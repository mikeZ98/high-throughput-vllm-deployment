"""Deterministic synthetic traffic: arrival processes, prompt-length tiers, request payloads.

Everything is derived from ``WorkloadProfile.seed`` with independent random
streams for arrivals, prompt tiers/lengths, request kinds and prompt text, so
changing one knob (e.g. the arrival pattern) never perturbs the others.
Prompt "tokens" are whitespace-separated words: exact for the mock engine's
tokenizer, and roughly 1.3 model tokens per word for real BPE tokenizers.
"""

from __future__ import annotations

import itertools
import random
from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum, unique
from typing import Annotated, Final, Self

from pydantic import Field, Strict, model_validator

from high_throughput_vllm_deployment.schemas._base import (
    ContractModel,
    FiniteFloat,
    NonEmptyStr,
    PositiveInt,
)
from high_throughput_vllm_deployment.schemas.inference import (
    ChatCompletionRequest,
    ChatMessage,
    CompletionRequest,
    StreamOptions,
)


@unique
class ArrivalPattern(StrEnum):
    UNIFORM = "uniform"  # fixed inter-arrival gap of 1 / rate
    POISSON = "poisson"  # exponential gaps with mean 1 / rate
    BURST = "burst"  # `burst_size` simultaneous requests every burst_size / rate


@unique
class PromptTier(StrEnum):
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"


TIER_TOKENS: Final[dict[PromptTier, int]] = {
    PromptTier.SHORT: 128,
    PromptTier.MEDIUM: 512,
    PromptTier.LONG: 2048,
}

_WORDS: Final[tuple[str, ...]] = tuple(
    (
        "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi "
        "rho sigma tau upsilon phi chi psi omega river mountain forest ocean desert valley "
        "signal vector matrix kernel buffer packet socket thread engine cache latency"
    ).split()
)

LaxArrivalPattern = Annotated[ArrivalPattern, Strict(False)]
LaxPromptTier = Annotated[PromptTier, Strict(False)]
Fraction = Annotated[FiniteFloat, Field(ge=0.0, le=1.0)]


class WorkloadProfile(ContractModel):
    """Shape of a synthetic workload."""

    model: NonEmptyStr
    num_requests: PositiveInt = 100
    request_rate: Annotated[FiniteFloat, Field(gt=0.0)] = 10.0
    arrival: LaxArrivalPattern = ArrivalPattern.POISSON
    burst_size: PositiveInt = 8
    prompt_mix: dict[LaxPromptTier, Annotated[FiniteFloat, Field(ge=0.0)]] = Field(
        default_factory=lambda: {PromptTier.SHORT: 1.0}
    )
    length_jitter: Annotated[FiniteFloat, Field(ge=0.0, le=0.9)] = 0.25
    max_tokens: PositiveInt = 128
    chat_fraction: Fraction = 1.0
    stream_fraction: Fraction = 1.0
    temperature: Annotated[FiniteFloat, Field(ge=0.0, le=2.0)] = 0.0
    seed: int = 0

    @model_validator(mode="after")
    def _mix_has_mass(self) -> Self:
        if not any(weight > 0 for weight in self.prompt_mix.values()):
            raise ValueError("prompt_mix needs at least one tier with positive weight")
        return self


@dataclass(frozen=True, slots=True)
class WorkItem:
    index: int
    arrival_s: float
    tier: PromptTier
    prompt_tokens: int
    stream: bool
    request: CompletionRequest | ChatCompletionRequest


def inter_arrival_gaps(
    pattern: ArrivalPattern, rate: float, burst_size: int, rng: random.Random
) -> Iterator[float]:
    """Infinite stream of gaps between consecutive arrivals (the first arrival is at t=0)."""
    if pattern is ArrivalPattern.UNIFORM:
        return itertools.repeat(1.0 / rate)
    if pattern is ArrivalPattern.POISSON:
        return (rng.expovariate(rate) for _ in itertools.count())
    period = burst_size / rate
    return itertools.cycle([0.0] * (burst_size - 1) + [period])


def prompt_length_bounds(tier: PromptTier, jitter: float) -> tuple[int, int]:
    nominal = TIER_TOKENS[tier]
    return max(1, round(nominal * (1 - jitter))), round(nominal * (1 + jitter))


def prompt_text(num_tokens: int, rng: random.Random) -> str:
    return " ".join(rng.choice(_WORDS) for _ in range(num_tokens))


def _rng(seed: int, stream: str) -> random.Random:
    """Independent, reproducible stream per concern (simulation, not security)."""
    return random.Random(f"{seed}:{stream}")  # noqa: S311


def generate_workload(profile: WorkloadProfile) -> list[WorkItem]:
    """Materialise ``profile.num_requests`` work items, deterministically."""
    arrivals_rng = _rng(profile.seed, "arrival")
    tier_rng = _rng(profile.seed, "tier")
    kind_rng = _rng(profile.seed, "kind")
    text_rng = _rng(profile.seed, "text")

    tiers = [tier for tier, weight in profile.prompt_mix.items() if weight > 0]
    weights = [profile.prompt_mix[tier] for tier in tiers]
    gaps = inter_arrival_gaps(
        profile.arrival, profile.request_rate, profile.burst_size, arrivals_rng
    )

    items: list[WorkItem] = []
    arrival = 0.0
    for index in range(profile.num_requests):
        if index:
            arrival += next(gaps)
        tier = tier_rng.choices(tiers, weights)[0]
        low, high = prompt_length_bounds(tier, profile.length_jitter)
        prompt_tokens = tier_rng.randint(low, high)
        is_chat = kind_rng.random() < profile.chat_fraction
        stream = is_chat and kind_rng.random() < profile.stream_fraction
        text = prompt_text(prompt_tokens, text_rng)
        request: CompletionRequest | ChatCompletionRequest
        if is_chat:
            request = ChatCompletionRequest(
                model=profile.model,
                messages=[ChatMessage(role="user", content=text)],
                max_tokens=profile.max_tokens,
                temperature=profile.temperature,
                stream=stream,
                stream_options=StreamOptions(include_usage=True) if stream else None,
            )
        else:
            request = CompletionRequest(
                model=profile.model,
                prompt=text,
                max_tokens=profile.max_tokens,
                temperature=profile.temperature,
            )
        items.append(WorkItem(index, arrival, tier, prompt_tokens, stream, request))
    return items
