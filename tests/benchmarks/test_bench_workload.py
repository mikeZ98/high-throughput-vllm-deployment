"""Workload generator: arrival statistics, prompt-length bounds, payload validity, determinism."""

from __future__ import annotations

import itertools
import math
import random
import statistics
from collections import Counter

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from high_throughput_vllm_deployment.benchmarks.workload import (
    TIER_TOKENS,
    ArrivalPattern,
    PromptTier,
    WorkloadProfile,
    generate_workload,
    inter_arrival_gaps,
    prompt_length_bounds,
)
from high_throughput_vllm_deployment.core.mock import count_tokens
from high_throughput_vllm_deployment.schemas.inference import (
    ChatCompletionRequest,
    CompletionRequest,
)

MODEL = "mock-vllm"


def _profile(**overrides: object) -> WorkloadProfile:
    return WorkloadProfile.model_validate({"model": MODEL} | overrides)


def _gaps(items: list[float]) -> list[float]:
    return [b - a for a, b in itertools.pairwise(items)]


# --------------------------------------------------------------------------- #
# Arrival processes                                                           #
# --------------------------------------------------------------------------- #


def test_uniform_arrivals_are_evenly_spaced() -> None:
    items = generate_workload(_profile(num_requests=50, request_rate=4.0, arrival="uniform"))
    arrivals = [item.arrival_s for item in items]
    assert arrivals[0] == 0.0
    assert _gaps(arrivals) == pytest.approx([0.25] * 49)


def test_poisson_arrivals_have_exponential_gaps() -> None:
    rate = 20.0
    items = generate_workload(_profile(num_requests=5000, request_rate=rate, arrival="poisson"))
    gaps = _gaps([item.arrival_s for item in items])
    mean = statistics.fmean(gaps)
    assert mean == pytest.approx(1 / rate, rel=0.05)
    assert statistics.stdev(gaps) / mean == pytest.approx(1.0, abs=0.05)  # CV of an exponential
    assert all(gap >= 0 for gap in gaps)
    # Memorylessness: P(gap > mean) = e^-1.
    assert sum(g > 1 / rate for g in gaps) / len(gaps) == pytest.approx(math.exp(-1), abs=0.03)


def test_burst_arrivals_cluster_and_preserve_the_mean_rate() -> None:
    items = generate_workload(
        _profile(num_requests=40, request_rate=10.0, arrival="burst", burst_size=8)
    )
    groups = [list(g) for _, g in itertools.groupby(items, key=lambda item: item.arrival_s)]
    assert [len(g) for g in groups] == [8] * 5
    assert [g[0].arrival_s for g in groups] == pytest.approx([0.0, 0.8, 1.6, 2.4, 3.2])
    assert len(items) / (groups[-1][0].arrival_s + 0.8) == pytest.approx(10.0)


def test_inter_arrival_gap_streams() -> None:
    rng = random.Random(0)  # noqa: S311 - test determinism only
    assert list(itertools.islice(inter_arrival_gaps(ArrivalPattern.UNIFORM, 2.0, 1, rng), 3)) == [
        0.5,
        0.5,
        0.5,
    ]
    burst = list(itertools.islice(inter_arrival_gaps(ArrivalPattern.BURST, 4.0, 3, rng), 6))
    assert burst == [0.0, 0.0, 0.75, 0.0, 0.0, 0.75]


@settings(max_examples=50, deadline=None)
@given(
    arrival=st.sampled_from(list(ArrivalPattern)),
    rate=st.floats(min_value=0.1, max_value=500.0),
    burst_size=st.integers(min_value=1, max_value=16),
    n=st.integers(min_value=1, max_value=60),
    seed=st.integers(),
)
def test_arrivals_start_at_zero_and_never_decrease(
    arrival: ArrivalPattern, rate: float, burst_size: int, n: int, seed: int
) -> None:
    items = generate_workload(
        _profile(
            num_requests=n, request_rate=rate, arrival=arrival, burst_size=burst_size, seed=seed
        )
    )
    arrivals = [item.arrival_s for item in items]
    assert arrivals[0] == 0.0
    assert arrivals == sorted(arrivals)
    assert [item.index for item in items] == list(range(n))


# --------------------------------------------------------------------------- #
# Prompt lengths and payloads                                                 #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("tier", "jitter", "bounds"),
    [
        (PromptTier.SHORT, 0.25, (96, 160)),
        (PromptTier.MEDIUM, 0.0, (512, 512)),
        (PromptTier.LONG, 0.5, (1024, 3072)),
        (PromptTier.SHORT, 0.9, (13, 243)),
    ],
)
def test_prompt_length_bounds(tier: PromptTier, jitter: float, bounds: tuple[int, int]) -> None:
    assert prompt_length_bounds(tier, jitter) == bounds


@settings(max_examples=40, deadline=None)
@given(
    tier=st.sampled_from(list(PromptTier)),
    jitter=st.floats(min_value=0.0, max_value=0.9),
    seed=st.integers(),
)
def test_prompts_have_exactly_the_sampled_length_within_tier_bounds(
    tier: PromptTier, jitter: float, seed: int
) -> None:
    items = generate_workload(
        _profile(num_requests=8, prompt_mix={tier: 1.0}, length_jitter=jitter, seed=seed)
    )
    low, high = prompt_length_bounds(tier, jitter)
    for item in items:
        assert item.tier is tier
        assert low <= item.prompt_tokens <= high
        assert isinstance(item.request, ChatCompletionRequest)
        assert count_tokens(item.request.messages[0].content) == item.prompt_tokens


def test_tier_mix_follows_weights() -> None:
    mix = {
        PromptTier.SHORT: 0.6,
        PromptTier.MEDIUM: 0.3,
        PromptTier.LONG: 0.1,
    }
    items = generate_workload(_profile(num_requests=4000, prompt_mix=mix))
    counts = Counter(item.tier for item in items)
    for tier, weight in mix.items():
        assert counts[tier] / len(items) == pytest.approx(weight, abs=0.025)
    mean_tokens = statistics.fmean(item.prompt_tokens for item in items)
    expected = sum(TIER_TOKENS[t] * w for t, w in mix.items())
    assert mean_tokens == pytest.approx(expected, rel=0.03)


def test_zero_weight_tiers_are_never_sampled() -> None:
    items = generate_workload(_profile(num_requests=200, prompt_mix={"short": 1.0, "long": 0.0}))
    assert {item.tier for item in items} == {PromptTier.SHORT}


def test_request_kinds_follow_fractions() -> None:
    completions = generate_workload(_profile(num_requests=20, chat_fraction=0.0, max_tokens=7))
    assert all(isinstance(item.request, CompletionRequest) for item in completions)
    assert not any(item.stream for item in completions)
    assert all(item.request.max_tokens == 7 for item in completions)

    streamed = generate_workload(_profile(num_requests=20, stream_fraction=1.0))
    for item in streamed:
        assert isinstance(item.request, ChatCompletionRequest) and item.stream
        assert item.request.stream is True
        assert item.request.stream_options is not None
        assert item.request.stream_options.include_usage

    blocking = generate_workload(_profile(num_requests=20, stream_fraction=0.0))
    assert all(
        isinstance(i.request, ChatCompletionRequest) and i.request.stream_options is None
        for i in blocking
    )

    mixed = generate_workload(_profile(num_requests=2000, chat_fraction=0.5, stream_fraction=0.5))
    chat = sum(isinstance(i.request, ChatCompletionRequest) for i in mixed) / len(mixed)
    streams = sum(i.stream for i in mixed) / len(mixed)
    assert chat == pytest.approx(0.5, abs=0.04)
    assert streams == pytest.approx(0.25, abs=0.04)


def test_generation_is_deterministic_per_seed() -> None:
    first = generate_workload(_profile(num_requests=30, seed=7))
    again = generate_workload(_profile(num_requests=30, seed=7))
    other = generate_workload(_profile(num_requests=30, seed=8))
    assert first == again
    assert first != other


def test_changing_arrivals_does_not_change_payloads() -> None:
    poisson = generate_workload(_profile(num_requests=30, arrival="poisson"))
    burst = generate_workload(_profile(num_requests=30, arrival="burst"))
    assert [i.request for i in poisson] == [i.request for i in burst]
    assert [i.arrival_s for i in poisson] != [i.arrival_s for i in burst]


@pytest.mark.parametrize(
    "overrides",
    [
        {"prompt_mix": {"short": 0.0}},
        {"prompt_mix": {"huge": 1.0}},
        {"request_rate": 0.0},
        {"length_jitter": 0.95},
        {"chat_fraction": 1.5},
        {"arrival": "gaussian"},
    ],
)
def test_invalid_profiles_are_rejected(overrides: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        _profile(**overrides)
