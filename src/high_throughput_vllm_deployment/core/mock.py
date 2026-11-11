"""Deterministic, GPU-free simulation of vLLM continuous batching and PagedAttention.

The engine runs a discrete-step scheduler modelled on vLLM's V0 scheduler:

* **Continuous batching** - every step decodes one token for each RUNNING
  sequence; new sequences join (and finished ones leave) between steps.
* **PagedAttention** - KV cache is a pool of fixed-size blocks. A sequence of
  ``L`` context tokens holds ``ceil(L / block_size)`` GPU blocks and grows by
  one block whenever decoding crosses a block boundary. Blocks are reclaimed
  as soon as a sequence finishes or is aborted.
* **Scheduling** - each step (1) reserves the next-token slot for every
  running sequence, preempting the *latest-arrived* running sequence when the
  GPU pool is exhausted; then (2) resumes SWAPPED sequences (swap-in) and
  admits WAITING ones (prefill) strictly in arrival order across both queues,
  bounded by ``max_num_seqs``, the per-step prefill budget
  ``max_num_batched_tokens`` and free blocks. Step (2) is skipped in a step
  that preempted, to avoid thrashing.
* **Preemption** - a victim is swapped to the CPU block pool (-> SWAPPED) when
  it fits there, otherwise its blocks are dropped and it re-enters WAITING to
  be recomputed (re-prefilled) later.
* **Timing** - a step lasts ``decode_step_s`` plus ``prefill_s_per_token`` for
  every prompt token prefilled in it, so time-to-first-token reflects queueing
  and prompt length while inter-token latency reflects batch steps.

FCFS preemption of the latest arrival guarantees progress: the earliest
running sequence always fits on its own (``max_model_len`` is validated
against the pool size), so the simulation can never livelock.

Generated text is a pure function of the request (prompt/messages, seed,
choice index): concurrency, batching and preemption change *when* tokens
arrive, never *which* tokens arrive.
"""

from __future__ import annotations

import asyncio
import bisect
import contextlib
import hashlib
import itertools
import json
import math
import re
import weakref
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass, field
from typing import Any, Final, Self

from pydantic import Field, model_validator

from high_throughput_vllm_deployment.core._clock import Sleep
from high_throughput_vllm_deployment.core.errors import (
    EngineError,
    EngineRequestError,
    EngineUnavailableError,
)
from high_throughput_vllm_deployment.schemas._base import (
    ContractModel,
    NonEmptyStr,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
)
from high_throughput_vllm_deployment.schemas.config import VllmDeploymentConfig
from high_throughput_vllm_deployment.schemas.enums import FinishReason, SchedulerState
from high_throughput_vllm_deployment.schemas.inference import (
    AssistantMessage,
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    DeltaMessage,
    SamplingParams,
    StreamChoice,
    StreamChunk,
    Usage,
)
from high_throughput_vllm_deployment.schemas.telemetry import ClusterTelemetry, KvCacheBlocks

CHAT_MESSAGE_OVERHEAD_TOKENS: Final = 4
CHAT_REPLY_PRIMING_TOKENS: Final = 2

_WORD: Final = re.compile(r"\S+")
VOCABULARY: Final[tuple[str, ...]] = tuple(
    (
        "the of and to in is that for it as with was on be by at this are from or an "
        "which but not have one all were they there can had has more when will their "
        "model token batch block cache engine request stream latency throughput kernel "
        "tensor memory scheduler prefill decode attention paged queue swap gpu"
    ).split()
)


def count_tokens(text: str) -> int:
    """Whitespace tokenizer: deterministic and dependency-free."""
    return len(_WORD.findall(text))


def chat_prompt_tokens(request: ChatCompletionRequest) -> int:
    """Prompt length of a chat request, including chat-template overhead."""
    return CHAT_REPLY_PRIMING_TOKENS + sum(
        count_tokens(message.content) + CHAT_MESSAGE_OVERHEAD_TOKENS for message in request.messages
    )


def _digest(*parts: bytes) -> bytes:
    return hashlib.sha256(b"\x1f".join(parts)).digest()


def token_text(key: bytes, position: int) -> str:
    """Token emitted at *position* of the sequence identified by *key*."""
    digest = _digest(key, position.to_bytes(8, "big"))
    return " " + VOCABULARY[int.from_bytes(digest[:4], "big") % len(VOCABULARY)]


def natural_length(key: bytes, max_natural_tokens: int) -> int:
    """Number of tokens before the simulated model emits EOS."""
    digest = _digest(key, b"eos")
    return 1 + int.from_bytes(digest[:4], "big") % max_natural_tokens


class MockEngineConfig(ContractModel):
    """Capacity and pacing of the simulated engine."""

    model_name: NonEmptyStr = "mock-vllm"
    block_size: PositiveInt = 16
    num_gpu_blocks: PositiveInt = 1024
    num_cpu_blocks: NonNegativeInt = 256
    max_num_seqs: PositiveInt = 256
    max_num_batched_tokens: PositiveInt = 8192
    max_model_len: PositiveInt = 4096
    decode_step_s: NonNegativeFloat = 0.02
    prefill_s_per_token: NonNegativeFloat = 1e-4
    max_natural_tokens: PositiveInt = 64
    ignore_eos: bool = Field(
        default=False,
        description="Always generate up to the length limit (vLLM's benchmark `ignore_eos`).",
    )
    created_epoch: NonNegativeInt = 1_700_000_000

    @model_validator(mode="after")
    def _schedulable(self) -> Self:
        if self.max_model_len > self.num_gpu_blocks * self.block_size:
            raise ValueError(
                "max_model_len exceeds the GPU KV cache capacity "
                f"({self.num_gpu_blocks} blocks x {self.block_size} tokens)"
            )
        if self.max_num_batched_tokens < self.max_model_len:
            raise ValueError("max_num_batched_tokens must be >= max_model_len")
        return self

    @classmethod
    def from_deployment(cls, config: VllmDeploymentConfig, **overrides: Any) -> Self:
        """Mirror a deployment's model name and length/batching limits."""
        base: dict[str, Any] = {
            "model_name": config.model,
            "max_model_len": config.max_model_len,
            "max_num_batched_tokens": config.max_num_batched_tokens,
        }
        return cls.model_validate(base | overrides)


@dataclass(slots=True)
class MockEngineStats:
    """Cumulative scheduler counters (mutable, engine-owned)."""

    steps: int = 0
    prefill_tokens: int = 0
    decode_tokens: int = 0
    swap_preemptions: int = 0
    recompute_preemptions: int = 0
    swap_ins: int = 0
    completed_sequences: int = 0
    aborted_sequences: int = 0
    peak_running: int = 0
    peak_waiting: int = 0
    peak_swapped: int = 0
    peak_gpu_blocks_used: int = 0


@dataclass(frozen=True, slots=True)
class _Event:
    choice_index: int
    text: str = ""
    finish_reason: FinishReason | None = None
    error: EngineError | None = None


@dataclass(eq=False, slots=True)
class _Sequence:
    seq_id: int
    choice_index: int
    prompt_tokens: int
    target_tokens: int
    finish_reason: FinishReason
    key: bytes
    sink: asyncio.Queue[_Event]
    state: SchedulerState = SchedulerState.WAITING
    generated: int = 0
    gpu_blocks: int = 0
    cpu_blocks: int = 0
    finished: bool = False

    @property
    def context_len(self) -> int:
        return self.prompt_tokens + self.generated


def _arrival(seq: _Sequence) -> int:
    return seq.seq_id


@dataclass(slots=True)
class _Submission:
    request_id: str
    created: int
    sink: asyncio.Queue[_Event] = field(default_factory=asyncio.Queue)
    sequences: list[_Sequence] = field(default_factory=list)


class MockVllmEngine:
    """:class:`InferenceEnginePort` backed by the in-process scheduler simulation."""

    def __init__(self, config: MockEngineConfig | None = None, *, sleep: Sleep = asyncio.sleep):
        self.config = config or MockEngineConfig()
        self.stats = MockEngineStats()
        self._sleep = sleep
        self._running: list[_Sequence] = []
        self._waiting: list[_Sequence] = []
        self._swapped: list[_Sequence] = []
        self._free_gpu = self.config.num_gpu_blocks
        self._free_cpu = self.config.num_cpu_blocks
        self._seq_ids = itertools.count()
        self._request_ids = itertools.count(1)
        self._clock_s = 0.0
        self._task: asyncio.Task[None] | None = None
        self._closed = False
        self._failure: BaseException | None = None
        self._sinks: weakref.WeakSet[asyncio.Queue[_Event]] = weakref.WeakSet()

    # ------------------------------------------------------------------ #
    # Introspection                                                      #
    # ------------------------------------------------------------------ #

    @property
    def clock_s(self) -> float:
        """Simulated seconds elapsed (sum of step durations)."""
        return self._clock_s

    @property
    def is_idle(self) -> bool:
        """No sequence is running, waiting or swapped out."""
        return not (self._running or self._waiting or self._swapped)

    @property
    def is_quiescent(self) -> bool:
        """Idle, and every emitted event has been consumed by its requester.

        A request's final token is emitted (making the engine idle) one
        scheduling turn before its consumer observes it; time may only be
        skipped once nothing is in flight anywhere.
        """
        return self.is_idle and all(sink.empty() for sink in self._sinks)

    def advance_idle_clock(self, seconds: float) -> None:
        """Skip simulated idle time (virtual-time load generation); only legal when idle."""
        if seconds < 0:
            raise ValueError("cannot move the simulated clock backwards")
        if not self.is_idle:
            raise RuntimeError("the simulated clock only advances by itself while busy")
        self._clock_s += seconds

    def blocks_for(self, tokens: int) -> int:
        return math.ceil(tokens / self.config.block_size)

    def snapshot(self) -> ClusterTelemetry:
        cfg = self.config
        used_gpu = cfg.num_gpu_blocks - self._free_gpu
        used_cpu = cfg.num_cpu_blocks - self._free_cpu
        return ClusterTelemetry(
            active_requests=len(self._running),
            waiting_requests=len(self._waiting),
            swapped_requests=len(self._swapped),
            gpu_cache_usage_pct=100.0 * used_gpu / cfg.num_gpu_blocks,
            cpu_cache_usage_pct=100.0 * used_cpu / cfg.num_cpu_blocks
            if cfg.num_cpu_blocks
            else 0.0,
            kv_cache=KvCacheBlocks(
                block_size=cfg.block_size,
                total_gpu_blocks=cfg.num_gpu_blocks,
                used_gpu_blocks=used_gpu,
                total_cpu_blocks=cfg.num_cpu_blocks,
                used_cpu_blocks=used_cpu,
            ),
        )

    # ------------------------------------------------------------------ #
    # Lifecycle                                                          #
    # ------------------------------------------------------------------ #

    async def aclose(self) -> None:
        """Stop the scheduler; in-flight requests finish with ``abort``."""
        self._closed = True
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        for seq in self._all_sequences():
            self._release(seq)
            self.stats.aborted_sequences += 1
            seq.sink.put_nowait(_Event(seq.choice_index, finish_reason=FinishReason.ABORT))

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        await self.aclose()

    def _all_sequences(self) -> list[_Sequence]:
        return [*self._running, *self._swapped, *self._waiting]

    def _ensure_scheduler(self) -> None:
        # The scheduler task exits as soon as the engine is idle, so an engine
        # that is never closed cannot leak a pending task; restart on demand.
        if self._task is None or self._task.done():
            self._task = asyncio.get_running_loop().create_task(self._run())

    async def _run(self) -> None:
        try:
            while self._running or self._waiting or self._swapped:
                duration = self._schedule()
                await self._sleep(duration)
                self._clock_s += duration
                self._emit()
        except Exception as exc:  # fail every waiter instead of hanging them
            self._failure = exc
            error = EngineUnavailableError(f"mock engine step failed: {exc!r}")
            for seq in self._all_sequences():
                self._release(seq)
                seq.sink.put_nowait(_Event(seq.choice_index, error=error))

    # ------------------------------------------------------------------ #
    # Scheduler                                                          #
    # ------------------------------------------------------------------ #

    def _reserve_next_slot(self, seq: _Sequence) -> bool:
        needed = self.blocks_for(seq.context_len + 1) - seq.gpu_blocks
        if needed > self._free_gpu:
            return False
        self._free_gpu -= needed
        seq.gpu_blocks += needed
        return True

    def _activate(self, seq: _Sequence) -> None:
        seq.state = SchedulerState.RUNNING
        bisect.insort(self._running, seq, key=_arrival)

    def _preempt(self, victim: _Sequence) -> None:
        self._running.remove(victim)
        self._free_gpu += victim.gpu_blocks
        if victim.gpu_blocks <= self._free_cpu:
            self._free_cpu -= victim.gpu_blocks
            victim.cpu_blocks, victim.gpu_blocks = victim.gpu_blocks, 0
            victim.state = SchedulerState.SWAPPED
            bisect.insort(self._swapped, victim, key=_arrival)
            self.stats.swap_preemptions += 1
        else:
            victim.gpu_blocks = 0
            victim.state = SchedulerState.WAITING
            bisect.insort(self._waiting, victim, key=_arrival)
            self.stats.recompute_preemptions += 1

    def _schedule(self) -> float:
        """Plan one step; returns its simulated duration in seconds."""
        cfg = self.config
        preempted = False

        # (1) Decode slots for running sequences, oldest first.
        for seq in list(self._running):
            if seq.state is not SchedulerState.RUNNING:
                continue  # already preempted for an older sequence this step
            while not self._reserve_next_slot(seq):
                victim = self._running[-1]
                self._preempt(victim)
                preempted = True
                if victim is seq:
                    break

        prefill_tokens = 0
        if not preempted:
            # (2) Resume SWAPPED and admit WAITING sequences strictly in arrival
            # order across both queues, stopping at the first that does not fit.
            # Preempted sequences arrived before anything never scheduled, so
            # they resume first; FCFS also holds when swap and recompute mix.
            while (self._swapped or self._waiting) and len(self._running) < cfg.max_num_seqs:
                seq = min(self._swapped[:1] + self._waiting[:1], key=_arrival)
                needed = self.blocks_for(seq.context_len + 1)
                if needed > self._free_gpu:
                    break
                if seq.state is SchedulerState.SWAPPED:
                    self._swapped.pop(0)
                    self._free_cpu += seq.cpu_blocks
                    seq.cpu_blocks = 0
                    self.stats.swap_ins += 1
                else:
                    if prefill_tokens + seq.context_len > cfg.max_num_batched_tokens:
                        break
                    self._waiting.pop(0)
                    prefill_tokens += seq.context_len
                self._free_gpu -= needed
                seq.gpu_blocks = needed
                self._activate(seq)

        stats = self.stats
        stats.steps += 1
        stats.prefill_tokens += prefill_tokens
        stats.peak_running = max(stats.peak_running, len(self._running))
        stats.peak_waiting = max(stats.peak_waiting, len(self._waiting))
        stats.peak_swapped = max(stats.peak_swapped, len(self._swapped))
        stats.peak_gpu_blocks_used = max(
            stats.peak_gpu_blocks_used, cfg.num_gpu_blocks - self._free_gpu
        )
        return cfg.decode_step_s + prefill_tokens * cfg.prefill_s_per_token

    def _emit(self) -> None:
        for seq in list(self._running):
            text = token_text(seq.key, seq.generated)
            seq.generated += 1
            self.stats.decode_tokens += 1
            done = seq.generated >= seq.target_tokens
            seq.sink.put_nowait(_Event(seq.choice_index, text, seq.finish_reason if done else None))
            if done:
                self._release(seq)
                self.stats.completed_sequences += 1

    def _release(self, seq: _Sequence) -> None:
        """Free a live sequence's blocks and drop it from its queue."""
        seq.finished = True
        container = {
            SchedulerState.RUNNING: self._running,
            SchedulerState.WAITING: self._waiting,
            SchedulerState.SWAPPED: self._swapped,
        }[seq.state]
        container.remove(seq)
        self._free_gpu += seq.gpu_blocks
        self._free_cpu += seq.cpu_blocks
        seq.gpu_blocks = seq.cpu_blocks = 0

    def _abort(self, sequences: Iterable[_Sequence]) -> None:
        for seq in sequences:
            if not seq.finished:
                self._release(seq)
                self.stats.aborted_sequences += 1

    def _close(self, submission: _Submission) -> None:
        """The consumer is gone: abort what is left and forget its undelivered events.

        A consumer cancelled after its final event was queued, but before it
        read it, would otherwise leave that event pending forever, and the
        engine would never become quiescent.
        """
        self._abort(submission.sequences)
        self._sinks.discard(submission.sink)

    # ------------------------------------------------------------------ #
    # Request admission                                                  #
    # ------------------------------------------------------------------ #

    def _validate(self, request: SamplingParams, prompt_tokens: int) -> None:
        cfg = self.config
        if request.model != cfg.model_name:
            raise EngineRequestError(
                f"model {request.model!r} does not exist (serving {cfg.model_name!r})",
                status_code=404,
            )
        if prompt_tokens >= cfg.max_model_len:
            raise EngineRequestError(
                f"prompt has {prompt_tokens} tokens; max_model_len is {cfg.max_model_len}"
            )
        if (
            request.max_tokens is not None
            and prompt_tokens + request.max_tokens > cfg.max_model_len
        ):
            raise EngineRequestError(
                f"prompt ({prompt_tokens}) + max_tokens ({request.max_tokens}) exceeds "
                f"max_model_len ({cfg.max_model_len})"
            )

    def _open(self, prefix: str) -> _Submission:
        if self._closed:
            raise EngineUnavailableError("mock engine is closed")
        request_id = f"{prefix}-mock-{next(self._request_ids):08d}"
        submission = _Submission(request_id, self.config.created_epoch + int(self._clock_s))
        self._sinks.add(submission.sink)
        return submission

    def _submit(
        self,
        submission: _Submission,
        *,
        request: SamplingParams,
        prompt_tokens: int,
        identity: dict[str, Any],
        choice_index: int,
    ) -> None:
        key = _digest(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode())
        limit = (
            request.max_tokens
            if request.max_tokens is not None
            else self.config.max_model_len - prompt_tokens
        )
        natural = natural_length(key, self.config.max_natural_tokens)
        if natural < limit and not self.config.ignore_eos:
            target, reason = natural, FinishReason.STOP
        else:
            target, reason = limit, FinishReason.LENGTH
        seq = _Sequence(
            seq_id=next(self._seq_ids),
            choice_index=choice_index,
            prompt_tokens=prompt_tokens,
            target_tokens=target,
            finish_reason=reason,
            key=key,
            sink=submission.sink,
        )
        self._waiting.append(seq)
        submission.sequences.append(seq)

    def _submit_chat(self, request: ChatCompletionRequest, prefix: str) -> tuple[_Submission, int]:
        prompt_tokens = chat_prompt_tokens(request)
        self._validate(request, prompt_tokens)
        submission = self._open(prefix)
        messages = [message.model_dump(mode="json") for message in request.messages]
        for choice in range(request.n):
            identity = {"messages": messages, "seed": request.seed, "choice": choice}
            self._submit(
                submission,
                request=request,
                prompt_tokens=prompt_tokens,
                identity=identity,
                choice_index=choice,
            )
        self._ensure_scheduler()
        return submission, prompt_tokens

    async def _collect(self, submission: _Submission) -> dict[int, tuple[str, FinishReason]]:
        pieces: dict[int, list[str]] = {seq.choice_index: [] for seq in submission.sequences}
        reasons: dict[int, FinishReason] = {}
        try:
            while len(reasons) < len(submission.sequences):
                event = await submission.sink.get()
                if event.error is not None:
                    raise event.error
                pieces[event.choice_index].append(event.text)
                if event.finish_reason is not None:
                    reasons[event.choice_index] = event.finish_reason
        finally:
            self._close(submission)
        return {index: ("".join(pieces[index]), reasons[index]) for index in sorted(reasons)}

    @staticmethod
    def _usage(prompt_tokens: int, submission: _Submission) -> Usage:
        completion = sum(seq.generated for seq in submission.sequences)
        return Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion,
            total_tokens=prompt_tokens + completion,
        )

    # ------------------------------------------------------------------ #
    # InferenceEnginePort                                                #
    # ------------------------------------------------------------------ #

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        prompts = [request.prompt] if isinstance(request.prompt, str) else list(request.prompt)
        counts = [count_tokens(prompt) for prompt in prompts]
        for count in counts:
            self._validate(request, count)
        submission = self._open("cmpl")
        for prompt_index, (prompt, count) in enumerate(zip(prompts, counts, strict=True)):
            for choice in range(request.n):
                self._submit(
                    submission,
                    request=request,
                    prompt_tokens=count,
                    identity={"prompt": prompt, "seed": request.seed, "choice": choice},
                    choice_index=prompt_index * request.n + choice,
                )
        self._ensure_scheduler()
        results = await self._collect(submission)
        return CompletionResponse(
            id=submission.request_id,
            created=submission.created,
            model=self.config.model_name,
            choices=[
                CompletionChoice(index=index, text=text, finish_reason=reason)
                for index, (text, reason) in results.items()
            ],
            usage=self._usage(sum(counts), submission),
        )

    async def chat_complete(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        submission, prompt_tokens = self._submit_chat(request, "chatcmpl")
        results = await self._collect(submission)
        return ChatCompletionResponse(
            id=submission.request_id,
            created=submission.created,
            model=self.config.model_name,
            choices=[
                ChatCompletionChoice(
                    index=index, message=AssistantMessage(content=text), finish_reason=reason
                )
                for index, (text, reason) in results.items()
            ],
            usage=self._usage(prompt_tokens, submission),
        )

    async def stream_chat(self, request: ChatCompletionRequest) -> AsyncIterator[StreamChunk]:
        submission, prompt_tokens = self._submit_chat(request, "chatcmpl")

        def chunk(choice: StreamChoice) -> StreamChunk:
            return StreamChunk(
                id=submission.request_id,
                created=submission.created,
                model=self.config.model_name,
                choices=[choice],
            )

        try:
            for seq in submission.sequences:
                yield chunk(
                    StreamChoice(
                        index=seq.choice_index, delta=DeltaMessage(role="assistant", content="")
                    )
                )
            remaining = len(submission.sequences)
            while remaining:
                event = await submission.sink.get()
                if event.error is not None:
                    raise event.error
                if event.finish_reason is not None:
                    remaining -= 1
                yield chunk(
                    StreamChoice(
                        index=event.choice_index,
                        delta=DeltaMessage(content=event.text),
                        finish_reason=event.finish_reason,
                    )
                )
            if request.stream_options is not None and request.stream_options.include_usage:
                yield StreamChunk(
                    id=submission.request_id,
                    created=submission.created,
                    model=self.config.model_name,
                    choices=[],
                    usage=self._usage(prompt_tokens, submission),
                )
        finally:
            self._close(submission)

    async def get_telemetry(self) -> ClusterTelemetry:
        return self.snapshot()

    async def is_healthy(self) -> bool:
        return not self._closed and self._failure is None
