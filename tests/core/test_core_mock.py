"""MockVllmEngine: continuous batching, PagedAttention accounting, preemption, determinism."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from engine_invariants import assert_scheduler_invariants
from high_throughput_vllm_deployment.core import instant_sleep
from high_throughput_vllm_deployment.core.errors import (
    EngineRequestError,
    EngineUnavailableError,
)
from high_throughput_vllm_deployment.core.mock import (
    CHAT_MESSAGE_OVERHEAD_TOKENS,
    CHAT_REPLY_PRIMING_TOKENS,
    MockEngineConfig,
    MockVllmEngine,
    chat_prompt_tokens,
    count_tokens,
)
from high_throughput_vllm_deployment.core.ports import InferenceEnginePort
from high_throughput_vllm_deployment.schemas.config import VllmDeploymentConfig
from high_throughput_vllm_deployment.schemas.enums import FinishReason, SchedulerState
from high_throughput_vllm_deployment.schemas.inference import (
    ChatCompletionRequest,
    ChatMessage,
    CompletionRequest,
    CompletionResponse,
    StreamChunk,
)
from high_throughput_vllm_deployment.schemas.telemetry import ClusterTelemetry

MODEL = "mock-vllm"
TIGHT = MockEngineConfig(
    block_size=4,
    num_gpu_blocks=24,
    num_cpu_blocks=64,
    max_model_len=64,
    max_num_batched_tokens=64,
    max_natural_tokens=40,
    ignore_eos=True,  # fixed output lengths make pressure scenarios exact
)
NATURAL = TIGHT.model_copy(update={"ignore_eos": False})


def _completion(prompt: str, max_tokens: int | None = None, **kwargs: Any) -> CompletionRequest:
    return CompletionRequest(model=MODEL, prompt=prompt, max_tokens=max_tokens, **kwargs)


def _chat(content: str, max_tokens: int | None = None, n: int = 1) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model=MODEL,
        messages=[ChatMessage(role="user", content=content)],
        max_tokens=max_tokens,
        n=n,
    )


class StepProbe:
    """Sleep hook: records each step's duration and post-schedule snapshot, checks invariants."""

    def __init__(self) -> None:
        self.engine: MockVllmEngine | None = None
        self.durations: list[float] = []
        self.snapshots: list[ClusterTelemetry] = []
        self._last = (0, 0, 0)

    def attach(self, config: MockEngineConfig) -> MockVllmEngine:
        self.engine = MockVllmEngine(config, sleep=self)
        return self.engine

    async def __call__(self, duration: float) -> None:
        assert self.engine is not None
        assert_scheduler_invariants(self.engine)
        stats = self.engine.stats
        preemptions = stats.swap_preemptions + stats.recompute_preemptions
        if preemptions > self._last[0]:  # a step that preempts must not admit or swap in
            assert (stats.prefill_tokens, stats.swap_ins) == self._last[1:], "thrashing step"
        self._last = (preemptions, stats.prefill_tokens, stats.swap_ins)
        self.durations.append(duration)
        self.snapshots.append(self.engine.snapshot())
        await asyncio.sleep(0)


async def _alone(config: MockEngineConfig, request: CompletionRequest) -> CompletionResponse:
    """Reference output: the request served by a fresh engine with no contention."""
    async with MockVllmEngine(config, sleep=instant_sleep) as engine:
        return await engine.complete(request)


def _essence(response: CompletionResponse) -> list[tuple[int, str, FinishReason | None]]:
    return [(c.index, c.text, c.finish_reason) for c in response.choices]


# --------------------------------------------------------------------------- #
# Configuration and helpers                                                   #
# --------------------------------------------------------------------------- #


def test_engine_satisfies_the_engine_port() -> None:
    port: InferenceEnginePort = MockVllmEngine()
    assert isinstance(port, InferenceEnginePort)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"num_gpu_blocks": 10, "block_size": 16, "max_model_len": 4096}, "GPU KV cache capacity"),
        ({"max_num_batched_tokens": 1024, "max_model_len": 2048}, "max_num_batched_tokens"),
    ],
)
def test_unschedulable_configs_are_rejected(overrides: dict[str, int], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        MockEngineConfig.model_validate(overrides)


def test_config_mirrors_a_deployment() -> None:
    deployment = VllmDeploymentConfig(
        model="org/model", max_model_len=2048, max_num_batched_tokens=4096
    )
    config = MockEngineConfig.from_deployment(deployment, num_gpu_blocks=256)
    assert (config.model_name, config.max_model_len, config.max_num_batched_tokens) == (
        "org/model",
        2048,
        4096,
    )
    assert config.num_gpu_blocks == 256


def test_token_counting() -> None:
    assert count_tokens("  one two\nthree  ") == 3
    request = ChatCompletionRequest(
        model=MODEL,
        messages=[
            ChatMessage(role="system", content="be brief"),
            ChatMessage(role="user", content="hi"),
        ],
    )
    expected = CHAT_REPLY_PRIMING_TOKENS + 3 + 2 * CHAT_MESSAGE_OVERHEAD_TOKENS
    assert chat_prompt_tokens(request) == expected


# --------------------------------------------------------------------------- #
# Request validation and lifecycle                                            #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("request_", "status", "match"),
    [
        (CompletionRequest(model="other", prompt="x"), 404, "does not exist"),
        (_completion("w " * 64), 400, "max_model_len is 64"),
        (_completion("w " * 60, max_tokens=5), 400, "exceeds max_model_len"),
        (CompletionRequest(model=MODEL, prompt=["ok", "w " * 70]), 400, "max_model_len"),
    ],
)
async def test_invalid_requests_are_rejected_atomically(
    request_: CompletionRequest, status: int, match: str
) -> None:
    async with MockVllmEngine(TIGHT, sleep=instant_sleep) as engine:
        with pytest.raises(EngineRequestError, match=match) as info:
            await engine.complete(request_)
        assert info.value.status_code == status
        assert engine.stats.steps == 0
        assert engine.snapshot().total_requests == 0


@pytest.mark.asyncio
async def test_closed_engine_refuses_work_and_reports_unhealthy() -> None:
    engine = MockVllmEngine(sleep=instant_sleep)
    assert await engine.is_healthy()
    await engine.aclose()  # never started: no scheduler task to cancel
    assert not await engine.is_healthy()
    with pytest.raises(EngineUnavailableError, match="closed"):
        await engine.complete(_completion("x"))
    with pytest.raises(EngineUnavailableError, match="closed"):
        await anext(engine.stream_chat(_chat("x")))


@pytest.mark.asyncio
async def test_scheduler_goes_idle_and_restarts_on_demand() -> None:
    async with MockVllmEngine(sleep=instant_sleep) as engine:
        first = await engine.complete(_completion("first request", max_tokens=3))
        await asyncio.sleep(0)
        assert engine._task is not None and engine._task.done()
        second = await engine.complete(_completion("second request", max_tokens=3))
    assert first.id == "cmpl-mock-00000001"
    assert second.id == "cmpl-mock-00000002"


@pytest.mark.asyncio
async def test_aclose_aborts_in_flight_requests() -> None:
    gate = asyncio.Event()
    steps = 0

    async def gated_sleep(_duration: float) -> None:
        nonlocal steps
        steps += 1
        if steps > 2:
            await gate.wait()  # park the scheduler mid-generation

    engine = MockVllmEngine(TIGHT, sleep=gated_sleep)
    pending = asyncio.ensure_future(engine.complete(_completion("abort me", max_tokens=30)))
    stream = engine.stream_chat(_chat("abort me too", max_tokens=30))
    streamed = [await anext(stream)]
    while steps <= 2:
        await asyncio.sleep(0)
    await engine.aclose()

    response = await pending
    assert response.choices[0].finish_reason is FinishReason.ABORT
    assert response.usage.completion_tokens == 2
    streamed += [chunk async for chunk in stream]
    assert streamed[-1].choices[0].finish_reason is FinishReason.ABORT
    assert engine.stats.aborted_sequences == 2
    assert engine.snapshot().kv_cache.used_gpu_blocks == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("streaming", [False, True])
async def test_scheduler_failure_fails_waiters_instead_of_hanging(streaming: bool) -> None:
    async def broken_sleep(_duration: float) -> None:
        raise RuntimeError("simulated device fault")

    engine = MockVllmEngine(TIGHT, sleep=broken_sleep)
    with pytest.raises(EngineUnavailableError, match="simulated device fault"):
        if streaming:
            async for _ in engine.stream_chat(_chat("x", max_tokens=4)):
                pass
        else:
            await engine.complete(_completion("x", max_tokens=4))
    assert not await engine.is_healthy()
    assert engine.snapshot().kv_cache.used_gpu_blocks == 0


@pytest.mark.asyncio
async def test_closing_a_stream_early_aborts_and_reclaims_blocks() -> None:
    async with MockVllmEngine(TIGHT, sleep=instant_sleep) as engine:
        stream = engine.stream_chat(_chat("stop listening", max_tokens=30))
        await anext(stream)
        await anext(stream)
        await stream.aclose()  # type: ignore[attr-defined]
        assert engine.stats.aborted_sequences == 1
        assert engine.snapshot().kv_cache.used_gpu_blocks == 0
        assert engine.snapshot().total_requests == 0


# --------------------------------------------------------------------------- #
# Outputs: determinism and token sequencing                                   #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_outputs_are_deterministic_and_finish_reasons_are_consistent() -> None:
    capped = _completion("the quick brown fox", max_tokens=3)
    free = _completion("the quick brown fox")
    first, second = await _alone(NATURAL, capped), await _alone(NATURAL, capped)
    assert _essence(first) == _essence(second)
    assert first.choices[0].finish_reason is FinishReason.LENGTH
    assert first.usage.completion_tokens == 3 == count_tokens(first.choices[0].text)

    natural = await _alone(NATURAL, free)
    assert natural.choices[0].finish_reason is FinishReason.STOP
    assert 3 < natural.usage.completion_tokens <= NATURAL.max_natural_tokens
    assert natural.choices[0].text.startswith(first.choices[0].text)  # same token stream

    forced = await _alone(TIGHT, free)  # ignore_eos: runs to max_model_len - prompt
    assert forced.choices[0].finish_reason is FinishReason.LENGTH
    assert forced.usage.completion_tokens == TIGHT.max_model_len - 4
    seeded = await _alone(NATURAL, _completion("the quick brown fox", max_tokens=3, seed=7))
    assert seeded.choices[0].text != first.choices[0].text


@pytest.mark.asyncio
async def test_batch_prompts_and_n_choices_follow_openai_indexing() -> None:
    request = CompletionRequest(model=MODEL, prompt=["alpha beta", "gamma"], n=2, max_tokens=4)
    async with MockVllmEngine(TIGHT, sleep=instant_sleep) as engine:
        response = await engine.complete(request)
    assert [c.index for c in response.choices] == [0, 1, 2, 3]
    assert response.usage.prompt_tokens == 3
    assert response.usage.completion_tokens == 16
    alpha = await _alone(TIGHT, _completion("alpha beta", max_tokens=4))
    assert response.choices[0].text == alpha.choices[0].text


@pytest.mark.asyncio
async def test_stream_matches_non_streaming_output_and_is_ordered() -> None:
    request = _chat("tell me about paged attention", max_tokens=12, n=3)
    async with MockVllmEngine(TIGHT, sleep=instant_sleep) as engine:
        chunks: list[StreamChunk] = [chunk async for chunk in engine.stream_chat(request)]
        reference = await engine.chat_complete(request)

    assert len({chunk.id for chunk in chunks}) == 1
    per_choice: dict[int, list[StreamChunk]] = {}
    for chunk in chunks:
        (choice,) = chunk.choices
        per_choice.setdefault(choice.index, []).append(chunk)
    assert sorted(per_choice) == [0, 1, 2]
    for index, choice_chunks in per_choice.items():
        deltas = [c.choices[0] for c in choice_chunks]
        assert deltas[0].delta.role == "assistant" and deltas[0].delta.content == ""
        assert all(d.finish_reason is None for d in deltas[:-1])
        assert deltas[-1].finish_reason is FinishReason.LENGTH
        text = "".join(d.delta.content or "" for d in deltas)
        assert text == reference.choices[index].message.content
    assert reference.usage.completion_tokens == 36


@pytest.mark.asyncio
@pytest.mark.parametrize("num_cpu_blocks", [64, 0], ids=["swap", "recompute"])
async def test_concurrent_outputs_are_invariant_under_preemption(num_cpu_blocks: int) -> None:
    config = TIGHT.model_copy(update={"num_cpu_blocks": num_cpu_blocks})
    requests = [_completion(f"prompt number {i} has words", max_tokens=30) for i in range(8)]
    probe = StepProbe()
    async with probe.attach(config) as engine:
        concurrent = await asyncio.gather(*(engine.complete(r) for r in requests))
        stats = engine.stats
    assert stats.swap_preemptions + stats.recompute_preemptions > 0, "test must exercise pressure"
    for request, response in zip(requests, concurrent, strict=True):
        assert _essence(response) == _essence(await _alone(config, request))
        assert response.usage.completion_tokens == 30


@settings(max_examples=40, deadline=None)
@given(
    block_size=st.sampled_from([1, 2, 4, 8]),
    extra_blocks=st.integers(min_value=0, max_value=24),
    num_cpu_blocks=st.integers(min_value=0, max_value=32),
    max_num_seqs=st.integers(min_value=1, max_value=8),
    jobs=st.lists(
        st.tuples(st.integers(min_value=1, max_value=10), st.integers(min_value=1, max_value=30)),
        min_size=1,
        max_size=10,
    ),
)
def test_scheduling_never_changes_outputs_and_always_drains(
    block_size: int,
    extra_blocks: int,
    num_cpu_blocks: int,
    max_num_seqs: int,
    jobs: list[tuple[int, int]],
) -> None:
    max_model_len = 48
    config = MockEngineConfig(
        block_size=block_size,
        num_gpu_blocks=-(-max_model_len // block_size) + extra_blocks,
        num_cpu_blocks=num_cpu_blocks,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_model_len,
        max_natural_tokens=20,
    )
    requests = [
        _completion(" ".join(f"w{i}x{k}" for k in range(words)), max_tokens=max_tokens)
        for i, (words, max_tokens) in enumerate(jobs)
    ]

    async def scenario() -> None:
        probe = StepProbe()
        async with probe.attach(config) as engine:
            results = await asyncio.gather(*(engine.complete(r) for r in requests))
            assert all(s.active_requests <= max_num_seqs for s in probe.snapshots)
            final = engine.snapshot()
            assert final.total_requests == 0
            assert final.kv_cache.used_gpu_blocks == final.kv_cache.used_cpu_blocks == 0
            assert engine.stats.completed_sequences == len(requests)
            assert engine.stats.swap_ins == engine.stats.swap_preemptions
        for request, response in zip(requests, results, strict=True):
            assert _essence(response) == _essence(await _alone(config, request))

    asyncio.run(scenario())


# --------------------------------------------------------------------------- #
# PagedAttention block accounting                                             #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_blocks_grow_with_context_and_are_reclaimed_on_completion() -> None:
    prompt_tokens, max_tokens = 5, 20
    probe = StepProbe()
    async with probe.attach(TIGHT) as engine:
        await engine.complete(_completion(" ".join(["w"] * prompt_tokens), max_tokens=max_tokens))
        after = engine.snapshot()

    used = [s.kv_cache.used_gpu_blocks for s in probe.snapshots]
    # At step k the sequence holds its prompt, k generated tokens and the slot for token k+1.
    assert used == [engine.blocks_for(prompt_tokens + k + 1) for k in range(max_tokens)]
    assert used == sorted(used) and used[-1] > used[0]
    assert after.kv_cache.used_gpu_blocks == 0
    assert after.gpu_cache_usage_pct == 0.0


@pytest.mark.asyncio
async def test_finished_sequences_return_blocks_while_others_keep_running() -> None:
    probe = StepProbe()
    async with probe.attach(TIGHT) as engine:
        await asyncio.gather(
            engine.complete(_completion("w w w w", max_tokens=2)),
            engine.complete(_completion("w w w w", max_tokens=12)),
        )
    active = [s.active_requests for s in probe.snapshots]
    used = [s.kv_cache.used_gpu_blocks for s in probe.snapshots]
    handover = active.index(1)
    assert active[:handover] == [2] * handover
    assert used[handover] < used[handover - 1], "short request's blocks were not reclaimed"


# --------------------------------------------------------------------------- #
# Cache pressure: WAITING, SWAPPED and recompute                              #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_gpu_exhaustion_swaps_latest_arrivals_to_cpu_and_back() -> None:
    probe = StepProbe()
    async with probe.attach(TIGHT) as engine:
        await asyncio.gather(
            *(engine.complete(_completion(f"req {i}", max_tokens=30)) for i in range(8))
        )
        stats = engine.stats
    assert stats.swap_preemptions > 0 and stats.recompute_preemptions == 0
    assert stats.swap_ins == stats.swap_preemptions
    assert max(s.swapped_requests for s in probe.snapshots) == stats.peak_swapped > 0
    assert max(s.cpu_cache_usage_pct for s in probe.snapshots) > 0.0
    assert stats.peak_gpu_blocks_used == TIGHT.num_gpu_blocks


@pytest.mark.asyncio
async def test_without_swap_space_preempted_requests_are_recomputed_from_waiting() -> None:
    config = TIGHT.model_copy(update={"num_cpu_blocks": 0})
    probe = StepProbe()
    prompts = [f"request {i} prompt" for i in range(8)]
    async with probe.attach(config) as engine:
        await asyncio.gather(*(engine.complete(_completion(p, max_tokens=30)) for p in prompts))
        stats = engine.stats
    assert stats.recompute_preemptions > 0 and stats.swap_preemptions == 0
    assert all(s.swapped_requests == 0 and s.cpu_cache_usage_pct == 0.0 for s in probe.snapshots)
    # Recompute re-prefills prompt + already generated tokens.
    assert stats.prefill_tokens > sum(count_tokens(p) for p in prompts)
    ever_ran = [i for i, s in enumerate(probe.snapshots) if s.active_requests == 8]
    assert ever_ran and any(s.waiting_requests > 0 for s in probe.snapshots[ever_ran[0] :])


@pytest.mark.asyncio
async def test_max_num_seqs_holds_excess_requests_in_waiting() -> None:
    config = TIGHT.model_copy(update={"max_num_seqs": 2})
    probe = StepProbe()
    async with probe.attach(config) as engine:
        await asyncio.gather(
            *(engine.complete(_completion(f"q {i}", max_tokens=3)) for i in range(5))
        )
    assert max(s.active_requests for s in probe.snapshots) == 2
    assert probe.snapshots[0].waiting_requests == 3
    assert engine.stats.peak_waiting == 3


@pytest.mark.asyncio
async def test_prefill_budget_limits_admissions_per_step() -> None:
    config = MockEngineConfig(
        max_model_len=16, max_num_batched_tokens=16, block_size=4, num_gpu_blocks=64
    )
    probe = StepProbe()
    async with probe.attach(config) as engine:
        await asyncio.gather(
            *(engine.complete(_completion("w " * 10, max_tokens=4)) for _ in range(3))
        )
    assert [s.active_requests for s in probe.snapshots[:3]] == [1, 2, 3]


@pytest.mark.asyncio
async def test_swapped_requests_take_priority_over_new_arrivals() -> None:
    trace: list[tuple[SchedulerState | None, int]] = []
    late_id: int | None = None

    async def tracing_sleep(_duration: float) -> None:
        state = next((s.state for s in engine._all_sequences() if s.seq_id == late_id), None)
        trace.append((state, len(engine._swapped)))
        await asyncio.sleep(0)

    engine = MockVllmEngine(TIGHT, sleep=tracing_sleep)
    async with engine:
        tasks = [
            asyncio.ensure_future(engine.complete(_completion(f"early {i}", max_tokens=30)))
            for i in range(8)
        ]
        while engine.snapshot().swapped_requests == 0:
            await asyncio.sleep(0)
        late = asyncio.ensure_future(engine.complete(_completion("late arrival", max_tokens=2)))
        await asyncio.sleep(0)
        late_id = max(s.seq_id for s in engine._all_sequences())
        await asyncio.gather(*tasks, late)

    observed = [(state, swapped) for state, swapped in trace if state is not None]
    admitted_at = next(
        i for i, (state, _) in enumerate(observed) if state is SchedulerState.RUNNING
    )
    assert admitted_at > 0, "late request must have been held back"
    assert all(state is SchedulerState.WAITING for state, _ in observed[:admitted_at])
    assert any(swapped > 0 for _, swapped in observed[:admitted_at])
    assert observed[admitted_at][1] == 0, "admitted while a swapped sequence was pending"


# --------------------------------------------------------------------------- #
# Timing and telemetry                                                        #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_step_durations_model_prefill_and_decode() -> None:
    config = MockEngineConfig(decode_step_s=0.02, prefill_s_per_token=0.001)
    probe = StepProbe()
    async with probe.attach(config) as engine:
        response = await engine.complete(_completion("w " * 100, max_tokens=5))
        clock = engine.clock_s
    assert probe.durations[0] == pytest.approx(0.02 + 100 * 0.001)
    assert probe.durations[1:] == pytest.approx([0.02] * 4)
    assert clock == pytest.approx(sum(probe.durations))
    assert response.created == config.created_epoch


@pytest.mark.asyncio
async def test_telemetry_reports_live_scheduler_state() -> None:
    config = TIGHT.model_copy(update={"num_cpu_blocks": 0})
    async with MockVllmEngine(config, sleep=instant_sleep) as engine:
        idle = await engine.get_telemetry()
        assert idle.cpu_cache_usage_pct == 0.0
        assert idle.kv_cache.total_gpu_blocks == 24
        task = asyncio.ensure_future(engine.complete(_completion("busy now", max_tokens=8)))
        await asyncio.sleep(0)
        busy = await engine.get_telemetry()
        await task
    assert busy.active_requests + busy.waiting_requests == 1


# --------------------------------------------------------------------------- #
# Usage chunk and quiescence                                                  #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_include_usage_appends_a_usage_only_chunk() -> None:
    from high_throughput_vllm_deployment.schemas.inference import StreamOptions

    request = _chat("count my tokens", max_tokens=5).model_copy(
        update={"stream_options": StreamOptions(include_usage=True)}
    )
    async with MockVllmEngine(TIGHT, sleep=instant_sleep) as engine:
        chunks = [chunk async for chunk in engine.stream_chat(request)]
        plain = [chunk async for chunk in engine.stream_chat(_chat("count my tokens", 5))]
    assert chunks[-1].choices == [] and chunks[-1].usage is not None
    assert chunks[-1].usage.completion_tokens == 5
    assert chunks[-1].usage.prompt_tokens == chat_prompt_tokens(request)
    assert all(chunk.usage is None for chunk in plain)


@pytest.mark.asyncio
async def test_quiescence_waits_for_results_to_be_consumed() -> None:
    def state(engine: MockVllmEngine) -> tuple[bool, bool]:
        # Read through a call: the properties change across awaits.
        return engine.is_idle, engine.is_quiescent

    async with MockVllmEngine(TIGHT, sleep=instant_sleep) as engine:
        assert state(engine) == (True, True)
        stream = engine.stream_chat(_chat("hold the last token", max_tokens=2))
        await anext(stream)  # role chunk; both tokens are produced while we do not read
        while not state(engine)[0]:
            await asyncio.sleep(0)
        assert state(engine) == (True, False)  # idle, but events still undelivered
        _ = [chunk async for chunk in stream]
        assert state(engine) == (True, True)


@pytest.mark.asyncio
async def test_abandoned_consumer_does_not_block_quiescence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A consumer cancelled after its final event was queued (before reading it)
    must not leave the engine non-quiescent forever (found by FSM fuzzing)."""
    async with MockVllmEngine(TIGHT, sleep=instant_sleep) as engine:
        task = asyncio.ensure_future(engine.complete(_completion("last words", max_tokens=1)))
        await asyncio.sleep(0)
        sinks = list(engine._sinks)  # keep the queue alive, as a retained traceback would
        emit = engine._emit

        def emit_then_cancel() -> None:
            emit()  # queues the final event...
            task.cancel()  # ...and the consumer is cancelled before it can read it

        monkeypatch.setattr(engine, "_emit", emit_then_cancel)
        with pytest.raises(asyncio.CancelledError):
            await task
        assert engine.is_idle
        assert not sinks[0].empty()  # the final event really is undelivered
        assert engine.is_quiescent
