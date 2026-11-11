"""Scheduler invariants of MockVllmEngine, shared by the core and hardening suites."""

from __future__ import annotations

from high_throughput_vllm_deployment.core.mock import MockVllmEngine
from high_throughput_vllm_deployment.schemas.enums import SchedulerState


def assert_scheduler_invariants(engine: MockVllmEngine) -> None:
    """Internal block accounting must balance exactly at every step."""
    cfg = engine.config
    running, waiting, swapped = engine._running, engine._waiting, engine._swapped
    for queue, state in (
        (running, SchedulerState.RUNNING),
        (waiting, SchedulerState.WAITING),
        (swapped, SchedulerState.SWAPPED),
    ):
        assert all(seq.state is state and not seq.finished for seq in queue)
        assert [seq.seq_id for seq in queue] == sorted(seq.seq_id for seq in queue)
    ids = [seq.seq_id for seq in (*running, *waiting, *swapped)]
    assert len(ids) == len(set(ids)), "a sequence is in two queues"
    assert len(running) <= cfg.max_num_seqs
    assert sum(seq.gpu_blocks for seq in running) == cfg.num_gpu_blocks - engine._free_gpu
    assert sum(seq.cpu_blocks for seq in swapped) == cfg.num_cpu_blocks - engine._free_cpu
    assert all(seq.gpu_blocks == engine.blocks_for(seq.context_len + 1) for seq in running)
    assert all(seq.gpu_blocks == seq.cpu_blocks == 0 for seq in waiting)
    assert all(seq.gpu_blocks == 0 and seq.cpu_blocks > 0 for seq in swapped)
    # FCFS: preemption always takes the latest arrival and admission/swap-in go
    # oldest-first, so every running sequence arrived before every queued one.
    queued = [seq.seq_id for seq in (*waiting, *swapped)]
    if running and queued:
        assert max(seq.seq_id for seq in running) < min(queued), "FCFS order violated"
