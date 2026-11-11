"""Prometheus text parsing and vLLM /metrics -> ClusterTelemetry mapping."""

from __future__ import annotations

import pytest

from high_throughput_vllm_deployment.core.errors import EngineProtocolError
from high_throughput_vllm_deployment.core.metrics import (
    MetricsParseError,
    parse_prometheus_text,
    telemetry_from_prometheus,
)

CACHE_INFO = (
    'vllm:cache_config_info{block_size="16",cache_dtype="auto",gpu_memory_utilization="0.9",'
    'num_cpu_blocks="512",num_gpu_blocks="2048"} 1.0'
)

VLLM_V0_METRICS = f"""\
# HELP vllm:num_requests_running Number of requests currently running on GPU.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{{model_name="mistralai/Mistral-7B-Instruct-v0.2"}} 12.0
# HELP vllm:num_requests_waiting Number of requests waiting to be processed.
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{{model_name="mistralai/Mistral-7B-Instruct-v0.2"}} 3.0
vllm:num_requests_swapped{{model_name="mistralai/Mistral-7B-Instruct-v0.2"}} 1.0
vllm:gpu_cache_usage_perc{{model_name="mistralai/Mistral-7B-Instruct-v0.2"}} 0.25
vllm:cpu_cache_usage_perc{{model_name="mistralai/Mistral-7B-Instruct-v0.2"}} 0.125
# HELP vllm:cache_config_info Information of the LLMEngine CacheConfig
# TYPE vllm:cache_config_info gauge
{CACHE_INFO}
vllm:time_to_first_token_seconds_bucket{{le="0.001",model_name="m"}} 0.0 1700000000000
vllm:time_to_first_token_seconds_bucket{{le="+Inf",model_name="m"}} 42.0
process_resident_memory_bytes 1.2e+09
"""

VLLM_V1_METRICS = """\
vllm:num_requests_running{engine="0",model_name="m"} 4.0
vllm:num_requests_running{engine="1",model_name="m"} 6.0
vllm:num_requests_waiting{engine="0",model_name="m"} 0.0
vllm:num_requests_waiting{engine="1",model_name="m"} 2.0
vllm:kv_cache_usage_perc{engine="0",model_name="m"} 0.5
vllm:kv_cache_usage_perc{engine="1",model_name="m"} 0.7
vllm:cache_config_info{block_size="16",num_gpu_blocks="1000",num_cpu_blocks="None"} 1.0
"""


# --------------------------------------------------------------------------- #
# Text exposition parser                                                      #
# --------------------------------------------------------------------------- #


def test_parser_groups_samples_and_ignores_comments() -> None:
    families = parse_prometheus_text(VLLM_V0_METRICS)
    running = families["vllm:num_requests_running"]
    assert [(s.value, dict(s.labels)) for s in running] == [
        (12.0, {"model_name": "mistralai/Mistral-7B-Instruct-v0.2"})
    ]
    assert families["process_resident_memory_bytes"][0].labels == {}
    assert families["vllm:time_to_first_token_seconds_bucket"][1].labels["le"] == "+Inf"


def test_parser_unescapes_label_values() -> None:
    sample = parse_prometheus_text('m{a="q\\"uote",b="back\\\\slash",c="new\\nline",} 1\n')["m"][0]
    assert dict(sample.labels) == {"a": 'q"uote', "b": "back\\slash", "c": "new\nline"}


def test_parser_accepts_special_float_values() -> None:
    families = parse_prometheus_text("a NaN\nb +Inf\nc -Inf\n")
    assert families["b"][0].value == float("inf")
    assert families["c"][0].value == float("-inf")


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("vllm:num_requests_running\n", "not a metric sample"),
        ("1metric 1\n", "not a metric sample"),
        ("metric one\n", "invalid sample value"),
        ("metric{a=unquoted} 1\n", "malformed label set"),
        ('metric{a="x" b="y"} 1\n', "malformed label set"),
    ],
)
def test_parser_rejects_malformed_lines(text: str, match: str) -> None:
    with pytest.raises(MetricsParseError, match=match):
        parse_prometheus_text(text)


def test_metrics_parse_error_is_a_protocol_error() -> None:
    assert issubclass(MetricsParseError, EngineProtocolError)


# --------------------------------------------------------------------------- #
# vLLM mapping                                                                #
# --------------------------------------------------------------------------- #


def test_v0_metrics_map_to_cluster_telemetry() -> None:
    telemetry = telemetry_from_prometheus(VLLM_V0_METRICS)
    assert (telemetry.active_requests, telemetry.waiting_requests, telemetry.swapped_requests) == (
        12,
        3,
        1,
    )
    assert telemetry.gpu_cache_usage_pct == 25.0
    assert telemetry.cpu_cache_usage_pct == 12.5
    kv = telemetry.kv_cache
    assert (kv.block_size, kv.total_gpu_blocks, kv.used_gpu_blocks) == (16, 2048, 512)
    assert (kv.total_cpu_blocks, kv.used_cpu_blocks) == (512, 64)


def test_v1_metrics_sum_counts_and_average_usage_across_series() -> None:
    telemetry = telemetry_from_prometheus(VLLM_V1_METRICS)
    assert (telemetry.active_requests, telemetry.waiting_requests) == (10, 2)
    assert telemetry.swapped_requests == 0
    assert telemetry.gpu_cache_usage_pct == pytest.approx(60.0)
    assert telemetry.cpu_cache_usage_pct == 0.0
    assert telemetry.kv_cache.used_gpu_blocks == 600
    assert telemetry.kv_cache.total_cpu_blocks == 0


def _without(text: str, metric: str) -> str:
    return "\n".join(line for line in text.splitlines() if not line.startswith(metric))


@pytest.mark.parametrize(
    ("text", "match"),
    [
        (_without(VLLM_V0_METRICS, "vllm:cache_config_info"), "cache_config_info is missing"),
        (_without(VLLM_V0_METRICS, "vllm:num_requests_running"), "num_requests_running is missing"),
        (_without(VLLM_V0_METRICS, "vllm:gpu_cache_usage_perc"), "gpu_cache_usage_perc or"),
        (
            VLLM_V0_METRICS.replace('block_size="16",', ""),
            "lacks a usable 'block_size'",
        ),
        (
            VLLM_V0_METRICS.replace('num_gpu_blocks="2048"', 'num_gpu_blocks="many"'),
            "is not an integer",
        ),
        (
            VLLM_V0_METRICS.replace("} 3.0", "} 2.5"),
            "must be a non-negative integer",
        ),
        (
            VLLM_V0_METRICS.replace("} 3.0", "} -1.0"),
            "must be a non-negative integer",
        ),
        (VLLM_V0_METRICS.replace("} 12.0", "} NaN"), "non-finite"),
        (VLLM_V0_METRICS.replace("} 0.25", "} 1.5"), "violate the telemetry contract"),
    ],
)
def test_incomplete_or_inconsistent_metrics_fail_closed(text: str, match: str) -> None:
    with pytest.raises(MetricsParseError, match=match):
        telemetry_from_prometheus(text)
