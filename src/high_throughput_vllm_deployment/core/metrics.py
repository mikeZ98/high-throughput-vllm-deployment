"""Prometheus text exposition parsing and vLLM ``/metrics`` -> :class:`ClusterTelemetry` mapping.

Both the V0 (``gpu_cache_usage_perc``, ``cpu_cache_usage_perc``,
``num_requests_swapped``) and V1 (``kv_cache_usage_perc``, no swap) metric
sets are understood. vLLM reports cache usage as a fraction in ``[0, 1]``;
:class:`ClusterTelemetry` stores percentages, and used block counts are
derived from the fraction and the ``cache_config_info`` block totals.
Series of the same metric (e.g. one per ``model_name`` or data-parallel rank)
are summed for counts and averaged for usage fractions.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from pydantic import ValidationError

from high_throughput_vllm_deployment.core.errors import EngineProtocolError
from high_throughput_vllm_deployment.schemas.telemetry import ClusterTelemetry, KvCacheBlocks

_SAMPLE: Final = re.compile(
    r"(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
    r"(?:\{(?P<labels>.*)\})?"
    r"[ \t]+(?P<value>\S+)"
    r"(?:[ \t]+-?[0-9]+)?"
)
_LABEL: Final = re.compile(
    r'[ \t]*([a-zA-Z_][a-zA-Z0-9_]*)[ \t]*=[ \t]*"((?:[^"\\]|\\.)*)"[ \t]*(?:,|$)'
)
_ESCAPES: Final = {"\\\\": "\\", '\\"': '"', "\\n": "\n"}

RUNNING: Final = "vllm:num_requests_running"
WAITING: Final = "vllm:num_requests_waiting"
SWAPPED: Final = "vllm:num_requests_swapped"
GPU_USAGE: Final = ("vllm:gpu_cache_usage_perc", "vllm:kv_cache_usage_perc")
CPU_USAGE: Final = "vllm:cpu_cache_usage_perc"
CACHE_INFO: Final = "vllm:cache_config_info"


class MetricsParseError(EngineProtocolError):
    """The metrics body is not valid Prometheus text exposition or lacks vLLM metrics."""


@dataclass(frozen=True, slots=True)
class Sample:
    name: str
    labels: Mapping[str, str]
    value: float


def _parse_labels(raw: str, lineno: int) -> dict[str, str]:
    labels: dict[str, str] = {}
    pos = 0
    raw = raw.rstrip(", \t")
    while pos < len(raw):
        match = _LABEL.match(raw, pos)
        if match is None:
            raise MetricsParseError(f"line {lineno}: malformed label set {{{raw}}}")
        value = re.sub(r"\\[\\\"n]", lambda m: _ESCAPES[m.group(0)], match.group(2))
        labels[match.group(1)] = value
        pos = match.end()
    return labels


def parse_prometheus_text(text: str) -> dict[str, list[Sample]]:
    """Parse a text-format exposition into samples grouped by metric name."""
    families: dict[str, list[Sample]] = defaultdict(list)
    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = _SAMPLE.fullmatch(stripped)
        if match is None:
            raise MetricsParseError(f"line {lineno}: not a metric sample: {stripped!r}")
        try:
            value = float(match.group("value"))
        except ValueError:
            raise MetricsParseError(
                f"line {lineno}: invalid sample value {match.group('value')!r}"
            ) from None
        labels = _parse_labels(match.group("labels") or "", lineno)
        families[match.group("name")].append(Sample(match.group("name"), labels, value))
    return dict(families)


def _finite_values(families: Mapping[str, Sequence[Sample]], name: str) -> list[float]:
    values = [sample.value for sample in families.get(name, ())]
    if any(not math.isfinite(value) for value in values):
        raise MetricsParseError(f"{name} has a non-finite value")
    return values


def _count(families: Mapping[str, Sequence[Sample]], name: str, *, required: bool) -> int:
    values = _finite_values(families, name)
    if not values:
        if required:
            raise MetricsParseError(f"required metric {name} is missing")
        return 0
    total = sum(values)
    if total != int(total) or total < 0:
        raise MetricsParseError(f"{name} must be a non-negative integer, got {total}")
    return int(total)


def _mean_fraction(families: Mapping[str, Sequence[Sample]], names: Sequence[str]) -> float | None:
    for name in names:
        values = _finite_values(families, name)
        if values:
            return sum(values) / len(values)
    return None


def _label_int(labels: Mapping[str, str], key: str, *, required: bool) -> int:
    raw = labels.get(key)
    if raw is None or raw in {"", "None"}:
        if required:
            raise MetricsParseError(f"{CACHE_INFO} lacks a usable {key!r} label")
        return 0
    try:
        return int(raw)
    except ValueError:
        raise MetricsParseError(f"{CACHE_INFO} label {key}={raw!r} is not an integer") from None


def _used_blocks(fraction: float, total: int) -> int:
    return min(total, max(0, round(fraction * total)))


def telemetry_from_samples(families: Mapping[str, Sequence[Sample]]) -> ClusterTelemetry:
    """Map parsed vLLM metric families onto the :class:`ClusterTelemetry` contract."""
    info = families.get(CACHE_INFO)
    if not info:
        raise MetricsParseError(f"required metric {CACHE_INFO} is missing")
    labels = info[0].labels
    block_size = _label_int(labels, "block_size", required=True)
    total_gpu = _label_int(labels, "num_gpu_blocks", required=True)
    total_cpu = _label_int(labels, "num_cpu_blocks", required=False)

    gpu_fraction = _mean_fraction(families, GPU_USAGE)
    if gpu_fraction is None:
        raise MetricsParseError(f"required metric {' or '.join(GPU_USAGE)} is missing")
    cpu_fraction = _mean_fraction(families, (CPU_USAGE,)) or 0.0

    try:
        return ClusterTelemetry(
            active_requests=_count(families, RUNNING, required=True),
            waiting_requests=_count(families, WAITING, required=True),
            swapped_requests=_count(families, SWAPPED, required=False),
            gpu_cache_usage_pct=gpu_fraction * 100.0,
            cpu_cache_usage_pct=cpu_fraction * 100.0,
            kv_cache=KvCacheBlocks(
                block_size=block_size,
                total_gpu_blocks=total_gpu,
                used_gpu_blocks=_used_blocks(gpu_fraction, total_gpu),
                total_cpu_blocks=total_cpu,
                used_cpu_blocks=_used_blocks(cpu_fraction, total_cpu),
            ),
        )
    except ValidationError as exc:
        raise MetricsParseError(f"metrics violate the telemetry contract: {exc}") from exc


def telemetry_from_prometheus(text: str) -> ClusterTelemetry:
    """Parse a vLLM ``/metrics`` body into :class:`ClusterTelemetry`."""
    return telemetry_from_samples(parse_prometheus_text(text))
