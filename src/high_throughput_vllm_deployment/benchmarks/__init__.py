"""Load generation and benchmark aggregation.

Only the pure building blocks are re-exported here. ``runner`` and
``mock_server`` are CLI entry points (``python -m``) and ``locustfile``
must never be imported implicitly: importing Locust monkey-patches the whole
process with gevent. Load it via the ``locust`` CLI, or with
``LOCUST_SKIP_MONKEY_PATCH=1`` in tests.
"""

from high_throughput_vllm_deployment.benchmarks.metrics_collector import (
    MetricsCollector,
    RequestRecord,
    categorize_exception,
    categorize_status,
    percentile,
    summarize,
)
from high_throughput_vllm_deployment.benchmarks.observer import StreamObserver, record_from_response
from high_throughput_vllm_deployment.benchmarks.workload import (
    ArrivalPattern,
    PromptTier,
    WorkItem,
    WorkloadProfile,
    generate_workload,
)

__all__ = [
    "ArrivalPattern",
    "MetricsCollector",
    "PromptTier",
    "RequestRecord",
    "StreamObserver",
    "WorkItem",
    "WorkloadProfile",
    "categorize_exception",
    "categorize_status",
    "generate_workload",
    "percentile",
    "record_from_response",
    "summarize",
]
