"""Locust entry point for the container image.

The image ships the application as bytecode only, but ``locust -f`` needs a
source file; importing the packaged module registers its event hooks, and the
re-exported user classes are what Locust discovers.
"""

from high_throughput_vllm_deployment.benchmarks.locustfile import (
    VllmChatUser,
    VllmFastChatUser,
)

__all__ = ["VllmChatUser", "VllmFastChatUser"]
