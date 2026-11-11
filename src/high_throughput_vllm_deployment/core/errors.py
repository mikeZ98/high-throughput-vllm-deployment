"""Engine error taxonomy shared by every :class:`InferenceEnginePort` implementation.

Callers handle one closed hierarchy regardless of whether the engine is the
real HTTP server or the in-process mock; transport-library exceptions never
leak through the port.
"""

from __future__ import annotations


class EngineError(Exception):
    """Base class for all engine failures."""


class EngineUnavailableError(EngineError):
    """The engine could not be reached, or kept failing transiently until retries ran out."""


class EngineTimeoutError(EngineError):
    """A connect, read, write or pool timeout elapsed."""


class EngineProtocolError(EngineError):
    """The engine answered with a payload that violates the wire contract."""


class EngineHTTPError(EngineError):
    """The engine answered with a non-retryable error status."""

    def __init__(self, message: str, *, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class EngineRequestError(EngineHTTPError):
    """The request itself was rejected (HTTP 4xx, or a mock-side validation failure)."""

    def __init__(self, message: str, *, status_code: int = 400) -> None:
        super().__init__(message, status_code=status_code)
