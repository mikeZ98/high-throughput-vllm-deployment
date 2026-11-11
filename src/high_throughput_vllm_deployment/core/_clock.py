"""Injectable time source shared by the client, the mock engine and collectors."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

Sleep = Callable[[float], Awaitable[None]]
"""``asyncio.sleep``-compatible coroutine function."""


async def instant_sleep(_seconds: float) -> None:
    """Yield to the event loop without waiting: simulated time, real ordering."""
    await asyncio.sleep(0)
