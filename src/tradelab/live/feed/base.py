"""Typed asynchronous feed contract."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any


def _missing(method: str) -> NotImplementedError:
    return NotImplementedError(f"FeedProvider.{method}() not implemented")


class Subscription:
    """Idempotent subscription handle compatible with callable broker handles."""

    def __init__(self, unsubscribe: Callable[[], None]) -> None:
        self._unsubscribe = unsubscribe
        self._closed = False

    def unsubscribe(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._unsubscribe()

    def __call__(self) -> None:
        self.unsubscribe()


FeedHandler = Callable[[dict[str, Any]], object]


class FeedProvider:
    async def connect(self) -> None:
        raise _missing("connect")

    async def disconnect(self) -> None:
        raise _missing("disconnect")

    async def subscribe_bars(
        self, symbol: str, interval: str, handler: FeedHandler
    ) -> Subscription:
        raise _missing("subscribeBars")

    async def subscribe_ticks(self, symbol: str, handler: FeedHandler) -> Subscription:
        raise _missing("subscribeTicks")

    async def get_historical_bars(
        self, symbol: str, interval: str, count: int
    ) -> Sequence[Mapping[str, object]]:
        raise _missing("getHistoricalBars")
