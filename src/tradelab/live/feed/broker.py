"""Feed provider delegating to a structural broker adapter."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, Protocol

from .base import FeedHandler, FeedProvider, Subscription


class FeedBroker(Protocol):
    def subscribe_bars(
        self, symbol: str, interval: str, handler: FeedHandler
    ) -> Awaitable[Callable[[], None] | Subscription]: ...

    def subscribe_trades(
        self, symbol: str, handler: FeedHandler
    ) -> Awaitable[Callable[[], None] | Subscription]: ...

    def get_historical_bars(
        self, symbol: str, interval: str, limit: int = 200
    ) -> Awaitable[Sequence[Mapping[str, object]]]: ...


def _subscription(value: Callable[[], None] | Subscription) -> Subscription:
    return value if isinstance(value, Subscription) else Subscription(value)


class BrokerFeed(FeedProvider):
    def __init__(self, *, broker: FeedBroker) -> None:
        self.broker = broker

    async def connect(self) -> None:
        return None

    async def disconnect(self) -> None:
        return None

    async def subscribe_bars(
        self, symbol: str, interval: str, handler: Callable[[dict[str, Any]], object]
    ) -> Subscription:
        return _subscription(await self.broker.subscribe_bars(symbol, interval, handler))

    async def subscribe_ticks(
        self, symbol: str, handler: Callable[[dict[str, Any]], object]
    ) -> Subscription:
        return _subscription(await self.broker.subscribe_trades(symbol, handler))

    async def get_historical_bars(
        self, symbol: str, interval: str, count: int
    ) -> Sequence[Mapping[str, object]]:
        return await self.broker.get_historical_bars(symbol, interval, count)


def create_broker_feed(*, broker: FeedBroker) -> BrokerFeed:
    return BrokerFeed(broker=broker)
