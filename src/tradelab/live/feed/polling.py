"""Deduplicating REST polling feed with cancellation-safe lifecycle."""

from __future__ import annotations

import asyncio
import inspect
import math
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import suppress
from typing import Any, Protocol

from tradelab.errors import ValidationError

from .base import FeedHandler, FeedProvider, Subscription


class HistoryBroker(Protocol):
    def get_historical_bars(
        self, symbol: str, interval: str, limit: int = 200
    ) -> Awaitable[Sequence[Mapping[str, object]]]: ...


def _number(value: object, default: float, minimum: float, name: str) -> float:
    if isinstance(value, bool):
        raise ValidationError(f"{name} must be finite")
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as error:
        raise ValidationError(f"{name} must be finite") from error
    if not math.isfinite(result):
        raise ValidationError(f"{name} must be finite")
    return max(minimum, result or default)


def _bar_time(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return -math.inf
    number = float(value)
    return number if math.isfinite(number) else -math.inf


class PollingFeed(FeedProvider):
    def __init__(
        self,
        *,
        broker: HistoryBroker,
        poll_interval_ms: object = 60_000,
        default_bars_per_poll: object = 2,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.broker = broker
        self.poll_interval_ms = _number(poll_interval_ms, 60_000, 500, "poll_interval_ms")
        bars = _number(default_bars_per_poll, 2, 1, "default_bars_per_poll")
        self.default_bars_per_poll = int(bars)
        self._sleep = sleep
        self.bar_subscriptions: dict[str, list[FeedHandler]] = {}
        self.tick_subscriptions: dict[str, list[FeedHandler]] = {}
        self.last_emitted_by_stream: dict[str, float] = {}
        self.connected = False
        self._poll_task: asyncio.Task[None] | None = None
        self._poll_lock = asyncio.Lock()

    @staticmethod
    def _key(symbol: str, interval: str) -> str:
        return f"{symbol}::{interval}"

    @property
    def polling(self) -> bool:
        return self._poll_task is not None and not self._poll_task.done()

    async def connect(self) -> None:
        self.connected = True

    async def disconnect(self) -> None:
        self.connected = False
        await self.stop_polling()

    def _subscribe(
        self, values: dict[str, list[FeedHandler]], key: str, handler: FeedHandler
    ) -> Subscription:
        if not callable(handler):
            raise ValidationError("feed handler must be callable")
        values.setdefault(key, []).append(handler)

        def unsubscribe() -> None:
            values[key] = [
                candidate for candidate in values.get(key, []) if candidate is not handler
            ]

        return Subscription(unsubscribe)

    async def subscribe_bars(
        self, symbol: str, interval: str, handler: FeedHandler
    ) -> Subscription:
        return self._subscribe(self.bar_subscriptions, self._key(symbol, interval), handler)

    async def subscribe_ticks(self, symbol: str, handler: FeedHandler) -> Subscription:
        return self._subscribe(self.tick_subscriptions, symbol, handler)

    async def get_historical_bars(
        self, symbol: str, interval: str, count: int
    ) -> Sequence[Mapping[str, object]]:
        return await self.broker.get_historical_bars(symbol, interval, count)

    async def poll_once(self) -> None:
        async with self._poll_lock:
            for stream in tuple(self.bar_subscriptions):
                symbol, interval = stream.split("::", 1)
                bars = await self.broker.get_historical_bars(
                    symbol, interval, self.default_bars_per_poll
                )
                ordered = sorted(
                    (dict(bar) for bar in bars), key=lambda bar: _bar_time(bar.get("time"))
                )
                last_seen = self.last_emitted_by_stream.get(stream, -math.inf)
                next_bars = [bar for bar in ordered if _bar_time(bar.get("time")) > last_seen]
                for bar in next_bars:
                    for handler in tuple(self.bar_subscriptions.get(stream, ())):
                        outcome = handler(dict(bar))
                        if inspect.isawaitable(outcome):
                            await outcome
                if next_bars:
                    self.last_emitted_by_stream[stream] = _bar_time(next_bars[-1].get("time"))

    async def _poll_loop(self) -> None:
        try:
            while self.connected:
                await self._sleep(self.poll_interval_ms / 1_000)
                if self.connected:
                    with suppress(Exception):
                        await self.poll_once()
        except asyncio.CancelledError:
            raise

    def start_polling(self) -> None:
        if self.polling:
            return
        if not self.connected:
            raise ValidationError("polling feed must be connected before polling")
        self._poll_task = asyncio.create_task(self._poll_loop())

    async def stop_polling(self) -> None:
        task, self._poll_task = self._poll_task, None
        if task is None:
            return
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


def create_polling_feed(**options: Any) -> PollingFeed:
    return PollingFeed(**options)
