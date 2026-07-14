"""Completion and deduplication for bar, tick, and polling streams."""

from __future__ import annotations

import math
import re
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from tradelab.engine.execution import estimate_bar_ms
from tradelab.errors import ValidationError
from tradelab.utils.time import is_session

from .events import EventBus

_INTERVAL = re.compile(r"^(\d+)(m|h|d)$")


def _interval_ms(value: object) -> int:
    match = _INTERVAL.fullmatch(str(value or "1m").strip().lower())
    if match is None:
        return 60_000
    amount = int(match.group(1))
    return amount * {"m": 60_000, "h": 3_600_000, "d": 86_400_000}[match.group(2)]


def _finite(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


class CandleAggregator:
    def __init__(
        self,
        *,
        mode: str = "stream",
        interval: str = "1m",
        grace_ms: object = 5_000,
        session: str = "AUTO",
    ) -> None:
        grace = _finite(grace_ms)
        if grace is None or grace < 0:
            raise ValidationError("grace_ms must be finite and non-negative")
        self.mode = mode
        self.interval = interval
        self.grace_ms = grace
        self.session = session
        self.interval_ms: float = _interval_ms(interval)
        self.current: dict[str, float] | None = None
        self.last_emitted_time = -math.inf
        self._events = EventBus()

    def on_bar(self, handler: Callable[[dict[str, Any]], object]) -> Callable[[], None]:
        return self._events.on("bar", handler)

    def emit_bar(self, bar: Mapping[str, Any] | None) -> None:
        if not isinstance(bar, Mapping):
            return
        timestamp = _finite(bar.get("time"))
        if timestamp is None or timestamp <= self.last_emitted_time:
            return
        self.last_emitted_time = timestamp
        self._events.emit("bar", dict(bar))

    def process_bar(self, bar: Mapping[str, Any], *, is_final: bool = True) -> None:
        if self.mode != "stream" or is_final:
            self.emit_bar(bar)

    def process_polled_bars(self, bars: Sequence[Mapping[str, Any]] = ()) -> None:
        for bar in sorted(bars, key=lambda item: float(item.get("time", -math.inf))):
            self.emit_bar(bar)

    def process_tick(self, raw_tick: Mapping[str, object]) -> None:
        timestamp = _finite(raw_tick.get("time"))
        price = next(
            (
                number
                for key in ("price", "last", "close", "bid", "ask")
                if (number := _finite(raw_tick.get(key))) is not None
            ),
            None,
        )
        volume = _finite(raw_tick.get("size", raw_tick.get("volume", 0)))
        if timestamp is None or price is None:
            return
        start = math.floor(timestamp / self.interval_ms) * self.interval_ms
        if self.current is None:
            self.current = self._new_bar(start, price, volume or 0, timestamp)
            return
        if start == self.current["time"]:
            self.current["high"] = max(self.current["high"], price)
            self.current["low"] = min(self.current["low"], price)
            self.current["close"] = price
            self.current["volume"] += volume or 0
            self.current["_lastTickTime"] = timestamp
        elif start > self.current["time"]:
            self.emit_bar(self._public_current())
            self.current = self._new_bar(start, price, volume or 0, timestamp)

    @staticmethod
    def _new_bar(start: float, price: float, volume: float, timestamp: float) -> dict[str, float]:
        return {
            "time": start,
            "open": price,
            "high": price,
            "low": price,
            "close": price,
            "volume": volume,
            "_lastTickTime": timestamp,
        }

    def _public_current(self) -> dict[str, float]:
        assert self.current is not None
        return {
            key: self.current[key] for key in ("time", "open", "high", "low", "close", "volume")
        }

    def force_close(self, time_ms: object | None = None) -> None:
        if self.current is None:
            return
        timestamp = time.time_ns() // 1_000_000 if time_ms is None else _finite(time_ms)
        if timestamp is None:
            raise ValidationError("time_ms must be finite")
        deadline = self.current["time"] + self.interval_ms + self.grace_ms
        session_open = is_session(self.current["time"] + self.interval_ms, self.session)
        if timestamp >= deadline or not session_open:
            self.emit_bar(self._public_current())
            self.current = None

    def estimate_from_series(self, candles: Sequence[Mapping[str, Any]]) -> float:
        estimated = estimate_bar_ms(candles)
        if math.isfinite(estimated) and estimated > 0:
            self.interval_ms = estimated
        return self.interval_ms


def create_candle_aggregator(**options: Any) -> CandleAggregator:
    return CandleAggregator(**options)
