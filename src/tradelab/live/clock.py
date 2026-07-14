"""Broker-synchronized clock with an injectable local time source."""

from __future__ import annotations

import math
import time
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from tradelab.errors import ValidationError


class ServerClock(Protocol):
    def get_server_time(self) -> Awaitable[object]: ...


def _system_now_ms() -> int:
    return time.time_ns() // 1_000_000


def _nonnegative(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValidationError(f"{name} must be finite and non-negative")
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as error:
        raise ValidationError(f"{name} must be finite and non-negative") from error
    if not math.isfinite(number) or number < 0:
        raise ValidationError(f"{name} must be finite and non-negative")
    return number


class BrokerClock:
    """Maintain broker-minus-local offset without changing the system clock."""

    def __init__(
        self,
        *,
        warn_threshold_ms: object = 2_000,
        now_ms: Callable[[], int] = _system_now_ms,
    ) -> None:
        self.warn_threshold_ms = _nonnegative(warn_threshold_ms, "warn_threshold_ms")
        self._now_ms = now_ms
        self.offset_ms = 0.0
        self.synced_at: int | None = None

    def now(self) -> float:
        return self._now_ms() + self.offset_ms

    def get_offset_ms(self) -> float:
        return self.offset_ms

    async def sync_with_broker(self, broker: object | None) -> dict[str, object]:
        getter = getattr(broker, "get_server_time", None)
        server_time: float | None = None
        if callable(getter):
            try:
                candidate = await getter()
                if not isinstance(candidate, bool):
                    numeric = float(candidate)
                    if math.isfinite(numeric):
                        server_time = numeric
            except Exception:
                server_time = None
        local_time = self._now_ms()
        self.offset_ms = server_time - local_time if server_time is not None else 0.0
        self.synced_at = local_time
        warning = None
        if abs(self.offset_ms) > self.warn_threshold_ms:
            warning = (
                f"clock offset {self.offset_ms:g}ms exceeds threshold {self.warn_threshold_ms:g}ms"
            )
        return {
            "serverTime": server_time,
            "localTime": local_time,
            "offsetMs": self.offset_ms,
            "warning": warning,
        }


def create_clock(**options: Any) -> BrokerClock:
    return BrokerClock(**options)
