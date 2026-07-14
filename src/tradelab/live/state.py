"""Persistence facade and broker restart reconciliation."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .storage import StorageProvider


def _system_now_ms() -> int:
    return time.time_ns() // 1_000_000


def _number(value: object) -> float:
    if isinstance(value, bool):
        return 0.0
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def _qty_close_enough(left: object, right: object, tolerance_pct: float = 0.05) -> bool:
    first, second = abs(_number(left)), abs(_number(right))
    if first == second == 0:
        return True
    return abs(first - second) / max(first, second, 1e-12) <= tolerance_pct


class StateManager:
    def __init__(
        self, *, storage: StorageProvider, now_ms: Callable[[], int] = _system_now_ms
    ) -> None:
        self.storage = storage
        self._now_ms = now_ms

    async def load(self, namespace: str) -> Any:
        return await self.storage.load(namespace)

    async def save(self, namespace: str, state: Mapping[str, object]) -> None:
        await self.storage.save(namespace, {**state, "savedAt": self._now_ms()})

    async def append_trade(self, namespace: str, trade: object) -> None:
        await self.storage.append_trade(namespace, trade)

    async def append_equity_point(self, namespace: str, point: object) -> None:
        await self.storage.append_equity_point(namespace, point)

    async def load_trades(self, namespace: str) -> list[Any]:
        return await self.storage.load_trades(namespace)

    async def load_equity_curve(self, namespace: str) -> list[Any]:
        return await self.storage.load_equity_curve(namespace)

    async def clear(self, namespace: str) -> None:
        await self.storage.clear(namespace)

    def reconcile(
        self,
        *,
        persisted_state: Mapping[str, Any] | None,
        broker_positions: Sequence[Mapping[str, Any]] = (),
        symbol: str,
    ) -> dict[str, Any]:
        report: dict[str, Any] = {
            "status": "ok",
            "action": "none",
            "message": "no reconciliation needed",
            "adoptedPosition": None,
            "mismatch": None,
        }
        raw_open = persisted_state.get("openPosition") if persisted_state else None
        persisted_open = raw_open if isinstance(raw_open, Mapping) else None
        broker_position = next(
            (position for position in broker_positions if position.get("symbol") == symbol), None
        )
        if persisted_open is not None and broker_position is not None:
            size = persisted_open.get("size", persisted_open.get("qty"))
            if persisted_open.get("side") == broker_position.get("side") and _qty_close_enough(
                size, broker_position.get("qty")
            ):
                report.update(
                    {
                        "action": "adopt-broker",
                        "message": "persisted and broker positions matched",
                        "adoptedPosition": {
                            **persisted_open,
                            "size": broker_position.get("qty"),
                            "entryFill": broker_position.get(
                                "avgEntry",
                                broker_position.get(
                                    "avg_entry",
                                    persisted_open.get("entryFill", persisted_open.get("entry")),
                                ),
                            ),
                        },
                    }
                )
                return report
            report.update(
                {
                    "status": "error",
                    "action": "mismatch",
                    "message": "persisted and broker positions mismatch",
                    "mismatch": {
                        "persisted": dict(persisted_open),
                        "broker": dict(broker_position),
                    },
                }
            )
            return report
        if persisted_open is not None:
            report.update(
                {
                    "status": "warn",
                    "action": "closed-externally",
                    "message": "persisted open position missing at broker",
                }
            )
        elif broker_position is not None:
            report.update(
                {
                    "status": "warn",
                    "action": "external-position",
                    "message": "broker has external position not present in persisted state",
                }
            )
        return report


def create_state_manager(*, storage: StorageProvider) -> StateManager:
    return StateManager(storage=storage)
