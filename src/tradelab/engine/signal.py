"""Signal normalization and safe callback invocation."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from datetime import UTC, datetime

from tradelab.errors import StrategyError


def _nullish(*values: object) -> object:
    return next((value for value in values if value is not None), None)


def as_number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def normalize_side(value: object) -> str | None:
    return (
        {"long": "long", "buy": "long", "short": "short", "sell": "short"}.get(value)
        if isinstance(value, str)
        else None
    )


def normalize_signal(
    raw: object, bar: Mapping[str, object], fallback_r: float
) -> dict[str, object] | None:
    if not isinstance(raw, Mapping):
        return None
    side = normalize_side(_nullish(raw.get("side"), raw.get("direction"), raw.get("action")))
    if side is None:
        return None
    entry = as_number(_nullish(raw.get("entry"), raw.get("limit"), raw.get("price")))
    entry = as_number(bar.get("close")) if entry is None else entry
    stop = as_number(_nullish(raw.get("stop"), raw.get("stopLoss"), raw.get("sl")))
    if entry is None or stop is None or not abs(entry - stop) > 0:
        return None
    target = as_number(_nullish(raw.get("takeProfit"), raw.get("target"), raw.get("tp")))
    rr = as_number(_nullish(raw.get("_rr"), raw.get("rr")))
    target_r = fallback_r if rr is None else rr
    if target is None and target_r > 0:
        target = (
            entry + abs(entry - stop) * target_r
            if side == "long"
            else entry - abs(entry - stop) * target_r
        )
    if target is None:
        return None
    result = dict(raw)
    result.update(
        {
            "side": side,
            "entry": entry,
            "stop": stop,
            "takeProfit": target,
            "qty": as_number(_nullish(raw.get("qty"), raw.get("size"))),
            "riskPct": as_number(raw.get("riskPct")),
            "riskFraction": as_number(raw.get("riskFraction")),
            "_rr": rr if rr is not None else raw.get("_rr"),
            "_initRisk": as_number(raw.get("_initRisk"))
            if as_number(raw.get("_initRisk")) is not None
            else raw.get("_initRisk"),
        }
    )
    return result


def call_signal_with_context(
    signal: Callable[[dict[str, object]], object],
    context: dict[str, object],
    index: int,
    bar: Mapping[str, object],
    symbol: str,
) -> object:
    try:
        return signal(context)
    except Exception as error:
        time = bar.get("time")
        formatted = "invalid-time"
        if isinstance(time, (int, float)) and not isinstance(time, bool):
            try:
                numeric_time = float(time)
                if math.isfinite(numeric_time):
                    formatted = (
                        datetime.fromtimestamp(numeric_time / 1_000, UTC)
                        .isoformat()
                        .replace("+00:00", ".000Z")
                    )
            except (OverflowError, OSError, ValueError):
                pass
        raise StrategyError(
            f"signal() threw at index={index}, time={formatted}, symbol={symbol}: {error}"
        ) from error
