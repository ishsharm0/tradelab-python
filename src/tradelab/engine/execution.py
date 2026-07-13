"""Pure, deterministic bar execution primitives."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, cast

from tradelab.errors import ValidationError
from tradelab.utils.time import NEW_YORK, minutes_et


def _number(value: object, name: str, default: float | None = None) -> float:
    if value is None and default is not None:
        return default
    if isinstance(value, bool):
        raise ValidationError(f"{name} must be numeric", context={name: value})
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ValidationError(f"{name} must be numeric", context={name: value}) from error
    if not math.isfinite(number):
        raise ValidationError(f"{name} must be finite", context={name: value})
    return number


def _cost_number(costs: Mapping[str, object], key: str, fallback: float) -> float:
    value = costs.get(key, fallback)
    return (
        float(value)
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        else fallback
    )


def apply_fill(
    price: float,
    side: str,
    *,
    slippage_bps: float = 0,
    fee_bps: float = 0,
    kind: str = "market",
    qty: float = 0,
    costs: Mapping[str, object] | None = None,
) -> dict[str, float]:
    """Apply JS-compatible spread, slippage, and commission to one fill."""
    base = _number(price, "price")
    model = costs or {}
    effective_slippage = _cost_number(model, "slippageBps", slippage_bps)
    if isinstance(model.get("slippageByKind"), Mapping):
        by_kind = cast(Mapping[str, Any], model["slippageByKind"])
        candidate = by_kind.get(kind)
        if (
            isinstance(candidate, (int, float))
            and not isinstance(candidate, bool)
            and math.isfinite(float(candidate))
        ):
            effective_slippage = float(candidate)
    elif kind == "limit":
        effective_slippage *= 0.25
    elif kind == "stop":
        effective_slippage *= 1.25
    half_spread = _cost_number(model, "spreadBps", 0) / 2
    slippage = (effective_slippage + half_spread) / 10_000 * base
    filled = base + slippage if side == "long" else base - slippage
    commission_bps = _cost_number(model, "commissionBps", fee_bps)
    per_unit = commission_bps / 10_000 * abs(filled) + _cost_number(model, "commissionPerUnit", 0)
    size = max(0.0, _number(qty, "qty"))
    gross = per_unit * size + _cost_number(model, "commissionPerOrder", 0)
    total = max(_cost_number(model, "minCommission", 0), gross)
    return {"price": filled, "fee": total / size if size > 0 else per_unit, "fee_total": total}


def clamp_stop(
    market_price: float, proposed_stop: float, side: str, oco: Mapping[str, object]
) -> float:
    eps = _cost_number(oco, "clampEpsBps", 0.25) / 10_000 * market_price
    return (
        min(proposed_stop, market_price - eps)
        if side == "long"
        else max(proposed_stop, market_price + eps)
    )


def touched_limit(
    side: str, limit_price: float, bar: Mapping[str, object], mode: str = "intrabar"
) -> bool:
    if mode == "close":
        return (
            _number(bar["close"], "close") <= limit_price
            if side == "long"
            else _number(bar["close"], "close") >= limit_price
        )
    return (
        _number(bar["low"], "low") <= limit_price
        if side == "long"
        else _number(bar["high"], "high") >= limit_price
    )


def oco_exit_check(
    *,
    side: str,
    stop: float,
    tp: float,
    bar: Mapping[str, object],
    mode: str = "intrabar",
    tie_break: str = "pessimistic",
) -> dict[str, str | float | None]:
    """Return the deterministic OCO winner for a bar."""
    if mode == "close":
        close = _number(bar["close"], "close")
        if (side == "long" and close <= stop) or (side == "short" and close >= stop):
            return {"hit": "SL", "px": stop}
        if (side == "long" and close >= tp) or (side == "short" and close <= tp):
            return {"hit": "TP", "px": tp}
        return {"hit": None, "px": None}
    hit_stop = (
        _number(bar["low"], "low") <= stop
        if side == "long"
        else _number(bar["high"], "high") >= stop
    )
    hit_target = (
        _number(bar["high"], "high") >= tp if side == "long" else _number(bar["low"], "low") <= tp
    )
    if hit_stop and hit_target:
        return {"hit": "TP", "px": tp} if tie_break == "optimistic" else {"hit": "SL", "px": stop}
    if hit_stop:
        return {"hit": "SL", "px": stop}
    if hit_target:
        return {"hit": "TP", "px": tp}
    return {"hit": None, "px": None}


def is_eod_bar(time_ms: int | float) -> bool:
    return minutes_et(time_ms) >= 16 * 60


def round_step(value: float, step: float = 0.001) -> float:
    return math.floor(value / step) * step


def estimate_bar_ms(candles: Sequence[Mapping[str, Any]]) -> float:
    deltas = [
        float(candles[index]["time"]) - float(candles[index - 1]["time"])
        for index in range(1, min(len(candles), 500))
        if float(candles[index]["time"]) > float(candles[index - 1]["time"])
    ]
    if not deltas:
        return 5 * 60 * 1_000
    deltas.sort()
    middle = len(deltas) // 2
    median = deltas[middle] if len(deltas) % 2 else (deltas[middle - 1] + deltas[middle]) / 2
    return max(60_000, min(median, 60 * 60_000))


def day_key_utc(time_ms: float) -> str:
    return datetime.fromtimestamp(time_ms / 1_000, UTC).strftime("%Y-%m-%d")


def day_key_et(time_ms: float) -> str:
    return datetime.fromtimestamp(time_ms / 1_000, UTC).astimezone(NEW_YORK).strftime("%Y-%m-%d")
