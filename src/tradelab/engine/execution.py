"""Pure, deterministic bar execution primitives."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, cast

from tradelab.errors import ValidationError
from tradelab.utils.time import minutes_et


def _number(value: object, name: str, default: float | None = None) -> float:
    if value is None and default is not None:
        return default
    if isinstance(value, bool):
        raise ValidationError(f"{name} must be numeric", context={name: value})
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as error:
        raise ValidationError(f"{name} must be numeric", context={name: value}) from error
    if not math.isfinite(number):
        raise ValidationError(f"{name} must be finite", context={name: value})
    return number


def _cost_number(costs: Mapping[str, object], key: str, fallback: float) -> float:
    value = costs.get(key, fallback)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return fallback
    try:
        number = float(value)
    except OverflowError:
        return fallback
    return number if math.isfinite(number) else fallback


def _side(value: object) -> str:
    if value not in {"long", "short"}:
        raise ValidationError("side must be long or short", context={"side": value})
    return str(value)


def _mode(value: object) -> str:
    if value not in {"intrabar", "close"}:
        raise ValidationError("mode must be intrabar or close", context={"mode": value})
    return str(value)


def _finite_result(value: float, name: str) -> float:
    if not math.isfinite(value):
        raise ValidationError(f"{name} must remain finite", context={name: value})
    return value


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
    direction = _side(side)
    fallback_slippage = _number(slippage_bps, "slippage_bps")
    fallback_fee = _number(fee_bps, "fee_bps")
    if costs is not None and not isinstance(costs, Mapping):
        raise ValidationError("costs must be a mapping", context={"costs": costs})
    model = costs or {}
    effective_slippage = _cost_number(model, "slippageBps", fallback_slippage)
    kind_override: float | None = None
    if isinstance(model.get("slippageByKind"), Mapping):
        by_kind = cast(Mapping[str, Any], model["slippageByKind"])
        candidate = by_kind.get(kind)
        if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
            try:
                numeric_candidate = float(candidate)
            except OverflowError:
                numeric_candidate = math.nan
            if math.isfinite(numeric_candidate):
                kind_override = numeric_candidate
    if kind_override is not None:
        effective_slippage = kind_override
    else:
        if kind == "limit":
            effective_slippage *= 0.25
        elif kind == "stop":
            effective_slippage *= 1.25
    half_spread = _cost_number(model, "spreadBps", 0) / 2
    slippage = (effective_slippage + half_spread) / 10_000 * base
    filled = _finite_result(
        base + slippage if direction == "long" else base - slippage, "filled price"
    )
    commission_bps = _cost_number(model, "commissionBps", fallback_fee)
    per_unit = commission_bps / 10_000 * abs(filled) + _cost_number(model, "commissionPerUnit", 0)
    size = max(0.0, _number(qty, "qty"))
    gross = per_unit * size + _cost_number(model, "commissionPerOrder", 0)
    total = max(_cost_number(model, "minCommission", 0), gross)
    return {
        "price": filled,
        "fee": _finite_result(total / size if size > 0 else per_unit, "fee"),
        "fee_total": _finite_result(total, "fee_total"),
    }


def clamp_stop(
    market_price: float, proposed_stop: float, side: str, oco: Mapping[str, object]
) -> float:
    market = _number(market_price, "market_price")
    proposed = _number(proposed_stop, "proposed_stop")
    direction = _side(side)
    if not isinstance(oco, Mapping):
        raise ValidationError("oco must be a mapping", context={"oco": oco})
    eps = _cost_number(oco, "clampEpsBps", 0.25) / 10_000 * market
    return _finite_result(
        min(proposed, market - eps) if direction == "long" else max(proposed, market + eps),
        "stop",
    )


def touched_limit(
    side: str, limit_price: float, bar: Mapping[str, object], mode: str = "intrabar"
) -> bool:
    direction = _side(side)
    trigger = _mode(mode)
    limit = _number(limit_price, "limit_price")
    if not isinstance(bar, Mapping):
        raise ValidationError("bar must be a mapping", context={"bar": bar})
    if trigger == "close":
        return (
            _number(bar.get("close"), "close") <= limit
            if direction == "long"
            else _number(bar.get("close"), "close") >= limit
        )
    return (
        _number(bar.get("low"), "low") <= limit
        if direction == "long"
        else _number(bar.get("high"), "high") >= limit
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
    direction = _side(side)
    trigger = _mode(mode)
    stop_value = _number(stop, "stop")
    target_value = _number(tp, "tp")
    if tie_break not in {"pessimistic", "optimistic"}:
        raise ValidationError(
            "tie_break must be pessimistic or optimistic", context={"tie_break": tie_break}
        )
    if not isinstance(bar, Mapping):
        raise ValidationError("bar must be a mapping", context={"bar": bar})
    if trigger == "close":
        close = _number(bar.get("close"), "close")
        if (direction == "long" and close <= stop_value) or (
            direction == "short" and close >= stop_value
        ):
            return {"hit": "SL", "px": stop_value}
        if (direction == "long" and close >= target_value) or (
            direction == "short" and close <= target_value
        ):
            return {"hit": "TP", "px": target_value}
        return {"hit": None, "px": None}
    hit_stop = (
        _number(bar.get("low"), "low") <= stop_value
        if direction == "long"
        else _number(bar.get("high"), "high") >= stop_value
    )
    hit_target = (
        _number(bar.get("high"), "high") >= target_value
        if direction == "long"
        else _number(bar.get("low"), "low") <= target_value
    )
    if hit_stop and hit_target:
        return (
            {"hit": "TP", "px": target_value}
            if tie_break == "optimistic"
            else {"hit": "SL", "px": stop_value}
        )
    if hit_stop:
        return {"hit": "SL", "px": stop_value}
    if hit_target:
        return {"hit": "TP", "px": target_value}
    return {"hit": None, "px": None}


def is_eod_bar(time_ms: int | float) -> bool:
    return minutes_et(time_ms) >= 16 * 60


def round_step(value: float, step: float = 0.001) -> float:
    number = _number(value, "value")
    increment = _number(step, "step")
    if increment <= 0:
        raise ValidationError("step must be positive", context={"step": step})
    quotient = _finite_result(number / increment, "step quotient")
    return _finite_result(math.floor(quotient) * increment, "rounded value")


def estimate_bar_ms(candles: Sequence[Mapping[str, Any]]) -> float:
    if isinstance(candles, (str, bytes)) or not isinstance(candles, Sequence):
        raise ValidationError("candles must be a sequence")
    deltas: list[float] = []
    previous: float | None = None
    for index, candle in enumerate(candles[:500]):
        if not isinstance(candle, Mapping):
            raise ValidationError("candle must be a mapping", context={"index": index})
        current = _number(candle.get("time"), "time")
        if previous is not None and current > previous:
            deltas.append(_finite_result(current - previous, "bar delta"))
        previous = current
    if not deltas:
        return 5 * 60 * 1_000
    deltas.sort()
    middle = len(deltas) // 2
    median = deltas[middle] if len(deltas) % 2 else (deltas[middle - 1] + deltas[middle]) / 2
    return max(60_000, min(median, 60 * 60_000))


def day_key_utc(time_ms: float) -> str:
    value = _number(time_ms, "time_ms")
    try:
        return datetime.fromtimestamp(value / 1_000, UTC).strftime("%Y-%m-%d")
    except (ValueError, OverflowError, OSError) as error:
        raise ValidationError(
            "time_ms must be Unix milliseconds", context={"time_ms": time_ms}
        ) from error


def day_key_et(time_ms: float) -> str:
    # Match the immutable JS oracle: it anchors the ET wall clock to the input's
    # UTC calendar date, so the resulting key is always that UTC date.
    return day_key_utc(time_ms)
