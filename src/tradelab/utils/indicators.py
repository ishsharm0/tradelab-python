"""Core technical-analysis primitives shared by indicators and strategies."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import TypeAlias, cast

from tradelab.errors import ValidationError
from tradelab.models import Candle

CandleInput: TypeAlias = Candle | Mapping[str, object]


def _period(value: int, *, name: str = "period") -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValidationError(f"{name} must be a positive integer", context={name: value})
    return value


def _number(value: object, *, field: str) -> float:
    if isinstance(value, bool):
        raise ValidationError(f"{field} must be numeric", context={field: value})
    try:
        result = float(cast(str | float | int, value))
    except (TypeError, ValueError) as error:
        raise ValidationError(f"{field} must be numeric", context={field: value}) from error
    if not math.isfinite(result):
        raise ValidationError(f"{field} must be finite", context={field: value})
    return result


def _sum_left_to_right(values: Iterable[float]) -> float:
    """Accumulate floats in JavaScript Number's explicit loop order."""
    total = 0.0
    for value in values:
        total += value
    return total


def candle_value(candle: CandleInput, field: str) -> float:
    """Return one validated numeric candle field from a model or mapping."""
    if isinstance(candle, Candle):
        return _number(getattr(candle, field), field=field)
    if not isinstance(candle, Mapping) or field not in candle:
        raise ValidationError(f"candle requires {field}", context={"field": field})
    return _number(candle[field], field=field)


def candle_time_ms(candle: CandleInput) -> int:
    """Return a candle timestamp as Unix milliseconds."""
    if isinstance(candle, Candle):
        return candle.time_ms
    if not isinstance(candle, Mapping) or "time" not in candle:
        raise ValidationError("candle requires time", context={"field": "time"})
    value = candle["time"]
    if isinstance(value, bool):
        raise ValidationError("time must be Unix milliseconds", context={"time": value})
    try:
        return int(cast(str | float | int, value))
    except (TypeError, ValueError, OverflowError) as error:
        raise ValidationError("time must be Unix milliseconds", context={"time": value}) from error


def ema(values: Sequence[float | int], period: int = 14) -> list[float]:
    """Return an EMA with JavaScript-compatible warmup and NaN carry-forward behavior."""
    lookback = _period(period)
    if not values:
        return []

    output: list[float] = []
    warmup_sum = 0.0
    smoothing = 2 / (lookback + 1)
    for index, raw_value in enumerate(values):
        if isinstance(raw_value, bool):
            raise ValidationError("EMA values must be numeric", context={"value": raw_value})
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as error:
            raise ValidationError(
                "EMA values must be numeric", context={"value": raw_value}
            ) from error
        if not math.isfinite(value):
            output.append(0.0 if index == 0 else output[-1])
        elif index < lookback:
            warmup_sum += value
            output.append(warmup_sum / lookback if index == lookback - 1 else value)
        else:
            output.append(value * smoothing + output[-1] * (1 - smoothing))
    return output


def atr(bars: Sequence[CandleInput], period: int = 14) -> list[float | None]:
    """Return Wilder's average true range, aligned to the source bars."""
    period = _period(period)
    if not bars:
        return []
    values = [
        (candle_value(bar, "high"), candle_value(bar, "low"), candle_value(bar, "close"))
        for bar in bars
    ]
    true_ranges: list[float] = []
    for index, (high, low, _close) in enumerate(values):
        if index == 0:
            true_ranges.append(high - low)
        else:
            previous_close = values[index - 1][2]
            true_ranges.append(
                max(high - low, abs(high - previous_close), abs(low - previous_close))
            )

    output: list[float | None] = [None] * len(true_ranges)
    if len(true_ranges) < period:
        return output
    previous_atr = _sum_left_to_right(true_ranges[:period]) / period
    output[period - 1] = previous_atr
    for index in range(period, len(true_ranges)):
        previous_atr = (previous_atr * (period - 1) + true_ranges[index]) / period
        output[index] = previous_atr
    return output


def swing_high(bars: Sequence[CandleInput], index: int, left: int = 2, right: int = 2) -> bool:
    """Whether a bar has a strictly higher high than its surrounding window."""
    left = _period(left, name="left")
    right = _period(right, name="right")
    if index < left or index + right >= len(bars):
        return False
    high = candle_value(bars[index], "high")
    return all(
        cursor == index or candle_value(bars[cursor], "high") < high
        for cursor in range(index - left, index + right + 1)
    )


def swing_low(bars: Sequence[CandleInput], index: int, left: int = 2, right: int = 2) -> bool:
    """Whether a bar has a strictly lower low than its surrounding window."""
    left = _period(left, name="left")
    right = _period(right, name="right")
    if index < left or index + right >= len(bars):
        return False
    low = candle_value(bars[index], "low")
    return all(
        cursor == index or candle_value(bars[cursor], "low") > low
        for cursor in range(index - left, index + right + 1)
    )


def detect_fvg(bars: Sequence[CandleInput], index: int) -> dict[str, float | str] | None:
    """Detect the three-bar fair-value gap ending at ``index``."""
    if index < 2 or index >= len(bars):
        return None
    first, third = bars[index - 2], bars[index]
    first_high, first_low = candle_value(first, "high"), candle_value(first, "low")
    third_high, third_low = candle_value(third, "high"), candle_value(third, "low")
    if first_high < third_low:
        return {
            "type": "bull",
            "top": first_high,
            "bottom": third_low,
            "mid": (first_high + third_low) / 2,
        }
    if first_low > third_high:
        return {
            "type": "bear",
            "top": third_high,
            "bottom": first_low,
            "mid": (third_high + first_low) / 2,
        }
    return None


def last_swing(
    bars: Sequence[CandleInput], index: int, direction: str
) -> dict[str, float | int] | None:
    """Return the preceding opposite swing for an up or down trend direction."""
    if direction not in {"up", "down"}:
        raise ValidationError("direction must be up or down", context={"direction": direction})
    for cursor in range(min(index - 1, len(bars) - 1), -1, -1):
        if direction == "up" and swing_low(bars, cursor):
            return {"idx": cursor, "price": candle_value(bars[cursor], "low")}
        if direction == "down" and swing_high(bars, cursor):
            return {"idx": cursor, "price": candle_value(bars[cursor], "high")}
    return None


def structure_state(
    bars: Sequence[CandleInput], index: int
) -> dict[str, dict[str, float | int] | None]:
    """Return latest structural swing low and high prior to an index."""
    return {
        "lastLow": last_swing(bars, index, "up"),
        "lastHigh": last_swing(bars, index, "down"),
    }


def bps_of(price: float, bps: float) -> float:
    """Return the price movement represented by a basis-point amount."""
    return _number(price, field="price") * (_number(bps, field="bps") / 10_000)


def pct(a: float, b: float) -> float:
    """Return fractional change from ``b`` to ``a``."""
    denominator = _number(b, field="b")
    if denominator == 0:
        raise ValidationError("b must not be zero", context={"b": b})
    return (_number(a, field="a") - denominator) / denominator
