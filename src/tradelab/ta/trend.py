"""Trend-following technical indicators."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

from tradelab.errors import ValidationError
from tradelab.utils.indicators import (
    CandleInput,
    _number,
    _period,
    atr,
    candle_time_ms,
    candle_value,
)


def supertrend(
    bars: Sequence[CandleInput], period: int = 10, mult: float = 3
) -> dict[str, list[float | None] | list[int | None]]:
    """Return Supertrend support/resistance line and trend direction."""
    period = _period(period)
    ranges = atr(bars, period)
    line: list[float | None] = [None] * len(bars)
    direction: list[int | None] = [None] * len(bars)
    previous_upper = float("inf")
    previous_lower = float("-inf")
    previous_direction = 1
    multiplier = _number(mult, field="mult")
    for index, range_ in enumerate(ranges):
        if range_ is None:
            continue
        high, low, close = (
            candle_value(bars[index], "high"),
            candle_value(bars[index], "low"),
            candle_value(bars[index], "close"),
        )
        midpoint = (high + low) / 2
        basic_upper, basic_lower = midpoint + multiplier * range_, midpoint - multiplier * range_
        previous_close = candle_value(bars[index - 1], "close") if index else close
        upper = (
            basic_upper
            if basic_upper < previous_upper or previous_close > previous_upper
            else previous_upper
        )
        lower = (
            basic_lower
            if basic_lower > previous_lower or previous_close < previous_lower
            else previous_lower
        )
        current_direction = previous_direction
        if previous_direction == 1 and close < lower:
            current_direction = -1
        elif previous_direction == -1 and close > upper:
            current_direction = 1
        line[index] = lower if current_direction == 1 else upper
        direction[index] = current_direction
        previous_upper, previous_lower, previous_direction = upper, lower, current_direction
    return {"line": line, "direction": direction}


def vwap(bars: Sequence[CandleInput]) -> list[float]:
    """Return UTC-session VWAP, falling back to an unweighted typical-price mean."""
    output: list[float] = []
    current_day: tuple[int, int, int] | None = None
    cumulative_pv = cumulative_volume = cumulative_typical_price = 0.0
    count = 0
    for bar in bars:
        day = datetime.fromtimestamp(candle_time_ms(bar) / 1_000, tz=UTC).date()
        day_key = (day.year, day.month, day.day)
        if day_key != current_day:
            current_day = day_key
            cumulative_pv = cumulative_volume = cumulative_typical_price = 0.0
            count = 0
        typical_price = (
            candle_value(bar, "high") + candle_value(bar, "low") + candle_value(bar, "close")
        ) / 3
        try:
            volume = candle_value(bar, "volume")
        except ValidationError:
            volume = 0.0
        cumulative_pv += typical_price * volume
        cumulative_volume += volume
        cumulative_typical_price += typical_price
        count += 1
        output.append(
            cumulative_pv / cumulative_volume
            if cumulative_volume > 0
            else cumulative_typical_price / count
        )
    return output
