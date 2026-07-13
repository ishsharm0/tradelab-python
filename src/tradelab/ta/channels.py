"""Volatility and price-range channel indicators."""

from __future__ import annotations

import math
from collections.abc import Sequence

from tradelab.utils.indicators import CandleInput, _number, _period, atr, candle_value, ema


def bollinger(
    closes: Sequence[float | int], period: int = 20, mult: float = 2
) -> dict[str, list[float | None]]:
    """Return SMA-centered Bollinger bands using population standard deviation."""
    period = _period(period)
    multiplier = _number(mult, field="mult")
    output: dict[str, list[float | None]] = {
        key: [None] * len(closes) for key in ("middle", "upper", "lower")
    }
    values = [_number(close, field="close") for close in closes]
    for index in range(period - 1, len(values)):
        window = values[index - period + 1 : index + 1]
        middle = sum(window) / period
        deviation = math.sqrt(sum((value - middle) ** 2 for value in window) / period)
        output["middle"][index] = middle
        output["upper"][index] = middle + multiplier * deviation
        output["lower"][index] = middle - multiplier * deviation
    return output


def donchian(bars: Sequence[CandleInput], period: int = 20) -> dict[str, list[float | None]]:
    """Return rolling high, low, and midpoint Donchian channels."""
    period = _period(period)
    output: dict[str, list[float | None]] = {
        key: [None] * len(bars) for key in ("upper", "lower", "middle")
    }
    for index in range(period - 1, len(bars)):
        window = bars[index - period + 1 : index + 1]
        upper = max(candle_value(bar, "high") for bar in window)
        lower = min(candle_value(bar, "low") for bar in window)
        output["upper"][index], output["lower"][index] = upper, lower
        output["middle"][index] = (upper + lower) / 2
    return output


def keltner(
    bars: Sequence[CandleInput], ema_period: int = 20, atr_period: int = 14, mult: float = 2
) -> dict[str, list[float | None]]:
    """Return EMA-centered Keltner channels whose width is a multiple of ATR."""
    ema_period = _period(ema_period, name="ema_period")
    atr_period = _period(atr_period, name="atr_period")
    middle_values = ema([candle_value(bar, "close") for bar in bars], ema_period)
    ranges = atr(bars, atr_period)
    output: dict[str, list[float | None]] = {
        key: [None] * len(bars) for key in ("upper", "lower", "middle")
    }
    multiplier = _number(mult, field="mult")
    for index, range_ in enumerate(ranges):
        if range_ is not None:
            middle = middle_values[index]
            output["middle"][index] = middle
            output["upper"][index] = middle + multiplier * range_
            output["lower"][index] = middle - multiplier * range_
    return output
