"""Momentum oscillator indicators."""

from __future__ import annotations

from collections.abc import Sequence

from tradelab.utils.indicators import (
    CandleInput,
    _number,
    _period,
    _sum_left_to_right,
    candle_value,
    ema,
)


def rsi(closes: Sequence[float | int], period: int = 14) -> list[float | None]:
    """Return Wilder's RSI with ``None`` values before the warmup completes."""
    period = _period(period)
    values = [_number(close, field="close") for close in closes]
    output: list[float | None] = [None] * len(values)
    if len(values) <= period:
        return output
    gain_sum = 0.0
    loss_sum = 0.0
    for index in range(1, period + 1):
        change = values[index] - values[index - 1]
        if change >= 0:
            gain_sum += change
        else:
            loss_sum -= change
    average_gain, average_loss = gain_sum / period, loss_sum / period
    output[period] = 100.0 if average_loss == 0 else 100 - 100 / (1 + average_gain / average_loss)
    for index in range(period + 1, len(values)):
        change = values[index] - values[index - 1]
        average_gain = (average_gain * (period - 1) + max(change, 0)) / period
        average_loss = (average_loss * (period - 1) + max(-change, 0)) / period
        output[index] = (
            100.0 if average_loss == 0 else 100 - 100 / (1 + average_gain / average_loss)
        )
    return output


def macd(
    closes: Sequence[float | int], fast: int = 12, slow: int = 26, signal_period: int = 9
) -> dict[str, list[float]]:
    """Return aligned MACD, signal, and histogram series."""
    _period(fast, name="fast")
    _period(slow, name="slow")
    _period(signal_period, name="signal_period")
    fast_ema, slow_ema = ema(closes, fast), ema(closes, slow)
    macd_line = [
        fast_value - slow_value for fast_value, slow_value in zip(fast_ema, slow_ema, strict=True)
    ]
    signal_line = ema(macd_line, signal_period)
    return {
        "macd": macd_line,
        "signal": signal_line,
        "histogram": [value - signal_line[index] for index, value in enumerate(macd_line)],
    }


def stochastic(
    bars: Sequence[CandleInput], k_period: int = 14, d_period: int = 3
) -> dict[str, list[float | None]]:
    """Return stochastic %K and its simple-moving-average %D signal."""
    k_period = _period(k_period, name="k_period")
    d_period = _period(d_period, name="d_period")
    ohlc_values = [
        (candle_value(bar, "high"), candle_value(bar, "low"), candle_value(bar, "close"))
        for bar in bars
    ]
    k: list[float | None] = [None] * len(ohlc_values)
    for index in range(k_period - 1, len(ohlc_values)):
        window = ohlc_values[index - k_period + 1 : index + 1]
        high = max(high for high, _, _ in window)
        low = min(low for _, low, _ in window)
        range_ = high - low
        k[index] = 0.0 if range_ == 0 else (ohlc_values[index][2] - low) / range_ * 100
    d: list[float | None] = [None] * len(ohlc_values)
    for index in range(k_period + d_period - 2, len(ohlc_values)):
        k_window = k[index - d_period + 1 : index + 1]
        if any(value is None for value in k_window):
            raise AssertionError("stochastic d window must contain initialized k values")
        d[index] = _sum_left_to_right(value for value in k_window if value is not None) / d_period
    return {"k": k, "d": d}
