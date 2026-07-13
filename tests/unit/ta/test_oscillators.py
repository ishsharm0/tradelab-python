"""Tests ported from the JavaScript oscillator module."""

from __future__ import annotations

import pytest

from tradelab import ValidationError
from tradelab.ta.oscillators import macd, rsi, stochastic


def test_rsi_of_a_strictly_rising_series_approaches_100() -> None:
    values = rsi([100 + index for index in range(30)], 14)

    assert len(values) == 30
    assert values[5] is None
    assert values[29] is not None and values[29] > 99


def test_rsi_of_a_strictly_falling_series_approaches_zero() -> None:
    values = rsi([100 - index for index in range(30)], 14)

    assert values[29] is not None and values[29] < 1


def test_macd_returns_aligned_macd_signal_and_histogram_arrays() -> None:
    values = [100 + __import__("math").sin(index / 3) * 5 for index in range(60)]
    result = macd(values, 12, 26, 9)

    assert set(result) == {"macd", "signal", "histogram"}
    assert all(len(series) == len(values) for series in result.values())
    assert result["macd"][59] is not None


def test_stochastic_k_stays_in_range_and_d_has_warmup() -> None:
    bars = [
        {
            "high": 101 + __import__("math").sin(index),
            "low": 99 + __import__("math").sin(index),
            "close": 100 + __import__("math").sin(index) * 0.5,
        }
        for index in range(40)
    ]
    result = stochastic(bars, 14, 3)

    assert result["k"][39] is not None and 0 <= result["k"][39] <= 100
    assert result["d"][:15] == [None] * 15


@pytest.mark.parametrize("periods", [(0, 3), (14, 0), (0, 0)])
def test_invalid_oscillator_periods_raise_validation_error(periods: tuple[int, int]) -> None:
    with pytest.raises(ValidationError):
        stochastic([{"high": 2, "low": 1, "close": 1.5}], *periods)
