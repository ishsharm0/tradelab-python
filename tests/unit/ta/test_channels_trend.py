"""Tests ported from JavaScript channel and trend modules."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from tradelab import ValidationError
from tradelab.ta.channels import bollinger, donchian, keltner
from tradelab.ta.trend import supertrend, vwap


def test_bollinger_middle_is_sma_and_width_is_symmetric() -> None:
    closes = [100 + (1 if index % 2 == 0 else -1) for index in range(30)]
    result = bollinger(closes, 20, 2)

    assert len(result["middle"]) == len(closes)
    index = 25
    upper, middle, lower = (
        result["upper"][index],
        result["middle"][index],
        result["lower"][index],
    )
    assert upper is not None and middle is not None and lower is not None
    assert upper > middle > lower
    assert upper - middle == pytest.approx(middle - lower)


def test_bollinger_accumulates_mean_and_deviation_left_to_right_like_javascript() -> None:
    mean_result = bollinger([1e16, 1, -1e16], 3)
    deviation_result = bollinger([1e16, -1e8, 1e4], 3)

    assert mean_result["middle"][-1] == 0.0
    assert deviation_result["upper"][-1] == 12_761_423_762_959_704.0
    assert deviation_result["lower"][-1] == -6_094_757_162_953_036.0


def test_donchian_tracks_rolling_high_and_low() -> None:
    bars = [{"high": 100 + index, "low": 90 + index, "close": 95 + index} for index in range(25)]

    result = donchian(bars, 20)

    assert result["upper"][24] == 124
    assert result["lower"][24] == 95
    assert result["middle"][24] == pytest.approx(109.5)


def test_keltner_is_centered_on_ema_with_atr_scaled_width() -> None:
    bars = [
        {"high": 101 + index * 0.1, "low": 99 + index * 0.1, "close": 100 + index * 0.1}
        for index in range(40)
    ]
    result = keltner(bars, 20, 14, 2)

    upper, middle, lower = result["upper"][39], result["middle"][39], result["lower"][39]
    assert upper is not None and middle is not None and lower is not None
    assert upper > middle > lower


def test_supertrend_marks_rising_series_as_an_uptrend() -> None:
    bars = [{"high": 101 + index, "low": 99 + index, "close": 100 + index} for index in range(40)]
    result = supertrend(bars, 10, 3)

    assert result["direction"][39] == 1
    line = result["line"][39]
    assert line is not None
    assert line < bars[39]["close"]


def test_vwap_resets_each_utc_day_and_falls_back_without_volume() -> None:
    day_one = int(datetime(2025, 1, 2, 14, 30, tzinfo=UTC).timestamp() * 1_000)
    day_two = int(datetime(2025, 1, 3, 14, 30, tzinfo=UTC).timestamp() * 1_000)
    bars = [
        {"time": day_one, "high": 102, "low": 98, "close": 100, "volume": 10},
        {"time": day_one + 60_000, "high": 104, "low": 100, "close": 103, "volume": 30},
        {"time": day_two, "high": 50, "low": 48, "close": 49, "volume": 0},
    ]

    result = vwap(bars)

    assert result[1] == pytest.approx(101.75)
    assert result[2] == pytest.approx(49)


@pytest.mark.parametrize("period", [0, -1])
def test_invalid_channel_period_raises_validation_error(period: int) -> None:
    with pytest.raises(ValidationError):
        bollinger([1, 2, 3], period)


def test_donchian_validates_short_malformed_bars_before_warmup() -> None:
    with pytest.raises(ValidationError):
        donchian([{"high": 1}], 20)


def test_supertrend_validates_short_bars_missing_close_before_warmup() -> None:
    with pytest.raises(ValidationError):
        supertrend([{"high": 1, "low": 0}], 10)
