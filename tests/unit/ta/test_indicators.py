"""Tests ported from the JavaScript indicator primitives."""

from __future__ import annotations

import math

import pytest

from tradelab import Candle, ValidationError
from tradelab.utils.indicators import (
    atr,
    bps_of,
    detect_fvg,
    ema,
    last_swing,
    pct,
    structure_state,
    swing_high,
    swing_low,
)


def _bars(count: int = 30) -> list[dict[str, float]]:
    return [
        {"high": 101.0 + index, "low": 99.0 + index, "close": 100.0 + index}
        for index in range(count)
    ]


def test_ema_seeds_with_the_simple_average_then_smooths() -> None:
    assert ema([1, 2, 3, 4, 5], 3) == pytest.approx([1, 2, 2, 3, 4])


def test_ema_preserves_original_nonfinite_carry_forward_behavior() -> None:
    values = ema([1, math.nan, 3], 2)

    assert values[0] == 1
    assert values[1] == 1
    assert values[2] == pytest.approx(7 / 3)


def test_atr_has_none_warmup_and_uses_wilder_smoothing() -> None:
    bars = _bars(4)

    assert atr(bars, 3) == [None, None, 2.0, 2.0]


def test_swing_fvg_and_structure_helpers_preserve_object_shapes() -> None:
    bars = [
        {"high": 5, "low": 1, "close": 3},
        {"high": 7, "low": 2, "close": 6},
        {"high": 6, "low": 0, "close": 4},
        {"high": 8, "low": 4, "close": 7},
        {"high": 5, "low": 2, "close": 3},
        {"high": 4, "low": 1, "close": 2},
    ]

    assert swing_high(bars, 3, left=1, right=1) is True
    assert swing_low(bars, 2, left=1, right=1) is True
    assert detect_fvg(
        [
            {"high": 10, "low": 8, "close": 9},
            {"high": 11, "low": 9, "close": 10},
            {"high": 14, "low": 12, "close": 13},
        ],
        2,
    ) == {"type": "bull", "top": 10.0, "bottom": 12.0, "mid": 11.0}
    assert last_swing(bars, 5, "down") == {"idx": 3, "price": 8.0}
    assert structure_state(bars, 5) == {
        "lastLow": {"idx": 2, "price": 0.0},
        "lastHigh": {"idx": 3, "price": 8.0},
    }


def test_indicators_accept_candle_objects() -> None:
    bars = [Candle(1_700_000_000_000 + index, 10, 12, 9, 11, 100) for index in range(3)]

    assert atr(bars, 2) == [None, 3.0, 3.0]


@pytest.mark.parametrize("period", [0, -1, 1.5, True])
def test_invalid_indicator_periods_raise_validation_error(period: object) -> None:
    with pytest.raises(ValidationError):
        ema([1, 2, 3], period)  # type: ignore[arg-type]


def test_invalid_indicator_input_raises_validation_error() -> None:
    with pytest.raises(ValidationError):
        atr([{"high": 2}], 2)


def test_basis_points_and_percent_change_match_javascript_primitives() -> None:
    assert bps_of(100, 25) == pytest.approx(0.25)
    assert pct(110, 100) == pytest.approx(0.1)
