"""Numerical parity checks against the committed JavaScript technical-analysis fixture."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import pytest

from tradelab.ta import (
    atr,
    bollinger,
    detect_fvg,
    donchian,
    ema,
    keltner,
    last_swing,
    macd,
    rsi,
    stochastic,
    structure_state,
    supertrend,
    swing_high,
    swing_low,
    vwap,
)


def _assert_nested_approx(actual: Any, expected: Any) -> None:
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert actual.keys() == expected.keys()
        for key, value in expected.items():
            _assert_nested_approx(actual[key], value)
    elif isinstance(expected, list):
        assert isinstance(actual, list)
        assert len(actual) == len(expected)
        for value, expected_value in zip(actual, expected, strict=True):
            _assert_nested_approx(value, expected_value)
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)
    else:
        assert actual == expected


def test_technical_analysis_matches_committed_javascript_fixture(
    load_fixture: Callable[[str], object],
) -> None:
    fixture = cast(dict[str, Any], load_fixture("ta.json"))
    inputs = fixture["input"]
    calls = inputs["calls"]
    candles = inputs["candles"]
    closes = inputs["closes"]

    actual = {
        "ema": ema(closes, **calls["ema"]),
        "atr": atr(candles, **calls["atr"]),
        "rsi": rsi(closes, **calls["rsi"]),
        "macd": macd(
            closes,
            fast=calls["macd"]["fast"],
            slow=calls["macd"]["slow"],
            signal_period=calls["macd"]["signalPeriod"],
        ),
        "stochastic": stochastic(
            candles,
            k_period=calls["stochastic"]["kPeriod"],
            d_period=calls["stochastic"]["dPeriod"],
        ),
        "bollinger": bollinger(closes, **calls["bollinger"]),
        "donchian": donchian(candles, **calls["donchian"]),
        "keltner": keltner(
            candles,
            ema_period=calls["keltner"]["emaPeriod"],
            atr_period=calls["keltner"]["atrPeriod"],
            mult=calls["keltner"]["mult"],
        ),
        "supertrend": supertrend(candles, **calls["supertrend"]),
        "vwap": vwap(candles),
        "swingHighAt9": swing_high(candles, **calls["swingHighAt9"]),
        "swingLowAt8": swing_low(candles, **calls["swingLowAt8"]),
        "fvgAt3": detect_fvg(candles, **calls["fvgAt3"]),
        "lastSwingAt12": last_swing(candles, **calls["lastSwingAt12"]),
        "structureAt12": structure_state(candles, **calls["structureAt12"]),
    }

    _assert_nested_approx(actual, fixture["output"])
