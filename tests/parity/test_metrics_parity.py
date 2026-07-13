"""Parity checks for metrics fixtures generated from the JavaScript implementation."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, cast

import pytest

from tradelab.metrics import BIG_NUMBER, build_metrics, clamp_finite, periods_per_year


def _snake(value: str) -> str:
    # JavaScript's PnL acronym becomes the established Python spelling ``pnl``.
    value = value.replace("PnL", "Pnl")
    result: list[str] = []
    for char in value:
        if char.isupper():
            result.extend(("_", char.lower()))
        else:
            result.append(char)
    return "".join(result)


def _translate_expected(value: Any) -> Any:
    if isinstance(value, dict):
        return {_snake(key): _translate_expected(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_translate_expected(item) for item in value]
    return value


def _assert_nested_approx(actual: Any, expected: Any) -> None:
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert actual.keys() == expected.keys()
        for key, value in expected.items():
            _assert_nested_approx(actual[key], value)
    elif isinstance(expected, list):
        assert isinstance(actual, list)
        assert len(actual) == len(expected)
        for item, expected_item in zip(actual, expected, strict=True):
            _assert_nested_approx(item, expected_item)
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)
    else:
        assert actual == expected


def test_metrics_match_committed_javascript_fixture(
    load_fixture: Callable[[str], object],
) -> None:
    fixture = cast(dict[str, Any], load_fixture("metrics.json"))
    inputs = fixture["input"]
    expected = fixture["output"]
    build_input = inputs["buildMetrics"]

    actual = {
        "big_number": BIG_NUMBER,
        "clamp_finite": [
            clamp_finite(math.inf, 0),
            clamp_finite(-math.inf, 0),
            clamp_finite(math.nan, 7),
        ],
        "periods_per_year": {
            "daily": periods_per_year("1d", None),
            "minute": periods_per_year("1m", None),
            "estimated": periods_per_year("custom", 3_600_000),
        },
        "build_metrics": build_metrics(
            closed=build_input["closed"],
            equity_start=build_input["equityStart"],
            equity_final=build_input["equityFinal"],
            candles=build_input["candles"],
            est_bar_ms=build_input["estBarMs"],
            eq_series=build_input.get("eqSeries"),
            interval=build_input.get("interval"),
            benchmark_returns=build_input.get("benchmarkReturns"),
        ),
    }

    _assert_nested_approx(actual, _translate_expected(expected))
