"""Generated JavaScript fixture parity for deterministic bar execution."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import pytest

from tradelab.engine.backtest import backtest
from tradelab.engine.financing import financing_cost, funding_events


def _camel_to_snake(value: str) -> str:
    value = value.replace("PnL", "Pnl")
    return "".join((f"_{char.lower()}" if char.isupper() else char) for char in value)


def _signal(spec: dict[str, Any]) -> Callable[[dict[str, object]], dict[str, object] | None]:
    assert spec["kind"] == "index-equals"
    return lambda context: spec["value"] if context["index"] == spec["index"] else None


def _assert_approx(actual: Any, expected: Any) -> None:
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert actual.keys() == expected.keys()
        for key, value in expected.items():
            _assert_approx(actual[key], value)
    elif isinstance(expected, list):
        assert isinstance(actual, list)
        assert len(actual) == len(expected)
        for value, expected_value in zip(actual, expected, strict=True):
            _assert_approx(value, expected_value)
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected, abs=1e-12, rel=1e-12)
    else:
        assert actual == expected


def test_backtest_matches_generated_javascript_fixture(
    load_fixture: Callable[[str], object],
) -> None:
    fixture = cast(dict[str, Any], load_fixture("backtest.json"))
    options = {_camel_to_snake(key): value for key, value in fixture["input"]["options"].items()}
    actual = backtest(**options, signal=_signal(fixture["input"]["signal"]))

    _assert_approx(actual, fixture["output"])


def test_financing_matches_generated_javascript_fixture(
    load_fixture: Callable[[str], object],
) -> None:
    fixture = cast(dict[str, Any], load_fixture("financing.json"))
    inputs, expected = fixture["input"], fixture["output"]
    funding = inputs["fundingEvents"]

    assert funding_events(**funding) == expected["fundingEvents"]
    assert financing_cost(**inputs["long"]) == pytest.approx(expected["long"])
    assert financing_cost(**inputs["short"]) == pytest.approx(expected["short"])
