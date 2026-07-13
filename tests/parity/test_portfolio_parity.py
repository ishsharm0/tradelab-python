"""Generated JavaScript fixture parity for shared-capital portfolios."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, cast

import pytest

from tradelab.engine.portfolio import backtest_portfolio


def _camel_to_snake(value: str) -> str:
    if value == "finalTP_R":
        return "final_tp_r"
    value = value.replace("PnL", "Pnl")
    return "".join((f"_{char.lower()}" if char.isupper() else char) for char in value)


def _signal(spec: Mapping[str, Any]) -> Callable[[dict[str, object]], object | None]:
    assert spec["kind"] == "index-equals"
    return lambda context: spec["value"] if context["index"] == spec["index"] else None


def _assert_approx(actual: Any, expected: Any) -> None:
    if isinstance(expected, dict):
        assert isinstance(actual, Mapping)
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


def test_portfolio_matches_generated_javascript_fixture(
    load_fixture: Callable[[str], object],
) -> None:
    fixture = cast(dict[str, Any], load_fixture("portfolio.json"))
    raw_options = fixture["input"]["options"]
    systems: list[dict[str, Any]] = []
    for raw_system in raw_options["systems"]:
        system = {
            _camel_to_snake(key): value for key, value in raw_system.items() if key != "signal"
        }
        system["signal"] = _signal(raw_system["signal"])
        systems.append(system)
    options = {
        _camel_to_snake(key): value for key, value in raw_options.items() if key != "systems"
    }

    actual = backtest_portfolio(systems=systems, **options)

    _assert_approx(actual, fixture["output"])
