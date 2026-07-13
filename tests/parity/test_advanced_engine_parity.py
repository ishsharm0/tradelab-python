"""Generated JavaScript parity for advanced deterministic engines."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, cast

import pytest

from tradelab.engine.backtest_ticks import backtest_ticks
from tradelab.engine.walk_forward import walk_forward_optimize


def _camel_to_snake(value: str) -> str:
    if value == "finalTP_R":
        return "final_tp_r"
    value = value.replace("PnL", "Pnl")
    return "".join((f"_{char.lower()}" if char.isupper() else char) for char in value)


def _signal(spec: dict[str, Any]) -> Callable[[dict[str, object]], object]:
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


def test_ticks_match_generated_javascript_fixture(
    load_fixture: Callable[[str], object],
) -> None:
    fixture = cast(dict[str, Any], load_fixture("ticks.json"))
    options = {_camel_to_snake(key): value for key, value in fixture["input"]["options"].items()}
    actual = backtest_ticks(**options, signal=_signal(fixture["input"]["signal"]))

    _assert_approx(actual, fixture["output"])


def _walk_signal_factory(spec: dict[str, Any]) -> Callable[[dict[str, Any]], object]:
    assert spec["kind"] == "index-equals-from-bar"

    def factory(params: dict[str, Any]) -> Callable[[dict[str, Any]], object]:
        def signal(context: dict[str, Any]) -> object:
            if context["index"] != spec["index"]:
                return None
            bar = context["bar"]
            return {
                "side": "long",
                "entry": bar["close"],
                "stop": bar["close"] - 1,
                "takeProfit": bar["close"] + params["target"],
                "qty": 1,
            }

        return signal

    return factory


def test_walk_forward_matches_generated_javascript_fixture(
    load_fixture: Callable[[str], object],
) -> None:
    fixture = cast(dict[str, Any], load_fixture("walkForward.json"))
    options = {_camel_to_snake(key): value for key, value in fixture["input"]["options"].items()}
    actual = walk_forward_optimize(
        **options,
        signal_factory=_walk_signal_factory(fixture["input"]["signalFactory"]),
    )

    _assert_approx(actual, fixture["output"])
