"""Grid, optimization, and walk-forward contract tests."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from tradelab.engine.grid import grid
from tradelab.engine.optimize import optimize
from tradelab.engine.walk_forward import walk_forward_optimize
from tradelab.errors import ValidationError

START = 1_704_205_800_000


def _bars(count: int = 18) -> list[dict[str, float | int]]:
    prices = [100 + index + (1 if index % 2 else 0) for index in range(count)]
    return [
        {
            "time": START + index * 60_000,
            "open": price,
            "high": price + 2,
            "low": price - 2,
            "close": price,
            "volume": 100,
        }
        for index, price in enumerate(prices)
    ]


def _hold_factory(params: dict[str, Any]) -> Callable[[dict[str, object]], object]:
    hold = int(params.get("hold", 1))

    def signal(context: dict[str, object]) -> object:
        if context["index"] != 1:
            return None
        bar = context["bar"]
        assert isinstance(bar, dict)
        close = float(bar["close"])
        return {
            "side": "long",
            "entry": close,
            "stop": close - 5,
            "takeProfit": close + 100,
            "qty": 1,
            "_maxBarsInTrade": hold,
        }

    return signal


def _sometimes_failing_factory(
    params: dict[str, Any],
) -> Callable[[dict[str, object]], object]:
    if params.get("fail"):
        raise RuntimeError(f"bad params {params['name']}")
    return _hold_factory(params)


def test_grid_uses_javascript_numeric_object_key_order_and_stable_cartesian_order() -> None:
    spec = {"10": ["ten"], "2": ["a", "b"], "z": [1, 2], "01": "fixed"}

    assert grid(spec) == [
        {"2": "a", "10": "ten", "z": 1, "01": "fixed"},
        {"2": "a", "10": "ten", "z": 2, "01": "fixed"},
        {"2": "b", "10": "ten", "z": 1, "01": "fixed"},
        {"2": "b", "10": "ten", "z": 2, "01": "fixed"},
    ]
    assert grid({}) == [{}]


def test_grid_treats_only_lists_as_sweeps_and_rejects_non_mapping_specs() -> None:
    assert grid() == [{}]
    assert grid({"fast": [3, 5], "fixed": (8, 13)}) == [
        {"fast": 3, "fixed": (8, 13)},
        {"fast": 5, "fixed": (8, 13)},
    ]
    with pytest.raises(ValidationError, match="grid spec must be a mapping"):
        grid([("fast", [3, 5])])  # type: ignore[arg-type]


def test_optimize_preserves_input_slots_failures_and_stable_tie_order() -> None:
    parameter_sets = [
        {"name": "first", "hold": 1},
        {"name": "broken", "fail": True},
        {"name": "second", "hold": 1},
    ]
    output = optimize(
        candles=_bars(),
        signal_factory=_sometimes_failing_factory,
        parameter_sets=parameter_sets,
        score_by="totalPnL",
        concurrency=2,
        backtest_options={
            "warmupBars": 1,
            "flattenAtClose": False,
            "scaleOutAtR": 0,
            "slippageBps": 0,
        },
    )

    assert [row["params"] for row in output["results"]] == parameter_sets
    assert "bad params broken" in output["results"][1]["error"]
    assert [row["params"]["name"] for row in output["leaderboard"]] == [
        "first",
        "second",
    ]
    assert output["best"] is output["leaderboard"][0]
    assert set(output["results"][0]["metrics"]) == {
        "trades",
        "winRate",
        "profitFactor",
        "expectancy",
        "totalR",
        "avgR",
        "sharpe",
        "sharpeAnnualized",
        "maxDrawdown",
        "calmar",
        "returnPct",
        "totalPnL",
        "finalEquity",
    }


def test_optimize_empty_parameter_sets_and_validation_boundaries() -> None:
    assert optimize(candles=_bars(), signal_factory=_hold_factory, parameter_sets=[]) == {
        "results": [],
        "leaderboard": [],
        "best": None,
    }
    with pytest.raises(ValidationError, match="concurrency must be a positive integer"):
        optimize(
            candles=_bars(),
            signal_factory=_hold_factory,
            parameter_sets=[{"hold": 1}],
            concurrency=0,
        )
    with pytest.raises(ValidationError, match="parameter_sets must be a sequence"):
        optimize(
            candles=_bars(),
            signal_factory=_hold_factory,
            parameter_sets="bad",
        )
    with pytest.raises(ValidationError, match="signal_factory must be callable"):
        optimize(
            candles=_bars(),
            signal_factory=None,
            parameter_sets=[{"hold": 1}],
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"parameter_sets": [1]}, "each parameter set must be a mapping"),
        ({"candles": "bad"}, "candles must be a sequence"),
        ({"interval": 60}, "interval must be a string or None"),
        ({"backtest_options": []}, "backtest_options must be a mapping"),
        ({"score_by": 1}, "score_by must be a string"),
        ({"concurrency": True}, "concurrency must be a positive integer"),
        ({"concurrency": "many"}, "concurrency must be a positive integer"),
    ],
)
def test_optimize_rejects_malformed_options(overrides: dict[str, object], message: str) -> None:
    options: dict[str, object] = {
        "candles": _bars(),
        "signal_factory": _hold_factory,
        "parameter_sets": [{"hold": 1}],
    }
    options.update(overrides)

    with pytest.raises(ValidationError, match=message):
        optimize(**options)


def test_optimize_records_a_non_callable_factory_result_as_a_failed_row() -> None:
    output = optimize(
        candles=_bars(),
        signal_factory=lambda _params: None,
        parameter_sets=[{"hold": 1}],
    )

    assert "signal_factory must return a signal callable" in output["results"][0]["error"]
    assert output["leaderboard"] == []
    assert output["best"] is None


def test_process_pool_infrastructure_failure_keeps_the_result_slot() -> None:
    local_factory = lambda params: _hold_factory(params)  # noqa: E731

    output = optimize(
        candles=_bars(),
        signal_factory=local_factory,
        parameter_sets=[{"hold": 1}],
        concurrency=1,
        use_process_pool=True,
    )

    assert output["results"][0]["params"] == {"hold": 1}
    assert "worker failed for params" in output["results"][0]["error"]
    assert output["leaderboard"] == []


def test_opt_in_process_pool_matches_serial_results() -> None:
    options: dict[str, Any] = {
        "candles": _bars(),
        "signal_factory": _hold_factory,
        "parameter_sets": [{"hold": 1}, {"hold": 2}],
        "score_by": "totalPnL",
        "concurrency": 2,
        "backtest_options": {
            "warmupBars": 1,
            "flattenAtClose": False,
            "scaleOutAtR": 0,
            "slippageBps": 0,
        },
    }

    serial = optimize(**options)
    parallel = optimize(**options, use_process_pool=True)

    assert parallel == serial


@pytest.mark.parametrize("unsafe", [float("nan"), 10**10_000], ids=["nan", "huge-int"])
def test_optimization_rejects_nonportable_parameter_values(unsafe: object) -> None:
    with pytest.raises(ValidationError):
        optimize(
            candles=_bars(),
            signal_factory=_hold_factory,
            parameter_sets=[{"hold": 1, "unsafe": unsafe}],
        )


def test_optimization_rejects_cyclic_parameters() -> None:
    cyclic: dict[str, object] = {}
    cyclic["self"] = cyclic
    with pytest.raises(ValidationError, match="cyclic"):
        optimize(candles=_bars(), signal_factory=_hold_factory, parameter_sets=[cyclic])


def test_walk_forward_rolling_and_anchored_ranges_use_fresh_signals() -> None:
    calls: list[dict[str, Any]] = []

    def factory(params: dict[str, Any]) -> Callable[[dict[str, object]], object]:
        calls.append(params)
        return _hold_factory(params)

    rolling = walk_forward_optimize(
        candles=_bars(18),
        signal_factory=factory,
        parameter_sets=[{"hold": 1}, {"hold": 2}],
        train_bars=6,
        test_bars=4,
        step_bars=4,
        score_by="totalPnL",
        backtest_options={"warmupBars": 1, "flattenAtClose": False, "scaleOutAtR": 0},
    )
    anchored = walk_forward_optimize(
        candles=_bars(18),
        signal_factory=factory,
        parameter_sets=[{"hold": 1}, {"hold": 2}],
        train_bars=6,
        test_bars=4,
        step_bars=4,
        mode="anchored",
        score_by="totalPnL",
        backtest_options={"warmupBars": 1, "flattenAtClose": False, "scaleOutAtR": 0},
    )

    assert rolling["windows"][1]["train"]["start"] == _bars(18)[4]["time"]
    assert anchored["windows"][0]["train"]["start"] == START
    assert anchored["windows"][1]["train"]["start"] == START
    expected_factory_calls = sum(2 + 1 for _ in rolling["windows"] + anchored["windows"])
    assert len(calls) == expected_factory_calls


def test_walk_forward_preserves_javascript_stability_mutation_order_quirk() -> None:
    result = walk_forward_optimize(
        candles=_bars(18),
        signal_factory=_hold_factory,
        parameter_sets=[{"hold": 1}],
        train_bars=6,
        test_bars=4,
        step_bars=4,
        score_by="totalPnL",
        backtest_options={"warmupBars": 1, "flattenAtClose": False, "scaleOutAtR": 0},
    )

    assert [window["stabilityScore"] for window in result["windows"]] == [1, 0.5, 0]
    assert result["bestParamsSummary"]["adjacentRepeatRate"] == 1
    assert result["bestParams"].winners == [{"hold": 1}] * 3
    assert result["bestParams"].stability is result["bestParamsSummary"]


def test_walk_forward_stitches_equity_strictly_forward_and_rolls_equity() -> None:
    result = walk_forward_optimize(
        candles=_bars(18),
        signal_factory=_hold_factory,
        parameter_sets=[{"hold": 1}],
        train_bars=6,
        test_bars=6,
        step_bars=2,
        score_by="totalPnL",
        backtest_options={
            "equity": 10_000,
            "warmupBars": 1,
            "flattenAtClose": False,
            "scaleOutAtR": 0,
            "slippageBps": 0,
        },
    )

    times = [point["time"] for point in result["eqSeries"]]
    assert times == sorted(times)
    expected: list[dict[str, object]] = []
    for window in result["windows"]:
        source = window["result"]["eqSeries"]
        if not expected:
            expected.extend(source)
        else:
            last_time = expected[-1]["time"]
            expected.extend(point for point in source if point["time"] > last_time)
    assert result["eqSeries"] == expected
    starts = [window["testMetrics"]["startEquity"] for window in result["windows"]]
    finals = [window["testMetrics"]["finalEquity"] for window in result["windows"]]
    assert starts[1:] == finals[:-1]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"candles": []}, "non-empty candles"),
        ({"signal_factory": None}, "signal_factory callable"),
        ({"parameter_sets": []}, "requires parameter_sets"),
        ({"train_bars": 0}, "train_bars.*positive"),
        ({"test_bars": 1.5}, "positive integer"),
        ({"step_bars": -1}, "step_bars.*positive"),
        ({"mode": "expanding"}, "rolling or anchored"),
        ({"parameter_sets": [1]}, "each parameter set must be a mapping"),
        ({"score_by": 1}, "score_by must be a string"),
        ({"backtest_options": []}, "backtest_options must be a mapping"),
        ({"backtest_options": {"equity": float("inf")}}, "equity must be finite"),
    ],
)
def test_walk_forward_validation(overrides: dict[str, object], message: str) -> None:
    options: dict[str, object] = {
        "candles": _bars(18),
        "signal_factory": _hold_factory,
        "parameter_sets": [{"hold": 1}],
        "train_bars": 6,
        "test_bars": 4,
        "step_bars": 4,
    }
    options.update(overrides)
    with pytest.raises(ValidationError, match=message):
        walk_forward_optimize(**options)


def test_walk_forward_reports_actionable_zero_window_error() -> None:
    with pytest.raises(ValidationError, match=r"produced zero windows.*need at least 12"):
        walk_forward_optimize(
            candles=_bars(10),
            signal_factory=_hold_factory,
            parameter_sets=[{"hold": 1}],
            train_bars=8,
            test_bars=4,
            step_bars=2,
        )


def test_walk_forward_rejects_a_non_callable_factory_result() -> None:
    with pytest.raises(ValidationError, match="return a signal callable"):
        walk_forward_optimize(
            candles=_bars(10),
            signal_factory=lambda _params: None,
            parameter_sets=[{"hold": 1}],
            train_bars=6,
            test_bars=4,
        )


def test_single_walk_forward_window_has_full_local_stability() -> None:
    result = walk_forward_optimize(
        candles=_bars(10),
        signal_factory=_hold_factory,
        parameter_sets=[{"hold": 1}],
        train_bars=6,
        test_bars=4,
        score_by="totalPnL",
        backtest_options={"warmupBars": 1, "flattenAtClose": False, "scaleOutAtR": 0},
    )

    assert result["windows"][0]["stabilityScore"] == 1
    assert result["bestParamsSummary"]["adjacentRepeatRate"] == 0
