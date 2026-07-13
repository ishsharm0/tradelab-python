"""Pinned JavaScript parity tests for the quantitative research toolkit."""

from __future__ import annotations

from typing import Any, cast

import pytest

from tradelab.research import (
    combinatorial_purged_splits,
    deflated_sharpe,
    moments,
    monte_carlo,
    normal_cdf,
    normal_ppf,
    probability_of_backtest_overfitting,
)


def _assert_nested_approx(actual: Any, expected: Any) -> None:
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key, value in expected.items():
            _assert_nested_approx(actual[key], value)
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for actual_value, expected_value in zip(actual, expected, strict=True):
            _assert_nested_approx(actual_value, expected_value)
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected)
    else:
        assert actual == expected


def test_research_fixture_parity(load_fixture: Any) -> None:
    research_fixture = cast(dict[str, Any], load_fixture("research.json"))
    inputs = research_fixture["input"]
    expected = research_fixture["output"]

    stats_input = inputs["stats"]
    _assert_nested_approx(
        {
            "normalCdf": normal_cdf(stats_input["normalCdf"]),
            "normalPpf": normal_ppf(stats_input["normalPpf"]),
            "moments": moments(stats_input["moments"]).__dict__,
        },
        expected["stats"],
    )

    cpcv_input = inputs["cpcv"]
    assert combinatorial_purged_splits(
        n_observations=cpcv_input["nObservations"],
        n_groups=cpcv_input["nGroups"],
        n_test_groups=cpcv_input["nTestGroups"],
        embargo=cpcv_input["embargo"],
    ) == [
        {"train": row["train"], "test": row["test"], "test_groups": row["testGroups"]}
        for row in expected["cpcv"]
    ]

    dsl_input = inputs["deflatedSharpe"]
    assert deflated_sharpe(
        sharpe=dsl_input["sharpe"],
        sample_size=dsl_input["sampleSize"],
        num_trials=dsl_input["numTrials"],
        sharpe_std=dsl_input["sharpeStd"],
        skew=dsl_input["skew"],
        kurtosis=dsl_input["kurtosis"],
    ) == pytest.approx(expected["deflatedSharpe"])

    mc_input = inputs["monteCarlo"]
    _assert_nested_approx(
        monte_carlo(
            trade_pnls=mc_input["tradePnls"],
            equity_start=mc_input["equityStart"],
            iterations=mc_input["iterations"],
            block_size=mc_input["blockSize"],
            seed=mc_input["seed"],
        ),
        {
            "iterations": expected["monteCarlo"]["iterations"],
            "block_size": expected["monteCarlo"]["blockSize"],
            "final_equity": expected["monteCarlo"]["finalEquity"],
            "max_drawdown": expected["monteCarlo"]["maxDrawdown"],
            "path_bands": expected["monteCarlo"]["pathBands"],
            "prob_profit": expected["monteCarlo"]["probProfit"],
        },
    )

    pbo_input = inputs["pbo"]
    _assert_nested_approx(
        probability_of_backtest_overfitting(
            pbo_input["performanceMatrix"], groups=pbo_input["groups"]
        ),
        {
            "pbo": expected["pbo"]["pbo"],
            "combos": expected["pbo"]["combos"],
            "median_logit": expected["pbo"]["medianLogit"],
        },
    )
