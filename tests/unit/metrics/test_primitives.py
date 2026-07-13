"""Focused compatibility tests for public metric primitives."""

from __future__ import annotations

import json
import math
from importlib.util import find_spec

import pytest

from tradelab.errors import ValidationError
from tradelab.metrics import BIG_NUMBER, benchmark_stats, clamp_finite, periods_per_year


@pytest.mark.parametrize(
    ("interval", "expected"),
    [
        ("1m", 98_280),
        ("2m", 49_140),
        ("5m", 19_656),
        ("15m", 6_552),
        ("30m", 3_276),
        ("1h", 1_638),
        ("60m", 1_638),
        ("1d", 252),
        ("1wk", 52),
        ("1mo", 12),
    ],
)
def test_periods_per_year_maps_known_intervals(interval: str, expected: int) -> None:
    assert periods_per_year(interval, None) == expected


def test_periods_per_year_uses_positive_javascript_rounding_for_estimates() -> None:
    # 31_536_000_000 / 12_614_400_000 is exactly 2.5; JS Math.round gives 3.
    assert periods_per_year("unknown", 12_614_400_000) == 3
    assert periods_per_year("1d", 12_614_400_000) == 252
    assert periods_per_year(None, None) == 252


def test_periods_per_year_preserves_large_integral_javascript_quotients() -> None:
    estimate = 31_536_000_000 / 9_007_199_254_740_991
    assert periods_per_year("custom", estimate) == 9_007_199_254_740_991


@pytest.mark.parametrize(
    ("value", "fallback", "expected"),
    [
        (math.inf, 7.0, BIG_NUMBER),
        (-math.inf, 7.0, -BIG_NUMBER),
        (math.nan, 7.0, 7.0),
        (None, -2.0, -2.0),
        (True, -2.0, -2.0),
        (1_000_000_001.0, 0.0, 1_000_000_001.0),
    ],
)
def test_clamp_finite_keeps_only_finite_non_boolean_numbers(
    value: object, fallback: float, expected: float
) -> None:
    assert clamp_finite(value, fallback) == expected


def test_clamp_finite_never_leaks_conversion_overflow_or_nonfinite_fallback() -> None:
    assert clamp_finite(10**10_000, 7.0) == BIG_NUMBER
    assert clamp_finite(-(10**10_000), 7.0) == -BIG_NUMBER
    assert clamp_finite(math.nan, math.inf) is None
    json.dumps(clamp_finite(math.nan, math.inf), allow_nan=False)


def test_benchmark_stats_handles_singletons_and_constant_sources() -> None:
    assert benchmark_stats([0.1], [0.2]) == {
        "alpha": 0.1,
        "beta": 0.0,
        "correlation": 0.0,
        "information_ratio": 0.0,
        "tracking_error": 0.0,
    }
    constant = benchmark_stats([0.1, 0.2], [0.05, 0.05])
    assert constant["beta"] == 0.0
    assert constant["correlation"] == 0.0


@pytest.mark.parametrize("multiplier", [2.0, -2.0])
def test_benchmark_stats_preserves_perfect_positive_and_negative_correlation(
    multiplier: float,
) -> None:
    benchmark = [0.01, -0.02, 0.03, -0.01]
    strategy = [multiplier * value for value in benchmark]
    stats = benchmark_stats(strategy, benchmark)
    assert stats["beta"] == pytest.approx(multiplier)
    assert stats["alpha"] == pytest.approx(0.0)
    assert stats["correlation"] == pytest.approx(1.0 if multiplier > 0 else -1.0)


def test_benchmark_stats_uses_explicit_left_to_right_population_moments() -> None:
    values = [1e16, 1.0, -1e16]
    stats = benchmark_stats(values, values)
    assert stats["alpha"] == 0.0
    assert stats["beta"] == 1.0
    assert stats["correlation"] == 1.0

    population = benchmark_stats([0.0, 2.0], [0.0, 0.0])
    assert population["tracking_error"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("strategy", "benchmark", "expected_information_ratio", "expected_tracking_error"),
    [
        ([1e308, -1e308], [1e308, -1e308], 0.0, 0.0),
        ([1e308, 1e308], [-1e308, -1e308], None, None),
    ],
)
def test_benchmark_stats_matches_json_serialized_js_for_overflowed_moments(
    strategy: list[float],
    benchmark: list[float],
    expected_information_ratio: float | None,
    expected_tracking_error: float | None,
) -> None:
    stats = benchmark_stats(strategy, benchmark)
    assert stats == {
        "alpha": None,
        "beta": None,
        "correlation": None,
        "information_ratio": expected_information_ratio,
        "tracking_error": expected_tracking_error,
    }
    json.dumps(stats, allow_nan=False)


def test_periods_per_year_handles_binary64_overflow_like_javascript() -> None:
    assert periods_per_year("custom", 10**10_000) == 252
    assert periods_per_year("custom", 5e-324) == math.inf


def test_benchmark_stats_returns_null_block_for_empty_or_mismatched_inputs() -> None:
    expected = {
        "alpha": None,
        "beta": None,
        "correlation": None,
        "information_ratio": None,
        "tracking_error": None,
    }
    assert benchmark_stats([], []) == expected
    assert benchmark_stats([0.01], [0.01, 0.02]) == expected


@pytest.mark.parametrize("strategy, benchmark", [(True, [0.0]), ([0.0], True)])
def test_benchmark_stats_returns_null_block_for_wrong_type_inputs(
    strategy: object, benchmark: object
) -> None:
    assert benchmark_stats(strategy, benchmark) == {
        "alpha": None,
        "beta": None,
        "correlation": None,
        "information_ratio": None,
        "tracking_error": None,
    }


@pytest.mark.parametrize(
    "strategy, benchmark",
    [([math.nan], [0.0]), ([0.0], [math.inf])],
)
def test_benchmark_stats_rejects_nonfinite_sequence_members(
    strategy: object, benchmark: object
) -> None:
    with pytest.raises(ValidationError):
        benchmark_stats(strategy, benchmark)


def test_metrics_public_functions_are_available_from_approved_modules() -> None:
    for module in ("annualize", "finite", "benchmark", "build"):
        assert find_spec(f"tradelab.metrics.{module}") is not None
