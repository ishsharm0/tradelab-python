"""Focused compatibility tests for public metric primitives."""

from __future__ import annotations

import math

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


@pytest.mark.parametrize(
    "strategy, benchmark",
    [([math.nan], [0.0]), ([0.0], [math.inf]), (True, [0.0])],
)
def test_benchmark_stats_rejects_malformed_or_nonfinite_public_inputs(
    strategy: object, benchmark: object
) -> None:
    with pytest.raises(ValidationError):
        benchmark_stats(strategy, benchmark)  # type: ignore[arg-type]
