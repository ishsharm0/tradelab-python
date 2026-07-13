"""Benchmark-relative statistics."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TypeGuard

from .finite import _finite_number

_NULL_BENCHMARK: dict[str, None] = {
    "alpha": None,
    "beta": None,
    "correlation": None,
    "information_ratio": None,
    "tracking_error": None,
}


def _is_sequence(value: object) -> TypeGuard[Sequence[object]]:
    return not isinstance(value, (str, bytes, bytearray)) and isinstance(value, Sequence)


def _sum(values: Sequence[float]) -> float:
    total = 0.0
    for value in values:
        total += value
    return total


def _mean(values: Sequence[float]) -> float:
    return _sum(values) / len(values) if values else 0.0


def _stddev(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    average = _mean(values)
    squared_differences: list[float] = []
    for value in values:
        difference = value - average
        squared_differences.append(difference * difference)
    return math.sqrt(_mean(squared_differences))


def benchmark_stats(strategy_returns: object, benchmark_returns: object) -> dict[str, float | None]:
    """Return population OLS and active-return metrics for aligned return series."""
    if not _is_sequence(strategy_returns) or not _is_sequence(benchmark_returns):
        return dict(_NULL_BENCHMARK)
    if not strategy_returns or len(strategy_returns) != len(benchmark_returns):
        return dict(_NULL_BENCHMARK)

    strategy = [_finite_number(value, "strategy_returns item") for value in strategy_returns]
    benchmark = [_finite_number(value, "benchmark_returns item") for value in benchmark_returns]
    mean_strategy = _mean(strategy)
    mean_benchmark = _mean(benchmark)
    covariance = 0.0
    variance_benchmark = 0.0
    variance_strategy = 0.0
    for index, strategy_value in enumerate(strategy):
        strategy_delta = strategy_value - mean_strategy
        benchmark_delta = benchmark[index] - mean_benchmark
        covariance += strategy_delta * benchmark_delta
        variance_benchmark += benchmark_delta * benchmark_delta
        variance_strategy += strategy_delta * strategy_delta

    beta = 0.0 if variance_benchmark == 0 else covariance / variance_benchmark
    alpha = mean_strategy - beta * mean_benchmark
    denominator = math.sqrt(variance_strategy * variance_benchmark)
    correlation = 0.0 if denominator == 0 else covariance / denominator
    active: list[float] = []
    for index, strategy_value in enumerate(strategy):
        active.append(strategy_value - benchmark[index])
    mean_active = _mean(active)
    tracking_error = _stddev(active)
    information_ratio = 0.0 if tracking_error == 0 else mean_active / tracking_error
    return {
        "alpha": alpha,
        "beta": beta,
        "correlation": correlation,
        "information_ratio": information_ratio,
        "tracking_error": tracking_error,
    }
