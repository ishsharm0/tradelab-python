"""Focused behavior tests for simulation and backtest-overfitting estimates."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import overload

import pytest

from tradelab.errors import ValidationError
from tradelab.research.monte_carlo import monte_carlo
from tradelab.research.pbo import probability_of_backtest_overfitting

_PNLS = [10, -5, 7, -3, 4, 6]


class CustomPnls(Sequence[int]):
    """A non-list/tuple sequence accepted by the public Monte Carlo API."""

    def __init__(self, values: list[int]) -> None:
        self._values = values

    @overload
    def __getitem__(self, index: int) -> int: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[int]: ...

    def __getitem__(self, index: int | slice) -> int | Sequence[int]:
        return self._values[index]

    def __len__(self) -> int:
        return len(self._values)


def test_monte_carlo_is_deterministic_and_uses_floor_percentiles() -> None:
    first = monte_carlo(trade_pnls=_PNLS, equity_start=1000, iterations=32, block_size=2, seed=42)
    second = monte_carlo(trade_pnls=_PNLS, equity_start=1000, iterations=32, block_size=2, seed=42)

    assert first == second
    assert first["final_equity"] == {"p5": 1006, "p25": 1015, "p50": 1022, "p75": 1026, "p95": 1034}
    assert first["path_bands"][1] == {"p5": 995, "p50": 1006, "p95": 1010}


def test_monte_carlo_accepts_a_caller_owned_rng() -> None:
    draws = iter([0.0, 0.5])

    result = monte_carlo(
        trade_pnls=[10, -4],
        equity_start=100,
        iterations=1,
        rng=lambda: next(draws),
    )

    assert result["final_equity"]["p50"] == 106


def test_monte_carlo_accepts_non_string_sequences() -> None:
    result = monte_carlo(trade_pnls=CustomPnls([10, -4]), equity_start=100, iterations=1, seed=1)

    assert result["iterations"] == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"trade_pnls": []},
        {"trade_pnls": [1, math.nan]},
        {"trade_pnls": [1], "iterations": 0},
        {"trade_pnls": [1], "iterations": True},
        {"trade_pnls": [1], "iterations": 1.5},
        {"trade_pnls": [1], "block_size": 0},
        {"trade_pnls": [1], "block_size": True},
        {"trade_pnls": [1], "block_size": 1.5},
        {"trade_pnls": [1], "equity_start": math.inf},
        {"trade_pnls": [1], "rng": 1},
    ],
)
def test_monte_carlo_rejects_invalid_numeric_inputs(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        monte_carlo(**kwargs)  # type: ignore[arg-type]


def test_pbo_matches_source_ranking_semantics() -> None:
    matrix = [
        [0.1, 0.2, -0.1, 0.3, 0.05, 0.2],
        [0.15, -0.1, 0.2, 0.1, 0.12, -0.05],
        [-0.05, 0.1, 0.05, -0.1, 0.2, 0.1],
    ]

    assert probability_of_backtest_overfitting(matrix, groups=4) == {
        "pbo": 1.0,
        "combos": 6,
        "median_logit": 0.0,
    }


@pytest.mark.parametrize(
    ("matrix", "groups"),
    [
        ([], 2),
        ([[0.1], [0.2]], 2),
        ([[0.1, 0.2], [0.3]], 2),
        ([[0.1, math.nan], [0.2, 0.3]], 2),
        ([[0.1, 0.2, 0.3, 0.4], [0.3, 0.4, 0.5, 0.6]], 3),
    ],
)
def test_pbo_rejects_malformed_matrices_and_groups(matrix: list[list[float]], groups: int) -> None:
    with pytest.raises(ValidationError):
        probability_of_backtest_overfitting(matrix, groups=groups)


@pytest.mark.parametrize(
    "matrix",
    [
        None,
        1,
        {},
        {"row": [0.1, 0.2]},
        [[10**100_000, 0.2], [0.3, 0.4]],
        [[1e308, -1e308, 1e308, -1e308], [0.1, 0.2, 0.3, 0.4]],
    ],
)
def test_pbo_wraps_malformed_top_level_and_huge_numbers_in_validation_errors(
    matrix: object,
) -> None:
    with pytest.raises(ValidationError, match="performance_matrix"):
        probability_of_backtest_overfitting(matrix, groups=2)  # type: ignore[arg-type]
