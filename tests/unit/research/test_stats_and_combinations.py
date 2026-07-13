"""Focused behavior tests for research statistics and combinations."""

from __future__ import annotations

import math

import pytest

from tradelab.errors import ValidationError
from tradelab.research.combinations import combinations
from tradelab.research.stats import moments, normal_cdf, normal_ppf


def test_normal_distribution_functions_match_pinned_approximations() -> None:
    assert normal_cdf(1.25) == pytest.approx(0.894350157794624)
    assert normal_ppf(0.9) == pytest.approx(1.2815515641401563)
    assert normal_ppf(0) == -math.inf
    assert normal_ppf(1) == math.inf


def test_moments_preserve_left_to_right_accumulation() -> None:
    values = [1e16, 1.0, -1e16]

    assert moments(values).mean == 0.0


def test_moments_returns_population_statistics() -> None:
    result = moments([1, 2, 4, 8, 16])

    assert result.mean == pytest.approx(6.2)
    assert result.std == pytest.approx(5.455272678794343)
    assert result.skew == pytest.approx(0.889048134816954)
    assert result.kurtosis == pytest.approx(2.3259408602150526)


@pytest.mark.parametrize("values", [[], [1.0, math.nan], [1.0, math.inf]])
def test_moments_rejects_empty_or_nonfinite_values(values: list[float]) -> None:
    with pytest.raises(ValidationError):
        moments(values)


def test_combinations_are_lexicographic() -> None:
    assert combinations(4, 2) == [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]


@pytest.mark.parametrize("n,k", [(-1, 0), (3, -1), (3, 4), (True, 1)])
def test_combinations_rejects_invalid_dimensions(n: int, k: int) -> None:
    with pytest.raises(ValidationError):
        combinations(n, k)
