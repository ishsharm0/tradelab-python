"""Deflated-Sharpe calculations for multiple strategy trials."""

from __future__ import annotations

import math
from typing import TypedDict

from tradelab.errors import ValidationError
from tradelab.research.stats import normal_cdf, normal_ppf

_EULER_MASCHERONI = 0.5772156649015329
Number = int | float


class SweepHaircut(TypedDict):
    """Expected null maximum Sharpe across a sweep of trials."""

    expected_max_sharpe: float
    num_trials: int


def _finite(value: Number, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{name} must be a finite number", context={name: value})
    try:
        normalized = float(value)
    except OverflowError as error:
        raise ValidationError(f"{name} must be a finite number", context={name: value}) from error
    if not math.isfinite(normalized):
        raise ValidationError(f"{name} must be a finite number", context={name: value})
    return normalized


def _positive_integer(value: Number, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValidationError(f"{name} must be a positive integer", context={name: value})
    _finite(value, name=name)
    return value


def sweep_haircut(*, num_trials: Number, sharpe_std: Number) -> SweepHaircut:
    """Return the expected maximum null Sharpe from independent trials."""
    trials = _positive_integer(num_trials, name="num_trials")
    standard_deviation = _finite(sharpe_std, name="sharpe_std")
    if standard_deviation < 0:
        raise ValidationError("sharpe_std must be non-negative", context={"sharpe_std": sharpe_std})
    if trials == 1:
        return {"expected_max_sharpe": 0.0, "num_trials": trials}
    a = normal_ppf(1 - 1 / trials)
    b = normal_ppf(1 - 1 / (trials * math.e))
    expected = standard_deviation * ((1 - _EULER_MASCHERONI) * a + _EULER_MASCHERONI * b)
    return {"expected_max_sharpe": expected, "num_trials": trials}


def deflated_sharpe(
    *,
    sharpe: Number,
    sample_size: Number,
    num_trials: Number = 1,
    sharpe_std: Number = 0,
    skew: Number = 0,
    kurtosis: Number = 3,
) -> float:
    """Return the deflated Sharpe probability adjusted for trial selection."""
    observed_sharpe = _finite(sharpe, name="sharpe")
    size = _positive_integer(sample_size, name="sample_size")
    skewness = _finite(skew, name="skew")
    kurt = _finite(kurtosis, name="kurtosis")
    null_sharpe = sweep_haircut(num_trials=num_trials, sharpe_std=sharpe_std)["expected_max_sharpe"]
    squared_sharpe = observed_sharpe * observed_sharpe
    variance_term = 1 - skewness * observed_sharpe + ((kurt - 1) / 4) * squared_sharpe
    if not math.isfinite(variance_term):
        raise ValidationError("Sharpe inputs produced a non-finite variance adjustment")
    denominator = math.sqrt(max(1e-12, variance_term))
    z_score = ((observed_sharpe - null_sharpe) * math.sqrt(max(1, size - 1))) / denominator
    return normal_cdf(z_score)
