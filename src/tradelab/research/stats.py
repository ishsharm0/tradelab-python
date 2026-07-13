"""Numerically explicit statistics used by TradeLab research tools."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from tradelab.errors import ValidationError

Number = int | float


@dataclass(frozen=True)
class Moments:
    """Population moments for a sequence of numeric values."""

    mean: float
    std: float
    skew: float
    kurtosis: float


def _finite_number(value: Number, *, name: str) -> float:
    """Normalize a finite real numeric value or raise a contextual error."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValidationError(f"{name} must be a finite number", context={name: value})
    return float(value)


def normal_cdf(value: Number) -> float:
    """Return the standard-normal CDF using Abramowitz and Stegun 7.1.26."""
    x = _finite_number(value, name="value")
    sign = -1 if x < 0 else 1
    absolute_x = abs(x) / math.sqrt(2)
    t = 1 / (1 + 0.3275911 * absolute_x)
    y = 1 - (
        ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592)
        * t
        * math.exp(-absolute_x * absolute_x)
    )
    return 0.5 * (1 + sign * y)


def normal_ppf(probability: Number) -> float:
    """Return the inverse standard-normal CDF with Acklam's approximation."""
    p = _finite_number(probability, name="probability")
    if p <= 0:
        return -math.inf
    if p >= 1:
        return math.inf

    a = (
        -3.969683028665376e1,
        2.209460984245205e2,
        -2.759285104469687e2,
        1.38357751867269e2,
        -3.066479806614716e1,
        2.506628277459239,
    )
    b = (
        -5.447609879822406e1,
        1.615858368580409e2,
        -1.556989798598866e2,
        6.680131188771972e1,
        -1.328068155288572e1,
    )
    c = (
        -7.784894002430293e-3,
        -3.223964580411365e-1,
        -2.400758277161838,
        -2.549732539343734,
        4.374664141464968,
        2.938163982698783,
    )
    d = (7.784695709041462e-3, 3.224671290700398e-1, 2.445134137142996, 3.754408661907416)
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        numerator = c[0]
        for coefficient in c[1:]:
            numerator = numerator * q + coefficient
        denominator = d[0]
        for coefficient in d[1:]:
            denominator = denominator * q + coefficient
        denominator = denominator * q + 1
        return numerator / denominator
    if p <= phigh:
        q = p - 0.5
        r = q * q
        numerator = a[0]
        for coefficient in a[1:]:
            numerator = numerator * r + coefficient
        numerator *= q
        denominator = b[0]
        for coefficient in b[1:]:
            denominator = denominator * r + coefficient
        denominator = denominator * r + 1
        return numerator / denominator
    q = math.sqrt(-2 * math.log(1 - p))
    numerator = c[0]
    for coefficient in c[1:]:
        numerator = numerator * q + coefficient
    denominator = d[0]
    for coefficient in d[1:]:
        denominator = denominator * q + coefficient
    denominator = denominator * q + 1
    return -(numerator / denominator)


def moments(values: Sequence[Number]) -> Moments:
    """Return population mean, deviation, skew, and Pearson kurtosis.

    Accumulation is deliberately left-to-right to match the JavaScript source.
    """
    if not values:
        raise ValidationError("values must be non-empty", context={"values": values})
    numeric_values = [_finite_number(value, name="values") for value in values]
    count = len(numeric_values)
    if count < 2:
        return Moments(mean=numeric_values[0], std=0.0, skew=0.0, kurtosis=3.0)

    total = 0.0
    for value in numeric_values:
        total += value
    mean = total / count
    second = 0.0
    third = 0.0
    fourth = 0.0
    for value in numeric_values:
        distance = value - mean
        second += distance * distance
        third += distance * distance * distance
        fourth += distance * distance * distance * distance
    second /= count
    third /= count
    fourth /= count
    std = math.sqrt(second)
    skew = 0.0 if std == 0 else third / std**3
    kurtosis = 3.0 if second == 0 else fourth / second**2
    return Moments(mean=mean, std=std, skew=skew, kurtosis=kurtosis)
