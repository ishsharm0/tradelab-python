"""Finite-number validation and JSON-safe metric values."""

from __future__ import annotations

import math
from typing import TypeAlias, TypeGuard

from tradelab.errors import ValidationError

BIG_NUMBER = 1e9
Number: TypeAlias = int | float


def _is_number(value: object) -> TypeGuard[Number]:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _finite_number(value: object, name: str) -> float:
    if not _is_number(value):
        raise ValidationError(f"{name} must be a finite non-boolean number")
    number = float(value)
    if not math.isfinite(number):
        raise ValidationError(f"{name} must be a finite non-boolean number")
    return number


def clamp_finite(value: object, fallback: float = 0.0) -> float:
    """Return a finite JSON-safe metric, retaining finite values without capping."""
    if _is_number(value):
        number = float(value)
        if number == math.inf:
            return BIG_NUMBER
        if number == -math.inf:
            return -BIG_NUMBER
        if math.isfinite(number):
            return number
    return fallback
