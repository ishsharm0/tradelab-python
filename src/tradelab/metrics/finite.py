"""Finite-number validation and JSON-safe metric values."""

from __future__ import annotations

import math
from typing import TypeAlias, TypeGuard, overload

from tradelab.errors import ValidationError

BIG_NUMBER = 1e9
Number: TypeAlias = int | float


def _is_number(value: object) -> TypeGuard[Number]:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _finite_number(value: object, name: str) -> float:
    if not _is_number(value):
        raise ValidationError(f"{name} must be a finite non-boolean number")
    try:
        number = float(value)
    except (OverflowError, ValueError):
        raise ValidationError(f"{name} must be a finite non-boolean number") from None
    if not math.isfinite(number):
        raise ValidationError(f"{name} must be a finite non-boolean number")
    return number


def _json_number(value: object) -> float | None:
    """Return the JSON representation of a JavaScript numeric object property."""
    if not _is_number(value):
        return None
    try:
        number = float(value)
    except (OverflowError, ValueError):
        return None
    return number if math.isfinite(number) else None


@overload
def clamp_finite(value: object) -> float: ...


@overload
def clamp_finite(value: object, fallback: Number | None) -> float | None: ...


def clamp_finite(value: object, fallback: Number | None = 0.0) -> float | None:
    """Return a finite JSON-safe metric, retaining finite values without capping."""
    if _is_number(value):
        try:
            number = float(value)
        except OverflowError:
            return BIG_NUMBER if value > 0 else -BIG_NUMBER
        except ValueError:
            number = math.nan
        if number == math.inf:
            return BIG_NUMBER
        if number == -math.inf:
            return -BIG_NUMBER
        if math.isfinite(number):
            return number
    return _json_number(fallback)
