"""Stable JavaScript-compatible Cartesian parameter grids."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tradelab.errors import ValidationError

_MAX_ARRAY_INDEX = 2**32 - 2


def _array_index(key: str) -> int | None:
    if not key or not key.isascii() or not key.isdigit():
        return None
    if len(key) > 1 and key[0] == "0":
        return None
    value = int(key)
    return value if value <= _MAX_ARRAY_INDEX else None


def _object_keys(spec: Mapping[Any, object]) -> list[str]:
    keys = [str(key) for key in spec]
    indexed = sorted(
        ((index, key) for key in keys if (index := _array_index(key)) is not None),
        key=lambda item: item[0],
    )
    ordinary = [key for key in keys if _array_index(key) is None]
    return [key for _, key in indexed] + ordinary


def grid(spec: Mapping[Any, object] | None = None) -> list[dict[str, Any]]:
    """Expand list values while holding scalar values fixed."""
    if spec is None:
        spec = {}
    if not isinstance(spec, Mapping):
        raise ValidationError("grid spec must be a mapping")
    source = {str(key): value for key, value in spec.items()}
    output: list[dict[str, Any]] = [{}]
    for key in _object_keys(spec):
        raw_value = source[key]
        values = raw_value if isinstance(raw_value, list) else [raw_value]
        output = [{**base, key: value} for base in output for value in values]
    return output
