"""Thread-safe runtime strategy registry."""

from __future__ import annotations

import math
from collections.abc import Mapping
from copy import deepcopy
from threading import RLock
from typing import Any, cast

from tradelab.errors import ValidationError

from .builtins import BUILTINS, SignalFactory

_LOCK = RLock()
_REGISTRY: dict[str, dict[str, Any]] = deepcopy(BUILTINS)


def _name(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError("strategy name must be a non-empty string")
    return value


def _metadata_json(value: object, active: set[int] | None = None) -> object:
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        if abs(value) > 2**53 - 1:
            raise ValidationError("strategy metadata integer exceeds portable JSON range")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValidationError("strategy metadata numbers must be finite")
        return value
    seen = active if active is not None else set()
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in seen:
            raise ValidationError("strategy metadata cannot be cyclic")
        seen.add(identity)
        try:
            return {str(key): _metadata_json(item, seen) for key, item in value.items()}
        finally:
            seen.remove(identity)
    if isinstance(value, (list, tuple)):
        identity = id(value)
        if identity in seen:
            raise ValidationError("strategy metadata cannot be cyclic")
        seen.add(identity)
        try:
            return [_metadata_json(item, seen) for item in value]
        finally:
            seen.remove(identity)
    raise ValidationError("strategy metadata must be finite JSON data")


def register_strategy(name: str, definition: Mapping[str, object]) -> None:
    """Atomically register or replace one BUILTINS-shaped definition."""
    validated_name = _name(name)
    factory = definition.get("factory") if isinstance(definition, Mapping) else None
    if not callable(factory):
        raise ValidationError(f'registerStrategy("{name}") requires a factory function')
    description = definition.get("description")
    if description is not None and not isinstance(description, str):
        raise ValidationError("strategy description must be a string or None")
    copied = {
        "description": description,
        "params": _metadata_json(definition.get("params")),
        "factory": factory,
    }
    with _LOCK:
        _REGISTRY[validated_name] = copied


def list_strategies() -> list[dict[str, object]]:
    """Return fresh metadata records in deterministic registration order."""
    with _LOCK:
        return [
            {
                "name": name,
                "description": deepcopy(definition.get("description")),
                "params": deepcopy(definition.get("params")),
            }
            for name, definition in _REGISTRY.items()
        ]


def get_strategy(name: str) -> SignalFactory:
    """Return the named signal factory or raise with all available names."""
    validated_name = _name(name)
    with _LOCK:
        definition = _REGISTRY.get(validated_name)
        if definition is None:
            available = ", ".join(_REGISTRY)
            raise ValidationError(f'Unknown strategy "{name}". Available: {available}')
        factory = definition["factory"]
    return cast(SignalFactory, factory)


__all__ = ["get_strategy", "list_strategies", "register_strategy"]
