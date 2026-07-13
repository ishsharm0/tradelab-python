"""Small, serializable domain models used throughout TradeLab."""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, fields, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, TypeAlias

from .errors import ValidationError

Primitive: TypeAlias = None | bool | int | float | str | list["Primitive"] | dict[str, "Primitive"]
MAX_SAFE_INTEGER = 2**53 - 1


def _utc_datetime(value: datetime | int) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValidationError("Candle time must be timezone-aware", context={"time": value})
        return value.astimezone(UTC)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(
            "Candle time must be Unix milliseconds or a datetime", context={"time": value}
        )
    return datetime.fromtimestamp(value / 1_000, tz=UTC)


@dataclass(frozen=True, slots=True, init=False)
class Candle:
    """An OHLCV market-data bar with its timestamp normalized to UTC."""

    time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float | None = None

    def __init__(
        self,
        time: datetime | int,
        open: float,
        high: float,
        low: float,
        close: float,
        volume: float | None = None,
    ) -> None:
        object.__setattr__(self, "time", _utc_datetime(time))
        object.__setattr__(self, "open", float(open))
        object.__setattr__(self, "high", float(high))
        object.__setattr__(self, "low", float(low))
        object.__setattr__(self, "close", float(close))
        object.__setattr__(self, "volume", float(volume) if volume is not None else None)

    @property
    def time_ms(self) -> int:
        """Timestamp as Unix milliseconds."""
        time = self.time
        if not isinstance(time, datetime):  # pragma: no cover - normalized in __post_init__
            raise AssertionError("Candle time was not normalized")
        return round(time.timestamp() * 1_000)

    def to_dict(self) -> dict[str, int | float | None]:
        """Return a stable primitive representation suitable for JSON encoding."""
        return {
            "time": self.time_ms,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }


@dataclass(frozen=True, slots=True)
class Signal:
    """A directional trading signal."""

    side: str
    stop: float | None = None

    def __post_init__(self) -> None:
        normalized = self.side.strip().lower()
        aliases = {"buy": "long", "sell": "short", "long": "long", "short": "short"}
        try:
            object.__setattr__(self, "side", aliases[normalized])
        except KeyError as error:
            raise ValidationError(
                "Signal side must be buy, sell, long, or short", context={"side": self.side}
            ) from error
        if self.stop is not None:
            object.__setattr__(self, "stop", float(self.stop))

    @property
    def normalized_side(self) -> str:
        """Return the canonical long or short form of the signal direction."""
        return self.side


def _freeze_json(value: object, active: set[int] | None = None) -> object:
    """Copy JSON data into immutable containers and reject invalid numerics."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        if abs(value) > MAX_SAFE_INTEGER:
            raise ValidationError("BacktestResult integer exceeds portable JSON range")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValidationError("BacktestResult requires finite JSON numbers")
        return value
    if isinstance(value, Mapping):
        seen = active if active is not None else set()
        identity = id(value)
        if identity in seen:
            raise ValidationError("BacktestResult cannot contain cyclic JSON data")
        seen.add(identity)
        try:
            return MappingProxyType(
                {str(key): _freeze_json(item, seen) for key, item in value.items()}
            )
        finally:
            seen.remove(identity)
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray)):
        seen = active if active is not None else set()
        identity = id(value)
        if identity in seen:
            raise ValidationError("BacktestResult cannot contain cyclic JSON data")
        seen.add(identity)
        try:
            return tuple(_freeze_json(item, seen) for item in value)
        finally:
            seen.remove(identity)
    return _freeze_json(to_primitive(value), active)


def _thaw_json(value: object) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


@dataclass(frozen=True, slots=True, init=False)
class BacktestResult(Mapping[str, Any]):
    """Immutable typed snapshot returned by every simulation engine."""

    _data: Mapping[str, object]

    def __init__(self, payload: Mapping[str, object]) -> None:
        frozen = _freeze_json(payload)
        if not isinstance(frozen, Mapping):  # pragma: no cover - guaranteed by input type
            raise ValidationError("BacktestResult payload must be a mapping")
        object.__setattr__(self, "_data", frozen)

    def __getitem__(self, key: str) -> Any:
        return _thaw_json(self._data[key])

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def to_dict(self) -> dict[str, Any]:
        """Return a fresh JSON-compatible dictionary."""
        return {key: _thaw_json(value) for key, value in self._data.items()}


def to_primitive(value: object) -> Primitive:
    """Recursively turn supported values into JSON-compatible primitives."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Candle):
        return to_primitive(value.to_dict())
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat() if value.tzinfo is not None else value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): to_primitive(item) for key, item in value.items()}
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: to_primitive(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Iterable):
        return [to_primitive(item) for item in value]
    return str(value)
