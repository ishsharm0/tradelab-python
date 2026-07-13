"""Small, serializable domain models used throughout TradeLab."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeAlias

from .errors import ValidationError

Primitive: TypeAlias = None | bool | int | float | str | list["Primitive"] | dict[str, "Primitive"]


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


@dataclass(frozen=True, slots=True)
class Candle:
    """An OHLCV market-data bar with its timestamp normalized to UTC."""

    time: datetime | int
    open: float
    high: float
    low: float
    close: float
    volume: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "time", _utc_datetime(self.time))
        object.__setattr__(self, "open", float(self.open))
        object.__setattr__(self, "high", float(self.high))
        object.__setattr__(self, "low", float(self.low))
        object.__setattr__(self, "close", float(self.close))
        if self.volume is not None:
            object.__setattr__(self, "volume", float(self.volume))

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
