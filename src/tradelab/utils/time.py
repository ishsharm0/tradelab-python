"""New York session and wall-clock utilities."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

from tradelab.errors import ValidationError

NEW_YORK = ZoneInfo("America/New_York")
Window = dict[str, int]


def _utc_datetime(time_ms: int | float | datetime) -> datetime:
    if isinstance(time_ms, datetime):
        if time_ms.tzinfo is None or time_ms.utcoffset() is None:
            raise ValidationError("time must be timezone-aware", context={"time": time_ms})
        return time_ms.astimezone(UTC)
    if isinstance(time_ms, bool) or not isinstance(time_ms, (int, float)):
        raise ValidationError("time must be Unix milliseconds", context={"time": time_ms})
    try:
        return datetime.fromtimestamp(time_ms / 1_000, tz=UTC)
    except (OverflowError, OSError, ValueError) as error:
        raise ValidationError(
            "time must be Unix milliseconds", context={"time": time_ms}
        ) from error


def _et_datetime(time_ms: int | float | datetime) -> datetime:
    return _utc_datetime(time_ms).astimezone(NEW_YORK)


def offset_et(time_ms: int | float | datetime) -> int:
    """Return the positive number of hours New York is behind UTC at a timestamp."""
    offset = _et_datetime(time_ms).utcoffset()
    if offset is None:  # pragma: no cover - ZoneInfo always supplies one for an aware datetime.
        raise AssertionError("New York timestamp has no UTC offset")
    return int(-offset.total_seconds() / 3_600)


def minutes_et(time_ms: int | float | datetime) -> int:
    """Return New York wall-clock minutes since midnight."""
    value = _et_datetime(time_ms)
    return value.hour * 60 + value.minute


def is_session(time_ms: int | float | datetime, session: str = "NYSE") -> bool:
    """Return whether a timestamp is within the named NYSE, futures, or AUTO session."""
    utc_value = _utc_datetime(time_ms)
    value = utc_value.astimezone(NEW_YORK)
    name = session.upper()
    if name not in {"NYSE", "FUT", "AUTO"}:
        raise ValidationError("session must be NYSE, FUT, or AUTO", context={"session": session})
    minutes = value.hour * 60 + value.minute
    if utc_value.weekday() >= 5:
        return name == "FUT" and (minutes >= 18 * 60 or minutes < 17 * 60)
    if name == "AUTO":
        return True
    if name == "FUT":
        return not 17 * 60 <= minutes < 18 * 60
    return 9 * 60 + 30 <= minutes <= 16 * 60


def parse_windows_csv(csv: str | None) -> list[Window] | None:
    """Parse comma-separated inclusive New York time windows such as ``09:30-16:00``."""
    if csv is None or not csv.strip():
        return None
    windows: list[Window] = []
    for token in csv.split(","):
        text = token.strip()
        if not text or text.count("-") != 1:
            raise ValidationError("invalid time window", context={"window": token})
        start, end = (part.strip() for part in text.split("-"))
        start_minutes = _parse_clock(start)
        end_minutes = _parse_clock(end)
        if start_minutes > end_minutes:
            raise ValidationError("window start must not be after end", context={"window": token})
        windows.append({"aMin": start_minutes, "bMin": end_minutes})
    return windows or None


def _parse_clock(value: str) -> int:
    parts = value.split(":")
    if len(parts) != 2:
        raise ValidationError("invalid wall-clock time", context={"time": value})
    try:
        hour, minute = (int(part) for part in parts)
    except ValueError as error:
        raise ValidationError("invalid wall-clock time", context={"time": value}) from error
    if not 0 <= hour <= 23 or not 0 <= minute <= 59:
        raise ValidationError("invalid wall-clock time", context={"time": value})
    return hour * 60 + minute


def in_windows_et(time_ms: int | float | datetime, windows: Sequence[Window] | None) -> bool:
    """Return whether a timestamp lies inside at least one inclusive NY time window."""
    if not windows:
        return True
    minutes = minutes_et(time_ms)
    return any(window["aMin"] <= minutes <= window["bMin"] for window in windows)
