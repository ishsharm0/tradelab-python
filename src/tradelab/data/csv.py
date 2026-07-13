"""Candle normalization, CSV ingestion, merging, and descriptive statistics."""

from __future__ import annotations

import csv
import math
import re
import warnings
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

from tradelab.errors import ValidationError
from tradelab.utils.time import NEW_YORK

_TIME_ALIASES = ("date", "datetime", "timestamp", "ts", "open time", "opentime")
_COLUMN_ALIASES = {
    "open": ("o",),
    "high": ("h",),
    "low": ("l",),
    "close": ("c", "adj close"),
    "volume": ("v", "vol", "quantity"),
}
_NUMERIC_TEXT = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)")
_DATE_ONLY = re.compile(r"\d{4}-\d{2}-\d{2}")


def _number(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return 0.0
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _nonnegative_int(value: object, name: str) -> int:
    numeric = _number(value)
    if numeric is None or not numeric.is_integer() or numeric < 0:
        raise ValidationError(f"{name} must be a non-negative integer")
    return int(numeric)


def _compact(value: float) -> int | float:
    return int(value) if value.is_integer() else value


def _unquote_token(value: str) -> str:
    text = value.strip()
    if text[:1] in {"'", '"'}:
        text = text[1:]
    if text[-1:] in {"'", '"'}:
        text = text[:-1]
    return text


def _time_ms(value: object, custom_parser: Callable[[object], object] | None = None) -> int | float:
    if value is None or value == "":
        raise ValueError("missing date")
    if custom_parser is not None:
        parsed = custom_parser(value)
        if isinstance(parsed, datetime):
            value = parsed
        elif parsed is None:
            return 0
        else:
            custom_number = _number(parsed)
            if custom_number is not None:
                return _compact(custom_number)
    if isinstance(value, datetime):
        aware = value.replace(tzinfo=NEW_YORK) if value.tzinfo is None else value.astimezone(UTC)
        return _compact(aware.timestamp() * 1_000)
    text = _unquote_token(str(value))
    numeric = _number(text)
    if numeric is not None:
        return _compact(numeric * 1_000 if numeric < 1e11 else numeric)
    if _DATE_ONLY.fullmatch(text):
        parsed = datetime.strptime(text, "%Y-%m-%d").replace(tzinfo=UTC)
        return _compact(parsed.timestamp() * 1_000)
    dotted = re.fullmatch(r"(\d{4})\.(\d{2})\.(\d{2})\s+(\d{2}):(\d{2})(?::(\d{2}))?", text)
    if dotted:
        year, month, day, hour, minute, second = (int(item or 0) for item in dotted.groups())
        return _compact(
            datetime(year, month, day, hour, minute, second, tzinfo=NEW_YORK).timestamp() * 1_000
        )
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=NEW_YORK)
        return _compact(parsed.timestamp() * 1_000)
    except (ValueError, OverflowError, OSError) as error:
        try:
            parsed = datetime.strptime(text, "%B %d, %Y %H:%M:%S").replace(tzinfo=NEW_YORK)
            return _compact(parsed.timestamp() * 1_000)
        except ValueError:
            raise ValueError(f"cannot parse date: {text}") from error


def _date_boundary(value: object, fallback: float) -> int | float:
    if value is None or value is False or value == "":
        return fallback
    if isinstance(value, (int, float, bool)):
        return fallback
    text = str(value).strip()
    if _NUMERIC_TEXT.fullmatch(text):
        return fallback
    try:
        return _time_ms(value)
    except (TypeError, ValueError, OverflowError, OSError):
        return fallback


def _first(row: Mapping[str, object], *keys: str) -> object:
    return next((row[key] for key in keys if key in row and row[key] is not None), None)


def normalize_candles(candles: object) -> list[dict[str, int | float]]:
    """Normalize aliases, repair OHLC ranges, sort, and first-deduplicate rows."""
    if isinstance(candles, (str, bytes, bytearray)) or not isinstance(candles, Sequence):
        return []
    parsed: list[dict[str, int | float]] = []
    for value in candles:
        if not isinstance(value, Mapping):
            continue
        try:
            time = _time_ms(_first(value, "time", "timestamp", "date"))
        except (TypeError, ValueError, OverflowError, OSError):
            continue
        open_ = _number(_first(value, "open", "o"))
        high = _number(_first(value, "high", "h"))
        low = _number(_first(value, "low", "l"))
        close = _number(_first(value, "close", "c"))
        if any(item is None for item in (open_, high, low, close)):
            continue
        assert open_ is not None and high is not None and low is not None and close is not None
        volume = _number(_first(value, "volume", "v"))
        parsed.append(
            {
                "time": time,
                "open": open_,
                "high": max(high, open_, close),
                "low": min(low, open_, close),
                "close": close,
                "volume": volume if volume is not None else 0.0,
            }
        )
    reordered = any(
        parsed[index]["time"] < parsed[index - 1]["time"] for index in range(1, len(parsed))
    )
    normalized = sorted(parsed, key=lambda candle: candle["time"])
    output: list[dict[str, int | float]] = []
    seen: set[int | float] = set()
    for candle in normalized:
        if candle["time"] not in seen:
            seen.add(candle["time"])
            output.append(candle)
    if reordered or len(output) != len(parsed):
        warnings.warn(
            "[tradelab] normalizeCandles() reordered or deduplicated candles "
            f"(input={len(candles)}, valid={len(parsed)}, output={len(output)})",
            RuntimeWarning,
            stacklevel=2,
        )
    return output


def merge_candles(*arrays: Sequence[object]) -> list[dict[str, int | float]]:
    return normalize_candles([item for values in arrays for item in values])


def _resolve_column(
    column: str | int | float | None, headers: Mapping[str, int], aliases: Sequence[str]
) -> int:
    if (
        isinstance(column, (int, float))
        and not isinstance(column, bool)
        and math.isfinite(float(column))
        and float(column).is_integer()
        and column >= 0
    ):
        return int(column)
    for candidate in (column, *aliases):
        if candidate is not None and str(candidate).strip().lower() in headers:
            return headers[str(candidate).strip().lower()]
    return -1


def load_candles_from_csv(
    file_path: str | Path,
    *,
    delimiter: str = ",",
    skip_rows: int = 0,
    has_header: bool = True,
    time_col: str | int | float = "time",
    open_col: str | int | float = "open",
    high_col: str | int | float = "high",
    low_col: str | int | float = "low",
    close_col: str | int | float = "close",
    volume_col: str | int | float = "volume",
    start_date: object = None,
    end_date: object = None,
    custom_date_parser: Callable[[object], object] | None = None,
    **aliases: object,
) -> list[dict[str, int | float]]:
    """Read a CSV file with named or numeric columns and inclusive date bounds."""
    path = Path(file_path)
    if not path.exists():
        raise ValidationError(f"CSV file not found: {path}")
    skip_rows = _nonnegative_int(aliases.get("skipRows", skip_rows), "skip_rows")
    has_header = bool(aliases.get("hasHeader", has_header))
    time_col = aliases.get("timeCol", time_col)  # type: ignore[assignment]
    open_col = aliases.get("openCol", open_col)  # type: ignore[assignment]
    high_col = aliases.get("highCol", high_col)  # type: ignore[assignment]
    low_col = aliases.get("lowCol", low_col)  # type: ignore[assignment]
    close_col = aliases.get("closeCol", close_col)  # type: ignore[assignment]
    volume_col = aliases.get("volumeCol", volume_col)  # type: ignore[assignment]
    start_date = aliases.get("startDate", start_date)
    end_date = aliases.get("endDate", end_date)
    parser = aliases.get("customDateParser", custom_date_parser)
    custom_date_parser = parser if callable(parser) else custom_date_parser
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) <= skip_rows:
        raise ValidationError(f"CSV file is empty: {path}")
    rows = [
        [_unquote_token(value) for value in row]
        for row in csv.reader(lines[skip_rows:], delimiter=delimiter)
    ]
    header = rows[0] if has_header else []
    headers = {value.strip().lower(): index for index, value in enumerate(header)}
    indexes = {
        "time": _resolve_column(time_col, headers, _TIME_ALIASES),
        "open": _resolve_column(open_col, headers, _COLUMN_ALIASES["open"]),
        "high": _resolve_column(high_col, headers, _COLUMN_ALIASES["high"]),
        "low": _resolve_column(low_col, headers, _COLUMN_ALIASES["low"]),
        "close": _resolve_column(close_col, headers, _COLUMN_ALIASES["close"]),
        "volume": _resolve_column(volume_col, headers, _COLUMN_ALIASES["volume"]),
    }
    if any(indexes[key] < 0 for key in ("time", "open", "high", "low", "close")):
        raise ValidationError(f"Could not resolve required CSV columns in {path.name}")
    minimum = _date_boundary(start_date, -math.inf)
    maximum = _date_boundary(end_date, math.inf)
    candles: list[dict[str, object]] = []
    for columns in rows[1 if has_header else 0 :]:
        try:
            time = _time_ms(columns[indexes["time"]], custom_date_parser)
            if time < minimum or time > maximum:
                continue
            candle: dict[str, object] = {
                "time": time,
                "open": columns[indexes["open"]],
                "high": columns[indexes["high"]],
                "low": columns[indexes["low"]],
                "close": columns[indexes["close"]],
                "volume": (
                    columns[indexes["volume"]] if 0 <= indexes["volume"] < len(columns) else 0
                ),
            }
            candles.append(candle)
        except (IndexError, TypeError, ValueError, OverflowError, OSError):
            continue
    return normalize_candles(candles)


def _iso(time_ms: int | float) -> str:
    return (
        datetime.fromtimestamp(time_ms / 1_000, UTC)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def candle_stats(candles: object) -> dict[str, object] | None:
    normalized = normalize_candles(candles)
    if not normalized:
        return None
    gaps = [
        normalized[index]["time"] - normalized[index - 1]["time"]
        for index in range(1, min(len(normalized), 500))
        if normalized[index]["time"] > normalized[index - 1]["time"]
    ]
    gaps.sort()
    gap = gaps[len(gaps) // 2] if gaps else 0
    interval_minutes = math.floor(gap / 60_000 + 0.5)
    first, last = normalized[0], normalized[-1]
    return {
        "count": len(normalized),
        "firstTime": _iso(first["time"]),
        "lastTime": _iso(last["time"]),
        "durationDays": (last["time"] - first["time"]) / 86_400_000,
        "estimatedIntervalMin": interval_minutes,
        "priceRange": {
            "low": min(candle["low"] for candle in normalized),
            "high": max(candle["high"] for candle in normalized),
        },
    }


__all__ = ["candle_stats", "load_candles_from_csv", "merge_candles", "normalize_candles"]
