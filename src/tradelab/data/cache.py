"""Atomic, JavaScript-compatible candle cache files."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

from tradelab.errors import ValidationError

from .csv import normalize_candles

_SAFE = frozenset("-_.ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")


def _js_string(value: object) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        if value == 0:
            return "0"
        if value.is_integer():
            return str(int(value))
    if isinstance(value, (list, tuple)):
        return ",".join("" if item is None else _js_string(item) for item in value)
    if isinstance(value, Mapping):
        return "[object Object]"
    return str(value)


def _safe_segment(value: object) -> str:
    units = _js_string(value).encode("utf-16-le", "surrogatepass")
    output: list[str] = []
    for index in range(0, len(units), 2):
        code = int.from_bytes(units[index : index + 2], "little")
        character = chr(code)
        output.append(character if code < 128 and character in _SAFE else "_")
    return "".join(output)


def cached_candles_path(
    symbol: object, interval: object, period: object, out_dir: str | Path = "output/data"
) -> Path:
    return Path(out_dir) / (
        f"candles-{_safe_segment(symbol)}-{_safe_segment(interval)}-{_safe_segment(period)}.json"
    )


def save_candles_to_cache(
    candles: object,
    *,
    symbol: object = "UNKNOWN",
    interval: object = "tf",
    period: object = "range",
    out_dir: str | Path = "output/data",
    source: object = None,
    now: datetime | None = None,
) -> Path:
    path = cached_candles_path(symbol, interval, period, out_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = normalize_candles(candles)
    payload = {
        "symbol": symbol,
        "interval": interval,
        "period": period,
        "source": source,
        "count": len(normalized),
        "asOf": (now or datetime.now(UTC))
        .astimezone(UTC)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z"),
        "candles": normalized,
    }
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        return path
    except (OSError, TypeError, ValueError) as error:
        raise ValidationError(f"Could not save candle cache: {path}") from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_candles_from_cache(
    symbol: object, interval: object, period: object, out_dir: str | Path = "output/data"
) -> list[dict[str, int | float]] | None:
    path = cached_candles_path(symbol, interval, period, out_dir)
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant {value}")
            ),
        )
    except (OSError, json.JSONDecodeError, UnicodeError, ValueError):
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("candles"), list):
        return None
    return normalize_candles(payload["candles"])


__all__ = ["cached_candles_path", "load_candles_from_cache", "save_candles_to_cache"]
