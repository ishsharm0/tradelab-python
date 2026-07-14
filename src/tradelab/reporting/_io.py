"""Strict serialization, safe filenames, and atomic reporting writes."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tradelab.errors import ValidationError

_SAFE = frozenset("-_.ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")


def _js_string(value: object) -> str:
    if value is None:
        return "undefined"
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


def safe_segment(value: object) -> str:
    """Apply the JavaScript ASCII filename sanitizer by UTF-16 code unit."""
    units = _js_string(value).encode("utf-16-le", "surrogatepass")
    output: list[str] = []
    for index in range(0, len(units), 2):
        code = int.from_bytes(units[index : index + 2], "little")
        character = chr(code)
        output.append(character if code < 128 and character in _SAFE else "_")
    return "".join(output)


def iso_milliseconds(value: object) -> str:
    """Return JavaScript ``Date.toISOString()``-style UTC text."""
    if isinstance(value, datetime):
        parsed = value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
    elif isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError("report timestamp must be numeric or datetime")
    else:
        try:
            parsed = datetime.fromtimestamp(float(value) / 1_000, UTC)
        except (OverflowError, OSError, ValueError) as error:
            raise ValidationError("report timestamp is outside the supported range") from error
    return parsed.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _json_value(value: object, active: set[int] | None = None) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValidationError("report data must use strict JSON with finite numbers")
        return value
    if isinstance(value, Mapping):
        seen = active if active is not None else set()
        identity = id(value)
        if identity in seen:
            raise ValidationError("report data must use strict JSON without cycles")
        seen.add(identity)
        try:
            return {str(key): _json_value(item, seen) for key, item in value.items()}
        finally:
            seen.remove(identity)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        seen = active if active is not None else set()
        identity = id(value)
        if identity in seen:
            raise ValidationError("report data must use strict JSON without cycles")
        seen.add(identity)
        try:
            return [_json_value(item, seen) for item in value]
        finally:
            seen.remove(identity)
    raise ValidationError(
        "report data must use strict JSON primitives",
        context={"type": type(value).__name__},
    )


def strict_json_dumps(value: object, *, html_safe: bool = False) -> str:
    """Serialize strict JSON and optionally make it safe inside a script element."""
    serialized = json.dumps(
        _json_value(value),
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        separators=(",", ": "),
    )
    if html_safe:
        serialized = (
            serialized.replace("<", "\\u003c")
            .replace("\u2028", "\\u2028")
            .replace("\u2029", "\\u2029")
        )
    return serialized


def atomic_write_text(path: Path, content: str) -> Path:
    """Replace a UTF-8 artifact atomically and remove failed temporary files."""
    temporary: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        return path
    except OSError as error:
        raise ValidationError(f"Could not write report artifact: {path}") from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
