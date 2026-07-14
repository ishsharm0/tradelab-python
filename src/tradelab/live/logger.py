"""Structured JSON-lines logging for live components."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any, Protocol, TextIO


class AnyEventBus(Protocol):
    def on_any(self, handler: Callable[[dict[str, Any]], object]) -> Callable[[], None]: ...


_PRIORITIES = {"debug": 10, "info": 20, "warn": 30, "error": 40, "silent": 100}


def _level(value: object) -> str:
    return str(value) if value in _PRIORITIES else "info"


class LiveLogger:
    def __init__(
        self,
        *,
        level: str = "info",
        stream: TextIO = sys.stdout,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self.level = _level(level)
        self.stream = stream
        self._now = now
        self._unsubscribe: Callable[[], None] | None = None

    def should_log(self, level: str) -> bool:
        return _PRIORITIES[_level(level)] >= _PRIORITIES[self.level]

    def write(
        self, level: str, message: object, fields: Mapping[str, object] | None = None
    ) -> None:
        normalized = _level(level)
        if not self.should_log(normalized):
            return
        record: dict[str, object] = {
            "t": self._now().astimezone(UTC).isoformat().replace("+00:00", "Z"),
            "level": normalized,
            "msg": str(message),
        }
        record.update(dict(fields or {}))
        self.stream.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")

    def debug(self, message: object, fields: Mapping[str, object] | None = None) -> None:
        self.write("debug", message, fields)

    def info(self, message: object, fields: Mapping[str, object] | None = None) -> None:
        self.write("info", message, fields)

    def warn(self, message: object, fields: Mapping[str, object] | None = None) -> None:
        self.write("warn", message, fields)

    def error(self, message: object, fields: Mapping[str, object] | None = None) -> None:
        self.write("error", message, fields)

    def attach(self, event_bus: AnyEventBus | None) -> Callable[[], None]:
        if event_bus is None or not callable(getattr(event_bus, "on_any", None)):
            return lambda: None
        self.detach()

        def handler(envelope: dict[str, Any]) -> None:
            event = str(envelope.get("event", ""))
            level = (
                "error"
                if event == "error"
                else "warn"
                if event.startswith("risk:") or event in {"reconnecting", "disconnected"}
                else "info"
            )
            self.write(level, event, {"event": event, "payload": envelope.get("payload", {})})

        self._unsubscribe = event_bus.on_any(handler)
        return self.detach

    def detach(self) -> None:
        if self._unsubscribe is not None:
            unsubscribe, self._unsubscribe = self._unsubscribe, None
            unsubscribe()


def create_logger(**options: Any) -> LiveLogger:
    return LiveLogger(**options)
