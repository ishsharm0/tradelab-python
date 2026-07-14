"""Synchronous ordered events shared by live components."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

EventHandler = Callable[[dict[str, Any]], object]

LIVE_EVENTS = (
    "signal",
    "order:submitted",
    "order:filled",
    "order:canceled",
    "order:rejected",
    "order:modified",
    "position:opened",
    "position:updated",
    "position:closed",
    "equity:update",
    "risk:warning",
    "risk:halt",
    "bar",
    "tick",
    "error",
    "connected",
    "disconnected",
    "reconnecting",
    "shutdown",
    "stateRestored",
    "reconciled",
)


class EventBus:
    """Small EventEmitter-compatible bus with a wildcard channel."""

    def __init__(self) -> None:
        self._handlers: dict[str, list[EventHandler]] = {}

    def on(self, event: str, handler: EventHandler) -> Callable[[], None]:
        if not isinstance(event, str) or not event:
            raise TypeError("event must be a non-empty string")
        if not callable(handler):
            raise TypeError("handler must be callable")
        self._handlers.setdefault(event, []).append(handler)
        removed = False

        def unsubscribe() -> None:
            nonlocal removed
            if removed:
                return
            removed = True
            self.off(event, handler)

        return unsubscribe

    def off(self, event: str, handler: EventHandler) -> None:
        current = self._handlers.get(event)
        if not current:
            return
        self._handlers[event] = [candidate for candidate in current if candidate is not handler]
        if not self._handlers[event]:
            self._handlers.pop(event, None)

    def emit(self, event: str, payload: Mapping[str, Any] | None = None) -> bool:
        value = dict(payload or {})
        for handler in tuple(self._handlers.get(event, ())):
            handler(value)
        return True

    def emit_event(self, event: str, payload: Mapping[str, Any] | None = None) -> bool:
        value = dict(payload or {})
        self.emit(event, value)
        self.emit("*", {"event": event, "payload": value})
        return True

    def on_any(self, handler: EventHandler) -> Callable[[], None]:
        return self.on("*", handler)

    def remove_all_listeners(self) -> None:
        self._handlers.clear()


def create_event_bus() -> EventBus:
    return EventBus()
