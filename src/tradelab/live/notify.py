"""Best-effort callbacks and webhooks for selected live events."""

from __future__ import annotations

import asyncio
import inspect
import math
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import suppress
from typing import Any, Protocol

import httpx

from tradelab.errors import ValidationError

Notifier = Callable[[dict[str, Any]], object]
WebhookClient = Callable[[str, Mapping[str, object]], Awaitable[None]]


class NotifierSource(Protocol):
    event_bus: Any


async def _post_webhook(url: str, envelope: Mapping[str, object]) -> None:
    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.post(url, json=dict(envelope))
        response.raise_for_status()


class NotifierHandle:
    def __init__(self, unsubscribe: Callable[[], None]) -> None:
        self._unsubscribe = unsubscribe
        self._tasks: set[asyncio.Task[None]] = set()
        self._closed = False

    def add(self, coroutine: Awaitable[None]) -> None:
        if self._closed:
            return
        task = asyncio.ensure_future(coroutine)
        self._tasks.add(task)

        def complete(done: asyncio.Task[None]) -> None:
            self._tasks.discard(done)
            if not done.cancelled():
                done.exception()

        task.add_done_callback(complete)

    async def drain(self) -> None:
        while self._tasks:
            await asyncio.gather(*tuple(self._tasks), return_exceptions=True)

    def __call__(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._unsubscribe()
        for task in tuple(self._tasks):
            task.cancel()


def _drawdown(value: object) -> float:
    if isinstance(value, bool):
        raise ValidationError("drawdown_pct must be finite and non-negative")
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as error:
        raise ValidationError("drawdown_pct must be finite and non-negative") from error
    if not math.isfinite(number) or number < 0:
        raise ValidationError("drawdown_pct must be finite and non-negative")
    return number


def attach_notifier(
    source: NotifierSource,
    *,
    on_event: Notifier | None = None,
    webhook_url: str | None = None,
    events: Sequence[str] = ("order:filled", "risk:halt"),
    drawdown_pct: object = 0,
    webhook_client: WebhookClient = _post_webhook,
) -> NotifierHandle:
    bus = getattr(source, "event_bus", None)
    if bus is None or not callable(getattr(bus, "on_any", None)):
        raise ValidationError("notifier source must expose event_bus.on_any")
    threshold = _drawdown(drawdown_pct)
    wanted = set(events)
    peak: float | None = None
    handle: NotifierHandle

    async def deliver(event: str, payload: Mapping[str, object]) -> None:
        envelope = {"event": event, "payload": dict(payload)}
        if on_event is not None:
            with suppress(Exception):
                outcome = on_event(envelope)
                if inspect.isawaitable(outcome):
                    await outcome
        if webhook_url:
            with suppress(Exception):
                await webhook_client(webhook_url, envelope)

    def handler(envelope: dict[str, Any]) -> None:
        nonlocal peak
        event = str(envelope.get("event", ""))
        raw_payload = envelope.get("payload")
        payload = dict(raw_payload) if isinstance(raw_payload, Mapping) else {}
        if event in wanted:
            handle.add(deliver(event, payload))
            return
        if threshold > 0 and event == "equity:update":
            equity = payload.get("equity")
            if isinstance(equity, bool) or not isinstance(equity, (int, float)):
                return
            value = float(equity)
            if not math.isfinite(value):
                return
            if peak is None or value > peak:
                peak = value
            if peak > 0:
                decline = (peak - value) / peak * 100
                if decline >= threshold:
                    handle.add(
                        deliver(
                            "drawdown:breach",
                            {"equity": value, "peak": peak, "drawdownPct": decline},
                        )
                    )

    unsubscribe = bus.on_any(handler)
    handle = NotifierHandle(unsubscribe)
    return handle
