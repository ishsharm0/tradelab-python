"""Clock, structured logging, and notifier runtime contracts."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from io import StringIO
from typing import Any

import pytest

import tradelab.live as live
from tradelab.live import EventBus
from tradelab.live.clock import BrokerClock
from tradelab.live.logger import LiveLogger
from tradelab.live.notify import attach_notifier


class ClockBroker:
    def __init__(self, server_time: object = 12_500, *, fails: bool = False) -> None:
        self.server_time = server_time
        self.fails = fails

    async def get_server_time(self) -> object:
        if self.fails:
            raise RuntimeError("clock unavailable")
        return self.server_time


@pytest.mark.asyncio
async def test_broker_clock_syncs_warns_and_falls_back_without_mutating_time_source() -> None:
    times = iter([10_000, 10_000, 11_000, 11_000])
    clock = BrokerClock(warn_threshold_ms=2_000, now_ms=lambda: next(times))
    report = await clock.sync_with_broker(ClockBroker())
    assert report == {
        "serverTime": 12_500.0,
        "localTime": 10_000,
        "offsetMs": 2_500.0,
        "warning": "clock offset 2500ms exceeds threshold 2000ms",
    }
    assert clock.now() == 12_500
    failed = await clock.sync_with_broker(ClockBroker(fails=True))
    assert failed["serverTime"] is None
    assert failed["offsetMs"] == 0
    assert clock.now() == 11_000


def test_logger_filters_json_lines_maps_bus_levels_and_detaches() -> None:
    stream = StringIO()
    logger = LiveLogger(
        level="warn",
        stream=stream,
        now=lambda: datetime(2025, 1, 2, tzinfo=UTC),
    )
    logger.info("hidden")
    logger.warn("visible", {"symbol": "A"})
    bus = EventBus()
    detach = logger.attach(bus)
    bus.emit_event("risk:halt", {"reason": "limit"})
    bus.emit_event("error", {"message": "boom"})
    detach()
    detach()
    bus.emit_event("risk:halt", {"reason": "ignored"})
    rows = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert [row["level"] for row in rows] == ["warn", "warn", "error"]
    assert rows[0]["symbol"] == "A"
    assert rows[1]["event"] == "risk:halt"


@pytest.mark.asyncio
async def test_notifier_delivers_selected_events_drawdown_and_swallows_failures() -> None:
    bus = EventBus()
    source = type("Source", (), {"event_bus": bus})()
    seen: list[dict[str, Any]] = []
    posted: list[tuple[str, Mapping[str, object]]] = []

    async def on_event(envelope: dict[str, Any]) -> None:
        seen.append(envelope)
        if envelope["event"] == "risk:halt":
            raise RuntimeError("non-fatal callback")

    async def webhook(url: str, envelope: Mapping[str, object]) -> None:
        posted.append((url, envelope))
        if envelope["event"] == "risk:halt":
            raise RuntimeError("non-fatal webhook")

    handle = attach_notifier(
        source,
        on_event=on_event,
        webhook_url="https://example.test/hook",
        webhook_client=webhook,
        drawdown_pct=5,
    )
    bus.emit_event("equity:update", {"equity": 100})
    bus.emit_event("equity:update", {"equity": 94})
    bus.emit_event("risk:halt", {"reason": "limit"})
    await handle.drain()
    assert [item["event"] for item in seen] == ["drawdown:breach", "risk:halt"]
    assert len(posted) == 2
    handle()
    handle()
    bus.emit_event("risk:halt", {})
    await asyncio.sleep(0)
    assert len(seen) == 2


def test_live_package_exports_runtime_factories() -> None:
    for name in (
        "BrokerClock",
        "LiveLogger",
        "PollingFeed",
        "BrokerFeed",
        "CandleAggregator",
        "StateManager",
        "LiveEngine",
        "LiveOrchestrator",
        "DashboardServer",
        "attach_notifier",
        "create_dashboard_server",
    ):
        assert hasattr(live, name), name


@pytest.mark.asyncio
async def test_clock_and_notifier_validate_finite_configuration() -> None:
    with pytest.raises(Exception, match="warn_threshold_ms"):
        BrokerClock(warn_threshold_ms=float("nan"))
    clock = BrokerClock(now_ms=lambda: 7)
    assert (await clock.sync_with_broker(None))["localTime"] == 7
    with pytest.raises(Exception, match="event_bus"):
        attach_notifier(object())  # type: ignore[arg-type]
    source = type("Source", (), {"event_bus": EventBus()})()
    with pytest.raises(Exception, match="drawdown_pct"):
        attach_notifier(source, drawdown_pct=-1)


def test_logger_normalizes_unknown_level_and_silent_suppresses() -> None:
    stream = StringIO()
    logger = LiveLogger(level="unknown", stream=stream)
    logger.debug("hidden")
    logger.error("shown")
    logger.level = "silent"
    logger.error("hidden too")
    assert len(stream.getvalue().splitlines()) == 1
    assert LiveLogger().attach(None)() is None
