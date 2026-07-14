"""Portfolio live-orchestrator contracts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import pytest

from tradelab.errors import ValidationError
from tradelab.live import JsonFileStorage, PaperEngine
from tradelab.live.orchestrator import LiveOrchestrator


@pytest.mark.asyncio
async def test_orchestrator_starts_systems_allocates_weights_and_aggregates(tmp_path: Path) -> None:
    broker = PaperEngine(equity=20_000)
    systems = [
        {"id": "a", "symbol": "AAA", "signal": lambda _context: None, "weight": 1},
        {"id": "b", "symbol": "BBB", "signal": lambda _context: None, "weight": 3},
    ]
    orchestrator = LiveOrchestrator(
        systems=systems,
        broker=broker,
        storage=JsonFileStorage(base_dir=tmp_path),
        allocation="weighted",
    )
    await orchestrator.start()
    status = orchestrator.get_status()
    assert [row["realizedEquity"] for row in status["systems"]] == [5_000, 15_000]
    assert status["aggregateEquity"] == 20_000
    assert status["running"] is True
    await orchestrator.stop()
    await orchestrator.stop()
    assert orchestrator.running is False


@pytest.mark.asyncio
async def test_orchestrator_flattens_every_engine_before_shared_broker_disconnect(
    tmp_path: Path,
) -> None:
    broker = PaperEngine(equity=20_000)
    orchestrator = LiveOrchestrator(
        systems=[
            {"id": "a", "symbol": "AAA", "signal": lambda _context: None},
            {"id": "b", "symbol": "BBB", "signal": lambda _context: None},
        ],
        broker=broker,
        storage=JsonFileStorage(base_dir=tmp_path),
    )
    await orchestrator.start()
    for engine, symbol in zip(orchestrator.engines, ("AAA", "BBB"), strict=True):
        engine.open_position = {
            "id": f"trade-{symbol}",
            "symbol": symbol,
            "side": "long",
            "size": 1,
            "entry": 100,
            "entryFill": 100,
            "stop": 98,
            "takeProfit": 104,
            "openTime": 1,
        }
        broker.positions[symbol] = {
            "symbol": symbol,
            "side": "long",
            "qty": 1,
            "avgEntry": 100,
        }
        broker.last_prices[symbol] = 101

    await orchestrator.stop(flatten_on_shutdown=True)

    assert await broker.get_positions() == []
    assert broker.is_connected() is False
    assert orchestrator.running is False


@pytest.mark.asyncio
async def test_orchestrator_rolls_back_started_engines_when_later_start_fails() -> None:
    states: list[str] = []

    class FakeRisk:
        def halt(self, reason: str) -> None:
            states.append(reason)

    class FakeEngine:
        def __init__(self, *, id: str, fail: bool = False, **_options: object) -> None:
            self.id = id
            self.fail = fail
            self.risk_manager = FakeRisk()
            self.event_bus = type("Bus", (), {"on_any": lambda _self, _handler: lambda: None})()

        async def start(self) -> None:
            if self.fail:
                raise RuntimeError("start failed")
            states.append(f"start:{self.id}")

        async def stop(self) -> None:
            states.append(f"stop:{self.id}")

        def get_status(self) -> dict[str, Any]:
            return {"id": self.id, "equity": 100, "openPosition": None}

    orchestrator = LiveOrchestrator(
        systems=[
            {"id": "one", "symbol": "A", "signal": lambda _context: None},
            {"id": "two", "symbol": "B", "signal": lambda _context: None, "fail": True},
        ],
        broker=PaperEngine(),
        engine_factory=cast(Any, FakeEngine),
    )
    with pytest.raises(RuntimeError, match="start failed"):
        await orchestrator.start()
    assert states == ["start:one", "stop:one"]
    assert orchestrator.engines == []


def test_portfolio_daily_loss_halts_every_engine_once() -> None:
    orchestrator = LiveOrchestrator(
        systems=[{"id": "a", "symbol": "A", "signal": lambda _context: None}],
        broker=PaperEngine(),
        max_daily_loss_pct=1,
        now_ms=lambda: 1_735_828_200_000,
    )
    halted: list[str] = []

    class Risk:
        def halt(self, reason: str) -> None:
            halted.append(reason)

    class Engine:
        risk_manager = Risk()

        def get_status(self) -> dict[str, object]:
            return {"equity": 98, "openPosition": None}

    orchestrator.engines = [Engine()]  # type: ignore[list-item]
    orchestrator.current_day = "2025-01-02"
    orchestrator.day_start_equity = 100
    events: list[dict[str, Any]] = []
    orchestrator.event_bus.on("risk:halt", events.append)
    orchestrator.check_portfolio_limits()
    orchestrator.check_portfolio_limits()
    assert halted == ["portfolio daily loss limit reached"]
    assert len(events) == 1


def test_orchestrator_validates_systems_and_broker() -> None:
    with pytest.raises(ValidationError, match="systems"):
        LiveOrchestrator(systems=[], broker=PaperEngine())
    with pytest.raises(ValidationError, match="broker"):
        LiveOrchestrator(systems=[{"symbol": "A", "signal": lambda _context: None}], broker=None)


@pytest.mark.asyncio
async def test_orchestrator_propagates_explicit_live_confirmation() -> None:
    captured: list[dict[str, object]] = []

    class FakeEngine:
        def __init__(self, **options: object) -> None:
            captured.append(dict(options))
            self.event_bus = options["event_bus"]
            self.risk_manager = object()

        async def start(self) -> None:
            return None

        async def stop(self) -> None:
            return None

        def get_status(self) -> dict[str, object]:
            return {"equity": 10_000, "openPosition": None}

    orchestrator = LiveOrchestrator(
        systems=[{"symbol": "A", "signal": lambda _context: None}],
        broker=PaperEngine(),
        confirm_live=True,
        engine_factory=cast(Any, FakeEngine),
    )
    await orchestrator.start()
    assert captured[0]["confirm_live"] is True
    await orchestrator.stop()
