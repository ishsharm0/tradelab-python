"""Runtime state persistence and restart reconciliation contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

from tradelab.live import JsonFileStorage
from tradelab.live.state import StateManager


@pytest.mark.asyncio
async def test_state_manager_delegates_atomic_state_and_journals(tmp_path: Path) -> None:
    manager = StateManager(storage=JsonFileStorage(base_dir=tmp_path), now_ms=lambda: 123)
    await manager.save("sys", {"equity": 5_000})
    await manager.append_trade("sys", {"id": 1})
    await manager.append_equity_point("sys", {"time": 1, "equity": 5_000})
    assert await manager.load("sys") == {"equity": 5_000, "savedAt": 123}
    assert await manager.load_trades("sys") == [{"id": 1}]
    assert await manager.load_equity_curve("sys") == [{"time": 1, "equity": 5_000}]
    await manager.clear("sys")
    assert await manager.load("sys") is None


def test_reconcile_adopts_matching_snake_case_broker_position() -> None:
    manager = StateManager(storage=JsonFileStorage())
    report = manager.reconcile(
        persisted_state={
            "openPosition": {
                "symbol": "AAPL",
                "side": "long",
                "size": 10,
                "entry": 100,
                "entryFill": 100,
            }
        },
        broker_positions=[{"symbol": "AAPL", "side": "long", "qty": 10.2, "avg_entry": 100.1}],
        symbol="AAPL",
    )
    assert report["action"] == "adopt-broker"
    assert report["adoptedPosition"]["size"] == 10.2
    assert report["adoptedPosition"]["entryFill"] == 100.1


@pytest.mark.parametrize(
    ("persisted", "positions", "action", "status"),
    [
        (
            {"openPosition": {"symbol": "A", "side": "long", "size": 2}},
            [{"symbol": "A", "side": "short", "qty": 2}],
            "mismatch",
            "error",
        ),
        (
            {"openPosition": {"symbol": "A", "side": "long", "size": 2}},
            [],
            "closed-externally",
            "warn",
        ),
        ({}, [{"symbol": "A", "side": "long", "qty": 2}], "external-position", "warn"),
        ({}, [], "none", "ok"),
    ],
)
def test_reconcile_reports_every_restart_outcome(
    persisted: dict[str, object],
    positions: list[dict[str, object]],
    action: str,
    status: str,
) -> None:
    report = StateManager(storage=JsonFileStorage()).reconcile(
        persisted_state=persisted, broker_positions=positions, symbol="A"
    )
    assert (report["action"], report["status"]) == (action, status)
