"""Typer CLI end-to-end contracts for local, credential-free workflows."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import tradelab.cli as cli
from tradelab.cli import app
from tradelab.live import JsonFileStorage

RUNNER = CliRunner()


def _csv(path: Path) -> Path:
    path.write_text(
        "time,open,high,low,close,volume\n"
        "2025-01-02T14:30:00Z,100,101,99,100,10\n"
        "2025-01-02T14:31:00Z,100,102,99,101,11\n"
        "2025-01-02T14:32:00Z,101,103,100,102,12\n",
        encoding="utf-8",
    )
    return path


def test_cli_help_and_version() -> None:
    result = RUNNER.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == "1.3.1"
    help_result = RUNNER.invoke(app, ["--help"])
    assert help_result.exit_code == 0
    assert "backtest" in help_result.stdout
    assert "walk-forward" in help_result.stdout


def test_import_csv_and_run_preset(tmp_path: Path) -> None:
    csv_path = _csv(tmp_path / "bars.csv")
    out_dir = tmp_path / "cache"
    imported = RUNNER.invoke(
        app,
        [
            "import-csv",
            str(csv_path),
            "--symbol",
            "TEST",
            "--interval",
            "1m",
            "--out-dir",
            str(out_dir),
        ],
    )
    assert imported.exit_code == 0, imported.output
    assert "Saved 3 candles" in imported.stdout
    assert (out_dir / "candles-TEST-1m-custom.json").exists()

    run = RUNNER.invoke(
        app,
        [
            "run",
            "buy-hold",
            "--source",
            "csv",
            "--csv-path",
            str(csv_path),
            "--params",
            '{"holdBars": 1}',
        ],
    )
    assert run.exit_code == 0, run.output
    assert "trade" in run.stdout.lower()


def test_backtest_csv_exports_artifacts_and_valid_json(tmp_path: Path) -> None:
    csv_path = _csv(tmp_path / "bars.csv")
    out_dir = tmp_path / "reports"
    result = RUNNER.invoke(
        app,
        [
            "backtest",
            "--source",
            "csv",
            "--csv-path",
            str(csv_path),
            "--symbol",
            "TEST",
            "--interval",
            "1m",
            "--strategy",
            "buy-hold",
            "--params",
            '{"holdBars": 1}',
            "--out-dir",
            str(out_dir),
            "--warmup-bars",
            "0",
            "--no-cache",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["symbol"] == "TEST"
    assert set(payload["outputs"]) == {"csv", "html", "markdown", "metrics"}
    assert Path(payload["outputs"]["html"]).exists()


def test_cli_rejects_bad_json_and_unknown_preset(tmp_path: Path) -> None:
    csv_path = _csv(tmp_path / "bars.csv")
    malformed = RUNNER.invoke(
        app,
        ["run", "buy-hold", "--source", "csv", "--csv-path", str(csv_path), "--params", "{"],
    )
    assert malformed.exit_code != 0
    assert "Invalid JSON" in malformed.output
    unknown = RUNNER.invoke(
        app,
        ["run", "not-real", "--source", "csv", "--csv-path", str(csv_path)],
    )
    assert unknown.exit_code != 0
    assert "Unknown strategy" in unknown.output


def test_cli_loads_a_python_strategy_module(tmp_path: Path) -> None:
    csv_path = _csv(tmp_path / "bars.csv")
    strategy_path = tmp_path / "custom_strategy.py"
    strategy_path.write_text(
        "def signal(context):\n"
        "    if context['index'] == 0:\n"
        "        close = context['bar']['close']\n"
        "        return {'side': 'long', 'qty': 1, 'stop': close - 2, 'rr': 2}\n"
        "    return None\n",
        encoding="utf-8",
    )
    result = RUNNER.invoke(
        app,
        [
            "backtest",
            "--source",
            "csv",
            "--csv-path",
            str(csv_path),
            "--strategy",
            str(strategy_path),
            "--warmup-bars",
            "0",
            "--out-dir",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["symbol"] == "DATA"


def test_portfolio_command_runs_multiple_csv_systems(tmp_path: Path) -> None:
    first = _csv(tmp_path / "first.csv")
    second = _csv(tmp_path / "second.csv")
    result = RUNNER.invoke(
        app,
        [
            "portfolio",
            "--csv-paths",
            f"{first},{second}",
            "--symbols",
            "ONE,TWO",
            "--params",
            '{"holdBars": 1}',
            "--out-dir",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["systems"] == 2
    assert Path(payload["metricsPath"]).exists()


def test_status_reads_one_persisted_namespace(tmp_path: Path) -> None:
    storage = JsonFileStorage(base_dir=tmp_path)
    asyncio.run(storage.save("paper-one", {"equity": 12_345, "savedAt": "now"}))
    asyncio.run(storage.append_trade("paper-one", {"pnl": 5}))
    result = RUNNER.invoke(app, ["status", "--dir", str(tmp_path), "--namespace", "paper-one"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["state"]["equity"] == 12_345
    assert payload["trades"] == 1


def test_paper_command_replays_csv_without_credentials(tmp_path: Path) -> None:
    csv_path = _csv(tmp_path / "paper.csv")
    result = RUNNER.invoke(
        app,
        [
            "paper",
            "--symbol",
            "TEST",
            "--interval",
            "1m",
            "--csv-path",
            str(csv_path),
            "--strategy",
            "buy-hold",
            "--warmup-bars",
            "0",
            "--state-dir",
            str(tmp_path / "state"),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["symbol"] == "TEST"
    assert payload["barsProcessed"] == 3
    assert payload["mode"] == "paper"


def test_paper_config_resolves_multiple_registered_and_local_strategies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    strategy_path = tmp_path / "local_strategy.py"
    strategy_path.write_text(
        "def create_signal(params):\n"
        "    marker = params['marker']\n"
        "    return lambda context: {'marker': marker, 'context': context}\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "paper.json"
    config_path.write_text(
        json.dumps(
            {
                "allocation": "weighted",
                "equity": 25_000,
                "maxDailyLossPct": 3,
                "systems": [
                    {
                        "id": "registered",
                        "symbol": "ONE",
                        "strategy": "buy-hold",
                        "params": {"holdBars": 2},
                        "weight": 1,
                    },
                    {
                        "id": "local",
                        "symbol": "TWO",
                        "strategy": "./local_strategy.py",
                        "params": {"marker": "loaded"},
                        "weight": 2,
                        "warmupBars": 7,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    captured: dict[str, Any] = {}
    events: list[str] = []

    class FakeOrchestrator:
        def __init__(self, **options: object) -> None:
            captured.update(options)

        async def start(self) -> None:
            events.append("start")

        async def stop(self) -> None:
            events.append("stop")

        def get_status(self) -> dict[str, object]:
            return {"running": True, "systems": [{"symbol": "ONE"}, {"symbol": "TWO"}]}

    monkeypatch.setattr(cli, "LiveOrchestrator", FakeOrchestrator)
    result = RUNNER.invoke(
        app,
        ["paper", "--config", str(config_path), "--state-dir", str(tmp_path / "state")],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["mode"] == "paper"
    assert len(payload["systems"]) == 2
    assert captured["allocation"] == "weighted"
    assert captured["equity"] == 25_000
    assert captured["max_daily_loss_pct"] == 3
    broker = captured["broker"]
    assert asyncio.run(broker.get_account())["equity"] == 25_000
    systems = captured["systems"]
    assert isinstance(systems, list)
    assert [system["id"] for system in systems] == ["registered", "local"]
    assert all(callable(system["signal"]) for system in systems)
    assert systems[1]["warmup_bars"] == 7
    assert systems[1]["signal"]({})["marker"] == "loaded"
    assert all("strategy" not in system and "params" not in system for system in systems)
    assert events == ["start", "stop"]


def test_paper_config_rejects_duplicate_symbols_that_share_broker_order_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _csv(tmp_path / "first.csv")

    class FakeOrchestrator:
        def __init__(self, **_options: object) -> None:
            pass

        async def start(self) -> None:
            return None

        async def stop(self) -> None:
            return None

        def get_status(self) -> dict[str, object]:
            return {"systems": []}

    monkeypatch.setattr(cli, "LiveOrchestrator", FakeOrchestrator)
    config_path = tmp_path / "duplicate-symbol.json"
    config_path.write_text(
        json.dumps(
            {
                "systems": [
                    {"id": "one", "symbol": "SPY", "csvPath": first.name},
                    {"id": "two", "symbol": "spy", "interval": "5m"},
                ]
            }
        ),
        encoding="utf-8",
    )
    result = RUNNER.invoke(app, ["paper", "--config", str(config_path)])
    assert result.exit_code != 0
    assert "duplicate symbol" in result.output


def test_paper_dashboard_and_watch_own_resource_lifecycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[Any] = []

    class FakeEngine:
        def __init__(self, **_options: object) -> None:
            events.append("engine:init")

        async def start(self) -> None:
            events.append("engine:start")

        async def stop(self) -> None:
            events.append("engine:stop")

        def get_status(self) -> dict[str, object]:
            return {"symbol": "SPY"}

    class FakeDashboard:
        command_token = "test-dashboard-token"

        async def start(self) -> str:
            events.append("dashboard:start")
            return "http://127.0.0.1:5432"

        async def close(self) -> None:
            events.append("dashboard:close")

    def dashboard_factory(**options: object) -> FakeDashboard:
        events.append(("dashboard:init", options["source"], options["port"]))
        return FakeDashboard()

    async def finish_watch() -> None:
        events.append("watch")

    monkeypatch.setattr(cli, "LiveEngine", FakeEngine)
    monkeypatch.setattr(cli, "create_dashboard_server", dashboard_factory)
    monkeypatch.setattr(cli, "_wait_until_cancelled", finish_watch)
    result = RUNNER.invoke(
        app,
        [
            "paper",
            "--dashboard",
            "--dashboard-port",
            "5432",
            "--watch",
            "--state-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "dashboard: http://127.0.0.1:5432" in result.stderr
    assert "dashboard token: test-dashboard-token" in result.stderr
    assert events[0:2] == ["engine:init", "engine:start"]
    assert events[2][0] == "dashboard:init"
    assert events[2][1].__class__ is FakeEngine
    assert events[2][2] == 5432
    assert events[3:] == ["dashboard:start", "watch", "dashboard:close", "engine:stop"]


def test_paper_closes_dashboard_and_stops_engine_when_dashboard_start_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []

    class FakeEngine:
        def __init__(self, **_options: object) -> None:
            pass

        async def start(self) -> None:
            events.append("engine:start")

        async def stop(self) -> None:
            events.append("engine:stop")

        def get_status(self) -> dict[str, object]:
            return {}

    class BrokenDashboard:
        async def start(self) -> str:
            events.append("dashboard:start")
            raise ValueError("dashboard failed")

        async def close(self) -> None:
            events.append("dashboard:close")

    monkeypatch.setattr(cli, "LiveEngine", FakeEngine)
    monkeypatch.setattr(cli, "create_dashboard_server", lambda **_options: BrokenDashboard())
    result = RUNNER.invoke(
        app,
        ["paper", "--dashboard", "--state-dir", str(tmp_path)],
    )

    assert result.exit_code != 0
    assert "dashboard failed" in result.output
    assert events == ["engine:start", "dashboard:start", "dashboard:close", "engine:stop"]


def test_live_help_exposes_config_dashboard_and_watch_options() -> None:
    result = RUNNER.invoke(app, ["live", "--help"])
    assert result.exit_code == 0, result.output
    assert "--config" in result.stdout
    assert "--dashboard" in result.stdout
    assert "--dashboard-port" in result.stdout
    assert "--watch" in result.stdout


def test_live_requires_watch_after_all_permission_gates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CertifiedBroker:
        def supports_order_updates(self) -> bool:
            return True

    monkeypatch.setenv("TRADELAB_ALLOW_LIVE", "true")
    monkeypatch.setattr(cli, "_live_broker", lambda _name: CertifiedBroker())
    result = RUNNER.invoke(
        app,
        ["live", "--broker", "certified", "--symbol", "AAPL", "--confirm-live"],
    )

    assert result.exit_code != 0
    assert "--watch" in result.output


def test_live_watch_cancels_pending_entry_and_delegates_exit_flatten_to_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []

    class CertifiedBroker:
        def supports_order_updates(self) -> bool:
            return True

        async def cancel_order(self, order_id: str) -> None:
            events.append(("cancel", order_id))

    broker = CertifiedBroker()

    class FakeEngine:
        def __init__(self, **options: object) -> None:
            self.broker = options["broker"]
            self.pending_exit_order_id = "exit-1"

        async def start(self) -> None:
            events.append("start")

        async def stop(self, *, flatten_on_shutdown: bool = False) -> None:
            events.append(("stop", flatten_on_shutdown, self.pending_exit_order_id))

        def get_status(self) -> dict[str, object]:
            return {
                "symbol": "AAPL",
                "openPosition": {
                    "side": "long",
                    "pendingExit": {"orderId": "exit-1", "clientOrderId": "client-exit-1"},
                },
                "pendingOrder": {"orderId": "pending-1"},
            }

    async def finish_watch() -> None:
        events.append("watch")

    monkeypatch.setenv("TRADELAB_ALLOW_LIVE", "true")
    monkeypatch.setattr(cli, "_live_broker", lambda _name: broker)
    monkeypatch.setattr(cli, "LiveEngine", FakeEngine)
    monkeypatch.setattr(cli, "_wait_until_cancelled", finish_watch, raising=False)
    result = RUNNER.invoke(
        app,
        [
            "live",
            "--broker",
            "certified",
            "--symbol",
            "AAPL",
            "--confirm-live",
            "--watch",
        ],
    )

    assert result.exit_code == 0, result.output
    assert events == [
        "start",
        "watch",
        ("cancel", "pending-1"),
        ("stop", True, "exit-1"),
    ]


def test_live_cleanup_surfaces_cancel_failure_after_attempting_flatten(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[object] = []

    class CertifiedBroker:
        def supports_order_updates(self) -> bool:
            return True

        async def cancel_order(self, order_id: str) -> None:
            events.append(("cancel", order_id))
            raise RuntimeError("cancel failed")

    broker = CertifiedBroker()

    class FakeEngine:
        def __init__(self, **_options: object) -> None:
            self.broker = broker

        async def start(self) -> None:
            events.append("start")

        async def stop(self, *, flatten_on_shutdown: bool = False) -> None:
            events.append(("stop", flatten_on_shutdown))

        def get_status(self) -> dict[str, object]:
            return {"pendingOrder": {"orderId": "pending-1"}}

    async def finish_watch() -> None:
        return None

    monkeypatch.setenv("TRADELAB_ALLOW_LIVE", "true")
    monkeypatch.setattr(cli, "_live_broker", lambda _name: broker)
    monkeypatch.setattr(cli, "LiveEngine", FakeEngine)
    monkeypatch.setattr(cli, "_wait_until_cancelled", finish_watch)
    result = RUNNER.invoke(
        app,
        [
            "live",
            "--broker",
            "certified",
            "--symbol",
            "AAPL",
            "--confirm-live",
            "--watch",
        ],
    )

    assert result.exit_code != 0
    assert isinstance(result.exception, RuntimeError)
    assert str(result.exception) == "cancel failed"
    assert events == ["start", ("cancel", "pending-1"), ("stop", True)]


def test_live_cleanup_still_flattens_when_status_snapshot_fails() -> None:
    events: list[object] = []

    class BrokenStatusEngine:
        def get_status(self) -> dict[str, object]:
            raise RuntimeError("status failed")

        async def stop(self, *, flatten_on_shutdown: bool = False) -> None:
            events.append(("stop", flatten_on_shutdown))

    with pytest.raises(RuntimeError, match="status failed"):
        asyncio.run(cli._stop_live_runtime(BrokenStatusEngine()))

    assert events == [("stop", True)]


def test_live_config_builds_orchestrator_and_safely_stops_every_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "live.json"
    config_path.write_text(
        json.dumps(
            {
                "allocation": "weighted",
                "systems": [
                    {"id": "one", "symbol": "ONE", "strategy": "buy-hold", "weight": 1},
                    {"id": "two", "symbol": "TWO", "strategy": "ema-cross", "weight": 2},
                ],
            }
        ),
        encoding="utf-8",
    )
    events: list[object] = []
    captured: dict[str, Any] = {}

    class CertifiedBroker:
        def supports_order_updates(self) -> bool:
            return True

        async def cancel_order(self, order_id: str) -> None:
            events.append(("cancel", order_id))

    broker = CertifiedBroker()

    class FakeEngine:
        def __init__(self, system_id: str) -> None:
            self.system_id = system_id
            self.broker = broker

        async def stop(self, *, flatten_on_shutdown: bool = False) -> None:
            events.append(("engine:stop", self.system_id, flatten_on_shutdown))

        def get_status(self) -> dict[str, object]:
            return {"pendingOrder": {"orderId": f"pending-{self.system_id}"}}

    class FakeOrchestrator:
        def __init__(self, **options: object) -> None:
            captured.update(options)
            systems = options["systems"]
            assert isinstance(systems, list)
            self.engines = [FakeEngine(str(system["id"])) for system in systems]

        async def start(self) -> None:
            events.append("orchestrator:start")

        async def stop(self, *, flatten_on_shutdown: bool = False) -> None:
            events.append(("orchestrator:stop", flatten_on_shutdown))
            for engine in reversed(self.engines):
                await engine.stop(flatten_on_shutdown=flatten_on_shutdown)

        def get_status(self) -> dict[str, object]:
            return {"systems": [{"id": "one"}, {"id": "two"}]}

    async def finish_watch() -> None:
        events.append("watch")

    monkeypatch.setenv("TRADELAB_ALLOW_LIVE", "true")
    monkeypatch.setattr(cli, "_live_broker", lambda _name: broker)
    monkeypatch.setattr(cli, "LiveOrchestrator", FakeOrchestrator)
    monkeypatch.setattr(cli, "_wait_until_cancelled", finish_watch)
    result = RUNNER.invoke(
        app,
        [
            "live",
            "--broker",
            "certified",
            "--config",
            str(config_path),
            "--confirm-live",
            "--watch",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["confirm_live"] is True
    assert captured["allocation"] == "weighted"
    systems = captured["systems"]
    assert isinstance(systems, list)
    assert all(callable(system["signal"]) for system in systems)
    assert events == [
        "orchestrator:start",
        "watch",
        ("cancel", "pending-two"),
        ("cancel", "pending-one"),
        ("orchestrator:stop", True),
        ("engine:stop", "two", True),
        ("engine:stop", "one", True),
    ]


def test_live_config_rejects_historical_csv_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv_path = _csv(tmp_path / "bars.csv")
    config_path = tmp_path / "live.json"
    config_path.write_text(
        json.dumps(
            {"systems": [{"symbol": "AAPL", "strategy": "buy-hold", "csvPath": csv_path.name}]}
        ),
        encoding="utf-8",
    )

    class CertifiedBroker:
        def supports_order_updates(self) -> bool:
            return True

    monkeypatch.setenv("TRADELAB_ALLOW_LIVE", "true")
    monkeypatch.setattr(cli, "_live_broker", lambda _name: CertifiedBroker())
    result = RUNNER.invoke(
        app,
        [
            "live",
            "--broker",
            "certified",
            "--config",
            str(config_path),
            "--confirm-live",
            "--watch",
        ],
    )

    assert result.exit_code != 0
    assert "csvPath is paper-only" in result.output


def test_live_command_fails_closed_for_rest_only_adapter(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "live.json"
    config_path.write_text(
        json.dumps({"systems": [{"symbol": "AAPL", "strategy": "buy-hold"}]}),
        encoding="utf-8",
    )
    monkeypatch.setenv("TRADELAB_ALLOW_LIVE", "true")
    result = RUNNER.invoke(
        app,
        [
            "live",
            "--broker",
            "alpaca",
            "--config",
            str(config_path),
            "--confirm-live",
        ],
    )
    assert result.exit_code != 0
    assert "streaming order updates" in result.output
