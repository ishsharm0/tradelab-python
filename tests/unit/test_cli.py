"""Typer CLI end-to-end contracts for local, credential-free workflows."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

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


def test_live_command_fails_closed_for_rest_only_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRADELAB_ALLOW_LIVE", "true")
    result = RUNNER.invoke(
        app,
        ["live", "--broker", "alpaca", "--symbol", "AAPL", "--confirm-live"],
    )
    assert result.exit_code != 0
    assert "streaming order updates" in result.output
