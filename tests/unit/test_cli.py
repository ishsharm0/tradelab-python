"""Typer CLI end-to-end contracts for local, credential-free workflows."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from tradelab.cli import app

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
