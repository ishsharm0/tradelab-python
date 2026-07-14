"""Command-line workflows for research, backtesting, data, and reports."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Coroutine, Mapping
from pathlib import Path
from typing import Any, TypeVar

import typer

from tradelab.data import get_historical_candles, load_candles_from_csv, save_candles_to_cache
from tradelab.engine import backtest, grid, walk_forward_optimize
from tradelab.errors import TradeLabError
from tradelab.reporting import export_backtest_artifacts, export_metrics_json, summarize
from tradelab.strategies import get_strategy, list_strategies

VERSION = "1.3.1"
T = TypeVar("T")

app = typer.Typer(
    name="tradelab",
    help="Agent-native Python trading engine for research, backtesting, and live execution.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)


def _run(awaitable: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(awaitable)


def _json_mapping(value: str, name: str) -> dict[str, object]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise typer.BadParameter(f"Invalid JSON value for {name}: {value[:120]}") from error
    if not isinstance(parsed, dict):
        raise typer.BadParameter(f"{name} must be a JSON object")
    return parsed


def _fail(error: Exception) -> None:
    raise typer.BadParameter(str(error)) from error


def _paths(value: Mapping[str, object]) -> dict[str, str | None]:
    return {key: None if path is None else str(path) for key, path in value.items()}


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(VERSION)
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        callback=_version_callback,
        is_eager=True,
        help="Print the installed TradeLab version.",
    ),
) -> None:
    """Run a TradeLab command."""
    del version


@app.command("list-strategies")
def list_strategies_command() -> None:
    """List built-in strategy names and tunable parameters."""
    typer.echo(json.dumps({"strategies": list_strategies()}, indent=2, allow_nan=False))


async def _candles(
    *,
    source: str,
    symbol: str | None,
    interval: str,
    period: str,
    csv_path: Path | None,
    cache: bool,
) -> list[dict[str, int | float]]:
    return await get_historical_candles(
        source=source,
        symbol=symbol,
        interval=interval,
        period=period,
        csv_path=csv_path,
        cache=cache,
    )


@app.command("backtest")
def backtest_command(
    source: str = typer.Option("yahoo", help="Data source: yahoo, csv, or auto."),
    symbol: str | None = typer.Option(None, help="Instrument symbol."),
    interval: str = typer.Option("1d", help="Bar interval."),
    period: str = typer.Option("1y", help="Historical period."),
    csv_path: Path | None = typer.Option(None, exists=True, dir_okay=False),
    strategy: str = typer.Option("ema-cross", help="Registered strategy name."),
    params: str = typer.Option("{}", help="Strategy parameters as JSON."),
    out_dir: Path = typer.Option(Path("output"), help="Artifact directory."),
    cache: bool = typer.Option(True, "--cache/--no-cache"),
    equity: float = typer.Option(10_000, min=0),
    risk_pct: float = typer.Option(1, min=0),
    warmup_bars: int = typer.Option(20, min=0),
) -> None:
    """Run a backtest from Yahoo or CSV and export all report artifacts."""
    try:
        strategy_params = _json_mapping(params, "params")
        candles = _run(
            _candles(
                source=source,
                symbol=symbol,
                interval=interval,
                period=period,
                csv_path=csv_path,
                cache=cache,
            )
        )
        signal = get_strategy(strategy)(strategy_params)
        result = backtest(
            candles=candles,
            symbol=symbol or "DATA",
            interval=interval,
            range=period if source != "csv" else "custom",
            equity=equity,
            risk_pct=risk_pct,
            warmup_bars=warmup_bars,
            signal=signal,
        )
        outputs = export_backtest_artifacts(result, out_dir=out_dir)
        metrics = result["metrics"]
        typer.echo(
            json.dumps(
                {
                    "symbol": result["symbol"],
                    "trades": metrics["trades"],
                    "winRate": metrics["winRate"],
                    "profitFactor": metrics["profitFactor"],
                    "finalEquity": metrics["finalEquity"],
                    "outputs": _paths(outputs),
                },
                indent=2,
                allow_nan=False,
            )
        )
    except typer.BadParameter:
        raise
    except (TradeLabError, OSError, ValueError) as error:
        _fail(error)


@app.command("run")
def run_preset_command(
    preset: str = typer.Argument(..., help="Registered strategy name."),
    source: str = typer.Option("yahoo"),
    symbol: str | None = typer.Option(None),
    interval: str = typer.Option("1d"),
    period: str = typer.Option("1y"),
    csv_path: Path | None = typer.Option(None, exists=True, dir_okay=False),
    params: str = typer.Option("{}", help="Strategy parameters as JSON."),
    cache: bool = typer.Option(True, "--cache/--no-cache"),
) -> None:
    """Run a registered preset and print its plain-English summary."""
    try:
        candles = _run(
            _candles(
                source=source,
                symbol=symbol,
                interval=interval,
                period=period,
                csv_path=csv_path,
                cache=cache,
            )
        )
        signal = get_strategy(preset)(_json_mapping(params, "params"))
        result = backtest(
            candles=candles,
            symbol=symbol or "PRESET",
            interval=interval,
            signal=signal,
            warmup_bars=0,
        )
        metrics = result["metrics"]
        typer.echo(
            summarize(
                {
                    "trades": metrics["trades"],
                    "winRate": metrics["winRate"],
                    "totalReturnPct": metrics["returnPct"] * 100,
                    "maxDrawdownPct": metrics["maxDrawdown"] * 100,
                    "sharpe": metrics["sharpe"],
                }
            )
        )
    except typer.BadParameter:
        raise
    except (TradeLabError, OSError, ValueError) as error:
        _fail(error)


@app.command("import-csv")
def import_csv_command(
    csv_path: Path = typer.Argument(..., exists=True, dir_okay=False),
    symbol: str = typer.Option("DATA"),
    interval: str = typer.Option("1d"),
    period: str = typer.Option("custom"),
    out_dir: Path = typer.Option(Path("output/data")),
) -> None:
    """Normalize a CSV and save it to the candle cache."""
    try:
        candles = load_candles_from_csv(csv_path)
        output = save_candles_to_cache(
            candles,
            symbol=symbol,
            interval=interval,
            period=period,
            out_dir=out_dir,
            source="csv",
        )
        typer.echo(f"Saved {len(candles)} candles to {output}")
    except (TradeLabError, OSError, ValueError) as error:
        _fail(error)


@app.command()
def prefetch(
    symbol: str = typer.Option("SPY"),
    interval: str = typer.Option("1d"),
    period: str = typer.Option("1y"),
    out_dir: Path = typer.Option(Path("output/data")),
) -> None:
    """Fetch Yahoo candles and persist the normalized cache file."""
    try:
        candles = _run(
            get_historical_candles(
                source="yahoo",
                symbol=symbol,
                interval=interval,
                period=period,
                cache=False,
            )
        )
        output = save_candles_to_cache(
            candles,
            symbol=symbol,
            interval=interval,
            period=period,
            out_dir=out_dir,
            source="yahoo",
        )
        typer.echo(f"Saved {len(candles)} candles to {output}")
    except (TradeLabError, OSError, ValueError) as error:
        _fail(error)


@app.command("walk-forward")
def walk_forward_command(
    source: str = typer.Option("yahoo"),
    symbol: str | None = typer.Option(None),
    interval: str = typer.Option("1d"),
    period: str = typer.Option("2y"),
    csv_path: Path | None = typer.Option(None, exists=True, dir_okay=False),
    strategy: str = typer.Option("ema-cross"),
    parameter_grid: str = typer.Option(
        '{"fast":[8,10,12],"slow":[20,30,40],"rr":[1.5,2,3]}',
        "--grid",
        help="Cartesian parameter grid as JSON.",
    ),
    train_bars: int = typer.Option(120, min=1),
    test_bars: int = typer.Option(40, min=1),
    step_bars: int | None = typer.Option(None, min=1),
    mode: str = typer.Option("rolling"),
    score_by: str = typer.Option("profitFactor"),
    out_dir: Path = typer.Option(Path("output")),
    cache: bool = typer.Option(True, "--cache/--no-cache"),
) -> None:
    """Run rolling or anchored train/test optimization."""
    try:
        candles = _run(
            _candles(
                source=source,
                symbol=symbol,
                interval=interval,
                period=period,
                csv_path=csv_path,
                cache=cache,
            )
        )
        factory = get_strategy(strategy)
        parameter_sets = grid(_json_mapping(parameter_grid, "grid"))
        result = walk_forward_optimize(
            candles=candles,
            parameter_sets=parameter_sets,
            train_bars=train_bars,
            test_bars=test_bars,
            step_bars=step_bars or test_bars,
            mode=mode,
            score_by=score_by,
            signal_factory=lambda values: factory(values),
            backtest_options={"symbol": symbol or "DATA", "interval": interval},
        )
        metrics_path = export_metrics_json(
            result,
            symbol=symbol or "DATA",
            interval=interval,
            range_=f"{train_bars}-{test_bars}",
            out_dir=out_dir,
        )
        typer.echo(
            json.dumps(
                {
                    "windows": len(result["windows"]),
                    "positions": len(result["positions"]),
                    "finalEquity": result["metrics"]["finalEquity"],
                    "bestParamsSummary": result["bestParamsSummary"],
                    "metricsPath": str(metrics_path),
                },
                indent=2,
                allow_nan=False,
            )
        )
    except typer.BadParameter:
        raise
    except (TradeLabError, OSError, ValueError) as error:
        _fail(error)


def entrypoint() -> None:
    app()


__all__ = ["VERSION", "app", "entrypoint"]
