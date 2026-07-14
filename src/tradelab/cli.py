"""Command-line workflows for research, backtesting, data, and reports."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
from collections.abc import Callable, Coroutine, Mapping
from pathlib import Path
from types import ModuleType
from typing import Any, TypeVar, cast

import typer

from tradelab.brokers import (
    AlpacaBroker,
    BinanceBroker,
    CoinbaseBroker,
    InteractiveBrokersBroker,
)
from tradelab.data import get_historical_candles, load_candles_from_csv, save_candles_to_cache
from tradelab.engine import backtest, backtest_portfolio, grid, walk_forward_optimize
from tradelab.errors import LiveTradingDisabledError, TradeLabError, ValidationError
from tradelab.live import (
    JsonFileStorage,
    LiveEngine,
    LiveOrchestrator,
    PaperEngine,
    create_dashboard_server,
)
from tradelab.reporting import export_backtest_artifacts, export_metrics_json, summarize
from tradelab.strategies import get_strategy, list_strategies

VERSION = "1.3.1"
T = TypeVar("T")
Strategy = Callable[[Mapping[str, Any]], object]
StrategyFactory = Callable[[Mapping[str, object]], Strategy]

_SYSTEM_OPTION_ALIASES = {
    "pollIntervalMs": "poll_interval_ms",
    "warmupBars": "warmup_bars",
    "riskPct": "risk_pct",
    "finalTpR": "final_tp_r",
    "flattenAtClose": "flatten_at_close",
    "qtyStep": "qty_step",
    "minQty": "min_qty",
    "maxLeverage": "max_leverage",
    "dailyMaxTrades": "daily_max_trades",
    "maxDailyLossPct": "max_daily_loss_pct",
    "entryChase": "entry_chase",
}
_SYSTEM_OPTIONS = {
    "id",
    "symbol",
    "interval",
    "mode",
    "weight",
    "poll_interval_ms",
    "warmup_bars",
    "risk_pct",
    "final_tp_r",
    "flatten_at_close",
    "qty_step",
    "min_qty",
    "max_leverage",
    "daily_max_trades",
    "max_daily_loss_pct",
    "entry_chase",
    "oco",
    "risk",
}

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


def _load_strategy_factory(value: str) -> StrategyFactory:
    """Resolve a registered strategy or an explicit local Python module."""
    path = Path(value).expanduser()
    if path.suffix != ".py" and not path.exists():
        return cast(StrategyFactory, get_strategy(value))
    if not path.is_file():
        raise ValidationError(f'strategy module "{path}" does not exist or is not a file')
    resolved = path.resolve()
    spec = importlib.util.spec_from_file_location(
        f"tradelab_user_strategy_{abs(hash(resolved))}", resolved
    )
    if spec is None or spec.loader is None:
        raise ValidationError(f'cannot load strategy module "{resolved}"')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return _factory_from_module(module, resolved)


def _factory_from_module(module: ModuleType, path: Path) -> StrategyFactory:
    factory = getattr(module, "create_signal", None)
    if callable(factory):
        return cast(StrategyFactory, factory)
    signal = getattr(module, "signal", None)
    if callable(signal):
        strategy = cast(Strategy, signal)
        return lambda _parameters: strategy
    raise ValidationError(
        f'strategy module "{path}" must define create_signal(params) or signal(context)'
    )


def _live_broker(name: str) -> object:
    normalized = name.strip().lower()
    factories: dict[str, Callable[[], object]] = {
        "alpaca": AlpacaBroker,
        "binance": BinanceBroker,
        "coinbase": CoinbaseBroker,
        "ib": InteractiveBrokersBroker,
        "interactive-brokers": InteractiveBrokersBroker,
    }
    factory = factories.get(normalized)
    if factory is None:
        raise ValidationError(f'unsupported broker "{name}"')
    return factory()


def _broker_config(name: str) -> dict[str, object]:
    prefix = name.strip().upper().replace("-", "_")
    return {
        "api_key": os.environ.get(f"TRADELAB_{prefix}_API_KEY", ""),
        "api_secret": os.environ.get(f"TRADELAB_{prefix}_API_SECRET", ""),
        "passphrase": os.environ.get(f"TRADELAB_{prefix}_PASSPHRASE", ""),
        "host": os.environ.get("TRADELAB_IB_HOST", "127.0.0.1"),
        "port": int(os.environ.get("TRADELAB_IB_PORT", "7497")),
        "client_id": int(os.environ.get("TRADELAB_IB_CLIENT_ID", "1")),
        "paper": False,
    }


def _load_live_config(path: Path) -> dict[str, object]:
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValidationError(f'config "{path}" is not valid JSON') from error
    if not isinstance(parsed, dict):
        raise ValidationError("config must be a JSON object")
    return cast(dict[str, object], parsed)


def _config_strategy(value: object, *, base_dir: Path, fallback: str) -> str:
    selected = fallback if value is None else value
    if not isinstance(selected, str) or not selected.strip():
        raise ValidationError("each configured system strategy must be a non-empty string")
    path = Path(selected).expanduser()
    if path.suffix == ".py" and not path.is_absolute():
        return str((base_dir / path).resolve())
    return selected


def _config_systems(
    config: Mapping[str, object],
    *,
    base_dir: Path,
    strategy: str,
    params: Mapping[str, object],
    defaults: Mapping[str, object],
) -> tuple[list[dict[str, object]], list[tuple[str, str, Path]]]:
    raw_systems = config.get("systems")
    if not isinstance(raw_systems, list) or not raw_systems:
        raise ValidationError("config requires a non-empty systems array")
    systems: list[dict[str, object]] = []
    replays: list[tuple[str, str, Path]] = []
    replay_sources: dict[tuple[str, str], Path] = {}
    configured_symbols: set[str] = set()
    for index, raw in enumerate(raw_systems):
        if not isinstance(raw, Mapping):
            raise ValidationError(f"config system {index + 1} must be a JSON object")
        raw_params = raw.get("params", params)
        if not isinstance(raw_params, Mapping):
            raise ValidationError(f"config system {index + 1} params must be a JSON object")
        selected_strategy = _config_strategy(
            raw.get("strategy"), base_dir=base_dir, fallback=strategy
        )
        system: dict[str, object] = {}
        for key, value in raw.items():
            if not isinstance(key, str):
                raise ValidationError(f"config system {index + 1} keys must be strings")
            canonical = _SYSTEM_OPTION_ALIASES.get(key, key)
            if canonical in _SYSTEM_OPTIONS:
                system[canonical] = value
        for key, value in defaults.items():
            system.setdefault(key, value)
        symbol = system.get("symbol")
        interval = system.get("interval", "1m")
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValidationError(f"config system {index + 1} requires symbol")
        if not isinstance(interval, str) or not interval.strip():
            raise ValidationError(f"config system {index + 1} requires interval")
        normalized_symbol = symbol.strip().upper()
        if normalized_symbol in configured_symbols:
            raise ValidationError(
                f'config contains duplicate symbol "{symbol.strip()}"; '
                "shared broker order events require one system per symbol"
            )
        configured_symbols.add(normalized_symbol)
        system["symbol"] = symbol.strip()
        system["interval"] = interval.strip()
        system["signal"] = _load_strategy_factory(selected_strategy)(dict(raw_params))
        systems.append(system)
        csv_value = raw.get("csvPath", raw.get("csv_path"))
        if csv_value is not None:
            if not isinstance(csv_value, str) or not csv_value.strip():
                raise ValidationError(f"config system {index + 1} csvPath must be a path string")
            csv_path = Path(csv_value).expanduser()
            if not csv_path.is_absolute():
                csv_path = base_dir / csv_path
            if not csv_path.is_file():
                raise ValidationError(f'configured CSV "{csv_path}" does not exist')
            market = (symbol.strip(), interval.strip())
            existing = replay_sources.get(market)
            if existing is not None and existing.resolve() != csv_path.resolve():
                raise ValidationError(
                    f"conflicting CSV paths configured for {market[0]} {market[1]}"
                )
            if existing is None:
                replay_sources[market] = csv_path
                replays.append((*market, csv_path))
    return systems, replays


def _orchestrator_options(
    config: Mapping[str, object], *, fallback_equity: float
) -> tuple[str, object, object]:
    allocation = config.get("allocation", "equal")
    if allocation == "weight":
        allocation = "weighted"
    if not isinstance(allocation, str):
        raise ValidationError("config allocation must be equal or weighted")
    return (
        allocation,
        config.get("equity", fallback_equity),
        config.get("maxDailyLossPct", config.get("max_daily_loss_pct", 0)),
    )


async def _wait_until_cancelled() -> None:
    await asyncio.Event().wait()


async def _stop_live_runtime(runtime: object) -> None:
    raw_engines = getattr(runtime, "engines", None)
    engines = list(raw_engines) if isinstance(raw_engines, list) else [runtime]
    failure: BaseException | None = None
    for engine in reversed(engines):
        getter = getattr(engine, "get_status", None)
        try:
            status = getter() if callable(getter) else {}
        except BaseException as error:
            failure = failure or error
            status = {}
        pending = status.get("pendingOrder") if isinstance(status, Mapping) else None
        order_id = pending.get("orderId") if isinstance(pending, Mapping) else None
        broker = getattr(engine, "broker", None)
        cancel = getattr(broker, "cancel_order", None)
        if order_id and callable(cancel):
            try:
                await cancel(str(order_id))
            except BaseException as error:
                failure = failure or error
    stop_runtime = getattr(runtime, "stop", None)
    if callable(stop_runtime):
        try:
            await stop_runtime(flatten_on_shutdown=True)
        except BaseException as error:
            failure = failure or error
    if failure is not None:
        raise failure


async def _run_owned_runtime(
    runtime: object,
    *,
    dashboard: bool,
    dashboard_port: int,
    watch: bool,
    work: Callable[[], Coroutine[Any, Any, None]] | None = None,
    live_cleanup: bool = False,
) -> dict[str, object]:
    dashboard_server: object | None = None
    try:
        await cast(Any, runtime).start()
        if dashboard:
            dashboard_server = create_dashboard_server(source=runtime, port=dashboard_port)
            url = await cast(Any, dashboard_server).start()
            typer.echo(f"dashboard: {url}", err=True)
            token = getattr(dashboard_server, "command_token", None)
            if isinstance(token, str) and token:
                typer.echo(f"dashboard token: {token}", err=True)
        if work is not None:
            await work()
        if watch:
            await _wait_until_cancelled()
        status = cast(Any, runtime).get_status()
        return dict(status) if isinstance(status, Mapping) else {}
    finally:
        try:
            if dashboard_server is not None:
                await cast(Any, dashboard_server).close()
        finally:
            if live_cleanup:
                await _stop_live_runtime(runtime)
            else:
                await cast(Any, runtime).stop()


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
        signal = _load_strategy_factory(strategy)(strategy_params)
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
        signal = _load_strategy_factory(preset)(_json_mapping(params, "params"))
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
        factory = _load_strategy_factory(strategy)
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


@app.command()
def portfolio(
    csv_paths: str = typer.Option(..., help="Comma-separated CSV paths."),
    symbols: str = typer.Option("", help="Comma-separated symbols."),
    strategy: str = typer.Option("buy-hold"),
    params: str = typer.Option("{}", help="Strategy parameters as JSON."),
    interval: str = typer.Option("mixed"),
    equity: float = typer.Option(10_000, min=0),
    out_dir: Path = typer.Option(Path("output")),
) -> None:
    """Run several CSV datasets against one shared-capital portfolio."""
    try:
        paths = [Path(value.strip()) for value in csv_paths.split(",") if value.strip()]
        if not paths:
            raise ValidationError("portfolio requires at least one CSV path")
        names = [value.strip() for value in symbols.split(",") if value.strip()]
        strategy_params = _json_mapping(params, "params")
        factory = _load_strategy_factory(strategy)
        systems = [
            {
                "symbol": names[index] if index < len(names) else f"asset-{index + 1}",
                "candles": load_candles_from_csv(path),
                "signal": factory(strategy_params),
                "warmup_bars": 0,
                "flatten_at_close": False,
            }
            for index, path in enumerate(paths)
        ]
        result = backtest_portfolio(
            systems=systems,
            equity=equity,
            interval=interval,
            collect_replay=False,
            collect_eq_series=True,
        )
        metrics_path = export_metrics_json(
            result,
            symbol="PORTFOLIO",
            interval=interval,
            range_="custom",
            out_dir=out_dir,
        )
        typer.echo(
            json.dumps(
                {
                    "systems": len(result["systems"]),
                    "positions": len(result["positions"]),
                    "finalEquity": result["metrics"]["finalEquity"],
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


@app.command()
def paper(
    symbol: str = typer.Option("SPY"),
    interval: str = typer.Option("1m"),
    strategy: str = typer.Option("buy-hold"),
    params: str = typer.Option("{}", help="Strategy parameters as JSON."),
    csv_path: Path | None = typer.Option(None, exists=True, dir_okay=False),
    config: Path | None = typer.Option(None, exists=True, dir_okay=False),
    state_dir: Path = typer.Option(Path("output/live-state")),
    equity: float = typer.Option(10_000, min=0),
    risk_pct: float = typer.Option(1, min=0),
    warmup_bars: int = typer.Option(0, min=0),
    dashboard: bool = typer.Option(False, "--dashboard", help="Start a loopback dashboard."),
    dashboard_port: int = typer.Option(4_317, "--dashboard-port", min=0, max=65_535),
    watch: bool = typer.Option(False, "--watch", help="Run until interrupted."),
) -> None:
    """Run one or more credential-free paper systems with optional CSV replay."""

    async def run() -> dict[str, object]:
        strategy_params = _json_mapping(params, "params")
        if config is not None:
            file_config = _load_live_config(config)
            systems, replays = _config_systems(
                file_config,
                base_dir=config.parent,
                strategy=strategy,
                params=strategy_params,
                defaults={
                    "symbol": symbol,
                    "interval": interval,
                    "warmup_bars": warmup_bars,
                    "risk_pct": risk_pct,
                },
            )
            allocation, configured_equity, max_daily_loss_pct = _orchestrator_options(
                file_config, fallback_equity=equity
            )
            broker = PaperEngine(equity=configured_equity)
            runtime: object = LiveOrchestrator(
                systems=systems,
                broker=broker,
                storage=JsonFileStorage(base_dir=state_dir),
                allocation=allocation,
                equity=configured_equity,
                max_daily_loss_pct=max_daily_loss_pct,
            )
            bars_processed = 0

            async def replay_config() -> None:
                nonlocal bars_processed
                for replay_symbol, replay_interval, replay_path in replays:
                    replay_bars = load_candles_from_csv(replay_path)
                    bars_processed += len(replay_bars)
                    for bar in replay_bars:
                        await broker.simulate_bar(replay_symbol, replay_interval, bar)

            status = await _run_owned_runtime(
                runtime,
                dashboard=dashboard,
                dashboard_port=dashboard_port,
                watch=watch,
                work=replay_config,
            )
            return {
                "mode": "paper",
                "barsProcessed": bars_processed,
                **status,
            }

        broker = PaperEngine(equity=equity)
        signal = _load_strategy_factory(strategy)(strategy_params)
        engine = LiveEngine(
            id=f"paper-{symbol}-{interval}",
            symbol=symbol,
            interval=interval,
            signal=signal,
            broker=broker,
            storage=JsonFileStorage(base_dir=state_dir),
            equity=equity,
            risk_pct=risk_pct,
            warmup_bars=warmup_bars,
        )
        bars = load_candles_from_csv(csv_path) if csv_path is not None else []

        async def replay() -> None:
            for bar in bars:
                await broker.simulate_bar(symbol, interval, bar)

        status = await _run_owned_runtime(
            engine,
            dashboard=dashboard,
            dashboard_port=dashboard_port,
            watch=watch,
            work=replay,
        )
        return {"mode": "paper", "barsProcessed": len(bars), **status}

    try:
        typer.echo(json.dumps(_run(run()), indent=2, allow_nan=False))
    except typer.BadParameter:
        raise
    except (TradeLabError, OSError, ValueError) as error:
        _fail(error)


@app.command()
def live(
    broker: str = typer.Option(..., help="Broker: alpaca, binance, coinbase, or ib."),
    symbol: str | None = typer.Option(None),
    interval: str = typer.Option("1m"),
    strategy: str = typer.Option("ema-cross"),
    params: str = typer.Option("{}", help="Strategy parameters as JSON."),
    config: Path | None = typer.Option(None, exists=True, dir_okay=False),
    state_dir: Path = typer.Option(Path("output/live-state")),
    confirm_live: bool = typer.Option(False, "--confirm-live"),
    warmup_bars: int = typer.Option(200, min=0),
    watch: bool = typer.Option(False, "--watch", help="Run until interrupted."),
    dashboard: bool = typer.Option(False, "--dashboard", help="Start a loopback dashboard."),
    dashboard_port: int = typer.Option(4_317, "--dashboard-port", min=0, max=65_535),
) -> None:
    """Start a permission-gated live engine; uncertified adapters fail closed."""
    try:
        if os.environ.get("TRADELAB_ALLOW_LIVE") != "true" or confirm_live is not True:
            raise LiveTradingDisabledError(
                "live requires TRADELAB_ALLOW_LIVE=true and --confirm-live"
            )
        adapter = _live_broker(broker)
        supports_updates = getattr(adapter, "supports_order_updates", None)
        if not callable(supports_updates) or supports_updates() is not True:
            raise LiveTradingDisabledError(
                f"{broker} lacks certified streaming order updates; live execution is disabled"
            )
        if watch is not True:
            raise LiveTradingDisabledError(
                "live requires --watch so submitted orders and positions remain managed"
            )
        strategy_params = _json_mapping(params, "params")

        async def run() -> dict[str, object]:
            if config is not None:
                file_config = _load_live_config(config)
                systems, replays = _config_systems(
                    file_config,
                    base_dir=config.parent,
                    strategy=strategy,
                    params=strategy_params,
                    defaults={"interval": interval, "warmup_bars": warmup_bars},
                )
                if replays:
                    raise ValidationError("csvPath is paper-only and cannot be used by live")
                allocation, configured_equity, max_daily_loss_pct = _orchestrator_options(
                    file_config, fallback_equity=10_000
                )
                runtime: object = LiveOrchestrator(
                    systems=systems,
                    broker=adapter,
                    storage=JsonFileStorage(base_dir=state_dir),
                    broker_config=_broker_config(broker),
                    confirm_live=True,
                    allocation=allocation,
                    equity=configured_equity,
                    max_daily_loss_pct=max_daily_loss_pct,
                )
            else:
                if symbol is None or not symbol.strip():
                    raise ValidationError("live requires --symbol when --config is not provided")
                signal = _load_strategy_factory(strategy)(strategy_params)
                runtime = LiveEngine(
                    id=f"live-{broker}-{symbol}-{interval}",
                    symbol=symbol,
                    interval=interval,
                    signal=signal,
                    broker=adapter,
                    broker_config=_broker_config(broker),
                    storage=JsonFileStorage(base_dir=state_dir),
                    warmup_bars=warmup_bars,
                    confirm_live=True,
                )
            status = await _run_owned_runtime(
                runtime,
                dashboard=dashboard,
                dashboard_port=dashboard_port,
                watch=True,
                live_cleanup=True,
            )
            return {"mode": "live", **status}

        typer.echo(json.dumps(_run(run()), indent=2, allow_nan=False))
    except typer.BadParameter:
        raise
    except (TradeLabError, OSError, ValueError) as error:
        _fail(error)


@app.command()
def status(
    state_dir: Path = typer.Option(Path("output/live-state"), "--dir", "--state-dir"),
    namespace: str | None = typer.Option(None, "--namespace", "--id"),
) -> None:
    """Read persisted live state without starting a broker or feed."""
    try:
        storage = JsonFileStorage(base_dir=state_dir)
        if namespace:
            state = _run(storage.load(namespace))
            trades = _run(storage.load_trades(namespace))
            equity = _run(storage.load_equity_curve(namespace))
            payload = {
                "namespace": namespace,
                "state": state,
                "trades": len(trades),
                "equityPoints": len(equity),
            }
        elif not state_dir.exists():
            payload = {"dir": str(state_dir), "namespaces": []}
        else:
            summaries: list[dict[str, object]] = []
            for directory in sorted(path for path in state_dir.iterdir() if path.is_dir()):
                state = _run(storage.load(directory.name))
                trades = _run(storage.load_trades(directory.name))
                state_mapping = state if isinstance(state, Mapping) else {}
                summaries.append(
                    {
                        "namespace": directory.name,
                        "savedAt": state_mapping.get("savedAt"),
                        "equity": state_mapping.get("equity"),
                        "openPosition": bool(state_mapping.get("openPosition")),
                        "trades": len(trades),
                    }
                )
            payload = {"dir": str(state_dir), "namespaces": summaries}
        typer.echo(json.dumps(payload, indent=2, allow_nan=False))
    except (TradeLabError, OSError, ValueError) as error:
        _fail(error)


def entrypoint() -> None:
    app()


__all__ = ["VERSION", "app", "entrypoint"]
