"""Injectable implementations for TradeLab's MCP tool surface."""

from __future__ import annotations

import asyncio
import inspect
import math
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from tradelab.errors import LiveTradingDisabledError

from .schemas import DESCRIPTIONS, SCHEMAS, TOOL_NAMES

ToolHandler = Callable[[Mapping[str, Any]], Awaitable[object]]
Dependency = Callable[..., object]

_METRIC_KEYS = (
    "trades",
    "winRate",
    "profitFactor",
    "expectancy",
    "totalR",
    "avgR",
    "sharpe",
    "sharpeAnnualized",
    "sortinoAnnualized",
    "maxDrawdown",
    "calmar",
    "returnPct",
    "totalPnL",
    "finalEquity",
    "exposurePct",
    "sideBreakdown",
)


@dataclass(slots=True)
class ToolDefinition:
    """One public MCP tool and its injectable async handler."""

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: ToolHandler


@dataclass(slots=True)
class McpDependencies:
    """Replaceable service boundary used by the MCP handlers."""

    get_historical_candles: Dependency
    candle_stats: Dependency
    list_strategies: Dependency
    get_strategy: Dependency
    backtest: Dependency
    walk_forward: Dependency
    expand_grid: Dependency
    monte_carlo: Dependency
    deflated_sharpe: Dependency
    session_manager: Any
    research_store: Any
    now_ms: Callable[[], int]


async def _maybe_await(value: object) -> object:
    if inspect.isawaitable(value):
        return await cast(Awaitable[object], value)
    return value


async def _call(function: Dependency, *args: object, **kwargs: object) -> object:
    return await _maybe_await(function(*args, **kwargs))


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return cast(Mapping[str, Any], value)


def _sequence(value: object, name: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be an array")
    return value


def _camelize_key(key: str) -> str:
    head, *tail = key.split("_")
    return head + "".join(part[:1].upper() + part[1:] for part in tail)


def _camelize(value: object) -> object:
    if isinstance(value, Mapping):
        return {_camelize_key(str(key)): _camelize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_camelize(item) for item in value]
    return value


def _summarize_metrics(value: object) -> dict[str, Any]:
    metrics = _mapping(value, "metrics")
    return {key: metrics.get(key) for key in _METRIC_KEYS}


async def _resolve_candles(args: Mapping[str, Any], deps: McpDependencies) -> list[Any]:
    inline = args.get("candles")
    if isinstance(inline, list) and inline:
        return inline
    data = args.get("data")
    if isinstance(data, Mapping):
        result = await _call(deps.get_historical_candles, data)
        return list(_sequence(result, "historical candles"))
    raise ValueError("Provide either `candles` (array) or `data` (getHistoricalCandles spec).")


def _result_mapping(value: object) -> Mapping[str, Any]:
    return _mapping(value, "backtest result")


def _score(metrics: Mapping[str, Any], score_by: str) -> tuple[float | None, float]:
    raw = metrics.get(score_by)
    if isinstance(raw, (int, float)) and not isinstance(raw, bool) and math.isfinite(raw):
        return float(raw), float(raw)
    return None, -math.inf


def _backtest_options(
    args: Mapping[str, Any], candles: Sequence[Any], signal: object, *, warmup: bool = False
) -> dict[str, Any]:
    options = {
        "candles": list(candles),
        "symbol": args.get("symbol", "UNKNOWN"),
        "interval": args.get("interval"),
        "signal": signal,
        "collectReplay": False,
    }
    if warmup:
        options["warmupBars"] = 0
    custom = args.get("backtestOptions")
    if isinstance(custom, Mapping):
        options.update(custom)
    return options


def _trade_preview(position: object) -> dict[str, Any]:
    item = _mapping(position, "position")
    exit_data = _mapping(item.get("exit"), "position exit")
    return {
        "side": item.get("side"),
        "entry": item.get("entryFill", item.get("entry")),
        "exit": exit_data.get("price"),
        "pnl": exit_data.get("pnl"),
        "reason": exit_data.get("reason"),
    }


def _session(deps: McpDependencies, session_id: object) -> Any:
    found = deps.session_manager.get(session_id)
    if found is None:
        raise ValueError(f'No session found with id "{session_id}"')
    return found


def _signal_order(signal: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "side": signal.get("side", signal.get("direction", signal.get("action"))),
        "type": signal.get("type", "market"),
        "qty": signal.get("qty", signal.get("size")),
        "risk_pct": signal.get("riskPct"),
        "stop": signal.get("stop", signal.get("stopLoss", signal.get("sl"))),
        "target": signal.get("target", signal.get("takeProfit", signal.get("tp"))),
        "rr": signal.get("rr", signal.get("_rr")),
        "limit_price": signal.get(
            "limitPrice", signal.get("limit", signal.get("entry", signal.get("price")))
        ),
    }


def default_dependencies(*, research_directory: str | Path | None = None) -> McpDependencies:
    """Create the production service graph without doing network or broker work."""
    from tradelab.data import candle_stats, get_historical_candles
    from tradelab.engine import backtest, grid, walk_forward_optimize
    from tradelab.live import SessionManager
    from tradelab.research import ResearchStore, deflated_sharpe, monte_carlo
    from tradelab.strategies import get_strategy, list_strategies

    async def load(spec: Mapping[str, Any]) -> object:
        options = dict(spec)
        if "csvPath" in options:
            options["csv_path"] = options.pop("csvPath")
        if "cacheDir" in options:
            options["cache_dir"] = options.pop("cacheDir")
        return await get_historical_candles(**options)

    def run_monte_carlo(options: Mapping[str, Any]) -> object:
        return monte_carlo(**options)

    def run_deflated_sharpe(options: Mapping[str, Any]) -> object:
        return deflated_sharpe(**options)

    store = ResearchStore(research_directory) if research_directory is not None else ResearchStore()
    return McpDependencies(
        get_historical_candles=load,
        candle_stats=candle_stats,
        list_strategies=list_strategies,
        get_strategy=get_strategy,
        backtest=backtest,
        walk_forward=walk_forward_optimize,
        expand_grid=grid,
        monte_carlo=run_monte_carlo,
        deflated_sharpe=run_deflated_sharpe,
        session_manager=SessionManager(),
        research_store=store,
        now_ms=lambda: time.time_ns() // 1_000_000,
    )


def build_tools(dependencies: McpDependencies | None = None) -> dict[str, ToolDefinition]:
    """Build all 25 tool definitions around one dependency graph."""
    deps = dependencies or default_dependencies()
    attached: dict[tuple[str, str], Callable[[Mapping[str, Any]], object]] = {}
    operation_lock = asyncio.Lock()
    halt_lock = asyncio.Lock()
    halt_generation = 0
    halting = False
    active_halts = 0
    halt_tasks: set[asyncio.Task[object]] = set()

    @asynccontextmanager
    async def live_operation() -> AsyncIterator[None]:
        generation = halt_generation
        if halting:
            raise LiveTradingDisabledError("live operation blocked by the kill switch")
        async with operation_lock:
            if halting or generation != halt_generation:
                raise LiveTradingDisabledError("live operation blocked by the kill switch")
            yield

    async def list_strategies_handler(_args: Mapping[str, Any]) -> object:
        return {"strategies": await _call(deps.list_strategies)}

    async def fetch_candles_handler(args: Mapping[str, Any]) -> object:
        values = list(
            _sequence(await _call(deps.get_historical_candles, args), "historical candles")
        )
        return {
            "count": len(values),
            "first": values[0] if values else None,
            "last": values[-1] if values else None,
        }

    async def run_backtest_handler(args: Mapping[str, Any]) -> object:
        candles = await _resolve_candles(args, deps)
        factory = await _call(deps.get_strategy, args.get("strategy"))
        signal = cast(Callable[[Mapping[str, Any]], object], factory)(
            cast(Mapping[str, Any], args.get("params") or {})
        )
        result = _result_mapping(
            await _call(deps.backtest, _backtest_options(args, candles, signal))
        )
        metrics = _summarize_metrics(result.get("metrics"))
        positions = list(_sequence(result.get("positions", []), "positions"))
        research_id = args.get("researchId")
        if research_id:
            verdict: dict[str, Any]
            try:
                probability = await _call(
                    deps.deflated_sharpe,
                    {
                        "sharpe": _mapping(result.get("metrics"), "metrics").get("sharpe"),
                        "sample_size": _mapping(result.get("metrics"), "metrics").get("trades"),
                        "num_trials": args.get("numTrials", 1),
                    },
                )
                finite = isinstance(probability, (int, float)) and math.isfinite(probability)
                verdict = {
                    "deflatedSharpe": probability if finite else None,
                    "overfit": bool(finite and cast(float, probability) < 0.9),
                    "note": (
                        f"PSR {cast(float, probability) * 100:.1f}%"
                        if finite
                        else "insufficient data"
                    ),
                }
            except Exception:
                verdict = {
                    "deflatedSharpe": None,
                    "overfit": False,
                    "note": "verdict unavailable",
                }
            with suppress(Exception):
                await _maybe_await(
                    deps.research_store.log(
                        research_id,
                        hypothesis=args.get("strategy"),
                        params=args.get("params") or {},
                        metrics=metrics,
                        verdict=verdict,
                    )
                )
        return {
            "symbol": result.get("symbol"),
            "interval": result.get("interval"),
            "metrics": metrics,
            "tradesPreview": [_trade_preview(item) for item in positions[:10]],
        }

    async def walk_forward_handler(args: Mapping[str, Any]) -> object:
        candles = await _resolve_candles(args, deps)
        factory = await _call(deps.get_strategy, args.get("strategy"))
        parameter_sets = await _call(deps.expand_grid, args.get("grid") or {})

        def signal_factory(params: Mapping[str, Any]) -> object:
            return cast(Callable[[Mapping[str, Any]], object], factory)(params)

        options = {
            "candles": candles,
            "mode": args.get("mode", "rolling"),
            "trainBars": args.get("trainBars"),
            "testBars": args.get("testBars"),
            "stepBars": args.get("stepBars", args.get("testBars")),
            "scoreBy": args.get("scoreBy", "profitFactor"),
            "parameterSets": parameter_sets,
            "signalFactory": signal_factory,
            "backtestOptions": {
                "interval": args.get("interval"),
                "collectReplay": False,
                **dict(cast(Mapping[str, Any], args.get("backtestOptions") or {})),
            },
        }
        value = _mapping(await _call(deps.walk_forward, options), "walk-forward result")
        windows = list(_sequence(value.get("windows", []), "walk-forward windows"))
        summaries = []
        for window in windows:
            item = _mapping(window, "walk-forward window")
            summaries.append(
                {
                    "bestParams": item.get("bestParams"),
                    "oosTrades": item.get("oosTrades"),
                    "profitable": item.get("profitable"),
                    "stabilityScore": item.get("stabilityScore"),
                }
            )
        return {
            "windows": len(windows),
            "metrics": _summarize_metrics(value.get("metrics")),
            "stability": value.get("bestParamsSummary"),
            "windowSummaries": summaries,
        }

    async def robustness_handler(args: Mapping[str, Any]) -> object:
        candles = await _resolve_candles(args, deps)
        factory = await _call(deps.get_strategy, args.get("strategy"))
        signal = cast(Callable[[Mapping[str, Any]], object], factory)(
            cast(Mapping[str, Any], args.get("params") or {})
        )
        result = _result_mapping(
            await _call(deps.backtest, _backtest_options(args, candles, signal, warmup=True))
        )
        metrics_map = _mapping(result.get("metrics"), "metrics")
        metrics = _summarize_metrics(metrics_map)
        positions = _sequence(result.get("positions", []), "positions")
        pnls = [
            _mapping(_mapping(item, "position").get("exit"), "position exit").get("pnl")
            for item in positions
        ]
        if len(pnls) < 2:
            return {
                "metrics": metrics,
                "monteCarlo": None,
                "deflatedSharpe": None,
                "note": f"Only {len(pnls)} trade(s), need at least 2 for statistical analysis.",
            }
        monte_carlo_result = await _call(
            deps.monte_carlo,
            {
                "trade_pnls": pnls,
                "equity_start": args.get("equityStart", 10_000),
                "iterations": args.get("iterations", 1000),
                "block_size": args.get("blockSize", 1),
                "seed": args.get("seed", "tradelab-mc"),
            },
        )
        deflated = await _call(
            deps.deflated_sharpe,
            {
                "sharpe": metrics_map.get("sharpe"),
                "sample_size": metrics_map.get("trades"),
                "num_trials": args.get("numTrials", 1),
                "sharpe_std": args.get("sharpeStd", 0),
                "skew": args.get("skew", 0),
                "kurtosis": args.get("kurtosis", 3),
            },
        )
        return {
            "metrics": metrics,
            "monteCarlo": _camelize(monte_carlo_result),
            "deflatedSharpe": deflated,
        }

    async def optimize_handler(args: Mapping[str, Any]) -> object:
        candles = await _resolve_candles(args, deps)
        factory = await _call(deps.get_strategy, args.get("strategy"))
        score_by = str(args.get("scoreBy", "profitFactor"))
        parameter_sets = _sequence(
            await _call(deps.expand_grid, args.get("grid") or {}), "parameter sets"
        )
        ranked: list[tuple[float, dict[str, Any]]] = []
        for raw_params in parameter_sets:
            params = _mapping(raw_params, "parameters")
            signal = cast(Callable[[Mapping[str, Any]], object], factory)(params)
            result = _result_mapping(
                await _call(deps.backtest, _backtest_options(args, candles, signal))
            )
            metrics = _mapping(result.get("metrics"), "metrics")
            public_score, sort_score = _score(metrics, score_by)
            ranked.append(
                (
                    sort_score,
                    {
                        "params": dict(params),
                        "score": public_score,
                        "metrics": _summarize_metrics(metrics),
                    },
                )
            )
        ranked.sort(key=lambda row: row[0], reverse=True)
        rows = [row for _, row in ranked]
        return {"leaderboard": rows, "best": rows[0] if rows else None}

    async def compare_handler(args: Mapping[str, Any]) -> object:
        candles = await _resolve_candles(args, deps)
        score_by = str(args.get("scoreBy", "profitFactor"))
        entries = _sequence(args.get("strategies", []), "strategies")
        ranked: list[tuple[float, dict[str, Any]]] = []
        for raw_entry in entries:
            entry = _mapping(raw_entry, "strategy entry")
            strategy = str(entry.get("strategy"))
            params = cast(Mapping[str, Any], entry.get("params") or {})
            factory = await _call(deps.get_strategy, strategy)
            signal = cast(Callable[[Mapping[str, Any]], object], factory)(params)
            result = _result_mapping(
                await _call(deps.backtest, _backtest_options(args, candles, signal))
            )
            metrics = _mapping(result.get("metrics"), "metrics")
            public_score, sort_score = _score(metrics, score_by)
            ranked.append(
                (
                    sort_score,
                    {
                        "strategy": strategy,
                        "params": dict(params),
                        "score": public_score,
                        "metrics": _summarize_metrics(metrics),
                    },
                )
            )
        ranked.sort(key=lambda row: row[0], reverse=True)
        return {"rankedBy": score_by, "results": [row for _, row in ranked]}

    async def candle_stats_handler(args: Mapping[str, Any]) -> object:
        candles = await _resolve_candles(args, deps)
        value = await _call(deps.candle_stats, candles)
        if value is None:
            return {"stats": None, "note": "No candles returned."}
        stats = _mapping(value, "candle stats")
        interval = stats.get("estimatedIntervalMin", 0)
        note = (
            f"Estimated bar interval ~{interval} min."
            if isinstance(interval, (int, float)) and interval > 0
            else "Could not estimate interval."
        )
        return {"stats": value, "note": note}

    async def create_session_handler(args: Mapping[str, Any]) -> object:
        async with live_operation():
            options: dict[str, Any] = {
                "id": args.get("sessionId"),
                "symbol": args.get("symbol"),
                "symbols": args.get("symbols"),
                "mode": args.get("mode", "paper"),
                "interval": args.get("interval", "1m"),
                "equity": args.get("equity", 10_000),
                "confirm_live": args.get("confirmLive", False),
            }
            if args.get("riskPct") is not None:
                options["risk_pct"] = args["riskPct"]
            if args.get("maxDailyLossPct") is not None:
                options["max_daily_loss_pct"] = args["maxDailyLossPct"]
            session = await _maybe_await(deps.session_manager.create(**options))
            if halting:
                raise LiveTradingDisabledError("session creation interrupted by the kill switch")
            session_id = str(cast(Any, session).id)
            for key in [key for key in attached if key[0] == session_id]:
                attached.pop(key, None)
            return cast(Any, session).get_status()

    async def list_sessions_handler(_args: Mapping[str, Any]) -> object:
        return [session.get_status() for session in deps.session_manager.list()]

    async def session_status_handler(args: Mapping[str, Any]) -> object:
        return await _maybe_await(_session(deps, args.get("sessionId")).refresh())

    async def feed_price_handler(args: Mapping[str, Any]) -> object:
        async with live_operation():
            session = _session(deps, args.get("sessionId"))
            bar = args.get("bar")
            price = args.get("price")
            if (
                bar is None
                and isinstance(price, (int, float))
                and not isinstance(price, bool)
                and math.isfinite(price)
            ):
                bar = {
                    "time": deps.now_ms(),
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "volume": 0,
                }
            if bar is None:
                raise ValueError("Provide either `bar` (OHLCV) or `price` (number)")
            symbol = args.get("symbol")
            await _maybe_await(session.push_bar(_mapping(bar, "bar"), symbol))
            effective_symbol = str(symbol or session.symbol)
            strategy_fn = attached.get((str(session.id), effective_symbol))
            status = _mapping(session.get_status(), "session status")
            positions = list(_sequence(status.get("positions", []), "positions"))
            symbol_positions = [
                item
                for item in positions
                if not _mapping(item, "position").get("symbol")
                or _mapping(item, "position").get("symbol") == effective_symbol
            ]
            if strategy_fn is not None and not symbol_positions:
                candles = session.candle_buffer_for(effective_symbol)
                context = {
                    "candles": candles,
                    "index": len(candles) - 1,
                    "bar": candles[-1] if candles else None,
                    "equity": status.get("equity"),
                    "openPosition": None,
                    "pendingOrder": None,
                }
                raw_signal = strategy_fn(context)
                if inspect.isawaitable(raw_signal):
                    raw_signal = await cast(Awaitable[object], raw_signal)
                if halting:
                    raise LiveTradingDisabledError("strategy order blocked by the kill switch")
                if isinstance(raw_signal, Mapping) and any(
                    raw_signal.get(key) for key in ("side", "direction", "action")
                ):
                    order = _signal_order(cast(Mapping[str, Any], raw_signal))
                    order["symbol"] = effective_symbol
                    await _maybe_await(session.place_order(**order))
            return session.get_status()

    async def place_order_handler(args: Mapping[str, Any]) -> object:
        async with live_operation():
            session = _session(deps, args.get("sessionId"))
            return await _maybe_await(
                session.place_order(
                    side=args.get("side"),
                    type=args.get("type", "market"),
                    qty=args.get("qty"),
                    risk_pct=args.get("riskPct"),
                    stop=args.get("stop"),
                    target=args.get("target"),
                    rr=args.get("rr"),
                    limit_price=args.get("limitPrice"),
                    symbol=args.get("symbol"),
                )
            )

    async def close_position_handler(args: Mapping[str, Any]) -> object:
        async with live_operation():
            return await _maybe_await(
                _session(deps, args.get("sessionId")).close_position(args.get("symbol"))
            )

    async def flatten_handler(args: Mapping[str, Any]) -> object:
        async with live_operation():
            await _maybe_await(_session(deps, args.get("sessionId")).flatten())
            return {"ok": True}

    async def cancel_order_handler(args: Mapping[str, Any]) -> object:
        async with live_operation():
            await _maybe_await(
                _session(deps, args.get("sessionId")).cancel_order(args.get("orderId"))
            )
            return {"ok": True}

    async def account_handler(args: Mapping[str, Any]) -> object:
        return await _maybe_await(_session(deps, args.get("sessionId")).get_account())

    async def positions_handler(args: Mapping[str, Any]) -> object:
        return await _maybe_await(_session(deps, args.get("sessionId")).get_positions())

    async def recent_events_handler(args: Mapping[str, Any]) -> object:
        return await _maybe_await(
            _session(deps, args.get("sessionId")).recent_events(args.get("limit", 50))
        )

    async def attach_strategy_handler(args: Mapping[str, Any]) -> object:
        async with live_operation():
            session = _session(deps, args.get("sessionId"))
            factory = await _call(deps.get_strategy, args.get("strategy"))
            if halting:
                raise LiveTradingDisabledError("strategy attachment blocked by the kill switch")
            params = cast(Mapping[str, Any], args.get("params") or {})
            signal = cast(Callable[[Mapping[str, Any]], object], factory)(params)
            symbol = str(args.get("symbol") or session.symbol)
            attached[(str(session.id), symbol)] = cast(
                Callable[[Mapping[str, Any]], object], signal
            )
            return {"ok": True, "strategy": args.get("strategy"), "params": dict(params)}

    async def halt_all_handler(_args: Mapping[str, Any]) -> object:
        nonlocal active_halts, halt_generation, halting

        async def finish_halt() -> object:
            nonlocal active_halts, halting
            try:
                async with operation_lock:
                    await _maybe_await(deps.session_manager.halt_all())
                return {"ok": True, "sessionsHalted": len(deps.session_manager.list())}
            finally:
                attached.clear()
                active_halts -= 1
                halting = active_halts > 0

        async with halt_lock:
            halt_generation += 1
            halting = True
            active_halts += 1
            attached.clear()
            task = asyncio.create_task(finish_halt())
            halt_tasks.add(task)
            task.add_done_callback(halt_tasks.discard)
        return await asyncio.shield(task)

    async def research_open_handler(args: Mapping[str, Any]) -> object:
        return _camelize(
            await _maybe_await(deps.research_store.open(args.get("id"), args.get("goal", "")))
        )

    async def research_log_handler(args: Mapping[str, Any]) -> object:
        return _camelize(
            await _maybe_await(
                deps.research_store.log(
                    args.get("id"),
                    hypothesis=args.get("hypothesis", ""),
                    params=args.get("params"),
                    metrics=args.get("metrics"),
                    verdict=args.get("verdict"),
                )
            )
        )

    async def research_recall_handler(args: Mapping[str, Any]) -> object:
        return _camelize(
            await _maybe_await(deps.research_store.recall(args.get("id"), args.get("limit", 10)))
        )

    async def research_close_handler(args: Mapping[str, Any]) -> object:
        return _camelize(await _maybe_await(deps.research_store.close(args.get("id"))))

    handlers: dict[str, ToolHandler] = {
        "list_strategies": list_strategies_handler,
        "fetch_candles": fetch_candles_handler,
        "run_backtest": run_backtest_handler,
        "walk_forward": walk_forward_handler,
        "analyze_robustness": robustness_handler,
        "optimize_strategy": optimize_handler,
        "compare_strategies": compare_handler,
        "candle_stats": candle_stats_handler,
        "create_session": create_session_handler,
        "list_sessions": list_sessions_handler,
        "session_status": session_status_handler,
        "feed_price": feed_price_handler,
        "place_order": place_order_handler,
        "close_position": close_position_handler,
        "flatten": flatten_handler,
        "cancel_order": cancel_order_handler,
        "account": account_handler,
        "positions": positions_handler,
        "recent_events": recent_events_handler,
        "attach_strategy": attach_strategy_handler,
        "halt_all": halt_all_handler,
        "research_open": research_open_handler,
        "research_log": research_log_handler,
        "research_recall": research_recall_handler,
        "research_close": research_close_handler,
    }
    return {
        name: ToolDefinition(name, DESCRIPTIONS[name], SCHEMAS[name], handlers[name])
        for name in TOOL_NAMES
    }


__all__ = [
    "McpDependencies",
    "ToolDefinition",
    "build_tools",
    "default_dependencies",
]
