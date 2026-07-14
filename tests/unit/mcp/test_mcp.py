"""MCP catalog, handler, serialization, and SDK server contracts."""

# ruff: noqa: E501 -- expected descriptions are exact immutable protocol strings.

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest
from mcp.server import Server
from mcp.types import (
    CallToolRequest,
    CallToolRequestParams,
    CallToolResult,
    ListToolsRequest,
    TextContent,
)

import tradelab.mcp.server as mcp_server
from tradelab.errors import LiveTradingDisabledError
from tradelab.mcp import (
    DESCRIPTIONS,
    SCHEMAS,
    TOOL_NAMES,
    McpDependencies,
    build_tools,
    create_server,
    default_dependencies,
    invoke_tool,
    run_stdio_server,
)

EXPECTED_NAMES = (
    "list_strategies",
    "fetch_candles",
    "run_backtest",
    "walk_forward",
    "analyze_robustness",
    "optimize_strategy",
    "compare_strategies",
    "candle_stats",
    "create_session",
    "list_sessions",
    "session_status",
    "feed_price",
    "place_order",
    "close_position",
    "flatten",
    "cancel_order",
    "account",
    "positions",
    "recent_events",
    "attach_strategy",
    "halt_all",
    "research_open",
    "research_log",
    "research_recall",
    "research_close",
)


def test_mcp_entrypoint_help_and_version_do_not_start_stdio(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(mcp_server.sys, "argv", ["tradelab-mcp", "--help"])
    mcp_server.entrypoint()
    help_output = capsys.readouterr().out
    assert "Usage: tradelab-mcp" in help_output
    assert "Model Context Protocol" in help_output

    monkeypatch.setattr(mcp_server.sys, "argv", ["tradelab-mcp", "--version"])
    mcp_server.entrypoint()
    assert capsys.readouterr().out.strip() == "1.3.1"


EXPECTED_DESCRIPTIONS = {
    "list_strategies": "List built-in trading strategies with their tunable parameters.",
    "fetch_candles": "Download/caches OHLCV candles from Yahoo or CSV. Returns a compact summary.",
    "run_backtest": "Run a single backtest using a named strategy + params. Returns a metrics summary and a small trade preview (no replay).",
    "walk_forward": "Walk-forward optimize a named strategy over a parameter grid. Returns out-of-sample metrics and winner stability.",
    "analyze_robustness": "Run a backtest on a named strategy then Monte Carlo + Deflated Sharpe on the realized trade PnLs. Degrades gracefully if fewer than 2 trades.",
    "optimize_strategy": "In-process grid sweep of a named strategy. Returns a leaderboard ranked by a chosen metric (default: profitFactor).",
    "compare_strategies": "Run several named strategies on the same candle dataset and return a ranked comparison.",
    "candle_stats": "Return shape statistics (count, date range, price range, estimated interval) for an inline candle array or a data spec. Useful for sanity-checking data before backtesting.",
    "create_session": "Create a new paper (default) or live (gated) trading session. Paper needs no credentials.",
    "list_sessions": "List all active trading sessions and their current status.",
    "session_status": "Get a full refreshed status snapshot for a session (positions, orders, equity, risk).",
    "feed_price": "Feed a price bar (or single price) to a session, advancing paper simulations and triggering fills.",
    "place_order": "Place a market or limit order in a session (optionally risk-sized with bracket stop/target).",
    "close_position": "Close the open position for a symbol in a session via an opposite market order.",
    "flatten": "Flatten all positions and cancel all open orders in a session.",
    "cancel_order": "Cancel a specific open order in a session.",
    "account": "Get the broker account details for a session (equity, cash, buying power).",
    "positions": "Get all open positions for a session.",
    "recent_events": "Get recent session events (fills, risk changes, bars) for monitoring.",
    "attach_strategy": "Attach a named built-in strategy to a session. It will auto-evaluate on each feed_price and place orders when flat.",
    "halt_all": "Emergency kill switch: flatten all positions and stop all trading sessions.",
    "research_open": "Open or resume a persistent research session for iterating on strategy hypotheses.",
    "research_log": "Append a tested hypothesis (params, metrics, optional overfitting verdict) to a research session.",
    "research_recall": "Recall recent research entries plus a synthesized summary (best Sharpe, overfit count).",
    "research_close": "Mark a research session complete and return its final record.",
}


def _metrics(score: float = 2.0) -> dict[str, Any]:
    return {
        "trades": 2,
        "winRate": 0.5,
        "profitFactor": score,
        "expectancy": 1,
        "totalR": 2,
        "avgR": 1,
        "sharpe": 1.1,
        "sharpeAnnualized": 1.4,
        "sortinoAnnualized": 1.8,
        "maxDrawdown": 0.1,
        "calmar": 1.5,
        "returnPct": 0.2,
        "totalPnL": 20,
        "finalEquity": 10_020,
        "exposurePct": 0.3,
        "sideBreakdown": {},
    }


def _result(score: float = 2.0, trade_count: int = 2) -> dict[str, Any]:
    positions = [
        {
            "side": "long",
            "entry": 100,
            "entryFill": 101,
            "exit": {"price": 102, "pnl": index + 1, "reason": "target"},
        }
        for index in range(trade_count)
    ]
    return {
        "symbol": "TEST",
        "interval": "1d",
        "metrics": _metrics(score),
        "positions": positions,
    }


class FakeStore:
    def __init__(self) -> None:
        self.entries: list[dict[str, Any]] = []

    def open(self, record_id: str, goal: str = "") -> dict[str, Any]:
        return {"id": record_id, "goal": goal, "created_at": "now", "entries": []}

    def log(self, record_id: str, **entry: Any) -> dict[str, Any]:
        value = {"id": record_id, **entry}
        self.entries.append(value)
        return value

    def recall(self, record_id: str, limit: int = 10) -> dict[str, Any]:
        return {"goal": record_id, "entries": self.entries[-limit:], "best_sharpe": None}

    def close(self, record_id: str) -> dict[str, Any]:
        return {"id": record_id, "closed_at": "later", "entries": self.entries}


class FakeSession:
    def __init__(self, session_id: str = "demo") -> None:
        self.id = session_id
        self.symbol = "AAPL"
        self.symbols = ["AAPL"]
        self.bars: list[dict[str, Any]] = []
        self.orders: list[dict[str, Any]] = []
        self.order_error: Exception | None = None
        self.canceled: list[str] = []
        self.flattened = False

    def get_status(self) -> dict[str, Any]:
        return {"id": self.id, "symbol": self.symbol, "equity": 10_000, "positions": []}

    async def refresh(self) -> dict[str, Any]:
        return self.get_status()

    async def push_bar(self, bar: Mapping[str, Any], symbol: str | None = None) -> None:
        self.bars.append({**bar, "symbol": symbol})

    def candle_buffer_for(self, symbol: str | None = None) -> list[dict[str, Any]]:
        return [dict(bar) for bar in self.bars]

    async def place_order(self, **order: Any) -> dict[str, Any]:
        if self.order_error is not None:
            raise self.order_error
        self.orders.append(order)
        return {"status": "filled", **order}

    async def close_position(self, symbol: str | None = None) -> dict[str, Any]:
        return {"closed": symbol or self.symbol}

    async def flatten(self) -> None:
        self.flattened = True

    async def cancel_order(self, order_id: str) -> None:
        self.canceled.append(order_id)

    async def get_account(self) -> dict[str, Any]:
        return {"equity": 10_000, "cash": 9_000}

    async def get_positions(self) -> list[dict[str, Any]]:
        return []

    def recent_events(self, limit: int = 50) -> list[dict[str, Any]]:
        return [{"limit": limit}]


class FakeManager:
    def __init__(self) -> None:
        self.sessions: dict[str, FakeSession] = {}
        self.created: dict[str, Any] = {}
        self.halted = False

    async def create(self, **options: Any) -> FakeSession:
        self.created = options
        session = FakeSession(str(options["id"]))
        symbols = options.get("symbols")
        if symbols:
            session.symbols = list(symbols)
            session.symbol = session.symbols[0]
        self.sessions[session.id] = session
        return session

    def get(self, session_id: str) -> FakeSession | None:
        return self.sessions.get(session_id)

    def list(self) -> list[FakeSession]:
        return list(self.sessions.values())

    async def halt_all(self) -> None:
        self.halted = True
        self.sessions.clear()


def _deps(
    *,
    backtest_result: dict[str, Any] | None = None,
    strategy_factory: Callable[[Mapping[str, Any]], Callable[[Mapping[str, Any]], object]]
    | None = None,
) -> McpDependencies:
    candles = [
        {"time": 1, "open": 100, "high": 101, "low": 99, "close": 100, "volume": 1},
        {"time": 2, "open": 101, "high": 102, "low": 100, "close": 101, "volume": 1},
    ]
    result = backtest_result or _result()
    factory = strategy_factory or (lambda _params: lambda _context: None)
    return McpDependencies(
        get_historical_candles=lambda _args: candles,
        candle_stats=lambda values: {
            "count": len(values),
            "firstTime": "a",
            "lastTime": "b",
            "estimatedIntervalMin": 1,
            "priceRange": {"low": 99, "high": 102},
        },
        list_strategies=lambda: [{"name": "ema-cross", "params": {}}],
        get_strategy=lambda _name: factory,
        backtest=lambda _options: result,
        walk_forward=lambda _options: {
            "windows": [
                {
                    "bestParams": {"fast": 3},
                    "oosTrades": 1,
                    "profitable": True,
                    "stabilityScore": 1,
                }
            ],
            "metrics": _metrics(),
            "bestParamsSummary": {"uniqueWinnerCount": 1},
        },
        expand_grid=lambda spec: [{"fast": value} for value in spec.get("fast", [3])],
        monte_carlo=lambda _options: {"final_equity": {"p5": 9_000, "p50": 10_000}},
        deflated_sharpe=lambda _options: 0.8,
        session_manager=FakeManager(),
        research_store=FakeStore(),
        now_ms=lambda: 123,
    )


def test_catalog_has_exact_25_names_descriptions_and_public_schemas() -> None:
    assert TOOL_NAMES == EXPECTED_NAMES
    assert DESCRIPTIONS == EXPECTED_DESCRIPTIONS
    assert tuple(SCHEMAS) == EXPECTED_NAMES
    assert len(SCHEMAS) == 25
    assert SCHEMAS["run_backtest"]["required"] == ["strategy"]
    assert SCHEMAS["run_backtest"]["properties"]["candles"]["items"]["required"] == [
        "time",
        "high",
        "low",
        "close",
    ]
    assert SCHEMAS["fetch_candles"]["properties"]["source"]["enum"] == [
        "yahoo",
        "csv",
        "auto",
    ]
    assert SCHEMAS["place_order"]["required"] == ["sessionId", "side"]
    assert SCHEMAS["research_open"] == {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {},
    }


@pytest.mark.asyncio
async def test_research_handlers_cover_the_full_compact_workflow() -> None:
    dependencies = _deps()
    tools = build_tools(dependencies)
    assert (await tools["list_strategies"].handler({}))["strategies"][0]["name"] == "ema-cross"
    fetched = await tools["fetch_candles"].handler({"symbol": "TEST"})
    assert fetched == {
        "count": 2,
        "first": {"time": 1, "open": 100, "high": 101, "low": 99, "close": 100, "volume": 1},
        "last": {"time": 2, "open": 101, "high": 102, "low": 100, "close": 101, "volume": 1},
    }
    backtested = await tools["run_backtest"].handler(
        {
            "data": {"symbol": "TEST"},
            "strategy": "ema-cross",
            "researchId": "automatic",
        }
    )
    assert set(backtested) == {"symbol", "interval", "metrics", "tradesPreview"}
    assert backtested["tradesPreview"][0] == {
        "side": "long",
        "entry": 101,
        "exit": 102,
        "pnl": 1,
        "reason": "target",
    }
    assert dependencies.research_store.entries[0]["verdict"]["deflatedSharpe"] == 0.8
    walked = await tools["walk_forward"].handler(
        {
            "candles": [fetched["first"], fetched["last"]],
            "strategy": "ema-cross",
            "trainBars": 1,
            "testBars": 1,
            "grid": {"fast": [3, 5]},
        }
    )
    assert walked["windows"] == 1
    assert walked["stability"] == {"uniqueWinnerCount": 1}

    robust = await tools["analyze_robustness"].handler(
        {"candles": [fetched["first"], fetched["last"]], "strategy": "ema-cross"}
    )
    assert robust["monteCarlo"]["finalEquity"]["p5"] == 9_000
    assert robust["deflatedSharpe"] == 0.8
    optimized = await tools["optimize_strategy"].handler(
        {
            "candles": [fetched["first"], fetched["last"]],
            "strategy": "ema-cross",
            "grid": {"fast": [3, 5]},
        }
    )
    assert len(optimized["leaderboard"]) == 2
    compared = await tools["compare_strategies"].handler(
        {
            "candles": [fetched["first"], fetched["last"]],
            "strategies": [{"strategy": "ema-cross"}, {"strategy": "buy-hold"}],
        }
    )
    assert compared["rankedBy"] == "profitFactor"
    assert len(compared["results"]) == 2
    stats = await tools["candle_stats"].handler({"candles": [fetched["first"], fetched["last"]]})
    assert stats["note"] == "Estimated bar interval ~1 min."


@pytest.mark.asyncio
async def test_robustness_and_candle_stats_degrade_gracefully() -> None:
    dependencies = _deps(backtest_result=_result(trade_count=1))
    tools = build_tools(dependencies)
    args = {"candles": [{"time": 1}], "strategy": "ema-cross"}
    robust = await tools["analyze_robustness"].handler(args)
    assert robust["monteCarlo"] is None
    assert "need at least 2" in robust["note"]
    dependencies.candle_stats = lambda _candles: None
    empty_tools = build_tools(dependencies)
    assert await empty_tools["candle_stats"].handler({"candles": [{"time": 1}]}) == {
        "stats": None,
        "note": "No candles returned.",
    }
    with pytest.raises(ValueError, match="Provide either"):
        await tools["run_backtest"].handler({"strategy": "ema-cross"})


@pytest.mark.asyncio
async def test_research_session_handlers_round_trip_with_camel_case_results() -> None:
    tools = build_tools(_deps())
    opened = await tools["research_open"].handler({"id": "x", "goal": "test"})
    assert opened["createdAt"] == "now"
    logged = await tools["research_log"].handler(
        {"id": "x", "hypothesis": "h", "params": {"a": 1}, "metrics": {"sharpe": 1.1}}
    )
    assert logged["hypothesis"] == "h"
    recalled = await tools["research_recall"].handler({"id": "x", "limit": 1})
    assert recalled["bestSharpe"] is None
    closed = await tools["research_close"].handler({"id": "x"})
    assert closed["closedAt"] == "later"


@pytest.mark.asyncio
async def test_live_handlers_use_injected_manager_and_attached_strategy() -> None:
    def factory(_params: Mapping[str, Any]) -> Callable[[Mapping[str, Any]], object]:
        return lambda _context: {"side": "long", "qty": 1, "stopLoss": 98, "takeProfit": 104}

    dependencies = _deps(strategy_factory=factory)
    manager = dependencies.session_manager
    tools = build_tools(dependencies)
    created = await tools["create_session"].handler(
        {"sessionId": "demo", "symbols": ["AAPL"], "riskPct": 2, "maxDailyLossPct": 3}
    )
    assert created["id"] == "demo"
    assert manager.created["risk_pct"] == 2
    assert len(await tools["list_sessions"].handler({})) == 1
    assert (await tools["session_status"].handler({"sessionId": "demo"}))["id"] == "demo"
    assert await tools["attach_strategy"].handler(
        {"sessionId": "demo", "strategy": "buy-hold", "params": {"holdBars": 5}}
    ) == {"ok": True, "strategy": "buy-hold", "params": {"holdBars": 5}}
    status = await tools["feed_price"].handler(
        {"sessionId": "demo", "price": 100, "symbol": "AAPL"}
    )
    assert status["id"] == "demo"
    session = manager.get("demo")
    assert session is not None
    assert session.bars[0]["time"] == 123
    assert session.orders[0]["stop"] == 98
    placed = await tools["place_order"].handler(
        {"sessionId": "demo", "side": "buy", "limitPrice": 99, "riskPct": 1}
    )
    assert placed["limit_price"] == 99
    assert await tools["close_position"].handler({"sessionId": "demo"}) == {"closed": "AAPL"}
    assert await tools["flatten"].handler({"sessionId": "demo"}) == {"ok": True}
    assert await tools["cancel_order"].handler({"sessionId": "demo", "orderId": "o1"}) == {
        "ok": True
    }
    assert (await tools["account"].handler({"sessionId": "demo"}))["cash"] == 9_000
    assert await tools["positions"].handler({"sessionId": "demo"}) == []
    assert await tools["recent_events"].handler({"sessionId": "demo", "limit": 7}) == [{"limit": 7}]
    halted = await tools["halt_all"].handler({})
    assert halted == {"ok": True, "sessionsHalted": 0}
    assert manager.halted is True


@pytest.mark.asyncio
async def test_live_handlers_report_unknown_sessions_and_missing_prices() -> None:
    tools = build_tools(_deps())
    with pytest.raises(ValueError, match='No session found with id "ghost"'):
        await tools["place_order"].handler({"sessionId": "ghost", "side": "long"})
    await tools["create_session"].handler({"sessionId": "demo", "symbol": "AAPL"})
    with pytest.raises(ValueError, match="Provide either"):
        await tools["feed_price"].handler({"sessionId": "demo"})


@pytest.mark.asyncio
async def test_attached_strategy_is_cleared_on_halt_and_session_recreation() -> None:
    def factory(_params: Mapping[str, Any]) -> Callable[[Mapping[str, Any]], object]:
        return lambda _context: {"side": "long", "qty": 1}

    dependencies = _deps(strategy_factory=factory)
    tools = build_tools(dependencies)
    await tools["create_session"].handler({"sessionId": "demo", "symbol": "AAPL"})
    await tools["attach_strategy"].handler(
        {"sessionId": "demo", "strategy": "buy-hold", "symbol": "AAPL"}
    )

    await tools["create_session"].handler({"sessionId": "demo", "symbol": "AAPL"})
    await tools["feed_price"].handler({"sessionId": "demo", "price": 100})
    recreated = dependencies.session_manager.get("demo")
    assert recreated is not None
    assert recreated.orders == []

    await tools["attach_strategy"].handler(
        {"sessionId": "demo", "strategy": "buy-hold", "symbol": "AAPL"}
    )
    await tools["halt_all"].handler({})
    await tools["create_session"].handler({"sessionId": "demo", "symbol": "AAPL"})
    await tools["feed_price"].handler({"sessionId": "demo", "price": 101})
    after_halt = dependencies.session_manager.get("demo")
    assert after_halt is not None
    assert after_halt.orders == []


@pytest.mark.asyncio
async def test_attached_strategy_and_order_failures_are_reported() -> None:
    dependencies = _deps(
        strategy_factory=lambda _params: (
            lambda _context: (_ for _ in ()).throw(RuntimeError("strategy failed"))
        )
    )
    tools = build_tools(dependencies)
    await tools["create_session"].handler({"sessionId": "demo", "symbol": "AAPL"})
    await tools["attach_strategy"].handler(
        {"sessionId": "demo", "strategy": "broken", "symbol": "AAPL"}
    )
    with pytest.raises(RuntimeError, match="strategy failed"):
        await tools["feed_price"].handler({"sessionId": "demo", "price": 100})

    dependencies = _deps(
        strategy_factory=lambda _params: lambda _context: {"side": "long", "qty": 1}
    )
    tools = build_tools(dependencies)
    await tools["create_session"].handler({"sessionId": "demo", "symbol": "AAPL"})
    await tools["attach_strategy"].handler(
        {"sessionId": "demo", "strategy": "buy-hold", "symbol": "AAPL"}
    )
    session = dependencies.session_manager.get("demo")
    assert session is not None
    session.order_error = RuntimeError("broker rejected")
    with pytest.raises(RuntimeError, match="broker rejected"):
        await tools["feed_price"].handler({"sessionId": "demo", "price": 100})


@pytest.mark.asyncio
async def test_halt_all_blocks_an_inflight_attached_strategy_order() -> None:
    evaluating = asyncio.Event()
    release = asyncio.Event()

    async def signal(_context: Mapping[str, Any]) -> dict[str, object]:
        evaluating.set()
        await release.wait()
        return {"side": "long", "qty": 1}

    dependencies = _deps(strategy_factory=lambda _params: signal)
    tools = build_tools(dependencies)
    await tools["create_session"].handler({"sessionId": "demo", "symbol": "AAPL"})
    await tools["attach_strategy"].handler(
        {"sessionId": "demo", "strategy": "slow", "symbol": "AAPL"}
    )
    session = dependencies.session_manager.get("demo")
    assert session is not None

    feed_task = asyncio.create_task(
        tools["feed_price"].handler({"sessionId": "demo", "price": 100})
    )
    await evaluating.wait()
    halt_task = asyncio.create_task(tools["halt_all"].handler({}))
    await asyncio.sleep(0)
    release.set()

    with pytest.raises(LiveTradingDisabledError, match="kill switch"):
        await feed_task
    await halt_task
    assert session.orders == []


@pytest.mark.asyncio
async def test_sdk_server_and_invocation_return_strict_text_results_and_errors() -> None:
    tools = build_tools(_deps())
    server = create_server(tools=tools)
    assert isinstance(server, Server)
    result = await invoke_tool(tools, "list_strategies", {})
    assert isinstance(result, CallToolResult)
    assert result.isError is False
    assert isinstance(result.content[0], TextContent)
    assert json.loads(result.content[0].text)["strategies"][0]["name"] == "ema-cross"

    async def non_finite(_args: Mapping[str, Any]) -> object:
        return {"bad": float("nan")}

    tools["list_strategies"].handler = non_finite
    bad = await invoke_tool(tools, "list_strategies", {})
    assert bad.isError is True
    assert bad.content[0].text.startswith("Error:")

    async def broken(_args: Mapping[str, Any]) -> object:
        raise RuntimeError("broken\ud800")

    tools["list_strategies"].handler = broken
    error = await invoke_tool(tools, "list_strategies", {})
    assert error.isError is True
    assert error.content[0].text == "Error: broken�"
    missing = await invoke_tool(tools, "ghost", {})
    assert missing.isError is True
    assert "Unknown tool" in missing.content[0].text


@pytest.mark.asyncio
async def test_sdk_request_handlers_publish_and_validate_the_exact_catalog() -> None:
    server = create_server(tools=build_tools(_deps()))
    listed = await server.request_handlers[ListToolsRequest](ListToolsRequest())
    assert [tool.name for tool in listed.root.tools] == list(EXPECTED_NAMES)
    assert listed.root.tools[2].inputSchema == SCHEMAS["run_backtest"]

    called = await server.request_handlers[CallToolRequest](
        CallToolRequest(params=CallToolRequestParams(name="list_strategies", arguments={}))
    )
    assert called.root.isError is False
    assert json.loads(called.root.content[0].text)["strategies"][0]["name"] == "ema-cross"


@pytest.mark.asyncio
async def test_strict_result_boundary_rejects_nonportable_json_shapes() -> None:
    tools = build_tools(_deps())

    async def return_value(_args: Mapping[str, Any]) -> object:
        return current[0]

    tools["list_strategies"].handler = return_value
    current: list[object] = [{"values": [1, 1.5, True, None, "ok"]}]
    assert (await invoke_tool(tools, "list_strategies", None)).isError is False

    cyclic_mapping: dict[str, object] = {}
    cyclic_mapping["self"] = cyclic_mapping
    cyclic_sequence: list[object] = []
    cyclic_sequence.append(cyclic_sequence)
    for invalid in (
        1 << 60,
        {1: "non-string key"},
        {"unsupported": {1, 2}},
        cyclic_mapping,
        cyclic_sequence,
    ):
        current[0] = invalid
        result = await invoke_tool(tools, "list_strategies", {})
        assert result.isError is True
        assert result.content[0].text.startswith("Error:")


def test_default_dependencies_are_constructed_without_external_io(tmp_path: Path) -> None:
    dependencies = default_dependencies(research_directory=tmp_path)
    assert len(build_tools(dependencies)) == 25
    assert dependencies.now_ms() > 0


@pytest.mark.asyncio
async def test_production_dependencies_use_csv_engine_research_and_store_without_network(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "prices.csv"
    csv_path.write_text(
        "time,open,high,low,close,volume\n"
        "2025-01-01T00:00:00Z,100,101,99,100,10\n"
        "2025-01-02T00:00:00Z,100,103,100,102,11\n"
        "2025-01-03T00:00:00Z,102,105,101,104,12\n",
        encoding="utf-8",
    )
    dependencies = default_dependencies(research_directory=tmp_path / "research")
    loaded = await dependencies.get_historical_candles(
        {
            "source": "csv",
            "csvPath": str(csv_path),
            "cacheDir": str(tmp_path / "cache"),
            "cache": False,
        }
    )
    assert len(loaded) == 3
    assert (
        dependencies.monte_carlo({"trade_pnls": [1, -1], "iterations": 5, "seed": "mcp-test"})[
            "final_equity"
        ]["p50"]
        == 10_000
    )
    assert 0 <= dependencies.deflated_sharpe({"sharpe": 1, "sample_size": 10, "num_trials": 1}) <= 1

    tools = build_tools(dependencies)
    result = await tools["run_backtest"].handler(
        {"candles": loaded, "strategy": "buy-hold", "symbol": "CSV"}
    )
    assert result["symbol"] == "CSV"
    walked = await tools["walk_forward"].handler(
        {
            "candles": loaded,
            "strategy": "buy-hold",
            "trainBars": 1,
            "testBars": 1,
            "grid": {},
        }
    )
    assert walked["windows"] == 2
    opened = await tools["research_open"].handler({"id": "csv", "goal": "offline"})
    assert opened["goal"] == "offline"


@pytest.mark.asyncio
async def test_stdio_runner_uses_the_sdk_streams(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[object] = []

    @asynccontextmanager
    async def streams() -> Any:
        yield "read", "write"

    class Application:
        def create_initialization_options(self) -> str:
            return "options"

        async def run(self, *args: object) -> None:
            captured.extend(args)

    monkeypatch.setattr(mcp_server, "stdio_server", streams)
    await run_stdio_server(Application())  # type: ignore[arg-type]
    assert captured == ["read", "write", "options"]


def test_entrypoint_and_version_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []

    def run(coroutine: Any) -> None:
        calls.append(coroutine)
        coroutine.close()

    def missing_version(_package: str) -> str:
        raise mcp_server.PackageNotFoundError

    monkeypatch.setattr(mcp_server.asyncio, "run", run)
    monkeypatch.setattr(mcp_server, "version", missing_version)
    monkeypatch.setattr(mcp_server.sys, "argv", ["tradelab-mcp"])
    assert isinstance(create_server(tools=build_tools(_deps())), Server)
    mcp_server.entrypoint()
    assert len(calls) == 1
