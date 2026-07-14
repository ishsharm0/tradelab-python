"""Source-compatible MCP tool catalog and public JSON schemas."""

# ruff: noqa: E501 -- descriptions are immutable protocol strings from the source server.

from __future__ import annotations

from typing import Any

_DRAFT = "http://json-schema.org/draft-07/schema#"

TOOL_NAMES = (
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

DESCRIPTIONS = {
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


def _object(
    properties: dict[str, Any] | None = None,
    required: list[str] | None = None,
    *,
    root: bool = False,
    passthrough: bool = False,
) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "object", "properties": properties or {}}
    if root:
        schema = {"$schema": _DRAFT, **schema}
    if required:
        schema["required"] = required
    if passthrough:
        schema["additionalProperties"] = {}
    return schema


def _record(value: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "propertyNames": {"type": "string"},
        "additionalProperties": value or {},
    }


def _array(items: dict[str, Any]) -> dict[str, Any]:
    return {"type": "array", "items": items}


_NUMBER = {"type": "number"}
_STRING = {"type": "string"}
_BOOLEAN = {"type": "boolean"}
_ANY: dict[str, Any] = {}
_CANDLE = _object(
    {
        "time": _NUMBER,
        "open": _NUMBER,
        "high": _NUMBER,
        "low": _NUMBER,
        "close": _NUMBER,
        "volume": _NUMBER,
    },
    ["time", "high", "low", "close"],
)
_BAR = _object(
    {
        "time": _NUMBER,
        "open": _NUMBER,
        "high": _NUMBER,
        "low": _NUMBER,
        "close": _NUMBER,
        "volume": _NUMBER,
    },
    ["time", "close"],
)
_DATA_PROPERTIES = {
    "source": {"type": "string", "enum": ["yahoo", "csv", "auto"]},
    "symbol": _STRING,
    "interval": _STRING,
    "period": _STRING,
    "csvPath": _STRING,
    "cache": _BOOLEAN,
}
_DATA = _object(_DATA_PROPERTIES, passthrough=True)
_CANDLES_AND_DATA = {"candles": _array(_CANDLE), "data": _DATA}
_PARAMS = _record()
_GRID = _record(_array(_ANY))
_BACKTEST_OPTIONS = _record()


def _root(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return _object(properties, required, root=True)


SCHEMAS: dict[str, dict[str, Any]] = {
    "list_strategies": _root({}),
    "fetch_candles": _root(_DATA_PROPERTIES),
    "run_backtest": _root(
        {
            **_CANDLES_AND_DATA,
            "symbol": _STRING,
            "interval": _STRING,
            "strategy": _STRING,
            "params": _PARAMS,
            "backtestOptions": _BACKTEST_OPTIONS,
        },
        ["strategy"],
    ),
    "walk_forward": _root(
        {
            **_CANDLES_AND_DATA,
            "interval": _STRING,
            "strategy": _STRING,
            "trainBars": _NUMBER,
            "testBars": _NUMBER,
            "stepBars": _NUMBER,
            "mode": {"type": "string", "enum": ["rolling", "anchored"]},
            "scoreBy": _STRING,
            "grid": _GRID,
            "backtestOptions": _BACKTEST_OPTIONS,
        },
        ["strategy", "trainBars", "testBars"],
    ),
    "analyze_robustness": _root(
        {
            **_CANDLES_AND_DATA,
            "symbol": _STRING,
            "interval": _STRING,
            "strategy": _STRING,
            "params": _PARAMS,
            "equityStart": _NUMBER,
            "iterations": _NUMBER,
            "blockSize": _NUMBER,
            "seed": {"anyOf": [_STRING, _NUMBER]},
            "numTrials": _NUMBER,
            "sharpeStd": _NUMBER,
            "skew": _NUMBER,
            "kurtosis": _NUMBER,
            "backtestOptions": _BACKTEST_OPTIONS,
        },
        ["strategy"],
    ),
    "optimize_strategy": _root(
        {
            **_CANDLES_AND_DATA,
            "symbol": _STRING,
            "interval": _STRING,
            "strategy": _STRING,
            "grid": _GRID,
            "scoreBy": _STRING,
            "backtestOptions": _BACKTEST_OPTIONS,
        },
        ["strategy"],
    ),
    "compare_strategies": _root(
        {
            **_CANDLES_AND_DATA,
            "symbol": _STRING,
            "interval": _STRING,
            "strategies": _array(
                _object(
                    {"strategy": _STRING, "params": _PARAMS},
                    ["strategy"],
                )
            ),
            "scoreBy": _STRING,
            "backtestOptions": _BACKTEST_OPTIONS,
        },
        ["strategies"],
    ),
    "candle_stats": _root(_CANDLES_AND_DATA),
    "create_session": _root(
        {
            "sessionId": _STRING,
            "symbol": _STRING,
            "symbols": _array(_STRING),
            "mode": {"type": "string", "enum": ["paper", "live"]},
            "interval": _STRING,
            "equity": _NUMBER,
            "riskPct": _NUMBER,
            "maxDailyLossPct": _NUMBER,
            "confirmLive": _BOOLEAN,
        },
        ["sessionId"],
    ),
    "list_sessions": _root({}),
    "session_status": _root({"sessionId": _STRING}, ["sessionId"]),
    "feed_price": _root(
        {"sessionId": _STRING, "bar": _BAR, "price": _NUMBER, "symbol": _STRING},
        ["sessionId"],
    ),
    "place_order": _root(
        {
            "sessionId": _STRING,
            "side": {"type": "string", "enum": ["long", "short", "buy", "sell"]},
            "type": {
                "type": "string",
                "enum": ["market", "limit", "stop", "stop_limit"],
            },
            "qty": _NUMBER,
            "riskPct": _NUMBER,
            "stop": _NUMBER,
            "target": _NUMBER,
            "rr": _NUMBER,
            "limitPrice": _NUMBER,
            "symbol": _STRING,
        },
        ["sessionId", "side"],
    ),
    "close_position": _root({"sessionId": _STRING, "symbol": _STRING}, ["sessionId"]),
    "flatten": _root({"sessionId": _STRING}, ["sessionId"]),
    "cancel_order": _root({"sessionId": _STRING, "orderId": _STRING}, ["sessionId", "orderId"]),
    "account": _root({"sessionId": _STRING}, ["sessionId"]),
    "positions": _root({"sessionId": _STRING}, ["sessionId"]),
    "recent_events": _root({"sessionId": _STRING, "limit": _NUMBER}, ["sessionId"]),
    "attach_strategy": _root(
        {
            "sessionId": _STRING,
            "strategy": _STRING,
            "params": _PARAMS,
            "symbol": _STRING,
        },
        ["sessionId", "strategy"],
    ),
    "halt_all": _root({}),
    "research_open": _root({}),
    "research_log": _root({}),
    "research_recall": _root({}),
    "research_close": _root({}),
}
