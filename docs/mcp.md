# MCP server

Install the package and configure an MCP client to launch `tradelab-mcp` over stdio:

```json
{
  "mcpServers": {
    "tradelab": {
      "command": "tradelab-mcp"
    }
  }
}
```

The server exposes 25 tools.

## Research and validation

- `list_strategies`
- `fetch_candles`
- `run_backtest`
- `walk_forward`
- `analyze_robustness`
- `optimize_strategy`
- `compare_strategies`
- `candle_stats`

## Persistent research loop

- `research_open`
- `research_log`
- `research_recall`
- `research_close`

## Paper and live sessions

- `create_session`
- `list_sessions`
- `session_status`
- `feed_price`
- `place_order`
- `close_position`
- `flatten`
- `cancel_order`
- `account`
- `positions`
- `recent_events`
- `attach_strategy`
- `halt_all`

Paper is the default. MCP live sessions cannot bypass `TRADELAB_ALLOW_LIVE=true`, explicit
live confirmation, broker credentials, risk controls, or the process-level kill switch.

Tool results are encoded as strict JSON text. Expected failures are returned as MCP tool errors
without tracebacks or credential material.
