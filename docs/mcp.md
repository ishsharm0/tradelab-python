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

Paper is the default. The bundled stdio server does not install a live broker factory; this keeps
credentials out of tool arguments and prevents an uncertified REST adapter from being mistaken
for a protected live connection. Programmatic servers may inject a factory, but live sessions
still cannot bypass `TRADELAB_ALLOW_LIVE=true`, explicit confirmation, genuine streamed order
updates, broker credentials, risk controls, or the cancellation-safe process kill switch.

Tool results are encoded as strict JSON text. Expected failures are returned as MCP tool errors
without tracebacks or credential material.
