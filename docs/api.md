# API map

The distribution name is `tradelab-python`; all imports begin with `tradelab`.

## Backtesting

| Function | Result |
| --- | --- |
| `backtest(...)` | Deterministic candle backtest |
| `await backtest_async(...)` | Async-signal candle backtest with budgets |
| `backtest_ticks(...)` | Tick/quote replay with deterministic queue fills |
| `backtest_portfolio(...)` | Shared-capital multi-system simulation |
| `walk_forward_optimize(...)` | Rolling or anchored out-of-sample validation |
| `optimize(...)` | Stable concurrent parameter sweep |
| `grid(spec)` | Cartesian parameter expansion |

Results use canonical camel-case keys such as `eqSeries`, `openPositions`, `finalEquity`,
and `profitFactor`. Python keyword options use snake case; compatibility aliases are accepted
where the JavaScript API exposed them.

## Data

- `await get_historical_candles(...)`
- `await fetch_historical(...)`
- `load_candles_from_csv(...)`
- `normalize_candles(...)`
- `merge_candles(...)`
- `candle_stats(...)`
- `save_candles_to_cache(...)`
- `load_candles_from_cache(...)`

## Research

- `monte_carlo(...)`
- `deflated_sharpe(...)`
- `sweep_haircut(...)`
- `probability_of_backtest_overfitting(...)`
- `combinatorial_purged_splits(...)`
- `create_research_store(...)`

## Reports

- `summarize(metrics)`
- `export_metrics_json(result, ...)`
- `export_trades_csv(trades, ...)`
- `export_markdown_report(result, ...)`
- `render_html_report(...)` / `export_html_report(...)`
- `export_backtest_artifacts(result, ...)`

HTML reports contain inline CSS and SVG only. They do not load a CDN, remote font, image,
script, or analytics endpoint.

## Strategies

`list_strategies()`, `get_strategy(name)`, and `register_strategy(name, definition)` manage
the process-local registry. Included strategies are `ema-cross`, `rsi-reversion`,
`donchian-breakout`, and `buy-hold`.

## Live runtime

`tradelab.live` exports `PaperEngine`, `TradingSession`, `SessionManager`, `LiveEngine`,
`LiveOrchestrator`, feed providers, candle aggregation, JSON storage, the authenticated
loopback dashboard, and Alpaca/Binance/Coinbase/Interactive Brokers adapters. External adapters
without authenticated reconnecting order streams are available for request mapping and data, but
fail closed when asked to provide managed live protection.

## Error model

All expected public failures derive from `TradeLabError`:

- `ValidationError`
- `StrategyError`
- `DataProviderError`
- `BrokerError`
- `RiskRejectedError`
- `LiveTradingDisabledError`

Each error may include a JSON-safe `context` mapping for programmatic diagnostics.
