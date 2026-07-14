# Live-trading safety

TradeLab defaults to research and paper execution. Installing the package, importing a broker,
or starting the MCP server does not authorize a live order.

## Live gate

A live session requires:

```bash
export TRADELAB_ALLOW_LIVE=true
```

The caller must also pass `confirm_live=True` and provide a non-paper credentialed broker.
Missing any condition raises `LiveTradingDisabledError` before the session starts.

## Risk controls

The live risk manager supports:

- percentage and dollar daily-loss limits;
- peak-to-trough drawdown halts;
- session windows in America/New_York;
- maximum daily trade counts;
- cooldowns after losses;
- per-position, gross, and net exposure caps;
- process-level `halt_all()` flattening.

Daily state resets at actual New York midnight across standard and daylight-saving time.

## Credentials

Pass credentials through environment variables or an external secret manager. Never place API
keys in strategy parameters, research logs, report metadata, CLI history, or committed files.
Use paper/testnet endpoints first and configure broker-side withdrawal restrictions and IP
allowlists where available.

## Operational checklist

1. Replay representative data with realistic costs.
2. Run walk-forward and robustness analysis.
3. Exercise the exact strategy in a paper session.
4. Verify quantity steps, minimum sizes, symbols, and bracket behavior.
5. Configure daily loss and exposure limits below broker limits.
6. Test `flatten()` and `halt_all()` before enabling live mode.
7. Start with the smallest accepted quantity and monitor broker acknowledgements.

No backtest or statistical result guarantees future performance.
