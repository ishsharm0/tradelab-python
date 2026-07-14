# TradeLab for Python

TradeLab is an agent-native trading toolkit for research, deterministic backtesting,
paper trading, and permission-gated live execution.

The distribution is named `tradelab-python`; the import package and command are both
`tradelab`.

```bash
pip install tradelab-python
```

Python 3.11 or newer is required.

## Why TradeLab

- One signal contract across bar backtests, async signals, tick replay, portfolios,
  walk-forward validation, paper sessions, and live adapters.
- Deterministic execution with explicit slippage, spreads, commissions, carry, funding,
  risk sizing, stops, targets, partial exits, and daily circuit breakers.
- Yahoo and CSV ingestion with normalization, chunking, retry control, throttling, and
  atomic local caches.
- Research statistics for Monte Carlo analysis, Deflated Sharpe, PBO, CPCV, and persistent
  hypothesis logs.
- Self-contained JSON, CSV, Markdown, and offline HTML reports.
- A 25-tool MCP server for research and paper/live session control.
- Strict typing, finite JSON outputs, and no implicit live-trading permission.

## Quick start

```python
import asyncio

from tradelab import backtest, get_historical_candles
from tradelab.reporting import export_backtest_artifacts
from tradelab.strategies import get_strategy


async def main() -> None:
    candles = await get_historical_candles(
        source="yahoo",
        symbol="SPY",
        interval="1d",
        period="2y",
        cache=True,
    )
    signal = get_strategy("ema-cross")({"fast": 10, "slow": 30, "rr": 2})
    result = backtest(
        candles=candles,
        symbol="SPY",
        interval="1d",
        equity=10_000,
        risk_pct=1,
        warmup_bars=50,
        costs={"slippageBps": 1, "commissionBps": 0.5},
        signal=signal,
    )
    print(result["metrics"])
    export_backtest_artifacts(result, out_dir="output")


asyncio.run(main())
```

`BacktestResult` behaves like an immutable mapping and can be converted to an independent
plain dictionary with `result.to_dict()`.

## Signal contract

A strategy receives a context mapping and returns `None` or an order intent:

```python
def signal(context: dict[str, object]) -> dict[str, object] | None:
    bar = context["bar"]
    if context["openPosition"] is not None:
        return None
    close = float(bar["close"])
    return {"side": "long", "entry": close, "stop": close * 0.97, "rr": 2}
```

Common fields are `side`, `entry`, `stop`, `takeProfit`, `rr`, `qty`, `size`, `riskPct`,
and `riskFraction`. Snake-case Python options are accepted by the Python API; canonical
camel-case result payloads remain compatible with the original TradeLab contracts.

## Data

```python
from tradelab.data import get_historical_candles, load_candles_from_csv

yahoo = await get_historical_candles(
    source="yahoo", symbol="QQQ", interval="1d", period="1y"
)
csv_rows = load_candles_from_csv("data/btc.csv")
```

All candles normalize to `time`, `open`, `high`, `low`, `close`, and `volume`. Cache writes
use atomic replacement and strict JSON. The Yahoo client accepts an injected
`httpx.AsyncClient`, clock, and sleeper for deterministic integration tests.

## Validation and portfolios

```python
from tradelab import backtest_portfolio, grid, walk_forward_optimize
from tradelab.strategies import get_strategy

factory = get_strategy("ema-cross")
walk_forward = walk_forward_optimize(
    candles=yahoo,
    train_bars=180,
    test_bars=60,
    parameter_sets=grid({"fast": [8, 10], "slow": [21, 30], "rr": [1.5, 2]}),
    signal_factory=lambda params: factory(params),
)

portfolio = backtest_portfolio(
    equity=100_000,
    max_daily_loss_pct=3,
    systems=[
        {"symbol": "SPY", "candles": spy, "signal": spy_signal, "weight": 2},
        {"symbol": "QQQ", "candles": qqq, "signal": qqq_signal, "weight": 1},
    ],
)
```

Portfolio equity points expose `lockedCapital` and `availableCapital`. Daily loss controls
reset at actual New York midnight, including DST boundaries.

## Paper and live sessions

```python
from tradelab.live import SessionManager

manager = SessionManager()
session = await manager.create(id="paper-spy", symbol="SPY", mode="paper", equity=25_000)
await session.push_bar(
    {"time": 1_735_828_200_000, "open": 100, "high": 101, "low": 99, "close": 100}
)
await session.place_order(side="long", risk_pct=1, stop=98, target=104)
print(session.get_status())
await manager.halt_all()
```

Paper mode needs no credentials. Live mode requires all three conditions:

1. `TRADELAB_ALLOW_LIVE=true`
2. `confirm_live=True`
3. a connected, credentialed non-paper broker adapter

The process-level `halt_all()` operation flattens and stops every managed session.

## Command line

```bash
tradelab backtest --source yahoo --symbol SPY --interval 1d --period 1y
tradelab backtest --source csv --csv-path ./data/spy.csv --strategy buy-hold
tradelab walk-forward --source yahoo --symbol QQQ --period 2y
tradelab run ema-cross --source yahoo --symbol SPY --params '{"fast": 8}'
tradelab prefetch --symbol SPY --interval 1d --period 1y
tradelab import-csv ./data/spy.csv --symbol SPY --interval 1d
```

## MCP server

```json
{
  "mcpServers": {
    "tradelab": {
      "command": "tradelab-mcp"
    }
  }
}
```

The server exposes research, robustness, persistent research-session, and paper/live
session tools over stdio. Live session creation remains subject to the same environment,
confirmation, and credential gates as the Python API.

## Development

```bash
uv sync --all-extras
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests
uv build
uv run twine check dist/*
```

The original JavaScript repository is used as an immutable parity oracle. Python adds
stricter finite-number, path-containment, atomic-write, concurrency, and timezone safety
where preserving a JavaScript defect would be dangerous.

## License

MIT
