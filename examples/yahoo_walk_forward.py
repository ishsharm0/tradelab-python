"""Fetch public Yahoo bars and run rolling walk-forward validation."""

from __future__ import annotations

import asyncio

from tradelab import get_historical_candles, grid, walk_forward_optimize
from tradelab.strategies import get_strategy


async def main() -> None:
    candles = await get_historical_candles(source="yahoo", symbol="SPY", interval="1d", period="5y")
    factory = get_strategy("ema-cross")
    result = walk_forward_optimize(
        candles=candles,
        parameter_sets=grid({"fast": [8, 10, 12], "slow": [21, 30, 40], "rr": [1.5, 2]}),
        train_bars=252,
        test_bars=63,
        signal_factory=lambda parameters: factory(parameters),
        backtest_options={"symbol": "SPY", "interval": "1d"},
    )
    print(result["bestParamsSummary"])
    print(result["metrics"])


if __name__ == "__main__":
    asyncio.run(main())
