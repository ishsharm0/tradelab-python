"""Unified CSV/Yahoo/cache data facade contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

from tradelab.data.facade import backtest_historical, get_historical_candles
from tradelab.data.yahoo import Candle
from tradelab.errors import ValidationError
from tradelab.models import BacktestResult


@pytest.mark.asyncio
async def test_auto_csv_loads_caches_and_derives_custom_period(tmp_path: Path) -> None:
    csv_path = tmp_path / "bars.csv"
    csv_path.write_text(
        "time,open,high,low,close\n2025-01-02T14:30:00Z,100,101,99,100.5\n",
        encoding="utf-8",
    )
    candles = await get_historical_candles(
        csv_path=csv_path,
        symbol="TEST",
        interval="1d",
        cache_dir=tmp_path,
    )
    assert len(candles) == 1
    assert (tmp_path / "candles-TEST-1d-custom.json").exists()


@pytest.mark.asyncio
async def test_yahoo_cache_hit_bypasses_fetch_and_refresh_replaces(tmp_path: Path) -> None:
    calls: list[tuple[str, str, object]] = []

    async def fetcher(
        symbol: str, interval: str, period: object, **_options: object
    ) -> list[Candle]:
        calls.append((symbol, interval, period))
        return [{"time": 1_700_000_000_000, "open": 2, "high": 3, "low": 1, "close": 2}]

    first = await get_historical_candles(
        symbol="SPY", interval="1d", period="1y", cache_dir=tmp_path, fetcher=fetcher
    )
    second = await get_historical_candles(
        symbol="SPY", interval="1d", period="1y", cache_dir=tmp_path, fetcher=fetcher
    )
    third = await get_historical_candles(
        symbol="SPY",
        interval="1d",
        period="1y",
        cache_dir=tmp_path,
        fetcher=fetcher,
        refresh=True,
    )
    assert first == second
    assert third[0]["time"] == 1_700_000_000_000
    assert calls == [("SPY", "1d", "1y"), ("SPY", "1d", "1y")]


@pytest.mark.asyncio
async def test_facade_validation_and_backtest_composition(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="CSV source requires"):
        await get_historical_candles(source="csv")
    with pytest.raises(ValidationError, match="Yahoo source requires"):
        await get_historical_candles(source="yahoo")
    with pytest.raises(ValidationError, match="Unsupported data source"):
        await get_historical_candles(source="fred")

    csv_path = tmp_path / "bars.csv"
    csv_path.write_text(
        "time,open,high,low,close\n"
        "2025-01-02T14:30:00Z,100,101,99,100\n"
        "2025-01-02T14:31:00Z,100,102,99,101\n",
        encoding="utf-8",
    )
    result = await backtest_historical(
        data={"csv_path": csv_path, "symbol": "TEST", "interval": "1m", "cache": False},
        backtest_options={"warmup_bars": 0, "signal": lambda _context: None},
    )
    assert isinstance(result, BacktestResult)
    assert result["symbol"] == "TEST"
