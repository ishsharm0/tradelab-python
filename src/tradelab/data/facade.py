"""Unified cached CSV and Yahoo historical-data workflows."""

from __future__ import annotations

import math
import re
from collections.abc import Awaitable, Callable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from tradelab.engine import backtest
from tradelab.errors import ValidationError
from tradelab.models import BacktestResult

from .cache import load_candles_from_cache, save_candles_to_cache
from .csv import _time_ms, load_candles_from_csv, normalize_candles
from .yahoo import Candle, fetch_historical

Fetcher = Callable[..., Awaitable[list[Candle]]]


def _range_period(start: object, end: object) -> str:
    if not start or not end:
        return "custom"

    def timestamp(value: object) -> float:
        if isinstance(value, (int, float, bool)):
            return math.nan
        if not isinstance(value, datetime) and re.fullmatch(
            r"[+-]?(?:\d+\.?\d*|\.\d+)", str(value).strip()
        ):
            return math.nan
        return float(_time_ms(value))

    try:
        start_ms, end_ms = timestamp(start), timestamp(end)
    except (ValueError, OverflowError, OSError):
        return "custom"
    if not math.isfinite(start_ms) or not math.isfinite(end_ms) or end_ms <= start_ms:
        return "custom"
    days = max(1, math.floor((end_ms - start_ms) / 86_400_000 + 0.5))
    return f"{days}d"


async def get_historical_candles(
    *,
    source: str = "auto",
    symbol: str | None = None,
    interval: str = "1d",
    period: object = None,
    cache: bool = True,
    refresh: bool = False,
    cache_dir: str | Path | None = None,
    csv: Mapping[str, object] | None = None,
    csv_path: str | Path | None = None,
    fetcher: Fetcher = fetch_historical,
    **options: Any,
) -> list[Candle]:
    """Resolve a CSV or Yahoo source, optionally reading/writing the local cache."""
    csv_options = dict(csv or {})
    file_path = (
        csv_path
        or csv_options.pop("file_path", None)
        or csv_options.pop("filePath", None)
        or csv_options.pop("path", None)
    )
    selected_source = "csv" if source == "auto" and file_path else source
    if selected_source == "auto":
        selected_source = "yahoo"
    effective_cache_dir = Path(cache_dir or Path.cwd() / "output" / "data")

    if selected_source == "csv":
        if not isinstance(file_path, (str, Path)) or not file_path:
            raise ValidationError('CSV source requires "csv_path" or "csv.file_path"')
        csv_options.update(options)
        loader: Callable[..., list[Candle]] = load_candles_from_csv
        rows = loader(file_path, **csv_options)
        if cache and symbol:
            resolved_period = period or _range_period(
                csv_options.get("start_date", csv_options.get("startDate")),
                csv_options.get("end_date", csv_options.get("endDate")),
            )
            save_candles_to_cache(
                rows,
                symbol=symbol,
                interval=interval,
                period=resolved_period,
                out_dir=effective_cache_dir,
                source="csv",
            )
        return rows

    if selected_source != "yahoo":
        raise ValidationError(f'Unsupported data source "{selected_source}"')
    if not symbol:
        raise ValidationError('Yahoo source requires "symbol"')
    resolved_period = period if period is not None else "1y"
    if cache and not refresh:
        cached = load_candles_from_cache(symbol, interval, resolved_period, effective_cache_dir)
        if cached:
            return cached
    rows = normalize_candles(await fetcher(symbol, interval, resolved_period, **options))
    if cache:
        save_candles_to_cache(
            rows,
            symbol=symbol,
            interval=interval,
            period=resolved_period,
            out_dir=effective_cache_dir,
            source="yahoo",
        )
    return rows


async def backtest_historical(
    *,
    data: Mapping[str, Any] | None = None,
    backtest_options: Mapping[str, Any] | None = None,
    **legacy: Any,
) -> BacktestResult:
    """Fetch historical candles and run the deterministic bar backtester."""
    data_options = dict(data or legacy)
    candles = await get_historical_candles(**data_options)
    options = {
        "candles": candles,
        "symbol": data_options.get("symbol"),
        "interval": data_options.get("interval"),
        "range": data_options.get("period", "custom"),
        **dict(backtest_options or {}),
    }
    return backtest(**options)


__all__ = ["backtest_historical", "get_historical_candles"]
