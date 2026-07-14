"""Historical market-data normalization and providers."""

from .cache import cached_candles_path, load_candles_from_cache, save_candles_to_cache
from .csv import candle_stats, load_candles_from_csv, merge_candles, normalize_candles
from .facade import backtest_historical, get_historical_candles
from .yahoo import fetch_historical, fetch_latest_candle

__all__ = [
    "backtest_historical",
    "cached_candles_path",
    "candle_stats",
    "fetch_historical",
    "fetch_latest_candle",
    "get_historical_candles",
    "load_candles_from_cache",
    "load_candles_from_csv",
    "merge_candles",
    "normalize_candles",
    "save_candles_to_cache",
]
