"""Annualization intervals compatible with the JavaScript metrics module."""

from __future__ import annotations

import math

from .finite import Number, _is_number

_MS_PER_YEAR = 365 * 24 * 60 * 60 * 1000
_INTERVAL_PERIODS: dict[str, int] = {
    "1m": 98_280,
    "2m": 49_140,
    "5m": 19_656,
    "15m": 6_552,
    "30m": 3_276,
    "1h": 1_638,
    "60m": 1_638,
    "1d": 252,
    "1wk": 52,
    "1mo": 12,
}


def _js_round_positive(value: float) -> int | float:
    """Return JavaScript ``Math.round`` for positive binary64 values."""
    if not math.isfinite(value):
        return value
    integer = math.floor(value)
    if value == integer:
        return integer
    return integer + 1 if value - integer >= 0.5 else integer


def periods_per_year(interval: str | None, est_bar_ms: Number | None) -> int | float:
    """Return annualization periods using known trading intervals or a 24/7 estimate."""
    if interval in _INTERVAL_PERIODS:
        return _INTERVAL_PERIODS[interval]
    if _is_number(est_bar_ms):
        try:
            bar_ms = float(est_bar_ms)
        except (OverflowError, ValueError):
            return 252
        if math.isfinite(bar_ms) and bar_ms > 0:
            return _js_round_positive(_MS_PER_YEAR / bar_ms)
    return 252
