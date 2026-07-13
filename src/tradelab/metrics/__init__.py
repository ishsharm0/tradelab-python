"""Performance analytics compatible with TradeLab's JavaScript metrics module."""

from .analytics import BIG_NUMBER, benchmark_stats, build_metrics, clamp_finite, periods_per_year

__all__ = [
    "BIG_NUMBER",
    "benchmark_stats",
    "build_metrics",
    "clamp_finite",
    "periods_per_year",
]
