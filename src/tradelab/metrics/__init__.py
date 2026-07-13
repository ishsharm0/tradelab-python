"""Performance analytics compatible with TradeLab's JavaScript metrics module."""

from .annualize import periods_per_year
from .benchmark import benchmark_stats
from .build import build_metrics
from .finite import BIG_NUMBER, clamp_finite

__all__ = [
    "BIG_NUMBER",
    "benchmark_stats",
    "build_metrics",
    "clamp_finite",
    "periods_per_year",
]
