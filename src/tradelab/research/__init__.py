"""Quantitative research tools with JavaScript-source-compatible semantics."""

from .combinations import combinations
from .cpcv import CpcvSplit, combinatorial_purged_splits
from .deflated_sharpe import SweepHaircut, deflated_sharpe, sweep_haircut
from .monte_carlo import MonteCarloResult, PathPercentileBands, PercentileBands, monte_carlo
from .pbo import PboResult, probability_of_backtest_overfitting
from .stats import Moments, moments, normal_cdf, normal_ppf
from .store import (
    BestSharpe,
    ResearchEntry,
    ResearchRecall,
    ResearchRecord,
    ResearchStore,
    best_sharpe,
    create_research_store,
)

__all__ = [
    "BestSharpe",
    "CpcvSplit",
    "Moments",
    "MonteCarloResult",
    "PathPercentileBands",
    "PboResult",
    "PercentileBands",
    "ResearchEntry",
    "ResearchRecall",
    "ResearchRecord",
    "ResearchStore",
    "SweepHaircut",
    "best_sharpe",
    "combinations",
    "combinatorial_purged_splits",
    "create_research_store",
    "deflated_sharpe",
    "moments",
    "monte_carlo",
    "normal_cdf",
    "normal_ppf",
    "probability_of_backtest_overfitting",
    "sweep_haircut",
]
