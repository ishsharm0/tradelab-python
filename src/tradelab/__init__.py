"""TradeLab's public Python API."""

from .engine import (
    BarSystemRunner,
    BudgetExceededError,
    LlmSignal,
    backtest,
    backtest_async,
    backtest_portfolio,
    backtest_ticks,
    financing_cost,
    funding_events,
    grid,
    optimize,
    walk_forward_optimize,
    with_budget,
)
from .errors import (
    BrokerError,
    DataProviderError,
    LiveTradingDisabledError,
    RiskRejectedError,
    StrategyError,
    TradeLabError,
    ValidationError,
)
from .models import BacktestResult, Candle, Signal, to_primitive
from .strategies import get_strategy, list_strategies, register_strategy

__all__ = [
    "BacktestResult",
    "BarSystemRunner",
    "BrokerError",
    "BudgetExceededError",
    "Candle",
    "DataProviderError",
    "LiveTradingDisabledError",
    "LlmSignal",
    "RiskRejectedError",
    "Signal",
    "StrategyError",
    "TradeLabError",
    "ValidationError",
    "backtest",
    "backtest_async",
    "backtest_portfolio",
    "backtest_ticks",
    "financing_cost",
    "funding_events",
    "get_strategy",
    "grid",
    "list_strategies",
    "optimize",
    "register_strategy",
    "to_primitive",
    "walk_forward_optimize",
    "with_budget",
]
