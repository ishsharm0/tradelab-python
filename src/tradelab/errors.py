"""Exceptions raised by TradeLab."""

from __future__ import annotations

from collections.abc import Mapping


class TradeLabError(Exception):
    """Base exception for expected TradeLab failures."""

    def __init__(self, message: str, *, context: Mapping[str, object] | None = None) -> None:
        self.message = message
        self.context = dict(context) if context is not None else {}
        super().__init__(message)


class ValidationError(TradeLabError):
    """Raised when a public value does not meet its contract."""


class StrategyError(TradeLabError):
    """Raised when a strategy cannot produce or process a signal."""


class DataProviderError(TradeLabError):
    """Raised when a market-data provider fails."""


class BrokerError(TradeLabError):
    """Raised when a broker operation fails."""


class RiskRejectedError(TradeLabError):
    """Raised when risk controls reject a trading action."""


class LiveTradingDisabledError(TradeLabError):
    """Raised when an operation requires live-trading permission."""
