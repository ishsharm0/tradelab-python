"""Typed broker adapter contract."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from .events import EventBus


def _missing(owner: str, method: str) -> NotImplementedError:
    return NotImplementedError(f"{owner}.{method}() not implemented")


class BrokerAdapter(EventBus):
    """Base interface implemented by paper and credentialed brokers."""

    async def connect(self, config: Mapping[str, object] | None = None) -> None:
        raise _missing("BrokerAdapter", "connect")

    async def disconnect(self) -> None:
        raise _missing("BrokerAdapter", "disconnect")

    def is_connected(self) -> bool:
        raise _missing("BrokerAdapter", "isConnected")

    async def get_account(self) -> dict[str, Any]:
        raise _missing("BrokerAdapter", "getAccount")

    async def get_positions(self) -> list[dict[str, Any]]:
        raise _missing("BrokerAdapter", "getPositions")

    async def get_server_time(self) -> int:
        return time.time_ns() // 1_000_000

    async def submit_order(self, order: Mapping[str, object]) -> dict[str, Any]:
        raise _missing("BrokerAdapter", "submitOrder")

    async def cancel_order(self, order_id: str) -> None:
        raise _missing("BrokerAdapter", "cancelOrder")

    async def modify_order(self, order_id: str, changes: Mapping[str, object]) -> dict[str, Any]:
        raise _missing("BrokerAdapter", "modifyOrder")

    async def get_open_orders(self) -> list[dict[str, Any]]:
        raise _missing("BrokerAdapter", "getOpenOrders")

    async def get_order_status(self, order_id: str) -> dict[str, Any]:
        raise _missing("BrokerAdapter", "getOrderStatus")

    async def subscribe_quotes(
        self, symbol: str, handler: Callable[[dict[str, Any]], object]
    ) -> Callable[[], None]:
        raise _missing("BrokerAdapter", "subscribeQuotes")

    async def subscribe_trades(
        self, symbol: str, handler: Callable[[dict[str, Any]], object]
    ) -> Callable[[], None]:
        raise _missing("BrokerAdapter", "subscribeTrades")

    async def subscribe_bars(
        self, symbol: str, interval: str, handler: Callable[[dict[str, Any]], object]
    ) -> Callable[[], None]:
        raise _missing("BrokerAdapter", "subscribeBars")

    async def get_historical_bars(
        self, symbol: str, interval: str, limit: int = 200
    ) -> Sequence[Mapping[str, object]]:
        raise _missing("BrokerAdapter", "getHistoricalBars")

    def supports_paper_native(self) -> bool:
        return False


@runtime_checkable
class SessionBroker(Protocol):
    """Structural broker boundary consumed by trading sessions."""

    async def connect(self, config: Mapping[str, object] | None = None) -> None: ...

    async def disconnect(self) -> None: ...

    def is_connected(self) -> bool: ...

    def on(self, event: str, handler: Callable[[dict[str, Any]], object]) -> Callable[[], None]: ...

    async def get_account(self) -> Mapping[str, Any]: ...

    async def get_positions(self) -> Sequence[Mapping[str, Any]]: ...

    async def submit_order(self, order: Mapping[str, object]) -> Mapping[str, Any]: ...

    async def cancel_order(self, order_id: str) -> None: ...

    async def get_open_orders(self) -> Sequence[Mapping[str, Any]]: ...
