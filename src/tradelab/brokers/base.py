"""Typed broker contracts and transport-independent lifecycle helpers."""

from __future__ import annotations

import inspect
import math
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from typing import Any, NotRequired, TypedDict

import httpx

from tradelab.errors import BrokerError

Clock = Callable[[], int]
EventHandler = Callable[[dict[str, Any]], object]
Unsubscribe = Callable[[], None]
HttpParam = str | int | float | bool | None


class Account(TypedDict):
    equity: float
    buyingPower: float
    cash: float
    currency: str
    marginUsed: float


class Position(TypedDict):
    symbol: str
    side: str
    qty: float
    avgEntry: float
    marketValue: float
    unrealizedPnl: float


class OrderReceipt(TypedDict):
    orderId: str
    clientOrderId: NotRequired[str | None]
    status: str
    filledQty: float
    avgFillPrice: NotRequired[float | None]
    filledAt: NotRequired[int | None]
    symbol: str
    side: str
    type: str
    qty: float
    rejectReason: NotRequired[str | None]
    limitPrice: NotRequired[float]
    stopPrice: NotRequired[float]


def system_clock_ms() -> int:
    return round(time.time() * 1_000)


def option(config: Mapping[str, object], *names: str, default: object = None) -> object:
    for name in names:
        if name in config:
            return config[name]
    return default


def number(value: object, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return default
    return parsed if math.isfinite(parsed) else default


def optional_number(value: object) -> float | None:
    """Coerce a finite provider number, preserving absent/invalid values as ``None``."""
    if value is None or value == "":
        return None
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) else None


def as_mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def as_rows(value: object) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        return []
    return [row for row in value if isinstance(row, Mapping)]


def _provider_message(payload: object) -> str | None:
    row = as_mapping(payload)
    error_response = as_mapping(row.get("error_response"))
    for value in (
        error_response.get("message"),
        row.get("message"),
        row.get("error"),
        row.get("msg"),
    ):
        if isinstance(value, str) and value:
            return value
    return None


class BrokerAdapter(ABC):
    """Base async broker interface with local event and subscription support."""

    broker_name = "broker"

    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        clock: Clock = system_clock_ms,
    ) -> None:
        self._client = client
        self._owns_client = False
        self._clock = clock
        self._connected = False
        self._events: dict[str, list[EventHandler]] = {}
        self._subscriptions: dict[str, dict[str, list[EventHandler]]] = {
            "bars": {},
            "quotes": {},
            "trades": {},
        }

    @abstractmethod
    async def connect(self, config: Mapping[str, object] | None = None) -> None:
        """Configure credentials/endpoints and establish the adapter lifecycle."""

    @abstractmethod
    async def get_account(self) -> Mapping[str, Any]:
        """Return normalized account balances."""

    @abstractmethod
    async def get_positions(self) -> Sequence[Mapping[str, Any]]:
        """Return normalized open positions."""

    @abstractmethod
    async def submit_order(self, order: Mapping[str, object]) -> Mapping[str, Any]:
        """Submit an order and return its normalized receipt."""

    @abstractmethod
    async def cancel_order(self, order_id: object) -> None:
        """Cancel an order when it is still open."""

    @abstractmethod
    async def modify_order(
        self, order_id: object, changes: Mapping[str, object]
    ) -> Mapping[str, Any]:
        """Modify an existing order and return its receipt."""

    @abstractmethod
    async def get_open_orders(self) -> Sequence[Mapping[str, Any]]:
        """Return currently open orders."""

    @abstractmethod
    async def get_order_status(self, order_id: object) -> Mapping[str, Any]:
        """Return one order's normalized status."""

    @abstractmethod
    async def get_historical_bars(
        self, symbol: str, interval: str, limit: int = 200
    ) -> Sequence[Mapping[str, object]]:
        """Return normalized OHLCV bars."""

    async def _open(self) -> None:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
            self._owns_client = True
        self._connected = True

    async def disconnect(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
        self._client = None if self._owns_client else self._client
        self._owns_client = False
        self._connected = False
        for subscriptions in self._subscriptions.values():
            subscriptions.clear()

    def is_connected(self) -> bool:
        return self._connected

    def supports_paper_native(self) -> bool:
        return False

    def on(self, event: str, handler: EventHandler) -> Unsubscribe:
        self._events.setdefault(event, []).append(handler)

        def unsubscribe() -> None:
            self.off(event, handler)

        return unsubscribe

    def off(self, event: str, handler: EventHandler) -> None:
        self._events[event] = [item for item in self._events.get(event, []) if item is not handler]

    async def _emit(self, event: str, payload: dict[str, Any]) -> None:
        for handler in tuple(self._events.get(event, ())):
            result = handler(payload)
            if inspect.isawaitable(result):
                await result

    def _subscribe(self, channel: str, key: str, handler: EventHandler) -> Unsubscribe:
        subscriptions = self._subscriptions[channel]
        subscriptions.setdefault(key, []).append(handler)

        def unsubscribe() -> None:
            subscriptions[key] = [
                item for item in subscriptions.get(key, []) if item is not handler
            ]

        return unsubscribe

    async def subscribe_quotes(self, symbol: str, handler: EventHandler) -> Unsubscribe:
        return self._subscribe("quotes", symbol, handler)

    async def subscribe_trades(self, symbol: str, handler: EventHandler) -> Unsubscribe:
        return self._subscribe("trades", symbol, handler)

    async def subscribe_bars(
        self, symbol: str, interval: str, handler: EventHandler
    ) -> Unsubscribe:
        return self._subscribe("bars", f"{symbol}::{interval}", handler)

    async def _publish(self, channel: str, key: str, payload: dict[str, Any]) -> None:
        for handler in tuple(self._subscriptions[channel].get(key, ())):
            result = handler(payload)
            if inspect.isawaitable(result):
                await result

    async def publish_quote(self, symbol: str, payload: dict[str, Any]) -> None:
        await self._publish("quotes", symbol, payload)

    async def publish_trade(self, symbol: str, payload: dict[str, Any]) -> None:
        await self._publish("trades", symbol, payload)

    async def publish_bar(self, symbol: str, interval: str, payload: dict[str, Any]) -> None:
        await self._publish("bars", f"{symbol}::{interval}", payload)

    def _require_client(self) -> httpx.AsyncClient:
        if not self._connected or self._client is None:
            raise BrokerError(f"{self.__class__.__name__} is not connected")
        return self._client

    async def _request_json(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        params: Mapping[str, object] | list[tuple[str, HttpParam]] | None = None,
        json: object = None,
    ) -> object:
        client = self._require_client()
        clean_params: list[tuple[str, HttpParam]] | None = None
        if params is not None:
            items = params.items() if isinstance(params, Mapping) else params
            clean_params = [
                (
                    key,
                    value if isinstance(value, (str, int, float, bool)) else str(value),
                )
                for key, value in items
                if value is not None
            ]
        try:
            response = await client.request(
                method,
                url,
                headers=headers,
                params=clean_params,
                json=json,
            )
        except httpx.HTTPError as error:
            raise BrokerError(
                f"{self.broker_name} {method} request failed: {error}",
                context={"broker": self.broker_name, "method": method, "url": url},
            ) from error
        try:
            payload: object = response.json() if response.content else {}
        except ValueError as error:
            raise BrokerError(
                f"{self.broker_name} returned invalid JSON",
                context={
                    "broker": self.broker_name,
                    "method": method,
                    "url": url,
                    "status_code": response.status_code,
                },
            ) from error
        if not response.is_success:
            message = _provider_message(payload) or (
                f"{self.broker_name} request failed ({response.status_code})"
            )
            raise BrokerError(
                message,
                context={
                    "broker": self.broker_name,
                    "method": method,
                    "url": url,
                    "status_code": response.status_code,
                },
            )
        return payload


__all__ = [
    "Account",
    "BrokerAdapter",
    "Clock",
    "EventHandler",
    "OrderReceipt",
    "Position",
    "Unsubscribe",
]
