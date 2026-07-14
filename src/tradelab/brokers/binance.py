"""Binance spot and USD-M futures REST adapter."""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import Callable, Mapping, Sequence

import httpx

from tradelab.data import normalize_candles
from tradelab.errors import ValidationError

from .base import (
    Account,
    BrokerAdapter,
    Clock,
    HttpParam,
    OrderReceipt,
    Position,
    as_mapping,
    as_rows,
    number,
    option,
    optional_number,
    system_clock_ms,
)

Signer = Callable[[str, str], str]


def _hmac_sign(secret: str, payload: str) -> str:
    return hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()


def _status(value: object) -> str:
    normalized = str(value or "").upper()
    if normalized == "PARTIALLY_FILLED":
        return "partially_filled"
    if normalized == "FILLED":
        return "filled"
    if normalized in {"CANCELED", "CANCELLED"}:
        return "canceled"
    if normalized == "REJECTED":
        return "rejected"
    if normalized in {"EXPIRED", "EXPIRED_IN_MATCH"}:
        return "expired"
    return "new"


def _receipt(value: object, *, submitted: bool = False) -> OrderReceipt:
    row = as_mapping(value)
    average = optional_number(row.get("avgPrice"))
    time_value = row.get("transactTime" if submitted else "updateTime")
    return {
        "orderId": str(row.get("orderId", "")),
        "clientOrderId": str(row["clientOrderId"])
        if row.get("clientOrderId") is not None
        else None,
        "status": _status(row.get("status")),
        "filledQty": number(row.get("executedQty")),
        "avgFillPrice": average,
        "filledAt": int(number(time_value)) if time_value is not None else None,
        "symbol": str(row.get("symbol", "")),
        "side": str(row.get("side", "")).lower(),
        "type": str(row.get("type", "")).lower(),
        "qty": number(row.get("origQty")),
        "rejectReason": str(row["rejectReason"]) if row.get("rejectReason") is not None else None,
    }


class BinanceBroker(BrokerAdapter):
    """Async Binance adapter supporting spot and USD-M futures endpoints."""

    broker_name = "binance"

    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        clock: Clock = system_clock_ms,
        signer: Signer = _hmac_sign,
    ) -> None:
        super().__init__(client=client, clock=clock)
        self._signer = signer
        self._api_key = ""
        self._api_secret = ""
        self._futures = False
        self._base_url = "https://api.binance.com"

    async def connect(self, config: Mapping[str, object] | None = None) -> None:
        values = config or {}
        self._api_key = str(option(values, "api_key", "apiKey", default="") or "").strip()
        self._api_secret = str(option(values, "api_secret", "apiSecret", default="") or "").strip()
        if not self._api_key or not self._api_secret:
            raise ValidationError("Binance requires non-empty api_key and api_secret")
        self._futures = bool(option(values, "futures", default=False))
        paper = bool(option(values, "paper", default=False))
        if self._futures and paper:
            default_url = "https://testnet.binancefuture.com"
        elif paper:
            default_url = "https://testnet.binance.vision"
        elif self._futures:
            default_url = "https://fapi.binance.com"
        else:
            default_url = "https://api.binance.com"
        self._base_url = str(option(values, "base_url", "baseUrl", default=default_url)).rstrip("/")
        await self._open()

    def supports_paper_native(self) -> bool:
        return True

    def _signed(self, params: Mapping[str, object] | None = None) -> list[tuple[str, HttpParam]]:
        values: list[tuple[str, HttpParam]] = [
            (
                key,
                value if isinstance(value, (str, int, float, bool)) else str(value),
            )
            for key, value in (params or {}).items()
            if value is not None
        ]
        values.append(("timestamp", self._clock()))
        encoded = str(httpx.QueryParams(values))
        values.append(("signature", self._signer(self._api_secret, encoded)))
        return values

    async def _request(
        self,
        method: str,
        path: str,
        *,
        signed: bool = False,
        params: Mapping[str, object] | None = None,
    ) -> object:
        values: Mapping[str, object] | list[tuple[str, HttpParam]] | None
        values = self._signed(params) if signed else params
        return await self._request_json(
            method,
            f"{self._base_url}{path}",
            headers={"content-type": "application/json", "X-MBX-APIKEY": self._api_key},
            params=values,
        )

    def _path(self, spot: str, futures: str) -> str:
        return futures if self._futures else spot

    async def get_server_time(self) -> int:
        path = self._path("/api/v3/time", "/fapi/v1/time")
        payload = as_mapping(await self._request("GET", path))
        return int(number(payload.get("serverTime"), float(self._clock())))

    async def get_account(self) -> Account:
        if self._futures:
            account = as_mapping(await self._request("GET", "/fapi/v2/account", signed=True))
            return {
                "equity": number(account.get("totalWalletBalance")),
                "buyingPower": number(account.get("availableBalance")),
                "cash": number(account.get("availableBalance")),
                "currency": "USDT",
                "marginUsed": number(account.get("totalPositionInitialMargin")),
            }
        account = as_mapping(await self._request("GET", "/api/v3/account", signed=True))
        free = sum(number(row.get("free")) for row in as_rows(account.get("balances")))
        return {
            "equity": free,
            "buyingPower": free,
            "cash": free,
            "currency": "USDT",
            "marginUsed": 0.0,
        }

    async def get_positions(self) -> list[Position]:
        if self._futures:
            rows = as_rows(await self._request("GET", "/fapi/v2/positionRisk", signed=True))
            positions: list[Position] = []
            for row in rows:
                amount = number(row.get("positionAmt"))
                if amount == 0:
                    continue
                positions.append(
                    {
                        "symbol": str(row.get("symbol", "")),
                        "side": "long" if amount >= 0 else "short",
                        "qty": abs(amount),
                        "avgEntry": number(row.get("entryPrice")),
                        "marketValue": abs(amount * number(row.get("markPrice"))),
                        "unrealizedPnl": number(row.get("unRealizedProfit")),
                    }
                )
            return positions
        account = as_mapping(await self._request("GET", "/api/v3/account", signed=True))
        return [
            {
                "symbol": f"{row.get('asset', '')}USDT",
                "side": "long",
                "qty": number(row.get("free")),
                "avgEntry": 0.0,
                "marketValue": number(row.get("free")),
                "unrealizedPnl": 0.0,
            }
            for row in as_rows(account.get("balances"))
            if number(row.get("free")) > 0
        ]

    def _order_params(self, order: Mapping[str, object]) -> dict[str, object]:
        order_type = str(order.get("type") or "market").lower()
        payload: dict[str, object] = {
            "symbol": order.get("symbol"),
            "side": str(order.get("side") or "").upper(),
            "quantity": str(order.get("qty")),
            "type": "STOP_LOSS_LIMIT" if order_type == "stop_limit" else order_type.upper(),
            "timeInForce": str(
                option(order, "time_in_force", "timeInForce", default="GTC")
            ).upper(),
            "newClientOrderId": option(order, "client_order_id", "clientOrderId"),
        }
        limit_price = option(order, "limit_price", "limitPrice")
        stop_price = option(order, "stop_price", "stopPrice")
        if limit_price is not None:
            payload["price"] = str(limit_price)
        if stop_price is not None:
            payload["stopPrice"] = str(stop_price)
        if payload["type"] == "MARKET":
            payload.pop("timeInForce")
        return payload

    async def submit_order(self, order: Mapping[str, object]) -> OrderReceipt:
        path = self._path("/api/v3/order", "/fapi/v1/order")
        result = await self._request("POST", path, signed=True, params=self._order_params(order))
        receipt = _receipt(result, submitted=True)
        await self._emit("order:submitted", dict(receipt))
        return receipt

    async def cancel_order(self, order_id: object) -> None:
        path = self._path("/api/v3/order", "/fapi/v1/order")
        await self._request("DELETE", path, signed=True, params={"orderId": order_id})
        await self._emit("order:canceled", {"orderId": str(order_id)})

    async def modify_order(self, order_id: object, changes: Mapping[str, object]) -> OrderReceipt:
        path = self._path("/api/v3/order", "/fapi/v1/order")
        result = await self._request(
            "PUT",
            path,
            signed=True,
            params={
                "orderId": order_id,
                "quantity": changes.get("qty"),
                "price": option(changes, "limit_price", "limitPrice"),
                "stopPrice": option(changes, "stop_price", "stopPrice"),
            },
        )
        receipt = _receipt(result)
        await self._emit("order:modified", dict(receipt))
        return receipt

    async def get_open_orders(self) -> list[OrderReceipt]:
        path = self._path("/api/v3/openOrders", "/fapi/v1/openOrders")
        return [_receipt(row) for row in as_rows(await self._request("GET", path, signed=True))]

    async def get_order_status(self, order_id: object) -> OrderReceipt:
        path = self._path("/api/v3/order", "/fapi/v1/order")
        return _receipt(await self._request("GET", path, signed=True, params={"orderId": order_id}))

    async def get_historical_bars(
        self, symbol: str, interval: str, limit: int = 200
    ) -> list[dict[str, int | float]]:
        path = self._path("/api/v3/klines", "/fapi/v1/klines")
        payload = await self._request(
            "GET", path, params={"symbol": symbol, "interval": interval, "limit": limit}
        )
        rows = payload if isinstance(payload, list) else []
        bars: list[dict[str, object]] = []
        for row in rows:
            if not isinstance(row, Sequence) or isinstance(row, (str, bytes)) or len(row) < 6:
                continue
            bars.append(
                {
                    "time": row[0],
                    "open": row[1],
                    "high": row[2],
                    "low": row[3],
                    "close": row[4],
                    "volume": row[5],
                }
            )
        return normalize_candles(bars)


def create_binance_broker(
    *,
    client: httpx.AsyncClient | None = None,
    clock: Clock = system_clock_ms,
    signer: Signer = _hmac_sign,
) -> BinanceBroker:
    """Create a Binance adapter from constructor options."""
    return BinanceBroker(client=client, clock=clock, signer=signer)


__all__ = ["BinanceBroker", "create_binance_broker"]
