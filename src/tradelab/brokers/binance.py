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
    receipt: OrderReceipt = {
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
        "qty": number(row.get("qty") if row.get("qty") is not None else row.get("origQty")),
        "rejectReason": str(row["rejectReason"]) if row.get("rejectReason") is not None else None,
    }
    limit_price = optional_number(row.get("price"))
    stop_price = optional_number(row.get("stopPrice"))
    if limit_price is not None and limit_price > 0:
        receipt["limitPrice"] = limit_price
    if stop_price is not None and stop_price > 0:
        receipt["stopPrice"] = stop_price
    return receipt


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
        self._quote_currency = "USDT"
        self._order_symbols: dict[str, str] = {}
        self._orders: dict[str, OrderReceipt] = {}

    async def connect(self, config: Mapping[str, object] | None = None) -> None:
        values = config or {}
        self._api_key = str(option(values, "api_key", "apiKey", default="") or "").strip()
        self._api_secret = str(option(values, "api_secret", "apiSecret", default="") or "").strip()
        if not self._api_key or not self._api_secret:
            raise ValidationError("Binance requires non-empty api_key and api_secret")
        self._futures = bool(option(values, "futures", default=False))
        self._quote_currency = str(
            option(values, "quote_currency", "quoteCurrency", default="USDT")
        ).upper()
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
        quote = next(
            (
                row
                for row in as_rows(account.get("balances"))
                if str(row.get("asset", "")).upper() == self._quote_currency
            ),
            {},
        )
        free = number(quote.get("free"))
        locked = number(quote.get("locked"))
        return {
            "equity": free + locked,
            "buyingPower": free,
            "cash": free,
            "currency": self._quote_currency,
            "marginUsed": locked,
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
        # Spot balances are assets, not valued positions. Without product metadata and a
        # mark-price conversion, inventing ``ASSETUSDT`` symbols or USD values is unsafe.
        await self._request("GET", "/api/v3/account", signed=True)
        return []

    def _order_params(self, order: Mapping[str, object]) -> dict[str, object]:
        order_type = str(order.get("type") or "market").lower()
        if order_type == "stop":
            provider_type = "STOP_MARKET" if self._futures else "STOP_LOSS"
        elif order_type == "stop_limit":
            provider_type = "STOP" if self._futures else "STOP_LOSS_LIMIT"
        else:
            provider_type = order_type.upper()
        payload: dict[str, object] = {
            "symbol": order.get("symbol"),
            "side": str(order.get("side") or "").upper(),
            "quantity": str(order.get("qty")),
            "type": provider_type,
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
        if order_type in {"stop", "stop_limit"} and stop_price is None:
            raise ValidationError("Binance stop orders require stop_price")
        if payload["type"] in {"MARKET", "STOP_LOSS", "STOP_MARKET"}:
            payload.pop("timeInForce")
        return payload

    async def submit_order(self, order: Mapping[str, object]) -> OrderReceipt:
        path = self._path("/api/v3/order", "/fapi/v1/order")
        result = await self._request("POST", path, signed=True, params=self._order_params(order))
        receipt = _receipt(result, submitted=True)
        if not receipt["symbol"]:
            receipt["symbol"] = str(order.get("symbol") or "")
        if receipt["orderId"] and receipt["symbol"]:
            self._order_symbols[receipt["orderId"]] = receipt["symbol"]
            self._orders[receipt["orderId"]] = receipt
        await self._emit("order:submitted", dict(receipt))
        return receipt

    def _order_reference(
        self, order_id: object, changes: Mapping[str, object] | None = None
    ) -> tuple[str, str]:
        if isinstance(order_id, Mapping):
            raw_id = option(order_id, "order_id", "orderId")
            raw_symbol = order_id.get("symbol")
        else:
            raw_id = order_id
            raw_symbol = None
        reference = str(raw_id or "").strip()
        symbol = str(
            raw_symbol or option(changes or {}, "symbol") or self._order_symbols.get(reference, "")
        ).strip()
        if not reference:
            raise ValidationError("Binance order reference requires orderId")
        if not symbol:
            raise ValidationError(
                "Binance order reference requires symbol; pass a receipt-like "
                "{'orderId': ..., 'symbol': ...} value"
            )
        return reference, symbol

    async def cancel_order(self, order_id: object) -> None:
        reference, symbol = self._order_reference(order_id)
        path = self._path("/api/v3/order", "/fapi/v1/order")
        await self._request(
            "DELETE", path, signed=True, params={"symbol": symbol, "orderId": reference}
        )
        await self._emit("order:canceled", {"orderId": reference, "symbol": symbol})

    async def modify_order(self, order_id: object, changes: Mapping[str, object]) -> OrderReceipt:
        reference, symbol = self._order_reference(order_id, changes)
        supplied = order_id if isinstance(order_id, Mapping) else {}
        current = self._orders.get(reference, _receipt(supplied))
        quantity = option(changes, "qty", "quantity")
        limit_price = option(changes, "limit_price", "limitPrice", "price")
        stop_price = option(changes, "stop_price", "stopPrice")
        if not self._futures:
            if limit_price is not None or stop_price is not None:
                raise ValidationError(
                    "Binance spot keep-priority amendments cannot change order price"
                )
            if optional_number(quantity) is None or number(quantity) <= 0:
                raise ValidationError("Binance spot amendments require a positive quantity")
            path = "/api/v3/order/amend/keepPriority"
            params: dict[str, object] = {
                "symbol": symbol,
                "orderId": reference,
                "newQty": quantity,
            }
        else:
            side = str(option(changes, "side", default=current.get("side", ""))).upper()
            quantity = quantity if quantity is not None else current.get("qty")
            limit_price = limit_price if limit_price is not None else current.get("limitPrice")
            if side not in {"BUY", "SELL"}:
                raise ValidationError("Binance futures modification requires order side")
            if optional_number(quantity) is None or number(quantity) <= 0:
                raise ValidationError("Binance futures modification requires positive quantity")
            if optional_number(limit_price) is None or number(limit_price) <= 0:
                raise ValidationError("Binance futures modification requires positive limit price")
            if current.get("type") not in {"", "limit"}:
                raise ValidationError("Binance futures only supports modifying limit orders")
            path = "/fapi/v1/order"
            params = {
                "symbol": symbol,
                "orderId": reference,
                "side": side,
                "quantity": quantity,
                "price": limit_price,
            }
        result = await self._request(
            "PUT",
            path,
            signed=True,
            params=params,
        )
        result_row = as_mapping(result)
        receipt = _receipt(result_row.get("amendedOrder", result_row))
        if not receipt["orderId"]:
            receipt["orderId"] = reference
        if not receipt["symbol"]:
            receipt["symbol"] = symbol
        if not receipt["side"]:
            receipt["side"] = current.get("side", "")
        if not receipt["type"]:
            receipt["type"] = current.get("type", "")
        if receipt["qty"] <= 0 and quantity is not None:
            receipt["qty"] = number(quantity)
        if "limitPrice" not in receipt and limit_price is not None:
            receipt["limitPrice"] = number(limit_price)
        if receipt["orderId"]:
            self._order_symbols[receipt["orderId"]] = receipt["symbol"]
            self._orders[receipt["orderId"]] = receipt
        await self._emit("order:modified", dict(receipt))
        return receipt

    async def get_open_orders(self) -> list[OrderReceipt]:
        path = self._path("/api/v3/openOrders", "/fapi/v1/openOrders")
        receipts = [_receipt(row) for row in as_rows(await self._request("GET", path, signed=True))]
        for receipt in receipts:
            if receipt["orderId"] and receipt["symbol"]:
                self._order_symbols[receipt["orderId"]] = receipt["symbol"]
                self._orders[receipt["orderId"]] = receipt
        return receipts

    async def get_order_status(self, order_id: object) -> OrderReceipt:
        reference, symbol = self._order_reference(order_id)
        path = self._path("/api/v3/order", "/fapi/v1/order")
        receipt = _receipt(
            await self._request(
                "GET", path, signed=True, params={"symbol": symbol, "orderId": reference}
            )
        )
        if not receipt["symbol"]:
            receipt["symbol"] = symbol
        self._order_symbols[receipt["orderId"] or reference] = receipt["symbol"]
        self._orders[receipt["orderId"] or reference] = receipt
        return receipt

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
