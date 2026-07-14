"""Alpaca Markets trading and market-data adapter."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from urllib.parse import quote

import httpx

from tradelab.data import normalize_candles
from tradelab.errors import ValidationError

from .base import (
    Account,
    BrokerAdapter,
    Clock,
    OrderReceipt,
    Position,
    as_mapping,
    as_rows,
    number,
    option,
    optional_number,
    system_clock_ms,
)


def _time_ms(value: object) -> int | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return round(
            datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC).timestamp() * 1_000
        )
    except ValueError:
        return None


def _status(value: object) -> str:
    normalized = str(value or "").lower()
    if normalized == "partially_filled":
        return "partially_filled"
    if normalized in {"filled", "rejected", "expired"}:
        return normalized
    if normalized in {"canceled", "cancelled"}:
        return "canceled"
    return "new"


def _receipt(value: object) -> OrderReceipt:
    order = as_mapping(value)
    average = optional_number(order.get("filled_avg_price"))
    return {
        "orderId": str(order.get("id", "")),
        "clientOrderId": str(order["client_order_id"])
        if order.get("client_order_id") is not None
        else None,
        "status": _status(order.get("status")),
        "filledQty": number(order.get("filled_qty")),
        "avgFillPrice": average,
        "filledAt": _time_ms(order.get("filled_at")),
        "symbol": str(order.get("symbol", "")),
        "side": str(order.get("side", "")).lower(),
        "type": str(order.get("type", "")).lower(),
        "qty": number(order.get("qty")),
        "rejectReason": str(order["reject_reason"])
        if order.get("reject_reason") is not None
        else None,
    }


class AlpacaBroker(BrokerAdapter):
    """Async Alpaca REST adapter with paper/live endpoint selection."""

    broker_name = "alpaca"

    def __init__(
        self, *, client: httpx.AsyncClient | None = None, clock: Clock = system_clock_ms
    ) -> None:
        super().__init__(client=client, clock=clock)
        self._api_key = ""
        self._api_secret = ""
        self._base_url = "https://api.alpaca.markets"
        self._data_url = "https://data.alpaca.markets"

    async def connect(self, config: Mapping[str, object] | None = None) -> None:
        values = config or {}
        self._api_key = str(option(values, "api_key", "apiKey", default="") or "").strip()
        self._api_secret = str(option(values, "api_secret", "apiSecret", default="") or "").strip()
        if not self._api_key or not self._api_secret:
            raise ValidationError("Alpaca requires non-empty api_key and api_secret")
        paper = bool(option(values, "paper", default=False))
        default_url = "https://paper-api.alpaca.markets" if paper else "https://api.alpaca.markets"
        self._base_url = str(option(values, "base_url", "baseUrl", default=default_url)).rstrip("/")
        self._data_url = str(
            option(values, "data_url", "dataUrl", default="https://data.alpaca.markets")
        ).rstrip("/")
        await self._open()

    def supports_paper_native(self) -> bool:
        return True

    def _headers(self) -> dict[str, str]:
        return {
            "content-type": "application/json",
            "APCA-API-KEY-ID": self._api_key,
            "APCA-API-SECRET-KEY": self._api_secret,
        }

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, object] | None = None,
        body: object = None,
        data_api: bool = False,
    ) -> object:
        base = self._data_url if data_api else self._base_url
        return await self._request_json(
            method,
            f"{base}{path}",
            headers=self._headers(),
            params=params,
            json=body,
        )

    async def get_account(self) -> Account:
        account = as_mapping(await self._request("GET", "/v2/account"))
        return {
            "equity": number(account.get("equity")),
            "buyingPower": number(account.get("buying_power")),
            "cash": number(account.get("cash")),
            "currency": str(account.get("currency") or "USD"),
            "marginUsed": number(account.get("initial_margin")),
        }

    async def get_positions(self) -> list[Position]:
        return [
            {
                "symbol": str(row.get("symbol", "")),
                "side": str(row.get("side") or "long").lower(),
                "qty": number(row.get("qty")),
                "avgEntry": number(row.get("avg_entry_price")),
                "marketValue": number(row.get("market_value")),
                "unrealizedPnl": number(row.get("unrealized_pl")),
            }
            for row in as_rows(await self._request("GET", "/v2/positions"))
        ]

    async def get_server_time(self) -> int:
        clock = as_mapping(await self._request("GET", "/v2/clock"))
        return _time_ms(clock.get("timestamp")) or self._clock()

    async def submit_order(self, order: Mapping[str, object]) -> OrderReceipt:
        body: dict[str, object] = {
            "symbol": order.get("symbol"),
            "side": order.get("side"),
            "type": order.get("type"),
            "qty": str(order.get("qty")),
            "time_in_force": option(order, "time_in_force", "timeInForce", default="day"),
            "client_order_id": option(order, "client_order_id", "clientOrderId"),
        }
        for source, target in (("limit_price", "limit_price"), ("stop_price", "stop_price")):
            value = option(order, source, "limitPrice" if source == "limit_price" else "stopPrice")
            if value is not None:
                body[target] = str(value)
        receipt = _receipt(await self._request("POST", "/v2/orders", body=body))
        await self._emit("order:submitted", dict(receipt))
        return receipt

    async def cancel_order(self, order_id: object) -> None:
        await self._request("DELETE", f"/v2/orders/{quote(str(order_id), safe='')}")
        await self._emit("order:canceled", {"orderId": str(order_id)})

    async def modify_order(self, order_id: object, changes: Mapping[str, object]) -> OrderReceipt:
        body: dict[str, str] = {}
        for aliases, target in (
            (("qty",), "qty"),
            (("limit_price", "limitPrice"), "limit_price"),
            (("stop_price", "stopPrice"), "stop_price"),
        ):
            value = option(changes, *aliases)
            if value is not None:
                body[target] = str(value)
        result = await self._request(
            "PATCH", f"/v2/orders/{quote(str(order_id), safe='')}", body=body
        )
        receipt = _receipt(result)
        await self._emit("order:modified", dict(receipt))
        return receipt

    async def get_open_orders(self) -> list[OrderReceipt]:
        rows = as_rows(await self._request("GET", "/v2/orders", params={"status": "open"}))
        return [_receipt(row) for row in rows]

    async def get_order_status(self, order_id: object) -> OrderReceipt:
        result = await self._request("GET", f"/v2/orders/{quote(str(order_id), safe='')}")
        return _receipt(result)

    async def get_historical_bars(
        self, symbol: str, interval: str, limit: int = 200
    ) -> list[dict[str, int | float]]:
        remaining = max(0, int(limit))
        token: str | None = None
        seen_tokens: set[str] = set()
        bars: list[dict[str, object]] = []
        while remaining > 0:
            params: dict[str, object] = {"timeframe": interval, "limit": remaining}
            if token:
                params["page_token"] = token
            payload = as_mapping(
                await self._request(
                    "GET",
                    f"/v2/stocks/{quote(symbol, safe='')}/bars",
                    params=params,
                    data_api=True,
                )
            )
            page = as_rows(payload.get("bars"))
            for bar in page[:remaining]:
                timestamp = _time_ms(bar.get("t"))
                if timestamp is not None:
                    bars.append(
                        {
                            "time": timestamp,
                            "open": bar.get("o"),
                            "high": bar.get("h"),
                            "low": bar.get("l"),
                            "close": bar.get("c"),
                            "volume": bar.get("v", 0),
                        }
                    )
            remaining = limit - len(bars)
            next_token = payload.get("next_page_token")
            next_value = str(next_token) if next_token else None
            if not next_value or next_value in seen_tokens or not page:
                break
            seen_tokens.add(next_value)
            token = next_value
        return normalize_candles(bars)


def create_alpaca_broker(
    *, client: httpx.AsyncClient | None = None, clock: Clock = system_clock_ms
) -> AlpacaBroker:
    """Create an Alpaca adapter from constructor options."""
    return AlpacaBroker(client=client, clock=clock)


__all__ = ["AlpacaBroker", "create_alpaca_broker"]
