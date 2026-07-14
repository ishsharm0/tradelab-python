"""Coinbase Advanced Trade REST adapter with deterministic JWT authentication."""

from __future__ import annotations

import base64
import binascii
import secrets
import uuid
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote, urlsplit

import httpx
import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec, ed25519

from tradelab.data import normalize_candles
from tradelab.errors import BrokerError, ValidationError

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

UuidFactory = Callable[[], str]
NonceFactory = Callable[[], str]
PrivateKey = ec.EllipticCurvePrivateKey | ed25519.Ed25519PrivateKey
JwtSigner = Callable[[Mapping[str, object], PrivateKey, str, Mapping[str, object]], str]


def _sign(
    payload: Mapping[str, object],
    private_key: PrivateKey,
    algorithm: str,
    headers: Mapping[str, object],
) -> str:
    return jwt.encode(dict(payload), private_key, algorithm=algorithm, headers=dict(headers))


def _private_key(secret: str) -> tuple[PrivateKey, str]:
    value = secret.replace("\\n", "\n") if "-----BEGIN" in secret else secret
    if value.lstrip().startswith("-----BEGIN"):
        try:
            key = serialization.load_pem_private_key(value.encode(), password=None)
        except (TypeError, ValueError) as error:
            raise ValidationError("Coinbase api_secret is not a valid private key") from error
    else:
        try:
            raw = base64.b64decode("".join(value.split()), validate=True)
        except (binascii.Error, ValueError) as error:
            raise ValidationError(
                "Coinbase api_secret must be an EC/Ed25519 PEM or base64 Ed25519 key"
            ) from error
        if len(raw) not in {32, 64}:
            raise ValidationError("Coinbase Ed25519 api_secret must decode to 32 or 64 bytes")
        key = ed25519.Ed25519PrivateKey.from_private_bytes(raw[:32])
    if isinstance(key, ed25519.Ed25519PrivateKey):
        return key, "EdDSA"
    if isinstance(key, ec.EllipticCurvePrivateKey) and isinstance(key.curve, ec.SECP256R1):
        return key, "ES256"
    raise ValidationError("Coinbase api_secret must use P-256 ECDSA or Ed25519")


def _time_ms(value: object) -> int | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)
    except ValueError:
        return None
    return round(timestamp.timestamp() * 1_000)


def _status(value: object) -> str:
    normalized = str(value or "").upper()
    if "PARTIALLY" in normalized:
        return "partially_filled"
    if "FILLED" in normalized:
        return "filled"
    if "CANCEL" in normalized:
        return "canceled"
    if "REJECT" in normalized:
        return "rejected"
    if "EXPIRE" in normalized:
        return "expired"
    return "new"


def _receipt(value: object, *, fallback_id: object = "") -> OrderReceipt:
    row = as_mapping(value)
    return {
        "orderId": str(row.get("order_id") or fallback_id),
        "clientOrderId": str(row["client_order_id"])
        if row.get("client_order_id") is not None
        else None,
        "status": _status(row.get("status")),
        "filledQty": number(row.get("filled_size")),
        "avgFillPrice": optional_number(row.get("average_filled_price")),
        "filledAt": _time_ms(row.get("last_fill_time")),
        "symbol": str(row.get("product_id", "")),
        "side": str(row.get("side", "")).lower(),
        "type": str(row.get("order_type", "")).lower(),
        "qty": number(row.get("base_size")),
        "rejectReason": str(row["reject_reason"]) if row.get("reject_reason") is not None else None,
    }


class CoinbaseBroker(BrokerAdapter):
    """Async Coinbase Advanced Trade adapter."""

    broker_name = "coinbase"

    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None = None,
        clock: Clock = system_clock_ms,
        signer: JwtSigner = _sign,
        uuid_factory: UuidFactory = lambda: str(uuid.uuid4()),
        nonce_factory: NonceFactory = secrets.token_hex,
    ) -> None:
        super().__init__(client=client, clock=clock)
        self._signer = signer
        self._uuid_factory = uuid_factory
        self._nonce_factory = nonce_factory
        self._api_key = ""
        self._api_secret = ""
        self._private_key: PrivateKey | None = None
        self._algorithm = ""
        self._quote_currency = "USD"
        self._base_url = "https://api.coinbase.com/api/v3/brokerage"

    async def connect(self, config: Mapping[str, object] | None = None) -> None:
        values = config or {}
        self._api_key = str(option(values, "api_key", "apiKey", default="") or "").strip()
        self._api_secret = str(option(values, "api_secret", "apiSecret", default="") or "").strip()
        if not self._api_key or not self._api_secret:
            raise ValidationError("Coinbase requires non-empty api_key and api_secret")
        self._private_key, self._algorithm = _private_key(self._api_secret)
        self._quote_currency = str(
            option(values, "quote_currency", "quoteCurrency", default="USD")
        ).upper()
        self._base_url = str(
            option(
                values,
                "base_url",
                "baseUrl",
                default="https://api.coinbase.com/api/v3/brokerage",
            )
        ).rstrip("/")
        await self._open()

    def _jwt(self, method: str, url: str) -> str:
        target = urlsplit(url)
        now = self._clock() // 1_000
        if self._private_key is None:
            raise BrokerError("Coinbase broker is not authenticated")
        return self._signer(
            {
                "iss": "cdp",
                "sub": self._api_key,
                "nbf": now,
                "exp": now + 120,
                "uri": f"{method.upper()} {target.netloc}{target.path}",
            },
            self._private_key,
            self._algorithm,
            {"kid": self._api_key, "nonce": self._nonce_factory()},
        )

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, object] | None = None,
        body: object = None,
    ) -> object:
        url = f"{self._base_url}{path}"
        response = await self._request_json(
            method,
            url,
            headers={
                "content-type": "application/json",
                "Authorization": f"Bearer {self._jwt(method, url)}",
            },
            params=params,
            json=body,
        )
        envelope = as_mapping(response)
        error = as_mapping(envelope.get("error_response"))
        if envelope.get("success") is False or error:
            message = str(error.get("message") or error.get("error") or "Coinbase request failed")
            raise BrokerError(
                message, context={"broker": self.broker_name, "method": method, "path": path}
            )
        return response

    async def _pages(
        self,
        path: str,
        key: str,
        *,
        params: Mapping[str, object] | None = None,
    ) -> list[Mapping[str, Any]]:
        query = dict(params or {})
        rows: list[Mapping[str, Any]] = []
        seen: set[str] = set()
        while True:
            payload = as_mapping(await self._request("GET", path, params=query))
            rows.extend(as_rows(payload.get(key)))
            cursor_value = payload.get("cursor")
            cursor = str(cursor_value) if cursor_value else ""
            if not payload.get("has_next") or not cursor or cursor in seen:
                return rows
            seen.add(cursor)
            query["cursor"] = cursor

    async def get_server_time(self) -> int:
        return self._clock()

    async def get_account(self) -> Account:
        accounts = await self._pages("/accounts", "accounts")
        quote = next(
            (
                row
                for row in accounts
                if str(row.get("currency", "")).upper() == self._quote_currency
            ),
            {},
        )
        available = quote.get("available_balance")
        hold = quote.get("hold")
        cash = number(available.get("value")) if isinstance(available, Mapping) else 0.0
        held = number(hold.get("value")) if isinstance(hold, Mapping) else 0.0
        return {
            "equity": cash + held,
            "buyingPower": cash,
            "cash": cash,
            "currency": self._quote_currency,
            "marginUsed": held,
        }

    async def get_positions(self) -> list[Position]:
        positions: list[Position] = []
        for row in await self._pages("/accounts", "accounts"):
            product_id = row.get("product_id")
            market_value = row.get("market_value")
            if not isinstance(product_id, str) or not product_id or market_value is None:
                continue
            balance = row.get("available_balance")
            qty = number(balance.get("value")) if isinstance(balance, Mapping) else 0.0
            if qty <= 0:
                continue
            positions.append(
                {
                    "symbol": product_id,
                    "side": "long",
                    "qty": qty,
                    "avgEntry": 0.0,
                    "marketValue": number(market_value),
                    "unrealizedPnl": number(row.get("unrealized_pnl")),
                }
            )
        return positions

    async def submit_order(self, order: Mapping[str, object]) -> OrderReceipt:
        order_type = str(order.get("type") or "market").lower()
        provided_client_id = option(order, "client_order_id", "clientOrderId")
        client_id = str(
            provided_client_id if provided_client_id is not None else self._uuid_factory()
        )
        qty = str(order.get("qty"))
        configuration: dict[str, object]
        if order_type == "market":
            configuration = {"market_market_ioc": {"base_size": qty}}
        elif order_type == "limit":
            configuration = {
                "limit_limit_gtc": {
                    "base_size": qty,
                    "limit_price": str(option(order, "limit_price", "limitPrice")),
                }
            }
        else:
            stop = option(order, "stop_price", "stopPrice")
            limit = option(order, "limit_price", "limitPrice", default=stop)
            configuration = {
                "stop_limit_stop_limit_gtc": {
                    "base_size": qty,
                    "stop_price": str(stop),
                    "limit_price": str(limit),
                }
            }
        body = {
            "client_order_id": client_id,
            "product_id": order.get("symbol"),
            "side": str(order.get("side") or "buy").upper(),
            "order_configuration": configuration,
        }
        response = as_mapping(await self._request("POST", "/orders", body=body))
        result = as_mapping(response.get("success_response") or response.get("order"))
        receipt = _receipt(result, fallback_id=response.get("order_id") or client_id)
        receipt["clientOrderId"] = client_id
        receipt["symbol"] = str(order.get("symbol", ""))
        receipt["side"] = str(order.get("side") or "buy").lower()
        receipt["type"] = order_type
        receipt["qty"] = number(order.get("qty"))
        await self._emit("order:submitted", dict(receipt))
        return receipt

    async def cancel_order(self, order_id: object) -> None:
        await self._request("POST", "/orders/batch_cancel", body={"order_ids": [str(order_id)]})
        await self._emit("order:canceled", {"orderId": str(order_id)})

    async def modify_order(self, order_id: object, changes: Mapping[str, object]) -> OrderReceipt:
        body: dict[str, object] = {"order_id": str(order_id)}
        for aliases, target in (
            (("qty",), "size"),
            (("limit_price", "limitPrice"), "limit_price"),
            (("stop_price", "stopPrice"), "stop_price"),
        ):
            value = option(changes, *aliases)
            if value is not None:
                body[target] = str(value)
        response = as_mapping(await self._request("POST", "/orders/edit", body=body))
        receipt = _receipt(response.get("success_response"), fallback_id=order_id)
        await self._emit("order:modified", dict(receipt))
        return receipt

    async def get_open_orders(self) -> list[OrderReceipt]:
        rows = await self._pages(
            "/orders/historical/batch", "orders", params={"order_status": "OPEN"}
        )
        return [_receipt(row) for row in rows]

    async def get_order_status(self, order_id: object) -> OrderReceipt:
        response = as_mapping(
            await self._request("GET", f"/orders/historical/{quote(str(order_id), safe='')}")
        )
        return _receipt(response.get("order"), fallback_id=order_id)

    @staticmethod
    def _granularity(interval: object) -> tuple[str, int]:
        values = {
            "1m": ("ONE_MINUTE", 60),
            "5m": ("FIVE_MINUTE", 300),
            "15m": ("FIFTEEN_MINUTE", 900),
            "30m": ("THIRTY_MINUTE", 1_800),
            "1h": ("ONE_HOUR", 3_600),
            "2h": ("TWO_HOUR", 7_200),
            "4h": ("FOUR_HOUR", 14_400),
            "6h": ("SIX_HOUR", 21_600),
            "1d": ("ONE_DAY", 86_400),
        }
        normalized = str(interval or "1m").lower()
        if normalized not in values:
            raise ValidationError(f'Unsupported Coinbase candle interval "{normalized}"')
        return values[normalized]

    async def get_historical_bars(
        self, symbol: str, interval: str, limit: int = 200
    ) -> list[dict[str, int | float]]:
        if limit <= 0 or limit > 350:
            raise ValidationError("Coinbase candle limit must be between 1 and 350")
        granularity, seconds = self._granularity(interval)
        end = self._clock() // 1_000
        start = end - seconds * limit
        payload = await self._request(
            "GET",
            f"/products/{quote(symbol, safe='')}/candles",
            params={
                "granularity": granularity,
                "start": start,
                "end": end,
                "limit": limit,
            },
        )
        mapping = as_mapping(payload)
        rows: object = mapping.get("candles") if mapping else payload
        values = rows if isinstance(rows, list) else []
        bars: list[dict[str, object]] = []
        for item in values:
            if isinstance(item, Mapping):
                start = item.get("start", item.get("time"))
                bars.append(
                    {
                        "time": number(start) * 1_000,
                        "low": item.get("low"),
                        "high": item.get("high"),
                        "open": item.get("open"),
                        "close": item.get("close"),
                        "volume": item.get("volume", 0),
                    }
                )
            elif isinstance(item, list) and len(item) >= 5:
                bars.append(
                    {
                        "time": number(item[0]) * 1_000,
                        "low": item[1],
                        "high": item[2],
                        "open": item[3],
                        "close": item[4],
                        "volume": item[5] if len(item) > 5 else 0,
                    }
                )
        return normalize_candles(bars)


def create_coinbase_broker(
    *,
    client: httpx.AsyncClient | None = None,
    clock: Clock = system_clock_ms,
    signer: JwtSigner = _sign,
    uuid_factory: UuidFactory = lambda: str(uuid.uuid4()),
    nonce_factory: NonceFactory = secrets.token_hex,
) -> CoinbaseBroker:
    """Create a Coinbase adapter from constructor options."""
    return CoinbaseBroker(
        client=client,
        clock=clock,
        signer=signer,
        uuid_factory=uuid_factory,
        nonce_factory=nonce_factory,
    )


__all__ = ["CoinbaseBroker", "create_coinbase_broker"]
