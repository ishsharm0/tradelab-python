"""Coinbase Advanced Trade JWT, pagination, orders, and candle contracts."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json

import httpx
import pytest

from tradelab.brokers import CoinbaseBroker
from tradelab.errors import BrokerError, ValidationError


def _response(request: httpx.Request, payload: object = None, status: int = 200) -> httpx.Response:
    return httpx.Response(status, json={} if payload is None else payload, request=request)


def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode()


def _jwt(secret: str, method: str, path: str) -> str:
    header = _b64(
        json.dumps({"alg": "HS256", "typ": "JWT", "kid": "key"}, separators=(",", ":")).encode()
    )
    payload = _b64(
        json.dumps(
            {
                "iss": "cdp",
                "sub": "key",
                "nbf": 1_699_999_995,
                "exp": 1_700_000_120,
                "uri": f"{method} api.coinbase.com{path}",
            },
            separators=(",", ":"),
        ).encode()
    )
    signing = f"{header}.{payload}"
    signature = _b64(hmac.new(secret.encode(), signing.encode(), hashlib.sha256).digest())
    return f"{signing}.{signature}"


@pytest.mark.asyncio
async def test_coinbase_requires_credentials_and_builds_exact_jwt_for_paginated_accounts() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.params.get("cursor"):
            return _response(
                request,
                {
                    "accounts": [{"currency": "BTC", "available_balance": {"value": "2"}}],
                    "has_next": False,
                },
            )
        return _response(
            request,
            {
                "accounts": [{"currency": "USD", "available_balance": {"value": "5000"}}],
                "has_next": True,
                "cursor": "next-account",
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        broker = CoinbaseBroker(client=client, clock=lambda: 1_700_000_000_000)
        with pytest.raises(ValidationError, match=r"api_key.*api_secret"):
            await broker.connect({})
        await broker.connect({"apiKey": "key", "apiSecret": "secret"})
        assert not broker.supports_paper_native()
        account = await broker.get_account()

    assert account == {
        "equity": 5002.0,
        "buyingPower": 5002.0,
        "cash": 5002.0,
        "currency": "USD",
        "marginUsed": 0.0,
    }
    assert requests[0].url == "https://api.coinbase.com/api/v3/brokerage/accounts"
    assert (
        requests[0].headers["Authorization"]
        == f"Bearer {_jwt('secret', 'GET', '/api/v3/brokerage/accounts')}"
    )
    assert dict(requests[1].url.params) == {"cursor": "next-account"}


@pytest.mark.asyncio
async def test_coinbase_positions_and_server_time_use_camel_normalization() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return _response(
            request,
            {
                "accounts": [
                    {"currency": "BTC-USD", "available_balance": {"value": "2"}},
                    {"currency": "EMPTY", "available_balance": {"value": "0"}},
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        broker = CoinbaseBroker(client=client, clock=lambda: 123)
        await broker.connect({"api_key": "key", "api_secret": "secret"})
        assert await broker.get_server_time() == 123
        assert await broker.get_positions() == [
            {
                "symbol": "BTCUSD",
                "side": "long",
                "qty": 2.0,
                "avgEntry": 0.0,
                "marketValue": 2.0,
                "unrealizedPnl": 0.0,
            }
        ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("order_type", "prices", "configuration"),
    [
        ("market", {}, {"market_market_ioc": {"base_size": "1"}}),
        (
            "limit",
            {"limitPrice": 100},
            {"limit_limit_gtc": {"base_size": "1", "limit_price": "100"}},
        ),
        (
            "stop_limit",
            {"stop_price": 99},
            {
                "stop_limit_stop_limit_gtc": {
                    "base_size": "1",
                    "stop_price": "99",
                    "limit_price": "99",
                }
            },
        ),
    ],
)
async def test_coinbase_submit_order_configurations_and_receipt(
    order_type: str, prices: dict[str, object], configuration: dict[str, object]
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _response(
            request,
            {
                "success_response": {
                    "order_id": "oid-1",
                    "status": "FILLED",
                    "filled_size": "1",
                    "average_filled_price": "100",
                    "last_fill_time": "2025-01-02T14:30:00Z",
                }
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        broker = CoinbaseBroker(
            client=client, clock=lambda: 1_700_000_000_000, uuid_factory=lambda: "uuid-1"
        )
        await broker.connect({"api_key": "key", "api_secret": "secret"})
        receipt = await broker.submit_order(
            {"symbol": "BTC-USD", "side": "buy", "type": order_type, "qty": 1, **prices}
        )

    body = json.loads(requests[0].content)
    assert body == {
        "client_order_id": "uuid-1",
        "product_id": "BTC-USD",
        "side": "BUY",
        "order_configuration": configuration,
    }
    assert receipt["orderId"] == "oid-1"
    assert receipt["status"] == "filled"
    assert receipt["avgFillPrice"] == 100


@pytest.mark.asyncio
async def test_coinbase_cancel_edit_status_and_paginated_open_orders() -> None:
    requests: list[httpx.Request] = []
    row = {
        "order_id": "oid-1",
        "client_order_id": "client-1",
        "status": "CANCELLED",
        "filled_size": "0",
        "product_id": "BTC-USD",
        "side": "SELL",
        "order_type": "LIMIT",
        "base_size": "1",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("batch_cancel"):
            return _response(request, {})
        if request.url.path.endswith("/edit"):
            return _response(request, {"success_response": row})
        if request.url.path.endswith("/batch"):
            if request.url.params.get("cursor"):
                return _response(request, {"orders": [row]})
            return _response(request, {"orders": [row], "has_next": True, "cursor": "next-order"})
        return _response(request, {"order": row})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        broker = CoinbaseBroker(client=client, clock=lambda: 1)
        await broker.connect({"api_key": "key", "api_secret": "secret"})
        await broker.cancel_order("oid-1")
        modified = await broker.modify_order("oid-1", {"qty": 0, "limitPrice": 0})
        opened = await broker.get_open_orders()
        status = await broker.get_order_status("oid-1")

    assert json.loads(requests[0].content) == {"order_ids": ["oid-1"]}
    assert json.loads(requests[1].content) == {"order_id": "oid-1", "size": "0", "limit_price": "0"}
    assert modified["orderId"] == "oid-1"
    assert len(opened) == 2 and status["status"] == "canceled"
    assert dict(requests[2].url.params) == {"order_status": "OPEN"}
    assert dict(requests[3].url.params) == {"order_status": "OPEN", "cursor": "next-order"}


@pytest.mark.asyncio
async def test_coinbase_candles_interval_conversion_and_provider_errors() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _response(
            request,
            {
                "candles": [
                    {
                        "start": 1_735_828_200,
                        "low": "99",
                        "high": "101",
                        "open": "100",
                        "close": "100.5",
                        "volume": "2",
                    }
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        broker = CoinbaseBroker(client=client, clock=lambda: 1)
        await broker.connect(
            {"api_key": "key", "api_secret": "secret", "baseUrl": "https://custom.coinbase/base"}
        )
        bars = await broker.get_historical_bars("BTC-USD", "2h", 1)
    assert requests[0].url.path == "/base/products/BTC-USD/candles"
    assert dict(requests[0].url.params) == {"granularity": "7200", "limit": "1"}
    assert bars[0]["time"] == 1_735_828_200_000

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: _response(request, {"error_response": {"message": "denied"}}, 403)
        )
    ) as client:
        broker = CoinbaseBroker(client=client, clock=lambda: 1)
        await broker.connect({"api_key": "key", "api_secret": "secret"})
        with pytest.raises(BrokerError, match="denied"):
            await broker.get_account()
