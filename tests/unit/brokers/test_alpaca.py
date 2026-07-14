"""Alpaca broker lifecycle, wire protocol, normalization, and failure contracts."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from tradelab.brokers import AlpacaBroker
from tradelab.errors import BrokerError, ValidationError


def _response(request: httpx.Request, payload: object = None, status: int = 200) -> httpx.Response:
    return httpx.Response(status, json={} if payload is None else payload, request=request)


@pytest.mark.asyncio
async def test_alpaca_requires_credentials_and_preserves_injected_client() -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: _response(request))
    ) as client:
        broker = AlpacaBroker(client=client)
        with pytest.raises(ValidationError, match=r"api_key.*api_secret"):
            await broker.connect({"paper": True})
        await broker.connect({"api_key": "key", "api_secret": "secret", "paper": True})
        assert broker.is_connected()
        assert broker.supports_paper_native()
        await broker.disconnect()
        assert not broker.is_connected()
        assert not client.is_closed


@pytest.mark.asyncio
async def test_alpaca_account_positions_clock_and_exact_auth_requests() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/v2/account":
            return _response(
                request,
                {
                    "equity": "10000.5",
                    "buying_power": "8000",
                    "cash": "5000",
                    "currency": "USD",
                    "initial_margin": "250",
                },
            )
        if request.url.path == "/v2/positions":
            return _response(
                request,
                [
                    {
                        "symbol": "AAPL",
                        "side": "LONG",
                        "qty": "2",
                        "avg_entry_price": "100",
                        "market_value": "220",
                        "unrealized_pl": "20",
                    }
                ],
            )
        return _response(request, {"timestamp": "2025-01-02T14:30:00Z"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        broker = AlpacaBroker(client=client, clock=lambda: 7)
        await broker.connect({"api_key": "key", "api_secret": "secret", "paper": True})
        account = await broker.get_account()
        assert account["equity"] == 10000.5
        assert account["buyingPower"] == 8000.0
        assert account["marginUsed"] == 250.0
        positions = await broker.get_positions()
        assert positions[0]["avgEntry"] == 100.0
        assert positions[0]["marketValue"] == 220.0
        assert positions[0]["unrealizedPnl"] == 20.0
        assert await broker.get_server_time() == 1_735_828_200_000

    assert [request.url.host for request in requests] == [
        "paper-api.alpaca.markets",
        "paper-api.alpaca.markets",
        "paper-api.alpaca.markets",
    ]
    for request in requests:
        assert request.headers["APCA-API-KEY-ID"] == "key"
        assert request.headers["APCA-API-SECRET-KEY"] == "secret"


@pytest.mark.asyncio
async def test_alpaca_order_methods_emit_and_normalize_exact_bodies() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "DELETE":
            return httpx.Response(204, request=request)
        payload = {
            "id": "123",
            "client_order_id": "client-1",
            "status": "partially_filled",
            "filled_qty": "0.5",
            "filled_avg_price": "101.25",
            "filled_at": "2025-01-02T14:30:00Z",
            "symbol": "AAPL",
            "side": "buy",
            "type": "limit",
            "qty": "1",
            "reject_reason": None,
        }
        if request.url.params.get("status") == "open":
            return _response(request, [payload])
        return _response(request, payload)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        broker = AlpacaBroker(client=client)
        await broker.connect({"api_key": "key", "api_secret": "secret", "paper": True})
        emitted: list[dict[str, Any]] = []
        broker.on("order:submitted", emitted.append)
        receipt = await broker.submit_order(
            {
                "symbol": "AAPL",
                "side": "buy",
                "type": "limit",
                "qty": 1,
                "limit_price": 100,
                "stop_price": 99,
                "time_in_force": "gtc",
                "client_order_id": "client-1",
            }
        )
        modified = await broker.modify_order("123", {"qty": 2, "limit_price": 102})
        await broker.cancel_order("123")
        opened = await broker.get_open_orders()
        status = await broker.get_order_status("123")

    assert receipt["status"] == "partially_filled"
    assert receipt["avgFillPrice"] == 101.25
    assert modified["orderId"] == "123"
    assert opened == [status]
    assert emitted == [receipt]
    assert json.loads(requests[0].content) == {
        "symbol": "AAPL",
        "side": "buy",
        "type": "limit",
        "qty": "1",
        "time_in_force": "gtc",
        "client_order_id": "client-1",
        "limit_price": "100",
        "stop_price": "99",
    }
    assert requests[1].method == "PATCH"
    assert json.loads(requests[1].content) == {"qty": "2", "limit_price": "102"}
    assert requests[2].method == "DELETE"
    assert dict(requests[3].url.params) == {"status": "open"}


@pytest.mark.asyncio
async def test_alpaca_is_live_adapter_and_accepts_trading_session_camel_case_orders() -> None:
    captured: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return _response(
            request,
            {
                "id": "camel-1",
                "client_order_id": "session-1",
                "status": "new",
                "symbol": "AAPL",
                "side": "buy",
                "type": "limit",
                "qty": "1",
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        broker = AlpacaBroker(client=client)
        await broker.connect({"apiKey": "key", "apiSecret": "secret", "paper": True})
        receipt = await broker.submit_order(
            {
                "symbol": "AAPL",
                "side": "buy",
                "type": "limit",
                "qty": 1,
                "limitPrice": 100,
                "clientOrderId": "session-1",
            }
        )

    assert receipt["orderId"] == "camel-1"
    assert receipt["clientOrderId"] == "session-1"
    assert captured[0]["limit_price"] == "100"


@pytest.mark.asyncio
async def test_alpaca_historical_bars_paginate_on_data_api_and_subscriptions_unsubscribe() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.params.get("page_token"):
            return _response(
                request,
                {"bars": [{"t": "2025-01-02T14:31:00Z", "o": 2, "h": 3, "l": 1, "c": 2, "v": 8}]},
            )
        return _response(
            request,
            {
                "bars": [{"t": "2025-01-02T14:30:00Z", "o": 1, "h": 2, "l": 0, "c": 1, "v": 7}],
                "next_page_token": "next",
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        broker = AlpacaBroker(client=client)
        await broker.connect({"api_key": "key", "api_secret": "secret"})
        bars = await broker.get_historical_bars("AAPL", "1Min", 2)
        seen: list[object] = []
        unsubscribe = await broker.subscribe_quotes("AAPL", seen.append)
        await broker.publish_quote("AAPL", {"bid": 1})
        unsubscribe()
        await broker.publish_quote("AAPL", {"bid": 2})

    assert [bar["time"] for bar in bars] == [1_735_828_200_000, 1_735_828_260_000]
    assert seen == [{"bid": 1}]
    assert requests[0].url.host == "data.alpaca.markets"
    assert dict(requests[0].url.params) == {"timeframe": "1Min", "limit": "2"}
    assert dict(requests[1].url.params) == {
        "timeframe": "1Min",
        "limit": "1",
        "page_token": "next",
    }


@pytest.mark.asyncio
async def test_alpaca_wraps_provider_and_transport_errors() -> None:
    def rejected(request: httpx.Request) -> httpx.Response:
        return _response(request, {"message": "insufficient buying power"}, 422)

    async with httpx.AsyncClient(transport=httpx.MockTransport(rejected)) as client:
        broker = AlpacaBroker(client=client)
        await broker.connect({"api_key": "key", "api_secret": "secret"})
        with pytest.raises(BrokerError, match="insufficient buying power") as captured:
            await broker.get_account()
        assert captured.value.context["status_code"] == 422

    def broken(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(broken)) as client:
        broker = AlpacaBroker(client=client)
        await broker.connect({"api_key": "key", "api_secret": "secret"})
        with pytest.raises(BrokerError, match="offline"):
            await broker.get_account()
