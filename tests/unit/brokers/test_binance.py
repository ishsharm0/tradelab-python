"""Binance spot/futures request signing, endpoints, and normalization contracts."""

from __future__ import annotations

import hashlib
import hmac
from urllib.parse import urlencode

import httpx
import pytest

from tradelab.brokers import BinanceBroker
from tradelab.errors import BrokerError, ValidationError


def _response(request: httpx.Request, payload: object = None, status: int = 200) -> httpx.Response:
    return httpx.Response(status, json={} if payload is None else payload, request=request)


@pytest.mark.asyncio
async def test_binance_spot_testnet_signs_exact_account_request_and_maps_balances() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _response(
            request,
            {"balances": [{"asset": "USDT", "free": "1000"}, {"asset": "BTC", "free": "2"}]},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        broker = BinanceBroker(client=client, clock=lambda: 1_700_000_000_000)
        with pytest.raises(ValidationError, match=r"api_key.*api_secret"):
            await broker.connect({})
        await broker.connect({"apiKey": "key", "apiSecret": "secret", "paper": True})
        account = await broker.get_account()
        positions = await broker.get_positions()

    unsigned = urlencode({"timestamp": 1_700_000_000_000})
    signature = hmac.new(b"secret", unsigned.encode(), hashlib.sha256).hexdigest()
    assert requests[0].url.host == "testnet.binance.vision"
    assert requests[0].url.path == "/api/v3/account"
    assert requests[0].url.query.decode() == f"{unsigned}&signature={signature}"
    assert requests[0].headers["X-MBX-APIKEY"] == "key"
    assert account == {
        "equity": 1000.0,
        "buyingPower": 1000.0,
        "cash": 1000.0,
        "currency": "USDT",
        "marginUsed": 0.0,
    }
    assert positions == []


@pytest.mark.asyncio
async def test_binance_spot_order_lifecycle_uses_signed_query_and_camel_aliases() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "DELETE":
            return _response(request, {"orderId": 7})
        row = {
            "orderId": 7,
            "clientOrderId": "client-7",
            "status": "PARTIALLY_FILLED",
            "executedQty": "0.2",
            "avgPrice": "101",
            "updateTime": 123,
            "transactTime": 122,
            "symbol": "BTCUSDT",
            "side": "BUY",
            "type": "LIMIT",
            "origQty": "1",
            "price": "100",
        }
        if request.method == "PUT":
            return _response(request, {"amendedOrder": {**row, "qty": "0.5"}})
        return _response(request, [row] if request.url.path.endswith("openOrders") else row)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        broker = BinanceBroker(client=client, clock=lambda: 111)
        await broker.connect({"api_key": "key", "api_secret": "secret"})
        receipt = await broker.submit_order(
            {
                "symbol": "BTCUSDT",
                "side": "buy",
                "type": "stop_limit",
                "qty": 1,
                "limitPrice": 100,
                "stopPrice": 99,
                "timeInForce": "gtc",
                "clientOrderId": "client-7",
            }
        )
        modified = await broker.modify_order("7", {"qty": 0.5})
        await broker.cancel_order("7")
        opened = await broker.get_open_orders()
        status = await broker.get_order_status("7")

    submit = dict(requests[0].url.params)
    assert requests[0].method == "POST"
    assert {
        "symbol": "BTCUSDT",
        "side": "BUY",
        "quantity": "1",
        "type": "STOP_LOSS_LIMIT",
        "timeInForce": "GTC",
        "newClientOrderId": "client-7",
        "price": "100",
        "stopPrice": "99",
    }.items() <= submit.items()
    assert receipt["orderId"] == "7"
    assert receipt["status"] == "partially_filled"
    assert receipt["avgFillPrice"] == 101
    assert modified["orderId"] == "7"
    assert modified["qty"] == 0.5
    assert opened == [status]
    assert requests[1].url.path == "/api/v3/order/amend/keepPriority"
    assert {
        "symbol": "BTCUSDT",
        "orderId": "7",
        "newQty": "0.5",
    }.items() <= dict(requests[1].url.params).items()
    assert requests[2].method == "DELETE"
    assert dict(requests[2].url.params)["orderId"] == "7"
    for request in (requests[1], requests[2], requests[4]):
        assert dict(request.url.params)["symbol"] == "BTCUSDT"


@pytest.mark.asyncio
async def test_binance_futures_modify_uses_required_side_quantity_and_price() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        params = dict(request.url.params)
        return _response(
            request,
            {
                "orderId": 17,
                "status": "NEW",
                "symbol": "ETHUSDT",
                "side": "SELL",
                "type": "LIMIT",
                "origQty": params.get("quantity", "2"),
                "price": params.get("price", "2000"),
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        broker = BinanceBroker(client=client, clock=lambda: 111)
        await broker.connect({"api_key": "key", "api_secret": "secret", "futures": True})
        receipt = await broker.submit_order(
            {
                "symbol": "ETHUSDT",
                "side": "sell",
                "type": "limit",
                "qty": 2,
                "limitPrice": 2000,
            }
        )
        modified = await broker.modify_order(receipt, {"qty": 1.5, "limitPrice": 1990})

    assert requests[1].url.path == "/fapi/v1/order"
    assert {
        "symbol": "ETHUSDT",
        "orderId": "17",
        "side": "SELL",
        "quantity": "1.5",
        "price": "1990",
    }.items() <= dict(requests[1].url.params).items()
    assert modified["side"] == "sell"
    assert modified["qty"] == 1.5
    assert modified["limitPrice"] == 1990


@pytest.mark.asyncio
async def test_binance_modify_rejects_fields_unsupported_by_provider_endpoint() -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: _response(request))
    ) as client:
        spot = BinanceBroker(client=client, clock=lambda: 1)
        await spot.connect({"api_key": "key", "api_secret": "secret"})
        with pytest.raises(ValidationError, match="quantity"):
            await spot.modify_order({"orderId": "1", "symbol": "BTCUSDT"}, {})
        with pytest.raises(ValidationError, match="price"):
            await spot.modify_order(
                {"orderId": "1", "symbol": "BTCUSDT"},
                {"qty": 0.5, "limitPrice": 10},
            )

        futures = BinanceBroker(client=client, clock=lambda: 1)
        await futures.connect({"api_key": "key", "api_secret": "secret", "futures": True})
        with pytest.raises(ValidationError, match="side"):
            await futures.modify_order(
                {"orderId": "2", "symbol": "BTCUSDT"},
                {"qty": 1, "limitPrice": 10},
            )


@pytest.mark.asyncio
async def test_binance_market_order_omits_time_in_force_and_normalizes_klines() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("klines"):
            return _response(
                request,
                [
                    [1_700_000_000_000, "1", "2", "0", "1", "3"],
                    [1_700_000_060_000, "100", "102", "99", "101", "10"],
                ],
            )
        if request.url.path.endswith("time"):
            return _response(request, {"serverTime": 456})
        return _response(
            request,
            {
                "orderId": 1,
                "status": "NEW",
                "symbol": "BTCUSDT",
                "side": "BUY",
                "type": "MARKET",
                "origQty": "1",
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        broker = BinanceBroker(client=client, clock=lambda: 111)
        await broker.connect(
            {"api_key": "key", "api_secret": "secret", "base_url": "https://custom.binance"}
        )
        await broker.submit_order({"symbol": "BTCUSDT", "side": "buy", "type": "market", "qty": 1})
        assert await broker.get_server_time() == 456
        bars = await broker.get_historical_bars("BTCUSDT", "1m", 2)

    assert requests[0].url.host == "custom.binance"
    assert "timeInForce" not in requests[0].url.params
    assert dict(requests[2].url.params) == {"symbol": "BTCUSDT", "interval": "1m", "limit": "2"}
    assert [bar["time"] for bar in bars] == [1_700_000_000_000, 1_700_000_060_000]


@pytest.mark.asyncio
async def test_binance_futures_selects_paths_and_maps_short_positions() -> None:
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        if request.url.path.endswith("/time"):
            return _response(request, {})
        if request.url.path.endswith("/account"):
            return _response(
                request,
                {
                    "totalWalletBalance": "500",
                    "availableBalance": "400",
                    "totalPositionInitialMargin": "20",
                },
            )
        if request.url.path.endswith("positionRisk"):
            return _response(
                request,
                [
                    {
                        "symbol": "ETHUSDT",
                        "positionAmt": "-2",
                        "entryPrice": "10",
                        "markPrice": "12",
                        "unRealizedProfit": "-4",
                    },
                    {"symbol": "EMPTY", "positionAmt": "0"},
                ],
            )
        return _response(request, [])

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        broker = BinanceBroker(client=client, clock=lambda: 999)
        await broker.connect(
            {"api_key": "key", "api_secret": "secret", "paper": True, "futures": True}
        )
        assert await broker.get_server_time() == 999
        account = await broker.get_account()
        positions = await broker.get_positions()
        await broker.get_open_orders()
        await broker.get_historical_bars("ETHUSDT", "1m", 1)

    assert account["equity"] == 500
    assert positions == [
        {
            "symbol": "ETHUSDT",
            "side": "short",
            "qty": 2.0,
            "avgEntry": 10.0,
            "marketValue": 24.0,
            "unrealizedPnl": -4.0,
        }
    ]
    assert paths == [
        "/fapi/v1/time",
        "/fapi/v2/account",
        "/fapi/v2/positionRisk",
        "/fapi/v1/openOrders",
        "/fapi/v1/klines",
    ]


@pytest.mark.asyncio
async def test_binance_wraps_provider_errors() -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: _response(request, {"msg": "bad signature"}, 401)
        )
    ) as client:
        broker = BinanceBroker(client=client, clock=lambda: 1)
        await broker.connect({"api_key": "key", "api_secret": "secret"})
        with pytest.raises(BrokerError, match="bad signature") as captured:
            await broker.get_account()
        assert captured.value.context["status_code"] == 401


@pytest.mark.asyncio
async def test_binance_order_references_accept_receipts_and_fail_closed_without_symbol() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _response(
            request,
            {
                "orderId": 9,
                "status": "NEW",
                "symbol": "ETHUSDT",
                "side": "SELL",
                "type": "LIMIT",
                "origQty": "1",
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        broker = BinanceBroker(client=client, clock=lambda: 1)
        await broker.connect({"api_key": "key", "api_secret": "secret"})
        receipt = await broker.get_order_status({"orderId": "9", "symbol": "ETHUSDT"})
        assert receipt["symbol"] == "ETHUSDT"
        with pytest.raises(ValidationError, match="symbol"):
            await broker.get_order_status("unknown")

    assert dict(requests[0].url.params)["symbol"] == "ETHUSDT"


@pytest.mark.asyncio
async def test_binance_maps_protective_stops_to_official_spot_and_futures_types() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _response(
            request,
            {
                "orderId": len(requests),
                "status": "NEW",
                "symbol": "BTCUSDT",
                "side": "SELL",
                "origQty": "1",
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        spot = BinanceBroker(client=client, clock=lambda: 1)
        await spot.connect({"api_key": "key", "api_secret": "secret"})
        await spot.submit_order(
            {"symbol": "BTCUSDT", "side": "sell", "type": "stop", "qty": 1, "stopPrice": 90}
        )
        futures = BinanceBroker(client=client, clock=lambda: 1)
        await futures.connect({"api_key": "key", "api_secret": "secret", "futures": True})
        await futures.submit_order(
            {"symbol": "BTCUSDT", "side": "sell", "type": "stop", "qty": 1, "stopPrice": 90}
        )

    spot_params = dict(requests[0].url.params)
    assert spot_params["type"] == "STOP_LOSS"
    assert "timeInForce" not in spot_params and "price" not in spot_params
    futures_params = dict(requests[1].url.params)
    assert futures_params["type"] == "STOP_MARKET"
    assert "timeInForce" not in futures_params and "price" not in futures_params
