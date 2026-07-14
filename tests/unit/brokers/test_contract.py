"""Live-session structural interoperability for credentialed broker adapters."""

from __future__ import annotations

import httpx
import pytest

from tradelab.brokers import (
    AlpacaBroker,
    BinanceBroker,
    CoinbaseBroker,
    InteractiveBrokersBroker,
    create_alpaca_broker,
    create_binance_broker,
    create_coinbase_broker,
    create_interactive_brokers_broker,
)
from tradelab.errors import BrokerError
from tradelab.live import SessionBroker, TradingSession


def test_all_brokers_satisfy_live_session_protocol_without_optional_imports() -> None:
    brokers = [
        AlpacaBroker(),
        BinanceBroker(),
        CoinbaseBroker(),
        InteractiveBrokersBroker(),
    ]
    assert all(isinstance(broker, SessionBroker) for broker in brokers)
    for broker in brokers:
        assert broker.supports_order_updates() is False
        session = TradingSession(symbol="AAPL", broker=broker)
        assert session.broker is broker


def test_provider_factories_return_fresh_typed_adapters() -> None:
    assert isinstance(create_alpaca_broker(), AlpacaBroker)
    assert isinstance(create_binance_broker(), BinanceBroker)
    assert isinstance(create_coinbase_broker(), CoinbaseBroker)
    assert isinstance(create_interactive_brokers_broker(), InteractiveBrokersBroker)


@pytest.mark.asyncio
async def test_rest_broker_fails_closed_when_session_requires_fill_updates() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v2/account":
            payload: object = {"equity": "10000", "buying_power": "8000", "cash": "5000"}
        elif request.url.path in {"/v2/positions", "/v2/orders"}:
            payload = []
        else:
            payload = {}
        return httpx.Response(200, json=payload, request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        broker = AlpacaBroker(client=client)
        await broker.connect({"api_key": "key", "api_secret": "secret", "paper": True})
        session = TradingSession(symbol="AAPL", broker=broker)
        with pytest.raises(BrokerError, match="live order updates"):
            await session.start()
        await broker.disconnect()
        assert not broker.is_connected()
        assert not client.is_closed
