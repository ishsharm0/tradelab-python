"""Interactive Brokers lazy dependency, gateway lifecycle, and fallback contracts."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from tradelab.brokers import InteractiveBrokersBroker
from tradelab.errors import BrokerError, ValidationError


class FakeIB:
    def __init__(self) -> None:
        self.connect_call: tuple[object, ...] | None = None
        self.disconnected = False

    async def connectAsync(
        self,
        host: str,
        port: int,
        clientId: int,
        timeout: float,
        readonly: bool,
    ) -> None:
        self.connect_call = (host, port, clientId, timeout, readonly)

    def disconnect(self) -> None:
        self.disconnected = True

    async def accountSummaryAsync(self) -> list[object]:
        return [
            SimpleNamespace(tag="NetLiquidation", value="10000", currency="USD"),
            SimpleNamespace(tag="BuyingPower", value="8000", currency="USD"),
            SimpleNamespace(tag="TotalCashValue", value="5000", currency="USD"),
            SimpleNamespace(tag="InitMarginReq", value="250", currency="USD"),
        ]

    def positions(self) -> list[object]:
        return [
            SimpleNamespace(
                contract=SimpleNamespace(symbol="AAPL"),
                position=-2,
                avgCost=100,
            )
        ]

    async def reqHistoricalDataAsync(self, _contract: object, **kwargs: object) -> list[object]:
        assert kwargs["barSizeSetting"] == "1 min"
        return [
            SimpleNamespace(
                date="2025-01-02T14:30:00Z",
                open=100,
                high=101,
                low=99,
                close=100.5,
                volume=10,
            )
        ]


def _missing(_name: str) -> object:
    raise ModuleNotFoundError("ib_insync")


@pytest.mark.asyncio
async def test_ib_dependency_is_lazy_and_missing_extra_error_is_actionable() -> None:
    broker = InteractiveBrokersBroker(module_loader=_missing)
    assert broker.supports_paper_native()
    with pytest.raises(BrokerError, match=r"tradelab\[ib\].*ib-insync"):
        await broker.connect({"paper": True})


@pytest.mark.asyncio
async def test_ib_connects_to_paper_gateway_maps_account_positions_and_disconnects() -> None:
    fake = FakeIB()
    broker = InteractiveBrokersBroker(ib_factory=lambda: fake)
    with pytest.raises(ValidationError, match="port"):
        await broker.connect({"port": 0})
    await broker.connect(
        {"paper": True, "host": "gateway", "clientId": 9, "timeout": 3, "readonly": True}
    )
    assert fake.connect_call == ("gateway", 7497, 9, 3.0, True)
    assert broker.is_connected()
    assert await broker.get_account() == {
        "equity": 10000.0,
        "buyingPower": 8000.0,
        "cash": 5000.0,
        "currency": "USD",
        "marginUsed": 250.0,
    }
    assert await broker.get_positions() == [
        {
            "symbol": "AAPL",
            "side": "short",
            "qty": 2.0,
            "avgEntry": 100.0,
            "marketValue": 200.0,
            "unrealizedPnl": 0.0,
        }
    ]
    await broker.disconnect()
    assert fake.disconnected and not broker.is_connected()


@pytest.mark.asyncio
async def test_ib_local_order_lifecycle_is_session_compatible() -> None:
    broker = InteractiveBrokersBroker(ib_factory=FakeIB, clock=lambda: 123)
    await broker.connect({"paper": False})
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
    assert receipt == {
        "orderId": "1",
        "clientOrderId": "session-1",
        "status": "new",
        "filledQty": 0.0,
        "avgFillPrice": None,
        "filledAt": None,
        "symbol": "AAPL",
        "side": "buy",
        "type": "limit",
        "qty": 1.0,
        "limitPrice": 100.0,
    }
    assert await broker.get_open_orders() == [receipt]
    modified = await broker.modify_order("1", {"qty": 2, "stop_price": 99})
    assert modified["qty"] == 2 and modified["stopPrice"] == 99
    await broker.cancel_order("1")
    assert (await broker.get_order_status("1"))["status"] == "canceled"
    assert await broker.get_open_orders() == []
    await broker.cancel_order("missing")
    with pytest.raises(BrokerError, match="not found"):
        await broker.modify_order("missing", {})


@pytest.mark.asyncio
async def test_ib_historical_bars_and_noop_subscriptions() -> None:
    created_contracts: list[tuple[str, str, str]] = []

    def stock(symbol: str, exchange: str, currency: str) -> object:
        created_contracts.append((symbol, exchange, currency))
        return object()

    fake = FakeIB()
    broker = InteractiveBrokersBroker(
        ib_factory=lambda: fake,
        contract_factory=stock,
    )
    await broker.connect({})
    bars = await broker.get_historical_bars("AAPL", "1m", 5)
    seen: list[dict[str, Any]] = []
    unsubscribe = await broker.subscribe_bars("AAPL", "1m", seen.append)
    await broker.publish_bar("AAPL", "1m", {"close": 1})
    unsubscribe()
    await broker.publish_bar("AAPL", "1m", {"close": 2})
    assert created_contracts == [("AAPL", "SMART", "USD")]
    assert bars == [
        {
            "time": 1_735_828_200_000,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 10.0,
        }
    ]
    assert seen == [{"close": 1}]
