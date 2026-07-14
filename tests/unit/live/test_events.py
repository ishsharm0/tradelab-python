"""Live event bus and adapter contracts."""

from __future__ import annotations

from typing import Any, cast

import pytest

from tradelab.live import (
    LIVE_EVENTS,
    BrokerAdapter,
    EventBus,
    StorageProvider,
    create_event_bus,
)


def test_event_bus_preserves_native_then_wildcard_order_and_unsubscribes() -> None:
    bus = EventBus()
    seen: list[object] = []
    stop_native = bus.on("signal", lambda payload: seen.append(("signal", payload)))
    stop_any = bus.on_any(lambda envelope: seen.append(envelope))

    assert bus.emit_event("signal", {"side": "long"}) is True
    assert seen == [
        ("signal", {"side": "long"}),
        {"event": "signal", "payload": {"side": "long"}},
    ]
    stop_native()
    stop_native()
    stop_any()
    assert bus.emit_event("signal", {}) is True
    assert len(seen) == 2


def test_live_event_names_match_javascript_contract() -> None:
    assert LIVE_EVENTS[0] == "signal"
    assert LIVE_EVENTS[-1] == "reconciled"
    assert len(LIVE_EVENTS) == len(set(LIVE_EVENTS)) == 21


@pytest.mark.asyncio
async def test_base_provider_and_broker_methods_are_explicitly_abstract() -> None:
    storage = StorageProvider()
    with pytest.raises(NotImplementedError, match=r"StorageProvider.load\(\)"):
        await storage.load("x")
    broker = BrokerAdapter()
    with pytest.raises(NotImplementedError, match=r"BrokerAdapter.connect\(\)"):
        await broker.connect()
    with pytest.raises(NotImplementedError, match="disconnect"):
        await broker.disconnect()
    with pytest.raises(NotImplementedError, match="isConnected"):
        broker.is_connected()
    with pytest.raises(NotImplementedError, match="getAccount"):
        await broker.get_account()
    with pytest.raises(NotImplementedError, match="getPositions"):
        await broker.get_positions()
    with pytest.raises(NotImplementedError, match="submitOrder"):
        await broker.submit_order({})
    with pytest.raises(NotImplementedError, match="cancelOrder"):
        await broker.cancel_order("x")
    with pytest.raises(NotImplementedError, match="modifyOrder"):
        await broker.modify_order("x", {})
    with pytest.raises(NotImplementedError, match="getOpenOrders"):
        await broker.get_open_orders()
    with pytest.raises(NotImplementedError, match="getOrderStatus"):
        await broker.get_order_status("x")

    def handler(_payload: dict[str, Any]) -> None:
        return None

    with pytest.raises(NotImplementedError, match="subscribeQuotes"):
        await broker.subscribe_quotes("A", handler)
    with pytest.raises(NotImplementedError, match="subscribeTrades"):
        await broker.subscribe_trades("A", handler)
    with pytest.raises(NotImplementedError, match="subscribeBars"):
        await broker.subscribe_bars("A", "1m", handler)
    with pytest.raises(NotImplementedError, match="getHistoricalBars"):
        await broker.get_historical_bars("A", "1m")

    with pytest.raises(NotImplementedError, match="save"):
        await storage.save("x", {})
    with pytest.raises(NotImplementedError, match="appendTrade"):
        await storage.append_trade("x", {})
    with pytest.raises(NotImplementedError, match="appendEquityPoint"):
        await storage.append_equity_point("x", {})
    with pytest.raises(NotImplementedError, match="loadTrades"):
        await storage.load_trades("x")
    with pytest.raises(NotImplementedError, match="loadEquityCurve"):
        await storage.load_equity_curve("x")
    with pytest.raises(NotImplementedError, match="clear"):
        await storage.clear("x")
    assert broker.supports_paper_native() is False
    assert isinstance(await broker.get_server_time(), int)


def test_listener_exceptions_propagate_without_emitting_wildcard() -> None:
    bus = EventBus()
    seen: list[dict[str, Any]] = []
    bus.on("error", lambda _payload: (_ for _ in ()).throw(RuntimeError("boom")))
    bus.on_any(seen.append)
    with pytest.raises(RuntimeError, match="boom"):
        bus.emit_event("error", {"message": "x"})
    assert seen == []


def test_event_bus_validates_handlers_and_can_remove_all_listeners() -> None:
    bus = create_event_bus()
    with pytest.raises(TypeError, match="event"):
        bus.on("", lambda _payload: None)
    with pytest.raises(TypeError, match="handler"):
        bus.on("x", cast(Any, None))
    bus.off("missing", lambda _payload: None)
    seen: list[dict[str, Any]] = []
    bus.on("x", seen.append)
    bus.remove_all_listeners()
    bus.emit("x", {"value": 1})
    assert seen == []
