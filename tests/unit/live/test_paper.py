"""Paper broker execution and event contracts."""

from __future__ import annotations

import asyncio
import json
import math
from typing import Any

import pytest

from tradelab.errors import ValidationError
from tradelab.live import PaperEngine


def _bar(
    time: int, price: float, *, high: float | None = None, low: float | None = None
) -> dict[str, float | int]:
    return {
        "time": time,
        "open": price,
        "high": price if high is None else high,
        "low": price if low is None else low,
        "close": price,
        "volume": 100,
    }


@pytest.mark.asyncio
async def test_market_round_trip_tracks_positions_cash_and_events() -> None:
    broker = PaperEngine(equity=1_000, slippage_bps=0, fee_bps=0)
    events: list[dict[str, Any]] = []
    broker.on("order:filled", events.append)
    await broker.connect({"account": "paper"})
    await broker.simulate_bar("AAPL", "1m", _bar(1, 100))
    opened = await broker.submit_order(
        {"symbol": "AAPL", "side": "buy", "type": "market", "qty": 2}
    )
    assert opened["status"] == "filled"
    assert (await broker.get_positions())[0] == {
        "symbol": "AAPL",
        "side": "long",
        "qty": 2,
        "avgEntry": 100,
        "marketValue": 200,
        "unrealizedPnl": 0,
    }
    await broker.simulate_bar("AAPL", "1m", _bar(2, 110))
    await broker.submit_order({"symbol": "AAPL", "side": "sell", "type": "market", "qty": 2})
    assert await broker.get_positions() == []
    assert (await broker.get_account())["equity"] == 1_020
    assert [event["orderId"] for event in events] == ["paper-1", "paper-2"]


@pytest.mark.asyncio
async def test_market_without_price_rejects_and_limit_stop_types_fill_on_bars() -> None:
    broker = PaperEngine(equity=10_000)
    await broker.connect()
    rejected = await broker.submit_order(
        {"symbol": "AAPL", "side": "buy", "type": "market", "qty": 1}
    )
    assert rejected["status"] == "rejected"
    assert rejected["rejectReason"] == "no price available for market order"

    limit = await broker.submit_order(
        {"symbol": "AAPL", "side": "buy", "type": "limit", "qty": 1, "limitPrice": 99}
    )
    stop = await broker.submit_order(
        {"symbol": "MSFT", "side": "buy", "type": "stop", "qty": 1, "stopPrice": 105}
    )
    stop_limit = await broker.submit_order(
        {
            "symbol": "TSLA",
            "side": "buy",
            "type": "stop_limit",
            "qty": 1,
            "stopPrice": 102,
            "limitPrice": 101,
        }
    )
    await broker.simulate_bar("AAPL", "1m", _bar(2, 100, low=98))
    await broker.simulate_bar("MSFT", "1m", _bar(2, 104, high=106))
    await broker.simulate_bar("TSLA", "1m", _bar(2, 101, high=103, low=100))
    assert (await broker.get_order_status(limit["orderId"]))["status"] == "filled"
    assert (await broker.get_order_status(stop["orderId"]))["status"] == "filled"
    assert (await broker.get_order_status(stop_limit["orderId"]))["status"] == "filled"


@pytest.mark.asyncio
async def test_wide_bar_does_not_fill_an_order_removed_by_prior_handler() -> None:
    broker = PaperEngine(equity=10_000)
    await broker.connect()
    first = await broker.submit_order(
        {"symbol": "A", "side": "sell", "type": "stop", "qty": 1, "stopPrice": 98}
    )
    sibling = await broker.submit_order(
        {"symbol": "A", "side": "sell", "type": "limit", "qty": 1, "limitPrice": 104}
    )

    def cancel_sibling(order: dict[str, Any]) -> None:
        if order["orderId"] == first["orderId"]:
            broker.cancel_order_nowait(sibling["orderId"])

    broker.on("order:filled", cancel_sibling)
    await broker.simulate_bar("A", "1m", _bar(2, 100, high=104, low=98))
    assert (await broker.get_order_status(first["orderId"]))["status"] == "filled"
    assert (await broker.get_order_status(sibling["orderId"]))["status"] == "canceled"
    assert (await broker.get_positions())[0]["side"] == "short"
    assert (await broker.get_positions())[0]["qty"] == 1


@pytest.mark.asyncio
async def test_subscriptions_history_modify_cancel_and_disconnect_are_idempotent() -> None:
    broker = PaperEngine()
    await broker.connect()
    seen: list[tuple[str, object]] = []

    async def slow_bar(bar: dict[str, Any]) -> None:
        await asyncio.sleep(0)
        seen.append(("bar", bar["close"]))

    stop_bar = await broker.subscribe_bars("A", "1m", slow_bar)
    stop_trade = await broker.subscribe_trades(
        "A", lambda trade: seen.append(("trade", trade["price"]))
    )
    broker.set_historical_bars("A", "1m", [_bar(1, 1), _bar(2, 2), _bar(3, 3)])
    assert len(await broker.get_historical_bars("A", "1m", 2)) == 2
    await broker.simulate_bar("A", "1m", _bar(4, 4))
    assert seen == [("bar", 4.0), ("trade", 4.0)]
    stop_bar()
    stop_bar()
    stop_trade()
    order = await broker.submit_order(
        {"symbol": "A", "side": "buy", "type": "limit", "qty": 2, "limitPrice": 1}
    )
    modified = await broker.modify_order(order["orderId"], {"qty": 3, "limitPrice": 2})
    assert (modified["qty"], modified["limitPrice"]) == (3, 2)
    await broker.cancel_order(order["orderId"])
    await broker.cancel_order(order["orderId"])
    await broker.disconnect()
    await broker.disconnect()
    assert broker.is_connected() is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("order", "message"),
    [
        ({"symbol": "A", "side": "hold", "qty": 1}, "side"),
        ({"symbol": "A", "side": "buy", "type": "iceberg", "qty": 1}, "type"),
        ({"symbol": "A", "side": "buy", "qty": math.inf}, "qty"),
        ({"symbol": "", "side": "buy", "qty": 1}, "symbol"),
    ],
)
async def test_order_validation_rejects_unsafe_values(
    order: dict[str, object], message: str
) -> None:
    broker = PaperEngine()
    with pytest.raises(ValidationError, match=message):
        await broker.submit_order(order)


@pytest.mark.asyncio
async def test_receipts_are_json_safe_snapshots() -> None:
    broker = PaperEngine()
    await broker.simulate_bar("A", "1m", _bar(1, 10))
    receipt = await broker.submit_order({"symbol": "A", "side": "buy", "type": "market", "qty": 1})
    assert json.loads(json.dumps(receipt, allow_nan=False)) == receipt
    receipt["status"] = "tampered"
    assert (await broker.get_order_status("paper-1"))["status"] == "filled"


@pytest.mark.asyncio
async def test_overflowing_fill_is_rejected_before_mutating_position() -> None:
    broker = PaperEngine()
    await broker.simulate_bar("A", "1m", _bar(1, 1e308))
    first = await broker.submit_order({"symbol": "A", "side": "buy", "type": "market", "qty": 1})
    assert first["status"] == "filled"
    before = broker.positions["A"].copy()
    with pytest.raises(ValidationError, match="finite"):
        await broker.submit_order({"symbol": "A", "side": "buy", "type": "market", "qty": 1})
    assert broker.positions["A"] == before
