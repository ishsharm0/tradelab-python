"""Trading-session permission, lifecycle, sizing, and OCO contracts."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from typing import Any, cast

import pytest

from tradelab.errors import LiveTradingDisabledError, RiskRejectedError, ValidationError
from tradelab.live import BrokerAdapter, EventBus, PaperEngine, SessionManager, TradingSession


class SnakeCaseBroker:
    """Credentialed-adapter shape used by the package's REST brokers."""

    def __init__(self) -> None:
        self.connected = False
        self.events = EventBus()
        self.orders: list[dict[str, object]] = []

    async def connect(self, config: Mapping[str, object] | None = None) -> None:
        self.connected = True

    async def disconnect(self) -> None:
        self.connected = False

    def is_connected(self) -> bool:
        return self.connected

    def supports_paper_native(self) -> bool:
        return True

    def on(self, event: str, handler: Callable[[dict[str, Any]], object]) -> Callable[[], None]:
        return self.events.on(event, handler)

    async def get_account(self) -> dict[str, object]:
        return {
            "equity": 12_000,
            "buying_power": 8_000,
            "cash": 6_000,
            "currency": "USD",
            "margin_used": 4_000,
        }

    async def get_positions(self) -> list[dict[str, object]]:
        return [
            {
                "symbol": "AAPL",
                "side": "long",
                "qty": 2,
                "avg_entry": 100,
                "market_value": 202,
                "unrealized_pnl": 2,
            }
        ]

    async def get_open_orders(self) -> list[dict[str, object]]:
        return []

    async def submit_order(self, order: Mapping[str, object]) -> dict[str, object]:
        self.orders.append(dict(order))
        receipt: dict[str, object] = {
            "order_id": "external-1",
            "client_order_id": order.get("clientOrderId"),
            "status": "new",
            "filled_qty": 0,
            "avg_fill_price": None,
            "filled_at": None,
            "symbol": order["symbol"],
            "side": order["side"],
            "type": order["type"],
            "qty": order["qty"],
            "reject_reason": None,
        }
        self.events.emit("order:submitted", receipt)
        return receipt

    async def cancel_order(self, order_id: str) -> None:
        self.events.emit("order:canceled", {"order_id": order_id})


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


async def _session(**kwargs: Any) -> TradingSession:
    broker = cast(BrokerAdapter, kwargs.pop("broker", PaperEngine(equity=10_000)))
    session = TradingSession(id="t1", symbol="AAPL", broker=broker, equity=10_000, **kwargs)
    await session.start()
    return session


@pytest.mark.asyncio
async def test_market_sizing_events_and_status_match_live_payloads() -> None:
    bus = EventBus()
    forwarded: list[dict[str, Any]] = []
    bus.on_any(forwarded.append)
    session = await _session(event_bus=bus, risk_pct=1)
    await session.push_bar(_bar(1, 100))
    receipt = await session.place_order(
        side="long",
        risk_pct=1,
        stop=98,
        target=104,
        rationale={"signal": "ema-cross"},
    )
    assert receipt["status"] == "filled"
    status = session.get_status()
    assert status["positions"][0]["qty"] == 50
    filled = next(event for event in session.recent_events() if event["event"] == "order:filled")
    assert filled["payload"]["sizing"] == {
        "entry": 100,
        "stop": 98,
        "target": 104,
        "rr": None,
        "riskFraction": 0.01,
        "riskAmount": 100,
        "qty": 50,
        "notional": 5_000,
    }
    assert filled["payload"]["rationale"] == {"signal": "ema-cross"}
    assert any(item["payload"].get("sessionId") == "t1" for item in forwarded)


@pytest.mark.asyncio
async def test_bracket_oco_and_wide_bar_fill_exactly_one_exit() -> None:
    session = await _session()
    await session.push_bar(_bar(1, 100))
    await session.place_order(side="long", qty=10, stop=98, target=104)
    assert len(session.get_status()["openOrders"]) == 2
    await session.push_bar(_bar(2, 100, high=104, low=98))
    assert session.get_status()["positions"] == []
    assert session.get_status()["openOrders"] == []
    closed = [event for event in session.recent_events() if event["event"] == "position:closed"]
    assert len(closed) == 1


@pytest.mark.asyncio
async def test_resting_limit_attaches_bracket_after_fill() -> None:
    session = await _session()
    await session.push_bar(_bar(1, 100))
    await session.place_order(
        side="long", type="limit", qty=10, limit_price=99, stop=97, target=105
    )
    assert session.brackets == {}
    await session.push_bar(_bar(2, 99, high=100, low=98))
    assert len(session.get_status()["positions"]) == 1
    assert len(session.get_status()["openOrders"]) == 2
    assert "AAPL" in session.brackets


@pytest.mark.asyncio
async def test_multisymbol_prices_oco_and_exposure_use_each_position_value() -> None:
    broker = PaperEngine(equity=10_000)
    session = TradingSession(
        id="multi",
        symbols=["AAPL", "MSFT"],
        broker=broker,
        equity=10_000,
        max_gross_exposure_pct=100,
        qty_step=1,
        min_qty=1,
    )
    await session.start()
    await session.push_bar(_bar(1, 200), symbol="AAPL")
    await session.place_order(side="long", qty=50, symbol="AAPL", stop=190, target=210)
    await session.push_bar(_bar(1, 50), symbol="MSFT")
    with pytest.raises(RiskRejectedError, match="gross exposure"):
        await session.place_order(side="long", qty=1, symbol="MSFT")
    assert session.last_price_for("AAPL") == 200
    with pytest.raises(ValidationError, match="symbol is required"):
        await session.place_order(side="long", qty=1)
    assert "AAPL" in session.brackets


@pytest.mark.asyncio
async def test_close_flatten_and_risk_halt_block_new_orders() -> None:
    session = await _session(max_daily_loss_pct=1)
    await session.push_bar(_bar(1, 100))
    await session.place_order(side="long", qty=100)
    await session.push_bar(_bar(2, 98))
    await session.close_position()
    assert session.get_status()["risk"]["halted"] is True
    with pytest.raises(RiskRejectedError, match="risk-halted"):
        await session.place_order(side="long", qty=1)
    await session.flatten()
    assert session.get_status()["positions"] == []


def test_live_permission_requires_env_confirmation_and_nonpaper_broker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TRADELAB_ALLOW_LIVE", raising=False)
    with pytest.raises(LiveTradingDisabledError, match="gated"):
        TradingSession(symbol="A", broker=PaperEngine(), mode="live", confirm_live=True)
    monkeypatch.setenv("TRADELAB_ALLOW_LIVE", "true")
    with pytest.raises(LiveTradingDisabledError, match="confirm_live"):
        TradingSession(symbol="A", broker=PaperEngine(), mode="live")
    with pytest.raises(LiveTradingDisabledError, match="credentialed"):
        TradingSession(symbol="A", broker=PaperEngine(), mode="live", confirm_live=True)


@pytest.mark.asyncio
async def test_live_session_normalizes_structural_external_broker_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRADELAB_ALLOW_LIVE", "true")
    broker = SnakeCaseBroker()
    session = TradingSession(
        id="external",
        symbol="AAPL",
        broker=broker,
        mode="live",
        confirm_live=True,
    )
    await session.start()

    position = session.get_status()["positions"][0]
    assert position == {
        "symbol": "AAPL",
        "side": "long",
        "qty": 2,
        "avgEntry": 100,
        "marketValue": 202,
        "unrealizedPnl": 2,
    }
    await session.push_bar(_bar(1, 101))
    receipt = await session.place_order(side="buy", qty=1)
    assert receipt["orderId"] == "external-1"
    submitted = [event for event in session.recent_events() if event["event"] == "order:submitted"][
        -1
    ]
    assert submitted["payload"]["clientOrderId"].startswith("external-entry-")
    assert "client_order_id" not in submitted["payload"]
    await session.stop()


@pytest.mark.asyncio
async def test_start_stop_are_idempotent_and_cleanup_broker_listeners() -> None:
    broker = PaperEngine()
    session = TradingSession(id="life", symbol="A", broker=broker)
    await session.start()
    await session.start()
    await session.stop()
    await session.stop()
    assert broker.is_connected() is False
    assert [event["event"] for event in session.events].count("connected") == 1
    assert [event["event"] for event in session.events].count("shutdown") == 1


@pytest.mark.asyncio
async def test_cancelled_start_disconnects_partial_broker() -> None:
    entered = asyncio.Event()

    class SlowPaper(PaperEngine):
        async def connect(self, config: Mapping[str, object] | None = None) -> None:
            self.connected = True
            entered.set()
            await asyncio.sleep(30)

    broker = SlowPaper()
    session = TradingSession(symbol="A", broker=broker)
    task = asyncio.create_task(session.start())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert broker.is_connected() is False
    assert session.running is False


@pytest.mark.asyncio
async def test_background_bracket_failures_are_drained_and_reported() -> None:
    session = await _session()

    async def fail_bracket() -> None:
        raise RuntimeError("bracket rejected")

    session._spawn(fail_bracket())
    await asyncio.sleep(0)
    with pytest.raises(RuntimeError, match="bracket rejected"):
        await session.refresh()
    assert session._tasks == set()
    await session.stop()


@pytest.mark.asyncio
async def test_session_manager_create_remove_halt_all_and_duplicate_safety() -> None:
    manager = SessionManager()
    first = await manager.create(id="one", symbol="A", equity=5_000)
    with pytest.raises(ValidationError, match="already exists"):
        await manager.create(id="one", symbol="A")
    second = await manager.create(id="two", symbol="B", equity=5_000)
    await first.push_bar(_bar(1, 10))
    await first.place_order(side="long", qty=2)
    await manager.remove("missing")
    await manager.remove("one", flatten=True)
    assert manager.get("one") is None
    assert (await first.get_positions()) == []
    await manager.halt_all()
    assert manager.list() == []
    assert second.running is False
