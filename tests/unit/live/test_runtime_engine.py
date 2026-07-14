"""Bar-driven live engine contracts."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import pytest

from tradelab.errors import LiveTradingDisabledError, ValidationError
from tradelab.live import JsonFileStorage, PaperEngine
from tradelab.live.engine import LiveEngine
from tradelab.live.feed import BrokerFeed


def _bar(
    time: int, close: float, *, high: float | None = None, low: float | None = None
) -> dict[str, float | int]:
    return {
        "time": time,
        "open": close,
        "high": close if high is None else high,
        "low": close if low is None else low,
        "close": close,
        "volume": 100,
    }


def _bars() -> list[dict[str, float | int]]:
    return [
        _bar(1_735_828_200_000, 100, high=100.5, low=99.8),
        _bar(1_735_828_260_000, 101, high=101.2, low=100.1),
        _bar(1_735_828_320_000, 102.5, high=103, low=101.2),
        _bar(1_735_828_380_000, 102.3, high=103, low=102),
    ]


@pytest.mark.asyncio
async def test_live_engine_awaits_signal_trades_persists_and_stops_cleanly(tmp_path: Path) -> None:
    bars = _bars()
    broker = PaperEngine(equity=10_000)
    broker.set_historical_bars("AAPL", "1m", bars[:1])

    async def signal(context: dict[str, object]) -> dict[str, object] | None:
        await asyncio.sleep(0)
        if context["index"] != 1:
            return None
        bar = context["bar"]
        assert isinstance(bar, Mapping)
        return {"side": "buy", "stop": float(bar["close"]) - 1, "rr": 1, "qty": 1}

    engine = LiveEngine(
        id="runtime-e2e",
        symbol="AAPL",
        signal=signal,
        broker=broker,
        storage=JsonFileStorage(base_dir=tmp_path),
        warmup_bars=1,
        flatten_at_close=False,
    )
    await engine.start()
    for bar in bars[1:]:
        await broker.simulate_bar("AAPL", "1m", bar)
    status = engine.get_status()
    assert status["trades"] == 1
    assert status["openPosition"] is None
    assert engine.trades[0]["exit"]["reason"] == "TP"
    assert (await engine.state_manager.load_trades("runtime-e2e"))[0]["symbol"] == "AAPL"
    await engine.stop()
    await engine.stop()
    assert broker.is_connected() is False
    assert engine.running is False


@pytest.mark.asyncio
async def test_handle_bar_serializes_slow_signals_and_deduplicates_time(tmp_path: Path) -> None:
    broker = PaperEngine()
    calls: list[int] = []

    async def signal(context: dict[str, object]) -> None:
        calls.append(cast(int, context["index"]))
        await asyncio.sleep(0.01)

    engine = LiveEngine(
        id="serialized",
        symbol="A",
        signal=signal,
        broker=broker,
        storage=JsonFileStorage(base_dir=tmp_path),
        warmup_bars=1,
    )
    await engine.start()
    bar = _bar(1_735_828_200_000, 10)
    await asyncio.gather(engine.handle_bar(bar), engine.handle_bar(bar))
    assert calls == [0]
    assert len(engine.candle_buffer) == 1
    await engine.stop()


@pytest.mark.asyncio
async def test_accepted_exit_is_not_resubmitted_on_later_triggering_bars(tmp_path: Path) -> None:
    broker = PaperEngine()
    engine = LiveEngine(
        id="single-exit",
        symbol="A",
        signal=lambda _context: None,
        broker=broker,
        storage=JsonFileStorage(base_dir=tmp_path),
        warmup_bars=0,
    )
    await engine.start()
    engine.open_position = {
        "id": "trade-1",
        "symbol": "A",
        "side": "long",
        "size": 1,
        "entry": 100,
        "entryFill": 100,
        "stop": 98,
        "takeProfit": 104,
        "openTime": 1,
        "_openedAtIndex": 0,
    }

    await engine.handle_bar(_bar(1_735_828_200_000, 98, high=99, low=97))
    await engine.handle_bar(_bar(1_735_828_260_000, 97, high=98, low=96))

    assert len(await broker.get_open_orders()) == 1
    assert engine.open_position["_pendingExitOrderId"] == "paper-1"
    assert engine.open_position["_pendingExitClientOrderId"].startswith("single-exit-exit-")
    assert engine.get_status()["openPosition"]["pendingExit"] == {
        "orderId": "paper-1",
        "clientOrderId": engine.open_position["_pendingExitClientOrderId"],
        "reason": "SL",
    }
    broker.positions["A"] = {
        "symbol": "A",
        "side": "long",
        "qty": 1,
        "avgEntry": 100,
    }
    broker.last_prices["A"] = 97
    await engine.stop(flatten_on_shutdown=True)
    assert engine.open_position is None
    assert await broker.get_positions() == []
    assert (await broker.get_order_status("paper-1"))["status"] == "canceled"


@pytest.mark.asyncio
async def test_synchronous_exit_rejection_without_event_halts_risk(tmp_path: Path) -> None:
    class SilentRejectPaper(PaperEngine):
        async def submit_order(self, order: Mapping[str, object]) -> dict[str, Any]:
            return {
                "orderId": "rejected-exit",
                "clientOrderId": order["clientOrderId"],
                "status": "rejected",
                "symbol": order["symbol"],
                "side": order["side"],
                "type": order["type"],
                "qty": order["qty"],
                "filledQty": 0,
                "rejectReason": "venue risk gate",
            }

    broker = SilentRejectPaper()
    engine = LiveEngine(
        id="rejected-exit",
        symbol="A",
        signal=lambda _context: None,
        broker=broker,
        storage=JsonFileStorage(base_dir=tmp_path),
        warmup_bars=0,
    )
    await engine.start()
    engine.open_position = {
        "id": "trade-1",
        "symbol": "A",
        "side": "long",
        "size": 1,
        "entry": 100,
        "entryFill": 100,
        "stop": 98,
        "takeProfit": 104,
        "openTime": 1,
        "_openedAtIndex": 0,
    }

    await engine.handle_bar(_bar(1_735_828_200_000, 98, high=99, low=97))

    assert engine.risk_manager.halted is True
    assert engine.risk_manager.halt_reason == "exit order rejected: venue risk gate"
    assert engine.open_position["_pendingExitOrderId"] == "rejected-exit"
    await engine.stop()


@pytest.mark.asyncio
async def test_start_failure_rolls_back_feed_broker_and_listeners(tmp_path: Path) -> None:
    class BrokenFeed(BrokerFeed):
        disconnected = False

        async def get_historical_bars(
            self, symbol: str, interval: str, count: int
        ) -> list[Mapping[str, object]]:
            raise RuntimeError("history unavailable")

        async def disconnect(self) -> None:
            self.disconnected = True

    broker = PaperEngine()
    feed = BrokenFeed(broker=broker)
    engine = LiveEngine(
        id="broken",
        symbol="A",
        signal=lambda _context: None,
        broker=broker,
        feed=feed,
        storage=JsonFileStorage(base_dir=tmp_path),
    )
    with pytest.raises(RuntimeError, match="history unavailable"):
        await engine.start()
    assert broker.is_connected() is False
    assert feed.disconnected is True
    assert engine.running is engine.connected is False
    assert engine._unsubscribers == []


@pytest.mark.asyncio
async def test_orchestrator_owned_start_failure_preserves_shared_transport(tmp_path: Path) -> None:
    class BrokenFeed(BrokerFeed):
        disconnected = False

        async def get_historical_bars(
            self, symbol: str, interval: str, count: int
        ) -> list[Mapping[str, object]]:
            raise RuntimeError("history unavailable")

        async def disconnect(self) -> None:
            self.disconnected = True

    broker = PaperEngine()
    feed = BrokenFeed(broker=broker)
    engine = LiveEngine(
        id="shared-start-failure",
        symbol="A",
        signal=lambda _context: None,
        broker=broker,
        feed=feed,
        storage=JsonFileStorage(base_dir=tmp_path),
    )

    with pytest.raises(RuntimeError, match="history unavailable"):
        await engine.start(
            disconnect_feed_on_failure=False,
            disconnect_broker_on_failure=False,
        )

    assert broker.is_connected() is True
    assert feed.disconnected is False
    assert engine.running is engine.connected is False
    await broker.disconnect()


@pytest.mark.asyncio
async def test_restart_mismatch_halts_risk_and_restores_state(tmp_path: Path) -> None:
    storage = JsonFileStorage(base_dir=tmp_path)
    await storage.save(
        "restart",
        {
            "openPosition": {
                "symbol": "A",
                "side": "long",
                "size": 2,
                "entry": 10,
                "entryFill": 10,
                "stop": 9,
                "takeProfit": 12,
                "openTime": 1,
            },
            "equity": 9_900,
            "candleBuffer": [],
            "lastBarTime": None,
            "dayPnl": -100,
            "dayTrades": 1,
            "tradeIdCounter": 3,
        },
    )
    broker = PaperEngine()
    await broker.simulate_bar("A", "1m", _bar(1, 10))
    await broker.submit_order({"symbol": "A", "side": "sell", "type": "market", "qty": 1})
    engine = LiveEngine(
        id="restart",
        symbol="A",
        signal=lambda _context: None,
        broker=broker,
        storage=storage,
        warmup_bars=1,
        use_broker_account_equity=False,
    )
    await engine.start()
    assert engine.equity == 9_900
    assert engine.trade_id_counter == 3
    assert engine.risk_manager.halted is True
    assert engine.risk_manager.halt_reason == "position mismatch on restart"
    await engine.stop()


@pytest.mark.asyncio
async def test_restart_halts_when_persisted_exit_is_not_open_at_broker(tmp_path: Path) -> None:
    storage = JsonFileStorage(base_dir=tmp_path)
    position = {
        "symbol": "A",
        "side": "long",
        "size": 1,
        "entry": 10,
        "entryFill": 10,
        "stop": 9,
        "takeProfit": 12,
        "openTime": 1,
        "_pendingExitOrderId": "vanished-order",
        "_pendingExitClientOrderId": "restart-exit-1",
    }
    await storage.save(
        "restart-exit",
        {
            "openPosition": position,
            "equity": 10_000,
            "candleBuffer": [],
            "lastBarTime": None,
            "dayPnl": 0,
            "dayTrades": 0,
            "tradeIdCounter": 1,
        },
    )
    broker = PaperEngine()
    broker.positions["A"] = {
        "symbol": "A",
        "side": "long",
        "qty": 1,
        "avgEntry": 10,
    }
    engine = LiveEngine(
        id="restart-exit",
        symbol="A",
        signal=lambda _context: None,
        broker=broker,
        storage=storage,
        warmup_bars=0,
        use_broker_account_equity=False,
    )

    await engine.start()

    assert engine.risk_manager.halted is True
    assert engine.risk_manager.halt_reason == "pending exit order missing on restart"
    await engine.stop()


@pytest.mark.asyncio
async def test_pending_limit_chases_then_expires_and_cancels(tmp_path: Path) -> None:
    broker = PaperEngine()
    broker.set_historical_bars("A", "1m", [_bar(1_735_828_200_000, 100)])

    def signal(context: dict[str, object]) -> dict[str, object] | None:
        if context["index"] != 1:
            return None
        return {
            "side": "buy",
            "entry": 99,
            "stop": 98,
            "rr": 2,
            "qty": 1,
            "_entryExpiryBars": 1,
            "_imb": {"mid": 99.5},
        }

    engine = LiveEngine(
        id="pending",
        symbol="A",
        signal=signal,
        broker=broker,
        storage=JsonFileStorage(base_dir=tmp_path),
        warmup_bars=1,
        entry_chase={"enabled": True, "afterBars": 1},
    )
    await engine.start()
    await broker.simulate_bar("A", "1m", _bar(1_735_828_260_000, 100, low=100))
    assert engine.pending_order is not None
    order_id = str(engine.pending_order["orderId"])
    await broker.simulate_bar("A", "1m", _bar(1_735_828_320_000, 100, low=100))
    assert engine.pending_order["entry"] == 99.5
    await broker.simulate_bar("A", "1m", _bar(1_735_828_380_000, 100, low=100))
    await broker.simulate_bar("A", "1m", _bar(1_735_828_440_000, 100, low=100))
    assert engine.pending_order is None
    assert (await broker.get_order_status(order_id))["status"] == "canceled"
    await engine.stop()


@pytest.mark.asyncio
async def test_engine_risk_gate_warns_before_oversized_order(tmp_path: Path) -> None:
    broker = PaperEngine()
    warnings: list[dict[str, Any]] = []
    engine = LiveEngine(
        id="risk-gate",
        symbol="A",
        signal=lambda context: {
            "side": "buy",
            "stop": float(cast(int | float, cast(Mapping[str, object], context["bar"])["close"]))
            - 1,
            "rr": 1,
            "qty": 1_000,
        },
        broker=broker,
        storage=JsonFileStorage(base_dir=tmp_path),
        risk={"max_position_pct": 10},
        warmup_bars=1,
    )
    engine.event_bus.on("risk:warning", warnings.append)
    await engine.start()
    await broker.simulate_bar("A", "1m", _bar(1_735_828_200_000, 100))
    assert engine.pending_order is None
    assert warnings[-1]["reason"] == "max position size exceeded"
    await engine.stop()


@pytest.mark.asyncio
async def test_async_signal_failure_is_contextual_and_engine_can_stop(tmp_path: Path) -> None:
    broker = PaperEngine()

    async def broken(_context: dict[str, object]) -> None:
        raise RuntimeError("model offline")

    engine = LiveEngine(
        id="signal-error",
        symbol="A",
        signal=broken,
        broker=broker,
        storage=JsonFileStorage(base_dir=tmp_path),
        warmup_bars=1,
    )
    await engine.start()
    with pytest.raises(Exception, match="model offline"):
        await broker.simulate_bar("A", "1m", _bar(1_735_828_200_000, 10))
    await engine.stop()


@pytest.mark.asyncio
async def test_stop_can_flatten_open_position_on_shutdown(tmp_path: Path) -> None:
    broker = PaperEngine()
    engine = LiveEngine(
        id="flatten-stop",
        symbol="A",
        signal=lambda context: {
            "side": "buy",
            "stop": float(cast(int | float, cast(Mapping[str, object], context["bar"])["close"]))
            - 1,
            "rr": 5,
            "qty": 1,
        },
        broker=broker,
        storage=JsonFileStorage(base_dir=tmp_path),
        warmup_bars=1,
    )
    await engine.start()
    await broker.simulate_bar("A", "1m", _bar(1_735_828_200_000, 10))
    assert engine.open_position is not None
    await engine.stop(flatten_on_shutdown=True)
    assert engine.open_position is None
    assert engine.trades[-1]["exit"]["reason"] == "SHUTDOWN"


def test_live_engine_requires_signal_broker_and_symbol() -> None:
    with pytest.raises(ValidationError, match="signal"):
        LiveEngine(symbol="A", signal=None, broker=PaperEngine())
    with pytest.raises(ValidationError, match="broker"):
        LiveEngine(symbol="A", signal=lambda _context: None, broker=None)
    with pytest.raises(ValidationError, match="symbol"):
        LiveEngine(symbol="", signal=lambda _context: None, broker=PaperEngine())

    class RestOnly:
        def supports_order_updates(self) -> bool:
            return False

    with pytest.raises(ValidationError, match="order updates"):
        LiveEngine(symbol="A", signal=lambda _context: None, broker=RestOnly())


def test_nonpaper_live_engine_requires_env_and_explicit_confirmation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class StreamingBroker:
        def supports_order_updates(self) -> bool:
            return True

    broker = StreamingBroker()
    monkeypatch.delenv("TRADELAB_ALLOW_LIVE", raising=False)
    with pytest.raises(LiveTradingDisabledError, match="TRADELAB_ALLOW_LIVE=true"):
        LiveEngine(symbol="A", signal=lambda _context: None, broker=broker, confirm_live=True)
    monkeypatch.setenv("TRADELAB_ALLOW_LIVE", "true")
    with pytest.raises(LiveTradingDisabledError, match="confirm_live=True"):
        LiveEngine(symbol="A", signal=lambda _context: None, broker=broker)
    engine = LiveEngine(symbol="A", signal=lambda _context: None, broker=broker, confirm_live=True)
    assert engine.broker is broker


@pytest.mark.asyncio
async def test_engine_start_unwires_partial_broker_registration(tmp_path: Path) -> None:
    class PartialWirePaper(PaperEngine):
        def on(self, event: str, handler: Any) -> Any:
            if event == "order:filled":
                raise RuntimeError("updates unavailable")
            return super().on(event, handler)

    broker = PartialWirePaper()
    engine = LiveEngine(
        symbol="A",
        signal=lambda _context: None,
        broker=broker,
        storage=JsonFileStorage(base_dir=tmp_path),
    )
    with pytest.raises(RuntimeError, match="updates unavailable"):
        await engine.start()
    assert broker._handlers == {}
