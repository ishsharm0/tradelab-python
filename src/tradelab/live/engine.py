"""Serialized bar-driven live engine reusing backtest signal and risk contracts."""

from __future__ import annotations

import asyncio
import inspect
import math
import os
import re
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import suppress
from datetime import UTC, datetime
from typing import Any, cast

from tradelab.data.csv import normalize_candles
from tradelab.engine.execution import is_eod_bar, oco_exit_check
from tradelab.engine.signal import call_signal_with_context, normalize_signal
from tradelab.errors import LiveTradingDisabledError, StrategyError, ValidationError
from tradelab.utils.position_sizing import calculate_position_size

from .clock import BrokerClock
from .events import EventBus
from .feed import BrokerFeed, FeedProvider, PollingFeed, Subscription
from .paper import PaperEngine
from .risk import RiskManager
from .state import StateManager
from .storage import JsonFileStorage, StorageProvider

Signal = Callable[[dict[str, object]], object]


def _system_now_ms() -> int:
    return time.time_ns() // 1_000_000


def _number(value: object, fallback: float | None = None) -> float | None:
    if isinstance(value, bool):
        return fallback
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return fallback
    return number if math.isfinite(number) else fallback


_ORDER_ALIASES = {
    "order_id": "orderId",
    "client_order_id": "clientOrderId",
    "filled_qty": "filledQty",
    "avg_fill_price": "avgFillPrice",
    "filled_at": "filledAt",
    "limit_price": "limitPrice",
    "stop_price": "stopPrice",
    "reject_reason": "rejectReason",
}


def _order(value: Mapping[str, Any]) -> dict[str, Any]:
    output = dict(value)
    for snake, camel in _ORDER_ALIASES.items():
        if camel not in output and snake in output:
            output[camel] = output[snake]
        output.pop(snake, None)
    return output


def _position(value: Mapping[str, Any]) -> dict[str, Any]:
    output = dict(value)
    for snake, camel in {
        "avg_entry": "avgEntry",
        "market_value": "marketValue",
        "unrealized_pnl": "unrealizedPnl",
    }.items():
        if camel not in output and snake in output:
            output[camel] = output[snake]
        output.pop(snake, None)
    return output


def _matches(reference: Mapping[str, Any] | None, order: Mapping[str, Any]) -> bool:
    if reference is None:
        return False
    return bool(
        (reference.get("orderId") and reference.get("orderId") == order.get("orderId"))
        or (
            reference.get("clientOrderId")
            and reference.get("clientOrderId") == order.get("clientOrderId")
        )
    )


def _snapshot(position: Mapping[str, Any], mark_price: float) -> dict[str, Any]:
    entry = float(position.get("entryFill", position.get("entry", 0)))
    size = float(position.get("size", 0))
    direction = 1 if position.get("side") == "long" else -1
    pending_exit = None
    if position.get("_pendingExitClientOrderId"):
        pending_exit = {
            "orderId": position.get("_pendingExitOrderId"),
            "clientOrderId": position.get("_pendingExitClientOrderId"),
            "reason": position.get("_pendingExitReason"),
        }
    return {
        "id": position.get("id"),
        "symbol": position.get("symbol"),
        "side": position.get("side"),
        "size": size,
        "entry": position.get("entry"),
        "entryFill": entry,
        "stop": position.get("stop"),
        "takeProfit": position.get("takeProfit"),
        "openTime": position.get("openTime"),
        "markPrice": mark_price,
        "unrealizedPnl": (mark_price - entry) * direction * size,
        "_initRisk": position.get("_initRisk"),
        "pendingExit": pending_exit,
    }


async def _maybe_await(value: object) -> object:
    return await value if inspect.isawaitable(value) else value


class LiveEngine:
    """Run one signal system against a broker/feed with restart-safe state."""

    def __init__(
        self,
        *,
        signal: Signal | None,
        broker: Any,
        symbol: str,
        id: str | None = None,
        interval: str = "1m",
        mode: str = "streaming",
        poll_interval_ms: object = 60_000,
        warmup_bars: int = 200,
        equity: object = 10_000,
        risk_pct: object = 1,
        final_tp_r: object = 3,
        flatten_at_close: bool = False,
        qty_step: object = 0.001,
        min_qty: object = 0.001,
        max_leverage: object = 2,
        daily_max_trades: object = 0,
        max_daily_loss_pct: object = 0,
        entry_chase: Mapping[str, object] | None = None,
        oco: Mapping[str, object] | None = None,
        risk: Mapping[str, object] | None = None,
        broker_config: Mapping[str, object] | None = None,
        confirm_live: bool = False,
        use_broker_account_equity: bool = True,
        feed: FeedProvider | None = None,
        event_bus: EventBus | None = None,
        storage: StorageProvider | None = None,
        clock: BrokerClock | None = None,
        now_ms: Callable[[], int] = _system_now_ms,
    ) -> None:
        if not callable(signal):
            raise ValidationError("live engine requires a signal function")
        if broker is None:
            raise ValidationError("live engine requires a broker adapter")
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValidationError("live engine requires symbol")
        if not isinstance(broker, PaperEngine):
            supports_updates = getattr(broker, "supports_order_updates", None)
            if not callable(supports_updates) or supports_updates() is not True:
                raise ValidationError("live engine requires genuine streaming order updates")
            if os.environ.get("TRADELAB_ALLOW_LIVE") != "true":
                raise LiveTradingDisabledError(
                    "live trading is gated: set TRADELAB_ALLOW_LIVE=true"
                )
            if confirm_live is not True:
                raise LiveTradingDisabledError("live engine requires confirm_live=True")
        if isinstance(warmup_bars, bool) or not isinstance(warmup_bars, int) or warmup_bars < 0:
            raise ValidationError("warmup_bars must be a non-negative integer")
        numeric = {
            "equity": equity,
            "risk_pct": risk_pct,
            "final_tp_r": final_tp_r,
            "qty_step": qty_step,
            "min_qty": min_qty,
            "max_leverage": max_leverage,
        }
        converted: dict[str, float] = {}
        for name, value in numeric.items():
            number = _number(value)
            if number is None or number < 0 or (name == "qty_step" and number == 0):
                raise ValidationError(f"{name} must be finite and non-negative")
            converted[name] = number
        self.signal = signal
        self.broker = broker
        self.symbol = symbol.strip()
        self.interval = interval
        self.mode = mode
        self.namespace = re.sub(r"[^a-zA-Z0-9._-]", "_", id or f"{self.symbol}-{interval}")
        self.poll_interval_ms = poll_interval_ms
        self.warmup_bars = warmup_bars
        self.risk_pct = converted["risk_pct"]
        self.final_tp_r = converted["final_tp_r"]
        self.flatten_at_close = flatten_at_close
        self.qty_step = converted["qty_step"]
        self.min_qty = converted["min_qty"]
        self.max_leverage = converted["max_leverage"]
        self.entry_chase = {
            "enabled": True,
            "afterBars": 2,
            "maxSlipR": 0.2,
            "convertOnExpiry": False,
            **dict(entry_chase or {}),
        }
        self.oco = dict(oco or {})
        self.broker_config = dict(broker_config or {})
        self.use_broker_account_equity = use_broker_account_equity
        if feed is not None:
            self.feed = feed
        elif mode == "polling":
            self.feed = PollingFeed(broker=broker, poll_interval_ms=poll_interval_ms)
        else:
            self.feed = BrokerFeed(broker=cast(Any, broker))
        self.event_bus = event_bus or EventBus()
        self.storage = storage or JsonFileStorage()
        self.state_manager = StateManager(storage=self.storage, now_ms=now_ms)
        risk_options = dict(risk or {})
        risk_options.setdefault("max_daily_loss_pct", max_daily_loss_pct)
        risk_options.setdefault("max_daily_trades", daily_max_trades)
        self.risk_manager = RiskManager(risk_options)
        self.clock = clock or BrokerClock(now_ms=now_ms)
        self._now_ms = now_ms

        self.running = False
        self.connected = False
        self.subscriptions: list[Subscription] = []
        self.candle_buffer: list[dict[str, int | float]] = []
        self.last_bar_time: int | float | None = None
        self.open_position: dict[str, Any] | None = None
        self.pending_order: dict[str, Any] | None = None
        self.trade_id_counter = 0
        self.trades: list[dict[str, Any]] = []
        self.eq_series: list[dict[str, float]] = []
        self.equity = converted["equity"]
        self.day_pnl = 0.0
        self.day_trades = 0
        self.started_at: int | None = None
        self._event_tasks: set[asyncio.Task[None]] = set()
        self._unsubscribers: list[Callable[[], None]] = []
        self._bar_lock = asyncio.Lock()
        self._lifecycle_lock = asyncio.Lock()

    def _emit(self, event: str, payload: Mapping[str, Any] | None = None) -> None:
        self.event_bus.emit_event(event, dict(payload or {}))

    def _spawn(self, awaitable: Awaitable[None]) -> None:
        self._event_tasks.add(asyncio.ensure_future(awaitable))

    async def _drain_event_tasks(self) -> None:
        while self._event_tasks:
            tasks = tuple(self._event_tasks)
            try:
                await asyncio.gather(*tasks)
            except BaseException:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise
            finally:
                self._event_tasks.difference_update(tasks)

    def _is_order_for_symbol(self, order: Mapping[str, Any]) -> bool:
        return not order.get("symbol") or order.get("symbol") == self.symbol

    def _forward_order(self, event: str, raw: Mapping[str, Any]) -> None:
        order = _order(raw)
        if self._is_order_for_symbol(order):
            self._emit(event, {**order, "symbol": order.get("symbol") or self.symbol})

    def _wire_broker(self) -> None:
        if self._unsubscribers:
            return
        try:
            self._unsubscribers.append(
                self.broker.on(
                    "order:submitted", lambda row: self._forward_order("order:submitted", row)
                )
            )
            self._unsubscribers.append(self.broker.on("order:filled", self._on_order_filled))
            self._unsubscribers.append(
                self.broker.on(
                    "order:canceled",
                    lambda row: self._on_order_terminal("order:canceled", row),
                )
            )
            self._unsubscribers.append(
                self.broker.on(
                    "order:rejected",
                    lambda row: self._on_order_terminal("order:rejected", row),
                )
            )
            self._unsubscribers.append(
                self.broker.on(
                    "order:modified", lambda row: self._forward_order("order:modified", row)
                )
            )
        except BaseException:
            self._unwire_broker()
            raise

    def _unwire_broker(self) -> None:
        for unsubscribe in self._unsubscribers:
            unsubscribe()
        self._unsubscribers.clear()

    def _on_order_filled(self, raw: dict[str, Any]) -> None:
        order = _order(raw)
        if not self._is_order_for_symbol(order):
            return
        self._emit("order:filled", {"symbol": self.symbol, **order})
        if _matches(self.pending_order, order):
            assert self.pending_order is not None
            pending = self.pending_order
            entry_fill = _number(order.get("avgFillPrice"), float(pending["entry"]))
            assert entry_fill is not None
            self.trade_id_counter += 1
            self.open_position = {
                "id": self.trade_id_counter,
                "symbol": self.symbol,
                "side": pending["side"],
                "entry": pending["entry"],
                "entryFill": entry_fill,
                "stop": pending["stop"],
                "takeProfit": pending["tp"],
                "size": _number(order.get("filledQty"), float(pending["fixedQty"])) or 0,
                "openTime": _number(
                    order.get("filledAt"), float(self.last_bar_time or self._now_ms())
                ),
                "_initRisk": abs(
                    _number(pending.get("meta", {}).get("_initRisk"), None)
                    or float(pending["entry"]) - float(pending["stop"])
                ),
                "_maxBarsInTrade": pending.get("meta", {}).get("_maxBarsInTrade"),
                "_maxHoldMin": pending.get("meta", {}).get("_maxHoldMin"),
                "_openedAtIndex": len(self.candle_buffer) - 1,
            }
            self.pending_order = None
            self.day_trades += 1
            self._emit(
                "position:opened",
                {"symbol": self.symbol, "position": _snapshot(self.open_position, entry_fill)},
            )
            self._spawn(self._persist_state())
            return
        if self.open_position and order.get("side") == self._opposite(self.open_position["side"]):
            closing = self.open_position
            fallback = _number(
                closing.get("_pendingExitPriceHint"),
                self._current_mark(float(closing["entryFill"])),
            )
            exit_price = _number(order.get("avgFillPrice"), fallback)
            assert exit_price is not None
            quantity = _number(order.get("filledQty"), float(closing["size"])) or 0
            direction = 1 if closing["side"] == "long" else -1
            pnl = (exit_price - float(closing["entryFill"])) * direction * quantity
            self.equity += pnl
            self.day_pnl += pnl
            self.open_position = None
            filled_at = _number(order.get("filledAt"), float(self._now_ms())) or self._now_ms()
            self.risk_manager.record_trade(pnl=pnl, time_ms=filled_at, equity=self.equity)
            trade = {
                "symbol": self.symbol,
                "id": closing["id"],
                "side": closing["side"],
                "entry": closing["entry"],
                "stop": closing["stop"],
                "takeProfit": closing["takeProfit"],
                "size": quantity,
                "openTime": closing["openTime"],
                "entryFill": closing["entryFill"],
                "_initRisk": closing.get("_initRisk"),
                "exit": {
                    "price": exit_price,
                    "time": filled_at,
                    "reason": closing.get("_pendingExitReason", "EXIT"),
                    "pnl": pnl,
                },
            }
            self.trades.append(trade)
            self._emit("position:closed", {"symbol": self.symbol, "trade": trade})
            self._spawn(self._finalize_trade(trade))

    def _on_order_terminal(self, event: str, raw: dict[str, Any]) -> None:
        order = _order(raw)
        if not self._is_order_for_symbol(order):
            return
        self._emit(event, {"symbol": self.symbol, **order})
        if _matches(self.pending_order, order):
            self.pending_order = None
            self._spawn(self._persist_state())
            return
        if self.open_position and _matches(
            {
                "orderId": self.open_position.get("_pendingExitOrderId"),
                "clientOrderId": self.open_position.get("_pendingExitClientOrderId"),
            },
            order,
        ):
            reason = f"exit order {event.removeprefix('order:')}"
            self.risk_manager.halt(reason)
            self._emit("risk:halt", {"symbol": self.symbol, "reason": reason})
            self._spawn(self._persist_state())

    async def _finalize_trade(self, trade: Mapping[str, Any]) -> None:
        await self.state_manager.append_trade(self.namespace, trade)
        await self._persist_state()

    @staticmethod
    def _opposite(side: object) -> str:
        return "sell" if side == "long" else "buy"

    def _append_bar(self, bar: dict[str, int | float]) -> None:
        self.candle_buffer.append(bar)
        maximum = max(10, self.warmup_bars + 100)
        if len(self.candle_buffer) > maximum:
            del self.candle_buffer[: len(self.candle_buffer) - maximum]
        self.last_bar_time = bar["time"]

    def _current_mark(self, fallback: float | None = None) -> float:
        if self.candle_buffer:
            return float(self.candle_buffer[-1]["close"])
        return float(fallback or 0)

    def _marked_equity(self, mark: float | None = None) -> float:
        if self.open_position is None:
            return self.equity
        price = (
            mark if mark is not None else self._current_mark(float(self.open_position["entryFill"]))
        )
        direction = 1 if self.open_position["side"] == "long" else -1
        return self.equity + (price - float(self.open_position["entryFill"])) * direction * float(
            self.open_position["size"]
        )

    def _signal_context(self, bar: Mapping[str, object]) -> dict[str, object]:
        mark = _number(bar["close"])
        assert mark is not None
        return {
            "candles": self.candle_buffer,
            "index": len(self.candle_buffer) - 1,
            "bar": bar,
            "equity": self._marked_equity(mark),
            "openPosition": _snapshot(self.open_position, mark) if self.open_position else None,
            "pendingOrder": self.pending_order,
        }

    async def _persist_state(self) -> None:
        await self.state_manager.save(
            self.namespace,
            {
                "openPosition": self.open_position,
                "pendingOrder": self.pending_order,
                "equity": self.equity,
                "candleBuffer": self.candle_buffer,
                "strategyState": {},
                "lastBarTime": self.last_bar_time,
                "dayPnl": self.day_pnl,
                "dayTrades": self.day_trades,
                "tradeIdCounter": self.trade_id_counter,
            },
        )

    async def _record_equity(self, time_ms: int | float, mark: float) -> None:
        point = {
            "time": float(time_ms),
            "timestamp": float(time_ms),
            "equity": self._marked_equity(mark),
        }
        self.eq_series.append(point)
        await self.state_manager.append_equity_point(self.namespace, point)
        self._emit("equity:update", {"symbol": self.symbol, **point})

    async def _submit_entry(self, decision: Mapping[str, Any], explicit_entry: bool) -> None:
        risk_fraction = _number(decision.get("riskFraction"))
        if risk_fraction is None:
            risk_pct = _number(decision.get("riskPct"))
            risk_fraction = risk_pct / 100 if risk_pct is not None else self.risk_pct / 100
        requested = _number(decision.get("qty"))
        if requested is None:
            requested = calculate_position_size(
                equity=self._marked_equity(float(decision["entry"])),
                entry=float(decision["entry"]),
                stop=float(decision["stop"]),
                risk_fraction=risk_fraction,
                qty_step=self.qty_step,
                min_qty=self.min_qty,
                max_leverage=self.max_leverage,
            )
        if requested < self.min_qty:
            return
        gate = self.risk_manager.can_open_position(
            time_ms=self.last_bar_time or self._now_ms(),
            position_count=1 if self.open_position else 0,
            position_value=abs(float(decision["entry"]) * requested),
            equity=self._marked_equity(float(decision["entry"])),
        )
        if not gate["ok"]:
            self._emit("risk:warning", {"symbol": self.symbol, "reason": gate["reason"]})
            return
        client_id = f"{self.namespace}-entry-{self._now_ms()}"
        expiry = int(_number(decision.get("_entryExpiryBars"), 5) or 5)
        index = len(self.candle_buffer) - 1
        order_type = "limit" if explicit_entry else "market"
        self.pending_order = {
            "side": decision["side"],
            "entry": decision["entry"],
            "stop": decision["stop"],
            "tp": decision["takeProfit"],
            "riskFrac": risk_fraction,
            "fixedQty": requested,
            "expiresAt": index + max(1, expiry),
            "startedAtIndex": index,
            "meta": dict(decision),
            "plannedRiskAbs": abs(float(decision["entry"]) - float(decision["stop"])),
            "orderId": None,
            "clientOrderId": client_id,
            "type": order_type,
            "_chasedCE": False,
        }
        receipt = _order(
            await self.broker.submit_order(
                {
                    "symbol": self.symbol,
                    "side": "buy" if decision["side"] == "long" else "sell",
                    "type": order_type,
                    "qty": requested,
                    "limitPrice": decision["entry"] if explicit_entry else None,
                    "clientOrderId": client_id,
                }
            )
        )
        await self._drain_event_tasks()
        if self.pending_order is None:
            return
        self.pending_order["orderId"] = receipt.get("orderId")
        self.pending_order["clientOrderId"] = receipt.get("clientOrderId") or client_id
        await self._persist_state()
        if receipt.get("status") == "filled" and self.pending_order is not None:
            self._on_order_filled(receipt)
            await self._drain_event_tasks()

    async def _submit_exit(self, reason: str, price: float, kind: str = "market") -> None:
        if self.open_position is None:
            return
        if self.open_position.get("_pendingExitClientOrderId"):
            return
        client_id = f"{self.namespace}-exit-{self._now_ms()}"
        self.open_position["_pendingExitReason"] = reason
        self.open_position["_pendingExitPriceHint"] = price
        self.open_position["_pendingExitClientOrderId"] = client_id
        try:
            receipt = _order(
                await self.broker.submit_order(
                    {
                        "symbol": self.symbol,
                        "side": self._opposite(self.open_position["side"]),
                        "type": kind,
                        "qty": self.open_position["size"],
                        "limitPrice": price if kind == "limit" else None,
                        "stopPrice": price if kind == "stop" else None,
                        "clientOrderId": client_id,
                    }
                )
            )
        except BaseException:
            self.risk_manager.halt("exit order submission outcome unknown")
            self._emit(
                "risk:halt",
                {"symbol": self.symbol, "reason": self.risk_manager.halt_reason},
            )
            await self._persist_state()
            raise
        if self.open_position is not None:
            self.open_position["_pendingExitOrderId"] = receipt.get("orderId")
        await self._drain_event_tasks()
        if receipt.get("status") == "rejected":
            if not self.risk_manager.halted:
                reason_text = str(receipt.get("rejectReason") or "broker rejected exit order")
                reason = f"exit order rejected: {reason_text}"
                self.risk_manager.halt(reason)
                self._emit("risk:halt", {"symbol": self.symbol, "reason": reason})
            await self._persist_state()
            return
        if receipt.get("status") == "filled" and self.open_position is not None:
            self._on_order_filled(receipt)
            await self._drain_event_tasks()
        await self._persist_state()

    async def _manage_pending(self) -> None:
        if self.pending_order is None:
            return
        index = len(self.candle_buffer) - 1
        if index > int(self.pending_order["expiresAt"]):
            order_id = self.pending_order.get("orderId")
            if order_id:
                try:
                    await self.broker.cancel_order(str(order_id))
                except Exception:
                    return
            self.pending_order = None
            await self._persist_state()
            return
        if bool(self.entry_chase.get("enabled")):
            meta = self.pending_order.get("meta")
            imbalance = meta.get("_imb") if isinstance(meta, Mapping) else None
            midpoint = _number(imbalance.get("mid")) if isinstance(imbalance, Mapping) else None
            elapsed = index - int(self.pending_order.get("startedAtIndex", index))
            after = int(_number(self.entry_chase.get("afterBars"), 2) or 2)
            if (
                midpoint is not None
                and not self.pending_order.get("_chasedCE")
                and elapsed >= max(1, after)
                and self.pending_order.get("orderId")
            ):
                try:
                    await self.broker.modify_order(
                        str(self.pending_order["orderId"]), {"limitPrice": midpoint}
                    )
                except Exception:
                    return
                self.pending_order["entry"] = midpoint
                self.pending_order["_chasedCE"] = True
                await self._persist_state()

    async def _manage_open_position(self, bar: Mapping[str, int | float]) -> None:
        if self.open_position is None:
            return
        if self.flatten_at_close and is_eod_bar(bar["time"]):
            await self._submit_exit("EOD", float(bar["close"]))
            return
        held = len(self.candle_buffer) - int(self.open_position.get("_openedAtIndex", 0))
        maximum = _number(self.open_position.get("_maxBarsInTrade"))
        if maximum is not None and maximum > 0 and held >= maximum:
            await self._submit_exit("TIME", float(bar["close"]))
            return
        outcome = oco_exit_check(
            side=str(self.open_position["side"]),
            stop=float(self.open_position["stop"]),
            tp=float(self.open_position["takeProfit"]),
            bar=bar,
            mode=str(self.oco.get("mode", "intrabar")),
            tie_break=str(self.oco.get("tieBreak", "pessimistic")),
        )
        if outcome["hit"]:
            await self._submit_exit(
                str(outcome["hit"]),
                float(cast(float, outcome["px"])),
                "limit" if outcome["hit"] == "TP" else "stop",
            )

    async def handle_bar(self, raw_bar: Mapping[str, object]) -> None:
        async with self._bar_lock:
            await self._drain_event_tasks()
            normalized = normalize_candles([raw_bar])
            if not normalized or not self.running:
                return
            bar = normalized[0]
            if self.last_bar_time is not None and bar["time"] <= self.last_bar_time:
                return
            self._append_bar(bar)
            self._emit("bar", {"symbol": self.symbol, "bar": bar})
            self.risk_manager.update(
                time_ms=bar["time"], equity=self._marked_equity(float(bar["close"]))
            )
            if self.open_position:
                await self._manage_open_position(bar)
            if self.pending_order:
                await self._manage_pending()
            can_trade = self.risk_manager.can_trade(time_ms=bar["time"])
            if not can_trade["ok"] and self.pending_order:
                order_id = self.pending_order.get("orderId")
                if order_id:
                    with suppress(Exception):
                        await self.broker.cancel_order(str(order_id))
                self.pending_order = None
                await self._persist_state()
            if not can_trade["ok"]:
                self._emit("risk:halt", {"symbol": self.symbol, "reason": can_trade["reason"]})
                await self._record_equity(bar["time"], float(bar["close"]))
                return
            if self.open_position is None and self.pending_order is None:
                context = self._signal_context(bar)
                signal_index = len(self.candle_buffer) - 1
                try:
                    raw_signal = call_signal_with_context(
                        self.signal, context, signal_index, bar, self.symbol
                    )
                    if inspect.isawaitable(raw_signal):
                        raw_signal = await raw_signal
                except StrategyError:
                    raise
                except Exception as error:
                    raise StrategyError(
                        f"signal() threw at index={context['index']}, symbol={self.symbol}: {error}"
                    ) from error
                if raw_signal:
                    self._emit(
                        "signal",
                        {
                            "symbol": self.symbol,
                            "t": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
                            "signal": raw_signal,
                        },
                    )
                decision = normalize_signal(raw_signal, bar, self.final_tp_r)
                if decision:
                    explicit = isinstance(raw_signal, Mapping) and any(
                        key in raw_signal for key in ("entry", "limit", "price")
                    )
                    await self._submit_entry(decision, explicit)
            await self._record_equity(bar["time"], float(bar["close"]))

    async def poll_once(self) -> None:
        poll = getattr(self.feed, "poll_once", None)
        if callable(poll):
            await _maybe_await(poll())
            return
        bars = await self.feed.get_historical_bars(self.symbol, self.interval, 2)
        for bar in sorted(bars, key=lambda item: _number(item.get("time"), -math.inf) or -math.inf):
            await self.handle_bar(bar)

    async def start(
        self,
        *,
        disconnect_feed_on_failure: bool = True,
        disconnect_broker_on_failure: bool = True,
    ) -> None:
        async with self._lifecycle_lock:
            if self.running:
                return
            try:
                if not self.broker.is_connected():
                    await self.broker.connect(self.broker_config)
                await self.feed.connect()
                self._wire_broker()
                clock = await self.clock.sync_with_broker(self.broker)
                if clock["warning"]:
                    self._emit("risk:warning", {"symbol": self.symbol, "reason": clock["warning"]})
                if self.use_broker_account_equity:
                    try:
                        account = await self.broker.get_account()
                        account_equity = _number(account.get("equity"))
                        if account_equity is not None and account_equity > 0:
                            self.equity = account_equity
                    except Exception:
                        pass
                persisted_raw = await self.state_manager.load(self.namespace)
                persisted = persisted_raw if isinstance(persisted_raw, Mapping) else None
                if persisted:
                    self.open_position = cast(dict[str, Any] | None, persisted.get("openPosition"))
                    self.pending_order = cast(dict[str, Any] | None, persisted.get("pendingOrder"))
                    self.equity = _number(persisted.get("equity"), self.equity) or self.equity
                    candles = persisted.get("candleBuffer")
                    self.candle_buffer = normalize_candles(
                        candles if isinstance(candles, Sequence) else []
                    )
                    self.last_bar_time = _number(persisted.get("lastBarTime"))
                    self.day_pnl = _number(persisted.get("dayPnl"), 0) or 0
                    self.day_trades = int(_number(persisted.get("dayTrades"), 0) or 0)
                    self.trade_id_counter = int(_number(persisted.get("tradeIdCounter"), 0) or 0)
                    self._emit(
                        "stateRestored", {"symbol": self.symbol, "namespace": self.namespace}
                    )
                warmup = await self.feed.get_historical_bars(
                    self.symbol, self.interval, max(1, self.warmup_bars)
                )
                for bar in normalize_candles(warmup or []):
                    if self.last_bar_time is None or bar["time"] > self.last_bar_time:
                        self._append_bar(bar)
                try:
                    positions = [_position(item) for item in await self.broker.get_positions()]
                except Exception:
                    positions = []
                reconcile = self.state_manager.reconcile(
                    persisted_state=persisted,
                    broker_positions=positions,
                    symbol=self.symbol,
                )
                pending_exit_reconcile_error: str | None = None
                if self.open_position and self.open_position.get("_pendingExitClientOrderId"):
                    try:
                        open_orders = [_order(item) for item in await self.broker.get_open_orders()]
                    except Exception:
                        pending_exit_reconcile_error = (
                            "unable to reconcile pending exit order on restart"
                        )
                    else:
                        expected_exit = {
                            "orderId": self.open_position.get("_pendingExitOrderId"),
                            "clientOrderId": self.open_position.get("_pendingExitClientOrderId"),
                        }
                        if not any(_matches(expected_exit, order) for order in open_orders):
                            pending_exit_reconcile_error = "pending exit order missing on restart"
                if reconcile["action"] == "adopt-broker" and reconcile["adoptedPosition"]:
                    self.open_position = {
                        **(self.open_position or {}),
                        **reconcile["adoptedPosition"],
                    }
                self.risk_manager.initialize(self.equity, self.last_bar_time or self._now_ms())
                self.risk_manager.day_trades = self.day_trades
                self.risk_manager.day_pnl = self.day_pnl
                if reconcile["action"] == "mismatch":
                    self.risk_manager.halt("position mismatch on restart")
                elif pending_exit_reconcile_error is not None:
                    self.risk_manager.halt(pending_exit_reconcile_error)
                self._emit("reconciled", {"symbol": self.symbol, "reconcile": reconcile})
                subscription = await self.feed.subscribe_bars(
                    self.symbol, self.interval, self.handle_bar
                )
                self.subscriptions.append(subscription)
                if self.mode == "polling":
                    start_polling = getattr(self.feed, "start_polling", None)
                    if callable(start_polling):
                        await _maybe_await(start_polling())
                self.started_at = self._now_ms()
                self.connected = True
                self.running = True
                self._emit("connected", {"symbol": self.symbol, "namespace": self.namespace})
                await self._persist_state()
            except BaseException:
                await self._rollback_start(
                    disconnect_feed=disconnect_feed_on_failure,
                    disconnect_broker=disconnect_broker_on_failure,
                )
                raise

    async def _rollback_start(
        self, *, disconnect_feed: bool = True, disconnect_broker: bool = True
    ) -> None:
        self.running = self.connected = False
        for subscription in self.subscriptions:
            subscription.unsubscribe()
        self.subscriptions.clear()
        self._unwire_broker()
        stop_polling = getattr(self.feed, "stop_polling", None)
        if callable(stop_polling):
            await _maybe_await(stop_polling())
        try:
            if disconnect_feed:
                await self.feed.disconnect()
        finally:
            if disconnect_broker and self.broker.is_connected():
                await self.broker.disconnect()

    async def stop(
        self,
        *,
        flatten_on_shutdown: bool = False,
        disconnect_feed: bool = True,
        disconnect_broker: bool = True,
    ) -> None:
        async with self._lifecycle_lock:
            if not self.connected and not self.running:
                return
            failure: BaseException | None = None
            try:
                await self._drain_event_tasks()
                if flatten_on_shutdown and self.open_position:
                    pending_exit_id = self.open_position.get("_pendingExitOrderId")
                    pending_exit_client_id = self.open_position.get("_pendingExitClientOrderId")
                    if pending_exit_client_id:
                        if not pending_exit_id:
                            raise RuntimeError(
                                "cannot safely flatten while an exit submission outcome is unknown"
                            )
                        await self.broker.cancel_order(str(pending_exit_id))
                        if self.open_position is not None:
                            for key in (
                                "_pendingExitOrderId",
                                "_pendingExitClientOrderId",
                                "_pendingExitReason",
                                "_pendingExitPriceHint",
                            ):
                                self.open_position.pop(key, None)
                    await self._submit_exit(
                        "SHUTDOWN", self._current_mark(float(self.open_position["entryFill"]))
                    )
                stop_polling = getattr(self.feed, "stop_polling", None)
                if callable(stop_polling):
                    await _maybe_await(stop_polling())
                for subscription in self.subscriptions:
                    subscription.unsubscribe()
                self.subscriptions.clear()
                await self._persist_state()
            except BaseException as error:
                failure = error
            finally:
                try:
                    if disconnect_feed:
                        await self.feed.disconnect()
                finally:
                    if disconnect_broker and self.broker.is_connected():
                        await self.broker.disconnect()
                    self._unwire_broker()
                    self.running = self.connected = False
                    self._emit("shutdown", {"symbol": self.symbol, "namespace": self.namespace})
            if failure is not None:
                raise failure

    def get_status(self) -> dict[str, Any]:
        mark = self._current_mark(
            float(self.open_position["entryFill"]) if self.open_position else None
        )
        return {
            "id": self.namespace,
            "symbol": self.symbol,
            "interval": self.interval,
            "running": self.running,
            "connected": self.connected,
            "startedAt": self.started_at,
            "lastBarTime": self.last_bar_time,
            "equity": self._marked_equity(),
            "realizedEquity": self.equity,
            "openPosition": _snapshot(self.open_position, mark) if self.open_position else None,
            "pendingOrder": self.pending_order,
            "dayPnl": self.day_pnl,
            "dayTrades": self.day_trades,
            "trades": len(self.trades),
            "risk": self.risk_manager.get_state(),
        }


def create_live_engine(**options: Any) -> LiveEngine:
    return LiveEngine(**options)
