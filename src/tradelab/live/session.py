"""Permission-gated trading sessions built on broker adapters."""

from __future__ import annotations

import asyncio
import inspect
import math
import os
import time
from collections.abc import Awaitable, Callable, Coroutine, Mapping, Sequence
from typing import Any, cast

from tradelab.engine.execution import round_step
from tradelab.errors import LiveTradingDisabledError, RiskRejectedError, ValidationError
from tradelab.models import BacktestResult
from tradelab.utils.position_sizing import calculate_position_size

from .broker import SessionBroker
from .events import EventBus
from .paper import PaperEngine
from .risk import RiskManager


def _now_ms() -> int:
    return time.time_ns() // 1_000_000


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValidationError(f"{name} must be finite", context={name: value})
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as error:
        raise ValidationError(f"{name} must be finite", context={name: value}) from error
    if not math.isfinite(number):
        raise ValidationError(f"{name} must be finite", context={name: value})
    return number


def _positive(value: object, name: str, *, allow_zero: bool = False) -> float:
    number = _finite(value, name)
    if number < 0 or (number == 0 and not allow_zero):
        raise ValidationError(f"{name} must be positive", context={name: value})
    return number


def _safe_dict(value: Mapping[str, Any]) -> dict[str, Any]:
    safe = BacktestResult({"value": value})["value"]
    if not isinstance(safe, dict):  # pragma: no cover - model invariant
        raise ValidationError("live payload must be JSON-safe")
    return safe


def _canonical(value: Mapping[str, Any], aliases: Mapping[str, str]) -> dict[str, Any]:
    output = dict(value)
    for snake, camel in aliases.items():
        if camel not in output and snake in output:
            output[camel] = output[snake]
        output.pop(snake, None)
    return _safe_dict(output)


_ORDER_ALIASES = {
    "order_id": "orderId",
    "client_order_id": "clientOrderId",
    "filled_qty": "filledQty",
    "avg_fill_price": "avgFillPrice",
    "filled_at": "filledAt",
    "limit_price": "limitPrice",
    "stop_price": "stopPrice",
    "time_in_force": "timeInForce",
    "reject_reason": "rejectReason",
}
_ACCOUNT_ALIASES = {"buying_power": "buyingPower", "margin_used": "marginUsed"}
_POSITION_ALIASES = {
    "avg_entry": "avgEntry",
    "avg_price": "avgPrice",
    "entry_price": "entryPrice",
    "market_value": "marketValue",
    "unrealized_pnl": "unrealizedPnl",
}


def _order_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    return _canonical(value, _ORDER_ALIASES)


def _account_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    return _canonical(value, _ACCOUNT_ALIASES)


def _position_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    return _canonical(value, _POSITION_ALIASES)


def _broker_side(side: object) -> str:
    normalized = str(side).lower()
    if normalized in {"long", "buy"}:
        return "buy"
    if normalized in {"short", "sell"}:
        return "sell"
    raise ValidationError("side must be long, short, buy, or sell")


def _opposite(side: object) -> str:
    return "sell" if str(side).lower() in {"long", "buy"} else "buy"


def _matches(reference: Mapping[str, Any] | None, order: Mapping[str, Any] | None) -> bool:
    if not reference or not order:
        return False
    return bool(
        (
            reference.get("orderId")
            and order.get("orderId")
            and reference["orderId"] == order["orderId"]
        )
        or (
            reference.get("clientOrderId")
            and order.get("clientOrderId")
            and reference["clientOrderId"] == order["clientOrderId"]
        )
    )


class TradingSession:
    """One paper or explicitly authorized live broker lifecycle."""

    def __init__(
        self,
        *,
        id: str | None = None,
        symbol: str | None = None,
        symbols: Sequence[str] | None = None,
        interval: str = "1m",
        broker: SessionBroker,
        mode: str = "paper",
        equity: object = 10_000,
        risk_pct: object = 1,
        max_daily_loss_pct: object = 0,
        max_position_pct: object = 1,
        max_gross_exposure_pct: object = 0,
        max_net_exposure_pct: object = 0,
        qty_step: object = 0.001,
        min_qty: object = 0.001,
        max_leverage: object = 2,
        confirm_live: bool = False,
        event_bus: EventBus | None = None,
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        if mode not in {"paper", "live"}:
            raise ValidationError("mode must be paper or live")
        if not isinstance(broker, SessionBroker):
            raise ValidationError("TradingSession requires a broker")
        if mode == "live":
            if not self.live_allowed():
                raise LiveTradingDisabledError(
                    "live trading is gated: set TRADELAB_ALLOW_LIVE=true"
                )
            if confirm_live is not True:
                raise LiveTradingDisabledError("live mode requires confirm_live=True")
            if isinstance(broker, PaperEngine):
                raise LiveTradingDisabledError("live mode requires a credentialed broker")
            supports_updates = getattr(broker, "supports_order_updates", None)
            if not callable(supports_updates) or supports_updates() is not True:
                raise LiveTradingDisabledError("live mode requires genuine streaming order updates")
        values = list(symbols) if symbols is not None else ([symbol] if symbol else [])
        cleaned = [str(value).strip() for value in values]
        if not cleaned or any(not value for value in cleaned):
            raise ValidationError("TradingSession requires a symbol or symbols")
        self.symbols = list(dict.fromkeys(cleaned))
        self.symbol = self.symbols[0]
        if not isinstance(interval, str) or not interval:
            raise ValidationError("interval must be a non-empty string")
        self.id = str(id or f"{self.symbol}-{interval}")
        self.interval = interval
        self.broker = broker
        self.mode = mode
        self.equity = _positive(equity, "equity", allow_zero=True)
        self._start_equity = self.equity
        self.risk_pct = _positive(risk_pct, "risk_pct", allow_zero=True)
        self.max_position_pct = _positive(max_position_pct, "max_position_pct", allow_zero=True)
        self.qty_step = _positive(qty_step, "qty_step")
        self.min_qty = _positive(min_qty, "min_qty", allow_zero=True)
        self.max_leverage = _positive(max_leverage, "max_leverage", allow_zero=True)
        self.event_bus = event_bus or EventBus()
        self.risk_manager = RiskManager(
            max_daily_loss_pct=max_daily_loss_pct,
            max_drawdown_pct=0,
            max_position_pct=self.max_position_pct * 100,
            max_gross_exposure_pct=max_gross_exposure_pct,
            max_net_exposure_pct=max_net_exposure_pct,
        )
        self.running = False
        self.events: list[dict[str, Any]] = []
        self.brackets: dict[str, dict[str, str]] = {}
        self._bracket_recoveries: dict[str, dict[str, str]] = {}
        self._oco_winners: dict[str, dict[str, str]] = {}
        self._pending_brackets: dict[str, dict[str, Any]] = {}
        self._entry_meta: dict[str, dict[str, Any]] = {}
        self._leg_meta: dict[str, dict[str, Any]] = {}
        self._cached_positions: list[dict[str, Any]] = []
        self._cached_open_orders: list[dict[str, Any]] = []
        self._last_prices: dict[str, float] = {}
        self._candle_buffers: dict[str, list[dict[str, Any]]] = {
            value: [] for value in self.symbols
        }
        self._was_halted = False
        self._coid_seq = 0
        self._clock_ms = clock_ms or _now_ms
        self._tasks: set[asyncio.Task[None]] = set()
        self._unsubscribers: list[Callable[[], None]] = []
        self._wired = False
        self._lifecycle_lock = asyncio.Lock()

    @staticmethod
    def live_allowed() -> bool:
        return os.environ.get("TRADELAB_ALLOW_LIVE") == "true"

    @property
    def last_price(self) -> float | None:
        return self._last_prices.get(self.symbol)

    @property
    def candle_buffer(self) -> list[dict[str, Any]]:
        return self._candle_buffers[self.symbol]

    def last_price_for(self, symbol: str | None = None) -> float | None:
        return self._last_prices.get(symbol or self.symbol)

    def candle_buffer_for(self, symbol: str | None = None) -> list[dict[str, Any]]:
        return list(self._candle_buffers.get(symbol or self.symbol, []))

    def _resolve_symbol(self, symbol: str | None) -> str:
        if symbol:
            if symbol not in self.symbols:
                raise ValidationError(f'symbol "{symbol}" is not configured for this session')
            return symbol
        if len(self.symbols) == 1:
            return self.symbol
        raise ValidationError("symbol is required for a multi-symbol session")

    def _record(self, event: str, payload: Mapping[str, Any] | None = None) -> None:
        value = _safe_dict(dict(payload or {}))
        message = {"event": event, "payload": value, "t": self._clock_ms()}
        self.events.append(message)
        if len(self.events) > 500:
            self.events.pop(0)
        forwarded = {"sessionId": self.id, "symbol": self.symbol, **value}
        self.event_bus.emit_event(event, forwarded)

    def _wire_broker_events(self) -> None:
        if self._wired:
            return
        try:
            self._unsubscribers.append(
                self.broker.on(
                    "order:filled", lambda order: self._on_broker_fill(_order_payload(order))
                )
            )
            self._unsubscribers.append(
                self.broker.on(
                    "order:submitted",
                    lambda order: self._record(
                        "order:submitted", self._with_meta(_order_payload(order))
                    ),
                )
            )
            self._unsubscribers.append(
                self.broker.on(
                    "order:canceled",
                    lambda order: self._on_terminal_order("order:canceled", _order_payload(order)),
                )
            )
            self._unsubscribers.append(
                self.broker.on(
                    "order:rejected",
                    lambda order: self._on_terminal_order("order:rejected", _order_payload(order)),
                )
            )
            self._unsubscribers.append(
                self.broker.on(
                    "equity:update",
                    lambda account: self._record("equity:update", _account_payload(account)),
                )
            )
        except BaseException:
            self._unwire_broker_events()
            raise
        self._wired = True

    def _unwire_broker_events(self) -> None:
        for unsubscribe in self._unsubscribers:
            unsubscribe()
        self._unsubscribers.clear()
        self._wired = False

    def _spawn(self, awaitable: Coroutine[Any, Any, None]) -> None:
        task: asyncio.Task[None] = asyncio.create_task(awaitable)
        self._tasks.add(task)

    async def _drain_tasks(self) -> None:
        while self._tasks:
            tasks = tuple(self._tasks)
            try:
                await asyncio.gather(*tasks)
            finally:
                self._tasks.difference_update(tasks)

    def _with_meta(self, order: Mapping[str, Any]) -> dict[str, Any]:
        value = dict(order)
        key = value.get("clientOrderId")
        if isinstance(key, str) and key in self._entry_meta:
            meta = self._entry_meta[key]
            value["sizing"] = meta["sizing"]
            if meta.get("rationale") is not None:
                value["rationale"] = meta["rationale"]
        elif isinstance(key, str) and key in self._leg_meta:
            value.update(self._leg_meta[key])
        return _safe_dict(value)

    def _on_terminal_order(self, event: str, order: dict[str, Any]) -> None:
        self._record(event, self._with_meta(order))
        for symbol, staged in tuple(self._pending_brackets.items()):
            if _matches(staged, order):
                self._pending_brackets.pop(symbol, None)
                break
        for symbol, bracket in tuple(self.brackets.items()):
            order_id = str(order.get("orderId") or "")
            if order_id not in {bracket.get("stopId"), bracket.get("targetId")}:
                continue
            if symbol in self._oco_winners:
                return
            leg = "stop" if order_id == bracket.get("stopId") else "target"
            reason = f"protective {leg} order {event.removeprefix('order:')}"
            self.risk_manager.halt(reason)
            self._record("risk:halt", {"symbol": symbol, "reason": reason})
            sibling = bracket.get("targetId") if leg == "stop" else bracket.get("stopId")
            if sibling:
                self._oco_winners[symbol] = {
                    "siblingId": str(sibling),
                    "winnerId": order_id,
                    "reason": "PROTECTIVE_LEG_TERMINATED",
                }
                self._spawn(self._cancel_oco_sibling(symbol, sibling))
            else:
                self.brackets.pop(symbol, None)
            return

    def _on_broker_fill(self, order: dict[str, Any]) -> None:
        self._record("order:filled", self._with_meta(order))
        for symbol, staged in tuple(self._pending_brackets.items()):
            if _matches(staged, order):
                self._pending_brackets.pop(symbol, None)
                parent_entry_id = staged.get("parent_entry_id")
                self._spawn(
                    self._attach_bracket(
                        side=staged.get("side"),
                        size=_finite(staged.get("size"), "bracket.size"),
                        stop=(
                            None
                            if staged.get("stop") is None
                            else _finite(staged.get("stop"), "bracket.stop")
                        ),
                        target=(
                            None
                            if staged.get("target") is None
                            else _finite(staged.get("target"), "bracket.target")
                        ),
                        rr=(
                            None
                            if staged.get("rr") is None
                            else _finite(staged.get("rr"), "bracket.rr")
                        ),
                        entry_ref=_finite(staged.get("entry_ref"), "bracket.entry_ref"),
                        symbol=symbol,
                        receipt=order,
                        parent_entry_id=(
                            str(parent_entry_id)
                            if parent_entry_id is not None
                            else str(order.get("clientOrderId") or "") or None
                        ),
                    )
                )
                return
        for symbol, bracket in tuple(self.brackets.items()):
            order_id = order.get("orderId")
            if order_id not in {bracket.get("stopId"), bracket.get("targetId")}:
                continue
            sibling = (
                bracket.get("targetId")
                if order_id == bracket.get("stopId")
                else bracket.get("stopId")
            )
            reason = "SL" if order_id == bracket.get("stopId") else "TP"
            if symbol not in self._oco_winners:
                self._oco_winners[symbol] = {
                    "siblingId": str(sibling or ""),
                    "winnerId": str(order_id or ""),
                    "reason": reason,
                }
                self._record("position:closed", {"symbol": symbol, "reason": reason})
            if sibling:
                cancel_nowait = getattr(self.broker, "cancel_order_nowait", None)
                if callable(cancel_nowait):
                    try:
                        cancel_nowait(sibling)
                    except Exception as error:
                        self._record(
                            "error",
                            {
                                "symbol": symbol,
                                "operation": "oco-cancel",
                                "orderId": sibling,
                                "message": str(error),
                            },
                        )
                        raise
                    else:
                        self.brackets.pop(symbol, None)
                        self._oco_winners.pop(symbol, None)
                else:
                    self._spawn(self._cancel_oco_sibling(symbol, sibling))
            else:
                self.brackets.pop(symbol, None)
                self._oco_winners.pop(symbol, None)
            return

    async def _cancel_oco_sibling(self, symbol: str, order_id: str) -> None:
        try:
            await self.broker.cancel_order(order_id)
        except Exception as error:
            self._record(
                "error",
                {
                    "symbol": symbol,
                    "operation": "oco-cancel",
                    "orderId": order_id,
                    "message": str(error),
                },
            )
            raise
        else:
            self.brackets.pop(symbol, None)
            self._oco_winners.pop(symbol, None)

    async def start(self) -> dict[str, Any]:
        async with self._lifecycle_lock:
            if self.running:
                return self.get_status()
            self._wire_broker_events()
            try:
                if not self.broker.is_connected():
                    await self.broker.connect({})
                account = _account_payload(await self.broker.get_account())
                account_equity = account.get("equity")
                if account_equity is not None:
                    self.equity = _positive(account_equity, "account.equity", allow_zero=True)
                    self._start_equity = self.equity
                self.risk_manager.initialize(self.equity, self._clock_ms())
                self.running = True
                self._record("connected", {"mode": self.mode})
                return await self.refresh()
            except BaseException:
                self.running = False
                try:
                    if self.broker.is_connected():
                        await self.broker.disconnect()
                finally:
                    self._unwire_broker_events()
                raise

    async def stop(self, *, flatten: bool = False) -> None:
        async with self._lifecycle_lock:
            if not self.running and not self.broker.is_connected():
                return
            try:
                await self._drain_tasks()
                if flatten:
                    await self.flatten()
            finally:
                self.running = False
                try:
                    if self.broker.is_connected():
                        await self.broker.disconnect()
                finally:
                    self._unwire_broker_events()
                    self._record("shutdown", {})

    async def push_bar(self, bar: Mapping[str, object], symbol: str | None = None) -> None:
        if not self.running:
            raise ValidationError("session not started")
        resolved = self._resolve_symbol(symbol)
        if not isinstance(bar, Mapping):
            raise ValidationError("bar must be a mapping")
        close = _finite(bar.get("close"), "bar.close")
        time_ms = _finite(bar.get("time"), "bar.time")
        clean = dict(bar)
        self._last_prices[resolved] = close
        simulate = getattr(self.broker, "simulate_bar", None)
        if callable(simulate):
            await simulate(resolved, self.interval, clean)
        await self._drain_tasks()
        buffer = self._candle_buffers.setdefault(resolved, [])
        buffer.append(_safe_dict(clean))
        if len(buffer) > 200:
            buffer.pop(0)
        self._record("bar", {"symbol": resolved, "close": close, "time": time_ms})
        await self._sync_equity_and_risk()
        await self.refresh()

    def _next_client_id(self, leg: str) -> str:
        self._coid_seq += 1
        return f"{self.id}-{leg}-{self._clock_ms()}-{self._coid_seq}"

    async def place_order(
        self,
        *,
        side: object,
        type: str = "market",
        qty: object | None = None,
        risk_pct: object | None = None,
        stop: object | None = None,
        target: object | None = None,
        rr: object | None = None,
        limit_price: object | None = None,
        rationale: Mapping[str, object] | None = None,
        symbol: str | None = None,
    ) -> dict[str, Any]:
        if not self.running:
            raise ValidationError("session not started")
        if self.risk_manager.halted:
            raise RiskRejectedError("session is risk-halted for the day")
        resolved = self._resolve_symbol(symbol)
        entry = (
            _finite(limit_price, "limit_price")
            if type == "limit"
            else self.last_price_for(resolved)
        )
        if entry is None:
            raise ValidationError("no price available; push_bar() a price first")
        fraction = (
            _positive(risk_pct, "risk_pct", allow_zero=True) / 100
            if risk_pct is not None
            else self.risk_pct / 100
        )
        stop_value = None if stop is None else _finite(stop, "stop")
        if qty is None:
            if stop_value is None:
                raise ValidationError("risk-based sizing requires a stop")
            size = calculate_position_size(
                equity=self.equity,
                entry=entry,
                stop=stop_value,
                risk_fraction=fraction,
                qty_step=self.qty_step,
                min_qty=self.min_qty,
                max_leverage=self.max_leverage,
            )
        else:
            size = _positive(qty, "qty")
        size = round_step(size, self.qty_step)
        if size < self.min_qty:
            raise ValidationError(f"sized below min_qty ({size})")
        rr_value = None if rr is None else _positive(rr, "rr")
        target_value = None if target is None else _finite(target, "target")
        direction = 1 if _broker_side(side) == "buy" else -1
        if target_value is None and rr_value is not None and stop_value is not None:
            target_value = entry + direction * rr_value * abs(entry - stop_value)
        sizing = {
            "entry": entry,
            "stop": stop_value,
            "target": target_value,
            "rr": rr_value,
            "riskFraction": fraction,
            "riskAmount": self.equity * fraction,
            "qty": size,
            "notional": size * entry,
        }
        positions = self._cached_positions
        gross = abs(size * entry)
        net = direction * size * entry
        for position in positions:
            price = position.get(
                "avgEntry", position.get("avgPrice", position.get("entryPrice", entry))
            )
            position_value = (
                _finite(position["marketValue"], "position.marketValue")
                if position.get("marketValue") is not None
                else _finite(position.get("qty", 0), "position.qty")
                * _finite(price, "position.price")
            )
            gross += abs(position_value)
            net += position_value if position.get("side") in {"long", "buy"} else -position_value
        gate = self.risk_manager.can_open_position(
            time_ms=self._clock_ms(),
            position_count=len(positions),
            position_value=abs(size * entry),
            gross_exposure=gross,
            net_exposure=net,
            equity=self.equity,
        )
        if not gate["ok"]:
            raise RiskRejectedError(f"risk rejected: {gate['reason']}")
        rationale_value = None if rationale is None else _safe_dict(dict(rationale))
        client_id = self._next_client_id("entry")
        self._entry_meta[client_id] = {"sizing": sizing, "rationale": rationale_value}
        receipt = _order_payload(
            await self.broker.submit_order(
                {
                    "symbol": resolved,
                    "side": _broker_side(side),
                    "type": type,
                    "qty": size,
                    "limitPrice": limit_price if type == "limit" else None,
                    "clientOrderId": client_id,
                }
            )
        )
        if stop_value is not None or target_value is not None or rr_value is not None:
            parent = receipt.get("clientOrderId") or client_id
            bracket = {
                "side": side,
                "size": size,
                "stop": stop_value,
                "target": target_value,
                "rr": rr_value,
                "entry_ref": entry,
            }
            if receipt.get("status") == "filled":
                await self._attach_bracket(
                    side=side,
                    size=size,
                    stop=stop_value,
                    target=target_value,
                    rr=rr_value,
                    entry_ref=entry,
                    receipt=receipt,
                    parent_entry_id=str(parent),
                    symbol=resolved,
                )
            elif receipt.get("status") != "rejected":
                self._pending_brackets[resolved] = {
                    **bracket,
                    "orderId": receipt.get("orderId"),
                    "clientOrderId": receipt.get("clientOrderId") or client_id,
                    "parent_entry_id": str(parent),
                }
        await self.refresh()
        return receipt

    async def _attach_bracket(
        self,
        *,
        side: object,
        size: float,
        stop: float | None,
        target: float | None,
        rr: float | None,
        entry_ref: float,
        receipt: Mapping[str, Any],
        parent_entry_id: str | None,
        symbol: str,
        **_ignored: object,
    ) -> None:
        entry_fill = _finite(receipt.get("avgFillPrice", entry_ref), "entry fill")
        risk = abs(entry_fill - stop) if stop is not None else None
        target_price = target
        if target_price is None and rr is not None and risk is not None:
            target_price = entry_fill + (1 if _broker_side(side) == "buy" else -1) * rr * risk
        bracket: dict[str, str] = {}
        stop_client_id: str | None = None
        if stop is not None:
            client_id = self._next_client_id("stop")
            stop_client_id = client_id
            if parent_entry_id:
                self._leg_meta[client_id] = {"parentEntryId": parent_entry_id, "leg": "stop"}
            order = _order_payload(
                await self.broker.submit_order(
                    {
                        "symbol": symbol,
                        "side": _opposite(side),
                        "type": "stop",
                        "qty": size,
                        "stopPrice": stop,
                        "clientOrderId": client_id,
                    }
                )
            )
            if order.get("status") == "rejected":
                reason = "protective stop order rejected"
                self.risk_manager.halt(reason)
                self._record("risk:halt", {"symbol": symbol, "reason": reason})
                raise RuntimeError(reason)
            bracket["stopId"] = str(order["orderId"])
        if target_price is not None:
            client_id = self._next_client_id("target")
            if parent_entry_id:
                self._leg_meta[client_id] = {"parentEntryId": parent_entry_id, "leg": "target"}
            try:
                order = _order_payload(
                    await self.broker.submit_order(
                        {
                            "symbol": symbol,
                            "side": _opposite(side),
                            "type": "limit",
                            "qty": size,
                            "limitPrice": target_price,
                            "clientOrderId": client_id,
                        }
                    )
                )
                if order.get("status") == "rejected":
                    raise RuntimeError("protective target order rejected")
            except BaseException:
                stop_id = bracket.get("stopId")
                if stop_id is not None:
                    recovery = {"stopId": stop_id}
                    if stop_client_id is not None:
                        recovery["clientOrderId"] = stop_client_id
                    self.brackets[symbol] = {"stopId": stop_id}
                    self._bracket_recoveries[symbol] = dict(recovery)
                    try:
                        await asyncio.shield(self.broker.cancel_order(stop_id))
                    except BaseException as compensation_error:
                        reason = "bracket compensation failed; trading halted"
                        self.risk_manager.halt(reason)
                        self._record(
                            "error",
                            {
                                "symbol": symbol,
                                "operation": "bracket-compensation",
                                "orderId": stop_id,
                                "message": reason,
                            },
                        )
                        self._record("risk:halt", {"reason": reason})
                        raise RuntimeError(reason) from compensation_error
                    else:
                        self.brackets.pop(symbol, None)
                        self._bracket_recoveries.pop(symbol, None)
                raise
            bracket["targetId"] = str(order["orderId"])
        self.brackets[symbol] = bracket

    async def _sync_equity_and_risk(self) -> None:
        account = _account_payload(await self.broker.get_account())
        if account.get("equity") is None:
            return
        next_equity = _positive(account["equity"], "account.equity", allow_zero=True)
        delta = next_equity - self.equity
        self.equity = next_equity
        if delta:
            self.risk_manager.record_trade(pnl=delta, time_ms=self._clock_ms(), equity=self.equity)
        else:
            self.risk_manager.update(time_ms=self._clock_ms(), equity=self.equity)
        halted = self.risk_manager.halted
        if halted and not self._was_halted:
            self._record("risk:halt", {"reason": self.risk_manager.halt_reason or "risk halt"})
        self._was_halted = halted

    async def close_position(self, symbol: str | None = None) -> dict[str, Any] | None:
        resolved = self._resolve_symbol(symbol)
        positions = [_position_payload(item) for item in await self.broker.get_positions()]
        position = next((item for item in positions if item.get("symbol") == resolved), None)
        if position is None:
            return None
        bracket = self.brackets.pop(resolved, None)
        self._bracket_recoveries.pop(resolved, None)
        if bracket:
            for order_id in bracket.values():
                await self.broker.cancel_order(order_id)
        receipt = _order_payload(
            await self.broker.submit_order(
                {
                    "symbol": resolved,
                    "side": _opposite(position["side"]),
                    "type": "market",
                    "qty": position["qty"],
                    "clientOrderId": self._next_client_id("close"),
                }
            )
        )
        await self._sync_equity_and_risk()
        await self.refresh()
        return receipt

    async def flatten(self) -> None:
        for position in [_position_payload(item) for item in await self.broker.get_positions()]:
            await self.close_position(str(position["symbol"]))
        for order in [_order_payload(item) for item in await self.broker.get_open_orders()]:
            await self.broker.cancel_order(str(order["orderId"]))
        self._pending_brackets.clear()
        self.brackets.clear()
        self._bracket_recoveries.clear()
        self._oco_winners.clear()
        await self.refresh()

    async def cancel_order(self, order_id: str) -> None:
        await self.broker.cancel_order(order_id)
        await self.refresh()

    async def get_account(self) -> dict[str, Any]:
        return _account_payload(await self.broker.get_account())

    async def get_positions(self) -> list[dict[str, Any]]:
        return [_position_payload(item) for item in await self.broker.get_positions()]

    def recent_events(self, limit: int = 50) -> list[dict[str, Any]]:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
            raise ValidationError("limit must be a non-negative integer")
        return [_safe_dict(event) for event in self.events[-limit:]] if limit else []

    def get_status(self) -> dict[str, Any]:
        risk = self.risk_manager.get_state()
        return _safe_dict(
            {
                "id": self.id,
                "symbol": self.symbol,
                "symbols": self.symbols,
                "interval": self.interval,
                "mode": self.mode,
                "running": self.running,
                "equity": self.equity,
                "dayPnl": risk.get("dayPnl", 0),
                "lastPrice": self.last_price,
                "positions": self._cached_positions,
                "openOrders": self._cached_open_orders,
                "risk": {"halted": bool(risk.get("halted")), **risk},
            }
        )

    async def refresh(self) -> dict[str, Any]:
        await self._drain_tasks()
        self._cached_positions = [
            _position_payload(item) for item in await self.broker.get_positions()
        ]
        self._cached_open_orders = [
            _order_payload(item) for item in await self.broker.get_open_orders()
        ]
        open_ids = {str(order.get("orderId")) for order in self._cached_open_orders}
        retried = False
        for symbol, recovery in tuple(self._bracket_recoveries.items()):
            stop_id = recovery.get("stopId")
            if stop_id and stop_id in open_ids:
                try:
                    await self.broker.cancel_order(stop_id)
                except Exception:
                    self._record(
                        "error",
                        {
                            "symbol": symbol,
                            "operation": "bracket-compensation",
                            "orderId": stop_id,
                            "message": "bracket compensation retry failed",
                        },
                    )
                    raise
                retried = True
            self.brackets.pop(symbol, None)
            self._bracket_recoveries.pop(symbol, None)
        for symbol, winner in tuple(self._oco_winners.items()):
            sibling = winner.get("siblingId")
            if sibling and sibling in open_ids:
                await self._cancel_oco_sibling(symbol, sibling)
                retried = True
            else:
                self.brackets.pop(symbol, None)
                self._oco_winners.pop(symbol, None)
        if retried:
            self._cached_open_orders = [
                _order_payload(item) for item in await self.broker.get_open_orders()
            ]
        account = _account_payload(await self.broker.get_account())
        if account.get("equity") is not None:
            self.equity = _positive(account["equity"], "account.equity", allow_zero=True)
        return self.get_status()


BrokerFactory = Callable[[dict[str, object]], SessionBroker | Awaitable[SessionBroker]]


class SessionManager:
    """Own and clean up named sessions without retaining stopped instances."""

    def __init__(self, *, broker_factory: BrokerFactory | None = None) -> None:
        self.sessions: dict[str, TradingSession] = {}
        self.broker_factory = broker_factory
        self._lock = asyncio.Lock()

    async def create(
        self,
        *,
        id: str | None = None,
        mode: str = "paper",
        symbol: str | None = None,
        symbols: Sequence[str] | None = None,
        interval: str = "1m",
        equity: object = 10_000,
        confirm_live: bool = False,
        broker: SessionBroker | None = None,
        **options: object,
    ) -> TradingSession:
        values = list(symbols) if symbols else ([symbol] if symbol else [])
        candidate_id = str(id or (f"{values[0]}-{interval}" if values else ""))
        if not candidate_id:
            raise ValidationError("session requires an id or symbol")
        async with self._lock:
            if candidate_id in self.sessions:
                raise ValidationError(f'session "{candidate_id}" already exists')
            resolved = broker
            owned = False
            try:
                if mode == "live":
                    if not TradingSession.live_allowed() or confirm_live is not True:
                        raise LiveTradingDisabledError(
                            "live mode requires TRADELAB_ALLOW_LIVE=true and confirm_live=True"
                        )
                    if resolved is None and self.broker_factory is not None:
                        created = self.broker_factory({"symbol": symbol, **options})
                        resolved = await created if inspect.isawaitable(created) else created
                        owned = True
                    if resolved is None:
                        raise LiveTradingDisabledError("live mode requires a credentialed broker")
                if resolved is None:
                    resolved = PaperEngine(equity=equity)
                    owned = True
                session = cast(
                    TradingSession,
                    cast(Any, TradingSession)(
                        id=candidate_id,
                        symbol=symbol,
                        symbols=symbols,
                        interval=interval,
                        broker=resolved,
                        mode=mode,
                        equity=equity,
                        confirm_live=confirm_live,
                        **options,
                    ),
                )
                await session.start()
            except BaseException:
                if owned and resolved is not None and resolved.is_connected():
                    await resolved.disconnect()
                raise
            self.sessions[session.id] = session
            return session

    def get(self, id: str) -> TradingSession | None:
        return self.sessions.get(id)

    def list(self) -> list[TradingSession]:
        return list(self.sessions.values())

    async def remove(self, id: str, *, flatten: bool = True) -> None:
        async with self._lock:
            session = self.sessions.get(id)
            if session is None:
                return
            await session.stop(flatten=flatten)
            self.sessions.pop(id, None)

    async def halt_all(self) -> None:
        async with self._lock:
            sessions = list(self.sessions.values())
            outcomes = await asyncio.gather(
                *(session.stop(flatten=True) for session in sessions), return_exceptions=True
            )
            self.sessions.clear()
            failure = next(
                (outcome for outcome in outcomes if isinstance(outcome, BaseException)), None
            )
            if failure is not None:
                raise failure


def create_session_manager(*, broker_factory: BrokerFactory | None = None) -> SessionManager:
    return SessionManager(broker_factory=broker_factory)
