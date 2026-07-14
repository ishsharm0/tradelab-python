"""In-process paper broker with deterministic bar-driven fills."""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from typing import Any

from tradelab.engine.execution import apply_fill, round_step, touched_limit
from tradelab.errors import ValidationError
from tradelab.models import BacktestResult

from .broker import BrokerAdapter


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


def _optional_finite(value: object, name: str) -> float | None:
    return None if value is None else _finite(value, name)


def _positive(value: object, name: str, *, allow_zero: bool = False) -> float:
    number = _finite(value, name)
    if number < 0 or (not allow_zero and number == 0):
        operator = "non-negative" if allow_zero else "positive"
        raise ValidationError(f"{name} must be {operator}", context={name: value})
    return number


def _side(value: object) -> str:
    normalized = str(value or "").lower()
    if normalized not in {"buy", "sell"}:
        raise ValidationError(f'unsupported paper order side "{value}"')
    return normalized


def _order_type(value: object) -> str:
    normalized = str(value or "market").lower()
    if normalized not in {"market", "limit", "stop", "stop_limit"}:
        raise ValidationError(f'unsupported paper order type "{value}"')
    return normalized


def _symbol(value: object) -> str:
    symbol = str(value or "").strip()
    if not symbol:
        raise ValidationError("symbol must be a non-empty string")
    return symbol


def _snapshot(value: Mapping[str, Any]) -> dict[str, Any]:
    result = BacktestResult({"value": value})["value"]
    if not isinstance(result, dict):  # pragma: no cover - guaranteed by model
        raise ValidationError("paper payload must be JSON-safe")
    return result


_ORDER_KEYS = (
    "orderId",
    "clientOrderId",
    "status",
    "filledQty",
    "avgFillPrice",
    "filledAt",
    "symbol",
    "side",
    "type",
    "qty",
    "limitPrice",
    "stopPrice",
    "timeInForce",
    "rejectReason",
)


def _clone_order(order: Mapping[str, Any]) -> dict[str, Any]:
    return _snapshot({key: order.get(key) for key in _ORDER_KEYS})


class PaperEngine(BrokerAdapter):
    """Broker adapter whose orders fill from explicitly simulated bars."""

    def __init__(
        self,
        *,
        equity: object = 10_000,
        currency: str = "USD",
        slippage_bps: object = 0,
        fee_bps: object = 0,
        costs: Mapping[str, object] | None = None,
        qty_step: object = 0.001,
    ) -> None:
        super().__init__()
        self.connected = False
        self.config: dict[str, object] = {}
        self.currency = str(currency)
        self.starting_equity = _positive(equity, "equity", allow_zero=True)
        self.cash = self.starting_equity
        self.slippage_bps = _finite(slippage_bps, "slippage_bps")
        self.fee_bps = _finite(fee_bps, "fee_bps")
        if costs is not None and not isinstance(costs, Mapping):
            raise ValidationError("costs must be a mapping")
        self.costs = deepcopy(dict(costs)) if costs is not None else None
        self.qty_step = _positive(qty_step, "qty_step")
        self.positions: dict[str, dict[str, Any]] = {}
        self.open_orders: dict[str, dict[str, Any]] = {}
        self.order_history: dict[str, dict[str, Any]] = {}
        self.last_prices: dict[str, float] = {}
        self._bar_subscribers: dict[str, list[Callable[[dict[str, Any]], object]]] = {}
        self._trade_subscribers: dict[str, list[Callable[[dict[str, Any]], object]]] = {}
        self._quote_subscribers: dict[str, list[Callable[[dict[str, Any]], object]]] = {}
        self._historical_bars: dict[str, list[dict[str, Any]]] = {}
        self._order_id_counter = 1

    async def connect(self, config: Mapping[str, object] | None = None) -> None:
        if config is not None and not isinstance(config, Mapping):
            raise ValidationError("broker config must be a mapping")
        self.config = dict(config or {})
        self.connected = True

    async def disconnect(self) -> None:
        self.connected = False
        self._bar_subscribers.clear()
        self._trade_subscribers.clear()
        self._quote_subscribers.clear()

    def is_connected(self) -> bool:
        return self.connected

    def supports_paper_native(self) -> bool:
        return True

    def _position_mark(self, position: Mapping[str, Any]) -> dict[str, float]:
        mark = self.last_prices.get(str(position["symbol"]), float(position["avgEntry"]))
        unrealized = (
            (mark - float(position["avgEntry"])) * float(position["qty"])
            if position["side"] == "long"
            else (float(position["avgEntry"]) - mark) * float(position["qty"])
        )
        return {
            "mark": mark,
            "marketValue": mark * float(position["qty"]),
            "unrealizedPnl": unrealized,
        }

    def _summary(self) -> tuple[float, float]:
        unrealized = 0.0
        market_value = 0.0
        for position in self.positions.values():
            marked = self._position_mark(position)
            unrealized += marked["unrealizedPnl"]
            market_value += marked["marketValue"]
        return unrealized, market_value

    async def get_account(self) -> dict[str, Any]:
        unrealized, market_value = self._summary()
        equity = self.cash + unrealized
        return _snapshot(
            {
                "equity": equity,
                "buyingPower": max(0.0, equity),
                "cash": self.cash,
                "currency": self.currency,
                "marginUsed": max(0.0, market_value - self.cash),
            }
        )

    async def get_positions(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for position in self.positions.values():
            marked = self._position_mark(position)
            rows.append(
                _snapshot(
                    {
                        "symbol": position["symbol"],
                        "side": position["side"],
                        "qty": position["qty"],
                        "avgEntry": position["avgEntry"],
                        "marketValue": marked["marketValue"],
                        "unrealizedPnl": marked["unrealizedPnl"],
                    }
                )
            )
        return rows

    @staticmethod
    def _stream_key(symbol: str, interval: str = "*") -> str:
        return f"{symbol}::{interval}"

    def _subscribe(
        self,
        subscribers: dict[str, list[Callable[[dict[str, Any]], object]]],
        key: str,
        handler: Callable[[dict[str, Any]], object],
    ) -> Callable[[], None]:
        if not callable(handler):
            raise ValidationError("subscription handler must be callable")
        subscribers.setdefault(key, []).append(handler)
        removed = False

        def unsubscribe() -> None:
            nonlocal removed
            if removed:
                return
            removed = True
            subscribers[key] = [
                candidate for candidate in subscribers.get(key, []) if candidate is not handler
            ]

        return unsubscribe

    async def subscribe_bars(
        self, symbol: str, interval: str, handler: Callable[[dict[str, Any]], object]
    ) -> Callable[[], None]:
        return self._subscribe(self._bar_subscribers, self._stream_key(symbol, interval), handler)

    async def subscribe_trades(
        self, symbol: str, handler: Callable[[dict[str, Any]], object]
    ) -> Callable[[], None]:
        return self._subscribe(self._trade_subscribers, symbol, handler)

    async def subscribe_quotes(
        self, symbol: str, handler: Callable[[dict[str, Any]], object]
    ) -> Callable[[], None]:
        return self._subscribe(self._quote_subscribers, symbol, handler)

    @staticmethod
    async def _emit_to(
        subscribers: Mapping[str, Sequence[Callable[[dict[str, Any]], object]]],
        key: str,
        payload: dict[str, Any],
    ) -> None:
        for handler in tuple(subscribers.get(key, ())):
            result = handler(_snapshot(payload))
            if inspect.isawaitable(result):
                await result

    def set_historical_bars(
        self, symbol: str, interval: str, bars: Sequence[Mapping[str, object]]
    ) -> None:
        if isinstance(bars, (str, bytes)) or not isinstance(bars, Sequence):
            raise ValidationError("bars must be a sequence")
        self._historical_bars[self._stream_key(symbol, interval)] = [dict(bar) for bar in bars]

    async def get_historical_bars(
        self, symbol: str, interval: str, limit: int = 200
    ) -> Sequence[Mapping[str, object]]:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
            raise ValidationError("limit must be a non-negative integer")
        values = self._historical_bars.get(self._stream_key(symbol, interval), [])
        return deepcopy(values[max(0, len(values) - limit) :])

    def _next_order_id(self) -> str:
        value = f"paper-{self._order_id_counter}"
        self._order_id_counter += 1
        return value

    def _record_order(self, order: Mapping[str, Any]) -> None:
        self.order_history[str(order["orderId"])] = deepcopy(dict(order))

    def _reject_order(self, order: dict[str, Any], reason: str) -> dict[str, Any]:
        order["status"] = "rejected"
        order["rejectReason"] = reason
        self._record_order(order)
        self.open_orders.pop(str(order["orderId"]), None)
        receipt = _clone_order(order)
        self.emit("order:rejected", receipt)
        return receipt

    def _fill_order(
        self,
        order: dict[str, Any],
        fill_price: float,
        kind: str = "market",
        fill_time: float | None = None,
    ) -> dict[str, Any]:
        quantity = _positive(order["qty"], "qty")
        side = _side(order["side"])
        fill = apply_fill(
            fill_price,
            "long" if side == "buy" else "short",
            slippage_bps=self.slippage_bps,
            fee_bps=self.fee_bps,
            kind=kind,
            qty=quantity,
            costs=self.costs,
        )
        fill_price = _finite(fill["price"], "fill.price")
        fee_total = _positive(fill["fee_total"], "fill.fee_total", allow_zero=True)
        _finite(fill_price * quantity, "fill.notional")
        incoming = quantity if side == "buy" else -quantity
        symbol = str(order["symbol"])
        position = self.positions.get(symbol)
        realized = 0.0
        next_position: dict[str, Any] | None
        if position is None:
            next_position = {
                "symbol": order["symbol"],
                "side": "long" if incoming > 0 else "short",
                "qty": abs(incoming),
                "avgEntry": fill_price,
            }
        else:
            existing = (
                float(position["qty"]) if position["side"] == "long" else -float(position["qty"])
            )
            if existing * incoming >= 0:
                total = _positive(abs(existing) + abs(incoming), "position.qty")
                weighted_cost = _finite(
                    abs(existing) * float(position["avgEntry"]) + abs(incoming) * fill_price,
                    "position.weighted_cost",
                )
                average = _finite(weighted_cost / total, "position.avgEntry")
                next_position = {
                    "symbol": order["symbol"],
                    "side": "long" if existing + incoming >= 0 else "short",
                    "qty": abs(existing + incoming),
                    "avgEntry": average,
                }
            else:
                closed = min(abs(existing), abs(incoming))
                realized = (
                    (fill_price - float(position["avgEntry"])) * closed
                    if position["side"] == "long"
                    else (float(position["avgEntry"]) - fill_price) * closed
                )
                realized = _finite(realized, "realizedPnl")
                remaining = existing + incoming
                if remaining == 0:
                    next_position = None
                elif existing * remaining > 0:
                    next_position = {
                        "symbol": order["symbol"],
                        "side": position["side"],
                        "qty": abs(remaining),
                        "avgEntry": position["avgEntry"],
                    }
                else:
                    next_position = {
                        "symbol": order["symbol"],
                        "side": "long" if remaining > 0 else "short",
                        "qty": abs(remaining),
                        "avgEntry": fill_price,
                    }
        next_cash = _finite(self.cash - fee_total + realized, "cash")
        if next_position is None:
            self.positions.pop(symbol, None)
        else:
            _snapshot(next_position)
            self.positions[symbol] = next_position
        self.cash = next_cash
        order.update(
            {
                "status": "filled",
                "filledQty": quantity,
                "avgFillPrice": fill_price,
                "filledAt": self.get_time_ms() if fill_time is None else fill_time,
            }
        )
        self._record_order(order)
        self.open_orders.pop(str(order["orderId"]), None)
        receipt = _clone_order(order)
        self.emit("order:filled", receipt)
        unrealized, _ = self._summary()
        self.emit(
            "equity:update",
            {
                "cash": self.cash,
                "realizedPnl": realized,
                "feeTotal": fee_total,
                "equity": self.cash + unrealized,
            },
        )
        return receipt

    @staticmethod
    def get_time_ms() -> int:
        import time

        return time.time_ns() // 1_000_000

    async def submit_order(self, order: Mapping[str, object]) -> dict[str, Any]:
        if not isinstance(order, Mapping):
            raise ValidationError("order must be a mapping")
        order_type = _order_type(order.get("type", "market"))
        normalized: dict[str, Any] = {
            "orderId": self._next_order_id(),
            "clientOrderId": order.get("clientOrderId", order.get("client_order_id")),
            "status": "new",
            "filledQty": 0.0,
            "avgFillPrice": None,
            "filledAt": None,
            "symbol": _symbol(order.get("symbol")),
            "side": _side(order.get("side")),
            "type": order_type,
            "qty": round_step(_positive(order.get("qty"), "qty"), self.qty_step),
            "limitPrice": _optional_finite(
                order.get("limitPrice", order.get("limit_price")), "limitPrice"
            ),
            "stopPrice": _optional_finite(
                order.get("stopPrice", order.get("stop_price")), "stopPrice"
            ),
            "timeInForce": str(order.get("timeInForce", order.get("time_in_force", "day"))),
            "rejectReason": None,
        }
        if normalized["qty"] <= 0:
            return self._reject_order(normalized, "invalid quantity")
        if order_type in {"limit", "stop_limit"} and normalized["limitPrice"] is None:
            raise ValidationError("limitPrice must be finite for limit orders")
        if order_type in {"stop", "stop_limit"} and normalized["stopPrice"] is None:
            raise ValidationError("stopPrice must be finite for stop orders")
        self._record_order(normalized)
        self.emit("order:submitted", _clone_order(normalized))
        if order_type == "market":
            price = self.last_prices.get(str(normalized["symbol"]))
            if price is None:
                price = normalized["limitPrice"] or normalized["stopPrice"]
            if price is None:
                return self._reject_order(normalized, "no price available for market order")
            return self._fill_order(normalized, float(price))
        self.open_orders[str(normalized["orderId"])] = normalized
        return _clone_order(normalized)

    def cancel_order_nowait(self, order_id: str) -> None:
        order = self.open_orders.get(str(order_id))
        if order is None:
            return
        order["status"] = "canceled"
        self._record_order(order)
        self.open_orders.pop(str(order_id), None)
        self.emit("order:canceled", _clone_order(order))

    async def cancel_order(self, order_id: str) -> None:
        self.cancel_order_nowait(order_id)

    async def modify_order(self, order_id: str, changes: Mapping[str, object]) -> dict[str, Any]:
        order = self.open_orders.get(str(order_id))
        if order is None:
            raise ValidationError(f'paper order "{order_id}" not found or already closed')
        if not isinstance(changes, Mapping):
            raise ValidationError("changes must be a mapping")
        if "qty" in changes:
            order["qty"] = round_step(_positive(changes["qty"], "qty"), self.qty_step)
        for camel, snake in (("limitPrice", "limit_price"), ("stopPrice", "stop_price")):
            if camel in changes or snake in changes:
                order[camel] = _finite(changes.get(camel, changes.get(snake)), camel)
        self._record_order(order)
        receipt = _clone_order(order)
        self.emit("order:modified", receipt)
        return receipt

    async def get_open_orders(self) -> list[dict[str, Any]]:
        return [_clone_order(order) for order in self.open_orders.values()]

    async def get_order_status(self, order_id: str) -> dict[str, Any]:
        order = self.open_orders.get(str(order_id)) or self.order_history.get(str(order_id))
        if order is None:
            raise ValidationError(f'paper order "{order_id}" not found')
        return _clone_order(order)

    async def simulate_bar(self, symbol: str, interval: str, bar: Mapping[str, object]) -> None:
        if not isinstance(bar, Mapping):
            raise ValidationError("bar must be a mapping")
        normalized = {
            "time": _finite(bar.get("time"), "bar.time"),
            "open": _finite(bar.get("open"), "bar.open"),
            "high": _finite(bar.get("high"), "bar.high"),
            "low": _finite(bar.get("low"), "bar.low"),
            "close": _finite(bar.get("close"), "bar.close"),
            "volume": _finite(bar.get("volume", 0), "bar.volume"),
        }
        if normalized["high"] < max(normalized["open"], normalized["close"], normalized["low"]):
            raise ValidationError("bar OHLC range is invalid")
        if normalized["low"] > min(normalized["open"], normalized["close"], normalized["high"]):
            raise ValidationError("bar OHLC range is invalid")
        clean_symbol = _symbol(symbol)
        self.last_prices[clean_symbol] = normalized["close"]
        await self._emit_to(
            self._bar_subscribers, self._stream_key(clean_symbol, interval), normalized
        )
        await self._emit_to(
            self._trade_subscribers,
            clean_symbol,
            {
                "time": normalized["time"],
                "price": normalized["close"],
                "size": normalized["volume"],
            },
        )
        for order in tuple(self.open_orders.values()):
            if order["symbol"] != clean_symbol or str(order["orderId"]) not in self.open_orders:
                continue
            if order["type"] == "limit":
                if touched_limit(
                    "long" if order["side"] == "buy" else "short",
                    float(order["limitPrice"]),
                    normalized,
                ):
                    self._fill_order(order, float(order["limitPrice"]), "limit", normalized["time"])
            elif order["type"] == "stop":
                touched = (
                    normalized["high"] >= float(order["stopPrice"])
                    if order["side"] == "buy"
                    else normalized["low"] <= float(order["stopPrice"])
                )
                if touched:
                    self._fill_order(order, float(order["stopPrice"]), "stop", normalized["time"])
            else:
                touched_stop = (
                    normalized["high"] >= float(order["stopPrice"])
                    if order["side"] == "buy"
                    else normalized["low"] <= float(order["stopPrice"])
                )
                order["_triggered"] = bool(order.get("_triggered")) or touched_stop
                if order["_triggered"] and touched_limit(
                    "long" if order["side"] == "buy" else "short",
                    float(order["limitPrice"]),
                    normalized,
                ):
                    self._fill_order(order, float(order["limitPrice"]), "limit", normalized["time"])


def create_paper_engine(
    *,
    equity: object = 10_000,
    currency: str = "USD",
    slippage_bps: object = 0,
    fee_bps: object = 0,
    costs: Mapping[str, object] | None = None,
    qty_step: object = 0.001,
) -> PaperEngine:
    return PaperEngine(
        equity=equity,
        currency=currency,
        slippage_bps=slippage_bps,
        fee_bps=fee_bps,
        costs=costs,
        qty_step=qty_step,
    )
