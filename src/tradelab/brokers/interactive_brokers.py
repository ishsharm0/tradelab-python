"""Interactive Brokers adapter with a lazy ``ib-insync`` dependency."""

from __future__ import annotations

import importlib
import inspect
import math
import re
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from types import ModuleType
from typing import Any, Protocol, cast

from tradelab.data import normalize_candles
from tradelab.errors import BrokerError, ValidationError

from .base import (
    Account,
    BrokerAdapter,
    Clock,
    OrderReceipt,
    Position,
    number,
    option,
    system_clock_ms,
)

Factory = Callable[[], object]
ModuleLoader = Callable[[str], object]
ContractFactory = Callable[[str, str, str], object]
OrderFactory = Callable[..., object]


class _IBModule(Protocol):
    IB: Factory
    Stock: ContractFactory


async def _resolve(value: object) -> object:
    return await value if inspect.isawaitable(value) else value


def _attribute(value: object, name: str, default: object = None) -> object:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _time_ms(value: object) -> int | None:
    if isinstance(value, datetime):
        timestamp = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
        return round(timestamp.astimezone(UTC).timestamp() * 1_000)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return round(float(value) * (1_000 if abs(float(value)) < 1e12 else 1))
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return round(parsed.astimezone(UTC).timestamp() * 1_000)
    return None


class InteractiveBrokersBroker(BrokerAdapter):
    """IB Gateway/TWS adapter whose optional SDK is imported only on connect."""

    broker_name = "interactive_brokers"

    def __init__(
        self,
        *,
        ib_factory: Factory | None = None,
        module_loader: ModuleLoader = importlib.import_module,
        contract_factory: ContractFactory | None = None,
        order_factory: OrderFactory | None = None,
        clock: Clock = system_clock_ms,
    ) -> None:
        super().__init__(clock=clock)
        self._ib_factory = ib_factory
        self._module_loader = module_loader
        self._contract_factory = contract_factory
        self._order_factory = order_factory
        self._ib: object | None = None
        self._module: ModuleType | object | None = None
        self._trades: dict[str, object] = {}

    def supports_paper_native(self) -> bool:
        return True

    async def connect(self, config: Mapping[str, object] | None = None) -> None:
        values = config or {}
        paper = bool(option(values, "paper", default=False))
        host = str(option(values, "host", default="127.0.0.1"))
        port_value = option(values, "port", default=7497 if paper else 7496)
        client_value = option(values, "client_id", "clientId", default=1)
        if isinstance(port_value, bool) or not isinstance(port_value, int) or port_value <= 0:
            raise ValidationError("Interactive Brokers port must be a positive integer")
        if isinstance(client_value, bool) or not isinstance(client_value, int) or client_value < 0:
            raise ValidationError("Interactive Brokers client_id must be a non-negative integer")
        if self._ib_factory is None:
            try:
                self._module = self._module_loader("ib_insync")
                factory = cast(_IBModule, self._module).IB
            except (ImportError, AttributeError) as error:
                raise BrokerError(
                    'Install the optional "tradelab[ib]" extra to provide ib-insync: '
                    'pip install "tradelab[ib]"'
                ) from error
        else:
            factory = self._ib_factory
        self._ib = factory()
        connector = getattr(self._ib, "connectAsync", None)
        if not callable(connector):
            connector = getattr(self._ib, "connect", None)
        if not callable(connector):
            self._ib = None
            raise BrokerError("Interactive Brokers client does not provide connect")
        try:
            await _resolve(
                connector(
                    host,
                    port_value,
                    clientId=client_value,
                    timeout=number(option(values, "timeout", default=4.0), 4.0),
                    readonly=bool(option(values, "readonly", default=False)),
                )
            )
        except Exception as error:
            self._ib = None
            raise BrokerError(f"Interactive Brokers connection failed: {error}") from error
        self._connected = True

    async def disconnect(self) -> None:
        if self._ib is not None:
            disconnect = getattr(self._ib, "disconnect", None)
            if callable(disconnect):
                await _resolve(disconnect())
        self._ib = None
        await super().disconnect()

    def _require_ib(self) -> object:
        if not self._connected or self._ib is None:
            raise BrokerError("InteractiveBrokersBroker is not connected")
        return self._ib

    async def get_server_time(self) -> int:
        return self._clock()

    async def get_account(self) -> Account:
        ib = self._require_ib()
        summary_method = getattr(ib, "accountSummaryAsync", None)
        resolved = await _resolve(summary_method()) if callable(summary_method) else []
        rows = resolved if isinstance(resolved, list) else []
        tags = {str(_attribute(row, "tag", "")): number(_attribute(row, "value")) for row in rows}
        currency = next(
            (str(_attribute(row, "currency")) for row in rows if _attribute(row, "currency")),
            "USD",
        )
        return {
            "equity": tags.get("NetLiquidation", 0.0),
            "buyingPower": tags.get("BuyingPower", 0.0),
            "cash": tags.get("TotalCashValue", 0.0),
            "currency": currency,
            "marginUsed": tags.get("InitMarginReq", 0.0),
        }

    async def get_positions(self) -> list[Position]:
        ib = self._require_ib()
        method = getattr(ib, "positions", None)
        values = await _resolve(method()) if callable(method) else []
        positions: list[Position] = []
        for row in values if isinstance(values, list) else []:
            qty = number(_attribute(row, "position"))
            if qty == 0:
                continue
            entry = number(_attribute(row, "avgCost"))
            contract = _attribute(row, "contract")
            positions.append(
                {
                    "symbol": str(_attribute(contract, "symbol", "")),
                    "side": "long" if qty >= 0 else "short",
                    "qty": abs(qty),
                    "avgEntry": entry,
                    "marketValue": abs(qty * entry),
                    "unrealizedPnl": 0.0,
                }
            )
        return positions

    async def submit_order(self, order: Mapping[str, object]) -> OrderReceipt:
        ib = self._require_ib()
        symbol = str(order.get("symbol") or "").strip()
        if not symbol:
            raise ValidationError("Interactive Brokers order requires a symbol")
        contract = await self._qualified_contract(symbol)
        ib_order = self._create_order(order)
        place = getattr(ib, "placeOrder", None)
        if not callable(place):
            raise BrokerError("Interactive Brokers client does not provide placeOrder")
        trade = await _resolve(place(contract, ib_order))
        receipt = self._trade_receipt(trade, fallback_contract=contract, fallback_order=ib_order)
        if not receipt["orderId"]:
            raise BrokerError("Interactive Brokers did not assign an order id")
        self._trades[receipt["orderId"]] = trade
        await self._emit("order:submitted", dict(receipt))
        return receipt

    async def cancel_order(self, order_id: object) -> None:
        reference = str(order_id)
        trade = self._trades.get(reference)
        if trade is None:
            return
        ib = self._require_ib()
        cancel = getattr(ib, "cancelOrder", None)
        if not callable(cancel):
            raise BrokerError("Interactive Brokers client does not provide cancelOrder")
        order = _attribute(trade, "order")
        result = await _resolve(cancel(order))
        if result is not None:
            trade = result
            self._trades[reference] = trade
        receipt = self._trade_receipt(trade)
        await self._emit("order:canceled", dict(receipt))

    async def modify_order(self, order_id: object, changes: Mapping[str, object]) -> OrderReceipt:
        reference = str(order_id)
        trade = self._trades.get(reference)
        if trade is None:
            raise BrokerError(f'IB order "{order_id}" not found')
        ib = self._require_ib()
        contract = _attribute(trade, "contract")
        order = _attribute(trade, "order")
        mutable_order = cast(Any, order)
        if changes.get("qty") is not None:
            mutable_order.totalQuantity = number(changes["qty"])
        limit = option(changes, "limit_price", "limitPrice")
        stop = option(changes, "stop_price", "stopPrice")
        if limit is not None:
            mutable_order.lmtPrice = number(limit)
        if stop is not None:
            mutable_order.auxPrice = number(stop)
            if str(_attribute(order, "orderType", "")).upper() == "LMT":
                mutable_order.orderType = "STP LMT"
        place = getattr(ib, "placeOrder", None)
        if not callable(place):
            raise BrokerError("Interactive Brokers client does not provide placeOrder")
        updated = await _resolve(place(contract, order))
        if updated is not None:
            trade = updated
            self._trades[reference] = trade
        receipt = self._trade_receipt(trade)
        await self._emit("order:modified", dict(receipt))
        return receipt

    async def get_open_orders(self) -> list[OrderReceipt]:
        ib = self._require_ib()
        method = getattr(ib, "openTrades", None)
        values = await _resolve(method()) if callable(method) else list(self._trades.values())
        trades = values if isinstance(values, list) else []
        receipts = [self._trade_receipt(trade) for trade in trades]
        for trade, receipt in zip(trades, receipts, strict=True):
            if receipt["orderId"]:
                self._trades[receipt["orderId"]] = trade
        return [receipt for receipt in receipts if receipt["status"] in {"new", "partially_filled"}]

    async def get_order_status(self, order_id: object) -> OrderReceipt:
        trade = self._trades.get(str(order_id))
        if trade is None:
            raise BrokerError(f'IB order "{order_id}" not found')
        return self._trade_receipt(trade)

    async def _qualified_contract(self, symbol: str) -> object:
        ib = self._require_ib()
        contract = self._contract(symbol)
        qualify = getattr(ib, "qualifyContractsAsync", None)
        if not callable(qualify):
            qualify = getattr(ib, "qualifyContracts", None)
        if not callable(qualify):
            raise BrokerError("Interactive Brokers client does not provide qualifyContracts")
        values = await _resolve(qualify(contract))
        if not isinstance(values, (list, tuple)) or not values:
            raise BrokerError(f'Interactive Brokers could not qualify contract "{symbol}"')
        return values[0]

    def _create_order(self, order: Mapping[str, object]) -> object:
        factory = self._order_factory
        if factory is None and self._module is not None:
            candidate = getattr(self._module, "Order", None)
            factory = candidate if callable(candidate) else None
        if factory is None:
            raise BrokerError("Interactive Brokers order factory is unavailable")
        kind = str(order.get("type") or "market").lower()
        order_type = {
            "market": "MKT",
            "limit": "LMT",
            "stop": "STP",
            "stop_limit": "STP LMT",
        }.get(kind)
        if order_type is None:
            raise ValidationError(f'Unsupported Interactive Brokers order type "{kind}"')
        values: dict[str, object] = {
            "action": str(order.get("side") or "").upper(),
            "totalQuantity": number(order.get("qty")),
            "orderType": order_type,
            "tif": str(option(order, "time_in_force", "timeInForce", default="GTC")).upper(),
            "transmit": True,
        }
        client_id = option(order, "client_order_id", "clientOrderId")
        if client_id is not None:
            values["orderRef"] = str(client_id)
        limit = option(order, "limit_price", "limitPrice")
        stop = option(order, "stop_price", "stopPrice")
        if limit is not None:
            values["lmtPrice"] = number(limit)
        if stop is not None:
            values["auxPrice"] = number(stop)
        return factory(**values)

    @staticmethod
    def _trade_receipt(
        trade: object, *, fallback_contract: object = None, fallback_order: object = None
    ) -> OrderReceipt:
        contract = _attribute(trade, "contract", fallback_contract)
        order = _attribute(trade, "order", fallback_order)
        status = _attribute(trade, "orderStatus", {})
        normalized_status = str(_attribute(status, "status", "")).upper().replace("_", "")
        state = {
            "FILLED": "filled",
            "CANCELLED": "canceled",
            "APICANCELLED": "canceled",
            "INACTIVE": "rejected",
            "PARTIALLYFILLED": "partially_filled",
        }.get(normalized_status, "new")
        filled = number(_attribute(status, "filled"))
        total = number(_attribute(order, "totalQuantity"))
        if state == "new" and 0 < filled < total:
            state = "partially_filled"
        fills = _attribute(trade, "fills", [])
        fill_rows = fills if isinstance(fills, list) else []
        execution = _attribute(fill_rows[-1], "execution") if fill_rows else None
        filled_at = _time_ms(_attribute(execution, "time"))
        average = number(_attribute(status, "avgFillPrice"))
        receipt: OrderReceipt = {
            "orderId": str(_attribute(order, "orderId", _attribute(order, "permId", "")) or ""),
            "clientOrderId": str(_attribute(order, "orderRef"))
            if _attribute(order, "orderRef") not in {None, ""}
            else None,
            "status": state,
            "filledQty": filled,
            "avgFillPrice": average if average > 0 else None,
            "filledAt": filled_at,
            "symbol": str(_attribute(contract, "symbol", "")),
            "side": str(_attribute(order, "action", "")).lower(),
            "type": {
                "MKT": "market",
                "LMT": "limit",
                "STP": "stop",
                "STP LMT": "stop_limit",
            }.get(str(_attribute(order, "orderType", "")).upper(), "unknown"),
            "qty": total,
        }
        limit = number(_attribute(order, "lmtPrice"))
        stop = number(_attribute(order, "auxPrice"))
        if limit > 0:
            receipt["limitPrice"] = limit
        if stop > 0:
            receipt["stopPrice"] = stop
        return receipt

    def _contract(self, symbol: str) -> object:
        factory = self._contract_factory
        if factory is None and self._module is not None:
            candidate = getattr(self._module, "Stock", None)
            factory = candidate if callable(candidate) else None
        if factory is None:
            raise BrokerError("Interactive Brokers contract factory is unavailable")
        return factory(symbol, "SMART", "USD")

    @staticmethod
    def _bar_size(interval: object) -> tuple[str, int]:
        match = re.fullmatch(r"(\d+)(m|h|d)", str(interval or "1m").lower())
        if match is None:
            return "1 min", 60
        amount = int(match.group(1))
        unit, seconds = {
            "m": ("min", 60),
            "h": ("hour", 3_600),
            "d": ("day", 86_400),
        }[match.group(2)]
        return f"{amount} {unit}", amount * seconds

    async def get_historical_bars(
        self, symbol: str, interval: str, limit: int = 200
    ) -> list[dict[str, int | float]]:
        if limit <= 0:
            return []
        ib = self._require_ib()
        request = getattr(ib, "reqHistoricalDataAsync", None)
        if not callable(request):
            return []
        bar_size, seconds = self._bar_size(interval)
        duration_days = max(1, math.ceil(max(1, limit) * seconds / 86_400))
        values = await _resolve(
            request(
                self._contract(symbol),
                endDateTime="",
                durationStr=f"{duration_days} D",
                barSizeSetting=bar_size,
                whatToShow="TRADES",
                useRTH=True,
                formatDate=2,
                keepUpToDate=False,
            )
        )
        bars: list[dict[str, object]] = []
        for row in values if isinstance(values, list) else []:
            timestamp = _time_ms(_attribute(row, "date"))
            if timestamp is None:
                continue
            bars.append(
                {
                    "time": timestamp,
                    "open": _attribute(row, "open"),
                    "high": _attribute(row, "high"),
                    "low": _attribute(row, "low"),
                    "close": _attribute(row, "close"),
                    "volume": _attribute(row, "volume", 0),
                }
            )
        return normalize_candles(bars)[-max(0, int(limit)) :]


def create_interactive_brokers_broker(
    *,
    ib_factory: Factory | None = None,
    module_loader: ModuleLoader = importlib.import_module,
    contract_factory: ContractFactory | None = None,
    order_factory: OrderFactory | None = None,
    clock: Clock = system_clock_ms,
) -> InteractiveBrokersBroker:
    """Create an Interactive Brokers adapter from constructor options."""
    return InteractiveBrokersBroker(
        ib_factory=ib_factory,
        module_loader=module_loader,
        contract_factory=contract_factory,
        order_factory=order_factory,
        clock=clock,
    )


__all__ = ["InteractiveBrokersBroker", "create_interactive_brokers_broker"]
