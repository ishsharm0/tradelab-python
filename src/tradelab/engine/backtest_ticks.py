"""Deterministic event-driven tick backtesting.

Signals create pending intents on the current tick. Market and limit intents can
only fill on a later tick, and every position still open after the final tick is
liquidated at that tick with reason ``EOT``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
from typing import Any

from tradelab.errors import ValidationError
from tradelab.metrics import build_metrics
from tradelab.models import BacktestResult
from tradelab.utils.position_sizing import calculate_position_size
from tradelab.utils.random import _javascript_seed_string, make_rng

from .execution import apply_fill, day_key_utc, oco_exit_check, round_step
from .financing import financing_cost
from .signal import as_number, call_signal_with_context, normalize_signal

_MISSING = object()


def _camel(value: str) -> str:
    parts = value.split("_")
    return parts[0] + "".join(part[:1].upper() + part[1:] for part in parts[1:])


def _js_key(value: object) -> object:
    if isinstance(value, Mapping):
        special = {
            "total_pnl": "totalPnL",
            "avg_pnl": "avgPnL",
            "profit_factor_leg": "profitFactor_leg",
            "profit_factor_pos": "profitFactor_pos",
            "win_rate_leg": "winRate_leg",
            "win_rate_pos": "winRate_pos",
        }
        return {
            special.get(str(key), _camel(str(key))): _js_key(item) for key, item in value.items()
        }
    if isinstance(value, list):
        return [_js_key(item) for item in value]
    return value


_SPECIAL_OPTION_KEYS = {"final_tp_r": "finalTP_R"}


def _option(options: Mapping[str, Any], snake: str, default: Any) -> Any:
    for key in (snake, _SPECIAL_OPTION_KEYS.get(snake), _camel(snake)):
        if key is not None and key in options and options[key] is not None:
            return options[key]
    return default


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValidationError(f"{name} must be finite", context={name: value})
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as error:
        raise ValidationError(f"{name} must be finite", context={name: value}) from error
    if not math.isfinite(result):
        raise ValidationError(f"{name} must be finite", context={name: value})
    return result


def _bounded(
    value: object,
    name: str,
    *,
    minimum: float = 0,
    maximum: float | None = None,
    exclusive_minimum: bool = False,
) -> float:
    result = _finite(value, name)
    too_small = result <= minimum if exclusive_minimum else result < minimum
    if too_small or (maximum is not None and result > maximum):
        suffix = f" and <= {maximum}" if maximum is not None else ""
        operator = ">" if exclusive_minimum else ">="
        raise ValidationError(f"{name} must be {operator} {minimum}{suffix}", context={name: value})
    return result


def _integer(value: object, name: str) -> int:
    result = _finite(value, name)
    if result < 0 or not result.is_integer():
        raise ValidationError(f"{name} must be an integer >= 0", context={name: value})
    return int(result)


def _js_number(value: object) -> float | None:
    """Implement the primitive subset of JavaScript ``Number`` used by tick rows."""
    if value is _MISSING:
        return None
    if value is None:
        return 0.0
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, str) and not value.strip():
        return 0.0
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _nullish(mapping: Mapping[str, object], *keys: str) -> object:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return _MISSING


def _compact_number(value: float) -> int | float:
    return int(value) if value.is_integer() else value


def _normalize_tick(value: object) -> dict[str, object] | None:
    if not isinstance(value, Mapping):
        return None
    time = _js_number(value.get("time", _MISSING))
    bid = _js_number(value.get("bid", _MISSING))
    ask = _js_number(value.get("ask", _MISSING))
    last = _js_number(_nullish(value, "price", "last", "close"))
    mid = (
        (bid + ask) / 2
        if bid is not None and ask is not None
        else (last if last is not None else bid if bid is not None else ask)
    )
    if time is None or mid is None:
        return None
    prices = [
        price
        for price in (
            _js_number(value.get("low", _MISSING)),
            _js_number(value.get("high", _MISSING)),
            bid,
            ask,
            last,
            mid,
        )
        if price is not None
    ]
    volume = _js_number(_nullish(value, "size", "volume"))
    tick = dict(value)
    tick.update(
        {
            "time": _compact_number(time),
            "open": _compact_number(mid),
            "high": _compact_number(max(prices) if prices else mid),
            "low": _compact_number(min(prices) if prices else mid),
            "close": _compact_number(mid),
            "volume": _compact_number(volume) if volume is not None else None,
        }
    )
    return tick


def _iso(time_ms: float) -> str:
    try:
        return (
            datetime.fromtimestamp(time_ms / 1_000, UTC)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z")
        )
    except (OSError, OverflowError, ValueError) as error:
        raise ValidationError(
            "tick time must be valid Unix milliseconds", context={"time": time_ms}
        ) from error


def _equity_point(time: int | float, equity: float) -> dict[str, int | float]:
    return {"time": time, "timestamp": time, "equity": equity}


def _tick_time(tick: Mapping[str, object]) -> int | float:
    value = tick.get("time")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    raise ValidationError("normalized tick time must be numeric")


def _tick_number(tick: Mapping[str, object], key: str) -> float:
    return _finite(tick.get(key), f"tick.{key}")


def _deterministic_fill(probability: float, seed_parts: Sequence[object]) -> bool:
    if probability >= 1:
        return True
    if probability <= 0:
        return False
    seed = "|".join(_javascript_seed_string(part) for part in seed_parts)
    return make_rng(seed)() <= probability


def _assert_json_safe(value: object) -> None:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValidationError("tick backtest result must be JSON-safe") from error


def backtest_ticks(
    options: Mapping[str, object] | None = None, /, **kwargs: object
) -> BacktestResult:
    """Run the immutable JavaScript tick chronology with deterministic fills."""
    raw: dict[str, Any] = dict(options or {})
    raw.update(kwargs)
    tick_values = _option(raw, "ticks", [])
    if isinstance(tick_values, (str, bytes, bytearray)) or not isinstance(tick_values, Sequence):
        raise ValidationError("ticks must be a sequence")
    if not tick_values:
        raise ValidationError("backtest_ticks requires non-empty ticks")
    signal_value = _option(raw, "signal", None)
    if not callable(signal_value):
        raise ValidationError("backtest_ticks requires a signal callable")
    signal: Callable[[dict[str, object]], object] = signal_value

    ticks = [tick for value in tick_values if (tick := _normalize_tick(value)) is not None]
    if not ticks:
        raise ValidationError(
            f"backtest_ticks could not normalize any ticks from {len(tick_values)} input rows"
        )

    symbol = str(_option(raw, "symbol", "UNKNOWN"))
    interval_value = _option(raw, "interval", None)
    if interval_value is not None and not isinstance(interval_value, str):
        raise ValidationError("interval must be a string or None")
    interval: str | None = interval_value
    range_value = _option(raw, "range", None)
    equity_start = _finite(_option(raw, "equity", 10_000), "equity")
    risk_pct = _finite(_option(raw, "risk_pct", 1), "risk_pct")
    slippage_bps = _finite(_option(raw, "slippage_bps", 1), "slippage_bps")
    fee_bps = _finite(_option(raw, "fee_bps", 0), "fee_bps")
    final_tp_r = _bounded(_option(raw, "final_tp_r", 3), "final_tp_r")
    max_daily_loss_pct = _bounded(_option(raw, "max_daily_loss_pct", 0), "max_daily_loss_pct")
    daily_max_trades = _integer(_option(raw, "daily_max_trades", 0), "daily_max_trades")
    qty_step = _bounded(_option(raw, "qty_step", 0.001), "qty_step", exclusive_minimum=True)
    min_qty = _bounded(_option(raw, "min_qty", 0.001), "min_qty")
    max_leverage = _bounded(_option(raw, "max_leverage", 2), "max_leverage")
    collect_eq_series = bool(_option(raw, "collect_eq_series", True))
    collect_replay = bool(_option(raw, "collect_replay", True))
    probability = _bounded(
        _option(raw, "queue_fill_probability", 1),
        "queue_fill_probability",
        maximum=1,
    )
    seed = _option(raw, "seed", "tradelab-ticks")
    _javascript_seed_string(seed)
    costs_value = _option(raw, "costs", None)
    if costs_value is not None and not isinstance(costs_value, Mapping):
        raise ValidationError("costs must be a mapping")
    costs: Mapping[str, object] | None = (
        deepcopy(dict(costs_value)) if isinstance(costs_value, Mapping) else None
    )
    oco_value = _option(raw, "oco", {})
    if not isinstance(oco_value, Mapping):
        raise ValidationError("oco must be a mapping")
    tie_break = oco_value.get("tieBreak", oco_value.get("tie_break", "pessimistic"))
    if tie_break not in {"pessimistic", "optimistic"}:
        raise ValidationError("oco.tieBreak must be pessimistic or optimistic")

    trades: list[dict[str, Any]] = []
    eq_series: list[dict[str, int | float]] = (
        [_equity_point(_tick_time(ticks[0]), equity_start)] if collect_eq_series else []
    )
    replay_frames: list[dict[str, object]] = []
    replay_events: list[dict[str, object]] = []
    history: list[dict[str, object]] = []
    open_position: dict[str, Any] | None = None
    pending: dict[str, Any] | None = None
    current_equity = equity_start
    current_day: str | None = None
    day_start_equity = equity_start
    day_pnl = 0.0
    day_trades = 0
    trade_id = 0

    def marked_equity(tick: Mapping[str, object]) -> float:
        if not open_position:
            return current_equity
        result = current_equity + (
            _tick_number(tick, "close") - float(open_position["entryFill"])
        ) * (1 if open_position["side"] == "long" else -1) * float(open_position["size"])
        if not math.isfinite(result):
            raise ValidationError("marked equity must remain finite")
        return result

    def record_frame(tick: Mapping[str, object]) -> None:
        equity_now = marked_equity(tick)
        if collect_eq_series:
            eq_series.append(_equity_point(_tick_time(tick), equity_now))
        if collect_replay:
            replay_frames.append(
                {
                    "t": _iso(_tick_number(tick, "time")),
                    "price": tick["close"],
                    "equity": equity_now,
                    "posSide": open_position["side"] if open_position else None,
                    "posSize": open_position["size"] if open_position else 0,
                }
            )

    def close_position(
        tick: Mapping[str, object], reason: str, raw_price: float, fill_kind: str
    ) -> None:
        nonlocal current_equity, day_pnl, open_position
        if not open_position:
            return
        position = open_position
        exit_side = "short" if position["side"] == "long" else "long"
        fill = apply_fill(
            raw_price,
            exit_side,
            slippage_bps=slippage_bps,
            fee_bps=fee_bps,
            kind=fill_kind,
            qty=float(position["size"]),
            costs=costs,
        )
        gross_pnl = (
            (fill["price"] - float(position["entryFill"]))
            * (1 if position["side"] == "long" else -1)
            * float(position["size"])
        )
        financing = financing_cost(
            side=str(position["side"]),
            notional=float(position["entryFill"]) * float(position["size"]),
            from_ms=float(position["openTime"]),
            to_ms=_tick_number(tick, "time"),
            costs=costs,
        )
        pnl = gross_pnl - float(position.get("entryFeeTotal", 0)) - fill["fee_total"] - financing
        if not math.isfinite(pnl) or not math.isfinite(current_equity + pnl):
            raise ValidationError("tick trade PnL must remain finite")
        current_equity += pnl
        day_pnl += pnl
        trade = dict(position)
        trade["exit"] = {
            "price": fill["price"],
            "time": tick["time"],
            "reason": reason,
            "pnl": pnl,
            "financing": financing,
        }
        trades.append(trade)
        if collect_replay:
            replay_events.append(
                {
                    "t": _iso(_tick_number(tick, "time")),
                    "price": fill["price"],
                    "type": "tp" if reason == "TP" else "sl" if reason == "SL" else "exit",
                    "side": position["side"],
                    "size": position["size"],
                    "tradeId": position["id"],
                    "reason": reason,
                    "pnl": pnl,
                }
            )
        open_position = None

    def open_pending(tick: Mapping[str, object], *, kind: str) -> None:
        nonlocal open_position, pending, day_trades, trade_id
        if pending is None:
            return
        entry = _tick_number(tick, "close") if kind == "market" else float(pending["entry"])
        raw_size = pending["fixedQty"]
        if raw_size is None:
            raw_size = calculate_position_size(
                equity=current_equity,
                entry=entry,
                stop=float(pending["stop"]),
                risk_fraction=float(pending["riskFrac"]),
                qty_step=qty_step,
                min_qty=min_qty,
                max_leverage=max_leverage,
            )
        size = round_step(float(raw_size), qty_step)
        if size >= min_qty:
            fill = apply_fill(
                entry,
                str(pending["side"]),
                slippage_bps=slippage_bps,
                fee_bps=fee_bps,
                kind=kind,
                qty=size,
                costs=costs,
            )
            trade_id += 1
            open_position = {
                "symbol": symbol,
                "id": trade_id,
                "side": pending["side"],
                "entry": entry,
                "stop": pending["stop"],
                "takeProfit": pending["takeProfit"],
                "size": size,
                "openTime": tick["time"],
                "entryFill": fill["price"],
                "entryFeeTotal": fill["fee_total"],
                "_initRisk": abs(entry - float(pending["stop"])),
            }
            day_trades += 1
            if collect_replay:
                replay_events.append(
                    {
                        "t": _iso(_tick_number(tick, "time")),
                        "price": fill["price"],
                        "type": "entry",
                        "side": open_position["side"],
                        "size": size,
                        "tradeId": trade_id,
                    }
                )
        pending = None

    for index, tick in enumerate(ticks):
        history.append(tick)
        next_day = day_key_utc(_tick_number(tick, "time"))
        if current_day is None or next_day != current_day:
            current_day = next_day
            day_start_equity = current_equity
            day_pnl = 0.0
            day_trades = 0

        if open_position:
            hit = oco_exit_check(
                side=str(open_position["side"]),
                stop=float(open_position["stop"]),
                tp=float(open_position["takeProfit"]),
                bar=tick,
                mode="intrabar",
                tie_break=str(tie_break),
            )
            if hit["hit"]:
                close_position(
                    tick,
                    str(hit["hit"]),
                    _finite(hit["px"], "oco exit price"),
                    "limit" if hit["hit"] == "TP" else "stop",
                )

        if not open_position and pending and index > int(pending["createdAtIndex"]):
            if pending["orderType"] == "market":
                open_pending(tick, kind="market")
            else:
                touched = (
                    _tick_number(tick, "low") <= float(pending["entry"])
                    if pending["side"] == "long"
                    else _tick_number(tick, "high") >= float(pending["entry"])
                )
                if touched and _deterministic_fill(
                    probability,
                    (
                        seed,
                        symbol,
                        tick["time"],
                        pending["entry"],
                        pending["stop"],
                        pending["side"],
                    ),
                ):
                    open_pending(tick, kind="limit")

        max_loss_dollars = abs(max_daily_loss_pct) / 100 * day_start_equity
        daily_loss_hit = max_daily_loss_pct > 0 and day_pnl <= -max_loss_dollars
        daily_trade_cap_hit = daily_max_trades > 0 and day_trades >= daily_max_trades
        if not open_position and not pending and not daily_loss_hit and not daily_trade_cap_hit:
            context: dict[str, object] = {
                "candles": history,
                "index": index,
                "bar": tick,
                "equity": marked_equity(tick),
                "openPosition": open_position,
                "pendingOrder": pending,
            }
            raw_signal = call_signal_with_context(signal, context, index, tick, symbol)
            next_signal = normalize_signal(raw_signal, tick, final_tp_r)
            if next_signal:
                explicit_entry = isinstance(raw_signal, Mapping) and any(
                    key in raw_signal for key in ("entry", "limit", "price")
                )
                risk_fraction = as_number(next_signal.get("riskFraction"))
                if risk_fraction is None:
                    signal_risk_pct = as_number(next_signal.get("riskPct"))
                    risk_fraction = (
                        signal_risk_pct / 100 if signal_risk_pct is not None else risk_pct / 100
                    )
                pending = {
                    "side": next_signal["side"],
                    "entry": next_signal["entry"],
                    "stop": next_signal["stop"],
                    "takeProfit": next_signal["takeProfit"],
                    "fixedQty": next_signal.get("qty"),
                    "riskFrac": risk_fraction,
                    "orderType": "limit" if explicit_entry else "market",
                    "createdAtIndex": index,
                }
        record_frame(tick)

    if open_position:
        final_tick = ticks[-1]
        close_position(final_tick, "EOT", _tick_number(final_tick, "close"), "market")
        record_frame(final_tick)

    est_bar_ms = (
        max(1.0, _tick_number(ticks[1], "time") - _tick_number(ticks[0], "time"))
        if len(ticks) > 1
        else 1.0
    )
    metrics = build_metrics(
        closed=trades,
        equity_start=equity_start,
        equity_final=current_equity,
        candles=ticks,
        est_bar_ms=est_bar_ms,
        eq_series=eq_series,
        interval=interval,
    )
    result = {
        "symbol": symbol,
        "interval": interval,
        "range": range_value,
        "trades": trades,
        "positions": trades,
        "openPositions": [],
        "metrics": _js_key(metrics),
        "eqSeries": eq_series,
        "replay": {"frames": replay_frames, "events": replay_events},
    }
    _assert_json_safe(result)
    return BacktestResult(result)
