"""One deterministic bar state machine and its synchronous backtest wrapper.

The runner deliberately leaves the final position open: a caller can continue it
incrementally, and synchronous :func:`backtest` is only a loop over ``step``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, overload

from tradelab.errors import ValidationError
from tradelab.metrics import build_metrics
from tradelab.models import Candle
from tradelab.utils.indicators import atr
from tradelab.utils.position_sizing import calculate_position_size

from .execution import (
    apply_fill,
    clamp_stop,
    day_key_et,
    day_key_utc,
    estimate_bar_ms,
    is_eod_bar,
    oco_exit_check,
    round_step,
    touched_limit,
)
from .financing import financing_cost
from .signal import as_number, call_signal_with_context, normalize_signal


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


def _option(options: Mapping[str, Any], snake: str, default: Any) -> Any:
    return options.get(snake, options.get(_camel(snake), default))


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValidationError(f"{name} must be finite", context={name: value})
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ValidationError(f"{name} must be finite", context={name: value}) from error
    if not math.isfinite(result):
        raise ValidationError(f"{name} must be finite", context={name: value})
    return result


def _time(value: object) -> int:
    if isinstance(value, bool):
        raise ValidationError("candle time must be Unix milliseconds", context={"time": value})
    try:
        result = int(_finite(value, "time"))
        datetime.fromtimestamp(result / 1_000, UTC)
    except (TypeError, ValueError, OverflowError, OSError) as error:
        raise ValidationError(
            "candle time must be Unix milliseconds", context={"time": value}
        ) from error
    return result


def _normalize_candle(value: object, index: int) -> dict[str, Any]:
    if isinstance(value, Candle):
        return dict(value.to_dict())
    if not isinstance(value, Mapping):
        raise ValidationError("candle must be a mapping", context={"index": index})
    required = ("time", "open", "high", "low", "close")
    if any(key not in value for key in required):
        raise ValidationError(
            "candle requires time, open, high, low, and close", context={"index": index}
        )
    candle: dict[str, Any] = {"time": _time(value["time"])}
    for key in required[1:]:
        candle[key] = _finite(value[key], f"candle.{key}")
    volume = value.get("volume")
    if volume is not None:
        candle["volume"] = _finite(volume, "candle.volume")
    high, low = float(candle["high"]), float(candle["low"])
    if (
        high < low
        or high < max(float(candle["open"]), float(candle["close"]))
        or low > min(float(candle["open"]), float(candle["close"]))
    ):
        raise ValidationError("candle OHLC range is invalid", context={"index": index})
    return candle


def _normalize_candles(value: object) -> list[dict[str, Any]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValidationError("candles must be a sequence")
    candles = [_normalize_candle(bar, index) for index, bar in enumerate(value)]
    if not candles:
        raise ValidationError("backtest requires non-empty candles")
    return candles


def _iso(time_ms: float) -> str:
    return (
        datetime.fromtimestamp(time_ms / 1_000, UTC)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _equity_point(time: float, equity: float) -> dict[str, float]:
    return {"time": time, "timestamp": time, "equity": equity}


class _StrictHistory(Sequence[dict[str, Any]]):
    """Read-only no-lookahead history view mirroring the JavaScript proxy."""

    def __init__(self, values: Sequence[dict[str, Any]], index: int) -> None:
        self._values = values
        self._index = index

    @overload
    def __getitem__(self, index: int) -> dict[str, Any]: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[dict[str, Any]]: ...

    def __getitem__(self, index: int | slice) -> dict[str, Any] | Sequence[dict[str, Any]]:
        if isinstance(index, int) and index >= len(self._values):
            raise RuntimeError(
                "strict mode: signal() tried to access "
                f"candles[{index}] beyond current index {self._index}"
            )
        return self._values[index]

    def __len__(self) -> int:
        return len(self._values)


class BarSystemRunner:
    """Incremental, deterministic implementation of the standalone bar oracle."""

    def __init__(self, options: Mapping[str, object] | None = None, /, **kwargs: object) -> None:
        raw: dict[str, Any] = dict(options or {})
        raw.update(kwargs)
        self.raw = raw
        self.candles = _normalize_candles(_option(raw, "candles", []))
        signal = _option(raw, "signal", None)
        if not callable(signal):
            raise ValidationError("backtest requires a signal callable")
        self.signal: Callable[[dict[str, object]], object] = signal
        self.symbol = str(_option(raw, "symbol", "UNKNOWN"))
        self.interval = _option(raw, "interval", None)
        self.range = _option(raw, "range", None)
        self.equity_start = _finite(_option(raw, "equity", 10_000), "equity")
        self.current_equity = self.equity_start
        self.risk_pct = _finite(
            _option(
                raw, "risk_pct", _option(raw, "riskFraction", 0.01) if "riskFraction" in raw else 1
            ),
            "risk_pct",
        )
        if "riskFraction" in raw or "risk_fraction" in raw:
            self.risk_pct = (
                _finite(
                    _option(raw, "risk_fraction", _option(raw, "riskFraction", 0.01)),
                    "risk_fraction",
                )
                * 100
            )
        self.warmup_bars = int(_option(raw, "warmup_bars", 200))
        self.slippage_bps = _finite(_option(raw, "slippage_bps", 1), "slippage_bps")
        self.fee_bps = _finite(_option(raw, "fee_bps", 0), "fee_bps")
        costs = _option(raw, "costs", None)
        self.costs: Mapping[str, object] | None = costs if isinstance(costs, Mapping) else None
        self.scale_out_at_r = _finite(_option(raw, "scale_out_at_r", 1), "scale_out_at_r")
        self.scale_out_frac = _finite(_option(raw, "scale_out_frac", 0.5), "scale_out_frac")
        self.final_tp_r = _finite(_option(raw, "final_tp_r", 3), "final_tp_r")
        self.max_daily_loss_pct = _finite(
            _option(raw, "max_daily_loss_pct", 2), "max_daily_loss_pct"
        )
        self.atr_trail_mult = _finite(_option(raw, "atr_trail_mult", 0), "atr_trail_mult")
        self.atr_trail_period = int(_option(raw, "atr_trail_period", 14))
        self.oco: dict[str, Any] = {
            "mode": "intrabar",
            "tieBreak": "pessimistic",
            "clampStops": True,
            "clampEpsBps": 0.25,
        }
        if isinstance(_option(raw, "oco", None), Mapping):
            self.oco.update(_option(raw, "oco", None))
        self.trigger = str(_option(raw, "trigger_mode", self.oco["mode"]))
        self.flatten_at_close = bool(_option(raw, "flatten_at_close", True))
        self.daily_max_trades = int(_option(raw, "daily_max_trades", 0))
        self.post_loss_cooldown_bars = int(_option(raw, "post_loss_cooldown_bars", 0))
        self.mfe_trail: dict[str, Any] = {"enabled": False, "armR": 1, "givebackR": 0.5}
        if isinstance(_option(raw, "mfe_trail", None), Mapping):
            self.mfe_trail.update(_option(raw, "mfe_trail", None))
        self.pyramiding: dict[str, Any] = {
            "enabled": False,
            "addAtR": 1,
            "addFrac": 0.25,
            "maxAdds": 1,
            "onlyAfterBreakEven": True,
        }
        if isinstance(_option(raw, "pyramiding", None), Mapping):
            self.pyramiding.update(_option(raw, "pyramiding", None))
        self.vol_scale: dict[str, Any] = {
            "enabled": False,
            "atrPeriod": self.atr_trail_period,
            "cutIfAtrX": 1.3,
            "cutFrac": 0.33,
            "noCutAboveR": 1.5,
        }
        if isinstance(_option(raw, "vol_scale", None), Mapping):
            self.vol_scale.update(_option(raw, "vol_scale", None))
        self.qty_step = _finite(_option(raw, "qty_step", 0.001), "qty_step")
        self.min_qty = _finite(_option(raw, "min_qty", 0.001), "min_qty")
        self.max_leverage = _finite(_option(raw, "max_leverage", 2), "max_leverage")
        self.entry_chase: dict[str, Any] = {
            "enabled": True,
            "afterBars": 2,
            "maxSlipR": 0.2,
            "convertOnExpiry": False,
        }
        if isinstance(_option(raw, "entry_chase", None), Mapping):
            self.entry_chase.update(_option(raw, "entry_chase", None))
        self.reanchor_stop_on_fill = bool(_option(raw, "reanchor_stop_on_fill", True))
        self.max_slip_r_on_fill = _finite(
            _option(raw, "max_slip_r_on_fill", 0.4), "max_slip_r_on_fill"
        )
        self.want_eq_series = bool(_option(raw, "collect_eq_series", True))
        self.want_replay = bool(_option(raw, "collect_replay", True))
        self.strict = bool(_option(raw, "strict", False))
        self.estimated_bar_ms = estimate_bar_ms(self.candles)
        need_atr = self.atr_trail_mult > 0 or bool(self.vol_scale["enabled"])
        self.atr_values = (
            atr(
                self.candles,
                int(
                    self.vol_scale["atrPeriod"]
                    if self.vol_scale["enabled"]
                    else self.atr_trail_period
                ),
            )
            if need_atr
            else None
        )
        self.closed: list[dict[str, Any]] = []
        self.open: dict[str, Any] | None = None
        self.pending: dict[str, Any] | None = None
        self.cooldown = 0
        self.current_day: str | None = None
        self.day_pnl = 0.0
        self.day_trades = 0
        self.day_equity_start = self.current_equity
        self.trade_id = 0
        self.start_index = min(max(1, self.warmup_bars), len(self.candles))
        self.history = self.candles[: self.start_index]
        self.index = self.start_index
        self.last_bar = self.history[-1] if self.history else None
        self.eq_series: list[dict[str, float]] = (
            [_equity_point(float(self.candles[0]["time"]), self.current_equity)]
            if self.want_eq_series
            else []
        )
        self.replay_frames: list[dict[str, object]] = []
        self.replay_events: list[dict[str, object]] = []

    def has_next(self) -> bool:
        return self.index < len(self.candles)

    def peek_time(self) -> float:
        return float(self.candles[self.index]["time"]) if self.has_next() else math.inf

    def get_mark_price(self) -> float | None:
        return float(self.last_bar["close"]) if self.last_bar else None

    def get_marked_equity(self) -> float:
        if not self.open or not self.last_bar:
            return self.current_equity
        return self.current_equity + (
            float(self.last_bar["close"]) - float(self.open["entryFill"])
        ) * (1 if self.open["side"] == "long" else -1) * float(self.open["size"])

    def _record_frame(self, bar: Mapping[str, Any]) -> None:
        time = float(bar["time"])
        if self.want_eq_series:
            self.eq_series.append(_equity_point(time, self.current_equity))
        if self.want_replay:
            self.replay_frames.append(
                {
                    "t": _iso(time),
                    "price": bar["close"],
                    "equity": self.current_equity,
                    "posSide": self.open["side"] if self.open else None,
                    "posSize": self.open["size"] if self.open else 0,
                }
            )

    def _close_leg(
        self, qty: float, exit_price: float, exit_fee: float, bar: Mapping[str, Any], reason: str
    ) -> None:
        assert self.open is not None
        open_pos = self.open
        entry_fill = float(open_pos["entryFill"])
        pnl = (exit_price - entry_fill) * (1 if open_pos["side"] == "long" else -1) * qty
        pnl -= float(open_pos.get("entryFeeTotal", 0)) * qty / float(open_pos["initSize"])
        financing = financing_cost(
            str(open_pos["side"]),
            entry_fill * qty,
            float(open_pos["openTime"]),
            float(bar["time"]),
            self.costs,
        )
        pnl -= exit_fee + financing
        self.current_equity += pnl
        self.day_pnl += pnl
        if self.want_eq_series:
            self.eq_series.append(_equity_point(float(bar["time"]), self.current_equity))
        remaining = float(open_pos["size"]) - qty
        event_type = {"SCALE": "scale-out", "TP": "tp", "SL": "sl", "EOD": "eod"}.get(
            reason, "exit" if remaining <= 0 else "scale-out"
        )
        if self.want_replay:
            self.replay_events.append(
                {
                    "t": _iso(float(bar["time"])),
                    "price": exit_price,
                    "type": event_type,
                    "side": open_pos["side"],
                    "size": qty,
                    "tradeId": open_pos["id"],
                    "reason": reason,
                    "pnl": pnl,
                }
            )
        record = dict(open_pos)
        record.update(
            {
                "size": qty,
                "exit": {
                    "price": exit_price,
                    "time": bar["time"],
                    "reason": reason,
                    "pnl": pnl,
                    "financing": financing,
                    "exitATR": open_pos.get("_lastATR"),
                },
                "mfeR": open_pos.get("_mfeR", 0),
                "maeR": open_pos.get("_maeR", 0),
                "adds": open_pos.get("_adds", 0),
            }
        )
        self.closed.append(record)
        open_pos["size"] = remaining
        open_pos["_realized"] = float(open_pos.get("_realized", 0)) + pnl

    def _force_exit(self, reason: str, bar: Mapping[str, Any]) -> None:
        if not self.open:
            return
        fill = apply_fill(
            float(bar["close"]),
            "short" if self.open["side"] == "long" else "long",
            slippage_bps=self.slippage_bps,
            fee_bps=self.fee_bps,
            qty=float(self.open["size"]),
            costs=self.costs,
        )
        self._close_leg(float(self.open["size"]), fill["price"], fill["fee_total"], bar, reason)
        self.cooldown = int(self.open.get("_cooldownBars", 0))
        self.open = None

    def _tighten_breakeven(self, bar: Mapping[str, Any]) -> None:
        assert self.open is not None
        realized = float(self.open.get("_realized", 0))
        if realized <= 0 or float(self.open["size"]) <= 0:
            return
        candidate = (
            float(self.open["entryFill"]) - abs(realized / float(self.open["size"]))
            if self.open["side"] == "long"
            else float(self.open["entryFill"]) + abs(realized / float(self.open["size"]))
        )
        candidate = (
            max(float(self.open["stop"]), candidate)
            if self.open["side"] == "long"
            else min(float(self.open["stop"]), candidate)
        )
        self.open["stop"] = (
            clamp_stop(float(bar["close"]), candidate, str(self.open["side"]), self.oco)
            if self.oco["clampStops"]
            else candidate
        )

    def _open_from_pending(self, bar: Mapping[str, Any], entry: float, kind: str) -> bool:
        if not self.pending:
            return False
        pending = self.pending
        planned = max(1e-8, float(pending["plannedRiskAbs"]))
        if abs(entry - float(pending["entry"])) / planned > self.max_slip_r_on_fill:
            return False
        stop = float(pending["stop"])
        if self.reanchor_stop_on_fill:
            stop = entry - planned if pending["side"] == "long" else entry + planned
        target = float(pending["tp"])
        rr = as_number(pending["meta"].get("_rr")) if isinstance(pending["meta"], Mapping) else None
        if self.reanchor_stop_on_fill and rr is not None:
            planned_target = (
                float(pending["entry"]) + rr * planned
                if pending["side"] == "long"
                else float(pending["entry"]) - rr * planned
            )
            if abs(target - planned_target) <= max(1e-8, planned * 1e-6):
                target = (
                    entry + rr * abs(entry - stop)
                    if pending["side"] == "long"
                    else entry - rr * abs(entry - stop)
                )
        requested = pending.get("fixedQty")
        size = (
            float(requested)
            if requested is not None
            else calculate_position_size(
                equity=self.current_equity,
                entry=entry,
                stop=stop,
                risk_fraction=float(pending["riskFrac"]),
                qty_step=self.qty_step,
                min_qty=self.min_qty,
                max_leverage=self.max_leverage,
            )
        )
        size = round_step(size, self.qty_step)
        if size < self.min_qty:
            return False
        fill = apply_fill(
            entry,
            str(pending["side"]),
            slippage_bps=self.slippage_bps,
            fee_bps=self.fee_bps,
            kind=kind,
            qty=size,
            costs=self.costs,
        )
        meta = dict(pending["meta"]) if isinstance(pending["meta"], Mapping) else {}
        self.trade_id += 1
        self.open = {
            "symbol": self.symbol,
            **meta,
            "id": self.trade_id,
            "side": pending["side"],
            "entry": entry,
            "stop": stop,
            "takeProfit": target,
            "size": size,
            "openTime": bar["time"],
            "entryFill": fill["price"],
            "entryFeeTotal": fill["fee_total"],
            "initSize": size,
            "baseSize": size,
            "_mfeR": 0,
            "_maeR": 0,
            "_adds": 0,
            "_initRisk": abs(entry - stop) or 1e-8,
        }
        if self.atr_values and self.atr_values[self.index] is not None:
            self.open["entryATR"] = self.atr_values[self.index]
            self.open["_lastATR"] = self.atr_values[self.index]
        self.day_trades += 1
        self.pending = None
        if self.want_replay:
            self.replay_events.append(
                {
                    "t": _iso(float(bar["time"])),
                    "price": fill["price"],
                    "type": "entry",
                    "side": self.open["side"],
                    "size": size,
                    "tradeId": self.trade_id,
                }
            )
        return True

    def _manage_open(self, bar: Mapping[str, Any]) -> None:
        assert self.open is not None
        open_pos = self.open
        risk = float(open_pos["_initRisk"])
        high_r = (
            (float(bar["high"]) - float(open_pos["entry"])) / risk
            if open_pos["side"] == "long"
            else (float(open_pos["entry"]) - float(bar["low"])) / risk
        )
        low_r = (
            (float(bar["low"]) - float(open_pos["entry"])) / risk
            if open_pos["side"] == "long"
            else (float(open_pos["entry"]) - float(bar["high"])) / risk
        )
        mark_r = (
            (float(bar["close"]) - float(open_pos["entry"])) / risk
            if open_pos["side"] == "long"
            else (float(open_pos["entry"]) - float(bar["close"])) / risk
        )
        if self.atr_values and self.atr_values[self.index] is not None:
            open_pos["_lastATR"] = self.atr_values[self.index]
        open_pos["_mfeR"] = max(float(open_pos.get("_mfeR", -math.inf)), high_r)
        open_pos["_maeR"] = min(float(open_pos.get("_maeR", math.inf)), low_r)

        def tighten(candidate: float) -> None:
            value = (
                max(float(open_pos["stop"]), candidate)
                if open_pos["side"] == "long"
                else min(float(open_pos["stop"]), candidate)
            )
            open_pos["stop"] = (
                clamp_stop(float(bar["close"]), value, str(open_pos["side"]), self.oco)
                if self.oco["clampStops"]
                else value
            )

        if (
            float(open_pos.get("_breakevenAtR", 0)) > 0
            and high_r >= float(open_pos["_breakevenAtR"])
            and not open_pos.get("_beArmed")
        ):
            tighten(float(open_pos["entry"]))
            open_pos["_beArmed"] = True
        if float(open_pos.get("_trailAfterR", 0)) > 0 and high_r >= float(open_pos["_trailAfterR"]):
            tighten(
                float(bar["close"]) - risk
                if open_pos["side"] == "long"
                else float(bar["close"]) + risk
            )
        if self.mfe_trail["enabled"] and float(open_pos["_mfeR"]) >= float(self.mfe_trail["armR"]):
            r = max(0.0, float(open_pos["_mfeR"]) - max(0.0, float(self.mfe_trail["givebackR"])))
            tighten(
                float(open_pos["entry"]) + r * risk
                if open_pos["side"] == "long"
                else float(open_pos["entry"]) - r * risk
            )
        if self.atr_trail_mult > 0 and self.atr_values and self.atr_values[self.index] is not None:
            atr_value = self.atr_values[self.index]
            assert atr_value is not None
            distance = atr_value * self.atr_trail_mult
            tighten(
                float(bar["close"]) - distance
                if open_pos["side"] == "long"
                else float(bar["close"]) + distance
            )
        # Volatility scale, then add, then scale-out: this ordering is observable.
        if (
            self.vol_scale["enabled"]
            and open_pos.get("entryATR")
            and float(open_pos["size"]) > self.min_qty
            and self.atr_values
            and self.atr_values[self.index] is not None
        ):
            atr_value = self.atr_values[self.index]
            assert atr_value is not None
            ratio = atr_value / max(1e-12, float(open_pos["entryATR"]))
            if (
                ratio >= float(self.vol_scale["cutIfAtrX"])
                and mark_r < float(self.vol_scale["noCutAboveR"])
                and not open_pos.get("_volCutDone")
            ):
                qty = round_step(
                    float(open_pos["size"]) * float(self.vol_scale["cutFrac"]), self.qty_step
                )
                if self.min_qty <= qty < float(open_pos["size"]):
                    fill = apply_fill(
                        float(bar["close"]),
                        "short" if open_pos["side"] == "long" else "long",
                        slippage_bps=self.slippage_bps,
                        fee_bps=self.fee_bps,
                        qty=qty,
                        costs=self.costs,
                    )
                    self._close_leg(qty, fill["price"], fill["fee_total"], bar, "SCALE")
                    self._tighten_breakeven(bar)
                    open_pos["_volCutDone"] = True
        added = False
        if self.pyramiding["enabled"] and int(open_pos.get("_adds", 0)) < int(
            self.pyramiding["maxAdds"]
        ):
            add_n = int(open_pos.get("_adds", 0)) + 1
            price = (
                float(open_pos["entry"]) + float(self.pyramiding["addAtR"]) * add_n * risk
                if open_pos["side"] == "long"
                else float(open_pos["entry"]) - float(self.pyramiding["addAtR"]) * add_n * risk
            )
            touched = (
                (
                    float(bar["high"]) >= price
                    if self.trigger == "intrabar"
                    else float(bar["close"]) >= price
                )
                if open_pos["side"] == "long"
                else (
                    float(bar["low"]) <= price
                    if self.trigger == "intrabar"
                    else float(bar["close"]) <= price
                )
            )
            be = (
                not self.pyramiding["onlyAfterBreakEven"]
                or (
                    open_pos["side"] == "long"
                    and float(open_pos["stop"]) >= float(open_pos["entry"])
                )
                or (
                    open_pos["side"] == "short"
                    and float(open_pos["stop"]) <= float(open_pos["entry"])
                )
            )
            qty = round_step(
                float(open_pos.get("baseSize", open_pos["initSize"]))
                * float(self.pyramiding["addFrac"]),
                self.qty_step,
            )
            if be and touched and qty >= self.min_qty:
                fill = apply_fill(
                    price,
                    str(open_pos["side"]),
                    slippage_bps=self.slippage_bps,
                    fee_bps=self.fee_bps,
                    kind="limit",
                    qty=qty,
                    costs=self.costs,
                )
                old = float(open_pos["size"])
                new = old + qty
                open_pos["entryFeeTotal"] = float(open_pos["entryFeeTotal"]) + fill["fee_total"]
                open_pos["entryFill"] = (
                    float(open_pos["entryFill"]) * old + fill["price"] * qty
                ) / new
                open_pos["size"] = new
                open_pos["initSize"] = float(open_pos["initSize"]) + qty
                open_pos["_adds"] = add_n
                added = True
        if not added and not open_pos.get("_scaled") and self.scale_out_at_r > 0:
            price = (
                float(open_pos["entry"]) + self.scale_out_at_r * risk
                if open_pos["side"] == "long"
                else float(open_pos["entry"]) - self.scale_out_at_r * risk
            )
            touched = (
                (
                    float(bar["high"]) >= price
                    if self.trigger == "intrabar"
                    else float(bar["close"]) >= price
                )
                if open_pos["side"] == "long"
                else (
                    float(bar["low"]) <= price
                    if self.trigger == "intrabar"
                    else float(bar["close"]) <= price
                )
            )
            qty = round_step(float(open_pos["size"]) * self.scale_out_frac, self.qty_step)
            if touched and self.min_qty <= qty < float(open_pos["size"]):
                fill = apply_fill(
                    price,
                    "short" if open_pos["side"] == "long" else "long",
                    slippage_bps=self.slippage_bps,
                    fee_bps=self.fee_bps,
                    kind="limit",
                    qty=qty,
                    costs=self.costs,
                )
                self._close_leg(qty, fill["price"], fill["fee_total"], bar, "SCALE")
                open_pos["_scaled"] = True
                open_pos["takeProfit"] = (
                    float(open_pos["entry"]) + self.final_tp_r * risk
                    if open_pos["side"] == "long"
                    else float(open_pos["entry"]) - self.final_tp_r * risk
                )
                self._tighten_breakeven(bar)
                open_pos["_beArmed"] = True
        hit = oco_exit_check(
            side=str(open_pos["side"]),
            stop=float(open_pos["stop"]),
            tp=float(open_pos["takeProfit"]),
            bar=bar,
            mode=str(self.oco["mode"]),
            tie_break=str(self.oco["tieBreak"]),
        )
        if hit["hit"]:
            kind = "limit" if hit["hit"] == "TP" else "stop"
            fill = apply_fill(
                _finite(hit["px"], "oco exit price"),
                "short" if open_pos["side"] == "long" else "long",
                slippage_bps=self.slippage_bps,
                fee_bps=self.fee_bps,
                kind=kind,
                qty=float(open_pos["size"]),
                costs=self.costs,
            )
            local = int(open_pos.get("_cooldownBars", 0))
            self._close_leg(
                float(open_pos["size"]), fill["price"], fill["fee_total"], bar, str(hit["hit"])
            )
            self.cooldown = (
                max(self.cooldown, self.post_loss_cooldown_bars)
                if hit["hit"] == "SL"
                else self.cooldown
            ) or local
            self.open = None

    def step(self) -> dict[str, object] | None:
        if not self.has_next():
            return None
        bar = self.candles[self.index]
        self.history.append(bar)
        self.last_bar = bar
        key = (
            day_key_et(float(bar["time"]))
            if self.flatten_at_close or self.trigger == "close"
            else day_key_utc(float(bar["time"]))
        )
        if key != self.current_day:
            self.current_day = key
            self.day_pnl = 0.0
            self.day_trades = 0
            self.day_equity_start = self.current_equity
        if (
            self.open
            and int(self.open.get("_maxBarsInTrade", 0)) > 0
            and max(
                1,
                round((float(bar["time"]) - float(self.open["openTime"])) / self.estimated_bar_ms),
            )
            >= int(self.open["_maxBarsInTrade"])
        ):
            self._force_exit("TIME", bar)
        if (
            self.open
            and as_number(self.open.get("_maxHoldMin"))
            and float(self.open["_maxHoldMin"]) > 0
            and (float(bar["time"]) - float(self.open["openTime"])) / 60_000
            >= float(self.open["_maxHoldMin"])
        ):
            self._force_exit("TIME", bar)
        if self.open and self.flatten_at_close and is_eod_bar(float(bar["time"])):
            self._force_exit("EOD", bar)
        if self.open:
            self._manage_open(bar)
        # Intentional standalone JS behavior: zero means 0-dollar limit and blocks entry.
        daily_loss_hit = self.day_pnl <= -abs(self.max_daily_loss_pct / 100 * self.day_equity_start)
        trade_cap = self.daily_max_trades > 0 and self.day_trades >= self.daily_max_trades
        if not self.open and self.pending:
            if self.index > int(self.pending["expiresAt"]) or daily_loss_hit or trade_cap:
                if self.entry_chase["enabled"] and self.entry_chase["convertOnExpiry"]:
                    self._open_from_pending(bar, float(bar["close"]), "market")
                else:
                    self.pending = None
            elif touched_limit(
                str(self.pending["side"]), float(self.pending["entry"]), bar, self.trigger
            ):
                self._open_from_pending(bar, float(self.pending["entry"]), "limit")
            elif self.entry_chase["enabled"]:
                elapsed = self.index - int(self.pending.get("startedAtIndex", self.index))
                meta = self.pending.get("meta")
                midpoint = (
                    as_number(meta.get("_imb", {}).get("mid"))
                    if isinstance(meta, Mapping) and isinstance(meta.get("_imb"), Mapping)
                    else None
                )
                if (
                    not self.pending.get("_chasedCE")
                    and midpoint is not None
                    and elapsed >= max(1, int(self.entry_chase["afterBars"]))
                ):
                    self.pending["entry"] = midpoint
                    self.pending["_chasedCE"] = True
                if self.pending and self.pending.get("_chasedCE"):
                    risk_ref = abs(
                        _finite(
                            (meta.get("_initRisk") if isinstance(meta, Mapping) else None)
                            or float(self.pending["entry"]) - float(self.pending["stop"]),
                            "pending initial risk",
                        )
                    )
                    direction = 1 if self.pending["side"] == "long" else -1
                    slipped_r = max(
                        0.0,
                        direction
                        * (float(bar["close"]) - float(self.pending["entry"]))
                        / max(1e-8, risk_ref),
                    )
                    if slipped_r > self.max_slip_r_on_fill:
                        self.pending = None
                    elif 0 < slipped_r <= float(self.entry_chase["maxSlipR"]):
                        self._open_from_pending(bar, float(bar["close"]), "market")
        if self.open or self.cooldown > 0:
            if self.cooldown > 0:
                self.cooldown -= 1
            self._record_frame(bar)
            self.index += 1
            return bar
        if daily_loss_hit or trade_cap:
            self.pending = None
            self._record_frame(bar)
            self.index += 1
            return bar
        if self.pending:
            self._record_frame(bar)
            self.index += 1
            return bar
        if self.strict and len(self.history) != self.index + 1:
            raise ValidationError(
                f"strict mode: signal() received {len(self.history)} candles at index {self.index}"
            )
        context: dict[str, Any] = {
            "candles": _StrictHistory(self.history, self.index) if self.strict else self.history,
            "index": self.index,
            "bar": bar,
            "equity": self.current_equity,
            "openPosition": self.open,
            "pendingOrder": self.pending,
        }
        raw = call_signal_with_context(self.signal, context, self.index, bar, self.symbol)
        next_signal = normalize_signal(raw, bar, self.final_tp_r)
        if next_signal:
            risk_frac = (
                next_signal.get("riskFraction")
                if next_signal.get("riskFraction") is not None
                else (
                    _finite(next_signal["riskPct"], "signal riskPct") / 100
                    if next_signal.get("riskPct") is not None
                    else self.risk_pct / 100
                )
            )
            expiry = as_number(next_signal.get("_entryExpiryBars")) or 5
            self.pending = {
                "side": next_signal["side"],
                "entry": next_signal["entry"],
                "stop": next_signal["stop"],
                "tp": next_signal["takeProfit"],
                "riskFrac": risk_frac,
                "fixedQty": next_signal.get("qty"),
                "expiresAt": self.index + max(1, int(expiry)),
                "startedAtIndex": self.index,
                "meta": next_signal,
                "plannedRiskAbs": abs(
                    _finite(
                        next_signal.get("_initRisk")
                        or _finite(next_signal["entry"], "signal entry")
                        - _finite(next_signal["stop"], "signal stop"),
                        "signal initial risk",
                    )
                ),
            }
            if touched_limit(
                str(self.pending["side"]), float(self.pending["entry"]), bar, self.trigger
            ):
                self._open_from_pending(bar, float(self.pending["entry"]), "limit")
        self._record_frame(bar)
        self.index += 1
        return bar

    def build_result(self) -> dict[str, Any]:
        metrics = build_metrics(
            closed=self.closed,
            equity_start=self.equity_start,
            equity_final=self.current_equity,
            candles=self.candles,
            est_bar_ms=self.estimated_bar_ms,
            eq_series=self.eq_series,
            interval=self.interval if isinstance(self.interval, str) else None,
            benchmark_returns=_option(self.raw, "benchmark_returns", None),
        )
        open_positions = []
        if self.open:
            mark = float(self.candles[-1]["close"])
            direction = 1 if self.open["side"] == "long" else -1
            open_positions = [
                {
                    "id": self.open["id"],
                    "symbol": self.open["symbol"],
                    "side": self.open["side"],
                    "size": self.open["size"],
                    "entry": self.open["entry"],
                    "entryFill": self.open["entryFill"],
                    "stop": self.open["stop"],
                    "takeProfit": self.open["takeProfit"],
                    "openTime": self.open["openTime"],
                    "markPrice": mark,
                    "unrealizedPnl": (mark - float(self.open["entryFill"]))
                    * direction
                    * float(self.open["size"]),
                    "_initRisk": self.open["_initRisk"],
                }
            ]
        return {
            "symbol": self.symbol,
            "interval": self.interval,
            "range": self.range,
            "trades": self.closed,
            "positions": [x for x in self.closed if x["exit"]["reason"] != "SCALE"],
            "openPositions": open_positions,
            "metrics": _js_key(metrics),
            "eqSeries": self.eq_series,
            "replay": {"frames": self.replay_frames, "events": self.replay_events},
        }


def backtest(options: Mapping[str, object] | None = None, /, **kwargs: object) -> dict[str, Any]:
    """Exhaust :class:`BarSystemRunner` synchronously without terminal liquidation."""
    runner = BarSystemRunner(options, **kwargs)
    while runner.has_next():
        runner.step()
    return runner.build_result()
