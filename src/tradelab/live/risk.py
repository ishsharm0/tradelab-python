"""Live-trading circuit breakers and exposure gates."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from typing import Any

from tradelab.engine.execution import day_key_et
from tradelab.errors import ValidationError
from tradelab.utils.time import in_windows_et, is_session, parse_windows_csv


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


def _nonnegative(value: object, name: str) -> float:
    number = _finite(value, name)
    if number < 0:
        raise ValidationError(f"{name} must be non-negative", context={name: value})
    return number


def _integer(value: object, name: str) -> int:
    number = _nonnegative(value, name)
    if not number.is_integer():
        raise ValidationError(f"{name} must be a non-negative integer", context={name: value})
    return int(number)


def _camel(name: str) -> str:
    parts = name.split("_")
    return parts[0] + "".join(part.title() for part in parts[1:])


def _option(options: Mapping[str, Any], name: str, default: Any) -> Any:
    for key in (name, _camel(name)):
        if key in options and options[key] is not None:
            return options[key]
    return default


class RiskManager:
    """Stateful ET-day risk gate mirroring the JavaScript manager."""

    def __init__(self, options: Mapping[str, object] | None = None, /, **kwargs: object) -> None:
        raw: dict[str, Any] = dict(options or {})
        raw.update(kwargs)
        self.max_daily_loss_pct = _nonnegative(
            _option(raw, "max_daily_loss_pct", 2), "max_daily_loss_pct"
        )
        absolute = _option(raw, "max_daily_loss_dollars", None)
        self.max_daily_loss_dollars = (
            None if absolute is None else _nonnegative(absolute, "max_daily_loss_dollars")
        )
        self.max_drawdown_pct = _nonnegative(
            _option(raw, "max_drawdown_pct", 20), "max_drawdown_pct"
        )
        self.max_positions = _integer(_option(raw, "max_positions", 10), "max_positions")
        self.max_position_pct = _nonnegative(
            _option(raw, "max_position_pct", 50), "max_position_pct"
        )
        self.max_daily_trades = _integer(_option(raw, "max_daily_trades", 0), "max_daily_trades")
        self.cooldown_after_loss_ms = _nonnegative(
            _option(raw, "cooldown_after_loss_ms", 0), "cooldown_after_loss_ms"
        )
        allowed_sessions = _option(raw, "allowed_sessions", "AUTO")
        if not isinstance(allowed_sessions, str) or allowed_sessions.upper() not in {
            "AUTO",
            "NYSE",
            "FUT",
        }:
            raise ValidationError("allowed_sessions must be AUTO, NYSE, or FUT")
        self.allowed_sessions = allowed_sessions.upper()
        raw_windows = _option(raw, "allowed_windows", None)
        if raw_windows is not None and not isinstance(raw_windows, str):
            raise ValidationError("allowed_windows must be a CSV string or None")
        self.allowed_windows = parse_windows_csv(raw_windows)
        self.max_gross_exposure_pct = _nonnegative(
            _option(raw, "max_gross_exposure_pct", 0), "max_gross_exposure_pct"
        )
        self.max_net_exposure_pct = _nonnegative(
            _option(raw, "max_net_exposure_pct", 0), "max_net_exposure_pct"
        )
        self.start_equity: float | None = None
        self.current_equity: float | None = None
        self.peak_equity: float | None = None
        self.current_day_key: str | None = None
        self.day_pnl = 0.0
        self.day_trades = 0
        self.last_loss_at: float | None = None
        self.halted = False
        self.halt_reason: str | None = None

    def initialize(self, equity: object, time_ms: object | None = None) -> None:
        value = _nonnegative(equity, "equity")
        timestamp = _finite(_now_ms() if time_ms is None else time_ms, "time_ms")
        key = day_key_et(timestamp)
        self.start_equity = value
        self.current_equity = value
        self.peak_equity = value
        self.current_day_key = key
        self.day_pnl = 0.0
        self.day_trades = 0
        self.last_loss_at = None
        self.halted = False
        self.halt_reason = None

    def _reset_day(self, time_ms: float) -> None:
        next_day = day_key_et(time_ms)
        if self.current_day_key == next_day:
            return
        self.current_day_key = next_day
        self.day_pnl = 0.0
        self.day_trades = 0
        self.halted = False
        self.halt_reason = None

    def update(self, *, time_ms: object, equity: object) -> None:
        timestamp = _finite(time_ms, "time_ms")
        value = _nonnegative(equity, "equity")
        if self.start_equity is None:
            self.initialize(value, timestamp)
            return
        self._reset_day(timestamp)
        self.current_equity = value
        if self.peak_equity is None or value > self.peak_equity:
            self.peak_equity = value
        self._maybe_halt_for_drawdown()
        self._maybe_halt_for_daily_loss()

    def _maybe_halt_for_drawdown(self) -> None:
        if self.halted or self.current_equity is None or not (self.peak_equity or 0) > 0:
            return
        assert self.peak_equity is not None
        drawdown = (self.peak_equity - self.current_equity) / self.peak_equity
        maximum = self.max_drawdown_pct / 100
        if maximum > 0 and drawdown >= maximum:
            self.halt(f"max drawdown reached ({drawdown * 100:.2f}%)")

    def _maybe_halt_for_daily_loss(self) -> None:
        if self.halted:
            return
        start = self.start_equity or 0
        pct_hit = self.max_daily_loss_pct > 0 and self.day_pnl <= -abs(
            start * self.max_daily_loss_pct / 100
        )
        dollars_hit = (
            self.max_daily_loss_dollars is not None and self.day_pnl <= -self.max_daily_loss_dollars
        )
        if pct_hit or dollars_hit:
            self.halt("daily loss limit reached")

    def is_session_allowed(self, time_ms: object) -> bool:
        timestamp = _finite(time_ms, "time_ms")
        return is_session(timestamp, self.allowed_sessions) and in_windows_et(
            timestamp, self.allowed_windows
        )

    def can_trade(self, *, time_ms: object | None = None) -> dict[str, object]:
        timestamp = _finite(_now_ms() if time_ms is None else time_ms, "time_ms")
        if self.halted:
            return {"ok": False, "reason": self.halt_reason or "risk halt active"}
        if not self.is_session_allowed(timestamp):
            return {"ok": False, "reason": "outside allowed session/window"}
        if (
            self.cooldown_after_loss_ms > 0
            and self.last_loss_at is not None
            and timestamp - self.last_loss_at < self.cooldown_after_loss_ms
        ):
            return {"ok": False, "reason": "cooldown after loss active"}
        return {"ok": True, "reason": None}

    def can_open_position(
        self,
        *,
        time_ms: object | None = None,
        position_count: object = 0,
        position_value: object = 0,
        equity: object | None = None,
        gross_exposure: object | None = None,
        net_exposure: object | None = None,
    ) -> dict[str, object]:
        decision = self.can_trade(time_ms=time_ms)
        if not decision["ok"]:
            return decision
        count = _integer(position_count, "position_count")
        value = _finite(position_value, "position_value")
        if self.max_positions > 0 and count >= self.max_positions:
            return {"ok": False, "reason": "max positions reached"}
        if self.max_daily_trades > 0 and self.day_trades >= self.max_daily_trades:
            return {"ok": False, "reason": "max daily trades reached"}
        resolved_equity = self.current_equity if equity is None else _nonnegative(equity, "equity")
        if (
            self.max_position_pct > 0
            and resolved_equity is not None
            and resolved_equity > 0
            and abs(value) / resolved_equity > self.max_position_pct / 100
        ):
            return {"ok": False, "reason": "max position size exceeded"}
        return self._check_exposure(
            gross_exposure=gross_exposure,
            net_exposure=net_exposure,
            equity=resolved_equity,
        )

    def check_exposure(
        self,
        *,
        gross_exposure: object | None = None,
        net_exposure: object | None = None,
        equity: object | None = None,
    ) -> dict[str, object]:
        resolved = self.current_equity if equity is None else _nonnegative(equity, "equity")
        return self._check_exposure(
            gross_exposure=gross_exposure, net_exposure=net_exposure, equity=resolved
        )

    def _check_exposure(
        self,
        *,
        gross_exposure: object | None,
        net_exposure: object | None,
        equity: float | None,
    ) -> dict[str, object]:
        gross = None if gross_exposure is None else _finite(gross_exposure, "gross_exposure")
        net = None if net_exposure is None else _finite(net_exposure, "net_exposure")
        if equity is not None and equity > 0:
            if (
                self.max_gross_exposure_pct > 0
                and gross is not None
                and abs(gross) / equity > self.max_gross_exposure_pct / 100
            ):
                return {"ok": False, "reason": "max gross exposure exceeded"}
            if (
                self.max_net_exposure_pct > 0
                and net is not None
                and abs(net) / equity > self.max_net_exposure_pct / 100
            ):
                return {"ok": False, "reason": "max net exposure exceeded"}
        return {"ok": True, "reason": None}

    def record_trade(
        self, *, pnl: object = 0, time_ms: object | None = None, equity: object | None = None
    ) -> None:
        timestamp = _finite(_now_ms() if time_ms is None else time_ms, "time_ms")
        realized = _finite(pnl, "pnl")
        next_equity = None if equity is None else _nonnegative(equity, "equity")
        self._reset_day(timestamp)
        self.day_pnl += realized
        self.day_trades += 1
        if realized < 0:
            self.last_loss_at = timestamp
        if next_equity is not None:
            self.current_equity = next_equity
            if self.peak_equity is None or next_equity > self.peak_equity:
                self.peak_equity = next_equity
        self._maybe_halt_for_daily_loss()
        self._maybe_halt_for_drawdown()

    def halt(self, reason: str = "manual halt") -> None:
        self.halted = True
        self.halt_reason = str(reason)

    def clear_halt(self) -> None:
        self.halted = False
        self.halt_reason = None

    def get_state(self) -> dict[str, object]:
        return {
            "startEquity": self.start_equity,
            "currentEquity": self.current_equity,
            "peakEquity": self.peak_equity,
            "dayPnl": self.day_pnl,
            "dayTrades": self.day_trades,
            "currentDayKey": self.current_day_key,
            "halted": self.halted,
            "haltReason": self.halt_reason,
            "lastLossAt": self.last_loss_at,
        }


def create_risk_manager(
    options: Mapping[str, object] | None = None, **kwargs: object
) -> RiskManager:
    return RiskManager(options, **kwargs)
