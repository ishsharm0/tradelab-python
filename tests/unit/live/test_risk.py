"""Live risk circuit-breaker contracts."""

from __future__ import annotations

import math
from datetime import datetime
from typing import cast
from zoneinfo import ZoneInfo

import pytest

from tradelab.errors import ValidationError
from tradelab.live import RiskManager

ET = ZoneInfo("America/New_York")


def _et_ms(hour: int, minute: int = 0, second: int = 0, *, day: int = 2) -> int:
    return round(datetime(2025, 1, day, hour, minute, second, tzinfo=ET).timestamp() * 1_000)


OPEN = _et_ms(9, 30)


def test_daily_loss_halts_and_actual_et_midnight_resets() -> None:
    risk = RiskManager(max_daily_loss_pct=1)
    risk.initialize(10_000, OPEN)
    risk.record_trade(pnl=-150, time_ms=OPEN + 1_000, equity=9_850)
    assert risk.can_trade(time_ms=OPEN + 2_000) == {
        "ok": False,
        "reason": "daily loss limit reached",
    }

    before_midnight = _et_ms(23, 59, 59, day=1)
    at_midnight = _et_ms(0)
    risk.initialize(10_000, before_midnight)
    risk.halt("test")
    risk.update(time_ms=at_midnight, equity=10_000)
    assert risk.get_state()["halted"] is False
    assert risk.get_state()["currentDayKey"] == "2025-01-02"


def test_drawdown_absolute_loss_and_cooldown_gates() -> None:
    risk = RiskManager(
        max_daily_loss_pct=0,
        max_daily_loss_dollars=50,
        max_drawdown_pct=10,
        cooldown_after_loss_ms=10_000,
    )
    risk.initialize(1_000, OPEN)
    risk.update(time_ms=OPEN + 1, equity=1_100)
    risk.update(time_ms=OPEN + 2, equity=980)
    assert cast(str, risk.get_state()["haltReason"]).startswith("max drawdown reached")
    risk.clear_halt()
    risk.record_trade(pnl=-60, time_ms=OPEN + 3, equity=1_040)
    assert risk.get_state()["haltReason"] == "daily loss limit reached"
    risk.clear_halt()
    assert risk.can_trade(time_ms=OPEN + 4)["reason"] == "cooldown after loss active"
    assert risk.can_trade(time_ms=OPEN + 10_004)["ok"] is True


def test_session_window_position_trade_and_exposure_limits() -> None:
    risk = RiskManager(
        allowed_windows="10:00-10:30",
        max_positions=1,
        max_daily_trades=1,
        max_position_pct=50,
        max_gross_exposure_pct=100,
        max_net_exposure_pct=50,
    )
    outside = _et_ms(9, 45)
    inside = _et_ms(10)
    risk.initialize(10_000, outside)
    assert risk.can_trade(time_ms=outside)["reason"] == "outside allowed session/window"
    assert risk.can_open_position(time_ms=inside, position_count=1)["reason"] == (
        "max positions reached"
    )
    assert (
        risk.can_open_position(time_ms=inside, position_value=5_001, equity=10_000)["reason"]
        == "max position size exceeded"
    )
    assert risk.check_exposure(gross_exposure=10_001, equity=10_000)["reason"] == (
        "max gross exposure exceeded"
    )
    assert risk.check_exposure(net_exposure=-5_001, equity=10_000)["reason"] == (
        "max net exposure exceeded"
    )
    risk.record_trade(pnl=1, time_ms=inside, equity=10_001)
    assert risk.can_open_position(time_ms=inside)["reason"] == "max daily trades reached"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_daily_loss_pct": math.nan}, "max_daily_loss_pct"),
        ({"max_positions": 1.5}, "max_positions"),
        ({"allowed_sessions": "CRYPTO"}, "allowed_sessions"),
        ({"allowed_windows": "25:00-26:00"}, "wall-clock"),
    ],
)
def test_risk_options_reject_nonfinite_and_malformed_values(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        RiskManager(**kwargs)


def test_updates_reject_nonfinite_values_without_mutating_state() -> None:
    risk = RiskManager()
    risk.initialize(1_000, OPEN)
    before = risk.get_state()
    with pytest.raises(ValidationError, match="equity"):
        risk.update(time_ms=OPEN, equity=math.inf)
    assert risk.get_state() == before
