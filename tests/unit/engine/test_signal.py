"""Signal aliases, normalization, and diagnostic callback contracts."""

from __future__ import annotations

import pytest

from tradelab.engine.signal import call_signal_with_context, normalize_signal
from tradelab.errors import StrategyError


def test_normalize_signal_supports_aliases_and_risk_reward_target() -> None:
    result = normalize_signal(
        {"action": "sell", "price": 100, "sl": 102, "rr": 2, "size": 3},
        {"close": 99},
        3,
    )

    assert result is not None
    assert result["side"] == "short"
    assert result["takeProfit"] == 96
    assert result["qty"] == 3


def test_normalize_signal_rejects_invalid_direction_and_zero_risk() -> None:
    assert normalize_signal({"side": "flat", "stop": 99}, {"close": 100}, 3) is None
    assert normalize_signal({"side": "long", "entry": 100, "stop": 100}, {"close": 100}, 3) is None


def test_callback_error_is_wrapped_with_bar_context() -> None:
    with pytest.raises(StrategyError, match=r"index=7.*symbol=NQ.*bad"):
        call_signal_with_context(
            lambda _context: (_ for _ in ()).throw(ValueError("bad")),
            {},
            7,
            {"time": 1_704_205_800_000},
            "NQ",
        )
