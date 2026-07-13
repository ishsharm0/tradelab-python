"""Focused contracts for deterministic execution primitives."""

from __future__ import annotations

import pytest

from tradelab.engine.execution import (
    apply_fill,
    clamp_stop,
    day_key_et,
    day_key_utc,
    estimate_bar_ms,
    oco_exit_check,
    round_step,
    touched_limit,
)


def test_apply_fill_composes_kind_slippage_spread_and_commissions() -> None:
    fill = apply_fill(
        100,
        "long",
        kind="limit",
        qty=2,
        costs={
            "slippageBps": 4,
            "spreadBps": 2,
            "commissionBps": 10,
            "commissionPerUnit": 0.5,
            "commissionPerOrder": 1,
            "minCommission": 0,
        },
    )

    assert fill["price"] == pytest.approx(100.02)
    assert fill["fee_total"] == pytest.approx(2.20004)
    assert fill["fee"] == pytest.approx(1.10002)


def test_oco_optimistic_tie_prefers_take_profit() -> None:
    exit_result = oco_exit_check(
        side="long",
        stop=99,
        tp=101,
        bar={"open": 100, "high": 101, "low": 99, "close": 100},
        tie_break="optimistic",
    )

    assert exit_result == {"hit": "TP", "px": 101.0}


def test_touched_limit_is_inclusive_for_long_and_short() -> None:
    bar = {"open": 100, "high": 102, "low": 98, "close": 100}

    assert touched_limit("long", 98, bar)
    assert touched_limit("short", 102, bar)


def test_execution_primitives_cover_close_mode_clamps_and_empty_estimates() -> None:
    bar = {"open": 100, "high": 102, "low": 98, "close": 101}
    assert not touched_limit("long", 100, bar, "close")
    assert oco_exit_check(side="short", stop=102, tp=98, bar=bar, mode="close") == {
        "hit": None,
        "px": None,
    }
    assert clamp_stop(100, 101, "long", {"clampEpsBps": 1}) < 100
    assert clamp_stop(100, 99, "short", {"clampEpsBps": 1}) > 100
    assert round_step(1.239, 0.01) == pytest.approx(1.23)
    assert estimate_bar_ms([]) == 300_000


def test_apply_fill_uses_stop_multiplier_and_minimum_commission() -> None:
    fill = apply_fill(
        100,
        "short",
        kind="stop",
        qty=1,
        costs={"slippageBps": 4, "commissionPerOrder": 0.1, "minCommission": 2},
    )
    assert fill["price"] == pytest.approx(99.95)
    assert fill["fee_total"] == 2


def test_execution_supports_per_kind_costs_and_day_keys() -> None:
    fill = apply_fill(
        100,
        "long",
        kind="market",
        qty=1,
        costs={"slippageByKind": {"market": 10}},
    )

    assert fill["price"] == pytest.approx(100.1)
    assert day_key_utc(1_704_205_800_000) == "2024-01-02"
    assert day_key_et(1_704_205_800_000) == "2024-01-02"
