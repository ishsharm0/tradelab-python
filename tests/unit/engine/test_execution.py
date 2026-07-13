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
from tradelab.errors import ValidationError


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


@pytest.mark.parametrize("slippage_by_kind", [{}, {"limit": "invalid"}])
def test_apply_fill_falls_back_to_kind_multiplier_for_invalid_override(
    slippage_by_kind: dict[str, object],
) -> None:
    fill = apply_fill(
        100,
        "long",
        kind="limit",
        costs={"slippageBps": 4, "slippageByKind": slippage_by_kind},
    )

    assert fill["price"] == pytest.approx(100.01)


def test_day_key_et_matches_javascript_utc_anchor_at_et_midnight_crossing() -> None:
    # The immutable oracle combines the UTC calendar date with the ET wall clock.
    # At 02:00 UTC the real New York date is the prior day, but the oracle key is not.
    assert day_key_et(1_704_161_600_000) == "2024-01-02"


@pytest.mark.parametrize(
    ("operation", "match"),
    [
        (lambda: apply_fill(10**10_000, "long"), "price"),
        (lambda: apply_fill(100, "invalid"), "side"),
        (lambda: apply_fill(100, "long", slippage_bps=float("nan")), "slippage_bps"),
        (lambda: apply_fill(1e308, "long", slippage_bps=1e308), "finite"),
        (lambda: clamp_stop(100, 99, "invalid", {}), "side"),
        (lambda: touched_limit("invalid", 100, {"low": 99, "high": 101}), "side"),
        (
            lambda: oco_exit_check(
                side="invalid",
                stop=99,
                tp=101,
                bar={"high": 101, "low": 99, "close": 100},
            ),
            "side",
        ),
        (lambda: round_step(1, 0), "step"),
        (lambda: round_step(1e308, 1e-308), "step quotient"),
        (lambda: round_step(float("nan")), "value"),
        (lambda: estimate_bar_ms([{"time": 0}, {"time": 10**10_000}]), "time"),
    ],
)
def test_execution_boundaries_raise_validation_error(operation: object, match: str) -> None:
    with pytest.raises(ValidationError, match=match):
        operation()  # type: ignore[operator]
