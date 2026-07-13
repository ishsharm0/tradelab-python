"""Behavioral and adversarial tests for aggregate performance metrics."""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping, Sequence
from typing import cast

import pytest
from hypothesis import given
from hypothesis import strategies as st

from tradelab.errors import ValidationError
from tradelab.metrics import BIG_NUMBER, build_metrics


def _leg(
    *,
    time: int,
    pnl: float,
    side: str = "long",
    entry: float = 100.0,
    entry_fill: float | None = 100.0,
    risk: float = 1.0,
    open_time: int | None = None,
    reason: str | None = None,
) -> dict[str, object]:
    exit_price = entry + pnl if side == "long" else entry - pnl
    return {
        "side": side,
        "entry": entry,
        "entryFill": entry_fill,
        "_initRisk": risk,
        "openTime": time - 60_000 if open_time is None else open_time,
        "exit": {
            "price": exit_price,
            "time": time,
            "reason": reason or ("TP" if pnl >= 0 else "SL"),
            "pnl": pnl,
        },
    }


def _metrics(
    closed: Sequence[Mapping[str, object]],
    *,
    equity_start: float = 1_000.0,
    equity_final: float | None = None,
    candles: Sequence[Mapping[str, object]] | None = None,
    est_bar_ms: float = 60_000.0,
    eq_series: Sequence[Mapping[str, object]] | None = None,
    interval: str | None = "1d",
    benchmark_returns: Sequence[float] | None = None,
) -> dict[str, object]:
    return build_metrics(
        closed=closed,
        equity_start=equity_start,
        equity_final=equity_start if equity_final is None else equity_final,
        candles=[] if candles is None else candles,
        est_bar_ms=est_bar_ms,
        eq_series=eq_series,
        interval=interval,
        benchmark_returns=benchmark_returns,
    )


def test_no_trades_or_candles_returns_a_complete_finite_shape() -> None:
    metrics = _metrics([])
    assert metrics["trades"] == 0
    assert metrics["exposure_pct"] == 0.0
    assert metrics["profit_factor"] == 0.0
    assert metrics["daily"] == {"count": 1, "win_rate": 0.0, "avg_return": 0.0}
    assert metrics["benchmark"] == {
        "alpha": None,
        "beta": None,
        "correlation": None,
        "information_ratio": None,
        "tracking_error": None,
    }
    json.dumps(metrics, allow_nan=False)


def test_flat_trade_counts_and_resets_streaks() -> None:
    metrics = _metrics([_leg(time=1, pnl=1), _leg(time=2, pnl=0), _leg(time=3, pnl=-1)])
    assert metrics["trades"] == 3
    assert metrics["max_consec_wins"] == 1
    assert metrics["max_consec_losses"] == 1


@pytest.mark.parametrize(
    ("pnl", "profit_factor", "win_rate"),
    [(5.0, BIG_NUMBER, 1.0), (-5.0, 0.0, 0.0)],
)
def test_wins_and_losses_only_use_profit_factor_sentinels(
    pnl: float, profit_factor: float, win_rate: float
) -> None:
    metrics = _metrics([_leg(time=1, pnl=pnl)])
    assert metrics["profit_factor"] == profit_factor
    assert metrics["profit_factor_pos"] == profit_factor
    assert metrics["win_rate"] == win_rate


def test_zero_risk_and_short_r_multiple_are_handled_like_javascript() -> None:
    zero_risk = _metrics([_leg(time=1, pnl=5, risk=0)])
    assert zero_risk["total_r"] == 0.0
    assert zero_risk["avg_r"] == 0.0

    short = _metrics([_leg(time=1, pnl=10, side="short", risk=2)])
    assert short["total_r"] == 5.0
    assert short["short"] == {"trades": 1, "win_rate": 1.0, "avg_pnl": 10.0, "avg_r": 5.0}


def test_entry_fill_zero_is_used_instead_of_entry() -> None:
    trade = _leg(time=1, pnl=0, entry=100, entry_fill=0, risk=1)
    trade["exit"] = {"price": 2, "time": 1, "reason": "TP", "pnl": 1}
    assert _metrics([trade])["total_r"] == 2.0


def test_scale_legs_affect_realized_pnl_drawdown_and_leg_stats_only() -> None:
    scale = _leg(time=1, pnl=-20, reason="SCALE")
    complete = _leg(time=2, pnl=10)
    metrics = _metrics([scale, complete], equity_start=1_000, equity_final=1_010)
    assert metrics["trades"] == 1
    assert metrics["total_pnl"] == -10.0
    assert metrics["max_drawdown"] == pytest.approx(0.02)
    assert metrics["profit_factor"] == BIG_NUMBER
    assert metrics["profit_factor_leg"] == 0.5
    assert metrics["win_rate_leg"] == 0.5


def test_minimal_scale_legs_need_only_exit_time_pnl_and_reason() -> None:
    later = {"exit": {"time": 2, "pnl": 10.0, "reason": "SCALE"}}
    earlier = {"exit": {"time": 1, "pnl": -20.0, "reason": "SCALE"}}
    metrics = _metrics([later, earlier], equity_start=1_000, equity_final=990)
    assert metrics["trades"] == 0
    assert metrics["total_pnl"] == -10.0
    assert metrics["max_drawdown"] == pytest.approx(0.02)
    assert metrics["profit_factor_leg"] == 0.5
    assert metrics["win_rate_leg"] == 0.5
    assert metrics["daily"] == {"count": 1, "win_rate": 0.0, "avg_return": -0.01}


def test_sort_is_stable_and_never_mutates_input() -> None:
    first = _leg(time=1, pnl=-10)
    second = _leg(time=1, pnl=20)
    closed = [first, second]
    original = copy.deepcopy(closed)
    metrics = _metrics(closed, equity_start=100)
    assert closed == original
    assert metrics["max_drawdown"] == pytest.approx(0.1)


def test_explicit_final_equity_diverges_from_realized_pnl_and_exposure_uses_js_rounding() -> None:
    trade = _leg(time=5, pnl=1, open_time=0)
    metrics = _metrics(
        [trade],
        equity_start=1_000,
        equity_final=1_500,
        candles=[{}, {}, {}, {}],
        est_bar_ms=2,
    )
    assert metrics["return_pct"] == 0.5
    assert metrics["total_pnl"] == 1.0
    assert metrics["exposure_pct"] == 0.75
    assert _metrics([trade], candles=[{}], est_bar_ms=2)["exposure_pct"] == 3.0


def test_daily_returns_honor_utc_order_equal_times_filtered_days_and_benchmark_length() -> None:
    day = 86_400_000
    metrics = _metrics(
        [],
        eq_series=[
            {"time": day + 4, "equity": 110.0},
            {"time": day + 2, "equity": 100.0},
            {"time": day + 4, "equity": 120.0},
            {"time": 1, "equity": 0.0},
            {"time": 2, "equity": 5.0},
        ],
        benchmark_returns=[0.1, 0.2],
    )
    assert metrics["daily"] == {"count": 1, "win_rate": 1.0, "avg_return": 0.2}
    assert cast(Mapping[str, object], metrics["benchmark"])["beta"] is None


def test_wrong_type_benchmark_returns_produces_null_benchmark() -> None:
    metrics = _metrics([], benchmark_returns=cast(Sequence[float], True))
    assert metrics["benchmark"] == {
        "alpha": None,
        "beta": None,
        "correlation": None,
        "information_ratio": None,
        "tracking_error": None,
    }


def test_sharpe_sortino_edge_cases_and_unclamped_finite_annualization() -> None:
    one_return = _metrics(
        [], eq_series=[{"time": 0, "equity": 1_000}, {"time": 1, "equity": 1_002}]
    )
    assert one_return["sharpe"] == BIG_NUMBER
    assert one_return["sortino_daily"] == BIG_NUMBER
    assert one_return["sharpe_annualized"] == pytest.approx(BIG_NUMBER * math.sqrt(252))

    zero_return = _metrics(
        [], eq_series=[{"time": 0, "equity": 1_000}, {"time": 1, "equity": 1_000}]
    )
    assert zero_return["sharpe"] == BIG_NUMBER
    assert zero_return["sortino_daily"] == 0.0


def test_percentiles_are_lower_order_statistics_for_two_elements() -> None:
    first = _leg(time=1, pnl=-1, risk=1)
    second = _leg(time=2, pnl=1, risk=1)
    metrics = _metrics([first, second])
    assert metrics["r_dist"] == {"p10": -1.0, "p25": -1.0, "p50": -1.0, "p75": -1.0, "p90": -1.0}
    assert metrics["hold_dist_min"] == {
        "p10": 1.0,
        "p25": 1.0,
        "p50": 1.0,
        "p75": 1.0,
        "p90": 1.0,
    }


def test_metrics_cancellation_uses_left_to_right_addition() -> None:
    metrics = _metrics([_leg(time=1, pnl=1e16), _leg(time=2, pnl=1.0), _leg(time=3, pnl=-1e16)])
    assert metrics["total_pnl"] == 0.0
    assert metrics["expectancy"] == 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"closed": {}},
        {"equity_start": math.inf},
        {"equity_final": math.nan},
        {"est_bar_ms": math.inf},
        {"closed": [_leg(time=1, pnl=math.inf)]},
    ],
)
def test_build_metrics_rejects_malformed_public_structures_and_required_nonfinite_values(
    kwargs: dict[str, object],
) -> None:
    baseline: dict[str, object] = {
        "closed": [],
        "equity_start": 1_000.0,
        "equity_final": 1_000.0,
        "candles": [],
        "est_bar_ms": 60_000.0,
    }
    baseline.update(kwargs)
    with pytest.raises(ValidationError):
        build_metrics(**baseline)  # type: ignore[arg-type]


@given(
    st.lists(
        st.floats(min_value=-1000, max_value=1000, allow_nan=False, allow_infinity=False),
        max_size=8,
    )
)
def test_finite_metric_outputs_are_json_safe_and_do_not_mutate_legs(pnls: list[float]) -> None:
    closed = [_leg(time=index + 1, pnl=pnl) for index, pnl in enumerate(pnls)]
    original = copy.deepcopy(closed)
    metrics = _metrics(closed, equity_final=1_000.0)
    assert closed == original
    side_breakdown = cast(Mapping[str, object], metrics["side_breakdown"])
    assert metrics["long"] is side_breakdown["long"]
    assert metrics["short"] is side_breakdown["short"]
    for key, value in metrics.items():
        if key != "benchmark" and isinstance(value, float):
            assert math.isfinite(value)
    json.dumps(metrics, allow_nan=False)
