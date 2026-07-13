"""Chronology tests for the shared deterministic bar runner."""

from __future__ import annotations

import pytest

from tradelab.engine.backtest import BarSystemRunner, backtest
from tradelab.errors import StrategyError, ValidationError


def _bars(*prices: tuple[float, float, float, float]) -> list[dict[str, float | int]]:
    start = 1_704_205_800_000
    return [
        {"time": start + i * 60_000, "open": o, "high": h, "low": low, "close": c}
        for i, (o, h, low, c) in enumerate(prices)
    ]


def test_new_touched_limit_fills_on_signal_bar_but_cannot_stop_until_later_bar() -> None:
    result = backtest(
        candles=_bars((100, 100, 100, 100), (100, 102, 98, 100), (100, 100, 98, 99)),
        warmup_bars=1,
        flatten_at_close=False,
        scale_out_at_r=0,
        slippage_bps=0,
        signal=lambda context: (
            {"side": "buy", "entry": 100, "stop": 99, "takeProfit": 102, "qty": 1}
            if context["index"] == 1
            else None
        ),
    )

    assert result["trades"][0]["openTime"] == 1_704_205_860_000
    assert result["trades"][0]["exit"]["reason"] == "SL"
    assert result["trades"][0]["exit"]["time"] > result["trades"][0]["openTime"]


def test_time_exit_precedes_oco_on_same_bar() -> None:
    result = backtest(
        candles=_bars((100, 100, 100, 100), (100, 100, 100, 100), (100, 111, 89, 105)),
        warmup_bars=1,
        flatten_at_close=False,
        scale_out_at_r=0,
        slippage_bps=0,
        signal=lambda context: (
            {
                "side": "long",
                "entry": 100,
                "stop": 90,
                "takeProfit": 110,
                "qty": 1,
                "_maxBarsInTrade": 1,
            }
            if context["index"] == 1
            else None
        ),
    )

    assert result["positions"][0]["exit"]["reason"] == "TIME"
    assert result["positions"][0]["exit"]["price"] == 105


def test_runner_and_synchronous_backtest_share_exact_result() -> None:
    options = {
        "candles": _bars((100, 100, 100, 100), (100, 101, 99, 100), (100, 103, 100, 102)),
        "warmupBars": 1,
        "flattenAtClose": False,
        "scaleOutAtR": 0,
        "slippageBps": 0,
        "signal": lambda context: (
            {"side": "long", "entry": 100, "stop": 99, "takeProfit": 102, "qty": 1}
            if context["index"] == 1
            else None
        ),
    }
    runner = BarSystemRunner(options)
    assert runner.peek_time() == 1_704_205_860_000
    assert runner.get_mark_price() == 100
    assert runner.get_marked_equity() == 10_000
    while runner.has_next():
        runner.step()
    assert runner.peek_time() == float("inf")

    assert runner.build_result() == backtest(**options)


def test_standalone_zero_daily_loss_limit_blocks_entries_like_oracle() -> None:
    result = backtest(
        candles=_bars((100, 100, 100, 100), (100, 101, 99, 100)),
        warmup_bars=1,
        flatten_at_close=False,
        max_daily_loss_pct=0,
        signal=lambda _context: {"side": "long", "entry": 100, "stop": 99, "qty": 1},
    )

    assert result["trades"] == []


def test_collection_flags_disable_equity_and_replay_without_changing_metrics() -> None:
    result = backtest(
        candles=_bars((100, 100, 100, 100), (100, 100, 100, 100)),
        warmup_bars=1,
        collect_eq_series=False,
        collect_replay=False,
        signal=lambda _context: None,
    )

    assert result["eqSeries"] == []
    assert result["replay"] == {"frames": [], "events": []}
    assert result["metrics"]["trades"] == 0


def test_signal_error_contains_index_time_and_symbol() -> None:
    with pytest.raises(StrategyError, match=r"index=1.*symbol=ES.*boom"):
        backtest(
            candles=_bars((100, 100, 100, 100), (100, 100, 100, 100)),
            warmup_bars=1,
            symbol="ES",
            signal=lambda _context: (_ for _ in ()).throw(RuntimeError("boom")),
        )


def test_strict_history_rejects_direct_lookahead() -> None:
    with pytest.raises(
        StrategyError, match=r"strict mode: signal\(\) tried to access candles\[2\]"
    ):
        backtest(
            candles=_bars((100, 100, 100, 100), (100, 100, 100, 100), (100, 100, 100, 100)),
            warmup_bars=1,
            strict=True,
            signal=lambda context: context["candles"][context["index"] + 1],
        )


def test_malformed_candle_is_rejected_at_public_boundary() -> None:
    with pytest.raises(ValidationError):
        backtest(
            candles=[{"time": 0, "open": 1, "high": 0, "low": 1, "close": 1}], signal=lambda _: None
        )


def test_scale_add_trails_and_oco_are_processed_in_deterministic_order() -> None:
    result = backtest(
        candles=_bars(
            (100, 100, 100, 100),
            (100, 100, 100, 100),
            (100, 105, 99, 103),
            (103, 108, 101, 106),
            (106, 120, 104, 115),
        ),
        warmup_bars=1,
        flatten_at_close=False,
        scale_out_at_r=1,
        scale_out_frac=0.5,
        final_tp_r=10,
        atr_trail_mult=0.25,
        atr_trail_period=2,
        vol_scale={"enabled": True, "atrPeriod": 2, "cutIfAtrX": 1, "cutFrac": 0.25},
        mfe_trail={"enabled": True, "armR": 1, "givebackR": 0.5},
        pyramiding={"enabled": True, "onlyAfterBreakEven": False, "addAtR": 1, "addFrac": 0.25},
        slippage_bps=0,
        signal=lambda context: (
            {"side": "long", "entry": 100, "stop": 98, "takeProfit": 120, "qty": 4}
            if context["index"] == 1
            else None
        ),
    )
    assert result["trades"][-1]["exit"]["reason"] == "SL"
    assert result["trades"][-1]["adds"] == 1


def test_pending_expiry_can_convert_to_market_and_open_position_stays_open() -> None:
    result = backtest(
        candles=_bars(
            (100, 100, 100, 100),
            (101, 101, 101, 101),
            (103, 103, 103, 103),
            (104, 104, 104, 104),
        ),
        warmup_bars=1,
        flatten_at_close=False,
        scale_out_at_r=0,
        entry_chase={"enabled": True, "convertOnExpiry": True},
        slippage_bps=0,
        signal=lambda context: (
            {
                "side": "short",
                "entry": 105,
                "stop": 120,
                "takeProfit": 80,
                "qty": 1,
                "_entryExpiryBars": 1,
            }
            if context["index"] == 1
            else None
        ),
    )
    assert result["openPositions"][0]["entry"] == 104


def test_pending_chase_reanchors_to_imbalance_midpoint() -> None:
    result = backtest(
        candles=_bars(
            (100, 100, 100, 100),
            (100, 100, 100, 100),
            (101, 101, 101, 101),
            (102, 102, 102, 102),
        ),
        warmup_bars=1,
        flatten_at_close=False,
        scale_out_at_r=0,
        entry_chase={"enabled": True, "afterBars": 1, "maxSlipR": 0.2},
        slippage_bps=0,
        signal=lambda context: (
            {
                "side": "long",
                "entry": 99,
                "stop": 94,
                "takeProfit": 120,
                "qty": 1,
                "_imb": {"mid": 100},
            }
            if context["index"] == 1
            else None
        ),
    )
    assert result["openPositions"][0]["entry"] == 101


def test_eod_exit_precedes_oco_and_daily_trade_cap_blocks_reentry() -> None:
    bars = [
        {"time": 1_704_229_080_000, "open": 100, "high": 100, "low": 100, "close": 100},
        {"time": 1_704_229_140_000, "open": 100, "high": 100, "low": 100, "close": 100},
        {"time": 1_704_229_200_000, "open": 100, "high": 101, "low": 99, "close": 100},
    ]
    result = backtest(
        candles=bars,
        warmup_bars=1,
        flatten_at_close=True,
        scale_out_at_r=0,
        daily_max_trades=1,
        slippage_bps=0,
        signal=lambda context: {
            "side": "long",
            "entry": 100,
            "stop": 99,
            "takeProfit": 101,
            "qty": 1,
        },
    )
    assert result["positions"][0]["exit"]["reason"] == "EOD"


def test_scale_out_precedes_later_oco_and_positions_exclude_scale_legs() -> None:
    result = backtest(
        candles=_bars(
            (100, 100, 100, 100),
            (100, 100, 100, 100),
            (100, 103, 100, 102),
            (102, 105, 101, 104),
        ),
        warmup_bars=1,
        flatten_at_close=False,
        scale_out_at_r=1,
        scale_out_frac=0.5,
        final_tp_r=2,
        slippage_bps=0,
        signal=lambda context: (
            {"side": "long", "entry": 100, "stop": 98, "takeProfit": 120, "qty": 4}
            if context["index"] == 1
            else None
        ),
    )

    assert [leg["exit"]["reason"] for leg in result["trades"]] == ["SCALE", "TP"]
    assert len(result["positions"]) == 1


def test_private_pending_rejection_and_breakeven_are_deterministic() -> None:
    runner = BarSystemRunner(
        candles=_bars((100, 100, 100, 100), (100, 100, 100, 100)),
        warmup_bars=1,
        flatten_at_close=False,
        signal=lambda _context: None,
    )
    bar = runner.candles[1]
    runner.pending = {
        "side": "long",
        "entry": 100,
        "stop": 99,
        "tp": 102,
        "riskFrac": 0.01,
        "fixedQty": 1,
        "plannedRiskAbs": 1,
        "meta": {},
    }
    assert not runner._open_from_pending(bar, 102, "market")
    runner.open = {
        "side": "long",
        "entryFill": 100,
        "stop": 98,
        "size": 1,
        "_realized": 2,
    }
    runner._tighten_breakeven(bar)
    assert runner.open["stop"] >= 98
