"""Chronology tests for the shared deterministic bar runner."""

from __future__ import annotations

from typing import cast

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


def test_max_bars_uses_javascript_rounding_at_half_bar() -> None:
    start = 1_704_205_800_000
    bars = [
        {"time": start + offset, "open": 100, "high": 100, "low": 100, "close": 100}
        for offset in (0, 120_000, 240_000, 420_000)
    ]
    result = backtest(
        candles=bars,
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
                "_maxBarsInTrade": 3,
            }
            if context["index"] == 1
            else None
        ),
    )

    assert result["positions"][0]["exit"]["reason"] == "TIME"


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


def test_built_result_cannot_mutate_runner_state() -> None:
    runner = BarSystemRunner(
        candles=_bars((100, 100, 100, 100), (100, 101, 99, 100), (100, 103, 100, 102)),
        warmup_bars=1,
        flatten_at_close=False,
        scale_out_at_r=0,
        slippage_bps=0,
        signal=lambda context: (
            {"side": "long", "entry": 100, "stop": 99, "takeProfit": 102, "qty": 1}
            if context["index"] == 1
            else None
        ),
    )
    while runner.has_next():
        runner.step()
    result = runner.build_result()
    result["trades"][0]["exit"]["pnl"] = 999

    assert runner.build_result()["trades"][0]["exit"]["pnl"] != 999


def test_runner_exposes_capital_pending_and_force_exit_controls() -> None:
    runner = BarSystemRunner(
        candles=_bars(
            (100, 100, 100, 100),
            (100, 100, 100, 100),
            (100, 100, 100, 100),
        ),
        warmup_bars=1,
        flatten_at_close=False,
        scale_out_at_r=0,
        slippage_bps=0,
        max_leverage=2,
        signal=lambda context: (
            {"side": "long", "entry": 100, "stop": 99, "takeProfit": 110, "qty": 2}
            if context["index"] == 1
            else None
        ),
    )

    runner.step()
    assert runner.get_locked_capital() == 100
    runner.pending = {"sentinel": True}
    runner.cancel_pending()
    assert runner.pending is None
    runner.force_exit("RISK")
    assert runner.open is None
    assert runner.closed[-1]["exit"]["reason"] == "RISK"


def test_runner_can_disable_trading_for_a_step_without_invoking_signal() -> None:
    calls: list[int] = []
    runner = BarSystemRunner(
        candles=_bars(
            (100, 100, 100, 100),
            (100, 100, 100, 100),
            (100, 100, 100, 100),
        ),
        warmup_bars=1,
        flatten_at_close=False,
        signal=lambda context: calls.append(int(context["index"])),
    )

    runner.step(can_trade=False)
    runner.step()

    assert calls == [2]


def test_runner_step_uses_shared_equity_and_entry_size_resolver() -> None:
    observed: dict[str, float] = {}

    def resolve_entry_size(request: dict[str, object]) -> float:
        observed["desired"] = cast(float, request["desired_size"])
        return 4

    runner = BarSystemRunner(
        candles=_bars((100, 100, 100, 100), (100, 100, 100, 100)),
        warmup_bars=1,
        flatten_at_close=False,
        scale_out_at_r=0,
        slippage_bps=0,
        signal=lambda context: (
            observed.update(equity=float(context["equity"]))
            or {"side": "long", "entry": 100, "stop": 99, "takeProfit": 110}
        ),
    )

    runner.step(signal_equity=1_000, resolve_entry_size=resolve_entry_size)

    assert observed == {"equity": 1_000, "desired": 10}
    assert runner.open is not None
    assert runner.open["size"] == 4


def test_runner_honors_nullish_defaults_risk_precedence_and_final_tp_alias() -> None:
    runner = BarSystemRunner(
        candles=_bars((100, 100, 100, 100), (100, 100, 100, 100)),
        warmupBars=1,
        symbol=None,
        riskFraction=None,
        riskPct=5,
        finalTP_R=7,
        signal=lambda _context: None,
    )

    assert runner.symbol == "UNKNOWN"
    assert runner.risk_pct == 5
    assert runner.final_tp_r == 7


def test_unrepresentable_risk_fraction_falls_back_without_overflow() -> None:
    runner = BarSystemRunner(
        candles=_bars((100, 100, 100, 100), (100, 100, 100, 100)),
        riskFraction=10**400,
        riskPct=5,
        signal=lambda _context: None,
    )

    assert runner.risk_pct == 5


def test_numeric_string_risk_fraction_falls_back_like_javascript() -> None:
    runner = BarSystemRunner(
        candles=_bars((100, 100, 100, 100), (100, 100, 100, 100)),
        riskFraction="0.5",
        riskPct=2,
        signal=lambda _context: None,
    )

    assert runner.risk_pct == 2


@pytest.mark.parametrize(
    "option",
    [
        {"qty_step": 0},
        {"min_qty": -1},
        {"max_leverage": -1},
        {"atr_trail_period": 0},
        {"scale_out_at_r": -1},
        {"scale_out_frac": 1.1},
        {"final_tp_r": -1},
        {"max_daily_loss_pct": -1},
        {"atr_trail_mult": -1},
        {"daily_max_trades": -1},
        {"post_loss_cooldown_bars": -1},
        {"max_slip_r_on_fill": -0.1},
        {"trigger_mode": "future"},
    ],
)
def test_invalid_numeric_option_domains_are_rejected_eagerly(option: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        BarSystemRunner(
            candles=_bars((100, 100, 100, 100), (100, 100, 100, 100)),
            signal=lambda _context: None,
            **option,
        )


@pytest.mark.parametrize(
    "option",
    [
        {"costs": "bad"},
        {"oco": {"mode": "future"}},
        {"oco": {"tie_break": "random"}},
        {"mfe_trail": {"arm_r": float("inf")}},
        {"pyramiding": {"max_adds": 1.5}},
        {"vol_scale": {"atr_period": float("inf")}},
        {"entry_chase": {"max_slip_r": -1}},
    ],
)
def test_invalid_nested_options_are_rejected_eagerly(option: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        BarSystemRunner(
            candles=_bars((100, 100, 100, 100), (100, 100, 100, 100)),
            signal=lambda _context: None,
            **option,
        )


def test_nested_python_aliases_are_canonicalized_and_costs_are_snapshotted() -> None:
    costs: dict[str, object] = {"funding": {"intervalMs": 1, "rateBps": 1}}
    runner = BarSystemRunner(
        candles=_bars((100, 100, 100, 100), (100, 100, 100, 100)),
        signal=lambda _context: None,
        costs=costs,
        oco={"tie_break": "optimistic", "clamp_eps_bps": 2},
        pyramiding={"max_adds": 3, "only_after_break_even": False},
        vol_scale={"atr_period": 2},
        entry_chase={"after_bars": 4, "convert_on_expiry": True},
    )
    costs["funding"] = {"intervalMs": 1, "rateBps": 999}

    assert runner.oco["tieBreak"] == "optimistic"
    assert runner.oco["clampEpsBps"] == 2
    assert runner.pyramiding["maxAdds"] == 3
    assert runner.pyramiding["onlyAfterBreakEven"] is False
    assert runner.vol_scale["atrPeriod"] == 2
    assert runner.entry_chase["afterBars"] == 4
    assert runner.entry_chase["convertOnExpiry"] is True
    assert runner.costs == {"funding": {"intervalMs": 1, "rateBps": 1}}


@pytest.mark.parametrize(
    ("hook", "value"),
    [
        ("signal_equity", float("inf")),
        ("signal_equity", 10**10_000),
        ("can_trade", "false"),
        ("resolve_entry_size", 3),
    ],
    ids=["infinite-equity", "huge-equity", "nonbool-gate", "noncallable-resolver"],
)
def test_step_hooks_validate_before_consuming_a_bar(hook: str, value: object) -> None:
    runner = BarSystemRunner(
        candles=_bars((100, 100, 100, 100), (100, 100, 100, 100)),
        warmup_bars=1,
        signal=lambda _context: None,
    )
    index = runner.index

    with pytest.raises(ValidationError):
        runner.step(**{hook: value})  # type: ignore[arg-type]
    assert runner.index == index


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


def test_candles_are_normalized_sorted_and_first_deduplicated_without_input_mutation() -> None:
    candles: list[dict[str, object]] = [
        {"timestamp": "2024-01-02T14:31:00Z", "o": 101, "h": 102, "l": 100, "c": 101},
        {"time": 1_704_205_800, "open": 100, "high": 101, "low": 99, "close": 100},
        {"time": 1_704_205_800_000, "open": 999, "high": 999, "low": 999, "close": 999},
    ]
    original = [dict(candle) for candle in candles]

    runner = BarSystemRunner(candles=candles, warmup_bars=1, signal=lambda _context: None)

    assert candles == original
    assert [bar["time"] for bar in runner.candles] == [
        1_704_205_800_000,
        1_704_205_860_000,
    ]
    assert runner.candles[0]["close"] == 100
    assert all(bar["volume"] == 0 for bar in runner.candles)


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


def test_rejected_touched_pending_order_is_canceled_before_next_signal() -> None:
    calls = 0

    def signal(_context: dict[str, object]) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {"side": "long", "entry": 100, "stop": 99, "takeProfit": 102, "qty": 0.0005}

    result = backtest(
        candles=_bars(
            (100, 100, 100, 100),
            (100, 101, 99, 100),
            (100, 101, 99, 100),
            (100, 101, 99, 100),
        ),
        warmup_bars=1,
        flatten_at_close=False,
        signal=signal,
    )

    assert calls == 3
    assert result["trades"] == []


def test_rejected_existing_pending_fill_is_canceled_immediately() -> None:
    calls: list[int] = []

    def signal(context: dict[str, object]) -> dict[str, object] | None:
        index = cast(int, context["index"])
        calls.append(index)
        return (
            {"side": "long", "entry": 90, "stop": 89, "takeProfit": 92, "qty": 0.0005}
            if index == 1
            else None
        )

    backtest(
        candles=_bars(
            (100, 100, 100, 100),
            (100, 101, 99, 100),
            (95, 96, 90, 95),
            (95, 96, 94, 95),
        ),
        warmup_bars=1,
        flatten_at_close=False,
        signal=signal,
    )

    assert calls == [1, 2, 3]


def test_rejected_expiry_conversion_is_canceled_immediately() -> None:
    calls: list[int] = []

    def signal(context: dict[str, object]) -> dict[str, object] | None:
        index = cast(int, context["index"])
        calls.append(index)
        return (
            {
                "side": "long",
                "entry": 90,
                "stop": 89,
                "takeProfit": 92,
                "qty": 0.0005,
                "_entryExpiryBars": 1,
            }
            if index == 1
            else None
        )

    backtest(
        candles=_bars(
            (100, 100, 100, 100),
            (100, 100, 100, 100),
            (100, 100, 100, 100),
            (100, 100, 100, 100),
        ),
        warmup_bars=1,
        flatten_at_close=False,
        max_slip_r_on_fill=20,
        entry_chase={"enabled": True, "convertOnExpiry": True},
        signal=signal,
    )

    assert calls == [1, 3]


def test_rejected_chase_fill_is_canceled_immediately() -> None:
    calls: list[int] = []

    def signal(context: dict[str, object]) -> dict[str, object] | None:
        index = cast(int, context["index"])
        calls.append(index)
        return (
            {
                "side": "long",
                "entry": 90,
                "stop": 89,
                "takeProfit": 92,
                "qty": 0.0005,
                "_imb": {"mid": 99},
            }
            if index == 1
            else None
        )

    backtest(
        candles=_bars(
            (100, 100, 100, 100),
            (100, 100, 100, 100),
            (99.1, 99.1, 99.1, 99.1),
            (99.1, 99.1, 99.1, 99.1),
        ),
        warmup_bars=1,
        flatten_at_close=False,
        entry_chase={"enabled": True, "afterBars": 1, "maxSlipR": 0.2},
        signal=signal,
    )

    assert calls == [1, 2, 3]


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
