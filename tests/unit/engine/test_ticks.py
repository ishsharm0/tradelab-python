"""Tick-engine chronology, validation, and execution-cost tests."""

from __future__ import annotations

import json
from typing import Any

import pytest

from tradelab.engine.backtest_ticks import backtest_ticks
from tradelab.errors import StrategyError, ValidationError

START = 1_704_205_800_000


def _ticks(*prices: float) -> list[dict[str, float]]:
    return [
        {"time": START + index * 60_000, "price": price, "size": 1}
        for index, price in enumerate(prices)
    ]


def _one_long(context: dict[str, object]) -> dict[str, object] | None:
    if context["index"] != 0:
        return None
    return {"side": "long", "stop": 95, "takeProfit": 110, "qty": 2}


def test_market_signal_fills_only_on_later_tick_and_liquidates_at_eot() -> None:
    result = backtest_ticks(
        ticks=_ticks(100, 101),
        signal=_one_long,
        slippage_bps=0,
        fee_bps=0,
    )

    trade = result["trades"][0]
    assert trade["openTime"] == START + 60_000
    assert trade["entry"] == 101
    assert trade["exit"] == {
        "price": 101,
        "time": START + 60_000,
        "reason": "EOT",
        "pnl": 0,
        "financing": 0,
    }
    assert result["openPositions"] == []
    assert [point["time"] for point in result["eqSeries"]] == [
        START,
        START,
        START + 60_000,
        START + 60_000,
    ]
    assert len(result["replay"]["frames"]) == 3
    assert result["replay"]["frames"][-2]["posSide"] == "long"
    assert result["replay"]["frames"][-1]["posSide"] is None


def test_limit_signal_cannot_fill_on_signal_tick_then_oco_uses_later_tick() -> None:
    ticks = [
        {"time": START, "bid": 99, "ask": 101, "low": 98, "high": 103, "size": 2},
        {"time": START + 60_000, "bid": 99, "ask": 101, "low": 99, "high": 101},
        {"time": START + 120_000, "price": 104, "low": 100, "high": 105},
    ]
    result = backtest_ticks(
        ticks=ticks,
        signal=lambda context: (
            {"side": "buy", "entry": 100, "stop": 98, "takeProfit": 104, "qty": 1}
            if context["index"] == 0
            else None
        ),
        slippage_bps=0,
    )

    trade = result["positions"][0]
    assert trade["openTime"] == START + 60_000
    assert trade["exit"]["time"] == START + 120_000
    assert trade["exit"]["reason"] == "TP"
    assert result["replay"]["events"][0]["type"] == "entry"
    assert result["replay"]["events"][1]["type"] == "tp"


def test_oco_tie_break_and_short_aliases_match_tick_oracle() -> None:
    result = backtest_ticks(
        ticks=[
            {"time": START, "price": 100},
            {"time": START + 60_000, "price": 100},
            {"time": START + 120_000, "bid": 89, "ask": 111},
        ],
        signal=lambda context: (
            {"direction": "sell", "stopLoss": 110, "target": 90, "size": 1}
            if context["index"] == 0
            else None
        ),
        slippage_bps=0,
        oco={"tieBreak": "optimistic"},
    )

    assert result["trades"][0]["side"] == "short"
    assert result["trades"][0]["exit"]["reason"] == "TP"
    assert result["trades"][0]["exit"]["price"] == 90


def test_queue_probability_is_seeded_and_zero_keeps_limit_pending() -> None:
    options: dict[str, Any] = {
        "ticks": _ticks(100, 100, 100),
        "signal": lambda context: (
            {"side": "long", "entry": 100, "stop": 99, "takeProfit": 102, "qty": 1}
            if context["index"] == 0
            else None
        ),
        "slippage_bps": 0,
        "queue_fill_probability": 0,
        "seed": "same-seed",
    }
    first = backtest_ticks(**options)
    second = backtest_ticks(**options)

    assert first == second
    assert first["trades"] == []
    assert all(frame["posSide"] is None for frame in first["replay"]["frames"])


def test_seeded_fractional_queue_fill_is_reproducible() -> None:
    options: dict[str, Any] = {
        "ticks": _ticks(100, 100, 100),
        "signal": lambda context: (
            {"side": "long", "entry": 100, "stop": 99, "takeProfit": 102, "qty": 1}
            if context["index"] == 0
            else None
        ),
        "slippage_bps": 0,
        "queue_fill_probability": 0.75,
        "seed": "queue-🌙",
    }

    assert backtest_ticks(**options) == backtest_ticks(**options)


def test_daily_trade_cap_prevents_a_second_entry() -> None:
    def signal(context: dict[str, object]) -> dict[str, object] | None:
        if context["index"] in {0, 2}:
            return {"side": "long", "stop": 99, "takeProfit": 101, "qty": 1}
        return None

    result = backtest_ticks(
        ticks=_ticks(100, 100, 101, 100, 101),
        signal=signal,
        daily_max_trades=1,
        slippage_bps=0,
    )

    assert len(result["trades"]) == 1


def test_daily_loss_gate_blocks_reentry_after_realized_loss() -> None:
    def signal(context: dict[str, object]) -> dict[str, object] | None:
        if context["index"] in {0, 2}:
            return {"side": "long", "stop": 99, "takeProfit": 110, "qty": 200}
        return None

    result = backtest_ticks(
        ticks=_ticks(100, 100, 99, 100, 100),
        signal=signal,
        max_daily_loss_pct=1,
        slippage_bps=0,
    )

    assert len(result["trades"]) == 1
    assert result["trades"][0]["exit"]["pnl"] == -200


def test_fill_costs_and_financing_flow_through_shared_primitives() -> None:
    day = 86_400_000
    result = backtest_ticks(
        ticks=[
            {"time": START, "price": 100},
            {"time": START + day, "price": 100},
        ],
        signal=_one_long,
        slippage_bps=0,
        fee_bps=0,
        costs={
            "commissionPerOrder": 1,
            "carry": {"longAnnualBps": 365},
        },
    )

    trade = result["trades"][0]
    assert trade["entryFeeTotal"] == 1
    assert trade["exit"]["financing"] == pytest.approx(0)
    assert trade["exit"]["pnl"] == pytest.approx(-2)


def test_collection_flags_and_json_safety() -> None:
    result = backtest_ticks(
        ticks=_ticks(100),
        signal=lambda _context: None,
        collect_eq_series=False,
        collect_replay=False,
        range={"from": START, "to": START},
    )

    assert result["eqSeries"] == []
    assert result["replay"] == {"frames": [], "events": []}
    assert json.loads(json.dumps(result.to_dict(), allow_nan=False)) == result.to_dict()


def test_signal_context_contains_normalized_tick_history() -> None:
    contexts: list[dict[str, object]] = []
    result = backtest_ticks(
        ticks=[
            {"time": START, "bid": 99, "ask": 101, "size": 3},
            {"time": "bad", "price": 200},
        ],
        symbol="ES",
        signal=lambda context: contexts.append(context),
    )

    bar = contexts[0]["bar"]
    assert isinstance(bar, dict)
    assert bar["open"] == bar["close"] == 100
    assert bar["low"] == 99
    assert bar["high"] == 101
    assert bar["volume"] == 3
    assert result["symbol"] == "ES"


def test_signal_errors_include_tick_index_time_and_symbol() -> None:
    with pytest.raises(StrategyError, match=r"index=0.*symbol=NQ.*boom"):
        backtest_ticks(
            ticks=_ticks(100),
            symbol="NQ",
            signal=lambda _context: (_ for _ in ()).throw(RuntimeError("boom")),
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"ticks": []}, "non-empty ticks"),
        ({"ticks": "bad"}, "ticks must be a sequence"),
        ({"ticks": [{"time": "bad"}]}, "could not normalize"),
        ({"signal": None}, "signal callable"),
        ({"queue_fill_probability": -0.1}, "queue_fill_probability"),
        ({"queue_fill_probability": 1.1}, "queue_fill_probability"),
        ({"qty_step": 0}, "qty_step"),
        ({"daily_max_trades": 1.5}, "daily_max_trades"),
        ({"max_daily_loss_pct": -1}, "max_daily_loss_pct"),
        ({"oco": {"tieBreak": "coinflip"}}, "tieBreak"),
        ({"costs": []}, "costs"),
    ],
)
def test_invalid_public_inputs_raise_validation_error(
    kwargs: dict[str, object], message: str
) -> None:
    options: dict[str, object] = {"ticks": _ticks(100), "signal": lambda _context: None}
    options.update(kwargs)
    with pytest.raises(ValidationError, match=message):
        backtest_ticks(**options)
