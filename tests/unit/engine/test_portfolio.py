"""Shared-capital portfolio execution contracts."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any

import pytest

from tradelab.engine.execution import day_key_et
from tradelab.engine.portfolio import backtest_portfolio
from tradelab.errors import ValidationError

Signal = Callable[[dict[str, object]], Mapping[str, object] | None]
BASE_TIME = 1_735_823_400_000


def _bar(
    time: int,
    close: float,
    *,
    high: float | None = None,
    low: float | None = None,
) -> dict[str, float]:
    return {
        "time": time,
        "open": close,
        "high": close if high is None else high,
        "low": close if low is None else low,
        "close": close,
        "volume": 100.0,
    }


def _flat_bars(count: int = 5, *, offset_ms: int = 0) -> list[dict[str, float]]:
    return [_bar(BASE_TIME + offset_ms + index * 300_000, 100.0) for index in range(count)]


def _never(_: dict[str, object]) -> None:
    return None


def _index_signal(index: int, value: Mapping[str, object]) -> Signal:
    return lambda context: value if context["index"] == index else None


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({}, "non-empty systems"),
        ({"systems": []}, "non-empty systems"),
        ({"systems": "not-systems"}, "non-empty systems"),
        (
            {"systems": [{"candles": _flat_bars(), "signal": _never}], "allocation": "bad"},
            "allocation",
        ),
        (
            {
                "systems": [{"candles": _flat_bars(), "signal": _never}],
                "processing_order": "bad",
            },
            "processing_order",
        ),
        (
            {
                "systems": [{"candles": _flat_bars(), "signal": _never, "weight": 0}],
                "allocation": "weight",
            },
            "positive allocation weights",
        ),
        (
            {"systems": [{"candles": _flat_bars(), "signal": _never}], "equity": math.inf},
            "equity",
        ),
    ],
)
def test_portfolio_rejects_invalid_top_level_configuration(
    kwargs: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        backtest_portfolio(**kwargs)


def test_equal_and_weighted_allocations_publish_original_system_metadata() -> None:
    systems = [
        {
            "symbol": "FIRST",
            "candles": _flat_bars(),
            "signal": _never,
            "weight": 2,
            "max_allocation": 400,
        },
        {
            "symbol": "SECOND",
            "candles": _flat_bars(),
            "signal": _never,
            "weight": 1,
            "max_allocation_pct": 0.2,
        },
    ]

    equal = backtest_portfolio(systems=systems, equity=1_000)
    assert [entry["weight"] for entry in equal["systems"]] == [0.5, 0.5]
    assert [entry["allocationCap"] for entry in equal["systems"]] == [400, 200]

    weighted = backtest_portfolio(systems=systems, equity=1_000, allocation="weight")
    assert [entry["symbol"] for entry in weighted["systems"]] == ["FIRST", "SECOND"]
    assert [entry["weight"] for entry in weighted["systems"]] == pytest.approx([2 / 3, 1 / 3])
    assert [entry["equity"] for entry in weighted["systems"]] == pytest.approx(
        [2_000 / 3, 1_000 / 3]
    )
    assert [entry["allocationCap"] for entry in weighted["systems"]] == [400, 200]


def test_weight_normalization_is_stable_for_extreme_finite_weights() -> None:
    result = backtest_portfolio(
        systems=[
            {"symbol": "A", "candles": _flat_bars(), "signal": _never, "weight": 1e308},
            {"symbol": "B", "candles": _flat_bars(), "signal": _never, "weight": 1e308},
        ],
        allocation="weight",
    )
    assert [entry["weight"] for entry in result["systems"]] == [0.5, 0.5]


def test_portfolio_risk_day_uses_actual_new_york_midnight() -> None:
    assert day_key_et(1_735_779_600_000) == "2025-01-01"  # 2025-01-02 01:00Z
    assert day_key_et(1_735_797_600_000) == "2025-01-02"  # 2025-01-02 06:00Z


def test_timestamp_merge_steps_only_active_runners_and_emits_one_portfolio_point() -> None:
    first = [_bar(BASE_TIME, 100), _bar(BASE_TIME + 600_000, 100)]
    second = [_bar(BASE_TIME + 300_000, 200), _bar(BASE_TIME + 900_000, 200)]

    result = backtest_portfolio(
        systems=[
            {"symbol": "A", "candles": first, "signal": _never, "warmup_bars": 1},
            {"symbol": "B", "candles": second, "signal": _never, "warmup_bars": 1},
        ]
    )

    assert [point["time"] for point in result["eqSeries"]] == [
        BASE_TIME,
        BASE_TIME + 600_000,
        BASE_TIME + 900_000,
    ]
    assert all(point["availableCapital"] == 10_000 for point in result["eqSeries"])


def test_later_system_uses_live_shared_equity_and_remaining_capital() -> None:
    candles = [
        _bar(BASE_TIME, 100),
        _bar(BASE_TIME + 300_000, 100),
        _bar(BASE_TIME + 600_000, 90, high=100, low=89),
        _bar(BASE_TIME + 900_000, 100),
        _bar(BASE_TIME + 1_200_000, 110, high=111, low=100),
    ]
    result = backtest_portfolio(
        equity=1_000,
        collect_replay=False,
        systems=[
            {
                "symbol": "AAA",
                "candles": candles,
                "warmup_bars": 1,
                "flatten_at_close": False,
                "scale_out_at_r": 0,
                "max_leverage": 1,
                "qty_step": 0.01,
                "signal": _index_signal(
                    1, {"side": "buy", "entry": 100, "stop": 90, "rr": 1, "qty": 5}
                ),
            },
            {
                "symbol": "BBB",
                "candles": candles,
                "warmup_bars": 1,
                "flatten_at_close": False,
                "scale_out_at_r": 0,
                "max_leverage": 1,
                "qty_step": 0.01,
                "signal": _index_signal(
                    3, {"side": "buy", "entry": 100, "stop": 90, "rr": 1, "qty": 5}
                ),
            },
        ],
    )

    assert [(trade["symbol"], trade["size"]) for trade in result["positions"]] == [
        ("AAA", 5),
        ("BBB", pytest.approx(4.74, abs=0.01)),
    ]
    assert any(point["lockedCapital"] > 0 for point in result["eqSeries"])
    assert all("availableCapital" in point for point in result["eqSeries"])


def test_portfolio_cap_is_applied_to_pyramid_additions() -> None:
    candles = [
        _bar(BASE_TIME, 100),
        _bar(BASE_TIME + 300_000, 100),
        _bar(BASE_TIME + 600_000, 101, high=101, low=100),
    ]
    result = backtest_portfolio(
        equity=1_000,
        collect_replay=False,
        systems=[
            {
                "symbol": "PYRAMID",
                "candles": candles,
                "warmup_bars": 1,
                "max_leverage": 1,
                "qty_step": 0.01,
                "scale_out_at_r": 0,
                "pyramiding": {
                    "enabled": True,
                    "add_at_r": 0.1,
                    "add_frac": 1,
                    "max_adds": 1,
                    "only_after_break_even": False,
                },
                "signal": _index_signal(
                    1, {"side": "long", "entry": 100, "stop": 90, "takeProfit": 200, "qty": 5}
                ),
            }
        ],
    )

    position = result["openPositions"][0]
    assert position["size"] == pytest.approx(9.95)
    last = result["eqSeries"][-1]
    assert last["lockedCapital"] <= result["systems"][0]["allocationCap"]


def test_daily_loss_halt_cancels_all_pending_and_forces_all_open_positions() -> None:
    candles = [
        _bar(BASE_TIME, 100),
        _bar(BASE_TIME + 300_000, 100),
        _bar(BASE_TIME + 600_000, 89, high=100, low=89),
        _bar(BASE_TIME + 900_000, 100),
        _bar(BASE_TIME + 1_200_000, 110, high=110, low=100),
    ]
    result = backtest_portfolio(
        equity=1_000,
        max_daily_loss_pct=5,
        collect_replay=True,
        systems=[
            {
                "symbol": "LOSER",
                "candles": candles,
                "warmup_bars": 1,
                "max_leverage": 1,
                "qty_step": 0.01,
                "scale_out_at_r": 0,
                "signal": _index_signal(
                    1, {"side": "long", "entry": 100, "stop": 90, "rr": 100, "qty": 5}
                ),
            },
            {
                "symbol": "BLOCKED",
                "candles": candles,
                "warmup_bars": 1,
                "max_leverage": 1,
                "qty_step": 0.01,
                "scale_out_at_r": 0,
                "signal": _index_signal(
                    3, {"side": "long", "entry": 100, "stop": 90, "rr": 1, "qty": 5}
                ),
            },
        ],
    )

    assert [(trade["symbol"], trade["exit"]["reason"]) for trade in result["positions"]] == [
        ("LOSER", "SL")
    ]
    assert result["openPositions"] == []
    assert all(event["symbol"] != "BLOCKED" for event in result["replay"]["events"])


def test_marked_daily_loss_forces_open_positions_at_their_latest_marks() -> None:
    candles = [
        _bar(BASE_TIME, 100),
        _bar(BASE_TIME + 300_000, 100),
        _bar(BASE_TIME + 600_000, 89, high=100, low=89),
    ]
    result = backtest_portfolio(
        equity=1_000,
        max_daily_loss_pct=5,
        collect_replay=False,
        systems=[
            {
                "symbol": "MARKED",
                "candles": candles,
                "warmup_bars": 1,
                "max_leverage": 1,
                "qty_step": 0.01,
                "scale_out_at_r": 0,
                "slippage_bps": 0,
                "fee_bps": 0,
                "signal": _index_signal(
                    1,
                    {
                        "side": "long",
                        "entry": 100,
                        "stop": 1,
                        "takeProfit": 1_000,
                        "qty": 5,
                    },
                ),
            }
        ],
    )

    assert len(result["positions"]) == 1
    assert result["positions"][0]["exit"] == {
        "price": 89,
        "time": BASE_TIME + 600_000,
        "reason": "PORTFOLIO_DAILY_LOSS",
        "pnl": -55,
        "financing": 0,
        "exitATR": None,
    }
    assert result["eqSeries"][-1]["lockedCapital"] == 0


def test_daily_portfolio_halt_resets_on_the_next_et_day() -> None:
    next_day = BASE_TIME + 86_400_000
    candles = [
        _bar(BASE_TIME, 100),
        _bar(BASE_TIME + 300_000, 100),
        _bar(BASE_TIME + 600_000, 89, high=100, low=89),
        _bar(next_day, 100),
        _bar(next_day + 300_000, 102, high=102, low=100),
    ]
    result = backtest_portfolio(
        equity=1_000,
        max_daily_loss_pct=5,
        collect_replay=False,
        systems=[
            {
                "symbol": "DAY-ONE",
                "candles": candles,
                "warmup_bars": 1,
                "scale_out_at_r": 0,
                "signal": _index_signal(
                    1, {"side": "long", "entry": 100, "stop": 90, "rr": 100, "qty": 5}
                ),
            },
            {
                "symbol": "DAY-TWO",
                "candles": candles,
                "warmup_bars": 1,
                "scale_out_at_r": 0,
                "slippage_bps": 0,
                "fee_bps": 0,
                "signal": _index_signal(
                    3,
                    {"side": "long", "entry": 100, "stop": 99, "takeProfit": 102, "qty": 1},
                ),
            },
        ],
    )

    assert [trade["symbol"] for trade in result["positions"]] == ["DAY-ONE", "DAY-TWO"]


def test_seeded_shuffle_changes_competition_but_keeps_output_ties_in_system_order() -> None:
    candles = [
        _bar(BASE_TIME, 100),
        _bar(BASE_TIME + 60_000, 100, high=101, low=89),
        _bar(BASE_TIME + 120_000, 100),
    ]
    systems = [
        {
            "symbol": symbol,
            "candles": candles,
            "warmup_bars": 1,
            "max_leverage": 1,
            "qty_step": 0.01,
            "scale_out_at_r": 0,
            "signal": _index_signal(
                1, {"side": "long", "entry": 90, "stop": 80, "takeProfit": 200, "qty": 100}
            ),
        }
        for symbol in ("AAA", "BBB")
    ]

    first = backtest_portfolio(
        systems=systems,
        equity=1_000,
        processing_order="shuffle",
        shuffle_seed=0,
        collect_replay=True,
    )
    second = backtest_portfolio(
        systems=systems,
        equity=1_000,
        processing_order="shuffle",
        shuffle_seed=0,
        collect_replay=True,
    )

    assert first.to_dict() == second.to_dict()
    assert [entry["symbol"] for entry in first["systems"]] == ["AAA", "BBB"]
    assert [event["symbol"] for event in first["replay"]["events"][:2]] == ["AAA", "BBB"]
    sizes = [entry["result"]["openPositions"][0]["size"] for entry in first["systems"]]
    assert sizes[0] != sizes[1]
    assert sizes[1] < sizes[0]  # seed 0 processes BBB first, leaving AAA the larger live cap


def test_collect_flags_do_not_change_realized_final_equity() -> None:
    candles = [
        _bar(BASE_TIME, 100),
        _bar(BASE_TIME + 300_000, 100),
        _bar(BASE_TIME + 600_000, 102, high=102, low=100),
    ]
    system = {
        "symbol": "ONLY",
        "candles": candles,
        "warmup_bars": 1,
        "scale_out_at_r": 0,
        "slippage_bps": 0,
        "fee_bps": 0,
        "signal": _index_signal(
            1, {"side": "long", "entry": 100, "stop": 99, "takeProfit": 102, "qty": 1}
        ),
    }
    collected = backtest_portfolio(systems=[system], equity=1_000)
    omitted = backtest_portfolio(
        systems=[system], equity=1_000, collect_eq_series=False, collect_replay=False
    )

    assert omitted["eqSeries"] == []
    assert omitted["replay"] == {"frames": [], "events": []}
    assert omitted["metrics"]["finalEquity"] == collected["metrics"]["finalEquity"] == 1_002


def test_open_positions_and_equal_timestamp_exits_remain_in_original_system_order() -> None:
    candles = [
        _bar(BASE_TIME, 100),
        _bar(BASE_TIME + 300_000, 100),
        _bar(BASE_TIME + 600_000, 102, high=102, low=100),
    ]
    result = backtest_portfolio(
        systems=[
            {
                "symbol": symbol,
                "candles": candles,
                "warmup_bars": 1,
                "scale_out_at_r": 0,
                "signal": _index_signal(
                    1,
                    {"side": "long", "entry": 100, "stop": 99, "takeProfit": 102, "qty": 1},
                ),
            }
            for symbol in ("FIRST", "SECOND")
        ],
        processing_order="shuffle",
        shuffle_seed=0,
        collect_replay=True,
    )

    assert [trade["symbol"] for trade in result["trades"]] == ["FIRST", "SECOND"]
    assert [trade["symbol"] for trade in result["positions"]] == ["FIRST", "SECOND"]
    assert [event["symbol"] for event in result["replay"]["events"]] == [
        "FIRST",
        "SECOND",
        "FIRST",
        "SECOND",
    ]
