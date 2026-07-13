"""Shared synchronous/asynchronous bar-runner contracts."""

from __future__ import annotations

import asyncio
from typing import cast

import pytest

from tradelab.engine.async_signal import BudgetExceededError
from tradelab.engine.backtest import BarSystemRunner, backtest
from tradelab.engine.backtest_async import backtest_async
from tradelab.errors import StrategyError, ValidationError


def _bars(count: int = 5) -> list[dict[str, float | int]]:
    start = 1_704_205_800_000
    return [
        {
            "time": start + index * 60_000,
            "open": 100 + index,
            "high": 101 + index,
            "low": 99 + index,
            "close": 100 + index,
        }
        for index in range(count)
    ]


@pytest.mark.asyncio
async def test_async_backtest_with_sync_signal_matches_sync_result_exactly() -> None:
    def signal(context: dict[str, object]) -> dict[str, object] | None:
        return (
            {"side": "long", "entry": 101, "stop": 100, "takeProfit": 103, "qty": 1}
            if context["index"] == 1
            else None
        )

    options = {
        "candles": _bars(),
        "warmup_bars": 1,
        "flatten_at_close": False,
        "scale_out_at_r": 0,
        "slippage_bps": 0,
        "signal": signal,
    }

    expected = backtest(options)
    actual = await backtest_async(options)

    assert actual.to_dict() == expected.to_dict()


@pytest.mark.asyncio
async def test_async_backtest_awaits_signal_and_only_budgets_eligible_bars() -> None:
    calls: list[int] = []

    async def signal(context: dict[str, object]) -> dict[str, object] | None:
        index = cast(int, context["index"])
        calls.append(index)
        if index > 1:
            await asyncio.sleep(0.05)
        return (
            {"side": "long", "entry": 101, "stop": 90, "takeProfit": 200, "qty": 1}
            if index == 1
            else None
        )

    result = await backtest_async(
        candles=_bars(),
        warmup_bars=1,
        flatten_at_close=False,
        signal_budget_ms=5,
        signal=signal,
    )

    assert calls == [1]
    assert len(result["openPositions"]) == 1


@pytest.mark.asyncio
async def test_async_budget_error_is_wrapped_with_strategy_context() -> None:
    async def slow(_context: dict[str, object]) -> None:
        await asyncio.sleep(0.05)

    with pytest.raises(StrategyError, match=r"index=1.*symbol=ES.*exceeded its 1ms") as caught:
        await backtest_async(
            candles=_bars(2),
            warmup_bars=1,
            symbol="ES",
            signal_budget_ms=1,
            signal=slow,
        )

    assert isinstance(caught.value.__cause__, BudgetExceededError)


@pytest.mark.asyncio
async def test_runner_step_async_accepts_sync_and_async_signals() -> None:
    sync_runner = BarSystemRunner(candles=_bars(2), warmup_bars=1, signal=lambda _context: None)
    async_runner = BarSystemRunner(
        candles=_bars(2),
        warmup_bars=1,
        signal=lambda _context: asyncio.sleep(0, result=None),
    )

    assert await sync_runner.step_async() is not None
    assert await async_runner.step_async() is not None


@pytest.mark.parametrize(
    "step_options",
    [
        {"signal_equity": 10**10_000},
        {"signal_equity": float("nan")},
        {"can_trade": "false"},
        {"resolve_entry_size": 3},
    ],
)
def test_step_hooks_are_validated_before_runner_state_advances(
    step_options: dict[str, object],
) -> None:
    runner = BarSystemRunner(candles=_bars(2), warmup_bars=1, signal=lambda _context: None)
    original_index = runner.index
    original_history = list(runner.history)

    with pytest.raises(ValidationError):
        runner.step(**step_options)  # type: ignore[arg-type]

    assert runner.index == original_index
    assert runner.history == original_history


def test_option_risk_fraction_and_trigger_match_javascript_validation() -> None:
    runner = BarSystemRunner(
        candles=_bars(2),
        warmup_bars=1,
        riskFraction="0.5",
        riskPct=2,
        signal=lambda _context: None,
    )
    assert runner.risk_pct == 2

    with pytest.raises(ValidationError, match="trigger_mode"):
        BarSystemRunner(
            candles=_bars(2),
            warmup_bars=1,
            trigger_mode="future",
            signal=lambda _context: None,
        )


def test_entry_size_resolver_caps_pyramid_adds_through_shared_step() -> None:
    fill_kinds: list[str] = []

    def resolver(request: dict[str, object]) -> float:
        fill_kinds.append(cast(str, request["fill_kind"]))
        return 4 if request["fill_kind"] == "limit" else 0

    runner = BarSystemRunner(
        candles=[
            {"time": 0, "open": 100, "high": 100, "low": 100, "close": 100},
            {"time": 60_000, "open": 100, "high": 100, "low": 100, "close": 100},
            {"time": 120_000, "open": 101, "high": 103, "low": 101, "close": 102},
        ],
        warmup_bars=1,
        flatten_at_close=False,
        scale_out_at_r=0,
        pyramiding={
            "enabled": True,
            "only_after_break_even": False,
            "add_at_r": 1,
            "add_frac": 0.5,
        },
        signal=lambda context: (
            {"side": "long", "entry": 100, "stop": 99, "takeProfit": 200, "qty": 4}
            if context["index"] == 1
            else None
        ),
    )

    runner.step(resolve_entry_size=resolver)
    runner.step(resolve_entry_size=resolver)

    assert fill_kinds == ["limit", "pyramid"]
    assert runner.open is not None
    assert runner.open["size"] == 4
    assert runner.open["_adds"] == 0


def test_pyramid_resolver_is_only_called_after_add_trigger_is_touched() -> None:
    fill_kinds: list[str] = []

    def resolver(request: dict[str, object]) -> float:
        fill_kinds.append(cast(str, request["fill_kind"]))
        return cast(float, request["desired_size"])

    runner = BarSystemRunner(
        candles=[
            {"time": 0, "open": 100, "high": 100, "low": 100, "close": 100},
            {"time": 60_000, "open": 100, "high": 100, "low": 100, "close": 100},
            {"time": 120_000, "open": 100, "high": 100.5, "low": 100, "close": 100},
        ],
        warmup_bars=1,
        flatten_at_close=False,
        scale_out_at_r=0,
        pyramiding={"enabled": True, "only_after_break_even": False, "add_at_r": 1},
        signal=lambda context: (
            {"side": "long", "entry": 100, "stop": 99, "takeProfit": 200, "qty": 4}
            if context["index"] == 1
            else None
        ),
    )

    runner.step(resolve_entry_size=resolver)
    runner.step(resolve_entry_size=resolver)

    assert fill_kinds == ["limit"]
