"""Asynchronous sibling of the deterministic bar backtester."""

from __future__ import annotations

import math
from collections.abc import Mapping

from tradelab.errors import ValidationError
from tradelab.models import BacktestResult

from .async_signal import with_budget
from .backtest import BarSystemRunner


async def backtest_async(
    options: Mapping[str, object] | None = None, /, **kwargs: object
) -> BacktestResult:
    """Run a backtest whose signal may be synchronous or awaitable."""
    raw = dict(options or {})
    raw.update(kwargs)
    signal = raw.get("signal")
    if not callable(signal):
        raise ValidationError("backtest_async requires a signal callable")
    budget_raw = raw.get("signal_budget_ms", raw.get("signalBudgetMs", 0))
    if isinstance(budget_raw, bool):
        raise ValidationError("signal_budget_ms must be finite")
    try:
        budget_ms = float(budget_raw if budget_raw is not None else 0)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as error:
        raise ValidationError("signal_budget_ms must be finite") from error
    if not math.isfinite(budget_ms):
        raise ValidationError("signal_budget_ms must be finite")

    async def budgeted_signal(context: dict[str, object]) -> object:
        return await with_budget(signal(context), budget_ms)

    raw["signal"] = budgeted_signal
    runner = BarSystemRunner(raw)
    while runner.has_next():
        await runner.step_async(signal_equity=runner.get_marked_equity())
    return runner.build_result()


__all__ = ["backtest_async"]
