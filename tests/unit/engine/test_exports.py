"""Advanced-engine public export contracts."""

from __future__ import annotations

import tradelab
import tradelab.engine as engine


def test_advanced_engines_are_exported_from_module_and_root() -> None:
    names = (
        "BudgetExceededError",
        "LlmSignal",
        "backtest_async",
        "backtest_portfolio",
        "backtest_ticks",
        "grid",
        "optimize",
        "walk_forward_optimize",
        "with_budget",
    )
    for name in names:
        assert getattr(tradelab, name) is getattr(engine, name)
