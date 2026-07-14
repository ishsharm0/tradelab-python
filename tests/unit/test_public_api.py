"""Release-level checks for documented import surfaces."""

from __future__ import annotations

import importlib
from importlib.metadata import version

import tradelab


def test_distribution_and_runtime_versions_match() -> None:
    assert tradelab.__version__ == version("tradelab-python") == "1.3.1"


def test_documented_namespaces_import_without_optional_dependencies() -> None:
    for name in (
        "tradelab.brokers",
        "tradelab.data",
        "tradelab.engine",
        "tradelab.live",
        "tradelab.mcp",
        "tradelab.metrics",
        "tradelab.reporting",
        "tradelab.research",
        "tradelab.strategies",
        "tradelab.ta",
        "tradelab.utils",
    ):
        assert importlib.import_module(name).__name__ == name


def test_documented_root_exports_are_public() -> None:
    expected = {
        "backtest",
        "backtest_async",
        "backtest_historical",
        "backtest_portfolio",
        "backtest_ticks",
        "build_metrics",
        "create_research_store",
        "export_backtest_artifacts",
        "get_historical_candles",
        "get_strategy",
        "grid",
        "list_strategies",
        "optimize",
        "walk_forward_optimize",
    }
    assert expected <= set(tradelab.__all__)
    assert all(hasattr(tradelab, name) for name in expected)
