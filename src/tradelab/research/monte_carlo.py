"""Deterministic block-bootstrap Monte Carlo simulation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TypedDict

from tradelab.errors import ValidationError
from tradelab.utils import make_rng, rand_int

Number = int | float


class PercentileBands(TypedDict):
    """Floor-index percentile values for a sorted distribution."""

    p5: float
    p25: float
    p50: float
    p75: float
    p95: float


class PathPercentileBands(TypedDict):
    """Floor-index percentile values for one path step."""

    p5: float
    p50: float
    p95: float


class MonteCarloResult(TypedDict):
    """Summary of deterministic block-bootstrap simulation paths."""

    iterations: int
    block_size: int
    final_equity: PercentileBands
    max_drawdown: PercentileBands
    path_bands: list[PathPercentileBands]
    prob_profit: float


def _finite_number(value: Number, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValidationError(f"{name} must be a finite number", context={name: value})
    return float(value)


def _percentile(sorted_values: list[float], percentile: float) -> float:
    if not sorted_values:
        return 0.0
    index = min(
        len(sorted_values) - 1,
        max(0, math.floor((len(sorted_values) - 1) * percentile)),
    )
    return sorted_values[index]


def _max_drawdown(equity_path: list[float]) -> float:
    peak = equity_path[0]
    maximum = 0.0
    for equity in equity_path:
        if equity > peak:
            peak = equity
        drawdown = (peak - equity) / peak if peak > 0 else 0.0
        if drawdown > maximum:
            maximum = drawdown
    return maximum


def _bands(sorted_values: list[float]) -> PercentileBands:
    return {
        "p5": _percentile(sorted_values, 0.05),
        "p25": _percentile(sorted_values, 0.25),
        "p50": _percentile(sorted_values, 0.5),
        "p75": _percentile(sorted_values, 0.75),
        "p95": _percentile(sorted_values, 0.95),
    }


def monte_carlo(
    *,
    trade_pnls: Sequence[Number],
    equity_start: Number = 10_000,
    iterations: Number = 1000,
    block_size: Number = 1,
    seed: object = "tradelab-mc",
) -> MonteCarloResult:
    """Block-bootstrap PnL paths using TradeLab's deterministic seeded RNG.

    Percentiles intentionally use the JavaScript source's floor-index selection,
    rather than interpolation.
    """
    if not isinstance(trade_pnls, (list, tuple)) or not trade_pnls:
        raise ValidationError(
            "trade_pnls must be a non-empty numeric sequence", context={"trade_pnls": trade_pnls}
        )
    pnls = [_finite_number(value, name="trade_pnls") for value in trade_pnls]
    starting_equity = _finite_number(equity_start, name="equity_start")
    iteration_value = _finite_number(iterations, name="iterations")
    block_value = _finite_number(block_size, name="block_size")
    run_count = math.floor(iteration_value)
    block = math.floor(block_value)
    if run_count < 1:
        raise ValidationError("iterations must be positive", context={"iterations": iterations})
    if block < 1:
        raise ValidationError("block_size must be positive", context={"block_size": block_size})

    rng = make_rng(seed)
    count = len(pnls)
    finals: list[float] = []
    drawdowns: list[float] = []
    path_samples: list[list[float]] = [[] for _ in range(count + 1)]
    for _ in range(run_count):
        path = [starting_equity]
        equity = starting_equity
        filled = 0
        while filled < count:
            start = rand_int(rng, count)
            for offset in range(block):
                if filled >= count:
                    break
                equity += pnls[(start + offset) % count]
                path.append(equity)
                filled += 1
        for step, equity_at_step in enumerate(path):
            path_samples[step].append(equity_at_step)
        finals.append(equity)
        drawdowns.append(_max_drawdown(path))

    sorted_finals = sorted(finals)
    sorted_drawdowns = sorted(drawdowns)
    path_bands: list[PathPercentileBands] = []
    for samples in path_samples:
        sorted_samples = sorted(samples)
        path_bands.append(
            {
                "p5": _percentile(sorted_samples, 0.05),
                "p50": _percentile(sorted_samples, 0.5),
                "p95": _percentile(sorted_samples, 0.95),
            }
        )
    profitable = 0
    for final_equity in finals:
        if final_equity > starting_equity:
            profitable += 1
    return {
        "iterations": run_count,
        "block_size": block,
        "final_equity": _bands(sorted_finals),
        "max_drawdown": _bands(sorted_drawdowns),
        "path_bands": path_bands,
        "prob_profit": profitable / iteration_value,
    }
