"""Carry and funding calculations used by every execution path."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

MS_PER_YEAR = 365 * 24 * 60 * 60 * 1_000


def funding_events(
    from_ms: float | None = None,
    to_ms: float | None = None,
    interval_ms: float | None = None,
    anchor_ms: float = 0,
    **aliases: Any,
) -> int:
    """Count cadence boundaries in the half-open interval ``(from_ms, to_ms]``."""
    from_value = float(aliases.get("fromMs", from_ms if from_ms is not None else 0))
    to_value = float(aliases.get("toMs", to_ms if to_ms is not None else 0))
    interval_value = float(aliases.get("intervalMs", interval_ms if interval_ms is not None else 0))
    anchor_value = float(aliases.get("anchorMs", anchor_ms))
    if interval_value <= 0 or to_value <= from_value:
        return 0
    first = math.floor((from_value - anchor_value) / interval_value) + 1
    last = math.floor((to_value - anchor_value) / interval_value)
    return max(0, last - first + 1)


def financing_cost(
    side: str,
    notional: float,
    from_ms: float | None = None,
    to_ms: float | None = None,
    costs: Mapping[str, object] | None = None,
    **aliases: Any,
) -> float:
    """Return cost to subtract from PnL (negative means a credit)."""
    from_value = float(aliases.get("fromMs", from_ms if from_ms is not None else 0))
    to_value = float(aliases.get("toMs", to_ms if to_ms is not None else 0))
    model = costs or {}
    amount = abs(float(notional))
    cost = 0.0
    carry = model.get("carry")
    if isinstance(carry, Mapping):
        annual_bps = carry.get("longAnnualBps" if side == "long" else "shortAnnualBps", 0)
        if isinstance(annual_bps, (int, float)) and not isinstance(annual_bps, bool):
            cost += (
                amount
                * (float(annual_bps) / 10_000)
                * max(0.0, to_value - from_value)
                / MS_PER_YEAR
            )
    funding = model.get("funding")
    if isinstance(funding, Mapping):
        interval = funding.get("intervalMs")
        rate = funding.get("rateBps")
        if (
            isinstance(interval, (int, float))
            and not isinstance(interval, bool)
            and isinstance(rate, (int, float))
            and not isinstance(rate, bool)
            and interval > 0
            and math.isfinite(float(rate))
        ):
            count = funding_events(
                from_value, to_value, float(interval), float(funding.get("anchorMs", 0))
            )
            cost += (1 if side == "long" else -1) * amount * float(rate) / 10_000 * count
    return cost
