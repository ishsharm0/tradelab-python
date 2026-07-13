"""Carry and funding calculations used by every execution path."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from tradelab.errors import ValidationError

MS_PER_YEAR = 365 * 24 * 60 * 60 * 1_000


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValidationError(f"{name} must be finite", context={name: value})
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as error:
        raise ValidationError(f"{name} must be finite", context={name: value}) from error
    if not math.isfinite(number):
        raise ValidationError(f"{name} must be finite", context={name: value})
    return number


def _finite_result(value: float, name: str) -> float:
    if not math.isfinite(value):
        raise ValidationError(f"{name} must remain finite", context={name: value})
    return value


def funding_events(
    from_ms: float | None = None,
    to_ms: float | None = None,
    interval_ms: float | None = None,
    anchor_ms: float = 0,
    **aliases: Any,
) -> int:
    """Count cadence boundaries in the half-open interval ``(from_ms, to_ms]``."""
    from_value = _finite(aliases.get("fromMs", from_ms if from_ms is not None else 0), "from_ms")
    to_value = _finite(aliases.get("toMs", to_ms if to_ms is not None else 0), "to_ms")
    interval_value = _finite(
        aliases.get("intervalMs", interval_ms if interval_ms is not None else 0), "interval_ms"
    )
    anchor_value = _finite(aliases.get("anchorMs", anchor_ms), "anchor_ms")
    if interval_value <= 0 or to_value <= from_value:
        return 0
    first_ratio = _finite_result((from_value - anchor_value) / interval_value, "first boundary")
    last_ratio = _finite_result((to_value - anchor_value) / interval_value, "last boundary")
    first = math.floor(first_ratio) + 1
    last = math.floor(last_ratio)
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
    if side not in {"long", "short"}:
        raise ValidationError("side must be long or short", context={"side": side})
    from_value = _finite(aliases.get("fromMs", from_ms if from_ms is not None else 0), "from_ms")
    to_value = _finite(aliases.get("toMs", to_ms if to_ms is not None else 0), "to_ms")
    if costs is not None and not isinstance(costs, Mapping):
        raise ValidationError("costs must be a mapping", context={"costs": costs})
    model = costs or {}
    amount = abs(_finite(notional, "notional"))
    cost = 0.0
    carry = model.get("carry")
    if carry is not None and not isinstance(carry, Mapping):
        raise ValidationError("costs.carry must be a mapping", context={"carry": carry})
    if isinstance(carry, Mapping):
        annual_bps = _finite(
            carry.get("longAnnualBps" if side == "long" else "shortAnnualBps", 0),
            "annual_bps",
        )
        cost = _finite_result(
            cost + amount * (annual_bps / 10_000) * max(0.0, to_value - from_value) / MS_PER_YEAR,
            "financing cost",
        )
    funding = model.get("funding")
    if funding is not None and not isinstance(funding, Mapping):
        raise ValidationError("costs.funding must be a mapping", context={"funding": funding})
    if isinstance(funding, Mapping):
        interval = funding.get("intervalMs")
        rate = funding.get("rateBps")
        interval_value = _finite(interval, "funding.interval_ms") if interval is not None else 0
        rate_value = _finite(rate, "funding.rate_bps") if rate is not None else 0
        if interval_value > 0 and rate is not None:
            anchor_value = _finite(funding.get("anchorMs", 0), "funding.anchor_ms")
            count = funding_events(from_value, to_value, interval_value, anchor_value)
            cost = _finite_result(
                cost + (1 if side == "long" else -1) * amount * rate_value / 10_000 * count,
                "financing cost",
            )
    return _finite_result(cost, "financing cost")
