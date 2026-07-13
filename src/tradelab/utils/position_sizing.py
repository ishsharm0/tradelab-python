"""Risk-based position sizing."""

from __future__ import annotations

import math

from tradelab.errors import ValidationError


def _finite(value: float, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValidationError(f"{name} must be finite", context={name: value})
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValidationError(f"{name} must be finite", context={name: value}) from error
    if not math.isfinite(result):
        raise ValidationError(f"{name} must be finite", context={name: value})
    return result


def calculate_position_size(
    *,
    equity: float,
    entry: float,
    stop: float,
    risk_fraction: float = 0.01,
    qty_step: float = 0.001,
    min_qty: float = 0.001,
    max_leverage: float = 2,
) -> float:
    """Calculate risk-capped quantity, rounded down to the configured quantity step."""
    equity_value = _finite(equity, name="equity")
    entry_value = _finite(entry, name="entry")
    stop_value = _finite(stop, name="stop")
    risk_fraction_value = _finite(risk_fraction, name="risk_fraction")
    qty_step_value = _finite(qty_step, name="qty_step")
    min_qty_value = _finite(min_qty, name="min_qty")
    max_leverage_value = _finite(max_leverage, name="max_leverage")
    if qty_step_value <= 0 or min_qty_value < 0 or max_leverage_value < 0:
        raise ValidationError(
            "qty_step must be positive and quantity/leverage limits cannot be negative",
            context={"qty_step": qty_step, "min_qty": min_qty, "max_leverage": max_leverage},
        )
    if equity_value <= 0:
        return 0.0
    risk_per_unit = abs(entry_value - stop_value)
    if risk_per_unit <= 0:
        return 0.0
    max_risk_dollars = max(0.0, equity_value * risk_fraction_value)
    quantity = max_risk_dollars / risk_per_unit
    leverage_cap_quantity = (equity_value * max_leverage_value) / max(1e-12, abs(entry_value))
    quantity = min(quantity, leverage_cap_quantity)
    quantity = math.floor(quantity / qty_step_value) * qty_step_value
    return quantity if quantity >= min_qty_value else 0.0
