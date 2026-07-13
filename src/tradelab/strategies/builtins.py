"""Built-in strategy factories mirrored from the immutable JavaScript package."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from tradelab.errors import ValidationError
from tradelab.ta import donchian, ema, rsi

SignalFunction = Callable[[dict[str, object]], dict[str, object] | None]
SignalFactory = Callable[[Mapping[str, object] | None], SignalFunction]


def _params(value: Mapping[str, object] | None) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _option(values: Mapping[str, object], camel: str, snake: str, default: object) -> object:
    if camel in values and values[camel] is not None:
        return values[camel]
    if snake in values and values[snake] is not None:
        return values[snake]
    return default


def _float(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValidationError(f"{name} must be finite")
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as error:
        raise ValidationError(f"{name} must be finite") from error
    if not math.isfinite(result):
        raise ValidationError(f"{name} must be finite")
    return result


def _finite_result(value: float, name: str) -> float:
    if not math.isfinite(value):
        raise ValidationError(f"{name} must remain finite")
    return value


def _required_number(values: Mapping[str, object], key: str, name: str) -> float:
    if key not in values:
        raise ValidationError(f"{name} is required")
    return _float(values[key], name)


def _int(value: object, name: str) -> int:
    result = _float(value, name)
    if not result.is_integer() or result <= 0:
        raise ValidationError(f"{name} must be a positive integer")
    return int(result)


def _candles(context: Mapping[str, object]) -> Sequence[Mapping[str, object]]:
    value = context.get("candles")
    return value if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else []


def _bar(context: Mapping[str, object]) -> Mapping[str, object]:
    value = context.get("bar")
    return value if isinstance(value, Mapping) else {}


def ema_cross_factory(values: Mapping[str, object] | None = None) -> SignalFunction:
    options = _params(values)
    fast = _int(_option(options, "fast", "fast", 10), "fast")
    slow = _int(_option(options, "slow", "slow", 30), "slow")
    reward_risk = _float(_option(options, "rr", "rr", 2), "rr")
    lookback = _int(_option(options, "lookback", "lookback", 15), "lookback")

    def signal(context: dict[str, object]) -> dict[str, object] | None:
        candles = _candles(context)
        bar = _bar(context)
        if len(candles) < slow + 2:
            return None
        closes = [_float(candle["close"], "candle.close") for candle in candles]
        fast_values, slow_values = ema(closes, fast), ema(closes, slow)
        last = len(closes) - 1
        if fast_values[last - 1] <= slow_values[last - 1] and fast_values[last] > slow_values[last]:
            stop = min(_float(candle["low"], "candle.low") for candle in candles[-lookback:])
            close = _float(bar["close"], "bar.close")
            if stop >= close:
                return None
            return {"side": "long", "entry": close, "stop": stop, "rr": reward_risk}
        return None

    return signal


def rsi_reversion_factory(values: Mapping[str, object] | None = None) -> SignalFunction:
    options = _params(values)
    period = _int(_option(options, "period", "period", 14), "period")
    oversold = _float(_option(options, "oversold", "oversold", 30), "oversold")
    stop_pct = _float(_option(options, "stopPct", "stop_pct", 2), "stop_pct")
    reward_risk = _float(_option(options, "rr", "rr", 1.5), "rr")

    def signal(context: dict[str, object]) -> dict[str, object] | None:
        candles = _candles(context)
        bar = _bar(context)
        if len(candles) < period + 2:
            return None
        value = rsi([_float(candle["close"], "candle.close") for candle in candles], period)[-1]
        if value is None or value > oversold:
            return None
        close = _required_number(bar, "close", "bar.close")
        stop = _finite_result(close * (1 - stop_pct / 100), "signal.stop")
        return {
            "side": "long",
            "entry": close,
            "stop": stop,
            "rr": reward_risk,
        }

    return signal


def donchian_breakout_factory(values: Mapping[str, object] | None = None) -> SignalFunction:
    options = _params(values)
    period = _int(_option(options, "period", "period", 20), "period")
    reward_risk = _float(_option(options, "rr", "rr", 2), "rr")

    def signal(context: dict[str, object]) -> dict[str, object] | None:
        candles = _candles(context)
        bar = _bar(context)
        if len(candles) < period + 2:
            return None
        channel = donchian(candles, period)
        prior_upper = channel["upper"][-2]
        prior_lower = channel["lower"][-2]
        close = _float(bar["close"], "bar.close")
        if prior_upper is not None and close > prior_upper:
            return {"side": "long", "entry": close, "stop": prior_lower, "rr": reward_risk}
        return None

    return signal


def buy_hold_factory(values: Mapping[str, object] | None = None) -> SignalFunction:
    options = _params(values)
    hold_bars = _float(_option(options, "holdBars", "hold_bars", 5), "hold_bars")
    stop_pct = _float(_option(options, "stopPct", "stop_pct", 10), "stop_pct")
    entered = False

    def signal(context: dict[str, object]) -> dict[str, object] | None:
        nonlocal entered
        if entered:
            return None
        close = _required_number(_bar(context), "close", "bar.close")
        stop = _finite_result(close * (1 - stop_pct / 100), "signal.stop")
        result = {
            "side": "long",
            "entry": close,
            "stop": stop,
            "rr": 5,
            "_maxBarsInTrade": hold_bars,
        }
        entered = True
        return result

    return signal


BUILTINS: dict[str, dict[str, Any]] = {
    "ema-cross": {
        "description": "Long when fast EMA crosses above slow EMA; stop at recent swing low.",
        "params": {
            "fast": {"type": "number", "default": 10, "description": "fast EMA period"},
            "slow": {"type": "number", "default": 30, "description": "slow EMA period"},
            "rr": {"type": "number", "default": 2, "description": "reward:risk target"},
            "lookback": {
                "type": "number",
                "default": 15,
                "description": "swing-low lookback for stop",
            },
        },
        "factory": ema_cross_factory,
    },
    "rsi-reversion": {
        "description": "Long when RSI dips below `oversold`; stop a fixed pct below entry.",
        "params": {
            "period": {"type": "number", "default": 14, "description": "RSI period"},
            "oversold": {"type": "number", "default": 30, "description": "RSI entry threshold"},
            "stopPct": {
                "type": "number",
                "default": 2,
                "description": "stop distance in percent",
            },
            "rr": {"type": "number", "default": 1.5, "description": "reward:risk target"},
        },
        "factory": rsi_reversion_factory,
    },
    "donchian-breakout": {
        "description": "Long on a close above the prior Donchian upper channel.",
        "params": {
            "period": {"type": "number", "default": 20, "description": "channel lookback"},
            "rr": {"type": "number", "default": 2, "description": "reward:risk target"},
        },
        "factory": donchian_breakout_factory,
    },
    "buy-hold": {
        "description": "Enter once at the first eligible bar and hold for `holdBars`.",
        "params": {
            "holdBars": {
                "type": "number",
                "default": 5,
                "description": "bars to hold before exit",
            },
            "stopPct": {
                "type": "number",
                "default": 10,
                "description": "protective stop distance in percent",
            },
        },
        "factory": buy_hold_factory,
    },
}
