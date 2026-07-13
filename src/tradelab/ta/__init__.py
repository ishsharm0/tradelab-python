"""Technical-analysis indicators with array-aligned outputs."""

from tradelab.ta.channels import bollinger, donchian, keltner
from tradelab.ta.oscillators import macd, rsi, stochastic
from tradelab.ta.trend import supertrend, vwap
from tradelab.utils.indicators import (
    atr,
    detect_fvg,
    ema,
    last_swing,
    structure_state,
    swing_high,
    swing_low,
)

__all__ = [
    "atr",
    "bollinger",
    "detect_fvg",
    "donchian",
    "ema",
    "keltner",
    "last_swing",
    "macd",
    "rsi",
    "stochastic",
    "structure_state",
    "supertrend",
    "swing_high",
    "swing_low",
    "vwap",
]
