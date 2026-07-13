"""Local deterministic pseudo-random number generators."""

from __future__ import annotations

import math
from collections.abc import Callable

from tradelab.errors import ValidationError

_MASK_32 = 0xFFFFFFFF
_UINT_32_SIZE = 4_294_967_296


def _imul(left: int, right: int) -> int:
    return (left * right) & _MASK_32


def _xmur3(seed: str) -> int:
    utf16 = seed.encode("utf-16-le", "surrogatepass")
    value = (1_779_033_703 ^ (len(utf16) // 2)) & _MASK_32
    for index in range(0, len(utf16), 2):
        code_unit = utf16[index] | (utf16[index + 1] << 8)
        value = _imul(value ^ code_unit, 3_432_918_353)
        value = ((value << 13) | (value >> 19)) & _MASK_32
    value = _imul(value ^ (value >> 16), 2_246_822_507)
    value = _imul(value ^ (value >> 13), 3_266_489_909)
    return (value ^ (value >> 16)) & _MASK_32


def _javascript_seed_string(seed: object) -> str:
    """Match JavaScript ``String`` for supported primitive seed values."""
    if isinstance(seed, str):
        return seed
    if seed is None:
        return "null"
    if isinstance(seed, bool):
        return "true" if seed else "false"
    if isinstance(seed, int):
        try:
            value = float(seed)
        except OverflowError:
            value = math.inf if seed > 0 else -math.inf
        return _javascript_number_string(value)
    if isinstance(seed, float):
        return _javascript_number_string(seed)
    raise ValidationError("seed must be a supported JavaScript primitive", context={"seed": seed})


def _javascript_number_string(value: float) -> str:
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "Infinity" if value > 0 else "-Infinity"
    if value == 0:
        return "0"

    representation = repr(value)
    if "e" not in representation:
        return representation.removesuffix(".0")

    mantissa, exponent_text = representation.split("e")
    exponent = int(exponent_text)
    if -6 <= exponent < 21:
        return _expand_exponential(mantissa, exponent)
    return f"{mantissa.removesuffix('.0')}e{exponent:+d}"


def _expand_exponential(mantissa: str, exponent: int) -> str:
    sign = ""
    if mantissa.startswith("-"):
        sign, mantissa = "-", mantissa[1:]
    integer, _, fraction = mantissa.partition(".")
    digits = integer + fraction
    decimal_index = len(integer) + exponent
    if decimal_index <= 0:
        return f"{sign}0.{'0' * -decimal_index}{digits}"
    if decimal_index >= len(digits):
        return f"{sign}{digits}{'0' * (decimal_index - len(digits))}"
    return f"{sign}{digits[:decimal_index]}.{digits[decimal_index:]}"


def make_rng(seed: object = "tradelab") -> Callable[[], float]:
    """Return a deterministic JavaScript-compatible generator without global state."""
    state = _xmur3(_javascript_seed_string(seed))

    def rng() -> float:
        nonlocal state
        state = (state + 0x6D2B79F5) & _MASK_32
        value = _imul(state ^ (state >> 15), state | 1)
        value ^= (value + _imul(value ^ (value >> 7), value | 61)) & _MASK_32
        return ((value ^ (value >> 14)) & _MASK_32) / _UINT_32_SIZE

    return rng


def rand_int(rng: Callable[[], float], max_exclusive: int) -> int:
    """Draw a bounded integer from a caller-owned deterministic generator."""
    if isinstance(max_exclusive, bool) or not isinstance(max_exclusive, int) or max_exclusive <= 0:
        raise ValidationError(
            "max_exclusive must be a positive integer", context={"max_exclusive": max_exclusive}
        )
    value = rng()
    if not 0 <= value < 1:
        raise ValidationError("rng must return a value in [0, 1)", context={"value": value})
    return int(value * max_exclusive)
