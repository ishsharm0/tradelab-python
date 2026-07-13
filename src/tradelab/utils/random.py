"""Local deterministic pseudo-random number generators."""

from __future__ import annotations

from collections.abc import Callable

from tradelab.errors import ValidationError

_MASK_32 = 0xFFFFFFFF
_UINT_32_SIZE = 4_294_967_296


def _imul(left: int, right: int) -> int:
    return (left * right) & _MASK_32


def _xmur3(seed: str) -> int:
    value = (1_779_033_703 ^ len(seed)) & _MASK_32
    for character in seed:
        value = _imul(value ^ ord(character), 3_432_918_353)
        value = ((value << 13) | (value >> 19)) & _MASK_32
    value = _imul(value ^ (value >> 16), 2_246_822_507)
    value = _imul(value ^ (value >> 13), 3_266_489_909)
    return (value ^ (value >> 16)) & _MASK_32


def make_rng(seed: object = "tradelab") -> Callable[[], float]:
    """Return a deterministic JavaScript-compatible generator without global state."""
    state = _xmur3(str(seed))

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
