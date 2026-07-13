"""Lexicographic index-combination generation."""

from __future__ import annotations

from tradelab.errors import ValidationError


def combinations(n: int, k: int) -> list[list[int]]:
    """Return all lexicographic ``k``-sized index combinations from ``range(n)``."""
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise ValidationError("n must be a non-negative integer", context={"n": n})
    if isinstance(k, bool) or not isinstance(k, int) or k < 0 or k > n:
        raise ValidationError("k must be an integer between zero and n", context={"n": n, "k": k})

    result: list[list[int]] = []
    current: list[int] = []

    def recurse(start: int) -> None:
        if len(current) == k:
            result.append(current.copy())
            return
        for index in range(start, n):
            current.append(index)
            recurse(index + 1)
            current.pop()

    recurse(0)
    return result
