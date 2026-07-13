"""Per-bar async budgets and a safe model-backed signal adapter."""

from __future__ import annotations

import asyncio
import inspect
import math
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, TypeVar, overload

T = TypeVar("T")


class BudgetExceededError(TimeoutError):
    """Raised when one signal decision exceeds its configured wall-time budget."""

    def __init__(self, budget_ms: float) -> None:
        self.budget_ms = budget_ms
        super().__init__(f"signal() exceeded its {budget_ms:g}ms per-bar budget")


@overload
async def with_budget(value: Awaitable[T], budget_ms: float | int | None = 0) -> T: ...


@overload
async def with_budget(value: T, budget_ms: float | int | None = 0) -> T: ...


async def with_budget(value: Any, budget_ms: float | int | None = 0) -> Any:
    """Resolve *value*, enforcing a per-bar timeout when the budget is positive."""
    if isinstance(budget_ms, bool):
        raise TypeError("budget_ms must be a finite number")
    try:
        budget = float(budget_ms or 0)
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError("budget_ms must be a finite number") from error
    if not math.isfinite(budget):
        raise ValueError("budget_ms must be finite")

    async def resolve() -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    if budget <= 0:
        return await resolve()
    try:
        async with asyncio.timeout(budget / 1_000):
            return await resolve()
    except TimeoutError as error:
        raise BudgetExceededError(budget) from error


class _NoLookahead(Sequence[Mapping[str, object]]):
    def __init__(self, candles: Sequence[Mapping[str, object]], index: int) -> None:
        self._candles = candles
        self._index = index

    @overload
    def __getitem__(self, index: int) -> Mapping[str, object]: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[Mapping[str, object]]: ...

    def __getitem__(
        self, index: int | slice
    ) -> Mapping[str, object] | Sequence[Mapping[str, object]]:
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self._candles))
            selected = list(range(start, stop, step))
            if selected and max(selected) > self._index:
                raise RuntimeError(
                    f"LlmSignal: lookahead access beyond current index {self._index}"
                )
            return self._candles[index]
        normalized = index if index >= 0 else len(self._candles) + index
        if normalized > self._index:
            raise RuntimeError(
                f"LlmSignal: lookahead access to candles[{index}] (current index {self._index})"
            )
        return self._candles[index]

    def __len__(self) -> int:
        return len(self._candles)


class LlmSignal:
    """Cache, budget, audit, and no-lookahead wrapper for async decisions."""

    def __init__(
        self,
        *,
        resolve: Callable[[dict[str, object]], object | Awaitable[object]],
        budget_ms: float = 0,
        on_error: str = "skip",
        clock_ms: Callable[[], float] | None = None,
    ) -> None:
        if not callable(resolve):
            raise TypeError("LlmSignal requires a resolve(context) function")
        if on_error not in {"skip", "throw"}:
            raise ValueError('on_error must be "skip" or "throw"')
        self.resolve = resolve
        self.budget_ms = budget_ms
        self.on_error = on_error
        self.log: list[dict[str, object]] = []
        self._cache: dict[object, object | None] = {}
        self._clock_ms = clock_ms or (lambda: time.perf_counter() * 1_000)

    async def signal(self, context: dict[str, object]) -> object | None:
        raw_bar = context.get("bar")
        bar = raw_bar if isinstance(raw_bar, Mapping) else {}
        key = bar.get("time") if bar.get("time") is not None else context.get("index")
        if key in self._cache:
            return self._cache[key]
        candles = context.get("candles")
        index = context.get("index")
        if not isinstance(candles, Sequence) or isinstance(candles, (str, bytes)):
            raise TypeError("LlmSignal context.candles must be a sequence")
        if not isinstance(index, int) or isinstance(index, bool):
            raise TypeError("LlmSignal context.index must be an integer")
        safe_context = dict(context)
        safe_context["candles"] = _NoLookahead(candles, index)
        started = self._clock_ms()
        try:
            result = await with_budget(self.resolve(safe_context), self.budget_ms)
            cached = result if result is not None else None
            self._cache[key] = cached
            self.log.append(
                {
                    "index": index,
                    "time": bar.get("time"),
                    "close": bar.get("close"),
                    "latencyMs": max(0.0, self._clock_ms() - started),
                    "result": cached,
                }
            )
            return cached
        except Exception as error:
            self.log.append(
                {
                    "index": index,
                    "time": bar.get("time"),
                    "close": bar.get("close"),
                    "latencyMs": max(0.0, self._clock_ms() - started),
                    "error": str(error),
                }
            )
            self._cache[key] = None
            if self.on_error == "throw":
                raise
            return None


__all__ = ["BudgetExceededError", "LlmSignal", "with_budget"]
