"""Async signal budgets and model-backed signal adapter contracts."""

from __future__ import annotations

import asyncio

import pytest

from tradelab.engine.async_signal import BudgetExceededError, LlmSignal, with_budget


@pytest.mark.asyncio
async def test_with_budget_accepts_values_and_disables_nonpositive_timeout() -> None:
    assert await with_budget(7, 1) == 7
    assert await with_budget(asyncio.sleep(0, result=8), 0) == 8


@pytest.mark.asyncio
async def test_with_budget_raises_typed_error_and_cancels_work() -> None:
    cancelled = asyncio.Event()

    async def slow() -> None:
        try:
            await asyncio.sleep(1)
        finally:
            cancelled.set()

    with pytest.raises(BudgetExceededError, match=r"exceeded its 1ms") as caught:
        await with_budget(slow(), 1)
    assert caught.value.budget_ms == 1
    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_llm_signal_caches_by_bar_time_and_blocks_lookahead() -> None:
    calls = 0

    async def resolve(context: dict[str, object]) -> dict[str, object]:
        nonlocal calls
        calls += 1
        candles = context["candles"]
        assert candles[1]["close"] == 101  # type: ignore[index]
        with pytest.raises(RuntimeError, match="lookahead"):
            _ = candles[2]  # type: ignore[index]
        return {"side": "long", "stop": 99}

    adapter = LlmSignal(resolve=resolve)
    context = {
        "candles": [{"close": 100}, {"close": 101}, {"close": 102}],
        "index": 1,
        "bar": {"time": 123, "close": 101},
    }

    first = await adapter.signal(context)
    second = await adapter.signal(context)
    assert first == second == {"side": "long", "stop": 99}
    assert calls == 1
    assert adapter.log[0]["index"] == 1


@pytest.mark.asyncio
async def test_llm_signal_throw_mode_throws_once_then_returns_cached_none() -> None:
    calls = 0

    async def fail(_context: dict[str, object]) -> None:
        nonlocal calls
        calls += 1
        raise RuntimeError("model unavailable")

    adapter = LlmSignal(resolve=fail, on_error="throw")
    context = {"candles": [], "index": 0, "bar": {"time": 1, "close": 2}}

    with pytest.raises(RuntimeError, match="model unavailable"):
        await adapter.signal(context)
    assert await adapter.signal(context) is None
    assert calls == 1
    assert adapter.log[0]["error"] == "model unavailable"


def test_llm_signal_validates_public_options() -> None:
    with pytest.raises(TypeError, match="resolve"):
        LlmSignal(resolve=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="on_error"):
        LlmSignal(resolve=lambda _: None, on_error="explode")
