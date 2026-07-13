"""Built-in strategy and registry parity contracts."""

from __future__ import annotations

import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import pytest

from tradelab.errors import ValidationError
from tradelab.strategies import get_strategy, list_strategies, register_strategy


def _bars(closes: list[float]) -> list[dict[str, float]]:
    return [
        {"time": index, "open": close, "high": close + 1, "low": close - 1, "close": close}
        for index, close in enumerate(closes)
    ]


def _context(bars: list[dict[str, float]]) -> dict[str, object]:
    return {"candles": bars, "bar": bars[-1], "index": len(bars) - 1}


def test_builtin_metadata_is_exact_ordered_and_defensive() -> None:
    metadata = list_strategies()[:4]
    assert [item["name"] for item in metadata] == [
        "ema-cross",
        "rsi-reversion",
        "donchian-breakout",
        "buy-hold",
    ]
    assert metadata[0]["description"] == (
        "Long when fast EMA crosses above slow EMA; stop at recent swing low."
    )
    assert metadata[1]["params"]["stopPct"]["default"] == 2  # type: ignore[index]
    metadata[0]["params"]["fast"]["default"] = 999  # type: ignore[index]
    assert list_strategies()[0]["params"]["fast"]["default"] == 10  # type: ignore[index]


def test_ema_cross_positive_case_and_warmup() -> None:
    signal = get_strategy("ema-cross")({"fast": 2, "slow": 4, "lookback": 3})
    assert signal(_context(_bars([8, 8, 8]))) is None
    assert signal(_context(_bars([8, 8, 8, 8, 8, 8, 8, 12]))) == {
        "side": "long",
        "entry": 12,
        "stop": 7,
        "rr": 2.0,
    }


def test_rsi_equality_and_donchian_strict_breakout() -> None:
    falling = _bars([10, 9, 8, 7])
    rsi_signal = get_strategy("rsi-reversion")({"period": 2, "oversold": 0, "stopPct": 10, "rr": 2})
    assert rsi_signal(_context(falling)) == {
        "side": "long",
        "entry": 7,
        "stop": pytest.approx(6.3),
        "rr": 2.0,
    }
    equal = _bars([10, 10, 10, 11])
    breakout = get_strategy("donchian-breakout")({"period": 2})
    assert breakout(_context(equal)) is None
    broken = _bars([10, 10, 10, 12])
    assert breakout(_context(broken)) == {"side": "long", "entry": 12, "stop": 9, "rr": 2.0}


def test_buy_hold_is_stateful_per_factory_and_supports_python_aliases() -> None:
    first = get_strategy("buy-hold")({"hold_bars": 3, "stop_pct": 10})
    second = get_strategy("buy-hold")({"holdBars": 4, "stopPct": 20})
    context = _context(_bars([100]))
    assert first(context) == {
        "side": "long",
        "entry": 100,
        "stop": 90,
        "rr": 5,
        "_maxBarsInTrade": 3.0,
    }
    assert first(context) is None
    assert second(context)["_maxBarsInTrade"] == 4  # type: ignore[index]


def test_strategy_canonical_camel_params_win_and_invalid_call_does_not_consume_state() -> None:
    signal = get_strategy("buy-hold")(
        {"holdBars": 9, "hold_bars": 3, "stopPct": 20, "stop_pct": 10}
    )
    with pytest.raises(ValidationError, match=r"bar\.close"):
        signal({})
    result = signal(_context(_bars([100])))
    assert result is not None
    assert result["_maxBarsInTrade"] == 9
    assert result["stop"] == 80


def test_builtins_reject_nonfinite_derived_stops() -> None:
    buy_hold = get_strategy("buy-hold")({"stopPct": -1e308})
    with pytest.raises(ValidationError, match="remain finite"):
        buy_hold(_context(_bars([1e308])))
    rsi_signal = get_strategy("rsi-reversion")({"period": 2, "oversold": 100, "stopPct": -1e308})
    with pytest.raises(ValidationError, match="remain finite"):
        rsi_signal(_context(_bars([1e308, 1e308, 1e308, 1e308])))


def test_registry_custom_replace_errors_and_concurrency() -> None:
    name = "unit-custom-registry"

    def factory(_params: object = None) -> Callable[[dict[str, object]], object]:
        return lambda _context: None

    register_strategy(name, {"description": "first", "params": {}, "factory": factory})
    before = [item["name"] for item in list_strategies()]
    register_strategy(name, {"description": "second", "params": {}, "factory": factory})
    assert [item["name"] for item in list_strategies()] == before
    assert get_strategy(name) is factory

    with pytest.raises(ValidationError, match="requires a factory"):
        register_strategy(name, {"factory": None})
    assert get_strategy(name) is factory
    with pytest.raises(ValidationError, match=r'Unknown strategy "NOPE".*ema-cross'):
        get_strategy("NOPE")

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: get_strategy(name), range(20)))
    assert all(result is factory for result in results)


def test_registry_is_exported_from_package_root() -> None:
    import tradelab

    assert tradelab.get_strategy is get_strategy
    assert tradelab.list_strategies is list_strategies
    assert tradelab.register_strategy is register_strategy


def test_registry_rejects_bad_names_and_poisoned_metadata_atomically() -> None:
    def factory(_params: object = None) -> Callable[[dict[str, object]], object]:
        return lambda _context: None

    before = list_strategies()
    bad_names: list[object] = ["", "   ", 1, []]
    for name in bad_names:
        with pytest.raises(ValidationError, match="strategy name"):
            register_strategy(name, {"factory": factory})  # type: ignore[arg-type]
    cyclic: dict[str, object] = {}
    cyclic["self"] = cyclic
    for params in (cyclic, {"bad": float("nan")}, {"bad": {1, 2}}):
        with pytest.raises(ValidationError, match="metadata"):
            register_strategy("poisoned", {"factory": factory, "params": params})
    assert list_strategies() == before
    json.dumps(list_strategies(), allow_nan=False)
