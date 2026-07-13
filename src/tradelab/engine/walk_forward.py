"""Rolling and anchored walk-forward optimization."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

from tradelab.errors import ValidationError
from tradelab.metrics import build_metrics
from tradelab.models import BacktestResult

from .backtest import backtest
from .execution import estimate_bar_ms

Signal = Callable[[dict[str, object]], object]
SignalFactory = Callable[[dict[str, Any]], Signal]


def _camel(value: str) -> str:
    parts = value.split("_")
    return parts[0] + "".join(part[:1].upper() + part[1:] for part in parts[1:])


def _js_key(value: object) -> object:
    if isinstance(value, Mapping):
        special = {
            "total_pnl": "totalPnL",
            "avg_pnl": "avgPnL",
            "profit_factor_leg": "profitFactor_leg",
            "profit_factor_pos": "profitFactor_pos",
            "win_rate_leg": "winRate_leg",
            "win_rate_pos": "winRate_pos",
        }
        return {
            special.get(str(key), _camel(str(key))): _js_key(item) for key, item in value.items()
        }
    if isinstance(value, list):
        return [_js_key(item) for item in value]
    return value


def _option(options: Mapping[str, Any], snake: str, default: Any) -> Any:
    for key in (snake, _camel(snake)):
        if key in options and options[key] is not None:
            return options[key]
    return default


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValidationError(f"{name} must be a positive integer")
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as error:
        raise ValidationError(f"{name} must be a positive integer") from error
    if not math.isfinite(number) or number <= 0 or not number.is_integer():
        raise ValidationError(f"{name} must be a positive integer")
    return int(number)


def _score(metrics: object, score_by: str) -> float:
    if not isinstance(metrics, Mapping):
        return -math.inf
    value = metrics.get(score_by)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return -math.inf
    number = float(value)
    return number if math.isfinite(number) else -math.inf


def _canonical_params(params: Mapping[str, object]) -> str:
    ordered = {key: params[key] for key in sorted(params)}
    try:
        return json.dumps(ordered, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValidationError("parameter sets must be JSON-safe") from error


def _window_ranges(
    length: int, train_bars: int, test_bars: int, step_bars: int, mode: str
) -> list[dict[str, int]]:
    ranges: list[dict[str, int]] = []
    start = 0
    while start + train_bars + test_bars <= length:
        train_start = 0 if mode == "anchored" else start
        train_end = train_bars + start if mode == "anchored" else start + train_bars
        test_start = train_end
        test_end = test_start + test_bars
        if test_end > length:
            break
        ranges.append(
            {
                "trainStart": train_start,
                "trainEnd": train_end,
                "testStart": test_start,
                "testEnd": test_end,
            }
        )
        start += step_bars
    return ranges


def _stitch_equity(target: list[dict[str, object]], source: object) -> None:
    if not isinstance(source, Sequence) or isinstance(source, (str, bytes)) or not source:
        return
    points = [dict(point) for point in source if isinstance(point, Mapping)]
    if not target:
        target.extend(points)
        return
    last_time = float(cast(Any, target[-1]["time"]))
    target.extend(point for point in points if float(point["time"]) > last_time)


def _candle_time(candle: object) -> object:
    return candle.get("time") if isinstance(candle, Mapping) else None


def _summarize_best_params(windows: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    summaries: dict[str, dict[str, Any]] = {}
    adjacent_repeats = 0
    for index, window in enumerate(windows):
        params = window["bestParams"]
        signature = window.get("bestParamsSignature") or _canonical_params(params)
        current = summaries.get(signature)
        if current is None:
            current = {
                "params": params,
                "wins": 0,
                "profitableWindows": 0,
                "oosTrades": 0,
            }
            summaries[signature] = current
        current["wins"] += 1
        current["profitableWindows"] += 1 if window["profitable"] else 0
        current["oosTrades"] += window["oosTrades"]
        if index > 0:
            previous = windows[index - 1]
            previous_signature = previous.get("bestParamsSignature") or _canonical_params(
                previous["bestParams"]
            )
            if previous_signature == signature:
                adjacent_repeats += 1
    leaderboard = list(summaries.values())
    leaderboard.sort(
        key=lambda value: (int(value["wins"]), int(value["profitableWindows"])),
        reverse=True,
    )
    adjacent_pairs = max(0, len(windows) - 1)
    return {
        "winners": [window["bestParams"] for window in windows],
        "stability": {
            "adjacentRepeatRate": adjacent_repeats / adjacent_pairs if adjacent_pairs else 0,
            "uniqueWinnerCount": len(summaries),
            "dominant": leaderboard[0] if leaderboard else None,
            "leaderboard": leaderboard,
        },
    }


class BestParamsList(list[dict[str, Any]]):
    """A list carrying the non-enumerable-like JavaScript summary properties."""

    def __init__(
        self,
        values: Sequence[dict[str, Any]],
        winners: list[dict[str, Any]],
        stability: dict[str, Any],
    ) -> None:
        super().__init__(values)
        self.winners = winners
        self.stability = stability


def walk_forward_optimize(
    options: Mapping[str, object] | None = None, /, **kwargs: object
) -> dict[str, Any]:
    """Select on each training window and evaluate on its following OOS window."""
    raw: dict[str, Any] = dict(options or {})
    raw.update(kwargs)
    candle_values = _option(raw, "candles", [])
    if isinstance(candle_values, (str, bytes, bytearray)) or not isinstance(
        candle_values, Sequence
    ):
        raise ValidationError("candles must be a sequence")
    if not candle_values:
        raise ValidationError("walk_forward_optimize requires non-empty candles")
    candles = list(candle_values)
    factory_value = _option(raw, "signal_factory", None)
    if not callable(factory_value):
        raise ValidationError("walk_forward_optimize requires a signal_factory callable")
    signal_factory: SignalFactory = factory_value
    parameter_values = _option(raw, "parameter_sets", [])
    if isinstance(parameter_values, (str, bytes, bytearray)) or not isinstance(
        parameter_values, Sequence
    ):
        raise ValidationError("parameter_sets must be a sequence")
    if not parameter_values:
        raise ValidationError("walk_forward_optimize requires parameter_sets")
    parameter_sets: list[dict[str, Any]] = []
    for index, params in enumerate(parameter_values):
        if not isinstance(params, Mapping):
            raise ValidationError("each parameter set must be a mapping", context={"index": index})
        normalized = {str(key): value for key, value in params.items()}
        _canonical_params(normalized)
        parameter_sets.append(normalized)
    safe_parameter_sets = BacktestResult({"parameterSets": parameter_sets})["parameterSets"]
    if not isinstance(safe_parameter_sets, list):  # pragma: no cover - model invariant
        raise ValidationError("parameter_sets must be JSON-safe")
    parameter_sets = safe_parameter_sets
    train_bars = _positive_integer(_option(raw, "train_bars", None), "train_bars")
    test_bars = _positive_integer(_option(raw, "test_bars", None), "test_bars")
    step_bars = _positive_integer(_option(raw, "step_bars", test_bars), "step_bars")
    mode = _option(raw, "mode", "rolling")
    if mode not in {"rolling", "anchored"}:
        raise ValidationError("mode must be rolling or anchored")
    score_by = _option(raw, "score_by", "profitFactor")
    if not isinstance(score_by, str):
        raise ValidationError("score_by must be a string")
    backtest_value = _option(raw, "backtest_options", {})
    if not isinstance(backtest_value, Mapping):
        raise ValidationError("backtest_options must be a mapping")
    backtest_options = dict(backtest_value)
    equity_value = backtest_options.get("equity", 10_000)
    rolling_equity = 10_000 if equity_value is None else float(equity_value)
    if not math.isfinite(rolling_equity):
        raise ValidationError("backtest_options.equity must be finite")

    ranges = _window_ranges(len(candles), train_bars, test_bars, step_bars, str(mode))
    if not ranges:
        required = train_bars + test_bars
        raise ValidationError(
            "walk_forward_optimize produced zero windows: "
            f"need at least {required} candles (train_bars={train_bars} + "
            f"test_bars={test_bars}) but got {len(candles)}"
        )

    windows: list[dict[str, Any]] = []
    all_trades: list[dict[str, object]] = []
    all_positions: list[dict[str, object]] = []
    eq_series: list[dict[str, object]] = []
    train_options = dict(backtest_options)
    train_options.update({"collectEqSeries": False, "collectReplay": False})
    test_options = dict(backtest_options)

    for window_range in ranges:
        train_slice = candles[window_range["trainStart"] : window_range["trainEnd"]]
        test_slice = candles[window_range["testStart"] : window_range["testEnd"]]
        if not train_slice or not test_slice:
            raise ValidationError(
                "walk_forward_optimize generated an empty window",
                context={
                    "train": len(train_slice),
                    "test": len(test_slice),
                    "range": window_range,
                },
            )
        best: dict[str, Any] | None = None
        for params in parameter_sets:
            signal = signal_factory(params)
            if not callable(signal):
                raise ValidationError("signal_factory must return a signal callable")
            call_options = dict(train_options)
            call_options.update(
                {"candles": train_slice, "equity": rolling_equity, "signal": signal}
            )
            train_result = backtest(call_options)
            score = _score(train_result.get("metrics"), score_by)
            if best is None or score > best["score"]:
                best = {
                    "params": params,
                    "score": score,
                    "metrics": train_result["metrics"],
                }
        if best is None:
            raise ValidationError("walk_forward_optimize could not select a parameter set")
        test_signal = signal_factory(best["params"])
        if not callable(test_signal):
            raise ValidationError("signal_factory must return a signal callable")
        call_options = dict(test_options)
        call_options.update(
            {"candles": test_slice, "equity": rolling_equity, "signal": test_signal}
        )
        test_result = backtest(call_options)
        test_metrics = test_result["metrics"]
        if not isinstance(test_metrics, Mapping):
            raise ValidationError("walk-forward test metrics must be a mapping")
        final_equity = test_metrics.get("finalEquity")
        if isinstance(final_equity, bool) or not isinstance(final_equity, (int, float)):
            raise ValidationError("walk-forward final equity must be numeric")
        rolling_equity = float(final_equity)
        all_trades.extend(test_result["trades"])
        all_positions.extend(test_result["positions"])
        _stitch_equity(eq_series, test_result["eqSeries"])
        params = best["params"]
        windows.append(
            {
                "train": {
                    "start": _candle_time(train_slice[0]),
                    "end": _candle_time(train_slice[-1]),
                },
                "test": {
                    "start": _candle_time(test_slice[0]),
                    "end": _candle_time(test_slice[-1]),
                },
                "bestParams": params,
                "trainScore": best["score"],
                "trainMetrics": best["metrics"],
                "testMetrics": test_metrics,
                "oosTrades": test_metrics["trades"],
                "profitable": float(test_metrics["totalPnL"]) > 0,
                "stabilityScore": 0,
                "bestParamsSignature": _canonical_params(params),
                "result": test_result.to_dict(),
            }
        )

    for index, window in enumerate(windows):
        signature = window["bestParamsSignature"]
        adjacent: list[int] = []
        if index > 0:
            adjacent.append(1 if windows[index - 1].get("bestParamsSignature") == signature else 0)
        if index + 1 < len(windows):
            adjacent.append(1 if windows[index + 1].get("bestParamsSignature") == signature else 0)
        window["stabilityScore"] = sum(adjacent) / len(adjacent) if adjacent else 1
        window.pop("bestParamsSignature", None)

    start_equity = backtest_options.get("equity", 10_000)
    if start_equity is None:
        start_equity = 10_000
    interval = backtest_options.get("interval")
    metrics = build_metrics(
        closed=all_trades,
        equity_start=float(start_equity),
        equity_final=rolling_equity,
        candles=candles,
        est_bar_ms=estimate_bar_ms(candles),
        eq_series=eq_series,
        interval=interval if isinstance(interval, str) else None,
    )
    summary = _summarize_best_params(windows)
    stability = summary["stability"]
    winners = summary["winners"]
    if not isinstance(stability, dict) or not isinstance(winners, list):
        raise ValidationError("walk-forward summary must be structured")
    best_params_values = [window["bestParams"] for window in windows]
    best_params = BestParamsList(best_params_values, winners, stability)
    output = {
        "windows": windows,
        "trades": all_trades,
        "positions": all_positions,
        "openPositions": [],
        "metrics": _js_key(metrics),
        "eqSeries": eq_series,
        "replay": {"frames": [], "events": []},
        "bestParams": best_params,
        "bestParamsSummary": stability,
    }
    try:
        json.dumps(output, allow_nan=False)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValidationError("walk-forward result must be JSON-safe") from error
    return output
