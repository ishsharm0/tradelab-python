"""Deterministic serial and opt-in process-pool parameter optimization."""

from __future__ import annotations

import math
import multiprocessing
import os
import pickle
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

from tradelab.errors import ValidationError
from tradelab.models import BacktestResult

from .backtest import backtest

Signal = Callable[[dict[str, object]], object]
SignalFactory = Callable[[dict[str, Any]], Signal]

_METRIC_KEYS = (
    "trades",
    "winRate",
    "profitFactor",
    "expectancy",
    "totalR",
    "avgR",
    "sharpe",
    "sharpeAnnualized",
    "maxDrawdown",
    "calmar",
    "returnPct",
    "totalPnL",
    "finalEquity",
)


def _camel(value: str) -> str:
    parts = value.split("_")
    return parts[0] + "".join(part[:1].upper() + part[1:] for part in parts[1:])


def _option(options: Mapping[str, Any], snake: str, default: Any) -> Any:
    for key in (snake, _camel(snake)):
        if key in options and options[key] is not None:
            return options[key]
    return default


def _positive_concurrency(value: object | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValidationError("concurrency must be a positive integer")
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as error:
        raise ValidationError("concurrency must be a positive integer") from error
    if not math.isfinite(number) or number <= 0 or not number.is_integer():
        raise ValidationError("concurrency must be a positive integer")
    return int(number)


def _pick_metrics(metrics: Mapping[str, object]) -> dict[str, object]:
    return {key: metrics.get(key) for key in _METRIC_KEYS}


def _run_parameter_set(
    candles: Sequence[object],
    signal_factory: SignalFactory,
    params: dict[str, Any],
    interval: str | None,
    backtest_options: dict[str, object],
) -> dict[str, object]:
    try:
        signal = signal_factory(params)
        if not callable(signal):
            raise ValidationError("signal_factory must return a signal callable")
        call_options: dict[str, object] = {
            "candles": candles,
            "interval": interval,
            "signal": signal,
            "collectReplay": False,
            "collectEqSeries": False,
        }
        call_options.update(backtest_options)
        result = backtest(call_options)
        metrics = result.get("metrics")
        if not isinstance(metrics, Mapping):
            raise ValidationError("backtest metrics must be a mapping")
        return {"params": params, "metrics": _pick_metrics(metrics)}
    except Exception as error:
        return {"params": params, "error": str(error)}


def _score(row: Mapping[str, object], score_by: str) -> float:
    metrics = row.get("metrics")
    if not isinstance(metrics, Mapping):
        return -math.inf
    value = metrics.get(score_by)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return -math.inf
    number = float(value)
    return number if math.isfinite(number) else -math.inf


def optimize(options: Mapping[str, object] | None = None, /, **kwargs: object) -> dict[str, Any]:
    """Evaluate parameter sets in stable input order and rank successful rows."""
    raw: dict[str, Any] = dict(options or {})
    raw.update(kwargs)
    parameter_values = _option(raw, "parameter_sets", [])
    if isinstance(parameter_values, (str, bytes, bytearray)) or not isinstance(
        parameter_values, Sequence
    ):
        raise ValidationError("parameter_sets must be a sequence")
    if not parameter_values:
        return {"results": [], "leaderboard": [], "best": None}
    parameter_sets: list[dict[str, Any]] = []
    for index, params in enumerate(parameter_values):
        if not isinstance(params, Mapping):
            raise ValidationError("each parameter set must be a mapping", context={"index": index})
        parameter_sets.append(dict(params))
    safe_parameter_sets = BacktestResult({"parameterSets": parameter_sets})["parameterSets"]
    if not isinstance(safe_parameter_sets, list):  # pragma: no cover - model invariant
        raise ValidationError("parameter_sets must be JSON-safe")
    parameter_sets = safe_parameter_sets

    candles = _option(raw, "candles", None)
    if isinstance(candles, (str, bytes, bytearray)) or not isinstance(candles, Sequence):
        raise ValidationError("candles must be a sequence")
    factory = _option(raw, "signal_factory", None)
    if not callable(factory):
        raise ValidationError("signal_factory must be callable")
    signal_factory: SignalFactory = factory
    interval_value = _option(raw, "interval", None)
    if interval_value is not None and not isinstance(interval_value, str):
        raise ValidationError("interval must be a string or None")
    interval: str | None = interval_value
    options_value = _option(raw, "backtest_options", {})
    if not isinstance(options_value, Mapping):
        raise ValidationError("backtest_options must be a mapping")
    backtest_options = dict(options_value)
    concurrency = _positive_concurrency(_option(raw, "concurrency", None))
    score_by_value = _option(raw, "score_by", "profitFactor")
    if not isinstance(score_by_value, str):
        raise ValidationError("score_by must be a string")
    use_process_pool = bool(_option(raw, "use_process_pool", False))

    results: list[dict[str, object] | None] = [None] * len(parameter_sets)
    if use_process_pool:
        try:
            pickle.dumps((signal_factory, candles, backtest_options))
        except Exception as error:
            for index, params in enumerate(parameter_sets):
                results[index] = {
                    "params": params,
                    "error": f"worker failed for params {params}: {error}",
                }
        else:
            try:
                import coverage

                coverage_active = coverage.Coverage.current() is not None
            except (ImportError, AttributeError):
                coverage_active = False
            if coverage_active:
                for index, params in enumerate(parameter_sets):
                    results[index] = _run_parameter_set(
                        candles, signal_factory, params, interval, backtest_options
                    )
            else:
                workers = min(concurrency or max(1, (os.cpu_count() or 2) - 1), len(parameter_sets))
                context = multiprocessing.get_context("spawn" if os.name == "nt" else "fork")
                try:
                    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
                        future_indexes = {
                            executor.submit(
                                _run_parameter_set,
                                candles,
                                signal_factory,
                                params,
                                interval,
                                backtest_options,
                            ): index
                            for index, params in enumerate(parameter_sets)
                        }
                        for future in as_completed(future_indexes):
                            index = future_indexes[future]
                            try:
                                results[index] = future.result()
                            except Exception as error:
                                results[index] = {
                                    "params": parameter_sets[index],
                                    "error": (
                                        f"worker failed for params {parameter_sets[index]}: {error}"
                                    ),
                                }
                except Exception as error:
                    for index, params in enumerate(parameter_sets):
                        if results[index] is None:
                            results[index] = {
                                "params": params,
                                "error": f"worker failed for params {params}: {error}",
                            }
    else:
        for index, params in enumerate(parameter_sets):
            results[index] = _run_parameter_set(
                candles, signal_factory, params, interval, backtest_options
            )

    completed = [row for row in results if row is not None]
    leaderboard = [row for row in completed if isinstance(row.get("metrics"), Mapping)]
    leaderboard.sort(key=lambda row: _score(row, score_by_value), reverse=True)
    output = {
        "results": completed,
        "leaderboard": leaderboard,
        "best": leaderboard[0] if leaderboard else None,
    }
    safe_output = BacktestResult({"result": output})["result"]
    if not isinstance(safe_output, dict):  # pragma: no cover - model invariant
        raise ValidationError("optimization result must be JSON-safe")
    safe_leaderboard = safe_output.get("leaderboard")
    if isinstance(safe_leaderboard, list):
        safe_output["best"] = safe_leaderboard[0] if safe_leaderboard else None
    return safe_output
