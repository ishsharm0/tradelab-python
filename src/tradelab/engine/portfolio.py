"""Deterministic shared-capital portfolio backtesting."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from tradelab.errors import ValidationError
from tradelab.metrics import build_metrics
from tradelab.models import BacktestResult

from .backtest import BarSystemRunner, _js_key
from .execution import day_key_et, estimate_bar_ms

_UINT32_MASK = 0xFFFF_FFFF


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValidationError(f"{name} must be finite", context={name: value})
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError) as error:
        raise ValidationError(f"{name} must be finite", context={name: value}) from error
    if not math.isfinite(number):
        raise ValidationError(f"{name} must be finite", context={name: value})
    return number


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValidationError(f"{name} must be boolean", context={name: value})
    return value


def _system_option(system: Mapping[str, Any], snake: str, camel: str) -> object:
    if snake in system and system[snake] is not None:
        return system[snake]
    return system.get(camel)


def _optional_positive(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _weight(value: object) -> float:
    return _optional_positive(value) or 0.0


def _default_system_cap(
    total_equity: float,
    cap_pct: float,
    max_allocation: object,
    max_allocation_pct: object,
) -> float:
    limits: list[float] = []
    if math.isfinite(cap_pct) and cap_pct > 0:
        limits.append(total_equity * cap_pct)
    absolute_limit = _optional_positive(max_allocation)
    if absolute_limit is not None:
        limits.append(absolute_limit)
    percent_limit = _optional_positive(max_allocation_pct)
    if percent_limit is not None:
        limits.append(total_equity * percent_limit)
    return min(limits) if limits else max(0.0, total_equity)


def _hash_order_score(index: int, time: float, seed: int) -> int:
    value = (
        (int(time) & _UINT32_MASK)
        ^ (((index + 1) * 0x9E37_79B1) & _UINT32_MASK)
        ^ (seed & _UINT32_MASK)
    ) & _UINT32_MASK
    value = ((value ^ (value >> 16)) * 0x85EB_CA6B) & _UINT32_MASK
    value = ((value ^ (value >> 13)) * 0xC2B2_AE35) & _UINT32_MASK
    return (value ^ (value >> 16)) & _UINT32_MASK


def _iso(time_ms: float) -> str:
    try:
        value = datetime.fromtimestamp(time_ms / 1_000, UTC)
    except (ValueError, OverflowError, OSError) as error:
        raise ValidationError("portfolio timestamp must be Unix milliseconds") from error
    return value.isoformat(timespec="milliseconds").replace("+00:00", "Z")


@dataclass(slots=True)
class _SystemRunner:
    index: int
    symbol: str
    system: Mapping[str, Any]
    default_cap_pct: float
    initial_reference_equity: float
    runner: BarSystemRunner

    def cap(self, total_equity: float) -> float:
        return _default_system_cap(
            max(0.0, total_equity),
            self.default_cap_pct,
            _system_option(self.system, "max_allocation", "maxAllocation"),
            _system_option(self.system, "max_allocation_pct", "maxAllocationPct"),
        )


@dataclass(frozen=True, slots=True)
class _PortfolioState:
    marked_equity: float
    locked_capital: float
    available_capital: float


def _portfolio_state(runners: Sequence[_SystemRunner], initial_equity: float) -> _PortfolioState:
    marked_equity = initial_equity
    locked_capital = 0.0
    for entry in runners:
        marked_equity += entry.runner.get_marked_equity() - entry.initial_reference_equity
        locked_capital += entry.runner.get_locked_capital()
    return _PortfolioState(marked_equity, locked_capital, marked_equity - locked_capital)


def _point(time: float, state: _PortfolioState) -> dict[str, float]:
    return {
        "time": time,
        "timestamp": time,
        "equity": state.marked_equity,
        "lockedCapital": state.locked_capital,
        "availableCapital": state.available_capital,
    }


def _next_time_and_active(
    runners: Sequence[_SystemRunner],
) -> tuple[float, list[_SystemRunner]]:
    next_time = math.inf
    active: list[_SystemRunner] = []
    for entry in runners:
        time = entry.runner.peek_time()
        if time < next_time:
            next_time = time
            active = [entry]
        elif time == next_time:
            active.append(entry)
    return next_time, active


def _initial_time(runners: Sequence[_SystemRunner]) -> float:
    times = [float(entry.runner.candles[0]["time"]) for entry in runners if entry.runner.candles]
    return min(times) if times else 0.0


def _force_exit_all(runners: Sequence[_SystemRunner], time: float) -> None:
    for entry in runners:
        if not entry.runner.open:
            continue
        price = entry.runner.get_mark_price()
        if price is None or not math.isfinite(price):
            continue
        entry.runner.force_exit("PORTFOLIO_DAILY_LOSS", {"time": time, "close": price}, price)


def _with_symbol(record: Mapping[str, Any], symbol: str) -> dict[str, Any]:
    output = dict(record)
    if not output.get("symbol"):
        output["symbol"] = symbol
    return output


def _combine_replay(
    system_results: Sequence[dict[str, Any]],
    eq_series: Sequence[Mapping[str, float]],
    collect_replay: bool,
) -> dict[str, list[dict[str, Any]]]:
    if not collect_replay:
        return {"frames": [], "events": []}
    events: list[dict[str, Any]] = []
    for entry in system_results:
        replay = entry["result"]["replay"]
        for event in replay.get("events", []):
            events.append(_with_symbol(event, str(entry["symbol"])))
    events.sort(key=lambda event: str(event.get("t", "")))
    frames = [
        {
            "t": _iso(float(point["time"])),
            "price": 0,
            "equity": point["equity"],
            "posSide": None,
            "posSize": 0,
            "lockedCapital": point["lockedCapital"],
            "availableCapital": point["availableCapital"],
        }
        for point in eq_series
    ]
    return {"frames": frames, "events": events}


def backtest_portfolio(
    *,
    systems: Sequence[Mapping[str, Any]] = (),
    equity: object = 10_000,
    interval: str | None = None,
    allocation: str = "equal",
    collect_eq_series: object = True,
    collect_replay: object = False,
    max_daily_loss_pct: object = 0,
    processing_order: str = "sequential",
    shuffle_seed: object = 0,
) -> BacktestResult:
    """Run multiple bar systems against a single marked-equity capital pool."""
    if (
        isinstance(systems, (str, bytes, bytearray))
        or not isinstance(systems, Sequence)
        or not systems
    ):
        raise ValidationError("backtest_portfolio requires a non-empty systems sequence")
    if allocation not in {"equal", "weight"}:
        raise ValidationError('allocation must be "equal" or "weight"')
    if processing_order not in {"sequential", "shuffle"}:
        raise ValidationError('processing_order must be "sequential" or "shuffle"')
    if interval is not None and not isinstance(interval, str):
        raise ValidationError("interval must be a string or None")
    initial_equity = _finite(equity, "equity")
    if initial_equity <= 0:
        raise ValidationError("equity must be positive", context={"equity": equity})
    want_eq_series = _boolean(collect_eq_series, "collect_eq_series")
    want_replay = _boolean(collect_replay, "collect_replay")
    daily_loss_pct = abs(_finite(max_daily_loss_pct, "max_daily_loss_pct"))
    seed_number = _finite(shuffle_seed, "shuffle_seed")
    if not seed_number.is_integer():
        raise ValidationError("shuffle_seed must be an integer")
    seed = int(seed_number)

    normalized_systems: list[Mapping[str, Any]] = []
    for index, system in enumerate(systems):
        if not isinstance(system, Mapping):
            raise ValidationError("portfolio system must be a mapping", context={"index": index})
        normalized_systems.append(system)
    weights = (
        [1.0] * len(normalized_systems)
        if allocation == "equal"
        else [_weight(system.get("weight", 0)) for system in normalized_systems]
    )
    max_weight = max(weights)
    if max_weight <= 0:
        raise ValidationError("backtest_portfolio requires positive allocation weights")
    scaled_weights = [weight / max_weight for weight in weights]
    scaled_total = sum(scaled_weights)
    if not math.isfinite(scaled_total) or scaled_total <= 0:
        raise ValidationError("portfolio allocation weights could not be normalized")

    runners: list[_SystemRunner] = []
    for index, system in enumerate(normalized_systems):
        cap_pct = scaled_weights[index] / scaled_total
        reference_equity = initial_equity * cap_pct
        raw_symbol = system.get("symbol")
        symbol = str(raw_symbol) if raw_symbol is not None else f"system-{index + 1}"
        runner_options = dict(system)
        runner_options.update(
            {
                "symbol": symbol,
                "equity": reference_equity,
                "collect_eq_series": want_eq_series,
                "collect_replay": want_replay,
            }
        )
        runners.append(
            _SystemRunner(
                index=index,
                symbol=symbol,
                system=system,
                default_cap_pct=cap_pct,
                initial_reference_equity=reference_equity,
                runner=BarSystemRunner(runner_options),
            )
        )

    state = _portfolio_state(runners, initial_equity)
    eq_series: list[dict[str, float]] = (
        [_point(_initial_time(runners), state)] if want_eq_series else []
    )
    current_day: str | None = None
    day_start_equity = initial_equity
    portfolio_halted = False

    while True:
        next_time, active = _next_time_and_active(runners)
        if not math.isfinite(next_time):
            break
        if processing_order == "shuffle":
            active.sort(
                key=lambda entry: (
                    _hash_order_score(entry.index, next_time, seed),
                    entry.index,
                )
            )
        else:
            active.sort(key=lambda entry: entry.index)

        day = day_key_et(next_time)
        if current_day is None or day != current_day:
            current_day = day
            state = _portfolio_state(runners, initial_equity)
            day_start_equity = state.marked_equity
            portfolio_halted = False

        for system_entry in active:
            state = _portfolio_state(runners, initial_equity)
            total_equity = state.marked_equity
            available_capital = max(0.0, state.available_capital)
            system_locked = system_entry.runner.get_locked_capital()
            system_cap = system_entry.cap(total_equity)
            system_remaining = max(0.0, system_cap - system_locked)

            def resolve_entry_size(
                request: dict[str, object],
                system: _SystemRunner = system_entry,
                available: float = available_capital,
                remaining: float = system_remaining,
            ) -> float:
                desired_size = _finite(request.get("desired_size"), "desired_size")
                entry_price = _finite(request.get("entry_price"), "entry_price")
                leverage = max(1.0, system.runner.max_leverage or 1.0)
                denominator = max(1e-12, abs(entry_price))
                by_available = available * leverage / denominator
                by_system_cap = remaining * leverage / denominator
                return min(desired_size, by_available, by_system_cap)

            system_entry.runner.step(
                signal_equity=total_equity,
                can_trade=not portfolio_halted,
                resolve_entry_size=resolve_entry_size,
            )
            state = _portfolio_state(runners, initial_equity)
            if (
                not portfolio_halted
                and daily_loss_pct > 0
                and state.marked_equity <= day_start_equity * (1 - daily_loss_pct / 100)
            ):
                portfolio_halted = True
                for entry in runners:
                    entry.runner.cancel_pending()
                _force_exit_all(runners, next_time)
                state = _portfolio_state(runners, initial_equity)

        if want_eq_series:
            eq_series.append(_point(next_time, state))

    system_results: list[dict[str, Any]] = []
    for entry in runners:
        result = entry.runner.build_result().to_dict()
        result["replay"]["events"] = [
            _with_symbol(event, entry.symbol) for event in result["replay"]["events"]
        ]
        system_results.append(
            {
                "symbol": entry.symbol,
                "weight": entry.default_cap_pct,
                "equity": entry.initial_reference_equity,
                "allocationCapPct": entry.default_cap_pct,
                "allocationCap": entry.cap(initial_equity),
                "result": result,
            }
        )

    trades = [
        _with_symbol(trade, str(entry["symbol"]))
        for entry in system_results
        for trade in entry["result"]["trades"]
    ]
    trades.sort(key=lambda trade: float(trade["exit"]["time"]))
    positions = [
        _with_symbol(position, str(entry["symbol"]))
        for entry in system_results
        for position in entry["result"]["positions"]
    ]
    positions.sort(key=lambda trade: float(trade["exit"]["time"]))
    open_positions = [
        _with_symbol(position, str(entry["symbol"]))
        for entry in system_results
        for position in entry["result"].get("openPositions", [])
    ]
    replay = _combine_replay(system_results, eq_series, want_replay)
    ordered_candles = sorted(
        (candle for entry in runners for candle in entry.runner.candles),
        key=lambda candle: float(candle["time"]),
    )
    metrics_interval = interval
    if metrics_interval is None:
        first_interval = normalized_systems[0].get("interval")
        metrics_interval = first_interval if isinstance(first_interval, str) else None
    final_state = _portfolio_state(runners, initial_equity)
    final_equity = eq_series[-1]["equity"] if eq_series else final_state.marked_equity
    metrics = build_metrics(
        closed=trades,
        equity_start=initial_equity,
        equity_final=final_equity,
        candles=ordered_candles,
        est_bar_ms=estimate_bar_ms(ordered_candles),
        eq_series=eq_series,
        interval=metrics_interval,
    )
    return BacktestResult(
        {
            "symbol": "PORTFOLIO",
            "interval": metrics_interval,
            "range": None,
            "trades": trades,
            "positions": positions,
            "openPositions": open_positions,
            "metrics": _js_key(metrics),
            "eqSeries": eq_series,
            "replay": replay,
            "systems": system_results,
        }
    )
