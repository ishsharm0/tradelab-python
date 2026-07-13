"""Aggregate metrics for completed backtest trade legs."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TypeAlias

from tradelab.errors import ValidationError

from .annualize import _js_round_positive, periods_per_year
from .benchmark import _mean, _stddev, benchmark_stats
from .finite import BIG_NUMBER, Number, _finite_number, _json_number, clamp_finite

ValidatedLeg: TypeAlias = tuple[Mapping[str, object], Mapping[str, object], float, float, bool]
_PERCENTILE_RANKS: tuple[tuple[str, float], ...] = (
    ("p10", 0.1),
    ("p25", 0.25),
    ("p50", 0.5),
    ("p75", 0.75),
    ("p90", 0.9),
)
_MS_PER_DAY = 86_400_000
_TIME_CLIP_LIMIT = 8_640_000_000_000_000


def _sequence(value: object, name: str) -> Sequence[object]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ValidationError(f"{name} must be a sequence")
    return value


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValidationError(f"{name} must be a mapping")
    return value


def _value(record: Mapping[str, object], camel: str, snake: str, default: object = None) -> object:
    if camel in record:
        return record[camel]
    if snake in record:
        return record[snake]
    return default


def _day_key_utc(time_ms: float) -> int:
    """Return an epoch-day key after ECMAScript Date TimeClip truncation."""
    clipped = math.trunc(time_ms)
    if abs(clipped) > _TIME_CLIP_LIMIT:
        raise ValidationError("equity timestamp is outside the ECMAScript TimeClip range")
    return clipped // _MS_PER_DAY


def _sortino(values: Sequence[float]) -> float:
    losses = [value for value in values if value < 0]
    downside_deviation = _stddev(losses if losses else [0.0])
    average = _mean(values)
    return math.inf if downside_deviation == 0 and average > 0 else (
        0.0 if downside_deviation == 0 else average / downside_deviation
    )


def _trade_r_multiple(trade: Mapping[str, object], exit_data: Mapping[str, object]) -> float:
    risk_value = _value(trade, "_initRisk", "init_risk", 0.0)
    initial_risk = _finite_number(risk_value, "trade initial risk") if risk_value else 0.0
    if initial_risk <= 0:
        return 0.0
    entry_fill = _value(trade, "entryFill", "entry_fill")
    entry_value = _value(trade, "entry", "entry") if entry_fill is None else entry_fill
    entry = _finite_number(entry_value, "trade entry")
    exit_price = _finite_number(_value(exit_data, "price", "price"), "exit price")
    side = _value(trade, "side", "side")
    per_unit = exit_price - entry if side == "long" else entry - exit_price
    return per_unit / initial_risk


def _streaks(labels: Sequence[str]) -> tuple[int, int]:
    wins = losses = max_wins = max_losses = 0
    for label in labels:
        if label == "win":
            wins += 1
            losses = 0
            max_wins = max(max_wins, wins)
        elif label == "loss":
            losses += 1
            wins = 0
            max_losses = max(max_losses, losses)
        else:
            wins = losses = 0
    return max_wins, max_losses


def _daily_returns(eq_series: Sequence[Mapping[str, float]]) -> list[float]:
    records: dict[int, dict[str, float]] = {}
    for point in eq_series:
        time = point["time"]
        equity = point["equity"]
        day = _day_key_utc(time)
        record = records.get(day)
        if record is None:
            record = {"open": equity, "close": equity, "first": time, "last": time}
            records[day] = record
        if time < record["first"]:
            record["first"] = time
            record["open"] = equity
        if time >= record["last"]:
            record["last"] = time
            record["close"] = equity
    returns: list[float] = []
    for record in records.values():
        opening = record["open"]
        closing = record["close"]
        if opening > 0 and math.isfinite(opening) and math.isfinite(closing):
            returns.append((closing - opening) / opening)
    return returns


def _percentile(values: Sequence[float], rank: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[math.floor((len(ordered) - 1) * rank)]


def _profit_factor(gross_profit: float, gross_loss: float) -> float:
    if gross_loss == 0:
        return BIG_NUMBER if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def _validated_leg(value: object, index: int) -> ValidatedLeg:
    trade = _mapping(value, f"closed[{index}]")
    exit_data = _mapping(_value(trade, "exit", "exit"), f"closed[{index}].exit")
    exit_time = _finite_number(_value(exit_data, "time", "time"), f"closed[{index}].exit.time")
    pnl = _finite_number(_value(exit_data, "pnl", "pnl"), f"closed[{index}].exit.pnl")
    is_scale = _value(exit_data, "reason", "reason") == "SCALE"
    if not is_scale:
        _finite_number(_value(exit_data, "price", "price"), f"closed[{index}].exit.price")
        _finite_number(_value(trade, "openTime", "open_time"), f"closed[{index}].open_time")
    return trade, exit_data, exit_time, pnl, is_scale


def _equity_points(value: object) -> list[Mapping[str, float]]:
    points = _sequence(value, "eq_series")
    output: list[Mapping[str, float]] = []
    for index, point_value in enumerate(points):
        point = _mapping(point_value, f"eq_series[{index}]")
        output.append(
            {
                "time": _finite_number(_value(point, "time", "time"), f"eq_series[{index}].time"),
                "equity": _finite_number(
                    _value(point, "equity", "equity"), f"eq_series[{index}].equity"
                ),
            }
        )
    return output


def build_metrics(
    *,
    closed: Sequence[Mapping[str, object]],
    equity_start: Number,
    equity_final: Number,
    candles: Sequence[Mapping[str, object]],
    est_bar_ms: Number,
    eq_series: Sequence[Mapping[str, object]] | None = None,
    interval: str | None = None,
    benchmark_returns: Sequence[Number] | None = None,
) -> dict[str, object]:
    """Build aggregate metrics with snake_case keys from realized trade-leg mappings."""
    closed_values = _sequence(closed, "closed")
    candle_values = _sequence(candles, "candles")
    start_equity = _finite_number(equity_start, "equity_start")
    final_equity = _finite_number(equity_final, "equity_final")
    bar_ms = _finite_number(est_bar_ms, "est_bar_ms")
    if bar_ms <= 0:
        raise ValidationError("est_bar_ms must be positive")
    legs = [_validated_leg(value, index) for index, value in enumerate(closed_values)]
    legs.sort(key=lambda leg: leg[2])
    for index, candle in enumerate(candle_values):
        _mapping(candle, f"candles[{index}]")

    completed_count = winning_trade_count = winning_leg_count = 0
    total_r = realized_pnl = gross_profit_positions = gross_loss_positions = 0.0
    gross_profit_legs = gross_loss_legs = 0.0
    open_bars: int | float = 0
    long_trades = long_wins = short_trades = short_wins = 0
    long_pnl = short_pnl = 0.0
    trade_rs: list[float] = []
    trade_pnls: list[float] = []
    trade_returns: list[float] = []
    hold_minutes: list[float] = []
    labels: list[str] = []
    long_rs: list[float] = []
    short_rs: list[float] = []
    peak_equity = current_equity = start_equity
    max_drawdown = 0.0

    for trade, exit_data, exit_time, pnl, is_scale in legs:
        realized_pnl += pnl
        if pnl > 0:
            gross_profit_legs += pnl
            winning_leg_count += 1
        elif pnl < 0:
            gross_loss_legs += abs(pnl)
        current_equity += pnl
        peak_equity = max(peak_equity, current_equity)
        drawdown = (peak_equity - current_equity) / max(1e-12, peak_equity)
        max_drawdown = max(max_drawdown, drawdown)
        if is_scale:
            continue

        completed_count += 1
        trade_pnls.append(pnl)
        trade_returns.append(pnl / max(1e-12, start_equity))
        trade_r = _trade_r_multiple(trade, exit_data)
        trade_rs.append(trade_r)
        total_r += trade_r
        labels.append("win" if pnl > 0 else "loss" if pnl < 0 else "flat")
        open_time = _finite_number(_value(trade, "openTime", "open_time"), "trade open_time")
        hold_minutes.append((exit_time - open_time) / 60_000)
        open_bars += max(1, _js_round_positive((exit_time - open_time) / bar_ms))
        if pnl > 0:
            winning_trade_count += 1
            gross_profit_positions += pnl
        elif pnl < 0:
            gross_loss_positions += abs(pnl)
        side = _value(trade, "side", "side")
        if side == "long":
            long_trades += 1
            long_pnl += pnl
            long_rs.append(trade_r)
            if pnl > 0:
                long_wins += 1
        elif side == "short":
            short_trades += 1
            short_pnl += pnl
            short_rs.append(trade_r)
            if pnl > 0:
                short_wins += 1

    supplied_equity_series = _equity_points(eq_series) if eq_series is not None else []
    equity_series: list[Mapping[str, float]]
    if supplied_equity_series:
        equity_series = supplied_equity_series
    else:
        first_time = legs[0][2] if legs else 0.0
        equity_series = [{"time": first_time, "equity": start_equity}]
        reconstructed_equity = start_equity
        for _, _, exit_time, pnl, _ in legs:
            reconstructed_equity += pnl
            equity_series.append({"time": exit_time, "equity": reconstructed_equity})
    daily_returns = _daily_returns(equity_series)
    daily_std = _stddev(daily_returns)
    sharpe_daily = math.inf if daily_std == 0 and daily_returns else (
        0.0 if daily_std == 0 else _mean(daily_returns) / daily_std
    )
    sortino_daily = _sortino(daily_returns)
    periods = periods_per_year(interval, bar_ms)
    max_wins, max_losses = _streaks(labels)
    profit_factor_positions = _profit_factor(gross_profit_positions, gross_loss_positions)
    profit_factor_legs = _profit_factor(gross_profit_legs, gross_loss_legs)
    return_pct = (final_equity - start_equity) / max(1e-12, start_equity)
    calmar = math.inf if max_drawdown == 0 and return_pct > 0 else (
        0.0 if max_drawdown == 0 else return_pct / max_drawdown
    )
    trade_return_std = _stddev(trade_returns)
    sharpe_per_trade = math.inf if trade_return_std == 0 and trade_returns else (
        0.0 if trade_return_std == 0 else _mean(trade_returns) / trade_return_std
    )
    daily_win_rate = (
        len([value for value in daily_returns if value > 0]) / len(daily_returns)
        if daily_returns
        else 0.0
    )
    side_breakdown: dict[str, dict[str, float | int | None]] = {
        "long": {
            "trades": long_trades,
            "win_rate": long_wins / long_trades if long_trades else 0.0,
            "avg_pnl": _json_number(long_pnl / long_trades) if long_trades else 0.0,
            "avg_r": _json_number(_mean(long_rs)),
        },
        "short": {
            "trades": short_trades,
            "win_rate": short_wins / short_trades if short_trades else 0.0,
            "avg_pnl": _json_number(short_pnl / short_trades) if short_trades else 0.0,
            "avg_r": _json_number(_mean(short_rs)),
        },
    }
    benchmark_values = [] if benchmark_returns is None else benchmark_returns
    benchmark = benchmark_stats(daily_returns, benchmark_values)
    clamped_daily_sharpe = clamp_finite(sharpe_daily)
    clamped_daily_sortino = clamp_finite(sortino_daily)
    return {
        "trades": completed_count,
        "win_rate": winning_trade_count / completed_count if completed_count else 0.0,
        "profit_factor": clamp_finite(profit_factor_positions),
        "expectancy": _json_number(_mean(trade_pnls)),
        "total_r": _json_number(total_r),
        "avg_r": _json_number(_mean(trade_rs)),
        "sharpe": clamped_daily_sharpe,
        "sharpe_annualized": clamp_finite(clamped_daily_sharpe * math.sqrt(periods)),
        "sortino_annualized": clamp_finite(clamped_daily_sortino * math.sqrt(periods)),
        "sharpe_per_trade": clamp_finite(sharpe_per_trade),
        "sortino_per_trade": clamp_finite(_sortino(trade_returns)),
        "annualization_periods": periods if math.isfinite(periods) else None,
        "max_drawdown": _json_number(max_drawdown),
        "max_drawdown_pct": _json_number(max_drawdown),
        "calmar": clamp_finite(calmar),
        "max_consec_wins": max_wins,
        "max_consec_losses": max_losses,
        "avg_hold": _json_number(_mean(hold_minutes)),
        "avg_hold_min": _json_number(_mean(hold_minutes)),
        "exposure_pct": _json_number(open_bars / max(1, len(candle_values))),
        "total_pnl": _json_number(realized_pnl),
        "return_pct": _json_number(return_pct),
        "final_equity": final_equity,
        "start_equity": start_equity,
        "profit_factor_pos": clamp_finite(profit_factor_positions),
        "profit_factor_leg": clamp_finite(profit_factor_legs),
        "win_rate_pos": winning_trade_count / completed_count if completed_count else 0.0,
        "win_rate_leg": winning_leg_count / len(legs) if legs else 0.0,
        "sharpe_daily": clamped_daily_sharpe,
        "sortino_daily": clamped_daily_sortino,
        "benchmark": benchmark,
        "side_breakdown": side_breakdown,
        "long": side_breakdown["long"],
        "short": side_breakdown["short"],
        "r_dist": {
            key: _json_number(_percentile(trade_rs, rank)) for key, rank in _PERCENTILE_RANKS
        },
        "hold_dist_min": {
            key: _json_number(_percentile(hold_minutes, rank))
            for key, rank in _PERCENTILE_RANKS
        },
        "daily": {
            "count": len(daily_returns),
            "win_rate": daily_win_rate,
            "avg_return": _json_number(_mean(daily_returns)),
        },
    }
