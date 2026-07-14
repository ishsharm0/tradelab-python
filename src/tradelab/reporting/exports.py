"""CSV, JSON, Markdown, HTML, and combined backtest artifact exports."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TypedDict

from tradelab.errors import ValidationError

from ._io import atomic_write_text, iso_milliseconds, safe_segment, strict_json_dumps
from .html import export_html_report
from .summary import render_markdown_report

_CSV_HEADER = (
    "time_open",
    "time_close",
    "side",
    "entry",
    "stop",
    "takeProfit",
    "exit",
    "reason",
    "size",
    "pnl",
    "R",
    "mfeR",
    "maeR",
    "adds",
    "entryATR",
    "exitATR",
)


class ArtifactPaths(TypedDict):
    """Paths produced by :func:`export_backtest_artifacts`."""

    csv: Path | None
    html: Path | None
    markdown: Path | None
    metrics: Path | None


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValidationError(f"{name} must be a mapping")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool):
        number = 1.0 if value else 0.0
    else:
        try:
            number = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError, OverflowError) as error:
            raise ValidationError(f"{name} must be numeric") from error
    if not math.isfinite(number):
        raise ValidationError(f"{name} must be finite")
    return number


def _fixed(value: object, digits: int, name: str) -> str:
    return f"{_number(value, name):.{digits}f}"


def _csv_cell(value: object) -> str:
    text = "" if value is None else str(value)
    if any(character in text for character in (",", '"', "\r", "\n")):
        return f'"{text.replace(chr(34), chr(34) * 2)}"'
    return text


def _trade_r_multiple(trade: Mapping[str, Any], exit_data: Mapping[str, Any]) -> float:
    initial_risk = _number(trade.get("_initRisk", 0) or 0, "trade._initRisk")
    if initial_risk <= 0:
        return 0.0
    entry_value = trade.get("entryFill")
    entry = _number(trade.get("entry") if entry_value is None else entry_value, "trade.entry")
    exit_price = _number(exit_data.get("price"), "trade.exit.price")
    per_unit = exit_price - entry if trade.get("side") == "long" else entry - exit_price
    return per_unit / initial_risk


def _trade_row(trade_value: object) -> tuple[str, ...]:
    trade = _mapping(trade_value, "trade")
    exit_data = _mapping(trade.get("exit"), "trade.exit")
    entry_atr = trade.get("entryATR")
    exit_atr = exit_data.get("exitATR")
    return (
        iso_milliseconds(trade.get("openTime")),
        iso_milliseconds(exit_data.get("time")),
        str(trade.get("side", "")),
        _fixed(trade.get("entry"), 6, "trade.entry"),
        _fixed(trade.get("stop"), 6, "trade.stop"),
        _fixed(trade.get("takeProfit"), 6, "trade.takeProfit"),
        _fixed(exit_data.get("price"), 6, "trade.exit.price"),
        str(exit_data.get("reason", "")),
        str(trade.get("size", "")),
        _fixed(exit_data.get("pnl"), 2, "trade.exit.pnl"),
        f"{_trade_r_multiple(trade, exit_data):.3f}",
        _fixed(trade.get("mfeR", 0) or 0, 3, "trade.mfeR"),
        _fixed(trade.get("maeR", 0) or 0, 3, "trade.maeR"),
        str(trade.get("adds", 0) or 0),
        "" if entry_atr is None else _fixed(entry_atr, 6, "trade.entryATR"),
        "" if exit_atr is None else _fixed(exit_atr, 6, "trade.exit.exitATR"),
    )


def export_trades_csv(
    closed_trades: Sequence[object] | None,
    *,
    symbol: object = "UNKNOWN",
    interval: object = "tf",
    range_: object = "range",
    out_dir: str | Path = "output",
) -> Path | None:
    """Write the JavaScript-compatible closed-trade ledger."""
    if not closed_trades:
        return None
    rows = [_CSV_HEADER, *(_trade_row(trade) for trade in closed_trades)]
    content = "\n".join(",".join(_csv_cell(value) for value in row) for row in rows)
    path = Path(out_dir) / (
        f"trades-{safe_segment(symbol)}-{safe_segment(interval)}-{safe_segment(range_)}.csv"
    )
    return atomic_write_text(path, content)


def _result_meta(
    result: Mapping[str, Any], symbol: object | None, interval: object | None, range_: object | None
) -> tuple[object, object, object]:
    return (
        result.get("symbol") if symbol is None else symbol,
        result.get("interval", "tf") if interval is None else interval,
        result.get("range", "range") if range_ is None else range_,
    )


def export_metrics_json(
    result: Mapping[str, Any],
    *,
    symbol: object | None = None,
    interval: object | None = None,
    range_: object | None = None,
    out_dir: str | Path = "output",
) -> Path:
    """Write strict, machine-readable metrics JSON."""
    metrics = result.get("metrics") if isinstance(result, Mapping) else None
    if not isinstance(metrics, Mapping):
        raise ValidationError("export_metrics_json requires a backtest result with metrics")
    symbol_value, interval_value, range_value = _result_meta(result, symbol, interval, range_)
    path = Path(out_dir) / (
        f"metrics-{safe_segment(symbol_value)}-{safe_segment(interval_value)}-"
        f"{safe_segment(range_value)}.json"
    )
    content = strict_json_dumps(metrics) + "\n"
    return atomic_write_text(path, content)


def export_markdown_report(
    result: Mapping[str, Any],
    *,
    symbol: object | None = None,
    interval: object | None = None,
    range_: object | None = None,
    out_dir: str | Path = "output",
    verdict: Mapping[str, object] | None = None,
) -> Path:
    """Write a deterministic Markdown backtest report."""
    metrics = result.get("metrics") if isinstance(result, Mapping) else None
    if not isinstance(metrics, Mapping):
        raise ValidationError("export_markdown_report requires a backtest result with metrics")
    symbol_value, interval_value, range_value = _result_meta(result, symbol, interval, range_)
    content = render_markdown_report(
        symbol=symbol_value,
        interval=interval_value,
        range_=range_value,
        metrics=metrics,
        verdict=verdict,
    )
    path = Path(out_dir) / (
        f"report-{safe_segment(symbol_value)}-{safe_segment(interval_value)}-"
        f"{safe_segment(range_value)}.md"
    )
    return atomic_write_text(path, content)


def export_backtest_artifacts(
    result: Mapping[str, Any] | None,
    *,
    symbol: object | None = None,
    interval: object | None = None,
    range_: object | None = None,
    out_dir: str | Path = "output",
    export_csv: bool = True,
    export_html: bool = True,
    export_markdown: bool = True,
    export_metrics: bool = True,
    csv_source: str = "positions",
) -> ArtifactPaths:
    """Write selected report artifacts using positions as the default CSV source."""
    if not isinstance(result, Mapping):
        raise ValidationError("export_backtest_artifacts requires a backtest result")
    symbol_value, interval_value, range_value = _result_meta(result, symbol, interval, range_)
    csv_values = result.get("trades") if csv_source == "trades" else result.get("positions")
    if csv_values is None:
        csv_values = result.get("trades", [])
    if not isinstance(csv_values, Sequence) or isinstance(csv_values, (str, bytes, bytearray)):
        raise ValidationError("backtest CSV source must be a sequence")
    metrics = result.get("metrics")
    eq_series = result.get("eqSeries")
    replay = result.get("replay")
    positions = result.get("positions", [])
    return {
        "csv": export_trades_csv(
            csv_values,
            symbol=symbol_value,
            interval=interval_value,
            range_=range_value,
            out_dir=out_dir,
        )
        if export_csv
        else None,
        "html": export_html_report(
            symbol=symbol_value,
            interval=interval_value,
            range_=range_value,
            metrics=metrics if isinstance(metrics, Mapping) else {},
            eq_series=eq_series if isinstance(eq_series, Sequence) else [],
            replay=replay if isinstance(replay, Mapping) else None,
            positions=positions if isinstance(positions, Sequence) else [],
            out_dir=out_dir,
        )
        if export_html
        else None,
        "markdown": export_markdown_report(
            result,
            symbol=symbol_value,
            interval=interval_value,
            range_=range_value,
            out_dir=out_dir,
        )
        if export_markdown
        else None,
        "metrics": export_metrics_json(
            result,
            symbol=symbol_value,
            interval=interval_value,
            range_=range_value,
            out_dir=out_dir,
        )
        if export_metrics
        else None,
    }


__all__ = [
    "ArtifactPaths",
    "export_backtest_artifacts",
    "export_markdown_report",
    "export_metrics_json",
    "export_trades_csv",
]
