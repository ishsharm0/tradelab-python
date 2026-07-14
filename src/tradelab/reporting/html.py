# ruff: noqa: E501
"""Self-contained, dependency-free HTML backtest dashboard rendering."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tradelab.errors import ValidationError

from ._io import atomic_write_text, iso_milliseconds, safe_segment, strict_json_dumps

_CSS = """
:root{color-scheme:dark;--bg:#07111f;--panel:#101d30;--text:#eef5ff;--muted:#9eb0c9;
--line:#26364d;--accent:#64e0c1;--negative:#fb7185}*{box-sizing:border-box}body{margin:0;
font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--text);
background:linear-gradient(180deg,#050a12,var(--bg));min-height:100vh}.report-shell{max-width:1180px;
margin:auto;padding:32px 22px 48px}.hero{display:flex;justify-content:space-between;align-items:flex-start;
gap:20px;flex-wrap:wrap;margin-bottom:22px}.eyebrow{color:var(--accent);font-size:.75rem;
letter-spacing:.14em;text-transform:uppercase;margin:0 0 8px}.hero h1{font-size:clamp(2rem,4vw,3rem);
line-height:1;margin:0 0 10px}.muted{color:var(--muted)}.pill{border:1px solid #315a59;
border-radius:999px;padding:9px 13px;color:#a8ffe8}.metric-grid,.panel-grid{display:grid;gap:14px}
.metric-grid{grid-template-columns:repeat(auto-fit,minmax(180px,1fr));margin-bottom:14px}
.panel-grid{grid-template-columns:repeat(12,minmax(0,1fr))}.card,.panel{background:var(--panel);
border:1px solid var(--line);border-radius:16px;box-shadow:0 18px 45px #0005}.card{padding:15px}
.metric-card__label{font-size:.75rem;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
.metric-card__value{font-size:1.5rem;font-weight:650;margin-top:7px}.metric-card__note{font-size:.82rem;
color:var(--muted);margin-top:5px}.panel{grid-column:span 6;padding:17px}.panel--wide{grid-column:span 12}
.panel h2{font-size:1rem;margin:0 0 4px}.chart{width:100%;height:auto;display:block;margin-top:12px}
.chart-grid{stroke:#314158;stroke-width:1}.chart-line{fill:none;stroke-linecap:round;stroke-linejoin:round;
stroke-width:3}.data-table{border-collapse:collapse;width:100%;font-size:.9rem;margin-top:10px}
.data-table th,.data-table td{text-align:left;padding:9px;border-bottom:1px solid var(--line)}
.data-table th{color:var(--muted)}.table-wrap{overflow:auto}.is-hidden{display:none}
@media(max-width:800px){.panel,.panel--wide{grid-column:span 12}.report-shell{padding:22px 14px}}
""".strip()


def _escape_html(value: object) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValidationError(f"{name} must be finite")
    return number


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _fmt(value: object, digits: int = 2) -> str:
    number = _finite(value)
    return "—" if number is None else f"{number:.{digits}f}"


def _fmt_pct(value: object, digits: int = 2) -> str:
    number = _finite(value)
    return "—" if number is None else f"{number * 100:.{digits}f}%"


def _nested(metrics: Mapping[str, object], key: str, child: str, default: object = 0) -> object:
    value = metrics.get(key)
    return value.get(child, default) if isinstance(value, Mapping) else default


def _line_svg(values: Sequence[float], *, color: str, label: str) -> str:
    width, height, padding = 720, 230, 24
    low, high = min(values), max(values)
    span = high - low or 1.0
    count = len(values)
    points = []
    for index, value in enumerate(values):
        x = padding + (width - padding * 2) * (index / max(1, count - 1))
        y = padding + (height - padding * 2) * ((high - value) / span)
        points.append(f"{x:.2f},{y:.2f}")
    return (
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{_escape_html(label)}">'
        f'<line class="chart-grid" x1="{padding}" y1="{height - padding}" '
        f'x2="{width - padding}" y2="{height - padding}"/>'
        f'<polyline class="chart-line" stroke="{color}" points="{" ".join(points)}"/>'
        "</svg>"
    )


def _bar_svg(values: Sequence[float], *, label: str) -> str:
    width, height, padding = 720, 230, 24
    maximum = max((abs(value) for value in values), default=1.0) or 1.0
    center = height / 2
    available = center - padding
    bar_width = (width - padding * 2) / max(1, len(values))
    bars: list[str] = []
    for index, value in enumerate(values):
        magnitude = abs(value) / maximum * available
        x = padding + index * bar_width + bar_width * 0.12
        y = center - magnitude if value >= 0 else center
        color = "#4ade80" if value >= 0 else "#fb7185"
        bars.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width * 0.76:.2f}" '
            f'height="{magnitude:.2f}" fill="{color}" rx="2"/>'
        )
    return (
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{_escape_html(label)}"><line class="chart-grid" x1="{padding}" '
        f'y1="{center}" x2="{width - padding}" y2="{center}"/>{"".join(bars)}</svg>'
    )


def _normalize_equity(eq_series: Sequence[object]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for index, value in enumerate(eq_series):
        if not isinstance(value, Mapping):
            raise ValidationError("eq_series points must be mappings", context={"index": index})
        output.append(
            {
                "time": value.get("time"),
                "t": iso_milliseconds(value.get("time")),
                "equity": _number(value.get("equity"), "eq_series equity"),
            }
        )
    return output


def _daily_pnl(points: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    by_day: dict[str, dict[str, float]] = {}
    for point in points:
        date = str(point["t"])[:10]
        time = _number(point["time"], "eq_series time")
        equity = _number(point["equity"], "eq_series equity")
        record = by_day.get(date)
        if record is None:
            by_day[date] = {
                "open": equity,
                "close": equity,
                "firstTime": time,
                "lastTime": time,
            }
        else:
            if time < record["firstTime"]:
                record["firstTime"] = time
                record["open"] = equity
            if time >= record["lastTime"]:
                record["lastTime"] = time
                record["close"] = equity
    return [
        {"date": date, "pnl": record["close"] - record["open"]}
        for date, record in sorted(by_day.items())
    ]


def _payload(eq_series: Sequence[object], replay: Mapping[str, object] | None) -> dict[str, Any]:
    points = _normalize_equity(eq_series)
    peak = _number(points[0]["equity"], "eq_series equity")
    drawdown: list[dict[str, object]] = []
    for point in points:
        equity = _number(point["equity"], "eq_series equity")
        peak = max(peak, equity)
        drawdown.append({"t": point["t"], "value": (equity - peak) / peak if peak > 0 else 0})
    frames = replay.get("frames", []) if replay else []
    events = replay.get("events", []) if replay else []
    if not isinstance(frames, Sequence) or isinstance(frames, (str, bytes, bytearray)):
        frames = []
    if not isinstance(events, Sequence) or isinstance(events, (str, bytes, bytearray)):
        events = []
    normalized_replay = {"frames": list(frames), "events": list(events)}
    return {
        "eqSeries": [{"t": point["t"], "equity": point["equity"]} for point in points],
        "drawdown": drawdown,
        "dailyPnl": _daily_pnl(points),
        "replay": normalized_replay,
        "hasReplay": bool(normalized_replay["frames"]),
    }


def _metric_cards(metrics: Mapping[str, object]) -> str:
    cards = (
        (
            "Net Return",
            _fmt_pct(metrics.get("returnPct", 0)),
            f"PnL {_fmt(metrics.get('totalPnL', 0))}",
        ),
        (
            "Win Rate",
            _fmt_pct(metrics.get("winRate", 0), 1),
            f"{metrics.get('trades', 0)} completed positions",
        ),
        (
            "Profit Factor",
            _fmt(metrics.get("profitFactor", 0)),
            f"Avg R {_fmt(metrics.get('avgR', 0))}",
        ),
        (
            "Drawdown",
            _fmt_pct(metrics.get("maxDrawdownPct", 0)),
            f"Calmar {_fmt(metrics.get('calmar', 0))}",
        ),
    )
    return "".join(
        '<article class="card"><div class="metric-card__label">'
        f'{_escape_html(label)}</div><div class="metric-card__value">{_escape_html(value)}</div>'
        f'<div class="metric-card__note">{_escape_html(note)}</div></article>'
        for label, value, note in cards
    )


def _summary_rows(metrics: Mapping[str, object]) -> str:
    rows = (
        ("Trades", metrics.get("trades", 0)),
        ("Win rate", _fmt_pct(metrics.get("winRate", 0), 1)),
        ("Profit factor", _fmt(metrics.get("profitFactor", 0))),
        ("Expectancy / trade", _fmt(metrics.get("expectancy", 0))),
        ("Total R", _fmt(metrics.get("totalR", 0))),
        ("Avg R / trade", _fmt(metrics.get("avgR", 0))),
        ("Max drawdown", _fmt_pct(metrics.get("maxDrawdownPct", 0))),
        ("Exposure", _fmt_pct(metrics.get("exposurePct", 0), 1)),
        ("Avg hold (min)", _fmt(metrics.get("avgHoldMin", 0), 1)),
        ("Daily Sharpe", _fmt(metrics.get("sharpeDaily", 0))),
    )
    return "".join(
        f"<tr><th>{_escape_html(label)}</th><td>{_escape_html(value)}</td></tr>"
        for label, value in rows
    )


def _breakdown_rows(metrics: Mapping[str, object]) -> str:
    rows = (
        (
            "Long",
            f"{_nested(metrics, 'long', 'trades')} trades, "
            f"{_fmt_pct(_nested(metrics, 'long', 'winRate'), 1)} win, "
            f"avg R {_fmt(_nested(metrics, 'long', 'avgR'))}",
        ),
        (
            "Short",
            f"{_nested(metrics, 'short', 'trades')} trades, "
            f"{_fmt_pct(_nested(metrics, 'short', 'winRate'), 1)} win, "
            f"avg R {_fmt(_nested(metrics, 'short', 'avgR'))}",
        ),
        (
            "R p50 / p90",
            f"{_fmt(_nested(metrics, 'rDist', 'p50'))} / {_fmt(_nested(metrics, 'rDist', 'p90'))}",
        ),
        (
            "Hold p50 / p90",
            f"{_fmt(_nested(metrics, 'holdDistMin', 'p50'), 1)} / "
            f"{_fmt(_nested(metrics, 'holdDistMin', 'p90'), 1)} min",
        ),
    )
    return "".join(
        f"<tr><th>{_escape_html(label)}</th><td>{_escape_html(value)}</td></tr>"
        for label, value in rows
    )


def _position_rows(positions: Sequence[object]) -> str:
    if not positions:
        return '<tr><td class="muted" colspan="7">No completed positions</td></tr>'
    rows: list[str] = []
    for value in reversed(positions[-25:]):
        if not isinstance(value, Mapping):
            continue
        exit_data = value.get("exit")
        exit_mapping = exit_data if isinstance(exit_data, Mapping) else {}
        entry = value.get("entryFill")
        if entry is None:
            entry = value.get("entry")
        cells = (
            iso_milliseconds(value.get("openTime")),
            value.get("side", ""),
            _fmt(entry, 4),
            _fmt(exit_mapping.get("price"), 4),
            exit_mapping.get("reason", "—"),
            _fmt(exit_mapping.get("pnl")),
            f"{_fmt(value.get('mfeR', 0))} / {_fmt(value.get('maeR', 0))}",
        )
        rows.append("<tr>" + "".join(f"<td>{_escape_html(cell)}</td>" for cell in cells) + "</tr>")
    return "".join(rows) or '<tr><td class="muted" colspan="7">No completed positions</td></tr>'


def render_html_report(
    *,
    symbol: object,
    interval: object,
    range_: object,
    metrics: Mapping[str, object],
    eq_series: Sequence[object],
    replay: Mapping[str, object] | None = None,
    positions: Sequence[object] = (),
) -> str:
    """Return a deterministic, self-contained offline dashboard."""
    if not eq_series:
        raise ValidationError("render_html_report requires a populated eq_series")
    payload = _payload(eq_series, replay)
    report_json = strict_json_dumps(payload, html_safe=True)
    title = f"{symbol} {interval} ({range_})"
    equity_values = [float(point["equity"]) for point in payload["eqSeries"]]
    drawdown_values = [float(point["value"]) for point in payload["drawdown"]]
    daily_values = [float(point["pnl"]) for point in payload["dailyPnl"]]
    replay_class = "" if payload["hasReplay"] else " is-hidden"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_escape_html(title)} backtest</title><style>{_CSS}</style></head><body>
<main class="report-shell"><header class="hero"><div><p class="eyebrow">Trading Engine Report</p>
<h1>{_escape_html(title)} backtest</h1><p class="muted">Start {_escape_html(_fmt(metrics.get("startEquity", 0)))}
 • End {_escape_html(_fmt(metrics.get("finalEquity", 0)))}</p></div>
<div class="pill">Return {_escape_html(_fmt_pct(metrics.get("returnPct", 0)))} • Max DD
 {_escape_html(_fmt_pct(metrics.get("maxDrawdownPct", 0)))}</div></header>
<section class="metric-grid">{_metric_cards(metrics)}</section><section class="panel-grid">
<article class="panel panel--wide"><h2>Equity Curve</h2><p class="muted">Realized equity through the run.</p>
{_line_svg(equity_values, color="#64e0c1", label="Equity curve")}</article>
<article class="panel"><h2>Summary</h2><table class="data-table"><tbody>{_summary_rows(metrics)}</tbody></table></article>
<article class="panel"><h2>Drawdown</h2>{_line_svg(drawdown_values, color="#fb7185", label="Drawdown")}</article>
<article class="panel"><h2>Daily PnL</h2>{_bar_svg(daily_values, label="Daily profit and loss")}</article>
<article class="panel"><h2>Breakdown</h2><table class="data-table"><tbody>{_breakdown_rows(metrics)}</tbody></table></article>
<article class="panel panel--wide{replay_class}"><h2>Executions</h2><p class="muted">Replay data embedded for offline inspection.</p></article>
<article class="panel panel--wide"><h2>Recent Positions</h2><div class="table-wrap"><table class="data-table">
<thead><tr><th>Opened</th><th>Side</th><th>Entry</th><th>Exit</th><th>Reason</th><th>PnL</th><th>MFE / MAE</th></tr></thead>
<tbody>{_position_rows(positions)}</tbody></table></div></article></section></main>
<script id="report-data" type="application/json">{report_json}</script></body></html>"""


def export_html_report(
    *,
    symbol: object,
    interval: object,
    range_: object,
    metrics: Mapping[str, object],
    eq_series: Sequence[object],
    replay: Mapping[str, object] | None = None,
    positions: Sequence[object] = (),
    out_dir: str | Path = "output",
) -> Path | None:
    """Atomically write a self-contained HTML dashboard."""
    if not eq_series:
        return None
    path = Path(out_dir) / (
        f"report-{safe_segment(symbol)}-{safe_segment(interval)}-{safe_segment(range_)}.html"
    )
    return atomic_write_text(
        path,
        render_html_report(
            symbol=symbol,
            interval=interval,
            range_=range_,
            metrics=metrics,
            eq_series=eq_series,
            replay=replay,
            positions=positions,
        ),
    )


__all__ = ["export_html_report", "render_html_report"]
