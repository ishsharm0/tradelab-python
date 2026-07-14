"""Plain-English and Markdown backtest summaries."""

from __future__ import annotations

import math
from collections.abc import Mapping


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _plain_number(value: float) -> str:
    return str(int(value)) if value.is_integer() else str(value)


def _pct(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}%"


def summarize(
    metrics: Mapping[str, object] | None = None,
    *,
    verdict: Mapping[str, object] | None = None,
) -> str:
    """Render the JavaScript-compatible one-paragraph metrics summary."""
    values = metrics or {}
    trades_value = _finite(values.get("trades"))
    trades = trades_value if trades_value is not None else 0.0
    win_rate = _finite(values.get("winRate"))
    drawdown = _finite(values.get("maxDrawdownPct"))
    if drawdown is None:
        fallback_drawdown = _finite(values.get("maxDrawdown"))
        drawdown = fallback_drawdown * 100 if fallback_drawdown is not None else None
    total_return = _finite(values.get("totalReturnPct"))
    sharpe = _finite(values.get("sharpe"))

    if trades == 0:
        return "Ran with 0 trades, so there is nothing to evaluate yet."

    parts = [f"Made {_plain_number(trades)} trades"]
    if win_rate is not None:
        rounded_win = math.floor(win_rate * 100 + 0.5)
        parts.append(f"won {rounded_win}% of them")
    if total_return is not None:
        parts.append(f"for a {_pct(total_return)} total return")
    if drawdown is not None:
        parts.append(f"with a worst drawdown of {_pct(drawdown)}")

    text = ", ".join(parts)
    if sharpe is not None:
        text += f" (Sharpe {sharpe:.2f})"
    text += "."

    if verdict and verdict.get("overfit"):
        note = verdict.get("note")
        suffix = f" ({note})" if note else ""
        text += f" Caution: robustness checks flag this result as likely overfit{suffix}."
    return text


def _escape_markdown(value: object) -> str:
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("\r", " ")
        .replace("\n", " ")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _metric_text(value: object) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "n/a"
        return format(value, ".12g")
    return str(value)


def render_markdown_report(
    *,
    symbol: object,
    interval: object,
    range_: object,
    metrics: Mapping[str, object],
    verdict: Mapping[str, object] | None = None,
) -> str:
    """Render a deterministic Markdown summary and scalar metrics table."""
    title = " ".join(
        (_escape_markdown(symbol), _escape_markdown(interval), f"({_escape_markdown(range_)})")
    )
    scalar_rows = [
        f"| {_escape_markdown(key)} | {_escape_markdown(_metric_text(value))} |"
        for key, value in sorted(metrics.items())
        if not isinstance(value, (Mapping, list, tuple))
    ]
    rows = "\n".join(scalar_rows) if scalar_rows else "| n/a | n/a |"
    return (
        f"# {title} backtest\n\n"
        f"{summarize(metrics, verdict=verdict)}\n\n"
        "## Metrics\n\n"
        "| Metric | Value |\n"
        "| --- | ---: |\n"
        f"{rows}\n"
    )


__all__ = ["render_markdown_report", "summarize"]
