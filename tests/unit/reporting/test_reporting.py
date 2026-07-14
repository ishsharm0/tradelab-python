"""Reporting exporters, summaries, and offline dashboard contracts."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

import tradelab.reporting._io as reporting_io
from tradelab.errors import ValidationError
from tradelab.reporting import (
    export_backtest_artifacts,
    export_html_report,
    export_markdown_report,
    export_metrics_json,
    export_trades_csv,
    render_html_report,
    render_markdown_report,
    summarize,
)

START = 1_735_689_600_000


def _metrics() -> dict[str, Any]:
    return {
        "trades": 3,
        "winRate": 2 / 3,
        "profitFactor": 1.8,
        "expectancy": 42.5,
        "totalR": 3.4,
        "avgR": 1.13,
        "maxDrawdownPct": 0.08,
        "exposurePct": 0.25,
        "avgHoldMin": 35,
        "sharpeDaily": 1.2,
        "returnPct": 0.12,
        "totalPnL": 120,
        "calmar": 1.5,
        "startEquity": 1000,
        "finalEquity": 1120,
        "long": {"trades": 2, "winRate": 0.5, "avgR": 0.8},
        "short": {"trades": 1, "winRate": 1, "avgR": 1.8},
        "rDist": {"p50": 1, "p90": 2.1},
        "holdDistMin": {"p50": 20, "p90": 60},
    }


def _trade(*, reason: str = "take_profit") -> dict[str, Any]:
    return {
        "openTime": START,
        "side": "long",
        "entry": 100,
        "entryFill": 100.25,
        "stop": 98,
        "takeProfit": 104,
        "size": 2,
        "_initRisk": 2.25,
        "mfeR": 1.2345,
        "maeR": -0.3456,
        "adds": 1,
        "entryATR": 1.5,
        "exit": {
            "time": START + 86_400_000,
            "price": 103.25,
            "reason": reason,
            "pnl": 6.0,
            "exitATR": 1.75,
        },
    }


def _result() -> dict[str, Any]:
    return {
        "symbol": "DEMO",
        "interval": "1d",
        "range": "1y",
        "metrics": _metrics(),
        "trades": [_trade()],
        "positions": [_trade()],
        "eqSeries": [
            {"time": START, "equity": 1000},
            {"time": START + 43_200_000, "equity": 1050},
            {"time": START + 86_400_000, "equity": 1120},
        ],
        "replay": {"frames": [], "events": []},
    }


def test_summarize_matches_javascript_contract_and_overfit_caveat() -> None:
    assert summarize(
        {
            "trades": 23,
            "winRate": 0.52,
            "maxDrawdownPct": 8.1,
            "totalReturnPct": 14.2,
            "sharpe": 1.3,
        }
    ) == (
        "Made 23 trades, won 52% of them, for a 14.2% total return, "
        "with a worst drawdown of 8.1% (Sharpe 1.30)."
    )
    assert summarize(
        {
            "trades": 5,
            "winRate": 0.8,
            "maxDrawdown": 0.03,
            "totalReturnPct": 40,
            "sharpe": 2.5,
        },
        verdict={"overfit": True, "note": "PBO high"},
    ).endswith("likely overfit (PBO high).")
    assert summarize({"trades": 0}) == ("Ran with 0 trades, so there is nothing to evaluate yet.")


def test_export_trades_csv_matches_javascript_columns_and_escapes_fields(tmp_path: Path) -> None:
    output = export_trades_csv(
        [_trade(reason='take, "profit"')],
        symbol="BRK/B",
        interval="1 d",
        range_="a:b",
        out_dir=tmp_path,
    )

    assert output == tmp_path / "trades-BRK_B-1_d-a_b.csv"
    assert output is not None
    rows = output.read_text(encoding="utf-8").splitlines()
    assert rows[0] == (
        "time_open,time_close,side,entry,stop,takeProfit,exit,reason,size,pnl,R,"
        "mfeR,maeR,adds,entryATR,exitATR"
    )
    assert rows[1] == (
        "2025-01-01T00:00:00.000Z,2025-01-02T00:00:00.000Z,long,100.000000,"
        '98.000000,104.000000,103.250000,"take, ""profit""",2,6.00,1.333,'
        "1.234,-0.346,1,1.500000,1.750000"
    )
    assert export_trades_csv([], out_dir=tmp_path / "unused") is None
    assert not (tmp_path / "unused").exists()


def test_export_metrics_json_is_strict_deterministic_and_atomic(tmp_path: Path) -> None:
    result = _result()
    output = export_metrics_json(result, out_dir=tmp_path)

    assert output.name == "metrics-DEMO-1d-1y.json"
    assert json.loads(output.read_text(encoding="utf-8")) == result["metrics"]
    assert output.read_text(encoding="utf-8").endswith("\n")

    original = output.read_text(encoding="utf-8")
    result["metrics"]["bad"] = float("nan")
    with pytest.raises(ValidationError, match="strict JSON"):
        export_metrics_json(result, out_dir=tmp_path)
    assert output.read_text(encoding="utf-8") == original
    assert not list(tmp_path.glob(f".{output.name}.*"))


def test_markdown_report_is_deterministic_and_escapes_active_markup(tmp_path: Path) -> None:
    metrics = {
        "trades": 2,
        "winRate": 0.5,
        "totalReturnPct": 4.2,
        "note": "a|b\n<script>",
    }
    markdown = render_markdown_report(symbol="<DEMO>", interval="1d", range_="1y", metrics=metrics)

    assert markdown.startswith("# &lt;DEMO&gt; 1d (1y) backtest\n")
    assert "Made 2 trades, won 50% of them, for a 4.2% total return." in markdown
    assert "a\\|b &lt;script&gt;" in markdown
    output = export_markdown_report(
        {"symbol": "<DEMO>", "interval": "1d", "range": "1y", "metrics": metrics},
        out_dir=tmp_path,
    )
    assert output.name == "report-_DEMO_-1d-1y.md"
    assert output.read_text(encoding="utf-8") == markdown


def test_render_html_report_is_offline_self_contained_and_script_safe() -> None:
    dangerous = "</script><img src=x>\u2028\u2029"
    html = render_html_report(
        symbol="DEMO<script>",
        interval="1d",
        range_="1y",
        metrics=_metrics(),
        eq_series=[
            {"time": START, "equity": 1000},
            {"time": START + 43_200_000, "equity": 1050},
            {"time": START + 86_400_000, "equity": 1025},
        ],
        replay={"frames": [{"t": START, "price": 10}], "events": [{"note": dangerous}]},
        positions=[_trade(reason=dangerous)],
    )

    assert "Trading Engine Report" in html
    assert "metric-card__value" in html
    assert 'id="report-data"' in html
    assert "<svg" in html
    assert "http://" not in html and "https://" not in html
    assert "<script src=" not in html
    assert "DEMO&lt;script&gt;" in html
    assert "</script><img" not in html
    assert "\\u003c/script>\\u003cimg src=x>\\u2028\\u2029" in html
    assert html == render_html_report(
        symbol="DEMO<script>",
        interval="1d",
        range_="1y",
        metrics=_metrics(),
        eq_series=[
            {"time": START, "equity": 1000},
            {"time": START + 43_200_000, "equity": 1050},
            {"time": START + 86_400_000, "equity": 1025},
        ],
        replay={"frames": [{"t": START, "price": 10}], "events": [{"note": dangerous}]},
        positions=[_trade(reason=dangerous)],
    )


def test_html_validation_and_export_contract(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="populated eq_series"):
        render_html_report(symbol="DEMO", interval="1d", range_="1y", metrics={}, eq_series=[])
    with pytest.raises(ValidationError, match="strict JSON"):
        render_html_report(
            symbol="DEMO",
            interval="1d",
            range_="1y",
            metrics={},
            eq_series=[{"time": START, "equity": 1000}],
            replay={"frames": [], "events": [{"value": float("inf")}]},
        )
    assert (
        export_html_report(
            symbol="DEMO", interval="1d", range_="1y", metrics={}, eq_series=[], out_dir=tmp_path
        )
        is None
    )
    output = export_html_report(
        symbol="DEMO",
        interval="1d",
        range_="1y",
        metrics=_metrics(),
        eq_series=_result()["eqSeries"],
        out_dir=tmp_path,
    )
    assert output is not None
    assert output.name == "report-DEMO-1d-1y.html"
    assert output.read_text(encoding="utf-8").startswith("<!doctype html>")


def test_export_backtest_artifacts_writes_all_formats_and_honors_csv_source(
    tmp_path: Path,
) -> None:
    result = _result()
    result["trades"] = [_trade(reason="trade-source")]
    result["positions"] = [_trade(reason="position-source")]

    outputs = export_backtest_artifacts(result, out_dir=tmp_path)
    assert set(outputs) == {"csv", "html", "markdown", "metrics"}
    assert all(path is not None and path.exists() for path in outputs.values())
    assert "position-source" in outputs["csv"].read_text(encoding="utf-8")

    trade_outputs = export_backtest_artifacts(
        result,
        out_dir=tmp_path / "trades",
        csv_source="trades",
        export_html=False,
        export_markdown=False,
        export_metrics=False,
    )
    assert "trade-source" in trade_outputs["csv"].read_text(encoding="utf-8")
    assert trade_outputs["html"] is None
    assert trade_outputs["markdown"] is None
    assert trade_outputs["metrics"] is None
    with pytest.raises(ValidationError, match="requires a backtest result"):
        export_backtest_artifacts(None)  # type: ignore[arg-type]


def test_atomic_replace_failure_preserves_existing_reporting_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "metrics-DEMO-1d-1y.json"
    output.write_text("existing", encoding="utf-8")
    monkeypatch.setattr(
        reporting_io.os,
        "replace",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("boom")),
    )

    with pytest.raises(ValidationError, match="Could not write report artifact"):
        export_metrics_json(_result(), out_dir=tmp_path)

    assert output.read_text(encoding="utf-8") == "existing"
    assert not list(tmp_path.glob(f".{output.name}.*"))


def test_reporting_safety_helpers_cover_javascript_names_dates_and_strict_json() -> None:
    assert reporting_io.safe_segment(None) == "undefined"
    assert reporting_io.safe_segment(True) == "true"
    assert reporting_io.safe_segment(False) == "false"
    assert reporting_io.safe_segment(-0.0) == "0"
    assert reporting_io.safe_segment(float("nan")) == "NaN"
    assert reporting_io.safe_segment(float("inf")) == "Infinity"
    assert reporting_io.safe_segment(float("-inf")) == "-Infinity"
    assert reporting_io.safe_segment(["a", None, {"x": 1}]) == "a___object_Object_"
    assert reporting_io.iso_milliseconds(datetime(2025, 1, 1, tzinfo=UTC)) == (
        "2025-01-01T00:00:00.000Z"
    )
    with pytest.raises(ValidationError, match="timestamp must be"):
        reporting_io.iso_milliseconds("bad")
    with pytest.raises(ValidationError, match="outside the supported range"):
        reporting_io.iso_milliseconds(10**30)

    assert json.loads(reporting_io.strict_json_dumps({"tuple": (1, True, None)})) == {
        "tuple": [1, True, None]
    }
    cyclic: list[object] = []
    cyclic.append(cyclic)
    with pytest.raises(ValidationError, match="without cycles"):
        reporting_io.strict_json_dumps(cyclic)
    with pytest.raises(ValidationError, match="JSON primitives"):
        reporting_io.strict_json_dumps({1, 2})


def test_export_validation_and_short_trade_zero_r_branches(tmp_path: Path) -> None:
    short = _trade()
    short.update({"side": "short", "_initRisk": 0, "entryATR": None})
    short["exit"].pop("exitATR")
    output = export_trades_csv([short], out_dir=tmp_path)
    assert output is not None
    assert ",0.000," in output.read_text(encoding="utf-8")

    with pytest.raises(ValidationError, match="trade must be a mapping"):
        export_trades_csv([1], out_dir=tmp_path)
    malformed = _trade()
    malformed["entry"] = "bad"
    with pytest.raises(ValidationError, match=r"trade\.entry must be numeric"):
        export_trades_csv([malformed], out_dir=tmp_path)
    with pytest.raises(ValidationError, match="with metrics"):
        export_metrics_json({}, out_dir=tmp_path)
    with pytest.raises(ValidationError, match="with metrics"):
        export_markdown_report({}, out_dir=tmp_path)

    result = _result()
    result.pop("positions")
    outputs = export_backtest_artifacts(
        result, out_dir=tmp_path / "fallback", export_html=False, export_markdown=False
    )
    assert outputs["csv"] is not None
    result["positions"] = "bad"
    with pytest.raises(ValidationError, match="CSV source must be a sequence"):
        export_backtest_artifacts(result, out_dir=tmp_path / "bad")


def test_html_normalizes_bad_replay_collections_and_empty_position_rows() -> None:
    html = render_html_report(
        symbol="DEMO",
        interval="1d",
        range_="1y",
        metrics={},
        eq_series=[
            {"time": START + 1_000, "equity": 1010},
            {"time": START, "equity": 1000},
            {"time": START + 2_000, "equity": 1005},
        ],
        replay={"frames": "bad", "events": 1},
        positions=[],
    )

    assert "No completed positions" in html
    assert "is-hidden" in html
    assert '"frames": []' in html
