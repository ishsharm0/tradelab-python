"""Contract tests for the generated JavaScript parity oracle."""

from __future__ import annotations

import json
from pathlib import Path

EXPECTED_FIXTURES = {
    "ta": "ta.json",
    "metrics": "metrics.json",
    "research": "research.json",
    "backtest": "backtest.json",
    "ticks": "ticks.json",
    "financing": "financing.json",
    "walkForward": "walkForward.json",
    "portfolio": "portfolio.json",
}


def test_manifest_identifies_the_javascript_oracle(load_fixture: object) -> None:
    """The manifest pins the reference implementation and deterministic fixture set."""
    manifest = load_fixture("manifest.json")

    assert isinstance(manifest, dict)
    assert manifest["sourceVersion"] == "1.3.1"
    assert manifest["seed"] == 42
    assert manifest["fixtures"] == EXPECTED_FIXTURES


def test_declared_fixture_files_exist_and_are_valid_json(
    fixture_dir: Path, load_fixture: object
) -> None:
    """Every manifest-declared oracle payload exists and is independently parseable."""
    manifest = load_fixture("manifest.json")

    assert isinstance(manifest, dict)
    assert isinstance(manifest["fixtures"], dict)
    for filename in manifest["fixtures"].values():
        path = fixture_dir / filename
        assert path.is_file(), f"missing declared fixture: {filename}"
        assert json.loads(path.read_text(encoding="utf-8")) is not None


def test_ta_fixture_covers_every_exported_indicator(load_fixture: object) -> None:
    """TA parity data includes a direct oracle result for every public indicator."""
    fixture = load_fixture("ta.json")

    assert isinstance(fixture, dict)
    assert isinstance(fixture["output"], dict)
    assert {
        "ema",
        "atr",
        "rsi",
        "macd",
        "stochastic",
        "bollinger",
        "donchian",
        "keltner",
        "supertrend",
        "vwap",
        "swingHighAt9",
        "swingLowAt8",
        "fvgAt3",
        "lastSwingAt12",
        "structureAt12",
    }.issubset(fixture["output"])
