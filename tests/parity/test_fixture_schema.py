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

EXPECTED_SOURCE_PATHS = {
    "annualize": "src/metrics/annualize.js",
    "backtest": "src/engine/backtest.js",
    "financing": "src/engine/execution.js",
    "finite": "src/metrics/finite.js",
    "metrics": "src/metrics/buildMetrics.js",
    "portfolio": "src/engine/portfolio.js",
    "research": "src/research/index.js",
    "ta": "src/ta/index.js",
    "ticks": "src/engine/backtestTicks.js",
    "walkForward": "src/engine/walkForward.js",
}

REQUIRED_INPUT_KEYS = {
    "ta.json": {"calls", "candles", "closes"},
    "metrics.json": {"annualize", "buildMetrics", "finite"},
    "research.json": {"cpcv", "deflatedSharpe", "monteCarlo", "pbo", "stats"},
    "backtest.json": {"options", "signal"},
    "ticks.json": {"options", "signal"},
    "financing.json": {"fundingEvents", "long", "short"},
    "walkForward.json": {"options", "signalFactory"},
    "portfolio.json": {"options"},
}

EXPECTED_TA_CALLS = {
    "atr",
    "bollinger",
    "donchian",
    "ema",
    "fvgAt3",
    "keltner",
    "lastSwingAt12",
    "macd",
    "rsi",
    "stochastic",
    "structureAt12",
    "supertrend",
    "swingHighAt9",
    "swingLowAt8",
    "vwap",
}


def test_manifest_identifies_the_javascript_oracle(load_fixture: object) -> None:
    """The manifest pins the reference implementation and deterministic fixture set."""
    manifest = load_fixture("manifest.json")

    assert isinstance(manifest, dict)
    assert manifest["sourceVersion"] == "1.3.1"
    assert manifest["seed"] == 42
    assert manifest["fixtures"] == EXPECTED_FIXTURES
    assert manifest["sourcePaths"] == EXPECTED_SOURCE_PATHS


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


def test_fixture_payloads_preserve_replayable_inputs_and_outputs(load_fixture: object) -> None:
    """Fixtures retain every call's serializable input and its JavaScript output."""
    for filename, required_keys in REQUIRED_INPUT_KEYS.items():
        payload = load_fixture(filename)

        assert isinstance(payload, dict)
        assert isinstance(payload.get("input"), dict)
        assert required_keys.issubset(payload["input"])
        assert isinstance(payload.get("output"), dict)

    ta_payload = load_fixture("ta.json")
    assert isinstance(ta_payload, dict)
    assert set(ta_payload["input"]["calls"]) == EXPECTED_TA_CALLS


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
