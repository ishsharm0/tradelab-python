"""Behavior, persistence, and validation tests for the synchronous research store."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tradelab.errors import ValidationError
from tradelab.research.store import ResearchStore


class FixedClock:
    """A deterministic timestamp source for store tests."""

    def __init__(self) -> None:
        self._timestamps = iter(
            [
                datetime(2024, 1, 1, tzinfo=UTC),
                datetime(2024, 1, 2, tzinfo=UTC),
                datetime(2024, 1, 3, tzinfo=UTC),
            ]
        )

    def __call__(self) -> datetime:
        return next(self._timestamps)


def test_store_open_log_recall_close_and_reopen(tmp_path: Path) -> None:
    store = ResearchStore(tmp_path, clock=FixedClock())

    assert (
        store.open("mean-reversion", "Validate a signal")["created_at"]
        == "2024-01-01T00:00:00+00:00"
    )
    entry = store.log(
        "mean-reversion",
        hypothesis="test",
        params={"lookback": 10},
        metrics={"sharpe": 1.5},
        verdict={"overfit": True},
    )
    assert entry["at"] == "2024-01-02T00:00:00+00:00"
    recalled = ResearchStore(tmp_path).recall("mean-reversion")
    assert recalled["goal"] == "Validate a signal"
    assert recalled["best_sharpe"] == {"sharpe": 1.5, "params": {"lookback": 10}}
    assert (
        recalled["summary"]
        == 'Best Sharpe so far: 1.50 via {"lookback":10}. 1 of 1 flagged overfit.'
    )
    assert store.close("mean-reversion")["closed_at"] == "2024-01-03T00:00:00+00:00"


def test_store_rejects_traversal_ids(tmp_path: Path) -> None:
    store = ResearchStore(tmp_path)

    for record_id in ("../unsafe", "nested/id", "", "unsafe id", "café"):
        with pytest.raises(ValidationError):
            store.open(record_id)


def test_store_persists_valid_json_atomically_and_cleans_temp_on_replace_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ResearchStore(tmp_path)
    store.open("atomic")
    assert json.loads((tmp_path / "atomic.json").read_text())["id"] == "atomic"

    def fail_replace(
        source: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        destination: str | bytes | os.PathLike[str] | os.PathLike[bytes],
    ) -> None:
        raise OSError("replace interrupted")

    monkeypatch.setattr("tradelab.research.store.os.replace", fail_replace)
    with pytest.raises(OSError, match="replace interrupted"):
        store.log("atomic", hypothesis="will not persist")
    assert not list(tmp_path.glob(".atomic.*.tmp"))
    record = ResearchStore(tmp_path).load("atomic")
    assert record is not None
    assert record["entries"] == []


def test_store_load_missing_and_recall_empty_use_source_defaults(tmp_path: Path) -> None:
    store = ResearchStore(tmp_path)

    assert store.load("missing") is None
    assert store.recall("missing") == {
        "goal": "",
        "entries": [],
        "best_sharpe": None,
        "summary": "No entries logged yet.",
    }
