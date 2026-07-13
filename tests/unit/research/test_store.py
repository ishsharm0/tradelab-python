"""Behavior, persistence, and validation tests for the synchronous research store."""

from __future__ import annotations

import json
import math
import multiprocessing
import os
import threading
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tradelab.errors import ValidationError
from tradelab.research.store import ResearchStore


def _write_entries_in_process(directory: str, prefix: str, start: object) -> None:
    """Exercise the public store from an independently spawned interpreter."""
    start.wait()  # type: ignore[attr-defined]
    store = ResearchStore(directory)
    for index in range(20):
        store.log("shared-process", hypothesis=f"{prefix}-{index}")


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


def test_store_rejects_strict_json_values_without_changing_the_original_file(
    tmp_path: Path,
) -> None:
    store = ResearchStore(tmp_path)
    store.open("strict")
    path = tmp_path / "strict.json"
    original = path.read_bytes()

    with pytest.raises(ValidationError):
        store.log("strict", params={"nested": [1, {"bad": math.nan}]})
    with pytest.raises(ValidationError):
        store.log("strict", metrics={"bad": object()})
    with pytest.raises(ValidationError):
        store.log("strict", verdict={"nested": [math.inf]})

    assert path.read_bytes() == original


def test_store_rejects_integers_outside_portable_binary64_range(tmp_path: Path) -> None:
    store = ResearchStore(tmp_path)
    store.open("portable-integer")
    path = tmp_path / "portable-integer.json"
    original = path.read_bytes()

    with pytest.raises(ValidationError):
        store.log("portable-integer", metrics={"sharpe": 10**10_000})
    with pytest.raises(ValidationError):
        store.log("portable-integer", params={"integer": 2**53})
    assert path.read_bytes() == original

    store.log("portable-integer", params={"integer": 2**53 - 1})

    assert path.read_bytes() != original


def test_store_normalizes_nested_json_values_and_requires_boolean_overfit(tmp_path: Path) -> None:
    store = ResearchStore(tmp_path)
    store.log(
        "typed",
        params={"nested": ({"values": (1, 2)},)},
        metrics={"series": [1.0, None]},
        verdict={"overfit": True, "notes": ("reviewed",)},
    )

    record = store.load("typed")
    assert record is not None
    assert record["entries"][0]["params"] == {"nested": [{"values": [1, 2]}]}
    assert record["entries"][0]["verdict"] == {"overfit": True, "notes": ["reviewed"]}

    with pytest.raises(ValidationError):
        store.log("typed", verdict={"overfit": 1})

    path = tmp_path / "typed.json"
    payload = json.loads(path.read_text())
    payload["entries"][0]["verdict"] = {"overfit": "yes"}
    path.write_text(json.dumps(payload))
    with pytest.raises(ValidationError):
        store.load("typed")


def test_store_rejects_corrupt_or_aliased_records_without_redirecting_writes(
    tmp_path: Path,
) -> None:
    store = ResearchStore(tmp_path)
    path = tmp_path / "requested.json"
    path.write_text(
        json.dumps(
            {
                "id": "other",
                "goal": "must stay put",
                "created_at": "2024-01-01T00:00:00+00:00",
                "closed_at": None,
                "entries": [],
            }
        )
    )
    original = path.read_bytes()

    with pytest.raises(ValidationError, match="requested"):
        store.load("requested")
    with pytest.raises(ValidationError, match="requested"):
        store.log("requested", hypothesis="must not write other.json")
    assert path.read_bytes() == original
    assert not (tmp_path / "other.json").exists()

    (tmp_path / "corrupt.json").write_text("{")
    with pytest.raises(ValidationError, match="corrupt"):
        store.load("corrupt")


def test_store_serializes_two_synchronized_writers_without_leaking_lock_files(
    tmp_path: Path,
) -> None:
    first = ResearchStore(tmp_path)
    second = ResearchStore(tmp_path)
    start = threading.Barrier(3)
    errors: list[BaseException] = []

    def write_entries(store: ResearchStore, prefix: str) -> None:
        try:
            start.wait()
            for index in range(25):
                store.log("shared", hypothesis=f"{prefix}-{index}")
        except BaseException as error:  # pragma: no cover - assertion below reports it
            errors.append(error)

    threads = [
        threading.Thread(target=write_entries, args=(first, "first")),
        threading.Thread(target=write_entries, args=(second, "second")),
    ]
    for thread in threads:
        thread.start()
    start.wait()
    for thread in threads:
        thread.join()

    assert not errors
    record = ResearchStore(tmp_path).load("shared")
    assert record is not None
    assert {entry["hypothesis"] for entry in record["entries"]} == {
        f"{prefix}-{index}" for prefix in ("first", "second") for index in range(25)
    }
    assert not list(tmp_path.glob("*.lock"))


def test_store_serializes_independent_processes_without_lost_updates(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    start = context.Event()
    processes = [
        context.Process(
            target=_write_entries_in_process,
            args=(str(tmp_path), prefix, start),
        )
        for prefix in ("first", "second")
    ]
    for process in processes:
        process.start()
    start.set()
    for process in processes:
        process.join(timeout=15)
        assert process.exitcode == 0

    record = ResearchStore(tmp_path).load("shared-process")
    assert record is not None
    assert {entry["hypothesis"] for entry in record["entries"]} == {
        f"{prefix}-{index}" for prefix in ("first", "second") for index in range(20)
    }
    assert not list(tmp_path.glob("*.lock"))
