"""Synchronous, atomically persisted research records."""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Callable, Mapping
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypedDict

from tradelab.errors import ValidationError

_DEFAULT_DIRECTORY = Path(".tradelab/research")
_SAFE_ID = re.compile(r"^[\w.-]+$", re.ASCII)


class ResearchEntry(TypedDict):
    """One recorded research hypothesis and its measured outcome."""

    at: str
    hypothesis: str
    params: dict[str, Any]
    metrics: dict[str, Any]
    verdict: dict[str, Any] | None


class ResearchRecord(TypedDict):
    """Persisted record for a named research investigation."""

    id: str
    goal: str
    created_at: str
    closed_at: str | None
    entries: list[ResearchEntry]


class BestSharpe(TypedDict):
    """The highest finite Sharpe and the corresponding parameter set."""

    sharpe: float
    params: dict[str, Any]


class ResearchRecall(TypedDict):
    """A compact record recap suitable for ordinary Python callers."""

    goal: str
    entries: list[ResearchEntry]
    best_sharpe: BestSharpe | None
    summary: str


def _timestamp(clock: Callable[[], datetime]) -> str:
    value = clock()
    if not isinstance(value, datetime):
        raise ValidationError("clock must return a datetime", context={"clock_value": value})
    return value.isoformat()


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _validate_id(record_id: str) -> str:
    if not isinstance(record_id, str) or not _SAFE_ID.fullmatch(record_id):
        raise ValidationError("invalid research id", context={"id": record_id})
    return record_id


def _entry_from_mapping(value: object) -> ResearchEntry | None:
    if not isinstance(value, Mapping):
        return None
    at = value.get("at")
    hypothesis = value.get("hypothesis")
    params = value.get("params")
    metrics = value.get("metrics")
    verdict = value.get("verdict")
    if not isinstance(at, str) or not isinstance(hypothesis, str):
        return None
    if not isinstance(params, dict) or not isinstance(metrics, dict):
        return None
    if verdict is not None and not isinstance(verdict, dict):
        return None
    return {
        "at": at,
        "hypothesis": hypothesis,
        "params": params,
        "metrics": metrics,
        "verdict": verdict,
    }


def _record_from_mapping(value: object) -> ResearchRecord | None:
    if not isinstance(value, Mapping):
        return None
    record_id = value.get("id")
    goal = value.get("goal")
    created_at = value.get("created_at")
    closed_at = value.get("closed_at")
    entries = value.get("entries")
    if (
        not isinstance(record_id, str)
        or not isinstance(goal, str)
        or not isinstance(created_at, str)
    ):
        return None
    if closed_at is not None and not isinstance(closed_at, str):
        return None
    if not isinstance(entries, list):
        return None
    normalized_entries: list[ResearchEntry] = []
    for entry in entries:
        normalized = _entry_from_mapping(entry)
        if normalized is None:
            return None
        normalized_entries.append(normalized)
    return {
        "id": record_id,
        "goal": goal,
        "created_at": created_at,
        "closed_at": closed_at,
        "entries": normalized_entries,
    }


def best_sharpe(entries: list[ResearchEntry]) -> BestSharpe | None:
    """Return the first entry with the greatest finite ``metrics.sharpe`` value."""
    best: BestSharpe | None = None
    for entry in entries:
        candidate = entry["metrics"].get("sharpe")
        if isinstance(candidate, bool) or not isinstance(candidate, (int, float)):
            continue
        sharpe = float(candidate)
        if not math_isfinite(sharpe):
            continue
        if best is None or sharpe > best["sharpe"]:
            best = {"sharpe": sharpe, "params": entry["params"]}
    return best


def math_isfinite(value: float) -> bool:
    """Keep finite-number validation local without importing a broad numeric stack."""
    return value != float("inf") and value != float("-inf") and value == value


class ResearchStore:
    """Pathlib-backed synchronous store for durable research records.

    Records are written through a same-directory temporary file, flushed and
    fsynced before being atomically replaced at their final path.
    """

    def __init__(
        self,
        directory: str | Path = _DEFAULT_DIRECTORY,
        *,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        self.directory = Path(directory)
        self._clock = clock

    def _path_for(self, record_id: str) -> Path:
        return self.directory / f"{_validate_id(record_id)}.json"

    def _new_record(self, record_id: str, goal: str = "") -> ResearchRecord:
        return {
            "id": _validate_id(record_id),
            "goal": goal,
            "created_at": _timestamp(self._clock),
            "closed_at": None,
            "entries": [],
        }

    def _save(self, record: ResearchRecord) -> ResearchRecord:
        path = self._path_for(record["id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path: Path | None = None
        try:
            descriptor, raw_temp_path = tempfile.mkstemp(
                prefix=f".{record['id']}.", suffix=".tmp", dir=path.parent, text=True
            )
            temp_path = Path(raw_temp_path)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(record, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, path)
        except BaseException:
            if temp_path is not None:
                with suppress(OSError):
                    temp_path.unlink(missing_ok=True)
            raise
        return record

    def load(self, record_id: str) -> ResearchRecord | None:
        """Load one record, returning ``None`` when its file is absent or unreadable."""
        path = self._path_for(record_id)
        try:
            with path.open(encoding="utf-8") as handle:
                return _record_from_mapping(json.load(handle))
        except (OSError, json.JSONDecodeError):
            return None

    def open(self, record_id: str, goal: str = "") -> ResearchRecord:
        """Load an existing record or persist a newly opened investigation."""
        record = self.load(record_id)
        return record if record is not None else self._save(self._new_record(record_id, goal))

    def log(
        self,
        record_id: str,
        *,
        hypothesis: str = "",
        params: Mapping[str, Any] | None = None,
        metrics: Mapping[str, Any] | None = None,
        verdict: Mapping[str, Any] | None = None,
    ) -> ResearchEntry:
        """Append and persist an entry, implicitly opening a missing record."""
        record = self.load(record_id)
        if record is None:
            record = self._new_record(record_id)
        entry: ResearchEntry = {
            "at": _timestamp(self._clock),
            "hypothesis": hypothesis,
            "params": dict(params or {}),
            "metrics": dict(metrics or {}),
            "verdict": dict(verdict) if verdict is not None else None,
        }
        record["entries"].append(entry)
        self._save(record)
        return entry

    def recall(self, record_id: str, limit: int = 10) -> ResearchRecall:
        """Return recent entries plus the source-compatible status summary."""
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
            raise ValidationError("limit must be a non-negative integer", context={"limit": limit})
        record = self.load(record_id)
        if record is None:
            return {
                "goal": "",
                "entries": [],
                "best_sharpe": None,
                "summary": "No entries logged yet.",
            }
        all_entries = record["entries"]
        best = best_sharpe(all_entries)
        flagged = sum(
            entry["verdict"].get("overfit") is True for entry in all_entries if entry["verdict"]
        )
        if not all_entries:
            summary = "No entries logged yet."
        elif best is None:
            summary = f"Best Sharpe so far: n/a. {flagged} of {len(all_entries)} flagged overfit."
        else:
            parameters = json.dumps(best["params"], separators=(",", ":"), ensure_ascii=False)
            summary = (
                f"Best Sharpe so far: {best['sharpe']:.2f} via {parameters}. "
                f"{flagged} of {len(all_entries)} flagged overfit."
            )
        return {
            "goal": record["goal"],
            "entries": all_entries[-limit:] if limit else [],
            "best_sharpe": best,
            "summary": summary,
        }

    def summary(self, record_id: str) -> str:
        """Return the concise source-compatible summary for a record."""
        return self.recall(record_id)["summary"]

    def best_sharpe(self, record_id: str) -> BestSharpe | None:
        """Return the record's best finite Sharpe result, if any."""
        record = self.load(record_id)
        return None if record is None else best_sharpe(record["entries"])

    def close(self, record_id: str) -> ResearchRecord:
        """Timestamp and persist record closure, implicitly creating it when missing."""
        record = self.load(record_id)
        if record is None:
            record = self._new_record(record_id)
        record["closed_at"] = _timestamp(self._clock)
        return self._save(record)


def create_research_store(
    directory: str | Path = _DEFAULT_DIRECTORY, *, clock: Callable[[], datetime] = _utc_now
) -> ResearchStore:
    """Create a synchronous :class:`ResearchStore` with optional deterministic clock."""
    return ResearchStore(directory, clock=clock)
