"""Synchronous, atomically persisted research records."""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypedDict

from tradelab.errors import ValidationError

_DEFAULT_DIRECTORY = Path(".tradelab/research")
_SAFE_ID = re.compile(r"^[\w.-]+$", re.ASCII)
_LOCK_TIMEOUT_SECONDS = 5.0
_STALE_LOCK_SECONDS = 30.0
_LOCK_POLL_SECONDS = 0.01
_MAX_SAFE_INTEGER = (1 << 53) - 1
_thread_lock_guard = threading.Lock()


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


class _ThreadRecordLock:
    """A typed wrapper around a re-entrant lock shared by store instances."""

    def __init__(self) -> None:
        self._lock = threading.RLock()

    def acquire(self) -> None:
        self._lock.acquire()

    def release(self) -> None:
        self._lock.release()


_thread_locks: dict[Path, _ThreadRecordLock] = {}


def _thread_lock_for(path: Path) -> _ThreadRecordLock:
    key = path.absolute()
    with _thread_lock_guard:
        lock = _thread_locks.get(key)
        if lock is None:
            lock = _ThreadRecordLock()
            _thread_locks[key] = lock
        return lock


class _InterprocessRecordLock:
    """Portable lock-file ownership with bounded waiting and stale-file recovery."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._acquired = False

    def _owner_is_alive(self) -> bool:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            pid = payload.get("pid") if isinstance(payload, dict) else None
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return False
        if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False
        return True

    def _remove_stale_file(self) -> bool:
        try:
            modified_at = self.path.stat().st_mtime
        except FileNotFoundError:
            return False
        except OSError as error:
            raise ValidationError(
                "unable to inspect research lock",
                context={"lock_path": str(self.path), "error": str(error)},
            ) from error
        if time.time() - modified_at <= _STALE_LOCK_SECONDS or self._owner_is_alive():
            return False
        try:
            self.path.unlink()
        except FileNotFoundError:
            return False
        except OSError as error:
            raise ValidationError(
                "unable to remove stale research lock",
                context={"lock_path": str(self.path), "error": str(error)},
            ) from error
        return True

    def acquire(self) -> None:
        deadline = time.monotonic() + _LOCK_TIMEOUT_SECONDS
        self.path.parent.mkdir(parents=True, exist_ok=True)
        while True:
            try:
                descriptor = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            except FileExistsError:
                if self._remove_stale_file():
                    continue
                if time.monotonic() >= deadline:
                    raise ValidationError(
                        "timed out acquiring research record lock",
                        context={
                            "lock_path": str(self.path),
                            "timeout_seconds": _LOCK_TIMEOUT_SECONDS,
                        },
                    ) from None
                time.sleep(_LOCK_POLL_SECONDS)
                continue
            except OSError as error:
                raise ValidationError(
                    "unable to acquire research record lock",
                    context={"lock_path": str(self.path), "error": str(error)},
                ) from error
            try:
                metadata = json.dumps(
                    {"pid": os.getpid(), "created_at": time.time()}
                ).encode("utf-8")
                os.write(descriptor, metadata)
            except OSError as error:
                with suppress(OSError):
                    self.path.unlink()
                raise ValidationError(
                    "unable to initialize research record lock",
                    context={"lock_path": str(self.path), "error": str(error)},
                ) from error
            finally:
                os.close(descriptor)
            self._acquired = True
            return

    def release(self) -> None:
        if not self._acquired:
            return
        self._acquired = False
        try:
            self.path.unlink()
        except FileNotFoundError:
            return
        except OSError as error:
            raise ValidationError(
                "unable to release research record lock",
                context={"lock_path": str(self.path), "error": str(error)},
            ) from error


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


def _normalize_json_value(value: object, *, field: str, seen: set[int] | None = None) -> Any:
    """Return portable JSON data, rejecting non-finite and unsupported values."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        if abs(value) > _MAX_SAFE_INTEGER:
            raise ValidationError(
                "research integers must be exact portable binary64 values",
                context={"field": field, "max_safe_integer": _MAX_SAFE_INTEGER},
            )
        try:
            normalized_number = float(value)
        except OverflowError as error:
            raise ValidationError(
                "research integers must fit the portable binary64 range", context={"field": field}
            ) from error
        if not math.isfinite(normalized_number):
            raise ValidationError(
                "research integers must fit the portable binary64 range", context={"field": field}
            )
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValidationError(
                "research data must use finite JSON numbers", context={"field": field}
            )
        return value
    if seen is None:
        seen = set()
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in seen:
            raise ValidationError("research data cannot contain cycles", context={"field": field})
        seen.add(identity)
        try:
            normalized: dict[str, Any] = {}
            for key, nested_value in value.items():
                if not isinstance(key, str):
                    raise ValidationError(
                        "research JSON object keys must be strings", context={"field": field}
                    )
                normalized[key] = _normalize_json_value(
                    nested_value, field=f"{field}.{key}", seen=seen
                )
            return normalized
        finally:
            seen.remove(identity)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        identity = id(value)
        if identity in seen:
            raise ValidationError("research data cannot contain cycles", context={"field": field})
        seen.add(identity)
        try:
            return [
                _normalize_json_value(nested_value, field=f"{field}[{index}]", seen=seen)
                for index, nested_value in enumerate(value)
            ]
        finally:
            seen.remove(identity)
    raise ValidationError(
        "research data must contain only portable JSON values",
        context={"field": field, "type": type(value).__name__},
    )


def _normalized_object(value: object, *, field: str) -> dict[str, Any]:
    normalized = _normalize_json_value(value, field=field)
    if not isinstance(normalized, dict):
        raise ValidationError("research data must be a JSON object", context={"field": field})
    return normalized


def _entry_from_mapping(value: object) -> ResearchEntry:
    if not isinstance(value, Mapping):
        raise ValidationError("research entry must be an object")
    at = value.get("at")
    hypothesis = value.get("hypothesis")
    params = value.get("params")
    metrics = value.get("metrics")
    verdict = value.get("verdict")
    if not isinstance(at, str) or not isinstance(hypothesis, str):
        raise ValidationError("research entry requires string at and hypothesis values")
    normalized_params = _normalized_object(params, field="params")
    normalized_metrics = _normalized_object(metrics, field="metrics")
    if verdict is None:
        normalized_verdict: dict[str, Any] | None = None
    else:
        normalized_verdict = _normalized_object(verdict, field="verdict")
        if "overfit" in normalized_verdict and type(normalized_verdict["overfit"]) is not bool:
            raise ValidationError("verdict.overfit must be a boolean")
    return {
        "at": at,
        "hypothesis": hypothesis,
        "params": normalized_params,
        "metrics": normalized_metrics,
        "verdict": normalized_verdict,
    }


def _record_from_mapping(value: object) -> ResearchRecord:
    if not isinstance(value, Mapping):
        raise ValidationError("research record must be an object")
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
        raise ValidationError("research record requires string id, goal, and created_at values")
    _validate_id(record_id)
    if closed_at is not None and not isinstance(closed_at, str):
        raise ValidationError("research record closed_at must be a string or null")
    if not isinstance(entries, list):
        raise ValidationError("research record entries must be an array")
    return {
        "id": record_id,
        "goal": goal,
        "created_at": created_at,
        "closed_at": closed_at,
        "entries": [_entry_from_mapping(entry) for entry in entries],
    }


def best_sharpe(entries: list[ResearchEntry]) -> BestSharpe | None:
    """Return the first entry with the greatest finite ``metrics.sharpe`` value."""
    best: BestSharpe | None = None
    for entry in entries:
        candidate = entry["metrics"].get("sharpe")
        if isinstance(candidate, bool) or not isinstance(candidate, (int, float)):
            continue
        try:
            sharpe = float(candidate)
        except OverflowError:
            continue
        if not math.isfinite(sharpe):
            continue
        if best is None or sharpe > best["sharpe"]:
            best = {"sharpe": sharpe, "params": entry["params"]}
    return best


class ResearchStore:
    """Pathlib-backed synchronous store for durable research records.

    Each mutation uses an in-process lock plus an atomic same-directory lock
    file, then writes through a fsynced temporary file and ``os.replace``.
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

    def _lock_path_for(self, record_id: str) -> Path:
        return self.directory / f".{_validate_id(record_id)}.lock"

    @contextmanager
    def _record_lock(self, record_id: str) -> Iterator[None]:
        lock_path = self._lock_path_for(record_id)
        thread_lock = _thread_lock_for(lock_path)
        thread_lock.acquire()
        interprocess_lock = _InterprocessRecordLock(lock_path)
        try:
            interprocess_lock.acquire()
            try:
                yield
            finally:
                interprocess_lock.release()
        finally:
            thread_lock.release()

    def _new_record(self, record_id: str, goal: str = "") -> ResearchRecord:
        if not isinstance(goal, str):
            raise ValidationError("goal must be a string", context={"goal": goal})
        return {
            "id": _validate_id(record_id),
            "goal": goal,
            "created_at": _timestamp(self._clock),
            "closed_at": None,
            "entries": [],
        }

    def _save(self, record_id: str, record: ResearchRecord) -> ResearchRecord:
        normalized = _record_from_mapping(record)
        if normalized["id"] != record_id:
            raise ValidationError(
                "research record id does not match target path",
                context={"requested_id": record_id, "record_id": normalized["id"]},
            )
        path = self._path_for(record_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path: Path | None = None
        try:
            descriptor, raw_temp_path = tempfile.mkstemp(
                prefix=f".{record_id}.", suffix=".tmp", dir=path.parent, text=True
            )
            temp_path = Path(raw_temp_path)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(normalized, handle, indent=2, ensure_ascii=False, allow_nan=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, path)
        except (TypeError, ValueError, OverflowError, RecursionError) as error:
            if temp_path is not None:
                with suppress(OSError):
                    temp_path.unlink(missing_ok=True)
            raise ValidationError(
                "research record could not be encoded as strict JSON",
                context={"record_id": record_id},
            ) from error
        except BaseException:
            if temp_path is not None:
                with suppress(OSError):
                    temp_path.unlink(missing_ok=True)
            raise
        return normalized

    def _load_unlocked(self, record_id: str) -> ResearchRecord | None:
        path = self._path_for(record_id)
        try:
            with path.open(encoding="utf-8") as handle:
                loaded = json.load(handle)
        except FileNotFoundError:
            return None
        except (json.JSONDecodeError, UnicodeDecodeError, RecursionError) as error:
            raise ValidationError(
                f"corrupt research record '{record_id}'", context={"path": str(path)}
            ) from error
        except OSError as error:
            raise ValidationError(
                f"unable to read research record '{record_id}'",
                context={"path": str(path), "error": str(error)},
            ) from error
        try:
            record = _record_from_mapping(loaded)
        except ValidationError as error:
            raise ValidationError(
                f"invalid research record '{record_id}'", context={"path": str(path)}
            ) from error
        if record["id"] != record_id:
            raise ValidationError(
                f"research record '{record_id}' contains a different id",
                context={"requested_id": record_id, "embedded_id": record["id"]},
            )
        return record

    def load(self, record_id: str) -> ResearchRecord | None:
        """Load one record, returning ``None`` only when its file is absent."""
        _validate_id(record_id)
        return self._load_unlocked(record_id)

    def open(self, record_id: str, goal: str = "") -> ResearchRecord:
        """Load an existing record or persist a newly opened investigation."""
        _validate_id(record_id)
        with self._record_lock(record_id):
            record = self._load_unlocked(record_id)
            if record is not None:
                return record
            return self._save(record_id, self._new_record(record_id, goal))

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
        _validate_id(record_id)
        if not isinstance(hypothesis, str):
            raise ValidationError("hypothesis must be a string", context={"hypothesis": hypothesis})
        entry = _entry_from_mapping(
            {
                "at": _timestamp(self._clock),
                "hypothesis": hypothesis,
                "params": {} if params is None else params,
                "metrics": {} if metrics is None else metrics,
                "verdict": verdict,
            }
        )
        with self._record_lock(record_id):
            record = self._load_unlocked(record_id)
            if record is None:
                record = self._new_record(record_id)
            record["entries"].append(entry)
            self._save(record_id, record)
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
            parameters = json.dumps(
                best["params"], separators=(",", ":"), ensure_ascii=False, allow_nan=False
            )
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
        _validate_id(record_id)
        with self._record_lock(record_id):
            record = self._load_unlocked(record_id)
            if record is None:
                record = self._new_record(record_id)
            record["closed_at"] = _timestamp(self._clock)
            return self._save(record_id, record)


def create_research_store(
    directory: str | Path = _DEFAULT_DIRECTORY, *, clock: Callable[[], datetime] = _utc_now
) -> ResearchStore:
    """Create a synchronous :class:`ResearchStore` with optional deterministic clock."""
    return ResearchStore(directory, clock=clock)
