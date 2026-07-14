"""Atomic JSON state and JSONL journals for live sessions."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import uuid
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any, TypeVar

from tradelab.errors import ValidationError
from tradelab.models import BacktestResult

T = TypeVar("T")


def _missing(method: str) -> NotImplementedError:
    return NotImplementedError(f"StorageProvider.{method}() not implemented")


class StorageProvider:
    async def load(self, namespace: str) -> Any:
        raise _missing("load")

    async def save(self, namespace: str, state: object) -> None:
        raise _missing("save")

    async def append_trade(self, namespace: str, trade: object) -> None:
        raise _missing("appendTrade")

    async def append_equity_point(self, namespace: str, point: object) -> None:
        raise _missing("appendEquityPoint")

    async def load_trades(self, namespace: str) -> list[Any]:
        raise _missing("loadTrades")

    async def load_equity_curve(self, namespace: str) -> list[Any]:
        raise _missing("loadEquityCurve")

    async def clear(self, namespace: str) -> None:
        raise _missing("clear")


_UNSAFE_NAMESPACE = re.compile(r"[^a-zA-Z0-9._-]")


def _safe_json(value: object) -> Any:
    return BacktestResult({"value": value})["value"]


async def _complete_io(operation: Callable[[], T]) -> T:
    """Do not release a storage lock while a cancelled thread still mutates disk."""

    task = asyncio.create_task(asyncio.to_thread(operation))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        with suppress(BaseException):
            await task
        raise


class JsonFileStorage(StorageProvider):
    """Zero-dependency persistence with per-namespace serialization."""

    def __init__(self, *, base_dir: str | os.PathLike[str] | None = None) -> None:
        root = Path(base_dir) if base_dir is not None else Path.cwd() / "output" / "live-state"
        self.base_dir = root.expanduser().absolute()
        self._locks: dict[str, asyncio.Lock] = {}

    def _namespace(self, namespace: object) -> str:
        value = _UNSAFE_NAMESPACE.sub("_", str(namespace or "default"))
        if value in {"", ".", ".."}:
            raise ValidationError("namespace must identify a contained directory")
        return value

    def namespace_dir(self, namespace: object) -> Path:
        name = self._namespace(namespace)
        candidate = self.base_dir / name
        if candidate.is_symlink():
            raise ValidationError("namespace cannot be a symbolic link")
        resolved = candidate.resolve(strict=False)
        base = self.base_dir.resolve(strict=False)
        if resolved.parent != base:
            raise ValidationError("namespace escapes configured base directory")
        return candidate

    def _path(self, namespace: object, filename: str) -> Path:
        directory = self.namespace_dir(namespace)
        path = directory / filename
        if path.is_symlink():
            raise ValidationError("storage file cannot be a symbolic link")
        return path

    def state_path(self, namespace: object) -> Path:
        return self._path(namespace, "state.json")

    def trades_path(self, namespace: object) -> Path:
        return self._path(namespace, "trades.jsonl")

    def equity_path(self, namespace: object) -> Path:
        return self._path(namespace, "equity.jsonl")

    def _lock(self, namespace: object) -> asyncio.Lock:
        name = self._namespace(namespace)
        return self._locks.setdefault(name, asyncio.Lock())

    @staticmethod
    def _read_json(path: Path) -> Any:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (json.JSONDecodeError, UnicodeError, OSError) as error:
            raise ValidationError(f"invalid JSON state file: {path.name}") from error

    @staticmethod
    def _write_atomic(path: Path, payload: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.parent / f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        try:
            with temporary.open("x", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            try:
                descriptor = os.open(path.parent, os.O_RDONLY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
            except OSError:
                pass
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _append_line(path: Path, payload: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        try:
            remaining = memoryview(f"{encoded}\n".encode())
            while remaining:
                written = os.write(descriptor, remaining)
                if written <= 0:
                    raise OSError("short write made no progress")
                remaining = remaining[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @staticmethod
    def _read_lines(path: Path) -> list[Any]:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return []
        except (UnicodeError, OSError) as error:
            raise ValidationError(f"invalid JSON lines file: {path.name}") from error
        output: list[Any] = []
        for index, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                output.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValidationError(f"invalid JSON line {index} in {path.name}") from error
        return output

    async def load(self, namespace: str) -> Any:
        async with self._lock(namespace):
            path = self.state_path(namespace)
            return await _complete_io(lambda: self._read_json(path))

    async def save(self, namespace: str, state: object) -> None:
        payload = _safe_json(state)
        async with self._lock(namespace):
            path = self.state_path(namespace)
            await _complete_io(lambda: self._write_atomic(path, payload))

    async def append_trade(self, namespace: str, trade: object) -> None:
        payload = _safe_json(trade)
        async with self._lock(namespace):
            path = self.trades_path(namespace)
            await _complete_io(lambda: self._append_line(path, payload))

    async def append_equity_point(self, namespace: str, point: object) -> None:
        payload = _safe_json(point)
        async with self._lock(namespace):
            path = self.equity_path(namespace)
            await _complete_io(lambda: self._append_line(path, payload))

    async def load_trades(self, namespace: str) -> list[Any]:
        async with self._lock(namespace):
            path = self.trades_path(namespace)
            return await _complete_io(lambda: self._read_lines(path))

    async def load_equity_curve(self, namespace: str) -> list[Any]:
        async with self._lock(namespace):
            path = self.equity_path(namespace)
            return await _complete_io(lambda: self._read_lines(path))

    async def clear(self, namespace: str) -> None:
        async with self._lock(namespace):
            directory = self.namespace_dir(namespace)
            if directory.exists():
                await _complete_io(lambda: shutil.rmtree(directory))


def create_json_file_storage(*, base_dir: str | os.PathLike[str] | None = None) -> JsonFileStorage:
    return JsonFileStorage(base_dir=base_dir)
