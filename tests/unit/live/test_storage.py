"""Atomic and traversal-safe live JSON persistence."""

from __future__ import annotations

import asyncio
import json
import math
import os
import threading
from pathlib import Path

import pytest

from tradelab.errors import ValidationError
from tradelab.live import JsonFileStorage


@pytest.mark.asyncio
async def test_storage_saves_loads_and_appends_json_atomically(tmp_path: Path) -> None:
    storage = JsonFileStorage(base_dir=tmp_path)
    await storage.save("alpha", {"equity": 1_000, "lastBarTime": 123})
    assert await storage.load("alpha") == {"equity": 1_000, "lastBarTime": 123}
    await storage.append_trade("alpha", {"id": 1, "side": "long"})
    await storage.append_trade("alpha", {"id": 2, "side": "short"})
    await storage.append_equity_point("alpha", {"time": 1, "equity": 1_000})
    assert [row["id"] for row in await storage.load_trades("alpha")] == [1, 2]
    assert await storage.load_equity_curve("alpha") == [{"time": 1, "equity": 1_000}]
    assert list((tmp_path / "alpha").glob("*.tmp")) == []
    json.loads((tmp_path / "alpha" / "state.json").read_text())


@pytest.mark.asyncio
async def test_concurrent_appends_remain_complete_json_lines(tmp_path: Path) -> None:
    storage = JsonFileStorage(base_dir=tmp_path)
    await asyncio.gather(*(storage.append_trade("parallel", {"id": index}) for index in range(50)))
    rows = await storage.load_trades("parallel")
    assert sorted(row["id"] for row in rows) == list(range(50))
    assert len((tmp_path / "parallel" / "trades.jsonl").read_text().splitlines()) == 50


@pytest.mark.asyncio
async def test_namespace_sanitization_never_escapes_base_and_clear_is_idempotent(
    tmp_path: Path,
) -> None:
    storage = JsonFileStorage(base_dir=tmp_path)
    await storage.save("../../escape", {"ok": True})
    assert storage.namespace_dir("../../escape").parent == tmp_path.resolve()
    assert not (tmp_path.parent / "escape").exists()
    with pytest.raises(ValidationError, match="namespace"):
        storage.namespace_dir("..")
    await storage.clear("../../escape")
    await storage.clear("../../escape")
    assert await storage.load("../../escape") is None


@pytest.mark.asyncio
async def test_storage_rejects_symlinked_namespace_and_files(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    base = tmp_path / "base"
    base.mkdir()
    (base / "evil").symlink_to(outside, target_is_directory=True)
    storage = JsonFileStorage(base_dir=base)
    with pytest.raises(ValidationError, match="symbolic link"):
        await storage.save("evil", {"owned": True})
    assert not (outside / "state.json").exists()

    safe = base / "safe"
    safe.mkdir()
    target = outside / "target.json"
    target.write_text("{}")
    (safe / "state.json").symlink_to(target)
    with pytest.raises(ValidationError, match="symbolic link"):
        await storage.save("safe", {"owned": True})
    assert target.read_text() == "{}"


@pytest.mark.asyncio
@pytest.mark.parametrize("unsafe", [math.nan, 10**10_000], ids=["nan", "huge"])
async def test_storage_rejects_nonportable_numbers(tmp_path: Path, unsafe: object) -> None:
    storage = JsonFileStorage(base_dir=tmp_path)
    with pytest.raises(ValidationError):
        await storage.save("unsafe", {"value": unsafe})
    assert not storage.state_path("unsafe").exists()


@pytest.mark.asyncio
async def test_storage_rejects_cycles_and_malformed_jsonl(tmp_path: Path) -> None:
    storage = JsonFileStorage(base_dir=tmp_path)
    cyclic: dict[str, object] = {}
    cyclic["self"] = cyclic
    with pytest.raises(ValidationError, match="cyclic"):
        await storage.append_trade("unsafe", cyclic)
    path = storage.trades_path("broken")
    path.parent.mkdir(parents=True)
    path.write_text('{"id":1}\nnot-json\n')
    with pytest.raises(ValidationError, match="invalid JSON"):
        await storage.load_trades("broken")


@pytest.mark.asyncio
async def test_cancelled_save_finishes_atomic_write_before_releasing_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage = JsonFileStorage(base_dir=tmp_path)
    entered = threading.Event()
    release = threading.Event()
    original = storage._write_atomic

    def delayed_write(path: Path, payload: object) -> None:
        entered.set()
        assert release.wait(timeout=2)
        original(path, payload)

    monkeypatch.setattr(storage, "_write_atomic", delayed_write)
    task = asyncio.create_task(storage.save("cancel", {"complete": True}))
    assert await asyncio.to_thread(entered.wait, 2)
    task.cancel()
    await asyncio.sleep(0)
    assert task.done() is False
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert await storage.load("cancel") == {"complete": True}
    assert list((tmp_path / "cancel").glob("*.tmp")) == []


def test_jsonl_append_retries_short_os_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "journal.jsonl"
    original = os.write
    calls = 0

    def short_write(descriptor: int, payload: bytes | bytearray | memoryview[int]) -> int:
        nonlocal calls
        calls += 1
        chunk = payload[: max(1, len(payload) // 2)]
        return original(descriptor, chunk)

    monkeypatch.setattr(os, "write", short_write)
    JsonFileStorage._append_line(path, {"id": 1, "value": "long-enough"})
    assert calls > 1
    assert json.loads(path.read_text()) == {"id": 1, "value": "long-enough"}
