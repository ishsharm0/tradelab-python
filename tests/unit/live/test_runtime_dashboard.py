"""Local FastAPI dashboard, command, and SSE contracts."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from tradelab.live import EventBus
from tradelab.live.dashboard import DashboardServer, create_dashboard_server


class Source:
    def __init__(self) -> None:
        self.event_bus = EventBus()
        self.refreshed = False
        self.calls: list[tuple[str, object | None]] = []

    async def refresh(self) -> None:
        self.refreshed = True

    def get_status(self) -> dict[str, object]:
        return {"symbol": "AAPL", "equity": 25_000, "refreshed": self.refreshed}

    async def flatten(self) -> None:
        self.calls.append(("flatten", None))

    async def stop(self) -> None:
        self.calls.append(("stop", None))

    async def close_position(self, symbol: str | None) -> None:
        self.calls.append(("close_position", symbol))

    async def cancel_order(self, order_id: str | None) -> None:
        self.calls.append(("cancel_order", order_id))


@pytest.mark.asyncio
async def test_dashboard_html_state_and_command_allowlist() -> None:
    source = Source()
    dashboard = create_dashboard_server(source=source, port=0, command_token="test-token")
    transport = httpx.ASGITransport(app=dashboard.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        root = await client.get("/")
        headers = {"X-Tradelab-Token": "test-token"}
        state = await client.get("/state", headers=headers)
        flatten = await client.post("/command", json={"type": "flatten"}, headers=headers)
        close = await client.post(
            "/command", json={"type": "closePosition", "symbol": "MSFT"}, headers=headers
        )
        cancel = await client.post(
            "/command", json={"type": "cancelOrder", "orderId": "order-1"}, headers=headers
        )
        rejected = await client.post("/command", json={"type": "__getattribute__"}, headers=headers)
    assert root.status_code == 200 and "tradelab live" in root.text
    assert state.json()["refreshed"] is True
    assert flatten.json() == close.json() == cancel.json() == {"ok": True}
    assert rejected.status_code == 400
    assert source.calls == [
        ("flatten", None),
        ("close_position", "MSFT"),
        ("cancel_order", "order-1"),
    ]
    await dashboard.close()


@pytest.mark.asyncio
async def test_dashboard_command_errors_stop_and_body_limit() -> None:
    source = Source()

    async def broken_flatten() -> None:
        raise RuntimeError("cannot flatten")

    source.flatten = broken_flatten  # type: ignore[method-assign]
    dashboard = DashboardServer(source=source, command_token="test-token")
    transport = httpx.ASGITransport(app=dashboard.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        headers = {"X-Tradelab-Token": "test-token"}
        stopped = await client.post("/command", json={"type": "stop"}, headers=headers)
        failed = await client.post("/command", json={"type": "flatten"}, headers=headers)
        malformed = await client.post(
            "/command",
            content="{",
            headers={**headers, "content-type": "application/json"},
        )
        oversized = await client.post(
            "/command",
            content="{}",
            headers={
                **headers,
                "content-length": "70000",
                "content-type": "application/json",
            },
        )
    assert stopped.status_code == 200
    assert failed.status_code == 500 and failed.json()["error"] == "command failed"
    assert malformed.status_code == 400
    assert oversized.status_code == 413
    await dashboard.close()


@pytest.mark.asyncio
async def test_dashboard_command_requires_secret_header() -> None:
    source = Source()
    dashboard = DashboardServer(source=source, command_token="known-secret")
    transport = httpx.ASGITransport(app=dashboard.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        missing = await client.post("/command", json={"type": "flatten"})
        wrong = await client.post(
            "/command",
            json={"type": "flatten"},
            headers={"X-Tradelab-Token": "wrong-secret"},
        )
        correct = await client.post(
            "/command",
            json={"type": "flatten"},
            headers={"X-Tradelab-Token": "known-secret"},
        )
        state = await client.get("/state")
        events = await client.get("/events")
    assert missing.status_code == wrong.status_code == 403
    assert missing.json() == wrong.json() == {"ok": False, "error": "forbidden"}
    assert correct.json() == {"ok": True}
    assert state.status_code == events.status_code == 403
    assert "known-secret" not in state.text
    assert source.calls == [("flatten", None)]
    await dashboard.close()


@pytest.mark.asyncio
async def test_sse_replays_bounded_events_and_removes_disconnected_client() -> None:
    source = Source()
    dashboard = DashboardServer(source=source, max_buffer=2, now_ms=lambda: 123)
    source.event_bus.emit_event("bar", {"n": 1})
    source.event_bus.emit_event("bar", {"n": 2})
    source.event_bus.emit_event("position:opened", {"symbol": "A"})
    stream = dashboard.event_stream()
    first = await anext(stream)
    second = await anext(stream)
    assert json.loads(first.removeprefix("data: "))["payload"] == {"n": 2}
    assert "position:opened" in second
    assert dashboard.client_count == 1
    await stream.aclose()
    assert dashboard.client_count == 0
    await dashboard.close()


@pytest.mark.asyncio
async def test_sse_slow_client_drops_oldest_and_close_unblocks_stream() -> None:
    source = Source()
    dashboard = DashboardServer(source=source, max_buffer=2)
    stream = dashboard.event_stream()
    waiting = asyncio.create_task(anext(stream))
    await asyncio.sleep(0)
    source.event_bus.emit_event("bar", {"n": 1})
    assert '"n":1' in await waiting
    for value in (2, 3, 4):
        source.event_bus.emit_event("bar", {"n": value})
    assert '"n":3' in await anext(stream)
    assert '"n":4' in await anext(stream)
    blocked = asyncio.create_task(anext(stream))
    await asyncio.sleep(0)
    await dashboard.close()
    with pytest.raises(StopAsyncIteration):
        await blocked
    assert dashboard.client_count == 0


@pytest.mark.asyncio
async def test_dashboard_actual_server_defaults_to_loopback_and_closes() -> None:
    dashboard = DashboardServer(source=Source(), port=0)
    assert dashboard.host == "127.0.0.1"
    url = await dashboard.start()
    assert url.startswith("http://127.0.0.1:")
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{url}/state", headers={"X-Tradelab-Token": dashboard.command_token}
        )
    assert response.status_code == 200
    await dashboard.close()
    await dashboard.close()
    with pytest.raises(RuntimeError, match="closed"):
        await dashboard.start()


def test_dashboard_requires_event_source_and_valid_limits() -> None:
    with pytest.raises(ValueError, match="event_bus"):
        DashboardServer(source=object())
    with pytest.raises(ValueError, match="max_buffer"):
        DashboardServer(source=Source(), max_buffer=0)
    with pytest.raises(ValueError, match="command_token"):
        DashboardServer(source=Source(), command_token="")
    with pytest.raises(ValueError, match="non-loopback"):
        DashboardServer(source=Source(), host="0.0.0.0")
    with pytest.raises(ValueError, match="TLS proxy"):
        DashboardServer(source=Source(), host="0.0.0.0", allow_remote=True)
