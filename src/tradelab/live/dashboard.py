"""Loopback-only FastAPI dashboard with bounded Server-Sent Events."""

from __future__ import annotations

import asyncio
import hmac
import inspect
import ipaddress
import json
import secrets
import time
from collections import deque
from collections.abc import AsyncGenerator, Callable, Mapping
from contextlib import suppress
from typing import Any

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>tradelab live</title></head>
<body><h1>tradelab live</h1><p>Use the authenticated local API to inspect runtime state.</p>
</body></html>"""


def _system_now_ms() -> int:
    return time.time_ns() // 1_000_000


async def _maybe_await(value: object) -> object:
    return await value if inspect.isawaitable(value) else value


class DashboardServer:
    """Own a FastAPI app and optional uvicorn lifecycle on loopback by default."""

    def __init__(
        self,
        *,
        source: object,
        host: str = "127.0.0.1",
        port: int = 4_317,
        max_buffer: int = 200,
        command_token: str | None = None,
        allow_remote: bool = False,
        now_ms: Callable[[], int] = _system_now_ms,
    ) -> None:
        event_bus = getattr(source, "event_bus", None)
        if event_bus is None or not callable(getattr(event_bus, "on_any", None)):
            raise ValueError("dashboard source must expose event_bus.on_any")
        if isinstance(max_buffer, bool) or not isinstance(max_buffer, int) or max_buffer <= 0:
            raise ValueError("max_buffer must be a positive integer")
        if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65_535:
            raise ValueError("port must be between 0 and 65535")
        if not isinstance(host, str) or not host:
            raise ValueError("host must be a non-empty string")
        try:
            loopback = host.lower() == "localhost" or ipaddress.ip_address(host).is_loopback
        except ValueError:
            loopback = False
        if not loopback:
            raise ValueError(
                "non-loopback dashboard binding is disabled; use an authenticated TLS proxy"
            )
        if command_token is not None and (not isinstance(command_token, str) or not command_token):
            raise ValueError("command_token must be a non-empty string")
        self.source = source
        self.host = host
        self.port = port
        self.max_buffer = max_buffer
        self.command_token = command_token or secrets.token_urlsafe(32)
        self._now_ms = now_ms
        self._recent: deque[dict[str, Any]] = deque(maxlen=max_buffer)
        self._clients: set[asyncio.Queue[str | None]] = set()
        self._closed = False
        self._uvicorn: uvicorn.Server | None = None
        self._server_task: asyncio.Task[None] | None = None
        self._unsubscribe = event_bus.on_any(self._on_event)
        self.app = self._build_app()

    @property
    def client_count(self) -> int:
        return len(self._clients)

    def _on_event(self, envelope: dict[str, Any]) -> None:
        raw = envelope.get("payload")
        message = {
            "event": str(envelope.get("event", "")),
            "payload": dict(raw) if isinstance(raw, Mapping) else {},
            "t": self._now_ms(),
        }
        self._recent.append(message)
        frame = f"data: {json.dumps(message, separators=(',', ':'), allow_nan=False)}\n\n"
        for queue in tuple(self._clients):
            if queue.full():
                with suppress(asyncio.QueueEmpty):
                    queue.get_nowait()
            queue.put_nowait(frame)

    async def event_stream(self) -> AsyncGenerator[str, None]:
        queue: asyncio.Queue[str | None] = asyncio.Queue(maxsize=self.max_buffer)
        self._clients.add(queue)
        try:
            for message in tuple(self._recent):
                yield f"data: {json.dumps(message, separators=(',', ':'), allow_nan=False)}\n\n"
            while True:
                frame = await queue.get()
                if frame is None:
                    return
                yield frame
        finally:
            self._clients.discard(queue)

    async def _state(self) -> dict[str, Any]:
        refresh = getattr(self.source, "refresh", None)
        if callable(refresh):
            with suppress(Exception):
                await _maybe_await(refresh())
        getter = getattr(self.source, "get_status", None)
        if not callable(getter):
            return {}
        result = await _maybe_await(getter())
        return dict(result) if isinstance(result, Mapping) else {}

    async def _command(self, request: Request) -> JSONResponse:
        if not self._authorized(request):
            return JSONResponse({"ok": False, "error": "forbidden"}, status_code=403)
        if int(request.headers.get("content-length", "0") or 0) > 65_536:
            return JSONResponse({"ok": False, "error": "command body too large"}, status_code=413)
        try:
            raw = await request.json()
        except (json.JSONDecodeError, ValueError):
            raw = {}
        command = raw if isinstance(raw, Mapping) else {}
        raw_type = command.get("type")
        command_type = raw_type if isinstance(raw_type, str) else None
        allowed = {
            "flatten": ("flatten", None),
            "stop": ("stop", None),
            "closePosition": ("close_position", command.get("symbol")),
            "cancelOrder": ("cancel_order", command.get("orderId")),
        }
        selected = allowed.get(command_type) if command_type is not None else None
        if selected is None:
            return JSONResponse(
                {"ok": False, "error": f'unsupported command "{command_type}"'},
                status_code=400,
            )
        method_name, argument = selected
        method = getattr(self.source, method_name, None)
        if not callable(method):
            return JSONResponse(
                {"ok": False, "error": f'unsupported command "{command_type}"'},
                status_code=400,
            )
        try:
            result = method() if argument is None else method(argument)
            await _maybe_await(result)
        except Exception:
            return JSONResponse({"ok": False, "error": "command failed"}, status_code=500)
        return JSONResponse({"ok": True})

    def _authorized(self, request: Request) -> bool:
        supplied = request.headers.get("X-Tradelab-Token", "")
        return hmac.compare_digest(supplied, self.command_token)

    def _build_app(self) -> FastAPI:
        app = FastAPI(title="tradelab live", docs_url=None, redoc_url=None)

        @app.get("/", response_class=HTMLResponse)
        async def index() -> str:
            return _HTML

        @app.get("/state")
        async def state(request: Request) -> JSONResponse:
            if not self._authorized(request):
                return JSONResponse({"ok": False, "error": "forbidden"}, status_code=403)
            return JSONResponse(await self._state())

        @app.post("/command")
        async def command(request: Request) -> JSONResponse:
            return await self._command(request)

        @app.get("/events", response_model=None)
        async def events(request: Request) -> Response:
            if not self._authorized(request):
                return JSONResponse({"ok": False, "error": "forbidden"}, status_code=403)
            return StreamingResponse(
                self.event_stream(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )

        return app

    async def start(self) -> str:
        if self._closed:
            raise RuntimeError("dashboard server is closed")
        if self._server_task is not None:
            return self.url
        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="warning",
            lifespan="off",
            access_log=False,
        )
        self._uvicorn = uvicorn.Server(config)
        self._server_task = asyncio.create_task(self._uvicorn.serve())
        while not self._uvicorn.started:
            if self._server_task.done():
                await self._server_task
                raise RuntimeError("dashboard server failed to start")
            await asyncio.sleep(0.01)
        if self._uvicorn.servers and self._uvicorn.servers[0].sockets:
            self.port = int(self._uvicorn.servers[0].sockets[0].getsockname()[1])
        return self.url

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._unsubscribe()
        for queue in tuple(self._clients):
            if queue.full():
                with suppress(asyncio.QueueEmpty):
                    queue.get_nowait()
            queue.put_nowait(None)
        task, server = self._server_task, self._uvicorn
        self._server_task = None
        if task is not None and server is not None:
            server.should_exit = True
            await task


def create_dashboard_server(**options: Any) -> DashboardServer:
    return DashboardServer(**options)
