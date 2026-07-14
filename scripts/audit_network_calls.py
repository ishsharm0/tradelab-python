#!/usr/bin/env python3
"""Run the test suite with real network sockets disabled.

HTTP requests are recorded after URL sanitization, so mocked adapter coverage can
be audited without printing credentials, headers, query strings, or bodies.
"""

from __future__ import annotations

import socket
import sys
from collections.abc import Callable
from typing import Any
from urllib.parse import urlsplit

import httpx
import pytest


class NetworkAudit:
    """Pytest plugin that blocks sockets and records sanitized HTTP destinations."""

    def __init__(self) -> None:
        self.requests: set[tuple[str, str, str]] = set()
        self._socket_connect: Callable[..., Any] | None = None
        self._create_connection: Callable[..., Any] | None = None
        self._send: Callable[..., Any] | None = None

    def pytest_sessionstart(self, session: pytest.Session) -> None:
        del session
        self._socket_connect = socket.socket.connect
        self._create_connection = socket.create_connection
        self._send = httpx.AsyncClient.send
        audit = self

        def blocked_connect(sock: socket.socket, address: object) -> None:
            del sock
            raise RuntimeError(f"unmocked network socket blocked: {address!r}")

        def blocked_create_connection(*args: object, **kwargs: object) -> None:
            del args, kwargs
            raise RuntimeError("unmocked network socket blocked")

        async def audited_send(
            client: httpx.AsyncClient, request: httpx.Request, *args: object, **kwargs: object
        ) -> httpx.Response:
            assert audit._send is not None
            parsed = urlsplit(str(request.url))
            audit.requests.add((request.method, parsed.hostname or "", parsed.path or "/"))
            response = await audit._send(client, request, *args, **kwargs)
            if not isinstance(response, httpx.Response):
                raise TypeError("httpx send returned a non-response value")
            return response

        socket.socket.connect = blocked_connect  # type: ignore[assignment]
        socket.create_connection = blocked_create_connection  # type: ignore[assignment]
        httpx.AsyncClient.send = audited_send  # type: ignore[assignment]

    def pytest_sessionfinish(self, session: pytest.Session, exitstatus: int) -> None:
        del session, exitstatus
        if self._socket_connect is not None:
            socket.socket.connect = self._socket_connect  # type: ignore[method-assign]
        if self._create_connection is not None:
            socket.create_connection = self._create_connection
        if self._send is not None:
            httpx.AsyncClient.send = self._send  # type: ignore[method-assign]
        if not self.requests:
            print("network audit: no HTTP requests observed")
            return
        print("network audit: mocked HTTP destinations (credentials and payloads omitted)")
        for method, host, path in sorted(self.requests):
            print(f"  {method} {host}{path}")


def main() -> int:
    targets = sys.argv[1:] or ["tests"]
    return pytest.main(["-q", "--disable-warnings", *targets], plugins=[NetworkAudit()])


if __name__ == "__main__":
    raise SystemExit(main())
