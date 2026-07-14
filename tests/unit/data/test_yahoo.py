"""Yahoo chart adapter request, retry, chunking, and normalization contracts."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from typing import Any

import httpx
import pytest

from tradelab.data.yahoo import fetch_historical, fetch_latest_candle
from tradelab.errors import DataProviderError, ValidationError


def _payload(timestamps: list[int] | None = None) -> dict[str, object]:
    values = timestamps or [1_735_828_200, 1_735_914_600]
    return {
        "chart": {
            "result": [
                {
                    "timestamp": values,
                    "indicators": {
                        "quote": [
                            {
                                "open": [100, 101],
                                "high": [99, 103],
                                "low": [102, 100],
                                "close": [101, 102],
                                "volume": [None, 1100],
                            }
                        ]
                    },
                }
            ],
            "error": None,
        }
    }


@pytest.mark.asyncio
async def test_fetch_historical_builds_exact_chart_request_and_sanitizes() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=_payload(), request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        candles = await fetch_historical(
            "BRK/B",
            "1d",
            "6months",
            include_pre_post=True,
            client=client,
            now_ms=lambda: 1_800_000_000_000,
            sleep=lambda _seconds: _done(),
            min_delay_seconds=0,
        )

    assert len(requests) == 1
    request = requests[0]
    assert request.url.raw_path.split(b"?")[0].endswith(b"/BRK%2FB")
    assert list(request.url.params.multi_items()) == [
        ("period1", "1784221200"),
        ("period2", "1800000000"),
        ("interval", "1d"),
        ("includePrePost", "true"),
        ("events", "div,splits"),
    ]
    assert request.headers["user-agent"].startswith("Mozilla/5.0")
    assert candles[0] == {
        "time": 1_735_828_200_000,
        "open": 100.0,
        "high": 101.0,
        "low": 100.0,
        "close": 101.0,
        "volume": 0.0,
    }


async def _done() -> None:
    return None


@pytest.mark.asyncio
async def test_retry_backoff_only_for_retryable_failures_and_clear_message() -> None:
    attempts = 0
    delays: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        raise httpx.ConnectError("fetch failed", request=request)

    async def record_sleep(seconds: float) -> None:
        delays.append(seconds)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(DataProviderError, match=r"Unable to reach Yahoo Finance.*CSV/cache"):
            await fetch_historical(
                "SPY",
                "1d",
                "6mo",
                client=client,
                now_ms=lambda: 1_800_000_000_000,
                sleep=record_sleep,
                min_delay_seconds=0,
            )
    assert attempts == 3
    assert delays == [0.5, 1.0]


@pytest.mark.asyncio
async def test_nonretryable_http_error_stops_immediately() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(400, text="bad symbol", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(DataProviderError, match="Yahoo API error 400"):
            await fetch_historical(
                "NOPE",
                client=client,
                now_ms=lambda: 1_800_000_000_000,
                sleep=lambda _seconds: _done(),
                min_delay_seconds=0,
            )
    assert attempts == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["timeout", "server"])
async def test_structural_transient_failures_retry_even_without_matching_text(
    failure: str,
) -> None:
    attempts = 0
    delays: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if failure == "timeout":
            raise httpx.ReadTimeout("", request=request)
        return httpx.Response(503, text="temporarily unavailable", request=request)

    async def sleep(seconds: float) -> None:
        delays.append(seconds)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(DataProviderError, match="after 3 attempts"):
            await fetch_historical(
                "SPY",
                client=client,
                now_ms=lambda: 1_800_000_000_000,
                sleep=sleep,
                min_delay_seconds=0,
            )
    assert attempts == 3
    assert delays == [0.5, 1.0]


@pytest.mark.asyncio
async def test_long_intraday_period_chunks_and_deduplicates_last_value() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        payload = _payload([100, 200])
        mutable: Any = payload
        quote_payload = mutable["chart"]["result"][0]["indicators"]["quote"][0]
        quote_payload["open"] = [calls, calls]
        quote_payload["high"] = [calls, calls]
        quote_payload["low"] = [calls, calls]
        quote_payload["close"] = [calls, calls]
        return httpx.Response(200, json=payload, request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        candles = await fetch_historical(
            "SPY",
            "1m",
            "8d",
            client=client,
            now_ms=lambda: 1_800_000_000_000,
            sleep=lambda _seconds: _done(),
            min_delay_seconds=0,
        )
    assert calls == 2
    assert [candle["open"] for candle in candles] == [2.0, 2.0]


@pytest.mark.asyncio
async def test_chart_errors_invalid_period_empty_results_and_latest() -> None:
    responses = iter(
        [
            {"chart": {"error": {"description": "delisted"}, "result": None}},
            {"chart": {"error": None, "result": []}},
            _payload(),
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=json.dumps(next(responses)), request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(DataProviderError, match="delisted"):
            await fetch_historical(
                "BAD",
                client=client,
                now_ms=lambda: 1_800_000_000_000,
                sleep=lambda _seconds: _done(),
                min_delay_seconds=0,
            )
        assert (
            await fetch_historical(
                "EMPTY",
                client=client,
                now_ms=lambda: 1_800_000_000_000,
                sleep=lambda _seconds: _done(),
                min_delay_seconds=0,
            )
            == []
        )
        latest = await fetch_latest_candle(
            "SPY",
            client=client,
            now_ms=lambda: 1_800_000_000_000,
            sleep=lambda _seconds: _done(),
            min_delay_seconds=0,
        )
    assert latest is not None and latest["close"] == 102
    with pytest.raises(ValidationError, match="Invalid period"):
        await fetch_historical("SPY", period="whenever")


@pytest.mark.asyncio
async def test_throttle_waits_for_remaining_global_window() -> None:
    clock = iter([1_000.0, 1_000.0, 1_100.0])
    delays: list[float] = []

    async def sleep(seconds: float) -> None:
        delays.append(seconds)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_payload(), request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await fetch_historical(
            "SPY",
            client=client,
            now_ms=lambda: next(clock),
            sleep=sleep,
            min_delay_seconds=0.4,
            _last_request_at_ms=800.0,
        )
    assert delays == [0.2]


def test_throttle_state_is_safe_across_multiple_event_loops() -> None:
    async def wave() -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_payload(), request=request)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await asyncio.gather(
                fetch_historical("SPY", client=client, min_delay_seconds=0),
                fetch_historical("QQQ", client=client, min_delay_seconds=0),
            )

    asyncio.run(wave())
    asyncio.run(wave())


def test_throttle_reservations_are_process_global_across_threads() -> None:
    barrier = threading.Barrier(2)
    requested: list[float] = []

    async def fetch_once() -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            requested.append(time.monotonic())
            return httpx.Response(200, json=_payload(), request=request)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await fetch_historical("SPY", client=client, min_delay_seconds=0.05)

    def worker() -> None:
        barrier.wait()
        asyncio.run(fetch_once())

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(requested) == 2
    assert abs(requested[1] - requested[0]) >= 0.045
