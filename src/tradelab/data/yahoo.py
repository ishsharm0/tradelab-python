"""Asynchronous Yahoo Finance chart adapter with deterministic retry and throttling."""

from __future__ import annotations

import asyncio
import math
import re
import threading
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any
from urllib.parse import quote

import httpx

from tradelab.errors import DataProviderError, ValidationError

DAY_MS = 24 * 60 * 60 * 1_000
_USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
_RETRYABLE = re.compile(r"too many requests|rate limit|429|timeout|fetch failed|network", re.I)

Clock = Callable[[], float]
Sleeper = Callable[[float], Awaitable[None]]
Candle = dict[str, int | float]


class _YahooHttpError(DataProviderError):
    def __init__(self, status_code: int, body: str) -> None:
        self.status_code = status_code
        super().__init__(f"Yahoo API error {status_code}: {body}")


_THROTTLE_GUARD = threading.Lock()
_next_request_at_ms = 0.0


def _system_now_ms() -> float:
    return time.time() * 1_000


def _normalize_interval(interval: object) -> str:
    return str(interval or "1d").strip()


def _period_ms(period: object) -> float:
    if isinstance(period, (int, float)) and not isinstance(period, bool):
        value = float(period)
        if math.isfinite(value):
            return value
    raw = str(period or "60d").strip().lower()
    normalized = re.sub(r"months?$", "mo", raw)
    normalized = re.sub(r"^(\d+)mons?$", r"\1mo", normalized)
    match = re.fullmatch(r"(\d+)(mo|m|h|d|w|y)", normalized, re.I)
    if match is None:
        raise ValidationError(f'Invalid period "{period}". Use values like "5d", "60d", "1y".')
    amount = int(match.group(1))
    unit = match.group(2).lower()
    factors = {
        "m": 60 * 1_000,
        "h": 60 * 60 * 1_000,
        "d": DAY_MS,
        "w": 7 * DAY_MS,
    }
    if unit == "mo":
        return float(math.floor(amount * 30.4375 * DAY_MS + 0.5))
    if unit == "y":
        return float(math.floor(amount * 365.25 * DAY_MS + 0.5))
    return float(amount * factors[unit])


def _max_days(interval: str) -> int:
    if re.search(r"(m|h)$", interval, re.I) is None:
        return 365 * 10
    minute_match = re.fullmatch(r"(\d+)m", interval, re.I)
    if minute_match is not None:
        minutes = int(minute_match.group(1))
        if minutes <= 2:
            return 7
        if minutes <= 30:
            return 60
        if minutes <= 60:
            return 730
        return 365
    if re.fullmatch(r"\d+h", interval, re.I):
        return 730
    return 60


def _finite(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _sanitize(candles: Sequence[Mapping[str, object]]) -> list[Candle]:
    deduped: dict[float, Candle] = {}
    for candle in candles:
        time_value = _finite(candle.get("time"))
        open_value = _finite(candle.get("open"))
        high_value = _finite(candle.get("high"))
        low_value = _finite(candle.get("low"))
        close_value = _finite(candle.get("close"))
        if None in (time_value, open_value, high_value, low_value, close_value):
            continue
        assert time_value is not None
        assert open_value is not None
        assert high_value is not None
        assert low_value is not None
        assert close_value is not None
        volume_value = _finite(candle.get("volume"))
        compact_time: int | float = int(time_value) if time_value.is_integer() else time_value
        deduped[time_value] = {
            "time": compact_time,
            "open": open_value,
            "high": max(high_value, open_value, close_value),
            "low": min(low_value, open_value, close_value),
            "close": close_value,
            "volume": volume_value if volume_value is not None else 0.0,
        }
    return [deduped[key] for key in sorted(deduped)]


async def _rate_limited_get(
    client: httpx.AsyncClient,
    url: str,
    params: list[tuple[str, str | int | float | bool | None]],
    *,
    now_ms: Clock,
    sleep: Sleeper,
    min_delay_seconds: float,
    local_last_request_at_ms: list[float] | None,
) -> httpx.Response:
    minimum_ms = max(0.0, min_delay_seconds * 1_000)
    if local_last_request_at_ms is not None:
        elapsed_ms = now_ms() - local_last_request_at_ms[0]
        if elapsed_ms < minimum_ms:
            await sleep((minimum_ms - elapsed_ms) / 1_000)
        local_last_request_at_ms[0] = now_ms()
    else:
        global _next_request_at_ms
        current_ms = now_ms()
        with _THROTTLE_GUARD:
            request_at_ms = max(current_ms, _next_request_at_ms)
            _next_request_at_ms = request_at_ms + minimum_ms
        if request_at_ms > current_ms:
            await sleep((request_at_ms - current_ms) / 1_000)
    return await client.get(url, params=params, headers={"User-Agent": _USER_AGENT})


def _chart_rows(payload: object) -> list[Candle]:
    if not isinstance(payload, Mapping):
        raise DataProviderError("Yahoo returned an invalid chart payload")
    chart = payload.get("chart")
    if not isinstance(chart, Mapping):
        raise DataProviderError("Yahoo returned an invalid chart payload")
    error = chart.get("error")
    if error:
        description = error.get("description") if isinstance(error, Mapping) else None
        raise DataProviderError(str(description or "Yahoo chart error"))
    results = chart.get("result")
    if not isinstance(results, Sequence) or isinstance(results, (str, bytes)) or not results:
        return []
    result = results[0]
    if not isinstance(result, Mapping):
        return []
    timestamps = result.get("timestamp")
    indicators = result.get("indicators")
    if not isinstance(timestamps, Sequence) or isinstance(timestamps, (str, bytes)):
        return []
    quote_rows = indicators.get("quote") if isinstance(indicators, Mapping) else None
    quote = (
        quote_rows[0]
        if isinstance(quote_rows, Sequence)
        and not isinstance(quote_rows, (str, bytes))
        and quote_rows
        else {}
    )
    if not isinstance(quote, Mapping):
        quote = {}
    columns: dict[str, Sequence[object]] = {}
    for name in ("open", "high", "low", "close", "volume"):
        value = quote.get(name)
        columns[name] = (
            value if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else []
        )
    rows: list[dict[str, object]] = []
    for index, timestamp in enumerate(timestamps):
        try:
            values = {name: columns[name][index] for name in ("open", "high", "low", "close")}
        except IndexError:
            continue
        if any(value is None for value in values.values()):
            continue
        numeric_time = _finite(timestamp)
        if numeric_time is None:
            continue
        rows.append(
            {
                "time": numeric_time * 1_000,
                **values,
                "volume": columns["volume"][index] if index < len(columns["volume"]) else 0,
            }
        )
    return _sanitize(rows)


async def _fetch_chart(
    client: httpx.AsyncClient,
    symbol: str,
    *,
    period1: float,
    period2: float,
    interval: str,
    include_pre_post: bool,
    period: object,
    max_retries: int,
    now_ms: Clock,
    sleep: Sleeper,
    min_delay_seconds: float,
    local_last_request_at_ms: list[float] | None,
) -> list[Candle]:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{quote(symbol, safe='')}"
    params: list[tuple[str, str | int | float | bool | None]] = [
        ("period1", str(math.floor(period1))),
        ("period2", str(math.floor(period2))),
        ("interval", interval),
        ("includePrePost", "true" if include_pre_post else "false"),
        ("events", "div,splits"),
    ]
    last_error: Exception | None = None
    attempts = max(1, max_retries)
    attempts_made = 0
    for attempt in range(attempts):
        attempts_made = attempt + 1
        try:
            response = await _rate_limited_get(
                client,
                url,
                params,
                now_ms=now_ms,
                sleep=sleep,
                min_delay_seconds=min_delay_seconds,
                local_last_request_at_ms=local_last_request_at_ms,
            )
            if not response.is_success:
                raise _YahooHttpError(response.status_code, response.text)
            return _chart_rows(response.json())
        except (httpx.HTTPError, DataProviderError, ValueError) as error:
            last_error = error
            retryable = (
                isinstance(error, httpx.TransportError)
                or (
                    isinstance(error, _YahooHttpError)
                    and (error.status_code in {408, 425, 429} or 500 <= error.status_code <= 599)
                )
                or _RETRYABLE.search(str(error)) is not None
            )
            if not retryable or attempt == attempts - 1:
                break
            await sleep(min(12.0, 0.5 * 2**attempt))
    detail = str(last_error or "unknown error")
    raise DataProviderError(
        f"Unable to reach Yahoo Finance for {symbol} {interval} {period} after {attempts_made} "
        f"attempts. Last error: {detail} Try again later, or fall back to a local "
        'CSV/cache workflow with get_historical_candles(source="csv", ...) or '
        "load_candles_from_cache(...)."
    ) from last_error


async def fetch_historical(
    symbol: str,
    interval: object = "5m",
    period: object = "60d",
    *,
    include_pre_post: bool = False,
    client: httpx.AsyncClient | None = None,
    now_ms: Clock = _system_now_ms,
    sleep: Sleeper = asyncio.sleep,
    max_retries: int = 3,
    min_delay_seconds: float = 0.4,
    _last_request_at_ms: float | None = None,
) -> list[Candle]:
    """Fetch a normalized historical range from Yahoo's public chart endpoint."""
    if not isinstance(symbol, str) or not symbol:
        raise ValidationError("symbol must be a non-empty string")
    normalized_interval = _normalize_interval(interval)
    span_ms = _period_ms(period)
    max_span_ms = _max_days(normalized_interval) * DAY_MS
    local_last = (
        [_last_request_at_ms or 0.0]
        if _last_request_at_ms is not None or now_ms is not _system_now_ms
        else None
    )
    if client is None:
        async with httpx.AsyncClient(timeout=30.0) as owned_client:
            return await fetch_historical(
                symbol,
                normalized_interval,
                period,
                include_pre_post=include_pre_post,
                client=owned_client,
                now_ms=now_ms,
                sleep=sleep,
                max_retries=max_retries,
                min_delay_seconds=min_delay_seconds,
                _last_request_at_ms=_last_request_at_ms,
            )

    end_ms = now_ms()
    if span_ms <= max_span_ms:
        direct_rows = await _fetch_chart(
            client,
            symbol,
            period1=max(0, math.floor(end_ms / 1_000) - math.floor(span_ms / 1_000)),
            period2=math.floor(end_ms / 1_000),
            interval=normalized_interval,
            include_pre_post=include_pre_post,
            period=period,
            max_retries=max_retries,
            now_ms=now_ms,
            sleep=sleep,
            min_delay_seconds=min_delay_seconds,
            local_last_request_at_ms=local_last,
        )
        return _sanitize(direct_rows)

    rows: list[Candle] = []
    remaining_ms = span_ms
    chunk_end_ms = end_ms
    while remaining_ms > 0:
        take_ms = min(remaining_ms, max_span_ms)
        chunk_start_ms = chunk_end_ms - take_ms
        rows.extend(
            await _fetch_chart(
                client,
                symbol,
                period1=math.floor(chunk_start_ms / 1_000),
                period2=math.floor(chunk_end_ms / 1_000),
                interval=normalized_interval,
                include_pre_post=include_pre_post,
                period=period,
                max_retries=max_retries,
                now_ms=now_ms,
                sleep=sleep,
                min_delay_seconds=min_delay_seconds,
                local_last_request_at_ms=local_last,
            )
        )
        remaining_ms -= take_ms
        chunk_end_ms = chunk_start_ms - 1_000
        if chunk_end_ms <= 0 or len(rows) > 2_000_000:
            break
    return _sanitize(rows)


async def fetch_latest_candle(
    symbol: str, interval: object = "1m", **options: Any
) -> Candle | None:
    bars = await fetch_historical(symbol, interval, "5d", **options)
    return bars[-1] if bars else None


__all__ = ["fetch_historical", "fetch_latest_candle"]
