"""Runtime feed and candle-completion contracts."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pytest

from tradelab.live.candle import CandleAggregator
from tradelab.live.feed import BrokerFeed, FeedProvider, PollingFeed


class HistoryBroker:
    def __init__(self) -> None:
        self.bars: list[dict[str, int | float]] = []
        self.bar_handler: Callable[[dict[str, Any]], object] | None = None

    async def get_historical_bars(
        self, symbol: str, interval: str, limit: int = 200
    ) -> Sequence[Mapping[str, object]]:
        return self.bars[-limit:]

    async def subscribe_bars(
        self, symbol: str, interval: str, handler: Callable[[dict[str, Any]], object]
    ) -> Callable[[], None]:
        self.bar_handler = handler
        return lambda: setattr(self, "bar_handler", None)

    async def subscribe_trades(
        self, symbol: str, handler: Callable[[dict[str, Any]], object]
    ) -> Callable[[], None]:
        return lambda: None


@pytest.mark.asyncio
async def test_feed_provider_contract_is_explicit() -> None:
    feed = FeedProvider()
    with pytest.raises(NotImplementedError, match="connect"):
        await feed.connect()
    with pytest.raises(NotImplementedError, match="disconnect"):
        await feed.disconnect()
    with pytest.raises(NotImplementedError, match="subscribeBars"):
        await feed.subscribe_bars("A", "1m", lambda _bar: None)
    with pytest.raises(NotImplementedError, match="subscribeTicks"):
        await feed.subscribe_ticks("A", lambda _tick: None)
    with pytest.raises(NotImplementedError, match="getHistoricalBars"):
        await feed.get_historical_bars("A", "1m", 2)


@pytest.mark.asyncio
async def test_polling_feed_orders_deduplicates_and_awaits_handlers() -> None:
    broker = HistoryBroker()
    feed = PollingFeed(broker=broker, default_bars_per_poll=5)
    seen: list[int | float] = []

    async def handler(bar: dict[str, Any]) -> None:
        await asyncio.sleep(0)
        seen.append(bar["time"])

    subscription = await feed.subscribe_bars("A", "1m", handler)
    broker.bars = [{"time": 2}, {"time": 1}]
    await feed.poll_once()
    await feed.poll_once()
    broker.bars.append({"time": 3})
    await feed.poll_once()
    assert seen == [1, 2, 3]
    subscription.unsubscribe()
    subscription.unsubscribe()
    broker.bars.append({"time": 4})
    await feed.poll_once()
    assert seen == [1, 2, 3]


@pytest.mark.asyncio
async def test_polling_loop_cancels_cleanly_and_can_reconnect() -> None:
    broker = HistoryBroker()
    entered = asyncio.Event()

    async def controlled_sleep(_seconds: float) -> None:
        entered.set()
        await asyncio.sleep(30)

    feed = PollingFeed(broker=broker, sleep=controlled_sleep)
    await feed.connect()
    feed.start_polling()
    await entered.wait()
    await feed.disconnect()
    assert feed.polling is False
    entered.clear()
    await feed.connect()
    feed.start_polling()
    await entered.wait()
    await feed.stop_polling()
    assert feed.polling is False


@pytest.mark.asyncio
async def test_broker_feed_delegates_subscriptions_and_history() -> None:
    broker = HistoryBroker()
    broker.bars = [{"time": 1}]
    feed = BrokerFeed(broker=broker)
    await feed.connect()
    stop = await feed.subscribe_bars("A", "1m", lambda _bar: None)
    assert broker.bar_handler is not None
    assert list(await feed.get_historical_bars("A", "1m", 1)) == [{"time": 1}]
    stop()
    await feed.disconnect()


@pytest.mark.asyncio
async def test_polling_feed_tick_subscription_validation_and_start_gate() -> None:
    broker = HistoryBroker()
    feed = PollingFeed(broker=broker)
    with pytest.raises(Exception, match="connected"):
        feed.start_polling()
    with pytest.raises(Exception, match="handler"):
        await feed.subscribe_ticks("A", None)  # type: ignore[arg-type]
    ticks: list[dict[str, Any]] = []
    stop = await feed.subscribe_ticks("A", ticks.append)
    stop()
    await feed.stop_polling()
    with pytest.raises(Exception, match="poll_interval_ms"):
        PollingFeed(broker=broker, poll_interval_ms=float("nan"))


def test_candle_aggregator_builds_ticks_deduplicates_and_force_closes() -> None:
    aggregator = CandleAggregator(mode="tick", interval="1m", grace_ms=1_000)
    bars: list[dict[str, Any]] = []
    aggregator.on_bar(bars.append)
    start = 1_735_828_200_000
    aggregator.process_tick({"time": start, "price": 100, "size": 1})
    aggregator.process_tick({"time": start + 30_000, "last": 101, "volume": 2})
    aggregator.process_tick({"time": start + 60_000, "price": 99, "size": 1})
    assert bars[0] == {
        "time": start,
        "open": 100.0,
        "high": 101.0,
        "low": 100.0,
        "close": 101.0,
        "volume": 3.0,
    }
    aggregator.force_close(start + 121_000)
    assert len(bars) == 2
    aggregator.process_polled_bars([bars[0], bars[1]])
    assert len(bars) == 2


def test_stream_candles_require_final_and_estimate_interval() -> None:
    aggregator = CandleAggregator(mode="stream", interval="bad")
    bars: list[dict[str, Any]] = []
    aggregator.on_bar(bars.append)
    bar = {"time": 1, "open": 1, "high": 1, "low": 1, "close": 1, "volume": 0}
    aggregator.process_bar(bar, is_final=False)
    aggregator.process_bar(bar, is_final=True)
    assert bars == [bar]
    assert aggregator.estimate_from_series([{"time": 0}, {"time": 120_000}]) == 120_000
