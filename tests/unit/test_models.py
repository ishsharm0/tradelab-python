from __future__ import annotations

from datetime import UTC, datetime
from typing import assert_type

from tradelab.models import Candle, Signal, to_primitive


def test_candle_accepts_unix_ms_and_serializes_stably() -> None:
    candle = Candle(
        time=1_700_000_000_000,
        open=10,
        high=12,
        low=9,
        close=11,
        volume=100,
    )

    assert candle.to_dict() == {
        "time": 1_700_000_000_000,
        "open": 10.0,
        "high": 12.0,
        "low": 9.0,
        "close": 11.0,
        "volume": 100.0,
    }


def test_candle_normalizes_timezone_aware_datetime_to_utc() -> None:
    candle = Candle(
        time=datetime(2023, 11, 14, 22, 13, 20, tzinfo=UTC),
        open=10,
        high=12,
        low=9,
        close=11,
    )

    assert candle.time == datetime(2023, 11, 14, 22, 13, 20, tzinfo=UTC)
    assert candle.time_ms == 1_700_000_000_000


def test_candle_time_is_statically_a_datetime() -> None:
    candle = Candle(time=1_700_000_000_000, open=10, high=12, low=9, close=11)

    assert_type(candle.time, datetime)


def test_signal_normalizes_side_aliases() -> None:
    assert Signal(side="buy", stop=9).normalized_side == "long"
    assert Signal(side="sell", stop=11).normalized_side == "short"


def test_to_primitive_recursively_serializes_models() -> None:
    payload = {"candles": [Candle(time=0, open=1, high=2, low=0.5, close=1.5)]}

    assert to_primitive(payload) == {
        "candles": [
            {
                "time": 0,
                "open": 1.0,
                "high": 2.0,
                "low": 0.5,
                "close": 1.5,
                "volume": None,
            }
        ]
    }
