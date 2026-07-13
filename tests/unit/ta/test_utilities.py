"""Tests for deterministic local random, sizing, and New York session utilities."""

from __future__ import annotations

import random as stdlib_random
from datetime import UTC, datetime

import pytest

from tradelab import ValidationError
from tradelab.utils.position_sizing import calculate_position_size
from tradelab.utils.random import make_rng, rand_int
from tradelab.utils.time import (
    in_windows_et,
    is_session,
    minutes_et,
    offset_et,
    parse_windows_csv,
)


def _ms(value: datetime) -> int:
    return int(value.timestamp() * 1_000)


def test_seeded_rng_is_deterministic_and_does_not_mutate_global_state() -> None:
    stdlib_random.seed(93)
    before = stdlib_random.random()
    rng_a = make_rng("abc")
    rng_b = make_rng("abc")
    values_a = [rng_a() for _ in range(3)]
    values_b = [rng_b() for _ in range(3)]
    after = stdlib_random.random()

    stdlib_random.seed(93)
    assert before == stdlib_random.random()
    assert after == stdlib_random.random()
    assert values_a == values_b
    assert all(0 <= value < 1 for value in values_a)
    assert all(0 <= rand_int(rng_a, 5) < 5 for _ in range(100))


def test_different_rng_seeds_diverge() -> None:
    assert make_rng("abc")() != make_rng("xyz")()


def test_rng_matches_javascript_for_astral_unicode_seed() -> None:
    rng = make_rng("💹")

    assert [rng() for _ in range(3)] == [
        0.28879048582166433,
        0.0992135307751596,
        0.5422337634954602,
    ]


def test_rng_coerces_integer_seeds_through_javascript_number_semantics() -> None:
    assert make_rng(9_007_199_254_740_993)() == 0.8206203512381762


@pytest.mark.parametrize(
    ("seed", "javascript_string"),
    [
        pytest.param(10**10_000, "Infinity", id="positive-infinity"),
        pytest.param(-(10**10_000), "-Infinity", id="negative-infinity"),
    ],
)
def test_rng_coerces_overflowing_integer_seeds_to_javascript_infinity(
    seed: int, javascript_string: str
) -> None:
    assert make_rng(seed)() == make_rng(javascript_string)()


@pytest.mark.parametrize(
    ("seed", "javascript_string"),
    [(7.0, "7"), (None, "null"), (True, "true"), (False, "false")],
)
def test_rng_matches_javascript_string_coercion(seed: object, javascript_string: str) -> None:
    object_seeded = make_rng(seed)
    string_seeded = make_rng(javascript_string)

    assert [object_seeded() for _ in range(3)] == [string_seeded() for _ in range(3)]


def test_position_size_applies_risk_leverage_and_quantity_step() -> None:
    assert calculate_position_size(equity=10_000, entry=100, stop=99) == pytest.approx(100)
    assert calculate_position_size(
        equity=10_000, entry=100, stop=99, max_leverage=0.5
    ) == pytest.approx(50)
    assert calculate_position_size(equity=0, entry=100, stop=99) == 0
    assert calculate_position_size(equity=10_000, entry=100, stop=100) == 0


def test_invalid_position_sizing_options_raise_validation_error() -> None:
    with pytest.raises(ValidationError):
        calculate_position_size(equity=100, entry=10, stop=9, qty_step=0)


def test_new_york_offset_minutes_and_sessions_handle_dst_boundaries() -> None:
    before_dst = _ms(datetime(2025, 3, 9, 6, 59, tzinfo=UTC))
    after_dst = _ms(datetime(2025, 3, 9, 7, 0, tzinfo=UTC))
    nyse_open = _ms(datetime(2025, 3, 10, 13, 30, tzinfo=UTC))
    futures_open = _ms(datetime(2025, 3, 9, 22, 0, tzinfo=UTC))

    assert offset_et(before_dst) == 5
    assert minutes_et(before_dst) == 119
    assert offset_et(after_dst) == 4
    assert minutes_et(after_dst) == 180
    assert is_session(nyse_open, "NYSE") is True
    assert is_session(nyse_open - 60_000, "NYSE") is False
    assert is_session(after_dst, "FUT") is True
    assert is_session(futures_open, "FUT") is True


def test_sessions_apply_utc_weekend_rule_before_auto_and_use_eastern_minutes() -> None:
    dst_transition = _ms(datetime(2025, 3, 9, 7, 0, tzinfo=UTC))
    saturday_1700 = _ms(datetime(2025, 3, 8, 22, 0, tzinfo=UTC))
    saturday_1800 = _ms(datetime(2025, 3, 8, 23, 0, tzinfo=UTC))

    assert is_session(dst_transition, "AUTO") is False
    assert is_session(dst_transition, "FUT") is True
    assert is_session(saturday_1700, "FUT") is False
    assert is_session(saturday_1800, "FUT") is True


def test_session_windows_are_inclusive_and_parsed_from_csv() -> None:
    timestamp = _ms(datetime(2025, 1, 2, 15, 30, tzinfo=UTC))  # 10:30 Eastern
    windows = parse_windows_csv("09:30 - 10:30, 13:00-14:00")

    assert windows == [{"aMin": 570, "bMin": 630}, {"aMin": 780, "bMin": 840}]
    assert in_windows_et(timestamp, windows) is True
    assert in_windows_et(timestamp, None) is True


@pytest.mark.parametrize("csv", ["9:x-10:00", "09:00", "25:00-26:00"])
def test_invalid_window_csv_raises_validation_error(csv: str) -> None:
    with pytest.raises(ValidationError):
        parse_windows_csv(csv)
