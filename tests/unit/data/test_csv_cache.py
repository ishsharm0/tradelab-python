"""Normalization, CSV, statistics, and atomic cache contracts."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from tradelab.data import (
    cached_candles_path,
    candle_stats,
    load_candles_from_cache,
    load_candles_from_csv,
    merge_candles,
    normalize_candles,
    save_candles_to_cache,
)
from tradelab.errors import ValidationError


def test_normalize_aliases_repairs_sorts_first_dedupes_and_warns() -> None:
    rows = [
        {"timestamp": "2024-01-02T00:01:00Z", "o": "2", "h": 1, "l": 3, "c": 2.5},
        {"time": 1_704_153_600, "open": 1, "high": 2, "low": 0, "close": 1},
        {"time": 1_704_153_600_000, "open": 99, "high": 99, "low": 99, "close": 99},
        {"time": "bad", "open": 1, "high": 2, "low": 0, "close": 1},
    ]
    with pytest.warns(RuntimeWarning, match=r"input=4, valid=3, output=2"):
        output = normalize_candles(rows)
    assert [row["time"] for row in output] == [1_704_153_600_000, 1_704_153_660_000]
    assert output[0]["close"] == 1
    assert output[1]["high"] == 2.5
    assert output[1]["low"] == 2
    assert output[1]["volume"] == 0


def test_normalize_preserves_fractional_ms_and_javascript_primitive_coercion() -> None:
    output = normalize_candles(
        [
            {
                "time": 1_704_153_600_000.6,
                "open": True,
                "high": "",
                "low": False,
                "close": 2,
            }
        ]
    )
    assert output == [
        {
            "time": 1_704_153_600_000.6,
            "open": 1.0,
            "high": 2.0,
            "low": 0.0,
            "close": 2.0,
            "volume": 0.0,
        }
    ]


def test_normalize_does_not_apply_csv_quote_stripping_to_ohlc_values() -> None:
    assert (
        normalize_candles(
            [{"time": 1_704_153_600_000, "open": 1, "high": 2, "low": 0, "close": "'2'"}]
        )
        == []
    )


def test_naive_dates_follow_javascript_new_york_local_time() -> None:
    output = normalize_candles(
        [{"time": "2024-01-02T09:30:00", "open": 1, "high": 2, "low": 0, "close": 1}]
    )
    assert output[0]["time"] == 1_704_205_800_000


def test_date_only_is_utc_and_common_javascript_text_dates_are_new_york_local() -> None:
    output = normalize_candles(
        [
            {"time": "2024-01-02", "open": 1, "high": 2, "low": 0, "close": 1},
            {
                "time": "January 2, 2024 09:30:00",
                "open": 2,
                "high": 3,
                "low": 1,
                "close": 2,
            },
        ]
    )

    assert [row["time"] for row in output] == [1_704_153_600_000, 1_704_205_800_000]


def test_merge_precedence_stats_upper_median_and_empty() -> None:
    start = 1_704_153_600_000
    first = [{"time": start, "open": 1, "high": 2, "low": 0, "close": 1}]
    second = [
        {"time": start, "open": 9, "high": 9, "low": 9, "close": 9},
        {"time": start + 60_000, "open": 2, "high": 3, "low": 1, "close": 2},
        {"time": start + 240_000, "open": 3, "high": 4, "low": 2, "close": 3},
    ]
    with pytest.warns(RuntimeWarning):
        merged = merge_candles(first, second)
    assert merged[0]["close"] == 1
    stats = candle_stats(merged)
    assert stats is not None
    assert stats["estimatedIntervalMin"] == 3
    assert stats["priceRange"] == {"low": 0.0, "high": 4.0}
    assert candle_stats([]) is None


def test_csv_headers_quotes_custom_parser_and_inclusive_bounds(tmp_path: object) -> None:
    path = tmp_path / "prices.csv"  # type: ignore[operator]
    path.write_text(
        'ignored\nDate,Open,High,Low,"Adj Close",Vol\n'
        '"2024-01-01T00:00:00Z",1,2,0,1.5,10\n'
        '"2024-01-02T00:00:00Z",2,3,1,2.5,bad\n',
        encoding="utf-8",
    )
    rows = load_candles_from_csv(
        path,
        skip_rows=1,
        start_date="2024-01-02T00:00:00Z",
        end_date="2024-01-02T00:00:00Z",
        custom_date_parser=lambda value: float("nan"),
    )
    assert len(rows) == 1
    assert rows[0]["close"] == 2.5
    assert rows[0]["volume"] == 0


def test_csv_headerless_indexes_and_errors(tmp_path: object) -> None:
    path = tmp_path / "plain.csv"  # type: ignore[operator]
    path.write_text("1704153600,1,2,0,1.5\n", encoding="utf-8")
    assert (
        load_candles_from_csv(
            path,
            has_header=False,
            time_col=0.0,
            open_col=1.0,
            high_col=2.0,
            low_col=3.0,
            close_col=4.0,
            volume_col=5.0,
            start_date=9_999_999_999_999,
        )[0]["volume"]
        == 0
    )
    with pytest.raises(ValidationError, match="not found"):
        load_candles_from_csv(path.with_name("missing.csv"))
    empty = tmp_path / "empty.csv"  # type: ignore[operator]
    empty.write_text("\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="empty"):
        load_candles_from_csv(empty)


def test_csv_trims_tokens_before_custom_parser_and_matches_null_and_string_bounds(
    tmp_path: object,
) -> None:
    path = tmp_path / "parser.csv"  # type: ignore[operator]
    path.write_text(
        "time,open,high,low,close\n"
        " '2024-01-01T00:00:00Z' ,'1',2,0,1\n"
        "2024-01-02T00:00:00Z,2,3,1,2\n",
        encoding="utf-8",
    )
    seen: list[object] = []

    with pytest.warns(RuntimeWarning, match="deduplicated candles"):
        parsed = load_candles_from_csv(
            path,
            custom_date_parser=lambda value: seen.append(value),
            start_date="1704153600000",
        )

    assert seen == ["2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z"]
    assert [row["time"] for row in parsed] == [0]
    assert parsed[0]["open"] == 1


def test_cache_path_utf16_safety_atomic_roundtrip_and_corruption(tmp_path: object) -> None:
    path = cached_candles_path("BRK/😀", "1 m", "../x", tmp_path)  # type: ignore[arg-type]
    assert path.name == "candles-BRK___-1_m-.._x.json"
    candles = [{"time": 0, "open": 1, "high": 2, "low": 0, "close": 1}]
    saved = save_candles_to_cache(
        candles,
        symbol="ES",
        interval="1m",
        period="1d",
        out_dir=tmp_path,  # type: ignore[arg-type]
        source="csv",
        now=datetime(2024, 1, 1, tzinfo=UTC),
    )
    payload = json.loads(saved.read_text(encoding="utf-8"))
    assert payload["count"] == 1
    assert payload["asOf"] == "2024-01-01T00:00:00.000Z"
    assert load_candles_from_cache("ES", "1m", "1d", tmp_path) == normalize_candles(candles)  # type: ignore[arg-type]
    assert not list(saved.parent.glob(f".{saved.name}.*"))
    saved.write_text("{", encoding="utf-8")
    assert load_candles_from_cache("ES", "1m", "1d", tmp_path) is None  # type: ignore[arg-type]


def test_cache_rejects_nonstandard_json_constants_and_js_stringifies_segments(
    tmp_path: object,
) -> None:
    path = cached_candles_path(None, True, False, tmp_path)  # type: ignore[arg-type]
    assert path.name == "candles-null-true-false.json"
    path.write_text(
        '{"meta":NaN,"candles":[{"time":1704153600000,"open":1,"high":2,"low":0,"close":1}]}',
        encoding="utf-8",
    )
    assert load_candles_from_cache(None, True, False, tmp_path) is None  # type: ignore[arg-type]


def test_cache_path_uses_javascript_string_coercion_for_numbers_arrays_and_objects(
    tmp_path: object,
) -> None:
    assert cached_candles_path(1.0, float("nan"), ["a", "b"], tmp_path).name == (  # type: ignore[arg-type]
        "candles-1-NaN-a_b.json"
    )
    assert cached_candles_path(float("inf"), float("-inf"), {"x": 1}, tmp_path).name == (  # type: ignore[arg-type]
        "candles-Infinity--Infinity-_object_Object_.json"
    )


@pytest.mark.parametrize("failure", ["serialize", "replace"])
def test_cache_atomic_failure_preserves_existing_file_and_removes_temporary(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    path = cached_candles_path("ES", "1m", "atomic", tmp_path)  # type: ignore[arg-type]
    path.write_text("existing", encoding="utf-8")
    if failure == "serialize":
        monkeypatch.setattr(
            "tradelab.data.cache.json.dump",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("boom")),
        )
    else:
        monkeypatch.setattr(
            "tradelab.data.cache.os.replace",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("boom")),
        )

    with pytest.raises(ValidationError, match="Could not save candle cache"):
        save_candles_to_cache(
            [{"time": 0, "open": 1, "high": 2, "low": 0, "close": 1}],
            symbol="ES",
            interval="1m",
            period="atomic",
            out_dir=tmp_path,  # type: ignore[arg-type]
        )

    assert path.read_text(encoding="utf-8") == "existing"
    assert not list(path.parent.glob(f".{path.name}.*"))


def test_cache_atomic_replace_failure_preserves_previous_file_and_cleans_temp(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    candles = [{"time": 0, "open": 1, "high": 2, "low": 0, "close": 1}]
    saved = save_candles_to_cache(
        candles,
        symbol="ES",
        interval="1m",
        period="1d",
        out_dir=tmp_path,  # type: ignore[arg-type]
    )
    original = saved.read_bytes()

    def fail_replace(_source: object, _destination: object) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr("tradelab.data.cache.os.replace", fail_replace)
    with pytest.raises(ValidationError, match="Could not save candle cache"):
        save_candles_to_cache(
            [{"time": 1, "open": 3, "high": 4, "low": 2, "close": 3}],
            symbol="ES",
            interval="1m",
            period="1d",
            out_dir=tmp_path,  # type: ignore[arg-type]
        )

    assert saved.read_bytes() == original
    assert not list(saved.parent.glob(f".{saved.name}.*"))
