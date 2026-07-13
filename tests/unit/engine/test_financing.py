"""Financing contracts mirrored from the immutable JavaScript oracle."""

from __future__ import annotations

import pytest

from tradelab.engine.financing import financing_cost, funding_events
from tradelab.errors import ValidationError


def test_funding_counts_from_exclusive_to_inclusive_boundaries() -> None:
    hour = 60 * 60 * 1_000

    assert funding_events(0, 24 * hour, 8 * hour, 0) == 3
    assert funding_events(8 * hour, 24 * hour, 8 * hour, 0) == 2
    assert funding_events(0, 8 * hour - 1, 8 * hour, 0) == 0


def test_financing_fixture_long_and_short() -> None:
    costs = {
        "carry": {"longAnnualBps": 365, "shortAnnualBps": 120},
        "funding": {"anchorMs": 0, "intervalMs": 28_800_000, "rateBps": 1.5},
    }
    from_ms = 1_704_205_800_000
    to_ms = 1_704_295_800_000

    assert financing_cost("long", 10_000, from_ms, to_ms, costs) == pytest.approx(5.541666666666666)
    assert financing_cost("short", 10_000, from_ms, to_ms, costs) == pytest.approx(
        -4.1575342465753415
    )


@pytest.mark.parametrize(
    "operation",
    [
        lambda: funding_events(10**10_000, 1, 1),
        lambda: funding_events(0, 1, float("nan")),
        lambda: funding_events(0, 1, 1, float("inf")),
        lambda: financing_cost("invalid", 100, 0, 1, {}),
        lambda: financing_cost("long", 10**10_000, 0, 1, {}),
        lambda: financing_cost("long", 100, 0, 1, "invalid"),  # type: ignore[arg-type]
        lambda: financing_cost("long", 100, 0, 1, {"carry": {"longAnnualBps": float("inf")}}),
        lambda: financing_cost(
            "long",
            100,
            0,
            1,
            {"funding": {"intervalMs": 1, "rateBps": 1, "anchorMs": float("nan")}},
        ),
        lambda: financing_cost("long", 1e308, 0, 1, {"carry": {"longAnnualBps": 1e308}}),
    ],
)
def test_financing_boundaries_raise_validation_error(operation: object) -> None:
    with pytest.raises(ValidationError):
        operation()  # type: ignore[operator]
