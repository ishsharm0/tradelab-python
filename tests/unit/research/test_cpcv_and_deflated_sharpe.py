"""Focused behavior tests for CPCV and deflated Sharpe calculations."""

from __future__ import annotations

import math
from collections.abc import Callable

import pytest

from tradelab.errors import ValidationError
from tradelab.research.cpcv import combinatorial_purged_splits
from tradelab.research.deflated_sharpe import deflated_sharpe, sweep_haircut


def test_cpcv_purges_embargoes_around_every_test_block() -> None:
    splits = combinatorial_purged_splits(n_observations=12, n_groups=4, n_test_groups=2, embargo=1)

    assert splits[1] == {"train": [4, 10, 11], "test": [0, 1, 2, 6, 7, 8], "test_groups": [0, 2]}
    assert len(splits) == 6


@pytest.mark.parametrize(
    ("kwargs"),
    [
        {"n_observations": 0},
        {"n_observations": 4, "n_groups": 1},
        {"n_observations": 4, "n_groups": 4, "n_test_groups": 4},
        {"n_observations": 4, "n_groups": 2, "n_test_groups": 1, "embargo": -1},
    ],
)
def test_cpcv_rejects_invalid_parameters(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValidationError):
        combinatorial_purged_splits(**kwargs)


def test_sweep_haircut_matches_the_expected_null_maximum() -> None:
    result = sweep_haircut(num_trials=12, sharpe_std=0.3)

    assert result["num_trials"] == 12
    assert result["expected_max_sharpe"] == pytest.approx(0.49944341610694054)


def test_deflated_sharpe_matches_fixture_and_bounds_probability() -> None:
    result = deflated_sharpe(
        sharpe=1.4,
        sample_size=64,
        num_trials=12,
        sharpe_std=0.3,
        skew=-0.2,
        kurtosis=3.4,
    )

    assert result == pytest.approx(0.9999974528453102)
    assert 0 <= result <= 1


@pytest.mark.parametrize(
    ("function", "kwargs"),
    [
        (sweep_haircut, {"num_trials": math.nan, "sharpe_std": 1}),
        (deflated_sharpe, {"sharpe": math.inf, "sample_size": 10}),
        (deflated_sharpe, {"sharpe": 1, "sample_size": 0}),
    ],
)
def test_sharpe_functions_reject_nonfinite_or_invalid_inputs(
    function: Callable[..., object], kwargs: dict[str, float | int]
) -> None:
    with pytest.raises(ValidationError):
        function(**kwargs)
