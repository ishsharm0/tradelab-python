"""Combinatorial purged cross-validation splits."""

from __future__ import annotations

from typing import TypedDict

from tradelab.errors import ValidationError
from tradelab.research.combinations import combinations


class CpcvSplit(TypedDict):
    """One CPCV split expressed as positional observation indices."""

    train: list[int]
    test: list[int]
    test_groups: list[int]


def _positive_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValidationError(f"{name} must be a positive integer", context={name: value})
    return value


def combinatorial_purged_splits(
    *, n_observations: int, n_groups: int = 6, n_test_groups: int = 2, embargo: int = 0
) -> list[CpcvSplit]:
    """Return all CPCV splits with an embargo around each selected test group."""
    observations = _positive_int(n_observations, name="n_observations")
    groups = _positive_int(n_groups, name="n_groups")
    test_groups_count = _positive_int(n_test_groups, name="n_test_groups")
    if groups < 2 or groups > observations:
        raise ValidationError(
            "n_groups must be between 2 and n_observations",
            context={"n_groups": groups, "n_observations": observations},
        )
    if test_groups_count >= groups:
        raise ValidationError(
            "n_test_groups must be less than n_groups",
            context={"n_test_groups": test_groups_count, "n_groups": groups},
        )
    if isinstance(embargo, bool) or not isinstance(embargo, int) or embargo < 0:
        raise ValidationError(
            "embargo must be a non-negative integer", context={"embargo": embargo}
        )

    bounds = [
        ((group * observations) // groups, ((group + 1) * observations) // groups)
        for group in range(groups)
    ]
    splits: list[CpcvSplit] = []
    for selected_groups in combinations(groups, test_groups_count):
        test_set: set[int] = set()
        purge_zones: list[tuple[int, int]] = []
        for group in selected_groups:
            start, end = bounds[group]
            for index in range(start, end):
                test_set.add(index)
            purge_zones.append((start - embargo, end + embargo))
        train: list[int] = []
        test: list[int] = []
        for index in range(observations):
            if index in test_set:
                test.append(index)
            elif not any(low <= index < high for low, high in purge_zones):
                train.append(index)
        splits.append({"train": train, "test": test, "test_groups": selected_groups})
    return splits
