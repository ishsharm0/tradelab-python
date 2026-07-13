"""Probability of backtest overfitting calculations."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TypedDict

from tradelab.errors import ValidationError
from tradelab.research.combinations import combinations

Number = int | float


class PboResult(TypedDict):
    """Combinatorial-symmetric cross-validation PBO summary."""

    pbo: float
    combos: int
    median_logit: float


def _sharpe(returns: list[float]) -> float:
    count = len(returns)
    if count < 2:
        return 0.0
    total = 0.0
    for value in returns:
        total += value
    mean = total / count
    variance = 0.0
    for value in returns:
        variance += (value - mean) ** 2
    variance /= count - 1
    standard_deviation = math.sqrt(variance)
    if standard_deviation == 0:
        if mean > 0:
            return math.inf
        if mean < 0:
            return -math.inf
        return 0.0
    return mean / standard_deviation


def _validated_matrix(performance_matrix: Sequence[Sequence[Number]]) -> list[list[float]]:
    if isinstance(performance_matrix, (str, bytes)) or len(performance_matrix) < 2:
        raise ValidationError(
            "performance_matrix needs at least two strategies",
            context={
                "n_strategies": len(performance_matrix)
                if not isinstance(performance_matrix, (str, bytes))
                else 0
            },
        )
    if not isinstance(performance_matrix[0], Sequence) or isinstance(
        performance_matrix[0], (str, bytes)
    ):
        raise ValidationError("performance_matrix rows must be numeric sequences")
    observations = len(performance_matrix[0])
    if observations < 2:
        raise ValidationError(
            "performance_matrix needs at least two observations",
            context={"n_observations": observations},
        )
    normalized: list[list[float]] = []
    for row_number, row in enumerate(performance_matrix):
        if (
            isinstance(row, (str, bytes))
            or not isinstance(row, Sequence)
            or len(row) != observations
        ):
            raise ValidationError(
                "performance_matrix rows must have equal length",
                context={"row": row_number, "expected_length": observations},
            )
        values: list[float] = []
        for value in row:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValidationError(
                    "performance_matrix values must be finite numbers",
                    context={"row": row_number, "value": value},
                )
            values.append(float(value))
        normalized.append(values)
    return normalized


def probability_of_backtest_overfitting(
    performance_matrix: Sequence[Sequence[Number]], *, groups: int = 16
) -> PboResult:
    """Estimate PBO using combinatorially symmetric cross-validation."""
    matrix = _validated_matrix(performance_matrix)
    if isinstance(groups, bool) or not isinstance(groups, int) or groups <= 0:
        raise ValidationError("groups must be a positive integer", context={"groups": groups})
    observations = len(matrix[0])
    group_count = min(groups, observations)
    if group_count % 2:
        raise ValidationError("groups must be even", context={"groups": group_count})

    group_indices: list[list[int]] = [[] for _ in range(group_count)]
    for index in range(observations):
        group_indices[(index * group_count) // observations].append(index)
    in_sample_combinations = combinations(group_count, group_count // 2)
    logits: list[float] = []
    overfit_count = 0
    for in_sample_groups in in_sample_combinations:
        in_sample_set = set(in_sample_groups)
        in_sample_indices: list[int] = []
        out_sample_indices: list[int] = []
        for group, indices in enumerate(group_indices):
            (in_sample_indices if group in in_sample_set else out_sample_indices).extend(indices)
        in_scores = [_sharpe([row[index] for index in in_sample_indices]) for row in matrix]
        out_scores = [_sharpe([row[index] for index in out_sample_indices]) for row in matrix]
        best_strategy = 0
        for strategy in range(1, len(matrix)):
            if in_scores[strategy] > in_scores[best_strategy]:
                best_strategy = strategy
        winner_out_score = out_scores[best_strategy]
        rank = 1
        for strategy, score in enumerate(out_scores):
            if strategy != best_strategy and score < winner_out_score:
                rank += 1
        relative_rank = rank / (len(matrix) + 1)
        logits.append(math.log(relative_rank / (1 - relative_rank)))
        if relative_rank <= 0.5:
            overfit_count += 1
    sorted_logits = sorted(logits)
    return {
        "pbo": overfit_count / len(in_sample_combinations),
        "combos": len(in_sample_combinations),
        "median_logit": sorted_logits[len(sorted_logits) // 2],
    }
