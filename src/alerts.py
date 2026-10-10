"""Deterministic checks for aggregate evaluation quality thresholds."""

import math
from typing import Any


_METRIC_NAMES = ("recall_at_k", "mrr", "ndcg_at_k")


def _validate_score(value: Any, name: str) -> float:
    """Return a finite score from 0 to 1, or explain why it is invalid."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number from 0 to 1.")

    try:
        score = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be a finite number from 0 to 1.") from error

    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError(f"{name} must be a finite number from 0 to 1.")
    return score


def check_thresholds(
    evaluation_result: dict[str, Any], thresholds: dict[str, float]
) -> dict[str, Any]:
    """Check configured minimum aggregate scores in a runner result.

    ``evaluation_result`` must contain an ``aggregate_metrics`` dictionary with
    valid ``recall_at_k``, ``mrr``, and ``ndcg_at_k`` scores. ``thresholds`` may
    contain any non-empty subset of those metric names. An empty mapping is
    rejected because it would perform no useful checks. Checks and alerts are
    always returned in the fixed metric order Recall@k, MRR, then nDCG@k.

    The result has an overall ``passed`` flag, a ``checks`` list with each
    metric's actual score, threshold, and pass status, and an ``alerts`` list
    containing the failed checks. A score equal to its threshold passes.
    Invalid or missing scores and invalid configurations raise ``ValueError``.
    """
    if not isinstance(evaluation_result, dict):
        raise ValueError("evaluation_result must be a dictionary.")

    aggregate_metrics = evaluation_result.get("aggregate_metrics")
    if not isinstance(aggregate_metrics, dict):
        raise ValueError("evaluation_result.aggregate_metrics must be a dictionary.")

    # Validate every runner metric, including metrics without a configured
    # threshold, so malformed evaluation output is never silently accepted.
    actual_scores: dict[str, float] = {}
    for metric_name in _METRIC_NAMES:
        if metric_name not in aggregate_metrics:
            raise ValueError(
                f"evaluation_result.aggregate_metrics is missing '{metric_name}'."
            )
        actual_scores[metric_name] = _validate_score(
            aggregate_metrics[metric_name],
            f"evaluation_result.aggregate_metrics.{metric_name}",
        )

    if not isinstance(thresholds, dict):
        raise ValueError("thresholds must be a dictionary.")
    if not thresholds:
        raise ValueError("thresholds must contain at least one metric.")

    unknown_metrics = [name for name in thresholds if name not in _METRIC_NAMES]
    if unknown_metrics:
        raise ValueError(
            f"Unsupported threshold metric '{unknown_metrics[0]}'. "
            f"Choose from: {', '.join(_METRIC_NAMES)}."
        )

    validated_thresholds = {
        metric_name: _validate_score(
            thresholds[metric_name], f"thresholds.{metric_name}"
        )
        for metric_name in _METRIC_NAMES
        if metric_name in thresholds
    }

    checks = []
    alerts = []
    for metric_name in _METRIC_NAMES:
        if metric_name not in validated_thresholds:
            continue

        actual_score = actual_scores[metric_name]
        threshold = validated_thresholds[metric_name]
        passed = actual_score >= threshold
        check = {
            "metric": metric_name,
            "actual_score": actual_score,
            "threshold": threshold,
            "passed": passed,
        }
        checks.append(check)
        if not passed:
            alerts.append(
                {
                    **check,
                    "message": (
                        f"{metric_name} score {actual_score} is below "
                        f"the minimum threshold {threshold}."
                    ),
                }
            )

    return {"passed": not alerts, "checks": checks, "alerts": alerts}
