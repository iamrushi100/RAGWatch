"""Tests for deterministic aggregate quality threshold checks."""

import pytest

from src.alerts import check_thresholds


@pytest.fixture
def evaluation_result():
    """Return a small result with the same shape as the evaluation runner."""
    return {
        "dataset_id": "tiny_sample",
        "k": 2,
        "per_question": [],
        "aggregate_metrics": {
            "recall_at_k": 0.8,
            "mrr": 0.6,
            "ndcg_at_k": 0.7,
        },
    }


def test_all_configured_metrics_pass_in_fixed_order(evaluation_result):
    result = check_thresholds(
        evaluation_result,
        {"ndcg_at_k": 0.7, "mrr": 0.5, "recall_at_k": 0.8},
    )

    assert result["passed"] is True
    assert [check["metric"] for check in result["checks"]] == [
        "recall_at_k",
        "mrr",
        "ndcg_at_k",
    ]
    assert all(check["passed"] for check in result["checks"])
    assert result["alerts"] == []


def test_one_metric_below_threshold_creates_one_alert(evaluation_result):
    result = check_thresholds(evaluation_result, {"mrr": 0.7})

    assert result["passed"] is False
    assert result["alerts"] == [
        {
            "metric": "mrr",
            "actual_score": 0.6,
            "threshold": 0.7,
            "passed": False,
            "message": "mrr score 0.6 is below the minimum threshold 0.7.",
        }
    ]


def test_multiple_metrics_below_threshold_create_multiple_alerts(evaluation_result):
    result = check_thresholds(
        evaluation_result, {"ndcg_at_k": 0.8, "recall_at_k": 0.9}
    )

    assert [alert["metric"] for alert in result["alerts"]] == [
        "recall_at_k",
        "ndcg_at_k",
    ]
    assert result["passed"] is False


def test_score_equal_to_threshold_passes(evaluation_result):
    result = check_thresholds(evaluation_result, {"recall_at_k": 0.8})

    assert result["checks"][0]["passed"] is True
    assert result["alerts"] == []


def test_thresholds_at_zero_and_one_are_valid(evaluation_result):
    evaluation_result["aggregate_metrics"].update(
        {"recall_at_k": 0.0, "mrr": 1.0}
    )

    result = check_thresholds(
        evaluation_result, {"recall_at_k": 0, "mrr": 1}
    )

    assert result["passed"] is True
    assert [check["threshold"] for check in result["checks"]] == [0.0, 1.0]


@pytest.mark.parametrize(
    "threshold",
    [True, False, float("nan"), float("inf"), float("-inf"), -0.01, 1.01, "0.5"],
)
def test_invalid_threshold_values_are_rejected(evaluation_result, threshold):
    with pytest.raises(ValueError, match="thresholds.mrr must be a finite number"):
        check_thresholds(evaluation_result, {"mrr": threshold})


def test_empty_threshold_configuration_is_rejected(evaluation_result):
    with pytest.raises(ValueError, match="at least one metric"):
        check_thresholds(evaluation_result, {})


def test_unknown_threshold_metric_is_rejected(evaluation_result):
    with pytest.raises(ValueError, match="Unsupported threshold metric"):
        check_thresholds(evaluation_result, {"precision": 0.5})


def test_thresholds_must_be_a_dictionary(evaluation_result):
    with pytest.raises(ValueError, match="thresholds must be a dictionary"):
        check_thresholds(evaluation_result, [])


def test_missing_aggregate_metric_is_rejected(evaluation_result):
    del evaluation_result["aggregate_metrics"]["ndcg_at_k"]

    with pytest.raises(ValueError, match="missing 'ndcg_at_k'"):
        check_thresholds(evaluation_result, {"mrr": 0.5})


def test_aggregate_metrics_must_be_a_dictionary(evaluation_result):
    evaluation_result["aggregate_metrics"] = None

    with pytest.raises(ValueError, match="aggregate_metrics must be a dictionary"):
        check_thresholds(evaluation_result, {"mrr": 0.5})


@pytest.mark.parametrize(
    "score",
    [True, float("nan"), float("inf"), float("-inf"), -0.01, 1.01, "0.5"],
)
def test_invalid_evaluation_score_is_rejected(evaluation_result, score):
    evaluation_result["aggregate_metrics"]["mrr"] = score

    with pytest.raises(
        ValueError,
        match="evaluation_result.aggregate_metrics.mrr must be a finite number",
    ):
        check_thresholds(evaluation_result, {"recall_at_k": 0.5})


def test_evaluation_result_must_be_a_dictionary():
    with pytest.raises(ValueError, match="evaluation_result must be a dictionary"):
        check_thresholds(None, {"mrr": 0.5})
