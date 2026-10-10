"""Tests for the controlled ranking-inversion experiment."""

import json
from pathlib import Path

import pytest

from src import regression_experiment


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DATASET = PROJECT_ROOT / "data" / "sample_dataset.json"


def test_experiment_is_deterministic():
    first = regression_experiment.run_regression_experiment(SAMPLE_DATASET, k=2)
    second = regression_experiment.run_regression_experiment(SAMPLE_DATASET, k=2)

    assert first == second


def test_both_conditions_use_the_same_loaded_questions_and_labels(monkeypatch):
    original_rank_documents = regression_experiment.rank_documents
    observed_calls = []

    def recording_ranker(dataset, question):
        query = next(item for item in dataset["queries"] if item["question"] == question)
        observed_calls.append(
            (id(dataset), question, tuple(tuple(label.items()) for label in query["relevance"]))
        )
        return original_rank_documents(dataset, question)

    monkeypatch.setattr(regression_experiment, "rank_documents", recording_ranker)

    regression_experiment.run_regression_experiment(SAMPLE_DATASET, k=2)

    assert len(observed_calls) == 2 * 6
    assert len({call[0] for call in observed_calls}) == 1
    for baseline_call, degraded_call in zip(observed_calls[:6], observed_calls[6:]):
        assert baseline_call == degraded_call


def test_experiment_reports_calculated_differences_and_detects_regression():
    result = regression_experiment.run_regression_experiment(SAMPLE_DATASET, k=2)

    for metric_name in ("recall_at_k", "mrr", "ndcg_at_k"):
        expected_difference = (
            result["degraded"]["aggregate_metrics"][metric_name]
            - result["baseline"]["aggregate_metrics"][metric_name]
        )
        assert result["candidate_minus_baseline"][metric_name] == pytest.approx(
            expected_difference
        )
    assert any(
        difference < 0
        for difference in result["candidate_minus_baseline"].values()
    )


def test_cli_saves_machine_readable_experiment_result(tmp_path: Path):
    output_path = tmp_path / "nested" / "regression.json"

    exit_code = regression_experiment.main(
        ["--dataset", str(SAMPLE_DATASET), "--k", "2", "--output", str(output_path)]
    )

    saved_result = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert saved_result == regression_experiment.run_regression_experiment(
        SAMPLE_DATASET, k=2
    )


@pytest.mark.parametrize("k", [0, -1, True, 1.5])
def test_experiment_rejects_invalid_k(k):
    with pytest.raises(ValueError, match="k must be a positive integer"):
        regression_experiment.run_regression_experiment(SAMPLE_DATASET, k=k)
