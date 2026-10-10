"""Tests for deterministic retrieval metric calculations."""

from pathlib import Path

import pytest

from src.dataset import load_dataset
from src.evaluation import recall_at_k


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DATASET = PROJECT_ROOT / "data" / "sample_dataset.json"


@pytest.fixture
def lighthouse_relevance():
    dataset = load_dataset(SAMPLE_DATASET)
    lighthouse_query = next(
        query
        for query in dataset["queries"]
        if query["question_id"] == "q_lighthouse_color_and_timing"
    )
    return lighthouse_query["relevance"]


def test_recall_counts_only_relevant_results_in_top_k(lighthouse_relevance):
    ranked_ids = ["doc_lighthouse_location", "doc_cafe_hours"]

    assert recall_at_k(lighthouse_relevance, ranked_ids, k=2) == 0.5


def test_recall_changes_with_k(lighthouse_relevance):
    ranked_ids = ["doc_lighthouse_location", "doc_lighthouse_signal"]

    assert recall_at_k(lighthouse_relevance, ranked_ids, k=1) == 0.5
    assert recall_at_k(lighthouse_relevance, ranked_ids, k=2) == 1.0


def test_k_larger_than_ranking_uses_all_returned_results(lighthouse_relevance):
    ranked_ids = ["doc_lighthouse_signal"]

    assert recall_at_k(lighthouse_relevance, ranked_ids, k=10) == 0.5


def test_returns_zero_when_no_relevant_document_is_retrieved(lighthouse_relevance):
    assert recall_at_k(lighthouse_relevance, ["doc_cafe_hours"], k=1) == 0.0


def test_returns_zero_when_question_has_no_relevant_labels():
    assert recall_at_k([], ["doc_cafe_hours"], k=1) == 0.0


def test_returns_zero_for_empty_ranking(lighthouse_relevance):
    assert recall_at_k(lighthouse_relevance, [], k=5) == 0.0


@pytest.mark.parametrize("k", [0, -1, True, 1.5])
def test_rejects_invalid_k(lighthouse_relevance, k):
    with pytest.raises(ValueError, match="k must be a positive integer"):
        recall_at_k(lighthouse_relevance, [], k=k)


def test_rejects_duplicate_ids_in_ranking(lighthouse_relevance):
    with pytest.raises(ValueError, match="Duplicate ranked document ID"):
        recall_at_k(
            lighthouse_relevance,
            ["doc_lighthouse_signal", "doc_lighthouse_signal"],
            k=2,
        )


def test_zero_grade_is_not_relevant():
    labels = [
        {"document_id": "doc_relevant", "grade": 2},
        {"document_id": "doc_irrelevant", "grade": 0},
    ]

    assert recall_at_k(labels, ["doc_irrelevant"], k=1) == 0.0
