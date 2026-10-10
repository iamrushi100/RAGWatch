"""Tests for deterministic retrieval metric calculations."""

import math
from pathlib import Path

import pytest

from src.dataset import load_dataset
from src.evaluation import ndcg_at_k, recall_at_k, reciprocal_rank


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


def test_reciprocal_rank_is_one_for_relevant_result_at_rank_one(lighthouse_relevance):
    assert reciprocal_rank(
        lighthouse_relevance,
        ["doc_lighthouse_signal", "doc_cafe_hours"],
    ) == 1.0


def test_reciprocal_rank_uses_first_relevant_result_at_later_rank(
    lighthouse_relevance,
):
    ranked_ids = [
        "doc_cafe_hours",
        "doc_bakery_bread",
        "doc_lighthouse_location",
        "doc_lighthouse_signal",
    ]

    assert reciprocal_rank(lighthouse_relevance, ranked_ids) == pytest.approx(1 / 3)


def test_reciprocal_rank_is_zero_when_no_relevant_result_is_retrieved(
    lighthouse_relevance,
):
    assert reciprocal_rank(lighthouse_relevance, ["doc_cafe_hours"]) == 0.0


def test_reciprocal_rank_is_zero_for_empty_ranking(lighthouse_relevance):
    assert reciprocal_rank(lighthouse_relevance, []) == 0.0


def test_reciprocal_rank_is_zero_when_no_relevant_labels_exist():
    assert reciprocal_rank([], ["doc_cafe_hours"]) == 0.0


def test_reciprocal_rank_rejects_invalid_relevance_grade():
    labels = [{"document_id": "doc_one", "grade": -1}]

    with pytest.raises(ValueError, match="grade must be a non-negative integer"):
        reciprocal_rank(labels, ["doc_one"])


def test_reciprocal_rank_rejects_duplicate_ranked_ids():
    labels = [{"document_id": "doc_one", "grade": 1}]

    with pytest.raises(ValueError, match="Duplicate ranked document ID"):
        reciprocal_rank(labels, ["doc_one", "doc_one"])


def test_reciprocal_rank_requires_lists():
    with pytest.raises(ValueError, match="relevance_labels must be a list"):
        reciprocal_rank(None, [])

    with pytest.raises(ValueError, match="ranked_document_ids must be a list"):
        reciprocal_rank([], None)


def test_ndcg_is_one_for_perfect_lighthouse_ranking(lighthouse_relevance):
    ranked_ids = ["doc_lighthouse_signal", "doc_lighthouse_location"]

    assert ndcg_at_k(lighthouse_relevance, ranked_ids, k=2) == pytest.approx(1.0)


def test_ndcg_is_lower_for_reversed_lighthouse_ranking(lighthouse_relevance):
    ranked_ids = ["doc_lighthouse_location", "doc_lighthouse_signal"]

    # DCG = 1/log2(2) + 7/log2(3); IDCG = 7/log2(2) + 1/log2(3).
    dcg = 1 + 7 / math.log2(3)
    idcg = 7 + 1 / math.log2(3)

    assert ndcg_at_k(lighthouse_relevance, ranked_ids, k=2) == pytest.approx(
        dcg / idcg
    )
    assert dcg / idcg < 1.0


def test_ndcg_changes_with_k(lighthouse_relevance):
    ranked_ids = ["doc_lighthouse_location", "doc_lighthouse_signal"]

    assert ndcg_at_k(lighthouse_relevance, ranked_ids, k=1) == pytest.approx(1 / 7)
    assert ndcg_at_k(lighthouse_relevance, ranked_ids, k=2) > 1 / 7


def test_ndcg_penalizes_missing_relevant_results(lighthouse_relevance):
    result = ndcg_at_k(
        lighthouse_relevance,
        ["doc_lighthouse_signal"],
        k=2,
    )

    expected = 7 / (7 + 1 / math.log2(3))
    assert result == pytest.approx(expected)


def test_ndcg_returns_zero_when_there_are_no_relevant_labels():
    assert ndcg_at_k([], ["doc_cafe_hours"], k=1) == 0.0


def test_ndcg_returns_zero_for_empty_ranking(lighthouse_relevance):
    assert ndcg_at_k(lighthouse_relevance, [], k=2) == 0.0


@pytest.mark.parametrize("k", [0, -1, True, 1.5])
def test_ndcg_rejects_invalid_k(lighthouse_relevance, k):
    with pytest.raises(ValueError, match="k must be a positive integer"):
        ndcg_at_k(lighthouse_relevance, [], k=k)


def test_ndcg_rejects_malformed_relevance_grade():
    labels = [{"document_id": "doc_one", "grade": -1}]

    with pytest.raises(ValueError, match="grade must be a non-negative integer"):
        ndcg_at_k(labels, ["doc_one"], k=1)


def test_ndcg_rejects_duplicate_ranked_ids():
    labels = [{"document_id": "doc_one", "grade": 1}]

    with pytest.raises(ValueError, match="Duplicate ranked document ID"):
        ndcg_at_k(labels, ["doc_one", "doc_one"], k=2)
