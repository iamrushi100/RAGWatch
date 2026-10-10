"""Tests for deterministic per-question failure analysis."""

from pathlib import Path

import pytest

from src.dataset import load_dataset
from src.failure_analysis import analyze_question
from src.retriever import rank_documents


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DATASET = PROJECT_ROOT / "data" / "sample_dataset.json"
LIGHTHOUSE_QUESTION_ID = "q_lighthouse_color_and_timing"


@pytest.fixture
def sample_dataset():
    return load_dataset(SAMPLE_DATASET)


def test_all_relevant_chunks_at_top_has_no_diagnostic_labels(sample_dataset):
    result = analyze_question(
        sample_dataset,
        LIGHTHOUSE_QUESTION_ID,
        [
            ("doc_lighthouse_signal::chunk-0000", 0.9),
            ("doc_lighthouse_location::chunk-0000", 0.5),
            ("doc_cafe_hours::chunk-0000", 0.1),
        ],
        k=2,
    )

    assert result == {
        "question_id": LIGHTHOUSE_QUESTION_ID,
        "relevant_chunk_ids": [
            "doc_lighthouse_location::chunk-0000",
            "doc_lighthouse_signal::chunk-0000",
        ],
        "retrieved_chunk_ids": [
            "doc_lighthouse_signal::chunk-0000",
            "doc_lighthouse_location::chunk-0000",
            "doc_cafe_hours::chunk-0000",
        ],
        "missing_relevant_chunk_ids": [],
        "misranked_relevant_chunk_ids": [],
        "diagnostic_labels": [],
    }


def test_one_or_more_relevant_chunks_missing_from_top_k(sample_dataset):
    result = analyze_question(
        sample_dataset,
        LIGHTHOUSE_QUESTION_ID,
        [("doc_lighthouse_signal::chunk-0000", 0.9), ("doc_cafe_hours::chunk-0000", 0.1)],
        k=1,
    )

    assert result["missing_relevant_chunk_ids"] == [
        "doc_lighthouse_location::chunk-0000"
    ]
    assert result["diagnostic_labels"] == ["retrieval_miss"]


def test_lower_grade_relevant_document_before_higher_grade_is_ranking_issue(
    sample_dataset,
):
    result = analyze_question(
        sample_dataset,
        LIGHTHOUSE_QUESTION_ID,
        [
            ("doc_lighthouse_location::chunk-0000", 0.9),
            ("doc_lighthouse_signal::chunk-0000", 0.5),
        ],
        k=2,
    )

    assert result["missing_relevant_chunk_ids"] == []
    assert result["misranked_relevant_chunk_ids"] == [
        "doc_lighthouse_location::chunk-0000"
    ]
    assert result["diagnostic_labels"] == ["ranking_issue"]


def test_no_relevant_documents_is_explicitly_labeled(sample_dataset):
    query = next(
        query
        for query in sample_dataset["queries"]
        if query["question_id"] == "q_library_loan_period"
    )
    query["relevance"] = []
    result = analyze_question(
        sample_dataset,
        "q_library_loan_period",
        [("doc_cafe_hours::chunk-0000", 0.9)],
        k=1,
    )

    assert result["relevant_chunk_ids"] == []
    assert result["missing_relevant_chunk_ids"] == []
    assert result["diagnostic_labels"] == ["no_relevant_documents"]


def test_empty_retrieval_results_are_explicit_and_relevant_docs_are_misses(
    sample_dataset,
):
    result = analyze_question(
        sample_dataset, LIGHTHOUSE_QUESTION_ID, [], k=2
    )

    assert result["retrieved_chunk_ids"] == []
    assert result["missing_relevant_chunk_ids"] == [
        "doc_lighthouse_location::chunk-0000",
        "doc_lighthouse_signal::chunk-0000",
    ]
    assert result["diagnostic_labels"] == [
        "empty_retrieval_results",
        "retrieval_miss",
    ]


def test_empty_ranking_with_no_relevant_docs_reports_both_edge_cases(sample_dataset):
    query = next(
        query
        for query in sample_dataset["queries"]
        if query["question_id"] == "q_library_loan_period"
    )
    query["relevance"] = []
    result = analyze_question(sample_dataset, "q_library_loan_period", [], k=2)

    assert result["diagnostic_labels"] == [
        "no_relevant_documents",
        "empty_retrieval_results",
    ]


def test_unknown_question_id_is_rejected(sample_dataset):
    with pytest.raises(ValueError, match="Question ID 'q_missing' was not found"):
        analyze_question(sample_dataset, "q_missing", [], k=2)


def test_malformed_ranking_pair_is_rejected(sample_dataset):
    with pytest.raises(ValueError, match=r"must be a \(chunk_id, score\) pair"):
        analyze_question(
            sample_dataset,
            LIGHTHOUSE_QUESTION_ID,
            ["doc_lighthouse_signal::chunk-0000"],
            k=2,
        )


def test_duplicate_ranked_chunk_ids_are_rejected(sample_dataset):
    with pytest.raises(ValueError, match="Duplicate ranked chunk ID"):
        analyze_question(
            sample_dataset,
            LIGHTHOUSE_QUESTION_ID,
            [
                ("doc_lighthouse_signal::chunk-0000", 0.9),
                ("doc_lighthouse_signal::chunk-0000", 0.8),
            ],
            k=2,
        )


def test_unknown_document_reference_in_ranking_is_rejected(sample_dataset):
    with pytest.raises(ValueError, match="unknown chunk_id 'doc_unknown::chunk-0000'"):
        analyze_question(
            sample_dataset,
            LIGHTHOUSE_QUESTION_ID,
            [("doc_unknown::chunk-0000", 0.9)],
            k=2,
        )


def test_unknown_document_reference_in_relevance_is_rejected(sample_dataset):
    lighthouse_query = next(
        query
        for query in sample_dataset["queries"]
        if query["question_id"] == LIGHTHOUSE_QUESTION_ID
    )
    lighthouse_query["relevance"].append(
        {"chunk_id": "doc_unknown::chunk-0000", "grade": 1}
    )

    with pytest.raises(
        ValueError, match="refers to unknown chunk_id 'doc_unknown::chunk-0000'"
    ):
        analyze_question(sample_dataset, LIGHTHOUSE_QUESTION_ID, [], k=2)


def test_output_is_deterministic_for_same_dataset_and_ranking(sample_dataset):
    ranked_results = rank_documents(
        sample_dataset, "What color is the lighthouse signal and when does it flash?"
    )

    first = analyze_question(
        sample_dataset, LIGHTHOUSE_QUESTION_ID, ranked_results, k=2
    )
    second = analyze_question(
        sample_dataset, LIGHTHOUSE_QUESTION_ID, ranked_results, k=2
    )

    assert first == second
