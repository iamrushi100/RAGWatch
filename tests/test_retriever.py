"""Focused tests for the first TF-IDF retrieval baseline."""

import json
from pathlib import Path

import pytest

from src.dataset import load_dataset
from src.retriever import rank_documents


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DATASET = PROJECT_ROOT / "data" / "sample_dataset.json"


@pytest.fixture
def sample_dataset():
    return load_dataset(SAMPLE_DATASET)


def test_relevant_library_chunk_ranks_first(sample_dataset):
    results = rank_documents(
        sample_dataset,
        "How long can readers borrow books from Pine Street Library?",
    )

    assert results[0][0] == "doc_library_returns::chunk-0000"
    assert results[0][1] > 0


def test_results_are_descending_and_scores_stay_with_chunk_ids(sample_dataset):
    results = rank_documents(sample_dataset, "Harbor Point lighthouse red light")

    assert len(results) == len(sample_dataset["chunks"])
    assert all(results[index][1] >= results[index + 1][1]
               for index in range(len(results) - 1))

    scores_by_id = dict(results)
    assert scores_by_id["doc_lighthouse_signal::chunk-0000"] > scores_by_id[
        "doc_cafe_hours::chunk-0000"
    ]


def test_equal_scores_use_chunk_id_as_stable_tie_breaker(tmp_path: Path):
    dataset_path = tmp_path / "tie_dataset.json"
    dataset_path.write_text(
        json.dumps(
            {
                "dataset_id": "tie_example",
                "schema_version": 1,
                "documents": [
                    {"document_id": "doc_zebra", "text": "same words here"},
                    {"document_id": "doc_apple", "text": "same words here"},
                ],
                "queries": [],
            }
        ),
        encoding="utf-8",
    )
    dataset = load_dataset(dataset_path)

    results = rank_documents(dataset, "same words")

    assert [document_id for document_id, _ in results] == [
        "doc_apple::chunk-0000",
        "doc_zebra::chunk-0000",
    ]
    assert results[0][1] == results[1][1]


@pytest.mark.parametrize("question", ["", "   ", "\n\t"])
def test_empty_question_is_rejected(sample_dataset, question):
    with pytest.raises(ValueError, match="question must be a non-empty string"):
        rank_documents(sample_dataset, question)


def test_empty_document_collection_is_rejected(tmp_path: Path):
    dataset_path = tmp_path / "empty_dataset.json"
    dataset_path.write_text(
        json.dumps(
            {
                "dataset_id": "empty_example",
                "schema_version": 1,
                "documents": [],
                "queries": [],
            }
        ),
        encoding="utf-8",
    )
    dataset = load_dataset(dataset_path)

    with pytest.raises(ValueError, match="dataset with no text chunks"):
        rank_documents(dataset, "Any question")
