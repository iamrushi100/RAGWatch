"""Focused tests for loading and validating evaluation datasets."""

import json
from pathlib import Path

import pytest

from src.dataset import DatasetError, load_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DATASET = PROJECT_ROOT / "data" / "sample_dataset.json"


def write_dataset(tmp_path: Path, data: object) -> Path:
    """Write a temporary JSON dataset and return its path."""
    file_path = tmp_path / "dataset.json"
    file_path.write_text(json.dumps(data), encoding="utf-8")
    return file_path


def test_loads_repository_sample_dataset():
    dataset = load_dataset(SAMPLE_DATASET)

    assert dataset["dataset_id"] == "ragwatch_tiny_sample_v2"
    assert len(dataset["documents"]) == 7
    assert len(dataset["queries"]) == 6
    assert len(dataset["chunks"]) == 12
    assert all("chunk_id" in label for query in dataset["queries"] for label in query["relevance"])


def test_legacy_source_label_is_normalized_when_source_has_one_chunk(tmp_path: Path):
    original = {
        "dataset_id": "legacy_single_chunk",
        "schema_version": 1,
        "documents": [{"document_id": "doc1", "text": "A short fact."}],
        "queries": [
            {
                "question_id": "q1",
                "question": "What fact?",
                "relevance": [{"document_id": "doc1", "grade": 3}],
            }
        ],
    }
    file_path = write_dataset(tmp_path, original)

    dataset = load_dataset(file_path)

    assert dataset["queries"][0]["relevance"] == [
        {"chunk_id": "doc1::chunk-0000", "grade": 3}
    ]
    # Normalization is in memory; the original fixture is not rewritten.
    assert json.loads(file_path.read_text(encoding="utf-8")) == original


def test_legacy_source_label_is_rejected_when_source_splits(tmp_path: Path):
    original = {
        "dataset_id": "legacy_split_source",
        "schema_version": 1,
        "documents": [{"document_id": "doc1", "text": "x" * 100}],
        "queries": [
            {
                "question_id": "q1",
                "question": "Which source?",
                "relevance": [{"document_id": "doc1", "grade": 2}],
            }
        ],
    }

    with pytest.raises(DatasetError, match="Relabel it with the specific relevant chunk_id"):
        load_dataset(write_dataset(tmp_path, original))


def test_reports_missing_file_clearly(tmp_path: Path):
    with pytest.raises(DatasetError, match="Could not read dataset file"):
        load_dataset(tmp_path / "missing.json")


def test_reports_invalid_json_clearly(tmp_path: Path):
    file_path = tmp_path / "broken.json"
    file_path.write_text("{not json", encoding="utf-8")

    with pytest.raises(DatasetError, match="contains invalid JSON"):
        load_dataset(file_path)


def test_requires_top_level_fields(tmp_path: Path):
    file_path = write_dataset(tmp_path, {"dataset_id": "tiny"})

    with pytest.raises(DatasetError, match="missing required top-level field"):
        load_dataset(file_path)


def test_rejects_duplicate_document_ids(tmp_path: Path):
    data = {
        "dataset_id": "tiny",
        "schema_version": 1,
        "documents": [
            {"document_id": "doc1", "text": "First."},
            {"document_id": "doc1", "text": "Second."},
        ],
        "queries": [],
    }

    with pytest.raises(DatasetError, match="Duplicate document_id 'doc1'"):
        load_dataset(write_dataset(tmp_path, data))


def test_rejects_duplicate_question_ids(tmp_path: Path):
    data = {
        "dataset_id": "tiny",
        "schema_version": 1,
        "documents": [],
        "queries": [
            {"question_id": "q1", "question": "First?", "relevance": []},
            {"question_id": "q1", "question": "Second?", "relevance": []},
        ],
    }

    with pytest.raises(DatasetError, match="Duplicate question_id 'q1'"):
        load_dataset(write_dataset(tmp_path, data))


def test_rejects_relevance_reference_to_unknown_document(tmp_path: Path):
    data = {
        "dataset_id": "tiny",
        "schema_version": 1,
        "documents": [{"document_id": "doc1", "text": "A fact."}],
        "queries": [
            {
                "question_id": "q1",
                "question": "What is the fact?",
                "relevance": [{"document_id": "missing", "grade": 2}],
            }
        ],
    }

    with pytest.raises(DatasetError, match="refers to unknown document_id 'missing'"):
        load_dataset(write_dataset(tmp_path, data))


@pytest.mark.parametrize("grade", [0, 4, "3", True])
def test_rejects_invalid_relevance_grades(tmp_path: Path, grade: object):
    data = {
        "dataset_id": "tiny",
        "schema_version": 1,
        "documents": [{"document_id": "doc1", "text": "A fact."}],
        "queries": [
            {
                "question_id": "q1",
                "question": "What is the fact?",
                "relevance": [{"document_id": "doc1", "grade": grade}],
            }
        ],
    }

    with pytest.raises(DatasetError, match="grade must be an integer from 1 to 3"):
        load_dataset(write_dataset(tmp_path, data))
