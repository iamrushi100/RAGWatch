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

    assert dataset["dataset_id"] == "ragwatch_tiny_sample_v1"
    assert len(dataset["documents"]) == 7
    assert len(dataset["queries"]) == 6


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
