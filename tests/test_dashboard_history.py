"""Tests for the dashboard's local SQLite history helpers."""

import sqlite3
from pathlib import Path

import pytest

from src.dashboard_history import (
    compare_history_runs,
    discover_history_databases,
    load_history_runs,
)
from src.history import initialize_database, save_run


def make_result(recall: float) -> dict:
    return {
        "dataset_id": "tiny_dataset",
        "k": 2,
        "aggregate_metrics": {
            "recall_at_k": recall,
            "mrr": recall,
            "ndcg_at_k": recall,
        },
        "per_question": [
            {
                "question_id": "q_one",
                "recall_at_k": recall,
                "reciprocal_rank": recall,
                "ndcg_at_k": recall,
            }
        ],
    }


def test_discovers_only_sqlite_files_under_the_data_directory(tmp_path: Path):
    data_directory = tmp_path / "data"
    nested_database = data_directory / "nested" / "run.sqlite3"
    nested_database.parent.mkdir(parents=True)
    nested_database.touch()
    other_database = data_directory / "history.db"
    other_database.touch()
    (data_directory / "sample.json").touch()

    assert discover_history_databases(data_directory) == [
        other_database,
        nested_database,
    ]


def test_loads_empty_initialized_history_database(tmp_path: Path):
    database_path = tmp_path / "history.sqlite3"
    initialize_database(database_path)

    assert load_history_runs(database_path) == []


def test_compares_saved_runs_using_history_api(tmp_path: Path):
    database_path = tmp_path / "history.sqlite3"
    initialize_database(database_path)
    baseline_id = save_run(database_path, make_result(0.5))
    candidate_id = save_run(database_path, make_result(0.75))

    comparison = compare_history_runs(database_path, baseline_id, candidate_id)

    assert comparison["aggregate_differences"] == pytest.approx(
        {"recall_at_k": 0.25, "mrr": 0.25, "ndcg_at_k": 0.25}
    )
    assert comparison["question_comparisons"][0]["status"] == "matched"


def test_invalid_database_path_raises_sqlite_error(tmp_path: Path):
    with pytest.raises(sqlite3.Error):
        load_history_runs(tmp_path / "missing" / "history.sqlite3")
