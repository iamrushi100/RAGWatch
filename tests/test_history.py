"""Tests for persisting evaluation runs in SQLite."""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.history import initialize_database, save_run


SAMPLE_RESULT = {
    "dataset_id": "ragwatch_tiny_sample_v1",
    "k": 2,
    "per_question": [
        {
            "question_id": "q_one",
            "recall_at_k": 1.0,
            "reciprocal_rank": 1.0,
            "ndcg_at_k": 1.0,
        },
        {
            "question_id": "q_two",
            "recall_at_k": 0.5,
            "reciprocal_rank": 0.5,
            "ndcg_at_k": 0.7,
        },
    ],
    "aggregate_metrics": {
        "recall_at_k": 0.75,
        "mrr": 0.75,
        "ndcg_at_k": 0.85,
    },
}


def test_initialization_creates_tables_and_foreign_key(tmp_path: Path):
    database_path = tmp_path / "history.sqlite3"

    initialize_database(database_path)

    connection = sqlite3.connect(database_path)
    try:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        foreign_keys = connection.execute(
            "PRAGMA foreign_key_list(question_results)"
        ).fetchall()
    finally:
        connection.close()

    assert {"evaluation_runs", "question_results"}.issubset(tables)
    assert any(row[2] == "evaluation_runs" for row in foreign_keys)


def test_save_run_persists_aggregate_and_question_rows_after_close(tmp_path: Path):
    database_path = tmp_path / "history.sqlite3"
    initialize_database(database_path)

    run_id = save_run(database_path, SAMPLE_RESULT)

    # Read using a separate connection to show saved data outlives the writer.
    connection = sqlite3.connect(database_path)
    try:
        run_row = connection.execute(
            """
            SELECT dataset_id, k, created_at, recall_at_k, mrr, ndcg_at_k
            FROM evaluation_runs WHERE run_id = ?
            """,
            (run_id,),
        ).fetchone()
        question_rows = connection.execute(
            """
            SELECT question_id, recall_at_k, reciprocal_rank, ndcg_at_k
            FROM question_results WHERE run_id = ? ORDER BY question_id
            """,
            (run_id,),
        ).fetchall()
    finally:
        connection.close()

    assert run_row is not None
    assert run_row[:2] == ("ragwatch_tiny_sample_v1", 2)
    assert run_row[3:] == (0.75, 0.75, 0.85)
    timestamp = datetime.fromisoformat(run_row[2].replace("Z", "+00:00"))
    assert timestamp.tzinfo == timezone.utc
    assert question_rows == [
        ("q_one", 1.0, 1.0, 1.0),
        ("q_two", 0.5, 0.5, 0.7),
    ]


def test_each_saved_run_gets_a_distinct_integer_id(tmp_path: Path):
    database_path = tmp_path / "history.sqlite3"
    initialize_database(database_path)

    first_run_id = save_run(database_path, SAMPLE_RESULT)
    second_run_id = save_run(database_path, SAMPLE_RESULT)

    assert isinstance(first_run_id, int)
    assert isinstance(second_run_id, int)
    assert first_run_id != second_run_id


def test_repeated_initialization_keeps_existing_runs(tmp_path: Path):
    database_path = tmp_path / "history.sqlite3"
    initialize_database(database_path)
    saved_run_id = save_run(database_path, SAMPLE_RESULT)

    initialize_database(database_path)

    connection = sqlite3.connect(database_path)
    try:
        saved_row = connection.execute(
            "SELECT run_id FROM evaluation_runs WHERE run_id = ?", (saved_run_id,)
        ).fetchone()
    finally:
        connection.close()
    assert saved_row == (saved_run_id,)


def test_failed_question_insert_rolls_back_the_entire_run(tmp_path: Path):
    database_path = tmp_path / "history.sqlite3"
    initialize_database(database_path)
    connection = sqlite3.connect(database_path)
    try:
        connection.execute(
            """
            CREATE TRIGGER reject_question
            BEFORE INSERT ON question_results
            WHEN NEW.question_id = 'q_two'
            BEGIN
                SELECT RAISE(ABORT, 'test failure');
            END;
            """
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(sqlite3.IntegrityError, match="test failure"):
        save_run(database_path, SAMPLE_RESULT)

    connection = sqlite3.connect(database_path)
    try:
        run_count = connection.execute(
            "SELECT COUNT(*) FROM evaluation_runs"
        ).fetchone()[0]
        question_count = connection.execute(
            "SELECT COUNT(*) FROM question_results"
        ).fetchone()[0]
    finally:
        connection.close()
    assert run_count == 0
    assert question_count == 0


def test_save_run_rejects_missing_metric_instead_of_omitting_it(tmp_path: Path):
    database_path = tmp_path / "history.sqlite3"
    initialize_database(database_path)
    incomplete_result = {
        **SAMPLE_RESULT,
        "aggregate_metrics": {"recall_at_k": 0.75, "mrr": 0.75},
    }

    with pytest.raises(ValueError, match="aggregate_metrics.ndcg_at_k"):
        save_run(database_path, incomplete_result)

    connection = sqlite3.connect(database_path)
    try:
        run_count = connection.execute(
            "SELECT COUNT(*) FROM evaluation_runs"
        ).fetchone()[0]
    finally:
        connection.close()
    assert run_count == 0


@pytest.mark.parametrize("invalid_score", [float("nan"), float("inf"), float("-inf")])
def test_save_run_rejects_non_finite_aggregate_scores(
    tmp_path: Path, invalid_score: float
):
    invalid_result = {
        **SAMPLE_RESULT,
        "aggregate_metrics": {
            **SAMPLE_RESULT["aggregate_metrics"],
            "mrr": invalid_score,
        },
    }

    with pytest.raises(ValueError, match="mrr must be a finite number from 0 to 1"):
        save_run(tmp_path / "history.sqlite3", invalid_result)
