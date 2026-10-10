"""Tests for persisting evaluation runs in SQLite."""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.history import (
    IncompatibleRunsError,
    RunNotFoundError,
    compare_runs,
    initialize_database,
    list_runs,
    save_run,
)


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


def make_result(
    *,
    dataset_id: str = "ragwatch_tiny_sample_v1",
    k: int = 2,
    aggregate_metrics: dict | None = None,
    per_question: list[dict] | None = None,
) -> dict:
    """Build an independent, valid runner-shaped result for history tests."""
    return {
        "dataset_id": dataset_id,
        "k": k,
        "aggregate_metrics": aggregate_metrics or dict(SAMPLE_RESULT["aggregate_metrics"]),
        "per_question": per_question
        if per_question is not None
        else [dict(question) for question in SAMPLE_RESULT["per_question"]],
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


def test_list_runs_returns_newest_first_with_aggregate_metrics(tmp_path: Path):
    database_path = tmp_path / "history.sqlite3"
    initialize_database(database_path)
    older_run_id = save_run(database_path, SAMPLE_RESULT)
    newer_run_id = save_run(database_path, SAMPLE_RESULT)

    runs = list_runs(database_path)

    assert [run["run_id"] for run in runs] == [newer_run_id, older_run_id]
    assert runs[0]["dataset_id"] == "ragwatch_tiny_sample_v1"
    assert runs[0]["k"] == 2
    assert runs[0]["created_at"].endswith("Z")
    assert runs[0]["aggregate_metrics"] == SAMPLE_RESULT["aggregate_metrics"]


def test_list_runs_returns_empty_list_for_empty_history(tmp_path: Path):
    database_path = tmp_path / "history.sqlite3"
    initialize_database(database_path)

    assert list_runs(database_path) == []


def test_compare_identical_runs_has_zero_differences(tmp_path: Path):
    database_path = tmp_path / "history.sqlite3"
    initialize_database(database_path)
    baseline_id = save_run(database_path, SAMPLE_RESULT)
    candidate_id = save_run(database_path, SAMPLE_RESULT)

    comparison = compare_runs(database_path, baseline_id, candidate_id)

    assert comparison["aggregate_differences"] == {
        "recall_at_k": 0.0,
        "mrr": 0.0,
        "ndcg_at_k": 0.0,
    }
    assert all(
        question["differences"] == {
            "recall_at_k": 0.0,
            "reciprocal_rank": 0.0,
            "ndcg_at_k": 0.0,
        }
        for question in comparison["question_comparisons"]
    )


def test_compare_reports_positive_and_negative_differences(tmp_path: Path):
    database_path = tmp_path / "history.sqlite3"
    initialize_database(database_path)
    baseline = make_result(
        aggregate_metrics={"recall_at_k": 0.4, "mrr": 0.8, "ndcg_at_k": 0.5}
    )
    candidate = make_result(
        aggregate_metrics={"recall_at_k": 0.7, "mrr": 0.3, "ndcg_at_k": 0.6}
    )
    baseline_id = save_run(database_path, baseline)
    candidate_id = save_run(database_path, candidate)

    comparison = compare_runs(database_path, baseline_id, candidate_id)

    assert comparison["aggregate_differences"] == pytest.approx(
        {"recall_at_k": 0.3, "mrr": -0.5, "ndcg_at_k": 0.1}
    )


def test_compare_matches_questions_by_id_when_row_order_differs(tmp_path: Path):
    database_path = tmp_path / "history.sqlite3"
    initialize_database(database_path)
    baseline_questions = [
        {"question_id": "q_one", "recall_at_k": 0.2, "reciprocal_rank": 0.4, "ndcg_at_k": 0.3},
        {"question_id": "q_two", "recall_at_k": 0.5, "reciprocal_rank": 0.6, "ndcg_at_k": 0.7},
    ]
    candidate_questions = [
        {"question_id": "q_two", "recall_at_k": 0.8, "reciprocal_rank": 0.9, "ndcg_at_k": 0.9},
        {"question_id": "q_one", "recall_at_k": 0.6, "reciprocal_rank": 0.2, "ndcg_at_k": 0.5},
    ]
    baseline_id = save_run(database_path, make_result(per_question=baseline_questions))
    candidate_id = save_run(database_path, make_result(per_question=candidate_questions))

    comparison = compare_runs(database_path, baseline_id, candidate_id)

    comparisons = {
        question["question_id"]: question
        for question in comparison["question_comparisons"]
    }
    assert comparisons["q_one"]["differences"] == pytest.approx(
        {"recall_at_k": 0.4, "reciprocal_rank": -0.2, "ndcg_at_k": 0.2}
    )
    assert comparisons["q_two"]["differences"] == pytest.approx(
        {"recall_at_k": 0.3, "reciprocal_rank": 0.3, "ndcg_at_k": 0.2}
    )


def test_compare_identifies_questions_present_in_only_one_run(tmp_path: Path):
    database_path = tmp_path / "history.sqlite3"
    initialize_database(database_path)
    baseline_questions = [
        dict(SAMPLE_RESULT["per_question"][0]),
        {"question_id": "q_removed", "recall_at_k": 0.2, "reciprocal_rank": 0.2, "ndcg_at_k": 0.2},
    ]
    candidate_questions = [
        dict(SAMPLE_RESULT["per_question"][0]),
        {"question_id": "q_added", "recall_at_k": 0.3, "reciprocal_rank": 0.3, "ndcg_at_k": 0.3},
    ]
    baseline_id = save_run(database_path, make_result(per_question=baseline_questions))
    candidate_id = save_run(database_path, make_result(per_question=candidate_questions))

    comparison = compare_runs(database_path, baseline_id, candidate_id)

    comparisons = {
        question["question_id"]: question
        for question in comparison["question_comparisons"]
    }
    assert comparisons["q_added"]["status"] == "candidate_only"
    assert comparisons["q_added"]["baseline_metrics"] is None
    assert comparisons["q_added"]["differences"] is None
    assert comparisons["q_removed"]["status"] == "baseline_only"
    assert comparisons["q_removed"]["candidate_metrics"] is None
    assert comparisons["q_removed"]["differences"] is None


@pytest.mark.parametrize(
    ("baseline_id", "candidate_id", "missing_name"),
    [(999, 1, "Baseline"), (1, 999, "Candidate")],
)
def test_compare_reports_missing_run_ids(
    tmp_path: Path, baseline_id: int, candidate_id: int, missing_name: str
):
    database_path = tmp_path / "history.sqlite3"
    initialize_database(database_path)
    if baseline_id == 1:
        save_run(database_path, SAMPLE_RESULT)

    with pytest.raises(RunNotFoundError, match=missing_name):
        compare_runs(database_path, baseline_id, candidate_id)


@pytest.mark.parametrize(
    ("candidate_changes", "message"),
    [({"dataset_id": "another_dataset"}, "different dataset IDs"), ({"k": 3}, "different k values")],
)
def test_compare_rejects_incompatible_runs(
    tmp_path: Path, candidate_changes: dict, message: str
):
    database_path = tmp_path / "history.sqlite3"
    initialize_database(database_path)
    baseline_id = save_run(database_path, SAMPLE_RESULT)
    candidate_result = make_result(**candidate_changes)
    candidate_id = save_run(database_path, candidate_result)

    with pytest.raises(IncompatibleRunsError, match=message):
        compare_runs(database_path, baseline_id, candidate_id)
