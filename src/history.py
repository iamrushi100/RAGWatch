"""Save completed evaluation results in a small local SQLite database."""

import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def _connect(database_path: str | Path) -> sqlite3.Connection:
    """Open a database connection with foreign-key checks enabled."""
    connection = sqlite3.connect(database_path)
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_database(database_path: str | Path) -> None:
    """Create the run-history tables if they do not already exist."""
    connection = _connect(database_path)
    try:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS evaluation_runs (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_id TEXT NOT NULL,
                k INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                recall_at_k REAL NOT NULL,
                mrr REAL NOT NULL,
                ndcg_at_k REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS question_results (
                run_id INTEGER NOT NULL,
                question_id TEXT NOT NULL,
                recall_at_k REAL NOT NULL,
                reciprocal_rank REAL NOT NULL,
                ndcg_at_k REAL NOT NULL,
                PRIMARY KEY (run_id, question_id),
                FOREIGN KEY (run_id) REFERENCES evaluation_runs(run_id)
                    ON DELETE CASCADE
            );
            """
        )
    finally:
        connection.close()


def _validate_result(result: object) -> tuple[str, int, dict[str, float], list[dict]]:
    """Check that every field required by the runner's result is present."""
    if not isinstance(result, dict):
        raise ValueError("evaluation_result must be a dictionary.")

    dataset_id = result.get("dataset_id")
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise ValueError("evaluation_result.dataset_id must be a non-empty string.")

    k = result.get("k")
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("evaluation_result.k must be a positive integer.")

    aggregate = result.get("aggregate_metrics")
    if not isinstance(aggregate, dict):
        raise ValueError("evaluation_result.aggregate_metrics must be a dictionary.")

    aggregate_scores = {
        "recall_at_k": _validate_score(aggregate, "recall_at_k", "aggregate_metrics"),
        "mrr": _validate_score(aggregate, "mrr", "aggregate_metrics"),
        "ndcg_at_k": _validate_score(aggregate, "ndcg_at_k", "aggregate_metrics"),
    }

    questions = result.get("per_question")
    if not isinstance(questions, list):
        raise ValueError("evaluation_result.per_question must be a list.")

    checked_questions = []
    seen_question_ids = set()
    for index, question in enumerate(questions):
        location = f"per_question[{index}]"
        if not isinstance(question, dict):
            raise ValueError(f"evaluation_result.{location} must be a dictionary.")

        question_id = question.get("question_id")
        if not isinstance(question_id, str) or not question_id.strip():
            raise ValueError(
                f"evaluation_result.{location}.question_id must be a non-empty string."
            )
        if question_id in seen_question_ids:
            raise ValueError(f"Duplicate question_id '{question_id}' in evaluation result.")
        seen_question_ids.add(question_id)

        checked_questions.append(
            {
                "question_id": question_id,
                "recall_at_k": _validate_score(question, "recall_at_k", location),
                "reciprocal_rank": _validate_score(question, "reciprocal_rank", location),
                "ndcg_at_k": _validate_score(question, "ndcg_at_k", location),
            }
        )

    return dataset_id, k, aggregate_scores, checked_questions


def _validate_score(values: dict, name: str, location: str) -> float:
    """Return a finite metric score from zero through one, or explain the error."""
    score = values.get(name)
    if (
        not isinstance(score, (int, float))
        or isinstance(score, bool)
        or not math.isfinite(score)
        or not 0 <= score <= 1
    ):
        raise ValueError(
            f"evaluation_result.{location}.{name} must be a finite number from 0 to 1."
        )
    return float(score)


def save_run(database_path: str | Path, evaluation_result: object) -> int:
    """Save one runner result atomically and return its integer run ID.

    Call ``initialize_database`` first. The timestamp is UTC in ISO 8601
    format with microseconds and a ``Z`` suffix, for example
    ``2026-10-10T12:30:45.123456Z``. The parent run row and every question
    row are committed together or rolled back together.
    """
    dataset_id, k, aggregate, questions = _validate_result(evaluation_result)
    created_at = datetime.now(timezone.utc).isoformat(timespec="microseconds")
    created_at = created_at.replace("+00:00", "Z")

    connection = _connect(database_path)
    try:
        with connection:
            cursor = connection.execute(
                """
                INSERT INTO evaluation_runs (
                    dataset_id, k, created_at, recall_at_k, mrr, ndcg_at_k
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    dataset_id,
                    k,
                    created_at,
                    aggregate["recall_at_k"],
                    aggregate["mrr"],
                    aggregate["ndcg_at_k"],
                ),
            )
            run_id = cursor.lastrowid
            if run_id is None:
                raise RuntimeError("SQLite did not return a run ID for the saved run.")

            connection.executemany(
                """
                INSERT INTO question_results (
                    run_id, question_id, recall_at_k, reciprocal_rank, ndcg_at_k
                ) VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        run_id,
                        question["question_id"],
                        question["recall_at_k"],
                        question["reciprocal_rank"],
                        question["ndcg_at_k"],
                    )
                    for question in questions
                ],
            )
        return int(run_id)
    finally:
        connection.close()
