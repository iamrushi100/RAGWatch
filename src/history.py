"""Save completed evaluation results in a small local SQLite database."""

import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


class RunNotFoundError(LookupError):
    """Raised when a requested saved evaluation run does not exist."""


class IncompatibleRunsError(ValueError):
    """Raised when two runs use different datasets or retrieval cutoffs."""


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


def list_runs(database_path: str | Path) -> list[dict]:
    """Return saved run summaries, newest first by increasing run ID.

    Each item contains ``run_id``, ``dataset_id``, ``k``, ``created_at``,
    and an ``aggregate_metrics`` dictionary. Results are ordered by descending
    run ID, which is the save order. An initialized database with no saved
    runs returns an empty list.
    """
    connection = _connect(database_path)
    try:
        rows = connection.execute(
            """
            SELECT run_id, dataset_id, k, created_at,
                   recall_at_k, mrr, ndcg_at_k
            FROM evaluation_runs
            ORDER BY run_id DESC
            """
        ).fetchall()
    finally:
        connection.close()

    return [
        {
            "run_id": row[0],
            "dataset_id": row[1],
            "k": row[2],
            "created_at": row[3],
            "aggregate_metrics": {
                "recall_at_k": row[4],
                "mrr": row[5],
                "ndcg_at_k": row[6],
            },
        }
        for row in rows
    ]


def _validate_run_id(run_id: int, name: str) -> None:
    if not isinstance(run_id, int) or isinstance(run_id, bool) or run_id < 1:
        raise ValueError(f"{name} must be a positive integer.")


def _read_run(connection: sqlite3.Connection, run_id: int) -> dict | None:
    row = connection.execute(
        """
        SELECT run_id, dataset_id, k, created_at,
               recall_at_k, mrr, ndcg_at_k
        FROM evaluation_runs
        WHERE run_id = ?
        """,
        (run_id,),
    ).fetchone()
    if row is None:
        return None

    question_rows = connection.execute(
        """
        SELECT question_id, recall_at_k, reciprocal_rank, ndcg_at_k
        FROM question_results
        WHERE run_id = ?
        ORDER BY question_id
        """,
        (run_id,),
    ).fetchall()
    return {
        "run_id": row[0],
        "dataset_id": row[1],
        "k": row[2],
        "created_at": row[3],
        "aggregate_metrics": {
            "recall_at_k": row[4],
            "mrr": row[5],
            "ndcg_at_k": row[6],
        },
        "questions": {
            question_row[0]: {
                "recall_at_k": question_row[1],
                "reciprocal_rank": question_row[2],
                "ndcg_at_k": question_row[3],
            }
            for question_row in question_rows
        },
    }


def compare_runs(
    database_path: str | Path, baseline_run_id: int, candidate_run_id: int
) -> dict:
    """Compare saved runs and return candidate-minus-baseline differences.

    Both IDs must exist, and the runs must have the same ``dataset_id`` and
    ``k``. Missing runs raise ``RunNotFoundError``; incompatible runs raise
    ``IncompatibleRunsError``. Question rows are matched by ``question_id``,
    and comparisons are returned in question-ID order. Questions found in
    only one run have a ``baseline_only`` or ``candidate_only`` status and
    ``None`` for their differences.
    """
    _validate_run_id(baseline_run_id, "baseline_run_id")
    _validate_run_id(candidate_run_id, "candidate_run_id")

    connection = _connect(database_path)
    try:
        baseline = _read_run(connection, baseline_run_id)
        if baseline is None:
            raise RunNotFoundError(f"Baseline run ID {baseline_run_id} was not found.")

        candidate = _read_run(connection, candidate_run_id)
        if candidate is None:
            raise RunNotFoundError(f"Candidate run ID {candidate_run_id} was not found.")

        if baseline["dataset_id"] != candidate["dataset_id"]:
            raise IncompatibleRunsError(
                "Cannot compare runs with different dataset IDs "
                f"('{baseline['dataset_id']}' and '{candidate['dataset_id']}')."
            )
        if baseline["k"] != candidate["k"]:
            raise IncompatibleRunsError(
                f"Cannot compare runs with different k values "
                f"({baseline['k']} and {candidate['k']})."
            )
    finally:
        connection.close()

    aggregate_metric_names = ("recall_at_k", "mrr", "ndcg_at_k")
    aggregate_differences = {
        name: candidate["aggregate_metrics"][name]
        - baseline["aggregate_metrics"][name]
        for name in aggregate_metric_names
    }

    question_metric_names = ("recall_at_k", "reciprocal_rank", "ndcg_at_k")
    question_comparisons = []
    all_question_ids = sorted(set(baseline["questions"]) | set(candidate["questions"]))
    for question_id in all_question_ids:
        baseline_metrics = baseline["questions"].get(question_id)
        candidate_metrics = candidate["questions"].get(question_id)

        if baseline_metrics is None:
            status = "candidate_only"
            differences = None
        elif candidate_metrics is None:
            status = "baseline_only"
            differences = None
        else:
            status = "matched"
            differences = {
                name: candidate_metrics[name] - baseline_metrics[name]
                for name in question_metric_names
            }

        question_comparisons.append(
            {
                "question_id": question_id,
                "status": status,
                "baseline_metrics": baseline_metrics,
                "candidate_metrics": candidate_metrics,
                "differences": differences,
            }
        )

    return {
        "baseline": {
            key: value for key, value in baseline.items() if key != "questions"
        },
        "candidate": {
            key: value for key, value in candidate.items() if key != "questions"
        },
        "aggregate_differences": aggregate_differences,
        "question_comparisons": question_comparisons,
    }
