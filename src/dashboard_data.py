"""Safe discovery and validation helpers for local JSON evaluation reports."""

import json
import math
from pathlib import Path
from typing import Any


class ReportLoadError(ValueError):
    """Raised when an evaluation report cannot be safely loaded or displayed."""


def _inside_directory(path: Path, directory: Path) -> bool:
    """Return whether a resolved path is inside the resolved report directory."""
    try:
        path.resolve().relative_to(directory.resolve())
    except (OSError, ValueError):
        return False
    return True


def discover_report_files(reports_directory: str | Path) -> list[Path]:
    """Return JSON report files inside a report directory in stable order.

    Symlinks that resolve outside the directory are ignored. A missing report
    directory yields an empty list so the dashboard can show a useful message.
    """
    root = Path(reports_directory)
    if not root.is_dir():
        return []

    files = []
    for candidate in root.rglob("*.json"):
        if candidate.is_file() and _inside_directory(candidate, root):
            files.append(candidate)
    return sorted(files, key=lambda path: path.relative_to(root).as_posix().casefold())


def _validate_score(value: Any, field_name: str) -> None:
    """Check that a displayed metric is a finite number from zero to one."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReportLoadError(f"Report field '{field_name}' must be a score from 0 to 1.")
    try:
        score = float(value)
    except OverflowError as error:
        raise ReportLoadError(
            f"Report field '{field_name}' must be a score from 0 to 1."
        ) from error
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ReportLoadError(f"Report field '{field_name}' must be a score from 0 to 1.")


def _validate_report_shape(report: Any) -> dict[str, Any]:
    """Validate fields the dashboard needs, including optional diagnostics."""
    if not isinstance(report, dict):
        raise ReportLoadError("The JSON report must contain an object at the top level.")

    dataset_id = report.get("dataset_id")
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise ReportLoadError("Report field 'dataset_id' must be a non-empty string.")

    k = report.get("k")
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ReportLoadError("Report field 'k' must be a positive integer.")

    aggregate = report.get("aggregate_metrics")
    if not isinstance(aggregate, dict):
        raise ReportLoadError("Report field 'aggregate_metrics' must be an object.")
    for name in ("recall_at_k", "mrr", "ndcg_at_k"):
        if name not in aggregate:
            raise ReportLoadError(f"Report is missing aggregate metric '{name}'.")
        _validate_score(aggregate[name], f"aggregate_metrics.{name}")

    questions = report.get("per_question")
    if not isinstance(questions, list):
        raise ReportLoadError("Report field 'per_question' must be a list.")
    for index, question in enumerate(questions):
        location = f"per_question[{index}]"
        if not isinstance(question, dict):
            raise ReportLoadError(f"Report field '{location}' must be an object.")
        question_id = question.get("question_id")
        if not isinstance(question_id, str) or not question_id.strip():
            raise ReportLoadError(f"Report field '{location}.question_id' must be a string.")
        for name in ("recall_at_k", "reciprocal_rank", "ndcg_at_k"):
            if name not in question:
                raise ReportLoadError(f"Report is missing '{location}.{name}'.")
            _validate_score(question[name], f"{location}.{name}")

    if "threshold_check" in report:
        threshold_check = report["threshold_check"]
        if (
            not isinstance(threshold_check, dict)
            or not isinstance(threshold_check.get("passed"), bool)
            or not isinstance(threshold_check.get("checks"), list)
            or not isinstance(threshold_check.get("alerts"), list)
        ):
            raise ReportLoadError("Report field 'threshold_check' has an invalid structure.")
        for index, check in enumerate(threshold_check["checks"]):
            if not isinstance(check, dict) or not isinstance(check.get("metric"), str):
                raise ReportLoadError(
                    f"Report field 'threshold_check.checks[{index}]' is invalid."
                )
            _validate_score(
                check.get("actual_score"),
                f"threshold_check.checks[{index}].actual_score",
            )
            _validate_score(
                check.get("threshold"), f"threshold_check.checks[{index}].threshold"
            )
            if not isinstance(check.get("passed"), bool):
                raise ReportLoadError(
                    f"Report field 'threshold_check.checks[{index}].passed' must be boolean."
                )
        for index, alert in enumerate(threshold_check["alerts"]):
            if not isinstance(alert, dict) or not isinstance(alert.get("message"), str):
                raise ReportLoadError(
                    f"Report field 'threshold_check.alerts[{index}]' is invalid."
                )

    if "failure_analysis" in report:
        analyses = report["failure_analysis"]
        if not isinstance(analyses, list):
            raise ReportLoadError("Report field 'failure_analysis' must be a list.")
        for index, analysis in enumerate(analyses):
            if not isinstance(analysis, dict):
                raise ReportLoadError(
                    f"Report field 'failure_analysis[{index}]' must be an object."
                )
            for name in (
                "question_id",
                "diagnostic_labels",
                "missing_relevant_document_ids",
            ):
                if name not in analysis:
                    raise ReportLoadError(
                        f"Report is missing 'failure_analysis[{index}].{name}'."
                    )
            if not isinstance(analysis["question_id"], str):
                raise ReportLoadError(
                    f"Report field 'failure_analysis[{index}].question_id' must be a string."
                )
            if not isinstance(analysis["diagnostic_labels"], list) or not all(
                isinstance(label, str) for label in analysis["diagnostic_labels"]
            ):
                raise ReportLoadError(
                    f"Report field 'failure_analysis[{index}].diagnostic_labels' "
                    "must be a list of strings."
                )
            if not isinstance(analysis["missing_relevant_document_ids"], list) or not all(
                isinstance(document_id, str)
                for document_id in analysis["missing_relevant_document_ids"]
            ):
                raise ReportLoadError(
                    f"Report field 'failure_analysis[{index}].missing_relevant_document_ids' "
                    "must be a list of strings."
                )

    return report


def load_report(report_path: str | Path, reports_directory: str | Path) -> dict[str, Any]:
    """Load one report only if it is a JSON file inside the approved directory."""
    root = Path(reports_directory)
    path = Path(report_path)
    if path.suffix.lower() != ".json" or not _inside_directory(path, root):
        raise ReportLoadError("Choose a JSON report stored under the reports directory.")

    try:
        with path.open("r", encoding="utf-8") as report_file:
            report = json.load(report_file)
    except (OSError, UnicodeError) as error:
        raise ReportLoadError(f"Could not read report '{path}': {error}") from error
    except json.JSONDecodeError as error:
        raise ReportLoadError(
            f"Report '{path}' contains invalid JSON at line {error.lineno}, "
            f"column {error.colno}."
        ) from error

    return _validate_report_shape(report)
