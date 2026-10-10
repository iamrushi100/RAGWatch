"""Tests for safe local dashboard report discovery and loading."""

import json
from pathlib import Path

import pytest

from src.dashboard_data import (
    ReportLoadError,
    discover_report_files,
    load_report,
)


SAMPLE_REPORT = {
    "dataset_id": "tiny_sample",
    "k": 2,
    "per_question": [
        {
            "question_id": "q_one",
            "recall_at_k": 1.0,
            "reciprocal_rank": 1.0,
            "ndcg_at_k": 1.0,
        }
    ],
    "aggregate_metrics": {"recall_at_k": 1.0, "mrr": 1.0, "ndcg_at_k": 1.0},
}


def write_report(path: Path, report: object = SAMPLE_REPORT) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def test_discovery_returns_nested_json_reports_in_stable_order(tmp_path: Path):
    reports = tmp_path / "reports"
    later = write_report(reports / "zeta" / "report.json")
    earlier = write_report(reports / "alpha.json")
    (reports / "notes.txt").write_text("not a report", encoding="utf-8")

    assert discover_report_files(reports) == [earlier, later]


def test_discovery_returns_empty_list_for_missing_reports_directory(tmp_path: Path):
    assert discover_report_files(tmp_path / "missing") == []


def test_loads_valid_report_with_optional_fields(tmp_path: Path):
    reports = tmp_path / "reports"
    report = {
        **SAMPLE_REPORT,
        "threshold_check": {
            "passed": False,
            "checks": [
                {
                    "metric": "mrr",
                    "actual_score": 0.5,
                    "threshold": 0.8,
                    "passed": False,
                }
            ],
            "alerts": [{"message": "mrr is below its minimum."}],
        },
        "failure_analysis": [
            {
                "question_id": "q_one",
                "diagnostic_labels": ["retrieval_miss"],
                "missing_relevant_document_ids": ["doc_missing"],
            }
        ],
    }
    path = write_report(reports / "report.json", report)

    assert load_report(path, reports) == report


def test_rejects_report_path_outside_reports_directory(tmp_path: Path):
    reports = tmp_path / "reports"
    external_path = write_report(tmp_path / "private.json")

    with pytest.raises(ReportLoadError, match="stored under the reports directory"):
        load_report(external_path, reports)


def test_reports_invalid_json_clearly(tmp_path: Path):
    reports = tmp_path / "reports"
    path = reports / "broken.json"
    path.parent.mkdir()
    path.write_text("{broken", encoding="utf-8")

    with pytest.raises(ReportLoadError, match="contains invalid JSON"):
        load_report(path, reports)


def test_rejects_report_missing_required_metric(tmp_path: Path):
    incomplete = {**SAMPLE_REPORT, "aggregate_metrics": {"mrr": 1.0}}
    path = write_report(tmp_path / "reports" / "incomplete.json", incomplete)

    with pytest.raises(ReportLoadError, match="missing aggregate metric 'recall_at_k'"):
        load_report(path, tmp_path / "reports")


def test_rejects_malformed_optional_failure_analysis(tmp_path: Path):
    report = {**SAMPLE_REPORT, "failure_analysis": [{"question_id": "q_one"}]}
    path = write_report(tmp_path / "reports" / "bad-optional.json", report)

    with pytest.raises(
        ReportLoadError,
        match=r"missing 'failure_analysis\[0\].diagnostic_labels'",
    ):
        load_report(path, tmp_path / "reports")


def test_reports_missing_file_clearly(tmp_path: Path):
    reports = tmp_path / "reports"
    reports.mkdir()

    with pytest.raises(ReportLoadError, match="Could not read report"):
        load_report(reports / "missing.json", reports)
