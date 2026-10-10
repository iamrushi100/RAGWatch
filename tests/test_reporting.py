"""Tests for JSON evaluation report export."""

import json
from pathlib import Path

import pytest

from src.reporting import export_json_report


SAMPLE_RESULT = {
    "dataset_id": "small_fixture",
    "k": 2,
    "per_question": [
        {
            "question_id": "q1",
            "recall_at_k": 1.0,
            "reciprocal_rank": 1.0,
            "ndcg_at_k": 1.0,
        }
    ],
    "aggregate_metrics": {"recall_at_k": 1.0, "mrr": 1.0, "ndcg_at_k": 1.0},
}


def test_export_writes_valid_json_with_runner_results(tmp_path: Path):
    report_path = tmp_path / "report.json"

    returned_path = export_json_report(SAMPLE_RESULT, report_path)

    assert returned_path == report_path
    assert json.loads(report_path.read_text(encoding="utf-8")) == SAMPLE_RESULT
    assert report_path.read_text(encoding="utf-8").endswith("\n")


def test_export_creates_nested_parent_directories(tmp_path: Path):
    report_path = tmp_path / "reports" / "daily" / "run.json"

    export_json_report(SAMPLE_RESULT, report_path)

    assert report_path.is_file()
    assert json.loads(report_path.read_text(encoding="utf-8")) == SAMPLE_RESULT


def test_export_surfaces_file_system_errors(tmp_path: Path):
    output_is_directory = tmp_path / "existing_directory"
    output_is_directory.mkdir()

    with pytest.raises(OSError):
        export_json_report(SAMPLE_RESULT, output_is_directory)
