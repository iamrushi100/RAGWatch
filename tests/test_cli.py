"""Tests for the RAGWatch command-line interface."""

import json
from pathlib import Path

import pytest

from src.cli import main
from src.history import list_runs


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DATASET = PROJECT_ROOT / "data" / "sample_dataset.json"


def test_cli_prints_question_and_aggregate_metrics(capsys):
    exit_code = main([str(SAMPLE_DATASET), "--k", "2"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "ragwatch_tiny_sample_v2" in output
    assert "q_library_loan_period" in output
    assert "Recall@2" in output
    assert "MRR" in output
    assert "nDCG@2" in output


def test_cli_rejects_invalid_k(capsys):
    with pytest.raises(SystemExit) as error:
        main([str(SAMPLE_DATASET), "--k", "0"])

    assert error.value.code == 2
    assert "k must be a positive integer" in capsys.readouterr().err


def test_cli_reports_invalid_dataset_path(capsys, tmp_path: Path):
    exit_code = main([str(tmp_path / "missing.json"), "--k", "2"])

    assert exit_code == 1
    assert "Error:" in capsys.readouterr().err


def test_cli_reports_malformed_dataset(capsys, tmp_path: Path):
    dataset_path = tmp_path / "malformed.json"
    dataset_path.write_text("{not valid JSON", encoding="utf-8")

    exit_code = main([str(dataset_path), "--k", "2"])

    assert exit_code == 1
    assert "invalid JSON" in capsys.readouterr().err


def test_cli_saves_json_report_to_nested_output_path(capsys, tmp_path: Path):
    output_path = tmp_path / "reports" / "nested" / "evaluation.json"

    exit_code = main(
        [str(SAMPLE_DATASET), "--k", "2", "--output", str(output_path)]
    )

    captured = capsys.readouterr()
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert report["dataset_id"] == "ragwatch_tiny_sample_v2"
    assert len(report["per_question"]) == 6
    assert "JSON report saved to" in captured.out
    assert "threshold_check" not in report
    assert "failure_analysis" not in report


def test_cli_optionally_saves_evaluation_to_sqlite_history(capsys, tmp_path: Path):
    database_path = tmp_path / "history.sqlite3"

    exit_code = main(
        [
            str(SAMPLE_DATASET),
            "--k",
            "2",
            "--history-db",
            str(database_path),
        ]
    )

    runs = list_runs(database_path)
    captured = capsys.readouterr()
    assert exit_code == 0
    assert len(runs) == 1
    assert runs[0]["dataset_id"] == "ragwatch_tiny_sample_v2"
    assert runs[0]["k"] == 2
    assert "SQLite history run saved: 1" in captured.out


def test_cli_reports_invalid_sqlite_history_path(capsys, tmp_path: Path):
    database_path = tmp_path / "missing-parent" / "history.sqlite3"

    exit_code = main(
        [str(SAMPLE_DATASET), "--k", "2", "--history-db", str(database_path)]
    )

    assert exit_code == 1
    assert "Error saving SQLite history" in capsys.readouterr().err


def test_cli_exports_requested_thresholds_and_failure_analysis(capsys, tmp_path: Path):
    output_path = tmp_path / "diagnostic-report.json"

    exit_code = main(
        [
            str(SAMPLE_DATASET),
            "--k",
            "2",
            "--output",
            str(output_path),
            "--min-mrr",
            "0.5",
            "--failure-analysis",
        ]
    )

    captured = capsys.readouterr()
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert report["threshold_check"]["checks"][0]["metric"] == "mrr"
    assert len(report["failure_analysis"]) == len(report["per_question"]) == 6
    assert "Threshold checks:" in captured.out
    assert "Failure analysis:" in captured.out


def test_cli_reports_invalid_threshold_configuration(capsys):
    exit_code = main(
        [str(SAMPLE_DATASET), "--k", "2", "--min-recall-at-k", "1.1"]
    )

    assert exit_code == 1
    assert "thresholds.recall_at_k must be a finite number from 0 to 1" in (
        capsys.readouterr().err
    )


def test_cli_reports_output_write_error(capsys, tmp_path: Path):
    output_is_directory = tmp_path / "existing_directory"
    output_is_directory.mkdir()

    exit_code = main(
        [str(SAMPLE_DATASET), "--k", "2", "--output", str(output_is_directory)]
    )

    assert exit_code == 1
    assert "Error writing JSON report" in capsys.readouterr().err
