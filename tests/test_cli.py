"""Tests for the RAGWatch command-line interface."""

import json
from pathlib import Path

import pytest

from src.cli import main


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DATASET = PROJECT_ROOT / "data" / "sample_dataset.json"


def test_cli_prints_question_and_aggregate_metrics(capsys):
    exit_code = main([str(SAMPLE_DATASET), "--k", "2"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "ragwatch_tiny_sample_v1" in output
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
    assert report["dataset_id"] == "ragwatch_tiny_sample_v1"
    assert len(report["per_question"]) == 6
    assert "JSON report saved to" in captured.out


def test_cli_reports_output_write_error(capsys, tmp_path: Path):
    output_is_directory = tmp_path / "existing_directory"
    output_is_directory.mkdir()

    exit_code = main(
        [str(SAMPLE_DATASET), "--k", "2", "--output", str(output_is_directory)]
    )

    assert exit_code == 1
    assert "Error writing JSON report" in capsys.readouterr().err
