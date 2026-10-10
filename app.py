"""Read-only Streamlit dashboard for existing RAGWatch JSON reports."""

from pathlib import Path
import sqlite3

import streamlit as st

from src.dashboard_data import ReportLoadError, discover_report_files, load_report
from src.dashboard_history import (
    compare_history_runs,
    discover_history_databases,
    load_history_runs,
)
from src.history import IncompatibleRunsError, RunNotFoundError


PROJECT_ROOT = Path(__file__).resolve().parent
REPORTS_DIRECTORY = PROJECT_ROOT / "reports"
DATA_DIRECTORY = PROJECT_ROOT / "data"


def main() -> None:
    """Display one selected report; this page never starts an evaluation run."""
    st.set_page_config(page_title="RAGWatch", page_icon="📊", layout="wide")
    st.title("RAGWatch")
    st.caption("Read-only RAG retrieval evaluation reports")

    st.header("JSON evaluation reports")
    report_files = discover_report_files(REPORTS_DIRECTORY)
    if not report_files:
        st.info("No JSON reports found. Export one with the RAGWatch CLI first.")
    else:
        selected_path = st.selectbox(
            "Choose an evaluation report",
            report_files,
            format_func=lambda path: path.relative_to(REPORTS_DIRECTORY).as_posix(),
        )
        try:
            report = load_report(selected_path, REPORTS_DIRECTORY)
        except ReportLoadError as error:
            st.error(str(error))
        else:
            st.subheader(f"Dataset: {report['dataset_id']} (k={report['k']})")
            aggregate = report["aggregate_metrics"]
            metric_columns = st.columns(3)
            metric_columns[0].metric(
                f"Recall@{report['k']}", f"{aggregate['recall_at_k']:.3f}"
            )
            metric_columns[1].metric("MRR", f"{aggregate['mrr']:.3f}")
            metric_columns[2].metric(
                f"nDCG@{report['k']}", f"{aggregate['ndcg_at_k']:.3f}"
            )

            st.subheader("Per-question metrics")
            st.dataframe(report["per_question"], use_container_width=True, hide_index=True)

            threshold_check = report.get("threshold_check")
            if threshold_check is not None:
                st.subheader("Threshold checks")
                if threshold_check["passed"]:
                    st.success("All configured thresholds passed.")
                else:
                    st.error("One or more configured thresholds failed.")
                if threshold_check["checks"]:
                    st.dataframe(
                        threshold_check["checks"],
                        use_container_width=True,
                        hide_index=True,
                    )
                for alert in threshold_check["alerts"]:
                    st.warning(alert["message"])

            failure_analysis = report.get("failure_analysis")
            if failure_analysis is not None:
                st.subheader("Per-question failure analysis")
                analysis_rows = [
                    {
                        "question_id": item["question_id"],
                        "diagnostic_labels": ", ".join(item["diagnostic_labels"])
                        or "none",
                        "missing_relevant_ids": ", ".join(
                            item.get(
                                "missing_relevant_chunk_ids",
                                item.get("missing_relevant_document_ids", []),
                            )
                        )
                        or "none",
                    }
                    for item in failure_analysis
                ]
                st.dataframe(analysis_rows, use_container_width=True, hide_index=True)

    st.divider()
    st.header("SQLite run history")
    database_files = discover_history_databases(DATA_DIRECTORY)
    if not database_files:
        st.info(
            "No SQLite history database found under data/. Run an evaluation with "
            "--history-db data/ragwatch_history.sqlite3 to create one."
        )
        return

    selected_database = st.selectbox(
        "Choose a history database",
        database_files,
        format_func=lambda path: path.relative_to(DATA_DIRECTORY).as_posix(),
    )
    try:
        runs = load_history_runs(selected_database)
    except (OSError, sqlite3.Error) as error:
        st.error(f"Could not read SQLite history: {error}")
        return

    if not runs:
        st.info("This history database has no saved evaluation runs yet.")
        return

    run_rows = [
        {
            "run_id": run["run_id"],
            "dataset_id": run["dataset_id"],
            "k": run["k"],
            "created_at_utc": run["created_at"],
            **run["aggregate_metrics"],
        }
        for run in runs
    ]
    st.subheader("Recent runs (newest first)")
    st.dataframe(run_rows, use_container_width=True, hide_index=True)

    if len(runs) < 2:
        st.info("Save at least two runs to compare them.")
        return

    run_ids = [run["run_id"] for run in runs]
    baseline_id = st.selectbox(
        "Baseline run",
        run_ids,
        index=len(run_ids) - 1,
        format_func=lambda run_id: f"Run {run_id}",
    )
    candidate_id = st.selectbox(
        "Candidate run",
        run_ids,
        index=0,
        format_func=lambda run_id: f"Run {run_id}",
    )
    try:
        comparison = compare_history_runs(
            selected_database, baseline_id, candidate_id
        )
    except RunNotFoundError as error:
        st.warning(str(error))
        return
    except IncompatibleRunsError as error:
        st.warning(str(error))
        return
    except (OSError, sqlite3.Error) as error:
        st.error(f"Could not compare saved runs: {error}")
        return

    st.subheader("Aggregate candidate minus baseline")
    st.dataframe(
        [
            {"metric": name, "difference": value}
            for name, value in comparison["aggregate_differences"].items()
        ],
        use_container_width=True,
        hide_index=True,
    )
    st.subheader("Per-question comparison")
    st.dataframe(
        comparison["question_comparisons"], use_container_width=True, hide_index=True
    )


if __name__ == "__main__":
    main()
