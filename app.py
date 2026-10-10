"""Read-only Streamlit dashboard for existing RAGWatch JSON reports."""

from pathlib import Path

import streamlit as st

from src.dashboard_data import ReportLoadError, discover_report_files, load_report


PROJECT_ROOT = Path(__file__).resolve().parent
REPORTS_DIRECTORY = PROJECT_ROOT / "reports"


def main() -> None:
    """Display one selected report; this page never starts an evaluation run."""
    st.set_page_config(page_title="RAGWatch", page_icon="📊", layout="wide")
    st.title("RAGWatch")
    st.caption("Read-only RAG retrieval evaluation reports")

    report_files = discover_report_files(REPORTS_DIRECTORY)
    if not report_files:
        st.info("No JSON reports found. Export a report with the RAGWatch CLI first.")
        return

    selected_path = st.selectbox(
        "Choose an evaluation report",
        report_files,
        format_func=lambda path: path.relative_to(REPORTS_DIRECTORY).as_posix(),
    )
    try:
        report = load_report(selected_path, REPORTS_DIRECTORY)
    except ReportLoadError as error:
        st.error(str(error))
        return

    st.subheader(f"Dataset: {report['dataset_id']} (k={report['k']})")
    aggregate = report["aggregate_metrics"]
    metric_columns = st.columns(3)
    metric_columns[0].metric(f"Recall@{report['k']}", f"{aggregate['recall_at_k']:.3f}")
    metric_columns[1].metric("MRR", f"{aggregate['mrr']:.3f}")
    metric_columns[2].metric(f"nDCG@{report['k']}", f"{aggregate['ndcg_at_k']:.3f}")

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
            st.dataframe(threshold_check["checks"], use_container_width=True, hide_index=True)
        for alert in threshold_check["alerts"]:
            st.warning(alert["message"])

    failure_analysis = report.get("failure_analysis")
    if failure_analysis is not None:
        st.subheader("Per-question failure analysis")
        analysis_rows = [
            {
                "question_id": item["question_id"],
                "diagnostic_labels": ", ".join(item["diagnostic_labels"]) or "none",
                "missing_relevant_document_ids": ", ".join(
                    item["missing_relevant_document_ids"]
                ) or "none",
            }
            for item in failure_analysis
        ]
        st.dataframe(analysis_rows, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
