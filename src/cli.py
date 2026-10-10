"""Command-line entry point for local RAGWatch evaluations.

Run from the project root with:

    python -m src.cli data/sample_dataset.json --k 2
"""

import argparse
from collections.abc import Sequence
import sqlite3
import sys

from src.history import initialize_database, save_run
from src.reporting import export_json_report
from src.runner import run_evaluation


def _positive_integer(value: str) -> int:
    """Convert a command-line value to a positive integer or show a clear error."""
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("k must be a positive integer") from error

    if number < 1:
        raise argparse.ArgumentTypeError("k must be a positive integer")
    return number


def main(arguments: Sequence[str] | None = None) -> int:
    """Run one evaluation and print its question-level and aggregate metrics."""
    parser = argparse.ArgumentParser(
        description="Run the local RAGWatch retrieval evaluation."
    )
    parser.add_argument("dataset_path", help="path to the evaluation dataset JSON file")
    parser.add_argument(
        "--k",
        type=_positive_integer,
        required=True,
        help="positive result cutoff for Recall@k and nDCG@k",
    )
    parser.add_argument(
        "--output",
        help="optional path for saving the complete evaluation result as JSON",
    )
    parser.add_argument(
        "--history-db",
        help="optional SQLite path for saving this evaluation in run history",
    )
    parser.add_argument(
        "--min-recall-at-k",
        type=float,
        help="optional minimum acceptable aggregate Recall@k score (0 to 1)",
    )
    parser.add_argument(
        "--min-mrr",
        type=float,
        help="optional minimum acceptable aggregate MRR score (0 to 1)",
    )
    parser.add_argument(
        "--min-ndcg-at-k",
        type=float,
        help="optional minimum acceptable aggregate nDCG@k score (0 to 1)",
    )
    parser.add_argument(
        "--failure-analysis",
        action="store_true",
        help="include deterministic per-question retrieval diagnostics",
    )
    args = parser.parse_args(arguments)

    thresholds = {}
    if args.min_recall_at_k is not None:
        thresholds["recall_at_k"] = args.min_recall_at_k
    if args.min_mrr is not None:
        thresholds["mrr"] = args.min_mrr
    if args.min_ndcg_at_k is not None:
        thresholds["ndcg_at_k"] = args.min_ndcg_at_k

    try:
        result = run_evaluation(
            args.dataset_path,
            k=args.k,
            thresholds=thresholds if thresholds else None,
            include_failure_analysis=args.failure_analysis,
        )
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    if args.history_db is not None:
        try:
            initialize_database(args.history_db)
            run_id = save_run(args.history_db, result)
        except (OSError, sqlite3.Error, ValueError) as error:
            print(f"Error saving SQLite history to '{args.history_db}': {error}", file=sys.stderr)
            return 1
        print(f"SQLite history run saved: {run_id}")

    if args.output:
        try:
            report_path = export_json_report(result, args.output)
        except OSError as error:
            print(f"Error writing JSON report to '{args.output}': {error}", file=sys.stderr)
            return 1
        print(f"JSON report saved to: {report_path}")

    print(f"Dataset: {result['dataset_id']} (k={result['k']})")
    print("Per-question results:")
    for question_result in result["per_question"]:
        print(
            f"  {question_result['question_id']}: "
            f"Recall@{args.k}={question_result['recall_at_k']:.3f}, "
            f"RR={question_result['reciprocal_rank']:.3f}, "
            f"nDCG@{args.k}={question_result['ndcg_at_k']:.3f}"
        )

    aggregate = result["aggregate_metrics"]
    print("Aggregate metrics:")
    print(f"  Recall@{args.k}: {aggregate['recall_at_k']:.3f}")
    print(f"  MRR: {aggregate['mrr']:.3f}")
    print(f"  nDCG@{args.k}: {aggregate['ndcg_at_k']:.3f}")

    threshold_check = result.get("threshold_check")
    if threshold_check is not None:
        print("Threshold checks:")
        print(f"  Overall: {'PASS' if threshold_check['passed'] else 'FAIL'}")
        for alert in threshold_check["alerts"]:
            print(f"  Alert: {alert['message']}")

    if args.failure_analysis:
        print("Failure analysis:")
        for analysis in result["failure_analysis"]:
            labels = ", ".join(analysis["diagnostic_labels"]) or "none"
            print(f"  {analysis['question_id']}: {labels}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
