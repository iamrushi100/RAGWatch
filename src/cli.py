"""Command-line entry point for local RAGWatch evaluations.

Run from the project root with:

    python -m src.cli data/sample_dataset.json --k 2
"""

import argparse
from collections.abc import Sequence
import sys

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
    args = parser.parse_args(arguments)

    try:
        result = run_evaluation(args.dataset_path, k=args.k)
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
