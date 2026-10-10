"""Run a local, controlled ranking-inversion regression experiment.

Run from the project root with::

    python -m src.regression_experiment
"""

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys
from typing import Any

from src.dataset import load_dataset
from src.reporting import export_json_report
from src.retriever import rank_documents
from src.runner import _evaluate_loaded_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "data" / "sample_dataset.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "retrieval-regression-experiment.json"


def _reverse_baseline_ranking(
    dataset: dict[str, Any], question: str
) -> list[tuple[str, float]]:
    """Simulate a severe ranking regression by reversing the baseline order."""
    return list(reversed(rank_documents(dataset, question)))


def run_regression_experiment(
    dataset_path: str | Path = DEFAULT_DATASET,
    k: int = 2,
) -> dict[str, Any]:
    """Compare TF-IDF with an intentionally reversed ranking on one dataset.

    Both conditions use the same loaded chunks, questions, relevance labels,
    and metric implementations. Only the order of each baseline ranking is
    reversed for the degraded condition.
    """
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive integer.")

    dataset = load_dataset(dataset_path)
    baseline = _evaluate_loaded_dataset(dataset, k, rank_documents)
    degraded = _evaluate_loaded_dataset(dataset, k, _reverse_baseline_ranking)

    metric_differences = {
        metric_name: (
            degraded["aggregate_metrics"][metric_name]
            - baseline["aggregate_metrics"][metric_name]
        )
        for metric_name in ("recall_at_k", "mrr", "ndcg_at_k")
    }
    return {
        "experiment_id": "controlled-ranking-inversion-v1",
        "dataset_id": dataset["dataset_id"],
        "k": k,
        "question_count": len(dataset["queries"]),
        "degradation": {
            "name": "reverse_baseline_ranking",
            "description": (
                "Reverse the complete TF-IDF ranking for every question. "
                "This deliberately severe perturbation demonstrates that "
                "the evaluation detects a ranking regression; it is not a "
                "realistic retrieval strategy."
            ),
        },
        "baseline": {
            "aggregate_metrics": baseline["aggregate_metrics"],
            "per_question": baseline["per_question"],
        },
        "degraded": {
            "aggregate_metrics": degraded["aggregate_metrics"],
            "per_question": degraded["per_question"],
        },
        "candidate_minus_baseline": metric_differences,
    }


def _positive_integer(value: str) -> int:
    """Parse a positive integer CLI option."""
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("k must be a positive integer") from error
    if number < 1:
        raise argparse.ArgumentTypeError("k must be a positive integer")
    return number


def main(arguments: Sequence[str] | None = None) -> int:
    """Run the experiment and save its result as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET),
        help="dataset JSON path (defaults to the fictional sample dataset)",
    )
    parser.add_argument(
        "--k", type=_positive_integer, default=2, help="positive metric cutoff"
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="JSON result path (defaults to reports/retrieval-regression-experiment.json)",
    )
    args = parser.parse_args(arguments)

    try:
        dataset_path = Path(args.dataset)
        output_path = Path(args.output)
        if output_path.resolve() == dataset_path.resolve():
            raise ValueError("The output path must not be the dataset path.")
        if output_path.exists() or output_path.is_symlink():
            raise ValueError(
                f"Output file '{output_path}' already exists; choose a new path."
            )

        result = run_regression_experiment(args.dataset, args.k)
        output_path = export_json_report(result, output_path)
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Regression experiment report saved to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
