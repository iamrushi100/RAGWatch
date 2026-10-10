"""Run the local retrieval evaluation for every question in a dataset."""

from pathlib import Path
from typing import Any

from src.dataset import load_dataset
from src.evaluation import ndcg_at_k, recall_at_k, reciprocal_rank
from src.retriever import rank_documents


def run_evaluation(dataset_path: str | Path, k: int) -> dict[str, Any]:
    """Load a dataset, evaluate each question, and return per-question and mean scores.

    Each question's metric values count equally in the aggregates: Recall@k
    and nDCG@k are averaged, and MRR is the average reciprocal rank. A
    question with no relevant labels therefore contributes the metrics'
    documented zero values. If the dataset has no questions, the per-question
    list is empty and each aggregate is 0.0. Errors during loading, retrieval,
    or metric calculation are allowed to stop the run; a failed question is
    never silently left out of the averages.
    """
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive integer.")

    dataset = load_dataset(dataset_path)
    per_question = []

    for query in dataset["queries"]:
        ranked_results = rank_documents(dataset, query["question"])
        ranked_document_ids = [document_id for document_id, _ in ranked_results]
        relevance_labels = query["relevance"]

        per_question.append(
            {
                "question_id": query["question_id"],
                "recall_at_k": recall_at_k(relevance_labels, ranked_document_ids, k),
                "reciprocal_rank": reciprocal_rank(
                    relevance_labels, ranked_document_ids
                ),
                "ndcg_at_k": ndcg_at_k(relevance_labels, ranked_document_ids, k),
            }
        )

    question_count = len(per_question)
    if question_count:
        aggregate_metrics = {
            "recall_at_k": sum(
                result["recall_at_k"] for result in per_question
            ) / question_count,
            "mrr": sum(
                result["reciprocal_rank"] for result in per_question
            ) / question_count,
            "ndcg_at_k": sum(
                result["ndcg_at_k"] for result in per_question
            ) / question_count,
        }
    else:
        aggregate_metrics = {
            "recall_at_k": 0.0,
            "mrr": 0.0,
            "ndcg_at_k": 0.0,
        }

    return {
        "dataset_id": dataset["dataset_id"],
        "k": k,
        "per_question": per_question,
        "aggregate_metrics": aggregate_metrics,
    }
