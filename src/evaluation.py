"""Deterministic retrieval evaluation metrics for RAGWatch."""

import math
from typing import Any


def _relevance_item_id(label: dict[str, Any], index: int) -> str:
    """Read a chunk ID, while retaining support for direct legacy metric calls."""
    has_chunk_id = "chunk_id" in label
    has_document_id = "document_id" in label
    if has_chunk_id == has_document_id:
        raise ValueError(
            f"relevance_labels[{index}] must contain exactly one of "
            "'chunk_id' or 'document_id'."
        )
    field_name = "chunk_id" if has_chunk_id else "document_id"
    item_id = label.get(field_name)
    if not isinstance(item_id, str) or not item_id.strip():
        raise ValueError(
            f"relevance_labels[{index}].{field_name} must be a non-empty string."
        )
    return item_id


def _validated_relevant_ids(
    relevance_labels: list[dict[str, Any]],
    ranked_document_ids: list[str],
) -> set[str]:
    """Validate one query's labels and ranking, returning positive-grade IDs."""
    if not isinstance(relevance_labels, list):
        raise ValueError("relevance_labels must be a list.")
    if not isinstance(ranked_document_ids, list):
        raise ValueError("ranked_document_ids must be a list.")

    relevant_ids: set[str] = set()
    seen_label_ids: set[str] = set()
    for index, label in enumerate(relevance_labels):
        if not isinstance(label, dict):
            raise ValueError(f"relevance_labels[{index}] must be a dictionary.")

        item_id = _relevance_item_id(label, index)
        if item_id in seen_label_ids:
            raise ValueError(f"Duplicate relevance label for item '{item_id}'.")
        seen_label_ids.add(item_id)

        grade = label.get("grade")
        if not isinstance(grade, int) or isinstance(grade, bool) or grade < 0:
            raise ValueError(
                f"relevance_labels[{index}].grade must be a non-negative integer."
            )
        if grade > 0:
            relevant_ids.add(item_id)

    seen_ranked_ids: set[str] = set()
    for index, document_id in enumerate(ranked_document_ids):
        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError(f"ranked_document_ids[{index}] must be a non-empty string.")
        if document_id in seen_ranked_ids:
            raise ValueError(f"Duplicate ranked document ID '{document_id}'.")
        seen_ranked_ids.add(document_id)

    return relevant_ids


def recall_at_k(
    relevance_labels: list[dict[str, Any]],
    ranked_document_ids: list[str],
    k: int,
) -> float:
    """Calculate Recall@k for one question.

    A relevance label with a grade greater than zero is relevant. The result
    is the number of relevant document IDs in the first ``k`` ranked results,
    divided by the total number of relevant document IDs in the labels. If
    there are no relevant labels, the result is 0.0 because recall has no
    positive denominator in that case. A ``k`` larger than the ranking simply
    includes all returned results.

    ``relevance_labels`` should be the question's ``relevance`` list from a
    dataset returned by ``src.dataset.load_dataset``. Ranked IDs may include
    irrelevant documents, but each ID must appear at most once.
    """
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive integer.")

    relevant_ids = _validated_relevant_ids(relevance_labels, ranked_document_ids)
    if not relevant_ids:
        return 0.0

    retrieved_relevant_ids = set(ranked_document_ids[:k]) & relevant_ids
    return len(retrieved_relevant_ids) / len(relevant_ids)


def reciprocal_rank(
    relevance_labels: list[dict[str, Any]],
    ranked_document_ids: list[str],
) -> float:
    """Return the reciprocal rank of the first relevant result for one query.

    A document is relevant when its grade is greater than zero. Ranking
    positions start at 1, so a relevant document at rank 1 contributes 1.0
    and one at rank 3 contributes 1/3. If no relevant document is retrieved,
    including when the ranking or relevant-label list is empty, return 0.0.

    The inputs use the same format and validation rules as ``recall_at_k``.
    This is the per-question value; a future evaluation runner can average
    these values across its query results to calculate MRR.
    """
    relevant_ids = _validated_relevant_ids(relevance_labels, ranked_document_ids)

    for rank, document_id in enumerate(ranked_document_ids, start=1):
        if document_id in relevant_ids:
            return 1 / rank

    return 0.0


def ndcg_at_k(
    relevance_labels: list[dict[str, Any]],
    ranked_document_ids: list[str],
    k: int,
) -> float:
    """Calculate normalized discounted cumulative gain for one question.

    For a result at rank ``r`` (starting at 1), its gain is ``2^grade - 1``
    and its discount is ``log2(r + 1)``. Documents missing from the relevance
    labels have grade 0 and contribute no gain. DCG is the discounted gain
    sum for the first ``k`` retrieved documents. IDCG uses the same formula
    on relevance grades sorted from highest to lowest, limited to ``k``.

    Return ``DCG / IDCG``. If IDCG is zero (there are no positive-grade
    documents), return 0.0, matching the zero-result behavior of the other
    metrics. A ``k`` larger than the ranking or the number of relevant
    documents simply uses the available results and labels.
    """
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive integer.")

    relevant_ids = _validated_relevant_ids(relevance_labels, ranked_document_ids)
    grades_by_id = {
        _relevance_item_id(label, index): label["grade"]
        for index, label in enumerate(relevance_labels)
    }

    def discounted_gain(grade: int, rank: int) -> float:
        return (2**grade - 1) / math.log2(rank + 1)

    dcg = sum(
        discounted_gain(grades_by_id.get(document_id, 0), rank)
        for rank, document_id in enumerate(ranked_document_ids[:k], start=1)
    )

    ideal_grades = sorted(
        (grades_by_id[document_id] for document_id in relevant_ids), reverse=True
    )[:k]
    idcg = sum(
        discounted_gain(grade, rank)
        for rank, grade in enumerate(ideal_grades, start=1)
    )

    if idcg == 0:
        return 0.0
    return dcg / idcg
