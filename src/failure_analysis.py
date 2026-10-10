"""Deterministic per-question retrieval failure diagnostics."""

import math
from typing import Any


def analyze_question(
    dataset: dict[str, Any],
    question_id: str,
    ranked_results: list[tuple[str, float]],
    k: int,
) -> dict[str, Any]:
    """Describe retrieval misses and grade-supported ranking issues for a question.

    ``dataset`` is expected to have the validated shape returned by
    ``src.dataset.load_dataset``. ``ranked_results`` uses the retriever's
    ``(document_id, score)`` pairs, in best-to-worst order. The returned
    retrieved IDs include the full ranking; ``k`` determines which relevant
    IDs count as covered or missed.

    A ``retrieval_miss`` means a positive-grade document is absent from that
    top-k list. A ``ranking_issue`` means a lower-grade relevant document is
    ranked before a higher-grade relevant document in the returned ranking.
    This describes an ordering mismatch supported by the labels; it does not
    identify its cause. ``no_relevant_documents`` and
    ``empty_retrieval_results`` make those edge cases explicit. Invalid inputs
    and unknown document IDs raise ``ValueError`` rather than being treated as
    successful analysis.
    """
    if not isinstance(dataset, dict):
        raise ValueError("dataset must be a dictionary.")
    if not isinstance(question_id, str) or not question_id.strip():
        raise ValueError("question_id must be a non-empty string.")
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive integer.")

    documents = dataset.get("documents")
    queries = dataset.get("queries")
    if not isinstance(documents, list):
        raise ValueError("dataset.documents must be a list.")
    if not isinstance(queries, list):
        raise ValueError("dataset.queries must be a list.")

    document_ids: set[str] = set()
    for index, document in enumerate(documents):
        if not isinstance(document, dict):
            raise ValueError(f"dataset.documents[{index}] must be a dictionary.")
        document_id = document.get("document_id")
        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError(
                f"dataset.documents[{index}].document_id must be a non-empty string."
            )
        if document_id in document_ids:
            raise ValueError(f"Duplicate document_id '{document_id}'.")
        document_ids.add(document_id)

    selected_query = None
    seen_question_ids: set[str] = set()
    for index, query in enumerate(queries):
        if not isinstance(query, dict):
            raise ValueError(f"dataset.queries[{index}] must be a dictionary.")
        candidate_id = query.get("question_id")
        if not isinstance(candidate_id, str) or not candidate_id.strip():
            raise ValueError(
                f"dataset.queries[{index}].question_id must be a non-empty string."
            )
        if candidate_id in seen_question_ids:
            raise ValueError(f"Duplicate question_id '{candidate_id}'.")
        seen_question_ids.add(candidate_id)
        if candidate_id == question_id:
            selected_query = query

    if selected_query is None:
        raise ValueError(f"Question ID '{question_id}' was not found in the dataset.")

    relevance = selected_query.get("relevance")
    if not isinstance(relevance, list):
        raise ValueError(f"Question '{question_id}' relevance must be a list.")

    grades_by_id: dict[str, int] = {}
    for index, label in enumerate(relevance):
        if not isinstance(label, dict):
            raise ValueError(
                f"Question '{question_id}' relevance[{index}] must be a dictionary."
            )
        relevant_id = label.get("document_id")
        if not isinstance(relevant_id, str) or not relevant_id.strip():
            raise ValueError(
                f"Question '{question_id}' relevance[{index}].document_id "
                "must be a non-empty string."
            )
        if relevant_id not in document_ids:
            raise ValueError(
                f"Question '{question_id}' refers to unknown document_id "
                f"'{relevant_id}'."
            )
        if relevant_id in grades_by_id:
            raise ValueError(
                f"Question '{question_id}' has duplicate relevance label "
                f"for document '{relevant_id}'."
            )
        grade = label.get("grade")
        if not isinstance(grade, int) or isinstance(grade, bool) or not 1 <= grade <= 3:
            raise ValueError(
                f"Question '{question_id}' relevance[{index}].grade "
                "must be an integer from 1 to 3."
            )
        grades_by_id[relevant_id] = grade

    if not isinstance(ranked_results, list):
        raise ValueError("ranked_results must be a list of (document_id, score) pairs.")

    ranked_document_ids = []
    seen_ranked_ids: set[str] = set()
    for index, result in enumerate(ranked_results):
        if not isinstance(result, (tuple, list)) or len(result) != 2:
            raise ValueError(
                f"ranked_results[{index}] must be a (document_id, score) pair."
            )
        document_id, score = result
        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError(
                f"ranked_results[{index}].document_id must be a non-empty string."
            )
        if document_id not in document_ids:
            raise ValueError(
                f"ranked_results[{index}] refers to unknown document_id "
                f"'{document_id}'."
            )
        if document_id in seen_ranked_ids:
            raise ValueError(f"Duplicate ranked document ID '{document_id}'.")
        seen_ranked_ids.add(document_id)
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError(f"ranked_results[{index}].score must be a finite number.")
        try:
            finite_score = math.isfinite(score)
        except OverflowError:
            finite_score = False
        if not finite_score:
            raise ValueError(f"ranked_results[{index}].score must be a finite number.")
        ranked_document_ids.append(document_id)

    top_k_ids = ranked_document_ids[:k]
    relevant_ids = sorted(grades_by_id)
    relevant_set = set(relevant_ids)
    retrieved_relevant_ids = [
        doc_id for doc_id in ranked_document_ids if doc_id in relevant_set
    ]
    missing_relevant_ids = [doc_id for doc_id in relevant_ids if doc_id not in top_k_ids]

    # A lower-grade relevant passage above a higher-grade one is a label-backed
    # ordering issue. Track the lower-grade passages while preserving rank order.
    misranked_ids = []
    for earlier_index, earlier_id in enumerate(retrieved_relevant_ids):
        earlier_grade = grades_by_id[earlier_id]
        has_higher_grade_below = any(
            grades_by_id[later_id] > earlier_grade
            for later_id in retrieved_relevant_ids[earlier_index + 1 :]
        )
        if has_higher_grade_below:
            misranked_ids.append(earlier_id)

    diagnostic_labels = []
    if not relevant_ids:
        diagnostic_labels.append("no_relevant_documents")
    if not ranked_document_ids:
        diagnostic_labels.append("empty_retrieval_results")
    if missing_relevant_ids:
        diagnostic_labels.append("retrieval_miss")
    if misranked_ids:
        diagnostic_labels.append("ranking_issue")

    return {
        "question_id": question_id,
        "relevant_document_ids": relevant_ids,
        "retrieved_document_ids": ranked_document_ids,
        "missing_relevant_document_ids": missing_relevant_ids,
        "misranked_relevant_document_ids": misranked_ids,
        "diagnostic_labels": diagnostic_labels,
    }
