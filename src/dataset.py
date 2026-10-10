"""Load and validate a RAGWatch evaluation dataset."""

import json
from pathlib import Path
from typing import Any


class DatasetError(ValueError):
    """Raised when a dataset file cannot be loaded or is invalid."""


def load_dataset(file_path: str | Path) -> dict[str, Any]:
    """Load a JSON dataset and validate its required fields and references.

    The returned dictionary has the same structure as the JSON file. This
    function checks the dataset shape and IDs; it does not evaluate retrieval.
    """
    path = Path(file_path)

    try:
        with path.open("r", encoding="utf-8") as dataset_file:
            data = json.load(dataset_file)
    except OSError as error:
        raise DatasetError(f"Could not read dataset file '{path}': {error}") from error
    except json.JSONDecodeError as error:
        raise DatasetError(
            f"Dataset file '{path}' contains invalid JSON: {error.msg} "
            f"(line {error.lineno}, column {error.colno})."
        ) from error

    if not isinstance(data, dict):
        raise DatasetError("The dataset must be a JSON object at the top level.")

    required_fields = {"dataset_id", "schema_version", "documents", "queries"}
    missing_fields = required_fields - data.keys()
    if missing_fields:
        names = ", ".join(sorted(missing_fields))
        raise DatasetError(f"The dataset is missing required top-level field(s): {names}.")

    if not isinstance(data["dataset_id"], str) or not data["dataset_id"].strip():
        raise DatasetError("'dataset_id' must be a non-empty string.")
    if (
        not isinstance(data["schema_version"], int)
        or isinstance(data["schema_version"], bool)
        or data["schema_version"] < 1
    ):
        raise DatasetError("'schema_version' must be a positive integer.")

    documents = data["documents"]
    if not isinstance(documents, list):
        raise DatasetError("'documents' must be a list.")

    document_ids: set[str] = set()
    for index, document in enumerate(documents):
        location = f"documents[{index}]"
        if not isinstance(document, dict):
            raise DatasetError(f"{location} must be a JSON object.")
        document_id = document.get("document_id")
        if not isinstance(document_id, str) or not document_id.strip():
            raise DatasetError(f"{location}.document_id must be a non-empty string.")
        if document_id in document_ids:
            raise DatasetError(f"Duplicate document_id '{document_id}'.")
        document_ids.add(document_id)
        text = document.get("text")
        if not isinstance(text, str) or not text.strip():
            raise DatasetError(f"{location}.text must be a non-empty string.")

    queries = data["queries"]
    if not isinstance(queries, list):
        raise DatasetError("'queries' must be a list.")

    question_ids: set[str] = set()
    for index, query in enumerate(queries):
        location = f"queries[{index}]"
        if not isinstance(query, dict):
            raise DatasetError(f"{location} must be a JSON object.")

        question_id = query.get("question_id")
        if not isinstance(question_id, str) or not question_id.strip():
            raise DatasetError(f"{location}.question_id must be a non-empty string.")
        if question_id in question_ids:
            raise DatasetError(f"Duplicate question_id '{question_id}'.")
        question_ids.add(question_id)

        question = query.get("question")
        if not isinstance(question, str) or not question.strip():
            raise DatasetError(f"{location}.question must be a non-empty string.")

        relevance = query.get("relevance")
        if not isinstance(relevance, list):
            raise DatasetError(f"{location}.relevance must be a list.")

        judged_document_ids: set[str] = set()
        for relevance_index, label in enumerate(relevance):
            label_location = f"{location}.relevance[{relevance_index}]"
            if not isinstance(label, dict):
                raise DatasetError(f"{label_location} must be a JSON object.")

            relevant_document_id = label.get("document_id")
            if not isinstance(relevant_document_id, str) or not relevant_document_id.strip():
                raise DatasetError(
                    f"{label_location}.document_id must be a non-empty string."
                )
            if relevant_document_id not in document_ids:
                raise DatasetError(
                    f"{label_location} refers to unknown document_id "
                    f"'{relevant_document_id}'."
                )
            if relevant_document_id in judged_document_ids:
                raise DatasetError(
                    f"{location}.relevance contains duplicate document_id "
                    f"'{relevant_document_id}'."
                )
            judged_document_ids.add(relevant_document_id)

            grade = label.get("grade")
            if not isinstance(grade, int) or isinstance(grade, bool) or not 1 <= grade <= 3:
                raise DatasetError(
                    f"{label_location}.grade must be an integer from 1 to 3."
                )

    return data
