"""Load and validate a RAGWatch evaluation dataset."""

import json
from pathlib import Path
from typing import Any

from src.chunking import chunk_document


class DatasetError(ValueError):
    """Raised when a dataset file cannot be loaded or is invalid."""


def load_dataset(file_path: str | Path) -> dict[str, Any]:
    """Load a JSON dataset and validate its required fields and references.

    The returned dictionary includes the JSON dataset fields plus a derived
    ``chunks`` list. For schema version 1, a relevance label that uses a source
    ``document_id`` is normalized in memory to its ``chunk_id`` when that
    source produces exactly one chunk. If the source splits into multiple
    chunks, loading fails and requires an explicit chunk-level relevance label.
    The source JSON file is not rewritten, and this function does not evaluate
    retrieval.
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
    if data["schema_version"] not in (1, 2):
        raise DatasetError("'schema_version' must be a supported version: 1 or 2.")

    documents = data["documents"]
    if not isinstance(documents, list):
        raise DatasetError("'documents' must be a list.")

    document_ids: set[str] = set()
    chunks = []
    chunks_by_source: dict[str, list[dict[str, Any]]] = {}
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
        document_chunks = chunk_document(document_id, text)
        chunks.extend(document_chunks)
        chunks_by_source[document_id] = document_chunks

    chunk_ids = {chunk["chunk_id"] for chunk in chunks}

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

            has_chunk_id = "chunk_id" in label
            has_document_id = "document_id" in label
            if has_chunk_id == has_document_id:
                raise DatasetError(
                    f"{label_location} must contain exactly one of 'chunk_id' "
                    "or legacy 'document_id'."
                )

            if has_chunk_id:
                relevant_chunk_id = label.get("chunk_id")
            else:
                if data["schema_version"] != 1:
                    raise DatasetError(
                        f"{label_location} in schema version 2 must use 'chunk_id'."
                    )
                legacy_document_id = label.get("document_id")
                if (
                    not isinstance(legacy_document_id, str)
                    or not legacy_document_id.strip()
                ):
                    raise DatasetError(
                        f"{label_location}.document_id must be a non-empty string."
                    )
                if legacy_document_id not in document_ids:
                    raise DatasetError(
                        f"{label_location} refers to unknown document_id "
                        f"'{legacy_document_id}'."
                    )
                source_chunks = chunks_by_source[legacy_document_id]
                if len(source_chunks) != 1:
                    raise DatasetError(
                        f"{label_location} uses legacy source-level document_id "
                        f"'{legacy_document_id}', but that source creates "
                        f"{len(source_chunks)} chunks. Relabel it with the "
                        "specific relevant chunk_id."
                    )
                relevant_chunk_id = source_chunks[0]["chunk_id"]
                label["chunk_id"] = relevant_chunk_id
                del label["document_id"]

            if not isinstance(relevant_chunk_id, str) or not relevant_chunk_id.strip():
                raise DatasetError(
                    f"{label_location}.chunk_id must be a non-empty string."
                )
            if relevant_chunk_id not in chunk_ids:
                raise DatasetError(
                    f"{label_location} refers to unknown chunk_id "
                    f"'{relevant_chunk_id}'."
                )
            if relevant_chunk_id in judged_document_ids:
                raise DatasetError(
                    f"{location}.relevance contains duplicate chunk_id "
                    f"'{relevant_chunk_id}'."
                )
            judged_document_ids.add(relevant_chunk_id)

            grade = label.get("grade")
            if not isinstance(grade, int) or isinstance(grade, bool) or not 1 <= grade <= 3:
                raise DatasetError(
                    f"{label_location}.grade must be an integer from 1 to 3."
                )

    # Chunks are derived at load time; the source documents remain unchanged
    # in the on-disk fixture and in this returned object.
    data["chunks"] = chunks
    return data
