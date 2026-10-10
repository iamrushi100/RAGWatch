"""A small TF-IDF retriever for validated RAGWatch datasets."""

from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def rank_documents(
    dataset: dict[str, Any], question: str
) -> list[tuple[str, float]]:
    """Return (chunk_id, similarity_score) pairs, best match first.

    The dataset must already have been validated by ``load_dataset`` in
    ``src.dataset``. Each chunk retains its ``source_document_id`` in the
    dataset for traceability. Equal scores are ordered by chunk ID
    alphabetically, so repeated calls produce a predictable ranking.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string.")

    chunks = dataset["chunks"]
    if not chunks:
        raise ValueError("Cannot retrieve from a dataset with no text chunks.")

    # Build one text per chunk and keep chunk IDs in the same row order.
    chunk_ids = [chunk["chunk_id"] for chunk in chunks]
    chunk_texts = [chunk["text"] for chunk in chunks]

    # TF-IDF turns words into weighted numeric features. Cosine similarity
    # compares the question vector with each document vector.
    vectorizer = TfidfVectorizer()
    document_vectors = vectorizer.fit_transform(chunk_texts)
    question_vector = vectorizer.transform([question])
    scores = cosine_similarity(question_vector, document_vectors)[0]

    # Sorting paired records keeps every score attached to its source ID.
    results = list(zip(chunk_ids, scores.tolist()))
    results.sort(key=lambda result: (-result[1], result[0]))
    return results
