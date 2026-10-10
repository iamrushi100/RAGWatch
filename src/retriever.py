"""A small TF-IDF retriever for validated RAGWatch datasets."""

from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def rank_documents(
    dataset: dict[str, Any], question: str
) -> list[tuple[str, float]]:
    """Return (document_id, similarity_score) pairs, best match first.

    The dataset must already have been validated by ``load_dataset`` in
    ``src.dataset``. Equal scores are ordered by document ID alphabetically,
    so repeated calls produce a predictable ranking.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string.")

    documents = dataset["documents"]
    if not documents:
        raise ValueError("Cannot retrieve from a dataset with no documents.")

    # Build one text per source and keep IDs in the same order as the rows.
    document_ids = [document["document_id"] for document in documents]
    document_texts = [document["text"] for document in documents]

    # TF-IDF turns words into weighted numeric features. Cosine similarity
    # compares the question vector with each document vector.
    vectorizer = TfidfVectorizer()
    document_vectors = vectorizer.fit_transform(document_texts)
    question_vector = vectorizer.transform([question])
    scores = cosine_similarity(question_vector, document_vectors)[0]

    # Sorting paired records keeps every score attached to its source ID.
    results = list(zip(document_ids, scores.tolist()))
    results.sort(key=lambda result: (-result[1], result[0]))
    return results
