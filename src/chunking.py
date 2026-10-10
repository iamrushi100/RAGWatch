"""Split source documents into stable, overlapping character chunks."""

from typing import Any


DEFAULT_CHUNK_SIZE = 72
DEFAULT_CHUNK_OVERLAP = 24


def _validate_settings(chunk_size: int, overlap: int) -> None:
    """Check chunk settings and reject booleans as accidental integers."""
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer.")
    if not isinstance(overlap, int) or isinstance(overlap, bool) or overlap < 0:
        raise ValueError("overlap must be a non-negative integer.")
    if overlap >= chunk_size:
        raise ValueError("overlap must be less than chunk_size.")


def chunk_document(
    document_id: str,
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    """Return fixed-size character chunks with IDs and source locations.

    IDs use ``<document_id>::chunk-<zero-based-index>`` with a four-digit
    minimum width, such as ``doc_notes::chunk-0001``. The index follows the
    deterministic left-to-right chunk sequence, so the same source ID and
    settings reproduce the same IDs. Each result retains the source ID and
    half-open character offsets for tracing a chunk back to its source text.

    Empty or whitespace-only text returns no chunks. Short text returns one
    chunk, and no emitted chunk contains only whitespace.
    """
    _validate_settings(chunk_size, overlap)
    if not isinstance(document_id, str) or not document_id.strip():
        raise ValueError("document_id must be a non-empty string.")
    if not isinstance(text, str):
        raise ValueError("text must be a string.")
    if not text.strip():
        return []

    chunks = []
    start = 0
    chunk_index = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end]
        if chunk_text.strip():
            chunks.append(
                {
                    "chunk_id": f"{document_id}::chunk-{chunk_index:04d}",
                    "source_document_id": document_id,
                    "chunk_index": chunk_index,
                    "start_char": start,
                    "end_char": end,
                    "text": chunk_text,
                }
            )

        if end == len(text):
            break
        start = end - overlap
        chunk_index += 1

    return chunks
