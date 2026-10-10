"""Focused tests for deterministic fixed-size character chunking."""

import pytest

from src.chunking import chunk_document


def test_short_text_produces_one_traceable_chunk():
    chunks = chunk_document("doc_short", "A short passage.", chunk_size=20, overlap=3)

    assert len(chunks) == 1
    assert chunks[0] == {
        "chunk_id": "doc_short::chunk-0000",
        "source_document_id": "doc_short",
        "chunk_index": 0,
        "start_char": 0,
        "end_char": 16,
        "text": "A short passage.",
    }


def test_exact_chunk_boundary_does_not_create_an_extra_empty_chunk():
    chunks = chunk_document("doc_exact", "abcdefghij", chunk_size=5, overlap=0)

    assert [chunk["text"] for chunk in chunks] == ["abcde", "fghij"]
    assert [chunk["chunk_id"] for chunk in chunks] == [
        "doc_exact::chunk-0000",
        "doc_exact::chunk-0001",
    ]


def test_chunks_have_configured_size_and_character_overlap():
    text = "abcdefghijklmno"

    chunks = chunk_document("doc_overlap", text, chunk_size=8, overlap=3)

    assert [chunk["text"] for chunk in chunks] == ["abcdefgh", "fghijklm", "klmno"]
    assert chunks[0]["text"][-3:] == chunks[1]["text"][:3]
    assert chunks[1]["text"][-3:] == chunks[2]["text"][:3]
    assert [(chunk["start_char"], chunk["end_char"]) for chunk in chunks] == [
        (0, 8),
        (5, 13),
        (10, 15),
    ]


def test_ids_are_stable_and_keep_source_document_traceability():
    first = chunk_document("doc_source", "0123456789", chunk_size=4, overlap=1)
    second = chunk_document("doc_source", "0123456789", chunk_size=4, overlap=1)

    assert [chunk["chunk_id"] for chunk in first] == [
        "doc_source::chunk-0000",
        "doc_source::chunk-0001",
        "doc_source::chunk-0002",
    ]
    assert first == second
    assert all(chunk["source_document_id"] == "doc_source" for chunk in first)


@pytest.mark.parametrize("text", ["", "   ", "\n\t"])
def test_empty_or_whitespace_text_produces_no_chunks(text):
    assert chunk_document("doc_empty", text) == []


@pytest.mark.parametrize(
    ("chunk_size", "overlap", "message"),
    [
        (0, 0, "chunk_size must be a positive integer"),
        (-1, 0, "chunk_size must be a positive integer"),
        (True, 0, "chunk_size must be a positive integer"),
        (5, -1, "overlap must be a non-negative integer"),
        (5, True, "overlap must be a non-negative integer"),
        (5, 5, "overlap must be less than chunk_size"),
    ],
)
def test_invalid_chunk_settings_are_rejected(chunk_size, overlap, message):
    with pytest.raises(ValueError, match=message):
        chunk_document("doc_invalid", "some text", chunk_size, overlap)
