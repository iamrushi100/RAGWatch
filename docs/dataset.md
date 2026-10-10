# Tiny Sample Evaluation Dataset

The hand-verifiable fixture is stored in [`data/sample_dataset.json`](../data/sample_dataset.json). Version 2 keeps seven fictional source documents and six questions. The loader derives twelve retrieval chunks from the source text using the documented defaults: 72 characters per chunk and 24 characters of overlap.

## JSON schema

The top-level object contains:

- `dataset_id`: stable name for this dataset version. It changed to `ragwatch_tiny_sample_v2` because relevance labels now refer to chunks.
- `schema_version`: the dataset shape version. Version 2 uses chunk-level relevance IDs.
- `documents`: original source passages. Each has a unique `document_id` and non-empty `text`; the original text and IDs are preserved.
- `queries`: questions and manually assigned relevance labels. Each has a unique `question_id`, question text, and a `relevance` list.

Each relevance entry has:

- `chunk_id`: the ID of a generated chunk that contains relevant evidence.
- `grade`: an integer from 1 through 3. Grade 3 means direct evidence; lower positive grades mean useful but less complete evidence. A chunk omitted from the exhaustive relevance list is treated as non-relevant.

The loader adds a `chunks` list to its returned in-memory dataset; chunks are derived and are not duplicated in the JSON file. Every chunk contains `chunk_id`, `source_document_id`, zero-based `chunk_index`, `start_char`, `end_char`, and the exact source-text slice. This preserves traceability without copying generated chunk records into the fixture.

## Chunking and IDs

`src.chunking.chunk_document` slices characters from left to right. The defaults are `chunk_size=72` and `overlap=24`; callers may choose other valid settings when creating chunks. Chunk size must be positive and overlap must satisfy `0 <= overlap < chunk_size`. Empty or whitespace-only text produces no chunks; short text produces one chunk.

An ID is formed as `<source_document_id>::chunk-<zero-based-index>`, padded to at least four digits. For example, the library source starts with `doc_library_returns::chunk-0000`. Its next chunk starts 48 characters into the text: the previous 72-character slice overlaps the next slice by 24 characters. Each chunk also includes `source_document_id` so a ranked chunk can be traced to its original passage.

The IDs are reproducible for the same source ID, text, and settings. If the text or chunk settings change, review the relevance labels and increment the dataset version/ID as appropriate. IDs identify chunk positions, not immutable content hashes.

Schema version 1 used `document_id` in relevance labels. The loader can normalize a legacy label in memory when that source produces exactly one chunk. If the source splits into multiple chunks, it raises an error asking for a specific `chunk_id`; it does not guess which chunk is relevant or rewrite the JSON file. Historical SQLite rows and saved JSON reports are not rewritten. Older failure reports may still contain document-level IDs, while newly evaluated reports contain chunk-level IDs.

## How it supports retrieval metrics

- **Recall@k:** counts relevant chunks among the first `k` ranked chunks and divides by the total number of relevant chunks for that question. The lighthouse question has two relevant chunks, so retrieving only one gives partial recall.
- **MRR:** finds the rank of the first retrieved chunk with a positive relevance grade and returns `1 / rank`; if none is retrieved, the contribution is zero.
- **nDCG@k:** uses grades to reward stronger evidence near the top. Gain is `2^grade - 1`, and the rank discount is `log2(rank + 1)`. It divides the discounted gain in the top `k` by the ideal discounted gain.

For `q_lighthouse_color_and_timing`, `doc_lighthouse_signal::chunk-0000` has grade 3 because it states the light's color and timing. `doc_lighthouse_location::chunk-0000` has grade 1 because it supplies useful context but not either requested detail. Suppose the ranking is:

1. `doc_lighthouse_location::chunk-0000` (grade 1)
2. `doc_lighthouse_signal::chunk-0000` (grade 3)
3. `doc_cafe_hours::chunk-0000` (grade 0)

At `k=2`, Recall@2 is `2 / 2 = 1.0`; at `k=1`, it is `1 / 2 = 0.5`. The first relevant chunk is at rank 1, so reciprocal rank is `1 / 1 = 1.0`.

For nDCG@2, the gains are `2^1 - 1 = 1` and `2^3 - 1 = 7`. Thus DCG is `(1 / log2(1 + 1)) + (7 / log2(2 + 1)) = 1 + 4.4165 = 5.4165`. The ideal order puts grade 3 first, so IDCG is `(7 / log2(1 + 1)) + (1 / log2(2 + 1)) = 7 + 0.6309 = 7.6309`. nDCG@2 is `5.4165 / 7.6309`, approximately `0.710`.

This is a hand-worked example, not a claim that the displayed ranking was produced by a run or benchmark.

## Limitations and intended use

This is a small test fixture for learning and manual inspection. It has only six fictional questions and seven short documents; several sources split into overlapping chunks. The examples do not represent real user language, ambiguous questions, or a broad domain. Relevance labels are judgments and should be reviewed whenever source text or chunk settings change.

Chunk-level labels make retrieval metrics sensitive to chunk boundaries. If relevant evidence straddles a boundary, overlap can preserve context, but labels still need manual verification. This fixture does not establish retriever quality, production performance, or drift-detection ability. Retrieval metrics do not measure generated-answer correctness or groundedness. No benchmark result is claimed here.
