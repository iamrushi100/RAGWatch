# Tiny Sample Evaluation Dataset

The hand-verifiable fixture is stored in [`data/sample_dataset.json`](../data/sample_dataset.json). It contains seven short fictional source documents and six questions. The examples use familiar facts so a person can check each relevance label by reading the text.

## JSON schema

The top-level object has three fields:

- `dataset_id`: stable name for this version of the sample dataset.
- `schema_version`: integer identifying the shape of the JSON records. Increase it if the data structure changes incompatibly.
- `documents`: the source passages available to the retriever.
- `queries`: the questions and their manually assigned relevance labels.

Each object in `documents` has:

- `document_id`: stable, unique identifier for the source passage. Queries refer to documents by this ID.
- `text`: the short fictional passage to retrieve.

Each object in `queries` has:

- `question_id`: stable, unique identifier for this evaluation question.
- `question`: the query text given to a retriever.
- `relevance`: a list of documents judged relevant to the question. Each list entry contains:
  - `document_id`: an ID that must exist in the top-level `documents` list.
  - `grade`: an integer relevance grade from 1 to 3. Grade 3 means directly answers the question; lower positive grades mean useful but less complete evidence. This fixture uses grade 1 for a partially related passage. A document omitted from a question's exhaustive relevance list is treated as non-relevant for that question.

The question and document IDs are unique within their respective lists. The relevance list must not repeat a document ID. A valid retriever ranking for a question is an ordered list of document IDs; metric code can translate each ranked ID to its grade, using grade 0 for documents absent from that question's relevance list.

## How it supports retrieval metrics

- **Recall@k:** For each question, count relevant documents (grade greater than 0) appearing in the first `k` ranked results and divide by the total number of relevant documents for that question. The lighthouse question has two relevant documents, so it can demonstrate partial recall when only one is retrieved in the top results.
- **MRR (mean reciprocal rank):** Find the rank of the first retrieved document with grade greater than 0. Its reciprocal rank is `1 / rank`; if no relevant document is retrieved, the contribution is 0. Graded relevance distinguishes strength for nDCG, while MRR treats any positive grade as relevant.
- **nDCG@k:** Use graded relevance to reward stronger evidence near the top. For this fixture, gain is `2^grade - 1`, and rank discount is `log2(rank + 1)`. DCG is the sum of discounted gains in the top `k`; IDCG is the DCG for the same relevant documents ordered from highest grade to lowest. nDCG is `DCG / IDCG` when IDCG is nonzero.

## Worked example

For `q_lighthouse_color_and_timing`, `doc_lighthouse_signal` has grade 3 because it gives the requested color and timing. `doc_lighthouse_location` has grade 1 because it gives useful lighthouse context (where it stands) but does not answer either requested detail. Suppose a retriever returns this ranking:

1. `doc_lighthouse_location` (grade 1)
2. `doc_lighthouse_signal` (grade 3)
3. `doc_cafe_hours` (not relevant; grade 0 for this question)

Using `k = 2`:

- **Recall@2:** both of the two relevant documents appear in the first two results, so Recall@2 is `2 / 2 = 1.0`. At `k = 1`, only one of the two appears, so Recall@1 is `1 / 2 = 0.5`.
- **Reciprocal rank:** the first relevant result is at rank 1, so this question contributes `1 / 1 = 1.0` to MRR. MRR across the dataset would average the per-question reciprocal ranks; it is not calculated here.
- **nDCG@2:** grade 1 has gain `2^1 - 1 = 1`, and grade 3 has gain `2^3 - 1 = 7`. DCG is `(1 / log2(1 + 1)) + (7 / log2(2 + 1)) = 1 + 4.4165 = 5.4165`. The ideal order puts grade 3 first and grade 1 second, so IDCG is `(7 / log2(1 + 1)) + (1 / log2(2 + 1)) = 7 + 0.6309 = 7.6309`. Therefore nDCG@2 is `5.4165 / 7.6309`, approximately `0.710`.

These values are hand-calculated from the documented formulas for the illustrative ranking above, not the result of running RAGWatch or a benchmark.

## Limitations and intended use

This is a tiny test fixture for learning, manual inspection, and later metric tests. The documents and questions are fictional, short, and unusually clear. Six questions cannot represent real users, varied writing styles, ambiguous information needs, or a broad domain. Labels are judgments made for this example and should be reviewed if the passages or questions change. The list assumes all relevant documents for each question are known; in a real corpus, an unlisted passage might be relevant but not yet judged.

The fixture does not establish retriever quality, production performance, or drift detection ability. It contains no generated answers and cannot measure answer correctness or groundedness. It is not a public benchmark, and no metric run or benchmark is claimed here.
