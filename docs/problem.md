# RAGWatch Problem and Project Goal

## The problem

A retrieval-augmented generation (RAG) application can keep running even
after it starts finding less useful source passages. A demo may still
look convincing, but changes to documents, chunking, or retrieval
settings can make the evidence less relevant. Without repeatable checks,
it is difficult to tell whether quality changed or which questions
explain the change.

RAGWatch is intended to help answer: **Did this RAG setup get worse, how
do we know, and which examples explain the change?** It evaluates a small
RAG retriever using known questions and relevant passages, records runs,
and supports comparison against a saved baseline. Where a measured score
crosses a documented threshold, it can report a regression alert.

## Intended user and goal

The intended user is a learner or developer who wants to understand and
demonstrate how to evaluate retrieval quality. The project is designed
for local use and as a portfolio learning project; it is not presented
as a commercial product or a production monitoring service.

**Project goal:** Build a beginner-friendly, standalone tool that
produces repeatable evidence about retrieval quality, tracks evaluation
runs, compares configurations, and explains failures in terms of
individual examples.

## MVP boundaries

The planned MVP centers on a small, documented document collection and
a manually checked evaluation set. It will include a simple retriever,
deterministic retrieval metrics (Recall@k, MRR, and nDCG@k), repeatable
evaluation runs, per-question failure analysis, SQLite run history,
baseline comparisons, and an explicit threshold alert. A basic
Streamlit dashboard will present runs and their evidence. Tests and a
small deterministic CI check are part of the planned portfolio-ready
release.

The initial retriever baseline is TF-IDF. More retrieval methods may be
considered after the baseline and metric tests work. FastAPI is a later,
conditional interface, and will be considered only after the core
evaluation functions work independently. LLM-based answer generation
and judging are optional extensions, not requirements for the core
evaluation engine; any judging must be capped and treated as an estimate.

## Limitations

- A small evaluation set may not represent real users or all kinds of
  questions, and its scores can be noisy.
- Metrics are only as trustworthy as the relevance labels. Labels need
  manual checking and should be versioned when changed.
- Controlled regression or drift simulations show behavior under the
  changes tested; they do not prove that RAGWatch detects every kind of
  real-world production drift.
- TF-IDF is a simple lexical baseline. It may miss relevant passages
  that use different wording, and it does not establish how an embedding
  retriever would perform.
- Retrieval metrics measure ranked evidence, not whether a generated
  answer is correct, useful, or grounded. Optional LLM judging is
  imperfect and is not ground truth.
- The project has no production traffic or large-scale validation, and
  it makes no production reliability claims.
- The core is intended not to require a paid API. Optional models or
  hosted services may have changing costs, terms, or local hardware
  requirements; they are outside the required local baseline.
- Deployment is a later attempt after local setup and tests are
  reliable, and may depend on hosting availability and limits.

No benchmark results or quality claims are made here. Results should be
documented only after the corresponding evaluation has actually run.
