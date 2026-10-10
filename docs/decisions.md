# RAGWatch Project Decisions

These decisions follow the repository's project plan. They establish
the initial direction; optional extensions remain conditional on the
core working and being tested.

## Standalone repository

- **Decision:** Keep RAGWatch in its own repository, with no integration
  into Project 1 or other projects.
- **Reason:** The project should be independently understandable,
  reproducible, and easy to demonstrate.
- **Alternative considered:** Reuse or connect another portfolio
  project's code.
- **Trade-off:** Some functionality may be duplicated, but setup and
  project boundaries remain clear.
- **Milestone:** Initial project blueprint.

## Local-first development

- **Decision:** Make a reproducible local workflow the primary
  development and acceptance path.
- **Reason:** Local development avoids making the core depend on a
  hosted service, paid API, or changing free-tier offer.
- **Alternative considered:** Build around a hosted service or deploy
  early.
- **Trade-off:** A local demo may be less convenient to share, and
  optional local models may need more compute.
- **Milestone:** Initial project blueprint.

## Deterministic retrieval metrics before LLM judging

- **Decision:** Implement and test Recall@k, MRR, and nDCG@k as the
  core evaluation before considering LLM answer judging.
- **Reason:** Retrieval metrics can be calculated repeatably from
  rankings and labels without model calls or API costs.
- **Alternative considered:** Start by judging generated answers with
  an LLM.
- **Trade-off:** Retrieval scores do not measure answer correctness or
  groundedness. LLM judging can be considered later, but must be
  optional, capped, cached where applicable, and described as an
  estimate.
- **Milestone:** Core evaluation phase.

## TF-IDF initial retrieval baseline

- **Decision:** Start with a simple TF-IDF lexical retriever.
- **Reason:** It provides a small, local baseline that is easier to
  inspect and evaluate before adding model downloads or more retrieval
  components.
- **Alternative considered:** Begin with sentence embeddings, a vector
  index, BM25, or hybrid retrieval.
- **Trade-off:** Lexical matching can miss relevant passages when the
  query and passage use different words. Other methods may be explored
  after baseline behavior and metric tests pass.
- **Milestone:** Retrieval baseline phase.

## SQLite run history

- **Decision:** Use SQLite to store evaluation run history.
- **Reason:** It can keep local run records without a separate database
  service.
- **Alternative considered:** Use files alone or operate a server
  database.
- **Trade-off:** SQLite is a good fit for this small local project but
  is not intended here as a multi-user, large-scale storage system.
- **Milestone:** Run history phase.

## Streamlit dashboard

- **Decision:** Use Streamlit for the dashboard.
- **Reason:** It supports a straightforward Python interface for
  presenting metrics, comparisons, history, and failure examples.
- **Alternative considered:** Build a custom frontend.
- **Trade-off:** Streamlit limits visual and interaction flexibility
  compared with a custom frontend, while keeping the initial dashboard
  smaller to build.
- **Milestone:** Dashboard phase, after core evaluation and history.

## FastAPI after core evaluation works

- **Decision:** Consider FastAPI only after the core evaluation
  functions work independently; the API remains conditional on time.
- **Reason:** The evaluation engine should be testable without a web
  layer, and the project plan prioritizes core behavior first.
- **Alternative considered:** Build API endpoints alongside the first
  retrieval and metric code.
- **Trade-off:** An API may be deferred or omitted if the core needs
  more work; this keeps the acceptance path focused.
- **Milestone:** Later interface phase.

## Deployment after local tests pass

- **Decision:** Attempt deployment only after local setup is
  reproducible and relevant tests pass.
- **Reason:** A reliable local project is the priority, and hosting
  availability and limits can change.
- **Alternative considered:** Make a hosted demo an early requirement.
- **Trade-off:** The project may finish without a public deployment if
  hosting is unsuitable, but remains demonstrable locally.
- **Milestone:** Later release phase.

## Small, focused implementation tasks

- **Decision:** Build the project incrementally, implementing and
  verifying one focused task at a time.
- **Reason:** This keeps the work understandable, limits unfinished
  features, and supports learning and review.
- **Alternative considered:** Generate the full planned application in
  one large step.
- **Trade-off:** Progress is staged and some optional features may not
  be reached, while each completed part is easier to inspect and
  explain.
- **Milestone:** Applies throughout development.
