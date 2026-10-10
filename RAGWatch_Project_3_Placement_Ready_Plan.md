# RAGWatch --- Project 3: RAG Evaluation & Quality Monitor

## Placement-Ready Development Plan

> **One-line pitch:** A standalone tool that evaluates a small RAG
> system, compares retrieval configurations, tracks evaluation runs, and
> raises evidence-based alerts when measured quality regresses.
>
> **Project goal:** Finish a demonstrable, honestly measured portfolio
> project that can be explained in five minutes and defended in a
> technical interview.
>
> **Working constraints:** Beginner-friendly; Windows + VS Code +
> Python + Codex; 4+ hours/day available; use ChatGPT and Codex
> efficiently; preserve the remaining 60% of ChatGPT Go usage/credits;
> no dependency on Project 1 or Project 2; deploy only after the local
> application is reliable.

------------------------------------------------------------------------

## 1. Final decisions

  -----------------------------------------------------------------------
  Area                                Decision
  ----------------------------------- -----------------------------------
  Repository                          New standalone repository named
                                      `ragwatch`

  Scope                               A small RAG system under test
                                      **plus** an evaluation/monitoring
                                      layer

  Teaching style                      Learn one concept, implement one
                                      small task, test it, explain it
                                      back

  Core language                       Python

  Initial retrieval baseline          TF-IDF / lexical baseline, then add
                                      one local embedding model

  Vector index                        Choose one lightweight option only
                                      after the baseline works; do not
                                      build FAISS and Chroma versions

  Keyword retrieval                   Add BM25 only after the first
                                      baseline and metrics pass

  Reranking                           Optional experiment, not a release
                                      blocker

  Answer generation                   Put behind an interface; use a
                                      capped optional provider/local
                                      model only after deterministic
                                      retrieval evaluation works

  Answer judging                      Optional/capped LLM-as-judge phase,
                                      cached; manually review at least 20
                                      examples if enough judged examples
                                      are produced

  Storage                             SQLite

  API                                 FastAPI

  Dashboard                           Streamlit

  Tests                               Pytest

  CI                                  GitHub Actions runs tests and a
                                      small deterministic regression
                                      evaluation

  Deployment                          Later; first make local setup
                                      reproducible

  Integration                         No integration with Project 1 in
                                      the initial project

  Budget rule                         No paid API is required for the
                                      core evaluation engine; do not make
                                      broad or repeated LLM calls without
                                      explicit approval
  -----------------------------------------------------------------------

## 2. Why this plan is tightened

The proposed 35-day plan has a strong direction, but it puts too many
expensive or parallel tasks into the must-have path: multiple embedding
models, BM25 and hybrid search, reranking, LLM answer generation, LLM
judging, drift tests, API, dashboard, Docker, CI, and deployment. For a
beginner working with limited AI-assistant usage, this creates a real
risk of building many partial features.

The adjusted approach preserves the project's strongest placement
evidence while making the first working version cheaper and more
finishable:

1.  **Build a tiny RAG baseline before adding libraries and model
    variants.**
2.  **Implement Recall@k, MRR, and nDCG@k by hand**, with tiny tests
    whose answers are known.
3.  **Keep deterministic retrieval evaluation as the core acceptance
    gate.** LLM answer judging is an additional layer because it can
    cost money and is itself imperfect.
4.  **Run one experiment at a time.** Start with chunking and
    lexical-vs-vector retrieval; add more embedding models or reranking
    only if the core is complete.
5.  **Treat drift as controlled simulation, not proof of production
    drift detection.**
6.  **Use one small CI evaluation on each push.** Keep a larger
    benchmark as a manual command.
7.  **Deploy after local reliability.** A live demo is useful, but it
    must not jeopardize completion of the core project.
8.  **Use ChatGPT for planning, concept teaching, and high-value
    reviews; Codex for one focused implementation task at a time.** Do
    not ask both tools to independently regenerate or review the whole
    project.

### Naming recommendation

Use **RAGWatch --- RAG Evaluation & Quality Monitor** as the
repository/product name. In the README, describe it as a *standalone RAG
evaluation and quality-monitoring learning project with controlled drift
simulations*. Avoid claiming real-world production drift detection
because the project will not have production traffic.

------------------------------------------------------------------------

## 3. Problem statement

A RAG system can look good in a demo but retrieve less useful evidence
after documents, chunking, retrieval settings, or embeddings change. The
application may still run while quality declines.

RAGWatch asks:

> **"Did this RAG configuration get worse, how do we know, and which
> examples explain the change?"**

It includes a small RAG system under test and an independent evaluator.
It records run configuration, dataset version, retrieved results,
metrics, and failures; compares a current run with a saved baseline; and
creates an alert when a documented threshold is crossed.

This is a learning and portfolio project---not a claim of production use
or a commercial product.

------------------------------------------------------------------------

## 4. MVP scope

### Required for the first portfolio-ready release

-   [ ] A small, documented collection of public or self-authored sample
    documents.
-   [ ] Fixed-size chunking with overlap and tests.
-   [ ] A simple retriever that returns ranked chunk IDs and scores.
-   [ ] A small labeled evaluation dataset with questions and relevant
    passage IDs.
-   [ ] Hand-built Recall@k, MRR, and nDCG@k with unit tests.
-   [ ] Repeatable evaluation runs with configuration and dataset
    version recorded.
-   [ ] Failure analysis showing the worst cases and the retrieved
    evidence.
-   [ ] SQLite run history and baseline-vs-current comparison.
-   [ ] A configurable threshold alert with a deterministic test that
    proves it fires.
-   [ ] A basic Streamlit dashboard for runs, metrics, comparisons, and
    failures.
-   [ ] A small FastAPI interface for running an evaluation and
    retrieving history, if time permits after the core CLI is complete.
-   [ ] Pytest suite and GitHub Actions regression gate.
-   [ ] README, architecture diagram, experiment results, limitations,
    and interview guide.
-   [ ] A reproducible local startup path and a later deployment
    attempt.

### Optional upgrades, only after the MVP works

-   Local sentence-transformer embeddings.
-   BM25 and hybrid retrieval comparison.
-   A second embedding model.
-   A cross-encoder reranker.
-   Answer generation with citations.
-   Groundedness and answer-relevance judging through a configurable LLM
    adapter.
-   Manual judge review and a judge-error report.
-   Cost-per-query tracking.
-   Bootstrap confidence intervals.
-   Webhook notifications.
-   Public hosted demo.

### Explicitly out of scope

-   Integration with Project 1 or any other portfolio project.
-   Kubernetes, Prometheus, or Grafana.
-   React or a custom frontend.
-   Fine-tuning models.
-   Authentication, multi-user support, or multi-tenant architecture.
-   Large-scale production claims.
-   Paid APIs as a hard requirement.
-   Running many embedding models or a large LLM judge sweep before the
    MVP is complete.

------------------------------------------------------------------------

## 5. Architecture

``` text
Sample Documents
      |
      v
Document Loader -> Text Cleaning -> Chunker
      |                              |
      |                              v
      |                       Chunk IDs + Metadata
      |                              |
      v                              v
                 Retriever (system under test)
                           |
Evaluation Dataset --------+
(question, relevant IDs,
 optional reference answer)
                           |
                           v
                   Evaluation Runner
                    /             \
                   v               v
          Retrieval Metrics   Optional Answer Evaluator
          Recall@k, MRR,      groundedness/relevance
          nDCG@k              (capped and cached)
                    \             /
                     v           v
                      Quality Report
                     /      |       \
                    v       v        v
              Failure    SQLite    Run comparison
              analysis   history       |
                                       v
                               Threshold / alert
                                       |
                           +-----------+-----------+
                           |                       |
                           v                       v
                    Streamlit UI             FastAPI
                           |
                           v
                GitHub Actions small regression gate
```

### Component responsibilities

-   **Document loader:** loads the project's sample documents and stable
    document IDs.
-   **Chunker:** splits text using a documented strategy and keeps
    source metadata.
-   **Retriever:** returns ranked chunk IDs for a query.
-   **Evaluation dataset:** stores questions and the relevant
    chunk/document IDs established by inspection.
-   **Evaluation runner:** executes the same cases against a specific
    configuration.
-   **Metrics module:** calculates deterministic retrieval metrics
    without relying on an evaluation library.
-   **Optional answer evaluator:** scores answer properties only when an
    answer generator and judging method are configured.
-   **Failure analysis:** shows low-scoring questions and whether
    relevant evidence appeared in the top-k results.
-   **SQLite store:** records run ID, timestamp, configuration, dataset
    version, metrics, and alert outcomes.
-   **Comparison/alert engine:** compares a run with a chosen baseline
    and applies explicit thresholds.
-   **Dashboard/API:** presents stored evidence and starts or retrieves
    runs.
-   **CI regression gate:** runs a small deterministic evaluation and
    fails if the agreed regression condition is violated.

**Important:** The evaluator should not silently change the dataset,
configuration, or threshold while comparing runs. A comparison is
meaningful only when those details are recorded.

------------------------------------------------------------------------

## 6. Dataset strategy

Dataset quality is one of the most important risks. A metric is only
meaningful if relevance labels are defensible.

### Recommended staged approach

**Stage A --- small hand-checked starter set**

-   Start with approximately 15--25 questions and a small document
    collection.
-   Include easy, medium, and harder questions.
-   Include a few questions whose answers are not in the documents, if
    answer-generation behavior is implemented.
-   Assign stable IDs to documents and chunks.
-   For each question, record the relevant source passage IDs.
-   Manually verify every label before using the set as a baseline.

**Stage B --- expand after the pipeline works**

-   Grow toward 50--100 questions if time permits.
-   Freeze a small golden set for regression tests.
-   Version the dataset and document every label change.
-   Keep retrieval-only benchmarks separate from answer-quality test
    sets when their labels or purpose differ.

### Public benchmark option

BEIR SciFact is a possible retrieval benchmark. Its Hugging Face dataset
card describes scientific-claim retrieval and lists a **CC BY-SA 4.0**
license. Review the current dataset card, attribution requirements, and
suitability before redistributing any subset. Do not assume every
external dataset has the same license.

Sources: - https://huggingface.co/datasets/BeIR/scifact -
https://openreview.net/forum?id=wCu6T5xFjeJ

### Dataset record (illustrative schema)

``` json
{
  "question_id": "q001",
  "question": "What does the sample document say about ...?",
  "relevant_passage_ids": ["doc01_chunk02"],
  "reference_answer": "A manually verified short answer.",
  "difficulty": "easy",
  "dataset_version": "v1"
}
```

The example is a schema illustration, not a real labeled example. Actual
labels must be verified against the chosen documents.

------------------------------------------------------------------------

## 7. Metrics and definitions

### Retrieval metrics --- implement by hand

-   **Recall@k:** number of relevant retrieved items in the top k
    divided by the total number of known relevant items for the query.
-   **MRR:** average reciprocal rank of the first relevant result across
    queries. A first relevant result at rank 1 contributes 1.0; rank 2
    contributes 0.5; rank 5 contributes 0.2. A query with no relevant
    result in the evaluated ranking contributes 0.
-   **nDCG@k:** ranking metric that rewards relevant results appearing
    higher. Use a documented relevance-graded formula; for binary
    labels, relevant items have gain 1 and irrelevant items gain 0.

Also record: - number of evaluated questions; - number of queries with
no relevant retrieved result; - per-question metric results; -
runtime/latency when measured consistently.

### Answer metrics --- optional layer

-   **Groundedness / faithfulness:** whether answer claims are supported
    by the retrieved context.
-   **Answer relevance:** whether the answer addresses the question.
-   **Correctness:** whether the answer matches the reference answer,
    when a reliable reference exists.

These are not interchangeable. A relevant answer can still contain
unsupported claims. LLM-as-judge outputs are estimates, not ground
truth. Cache judgments, cap the number of calls, manually inspect a
sample, and record judge/model/prompt versions.

### Alerting

Start with a simple configurable regression rule. For example, alert if
Recall@k falls by more than a configured absolute amount compared with
the selected baseline. Document whether the threshold is an absolute
score difference or a relative percentage.

A small evaluation set can produce noisy changes. Show the number of
questions behind a score and document that limitation. Bootstrap
confidence intervals are optional, not a blocker.

------------------------------------------------------------------------

## 8. Controlled drift simulations

Use "simulated drift" or "controlled regression" language in the README.
These experiments demonstrate behavior under known changes; they do not
prove the system can detect every real-world drift event.

1.  **Corpus growth:** add similar or distracting documents and measure
    whether retrieval metrics change.
2.  **Document change:** edit a source document and check whether
    expected evidence and labels remain valid.
3.  **Embedding/index mismatch:** demonstrate the error condition safely
    in a test harness. Do not silently compare vectors produced by
    incompatible embedding spaces; the application should ideally detect
    incompatible model/index metadata and reject the configuration.
4.  **Chunking/configuration change:** change chunk size or overlap,
    rebuild the index, and compare results.

For every simulation: - record the starting configuration; - change one
factor at a time; - record dataset and index versions; - save run
outputs; - explain whether the result was expected; - include the actual
measured numbers only after the experiment has run.

Do not promise that every simulation must lower quality. Some changes
may improve it or have no measurable effect. The test should prove that
the system records and reports what actually happened.

------------------------------------------------------------------------

## 9. Technology choices

  -----------------------------------------------------------------------
  Concern                 Initial choice          Reason / rule
  ----------------------- ----------------------- -----------------------
  Language                Python                  Main learning and
                                                  implementation language

  Baseline retrieval      TF-IDF using            Simple, local,
                          scikit-learn            inexpensive baseline

  Embeddings              One small               Add only after baseline
                          sentence-transformer    tests pass
                          model, later            

  Vector index            Select one lightweight  Avoid learning two
                          library later           indexes at once

  Keyword search          `rank_bm25`, later      Useful comparison after
                                                  baseline

  Data processing         Standard library first; Avoid dependencies
                          Pandas when useful      without a clear need

  Storage                 SQLite                  No separate database
                                                  service

  API                     FastAPI                 Add after evaluation
                                                  functions work
                                                  independently

  Dashboard               Streamlit               Faster to build than a
                                                  custom frontend

  Charts                  Plotly, if useful       Interactive
                                                  experiment/history
                                                  charts

  Tests                   Pytest                  Unit, integration,
                                                  regression tests

  CI                      GitHub Actions          Repeatable test and
                                                  regression gate

  Packaging               Docker                  After local setup and
                                                  tests are stable

  Deployment              Try a suitable free     Availability, limits,
                          option later            and policies can change
  -----------------------------------------------------------------------

### Model and API rule

Start without any paid API dependency. If answer generation or LLM
judging is added, use a small capped evaluation set, cache outputs, make
the provider configurable, and retain a deterministic test path that
does not call an external model.

Do not assume a "free tier" is permanent or unlimited. Verify the
provider's current pricing and terms before enabling it. A local model
avoids API billing but may require substantial RAM/compute and may be
slower on a CPU.

------------------------------------------------------------------------

## 10. Five-week execution plan

Five weeks is a target, not a guarantee. Adjust after Week 1 based on
actual progress. With 4+ hours/day, reserve a separate daily slot for
DSA, SQL, and core CS revision rather than using every hour on the
project.

### Week 1 --- Scope, baseline RAG, and dataset

**Tasks** - Create the repository, virtual environment, `AGENTS.md`,
README, progress notes, and decision log. - Write the problem statement
and architecture. - Create a small sample document collection. -
Implement fixed-size chunking with overlap. - Implement a simple
retriever and inspect its ranked results. - Create and manually verify
the first labeled evaluation cases.

**Learn** - Git, modules, functions, data structures, file handling,
chunking, retrieval, labels.

**Done when** - A local script can retrieve ranked source chunks for
sample questions. - Dataset validation catches missing IDs and malformed
records. - Tests cover chunking and basic retrieval behavior. - The
project blueprint is committed.

**Cut rule** - Do not add LLM judging, multiple embeddings, reranking,
API, or dashboard during this week.

### Week 2 --- Metrics and repeatable evaluation

**Tasks** - Implement Recall@k, MRR, and nDCG@k by hand. - Write tests
with tiny examples and exact expected values. - Build an evaluation
runner. - Save per-question outputs and aggregate scores. - Add
dataset/configuration version fields.

**Learn** - Ranking metrics, Python testing, data validation,
reproducible experiments.

**Done when** - Metric tests pass, including empty rankings and no
relevant results. - A complete evaluation can be repeated on the fixed
dataset. - Every aggregate score can be traced to per-question results.

### Week 3 --- Run history, comparisons, and failure analysis

**Tasks** - Store run metadata and metrics in SQLite. - Compare baseline
and current runs. - Implement a configurable threshold alert. - Create a
failure-analysis report. - Add controlled regression tests.

**Learn** - SQL basics, persistence, baseline comparisons, thresholds,
debugging.

**Done when** - At least two runs can be stored and compared. - A
deliberately degraded fixture triggers the alert. - A non-degraded
fixture does not trigger the same alert. - The report identifies the
worst examples and relevant evidence.

### Week 4 --- Dashboard, API, and optional answer evaluation

**Tasks** - Build the basic Streamlit dashboard from stored run
results. - Add FastAPI endpoints only after core functions are stable. -
If time and cost permit, add one answer-generation path and a
capped/cached judge. - Manually review at least 20 judged examples if
enough examples are generated. - Record judge limitations and
disagreements.

**Learn** - REST APIs, schemas, dashboard design, groundedness, model
evaluation limits.

**Done when** - A user can inspect run history, metrics, comparisons,
alerts, and failure examples. - API tests pass if the API is included. -
Answer judging is clearly labeled as approximate and optional.

**Cut rule** - If behind schedule, finish the dashboard and defer LLM
judging rather than compromising metric correctness and CI.

### Week 5 --- CI, Docker, documentation, deployment, interview preparation

**Tasks** - Add GitHub Actions for unit tests and a small deterministic
regression evaluation. - Add Docker only after the local run path is
stable. - Reproduce setup from a clean environment. - Finalize README,
architecture, dataset attribution, experiment table, failure analysis,
limitations, and case study. - Deploy if the free hosting option
supports the app and its resource requirements. - Prepare 30-second and
five-minute explanations.

**Learn** - CI/CD basics, reproducibility, packaging, deployment,
technical communication.

**Done when** - A clean clone can run the documented local workflow. -
CI passes on the baseline and fails on a deliberately failing regression
fixture. - Results in the README are generated by committed evaluation
commands. - You can explain metric definitions, architecture, one
experiment, one failure, and project limitations.

------------------------------------------------------------------------

## 11. Repository structure

Create folders gradually. Do not generate all files at once.

``` text
ragwatch/
├── AGENTS.md
├── README.md
├── requirements.txt             # or pyproject.toml; choose one packaging approach
├── .gitignore
├── .env.example                 # only if configuration requires it
├── docs/
│   ├── problem.md
│   ├── requirements.md
│   ├── architecture.md
│   ├── decisions.md
│   ├── metrics.md
│   ├── experiments.md
│   ├── failure_analysis.md
│   ├── case_study.md
│   ├── interview-guide.md
│   └── progress.md
├── data/
│   ├── documents/
│   └── eval/
├── src/
│   └── ragwatch/
│       ├── rag/
│       ├── evaluation/
│       ├── monitoring/
│       ├── storage/
│       └── api/
├── dashboard/
│   └── app.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── regression/
├── scripts/
├── reports/
└── .github/
    └── workflows/
        └── tests.yml
```

Pick either `requirements.txt` or `pyproject.toml` as the primary
dependency/configuration mechanism; avoid maintaining duplicate
dependency lists without a reason.

------------------------------------------------------------------------

## 12. ChatGPT + Codex low-usage workflow

The user has approximately 60% of their ChatGPT Go usage/credits
remaining. Exact quota mechanics can vary by feature and plan, so this
plan does not assume a fixed number of messages or tokens. The operating
principle is to reserve usage for high-value tasks.

### Tool responsibilities

**ChatGPT** - Explain concepts before implementation. - Help define
acceptance criteria. - Produce one focused Codex prompt per milestone. -
Review confusing errors or high-risk design decisions. - Help prepare
interview explanations.

**Codex in VS Code** - Inspect the real workspace. - Implement the
specific task requested. - Add or run tests for that task. - Summarize
changed files and commands/results. - Update documentation when
explicitly included in the task.

**Developer** - Read the diff. - Run the command locally. - Ask for
clarification on code not understood. - Confirm results before updating
progress notes. - Commit verified work.

### Cost-saving rules

1.  **One task, one prompt, one verification cycle.** Avoid asking
    multiple agents to implement the same feature.
2.  Give Codex relevant file paths and a clear acceptance checklist; do
    not paste the entire project into every prompt.
3.  Ask Codex to inspect before editing and restrict it to named files.
4.  Do not ask for a full repository review after every tiny change.
    Review the diff and tests; request help only when something is
    unclear or high risk.
5.  Prefer local deterministic tests over repeated LLM calls.
6.  Cache any LLM-generated answers or judge outputs using a stable key
    based on input, model, and prompt/version.
7.  Use a tiny development dataset and a separate small CI dataset. Run
    the larger evaluation manually.
8.  Never send secrets, credentials, or private documents to an AI tool.
9.  Keep `AGENTS.md`, `docs/progress.md`, and `docs/decisions.md`
    concise and current.
10. Commit each verified milestone. A commit is a recovery point and
    reduces the need to regenerate working code.
11. Do not ask Codex to implement future phases while current tests
    fail.
12. Stop adding features when the remaining work threatens the five-week
    delivery goal.

### Standard Codex prompt

``` text
Project: RAGWatch, a standalone Python RAG evaluation project.
Read AGENTS.md and inspect only the files relevant to this task.

Task: [one small, specific task].

Acceptance criteria:
- [observable behavior 1]
- [observable behavior 2]
- [tests that must pass]

Constraints:
- Implement only this task.
- Keep code beginner-readable and typed where appropriate.
- Do not add dependencies unless necessary; explain any proposed dependency.
- Add or update focused tests.
- Do not fabricate results.
- Do not modify unrelated files.

When finished, report:
1. Files changed.
2. Design choices in simple English.
3. Exact test commands run and their actual results.
4. Known limitations.
5. One short explanation I should be able to give in an interview.
Do not start the next task.
```

### Standard explain-back prompt

``` text
Explain this specific file in beginner-friendly language.
Cover its purpose, inputs, outputs, important functions, and one example.
Then ask me 3 short questions to check my understanding.
Do not rewrite the file.
```

------------------------------------------------------------------------

## 13. Progress and Git checkpoints

After each working milestone, update `docs/progress.md` with exactly
these three items:

1.  **Completed:** what changed.
2.  **Verified:** test output, evaluation run ID, screenshot, or command
    result.
3.  **Next:** the single next task.

Maintain `docs/decisions.md` with short entries:

-   Decision
-   Reason
-   Alternative considered
-   Consequence/trade-off
-   Date or milestone

Use small commits such as:

``` text
docs: establish project scope
feat: add fixed-size text chunking
test: cover retrieval metric edge cases
feat: persist evaluation runs
feat: add baseline regression alerts
```

Before each commit:

``` powershell
git status
git diff --check
git diff
pytest
```

Run the relevant tests for the changed area; run the full suite at
milestone boundaries. Never commit generated secrets, virtual
environments, local databases, or fabricated reports.

------------------------------------------------------------------------

## 14. Definition of done

### Core requirements

-   [ ] Standalone repository and reproducible local setup.
-   [ ] Small, manually verified evaluation dataset.
-   [ ] Baseline retriever returns ranked, identifiable source chunks.
-   [ ] Recall@k, MRR, and nDCG@k implemented by hand with tests.
-   [ ] Versioned evaluation runs and per-question results.
-   [ ] SQLite history and baseline comparison.
-   [ ] Configurable regression alert with positive and negative tests.
-   [ ] Failure-analysis report with real examples.
-   [ ] Streamlit dashboard.
-   [ ] FastAPI interface, if it fits after the core.
-   [ ] GitHub Actions test/regression gate.
-   [ ] README, architecture, experiment table, limitations, and
    attribution.

### Advanced / conditional

-   [ ] Answer generation and judge evaluation, if budget and time
    allow.
-   [ ] Manual judge review with recorded disagreements.
-   [ ] BM25/hybrid/embedding/reranking experiments, only if core is
    stable.
-   [ ] Docker and hosted demo, after local reproducibility.

### Interview readiness

-   [ ] Explain Recall@k, MRR, and nDCG@k with a small example.
-   [ ] Explain the dataset and its weaknesses.
-   [ ] Explain why simulated drift is not the same as production drift.
-   [ ] Show one real experiment and one real failure.
-   [ ] Explain the alert threshold and how it is tested.
-   [ ] Explain the limitations of LLM-as-judge.
-   [ ] Explain which AI tools were used and what you personally
    reviewed and understood.
-   [ ] Deliver a clear 30-second and five-minute pitch.

A feature is not complete merely because code exists. It is complete
when the behavior is tested, results are real, and you can explain the
implementation.

------------------------------------------------------------------------

## 15. Interview pitches

### 30 seconds

"RAG systems can continue running even when retrieval quality gets
worse. I built RAGWatch, a standalone evaluation and quality-monitoring
tool that measures retrieval metrics, stores versioned runs, compares
configurations, analyzes failed questions, and raises threshold-based
regression alerts. I tested its monitoring behavior with controlled
changes and documented the limits of the dataset and drift simulations."

### Five minutes

1.  **Problem (30 seconds):** why a working RAG demo does not prove
    sustained quality.
2.  **Architecture (60 seconds):** documents, chunker, retriever,
    evaluation dataset, metrics, SQLite, alerts, dashboard.
3.  **Metrics (60 seconds):** explain Recall@k, MRR, and nDCG@k with a
    small example.
4.  **Experiment (60 seconds):** show one actual comparison from the
    saved results.
5.  **Failure and limitation (60 seconds):** explain a real failed case,
    what caused it, and what the test set cannot prove.
6.  **Next step (30 seconds):** explain how you would extend it using
    real traffic and a larger labeled dataset.

Do not fill this pitch with numbers until the corresponding experiment
has run.

------------------------------------------------------------------------

## 16. Risks and cut rules

  -----------------------------------------------------------------------
  Risk                                Response
  ----------------------------------- -----------------------------------
  Too much scope                      Core retrieval metrics, history,
                                      alerts, tests, and documentation
                                      come first

  LLM usage/cost                      Keep LLM answer judging optional,
                                      capped, and cached

  Dataset labels are weak             Manually verify labels and document
                                      the process

  Small test set creates noisy scores Show sample size and limitations;
                                      consider confidence intervals later

  Embedding model is too slow locally Start with TF-IDF and benchmark
                                      before adding embeddings

  Model/index mismatch                Record model metadata and reject
                                      incompatible index configurations

  Deployment dependency issues        Keep local reproducibility as the
                                      primary acceptance gate

  AI-generated code is not understood Explain-back check and focused code
                                      review before commit

  Phase exceeds estimate              If a phase takes more than 1.5× its
                                      estimate, cut optional features
                                      rather than skip tests

  Live hosting is unavailable or      Provide reproducible local demo
  unsuitable                          instructions and screenshots; do
                                      not make hosting a blocker
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## 17. First session: do only this

Do not generate the whole repository.

1.  Create the `ragwatch` folder and initialize Git.
2.  Create and activate `.venv`.
3.  Add `.gitignore` and `AGENTS.md`.
4.  Create `docs/problem.md` and `docs/decisions.md`.
5.  Ask Codex to create only `docs/requirements.md`.
6.  Review the requirements before asking Codex for
    `docs/architecture.md`.
7.  Commit the verified blueprint.

### First Codex prompt

``` text
We are starting RAGWatch, a standalone RAG evaluation and quality monitor.
Read AGENTS.md and inspect the workspace.

Create ONLY docs/problem.md and docs/decisions.md.
Do not write application code or install packages.

problem.md should state the problem, intended user, one-line goal,
MVP boundaries, and limitations in plain English.

decisions.md should record the current choices:
- standalone repository, no Project 1 integration
- local-first development
- deterministic retrieval metrics first
- optional and capped LLM judging later
- Streamlit dashboard
- deployment only after local tests pass
- use small, focused implementation tasks

Do not create any other files. Summarize what you created and stop.
```

------------------------------------------------------------------------

## 18. Research references

These references informed the dataset and deployment notes. Check the
current terms and resource limits when implementation begins; free tiers
and platform policies can change.

1.  **BEIR SciFact dataset card** --- dataset purpose, structure, and
    listed license:\
    https://huggingface.co/datasets/BeIR/scifact
2.  **BEIR paper** --- benchmark background and citation:\
    https://openreview.net/forum?id=wCu6T5xFjeJ
3.  **Streamlit Community Cloud documentation** --- deployment flow,
    repository selection, Python version, and secrets setup:\
    https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
4.  **Streamlit deployment overview** --- supported deployment options
    and Community Cloud context:\
    https://docs.streamlit.io/deploy
5.  **Hugging Face Spaces hardware documentation** --- hardware
    availability and pricing; verify current account/Space requirements
    before selecting it:\
    https://huggingface.co/docs/hub/main/spaces-gpus

### Research limitations

-   Hosting terms and free-tier limits can change; deployment is
    deliberately postponed.
-   The SciFact dataset is a candidate, not a mandatory dependency.
    Review licensing and attribution before bundling or redistributing
    data.
-   Model/API pricing and free access are not assumed. Verify the
    selected provider at implementation time.
-   The five-week schedule is a planning estimate, not a guarantee.

------------------------------------------------------------------------

**Final principle:** Build the smallest version that produces
trustworthy, reproducible evidence. Then expand it. A finished project
with real results, honest limitations, passing tests, and a clear
explanation is more valuable than a larger project with unverified
claims.
