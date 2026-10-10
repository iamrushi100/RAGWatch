# RAGWatch - Agent Instructions

## Project Goal
Build a beginner-friendly, standalone RAG evaluation and
quality-monitoring application suitable for a placement portfolio.

## Working Rules
- Read this file before starting a task.
- Implement only the task explicitly requested.
- Inspect relevant files before editing them.
- Do not generate the entire application at once.
- Keep code simple, readable, and beginner-friendly.
- Explain important design choices in plain English.
- Do not add dependencies without explaining why they are needed.
- Do not invent test results, benchmark results, or metrics.
- Do not modify unrelated files.
- Never expose or commit API keys, passwords, or secrets.

## Development Priorities
1. Establish project requirements and architecture.
2. Build a small, deterministic RAG retrieval baseline.
3. Implement and test Recall@k, MRR, and nDCG@k.
4. Add repeatable evaluation runs and failure analysis.
5. Add SQLite history, comparisons, and regression alerts.
6. Build the dashboard and optional API.
7. Add continuous integration and finalize documentation.

## Cost and Scope
- The core project must not require paid APIs.
- LLM-based answer judging is optional and must be capped.
- Avoid unnecessary model downloads and repeated AI calls.
- Do not implement future phases before current tests pass.

## Verification
- Add focused tests for implemented behavior.
- Report exact commands run and actual outcomes.
- Identify untested behavior and known limitations.
- Stop after completing the requested task.

## Learning
For every implementation task, explain:
- What changed and why.
- How the implementation works.
- One concept the developer should understand for interviews.
