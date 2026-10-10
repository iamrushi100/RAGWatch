# RAGWatch Progress

## Completed
- Created the project workspace and initialized Git.
- Created the .venv Python virtual environment.
- Created the initial project directories and .gitignore.
- Added optional JSON evaluation report export through the CLI.
- Added deterministic, readable JSON output and creation of missing parent directories.
- Added opt-in CLI and runner integration for aggregate threshold checks and per-question failure analysis. Requested diagnostics are exported as top-level `threshold_check` and `failure_analysis` fields; default report fields remain unchanged.
- Added a read-only Streamlit dashboard at `app.py` for existing JSON reports. Launch locally with `streamlit run app.py` after installing `requirements.txt`.
- Added fixed-size, overlapping character chunking and changed the sample dataset to chunk-level relevance labels. Original source text and IDs remain available for traceability.
- Added a controlled ranking-inversion experiment. It evaluates TF-IDF and the reversed ranking against the same loaded dataset and exports both metric summaries plus candidate-minus-baseline differences. See `docs/retrieval_regression.md`.
- Connected the evaluation CLI to the existing SQLite run-history API with optional `--history-db` storage, and added a read-only run list and comparison section to the Streamlit dashboard.

## SQLite history usage
- Save an evaluation run to the dashboard-visible database: `python -m src.cli data/sample_dataset.json --k 2 --history-db data/ragwatch_history.sqlite3`
- Launch the dashboard to inspect and compare saved runs: `streamlit run app.py`
- History storage is optional. Without `--history-db`, CLI evaluation and JSON report behavior remains unchanged.

## Verified
- Python 3.13.0 is available.
- Git is available and the repository is initialized.
- Virtual environment activation succeeded.
- Command: `python -m pytest -p no:cacheprovider tests/test_cli.py tests/test_reporting.py tests/test_evaluation.py tests/test_dataset.py tests/test_retriever.py --basetemp .pytest_tmp`
- Result: 68 passed in 2.30s.
- Command: `python -m pytest -p no:cacheprovider tests/test_cli.py tests/test_reporting.py tests/test_evaluation.py tests/test_alerts.py tests/test_failure_analysis.py --basetemp .pytest_tmp`
- Result: 93 passed.
- Command: `python -m pytest -p no:cacheprovider tests --basetemp .pytest_tmp`
- Result: 130 passed.
- Command: `python -m pytest -p no:cacheprovider tests/test_dashboard_data.py --basetemp .pytest_tmp`
- Result: 8 passed.
- Command: `python -m pytest -p no:cacheprovider tests --basetemp .pytest_tmp`
- Result: 138 passed.
- Command: `python -m pytest -p no:cacheprovider tests/test_chunking.py tests/test_dataset.py tests/test_retriever.py tests/test_evaluation.py tests/test_failure_analysis.py tests/test_dashboard_data.py --basetemp .pytest_tmp`
- Result: 99 passed.
- Command: `python -m pytest -p no:cacheprovider tests --basetemp .pytest_tmp`
- Result: 155 passed.
- Focused regression experiment tests and related evaluation tests: `python -m pytest -p no:cacheprovider tests/test_regression_experiment.py tests/test_evaluation.py tests/test_retriever.py tests/test_dataset.py --basetemp .pytest_tmp_regression`
- Result: 73 passed.
- Full suite after the experiment changes: `python -m pytest -p no:cacheprovider tests --basetemp .pytest_tmp_regression_full`
- Result: 163 passed.
- Experiment report generated locally with `python -m src.regression_experiment` at `reports/retrieval-regression-experiment.json`.

## Next
- Connect the SQLite run history and comparisons to the CLI/dashboard workflow in a focused follow-up.
