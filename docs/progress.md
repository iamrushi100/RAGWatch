# RAGWatch Progress

## Completed
- Created the project workspace and initialized Git.
- Created the .venv Python virtual environment.
- Created the initial project directories and .gitignore.
- Added optional JSON evaluation report export through the CLI.
- Added deterministic, readable JSON output and creation of missing parent directories.
- Added opt-in CLI and runner integration for aggregate threshold checks and per-question failure analysis. Requested diagnostics are exported as top-level `threshold_check` and `failure_analysis` fields; default report fields remain unchanged.
- Added a read-only Streamlit dashboard at `app.py` for existing JSON reports. Launch locally with `streamlit run app.py` after installing `requirements.txt`.

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

## Next
- Review the report-export changes before committing this milestone.
- Plan the next implementation task.
