"""Safe discovery helpers for local SQLite history databases in the dashboard."""

from pathlib import Path

from src.history import compare_runs, list_runs


def discover_history_databases(data_directory: str | Path) -> list[Path]:
    """Find SQLite files contained in the project's data directory.

    Limiting discovery to this directory keeps the dashboard from accepting
    arbitrary local filesystem paths. Symlinks that resolve outside it are
    ignored.
    """
    root = Path(data_directory)
    if not root.is_dir():
        return []

    files = []
    for pattern in ("*.sqlite3", "*.db"):
        for candidate in root.rglob(pattern):
            try:
                candidate.resolve().relative_to(root.resolve())
            except (OSError, ValueError):
                continue
            if candidate.is_file():
                files.append(candidate)

    return sorted(set(files), key=lambda path: path.relative_to(root).as_posix().casefold())


def load_history_runs(database_path: str | Path) -> list[dict]:
    """Return run summaries through the existing SQLite history API."""
    return list_runs(database_path)


def compare_history_runs(
    database_path: str | Path, baseline_run_id: int, candidate_run_id: int
) -> dict:
    """Compare runs through the existing history API and its validation."""
    return compare_runs(database_path, baseline_run_id, candidate_run_id)
