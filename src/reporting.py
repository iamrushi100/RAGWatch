"""Write evaluation results as readable, deterministic JSON reports."""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def export_json_report(
    evaluation_result: Mapping[str, Any], output_path: str | Path
) -> Path:
    """Save an evaluation runner result as formatted JSON and return its path.

    Object keys are sorted and a final newline is included so the same result
    produces stable, easy-to-read report text. Missing parent directories are
    created. File-system errors are left for the caller to report clearly.
    """
    path = Path(output_path)
    report_text = json.dumps(evaluation_result, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report_text, encoding="utf-8")
    return path
