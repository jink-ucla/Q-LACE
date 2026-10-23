from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "filter_json_records_by_patient_ids.py"


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_filters_rows_using_patient_or_subject_ids(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    exclude_path = tmp_path / "exclude.jsonl"
    output_path = tmp_path / "output.jsonl"
    summary_path = tmp_path / "summary.json"

    write_jsonl(
        input_path,
        [
            {"source_id": "a", "patient_id": "p1"},
            {"source_id": "b", "patient_id": "p2"},
            {"source_id": "c", "subject_id": "p3"},
            {"source_id": "d"},
        ],
    )
    write_jsonl(
        exclude_path,
        [
            {"subject_id": "p2"},
            {"patient_id": "p3"},
        ],
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--input",
            str(input_path),
            "--exclude-records",
            str(exclude_path),
            "--output",
            str(output_path),
            "--summary-json",
            str(summary_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads(completed.stdout)
    assert summary["input_count"] == 4
    assert summary["kept_count"] == 2
    assert summary["removed_count"] == 2
    assert summary["removed_patient_ids"] == ["p2", "p3"]

    kept = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert [row["source_id"] for row in kept] == ["a", "d"]
