#!/usr/bin/env python3
"""Filter JSON/JSONL records by patient identifiers extracted from other records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", dest="input_path", type=Path, required=True)
    parser.add_argument(
        "--exclude-records",
        dest="exclude_paths",
        action="append",
        type=Path,
        required=True,
        help="JSON/JSONL file containing benchmark or gold records whose patient IDs should be excluded.",
    )
    parser.add_argument("--output", dest="output_path", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, default=None)
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                payload = json.loads(stripped)
                if not isinstance(payload, dict):
                    raise SystemExit(f"Expected object on line {line_number} of {path}")
                records.append(payload)
        return records

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list) and all(isinstance(item, dict) for item in payload):
        return list(payload)
    if isinstance(payload, dict):
        for key in ("records", "rows", "items", "data", "examples", "samples"):
            value = payload.get(key)
            if isinstance(value, list) and all(isinstance(item, dict) for item in value):
                return list(value)
    raise SystemExit(f"Unsupported JSON structure in {path}")


def extract_patient_id(record: dict[str, Any]) -> str | None:
    for key in ("patient_id", "subject_id"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, int):
            return str(value)
    return None


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def main() -> int:
    args = parse_args()
    input_records = load_records(args.input_path)
    exclude_records: list[dict[str, Any]] = []
    for path in args.exclude_paths:
        exclude_records.extend(load_records(path))

    excluded_patient_ids = {
        patient_id
        for patient_id in (extract_patient_id(record) for record in exclude_records)
        if patient_id is not None
    }
    kept_records: list[dict[str, Any]] = []
    removed_records: list[dict[str, Any]] = []
    for record in input_records:
        patient_id = extract_patient_id(record)
        if patient_id is not None and patient_id in excluded_patient_ids:
            removed_records.append(record)
        else:
            kept_records.append(record)

    write_jsonl(args.output_path, kept_records)
    summary = {
        "input_path": str(args.input_path),
        "output_path": str(args.output_path),
        "exclude_paths": [str(path) for path in args.exclude_paths],
        "input_count": len(input_records),
        "kept_count": len(kept_records),
        "removed_count": len(removed_records),
        "excluded_patient_id_count": len(excluded_patient_ids),
        "removed_patient_ids": sorted(
            {
                patient_id
                for patient_id in (extract_patient_id(record) for record in removed_records)
                if patient_id is not None
            }
        ),
    }
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
