#!/usr/bin/env python3
"""Build Stage-1 temporal-control datasets from benchmark-style JSONL."""

from __future__ import annotations

import argparse
import base64
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_JSONL = REPO_ROOT / "share_data" / "ms_cxr_t_longitudinal_stage1" / "test.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "share_data" / "ms_cxr_t_longitudinal_stage1_temporal_controls"
CONTROL_NAMES = ("baseline", "prior_drop", "duplicate_pair", "time_reversal", "random_prior")
PRIOR_CURRENT_FIELD_PAIRS = (
    ("prior_image_path", "current_image_path"),
    ("prior_view_position", "current_view_position"),
    ("prior_dicom_id", "dicom_id"),
    ("prior_study_id", "study_id"),
    ("prior_report_sections", "current_report_sections"),
)
BLANK_PRIOR_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/Zp8AAAAASUVORK5CYII="
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT_JSONL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed for random-prior donor selection.")
    return parser.parse_args(argv)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if not isinstance(row, dict):
                raise SystemExit(f"Expected object record on line {line_number} of {path}")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def ensure_blank_prior_asset(output_dir: Path) -> Path:
    asset_path = output_dir / "control_assets" / "prior_drop_blank.png"
    asset_path.parent.mkdir(parents=True, exist_ok=True)
    if not asset_path.exists():
        asset_path.write_bytes(BLANK_PRIOR_PNG_BYTES)
    return asset_path


def normalize_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip()


def anchor_source_id(row: dict[str, Any], row_index: int) -> str:
    for key in ("source_id", "example_id", "query_id", "pair_id"):
        value = normalize_text(row.get(key))
        if value:
            return value
    return f"row-{row_index}"


def clone_row(row: dict[str, Any]) -> dict[str, Any]:
    cloned = deepcopy(row)
    metadata = cloned.get("metadata")
    if not isinstance(metadata, dict):
        cloned["metadata"] = {}
    return cloned


def suffix_identifier(value: Any, control_name: str) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return value
    return f"{stripped}::control::{control_name}"


def invert_query_change(value: Any) -> str | None:
    normalized = normalize_text(value).lower()
    if normalized == "improving":
        return "worsening"
    if normalized == "worsening":
        return "improving"
    if normalized == "stable":
        return "stable"
    return None


def hashed_choice(options: list[int], *, seed: int, anchor_id: str) -> int:
    if not options:
        raise ValueError("options must be non-empty")
    digest = hashlib.sha1(f"{seed}:{anchor_id}".encode("utf-8")).hexdigest()
    return options[int(digest[:8], 16) % len(options)]


def choose_random_prior_donor(rows: list[dict[str, Any]], row_index: int, *, seed: int) -> tuple[int, str]:
    target = rows[row_index]
    target_anchor = anchor_source_id(target, row_index)
    target_patient = normalize_text(target.get("patient_id"))
    target_split = normalize_text(target.get("split"))

    def row_anchor(index: int) -> str:
        return anchor_source_id(rows[index], index)

    all_indices = list(range(len(rows)))
    candidate_groups = (
        (
            "same_split_different_patient",
            [
                index
                for index in all_indices
                if index != row_index
                and normalize_text(rows[index].get("split")) == target_split
                and normalize_text(rows[index].get("patient_id")) != target_patient
            ],
        ),
        (
            "global_different_patient",
            [
                index
                for index in all_indices
                if index != row_index and normalize_text(rows[index].get("patient_id")) != target_patient
            ],
        ),
        (
            "same_split_different_row",
            [
                index
                for index in all_indices
                if index != row_index and normalize_text(rows[index].get("split")) == target_split
            ],
        ),
        ("global_different_row", [index for index in all_indices if index != row_index]),
        ("self_fallback", [row_index]),
    )

    for selection_mode, candidates in candidate_groups:
        if not candidates:
            continue
        ordered = sorted(candidates, key=row_anchor)
        return hashed_choice(ordered, seed=seed, anchor_id=target_anchor), selection_mode
    raise AssertionError("Random-prior donor selection unexpectedly failed")


def apply_control_identifiers(
    row: dict[str, Any],
    *,
    control_name: str,
    original_anchor_id: str,
    row_index: int,
    seed: int,
) -> None:
    for key in ("source_id", "example_id", "query_id"):
        if key in row:
            row[key] = suffix_identifier(row.get(key), control_name)
    if "source_id" not in row or not normalize_text(row.get("source_id")):
        row["source_id"] = suffix_identifier(original_anchor_id, control_name)
    row["control_name"] = control_name
    metadata = row["metadata"]
    metadata.update(
        {
            "temporal_control_name": control_name,
            "temporal_control_source_id": original_anchor_id,
            "temporal_control_row_index": row_index,
            "temporal_control_seed": seed,
            "temporal_control_builder": "build_stage1_temporal_controls.py",
        }
    )


def build_baseline_row(
    raw_row: dict[str, Any],
    *,
    row_index: int,
    seed: int,
) -> dict[str, Any]:
    row = clone_row(raw_row)
    anchor_id = anchor_source_id(raw_row, row_index)
    apply_control_identifiers(row, control_name="baseline", original_anchor_id=anchor_id, row_index=row_index, seed=seed)
    row["metadata"]["temporal_control_modified_fields"] = []
    return row


def build_prior_drop_row(
    raw_row: dict[str, Any],
    *,
    row_index: int,
    seed: int,
    blank_prior_path: Path,
) -> dict[str, Any]:
    row = clone_row(raw_row)
    anchor_id = anchor_source_id(raw_row, row_index)
    apply_control_identifiers(row, control_name="prior_drop", original_anchor_id=anchor_id, row_index=row_index, seed=seed)

    modified_fields: list[str] = []
    if "prior_image_path" in row:
        row["prior_image_path"] = str(blank_prior_path)
        modified_fields.append("prior_image_path")
    if "prior_view_position" in row:
        row["prior_view_position"] = "CONTROL_BLANK"
        modified_fields.append("prior_view_position")
    if "prior_dicom_id" in row:
        row["prior_dicom_id"] = "CONTROL_BLANK_PRIOR"
        modified_fields.append("prior_dicom_id")
    if "prior_study_id" in row:
        row["prior_study_id"] = "CONTROL_BLANK_PRIOR"
        modified_fields.append("prior_study_id")
    if "prior_report_sections" in row:
        row["prior_report_sections"] = {}
        modified_fields.append("prior_report_sections")

    row["metadata"]["temporal_control_blank_prior_image_path"] = str(blank_prior_path)
    row["metadata"]["temporal_control_modified_fields"] = modified_fields
    return row


def build_duplicate_pair_row(
    raw_row: dict[str, Any],
    *,
    row_index: int,
    seed: int,
) -> dict[str, Any]:
    row = clone_row(raw_row)
    anchor_id = anchor_source_id(raw_row, row_index)
    apply_control_identifiers(
        row,
        control_name="duplicate_pair",
        original_anchor_id=anchor_id,
        row_index=row_index,
        seed=seed,
    )

    modified_fields: list[str] = []
    for prior_key, current_key in PRIOR_CURRENT_FIELD_PAIRS:
        if current_key in raw_row:
            row[prior_key] = deepcopy(raw_row[current_key])
            modified_fields.append(prior_key)
    row["metadata"]["temporal_control_modified_fields"] = modified_fields
    row["metadata"]["temporal_control_duplicate_strategy"] = "current_as_prior"
    return row


def build_time_reversal_row(
    raw_row: dict[str, Any],
    *,
    row_index: int,
    seed: int,
) -> dict[str, Any]:
    row = clone_row(raw_row)
    anchor_id = anchor_source_id(raw_row, row_index)
    apply_control_identifiers(
        row,
        control_name="time_reversal",
        original_anchor_id=anchor_id,
        row_index=row_index,
        seed=seed,
    )

    modified_fields: list[str] = []
    for prior_key, current_key in PRIOR_CURRENT_FIELD_PAIRS:
        prior_present = prior_key in raw_row
        current_present = current_key in raw_row
        if not prior_present and not current_present:
            continue
        row[prior_key] = deepcopy(raw_row.get(current_key))
        row[current_key] = deepcopy(raw_row.get(prior_key))
        modified_fields.extend([prior_key, current_key])

    row["metadata"]["temporal_control_modified_fields"] = modified_fields
    row["metadata"]["temporal_control_expected_reversed_query_change"] = invert_query_change(raw_row.get("query_change"))
    return row


def build_random_prior_row(
    raw_row: dict[str, Any],
    *,
    donor_row: dict[str, Any],
    donor_index: int,
    row_index: int,
    seed: int,
    selection_mode: str,
) -> dict[str, Any]:
    row = clone_row(raw_row)
    anchor_id = anchor_source_id(raw_row, row_index)
    apply_control_identifiers(
        row,
        control_name="random_prior",
        original_anchor_id=anchor_id,
        row_index=row_index,
        seed=seed,
    )

    modified_fields: list[str] = []
    for prior_key, _current_key in PRIOR_CURRENT_FIELD_PAIRS:
        if prior_key in donor_row:
            row[prior_key] = deepcopy(donor_row[prior_key])
            modified_fields.append(prior_key)

    row["metadata"]["temporal_control_modified_fields"] = modified_fields
    row["metadata"]["temporal_control_random_prior_selection"] = selection_mode
    row["metadata"]["temporal_control_random_prior_source_id"] = anchor_source_id(donor_row, donor_index)
    row["metadata"]["temporal_control_random_prior_patient_id"] = normalize_text(donor_row.get("patient_id"))
    row["metadata"]["temporal_control_random_prior_split"] = normalize_text(donor_row.get("split"))
    row["metadata"]["temporal_control_random_prior_self_fallback"] = donor_index == row_index
    return row


def build_summary(
    input_rows: list[dict[str, Any]],
    rows_by_control: dict[str, list[dict[str, Any]]],
    *,
    seed: int,
    blank_prior_path: Path,
) -> dict[str, Any]:
    all_controls = [row for control_name in CONTROL_NAMES for row in rows_by_control[control_name]]
    input_split_counts = Counter(normalize_text(row.get("split")) or "unknown" for row in input_rows)
    control_counts = {control_name: len(rows_by_control[control_name]) for control_name in CONTROL_NAMES}
    control_split_counts: dict[str, dict[str, int]] = {}
    control_query_change_counts: dict[str, dict[str, int]] = {}
    random_prior_selection_counts = Counter()
    random_prior_self_fallback_count = 0

    for control_name, rows in rows_by_control.items():
        control_split_counts[control_name] = dict(
            Counter(normalize_text(row.get("split")) or "unknown" for row in rows)
        )
        control_query_change_counts[control_name] = dict(
            Counter(normalize_text(row.get("query_change")) or "unknown" for row in rows)
        )
        if control_name == "random_prior":
            for row in rows:
                metadata = row.get("metadata") or {}
                if not isinstance(metadata, dict):
                    continue
                selection = normalize_text(metadata.get("temporal_control_random_prior_selection")) or "unknown"
                random_prior_selection_counts[selection] += 1
                if bool(metadata.get("temporal_control_random_prior_self_fallback")):
                    random_prior_self_fallback_count += 1

    return {
        "input_row_count": len(input_rows),
        "count": len(all_controls),
        "control_order": list(CONTROL_NAMES),
        "control_counts": control_counts,
        "input_split_counts": dict(input_split_counts),
        "control_split_counts": control_split_counts,
        "control_query_change_counts": control_query_change_counts,
        "seed": seed,
        "blank_prior_image_path": str(blank_prior_path),
        "random_prior_selection_counts": dict(random_prior_selection_counts),
        "random_prior_self_fallback_count": random_prior_self_fallback_count,
    }


def build_dataset(args: argparse.Namespace) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    output_dir = Path(args.output_dir).resolve()
    input_rows = load_jsonl(Path(args.input_jsonl))
    blank_prior_path = ensure_blank_prior_asset(output_dir)

    rows_by_control: dict[str, list[dict[str, Any]]] = {control_name: [] for control_name in CONTROL_NAMES}
    for row_index, raw_row in enumerate(input_rows):
        rows_by_control["baseline"].append(build_baseline_row(raw_row, row_index=row_index, seed=args.seed))
        rows_by_control["prior_drop"].append(
            build_prior_drop_row(raw_row, row_index=row_index, seed=args.seed, blank_prior_path=blank_prior_path)
        )
        rows_by_control["duplicate_pair"].append(build_duplicate_pair_row(raw_row, row_index=row_index, seed=args.seed))
        rows_by_control["time_reversal"].append(build_time_reversal_row(raw_row, row_index=row_index, seed=args.seed))

        donor_index, selection_mode = choose_random_prior_donor(input_rows, row_index, seed=args.seed)
        rows_by_control["random_prior"].append(
            build_random_prior_row(
                raw_row,
                donor_row=input_rows[donor_index],
                donor_index=donor_index,
                row_index=row_index,
                seed=args.seed,
                selection_mode=selection_mode,
            )
        )

    summary = build_summary(input_rows, rows_by_control, seed=args.seed, blank_prior_path=blank_prior_path)
    return rows_by_control, summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows_by_control, summary = build_dataset(args)
    all_controls = [row for control_name in CONTROL_NAMES for row in rows_by_control[control_name]]

    for control_name, rows in rows_by_control.items():
        write_jsonl(output_dir / f"{control_name}.jsonl", rows)
    write_jsonl(output_dir / "all_controls.jsonl", all_controls)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({"output_dir": str(output_dir), "summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
