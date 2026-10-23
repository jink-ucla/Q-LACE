# Portable supplementary copy: default paths/mount aliases sanitized; original hash is in FILE_MANIFEST.json.
#!/usr/bin/env python3
"""Build an eval-only Stage-1 longitudinal dataset from Chest ImaGenome gold comparisons."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from medgemma_visual_rft.longitudinal_schema import serialize_longitudinal_solution


DEFAULT_GOLD_TSV = Path(
    "./data/physionet.org/files/chest-imagenome/1.0.0/gold_dataset/gold_object_comparison_with_coordinates.txt"
)
DEFAULT_MIMIC_METADATA_CSV = Path(
    "./data/physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-metadata.csv"
)
DEFAULT_MIMIC_JPG_ROOT = Path(
    "./data/physionet.org/files/mimic-cxr-jpg/2.0.0/files"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "share_data" / "chest_imagenome_gold_stage1_eval"
UNSPECIFIED_ANATOMY = "unspecified"
COMPARISON_MAP = {
    "improved": "improving",
    "no change": "stable",
    "worsened": "worsening",
}
MS_CXR_T_HEAD_MAP = {
    "consolidation": "consolidation",
    "edema": "edema",
    "pleural effusion": "pleural effusion",
    "pneumonia": "pneumonia",
    "pneumothorax": "pneumothorax",
    "pulmonary edema/hazy opacity": "edema",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-tsv", type=Path, default=DEFAULT_GOLD_TSV)
    parser.add_argument("--mimic-metadata-csv", type=Path, default=DEFAULT_MIMIC_METADATA_CSV)
    parser.add_argument("--mimic-jpg-root", type=Path, default=DEFAULT_MIMIC_JPG_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--same-view-only", action="store_true")
    parser.add_argument(
        "--allowed-label-names",
        nargs="+",
        default=None,
        help="Optional allowlist of label names to keep.",
    )
    parser.add_argument(
        "--normalize-to-ms-cxr-t-heads",
        action="store_true",
        help=(
            "Project Chest ImaGenome labels onto the 5 MS-CXR-T finding heads for "
            "eval-only use. Original Chest ImaGenome labels are preserved in metadata."
        ),
    )
    return parser.parse_args()


def normalize_token(value: str) -> str:
    return " ".join(value.strip().split())


def slugify(value: str) -> str:
    cleaned = normalize_token(value).lower()
    slug_chars = [char if char.isalnum() else "_" for char in cleaned]
    slug = "".join(slug_chars)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")


def load_tsv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def load_metadata_index(metadata_csv: Path, *, needed_dicom_ids: set[str]) -> dict[str, dict[str, str]]:
    metadata_index: dict[str, dict[str, str]] = {}
    with metadata_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            dicom_id = normalize_token(row.get("dicom_id") or "")
            if dicom_id not in needed_dicom_ids:
                continue
            metadata_index[dicom_id] = {
                "subject_id": normalize_token(row.get("subject_id") or ""),
                "study_id": normalize_token(row.get("study_id") or ""),
                "view_position": normalize_token(row.get("ViewPosition") or "").upper(),
            }
    return metadata_index


def resolve_mimic_image_path(
    *,
    mimic_jpg_root: Path,
    subject_id: str,
    study_id: str,
    dicom_id: str,
) -> Path:
    subject_prefix = f"p{subject_id[:2]}"
    return mimic_jpg_root / subject_prefix / f"p{subject_id}" / f"s{study_id}" / f"{dicom_id}.jpg"


def build_problem(label_name: str) -> str:
    return (
        "Image 1 is the prior chest X-ray. Image 2 is the current chest X-ray. "
        "You must answer only the queried interval-change task. "
        f"Query: compared with the prior study, what is the interval change for {label_name}? "
        'Return exactly one JSON object with the form {"changes":[...]} and nothing else. '
        "Do not use markdown fences, code blocks, or a top-level JSON array. "
        "Each item in changes must contain exactly anatomy, finding, and change. "
        "The change value must be improving, stable, or worsening. "
        'If the queried change is not supported, return exactly {"changes":[]}.'
    )


def build_query_text(label_name: str) -> str:
    return f"compared with the prior study, what is the interval change for {label_name}?"


def normalize_label_name_for_eval(
    label_name: str,
    *,
    normalize_to_ms_cxr_t_heads: bool,
) -> str | None:
    normalized = normalize_token(label_name)
    if not normalize_to_ms_cxr_t_heads:
        return normalized
    return MS_CXR_T_HEAD_MAP.get(normalized.lower())


def build_dataset(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    raw_rows = load_tsv_rows(args.gold_tsv)
    allowed_label_names = (
        {normalize_token(value).lower() for value in args.allowed_label_names}
        if args.allowed_label_names
        else None
    )
    dataset_name = "chest_imagenome_gold_object_comparison"
    if args.normalize_to_ms_cxr_t_heads:
        dataset_name = f"{dataset_name}_ms_cxr_t_heads"

    needed_dicom_ids: set[str] = set()
    for row in raw_rows:
        current_image_id = normalize_token(row.get("current_image_id") or "")
        previous_image_id = normalize_token(row.get("previous_image_id") or "")
        if current_image_id:
            needed_dicom_ids.add(current_image_id)
        if previous_image_id:
            needed_dicom_ids.add(previous_image_id)
    metadata_index = load_metadata_index(args.mimic_metadata_csv, needed_dicom_ids=needed_dicom_ids)

    grouped_rows: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    skipped = Counter()
    comparison_counts = Counter()
    label_counts = Counter()
    original_label_counts = Counter()
    view_pair_counts = Counter()
    patients = set()

    for row in raw_rows:
        original_label_name = normalize_token(row.get("label_name") or "")
        comparison_raw = normalize_token(row.get("comparison") or "").lower()
        current_image_id = normalize_token(row.get("current_image_id") or "")
        previous_image_id = normalize_token(row.get("previous_image_id") or "")
        patient_id = normalize_token(row.get("patient_id") or "")
        if (
            not original_label_name
            or not comparison_raw
            or not current_image_id
            or not previous_image_id
            or not patient_id
        ):
            skipped["missing_required_field"] += 1
            continue
        if ";;" in comparison_raw:
            skipped["ambiguous_multi_label_comparison"] += 1
            continue
        comparison = COMPARISON_MAP.get(comparison_raw)
        if comparison is None:
            skipped["unsupported_comparison"] += 1
            continue
        label_name = normalize_label_name_for_eval(
            original_label_name,
            normalize_to_ms_cxr_t_heads=args.normalize_to_ms_cxr_t_heads,
        )
        if label_name is None:
            skipped["label_name_not_mapped_to_ms_cxr_t_head"] += 1
            continue
        if allowed_label_names is not None and label_name.lower() not in allowed_label_names:
            skipped["label_name_filtered"] += 1
            continue
        current_meta = metadata_index.get(current_image_id)
        previous_meta = metadata_index.get(previous_image_id)
        if current_meta is None or previous_meta is None:
            skipped["missing_metadata"] += 1
            continue
        current_image_path = resolve_mimic_image_path(
            mimic_jpg_root=args.mimic_jpg_root,
            subject_id=current_meta["subject_id"],
            study_id=current_meta["study_id"],
            dicom_id=current_image_id,
        )
        previous_image_path = resolve_mimic_image_path(
            mimic_jpg_root=args.mimic_jpg_root,
            subject_id=previous_meta["subject_id"],
            study_id=previous_meta["study_id"],
            dicom_id=previous_image_id,
        )
        if not current_image_path.exists() or not previous_image_path.exists():
            skipped["missing_image_path"] += 1
            continue
        current_view = current_meta["view_position"]
        previous_view = previous_meta["view_position"]
        if args.same_view_only and current_view and previous_view and current_view != previous_view:
            skipped["mixed_view_pair"] += 1
            continue
        group_key = (patient_id, previous_image_id, current_image_id, label_name.lower())
        grouped_rows[group_key].append(
            {
                **row,
                "label_name": label_name,
                "original_label_name": original_label_name,
                "comparison": comparison,
                "patient_id": patient_id,
                "subject_id": current_meta["subject_id"] or patient_id,
                "previous_study_id": previous_meta["study_id"],
                "study_id": current_meta["study_id"],
                "previous_view_position": previous_view,
                "current_view_position": current_view,
                "previous_image_path": str(previous_image_path),
                "current_image_path": str(current_image_path),
            }
        )

    query_rows: list[dict[str, Any]] = []
    pair_rows_by_id: dict[str, dict[str, Any]] = {}

    for (patient_id, previous_image_id, current_image_id, _label_name_key), rows in sorted(grouped_rows.items()):
        comparisons = {row["comparison"] for row in rows}
        if len(comparisons) != 1:
            skipped["conflicting_grouped_comparison"] += 1
            continue
        selected = rows[0]
        comparison = selected["comparison"]
        label_name = selected["label_name"]
        original_label_names = sorted(
            {
                normalize_token(str(row.get("original_label_name") or ""))
                for row in rows
                if normalize_token(str(row.get("original_label_name") or ""))
            }
        )
        pair_id = (
            f"chestimagenome:{patient_id}:{selected['previous_study_id']}:{selected['study_id']}:"
            f"{previous_image_id}:{current_image_id}"
        )
        source_id = f"{pair_id}::stage1::{slugify(label_name)}"
        problem = build_problem(label_name)
        solution = serialize_longitudinal_solution(
            [{"anatomy": UNSPECIFIED_ANATOMY, "finding": label_name, "change": comparison}]
        )
        view_pair = f"{selected['previous_view_position'] or 'UNK'}->{selected['current_view_position'] or 'UNK'}"
        query_rows.append(
            {
                "source_id": source_id,
                "pair_id": pair_id,
                "split": "test",
                "patient_id": patient_id,
                "subject_id": selected["subject_id"],
                "prior_study_id": selected["previous_study_id"],
                "study_id": selected["study_id"],
                "prior_dicom_id": previous_image_id,
                "dicom_id": current_image_id,
                "prior_image_path": selected["previous_image_path"],
                "current_image_path": selected["current_image_path"],
                "prior_view_position": selected["previous_view_position"],
                "current_view_position": selected["current_view_position"],
                "problem": problem,
                "query": build_query_text(label_name),
                "query_finding": label_name,
                "query_change": comparison,
                "solution": solution,
                "task_mode": "stage1",
                "label_status": dataset_name,
                "metadata": {
                    "dataset_name": dataset_name,
                    "patient_id": patient_id,
                    "relationship_id": normalize_token(selected.get("relationship_id") or ""),
                    "bbox": normalize_token(selected.get("bbox") or ""),
                    "attribute": normalize_token(selected.get("attribute") or ""),
                    "sentence": normalize_token(selected.get("sentence") or ""),
                    "label_name": original_label_names[0] if original_label_names else "",
                    "original_label_name": original_label_names[0] if original_label_names else "",
                    "original_label_names": original_label_names,
                    "projected_query_finding": label_name,
                    "normalize_to_ms_cxr_t_heads": args.normalize_to_ms_cxr_t_heads,
                    "current_image_id": current_image_id,
                    "previous_image_id": previous_image_id,
                    "duplicate_group_size": len(rows),
                    "source_comparison": normalize_token(rows[0].get("comparison") or ""),
                },
            }
        )
        pair_rows_by_id.setdefault(
            pair_id,
            {
                "pair_id": pair_id,
                "source_id": pair_id,
                "split": "test",
                "patient_id": patient_id,
                "subject_id": selected["subject_id"],
                "prior_study_id": selected["previous_study_id"],
                "study_id": selected["study_id"],
                "prior_dicom_id": previous_image_id,
                "dicom_id": current_image_id,
                "prior_image_path": selected["previous_image_path"],
                "current_image_path": selected["current_image_path"],
                "prior_view_position": selected["previous_view_position"],
                "current_view_position": selected["current_view_position"],
                "dataset_name": dataset_name,
            },
        )
        comparison_counts[comparison] += 1
        label_counts[label_name] += 1
        for original_label_name in original_label_names:
            original_label_counts[original_label_name] += 1
        view_pair_counts[view_pair] += 1
        patients.add(patient_id)

    pair_rows = list(pair_rows_by_id.values())
    summary = {
        "dataset_name": dataset_name,
        "gold_tsv": str(args.gold_tsv),
        "mimic_metadata_csv": str(args.mimic_metadata_csv),
        "mimic_jpg_root": str(args.mimic_jpg_root),
        "same_view_only": args.same_view_only,
        "normalize_to_ms_cxr_t_heads": args.normalize_to_ms_cxr_t_heads,
        "allowed_label_names": sorted(allowed_label_names) if allowed_label_names is not None else None,
        "query_row_count": len(query_rows),
        "pair_row_count": len(pair_rows),
        "patient_count": len(patients),
        "comparison_counts": dict(comparison_counts),
        "top_label_counts": dict(label_counts.most_common(25)),
        "top_original_label_counts": dict(original_label_counts.most_common(25)),
        "view_pair_counts": dict(view_pair_counts),
        "skipped_counts": dict(skipped),
    }
    return query_rows, pair_rows, summary


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    query_rows, pair_rows, summary = build_dataset(args)
    write_jsonl(args.output_dir / "all.jsonl", query_rows)
    write_jsonl(args.output_dir / "test.jsonl", query_rows)
    write_jsonl(args.output_dir / "benchmark_pairs.jsonl", pair_rows)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
