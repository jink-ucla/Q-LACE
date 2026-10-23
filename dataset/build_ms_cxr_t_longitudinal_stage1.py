# Portable supplementary copy: default paths/mount aliases sanitized; original hash is in FILE_MANIFEST.json.
#!/usr/bin/env python3
"""Build a query-conditioned Stage-1 longitudinal benchmark dataset from MS-CXR-T."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from medgemma_visual_rft.longitudinal_schema import serialize_longitudinal_solution


DEFAULT_MS_CXR_T_CSV = Path(
    "./data/physionet.org/files/ms-cxr-t/1.0.0/MS_CXR_T_temporal_image_classification_v1.0.0.csv"
)
DEFAULT_MIMIC_METADATA_CSV = Path(
    "./data/physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-metadata.csv"
)
DEFAULT_MIMIC_JPG_ROOT = Path(
    "./data/physionet.org/files/mimic-cxr-jpg/2.0.0/files"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "share_data" / "ms_cxr_t_longitudinal_stage1"
DEFAULT_ALLOWED_LABEL_QUALITIES = ("one_expert", "multiple_experts", "disagreement")
DEFAULT_FINDINGS = (
    ("consolidation", "consolidation_progression", "consolidation_label_quality"),
    ("edema", "edema_progression", "edema_label_quality"),
    ("pleural effusion", "pleural_effusion_progression", "pleural_effusion_label_quality"),
    ("pneumonia", "pneumonia_progression", "pneumonia_label_quality"),
    ("pneumothorax", "pneumothorax_progression", "pneumothorax_label_quality"),
)
ALLOWED_PROGRESSION = frozenset({"improving", "stable", "worsening"})
UNSPECIFIED_ANATOMY = "unspecified"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ms-cxr-t-csv", type=Path, default=DEFAULT_MS_CXR_T_CSV)
    parser.add_argument("--mimic-metadata-csv", type=Path, default=DEFAULT_MIMIC_METADATA_CSV)
    parser.add_argument("--mimic-jpg-root", type=Path, default=DEFAULT_MIMIC_JPG_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--split-policy",
        choices=("all_test", "patient_hash"),
        default="all_test",
        help="`all_test` keeps every benchmark row in test; `patient_hash` creates a deterministic synthetic patient split.",
    )
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument(
        "--allowed-label-qualities",
        nargs="+",
        default=list(DEFAULT_ALLOWED_LABEL_QUALITIES),
        help="Keep only findings whose MS-CXR-T label quality is in this allowlist.",
    )
    parser.add_argument("--same-view-only", action="store_true")
    return parser.parse_args()


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def normalize_token(value: str) -> str:
    return " ".join(value.strip().split())


def extract_dicom_leaf(value: str) -> str:
    cleaned = normalize_token(value)
    return cleaned.rsplit("/", 1)[-1]


def normalize_progression(value: str) -> str | None:
    cleaned = normalize_token(value).lower()
    if not cleaned:
        return None
    if cleaned not in ALLOWED_PROGRESSION:
        raise SystemExit(f"Unsupported MS-CXR-T progression value: {value!r}")
    return cleaned


def normalize_quality(value: str) -> str | None:
    cleaned = normalize_token(value).lower()
    return cleaned or None


def build_problem(finding: str) -> str:
    return (
        "Image 1 is the prior chest X-ray. Image 2 is the current chest X-ray. "
        "You must answer only the queried interval-change task. "
        f"Query: compared with the prior study, what is the interval change for {finding}? "
        'Return exactly one JSON object with the form {"changes":[...]} and nothing else. '
        "Do not use markdown fences, code blocks, or a top-level JSON array. "
        "Each item in changes must contain exactly anatomy, finding, and change. "
        "The change value must be improving, stable, or worsening. "
        'If the queried change is not supported, return exactly {"changes":[]}.'
    )


def build_query_text(finding: str) -> str:
    return f"compared with the prior study, what is the interval change for {finding}?"


def build_pair_id(subject_id: str, previous_study_id: str, study_id: str) -> str:
    return f"mscxrt:{subject_id}:{previous_study_id}:{study_id}"


def build_example_id(pair_id: str, finding_slug: str) -> str:
    return f"{pair_id}::stage1::{finding_slug.replace(' ', '_')}"


def resolve_mimic_image_path(
    *,
    mimic_jpg_root: Path,
    subject_id: str,
    study_id: str,
    dicom_id: str,
) -> Path:
    normalized = normalize_token(dicom_id)
    if "/" in normalized:
        return mimic_jpg_root / f"{normalized}.jpg"
    subject_prefix = f"p{subject_id[:2]}"
    return mimic_jpg_root / subject_prefix / f"p{subject_id}" / f"s{study_id}" / f"{normalized}.jpg"


def load_view_positions(metadata_csv: Path, needed_dicom_ids: set[str]) -> dict[str, str]:
    view_positions: dict[str, str] = {}
    with metadata_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            dicom_id = normalize_token(row.get("dicom_id") or "")
            if dicom_id not in needed_dicom_ids:
                continue
            view_positions[dicom_id] = normalize_token(row.get("ViewPosition") or "").upper()
    return view_positions


def split_for_subject(
    subject_id: str,
    *,
    split_policy: str,
    split_seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> str:
    if split_policy == "all_test":
        return "test"
    digest = hashlib.sha1(f"{split_seed}:{subject_id}".encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    if bucket < train_ratio:
        return "train"
    if bucket < train_ratio + val_ratio:
        return "val"
    return "test"


def validate_split_ratios(*, train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise SystemExit("train/val/test ratios must sum to 1.0")


def build_dataset(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    validate_split_ratios(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    allowed_label_qualities = {normalize_quality(value) for value in args.allowed_label_qualities}
    if None in allowed_label_qualities:
        raise SystemExit("allowed label qualities must not contain empty values")

    csv_rows = load_csv_rows(args.ms_cxr_t_csv)
    needed_dicom_ids: set[str] = set()
    for row in csv_rows:
        needed_dicom_ids.add(extract_dicom_leaf(row["dicom_id"]))
        needed_dicom_ids.add(extract_dicom_leaf(row["previous_dicom_id"]))
    view_positions = load_view_positions(args.mimic_metadata_csv, needed_dicom_ids)

    query_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    summary_counters: dict[str, Counter[str]] = {
        "finding_counts": Counter(),
        "change_counts": Counter(),
        "label_quality_counts": Counter(),
        "split_counts": Counter(),
        "view_pair_counts": Counter(),
    }
    skipped = Counter()
    patients_by_split: dict[str, set[str]] = defaultdict(set)

    for csv_index, row in enumerate(csv_rows):
        subject_id = normalize_token(row["subject_id"])
        study_id = normalize_token(row["study_id"])
        previous_study_id = normalize_token(row["previous_study_id"])
        raw_dicom_id = normalize_token(row["dicom_id"])
        raw_previous_dicom_id = normalize_token(row["previous_dicom_id"])
        dicom_id = extract_dicom_leaf(raw_dicom_id)
        previous_dicom_id = extract_dicom_leaf(raw_previous_dicom_id)
        pair_id = build_pair_id(subject_id, previous_study_id, study_id)
        prior_image_path = resolve_mimic_image_path(
            mimic_jpg_root=args.mimic_jpg_root,
            subject_id=subject_id,
            study_id=previous_study_id,
            dicom_id=raw_previous_dicom_id,
        )
        current_image_path = resolve_mimic_image_path(
            mimic_jpg_root=args.mimic_jpg_root,
            subject_id=subject_id,
            study_id=study_id,
            dicom_id=raw_dicom_id,
        )
        if not prior_image_path.exists() or not current_image_path.exists():
            skipped["missing_image_path"] += 1
            continue

        prior_view_position = view_positions.get(previous_dicom_id, "")
        current_view_position = view_positions.get(dicom_id, "")
        if not prior_view_position or not current_view_position:
            skipped["missing_view_position"] += 1
        if args.same_view_only and prior_view_position and current_view_position and prior_view_position != current_view_position:
            skipped["mixed_view_pair"] += 1
            continue

        split = split_for_subject(
            subject_id,
            split_policy=args.split_policy,
            split_seed=args.split_seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
        )
        view_pair = f"{prior_view_position or 'UNK'}->{current_view_position or 'UNK'}"
        pair_rows.append(
            {
                "pair_id": pair_id,
                "source_id": pair_id,
                "patient_id": subject_id,
                "subject_id": subject_id,
                "split": split,
                "prior_study_id": previous_study_id,
                "study_id": study_id,
                "prior_dicom_id": previous_dicom_id,
                "dicom_id": dicom_id,
                "prior_image_path": str(prior_image_path),
                "current_image_path": str(current_image_path),
                "prior_view_position": prior_view_position or None,
                "current_view_position": current_view_position or None,
                "metadata": {
                    "benchmark_name": "ms_cxr_t",
                    "benchmark_task": "temporal_image_classification",
                    "split_policy": args.split_policy,
                    "csv_row_index": csv_index,
                    "raw_previous_dicom_id": raw_previous_dicom_id,
                    "raw_dicom_id": raw_dicom_id,
                },
            }
        )

        emitted_query_count = 0
        for query_index, (finding, progression_key, quality_key) in enumerate(DEFAULT_FINDINGS):
            progression = normalize_progression(row.get(progression_key) or "")
            if progression is None:
                continue
            quality = normalize_quality(row.get(quality_key) or "")
            if quality not in allowed_label_qualities:
                skipped["filtered_label_quality"] += 1
                continue

            example_id = build_example_id(pair_id, finding)
            solution = serialize_longitudinal_solution(
                [
                    {
                        "anatomy": UNSPECIFIED_ANATOMY,
                        "finding": finding,
                        "change": progression,
                    }
                ]
            )
            query_rows.append(
                {
                    "pair_id": pair_id,
                    "example_id": example_id,
                    "source_id": example_id,
                    "base_source_id": pair_id,
                    "query_id": example_id,
                    "patient_id": subject_id,
                    "subject_id": subject_id,
                    "split": split,
                    "prior_study_id": previous_study_id,
                    "study_id": study_id,
                    "prior_dicom_id": previous_dicom_id,
                    "dicom_id": dicom_id,
                    "prior_image_path": str(prior_image_path),
                    "current_image_path": str(current_image_path),
                    "problem": build_problem(finding),
                    "query": build_query_text(finding),
                    "solution": solution,
                    "label_status": f"ms_cxr_t_{quality}",
                    "weak_changes_count": 1,
                    "query_anatomy": UNSPECIFIED_ANATOMY,
                    "query_finding": finding,
                    "query_change": progression,
                    "query_index": query_index,
                    "task_mode": "stage1",
                    "curriculum_stage": 1,
                    "prior_view_position": prior_view_position or None,
                    "current_view_position": current_view_position or None,
                    "metadata": {
                        "benchmark_name": "ms_cxr_t",
                        "benchmark_task": "temporal_image_classification",
                        "split_policy": args.split_policy,
                        "base_source_id": pair_id,
                        "query_id": example_id,
                        "query_anatomy": UNSPECIFIED_ANATOMY,
                        "query_finding": finding,
                        "query_change": progression,
                        "query_index": query_index,
                        "label_quality": quality,
                        "curriculum_stage": 1,
                        "task_mode": "stage1",
                        "sft_builder": "build_ms_cxr_t_longitudinal_stage1.py",
                        "supervision_source": "benchmark_labels",
                        "csv_row_index": csv_index,
                        "raw_previous_dicom_id": raw_previous_dicom_id,
                        "raw_dicom_id": raw_dicom_id,
                    },
                }
            )
            emitted_query_count += 1
            summary_counters["finding_counts"][finding] += 1
            summary_counters["change_counts"][progression] += 1
            summary_counters["label_quality_counts"][quality] += 1
            summary_counters["split_counts"][split] += 1
            summary_counters["view_pair_counts"][view_pair] += 1
            patients_by_split[split].add(subject_id)

        if emitted_query_count == 0:
            skipped["pairs_without_kept_labels"] += 1

    query_rows.sort(key=lambda row: (row["split"], row["patient_id"], row["pair_id"], row["source_id"]))
    pair_rows.sort(key=lambda row: (row["split"], row["patient_id"], row["pair_id"]))
    summary = {
        "count": len(query_rows),
        "pair_count": len(pair_rows),
        "split_policy": args.split_policy,
        "allowed_label_qualities": sorted(value for value in allowed_label_qualities if value is not None),
        "same_view_only": bool(args.same_view_only),
        "split_counts": dict(summary_counters["split_counts"]),
        "patient_counts_by_split": {
            split: len(subject_ids) for split, subject_ids in sorted(patients_by_split.items())
        },
        "finding_counts": dict(summary_counters["finding_counts"]),
        "change_counts": dict(summary_counters["change_counts"]),
        "label_quality_counts": dict(summary_counters["label_quality_counts"]),
        "view_pair_counts": dict(summary_counters["view_pair_counts"]),
        "skipped_counts": dict(skipped),
    }
    return query_rows, pair_rows, summary


def write_readme(path: Path, *, args: argparse.Namespace, summary: dict[str, Any]) -> None:
    lines = [
        "# MS-CXR-T Longitudinal Stage-1 Dataset",
        "",
        "This artifact expands the local MS-CXR-T temporal image-classification CSV into query-conditioned Stage-1 rows compatible with the current longitudinal MedGemma train/eval scripts.",
        "",
        "Source:",
        f"- benchmark CSV: `{args.ms_cxr_t_csv}`",
        f"- MIMIC metadata CSV: `{args.mimic_metadata_csv}`",
        f"- MIMIC JPG root: `{args.mimic_jpg_root}`",
        "",
        "Policy:",
        f"- split policy: `{args.split_policy}`",
        "- `all_test` is a local evaluation-only fallback and does not claim an official MS-CXR-T split",
        "- each labeled finding becomes one query-conditioned Stage-1 example",
        "- benchmark rows use `anatomy=\"unspecified\"` because the current task is finding-level, not anatomy-grounded",
        f"- allowed label qualities: `{summary['allowed_label_qualities']}`",
        f"- same-view only: `{summary['same_view_only']}`",
        "",
        "Summary:",
        f"- query rows: `{summary['count']}`",
        f"- pair rows: `{summary['pair_count']}`",
        f"- split counts: `{summary['split_counts']}`",
        f"- patient counts by split: `{summary['patient_counts_by_split']}`",
        f"- finding counts: `{summary['finding_counts']}`",
        f"- change counts: `{summary['change_counts']}`",
        f"- label quality counts: `{summary['label_quality_counts']}`",
        f"- view pair counts: `{summary['view_pair_counts']}`",
        f"- skipped counts: `{summary['skipped_counts']}`",
        "",
        "Files:",
        "- `train.jsonl`, `val.jsonl`, `test.jsonl`, `all.jsonl`: query-conditioned Stage-1 rows",
        "- `benchmark_pairs.jsonl`: deduplicated pair manifest for overlap audits",
        "- `summary.json`: machine-readable build summary",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    query_rows, pair_rows, summary = build_dataset(args)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    split_rows = {
        split: [row for row in query_rows if row["split"] == split]
        for split in ("train", "val", "test")
    }
    for split, rows in split_rows.items():
        write_jsonl(output_dir / f"{split}.jsonl", rows)
    write_jsonl(output_dir / "all.jsonl", query_rows)
    write_jsonl(output_dir / "benchmark_pairs.jsonl", pair_rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_readme(output_dir / "README.md", args=args, summary=summary)
    print(json.dumps({"output_dir": str(output_dir), "summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
