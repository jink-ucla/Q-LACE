#!/usr/bin/env python3
"""Build a benchmark-aligned Stage-1 weak training set from longitudinal weak tuples."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from medgemma_visual_rft.longitudinal_projection import (
    ProjectedMsCxrTChange,
    select_representative_ms_cxr_t_changes,
)
from medgemma_visual_rft.longitudinal_schema import serialize_longitudinal_solution


DEFAULT_INPUT_JSONL = (
    REPO_ROOT / "share_data" / "mimic_cxr_longitudinal_weak_tuples_preview_ms_cxr_t_excluded" / "weak_tuples.jsonl"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "share_data" / "mimic_cxr_longitudinal_stage1_benchmark"
DEFAULT_INCLUDE_STATUSES = ("weak_candidate",)
UNSPECIFIED_ANATOMY = "unspecified"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT_JSONL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--task-mode",
        choices=("stage1",),
        default="stage1",
        help="Reserved for wrapper compatibility; this builder only emits Stage-1 rows.",
    )
    parser.add_argument(
        "--same-view-only",
        dest="same_view_only",
        action="store_true",
        help="Require prior/current view positions to match exactly.",
    )
    parser.add_argument(
        "--no-same-view-only",
        dest="same_view_only",
        action="store_false",
        help="Allow mixed-view pairs.",
    )
    parser.set_defaults(same_view_only=True)
    parser.add_argument(
        "--include-statuses",
        nargs="+",
        default=list(DEFAULT_INCLUDE_STATUSES),
        help="Weak label statuses to keep.",
    )
    parser.add_argument(
        "--split-policy",
        choices=("patient_hash",),
        default="patient_hash",
        help="This builder uses deterministic patient-hash splits so it scales beyond the preview.",
    )
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--max-rows", type=int, default=None)
    return parser.parse_args()


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


def validate_split_ratios(*, train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise SystemExit("train/val/test ratios must sum to 1.0")


def normalize_view_position(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().upper()


def is_same_view_pair(row: dict[str, Any]) -> bool:
    prior_view = normalize_view_position(row.get("prior_view_position"))
    current_view = normalize_view_position(row.get("current_view_position"))
    return bool(prior_view) and prior_view == current_view


def split_for_subject(
    subject_id: str,
    *,
    split_seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> str:
    validate_split_ratios(train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio)
    digest = hashlib.sha1(f"{split_seed}:{subject_id}".encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    if bucket < train_ratio:
        return "train"
    if bucket < train_ratio + val_ratio:
        return "val"
    return "test"


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


def build_example_id(pair_id: str, benchmark_finding: str) -> str:
    return f"{pair_id}::stage1_benchmark::{benchmark_finding}"


def build_stage1_row(
    raw_row: dict[str, Any],
    *,
    split: str,
    query_index: int,
    projected_change: ProjectedMsCxrTChange,
    same_view_only: bool,
) -> dict[str, Any]:
    example_id = build_example_id(str(raw_row["pair_id"]), projected_change.benchmark_finding)
    solution = serialize_longitudinal_solution(
        [
            {
                "anatomy": UNSPECIFIED_ANATOMY,
                "finding": projected_change.display_finding,
                "change": projected_change.change,
            }
        ]
    )
    representative = projected_change.representative
    metadata = raw_row.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    return {
        "pair_id": raw_row["pair_id"],
        "example_id": example_id,
        "source_id": example_id,
        "base_source_id": raw_row.get("source_id") or raw_row["pair_id"],
        "query_id": example_id,
        "patient_id": str(raw_row["patient_id"]),
        "split": split,
        "prior_image_path": raw_row["prior_image_path"],
        "current_image_path": raw_row["current_image_path"],
        "problem": build_problem(projected_change.display_finding),
        "query": build_query_text(projected_change.display_finding),
        "solution": solution,
        "label_status": raw_row["weak"]["label_status"],
        "weak_changes_count": 1,
        "query_anatomy": UNSPECIFIED_ANATOMY,
        "query_finding": projected_change.display_finding,
        "query_change": projected_change.change,
        "query_index": query_index,
        "task_mode": "stage1",
        "curriculum_stage": 1,
        "prior_view_position": raw_row.get("prior_view_position"),
        "current_view_position": raw_row.get("current_view_position"),
        "prior_report_sections": raw_row.get("prior_report_sections") or {},
        "current_report_sections": raw_row.get("current_report_sections") or {},
        "metadata": {
            **metadata,
            "task_mode": "stage1",
            "curriculum_stage": 1,
            "same_view_only": same_view_only,
            "base_source_id": raw_row.get("source_id") or raw_row["pair_id"],
            "query_id": example_id,
            "query_anatomy": UNSPECIFIED_ANATOMY,
            "query_finding": projected_change.display_finding,
            "query_change": projected_change.change,
            "query_index": query_index,
            "projected_benchmark_finding": projected_change.benchmark_finding,
            "representative_anatomy": representative.anatomy,
            "representative_finding": representative.finding,
            "representative_change": representative.change,
            "representative_confidence": projected_change.representative_confidence,
            "representative_evidence": projected_change.representative_evidence,
            "representative_source": projected_change.representative_source,
            "sft_builder": "build_mimic_cxr_longitudinal_stage1_benchmark.py",
            "task_family": "longitudinal_cxr_change_rft_stage1_benchmark",
            "supervision_source": "weak_tuples",
        },
    }


def build_summary(
    rows: list[dict[str, Any]],
    *,
    input_row_count: int,
    same_view_only: bool,
    split_policy: str,
    included_statuses: list[str],
    skipped_counts: Counter[str],
) -> dict[str, Any]:
    split_counts = Counter(row["split"] for row in rows)
    label_status_counts = Counter(row["label_status"] for row in rows)
    by_split_and_status: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    patients_by_split: dict[str, set[str]] = defaultdict(set)
    query_change_counts = Counter()
    query_finding_counts = Counter()
    representative_finding_counts = Counter()
    representative_anatomy_counts = Counter()
    view_pair_counts = Counter()

    for row in rows:
        by_split_and_status[row["split"]][row["label_status"]] += 1
        patients_by_split[row["split"]].add(row["patient_id"])
        query_change_counts[row["query_change"]] += 1
        query_finding_counts[row["query_finding"]] += 1
        metadata = row["metadata"]
        representative_finding_counts[str(metadata["representative_finding"])] += 1
        representative_anatomy_counts[str(metadata["representative_anatomy"])] += 1
        prior_view = normalize_view_position(row.get("prior_view_position")) or "UNK"
        current_view = normalize_view_position(row.get("current_view_position")) or "UNK"
        view_pair_counts[f"{prior_view}->{current_view}"] += 1

    return {
        "task_mode": "stage1",
        "count": len(rows),
        "input_row_count": input_row_count,
        "pair_count": len({str(row["pair_id"]) for row in rows}),
        "same_view_only": same_view_only,
        "split_policy": split_policy,
        "include_statuses": included_statuses,
        "split_counts": dict(split_counts),
        "label_status_counts": dict(label_status_counts),
        "split_label_status_counts": {
            split: dict(counts) for split, counts in sorted(by_split_and_status.items())
        },
        "patient_counts_by_split": {
            split: len(patients) for split, patients in sorted(patients_by_split.items())
        },
        "query_change_counts": dict(query_change_counts),
        "query_finding_counts": dict(query_finding_counts),
        "representative_finding_counts": dict(representative_finding_counts),
        "representative_anatomy_counts": dict(representative_anatomy_counts),
        "view_pair_counts": dict(view_pair_counts),
        "skipped_counts": dict(skipped_counts),
    }


def write_readme(
    path: Path,
    *,
    input_jsonl: Path,
    summary: dict[str, Any],
) -> None:
    lines = [
        "# MIMIC-CXR Longitudinal Stage-1 Benchmark-Aligned Weak Dataset",
        "",
        "This artifact projects weak longitudinal tuples onto the 5 MS-CXR-T finding heads so the weak-train path matches the query-conditioned benchmark task.",
        "",
        "Source:",
        f"- weak tuple input: `{input_jsonl}`",
        "",
        "Policy:",
        "- Stage-1 only",
        "- query is benchmark-like and finding-level",
        '- solution stays on the tuple bridge with `{"changes":[...]}`',
        f"- query anatomy is fixed to `{UNSPECIFIED_ANATOMY}` to match the benchmark-facing Stage-1 task",
        "- at most one query row is emitted per pair per benchmark finding",
        "- representative weak tuples are chosen deterministically when multiple tuples map to the same benchmark head",
        f"- same-view only: `{summary['same_view_only']}`",
        f"- split policy: `{summary['split_policy']}`",
        f"- kept label statuses: `{summary['include_statuses']}`",
        "",
        "Summary:",
        f"- rows: `{summary['count']}`",
        f"- input rows considered: `{summary['input_row_count']}`",
        f"- pair count: `{summary['pair_count']}`",
        f"- split counts: `{summary['split_counts']}`",
        f"- patient counts by split: `{summary['patient_counts_by_split']}`",
        f"- label status counts: `{summary['label_status_counts']}`",
        f"- query change counts: `{summary['query_change_counts']}`",
        f"- query finding counts: `{summary['query_finding_counts']}`",
        f"- representative finding counts: `{summary['representative_finding_counts']}`",
        f"- representative anatomy counts: `{summary['representative_anatomy_counts']}`",
        f"- view pair counts: `{summary['view_pair_counts']}`",
        f"- skipped counts: `{summary['skipped_counts']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_dataset(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = load_jsonl(args.input_jsonl)
    if args.max_rows is not None:
        rows = rows[: args.max_rows]
    included_statuses = list(args.include_statuses or DEFAULT_INCLUDE_STATUSES)
    kept_statuses = set(included_statuses)
    skipped = Counter()
    output_rows: list[dict[str, Any]] = []

    for raw_row in rows:
        weak = raw_row.get("weak") or {}
        label_status = weak.get("label_status")
        if label_status not in kept_statuses:
            skipped["filtered_label_status"] += 1
            continue
        if args.same_view_only and not is_same_view_pair(raw_row):
            skipped["mixed_view_pair"] += 1
            continue
        projected_changes = select_representative_ms_cxr_t_changes(weak.get("changes") or [])
        if not projected_changes:
            skipped["pairs_without_benchmark_projection"] += 1
            continue

        split = split_for_subject(
            str(raw_row["patient_id"]),
            split_seed=args.split_seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
        )
        for query_index, projected_change in enumerate(projected_changes):
            output_rows.append(
                build_stage1_row(
                    raw_row,
                    split=split,
                    query_index=query_index,
                    projected_change=projected_change,
                    same_view_only=bool(args.same_view_only),
                )
            )

    output_rows.sort(key=lambda row: (row["split"], row["patient_id"], row["pair_id"], row["source_id"]))
    summary = build_summary(
        output_rows,
        input_row_count=len(rows),
        same_view_only=bool(args.same_view_only),
        split_policy=args.split_policy,
        included_statuses=included_statuses,
        skipped_counts=skipped,
    )
    return output_rows, summary


def main() -> int:
    args = parse_args()
    output_rows, summary = build_dataset(args)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        write_jsonl(output_dir / f"{split}.jsonl", [row for row in output_rows if row["split"] == split])
    write_jsonl(output_dir / "all.jsonl", output_rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_readme(output_dir / "README.md", input_jsonl=args.input_jsonl, summary=summary)
    print(json.dumps({"output_dir": str(output_dir), "summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
