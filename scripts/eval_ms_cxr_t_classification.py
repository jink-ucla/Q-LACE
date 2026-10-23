#!/usr/bin/env python3
"""Evaluate benchmark-facing Stage-1 classification predictions on MS-CXR-T rows."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from medgemma_visual_rft.longitudinal_classification import (
    DEFAULT_UNSUPPORTED_LABEL,
    bootstrap_classification_confidence_intervals,
    summarize_classification_records,
)
from medgemma_visual_rft.longitudinal_projection import normalize_ms_cxr_t_finding, project_changes_to_ms_cxr_t_progression


DEFAULT_REFERENCE_JSONL = REPO_ROOT / "share_data" / "ms_cxr_t_longitudinal_stage1" / "test.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "medgemma_runs" / "eval_ms_cxr_t_classification"
DEFAULT_LABELS = ("improving", "stable", "worsening")
CONTROL_METADATA_KEYS = (
    "control_name",
    "control_group_id",
    "control_pair_id",
    "temporal_control_group_id",
    "paired_source_id",
    "partner_source_id",
    "control_partner_source_id",
    "baseline_source_id",
    "time_reversal_source_id",
)
TRC_BASELINE_NAME = "baseline"
TRC_REVERSAL_NAME = "time_reversal"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-jsonl", type=Path, default=DEFAULT_REFERENCE_JSONL)
    parser.add_argument("--predictions-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--prediction-key", default="pred_label")
    parser.add_argument(
        "--labels",
        nargs="+",
        default=list(DEFAULT_LABELS),
        help="Closed-set classification labels to score.",
    )
    parser.add_argument(
        "--null-label",
        default="",
        help="Optional label to assign when a generative prediction projects to null for the queried finding.",
    )
    parser.add_argument(
        "--score-only-present",
        action="store_true",
        help="Restrict scoring to rows with a matched prediction. Useful for smoke runs.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=0,
        help="Enable patient-level bootstrap CIs for accuracy and macro-F1 when > 0.",
    )
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument(
        "--bootstrap-confidence-level",
        type=float,
        default=0.95,
        help="Confidence level for patient-level bootstrap intervals.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise SystemExit(f"Expected object on line {line_number} of {path}")
            rows.append(payload)
    return rows


def clean_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def canonical_control_name(value: Any) -> str | None:
    cleaned = clean_optional_str(value)
    if cleaned is None:
        return None
    return cleaned.casefold().replace("-", "_").replace(" ", "_")


def extract_control_metadata(reference_row: dict[str, Any]) -> dict[str, Any]:
    extracted: dict[str, Any] = {}
    candidate_sources: list[dict[str, Any]] = [reference_row]
    metadata = reference_row.get("metadata")
    if isinstance(metadata, dict):
        candidate_sources.append(metadata)
    for source in candidate_sources:
        for key, value in source.items():
            if key == "metadata" or value is None:
                continue
            if key in CONTROL_METADATA_KEYS or key.startswith("control_"):
                extracted.setdefault(key, value)
    return extracted


def pred_changes_to_label(
    pred_changes: list[Any],
    *,
    query_finding: str,
    null_label: str | None,
) -> str | None:
    normalized_query_finding = normalize_ms_cxr_t_finding(query_finding)
    if normalized_query_finding is None:
        return None
    changes: list[dict[str, str]] = []
    for item in pred_changes:
        if isinstance(item, dict):
            anatomy = str(item.get("anatomy") or "")
            finding = str(item.get("finding") or "")
            change = str(item.get("change") or "")
        elif isinstance(item, list) and len(item) == 3:
            anatomy, finding, change = (str(value) for value in item)
        else:
            continue
        changes.append({"anatomy": anatomy, "finding": finding, "change": change})
    projected = project_changes_to_ms_cxr_t_progression(changes)
    label = projected.get(normalized_query_finding)
    if label is None:
        return null_label
    return label


def canonicalize_prediction(
    prediction_row: dict[str, Any],
    *,
    prediction_key: str,
    query_finding: str,
    null_label: str | None,
) -> str:
    direct = prediction_row.get(prediction_key)
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    fallback = prediction_row.get("pred_label")
    if isinstance(fallback, str) and fallback.strip():
        return fallback.strip()
    if isinstance(prediction_row.get("pred_changes"), list):
        label = pred_changes_to_label(
            prediction_row["pred_changes"],
            query_finding=query_finding,
            null_label=null_label,
        )
        if label is not None:
            return label
    if isinstance(prediction_row.get("completion_label"), str) and prediction_row["completion_label"].strip():
        return prediction_row["completion_label"].strip()
    return "__invalid__"


def reverse_progression_label(label: str) -> str | None:
    return {
        "improving": "worsening",
        "stable": "stable",
        "worsening": "improving",
        "no_supported_evidence": "no_supported_evidence",
    }.get(label)


def build_joined_records(
    references: list[dict[str, Any]],
    prediction_by_source: dict[str, dict[str, Any]],
    *,
    prediction_key: str,
    null_label: str | None,
    score_only_present: bool,
) -> tuple[list[dict[str, Any]], int]:
    joined_records: list[dict[str, Any]] = []
    missing_predictions = 0
    for reference in references:
        source_id = str(reference["source_id"])
        prediction_row = prediction_by_source.get(source_id)
        if prediction_row is None:
            missing_predictions += 1
            if score_only_present:
                continue
            pred_label = "__missing__"
        else:
            pred_label = canonicalize_prediction(
                prediction_row,
                prediction_key=prediction_key,
                query_finding=str(reference["query_finding"]),
                null_label=null_label,
            )
        joined_records.append(
            {
                "source_id": source_id,
                "pair_id": str(reference["pair_id"]),
                "patient_id": str(reference.get("patient_id") or reference.get("subject_id") or ""),
                "split": str(reference["split"]),
                "query_finding": str(reference["query_finding"]),
                "gold_label": str(reference["query_change"]),
                "pred_label": pred_label,
            }
        )
        control_metadata = extract_control_metadata(reference)
        if control_metadata:
            joined_records[-1]["control_metadata"] = control_metadata
            for key, value in control_metadata.items():
                joined_records[-1].setdefault(key, value)
    return joined_records, missing_predictions


def summarize_by_control_name(
    records: list[dict[str, Any]],
    *,
    labels: list[str] | tuple[str, ...],
    unsupported_label: str = DEFAULT_UNSUPPORTED_LABEL,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        control_name = canonical_control_name(record.get("control_name"))
        if control_name is None:
            continue
        grouped[control_name].append(record)
    return {
        control_name: summarize_classification_records(
            control_records,
            labels=labels,
            unsupported_label=unsupported_label,
        )
        for control_name, control_records in sorted(grouped.items())
    }


def compute_temporal_reversal_consistency(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [
        record for record in records if canonical_control_name(record.get("control_name")) in {TRC_BASELINE_NAME, TRC_REVERSAL_NAME}
    ]
    if not candidates:
        return None

    by_source_id = {str(record["source_id"]): record for record in candidates}
    paired_source_keys: set[tuple[str, str]] = set()
    pairs: list[tuple[dict[str, Any], dict[str, Any], str]] = []

    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for record in candidates:
        control_name = canonical_control_name(record.get("control_name"))
        group_id = clean_optional_str(
            record.get("control_group_id")
            or record.get("control_pair_id")
            or record.get("temporal_control_group_id")
        )
        if control_name is None or group_id is None:
            continue
        grouped[group_id][control_name].append(record)

    for grouped_records in grouped.values():
        baselines = grouped_records.get(TRC_BASELINE_NAME, [])
        reversals = grouped_records.get(TRC_REVERSAL_NAME, [])
        if len(baselines) != 1 or len(reversals) != 1:
            continue
        baseline = baselines[0]
        reversal = reversals[0]
        pair_key = tuple(sorted((str(baseline["source_id"]), str(reversal["source_id"]))))
        if pair_key in paired_source_keys:
            continue
        paired_source_keys.add(pair_key)
        pairs.append((baseline, reversal, "group_id"))

    for record in candidates:
        source_id = str(record["source_id"])
        if any(source_id in pair_key for pair_key in paired_source_keys):
            continue
        control_name = canonical_control_name(record.get("control_name"))
        if control_name == TRC_BASELINE_NAME:
            partner_id = clean_optional_str(
                record.get("time_reversal_source_id")
                or record.get("paired_source_id")
                or record.get("partner_source_id")
                or record.get("control_partner_source_id")
            )
            expected_partner_name = TRC_REVERSAL_NAME
        elif control_name == TRC_REVERSAL_NAME:
            partner_id = clean_optional_str(
                record.get("baseline_source_id")
                or record.get("paired_source_id")
                or record.get("partner_source_id")
                or record.get("control_partner_source_id")
            )
            expected_partner_name = TRC_BASELINE_NAME
        else:
            continue
        if partner_id is None or partner_id not in by_source_id:
            continue
        partner = by_source_id[partner_id]
        if canonical_control_name(partner.get("control_name")) != expected_partner_name:
            continue
        baseline = record if control_name == TRC_BASELINE_NAME else partner
        reversal = partner if control_name == TRC_BASELINE_NAME else record
        pair_key = tuple(sorted((str(baseline["source_id"]), str(reversal["source_id"]))))
        if pair_key in paired_source_keys:
            continue
        paired_source_keys.add(pair_key)
        pairs.append((baseline, reversal, "partner_id"))

    if not pairs:
        return None

    consistent_count = 0
    by_finding_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for baseline, reversal, _ in pairs:
        expected_reversal = reverse_progression_label(str(baseline["pred_label"]))
        consistent = expected_reversal is not None and str(reversal["pred_label"]) == expected_reversal
        if consistent:
            consistent_count += 1
        finding = str(baseline.get("query_finding") or reversal.get("query_finding") or "unknown")
        by_finding_counts[finding]["count"] += 1
        if consistent:
            by_finding_counts[finding]["consistent_count"] += 1

    return {
        "count": len(pairs),
        "consistent_count": consistent_count,
        "rate": float(consistent_count / len(pairs)),
        "pairing_strategy_counts": dict(Counter(strategy for _, _, strategy in pairs)),
        "unpaired_control_record_count": len(candidates) - (2 * len(pairs)),
        "by_finding": {
            finding: {
                "count": int(counts["count"]),
                "consistent_count": int(counts["consistent_count"]),
                "rate": float(counts["consistent_count"] / counts["count"]) if counts["count"] else 0.0,
            }
            for finding, counts in sorted(by_finding_counts.items())
        },
    }


def build_classification_summary(
    joined_records: list[dict[str, Any]],
    *,
    labels: list[str] | tuple[str, ...],
    unsupported_label: str = DEFAULT_UNSUPPORTED_LABEL,
    bootstrap_samples: int = 0,
    bootstrap_seed: int = 42,
    bootstrap_confidence_level: float = 0.95,
) -> dict[str, Any]:
    summary = summarize_classification_records(
        joined_records,
        labels=labels,
        unsupported_label=unsupported_label,
    )
    by_control_name = summarize_by_control_name(
        joined_records,
        labels=labels,
        unsupported_label=unsupported_label,
    )
    if by_control_name:
        summary["by_control_name"] = by_control_name
        summary["by_control"] = by_control_name
    temporal_reversal_consistency = compute_temporal_reversal_consistency(joined_records)
    if temporal_reversal_consistency is not None:
        summary["temporal_reversal_consistency"] = temporal_reversal_consistency
    if bootstrap_samples > 0:
        summary["patient_bootstrap_ci"] = bootstrap_classification_confidence_intervals(
            joined_records,
            labels=labels,
            unsupported_label=unsupported_label,
            unit_key="patient_id",
            num_samples=bootstrap_samples,
            confidence_level=bootstrap_confidence_level,
            seed=bootstrap_seed,
        )
    return summary


def main() -> int:
    args = parse_args()
    references = load_jsonl(args.reference_jsonl)
    predictions = load_jsonl(args.predictions_jsonl)
    prediction_by_source = {
        str(row.get("source_id") or row.get("query_id") or row.get("example_id")): row for row in predictions
    }

    null_label = args.null_label.strip() or None
    joined_records, missing_predictions = build_joined_records(
        references,
        prediction_by_source,
        prediction_key=args.prediction_key,
        null_label=null_label,
        score_only_present=args.score_only_present,
    )

    requested_labels = list(args.labels)
    scoring_labels = list(requested_labels)
    if null_label is not None and null_label not in scoring_labels:
        scoring_labels.append(null_label)

    summary = build_classification_summary(
        joined_records,
        labels=scoring_labels,
        unsupported_label=null_label or "no_supported_evidence",
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_confidence_level=args.bootstrap_confidence_level,
    )
    summary["missing_prediction_count"] = missing_predictions
    summary["args"] = {
        "reference_jsonl": str(args.reference_jsonl),
        "predictions_jsonl": str(args.predictions_jsonl),
        "prediction_key": args.prediction_key,
        "requested_labels": requested_labels,
        "labels": scoring_labels,
        "null_label": null_label,
        "score_only_present": args.score_only_present,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "bootstrap_confidence_level": args.bootstrap_confidence_level,
    }

    run_name = args.run_name or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "joined_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for record in joined_records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
