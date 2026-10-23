#!/usr/bin/env python3
"""Explicit-input CPU diagnostics; analysis writes aggregate JSON only.

Adapted from Q-LACE canonical/temporal diagnostic metric definitions in
Visual-RFT/scripts/cmig_canonical_diagnostics.py and cmig_temporal_validation.py.
Changes: no archived-path main, strict row-ID/physical validation, explicit local
manifests, portable temporal preparation and aggregate-only output. Apache-2.0.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import cmig_medgemma_baseline as harness
from medgemma_visual_rft.longitudinal_classification import compute_multiclass_metrics
from tools.build_query_swap_controls import build_swaps, read_manifest, reindex

LABELS = ("improving", "stable", "worsening")
NSE = "no_supported_evidence"
INVALID = "__invalid__"
CONDITIONS = ("baseline", "prior_drop", "duplicate_pair", "time_reversal", "random_prior")


def reverse(label: str) -> str | None:
    return {"improving": "worsening", "worsening": "improving", "stable": "stable", NSE: NSE}.get(label)


def checked_predictions(rows: list[dict], records: list[dict]) -> dict[str, str]:
    reference = {row["row_id"]: row for row in rows}
    output = {}
    for record in records:
        row_id = record.get("row_id")
        if row_id not in reference or row_id in output:
            raise ValueError("Unexpected or repeated prediction row ID")
        row = reference[row_id]
        for key in ("row_index", "physical_key", "query_finding", "target_label", "patient_id"):
            if record.get(key) != row[key]:
                raise ValueError(f"Prediction identity mismatch: {key}")
        prediction = record.get("pred_label")
        output[row_id] = prediction if prediction in (*LABELS, NSE) else INVALID
    if set(reference) != set(output):
        raise ValueError("Missing prediction records; partial analyses are not permitted")
    return output


def fraction(numerator: int, denominator: int) -> dict:
    return {"numerator": numerator, "denominator": denominator,
            "estimate": numerator / denominator if denominator else None}


def classification(rows: list[dict], predictions: dict[str, str]) -> dict:
    pred = [predictions[r["row_id"]] for r in rows]
    metrics = compute_multiclass_metrics([r["target_label"] for r in rows], pred, labels=LABELS)
    return {"rows": len(rows), "accuracy": metrics["accuracy"],
            "supported_macro_f1": metrics["macro_f1"],
            "prediction_counts": dict(Counter(pred)), "confusion_matrix": metrics["confusion_matrix"],
            "invalid_output_count": pred.count(INVALID), "no_evidence_output_count": pred.count(NSE)}


def query_metrics(rows: list[dict], predictions: dict[str, str],
                  swap_rows: list[dict] | None = None,
                  swap_predictions: dict[str, str] | None = None) -> dict:
    controls = build_swaps(rows)
    if swap_rows is not None:
        if len(swap_rows) != len(controls) or swap_predictions is None:
            raise ValueError("Swap manifest must preserve every directed occurrence")
        fields = ("row_id", "physical_key", "patient_id", "target_label",
                  "query_swap_original_row_id", "query_swap_target_row_id",
                  "query_swap_original_row_index", "query_swap_target_row_index")
        for expected, actual in zip(controls, swap_rows):
            if any(expected.get(key) != actual.get(key) for key in fields):
                raise ValueError("Swap manifest physical identity or occurrence lineage differs")
    lookup = {r["row_id"]: r for r in rows}
    observations = []
    for index, control in enumerate(controls):
        source = lookup[control["query_swap_original_row_id"]]
        target = lookup[control["query_swap_target_row_id"]]
        original = predictions[source["row_id"]]
        swapped = (swap_predictions[swap_rows[index]["row_id"]] if swap_rows is not None
                   else predictions[target["row_id"]])
        observations.append({"different": source["target_label"] != target["target_label"],
            "changed": original != swapped,
            "both_correct": original == source["target_label"] and swapped == target["target_label"],
            "target_correct": swapped == target["target_label"],
            "invalid_pair": INVALID in (original, swapped), "nse_pair": NSE in (original, swapped)})
    different = [o for o in observations if o["different"]]
    same = [o for o in observations if not o["different"]]
    return {"schema": "qlace-public-query-diagnostics-v1", "manifest_rows": len(rows),
        "directed_swap_occurrences": len(observations), "different_label_occurrences": len(different),
        "same_label_occurrences": len(same),
        "mode": "separately_generated_swaps" if swap_rows is not None else "same-input_target_prediction_reuse",
        "classification": classification(rows, predictions),
        "required_change_compliance": fraction(sum(o["changed"] for o in different), len(different)),
        "paired_target_success": fraction(sum(o["both_correct"] for o in different), len(different)),
        "same_label_invariance": fraction(sum(not o["changed"] for o in same), len(same)),
        "same_label_paired_success": fraction(sum(o["both_correct"] for o in same), len(same)),
        "target_accuracy": fraction(sum(o["target_correct"] for o in observations), len(observations)),
        "invalid_prediction_pair_count": sum(o["invalid_pair"] for o in observations),
        "no_evidence_prediction_pair_count": sum(o["nse_pair"] for o in observations)}


def prepare_temporal(rows: list[dict], blank_prior: Path, seed: int) -> dict[str, list[dict]]:
    # Donor choice is deterministic and restricted to a different patient.
    result = {name: [] for name in CONDITIONS}
    for index, row in enumerate(rows):
        donors = [r for r in rows if r["patient_id"] != row["patient_id"]]
        if not donors:
            raise ValueError("Random-prior control requires another patient; no self fallback")
        donor = donors[int(harness.digest_json([seed, row["row_id"]])[:16], 16) % len(donors)]
        for condition in CONDITIONS:
            modified = {**row, "control_name": condition,
                        "temporal_anchor_row_id": row["row_id"], "temporal_anchor_row_index": index}
            if condition == "prior_drop":
                modified["prior_image_path"] = str(blank_prior.resolve())
            elif condition == "duplicate_pair":
                modified["prior_image_path"] = row["current_image_path"]
            elif condition == "time_reversal":
                modified["prior_image_path"], modified["current_image_path"] = row["current_image_path"], row["prior_image_path"]
                modified["target_label"] = reverse(row["target_label"])
            elif condition == "random_prior":
                modified["prior_image_path"] = donor["prior_image_path"]
            result[condition].append(reindex(modified, index))
    return result


def temporal_metrics(manifests: dict[str, list[dict]], predictions: dict[str, dict[str, str]]) -> dict:
    if set(manifests) != set(CONDITIONS) or set(predictions) != set(CONDITIONS):
        raise ValueError("Supply exactly all five temporal conditions")
    baseline = manifests["baseline"]
    if not baseline or any(r.get("temporal_anchor_row_id") != r["row_id"] for r in baseline):
        raise ValueError("Suite baseline must retain original occurrence anchors")
    for condition in CONDITIONS:
        group = manifests[condition]
        if len(group) != len(baseline):
            raise ValueError("Every temporal condition must preserve all baseline occurrences")
        for index, (base, row) in enumerate(zip(baseline, group)):
            if (row.get("temporal_anchor_row_id") != base.get("temporal_anchor_row_id") or
                row.get("temporal_anchor_row_index") != index or
                row["patient_id"] != base["patient_id"] or row["query_finding"] != base["query_finding"]):
                raise ValueError("Temporal occurrence lineage mismatch")
            expected_gold = reverse(base["target_label"]) if condition == "time_reversal" else base["target_label"]
            if row["target_label"] != expected_gold:
                raise ValueError("Temporal target-label transformation mismatch")
            if condition == "time_reversal":
                if row["physical_key"][:2] != list(reversed(base["physical_key"][:2])):
                    raise ValueError("Reversal must exchange actual image inputs")
            elif condition == "duplicate_pair":
                if row["physical_key"][:2] != [base["current_image_path"]] * 2:
                    raise ValueError("Duplicate control must use current image twice")
            elif row["current_image_path"] != base["current_image_path"]:
                raise ValueError("Temporal current-image mismatch")
            if condition == "random_prior" and not any(
                donor["patient_id"] != base["patient_id"] and
                donor["prior_image_path"] == row["prior_image_path"] for donor in baseline
            ):
                raise ValueError("Random prior must come from a supplied different patient")
    if len({r["prior_image_path"] for r in manifests["prior_drop"]}) != 1:
        raise ValueError("Prior-drop condition must use one shared blank asset")
    condition_scores = {name: classification(manifests[name], predictions[name])
                        for name in CONDITIONS if name != "duplicate_pair"}
    base_pred = [predictions["baseline"][r["row_id"]] for r in baseline]
    rev_pred = [predictions["time_reversal"][r["row_id"]] for r in manifests["time_reversal"]]
    dup_pred = [predictions["duplicate_pair"][r["row_id"]] for r in manifests["duplicate_pair"]]
    consistent = [reverse(a) is not None and reverse(a) == b for a, b in zip(base_pred, rev_pred)]
    strict = [a in LABELS and reverse(a) == b for a, b in zip(base_pred, rev_pred)]
    n = len(baseline)
    return {"schema": "qlace-public-temporal-diagnostics-v1", "occurrences_per_condition": n,
        "baseline_policy": "Deltas use only this temporal suite's supplied baseline",
        "by_condition": condition_scores,
        "prior_drop_delta_supported_macro_f1": condition_scores["prior_drop"]["supported_macro_f1"] - condition_scores["baseline"]["supported_macro_f1"],
        "random_prior_delta_supported_macro_f1": condition_scores["random_prior"]["supported_macro_f1"] - condition_scores["baseline"]["supported_macro_f1"],
        "temporal_reversal_consistency": fraction(sum(consistent), n),
        "supported_only_temporal_reversal_consistency": fraction(sum(strict), n),
        "baseline_and_reversal_paired_success": fraction(sum(a == row["target_label"] and b == reverse(row["target_label"]) for row, a, b in zip(baseline, base_pred, rev_pred)), n),
        "duplicate_stable_output_rate": fraction(dup_pred.count("stable"), n),
        "duplicate_nonstable_output_rate": fraction(n - dup_pred.count("stable"), n),
        "duplicate_change_label_output_rate": fraction(sum(p in ("improving", "worsening") for p in dup_pred), n),
        "duplicate_prediction_counts": dict(Counter(dup_pred))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    query = commands.add_parser("query")
    query.add_argument("--manifest", type=Path, required=True)
    query.add_argument("--predictions", type=Path, required=True)
    query.add_argument("--swap-manifest", type=Path)
    query.add_argument("--swap-predictions", type=Path)
    query.add_argument("--output", type=Path, required=True)
    prepare = commands.add_parser("prepare-temporal")
    prepare.add_argument("--manifest", type=Path, required=True)
    prepare.add_argument("--blank-prior-image", type=Path, required=True)
    prepare.add_argument("--seed", type=int, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    temporal = commands.add_parser("temporal")
    temporal.add_argument("--suite-dir", type=Path, required=True)
    temporal.add_argument("--predictions", type=Path, required=True,
                          help="Directory containing CONDITION/predictions.jsonl for all five conditions")
    temporal.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare-temporal":
        if not args.blank_prior_image.is_file():
            parser.error("Supply an existing local blank image")
        from PIL import Image
        with Image.open(args.blank_prior_image) as blank:
            if any(extrema != (0, 0) for extrema in blank.convert("RGB").getextrema()):
                parser.error("The supplied prior-drop asset must be uniformly black")
        groups = prepare_temporal(read_manifest(args.manifest), args.blank_prior_image, args.seed)
        for name, rows in groups.items():
            destination = args.output_dir / f"{name}.jsonl"
            if destination.resolve() == args.manifest.resolve():
                parser.error("Output must not overwrite the input manifest")
            harness.write_jsonl(destination, rows)
        print("Created five local temporal manifests")
        return
    if args.command == "query":
        if bool(args.swap_manifest) != bool(args.swap_predictions):
            parser.error("Supply both swap manifest and predictions, or neither")
        rows = read_manifest(args.manifest)
        predictions = checked_predictions(rows, harness.load_jsonl(args.predictions))
        swap_rows = read_manifest(args.swap_manifest) if args.swap_manifest else None
        swap_predictions = checked_predictions(swap_rows, harness.load_jsonl(args.swap_predictions)) if swap_rows is not None else None
        result = query_metrics(rows, predictions, swap_rows, swap_predictions)
        inputs = [args.manifest, args.predictions, args.swap_manifest, args.swap_predictions]
    else:
        manifests = {name: read_manifest(args.suite_dir / f"{name}.jsonl") for name in CONDITIONS}
        predictions = {name: checked_predictions(rows, harness.load_jsonl(args.predictions / name / "predictions.jsonl"))
                       for name, rows in manifests.items()}
        result = temporal_metrics(manifests, predictions)
        inputs = [args.suite_dir / f"{name}.jsonl" for name in CONDITIONS] + [args.predictions / name / "predictions.jsonl" for name in CONDITIONS]
    if any(path is not None and path.resolve() == args.output.resolve() for path in inputs):
        parser.error("Aggregate output must not overwrite an input")
    harness.write_json(args.output, result)


if __name__ == "__main__":
    main()
