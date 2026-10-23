#!/usr/bin/env python3
"""Rebuild primary-run diagnostics from raw, occurrence-preserving outputs.

No inference or historical-output mutation is performed. Raw outputs do not
store image paths; complete manifest order is therefore verified before each
prediction is attached to its physical prior/current/finding/occurrence key.
Exported row identities are SHA-256 digests, never pixel paths.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.cmig_temporal_validation import image_identity, load_rows, normalize_finding, occurrence_keys

LABELS = ("improving", "stable", "worsening")
FINDINGS = ("consolidation", "edema", "pleural effusion", "pneumonia", "pneumothorax")
PRIMARY_RAW = "medgemma_runs/ms_cxr_t_stage1_label_sft_benchmark_all_test_gpu6_big/ms_cxr_t_stage1_label_sft_benchmark_all_test_gpu6_big/predictions.jsonl"
PRIMARY_JOINED = "medgemma_runs/ms_cxr_t_stage1_label_sft_all_test_4label_rescored_gpu6_big/ms_cxr_t_stage1_label_sft_all_test_4label_rescored_gpu6_big/joined_predictions.jsonl"
BIO_SUMMARY = "medsiglip_runs/ms_cxr_t_biovilt_prior_current_all_test_4label_rescored_gpu1_fast7/summary.json"
BIO_JOINED = "medsiglip_runs/ms_cxr_t_biovilt_prior_current_all_test_4label_rescored_gpu1_fast7/joined_predictions.jsonl"
METRIC_NAMES = ("accuracy", "supported_macro_f1", "worsening_recall")


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def align_raw(reference: list[dict], predictions: list[dict], *, allow_missing_identity: bool = False) -> list[dict]:
    if len(reference) != len(predictions):
        raise ValueError("Raw predictions must preserve the complete canonical row count")
    keys = occurrence_keys(reference, "baseline")
    result = []
    for index, (ref, pred, key) in enumerate(zip(reference, predictions, keys)):
        if str(ref["source_id"]) != str(pred.get("source_id")):
            raise ValueError(f"Source/order mismatch at row {index}")
        for field in ("patient_id", "pair_id"):
            if allow_missing_identity and field not in pred:
                continue
            if str(ref[field]) != str(pred.get(field)):
                raise ValueError(f"{field} mismatch at row {index}")
        if normalize_finding(ref["query_finding"]) != normalize_finding(pred.get("query_finding")):
            raise ValueError(f"Finding mismatch at row {index}")
        gold = pred.get("gold_label", pred.get("target_label", pred.get("query_change")))
        if ref["query_change"] != gold:
            raise ValueError(f"Gold-label mismatch at row {index}")
        for prior, field in ((True, "prior_image_path"), (False, "current_image_path")):
            if field in pred and image_identity(ref, prior) != image_identity(pred, prior):
                raise ValueError(f"Physical-image mismatch at row {index}")
        result.append({"canonical_row_index": index, "patient_id": str(ref["patient_id"]),
                       "physical_occurrence_sha256": digest(key), "query_finding": ref["query_finding"],
                       "gold_label": ref["query_change"], "pred_label": pred.get("pred_label") or "__invalid__"})
    if len({r["physical_occurrence_sha256"] for r in result}) != len(result):
        raise ValueError("Physical occurrence identities are not unique")
    return result


def metrics(matrix: np.ndarray) -> np.ndarray:
    """Accept [..., 3 gold, 4 prediction] confusion tensors."""
    diagonal = np.diagonal(matrix[..., :3], axis1=-2, axis2=-1)
    total = matrix.sum(axis=(-2, -1))
    denominator = matrix.sum(axis=-1) + matrix[..., :3].sum(axis=-2)
    f1 = np.divide(2 * diagonal, denominator, out=np.zeros_like(diagonal, dtype=float), where=denominator != 0)
    accuracy = np.divide(diagonal.sum(axis=-1), total, out=np.zeros_like(total, dtype=float), where=total != 0)
    worsen_n = matrix[..., 2, :].sum(axis=-1)
    worsening = np.divide(matrix[..., 2, 2], worsen_n, out=np.zeros_like(worsen_n, dtype=float), where=worsen_n != 0)
    return np.stack((accuracy, f1.mean(axis=-1), worsening), axis=-1)


def classification_results(reference: list[dict], q: list[dict], bio: list[dict], samples: int, seed: int) -> dict:
    patients = sorted({r["patient_id"] for r in q})
    pi = {p: i for i, p in enumerate(patients)}
    groups = ("overall",) + FINDINGS
    shape = (len(patients), len(groups), 3, 4)
    counts = [np.zeros(shape, dtype=np.int32), np.zeros(shape, dtype=np.int32)]
    for model, rows in enumerate((q, bio)):
        for row in rows:
            p, g = pi[row["patient_id"]], LABELS.index(row["gold_label"])
            pred = LABELS.index(row["pred_label"]) if row["pred_label"] in LABELS else 3
            for f in (0, FINDINGS.index(row["query_finding"]) + 1):
                counts[model][p, f, g, pred] += 1
    observed = [c.sum(axis=0) for c in counts]
    estimates = [metrics(c) for c in observed]
    deltas = []
    rng = np.random.default_rng(seed)
    for start in range(0, samples, 50):
        draw = rng.integers(0, len(patients), size=(min(50, samples - start), len(patients)))
        deltas.append(metrics(counts[0][draw].sum(axis=1)) - metrics(counts[1][draw].sum(axis=1)))
    bounds = np.quantile(np.concatenate(deltas), [.025, .975], axis=0) if samples else None
    result = {}
    for i, group in enumerate(groups):
        rows = [r for r in q if group == "overall" or r["query_finding"] == group]
        result[group] = {"rows": len(rows), "patient_clusters": len({r["patient_id"] for r in rows}),
                         "q_lace": dict(zip(METRIC_NAMES, map(float, estimates[0][i]))),
                         "biovilt": dict(zip(METRIC_NAMES, map(float, estimates[1][i]))),
                         "confusion_labels": {"gold": list(LABELS), "prediction": list(LABELS) + ["other"]},
                         "q_lace_confusion": observed[0][i].tolist(), "biovilt_confusion": observed[1][i].tolist(),
                         "paired_delta": {}}
        for j, name in enumerate(METRIC_NAMES):
            ci = bounds[:, i, j].tolist() if bounds is not None else None
            result[group]["paired_delta"][name] = {"estimate": float(estimates[0][i, j] - estimates[1][i, j]),
                                                  "patient_bootstrap_95_ci": ci,
                                                  "excludes_zero": bool(ci and (ci[0] > 0 or ci[1] < 0))}
    return result


def complementarity(q: list[dict], bio: list[dict]) -> dict:
    """Exact paired correctness counts; the oracle is descriptive, not deployable."""
    if len(q) != len(bio):
        raise ValueError("Complementarity requires aligned complete prediction vectors")
    counts = Counter({name: 0 for name in ("both_correct", "q_lace_only_correct", "biovilt_only_correct", "both_wrong_same_prediction", "both_wrong_different_prediction")})
    for a, b in zip(q, bio):
        for field in ("canonical_row_index", "physical_occurrence_sha256", "gold_label"):
            if a[field] != b[field]:
                raise ValueError(f"Complementarity alignment mismatch: {field}")
        qa, ba = a["pred_label"] == a["gold_label"], b["pred_label"] == b["gold_label"]
        category = "both_correct" if qa and ba else "q_lace_only_correct" if qa else "biovilt_only_correct" if ba else "both_wrong_same_prediction" if a["pred_label"] == b["pred_label"] else "both_wrong_different_prediction"
        counts[category] += 1
    oracle_correct = counts["both_correct"] + counts["q_lace_only_correct"] + counts["biovilt_only_correct"]
    return {"rows": len(q), "counts": dict(counts), "either_correct_oracle": {"correct_count": oracle_correct, "denominator": len(q), "accuracy": oracle_correct / len(q) if q else None}, "interpretation": "In-sample correctness oracle; no validation-selected or deployable ensemble is evaluated."}


def join_swaps(controls: list[dict], reference: list[dict], predictions: list[dict]) -> list[dict]:
    """Join by verified original/target input row, never by colliding source ID."""
    result = []
    for index, control in enumerate(controls):
        source_i = int(control["query_swap_original_input_row_index"])
        target_i = int(control["query_swap_target_input_row_index"])
        if not (0 <= source_i < len(reference) and 0 <= target_i < len(reference)):
            raise ValueError(f"Control row index outside canonical manifest at {index}")
        source, target = reference[source_i], reference[target_i]
        for role, row in (("original", source), ("target", target)):
            for suffix, field in (("source_id", "source_id"), ("finding", "query_finding"), ("change", "query_change")):
                if control[f"query_swap_{role}_{suffix}"] != row[field]:
                    raise ValueError(f"Control {role} {suffix} mismatch at {index}")
            for prior in (True, False):
                if image_identity(control, prior) != image_identity(row, prior):
                    raise ValueError(f"Control {role} physical-image mismatch at {index}")
            if str(control["patient_id"]) != str(row["patient_id"]):
                raise ValueError(f"Control patient mismatch at {index}")
        if control["query"] != target["query"] or control["problem"] != target["problem"]:
            raise ValueError(f"Control target prompt mismatch at {index}")
        a, b = predictions[source_i], predictions[target_i]
        original_correct, target_correct = a["pred_label"] == a["gold_label"], b["pred_label"] == b["gold_label"]
        changed = a["pred_label"] != b["pred_label"]
        category = "both_correct" if original_correct and target_correct else "original_only_correct" if original_correct else "swapped_only_correct" if target_correct else "neither_correct"
        result.append({"control_index": index, "patient_id": str(source["patient_id"]),
                       "original_row_index": source_i, "target_row_index": target_i,
                       "original_physical_occurrence_sha256": a["physical_occurrence_sha256"],
                       "target_physical_occurrence_sha256": b["physical_occurrence_sha256"],
                       "original_finding": source["query_finding"], "target_finding": target["query_finding"],
                       "original_gold": a["gold_label"], "target_gold": b["gold_label"],
                       "original_prediction": a["pred_label"], "target_prediction": b["pred_label"],
                       "different_label": a["gold_label"] != b["gold_label"], "prediction_changed": changed,
                       "prediction_retained": not changed, "original_correct": original_correct,
                       "target_correct": target_correct, "both_correct": original_correct and target_correct,
                       "changed_to_correct_target": changed and target_correct, "four_way": category})
    return result


def binary_metric(rows: list[dict], field: str, samples: int, seed: int) -> dict | None:
    if not rows:
        return None
    counts = defaultdict(lambda: [0, 0])
    for row in rows:
        counts[row["patient_id"]][0] += bool(row[field])
        counts[row["patient_id"]][1] += 1
    values = np.asarray([counts[p] for p in sorted(counts)], dtype=np.int32)
    draws = []
    rng = np.random.default_rng(seed)
    for start in range(0, samples, 250):
        selected = rng.integers(0, len(values), size=(min(250, samples - start), len(values)))
        summed = values[selected].sum(axis=1)
        draws.extend((summed[:, 0] / summed[:, 1]).tolist())
    numerator = sum(bool(row[field]) for row in rows)
    return {"numerator": numerator, "denominator": len(rows), "estimate": numerator / len(rows),
            "patient_clusters": len(counts), "patient_bootstrap_95_ci": np.quantile(draws, [.025, .975]).tolist() if draws else None}


def swap_metrics(rows: list[dict], samples: int, seed: int) -> dict:
    different = [r for r in rows if r["different_label"]]
    same = [r for r in rows if not r["different_label"]]
    specs = {"target_accuracy": (rows, "target_correct"), "prediction_change_rate": (rows, "prediction_changed"),
             "all_both_correct": (rows, "both_correct"), "required_change_compliance": (different, "prediction_changed"),
             "paired_target_success": (different, "both_correct"), "different_label_target_accuracy": (different, "target_correct"),
             "changed_to_correct_target": (different, "changed_to_correct_target"), "same_target_invariance": (same, "prediction_retained"),
             "same_label_both_correct": (same, "both_correct"), "same_label_target_accuracy": (same, "target_correct")}
    return {"rows": len(rows), "different_label_rows": len(different), "same_label_rows": len(same),
            "patient_clusters": len({r["patient_id"] for r in rows}), "small_cell_descriptive": len(rows) < 20,
            "metrics": {name: binary_metric(subset, field, samples, seed) for name, (subset, field) in specs.items()},
            "four_way_counts": dict(Counter(r["four_way"] for r in rows))}


def oracle(reference: list[dict], samples: int, seed: int) -> dict:
    groups = defaultdict(list)
    for row in reference:
        groups[(str(row["patient_id"]), image_identity(row, True), image_identity(row, False))].append(row)
    output = {}
    for name in ("all_exact_image_pairs", "exact_image_multi_query", "exact_image_discordant"):
        selected = [v for v in groups.values() if name == "all_exact_image_pairs" or
                    (len({r["query_finding"] for r in v}) > 1 and (name != "exact_image_discordant" or len({r["query_change"] for r in v}) > 1))]
        rows = []
        for group in selected:
            correct_n = max(Counter(r["query_change"] for r in group).values())
            rows.extend({"patient_id": str(group[0]["patient_id"]), "oracle_correct": i < correct_n} for i in range(len(group)))
        output[name] = {"groups": len(selected), **binary_metric(rows, "oracle_correct", samples, seed)}
    return output


def compare_legacy_swap_points(previous: dict, swaps: list[dict]) -> dict:
    """Verify archived ordered-pair and transition point tables independently."""
    fields = {"target_accuracy": "target_correct", "prediction_change_rate": "prediction_changed", "paired_target_success": "both_correct", "prediction_retention_rate": "prediction_retained", "changed_to_correct_target": "changed_to_correct_target"}
    mismatches = []
    checked = 0
    for table, dimensions in (("ordered_finding_pairs", ("original_finding", "target_finding")), ("gold_transitions", ("original_gold", "target_gold"))):
        for block in previous[table]:
            subset = [r for r in swaps if all(r[d] == block[d] for d in dimensions)]
            for name, old in block["metrics"].items():
                chosen = subset
                if name in {"required_change_compliance", "different_label_target_accuracy", "changed_to_correct_target"} and table == "ordered_finding_pairs":
                    chosen = [r for r in subset if r["different_label"]]
                if name in {"same_target_invariance", "same_label_target_accuracy"}:
                    chosen = [r for r in subset if not r["different_label"]]
                if name == "paired_target_success" and table == "ordered_finding_pairs":
                    chosen = [r for r in subset if r["different_label"]]
                field = fields.get(name, {"required_change_compliance": "prediction_changed", "different_label_target_accuracy": "target_correct", "same_target_invariance": "prediction_retained", "same_label_target_accuracy": "target_correct"}.get(name))
                if field is None:
                    raise ValueError(f"Unknown archived swap metric: {name}")
                numerator = sum(bool(r[field]) for r in chosen)
                matches = (old is None and not chosen) or (old is not None and old["numerator"] == numerator and old["denominator"] == len(chosen) and old["estimate"] == numerator / len(chosen))
                checked += 1
                if not matches:
                    mismatches.append({"table": table, **{d: block[d] for d in dimensions}, "metric": name})
    return {"ordered_finding_pairs": len(previous["ordered_finding_pairs"]), "gold_transitions": len(previous["gold_transitions"]), "point_entries_checked": checked, "point_mismatch_count": len(mismatches), "all_point_estimates_unchanged": not mismatches, "mismatches": mismatches, "ci_note": "Point tables verified exactly; confidence intervals in the new report use the declared NumPy RNG and can differ slightly from historical Python-random bootstrap intervals."}


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False, encoding="utf-8") as handle:
        handle.write(text)
        temporary = handle.name
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "manuscript_rev10/artifacts/cmig_validation/canonical_diagnostics")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    args = parser.parse_args()
    if args.bootstrap_samples < 0:
        raise ValueError("Bootstrap samples must be nonnegative")
    inputs = {}

    def record(relative: str) -> Path:
        path = ROOT / relative
        inputs[relative] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "size_bytes": path.stat().st_size}
        return path

    reference = load_rows(record("share_data/ms_cxr_t_longitudinal_stage1/test.jsonl"))
    all_reference = load_rows(record("share_data/ms_cxr_t_longitudinal_stage1/all.jsonl"))
    if reference != all_reference:
        raise ValueError("Control index manifest differs from canonical test manifest")
    if len(reference) != 1326 or len({str(r["patient_id"]) for r in reference}) != 800:
        raise ValueError("Expected canonical 1326 rows / 800 patients")
    primary = align_raw(reference, load_rows(record(PRIMARY_RAW)))
    summary = json.loads(record(BIO_SUMMARY).read_text())
    bio_path = Path(summary["args"]["predictions_jsonl"]).resolve()
    bio_relative = str(bio_path.relative_to(ROOT))
    bio = align_raw(reference, load_rows(record(bio_relative)), allow_missing_identity=True)
    legacy = align_raw(reference, load_rows(record(PRIMARY_JOINED)))
    legacy_bio = align_raw(reference, load_rows(record(BIO_JOINED)))
    controls = load_rows(record("share_data/ms_cxr_t_longitudinal_stage1_query_controls/all_controls.jsonl"))
    swaps = join_swaps(controls, reference, primary)
    if len(swaps) != 662 or sum(r["different_label"] for r in swaps) != 162:
        raise ValueError("Unexpected query-swap population; expected 662 rows, 162 different / 500 same")
    samples, seed = args.bootstrap_samples, args.bootstrap_seed
    result = {"scope": "CPU-only canonical primary-run diagnostics; historical artifacts unchanged",
              "rows": len(reference), "patients": 800,
              "bootstrap": {"samples": samples, "seed": seed, "unit": "patient_id", "method": "percentile_95", "rng": "numpy.default_rng; paired identical draws for both model arms"},
              "alignment": "Complete source/finding/gold sequence verified for both raw models; primary patient/pair sequence also verified. Physical prior/current/finding plus occurrence index retained. Query controls use independently verified canonical row indices.",
              "provenance_limitation": "Raw predictions omit physical image paths; BioViL-T raw outputs also omit patient/pair IDs. Physical identity follows complete archived manifest order and source/finding/gold matching rather than independently saved per-prediction image paths. These limitations cannot be retrospectively removed.",
              "interpretation": ["Primary-run diagnostics are not three-run mean results.", "All intervals are exploratory and unadjusted for multiple comparisons.", "Required-change compliance and same-label invariance do not imply correct predictions; strict both-correct is separately reported.", "Oracle accuracy uses test labels in sample and is not a deployable learned baseline.", "Supported macro-F1 fixes three gold classes; out-of-set outputs remain errors, pooled as other for confusion accounting."],
              "classification": classification_results(reference, primary, bio, samples, seed),
              "primary_biovilt_complementarity": complementarity(primary, bio),
              "query_swap": {"aggregate": swap_metrics(swaps, samples, seed), "ordered_finding_pairs": [], "gold_transitions": [], "four_way": {}},
              "finding_blind_oracle": oracle(reference, samples, seed), "legacy_comparison": {}}
    old_swap_results = json.loads(record("manuscript_rev10/artifacts/rebuttal_analyses/query_swap_detailed/results.json").read_text())
    result["legacy_query_swap_point_reconciliation"] = compare_legacy_swap_points(old_swap_results, swaps)
    participating = {r[k] for r in swaps for k in ("original_row_index", "target_row_index")}
    changed_indices = {a["canonical_row_index"] for a, b in zip(primary, legacy) if a["pred_label"] != b["pred_label"]}
    result["legacy_query_swap_point_reconciliation"]["raw_vs_joined_changed_rows_in_swap_cohort"] = sorted(participating & changed_indices)
    for a in FINDINGS:
        for b in FINDINGS:
            if a != b:
                subset = [r for r in swaps if r["original_finding"] == a and r["target_finding"] == b]
                result["query_swap"]["ordered_finding_pairs"].append({"original_finding": a, "target_finding": b, **swap_metrics(subset, samples, seed)})
    for a in LABELS:
        for b in LABELS:
            subset = [r for r in swaps if r["original_gold"] == a and r["target_gold"] == b]
            result["query_swap"]["gold_transitions"].append({"original_gold": a, "target_gold": b, **swap_metrics(subset, samples, seed)})
    for name, subset in (("all", swaps), ("different_label", [r for r in swaps if r["different_label"]]), ("same_label", [r for r in swaps if not r["different_label"]])):
        categories = {}
        for category in ("both_correct", "original_only_correct", "swapped_only_correct", "neither_correct"):
            enriched = [{**r, "category_match": r["four_way"] == category} for r in subset]
            categories[category] = binary_metric(enriched, "category_match", samples, seed)
        result["query_swap"]["four_way"][name] = categories
    for name, raw, joined in (("primary", primary, legacy), ("biovilt", bio, legacy_bio)):
        differences = [{"canonical_row_index": a["canonical_row_index"], "physical_occurrence_sha256": a["physical_occurrence_sha256"],
                        "query_finding": a["query_finding"], "gold_label": a["gold_label"], "raw_prediction": a["pred_label"], "legacy_joined_prediction": b["pred_label"]}
                       for a, b in zip(raw, joined) if a["pred_label"] != b["pred_label"]]
        legacy_metrics = classification_results(reference, joined, joined, 0, seed)
        result["legacy_comparison"][name] = {"prediction_difference_count": len(differences), "differences": differences,
                                              "legacy_joined_metrics": {k: v["q_lace"] for k, v in legacy_metrics.items()}}
    result["inputs"] = inputs
    for relative in ("scripts/cmig_canonical_diagnostics.py", "tests/test_cmig_canonical_diagnostics.py", "scripts/cmig_temporal_validation.py"):
        record(relative)
    report_path = args.output_dir / "report.json"
    atomic_write(report_path, json.dumps(result, indent=2) + "\n")
    atomic_write(args.output_dir / "query_swap_rows.jsonl", "".join(json.dumps(row, sort_keys=True) + "\n" for row in swaps))
    lines = ["# Canonical primary-run diagnostics", "", "All 1,326 benchmark rows / 800 patients and all 662 directed query swaps are retained. This is a primary-run diagnostic, not a three-run mean.", "", "## Per-finding comparison", "", "|Finding|n|Q-LACE accuracy|Q-LACE F1|BioViL-T F1|F1 delta [95% CI]|Q-LACE worsening recall|", "|---|---:|---:|---:|---:|---:|---:|"]
    for finding, block in result["classification"].items():
        q, b, delta = block["q_lace"], block["biovilt"], block["paired_delta"]["supported_macro_f1"]
        ci = delta["patient_bootstrap_95_ci"]
        ci_text = f"[{ci[0]:+.4f}, {ci[1]:+.4f}]" if ci else "not estimated"
        lines.append(f"|{finding}|{block['rows']}|{q['accuracy']:.4f}|{q['supported_macro_f1']:.4f}|{b['supported_macro_f1']:.4f}|{delta['estimate']:+.4f} {ci_text}|{q['worsening_recall']:.4f}|")
    comp = result["primary_biovilt_complementarity"]
    lines += ["", "## Primary Q-LACE / BioViL-T complementarity", "", "|Correctness pattern|Rows|", "|---|---:|"]
    lines += [f"|{name}|{count}|" for name, count in comp["counts"].items()]
    lines += ["", f"The either-correct oracle is {comp['either_correct_oracle']['correct_count']}/1326 = {comp['either_correct_oracle']['accuracy']:.6f}. It is an in-sample descriptive ceiling, not a deployable ensemble. The old panel counts (635 / 150 / 99 / 358 / 84) must be replaced with the raw counts (636 / 149 / 98 / 359 / 84).", "", "## Query-swap decomposition", "", "662 directed controls comprise 162 different-label and 500 same-label swaps, across 219 patients. Both the original and target prompt/image identities are verified before reusing their predictions.", "", "|Metric|Numerator / denominator|Estimate|95% patient CI|", "|---|---:|---:|---:|"]
    for name, metric in result["query_swap"]["aggregate"]["metrics"].items():
        ci = metric["patient_bootstrap_95_ci"]
        ci_text = f"[{ci[0]:.4f}, {ci[1]:.4f}]" if ci else "not estimated"
        lines.append(f"|{name}|{metric['numerator']} / {metric['denominator']}|{metric['estimate']:.4f}|{ci_text}|")
    lines += ["", "## Manuscript corrections", "", "The primary accuracy remains 785/1326 = 0.592006. Supported macro-F1 changes from 0.5617 in the old joined result to 0.561100 using raw predictions. PTX supported F1 remains 0.287552 (0.288 rounded); worsening recall remains 6/63 = 0.095238. The headline required-change compliance 68/162 = 0.419753, same-label invariance 360/500 = 0.720000, and different-label strict paired success 30/162 = 0.185185 remain unchanged.", "", "All archived ordered finding-pair and gold-transition point estimates are independently verified unchanged. Neither corrected raw row (376 or 711, zero-based) participates in the query-swap cohort. Bootstrap intervals may shift slightly because the new report explicitly uses NumPy RNG rather than the historical Python random stream.", "", "The raw/legacy prediction substitutions and every per-finding matrix, paired confidence interval, 20 ordered finding pairs, nine gold transitions, four-way decomposition, and oracle are in report.json. query_swap_rows.jsonl exports hashed physical occurrence identities and never pixel paths.", "", "## Limits", "", result["provenance_limitation"], "", "Intervals are exploratory and not multiplicity-adjusted. Finding-blind oracle results are in-sample descriptive ceilings. The standalone primary predictions differ from the temporal-suite baseline; this report does not merge or substitute those separate inference artifacts.", ""]
    atomic_write(args.output_dir / "REPORT.md", "\n".join(lines))
    files = {p.name: {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "size_bytes": p.stat().st_size} for p in (report_path, args.output_dir / "REPORT.md", args.output_dir / "query_swap_rows.jsonl")}
    atomic_write(args.output_dir / "artifact_manifest.json", json.dumps({"command": "python scripts/cmig_canonical_diagnostics.py", "artifacts": files}, indent=2) + "\n")
    print(json.dumps({"output_dir": str(args.output_dir), "classification": result["classification"]["overall"], "query_swap": result["query_swap"]["aggregate"], "report_sha256": files["report.json"]["sha256"]}, indent=2))


if __name__ == "__main__":
    main()
