# Portable supplementary copy: default paths/mount aliases sanitized; original hash is in FILE_MANIFEST.json.
#!/usr/bin/env python3
"""Audit archived temporal controls and synthetic abstention counts without inference.

The historical evaluator builds a last-write-wins source_id dictionary. This
script instead validates the archived manifest's complete order and assigns
each prediction a physical image key plus an occurrence index. It retains
condition-specific image changes and all canonical benchmark rows.
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from medgemma_visual_rft.longitudinal_classification import compute_multiclass_metrics
from scripts.eval_ms_cxr_t_classification import compute_temporal_reversal_consistency

SUPPORTED = ("improving", "stable", "worsening")
NSE = "no_supported_evidence"
CONDITIONS = ("baseline", "prior_drop", "duplicate_pair", "time_reversal", "random_prior")
TEMPORAL_SOURCES = {
    "MedSigLIP current-only": "medsiglip_runs/medsiglip_stage1_current_only_full_rev4_gpu1_controls/eval_predictions.jsonl",
    "MedSigLIP prior/current": "medsiglip_runs/medsiglip_stage1_prior_current_full_rev4_gpu1_controls/eval_predictions.jsonl",
    "BioViL-T current-only": "medsiglip_runs/biovilt_stage1_current_only_controls_rev3_gpu2/eval_predictions.jsonl",
    "BioViL-T prior/current": "medsiglip_runs/biovilt_stage1_prior_current_controls_rev3_gpu1/eval_predictions.jsonl",
    "DenseNet current-only": "medsiglip_runs/torchxrayvision_densenet_stage1_current_only_full_rev8_controls_gpu4_fast/eval_predictions.jsonl",
    "DenseNet prior/current": "medsiglip_runs/torchxrayvision_densenet_stage1_prior_current_full_rev8_controls_gpu4_fast/eval_predictions.jsonl",
    "MedGemma historical prompt-only": "medgemma_runs/ms_cxr_t_stage1_promptonly_controls_gpu0/ms_cxr_t_stage1_promptonly_controls_gpu0/predictions.jsonl",
    "Q-LACE primary run": "medgemma_runs/ms_cxr_t_stage1_label_sft_controls_gpu0/ms_cxr_t_stage1_label_sft_controls_gpu0/predictions.jsonl",
}
NSE_SOURCES = {
    "BioViL-T prior/current": "medsiglip_runs/biovilt_stage1_prior_current_nse_stress_gpu1b_true_nse_4label/joined_predictions.jsonl",
    "Q-LACE primary run": "medgemma_runs/ms_cxr_t_nse_stress_stage1_sft_gpu2_4label/joined_predictions.jsonl",
    "MedGemma historical prompt-only": "medgemma_runs/ms_cxr_t_nse_stress_promptonly_gpu2/ms_cxr_t_nse_stress_promptonly_gpu2/predictions.jsonl",
    "Qwen2.5-VL-7B historical prompt-only": "medgemma_runs/ms_cxr_t_nse_stress_qwen25vl_gpu1/ms_cxr_t_nse_stress_qwen25vl_gpu1/predictions.jsonl",
}


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def normalize_finding(value: Any) -> str:
    return str(value).strip().replace("_", " ").casefold()


def image_identity(row: dict[str, Any], prior: bool) -> str:
    # Paths represent the actual input. Some old controls leave stale DICOM IDs.
    key = "prior_image_path" if prior else "current_image_path"
    value = row.get(key) or row.get("prior_dicom_id" if prior else "dicom_id")
    if not value:
        raise ValueError(f"Missing physical image identity: {key}")
    return str(value).replace("/legacy-data-alias/", "/data/")


def physical_key(row: dict[str, Any], condition: str) -> tuple[str, ...]:
    return (str(row.get("patient_id") or row.get("subject_id") or ""),
            image_identity(row, True), image_identity(row, False),
            normalize_finding(row["query_finding"]), condition)


def occurrence_keys(rows: list[dict[str, Any]], condition: str | None = None) -> list[tuple[Any, ...]]:
    seen: Counter = Counter()
    keys = []
    for row in rows:
        base = physical_key(row, condition or row["control_name"])
        occurrence = seen[base]
        seen[base] += 1
        keys.append((*base, occurrence))
    return keys


def source_collision_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["source_id"])].append(row)
    collisions = []
    for source_id, group in groups.items():
        if len(group) > 1:
            collisions.append({"source_id": source_id, "row_count": len(group),
                               "distinct_prior_images": sorted({image_identity(r, True) for r in group}),
                               "distinct_current_images": sorted({image_identity(r, False) for r in group}),
                               "gold_labels": [r["query_change"] for r in group]})
    return {"row_count": len(rows), "unique_source_ids": len(groups),
            "rows_lost_by_source_id_deduplication": len(rows) - len(groups),
            "collision_group_count": len(collisions), "collision_groups": collisions}


def align_manifest(manifest: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Require complete archived order; refuse lossy joins or guessed repairs.

    Historical raw outputs lack physical image paths. Full sequence, finding,
    and gold-label checks make occurrence alignment auditable, but cannot prove
    the absent paths retrospectively. The report states that limitation.
    """
    if len(manifest) != len(predictions):
        raise ValueError(f"Row count mismatch: {len(manifest)} != {len(predictions)}")
    keys = occurrence_keys(manifest)
    joined = []
    for index, (ref, pred, key) in enumerate(zip(manifest, predictions, keys)):
        if str(ref["source_id"]) != str(pred.get("source_id")):
            raise ValueError(f"Manifest order/source mismatch at row {index}")
        if normalize_finding(ref["query_finding"]) != normalize_finding(pred.get("query_finding")):
            raise ValueError(f"Finding mismatch at row {index}")
        pred_gold = pred.get("gold_label", pred.get("target_label", pred.get("query_change")))
        if pred_gold != ref["query_change"]:
            raise ValueError(f"Gold label mismatch at row {index}")
        for field in ("prior_image_path", "current_image_path"):
            if field in pred and image_identity(pred, field.startswith("prior")) != image_identity(ref, field.startswith("prior")):
                raise ValueError(f"Physical image mismatch at row {index}: {field}")
        joined.append({**ref, "physical_occurrence_key": list(key),
                       "pred_label": pred.get("pred_label") or "__invalid__"})
    if len(set(keys)) != len(keys):
        raise AssertionError("Occurrence keys must be unique")
    return joined


def classification(rows: list[dict[str, Any]]) -> dict[str, Any]:
    gold = [r["query_change"] for r in rows]
    pred = [r["pred_label"] for r in rows]
    metrics = compute_multiclass_metrics(gold, pred, labels=SUPPORTED + (NSE,))
    return {"count": len(rows), "accuracy": metrics["accuracy"],
            "supported_macro_f1": sum(metrics["per_class"][label]["f1"] for label in SUPPORTED) / 3,
            "legacy_four_class_macro_f1": metrics["macro_f1"],
            "correct_count": sum(a == b for a, b in zip(gold, pred)),
            "prediction_counts": dict(Counter(pred)), "gold_counts": dict(Counter(gold)),
            "per_class": metrics["per_class"], "confusion_matrix": metrics["confusion_matrix"]}


def temporal_metrics(rows: list[dict[str, Any]], canonical: list[dict[str, Any]]) -> dict[str, Any]:
    groups = {condition: [r for r in rows if r["control_name"] == condition] for condition in CONDITIONS}
    if any(len(group) != len(canonical) for group in groups.values()):
        raise ValueError("Each temporal condition must preserve all canonical rows")
    # The archived builder emits one block per condition in canonical order.
    for condition, group in groups.items():
        for index, (row, anchor) in enumerate(zip(group, canonical)):
            if row.get("control_base_source_id") != anchor["source_id"]:
                raise ValueError(f"Canonical lineage mismatch: {condition} row {index}")
            if normalize_finding(row["query_finding"]) != normalize_finding(anchor["query_finding"]):
                raise ValueError(f"Canonical query mismatch: {condition} row {index}")
            if str(row["patient_id"]) != str(anchor["patient_id"]):
                raise ValueError(f"Canonical patient mismatch: {condition} row {index}")
            if condition != "time_reversal" and image_identity(row, False) != image_identity(anchor, False):
                raise ValueError(f"Current input mismatch: {condition} row {index}")
            if condition == "duplicate_pair" and image_identity(row, True) != image_identity(row, False):
                raise ValueError("Duplicate-pair inputs are not identical")
    baseline, reversal = groups["baseline"], groups["time_reversal"]
    for base, rev, anchor in zip(baseline, reversal, canonical):
        if physical_key(base, "baseline") != physical_key(anchor, "baseline"):
            raise ValueError("Baseline physical inputs differ from canonical manifest")
        if image_identity(base, True) != image_identity(rev, False) or image_identity(base, False) != image_identity(rev, True):
            raise ValueError("Time reversal does not exchange actual image inputs")
        if reverse(base["query_change"]) != rev["query_change"]:
            raise ValueError("Archived reversal gold label is inconsistent")
    correct = [reverse(base["pred_label"]) == rev["pred_label"]
               for base, rev in zip(baseline, reversal)]
    both = [base["pred_label"] == base["query_change"] and rev["pred_label"] == rev["query_change"]
            for base, rev in zip(baseline, reversal)]
    supported_consistent = [base["pred_label"] in SUPPORTED and reverse(base["pred_label"]) == rev["pred_label"]
                            for base, rev in zip(baseline, reversal)]
    directional_gold = [(base, rev) for base, rev in zip(baseline, reversal)
                        if base["query_change"] in {"improving", "worsening"}]
    per_condition = {condition: classification(group) for condition, group in groups.items()}
    stable_gold = [r for r in groups["duplicate_pair"] if r["query_change"] == "stable"]
    stable_count = sum(r["pred_label"] == "stable" for r in stable_gold)
    duplicate_stable = sum(r["pred_label"] == "stable" for r in groups["duplicate_pair"])
    return {"by_condition": per_condition,
            "prior_drop_delta_supported_macro_f1": per_condition["prior_drop"]["supported_macro_f1"] - per_condition["baseline"]["supported_macro_f1"],
            "random_prior_delta_supported_macro_f1": per_condition["random_prior"]["supported_macro_f1"] - per_condition["baseline"]["supported_macro_f1"],
            "temporal_reversal_consistency": {"count": len(correct), "consistent_count": sum(correct), "rate": sum(correct) / len(correct),
                                               "definition": "Baseline/reversal predictions invert improving/worsening and preserve stable/NSE; invalid predictions fail.",
                                               "supported_labels_only_consistent_count": sum(supported_consistent),
                                               "supported_labels_only_rate": sum(supported_consistent) / len(correct),
                                               "directional_gold_count": len(directional_gold),
                                               "directional_gold_consistent_count": sum(reverse(a["pred_label"]) == b["pred_label"] for a, b in directional_gold),
                                               "baseline_and_reversal_both_correct_count": sum(both), "baseline_and_reversal_both_correct_rate": sum(both) / len(both)},
            "duplicate_pair": {"count": len(canonical), "stable_output_count": duplicate_stable,
                               "stable_output_rate_all_rows": duplicate_stable / len(canonical),
                               "false_change_output_count": sum(r["pred_label"] in {"improving", "worsening"} for r in groups["duplicate_pair"]),
                               "legacy_stable_recall_denominator": len(stable_gold), "legacy_stable_recall_numerator": stable_count,
                               "legacy_stable_recall": stable_count / len(stable_gold)}}


def reverse(label: str) -> str | None:
    return {"improving": "worsening", "worsening": "improving", "stable": "stable", NSE: NSE}.get(label)


def nse_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(r["pred_label"] for r in rows)
    n, k = len(rows), counts[NSE]
    conventional_f1 = 2 * k / (n + k) if n else 0.0
    return {"count": n, "prediction_counts": dict(counts), "abstention_count": k,
            "abstention_rate": k / n if n else 0.0,
            "supported_output_count": sum(counts[label] for label in SUPPORTED),
            "supported_output_rate": sum(counts[label] for label in SUPPORTED) / n if n else 0.0,
            "synthetic_nse_class_f1_not_clinical_performance": conventional_f1,
            "legacy_four_class_macro_f1_not_abstention_f1": conventional_f1 / 4}


def compare_prediction_vectors(reference: list[dict[str, Any]], left: list[dict[str, Any]],
                               right: list[dict[str, Any]], *, allow_generic_finding: bool = False) -> dict[str, Any]:
    """Compare matched complete vectors, allowing only the baseline ID suffix."""
    if len(reference) != len(left) or len(reference) != len(right):
        raise ValueError("Comparison vectors must preserve all reference rows")
    source_counts = Counter(r["source_id"] for r in reference)
    differences = []
    for index, (ref, a, b) in enumerate(zip(reference, left, right)):
        for item in (a, b):
            source_id = str(item["source_id"]).removesuffix("::control::baseline")
            if source_id != ref["source_id"]:
                raise ValueError(f"Comparison source/order mismatch at {index}")
            acceptable_findings = {normalize_finding(ref["query_finding"])}
            if allow_generic_finding:
                acceptable_findings.add("the unspecified finding")
            if normalize_finding(item["query_finding"]) not in acceptable_findings:
                raise ValueError(f"Comparison finding mismatch at {index}")
            if item.get("gold_label", item.get("target_label", item.get("query_change"))) != ref["query_change"]:
                raise ValueError(f"Comparison gold mismatch at {index}")
        if a["pred_label"] != b["pred_label"]:
            differences.append({"canonical_row_index_zero_based": index, "source_id": ref["source_id"],
                                "physical_key": list(physical_key(ref, "baseline")),
                                "source_id_collision_group": source_counts[ref["source_id"]] > 1,
                                "gold_label": ref["query_change"], "left_prediction": a["pred_label"],
                                "right_prediction": b["pred_label"],
                                "left_correct": a["pred_label"] == ref["query_change"],
                                "right_correct": b["pred_label"] == ref["query_change"]})
    return {"count": len(reference), "different_prediction_count": len(differences),
            "differences_with_colliding_source_id": sum(r["source_id_collision_group"] for r in differences),
            "differences_with_unique_source_id": sum(not r["source_id_collision_group"] for r in differences),
            "left_correct_count": sum(r["pred_label"] == ref["query_change"] for r, ref in zip(left, reference)),
            "right_correct_count": sum(r["pred_label"] == ref["query_change"] for r, ref in zip(right, reference)),
            "differences": differences}


def atomic_write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False, encoding="utf-8") as handle:
        handle.write(value)
        temporary = handle.name
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "manuscript_rev10/artifacts/cmig_validation/temporal")
    args = parser.parse_args()
    inputs = {}

    def record(relative: str) -> Path:
        path = ROOT / relative
        inputs[relative] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "size_bytes": path.stat().st_size}
        return path

    canonical = load_rows(record("share_data/ms_cxr_t_longitudinal_stage1/test.jsonl"))
    manifest = load_rows(record("share_data/ms_cxr_t_longitudinal_stage1_controls/all_controls.jsonl"))
    assert len(canonical) == 1326 and len(manifest) == 6630
    assert len(set(occurrence_keys(canonical, "baseline"))) == 1326
    output = {"scope": "CPU-only archived temporal-control and synthetic-candidate audit; historical files unchanged",
              "canonical_collision_audit": source_collision_audit(canonical),
              "control_manifest_audit": {condition: {"row_count": sum(r["control_name"] == condition for r in manifest),
                                                     "unique_physical_inputs_without_occurrence": len({physical_key(r, condition) for r in manifest if r["control_name"] == condition}),
                                                     "gold_label_counts": dict(Counter(r["query_change"] for r in manifest if r["control_name"] == condition))}
                                         for condition in CONDITIONS},
              "alignment": "Complete source-ID sequence, finding, gold label, condition blocks and physical manifest transformations verified; occurrences preserve repeated IDs.",
              "alignment_limitation": "Historical raw predictions omit image paths; within-collision image provenance follows archived manifest order, not independently stored prediction paths.",
              "paper_run_coverage": {"Q-LACE primary run": "complete 6630-row suite", "Q-LACE run 2": "no archived temporal suite found", "Q-LACE run 3": "no archived temporal suite found"},
              "temporal": {}, "synthetic_candidates": {},
              "interpretation": ["All F1 values use declared fixed label sets. Supported macro-F1 excludes NSE from averaging without removing any row or prediction error.",
                                 "Random-prior and prior-drop preserve original labels as diagnostics; they are not adjudicated labels for altered images.",
                                 "Duplicate-pair manifest preserves original change gold. Legacy stable recall uses only original stable rows; stable output rate over all 1326 rows is separately reported.",
                                 "Synthetic candidate targets are not adjudicated negatives. Remove candidate F1 from the manuscript and retain abstention rates and output counts."]}
    table = ["system\tbaseline_f1_3\tprior_drop_f1_3\tdelta_drop_f1_3\trandom_prior_f1_3\treversal_f1_3\ttrc_numerator\ttrc_denominator\ttrc\tduplicate_stable_all_rate\tlegacy_duplicate_stable_recall"]
    for name, relative in TEMPORAL_SOURCES.items():
        predictions = load_rows(record(relative))
        joined = align_manifest(manifest, predictions)
        metrics = temporal_metrics(joined, canonical)
        last_by_source = {r["source_id"]: r for r in predictions}
        legacy = [{**ref, "pred_label": last_by_source[ref["source_id"]]["pred_label"]} for ref in manifest]
        legacy_metrics = {c: classification([r for r in legacy if r["control_name"] == c]) for c in CONDITIONS}
        legacy_trc = compute_temporal_reversal_consistency(legacy)
        metrics["legacy_last_write_wins_replay"] = {"by_condition": legacy_metrics, "temporal_reversal_consistency": legacy_trc,
                                                       "prediction_rows_overwritten": sum(a["pred_label"] != b["pred_label"] for a, b in zip(joined, legacy))}
        metrics["source"] = relative
        output["temporal"][name] = metrics
        c, trc, d = metrics["by_condition"], metrics["temporal_reversal_consistency"], metrics["duplicate_pair"]
        table.append("\t".join(map(str, [name, c["baseline"]["supported_macro_f1"], c["prior_drop"]["supported_macro_f1"], metrics["prior_drop_delta_supported_macro_f1"], c["random_prior"]["supported_macro_f1"], c["time_reversal"]["supported_macro_f1"], trc["consistent_count"], trc["count"], trc["rate"], d["stable_output_rate_all_rows"], d["legacy_stable_recall"]])))
    candidate_manifest = load_rows(record("share_data/ms_cxr_t_nse_stress_candidates/test.jsonl"))
    assert len(candidate_manifest) == 500
    for name, relative in NSE_SOURCES.items():
        predictions = load_rows(record(relative))
        assert len(predictions) == len(candidate_manifest)
        assert [r["source_id"] for r in predictions] == [r["source_id"] for r in candidate_manifest]
        output["synthetic_candidates"][name] = {"source": relative, **nse_counts(predictions)}
    standalone_dir = "medgemma_runs/ms_cxr_t_stage1_label_sft_benchmark_all_test_gpu6_big/ms_cxr_t_stage1_label_sft_benchmark_all_test_gpu6_big"
    control_dir = str(Path(TEMPORAL_SOURCES["Q-LACE primary run"]).parent)
    joined_dir = "medgemma_runs/ms_cxr_t_stage1_label_sft_all_test_4label_rescored_gpu6_big/ms_cxr_t_stage1_label_sft_all_test_4label_rescored_gpu6_big"
    standalone_rows = load_rows(record(standalone_dir + "/predictions.jsonl"))
    standalone_joined = load_rows(record(joined_dir + "/joined_predictions.jsonl"))
    control_baseline = load_rows(ROOT / TEMPORAL_SOURCES["Q-LACE primary run"])[:len(canonical)]
    standalone_args = json.loads(record(standalone_dir + "/summary.json").read_text())["args"]
    control_args = json.loads(record(control_dir + "/summary.json").read_text())["args"]
    raw_comparison = compare_prediction_vectors(canonical, standalone_rows, control_baseline)
    arg_keys = sorted(set(standalone_args) | set(control_args))

    def rendered_prefix(row: dict[str, Any]) -> str:
        value = row.get("generated_text", "")
        if "\nmodel\n" not in value:
            raise ValueError("Missing archived rendered model-turn boundary")
        return value.rsplit("\nmodel\n", 1)[0]

    prompt_equal = [rendered_prefix(a) == rendered_prefix(b) for a, b in zip(standalone_rows, control_baseline)]
    output["primary_baseline_artifact_reconciliation"] = {
        "left": standalone_dir + "/predictions.jsonl", "right": TEMPORAL_SOURCES["Q-LACE primary run"],
        "standalone_raw_vs_temporal_baseline_raw": raw_comparison,
        "standalone_raw_vs_manuscript_joined": compare_prediction_vectors(canonical, standalone_rows, standalone_joined),
        "manuscript_joined_vs_temporal_baseline_raw": compare_prediction_vectors(canonical, standalone_joined, control_baseline),
        "identical_rendered_prompt_prefix_count": sum(prompt_equal),
        "rendered_prompt_prefix_count": len(prompt_equal),
        "prompt_prefix_digest_standalone": hashlib.sha256(json.dumps([rendered_prefix(r) for r in standalone_rows]).encode()).hexdigest(),
        "prompt_prefix_digest_controls": hashlib.sha256(json.dumps([rendered_prefix(r) for r in control_baseline]).encode()).hexdigest(),
        "identical_saved_inference_arguments": {key: standalone_args.get(key) for key in arg_keys if standalone_args.get(key) == control_args.get(key)},
        "different_saved_inference_arguments": {key: {"standalone": standalone_args.get(key), "controls": control_args.get(key)} for key in arg_keys if standalone_args.get(key) != control_args.get(key)},
        "standalone_model_path_currently_exists": Path(standalone_args["model_path"]).exists(),
        "controls_model_path_currently_exists": Path(control_args["model_path"]).exists(),
        "conclusion": "The same 1326 physical benchmark rows produce 11 raw prediction differences, all outside source-ID collision groups. Observed prompts and saved generation settings match. Model path locations differ; historical /tmp backbone no longer exists. The saved artifacts lack historical backbone hashes and complete software/device/preprocessing snapshots, so the cause cannot be established. These are separate inference artifacts, not a different population or solely deduplication effects.",
    }
    full_summaries = [joined_dir + "/summary.json",
                      "medgemma_runs/ms_cxr_t_stage1_label_sft_seed43_all_test_4label_gpu2/summary.json",
                      "medgemma_runs/ms_cxr_t_stage1_label_sft_seed46_all_test_4label_gpu1/summary.json"]
    integrity = {}
    for index, (summary_relative, internal_seed) in enumerate(zip(full_summaries, (42, 43, 46)), start=1):
        summary = json.loads(record(summary_relative).read_text())
        raw_path = Path(summary["args"]["predictions_jsonl"])
        raw_relative = str(raw_path.resolve().relative_to(ROOT)) if raw_path.is_absolute() else str(raw_path)
        raw = load_rows(record(raw_relative))
        joined_relative = str(Path(summary_relative).parent / "joined_predictions.jsonl")
        joined = load_rows(record(joined_relative))
        generic_base = f"medgemma_runs/ms_cxr_t_generic_query_seed{internal_seed}_all_test"
        generic_raw_relative = f"{generic_base}/{Path(generic_base).name}/predictions.jsonl"
        generic_joined_relative = f"{generic_base}_4label_scored/joined_predictions.jsonl"
        generic_raw = load_rows(record(generic_raw_relative))
        generic_joined = load_rows(record(generic_joined_relative))
        if not all(r["query_finding"] == "the unspecified finding" for r in generic_raw):
            raise ValueError("Generic control does not uniformly remove finding identity")
        raw_metrics = classification([{**ref, "pred_label": pred["pred_label"]} for ref, pred in zip(canonical, raw)])
        generic_raw_metrics = classification([{**ref, "pred_label": pred["pred_label"]} for ref, pred in zip(canonical, generic_raw)])
        integrity[f"Run {index}"] = {
            "full_raw_source": raw_relative, "full_joined_source": joined_relative,
            "full_raw_vs_joined": compare_prediction_vectors(canonical, raw, joined),
            "full_raw_metrics": raw_metrics,
            "generic_raw_source": generic_raw_relative, "generic_joined_source": generic_joined_relative,
            "generic_raw_vs_joined": compare_prediction_vectors(canonical, generic_raw, generic_joined, allow_generic_finding=True),
            "generic_raw_metrics": generic_raw_metrics,
            "generic_actual_finding_string": "the unspecified finding",
            "all_full_raw_patient_ids_match_canonical": all(str(r["patient_id"]) == str(ref["patient_id"]) for r, ref in zip(raw, canonical)),
            "all_generic_raw_patient_ids_match_canonical": all(str(r["patient_id"]) == str(ref["patient_id"]) for r, ref in zip(generic_raw, canonical)),
        }
    output["three_paper_runs_raw_vs_joined_integrity"] = integrity
    for relative in ("scripts/cmig_temporal_validation.py", "scripts/eval_ms_cxr_t_classification.py", "medgemma_visual_rft/longitudinal_classification.py", "tests/test_cmig_temporal_validation.py"):
        record(relative)
    output["inputs"] = inputs
    report = ["# CMIG temporal-control and candidate abstention validation", "", "Generated by `python scripts/cmig_temporal_validation.py`; no inference.", "",
              "All eight temporal systems retain 1,326 rows per condition (6,630 total). The canonical source ID has 12 collisions, each involving a distinct prior image; it has only 1,314 unique values.", "",
              "The old evaluator did not reduce classification denominators: it reused the last prediction for colliding IDs. Its reversal pairing excluded both members of each collision, scoring only 1,302 pairs. Corrected reversal pairing scores all 1,326.", "",
              "Only the primary Q-LACE temporal suite exists. These are single-run diagnostics, not three-run means.", "",
              "## Corrected supported three-class metrics", "", "|System|Baseline F1|Prior drop F1|Random prior F1|Reversal F1|TRC (all 1326)|Duplicate stable output rate|", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, m in output["temporal"].items():
        c = m["by_condition"]
        report.append(f"|{name}|{c['baseline']['supported_macro_f1']:.4f}|{c['prior_drop']['supported_macro_f1']:.4f}|{c['random_prior']['supported_macro_f1']:.4f}|{c['time_reversal']['supported_macro_f1']:.4f}|{m['temporal_reversal_consistency']['rate']:.4f}|{m['duplicate_pair']['stable_output_rate_all_rows']:.4f}|")
    report += ["", "The duplicate-pair stable output rate covers all rows; the old stable recall covers only 530 originally stable rows. This distinction must be named explicitly.", "", "## Synthetic candidates: behavior, not validated clinical negatives", "", "|System|NSE count / 500|Abstention rate|Old column (4-class macro-F1)|NSE class F1 under synthetic labels|", "|---|---:|---:|---:|---:|"]
    for name, m in output["synthetic_candidates"].items():
        report.append(f"|{name}|{m['abstention_count']} / {m['count']}|{m['abstention_rate']:.3f}|{m['legacy_four_class_macro_f1_not_abstention_f1']:.6f}|{m['synthetic_nse_class_f1_not_clinical_performance']:.6f}|")
    report += ["", "The manuscript column is confirmed to be fixed four-class macro-F1, one quarter of NSE class F1. Remove the F1 column and retain abstention rates and counts; no candidate was adjudicated as negative.", "", "## Standalone primary versus temporal-suite baseline", "", output["primary_baseline_artifact_reconciliation"]["conclusion"], "", "Standalone raw predictions have 785 correct rows; temporal-baseline raw predictions have 781. The manuscript joined primary output differs from standalone raw on two collision rows, while preserving its 785 correct total. Relative to that joined artifact, temporal baseline differs on 13 rows (11 unique-ID rows plus two colliding-ID rows). All 1326 observable rendered prompt prefixes match exactly; saved generation budget is 16 tokens, label mode, no prefill, repetition penalty 1.0 and JSON stop disabled in both runs. Adapter paths are identical. Only input manifest and backbone path differ among saved inference arguments.", "", "## Three paper-run integrity", "", "Raw prediction sources are recovered from saved scorer arguments. All full-query and generic raw files contain 1326 rows and match canonical source/patient order; all generic raw prompts use the unspecified-finding string.", "", "|Paper run|Full raw versus joined differences|Generic raw versus joined differences|Raw full accuracy|Raw full supported F1|", "|---|---:|---:|---:|---:|"]
    for name, entry in integrity.items():
        report.append(f"|{name}|{entry['full_raw_vs_joined']['different_prediction_count']}|{entry['generic_raw_vs_joined']['different_prediction_count']}|{entry['full_raw_metrics']['accuracy']:.6f}|{entry['full_raw_metrics']['supported_macro_f1']:.6f}|")
    report += ["", "The old full-query joined files introduce two net extra correct predictions across the three paper runs. Aggregates must use raw occurrence-preserving predictions. The existing three-run recomputation already reads generic raw files, so its generic arm does not inherit the generic joined substitution.", "", "## Provenance limitation", "", output["alignment_limitation"], "", "All source paths, hashes, confusion matrices, exact counts, and legacy replays are in `report.json`. Historical artifacts were preserved.", ""]
    atomic_write(args.output_dir / "report.json", json.dumps(output, indent=2) + "\n")
    atomic_write(args.output_dir / "temporal_metrics.tsv", "\n".join(table) + "\n")
    atomic_write(args.output_dir / "REPORT.md", "\n".join(report))
    print(args.output_dir)


if __name__ == "__main__":
    main()
