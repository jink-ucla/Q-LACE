#!/usr/bin/env python3
"""CPU-only, patient-paired analysis of the five prespecified CMIG passes.

Requires every full pass and verifies contracts before reading any outcome.
Historical outputs retain their original output contract and parser; they are
bridges, not identically configured inference. Never selects a prompt.
"""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import statistics
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "manuscript_rev10/artifacts/cmig_validation/baseline"
SPEC = importlib.util.spec_from_file_location("cmig_frozen_harness", ROOT / "scripts/cmig_medgemma_baseline.py")
HARNESS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HARNESS)
EXECUTING_SCRIPT_SHA256 = HARNESS.sha256_file(Path(__file__))
LABELS = HARNESS.LABELS
PREDICTION_LABELS = (*LABELS, "no_supported_evidence", "INVALID")
METRICS = ("accuracy", "supported_macro_f1", "pooled_class_balanced_accuracy", "mean_finding_class_balanced_accuracy")
NEW_ARMS = ("base_native", "qlace_run1_native", "qlace_run2_native", "qlace_run3_native", "base_squarepad")
HISTORICAL_ARMS = ("historical_prompt", "historical_qlace_run1", "historical_qlace_run2", "historical_qlace_run3")
GENERIC_ARMS = ("historical_generic_run1", "historical_generic_run2", "historical_generic_run3")
ARM_LABELS = {
    "base_native": "MedGemma, fixed native MCQ",
    "qlace_run1_native": "Q-LACE Run 1, fixed native MCQ",
    "qlace_run2_native": "Q-LACE Run 2, fixed native MCQ",
    "qlace_run3_native": "Q-LACE Run 3, fixed native MCQ",
    "base_squarepad": "MedGemma, square-padding sensitivity",
    "historical_prompt": "MedGemma, historical four-label interface",
    "historical_qlace_run1": "Q-LACE Run 1, historical interface",
    "historical_qlace_run2": "Q-LACE Run 2, historical interface",
    "historical_qlace_run3": "Q-LACE Run 3, historical interface",
    "historical_generic_run1": "Finding-blind Run 1, historical interface",
    "historical_generic_run2": "Finding-blind Run 2, historical interface",
    "historical_generic_run3": "Finding-blind Run 3, historical interface",
    "historical_biovilt": "BioViL-T, original classifier outputs",
}
# Numeric training identifiers occur only in this internal reproducibility map;
# generated report labels use Run 1--3.
INTERNAL_RUNS = {
    "historical_prompt": "medgemma_runs/ms_cxr_t_stage1_promptonly_all_test_4label_rescored_gpu7/joined_predictions.jsonl",
    "historical_qlace_run1": "medgemma_runs/ms_cxr_t_stage1_label_sft_all_test_4label_rescored_gpu6_big/ms_cxr_t_stage1_label_sft_all_test_4label_rescored_gpu6_big/joined_predictions.jsonl",
    "historical_qlace_run2": "medgemma_runs/ms_cxr_t_stage1_label_sft_seed43_all_test_4label_gpu2/joined_predictions.jsonl",
    "historical_qlace_run3": "medgemma_runs/ms_cxr_t_stage1_label_sft_seed46_all_test_4label_gpu1/joined_predictions.jsonl",
    "historical_biovilt": "medsiglip_runs/ms_cxr_t_biovilt_prior_current_all_test_4label_rescored_gpu1_fast7/joined_predictions.jsonl",
}
INTERNAL_RAW_RUNS = {
    "historical_prompt": "medgemma_runs/ms_cxr_t_stage1_promptonly_label_benchmark_gpu7/predictions.jsonl",
    "historical_qlace_run1": "medgemma_runs/ms_cxr_t_stage1_label_sft_benchmark_all_test_gpu6_big/ms_cxr_t_stage1_label_sft_benchmark_all_test_gpu6_big/predictions.jsonl",
    "historical_qlace_run2": "medgemma_runs/ms_cxr_t_stage1_label_sft_seed43_all_test_gpu2/ms_cxr_t_stage1_label_sft_seed43_all_test_gpu2/predictions.jsonl",
    "historical_qlace_run3": "medgemma_runs/ms_cxr_t_stage1_label_sft_seed46_all_test_gpu1/ms_cxr_t_stage1_label_sft_seed46_all_test_gpu1/predictions.jsonl",
    "historical_generic_run1": "medgemma_runs/ms_cxr_t_generic_query_seed42_all_test/ms_cxr_t_generic_query_seed42_all_test/predictions.jsonl",
    "historical_generic_run2": "medgemma_runs/ms_cxr_t_generic_query_seed43_all_test/ms_cxr_t_generic_query_seed43_all_test/predictions.jsonl",
    "historical_generic_run3": "medgemma_runs/ms_cxr_t_generic_query_seed46_all_test/ms_cxr_t_generic_query_seed46_all_test/predictions.jsonl",
    "historical_biovilt": "medsiglip_runs/biovilt_stage1_prior_current_full_gpu1_fast7_benchmark_all_test/eval_predictions.jsonl",
}
INTERNAL_ADAPTERS = {
    "qlace_run1_native": "medgemma_runs/longitudinal_stage1_label_sft_gpu6_big/adapter",
    "qlace_run2_native": "medgemma_runs/longitudinal_stage1_label_sft_seed43_gpu2/adapter",
    "qlace_run3_native": "medgemma_runs/longitudinal_stage1_label_sft_seed46_gpu1_fast/adapter",
}
CONTRASTS = {
    "primary_adaptation_same_contract": (["qlace_run1_native", "qlace_run2_native", "qlace_run3_native"], ["base_native"]),
    "secondary_native_vs_historical_prompt": (["base_native"], ["historical_prompt"]),
    "secondary_squarepad_vs_native": (["base_squarepad"], ["base_native"]),
    "bridge_adapters_native_vs_historical": (["qlace_run1_native", "qlace_run2_native", "qlace_run3_native"], ["historical_qlace_run1", "historical_qlace_run2", "historical_qlace_run3"]),
    "historical_adaptation_original_contract": (["historical_qlace_run1", "historical_qlace_run2", "historical_qlace_run3"], ["historical_prompt"]),
    "historical_full_vs_matched_finding_blind": (["historical_qlace_run1", "historical_qlace_run2", "historical_qlace_run3"], list(GENERIC_ARMS)),
    "historical_full_vs_biovilt": (["historical_qlace_run1", "historical_qlace_run2", "historical_qlace_run3"], ["historical_biovilt"]),
}


def check_fingerprint(fingerprint: dict[str, Any]) -> None:
    files = fingerprint["files"]
    if not files or len({entry["name"] for entry in files}) != len(files):
        raise ValueError("Empty or duplicate checkpoint fingerprint")
    if HARNESS.digest_json(files) != fingerprint["files_sha256"]:
        raise ValueError("Checkpoint file-list fingerprint is internally inconsistent")
    for entry in files:
        if len(entry["sha256"]) != 64 or not isinstance(entry["bytes"], int) or entry["bytes"] < 0:
            raise ValueError("Malformed checkpoint file fingerprint")


def load_new_run(rows: list[dict[str, Any]], directory: Path, expected_protocol: dict[str, Any],
                 expected_adapter: Path | None, script_hash: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    required = [directory / name for name in ("run_manifest.json", "predictions.jsonl", "metrics.json")]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ValueError(f"Full run is not finalized: {missing}")
    provenance = json.loads(required[0].read_text())
    contract = provenance["contract"]
    if contract.get("smoke_only") is not False:
        raise ValueError(f"Smoke output cannot enter a full comparison: {directory}")
    if contract["manifest_sha256"] != HARNESS.digest_json(rows):
        raise ValueError("Run/manifest semantic hashes differ")
    if contract["selected_row_ids"] != [r["row_id"] for r in rows]:
        raise ValueError("Run selection does not preserve all canonical rows")
    if contract["protocol"] != expected_protocol or contract["protocol_sha256"] != HARNESS.digest_json(expected_protocol):
        raise ValueError("Run did not use the fixed expected protocol")
    if contract["script_sha256"] != script_hash:
        raise ValueError("Inference code hash differs from the frozen harness")
    if contract["dtype"] != "bfloat16" or contract["do_sample"] or contract["cpu_offload"]:
        raise ValueError("Inference precision/decoding/offload contract differs")
    if contract["attention_implementation"] != "eager" or contract["use_cache"] is not True:
        raise ValueError("Inference attention/cache contract differs")
    check_fingerprint(contract["model"])
    adapter = contract["adapter"]
    if expected_adapter is None:
        if adapter is not None:
            raise ValueError("Base-model arm unexpectedly contains an adapter")
    else:
        if adapter is None or Path(adapter["path"]).resolve() != expected_adapter.resolve():
            raise ValueError("Adapter identity is not the prespecified manuscript run")
        check_fingerprint(adapter)
    predictions = HARNESS.load_jsonl(required[1])
    lookup = HARNESS.validate_predictions(rows, predictions)
    predictions = [lookup[row["row_id"]] for row in rows]
    for pred in predictions:
        if pred["protocol_sha256"] != contract["protocol_sha256"]:
            raise ValueError("Per-row protocol differs from run contract")
        parsed, error = HARNESS.parse_answer(pred["raw_text"], expected_protocol["option_mapping"])
        if parsed != pred["pred_label"] or bool(error) != bool(pred["parse_error"]):
            raise ValueError("Saved strict parsing differs from frozen parser")
        tokens = pred["generated_token_ids"]
        if len(tokens) != pred["generated_tokens"] or len(tokens) > expected_protocol["max_new_tokens"]:
            raise ValueError("Generation token count violates saved contract")
        if not isinstance(pred["eos_observed"], bool) or not isinstance(pred["truncated"], bool):
            raise ValueError("EOS/truncation observations are missing or malformed")
        if pred["truncated"] != (len(tokens) >= expected_protocol["max_new_tokens"] and not pred["eos_observed"]):
            raise ValueError("Truncation status is internally inconsistent")
        for key in ("prior_image_sha256", "current_image_sha256"):
            if not isinstance(pred.get(key), str) or len(pred[key]) != 64:
                raise ValueError("Physical image content hash missing")
    metrics = json.loads(required[2].read_text())
    if metrics.get("smoke_only") is not False or metrics.get("n_full_manifest_rows") != len(rows):
        raise ValueError("Final metrics do not describe the full manifest")
    recomputed = HARNESS.score_predictions(rows, predictions)
    for key in ("accuracy", "supported_macro_f1", "class_balanced_accuracy", "invalid_count", "truncated_count", "n_rows"):
        if not np.isclose(metrics[key], recomputed[key], rtol=0, atol=1e-12):
            raise ValueError(f"Stored final metric does not reproduce: {key}")
    return predictions, provenance


def align_historical(rows: list[dict[str, Any]], old_rows: list[dict[str, Any]],
                     *, finding_blind: bool = False, reduced_metadata: bool = False) -> list[dict[str, Any]]:
    if len(old_rows) != len(rows):
        raise ValueError("Historical row population differs")
    result = []
    for index, (row, old) in enumerate(zip(rows, old_rows)):
        fields = ("source_id",) if reduced_metadata else ("source_id", "pair_id", "patient_id")
        for field in fields:
            if str(old[field]) != str(row[field]):
                raise ValueError(f"Historical occurrence alignment failed at row {index}: {field}")
        expected_finding = "the unspecified finding" if finding_blind else row["query_finding"]
        if str(old["query_finding"]).replace("_", " ") != expected_finding:
            raise ValueError(f"Historical finding/blinding validation failed at row {index}")
        gold = old.get("gold_label", old.get("target_label", old.get("query_change")))
        if gold != row["target_label"]:
            raise ValueError(f"Historical gold alignment failed at row {index}")
        for field in ("prior_image_path", "current_image_path"):
            if field in old and Path(old[field]).resolve() != Path(row[field]).resolve():
                raise ValueError(f"Historical physical image mismatch: {field}")
        prediction = old.get("pred_label")
        if prediction not in PREDICTION_LABELS:
            prediction = None
        result.append({**row, "pred_label": prediction, "historical_parse_success": old.get("parse_success"),
                       "eos_observed": None, "truncated": None})
    return result


def diagnostic_counts(records: list[dict[str, Any]], historical: bool) -> dict[str, Any]:
    result = {
        "n_rows": len(records), "unsupported_output_count": sum(r.get("pred_label") == "no_supported_evidence" for r in records),
        "invalid_output_count": sum(r.get("pred_label") not in PREDICTION_LABELS[:-1] for r in records),
        "predicted_label_counts": dict(Counter(r.get("pred_label") or "INVALID" for r in records)),
    }
    if historical:
        known = [r["historical_parse_success"] for r in records if r.get("historical_parse_success") is not None]
        result.update({"historical_parse_success_observed_rows": len(known),
                       "parse_failure_count": sum(not bool(value) for value in known) if known else None,
                       "truncated_count": None, "eos_observed_count": None, "no_eos_count": None,
                       "generation_diagnostic_note": "Historical exports lack token IDs and EOS/truncation metadata; unavailable is not zero."})
    else:
        result.update({"parse_failure_count": sum(bool(r["parse_error"]) for r in records),
                       "truncated_count": sum(r["truncated"] for r in records),
                       "eos_observed_count": sum(r["eos_observed"] for r in records),
                       "no_eos_count": sum(not r["eos_observed"] for r in records),
                       "generated_tokens_mean": statistics.mean(r["generated_tokens"] for r in records),
                       "generated_tokens_max": max(r["generated_tokens"] for r in records)})
    return result


def patient_confusions(rows: list[dict[str, Any]], arms: dict[str, list[dict[str, Any]]]) -> tuple[np.ndarray, list[str], list[str], list[str]]:
    patients = sorted({r["patient_id"] for r in rows})
    findings = sorted({r["query_finding"] for r in rows})
    names = list(arms)
    pmap, fmap = ({v: i for i, v in enumerate(values)} for values in (patients, findings))
    gold_map = {v: i for i, v in enumerate(LABELS)}
    pred_map = {v: i for i, v in enumerate(PREDICTION_LABELS)}
    counts = np.zeros((len(patients), len(names), len(findings), 3, 5), dtype=np.int64)
    for arm_index, name in enumerate(names):
        records = arms[name]
        if len(records) != len(rows):
            raise ValueError("Arm length mismatch")
        for row, pred in zip(rows, records):
            if row["row_id"] != pred["row_id"]:
                raise ValueError("Arm order mismatch")
            counts[pmap[row["patient_id"]], arm_index, fmap[row["query_finding"]],
                   gold_map[row["target_label"]], pred_map.get(pred.get("pred_label"), 4)] += 1
    if not np.all(counts.sum(axis=(0, 2, 3, 4)) == len(rows)):
        raise ValueError("Confusion counts lost canonical rows")
    return counts, patients, findings, names


def metrics_from_counts(counts: np.ndarray) -> np.ndarray:
    """Vectorized metrics; final dimensions are finding, gold class, prediction."""
    pooled = counts.sum(axis=-3)
    tp = np.stack([pooled[..., i, i] for i in range(3)], axis=-1)
    support = pooled.sum(axis=-1)
    predicted = pooled[..., :3].sum(axis=-2)
    denominator = support + predicted
    f1 = np.divide(2 * tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator != 0).mean(axis=-1)
    total = pooled.sum(axis=(-1, -2))
    accuracy = np.divide(tp.sum(axis=-1), total, out=np.zeros_like(total, dtype=float), where=total != 0)
    recalls = np.divide(tp, support, out=np.zeros_like(tp, dtype=float), where=support != 0)
    n_supported = (support > 0).sum(axis=-1)
    balanced = np.divide(recalls.sum(axis=-1), n_supported, out=np.zeros_like(total, dtype=float), where=n_supported != 0)
    local_tp = np.stack([counts[..., i, i] for i in range(3)], axis=-1)
    local_support = counts.sum(axis=-1)
    local_recalls = np.divide(local_tp, local_support, out=np.zeros_like(local_tp, dtype=float), where=local_support != 0)
    local_n = (local_support > 0).sum(axis=-1)
    local_balanced = np.divide(local_recalls.sum(axis=-1), local_n, out=np.zeros_like(local_n, dtype=float), where=local_n != 0)
    finding_mean = local_balanced.mean(axis=-1)
    return np.stack((accuracy, f1, balanced, finding_mean), axis=-1)


def paired_bootstrap(counts: np.ndarray, *, replicates: int = 10000,
                     seed: int = 20261020, batch_size: int = 100) -> np.ndarray:
    if replicates < 1:
        raise ValueError("At least one bootstrap replicate is required")
    rng = np.random.default_rng(seed)
    n_patients = counts.shape[0]
    flat = counts.reshape(n_patients, -1)
    results = []
    for start in range(0, replicates, batch_size):
        size = min(batch_size, replicates - start)
        draws = rng.integers(0, n_patients, size=(size, n_patients))
        weights = np.zeros((size, n_patients), dtype=np.int64)
        np.add.at(weights, (np.arange(size)[:, None], draws), 1)
        boot_counts = (weights @ flat).reshape((size,) + counts.shape[1:])
        results.append(metrics_from_counts(boot_counts))
    return np.concatenate(results, axis=0)


def metric_intervals(point: np.ndarray, samples: np.ndarray) -> dict[str, Any]:
    limits = np.quantile(samples, (0.025, 0.975), axis=0, method="linear")
    return {name: {"estimate": float(point[index]), "ci_95": [float(limits[0, index]), float(limits[1, index])]}
            for index, name in enumerate(METRICS)}


def build_summary(rows: list[dict[str, Any]], arms: dict[str, list[dict[str, Any]]],
                  *, replicates: int = 10000, seed: int = 20261020) -> dict[str, Any]:
    counts, patients, findings, names = patient_confusions(rows, arms)
    point = metrics_from_counts(counts.sum(axis=0))
    bootstrap = paired_bootstrap(counts, replicates=replicates, seed=seed)
    lookup = {name: index for index, name in enumerate(names)}
    arm_results = {}
    for index, name in enumerate(names):
        finding_counts = counts[:, index].sum(axis=0)
        finding_results = {}
        for finding_index, finding in enumerate(findings):
            matrix = finding_counts[finding_index]
            local = metrics_from_counts(matrix[None, ...])
            finding_results[finding] = {
                "n_rows": int(matrix.sum()), "accuracy": float(local[0]),
                "supported_macro_f1": float(local[1]), "class_balanced_accuracy": float(local[2]),
                "confusion_matrix": matrix.tolist(),
                "gold_rows": list(LABELS), "prediction_columns": list(PREDICTION_LABELS),
            }
        arm_results[name] = {"label": ARM_LABELS.get(name, name),
                             "metrics": metric_intervals(point[index], bootstrap[:, index]),
                             "generation": diagnostic_counts(arms[name], name.startswith("historical_")),
                             "per_finding": finding_results}
    contrasts = {}
    for name, (left_names, right_names) in CONTRASTS.items():
        if any(value not in lookup for value in left_names + right_names):
            continue
        left = [lookup[v] for v in left_names]
        right = [lookup[v] for v in right_names]
        delta = point[left].mean(axis=0) - point[right].mean(axis=0)
        samples = bootstrap[:, left].mean(axis=1) - bootstrap[:, right].mean(axis=1)
        result = {"left_arms": left_names, "right_arms": right_names,
                  "metrics": metric_intervals(delta, samples),
                  "uncertainty_scope": "Paired patient resampling; fixed observed model runs; no uncertainty over newly trained random seeds."}
        if len(left) == 3:
            per_run_delta = point[left] - point[right].mean(axis=0)
            if len(right) == 3:
                per_run_delta = point[left] - point[right]
            result["observed_run_sample_sd"] = {metric: float(np.std(per_run_delta[:, k], ddof=1)) for k, metric in enumerate(METRICS)}
        contrasts[name] = result
    aggregates = {}
    for label, run_names in (("qlace_native_three_runs", list(NEW_ARMS[1:4])),
                              ("qlace_historical_three_runs", list(HISTORICAL_ARMS[1:])),
                              ("finding_blind_historical_three_runs", list(GENERIC_ARMS))):
        if not all(name in lookup for name in run_names):
            continue
        selected = [lookup[name] for name in run_names]
        aggregates[label] = {
            "run_count": 3, "arms": run_names,
            "metrics": metric_intervals(point[selected].mean(axis=0), bootstrap[:, selected].mean(axis=1)),
            "sample_sd": {metric: float(np.std(point[selected, k], ddof=1)) for k, metric in enumerate(METRICS)},
            "ci_scope": "Patient bootstrap conditional on the three observed runs; sample SD separately describes observed run variability.",
        }
    return {
        "schema": "cmig-baseline-comparison-v1", "n_rows": len(rows), "n_patients": len(patients),
        "findings": findings, "manifest_sha256": HARNESS.digest_json(rows),
        "bootstrap": {"replicates": replicates, "unit": "patient_id", "paired_across_all_arms": True,
                      "confidence_level": 0.95, "interval": "percentile, NumPy linear quantile",
                      "seed_uncertainty": "Conditional on observed runs; not a population-of-training-seeds interval."},
        "metric_definitions": {
            "accuracy": "Correct labels / all original row occurrences; NSE and invalid outputs are incorrect.",
            "supported_macro_f1": "Unweighted mean F1 across improving/stable/worsening only; no zero-support NSE class added.",
            "pooled_class_balanced_accuracy": "Mean per-change-class recall after pooling all finding rows.",
            "mean_finding_class_balanced_accuracy": "Unweighted mean of each finding's mean per-change-class recall. All five findings retained.",
            "zero_support_bootstrap_policy": "A zero-gold-support class is omitted from that bootstrap replicate's recall mean; macro-F1 retains the fixed three classes.",
            "published_macro_accuracy_caveat": "Neither balanced-accuracy bridge is asserted to reproduce Google's unavailable aggregation code.",
        },
        "output_contracts": {
            "new_native_and_padding": "User-only, explicit prior/current image annotations, three-option MCQ, no NSE, greedy decoding, 128-token cap, strict whole-response parser.",
            "historical": "Historical four-label canonical-label interface with system instruction, NSE permitted and 16-token cap for prompt-only; retain historical parsed labels. Adapter-specific historical generation provenance is in internal_manifest.json.",
            "bridge_limits": "Interface bridges change multiple prompt/output/preprocessing details together; they do not isolate one causal factor. Original historical generation exports lack image paths; canonical occurrence order and source/pair/patient/finding/gold were checked. BioViL-T classifier export retains source/finding/gold but lacks pair/patient fields. Historical EOS and truncation are unavailable.",
        },
        "arms": arm_results, "aggregates": aggregates, "contrasts": contrasts,
        "selection_statement": "The prompt, both preprocessing configurations and all five passes were fixed before new outcomes; no test-outcome prompt selection.",
        "hardware_caveat": "GPU allocation mixes H100 NVL and L40S; native and square-padding base runs are not hardware matched. Numerical equivalence to Google's executing environment is not established.",
    }


def markdown_report(summary: dict[str, Any]) -> str:
    lines = ["# Prespecified MedGemma baseline reconciliation", "",
             f"All {summary['n_rows']:,} original rows; {summary['n_patients']:,} patient clusters. "
             f"{summary['bootstrap']['replicates']:,} paired patient bootstrap replicates.", "",
             "| Interface/model | Accuracy | Supported macro-F1 | Pooled balanced accuracy | Mean finding balanced accuracy |", "|---|---:|---:|---:|---:|"]
    for name in (*NEW_ARMS, *HISTORICAL_ARMS, *GENERIC_ARMS, "historical_biovilt"):
        if name in summary["arms"]:
            arm = summary["arms"][name]
            values = [f"{arm['metrics'][key]['estimate']:.4f}" for key in METRICS]
            lines.append("| " + " | ".join([arm["label"], *values]) + " |")
    lines.extend(["", "## Prespecified paired contrasts", "",
                  "| Contrast (left minus right) | Accuracy delta [95% CI] | Macro-F1 delta [95% CI] |", "|---|---:|---:|"])
    for name, result in summary["contrasts"].items():
        values = []
        for key in METRICS[:2]:
            value = result["metrics"][key]
            values.append(f"{value['estimate']:+.4f} [{value['ci_95'][0]:+.4f}, {value['ci_95'][1]:+.4f}]")
        lines.append("| " + " | ".join([name.replace("_", " "), *values]) + " |")
    lines.extend(["", "The primary new-inference contrast compares the average metric across three existing Q-LACE runs against the frozen base under the identical native MCQ contract. The historical raw-only report contains no new-inference contrast. These confidence intervals condition on the observed runs; run sample SD is reported separately in the JSON.", "",
                  "Historical outputs retain their original parser and four-label interface. The historical prompt allowed NSE and had a 16-token generation cap. New runs use a user-only three-option prompt and 128-token cap. Interface bridges combine these changes; they cannot identify a single cause. Historical EOS/truncation metadata are unavailable, not zero.", "",
                  "The pooled and finding-averaged balanced accuracies are explicitly defined bridges. Google's exact aggregation implementation is unavailable, so neither is called an exact reproduction of its published macro accuracy. Both fixed preprocessing arms are retained regardless of their outcomes.", "",
                  "GPU allocation mixes H100 NVL and L40S; native and square-padding base runs are not hardware matched. The preprocessing contrast does not isolate padding with guaranteed hardware-identical arithmetic. Numerical equivalence to Google's executing environment is unverified. See the separate runtime integrity report for actual-import versus package-metadata provenance.", ""])
    if "historical_display_reconciliation" in summary:
        lines.extend(["## Historical display reconciliation", "",
                      "Historical comparison rows above use original raw generation outputs. Legacy joined files are preserved and rescored below because source-ID joining substituted predictions on collision rows.", "",
                      "| Historical arm | Changed labels (joined vs raw) | Legacy joined accuracy / F1 | Raw accuracy / F1 |", "|---|---:|---:|---:|"])
        for name, entry in summary["historical_display_reconciliation"].items():
            if entry.get("raw_available"):
                legacy, raw = entry["legacy_joined_metrics"], entry["raw_metrics"]
                lines.append(f"| {ARM_LABELS[name]} | {entry['different_prediction_rows']} | {legacy['accuracy']:.4f} / {legacy['supported_macro_f1']:.4f} | {raw['accuracy']:.4f} / {raw['supported_macro_f1']:.4f} |")
            else:
                lines.append(f"| {ARM_LABELS[name]} | unavailable | retained separately | raw unavailable |")
        lines.append("")
    return "\n".join(lines)


def collect_historical(rows: list[dict[str, Any]]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any], dict[str, Any]]:
    arms, reconciliation, input_files = {}, {}, {}
    for name, relative in INTERNAL_RAW_RUNS.items():
        path = ROOT / relative
        joined = None
        if name in INTERNAL_RUNS:
            joined_path = ROOT / INTERNAL_RUNS[name]
            if joined_path.is_file():
                joined = align_historical(rows, HARNESS.load_jsonl(joined_path),
                                           reduced_metadata=name == "historical_biovilt")
                joined_counts, _, _, _ = patient_confusions(rows, {name: joined})
                joined_metrics = metrics_from_counts(joined_counts.sum(axis=0))[0]
                reconciliation[name] = {"raw_available": False,
                                        "legacy_joined_metrics": dict(zip(METRICS, map(float, joined_metrics)))}
                input_files[f"{name}_legacy_joined"] = {"path": str(joined_path.resolve()), "sha256": HARNESS.sha256_file(joined_path)}
                summary_path = joined_path.with_name("summary.json")
                if summary_path.is_file():
                    input_files[f"{name}_legacy_joined"]["historical_scoring_args"] = json.loads(summary_path.read_text()).get("args")
        if not path.is_file():
            reconciliation.setdefault(name, {})["unavailable_reason"] = "Original raw output missing; joined labels are not substituted for raw inference."
            continue
        records = align_historical(rows, HARNESS.load_jsonl(path),
                                   finding_blind=name in GENERIC_ARMS,
                                   reduced_metadata=name == "historical_biovilt")
        arms[name] = records
        input_files[name] = {"path": str(path.resolve()), "sha256": HARNESS.sha256_file(path),
                             "alignment": "Canonical occurrence order plus source/finding/gold; original classifier lacks patient/pair fields." if name == "historical_biovilt" else "Canonical occurrence order plus source/pair/patient/finding/gold; blinded query validated separately."}
        raw_summary = path.with_name("summary.json")
        if raw_summary.is_file():
            input_files[name]["generation_args"] = json.loads(raw_summary.read_text()).get("args")
        if joined is not None:
            differing = [row["row_id"] for row, raw, old in zip(rows, records, joined) if raw["pred_label"] != old["pred_label"]]
            raw_counts, _, _, _ = patient_confusions(rows, {name: records})
            raw_metrics = metrics_from_counts(raw_counts.sum(axis=0))[0]
            reconciliation[name].update({"raw_available": True, "different_prediction_rows": len(differing),
                                         "raw_metrics": dict(zip(METRICS, map(float, raw_metrics))),
                                         "reconciliation_note": "Raw occurrence order supersedes source-ID-joined labels; legacy display retained."})
            input_files[name]["differing_row_ids"] = differing
    return arms, reconciliation, input_files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=BASELINE)
    parser.add_argument("--manifest", type=Path, default=BASELINE / "manifest/evaluation_manifest.jsonl")
    parser.add_argument("--output-dir", type=Path, default=BASELINE / "comparison")
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261020)
    parser.add_argument("--historical-only", action="store_true", help="Reconcile original raw historical outputs while new runs remain in progress; separate filenames")
    args = parser.parse_args()
    rows = HARNESS.read_manifest(args.manifest, 1326)
    if args.historical_only:
        arms, reconciliation, input_files = collect_historical(rows)
        if not all(name in arms for name in (*HISTORICAL_ARMS[1:], *GENERIC_ARMS)):
            raise ValueError("Original raw full-query and finding-blind outputs are required for historical-only analysis")
        summary = build_summary(rows, arms, replicates=args.bootstrap_replicates, seed=args.bootstrap_seed)
        summary["analysis_scope"] = "Original historical raw predictions only; new MCQ outcomes not read."
        summary["historical_display_reconciliation"] = reconciliation
        HARNESS.write_json(args.output_dir / "historical_raw_10k.json", summary)
        HARNESS.write_json(args.output_dir / "historical_internal_manifest.json", {
            "inputs": input_files, "bootstrap_seed": args.bootstrap_seed, "numpy_version": np.__version__,
            "script_sha256": EXECUTING_SCRIPT_SHA256, "inference_performed": False})
        HARNESS.atomic_write(args.output_dir / "HISTORICAL_RAW_REPORT.md", markdown_report(summary))
        print(json.dumps({"historical_full": summary["aggregates"]["qlace_historical_three_runs"],
                          "matched_control": summary["contrasts"]["historical_full_vs_matched_finding_blind"]}, indent=2))
        return
    protocols = {
        "native": HARNESS.validate_protocol(json.loads((BASELINE / "protocol_native_mcq.json").read_text())),
        "squarepad": HARNESS.validate_protocol(json.loads((BASELINE / "protocol_square_pad_mcq.json").read_text())),
    }
    preflight = json.loads((BASELINE / "cpu_preflight.json").read_text())
    for filename in ("protocol_native_mcq.json", "protocol_square_pad_mcq.json"):
        if HARNESS.sha256_file(BASELINE / filename) != preflight["file_sha256"][filename]:
            raise ValueError("Protocol file changed after CPU preflight")
    script_hash = preflight["file_sha256"]["scripts/cmig_medgemma_baseline.py"]
    if HARNESS.sha256_file(ROOT / "scripts/cmig_medgemma_baseline.py") != script_hash:
        raise ValueError("Frozen inference harness changed after preflight")
    arms, provenance, input_files = {}, {}, {}
    for name in NEW_ARMS:
        adapter = ROOT / INTERNAL_ADAPTERS[name] if name in INTERNAL_ADAPTERS else None
        predictions, manifest = load_new_run(rows, args.runs_root / name,
                                             protocols["squarepad" if name == "base_squarepad" else "native"],
                                             adapter, script_hash)
        arms[name], provenance[name] = predictions, manifest
        for filename in ("run_manifest.json", "predictions.jsonl", "metrics.json"):
            path = args.runs_root / name / filename
            input_files[f"{name}/{filename}"] = {"path": str(path.resolve()), "sha256": HARNESS.sha256_file(path)}
    base_contract = provenance["base_native"]["contract"]
    for name in NEW_ARMS:
        contract = provenance[name]["contract"]
        if contract["model"]["files"] != base_contract["model"]["files"]:
            raise ValueError(f"Different backbone checkpoint content in {name}")
        for setting in ("dtype", "attention_implementation", "do_sample", "seed", "use_cache", "cpu_offload"):
            if contract[setting] != base_contract[setting]:
                raise ValueError(f"Unmatched inference contract field: {setting}")
        for base_row, other in zip(arms["base_native"], arms[name]):
            for field in ("prior_image_sha256", "current_image_sha256"):
                if base_row[field] != other[field]:
                    raise ValueError(f"Image content differs across arms: {name}, {base_row['row_id']}")
    historical, reconciliation, historical_files = collect_historical(rows)
    arms.update(historical)
    input_files.update(historical_files)
    summary = build_summary(rows, arms, replicates=args.bootstrap_replicates, seed=args.bootstrap_seed)
    summary["historical_display_reconciliation"] = reconciliation
    internal = {"script_sha256": EXECUTING_SCRIPT_SHA256, "inputs": input_files,
                "run_manifests": provenance, "adapter_paths": INTERNAL_ADAPTERS,
                "bootstrap_seed": args.bootstrap_seed, "numpy_version": np.__version__,
                "source_mapping_reference": "manuscript_rev10/artifacts/rebuttal_analyses/recompute_three_random_seeds.py",
                "manifest_path": str(args.manifest.resolve()), "manifest_file_sha256": HARNESS.sha256_file(args.manifest),
                "inference_performed": False}
    HARNESS.write_json(args.output_dir / "comparison.json", summary)
    HARNESS.write_json(args.output_dir / "internal_manifest.json", internal)
    HARNESS.atomic_write(args.output_dir / "REPORT.md", markdown_report(summary))
    print(json.dumps({"output_dir": str(args.output_dir), "n_rows": len(rows),
                      "primary": summary["contrasts"]["primary_adaptation_same_contract"]}, indent=2))


if __name__ == "__main__":
    main()
