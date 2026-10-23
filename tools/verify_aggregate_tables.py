#!/usr/bin/env python3
"""Recompute point metrics and run summaries from bundled aggregate counts.

Patient-bootstrap intervals are copied from the study's aggregate record, not
regenerated: aggregate confusion matrices contain no patient-level resamples.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
from pathlib import Path
from statistics import fmean, stdev
try:
    from .paper_run_scope import validate as validate_paper_run_scope
except ImportError:  # Direct execution: python tools/verify_aggregate_tables.py
    from paper_run_scope import validate as validate_paper_run_scope


def equal(actual: float, expected: float, name: str) -> None:
    if not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12):
        raise ValueError(f"Aggregate arithmetic mismatch: {name}")


def metrics(matrix: list[list[int]]) -> dict[str, float]:
    if (len(matrix) != 3 or any(len(row) != 5 for row in matrix)
            or any(not isinstance(v, int) or v < 0 for row in matrix for v in row)):
        raise ValueError("Expected nonnegative integer counts for three targets and five outputs")
    support = [sum(row) for row in matrix]
    if min(support) <= 0:
        raise ValueError("All three target classes must have support")
    predicted = [sum(row[i] for row in matrix) for i in range(3)]
    return {
        "accuracy": sum(matrix[i][i] for i in range(3)) / sum(support),
        "supported_macro_f1": fmean(2 * matrix[i][i] / (support[i] + predicted[i]) for i in range(3)),
        "class_balanced_accuracy": fmean(matrix[i][i] / support[i] for i in range(3)),
    }


def verify(payload: dict) -> dict[str, int | str]:
    validate_paper_run_scope(payload)
    arms = payload["arms"]
    for name, arm in arms.items():
        pooled = [[0] * 5 for _ in range(3)]
        finding_balanced = []
        if set(arm["per_finding"]) != set(payload["findings"]):
            raise ValueError(f"Finding population mismatch: {name}")
        for finding, record in arm["per_finding"].items():
            if (record["gold_rows"] != ["improving", "stable", "worsening"]
                    or record["prediction_columns"] != ["improving", "stable", "worsening", "no_supported_evidence", "INVALID"]):
                raise ValueError("Unsupported confusion-matrix label order")
            matrix = record["confusion_matrix"]
            calculated = metrics(matrix)
            equal(sum(map(sum, matrix)), record["n_rows"], f"{name}/{finding}/rows")
            for metric, value in calculated.items():
                equal(value, record[metric], f"{name}/{finding}/{metric}")
            finding_balanced.append(calculated["class_balanced_accuracy"])
            for i in range(3):
                for j in range(5):
                    pooled[i][j] += matrix[i][j]
        calculated = metrics(pooled)
        calculated["pooled_class_balanced_accuracy"] = calculated.pop("class_balanced_accuracy")
        calculated["mean_finding_class_balanced_accuracy"] = fmean(finding_balanced)
        equal(sum(map(sum, pooled)), payload["n_rows"], f"{name}/rows")
        for metric, value in calculated.items():
            equal(value, arm["metrics"][metric]["estimate"], f"{name}/{metric}")

    def values(names: list[str], metric: str) -> list[float]:
        return [arms[name]["metrics"][metric]["estimate"] for name in names]

    for name, entry in payload["aggregates"].items():
        if entry["run_count"] != len(entry["arms"]):
            raise ValueError(f"Run-count mismatch: {name}")
        for metric, record in entry["metrics"].items():
            observations = values(entry["arms"], metric)
            equal(fmean(observations), record["estimate"], f"{name}/{metric}/mean")
            equal(stdev(observations), entry["sample_sd"][metric], f"{name}/{metric}/sample SD")
    for name, entry in payload["contrasts"].items():
        for metric, record in entry["metrics"].items():
            left, right = values(entry["left_arms"], metric), values(entry["right_arms"], metric)
            equal(fmean(left) - fmean(right), record["estimate"], f"{name}/{metric}/difference")
            if "observed_run_sample_sd" in entry:
                if len(right) == 1:
                    right *= len(left)
                if len(left) != len(right):
                    raise ValueError(f"Unequal paired run populations: {name}")
                equal(stdev(a - b for a, b in zip(left, right)), entry["observed_run_sample_sd"][metric],
                      f"{name}/{metric}/difference sample SD")
    return {"arms": len(arms), "finding_matrices": len(arms) * len(payload["findings"]),
            "run_aggregates": len(payload["aggregates"]), "contrasts": len(payload["contrasts"]),
            "interval_scope": "Stored patient-bootstrap intervals copied, not regenerated"}


def table_csv(payload: dict) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(("comparison", "metric", "estimate", "ci_lower_95", "ci_upper_95", "sample_sd"))
    for family in ("aggregates", "arms", "contrasts"):
        for name, entry in sorted(payload[family].items()):
            for metric, value in sorted(entry["metrics"].items()):
                writer.writerow((name, metric, value["estimate"], *value.get("ci_95", ("", "")),
                                 entry.get("sample_sd", {}).get(metric, "")))
    return stream.getvalue().encode()


def verify_sparse_metrics(record: dict, name: str) -> None:
    """Check supported-label metrics from a named aggregate confusion matrix."""
    labels = ("improving", "stable", "worsening")
    confusion = record["confusion_matrix"]
    if not isinstance(confusion, dict) or set(confusion) - set(labels):
        raise ValueError(f"Unexpected target labels: {name}")
    matrix = []
    for gold in labels:
        row = confusion.get(gold, {})
        if any(not isinstance(value, int) or value < 0 for value in row.values()):
            raise ValueError(f"Invalid aggregate counts: {name}")
        matrix.append([*(row.get(predicted, 0) for predicted in labels),
                       row.get("no_supported_evidence", 0),
                       sum(value for label, value in row.items()
                           if label not in (*labels, "no_supported_evidence"))])
    calculated = metrics(matrix)
    total = sum(map(sum, matrix))
    equal(total, record["count"], f"{name}/count")
    for metric in ("accuracy", "supported_macro_f1"):
        equal(calculated[metric], record[metric], f"{name}/{metric}")
    if "correct_count" in record:
        equal(sum(matrix[i][i] for i in range(3)), record["correct_count"], f"{name}/correct count")


def verify_candidate_behavior(record: dict, name: str) -> None:
    counts, total = record["prediction_counts"], record["count"]
    if not isinstance(total, int) or total <= 0 or any(
            not isinstance(value, int) or value < 0 for value in counts.values()):
        raise ValueError(f"Invalid candidate counts: {name}")
    equal(sum(counts.values()), total, f"{name}/candidate count")
    abstentions = counts.get("no_supported_evidence", 0)
    supported = sum(counts.get(label, 0) for label in ("improving", "stable", "worsening"))
    for key, expected in (("abstention_count", abstentions), ("abstention_rate", abstentions / total),
                          ("supported_output_count", supported), ("supported_output_rate", supported / total)):
        equal(expected, record[key], f"{name}/{key}")


def verify_additional_results(root: Path) -> dict:
    files = ("query_interventions_and_scale.json", "catalogue_and_frozen_heads.json",
             "secondary_diagnostics.json", "temporal_details.json", "data_training_runtime.json")
    paths = [root / "aggregates" / name for name in files]
    if not any(path.exists() for path in paths):
        return {"additional_classification_records": 0, "candidate_behavior_records": 0}
    if not all(path.is_file() for path in paths):
        raise ValueError("Expanded manuscript result set is incomplete")
    classification_count = behavior_count = 0

    def walk(value, name):
        nonlocal classification_count, behavior_count
        if isinstance(value, dict):
            if {"confusion_matrix", "count", "accuracy", "supported_macro_f1"} <= value.keys():
                verify_sparse_metrics(value, name)
                classification_count += 1
            if {"prediction_counts", "count", "abstention_rate", "supported_output_rate"} <= value.keys():
                verify_candidate_behavior(value, name)
                behavior_count += 1
            for key, item in value.items():
                walk(item, f"{name}/{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, f"{name}/{index}")

    for path in paths:
        walk(json.loads(path.read_text()), path.name)
    data = json.loads(paths[-1].read_text())
    for row in data["distributions"]:
        equal(sum(row["label_counts"].values()), row["rows"], f"{row['name']}/label count")
        equal(sum(row["finding_counts"].values()), row["rows"], f"{row['name']}/finding count")
    trainability = data["configured_trainability"]
    equal(sum(trainability["parameters_by_category"].values()), trainability["trainable_unique_parameters"],
          "configured trainable parameter sum")
    compute = data["historical_compute"]
    equal(sum(row["run_count"] for row in compute["groups"].values()), compute["adapter_run_count"],
          "reported adapter run count")
    equal(sum(row["single_gpu_hours"] for row in compute["groups"].values()), compute["adapter_single_gpu_hours"],
          "reported adapter hours")
    return {"additional_classification_records": classification_count, "candidate_behavior_records": behavior_count,
            "additional_result_files": len(paths)}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=root / "aggregates/baseline_comparison.json")
    parser.add_argument("--check-csv", type=Path, default=root / "aggregates/baseline_metrics.csv")
    parser.add_argument("--output", type=Path, help="Optional regenerated CSV; preserve packaged inputs")
    args = parser.parse_args()
    payload = json.loads(args.source.read_text())
    result = verify(payload)
    result.update(verify_additional_results(args.source.parent.parent))
    data = table_csv(payload)
    if args.check_csv.read_bytes() != data:
        raise ValueError("Bundled CSV differs from the aggregate JSON projection")
    if args.output:
        if args.output.resolve() in (args.source.resolve(), args.check_csv.resolve()):
            parser.error("Use a separate output path to preserve packaged inputs")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(data)
    print(json.dumps({"status": "passed", **result}, indent=2))


if __name__ == "__main__":
    main()
