"""Helpers for benchmark-facing longitudinal classification metrics."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
import math
import random
from typing import Any


DEFAULT_PROGRESS_LABELS = ("improving", "stable", "worsening")
DEFAULT_UNSUPPORTED_LABEL = "no_supported_evidence"


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def _record_value(record: Any, key: str) -> Any:
    if isinstance(record, dict):
        return record[key]
    return getattr(record, key)


def f1_from_counts(*, true_positive: int, false_positive: int, false_negative: int) -> float:
    precision = safe_divide(true_positive, true_positive + false_positive)
    recall = safe_divide(true_positive, true_positive + false_negative)
    if precision + recall == 0:
        return 0.0
    return float((2.0 * precision * recall) / (precision + recall))


def infer_label_space(
    gold_labels: Sequence[str],
    pred_labels: Sequence[str] | None = None,
    *,
    preferred_order: Sequence[str] = DEFAULT_PROGRESS_LABELS + (DEFAULT_UNSUPPORTED_LABEL,),
) -> tuple[str, ...]:
    observed = {label for label in gold_labels if label}
    if pred_labels is not None:
        observed.update(label for label in pred_labels if label)
    ordered = [label for label in preferred_order if label in observed]
    extras = sorted(label for label in observed if label not in preferred_order)
    return tuple(ordered + extras)


def compute_multiclass_metrics(
    gold_labels: Sequence[str],
    pred_labels: Sequence[str],
    *,
    labels: Sequence[str] | None = None,
    unsupported_label: str = DEFAULT_UNSUPPORTED_LABEL,
) -> dict[str, Any]:
    if len(gold_labels) != len(pred_labels):
        raise ValueError("gold_labels and pred_labels must have the same length")
    label_space = tuple(labels or infer_label_space(gold_labels, pred_labels))
    label_set = set(label_space)
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    invalid_prediction_count = 0

    for gold, pred in zip(gold_labels, pred_labels):
        confusion[gold][pred] += 1
        if pred not in label_set:
            invalid_prediction_count += 1

    per_class: dict[str, dict[str, float | int]] = {}
    for label in label_space:
        true_positive = sum(1 for gold, pred in zip(gold_labels, pred_labels) if gold == label and pred == label)
        false_positive = sum(1 for gold, pred in zip(gold_labels, pred_labels) if gold != label and pred == label)
        false_negative = sum(1 for gold, pred in zip(gold_labels, pred_labels) if gold == label and pred != label)
        support = sum(1 for gold in gold_labels if gold == label)
        pred_count = sum(1 for pred in pred_labels if pred == label)
        precision = safe_divide(true_positive, true_positive + false_positive)
        recall = safe_divide(true_positive, true_positive + false_negative)
        per_class[label] = {
            "support": support,
            "pred_count": pred_count,
            "precision": precision,
            "recall": recall,
            "f1": f1_from_counts(
                true_positive=true_positive,
                false_positive=false_positive,
                false_negative=false_negative,
            ),
        }

    accuracy = safe_divide(
        sum(1 for gold, pred in zip(gold_labels, pred_labels) if gold == pred),
        len(gold_labels),
    )
    macro_f1 = safe_divide(
        sum(float(stats["f1"]) for stats in per_class.values()),
        len(label_space),
    ) if label_space else 0.0
    unsupported_positive_rate: float | None = None
    unsupported_support = sum(1 for gold in gold_labels if gold == unsupported_label)
    if unsupported_support > 0:
        unsupported_positive_rate = safe_divide(
            sum(1 for gold, pred in zip(gold_labels, pred_labels) if gold == unsupported_label and pred != unsupported_label),
            unsupported_support,
        )

    return {
        "count": len(gold_labels),
        "labels": list(label_space),
        "accuracy": accuracy,
        "exact_match_accuracy": accuracy,
        "macro_f1": macro_f1,
        "invalid_prediction_rate": safe_divide(invalid_prediction_count, len(gold_labels)),
        "label_counts": dict(Counter(gold_labels)),
        "prediction_counts": dict(Counter(pred_labels)),
        "per_class": per_class,
        "confusion_matrix": {
            gold: dict(pred_counts) for gold, pred_counts in sorted(confusion.items())
        },
        "unsupported_positive_rate": unsupported_positive_rate,
    }


def multiclass_accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    return compute_multiclass_metrics(y_true, y_pred)["accuracy"]


def multiclass_per_class_f1(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    *,
    labels: Sequence[str] | None = None,
) -> dict[str, float]:
    summary = compute_multiclass_metrics(y_true, y_pred, labels=labels)
    return {label: float(stats["f1"]) for label, stats in summary["per_class"].items()}


def multiclass_macro_f1(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    *,
    labels: Sequence[str] | None = None,
) -> float:
    return compute_multiclass_metrics(y_true, y_pred, labels=labels)["macro_f1"]


def unsupported_positive_rate(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    *,
    labels: Sequence[str] | None = None,
    unsupported_label: str = DEFAULT_UNSUPPORTED_LABEL,
) -> float | None:
    return compute_multiclass_metrics(
        y_true,
        y_pred,
        labels=labels,
        unsupported_label=unsupported_label,
    )["unsupported_positive_rate"]


def multiclass_summary(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    *,
    labels: Sequence[str] | None = None,
    unsupported_label: str = DEFAULT_UNSUPPORTED_LABEL,
) -> dict[str, Any]:
    return compute_multiclass_metrics(
        y_true,
        y_pred,
        labels=labels,
        unsupported_label=unsupported_label,
    )


def summarize_by_finding(
    records: Iterable[dict[str, Any]],
    *,
    gold_key: str = "gold_label",
    pred_key: str = "pred_label",
    finding_key: str = "query_finding",
    labels: Sequence[str] | None = None,
    unsupported_label: str = DEFAULT_UNSUPPORTED_LABEL,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record[finding_key])].append(record)
    summary: dict[str, dict[str, Any]] = {}
    for finding, finding_records in sorted(grouped.items()):
        summary[finding] = compute_multiclass_metrics(
            [str(record[gold_key]) for record in finding_records],
            [str(record[pred_key]) for record in finding_records],
            labels=labels,
            unsupported_label=unsupported_label,
        )
    return summary


def per_finding_accuracy(
    records: Iterable[Any],
    *,
    finding_key: str = "finding",
    true_key: str = "y_true",
    pred_key: str = "y_pred",
) -> dict[str, float]:
    grouped: dict[str, list[Any]] = defaultdict(list)
    for record in records:
        grouped[str(_record_value(record, finding_key))].append(record)
    return {
        finding: multiclass_accuracy(
            [str(_record_value(record, true_key)) for record in grouped_records],
            [str(_record_value(record, pred_key)) for record in grouped_records],
        )
        for finding, grouped_records in sorted(grouped.items())
    }


def per_finding_macro_f1(
    records: Iterable[Any],
    *,
    finding_key: str = "finding",
    true_key: str = "y_true",
    pred_key: str = "y_pred",
    labels: Sequence[str] | None = None,
) -> dict[str, float]:
    grouped: dict[str, list[Any]] = defaultdict(list)
    for record in records:
        grouped[str(_record_value(record, finding_key))].append(record)
    return {
        finding: multiclass_macro_f1(
            [str(_record_value(record, true_key)) for record in grouped_records],
            [str(_record_value(record, pred_key)) for record in grouped_records],
            labels=labels,
        )
        for finding, grouped_records in sorted(grouped.items())
    }


def summarize_classification_records(
    records: Sequence[dict[str, Any]],
    *,
    gold_key: str = "gold_label",
    pred_key: str = "pred_label",
    finding_key: str = "query_finding",
    labels: Sequence[str] | None = None,
    unsupported_label: str = DEFAULT_UNSUPPORTED_LABEL,
) -> dict[str, Any]:
    gold_labels = [str(record[gold_key]) for record in records]
    pred_labels = [str(record[pred_key]) for record in records]
    resolved_labels = tuple(labels or infer_label_space(gold_labels, pred_labels))
    overall = compute_multiclass_metrics(
        gold_labels,
        pred_labels,
        labels=resolved_labels,
        unsupported_label=unsupported_label,
    )
    return {
        **overall,
        "by_finding": summarize_by_finding(
            records,
            gold_key=gold_key,
            pred_key=pred_key,
            finding_key=finding_key,
            labels=resolved_labels,
            unsupported_label=unsupported_label,
        ),
    }


def _quantile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("sorted_values must not be empty")
    if q <= 0.0:
        return float(sorted_values[0])
    if q >= 1.0:
        return float(sorted_values[-1])
    position = q * (len(sorted_values) - 1)
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return float(sorted_values[lower_index])
    lower_value = float(sorted_values[lower_index])
    upper_value = float(sorted_values[upper_index])
    fraction = position - lower_index
    return lower_value + ((upper_value - lower_value) * fraction)


def bootstrap_classification_confidence_intervals(
    records: Sequence[dict[str, Any]],
    *,
    gold_key: str = "gold_label",
    pred_key: str = "pred_label",
    labels: Sequence[str] | None = None,
    unsupported_label: str = DEFAULT_UNSUPPORTED_LABEL,
    unit_key: str = "patient_id",
    num_samples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "enabled": False,
        "unit_key": unit_key,
        "num_samples": int(num_samples),
        "confidence_level": float(confidence_level),
        "seed": int(seed),
    }
    if num_samples <= 0:
        result["reason"] = "disabled"
        return result
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")

    unit_to_records: dict[str, list[dict[str, Any]]] = defaultdict(list)
    missing_unit_record_count = 0
    for record in records:
        raw_unit = record.get(unit_key)
        unit = None if raw_unit is None else str(raw_unit).strip()
        if not unit:
            missing_unit_record_count += 1
            continue
        unit_to_records[unit].append(record)

    result["missing_unit_record_count"] = missing_unit_record_count
    result["num_units"] = len(unit_to_records)
    if missing_unit_record_count > 0:
        result["reason"] = "missing_unit_ids"
        return result
    if len(unit_to_records) < 2:
        result["reason"] = "insufficient_unique_units"
        return result

    gold_labels = [str(record[gold_key]) for record in records]
    pred_labels = [str(record[pred_key]) for record in records]
    resolved_labels = tuple(labels or infer_label_space(gold_labels, pred_labels))
    observed = compute_multiclass_metrics(
        gold_labels,
        pred_labels,
        labels=resolved_labels,
        unsupported_label=unsupported_label,
    )

    unit_ids = sorted(unit_to_records)
    rng = random.Random(seed)
    accuracy_values: list[float] = []
    macro_f1_values: list[float] = []
    for _ in range(num_samples):
        sampled_records: list[dict[str, Any]] = []
        for _ in range(len(unit_ids)):
            sampled_records.extend(unit_to_records[rng.choice(unit_ids)])
        sampled_summary = compute_multiclass_metrics(
            [str(record[gold_key]) for record in sampled_records],
            [str(record[pred_key]) for record in sampled_records],
            labels=resolved_labels,
            unsupported_label=unsupported_label,
        )
        accuracy_values.append(float(sampled_summary["accuracy"]))
        macro_f1_values.append(float(sampled_summary["macro_f1"]))

    accuracy_values.sort()
    macro_f1_values.sort()
    alpha = 1.0 - confidence_level
    lower_q = alpha / 2.0
    upper_q = 1.0 - lower_q

    result["enabled"] = True
    result["metrics"] = {
        "accuracy": {
            "observed": float(observed["accuracy"]),
            "bootstrap_mean": safe_divide(sum(accuracy_values), len(accuracy_values)),
            "ci_lower": _quantile(accuracy_values, lower_q),
            "ci_upper": _quantile(accuracy_values, upper_q),
        },
        "macro_f1": {
            "observed": float(observed["macro_f1"]),
            "bootstrap_mean": safe_divide(sum(macro_f1_values), len(macro_f1_values)),
            "ci_lower": _quantile(macro_f1_values, lower_q),
            "ci_upper": _quantile(macro_f1_values, upper_q),
        },
    }
    return result
