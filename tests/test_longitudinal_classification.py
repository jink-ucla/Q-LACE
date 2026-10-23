from __future__ import annotations

from dataclasses import dataclass
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from medgemma_visual_rft.longitudinal_classification import (  # noqa: E402
    multiclass_accuracy,
    multiclass_macro_f1,
    multiclass_per_class_f1,
    multiclass_summary,
    per_finding_accuracy,
    per_finding_macro_f1,
    unsupported_positive_rate,
)


@dataclass(frozen=True)
class FindingRecord:
    finding: str
    y_true: str
    y_pred: str


def test_multiclass_summary_tracks_accuracy_and_macro_f1() -> None:
    y_true = ["improving", "stable", "worsening", "stable"]
    y_pred = ["improving", "worsening", "worsening", "stable"]

    summary = multiclass_summary(y_true, y_pred, labels=["improving", "stable", "worsening"])

    assert summary["count"] == 4
    assert summary["accuracy"] == 0.75
    assert summary["exact_match_accuracy"] == 0.75
    assert round(summary["macro_f1"], 6) == round((1.0 + 0.6666666667 + 0.6666666667) / 3.0, 6)
    assert summary["invalid_prediction_rate"] == 0.0


def test_per_class_f1_works_without_extra_labels() -> None:
    y_true = ["stable", "stable", "worsening", "improving"]
    y_pred = ["stable", "worsening", "worsening", "improving"]

    per_class = multiclass_per_class_f1(y_true, y_pred)

    assert per_class["stable"] == 2.0 / 3.0
    assert per_class["worsening"] == 2.0 / 3.0
    assert per_class["improving"] == 1.0
    assert multiclass_accuracy(y_true, y_pred) == 0.75
    assert round(multiclass_macro_f1(y_true, y_pred), 6) == round((2.0 / 3.0 + 2.0 / 3.0 + 1.0) / 3.0, 6)


def test_unsupported_positive_rate_returns_none_when_class_absent() -> None:
    y_true = ["stable", "worsening"]
    y_pred = ["stable", "improving"]

    assert unsupported_positive_rate(y_true, y_pred) is None


def test_unsupported_positive_rate_counts_false_supported_predictions() -> None:
    y_true = ["no_supported_evidence", "no_supported_evidence", "stable"]
    y_pred = ["stable", "no_supported_evidence", "stable"]

    assert unsupported_positive_rate(y_true, y_pred) == 0.5


def test_per_finding_helpers_accept_mapping_and_attribute_records() -> None:
    mapping_records = [
        {"finding": "edema", "y_true": "stable", "y_pred": "stable"},
        {"finding": "edema", "y_true": "worsening", "y_pred": "stable"},
    ]
    attr_records = [
        FindingRecord(finding="pneumonia", y_true="worsening", y_pred="worsening"),
        FindingRecord(finding="pneumonia", y_true="stable", y_pred="worsening"),
    ]

    accuracy = per_finding_accuracy([*mapping_records, *attr_records])
    macro_f1 = per_finding_macro_f1(
        [*mapping_records, *attr_records],
        labels=["stable", "worsening"],
    )

    assert accuracy == {"edema": 0.5, "pneumonia": 0.5}
    assert round(macro_f1["edema"], 6) == round((2.0 / 3.0 + 0.0) / 2.0, 6)
    assert round(macro_f1["pneumonia"], 6) == round((0.0 + 2.0 / 3.0) / 2.0, 6)
