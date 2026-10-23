from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("cmig_temporal_validation", ROOT / "scripts/cmig_temporal_validation.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def row(prior="prior-a", source="same", label="worsening", condition="baseline"):
    return {"source_id": source, "patient_id": "p", "prior_image_path": prior,
            "current_image_path": "current", "query_finding": "pleural effusion",
            "query_change": label, "control_name": condition, "pred_label": label}


def test_physical_keys_preserve_distinct_priors_and_identical_control_occurrences():
    rows = [row(), row("prior-b")]
    assert len(set(module.occurrence_keys(rows))) == 2
    assert len(set(module.occurrence_keys([rows[0], deepcopy(rows[0])]))) == 2
    audit = module.source_collision_audit(rows)
    assert audit["rows_lost_by_source_id_deduplication"] == 1
    assert len(audit["collision_groups"][0]["distinct_prior_images"]) == 2


def test_align_preserves_collision_predictions_instead_of_last_write_wins():
    manifest = [row(), row("prior-b", label="stable")]
    predictions = [{"source_id": "same", "query_finding": "pleural_effusion", "gold_label": r["query_change"], "pred_label": pred}
                   for r, pred in zip(manifest, ["improving", "stable"])]
    assert [r["pred_label"] for r in module.align_manifest(manifest, predictions)] == ["improving", "stable"]
    predictions.pop()
    with pytest.raises(ValueError, match="Row count"):
        module.align_manifest(manifest, predictions)


def test_alignment_rejects_order_and_physical_mismatch():
    manifest = [row(source="a"), row(source="b")]
    with pytest.raises(ValueError, match="order/source"):
        module.align_manifest(manifest, list(reversed(manifest)))
    wrong = deepcopy(manifest)
    wrong[0]["prior_image_path"] = "wrong"
    with pytest.raises(ValueError, match="Physical image"):
        module.align_manifest(manifest, wrong)


def test_supported_f1_keeps_nse_errors_and_legacy_fixed_four_classes():
    rows = [row(label=label) for label in module.SUPPORTED]
    rows[0]["pred_label"] = module.NSE
    result = module.classification(rows)
    assert result["count"] == 3
    assert result["supported_macro_f1"] == pytest.approx(2 / 3)
    assert result["legacy_four_class_macro_f1"] == pytest.approx(1 / 2)


def test_nse_column_is_four_class_macro_f1_not_nse_f1():
    rows = [{"pred_label": module.NSE}] * 211 + [{"pred_label": "worsening"}] * 289
    result = module.nse_counts(rows)
    assert result["abstention_rate"] == pytest.approx(.422)
    assert result["synthetic_nse_class_f1_not_clinical_performance"] == pytest.approx(422 / 711)
    assert result["legacy_four_class_macro_f1_not_abstention_f1"] == pytest.approx(.14838255977496484)


def test_reversal_pairs_all_occurrences_and_duplicate_denominators_are_explicit():
    canonical = [row(), row("prior-b", label="stable")]
    controls = []
    for condition in module.CONDITIONS:
        for anchor in canonical:
            item = deepcopy(anchor)
            item["control_name"] = condition
            item["control_base_source_id"] = anchor["source_id"]
            if condition == "time_reversal":
                item["prior_image_path"], item["current_image_path"] = item["current_image_path"], item["prior_image_path"]
                item["query_change"] = module.reverse(item["query_change"])
                item["pred_label"] = item["query_change"]
            if condition == "duplicate_pair":
                item["pred_label"] = "stable"
                item["prior_image_path"] = item["current_image_path"]
            controls.append(item)
    result = module.temporal_metrics(controls, canonical)
    assert result["temporal_reversal_consistency"]["count"] == 2
    assert result["temporal_reversal_consistency"]["rate"] == 1
    assert result["duplicate_pair"]["count"] == 2
    assert result["duplicate_pair"]["legacy_stable_recall_denominator"] == 1
    assert result["duplicate_pair"]["stable_output_rate_all_rows"] == 1


def test_actual_path_takes_precedence_over_stale_control_dicom_id():
    item = row()
    item["prior_dicom_id"] = "original-prior-dicom"
    item["prior_image_path"] = "blank-image.jpg"
    assert module.image_identity(item, True) == "blank-image.jpg"


def test_prediction_comparison_separates_collisions_from_independent_inference_changes():
    reference = [row(source="unique"), row(source="collision"), row("prior-b", source="collision")]
    left = deepcopy(reference)
    right = deepcopy(reference)
    right[0]["pred_label"] = "stable"
    right[1]["pred_label"] = "stable"
    for item in right:
        item["source_id"] += "::control::baseline"
    result = module.compare_prediction_vectors(reference, left, right)
    assert result["different_prediction_count"] == 2
    assert result["differences_with_colliding_source_id"] == 1
    assert result["differences_with_unique_source_id"] == 1
