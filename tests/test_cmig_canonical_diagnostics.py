from copy import deepcopy

import numpy as np
import pytest

from scripts import cmig_canonical_diagnostics as mod


def reference(prior="a", finding="edema", source="collision", label="improving"):
    return {"source_id": source, "pair_id": "pair", "patient_id": "p", "prior_image_path": prior,
            "current_image_path": "current", "query_finding": finding, "query_change": label,
            "query": finding, "problem": "question: " + finding}


def raw(ref, prediction):
    return {k: v for k, v in {**ref, "pred_label": prediction}.items() if not k.endswith("image_path")}


def test_raw_alignment_preserves_source_collisions_and_occurrences():
    refs = [reference(), reference("b"), reference()]
    aligned = mod.align_raw(refs, [raw(r, p) for r, p in zip(refs, ("improving", "stable", "worsening"))])
    assert [r["pred_label"] for r in aligned] == ["improving", "stable", "worsening"]
    assert len({r["physical_occurrence_sha256"] for r in aligned}) == 3
    assert "image_path" not in str(aligned)


def test_alignment_rejects_missing_rows_order_and_physical_mismatch():
    refs = [reference(source="a"), reference(source="b")]
    with pytest.raises(ValueError, match="row count"):
        mod.align_raw(refs, [raw(refs[0], "stable")])
    with pytest.raises(ValueError, match="Source/order"):
        mod.align_raw(refs, [raw(r, "stable") for r in reversed(refs)])
    predictions = deepcopy(refs)
    predictions[0]["prior_image_path"] = "wrong"
    with pytest.raises(ValueError, match="Physical-image"):
        mod.align_raw(refs, predictions)


def control(refs, i, j):
    value = deepcopy(refs[j])
    for role, index in (("original", i), ("target", j)):
        value[f"query_swap_{role}_input_row_index"] = index
        for suffix, field in (("source_id", "source_id"), ("finding", "query_finding"), ("change", "query_change")):
            value[f"query_swap_{role}_{suffix}"] = refs[index][field]
    return value


def test_swap_uses_validated_row_indices_not_colliding_source_id():
    refs = [reference("other"), reference(), reference(finding="pneumonia", label="worsening")]
    predictions = mod.align_raw(refs, [raw(r, p) for r, p in zip(refs, ("stable", "improving", "worsening"))])
    joined = mod.join_swaps([control(refs, 1, 2)], refs, predictions)
    assert joined[0]["both_correct"] and joined[0]["prediction_changed"]
    assert joined[0]["original_row_index"] == 1
    broken = control(refs, 0, 2)
    with pytest.raises(ValueError, match="physical-image"):
        mod.join_swaps([broken], refs, predictions)


def test_supported_macro_f1_counts_out_of_set_predictions_as_errors():
    matrix = np.zeros((3, 4), dtype=int)
    matrix[0, 3] = matrix[1, 1] = matrix[2, 2] = 1
    assert mod.metrics(matrix).tolist() == pytest.approx([2 / 3, 2 / 3, 1])


def test_missing_raw_identity_requires_explicit_provenance_exception():
    ref = reference()
    prediction = {"source_id": ref["source_id"], "query_finding": "edema", "gold_label": "improving", "pred_label": "stable"}
    with pytest.raises(ValueError, match="patient_id"):
        mod.align_raw([ref], [prediction])
    assert mod.align_raw([ref], [prediction], allow_missing_identity=True)[0]["pred_label"] == "stable"


def test_patient_bootstrap_keeps_patient_groups_and_exact_counts():
    rows = [{"patient_id": "a", "correct": True}] * 3 + [{"patient_id": "b", "correct": False}]
    result = mod.binary_metric(rows, "correct", 100, 42)
    assert result["numerator"] == 3 and result["denominator"] == 4
    assert result["patient_clusters"] == 2
    assert result["patient_bootstrap_95_ci"] == [0.0, 1.0]
    assert mod.binary_metric([], "correct", 100, 42) is None


def test_joint_correctness_is_not_compliance_or_invariance():
    refs = [reference(), reference(finding="pneumonia", label="worsening")]
    predictions = mod.align_raw(refs, [raw(refs[0], "stable"), raw(refs[1], "worsening")])
    swaps = mod.join_swaps([control(refs, 0, 1)], refs, predictions)
    result = mod.swap_metrics(swaps, 0, 42)
    assert result["metrics"]["required_change_compliance"]["estimate"] == 1
    assert result["metrics"]["paired_target_success"]["estimate"] == 0


def test_complementarity_counts_all_five_patterns_and_refuses_misalignment():
    refs = [reference(source=str(i)) for i in range(5)]
    q = mod.align_raw(refs, [raw(r, p) for r, p in zip(refs, ("improving", "improving", "stable", "stable", "stable"))])
    bio = mod.align_raw(refs, [raw(r, p) for r, p in zip(refs, ("improving", "stable", "improving", "stable", "worsening"))])
    result = mod.complementarity(q, bio)
    assert list(result["counts"].values()) == [1, 1, 1, 1, 1]
    assert result["either_correct_oracle"]["correct_count"] == 3
    assert result["either_correct_oracle"]["accuracy"] == pytest.approx(.6)
    with pytest.raises(ValueError, match="alignment"):
        mod.complementarity(q, list(reversed(bio)))


def test_legacy_transition_success_includes_same_label_cells():
    swaps = [{"original_gold": "stable", "target_gold": "stable", "different_label": False,
              "target_correct": True, "both_correct": True}]
    old = {"ordered_finding_pairs": [], "gold_transitions": [{"original_gold": "stable", "target_gold": "stable", "metrics": {"paired_target_success": {"numerator": 1, "denominator": 1, "estimate": 1.0}}}]}
    result = mod.compare_legacy_swap_points(old, swaps)
    assert result["point_entries_checked"] == 1
    assert result["all_point_estimates_unchanged"]
