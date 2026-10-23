"""Synthetic arithmetic and partial-package rejection checks."""
from copy import deepcopy
import json

import pytest

from tools.verify_aggregate_tables import verify_sparse_metrics, verify_candidate_behavior, verify_additional_results


def test_sparse_confusion_counts_include_abstentions_and_invalids():
    record = {"count": 6, "correct_count": 3, "accuracy": 0.5,
              "supported_macro_f1": 2 / 3,
              "confusion_matrix": {"improving": {"improving": 1, "no_supported_evidence": 1},
                                   "stable": {"stable": 1, "INVALID": 1},
                                   "worsening": {"worsening": 1, "other": 1}}}
    verify_sparse_metrics(record, "synthetic")
    bad = deepcopy(record)
    bad["accuracy"] = 1.0
    with pytest.raises(ValueError, match="arithmetic mismatch"):
        verify_sparse_metrics(bad, "synthetic")


def test_candidate_behavior_checks_counts_without_a_reference_f1():
    record = {"count": 4, "prediction_counts": {"stable": 3, "no_supported_evidence": 1},
              "abstention_count": 1, "abstention_rate": 0.25,
              "supported_output_count": 3, "supported_output_rate": 0.75}
    verify_candidate_behavior(record, "synthetic")
    record["abstention_rate"] = 0.5
    with pytest.raises(ValueError, match="arithmetic mismatch"):
        verify_candidate_behavior(record, "synthetic")


def test_partial_expanded_result_set_is_rejected(tmp_path):
    folder = tmp_path / "aggregates"
    folder.mkdir()
    (folder / "query_interventions_and_scale.json").write_text(json.dumps({}))
    with pytest.raises(ValueError, match="incomplete"):
        verify_additional_results(tmp_path)
