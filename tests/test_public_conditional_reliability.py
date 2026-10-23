"""Synthetic tests for portable occurrence-preserving public workflows."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.build_query_swap_controls import build_swaps, read_manifest, reindex
from tools.analyze_conditional_reliability import (
    CONDITIONS, INVALID, NSE, checked_predictions, prepare_temporal, query_metrics,
    reverse, temporal_metrics,
)


def manifest(specs=None):
    specs = specs or [
        ("example-person-a", "prior-a.png", "current-a.png", "edema", "improving"),
        ("example-person-a", "prior-a.png", "current-a.png", "consolidation", "worsening"),
        ("example-person-b", "prior-b.png", "current-b.png", "edema", "stable"),
        ("example-person-b", "prior-b.png", "current-b.png", "consolidation", "stable"),
    ]
    return [reindex({"patient_id": patient, "prior_image_path": prior,
                     "current_image_path": current, "query_finding": finding,
                     "target_label": gold, "source_id": "example-colliding-source",
                     "pair_id": "example-reused-pair"}, index)
            for index, (patient, prior, current, finding, gold) in enumerate(specs)]


def records(rows, labels=None):
    labels = labels if labels is not None else [row["target_label"] for row in rows]
    return [{**row, "pred_label": label} for row, label in zip(rows, labels)]


def predictions(rows, labels=None):
    return checked_predictions(rows, records(rows, labels))


def test_exact_images_separate_reused_pair_ids():
    rows = manifest([
        ("example-person-a", "prior-one.png", "current.png", "edema", "improving"),
        ("example-person-a", "prior-one.png", "current.png", "consolidation", "worsening"),
        ("example-person-a", "prior-two.png", "current.png", "edema", "stable"),
        ("example-person-a", "prior-two.png", "current.png", "consolidation", "stable"),
    ])
    swaps = build_swaps(rows)
    assert len(swaps) == 4
    lookup = {row["row_id"]: row for row in rows}
    for swap in swaps:
        source = lookup[swap["query_swap_original_row_id"]]
        target = lookup[swap["query_swap_target_row_id"]]
        assert source["physical_key"][:2] == target["physical_key"][:2]
    result = query_metrics(rows, predictions(rows))
    assert result["different_label_occurrences"] == 2
    assert result["same_label_occurrences"] == 2


def test_duplicate_occurrences_remain_distinct():
    rows = manifest([
        ("example-person-a", "prior.png", "current.png", "edema", "improving"),
        ("example-person-a", "prior.png", "current.png", "edema", "improving"),
        ("example-person-a", "prior.png", "current.png", "consolidation", "worsening"),
    ])
    assert rows[0]["physical_key"] == rows[1]["physical_key"]
    assert rows[0]["row_id"] != rows[1]["row_id"]
    swaps = build_swaps(rows)
    assert len(swaps) == 4
    assert len({row["row_id"] for row in swaps}) == 4
    assert len({(row["query_swap_original_row_id"], row["query_swap_target_row_id"]) for row in swaps}) == 4


def test_compliance_is_not_paired_target_success():
    rows = manifest()
    result = query_metrics(rows, predictions(rows, ["stable", "worsening", "stable", "stable"]))
    assert result["required_change_compliance"]["estimate"] == 1
    assert result["paired_target_success"]["estimate"] == 0
    assert result["same_label_invariance"]["estimate"] == 1


def test_prediction_records_are_complete_unique_and_physical():
    rows = manifest()
    complete = records(rows)
    with pytest.raises(ValueError, match="Missing"):
        checked_predictions(rows, complete[:-1])
    with pytest.raises(ValueError, match="repeated"):
        checked_predictions(rows, complete + complete[:1])
    altered = deepcopy(complete)
    altered[0]["physical_key"][0] = "different-prior.png"
    with pytest.raises(ValueError, match="identity"):
        checked_predictions(rows, altered)
    assert checked_predictions(rows, list(reversed(complete))) == predictions(rows)


def test_invalid_and_nse_remain_errors_without_dropping_rows():
    rows = manifest()
    result = query_metrics(rows, predictions(rows, [None, NSE, "stable", "malformed"]))
    assert result["classification"]["rows"] == 4
    assert result["classification"]["accuracy"] == .25
    assert result["classification"]["invalid_output_count"] == 2
    assert result["classification"]["no_evidence_output_count"] == 1
    assert result["invalid_prediction_pair_count"] == 4
    assert result["no_evidence_prediction_pair_count"] == 2


def test_optional_generated_swaps_validate_lineage():
    rows = manifest()
    swaps = build_swaps(rows)
    result = query_metrics(rows, predictions(rows), swaps, predictions(swaps))
    assert result["mode"] == "separately_generated_swaps"
    assert result["paired_target_success"]["estimate"] == 1
    corrupted = deepcopy(swaps)
    corrupted[0]["query_swap_original_row_id"] = rows[-1]["row_id"]
    with pytest.raises(ValueError, match="lineage"):
        query_metrics(rows, predictions(rows), corrupted, predictions(swaps))


def test_temporal_preparation_transforms_labels_and_retains_occurrences(tmp_path):
    rows = manifest()
    groups = prepare_temporal(rows, tmp_path / "blank.png", seed=13)
    assert set(groups) == set(CONDITIONS)
    for name, group in groups.items():
        assert len(group) == len(rows)
        assert len({row["row_id"] for row in group}) == len(rows)
        assert [row["temporal_anchor_row_id"] for row in group] == [row["row_id"] for row in rows]
        for original, row in zip(rows, group):
            if name == "time_reversal":
                assert row["physical_key"][:2] == list(reversed(original["physical_key"][:2]))
                assert row["target_label"] == reverse(original["target_label"])
            else:
                assert row["target_label"] == original["target_label"]
    assert reverse("improving") == "worsening"
    assert reverse("worsening") == "improving"
    assert reverse("stable") == "stable"
    assert reverse(NSE) == NSE
    assert reverse(INVALID) is None


def test_temporal_metrics_use_suite_baseline_and_all_duplicate_rows(tmp_path):
    groups = prepare_temporal(manifest(), tmp_path / "blank.png", seed=13)
    pred = {name: predictions(group) for name, group in groups.items()}
    pred["prior_drop"] = predictions(groups["prior_drop"], [None] * 4)
    pred["duplicate_pair"] = predictions(groups["duplicate_pair"], ["stable", "worsening", NSE, None])
    result = temporal_metrics(groups, pred)
    assert result["prior_drop_delta_supported_macro_f1"] == -1
    assert result["random_prior_delta_supported_macro_f1"] == 0
    assert result["temporal_reversal_consistency"]["estimate"] == 1
    assert result["duplicate_stable_output_rate"]["estimate"] == .25
    assert result["duplicate_nonstable_output_rate"]["estimate"] == .75
    assert result["duplicate_change_label_output_rate"]["estimate"] == .25
    assert "duplicate_pair" not in result["by_condition"]


def test_nse_consistency_is_separate_from_supported_consistency(tmp_path):
    groups = prepare_temporal(manifest(), tmp_path / "blank.png", seed=13)
    pred = {name: predictions(group, [NSE, None, "stable", "stable"]) for name, group in groups.items()}
    result = temporal_metrics(groups, pred)
    assert result["temporal_reversal_consistency"]["numerator"] == 3
    assert result["supported_only_temporal_reversal_consistency"]["numerator"] == 2
    assert result["baseline_and_reversal_paired_success"]["numerator"] == 2


@pytest.mark.parametrize("corruption", ["label", "images", "lineage", "donor", "missing"])
def test_temporal_manifest_corruption_is_rejected(tmp_path, corruption):
    groups = prepare_temporal(manifest(), tmp_path / "blank.png", seed=13)
    pred = {name: predictions(group) for name, group in groups.items()}
    if corruption == "label":
        groups["time_reversal"][0]["target_label"] = "improving"
    elif corruption == "images":
        groups["time_reversal"][0]["physical_key"][0] = "wrong.png"
    elif corruption == "lineage":
        groups["duplicate_pair"][0]["temporal_anchor_row_index"] = 1
    elif corruption == "donor":
        groups["random_prior"][0]["prior_image_path"] = "unverified.png"
    else:
        groups["baseline"] = groups["baseline"][:-1]
    with pytest.raises(ValueError):
        temporal_metrics(groups, pred)


def test_random_prior_never_falls_back_to_same_patient(tmp_path):
    rows = manifest()[:2]
    with pytest.raises(ValueError, match="another patient"):
        prepare_temporal(rows, tmp_path / "blank.png", seed=13)


def test_manifest_row_hashes_checked(tmp_path):
    path = tmp_path / "manifest.jsonl"
    rows = manifest()
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert read_manifest(path) == rows
    rows[1]["row_id"] = rows[0]["row_id"]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError):
        read_manifest(path)


def test_query_cli_outputs_only_aggregate_results(tmp_path):
    rows = manifest()
    manifest_path, pred_path, output = [tmp_path / name for name in ("manifest.jsonl", "predictions.jsonl", "aggregate.json")]
    for path, values in ((manifest_path, rows), (pred_path, records(rows))):
        path.write_text("".join(json.dumps(row) + "\n" for row in values))
    subprocess.run([sys.executable, str(ROOT / "tools/analyze_conditional_reliability.py"), "query",
                    "--manifest", str(manifest_path), "--predictions", str(pred_path),
                    "--output", str(output)], check=True, capture_output=True, text=True)
    result = json.loads(output.read_text())
    assert result["paired_target_success"]["estimate"] == 1
    text = output.read_text()
    assert "example-person" not in text
    assert "prior-a.png" not in text
    assert "row_id" not in text


def test_preparation_and_temporal_cli_round_trip(tmp_path):
    from PIL import Image
    source = tmp_path / "manifest.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in manifest()))
    blank = tmp_path / "blank.png"
    Image.new("RGB", (2, 2), color="black").save(blank)
    script = str(ROOT / "tools/analyze_conditional_reliability.py")
    suite = tmp_path / "suite"
    subprocess.run([sys.executable, script, "prepare-temporal", "--manifest", str(source),
                    "--blank-prior-image", str(blank), "--seed", "13", "--output-dir", str(suite)],
                   check=True, capture_output=True, text=True)
    for condition in CONDITIONS:
        rows = read_manifest(suite / f"{condition}.jsonl")
        path = tmp_path / "predictions" / condition / "predictions.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in records(rows)))
    output = tmp_path / "aggregate.json"
    subprocess.run([sys.executable, script, "temporal", "--suite-dir", str(suite),
                    "--predictions", str(tmp_path / "predictions"), "--output", str(output)],
                   check=True, capture_output=True, text=True)
    assert json.loads(output.read_text())["temporal_reversal_consistency"]["estimate"] == 1
