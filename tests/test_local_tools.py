"""Synthetic fixtures and aggregate-only checks; no retained research cases."""
import csv
from copy import deepcopy
import json
import hashlib
from pathlib import Path
import subprocess
import sys

import pytest

from tools import build_master_csv as master
from tools.make_finding_blind import finding_blind
from tools.run_paper_training import command
from scripts import cmig_medgemma_baseline as harness
from tools.verify_manifest import verify
from tools import verify_aggregate_tables as aggregate


def csv_file(path, fields, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def inputs(tmp_path):
    # Clearly fabricated small identifiers: not real dataset records.
    key = {"subject_id": "1100", "study_id": "2200", "dicom_id": "synthetic-image"}
    meta, splits = tmp_path / "metadata.csv", tmp_path / "split.csv"
    csv_file(meta, (*master.KEYS, "ViewPosition"), [{**key, "ViewPosition": " ap "}])
    csv_file(splits, (*master.KEYS, "split"), [{**key, "split": "valid"}])
    image = tmp_path / "images/p11/p1100/s2200/synthetic-image.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"synthetic-placeholder-not-an-image")
    return meta, splits, image


def test_master_reconstructs_join_normalized_split_view_and_path(tmp_path):
    meta, splits, image = inputs(tmp_path)
    output = tmp_path / "master.csv"
    result = master.build_master(meta, splits, tmp_path / "images", output)
    assert result == {"metadata_rows": 1, "split_rows": 1, "output_rows": 1}
    with output.open() as handle:
        row = list(csv.DictReader(handle))[0]
    assert row["Path"] == str(image.resolve())
    assert row["split"] == "validate" and row["ViewPosition"] == "AP"


def test_master_rejects_duplicate_join_key_without_writing_output(tmp_path):
    meta, splits, _ = inputs(tmp_path)
    with splits.open("a") as handle:
        handle.write("1100,2200,synthetic-image,train\n")
    with pytest.raises(ValueError, match="Duplicate split"):
        master.build_master(meta, splits, tmp_path / "images", tmp_path / "output.csv")
    assert not (tmp_path / "output.csv").exists()


def test_master_rejects_unmatched_split_or_missing_image(tmp_path):
    meta, splits, image = inputs(tmp_path)
    image.unlink()
    with pytest.raises(ValueError, match="does not exist"):
        master.build_master(meta, splits, tmp_path / "images", tmp_path / "output.csv")
    assert master.build_master(meta, splits, tmp_path / "images", tmp_path / "output.csv", check_images=False)["output_rows"] == 1
    with splits.open("a") as handle:
        handle.write("1100,2200,synthetic-extra,test\n")
    with pytest.raises(ValueError, match="populations differ"):
        master.build_master(meta, splits, tmp_path / "images", tmp_path / "output.csv", check_images=False)


def test_master_rejects_source_overwrite_unsafe_and_unknown_split(tmp_path):
    meta, splits, _ = inputs(tmp_path)
    with pytest.raises(ValueError, match="overwrite"):
        master.build_master(meta, splits, tmp_path / "images", meta)
    with pytest.raises(ValueError, match="unsafe"):
        master.identifier("../outside")
    with pytest.raises(ValueError, match="unsafe"):
        master.identifier("", "p")
    csv_file(splits, (*master.KEYS, "split"), [{"subject_id": "1100", "study_id": "2200", "dicom_id": "synthetic-image", "split": "unknown"}])
    with pytest.raises(ValueError, match="Unknown split"):
        master.build_master(meta, splits, tmp_path / "images", tmp_path / "output.csv")


def test_master_rejects_empty_source_population(tmp_path):
    meta, splits, _ = inputs(tmp_path)
    csv_file(splits, (*master.KEYS, "split"), [])
    with pytest.raises(ValueError, match="Empty official"):
        master.build_master(meta, splits, tmp_path / "images", tmp_path / "output.csv")


def test_paper_command_has_exact_caps_dense_modules_and_no_temp_override():
    cmd = command(Path("chosen-model"), Path("chosen-data"), Path("chosen-output"), 123)
    assert cmd[cmd.index("--max-train-samples") + 1] == "2048"
    assert cmd[cmd.index("--max-eval-samples") + 1] == "512"
    assert cmd[cmd.index("--num-train-epochs") + 1] == "1"
    assert cmd[cmd.index("--gradient-accumulation-steps") + 1] == "1"
    assert cmd[cmd.index("--target-modules") + 1] == "all-linear"
    assert cmd[cmd.index("--modules-to-save") + 1:cmd.index("--modules-to-save") + 3] == ["lm_head", "embed_tokens"]
    assert "--no-load-in-4bit" in cmd and "--no-eval" not in cmd
    assert cmd[cmd.index("--model") + 1] == "chosen-model"
    fast = command(Path("chosen-model"), Path("chosen-data"), Path("chosen-output"), 123, no_eval=True)
    assert "--no-eval" in fast
    assert fast[fast.index("--save-steps") + 1] == "2048"


def test_finding_blind_preserves_targets_order_and_images_without_mutation():
    row = {"source_id": "fabricated-row", "query_finding": "edema", "query_change": "stable",
           "prior_image_path": "/fake/prior.jpg", "current_image_path": "/fake/current.jpg",
           "metadata": {"query_finding": "edema"}, "problem": "edema"}
    original = deepcopy(row)
    output = finding_blind(row)
    assert row == original
    assert output["query_finding"] == "the unspecified finding"
    for key in ("source_id", "query_change", "prior_image_path", "current_image_path"):
        assert output[key] == row[key]
    assert "edema" not in output["problem"]


def test_explicit_local_comparator_cli_uses_all_occurrences_and_paired_outputs(tmp_path):
    root = Path(__file__).resolve().parents[1]
    source = tmp_path / "synthetic_source.jsonl"
    labels = ("improving", "stable", "worsening")
    records = [{"source_id": "synthetic-collision", "patient_id": f"synthetic-patient-{i}",
                "prior_image_path": f"/fake/prior-{i}.jpg", "current_image_path": "/fake/current.jpg",
                "query_finding": "edema", "query_change": label} for i, label in enumerate(labels)]
    source.write_text("".join(json.dumps(r) + "\n" for r in records))
    rows, _ = harness.build_manifest(source, expected_rows=3, check_images=False)
    manifest = tmp_path / "manifest.jsonl"
    harness.write_jsonl(manifest, rows)
    base, adapted = tmp_path / "base.jsonl", tmp_path / "adapted.jsonl"
    harness.write_jsonl(base, [{**r, "pred_label": "stable"} for r in reversed(rows)])
    harness.write_jsonl(adapted, [{**r, "pred_label": r["target_label"]} for r in rows])
    output = tmp_path / "aggregate.json"
    subprocess.run([sys.executable, str(root / "tools/compare_local_predictions.py"),
                    "--manifest", str(manifest), "--arm", f"base={base}",
                    "--arm", f"adapted={adapted}", "--output", str(output),
                    "--bootstrap-replicates", "50"], check=True, cwd=root)
    result = json.loads(output.read_text())
    assert result["rows"] == 3 and result["patient_count"] == 3
    assert result["contrasts"]["adapted_minus_base"]["accuracy"]["estimate"] == pytest.approx(2 / 3)
    assert "synthetic-patient" not in output.read_text()


def test_manifest_verifier_detects_modified_payload_and_unsafe_paths(tmp_path):
    data = b"synthetic payload"
    (tmp_path / "example.txt").write_bytes(data)
    entry = {"file": "example.txt", "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    manifest = tmp_path / "FILE_MANIFEST.json"
    manifest.write_text(json.dumps({"files": [entry]}))
    assert verify(tmp_path) == 1
    (tmp_path / "example.txt").write_bytes(b"modified")
    with pytest.raises(ValueError, match="integrity"):
        verify(tmp_path)
    entry["file"] = "../outside.txt"
    manifest.write_text(json.dumps({"files": [entry]}))
    with pytest.raises(ValueError, match="manifest path"):
        verify(tmp_path)


def test_aggregate_counts_keep_invalid_and_nse_outputs_as_errors():
    # Every target has two rows; one correct and one unsupported/invalid.
    matrix = [[1, 0, 0, 1, 0], [0, 1, 0, 0, 1], [0, 0, 1, 1, 0]]
    result = aggregate.metrics(matrix)
    assert result["accuracy"] == pytest.approx(0.5)
    assert result["class_balanced_accuracy"] == pytest.approx(0.5)
    assert result["supported_macro_f1"] == pytest.approx(2 / 3)
    with pytest.raises(ValueError, match="three target classes"):
        aggregate.metrics([[1, 0, 0, 0, 0], [0] * 5, [0, 0, 1, 0, 0]])


def test_shipped_aggregate_arithmetic_and_csv_reproduction():
    root = Path(__file__).resolve().parents[1]
    payload = json.loads((root / "aggregates/baseline_comparison.json").read_text())
    result = aggregate.verify(payload)
    assert result["arms"] == 13 and result["finding_matrices"] == 65
    assert result["run_aggregates"] == 3
    assert aggregate.table_csv(payload) == (root / "aggregates/baseline_metrics.csv").read_bytes()
    changed = deepcopy(payload)
    changed["aggregates"]["qlace_historical_three_runs"]["metrics"]["accuracy"]["estimate"] += 0.001
    with pytest.raises(ValueError, match="arithmetic mismatch"):
        aggregate.verify(changed)
