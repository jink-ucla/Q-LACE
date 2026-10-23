from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "dataset" / "build_stage1_temporal_controls.py"


def load_module():
    spec = importlib.util.spec_from_file_location("build_stage1_temporal_controls", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load build_stage1_temporal_controls module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_row(
    *,
    source_id: str,
    pair_id: str,
    patient_id: str,
    split: str = "test",
    prior_image_path: str,
    current_image_path: str,
    prior_study_id: str,
    study_id: str,
    prior_dicom_id: str,
    dicom_id: str,
    prior_view_position: str = "PA",
    current_view_position: str = "PA",
    query_change: str = "worsening",
) -> dict[str, object]:
    return {
        "pair_id": pair_id,
        "example_id": source_id,
        "source_id": source_id,
        "query_id": source_id,
        "base_source_id": pair_id,
        "patient_id": patient_id,
        "subject_id": patient_id,
        "split": split,
        "prior_study_id": prior_study_id,
        "study_id": study_id,
        "prior_dicom_id": prior_dicom_id,
        "dicom_id": dicom_id,
        "prior_image_path": prior_image_path,
        "current_image_path": current_image_path,
        "prior_view_position": prior_view_position,
        "current_view_position": current_view_position,
        "prior_report_sections": {"findings": f"prior report {source_id}"},
        "current_report_sections": {"findings": f"current report {source_id}"},
        "problem": "Stage-1 prompt",
        "query": "compared with the prior study, what is the interval change for edema?",
        "solution": '{"changes":[{"anatomy":"unspecified","finding":"edema","change":"worsening"}]}',
        "query_anatomy": "unspecified",
        "query_finding": "edema",
        "query_change": query_change,
        "query_index": 0,
        "task_mode": "stage1",
        "curriculum_stage": 1,
        "metadata": {"benchmark_name": "ms_cxr_t", "original_source_id": source_id},
    }


def test_build_dataset_applies_expected_controls(tmp_path: Path) -> None:
    module = load_module()
    rows = [
        make_row(
            source_id="row-a",
            pair_id="pair-a",
            patient_id="p1",
            prior_image_path="/tmp/p1-prior.jpg",
            current_image_path="/tmp/p1-current.jpg",
            prior_study_id="s1",
            study_id="s2",
            prior_dicom_id="d1",
            dicom_id="d2",
            query_change="worsening",
        ),
        make_row(
            source_id="row-b",
            pair_id="pair-b",
            patient_id="p2",
            prior_image_path="/tmp/p2-prior.jpg",
            current_image_path="/tmp/p2-current.jpg",
            prior_study_id="s3",
            study_id="s4",
            prior_dicom_id="d3",
            dicom_id="d4",
            query_change="improving",
        ),
    ]
    input_jsonl = tmp_path / "input.jsonl"
    output_dir = tmp_path / "controls"
    module.write_jsonl(input_jsonl, rows)

    rows_by_control, summary = module.build_dataset(
        SimpleNamespace(
            input_jsonl=input_jsonl,
            output_dir=output_dir,
            seed=17,
        )
    )

    assert summary["input_row_count"] == 2
    assert summary["count"] == 10
    assert summary["control_counts"] == {name: 2 for name in module.CONTROL_NAMES}
    assert summary["random_prior_selection_counts"] == {"same_split_different_patient": 2}

    baseline_row = rows_by_control["baseline"][0]
    assert baseline_row["pair_id"] == "pair-a"
    assert baseline_row["source_id"] == "row-a::control::baseline"
    assert baseline_row["query_change"] == "worsening"
    assert baseline_row["metadata"]["temporal_control_name"] == "baseline"
    assert baseline_row["metadata"]["temporal_control_source_id"] == "row-a"
    assert baseline_row["metadata"]["temporal_control_modified_fields"] == []

    prior_drop_row = rows_by_control["prior_drop"][0]
    assert prior_drop_row["prior_image_path"] == str(output_dir / "control_assets" / "prior_drop_blank.png")
    assert Path(prior_drop_row["prior_image_path"]).exists()
    assert prior_drop_row["current_image_path"] == "/tmp/p1-current.jpg"
    assert prior_drop_row["prior_view_position"] == "CONTROL_BLANK"
    assert prior_drop_row["prior_report_sections"] == {}
    assert prior_drop_row["metadata"]["temporal_control_blank_prior_image_path"] == str(
        output_dir / "control_assets" / "prior_drop_blank.png"
    )

    duplicate_row = rows_by_control["duplicate_pair"][0]
    assert duplicate_row["prior_image_path"] == "/tmp/p1-current.jpg"
    assert duplicate_row["prior_study_id"] == "s2"
    assert duplicate_row["prior_dicom_id"] == "d2"
    assert duplicate_row["prior_report_sections"] == {"findings": "current report row-a"}
    assert duplicate_row["metadata"]["temporal_control_duplicate_strategy"] == "current_as_prior"

    reversed_row = rows_by_control["time_reversal"][0]
    assert reversed_row["prior_image_path"] == "/tmp/p1-current.jpg"
    assert reversed_row["current_image_path"] == "/tmp/p1-prior.jpg"
    assert reversed_row["prior_study_id"] == "s2"
    assert reversed_row["study_id"] == "s1"
    assert reversed_row["prior_dicom_id"] == "d2"
    assert reversed_row["dicom_id"] == "d1"
    assert reversed_row["metadata"]["temporal_control_expected_reversed_query_change"] == "improving"

    random_prior_row = rows_by_control["random_prior"][0]
    assert random_prior_row["prior_image_path"] == "/tmp/p2-prior.jpg"
    assert random_prior_row["prior_study_id"] == "s3"
    assert random_prior_row["prior_dicom_id"] == "d3"
    assert random_prior_row["prior_report_sections"] == {"findings": "prior report row-b"}
    assert random_prior_row["metadata"]["temporal_control_random_prior_source_id"] == "row-b"
    assert random_prior_row["metadata"]["temporal_control_random_prior_patient_id"] == "p2"
    assert random_prior_row["metadata"]["temporal_control_random_prior_selection"] == "same_split_different_patient"
    assert random_prior_row["metadata"]["temporal_control_random_prior_self_fallback"] is False


def test_main_writes_control_jsonls_and_summary(tmp_path: Path) -> None:
    module = load_module()
    rows = [
        make_row(
            source_id="row-a",
            pair_id="pair-a",
            patient_id="p1",
            prior_image_path="/tmp/p1-prior.jpg",
            current_image_path="/tmp/p1-current.jpg",
            prior_study_id="s1",
            study_id="s2",
            prior_dicom_id="d1",
            dicom_id="d2",
            query_change="stable",
        )
    ]
    input_jsonl = tmp_path / "input.jsonl"
    output_dir = tmp_path / "controls"
    module.write_jsonl(input_jsonl, rows)

    exit_code = module.main(
        [
            "--input-jsonl",
            str(input_jsonl),
            "--output-dir",
            str(output_dir),
            "--seed",
            "5",
        ]
    )
    assert exit_code == 0

    for filename in (
        "baseline.jsonl",
        "prior_drop.jsonl",
        "duplicate_pair.jsonl",
        "time_reversal.jsonl",
        "random_prior.jsonl",
        "all_controls.jsonl",
        "summary.json",
    ):
        assert (output_dir / filename).exists(), filename

    all_controls_rows = module.load_jsonl(output_dir / "all_controls.jsonl")
    assert len(all_controls_rows) == 5
    assert [row["control_name"] for row in all_controls_rows] == list(module.CONTROL_NAMES)
    assert len({row["source_id"] for row in all_controls_rows}) == 5

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["control_counts"] == {name: 1 for name in module.CONTROL_NAMES}
    assert summary["random_prior_self_fallback_count"] == 1
    assert summary["blank_prior_image_path"] == str(output_dir / "control_assets" / "prior_drop_blank.png")


def test_main_with_relative_output_dir_emits_absolute_blank_prior_paths(tmp_path: Path, monkeypatch) -> None:
    module = load_module()
    rows = [
        make_row(
            source_id="row-a",
            pair_id="pair-a",
            patient_id="p1",
            prior_image_path="/tmp/p1-prior.jpg",
            current_image_path="/tmp/p1-current.jpg",
            prior_study_id="s1",
            study_id="s2",
            prior_dicom_id="d1",
            dicom_id="d2",
            query_change="stable",
        )
    ]
    input_jsonl = tmp_path / "input.jsonl"
    module.write_jsonl(input_jsonl, rows)

    monkeypatch.chdir(tmp_path)
    relative_output_dir = Path("controls_rel")

    exit_code = module.main(
        [
            "--input-jsonl",
            str(input_jsonl),
            "--output-dir",
            str(relative_output_dir),
            "--seed",
            "5",
        ]
    )
    assert exit_code == 0

    output_dir = (tmp_path / relative_output_dir).resolve()
    all_controls_rows = module.load_jsonl(output_dir / "all_controls.jsonl")
    prior_drop_rows = [row for row in all_controls_rows if row["control_name"] == "prior_drop"]
    assert len(prior_drop_rows) == 1
    assert Path(prior_drop_rows[0]["prior_image_path"]).is_absolute()
    assert Path(prior_drop_rows[0]["prior_image_path"]).exists()

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert Path(summary["blank_prior_image_path"]).is_absolute()
    assert Path(summary["blank_prior_image_path"]).exists()
