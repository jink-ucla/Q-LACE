from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "audit_longitudinal_benchmark_overlap.py"


def _load_audit_module():
    spec = importlib.util.spec_from_file_location("audit_longitudinal_benchmark_overlap", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")


def test_audit_overlap_counts_identifier_categories_and_report_text(tmp_path):
    module = _load_audit_module()
    local_dir = tmp_path / "local"
    benchmark_dir = tmp_path / "benchmark"
    local_dir.mkdir()
    benchmark_dir.mkdir()

    _write_jsonl(
        local_dir / "manifest.jsonl",
        [
            {
                "pair_id": "1:10:11",
                "patient_id": "1",
                "source_id": "1:10:11",
                "prior_study_id": "10",
                "current_study_id": "11",
                "prior_dicom_id": "d10",
                "current_dicom_id": "d11",
                "prior_image_path": "/tmp/local/prior-a.png",
                "current_image_path": "/tmp/local/current-a.png",
                "problem": "Compare the studies.",
                "metadata": {
                    "builder": "x.py",
                    "task_family": "longitudinal_cxr_change_rft",
                    "label_status": "unlabeled_manifest",
                },
                "prior_report_sections": {"findings": "No acute disease."},
                "current_report_sections": {"impression": "No acute disease."},
            },
            {
                "pair_id": "2:20:21",
                "patient_id": "2",
                "source_id": "2:20:21",
                "prior_study_id": "20",
                "current_study_id": "21",
                "prior_dicom_id": "d20",
                "current_dicom_id": "d21",
                "prior_image_path": "/tmp/local/prior-b.png",
                "current_image_path": "/tmp/local/current-b.png",
                "problem": "Compare the studies.",
                "metadata": {
                    "builder": "x.py",
                    "task_family": "longitudinal_cxr_change_rft",
                    "label_status": "unlabeled_manifest",
                },
                "prior_report_sections": {"findings": "Stable exam."},
                "current_report_sections": {"impression": "Stable exam."},
            },
        ],
    )
    benchmark_records = [
        {
            "pair_id": "1:10:11",
            "patient_id": "1",
            "study_id": "11",
            "dicom_id": "d11",
            "image_path": "/export/other/current-a.png",
            "report_text": "No acute disease.",
        },
        {
            "pair_id": "9:90:91",
            "patient_id": "9",
            "study_id": "91",
            "dicom_id": "d91",
            "image_path": "/export/other/unrelated.png",
            "report_text": "No focal consolidation.",
        },
    ]
    with (benchmark_dir / "export.json").open("w", encoding="utf-8") as handle:
        json.dump(benchmark_records, handle)

    summary = module.audit_overlap(
        [local_dir],
        [benchmark_dir],
        include_report_text=True,
        sample_limit=3,
    )

    assert summary["records"] == {"local": 2, "benchmark": 2}
    assert summary["overlap"]["categories"]["patient_id"]["shared_unique_values"] == 1
    assert summary["overlap"]["categories"]["study_id"]["shared_unique_values"] == 1
    assert summary["overlap"]["categories"]["dicom_id"]["shared_unique_values"] == 1
    assert summary["overlap"]["categories"]["pair_id"]["shared_unique_values"] == 1
    assert summary["overlap"]["categories"]["image_path"]["shared_unique_values"] == 0
    assert summary["overlap"]["categories"]["image_basename"]["shared_unique_values"] == 1
    assert summary["overlap"]["categories"]["patient_id"]["local_rows_with_shared_values"] == 1
    assert summary["overlap"]["categories"]["patient_id"]["benchmark_rows_with_shared_values"] == 1
    assert summary["overlap"]["report_text"]["shared_unique_texts"] == 1
    assert summary["overlap"]["report_text"]["shared_samples"][0]["preview"] == "no acute disease."
    assert summary["overlap"]["any_identifier"]["shared_unique_values"] >= 5

    report = module.render_report(summary)
    assert "Longitudinal Benchmark Overlap Audit" in report
    assert "patient_id: shared=1" in report
    assert "Report-text overlap" in report


def test_audit_overlap_cli_writes_json_and_report_to_paths(tmp_path, capsys):
    module = _load_audit_module()
    local_file = tmp_path / "local.jsonl"
    benchmark_file = tmp_path / "benchmark.json"
    _write_jsonl(
        local_file,
        [
            {
                "pair_id": "1:10:11",
                "patient_id": "1",
                "source_id": "1:10:11",
                "prior_study_id": "10",
                "current_study_id": "11",
                "prior_dicom_id": "d10",
                "current_dicom_id": "d11",
                "prior_image_path": "/tmp/local/prior-a.png",
                "current_image_path": "/tmp/local/current-a.png",
                "problem": "Compare the studies.",
                "metadata": {
                    "builder": "x.py",
                    "task_family": "longitudinal_cxr_change_rft",
                    "label_status": "unlabeled_manifest",
                },
                "prior_report_sections": {"findings": "No acute disease."},
                "current_report_sections": {"impression": "No acute disease."},
            }
        ],
    )
    with benchmark_file.open("w", encoding="utf-8") as handle:
        json.dump(
            [
                {
                    "pair_id": "1:10:11",
                    "patient_id": "1",
                    "study_id": "11",
                    "dicom_id": "d11",
                    "image_path": "/export/other/current-a.png",
                    "report_text": "No acute disease.",
                }
            ],
            handle,
        )

    json_output = tmp_path / "out" / "summary.json"
    report_output = tmp_path / "out" / "report.txt"
    exit_code = module.main(
        [
            "--local",
            str(local_file),
            "--benchmark",
            str(benchmark_file),
            "--include-report-text",
            "--json-output",
            str(json_output),
            "--report-output",
            str(report_output),
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.out.strip() == ""
    assert captured.err.strip() == ""
    summary = json.loads(json_output.read_text(encoding="utf-8"))
    assert summary["records"] == {"local": 1, "benchmark": 1}
    assert summary["overlap"]["report_text"]["shared_unique_texts"] == 1
    assert "Longitudinal Benchmark Overlap Audit" in report_output.read_text(encoding="utf-8")


def test_audit_overlap_rejects_unsupported_json_structure(tmp_path):
    module = _load_audit_module()
    bad_file = tmp_path / "bad.json"
    bad_file.write_text(json.dumps({"unexpected": {"nested": "structure"}}), encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported JSON structure"):
        module.audit_overlap([bad_file], [bad_file])
