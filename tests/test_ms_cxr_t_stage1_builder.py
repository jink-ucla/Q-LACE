from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "dataset" / "build_ms_cxr_t_longitudinal_stage1.py"


def load_module():
    spec = importlib.util.spec_from_file_location("build_ms_cxr_t_longitudinal_stage1", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load MS-CXR-T builder module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def touch_image(root: Path, subject_id: str, study_id: str, dicom_id: str) -> Path:
    image_path = root / f"p{subject_id[:2]}" / f"p{subject_id}" / f"s{study_id}" / f"{dicom_id}.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"fake-jpg")
    return image_path


def build_args(
    *,
    csv_path: Path,
    metadata_csv: Path,
    jpg_root: Path,
    output_dir: Path,
    split_policy: str = "all_test",
    same_view_only: bool = False,
) -> argparse.Namespace:
    return argparse.Namespace(
        ms_cxr_t_csv=csv_path,
        mimic_metadata_csv=metadata_csv,
        mimic_jpg_root=jpg_root,
        output_dir=output_dir,
        split_policy=split_policy,
        split_seed=7,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        allowed_label_qualities=["one_expert", "multiple_experts", "disagreement"],
        same_view_only=same_view_only,
    )


def test_build_dataset_emits_stage1_rows_and_pair_manifest(tmp_path: Path) -> None:
    module = load_module()
    jpg_root = tmp_path / "mimic_files"
    csv_path = tmp_path / "ms_cxr_t.csv"
    metadata_csv = tmp_path / "metadata.csv"

    write_csv(
        csv_path,
        [
            "dicom_id",
            "previous_dicom_id",
            "previous_study_id",
            "study_id",
            "subject_id",
            "consolidation_progression",
            "consolidation_label_quality",
            "edema_progression",
            "edema_label_quality",
            "pleural_effusion_progression",
            "pleural_effusion_label_quality",
            "pneumonia_progression",
            "pneumonia_label_quality",
            "pneumothorax_progression",
            "pneumothorax_label_quality",
        ],
        [
            {
                "dicom_id": "p10/p10000001/s20000002/curr-1",
                "previous_dicom_id": "p10/p10000001/s20000001/prev-1",
                "previous_study_id": "20000001",
                "study_id": "20000002",
                "subject_id": "10000001",
                "consolidation_progression": "worsening",
                "consolidation_label_quality": "one_expert",
                "edema_progression": "",
                "edema_label_quality": "",
                "pleural_effusion_progression": "stable",
                "pleural_effusion_label_quality": "multiple_experts",
                "pneumonia_progression": "",
                "pneumonia_label_quality": "",
                "pneumothorax_progression": "",
                "pneumothorax_label_quality": "",
            },
            {
                "dicom_id": "p10/p10000002/s30000002/curr-2",
                "previous_dicom_id": "p10/p10000002/s30000001/prev-2",
                "previous_study_id": "30000001",
                "study_id": "30000002",
                "subject_id": "10000002",
                "consolidation_progression": "",
                "consolidation_label_quality": "",
                "edema_progression": "improving",
                "edema_label_quality": "disagreement",
                "pleural_effusion_progression": "",
                "pleural_effusion_label_quality": "",
                "pneumonia_progression": "",
                "pneumonia_label_quality": "",
                "pneumothorax_progression": "",
                "pneumothorax_label_quality": "",
            },
        ],
    )
    write_csv(
        metadata_csv,
        ["dicom_id", "ViewPosition"],
        [
            {"dicom_id": "prev-1", "ViewPosition": "AP"},
            {"dicom_id": "curr-1", "ViewPosition": "AP"},
            {"dicom_id": "prev-2", "ViewPosition": "PA"},
            {"dicom_id": "curr-2", "ViewPosition": "PA"},
        ],
    )

    touch_image(jpg_root, "10000001", "20000001", "prev-1")
    touch_image(jpg_root, "10000001", "20000002", "curr-1")
    touch_image(jpg_root, "10000002", "30000001", "prev-2")
    touch_image(jpg_root, "10000002", "30000002", "curr-2")

    query_rows, pair_rows, summary = module.build_dataset(
        build_args(
            csv_path=csv_path,
            metadata_csv=metadata_csv,
            jpg_root=jpg_root,
            output_dir=tmp_path / "out",
        )
    )

    assert len(query_rows) == 3
    assert len(pair_rows) == 2
    assert summary["split_policy"] == "all_test"
    assert summary["split_counts"] == {"test": 3}
    assert summary["finding_counts"] == {
        "consolidation": 1,
        "edema": 1,
        "pleural effusion": 1,
    }
    assert summary["change_counts"] == {"improving": 1, "stable": 1, "worsening": 1}

    first = query_rows[0]
    assert first["task_mode"] == "stage1"
    assert first["split"] == "test"
    assert first["query_anatomy"] == "unspecified"
    assert first["query_finding"] in {"consolidation", "edema", "pleural effusion"}
    assert Path(first["prior_image_path"]).is_absolute()
    assert Path(first["current_image_path"]).is_absolute()
    solution = json.loads(first["solution"])
    assert set(solution.keys()) == {"changes"}
    assert len(solution["changes"]) == 1
    assert solution["changes"][0]["anatomy"] == "unspecified"

    pair_row = pair_rows[0]
    assert pair_row["prior_dicom_id"].startswith("prev-")
    assert pair_row["dicom_id"].startswith("curr-")
    assert pair_row["subject_id"] == pair_row["patient_id"]


def test_same_view_filter_and_patient_hash_keep_subjects_together(tmp_path: Path) -> None:
    module = load_module()
    jpg_root = tmp_path / "mimic_files"
    csv_path = tmp_path / "ms_cxr_t.csv"
    metadata_csv = tmp_path / "metadata.csv"

    write_csv(
        csv_path,
        [
            "dicom_id",
            "previous_dicom_id",
            "previous_study_id",
            "study_id",
            "subject_id",
            "consolidation_progression",
            "consolidation_label_quality",
            "edema_progression",
            "edema_label_quality",
            "pleural_effusion_progression",
            "pleural_effusion_label_quality",
            "pneumonia_progression",
            "pneumonia_label_quality",
            "pneumothorax_progression",
            "pneumothorax_label_quality",
        ],
        [
            {
                "dicom_id": "p10/p10000011/s40000002/curr-a",
                "previous_dicom_id": "p10/p10000011/s40000001/prev-a",
                "previous_study_id": "40000001",
                "study_id": "40000002",
                "subject_id": "10000011",
                "consolidation_progression": "stable",
                "consolidation_label_quality": "one_expert",
                "edema_progression": "",
                "edema_label_quality": "",
                "pleural_effusion_progression": "",
                "pleural_effusion_label_quality": "",
                "pneumonia_progression": "",
                "pneumonia_label_quality": "",
                "pneumothorax_progression": "",
                "pneumothorax_label_quality": "",
            },
            {
                "dicom_id": "p10/p10000011/s40000004/curr-b",
                "previous_dicom_id": "p10/p10000011/s40000003/prev-b",
                "previous_study_id": "40000003",
                "study_id": "40000004",
                "subject_id": "10000011",
                "consolidation_progression": "",
                "consolidation_label_quality": "",
                "edema_progression": "worsening",
                "edema_label_quality": "one_expert",
                "pleural_effusion_progression": "",
                "pleural_effusion_label_quality": "",
                "pneumonia_progression": "",
                "pneumonia_label_quality": "",
                "pneumothorax_progression": "",
                "pneumothorax_label_quality": "",
            },
            {
                "dicom_id": "p10/p10000012/s50000002/curr-c",
                "previous_dicom_id": "p10/p10000012/s50000001/prev-c",
                "previous_study_id": "50000001",
                "study_id": "50000002",
                "subject_id": "10000012",
                "consolidation_progression": "improving",
                "consolidation_label_quality": "one_expert",
                "edema_progression": "",
                "edema_label_quality": "",
                "pleural_effusion_progression": "",
                "pleural_effusion_label_quality": "",
                "pneumonia_progression": "",
                "pneumonia_label_quality": "",
                "pneumothorax_progression": "",
                "pneumothorax_label_quality": "",
            },
        ],
    )
    write_csv(
        metadata_csv,
        ["dicom_id", "ViewPosition"],
        [
            {"dicom_id": "prev-a", "ViewPosition": "AP"},
            {"dicom_id": "curr-a", "ViewPosition": "PA"},
            {"dicom_id": "prev-b", "ViewPosition": "AP"},
            {"dicom_id": "curr-b", "ViewPosition": "AP"},
            {"dicom_id": "prev-c", "ViewPosition": "PA"},
            {"dicom_id": "curr-c", "ViewPosition": "PA"},
        ],
    )

    for subject_id, previous_study_id, previous_dicom_id, study_id, dicom_id in (
        ("10000011", "40000001", "prev-a", "40000002", "curr-a"),
        ("10000011", "40000003", "prev-b", "40000004", "curr-b"),
        ("10000012", "50000001", "prev-c", "50000002", "curr-c"),
    ):
        touch_image(jpg_root, subject_id, previous_study_id, previous_dicom_id)
        touch_image(jpg_root, subject_id, study_id, dicom_id)

    query_rows, _, summary = module.build_dataset(
        build_args(
            csv_path=csv_path,
            metadata_csv=metadata_csv,
            jpg_root=jpg_root,
            output_dir=tmp_path / "out",
            split_policy="patient_hash",
            same_view_only=True,
        )
    )

    assert summary["skipped_counts"]["mixed_view_pair"] == 1
    assert len(query_rows) == 2
    subject_to_splits: dict[str, set[str]] = {}
    for row in query_rows:
        subject_to_splits.setdefault(row["patient_id"], set()).add(row["split"])
    assert len(subject_to_splits["10000011"]) == 1
    assert len(subject_to_splits["10000012"]) == 1
