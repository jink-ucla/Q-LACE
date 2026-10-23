from __future__ import annotations

import csv
import importlib.util
from types import SimpleNamespace
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCRIPT_PATH = ROOT / "dataset" / "build_chest_imagenome_gold_stage1.py"


def load_module():
    spec = importlib.util.spec_from_file_location("build_chest_imagenome_gold_stage1", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load build_chest_imagenome_gold_stage1 module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_metadata_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "dicom_id",
        "subject_id",
        "study_id",
        "ViewPosition",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_gold_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "patient_id",
        "relationship_id",
        "subject_id",
        "object_id",
        "bbox",
        "comparison",
        "attribute",
        "sentence",
        "bbox_coord_224_subject",
        "bbox_coord_224_object",
        "bbox_coord_original_subject",
        "bbox_coord_original_object",
        "label_name",
        "current_image_id",
        "previous_image_id",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def touch_image(root: Path, subject_id: str, study_id: str, dicom_id: str) -> None:
    subject_prefix = f"p{subject_id[:2]}"
    image_path = root / subject_prefix / f"p{subject_id}" / f"s{study_id}" / f"{dicom_id}.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(b"")


def test_build_dataset_dedupes_and_filters_ambiguous_rows(tmp_path: Path) -> None:
    module = load_module()
    metadata_csv = tmp_path / "metadata.csv"
    jpg_root = tmp_path / "files"
    gold_tsv = tmp_path / "gold.tsv"

    write_metadata_csv(
        metadata_csv,
        [
            {"dicom_id": "prev-a", "subject_id": "12345678", "study_id": "111", "ViewPosition": "PA"},
            {"dicom_id": "curr-a", "subject_id": "12345678", "study_id": "222", "ViewPosition": "PA"},
            {"dicom_id": "prev-b", "subject_id": "87654321", "study_id": "333", "ViewPosition": "AP"},
            {"dicom_id": "curr-b", "subject_id": "87654321", "study_id": "444", "ViewPosition": "PA"},
        ],
    )
    for subject_id, study_id, dicom_id in [
        ("12345678", "111", "prev-a"),
        ("12345678", "222", "curr-a"),
        ("87654321", "333", "prev-b"),
        ("87654321", "444", "curr-b"),
    ]:
        touch_image(jpg_root, subject_id, study_id, dicom_id)

    write_gold_tsv(
        gold_tsv,
        [
            {
                "patient_id": "12345678",
                "relationship_id": "rel-1",
                "subject_id": "curr-a_object",
                "object_id": "prev-a_object",
                "bbox": "lung",
                "comparison": "worsened",
                "attribute": "attr-1",
                "sentence": "Sentence A",
                "bbox_coord_224_subject": "[1,2,3,4]",
                "bbox_coord_224_object": "[1,2,3,4]",
                "bbox_coord_original_subject": "[10,20,30,40]",
                "bbox_coord_original_object": "[10,20,30,40]",
                "label_name": "pneumonia",
                "current_image_id": "curr-a",
                "previous_image_id": "prev-a",
            },
            {
                "patient_id": "12345678",
                "relationship_id": "rel-2",
                "subject_id": "curr-a_object",
                "object_id": "prev-a_object",
                "bbox": "lung",
                "comparison": "worsened",
                "attribute": "attr-2",
                "sentence": "Sentence B",
                "bbox_coord_224_subject": "[1,2,3,4]",
                "bbox_coord_224_object": "[1,2,3,4]",
                "bbox_coord_original_subject": "[10,20,30,40]",
                "bbox_coord_original_object": "[10,20,30,40]",
                "label_name": "pneumonia",
                "current_image_id": "curr-a",
                "previous_image_id": "prev-a",
            },
            {
                "patient_id": "87654321",
                "relationship_id": "rel-3",
                "subject_id": "curr-b_object",
                "object_id": "prev-b_object",
                "bbox": "lung",
                "comparison": "no change;;worsened",
                "attribute": "attr-3",
                "sentence": "Sentence C",
                "bbox_coord_224_subject": "[1,2,3,4]",
                "bbox_coord_224_object": "[1,2,3,4]",
                "bbox_coord_original_subject": "[10,20,30,40]",
                "bbox_coord_original_object": "[10,20,30,40]",
                "label_name": "edema",
                "current_image_id": "curr-b",
                "previous_image_id": "prev-b",
            },
        ],
    )

    args = SimpleNamespace(
        gold_tsv=gold_tsv,
        mimic_metadata_csv=metadata_csv,
        mimic_jpg_root=jpg_root,
        output_dir=tmp_path / "out",
        same_view_only=False,
        allowed_label_names=None,
        normalize_to_ms_cxr_t_heads=False,
    )

    query_rows, pair_rows, summary = module.build_dataset(args)

    assert len(query_rows) == 1
    assert len(pair_rows) == 1
    row = query_rows[0]
    assert row["query_finding"] == "pneumonia"
    assert row["query_change"] == "worsening"
    assert row["metadata"]["duplicate_group_size"] == 2
    assert summary["comparison_counts"] == {"worsening": 1}
    assert summary["skipped_counts"]["ambiguous_multi_label_comparison"] == 1


def test_same_view_only_filters_mixed_view_pairs(tmp_path: Path) -> None:
    module = load_module()
    metadata_csv = tmp_path / "metadata.csv"
    jpg_root = tmp_path / "files"
    gold_tsv = tmp_path / "gold.tsv"

    write_metadata_csv(
        metadata_csv,
        [
            {"dicom_id": "prev-a", "subject_id": "12345678", "study_id": "111", "ViewPosition": "PA"},
            {"dicom_id": "curr-a", "subject_id": "12345678", "study_id": "222", "ViewPosition": "AP"},
        ],
    )
    touch_image(jpg_root, "12345678", "111", "prev-a")
    touch_image(jpg_root, "12345678", "222", "curr-a")
    write_gold_tsv(
        gold_tsv,
        [
            {
                "patient_id": "12345678",
                "relationship_id": "rel-1",
                "subject_id": "curr-a_object",
                "object_id": "prev-a_object",
                "bbox": "lung",
                "comparison": "improved",
                "attribute": "attr-1",
                "sentence": "Sentence A",
                "bbox_coord_224_subject": "[1,2,3,4]",
                "bbox_coord_224_object": "[1,2,3,4]",
                "bbox_coord_original_subject": "[10,20,30,40]",
                "bbox_coord_original_object": "[10,20,30,40]",
                "label_name": "edema",
                "current_image_id": "curr-a",
                "previous_image_id": "prev-a",
            },
        ],
    )

    args = SimpleNamespace(
        gold_tsv=gold_tsv,
        mimic_metadata_csv=metadata_csv,
        mimic_jpg_root=jpg_root,
        output_dir=tmp_path / "out",
        same_view_only=True,
        allowed_label_names=None,
        normalize_to_ms_cxr_t_heads=False,
    )

    query_rows, pair_rows, summary = module.build_dataset(args)

    assert query_rows == []
    assert pair_rows == []
    assert summary["skipped_counts"]["mixed_view_pair"] == 1


def test_normalize_to_ms_cxr_t_heads_projects_labels_and_preserves_original_metadata(
    tmp_path: Path,
) -> None:
    module = load_module()
    metadata_csv = tmp_path / "metadata.csv"
    jpg_root = tmp_path / "files"
    gold_tsv = tmp_path / "gold.tsv"

    write_metadata_csv(
        metadata_csv,
        [
            {"dicom_id": "prev-a", "subject_id": "12345678", "study_id": "111", "ViewPosition": "PA"},
            {"dicom_id": "curr-a", "subject_id": "12345678", "study_id": "222", "ViewPosition": "PA"},
            {"dicom_id": "prev-b", "subject_id": "87654321", "study_id": "333", "ViewPosition": "PA"},
            {"dicom_id": "curr-b", "subject_id": "87654321", "study_id": "444", "ViewPosition": "PA"},
        ],
    )
    for subject_id, study_id, dicom_id in [
        ("12345678", "111", "prev-a"),
        ("12345678", "222", "curr-a"),
        ("87654321", "333", "prev-b"),
        ("87654321", "444", "curr-b"),
    ]:
        touch_image(jpg_root, subject_id, study_id, dicom_id)

    write_gold_tsv(
        gold_tsv,
        [
            {
                "patient_id": "12345678",
                "relationship_id": "rel-1",
                "subject_id": "curr-a_object",
                "object_id": "prev-a_object",
                "bbox": "lung",
                "comparison": "improved",
                "attribute": "attr-1",
                "sentence": "Sentence A",
                "bbox_coord_224_subject": "[1,2,3,4]",
                "bbox_coord_224_object": "[1,2,3,4]",
                "bbox_coord_original_subject": "[10,20,30,40]",
                "bbox_coord_original_object": "[10,20,30,40]",
                "label_name": "pulmonary edema/hazy opacity",
                "current_image_id": "curr-a",
                "previous_image_id": "prev-a",
            },
            {
                "patient_id": "87654321",
                "relationship_id": "rel-2",
                "subject_id": "curr-b_object",
                "object_id": "prev-b_object",
                "bbox": "lung",
                "comparison": "worsened",
                "attribute": "attr-2",
                "sentence": "Sentence B",
                "bbox_coord_224_subject": "[1,2,3,4]",
                "bbox_coord_224_object": "[1,2,3,4]",
                "bbox_coord_original_subject": "[10,20,30,40]",
                "bbox_coord_original_object": "[10,20,30,40]",
                "label_name": "pleural effusion",
                "current_image_id": "curr-b",
                "previous_image_id": "prev-b",
            },
            {
                "patient_id": "12345678",
                "relationship_id": "rel-3",
                "subject_id": "curr-a_object",
                "object_id": "prev-a_object",
                "bbox": "lung",
                "comparison": "no change",
                "attribute": "attr-3",
                "sentence": "Sentence C",
                "bbox_coord_224_subject": "[1,2,3,4]",
                "bbox_coord_224_object": "[1,2,3,4]",
                "bbox_coord_original_subject": "[10,20,30,40]",
                "bbox_coord_original_object": "[10,20,30,40]",
                "label_name": "mass/nodule (not otherwise specified)",
                "current_image_id": "curr-a",
                "previous_image_id": "prev-a",
            },
        ],
    )

    args = SimpleNamespace(
        gold_tsv=gold_tsv,
        mimic_metadata_csv=metadata_csv,
        mimic_jpg_root=jpg_root,
        output_dir=tmp_path / "out",
        same_view_only=False,
        allowed_label_names=["edema"],
        normalize_to_ms_cxr_t_heads=True,
    )

    query_rows, pair_rows, summary = module.build_dataset(args)

    assert len(query_rows) == 1
    assert len(pair_rows) == 1
    row = query_rows[0]
    assert row["query_finding"] == "edema"
    assert row["query_change"] == "improving"
    assert row["metadata"]["label_name"] == "pulmonary edema/hazy opacity"
    assert row["metadata"]["original_label_name"] == "pulmonary edema/hazy opacity"
    assert row["metadata"]["original_label_names"] == ["pulmonary edema/hazy opacity"]
    assert row["metadata"]["projected_query_finding"] == "edema"
    assert row["metadata"]["normalize_to_ms_cxr_t_heads"] is True
    assert summary["dataset_name"] == "chest_imagenome_gold_object_comparison_ms_cxr_t_heads"
    assert summary["normalize_to_ms_cxr_t_heads"] is True
    assert summary["comparison_counts"] == {"improving": 1}
    assert summary["top_label_counts"] == {"edema": 1}
    assert summary["top_original_label_counts"] == {"pulmonary edema/hazy opacity": 1}
    assert summary["skipped_counts"]["label_name_filtered"] == 1
    assert summary["skipped_counts"]["label_name_not_mapped_to_ms_cxr_t_head"] == 1
