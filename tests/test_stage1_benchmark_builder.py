from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "dataset" / "build_mimic_cxr_longitudinal_stage1_benchmark.py"


def load_module():
    spec = importlib.util.spec_from_file_location("build_mimic_cxr_longitudinal_stage1_benchmark", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load Stage-1 benchmark builder module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_row(
    *,
    pair_id: str,
    patient_id: str,
    prior_view: str = "PA",
    current_view: str = "PA",
    label_status: str = "weak_candidate",
    changes: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "pair_id": pair_id,
        "source_id": pair_id,
        "patient_id": patient_id,
        "prior_image_path": "/tmp/prior.jpg",
        "current_image_path": "/tmp/current.jpg",
        "prior_view_position": prior_view,
        "current_view_position": current_view,
        "weak": {
            "label_status": label_status,
            "changes": changes or [],
        },
        "metadata": {"builder": "test"},
    }


def test_builder_projects_to_single_benchmark_head_with_unspecified_solution(tmp_path: Path) -> None:
    module = load_module()
    rows = [
        make_row(
            pair_id="p1:s1:s2",
            patient_id="p1",
            changes=[
                {
                    "anatomy": "chest",
                    "finding": "pleural effusion",
                    "change": "stable",
                    "confidence": 0.5,
                    "evidence": "trace pleural effusion is unchanged",
                    "source": "weak",
                },
                {
                    "anatomy": "left pleura",
                    "finding": "pleural effusion",
                    "change": "stable",
                    "confidence": 0.9,
                    "evidence": "small left pleural effusion is stable",
                    "source": "weak",
                },
                {
                    "anatomy": "lungs",
                    "finding": "edema",
                    "change": "improving",
                    "confidence": 0.8,
                    "evidence": "pulmonary edema is improving",
                    "source": "weak",
                },
                {
                    "anatomy": "chest",
                    "finding": "opacity",
                    "change": "worsening",
                    "confidence": 0.7,
                    "evidence": "opacity is worse",
                    "source": "weak",
                },
            ],
        )
    ]
    input_jsonl = tmp_path / "weak_tuples.jsonl"
    module.write_jsonl(input_jsonl, rows)

    dataset_rows, summary = module.build_dataset(
        SimpleNamespace(
            input_jsonl=input_jsonl,
            same_view_only=True,
            include_statuses=["weak_candidate"],
            split_policy="patient_hash",
            split_seed=42,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            max_rows=None,
        )
    )

    assert summary["count"] == 2
    assert summary["query_finding_counts"] == {"edema": 1, "pleural effusion": 1}
    assert summary["query_change_counts"] == {"improving": 1, "stable": 1}

    pleural_effusion_row = next(row for row in dataset_rows if row["query_finding"] == "pleural effusion")
    assert pleural_effusion_row["query_anatomy"] == "unspecified"
    assert pleural_effusion_row["solution"] == (
        '{"changes":[{"anatomy":"unspecified","finding":"pleural effusion","change":"stable"}]}'
    )
    assert pleural_effusion_row["metadata"]["representative_anatomy"] == "left pleura"
    assert pleural_effusion_row["metadata"]["representative_confidence"] == 0.9
    assert pleural_effusion_row["metadata"]["projected_benchmark_finding"] == "pleural_effusion"


def test_builder_skips_mixed_view_and_non_projected_rows(tmp_path: Path) -> None:
    module = load_module()
    rows = [
        make_row(
            pair_id="p1:s1:s2",
            patient_id="p1",
            prior_view="PA",
            current_view="AP",
            changes=[
                {
                    "anatomy": "chest",
                    "finding": "pneumothorax",
                    "change": "worsening",
                    "confidence": 0.8,
                    "evidence": "new pneumothorax",
                    "source": "weak",
                }
            ],
        ),
        make_row(
            pair_id="p2:s2:s3",
            patient_id="p2",
            changes=[
                {
                    "anatomy": "chest",
                    "finding": "opacity",
                    "change": "stable",
                    "confidence": 0.8,
                    "evidence": "opacity unchanged",
                    "source": "weak",
                }
            ],
        ),
        make_row(
            pair_id="p3:s3:s4",
            patient_id="p3",
            label_status="weak_stable_empty",
            changes=[],
        ),
    ]
    input_jsonl = tmp_path / "weak_tuples.jsonl"
    module.write_jsonl(input_jsonl, rows)

    dataset_rows, summary = module.build_dataset(
        SimpleNamespace(
            input_jsonl=input_jsonl,
            same_view_only=True,
            include_statuses=["weak_candidate"],
            split_policy="patient_hash",
            split_seed=42,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            max_rows=None,
        )
    )

    assert dataset_rows == []
    assert summary["count"] == 0
    assert summary["skipped_counts"] == {
        "mixed_view_pair": 1,
        "pairs_without_benchmark_projection": 1,
        "filtered_label_status": 1,
    }
