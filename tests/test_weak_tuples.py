from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from dataset.build_mimic_cxr_longitudinal_weak_tuples import (  # noqa: E402
    build_summary,
    extract_weak_changes,
)


def _base_row() -> dict[str, object]:
    return {
        "pair_id": "123:1:2",
        "patient_id": "123",
        "source_id": "123:1:2",
        "split": "train",
        "prior_report_sections": {
            "findings": "No pleural effusion.",
            "impression": "No acute cardiopulmonary process.",
        },
        "current_report_sections": {},
        "metadata": {"label_status": "unlabeled_manifest"},
    }


def test_extract_weak_changes_finds_change_tuple():
    row = _base_row()
    row["current_report_sections"] = {
        "comparison": "Compared to the prior study.",
        "findings": "There is a small left pleural effusion which has increased since the prior study.",
        "impression": "Worsening left pleural effusion.",
    }
    weak = extract_weak_changes(row, min_confidence=0.0)
    assert weak["label_status"] == "weak_candidate"
    assert weak["changes"]
    first = weak["changes"][0]
    assert first["change"] == "worsening"
    assert first["finding"] == "pleural effusion"
    assert first["anatomy"] in {"left pleura", "chest"}


def test_extract_weak_changes_emits_empty_list_for_stable_reports():
    row = _base_row()
    row["current_report_sections"] = {
        "comparison": "Chest radiograph comparison.",
        "findings": "No focal consolidation, pleural effusion, or pneumothorax.",
        "impression": "No acute cardiopulmonary abnormality.",
    }
    weak = extract_weak_changes(row, min_confidence=0.0)
    assert weak["changes"] == []
    assert weak["label_status"] == "weak_stable_empty"


def test_extract_weak_changes_drops_negated_new_findings():
    row = _base_row()
    row["current_report_sections"] = {
        "findings": "No new focal consolidation or pleural effusion.",
        "impression": "No acute cardiopulmonary process.",
    }
    weak = extract_weak_changes(row, min_confidence=0.0)
    assert weak["changes"] == []


def test_build_summary_counts_rows_and_tuples():
    rows = [
        {"weak": {"label_status": "weak_candidate", "changes": [{"anatomy": "chest", "finding": "edema", "change": "worsening"}]}},
        {"weak": {"label_status": "weak_stable_empty", "changes": []}},
        {"weak": {"label_status": "weak_unlabeled", "changes": []}},
    ]
    summary = build_summary(rows)
    assert summary["count"] == 3
    assert summary["total_candidate_tuples"] == 1
    assert summary["label_status_counts"]["weak_candidate"] == 1
    assert summary["change_counts"]["worsening"] == 1
