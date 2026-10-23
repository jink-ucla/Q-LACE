from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from medgemma_visual_rft.longitudinal_projection import (  # noqa: E402
    iter_ms_cxr_t_labeled_findings,
    project_changes_to_ms_cxr_t_progression,
    select_representative_ms_cxr_t_changes,
)
from medgemma_visual_rft.longitudinal_schema import IntervalChange  # noqa: E402


def test_projection_uses_finding_precedence_across_multiple_anatomies() -> None:
    projected = project_changes_to_ms_cxr_t_progression(
        [
            IntervalChange(anatomy="left lower lung zone", finding="pleural effusion", change="stable"),
            IntervalChange(anatomy="right lower lung zone", finding="pleural effusion", change="worsening"),
            IntervalChange(anatomy="bilateral lungs", finding="edema", change="improving"),
        ]
    )

    assert projected["pleural_effusion"] == "worsening"
    assert projected["edema"] == "improving"
    assert projected["consolidation"] is None


def test_projection_keeps_stable_empty_and_unsupported_findings_as_null() -> None:
    projected = project_changes_to_ms_cxr_t_progression(
        [
            IntervalChange(anatomy="chest", finding="opacity", change="stable"),
            IntervalChange(anatomy="chest", finding="atelectasis", change="worsening"),
        ]
    )

    assert all(value is None for value in projected.values())


def test_iter_labeled_findings_skips_null_heads() -> None:
    labeled = list(
        iter_ms_cxr_t_labeled_findings(
            [
                {"anatomy": "right lung", "finding": "pneumothorax", "change": "stable"},
                {"anatomy": "left base", "finding": "pleural effusion", "change": "improving"},
            ]
        )
    )

    assert labeled == [
        ("pleural_effusion", "improving"),
        ("pneumothorax", "stable"),
    ]


def test_representative_projection_prefers_confidence_then_specificity() -> None:
    projected = select_representative_ms_cxr_t_changes(
        [
            {
                "anatomy": "chest",
                "finding": "pleural effusion",
                "change": "stable",
                "confidence": 0.95,
                "evidence": "trace pleural effusion is unchanged",
                "source": "weak",
            },
            {
                "anatomy": "left pleura",
                "finding": "pleural effusion",
                "change": "stable",
                "confidence": 0.95,
                "evidence": "small left pleural effusion is stable",
                "source": "weak",
            },
            {
                "anatomy": "bilateral lungs",
                "finding": "edema",
                "change": "improving",
                "confidence": 0.70,
                "evidence": "edema is improving",
                "source": "weak",
            },
        ]
    )

    assert [item.benchmark_finding for item in projected] == ["edema", "pleural_effusion"]
    pleural_effusion = projected[1]
    assert pleural_effusion.change == "stable"
    assert pleural_effusion.representative.anatomy == "left pleura"
    assert pleural_effusion.representative_confidence == 0.95
    assert pleural_effusion.display_finding == "pleural effusion"
