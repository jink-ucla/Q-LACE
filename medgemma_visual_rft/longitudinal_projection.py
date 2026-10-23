"""Projection helpers for benchmark-facing longitudinal evaluation views."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable, Mapping
from typing import Any

from medgemma_visual_rft.longitudinal_schema import IntervalChange


MS_CXR_T_FINDINGS = (
    "consolidation",
    "edema",
    "pleural_effusion",
    "pneumonia",
    "pneumothorax",
)

_FINDING_ALIASES = {
    "consolidation": "consolidation",
    "edema": "edema",
    "pleural effusion": "pleural_effusion",
    "pleural_effusion": "pleural_effusion",
    "pneumonia": "pneumonia",
    "pneumothorax": "pneumothorax",
}

_CHANGE_PRECEDENCE = {
    "worsening": 3,
    "improving": 2,
    "stable": 1,
}

MS_CXR_T_FINDING_DISPLAY = {
    "consolidation": "consolidation",
    "edema": "edema",
    "pleural_effusion": "pleural effusion",
    "pneumonia": "pneumonia",
    "pneumothorax": "pneumothorax",
}


@dataclass(frozen=True)
class ProjectedMsCxrTChange:
    benchmark_finding: str
    display_finding: str
    change: str
    representative: IntervalChange
    representative_confidence: float | None
    representative_evidence: str | None
    representative_source: str | None


def _coerce_interval_change(item: IntervalChange | Mapping[str, Any]) -> IntervalChange:
    if isinstance(item, IntervalChange):
        return item
    if not isinstance(item, Mapping):
        raise TypeError("Expected IntervalChange or mapping")
    return IntervalChange(
        anatomy=str(item["anatomy"]),
        finding=str(item["finding"]),
        change=str(item["change"]),
        box_current=None,
    )


def normalize_ms_cxr_t_finding(value: str) -> str | None:
    return _FINDING_ALIASES.get(value.strip().lower())


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return float(stripped)
        except ValueError:
            return None
    return None


def _optional_clean_str(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _anatomy_specificity(anatomy: str) -> int:
    normalized = anatomy.strip().lower()
    if normalized in {"", "chest", "unspecified"}:
        return 0
    if any(
        token in normalized
        for token in (
            "left",
            "right",
            "upper",
            "lower",
            "middle",
            "mid",
            "lobe",
            "zone",
            "base",
            "basilar",
            "apical",
            "retrocardiac",
            "costophrenic",
        )
    ):
        return 2
    return 1


def select_representative_ms_cxr_t_changes(
    changes: Iterable[IntervalChange | Mapping[str, Any]],
) -> tuple[ProjectedMsCxrTChange, ...]:
    grouped: dict[str, list[tuple[IntervalChange, float | None, str, str]]] = {
        finding: [] for finding in MS_CXR_T_FINDINGS
    }

    for item in changes:
        interval_change = _coerce_interval_change(item)
        normalized_finding = normalize_ms_cxr_t_finding(interval_change.finding)
        if normalized_finding is None:
            continue
        if interval_change.change not in _CHANGE_PRECEDENCE:
            continue
        confidence = _optional_float(item.get("confidence")) if isinstance(item, Mapping) else None
        evidence = _optional_clean_str(item.get("evidence")) if isinstance(item, Mapping) else None
        source = _optional_clean_str(item.get("source")) if isinstance(item, Mapping) else None
        grouped[normalized_finding].append((interval_change, confidence, evidence or "", source or ""))

    selected: list[ProjectedMsCxrTChange] = []
    for finding in MS_CXR_T_FINDINGS:
        candidates = grouped[finding]
        if not candidates:
            continue
        best_interval_change, confidence, evidence, source = sorted(
            candidates,
            key=lambda item: (
                -_CHANGE_PRECEDENCE[item[0].change],
                -(item[1] if item[1] is not None else -1.0),
                -_anatomy_specificity(item[0].anatomy),
                item[0].anatomy.lower(),
                item[2].lower(),
                item[3].lower(),
            ),
        )[0]
        selected.append(
            ProjectedMsCxrTChange(
                benchmark_finding=finding,
                display_finding=MS_CXR_T_FINDING_DISPLAY[finding],
                change=best_interval_change.change,
                representative=best_interval_change,
                representative_confidence=confidence,
                representative_evidence=evidence or None,
                representative_source=source or None,
            )
        )
    return tuple(selected)


def project_changes_to_ms_cxr_t_progression(
    changes: Iterable[IntervalChange | Mapping[str, Any]],
) -> dict[str, str | None]:
    """Project anatomy-grounded tuples to the 5 benchmark disease heads.

    The collapse rule intentionally ignores anatomy and resolves multiple tuples
    for the same disease with the precedence:

        worsening > improving > stable > null
    """

    projected: dict[str, str | None] = {finding: None for finding in MS_CXR_T_FINDINGS}
    for item in select_representative_ms_cxr_t_changes(changes):
        projected[item.benchmark_finding] = item.change
    return projected


def iter_ms_cxr_t_labeled_findings(
    changes: Iterable[IntervalChange | Mapping[str, Any]],
) -> Iterable[tuple[str, str]]:
    projected = project_changes_to_ms_cxr_t_progression(changes)
    for finding in MS_CXR_T_FINDINGS:
        label = projected[finding]
        if label is not None:
            yield finding, label
