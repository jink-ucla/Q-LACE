"""Validation helpers for unlabeled prior/current MIMIC-CXR chronology manifests.

This contract is intentionally separate from the single-image localization
pilot schema. The manifest rows here are pairing/provenance artifacts rather
than supervised training examples.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


MANIFEST_REQUIRED_KEYS = (
    "pair_id",
    "patient_id",
    "source_id",
    "split",
    "prior_study_id",
    "current_study_id",
    "prior_dicom_id",
    "current_dicom_id",
    "prior_image_path",
    "current_image_path",
    "prior_view_position",
    "current_view_position",
    "problem",
    "metadata",
)

SUPPORTED_SPLITS = frozenset({"train", "validate", "test"})
SUPPORTED_VIEWS = frozenset({"AP", "PA"})
CANONICAL_REPORT_SECTION_KEYS = frozenset({"comparison", "findings", "impression"})


class LongitudinalManifestValidationError(ValueError):
    """Raised when a chronology manifest row violates the current contract."""


def load_manifest_jsonl(path: str | Path) -> list[dict[str, Any]]:
    resolved = Path(path)
    rows: list[dict[str, Any]] = []
    with resolved.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise LongitudinalManifestValidationError(
                    f"Invalid JSON on line {line_number} of {resolved}: {exc}"
                ) from exc
            if not isinstance(payload, dict):
                raise LongitudinalManifestValidationError(
                    f"Expected an object on line {line_number} of {resolved}"
                )
            rows.append(payload)
    return rows


def _require_str(record: Mapping[str, Any], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str):
        raise LongitudinalManifestValidationError(f"`{key}` must be a string")
    if not value.strip():
        raise LongitudinalManifestValidationError(f"`{key}` must not be empty")
    return value


def _validate_image_path(path_value: str, *, key: str) -> str:
    path = Path(path_value)
    if not path.is_absolute():
        raise LongitudinalManifestValidationError(f"`{key}` must be an absolute path")
    if not path.exists():
        raise LongitudinalManifestValidationError(f"Image file does not exist: {path}")
    return path_value


def _validate_report_path(path_value: Any, *, key: str) -> str | None:
    if path_value is None:
        return None
    if not isinstance(path_value, str):
        raise LongitudinalManifestValidationError(f"`{key}` must be a string when present")
    if not path_value.strip():
        raise LongitudinalManifestValidationError(f"`{key}` must not be empty when present")
    return path_value


def _validate_report_sections(value: Any, *, key: str) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise LongitudinalManifestValidationError(f"`{key}` must be a JSON object")
    normalized: dict[str, str] = {}
    for section_name, section_text in value.items():
        if not isinstance(section_name, str) or not section_name.strip():
            raise LongitudinalManifestValidationError(
                f"`{key}` keys must be non-empty strings"
            )
        if not isinstance(section_text, str):
            raise LongitudinalManifestValidationError(
                f"`{key}[{section_name!r}]` must be a string"
            )
        normalized_name = section_name.strip().lower()
        normalized_text = section_text.strip()
        if not normalized_text:
            raise LongitudinalManifestValidationError(
                f"`{key}[{section_name!r}]` must not be empty"
            )
        normalized[normalized_name] = normalized_text
    return normalized


def validate_manifest_record(record: Mapping[str, Any]) -> dict[str, Any]:
    pair_id = _require_str(record, "pair_id")
    patient_id = _require_str(record, "patient_id")
    source_id = _require_str(record, "source_id")
    split = _require_str(record, "split")
    if split not in SUPPORTED_SPLITS:
        raise LongitudinalManifestValidationError(
            f"`split` must be one of {sorted(SUPPORTED_SPLITS)}, got {split!r}"
        )
    if source_id != pair_id:
        raise LongitudinalManifestValidationError("`source_id` must match `pair_id`")
    if not pair_id.startswith(f"{patient_id}:"):
        raise LongitudinalManifestValidationError("`pair_id` must start with `patient_id:`")

    prior_study_id = _require_str(record, "prior_study_id")
    current_study_id = _require_str(record, "current_study_id")
    prior_dicom_id = _require_str(record, "prior_dicom_id")
    current_dicom_id = _require_str(record, "current_dicom_id")
    if prior_study_id == current_study_id:
        raise LongitudinalManifestValidationError("Prior and current study ids must differ")
    if prior_dicom_id == current_dicom_id:
        raise LongitudinalManifestValidationError("Prior and current dicom ids must differ")

    prior_image_path = _validate_image_path(_require_str(record, "prior_image_path"), key="prior_image_path")
    current_image_path = _validate_image_path(
        _require_str(record, "current_image_path"), key="current_image_path"
    )
    if prior_image_path == current_image_path:
        raise LongitudinalManifestValidationError("Prior and current image paths must differ")

    prior_view_position = _require_str(record, "prior_view_position").upper()
    current_view_position = _require_str(record, "current_view_position").upper()
    if prior_view_position not in SUPPORTED_VIEWS:
        raise LongitudinalManifestValidationError(
            f"`prior_view_position` must be one of {sorted(SUPPORTED_VIEWS)}"
        )
    if current_view_position not in SUPPORTED_VIEWS:
        raise LongitudinalManifestValidationError(
            f"`current_view_position` must be one of {sorted(SUPPORTED_VIEWS)}"
        )

    delta_days = record.get("delta_days")
    if delta_days is not None:
        if type(delta_days) is not int:  # noqa: E721 - reject bools and non-ints.
            raise LongitudinalManifestValidationError("`delta_days` must be an integer when present")
        if delta_days < 0:
            raise LongitudinalManifestValidationError("`delta_days` must be non-negative")

    problem = _require_str(record, "problem")

    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        raise LongitudinalManifestValidationError("`metadata` must be a JSON object")
    builder = _require_str(metadata, "builder")
    task_family = _require_str(metadata, "task_family")
    label_status = _require_str(metadata, "label_status")
    if task_family != "longitudinal_cxr_change_rft":
        raise LongitudinalManifestValidationError(
            f"`metadata.task_family` must be 'longitudinal_cxr_change_rft', got {task_family!r}"
        )
    if label_status != "unlabeled_manifest":
        raise LongitudinalManifestValidationError(
            f"`metadata.label_status` must be 'unlabeled_manifest', got {label_status!r}"
        )

    prior_report_path = _validate_report_path(record.get("prior_report_path"), key="prior_report_path")
    current_report_path = _validate_report_path(
        record.get("current_report_path"), key="current_report_path"
    )
    prior_report_sections = _validate_report_sections(
        record.get("prior_report_sections"), key="prior_report_sections"
    )
    current_report_sections = _validate_report_sections(
        record.get("current_report_sections"), key="current_report_sections"
    )

    return {
        "pair_id": pair_id,
        "patient_id": patient_id,
        "source_id": source_id,
        "split": split,
        "prior_study_id": prior_study_id,
        "current_study_id": current_study_id,
        "prior_dicom_id": prior_dicom_id,
        "current_dicom_id": current_dicom_id,
        "prior_image_path": prior_image_path,
        "current_image_path": current_image_path,
        "prior_view_position": prior_view_position,
        "current_view_position": current_view_position,
        "delta_days": delta_days,
        "problem": problem,
        "prior_report_path": prior_report_path,
        "current_report_path": current_report_path,
        "prior_report_sections": prior_report_sections,
        "current_report_sections": current_report_sections,
        "metadata": {
            "builder": builder,
            "task_family": task_family,
            "label_status": label_status,
        },
    }
