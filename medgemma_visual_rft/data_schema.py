"""Canonical schema helpers for MedGemma Visual-RFT pilot records.

The pilot dataset is stored with a few convenience fields, but the locked
record contract for training/evaluation is the minimal 7-field schema:

    image_path, problem, solution, label, box_2d, split, source_id

This module validates that contract and can canonicalize the raw pilot rows
into that shape without involving parser or reward code.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping


CANONICAL_KEYS = (
    "image_path",
    "problem",
    "solution",
    "label",
    "box_2d",
    "split",
    "source_id",
)

SUPPORTED_SPLITS = frozenset({"train", "val", "test"})

_SOLUTION_RE = re.compile(
    r"^<think>(?P<think>.*?)</think><answer>(?P<answer>.*)</answer>$",
    re.DOTALL,
)


class SchemaValidationError(ValueError):
    """Raised when a record violates the canonical schema contract."""


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSONL file into memory."""

    resolved = Path(path)
    rows: list[dict[str, Any]] = []
    with resolved.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:  # pragma: no cover - defensive.
                raise SchemaValidationError(
                    f"Invalid JSON on line {line_number} of {resolved}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise SchemaValidationError(
                    f"Expected an object on line {line_number} of {resolved}"
                )
            rows.append(row)
    return rows


def _require_str(record: Mapping[str, Any], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str):
        raise SchemaValidationError(f"`{key}` must be a string")
    if not value.strip():
        raise SchemaValidationError(f"`{key}` must not be empty")
    return value


def _require_split(record: Mapping[str, Any], split: str | None) -> str:
    split_value = split if split is not None else record.get("split")
    if not isinstance(split_value, str):
        raise SchemaValidationError("`split` must be provided as a string")
    if split_value not in SUPPORTED_SPLITS:
        raise SchemaValidationError(
            f"`split` must be one of {sorted(SUPPORTED_SPLITS)}, got {split_value!r}"
        )
    return split_value


def _validate_image_path(image_path: str, dataset_root: str | Path | None) -> None:
    path = Path(image_path)
    if path.is_absolute():
        raise SchemaValidationError("`image_path` must be relative to the dataset root")
    if ".." in path.parts:
        raise SchemaValidationError("`image_path` must not traverse parent directories")
    if dataset_root is not None:
        resolved = Path(dataset_root) / path
        if not resolved.exists():
            raise SchemaValidationError(f"Image file does not exist: {resolved}")


def _require_box_2d(value: Any) -> list[int]:
    if not isinstance(value, list):
        raise SchemaValidationError("`box_2d` must be a JSON list")
    if len(value) != 4:
        raise SchemaValidationError("`box_2d` must contain exactly 4 coordinates")
    coords: list[int] = []
    for index, coord in enumerate(value):
        if type(coord) is not int:  # noqa: E721 - reject bools and non-ints.
            raise SchemaValidationError(f"`box_2d[{index}]` must be an integer")
        if coord < 0 or coord > 1000:
            raise SchemaValidationError(f"`box_2d[{index}]` must be within [0, 1000]")
        coords.append(coord)
    y0, x0, y1, x1 = coords
    if y0 >= y1:
        raise SchemaValidationError("`box_2d` must satisfy y0 < y1")
    if x0 >= x1:
        raise SchemaValidationError("`box_2d` must satisfy x0 < x1")
    return coords


def _require_label(value: Any) -> str:
    if not isinstance(value, str):
        raise SchemaValidationError("`label` must be a string")
    normalized = " ".join(value.split())
    if not normalized:
        raise SchemaValidationError("`label` must not be empty")
    if value != normalized:
        raise SchemaValidationError("`label` must not contain leading/trailing or repeated whitespace")
    return value


def _parse_solution(solution: str) -> tuple[str, list[int]]:
    match = _SOLUTION_RE.fullmatch(solution)
    if not match:
        raise SchemaValidationError(
            "Solution must match `<think>...</think><answer>[{\"box_2d\": [...], \"label\": \"...\"}]</answer>`"
        )
    answer_text = match.group("answer")
    try:
        payload = json.loads(answer_text)
    except json.JSONDecodeError as exc:
        raise SchemaValidationError(f"Invalid JSON inside `<answer>`: {exc}") from exc
    if not isinstance(payload, list):
        raise SchemaValidationError("`<answer>` payload must be a JSON list")
    if len(payload) != 1:
        raise SchemaValidationError("`<answer>` payload must contain exactly one object")
    item = payload[0]
    if not isinstance(item, dict):
        raise SchemaValidationError("`<answer>` payload item must be a JSON object")
    if set(item.keys()) != {"box_2d", "label"}:
        raise SchemaValidationError("`<answer>` payload objects must contain only `box_2d` and `label`")
    box_2d = _require_box_2d(item["box_2d"])
    label = _require_label(item["label"])
    return label, box_2d


def validate_canonical_record(
    record: Mapping[str, Any],
    *,
    dataset_root: str | Path | None = None,
) -> dict[str, Any]:
    """Validate a record that already follows the canonical 7-field schema."""

    image_path = _require_str(record, "image_path")
    problem = _require_str(record, "problem")
    solution = _require_str(record, "solution")
    label = _require_str(record, "label")
    split = _require_split(record, None)
    source_id = _require_str(record, "source_id")
    box_2d = _require_box_2d(record.get("box_2d"))

    parsed_label, parsed_box_2d = _parse_solution(solution)
    if label != parsed_label:
        raise SchemaValidationError("`label` must match the label encoded in `solution`")
    if box_2d != parsed_box_2d:
        raise SchemaValidationError("`box_2d` must match the coordinates encoded in `solution`")

    _validate_image_path(image_path, dataset_root)

    return {
        "image_path": image_path,
        "problem": problem,
        "solution": solution,
        "label": label,
        "box_2d": box_2d,
        "split": split,
        "source_id": source_id,
    }


def canonicalize_record(
    record: Mapping[str, Any],
    *,
    split: str | None = None,
    dataset_root: str | Path | None = None,
) -> dict[str, Any]:
    """Project a raw pilot row into the canonical 7-field schema.

    The pilot corpus stores useful provenance metadata alongside the core
    fields. This helper extracts the solution payload, checks that the raw row
    is internally consistent, and returns the minimal canonical record.
    """

    image_path = _require_str(record, "image_path")
    problem = _require_str(record, "problem")
    solution = _require_str(record, "solution")
    source_id = _require_str(record, "source_id")
    split_value = _require_split(record, split)
    parsed_label, parsed_box_2d = _parse_solution(solution)

    _validate_image_path(image_path, dataset_root)

    if "label" in record and record["label"] is not None and str(record["label"]) != parsed_label:
        raise SchemaValidationError("Raw `label` field does not match the label encoded in `solution`")
    if "box_2d" in record and record["box_2d"] is not None and list(record["box_2d"]) != parsed_box_2d:
        raise SchemaValidationError("Raw `box_2d` field does not match the coordinates encoded in `solution`")

    return validate_canonical_record(
        {
            "image_path": image_path,
            "problem": problem,
            "solution": solution,
            "label": parsed_label,
            "box_2d": parsed_box_2d,
            "split": split_value,
            "source_id": source_id,
        },
        dataset_root=dataset_root,
    )
