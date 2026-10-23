"""Strict schema helpers for longitudinal CXR change extraction outputs.

This module intentionally stays separate from the single-image localization
parser/reward stack. The longitudinal track has a different contract:

    {"changes": [{"anatomy": "...", "finding": "...", "change": "..."}]}

The stronger variant optionally adds:

    "box_current": [y0, x0, y1, x1]

The helpers below validate that contract and return canonicalized Python
objects that can later be consumed by reward code or evaluation scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Iterable, Mapping


DEFAULT_CHANGE_STATES = frozenset({"improving", "stable", "worsening"})

_SPACE_RE = re.compile(r"\s+")


class LongitudinalParseError(ValueError):
    """Base class for longitudinal output parse failures."""


class LongitudinalJsonParseError(LongitudinalParseError):
    """Raised when the completion is not valid strict JSON."""


class LongitudinalSchemaValidationError(LongitudinalParseError):
    """Raised when a parsed JSON object violates the output schema."""


@dataclass(frozen=True)
class IntervalChange:
    anatomy: str
    finding: str
    change: str
    box_current: tuple[int, int, int, int] | None = None


@dataclass(frozen=True)
class LongitudinalParseResult:
    changes: tuple[IntervalChange, ...]
    raw_text: str
    normalized_text: str


def _normalize_text(value: str, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise LongitudinalSchemaValidationError(f"`{field_name}` must be a string")
    normalized = _SPACE_RE.sub(" ", value.strip())
    if not normalized:
        raise LongitudinalSchemaValidationError(f"`{field_name}` must not be empty")
    if value != normalized:
        raise LongitudinalSchemaValidationError(
            f"`{field_name}` must not contain leading/trailing or repeated whitespace"
        )
    return normalized


def _validate_box_current(value: Any) -> tuple[int, int, int, int]:
    if not isinstance(value, list):
        raise LongitudinalSchemaValidationError("`box_current` must be a JSON list")
    if len(value) != 4:
        raise LongitudinalSchemaValidationError("`box_current` must contain exactly 4 coordinates")
    coords: list[int] = []
    for index, coord in enumerate(value):
        if type(coord) is not int:  # noqa: E721 - reject bools and non-ints.
            raise LongitudinalSchemaValidationError(f"`box_current[{index}]` must be an integer")
        if coord < 0 or coord > 1000:
            raise LongitudinalSchemaValidationError(
                f"`box_current[{index}]` must be within [0, 1000]"
            )
        coords.append(coord)
    y0, x0, y1, x1 = coords
    if y0 >= y1:
        raise LongitudinalSchemaValidationError("`box_current` must satisfy y0 < y1")
    if x0 >= x1:
        raise LongitudinalSchemaValidationError("`box_current` must satisfy x0 < x1")
    return (y0, x0, y1, x1)


def _normalize_vocab(values: Iterable[str] | None) -> frozenset[str] | None:
    if values is None:
        return None
    return frozenset(_normalize_text(value, field_name="vocabulary item") for value in values)


def _validate_change_item(
    item: Mapping[str, Any],
    *,
    allowed_findings: frozenset[str] | None,
    allowed_changes: frozenset[str] | None,
    allowed_anatomies: frozenset[str] | None,
    require_box_current: bool,
) -> IntervalChange:
    if not isinstance(item, Mapping):
        raise LongitudinalSchemaValidationError("Each entry in `changes` must be a JSON object")

    allowed_keys = {"anatomy", "finding", "change"}
    if "box_current" in item or require_box_current:
        allowed_keys.add("box_current")
    if set(item.keys()) != allowed_keys:
        raise LongitudinalSchemaValidationError(
            f"`changes` entries must contain exactly {sorted(allowed_keys)}"
        )

    anatomy = _normalize_text(item.get("anatomy"), field_name="anatomy")
    finding = _normalize_text(item.get("finding"), field_name="finding")
    change = _normalize_text(item.get("change"), field_name="change")

    if allowed_anatomies is not None and anatomy not in allowed_anatomies:
        raise LongitudinalSchemaValidationError(f"Unsupported anatomy: {anatomy!r}")
    if allowed_findings is not None and finding not in allowed_findings:
        raise LongitudinalSchemaValidationError(f"Unsupported finding: {finding!r}")
    if allowed_changes is not None and change not in allowed_changes:
        raise LongitudinalSchemaValidationError(f"Unsupported change state: {change!r}")

    box_current: tuple[int, int, int, int] | None = None
    if require_box_current:
        box_current = _validate_box_current(item.get("box_current"))
    elif "box_current" in item:
        box_current = _validate_box_current(item["box_current"])

    return IntervalChange(
        anatomy=anatomy,
        finding=finding,
        change=change,
        box_current=box_current,
    )


def canonical_change_sort_key(item: IntervalChange) -> tuple[str, str, str, tuple[int, int, int, int] | None]:
    return (item.anatomy, item.finding, item.change, item.box_current)


def serialize_longitudinal_solution(
    changes: Iterable[IntervalChange | Mapping[str, Any]],
) -> str:
    """Serialize a deterministic strict-JSON longitudinal target."""

    normalized_changes: list[IntervalChange] = []
    for item in changes:
        if isinstance(item, IntervalChange):
            normalized_changes.append(item)
            continue
        if not isinstance(item, Mapping):
            raise LongitudinalSchemaValidationError(
                "serialize_longitudinal_solution expects IntervalChange objects or mappings"
            )
        filtered = {
            key: item[key]
            for key in ("anatomy", "finding", "change", "box_current")
            if key in item
        }
        normalized_changes.append(
            _validate_change_item(
                filtered,
                allowed_findings=None,
                allowed_changes=None,
                allowed_anatomies=None,
                require_box_current=False,
            )
        )

    payload = {
        "changes": [
            {
                **(
                    {"box_current": list(change.box_current)}
                    if change.box_current is not None
                    else {}
                ),
                "anatomy": change.anatomy,
                "finding": change.finding,
                "change": change.change,
            }
            for change in sorted(normalized_changes, key=canonical_change_sort_key)
        ]
    }
    # Preserve the canonical key order required by the contract.
    for item in payload["changes"]:
        if "box_current" in item:
            box_current = item.pop("box_current")
            ordered = {
                "anatomy": item["anatomy"],
                "finding": item["finding"],
                "change": item["change"],
                "box_current": box_current,
            }
            item.clear()
            item.update(ordered)
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))


def parse_longitudinal_completion(
    text: str,
    *,
    allowed_findings: Iterable[str] | None = None,
    allowed_changes: Iterable[str] | None = DEFAULT_CHANGE_STATES,
    allowed_anatomies: Iterable[str] | None = None,
    require_box_current: bool = False,
) -> LongitudinalParseResult:
    """Parse a strict JSON longitudinal completion.

    The completion must be *only* the JSON object. Free-form prose, markdown
    fences, and wrapper tags are treated as parse failures by design.
    """

    if not isinstance(text, str):
        raise LongitudinalSchemaValidationError("Completion must be a string")
    normalized_text = text.strip()
    if not normalized_text:
        raise LongitudinalSchemaValidationError("Completion must not be empty")

    try:
        payload = json.loads(normalized_text)
    except json.JSONDecodeError as exc:
        raise LongitudinalJsonParseError(f"Completion is not valid strict JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise LongitudinalSchemaValidationError("Top-level payload must be a JSON object")
    if set(payload.keys()) != {"changes"}:
        raise LongitudinalSchemaValidationError("Top-level payload must contain only `changes`")

    changes_value = payload["changes"]
    if not isinstance(changes_value, list):
        raise LongitudinalSchemaValidationError("`changes` must be a JSON list")

    normalized_findings = _normalize_vocab(allowed_findings)
    normalized_changes = _normalize_vocab(allowed_changes)
    normalized_anatomies = _normalize_vocab(allowed_anatomies)

    changes = tuple(
        sorted(
            (
                _validate_change_item(
                    item,
                    allowed_findings=normalized_findings,
                    allowed_changes=normalized_changes,
                    allowed_anatomies=normalized_anatomies,
                    require_box_current=require_box_current,
                )
                for item in changes_value
            ),
            key=canonical_change_sort_key,
        )
    )
    return LongitudinalParseResult(
        changes=changes,
        raw_text=text,
        normalized_text=normalized_text,
    )
