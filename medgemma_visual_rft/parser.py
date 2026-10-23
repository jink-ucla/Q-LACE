from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

THINK_START_TAG = "<think>"
THINK_END_TAG = "</think>"
ANSWER_START_TAG = "<answer>"
ANSWER_END_TAG = "</answer>"
FINAL_ANSWER_PREFIX = "Final Answer:"
COORD_MIN = 0
COORD_MAX = 1000

_WRAPPED_COMPLETION_RE = re.compile(
    r"^\s*"
    r"<think>(?P<think>.*?)</think>"
    r"\s*"
    r"<answer>(?P<answer>.*?)</answer>"
    r"\s*$",
    re.DOTALL,
)


class ParseError(ValueError):
    """Base class for strict MedGemma parser failures."""


class PrefixNormalizationError(ParseError):
    """Raised when the optional prefix normalization step cannot apply."""


class WrapperParseError(ParseError):
    """Raised when the completion does not match the required tag wrapper."""


class JsonParseError(ParseError):
    """Raised when the answer payload is not valid JSON."""


class SchemaValidationError(ParseError):
    """Raised when the JSON payload is structurally invalid."""


class BoxValidationError(ParseError):
    """Raised when a box_2d value is malformed or out of bounds."""


@dataclass(frozen=True, slots=True)
class Box2DLabel:
    box_2d: tuple[int, int, int, int]
    label: str

    def as_dict(self) -> dict[str, Any]:
        return {"box_2d": list(self.box_2d), "label": self.label}


@dataclass(frozen=True, slots=True)
class ParseResult:
    think: str
    items: tuple[Box2DLabel, ...]
    raw_text: str
    normalized_text: str
    final_answer_prefix_stripped: bool = False

    @property
    def answer(self) -> list[dict[str, Any]]:
        return [item.as_dict() for item in self.items]


def normalize_final_answer_prefix(text: str) -> tuple[str, bool]:
    """Strip a single leading ``Final Answer:`` prefix if present."""

    stripped = text.lstrip()
    if not stripped.startswith(FINAL_ANSWER_PREFIX):
        return text, False
    remainder = stripped[len(FINAL_ANSWER_PREFIX) :]
    return remainder.lstrip(), True


def parse_wrapped_completion(text: str) -> tuple[str, str, str, bool]:
    """Return ``(think, answer, normalized_text, prefix_stripped)`` for valid output."""

    normalized_text, prefix_stripped = normalize_final_answer_prefix(text)
    match = _WRAPPED_COMPLETION_RE.fullmatch(normalized_text)
    if match is None:
        raise WrapperParseError(
            "Completion must match <think>...</think><answer>...</answer> with no extra text."
        )
    think = match.group("think")
    answer = match.group("answer")
    return think, answer, normalized_text, prefix_stripped


def parse_answer_list(answer_text: str) -> tuple[Box2DLabel, ...]:
    """Parse and strictly validate the JSON answer list."""

    try:
        payload = json.loads(answer_text)
    except json.JSONDecodeError as exc:
        raise JsonParseError(f"Answer block is not valid JSON: {exc.msg}") from exc

    if not isinstance(payload, list):
        raise SchemaValidationError("Answer payload must be a JSON list.")
    if not payload:
        raise SchemaValidationError("Answer payload must contain at least one object.")

    items: list[Box2DLabel] = []
    for index, raw_item in enumerate(payload):
        if not isinstance(raw_item, dict):
            raise SchemaValidationError(f"Answer item at index {index} must be a JSON object.")

        expected_keys = {"box_2d", "label"}
        actual_keys = set(raw_item.keys())
        if actual_keys != expected_keys:
            extra = sorted(actual_keys - expected_keys)
            missing = sorted(expected_keys - actual_keys)
            raise SchemaValidationError(
                "Answer item must contain exactly the keys 'box_2d' and 'label'. "
                f"Missing: {missing or '[]'} Extra: {extra or '[]'}."
            )

        label = raw_item["label"]
        if not isinstance(label, str):
            raise SchemaValidationError(f"Answer item at index {index} has a non-string label.")
        label = label.strip()
        if not label:
            raise SchemaValidationError(f"Answer item at index {index} has an empty label.")

        box = _parse_box_2d(raw_item["box_2d"], index=index)
        items.append(Box2DLabel(box_2d=box, label=label))

    return tuple(items)


def parse_completion(text: str) -> ParseResult:
    """Parse a strict MedGemma completion into structured boxes and labels."""

    think, answer_text, normalized_text, prefix_stripped = parse_wrapped_completion(text)
    items = parse_answer_list(answer_text)
    return ParseResult(
        think=think,
        items=items,
        raw_text=text,
        normalized_text=normalized_text,
        final_answer_prefix_stripped=prefix_stripped,
    )


def _parse_box_2d(raw_box: Any, *, index: int) -> tuple[int, int, int, int]:
    if not isinstance(raw_box, list):
        raise BoxValidationError(f"Answer item at index {index} must use a list for box_2d.")
    if len(raw_box) != 4:
        raise BoxValidationError(f"Answer item at index {index} must have exactly 4 box coordinates.")

    coords: list[int] = []
    for coord_index, value in enumerate(raw_box):
        if type(value) is not int:
            raise BoxValidationError(
                f"Answer item at index {index} has a non-integer coordinate at position {coord_index}."
            )
        if value < COORD_MIN or value > COORD_MAX:
            raise BoxValidationError(
                f"Answer item at index {index} has a coordinate outside the range 0..1000."
            )
        coords.append(value)

    y0, x0, y1, x1 = coords
    if not (y0 < y1):
        raise BoxValidationError(f"Answer item at index {index} must satisfy y0 < y1.")
    if not (x0 < x1):
        raise BoxValidationError(f"Answer item at index {index} must satisfy x0 < x1.")

    return y0, x0, y1, x1
