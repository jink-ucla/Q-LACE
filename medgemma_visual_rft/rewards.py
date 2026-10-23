"""Reward helpers for the MedGemma Visual-RFT pilot.

This module is intentionally trainer-agnostic. It provides:
- strict shell/format checks for the MedGemma output contract
- strict parse gating for the single-box pilot schema
- normalized IoU/GIoU helpers for `[y0, x0, y1, x1]` boxes in `0..1000`
- deterministic single-box reward composition

The canonical training target does not include a leading ``Final Answer:``
prefix. If that prefix appears, normalization is handled only by the parser
helper; the raw-shell format reward still stays strict.
"""

from __future__ import annotations

import importlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

__all__ = [
    "ParsedSingleBox",
    "DEFAULT_FORMAT_WEIGHT",
    "DEFAULT_GEO_WEIGHT",
    "DEFAULT_LABEL_WEIGHT",
    "DEFAULT_METRIC",
    "format_reward",
    "parse_gate_reward",
    "normalized_iou",
    "normalized_giou",
    "single_box_reward",
    "single_box_iou_reward",
    "single_box_giou_reward",
    "reward_funcs_registry",
]

# Default reward composition for the single-box pilot.
DEFAULT_FORMAT_WEIGHT = 0.10
DEFAULT_LABEL_WEIGHT = 0.20
DEFAULT_GEO_WEIGHT = 0.70
DEFAULT_METRIC = "giou"

_SHELL_RE = re.compile(
    r"^\s*<think>(?P<think>.*?)</think>\s*<answer>(?P<answer>.*?)</answer>\s*$",
    re.DOTALL,
)
_JSON_OBJECT_KEYS = {"box_2d", "label"}


@dataclass(frozen=True)
class ParsedSingleBox:
    """Canonical single-box record extracted from a completion or solution."""

    box_2d: tuple[int, int, int, int]
    label: str
    think: str = ""
    raw_answer: str = ""


@dataclass(frozen=True)
class _ParserAPI:
    parse_completion: Callable[[str], Any] | None = None
    parse_solution: Callable[[str], Any] | None = None


_PARSER_API: _ParserAPI | None = None


def _load_optional_parser_api() -> _ParserAPI:
    """Load a parser helper module if the project already has one."""

    candidate_modules = (
        "medgemma_visual_rft.parser",
        "medgemma_visual_rft.parsers",
        "medgemma_visual_rft.output_parser",
    )
    for module_name in candidate_modules:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue

        parse_completion = None
        parse_solution = None
        for attr in (
            "parse_single_box_completion",
            "parse_completion",
            "parse_output",
            "parse_answer",
        ):
            candidate = getattr(module, attr, None)
            if callable(candidate):
                parse_completion = candidate
                break
        for attr in (
            "parse_single_box_solution",
            "parse_solution",
            "parse_reference",
            "parse_answer",
        ):
            candidate = getattr(module, attr, None)
            if callable(candidate):
                parse_solution = candidate
                break
        return _ParserAPI(parse_completion=parse_completion, parse_solution=parse_solution)

    return _ParserAPI()


def _get_parser_api() -> _ParserAPI:
    global _PARSER_API
    if _PARSER_API is None:
        _PARSER_API = _load_optional_parser_api()
    return _PARSER_API


def _flatten_text(value: Any) -> str:
    """Best-effort conversion from TRL chat-style objects into plain text."""

    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, Mapping):
        if "content" in value:
            return _flatten_text(value["content"])
        if "text" in value:
            return _flatten_text(value["text"])
        if "value" in value:
            return _flatten_text(value["value"])
        return json.dumps(value, ensure_ascii=True)
    if isinstance(value, (list, tuple)):
        parts: list[str] = []
        for item in value:
            if isinstance(item, Mapping):
                if item.get("type") == "image":
                    continue
                if "text" in item:
                    parts.append(_flatten_text(item["text"]))
                    continue
                if "content" in item:
                    parts.append(_flatten_text(item["content"]))
                    continue
                if "value" in item:
                    parts.append(_flatten_text(item["value"]))
                    continue
            parts.append(_flatten_text(item))
        return "".join(parts)
    return str(value)


def _extract_completion_text(completion: Any) -> str:
    """Return the assistant text from a TRL completion object or raw string."""

    if isinstance(completion, str):
        return completion
    if isinstance(completion, Mapping):
        if "content" in completion:
            return _flatten_text(completion["content"])
        if "text" in completion:
            return _flatten_text(completion["text"])
    if isinstance(completion, (list, tuple)):
        if completion and isinstance(completion[0], Mapping) and "content" in completion[0]:
            return _flatten_text(completion[0]["content"])
        return _flatten_text(completion)
    return _flatten_text(completion)


def _extract_reference_text(solution: Any) -> str:
    """Return a canonical solution string from text or structured data."""

    return _flatten_text(solution)


def _match_shell(text: str) -> re.Match[str] | None:
    return _SHELL_RE.fullmatch(text)


def _coerce_int(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("box coordinates must be integers")
    return value


def _coerce_box(box_value: Any) -> tuple[int, int, int, int]:
    if not isinstance(box_value, (list, tuple)) or len(box_value) != 4:
        raise ValueError("box_2d must be a length-4 list")
    y0 = _coerce_int(box_value[0])
    x0 = _coerce_int(box_value[1])
    y1 = _coerce_int(box_value[2])
    x1 = _coerce_int(box_value[3])
    if not (0 <= y0 <= 1000 and 0 <= x0 <= 1000 and 0 <= y1 <= 1000 and 0 <= x1 <= 1000):
        raise ValueError("box_2d coordinates must be in the 0..1000 range")
    if y1 <= y0 or x1 <= x0:
        raise ValueError("box_2d coordinates must satisfy y0 < y1 and x0 < x1")
    return (y0, x0, y1, x1)


def _normalize_label(label: Any) -> str:
    if not isinstance(label, str):
        raise ValueError("label must be a string")
    normalized = " ".join(label.split()).strip()
    if not normalized:
        raise ValueError("label must be non-empty")
    return normalized


def _parse_payload_text(text: Any, *, role: str) -> ParsedSingleBox:
    """Parse a MedGemma completion or solution into the canonical record."""

    if role not in {"completion", "solution"}:
        raise ValueError(f"unknown parse role: {role}")

    parser_api = _get_parser_api()
    if role == "completion" and parser_api.parse_completion is not None:
        parsed = parser_api.parse_completion(_extract_completion_text(text))
        return _coerce_parsed_record(parsed)
    if role == "solution" and parser_api.parse_solution is not None:
        parsed = parser_api.parse_solution(_extract_reference_text(text))
        return _coerce_parsed_record(parsed)

    raw_text = _extract_completion_text(text) if role == "completion" else _extract_reference_text(text)
    match = _match_shell(raw_text)
    if match is None:
        raise ValueError("completion does not match the required shell")

    think = match.group("think").strip()
    answer_text = match.group("answer").strip()
    payload = json.loads(answer_text)
    if not isinstance(payload, list) or len(payload) != 1:
        raise ValueError("answer must be a JSON list with exactly one object")

    item = payload[0]
    if not isinstance(item, Mapping):
        raise ValueError("answer item must be a JSON object")

    if set(item.keys()) != _JSON_OBJECT_KEYS:
        raise ValueError("answer object must contain exactly box_2d and label")

    box_2d = _coerce_box(item["box_2d"])
    label = _normalize_label(item["label"])
    return ParsedSingleBox(box_2d=box_2d, label=label, think=think, raw_answer=answer_text)


def _coerce_parsed_record(parsed: Any) -> ParsedSingleBox:
    """Normalize parser-module outputs into the local record shape."""

    if isinstance(parsed, ParsedSingleBox):
        return parsed
    if hasattr(parsed, "items"):
        items = list(getattr(parsed, "items"))
        if len(items) != 1:
            raise ValueError("parsed record must contain exactly one item")
        item = items[0]
        if hasattr(item, "box_2d") and hasattr(item, "label"):
            return ParsedSingleBox(
                box_2d=_coerce_box(getattr(item, "box_2d")),
                label=_normalize_label(getattr(item, "label")),
                think=_flatten_text(getattr(parsed, "think", "")),
                raw_answer=_flatten_text(getattr(parsed, "raw_text", getattr(parsed, "normalized_text", ""))),
            )
    if isinstance(parsed, Mapping):
        box_value = parsed.get("box_2d", parsed.get("box"))
        label_value = parsed.get("label")
        if box_value is None or label_value is None:
            raise ValueError("parsed record is missing box_2d or label")
        think_value = _flatten_text(parsed.get("think", ""))
        raw_answer = _flatten_text(parsed.get("raw_answer", parsed.get("answer", "")))
        return ParsedSingleBox(
            box_2d=_coerce_box(box_value),
            label=_normalize_label(label_value),
            think=think_value,
            raw_answer=raw_answer,
        )
    if hasattr(parsed, "box_2d") and hasattr(parsed, "label"):
        return ParsedSingleBox(
            box_2d=_coerce_box(getattr(parsed, "box_2d")),
            label=_normalize_label(getattr(parsed, "label")),
            think=_flatten_text(getattr(parsed, "think", "")),
            raw_answer=_flatten_text(getattr(parsed, "raw_answer", getattr(parsed, "answer", ""))),
        )
    raise TypeError(f"unsupported parser output type: {type(parsed)!r}")


def _extract_batch_field(
    batch: Sequence[Any] | Iterable[Any],
    *,
    field_name: str,
) -> list[Any]:
    values = list(batch)
    extracted: list[Any] = []
    for item in values:
        if isinstance(item, Mapping) and field_name in item:
            extracted.append(item[field_name])
        else:
            extracted.append(item)
    return extracted


def format_reward(completions: Sequence[Any], **kwargs: Any) -> list[float]:
    """Return 1.0 when the completion obeys the exact <think>/<answer> shell."""

    rewards: list[float] = []
    for completion in completions:
        text = _extract_completion_text(completion)
        rewards.append(1.0 if _match_shell(text) is not None else 0.0)
    return rewards


def parse_gate_reward(completions: Sequence[Any], solution: Sequence[Any], **kwargs: Any) -> list[float]:
    """Return 1.0 only when both completion and reference parse cleanly."""

    rewards: list[float] = []
    for completion, ref in zip(completions, solution):
        try:
            _parse_payload_text(completion, role="completion")
            _parse_payload_text(ref, role="solution")
            rewards.append(1.0)
        except Exception:
            rewards.append(0.0)
    return rewards


def _validate_box_pair(
    pred_box: Sequence[int],
    gt_box: Sequence[int],
) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float]]:
    pred = _coerce_box(pred_box)
    gt = _coerce_box(gt_box)
    return tuple(float(v) for v in pred), tuple(float(v) for v in gt)


def normalized_iou(pred_box: Sequence[int], gt_box: Sequence[int]) -> float:
    """Compute IoU for normalized `[y0, x0, y1, x1]` boxes in the 0..1000 range."""

    try:
        (py0, px0, py1, px1), (gy0, gx0, gy1, gx1) = _validate_box_pair(pred_box, gt_box)
    except Exception:
        return 0.0

    inter_y0 = max(py0, gy0)
    inter_x0 = max(px0, gx0)
    inter_y1 = min(py1, gy1)
    inter_x1 = min(px1, gx1)

    inter_h = max(0.0, inter_y1 - inter_y0)
    inter_w = max(0.0, inter_x1 - inter_x0)
    inter_area = inter_h * inter_w

    pred_area = max(0.0, py1 - py0) * max(0.0, px1 - px0)
    gt_area = max(0.0, gy1 - gy0) * max(0.0, gx1 - gx0)
    union_area = pred_area + gt_area - inter_area
    if union_area <= 0.0:
        return 0.0
    return float(inter_area / union_area)


def normalized_giou(pred_box: Sequence[int], gt_box: Sequence[int]) -> float:
    """Compute a [0, 1] normalized GIoU for normalized `[y0, x0, y1, x1]` boxes."""

    try:
        (py0, px0, py1, px1), (gy0, gx0, gy1, gx1) = _validate_box_pair(pred_box, gt_box)
    except Exception:
        return 0.0

    inter_y0 = max(py0, gy0)
    inter_x0 = max(px0, gx0)
    inter_y1 = min(py1, gy1)
    inter_x1 = min(px1, gx1)

    inter_h = max(0.0, inter_y1 - inter_y0)
    inter_w = max(0.0, inter_x1 - inter_x0)
    inter_area = inter_h * inter_w

    pred_area = max(0.0, py1 - py0) * max(0.0, px1 - px0)
    gt_area = max(0.0, gy1 - gy0) * max(0.0, gx1 - gx0)
    union_area = pred_area + gt_area - inter_area
    if union_area <= 0.0:
        return 0.0

    iou = inter_area / union_area

    enc_y0 = min(py0, gy0)
    enc_x0 = min(px0, gx0)
    enc_y1 = max(py1, gy1)
    enc_x1 = max(px1, gx1)
    enc_area = max(0.0, enc_y1 - enc_y0) * max(0.0, enc_x1 - enc_x0)
    if enc_area <= 0.0:
        return max(0.0, min(1.0, iou))

    giou = iou - ((enc_area - union_area) / enc_area)
    return max(0.0, min(1.0, (giou + 1.0) / 2.0))


def _geometry_reward(
    pred_box: Sequence[int],
    gt_box: Sequence[int],
    metric: str,
) -> float:
    metric_name = metric.lower().strip()
    if metric_name == "iou":
        return normalized_iou(pred_box, gt_box)
    if metric_name == "giou":
        return normalized_giou(pred_box, gt_box)
    raise ValueError(f"unsupported metric: {metric}")


def single_box_reward(
    completions: Sequence[Any],
    solution: Sequence[Any],
    **kwargs: Any,
) -> list[float]:
    """Deterministic single-box reward for the MedGemma pilot.

    Reward composition:
    - hard zero if the completion or reference fails the parse gate
    - small format bonus
    - small label exact-match bonus
    - main normalized geometry reward

    The default weights sum to 1.0 and can be overridden via kwargs:
    - format_weight
    - label_weight
    - geo_weight
    - metric (`iou` or `giou`)
    """

    format_weight = float(kwargs.get("format_weight", DEFAULT_FORMAT_WEIGHT))
    label_weight = float(kwargs.get("label_weight", DEFAULT_LABEL_WEIGHT))
    geo_weight = float(kwargs.get("geo_weight", DEFAULT_GEO_WEIGHT))
    metric = str(kwargs.get("metric", DEFAULT_METRIC))
    if not math.isclose(format_weight + label_weight + geo_weight, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("reward weights must sum to 1.0")

    rewards: list[float] = []
    for completion, ref in zip(completions, solution):
        try:
            pred = _parse_payload_text(completion, role="completion")
            gt = _parse_payload_text(ref, role="solution")
        except Exception:
            rewards.append(0.0)
            continue

        format_ok = 1.0 if _match_shell(_extract_completion_text(completion)) is not None else 0.0
        label_ok = 1.0 if _normalize_label(pred.label).casefold() == _normalize_label(gt.label).casefold() else 0.0
        geometry_ok = _geometry_reward(pred.box_2d, gt.box_2d, metric=metric)

        reward = (format_weight * format_ok) + (label_weight * label_ok) + (geo_weight * geometry_ok)
        rewards.append(float(max(0.0, min(1.0, reward))))

    return rewards


def single_box_iou_reward(
    completions: Sequence[Any],
    solution: Sequence[Any],
    **kwargs: Any,
) -> list[float]:
    """Wrapper that locks single-box reward to IoU."""

    return single_box_reward(completions, solution, metric="iou", **kwargs)


def single_box_giou_reward(
    completions: Sequence[Any],
    solution: Sequence[Any],
    **kwargs: Any,
) -> list[float]:
    """Wrapper that locks single-box reward to normalized GIoU."""

    return single_box_reward(completions, solution, metric="giou", **kwargs)


reward_funcs_registry: dict[str, Callable[..., list[float]]] = {
    "format": format_reward,
    "parse_gate": parse_gate_reward,
    "single_box_iou": single_box_iou_reward,
    "single_box_giou": single_box_giou_reward,
}
