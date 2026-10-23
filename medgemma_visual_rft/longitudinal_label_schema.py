"""Helpers for direct Stage-1 query-label longitudinal outputs."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Iterable


DEFAULT_STAGE1_LABELS = ("improving", "stable", "worsening", "no_supported_evidence")

_SPACE_RE = re.compile(r"\s+")
_LABEL_ALIASES = {
    "improving": "improving",
    "stable": "stable",
    "worsening": "worsening",
    "no_supported_evidence": "no_supported_evidence",
    "no supported evidence": "no_supported_evidence",
    "no-supported-evidence": "no_supported_evidence",
    "not_supported": "no_supported_evidence",
    "unsupported": "no_supported_evidence",
}


class Stage1LabelParseError(ValueError):
    """Raised when a Stage-1 label completion is invalid."""


@dataclass(frozen=True)
class Stage1LabelParseResult:
    label: str
    raw_text: str
    normalized_text: str


def canonicalize_stage1_label(
    value: str,
    *,
    allowed_labels: Iterable[str] = DEFAULT_STAGE1_LABELS,
) -> str:
    if not isinstance(value, str):
        raise Stage1LabelParseError("Stage-1 label must be a string")
    normalized = _SPACE_RE.sub(" ", value.strip())
    # Prompt-only comparators often obey the label-only instruction but add
    # terminal punctuation, e.g. "Worsening.". Treat that as the same label.
    normalized = normalized.strip("`\"'.,:;!?()[]{}")
    if not normalized:
        raise Stage1LabelParseError("Stage-1 label must not be empty")
    canonical = _LABEL_ALIASES.get(normalized.lower(), normalized.lower())
    allowed = tuple(allowed_labels)
    if canonical not in allowed:
        raise Stage1LabelParseError(
            f"Unsupported Stage-1 label {value!r}; expected one of {sorted(allowed)!r}"
        )
    return canonical


def serialize_stage1_label_solution(label: str) -> str:
    return canonicalize_stage1_label(label)


def parse_stage1_label_completion(
    text: str,
    *,
    allowed_labels: Iterable[str] = DEFAULT_STAGE1_LABELS,
) -> Stage1LabelParseResult:
    if not isinstance(text, str):
        raise Stage1LabelParseError("Completion must be a string")
    normalized_text = text.strip()
    if not normalized_text:
        raise Stage1LabelParseError("Completion must not be empty")
    label_text = normalized_text
    if normalized_text.startswith("{"):
        try:
            payload = json.loads(normalized_text)
        except json.JSONDecodeError as exc:
            raise Stage1LabelParseError(f"Completion is not valid JSON: {exc}") from exc
        if not isinstance(payload, dict) or set(payload.keys()) != {"answer"}:
            raise Stage1LabelParseError('JSON completion must be exactly {"answer": "..."}')
        label_text = payload["answer"]
    label = canonicalize_stage1_label(label_text, allowed_labels=allowed_labels)
    return Stage1LabelParseResult(
        label=label,
        raw_text=text,
        normalized_text=normalized_text,
    )


def build_stage1_label_problem(
    *,
    query_finding: str,
    query_anatomy: str | None = None,
    allowed_labels: Iterable[str] = DEFAULT_STAGE1_LABELS,
) -> str:
    finding = _SPACE_RE.sub(" ", query_finding.strip())
    if not finding:
        raise Stage1LabelParseError("query_finding must be non-empty for Stage-1 label prompts")
    anatomy = _SPACE_RE.sub(" ", (query_anatomy or "").strip())
    question_suffix = f" for {finding}"
    if anatomy and anatomy != "unspecified":
        question_suffix = f" in {anatomy}{question_suffix}"
    label_list = ", ".join(canonicalize_stage1_label(label, allowed_labels=allowed_labels) for label in allowed_labels)
    return (
        "Image 1 is the prior chest X-ray. Image 2 is the current chest X-ray. "
        "Answer only with one canonical label and nothing else. "
        f"Allowed labels: {label_list}. "
        f"Query: compared with the prior study, what is the interval change{question_suffix}?"
    )
