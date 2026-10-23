"""Helpers for stopping longitudinal JSON generation as soon as it closes cleanly."""

from __future__ import annotations

from medgemma_visual_rft.longitudinal_schema import LongitudinalParseError, parse_longitudinal_completion


def extract_balanced_json_object_prefix(text: str) -> str | None:
    """Return the first balanced top-level JSON object prefix, if present."""

    if not isinstance(text, str):
        return None
    stripped = text.lstrip()
    if not stripped.startswith("{"):
        return None

    depth = 0
    in_string = False
    escaped = False
    for index, char in enumerate(stripped):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue
        if char == "{":
            depth += 1
            continue
        if char == "}":
            depth -= 1
            if depth < 0:
                return None
            if depth == 0:
                return stripped[: index + 1]

    return None


def has_valid_longitudinal_json_prefix(text: str) -> bool:
    """Return True once a strict longitudinal JSON object has fully closed."""

    prefix = extract_balanced_json_object_prefix(text)
    if prefix is None:
        return False
    try:
        parse_longitudinal_completion(prefix)
    except LongitudinalParseError:
        return False
    return True
