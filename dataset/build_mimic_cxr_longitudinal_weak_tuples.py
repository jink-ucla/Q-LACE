#!/usr/bin/env python3
"""Bootstrap conservative interval-change tuples from report text in a chronology manifest."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_MANIFEST = REPO_ROOT / "share_data" / "mimic_cxr_longitudinal_manifest_preview" / "manifest.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "share_data" / "mimic_cxr_longitudinal_weak_tuples_preview"
RULE_NAME = "report_heuristic_v1"

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_NEGATION_RE = re.compile(r"\b(no|without|free of|negative for|absence of)\b")

_WORSENING_RE = re.compile(
    r"\b(increased|increasing|worsen(?:ed|ing)?|worse|bigger|larger|more prominent|"
    r"progress(?:ed|ion)?|new)\b"
)
_IMPROVING_RE = re.compile(
    r"\b(decreased|decreasing|improved|improving|smaller|less prominent|resolved|"
    r"resolution|improvement|better)\b"
)
_STABLE_RE = re.compile(
    r"\b(no significant change|no interval change|stable|unchanged|not significantly changed|"
    r"without change|no acute cardiopulmonary abnormality|no acute intrathoracic process|"
    r"no acute cardiopulmonary process)\b"
)

FINDING_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("pleural effusion", re.compile(r"\bpleural effusions?\b")),
    ("pneumothorax", re.compile(r"\bpneumothorax\b")),
    ("consolidation", re.compile(r"\bconsolidation\b")),
    ("pneumonia", re.compile(r"\bpneumonia\b")),
    ("edema", re.compile(r"\bedema(?:tous)?\b|\bpulmonary edema\b")),
    ("atelectasis", re.compile(r"\batelecta(?:sis|tic)\b")),
    ("opacity", re.compile(r"\bopacit(?:y|ies)\b")),
    ("cardiomegaly", re.compile(r"\bcardiomegaly\b|\benlarged cardiac silhouette\b")),
    ("vascular congestion", re.compile(r"\bvascular congestion\b|\bcongestive change\b")),
    ("low lung volumes", re.compile(r"\blow lung volume[s]?\b")),
    ("pleural thickening", re.compile(r"\bpleural thickening\b")),
    ("rib fracture", re.compile(r"\brib fracture[s]?\b|\brib deformit(?:y|ies)\b")),
]

ANATOMY_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("left lower lobe", re.compile(r"\bleft lower lobe\b|\bleft bas(?:e|ilar)\b")),
    ("right lower lobe", re.compile(r"\bright lower lobe\b|\bright bas(?:e|ilar)\b")),
    ("left upper lobe", re.compile(r"\bleft upper lobe\b")),
    ("right upper lobe", re.compile(r"\bright upper lobe\b")),
    ("bibasilar", re.compile(r"\bbibasilar\b")),
    ("bilateral lungs", re.compile(r"\bbilateral\b")),
    ("left pleura", re.compile(r"\bleft pleural\b|\bleft pleura\b")),
    ("right pleura", re.compile(r"\bright pleural\b|\bright pleura\b")),
    ("cardiomediastinal silhouette", re.compile(r"\bcardiomediastinal silhouette\b")),
    ("heart", re.compile(r"\bheart\b|\bcardiac\b")),
    ("mediastinum", re.compile(r"\bmediastin(?:um|al)\b")),
    ("hilar regions", re.compile(r"\bhilar\b")),
    ("lungs", re.compile(r"\blungs?\b")),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, default=DEFAULT_INPUT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument(
        "--exclude-records",
        action="append",
        type=Path,
        default=None,
        help="Optional JSON/JSONL file containing benchmark or gold records whose patient IDs should be excluded.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def load_json_records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    payload = json.loads(stripped)
                    if not isinstance(payload, dict):
                        raise SystemExit(f"Expected object records in {path}")
                    records.append(payload)
        return records

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list) and all(isinstance(item, dict) for item in payload):
        return list(payload)
    if isinstance(payload, dict):
        for key in ("records", "rows", "items", "data", "examples", "samples"):
            value = payload.get(key)
            if isinstance(value, list) and all(isinstance(item, dict) for item in value):
                return list(value)
    raise SystemExit(f"Unsupported JSON structure in {path}")


def load_excluded_patient_ids(paths: list[Path] | None) -> set[str]:
    if not paths:
        return set()
    excluded: set[str] = set()
    for path in paths:
        for record in load_json_records(path):
            for key in ("patient_id", "subject_id"):
                value = record.get(key)
                if isinstance(value, str) and value.strip():
                    excluded.add(value.strip())
                    break
                if isinstance(value, int):
                    excluded.add(str(value))
                    break
    return excluded


def sentence_split(text: str) -> list[str]:
    cleaned = " ".join(text.split())
    if not cleaned:
        return []
    return [part.strip() for part in _SENTENCE_SPLIT_RE.split(cleaned) if part.strip()]


def infer_change(sentence: str) -> str | None:
    if _STABLE_RE.search(sentence):
        return "stable"
    if _IMPROVING_RE.search(sentence):
        return "improving"
    if _WORSENING_RE.search(sentence):
        return "worsening"
    return None


def infer_finding(sentence: str) -> str | None:
    for label, pattern in FINDING_PATTERNS:
        if pattern.search(sentence):
            return label
    return None


def infer_anatomy(sentence: str) -> str:
    for label, pattern in ANATOMY_PATTERNS:
        if pattern.search(sentence):
            return label
    return "chest"


def confidence_for(change: str, finding: str | None, anatomy: str, sentence: str) -> float:
    confidence = 0.35
    if change == "stable":
        confidence += 0.25
    if finding is not None:
        confidence += 0.20
    if anatomy != "chest":
        confidence += 0.10
    if len(sentence.split()) <= 25:
        confidence += 0.10
    return min(confidence, 1.0)


def build_evidence_text(row: dict[str, Any]) -> list[str]:
    sections = row.get("current_report_sections") or {}
    ordered_keys = ("comparison", "findings", "impression")
    evidence = [sections[key].strip() for key in ordered_keys if isinstance(sections.get(key), str) and sections[key].strip()]
    return evidence


def extract_weak_changes(row: dict[str, Any], *, min_confidence: float) -> dict[str, Any]:
    evidence = build_evidence_text(row)
    sentences = sentence_split(" ".join(evidence))
    stable_seen = False
    dedup: dict[tuple[str, str, str], dict[str, Any]] = {}

    for sentence in sentences:
        lowered = sentence.lower()
        change = infer_change(lowered)
        if change is None:
            continue
        if change != "stable" and _NEGATION_RE.search(lowered):
            continue

        finding = infer_finding(lowered)
        anatomy = infer_anatomy(lowered)
        confidence = confidence_for(change, finding, anatomy, lowered)

        if change == "stable" and finding is None:
            stable_seen = True
            continue
        if finding is None:
            continue
        if confidence < min_confidence:
            continue

        key = (anatomy, finding, change)
        candidate = {
            "anatomy": anatomy,
            "finding": finding,
            "change": change,
            "evidence": sentence.strip(),
            "confidence": round(confidence, 4),
            "source": RULE_NAME,
        }
        existing = dedup.get(key)
        if existing is None or candidate["confidence"] > existing["confidence"]:
            dedup[key] = candidate

    changes = [dedup[key] for key in sorted(dedup)]
    if changes:
        label_status = "weak_candidate"
    elif stable_seen:
        label_status = "weak_stable_empty"
    else:
        label_status = "weak_unlabeled"

    return {
        "changes": changes,
        "label_status": label_status,
        "evidence": evidence,
        "heuristic": {
            "rule": RULE_NAME,
            "confidence_floor": min_confidence,
        },
    }


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    label_status_counts = Counter(row["weak"]["label_status"] for row in rows)
    change_counts = Counter()
    finding_counts = Counter()
    anatomy_counts = Counter()
    total_tuples = 0

    for row in rows:
        for change in row["weak"]["changes"]:
            total_tuples += 1
            change_counts[change["change"]] += 1
            finding_counts[change["finding"]] += 1
            anatomy_counts[change["anatomy"]] += 1

    return {
        "count": len(rows),
        "label_status_counts": dict(label_status_counts),
        "total_candidate_tuples": total_tuples,
        "change_counts": dict(change_counts),
        "top_findings": finding_counts.most_common(10),
        "top_anatomies": anatomy_counts.most_common(10),
    }


def write_readme(path: Path, *, input_manifest: Path, summary: dict[str, Any], min_confidence: float) -> None:
    lines = [
        "# MIMIC-CXR Longitudinal Weak Tuples Preview",
        "",
        "This artifact is a conservative weak-supervision bridge from chronology manifests to structured interval-change tuples.",
        "",
        "Source:",
        f"- input manifest: `{input_manifest}`",
        "",
        "Heuristic policy:",
        "- use only current report `comparison`, `findings`, and `impression` sections",
        "- detect `improving`, `stable`, `worsening` with closed regex rules",
        "- map findings and coarse anatomy with closed regex buckets",
        "- drop non-stable sentences with strong negation cues to avoid false positives such as `no new focal consolidation`",
        "- emit `changes: []` only for explicit stable-empty cases",
        "",
        f"Confidence floor: `{min_confidence}`",
        "",
        "Summary:",
        f"- rows: `{summary['count']}`",
        f"- label status counts: `{summary['label_status_counts']}`",
        f"- total candidate tuples: `{summary['total_candidate_tuples']}`",
        f"- change counts: `{summary['change_counts']}`",
        f"- top findings: `{summary['top_findings']}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.input_manifest)
    excluded_patient_ids = load_excluded_patient_ids(args.exclude_records)
    if excluded_patient_ids:
        rows = [row for row in rows if str(row.get("patient_id") or "").strip() not in excluded_patient_ids]
    if args.max_rows is not None:
        rows = rows[: args.max_rows]

    output_rows: list[dict[str, Any]] = []
    for row in rows:
        enriched = dict(row)
        metadata = dict(enriched.get("metadata") or {})
        weak = extract_weak_changes(enriched, min_confidence=args.min_confidence)
        metadata["label_status"] = weak["label_status"]
        enriched["metadata"] = metadata
        enriched["weak"] = weak
        output_rows.append(enriched)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    weak_path = output_dir / "weak_tuples.jsonl"
    write_jsonl(weak_path, output_rows)
    summary = build_summary(output_rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_readme(
        output_dir / "README.md",
        input_manifest=args.input_manifest,
        summary=summary,
        min_confidence=args.min_confidence,
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "weak_tuples_path": str(weak_path),
                "summary": summary,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
