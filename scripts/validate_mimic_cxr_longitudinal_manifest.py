#!/usr/bin/env python3
"""Validate a MIMIC-CXR prior/current chronology manifest and print a compact summary."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from medgemma_visual_rft.longitudinal_manifest import (  # noqa: E402
    CANONICAL_REPORT_SECTION_KEYS,
    load_manifest_jsonl,
    validate_manifest_record,
)


DEFAULT_MANIFEST = REPO_ROOT / "share_data" / "mimic_cxr_longitudinal_manifest_preview" / "manifest.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Path to the chronology manifest JSONL.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the number of rows to validate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_manifest_jsonl(args.manifest)
    if args.limit is not None:
        rows = rows[: args.limit]

    validated = [validate_manifest_record(row) for row in rows]
    split_counts = Counter(row["split"] for row in validated)
    view_pairs = Counter(
        f"{row['prior_view_position']}->{row['current_view_position']}" for row in validated
    )
    report_key_counts = Counter()
    canonical_report_presence = Counter()
    for row in validated:
        for key in row["prior_report_sections"]:
            report_key_counts[f"prior:{key}"] += 1
        for key in row["current_report_sections"]:
            report_key_counts[f"current:{key}"] += 1
        for key in CANONICAL_REPORT_SECTION_KEYS:
            if key in row["prior_report_sections"] and key in row["current_report_sections"]:
                canonical_report_presence[key] += 1

    summary = {
        "manifest": str(args.manifest),
        "validated_rows": len(validated),
        "patients": len({row["patient_id"] for row in validated}),
        "split_counts": dict(split_counts),
        "view_pairs": dict(view_pairs),
        "pairs_with_prior_and_current_canonical_sections": dict(canonical_report_presence),
        "top_report_section_keys": report_key_counts.most_common(12),
    }
    json.dump(summary, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
