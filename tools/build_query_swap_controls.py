#!/usr/bin/env python3
"""Build local directed query swaps from an explicitly supplied native manifest.

Adapted from Visual-RFT/scripts/build_query_swap_controls.py for Q-LACE:
removed historical paths and prediction reuse; retained every input occurrence
and required exact patient/prior/current identity. Apache-2.0; see LICENSE.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import cmig_medgemma_baseline as harness


def read_manifest(path: Path) -> list[dict]:
    rows = harness.load_jsonl(path)
    if not rows:
        raise ValueError("Manifest must contain at least one occurrence")
    rows = harness.read_manifest(path, expected_rows=len(rows))
    if any(not isinstance(r.get("patient_id"), str) or not r["patient_id"].strip() for r in rows):
        raise ValueError("Every occurrence requires a patient identity")
    return rows


def reindex(row: dict, index: int) -> dict:
    key = [row["prior_image_path"], row["current_image_path"], row["query_finding"]]
    return {**row, "row_index": index, "physical_key": key,
            "physical_key_sha256": harness.digest_json(key),
            "row_id": f"{index:06d}-{harness.digest_json(key)[:24]}"}


def build_swaps(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["patient_id"], row["prior_image_path"], row["current_image_path"])].append(row)
    controls = []
    for source in rows:
        key = (source["patient_id"], source["prior_image_path"], source["current_image_path"])
        for target in groups[key]:
            if source["query_finding"] == target["query_finding"]:
                continue
            controls.append(reindex({**target, "control_name": "query_swap",
                "query_swap_original_row_id": source["row_id"],
                "query_swap_target_row_id": target["row_id"],
                "query_swap_original_row_index": source["row_index"],
                "query_swap_target_row_index": target["row_index"]}, len(controls)))
    return controls


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="Restricted local JSONL manifest; never commit generated case records")
    args = parser.parse_args()
    if args.output.resolve() == args.manifest.resolve():
        parser.error("Output must not overwrite the source manifest")
    controls = build_swaps(read_manifest(args.manifest))
    if not controls:
        parser.error("No exact image pair has distinct finding queries")
    harness.write_jsonl(args.output, controls)
    print(f"Created {len(controls)} directed query-swap occurrences")


if __name__ == "__main__":
    main()
