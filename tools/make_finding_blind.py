#!/usr/bin/env python3
"""Replace finding identity without changing row order, images, or targets."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path


def finding_blind(row: dict) -> dict:
    output = deepcopy(row)
    output["query_finding"] = "the unspecified finding"
    metadata = output.get("metadata")
    if isinstance(metadata, dict) and "query_finding" in metadata:
        metadata["query_finding"] = "the unspecified finding"
    # Label-mode training rebuilds its actual prompt from query_finding. Clear
    # stale natural-language fields rather than leaking the original finding.
    output["query"] = "Compared with the prior study, what is the interval change for the unspecified finding?"
    output["problem"] = output["query"]
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error("Use a separate output path")
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(finding_blind(row)) + "\n")
    print(f"Preserved {len(rows)} rows in their original order; output remains credential-restricted.")


if __name__ == "__main__":
    main()
