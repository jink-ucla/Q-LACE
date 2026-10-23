#!/usr/bin/env python3
"""Score explicit local occurrence-preserving outputs with paired patient CIs.

No hidden historical path mapping is used. Restricted inputs remain local;
only aggregate statistics are written. This is not an original-run fingerprint
verifier, which would require the unavailable private run snapshots.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import cmig_baseline_compare as compare


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--arm", required=True, action="append", help="Unique alias=prediction.jsonl; repeated")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261020)
    args = parser.parse_args()
    rows = compare.HARNESS.load_jsonl(args.manifest)
    if not rows:
        parser.error("Empty manifest")
    rows = compare.HARNESS.read_manifest(args.manifest, len(rows))
    arms = {}
    for spec in args.arm:
        if "=" not in spec:
            parser.error("Use alias=path")
        alias, path = spec.split("=", 1)
        if not alias or alias in arms or not alias.replace("_", "").isalnum():
            parser.error("Arm aliases must be unique alphanumeric/underscore names")
        records = compare.HARNESS.load_jsonl(Path(path))
        lookup = compare.HARNESS.validate_predictions(rows, records)
        arms[alias] = [lookup[r["row_id"]] for r in rows]
    counts, _, _, _ = compare.patient_confusions(rows, arms)
    samples = compare.paired_bootstrap(counts, replicates=args.bootstrap_replicates,
                                      seed=args.bootstrap_seed)
    import numpy as np
    point = compare.metrics_from_counts(counts.sum(axis=0))
    result = {"rows": len(rows), "patient_count": len({r["patient_id"] for r in rows}),
              "scope": "Paired patient bootstrap conditional on supplied fixed runs; no new-seed uncertainty",
              "arms": {}, "contrasts": {}}
    names = list(arms)
    for i, alias in enumerate(names):
        result["arms"][alias] = {metric: {"estimate": float(point[i, k]),
            "ci_95": np.quantile(samples[:, i, k], [.025, .975]).tolist()}
            for k, metric in enumerate(compare.METRICS)}
        for j in range(i):
            result["contrasts"][f"{alias}_minus_{names[j]}"] = {metric: {
                "estimate": float(point[i, k] - point[j, k]),
                "ci_95": np.quantile(samples[:, i, k] - samples[:, j, k], [.025, .975]).tolist()}
                for k, metric in enumerate(compare.METRICS)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
