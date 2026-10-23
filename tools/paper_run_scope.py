"""Enforce the three random-seed runs reported in the manuscript."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


GROUPS = {
    "qlace_historical_three_runs": [f"historical_qlace_run{i}" for i in (1, 2, 3)],
    "finding_blind_historical_three_runs": [f"historical_generic_run{i}" for i in (1, 2, 3)],
    "qlace_native_three_runs": [f"qlace_run{i}_native" for i in (1, 2, 3)],
}
BASELINES = {"historical_biovilt", "historical_prompt", "base_native", "base_squarepad"}
CONTRASTS = {
    "bridge_adapters_native_vs_historical", "historical_adaptation_original_contract",
    "historical_full_vs_biovilt", "historical_full_vs_matched_finding_blind",
    "primary_adaptation_same_contract", "secondary_native_vs_historical_prompt",
    "secondary_squarepad_vs_native",
}


def validate(comparison: dict, reconstructed_runs: list[dict] | None = None, figure: dict | None = None) -> dict:
    allowed = BASELINES | {arm for group in GROUPS.values() for arm in group}
    if set(comparison["arms"]) != allowed:
        raise ValueError("Result arms differ from the manuscript's three-run scope")
    if set(comparison["aggregates"]) != set(GROUPS):
        raise ValueError("Aggregate groups differ from the manuscript's three-run scope")
    for name, members in GROUPS.items():
        group = comparison["aggregates"][name]
        if group["run_count"] != 3 or group["arms"] != members:
            raise ValueError(f"Expected three ordered runs in {name}")
    if set(comparison["contrasts"]) != CONTRASTS:
        raise ValueError("Contrasts differ from the manuscript's reported comparisons")
    for name, record in comparison["contrasts"].items():
        if not set(record["left_arms"] + record["right_arms"]) <= allowed:
            raise ValueError(f"Result membership differs in {name}")
    if reconstructed_runs is not None and [r["run"] for r in reconstructed_runs] != ["Run 1", "Run 2", "Run 3"]:
        raise ValueError("Training run membership must match Run 1--3")
    if figure is not None:
        mean = comparison["aggregates"]["qlace_historical_three_runs"]
        primary = comparison["arms"]["historical_qlace_run1"]
        for public, metric in (("accuracy", "accuracy"), ("macro_f1", "supported_macro_f1")):
            for plotted, expected in ((figure["primary"][public], primary["metrics"][metric]["estimate"]),
                                      (figure["three_random_seed_mean"][public], mean["metrics"][metric]["estimate"])):
                if not math.isclose(plotted, expected, rel_tol=0, abs_tol=1e-12):
                    raise ValueError("Figure annotation differs from the reported three-run results")
    return {
        "reported_random_seed_runs": 3, "run_names": ["Run 1", "Run 2", "Run 3"],
        "aggregate_membership": GROUPS, "analysis_arm_count": len(allowed),
        "interpretation": "Native-interface arms re-evaluate the same three adapters. Finding-blind controls use the corresponding three random seeds.",
        "published_single_run_variants": "Scale, budget and synthetic-abstention tables describe separate experimental conditions.",
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root)
    args = parser.parse_args()
    comparison = json.loads((args.root / "aggregates/baseline_comparison.json").read_text())
    data = json.loads((args.root / "aggregates/data_training_runtime.json").read_text())
    figure = json.loads((args.root / "aggregates/figure_A1_data.json").read_text())
    result = validate(comparison, data["reconstructed_training_runs"], figure)
    if json.loads((args.root / "PAPER_RUN_SCOPE.json").read_text()) != result:
        raise ValueError("Packaged run-scope record differs from the verified results")
    print(json.dumps({"status": "passed", **result}, indent=2))


if __name__ == "__main__":
    main()
