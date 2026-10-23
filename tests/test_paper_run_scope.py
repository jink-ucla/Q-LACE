"""Check consistency of reported run membership."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from tools.paper_run_scope import validate


def payload():
    root = Path(__file__).resolve().parents[1]
    return json.loads((root / "aggregates/baseline_comparison.json").read_text())


def test_only_reported_three_run_groups_are_accepted():
    assert validate(payload())["reported_random_seed_runs"] == 3


def test_extra_result_arm_is_rejected():
    data = payload()
    data["arms"]["historical_qlace_run4"] = deepcopy(data["arms"]["historical_qlace_run1"])
    with pytest.raises(ValueError, match="three-run scope"):
        validate(data)


def test_extra_seed_in_mean_is_rejected():
    data = payload()
    data["aggregates"]["qlace_historical_three_runs"]["arms"].append("historical_qlace_run4")
    with pytest.raises(ValueError, match="Expected"):
        validate(data)


def test_training_membership_mismatch_is_rejected():
    rows = [{"run": f"Run {i}"} for i in (1, 2, 3, 4)]
    with pytest.raises(ValueError, match="Training run membership"):
        validate(payload(), rows)
