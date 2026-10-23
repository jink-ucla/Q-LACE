"""Synthetic release-boundary regressions; no private inputs."""
import json
from pathlib import Path
import subprocess

import pytest

from tools.check_public_release import check_aggregate, check_history, check_path, check_payload, check_text


def test_source_and_aggregate_paths_are_allowed():
    for name in ("LICENSE", "README.md", ".github/workflows/cpu-checks.yml", "aggregates/manuscript_tables/table_1.csv"):
        check_path(name)


@pytest.mark.parametrize("name", ["../outside.py", "/absolute.py", "models/weights.json", "source/image.jpg", ".env", "local_artifacts/manifest.json", "tests/__pycache__/compiled.pyc"])
def test_restricted_or_binary_payload_paths_are_rejected(name):
    with pytest.raises(ValueError):
        check_path(name)


def test_secret_scan_reports_category_without_echoing_content():
    token = ("h" + "f_") + "SYNTHETIC" * 5
    with pytest.raises(ValueError, match="access_token") as error:
        check_text(token.encode(), "synthetic.txt")
    assert token not in str(error.value)
    with pytest.raises(ValueError, match="Binary"):
        check_text(b"synthetic\0payload", "synthetic.txt")
    check_text(b"Use your own token through the standard authentication flow.", "synthetic.md")


def test_case_records_are_not_aggregate_statistics():
    check_aggregate({"patient_count": 12, "unit": "patient_id", "accuracy": 0.5}, "synthetic.json")
    with pytest.raises(ValueError, match="Case-record"):
        check_aggregate({"records": [{"patient_id": "synthetic-local"}]}, "synthetic.json")


def test_unknown_worktree_file_is_not_silently_added_to_manifest(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "README.md").write_text("Synthetic release\n")
    (tmp_path / "FILE_MANIFEST.json").write_text(json.dumps({"files": []}))
    with pytest.raises(ValueError, match="explicit file manifest"):
        check_payload(tmp_path)


def test_history_scan_includes_unreachable_git_blobs(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    assert check_history(tmp_path)["git_objects"] == 0
    token = ("g" + "hp_") + "SYNTHETIC" * 5
    subprocess.run(["git", "-C", str(tmp_path), "hash-object", "-w", "--stdin"], input=token.encode(), capture_output=True, check=True)
    with pytest.raises(ValueError, match="access_token"):
        check_history(tmp_path)
