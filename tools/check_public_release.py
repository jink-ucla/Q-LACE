#!/usr/bin/env python3
"""Check the release payload, aggregate boundaries, and local Git objects."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.verify_manifest import verify

MAX_TEXT_BYTES = 2 * 1024 * 1024
TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".csv", ".tsv", ".yml", ".yaml", ".cff", ".toml"}
TEXT_NAMES = {"LICENSE", "NOTICE", ".gitignore", ".gitattributes"}
BLOCKED_DIRS = {"data", "share_data", "local_artifacts", "models", "checkpoints", "predictions", "outputs", "runs", "logs", "reproduced", "wandb"}
RECORD_KEYS = {"patient_id", "subject_id", "study_id", "dicom_id", "prior_study_id", "prior_dicom_id", "image_path", "prior_image_path", "current_image_path", "report_text", "raw_report", "generated_text", "case_id"}
PATTERNS = {
    "access_token": re.compile(r"(?:hf_[A-Za-z0-9]{25,}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|sk-[A-Za-z0-9_-]{32,})"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "cloud_access_key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "private_mount": re.compile(r"/(?:cvibraid|cvib2|Users)/|/" + r"home/[^\s/]+/"),
}
DATA_ID = re.compile(r"(?:\bp\d{8}/s\d{8}\b|\bp\d{8}_s\d{8}\b)")


def git(root: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.PIPE)


def payload_paths(root: Path) -> list[str]:
    """Git's candidate set includes tracked files even if subsequently ignored."""
    paths = git(root, "ls-files", "--cached", "--others", "--exclude-standard", "-z").decode().split("\0")
    return sorted(set(name for name in paths if name))


def check_path(name: str) -> None:
    p = Path(name)
    if p.is_absolute() or ".." in p.parts or not p.parts or p.parts[0] in BLOCKED_DIRS:
        raise ValueError(f"Non-release path: {name}")
    if p.name not in TEXT_NAMES and p.suffix.lower() not in TEXT_SUFFIXES:
        raise ValueError(f"Non-text release file: {name}")
    if p.name.startswith(".env") or ".git" in p.parts or "__pycache__" in p.parts:
        raise ValueError(f"Private/cache path: {name}")


def check_text(blob: bytes, where: str) -> None:
    if len(blob) > MAX_TEXT_BYTES or b"\0" in blob:
        raise ValueError(f"Binary or oversized release payload: {where}")
    try:
        text = blob.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Non-UTF-8 payload: {where}") from exc
    for label, pattern in PATTERNS.items():
        if pattern.search(text):
            # Report a location/category, never the matched credential/content.
            raise ValueError(f"Release scan detected {label}: {where}")


def check_aggregate(value, where: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in RECORD_KEYS:
                raise ValueError(f"Case-record field in aggregate: {where}")
            check_aggregate(item, where)
    elif isinstance(value, list):
        for item in value:
            check_aggregate(item, where)
    elif isinstance(value, str) and DATA_ID.search(value):
        raise ValueError(f"Identifier-shaped record in aggregate: {where}")


def check_payload(root: Path, *, require_manifest: bool = True) -> dict:
    names = payload_paths(root)
    if not names:
        raise ValueError("Empty release payload")
    for name in names:
        check_path(name)
        path = root / name
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"Linked or out-of-root file: {name}")
        data = path.read_bytes()
        check_text(data, name)
        if name.startswith("aggregates/"):
            if DATA_ID.search(data.decode("utf-8")):
                raise ValueError(f"Identifier-shaped record in aggregate: {name}")
            if path.suffix == ".json":
                check_aggregate(json.loads(data), name)
    if require_manifest:
        count = verify(root)
        manifest = json.loads((root / "FILE_MANIFEST.json").read_text())
        expected = {entry["file"] for entry in manifest["files"]} | {"FILE_MANIFEST.json"}
        if expected != set(names):
            raise ValueError("Current release files differ from the explicit file manifest")
    else:
        count = None
    return {"payload_files": len(names), "manifest_payloads": count}


def check_history(root: Path) -> dict:
    objects = git(root, "cat-file", "--batch-all-objects", "--batch-check=%(objectname) %(objecttype)").decode().splitlines()
    blobs = 0
    for line in objects:
        oid, kind = line.split()
        if kind in {"blob", "commit", "tag"}:
            check_text(git(root, "cat-file", kind, oid), f"Git {kind} {oid[:12]}")
            blobs += kind == "blob"
    commits = git(root, "rev-list", "--all").decode().splitlines()
    for commit in commits:
        for name in git(root, "ls-tree", "-r", "--name-only", "-z", commit).decode().split("\0"):
            if name:
                check_path(name)
                if name.startswith("aggregates/") and name.endswith(".json"):
                    data = git(root, "show", f"{commit}:{name}")
                    check_aggregate(json.loads(data), f"Git aggregate {commit[:12]}:{name}")
    return {"commits": len(commits), "git_objects": len(objects), "git_blobs": blobs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--history", action="store_true", help="Also scan all local Git blobs and reachable commit trees")
    args = parser.parse_args()
    result = {"status": "passed", **check_payload(args.root)}
    if args.history:
        result["history"] = check_history(args.root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
