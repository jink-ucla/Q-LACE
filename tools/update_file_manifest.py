#!/usr/bin/env python3
"""Regenerate the public payload manifest after reviewing intended changes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.check_public_release import check_payload, payload_paths


def build(root: Path) -> dict:
    check_payload(root, require_manifest=False)
    source = root / "provenance/s1_file_manifest.json"
    originals = {entry["file"]: entry for entry in json.loads(source.read_text())["files"]}
    entries = []
    for name in payload_paths(root):
        if name == "FILE_MANIFEST.json":
            continue
        data = (root / name).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        entry = {"file": name, "bytes": len(data), "sha256": digest}
        if name in originals:
            entry.update({"s1_sha256": originals[name]["sha256"],
                          "source": originals[name]["source"],
                          "transformations": originals[name].get("transformations", [])})
            if digest != originals[name]["sha256"]:
                entry["transformations"] = [*entry["transformations"], "Public-repository adaptation; see PORTABILITY.md"]
        else:
            entry["source"] = "Q-LACE public-repository documentation, tooling or preserved provenance"
        entries.append(entry)
    return {"schema": "qlace-public-file-manifest-v1", "contents": "Code, protocols, synthetic tests and manuscript aggregate results", "files": entries}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    result = build(args.root)
    (args.root / "FILE_MANIFEST.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"Recorded {len(result['files'])} public payload files")


if __name__ == "__main__":
    main()
