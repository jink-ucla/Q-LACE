#!/usr/bin/env python3
"""Verify the extracted software payload without network or restricted inputs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def verify(root: Path) -> int:
    manifest = json.loads((root / "FILE_MANIFEST.json").read_text())
    seen = set()
    for entry in manifest["files"]:
        relative = Path(entry["file"])
        if relative.is_absolute() or ".." in relative.parts or entry["file"] in seen:
            raise ValueError("Invalid or duplicate manifest path")
        seen.add(entry["file"])
        path = root / relative
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Missing, linked, or out-of-root manifest file")
        data = path.read_bytes()
        if len(data) != entry["bytes"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise ValueError(f"Payload integrity mismatch: {relative}")
    return len(seen)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(f"Verified {verify(args.root)} payload files against FILE_MANIFEST.json")


if __name__ == "__main__":
    main()
