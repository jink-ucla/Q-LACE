#!/usr/bin/env python3
"""Reconstruct a chronology-builder input locally from official MIMIC metadata.

No research data are bundled. Outputs contain credential-restricted identifiers
and local paths and must remain on authorized storage.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

KEYS = ("subject_id", "study_id", "dicom_id")
FIELDS = (*KEYS, "Path", "ViewPosition", "split")


def identifier(value: str, prefix: str = "") -> str:
    value = value.strip()
    if prefix and value.startswith(prefix):
        value = value[len(prefix):]
    if not value or "/" in value or "\\" in value or value in (".", ".."):
        raise ValueError("Empty or unsafe identifier")
    return value


def key(row: dict[str, str]) -> tuple[str, str, str]:
    return (identifier(row["subject_id"], "p"),
            identifier(row["study_id"], "s"), identifier(row["dicom_id"]))


def build_master(metadata: Path, splits: Path, jpg_root: Path, output: Path,
                 *, check_images: bool = True) -> dict[str, int]:
    if output.resolve() in (metadata.resolve(), splits.resolve()):
        raise ValueError("Output must not overwrite a source CSV")
    split_index: dict[tuple[str, str, str], str] = {}
    with splits.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not set((*KEYS, "split")).issubset(reader.fieldnames or ()):
            raise ValueError("Split CSV lacks required official columns")
        for row in reader:
            row_key, value = key(row), row["split"].strip().lower()
            if value == "valid":
                value = "validate"
            if value not in ("train", "validate", "test"):
                raise ValueError("Unknown split value")
            if row_key in split_index:
                raise ValueError("Duplicate split key")
            split_index[row_key] = value
    if not split_index:
        raise ValueError("Empty official split population")
    rows, seen = [], set()
    with metadata.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not set((*KEYS, "ViewPosition")).issubset(reader.fieldnames or ()):
            raise ValueError("Metadata CSV lacks required official columns")
        for row in reader:
            row_key = key(row)
            if row_key in seen:
                raise ValueError("Duplicate metadata key")
            seen.add(row_key)
            if row_key not in split_index:
                raise ValueError("Metadata image lacks an official split")
            subject, study, dicom = row_key
            image = jpg_root.resolve() / f"p{subject[:2]}" / f"p{subject}" / f"s{study}" / f"{dicom}.jpg"
            if check_images and not image.is_file():
                raise ValueError("A derived image path does not exist")
            rows.append(dict(zip(KEYS, row_key), Path=str(image),
                             ViewPosition=row["ViewPosition"].strip().upper(),
                             split=split_index[row_key]))
    if set(split_index) != seen:
        raise ValueError("Official split and metadata image populations differ")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return {"metadata_rows": len(rows), "split_rows": len(split_index), "output_rows": len(rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-csv", required=True, type=Path)
    parser.add_argument("--split-csv", required=True, type=Path)
    parser.add_argument("--jpg-root", required=True, type=Path, help="Directory containing pXX image directories")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--skip-image-check", action="store_true", help="Metadata-only preparation; validate paths before use")
    args = parser.parse_args()
    print(build_master(args.metadata_csv, args.split_csv, args.jpg_root, args.output,
                       check_images=not args.skip_image_check))


if __name__ == "__main__":
    main()
