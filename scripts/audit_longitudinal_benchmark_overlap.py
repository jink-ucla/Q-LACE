#!/usr/bin/env python3
"""Audit overlap between local longitudinal data and external benchmark exports.

The tool compares patient, study, dicom, pair, and image-path identifiers and can
optionally include canonicalized report-text overlap. It emits a JSON summary and a
human-readable report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


DEFAULT_REPORT_TEXT_KEYS = (
    "report_text",
    "report",
    "findings",
    "impression",
    "comparison",
)
DEFAULT_REPORT_SECTION_KEYS = (
    "prior_report_sections",
    "current_report_sections",
    "report_sections",
)
IDENTIFIER_CATEGORIES = ("patient_id", "study_id", "dicom_id", "pair_id", "image_path", "image_basename")
IDENTIFIER_KEY_ALIASES = {
    "patient_id": ("patient_id", "subject_id"),
    "study_id": ("study_id", "prior_study_id", "current_study_id", "study_uid", "accession_number"),
    "dicom_id": ("dicom_id", "prior_dicom_id", "current_dicom_id", "sop_instance_uid"),
    "pair_id": ("pair_id", "source_id", "pair_uid", "example_id"),
    "image_path": ("image_path", "prior_image_path", "current_image_path", "path"),
}


def _normalize_identifier(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().replace("\\", "/")
    return normalized or None


def _normalize_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = re.sub(r"\s+", " ", value).strip().lower()
    return normalized or None


def _preview_text(value: str, limit: int = 120) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def _digest_text(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]


def _iter_input_files(paths: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for raw_path in paths:
        path = raw_path.expanduser()
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file() and child.suffix.lower() in {".json", ".jsonl", ".ndjson"}:
                    files.append(child)
        else:
            files.append(path)
    unique_files: list[Path] = []
    seen: set[Path] = set()
    for path in files:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_files.append(resolved)
    return unique_files


def _load_json_records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                payload = json.loads(stripped)
                if not isinstance(payload, dict):
                    raise ValueError(f"Expected object on line {line_number} of {path}")
                records.append(payload)
        return records

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list):
        if not all(isinstance(item, dict) for item in payload):
            raise ValueError(f"Expected a list of objects in {path}")
        return list(payload)
    if isinstance(payload, dict):
        for key in ("records", "rows", "items", "data", "examples", "samples"):
            value = payload.get(key)
            if isinstance(value, list) and all(isinstance(item, dict) for item in value):
                return list(value)
        if any(key in payload for key in ("pair_id", "patient_id", "study_id", "dicom_id", "image_path")):
            return [payload]
    raise ValueError(f"Unsupported JSON structure in {path}")


def _load_records_from_paths(paths: Iterable[Path]) -> tuple[list[dict[str, Any]], list[str]]:
    files = _iter_input_files(paths)
    records: list[dict[str, Any]] = []
    file_labels: list[str] = []
    for file_path in files:
        file_records = _load_json_records(file_path)
        records.extend(file_records)
        file_labels.append(str(file_path))
    return records, file_labels


def _collect_strings(value: Any) -> list[str]:
    collected: list[str] = []
    if isinstance(value, str):
        normalized = _normalize_text(value)
        if normalized:
            collected.append(normalized)
    elif isinstance(value, dict):
        for item in value.values():
            collected.extend(_collect_strings(item))
    elif isinstance(value, list):
        for item in value:
            collected.extend(_collect_strings(item))
    return collected


def _extract_identifier_values(record: dict[str, Any]) -> dict[str, set[str]]:
    values: dict[str, set[str]] = {category: set() for category in IDENTIFIER_CATEGORIES}

    for category, aliases in IDENTIFIER_KEY_ALIASES.items():
        for key in aliases:
            normalized = _normalize_identifier(record.get(key))
            if normalized:
                values[category].add(normalized)

    for key in ("prior_image_path", "current_image_path", "image_path"):
        normalized = _normalize_identifier(record.get(key))
        if normalized:
            values["image_path"].add(normalized)
            values["image_basename"].add(Path(normalized).name)

    if "path" in record and isinstance(record["path"], str):
        normalized = _normalize_identifier(record["path"])
        if normalized:
            values["image_path"].add(normalized)
            values["image_basename"].add(Path(normalized).name)

    return values


def _extract_report_text_values(
    record: dict[str, Any],
    *,
    report_text_keys: Iterable[str],
    report_section_keys: Iterable[str],
) -> set[str]:
    texts: set[str] = set()
    for key in report_text_keys:
        normalized = _normalize_text(record.get(key))
        if normalized:
            texts.add(normalized)
    for key in report_section_keys:
        texts.update(_collect_strings(record.get(key)))
    return texts


def _build_index(records: list[dict[str, Any]], *, include_report_text: bool, report_text_keys: tuple[str, ...], report_section_keys: tuple[str, ...]) -> dict[str, Any]:
    record_values: dict[str, list[set[str]]] = {}
    unique_values: dict[str, set[str]] = {category: set() for category in IDENTIFIER_CATEGORIES}
    unique_values["report_text"] = set()

    for category in IDENTIFIER_CATEGORIES:
        record_values[category] = []

    if include_report_text:
        record_values["report_text"] = []

    for record in records:
        identifier_values = _extract_identifier_values(record)
        for category, values in identifier_values.items():
            record_values[category].append(values)
            unique_values[category].update(values)
        if include_report_text:
            report_values = _extract_report_text_values(
                record,
                report_text_keys=report_text_keys,
                report_section_keys=report_section_keys,
            )
            record_values["report_text"].append(report_values)
            unique_values["report_text"].update(report_values)

    return {
        "record_values": record_values,
        "unique_values": unique_values,
    }


def _summarize_category(
    local_record_values: list[set[str]],
    benchmark_record_values: list[set[str]],
    local_unique: set[str],
    benchmark_unique: set[str],
    *,
    sample_limit: int,
) -> dict[str, Any]:
    shared = sorted(local_unique & benchmark_unique)
    local_rows = sum(1 for values in local_record_values if values & benchmark_unique)
    benchmark_rows = sum(1 for values in benchmark_record_values if values & local_unique)
    return {
        "local_unique_values": len(local_unique),
        "benchmark_unique_values": len(benchmark_unique),
        "shared_unique_values": len(shared),
        "local_rows_with_shared_values": local_rows,
        "benchmark_rows_with_shared_values": benchmark_rows,
        "shared_samples": shared[:sample_limit],
    }


def _summarize_report_text(
    local_record_values: list[set[str]],
    benchmark_record_values: list[set[str]],
    local_unique: set[str],
    benchmark_unique: set[str],
    *,
    sample_limit: int,
) -> dict[str, Any]:
    shared = sorted(local_unique & benchmark_unique)
    local_rows = sum(1 for values in local_record_values if values & benchmark_unique)
    benchmark_rows = sum(1 for values in benchmark_record_values if values & local_unique)
    return {
        "enabled": True,
        "local_unique_texts": len(local_unique),
        "benchmark_unique_texts": len(benchmark_unique),
        "shared_unique_texts": len(shared),
        "local_rows_with_shared_text": local_rows,
        "benchmark_rows_with_shared_text": benchmark_rows,
        "shared_samples": [
            {
                "sha1_12": _digest_text(text),
                "preview": _preview_text(text),
            }
            for text in shared[:sample_limit]
        ],
    }


def audit_overlap(
    local_paths: Iterable[Path],
    benchmark_paths: Iterable[Path],
    *,
    include_report_text: bool = False,
    report_text_keys: Iterable[str] | None = None,
    report_section_keys: Iterable[str] | None = None,
    sample_limit: int = 5,
) -> dict[str, Any]:
    local_records, local_files = _load_records_from_paths(local_paths)
    benchmark_records, benchmark_files = _load_records_from_paths(benchmark_paths)
    report_text_key_tuple = tuple(report_text_keys or DEFAULT_REPORT_TEXT_KEYS)
    report_section_key_tuple = tuple(report_section_keys or DEFAULT_REPORT_SECTION_KEYS)

    local_index = _build_index(
        local_records,
        include_report_text=include_report_text,
        report_text_keys=report_text_key_tuple,
        report_section_keys=report_section_key_tuple,
    )
    benchmark_index = _build_index(
        benchmark_records,
        include_report_text=include_report_text,
        report_text_keys=report_text_key_tuple,
        report_section_keys=report_section_key_tuple,
    )

    categories = {}
    for category in IDENTIFIER_CATEGORIES:
        categories[category] = _summarize_category(
            local_index["record_values"][category],
            benchmark_index["record_values"][category],
            local_index["unique_values"][category],
            benchmark_index["unique_values"][category],
            sample_limit=sample_limit,
        )

    any_local = set().union(*(local_index["unique_values"][category] for category in IDENTIFIER_CATEGORIES))
    any_benchmark = set().union(*(benchmark_index["unique_values"][category] for category in IDENTIFIER_CATEGORIES))
    any_shared = sorted(any_local & any_benchmark)
    local_rows_any = sum(1 for row_values in zip(*local_index["record_values"].values()) if set().union(*row_values) & any_benchmark)
    benchmark_rows_any = sum(1 for row_values in zip(*benchmark_index["record_values"].values()) if set().union(*row_values) & any_local)

    summary: dict[str, Any] = {
        "local_inputs": local_files,
        "benchmark_inputs": benchmark_files,
        "records": {
            "local": len(local_records),
            "benchmark": len(benchmark_records),
        },
        "overlap": {
            "any_identifier": {
                "local_unique_values": len(any_local),
                "benchmark_unique_values": len(any_benchmark),
                "shared_unique_values": len(any_shared),
                "local_rows_with_shared_values": local_rows_any,
                "benchmark_rows_with_shared_values": benchmark_rows_any,
                "shared_samples": any_shared[:sample_limit],
            },
            "categories": categories,
        },
    }

    if include_report_text:
        summary["overlap"]["report_text"] = _summarize_report_text(
            local_index["record_values"]["report_text"],
            benchmark_index["record_values"]["report_text"],
            local_index["unique_values"]["report_text"],
            benchmark_index["unique_values"]["report_text"],
            sample_limit=sample_limit,
        )
    else:
        summary["overlap"]["report_text"] = {"enabled": False}

    return summary


def render_report(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("Longitudinal Benchmark Overlap Audit")
    lines.append(f"Local inputs: {len(summary['local_inputs'])}")
    lines.append(f"Benchmark inputs: {len(summary['benchmark_inputs'])}")
    lines.append(
        f"Records: local={summary['records']['local']} benchmark={summary['records']['benchmark']}"
    )
    lines.append("")
    lines.append("Identifier overlap")
    any_overlap = summary["overlap"]["any_identifier"]
    lines.append(
        "  any_identifier: shared={shared} local_rows={local_rows}/{local_total} benchmark_rows={benchmark_rows}/{benchmark_total}".format(
            shared=any_overlap["shared_unique_values"],
            local_rows=any_overlap["local_rows_with_shared_values"],
            local_total=summary["records"]["local"],
            benchmark_rows=any_overlap["benchmark_rows_with_shared_values"],
            benchmark_total=summary["records"]["benchmark"],
        )
    )
    if any_overlap["shared_samples"]:
        lines.append(f"    samples: {', '.join(any_overlap['shared_samples'])}")
    for category in IDENTIFIER_CATEGORIES:
        item = summary["overlap"]["categories"][category]
        lines.append(
            "  {category}: shared={shared} local_rows={local_rows}/{local_total} benchmark_rows={benchmark_rows}/{benchmark_total}".format(
                category=category,
                shared=item["shared_unique_values"],
                local_rows=item["local_rows_with_shared_values"],
                local_total=summary["records"]["local"],
                benchmark_rows=item["benchmark_rows_with_shared_values"],
                benchmark_total=summary["records"]["benchmark"],
            )
        )
        if item["shared_samples"]:
            lines.append(f"    samples: {', '.join(item['shared_samples'])}")

    report_text = summary["overlap"]["report_text"]
    lines.append("")
    lines.append("Report-text overlap")
    if not report_text.get("enabled"):
        lines.append("  disabled")
    else:
        lines.append(
            "  shared={shared} local_rows={local_rows}/{local_total} benchmark_rows={benchmark_rows}/{benchmark_total}".format(
                shared=report_text["shared_unique_texts"],
                local_rows=report_text["local_rows_with_shared_text"],
                local_total=summary["records"]["local"],
                benchmark_rows=report_text["benchmark_rows_with_shared_text"],
                benchmark_total=summary["records"]["benchmark"],
            )
        )
        for item in report_text["shared_samples"]:
            lines.append(f"    {item['sha1_12']}: {item['preview']}")

    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local",
        dest="local_inputs",
        action="append",
        type=Path,
        required=True,
        help="Local manifest or dataset file, or directory containing JSON/JSONL exports.",
    )
    parser.add_argument(
        "--benchmark",
        dest="benchmark_inputs",
        action="append",
        type=Path,
        required=True,
        help="External benchmark export file, or directory containing JSON/JSONL exports.",
    )
    parser.add_argument(
        "--include-report-text",
        action="store_true",
        help="Also compute overlap on canonicalized report text and report sections.",
    )
    parser.add_argument(
        "--report-text-key",
        dest="report_text_keys",
        action="append",
        default=None,
        help="Top-level string field to include in report-text overlap. May be repeated.",
    )
    parser.add_argument(
        "--report-section-key",
        dest="report_section_keys",
        action="append",
        default=None,
        help="Top-level dict field to include in report-text overlap. May be repeated.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=5,
        help="Maximum number of shared identifier samples to retain per category.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional path to write the JSON summary.",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=None,
        help="Optional path to write the human-readable report.",
    )
    return parser.parse_args(argv)


def _write_text(destination: Path | None, text: str, *, stream: Any) -> None:
    if destination is None:
        stream.write(text)
        if not text.endswith("\n"):
            stream.write("\n")
        stream.flush()
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text + ("\n" if not text.endswith("\n") else ""), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = audit_overlap(
        args.local_inputs,
        args.benchmark_inputs,
        include_report_text=args.include_report_text,
        report_text_keys=args.report_text_keys,
        report_section_keys=args.report_section_keys,
        sample_limit=args.sample_limit,
    )
    report = render_report(summary)
    _write_text(args.report_output, report, stream=sys.stderr)
    json_text = json.dumps(summary, indent=2, sort_keys=True)
    _write_text(args.json_output, json_text, stream=sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
