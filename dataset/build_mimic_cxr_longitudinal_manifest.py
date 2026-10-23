# Portable supplementary copy: default paths/mount aliases sanitized; original hash is in FILE_MANIFEST.json.
#!/usr/bin/env python3
"""Build a leakage-safe prior/current CXR manifest from the local MIMIC-CXR JPG mirror.

This builder is intentionally conservative:
- it emits chronology/pairing manifests, not a fully labeled training set
- it keeps the single-image localization pilot isolated
- it prefers one representative frontal image per study
- it can optionally recover lightweight report sections from `mimic-cxr-reports.zip`

The resulting JSONL is meant to be the first executable artifact for the
longitudinal CXR Change-RFT track.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
from typing import Any


DEFAULT_MIMIC_ROOT = Path("./data/physionet.org/files/mimic-cxr-jpg/2.0.0")
DEFAULT_MASTER_CSV = DEFAULT_MIMIC_ROOT / "master_modified.csv"
DEFAULT_METADATA_CSV = DEFAULT_MIMIC_ROOT / "mimic-cxr-2.0.0-metadata.csv"
DEFAULT_REPORTS_ZIP = DEFAULT_MIMIC_ROOT / "mimic-cxr-reports.zip"
DEFAULT_OUTPUT_DIR = Path("share_data/mimic_cxr_longitudinal_manifest_preview")
DEFAULT_ALLOWED_VIEWS = ("PA", "AP")
DEFAULT_PROMPT = (
    "Given the prior and current chest X-rays from the same patient, "
    "compare the studies and describe interval changes using a structured JSON schema."
)

_SECTION_HEADER_RE = re.compile(r"^\s*([A-Z][A-Z /-]+):\s*(.*)$")


@dataclass(frozen=True)
class StudyImage:
    subject_id: str
    study_id: str
    dicom_id: str
    image_path: str
    view_position: str
    split: str
    findings: str
    impression: str
    study_date: str | None
    study_time: str | None
    study_datetime: datetime | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimic-root", type=Path, default=DEFAULT_MIMIC_ROOT)
    parser.add_argument("--master-csv", type=Path, default=DEFAULT_MASTER_CSV)
    parser.add_argument("--metadata-csv", type=Path, default=DEFAULT_METADATA_CSV)
    parser.add_argument("--reports-zip", type=Path, default=DEFAULT_REPORTS_ZIP)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--allowed-views", nargs="+", default=list(DEFAULT_ALLOWED_VIEWS))
    parser.add_argument("--max-pairs", type=int, default=128, help="Optional cap for preview manifests.")
    parser.add_argument(
        "--include-report-sections",
        action="store_true",
        help="Recover COMPARISON/FINDINGS/IMPRESSION text for each study from the reports zip.",
    )
    parser.add_argument(
        "--require-reports",
        action="store_true",
        help="Keep only pairs where both prior/current report files exist in the zip.",
    )
    parser.add_argument(
        "--min-days-between-studies",
        type=int,
        default=0,
        help="Optional minimum day gap between adjacent studies.",
    )
    parser.add_argument(
        "--baseline-problem",
        type=str,
        default=DEFAULT_PROMPT,
        help="A qualitative baseline prompt to store in the manifest for later use.",
    )
    parser.add_argument(
        "--exclude-records",
        action="append",
        type=Path,
        default=None,
        help="Optional JSON/JSONL file containing benchmark or gold records whose patient IDs should be excluded.",
    )
    return parser.parse_args()


def normalize_study_id(value: str | int) -> str:
    text = str(value).strip()
    return text[1:] if text.startswith("s") else text


def parse_datetime(study_date: str | None, study_time: str | None) -> datetime | None:
    if not study_date:
        return None
    time_text = (study_time or "").split(".")[0]
    if time_text:
        time_text = time_text.zfill(6)
    else:
        time_text = "000000"
    try:
        return datetime.strptime(f"{study_date}{time_text}", "%Y%m%d%H%M%S")
    except ValueError:
        return None


def load_study_times(metadata_csv: Path) -> dict[tuple[str, str], tuple[str | None, str | None, datetime | None]]:
    study_times: dict[tuple[str, str], tuple[str | None, str | None, datetime | None]] = {}
    with metadata_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            subject_id = str(row["subject_id"]).strip()
            study_id = normalize_study_id(row["study_id"])
            study_date = str(row.get("StudyDate") or "").strip() or None
            study_time = str(row.get("StudyTime") or "").strip() or None
            current = parse_datetime(study_date, study_time)
            key = (subject_id, study_id)
            existing = study_times.get(key)
            if existing is None:
                study_times[key] = (study_date, study_time, current)
                continue
            _, _, prev_dt = existing
            if prev_dt is None and current is not None:
                study_times[key] = (study_date, study_time, current)
            elif prev_dt is not None and current is not None and current < prev_dt:
                study_times[key] = (study_date, study_time, current)
    return study_times


def view_rank(view_position: str) -> tuple[int, str]:
    view = (view_position or "").strip().upper()
    if view == "PA":
        return (0, view)
    if view == "AP":
        return (1, view)
    return (9, view)


def normalize_split(split: str) -> str:
    value = split.strip().lower()
    if value == "valid":
        return "validate"
    return value


def load_json_records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    payload = json.loads(stripped)
                    if not isinstance(payload, dict):
                        raise SystemExit(f"Expected object records in {path}")
                    records.append(payload)
        return records

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, list) and all(isinstance(item, dict) for item in payload):
        return list(payload)
    if isinstance(payload, dict):
        for key in ("records", "rows", "items", "data", "examples", "samples"):
            value = payload.get(key)
            if isinstance(value, list) and all(isinstance(item, dict) for item in value):
                return list(value)
    raise SystemExit(f"Unsupported JSON structure in {path}")


def load_excluded_patient_ids(paths: list[Path] | None) -> set[str]:
    if not paths:
        return set()
    excluded: set[str] = set()
    for path in paths:
        for record in load_json_records(path):
            for key in ("patient_id", "subject_id"):
                value = record.get(key)
                if isinstance(value, str) and value.strip():
                    excluded.add(value.strip())
                    break
                if isinstance(value, int):
                    excluded.add(str(value))
                    break
    return excluded


def choose_best_frontal_per_study(
    master_csv: Path,
    study_times: dict[tuple[str, str], tuple[str | None, str | None, datetime | None]],
    *,
    allowed_views: set[str],
) -> dict[tuple[str, str], StudyImage]:
    best: dict[tuple[str, str], StudyImage] = {}
    with master_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            view = str(row.get("ViewPosition") or "").strip().upper()
            if view not in allowed_views:
                continue
            subject_id = str(row["subject_id"]).strip()
            study_id = normalize_study_id(row["study_id"])
            image_path = str(row["Path"]).strip()
            if not image_path:
                continue
            key = (subject_id, study_id)
            study_date, study_time, study_dt = study_times.get(key, (None, None, None))
            candidate = StudyImage(
                subject_id=subject_id,
                study_id=study_id,
                dicom_id=str(row["dicom_id"]).strip(),
                image_path=image_path,
                view_position=view,
                split=normalize_split(str(row.get("split") or "").strip()),
                findings=str(row.get("findings") or "").strip(),
                impression=str(row.get("impression") or "").strip(),
                study_date=study_date,
                study_time=study_time,
                study_datetime=study_dt,
            )
            existing = best.get(key)
            if existing is None:
                best[key] = candidate
                continue
            if (view_rank(candidate.view_position), candidate.dicom_id) < (
                view_rank(existing.view_position),
                existing.dicom_id,
            ):
                best[key] = candidate
    return best


def report_zip_path(subject_id: str, study_id: str) -> str:
    patient_dir = f"p{subject_id[:2]}"
    return f"files/{patient_dir}/p{subject_id}/s{normalize_study_id(study_id)}.txt"


def extract_report_sections(text: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {}
    current_key: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        match = _SECTION_HEADER_RE.match(line)
        if match:
            heading = match.group(1).strip().lower().replace(" ", "_")
            heading = heading.replace("comparisons", "comparison")
            current_key = heading
            sections.setdefault(current_key, [])
            tail = match.group(2).strip()
            if tail:
                sections[current_key].append(tail)
            continue
        if current_key is not None:
            stripped = line.strip()
            if stripped:
                sections[current_key].append(stripped)
    return {key: " ".join(value).strip() for key, value in sections.items() if value}


def maybe_load_report_sections(report_zip: zipfile.ZipFile | None, subject_id: str, study_id: str) -> tuple[str, dict[str, str]]:
    relpath = report_zip_path(subject_id, study_id)
    if report_zip is None:
        return relpath, {}
    try:
        raw = report_zip.read(relpath).decode("utf-8", errors="ignore")
    except KeyError:
        return relpath, {}
    return relpath, extract_report_sections(raw)


def build_pairs(
    studies: dict[tuple[str, str], StudyImage],
    *,
    report_zip: zipfile.ZipFile | None,
    baseline_problem: str,
    require_reports: bool,
    min_days_between_studies: int,
    max_pairs: int | None,
    excluded_patient_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    by_subject: dict[str, list[StudyImage]] = defaultdict(list)
    for item in studies.values():
        by_subject[item.subject_id].append(item)

    records: list[dict[str, Any]] = []
    for subject_id, items in by_subject.items():
        if excluded_patient_ids and subject_id in excluded_patient_ids:
            continue
        ordered = sorted(
            items,
            key=lambda item: (
                item.study_datetime or datetime.min,
                item.study_id,
                item.dicom_id,
            ),
        )
        for index in range(1, len(ordered)):
            prior = ordered[index - 1]
            current = ordered[index]
            if prior.split and current.split and prior.split != current.split:
                continue

            delta_days: int | None = None
            if prior.study_datetime is not None and current.study_datetime is not None:
                delta_days = (current.study_datetime.date() - prior.study_datetime.date()).days
                if delta_days < min_days_between_studies:
                    continue

            prior_report_relpath, prior_sections = maybe_load_report_sections(report_zip, prior.subject_id, prior.study_id)
            current_report_relpath, current_sections = maybe_load_report_sections(report_zip, current.subject_id, current.study_id)
            if require_reports and (not prior_sections or not current_sections):
                continue

            pair_id = f"{subject_id}:{prior.study_id}:{current.study_id}"
            split = current.split or prior.split or ""
            record: dict[str, Any] = {
                "pair_id": pair_id,
                "patient_id": subject_id,
                "source_id": pair_id,
                "split": split,
                "prior_study_id": prior.study_id,
                "current_study_id": current.study_id,
                "prior_dicom_id": prior.dicom_id,
                "current_dicom_id": current.dicom_id,
                "prior_image_path": prior.image_path,
                "current_image_path": current.image_path,
                "prior_view_position": prior.view_position,
                "current_view_position": current.view_position,
                "prior_study_date": prior.study_date,
                "current_study_date": current.study_date,
                "prior_study_time": prior.study_time,
                "current_study_time": current.study_time,
                "delta_days": delta_days,
                "problem": baseline_problem,
                "prior_report_path": prior_report_relpath,
                "current_report_path": current_report_relpath,
                "prior_report_sections": prior_sections,
                "current_report_sections": current_sections,
                "metadata": {
                    "builder": "build_mimic_cxr_longitudinal_manifest.py",
                    "task_family": "longitudinal_cxr_change_rft",
                    "label_status": "unlabeled_manifest",
                },
            }
            records.append(record)
            if max_pairs is not None and len(records) >= max_pairs:
                return records
    return records


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    split_counts = Counter(str(row.get("split") or "") for row in rows)
    view_pairs = Counter(
        f"{row.get('prior_view_position','')}->{row.get('current_view_position','')}" for row in rows
    )
    delta_days = [
        int(row["delta_days"])
        for row in rows
        if row.get("delta_days") is not None and isinstance(row.get("delta_days"), int)
    ]
    report_pair_count = sum(
        1
        for row in rows
        if row.get("prior_report_sections") and row.get("current_report_sections")
    )
    patients = {str(row["patient_id"]) for row in rows}
    summary: dict[str, Any] = {
        "count": len(rows),
        "patient_count": len(patients),
        "split_counts": dict(split_counts),
        "view_pairs": dict(view_pairs),
        "with_both_report_sections": report_pair_count,
    }
    if delta_days:
        summary["delta_days"] = {
            "min": min(delta_days),
            "median": statistics.median(delta_days),
            "max": max(delta_days),
            "mean": statistics.fmean(delta_days),
        }
    return summary


def write_readme(path: Path, args: argparse.Namespace, summary: dict[str, Any]) -> None:
    lines = [
        "# MIMIC-CXR Longitudinal Manifest Preview",
        "",
        "This artifact is a chronology manifest, not a labeled training dataset.",
        "",
        "Purpose:",
        "- reconstruct prior/current frontal CXR pairs from the local MIMIC-CXR JPG mirror",
        "- preserve patient-level chronology and split provenance",
        "- optionally recover lightweight report sections for later weak supervision or audit",
        "",
        "Source files:",
        f"- `master_csv`: `{args.master_csv}`",
        f"- `metadata_csv`: `{args.metadata_csv}`",
        f"- `reports_zip`: `{args.reports_zip}`",
        "",
        "Builder policy:",
        f"- allowed views: `{', '.join(args.allowed_views)}`",
        "- one representative frontal image per study",
        "- pair adjacent studies within the same patient",
        "- split is inherited from the chosen MIMIC metadata rows and normalized so `valid -> validate`",
        "",
        "Manifest fields:",
        "- `pair_id`, `patient_id`, `source_id`, `split`",
        "- `prior_*` and `current_*` study/image identifiers",
        "- `delta_days`",
        "- `problem` baseline prompt",
        "- optional `prior_report_sections` / `current_report_sections` if report extraction is enabled",
        "- `metadata.label_status = unlabeled_manifest`",
        "",
        "Summary:",
        f"- pair count: `{summary.get('count', 0)}`",
        f"- patient count: `{summary.get('patient_count', 0)}`",
        f"- split counts: `{summary.get('split_counts', {})}`",
        f"- pairs with both report sections: `{summary.get('with_both_report_sections', 0)}`",
    ]
    if "delta_days" in summary:
        lines.extend(
            [
                "- delta_days stats:",
                f"  - min: `{summary['delta_days']['min']}`",
                f"  - median: `{summary['delta_days']['median']}`",
                f"  - mean: `{summary['delta_days']['mean']}`",
                f"  - max: `{summary['delta_days']['max']}`",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    allowed_views = {view.strip().upper() for view in args.allowed_views}
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    excluded_patient_ids = load_excluded_patient_ids(args.exclude_records)

    study_times = load_study_times(args.metadata_csv)
    studies = choose_best_frontal_per_study(
        args.master_csv,
        study_times,
        allowed_views=allowed_views,
    )

    report_zip: zipfile.ZipFile | None = None
    if args.include_report_sections or args.require_reports:
        report_zip = zipfile.ZipFile(args.reports_zip)
    try:
        rows = build_pairs(
            studies,
            report_zip=report_zip,
            baseline_problem=args.baseline_problem,
            require_reports=bool(args.require_reports),
            min_days_between_studies=int(args.min_days_between_studies),
            max_pairs=args.max_pairs,
            excluded_patient_ids=excluded_patient_ids,
        )
    finally:
        if report_zip is not None:
            report_zip.close()

    manifest_path = output_dir / "manifest.jsonl"
    write_jsonl(manifest_path, rows)
    summary = build_summary(rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_readme(output_dir / "README.md", args, summary)

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "manifest_path": str(manifest_path),
                "summary": summary,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
