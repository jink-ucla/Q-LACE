# Portable supplementary copy: default paths/mount aliases sanitized; original hash is in FILE_MANIFEST.json.
#!/usr/bin/env python3
"""Auditable three-option, row-preserving MedGemma baseline (no training).

``prepare`` and ``score`` use only the standard library. ``generate`` requires a
visible CUDA GPU; there is deliberately no CPU/offload inference fallback.
The protocol JSON must be pinned before inference, independently of test scores.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import re
import sys
import tempfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "manuscript_rev10/artifacts/cmig_validation/baseline"
DEFAULT_MODEL = Path("./medgemma/google-medgemma-1.5-4b-it")
LABELS = ("improving", "stable", "worsening")
FINDINGS = {"consolidation", "edema", "pleural effusion", "pneumonia", "pneumothorax"}
SCHEMA = "cmig-medgemma-three-option-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
                                     separators=(",", ":")).encode()).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def write_json(path: Path, value: Any) -> None:
    atomic_write(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    atomic_write(path, "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def canonical_label(value: Any) -> str:
    aliases = {"improved": "improving", "improving": "improving", "stable": "stable",
               "worsened": "worsening", "worsening": "worsening"}
    key = str(value).strip().lower()
    if key not in aliases:
        raise ValueError(f"Unsupported three-class gold label: {value!r}")
    return aliases[key]


def build_manifest(source: Path, *, expected_rows: int = 1326,
                   check_images: bool = True) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_rows = load_jsonl(source)
    if len(raw_rows) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows; found {len(raw_rows)}")
    result = []
    for index, raw in enumerate(raw_rows):
        finding = str(raw.get("query_finding", "")).strip().lower().replace("_", " ")
        if finding not in FINDINGS:
            raise ValueError(f"Unsupported finding at row {index}: {finding!r}")
        paths = {}
        for role, field in (("prior", "prior_image_path"), ("current", "current_image_path")):
            path = Path(raw[field])
            if not path.is_absolute():
                path = ROOT / path
            path = path.resolve()
            if check_images and not path.is_file():
                raise ValueError(f"Missing {role} image for row {index}: {path}")
            paths[field] = str(path)
        physical_key = [paths["prior_image_path"], paths["current_image_path"], finding]
        row_id = f"{index:06d}-{digest_json(physical_key)[:24]}"
        result.append({
            "row_id": row_id, "row_index": index, "physical_key": physical_key,
            "physical_key_sha256": digest_json(physical_key), **paths,
            "prior_dicom_id": raw.get("prior_dicom_id"),
            "current_dicom_id": raw.get("dicom_id"),
            "source_id": raw.get("source_id"), "pair_id": raw.get("pair_id"),
            "patient_id": str(raw.get("patient_id", raw.get("subject_id", ""))),
            "split": raw.get("split"), "query_finding": finding,
            "target_label": canonical_label(raw.get("query_change")),
            "prior_view_position": raw.get("prior_view_position"),
            "current_view_position": raw.get("current_view_position"),
            "source_csv_row_index": (raw.get("metadata") or {}).get("csv_row_index"),
        })
        if not result[-1]["patient_id"]:
            raise ValueError(f"Missing patient ID at row {index}")
    source_counts = Counter(row["source_id"] for row in result)
    physical_counts = Counter(row["physical_key_sha256"] for row in result)
    meta = {
        "schema": SCHEMA, "source_path": str(source.resolve()),
        "source_sha256": sha256_file(source), "manifest_sha256": digest_json(result),
        "n_rows": len(result), "n_patients": len({r["patient_id"] for r in result}),
        "n_physical_image_pairs": len({tuple(r["physical_key"][:2]) for r in result}),
        "n_distinct_physical_finding_keys": len(physical_counts),
        "duplicate_source_id_extra_rows": sum(n - 1 for n in source_counts.values()),
        "duplicate_physical_finding_extra_rows": sum(n - 1 for n in physical_counts.values()),
        "finding_counts": dict(Counter(r["query_finding"] for r in result)),
        "label_counts": dict(Counter(r["target_label"] for r in result)),
        "row_policy": "Preserve every input row and its order; never deduplicate source_id.",
        "identity": "row_index plus SHA256(real prior path, real current path, finding); gold excluded",
        "image_existence_checked": check_images,
    }
    return result, meta


def read_manifest(path: Path, expected_rows: int = 1326) -> list[dict[str, Any]]:
    rows = load_jsonl(path)
    if len(rows) != expected_rows or len({r["row_id"] for r in rows}) != len(rows):
        raise ValueError("Manifest row count or unique row IDs failed")
    if [r["row_index"] for r in rows] != list(range(len(rows))):
        raise ValueError("Manifest row order/index is not contiguous")
    for row in rows:
        key = [row["prior_image_path"], row["current_image_path"], row["query_finding"]]
        expected_id = f"{row['row_index']:06d}-{digest_json(key)[:24]}"
        if row["physical_key"] != key or row["row_id"] != expected_id:
            raise ValueError(f"Manifest physical identity mismatch: {row['row_id']}")
        canonical_label(row["target_label"])
    return rows


def validate_protocol(protocol: dict[str, Any]) -> dict[str, Any]:
    required = {"protocol_id", "question_template", "option_mapping", "content_layout",
                "system_prompt", "max_new_tokens", "provenance_status", "source_reference"}
    missing = required - protocol.keys()
    if missing:
        raise ValueError(f"Protocol is missing {sorted(missing)}")
    mapping = protocol["option_mapping"]
    if set(mapping) != {"A", "B", "C"} or set(mapping.values()) != set(LABELS):
        raise ValueError("Option mapping must be an A/B/C bijection onto the three supported labels")
    layout = protocol["content_layout"]
    allowed = {"prior_annotation", "prior_image", "current_annotation", "current_image", "question"}
    if set(layout) - allowed or any(layout.count(v) != 1 for v in ("prior_image", "current_image", "question")):
        raise ValueError("Layout must include each image and question exactly once")
    if not isinstance(protocol["max_new_tokens"], int) or protocol["max_new_tokens"] < 1:
        raise ValueError("max_new_tokens must be a positive integer")
    protocol["question_template"].format(finding="edema")
    if "{finding}" not in protocol["question_template"]:
        raise ValueError("Question template must explicitly interpolate finding")
    if not protocol.get("frozen_before_inference", False):
        raise ValueError("Protocol must explicitly be frozen_before_inference=true")
    return protocol


def parse_answer(text: str, option_mapping: dict[str, str]) -> tuple[str | None, str | None]:
    # Only accept a complete single-option or single-supported-label response.
    # Do not recover labels embedded in explanations, JSON, or contradictory prose.
    cleaned = text.strip()
    final = re.fullmatch(r"Final Answer:\s*([ABC])[.]?", cleaned, re.IGNORECASE)
    if final:
        return option_mapping[final.group(1).upper()], None
    option = re.fullmatch(r"(?:\(([ABC])\)|([ABC]))[.]?", cleaned, re.IGNORECASE)
    if option:
        return option_mapping[(option.group(1) or option.group(2)).upper()], None
    label = re.fullmatch(r"(improved|improving|stable|worsened|worsening)[.]?", cleaned, re.IGNORECASE)
    if label:
        return canonical_label(label.group(1)), None
    return None, "Completion is not exactly one option or supported label"


def prepare_image(image: Any, protocol: dict[str, Any]) -> Any:
    """Optional historical-padding sensitivity; primary keeps native aspect."""
    if not protocol.get("external_square_padding", False):
        return image
    from PIL import Image
    edge = max(image.size)
    padded = Image.new("RGB", (edge, edge), color=(0, 0, 0))
    padded.paste(image, ((edge - image.width) // 2, (edge - image.height) // 2))
    return padded


def make_messages(protocol: dict[str, Any], row: dict[str, Any], prior: Any,
                  current: Any) -> list[dict[str, Any]]:
    text_values = {
        "prior_annotation": protocol.get("prior_annotation", ""),
        "current_annotation": protocol.get("current_annotation", ""),
        "question": protocol["question_template"].format(finding=row["query_finding"]),
    }
    images = {"prior_image": prior, "current_image": current}
    content = []
    for part in protocol["content_layout"]:
        if part in images:
            content.append({"type": "image", "image": images[part]})
        elif text_values[part]:
            content.append({"type": "text", "text": text_values[part]})
    messages = []
    if protocol["system_prompt"].strip():
        messages.append({"role": "system", "content": [{"type": "text", "text": protocol["system_prompt"]}]})
    messages.append({"role": "user", "content": content})
    return messages


def validate_predictions(rows: list[dict[str, Any]], predictions: list[dict[str, Any]],
                         *, allow_partial: bool = False) -> dict[str, dict[str, Any]]:
    reference = {r["row_id"]: r for r in rows}
    output = {}
    for pred in predictions:
        row_id = pred["row_id"]
        if row_id not in reference or row_id in output:
            raise ValueError(f"Unexpected or repeated prediction row ID: {row_id}")
        row = reference[row_id]
        for key in ("row_index", "physical_key", "query_finding", "target_label", "patient_id"):
            if pred.get(key) != row[key]:
                raise ValueError(f"Prediction {key} mismatch for {row_id}")
        if pred.get("pred_label") is not None and pred["pred_label"] not in LABELS:
            raise ValueError(f"Unsupported prediction label for {row_id}")
        output[row_id] = pred
    if not allow_partial and set(reference) != set(output):
        raise ValueError(f"Incomplete predictions: {len(output)}/{len(reference)}")
    return output


def classification_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(records)
    if not n:
        raise ValueError("Cannot score an empty population")
    per_class = {}
    for label in LABELS:
        tp = sum(r["target_label"] == label and r.get("pred_label") == label for r in records)
        fp = sum(r["target_label"] != label and r.get("pred_label") == label for r in records)
        fn = sum(r["target_label"] == label and r.get("pred_label") != label for r in records)
        support = tp + fn
        per_class[label] = {"support": support, "tp": tp, "fp": fp, "fn": fn,
                            "recall": tp / support if support else None,
                            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}
    recalls = [v["recall"] for v in per_class.values() if v["recall"] is not None]
    return {
        "n_rows": n,
        "accuracy": sum(r.get("pred_label") == r["target_label"] for r in records) / n,
        "supported_macro_f1": sum(v["f1"] for v in per_class.values()) / 3,
        "class_balanced_accuracy": sum(recalls) / len(recalls),
        "invalid_count": sum(r.get("pred_label") is None for r in records),
        "truncated_count": sum(bool(r.get("truncated")) for r in records),
        "per_class": per_class,
        "predicted_label_counts": dict(Counter(r.get("pred_label") or "INVALID" for r in records)),
    }


def score_predictions(rows: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    lookup = validate_predictions(rows, predictions)
    ordered = [lookup[row["row_id"]] for row in rows]
    grouped = defaultdict(list)
    for record in ordered:
        grouped[record["query_finding"]].append(record)
    per_finding = {name: classification_metrics(records) for name, records in sorted(grouped.items())}
    result = classification_metrics(ordered)
    result.update({
        "schema": SCHEMA, "n_patients": len({r["patient_id"] for r in rows}),
        "manifest_sha256": digest_json(rows), "per_finding": per_finding,
        "mean_per_finding_accuracy": sum(m["accuracy"] for m in per_finding.values()) / len(per_finding),
        "mean_per_finding_class_balanced_accuracy": sum(m["class_balanced_accuracy"] for m in per_finding.values()) / len(per_finding),
        "published_macro_accuracy_definition": "UNVERIFIED: use explicitly named metrics; no alias to published score",
        "metric_definitions": {
            "accuracy": "Correct predictions / all original rows; invalid outputs count incorrect.",
            "supported_macro_f1": "Unweighted mean F1 over exactly improving, stable, worsening; invalid outputs incur false negatives.",
            "class_balanced_accuracy": "Unweighted mean recall over gold-supported change classes.",
            "mean_per_finding_accuracy": "Unweighted mean of within-finding row accuracy.",
            "mean_per_finding_class_balanced_accuracy": "Unweighted mean of within-finding gold-supported class recall averages.",
        },
    })
    return result


def checkpoint_fingerprint(directory: Path) -> dict[str, Any]:
    directory = directory.resolve()
    files = sorted(p for p in directory.iterdir() if p.is_file() and p.suffix in {".json", ".jinja", ".model", ".safetensors", ".bin"})
    if not files:
        raise ValueError(f"No checkpoint files found in {directory}")
    entries = [{"name": p.name, "bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in files]
    return {"path": str(directory), "files": entries, "files_sha256": digest_json(entries)}


def generation_contract(args: argparse.Namespace, rows: list[dict[str, Any]],
                        protocol: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": SCHEMA, "manifest_sha256": digest_json(rows),
        "protocol": protocol, "protocol_sha256": digest_json(protocol),
        "model": checkpoint_fingerprint(args.model_path),
        "adapter": checkpoint_fingerprint(args.adapter_path) if args.adapter_path else None,
        "seed": args.seed, "dtype": "bfloat16", "attention_implementation": "eager",
        "do_sample": False, "device": "cuda:0", "cpu_offload": False,
        "use_cache": True, "script_sha256": sha256_file(Path(__file__)),
    }


def generate(args: argparse.Namespace) -> None:
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for inference; no CPU fallback is permitted")
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    rows = read_manifest(args.manifest, args.expected_rows)
    protocol = validate_protocol(json.loads(args.protocol.read_text()))
    if args.limit is not None and args.limit < 1:
        raise ValueError("Smoke limit must be positive")
    selected = rows if args.limit is None else rows[:args.limit]
    contract = generation_contract(args, rows, protocol)
    contract.update({"selected_row_ids": [r["row_id"] for r in selected], "smoke_only": args.limit is not None})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_path = args.output_dir / "run_manifest.json"
    prediction_path = args.output_dir / "predictions.jsonl"
    if run_path.exists():
        previous = json.loads(run_path.read_text())
        if previous["contract"] != contract:
            raise ValueError("Resume contract mismatch; use a separate output directory")
    else:
        if prediction_path.exists():
            raise ValueError("Predictions exist without a provenance manifest")
        versions = {}
        for package in ("torch", "transformers", "peft", "pillow", "accelerate"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        write_json(run_path, {"contract": contract, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                              "python": sys.version, "package_versions": versions,
                              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                              "gpu_name": torch.cuda.get_device_name(0), "cuda_version": torch.version.cuda})
    existing = load_jsonl(prediction_path) if prediction_path.exists() else []
    completed = validate_predictions(selected, existing, allow_partial=True)
    if len(completed) == len(selected):
        print(f"Already complete: {len(selected)} rows", flush=True)
        return
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    processor = AutoProcessor.from_pretrained(str(args.model_path), local_files_only=True, use_fast=True)
    model = AutoModelForImageTextToText.from_pretrained(
        str(args.model_path), torch_dtype=torch.bfloat16, device_map={"": "cuda:0"},
        local_files_only=True, attn_implementation="eager")
    if args.adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(args.adapter_path), is_trainable=False,
                                          local_files_only=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    eos = model.generation_config.eos_token_id
    if eos is None:
        eos = processor.tokenizer.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos]) - {None}
    for row in selected:
        if row["row_id"] in completed:
            continue
        with Image.open(row["prior_image_path"]) as prior_file, Image.open(row["current_image_path"]) as current_file:
            prior, current = prior_file.convert("RGB"), current_file.convert("RGB")
        original_sizes = {"prior": list(prior.size), "current": list(current.size)}
        prior, current = prepare_image(prior, protocol), prepare_image(current, protocol)
        messages = make_messages(protocol, row, prior, current)
        template_kwargs = {}
        if "do_pan_and_scan" in protocol:
            template_kwargs["do_pan_and_scan"] = protocol["do_pan_and_scan"]
        batch = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                               return_dict=True, return_tensors="pt", **template_kwargs)
        batch = {key: value.to(device="cuda:0", dtype=torch.bfloat16) if torch.is_tensor(value) and value.is_floating_point()
                 else value.to("cuda:0") if torch.is_tensor(value) else value for key, value in batch.items()}
        input_length = batch["input_ids"].shape[1]
        with torch.inference_mode():
            output = model.generate(**batch, do_sample=False, max_new_tokens=protocol["max_new_tokens"],
                                    eos_token_id=eos, pad_token_id=processor.tokenizer.pad_token_id,
                                    use_cache=True)
        token_ids = output[0, input_length:].detach().cpu().tolist()
        raw_text = processor.tokenizer.decode(token_ids, skip_special_tokens=True)
        prediction, parse_error = parse_answer(raw_text, protocol["option_mapping"])
        eos_observed = any(token in eos_ids for token in token_ids)
        record = {**row, "pred_label": prediction, "parse_error": parse_error,
                  "raw_text": raw_text, "raw_text_with_special_tokens": processor.tokenizer.decode(token_ids, skip_special_tokens=False),
                  "generated_token_ids": token_ids, "generated_tokens": len(token_ids),
                  "eos_observed": eos_observed,
                  "truncated": len(token_ids) >= protocol["max_new_tokens"] and not eos_observed,
                  "input_tokens": input_length,
                  "prior_image_sha256": sha256_file(Path(row["prior_image_path"])),
                  "current_image_sha256": sha256_file(Path(row["current_image_path"])),
                  "original_image_sizes": original_sizes,
                  "image_sizes": {"prior": list(prior.size), "current": list(current.size)},
                  "pixel_values_shape": list(batch["pixel_values"].shape),
                  "prompt_rendered": processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False),
                  "protocol_sha256": digest_json(protocol)}
        completed[row["row_id"]] = record
        write_jsonl(prediction_path, [completed[r["row_id"]] for r in selected if r["row_id"] in completed])
        print(f"Completed {len(completed)}/{len(selected)}; row {row['row_index']}", flush=True)
    predictions = [completed[r["row_id"]] for r in selected]
    result = score_predictions(selected, predictions)
    result["smoke_only"] = args.limit is not None
    result["n_full_manifest_rows"] = len(rows)
    write_json(args.output_dir / "metrics.json", result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--input-jsonl", type=Path, default=ROOT / "share_data/ms_cxr_t_longitudinal_stage1/test.jsonl")
    prepare.add_argument("--output-dir", type=Path, default=ARTIFACTS / "manifest")
    prepare.add_argument("--expected-rows", type=int, default=1326)
    infer = commands.add_parser("generate")
    infer.add_argument("--manifest", type=Path, default=ARTIFACTS / "manifest/evaluation_manifest.jsonl")
    infer.add_argument("--expected-rows", type=int, default=1326)
    infer.add_argument("--protocol", type=Path, required=True)
    infer.add_argument("--output-dir", type=Path, required=True)
    infer.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    infer.add_argument("--adapter-path", type=Path)
    infer.add_argument("--seed", type=int, default=42)
    infer.add_argument("--limit", type=int, help="Smoke only: separate output directory required")
    score = commands.add_parser("score")
    score.add_argument("--manifest", type=Path, default=ARTIFACTS / "manifest/evaluation_manifest.jsonl")
    score.add_argument("--predictions", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--expected-rows", type=int, default=1326)
    args = parser.parse_args()
    if args.command == "prepare":
        rows, meta = build_manifest(args.input_jsonl, expected_rows=args.expected_rows)
        write_jsonl(args.output_dir / "evaluation_manifest.jsonl", rows)
        write_json(args.output_dir / "manifest_summary.json", meta)
        print(json.dumps(meta, indent=2))
    elif args.command == "score":
        write_json(args.output, score_predictions(read_manifest(args.manifest, args.expected_rows),
                                                  load_jsonl(args.predictions)))
    else:
        generate(args)


if __name__ == "__main__":
    main()
