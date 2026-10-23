# Portable supplementary copy: default paths/mount aliases sanitized; original hash is in FILE_MANIFEST.json.
#!/usr/bin/env python3
"""MedGemma-native paired-image SFT warm-start for longitudinal CXR change extraction."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from PIL import Image

from medgemma_visual_rft.longitudinal_label_schema import (
    build_stage1_label_problem,
    canonicalize_stage1_label,
    serialize_stage1_label_solution,
)
from medgemma_visual_rft.longitudinal_schema import parse_longitudinal_completion


DEFAULT_MODEL_PATH = Path("./medgemma/google-medgemma-1.5-4b-it")
DEFAULT_DATASET_DIR = REPO_ROOT / "share_data" / "mimic_cxr_longitudinal_sft_preview"
DEFAULT_TRAIN_JSONL = DEFAULT_DATASET_DIR / "train.jsonl"
DEFAULT_EVAL_JSONL = DEFAULT_DATASET_DIR / "val.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "medgemma_runs" / "longitudinal_sft_warmstart"
DEFAULT_JSON_SYSTEM_PROMPT = (
    "You are an expert medical imaging assistant. Output raw JSON only. "
    "Do not use markdown fences, code blocks, or extra explanatory text."
)
DEFAULT_LABEL_SYSTEM_PROMPT = (
    "You are an expert medical imaging assistant. Output exactly one canonical answer label and nothing else."
)
DEFAULT_MAX_SEQ_LENGTH = 4096


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--train-jsonl", type=Path, default=DEFAULT_TRAIN_JSONL)
    parser.add_argument("--eval-jsonl", type=Path, default=DEFAULT_EVAL_JSONL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--system-prompt", default=DEFAULT_JSON_SYSTEM_PROMPT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=0.3)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH)
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--load-in-4bit", action="store_true", default=True)
    parser.add_argument("--no-load-in-4bit", dest="load_in_4bit", action="store_false")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", default="all-linear")
    parser.add_argument("--modules-to-save", nargs="+", default=["lm_head", "embed_tokens"])
    parser.add_argument("--report-to", default="tensorboard")
    parser.add_argument("--resume-from-checkpoint", type=Path, default=None)
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--stage1-answer-mode",
        choices=("tuple_json", "label"),
        default="tuple_json",
        help="Keep the legacy tuple JSON bridge or override Stage-1 rows to direct label outputs.",
    )
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def maybe_import_dependency(name: str, message: str) -> Any:
    try:
        module = __import__(name, fromlist=["*"])
    except ImportError as exc:
        raise SystemExit(message) from exc
    return module


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if not isinstance(row, dict):
                raise SystemExit(f"Expected object on line {line_number} of {path}")
            rows.append(row)
    return rows


def split_name_from_path(path: Path) -> str:
    stem = path.stem.lower()
    if stem == "validation":
        return "val"
    if stem not in {"train", "val", "test"}:
        raise SystemExit(f"Unsupported split file name: {path.name}")
    return stem


def _require_nonempty_str(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise SystemExit(f"`{key}` must be a non-empty string for row {row.get('source_id')!r}")
    return value


def _optional_clean_str(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def resolve_image_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        raise SystemExit(f"Expected absolute image path in longitudinal dataset, got: {path_str}")
    if not path.exists():
        raise SystemExit(f"Image path does not exist: {path}")
    return path


def canonicalize_longitudinal_row(
    raw_row: dict[str, Any],
    *,
    split: str,
    stage1_answer_mode: str,
) -> dict[str, Any]:
    source_id = _require_nonempty_str(raw_row, "source_id")
    if raw_row.get("split") != split:
        raise SystemExit(
            f"Row {source_id!r} split mismatch: file implies {split!r}, row contains {raw_row.get('split')!r}"
        )
    prior_image_path = str(resolve_image_path(_require_nonempty_str(raw_row, "prior_image_path")))
    current_image_path = str(resolve_image_path(_require_nonempty_str(raw_row, "current_image_path")))
    patient_id = _require_nonempty_str(raw_row, "patient_id")
    pair_id = _require_nonempty_str(raw_row, "pair_id")
    label_status = _require_nonempty_str(raw_row, "label_status")
    metadata = raw_row.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise SystemExit(f"`metadata` must be a JSON object for row {source_id!r}")
    prior_view_position = _optional_clean_str(raw_row.get("prior_view_position"))
    current_view_position = _optional_clean_str(raw_row.get("current_view_position"))
    task_mode = (
        _optional_clean_str(raw_row.get("task_mode"))
        or _optional_clean_str(metadata.get("task_mode"))
        or "stage2_open_ended"
    )
    query_anatomy = _optional_clean_str(raw_row.get("query_anatomy")) or _optional_clean_str(
        metadata.get("query_anatomy")
    )
    query_finding = _optional_clean_str(raw_row.get("query_finding")) or _optional_clean_str(
        metadata.get("query_finding")
    )
    query_change = _optional_clean_str(raw_row.get("query_change")) or _optional_clean_str(
        metadata.get("query_change")
    )

    target_changes: list[dict[str, Any]]
    if task_mode == "stage1" and stage1_answer_mode == "label":
        if query_finding is None or query_change is None:
            raise SystemExit(
                f"Stage-1 label mode requires query_finding and query_change for row {source_id!r}"
            )
        problem = build_stage1_label_problem(query_finding=query_finding, query_anatomy=query_anatomy)
        solution = serialize_stage1_label_solution(query_change)
        target_changes = []
    else:
        problem = _require_nonempty_str(raw_row, "problem")
        solution = _require_nonempty_str(raw_row, "solution")
        parsed = parse_longitudinal_completion(solution)
        target_changes = [
            {
                "anatomy": item.anatomy,
                "finding": item.finding,
                "change": item.change,
                "box_current": list(item.box_current) if item.box_current is not None else None,
            }
            for item in parsed.changes
        ]

    return {
        "source_id": source_id,
        "pair_id": pair_id,
        "patient_id": patient_id,
        "split": split,
        "prior_image_path": prior_image_path,
        "current_image_path": current_image_path,
        "problem": problem,
        "solution": solution,
        "label_status": label_status,
        "task_mode": task_mode,
        "query_anatomy": query_anatomy,
        "query_finding": query_finding,
        "query_change": canonicalize_stage1_label(query_change) if query_change is not None else None,
        "prior_view_position": prior_view_position,
        "current_view_position": current_view_position,
        "same_view_pair": (
            prior_view_position is not None
            and current_view_position is not None
            and prior_view_position == current_view_position
        ),
        "target_changes": target_changes,
    }


def load_longitudinal_split(jsonl_path: Path, *, stage1_answer_mode: str) -> list[dict[str, Any]]:
    split = split_name_from_path(jsonl_path)
    return [
        canonicalize_longitudinal_row(row, split=split, stage1_answer_mode=stage1_answer_mode)
        for row in load_jsonl(jsonl_path)
    ]


def maybe_subsample(rows: list[dict[str, Any]], limit: int | None, *, seed: int) -> list[dict[str, Any]]:
    if limit is None or limit >= len(rows):
        return rows
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    keep = sorted(indices[:limit])
    return [rows[index] for index in keep]


def build_messages(
    system_prompt: str,
    example: dict[str, Any],
    prior_image: Image.Image,
    current_image: Image.Image,
) -> list[dict[str, Any]]:
    import os as _os
    img = [{"type": "image", "image": prior_image}, {"type": "image", "image": current_image}]
    txt = [{"type": "text", "text": example["problem"]}]
    # FINDING_FIRST=1 places the query text BEFORE the images so image tokens can
    # causally attend to the queried finding (conditioning-by-ordering intervention).
    user_content = (txt + img) if _os.environ.get("FINDING_FIRST") == "1" else (img + txt)
    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": user_content},
        {
            "role": "assistant",
            "content": [{"type": "text", "text": example["solution"]}],
        },
    ]


@lru_cache(maxsize=2048)
def load_rgb_image(path_str: str) -> Image.Image:
    with Image.open(path_str) as image:
        return image.convert("RGB")


def build_collator(processor: Any, *, system_prompt: str) -> Any:
    pad_token_id = processor.tokenizer.pad_token_id
    special_tokens_map = getattr(processor.tokenizer, "special_tokens_map", {})
    boi_token_id = None
    if isinstance(special_tokens_map, dict) and "boi_token" in special_tokens_map:
        boi_token = special_tokens_map["boi_token"]
        boi_token_id = processor.tokenizer.convert_tokens_to_ids(boi_token)

    def collate_fn(examples: list[dict[str, Any]]) -> dict[str, Any]:
        texts: list[str] = []
        images: list[list[Image.Image]] = []
        for example in examples:
            prior_image = load_rgb_image(example["prior_image_path"])
            current_image = load_rgb_image(example["current_image_path"])
            texts.append(
                processor.apply_chat_template(
                    build_messages(system_prompt, example, prior_image, current_image),
                    add_generation_prompt=False,
                    tokenize=False,
                ).strip()
            )
            images.append([prior_image.copy(), current_image.copy()])

        batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
        labels = batch["input_ids"].clone()
        labels[labels == pad_token_id] = -100
        if boi_token_id is not None:
            labels[labels == boi_token_id] = -100
        labels[labels == 262144] = -100
        batch["labels"] = labels
        return batch

    return collate_fn


def build_model_and_processor(args: argparse.Namespace) -> tuple[Any, Any]:
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this MedGemma longitudinal SFT entrypoint.")

    dtype = resolve_dtype(args.dtype)
    if dtype == torch.bfloat16 and torch.cuda.get_device_capability()[0] < 8:
        raise SystemExit("Selected bfloat16, but the active GPU does not advertise bf16 support.")

    model_kwargs: dict[str, Any] = {
        "attn_implementation": "eager",
        "torch_dtype": dtype,
        "device_map": "auto",
        "low_cpu_mem_usage": True,
    }
    if args.load_in_4bit:
        _ = maybe_import_dependency(
            "bitsandbytes",
            "bitsandbytes is required for --load-in-4bit. Install it or rerun with --no-load-in-4bit.",
        )
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_storage=dtype,
        )

    model = AutoModelForImageTextToText.from_pretrained(str(args.model), **model_kwargs)
    processor = AutoProcessor.from_pretrained(str(args.model), use_fast=True)
    processor.tokenizer.padding_side = "right"
    return model, processor


def build_processor_only(args: argparse.Namespace) -> Any:
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(str(args.model), use_fast=True)
    processor.tokenizer.padding_side = "right"
    return processor


def build_peft_config(args: argparse.Namespace) -> Any:
    from peft import LoraConfig

    return LoraConfig(
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        r=args.lora_r,
        bias="none",
        target_modules=args.target_modules,
        task_type="CAUSAL_LM",
        modules_to_save=args.modules_to_save,
    )


def build_trainer(
    args: argparse.Namespace,
    *,
    model: Any,
    processor: Any,
    train_dataset: Any,
    eval_dataset: Any | None,
    collator: Any,
) -> Any:
    from peft import prepare_model_for_kbit_training
    from trl import SFTConfig, SFTTrainer

    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    dtype = resolve_dtype(args.dtype)
    trainer_args = SFTConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=args.eval_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="linear",
        report_to=args.report_to,
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
        label_names=["labels"],
        max_length=args.max_seq_length,
        packing=False,
        seed=args.seed,
    )

    return SFTTrainer(
        model=model,
        args=trainer_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=build_peft_config(args),
        processing_class=processor,
        data_collator=collator,
    )


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def write_run_manifest(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]] | None,
) -> None:
    label_status_counts = {}
    task_mode_counts = {}
    query_supervision = {}
    view_pair_counts = {}
    for split_name, rows in (("train", train_rows), ("eval", eval_rows or [])):
        counts: dict[str, int] = {}
        task_counts: dict[str, int] = {}
        query_counts = {
            "query_rows": 0,
            "same_view_rows": 0,
        }
        split_view_pairs: dict[str, int] = {}
        for row in rows:
            counts[row["label_status"]] = counts.get(row["label_status"], 0) + 1
            task_counts[row["task_mode"]] = task_counts.get(row["task_mode"], 0) + 1
            if row.get("query_finding") is not None:
                query_counts["query_rows"] += 1
            if row.get("same_view_pair"):
                query_counts["same_view_rows"] += 1
            prior_view = row.get("prior_view_position")
            current_view = row.get("current_view_position")
            if prior_view and current_view:
                key = f"{prior_view}->{current_view}"
                split_view_pairs[key] = split_view_pairs.get(key, 0) + 1
        label_status_counts[split_name] = counts
        task_mode_counts[split_name] = task_counts
        query_supervision[split_name] = query_counts
        view_pair_counts[split_name] = split_view_pairs

    manifest = {
        "objective": "MedGemma SFT warm-start for paired-image longitudinal CXR change extraction",
        "model": str(args.model),
        "train_jsonl": str(args.train_jsonl),
        "eval_jsonl": str(args.eval_jsonl) if args.eval_jsonl else None,
        "output_dir": str(output_dir),
        "seed": args.seed,
        "train_count": len(train_rows),
        "eval_count": len(eval_rows) if eval_rows is not None else 0,
        "label_status_counts": label_status_counts,
        "task_mode_counts": task_mode_counts,
        "query_supervision": query_supervision,
        "view_pair_counts": view_pair_counts,
        "max_train_samples": args.max_train_samples,
        "max_eval_samples": args.max_eval_samples,
        "dtype": args.dtype,
        "load_in_4bit": args.load_in_4bit,
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": args.target_modules,
            "modules_to_save": args.modules_to_save,
        },
        "prompt_contract": {
            "system_prompt": args.system_prompt,
            "answer_format": (
                "one canonical label: improving|stable|worsening|no_supported_evidence"
                if args.stage1_answer_mode == "label"
                else '{"changes":[{"anatomy":"...","finding":"...","change":"..."}]}'
            ),
            "stage1_answer_mode": args.stage1_answer_mode,
            "wrappers_allowed": False,
            "assistant_prefill_in_training": False,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.stage1_answer_mode == "label" and args.system_prompt == DEFAULT_JSON_SYSTEM_PROMPT:
        args.system_prompt = DEFAULT_LABEL_SYSTEM_PROMPT
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_rows = load_longitudinal_split(args.train_jsonl, stage1_answer_mode=args.stage1_answer_mode)
    eval_rows = (
        None
        if args.no_eval
        else load_longitudinal_split(args.eval_jsonl, stage1_answer_mode=args.stage1_answer_mode)
    )
    train_rows = maybe_subsample(train_rows, args.max_train_samples, seed=args.seed)
    if eval_rows is not None:
        eval_rows = maybe_subsample(eval_rows, args.max_eval_samples, seed=args.seed + 1)

    write_run_manifest(args.output_dir, args=args, train_rows=train_rows, eval_rows=eval_rows)

    if args.dry_run:
        processor = build_processor_only(args)
        collator = build_collator(processor, system_prompt=args.system_prompt)
        preview_batch = collator([train_rows[0]])
        preview = {
            "objective": "MedGemma longitudinal SFT dry-run",
            "output_dir": str(args.output_dir),
            "train_count": len(train_rows),
            "eval_count": len(eval_rows) if eval_rows is not None else 0,
            "batch_keys": sorted(preview_batch.keys()),
            "input_ids_shape": list(preview_batch["input_ids"].shape),
            "labels_shape": list(preview_batch["labels"].shape),
        }
        print(json.dumps(preview, indent=2))
        return 0

    try:
        from datasets import Dataset
    except ImportError as exc:
        raise SystemExit(
            "datasets is required to build the trainer datasets. Install it or provide a Python environment "
            "with Hugging Face datasets available."
        ) from exc

    train_dataset = Dataset.from_list(train_rows)
    eval_dataset = Dataset.from_list(eval_rows) if eval_rows is not None else None

    model, processor = build_model_and_processor(args)
    collator = build_collator(processor, system_prompt=args.system_prompt)
    trainer = build_trainer(
        args,
        model=model,
        processor=processor,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        collator=collator,
    )

    train_result = trainer.train(
        resume_from_checkpoint=str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None
    )
    trainer.save_model(str(args.output_dir))
    processor.save_pretrained(str(args.output_dir))

    metrics = {"train": jsonable(train_result.metrics)}
    if eval_dataset is not None:
        metrics["eval"] = jsonable(trainer.evaluate())
    (args.output_dir / "metrics.json").write_text(json.dumps(jsonable(metrics), indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "metrics": jsonable(metrics)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
