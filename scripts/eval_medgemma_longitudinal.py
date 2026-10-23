# Portable supplementary copy: default paths/mount aliases sanitized; original hash is in FILE_MANIFEST.json.
#!/usr/bin/env python3
"""Evaluate MedGemma on paired-image longitudinal CXR JSON outputs."""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")

from PIL import Image
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, StoppingCriteria, StoppingCriteriaList

from medgemma_visual_rft.longitudinal_classification import summarize_classification_records
from medgemma_visual_rft.longitudinal_label_schema import (
    Stage1LabelParseError,
    build_stage1_label_problem,
    canonicalize_stage1_label,
    parse_stage1_label_completion,
)
from medgemma_visual_rft.longitudinal_decoding import has_valid_longitudinal_json_prefix
from medgemma_visual_rft.longitudinal_schema import (
    LongitudinalJsonParseError,
    LongitudinalParseError,
    IntervalChange,
    parse_longitudinal_completion,
)

try:
    from peft import PeftModel
except Exception:
    PeftModel = None  # type: ignore[assignment]


DEFAULT_MODEL_PATH = Path("./medgemma/google-medgemma-1.5-4b-it")
DEFAULT_INPUT_JSONL = REPO_ROOT / "share_data" / "mimic_cxr_longitudinal_sft_preview" / "all.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "medgemma_runs" / "eval_medgemma_longitudinal"
DEFAULT_JSON_SYSTEM_PROMPT = (
    "You are an expert medical imaging assistant. Output raw JSON only. "
    "Do not use markdown fences, code blocks, or extra explanatory text."
)
DEFAULT_LABEL_SYSTEM_PROMPT = (
    "You are an expert medical imaging assistant. Output exactly one canonical answer label and nothing else."
)


class StrictLongitudinalJsonStopper(StoppingCriteria):
    def __init__(
        self,
        *,
        tokenizer: Any,
        prompt_length: int,
        assistant_prefill: str | None,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length
        self.assistant_prefill = assistant_prefill or ""
        self._last_checked_length = -1

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any) -> bool:
        current_length = int(input_ids.shape[1])
        if current_length <= self.prompt_length or current_length == self._last_checked_length:
            return False
        self._last_checked_length = current_length
        suffix_ids = input_ids[0, self.prompt_length :]
        suffix_text = self.tokenizer.decode(suffix_ids, skip_special_tokens=True)
        return has_valid_longitudinal_json_prefix(f"{self.assistant_prefill}{suffix_text}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT_JSONL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_JSON_SYSTEM_PROMPT,
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument(
        "--json-stop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop generation as soon as a strict longitudinal JSON object closes.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--splits", nargs="*", default=None)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--assistant-prefill", type=str, default=None)
    parser.add_argument(
        "--stage1-answer-mode",
        choices=("tuple_json", "label"),
        default="tuple_json",
        help="Keep the legacy tuple JSON bridge or override Stage-1 rows to direct label outputs.",
    )
    parser.add_argument(
        "--preserve-stage1-problem",
        action="store_true",
        help=(
            "In Stage-1 label mode, use each input row's problem text instead of rebuilding "
            "the prompt from query_finding. This is intended for prompt/query intervention datasets."
        ),
    )
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Restrict model and processor loading to local files only.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow model-family-specific remote code when loading local or HF checkpoints.",
    )
    parser.add_argument(
        "--attn-implementation",
        choices=("auto", "eager", "sdpa", "flash_attention_2"),
        default="eager",
        help="Attention backend to request from transformers. Use auto to keep the model default.",
    )
    parser.add_argument(
        "--min-pixels",
        type=int,
        default=None,
        help="Optional lower image pixel budget for processors that support dynamic resizing.",
    )
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=None,
        help="Optional upper image pixel budget for processors that support dynamic resizing.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=0,
        help="Print generation progress every N rows when N > 0.",
    )
    parser.add_argument(
        "--generation-use-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the model KV cache during generation. Disable for long diagnostic sweeps if memory grows.",
    )
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def optional_clean_str(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def resolve_image_path(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        repo_relative = (REPO_ROOT / path).resolve()
        if repo_relative.exists():
            return repo_relative
        raise SystemExit(f"Expected absolute image path, got {path_str}")
    if not path.exists():
        raise SystemExit(f"Image path does not exist: {path}")
    return path


def canonicalize_row(
    raw_row: dict[str, Any],
    *,
    stage1_answer_mode: str,
    preserve_stage1_problem: bool = False,
) -> dict[str, Any]:
    metadata = raw_row.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise SystemExit(f"`metadata` must be a JSON object for row {raw_row.get('source_id')!r}")
    prior_view_position = optional_clean_str(raw_row.get("prior_view_position"))
    current_view_position = optional_clean_str(raw_row.get("current_view_position"))
    task_mode = (
        optional_clean_str(raw_row.get("task_mode"))
        or optional_clean_str(metadata.get("task_mode"))
        or "stage2_open_ended"
    )
    query_anatomy = optional_clean_str(raw_row.get("query_anatomy")) or optional_clean_str(
        metadata.get("query_anatomy")
    )
    query_finding = optional_clean_str(raw_row.get("query_finding")) or optional_clean_str(
        metadata.get("query_finding")
    )
    query_change = optional_clean_str(raw_row.get("query_change")) or optional_clean_str(
        metadata.get("query_change")
    )
    solution = str(raw_row.get("solution") or "").strip()
    problem = str(raw_row["problem"])
    target_changes: tuple[IntervalChange, ...] = ()
    if task_mode == "stage1" and stage1_answer_mode == "label":
        if query_finding is None or query_change is None:
            raise SystemExit(
                f"Stage-1 label mode requires query_finding and query_change for row {raw_row.get('source_id')!r}"
            )
        if not preserve_stage1_problem:
            problem = build_stage1_label_problem(query_finding=query_finding, query_anatomy=query_anatomy)
    else:
        parsed = parse_longitudinal_completion(solution)
        target_changes = parsed.changes
    return {
        "source_id": str(raw_row["source_id"]),
        "pair_id": str(raw_row["pair_id"]),
        "patient_id": str(raw_row["patient_id"]),
        "split": str(raw_row["split"]),
        "prior_image_path": str(resolve_image_path(str(raw_row["prior_image_path"]))),
        "current_image_path": str(resolve_image_path(str(raw_row["current_image_path"]))),
        "problem": problem,
        "solution": solution,
        "label_status": str(raw_row.get("label_status") or ""),
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


def maybe_filter_rows(
    rows: list[dict[str, Any]],
    *,
    splits: Iterable[str] | None,
    shuffle: bool,
    seed: int,
    limit: int | None,
) -> list[dict[str, Any]]:
    if splits is not None:
        allowed = set(splits)
        rows = [row for row in rows if row["split"] in allowed]
    if shuffle:
        random.Random(seed).shuffle(rows)
    if limit is not None:
        rows = rows[:limit]
    return rows


def pick_input_device(model: AutoModelForImageTextToText) -> torch.device:
    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict):
        for device in device_map.values():
            if isinstance(device, int):
                return torch.device(f"cuda:{device}")
            if isinstance(device, str) and device.startswith("cuda"):
                return torch.device(device)
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device("cpu")


def move_batch_to_device(batch: dict[str, Any], device: torch.device, dtype: torch.dtype) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            if value.is_floating_point():
                moved[key] = value.to(device=device, dtype=dtype)
            else:
                moved[key] = value.to(device=device)
        else:
            moved[key] = value
    return moved


def build_messages(
    system_prompt: str,
    user_prompt: str,
    prior_image: Image.Image,
    current_image: Image.Image,
    *,
    assistant_prefill: str | None,
) -> list[dict[str, Any]]:
    import os as _os
    messages: list[dict[str, Any]] = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
    _img = [{"type": "image", "image": prior_image}, {"type": "image", "image": current_image}]
    _txt = [{"type": "text", "text": user_prompt}]
    _user = (_txt + _img) if _os.environ.get("FINDING_FIRST") == "1" else (_img + _txt)
    messages.append({"role": "user", "content": _user})
    if assistant_prefill is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": assistant_prefill}]})
    return messages


def decode_response(
    processor: AutoProcessor,
    generated_sequence: torch.Tensor,
    input_ids: torch.Tensor,
    *,
    assistant_prefill: str | None,
) -> tuple[str, str]:
    generated_text = processor.post_process_image_text_to_text(
        generated_sequence.detach().cpu(),
        skip_special_tokens=True,
    )[0]
    input_text = processor.post_process_image_text_to_text(
        input_ids.detach().cpu(),
        skip_special_tokens=True,
    )[0]
    if generated_text.startswith(input_text):
        suffix = generated_text[len(input_text) :].strip()
    else:
        suffix = generated_text.strip()
    completion = f"{assistant_prefill}{suffix}" if assistant_prefill is not None else suffix
    return completion, generated_text


def load_model_and_processor(
    model_path: str,
    *,
    adapter_path: str | None,
    dtype: torch.dtype,
    local_files_only: bool,
    trust_remote_code: bool,
    attn_implementation: str,
    min_pixels: int | None,
    max_pixels: int | None,
) -> tuple[AutoProcessor, AutoModelForImageTextToText]:
    processor_kwargs: dict[str, Any] = {
        "use_fast": True,
        "local_files_only": local_files_only,
        "trust_remote_code": trust_remote_code,
    }
    if min_pixels is not None:
        processor_kwargs["min_pixels"] = min_pixels
    if max_pixels is not None:
        processor_kwargs["max_pixels"] = max_pixels
    processor = AutoProcessor.from_pretrained(model_path, **processor_kwargs)

    model_kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "device_map": "auto",
        "offload_buffers": True,
        "local_files_only": local_files_only,
        "trust_remote_code": trust_remote_code,
    }
    if attn_implementation != "auto":
        model_kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForImageTextToText.from_pretrained(model_path, **model_kwargs)
    if adapter_path is not None:
        if PeftModel is None:
            raise RuntimeError("adapter_path was provided but `peft` is not available.")
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    model.eval()
    return processor, model


def resolve_generation_eos_token_id(model: AutoModelForImageTextToText, processor: AutoProcessor) -> Any:
    generation_eos = getattr(model.generation_config, "eos_token_id", None)
    if generation_eos is not None:
        return generation_eos
    return processor.tokenizer.eos_token_id


def normalize_change_set(changes: Iterable[IntervalChange]) -> set[tuple[str, str, str]]:
    return {(item.anatomy, item.finding, item.change) for item in changes}


def tuple_metrics(pred: set[tuple[str, str, str]], target: set[tuple[str, str, str]]) -> dict[str, float]:
    overlap = len(pred & target)
    precision = overlap / max(1, len(pred))
    recall = overlap / max(1, len(target))
    f1 = 0.0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)
    exact = 1.0 if pred == target else 0.0
    return {
        "overlap": float(overlap),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "exact_set_match": float(exact),
    }


def evaluate_single(
    *,
    model: AutoModelForImageTextToText,
    processor: AutoProcessor,
    record: dict[str, Any],
    args: argparse.Namespace,
    dtype: torch.dtype,
) -> dict[str, Any]:
    prior_image = Image.open(record["prior_image_path"]).convert("RGB")
    current_image = Image.open(record["current_image_path"]).convert("RGB")
    messages = build_messages(
        args.system_prompt,
        record["problem"],
        prior_image,
        current_image,
        assistant_prefill=args.assistant_prefill,
    )
    batch = processor.apply_chat_template(
        messages,
        add_generation_prompt=args.assistant_prefill is None,
        continue_final_message=args.assistant_prefill is not None,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    input_device = pick_input_device(model)
    batch = move_batch_to_device(batch, input_device, dtype)

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": bool(args.do_sample),
        "pad_token_id": processor.tokenizer.pad_token_id,
        "eos_token_id": resolve_generation_eos_token_id(model, processor),
        "use_cache": bool(args.generation_use_cache),
    }
    if args.repetition_penalty != 1.0:
        generation_kwargs["repetition_penalty"] = args.repetition_penalty
    if args.do_sample:
        generation_kwargs["temperature"] = args.temperature if args.temperature > 0 else 0.7
        generation_kwargs["top_p"] = args.top_p
    if args.json_stop and not (record["task_mode"] == "stage1" and args.stage1_answer_mode == "label"):
        generation_kwargs["stopping_criteria"] = StoppingCriteriaList(
            [
                StrictLongitudinalJsonStopper(
                    tokenizer=processor.tokenizer,
                    prompt_length=int(batch["input_ids"].shape[1]),
                    assistant_prefill=args.assistant_prefill,
                )
            ]
        )

    with torch.inference_mode():
        generated = model.generate(**batch, **generation_kwargs)

    completion, generated_text = decode_response(
        processor,
        generated,
        batch["input_ids"],
        assistant_prefill=args.assistant_prefill,
    )
    base_record = {
        "source_id": record["source_id"],
        "pair_id": record["pair_id"],
        "patient_id": record["patient_id"],
        "split": record["split"],
        "label_status": record["label_status"],
        "task_mode": record["task_mode"],
        "query_anatomy": record["query_anatomy"],
        "query_finding": record["query_finding"],
        "query_change": record["query_change"],
        "prior_view_position": record["prior_view_position"],
        "current_view_position": record["current_view_position"],
        "same_view_pair": record["same_view_pair"],
        "completion": completion,
        "generated_text": generated_text,
    }

    if record["task_mode"] == "stage1" and args.stage1_answer_mode == "label":
        parse_success = 0.0
        parse_error: str | None = None
        pred_label: str | None = None
        try:
            pred_label = parse_stage1_label_completion(completion).label
            parse_success = 1.0
        except Stage1LabelParseError as exc:
            parse_error = f"{type(exc).__name__}: {exc}"
        target_label = record["query_change"]
        return {
            **base_record,
            "parse_success": parse_success,
            "parse_error": parse_error,
            "pred_label": pred_label,
            "target_label": target_label,
            "label_exact_match": 1.0 if pred_label == target_label else 0.0,
        }

    json_valid = 1.0
    try:
        json.loads(completion)
    except json.JSONDecodeError:
        json_valid = 0.0

    parse_success = 0.0
    parse_error = None
    pred_changes: tuple[IntervalChange, ...] = ()
    try:
        parsed = parse_longitudinal_completion(completion)
        pred_changes = parsed.changes
        parse_success = 1.0
    except LongitudinalParseError as exc:
        parse_error = f"{type(exc).__name__}: {exc}"

    pred_set = normalize_change_set(pred_changes)
    target_set = normalize_change_set(record["target_changes"])
    metrics = tuple_metrics(pred_set, target_set)
    return {
        **base_record,
        "json_valid": json_valid,
        "parse_success": parse_success,
        "pred_tuple_count": len(pred_changes),
        "target_tuple_count": len(record["target_changes"]),
        "empty_tuple_prediction": 1.0 if len(pred_changes) == 0 else 0.0,
        "parse_error": parse_error,
        "weak_overlap_precision": metrics["precision"],
        "weak_overlap_recall": metrics["recall"],
        "weak_overlap_f1": metrics["f1"],
        "exact_set_match": metrics["exact_set_match"],
        "pred_changes": [list(item) for item in sorted(pred_set)],
        "target_changes": [list(item) for item in sorted(target_set)],
    }


def mean_metric(records: list[dict[str, Any]], key: str) -> float:
    if not records:
        return 0.0
    return float(sum(float(record.get(key, 0.0)) for record in records) / len(records))


def summarize_by_split(records: list[dict[str, Any]], key: str) -> dict[str, float]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["split"]].append(record)
    return {split: mean_metric(group_records, key) for split, group_records in sorted(grouped.items())}


def counter_dict(values: Iterable[str]) -> dict[str, int]:
    return dict(Counter(values))


def resume_key(record: dict[str, Any]) -> str:
    """Row-level key for append/resume-safe generation.

    `source_id` can identify an image pair in some exports, so include the
    finding/anatomy/label fields that make the query row distinct.
    """

    label = record.get("target_label", record.get("query_change", ""))
    return json.dumps(
        [
            str(record.get("source_id", "")),
            str(record.get("pair_id", "")),
            str(record.get("query_finding", "")),
            str(record.get("query_anatomy", "")),
            str(label),
        ],
        ensure_ascii=True,
    )


def main() -> int:
    args = parse_args()
    if args.stage1_answer_mode == "label" and args.system_prompt == DEFAULT_JSON_SYSTEM_PROMPT:
        args.system_prompt = DEFAULT_LABEL_SYSTEM_PROMPT
    dtype = resolve_dtype(args.dtype)
    rows = [
        canonicalize_row(
            row,
            stage1_answer_mode=args.stage1_answer_mode,
            preserve_stage1_problem=args.preserve_stage1_problem,
        )
        for row in load_jsonl(args.input_jsonl)
    ]
    rows = maybe_filter_rows(rows, splits=args.splits, shuffle=args.shuffle, seed=args.seed, limit=args.limit)

    run_name = args.run_name or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = run_dir / "predictions.jsonl"

    records: list[dict[str, Any]] = []
    completed_keys: set[str] = set()
    if prediction_path.exists():
        for record in load_jsonl(prediction_path):
            key = resume_key(record)
            if key in completed_keys:
                continue
            completed_keys.add(key)
            records.append(record)

    rows_to_run = [row for row in rows if resume_key(row) not in completed_keys]
    total_rows = len(rows)
    if rows_to_run:
        processor, model = load_model_and_processor(
            str(args.model_path),
            adapter_path=args.adapter_path,
            dtype=dtype,
            local_files_only=bool(args.local_files_only),
            trust_remote_code=bool(args.trust_remote_code),
            attn_implementation=args.attn_implementation,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
        )

        with prediction_path.open("a", encoding="utf-8") as handle:
            for row in rows_to_run:
                record = evaluate_single(model=model, processor=processor, record=row, args=args, dtype=dtype)
                records.append(record)
                completed = len(records)
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if args.progress_every > 0 and (
                    completed == 1 or completed % args.progress_every == 0 or completed == total_rows
                ):
                    print(f"[eval_medgemma_longitudinal] generated {completed}/{total_rows}", flush=True)
    elif args.progress_every > 0:
        print(f"[eval_medgemma_longitudinal] generated {len(records)}/{total_rows} (resumed)", flush=True)

    shared_summary = {
        "count": len(records),
        "parse_success_rate": mean_metric(records, "parse_success"),
        "label_status_counts": dict(Counter(record["label_status"] for record in records)),
        "task_mode_counts": counter_dict(record["task_mode"] for record in records),
        "view_pair_counts": counter_dict(
            f"{record['prior_view_position']}->{record['current_view_position']}"
            for record in records
            if record["prior_view_position"] and record["current_view_position"]
        ),
        "same_view_counts": {
            "true": sum(1 for record in records if record["same_view_pair"] is True),
            "false": sum(1 for record in records if record["same_view_pair"] is False),
            "unknown": sum(
                1
                for record in records
                if record["prior_view_position"] is None or record["current_view_position"] is None
            ),
        },
        "args": {
            "input_jsonl": str(args.input_jsonl),
            "model_path": args.model_path,
            "adapter_path": args.adapter_path,
            "splits": list(args.splits) if args.splits is not None else None,
            "assistant_prefill": args.assistant_prefill,
            "max_new_tokens": args.max_new_tokens,
            "repetition_penalty": args.repetition_penalty,
            "json_stop": args.json_stop,
            "stage1_answer_mode": args.stage1_answer_mode,
            "preserve_stage1_problem": args.preserve_stage1_problem,
            "attn_implementation": args.attn_implementation,
            "min_pixels": args.min_pixels,
            "max_pixels": args.max_pixels,
        },
    }
    if args.stage1_answer_mode == "label" and all(record["task_mode"] == "stage1" for record in records):
        classification = summarize_classification_records(
            records,
            gold_key="target_label",
            pred_key="pred_label",
            finding_key="query_finding",
        )
        summary = {
            **shared_summary,
            "answer_mode": "label",
            "accuracy": classification["accuracy"],
            "exact_match_accuracy": classification["exact_match_accuracy"],
            "macro_f1": classification["macro_f1"],
            "invalid_prediction_rate": classification["invalid_prediction_rate"],
            "unsupported_positive_rate": classification["unsupported_positive_rate"],
            "prediction_counts": classification["prediction_counts"],
            "per_class": classification["per_class"],
            "by_finding": classification["by_finding"],
        }
    else:
        summary = {
            **shared_summary,
            "answer_mode": "tuple_json",
            "json_valid_rate": mean_metric(records, "json_valid"),
            "mean_tuple_count": mean_metric(records, "pred_tuple_count"),
            "empty_tuple_rate": mean_metric(records, "empty_tuple_prediction"),
            "weak_overlap_precision": mean_metric(records, "weak_overlap_precision"),
            "weak_overlap_recall": mean_metric(records, "weak_overlap_recall"),
            "weak_overlap_f1": mean_metric(records, "weak_overlap_f1"),
            "exact_set_match_rate": mean_metric(records, "exact_set_match"),
            "by_split": {
                "json_valid_rate": summarize_by_split(records, "json_valid"),
                "parse_success_rate": summarize_by_split(records, "parse_success"),
                "weak_overlap_f1": summarize_by_split(records, "weak_overlap_f1"),
                "exact_set_match_rate": summarize_by_split(records, "exact_set_match"),
            },
        }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
