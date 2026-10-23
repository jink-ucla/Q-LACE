#!/usr/bin/env python3
"""Print a fully specified paper-recipe command, or run it only with --execute."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys


def command(model: Path, data: Path, output: Path, seed: int, *, no_eval: bool = False) -> list[str]:
    root = Path(__file__).resolve().parents[1]
    cmd = [sys.executable, str(root / "scripts/train_medgemma_longitudinal_sft.py"),
            "--model", str(model), "--train-jsonl", str(data / "train.jsonl"),
            "--eval-jsonl", str(data / "val.jsonl"), "--output-dir", str(output),
            "--seed", str(seed), "--num-train-epochs", "1", "--max-train-samples", "2048",
            "--max-eval-samples", "512", "--per-device-train-batch-size", "1",
            "--per-device-eval-batch-size", "1", "--gradient-accumulation-steps", "1",
            "--learning-rate", "2e-4", "--max-seq-length", "1024", "--warmup-ratio", "0.03",
            "--max-grad-norm", "0.3", "--dtype", "bfloat16", "--no-load-in-4bit",
            "--stage1-answer-mode", "label", "--lora-r", "16", "--lora-alpha", "16",
            "--lora-dropout", "0.05", "--target-modules", "all-linear",
            "--modules-to-save", "lm_head", "embed_tokens", "--report-to", "none",
            "--eval-steps", "100", "--save-steps", "2048" if no_eval else "100",
            "--save-total-limit", "1" if no_eval else "2"]
    if no_eval:
        cmd.append("--no-eval")
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--no-eval", action="store_true", help="Disable validation, as in the third archived run and finding-blind controls")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    cmd = command(args.model, args.data, args.output, args.seed, no_eval=args.no_eval)
    print(shlex.join(cmd), flush=True)
    if args.execute:
        for path in (args.model, args.data / "train.jsonl", args.data / "val.jsonl"):
            if not path.exists():
                parser.error("Explicit model/data input does not exist")
        env = dict(os.environ, USE_TF="0", TRANSFORMERS_NO_TF="1", TOKENIZERS_PARALLELISM="false")
        env.pop("FINDING_FIRST", None)
        subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
