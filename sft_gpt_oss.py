#!/usr/bin/env python3
"""Compatibility entrypoint for power-system SFT.

This wrapper preserves the older `sft_gpt_oss.py` CLI while delegating all
training behavior to `gpt_oss_power_sft_revised.py`, which is the canonical
Gemma 4 training path.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path


DEFAULT_TRAIN_FILE = "/scratch/yx3882/psse_agent/out_traces_balanced/sft_traces.train.jsonl"
DEFAULT_VAL_FILE = "/scratch/yx3882/psse_agent/out_traces_balanced/sft_traces.valid.jsonl"
DEFAULT_TEST_FILE = "/scratch/yx3882/psse_agent/out_traces_balanced/sft_traces.test.jsonl"
DEFAULT_MODEL_NAME = "unsloth/Gemma-4-26B-A4B-it"
DEFAULT_OUTPUT_DIR = "/scratch/yx3882/psse_agent/outputs/gemma4_power_agent"
DEFAULT_MAX_SEQ_LENGTH = 4096


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compatibility wrapper for power-system SFT"
    )
    parser.add_argument("--train-file", default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--val-file", default=DEFAULT_VAL_FILE)
    parser.add_argument("--test-file", default=DEFAULT_TEST_FILE)
    parser.add_argument("--include-test-in-train", action="store_true", default=False)

    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH)
    parser.add_argument("--load-in-4bit", action="store_true", default=False)
    parser.add_argument("--no-load-in-4bit", dest="load_in_4bit", action="store_false")
    parser.add_argument("--load-in-16bit", action="store_true", default=True)
    parser.add_argument("--no-load-in-16bit", dest="load_in_16bit", action="store_false")

    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)

    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--dataset-num-proc", type=int, default=4)
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--run-name", default="")

    parser.add_argument("--drop-too-long-targets", action="store_true", default=True)
    parser.add_argument("--keep-too-long-targets", dest="drop_too_long_targets", action="store_false")
    parser.add_argument("--include-tool-schemas", action="store_true", default=True)
    parser.add_argument("--no-include-tool-schemas", dest="include_tool_schemas", action="store_false")
    parser.add_argument("--tools-file", default="")
    return parser.parse_args(argv)


def resolve_existing_path(path_str: str) -> Path | None:
    candidates = [
        Path(path_str),
        Path.cwd() / path_str,
        Path(__file__).resolve().parent / path_str,
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    return None


def merge_train_and_test(train_file: str, test_file: str) -> str | None:
    train_path = resolve_existing_path(train_file)
    test_path = resolve_existing_path(test_file)
    if train_path is None:
        return None
    if test_path is None:
        print(f"[warn] --include-test-in-train set but test file not found: {test_file!r}")
        return str(train_path)

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".jsonl",
        prefix="gpt_oss_train_plus_test_",
        delete=False,
    ) as handle:
        with train_path.open("r", encoding="utf-8") as src:
            shutil.copyfileobj(src, handle)
        with test_path.open("r", encoding="utf-8") as src:
            shutil.copyfileobj(src, handle)
        temp_path = handle.name

    print(f"[info] merged train + test into temporary file: {temp_path}")
    return temp_path


def build_forward_argv(args: argparse.Namespace, train_file: str) -> list[str]:
    forwarded = [
        "--train-file", train_file,
        "--val-file", args.val_file,
        "--model-name", args.model_name,
        "--output-dir", args.output_dir,
        "--max-seq-length", str(args.max_seq_length),
        "--dataset-num-proc", str(args.dataset_num_proc),
        "--lora-r", str(args.lora_r),
        "--lora-alpha", str(args.lora_alpha),
        "--lora-dropout", str(args.lora_dropout),
        "--per-device-train-batch-size", str(args.per_device_train_batch_size),
        "--per-device-eval-batch-size", str(args.per_device_eval_batch_size),
        "--gradient-accumulation-steps", str(args.gradient_accumulation_steps),
        "--warmup-steps", str(args.warmup_steps),
        "--num-train-epochs", str(args.num_train_epochs),
        "--max-steps", str(args.max_steps),
        "--learning-rate", str(args.learning_rate),
        "--weight-decay", str(args.weight_decay),
        "--lr-scheduler-type", args.lr_scheduler_type,
        "--logging-steps", str(args.logging_steps),
        "--save-steps", str(args.save_steps),
        "--eval-steps", str(args.eval_steps),
        "--save-total-limit", str(args.save_total_limit),
        "--seed", str(args.seed),
        "--report-to", args.report_to,
    ]

    if args.run_name:
        forwarded.extend(["--run-name", args.run_name])
    if args.load_in_4bit:
        forwarded.append("--load-in-4bit")
    else:
        forwarded.append("--no-load-in-4bit")
    if args.load_in_16bit:
        forwarded.append("--load-in-16bit")
    else:
        forwarded.append("--no-load-in-16bit")
    if args.drop_too_long_targets:
        forwarded.append("--drop-too-long-targets")
    else:
        forwarded.append("--keep-too-long-targets")
    if args.include_tool_schemas:
        forwarded.append("--include-tool-schemas")
    else:
        forwarded.append("--no-include-tool-schemas")
    if args.tools_file:
        forwarded.extend(["--tools-file", args.tools_file])
    return forwarded


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    from gpt_oss_power_sft_revised import main as revised_main

    merged_train_file: str | None = None
    try:
        train_file = args.train_file
        if args.include_test_in_train:
            merged_train_file = merge_train_and_test(args.train_file, args.test_file)
            if merged_train_file is not None:
                train_file = merged_train_file

        revised_main(build_forward_argv(args, train_file))
    finally:
        if merged_train_file is not None:
            Path(merged_train_file).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
