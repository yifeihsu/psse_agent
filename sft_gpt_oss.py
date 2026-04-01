#!/usr/bin/env python3
"""SFT fine-tuning for gpt-oss-20b using preprocessed power-system traces.

Usage:
    # After running preprocess.py which produces data/split_*.jsonl:
    python sft_gpt_oss.py

    # With WandB logging:
    python sft_gpt_oss.py --report-to wandb

    # Merge test split into training data:
    python sft_gpt_oss.py --include-test-in-train

    # Custom paths / hyperparams:
    python sft_gpt_oss.py --train-file data/split_train.jsonl \
                           --val-file data/split_val.jsonl   \
                           --test-file data/split_test.jsonl \
                           --max-seq-length 16384             \
                           --num-train-epochs 2
"""

from __future__ import annotations

import unsloth  # must be first to apply optimizations before other imports

import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_TRAIN_FILE = "/home/hk4488/psse_agent/data/split_train.jsonl"
DEFAULT_VAL_FILE   = "/home/hk4488/psse_agent/data/split_val.jsonl"
DEFAULT_TEST_FILE  = "/home/hk4488/psse_agent/data/split_test.jsonl"
DEFAULT_MODEL_NAME = "unsloth/gpt-oss-20b-unsloth-bnb-4bit"
DEFAULT_OUTPUT_DIR = "outputs/gpt_oss_power_sft"
DEFAULT_MAX_SEQ_LENGTH = 16384

# GPT-OSS chat template boundary tokens used by train_on_responses_only
GPT_OSS_INSTRUCTION_PART = "<|start|>user<|message|>"
GPT_OSS_RESPONSE_PART = "<|start|>assistant<|message|>"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SFT fine-tune gpt-oss-20b on preprocessed PSSE power-system traces"
    )
    # Data
    parser.add_argument("--train-file", default=DEFAULT_TRAIN_FILE,
                        help="Training JSONL from preprocess.py (default: %(default)s)")
    parser.add_argument("--val-file", default=DEFAULT_VAL_FILE,
                        help="Validation JSONL from preprocess.py (default: %(default)s)")
    parser.add_argument("--test-file", default=DEFAULT_TEST_FILE,
                        help="Test JSONL from preprocess.py (default: %(default)s)")
    parser.add_argument("--include-test-in-train", action="store_true", default=False,
                        help="Concatenate test split into training data before fine-tuning")
    # Model
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME,
                        help="Unsloth model ID or local path (default: %(default)s)")
    parser.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH,
                        help="Maximum sequence length (default: %(default)s)")
    parser.add_argument("--load-in-4bit", action="store_true", default=True,
                        help="Use 4-bit quantisation (default: True)")
    parser.add_argument("--no-load-in-4bit", dest="load_in_4bit", action="store_false")
    # LoRA
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank (default: %(default)s)")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha (default: %(default)s)")
    parser.add_argument("--lora-dropout", type=float, default=0.0,
                        help="LoRA dropout (default: %(default)s)")
    # Training
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help="Output directory for checkpoints (default: %(default)s)")
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1,
                        help="Overrides epochs when > 0 (default: disabled)")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--dataset-num-proc", type=int, default=4)
    parser.add_argument("--report-to", default="none",
                        help="Logging backend: 'wandb', 'tensorboard', or 'none' (default: %(default)s)")
    parser.add_argument("--run-name", default="",
                        help="WandB / experiment run name (optional)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def resolve_path(path_str: str, label: str) -> str | None:
    """Resolve a dataset file path, searching common locations."""
    candidates = [
        Path(path_str),
        Path.cwd() / path_str,
        Path(__file__).resolve().parent / path_str,
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.stat().st_size > 0:
            return str(candidate)
    print(f"[warn] {label} file not found: {path_str!r}")
    return None


def formatting_prompts_func(examples, tokenizer):
    """Apply gpt-oss chat template to a batch of conversations."""
    texts = []
    for conv in examples["messages"]:
        try:
            # Arrow schema-unification adds None for keys absent in some samples
            # (e.g. tool_calls=None on assistant messages that didn't call tools).
            # The GPT-OSS Jinja template does `if "tool_calls" in message` (key check)
            # then immediately indexes `tool_calls[0]`, crashing when the value is None.
            # Strip all None-valued keys before rendering to avoid this.
            clean_conv = [{k: v for k, v in m.items() if v is not None} for m in conv]
            texts.append(tokenizer.apply_chat_template(clean_conv, tokenize=False, add_generation_prompt=False))
        except Exception as e:
            import traceback as _tb
            print(f"\n[ERROR] apply_chat_template failed: {e}")
            print(f"  Template lines: {len(getattr(tokenizer, 'chat_template', '') .split(chr(10)))}")
            print(f"  Conversation roles: {[m.get('role') for m in clean_conv]}")
            for m in clean_conv:
                if m.get('role') == 'assistant':
                    print(f"  assistant msg keys: {list(m.keys())}, tool_calls={m.get('tool_calls')}")
            print(_tb.format_exc())
            raise
    return {"text": texts}


def load_split(path: str, tokenizer, num_proc: int) -> Dataset | None:
    """Load a JSONL split, apply chat template, return HF Dataset."""
    resolved = resolve_path(path, path)
    if resolved is None:
        return None

    raw = load_dataset("json", data_files={"split": resolved}, split="split")
    if len(raw) == 0:
        print(f"[warn] dataset at {path!r} is empty")
        return None

    dataset = raw.map(
        lambda ex: formatting_prompts_func(ex, tokenizer),
        batched=True,
        num_proc=1,  # keep 1 — tokenizer is not picklable across workers
        desc=f"Formatting {Path(path).name}",
    )
    return dataset


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # 1. Load model + tokenizer via Unsloth
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Loading model: {args.model_name}")
    print(f"{'='*60}\n")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        dtype=None,                          # auto-detect bf16 / fp16
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        full_finetuning=False,
    )

    # ------------------------------------------------------------------
    # 2. Attach LoRA adapters
    # ------------------------------------------------------------------
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
        use_rslora=False,
        loftq_config=None,
    )

    # ------------------------------------------------------------------
    # 3. Load preprocessed splits (output of preprocess.py)
    # ------------------------------------------------------------------
    print("Loading preprocessed datasets …")
    tmpl = getattr(tokenizer, "chat_template", "") or ""
    print(f"[DIAG] tokenizer.chat_template: {len(tmpl.split(chr(10)))} lines, hash={hash(tmpl)}")
    train_dataset = load_split(args.train_file, tokenizer, args.dataset_num_proc)
    eval_dataset  = load_split(args.val_file,   tokenizer, args.dataset_num_proc)
    test_dataset  = load_split(args.test_file,  tokenizer, args.dataset_num_proc)

    if train_dataset is None:
        raise FileNotFoundError(
            f"Training file not found: {args.train_file!r}\n"
            "Run preprocess.py first to generate the split files."
        )

    # Optionally merge test split into training data
    if args.include_test_in_train:
        if test_dataset is not None:
            from datasets import concatenate_datasets
            train_dataset = concatenate_datasets([train_dataset, test_dataset])
            print(f"  [info] test split merged into train")
        else:
            print(f"  [warn] --include-test-in-train set but test file not found, skipping")

    print(f"  train : {len(train_dataset):>6} samples  ({args.train_file}"
          + (" + test" if args.include_test_in_train and test_dataset else "") + ")")
    if eval_dataset is not None:
        print(f"  val   : {len(eval_dataset):>6} samples  ({args.val_file})")
    else:
        print("  val   : skipped (file not found or empty)")
    if test_dataset is not None and not args.include_test_in_train:
        print(f"  test  : {len(test_dataset):>6} samples  ({args.test_file})  [held out]")

    # Quick preview
    print("\n--- First training sample (first 800 chars) ---")
    print(train_dataset[0]["text"][:800])
    print("---\n")

    # ------------------------------------------------------------------
    # 4. Build SFTTrainer
    # ------------------------------------------------------------------
    eval_strategy = "steps" if eval_dataset is not None else "no"
    eval_steps_arg = args.eval_steps if eval_dataset is not None else None

    run_name = args.run_name or f"gpt-oss-power-sft-r{args.lora_r}"

    training_args = SFTConfig(
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        eval_strategy=eval_strategy,
        eval_steps=eval_steps_arg,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        optim="adamw_8bit",
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        seed=args.seed,
        output_dir=args.output_dir,
        report_to=args.report_to,
        run_name=run_name,
        dataset_text_field="text",
        dataset_num_proc=args.dataset_num_proc,
        max_length=args.max_seq_length,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
    )

    # ------------------------------------------------------------------
    # 5. Mask user/system tokens — train only on assistant responses
    #    Uses gpt-oss Harmony format boundary tokens
    # ------------------------------------------------------------------
    trainer = train_on_responses_only(
        trainer,
        instruction_part=GPT_OSS_INSTRUCTION_PART,
        response_part=GPT_OSS_RESPONSE_PART,
    )

    # Sanity-check masking on first sample
    sample_ids = trainer.train_dataset[0]["input_ids"]
    sample_labels = trainer.train_dataset[0]["labels"]
    unmasked_tokens = sum(1 for lbl in sample_labels if lbl != -100)
    print(f"[check] sample 0: {len(sample_ids)} tokens, {unmasked_tokens} unmasked (assistant) tokens")
    if unmasked_tokens == 0:
        print("[warn]  No unmasked tokens — check that GPT-OSS boundary tokens match your tokenizer.")

    # ------------------------------------------------------------------
    # 6. GPU memory snapshot before training
    # ------------------------------------------------------------------
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        reserved_gb = round(torch.cuda.max_memory_reserved() / 1024 ** 3, 2)
        total_gb = round(gpu.total_memory / 1024 ** 3, 2)
        print(f"\nGPU: {gpu.name}  |  Reserved: {reserved_gb} GB / {total_gb} GB total\n")

    # ------------------------------------------------------------------
    # 7. Train
    # ------------------------------------------------------------------
    print(f"{'='*60}")
    print("Starting training …")
    print(f"{'='*60}\n")
    trainer_stats = trainer.train()
    print(f"\nTraining metrics: {trainer_stats.metrics}")

    # ------------------------------------------------------------------
    # 8. Save LoRA adapter + tokenizer
    # ------------------------------------------------------------------
    save_dir = Path(args.output_dir) / "lora"
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))
    print(f"\nSaved LoRA adapter → {save_dir}")

    # GPU memory snapshot after training
    if torch.cuda.is_available():
        used_gb = round(torch.cuda.max_memory_reserved() / 1024 ** 3, 2)
        print(f"Peak GPU memory during training: {used_gb} GB")

    print(f"\n{'='*60}")
    print("Done. Fine-tuning complete.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
