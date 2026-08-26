"""LoRA supervised fine-tuning without the release receipt machinery.

``psse_env.sft`` performs the same training behind provenance validation,
study-manifest pinning, baseline admission and accelerator attestation.  This
module runs the training itself.

Rendering, tokenization and assistant-only masking are imported from
``psse_env.sft`` unchanged: those decide *what the model learns*, and a
divergent reimplementation would quietly invalidate any comparison with the
release path.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from psse_env.sft.collator import AssistantOnlyCollator
from psse_env.sft.gates import prepare_example

from .model import load_model_and_processor, lora_target_modules

DEFAULT_MAX_LENGTH = 8192


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def write_jsonl(path: str | Path, rows: Sequence[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def prepare_rows(
    rows: Sequence[dict[str, Any]],
    processor: Any,
    *,
    max_length: int,
    label: str = "row",
) -> tuple[list[dict[str, list[int]]], list[str]]:
    """Render and mask rows, reporting the ones that could not be prepared."""

    prepared: list[dict[str, list[int]]] = []
    failures: list[str] = []
    for index, row in enumerate(rows):
        try:
            example = prepare_example(
                row,
                processor,
                max_length=max_length,
                row_label=f"{label}[{index}]",
            )
        except Exception as exc:  # noqa: BLE001 - report, do not abort the run
            failures.append(f"{label}[{index}]: {type(exc).__name__}: {exc}")
            continue
        if example.target_truncated:
            failures.append(f"{label}[{index}]: supervised target was truncated")
            continue
        prepared.append(example.model_record())
    return prepared, failures


def build_config(
    *,
    output_dir: str | Path,
    epochs: float,
    batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    seed: int,
    has_validation: bool,
    report_to: str = "none",
    run_name: str | None = None,
) -> Any:
    import inspect

    from trl import SFTConfig

    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "num_train_epochs": epochs,
        "optim": "adamw_torch",
        "lr_scheduler_type": "linear",
        "logging_steps": 10,
        "eval_strategy": "epoch" if has_validation else "no",
        "save_strategy": "epoch",
        "seed": seed,
        "bf16": True,
        "fp16": False,
        # The rows arrive pretokenized with ``-100`` outside the assistant
        # target, so TRL must not re-derive either the packing or the loss mask.
        "packing": False,
        "completion_only_loss": False,
        "remove_unused_columns": False,
        "dataset_kwargs": {"skip_prepare_dataset": True},
        "max_length": None,
        "report_to": report_to,
        "run_name": run_name,
        "logging_dir": str(Path(output_dir) / "tb") if report_to == "tensorboard" else None,
    }
    supported = inspect.signature(SFTConfig.__init__).parameters
    return SFTConfig(
        **{key: value for key, value in kwargs.items() if key in supported}
    )


def train(
    *,
    train_path: str | Path,
    validation_path: str | Path | None,
    output_dir: str | Path,
    model_id: str | None = None,
    revision: str | None = None,
    initial_adapter: str | Path | None = None,
    max_length: int = DEFAULT_MAX_LENGTH,
    epochs: float = 2.0,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 1e-4,
    lora_rank: int = 16,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
    seed: int = 20260823,
    report_to: str = "none",
    run_name: str | None = None,
) -> dict[str, Any]:
    """Fine-tune a LoRA adapter and return a short training summary."""

    import inspect

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer

    model, processor, resolved_id, resolved_revision = load_model_and_processor(
        model_id=model_id,
        revision=revision,
        adapter_path=initial_adapter,
        load_in_4bit=True,
        for_training=True,
    )

    train_rows = read_jsonl(train_path)
    prepared_train, train_failures = prepare_rows(
        train_rows, processor, max_length=max_length, label="train"
    )
    if not prepared_train:
        raise RuntimeError("No training rows survived preparation.")

    prepared_validation: list[dict[str, list[int]]] = []
    validation_failures: list[str] = []
    if validation_path is not None:
        validation_rows = read_jsonl(validation_path)
        prepared_validation, validation_failures = prepare_rows(
            validation_rows, processor, max_length=max_length, label="validation"
        )

    if initial_adapter is None:
        model = prepare_model_for_kbit_training(model)
        # Training never reads the KV cache, and leaving it enabled costs a
        # large amount of memory per step.  Omitting this ran roughly five
        # times slower than the release path on the same data.
        config_object = getattr(model, "config", None)
        if config_object is not None and hasattr(config_object, "use_cache"):
            config_object.use_cache = False
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=lora_target_modules(model),
        )
        model = get_peft_model(model, lora_config)

    config = build_config(
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        seed=seed,
        has_validation=bool(prepared_validation),
        report_to=report_to,
        run_name=run_name or Path(output_dir).name,
    )

    # TRL requires an Arrow dataset rather than a list of records.  The rows
    # are already tokenized with the loss mask applied, and the collator only
    # pads, so wrapping them changes nothing about what is learned.
    from datasets import Dataset

    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": config,
        "train_dataset": Dataset.from_list(prepared_train),
        "data_collator": AssistantOnlyCollator(processor),
    }
    if prepared_validation:
        trainer_kwargs["eval_dataset"] = Dataset.from_list(prepared_validation)
    supported = inspect.signature(SFTTrainer.__init__).parameters
    if "processing_class" in supported:
        trainer_kwargs["processing_class"] = processor

    trainer = SFTTrainer(
        **{key: value for key, value in trainer_kwargs.items() if key in supported}
    )
    result = trainer.train()

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(destination))

    summary = {
        "model_id": resolved_id,
        "model_revision": resolved_revision,
        "initial_adapter": str(initial_adapter) if initial_adapter else None,
        "output_dir": str(destination),
        "train_rows": len(train_rows),
        "prepared_train_rows": len(prepared_train),
        "prepared_validation_rows": len(prepared_validation),
        "train_preparation_failures": train_failures[:20],
        "validation_preparation_failures": validation_failures[:20],
        "max_length": max_length,
        "epochs": epochs,
        "seed": seed,
        "training_loss": float(getattr(result, "training_loss", float("nan"))),
        "release_evidence": False,
    }
    (destination / "research_training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-id")
    parser.add_argument("--revision")
    parser.add_argument("--initial-adapter")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument(
        "--report-to",
        choices=("none", "wandb", "tensorboard"),
        default="none",
        help=(
            "Metric streaming. wandb needs a prior `wandb login` (use "
            "WANDB_MODE=offline on machines without internet, then `wandb "
            "sync`); tensorboard writes to <output-dir>/tb with no account."
        ),
    )
    parser.add_argument("--run-name", help="Run label in the tracker UI.")
    args = parser.parse_args(argv)

    summary = train(
        train_path=args.train,
        validation_path=args.validation,
        output_dir=args.output_dir,
        model_id=args.model_id,
        revision=args.revision,
        initial_adapter=args.initial_adapter,
        max_length=args.max_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        seed=args.seed,
        report_to=args.report_to,
        run_name=args.run_name,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
