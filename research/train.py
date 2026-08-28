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
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Sequence

from psse_env.sft.collator import AssistantOnlyCollator
from psse_env.sft.gates import prepare_example

from .model import load_model_and_processor, lora_target_modules

DEFAULT_MAX_LENGTH = 8192
TRAINING_STEP_EXPOSURE_KEY = "_research_training_step_exposure"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def summarize_prepared_rows(
    rows: Sequence[dict[str, list[int]]],
) -> dict[str, int | float | None]:
    """Summarize the assistant-target tokens available in a prepared corpus.

    Prompt tokens carry the ``-100`` ignore label, so counting the remaining
    labels measures the supervised target-token corpus rather than total prompt
    length. This is a corpus statistic; actual sampled exposure is recorded by
    :class:`ExposureCountingCollator` during training.
    """

    per_row = [
        sum(1 for token in row.get("labels", []) if token != -100)
        for row in rows
    ]
    total = sum(per_row)
    return {
        "rows": len(per_row),
        "supervised_tokens": total,
        "mean_supervised_tokens_per_row": (
            total / len(per_row) if per_row else None
        ),
        "min_supervised_tokens_per_row": min(per_row) if per_row else None,
        "max_supervised_tokens_per_row": max(per_row) if per_row else None,
    }


class ExposureCountingCollator:
    """Pad rows while counting the exact examples and tokens actually collated."""

    def __init__(self, processor: Any) -> None:
        self.base = AssistantOnlyCollator(processor)
        self._counts: dict[str, dict[str, int]] = defaultdict(
            lambda: {
                "batches": 0,
                "rows": 0,
                "input_tokens": 0,
                "supervised_tokens": 0,
            }
        )

    def __call__(self, features: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        batch_exposure = {
            "batches": 1,
            "rows": len(features),
            "input_tokens": 0,
            "supervised_tokens": 0,
        }
        for feature in features:
            split = str(feature.get("_exposure_split") or "unknown")
            labels = list(feature.get("labels") or [])
            attention = list(feature.get("attention_mask") or [])
            counts = self._counts[split]
            counts["rows"] += 1
            input_tokens = sum(int(value) != 0 for value in attention)
            supervised_tokens = sum(token != -100 for token in labels)
            counts["input_tokens"] += input_tokens
            counts["supervised_tokens"] += supervised_tokens
            batch_exposure["input_tokens"] += input_tokens
            batch_exposure["supervised_tokens"] += supervised_tokens
        splits = {
            str(feature.get("_exposure_split") or "unknown")
            for feature in features
        }
        for split in splits:
            self._counts[split]["batches"] += 1
        batch = self.base(features)
        if splits == {"train"}:
            batch[TRAINING_STEP_EXPOSURE_KEY] = batch_exposure
        return batch

    def summary(self, split: str) -> dict[str, int]:
        return dict(
            self._counts.get(
                split,
                {
                    "batches": 0,
                    "rows": 0,
                    "input_tokens": 0,
                    "supervised_tokens": 0,
                },
            )
        )


def pop_training_step_exposure(inputs: dict[str, Any]) -> dict[str, int]:
    """Remove and validate CPU-counted exposure before model forwarding."""

    exposure = inputs.pop(TRAINING_STEP_EXPOSURE_KEY, None)
    if not isinstance(exposure, Mapping):
        raise RuntimeError("training-step exposure metadata is missing")
    expected_keys = {"batches", "rows", "input_tokens", "supervised_tokens"}
    if set(exposure) != expected_keys or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in exposure.values()
    ):
        raise RuntimeError("training-step exposure metadata is invalid")
    if exposure["batches"] != 1 or exposure["rows"] <= 0:
        raise RuntimeError("training-step exposure batch is empty")
    labels = inputs.get("labels")
    label_shape = getattr(labels, "shape", ())
    if not label_shape or int(label_shape[0]) != exposure["rows"]:
        raise RuntimeError("training-step exposure row count disagrees with labels")
    return {key: int(exposure[key]) for key in expected_keys}


class TrainingStepExposureCounter:
    """Accumulate only batches whose ``training_step`` completed successfully."""

    def __init__(self) -> None:
        self._counts = {
            "batches": 0,
            "rows": 0,
            "input_tokens": 0,
            "supervised_tokens": 0,
        }

    def add(self, exposure: Mapping[str, int]) -> None:
        for key in self._counts:
            self._counts[key] += int(exposure[key])

    def summary(self) -> dict[str, int]:
        return dict(self._counts)


def build_config(
    *,
    output_dir: str | Path,
    epochs: float,
    batch_size: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    seed: int,
    has_validation: bool,
    max_steps: int = -1,
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
        "max_steps": max_steps,
        "optim": "adamw_torch",
        "lr_scheduler_type": "linear",
        "logging_steps": 10,
        "dataloader_num_workers": 0,
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
    max_steps: int = -1,
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

    train_sha256_before = file_sha256(train_path)
    validation_sha256_before = (
        file_sha256(validation_path) if validation_path else None
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

    resolved_lora_targets: tuple[str, ...] | None = None
    if initial_adapter is None:
        model = prepare_model_for_kbit_training(model)
        # Training never reads the KV cache, and leaving it enabled costs a
        # large amount of memory per step.  Omitting this ran roughly five
        # times slower than the release path on the same data.
        config_object = getattr(model, "config", None)
        if config_object is not None and hasattr(config_object, "use_cache"):
            config_object.use_cache = False
        resolved_lora_targets = lora_target_modules(model)
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=resolved_lora_targets,
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
        max_steps=max_steps,
        report_to=report_to,
        run_name=run_name or Path(output_dir).name,
    )

    # TRL requires an Arrow dataset rather than a list of records.  The rows
    # are already tokenized with the loss mask applied, and the collator only
    # pads, so wrapping them changes nothing about what is learned.
    from datasets import Dataset

    collator = ExposureCountingCollator(processor)
    prepared_train_dataset = [
        {**row, "_exposure_split": "train"} for row in prepared_train
    ]
    prepared_validation_dataset = [
        {**row, "_exposure_split": "validation"}
        for row in prepared_validation
    ]
    trainer_kwargs: dict[str, Any] = {
        "model": model,
        "args": config,
        "train_dataset": Dataset.from_list(prepared_train_dataset),
        "data_collator": collator,
    }
    if prepared_validation:
        trainer_kwargs["eval_dataset"] = Dataset.from_list(
            prepared_validation_dataset
        )
    supported = inspect.signature(SFTTrainer.__init__).parameters
    if "processing_class" in supported:
        trainer_kwargs["processing_class"] = processor

    training_step_counter = TrainingStepExposureCounter()

    class ExposureCountingSFTTrainer(SFTTrainer):
        def training_step(
            self,
            model: Any,
            inputs: dict[str, Any],
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            exposure = pop_training_step_exposure(inputs)
            result = super().training_step(model, inputs, *args, **kwargs)
            training_step_counter.add(exposure)
            return result

    trainer = ExposureCountingSFTTrainer(
        **{key: value for key, value in trainer_kwargs.items() if key in supported}
    )
    result = trainer.train()
    training_loss = float(getattr(result, "training_loss", float("nan")))
    if not math.isfinite(training_loss):
        raise RuntimeError(f"training produced a non-finite loss: {training_loss!r}")

    train_sha256_after = file_sha256(train_path)
    validation_sha256_after = (
        file_sha256(validation_path) if validation_path else None
    )
    if train_sha256_after != train_sha256_before:
        raise RuntimeError("training data changed while the run was in progress")
    if validation_sha256_after != validation_sha256_before:
        raise RuntimeError("validation data changed while the run was in progress")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(destination))

    summary = {
        "schema": "research_training_summary_v2",
        "model_id": resolved_id,
        "model_revision": resolved_revision,
        "initial_adapter": str(initial_adapter) if initial_adapter else None,
        "output_dir": str(destination),
        "train_path": str(train_path),
        "train_sha256": train_sha256_before,
        "validation_path": str(validation_path) if validation_path else None,
        "validation_sha256": validation_sha256_before,
        "train_rows": len(train_rows),
        "prepared_train_rows": len(prepared_train),
        "prepared_validation_rows": len(prepared_validation),
        "train_target_token_corpus_summary": summarize_prepared_rows(
            prepared_train
        ),
        "validation_target_token_corpus_summary": summarize_prepared_rows(
            prepared_validation
        ),
        "training_step_train_exposure": training_step_counter.summary(),
        "training_step_exposure_scope": "local_process_training_step",
        "collated_train_exposure": collator.summary("train"),
        "collated_validation_exposure": collator.summary("validation"),
        "collated_exposure_scope": "local_process_collator_with_lookahead",
        "world_size": int(getattr(trainer.args, "world_size", 1)),
        "train_preparation_failures": train_failures[:20],
        "validation_preparation_failures": validation_failures[:20],
        "max_length": max_length,
        "epochs": epochs,
        "batch_size": batch_size,
        "actual_per_device_train_batch_size": int(
            getattr(trainer.args, "per_device_train_batch_size", -1)
        ),
        "auto_find_batch_size": bool(
            getattr(trainer.args, "auto_find_batch_size", False)
        ),
        "dataloader_num_workers": int(
            getattr(trainer.args, "dataloader_num_workers", -1)
        ),
        "packing": bool(getattr(trainer.args, "packing", False)),
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "lora_target_modules": list(resolved_lora_targets or ()),
        "quantization": {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_compute_dtype": "bfloat16",
        },
        "max_steps": max_steps,
        "optimizer_updates_completed": int(getattr(trainer.state, "global_step", 0)),
        "trainer_epoch": getattr(trainer.state, "epoch", None),
        "seed": seed,
        "training_loss": training_loss,
        "release_evidence": False,
    }
    (destination / "research_training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
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
        "--max-steps",
        type=int,
        default=-1,
        help="Fixed optimizer-update budget; -1 trains by epochs. Use to hold "
        "update count constant across arms with different data sizes.",
    )
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
        max_steps=args.max_steps,
        seed=args.seed,
        report_to=args.report_to,
        run_name=args.run_name,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
