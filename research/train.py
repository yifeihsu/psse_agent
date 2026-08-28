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
import os
import re
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Sequence

from psse_env.sft.collator import AssistantOnlyCollator
from psse_env.sft.gates import prepare_example

from .model import load_model_and_processor, lora_target_modules

DEFAULT_MAX_LENGTH = 8192
TRAINING_STEP_EXPOSURE_KEY = "_research_training_step_exposure"
RESTART_LEDGER_NAME = "research_exposure_checkpoint.json"
RESTART_COMPLETE_NAME = "research_checkpoint_complete.json"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    """Hash a JSON value using the study's stable canonical encoding."""

    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _checkpoint_step(path: Path) -> int:
    prefix = "checkpoint-"
    if not path.name.startswith(prefix):
        raise ValueError(f"not a Trainer checkpoint directory: {path}")
    try:
        step = int(path.name[len(prefix) :])
    except ValueError as exc:
        raise ValueError(f"invalid Trainer checkpoint suffix: {path}") from exc
    if step <= 0:
        raise ValueError(f"invalid Trainer checkpoint step: {path}")
    return step


def finalize_restart_checkpoint(
    checkpoint: str | Path,
    ledger: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically mark a full Trainer checkpoint complete after hashing it."""

    root = Path(checkpoint)
    if not root.is_dir():
        raise RuntimeError(f"Trainer checkpoint directory is missing: {root}")
    step = _checkpoint_step(root)
    if int(ledger.get("global_step") or -1) != step:
        raise RuntimeError("checkpoint suffix and research ledger step disagree")

    _atomic_write_json(root / RESTART_LEDGER_NAME, ledger)
    trainer_state_path = root / "trainer_state.json"
    if not trainer_state_path.is_file():
        raise RuntimeError("restart checkpoint lacks trainer_state.json")
    trainer_state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
    if int(trainer_state.get("global_step") or -1) != step:
        raise RuntimeError("checkpoint suffix and Trainer state step disagree")

    required = ("optimizer.pt", "scheduler.pt", "adapter_config.json")
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise RuntimeError(f"restart checkpoint lacks required files: {missing}")
    if not any((root / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")):
        raise RuntimeError("restart checkpoint lacks adapter weights")
    if not any(root.glob("rng_state*.pth")):
        raise RuntimeError("restart checkpoint lacks RNG state")

    files: dict[str, dict[str, Any]] = {}
    for item in sorted(root.rglob("*")):
        if not item.is_file() or item.name == RESTART_COMPLETE_NAME or ".tmp-" in item.name:
            continue
        relative = item.relative_to(root).as_posix()
        files[relative] = {"bytes": item.stat().st_size, "sha256": file_sha256(item)}
    manifest = {
        "schema": "research_checkpoint_complete_v1",
        "global_step": step,
        "run_binding_sha256": ledger.get("run_binding_sha256"),
        "files": files,
    }
    _atomic_write_json(root / RESTART_COMPLETE_NAME, manifest)
    return manifest


def validate_restart_checkpoint(
    checkpoint: str | Path,
    *,
    expected_run_binding_sha256: str | None = None,
) -> dict[str, Any]:
    """Verify a completed restart checkpoint and return its exposure ledger."""

    root = Path(checkpoint).resolve(strict=True)
    step = _checkpoint_step(root)
    ledger_path = root / RESTART_LEDGER_NAME
    complete_path = root / RESTART_COMPLETE_NAME
    if not ledger_path.is_file() or not complete_path.is_file():
        raise ValueError("resume checkpoint lacks a completed research manifest")
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    manifest = json.loads(complete_path.read_text(encoding="utf-8"))
    if (
        ledger.get("schema") != "research_exposure_checkpoint_v2"
        or manifest.get("schema") != "research_checkpoint_complete_v1"
        or int(ledger.get("global_step") or -1) != step
        or int(manifest.get("global_step") or -1) != step
        or ledger.get("run_binding_sha256") != manifest.get("run_binding_sha256")
    ):
        raise ValueError("resume checkpoint manifests disagree")
    if (
        expected_run_binding_sha256 is not None
        and ledger.get("run_binding_sha256") != expected_run_binding_sha256
    ):
        raise ValueError("resume checkpoint belongs to a different research run")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("resume checkpoint completion manifest has no files")
    required = {
        RESTART_LEDGER_NAME,
        "trainer_state.json",
        "optimizer.pt",
        "scheduler.pt",
        "adapter_config.json",
    }
    if not required.issubset(files):
        raise ValueError("resume checkpoint manifest omits required state")
    if not ({"adapter_model.safetensors", "adapter_model.bin"} & set(files)):
        raise ValueError("resume checkpoint manifest omits adapter weights")
    if not any(Path(relative).name.startswith("rng_state") for relative in files):
        raise ValueError("resume checkpoint manifest omits RNG state")
    for relative, evidence in files.items():
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError("resume checkpoint manifest escapes its directory") from exc
        if (
            not path.is_file()
            or not isinstance(evidence, dict)
            or path.stat().st_size != evidence.get("bytes")
            or file_sha256(path) != evidence.get("sha256")
        ):
            raise ValueError(f"resume checkpoint file verification failed: {relative}")
    trainer_state = json.loads((root / "trainer_state.json").read_text(encoding="utf-8"))
    if int(trainer_state.get("global_step") or -1) != step:
        raise ValueError("resume checkpoint Trainer state step disagrees")
    return ledger


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


def sampled_rows_after_updates(
    *,
    train_rows: int,
    updates: int,
    batch_size: int,
    gradient_accumulation_steps: int,
) -> int:
    """Return the exact rows consumed after a deterministic number of updates."""

    values = (train_rows, batch_size, gradient_accumulation_steps)
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in values
    ) or isinstance(updates, bool) or not isinstance(updates, int) or updates < 0:
        raise ValueError("sampled-row schedule inputs are invalid")
    microbatches_per_epoch = (train_rows + batch_size - 1) // batch_size
    updates_per_epoch = (
        microbatches_per_epoch + gradient_accumulation_steps - 1
    ) // gradient_accumulation_steps
    full_epochs, residual_updates = divmod(updates, updates_per_epoch)
    residual_microbatches = min(
        residual_updates * gradient_accumulation_steps,
        microbatches_per_epoch,
    )
    return full_epochs * train_rows + min(
        residual_microbatches * batch_size,
        train_rows,
    )


def sampled_batches_after_updates(
    *,
    train_rows: int,
    updates: int,
    batch_size: int,
    gradient_accumulation_steps: int,
) -> int:
    """Return microbatches that reach ``training_step`` after N updates."""

    values = (train_rows, batch_size, gradient_accumulation_steps)
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in values
    ) or isinstance(updates, bool) or not isinstance(updates, int) or updates < 0:
        raise ValueError("sampled-batch schedule inputs are invalid")
    microbatches_per_epoch = (train_rows + batch_size - 1) // batch_size
    updates_per_epoch = (
        microbatches_per_epoch + gradient_accumulation_steps - 1
    ) // gradient_accumulation_steps
    full_epochs, residual_updates = divmod(updates, updates_per_epoch)
    residual_microbatches = min(
        residual_updates * gradient_accumulation_steps,
        microbatches_per_epoch,
    )
    return full_epochs * microbatches_per_epoch + residual_microbatches


def validate_sampled_exposure(
    counts: Mapping[str, Any] | None,
    *,
    train_rows: int,
    updates: int,
    batch_size: int,
    gradient_accumulation_steps: int,
) -> None:
    """Fail if a persisted exposure ledger disagrees with its global step."""

    if not isinstance(counts, Mapping):
        raise ValueError("training-step exposure ledger is missing")
    expected_rows = sampled_rows_after_updates(
        train_rows=train_rows,
        updates=updates,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    expected_batches = sampled_batches_after_updates(
        train_rows=train_rows,
        updates=updates,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    if counts.get("rows") != expected_rows or counts.get("batches") != expected_batches:
        raise ValueError(
            "training-step exposure ledger disagrees with optimizer progress"
        )


class ExposureCountingCollator:
    """Pad rows while counting the exact examples and tokens actually collated."""

    def __init__(
        self,
        processor: Any,
        initial_counts: Mapping[str, Mapping[str, int]] | None = None,
    ) -> None:
        self.base = AssistantOnlyCollator(processor)
        self._counts: dict[str, dict[str, int]] = defaultdict(
            lambda: {
                "batches": 0,
                "rows": 0,
                "input_tokens": 0,
                "supervised_tokens": 0,
            }
        )
        for split, counts in (initial_counts or {}).items():
            expected = {"batches", "rows", "input_tokens", "supervised_tokens"}
            if set(counts) != expected or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in counts.values()
            ):
                raise ValueError("initial collator exposure counts are invalid")
            self._counts[str(split)].update(
                {key: int(value) for key, value in counts.items()}
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

    def __init__(self, initial_counts: Mapping[str, int] | None = None) -> None:
        self._counts = {
            "batches": 0,
            "rows": 0,
            "input_tokens": 0,
            "supervised_tokens": 0,
        }
        if initial_counts is not None:
            if set(initial_counts) != set(self._counts) or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in initial_counts.values()
            ):
                raise ValueError("initial training-step exposure counts are invalid")
            self._counts.update(
                {key: int(value) for key, value in initial_counts.items()}
            )

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
    restart_save_steps: int | None = None,
    restart_save_total_limit: int | None = None,
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
        "save_strategy": "steps" if restart_save_steps is not None else "epoch",
        "save_only_model": False,
        "ignore_data_skip": False,
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
    if restart_save_steps is not None:
        kwargs["save_steps"] = restart_save_steps
        kwargs["save_total_limit"] = restart_save_total_limit
    supported = inspect.signature(SFTConfig.__init__).parameters
    required_restart_fields = {
        "save_strategy",
        "save_steps",
        "save_total_limit",
        "save_only_model",
        "ignore_data_skip",
    }
    if restart_save_steps is not None and not required_restart_fields.issubset(
        supported
    ):
        missing = sorted(required_restart_fields - set(supported))
        raise RuntimeError(f"SFTConfig lacks required restart fields: {missing}")
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
    resume_from_checkpoint: str | Path | None = None,
    restart_save_steps: int | None = None,
    restart_save_total_limit: int | None = None,
    milestone_steps: Mapping[str, int] | None = None,
    milestone_output_dir: str | Path | None = None,
    pass_normalized_total_rows: int | None = None,
    restart_contract: Mapping[str, Any] | None = None,
    seed: int = 20260823,
    report_to: str = "none",
    run_name: str | None = None,
) -> dict[str, Any]:
    """Fine-tune a LoRA adapter and return a short training summary."""

    import inspect

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import TrainerCallback
    from trl import SFTTrainer

    if resume_from_checkpoint is not None and initial_adapter is not None:
        raise ValueError("resume_from_checkpoint and initial_adapter are mutually exclusive")
    if bool(restart_save_steps is not None) != bool(
        restart_save_total_limit is not None
    ):
        raise ValueError("restart save steps and total limit are required together")
    if restart_save_steps is not None and restart_save_steps <= 0:
        raise ValueError("restart_save_steps must be positive")
    if restart_save_total_limit is not None and restart_save_total_limit <= 0:
        raise ValueError("restart_save_total_limit must be positive")
    if pass_normalized_total_rows is not None and pass_normalized_total_rows <= 0:
        raise ValueError("pass_normalized_total_rows must be positive")
    try:
        restart_contract_value = json.loads(
            json.dumps(restart_contract or {}, allow_nan=False)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("restart_contract must be a JSON mapping") from exc
    if not isinstance(restart_contract_value, dict):
        raise ValueError("restart_contract must be a JSON mapping")
    milestone_plan = dict(milestone_steps or {})
    if bool(milestone_plan) != (milestone_output_dir is not None):
        raise ValueError("milestone steps and milestone output directory are required together")
    checked_milestone_steps: list[int] = []
    for label, step in milestone_plan.items():
        if (
            not isinstance(label, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", label) is None
            or isinstance(step, bool)
            or not isinstance(step, int)
            or step <= 0
            or (max_steps > 0 and step > max_steps)
        ):
            raise ValueError("milestone labels and steps are invalid")
        checked_milestone_steps.append(step)
    if len(set(checked_milestone_steps)) != len(checked_milestone_steps):
        raise ValueError("milestone labels and steps are invalid")

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
    if (
        pass_normalized_total_rows is not None
        and pass_normalized_total_rows != 2 * len(prepared_train)
    ):
        raise ValueError("pass-normalized schedule must target exactly two corpus passes")
    if pass_normalized_total_rows is not None:
        updates_per_epoch = (
            (len(prepared_train) + batch_size - 1) // batch_size
            + gradient_accumulation_steps
            - 1
        ) // gradient_accumulation_steps
        if max_steps != 2 * updates_per_epoch:
            raise ValueError(
                "pass-normalized schedule max_steps must end at two corpus passes"
            )

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

    run_binding = {
        "schema": "research_restart_run_binding_v1",
        "model_id": resolved_id,
        "model_revision": resolved_revision,
        "train_sha256": train_sha256_before,
        "validation_sha256": validation_sha256_before,
        "prepared_train_sha256": canonical_json_sha256(prepared_train),
        "prepared_validation_sha256": canonical_json_sha256(prepared_validation),
        "prepared_train_rows": len(prepared_train),
        "prepared_validation_rows": len(prepared_validation),
        "max_length": max_length,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "lora_target_modules": list(resolved_lora_targets or ()),
        "max_steps": max_steps,
        "pass_normalized_total_rows": pass_normalized_total_rows,
        "milestone_steps": dict(sorted(milestone_plan.items())),
        "restart_save_steps": restart_save_steps,
        "restart_save_total_limit": restart_save_total_limit,
        "ignore_data_skip": False,
        "seed": seed,
        "external_contract": restart_contract_value,
    }
    run_binding_sha256 = canonical_json_sha256(run_binding)
    resume_path = (
        Path(resume_from_checkpoint).resolve(strict=True)
        if resume_from_checkpoint is not None
        else None
    )
    resume_global_step = 0
    initial_training_step_exposure: Mapping[str, int] | None = None
    initial_collated_exposure: Mapping[str, Mapping[str, int]] | None = None
    if resume_path is not None:
        ledger = validate_restart_checkpoint(
            resume_path,
            expected_run_binding_sha256=run_binding_sha256,
        )
        resume_global_step = int(ledger["global_step"])
        initial_training_step_exposure = ledger.get("training_step_train_exposure")
        initial_collated_exposure = ledger.get("collated_exposure")
        validate_sampled_exposure(
            initial_training_step_exposure,
            train_rows=len(prepared_train),
            updates=resume_global_step,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )

    config = build_config(
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        seed=seed,
        has_validation=bool(prepared_validation),
        max_steps=max_steps,
        restart_save_steps=restart_save_steps,
        restart_save_total_limit=restart_save_total_limit,
        report_to=report_to,
        run_name=run_name or Path(output_dir).name,
    )

    # TRL requires an Arrow dataset rather than a list of records.  The rows
    # are already tokenized with the loss mask applied, and the collator only
    # pads, so wrapping them changes nothing about what is learned.
    from datasets import Dataset

    collator = ExposureCountingCollator(processor, initial_collated_exposure)
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

    training_step_counter = TrainingStepExposureCounter(
        initial_training_step_exposure
    )

    class ExposureCountingSFTTrainer(SFTTrainer):
        def create_scheduler(
            self, num_training_steps: int, optimizer: Any | None = None
        ) -> Any:
            if pass_normalized_total_rows is None:
                return super().create_scheduler(num_training_steps, optimizer)
            if self.lr_scheduler is None:
                from torch.optim.lr_scheduler import LambdaLR

                active_optimizer = optimizer if optimizer is not None else self.optimizer
                if active_optimizer is None:
                    raise RuntimeError("optimizer must exist before scheduler creation")

                def row_fraction(step: int) -> float:
                    sampled = sampled_rows_after_updates(
                        train_rows=len(prepared_train),
                        updates=int(step),
                        batch_size=batch_size,
                        gradient_accumulation_steps=gradient_accumulation_steps,
                    )
                    return max(
                        0.0,
                        1.0 - sampled / pass_normalized_total_rows,
                    )

                self.lr_scheduler = LambdaLR(active_optimizer, row_fraction)
                self._created_lr_scheduler = True
            return self.lr_scheduler

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

    milestone_root = Path(milestone_output_dir) if milestone_output_dir else None
    step_to_milestone = {step: label for label, step in milestone_plan.items()}

    class ResearchCheckpointCallback(TrainerCallback):
        """Persist restart exposure ledgers and adapter-only pass milestones."""

        @staticmethod
        def payload(global_step: int) -> dict[str, Any]:
            training_exposure = training_step_counter.summary()
            validate_sampled_exposure(
                training_exposure,
                train_rows=len(prepared_train),
                updates=global_step,
                batch_size=batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
            )
            return {
                "schema": "research_exposure_checkpoint_v2",
                "global_step": int(global_step),
                "run_binding": run_binding,
                "run_binding_sha256": run_binding_sha256,
                "training_step_train_exposure": training_exposure,
                "collated_exposure": {
                    "train": collator.summary("train"),
                    "validation": collator.summary("validation"),
                },
            }

        def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            payload = self.payload(int(state.global_step))
            finalize_restart_checkpoint(checkpoint, payload)

            label = step_to_milestone.get(int(state.global_step))
            if label is None or milestone_root is None:
                return control
            destination = milestone_root / label
            marker = destination / "research_milestone.json"
            if destination.exists():
                if not marker.is_file():
                    raise RuntimeError(f"incomplete milestone directory: {destination}")
                existing = json.loads(marker.read_text(encoding="utf-8"))
                if (
                    existing.get("global_step") != int(state.global_step)
                    or existing.get("run_binding_sha256") != run_binding_sha256
                ):
                    raise RuntimeError(f"milestone binding drift: {destination}")
                adapter_files = existing.get("adapter_files")
                if (
                    not isinstance(adapter_files, dict)
                    or "adapter_config.json" not in adapter_files
                    or not (
                        {"adapter_model.safetensors", "adapter_model.bin"}
                        & set(adapter_files)
                    )
                ):
                    raise RuntimeError(f"milestone lacks adapter manifest: {destination}")
                for relative, evidence in adapter_files.items():
                    item = (destination / relative).resolve()
                    try:
                        item.relative_to(destination.resolve())
                    except ValueError as exc:
                        raise RuntimeError(
                            f"milestone manifest escapes its directory: {destination}"
                        ) from exc
                    if (
                        not item.is_file()
                        or not isinstance(evidence, dict)
                        or item.stat().st_size != evidence.get("bytes")
                        or file_sha256(item) != evidence.get("sha256")
                    ):
                        raise RuntimeError(f"milestone adapter drift: {item}")
                restart_evidence = destination / "research_restart_checkpoint_complete.json"
                if (
                    not restart_evidence.is_file()
                    or file_sha256(restart_evidence)
                    != existing.get("restart_checkpoint_manifest_sha256")
                ):
                    raise RuntimeError(
                        f"milestone restart evidence drift: {destination}"
                    )
                return control
            milestone_root.mkdir(parents=True, exist_ok=True)
            temporary = Path(
                tempfile.mkdtemp(prefix=f".{label}.tmp-", dir=milestone_root)
            )
            model = kwargs.get("model")
            if model is None:
                raise RuntimeError("Trainer callback did not provide the model")
            model.save_pretrained(str(temporary))
            adapter_files: dict[str, dict[str, Any]] = {}
            for item in sorted(temporary.iterdir()):
                if item.is_file():
                    adapter_files[item.name] = {
                        "bytes": item.stat().st_size,
                        "sha256": file_sha256(item),
                    }
            if "adapter_config.json" not in adapter_files or not (
                {"adapter_model.safetensors", "adapter_model.bin"}
                & set(adapter_files)
            ):
                raise RuntimeError("milestone adapter save is incomplete")
            restart_manifest_source = checkpoint / RESTART_COMPLETE_NAME
            restart_evidence = (
                temporary / "research_restart_checkpoint_complete.json"
            )
            shutil.copyfile(restart_manifest_source, restart_evidence)
            restart_manifest = json.loads(
                restart_evidence.read_text(encoding="utf-8")
            )
            if (
                restart_manifest.get("schema")
                != "research_checkpoint_complete_v1"
                or restart_manifest.get("global_step") != int(state.global_step)
                or restart_manifest.get("run_binding_sha256")
                != run_binding_sha256
            ):
                raise RuntimeError("milestone restart manifest binding drift")
            milestone_payload = {
                **payload,
                "schema": "research_exposure_milestone_v1",
                "label": label,
                "adapter_files": adapter_files,
                "restart_checkpoint": str(checkpoint),
                "restart_checkpoint_manifest_sha256": file_sha256(
                    restart_evidence
                ),
                "restart_checkpoint_manifest_evidence": (
                    "research_restart_checkpoint_complete.json"
                ),
            }
            _atomic_write_json(
                temporary / "research_milestone.json", milestone_payload
            )
            temporary.rename(destination)
            return control

        def on_step_end(
            self, args: Any, state: Any, control: Any, **kwargs: Any
        ) -> Any:
            label = step_to_milestone.get(int(state.global_step))
            if label is not None and milestone_root is not None:
                # The adapter milestone is only published from ``on_save``, after
                # the full optimizer/scheduler/RNG checkpoint has completed.
                control.should_save = True
            return control

    trainer = ExposureCountingSFTTrainer(
        **{key: value for key, value in trainer_kwargs.items() if key in supported}
    )
    trainer.add_callback(ResearchCheckpointCallback())
    result = trainer.train(
        resume_from_checkpoint=str(resume_path) if resume_path is not None else None
    )
    training_loss = float(getattr(result, "training_loss", float("nan")))
    if not math.isfinite(training_loss):
        raise RuntimeError(f"training produced a non-finite loss: {training_loss!r}")
    final_global_step = int(getattr(trainer.state, "global_step", 0))
    final_training_step_exposure = training_step_counter.summary()
    validate_sampled_exposure(
        final_training_step_exposure,
        train_rows=len(prepared_train),
        updates=final_global_step,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )

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
        "training_step_train_exposure": final_training_step_exposure,
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
        "resume_from_checkpoint": str(resume_path) if resume_path is not None else None,
        "resume_global_step": resume_global_step,
        "restart_save_steps": restart_save_steps,
        "restart_save_total_limit": restart_save_total_limit,
        "milestone_steps": dict(sorted(milestone_plan.items())),
        "milestone_output_dir": str(milestone_root) if milestone_root else None,
        "lr_schedule": (
            "linear_sampled_rows_over_2N"
            if pass_normalized_total_rows is not None
            else "linear_optimizer_steps"
        ),
        "pass_normalized_total_rows": pass_normalized_total_rows,
        "restart_run_binding": run_binding,
        "restart_run_binding_sha256": run_binding_sha256,
        "sampled_rows_at_completion": sampled_rows_after_updates(
            train_rows=len(prepared_train),
            updates=final_global_step,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
        ),
        "optimizer_updates_completed": final_global_step,
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
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--restart-save-steps", type=int)
    parser.add_argument("--restart-save-total-limit", type=int)
    parser.add_argument(
        "--milestone-step",
        action="append",
        default=[],
        metavar="LABEL=STEP",
        help="Save an adapter-only milestone at this optimizer step; repeatable.",
    )
    parser.add_argument("--milestone-output-dir")
    parser.add_argument("--pass-normalized-total-rows", type=int)
    parser.add_argument(
        "--restart-contract",
        help="JSON object with the external arm/config/source contract bound to checkpoints.",
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

    milestone_steps: dict[str, int] = {}
    for item in args.milestone_step:
        try:
            label, raw_step = item.split("=", 1)
            step = int(raw_step)
        except (TypeError, ValueError) as exc:
            parser.error(f"invalid --milestone-step {item!r}: {exc}")
        if label in milestone_steps:
            parser.error(f"duplicate milestone label: {label}")
        milestone_steps[label] = step

    restart_contract: Mapping[str, Any] | None = None
    if args.restart_contract:
        try:
            restart_contract = json.loads(
                Path(args.restart_contract).read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            parser.error(f"invalid --restart-contract: {exc}")
        if not isinstance(restart_contract, dict):
            parser.error("--restart-contract must contain a JSON object")

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
        resume_from_checkpoint=args.resume_from_checkpoint,
        restart_save_steps=args.restart_save_steps,
        restart_save_total_limit=args.restart_save_total_limit,
        milestone_steps=milestone_steps,
        milestone_output_dir=args.milestone_output_dir,
        pass_normalized_total_rows=args.pass_normalized_total_rows,
        restart_contract=restart_contract,
        seed=args.seed,
        report_to=args.report_to,
        run_name=args.run_name,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
