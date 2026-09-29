"""LoRA and TRL training configuration used by the research trainer (optional dependencies)."""

from __future__ import annotations

import inspect
import random
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from .gates import (
    TOKEN_TYPE_INPUT_NAMES,
    GateError,
    PreparedExample,
    processor_token_type_input_names,
)


LORA_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
MAX_TRAINING_SEED = (2**32) - 1
TRAINING_RNG_ENGINES = (
    "python_random",
    "numpy_random",
    "torch_cpu",
    "torch_cuda_all",
)
ROUND1_STUDY_VARIANT_TO_VIEW = {
    "natural_dagger": "natural-only",
    "natural_dagger_probes": "full",
}


def validate_training_seed(seed: Any) -> int:
    """Return one deterministic trainer seed in NumPy's supported domain."""

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise GateError("seed must be an integer between 0 and 4294967295.")
    if seed < 0 or seed > MAX_TRAINING_SEED:
        raise GateError("seed must be an integer between 0 and 4294967295.")
    return seed


@dataclass(frozen=True)
class LoraSettings:
    rank: int = 16
    alpha: int = 16
    dropout: float = 0.0
    target_modules: tuple[str, ...] = LORA_TARGET_MODULES

    def kwargs(self) -> dict[str, Any]:
        return {
            "r": self.rank,
            "lora_alpha": self.alpha,
            "lora_dropout": self.dropout,
            "target_modules": list(self.target_modules),
            "bias": "none",
            "task_type": "CAUSAL_LM",
        }


@dataclass(frozen=True)
class TrainerSettings:
    model_name: str = "unsloth/gemma-4-31B-it"
    revision: str = ""
    output_dir: str = "outputs/dagger_gemma4_pilot"
    max_length: int = 4096
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    epochs: float = 1.0
    max_steps: int = -1
    optimizer: str = "adamw_torch"
    lr_scheduler_type: str = "linear"
    logging_steps: int = 1
    save_steps: int = 25
    eval_steps: int = 25
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    seed: int = 3407
    bf16: bool = True
    fp16: bool = False
    load_in_4bit: bool = False
    local_files_only: bool = True
    trust_remote_code: bool = False
    allow_prompt_truncation: bool = False
    allow_nonrelease_artifacts: bool = False
    required_processor_loader: str | None = None
    report_to: str = "none"
    run_name: str | None = None
    initial_adapter_path: str | None = None
    initial_adapter_revision: str | None = None
    parent_checkpoint_receipt_path: str | None = None
    round1_provenance_path: str | None = None
    round1_preflight_path: str | None = None
    reviewed_source_commit: str | None = None
    round1_view: str | None = None
    study_variant: str | None = None
    study_manifest_path: str | None = None

    def validate(self) -> None:
        if "gemma-4" not in self.model_name.lower():
            raise GateError(f"TrainerSettings requires a Gemma 4 model id, got {self.model_name!r}.")
        if re.fullmatch(r"[0-9a-fA-F]{40}", self.revision) is None:
            raise GateError("TrainerSettings.revision must be a pinned 40-character commit hash.")
        if (
            self.max_length <= 0
            or self.batch_size <= 0
            or self.gradient_accumulation_steps <= 0
        ):
            raise GateError(
                "max_length, batch_size, and gradient_accumulation_steps must be positive."
            )
        if self.learning_rate <= 0:
            raise GateError("learning_rate must be positive.")
        if self.optimizer != "adamw_torch":
            raise GateError("optimizer must be the pinned value 'adamw_torch'.")
        if self.lr_scheduler_type != "linear":
            raise GateError("lr_scheduler_type must be the pinned value 'linear'.")
        validate_training_seed(self.seed)
        if self.bf16 and self.fp16:
            raise GateError("bf16 and fp16 cannot both be enabled.")
        if self.eval_strategy not in {"epoch", "steps"}:
            raise GateError("eval_strategy must be 'epoch' or 'steps'.")
        if self.save_strategy not in {"epoch", "steps"}:
            raise GateError("save_strategy must be 'epoch' or 'steps'.")
        if self.eval_strategy == "steps" and self.eval_steps <= 0:
            raise GateError("eval_steps must be positive for step-based validation.")
        if self.required_processor_loader not in {None, "AutoProcessor"}:
            raise GateError(
                "required_processor_loader must be None or 'AutoProcessor'."
            )
        if not isinstance(self.report_to, str) or self.report_to not in {
            "none",
            "wandb",
        }:
            raise GateError("report_to must be 'none' or 'wandb'.")
        if self.run_name is not None and (
            not isinstance(self.run_name, str) or not self.run_name.strip()
        ):
            raise GateError("run_name must be None or a non-empty string.")
        initial_path = str(self.initial_adapter_path or "").strip()
        initial_revision = str(self.initial_adapter_revision or "").strip()
        parent_receipt_path = str(
            self.parent_checkpoint_receipt_path or ""
        ).strip()
        if bool(initial_path) != bool(initial_revision):
            raise GateError(
                "initial_adapter_path and initial_adapter_revision must be supplied together."
            )
        if initial_path:
            if not Path(initial_path).expanduser().is_absolute():
                raise GateError("initial_adapter_path must be an absolute path.")
            if re.fullmatch(r"[0-9a-fA-F]{64}", initial_revision) is None:
                raise GateError(
                    "initial_adapter_revision must be a 64-hex checkpoint tree SHA-256."
                )
            initial = Path(initial_path).expanduser().resolve(strict=False)
            output = Path(self.output_dir).expanduser().resolve(strict=False)
            if (
                initial == output
                or initial in output.parents
                or output in initial.parents
            ):
                raise GateError(
                    "output_dir and initial_adapter_path must not overlap."
                )
        if parent_receipt_path:
            if not initial_path:
                raise GateError(
                    "parent_checkpoint_receipt_path requires an initial adapter."
                )
            if not Path(parent_receipt_path).expanduser().is_absolute():
                raise GateError("parent_checkpoint_receipt_path must be absolute.")
        round1_binding = (
            str(self.round1_provenance_path or "").strip(),
            str(self.round1_preflight_path or "").strip(),
            str(self.reviewed_source_commit or "").strip(),
            str(self.round1_view or "").strip(),
        )
        if any(round1_binding) != all(round1_binding):
            raise GateError(
                "round1_provenance_path, round1_preflight_path, "
                "reviewed_source_commit, and round1_view must be supplied "
                "together."
            )
        if initial_path and not all(round1_binding):
            raise GateError(
                "Warm-start training or smoke requires the complete Round-1 "
                "source binding."
            )
        if all(round1_binding):
            if not initial_path or not initial_revision:
                raise GateError(
                    "Round-1 source binding requires an immutable initial adapter."
                )
            if re.fullmatch(r"[0-9a-fA-F]{40}", round1_binding[2]) is None:
                raise GateError(
                    "reviewed_source_commit must be a 40-character commit hash."
                )
            if self.round1_view not in {"full", "natural-only"}:
                raise GateError(
                    "round1_view must be selected explicitly as full or "
                    "natural-only."
                )
        if self.study_variant is not None and self.study_variant not in {
            "bc0",
            "natural_dagger",
            "natural_dagger_probes",
        }:
            raise GateError(
                "study_variant must be bc0, natural_dagger, or "
                "natural_dagger_probes."
            )
        if self.study_variant == "bc0" and initial_path:
            raise GateError("The BC0 study variant cannot warm-start from an adapter.")
        if self.study_variant == "bc0" and parent_receipt_path:
            raise GateError(
                "The BC0 study variant cannot bind a parent checkpoint receipt."
            )
        if self.study_variant in {"natural_dagger", "natural_dagger_probes"} and not initial_path:
            raise GateError(
                f"The {self.study_variant} study variant requires a same-seed BC0 adapter."
            )
        if (
            self.study_variant in {"natural_dagger", "natural_dagger_probes"}
            and not parent_receipt_path
        ):
            raise GateError(
                f"The {self.study_variant} study variant requires the same-seed "
                "BC0 parent checkpoint receipt."
            )
        expected_view = ROUND1_STUDY_VARIANT_TO_VIEW.get(self.study_variant or "")
        if expected_view is not None and self.round1_view != expected_view:
            raise GateError(
                f"Study variant {self.study_variant} requires "
                f"round1_view={expected_view}."
            )
        if self.study_variant == "bc0" and self.round1_view is not None:
            raise GateError("The BC0 study variant cannot select a Round-1 view.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_lora_config(settings: LoraSettings) -> Any:
    try:
        from peft import LoraConfig
    except Exception as exc:  # pragma: no cover - depends on training environment.
        raise GateError(f"PEFT is required for LoRA training: {exc}") from exc
    return LoraConfig(**settings.kwargs())


def resolve_language_lora_targets(model: Any, suffixes: Sequence[str] = LORA_TARGET_MODULES) -> tuple[str, ...]:
    """Resolve exact text-tower paths so PEFT never wraps vision/audio projections."""
    selected: list[str] = []
    for name, module in model.named_modules():
        if ".language_model." not in f".{name}." and not name.startswith("language_model."):
            continue
        if not any(name.endswith(suffix) for suffix in suffixes):
            continue
        # Gemma 4 vision/audio projections may be Gemma4ClippableLinear
        # wrappers. They are excluded by the tower check, but handle a future
        # language wrapper explicitly by targeting its supported inner linear.
        inner = getattr(module, "linear", None)
        if inner is not None and type(module).__name__ == "Gemma4ClippableLinear":
            selected.append(f"{name}.linear")
        else:
            selected.append(name)
    if not selected:
        raise GateError(
            "No language-model LoRA projection modules were found; refusing to broaden adapters to vision/audio towers."
        )
    return tuple(selected)


def _supported_kwargs(callable_object: Any, kwargs: Mapping[str, Any]) -> dict[str, Any]:
    parameters = inspect.signature(callable_object).parameters
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return dict(kwargs)
    return {key: value for key, value in kwargs.items() if key in parameters}


def infer_required_side_input_names(model: Any, processor: Any, model_name: str) -> tuple[str, ...]:
    """Discover token-aligned side inputs required by the training model."""
    discovered = set(processor_token_type_input_names(processor))
    try:
        discovered.update(inspect.signature(model.forward).parameters)
    except (TypeError, ValueError):
        pass

    lowered = model_name.lower()
    if "gemma-4" in lowered or "gemma4" in lowered:
        discovered.add("mm_token_type_ids")
    return tuple(name for name in TOKEN_TYPE_INPUT_NAMES if name in discovered)


def ensure_required_side_inputs(
    examples: Sequence[PreparedExample],
    required_names: Sequence[str],
) -> list[PreparedExample]:
    """Preserve processor values and fill missing text-only side inputs with zeros."""
    required = tuple(dict.fromkeys(required_names))
    unsupported = sorted(set(required) - set(TOKEN_TYPE_INPUT_NAMES))
    if unsupported:
        raise GateError(f"Unsupported token-aligned side inputs requested: {unsupported}.")

    enriched: list[PreparedExample] = []
    for example in examples:
        side_inputs = {key: list(values) for key, values in example.side_inputs.items()}
        for name in required:
            values = side_inputs.setdefault(name, [0] * len(example.input_ids))
            if len(values) != len(example.input_ids):
                raise GateError(f"Prepared example has unaligned {name}.")
        enriched.append(replace(example, side_inputs=side_inputs))
    return enriched


def trl_config_kwargs(settings: TrainerSettings, *, has_validation: bool) -> dict[str, Any]:
    return {
        "output_dir": settings.output_dir,
        "per_device_train_batch_size": settings.batch_size,
        "per_device_eval_batch_size": settings.batch_size,
        "gradient_accumulation_steps": settings.gradient_accumulation_steps,
        "learning_rate": settings.learning_rate,
        "num_train_epochs": settings.epochs,
        "max_steps": settings.max_steps,
        "optim": settings.optimizer,
        "lr_scheduler_type": settings.lr_scheduler_type,
        "logging_steps": settings.logging_steps,
        "save_steps": settings.save_steps,
        "eval_steps": (
            settings.eval_steps
            if has_validation and settings.eval_strategy == "steps"
            else None
        ),
        "eval_strategy": settings.eval_strategy if has_validation else "no",
        "save_strategy": settings.save_strategy,
        "seed": settings.seed,
        "bf16": settings.bf16,
        "fp16": settings.fp16,
        "packing": False,
        "completion_only_loss": False,
        "remove_unused_columns": False,
        "dataset_kwargs": {"skip_prepare_dataset": True},
        "max_length": None,
        "report_to": settings.report_to,
        "run_name": settings.run_name,
        "load_best_model_at_end": getattr(
            settings, "load_best_model_at_end", False
        ),
        "metric_for_best_model": getattr(
            settings, "metric_for_best_model", None
        ),
        "greater_is_better": getattr(settings, "greater_is_better", None),
    }


def build_trl_config(settings: TrainerSettings, *, has_validation: bool) -> Any:
    try:
        from trl import SFTConfig
    except Exception as exc:  # pragma: no cover - depends on training environment.
        raise GateError(f"TRL is required for SFT training: {exc}") from exc
    return SFTConfig(**_supported_kwargs(SFTConfig.__init__, trl_config_kwargs(settings, has_validation=has_validation)))


def _snapshot_trainable_parameters(model: Any) -> dict[str, Any]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _restore_trainable_parameters(
    model: Any,
    snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    current = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if set(current) != set(snapshot):
        missing = sorted(set(snapshot) - set(current))
        added = sorted(set(current) - set(snapshot))
        raise GateError(
            "Trainable parameter set changed during smoke: "
            f"missing={missing}, added={added}."
        )
    for name, parameter in current.items():
        parameter.data.copy_(
            snapshot[name].to(parameter.device, dtype=parameter.dtype)
        )
    model.zero_grad(set_to_none=True)
    return {
        "performed": True,
        "restored_parameter_tensors": len(current),
        "restored_parameter_elements": sum(
            int(parameter.numel()) for parameter in current.values()
        ),
    }


def _seed_training_rngs(seed: int) -> dict[str, Any]:
    """Seed every required RNG engine and return its receipt-bound record."""

    normalized_seed = validate_training_seed(seed)
    try:
        import numpy as np
        import torch
    except Exception as exc:  # pragma: no cover - release training requires both.
        raise GateError(f"Training RNG dependencies are unavailable: {exc}") from exc
    random.seed(normalized_seed)
    np.random.seed(normalized_seed)
    torch.manual_seed(normalized_seed)
    # Call unconditionally. CPU-only builds treat this as a no-op, while the
    # release path separately requires and attests one approved CUDA device.
    torch.cuda.manual_seed_all(normalized_seed)
    return {
        "status": "applied",
        "seed": normalized_seed,
        "engines": list(TRAINING_RNG_ENGINES),
    }


def _records_dataset(examples: Sequence[PreparedExample]) -> Any:
    try:
        from datasets import Dataset
    except Exception as exc:  # pragma: no cover
        raise GateError(f"datasets is required for TRL training: {exc}") from exc
    return Dataset.from_list([example.model_record() for example in examples])
