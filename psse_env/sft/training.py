"""Optional-dependency LoRA/TRL training configuration and entrypoint."""

from __future__ import annotations

import gc
import inspect
import json
import random
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from .collator import AssistantOnlyCollator
from .gates import (
    TOKEN_TYPE_INPUT_NAMES,
    GateError,
    PreparedExample,
    audit_dataset,
    load_exact_processor,
    load_jsonl,
    processor_token_type_input_names,
    validate_grouped_pilot,
)
from .provenance import (
    git_source_state,
    stable_json_sha256,
    validate_generation_provenance,
)
from .smoke import (
    TARGETED_MIN_RELATIVE_LOSS_REDUCTION,
    TARGETED_RECOVERY_CASES,
    TARGETED_TINY_OVERFIT_LEARNING_RATES,
    TargetedSweepResult,
    run_generation_tool_call_smoke,
    run_targeted_recovery_smoke,
    run_training_smoke,
    select_targeted_recovery_slice,
    targeted_recovery_row_selection,
)


LORA_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


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
    round1_provenance_path: str | None = None
    round1_preflight_path: str | None = None
    reviewed_source_commit: str | None = None

    def validate(self) -> None:
        if "gemma-4" not in self.model_name.lower():
            raise GateError(f"TrainerSettings requires a Gemma 4 model id, got {self.model_name!r}.")
        if re.fullmatch(r"[0-9a-fA-F]{40}", self.revision) is None:
            raise GateError("TrainerSettings.revision must be a pinned 40-character commit hash.")
        if self.max_length <= 0 or self.batch_size <= 0:
            raise GateError("max_length and batch_size must be positive.")
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
        round1_binding = (
            str(self.round1_provenance_path or "").strip(),
            str(self.round1_preflight_path or "").strip(),
            str(self.reviewed_source_commit or "").strip(),
        )
        if any(round1_binding) != all(round1_binding):
            raise GateError(
                "round1_provenance_path, round1_preflight_path, and "
                "reviewed_source_commit must be supplied together."
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
    }


def build_trl_config(settings: TrainerSettings, *, has_validation: bool) -> Any:
    try:
        from trl import SFTConfig
    except Exception as exc:  # pragma: no cover - depends on training environment.
        raise GateError(f"TRL is required for SFT training: {exc}") from exc
    return SFTConfig(**_supported_kwargs(SFTConfig.__init__, trl_config_kwargs(settings, has_validation=has_validation)))


def _load_model(settings: TrainerSettings) -> tuple[Any, dict[str, Any]]:
    """Load the byte-verified pinned base snapshot used by release evaluation.

    Training must consume exactly the snapshot the release evaluator attests,
    so the checkpoint gate's paired comparison is against the same base bytes.
    Returns the model together with a durable snapshot attestation record.
    """
    from psse_env.dagger.release_factories import (
        BASE_MODEL_ID,
        BASE_MODEL_REVISION,
        BASE_SNAPSHOT_FILE_MANIFEST,
        BASE_SNAPSHOT_OPTIONAL_FILE_MANIFEST,
        _require_loaded_from_snapshot,
        _resolve_base_snapshot,
        _verify_snapshot_tree,
    )

    if settings.model_name != BASE_MODEL_ID or settings.revision != BASE_MODEL_REVISION:
        raise GateError(
            "Release SFT trains only the reviewed base snapshot "
            f"{BASE_MODEL_ID}@{BASE_MODEL_REVISION}; got "
            f"{settings.model_name!r}@{settings.revision!r}."
        )
    if not settings.local_files_only or settings.trust_remote_code:
        raise GateError(
            "Release SFT requires local_files_only=True and trust_remote_code=False."
        )
    snapshot = _resolve_base_snapshot()
    try:
        from transformers import AutoModelForImageTextToText
    except Exception as exc:  # pragma: no cover
        raise GateError(f"Transformers model classes are unavailable: {exc}") from exc
    kwargs: dict[str, Any] = {
        "local_files_only": True,
        "trust_remote_code": False,
        "device_map": "auto",
    }
    if settings.load_in_4bit:
        try:
            import torch
            from transformers import BitsAndBytesConfig
        except Exception as exc:  # pragma: no cover
            raise GateError(f"4-bit QLoRA requires torch and BitsAndBytesConfig: {exc}") from exc
        compute_dtype = torch.bfloat16 if settings.bf16 else torch.float16
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        kwargs["dtype"] = compute_dtype
    try:
        model = AutoModelForImageTextToText.from_pretrained(str(snapshot), **kwargs)
    except Exception as exc:
        raise GateError(
            "Exact pinned Gemma training-model load failed; no loader fallback was used: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    config = getattr(model, "config", None)
    if getattr(config, "model_type", None) != "gemma4":
        raise GateError("Loaded training model config is not Gemma 4")
    if type(model).__name__ != "Gemma4ForConditionalGeneration":
        raise GateError(
            "Loaded training model class is not the reviewed Gemma 4 conditional model: "
            f"{type(model).__name__}"
        )
    _require_loaded_from_snapshot(config, snapshot, label="training model")
    # The shared Hub cache was verified before the loader opened it.  Hash the
    # exact tree again after the read so a concurrent replacement cannot leave
    # the loaded model carrying only a pre-load attestation.  The processor is
    # loaded by (id, pinned revision, local_files_only), which the Hub resolves
    # from this same verified snapshot directory.
    _verify_snapshot_tree(
        snapshot,
        BASE_SNAPSHOT_FILE_MANIFEST,
        BASE_SNAPSHOT_OPTIONAL_FILE_MANIFEST,
    )
    attestation = {
        "model_id": BASE_MODEL_ID,
        "model_revision": BASE_MODEL_REVISION,
        "snapshot_path": str(snapshot),
        "verified_files": sorted(BASE_SNAPSHOT_FILE_MANIFEST),
        "model_class": type(model).__name__,
    }
    return model, attestation


def _validated_round1_generation_provenance_id(
    *,
    settings: TrainerSettings,
    train_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
    train_file: str | Path,
    validation_file: str | Path,
) -> str | None:
    """Authenticate the sole auxiliary SFT source before grouped ingestion."""

    has_ineligible_rows = any(
        row.get("production_label_eligible") is not True
        for row in [*train_rows, *validation_rows]
    )
    # Import only after this module is fully initialized.  The source gate
    # imports the aggregate builder, whose reviewed release factory imports
    # training helpers from this module.
    from .round1_source_gate import (
        round1_source_binding_required,
        validate_round1_source_mix_gate,
    )

    path_requires_binding = round1_source_binding_required(
        train_file,
        validation_file,
    )
    configured = all(
        (
            settings.round1_provenance_path,
            settings.round1_preflight_path,
            settings.reviewed_source_commit,
            settings.initial_adapter_revision,
        )
    )
    if not configured:
        if has_ineligible_rows or path_requires_binding:
            raise GateError(
                "Round-1 or non-production-label SFT rows require provenance, "
                "preflight, reviewed source commit, and initial-adapter revision "
                "validated through the Round-1 source gate."
            )
        return None

    report = validate_round1_source_mix_gate(
        settings.round1_provenance_path,
        settings.round1_preflight_path,
        reviewed_source_commit=str(settings.reviewed_source_commit).lower(),
        initial_adapter_revision=str(settings.initial_adapter_revision).lower(),
        train_path=train_file,
        validation_path=validation_file,
    )
    provenance_id = report.get("generation_provenance_id")
    if report.get("passed") is not True or not isinstance(
        provenance_id, str
    ) or re.fullmatch(r"[0-9a-f]{64}", provenance_id) is None:
        raise GateError(
            "Round-1 source gate did not return a valid generation provenance ID."
        )
    expected_content = report.get("canonical_dataset_content_sha256")
    actual_content = {
        "train": stable_json_sha256(list(train_rows)),
        "validation": stable_json_sha256(list(validation_rows)),
    }
    if not isinstance(expected_content, Mapping) or any(
        expected_content.get(split) != digest
        for split, digest in actual_content.items()
    ):
        raise GateError(
            "Round-1 source gate authenticated dataset bytes different from "
            "the rows already loaded for training."
        )
    return provenance_id


def _prepare_pilot(
    *,
    train_file: str | Path,
    validation_file: str | Path,
    settings: TrainerSettings,
    pilot_minimum_rows: int,
    pilot_maximum_rows: int,
) -> tuple[Any, list[PreparedExample], list[PreparedExample]]:
    settings.validate()
    train_rows = load_jsonl(train_file)
    validation_rows = load_jsonl(validation_file)
    validated_round1_id = _validated_round1_generation_provenance_id(
        settings=settings,
        train_rows=train_rows,
        validation_rows=validation_rows,
        train_file=train_file,
        validation_file=validation_file,
    )
    grouped = validate_grouped_pilot(
        {"train": train_rows, "validation": validation_rows},
        group_key="physical_root_fingerprint",
        required_protocol="canonical",
        minimum_rows=pilot_minimum_rows,
        maximum_rows=pilot_maximum_rows,
        validated_round1_generation_provenance_id=validated_round1_id,
    )
    if not grouped.passed:
        raise GateError("Grouped pilot gate failed: " + " | ".join(grouped.failures))
    generation = validate_generation_provenance(
        repo_root=Path(__file__).resolve().parents[2],
        datasets={"train": train_file, "validation": validation_file},
        rows=train_rows + validation_rows,
    )
    if not generation["passed"] and not settings.allow_nonrelease_artifacts:
        raise GateError(
            "Generation provenance gate failed: "
            + " | ".join(generation["failures"])
        )
    processor, processor_loader = load_exact_processor(
        settings.model_name,
        settings.revision,
        local_files_only=settings.local_files_only,
        trust_remote_code=settings.trust_remote_code,
    )
    if (
        settings.required_processor_loader is not None
        and processor_loader != settings.required_processor_loader
    ):
        raise GateError(
            "Release SFT requires "
            f"{settings.required_processor_loader}; loaded {processor_loader}. "
            "Repair the pinned processor cache before running a model stage."
        )
    train_gate = audit_dataset(
        train_rows,
        processor,
        max_length=settings.max_length,
        allow_prompt_truncation=settings.allow_prompt_truncation,
        require_current_registry=True,
    )
    validation_gate = audit_dataset(
        validation_rows,
        processor,
        max_length=settings.max_length,
        allow_prompt_truncation=settings.allow_prompt_truncation,
        require_current_registry=True,
    )
    failures = train_gate.failures + validation_gate.failures
    if failures:
        raise GateError("Tokenizer/mask gate failed: " + " | ".join(failures))
    return processor, train_gate.prepared, validation_gate.prepared


def _attach_lora(model: Any, settings: TrainerSettings, lora: LoraSettings) -> Any:
    try:
        from peft import get_peft_model, prepare_model_for_kbit_training
    except Exception as exc:  # pragma: no cover
        raise GateError(f"PEFT training dependencies are unavailable: {exc}") from exc
    if settings.load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    config = getattr(model, "config", None)
    if config is not None and hasattr(config, "use_cache"):
        config.use_cache = False
    targets = resolve_language_lora_targets(model, lora.target_modules)
    scoped = LoraSettings(
        rank=lora.rank,
        alpha=lora.alpha,
        dropout=lora.dropout,
        target_modules=targets,
    )
    return get_peft_model(model, build_lora_config(scoped))


def _inspect_initial_adapter(
    settings: TrainerSettings,
) -> dict[str, Any] | None:
    """Fail before base-model load unless the warm-start tree has exact bytes."""

    initial_path = str(settings.initial_adapter_path or "").strip()
    expected_revision = str(settings.initial_adapter_revision or "").strip().lower()
    if not initial_path and not expected_revision:
        return None
    if not initial_path or not expected_revision:
        raise GateError("Warm-start adapter identity is incomplete.")
    from psse_env.dagger.release_factories import inspect_release_checkpoint

    try:
        inspection = inspect_release_checkpoint(initial_path)
    except (OSError, ValueError, GateError) as exc:
        raise GateError(f"Initial LoRA adapter inspection failed: {exc}") from exc
    actual_revision = str(inspection.get("tree_sha256") or "").lower()
    if actual_revision != expected_revision:
        raise GateError(
            "Initial LoRA adapter tree digest mismatch: "
            f"expected {expected_revision}, computed {actual_revision}."
        )
    return inspection


def _load_trainable_initial_adapter(
    model: Any,
    settings: TrainerSettings,
    inspection: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    """Load one immutable, byte-verified LoRA checkpoint for warm-start SFT."""

    from psse_env.dagger.release_factories import (
        _copy_verified_checkpoint_tree,
        checkpoint_tree_sha256,
    )

    initial_path = str(settings.initial_adapter_path or "").strip()
    expected_revision = str(settings.initial_adapter_revision or "").strip().lower()
    actual_revision = str(inspection.get("tree_sha256") or "").lower()

    snapshot_owner = None
    try:
        snapshot, snapshot_owner = _copy_verified_checkpoint_tree(
            initial_path,
            expected_revision,
        )
        try:
            from peft import PeftModel, prepare_model_for_kbit_training
        except Exception as exc:  # pragma: no cover - optional live dependency.
            raise GateError(f"PEFT warm-start dependencies are unavailable: {exc}") from exc
        if settings.load_in_4bit:
            model = prepare_model_for_kbit_training(model)
        config = getattr(model, "config", None)
        if config is not None and hasattr(config, "use_cache"):
            config.use_cache = False
        try:
            model = PeftModel.from_pretrained(
                model,
                str(snapshot),
                is_trainable=True,
                local_files_only=True,
            )
        except Exception as exc:  # pragma: no cover - live checkpoint state.
            raise GateError(
                "Exact initial LoRA adapter load failed; training was not started: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if checkpoint_tree_sha256(snapshot) != expected_revision:
            raise GateError(
                "Private initial adapter snapshot changed while PEFT loaded it."
            )
        trainable = [
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        if not trainable:
            raise GateError(
                "Initial LoRA adapter loaded with no trainable parameters."
            )
    finally:
        if snapshot_owner is not None:
            snapshot_owner.cleanup()

    return model, {
        "attestation_schema_version": 1,
        "initial_adapter_path": str(inspection["path"]),
        "initial_adapter_revision": expected_revision,
        "tree_sha256": actual_revision,
        "file_count": int(inspection["file_count"]),
        "total_bytes": int(inspection["total_bytes"]),
        "private_copy_verified": True,
        "peft_load": {
            "is_trainable": True,
            "local_files_only": True,
        },
        "trainable_parameter_names": sorted(trainable),
    }


def _attach_trainable_adapter(
    model: Any,
    settings: TrainerSettings,
    lora: LoraSettings,
    initial_adapter_inspection: Mapping[str, Any] | None = None,
) -> tuple[Any, dict[str, Any] | None]:
    if settings.initial_adapter_path is None:
        return _attach_lora(model, settings, lora), None
    if initial_adapter_inspection is None:
        raise GateError("Warm-start adapter preflight inspection is missing.")
    return _load_trainable_initial_adapter(
        model,
        settings,
        initial_adapter_inspection,
    )


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


def _write_initial_adapter_attestation(
    *,
    settings: TrainerSettings,
    adapter_attestation: Mapping[str, Any],
    smoke_restore: Mapping[str, Any],
    base_snapshot_attestation: Mapping[str, Any],
) -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    source = git_source_state(repo_root)
    payload = {
        **dict(adapter_attestation),
        "training_source": {
            "source_commit": source.get("source_commit"),
            "release_eligible_source": source.get("release_eligible_source"),
        },
        "base_model": {
            "model_id": base_snapshot_attestation.get("model_id"),
            "model_revision": base_snapshot_attestation.get("model_revision"),
        },
        "output_dir": str(
            Path(settings.output_dir).expanduser().resolve(strict=False)
        ),
        "output_input_overlap": False,
        "training_configuration": {
            "learning_rate": settings.learning_rate,
            "epochs": settings.epochs,
            "max_steps": settings.max_steps,
            "batch_size": settings.batch_size,
            "gradient_accumulation_steps": settings.gradient_accumulation_steps,
            "seed": settings.seed,
            "report_to": settings.report_to,
            "run_name": settings.run_name,
        },
        "smoke_restore": dict(smoke_restore),
    }
    output = Path(settings.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "initial_adapter_attestation.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def run_lora_smoke(
    *,
    train_file: str | Path,
    validation_file: str | Path,
    settings: TrainerSettings,
    lora: LoraSettings = LoraSettings(),
    mode: str = "one-batch",
    pilot_minimum_rows: int = 32,
    pilot_maximum_rows: int = 128,
    tiny_overfit_steps: int = 20,
    targeted_recovery: bool = False,
    minimum_relative_loss_reduction: float = (
        TARGETED_MIN_RELATIVE_LOSS_REDUCTION
    ),
) -> Any:
    """Gate and run LoRA forward/backward, stopping before TRL training."""
    if mode not in {"one-batch", "tiny-overfit"}:
        raise ValueError("mode must be 'one-batch' or 'tiny-overfit'.")
    if targeted_recovery and mode != "tiny-overfit":
        raise ValueError(
            "The targeted recovery slice is valid only for tiny-overfit mode."
        )
    processor, train_examples, _ = _prepare_pilot(
        train_file=train_file,
        validation_file=validation_file,
        settings=settings,
        pilot_minimum_rows=pilot_minimum_rows,
        pilot_maximum_rows=pilot_maximum_rows,
    )
    train_rows = load_jsonl(train_file)
    if targeted_recovery:
        # Fail before allocating the 31B model when any reviewed case is absent.
        targeted_recovery_row_selection(train_rows)
    initial_adapter_inspection = _inspect_initial_adapter(settings)
    model, _snapshot_attestation = _load_model(settings)
    required_side_inputs = infer_required_side_input_names(model, processor, settings.model_name)
    train_examples = ensure_required_side_inputs(train_examples, required_side_inputs)
    selected_recovery = (
        select_targeted_recovery_slice(train_rows, train_examples)
        if targeted_recovery
        else None
    )
    model, _initial_adapter_attestation = _attach_trainable_adapter(
        model,
        settings,
        lora,
        initial_adapter_inspection,
    )
    if selected_recovery is not None:
        return run_targeted_recovery_smoke(
            model,
            processor,
            selected_recovery,
            steps=tiny_overfit_steps,
            learning_rate=settings.learning_rate,
            minimum_relative_loss_reduction=minimum_relative_loss_reduction,
        )
    steps = 1 if mode == "one-batch" else tiny_overfit_steps
    result = run_training_smoke(
        model,
        processor,
        train_examples,
        steps=steps,
        learning_rate=settings.learning_rate,
        batch_size=settings.batch_size,
        require_loss_decrease=mode == "tiny-overfit",
    )
    if mode == "tiny-overfit":
        tool_example = next(
            (example for example in train_examples if example.expected_tool_call is not None),
            None,
        )
        if tool_example is None:
            raise GateError("Tiny-overfit generation gate found no tool-call target row.")
        parsed = run_generation_tool_call_smoke(model, processor, tool_example)
        result = replace(
            result,
            generation_round_trip=True,
            generated_tool_name=parsed.name,
        )
    return result


def _reset_targeted_smoke_rng(seed: int) -> None:
    """Reset every available RNG so each LR sees the same LoRA initialization."""

    random.seed(int(seed))
    try:
        import numpy as np

        np.random.seed(int(seed))
    except ImportError:  # pragma: no cover - numpy is present in release env.
        pass
    try:
        import torch

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    except ImportError:  # pragma: no cover - live smoke requires torch anyway.
        pass


def _release_targeted_smoke_memory() -> None:
    """Release one diagnostic model before loading the next LR candidate."""

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:  # pragma: no cover - live smoke requires torch anyway.
        pass


def run_targeted_lora_smoke_sweep(
    *,
    train_file: str | Path,
    validation_file: str | Path,
    settings: TrainerSettings,
    lora: LoraSettings = LoraSettings(),
    pilot_minimum_rows: int = 32,
    pilot_maximum_rows: int = 128,
    tiny_overfit_steps: int = 20,
    minimum_relative_loss_reduction: float = (
        TARGETED_MIN_RELATIVE_LOSS_REDUCTION
    ),
    learning_rates: Sequence[float] = TARGETED_TINY_OVERFIT_LEARNING_RATES,
) -> TargetedSweepResult:
    """Run the reviewed three-rate diagnostic sweep from identical starts."""

    normalized_rates = tuple(float(value) for value in learning_rates)
    if normalized_rates != TARGETED_TINY_OVERFIT_LEARNING_RATES:
        raise ValueError(
            "Targeted recovery LR sweep must be exactly 1e-4, 3e-4, 1e-3."
        )
    # This cheap semantic check prevents three expensive model allocations when
    # the aggregate does not yet contain the exact reviewed recovery slice.
    targeted_recovery_row_selection(load_jsonl(train_file))
    runs: list[dict[str, Any]] = []
    successful: list[float] = []
    for learning_rate in normalized_rates:
        _reset_targeted_smoke_rng(settings.seed)
        diagnostic_settings = replace(settings, learning_rate=learning_rate)
        try:
            result = run_lora_smoke(
                train_file=train_file,
                validation_file=validation_file,
                settings=diagnostic_settings,
                lora=lora,
                mode="tiny-overfit",
                pilot_minimum_rows=pilot_minimum_rows,
                pilot_maximum_rows=pilot_maximum_rows,
                tiny_overfit_steps=tiny_overfit_steps,
                targeted_recovery=True,
                minimum_relative_loss_reduction=(
                    minimum_relative_loss_reduction
                ),
            )
        except (GateError, ValueError) as exc:
            runs.append(
                {
                    "passed": False,
                    "learning_rate": learning_rate,
                    "error": str(exc),
                }
            )
            _release_targeted_smoke_memory()
            continue
        run = result.to_dict()
        run["passed"] = True
        runs.append(run)
        successful.append(learning_rate)
        _release_targeted_smoke_memory()

    best_rate: float | None = None
    successful_runs = [run for run in runs if run.get("passed") is True]
    if successful_runs:
        best = max(
            successful_runs,
            key=lambda run: float(run.get("relative_loss_reduction") or 0.0),
        )
        best_rate = float(best["learning_rate"])
    return TargetedSweepResult(
        passed=bool(successful),
        learning_rates=normalized_rates,
        minimum_relative_loss_reduction=float(minimum_relative_loss_reduction),
        required_cases=TARGETED_RECOVERY_CASES,
        successful_learning_rates=tuple(successful),
        best_diagnostic_learning_rate=best_rate,
        runs=tuple(runs),
    )


def _records_dataset(examples: Sequence[PreparedExample]) -> Any:
    try:
        from datasets import Dataset
    except Exception as exc:  # pragma: no cover
        raise GateError(f"datasets is required for TRL training: {exc}") from exc
    return Dataset.from_list([example.model_record() for example in examples])


def run_lora_training(
    *,
    train_file: str | Path,
    validation_file: str | Path,
    settings: TrainerSettings,
    lora: LoraSettings = LoraSettings(),
    pilot_minimum_rows: int = 32,
    pilot_maximum_rows: int = 128,
    smoke_steps: int = 1,
) -> Any:
    """Run all pilot gates, a real optimizer smoke step, then TRL LoRA SFT."""
    # Full training is always release-facing, including programmatic callers
    # that bypass the CLI.  Keep the loader fallback available for standalone
    # diagnostics and optional smoke runs, but never for checkpoint-producing
    # training.
    settings = replace(settings, required_processor_loader="AutoProcessor")
    processor, train_examples, validation_examples = _prepare_pilot(
        train_file=train_file,
        validation_file=validation_file,
        settings=settings,
        pilot_minimum_rows=pilot_minimum_rows,
        pilot_maximum_rows=pilot_maximum_rows,
    )
    initial_adapter_inspection = _inspect_initial_adapter(settings)
    model, snapshot_attestation = _load_model(settings)
    required_side_inputs = infer_required_side_input_names(model, processor, settings.model_name)
    train_examples = ensure_required_side_inputs(train_examples, required_side_inputs)
    validation_examples = ensure_required_side_inputs(validation_examples, required_side_inputs)
    try:
        from trl import SFTTrainer
    except Exception as exc:  # pragma: no cover
        raise GateError(f"TRL training dependencies are unavailable: {exc}") from exc
    model, initial_adapter_attestation = _attach_trainable_adapter(
        model,
        settings,
        lora,
        initial_adapter_inspection,
    )
    trainable_snapshot = _snapshot_trainable_parameters(model)
    run_training_smoke(
        model,
        processor,
        train_examples,
        steps=smoke_steps,
        learning_rate=settings.learning_rate,
        batch_size=settings.batch_size,
        require_loss_decrease=smoke_steps > 1,
    )
    # The smoke gate proves the stack but is not an undocumented training step.
    # Restore pristine LoRA initialization before constructing TRL's optimizer.
    smoke_restore = _restore_trainable_parameters(model, trainable_snapshot)
    if initial_adapter_attestation is not None:
        _write_initial_adapter_attestation(
            settings=settings,
            adapter_attestation=initial_adapter_attestation,
            smoke_restore=smoke_restore,
            base_snapshot_attestation=snapshot_attestation,
        )
    config = build_trl_config(settings, has_validation=True)
    trainer_kwargs = {
        "model": model,
        "args": config,
        "train_dataset": _records_dataset(train_examples),
        "eval_dataset": _records_dataset(validation_examples),
        "data_collator": AssistantOnlyCollator(processor),
    }
    signature = inspect.signature(SFTTrainer.__init__).parameters
    if "processing_class" in signature:
        trainer_kwargs["processing_class"] = processor
    elif "tokenizer" in signature:
        trainer_kwargs["tokenizer"] = processor
    trainer = SFTTrainer(**trainer_kwargs)
    trainer.data_collator = AssistantOnlyCollator(processor)
    result = trainer.train()
    tool_example = next(
        (example for example in train_examples if example.expected_tool_call is not None),
        None,
    )
    if tool_example is None:
        raise GateError("Post-train generation gate found no tool-call target row.")
    run_generation_tool_call_smoke(model, processor, tool_example)
    output = Path(settings.output_dir) / "lora"
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output))
    if hasattr(processor, "save_pretrained"):
        processor.save_pretrained(str(output))
    _normalize_adapter_base_reference(output, snapshot_attestation)
    import json as _json

    attestation_path = Path(settings.output_dir) / "base_snapshot_attestation.json"
    attestation_path.write_text(
        _json.dumps(snapshot_attestation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _normalize_adapter_base_reference(
    adapter_dir: Path, attestation: Mapping[str, Any]
) -> None:
    """Rewrite the adapter's base reference from the snapshot path to the ID.

    The model is loaded from the verified local snapshot directory, so PEFT
    records that path in ``adapter_config.json``.  The checkpoint gate requires
    ``base_model_name_or_path`` to equal the pinned Hub model ID; leaving the
    machine-local path would make every trained adapter unpromotable.
    """
    import json as _json

    config_path = adapter_dir / "adapter_config.json"
    if not config_path.is_file():
        raise GateError("PEFT save produced no adapter_config.json to normalize.")
    adapter_config = _json.loads(config_path.read_text(encoding="utf-8"))
    recorded = adapter_config.get("base_model_name_or_path")
    if recorded not in (attestation["snapshot_path"], attestation["model_id"]):
        raise GateError(
            "Saved adapter references an unexpected base model: "
            f"{recorded!r} is neither the verified snapshot path nor the pinned ID."
        )
    adapter_config["base_model_name_or_path"] = attestation["model_id"]
    config_path.write_text(
        _json.dumps(adapter_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
