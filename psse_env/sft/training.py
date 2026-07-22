"""Optional-dependency LoRA/TRL training configuration and entrypoint."""

from __future__ import annotations

import inspect
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
from .provenance import validate_generation_provenance
from .smoke import run_generation_tool_call_smoke, run_training_smoke


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

    def validate(self) -> None:
        if "gemma-4" not in self.model_name.lower():
            raise GateError(f"TrainerSettings requires a Gemma 4 model id, got {self.model_name!r}.")
        import re
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
        "report_to": "none",
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
    grouped = validate_grouped_pilot(
        {"train": train_rows, "validation": validation_rows},
        group_key="physical_root_fingerprint",
        required_protocol="canonical",
        minimum_rows=pilot_minimum_rows,
        maximum_rows=pilot_maximum_rows,
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
) -> Any:
    """Gate and run LoRA forward/backward, stopping before TRL training."""
    if mode not in {"one-batch", "tiny-overfit"}:
        raise ValueError("mode must be 'one-batch' or 'tiny-overfit'.")
    processor, train_examples, _ = _prepare_pilot(
        train_file=train_file,
        validation_file=validation_file,
        settings=settings,
        pilot_minimum_rows=pilot_minimum_rows,
        pilot_maximum_rows=pilot_maximum_rows,
    )
    model, _snapshot_attestation = _load_model(settings)
    required_side_inputs = infer_required_side_input_names(model, processor, settings.model_name)
    train_examples = ensure_required_side_inputs(train_examples, required_side_inputs)
    model = _attach_lora(model, settings, lora)
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
    model, snapshot_attestation = _load_model(settings)
    required_side_inputs = infer_required_side_input_names(model, processor, settings.model_name)
    train_examples = ensure_required_side_inputs(train_examples, required_side_inputs)
    validation_examples = ensure_required_side_inputs(validation_examples, required_side_inputs)
    try:
        from trl import SFTTrainer
    except Exception as exc:  # pragma: no cover
        raise GateError(f"TRL training dependencies are unavailable: {exc}") from exc
    model = _attach_lora(model, settings, lora)
    trainable_snapshot = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
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
    for name, parameter in model.named_parameters():
        if name in trainable_snapshot:
            parameter.data.copy_(trainable_snapshot[name].to(parameter.device, dtype=parameter.dtype))
    model.zero_grad(set_to_none=True)
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
