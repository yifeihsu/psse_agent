"""Optional-dependency LoRA/TRL training configuration and entrypoint."""

from __future__ import annotations

import inspect
from dataclasses import replace
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .collator import AssistantOnlyCollator
from .gates import GateError, PreparedExample, audit_dataset, load_exact_processor, load_jsonl, validate_grouped_pilot
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
    seed: int = 3407
    bf16: bool = True
    fp16: bool = False
    load_in_4bit: bool = False
    local_files_only: bool = True
    trust_remote_code: bool = False
    allow_prompt_truncation: bool = False

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
        "eval_steps": settings.eval_steps if has_validation else None,
        "eval_strategy": "steps" if has_validation else "no",
        "save_strategy": "steps",
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


def _load_model(settings: TrainerSettings) -> Any:
    try:
        from transformers import AutoModelForCausalLM
    except Exception as exc:  # pragma: no cover
        raise GateError(f"Transformers model classes are unavailable: {exc}") from exc
    kwargs: dict[str, Any] = {
        "revision": settings.revision,
        "local_files_only": settings.local_files_only,
        "trust_remote_code": settings.trust_remote_code,
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
    errors: list[str] = []
    image_text_cls = None
    try:
        from transformers import AutoModelForImageTextToText
        image_text_cls = AutoModelForImageTextToText
    except Exception:
        pass
    for model_cls in (image_text_cls, AutoModelForCausalLM):
        if model_cls is None:
            continue
        try:
            return model_cls.from_pretrained(settings.model_name, **kwargs)
        except Exception as exc:
            errors.append(f"{model_cls.__name__}: {type(exc).__name__}: {exc}")
    raise GateError("Unable to load the pinned Gemma 4 training model. " + " | ".join(errors))


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
        minimum_rows=pilot_minimum_rows,
        maximum_rows=pilot_maximum_rows,
    )
    if not grouped.passed:
        raise GateError("Grouped pilot gate failed: " + " | ".join(grouped.failures))
    processor, _ = load_exact_processor(
        settings.model_name,
        settings.revision,
        local_files_only=settings.local_files_only,
        trust_remote_code=settings.trust_remote_code,
    )
    train_gate = audit_dataset(
        train_rows,
        processor,
        max_length=settings.max_length,
        allow_prompt_truncation=settings.allow_prompt_truncation,
    )
    validation_gate = audit_dataset(
        validation_rows,
        processor,
        max_length=settings.max_length,
        allow_prompt_truncation=settings.allow_prompt_truncation,
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
    model = _attach_lora(_load_model(settings), settings, lora)
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
    processor, train_examples, validation_examples = _prepare_pilot(
        train_file=train_file,
        validation_file=validation_file,
        settings=settings,
        pilot_minimum_rows=pilot_minimum_rows,
        pilot_maximum_rows=pilot_maximum_rows,
    )
    model = _load_model(settings)
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
    return result
