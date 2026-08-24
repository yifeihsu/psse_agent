"""Research-only Gemma 4 LoRA continuation entry point.

This keeps the scientifically relevant processor, masking, schema, optimizer,
LoRA, and TRL checks while intentionally omitting release provenance,
attestation, baseline-approval, and checkpoint-promotion policy.
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from psse_env.research_models import (
    DEFAULT_RESEARCH_MODEL,
    RESEARCH_MODEL_SPECS,
    ResearchModelSpec,
    assert_adapter_model_compatible,
    get_research_model_spec,
    resolve_research_model_spec,
)

from .collator import AssistantOnlyCollator
from .gates import GateError, PreparedExample, audit_dataset, load_exact_processor, load_jsonl
from .research_rows import normalize_research_rows
from .smoke import generate_single_tool_call, run_training_smoke
from .training import (
    LoraSettings,
    _records_dataset,
    _restore_trainable_parameters,
    _seed_training_rngs,
    _snapshot_trainable_parameters,
    build_lora_config,
    build_trl_config,
    ensure_required_side_inputs,
    infer_required_side_input_names,
    resolve_language_lora_targets,
    validate_training_seed,
)


RESEARCH_TRAINING_CONTRACT = "research_gemma_lora_training_v1"


@dataclass(frozen=True)
class ResearchTrainerSettings:
    model_name: str
    revision: str
    output_dir: str
    architecture: str | None = None
    max_length: int = 16384
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 3e-5
    epochs: float = 1.0
    max_steps: int = -1
    optimizer: str = "adamw_torch"
    lr_scheduler_type: str = "linear"
    logging_steps: int = 1
    save_steps: int = 8
    eval_steps: int = 8
    eval_strategy: str = "steps"
    save_strategy: str = "steps"
    seed: int = 20260720
    bf16: bool = True
    fp16: bool = False
    load_in_4bit: bool = True
    local_files_only: bool = True
    trust_remote_code: bool = False
    allow_prompt_truncation: bool = True
    report_to: str = "none"
    run_name: str | None = None
    initial_adapter_path: str | None = None
    resume_from_checkpoint: str | bool | None = None

    def resolved_model_spec(self) -> ResearchModelSpec:
        try:
            return resolve_research_model_spec(
                model=self.model_name,
                revision=self.revision,
                architecture=self.architecture,
            )
        except ValueError as exc:
            raise GateError(str(exc)) from exc

    def validate(self) -> None:
        spec = self.resolved_model_spec()
        if len(spec.revision) != 40 or any(
            character not in "0123456789abcdefABCDEF" for character in spec.revision
        ):
            raise GateError("Research model revision must be a pinned 40-character commit")
        if self.max_length <= 0 or self.batch_size <= 0 or self.gradient_accumulation_steps <= 0:
            raise GateError("max_length, batch_size, and gradient accumulation must be positive")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise GateError("learning_rate must be finite and positive")
        if not math.isfinite(self.epochs) or self.epochs <= 0:
            raise GateError("epochs must be finite and positive")
        if self.optimizer != "adamw_torch" or self.lr_scheduler_type != "linear":
            raise GateError("Research training currently supports adamw_torch with a linear scheduler")
        if self.eval_strategy not in {"epoch", "steps"} or self.save_strategy not in {"epoch", "steps"}:
            raise GateError("eval_strategy and save_strategy must be epoch or steps")
        if self.eval_strategy == "steps" and self.eval_steps <= 0:
            raise GateError("eval_steps must be positive")
        if self.save_strategy == "steps" and self.save_steps <= 0:
            raise GateError("save_steps must be positive")
        if self.bf16 and self.fp16:
            raise GateError("bf16 and fp16 cannot both be enabled")
        if self.report_to not in {"none", "wandb"}:
            raise GateError("report_to must be none or wandb")
        validate_training_seed(self.seed)
        if self.initial_adapter_path:
            adapter = Path(self.initial_adapter_path).expanduser().resolve(strict=True)
            output = Path(self.output_dir).expanduser().resolve(strict=False)
            if not adapter.is_dir():
                raise GateError(f"Initial adapter is not a directory: {adapter}")
            if adapter == output or adapter in output.parents or output in adapter.parents:
                raise GateError("Output and initial adapter paths must not overlap")
            adapter_config = _validate_adapter_files(adapter)
            try:
                assert_adapter_model_compatible(
                    spec.model_id, adapter_config.get("base_model_name_or_path")
                )
            except ValueError as exc:
                raise GateError(str(exc)) from exc


def _row_root(row: Mapping[str, Any]) -> str:
    metadata = row.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    return str(
        row.get("physical_root_fingerprint")
        or metadata.get("physical_root_fingerprint")
        or ""
    ).strip()


def validate_research_splits(
    train_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not train_rows or not validation_rows:
        raise GateError("Research training and validation splits must both be non-empty")
    roots: dict[str, set[str]] = {"train": set(), "validation": set()}
    for split, rows in (("train", train_rows), ("validation", validation_rows)):
        for index, row in enumerate(rows):
            root = _row_root(row)
            if not root:
                raise GateError(f"{split}[{index}] lacks physical_root_fingerprint")
            roots[split].add(root)
            metadata = row.get("metadata")
            protocol = metadata.get("protocol") if isinstance(metadata, Mapping) else None
            if protocol != "canonical":
                raise GateError(f"{split}[{index}] is not a canonical-protocol SFT row")
    overlap = roots["train"] & roots["validation"]
    if overlap:
        raise GateError(
            "Research train/validation physical roots overlap: "
            + ", ".join(sorted(overlap)[:8])
        )
    return {
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "train_roots": len(roots["train"]),
        "validation_roots": len(roots["validation"]),
        "overlap": [],
    }


def prepare_research_examples(
    *,
    train_file: str | Path,
    validation_file: str | Path,
    settings: ResearchTrainerSettings,
) -> tuple[Any, list[PreparedExample], list[PreparedExample], dict[str, Any]]:
    settings.validate()
    raw_train_rows = load_jsonl(train_file)
    raw_validation_rows = load_jsonl(validation_file)
    split_report = validate_research_splits(raw_train_rows, raw_validation_rows)
    train_rows, train_normalization = normalize_research_rows(
        raw_train_rows, source_label="train"
    )
    validation_rows, validation_normalization = normalize_research_rows(
        raw_validation_rows, source_label="validation"
    )
    processor, processor_loader = load_exact_processor(
        settings.model_name,
        settings.revision,
        local_files_only=settings.local_files_only,
        trust_remote_code=settings.trust_remote_code,
    )
    if processor_loader != "AutoProcessor":
        raise GateError(
            "Research Gemma training requires AutoProcessor so chat rendering matches inference"
        )
    if not callable(getattr(processor, "apply_chat_template", None)):
        raise GateError("Research Gemma processor lacks apply_chat_template")
    train_audit = audit_dataset(
        train_rows,
        processor,
        max_length=settings.max_length,
        allow_prompt_truncation=settings.allow_prompt_truncation,
        require_current_registry=False,
    )
    validation_audit = audit_dataset(
        validation_rows,
        processor,
        max_length=settings.max_length,
        allow_prompt_truncation=settings.allow_prompt_truncation,
        require_current_registry=False,
    )
    failures = [*train_audit.failures, *validation_audit.failures]
    if failures:
        raise GateError("Research tokenizer/mask/schema gate failed: " + " | ".join(failures))
    return (
        processor,
        train_audit.prepared,
        validation_audit.prepared,
        {
            "processor_loader": processor_loader,
            "splits": split_report,
            "row_normalization": {
                "train": train_normalization,
                "validation": validation_normalization,
            },
            "train": train_audit.to_dict(),
            "validation": validation_audit.to_dict(),
        },
    )


def _load_research_base(settings: ResearchTrainerSettings) -> Any:
    spec = settings.resolved_model_spec()
    try:
        import torch
        from transformers import AutoModelForMultimodalLM
    except Exception as exc:  # pragma: no cover - optional live dependencies.
        raise GateError(
            "Research small-model dependencies are unavailable. Install the "
            "research dependency overlay (Transformers >=5.10): "
            f"{exc}"
        ) from exc
    kwargs: dict[str, Any] = {
        "revision": settings.revision,
        "local_files_only": settings.local_files_only,
        "trust_remote_code": settings.trust_remote_code,
        "device_map": "auto",
    }
    if settings.load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
        except Exception as exc:  # pragma: no cover
            raise GateError(f"4-bit QLoRA dependencies are unavailable: {exc}") from exc
        compute_dtype = torch.bfloat16 if settings.bf16 else torch.float16
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        kwargs["dtype"] = compute_dtype
    elif settings.bf16:
        kwargs["dtype"] = torch.bfloat16
    elif settings.fp16:
        kwargs["dtype"] = torch.float16
    try:
        model = AutoModelForMultimodalLM.from_pretrained(settings.model_name, **kwargs)
    except Exception as exc:  # pragma: no cover - live model/cache state.
        raise GateError(
            f"Research Gemma base-model load failed: {type(exc).__name__}: {exc}"
        ) from exc
    observed_architecture = getattr(getattr(model, "config", None), "model_type", None)
    if observed_architecture != spec.architecture:
        raise GateError(
            "Loaded research model architecture mismatch: expected "
            f"{spec.architecture!r}, got {observed_architecture!r}"
        )
    missing = [
        name
        for name in ("generate", "get_input_embeddings")
        if not callable(getattr(model, name, None))
    ]
    if missing:
        raise GateError(
            "Loaded research model lacks required capabilities: " + ", ".join(missing)
        )
    return model


def _validate_adapter_files(path: Path) -> dict[str, Any]:
    config_path = path / "adapter_config.json"
    if not config_path.is_file() or not any(
        (path / name).is_file()
        for name in ("adapter_model.safetensors", "adapter_model.bin")
    ):
        raise GateError("Warm-start path lacks LoRA adapter config or weights")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise GateError(f"Warm-start adapter config is invalid: {exc}") from exc
    if str(config.get("peft_type") or "").upper() != "LORA":
        raise GateError("Warm-start adapter is not LoRA")
    return config


def attach_research_adapter(
    model: Any,
    *,
    settings: ResearchTrainerSettings,
    lora: LoraSettings,
) -> tuple[Any, dict[str, Any]]:
    try:
        from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training
    except Exception as exc:  # pragma: no cover
        raise GateError(f"PEFT training dependencies are unavailable: {exc}") from exc
    if settings.load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    config = getattr(model, "config", None)
    if config is not None and hasattr(config, "use_cache"):
        config.use_cache = False
    if settings.initial_adapter_path:
        path = Path(settings.initial_adapter_path).expanduser().resolve(strict=True)
        adapter_config = _validate_adapter_files(path)
        try:
            model = PeftModel.from_pretrained(
                model,
                str(path),
                is_trainable=True,
                local_files_only=settings.local_files_only,
            )
        except Exception as exc:  # pragma: no cover - live adapter state.
            raise GateError(
                f"Research warm-start adapter failed to load: {type(exc).__name__}: {exc}"
            ) from exc
        attachment = {
            "mode": "warm_start",
            "adapter_path": str(path),
            "adapter_recorded_base": adapter_config.get("base_model_name_or_path"),
            "tree_digest_enforced": False,
        }
    else:
        targets = resolve_language_lora_targets(model, lora.target_modules)
        scoped = LoraSettings(
            rank=lora.rank,
            alpha=lora.alpha,
            dropout=lora.dropout,
            target_modules=targets,
        )
        model = get_peft_model(model, build_lora_config(scoped))
        attachment = {"mode": "cold_start", "target_modules": list(targets)}
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable:
        raise GateError("Research adapter has no trainable parameters")
    attachment["trainable_parameter_tensors"] = len(trainable)
    return model, attachment


def _changed_trainable_parameters(model: Any, before: Mapping[str, Any]) -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise GateError(f"torch is unavailable for adapter-delta validation: {exc}") from exc
    current = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if set(current) != set(before):
        raise GateError("Trainable parameter names changed during research training")
    changed_tensors = 0
    changed_elements = 0
    squared_delta = 0.0
    for name, parameter in current.items():
        observed = parameter.detach().cpu()
        reference = before[name]
        if not torch.equal(observed, reference):
            changed_tensors += 1
            changed_elements += int(observed.numel())
            delta = observed.float() - reference.float()
            squared_delta += float((delta * delta).sum().item())
    report = {
        "changed_tensors": changed_tensors,
        "changed_elements": changed_elements,
        "delta_l2": math.sqrt(squared_delta),
    }
    if changed_tensors == 0:
        raise GateError("TRL completed without changing any trainable adapter tensor")
    return report


def _reload_saved_adapter(
    settings: ResearchTrainerSettings,
    processor: Any,
    output: Path,
    canary_examples: Sequence[PreparedExample],
    *,
    canary_count: int,
) -> dict[str, Any]:
    """Reconstruct the saved adapter on a fresh base-model instance."""

    report: dict[str, Any] = {
        "adapter_reloaded": False,
        "fresh_base_reconstructed": False,
        "canary_mode": "parseable_single_tool_call_after_reload",
        "canaries_requested": int(canary_count),
        "canaries_selected": 0,
        "generation_canary_pass": False,
        "canaries": [],
    }
    model: Any | None = None
    base: Any | None = None
    try:
        from peft import PeftModel

        base = _load_research_base(settings)
        report["fresh_base_reconstructed"] = True
        model = PeftModel.from_pretrained(
            base,
            str(output),
            is_trainable=False,
            local_files_only=settings.local_files_only,
        )
        peft_config = getattr(model, "peft_config", None)
        if not isinstance(peft_config, Mapping) or not peft_config:
            raise GateError("Saved adapter was not registered on the fresh base")
        model.eval()
        report["adapter_reloaded"] = True
        selected = [
            example for example in canary_examples if example.expected_tool_call is not None
        ][:canary_count]
        report["canaries_selected"] = len(selected)
        for index, example in enumerate(selected):
            expected = example.expected_tool_call
            try:
                parsed = generate_single_tool_call(
                    model,
                    processor,
                    example,
                )
                report["canaries"].append(
                    {
                        "index": index,
                        "passed": True,
                        "generated_tool": parsed.name,
                        "generated_arguments": dict(parsed.arguments),
                        "expected_tool": expected.name if expected is not None else None,
                        "expected_arguments": (
                            dict(expected.arguments) if expected is not None else None
                        ),
                        "target_tool_match": bool(
                            expected is not None and parsed.name == expected.name
                        ),
                        "exact_action_match": parsed == expected,
                    }
                )
            except Exception as exc:
                report["canaries"].append(
                    {
                        "index": index,
                        "passed": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    except Exception as exc:  # Preserve a usable adapter after any ordinary API failure.
        report["reload_error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if model is not None:
            del model
        if base is not None:
            del base
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    report["canaries_passed"] = sum(row.get("passed") is True for row in report["canaries"])
    report["generation_canary_pass"] = (
        report["adapter_reloaded"] is True
        and report["canaries_requested"] > 0
        and report["canaries_selected"] == report["canaries_requested"]
        and report["canaries_passed"] == report["canaries_requested"]
    )
    return report


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _completed_adapter_settings(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return settings that must still match after training has completed."""
    comparable = dict(value)
    # Resume selection controls how Trainer reaches the completed adapter.  It is
    # no longer scientifically relevant once the atomically published lora/ tree
    # exists, and can legitimately differ on a Slurm requeue that only finalizes
    # reload/report artifacts.
    comparable.pop("resume_from_checkpoint", None)
    return _jsonable(comparable)


def _write_report(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def run_research_training(
    *,
    train_file: str | Path,
    validation_file: str | Path,
    settings: ResearchTrainerSettings,
    lora: LoraSettings = LoraSettings(),
    smoke_steps: int = 1,
    reload_canaries: int = 1,
) -> dict[str, Any]:
    settings.validate()
    model_selection = asdict(settings.resolved_model_spec())
    if smoke_steps <= 0 or reload_canaries <= 0:
        raise ValueError("smoke_steps and reload_canaries must be positive")
    output_dir = Path(settings.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_output = output_dir / "lora"
    inputs = {
        "train": str(Path(train_file).expanduser().resolve()),
        "validation": str(Path(validation_file).expanduser().resolve()),
    }

    processor, train_examples, validation_examples, data_report = prepare_research_examples(
        train_file=train_file,
        validation_file=validation_file,
        settings=settings,
    )
    training_stage_path = output_dir / "training_stage.json"
    if adapter_output.exists():
        completion_errors: list[str] = []
        try:
            saved_adapter_config = _validate_adapter_files(adapter_output)
            recorded_base = str(
                saved_adapter_config.get("base_model_name_or_path") or ""
            )
            if recorded_base and recorded_base != settings.model_name:
                completion_errors.append("saved_adapter_base_model_mismatch")
        except Exception as exc:
            completion_errors.append(f"saved_adapter_invalid:{type(exc).__name__}:{exc}")
        training_stage: dict[str, Any] = {}
        if training_stage_path.is_file():
            try:
                loaded_stage = json.loads(training_stage_path.read_text(encoding="utf-8"))
                if not isinstance(loaded_stage, Mapping):
                    raise ValueError("training_stage.json is not an object")
                training_stage = dict(loaded_stage)
            except Exception as exc:
                completion_errors.append(
                    f"training_stage_invalid:{type(exc).__name__}:{exc}"
                )
        else:
            completion_errors.append("training_stage_missing")
        stage_metrics = training_stage.get("training_metrics")
        stage_metrics = stage_metrics if isinstance(stage_metrics, Mapping) else {}
        stage_loss = stage_metrics.get("train_loss")
        if (
            isinstance(stage_loss, bool)
            or not isinstance(stage_loss, (int, float))
            or not math.isfinite(float(stage_loss))
        ):
            completion_errors.append("finite_training_loss_not_preserved")
        stage_delta = training_stage.get("adapter_delta")
        stage_delta = stage_delta if isinstance(stage_delta, Mapping) else {}
        if int(stage_delta.get("changed_tensors") or 0) <= 0:
            completion_errors.append("adapter_delta_not_preserved")
        stage_settings = training_stage.get("settings")
        stage_settings = stage_settings if isinstance(stage_settings, Mapping) else {}
        if _completed_adapter_settings(stage_settings) != _completed_adapter_settings(
            asdict(settings)
        ):
            completion_errors.append("training_stage_settings_mismatch")
        stage_lora = training_stage.get("lora")
        stage_lora = stage_lora if isinstance(stage_lora, Mapping) else {}
        if _jsonable(stage_lora) != _jsonable(asdict(lora)):
            completion_errors.append("training_stage_lora_mismatch")
        stage_inputs = training_stage.get("inputs")
        stage_inputs = stage_inputs if isinstance(stage_inputs, Mapping) else {}
        if _jsonable(stage_inputs) != _jsonable(inputs):
            completion_errors.append("training_stage_inputs_mismatch")
        stage_data = training_stage.get("data")
        stage_data = stage_data if isinstance(stage_data, Mapping) else {}
        if _jsonable(stage_data) != _jsonable(data_report):
            completion_errors.append("training_stage_data_mismatch")
        reload_report = _reload_saved_adapter(
            settings,
            processor,
            adapter_output,
            [*validation_examples, *train_examples],
            canary_count=reload_canaries,
        )
        report = {
            "passed": not completion_errors
            and reload_report.get("generation_canary_pass") is True,
            "contract": RESEARCH_TRAINING_CONTRACT,
            "release_eligible": False,
            "resumed_saved_adapter_finalization": True,
            "model_selection": model_selection,
            "settings": asdict(settings),
            "lora": asdict(lora),
            "data": data_report,
            "saved_adapter": str(adapter_output),
            "preserved_training_stage": training_stage,
            "completion_errors": completion_errors,
            "reload": reload_report,
        }
        _write_report(output_dir / "research_run.json", report)
        return report

    _seed_training_rngs(settings.seed)
    model = _load_research_base(settings)
    required_side_inputs = infer_required_side_input_names(
        model, processor, settings.model_name
    )
    train_examples = ensure_required_side_inputs(train_examples, required_side_inputs)
    validation_examples = ensure_required_side_inputs(
        validation_examples, required_side_inputs
    )
    model, attachment = attach_research_adapter(
        model, settings=settings, lora=lora
    )
    pristine = _snapshot_trainable_parameters(model)
    smoke = run_training_smoke(
        model,
        processor,
        train_examples,
        steps=smoke_steps,
        learning_rate=settings.learning_rate,
        batch_size=settings.batch_size,
        require_loss_decrease=smoke_steps > 1,
    )
    smoke_restore = _restore_trainable_parameters(model, pristine)

    try:
        from trl import SFTTrainer
    except Exception as exc:  # pragma: no cover
        raise GateError(f"TRL training dependencies are unavailable: {exc}") from exc
    config = build_trl_config(settings, has_validation=True)
    trainer_kwargs: dict[str, Any] = {
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
    train_kwargs = {}
    if settings.resume_from_checkpoint is not None:
        train_kwargs["resume_from_checkpoint"] = settings.resume_from_checkpoint
    result = trainer.train(**train_kwargs)
    metrics = dict(getattr(result, "metrics", {}) or {})
    train_loss = metrics.get("train_loss")
    if isinstance(train_loss, bool) or not isinstance(train_loss, (int, float)) or not math.isfinite(float(train_loss)):
        raise GateError(f"TRL did not report a finite train_loss: {train_loss!r}")
    delta = _changed_trainable_parameters(model, pristine)

    if hasattr(trainer, "save_state"):
        trainer.save_state()
    training_stage = {
        "contract": RESEARCH_TRAINING_CONTRACT,
        "model_selection": model_selection,
        "settings": asdict(settings),
        "lora": asdict(lora),
        "inputs": inputs,
        "data": data_report,
        "required_side_inputs": list(required_side_inputs),
        "adapter_attachment": attachment,
        "optimizer_smoke": smoke.to_dict(),
        "smoke_restore": smoke_restore,
        "training_metrics": _jsonable(metrics),
        "adapter_delta": delta,
    }
    _write_report(training_stage_path, training_stage)

    staging_output = output_dir / f".lora.tmp-{os.getpid()}"
    if staging_output.exists():
        raise GateError(
            f"Preserving an existing interrupted adapter staging directory: {staging_output}"
        )
    model.save_pretrained(str(staging_output), safe_serialization=True)
    _validate_adapter_files(staging_output)
    os.replace(staging_output, adapter_output)

    del result, trainer, trainer_kwargs, model
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    reload_report = _reload_saved_adapter(
        settings,
        processor,
        adapter_output,
        [*validation_examples, *train_examples],
        canary_count=reload_canaries,
    )
    passed = reload_report.get("generation_canary_pass") is True
    report = {
        "passed": passed,
        "contract": RESEARCH_TRAINING_CONTRACT,
        "release_eligible": False,
        **training_stage,
        "saved_adapter": str(adapter_output),
        "processor_assets_embedded": False,
        "reload": reload_report,
    }
    _write_report(output_dir / "research_run.json", report)
    return report


def _seed(value: str) -> int:
    try:
        parsed = int(value)
        validate_training_seed(parsed)
    except (ValueError, GateError) as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return parsed


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Research-only Gemma 4 LoRA/TRL continuation"
    )
    result.add_argument(
        "--model-choice",
        choices=tuple(RESEARCH_MODEL_SPECS),
        default=DEFAULT_RESEARCH_MODEL.key,
        help="Atomic pinned model preset; 12b is the reportable pilot and e4b the fast smoke",
    )
    result.add_argument(
        "--model",
        help="Advanced override for the preset model id or a local snapshot",
    )
    result.add_argument(
        "--revision",
        help="Advanced revision override; known presets supply a pinned revision",
    )
    result.add_argument(
        "--architecture",
        choices=("gemma4", "gemma4_unified"),
        help="Required only for an unregistered custom model",
    )
    result.add_argument("--train", required=True, type=Path)
    result.add_argument("--validation", required=True, type=Path)
    result.add_argument("--initial-adapter", type=Path)
    result.add_argument("--output-dir", required=True, type=Path)
    result.add_argument("--max-length", type=int, default=16384)
    result.add_argument("--strict-prompt-length", action="store_true")
    result.add_argument("--batch-size", type=int, default=1)
    result.add_argument("--gradient-accumulation-steps", type=int, default=4)
    result.add_argument("--learning-rate", type=float, default=3e-5)
    result.add_argument("--epochs", type=float, default=1.0)
    result.add_argument("--max-steps", type=int, default=-1)
    result.add_argument("--logging-steps", type=int, default=1)
    result.add_argument("--save-steps", type=int, default=8)
    result.add_argument("--eval-steps", type=int, default=8)
    result.add_argument("--seed", type=_seed, default=20260720)
    result.add_argument("--lora-rank", type=int, default=16)
    result.add_argument("--lora-alpha", type=int, default=16)
    result.add_argument("--lora-dropout", type=float, default=0.0)
    result.add_argument("--smoke-steps", type=int, default=1)
    result.add_argument("--reload-canaries", type=int, default=1)
    result.add_argument("--allow-download", action="store_true")
    result.add_argument("--trust-remote-code", action="store_true")
    result.add_argument("--no-load-in-4bit", action="store_true")
    result.add_argument("--fp16", action="store_true")
    result.add_argument("--no-bf16", action="store_true")
    result.add_argument("--report-to", choices=("none", "wandb"), default="none")
    result.add_argument("--run-name")
    result.add_argument(
        "--resume-from-checkpoint",
        help="Use 'auto' for the latest Trainer checkpoint or supply a checkpoint path",
    )
    return result


def _resume_value(value: str | None) -> str | bool | None:
    if value is None:
        return None
    if value.lower() == "auto":
        return True
    return str(Path(value).expanduser().resolve(strict=True))


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        selected = get_research_model_spec(args.model_choice)
        model_spec = resolve_research_model_spec(
            model=args.model or selected.model_id,
            revision=args.revision,
            architecture=args.architecture,
            default=selected,
        )
        settings = ResearchTrainerSettings(
            model_name=model_spec.model_id,
            revision=model_spec.revision,
            architecture=model_spec.architecture,
            output_dir=str(args.output_dir.expanduser().resolve()),
            max_length=args.max_length,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            max_steps=args.max_steps,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            eval_steps=args.eval_steps,
            seed=args.seed,
            bf16=not args.no_bf16 and not args.fp16,
            fp16=args.fp16,
            load_in_4bit=not args.no_load_in_4bit,
            local_files_only=not args.allow_download,
            trust_remote_code=args.trust_remote_code,
            allow_prompt_truncation=not args.strict_prompt_length,
            report_to=args.report_to,
            run_name=args.run_name,
            initial_adapter_path=(
                str(args.initial_adapter.expanduser().resolve(strict=True))
                if args.initial_adapter is not None
                else None
            ),
            resume_from_checkpoint=_resume_value(args.resume_from_checkpoint),
        )
        report = run_research_training(
            train_file=args.train,
            validation_file=args.validation,
            settings=settings,
            lora=LoraSettings(
                rank=args.lora_rank,
                alpha=args.lora_alpha,
                dropout=args.lora_dropout,
            ),
            smoke_steps=args.smoke_steps,
            reload_canaries=args.reload_canaries,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "passed": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(_jsonable(report), indent=2, sort_keys=True))
    return 0 if report.get("passed") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
