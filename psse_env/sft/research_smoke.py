"""Bounded live GPU smoke for the E4B and Unified 12B research paths."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from psse_env.dagger.evaluator import evaluate_rollout_suites
from psse_env.dagger.release_factories import production_environment_factory
from psse_env.dagger.research_policy_factory import research_gemma_policy_factory
from psse_env.research_models import (
    GEMMA4_12B,
    GEMMA4_E4B,
    get_research_model_spec,
)

from .collator import AssistantOnlyCollator
from .gates import GateError, PreparedExample, load_jsonl
from .research_cli import (
    ResearchTrainerSettings,
    _load_research_base,
    _validate_adapter_files,
    attach_research_adapter,
    prepare_research_examples,
)
from .smoke import run_training_smoke
from .training import (
    LoraSettings,
    _restore_trainable_parameters,
    _seed_training_rngs,
    _snapshot_trainable_parameters,
    ensure_required_side_inputs,
    infer_required_side_input_names,
)


SMOKE_CONTRACT = "research_gemma4_bounded_gpu_smoke_v1"
PROBE_STAGES = (
    "initial_wls",
    "context_request",
    "measurement_correction",
    "parameter_correction",
    "candidate_verification",
    "candidate_commit",
    "candidate_rollback",
    "unsupported_correction_recovery",
    "invalid_precondition_recovery",
    "long_bounded_history",
)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


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


def _state(row: Mapping[str, Any]) -> Mapping[str, Any]:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return {}
    users = [
        message
        for message in messages
        if isinstance(message, Mapping) and message.get("role") == "user"
    ]
    if not users or not isinstance(users[-1].get("content"), str):
        return {}
    try:
        payload = json.loads(users[-1]["content"])
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    state = payload.get("state") if isinstance(payload, Mapping) else None
    return state if isinstance(state, Mapping) else {}


def _target(row: Mapping[str, Any]) -> dict[str, Any]:
    messages = row.get("messages")
    target = messages[-1] if isinstance(messages, list) and messages else None
    calls = target.get("tool_calls") if isinstance(target, Mapping) else None
    call = calls[0] if isinstance(calls, list) and len(calls) == 1 else None
    function = call.get("function") if isinstance(call, Mapping) else None
    arguments = function.get("arguments") if isinstance(function, Mapping) else None
    if not isinstance(function, Mapping) or not isinstance(arguments, Mapping):
        return {}
    return {
        "tool": str(function.get("name") or ""),
        "arguments": copy.deepcopy(dict(arguments)),
    }


def _history_length(row: Mapping[str, Any]) -> int:
    history = _state(row).get("history_window")
    return len(history) if isinstance(history, list) else 0


def _recovery_stratum(row: Mapping[str, Any]) -> str:
    metadata = row.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    labels = metadata.get("labels")
    labels = labels if isinstance(labels, Mapping) else {}
    return str(
        row.get("recovery_stratum")
        or metadata.get("recovery_stratum")
        or labels.get("recovery_stratum")
        or ""
    )


def _has_candidate(row: Mapping[str, Any]) -> bool:
    state = _state(row)
    return bool(state.get("has_open_candidate") or state.get("candidate_state_id"))


def _matches_probe(stage: str, row: Mapping[str, Any]) -> bool:
    target = _target(row).get("tool")
    stratum = _recovery_stratum(row)
    if stage == "initial_wls":
        return target == "wls_from_path" and not _has_candidate(row) and _history_length(row) == 0
    if stage == "context_request":
        return not stratum and target in {
            "get_measurement_context",
            "get_parameter_context",
            "get_topology_context",
        }
    if stage == "measurement_correction":
        return target == "correct_measurements_from_path"
    if stage == "parameter_correction":
        return target == "correct_parameters_from_path"
    if stage == "candidate_verification":
        return target == "wls_from_path" and _has_candidate(row)
    if stage == "candidate_commit":
        return target == "commit_state"
    if stage == "candidate_rollback":
        return target == "rollback_state"
    if stage == "unsupported_correction_recovery":
        return stratum == "unsupported_correction_recovery"
    if stage == "invalid_precondition_recovery":
        return stratum in {
            "invalid_precondition_repair",
            "premature_commit_recovery",
            "premature_escalation_recovery",
        }
    if stage == "long_bounded_history":
        return _history_length(row) >= 4
    raise ValueError(f"Unknown probe stage {stage!r}")


def select_probe_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Select ten fixed stage probes, keeping every example identity distinct."""

    ordered = sorted(
        (copy.deepcopy(dict(row)) for row in rows),
        key=lambda row: (
            -_history_length(row),
            str(row.get("example_id") or ""),
        ),
    )
    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    for stage in PROBE_STAGES:
        match = next(
            (
                row
                for row in ordered
                if str(row.get("example_id") or "") not in used
                and _matches_probe(stage, row)
            ),
            None,
        )
        if match is None:
            raise GateError(f"Probe corpus has no representative row for {stage}")
        example_id = str(match.get("example_id") or "")
        if not example_id or not _state(match) or not _target(match):
            raise GateError(f"Probe row for {stage} is incomplete")
        match["_research_probe_stage"] = stage
        used.add(example_id)
        selected.append(match)
    return selected


def validate_selected_probe_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    materialized = [copy.deepcopy(dict(row)) for row in rows]
    observed = tuple(str(row.get("_research_probe_stage") or "") for row in materialized)
    if observed != PROBE_STAGES:
        raise GateError(
            "Saved smoke probes do not contain the exact ordered ten-stage contract"
        )
    if any(not _state(row) or not _target(row) for row in materialized):
        raise GateError("Saved smoke probe is missing canonical state or target")
    return materialized


def _model_device(model: Any) -> Any:
    try:
        return next(parameter for parameter in model.parameters()).device
    except StopIteration as exc:
        raise GateError("Research smoke model has no parameters") from exc


def _loss(output: Any) -> Any:
    value = getattr(output, "loss", None)
    if value is None and isinstance(output, Mapping):
        value = output.get("loss")
    if value is None:
        raise GateError("Research smoke forward output has no loss")
    return value


def _cuda_sync(torch: Any) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run_cyclic_overfit(
    model: Any,
    processor: Any,
    examples: Sequence[PreparedExample],
    *,
    steps: int,
    learning_rate: float,
) -> dict[str, Any]:
    """Retain a 20-step cyclic microbatch overfit and prove mean loss falls."""

    try:
        import torch
    except Exception as exc:  # pragma: no cover - live GPU dependency.
        raise GateError(f"torch is required for cyclic overfit: {exc}") from exc
    selected = list(examples[: min(10, len(examples))])
    if steps <= 1 or not selected:
        raise ValueError("cyclic overfit requires multiple steps and examples")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise GateError("Cyclic overfit found no trainable parameters")
    before = [parameter.detach().clone() for parameter in trainable]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=0.0)
    collator = AssistantOnlyCollator(processor)
    device = _model_device(model)

    def mean_losses() -> tuple[float, list[float]]:
        observed: list[float] = []
        model.eval()
        with torch.inference_mode():
            for example in selected:
                batch = {
                    key: value.to(device)
                    for key, value in collator([example]).items()
                }
                numeric = float(_loss(model(**batch)).detach().cpu())
                if not math.isfinite(numeric):
                    raise GateError(f"Cyclic overfit evaluation produced loss {numeric}")
                observed.append(numeric)
        return sum(observed) / len(observed), observed

    initial_mean, initial_losses = mean_losses()
    step_losses: list[float] = []
    step_seconds: list[float] = []
    minimum_gradient_norm = math.inf
    maximum_gradient_norm = 0.0
    model.train()
    for step in range(steps):
        example = selected[step % len(selected)]
        optimizer.zero_grad(set_to_none=True)
        batch = {
            key: value.to(device) for key, value in collator([example]).items()
        }
        _cuda_sync(torch)
        started = time.perf_counter()
        loss = _loss(model(**batch))
        numeric = float(loss.detach().cpu())
        if not math.isfinite(numeric):
            raise GateError(f"Cyclic overfit produced non-finite loss {numeric}")
        loss.backward()
        squared_norm = 0.0
        for parameter in trainable:
            if parameter.grad is None:
                continue
            norm = float(parameter.grad.detach().float().norm().cpu())
            if not math.isfinite(norm):
                raise GateError("Cyclic overfit produced non-finite gradients")
            squared_norm += norm * norm
        gradient_norm = math.sqrt(squared_norm)
        if gradient_norm <= 0.0:
            raise GateError("Cyclic overfit produced a zero gradient")
        minimum_gradient_norm = min(minimum_gradient_norm, gradient_norm)
        maximum_gradient_norm = max(maximum_gradient_norm, gradient_norm)
        optimizer.step()
        _cuda_sync(torch)
        step_seconds.append(time.perf_counter() - started)
        step_losses.append(numeric)

    final_mean, final_losses = mean_losses()
    changed = any(
        not torch.equal(reference, parameter.detach())
        for reference, parameter in zip(before, trainable)
    )
    if not changed or not final_mean < initial_mean:
        raise GateError(
            "Cyclic tiny-overfit failed: "
            f"parameter_changed={changed}, initial_mean={initial_mean:.6g}, "
            f"final_mean={final_mean:.6g}"
        )
    return {
        "passed": True,
        "steps": int(steps),
        "examples": len(selected),
        "learning_rate": float(learning_rate),
        "initial_mean_loss": initial_mean,
        "final_mean_loss": final_mean,
        "relative_loss_reduction": (initial_mean - final_mean)
        / max(abs(initial_mean), 1e-12),
        "initial_losses": initial_losses,
        "final_losses": final_losses,
        "step_losses": step_losses,
        "mean_step_seconds": sum(step_seconds) / len(step_seconds),
        "maximum_step_seconds": max(step_seconds),
        "minimum_gradient_norm": minimum_gradient_norm,
        "maximum_gradient_norm": maximum_gradient_norm,
        "parameter_changed": changed,
    }


def _model_report(model: Any, attachment: Mapping[str, Any], settings: ResearchTrainerSettings) -> dict[str, Any]:
    total = sum(int(parameter.numel()) for parameter in model.parameters())
    trainable = sum(
        int(parameter.numel())
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    device_map = getattr(model, "hf_device_map", None)
    mapped = (
        {str(key): str(value) for key, value in device_map.items()}
        if isinstance(device_map, Mapping)
        else {}
    )
    offload = {
        key: value
        for key, value in mapped.items()
        if value.lower() in {"cpu", "disk", "meta"}
    }
    return {
        "loaded_class": f"{type(model).__module__}.{type(model).__name__}",
        "config_model_type": str(getattr(getattr(model, "config", None), "model_type", "")),
        "load_in_4bit_requested": settings.load_in_4bit,
        "is_loaded_in_4bit": bool(getattr(model, "is_loaded_in_4bit", False)),
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_fraction": trainable / total if total else 0.0,
        "adapter_attachment": copy.deepcopy(dict(attachment)),
        "device_map": mapped,
        "offload_entries": offload,
    }


def _environment() -> dict[str, Any]:
    packages = {}
    for name in (
        "torch",
        "transformers",
        "tokenizers",
        "peft",
        "trl",
        "accelerate",
        "bitsandbytes",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = None
    lock = Path(__file__).resolve().parents[1] / "requirements-sft-research.txt"
    lock_sha256 = (
        hashlib.sha256(lock.read_bytes()).hexdigest() if lock.is_file() else None
    )
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "source_commit_recorded_only": commit,
        "dependency_lock": str(lock),
        "dependency_lock_sha256_recorded_only": lock_sha256,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def _probe_policy(policy: Any, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for row in rows:
        stage = str(row["_research_probe_stage"])
        expected = _target(row)
        started = time.perf_counter()
        try:
            canonical_act = getattr(policy, "act_model_observation", None)
            if not callable(canonical_act):
                canonical_act = policy.act
            observed = canonical_act(_state(row))
            error = None
        except Exception as exc:
            observed = None
            error = f"{type(exc).__name__}: {exc}"
        elapsed = time.perf_counter() - started
        metrics = _jsonable(getattr(policy, "last_action_metrics", {}) or {})
        valid = isinstance(observed, Mapping) and isinstance(
            observed.get("arguments"), Mapping
        ) and isinstance(observed.get("tool"), str)
        records.append(
            {
                "stage": stage,
                "example_id": str(row.get("example_id") or ""),
                "expected": expected,
                "observed": _jsonable(observed),
                "schema_valid_single_call": valid,
                "exact_action_match": valid and dict(observed) == expected,
                "generation_seconds": elapsed,
                "prompt_tokens": metrics.get("prompt_tokens"),
                "original_prompt_tokens": metrics.get("original_prompt_tokens"),
                "generated_tokens": metrics.get("generated_tokens"),
                "hit_max_new_tokens": metrics.get("hit_max_new_tokens"),
                "truncated_input_tokens": metrics.get("truncated_input_tokens"),
                "forced_tool_prefix": metrics.get("forced_tool_prefix"),
                "repetition_loop_detected": metrics.get(
                    "repetition_loop_detected"
                ),
                "repetition_span_tokens": metrics.get("repetition_span_tokens"),
                "repetition_ngram_width": metrics.get("repetition_ngram_width"),
                "parse_error": error,
                "generation_metrics": metrics,
            }
        )
    return {
        "requested": len(rows),
        "schema_valid_single_calls": sum(
            row["schema_valid_single_call"] for row in records
        ),
        "exact_action_matches": sum(row["exact_action_match"] for row in records),
        "maximum_token_hits": sum(row["hit_max_new_tokens"] is True for row in records),
        "input_truncations": sum(
            int(row["truncated_input_tokens"] or 0) > 0 for row in records
        ),
        "repetition_checks_available": sum(
            isinstance(row["repetition_loop_detected"], bool) for row in records
        ),
        "repetition_loops": sum(
            row["repetition_loop_detected"] is True for row in records
        ),
        "records": records,
    }


def _closed_loop_slice(
    suites: Mapping[str, Sequence[Mapping[str, Any]]], *, count: int
) -> dict[str, list[dict[str, Any]]]:
    priorities = (
        "standard_success",
        "invalid_action_recovery",
        "forced_error_recovery",
        "partial_success_retention",
        "efficiency",
    )
    ordered_names = [name for name in priorities if name in suites]
    ordered_names.extend(sorted(set(suites) - set(ordered_names)))
    selected: dict[str, list[dict[str, Any]]] = {}
    used_roots: set[str] = set()
    for name in ordered_names:
        for scenario in suites[name]:
            grouping = scenario.get("grouping")
            grouping = grouping if isinstance(grouping, Mapping) else scenario
            root = str(grouping.get("physical_root_fingerprint") or "")
            if root and root in used_roots:
                continue
            selected.setdefault(name, []).append(copy.deepcopy(dict(scenario)))
            if root:
                used_roots.add(root)
            if sum(len(rows) for rows in selected.values()) == count:
                return selected
            break
    raise GateError(f"Closed-loop suite contains fewer than {count} unique-root scenarios")


def _run_closed_loop(
    policy: Any,
    suite_path: Path,
    *,
    scenarios: int,
    max_steps: int,
    seed: int,
) -> dict[str, Any]:
    payload = json.loads(suite_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise GateError("Closed-loop scenario suite must be a JSON object")
    suites = {
        str(name): list(rows)
        for name, rows in payload.items()
        if isinstance(rows, list)
    }
    selected = _closed_loop_slice(suites, count=scenarios)
    result = evaluate_rollout_suites(
        selected,
        env_factory=production_environment_factory,
        policy_factory=lambda **_kwargs: policy,
        max_steps=max_steps,
        seed=seed,
        required_suites=list(selected),
        minimum_suites=1,
        minimum_episodes_per_suite=1,
        minimum_roots_per_suite=1,
        require_release_environment=False,
        require_policy_identity=False,
    ).as_dict()
    episodes = result.get("suite_metrics", {}).get("episodes", [])
    dispositions = []
    for episode in episodes:
        terminal = episode.get("terminal") is True
        at_horizon = int(episode.get("steps") or 0) >= max_steps
        dispositions.append(
            {
                "scenario_id": episode.get("scenario_id"),
                "physical_root": episode.get("physical_root"),
                "terminal": terminal,
                "terminal_outcome": episode.get("terminal_outcome"),
                "horizon_disposition": not terminal and at_horizon,
                "steps": episode.get("steps"),
                "evaluator_error": episode.get("evaluator_error"),
            }
        )
    passed = len(dispositions) == scenarios and all(
        not row["evaluator_error"]
        and (row["terminal"] or row["horizon_disposition"])
        for row in dispositions
    )
    return {
        "passed": passed,
        "suite_path": str(suite_path),
        "scenarios_requested": scenarios,
        "max_steps": max_steps,
        "dispositions": dispositions,
        "evaluation": result,
    }


def _assert_same_smoke_request(
    observed: Mapping[str, Any], expected: Mapping[str, Any], *, source: Path
) -> None:
    """Reject stale result reuse while keeping commit/hash metadata descriptive."""

    mismatches = [
        field
        for field in ("contract", "model", "settings", "inputs")
        if _jsonable(observed.get(field)) != _jsonable(expected.get(field))
    ]
    if mismatches:
        raise GateError(
            f"Existing smoke state does not match this request at {source}: "
            + ", ".join(mismatches)
        )


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    spec = get_research_model_spec(args.model_choice)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    settings = ResearchTrainerSettings(
        model_name=spec.model_id,
        revision=spec.revision,
        architecture=spec.architecture,
        output_dir=str(output),
        max_length=args.max_length,
        batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=args.learning_rate,
        max_steps=args.overfit_steps,
        seed=args.seed,
        load_in_4bit=not args.no_load_in_4bit,
        local_files_only=not args.allow_download,
        trust_remote_code=spec.trust_remote_code,
        allow_prompt_truncation=False,
    )
    settings.validate()
    run_state: dict[str, Any] = {
        "contract": SMOKE_CONTRACT,
        "passed": False,
        "release_eligible": False,
        "model": _jsonable(spec.__dict__),
        "settings": _jsonable(settings.__dict__),
        "environment": _environment(),
        "inputs": {
            "train": str(args.train.expanduser().resolve(strict=True)),
            "validation": str(args.validation.expanduser().resolve(strict=True)),
            "probe_file": str(args.probe_file.expanduser().resolve(strict=True)),
            "closed_loop_suite": str(
                args.closed_loop_suite.expanduser().resolve(strict=True)
            ),
        },
    }
    final_path = output / "research_smoke.json"
    if final_path.is_file():
        prior = json.loads(final_path.read_text(encoding="utf-8"))
        if not isinstance(prior, Mapping) or prior.get("passed") is not True:
            raise GateError(f"Preserving an invalid smoke report: {final_path}")
        _assert_same_smoke_request(prior, run_state, source=final_path)
        return dict(prior)

    adapter_output = output / "lora"
    resume_post_training = adapter_output.exists()
    if resume_post_training:
        saved_stage = output / "stage_adapter_saved.json"
        if not saved_stage.is_file():
            raise GateError(
                "Preserving an adapter with no completed training-stage receipt: "
                f"{adapter_output}"
            )
        preserved = json.loads(saved_stage.read_text(encoding="utf-8"))
        if not isinstance(preserved, Mapping):
            raise GateError(f"Saved adapter stage is not an object: {saved_stage}")
        _assert_same_smoke_request(preserved, run_state, source=saved_stage)
        adapter_config = _validate_adapter_files(adapter_output)
        recorded_base = str(adapter_config.get("base_model_name_or_path") or "")
        if recorded_base != spec.model_id:
            raise GateError(
                f"Saved adapter records base {recorded_base!r}, expected {spec.model_id!r}"
            )
        run_state = copy.deepcopy(dict(preserved))
        run_state["post_training_resume_environment"] = _environment()
    else:
        _write_json(output / "run_config.json", _jsonable(run_state))

    try:
        import torch
    except Exception as exc:  # pragma: no cover - live dependency.
        raise GateError(f"torch is required for the GPU smoke: {exc}") from exc
    if not torch.cuda.is_available():
        raise GateError("Bounded research smoke requires one visible CUDA GPU")
    if not resume_post_training:
        _seed_training_rngs(args.seed)
        processor, train_examples, validation_examples, data_report = (
            prepare_research_examples(
                train_file=args.train,
                validation_file=args.validation,
                settings=settings,
            )
        )
        run_state["data"] = data_report
        _write_json(output / "stage_processor.json", _jsonable(run_state))

        torch.cuda.reset_peak_memory_stats()
        _cuda_sync(torch)
        load_started = time.perf_counter()
        model = _load_research_base(settings)
        _cuda_sync(torch)
        load_seconds = time.perf_counter() - load_started
        required_side_inputs = infer_required_side_input_names(
            model, processor, spec.model_id
        )
        train_examples = ensure_required_side_inputs(
            train_examples, required_side_inputs
        )
        validation_examples = ensure_required_side_inputs(
            validation_examples, required_side_inputs
        )
        model, attachment = attach_research_adapter(
            model,
            settings=settings,
            lora=LoraSettings(
                rank=args.lora_rank,
                alpha=args.lora_alpha,
                dropout=0.0,
            ),
        )
        model_report = _model_report(model, attachment, settings)
        if model_report["offload_entries"]:
            raise GateError(
                "Research smoke does not permit CPU/disk/meta offload: "
                + json.dumps(model_report["offload_entries"], sort_keys=True)
            )
        if model_report["trainable_parameters"] <= 0:
            raise GateError("Research smoke adapter has no trainable parameters")
        if settings.load_in_4bit and not model_report["is_loaded_in_4bit"]:
            raise GateError(
                "Four-bit loading was requested but the live model does not report it"
            )

        pristine = _snapshot_trainable_parameters(model)
        one_step_started = time.perf_counter()
        one_step = run_training_smoke(
            model,
            processor,
            train_examples,
            steps=1,
            learning_rate=args.learning_rate,
            batch_size=1,
            require_loss_decrease=False,
        )
        one_step_seconds = time.perf_counter() - one_step_started
        restored = _restore_trainable_parameters(model, pristine)
        overfit = run_cyclic_overfit(
            model,
            processor,
            train_examples,
            steps=args.overfit_steps,
            learning_rate=args.learning_rate,
        )
        peak_allocated = int(torch.cuda.max_memory_allocated())
        peak_reserved = int(torch.cuda.max_memory_reserved())
        training = {
            "load_seconds": load_seconds,
            "required_side_inputs": list(required_side_inputs),
            "model": model_report,
            "one_optimizer_step": {
                **one_step.to_dict(),
                "seconds": one_step_seconds,
                "restored_before_overfit": restored,
            },
            "retained_tiny_overfit": overfit,
            "cuda_peak_allocated_bytes": peak_allocated,
            "cuda_peak_reserved_bytes": peak_reserved,
        }
        run_state["training"] = training
        _write_json(output / "stage_training.json", _jsonable(run_state))

        staging = output / f".lora.tmp-{os.getpid()}"
        model.save_pretrained(str(staging), safe_serialization=True)
        adapter_config = _validate_adapter_files(staging)
        recorded_base = str(adapter_config.get("base_model_name_or_path") or "")
        if recorded_base != spec.model_id:
            raise GateError(
                f"Saved adapter records base {recorded_base!r}, "
                f"expected {spec.model_id!r}"
            )
        os.replace(staging, adapter_output)
        run_state["adapter"] = {
            "path": str(adapter_output),
            "recorded_base_model": recorded_base,
            "processor_assets_embedded": False,
            "saved": True,
        }
        _write_json(output / "stage_adapter_saved.json", _jsonable(run_state))

        del model
        gc.collect()
        torch.cuda.empty_cache()
        _cuda_sync(torch)
    reload_started = time.perf_counter()
    policy = research_gemma_policy_factory(
        adapter_output,
        base_model=spec.model_id,
        base_revision=spec.revision,
        load_in_4bit=settings.load_in_4bit,
        local_files_only=True,
        trust_remote_code=spec.trust_remote_code,
        prompt_profile=spec.prompt_profile,
        architecture=spec.architecture,
        cache=False,
    )
    reload_seconds = time.perf_counter() - reload_started
    reload_identity = _jsonable(policy.research_policy_identity)
    if reload_identity.get("processor_source") != spec.model_id:
        raise GateError(
            "Reloaded research policy did not use the exact pinned base processor: "
            f"{reload_identity.get('processor_source')!r}"
        )

    selected_probes = validate_selected_probe_rows(load_jsonl(args.probe_file))
    probes = _probe_policy(policy, selected_probes)
    probes["reload_seconds"] = reload_seconds
    probes["reload_identity"] = reload_identity
    run_state["probes"] = probes
    _write_json(output / "stage_probes.json", _jsonable(run_state))
    if probes["input_truncations"]:
        raise GateError("Generation probes truncated one or more input prompts")
    if spec == GEMMA4_E4B:
        probe_gate = probes["exact_action_matches"] >= 1
    elif spec == GEMMA4_12B:
        probe_gate = (
            probes["schema_valid_single_calls"] >= 9
            and probes["maximum_token_hits"] == 0
            and probes["repetition_checks_available"] == len(selected_probes)
            and probes["repetition_loops"] == 0
        )
    else:  # Parser limits ordinary use, but keep custom failures explicit.
        probe_gate = probes["schema_valid_single_calls"] >= 1
    probes["passed"] = probe_gate
    if not probe_gate:
        raise GateError(
            f"{spec.key} generation probe gate failed: "
            f"valid={probes['schema_valid_single_calls']}/10, "
            f"exact={probes['exact_action_matches']}/10, "
            f"max_token_hits={probes['maximum_token_hits']}, "
            f"repetition_loops={probes['repetition_loops']}, "
            "repetition_checks="
            f"{probes['repetition_checks_available']}/{len(selected_probes)}"
        )
    run_state["probes"] = probes
    _write_json(output / "stage_probes.json", _jsonable(run_state))

    closed_loop = _run_closed_loop(
        policy,
        args.closed_loop_suite.expanduser().resolve(strict=True),
        scenarios=args.closed_loop_scenarios,
        max_steps=args.closed_loop_max_steps,
        seed=args.seed,
    )
    run_state["closed_loop"] = closed_loop
    _write_json(output / "stage_closed_loop.json", _jsonable(run_state))
    if not closed_loop["passed"]:
        raise GateError("Closed-loop smoke did not produce clean terminal/horizon dispositions")
    run_state["passed"] = True
    _write_json(final_path, _jsonable(run_state))
    return run_state


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Run one bounded Gemma 4 GPU smoke")
    result.add_argument("--model-choice", choices=("e4b", "12b"), required=True)
    result.add_argument("--train", required=True, type=Path)
    result.add_argument("--validation", required=True, type=Path)
    result.add_argument("--probe-file", required=True, type=Path)
    result.add_argument("--closed-loop-suite", required=True, type=Path)
    result.add_argument("--output-dir", required=True, type=Path)
    result.add_argument("--max-length", type=int, default=16384)
    result.add_argument("--overfit-steps", type=int, default=20)
    result.add_argument("--learning-rate", type=float, default=1e-4)
    result.add_argument("--lora-rank", type=int, default=16)
    result.add_argument("--lora-alpha", type=int, default=16)
    result.add_argument("--closed-loop-scenarios", type=int, default=3)
    result.add_argument("--closed-loop-max-steps", type=int, default=8)
    result.add_argument("--seed", type=int, default=20260720)
    result.add_argument("--allow-download", action="store_true")
    result.add_argument("--no-load-in-4bit", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        report = run_smoke(args)
    except Exception as exc:
        failure = {
            "contract": SMOKE_CONTRACT,
            "passed": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        try:
            _write_json(args.output_dir.expanduser().resolve() / "failure.json", failure)
        except Exception:
            pass
        print(json.dumps(failure, indent=2, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(_jsonable(report), indent=2, sort_keys=True))
    return 0 if report.get("passed") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
