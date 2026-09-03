"""Compact research-only postflight for full Gemma 4 BC0 training."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from psse_env.research_models import GEMMA4_12B


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_full_bc0_postflight(
    output_dir: Path,
    *,
    expected_train_rows: int,
    expected_validation_rows: int,
    expected_eval_steps: Sequence[int],
    minimum_global_step: int,
    maximum_global_step: int,
) -> dict[str, Any]:
    root = output_dir.expanduser().resolve(strict=True)
    report = _load_object(root / "research_run.json")
    state = _load_object(root / "trainer_state.json")
    adapter = _load_object(root / "lora" / "adapter_config.json")
    stage_value = report.get("preserved_training_stage", report)
    stage = dict(stage_value) if isinstance(stage_value, Mapping) else {}
    model = report.get("model_selection", {})
    model = model if isinstance(model, Mapping) else {}
    settings = stage.get("settings", {})
    settings = settings if isinstance(settings, Mapping) else {}
    metrics = stage.get("training_metrics", {})
    metrics = metrics if isinstance(metrics, Mapping) else {}
    delta = stage.get("adapter_delta", {})
    delta = delta if isinstance(delta, Mapping) else {}
    reload = report.get("reload", {})
    reload = reload if isinstance(reload, Mapping) else {}
    splits = report.get("data", {}).get("splits", {})
    splits = splits if isinstance(splits, Mapping) else {}

    evaluations = [
        {"step": int(row["step"]), "eval_loss": float(row["eval_loss"])}
        for row in state.get("log_history", [])
        if isinstance(row, Mapping)
        and _finite_number(row.get("step"))
        and float(row["step"]).is_integer()
        and _finite_number(row.get("eval_loss"))
    ]
    observed_eval_steps = {row["step"] for row in evaluations}
    expected_steps = {int(step) for step in expected_eval_steps}
    checkpoints = sorted(
        path.name for path in root.glob("checkpoint-*") if path.is_dir()
    )
    checkpoint_names = set(checkpoints)
    best_checkpoint = str(state.get("best_model_checkpoint") or "")
    best_checkpoint_name = Path(best_checkpoint).name if best_checkpoint else ""
    best_metric = state.get("best_metric")
    best_step = next(
        (
            row["step"]
            for row in evaluations
            if f"checkpoint-{row['step']}" == best_checkpoint_name
        ),
        None,
    )
    best_checkpoint_eval_loss = next(
        (
            row["eval_loss"]
            for row in evaluations
            if row["step"] == best_step
        ),
        None,
    )
    minimum_eval_loss = min(
        (row["eval_loss"] for row in evaluations), default=None
    )
    best_checkpoint_is_minimum = bool(
        _finite_number(best_metric)
        and _finite_number(best_checkpoint_eval_loss)
        and minimum_eval_loss is not None
        and math.isclose(
            float(best_metric),
            float(best_checkpoint_eval_loss),
            rel_tol=1e-7,
            abs_tol=1e-9,
        )
        and math.isclose(
            float(best_checkpoint_eval_loss),
            float(minimum_eval_loss),
            rel_tol=1e-7,
            abs_tol=1e-9,
        )
    )

    adapter_weights = [
        path
        for name in ("adapter_model.safetensors", "adapter_model.bin")
        if (path := root / "lora" / name).is_file()
        and not path.is_symlink()
        and path.stat().st_size > 0
    ]
    published_adapter = adapter_weights[0] if adapter_weights else None
    best_adapter = (
        root / best_checkpoint_name / published_adapter.name
        if best_checkpoint_name and published_adapter is not None
        else None
    )
    published_adapter_sha256 = (
        _sha256(published_adapter) if published_adapter is not None else None
    )
    best_adapter_sha256 = (
        _sha256(best_adapter)
        if best_adapter is not None
        and best_adapter.is_file()
        and not best_adapter.is_symlink()
        and best_adapter.stat().st_size > 0
        else None
    )
    global_step = state.get("global_step")
    train_loss = metrics.get("train_loss")
    checks = {
        "research_run": report.get("passed") is True
        and report.get("completion_errors", []) == [],
        "model": model.get("model_id") == GEMMA4_12B.model_id
        and model.get("revision") == GEMMA4_12B.revision
        and model.get("architecture") == GEMMA4_12B.architecture,
        "splits": splits.get("train_rows") == expected_train_rows
        and splits.get("validation_rows") == expected_validation_rows
        and int(splits.get("train_roots") or 0) > 0
        and int(splits.get("validation_roots") or 0) > 0
        and splits.get("overlap") == [],
        "training": _finite_number(train_loss)
        and int(delta.get("changed_tensors") or 0) > 0
        and isinstance(global_step, int)
        and not isinstance(global_step, bool)
        and minimum_global_step <= global_step <= maximum_global_step,
        "evaluation": expected_steps.issubset(observed_eval_steps),
        "checkpoints": {
            f"checkpoint-{step}" for step in expected_steps
        }.issubset(checkpoint_names),
        "best_eval_checkpoint": settings.get("load_best_model_at_end") is True
        and settings.get("metric_for_best_model") == "eval_loss"
        and settings.get("greater_is_better") is False
        and best_checkpoint_name in checkpoint_names
        and best_step in observed_eval_steps
        and best_checkpoint_is_minimum
        and published_adapter_sha256 == best_adapter_sha256,
        "adapter": str(adapter.get("peft_type") or "").upper() == "LORA"
        and adapter.get("base_model_name_or_path") == GEMMA4_12B.model_id
        and bool(adapter_weights),
        "reload": reload.get("canary_mode")
        == "parseable_single_tool_call_after_reload"
        and reload.get("fresh_base_reconstructed") is True
        and reload.get("adapter_reloaded") is True
        and reload.get("canaries_requested") == 1
        and reload.get("canaries_selected") == 1
        and reload.get("canaries_passed") == 1
        and reload.get("generation_canary_pass") is True,
    }
    result = {
        "contract": "research_gemma4_full_bc0_postflight_v1",
        "passed": all(checks.values()),
        "checks": checks,
        "model_selection": dict(model),
        "splits": dict(splits),
        "global_step": global_step,
        "train_loss": train_loss,
        "changed_adapter_tensors": delta.get("changed_tensors"),
        "expected_eval_steps": sorted(expected_steps),
        "finite_evaluations": evaluations,
        "checkpoints": checkpoints,
        "best_model_checkpoint": best_checkpoint,
        "best_checkpoint_step": best_step,
        "best_checkpoint_eval_loss": best_checkpoint_eval_loss,
        "best_metric": best_metric,
        "minimum_eval_loss": minimum_eval_loss,
        "adapter_weight_files": [path.name for path in adapter_weights],
        "published_adapter_sha256": published_adapter_sha256,
        "best_checkpoint_adapter_sha256": best_adapter_sha256,
        "reload": {
            "canary_mode": reload.get("canary_mode"),
            "fresh_base_reconstructed": reload.get("fresh_base_reconstructed"),
            "adapter_reloaded": reload.get("adapter_reloaded"),
            "canaries_passed": reload.get("canaries_passed"),
            "generation_canary_pass": reload.get("generation_canary_pass"),
            "canaries": reload.get("canaries", []),
        },
    }
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output-dir", required=True, type=Path)
    result.add_argument("--expected-train-rows", required=True, type=int)
    result.add_argument("--expected-validation-rows", required=True, type=int)
    result.add_argument(
        "--expected-eval-step", required=True, action="append", type=int
    )
    result.add_argument("--minimum-global-step", required=True, type=int)
    result.add_argument("--maximum-global-step", required=True, type=int)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        result = build_full_bc0_postflight(
            args.output_dir,
            expected_train_rows=args.expected_train_rows,
            expected_validation_rows=args.expected_validation_rows,
            expected_eval_steps=args.expected_eval_step,
            minimum_global_step=args.minimum_global_step,
            maximum_global_step=args.maximum_global_step,
        )
        output = args.output_dir.expanduser().resolve() / "full_bc0_postflight.json"
        temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
        temporary.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "passed": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                indent=2,
            )
        )
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
