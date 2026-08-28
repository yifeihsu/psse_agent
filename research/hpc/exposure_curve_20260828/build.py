"""Fail-closed builder and gates for the 2026-08-28 exposure-curve cell."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

from research.hpc.occupancy_cell_20260827 import build as screening


CONTRACT = "research_occupancy_exposure_curve_v1"
SEED = 20260823
MODELS = screening.MODELS
EXPECTED_ENVIRONMENT = screening.EXPECTED_ENVIRONMENT
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")
GPU_FEATURE_UNION = "a100|h100|h200|rtx6000"
GPU_FAMILY_UNION = "A100|H100|H200|RTX6000"
CHECKPOINT_TARGETS = {
    "p075": Fraction(3, 4),
    "p100": Fraction(1, 1),
    "p150": Fraction(3, 2),
    "p200": Fraction(2, 1),
}
COMMON_SPLIT_SHA256 = (
    "62cbc48f45dadcea6b3635d072ecb13da0e4c60c0fe143ba194818967dae888b"
)

ARMS: dict[str, dict[str, Any]] = {
    "A": {"lane": "e2b", "inclusion": "selective"},
    "B": {"lane": "e2b", "inclusion": "learner_full"},
    "C": {"lane": "e2b", "inclusion": "full_occupancy"},
    "D": {"lane": "12b", "inclusion": "selective"},
    "E": {"lane": "12b", "inclusion": "learner_full"},
    "F": {"lane": "12b", "inclusion": "full_occupancy"},
}

EXPECTED_COUNTS: dict[str, dict[str, Any]] = {
    "A": {
        "round1_rows_collected": 3696,
        "supervisable_rows": 3695,
        "teacher_abstention_rows": 49,
        "round1_audited_rows": 3646,
        "round1_rows_selected": 799,
        "invalid_learner_action_rows_retained": 5,
        "round1_rows_added_to_train": 662,
        "round1_rows_added_to_validation": 137,
        "round1_split_universe_episodes": 333,
        "train_rows": 1358,
        "validation_rows": 340,
        "round1_selected_state_origin_breakdown": {"learner_policy": 799},
    },
    "B": {
        "round1_rows_collected": 3696,
        "supervisable_rows": 3695,
        "teacher_abstention_rows": 49,
        "round1_audited_rows": 3646,
        "round1_rows_selected": 2323,
        "invalid_learner_action_rows_retained": 11,
        "round1_rows_added_to_train": 1870,
        "round1_rows_added_to_validation": 453,
        "round1_split_universe_episodes": 333,
        "train_rows": 2566,
        "validation_rows": 656,
        "round1_selected_state_origin_breakdown": {"learner_policy": 2323},
    },
    "C": {
        "round1_rows_collected": 3696,
        "supervisable_rows": 3695,
        "teacher_abstention_rows": 49,
        "round1_audited_rows": 3646,
        "round1_rows_selected": 3646,
        "invalid_learner_action_rows_retained": 15,
        "round1_rows_added_to_train": 2934,
        "round1_rows_added_to_validation": 712,
        "round1_split_universe_episodes": 333,
        "train_rows": 3630,
        "validation_rows": 915,
        "round1_selected_state_origin_breakdown": {
            "expert_policy": 990,
            "initial": 333,
            "learner_policy": 2323,
        },
    },
    "D": {
        "round1_rows_collected": 3628,
        "supervisable_rows": 3619,
        "teacher_abstention_rows": 18,
        "round1_audited_rows": 3601,
        "round1_rows_selected": 764,
        "invalid_learner_action_rows_retained": 8,
        "round1_rows_added_to_train": 617,
        "round1_rows_added_to_validation": 147,
        "round1_split_universe_episodes": 333,
        "train_rows": 1313,
        "validation_rows": 350,
        "round1_selected_state_origin_breakdown": {"learner_policy": 764},
    },
    "E": {
        "round1_rows_collected": 3628,
        "supervisable_rows": 3619,
        "teacher_abstention_rows": 18,
        "round1_audited_rows": 3601,
        "round1_rows_selected": 2283,
        "invalid_learner_action_rows_retained": 13,
        "round1_rows_added_to_train": 1837,
        "round1_rows_added_to_validation": 446,
        "round1_split_universe_episodes": 333,
        "train_rows": 2533,
        "validation_rows": 649,
        "round1_selected_state_origin_breakdown": {"learner_policy": 2283},
    },
    "F": {
        "round1_rows_collected": 3628,
        "supervisable_rows": 3619,
        "teacher_abstention_rows": 18,
        "round1_audited_rows": 3601,
        "round1_rows_selected": 3601,
        "invalid_learner_action_rows_retained": 22,
        "round1_rows_added_to_train": 2880,
        "round1_rows_added_to_validation": 721,
        "round1_split_universe_episodes": 333,
        "train_rows": 3576,
        "validation_rows": 924,
        "round1_selected_state_origin_breakdown": {
            "expert_policy": 985,
            "initial": 333,
            "learner_policy": 2283,
        },
    },
}

EXPECTED_MISSING_TARGET_ROWS = {"e2b": 1, "12b": 9}
INPUT_KEYS = {
    "e2b_round0_train",
    "e2b_round0_validation",
    "e2b_round1_rows",
    "12b_round0_train",
    "12b_round0_validation",
    "12b_round1_rows",
    "evaluation_scenarios",
}


sha256 = screening.sha256
atomic_json = screening.atomic_json
load_json = screening.load_json
tree_digest = screening.tree_digest
snapshot_manifest = screening.snapshot_manifest
environment_receipt = screening.environment_receipt


def _absolute(value: Any, label: str) -> Path:
    path = Path(str(value))
    if not path.is_absolute() or any(char in str(path) for char in "\n\r\t,"):
        raise ValueError(f"{label} must be an absolute export-safe path")
    return path


def canonical_split_sha256(values: Sequence[str]) -> str:
    payload = json.dumps(
        sorted(values), sort_keys=True, separators=(",", ":")
    ).encode("utf-8") + b"\n"
    return hashlib.sha256(payload).hexdigest()


def _ceil_fraction(value: Fraction) -> int:
    return (value.numerator + value.denominator - 1) // value.denominator


def checkpoint_plan(train_rows: int) -> dict[str, dict[str, int | float]]:
    if isinstance(train_rows, bool) or not isinstance(train_rows, int) or train_rows <= 0:
        raise ValueError("train_rows must be a positive integer")
    updates_per_epoch = (train_rows + 3) // 4
    result: dict[str, dict[str, int | float]] = {}
    for label, target in CHECKPOINT_TARGETS.items():
        if target <= 1:
            step = _ceil_fraction(target * train_rows / 4)
        else:
            step = updates_per_epoch + _ceil_fraction(
                (target - 1) * train_rows / 4
            )
        expected = screening.expected_train_exposure(train_rows, step, 1, 4)
        sampled_rows = int(expected["training_step_rows"])
        result[label] = {
            "target_pass_numerator": target.numerator,
            "target_pass_denominator": target.denominator,
            "target_pass": float(target),
            "optimizer_step": step,
            "sampled_rows": sampled_rows,
            "realized_passes": sampled_rows / train_rows,
        }
    return result


def restart_contract(
    *,
    arm: str,
    config_sha256: str,
    source_tree_sha256: str,
    aggregate_report_sha256: str,
    exposure_sha256: str,
    train_sha256: str,
    validation_sha256: str,
    checkpoint_plan_value: Mapping[str, Any],
    evaluation_scenarios_sha256: str,
) -> dict[str, Any]:
    """Return the external immutable arm contract bound into Trainer checkpoints."""

    return {
        "schema": "research_exposure_curve_arm_contract_v1",
        "cell_contract": CONTRACT,
        "arm": arm,
        "lane": ARMS[arm]["lane"],
        "inclusion": ARMS[arm]["inclusion"],
        "seed": SEED,
        "config_sha256": config_sha256,
        "source_tree_sha256": source_tree_sha256,
        "aggregate_report_sha256": aggregate_report_sha256,
        "exposure_sha256": exposure_sha256,
        "train_sha256": train_sha256,
        "validation_sha256": validation_sha256,
        "checkpoint_plan": dict(checkpoint_plan_value),
        "evaluation_scenarios_sha256": evaluation_scenarios_sha256,
        "gpu_scheduler_union": GPU_FEATURE_UNION,
        "release_evidence": False,
    }


def parse_selected_arms(value: str) -> list[str]:
    selected = value.split(",")
    if (
        not selected
        or len(set(selected)) != len(selected)
        or any(arm not in ARMS for arm in selected)
    ):
        raise ValueError(
            "selected_arms must be a unique comma-separated subset of ARMS"
        )
    return selected


def validate_scheduler_union(value: str) -> None:
    expected = (
        f"e2b={GPU_FEATURE_UNION},12b={GPU_FEATURE_UNION};"
        f"families=e2b={GPU_FAMILY_UNION},12b={GPU_FAMILY_UNION}"
    )
    if value != expected:
        raise ValueError("this experiment requires the complete authorized GPU union")


def validate_config(
    path: str | Path, expected_sha: str | None = None
) -> tuple[dict[str, Any], str]:
    config_path = Path(path).resolve(strict=True)
    config_sha = sha256(config_path)
    if expected_sha and config_sha != expected_sha.lower():
        raise ValueError("configuration SHA-256 changed after submission")
    config = load_json(config_path)
    if config.get("contract") != CONTRACT:
        raise ValueError(f"config contract must be {CONTRACT!r}")
    source_commit = str(config.get("source_commit", "")).lower()
    if not HEX40.fullmatch(source_commit):
        raise ValueError("source_commit must be a 40-character commit")
    source = _absolute(config.get("source_root"), "source_root").resolve(strict=True)
    root = _absolute(config.get("cell_root"), "cell_root").resolve()
    # Preserve the configured invocation path: this research environment is an
    # overlay whose ``bin/python`` symlink selects a different site-packages
    # stack from the canonical base interpreter.  Resolving the symlink would
    # silently replace Transformers/TRL/PEFT with the base environment.
    python = Path(os.path.abspath(_absolute(config.get("python"), "python")))
    if not python.is_file():
        raise ValueError(f"configured Python does not exist: {python}")
    hf_home = _absolute(config.get("hf_home"), "hf_home").resolve(strict=True)
    if not os.access(python, os.X_OK):
        raise ValueError(f"configured Python is not executable: {python}")
    expected_tree = str(config.get("source_tree_sha256", "")).lower()
    if not HEX64.fullmatch(expected_tree) or tree_digest(source) != expected_tree:
        raise ValueError("source tree SHA-256 mismatch")
    if root == source or root.is_relative_to(source):
        raise ValueError("cell_root must not be inside source_root")
    snapshots = config.get("model_snapshot_sha256")
    if not isinstance(snapshots, dict) or set(snapshots) != set(MODELS):
        raise ValueError("model_snapshot_sha256 must contain both model lanes")
    for lane, digest in snapshots.items():
        if not HEX64.fullmatch(str(digest).lower()):
            raise ValueError(f"model_snapshot_sha256.{lane} must be a SHA-256")
        snapshots[lane] = str(digest).lower()
    inputs = config.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != INPUT_KEYS:
        raise ValueError(f"inputs must contain exactly {sorted(INPUT_KEYS)}")
    for name, record in inputs.items():
        if not isinstance(record, Mapping):
            raise ValueError(f"inputs.{name} is not an object")
        item = _absolute(record.get("path"), f"inputs.{name}.path").resolve(
            strict=True
        )
        digest = str(record.get("sha256", "")).lower()
        if not HEX64.fullmatch(digest) or sha256(item) != digest:
            raise ValueError(f"input hash mismatch: {name}")
        record["sha256"] = digest
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from research.collect import load_scenarios

    scenarios = load_scenarios(inputs["evaluation_scenarios"]["path"])
    if len(scenarios) != 65:
        raise ValueError("evaluation suite must contain exactly 65 scenarios")
    config.update(
        source_root=str(source),
        cell_root=str(root),
        python=str(python),
        hf_home=str(hf_home),
        source_commit=source_commit,
        source_tree_sha256=expected_tree,
    )
    return config, config_sha


def input_path(config: Mapping[str, Any], key: str) -> Path:
    return Path(config["inputs"][key]["path"]).resolve(strict=True)


def verify_model_snapshot(
    config: Mapping[str, Any], lane: str
) -> dict[str, Any]:
    return screening.verify_model_snapshot(config, lane)


def _validate_aggregate_report(arm: str, report: Mapping[str, Any]) -> None:
    expected = EXPECTED_COUNTS[arm]
    drift = {
        key: (report.get(key), value)
        for key, value in expected.items()
        if report.get(key) != value
    }
    if drift:
        raise ValueError(f"arm {arm} aggregate count drift: {drift}")
    if report.get("round1_split_universe") != "all_audited_rows_before_inclusion":
        raise ValueError(f"arm {arm} did not use the audited split universe")
    if report.get("release_evidence") is not False:
        raise ValueError(f"arm {arm} is not marked research-only")
    held_out = report.get("round1_split_assignment_validation_episodes")
    if (
        not isinstance(held_out, list)
        or len(held_out) != 66
        or any(not isinstance(value, str) or not value for value in held_out)
    ):
        raise ValueError(f"arm {arm} must bind exactly 66 validation roots")
    if canonical_split_sha256(held_out) != COMMON_SPLIT_SHA256:
        raise ValueError(f"arm {arm} common root split drift")
    lane = ARMS[arm]["lane"]
    missing_targets = int(report["round1_rows_collected"]) - int(
        report["supervisable_rows"]
    )
    if missing_targets != EXPECTED_MISSING_TARGET_ROWS[lane]:
        raise ValueError(f"arm {arm} missing-target count drift")


def _validate_exposure(
    report: Mapping[str, Any], *, train_rows: int, validation_rows: int, lane: str
) -> None:
    if (report.get("model_id"), report.get("model_revision")) != MODELS[lane]:
        raise ValueError(f"{lane} processor identity drift")
    for split, rows in (("train", train_rows), ("validation", validation_rows)):
        item = report.get(split)
        if not isinstance(item, Mapping):
            raise ValueError(f"missing {lane} {split} exposure")
        if item.get("input_rows") != rows:
            raise ValueError(f"{lane} {split} input row count drift")
        prepared = item.get("prepared") or {}
        if item.get("preparation_failure_count") != 0 or prepared.get("rows") != rows:
            raise ValueError(f"{lane} {split} preparation drift")
        if int(prepared.get("supervised_tokens") or 0) <= 0:
            raise ValueError(f"{lane} {split} has no supervised tokens")


def build(config_path: Path, expected_sha: str, attempt: Path) -> dict[str, Any]:
    config, config_sha = validate_config(config_path, expected_sha)
    environment = environment_receipt(config)
    environment["model_snapshots"] = {
        lane: verify_model_snapshot(config, lane) for lane in MODELS
    }
    cell_root = Path(config["cell_root"])
    attempt = attempt.resolve()
    if not attempt.is_relative_to(cell_root / "build"):
        raise ValueError("build attempt must be below <cell_root>/build")
    attempt.mkdir(parents=True, exist_ok=False)
    from research.exposure import summarize_dataset
    from research.model import load_processor
    from research.run_dagger import build_aggregate

    built: dict[str, dict[str, Any]] = {}
    processors: dict[str, Any] = {}
    for arm, spec in ARMS.items():
        lane = spec["lane"]
        destination = attempt / f"aggregate_{arm}"
        report = build_aggregate(
            round0_train=input_path(config, f"{lane}_round0_train"),
            round0_validation=input_path(config, f"{lane}_round0_validation"),
            round1_rows=input_path(config, f"{lane}_round1_rows"),
            output_dir=destination,
            inclusion=spec["inclusion"],
            seed=SEED,
        )
        report["release_evidence"] = False
        _validate_aggregate_report(arm, report)
        report["missing_teacher_target_rows"] = EXPECTED_MISSING_TARGET_ROWS[lane]
        report["teacher_exclusion_rows"] = (
            EXPECTED_MISSING_TARGET_ROWS[lane]
            + int(report["teacher_abstention_rows"])
        )
        report["invalid_action_retention_semantics"] = (
            "learner_action_validity_is_never_a_filter;occupancy_rule_still_applies"
        )
        report_path = destination / "aggregate_report.json"
        atomic_json(report_path, report, exclusive=True)
        if lane not in processors:
            processors[lane], resolved_id, resolved_revision = load_processor(
                model_id=MODELS[lane][0], revision=MODELS[lane][1]
            )
            if (resolved_id, resolved_revision) != MODELS[lane]:
                raise ValueError(f"processor identity drift for {lane}")
        train = Path(report["train"])
        validation = Path(report["validation"])
        exposure = {
            "model_id": MODELS[lane][0],
            "model_revision": MODELS[lane][1],
            "max_length": 8192,
            "train": summarize_dataset(
                train, processors[lane], max_length=8192, label="train"
            ),
            "validation": summarize_dataset(
                validation,
                processors[lane],
                max_length=8192,
                label="validation",
            ),
            "release_evidence": False,
        }
        _validate_exposure(
            exposure,
            train_rows=EXPECTED_COUNTS[arm]["train_rows"],
            validation_rows=EXPECTED_COUNTS[arm]["validation_rows"],
            lane=lane,
        )
        exposure_path = attempt / f"exposure_{arm}.json"
        atomic_json(exposure_path, exposure, exclusive=True)
        plan = checkpoint_plan(EXPECTED_COUNTS[arm]["train_rows"])
        aggregate_report_sha256 = sha256(report_path)
        exposure_sha256 = sha256(exposure_path)
        external_contract = restart_contract(
            arm=arm,
            config_sha256=config_sha,
            source_tree_sha256=config["source_tree_sha256"],
            aggregate_report_sha256=aggregate_report_sha256,
            exposure_sha256=exposure_sha256,
            train_sha256=report["train_sha256"],
            validation_sha256=report["validation_sha256"],
            checkpoint_plan_value=plan,
            evaluation_scenarios_sha256=config["inputs"]["evaluation_scenarios"][
                "sha256"
            ],
        )
        contract_path = attempt / f"restart_contract_{arm}.json"
        atomic_json(contract_path, external_contract, exclusive=True)
        built[arm] = {
            **spec,
            "model_id": MODELS[lane][0],
            "model_revision": MODELS[lane][1],
            "train": str(train),
            "train_sha256": report["train_sha256"],
            "validation": str(validation),
            "validation_sha256": report["validation_sha256"],
            "aggregate_report": str(report_path),
            "aggregate_report_sha256": aggregate_report_sha256,
            "teacher_abstentions": report["teacher_abstentions"],
            "teacher_abstentions_sha256": report["teacher_abstentions_sha256"],
            "exposure": str(exposure_path),
            "exposure_sha256": exposure_sha256,
            "restart_contract": str(contract_path),
            "restart_contract_sha256": sha256(contract_path),
            "restart_contract_value": external_contract,
            "checkpoint_plan": plan,
            "max_steps": int(plan["p200"]["optimizer_step"]),
        }
    split_assignments = {
        frozenset(
            load_json(record["aggregate_report"])[
                "round1_split_assignment_validation_episodes"
            ]
        )
        for record in built.values()
    }
    if len(split_assignments) != 1:
        raise ValueError("the six arms do not share one audited-root split")
    for lane in MODELS:
        abstention_hashes = {
            record["teacher_abstentions_sha256"]
            for record in built.values()
            if record["lane"] == lane
        }
        if len(abstention_hashes) != 1:
            raise ValueError(f"{lane} teacher-abstention stratum drift")
    receipt = {
        "contract": CONTRACT,
        "stage": "build_complete",
        "config": str(config_path.resolve()),
        "config_sha256": config_sha,
        "source_commit": config["source_commit"],
        "source_commit_binding": "informational_label_tree_sha256_authoritative",
        "source_tree_sha256": config["source_tree_sha256"],
        "environment": environment,
        "common_split_sha256": COMMON_SPLIT_SHA256,
        "seed": SEED,
        "arms": built,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_restart_count": int(os.environ.get("SLURM_RESTART_COUNT", "0")),
        "release_evidence": False,
    }
    receipt_path = attempt / "build_receipt.json"
    atomic_json(receipt_path, receipt, exclusive=True)
    pointer = {
        "contract": CONTRACT,
        "receipt": str(receipt_path),
        "receipt_sha256": sha256(receipt_path),
        "config_sha256": config_sha,
        "source_tree_sha256": config["source_tree_sha256"],
    }
    atomic_json(cell_root / "build" / "completed.json", pointer, exclusive=True)
    return pointer


def verify_build(
    config_path: Path, expected_sha: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config, config_sha = validate_config(config_path, expected_sha)
    pointer = load_json(Path(config["cell_root"]) / "build" / "completed.json")
    receipt_path = Path(pointer["receipt"]).resolve(strict=True)
    if pointer.get("receipt_sha256") != sha256(receipt_path):
        raise ValueError("build receipt hash mismatch")
    receipt = load_json(receipt_path)
    if (
        pointer.get("contract") != CONTRACT
        or receipt.get("contract") != CONTRACT
        or receipt.get("stage") != "build_complete"
        or receipt.get("config_sha256") != config_sha
        or receipt.get("source_tree_sha256") != config["source_tree_sha256"]
        or receipt.get("common_split_sha256") != COMMON_SPLIT_SHA256
        or receipt.get("seed") != SEED
        or set(receipt.get("arms", {})) != set(ARMS)
    ):
        raise ValueError("build receipt is not bound to this experiment")
    for arm, record in receipt["arms"].items():
        spec = ARMS[arm]
        expected_plan = checkpoint_plan(EXPECTED_COUNTS[arm]["train_rows"])
        if (
            (record.get("lane"), record.get("inclusion"))
            != (spec["lane"], spec["inclusion"])
            or (record.get("model_id"), record.get("model_revision"))
            != MODELS[spec["lane"]]
            or record.get("checkpoint_plan") != expected_plan
            or record.get("max_steps") != expected_plan["p200"]["optimizer_step"]
        ):
            raise ValueError(f"arm {arm} build contract drift")
        for key in (
            "train",
            "validation",
            "aggregate_report",
            "teacher_abstentions",
            "exposure",
            "restart_contract",
        ):
            path = Path(record[key]).resolve(strict=True)
            if sha256(path) != record[f"{key}_sha256"]:
                raise ValueError(f"arm {arm} {key} changed after build")
        if load_json(record["restart_contract"]) != record.get(
            "restart_contract_value"
        ):
            raise ValueError(f"arm {arm} restart contract drift")
    return config, pointer, receipt


def arm_values(config_path: Path, expected_sha: str, arm: str) -> list[str]:
    _, _, receipt = verify_build(config_path, expected_sha)
    record = receipt["arms"][arm]
    milestones = ",".join(
        f"{label}={value['optimizer_step']}"
        for label, value in record["checkpoint_plan"].items()
    )
    return [
        record["model_id"],
        record["model_revision"],
        str(record["max_steps"]),
        record["train"],
        record["validation"],
        record["train_sha256"],
        record["validation_sha256"],
        record["lane"],
        record["inclusion"],
        record["exposure"],
        record["exposure_sha256"],
        record["restart_contract"],
        str(2 * EXPECTED_COUNTS[arm]["train_rows"]),
        milestones,
    ]


def _valid_resume_checkpoint(path: Path, record: Mapping[str, Any]) -> int | None:
    if not path.is_dir() or path.is_symlink():
        return None
    suffix = path.name.removeprefix("checkpoint-")
    if not suffix.isdigit():
        return None
    step = int(suffix)
    arm_id = next(
        arm
        for arm, spec in ARMS.items()
        if spec["lane"] == record["lane"]
        and spec["inclusion"] == record["inclusion"]
    )
    train_rows = EXPECTED_COUNTS[arm_id]["train_rows"]
    expected_exposure = screening.expected_train_exposure(train_rows, step, 1, 4)
    try:
        from research.train import validate_restart_checkpoint

        ledger = validate_restart_checkpoint(path)
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError):
        return None
    binding = ledger.get("run_binding") or {}
    exposure = ledger.get("training_step_train_exposure") or {}
    if (
        ledger.get("schema") != "research_exposure_checkpoint_v2"
        or ledger.get("global_step") != step
        or binding.get("external_contract") != record["restart_contract_value"]
        or binding.get("model_id") != record["model_id"]
        or binding.get("model_revision") != record["model_revision"]
        or binding.get("train_sha256") != record["train_sha256"]
        or binding.get("validation_sha256") != record["validation_sha256"]
        or binding.get("max_steps") != record["max_steps"]
        or binding.get("seed") != SEED
        or binding.get("pass_normalized_total_rows")
        != 2 * train_rows
        or exposure.get("rows") != expected_exposure["training_step_rows"]
        or exposure.get("batches") != expected_exposure["training_step_batches"]
        or not 0 < step <= int(record["max_steps"])
    ):
        return None
    return step


def find_resume_checkpoint(
    config_path: Path,
    expected_sha: str,
    arm: str,
    job_id: str,
    restart: int,
) -> str:
    config, _, receipt = verify_build(config_path, expected_sha)
    if not job_id.isdigit() or restart < 0:
        raise ValueError("invalid Slurm job/restart identity")
    job_root = Path(config["cell_root"]) / "runs" / arm / f"job-{job_id}"
    candidates: list[tuple[int, Path]] = []
    for attempt in job_root.glob("attempt-r*"):
        raw_restart = attempt.name.removeprefix("attempt-r")
        if not raw_restart.isdigit() or int(raw_restart) >= restart:
            continue
        adapter = attempt / "adapter"
        if not adapter.is_dir() or adapter.is_symlink():
            continue
        for checkpoint in adapter.glob("checkpoint-*"):
            step = _valid_resume_checkpoint(checkpoint, receipt["arms"][arm])
            if step is not None:
                candidates.append((step, checkpoint.resolve()))
    if not candidates:
        return ""
    candidates.sort(key=lambda item: (item[0], str(item[1])))
    return str(candidates[-1][1])


def _milestone_receipt(
    marker_path: Path, record: Mapping[str, Any], label: str
) -> dict[str, Any]:
    marker = load_json(marker_path)
    plan = record["checkpoint_plan"][label]
    arm_id = next(
        arm
        for arm, spec in ARMS.items()
        if spec["lane"] == record["lane"]
        and spec["inclusion"] == record["inclusion"]
    )
    expected = screening.expected_train_exposure(
        EXPECTED_COUNTS[arm_id]["train_rows"],
        int(plan["optimizer_step"]),
        1,
        4,
    )
    exposure = marker.get("training_step_train_exposure") or {}
    binding = marker.get("run_binding") or {}
    from research.train import canonical_json_sha256

    train_rows = EXPECTED_COUNTS[arm_id]["train_rows"]
    validation_rows = EXPECTED_COUNTS[arm_id]["validation_rows"]
    expected_milestones = {
        item: value["optimizer_step"]
        for item, value in record["checkpoint_plan"].items()
    }
    adapter = marker_path.parent.resolve(strict=True)
    marker_adapter_files = marker.get("adapter_files")
    adapter_manifest = (
        marker_adapter_files if isinstance(marker_adapter_files, Mapping) else {}
    )
    excluded = {
        "research_milestone.json",
        "research_restart_checkpoint_complete.json",
    }
    children = list(adapter.iterdir())
    actual_adapter_files = {
        path.name for path in children if path.name not in excluded and path.is_file()
    }
    adapter_manifest_valid = (
        isinstance(marker_adapter_files, Mapping)
        and set(adapter_manifest) == actual_adapter_files
        and "adapter_config.json" in adapter_manifest
        and bool(
            {"adapter_model.safetensors", "adapter_model.bin"}
            & set(adapter_manifest)
        )
        and all(
            path.name in excluded or (path.is_file() and not path.is_symlink())
            for path in children
        )
    )
    if adapter_manifest_valid:
        for name, evidence in adapter_manifest.items():
            path = adapter / name
            if (
                Path(name).name != name
                or not isinstance(evidence, Mapping)
                or isinstance(evidence.get("bytes"), bool)
                or not isinstance(evidence.get("bytes"), int)
                or evidence["bytes"] < 0
                or HEX64.fullmatch(str(evidence.get("sha256", ""))) is None
                or path.stat().st_size != evidence["bytes"]
                or sha256(path) != evidence["sha256"]
            ):
                adapter_manifest_valid = False
                break

    restart_evidence_name = marker.get("restart_checkpoint_manifest_evidence")
    restart_evidence = adapter / "research_restart_checkpoint_complete.json"
    restart_manifest: Mapping[str, Any] = {}
    restart_evidence_valid = (
        restart_evidence_name == "research_restart_checkpoint_complete.json"
        and restart_evidence.is_file()
        and not restart_evidence.is_symlink()
    )
    if restart_evidence_valid:
        try:
            restart_manifest = load_json(restart_evidence)
        except (OSError, ValueError, json.JSONDecodeError):
            restart_evidence_valid = False
    restart_checkpoint = Path(str(marker.get("restart_checkpoint", "")))
    checks = {
        "schema": marker.get("schema") == "research_exposure_milestone_v1",
        "label": marker.get("label") == label,
        "step": marker.get("global_step") == plan["optimizer_step"],
        "restart_checkpoint": restart_checkpoint.is_absolute()
        and restart_checkpoint.name == f"checkpoint-{plan['optimizer_step']}",
        "max_steps": binding.get("max_steps") == record["max_steps"],
        "external_contract": binding.get("external_contract")
        == record["restart_contract_value"],
        "binding_hash": marker.get("run_binding_sha256")
        == canonical_json_sha256(binding),
        "train_hash": binding.get("train_sha256") == record["train_sha256"],
        "validation_hash": binding.get("validation_sha256")
        == record["validation_sha256"],
        "pass_normalized": binding.get("pass_normalized_total_rows")
        == 2 * train_rows,
        "prepared_rows": binding.get("prepared_train_rows") == train_rows
        and binding.get("prepared_validation_rows") == validation_rows,
        "model": (binding.get("model_id"), binding.get("model_revision"))
        == (record["model_id"], record["model_revision"]),
        "recipe": binding.get("max_length") == 8192
        and binding.get("batch_size") == 1
        and binding.get("gradient_accumulation_steps") == 4
        and binding.get("learning_rate") == 0.0001
        and binding.get("lora_rank") == 16
        and binding.get("lora_alpha") == 16
        and binding.get("lora_dropout") == 0.0
        and binding.get("seed") == SEED,
        "restart_policy": binding.get("restart_save_steps") == 25
        and binding.get("restart_save_total_limit") == 2,
        "milestone_plan": binding.get("milestone_steps") == expected_milestones,
        "rows": exposure.get("rows") == expected["training_step_rows"],
        "batches": exposure.get("batches") == expected["training_step_batches"],
        "tokens": isinstance(exposure.get("input_tokens"), int)
        and isinstance(exposure.get("supervised_tokens"), int)
        and exposure.get("input_tokens", 0)
        >= exposure.get("supervised_tokens", 0)
        > 0,
        "restart_manifest_hash": HEX64.fullmatch(
            str(marker.get("restart_checkpoint_manifest_sha256", ""))
        )
        is not None
        and restart_evidence_valid
        and sha256(restart_evidence)
        == marker.get("restart_checkpoint_manifest_sha256"),
        "restart_manifest_binding": restart_evidence_valid
        and restart_manifest.get("schema") == "research_checkpoint_complete_v1"
        and restart_manifest.get("global_step") == plan["optimizer_step"]
        and restart_manifest.get("run_binding_sha256")
        == marker.get("run_binding_sha256"),
        "adapter_manifest": adapter_manifest_valid,
    }
    if not all(checks.values()):
        raise ValueError(
            f"milestone {label} gate failed: "
            f"{[key for key, value in checks.items() if not value]}"
        )
    return {
        "label": label,
        "target_pass": plan["target_pass"],
        "realized_passes": plan["realized_passes"],
        "optimizer_step": plan["optimizer_step"],
        "sampled_rows": plan["sampled_rows"],
        "adapter": str(adapter.resolve()),
        "adapter_files": dict(adapter_manifest),
        "marker": str(marker_path.resolve()),
        "marker_sha256": sha256(marker_path),
        "training_step_train_exposure": exposure,
        "run_binding_sha256": marker.get("run_binding_sha256"),
        "restart_checkpoint": marker.get("restart_checkpoint"),
        "restart_checkpoint_manifest_sha256": marker.get(
            "restart_checkpoint_manifest_sha256"
        ),
        "restart_checkpoint_manifest_evidence": str(restart_evidence.resolve()),
    }


def _attempt_gpu_receipts(job_root: Path) -> list[dict[str, Any]]:
    result = []
    for path in sorted(job_root.glob("attempt-r*/allocated_gpu.json")):
        payload = load_json(path)
        if (
            payload.get("torch_cuda_device_count") != 1
            or payload.get("torch_device_index") != 0
            or payload.get("slurm_job_constraint") != GPU_FEATURE_UNION
            or payload.get("slurm_job_constraint_source") != "sacct"
            or not isinstance(payload.get("name"), str)
            or int(payload.get("total_memory_bytes") or 0) <= 0
        ):
            raise ValueError(f"invalid GPU attempt receipt: {path}")
        result.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "identity": payload,
            }
        )
    if not result:
        raise ValueError("no GPU attempt receipt exists")
    return result


def gate_training(
    config_path: Path,
    expected_sha: str,
    arm: str,
    summary_path: Path,
    job_id: str,
    output: Path,
) -> dict[str, Any]:
    config, _, receipt = verify_build(config_path, expected_sha)
    record = receipt["arms"][arm]
    summary = load_json(summary_path)
    train_rows = EXPECTED_COUNTS[arm]["train_rows"]
    expected = screening.expected_train_exposure(
        train_rows, record["max_steps"], 1, 4
    )
    step_exposure = summary.get("training_step_train_exposure") or {}
    milestones_root = (
        Path(config["cell_root"])
        / "runs"
        / arm
        / f"job-{job_id}"
        / "milestones"
    )
    milestones = {
        label: _milestone_receipt(
            milestones_root / label / "research_milestone.json", record, label
        )
        for label in CHECKPOINT_TARGETS
    }
    expected_milestone_steps = {
        label: value["optimizer_step"]
        for label, value in record["checkpoint_plan"].items()
    }
    checks = {
        "schema": summary.get("schema") == "research_training_summary_v2",
        "model": (summary.get("model_id"), summary.get("model_revision"))
        == (record["model_id"], record["model_revision"]),
        "fresh_base": summary.get("initial_adapter") is None,
        "data": summary.get("train_sha256") == record["train_sha256"]
        and summary.get("validation_sha256") == record["validation_sha256"],
        "updates": summary.get("optimizer_updates_completed")
        == record["max_steps"]
        and summary.get("max_steps") == record["max_steps"],
        "two_pass_rows": step_exposure.get("rows")
        == expected["training_step_rows"]
        == 2 * train_rows,
        "two_pass_batches": step_exposure.get("batches")
        == expected["training_step_batches"],
        "exposure_tokens": isinstance(step_exposure.get("input_tokens"), int)
        and isinstance(step_exposure.get("supervised_tokens"), int)
        and step_exposure.get("input_tokens", 0)
        >= step_exposure.get("supervised_tokens", 0)
        > 0,
        "milestone_plan": summary.get("milestone_steps")
        == expected_milestone_steps,
        "milestone_root": Path(str(summary.get("milestone_output_dir"))).resolve()
        == milestones_root.resolve(),
        "restart_policy": summary.get("restart_save_steps") == 25
        and summary.get("restart_save_total_limit") == 2,
        "pass_normalized_lr": summary.get("lr_schedule")
        == "linear_sampled_rows_over_2N"
        and summary.get("pass_normalized_total_rows") == 2 * train_rows
        and summary.get("sampled_rows_at_completion") == 2 * train_rows,
        "restart_contract": (summary.get("restart_run_binding") or {}).get(
            "external_contract"
        )
        == record["restart_contract_value"],
        "recipe": summary.get("seed") == SEED
        and summary.get("max_length") == 8192
        and summary.get("epochs") == 2.0
        and summary.get("batch_size") == 1
        and summary.get("gradient_accumulation_steps") == 4
        and summary.get("learning_rate") == 0.0001
        and summary.get("lora_rank") == 16
        and summary.get("lora_alpha") == 16
        and summary.get("lora_dropout") == 0.0,
        "no_preparation_failures": not summary.get("train_preparation_failures")
        and not summary.get("validation_preparation_failures"),
        "single_process": summary.get("world_size") == 1,
        "finite_loss": isinstance(summary.get("training_loss"), (int, float))
        and math.isfinite(float(summary["training_loss"])),
        "research_only": summary.get("release_evidence") is False,
    }
    if not all(checks.values()):
        raise ValueError(
            "training curve gate failed: "
            f"{[key for key, value in checks.items() if not value]}"
        )
    job_root = Path(config["cell_root"]) / "runs" / arm / f"job-{job_id}"
    gpu_attempts = _attempt_gpu_receipts(job_root)
    gate = {
        "contract": CONTRACT,
        "stage": "training_curve_gate",
        "arm": arm,
        "checks": checks,
        "summary": str(summary_path.resolve()),
        "summary_sha256": sha256(summary_path),
        "milestones": milestones,
        "checkpoint_plan": record["checkpoint_plan"],
        "training_step_train_exposure": step_exposure,
        "gpu_attempts": gpu_attempts,
        "observed_training_gpu_names": sorted(
            {item["identity"]["name"] for item in gpu_attempts}
        ),
        "hardware_match_claim": False,
        "release_evidence": False,
    }
    atomic_json(output, gate, exclusive=True)
    return gate


def publish_training(
    config_path: Path,
    expected_sha: str,
    arm: str,
    job_id: str,
    gate_path: Path,
) -> dict[str, Any]:
    config, _, _ = verify_build(config_path, expected_sha)
    gate = load_json(gate_path)
    if gate.get("arm") != arm or sha256(gate["summary"]) != gate["summary_sha256"]:
        raise ValueError("invalid training curve gate")
    pointer = {
        "contract": CONTRACT,
        "stage": "training_complete",
        "arm": arm,
        "job_id": job_id,
        "training_gate": str(gate_path.resolve()),
        "training_gate_sha256": sha256(gate_path),
        "milestones": gate["milestones"],
        "release_evidence": False,
    }
    target = (
        Path(config["cell_root"])
        / "runs"
        / arm
        / f"job-{job_id}"
        / "training.completed.json"
    )
    atomic_json(target, pointer, exclusive=True)
    return pointer


def verify_training(
    config_path: Path, expected_sha: str, arm: str, job_id: str
) -> dict[str, Any]:
    config, _, receipt = verify_build(config_path, expected_sha)
    pointer = load_json(
        Path(config["cell_root"])
        / "runs"
        / arm
        / f"job-{job_id}"
        / "training.completed.json"
    )
    if pointer.get("arm") != arm or pointer.get("job_id") != job_id:
        raise ValueError("training completion identity mismatch")
    gate_path = Path(pointer["training_gate"])
    if sha256(gate_path) != pointer["training_gate_sha256"]:
        raise ValueError("training completion gate hash mismatch")
    gate = load_json(gate_path)
    for label, value in gate["milestones"].items():
        current = _milestone_receipt(
            Path(value["marker"]), receipt["arms"][arm], label
        )
        if current != value:
            raise ValueError("training milestone evidence changed")
    return pointer


def gate_evaluation(
    config_path: Path,
    expected_sha: str,
    arm: str,
    job_id: str,
    milestone: str,
    evaluation_path: Path,
    output: Path,
) -> dict[str, Any]:
    config, _, receipt = verify_build(config_path, expected_sha)
    training = verify_training(config_path, expected_sha, arm, job_id)
    checkpoint = training["milestones"][milestone]
    report = load_json(evaluation_path)
    episodes = report.get("per_episode")
    expected_label = f"exposure-curve-{arm}-{milestone}"
    scenarios = config["inputs"]["evaluation_scenarios"]
    record = receipt["arms"][arm]
    checks = {
        "schema": report.get("evaluation_schema_version") == 2,
        "label": report.get("label") == expected_label,
        "model": (report.get("model_id"), report.get("model_revision"))
        == (record["model_id"], record["model_revision"]),
        "adapter": Path(str(report.get("adapter"))).resolve()
        == Path(checkpoint["adapter"]).resolve(),
        "scenario_hash": report.get("scenarios_sha256") == scenarios["sha256"],
        "episodes": report.get("episodes") == 65
        and isinstance(episodes, list)
        and len(episodes) == 65,
        "full_action_arguments": isinstance(episodes, list)
        and all(
            isinstance(row.get("actions"), list)
            and all(
                isinstance(action, Mapping)
                and isinstance(action.get("arguments"), Mapping)
                for action in row["actions"]
            )
            for row in episodes
        ),
        "identity": isinstance(episodes, list)
        and all(
            row.get("scenario_id")
            and row.get("root_scenario_id")
            and row.get("physical_root_fingerprint")
            for row in episodes
        ),
        "guards_disabled": report.get("guards_enabled") is False
        and report.get("release_evidence") is False,
    }
    if not all(checks.values()):
        raise ValueError(
            f"evaluation gate failed: {[key for key, value in checks.items() if not value]}"
        )
    gate = {
        "contract": CONTRACT,
        "stage": "checkpoint_evaluation_gate",
        "arm": arm,
        "milestone": milestone,
        "checks": checks,
        "checkpoint": checkpoint,
        "evaluation": str(evaluation_path.resolve()),
        "evaluation_sha256": sha256(evaluation_path),
        "training_completion": training,
        "release_evidence": False,
    }
    atomic_json(output, gate, exclusive=True)
    return gate


def publish_evaluation(
    config_path: Path,
    expected_sha: str,
    arm: str,
    job_id: str,
    milestone: str,
    gate_path: Path,
) -> dict[str, Any]:
    config, _, _ = verify_build(config_path, expected_sha)
    gate = load_json(gate_path)
    if gate.get("arm") != arm or gate.get("milestone") != milestone:
        raise ValueError("evaluation gate identity mismatch")
    pointer = {
        "contract": CONTRACT,
        "stage": "checkpoint_evaluation_complete",
        "arm": arm,
        "job_id": job_id,
        "milestone": milestone,
        "evaluation_gate": str(gate_path.resolve()),
        "evaluation_gate_sha256": sha256(gate_path),
        "evaluation": gate["evaluation"],
        "evaluation_sha256": gate["evaluation_sha256"],
        "release_evidence": False,
    }
    target = (
        Path(config["cell_root"])
        / "runs"
        / arm
        / f"job-{job_id}"
        / "evaluations"
        / milestone
        / "completed.json"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(target, pointer, exclusive=True)
    return pointer


def verify_evaluation(
    config_path: Path,
    expected_sha: str,
    arm: str,
    job_id: str,
    milestone: str,
) -> dict[str, Any]:
    config, _, _ = verify_build(config_path, expected_sha)
    pointer = load_json(
        Path(config["cell_root"])
        / "runs"
        / arm
        / f"job-{job_id}"
        / "evaluations"
        / milestone
        / "completed.json"
    )
    if (
        pointer.get("arm") != arm
        or pointer.get("job_id") != job_id
        or pointer.get("milestone") != milestone
    ):
        raise ValueError("checkpoint evaluation completion identity mismatch")
    for key in ("evaluation_gate", "evaluation"):
        if sha256(pointer[key]) != pointer[f"{key}_sha256"]:
            raise ValueError(f"checkpoint evaluation {key} changed")
    return pointer


def publish_arm(
    config_path: Path, expected_sha: str, arm: str, job_id: str
) -> dict[str, Any]:
    config, _, _ = verify_build(config_path, expected_sha)
    training = verify_training(config_path, expected_sha, arm, job_id)
    evaluations = {
        label: verify_evaluation(config_path, expected_sha, arm, job_id, label)
        for label in CHECKPOINT_TARGETS
    }
    pointer = {
        "contract": CONTRACT,
        "stage": "arm_complete",
        "arm": arm,
        "job_id": job_id,
        "training": training,
        "evaluations": evaluations,
        "release_evidence": False,
    }
    target = (
        Path(config["cell_root"])
        / "runs"
        / arm
        / f"job-{job_id}"
        / "completed.json"
    )
    atomic_json(target, pointer, exclusive=True)
    return pointer


def verify_arm(
    config_path: Path, expected_sha: str, arm: str, job_id: str
) -> dict[str, Any]:
    config, _, _ = verify_build(config_path, expected_sha)
    pointer = load_json(
        Path(config["cell_root"])
        / "runs"
        / arm
        / f"job-{job_id}"
        / "completed.json"
    )
    if pointer.get("arm") != arm or pointer.get("job_id") != job_id:
        raise ValueError("arm completion identity mismatch")
    verify_training(config_path, expected_sha, arm, job_id)
    for label in CHECKPOINT_TARGETS:
        verify_evaluation(config_path, expected_sha, arm, job_id, label)
    return pointer


def evaluation_values(
    config_path: Path, expected_sha: str, arm: str, job_id: str
) -> list[str]:
    pointer = verify_arm(config_path, expected_sha, arm, job_id)
    return [
        f"exposure-curve-{arm}-{label}\t{pointer['evaluations'][label]['evaluation']}"
        for label in CHECKPOINT_TARGETS
    ]


def gate_audit(
    config_path: Path,
    expected_sha: str,
    arm: str,
    parent_job: str,
    full: Path,
    summary: Path,
    output: Path,
) -> dict[str, Any]:
    config, _, _ = verify_build(config_path, expected_sha)
    completed = verify_arm(config_path, expected_sha, arm, parent_job)
    report = load_json(full)
    compact = load_json(summary)
    checks: dict[str, bool] = {
        "scenario_hash": report.get("scenarios_sha256")
        == config["inputs"]["evaluation_scenarios"]["sha256"],
        "evaluation_set": set(report.get("evaluations", {}))
        == {f"exposure-curve-{arm}-{label}" for label in CHECKPOINT_TARGETS},
        "compact_bound": compact.get("full_report_sha256") == sha256(full),
        "expert_baseline": compact.get("expert_baseline_problem_episodes") == 0,
        "research_only": report.get("release_evidence") is False,
    }
    physical: dict[str, Any] = {}
    for label in CHECKPOINT_TARGETS:
        key = f"exposure-curve-{arm}-{label}"
        evaluation = report.get("evaluations", {}).get(key, {})
        metrics = evaluation.get("summary", {})
        source = evaluation.get("source_evaluation", {})
        expected_eval = completed["evaluations"][label]
        checks[f"{label}_evaluation_hash"] = (
            source.get("sha256") == expected_eval["evaluation_sha256"]
        )
        checks[f"{label}_episodes"] = (
            metrics.get("episodes") == 65
            and metrics.get("physical_assessable_episodes") == 65
        )
        checks[f"{label}_replay"] = (
            metrics.get("replay_mismatch_episodes") == 0
            and metrics.get("unclassified_episodes") == 0
        )
        checks[f"{label}_identity"] = (
            evaluation.get("source_scenario_binding")
            == "sha256_and_per_episode_identity"
            and metrics.get("scenario_alignment_counts")
            == {"validated_physical_root_fingerprint": 65}
        )
        physical[label] = metrics
    if not all(checks.values()):
        raise ValueError(
            f"physical audit gate failed: {[key for key, value in checks.items() if not value]}"
        )
    gate = {
        "contract": CONTRACT,
        "stage": "physical_curve_audit_gate",
        "arm": arm,
        "checks": checks,
        "full_report": str(full.resolve()),
        "full_report_sha256": sha256(full),
        "summary_report": str(summary.resolve()),
        "summary_report_sha256": sha256(summary),
        "physical_summaries": physical,
        "release_evidence": False,
    }
    atomic_json(output, gate, exclusive=True)
    return gate


def validate_submission_jobs(payload: Mapping[str, Any]) -> None:
    selected = payload.get("selected_arms")
    if (
        not isinstance(selected, list)
        or any(not isinstance(arm, str) for arm in selected)
        or parse_selected_arms(",".join(selected)) != selected
    ):
        raise ValueError("submission receipt selected_arms are invalid")
    expected = {"build"}
    for arm in selected:
        expected.update({f"arm:{arm}", f"audit:{arm}"})
    jobs = payload.get("jobs")
    if not isinstance(jobs, Mapping) or set(jobs) != expected:
        raise ValueError("submission jobs do not match the selected six-arm graph")
    job_ids = list(jobs.values())
    if (
        any(not isinstance(value, str) or not value.isdigit() for value in job_ids)
        or len(set(job_ids)) != len(job_ids)
    ):
        raise ValueError("submission job IDs must be unique decimal strings")


def submission_update(
    config_path: Path, expected_sha: str, action: str, **kwargs: str
) -> dict[str, Any]:
    config, config_sha = validate_config(config_path, expected_sha)
    target = Path(config["cell_root"]) / "submission.json"
    if action == "begin":
        validate_scheduler_union(kwargs["gpu_constraints"])
        payload = {
            "contract": CONTRACT,
            "status": "submitting",
            "config": str(config_path.resolve()),
            "config_sha256": config_sha,
            "source_commit": config["source_commit"],
            "source_tree_sha256": config["source_tree_sha256"],
            "gpu_constraints": kwargs["gpu_constraints"],
            "selected_arms": parse_selected_arms(kwargs["selected_arms"]),
            "jobs": {},
            "release_evidence": False,
        }
        atomic_json(target, payload, exclusive=True)
        return payload
    payload = load_json(target)
    if payload.get("config_sha256") != config_sha:
        raise ValueError("submission receipt configuration mismatch")
    if action == "job":
        role = kwargs["role"]
        arm = kwargs.get("arm")
        if role in {"arm", "audit"} and arm not in payload["selected_arms"]:
            raise ValueError("refusing to record an unselected arm")
        key = role + (f":{arm}" if arm else "")
        if key in payload["jobs"]:
            raise ValueError("submission job already recorded")
        payload["jobs"][key] = kwargs["job_id"]
    elif action in {"finish", "fail"}:
        if action == "finish":
            validate_submission_jobs(payload)
        payload["status"] = "submitted" if action == "finish" else "submission_failed"
        if kwargs.get("detail"):
            payload["detail"] = kwargs["detail"]
    atomic_json(target, payload)
    return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)
    tree = sub.add_parser("tree-digest")
    tree.add_argument("--root", required=True, type=Path)
    digest = sub.add_parser("sha256")
    digest.add_argument("--path", required=True, type=Path)
    environment = sub.add_parser("environment")
    environment.add_argument("--config", required=True, type=Path)
    environment.add_argument("--expected-config-sha", required=True)
    snapshot = sub.add_parser("snapshot-digest")
    snapshot.add_argument("--hf-home", required=True, type=Path)
    snapshot.add_argument("--lane", required=True, choices=MODELS)
    cache = sub.add_parser("verify-model-cache")
    cache.add_argument("--config", required=True, type=Path)
    cache.add_argument("--expected-config-sha", required=True)
    cache.add_argument("--arm", required=True, choices=ARMS)
    for name in ("config-values", "verify-build"):
        item = sub.add_parser(name)
        item.add_argument("--config", required=True, type=Path)
        item.add_argument("--expected-config-sha", required=True)
    item = sub.add_parser("build")
    item.add_argument("--config", required=True, type=Path)
    item.add_argument("--expected-config-sha", required=True)
    item.add_argument("--attempt", required=True, type=Path)
    item = sub.add_parser("arm-values")
    item.add_argument("--config", required=True, type=Path)
    item.add_argument("--expected-config-sha", required=True)
    item.add_argument("--arm", required=True, choices=ARMS)
    item = sub.add_parser("find-resume")
    item.add_argument("--config", required=True, type=Path)
    item.add_argument("--expected-config-sha", required=True)
    item.add_argument("--arm", required=True, choices=ARMS)
    item.add_argument("--job-id", required=True)
    item.add_argument("--restart", required=True, type=int)
    item = sub.add_parser("gate-training")
    item.add_argument("--config", required=True, type=Path)
    item.add_argument("--expected-config-sha", required=True)
    item.add_argument("--arm", required=True, choices=ARMS)
    item.add_argument("--job-id", required=True)
    item.add_argument("--summary", required=True, type=Path)
    item.add_argument("--output", required=True, type=Path)
    item = sub.add_parser("publish-training")
    item.add_argument("--config", required=True, type=Path)
    item.add_argument("--expected-config-sha", required=True)
    item.add_argument("--arm", required=True, choices=ARMS)
    item.add_argument("--job-id", required=True)
    item.add_argument("--gate", required=True, type=Path)
    for name in ("verify-training", "publish-arm", "verify-arm", "evaluation-values"):
        item = sub.add_parser(name)
        item.add_argument("--config", required=True, type=Path)
        item.add_argument("--expected-config-sha", required=True)
        item.add_argument("--arm", required=True, choices=ARMS)
        item.add_argument("--job-id", required=True)
    item = sub.add_parser("gate-evaluation")
    item.add_argument("--config", required=True, type=Path)
    item.add_argument("--expected-config-sha", required=True)
    item.add_argument("--arm", required=True, choices=ARMS)
    item.add_argument("--job-id", required=True)
    item.add_argument("--milestone", required=True, choices=CHECKPOINT_TARGETS)
    item.add_argument("--evaluation", required=True, type=Path)
    item.add_argument("--output", required=True, type=Path)
    item = sub.add_parser("publish-evaluation")
    item.add_argument("--config", required=True, type=Path)
    item.add_argument("--expected-config-sha", required=True)
    item.add_argument("--arm", required=True, choices=ARMS)
    item.add_argument("--job-id", required=True)
    item.add_argument("--milestone", required=True, choices=CHECKPOINT_TARGETS)
    item.add_argument("--gate", required=True, type=Path)
    item = sub.add_parser("verify-evaluation")
    item.add_argument("--config", required=True, type=Path)
    item.add_argument("--expected-config-sha", required=True)
    item.add_argument("--arm", required=True, choices=ARMS)
    item.add_argument("--job-id", required=True)
    item.add_argument("--milestone", required=True, choices=CHECKPOINT_TARGETS)
    item = sub.add_parser("gate-audit")
    item.add_argument("--config", required=True, type=Path)
    item.add_argument("--expected-config-sha", required=True)
    item.add_argument("--arm", required=True, choices=ARMS)
    item.add_argument("--parent-job", required=True)
    item.add_argument("--full", required=True, type=Path)
    item.add_argument("--summary", required=True, type=Path)
    item.add_argument("--output", required=True, type=Path)
    for name in (
        "submission-begin",
        "submission-job",
        "submission-finish",
        "submission-fail",
    ):
        item = sub.add_parser(name)
        item.add_argument("--config", required=True, type=Path)
        item.add_argument("--expected-config-sha", required=True)
        if name == "submission-begin":
            item.add_argument("--gpu-constraints", required=True)
            item.add_argument("--selected-arms", required=True)
        elif name == "submission-job":
            item.add_argument("--role", choices=("build", "arm", "audit"), required=True)
            item.add_argument("--arm", choices=ARMS)
            item.add_argument("--job-id", required=True)
        elif name == "submission-fail":
            item.add_argument("--detail", default="sbatch failed")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "tree-digest":
        print(tree_digest(args.root))
        return 0
    if args.command == "sha256":
        print(sha256(args.path))
        return 0
    if args.command == "snapshot-digest":
        print(json.dumps(snapshot_manifest(args.hf_home, args.lane), sort_keys=True))
        return 0
    if args.command == "environment":
        config, _ = validate_config(args.config, args.expected_config_sha)
        print(json.dumps(environment_receipt(config), sort_keys=True))
        return 0
    if args.command == "verify-model-cache":
        config, _ = validate_config(args.config, args.expected_config_sha)
        lane = ARMS[args.arm]["lane"]
        print(json.dumps(verify_model_snapshot(config, lane), sort_keys=True))
        return 0
    if args.command == "config-values":
        config, _ = validate_config(args.config, args.expected_config_sha)
        print(config["cell_root"])
        print(config["source_root"])
        print(config["python"])
        print(config["hf_home"])
        return 0
    if args.command == "build":
        print(json.dumps(build(args.config, args.expected_config_sha, args.attempt)))
        return 0
    if args.command == "verify-build":
        _, pointer, _ = verify_build(args.config, args.expected_config_sha)
        print(json.dumps(pointer))
        return 0
    if args.command == "arm-values":
        print("\n".join(arm_values(args.config, args.expected_config_sha, args.arm)))
        return 0
    if args.command == "find-resume":
        print(
            find_resume_checkpoint(
                args.config,
                args.expected_config_sha,
                args.arm,
                args.job_id,
                args.restart,
            )
        )
        return 0
    if args.command == "gate-training":
        gate_training(
            args.config,
            args.expected_config_sha,
            args.arm,
            args.summary,
            args.job_id,
            args.output,
        )
        return 0
    if args.command == "publish-training":
        print(
            json.dumps(
                publish_training(
                    args.config,
                    args.expected_config_sha,
                    args.arm,
                    args.job_id,
                    args.gate,
                )
            )
        )
        return 0
    if args.command == "verify-training":
        print(
            json.dumps(
                verify_training(
                    args.config, args.expected_config_sha, args.arm, args.job_id
                )
            )
        )
        return 0
    if args.command == "gate-evaluation":
        gate_evaluation(
            args.config,
            args.expected_config_sha,
            args.arm,
            args.job_id,
            args.milestone,
            args.evaluation,
            args.output,
        )
        return 0
    if args.command == "publish-evaluation":
        print(
            json.dumps(
                publish_evaluation(
                    args.config,
                    args.expected_config_sha,
                    args.arm,
                    args.job_id,
                    args.milestone,
                    args.gate,
                )
            )
        )
        return 0
    if args.command == "verify-evaluation":
        print(
            json.dumps(
                verify_evaluation(
                    args.config,
                    args.expected_config_sha,
                    args.arm,
                    args.job_id,
                    args.milestone,
                )
            )
        )
        return 0
    if args.command == "publish-arm":
        print(
            json.dumps(
                publish_arm(
                    args.config, args.expected_config_sha, args.arm, args.job_id
                )
            )
        )
        return 0
    if args.command == "verify-arm":
        print(
            json.dumps(
                verify_arm(
                    args.config, args.expected_config_sha, args.arm, args.job_id
                )
            )
        )
        return 0
    if args.command == "evaluation-values":
        print(
            "\n".join(
                evaluation_values(
                    args.config, args.expected_config_sha, args.arm, args.job_id
                )
            )
        )
        return 0
    if args.command == "gate-audit":
        gate_audit(
            args.config,
            args.expected_config_sha,
            args.arm,
            args.parent_job,
            args.full,
            args.summary,
            args.output,
        )
        return 0
    action = args.command.removeprefix("submission-")
    values = vars(args)
    payload = submission_update(
        args.config,
        args.expected_config_sha,
        action,
        **{
            key: str(values[key])
            for key in (
                "gpu_constraints",
                "selected_arms",
                "role",
                "arm",
                "job_id",
                "detail",
            )
            if values.get(key) is not None
        },
    )
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
