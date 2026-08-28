"""CPU-only contract tests for the six-arm exposure-curve cell."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.hpc.exposure_curve_20260828 import build as curve
from research.train import canonical_json_sha256, finalize_restart_checkpoint


EXPECTED_PLANS = {
    "A": (1358, 340, 680, (255, 340, 510, 680)),
    "B": (2566, 656, 1284, (482, 642, 963, 1284)),
    "C": (3630, 915, 1816, (681, 908, 1362, 1816)),
    "D": (1313, 350, 658, (247, 329, 494, 658)),
    "E": (2533, 649, 1268, (475, 634, 951, 1268)),
    "F": (3576, 924, 1788, (671, 894, 1341, 1788)),
}


def test_six_arm_factorial_and_exact_two_pass_plans() -> None:
    assert curve.ARMS == {
        "A": {"lane": "e2b", "inclusion": "selective"},
        "B": {"lane": "e2b", "inclusion": "learner_full"},
        "C": {"lane": "e2b", "inclusion": "full_occupancy"},
        "D": {"lane": "12b", "inclusion": "selective"},
        "E": {"lane": "12b", "inclusion": "learner_full"},
        "F": {"lane": "12b", "inclusion": "full_occupancy"},
    }
    for arm, (train_rows, validation_rows, max_steps, steps) in EXPECTED_PLANS.items():
        assert curve.EXPECTED_COUNTS[arm]["train_rows"] == train_rows
        assert curve.EXPECTED_COUNTS[arm]["validation_rows"] == validation_rows
        plan = curve.checkpoint_plan(train_rows)
        assert tuple(value["optimizer_step"] for value in plan.values()) == steps
        assert plan["p100"]["sampled_rows"] == train_rows
        assert plan["p200"]["sampled_rows"] == 2 * train_rows
        assert plan["p200"]["optimizer_step"] == max_steps
        assert plan["p075"]["realized_passes"] >= 0.75
        assert plan["p150"]["realized_passes"] >= 1.5


def test_checkpoint_plan_rejects_nonpositive_or_boolean_rows() -> None:
    for value in (0, -1, True):
        with pytest.raises(ValueError, match="positive integer"):
            curve.checkpoint_plan(value)


def test_teacher_exclusions_and_invalid_retention_are_pinned() -> None:
    assert curve.EXPECTED_MISSING_TARGET_ROWS == {"e2b": 1, "12b": 9}
    assert [curve.EXPECTED_COUNTS[arm]["teacher_abstention_rows"] for arm in "ABC"] == [49] * 3
    assert [curve.EXPECTED_COUNTS[arm]["teacher_abstention_rows"] for arm in "DEF"] == [18] * 3
    assert [
        curve.EXPECTED_COUNTS[arm]["invalid_learner_action_rows_retained"]
        for arm in "ABCDEF"
    ] == [5, 11, 15, 8, 13, 22]


def test_common_split_digest_is_canonical_and_order_independent() -> None:
    assert curve.canonical_split_sha256(["b", "a"]) == curve.canonical_split_sha256(
        ["a", "b"]
    )
    assert len(curve.COMMON_SPLIT_SHA256) == 64


def test_scheduler_union_refuses_h200_only_or_partial_union() -> None:
    accepted = (
        "e2b=a100|h100|h200|rtx6000,12b=a100|h100|h200|rtx6000;"
        "families=e2b=A100|H100|H200|RTX6000,12b=A100|H100|H200|RTX6000"
    )
    curve.validate_scheduler_union(accepted)
    for invalid in (
        "e2b=h200,12b=h200;families=e2b=H200,12b=H200",
        accepted.replace("|rtx6000", ""),
        accepted.replace("|RTX6000", ""),
    ):
        with pytest.raises(ValueError, match="complete authorized GPU union"):
            curve.validate_scheduler_union(invalid)


def test_selected_arms_and_submission_graph_are_fail_closed() -> None:
    assert curve.parse_selected_arms("A,B,C,D,E,F") == list("ABCDEF")
    for invalid in ("", "A,A", "G", "A, B"):
        with pytest.raises(ValueError, match="selected_arms"):
            curve.parse_selected_arms(invalid)
    jobs = {"build": "1"}
    for index, arm in enumerate("ABCDEF", start=2):
        jobs[f"arm:{arm}"] = str(index)
        jobs[f"audit:{arm}"] = str(index + 6)
    curve.validate_submission_jobs({"selected_arms": list("ABCDEF"), "jobs": jobs})
    with pytest.raises(ValueError, match="selected six-arm graph"):
        curve.validate_submission_jobs(
            {"selected_arms": list("ABCDEF"), "jobs": {**jobs, "arm:G": "99"}}
        )


def _record(arm: str = "A") -> dict:
    train_rows, _, max_steps, _ = EXPECTED_PLANS[arm]
    plan = curve.checkpoint_plan(train_rows)
    external = curve.restart_contract(
        arm=arm,
        config_sha256="1" * 64,
        source_tree_sha256="2" * 64,
        aggregate_report_sha256="3" * 64,
        exposure_sha256="4" * 64,
        train_sha256="5" * 64,
        validation_sha256="6" * 64,
        checkpoint_plan_value=plan,
        evaluation_scenarios_sha256="7" * 64,
    )
    lane = curve.ARMS[arm]["lane"]
    return {
        **curve.ARMS[arm],
        "model_id": curve.MODELS[lane][0],
        "model_revision": curve.MODELS[lane][1],
        "train_sha256": "5" * 64,
        "validation_sha256": "6" * 64,
        "max_steps": max_steps,
        "checkpoint_plan": plan,
        "restart_contract_value": external,
    }


def _run_binding(record: dict, arm: str = "A") -> dict:
    train_rows = EXPECTED_PLANS[arm][0]
    validation_rows = EXPECTED_PLANS[arm][1]
    return {
        "model_id": record["model_id"],
        "model_revision": record["model_revision"],
        "train_sha256": record["train_sha256"],
        "validation_sha256": record["validation_sha256"],
        "prepared_train_rows": train_rows,
        "prepared_validation_rows": validation_rows,
        "max_length": 8192,
        "batch_size": 1,
        "gradient_accumulation_steps": 4,
        "learning_rate": 0.0001,
        "lora_rank": 16,
        "lora_alpha": 16,
        "lora_dropout": 0.0,
        "max_steps": record["max_steps"],
        "milestone_steps": {
            label: value["optimizer_step"]
            for label, value in record["checkpoint_plan"].items()
        },
        "restart_save_steps": 25,
        "restart_save_total_limit": 2,
        "seed": curve.SEED,
        "pass_normalized_total_rows": 2 * train_rows,
        "external_contract": record["restart_contract_value"],
    }


def _complete_checkpoint(
    root: Path,
    step: int,
    record: dict,
    *,
    exposure_row_delta: int = 0,
) -> Path:
    checkpoint = root / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True)
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": step}), encoding="utf-8"
    )
    for name in (
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
        "adapter_config.json",
        "adapter_model.safetensors",
    ):
        (checkpoint / name).write_bytes(name.encode("ascii"))
    binding = _run_binding(record)
    ledger = {
        "schema": "research_exposure_checkpoint_v2",
        "global_step": step,
        "run_binding": binding,
        "run_binding_sha256": canonical_json_sha256(binding),
        "training_step_train_exposure": {
            "batches": step * 4,
            "rows": step * 4 + exposure_row_delta,
            "input_tokens": step * 40,
            "supervised_tokens": step * 4,
        },
        "collated_exposure": {},
    }
    finalize_restart_checkpoint(checkpoint, ledger)
    return checkpoint


def test_resume_checkpoint_requires_completed_hash_manifest(tmp_path: Path) -> None:
    record = _record()
    checkpoint = _complete_checkpoint(tmp_path, 25, record)
    assert curve._valid_resume_checkpoint(checkpoint, record) == 25

    (checkpoint / "optimizer.pt").write_bytes(b"mutated")
    assert curve._valid_resume_checkpoint(checkpoint, record) is None


def test_resume_checkpoint_rejects_self_hashed_bad_exposure(tmp_path: Path) -> None:
    record = _record()
    checkpoint = _complete_checkpoint(
        tmp_path, 25, record, exposure_row_delta=-1
    )
    assert curve._valid_resume_checkpoint(checkpoint, record) is None


def test_milestone_receipt_binds_pass_exposure_and_external_contract(
    tmp_path: Path,
) -> None:
    record = _record()
    label = "p075"
    plan = record["checkpoint_plan"][label]
    expected = curve.screening.expected_train_exposure(
        EXPECTED_PLANS["A"][0], int(plan["optimizer_step"]), 1, 4
    )
    binding = _run_binding(record)
    milestone = tmp_path / label
    milestone.mkdir()
    (milestone / "adapter_config.json").write_text("{}", encoding="utf-8")
    (milestone / "adapter_model.safetensors").write_bytes(b"adapter")
    adapter_files = {
        name: {
            "bytes": (milestone / name).stat().st_size,
            "sha256": curve.sha256(milestone / name),
        }
        for name in ("adapter_config.json", "adapter_model.safetensors")
    }
    restart_evidence = milestone / "research_restart_checkpoint_complete.json"
    restart_evidence.write_text(
        json.dumps(
            {
                "schema": "research_checkpoint_complete_v1",
                "global_step": plan["optimizer_step"],
                "run_binding_sha256": canonical_json_sha256(binding),
            }
        ),
        encoding="utf-8",
    )
    marker = {
        "schema": "research_exposure_milestone_v1",
        "label": label,
        "global_step": plan["optimizer_step"],
        "run_binding": binding,
        "run_binding_sha256": canonical_json_sha256(binding),
        "training_step_train_exposure": {
            "batches": expected["training_step_batches"],
            "rows": expected["training_step_rows"],
            "input_tokens": 100,
            "supervised_tokens": 10,
        },
        "adapter_files": adapter_files,
        "restart_checkpoint": str(tmp_path / "checkpoint-255"),
        "restart_checkpoint_manifest_sha256": curve.sha256(restart_evidence),
        "restart_checkpoint_manifest_evidence": restart_evidence.name,
    }
    marker_path = milestone / "research_milestone.json"
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    receipt = curve._milestone_receipt(marker_path, record, label)
    assert receipt["optimizer_step"] == 255
    assert receipt["sampled_rows"] == 1020

    marker["training_step_train_exposure"]["rows"] -= 1
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    with pytest.raises(ValueError, match="milestone p075 gate failed"):
        curve._milestone_receipt(marker_path, record, label)

    marker["training_step_train_exposure"]["rows"] += 1
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    (milestone / "adapter_model.safetensors").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="milestone p075 gate failed"):
        curve._milestone_receipt(marker_path, record, label)

    (milestone / "adapter_model.safetensors").write_bytes(b"adapter")
    restart_evidence.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="milestone p075 gate failed"):
        curve._milestone_receipt(marker_path, record, label)


def test_launcher_hardcodes_union_and_never_selects_partition() -> None:
    script = (
        Path(__file__).parent / "hpc" / "exposure_curve_20260828" / "submit.sh"
    ).read_text(encoding="utf-8")
    assert "a100|h100|h200|rtx6000" in script
    assert "--constraint=\"$GPU_FEATURE_UNION\"" in script
    assert "--partition" not in script
    run_script = (
        Path(__file__).parent / "hpc" / "exposure_curve_20260828" / "run_arm.sh"
    ).read_text(encoding="utf-8")
    for value in (
        "--pass-normalized-total-rows",
        "--restart-save-steps 25",
        "--restart-contract",
        "--milestone-output-dir",
    ):
        assert value in run_script


def test_shell_wrappers_bootstrap_isolated_source_before_importing_builder() -> None:
    script_root = Path(__file__).parent / "hpc" / "exposure_curve_20260828"
    for name in ("submit.sh", "build.sh", "run_arm.sh", "audit.sh"):
        script = (script_root / name).read_text(encoding="utf-8")
        bootstrap = 'export PYTHONPATH="$CONFIG_SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}"'
        assert bootstrap in script
        assert script.index(bootstrap) < script.index('"$SCRIPT_DIR/build.py"')
