from __future__ import annotations

import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
from pathlib import Path
from typing import Any
from unittest import mock

from psse_env.actions import (
    ASK_FOR_MORE_EVIDENCE,
    RECOVERY_OPTIONS_EXHAUSTED_REQUEST,
)
from psse_env.dagger.evaluation_gate import (
    EvaluationGateResult,
    _audited_post_correction_handoff,
    _trace_action_schema_failure,
    current_registry_sha256,
    main as gate_main,
    validate_evaluation_artifact,
)
from psse_env.dagger.release_audit import (
    POST_CORRECTION_COMPLETION_CONTRACT,
    ReleaseAuditTolerances,
)
from psse_env.dagger.evaluator import (
    EVALUATION_SUITES,
    STUDY_EVALUATION_SCHEMA_VERSION,
    fingerprint_evaluation_suites,
    load_evaluation_suites,
    objective_recovery_action_assessment,
    objective_tool_evidence,
    study_objective_episode_evidence_marker,
    trace_progress_evidence,
)
from psse_env.sft.provenance import file_sha256, stable_json_sha256
from psse_env.sft.release_hardware import normalize_accelerator_class
from psse_env.state_store import policy_safe_copy


REPO_ROOT = Path(__file__).resolve().parents[2]
COMMIT = "a" * 40
MODEL_REVISION = "c" * 40
SEED = 19
MAX_STEPS = 8
TEST_POLICY_ID = "test-hard-gate-v3"
TEST_SUITES = ("standard_success", "efficiency")
SUITE_MANIFEST_FIELDS = (
    "suite_manifest",
    "suite_content_hashes",
    "suite_root_set_hashes",
    "suite_content_sha256",
    "root_set_sha256",
)


def _replace_exact_state_identity(
    value: Any, *, original: str, replacement: str
) -> Any:
    if isinstance(value, dict):
        return {
            key: _replace_exact_state_identity(
                item,
                original=original,
                replacement=replacement,
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _replace_exact_state_identity(
                item,
                original=original,
                replacement=replacement,
            )
            for item in value
        ]
    return replacement if value == original else value


def _coordinated_trace_state_substitution(
    episode: dict, *, replacement: str
) -> tuple[str, str]:
    """Rewrite every trace copy and every reproducible trace certificate."""

    original = episode["audit"]["initial_active_state_id"]
    first_before_hash = episode["trace"][0]["state_before_sha256"]
    episode["trace"] = _replace_exact_state_identity(
        episode["trace"],
        original=original,
        replacement=replacement,
    )
    partial_opportunities = episode["evaluation_intervention"][
        "retention_opportunity_count"
    ]
    policy_ordinal = 0
    for row in episode["trace"]:
        row["state_before_sha256"] = stable_json_sha256(row["state_before"])
        row["state_after_sha256"] = stable_json_sha256(row["state_after"])
        row["objective_tool_evidence"] = objective_tool_evidence(
            row["action"], row["policy_tool_output"]
        )
        if row["intervention"] is not True:
            row["observation_hash"] = stable_json_sha256(
                row["policy_observation"]
            )
            row["objective_action_assessment"] = (
                objective_recovery_action_assessment(
                    row["policy_observation"],
                    scenario_family=episode["family"],
                    error_cardinality=episode["cardinality"],
                    partial_success_opportunity=bool(
                        partial_opportunities and policy_ordinal == 0
                    ),
                )
            )
            policy_ordinal += 1
    handoff = episode["audit"].get("post_correction_handoff_assessment")
    if isinstance(handoff, dict):
        episode["audit"]["post_correction_handoff_assessment"] = (
            _replace_exact_state_identity(
                handoff,
                original=original,
                replacement=replacement,
            )
        )
    return original, first_before_hash


def _intervention(suite: str) -> dict:
    if suite == "standard_success":
        return {"intervention_schema_version": 1, "kind": "none"}
    if suite == "forced_error_recovery":
        return {
            "intervention_schema_version": 1,
            "kind": "pre_policy_failure",
            "failure_mode": "well_formed",
            "error_code": "injected_transient_tool_failure",
        }
    if suite == "invalid_action_recovery":
        return {
            "intervention_schema_version": 1,
            "kind": "pre_policy_failure",
            "failure_mode": "malformed",
            "error_code": "injected_invalid_action",
        }
    if suite == "partial_success_retention":
        return {
            "intervention_schema_version": 1,
            "kind": "committed_partial_correction",
            "setup_actions": [
                {
                    "tool": "get_measurement_context",
                    "arguments": {"state_id": "$active"},
                },
                {
                    "tool": "correct_measurements",
                    "arguments": {
                        "state_id": "$active",
                        "measurement_updates": {"0": 0.0},
                    },
                },
                {"tool": "run_wls", "arguments": {"state_id": "$candidate"}},
                {
                    "tool": "commit_state",
                    "arguments": {"candidate_state_id": "$candidate"},
                },
            ],
            "retention_required": True,
        }
    if suite == "efficiency":
        return {
            "intervention_schema_version": 1,
            "kind": "efficiency_budget",
            "limits": {
                "maximum_policy_steps": MAX_STEPS,
                "maximum_wls_calls": 4,
                "maximum_specialized_tool_calls": 4,
            },
        }
    raise AssertionError(suite)


def _suite_payload(
    *,
    names: tuple[str, ...] = TEST_SUITES,
    roots_per_suite: int = 1,
    suffix: str = "",
) -> dict[str, list[dict]]:
    return {
        suite: [
            {
                "scenario_schema_version": 1,
                "execution": {
                    "scenario_id": f"{suite}-case-{index}{suffix}",
                    "case": "ieee14",
                    "measurements": [],
                },
                "audit": {
                    "evaluation_intervention": _intervention(suite),
                    "truth": {
                        "clean_case": "ieee14",
                        "clean_measurements": [],
                        "truth_complete": True,
                    }
                },
                "grouping": {
                    "physical_root_fingerprint": f"{suite}-root-{index}{suffix}",
                    "scenario_family": "no_error",
                    "error_cardinality": 0,
                    "case_id": "ieee14",
                    "split": "test",
                    "source_tier": "frozen-test",
                },
            }
            for index in range(roots_per_suite)
        ]
        for suite in names
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _fingerprint(
    suite_path: Path,
    *,
    required_suites: tuple[str, ...] = TEST_SUITES,
    seed: int = SEED,
    minimum_roots: int | dict[str, int] = 1,
) -> dict:
    return fingerprint_evaluation_suites(
        load_evaluation_suites(suite_path),
        seed=seed,
        required_suites=required_suites,
        minimum_suites=len(required_suites),
        minimum_episodes_per_suite=1,
        minimum_roots_per_suite=minimum_roots,
    )


def _manifest_identity(contract: dict) -> dict:
    return {
        name: copy.deepcopy(contract[name]) for name in SUITE_MANIFEST_FIELDS
    }


def _source_descriptor(path: Path) -> dict:
    return {
        "path": str(path.resolve().relative_to(REPO_ROOT)),
        "location": "repository",
        "sha256": file_sha256(path),
    }


def _policy(suite_path: Path, contract: dict) -> dict:
    source_hash = file_sha256(Path(__file__))
    required_suites = list(contract["suite_names"])
    total_roots = sum(
        int(row["distinct_physical_roots"])
        for row in contract["suite_manifest"].values()
    )
    return {
        "policy_schema_version": 3,
        "policy_id": TEST_POLICY_ID,
        "approved_factories": {
            "environment": [
                {
                    "import_spec": "tests:environment",
                    "source_sha256": source_hash,
                }
            ],
            "expert_policy": [
                {
                    "import_spec": "tests:policy",
                    "source_sha256": source_hash,
                }
            ],
            "model_policy": [
                {
                    "import_spec": "tests:policy",
                    "source_sha256": source_hash,
                }
            ],
            "case_loader": [],
        },
        "role_policy": {
            "expert-baseline": "teacher_release",
            "base-baseline": "identity_and_measurement_only",
            "checkpoint-promotion": "bc0_promotion",
        },
        "suite_policy": {
            "status": "pinned",
            "approved_suite_sha256": file_sha256(suite_path),
            "approved_suite_manifest": _manifest_identity(contract),
            "required_suites": required_suites,
            "evaluator_seed": SEED,
            "max_steps": MAX_STEPS,
            "scenario_schema_version": 1,
            "minimum_physical_roots_per_suite": {
                name: 1 for name in required_suites
            },
        },
        "hard_constraints": {
            "maximum_false_commit_count": 0,
            "maximum_false_finalization_count": 0,
            "maximum_false_rollback_count": 0,
            "maximum_healthy_component_corruption_episodes": 0,
            "maximum_invalid_action_rate": 0.2,
            "maximum_invalid_actions_per_episode": 1,
            "maximum_loop_episode_rate": 0.05,
            "maximum_steps_per_episode": MAX_STEPS,
            "minimum_terminal_rate": 1.0,
        },
        "family_policy": {
            "no_error": {
                "minimum_physical_roots": total_roots,
                "minimum_audited_completion_rate": 1.0,
                "maximum_unqualified_operator_escalation_rate": 0.0,
            }
        },
    }


def _policy_identity(
    *,
    explicit_identity: str | None,
    model_id: str | None,
    model_revision: str | None,
) -> dict:
    return {
        "explicit_policy_identity": explicit_identity,
        "model_id": model_id,
        "model_revision": model_revision,
    }


def _runtime_environment(
    *,
    accelerator_name: str | None,
) -> dict:
    if accelerator_name is None:
        accelerator = {
            "backend": "cpu",
            "cuda_available": False,
            "torch_cuda_version": None,
            "driver_version": None,
            "device_count": 0,
            "bf16_supported": False,
            "devices": [],
        }
    else:
        total_memory_bytes = 143_771_721_728
        accelerator = {
            "backend": "cuda",
            "cuda_available": True,
            "torch_cuda_version": "12.8",
            "driver_version": "570.124.06",
            "device_count": 1,
            "bf16_supported": True,
            "devices": [
                {
                    "index": 0,
                    "name": accelerator_name,
                    "total_memory_bytes": total_memory_bytes,
                    "compute_capability": [9, 0],
                    "accelerator_class": normalize_accelerator_class(
                        accelerator_name,
                        total_memory_bytes,
                    ),
                }
            ],
        }
    return {
        "python_implementation": "CPython",
        "python_version": "3.12.0",
        "platform": "test-platform",
        "packages": {"torch": "2.10.0+cu128"},
        "accelerator": accelerator,
    }


def _canonical_policy_trace_row(
    index: int,
    *,
    tool: str,
    arguments: dict | None = None,
    advanced: bool = True,
    terminal_outcome: str | None = None,
) -> dict:
    default_arguments = (
        {"state_id": "fixture-active"} if tool == "run_wls" else {}
    )
    before = {
        "active_state_id": "fixture-active",
        "candidate_state_id": None,
        "accepted_corrections": [],
        "explained_anomalies": [],
        "trace_cursor": index,
    }
    after = {**before, "trace_cursor": index + 1}
    terminal = terminal_outcome is not None
    progress = trace_progress_evidence(
        before=before,
        after=after,
        output={"state_mutated": bool(advanced and not terminal)},
        terminal=terminal,
    )
    action = {
        "tool": tool,
        "arguments": default_arguments if arguments is None else arguments,
    }
    tool_output = {
        "execution_status": "success",
        "error_code": None,
        "state_mutated": bool(advanced and not terminal),
    }
    if tool == "run_wls":
        state_hash = stable_json_sha256(["runtime-state", index])
        tool_output["tool_metrics"] = {
            "state_id": action["arguments"]["state_id"],
            "state_hash": state_hash,
        }
    prior_history = []
    for prior_index in range(index):
        prior_action = copy.deepcopy(action)
        prior_output = copy.deepcopy(tool_output)
        prior_output["state_mutated"] = bool(advanced)
        if tool == "run_wls":
            prior_hash = stable_json_sha256(["runtime-state", prior_index])
            prior_output["tool_metrics"]["state_hash"] = prior_hash
        prior_history.append(
            {
                "state_id": "fixture-active",
                "candidate_state_id": None,
                "action": prior_action,
                "tool_output": prior_output,
            }
        )
    observation = {
        "active_state_id": "fixture-active",
        "candidate_state_id": None,
        "remaining_budget": max(MAX_STEPS - index, 0),
        "history_window": prior_history[-4:],
        "last_tool": (
            prior_history[-1]["action"]["tool"] if prior_history else None
        ),
        "last_tool_status": (
            prior_history[-1]["tool_output"]["execution_status"]
            if prior_history
            else None
        ),
        "last_tool_output": (
            copy.deepcopy(prior_history[-1]["tool_output"])
            if prior_history
            else {}
        ),
    }
    runtime_hash = (
        tool_output.get("tool_metrics", {}).get("state_hash")
        if tool == "run_wls"
        else None
    )
    return {
        "step": index,
        "intervention": False,
        "observation_hash": stable_json_sha256(observation),
        "policy_observation": observation,
        "objective_action_assessment": objective_recovery_action_assessment(
            observation,
            scenario_family="no_error",
            error_cardinality=0,
        ),
        "policy_tool_output": tool_output,
        "objective_tool_evidence": objective_tool_evidence(action, tool_output),
        "action": action,
        "execution_status": "success",
        "advanced": advanced,
        "error_code": None,
        "candidate_disposition_offline": None,
        "tool_regret": None,
        "runtime_state_hash": runtime_hash,
        "terminal_outcome": terminal_outcome,
        **progress,
    }


def _episode(
    manifest_row: dict,
    *,
    identity: dict,
    performance_ok: bool,
) -> dict:
    terminal = bool(performance_ok)
    outcome = "resolved" if performance_ok else None
    checks = {
        "accepted_correction_targets": {"status": "passed"},
        "healthy_measurements_preserved": {"status": "passed"},
        "healthy_case_components_preserved": {"status": "passed"},
        "accepted_target_nonregression": {"status": "passed"},
    }
    if performance_ok:
        checks.update(
            {
                "remaining_true_faults": {"status": "passed"},
                "final_measurements_match_clean": {"status": "passed"},
                "final_case_matches_clean": {"status": "passed"},
            }
        )
    intervention = copy.deepcopy(manifest_row["evaluation_intervention"])
    kind = intervention["kind"]
    pre_policy_steps = (
        1
        if kind == "pre_policy_failure"
        else len(intervention.get("setup_actions") or [])
        if kind == "committed_partial_correction"
        else 0
    )
    # Policy-action metrics exclude evaluator-injected challenge transitions.
    invalid_actions = 0
    retention_opportunities = int(kind == "committed_partial_correction")
    active_state_id = f"{manifest_row['episode_key']}:active"
    candidate_state_id = f"{manifest_row['episode_key']}:candidate"
    current_state: dict = {
        "active_state_id": active_state_id,
        "candidate_state_id": None,
        "accepted_corrections": [],
        "explained_anomalies": [],
        "trace_cursor": 0,
    }

    def progress_to(
        after: dict,
        *,
        state_mutated: bool,
        terminal_after: bool = False,
    ) -> dict:
        nonlocal current_state
        evidence = trace_progress_evidence(
            before=current_state,
            after=after,
            output={"state_mutated": state_mutated},
            terminal=terminal_after,
        )
        current_state = copy.deepcopy(after)
        return evidence

    if kind == "pre_policy_failure":
        injected_action = (
            {"tool": "run_wls", "arguments": {"state_id": active_state_id}}
            if intervention.get("failure_mode") == "well_formed"
            else {
                "tool": "__invalid_action__",
                "arguments": {"error_code": intervention["error_code"]},
            }
        )
        pre_trace = [
            {
                "step": 0,
                "intervention": True,
                "observation_hash": None,
                "action": injected_action,
                "execution_status": "failure",
                "advanced": False,
                "error_code": intervention["error_code"],
                "candidate_disposition_offline": None,
                "tool_regret": None,
                "runtime_state_hash": None,
                "terminal_outcome": None,
                **progress_to(copy.deepcopy(current_state), state_mutated=False),
            }
        ]
    elif kind == "committed_partial_correction":
        pre_trace = []
        for index, action in enumerate(intervention["setup_actions"]):
            resolved = copy.deepcopy(action)
            for field, value in list(resolved["arguments"].items()):
                if value == "$active":
                    resolved["arguments"][field] = active_state_id
                elif value == "$candidate":
                    resolved["arguments"][field] = candidate_state_id
            after_state = copy.deepcopy(current_state)
            after_state["trace_cursor"] = int(current_state["trace_cursor"]) + 1
            state_mutated = index in {1, 2, 3}
            if index == 1:
                after_state["candidate_state_id"] = candidate_state_id
            elif index == 2:
                after_state["candidate_verified"] = True
            elif index == 3:
                after_state["active_state_id"] = candidate_state_id
                after_state["candidate_state_id"] = None
                after_state["accepted_corrections"] = [
                    {"candidate_state_id": candidate_state_id}
                ]
            pre_trace.append(
                {
                    "step": index,
                    "intervention": True,
                    "observation_hash": None,
                    "action": resolved,
                    "execution_status": "success",
                    "advanced": index != 0,
                    "error_code": None,
                    "candidate_disposition_offline": (
                        "ACCEPT_PARTIAL"
                        if index == len(intervention["setup_actions"]) - 1
                        else None
                    ),
                    "tool_regret": None,
                    "runtime_state_hash": None,
                    "terminal_outcome": None,
                    **progress_to(
                        after_state,
                        state_mutated=state_mutated,
                    ),
                }
            )
    else:
        pre_trace = []
    policy_state_id = str(current_state["active_state_id"])
    policy_tools = (
        {"tool": "run_wls", "arguments": {"state_id": policy_state_id}},
        {"tool": "finalize_diagnosis", "arguments": {}},
    )
    policy_trace = []
    for index, action in enumerate(policy_tools):
        final_policy_row = index == len(policy_tools) - 1
        terminal_after = bool(final_policy_row and terminal)
        after_state = copy.deepcopy(current_state)
        after_state["trace_cursor"] = int(current_state["trace_cursor"]) + 1
        advanced = terminal_after
        policy_trace.append(
            {
                "step": pre_policy_steps + index,
                "intervention": False,
                "observation_hash": stable_json_sha256(
                    [manifest_row["episode_key"], index]
                ),
                "action": action,
                "execution_status": "success",
                "advanced": advanced,
                "error_code": None,
                "candidate_disposition_offline": None,
                "tool_regret": None,
                "runtime_state_hash": None,
                "terminal_outcome": outcome if index == 1 else None,
                **progress_to(
                    after_state,
                    state_mutated=False,
                    terminal_after=terminal_after,
                ),
            }
        )
    handoff_assessment = {
        "assessment_version": POST_CORRECTION_COMPLETION_CONTRACT,
        "status": "not_applicable",
        "eligible": False,
        "reasons": [],
        "actual_terminal_outcome": outcome,
        "runtime_contract": {
            "contract": POST_CORRECTION_COMPLETION_CONTRACT,
            "passed": False,
            "failures": ["handoff_marker_missing"],
            "active_state_id": str(current_state["active_state_id"]),
            "active_state_hash": None,
            "accepted_correction_count": len(
                current_state["accepted_corrections"]
            ),
            "post_correction_confirmation_handoff": False,
        },
        "counterfactual_completion_audit": None,
    }
    trace = [*pre_trace, *policy_trace]
    history: list[dict] = []
    policy_ordinal = 0
    for row in trace:
        output = {
            "execution_status": row["execution_status"],
            "error_code": row["error_code"],
            "state_mutated": row["state_mutated"],
        }
        action = row["action"]
        if (
            action["tool"] in {"run_wls", "verify_candidate"}
            and row["execution_status"] == "success"
        ):
            state_hash = stable_json_sha256(
                [manifest_row["episode_key"], "runtime-state", row["step"]]
            )
            output["tool_metrics"] = {
                "state_id": action["arguments"]["state_id"],
                "state_hash": state_hash,
            }
            row["runtime_state_hash"] = state_hash
        else:
            row["runtime_state_hash"] = None
        row["policy_tool_output"] = copy.deepcopy(output)
        row["objective_tool_evidence"] = objective_tool_evidence(action, output)
        if row["intervention"] is True:
            row["policy_observation"] = None
            row["objective_action_assessment"] = None
        else:
            observation = {
                "active_state_id": row["state_before"]["active_state_id"],
                "candidate_state_id": row["state_before"]["candidate_state_id"],
                "remaining_budget": max(2 - policy_ordinal, 0),
                "history_window": policy_safe_copy(history[-4:]),
                "last_tool": (
                    history[-1]["action"]["tool"] if history else None
                ),
                "last_tool_status": (
                    history[-1]["tool_output"]["execution_status"]
                    if history
                    else None
                ),
                "last_tool_output": (
                    copy.deepcopy(history[-1]["tool_output"])
                    if history
                    else {}
                ),
            }
            row["policy_observation"] = observation
            row["observation_hash"] = stable_json_sha256(observation)
            row["objective_action_assessment"] = (
                objective_recovery_action_assessment(
                    observation,
                    scenario_family=manifest_row["family"],
                    error_cardinality=manifest_row["cardinality"],
                    partial_success_opportunity=bool(
                        retention_opportunities and policy_ordinal == 0
                    ),
                )
            )
            policy_ordinal += 1
        history.append(
            {
                "state_id": row["state_before"]["active_state_id"],
                "candidate_state_id": row["state_before"]["candidate_state_id"],
                "action": copy.deepcopy(action),
                "tool_output": copy.deepcopy(output),
            }
        )
    return {
        "episode_key": manifest_row["episode_key"],
        "scenario_id": manifest_row["scenario_id"],
        "seed": manifest_row["seed"],
        "suite": manifest_row["suite"],
        "family": manifest_row["family"],
        "cardinality": manifest_row["cardinality"],
        "case": manifest_row["case"],
        "split": manifest_row["split"],
        "source_tier": manifest_row["source_tier"],
        "physical_root": manifest_row["physical_root"],
        "steps": 2 + pre_policy_steps,
        "policy_steps": 2,
        "terminal": terminal,
        "terminal_outcome": outcome,
        "final_physical_success": bool(performance_ok),
        "physical_correctness_known": True,
        "final_physical_correct": bool(performance_ok),
        "healthy_preservation_known": True,
        "healthy_components_preserved": True,
        "false_commit_count": 0 if performance_ok else 1,
        "false_rollback_count": 0,
        "false_finalization_count": 0,
        "partial_fix_count": retention_opportunities,
        "retained_partial_fix_count": retention_opportunities,
        "invalid_action_count": invalid_actions,
        "recovered_invalid_action_count": invalid_actions,
        "loop_detected": False,
        "wls_calls": 1,
        "specialized_tool_calls": 0,
        "tool_regret_total": 0.0,
        "tool_regret_samples": 0,
        "evaluation_intervention": {
            "contract": intervention,
            "applied": True,
            "pre_policy_step_count": pre_policy_steps,
            "injected_failure_count": int(kind == "pre_policy_failure"),
            "injected_invalid_action_count": int(
                kind == "pre_policy_failure"
                and intervention.get("failure_mode") == "malformed"
            ),
            "recovered_failure_count": int(
                kind == "pre_policy_failure" and performance_ok
            ),
            "retention_opportunity_count": retention_opportunities,
            "retained_opportunity_count": retention_opportunities,
        },
        "trace": trace,
        "evaluator_error": None,
        "release_environment_attestation": {
            "passed": True,
            "production_dataset_mode": True,
            "candidate_quality_oracle_mode": "deployment",
            "failures": [],
        },
        "policy_identity_attestation": {
            "passed": True,
            "required": copy.deepcopy(identity),
            "actual": copy.deepcopy(identity),
            "failures": [],
        },
        "objective_evidence": study_objective_episode_evidence_marker(),
        "audit": {
            "audit_mode": "strict_release_audit",
            "evidence_complete": True,
            "quarantined": False,
            "initial_active_state_id": active_state_id,
            "final_active_state_id": str(current_state["active_state_id"]),
            "post_correction_handoff_assessment": handoff_assessment,
            "strict_release_audit": {
                "audit_version": "strict_offline_episode_truth_v3",
                "terminal": terminal,
                "terminal_outcome": outcome,
                "scenario_family": manifest_row["family"],
                "physical_root_fingerprint": manifest_row["physical_root"],
                "problems": [],
                "quarantined": False,
                "checks": checks,
            },
        },
    }


def _overall(episodes: list[dict]) -> dict:
    total = len(episodes)
    terminal = sum(row["terminal"] is True for row in episodes)
    resolved = sum(
        row["terminal"] is True
        and row["terminal_outcome"] == "resolved"
        and row["final_physical_success"] is True
        for row in episodes
    )
    escalated = sum(
        row["terminal"] is True
        and row["terminal_outcome"] == "operator_escalation"
        for row in episodes
    )
    audited_handoffs = sum(
        _audited_post_correction_handoff(row)[0] for row in episodes
    )
    audited_completions = resolved + audited_handoffs
    unqualified_escalations = escalated - audited_handoffs
    injected = sum(
        row["evaluation_intervention"]["injected_failure_count"]
        for row in episodes
    )
    recovered_injected = sum(
        row["evaluation_intervention"]["recovered_failure_count"]
        for row in episodes
    )
    return {
        "episodes": total,
        "terminal_episodes": terminal,
        "terminal_rate": terminal / total if total else 0.0,
        "resolved_episodes": resolved,
        "resolution_rate": resolved / total if total else 0.0,
        "operator_escalation_episodes": escalated,
        "operator_escalation_rate": escalated / total if total else 0.0,
        "audited_post_correction_handoff_episodes": audited_handoffs,
        "audited_post_correction_handoff_rate": (
            audited_handoffs / total if total else 0.0
        ),
        "audited_completion_episodes": audited_completions,
        "audited_completion_rate": (
            audited_completions / total if total else 0.0
        ),
        "unqualified_operator_escalation_episodes": (
            unqualified_escalations
        ),
        "unqualified_operator_escalation_rate": (
            unqualified_escalations / total if total else 0.0
        ),
        "false_commit_count": sum(row["false_commit_count"] for row in episodes),
        "false_rollback_count": sum(
            row["false_rollback_count"] for row in episodes
        ),
        "false_finalization_count": sum(
            row["false_finalization_count"] for row in episodes
        ),
        "healthy_component_corruption_episodes": sum(
            row["healthy_preservation_known"] is True
            and row["healthy_components_preserved"] is False
            for row in episodes
        ),
        "invalid_action_count": sum(row["invalid_action_count"] for row in episodes),
        "injected_failure_count": injected,
        "recovered_injected_failures": recovered_injected,
        "injected_failure_recovery_rate": (
            recovered_injected / injected if injected else 0.0
        ),
        "episodes_with_injected_failures": sum(
            row["evaluation_intervention"]["injected_failure_count"] > 0
            for row in episodes
        ),
        "loop_episodes": sum(row["loop_detected"] is True for row in episodes),
        "evaluator_error_episodes": sum(
            row["evaluator_error"] is not None for row in episodes
        ),
    }


def _rehash(artifact: dict) -> None:
    unsigned = copy.deepcopy(artifact)
    unsigned.pop("content_sha256", None)
    artifact["content_sha256"] = stable_json_sha256(unsigned)


def _rehash_provenance(artifact: dict) -> None:
    identity_core = copy.deepcopy(artifact["provenance"])
    for field in ("identity_sha256", "release_eligible", "release_failures"):
        identity_core.pop(field, None)
    artifact["provenance"]["identity_sha256"] = stable_json_sha256(identity_core)
    _rehash(artifact)


def _set_operator_escalation(
    artifact: dict,
    episode_index: int,
    *,
    healthy_summary_known: bool = True,
) -> None:
    suite_metrics = artifact["evaluation"]["suite_metrics"]
    episode = suite_metrics["episodes"][episode_index]
    episode["terminal_outcome"] = "operator_escalation"
    episode["final_physical_success"] = False
    episode["physical_correctness_known"] = False
    episode["final_physical_correct"] = False
    episode["healthy_preservation_known"] = healthy_summary_known
    episode["healthy_components_preserved"] = healthy_summary_known
    episode["audit"]["strict_release_audit"][
        "terminal_outcome"
    ] = "operator_escalation"
    episode["audit"]["post_correction_handoff_assessment"][
        "actual_terminal_outcome"
    ] = "operator_escalation"
    episode["trace"][-1]["terminal_outcome"] = "operator_escalation"
    episode["trace"][-1]["terminal_after"] = True
    suite_metrics["overall"] = _overall(suite_metrics["episodes"])
    _rehash(artifact)


def _set_audited_post_correction_handoff(
    artifact: dict,
    episode_index: int,
) -> None:
    """Replace one resolved fixture with a fully bound audited handoff."""

    _set_operator_escalation(artifact, episode_index)
    suite_metrics = artifact["evaluation"]["suite_metrics"]
    episode = suite_metrics["episodes"][episode_index]
    final_row = episode["trace"][-1]
    accepted_count = final_row["state_after"]["accepted_correction_count"]
    if accepted_count < 1:
        raise AssertionError("audited handoff fixture requires a committed correction")
    active_state_id = final_row["state_after"]["active_state_id"]
    final_row["action"] = {
        "tool": ASK_FOR_MORE_EVIDENCE,
        "arguments": {
            "state_id": active_state_id,
            "request": RECOVERY_OPTIONS_EXHAUSTED_REQUEST,
        },
    }
    final_row["runtime_state_hash"] = final_row["state_after_sha256"]
    passed_check = {"status": "passed", "problems": []}
    counterfactual_checks = {
        "accepted_correction_targets": copy.deepcopy(passed_check),
        "accepted_target_nonregression": {
            **copy.deepcopy(passed_check),
            "target_evidence": [
                {
                    "family": "measurement",
                    "index0": 0,
                    "status": "passed",
                    "initial_distance": 1.0,
                    "final_distance": 0.0,
                    "tolerance": 1.0e-6,
                }
            ],
        },
        "remaining_true_faults": {
            **copy.deepcopy(passed_check),
            "derived_remaining_fault_count": 0,
            "evidence_source": "offline_scenario_truth_derivation",
        },
        "healthy_measurements_preserved": copy.deepcopy(passed_check),
        "healthy_case_components_preserved": copy.deepcopy(passed_check),
        "final_measurements_match_clean": copy.deepcopy(passed_check),
        "final_case_matches_clean": copy.deepcopy(passed_check),
    }
    counterfactual = {
        "audit_version": "strict_offline_episode_truth_v3",
        "scenario_id": episode["scenario_id"],
        "physical_root_fingerprint": episode["physical_root"],
        "scenario_family": episode["family"],
        "terminal": True,
        "terminal_outcome": "resolved",
        "checks": counterfactual_checks,
        "tolerances": asdict(ReleaseAuditTolerances()),
        "problems": [],
        "quarantined": False,
    }
    episode["audit"]["post_correction_handoff_assessment"] = {
        "assessment_version": POST_CORRECTION_COMPLETION_CONTRACT,
        "status": "passed",
        "eligible": True,
        "reasons": [],
        "actual_terminal_outcome": "operator_escalation",
        "runtime_contract": {
            "contract": POST_CORRECTION_COMPLETION_CONTRACT,
            "passed": True,
            "failures": [],
            "active_state_id": active_state_id,
            "active_state_hash": final_row["state_after_sha256"],
            "accepted_correction_count": accepted_count,
            "post_correction_confirmation_handoff": True,
        },
        "counterfactual_completion_audit": counterfactual,
    }
    suite_metrics["overall"] = _overall(suite_metrics["episodes"])
    _rehash(artifact)


def _insert_nonadvancing_repeat(episode: dict) -> None:
    """Insert a genuine repeated action in one no-progress policy epoch."""

    first_policy_index = next(
        index
        for index, row in enumerate(episode["trace"])
        if row["intervention"] is False
    )
    source = episode["trace"][first_policy_index]
    repeated = copy.deepcopy(source)
    repeated["observation_hash"] = stable_json_sha256(
        [episode["episode_key"], "repeated-no-progress"]
    )
    repeated["advanced"] = False
    repeated["terminal_outcome"] = None
    repeated["state_before"] = copy.deepcopy(source["state_after"])
    repeated["state_after"] = copy.deepcopy(source["state_after"])
    repeated["state_before_sha256"] = source["state_after_sha256"]
    repeated["state_after_sha256"] = source["state_after_sha256"]
    repeated["state_mutated"] = False
    repeated["policy_tool_output"]["state_mutated"] = False
    repeated["terminal_after"] = False
    episode["trace"].insert(first_policy_index + 1, repeated)
    for index, row in enumerate(episode["trace"]):
        row["step"] = index
    episode["steps"] += 1
    episode["policy_steps"] += 1
    if repeated["action"]["tool"] in {"run_wls", "verify_candidate"}:
        episode["wls_calls"] += 1
    episode["loop_detected"] = True
    history: list[dict] = []
    policy_ordinal = 0
    partial_opportunity = int(
        episode["evaluation_intervention"]["retention_opportunity_count"]
    )
    for row in episode["trace"]:
        output = row["policy_tool_output"]
        row["objective_tool_evidence"] = objective_tool_evidence(
            row["action"], output
        )
        metrics = output.get("tool_metrics")
        row["runtime_state_hash"] = (
            metrics.get("state_hash") if isinstance(metrics, dict) else None
        )
        if not row["intervention"]:
            observation = copy.deepcopy(row["policy_observation"])
            observation.update(
                {
                    "active_state_id": row["state_before"]["active_state_id"],
                    "candidate_state_id": row["state_before"][
                        "candidate_state_id"
                    ],
                    "history_window": policy_safe_copy(history[-4:]),
                    "last_tool": (
                        history[-1]["action"]["tool"] if history else None
                    ),
                    "last_tool_status": (
                        history[-1]["tool_output"]["execution_status"]
                        if history
                        else None
                    ),
                    "last_tool_output": (
                        copy.deepcopy(history[-1]["tool_output"])
                        if history
                        else {}
                    ),
                }
            )
            row["policy_observation"] = observation
            row["observation_hash"] = stable_json_sha256(observation)
            row["objective_action_assessment"] = (
                objective_recovery_action_assessment(
                    observation,
                    scenario_family=episode["family"],
                    error_cardinality=episode["cardinality"],
                    partial_success_opportunity=bool(
                        partial_opportunity and policy_ordinal == 0
                    ),
                )
            )
            policy_ordinal += 1
        history.append(
            {
                "state_id": row["state_before"]["active_state_id"],
                "candidate_state_id": row["state_before"]["candidate_state_id"],
                "action": copy.deepcopy(row["action"]),
                "tool_output": copy.deepcopy(output),
            }
        )


def _artifact(
    suite_path: Path,
    contract: dict,
    *,
    explicit_identity: str | None = "expert-v1",
    model_id: str | None = None,
    model_revision: str | None = None,
    performance_ok: bool = True,
    accelerator_name: str = "NVIDIA H200",
) -> dict:
    source_file = Path(__file__)
    evaluator_file = REPO_ROOT / "psse_env/dagger/evaluator.py"
    identity = _policy_identity(
        explicit_identity=explicit_identity,
        model_id=model_id,
        model_revision=model_revision,
    )
    episodes = [
        _episode(
            manifest_row,
            identity=identity,
            performance_ok=performance_ok,
        )
        for manifest_row in contract["episode_manifest"]
    ]
    environment_contract = {
        "production_dataset_mode": True,
        "candidate_quality_oracle_mode": "deployment",
    }
    configuration = {
        **copy.deepcopy(contract),
        "seed": SEED,
        "max_steps": MAX_STEPS,
        "required_suites": sorted(contract["suite_names"]),
        "minimum_suites": len(contract["suite_names"]),
        "minimum_episodes_per_suite": 1,
        "minimum_roots_per_suite": 1,
        "release_scenario_schema_validation": {
            "passed": True,
            "scenario_schema_version": 1,
        },
        "release_environment_validation": {
            "passed": True,
            "episodes_checked": len(episodes),
            "required": copy.deepcopy(environment_contract),
            "observed": [copy.deepcopy(environment_contract)],
            "failures": [],
        },
        "policy_identity_validation": {
            "passed": True,
            "episodes_checked": len(episodes),
            "required": copy.deepcopy(identity),
            "observed": [copy.deepcopy(identity)],
            "failures": [],
        },
        "custom_callback_validation": {
            "passed": True,
            "physical_audit_callback": False,
            "tool_cost_callback": False,
        },
    }
    core = {
        "provenance_schema_version": 1,
        "source_state": {
            "source_commit": COMMIT,
            "source_worktree_dirty": False,
            "tracked_diff_hash": "0" * 64,
            "untracked_source_files": [],
            "release_eligible_source": True,
        },
        "input_suite": {
            "provided_path": str(suite_path),
            "resolved_path": str(suite_path.resolve()),
            "sha256": file_sha256(suite_path),
            "size_bytes": suite_path.stat().st_size,
        },
        "factories": {
            "environment": {
                "import_spec": "tests:environment",
                "source": _source_descriptor(source_file),
            },
            "policy": {
                "import_spec": "tests:policy",
                "source": _source_descriptor(source_file),
            },
            "case_loader": None,
        },
        "policy_identity": copy.deepcopy(identity),
        "protocol_registry": {
            "protocol": "canonical",
            "registry_sha256": current_registry_sha256("canonical"),
        },
        "runtime_environment": _runtime_environment(
            accelerator_name=(
                accelerator_name if explicit_identity is None else None
            )
        ),
        "evaluator_source": _source_descriptor(evaluator_file),
    }
    core["identity_sha256"] = stable_json_sha256(core)
    provenance = {**core, "release_eligible": True, "release_failures": []}
    artifact = {
        "artifact_schema_version": STUDY_EVALUATION_SCHEMA_VERSION,
        "artifact_type": "closed_loop_release_evaluation",
        "release_eligible": True,
        "release_failures": [],
        "provenance": provenance,
        "evaluation": {
            "score": -1.0e12,
            "metrics": {},
            "suite_metrics": {
                "schema_version": STUDY_EVALUATION_SCHEMA_VERSION,
                "configuration": configuration,
                "overall": _overall(episodes),
                "episodes": episodes,
            },
        },
    }
    _rehash(artifact)
    # Gate fixtures represent artifacts after the evaluator has persisted and
    # the validator has decoded them, including JSON's string-only object keys.
    return json.loads(json.dumps(artifact))


def _validate(
    artifact: dict,
    *,
    role: str,
    policy: dict,
    suite_path: Path,
    explicit_identity: str | None = "expert-v1",
    model_id: str | None = None,
    model_revision: str | None = None,
    reference_artifact: dict | None = None,
    reference_model_id: str | None = None,
    reference_model_revision: str | None = None,
    required_policy_id: str = TEST_POLICY_ID,
) -> EvaluationGateResult:
    identity_kwargs = (
        {"expected_policy_identity": explicit_identity}
        if explicit_identity is not None
        else {
            "expected_model_id": model_id,
            "expected_model_revision": model_revision,
        }
    )
    return validate_evaluation_artifact(
        artifact,
        role=role,
        policy=policy,
        expected_source_commit=COMMIT,
        expected_suite_path=suite_path,
        expected_protocol="canonical",
        expected_registry_sha256=current_registry_sha256("canonical"),
        reference_artifact=reference_artifact,
        reference_model_id=reference_model_id,
        reference_model_revision=reference_model_revision,
        required_gate_policy_id=required_policy_id,
        repo_root=REPO_ROOT,
        require_current_clean_source=False,
        **identity_kwargs,
    )


class EvaluationGateV3Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.suite_path = self.root / "frozen-suite.json"
        _write_json(self.suite_path, _suite_payload())
        self.contract = _fingerprint(self.suite_path)
        self.policy = _policy(self.suite_path, self.contract)

    def _fixture_for_suite(
        self, suite: str
    ) -> tuple[Path, dict, dict, dict]:
        suite_path = self.root / f"{suite}-suite.json"
        _write_json(suite_path, _suite_payload(names=(suite,)))
        contract = _fingerprint(suite_path, required_suites=(suite,))
        policy = _policy(suite_path, contract)
        return suite_path, contract, policy, _artifact(suite_path, contract)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_exact_pinned_suite_and_expert_artifact_pass(self) -> None:
        result = _validate(
            _artifact(self.suite_path, self.contract),
            role="expert-baseline",
            policy=self.policy,
            suite_path=self.suite_path,
        )
        self.assertTrue(result.passed, result.failures)
        self.assertTrue(result.evidence_passed)
        self.assertTrue(result.performance_passed)
        self.assertTrue(result.performance_enforced)
        self.assertEqual(result.validation_role, "expert-baseline")
        self.assertEqual(result.frozen_suite_sha256, file_sha256(self.suite_path))

    def test_exact_pinned_suite_survives_json_key_normalization(self) -> None:
        suite_path, _contract, policy, artifact = self._fixture_for_suite(
            "partial_success_retention"
        )
        artifact_path = self.root / "persisted-expert-artifact.json"
        _write_json(
            artifact_path,
            artifact,
        )

        result = _validate(
            artifact_path,
            role="expert-baseline",
            policy=policy,
            suite_path=suite_path,
        )

        self.assertTrue(result.passed, result.failures)
        self.assertTrue(result.evidence_passed, result.evidence_failures)

    def test_json_key_normalization_does_not_hide_partial_value_tampering(self) -> None:
        suite_path, _contract, policy, artifact = self._fixture_for_suite(
            "partial_success_retention"
        )
        persisted = json.loads(json.dumps(artifact))
        persisted["evaluation"]["suite_metrics"]["configuration"][
            "episode_manifest"
        ][0]["evaluation_intervention"]["setup_actions"][1]["arguments"][
            "measurement_updates"
        ]["0"] = 1.0
        episode = persisted["evaluation"]["suite_metrics"]["episodes"][0]
        episode["evaluation_intervention"]["contract"]["setup_actions"][1][
            "arguments"
        ]["measurement_updates"]["0"] = 1.0
        episode["trace"][1]["action"]["arguments"]["measurement_updates"][
            "0"
        ] = 1.0
        _rehash(persisted)

        result = _validate(
            persisted,
            role="expert-baseline",
            policy=policy,
            suite_path=suite_path,
        )

        self.assertFalse(result.evidence_passed)
        self.assertTrue(
            any(
                "does not match" in failure
                for failure in result.evidence_failures
            ),
            result.evidence_failures,
        )

    def test_generic_operator_escalation_is_unqualified_not_completion(self) -> None:
        artifact = _artifact(self.suite_path, self.contract)
        _set_operator_escalation(artifact, 0)

        result = _validate(
            artifact,
            role="expert-baseline",
            policy=self.policy,
            suite_path=self.suite_path,
        )

        self.assertTrue(result.evidence_passed, result.evidence_failures)
        self.assertFalse(result.performance_passed)
        self.assertFalse(result.passed)
        overall = artifact["evaluation"]["suite_metrics"]["overall"]
        self.assertEqual(overall["operator_escalation_episodes"], 1)
        self.assertEqual(overall["audited_post_correction_handoff_episodes"], 0)
        self.assertEqual(overall["audited_completion_episodes"], 1)
        self.assertEqual(overall["unqualified_operator_escalation_episodes"], 1)
        self.assertTrue(
            any(
                "audited completion failed" in failure
                for failure in result.performance_failures
            ),
            result.performance_failures,
        )
        self.assertTrue(
            any(
                "unqualified operator escalation failed" in failure
                for failure in result.performance_failures
            ),
            result.performance_failures,
        )

    def test_state_bound_audited_handoff_satisfies_completion_policy(self) -> None:
        suite_path, _contract, policy, artifact = self._fixture_for_suite(
            "partial_success_retention"
        )
        _set_audited_post_correction_handoff(artifact, 0)

        result = _validate(
            artifact,
            role="expert-baseline",
            policy=policy,
            suite_path=suite_path,
        )

        self.assertTrue(result.passed, result.failures)
        overall = artifact["evaluation"]["suite_metrics"]["overall"]
        self.assertEqual(overall["resolved_episodes"], 0)
        self.assertEqual(overall["operator_escalation_episodes"], 1)
        self.assertEqual(overall["audited_post_correction_handoff_episodes"], 1)
        self.assertEqual(overall["audited_completion_episodes"], 1)
        self.assertEqual(overall["unqualified_operator_escalation_episodes"], 0)
        observed = result.observed["families"]["no_error"]
        self.assertEqual(observed["resolved_roots"], 0)
        self.assertEqual(observed["operator_escalation_roots"], 1)
        self.assertEqual(observed["audited_completion_roots"], 1)
        self.assertEqual(observed["unqualified_operator_escalation_roots"], 0)

    def test_audited_handoff_is_recomputed_from_final_trace_binding(self) -> None:
        suite_path, _contract, policy, artifact = self._fixture_for_suite(
            "partial_success_retention"
        )
        _set_audited_post_correction_handoff(artifact, 0)
        episode = artifact["evaluation"]["suite_metrics"]["episodes"][0]
        episode["audit"]["post_correction_handoff_assessment"][
            "runtime_contract"
        ]["active_state_id"] = "forged-active-state"
        artifact["evaluation"]["suite_metrics"]["overall"] = _overall(
            [episode]
        )
        _rehash(artifact)

        result = _validate(
            artifact,
            role="expert-baseline",
            policy=policy,
            suite_path=suite_path,
        )

        self.assertFalse(result.evidence_passed)
        self.assertTrue(
            any(
                "final action state id is unbound" in failure
                for failure in result.evidence_failures
            ),
            result.evidence_failures,
        )

    def test_failed_claimed_handoff_is_evidence_failure(self) -> None:
        suite_path, _contract, policy, artifact = self._fixture_for_suite(
            "partial_success_retention"
        )
        _set_audited_post_correction_handoff(artifact, 0)
        episode = artifact["evaluation"]["suite_metrics"]["episodes"][0]
        assessment = episode["audit"]["post_correction_handoff_assessment"]
        assessment["status"] = "failed"
        assessment["eligible"] = False
        assessment["reasons"] = ["counterfactual_completion_quarantined"]
        artifact["evaluation"]["suite_metrics"]["overall"] = _overall(
            [episode]
        )
        _rehash(artifact)

        result = _validate(
            artifact,
            role="expert-baseline",
            policy=policy,
            suite_path=suite_path,
        )

        self.assertFalse(result.evidence_passed)
        self.assertTrue(
            any(
                "claimed post-correction handoff assessment failed" in failure
                for failure in result.evidence_failures
            ),
            result.evidence_failures,
        )

    def test_audited_handoff_rejects_substituted_runtime_state_hash(self) -> None:
        suite_path, _contract, policy, artifact = self._fixture_for_suite(
            "partial_success_retention"
        )
        _set_audited_post_correction_handoff(artifact, 0)
        episode = artifact["evaluation"]["suite_metrics"]["episodes"][0]
        episode["audit"]["post_correction_handoff_assessment"][
            "runtime_contract"
        ]["active_state_hash"] = "f" * 64
        artifact["evaluation"]["suite_metrics"]["overall"] = _overall(
            [episode]
        )
        _rehash(artifact)

        result = _validate(
            artifact,
            role="expert-baseline",
            policy=policy,
            suite_path=suite_path,
        )

        self.assertFalse(result.evidence_passed)
        self.assertTrue(
            any(
                "final state hash is unbound" in failure
                for failure in result.evidence_failures
            ),
            result.evidence_failures,
        )

    def test_false_rollback_ceiling_cannot_be_weakened(self) -> None:
        policy = copy.deepcopy(self.policy)
        policy["hard_constraints"]["maximum_false_rollback_count"] = 1

        with self.assertRaisesRegex(
            ValueError,
            "maximum_false_rollback_count=0",
        ):
            _validate(
                _artifact(self.suite_path, self.contract),
                role="expert-baseline",
                policy=policy,
                suite_path=self.suite_path,
            )

    def test_episode_intervention_must_match_the_frozen_suite(self) -> None:
        artifact = _artifact(self.suite_path, self.contract)
        episode = artifact["evaluation"]["suite_metrics"]["episodes"][0]
        episode["evaluation_intervention"]["contract"]["kind"] = "forged"
        _rehash(artifact)

        result = _validate(
            artifact,
            role="expert-baseline",
            policy=self.policy,
            suite_path=self.suite_path,
        )
        self.assertFalse(result.evidence_passed)
        self.assertTrue(
            any(
                "intervention does not match the frozen suite" in failure
                for failure in result.evidence_failures
            ),
            result.evidence_failures,
        )

    def test_trace_actions_require_registered_tools_and_schema_valid_arguments(
        self,
    ) -> None:
        for tamper, expected_failure in (
            ("unknown_tool", "is not in the unified release registry"),
            ("unknown_argument", "contains unsupported argument"),
        ):
            with self.subTest(tamper=tamper):
                artifact = _artifact(self.suite_path, self.contract)
                row = artifact["evaluation"]["suite_metrics"]["episodes"][0][
                    "trace"
                ][0]
                if tamper == "unknown_tool":
                    row["action"] = {"tool": "unknown_release_tool", "arguments": {}}
                else:
                    row["action"]["arguments"]["unexpected"] = True
                _rehash(artifact)

                result = _validate(
                    artifact,
                    role="expert-baseline",
                    policy=self.policy,
                    suite_path=self.suite_path,
                )
                self.assertFalse(result.evidence_passed)
                self.assertTrue(
                    any(
                        expected_failure in failure
                        for failure in result.evidence_failures
                    ),
                    result.evidence_failures,
                )

    def test_trace_measurement_correction_accepts_both_execution_forms(self) -> None:
        actions = (
            {
                "tool": "correct_measurements",
                "arguments": {
                    "state_id": "fixture-active",
                    "measurement_updates": {"7": 1.0},
                },
            },
            {
                "tool": "correct_measurements",
                "arguments": {
                    "state_id": "fixture-active",
                    "suspect_group": [7],
                },
            },
        )
        for action in actions:
            with self.subTest(arguments=action["arguments"]):
                self.assertIsNone(_trace_action_schema_failure(action, index=3))

        malformed = {
            "tool": "correct_measurements",
            "arguments": {
                "state_id": "fixture-active",
                "suspect_group": "7",
            },
        }
        failure = _trace_action_schema_failure(malformed, index=3)
        self.assertIsNotNone(failure)
        self.assertIn("must have JSON type array", str(failure))

    def test_trace_target_only_fallback_does_not_hide_malformed_values(self) -> None:
        malformed_actions = (
            {
                "tool": "correct_measurements",
                "arguments": {
                    "state_id": "fixture-active",
                    "measurement_updates": {"7": "not-a-number"},
                },
            },
            {
                "tool": "correct_parameters",
                "arguments": {
                    "state_id": "fixture-active",
                    "line_index": 7,
                    "field": 123,
                    "value": "not-a-number",
                },
            },
            {
                "tool": "correct_topology",
                "arguments": {
                    "state_id": "fixture-active",
                    "line_index": 7,
                    "status": 0,
                    "status_field": 123,
                },
            },
        )
        for action in malformed_actions:
            with self.subTest(tool=action["tool"]):
                self.assertIsNotNone(
                    _trace_action_schema_failure(action, index=4)
                )

        extra_target_only_field = {
            "tool": "correct_measurements",
            "arguments": {
                "state_id": "fixture-active",
                "suspect_group": [7],
                "measurement_updates": {"7": 1.0},
            },
        }
        failure = _trace_action_schema_failure(extra_target_only_field, index=4)
        self.assertIsNotNone(failure)
        self.assertIn("unsupported argument", str(failure))

    def test_trace_rejects_hidden_loop_forged_advancement_and_misplaced_terminal(
        self,
    ) -> None:
        for tamper, expected_failure in (
            ("hidden_loop", "loop_detected does not match"),
            ("forged_advancement", "advanced does not match progress evidence"),
            ("misplaced_terminal", "exactly one matching marker"),
        ):
            with self.subTest(tamper=tamper):
                artifact = _artifact(self.suite_path, self.contract)
                metrics = artifact["evaluation"]["suite_metrics"]
                episode = metrics["episodes"][0]
                if tamper == "hidden_loop":
                    _insert_nonadvancing_repeat(episode)
                    episode["loop_detected"] = False
                    metrics["overall"] = _overall(metrics["episodes"])
                elif tamper == "forged_advancement":
                    episode["trace"][0]["advanced"] = True
                else:
                    episode["trace"][0]["terminal_outcome"] = "resolved"
                    episode["trace"][-1]["terminal_outcome"] = None
                _rehash(artifact)

                result = _validate(
                    artifact,
                    role="expert-baseline",
                    policy=self.policy,
                    suite_path=self.suite_path,
                )
                self.assertFalse(result.evidence_passed)
                self.assertTrue(
                    any(
                        expected_failure in failure
                        for failure in result.evidence_failures
                    ),
                    result.evidence_failures,
                )

    def test_pre_policy_failure_prefix_rejects_action_and_outcome_tampering(
        self,
    ) -> None:
        for tamper, expected_failure in (
            ("action", "pre-policy failure trace action does not match"),
            ("status", "pre-policy failure trace outcome does not match"),
            ("error", "pre-policy failure trace outcome does not match"),
            ("advanced", "pre-policy failure trace outcome does not match"),
        ):
            with self.subTest(tamper=tamper):
                suite_path, _contract, policy, artifact = self._fixture_for_suite(
                    "forced_error_recovery"
                )
                row = artifact["evaluation"]["suite_metrics"]["episodes"][0][
                    "trace"
                ][0]
                if tamper == "action":
                    row["action"] = {
                        "tool": "finalize_diagnosis",
                        "arguments": {},
                    }
                elif tamper == "status":
                    row["execution_status"] = "success"
                elif tamper == "error":
                    row["error_code"] = "forged_failure"
                else:
                    row["advanced"] = True
                _rehash(artifact)

                result = _validate(
                    artifact,
                    role="expert-baseline",
                    policy=policy,
                    suite_path=suite_path,
                )
                self.assertFalse(result.evidence_passed)
                self.assertTrue(
                    any(
                        expected_failure in failure
                        for failure in result.evidence_failures
                    ),
                    result.evidence_failures,
                )

    def test_injected_failure_recovery_rejects_count_and_chronology_tampering(
        self,
    ) -> None:
        for tamper, expected_failure in (
            ("count", "recovered_failure_count does not match the trace"),
            ("chronology", "advanced does not match progress evidence"),
        ):
            with self.subTest(tamper=tamper):
                suite_path, _contract, policy, artifact = self._fixture_for_suite(
                    "forced_error_recovery"
                )
                episode = artifact["evaluation"]["suite_metrics"]["episodes"][0]
                if tamper == "count":
                    episode["evaluation_intervention"][
                        "recovered_failure_count"
                    ] = 0
                else:
                    for row in episode["trace"][1:]:
                        row["advanced"] = False
                _rehash(artifact)

                result = _validate(
                    artifact,
                    role="expert-baseline",
                    policy=policy,
                    suite_path=suite_path,
                )
                self.assertFalse(result.evidence_passed)
                self.assertTrue(
                    any(
                        expected_failure in failure
                        for failure in result.evidence_failures
                    ),
                    result.evidence_failures,
                )

    def test_policy_invalid_recovery_rejects_count_and_chronology_tampering(
        self,
    ) -> None:
        for tamper, expected_failure in (
            ("count", "invalid_action_count does not match the policy trace"),
            (
                "chronology",
                "recovered_invalid_action_count does not match the policy trace",
            ),
        ):
            with self.subTest(tamper=tamper):
                suite_path, _contract, policy, artifact = self._fixture_for_suite(
                    "invalid_action_recovery"
                )
                episode = artifact["evaluation"]["suite_metrics"]["episodes"][0]
                episode["invalid_action_count"] = 1
                if tamper == "chronology":
                    episode["recovered_invalid_action_count"] = 1
                    row = episode["trace"][-1]
                    row["action"] = {
                        "tool": "__invalid_action__",
                        "arguments": {"error_code": "forged_policy_invalid"},
                    }
                    row["execution_status"] = "failure"
                    row["advanced"] = False
                    row["error_code"] = "forged_policy_invalid"
                    row["terminal_outcome"] = None
                _rehash(artifact)

                result = _validate(
                    artifact,
                    role="expert-baseline",
                    policy=policy,
                    suite_path=suite_path,
                )
                self.assertFalse(result.evidence_passed)
                self.assertTrue(
                    any(
                        expected_failure in failure
                        for failure in result.evidence_failures
                    ),
                    result.evidence_failures,
                )

    def test_partial_prefix_rejects_alias_tool_status_and_disposition_tampering(
        self,
    ) -> None:
        for tamper, expected_failure in (
            ("alias", "has an unresolved state alias"),
            ("tool", "action does not match its contract"),
            ("status", "outcome is invalid"),
            ("disposition", "outcome is invalid"),
        ):
            with self.subTest(tamper=tamper):
                suite_path, _contract, policy, artifact = self._fixture_for_suite(
                    "partial_success_retention"
                )
                episode = artifact["evaluation"]["suite_metrics"]["episodes"][0]
                prefix = episode["trace"][:4]
                if tamper == "alias":
                    prefix[0]["action"]["arguments"]["state_id"] = "$active"
                elif tamper == "tool":
                    prefix[0]["action"]["tool"] = "run_wls"
                elif tamper == "status":
                    prefix[0]["execution_status"] = "failure"
                    prefix[0]["advanced"] = False
                    prefix[0]["error_code"] = "forged_setup_failure"
                else:
                    prefix[-1]["candidate_disposition_offline"] = None
                _rehash(artifact)

                result = _validate(
                    artifact,
                    role="expert-baseline",
                    policy=policy,
                    suite_path=suite_path,
                )
                self.assertFalse(result.evidence_passed)
                self.assertTrue(
                    any(
                        expected_failure in failure
                        for failure in result.evidence_failures
                    ),
                    result.evidence_failures,
                )

    def test_efficiency_intervention_enforces_its_per_episode_limits(self) -> None:
        artifact = _artifact(self.suite_path, self.contract)
        episode = next(
            row
            for row in artifact["evaluation"]["suite_metrics"]["episodes"]
            if row["suite"] == "efficiency"
        )
        episode["policy_steps"] = MAX_STEPS + 1
        episode["steps"] = MAX_STEPS + 1
        episode["wls_calls"] = 0
        episode["trace"] = [
            _canonical_policy_trace_row(
                index,
                tool="finalize_diagnosis",
                advanced=True,
                terminal_outcome=(
                    "resolved" if index == MAX_STEPS else None
                ),
            )
            for index in range(MAX_STEPS + 1)
        ]
        episode["audit"]["initial_active_state_id"] = episode["trace"][0][
            "state_before"
        ]["active_state_id"]
        episode["audit"]["final_active_state_id"] = episode["trace"][-1][
            "state_after"
        ]["active_state_id"]
        _rehash(artifact)

        result = _validate(
            artifact,
            role="expert-baseline",
            policy=self.policy,
            suite_path=self.suite_path,
        )
        self.assertTrue(result.evidence_passed, result.evidence_failures)
        self.assertFalse(result.performance_passed)
        self.assertTrue(
            any(
                "episode efficiency limit maximum_policy_steps failed" in failure
                for failure in result.performance_failures
            ),
            result.performance_failures,
        )

    def test_efficiency_call_limit_is_recomputed_from_policy_trace(self) -> None:
        artifact = _artifact(self.suite_path, self.contract)
        episode = next(
            row
            for row in artifact["evaluation"]["suite_metrics"]["episodes"]
            if row["suite"] == "efficiency"
        )
        episode["policy_steps"] = 5
        episode["steps"] = 5
        episode["wls_calls"] = 5
        episode["trace"] = [
            _canonical_policy_trace_row(
                index,
                tool="run_wls",
                advanced=True,
                terminal_outcome="resolved" if index == 4 else None,
            )
            for index in range(5)
        ]
        episode["audit"]["initial_active_state_id"] = episode["trace"][0][
            "state_before"
        ]["active_state_id"]
        episode["audit"]["final_active_state_id"] = episode["trace"][-1][
            "state_after"
        ]["active_state_id"]
        _rehash(artifact)

        result = _validate(
            artifact,
            role="expert-baseline",
            policy=self.policy,
            suite_path=self.suite_path,
        )
        self.assertTrue(result.evidence_passed, result.evidence_failures)
        self.assertFalse(result.performance_passed)
        self.assertTrue(
            any(
                "episode efficiency limit maximum_wls_calls failed" in failure
                for failure in result.performance_failures
            ),
            result.performance_failures,
        )

    def test_partial_suite_requires_a_retained_committed_opportunity(self) -> None:
        suite_path = self.root / "partial-suite.json"
        _write_json(
            suite_path,
            _suite_payload(names=("partial_success_retention",)),
        )
        contract = _fingerprint(
            suite_path, required_suites=("partial_success_retention",)
        )
        policy = _policy(suite_path, contract)
        artifact = _artifact(suite_path, contract)
        episode = artifact["evaluation"]["suite_metrics"]["episodes"][0]
        episode["retained_partial_fix_count"] = 0
        episode["evaluation_intervention"]["retained_opportunity_count"] = 0
        _rehash(artifact)

        result = _validate(
            artifact,
            role="expert-baseline",
            policy=policy,
            suite_path=suite_path,
            required_policy_id=TEST_POLICY_ID,
        )
        self.assertTrue(result.evidence_passed, result.evidence_failures)
        self.assertFalse(result.performance_passed)
        self.assertTrue(
            any(
                "committed partial correction was not retained" in failure
                for failure in result.performance_failures
            ),
            result.performance_failures,
        )

    def test_policy_pinned_suite_rejects_substitution(self) -> None:
        substitute_path = self.root / "easier-suite.json"
        _write_json(substitute_path, _suite_payload(suffix="-substitute"))
        substitute_contract = _fingerprint(substitute_path)
        substitute_artifact = _artifact(substitute_path, substitute_contract)

        result = _validate(
            substitute_artifact,
            role="expert-baseline",
            policy=self.policy,
            suite_path=substitute_path,
        )
        self.assertFalse(result.passed)
        self.assertTrue(
            any("policy-pinned SHA-256" in failure for failure in result.failures),
            result.failures,
        )
        self.assertTrue(
            any("packaged policy" in failure for failure in result.failures),
            result.failures,
        )

    def test_release_gate_rejects_legacy_flat_suite_schema(self) -> None:
        legacy_path = self.root / "legacy-flat-suite.json"
        _write_json(
            legacy_path,
            {
                suite: [
                    {
                        "scenario_id": f"legacy-{suite}",
                        "physical_root_fingerprint": f"legacy-root-{suite}",
                        "scenario_family": "no_error",
                        "error_cardinality": 0,
                        "network_case": "ieee14",
                        "split": "test",
                        "source_tier": "legacy",
                    }
                ]
                for suite in TEST_SUITES
            },
        )

        with self.assertRaisesRegex(ValueError, "release scenario schema validation"):
            _validate(
                _artifact(self.suite_path, self.contract),
                role="expert-baseline",
                policy=self.policy,
                suite_path=legacy_path,
            )
    def test_episode_without_suite_membership_fails_closed(self) -> None:
        artifact = _artifact(self.suite_path, self.contract)
        del artifact["evaluation"]["suite_metrics"]["episodes"][0]["suite"]
        _rehash(artifact)

        result = _validate(
            artifact,
            role="expert-baseline",
            policy=self.policy,
            suite_path=self.suite_path,
        )
        self.assertFalse(result.passed)
        self.assertTrue(
            any("missing required evidence" in failure for failure in result.failures),
            result.failures,
        )
        self.assertTrue(
            any("suite membership" in failure for failure in result.failures),
            result.failures,
        )

    def test_evaluator_seed_and_max_steps_are_policy_bound(self) -> None:
        for field, value in (("seed", SEED + 1), ("max_steps", MAX_STEPS - 1)):
            with self.subTest(field=field):
                artifact = _artifact(self.suite_path, self.contract)
                artifact["evaluation"]["suite_metrics"]["configuration"][field] = value
                _rehash(artifact)
                result = _validate(
                    artifact,
                    role="expert-baseline",
                    policy=self.policy,
                    suite_path=self.suite_path,
                )
                self.assertFalse(result.passed)
                self.assertTrue(
                    any(
                        f"configuration {field}" in failure
                        for failure in result.failures
                    ),
                    result.failures,
                )

    def test_custom_callback_isolation_evidence_is_mandatory(self) -> None:
        for callback_evidence in (
            None,
            {
                "passed": False,
                "physical_audit_callback": False,
                "tool_cost_callback": True,
            },
        ):
            with self.subTest(callback_evidence=callback_evidence):
                artifact = _artifact(self.suite_path, self.contract)
                configuration = artifact["evaluation"]["suite_metrics"][
                    "configuration"
                ]
                if callback_evidence is None:
                    configuration.pop("custom_callback_validation")
                else:
                    configuration["custom_callback_validation"] = callback_evidence
                _rehash(artifact)
                result = _validate(
                    artifact,
                    role="expert-baseline",
                    policy=self.policy,
                    suite_path=self.suite_path,
                )
                self.assertFalse(result.evidence_passed)
                self.assertTrue(
                    any("custom-callback" in failure for failure in result.failures),
                    result.failures,
                )

    def test_release_scenario_schema_evidence_is_mandatory(self) -> None:
        for schema_evidence in (
            None,
            {"passed": False, "scenario_schema_version": 1},
            {"passed": True, "scenario_schema_version": 1.0},
        ):
            with self.subTest(schema_evidence=schema_evidence):
                artifact = _artifact(self.suite_path, self.contract)
                configuration = artifact["evaluation"]["suite_metrics"][
                    "configuration"
                ]
                if schema_evidence is None:
                    configuration.pop("release_scenario_schema_validation")
                else:
                    configuration["release_scenario_schema_validation"] = (
                        schema_evidence
                    )
                _rehash(artifact)
                result = _validate(
                    artifact,
                    role="expert-baseline",
                    policy=self.policy,
                    suite_path=self.suite_path,
                )
                self.assertFalse(result.evidence_passed)
                self.assertTrue(
                    any("schema-v1" in failure for failure in result.failures),
                    result.failures,
                )

    def test_schema_versions_require_exact_integers(self) -> None:
        artifact = _artifact(self.suite_path, self.contract)
        artifact["artifact_schema_version"] = float(
            STUDY_EVALUATION_SCHEMA_VERSION
        )
        artifact["evaluation"]["suite_metrics"]["schema_version"] = float(
            STUDY_EVALUATION_SCHEMA_VERSION
        )
        artifact["provenance"]["provenance_schema_version"] = 1.0
        _rehash_provenance(artifact)
        _rehash(artifact)
        result = _validate(
            artifact,
            role="expert-baseline",
            policy=self.policy,
            suite_path=self.suite_path,
        )
        self.assertFalse(result.evidence_passed)
        self.assertTrue(
            any("artifact_schema_version" in failure for failure in result.failures),
            result.failures,
        )
        self.assertTrue(
            any("provenance_schema_version" in failure for failure in result.failures),
            result.failures,
        )
        self.assertTrue(
            any("suite report schema_version" in failure for failure in result.failures),
            result.failures,
        )

    def test_schema4_objective_episode_evidence_is_mandatory(self) -> None:
        artifact = _artifact(self.suite_path, self.contract)
        episode = artifact["evaluation"]["suite_metrics"]["episodes"][0]
        episode.pop("objective_evidence")
        _rehash(artifact)

        result = _validate(
            artifact,
            role="expert-baseline",
            policy=self.policy,
            suite_path=self.suite_path,
        )

        self.assertFalse(result.evidence_passed)
        self.assertTrue(
            any("objective evidence" in failure for failure in result.failures),
            result.failures,
        )

    def test_schema4_release_gate_recomputes_canonical_objective_evidence(
        self,
    ) -> None:
        mutations = ("leakage", "assessment", "tool_evidence")
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                artifact = _artifact(self.suite_path, self.contract)
                episode = artifact["evaluation"]["suite_metrics"]["episodes"][0]
                row = next(
                    item for item in episode["trace"] if not item["intervention"]
                )
                if mutation == "leakage":
                    row["policy_observation"]["HiddenTruth"] = ["private"]
                    row["observation_hash"] = stable_json_sha256(
                        row["policy_observation"]
                    )
                    row["objective_action_assessment"] = (
                        objective_recovery_action_assessment(
                            row["policy_observation"],
                            scenario_family=episode["family"],
                            error_cardinality=episode["cardinality"],
                        )
                    )
                elif mutation == "assessment":
                    row["objective_action_assessment"]["expected_action"] = {
                        "tool": "finalize_diagnosis",
                        "arguments": {},
                    }
                else:
                    row["objective_tool_evidence"][
                        "chi_square_statistic"
                    ] = 0.1
                _rehash(artifact)

                result = _validate(
                    artifact,
                    role="expert-baseline",
                    policy=self.policy,
                    suite_path=self.suite_path,
                )
                self.assertFalse(result.evidence_passed)
                expected = (
                    "privileged fields"
                    if mutation == "leakage"
                    else "not reproducible"
                )
                self.assertTrue(
                    any(expected in failure for failure in result.evidence_failures),
                    result.evidence_failures,
                )

    def test_schema4_release_gate_rejects_coordinated_observation_swaps(
        self,
    ) -> None:
        for mutation in ("state", "history"):
            with self.subTest(mutation=mutation):
                artifact = _artifact(self.suite_path, self.contract)
                episode = artifact["evaluation"]["suite_metrics"]["episodes"][0]
                row = next(
                    item for item in episode["trace"] if not item["intervention"]
                )
                if mutation == "state":
                    row["policy_observation"]["active_state_id"] = "forged-active"
                else:
                    forged_output = {
                        "execution_status": "failure",
                        "error_code": "fabricated_failure",
                        "state_mutated": False,
                    }
                    row["policy_observation"].update(
                        {
                            "history_window": [
                                {
                                    "state_id": row["state_before"][
                                        "active_state_id"
                                    ],
                                    "candidate_state_id": None,
                                    "action": {
                                        "tool": "run_wls",
                                        "arguments": {
                                            "state_id": row["state_before"][
                                                "active_state_id"
                                            ]
                                        },
                                    },
                                    "tool_output": forged_output,
                                }
                            ],
                            "last_tool": "run_wls",
                            "last_tool_status": "failure",
                            "last_tool_output": forged_output,
                        }
                    )
                row["observation_hash"] = stable_json_sha256(
                    row["policy_observation"]
                )
                row["objective_action_assessment"] = (
                    objective_recovery_action_assessment(
                        row["policy_observation"],
                        scenario_family=episode["family"],
                        error_cardinality=episode["cardinality"],
                    )
                )
                _rehash(artifact)
                result = _validate(
                    artifact,
                    role="expert-baseline",
                    policy=self.policy,
                    suite_path=self.suite_path,
                )
                self.assertFalse(result.evidence_passed)
                expected = (
                    "contradicts state_before"
                    if mutation == "state"
                    else "history is not derived from trace"
                )
                self.assertTrue(
                    any(expected in failure for failure in result.evidence_failures),
                    result.evidence_failures,
                )

    def test_schema4_release_gate_rejects_rehashed_trace_state_substitution(
        self,
    ) -> None:
        artifact = _artifact(self.suite_path, self.contract)
        episode = next(
            item
            for item in artifact["evaluation"]["suite_metrics"]["episodes"]
            if item["suite"] == "standard_success"
        )
        original, old_snapshot_hash = _coordinated_trace_state_substitution(
            episode,
            replacement="coordinated-forged-active",
        )
        self.assertEqual(episode["audit"]["initial_active_state_id"], original)
        self.assertNotEqual(
            episode["trace"][0]["state_before_sha256"], old_snapshot_hash
        )
        _rehash(artifact)

        result = _validate(
            artifact,
            role="expert-baseline",
            policy=self.policy,
            suite_path=self.suite_path,
        )

        self.assertFalse(result.evidence_passed)
        self.assertTrue(
            any(
                "evaluator-owned initial state identity" in failure
                for failure in result.evidence_failures
            ),
            result.evidence_failures,
        )

    def test_artifact_provenance_and_factory_integrity_remain_fail_closed(
        self,
    ) -> None:
        artifact = _artifact(self.suite_path, self.contract)
        artifact["evaluation"]["score"] = 1.0e100
        result = _validate(
            artifact,
            role="expert-baseline",
            policy=self.policy,
            suite_path=self.suite_path,
        )
        self.assertFalse(result.passed)
        self.assertTrue(
            any("content_sha256" in failure for failure in result.failures),
            result.failures,
        )

        artifact = _artifact(self.suite_path, self.contract)
        artifact["provenance"]["source_state"]["source_commit"] = "e" * 40
        _rehash(artifact)
        result = _validate(
            artifact,
            role="expert-baseline",
            policy=self.policy,
            suite_path=self.suite_path,
        )
        self.assertFalse(result.passed)
        self.assertTrue(
            any("identity_sha256" in failure for failure in result.failures),
            result.failures,
        )

        unapproved = copy.deepcopy(self.policy)
        unapproved["approved_factories"]["environment"] = []
        result = _validate(
            _artifact(self.suite_path, self.contract),
            role="expert-baseline",
            policy=unapproved,
            suite_path=self.suite_path,
        )
        self.assertFalse(result.passed)
        self.assertTrue(
            any("environment factory" in failure for failure in result.failures),
            result.failures,
        )

    def test_episode_root_and_strict_audit_cannot_be_relabelled(self) -> None:
        artifact = _artifact(self.suite_path, self.contract)
        episodes = artifact["evaluation"]["suite_metrics"]["episodes"]
        episodes[1]["physical_root"] = episodes[0]["physical_root"]
        episodes[1]["audit"]["strict_release_audit"][
            "physical_root_fingerprint"
        ] = episodes[0]["physical_root"]
        _rehash(artifact)
        result = _validate(
            artifact,
            role="expert-baseline",
            policy=self.policy,
            suite_path=self.suite_path,
        )
        self.assertFalse(result.passed)
        self.assertTrue(
            any("globally duplicate" in failure for failure in result.failures),
            result.failures,
        )

    def test_frozen_manifest_binds_family_and_all_grouping_dimensions(self) -> None:
        artifact = _artifact(self.suite_path, self.contract)
        episode = artifact["evaluation"]["suite_metrics"]["episodes"][0]
        episode["family"] = "hif"
        episode["audit"]["strict_release_audit"]["scenario_family"] = "hif"
        _rehash(artifact)

        result = _validate(
            artifact,
            role="expert-baseline",
            policy=self.policy,
            suite_path=self.suite_path,
        )
        self.assertFalse(result.passed)
        self.assertTrue(
            any("frozen suite manifest" in failure for failure in result.failures),
            result.failures,
        )

        for field in ("cardinality", "case", "split", "source_tier"):
            with self.subTest(field=field):
                artifact = _artifact(self.suite_path, self.contract)
                artifact["evaluation"]["suite_metrics"]["episodes"][0][field] = (
                    "altered"
                )
                _rehash(artifact)
                result = _validate(
                    artifact,
                    role="expert-baseline",
                    policy=self.policy,
                    suite_path=self.suite_path,
                )
                self.assertFalse(result.passed)
                self.assertTrue(
                    any(
                        "frozen suite manifest" in failure
                        for failure in result.failures
                    ),
                    result.failures,
                )

    def test_episode_counters_require_exact_json_integers(self) -> None:
        artifact = _artifact(self.suite_path, self.contract)
        artifact["evaluation"]["suite_metrics"]["episodes"][0][
            "invalid_action_count"
        ] = "1"
        _rehash(artifact)
        result = _validate(
            artifact,
            role="expert-baseline",
            policy=self.policy,
            suite_path=self.suite_path,
        )
        self.assertFalse(result.passed)
        self.assertFalse(result.evidence_passed)
        self.assertTrue(
            any("must be a non-negative integer" in failure for failure in result.failures),
            result.failures,
        )

    def test_evaluator_source_is_anchored_to_the_real_evaluator(self) -> None:
        artifact = _artifact(self.suite_path, self.contract)
        artifact["provenance"]["evaluator_source"] = _source_descriptor(Path(__file__))
        _rehash_provenance(artifact)
        result = _validate(
            artifact,
            role="expert-baseline",
            policy=self.policy,
            suite_path=self.suite_path,
        )
        self.assertFalse(result.passed)
        self.assertTrue(
            any("evaluator source identity" in failure for failure in result.failures),
            result.failures,
        )

        artifact = _artifact(self.suite_path, self.contract)
        artifact["evaluation"]["suite_metrics"]["episodes"][0]["audit"][
            "strict_release_audit"
        ]["terminal_outcome"] = "operator_escalation"
        _rehash(artifact)
        result = _validate(
            artifact,
            role="expert-baseline",
            policy=self.policy,
            suite_path=self.suite_path,
        )
        self.assertFalse(result.passed)
        self.assertTrue(
            any("audit evidence" in failure for failure in result.failures),
            result.failures,
        )

    def test_provenance_for_suite_a_cannot_cover_results_from_suite_b(self) -> None:
        substitute_path = self.root / "suite-b.json"
        _write_json(substitute_path, _suite_payload(suffix="-suite-b"))
        substitute_contract = _fingerprint(substitute_path)
        artifact = _artifact(substitute_path, substitute_contract)
        input_suite = artifact["provenance"]["input_suite"]
        input_suite.update(
            {
                "provided_path": str(self.suite_path),
                "resolved_path": str(self.suite_path.resolve()),
                "sha256": file_sha256(self.suite_path),
                "size_bytes": self.suite_path.stat().st_size,
            }
        )
        identity_core = copy.deepcopy(artifact["provenance"])
        for field in ("identity_sha256", "release_eligible", "release_failures"):
            identity_core.pop(field, None)
        artifact["provenance"]["identity_sha256"] = stable_json_sha256(
            identity_core
        )
        _rehash(artifact)

        result = _validate(
            artifact,
            role="expert-baseline",
            policy=self.policy,
            suite_path=self.suite_path,
        )
        self.assertFalse(result.passed)
        self.assertTrue(
            any(
                "recomputed expected suite" in failure
                or "frozen suite manifest" in failure
                for failure in result.failures
            ),
            result.failures,
        )

    def test_base_records_performance_failure_but_does_not_block_measurement(self) -> None:
        base_artifact = _artifact(
            self.suite_path,
            self.contract,
            explicit_identity=None,
            model_id="base/gemma",
            model_revision=MODEL_REVISION,
            performance_ok=False,
        )
        result = _validate(
            base_artifact,
            role="base-baseline",
            policy=self.policy,
            suite_path=self.suite_path,
            explicit_identity=None,
            model_id="base/gemma",
            model_revision=MODEL_REVISION,
        )
        self.assertTrue(result.passed, result.failures)
        self.assertTrue(result.evidence_passed)
        self.assertFalse(result.performance_passed)
        self.assertFalse(result.performance_enforced)
        self.assertTrue(result.performance_failures)

        base_artifact["evaluation"]["suite_metrics"]["episodes"][0]["audit"][
            "evidence_complete"
        ] = False
        _rehash(base_artifact)
        incomplete = _validate(
            base_artifact,
            role="base-baseline",
            policy=self.policy,
            suite_path=self.suite_path,
            explicit_identity=None,
            model_id="base/gemma",
            model_revision=MODEL_REVISION,
        )
        self.assertFalse(incomplete.passed)
        self.assertFalse(incomplete.evidence_passed)
        self.assertTrue(
            any("evidence is incomplete" in failure for failure in incomplete.failures),
            incomplete.failures,
        )

    def test_truth_audited_false_finalization_is_performance_not_evidence(
        self,
    ) -> None:
        base_artifact = _artifact(
            self.suite_path,
            self.contract,
            explicit_identity=None,
            model_id="base/gemma",
            model_revision=MODEL_REVISION,
        )
        metrics = base_artifact["evaluation"]["suite_metrics"]
        for episode in metrics["episodes"]:
            episode["final_physical_success"] = False
            episode["final_physical_correct"] = False
            episode["false_finalization_count"] = 1
            strict = episode["audit"]["strict_release_audit"]
            strict["problems"] = ["remaining_true_faults"]
            strict["quarantined"] = True
            episode["audit"]["quarantined"] = True
            for name in (
                "remaining_true_faults",
                "final_measurements_match_clean",
                "final_case_matches_clean",
            ):
                strict["checks"][name] = {"status": "failed"}
        metrics["overall"] = _overall(metrics["episodes"])
        _rehash(base_artifact)

        result = _validate(
            base_artifact,
            role="base-baseline",
            policy=self.policy,
            suite_path=self.suite_path,
            explicit_identity=None,
            model_id="base/gemma",
            model_revision=MODEL_REVISION,
        )
        self.assertTrue(result.passed, result.evidence_failures)
        self.assertTrue(result.evidence_passed)
        self.assertFalse(result.performance_passed)
        self.assertTrue(
            any(
                "did not achieve complete physical safety" in failure
                for failure in result.performance_failures
            ),
            result.performance_failures,
        )

    def test_same_performance_failure_blocks_expert_and_checkpoint(self) -> None:
        expert = _validate(
            _artifact(
                self.suite_path,
                self.contract,
                performance_ok=False,
            ),
            role="expert-baseline",
            policy=self.policy,
            suite_path=self.suite_path,
        )
        self.assertFalse(expert.passed)
        self.assertTrue(expert.evidence_passed)
        self.assertFalse(expert.performance_passed)
        self.assertTrue(expert.performance_enforced)

        checkpoint_artifact = _artifact(
            self.suite_path,
            self.contract,
            explicit_identity=None,
            model_id="checkpoint/bc0",
            model_revision=MODEL_REVISION,
            performance_ok=False,
        )
        reference_artifact = _artifact(
            self.suite_path,
            self.contract,
            explicit_identity=None,
            model_id="base/gemma",
            model_revision=MODEL_REVISION,
            performance_ok=True,
        )
        checkpoint = _validate(
            checkpoint_artifact,
            role="checkpoint-promotion",
            policy=self.policy,
            suite_path=self.suite_path,
            explicit_identity=None,
            model_id="checkpoint/bc0",
            model_revision=MODEL_REVISION,
            reference_artifact=reference_artifact,
            reference_model_id="base/gemma",
            reference_model_revision=MODEL_REVISION,
        )
        self.assertFalse(checkpoint.passed)
        self.assertTrue(checkpoint.evidence_passed)
        self.assertFalse(checkpoint.performance_passed)
        self.assertTrue(checkpoint.performance_enforced)

    def test_checkpoint_requires_an_evidence_valid_reference(self) -> None:
        checkpoint_artifact = _artifact(
            self.suite_path,
            self.contract,
            explicit_identity=None,
            model_id="checkpoint/bc0",
            model_revision=MODEL_REVISION,
        )
        with self.assertRaisesRegex(ValueError, "requires reference_artifact"):
            _validate(
                checkpoint_artifact,
                role="checkpoint-promotion",
                policy=self.policy,
                suite_path=self.suite_path,
                explicit_identity=None,
                model_id="checkpoint/bc0",
                model_revision=MODEL_REVISION,
            )

        invalid_reference = _artifact(
            self.suite_path,
            self.contract,
            explicit_identity=None,
            model_id="base/gemma",
            model_revision=MODEL_REVISION,
        )
        invalid_reference["evaluation"]["suite_metrics"]["episodes"][0][
            "release_environment_attestation"
        ]["passed"] = False
        _rehash(invalid_reference)
        result = _validate(
            checkpoint_artifact,
            role="checkpoint-promotion",
            policy=self.policy,
            suite_path=self.suite_path,
            explicit_identity=None,
            model_id="checkpoint/bc0",
            model_revision=MODEL_REVISION,
            reference_artifact=invalid_reference,
            reference_model_id="base/gemma",
            reference_model_revision=MODEL_REVISION,
        )
        self.assertFalse(result.passed)
        self.assertTrue(
            any("reference" in failure.lower() for failure in result.failures),
            result.failures,
        )

    def test_checkpoint_cannot_reference_itself_or_change_policy_factory(self) -> None:
        checkpoint_artifact = _artifact(
            self.suite_path,
            self.contract,
            explicit_identity=None,
            model_id="checkpoint/bc0",
            model_revision=MODEL_REVISION,
        )
        with self.assertRaisesRegex(ValueError, "identities must differ"):
            _validate(
                checkpoint_artifact,
                role="checkpoint-promotion",
                policy=self.policy,
                suite_path=self.suite_path,
                explicit_identity=None,
                model_id="checkpoint/bc0",
                model_revision=MODEL_REVISION,
                reference_artifact=checkpoint_artifact,
                reference_model_id="checkpoint/bc0",
                reference_model_revision=MODEL_REVISION,
            )

        reference_artifact = _artifact(
            self.suite_path,
            self.contract,
            explicit_identity=None,
            model_id="base/gemma",
            model_revision=MODEL_REVISION,
        )
        reference_artifact["provenance"]["factories"]["policy"][
            "import_spec"
        ] = "tests:alternate_policy"
        _rehash_provenance(reference_artifact)
        policy = copy.deepcopy(self.policy)
        policy["approved_factories"]["model_policy"].append(
            {
                "import_spec": "tests:alternate_policy",
                "source_sha256": file_sha256(Path(__file__)),
            }
        )
        result = _validate(
            checkpoint_artifact,
            role="checkpoint-promotion",
            policy=policy,
            suite_path=self.suite_path,
            explicit_identity=None,
            model_id="checkpoint/bc0",
            model_revision=MODEL_REVISION,
            reference_artifact=reference_artifact,
            reference_model_id="base/gemma",
            reference_model_revision=MODEL_REVISION,
        )
        self.assertFalse(result.passed)
        self.assertFalse(result.evidence_passed)
        self.assertTrue(
            any("policy_factory" in failure for failure in result.failures),
            result.failures,
        )

    def test_exact_paired_checkpoint_and_reference_pass(self) -> None:
        checkpoint_artifact = _artifact(
            self.suite_path,
            self.contract,
            explicit_identity=None,
            model_id="checkpoint/bc0",
            model_revision=MODEL_REVISION,
        )
        reference_artifact = _artifact(
            self.suite_path,
            self.contract,
            explicit_identity=None,
            model_id="base/gemma",
            model_revision=MODEL_REVISION,
        )
        result = _validate(
            checkpoint_artifact,
            role="checkpoint-promotion",
            policy=self.policy,
            suite_path=self.suite_path,
            explicit_identity=None,
            model_id="checkpoint/bc0",
            model_revision=MODEL_REVISION,
            reference_artifact=reference_artifact,
            reference_model_id="base/gemma",
            reference_model_revision=MODEL_REVISION,
        )
        self.assertTrue(result.passed, result.failures)
        self.assertTrue(result.evidence_passed)
        self.assertTrue(result.performance_passed)
        self.assertTrue(result.comparison_passed)
        self.assertEqual(
            result.observed["paired_nonregression"]["paired_episodes"],
            len(self.contract["episode_manifest"]),
        )

    def test_model_role_requires_complete_accelerator_attestation(self) -> None:
        base_artifact = _artifact(
            self.suite_path,
            self.contract,
            explicit_identity=None,
            model_id="base/gemma",
            model_revision=MODEL_REVISION,
        )
        del base_artifact["provenance"]["runtime_environment"]["accelerator"]
        _rehash_provenance(base_artifact)

        result = _validate(
            base_artifact,
            role="base-baseline",
            policy=self.policy,
            suite_path=self.suite_path,
            explicit_identity=None,
            model_id="base/gemma",
            model_revision=MODEL_REVISION,
        )

        self.assertFalse(result.evidence_passed)
        self.assertTrue(
            any(
                "accelerator attestation is missing" in failure
                for failure in result.evidence_failures
            ),
            result.evidence_failures,
        )

    def test_model_role_rejects_unapproved_accelerator_class(self) -> None:
        base_artifact = _artifact(
            self.suite_path,
            self.contract,
            explicit_identity=None,
            model_id="base/gemma",
            model_revision=MODEL_REVISION,
            accelerator_name="NVIDIA A100-SXM4-80GB",
        )

        result = _validate(
            base_artifact,
            role="base-baseline",
            policy=self.policy,
            suite_path=self.suite_path,
            explicit_identity=None,
            model_id="base/gemma",
            model_revision=MODEL_REVISION,
        )

        self.assertFalse(result.evidence_passed)
        self.assertTrue(
            any(
                "not an approved release accelerator" in failure
                for failure in result.evidence_failures
            ),
            result.evidence_failures,
        )
        self.assertEqual(result.observed["accelerator_classes"], [])

    def test_model_role_requires_bf16_and_exactly_one_device(self) -> None:
        def disable_bf16(artifact: dict) -> None:
            artifact["provenance"]["runtime_environment"]["accelerator"][
                "bf16_supported"
            ] = False

        def expose_second_device(artifact: dict) -> None:
            accelerator = artifact["provenance"]["runtime_environment"][
                "accelerator"
            ]
            second = copy.deepcopy(accelerator["devices"][0])
            second["index"] = 1
            accelerator["devices"].append(second)
            accelerator["device_count"] = 2

        def remove_driver_version(artifact: dict) -> None:
            artifact["provenance"]["runtime_environment"]["accelerator"][
                "driver_version"
            ] = None

        cases = (
            ("bf16", disable_bf16, "BF16 support"),
            ("two_devices", expose_second_device, "exactly one"),
            ("driver", remove_driver_version, "driver version"),
        )
        for label, mutate, expected_failure in cases:
            with self.subTest(label=label):
                artifact = _artifact(
                    self.suite_path,
                    self.contract,
                    explicit_identity=None,
                    model_id="base/gemma",
                    model_revision=MODEL_REVISION,
                )
                mutate(artifact)
                _rehash_provenance(artifact)

                result = _validate(
                    artifact,
                    role="base-baseline",
                    policy=self.policy,
                    suite_path=self.suite_path,
                    explicit_identity=None,
                    model_id="base/gemma",
                    model_revision=MODEL_REVISION,
                )

                self.assertFalse(result.evidence_passed)
                self.assertTrue(
                    any(
                        expected_failure in failure
                        for failure in result.evidence_failures
                    ),
                    result.evidence_failures,
                )

    def test_checkpoint_and_base_allow_different_approved_accelerator_classes(
        self,
    ) -> None:
        cases = (
            ("NVIDIA H200", "NVIDIA H100 80GB HBM3", "h200", "h100"),
            (
                "NVIDIA H100 80GB HBM3",
                "NVIDIA RTX PRO 6000 Blackwell Server Edition",
                "h100",
                "rtx6000",
            ),
            (
                "NVIDIA RTX PRO 6000 Blackwell Server Edition",
                "NVIDIA H200",
                "rtx6000",
                "h200",
            ),
        )
        for candidate_name, reference_name, candidate_class, reference_class in cases:
            with self.subTest(
                candidate_class=candidate_class,
                reference_class=reference_class,
            ):
                checkpoint_artifact = _artifact(
                    self.suite_path,
                    self.contract,
                    explicit_identity=None,
                    model_id="checkpoint/bc0",
                    model_revision=MODEL_REVISION,
                    accelerator_name=candidate_name,
                )
                reference_artifact = _artifact(
                    self.suite_path,
                    self.contract,
                    explicit_identity=None,
                    model_id="base/gemma",
                    model_revision=MODEL_REVISION,
                    accelerator_name=reference_name,
                )

                result = _validate(
                    checkpoint_artifact,
                    role="checkpoint-promotion",
                    policy=self.policy,
                    suite_path=self.suite_path,
                    explicit_identity=None,
                    model_id="checkpoint/bc0",
                    model_revision=MODEL_REVISION,
                    reference_artifact=reference_artifact,
                    reference_model_id="base/gemma",
                    reference_model_revision=MODEL_REVISION,
                )

                self.assertTrue(result.passed, result.failures)
                self.assertTrue(result.evidence_passed)
                self.assertTrue(result.performance_passed)
                self.assertTrue(result.comparison_passed)
                paired = result.observed["paired_nonregression"]
                self.assertEqual(
                    paired["candidate_accelerator_classes"],
                    [candidate_class],
                )
                self.assertEqual(
                    paired["reference_accelerator_classes"],
                    [reference_class],
                )
                self.assertEqual(
                    paired["accelerator_compatibility"],
                    {
                        "policy": "individually_approved_release_classes",
                        "compatible": True,
                        "same_class": False,
                    },
                )

    def test_expert_cpu_artifact_does_not_require_cuda_attestation(self) -> None:
        expert_artifact = _artifact(self.suite_path, self.contract)
        del expert_artifact["provenance"]["runtime_environment"]["accelerator"]
        _rehash_provenance(expert_artifact)

        result = _validate(
            expert_artifact,
            role="expert-baseline",
            policy=self.policy,
            suite_path=self.suite_path,
        )

        self.assertTrue(result.evidence_passed, result.evidence_failures)
        self.assertEqual(result.observed["accelerator_classes"], [])

    def test_per_root_regression_fails_with_equal_aggregate_outcomes(self) -> None:
        policy = copy.deepcopy(self.policy)
        policy["family_policy"]["no_error"].update(
            {
                "minimum_audited_completion_rate": 0.5,
                "maximum_unqualified_operator_escalation_rate": 0.5,
            }
        )
        checkpoint_artifact = _artifact(
            self.suite_path,
            self.contract,
            explicit_identity=None,
            model_id="checkpoint/bc0",
            model_revision=MODEL_REVISION,
        )
        reference_artifact = _artifact(
            self.suite_path,
            self.contract,
            explicit_identity=None,
            model_id="base/gemma",
            model_revision=MODEL_REVISION,
        )
        _set_operator_escalation(reference_artifact, 0)
        _set_operator_escalation(checkpoint_artifact, 1)

        self.assertEqual(
            checkpoint_artifact["evaluation"]["suite_metrics"]["overall"],
            reference_artifact["evaluation"]["suite_metrics"]["overall"],
        )
        result = _validate(
            checkpoint_artifact,
            role="checkpoint-promotion",
            policy=policy,
            suite_path=self.suite_path,
            explicit_identity=None,
            model_id="checkpoint/bc0",
            model_revision=MODEL_REVISION,
            reference_artifact=reference_artifact,
            reference_model_id="base/gemma",
            reference_model_revision=MODEL_REVISION,
        )
        self.assertFalse(result.passed)
        self.assertTrue(result.evidence_passed, result.evidence_failures)
        self.assertFalse(result.performance_passed)
        self.assertFalse(result.comparison_passed)
        paired = result.observed["paired_nonregression"]
        self.assertEqual(len(paired["regressions"]), 1)
        self.assertEqual(
            paired["regressions"][0]["reason"],
            "safety_ordinal",
        )
        self.assertEqual(
            paired["regressions"][0]["episode_key"],
            checkpoint_artifact["evaluation"]["suite_metrics"]["episodes"][1][
                "episode_key"
            ],
        )

    def test_loop_regression_on_safe_escalation_cannot_hide_in_aggregate(self) -> None:
        policy = copy.deepcopy(self.policy)
        policy["hard_constraints"]["maximum_loop_episode_rate"] = 0.5
        policy["family_policy"]["no_error"].update(
            {
                "minimum_audited_completion_rate": 0.5,
                "maximum_unqualified_operator_escalation_rate": 0.5,
            }
        )
        checkpoint_artifact = _artifact(
            self.suite_path,
            self.contract,
            explicit_identity=None,
            model_id="checkpoint/bc0",
            model_revision=MODEL_REVISION,
        )
        reference_artifact = _artifact(
            self.suite_path,
            self.contract,
            explicit_identity=None,
            model_id="base/gemma",
            model_revision=MODEL_REVISION,
        )
        # Reproduce the evaluator's former unresolved-episode summary while
        # retaining a fully passing strict healthy-preservation audit.
        _set_operator_escalation(
            reference_artifact, 0, healthy_summary_known=False
        )
        _set_operator_escalation(
            checkpoint_artifact, 0, healthy_summary_known=False
        )
        candidate_episode = checkpoint_artifact["evaluation"]["suite_metrics"][
            "episodes"
        ][0]
        _insert_nonadvancing_repeat(candidate_episode)
        checkpoint_artifact["evaluation"]["suite_metrics"]["overall"] = _overall(
            checkpoint_artifact["evaluation"]["suite_metrics"]["episodes"]
        )
        _rehash(checkpoint_artifact)

        result = _validate(
            checkpoint_artifact,
            role="checkpoint-promotion",
            policy=policy,
            suite_path=self.suite_path,
            explicit_identity=None,
            model_id="checkpoint/bc0",
            model_revision=MODEL_REVISION,
            reference_artifact=reference_artifact,
            reference_model_id="base/gemma",
            reference_model_revision=MODEL_REVISION,
        )

        self.assertTrue(result.evidence_passed, result.evidence_failures)
        self.assertFalse(result.comparison_passed)
        self.assertFalse(result.passed)
        regressions = result.observed["paired_nonregression"]["regressions"]
        self.assertEqual(len(regressions), 1)
        self.assertEqual(regressions[0]["reason"], "safety_ordinal")
        self.assertEqual(regressions[0]["reference_safety_ordinal"], 1)
        self.assertEqual(regressions[0]["candidate_safety_ordinal"], 0)

    def test_unconfigured_policy_fails_closed(self) -> None:
        default_suite_path = self.root / "default-unconfigured-suite.json"
        policy_path = self.root / "unconfigured-policy.json"
        _write_json(
            default_suite_path,
            _suite_payload(
                names=tuple(EVALUATION_SUITES),
                roots_per_suite=20,
                suffix="-default",
            ),
        )
        unconfigured = copy.deepcopy(self.policy)
        unconfigured["approved_factories"] = {
            role: [] for role in unconfigured["approved_factories"]
        }
        unconfigured["suite_policy"]["status"] = "unconfigured"
        unconfigured["suite_policy"]["approved_suite_sha256"] = None
        unconfigured["suite_policy"]["approved_suite_manifest"] = None
        _write_json(policy_path, unconfigured)
        result = validate_evaluation_artifact(
            _artifact(self.suite_path, self.contract),
            role="expert-baseline",
            policy=policy_path,
            expected_source_commit=COMMIT,
            expected_suite_path=default_suite_path,
            expected_protocol="canonical",
            expected_registry_sha256=current_registry_sha256("canonical"),
            expected_policy_identity="expert-v1",
            required_gate_policy_id=TEST_POLICY_ID,
            repo_root=REPO_ROOT,
            require_current_clean_source=False,
        )
        self.assertFalse(result.passed)
        self.assertTrue(
            any("suite policy is not pinned" in failure for failure in result.failures),
            result.failures,
        )

    def test_cli_accepts_suite_path_and_rejects_hash_only(self) -> None:
        artifact_path = self.root / "expert-evaluation.json"
        policy_path = self.root / "policy.json"
        report_path = self.root / "gate-report.json"
        _write_json(artifact_path, _artifact(self.suite_path, self.contract))
        _write_json(policy_path, self.policy)
        clean_source = {
            "source_commit": COMMIT,
            "release_eligible_source": True,
            "source_worktree_dirty": False,
        }
        path_arguments = [
            "--artifact",
            str(artifact_path),
            "--role",
            "expert-baseline",
            "--policy",
            str(policy_path),
            "--required-gate-policy-id",
            TEST_POLICY_ID,
            "--expected-source-commit",
            COMMIT,
            "--expected-suite",
            str(self.suite_path),
            "--expected-policy-identity",
            "expert-v1",
            "--report-output",
            str(report_path),
        ]
        with mock.patch(
            "psse_env.dagger.evaluation_gate.git_source_state",
            return_value=clean_source,
        ), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(gate_main(path_arguments), 0)
        self.assertTrue(json.loads(report_path.read_text(encoding="utf-8"))["passed"])

        hash_only = [
            "--artifact",
            str(artifact_path),
            "--role",
            "expert-baseline",
            "--policy",
            str(policy_path),
            "--required-gate-policy-id",
            TEST_POLICY_ID,
            "--expected-source-commit",
            COMMIT,
            "--expected-suite-sha256",
            file_sha256(self.suite_path),
            "--expected-policy-identity",
            "expert-v1",
        ]
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            gate_main(hash_only)

    def test_checkpoint_cli_requires_and_accepts_reference_identity(self) -> None:
        checkpoint_path = self.root / "checkpoint-evaluation.json"
        reference_path = self.root / "reference-evaluation.json"
        policy_path = self.root / "policy.json"
        _write_json(
            checkpoint_path,
            _artifact(
                self.suite_path,
                self.contract,
                explicit_identity=None,
                model_id="checkpoint/bc0",
                model_revision=MODEL_REVISION,
            ),
        )
        _write_json(
            reference_path,
            _artifact(
                self.suite_path,
                self.contract,
                explicit_identity=None,
                model_id="base/gemma",
                model_revision=MODEL_REVISION,
            ),
        )
        _write_json(policy_path, self.policy)
        arguments = [
            "--artifact",
            str(checkpoint_path),
            "--role",
            "checkpoint-promotion",
            "--policy",
            str(policy_path),
            "--required-gate-policy-id",
            TEST_POLICY_ID,
            "--expected-source-commit",
            COMMIT,
            "--expected-suite",
            str(self.suite_path),
            "--expected-model-id",
            "checkpoint/bc0",
            "--expected-model-revision",
            MODEL_REVISION,
        ]
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            gate_main(arguments)

        clean_source = {
            "source_commit": COMMIT,
            "release_eligible_source": True,
            "source_worktree_dirty": False,
        }
        with mock.patch(
            "psse_env.dagger.evaluation_gate.git_source_state",
            return_value=clean_source,
        ), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(
                gate_main(
                    [
                        *arguments,
                        "--reference-artifact",
                        str(reference_path),
                        "--reference-model-id",
                        "base/gemma",
                        "--reference-model-revision",
                        MODEL_REVISION,
                    ]
                ),
                0,
            )


if __name__ == "__main__":
    unittest.main()
