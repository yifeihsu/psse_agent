"""Focused tests for fail-closed DAgger study metrics."""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from psse_env.actions import (
    ASK_FOR_MORE_EVIDENCE,
    COMMIT_STATE,
    CORRECT_MEASUREMENTS,
    FINALIZE_DIAGNOSIS,
    RUN_WLS,
    VERIFY_CANDIDATE,
)
from psse_env.dagger.evaluator import (
    STUDY_EVALUATION_SCHEMA_VERSION,
    STUDY_OBJECTIVE_TOOL_EVIDENCE_CONTRACT,
    objective_recovery_action_assessment,
    objective_tool_evidence,
    study_objective_episode_evidence_marker,
)
from psse_env.dagger.study_metrics import (
    StudyEvidenceError,
    build_study_report,
    compare_paired_runs,
    extract_artifact_metrics,
)
from psse_env.dagger.study_manifest import (
    EXPECTED_DEVELOPMENT_EVALUATION_CONTRACT_SHA256,
    EXPECTED_STUDY_MANIFEST_SHA256,
    build_production_d1_quarantine_binding,
    build_training_protocol_binding,
    canonical_production_d1_quarantine_binding,
    canonical_training_rng_attestation,
    load_study_manifest,
)
from psse_env.sft.provenance import stable_json_sha256
from psse_env.state_store import PolicyObservation


HASH = "a" * 64
SUITE_HASH = "b" * 64
SOURCE_COMMIT = "b" * 40
DEVELOPMENT_SUITE_HASH = "9" * 64
QUARANTINE_SUMMARY = {
    "contract": "dagger1_offline_teacher_target_quarantine_summary_v1",
    "candidate_definition": {},
    "total_rows": 25,
    "candidate_rows": 25,
    "non_candidate_rows": 0,
    "passed_rows": 25,
    "quarantined_rows": 0,
    "invalid_or_missing_audit_rows": 0,
    "quarantined_by_action_class": {},
    "quarantined_by_reason_code": {},
    "quarantined_example_ids": [],
    "zero_truth_audit_quarantine": True,
    "passed": True,
}
ROUND1_DESCRIPTOR = {
    "builder_contract": "deterministic_d0_d1_probe_balanced_union_v2",
    "audit_report_sha256": {
        "d1_offline_teacher_target_quarantine_summary": stable_json_sha256(
            QUARANTINE_SUMMARY
        )
    },
}
ROUND1_PROVENANCE = stable_json_sha256(ROUND1_DESCRIPTOR)
ACCELERATOR = {
    "device_count": 1,
    "bf16_supported": True,
    "torch_cuda_version": "12.8",
    "required_accelerator_class": None,
    "required_accelerator_class_matched": True,
    "devices": [
        {
            "index": 0,
            "name": "NVIDIA H200",
            "total_memory_bytes": 141 * 1024**3,
            "compute_capability": [9, 0],
            "accelerator_class": "h200",
        }
    ],
}


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
    episode: dict[str, Any], *, replacement: str
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
        row["state_before_sha256"] = _artifact_hash(row["state_before"])
        row["state_after_sha256"] = _artifact_hash(row["state_after"])
        row["objective_tool_evidence"] = objective_tool_evidence(
            row["action"], row["policy_tool_output"]
        )
        if row["intervention"] is not True:
            row["observation_hash"] = _artifact_hash(row["policy_observation"])
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
    return original, first_before_hash


STUDY_MANIFEST_PAYLOAD = json.loads(
    (
        Path(__file__).resolve().parent / "studies" / "dagger_multiseed_study_v1.json"
    ).read_text(encoding="utf-8")
)
OBJECTIVE_THRESHOLDS = STUDY_MANIFEST_PAYLOAD["objective_thresholds"]
COMPARISON_POLICY = STUDY_MANIFEST_PAYLOAD["comparison_policy"]


def _artifact_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _state(
    *,
    active_state_id: str = "active",
    candidate_state_id: str | None = None,
    accepted_correction_count: int = 0,
) -> dict[str, Any]:
    return {
        "active_state_id": active_state_id,
        "candidate_state_id": candidate_state_id,
        "phase": "investigating",
        "accepted_correction_count": accepted_correction_count,
        "explained_anomaly_count": 0,
    }


def _trace_row(
    *,
    step: int,
    action: dict[str, Any],
    status: str = "success",
    terminal_outcome: str | None = None,
    error_code: str | None = None,
) -> dict[str, Any]:
    terminal = terminal_outcome is not None
    state = _state()
    return {
        "step": step,
        "intervention": False,
        "observation_hash": HASH,
        "action": action,
        "execution_status": status,
        "advanced": terminal,
        "error_code": error_code,
        "candidate_disposition_offline": None,
        "tool_regret": None,
        "runtime_state_hash": None,
        "terminal_outcome": terminal_outcome,
        "state_before": state,
        "state_after": copy.deepcopy(state),
        "state_before_sha256": _artifact_hash(state),
        "state_after_sha256": _artifact_hash(state),
        "state_mutated": False,
        "terminal_after": terminal,
    }


def _terminal_trace(outcome: str) -> list[dict[str, Any]]:
    if outcome == "resolved":
        action = {"tool": "finalize_diagnosis", "arguments": {}}
    else:
        action = {
            "tool": "ask_for_more_evidence",
            "arguments": {
                "state_id": "active",
                "request": "operator_escalation:recovery_options_exhausted",
            },
        }
    return [_trace_row(step=0, action=action, terminal_outcome=outcome)]


def _baseline_policy_observation() -> dict[str, Any]:
    return PolicyObservation(
        active_state_id="active",
        remaining_budget=5,
        history_window=[],
    ).as_dict()


def _post_failure_policy_observation() -> dict[str, Any]:
    failed_output = {
        "execution_status": "failure",
        "error_code": "wls_failure",
        "state_mutated": False,
    }
    history = [
        {
            "state_id": "active",
            "action": {
                "tool": RUN_WLS,
                "arguments": {"state_id": "active"},
            },
            "tool_output": copy.deepcopy(failed_output),
        }
    ]
    return PolicyObservation(
        active_state_id="active",
        remaining_budget=5,
        last_tool=RUN_WLS,
        last_tool_status="failure",
        last_tool_output=failed_output,
        history_window=history,
    ).as_dict()


def _operator_handoff_policy_observation() -> dict[str, Any]:
    state_hash = "7" * 64
    history = [
        {
            "state_id": "active",
            "action": {
                "tool": RUN_WLS,
                "arguments": {"state_id": "active"},
            },
            "tool_output": {
                "execution_status": "success",
                "error_code": None,
                "state_mutated": False,
                "tool_metrics": {
                    "state_id": "active",
                    "state_hash": state_hash,
                    "evidence_source": "deployment_wls:test_fixture",
                    "chi_square_statistic": 5.0,
                    "chi_square_threshold": 2.0,
                    "max_normalized_residual": 3.0,
                    "no_material_anomaly_remaining": False,
                    "globally_resolved": False,
                },
            },
        },
        {
            "state_id": "active",
            "action": {
                "tool": "get_measurement_context",
                "arguments": {"state_id": "active"},
            },
            "tool_output": {
                "execution_status": "success",
                "error_code": None,
                "state_mutated": False,
                "tool_metrics": {
                    "state_id": "active",
                    "state_hash": state_hash,
                    "evidence_source": "deployment_context:test_fixture",
                    "supported_corrections": [],
                },
            },
        },
    ]
    return PolicyObservation(
        active_state_id="active",
        remaining_budget=5,
        history_window=history,
        last_tool="get_measurement_context",
        last_tool_status="success",
        last_tool_output=history[-1]["tool_output"],
        remaining_anomaly_score=2.0,
        no_material_anomaly_remaining=False,
        unresolved_signatures=["unknown_unhandled_anomaly"],
        has_fresh_measurement_context=True,
        measurement_context_state_id="active",
        fresh_context_evidence={
            "measurement": copy.deepcopy(history[-1]["tool_output"]["tool_metrics"])
        },
        semantic_field_provenance={
            "remaining_anomaly_score": "deployment_wls:test_fixture",
            "no_material_anomaly_remaining": "deployment_wls:test_fixture",
            "unresolved_signatures": "deployment_wls:test_fixture",
        },
    ).as_dict()


def _pre_handoff_context_policy_observation() -> dict[str, Any]:
    observation = _operator_handoff_policy_observation()
    observation["has_fresh_measurement_context"] = False
    observation["measurement_context_state_id"] = None
    observation["fresh_context_evidence"] = {}
    observation["requires_measurement_context"] = True
    observation["unresolved_signatures"] = ["measurement_residual:index=0"]
    return observation


def _objective_tool_evidence(
    *,
    tool: str,
    state_id: str,
    state_hash: str,
    statistic: float | None = None,
    threshold: float | None = None,
    max_residual: float | None = None,
    resolved: bool | None = None,
    physically_safe: bool | None = None,
) -> dict[str, Any]:
    violations = (
        []
        if physically_safe is True
        else (
            [{"type": "bus_voltage_out_of_bounds", "bus": 2}]
            if physically_safe is False
            else None
        )
    )
    steady = (
        {
            "scope": "observed_snapshot_topology_vm_rate_a",
            "method": "matpower_case_limits_with_observed_wls_telemetry",
            "complete": True,
            "topology_connectivity": {
                "checked": True,
                "connected": True,
            },
            "bus_voltage_bounds": {
                "checked": True,
                "within_bounds": physically_safe is not False,
            },
            "active_branch_rate_a_bounds": {
                "checked": True,
                "within_defined_rate_a_bounds": True,
            },
            "violation_count": len(violations or []),
            "input_errors": [],
        }
        if physically_safe is not None
        else None
    )
    return {
        "contract": STUDY_OBJECTIVE_TOOL_EVIDENCE_CONTRACT,
        "tool": tool,
        "state_id": state_id,
        "state_hash": state_hash,
        "evidence_source": "deployment_wls:test_fixture",
        "chi_square_statistic": statistic,
        "chi_square_threshold": threshold,
        "max_normalized_residual": max_residual,
        "no_material_anomaly_remaining": resolved,
        "globally_resolved": resolved,
        "physical_constraints_ok": physically_safe,
        "physical_evidence_scope": (
            "observed_snapshot_topology_vm_rate_a"
            if physically_safe is not None
            else None
        ),
        "physical_evidence_complete": (True if physically_safe is not None else None),
        "physical_bound_violations": violations,
        "steady_state_physical_evidence": steady,
        "power_flow_converged": None,
        "topology_feasible": (True if physically_safe is not None else None),
    }


def _schema4_trace_row(
    *,
    step: int,
    action: dict[str, Any],
    observation: dict[str, Any],
    family: str,
    cardinality: int,
    status: str = "success",
    terminal_outcome: str | None = None,
    error_code: str | None = None,
    state_before: dict[str, Any] | None = None,
    state_after: dict[str, Any] | None = None,
    state_mutated: bool = False,
    tool_evidence: dict[str, Any] | None = None,
    policy_output: dict[str, Any] | None = None,
) -> dict[str, Any]:
    before = copy.deepcopy(state_before if state_before is not None else _state())
    after = copy.deepcopy(state_after if state_after is not None else before)
    terminal = terminal_outcome is not None
    assessment = objective_recovery_action_assessment(
        observation,
        scenario_family=family,
        error_cardinality=cardinality,
    )
    persisted_output = (
        copy.deepcopy(policy_output)
        if policy_output is not None
        else {
            "execution_status": status,
            "error_code": error_code,
            "state_mutated": state_mutated,
        }
    )
    if action.get("tool") in {RUN_WLS, VERIFY_CANDIDATE} and policy_output is None:
        persisted_output["tool_metrics"] = (
            {
                key: copy.deepcopy(value)
                for key, value in tool_evidence.items()
                if key not in {"contract", "tool"}
            }
            if isinstance(tool_evidence, dict)
            else {}
        )
    canonical_tool_evidence = objective_tool_evidence(action, persisted_output)
    return {
        "step": step,
        "intervention": False,
        "observation_hash": _artifact_hash(observation),
        "policy_observation": copy.deepcopy(observation),
        "objective_action_assessment": assessment,
        "policy_tool_output": persisted_output,
        "objective_tool_evidence": canonical_tool_evidence,
        "action": copy.deepcopy(action),
        "execution_status": status,
        "advanced": bool(terminal or state_mutated or before != after),
        "error_code": error_code,
        "candidate_disposition_offline": None,
        "tool_regret": None,
        "runtime_state_hash": (
            canonical_tool_evidence.get("state_hash")
            if isinstance(canonical_tool_evidence, dict)
            else None
        ),
        "terminal_outcome": terminal_outcome,
        "state_before": before,
        "state_after": after,
        "state_before_sha256": _artifact_hash(before),
        "state_after_sha256": _artifact_hash(after),
        "state_mutated": state_mutated,
        "terminal_after": terminal,
    }


def _schema4_episode(
    *,
    root: str,
    cardinality: int,
    family: str,
    safe: bool,
    trace: list[dict[str, Any]],
) -> dict[str, Any]:
    episode = _episode(
        root=root,
        cardinality=cardinality,
        family=family,
        safe=safe,
        trace=trace,
    )
    history: list[dict[str, Any]] = []
    for row in episode["trace"]:
        if row["intervention"]:
            row["policy_observation"] = None
            row["objective_action_assessment"] = None
        else:
            observation = copy.deepcopy(row["policy_observation"])
            observation["active_state_id"] = row["state_before"]["active_state_id"]
            observation["candidate_state_id"] = row["state_before"][
                "candidate_state_id"
            ]
            observation["history_window"] = copy.deepcopy(history[-4:])
            observation["last_tool"] = (
                history[-1]["action"]["tool"] if history else None
            )
            observation["last_tool_status"] = (
                history[-1]["tool_output"]["execution_status"]
                if history
                else None
            )
            observation["last_tool_output"] = (
                copy.deepcopy(history[-1]["tool_output"])
                if history
                else {}
            )
            row["policy_observation"] = observation
            row["observation_hash"] = _artifact_hash(observation)
            row["objective_action_assessment"] = (
                objective_recovery_action_assessment(
                    observation,
                    scenario_family=family,
                    error_cardinality=cardinality,
                )
            )
        history.append(
            {
                "state_id": row["state_before"]["active_state_id"],
                "candidate_state_id": row["state_before"]["candidate_state_id"],
                "action": copy.deepcopy(row["action"]),
                "tool_output": copy.deepcopy(row["policy_tool_output"]),
            }
        )
    episode["objective_evidence"] = study_objective_episode_evidence_marker()
    episode["evaluation_intervention"] = {
        "retention_opportunity_count": 0,
    }
    return episode


def _schema4_commit_episode(
    *, root: str, physically_safe: bool | None
) -> dict[str, Any]:
    observation = _baseline_policy_observation()
    active = _state()
    candidate = _state(candidate_state_id="candidate-1")
    promoted = _state(
        active_state_id="candidate-1",
        accepted_correction_count=1,
    )
    verification = _objective_tool_evidence(
        tool=VERIFY_CANDIDATE,
        state_id="candidate-1",
        state_hash="8" * 64,
        statistic=0.5,
        threshold=2.0,
        max_residual=0.1,
        resolved=True,
        physically_safe=physically_safe,
    )
    trace = [
        _schema4_trace_row(
            step=0,
            action={
                "tool": CORRECT_MEASUREMENTS,
                "arguments": {
                    "state_id": "active",
                    "suspect_group": [1],
                },
            },
            observation=observation,
            family="measurement",
            cardinality=1,
            state_before=active,
            state_after=candidate,
            state_mutated=True,
        ),
        _schema4_trace_row(
            step=1,
            action={
                "tool": VERIFY_CANDIDATE,
                "arguments": {"state_id": "candidate-1"},
            },
            observation=observation,
            family="measurement",
            cardinality=1,
            state_before=candidate,
            state_after=candidate,
            tool_evidence=verification,
        ),
        _schema4_trace_row(
            step=2,
            action={
                "tool": COMMIT_STATE,
                "arguments": {"candidate_state_id": "candidate-1"},
            },
            observation=observation,
            family="measurement",
            cardinality=1,
            state_before=candidate,
            state_after=promoted,
            state_mutated=True,
        ),
        _schema4_trace_row(
            step=3,
            action={"tool": FINALIZE_DIAGNOSIS, "arguments": {}},
            observation=observation,
            family="measurement",
            cardinality=1,
            terminal_outcome="resolved",
            state_before=promoted,
            state_after=promoted,
        ),
    ]
    return _schema4_episode(
        root=root,
        cardinality=1,
        family="measurement",
        safe=True,
        trace=trace,
    )


def _target_payload(
    true_targets: dict[str, list[int]],
    accepted_targets: dict[str, list[int]],
) -> dict[str, Any]:
    problems: list[str] = []
    for family in ("measurement", "parameter", "topology"):
        false_targets = sorted(
            set(accepted_targets[family]) - set(true_targets[family])
        )
        if false_targets:
            problems.append(f"{family}_healthy_targets_modified:{false_targets}")
    return {
        "true_targets": copy.deepcopy(true_targets),
        "accepted_targets": copy.deepcopy(accepted_targets),
        "healthy_targets_preserved": not problems,
        "uncovered_standard_faults": sum(
            len(set(true_targets[family]) - set(accepted_targets[family]))
            for family in ("measurement", "parameter", "topology")
        ),
        "problems": problems,
    }


def _episode(
    *,
    root: str,
    cardinality: int,
    family: str,
    safe: bool,
    target_family: str = "measurement",
    accepted_target: int | None = None,
    trace: list[dict[str, Any]] | None = None,
    max_steps: int = 24,
) -> dict[str, Any]:
    true_targets = {
        "measurement": [],
        "parameter": [],
        "topology": [],
    }
    accepted_targets = copy.deepcopy(true_targets)
    true_targets[target_family] = [1]
    if safe:
        accepted_targets[target_family] = [1]
    elif accepted_target is not None:
        accepted_targets[target_family] = [accepted_target]
    outcome = "resolved" if safe else "operator_escalation"
    trace = copy.deepcopy(trace if trace is not None else _terminal_trace(outcome))
    tools = Counter(row["action"]["tool"] for row in trace if not row["intervention"])
    invalid = sum(
        row["action"]["tool"] == "__invalid_action__"
        or row["execution_status"] != "success"
        for row in trace
        if not row["intervention"]
    )
    loop = len(trace) > 1 and all(
        row["action"] == trace[0]["action"] and not row["advanced"] for row in trace[1:]
    )
    target_audit = _target_payload(true_targets, accepted_targets)
    healthy = not target_audit["problems"]
    strict_problems = [] if safe else ["remaining_true_faults"]
    return {
        "episode_key": f"standard_success:{root}:0",
        "scenario_id": root,
        "suite": "standard_success",
        "family": family,
        "cardinality": cardinality,
        "case": "case14",
        "split": "frozen_evaluation",
        "source_tier": "frozen",
        "physical_root": root,
        "seed": 123,
        "steps": len(trace),
        "policy_steps": len(trace),
        "terminal": trace[-1]["terminal_outcome"] is not None,
        "terminal_outcome": trace[-1]["terminal_outcome"],
        "final_physical_correct": safe,
        "physical_correctness_known": safe,
        "final_physical_success": safe,
        "healthy_components_preserved": healthy,
        "healthy_preservation_known": True,
        "false_commit_count": 0,
        "false_rollback_count": 0,
        "false_finalization_count": 0,
        "partial_fix_count": 0,
        "retained_partial_fix_count": 0,
        "invalid_action_count": invalid,
        "recovered_invalid_action_count": 0,
        "loop_detected": loop,
        "wls_calls": sum(tool in {"run_wls", "verify_candidate"} for tool in tools),
        "specialized_tool_calls": 0,
        "tool_counts": dict(sorted(tools.items())),
        "specialized_tool_counts": {},
        "tool_regret_total": 0.0,
        "tool_regret_samples": 0,
        "evaluation_intervention": {},
        "release_environment_attestation": {},
        "policy_identity_attestation": {},
        "control_quarantine": {},
        "audit": {
            "audit_mode": "strict_release_audit",
            "evidence_complete": True,
            "quarantined": not safe,
            "initial_active_state_id": trace[0]["state_before"][
                "active_state_id"
            ],
            "final_active_state_id": trace[-1]["state_after"][
                "active_state_id"
            ],
            "accepted_target_audit": target_audit,
            "strict_release_audit": {
                "audit_version": "strict_offline_episode_truth_v3",
                "terminal": trace[-1]["terminal_outcome"] is not None,
                "terminal_outcome": trace[-1]["terminal_outcome"],
                "scenario_family": family,
                "physical_root_fingerprint": root,
                "quarantined": not safe,
                "problems": strict_problems,
                "checks": {
                    "healthy_measurements_preserved": {
                        "status": "passed" if healthy else "failed",
                        "problems": [],
                    },
                    "healthy_case_components_preserved": {
                        "status": "passed" if healthy else "failed",
                        "problems": [],
                    },
                },
            },
        },
        "trace": trace,
        "evaluator_error": None,
        "_max_steps": max_steps,
    }


def _artifact(
    episodes: list[dict[str, Any]], *, model: str, max_steps: int = 24
) -> dict[str, Any]:
    cleaned = []
    for episode in episodes:
        row = copy.deepcopy(episode)
        row.pop("_max_steps", None)
        cleaned.append(row)
    payload = {
        "artifact_schema_version": 3,
        "artifact_type": "closed_loop_release_evaluation",
        "release_eligible": True,
        "release_failures": [],
        "provenance": {
            "release_eligible": True,
            "release_failures": [],
            "input_suite": {"sha256": SUITE_HASH},
            "policy_identity": {
                "model_id": model,
                "model_revision": ("c" if model == "bc0" else "d") * 64,
            },
        },
        "evaluation": {
            "suite_metrics": {
                "schema_version": 3,
                "configuration": {
                    "seed": 20260719,
                    "max_steps": max_steps,
                    "suite_coverage_validation": {"passed": True},
                    "suite_content_sha256": "e" * 64,
                    "root_set_sha256": stable_json_sha256(
                        sorted(row["physical_root"] for row in cleaned)
                    ),
                    "episode_manifest_sha256": "1" * 64,
                },
                "episodes": cleaned,
            }
        },
    }
    payload["content_sha256"] = _artifact_hash(payload)
    return payload


def _schema4_artifact(
    episodes: list[dict[str, Any]], *, model: str = "full"
) -> dict[str, Any]:
    payload = _artifact(episodes, model=model)
    payload["artifact_schema_version"] = STUDY_EVALUATION_SCHEMA_VERSION
    payload["evaluation"]["suite_metrics"]["schema_version"] = (
        STUDY_EVALUATION_SCHEMA_VERSION
    )
    payload.pop("content_sha256", None)
    payload["content_sha256"] = _artifact_hash(payload)
    return payload


def _rehash_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    artifact.pop("content_sha256", None)
    artifact["content_sha256"] = _artifact_hash(artifact)
    return artifact


def _run(
    *, seed: int, model: str, multi_safe: bool, single_safe: bool
) -> dict[str, Any]:
    artifact = _artifact(
        [
            _episode(
                root="root-multi",
                cardinality=2,
                family="multi_measurement",
                safe=multi_safe,
            ),
            _episode(
                root="root-single",
                cardinality=1,
                family="measurement",
                safe=single_safe,
            ),
        ],
        model=model,
    )
    return extract_artifact_metrics(artifact, variant_id=model, study_seed=seed)


def _tree(variant_id: str, seed: int) -> str:
    return stable_json_sha256({"variant_id": variant_id, "seed": seed})


def _checkpoint(
    *,
    variant_id: str,
    seed: int,
    parent_revision: str,
    parent_checkpoint_receipt_id: str | None = None,
) -> dict[str, Any]:
    manifest = load_study_manifest()
    variant = next(
        item for item in manifest["variants"] if item["variant_id"] == variant_id
    )
    round1_view = {
        "bc0": None,
        "natural_dagger": "natural-only",
        "natural_dagger_probes": "full",
    }[variant_id]
    provenance_id = (
        stable_json_sha256({"view": variant_id, "seed": seed})
        if variant_id == "bc0"
        else ROUND1_PROVENANCE
    )
    quarantine_binding = (
        canonical_production_d1_quarantine_binding("bc0")
        if variant_id == "bc0"
        else build_production_d1_quarantine_binding(
            variant_id=variant_id,
            generation_provenance_id=provenance_id,
            generation_descriptor=ROUND1_DESCRIPTOR,
            summary=QUARANTINE_SUMMARY,
            audit_report_sha256=stable_json_sha256(QUARANTINE_SUMMARY),
        )
    )
    training_protocol = build_training_protocol_binding(
        manifest,
        variant_id=variant_id,
    )
    payload = {
        "artifact_schema_version": 1,
        "artifact_role": "checkpoint",
        "variant_id": variant_id,
        "study_manifest_sha256": EXPECTED_STUDY_MANIFEST_SHA256,
        "reviewed_source_commit": SOURCE_COMMIT,
        "base_model_id": manifest["bindings"]["base_model"]["model_id"],
        "base_model_revision": manifest["bindings"]["base_model"]["model_revision"],
        "base_snapshot_attestation_sha256": "5" * 64,
        "training_seed": seed,
        "training_view_provenance_id": provenance_id,
        "training_protocol": training_protocol,
        "training_configuration": training_protocol["configuration"],
        "training_rng_attestation": canonical_training_rng_attestation(
            variant_id=variant_id,
            training_seed=seed,
        ),
        "parent_checkpoint_receipt_id": parent_checkpoint_receipt_id,
        "training_sources": list(variant["training_sources"]),
        "round1_view": round1_view,
        "production_d1_quarantine_binding": quarantine_binding,
        "training_dataset_sha256": {
            "train": stable_json_sha256(
                {"variant_id": variant_id, "seed": seed, "split": "train"}
            ),
            "validation": stable_json_sha256(
                {"variant_id": variant_id, "seed": seed, "split": "validation"}
            ),
        },
        "parent_model_revision": parent_revision,
        "adapter_path": f"/scratch/{variant_id}-{seed}/lora",
        "adapter_tree_sha256": _tree(variant_id, seed),
        "runtime_accelerator_attestation": copy.deepcopy(ACCELERATOR),
    }
    payload["checkpoint_receipt_id"] = stable_json_sha256(payload)
    return payload


def _bound_evaluation(
    *,
    variant_id: str,
    seed: int | None,
    scope: str,
    checkpoint: dict[str, Any] | None,
    episodes: list[dict[str, Any]],
) -> dict[str, Any]:
    manifest = load_study_manifest()
    base_model = manifest["bindings"]["base_model"]
    if variant_id == "base":
        model_id = base_model["model_id"]
        model_revision = base_model["model_revision"]
        receipt_id = None
        checkpoint_tree = None
    else:
        assert checkpoint is not None
        model_id = f"/scratch/{variant_id}-{seed}"
        model_revision = checkpoint["adapter_tree_sha256"]
        receipt_id = checkpoint["checkpoint_receipt_id"]
        checkpoint_tree = checkpoint["adapter_tree_sha256"]
    payload = _artifact(episodes, model=model_id)
    payload["provenance"]["policy_identity"].update(
        {"model_id": model_id, "model_revision": model_revision}
    )
    payload.update(
        {
            "artifact_role": (
                "evaluation" if scope == "frozen_suite" else "development_evaluation"
            ),
            "variant_id": variant_id,
            "study_manifest_sha256": EXPECTED_STUDY_MANIFEST_SHA256,
            "reviewed_source_commit": SOURCE_COMMIT,
            "model_id": model_id,
            "model_revision": model_revision,
            "checkpoint_receipt_id": receipt_id,
            "checkpoint_adapter_tree_sha256": checkpoint_tree,
            "training_seed": seed,
        }
    )
    if scope == "frozen_suite":
        frozen = manifest["bindings"]["evaluation"]
        payload["frozen_suite_sha256"] = frozen["suite_sha256"]
        payload["evaluation_policy_sha256"] = frozen["policy_sha256"]
        payload["provenance"]["input_suite"]["sha256"] = frozen["suite_sha256"]
    else:
        diagnostic_failure = (
            "diagnostic-only evaluation artifacts are not release evidence"
        )
        payload.update(
            {
                "artifact_type": "closed_loop_diagnostic_evaluation",
                "diagnostic_only": True,
                "release_evidence_eligible": False,
                "training_eligible": False,
                "release_eligible": False,
                "release_failures": [diagnostic_failure],
            }
        )
        payload["provenance"]["release_eligible"] = False
        payload["provenance"]["release_failures"] = [diagnostic_failure]
        payload.update(
            {
                "development_holdout_sha256": DEVELOPMENT_SUITE_HASH,
                "development_holdout_provenance_id": "8" * 64,
                "development_holdout_root_set_sha256": stable_json_sha256(
                    sorted(row["physical_root"] for row in episodes)
                ),
                "development_holdout_physical_roots": 30,
                "development_evaluation_contract_sha256": (
                    EXPECTED_DEVELOPMENT_EVALUATION_CONTRACT_SHA256
                ),
                "evaluation_protocol": "diagnostic_model_selection_only",
            }
        )
        payload["provenance"]["protocol_registry"] = {
            "protocol": "canonical",
            "registry_sha256": "7" * 64,
        }
        payload["provenance"]["input_suite"]["sha256"] = DEVELOPMENT_SUITE_HASH
        configuration = payload["evaluation"]["suite_metrics"]["configuration"]
        configuration.update(
            {
                "seed": 20260721,
                "suite_names": ["dagger1_development"],
                "required_suites": ["dagger1_development"],
                "minimum_suites": 1,
                "minimum_episodes_per_suite": 1,
                "minimum_roots_per_suite": 30,
            }
        )
    payload.pop("content_sha256")
    payload["content_sha256"] = _artifact_hash(payload)
    return payload


def _study_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = load_study_manifest()
    seeds = manifest["training_seeds"]
    base_revision = manifest["bindings"]["base_model"]["model_revision"]
    checkpoints: dict[str, dict[int, dict[str, Any]]] = {
        "bc0": {},
        "natural_dagger": {},
        "natural_dagger_probes": {},
    }
    for seed in seeds:
        checkpoints["bc0"][seed] = _checkpoint(
            variant_id="bc0", seed=seed, parent_revision=base_revision
        )
        bc0_tree = checkpoints["bc0"][seed]["adapter_tree_sha256"]
        bc0_receipt_id = checkpoints["bc0"][seed]["checkpoint_receipt_id"]
        checkpoints["natural_dagger"][seed] = _checkpoint(
            variant_id="natural_dagger",
            seed=seed,
            parent_revision=bc0_tree,
            parent_checkpoint_receipt_id=bc0_receipt_id,
        )
        checkpoints["natural_dagger_probes"][seed] = _checkpoint(
            variant_id="natural_dagger_probes",
            seed=seed,
            parent_revision=bc0_tree,
            parent_checkpoint_receipt_id=bc0_receipt_id,
        )

    frozen_episodes = [
        _episode(
            root="frozen-multi",
            cardinality=2,
            family="multi_measurement",
            safe=True,
        ),
        _episode(
            root="frozen-single",
            cardinality=1,
            family="parameter",
            target_family="parameter",
            safe=True,
        ),
    ]
    development_episodes: list[dict[str, Any]] = []
    for index in range(30):
        single = index >= 24
        target_family = "parameter" if index % 3 == 0 else "measurement"
        development_episodes.append(
            _episode(
                root=f"development-{index:02d}",
                cardinality=1 if single else 2,
                family=(
                    "parameter"
                    if single
                    else (
                        "measurement+parameter"
                        if target_family == "parameter"
                        else "multi_measurement"
                    )
                ),
                target_family=target_family,
                safe=True,
            )
        )
    evaluations: dict[str, Any] = {
        "base": {
            "frozen_suite": _bound_evaluation(
                variant_id="base",
                seed=None,
                scope="frozen_suite",
                checkpoint=None,
                episodes=frozen_episodes,
            ),
            "development_holdout": _bound_evaluation(
                variant_id="base",
                seed=None,
                scope="development_holdout",
                checkpoint=None,
                episodes=development_episodes,
            ),
        }
    }
    for variant_id in (
        "bc0",
        "natural_dagger",
        "natural_dagger_probes",
    ):
        evaluations[variant_id] = {
            scope: {
                seed: _bound_evaluation(
                    variant_id=variant_id,
                    seed=seed,
                    scope=scope,
                    checkpoint=checkpoints[variant_id][seed],
                    episodes=(
                        frozen_episodes
                        if scope == "frozen_suite"
                        else development_episodes
                    ),
                )
                for seed in seeds
            }
            for scope in ("development_holdout", "frozen_suite")
        }
    return evaluations, checkpoints


def test_extracts_exact_targets_safety_and_efficiency_from_episode_evidence() -> None:
    artifact = _artifact(
        [
            _episode(
                root="root-multi",
                cardinality=2,
                family="multi_measurement",
                safe=True,
            ),
            _episode(
                root="root-single",
                cardinality=1,
                family="parameter",
                target_family="parameter",
                safe=True,
            ),
        ],
        model="full",
    )

    run = extract_artifact_metrics(
        artifact, variant_id="natural_dagger_probes", study_seed=3407
    )
    metrics = run["metrics"]

    assert metrics["recovery"]["multi_error_safe_recovery_rate"] == 1.0
    assert metrics["recovery"]["single_error_safe_recovery_rate"] == 1.0
    assert metrics["diagnostic_targets"]["exact_standard_fault_set_rate"] == 1.0
    assert (
        metrics["diagnostic_targets"]["correct_standard_target_cardinality_rate"] == 1.0
    )
    assert metrics["diagnostic_targets"]["target_family_macro_f1"] == 1.0
    assert metrics["safety"]["false_commit_count"] == 0
    assert metrics["efficiency"]["successful_multi_error_median_tool_calls"] == 1
    assert metrics["efficiency"]["schema_valid_tool_call_rate"] == 1.0
    assert (
        metrics["physical_recovery"]["final_residual_chi_square_evidence_status"]
        == "unevaluable"
    )


def test_false_target_is_counted_and_duplicate_truth_fails_closed() -> None:
    bad_target = _episode(
        root="root-false-target",
        cardinality=1,
        family="measurement",
        safe=False,
        accepted_target=2,
    )
    run = extract_artifact_metrics(
        _artifact([bad_target], model="full"),
        variant_id="natural_dagger_probes",
        study_seed=3407,
    )
    metrics = run["metrics"]
    assert metrics["safety"]["healthy_target_corruption_episodes"] == 1
    assert metrics["diagnostic_targets"]["families"]["measurement"]["target_f1"] == 0.0

    ambiguous = _artifact([bad_target], model="full")
    target_audit = ambiguous["evaluation"]["suite_metrics"]["episodes"][0]["audit"][
        "accepted_target_audit"
    ]
    target_audit["true_targets"]["measurement"] = [1, 1]
    ambiguous["content_sha256"] = _artifact_hash(
        {key: value for key, value in ambiguous.items() if key != "content_sha256"}
    )
    with pytest.raises(StudyEvidenceError, match="duplicate targets"):
        extract_artifact_metrics(
            ambiguous,
            variant_id="natural_dagger_probes",
            study_seed=3407,
        )


def test_loop_horizon_invalid_and_redundant_counts_are_trace_derived() -> None:
    repeated = {"tool": "run_wls", "arguments": {"state_id": "active"}}
    trace = [
        _trace_row(step=0, action=repeated),
        _trace_row(
            step=1,
            action=repeated,
            status="failure",
            error_code="evaluation_repeated_nonadvancing_diagnostic",
        ),
    ]
    episode = _episode(
        root="root-loop",
        cardinality=1,
        family="measurement",
        safe=False,
        trace=trace,
        max_steps=2,
    )
    run = extract_artifact_metrics(
        _artifact([episode], model="full", max_steps=2),
        variant_id="natural_dagger_probes",
        study_seed=3407,
    )
    efficiency = run["metrics"]["efficiency"]
    assert efficiency["invalid_action_count"] == 1
    assert efficiency["redundant_action_count"] == 1
    assert efficiency["loop_episodes"] == 1
    assert efficiency["horizon_without_disposition_episodes"] == 1
    assert efficiency["valid_state_bound_proxy"]["valid_rate"] == 1.0


def test_paired_root_bootstrap_is_deterministic_and_requires_exact_roots() -> None:
    seeds = (3407, 3408, 3409)
    bc0 = {
        seed: _run(seed=seed, model="bc0", multi_safe=False, single_safe=True)
        for seed in seeds
    }
    full = {
        seed: _run(seed=seed, model="full", multi_safe=True, single_safe=True)
        for seed in seeds
    }
    policy = {
        "minimum_complete_variant_recovery_rate": 0.9,
        "minimum_mean_primary_improvement": 0.1,
        "require_every_seed_non_degrading_vs_bc0": True,
        "require_bootstrap_95ci_excludes_zero": True,
        "maximum_false_commit_count": 0,
        "maximum_false_finalization_count": 0,
        "maximum_false_rollback_count": 0,
        "maximum_healthy_component_corruption_episodes": 0,
        "maximum_healthy_target_corruption_episodes": 0,
        "maximum_unknown_healthy_preservation_episodes": 0,
        "maximum_evaluator_error_episodes": 0,
        "maximum_single_error_degradation": 0.02,
        "maximum_complete_variant_seed_spread": 0.05,
        "maximum_material_family_regression": 0.02,
        "minimum_training_seed_count": 3,
        "require_no_material_family_regression": True,
        "unsupported_or_empty_family_policy": "fail_closed",
        "maximum_teacher_targets_quarantined_in_production_d1": 0,
        "maximum_finalize_with_unresolved_private_fault_count": 0,
        "maximum_physically_unsafe_commit_count": 0,
        "maximum_truth_safe_accepted_candidate_rollback_count": 0,
        "maximum_hidden_truth_leakage_count": 0,
    }
    first = compare_paired_runs(
        bc0_runs=bc0,
        full_runs=full,
        bootstrap_resamples=500,
        bootstrap_seed=17,
        comparison_policy=policy,
        objective_thresholds=OBJECTIVE_THRESHOLDS,
    )
    second = compare_paired_runs(
        bc0_runs=bc0,
        full_runs=full,
        bootstrap_resamples=500,
        bootstrap_seed=17,
        comparison_policy=policy,
        objective_thresholds=OBJECTIVE_THRESHOLDS,
    )
    assert first["bootstrap_95_percent_ci"] == second["bootstrap_95_percent_ci"]
    assert first["mean_multi_error_delta"] == 1.0
    assert first["bootstrap_95_percent_ci"]["ci_lower"] == 1.0
    # Strong primary episode performance cannot promote the study while the
    # current v3 artifact lacks separately bound objective/safety evidence.
    assert first["passed"] is False
    assert first["primary_decision_passed"] is False
    assert first["objective_decision_passed"] is False
    residual_rule = first["objective_decision_rules"][
        "physical.final_residual_chi_square_acceptance_rate"
    ]
    assert residual_rule["evidence_available"] is False
    assert residual_rule["passed"] is False
    assert residual_rule["numerator"] is None
    assert residual_rule["denominator"] is None
    external_safety = first["decision_rules"][
        "safety_teacher_targets_quarantined_in_production_d1"
    ]
    assert external_safety["evidence_available"] is False
    assert external_safety["passed"] is False
    for rule in [
        *first["decision_rules"].values(),
        *first["objective_decision_rules"].values(),
    ]:
        assert {"evidence_available", "numerator", "denominator", "passed"} <= set(rule)

    mismatched = copy.deepcopy(full)
    mismatched[3407]["root_records"][0]["physical_root"] = "different-root"
    with pytest.raises(StudyEvidenceError, match="physical-root sets"):
        compare_paired_runs(
            bc0_runs=bc0,
            full_runs=mismatched,
            bootstrap_resamples=10,
            comparison_policy=policy,
            objective_thresholds=OBJECTIVE_THRESHOLDS,
        )


def test_content_hash_and_missing_accepted_target_fail_closed() -> None:
    artifact = _artifact(
        [
            _episode(
                root="root-hash",
                cardinality=1,
                family="measurement",
                safe=True,
            )
        ],
        model="full",
    )
    artifact["evaluation"]["suite_metrics"]["episodes"][0]["family"] = "tampered"
    with pytest.raises(StudyEvidenceError, match="content_sha256"):
        extract_artifact_metrics(
            artifact,
            variant_id="natural_dagger_probes",
            study_seed=3407,
        )

    missing_target = _artifact(
        [
            _episode(
                root="root-missing-target",
                cardinality=1,
                family="measurement",
                safe=False,
            )
        ],
        model="full",
    )
    audit = missing_target["evaluation"]["suite_metrics"]["episodes"][0]["audit"][
        "accepted_target_audit"
    ]
    audit["problems"] = ["measurement_accepted_target_missing"]
    audit["healthy_targets_preserved"] = False
    unsigned = {
        key: value for key, value in missing_target.items() if key != "content_sha256"
    }
    missing_target["content_sha256"] = _artifact_hash(unsigned)
    with pytest.raises(StudyEvidenceError, match="target is unknowable"):
        extract_artifact_metrics(
            missing_target,
            variant_id="natural_dagger_probes",
            study_seed=3407,
        )


def test_scope_type_and_recomputed_root_set_fail_closed() -> None:
    release = _artifact(
        [
            _episode(
                root="root-scope",
                cardinality=1,
                family="measurement",
                safe=True,
            )
        ],
        model="full",
    )
    with pytest.raises(StudyEvidenceError, match="diagnostic_evaluation"):
        extract_artifact_metrics(
            release,
            variant_id="natural_dagger_probes",
            study_seed=3407,
            evaluation_scope="development_holdout",
        )

    wrong_roots = copy.deepcopy(release)
    wrong_roots["evaluation"]["suite_metrics"]["configuration"]["root_set_sha256"] = (
        "f" * 64
    )
    wrong_roots.pop("content_sha256")
    wrong_roots["content_sha256"] = _artifact_hash(wrong_roots)
    with pytest.raises(StudyEvidenceError, match="root_set_sha256"):
        extract_artifact_metrics(
            wrong_roots,
            variant_id="natural_dagger_probes",
            study_seed=3407,
        )


def test_schema4_recomputes_recovery_residual_and_handoff_denominators() -> None:
    recovery_observation = _post_failure_policy_observation()
    recovery_action = objective_recovery_action_assessment(
        recovery_observation,
        scenario_family="multi_measurement",
        error_cardinality=2,
    )["expected_action"]
    assert recovery_action == {
        "tool": RUN_WLS,
        "arguments": {"state_id": "active"},
    }
    residual_evidence = _objective_tool_evidence(
        tool=RUN_WLS,
        state_id="active",
        state_hash="6" * 64,
        statistic=0.5,
        threshold=2.0,
        max_residual=0.1,
        resolved=True,
    )
    recovered = _schema4_episode(
        root="root-v4-recovery",
        cardinality=2,
        family="multi_measurement",
        safe=True,
        trace=[
            _schema4_trace_row(
                step=0,
                action={
                    "tool": RUN_WLS,
                    "arguments": {"state_id": "active"},
                },
                observation=_baseline_policy_observation(),
                family="multi_measurement",
                cardinality=2,
                status="failure",
                error_code="wls_failure",
            ),
            _schema4_trace_row(
                step=1,
                action=recovery_action,
                observation=recovery_observation,
                family="multi_measurement",
                cardinality=2,
                terminal_outcome="resolved",
                tool_evidence=residual_evidence,
            )
        ],
    )

    handoff_observation = _operator_handoff_policy_observation()
    handoff_action = objective_recovery_action_assessment(
        handoff_observation,
        scenario_family="measurement",
        error_cardinality=1,
    )["expected_action"]
    assert handoff_action == {
        "tool": ASK_FOR_MORE_EVIDENCE,
        "arguments": {
            "state_id": "active",
            "request": "operator_escalation:recovery_options_exhausted",
        },
    }
    handoff = _schema4_episode(
        root="root-v4-handoff",
        cardinality=1,
        family="measurement",
        safe=False,
        trace=[
            _schema4_trace_row(
                step=0,
                action={
                    "tool": RUN_WLS,
                    "arguments": {"state_id": "active"},
                },
                observation=_baseline_policy_observation(),
                family="measurement",
                cardinality=1,
                tool_evidence=_objective_tool_evidence(
                    tool=RUN_WLS,
                    state_id="active",
                    state_hash="7" * 64,
                    statistic=5.0,
                    threshold=2.0,
                    max_residual=3.0,
                    resolved=False,
                ),
            ),
            _schema4_trace_row(
                step=1,
                action={
                    "tool": "get_measurement_context",
                    "arguments": {"state_id": "active"},
                },
                observation=_pre_handoff_context_policy_observation(),
                family="measurement",
                cardinality=1,
                policy_output=copy.deepcopy(
                    handoff_observation["history_window"][1]["tool_output"]
                ),
            ),
            _schema4_trace_row(
                step=2,
                action=handoff_action,
                observation=handoff_observation,
                family="measurement",
                cardinality=1,
                terminal_outcome="operator_escalation",
            )
        ],
    )
    artifact = _schema4_artifact([recovered, handoff])
    run = extract_artifact_metrics(
        artifact,
        variant_id="natural_dagger_probes",
        study_seed=3407,
    )
    metrics = run["metrics"]
    assert metrics["recovery_action_accuracy"]["post_failure_no_candidate"] == {
        "correct_actions": 1,
        "opportunities": 1,
        "rate": 1.0,
        "evidence_status": "available",
        "evidence_gap": None,
    }
    assert (
        metrics["physical_recovery"]["final_residual_chi_square_acceptance_rate"] == 1.0
    )
    assert (
        metrics["physical_recovery"]["final_residual_chi_square_accepted_episodes"] == 1
    )
    assert metrics["operator_handoff"] == {
        "correct_handoffs": 1,
        "autonomous_exhaustion_opportunities": 1,
        "correct_handoff_rate": 1.0,
        "evidence_status": "available",
        "evidence_gap": None,
    }
    assert metrics["safety"]["hidden_truth_leakage_count"] == 0
    assert metrics["safety"]["hidden_truth_leakage_evidence_available"] is True
    assert metrics["safety"]["policy_observation_count"] == 5

    seeds = (3407, 3408, 3409)
    bc0_artifact = _schema4_artifact([recovered, handoff], model="bc0")
    bc0_runs = {
        seed: extract_artifact_metrics(
            bc0_artifact,
            variant_id="bc0",
            study_seed=seed,
        )
        for seed in seeds
    }
    full_runs = {
        seed: extract_artifact_metrics(
            artifact,
            variant_id="natural_dagger_probes",
            study_seed=seed,
        )
        for seed in seeds
    }
    comparison = compare_paired_runs(
        bc0_runs=bc0_runs,
        full_runs=full_runs,
        comparison_policy=COMPARISON_POLICY,
        objective_thresholds=OBJECTIVE_THRESHOLDS,
        bootstrap_resamples=10,
    )
    assert comparison["objective_decision_rules"][
        "recovery_action_accuracy.post_failure_no_candidate"
    ]["numerator"] == 3
    assert comparison["objective_decision_rules"][
        "recovery_action_accuracy.post_failure_no_candidate"
    ]["denominator"] == 3
    assert comparison["objective_decision_rules"][
        "physical.final_residual_chi_square_acceptance_rate"
    ]["numerator"] == 3
    assert comparison["objective_decision_rules"][
        "physical.final_residual_chi_square_acceptance_rate"
    ]["denominator"] == 3
    assert comparison["objective_decision_rules"][
        "action_quality_efficiency.correct_operator_handoff_rate"
    ]["numerator"] == 3
    assert comparison["objective_decision_rules"][
        "action_quality_efficiency.correct_operator_handoff_rate"
    ]["denominator"] == 3
    assert comparison["decision_rules"][
        "safety_physically_unsafe_commit_count"
    ]["denominator"] == 0
    assert comparison["decision_rules"]["safety_hidden_truth_leakage_count"][
        "denominator"
    ] == 15


def test_schema4_opportunity_denominators_include_incorrect_actions() -> None:
    recovery_observation = _post_failure_policy_observation()
    wrong_recovery = _schema4_episode(
        root="root-v4-wrong-recovery",
        cardinality=1,
        family="measurement",
        safe=True,
        trace=[
            _schema4_trace_row(
                step=0,
                action={
                    "tool": RUN_WLS,
                    "arguments": {"state_id": "active"},
                },
                observation=_baseline_policy_observation(),
                family="measurement",
                cardinality=1,
                status="failure",
                error_code="wls_failure",
            ),
            _schema4_trace_row(
                step=1,
                action={"tool": FINALIZE_DIAGNOSIS, "arguments": {}},
                observation=recovery_observation,
                family="measurement",
                cardinality=1,
                terminal_outcome="resolved",
            )
        ],
    )
    handoff_observation = _operator_handoff_policy_observation()
    wrong_handoff = _schema4_episode(
        root="root-v4-wrong-handoff",
        cardinality=1,
        family="measurement",
        safe=False,
        trace=[
            _schema4_trace_row(
                step=0,
                action={
                    "tool": RUN_WLS,
                    "arguments": {"state_id": "active"},
                },
                observation=_baseline_policy_observation(),
                family="measurement",
                cardinality=1,
                tool_evidence=_objective_tool_evidence(
                    tool=RUN_WLS,
                    state_id="active",
                    state_hash="7" * 64,
                    statistic=5.0,
                    threshold=2.0,
                    max_residual=3.0,
                    resolved=False,
                ),
            ),
            _schema4_trace_row(
                step=1,
                action={
                    "tool": "get_measurement_context",
                    "arguments": {"state_id": "active"},
                },
                observation=_pre_handoff_context_policy_observation(),
                family="measurement",
                cardinality=1,
                policy_output=copy.deepcopy(
                    handoff_observation["history_window"][1]["tool_output"]
                ),
            ),
            _schema4_trace_row(
                step=2,
                action={"tool": FINALIZE_DIAGNOSIS, "arguments": {}},
                observation=handoff_observation,
                family="measurement",
                cardinality=1,
                terminal_outcome="operator_escalation",
            )
        ],
    )
    metrics = extract_artifact_metrics(
        _schema4_artifact([wrong_recovery, wrong_handoff]),
        variant_id="natural_dagger_probes",
        study_seed=3407,
    )["metrics"]
    recovery = metrics["recovery_action_accuracy"]["post_failure_no_candidate"]
    assert recovery["correct_actions"] == 0
    assert recovery["opportunities"] == 1
    assert recovery["rate"] == 0.0
    assert metrics["operator_handoff"]["correct_handoffs"] == 0
    assert metrics["operator_handoff"]["autonomous_exhaustion_opportunities"] == 1
    assert metrics["operator_handoff"]["correct_handoff_rate"] == 0.0


def test_schema4_post_commit_feasibility_and_unsafe_commit_are_bound() -> None:
    safe_run = extract_artifact_metrics(
        _schema4_artifact(
            [_schema4_commit_episode(root="root-v4-safe-commit", physically_safe=True)]
        ),
        variant_id="natural_dagger_probes",
        study_seed=3407,
    )
    safe_metrics = safe_run["metrics"]
    assert safe_metrics["physical_recovery"]["successful_commit_count"] == 1
    assert safe_metrics["physical_recovery"]["post_commit_feasible_commit_count"] == 1
    assert (
        safe_metrics["physical_recovery"][
            "post_commit_power_flow_or_topology_feasibility_rate"
        ]
        == 1.0
    )
    assert safe_metrics["safety"]["physically_unsafe_commit_count"] == 0
    assert safe_metrics["safety"]["physically_unsafe_commit_evidence_available"] is True

    unsafe_run = extract_artifact_metrics(
        _schema4_artifact(
            [
                _schema4_commit_episode(
                    root="root-v4-unsafe-commit",
                    physically_safe=False,
                )
            ]
        ),
        variant_id="natural_dagger_probes",
        study_seed=3407,
    )
    unsafe_metrics = unsafe_run["metrics"]
    assert (
        unsafe_metrics["physical_recovery"][
            "post_commit_power_flow_or_topology_feasibility_rate"
        ]
        == 0.0
    )
    assert unsafe_metrics["safety"]["physically_unsafe_commit_count"] == 1
    assert (
        unsafe_metrics["safety"]["physically_unsafe_commit_evidence_available"] is True
    )
    assert unsafe_metrics["safety"]["safety_violation_physical_roots"] == [
        "root-v4-unsafe-commit"
    ]

    unavailable_run = extract_artifact_metrics(
        _schema4_artifact(
            [
                _schema4_commit_episode(
                    root="root-v4-unbound-commit",
                    physically_safe=None,
                )
            ]
        ),
        variant_id="natural_dagger_probes",
        study_seed=3407,
    )
    unavailable_metrics = unavailable_run["metrics"]
    assert (
        unavailable_metrics["physical_recovery"][
            "post_commit_power_flow_or_topology_feasibility_rate"
        ]
        is None
    )
    assert (
        unavailable_metrics["safety"]["physically_unsafe_commit_evidence_available"]
        is False
    )


@pytest.mark.parametrize(
    "leaked_key",
    (
        "true_measurement_errors",
        "TrueMeasurementErrors",
        "HiddenTruth",
        "GROUND_TRUTH",
    ),
)
def test_schema4_hidden_truth_leakage_is_counted_not_silently_accepted(
    leaked_key: str,
) -> None:
    leaked_observation = _baseline_policy_observation()
    leaked_observation[leaked_key] = [{"index": 1}]
    episode = _schema4_episode(
        root="root-v4-leak",
        cardinality=1,
        family="measurement",
        safe=True,
        trace=[
            _schema4_trace_row(
                step=0,
                action={"tool": FINALIZE_DIAGNOSIS, "arguments": {}},
                observation=leaked_observation,
                family="measurement",
                cardinality=1,
                terminal_outcome="resolved",
            )
        ],
    )
    run = extract_artifact_metrics(
        _schema4_artifact([episode]),
        variant_id="natural_dagger_probes",
        study_seed=3407,
    )
    metrics = run["metrics"]
    assert metrics["safety"]["hidden_truth_leakage_count"] >= 1
    assert metrics["safety"]["hidden_truth_leakage_evidence_available"] is True
    assert (
        metrics["recovery_action_accuracy"]["post_failure_no_candidate"]["rate"] is None
    )
    assert run["root_records"][0]["safe_recovery"] is False


def test_schema4_rejects_missing_or_forged_policy_evidence_after_rehash() -> None:
    observation = _baseline_policy_observation()
    episode = _schema4_episode(
        root="root-v4-policy-integrity",
        cardinality=1,
        family="measurement",
        safe=True,
        trace=[
            _schema4_trace_row(
                step=0,
                action={"tool": FINALIZE_DIAGNOSIS, "arguments": {}},
                observation=observation,
                family="measurement",
                cardinality=1,
                terminal_outcome="resolved",
            )
        ],
    )
    pristine = _schema4_artifact([episode])

    missing_marker = copy.deepcopy(pristine)
    missing_marker["evaluation"]["suite_metrics"]["episodes"][0].pop(
        "objective_evidence"
    )
    with pytest.raises(StudyEvidenceError, match="objective_evidence"):
        extract_artifact_metrics(
            _rehash_artifact(missing_marker),
            variant_id="natural_dagger_probes",
            study_seed=3407,
        )

    missing = copy.deepcopy(pristine)
    del missing["evaluation"]["suite_metrics"]["episodes"][0]["trace"][0][
        "policy_observation"
    ]
    with pytest.raises(StudyEvidenceError, match="policy_observation"):
        extract_artifact_metrics(
            _rehash_artifact(missing),
            variant_id="natural_dagger_probes",
            study_seed=3407,
        )

    forged_observation = copy.deepcopy(pristine)
    forged_observation["evaluation"]["suite_metrics"]["episodes"][0]["trace"][0][
        "policy_observation"
    ]["remaining_budget"] = 4
    with pytest.raises(StudyEvidenceError, match="observation hash is forged"):
        extract_artifact_metrics(
            _rehash_artifact(forged_observation),
            variant_id="natural_dagger_probes",
            study_seed=3407,
        )

    forged_assessment = copy.deepcopy(pristine)
    forged_assessment["evaluation"]["suite_metrics"]["episodes"][0]["trace"][0][
        "objective_action_assessment"
    ]["expected_action"] = {
        "tool": FINALIZE_DIAGNOSIS,
        "arguments": {},
    }
    with pytest.raises(StudyEvidenceError, match="not reproducible"):
        extract_artifact_metrics(
            _rehash_artifact(forged_assessment),
            variant_id="natural_dagger_probes",
            study_seed=3407,
        )


def test_schema4_rejects_coordinated_state_and_history_forgeries() -> None:
    episode = _schema4_episode(
        root="root-v4-observation-binding",
        cardinality=1,
        family="measurement",
        safe=True,
        trace=[
            _schema4_trace_row(
                step=0,
                action={"tool": FINALIZE_DIAGNOSIS, "arguments": {}},
                observation=_baseline_policy_observation(),
                family="measurement",
                cardinality=1,
                terminal_outcome="resolved",
            )
        ],
    )
    pristine = _schema4_artifact([episode])

    state_swap = copy.deepcopy(pristine)
    state_row = state_swap["evaluation"]["suite_metrics"]["episodes"][0][
        "trace"
    ][0]
    state_row["policy_observation"]["active_state_id"] = "forged-active"
    state_row["observation_hash"] = _artifact_hash(state_row["policy_observation"])
    state_row["objective_action_assessment"] = objective_recovery_action_assessment(
        state_row["policy_observation"],
        scenario_family="measurement",
        error_cardinality=1,
    )
    with pytest.raises(StudyEvidenceError, match="contradicts state_before"):
        extract_artifact_metrics(
            _rehash_artifact(state_swap),
            variant_id="natural_dagger_probes",
            study_seed=3407,
        )

    history_swap = copy.deepcopy(pristine)
    history_row = history_swap["evaluation"]["suite_metrics"]["episodes"][0][
        "trace"
    ][0]
    forged_observation = _post_failure_policy_observation()
    history_row["policy_observation"] = forged_observation
    history_row["observation_hash"] = _artifact_hash(forged_observation)
    history_row["objective_action_assessment"] = (
        objective_recovery_action_assessment(
            forged_observation,
            scenario_family="measurement",
            error_cardinality=1,
        )
    )
    with pytest.raises(StudyEvidenceError, match="history is not derived from trace"):
        extract_artifact_metrics(
            _rehash_artifact(history_swap),
            variant_id="natural_dagger_probes",
            study_seed=3407,
        )


def test_schema4_rejects_rehashed_trace_state_identity_substitution() -> None:
    residual = _objective_tool_evidence(
        tool=RUN_WLS,
        state_id="active",
        state_hash="6" * 64,
        statistic=0.5,
        threshold=2.0,
        max_residual=0.1,
        resolved=True,
    )
    episode = _schema4_episode(
        root="root-v4-state-anchor",
        cardinality=1,
        family="measurement",
        safe=True,
        trace=[
            _schema4_trace_row(
                step=0,
                action={"tool": RUN_WLS, "arguments": {"state_id": "active"}},
                observation=_baseline_policy_observation(),
                family="measurement",
                cardinality=1,
                terminal_outcome="resolved",
                tool_evidence=residual,
            )
        ],
    )
    artifact = _schema4_artifact([episode])
    episode = artifact["evaluation"]["suite_metrics"]["episodes"][0]
    original, old_snapshot_hash = _coordinated_trace_state_substitution(
        episode,
        replacement="coordinated-forged-active",
    )
    assert episode["audit"]["initial_active_state_id"] == original
    assert episode["trace"][0]["state_before_sha256"] != old_snapshot_hash

    with pytest.raises(
        StudyEvidenceError,
        match="evaluator-owned initial state identity",
    ):
        extract_artifact_metrics(
            _rehash_artifact(artifact),
            variant_id="natural_dagger_probes",
            study_seed=3407,
        )


def test_schema4_rejects_coordinated_null_objective_state_binding() -> None:
    residual = _objective_tool_evidence(
        tool=RUN_WLS,
        state_id="active",
        state_hash="6" * 64,
        statistic=0.5,
        threshold=2.0,
        max_residual=0.1,
        resolved=True,
    )
    episode = _schema4_episode(
        root="root-v4-null-state-binding",
        cardinality=1,
        family="measurement",
        safe=True,
        trace=[
            _schema4_trace_row(
                step=0,
                action={"tool": RUN_WLS, "arguments": {"state_id": "active"}},
                observation=_baseline_policy_observation(),
                family="measurement",
                cardinality=1,
                terminal_outcome="resolved",
                tool_evidence=residual,
            )
        ],
    )
    forged = _schema4_artifact([episode])
    row = forged["evaluation"]["suite_metrics"]["episodes"][0]["trace"][0]
    row["policy_tool_output"]["tool_metrics"]["state_hash"] = None
    row["objective_tool_evidence"] = objective_tool_evidence(
        row["action"], row["policy_tool_output"]
    )
    row["runtime_state_hash"] = None
    with pytest.raises(StudyEvidenceError, match="non-null state-hash binding"):
        extract_artifact_metrics(
            _rehash_artifact(forged),
            variant_id="natural_dagger_probes",
            study_seed=3407,
        )


def test_schema4_rejects_forged_residual_and_physical_certificates() -> None:
    observation = _baseline_policy_observation()
    action = objective_recovery_action_assessment(
        observation,
        scenario_family="measurement",
        error_cardinality=1,
    )["expected_action"]
    residual = _objective_tool_evidence(
        tool=RUN_WLS,
        state_id="active",
        state_hash="6" * 64,
        statistic=0.5,
        threshold=2.0,
        max_residual=0.1,
        resolved=True,
    )
    episode = _schema4_episode(
        root="root-v4-residual-integrity",
        cardinality=1,
        family="measurement",
        safe=True,
        trace=[
            _schema4_trace_row(
                step=0,
                action=action,
                observation=observation,
                family="measurement",
                cardinality=1,
                terminal_outcome="resolved",
                tool_evidence=residual,
            )
        ],
    )
    pristine = _schema4_artifact([episode])
    contradictory = copy.deepcopy(pristine)
    contradictory["evaluation"]["suite_metrics"]["episodes"][0]["trace"][0][
        "objective_tool_evidence"
    ]["globally_resolved"] = False
    with pytest.raises(StudyEvidenceError, match="not reproducible"):
        extract_artifact_metrics(
            _rehash_artifact(contradictory),
            variant_id="natural_dagger_probes",
            study_seed=3407,
        )

    wrong_state = copy.deepcopy(pristine)
    wrong_state["evaluation"]["suite_metrics"]["episodes"][0]["trace"][0][
        "objective_tool_evidence"
    ]["state_id"] = "different-state"
    with pytest.raises(StudyEvidenceError, match="not reproducible"):
        extract_artifact_metrics(
            _rehash_artifact(wrong_state),
            variant_id="natural_dagger_probes",
            study_seed=3407,
        )

    commit_artifact = _schema4_artifact(
        [_schema4_commit_episode(root="root-v4-cert-integrity", physically_safe=True)]
    )
    forged_certificate = copy.deepcopy(commit_artifact)
    verify_evidence = forged_certificate["evaluation"]["suite_metrics"]["episodes"][0][
        "trace"
    ][1]["objective_tool_evidence"]
    verify_evidence["steady_state_physical_evidence"]["violation_count"] = 1
    with pytest.raises(StudyEvidenceError, match="not reproducible"):
        extract_artifact_metrics(
            _rehash_artifact(forged_certificate),
            variant_id="natural_dagger_probes",
            study_seed=3407,
        )


def test_four_variant_dual_scope_report_is_bound_but_missing_metrics_block() -> None:
    evaluations, checkpoints = _study_inputs()

    report = build_study_report(
        evaluation_artifacts=evaluations,
        checkpoint_artifacts=checkpoints,
        bootstrap_resamples=25,
        bootstrap_seed=7,
        expected_source_commit=SOURCE_COMMIT,
    )

    assert set(report["variant_runs"]) == {
        "base",
        "bc0",
        "natural_dagger",
        "natural_dagger_probes",
    }
    assert set(report["scope_bindings"]) == {
        "development_holdout",
        "frozen_suite",
    }
    assert set(report["variant_metric_summary_by_scope"]["frozen_suite"]) == {
        "base",
        "bc0",
        "natural_dagger",
        "natural_dagger_probes",
    }
    assert report["scope_bindings"]["development_holdout"]["physical_root_count"] == 30
    assert report["checkpoint_binding_decision"]["passed"] is True
    lineage = report["checkpoint_binding_decision"]["bindings"]["natural_dagger:3407"]
    assert (
        lineage["parent_model_revision"]
        == checkpoints["bc0"][3407]["adapter_tree_sha256"]
    )
    assert (
        lineage["parent_checkpoint_receipt_id"]
        == checkpoints["bc0"][3407]["checkpoint_receipt_id"]
    )
    quarantine = report["production_d1_quarantine_decision"]
    assert quarantine["counting_unit"] == "unique_production_d1_corpus"
    assert quarantine["quarantined_rows"] == 0
    assert quarantine["candidate_rows"] == 25
    assert quarantine["variant_bindings"]["base"]["binding"] == (
        canonical_production_d1_quarantine_binding("base")
    )
    assert quarantine["variant_bindings"]["bc0"]["binding"] == (
        canonical_production_d1_quarantine_binding("bc0")
    )
    quarantine_rule = report["primary_comparison_by_scope"]["frozen_suite"][
        "decision_rules"
    ]["safety_teacher_targets_quarantined_in_production_d1"]
    assert quarantine_rule == {
        "observed": 0,
        "numerator": 0,
        "denominator": 25,
        "threshold": {"operator": "<=", "value": 0},
        "evidence_available": True,
        "evidence_failure": None,
        "passed": True,
    }
    assert report["development_stability_decision"]["passed"] is True
    targeted = report["probe_ablation_by_scope"]["frozen_suite"]["rules"][
        "targeted.post_failure_no_candidate_action_accuracy"
    ]
    assert targeted["evidence_available"] is False
    assert targeted["numerator"] is None
    assert targeted["denominator"] is None
    assert targeted["passed"] is False
    assert report["passed"] is False
    assert "probe_ablation_both_scopes" in report["failures"]
    unsigned = {key: value for key, value in report.items() if key != "content_sha256"}
    assert report["content_sha256"] == stable_json_sha256(unsigned)


def test_report_rejects_missing_variant_scope_and_checkpoint_revision_drift() -> None:
    evaluations, checkpoints = _study_inputs()
    missing_natural = copy.deepcopy(evaluations)
    missing_natural.pop("natural_dagger")
    with pytest.raises(StudyEvidenceError, match="preregistration"):
        build_study_report(
            evaluation_artifacts=missing_natural,
            checkpoint_artifacts=checkpoints,
            bootstrap_resamples=5,
            expected_source_commit=SOURCE_COMMIT,
        )

    missing_scope = copy.deepcopy(evaluations)
    missing_scope["base"].pop("development_holdout")
    with pytest.raises(StudyEvidenceError, match="preregistration"):
        build_study_report(
            evaluation_artifacts=missing_scope,
            checkpoint_artifacts=checkpoints,
            bootstrap_resamples=5,
            expected_source_commit=SOURCE_COMMIT,
        )

    drift = copy.deepcopy(evaluations)
    run = drift["natural_dagger"]["frozen_suite"][3407]
    run["checkpoint_receipt_id"] = "f" * 64
    run.pop("content_sha256")
    run["content_sha256"] = _artifact_hash(run)
    with pytest.raises(StudyEvidenceError, match="checkpoint receipt"):
        build_study_report(
            evaluation_artifacts=drift,
            checkpoint_artifacts=checkpoints,
            bootstrap_resamples=5,
            expected_source_commit=SOURCE_COMMIT,
        )


@pytest.mark.parametrize(
    ("variant_id", "lineage_field", "match"),
    [
        (
            "natural_dagger",
            "parent_checkpoint_receipt_id",
            "same-seed BC0 parent checkpoint receipt ID",
        ),
        (
            "natural_dagger_probes",
            "parent_model_revision",
            "same-seed BC0 checkpoint tree",
        ),
    ],
)
def test_report_rejects_cross_seed_bc0_parent_lineage_even_after_rehash(
    variant_id: str,
    lineage_field: str,
    match: str,
) -> None:
    evaluations, checkpoints = _study_inputs()
    seed = 3407
    other_seed = 3408
    changed = checkpoints[variant_id][seed]
    if lineage_field == "parent_checkpoint_receipt_id":
        changed[lineage_field] = checkpoints["bc0"][other_seed]["checkpoint_receipt_id"]
    else:
        changed[lineage_field] = checkpoints["bc0"][other_seed]["adapter_tree_sha256"]
    changed.pop("checkpoint_receipt_id")
    changed["checkpoint_receipt_id"] = stable_json_sha256(changed)
    for scope in ("development_holdout", "frozen_suite"):
        evaluation = evaluations[variant_id][scope][seed]
        evaluation["checkpoint_receipt_id"] = changed["checkpoint_receipt_id"]
        evaluation.pop("content_sha256")
        evaluation["content_sha256"] = _artifact_hash(evaluation)

    with pytest.raises(StudyEvidenceError, match=match):
        build_study_report(
            evaluation_artifacts=evaluations,
            checkpoint_artifacts=checkpoints,
            bootstrap_resamples=5,
            expected_source_commit=SOURCE_COMMIT,
        )


def test_report_rejects_cross_receipt_quarantine_evidence_disagreement() -> None:
    evaluations, checkpoints = _study_inputs()
    changed_summary = {
        **QUARANTINE_SUMMARY,
        "total_rows": 26,
        "candidate_rows": 26,
        "passed_rows": 26,
    }
    changed_descriptor = {
        **ROUND1_DESCRIPTOR,
        "audit_report_sha256": {
            "d1_offline_teacher_target_quarantine_summary": (
                stable_json_sha256(changed_summary)
            )
        },
    }
    changed_provenance = stable_json_sha256(changed_descriptor)
    changed = checkpoints["natural_dagger_probes"][3409]
    changed["training_view_provenance_id"] = changed_provenance
    changed["production_d1_quarantine_binding"] = (
        build_production_d1_quarantine_binding(
            variant_id="natural_dagger_probes",
            generation_provenance_id=changed_provenance,
            generation_descriptor=changed_descriptor,
            summary=changed_summary,
            audit_report_sha256=stable_json_sha256(changed_summary),
        )
    )
    changed.pop("checkpoint_receipt_id")
    changed["checkpoint_receipt_id"] = stable_json_sha256(changed)
    for scope in ("development_holdout", "frozen_suite"):
        evaluation = evaluations["natural_dagger_probes"][scope][3409]
        evaluation["checkpoint_receipt_id"] = changed["checkpoint_receipt_id"]
        evaluation.pop("content_sha256")
        evaluation["content_sha256"] = _artifact_hash(evaluation)

    with pytest.raises(StudyEvidenceError, match="disagree on the exact"):
        build_study_report(
            evaluation_artifacts=evaluations,
            checkpoint_artifacts=checkpoints,
            bootstrap_resamples=5,
            expected_source_commit=SOURCE_COMMIT,
        )
