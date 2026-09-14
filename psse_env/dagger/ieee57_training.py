"""Observable, step-audited balanced supervision and fresh-controller replay.

This is an expert-only mechanics pilot, not DAgger learner recovery. Private
truth can quarantine a fixed target but cannot replace it or steer the teacher.
"""
from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from typing import Any, Mapping

from psse_env.actions import invalid_action, safe_normalize_action
from psse_env.state_store import policy_safe_copy
from psse_env.dagger.dataset_builder import (
    alias_model_visible_state, bind_controller_action, examples_to_chat_sft,
    prepare_model_policy_observation, validate_policy_payload,
)
from psse_env.dagger.offline_teacher_target_audit import (
    offline_teacher_target_audit, validate_offline_teacher_target_audit_metadata,
)
from psse_env.dagger.protocol_bridge import (
    canonical_to_internal_action, internal_to_canonical_action,
)
from psse_env.dagger.release_factories import (
    deterministic_case_loader, select_observable_expert_actions,
)
from psse_env.dagger.sft_audit import audit_chat_sft_rows, audit_teacher_realizability

CONTRACT = "ieee57_observable_audited_balanced_pilot_v1"


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def private_runtime_scenario(envelope: Mapping[str, Any]) -> dict:
    """The environment isolates these fields in OracleState, never the teacher."""
    runtime = copy.deepcopy(dict(envelope["execution"]))
    truth = copy.deepcopy(dict(envelope["audit"]["truth"]))
    clean = truth.pop("clean_state", {})
    for key, alias in (("case", "clean_case"), ("measurements", "clean_measurements")):
        if key in clean:
            truth[alias] = clean[key]
    runtime.update(truth)
    if "release_audit" in envelope["audit"]:
        runtime["release_audit"] = copy.deepcopy(envelope["audit"]["release_audit"])
    return runtime


def _observation(env, history):
    value = env.get_policy_observation(history)
    obs = value.as_dict() if hasattr(value, "as_dict") else copy.deepcopy(dict(value))
    obs = json.loads(json.dumps(obs, sort_keys=True, allow_nan=False))
    validate_policy_payload(obs)
    return obs


def _model_view(observation):
    return prepare_model_policy_observation(
        observation, history=observation.get("history_window", []),
        max_history_events=8, max_history_chars=4096,
        alias_before_compaction=True,
    )


def _canonical(action, aliases):
    mapped = alias_model_visible_state(
        action, aliases["state_aliases"], episode_aliases=aliases["episode_aliases"],
        hash_aliases=aliases["hash_aliases"],
    )
    return internal_to_canonical_action(mapped)


def _detector_outputs(output, tool=None):
    """Check actual emitted detector settings, including nested verification."""
    from .ieee57_runtime import validate_ieee57_wls_metrics
    found = 0
    def visit(value):
        nonlocal found
        if isinstance(value, Mapping):
            if "anomaly_detection_rule" in value:
                validate_ieee57_wls_metrics(value)
                found += 1
            if "chi_square_alpha" in value and value["chi_square_alpha"] is not None:
                if abs(float(value["chi_square_alpha"]) - .05) > 1e-12:
                    raise ValueError("IEEE57 emitted chi-square setting drift")
                found += 1
            if "normalized_residual_threshold" in value and value["normalized_residual_threshold"] is not None:
                if abs(float(value["normalized_residual_threshold"]) - 4.) > 1e-12:
                    raise ValueError("IEEE57 emitted normalized-residual setting drift")
                found += 1
            for child in value.values():
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)
    visit(output)
    if tool in {"run_wls", "verify_candidate"} and output.get("execution_status") == "success":
        validate_ieee57_wls_metrics(output.get("tool_metrics", {}))
        found += 1
    return found


def supervision_reasons(row: Mapping[str, Any]) -> list[str]:
    reasons = []
    if not row.get("training_decision_evidence_verified"):
        reasons.append("observable_evidence_gate_failed")
    try:
        validate_offline_teacher_target_audit_metadata(
            row.get("offline_teacher_target_audit"), require_passed=True,
        )
    except ValueError:
        reasons.append("offline_teacher_target_audit_failed")
    if row.get("tool_output", {}).get("execution_status") != "success":
        reasons.append("protocol_execution_not_successful")
    if row.get("target_selection_changed_during_audit"):
        reasons.append("target_mutated_after_selection")
    if row.get("canonical_action") is None:
        reasons.append("canonical_mapping_failed")
    if not row.get("parent_construction_fingerprint") or not row.get("parent_plan_sha256"):
        reasons.append("parent_provenance_missing")
    return reasons


def collect_episode(envelope, *, max_steps=40, environment_factory=None):
    from .ieee57_runtime import ieee57_environment_factory, ieee57_expert_oracle_factory, validate_ieee57_runtime
    factory = environment_factory or ieee57_environment_factory
    env = factory()
    validate_ieee57_runtime(env)
    runtime = private_runtime_scenario(envelope)
    if len(runtime["measurements"]) != 491:
        raise ValueError("Balanced IEEE57 collection requires 491 measurements")
    env.reset(runtime)
    expert = ieee57_expert_oracle_factory()
    history, rows = [], []
    grouping = envelope["grouping"]
    for step in range(max_steps):
        validate_ieee57_runtime(env)
        observation = _observation(env, history)
        # Fix the observable target BEFORE obtaining private state or truth.
        selection = select_observable_expert_actions(
            policy_observation=observation, expert_oracle=expert,
        )
        # Match ObservableExpertPolicy's declared no-action behavior. It is an
        # invalid, quarantined transition, never a truth-selected replacement.
        action = safe_normalize_action(copy.deepcopy(selection.preferred_action)
            if selection.preferred_action is not None else invalid_action("observable_expert_returned_no_action"))
        before_target = digest(action)
        evidence_error = None
        try:
            env.assert_training_decision_evidence(action)
        except ValueError as exc:
            evidence_error = str(exc)
        audit = offline_teacher_target_audit(
            preferred_action=action, oracle_state=env.get_oracle_state(history),
            policy_observation=observation, scenario=runtime, env=env,
            observable_evidence_passed=evidence_error is None,
            case_loader=deterministic_case_loader,
        )
        if _model_view(_observation(env, history))[0] != _model_view(observation)[0]:
            raise ValueError("Private target audit changed observable execution state")
        model_view, aliases = _model_view(observation)
        canonical_error = None
        try:
            canonical_action = _canonical(action, aliases)
        except ValueError as exc:
            canonical_action, canonical_error = None, str(exc)
        next_state, output = env.step(action)
        validate_ieee57_runtime(env)
        detector_fields = _detector_outputs(output, action["tool"])
        next_history = history + [{"state_id": observation.get("active_state_id"),
                                  "action": policy_safe_copy(action), "tool_output": policy_safe_copy(output)}]
        next_view, _ = _model_view(_observation(env, next_history))
        row = {
            "example_id": f"{runtime['scenario_id']}:step:{step}",
            "scenario_id": runtime["scenario_id"], "root_scenario_id": grouping["root_scenario_id"],
            "physical_root_fingerprint": grouping["physical_root_fingerprint"],
            "scenario_family": grouping["scenario_family"], "error_cardinality": grouping["error_cardinality"],
            "source_realization_id": grouping["source_realization_id"],
            "parent_construction_fingerprint": grouping["parent_construction_fingerprint"],
            "parent_plan_sha256": grouping["parent_plan_sha256"],
            "dataset_split": grouping.get("dataset_split", grouping["split"]),
            "network_case": "case57", "step": step, "iteration": 0,
            "dataset_mode": "production", "collector_contract": CONTRACT,
            "dataset_source": "fresh_parent_assigned_observable_expert",
            "supervision_policy": CONTRACT, "policy_observation": observation,
            "history_window": copy.deepcopy(observation.get("history_window", [])),
            "preferred_action": action, "valid_next_actions": [action],
            "deferred_expert_actions": list(selection.actions[1:]),
            "target_selection_basis": selection.selection_basis,
            "target_fixed_before_private_audit": True,
            "target_selection_changed_during_audit": digest(action) != before_target,
            "training_decision_evidence_verified": evidence_error is None,
            "observable_evidence_error": evidence_error,
            "offline_teacher_target_audit": audit,
            "tool_output": policy_safe_copy(output),
            "model_view": model_view, "canonical_action": canonical_action,
            "canonical_mapping_error": canonical_error,
            "next_model_view": next_view,
            "detector_output_fields_checked": detector_fields,
            "terminal_after_action": bool(env.is_terminal(next_state)),
            "terminal_outcome": env.terminal_outcome,
            "semantic_field_provenance": copy.deepcopy(observation.get("semantic_field_provenance", {})),
        }
        row["quarantine_reasons"] = supervision_reasons(row)
        row["production_label_eligible"] = not row["quarantine_reasons"]
        rows.append(row)
        history = next_history
        if env.is_terminal(next_state):
            break
    return rows


def export_audited_rows(rows):
    """Only train labels; validation/test trajectories never enter this export."""
    accepted, quarantined = [], []
    for original in rows:
        row = copy.deepcopy(original)
        if row["dataset_split"] != "train":
            continue
        reasons = supervision_reasons(row)
        if row.get("quarantine_reasons") != reasons or row.get("production_label_eligible") != (not reasons):
            raise ValueError("Supervision audit ledger was modified")
        if reasons:
            quarantined.append({"example_id": row["example_id"], "family": row["scenario_family"],
                                "tool": row["preferred_action"]["tool"], "reasons": reasons})
        else:
            accepted.append(row)
    exported = examples_to_chat_sft(accepted, protocol="canonical", alias_before_compaction=True)
    if len(exported) != len(accepted):
        raise ValueError("Accepted target disappeared during export")
    for raw, chat in zip(accepted, exported):
        for key in ("source_realization_id", "parent_construction_fingerprint", "parent_plan_sha256", "dataset_split"):
            chat[key] = raw[key]
        chat["metadata"]["ieee57_supervision"] = {
            "contract": CONTRACT, "target_audit_passed": True,
            "parent_plan_sha256": raw["parent_plan_sha256"],
            "model_view_contract": "alias_before_compaction_v1",
        }
        if json.loads(chat["messages"][1]["content"])["state"] != raw["model_view"]:
            raise ValueError("Export changed the audited visible observation")
    realizability = audit_teacher_realizability(exported, conflict_tolerance=0)
    if not realizability["passed"]:
        raise ValueError(f"Observable targets conflict: {realizability}")
    structure = audit_chat_sft_rows(exported)
    if not structure["passed"]:
        raise ValueError(f"Canonical SFT audit failed: {structure}")
    return exported, {"accepted": len(accepted), "quarantined": len(quarantined),
                      "quarantine": quarantined, "realizability": realizability, "structure": structure}


def replay_episode(envelope, rows, exported_rows, *, environment_factory=None):
    """Replay raw prefixes, and execute exported targets using fresh aliases."""
    from .ieee57_runtime import ieee57_environment_factory, ieee57_expert_oracle_factory, validate_ieee57_runtime
    env = (environment_factory or ieee57_environment_factory)()
    validate_ieee57_runtime(env)
    runtime = private_runtime_scenario(envelope)
    runtime["scenario_id"] += "_fresh_replay"
    env.reset(runtime)
    exported = {row["example_id"]: row for row in exported_rows}
    if len(exported) != len(exported_rows):
        raise ValueError("Duplicate exported target IDs")
    if set(exported) - {row["example_id"] for row in rows}:
        raise ValueError("Export contains a target absent from this raw episode")
    history, checks = [], []
    expert = ieee57_expert_oracle_factory()
    for original in rows:
        validate_ieee57_runtime(env)
        observation = _observation(env, history)
        view, aliases = _model_view(observation)
        if view != original["model_view"]:
            raise ValueError(f"Replay visible-state mismatch at {original['example_id']}")
        action = original["canonical_action"]
        row = exported.get(original["example_id"])
        if row is not None:
            if json.loads(row["messages"][1]["content"])["state"] != view:
                raise ValueError("Exported replay observation mismatch")
            call = row["messages"][2]["tool_calls"][0]["function"]
            action = {"tool": call["name"], "arguments": call["arguments"]}
            if action != original["canonical_action"]:
                raise ValueError("Exported target differs from fixed audited target")
        selected = select_observable_expert_actions(policy_observation=observation, expert_oracle=expert)
        preferred = safe_normalize_action(selected.preferred_action if selected.preferred_action is not None
                                         else invalid_action("observable_expert_returned_no_action"))
        if action is None:
            if row is not None or original["preferred_action"]["tool"] != "__invalid_action__":
                raise ValueError("Unreplayable canonical target")
            rebound = copy.deepcopy(original["preferred_action"])
        else:
            rebound = bind_controller_action(canonical_to_internal_action(action), aliases["state_aliases"])
        if preferred != rebound:
            raise ValueError("Observable teacher decision changed on replay")
        evidence = True
        try:
            env.assert_training_decision_evidence(rebound)
        except ValueError:
            evidence = False
        audit = offline_teacher_target_audit(
            preferred_action=rebound, oracle_state=env.get_oracle_state(history),
            policy_observation=observation, scenario=runtime, env=env,
            observable_evidence_passed=evidence, case_loader=deterministic_case_loader,
        )
        if _model_view(_observation(env, history))[0] != view:
            raise ValueError("Replay private audit changed observable execution state")
        if row is not None:
            validate_offline_teacher_target_audit_metadata(audit, require_passed=True)
        next_state, output = env.step(rebound)
        validate_ieee57_runtime(env)
        _detector_outputs(output, rebound["tool"])
        if output.get("execution_status") != original["tool_output"].get("execution_status"):
            raise ValueError("Replay protocol outcome changed")
        if row is not None and output.get("execution_status") != "success":
            raise ValueError("Exported target did not execute successfully")
        checks.append({"example_id": original["example_id"], "exported": row is not None,
                       "fresh_state_id": observation["active_state_id"],
                       "original_state_id": original["policy_observation"]["active_state_id"],
                       "fresh_alias_rebinding": observation["active_state_id"] != original["policy_observation"]["active_state_id"],
                       "offline_target_audit_passed": audit["passed"], "passed": True})
        history.append({"state_id": observation.get("active_state_id"), "action": rebound,
                        "tool_output": policy_safe_copy(output)})
        next_view, _ = _model_view(_observation(env, history))
        if (next_view != original["next_model_view"]
                or bool(env.is_terminal(next_state)) != original["terminal_after_action"]
                or env.terminal_outcome != original["terminal_outcome"]):
            raise ValueError(f"Replay post-action state or terminal outcome changed at {original['example_id']}")
    if not all(row["fresh_alias_rebinding"] for row in checks):
        raise ValueError("Replay did not use fresh controller identities")
    return {"scenario_id": envelope["execution"]["scenario_id"], "steps": len(checks),
            "exported_targets_replayed": sum(row["exported"] for row in checks),
            "passed": True, "checks": checks}


def supervision_counts(rows):
    counts = Counter()
    for row in rows:
        verdict = "accepted" if row["production_label_eligible"] else "quarantined"
        counts[(row["dataset_split"], row["scenario_family"], row["preferred_action"]["tool"], verdict)] += 1
    return [{"split": split, "family": family, "tool": tool, "verdict": verdict, "count": count}
            for (split, family, tool, verdict), count in sorted(counts.items())]
