from __future__ import annotations

import copy
from typing import Any, Iterable, Mapping

from psse_env.actions import (
    CORRECT_MEASUREMENTS,
    CORRECT_PARAMETERS,
    CORRECT_TOPOLOGY,
    GET_MEASUREMENT_CONTEXT,
    GET_PARAMETER_CONTEXT,
    GET_TOPOLOGY_CONTEXT,
    RUN_WLS,
)
from psse_env.state_store import policy_safe_copy
from psse_env.oracle.process_validity import ProcessValidityOracle
from .error_injectors import InjectedAction, plausible_wrong_actions


class CounterfactualGenerator:
    """Bootstrap recovery rows by executing isolated transactional branches."""

    def __init__(self, *, env: Any, expert_oracle: Any) -> None:
        self.env = env
        self.expert_oracle = expert_oracle

    def generate(
        self,
        *,
        scenario: Mapping[str, Any],
        wrong_actions: Iterable[InjectedAction] | None = None,
        expert_actions: Iterable[Mapping[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        self.env.reset(scenario)
        root_observation = self.env.get_policy_observation([])
        root_oracle = self.env.get_oracle_state([])
        if expert_actions is None:
            expert_actions = list(self.expert_oracle.next_actions(root_oracle, []))
            expert_actions.extend(self._truth_correction_actions(root_oracle))
        physical_state = self.env.store.get_state(root_observation.active_state_id)
        injected = list(
            wrong_actions
            or plausible_wrong_actions(
                root_observation.as_dict(), expert_actions, physical_state=physical_state
            )
        )
        return self.generate_from_current(
            injected,
            root_scenario_id=str(
                scenario.get("root_scenario_id", scenario.get("scenario_id", "scenario"))
            ),
            physical_root_fingerprint=scenario.get("physical_root_fingerprint"),
        )

    def generate_from_current(
        self,
        injected_actions: Iterable[InjectedAction],
        *,
        root_scenario_id: str,
        physical_root_fingerprint: str | None = None,
    ) -> list[dict[str, Any]]:
        root_hash = self.env.store.episode_hash()
        rows: list[dict[str, Any]] = []
        for index, injected in enumerate(injected_actions):
            branch = self.env.clone()
            for setup_action in injected.setup_actions:
                bound_setup = self._bind_dynamic_state_ids(branch, setup_action)
                self._ensure_correction_context(branch, bound_setup, injected.family)
                branch.step(bound_setup)
            bound_action = self._bind_dynamic_state_ids(branch, injected.action)
            self._ensure_correction_context(branch, bound_action, injected.family)
            injection_parent = branch.get_policy_observation(branch.history)
            injection_validity = self._process_validity(branch, bound_action)
            next_state, output = branch.step(copy.deepcopy(bound_action))
            post_injection = branch.get_policy_observation(branch.history)
            injection_transition = {
                "parent_state_summary": injection_parent.as_dict(),
                "action": copy.deepcopy(bound_action),
                "tool_output": policy_safe_copy(output),
                "next_state_summary": post_injection.as_dict(),
                "candidate_state_summary": post_injection.as_dict() if post_injection.candidate_state_id else {},
            }
            verification_output = None
            verification_transition = None
            if next_state.get("has_unverified_candidate") and injected.family != "skipped_verification":
                candidate_id = next_state["candidate_state_id"]
                verification_parent = branch.get_policy_observation(branch.history)
                verification_action = {"tool": RUN_WLS, "arguments": {"state_id": candidate_id}}
                verification_validity = self._process_validity(branch, verification_action)
                next_state, verification_output = branch.step(
                    verification_action
                )
                post_verification = branch.get_policy_observation(branch.history)
                verification_transition = {
                    "parent_state_summary": verification_parent.as_dict(),
                    "action": verification_action,
                    "tool_output": policy_safe_copy(verification_output),
                    "next_state_summary": post_verification.as_dict(),
                    "candidate_state_summary": post_verification.as_dict(),
                }
            history = copy.deepcopy(branch.history)
            reached_oracle = branch.get_oracle_state(history)
            recovery_actions = self.expert_oracle.next_actions(reached_oracle, history)
            disposition = reached_oracle.get("candidate_disposition")
            assessment = getattr(reached_oracle, "candidate_assessment", {}) or {}
            injection_transition["labels"] = {
                "process_valid": bool(injection_validity.get("process_valid")),
            }
            if verification_transition is not None:
                verification_transition["labels"] = {
                    "process_valid": bool(verification_validity.get("process_valid")),
                    "candidate_disposition": disposition,
                    "progress_class": assessment.get("progress_class"),
                }
            reached_observation = branch.get_policy_observation(history)
            recovery_output = None
            continuation_actions: list[dict[str, Any]] = []
            recovered_observation = reached_observation
            recovery_validity: Mapping[str, Any] = {"process_valid": False}
            target_evidence_sufficient: bool | None = None
            target_evidence_error: str | None = None
            if recovery_actions:
                if bool(getattr(branch, "production_dataset_mode", False)):
                    assertion = getattr(branch, "assert_training_decision_evidence", None)
                    if not callable(assertion):
                        target_evidence_sufficient = False
                        target_evidence_error = "training_decision_evidence_assertion_unavailable"
                    else:
                        try:
                            assertion(recovery_actions[0])
                        except ValueError as exc:
                            target_evidence_sufficient = False
                            target_evidence_error = str(exc)
                        else:
                            target_evidence_sufficient = True
                recovery_branch = branch.clone()
                recovery_validity = self._process_validity(recovery_branch, recovery_actions[0])
                _, recovery_output = recovery_branch.step(copy.deepcopy(recovery_actions[0]))
                recovered_history = copy.deepcopy(recovery_branch.history)
                recovered_observation = recovery_branch.get_policy_observation(recovered_history)
                if not recovery_branch.is_terminal():
                    continuation_actions = self.expert_oracle.next_actions(
                        recovery_branch.get_oracle_state(recovered_history),
                        recovered_history,
                    )
            if disposition == "REJECT":
                state_class = "rejected_candidate_recovery"
            elif disposition == "ACCEPT_PARTIAL":
                state_class = "accepted_partial_continuation"
            else:
                state_class = "invalid_precondition_recovery"
            # The wrong branch may be constructed from hidden truth.  Its
            # recovery trace is useful auxiliary supervision, but it must not
            # silently enter the production single-label SFT corpus even when
            # the backing environment uses deployment providers.
            dataset_mode = "synthetic_counterfactual"
            rows.append(
                {
                    "example_id": f"counterfactual_{root_scenario_id}_{index}",
                    "scenario_id": root_scenario_id,
                    "root_scenario_id": root_scenario_id,
                    "physical_root_fingerprint": physical_root_fingerprint,
                    "dataset_mode": dataset_mode,
                    "dataset_source": "synthetic_counterfactual",
                    "production_label_eligible": False,
                    "target_evidence_sufficient": target_evidence_sufficient,
                    "target_evidence_error": target_evidence_error,
                    "branch_family": injected.family,
                    "policy_observation": reached_observation.as_dict(),
                    "parent_state_summary": reached_observation.as_dict(),
                    "state_summary": reached_observation.as_dict(),
                    "history_window": copy.deepcopy(reached_observation.history_window),
                    "executed_action": copy.deepcopy(recovery_actions[0]) if recovery_actions else None,
                    "executed_by": "expert_recovery" if recovery_actions else None,
                    "state_visited_by": "counterfactual_injection",
                    "tool_output": policy_safe_copy(recovery_output or {}),
                    "next_state_summary": recovered_observation.as_dict(),
                    "candidate_state_summary": (
                        recovered_observation.as_dict() if recovered_observation.candidate_state_id else {}
                    ),
                    "injected_action": copy.deepcopy(bound_action),
                    "injected_tool_output": policy_safe_copy(output),
                    "injection_transition": injection_transition,
                    "verification_transition": verification_transition,
                    "verification_output": policy_safe_copy(verification_output),
                    "preferred_action": copy.deepcopy(recovery_actions[0]) if recovery_actions else None,
                    "valid_next_actions": copy.deepcopy(recovery_actions),
                    "recovery_output": copy.deepcopy(recovery_output),
                    "continuation_actions": copy.deepcopy(continuation_actions),
                    "labels": {
                        "candidate_disposition": disposition,
                        "process_valid": bool(recovery_validity.get("process_valid")),
                        "state_class": state_class,
                        "dataset_mode": dataset_mode,
                        "dataset_source": "synthetic_counterfactual",
                        "production_label_eligible": False,
                        "target_evidence_sufficient": target_evidence_sufficient,
                    },
                    "state_class": state_class,
                }
            )
        if self.env.store.episode_hash() != root_hash:
            raise RuntimeError("Counterfactual branch mutated the root environment.")
        return rows

    def _process_validity(self, branch: Any, action: Mapping[str, Any]) -> Mapping[str, Any]:
        process_oracle = getattr(self.expert_oracle, "process_oracle", None)
        if process_oracle is None:
            process_oracle = ProcessValidityOracle()
        return process_oracle.check(
            branch.current_state(),
            action,
            store=getattr(branch, "store", None),
        )

    @staticmethod
    def _truth_correction_actions(oracle_state: Any) -> list[dict[str, Any]]:
        """Materialize bounded synthetic corrections when hints stop at context gathering."""
        active_id = oracle_state.get("active_state_id")
        actions: list[dict[str, Any]] = []
        measurement_errors = list(getattr(oracle_state, "true_measurement_errors", []) or [])
        for fault in measurement_errors:
            if not isinstance(fault, Mapping):
                continue
            index = fault.get("index", fault.get("index0", fault.get("measurement_index")))
            value = fault.get("clean", fault.get("clean_value", fault.get("true_value")))
            if value is None and index is not None:
                clean_measurements = getattr(oracle_state, "clean_measurements", None)
                try:
                    if isinstance(clean_measurements, (list, tuple)):
                        value = clean_measurements[int(index)]
                except (TypeError, ValueError, IndexError):
                    value = None
            if index is not None and value is not None:
                actions.append(
                    {
                        "tool": CORRECT_MEASUREMENTS,
                        "arguments": {"state_id": active_id, "measurement_updates": {index: value}},
                    }
                )
        parameter_errors = list(getattr(oracle_state, "true_parameter_errors", []) or [])
        for fault in parameter_errors:
            if not isinstance(fault, Mapping):
                continue
            target_key = next(
                (
                    key
                    for key in ("branch_id", "cb_name", "branch_row0", "line_index1", "line_index")
                    if fault.get(key) is not None
                ),
                None,
            )
            line_index = fault.get(target_key) if target_key else None
            value = fault.get("clean", fault.get("clean_value", fault.get("true_value")))
            if value is None and line_index is not None:
                clean_values = getattr(oracle_state, "hidden_truth", {}).get("clean_parameter_values")
                if isinstance(clean_values, Mapping):
                    value = clean_values.get(line_index, clean_values.get(str(line_index)))
                elif isinstance(clean_values, (list, tuple)):
                    try:
                        normalized_index = int(line_index) - (1 if target_key == "line_index1" else 0)
                        value = clean_values[normalized_index]
                    except (TypeError, ValueError, IndexError):
                        value = None
            arguments: dict[str, Any] = {"state_id": active_id}
            if target_key is not None:
                arguments[target_key] = line_index
            if fault.get("parameter") is not None:
                arguments["parameter"] = fault["parameter"]
            if value is not None:
                arguments["value"] = value
            if len(arguments) > 1:
                actions.append({"tool": CORRECT_PARAMETERS, "arguments": arguments})
        topology_errors = list(getattr(oracle_state, "true_topology_errors", []) or [])
        for fault in topology_errors:
            if not isinstance(fault, Mapping):
                continue
            target_key = next(
                (
                    key
                    for key in ("branch_id", "cb_name", "branch_row0", "line_index1", "line_index")
                    if fault.get(key) is not None
                ),
                None,
            )
            branch_id = fault.get(target_key) if target_key else None
            status = fault.get("expected_status", fault.get("clean", fault.get("true_value")))
            arguments = {"state_id": active_id}
            if target_key is not None:
                arguments[target_key] = branch_id
            if status is not None:
                arguments["status"] = status
            if len(arguments) > 1:
                actions.append({"tool": CORRECT_TOPOLOGY, "arguments": arguments})
        return actions

    @staticmethod
    def _bind_dynamic_state_ids(branch: Any, action: Mapping[str, Any]) -> dict[str, Any]:
        bound = copy.deepcopy(dict(action))
        arguments = bound.get("arguments")
        arguments = dict(arguments) if isinstance(arguments, Mapping) else {}
        state = branch.current_state()
        if arguments.get("state_id") == "__active__":
            arguments["state_id"] = state.get("active_state_id")
        elif arguments.get("state_id") == "__candidate__":
            arguments["state_id"] = state.get("candidate_state_id")
        if arguments.get("candidate_state_id") == "__candidate__":
            arguments["candidate_state_id"] = state.get("candidate_state_id")
        bound["arguments"] = arguments
        return bound

    @staticmethod
    def _ensure_correction_context(branch: Any, action: Mapping[str, Any], family: str) -> None:
        correction_tool = action.get("tool")
        context_tool = {
            CORRECT_MEASUREMENTS: GET_MEASUREMENT_CONTEXT,
            CORRECT_PARAMETERS: GET_PARAMETER_CONTEXT,
            CORRECT_TOPOLOGY: GET_TOPOLOGY_CONTEXT,
        }.get(correction_tool)
        if context_tool is None:
            return
        observation = branch.get_policy_observation(branch.history)
        if (
            correction_tool != CORRECT_MEASUREMENTS
            or observation.requires_measurement_context
            or family == "wrong_fault_family"
        ):
            branch.step(
                {
                    "tool": context_tool,
                    "arguments": {"state_id": observation.active_state_id},
                }
            )
