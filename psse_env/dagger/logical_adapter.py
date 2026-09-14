"""Pure logical-topology adapter over the production transactional controller.

This opt-in scope preserves raw physical-section sensors. It does not implement
mixed meter/parameter repair, AC operating-limit certification, or phase tools.
Private status truth audits outcomes; it never selects actions or dispositions.
"""
from __future__ import annotations

import copy
from collections.abc import Mapping

from logical_topology.atomic import validate_atomic_arguments
from logical_topology.provider import LogicalTopologyProviders, _document
from logical_topology.runtime import evidence_hash
from psse_env.actions import (
    ASK_FOR_MORE_EVIDENCE, COMMIT_STATE, CORRECT_TOPOLOGY, FINALIZE_DIAGNOSIS,
    GET_TOPOLOGY_CONTEXT, POST_CORRECTION_CONFIRMATION_SIGNATURE, ROLLBACK_STATE,
    RUN_WLS, VERIFY_CANDIDATE, safe_normalize_action, waveform_anomaly_signatures,
)
from psse_env.oracle.candidate_quality import CandidateAssessment, CandidateDisposition, CandidateQualityOracle
from psse_env.oracle.process_validity import ProcessValidityOracle
from psse_env.state_store import find_forbidden_policy_paths, policy_safe_copy
from psse_env.transactional_env import TransactionalPSSEEnv
from .logical_protocol import ALLOWED_INTERNAL, canonical_to_internal_action, internal_to_canonical_action

CONTRACT = "ieee57_pure_logical_transaction_v1"
INCONCLUSIVE_REQUEST = "operator_escalation:logical_topology_inconclusive"
COMPLETION_SCOPE = "pure_logical_topology_fixed_raw_evidence_within_declared_hypothesis_scope"


def _logical(state):
    return state["metadata"]["logical_topology"]


def _statuses(state):
    return _logical(state)["current_statuses"]


def _fresh_context(state):
    context = (state.get("fresh_context_evidence") or {}).get("topology") or {}
    return context if (state.get("has_fresh_topology_context")
                       and context.get("state_id") == state.get("active_state_id")) else {}


def _clean_fit(metrics):
    return (metrics.get("state_estimation_converged") is True
            and metrics.get("observable") is True
            and metrics.get("chi_square_alpha") == .05
            and metrics.get("normalized_residual_threshold") == 4.0
            and metrics.get("chi_square_alarm") is False
            and metrics.get("normalized_residual_alarm") is False
            and metrics.get("no_material_anomaly_remaining") is True)


class LogicalProcessOracle(ProcessValidityOracle):
    def _has_correction_payload(self, tool, args):
        if tool != CORRECT_TOPOLOGY:
            return False
        try:
            validate_atomic_arguments(args)
            return True
        except ValueError:
            return False

    def check(self, state, action, *, store=None):
        action = safe_normalize_action(action)
        if action["tool"] not in ALLOWED_INTERNAL:
            return {"process_valid": False, "reason": "pure_logical_route_unsupported",
                    "error_code": "pure_logical_route_unsupported", "error_detail": action["tool"],
                    "valid_next_actions": []}
        if action["tool"] == CORRECT_TOPOLOGY:
            # Pure logical deployments do not declare harmonic/phase channels;
            # the legacy waveform-acquisition preconditions belong to a wider
            # experiment contract. Preserve actual waveform evidence as a block.
            args = action["arguments"]
            error = None
            if state.get("has_open_candidate"):
                error = ("candidate_lifecycle_violation", "correction_with_open_candidate")
            elif not self._known_current_state(store, args.get("state_id")):
                error = ("unknown_state_id", str(args.get("state_id")))
            elif args.get("state_id") != state.get("active_state_id"):
                error = ("state_reference_mismatch", "correction_state_not_active")
            elif waveform_anomaly_signatures(state.get("unresolved_signatures") or []):
                error = ("pure_logical_route_unsupported", "waveform_evidence_requires_separate_adapter")
            elif not self._has_correction_payload(CORRECT_TOPOLOGY, args):
                error = ("schema_error", "invalid_atomic_logical_payload")
            else:
                error = self._context_supported_correction_failure(state, action)
            return {"process_valid": error is None, "reason": error[1] if error else None,
                    "error_code": error[0] if error else None, "error_detail": error[1] if error else None,
                    "valid_next_actions": []}
        return super().check(state, action, store=store)

    def _terminal_condition_met(self, state):
        context = _fresh_context(state)
        return (context.get("logical_decision") == "keep_current"
                and context.get("scope_complete") is True
                and state.get("no_material_anomaly_remaining") is True
                and not state.get("has_open_candidate"))


class LogicalCandidateQualityOracle(CandidateQualityOracle):
    """Observable disposition, with private exact-status retirement separate."""
    def __init__(self):
        super().__init__(mode="deployment")

    def label_candidate(self, *, parent_state, source_action, candidate_state,
                        verification_output, hidden_truth=None):
        # hidden_truth deliberately cannot influence acceptance or rejection.
        valid = (verification_output.get("logical_transition_verified") is True
                 and _clean_fit(verification_output))
        return CandidateAssessment(
            disposition=CandidateDisposition.ACCEPT_FINAL if valid else CandidateDisposition.REJECT,
            progress_class="logical_complete_correction" if valid else "unsupported_logical_correction",
            rationale_codes=["observable_certificate_and_exact_transition_verified" if valid
                             else "observable_logical_transition_or_fit_failed"],
            unresolved_signatures=[] if valid else ["logical_topology_investigation_required"])

    def matched_fault_indices(self, action, hidden_truth, *, parent_state=None, candidate_state=None):
        if not parent_state or not candidate_state:
            return []
        changes = (action.get("arguments") or {}).get("desired_statuses") or {}
        parent, candidate = _statuses(parent_state), _statuses(candidate_state)
        return [index for index, fault in enumerate(hidden_truth.get("true_topology_errors") or [])
                if (fault.get("device_id") in changes
                    and parent.get(fault["device_id"]) != fault.get("true_status")
                    and candidate.get(fault["device_id"]) == fault.get("true_status"))]


class LogicalTopologyEnv(TransactionalPSSEEnv):
    provider_kind = "deployment"

    def __init__(self, *, providers=None, scan_options=None, derived_case_dir=None, **kwargs):
        self.logical_providers = providers or LogicalTopologyProviders(
            atomic_actions=True, scan_options=scan_options, **({"derived_case_dir": derived_case_dir} if derived_case_dir else {}))
        if not self.logical_providers.atomic_actions:
            raise ValueError("logical agent adapter requires atomic certificate-bound actions")
        if self.logical_providers.chi2_alpha != .05 or self.logical_providers.normalized_residual_threshold != 4.0:
            raise ValueError("IEEE57 logical pilot pins chi2 alpha .05 and normalized residual threshold 4.0")
        forbidden = {"wls_runner", "process_oracle", "candidate_quality_oracle", "production_dataset_mode",
                     "context_providers", "correction_executors", "evidence_providers"} & set(kwargs)
        if forbidden:
            raise ValueError(f"logical execution contract cannot override {sorted(forbidden)}")
        p = self.logical_providers
        super().__init__(wls_runner=self._logical_wls, process_oracle=LogicalProcessOracle(),
            candidate_quality_oracle=LogicalCandidateQualityOracle(), production_dataset_mode=True,
            context_providers={GET_TOPOLOGY_CONTEXT: p.get_topology_context},
            correction_executors={CORRECT_TOPOLOGY: p.correct_topology}, **kwargs)

    def validate_production_configuration(self):
        # This named adapter intentionally exposes only the pure logical tools.
        if set(self.context_providers) != {GET_TOPOLOGY_CONTEXT} or set(self.correction_executors) != {CORRECT_TOPOLOGY}:
            raise ValueError("pure logical provider contract changed")
        return {"production_dataset_mode": True, "contract": CONTRACT, "completion_scope": COMPLETION_SCOPE,
                "chi2_alpha": .05, "normalized_residual_threshold": 4.0,
                "mixed_measurement_parameter_routes": "unsupported"}

    def reset(self, scenario):
        result = super().reset(scenario)
        self.logical_providers._contexts.clear()
        self._logical_root_id = self.store.active_state_id
        self.context_flags["requires_measurement_context"] = False
        self.logical_providers._context(self.store.get_state(self._logical_root_id))
        return self.current_state()

    def _verify_transition(self, parent, candidate):
        action = candidate.get("source_action") or {}
        args = action.get("arguments") or {}
        validate_atomic_arguments(args)
        if action.get("tool") != CORRECT_TOPOLOGY or args["state_id"] != parent["state_id"]:
            raise ValueError("candidate parent/source action mismatch")
        provider = self.logical_providers
        parent_entry = provider._context(parent, create=False)
        audit = parent_entry["audit"] or {}
        if (audit.get("unique_candidate_id") != args["candidate_id"]
            or evidence_hash(audit.get("certificate")) != args["certificate_hash"]):
            raise ValueError("parent certificate no longer matches the requested candidate")
        chosen = next((row for row in audit.get("plausible_candidates", []) if row["candidate_id"] == args["candidate_id"]), None)
        if chosen is None or chosen["changes"] != args["desired_statuses"]:
            raise ValueError("desired statuses do not equal the complete certified change")
        # Read candidate files before any derivation; verification must never
        # regenerate a derived file and conceal an external model modification.
        candidate_entry = provider._context(candidate)
        expected_runtime = copy.deepcopy(parent_entry["runtime"])
        expected = expected_runtime.apply(args["candidate_id"])
        actual = candidate_entry["runtime"].snapshot()
        for key in ("inventory", "current_case", "current_statuses", "measurement_inventory", "observations"):
            if evidence_hash(actual[key]) != evidence_hash(expected[key]):
                raise ValueError(f"candidate differs from certified intervention: {key}")
        stored_certificate = _logical(candidate).get("last_identification_certificate")
        if evidence_hash(stored_certificate) != args["certificate_hash"]:
            raise ValueError("candidate certificate provenance changed")
        return {"logical_transition_verified": True, "logical_certificate_hash": args["certificate_hash"],
                "logical_desired_statuses": copy.deepcopy(args["desired_statuses"]),
                "measurements_and_covariance_preserved": True,
                "completion_scope": COMPLETION_SCOPE}

    def _logical_wls(self, state):
        metrics = self.logical_providers.run_wls(state)
        if state.get("parent_state_id") and (state.get("source_action") or {}).get("tool") == CORRECT_TOPOLOGY:
            try:
                parent = self.store.get_state(state["parent_state_id"])
                metrics.update(self._verify_transition(parent, state))
            except (ValueError, KeyError) as exc:
                metrics.update(logical_transition_verified=False, logical_transition_error=str(exc))
        return metrics

    @staticmethod
    def _target_decision_evidence_missing(metrics, disposition, **kwargs):
        if disposition == CandidateDisposition.ACCEPT_FINAL.value:
            return [] if metrics.get("logical_transition_verified") is True and _clean_fit(metrics) else ["logical_certificate_or_fit_missing"]
        if disposition == CandidateDisposition.REJECT.value:
            return [] if metrics.get("logical_transition_verified") is False or not _clean_fit(metrics) else ["logical_rejection_evidence_missing"]
        return ["pure_logical_partial_or_inconclusive_transaction_unsupported"]

    def candidate_decision_evidence(self, candidate_state_id=None):
        report = super().candidate_decision_evidence(candidate_state_id)
        candidate_id = candidate_state_id or self.current_candidate_id
        if report["sufficient"] and candidate_id:
            candidate = self.store.get_state(candidate_id)
            if candidate.get("candidate_disposition") == CandidateDisposition.ACCEPT_FINAL.value:
                try:
                    self._verify_transition(self.store.get_state(candidate["parent_state_id"]), candidate)
                except (ValueError, KeyError) as exc:
                    report["sufficient"] = False
                    report["missing"].append(f"logical_commit_recheck:{exc}")
        return report

    def _persist_observable_semantics(self, metrics, *, source, replace_missing=False):
        previous_resolution = self.context_flags.get("no_material_anomaly_remaining", False)
        previous_source = self.context_flags.get("semantic_field_provenance", {}).get("no_material_anomaly_remaining")
        super()._persist_observable_semantics(metrics, source=source, replace_missing=replace_missing)
        # Replaced by explicit same-state logical confirmation in the process
        # oracle. The legacy measurement route remains guarded and uncalled.
        self.context_flags["unresolved_signatures"] = [item for item in self.context_flags.get("unresolved_signatures", [])
                                                       if item != POST_CORRECTION_CONFIRMATION_SIGNATURE]
        if "no_material_anomaly_remaining" in metrics:
            self.context_flags["no_material_anomaly_remaining"] = _clean_fit(metrics)
            self._set_semantic_provenance("no_material_anomaly_remaining", source)
        elif not replace_missing:
            self.context_flags["no_material_anomaly_remaining"] = previous_resolution
            if previous_source:
                self._set_semantic_provenance("no_material_anomaly_remaining", previous_source)

    def _step_context(self, action):
        output = super()._step_context(action)
        if output["execution_status"] == "success":
            context = self.context_flags["fresh_context_evidence"]["topology"]
            metrics = output["tool_metrics"]
            for key in ("logical_decision", "scope_complete", "hypothesis_scope", "unique_candidate_id",
                        "tested_candidate_count", "plausible_candidate_count", "unresolved_candidate_count", "certificate", "logical_binding_key"):
                if key in metrics:
                    context[key] = policy_safe_copy(metrics[key])
        return output

    def assert_training_decision_evidence(self, action):
        action = safe_normalize_action(action)
        if action["tool"] == GET_TOPOLOGY_CONTEXT:
            # A declared pure-topology screening rule runs after WLS on every
            # root, including non-alarming roots; no hidden family routing.
            self._assert_active_wls()
            return
        if action["tool"] == FINALIZE_DIAGNOSIS:
            self._assert_context_binding()
            if not self.process_oracle._terminal_condition_met(self.current_state()):
                raise ValueError("fresh complete logical confirmation required")
            return
        if action["tool"] == ASK_FOR_MORE_EVIDENCE:
            if not self._inconclusive_supported(action):
                raise ValueError("logical handoff requires same-state inconclusive evidence")
            return
        return super().assert_training_decision_evidence(action)

    def _inconclusive_supported(self, action):
        try:
            self._assert_context_binding()
        except (ValueError, KeyError):
            return False
        state = self.current_state()
        context = _fresh_context(state)
        complete_confirmation = (context.get("logical_decision") == "keep_current"
                                 and context.get("scope_complete") is True
                                 and state.get("no_material_anomaly_remaining") is True)
        return (action["arguments"].get("request") == INCONCLUSIVE_REQUEST and bool(context)
                and not complete_confirmation and not context.get("supported_corrections"))

    def _assert_active_wls(self):
        active = self.store.get_state(self.store.active_state_id)
        metrics = self._latest_bound_successful_tool_metrics((RUN_WLS,)) or {}
        entry = self.logical_providers._context(active, create=False)
        if (metrics.get("state_id") != active["state_id"] or metrics.get("state_hash") != active["state_hash"]
            or metrics.get("logical_binding_key") != entry["binding_key"]):
            raise ValueError("logical context requires successful WLS bound to the current active model, evidence, and configuration")

    def _assert_context_binding(self):
        active = self.store.get_state(self.store.active_state_id)
        context = _fresh_context(self.current_state())
        entry = self.logical_providers._context(active, create=False)
        audit = entry["audit"] or {}
        if (not context or not audit or context.get("state_hash") != active["state_hash"]
            or context.get("logical_binding_key") != entry["binding_key"]
            or context.get("logical_decision") != audit.get("decision")
            or context.get("scope_complete") != audit.get("scope_complete")
            or evidence_hash(context.get("certificate")) != evidence_hash(audit.get("certificate"))):
            raise ValueError("fresh logical context must match the current model, raw evidence, scan configuration, and audit")

    def dispatch_valid_action(self, action):
        if action["tool"] in {GET_TOPOLOGY_CONTEXT, FINALIZE_DIAGNOSIS}:
            try:
                if action["tool"] == GET_TOPOLOGY_CONTEXT:
                    self._assert_active_wls()
                else:
                    self._assert_context_binding()
            except (ValueError, KeyError) as exc:
                return self.record_noop_failure(action=action, error_code="stale_logical_evidence", error_detail=str(exc), valid_next_actions=[])
        if action["tool"] == ASK_FOR_MORE_EVIDENCE:
            if not self._inconclusive_supported(action):
                return self.record_noop_failure(action=action, error_code="logical_handoff_unsupported", valid_next_actions=[])
            self.terminal, self.terminal_outcome = True, "inconclusive"
            return self._standard_output(execution_status="success", state_mutated=False,
                tool_metrics={"terminal_outcome": "inconclusive", "request": INCONCLUSIVE_REQUEST,
                              "logical_decision": _fresh_context(self.current_state()).get("logical_decision"),
                              "completion_scope": COMPLETION_SCOPE})
        return super().dispatch_valid_action(action)

    def clone(self):
        # Keep the provider cache, WLS, and transaction store on the same cloned
        # environment. Generic collaborator cloning would split those bindings.
        return copy.deepcopy(self)

    def step_canonical(self, action):
        return self.step(canonical_to_internal_action(action))

    def private_target_audit(self, candidate_state_id):
        candidate = self.store.get_state(candidate_state_id)
        parent = self.store.get_state(candidate["parent_state_id"])
        truth = self._oracle_payload.get("logical_true_statuses")
        before, after = _statuses(parent), _statuses(candidate)
        changes = {key: value for key, value in after.items() if before.get(key) != value}
        requested = (candidate.get("source_action", {}).get("arguments") or {}).get("desired_statuses", {})
        fixed = (parent["measurements"] == candidate["measurements"]
                 and _document(_logical(parent)["measurement_inventory"], "measurement inventory")
                 == _document(_logical(candidate)["measurement_inventory"], "measurement inventory"))
        try:
            self._verify_transition(parent, candidate)
            intervention_preserved = True
        except (ValueError, KeyError):
            intervention_preserved = False
        correct = (all(key in truth and after[key] == truth[key] and before[key] != truth[key] for key in changes)
                   if isinstance(truth, Mapping) and changes else None)
        healthy = (not any(before.get(key) == truth.get(key) for key in changes)
                   if isinstance(truth, Mapping) else None)
        return {"candidate_state_id": candidate_state_id, "logical_device_changes": changes,
                "requested_statuses_correct": correct, "healthy_component_preserved": healthy,
                "only_declared_statuses_changed": changes == requested,
                "measurements_and_covariance_preserved": fixed,
                "exact_certified_intervention_preserved": intervention_preserved,
                "accepted_target_audit_passed": bool(correct and healthy and fixed and intervention_preserved and changes == requested)}

    def private_outcome_audit(self):
        root = self.store.get_state(self._logical_root_id)
        active = self.store.get_state(self.store.active_state_id)
        original, final = _statuses(root), _statuses(active)
        truth = self._oracle_payload.get("logical_true_statuses")
        sensors = _document(_logical(root)["measurement_inventory"], "measurement inventory")
        final_sensors = _document(_logical(active)["measurement_inventory"], "measurement inventory")
        fixed = root["measurements"] == active["measurements"] and sensors == final_sensors
        accepted_audits = [self.private_target_audit(row["candidate_state_id"])
                           for row in self.context_flags.get("accepted_corrections", [])]
        accepted_safe = all(row["accepted_target_audit_passed"] for row in accepted_audits)
        report = {"contract": "logical_private_status_audit_v1", "raw_measurement_count": len(root["measurements"]),
                  "measurements_and_covariance_preserved": fixed, "truth_available": isinstance(truth, Mapping),
                  "logical_changes": {key: value for key, value in final.items() if value != original.get(key)},
                  "terminal_outcome": self.terminal_outcome, "completion_scope": COMPLETION_SCOPE,
                  "accepted_target_audits": accepted_audits}
        if isinstance(truth, Mapping) and set(truth) == set(original) == set(final):
            remaining = [key for key in truth if final[key] != truth[key]]
            healthy_changed = [key for key in truth if original[key] == truth[key] and final[key] != original[key]]
            report.update(exact_statuses_recovered=not remaining, remaining_logical_device_ids=remaining,
                          healthy_component_preserved=not healthy_changed, healthy_device_changes=healthy_changed,
                          strict_resolved=self.terminal_outcome == "resolved" and not remaining and fixed and not healthy_changed and accepted_safe)
        else:
            report.update(exact_statuses_recovered=None, healthy_component_preserved=None, strict_resolved=False)
        return report


def logical_environment_factory(**kwargs):
    return LogicalTopologyEnv(**kwargs)


def logical_scenario(providers, *, case, inventory, reported_statuses, sensors, observations,
                     scenario_id, parent_id, true_statuses=None):
    payload = providers.state_payload(case, inventory, reported_statuses, sensors, observations)
    payload.update(scenario_id=scenario_id)
    payload["metadata"]["physical_parent_id"] = str(parent_id)
    if true_statuses is not None:
        if set(true_statuses) != set(reported_statuses):
            raise ValueError("private logical truth must cover exactly the full declared device inventory")
        payload["hidden_truth"] = {"truth_complete": True, "logical_true_statuses": copy.deepcopy(true_statuses),
            "true_topology_errors": [{"family": "topology", "device_id": device, "true_status": true_statuses[device]}
                                     for device in true_statuses if reported_statuses[device] != true_statuses[device]]}
    return payload


def observable_logical_teacher(observation):
    """Policy over observable context only; never receives the environment/truth."""
    state = observation.as_dict() if hasattr(observation, "as_dict") else dict(observation)
    forbidden = find_forbidden_policy_paths(state)
    if forbidden:
        raise ValueError(f"private fields supplied to logical teacher: {forbidden}")
    active, candidate = state["active_state_id"], state.get("candidate_state_id")
    if state.get("has_unverified_candidate"):
        return {"tool": VERIFY_CANDIDATE, "arguments": {"state_id": candidate}}
    if state.get("has_verified_candidate"):
        metrics = state.get("last_verification") or {}
        commit = metrics.get("logical_transition_verified") is True and _clean_fit(metrics)
        return {"tool": COMMIT_STATE if commit else ROLLBACK_STATE, "arguments": {"candidate_state_id": candidate}}
    context = _fresh_context(state)
    if context:
        supported = context.get("supported_corrections") or []
        if supported:
            return copy.deepcopy(supported[0])
        if context.get("logical_decision") == "keep_current" and context.get("scope_complete") is True and state.get("no_material_anomaly_remaining") is True:
            return {"tool": FINALIZE_DIAGNOSIS, "arguments": {}}
        return {"tool": ASK_FOR_MORE_EVIDENCE, "arguments": {"state_id": active, "request": INCONCLUSIVE_REQUEST}}
    # Every root and committed state gets an explicit WLS call before screening.
    if state.get("last_tool") == RUN_WLS and state.get("last_tool_status") == "success":
        return {"tool": GET_TOPOLOGY_CONTEXT, "arguments": {"state_id": active}}
    return {"tool": RUN_WLS, "arguments": {"state_id": active}}


def run_logical_episode(env, scenario, *, max_steps=16):
    env.reset(scenario)
    steps = []
    for _ in range(max_steps):
        observation = env.get_policy_observation().as_dict()
        action = observable_logical_teacher(observation)
        env.assert_training_decision_evidence(action)
        validity = env.process_oracle.check(env.current_state(), action, store=env.store)
        canonical = internal_to_canonical_action(action)
        if canonical_to_internal_action(canonical) != action:
            raise ValueError("logical canonical round-trip changed action")
        _, output = env.step_canonical(canonical)
        steps.append({"observation": observation, "action": canonical, "output": policy_safe_copy(output),
                      "observable_target_audit": {"supported": validity["process_valid"], "protocol_roundtrip": True,
                                                  "execution_success": output["execution_status"] == "success"}})
        if env.terminal or output["execution_status"] != "success":
            break
    return {"contract": CONTRACT, "physical_parent_id": scenario["metadata"]["physical_parent_id"],
            "scenario_id": scenario["scenario_id"], "steps": steps, "private_audit": env.private_outcome_audit()}
