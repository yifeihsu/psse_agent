from __future__ import annotations

from typing import Any, Mapping

from psse_env.actions import (
    ASK_FOR_MORE_EVIDENCE,
    COMMIT_STATE,
    CONTEXT_TOOLS,
    CORRECT_MEASUREMENTS,
    CORRECT_PARAMETERS,
    CORRECT_TOPOLOGY,
    CORRECTION_TOOLS,
    DIAGNOSTIC_TOOLS,
    FINALIZE_DIAGNOSIS,
    GET_MEASUREMENT_CONTEXT,
    GET_PARAMETER_CONTEXT,
    GET_TOPOLOGY_CONTEXT,
    INVALID_ACTION,
    MACRO_ACTIONS,
    POST_CORRECTION_CONFIRMATION_SIGNATURE,
    RECOVERY_OPTIONS_EXHAUSTED_REQUEST,
    ROLLBACK_STATE,
    RUN_ALTERNATIVE_TEST,
    RUN_WLS,
    VERIFY_CANDIDATE,
    action_signature,
    safe_normalize_action,
    terminal_explanation_signatures,
    unexplained_signatures,
)
from psse_env.state_store import SYNTHETIC_TERMINAL_COMPATIBILITY_KEY


_CORRECTION_CONTEXT_FAMILY = {
    CORRECT_MEASUREMENTS: "measurement",
    CORRECT_PARAMETERS: "parameter",
    CORRECT_TOPOLOGY: "topology",
}
_CONTEXT_TOOL_FOR_FAMILY = {
    "measurement": GET_MEASUREMENT_CONTEXT,
    "parameter": GET_PARAMETER_CONTEXT,
    "topology": GET_TOPOLOGY_CONTEXT,
}
_CORRECTION_TOOL_FOR_FAMILY = {
    family: tool for tool, family in _CORRECTION_CONTEXT_FAMILY.items()
}


def post_correction_confirmation_required(state: Mapping[str, Any]) -> bool:
    """Whether an autonomous correction would hit the confirmation guard.

    This is the canonical, policy-visible boundary shared by the production
    process gate and the recovery-probe generator.  Keep candidate lifecycle in
    the predicate: an open transaction is handled by the earlier lifecycle
    guard and is not evidence that the post-correction boundary was reached.
    """

    has_open_candidate = bool(
        state.get("has_open_candidate")
        or state.get("has_unverified_candidate")
        or state.get("has_verified_candidate")
    )
    if has_open_candidate or not state.get("accepted_corrections"):
        return False
    signatures = {
        str(item) for item in (state.get("unresolved_signatures") or [])
    }
    return signatures == {POST_CORRECTION_CONFIRMATION_SIGNATURE}
_RECOVERY_FAMILY_ORDER = {
    "measurement": ("parameter", "topology"),
    "parameter": ("topology", "measurement"),
    "topology": ("parameter", "measurement"),
}
_MAX_CORRECTION_RECOVERY_ACTIONS = 2


class ProcessValidityOracle:
    """Single authoritative legality and state-reference gate."""

    def __init__(
        self,
        *,
        anomaly_threshold: float = 1.0,
        executor_hydrated_corrections: bool = False,
    ) -> None:
        self.anomaly_threshold = float(anomaly_threshold)
        # Deployment correction executors compute replacement values from the
        # identified target (suspect indices, branch row), matching the
        # canonical correct_*_from_path protocol.  Enabling this accepts
        # target-only correction payloads; the payload stays bounded because
        # the model supplies no physical values at all.
        self.executor_hydrated_corrections = bool(executor_hydrated_corrections)

    def check(
        self,
        state: Any,
        action: Mapping[str, Any] | str,
        *,
        store: Any | None = None,
    ) -> dict[str, Any]:
        normalized = safe_normalize_action(action)
        tool = normalized["tool"]
        args = normalized["arguments"]
        error_code: str | None = None
        error_detail: str | None = None
        active_id = state.get("active_state_id")
        candidate_id = state.get("candidate_state_id")
        has_open_candidate = bool(
            state.get("has_open_candidate")
            or state.get("has_unverified_candidate")
            or state.get("has_verified_candidate")
        )
        has_unverified_candidate = bool(state.get("has_unverified_candidate"))
        has_verified_candidate = bool(state.get("has_verified_candidate"))

        if tool == INVALID_ACTION:
            error_code = str(args.get("error_code") or "schema_error")
            error_detail = str(args.get("error_detail") or "malformed_policy_action")
        elif tool not in MACRO_ACTIONS:
            error_code, error_detail = "unknown_tool", str(tool)
        elif tool in CORRECTION_TOOLS:
            requested = args.get("state_id") or active_id
            confirmation_required = post_correction_confirmation_required(state)
            if has_open_candidate:
                error_code, error_detail = "candidate_lifecycle_violation", "correction_with_open_candidate"
            elif not self._known_current_state(store, requested):
                error_code, error_detail = "unknown_state_id", str(requested)
            elif str(requested) != str(active_id):
                error_code, error_detail = "state_reference_mismatch", "correction_state_not_active"
            elif confirmation_required:
                # This controller marker is created only after an accepted
                # correction reaches observable statistical quiescence.  It
                # requests one same-state investigation followed by operator
                # handoff; provider suggestions remain review evidence, not a
                # license for another autonomous transaction.  Without this
                # process-level guard an off-policy learner can ignore the
                # expert handoff, open a healthy-target candidate, and then
                # make the expert disposition an impossible safety choice.
                family = _CORRECTION_CONTEXT_FAMILY[tool]
                error_code = "post_correction_confirmation_required"
                error_detail = (
                    f"{family}_autonomous_correction_blocked_for_operator_review"
                )
            elif tool == CORRECT_PARAMETERS and not self._context_is_fresh(state, "parameter"):
                error_code, error_detail = "missing_precondition", "parameter_context_missing"
            elif tool == CORRECT_TOPOLOGY and not self._context_is_fresh(state, "topology"):
                error_code, error_detail = "missing_precondition", "topology_context_missing"
            elif (
                tool == CORRECT_MEASUREMENTS
                and (
                    state.get("requires_measurement_context")
                    or self.executor_hydrated_corrections
                    or state.get("require_context_supported_corrections")
                )
                and not self._context_is_fresh(state, "measurement")
            ):
                error_code, error_detail = "missing_precondition", "measurement_context_missing"
            elif not self._has_correction_payload(tool, args):
                error_code, error_detail = "schema_error", "empty_correction_payload"
            elif (
                self.executor_hydrated_corrections
                or state.get("require_context_supported_corrections")
            ) and not state.get("audited_evaluation_setup_correction"):
                # Deployment executors hydrate physical values from a bounded
                # target.  The model-visible action is therefore legal only
                # when the exact target was emitted by a fresh context bound
                # to this active state.  This closes the former gap where an
                # arbitrary branch target could reach the expensive executor
                # merely because some context action had run earlier.
                #
                # The one exception is a private evaluator setup transition
                # used to construct the frozen partial-success starting state.
                # TransactionalPSSEEnv creates that flag internally for one
                # measurement update; it is not an action argument or policy
                # observation and therefore cannot weaken model execution.
                contract_failure = self._context_supported_correction_failure(
                    state, normalized
                )
                if contract_failure is not None:
                    error_code, error_detail = contract_failure
        elif tool in CONTEXT_TOOLS:
            requested = args.get("state_id") or active_id
            if has_open_candidate:
                error_code, error_detail = "candidate_lifecycle_violation", "context_request_with_open_candidate"
            elif not self._known_current_state(store, requested):
                error_code, error_detail = "unknown_state_id", str(requested)
            elif str(requested) != str(active_id):
                error_code, error_detail = "state_reference_mismatch", "context_state_not_active"
        elif tool in {RUN_WLS, VERIFY_CANDIDATE}:
            expected = candidate_id if has_open_candidate else active_id
            requested = args.get("state_id") or expected
            if tool == VERIFY_CANDIDATE and not has_open_candidate:
                error_code, error_detail = "candidate_lifecycle_violation", "verify_without_candidate"
            elif has_verified_candidate:
                error_code, error_detail = "candidate_lifecycle_violation", "candidate_already_verified"
            elif not self._known_current_state(store, requested):
                error_code, error_detail = "unknown_state_id", str(requested)
            elif str(requested) != str(expected):
                error_code, error_detail = "state_reference_mismatch", "wls_state_not_current_target"
        elif tool == COMMIT_STATE:
            requested = args.get("candidate_state_id") or candidate_id
            if not has_verified_candidate:
                error_code, error_detail = "candidate_lifecycle_violation", "commit_without_verified_candidate"
            elif str(requested) != str(candidate_id):
                error_code, error_detail = "state_reference_mismatch", "commit_candidate_not_current"
            elif not self._known_current_state(store, requested):
                error_code, error_detail = "unknown_state_id", str(requested)
            elif state.get("candidate_disposition") not in {"ACCEPT_PARTIAL", "ACCEPT_FINAL"}:
                error_code, error_detail = "candidate_lifecycle_violation", "commit_rejected_or_inconclusive_candidate"
        elif tool == ROLLBACK_STATE:
            requested = args.get("candidate_state_id") or candidate_id
            if not has_verified_candidate:
                error_code, error_detail = "candidate_lifecycle_violation", "rollback_without_verified_candidate"
            elif str(requested) != str(candidate_id):
                error_code, error_detail = "state_reference_mismatch", "rollback_candidate_not_current"
            elif not self._known_current_state(store, requested):
                error_code, error_detail = "unknown_state_id", str(requested)
            elif state.get("candidate_disposition") not in {"REJECT", "INCONCLUSIVE"}:
                error_code, error_detail = "candidate_lifecycle_violation", "rollback_accepted_candidate"
        elif tool == FINALIZE_DIAGNOSIS:
            if has_open_candidate:
                error_code, error_detail = "candidate_lifecycle_violation", "finalize_with_open_candidate"
            elif not self._terminal_condition_met(state):
                error_code, error_detail = "terminal_condition_not_met", "unresolved_or_unverified_anomaly"
        elif tool in {ASK_FOR_MORE_EVIDENCE, RUN_ALTERNATIVE_TEST} or tool in DIAGNOSTIC_TOOLS:
            # Specialized diagnostics are read-only evidence actions: legal on
            # the active state during investigation, and on an open candidate
            # only when its verification came back inconclusive.
            expected = candidate_id if has_open_candidate else active_id
            requested = args.get("state_id") or expected
            if has_open_candidate and state.get("candidate_disposition") != "INCONCLUSIVE":
                error_code, error_detail = "candidate_lifecycle_violation", "evidence_action_requires_inconclusive_candidate"
            elif not self._known_current_state(store, requested):
                error_code, error_detail = "unknown_state_id", str(requested)
            elif str(requested) != str(expected):
                error_code, error_detail = "state_reference_mismatch", "evidence_state_not_current_target"

        process_valid = error_code is None
        repairs = [] if process_valid else self.repair_actions(state, error_code, error_detail)
        return {
            "process_valid": process_valid,
            "reason": error_detail,
            "error_code": error_code,
            "error_detail": error_detail,
            "valid_next_actions": repairs,
        }

    @staticmethod
    def _known_current_state(store: Any | None, state_id: Any) -> bool:
        if state_id is None:
            return False
        return True if store is None else bool(store.exists(str(state_id)))

    def _has_correction_payload(self, tool: str, args: Mapping[str, Any]) -> bool:
        # ``safe_normalize_action`` has already flattened the optional legacy
        # ``modification`` wrapper.  Keeping this check tool-specific prevents
        # an unrestricted whole-vector replacement from masquerading as the
        # bounded measurement-correction macro.
        payload = dict(args)
        payload.pop("state_id", None)
        payload.pop("candidate_state_id", None)
        if tool == CORRECT_MEASUREMENTS:
            if "measurements" in payload:
                return False
            updates = payload.get("measurement_updates")
            if bool(updates) and isinstance(updates, Mapping):
                return True
            if self.executor_hydrated_corrections:
                group = payload.get("suspect_group")
                return (
                    isinstance(group, (list, tuple))
                    and bool(group)
                    and all(
                        isinstance(index, int) and not isinstance(index, bool)
                        for index in group
                    )
                )
            return False
        if tool == CORRECT_PARAMETERS:
            has_target = any(
                payload.get(key) is not None
                for key in ("line_index", "line_index1", "branch_row0", "branch_id", "cb_name")
            )
            has_value = any(
                payload.get(key) is not None
                for key in ("value", "corrected_value", "new_value", "multiplier")
            )
            if self.executor_hydrated_corrections and has_target:
                return True
            return (has_target and has_value) or any(
                key in payload for key in ("case", "case_updates")
            )
        if tool == CORRECT_TOPOLOGY:
            has_target = any(
                payload.get(key) is not None
                for key in ("branch_id", "cb_name", "line_index", "line_index1", "branch_row0")
            )
            return (has_target and any(
                payload.get(key) is not None for key in ("status", "expected_status")
            )) or any(key in payload for key in ("case", "case_updates"))
        return False

    @staticmethod
    def _context_is_fresh(state: Any, context_family: str) -> bool:
        if not state.get(f"has_fresh_{context_family}_context"):
            return False
        context_state_id = state.get(f"{context_family}_context_state_id")
        return context_state_id is not None and str(context_state_id) == str(state.get("active_state_id"))

    def _terminal_condition_met(self, state: Any) -> bool:
        signatures = terminal_explanation_signatures(
            state.get("unresolved_signatures") or []
        )
        anomalies_explained = bool(signatures) and not unexplained_signatures(
            signatures, state.get("explained_anomalies")
        )
        synthetic_terminal_eligible = bool(
            state.get(SYNTHETIC_TERMINAL_COMPATIBILITY_KEY)
            and state.get("oracle_terminal_eligible")
        )
        if synthetic_terminal_eligible:
            # Legacy synthetic fixtures may close from private truth, but the
            # capability bit is never part of a PolicyObservation and is never
            # injected by a production environment.
            return True
        if state.get("accepted_corrections"):
            # Candidate WLS evidence is sufficient to accept a transaction,
            # not to certify the whole corrected state as release-final.
            # A later diagnostic explanation closes only its own signature;
            # it is not an independent certificate for prior corrections.
            return False
        if state.get("no_material_anomaly_remaining"):
            return True
        remaining = state.get("remaining_anomaly_score")
        if remaining is not None and float(remaining) < self.anomaly_threshold:
            return True
        # A residual anomaly is diagnosed, not unresolved, once every
        # observable signature is accounted for by a recorded diagnostic
        # explanation (localized harmonic source, estimated HIF).
        return anomalies_explained

    def _context_supported_correction_failure(
        self,
        state: Any,
        action: Mapping[str, Any],
    ) -> tuple[str, str] | None:
        """Validate the exact same-state correction inventory contract."""

        normalized = safe_normalize_action(action)
        tool = normalized["tool"]
        family = _CORRECTION_CONTEXT_FAMILY[tool]
        active_id = state.get("active_state_id")
        if not self._context_is_fresh(state, family):
            return "missing_precondition", f"{family}_context_missing"

        fresh = state.get("fresh_context_evidence")
        evidence = fresh.get(family) if isinstance(fresh, Mapping) else None
        if (
            not isinstance(evidence, Mapping)
            or active_id is None
            or str(evidence.get("state_id") or "") != str(active_id)
        ):
            return "missing_precondition", f"{family}_context_missing"

        if family in {"parameter", "topology"}:
            # Both a complete negative screen and unavailable/inconclusive
            # evidence prohibit a branch correction in the current control
            # epoch.  Re-running arbitrary targets cannot change that fact.
            if evidence.get("route_status") != "actionable":
                route_status = str(
                    evidence.get("route_status") or "missing_route_status"
                )
                route_reason = str(
                    evidence.get("route_status_reason")
                    or "missing_route_status_reason"
                )
                return (
                    "correction_route_not_actionable",
                    (
                        f"{family}_route_not_actionable:"
                        f"{route_status}:{route_reason}"
                    ),
                )

        supported = evidence.get("supported_corrections")
        if not isinstance(supported, (list, tuple)):
            return "missing_precondition", f"{family}_context_missing"

        requested_arguments = dict(normalized["arguments"])
        requested_arguments.setdefault("state_id", active_id)
        requested_signature = action_signature(
            {"tool": tool, "arguments": requested_arguments}
        )
        supported_signatures: set[str] = set()
        for raw_action in supported:
            if not isinstance(raw_action, Mapping):
                return "missing_precondition", f"{family}_context_missing"
            supported_action = safe_normalize_action(raw_action)
            if supported_action["tool"] != tool:
                return "missing_precondition", f"{family}_context_missing"
            supported_state_id = supported_action["arguments"].get("state_id")
            if (
                supported_state_id is None
                or str(supported_state_id) != str(active_id)
            ):
                return "missing_precondition", f"{family}_context_missing"
            supported_signatures.add(action_signature(supported_action))

        if requested_signature not in supported_signatures:
            return (
                "correction_not_supported_by_current_context",
                f"{family}_correction_not_supported_by_context",
            )
        return None

    def _supported_corrections_for_family(
        self,
        state: Any,
        family: str,
    ) -> list[dict[str, Any]]:
        """Return only well-formed same-state actionable corrections."""

        if not self._context_is_fresh(state, family):
            return []
        active_id = state.get("active_state_id")
        fresh = state.get("fresh_context_evidence")
        evidence = fresh.get(family) if isinstance(fresh, Mapping) else None
        if (
            not isinstance(evidence, Mapping)
            or active_id is None
            or str(evidence.get("state_id") or "") != str(active_id)
        ):
            return []
        if (
            family in {"parameter", "topology"}
            and evidence.get("route_status") != "actionable"
        ):
            return []
        raw_supported = evidence.get("supported_corrections")
        if not isinstance(raw_supported, (list, tuple)):
            return []

        expected_tool = _CORRECTION_TOOL_FOR_FAMILY[family]
        supported: list[dict[str, Any]] = []
        seen: set[str] = set()
        for raw_action in raw_supported:
            if not isinstance(raw_action, Mapping):
                return []
            normalized = safe_normalize_action(raw_action)
            if normalized["tool"] != expected_tool:
                return []
            requested = normalized["arguments"].get("state_id")
            if requested is None or str(requested) != str(active_id):
                return []
            if not self._has_correction_payload(
                expected_tool, normalized["arguments"]
            ):
                return []
            signature = action_signature(normalized)
            if signature not in seen:
                supported.append(normalized)
                seen.add(signature)
        return supported

    def _bounded_correction_recovery_actions(
        self,
        state: Any,
        *,
        failed_family: str,
        allow_same_family: bool,
    ) -> list[dict[str, Any]]:
        """Offer at most two observable, non-escalating recovery actions."""

        if state.get("has_open_candidate") or state.get(
            "has_unverified_candidate"
        ) or state.get("has_verified_candidate"):
            return self._safe_actions_for_state(state)
        active_id = state.get("active_state_id")
        if active_id is None:
            return []

        actions: list[dict[str, Any]] = []
        if allow_same_family:
            actions.extend(
                self._supported_corrections_for_family(state, failed_family)
            )

        for family in _RECOVERY_FAMILY_ORDER[failed_family]:
            if len(actions) >= _MAX_CORRECTION_RECOVERY_ACTIONS:
                break
            alternatives = self._supported_corrections_for_family(state, family)
            if alternatives:
                actions.append(alternatives[0])
            elif not self._context_is_fresh(state, family):
                actions.append(
                    {
                        "tool": _CONTEXT_TOOL_FOR_FAMILY[family],
                        "arguments": {"state_id": active_id},
                    }
                )

        if not actions:
            actions = [{"tool": RUN_WLS, "arguments": {"state_id": active_id}}]
        return actions[:_MAX_CORRECTION_RECOVERY_ACTIONS]

    def repair_actions(
        self,
        state: Any,
        error_code: str | None,
        error_detail: str | None = None,
    ) -> list[dict[str, Any]]:
        active_id = state.get("active_state_id")
        candidate_id = state.get("candidate_state_id")
        if error_code in {"json_parse_error", "argument_decode_error", "schema_error", "policy_exception"}:
            return self._safe_actions_for_state(state)
        if error_code == "missing_precondition":
            tool = {
                "parameter_context_missing": GET_PARAMETER_CONTEXT,
                "topology_context_missing": GET_TOPOLOGY_CONTEXT,
                "measurement_context_missing": GET_MEASUREMENT_CONTEXT,
            }.get(error_detail, GET_PARAMETER_CONTEXT)
            return [{"tool": tool, "arguments": {"state_id": active_id}}]
        if error_code == "post_correction_confirmation_required":
            # The controller marker names the canonical measurement-context
            # confirmation protocol.  Its repair must not vary with whichever
            # off-policy correction family the learner happened to attempt.
            if not self._context_is_fresh(state, "measurement"):
                return [
                    {
                        "tool": GET_MEASUREMENT_CONTEXT,
                        "arguments": {"state_id": active_id},
                    }
                ]
            return [
                {
                    "tool": ASK_FOR_MORE_EVIDENCE,
                    "arguments": {
                        "state_id": active_id,
                        "request": RECOVERY_OPTIONS_EXHAUSTED_REQUEST,
                    },
                }
            ]
        if error_code in {
            "correction_not_supported_by_current_context",
            "correction_route_not_actionable",
            "parameter_scans_missing",
            "topology_correction_unsupported",
        }:
            if error_code == "parameter_scans_missing":
                failed_family = "parameter"
            elif error_code == "topology_correction_unsupported":
                failed_family = "topology"
            else:
                failed_family = next(
                    (
                        family
                        for family in _CONTEXT_TOOL_FOR_FAMILY
                        if str(error_detail or "").startswith(f"{family}_")
                    ),
                    "parameter",
                )
            return self._bounded_correction_recovery_actions(
                state,
                failed_family=failed_family,
                allow_same_family=(
                    error_code == "correction_not_supported_by_current_context"
                ),
            )
        if error_code in {"state_reference_mismatch", "unknown_state_id"}:
            return self._safe_actions_for_state(state)
        if error_code == "candidate_lifecycle_violation":
            if state.get("has_unverified_candidate") and candidate_id:
                return [{"tool": RUN_WLS, "arguments": {"state_id": candidate_id}}]
            disposition = state.get("candidate_disposition")
            if disposition in {"ACCEPT_PARTIAL", "ACCEPT_FINAL"} and candidate_id:
                return [{"tool": COMMIT_STATE, "arguments": {"candidate_state_id": candidate_id}}]
            if disposition in {"REJECT", "INCONCLUSIVE"} and candidate_id:
                return [{"tool": ROLLBACK_STATE, "arguments": {"candidate_state_id": candidate_id}}]
        if error_code == "terminal_condition_not_met" and active_id:
            return [{"tool": RUN_WLS, "arguments": {"state_id": active_id}}]
        return []

    @staticmethod
    def _safe_actions_for_state(state: Any) -> list[dict[str, Any]]:
        active_id = state.get("active_state_id")
        candidate_id = state.get("candidate_state_id")
        if state.get("has_unverified_candidate") and candidate_id:
            return [{"tool": RUN_WLS, "arguments": {"state_id": candidate_id}}]
        if state.get("has_verified_candidate") and candidate_id:
            if state.get("candidate_disposition") in {"ACCEPT_PARTIAL", "ACCEPT_FINAL"}:
                return [{"tool": COMMIT_STATE, "arguments": {"candidate_state_id": candidate_id}}]
            return [{"tool": ROLLBACK_STATE, "arguments": {"candidate_state_id": candidate_id}}]
        return [{"tool": RUN_WLS, "arguments": {"state_id": active_id}}] if active_id else []
