from __future__ import annotations

from typing import Any, Mapping, Sequence

from psse_env.actions import CORRECT_MEASUREMENTS, CORRECT_PARAMETERS, GET_PARAMETER_CONTEXT
from psse_env.oracle.expert_types import (
    ExpertActionProposal,
    dominance_confidence,
    history_action_tool,
    matching_evidence_codes,
    normalized_hint_actions,
    policy_state_view,
    recovery_record_applies_to_state,
    state_value,
)


class ParameterExpert:
    """Propose branch-parameter diagnosis after observable or historical evidence."""

    source_expert = "parameter_expert"
    _TOOLS = {GET_PARAMETER_CONTEXT, CORRECT_PARAMETERS}

    def propose(
        self,
        state: Any,
        history: Sequence[Mapping[str, Any]] | None = None,
        *,
        oracle_hints: Sequence[Mapping[str, Any]] | None = None,
        oracle_fault_present: bool = False,
    ) -> list[ExpertActionProposal]:
        del history
        if oracle_hints is None:
            oracle_hints = state_value(state, "oracle_action_hints", []) or []
        oracle_fault_present = bool(
            oracle_fault_present or state_value(state, "true_parameter_errors", [])
        )
        state = policy_state_view(state)
        active_id = state_value(state, "active_state_id")
        context_state_id = state_value(state, "parameter_context_state_id")
        has_context = bool(
            state_value(state, "has_fresh_parameter_context", False)
            and context_state_id is not None
            and str(context_state_id) == str(active_id)
        )
        rejected_measurement = self._rejected_measurement(state)
        partial_measurement = self._accepted_partial_measurement(state)
        priority_evidence: list[str] = []
        if rejected_measurement:
            priority_evidence.append("measurement_correction_rejected")
        if partial_measurement:
            priority_evidence.append("measurement_correction_accepted_partial")

        unresolved = state_value(state, "unresolved_signatures", [])
        parameter_codes = matching_evidence_codes(
            unresolved,
            "parameter",
            "impedance",
            "reactance",
            "resistance",
            "admittance",
            "multiplier",
        )
        measurement_dominant = bool(
            matching_evidence_codes(unresolved, "wls_residual_outlier_dominant")
        )
        branch_dominant = bool(
            matching_evidence_codes(
                unresolved, "wls_branch_multiplier_dominant"
            )
        )
        explicit_parameter_codes = [
            code
            for code in parameter_codes
            if not str(code).startswith("wls_branch_multiplier")
        ]
        # Dominance routing is deliberate: a dominant measurement outlier
        # contaminates the branch-multiplier scans just as a structural error
        # contaminates residuals, so the parameter path stays suppressed until
        # branch evidence is dominant or explicit.  Forcing parameter-first
        # here was measured to chase verification-rejected phantom lines until
        # the recovery budget exhausted (4 of 19 mixed episodes regressed).
        suppress_parameter_hints = bool(
            measurement_dominant
            and not branch_dominant
            and not explicit_parameter_codes
            and not partial_measurement
        )
        proposals: list[ExpertActionProposal] = []
        hint_actions = (
            []
            if suppress_parameter_hints
            else normalized_hint_actions(
                list(oracle_hints or ()),
                allowed_tools={CORRECT_PARAMETERS},
                active_state_id=active_id,
            )
        )
        for action in hint_actions:
            proposals.append(
                ExpertActionProposal(
                    action=action,
                    source_expert=self.source_expert,
                    confidence=0.995 if priority_evidence else 0.98,
                    evidence_codes=[
                        "oracle_action_hint",
                        "parameter_family",
                        *priority_evidence,
                    ],
                    admissible=action["tool"] != CORRECT_PARAMETERS or has_context,
                    estimated_immediate_risk=(
                        0.14 if action["tool"] == CORRECT_PARAMETERS else 0.02
                    ),
                )
            )
        # A rejected or partial measurement correction is useful priority
        # evidence only when the *current* active state still carries
        # observable branch-family evidence.  It must not manufacture a
        # parameter hypothesis after a commit cleared those signatures, and a
        # clearly measurement-dominant solve must not be reinterpreted as a
        # branch fault merely because one bounded measurement candidate was
        # rejected.
        parameter_signal = bool(parameter_codes) and not suppress_parameter_hints
        diagnosis_needed = parameter_signal
        if diagnosis_needed and not has_context and active_id:
            # Once measurement work has failed or only partially succeeded,
            # parameter diagnosis must outrank even another measurement hint.
            confidence = (
                0.999
                if priority_evidence
                else dominance_confidence(0.87, parameter_codes)
            )
            evidence = ["parameter_context_missing", *priority_evidence]
            if parameter_signal:
                evidence.extend(["parameter_anomaly_evidence", *parameter_codes])
            proposals.append(
                ExpertActionProposal(
                    action={"tool": GET_PARAMETER_CONTEXT, "arguments": {"state_id": active_id}},
                    source_expert=self.source_expert,
                    confidence=confidence,
                    evidence_codes=evidence,
                    admissible=True,
                    estimated_immediate_risk=0.01,
                )
            )
        return proposals

    @staticmethod
    def _rejected_measurement(state: Any) -> bool:
        active_id = state_value(state, "active_state_id")
        for item in state_value(state, "rejected_hypotheses", []) or []:
            if not recovery_record_applies_to_state(item, active_id):
                continue
            if history_action_tool(item) == CORRECT_MEASUREMENTS:
                return True
            if isinstance(item, Mapping):
                family = item.get("family") or item.get("action_family") or item.get("error_family")
                if str(family).lower() == "measurement":
                    return True
                if "correct_measurements:" in str(item.get("action_signature") or ""):
                    return True
        return False

    @staticmethod
    def _accepted_partial_measurement(state: Any) -> bool:
        for item in state_value(state, "accepted_corrections", []) or []:
            if history_action_tool(item) != CORRECT_MEASUREMENTS:
                if not isinstance(item, Mapping) or str(
                    item.get("family") or item.get("action_family") or ""
                ).lower() != "measurement":
                    continue
            # A committed correction is observable.  If the episode has not
            # terminated, it is operationally a partial fix; the privileged
            # candidate disposition must not be persisted in policy memory.
            if not state_value(state, "no_material_anomaly_remaining", False):
                return True
        return False
