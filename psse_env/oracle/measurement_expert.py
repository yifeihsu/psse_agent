from __future__ import annotations

from typing import Any, Mapping, Sequence

from psse_env.actions import CORRECT_MEASUREMENTS, GET_MEASUREMENT_CONTEXT, RUN_WLS
from psse_env.oracle.expert_types import (
    ExpertActionProposal,
    matching_evidence_codes,
    normalized_hint_actions,
    policy_state_view,
    state_value,
)


class MeasurementExpert:
    """Propose measurement-family diagnosis and correction actions."""

    source_expert = "measurement_expert"
    _TOOLS = {GET_MEASUREMENT_CONTEXT, CORRECT_MEASUREMENTS}

    def propose(
        self,
        state: Any,
        history: Sequence[Mapping[str, Any]] | None = None,
        *,
        oracle_hints: Sequence[Mapping[str, Any]] | None = None,
        oracle_fault_present: bool = False,
    ) -> list[ExpertActionProposal]:
        del history  # The compact policy observation already carries the needed belief state.
        if oracle_hints is None:
            oracle_hints = state_value(state, "oracle_action_hints", []) or []
        oracle_fault_present = bool(
            oracle_fault_present or state_value(state, "true_measurement_errors", [])
        )
        state = policy_state_view(state)
        active_id = state_value(state, "active_state_id")
        requires_context = bool(state_value(state, "requires_measurement_context", False))
        context_state_id = state_value(state, "measurement_context_state_id")
        has_context = bool(
            state_value(state, "has_fresh_measurement_context", False)
            and context_state_id is not None
            and str(context_state_id) == str(active_id)
        )
        proposals: list[ExpertActionProposal] = []

        for action in normalized_hint_actions(
            list(oracle_hints or ()),
            allowed_tools={CORRECT_MEASUREMENTS},
            active_state_id=active_id,
        ):
            proposals.append(
                ExpertActionProposal(
                    action=action,
                    source_expert=self.source_expert,
                    confidence=0.98,
                    evidence_codes=["oracle_action_hint", "measurement_family"],
                    admissible=(
                        action["tool"] != CORRECT_MEASUREMENTS
                        or not requires_context
                        or has_context
                    ),
                    estimated_immediate_risk=0.12 if action["tool"] == CORRECT_MEASUREMENTS else 0.02,
                )
            )

        unresolved = state_value(state, "unresolved_signatures", [])
        measurement_codes = matching_evidence_codes(
            unresolved,
            "measurement",
            "bad_data",
            "large_residual",
            "meter",
            "residual_outlier",
        )
        measurement_signal = bool(measurement_codes)
        # A global safety requirement for measurement context is not itself
        # evidence of a measurement fault.  Context routing must be supported
        # by an observable signature (or private teacher supervision that the
        # production collector independently audits against that signature).
        if measurement_signal and not has_context and active_id:
            evidence = ["measurement_context_missing"]
            evidence.extend(["measurement_anomaly_evidence", *measurement_codes])
            proposals.append(
                ExpertActionProposal(
                    action={"tool": GET_MEASUREMENT_CONTEXT, "arguments": {"state_id": active_id}},
                    source_expert=self.source_expert,
                    confidence=0.87,
                    evidence_codes=evidence,
                    admissible=True,
                    estimated_immediate_risk=0.01,
                )
            )

        # WLS is the safe observable baseline when no domain has yet produced
        # enough evidence for a correction.  Its modest score lets a concrete
        # parameter/topology proposal outrank it.
        if not proposals and active_id:
            proposals.append(
                ExpertActionProposal(
                    action={"tool": RUN_WLS, "arguments": {"state_id": active_id}},
                    source_expert=self.source_expert,
                    confidence=0.50,
                    evidence_codes=["observable_baseline_diagnostic"],
                    admissible=True,
                    estimated_immediate_risk=0.01,
                )
            )
        return proposals
