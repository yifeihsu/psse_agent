from __future__ import annotations

from typing import Any, Mapping, Sequence

from psse_env.actions import CORRECT_MEASUREMENTS, CORRECT_TOPOLOGY, GET_TOPOLOGY_CONTEXT
from psse_env.oracle.expert_types import (
    ExpertActionProposal,
    dominance_confidence,
    history_action_tool,
    matching_evidence_codes,
    normalized_hint_actions,
    policy_state_view,
    state_value,
)


class TopologyExpert:
    """Propose topology-context and topology-correction actions."""

    source_expert = "topology_expert"
    _TOOLS = {GET_TOPOLOGY_CONTEXT, CORRECT_TOPOLOGY}

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
            oracle_fault_present or state_value(state, "true_topology_errors", [])
        )
        state = policy_state_view(state)
        active_id = state_value(state, "active_state_id")
        context_state_id = state_value(state, "topology_context_state_id")
        has_context = bool(
            state_value(state, "has_fresh_topology_context", False)
            and context_state_id is not None
            and str(context_state_id) == str(active_id)
        )
        proposals: list[ExpertActionProposal] = []
        hint_actions = normalized_hint_actions(
            list(oracle_hints or ()),
            allowed_tools={CORRECT_TOPOLOGY},
            active_state_id=active_id,
        )
        for action in hint_actions:
            proposals.append(
                ExpertActionProposal(
                    action=action,
                    source_expert=self.source_expert,
                    confidence=0.98,
                    evidence_codes=["oracle_action_hint", "topology_family"],
                    admissible=action["tool"] != CORRECT_TOPOLOGY or has_context,
                    estimated_immediate_risk=0.18 if action["tool"] == CORRECT_TOPOLOGY else 0.02,
                )
            )

        unresolved = state_value(state, "unresolved_signatures", [])
        topology_codes = matching_evidence_codes(
            unresolved,
            "topology",
            "breaker",
            "switch",
            "line_status",
            "connectivity",
            "islanding",
        )
        measurement_dominant = bool(
            matching_evidence_codes(unresolved, "wls_residual_outlier_dominant")
        )
        branch_dominant = bool(
            matching_evidence_codes(
                unresolved, "wls_branch_multiplier_dominant"
            )
        )
        explicit_topology_codes = [
            code
            for code in topology_codes
            if not str(code).startswith("wls_branch_multiplier")
        ]
        partial_measurement = self._accepted_partial_measurement(state)
        # The WLS solve emits a few bounded cross-family findings even for a
        # clear gross-measurement case.  A non-dominant branch multiplier is
        # useful ambiguity evidence only when measurement evidence is not
        # currently dominant; otherwise topology flips would turn measurement
        # recovery into a long sequence of safely rejected but pointless
        # candidates.  Explicit topology/connectivity sensor signatures are
        # unaffected because they do not coexist with this WLS-only dominance
        # relation.
        topology_signal = bool(topology_codes) and not (
            measurement_dominant
            and not branch_dominant
            and not explicit_topology_codes
            and not partial_measurement
        )
        if topology_signal and not has_context and active_id:
            evidence = ["topology_context_missing"]
            evidence.extend(["topology_anomaly_evidence", *topology_codes])
            proposals.append(
                ExpertActionProposal(
                    action={"tool": GET_TOPOLOGY_CONTEXT, "arguments": {"state_id": active_id}},
                    source_expert=self.source_expert,
                    confidence=dominance_confidence(0.87, topology_codes),
                    evidence_codes=evidence,
                    admissible=True,
                    estimated_immediate_risk=0.01,
                )
            )
        return proposals

    @staticmethod
    def _accepted_partial_measurement(state: Any) -> bool:
        for item in state_value(state, "accepted_corrections", []) or []:
            if history_action_tool(item) != CORRECT_MEASUREMENTS:
                if not isinstance(item, Mapping) or str(
                    item.get("family") or item.get("action_family") or ""
                ).lower() != "measurement":
                    continue
            if not state_value(state, "no_material_anomaly_remaining", False):
                return True
        return False
