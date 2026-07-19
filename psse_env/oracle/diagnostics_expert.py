"""Routing expert for the specialized diagnostic tools.

Routes harmonic and high-impedance-fault investigation from observable
signals only: which telemetry channels exist on the active state
(``available_evidence``), which anomaly signatures are unresolved, and what
earlier diagnostics in this episode already produced.  Privileged fault
flags and oracle hints may raise a proposal's rank, matching the contract of
the other family experts, but they never create a route that observable
evidence cannot justify.

The intended escalation ladders are:

- harmonic: ``get_harmonic_context`` -> ``run_hse_from_path``;
- HIF: ``run_three_phase_nlm_from_path`` (line-level localization) ->
  ``estimate_hif_location_magnitude_multiscan_from_path`` when a persistent
  scan window exists, else ``estimate_hif_location_magnitude_from_path``.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from psse_env.actions import (
    ANOMALY_FAMILY_MARKERS,
    DIAGNOSTIC_TOOLS,
    ESTIMATE_HIF_FROM_PATH,
    ESTIMATE_HIF_MULTISCAN_FROM_PATH,
    GET_HARMONIC_CONTEXT,
    RUN_HSE_FROM_PATH,
    RUN_THREE_PHASE_NLM_FROM_PATH,
    safe_normalize_action,
)
from psse_env.oracle.expert_types import (
    ExpertActionProposal,
    matching_evidence_codes,
    normalized_hint_actions,
    policy_state_view,
    state_value,
)


# One shared vocabulary with the environment's explained-anomaly recording,
# so a signature that routes to a diagnostic is the same signature that the
# diagnostic's explanation later accounts for.
HARMONIC_MARKERS = ANOMALY_FAMILY_MARKERS["harmonic"]
HIF_MARKERS = ANOMALY_FAMILY_MARKERS["hif"]


class DiagnosticsExpert:
    """Propose harmonic and HIF diagnostic actions from observable evidence."""

    source_expert = "diagnostics_expert"

    def propose(
        self,
        state: Any,
        history: Sequence[Mapping[str, Any]] | None = None,
        *,
        oracle_hints: Sequence[Mapping[str, Any]] | None = None,
        harmonic_fault_present: bool = False,
        hif_fault_present: bool = False,
    ) -> list[ExpertActionProposal]:
        if oracle_hints is None:
            oracle_hints = state_value(state, "oracle_action_hints", []) or []
        state = policy_state_view(state)
        active_id = state_value(state, "active_state_id")
        if not active_id:
            return []
        available = {str(item) for item in state_value(state, "available_evidence", []) or []}
        unresolved = state_value(state, "unresolved_signatures", [])
        harmonic_codes = matching_evidence_codes(unresolved, *HARMONIC_MARKERS)
        hif_codes = matching_evidence_codes(unresolved, *HIF_MARKERS)
        completed = self._completed_diagnostics(history or [])
        proposals: list[ExpertActionProposal] = []

        for action in normalized_hint_actions(
            list(oracle_hints or ()),
            allowed_tools=set(DIAGNOSTIC_TOOLS),
            active_state_id=active_id,
        ):
            proposals.append(
                ExpertActionProposal(
                    action=action,
                    source_expert=self.source_expert,
                    confidence=0.97,
                    evidence_codes=["oracle_action_hint", "diagnostic_family"],
                    admissible=True,
                    estimated_immediate_risk=0.02,
                )
            )

        harmonic_signal = bool(harmonic_codes) or harmonic_fault_present
        if harmonic_signal and "harmonic_measurements" in available:
            if GET_HARMONIC_CONTEXT not in completed:
                proposals.append(
                    self._proposal(
                        GET_HARMONIC_CONTEXT,
                        {"state_id": active_id},
                        confidence=0.86,
                        evidence=[
                            "harmonic_telemetry_available",
                            *(harmonic_codes or ["privileged_harmonic_ranking"]),
                        ],
                    )
                )
            elif RUN_HSE_FROM_PATH not in completed:
                proposals.append(
                    self._proposal(
                        RUN_HSE_FROM_PATH,
                        {"state_id": active_id},
                        confidence=0.90,
                        evidence=["harmonic_context_acquired", "hse_localization_pending"],
                    )
                )

        hif_signal = bool(hif_codes) or hif_fault_present
        hif_branch = self._nlm_top_branch(completed.get(RUN_THREE_PHASE_NLM_FROM_PATH))
        if hif_signal and "nlm_diagnostic" in available and (
            RUN_THREE_PHASE_NLM_FROM_PATH not in completed
        ):
            proposals.append(
                self._proposal(
                    RUN_THREE_PHASE_NLM_FROM_PATH,
                    {"state_id": active_id},
                    confidence=0.85,
                    evidence=[
                        "nlm_telemetry_available",
                        *(hif_codes or ["privileged_hif_ranking"]),
                    ],
                )
            )
        if hif_branch is not None:
            follow_up_arguments = {
                "state_id": active_id,
                "candidate_branch_row0": int(hif_branch),
            }
            if (
                "hif_scan_window" in available
                and ESTIMATE_HIF_MULTISCAN_FROM_PATH not in completed
            ):
                proposals.append(
                    self._proposal(
                        ESTIMATE_HIF_MULTISCAN_FROM_PATH,
                        follow_up_arguments,
                        confidence=0.91,
                        evidence=[
                            "nlm_branch_localized",
                            "persistent_scan_window_available",
                        ],
                    )
                )
            elif ESTIMATE_HIF_FROM_PATH not in completed:
                proposals.append(
                    self._proposal(
                        ESTIMATE_HIF_FROM_PATH,
                        follow_up_arguments,
                        confidence=0.89,
                        evidence=["nlm_branch_localized", "single_scan_estimation"],
                    )
                )
        return proposals

    def _proposal(
        self,
        tool: str,
        arguments: Mapping[str, Any],
        *,
        confidence: float,
        evidence: list[str],
    ) -> ExpertActionProposal:
        return ExpertActionProposal(
            action={"tool": tool, "arguments": dict(arguments)},
            source_expert=self.source_expert,
            confidence=confidence,
            evidence_codes=evidence,
            admissible=True,
            estimated_immediate_risk=0.01,
        )

    @staticmethod
    def _completed_diagnostics(
        history: Sequence[Mapping[str, Any]],
    ) -> dict[str, Mapping[str, Any]]:
        """Map successfully executed diagnostic tools to their latest metrics.

        Accepts both raw collector transitions (``action``/``tool_output``)
        and the summarized model history window (``tool``/``outcome``/
        ``observable_metrics``).
        """
        completed: dict[str, Mapping[str, Any]] = {}
        for item in history:
            if not isinstance(item, Mapping):
                continue
            action = item.get("action") or item.get("executed_action") or item
            try:
                tool = safe_normalize_action(action)["tool"]
            except Exception:
                continue
            if tool not in DIAGNOSTIC_TOOLS:
                continue
            output = item.get("tool_output")
            outcome = item.get("outcome")
            if isinstance(output, Mapping):
                status = output.get("execution_status")
                metrics = output.get("tool_metrics")
            elif isinstance(outcome, Mapping):
                status = outcome.get("execution_status")
                metrics = item.get("observable_metrics")
            else:
                status, metrics = None, None
            if status == "success":
                completed[tool] = metrics if isinstance(metrics, Mapping) else {}
        return completed

    @staticmethod
    def _nlm_top_branch(metrics: Mapping[str, Any] | None) -> int | None:
        if not isinstance(metrics, Mapping):
            return None
        summary = metrics.get("nlm_summary")
        if not isinstance(summary, Mapping):
            return None
        groups = summary.get("top_hif_groups")
        if not isinstance(groups, Sequence):
            return None
        for group in groups:
            if isinstance(group, Mapping) and group.get("branch_row0") is not None:
                try:
                    return int(group["branch_row0"])
                except (TypeError, ValueError):
                    continue
        return None


__all__ = ["DiagnosticsExpert", "HARMONIC_MARKERS", "HIF_MARKERS"]
