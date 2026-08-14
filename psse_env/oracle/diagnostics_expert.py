"""Routing expert for the specialized diagnostic tools.

Routes harmonic, three-phase-unbalance, and high-impedance-fault
investigation from observable signals only: which telemetry channels exist on the active state
(``available_evidence``), which anomaly signatures are unresolved, and what
earlier diagnostics in this episode already produced.  Privileged fault
flags and oracle hints are deliberately ignored here: holding the policy
observation fixed must hold the production target action fixed too.

The intended escalation ladders are:

- harmonic: ``get_harmonic_context`` -> ``run_hse_from_path``;
- three-phase unbalance: ``run_three_phase_nlm_from_path`` -> an observable
  non-HIF unbalance classification (recorded by the provider);
- HIF: ``run_three_phase_nlm_from_path`` (line-level localization) ->
  ``estimate_hif_location_magnitude_multiscan_from_path`` when a persistent
  scan window exists, else ``estimate_hif_location_magnitude_from_path``.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from psse_env.actions import (
    ANOMALY_FAMILY_MARKERS,
    ASK_FOR_MORE_EVIDENCE,
    DIAGNOSTIC_TOOLS,
    ESTIMATE_HIF_FROM_PATH,
    ESTIMATE_HIF_MULTISCAN_FROM_PATH,
    GET_HARMONIC_CONTEXT,
    HIF_DIAGNOSTICS_EXHAUSTED_REQUEST,
    RUN_HSE_FROM_PATH,
    RUN_THREE_PHASE_NLM_FROM_PATH,
    safe_normalize_action,
    unexplained_signatures,
)
from psse_env.oracle.expert_types import (
    ExpertActionProposal,
    matching_evidence_codes,
    policy_state_view,
    state_value,
)


# One shared vocabulary with the environment's explained-anomaly recording,
# so a signature that routes to a diagnostic is the same signature that the
# diagnostic's explanation later accounts for.
HARMONIC_MARKERS = ANOMALY_FAMILY_MARKERS["harmonic"]
UNBALANCE_MARKERS = ANOMALY_FAMILY_MARKERS["three_phase_unbalance"]
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
        # Keep these parameters for API compatibility with the orchestrator,
        # but never let privileged data affect a production diagnostic label.
        del oracle_hints, harmonic_fault_present, hif_fault_present
        state = policy_state_view(state)
        active_id = state_value(state, "active_state_id")
        if not active_id:
            return []
        available = {str(item) for item in state_value(state, "available_evidence", []) or []}
        unresolved = unexplained_signatures(
            state_value(state, "unresolved_signatures", []),
            state_value(state, "explained_anomalies", []),
        )
        harmonic_codes = matching_evidence_codes(unresolved, *HARMONIC_MARKERS)
        unbalance_codes = matching_evidence_codes(unresolved, *UNBALANCE_MARKERS)
        hif_codes = matching_evidence_codes(unresolved, *HIF_MARKERS)
        completed = self._completed_diagnostics(
            history or [], active_state_id=str(active_id)
        )
        proposals: list[ExpertActionProposal] = []

        harmonic_signal = bool(harmonic_codes)
        if harmonic_signal and "harmonic_measurements" in available:
            if GET_HARMONIC_CONTEXT not in completed:
                proposals.append(
                    self._proposal(
                        GET_HARMONIC_CONTEXT,
                        {"state_id": active_id},
                        confidence=0.86,
                        evidence=[
                            "harmonic_telemetry_available",
                            *harmonic_codes,
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

        hif_signal = bool(hif_codes)
        unbalance_signal = bool(unbalance_codes)
        nlm_channel_available = bool(
            "nlm_diagnostic" in available
            or (unbalance_signal and "three_phase_voltages" in available)
        )
        hif_branch = self._nlm_top_branch(completed.get(RUN_THREE_PHASE_NLM_FROM_PATH))
        if (hif_signal or unbalance_signal) and nlm_channel_available and (
            RUN_THREE_PHASE_NLM_FROM_PATH not in completed
        ):
            evidence_codes = hif_codes if hif_signal else unbalance_codes
            proposals.append(
                self._proposal(
                    RUN_THREE_PHASE_NLM_FROM_PATH,
                    {"state_id": active_id},
                    confidence=0.85,
                    evidence=[
                        (
                            "nlm_telemetry_available"
                            if "nlm_diagnostic" in available
                            else "three_phase_voltage_telemetry_available"
                        ),
                        *evidence_codes,
                    ],
                )
            )
        # A ranked NLM branch is not itself proof of HIF.  Escalation requires
        # an independently observable HIF-specific signature; an unbalance or
        # imbalance signature alone stops at the non-HIF classification rung.
        if hif_signal and hif_branch is not None:
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
            required_estimators = [ESTIMATE_HIF_FROM_PATH]
            if "hif_scan_window" in available:
                required_estimators.insert(0, ESTIMATE_HIF_MULTISCAN_FROM_PATH)
            if all(
                self._diagnostic_rejected(completed.get(tool))
                for tool in required_estimators
            ):
                # A rejected model fit is not a clean bill of health.  End the
                # autonomous ladder with an explicit operator handoff request;
                # the environment independently re-audits the full, same-state
                # history before it may treat this request as terminal.
                proposals.append(
                    self._proposal(
                        ASK_FOR_MORE_EVIDENCE,
                        {
                            "state_id": active_id,
                            "request": HIF_DIAGNOSTICS_EXHAUSTED_REQUEST,
                        },
                        confidence=0.95,
                        evidence=[
                            "hif_signature_unexplained",
                            "configured_hif_diagnostics_rejected",
                            "operator_handoff_required",
                        ],
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
        *,
        active_state_id: str | None = None,
    ) -> dict[str, Mapping[str, Any]]:
        """Map attempted diagnostic tools to their latest observable outcome.

        Accepts both raw collector transitions (``action``/``tool_output``)
        and the summarized model history window (``tool``/``outcome``/
        ``observable_metrics``). Failed HIF estimators are retained so the
        ladder can fall back from multiscan to single-scan. Execution failure
        is never converted into a diagnostic rejection: if every configured
        estimator fails operationally, the release terminality gate must
        expose that infrastructure defect.
        """
        completed: dict[str, Mapping[str, Any]] = {}
        for item in history:
            if not isinstance(item, Mapping):
                continue
            action = item.get("action") or item.get("executed_action") or item
            try:
                normalized = safe_normalize_action(action)
                tool = normalized["tool"]
            except Exception:
                continue
            if tool not in DIAGNOSTIC_TOOLS:
                continue
            requested_state = normalized["arguments"].get("state_id")
            if (
                active_state_id is not None
                and requested_state is not None
                and str(requested_state) != str(active_state_id)
                and str(requested_state) not in {"active", "s0"}
            ):
                continue
            output = item.get("tool_output")
            outcome = item.get("outcome")
            if isinstance(output, Mapping):
                status = output.get("execution_status")
                metrics = output.get("tool_metrics")
                error_code = output.get("error_code")
            elif isinstance(outcome, Mapping):
                status = outcome.get("execution_status")
                metrics = item.get("observable_metrics")
                error_code = outcome.get("error_code")
            else:
                status, metrics, error_code = None, None, None
            if status in {"success", "failure"}:
                observed = dict(metrics) if isinstance(metrics, Mapping) else {}
                observed["_execution_status"] = status
                if error_code is not None:
                    observed["_error_code"] = str(error_code)
                completed[tool] = observed
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

    @staticmethod
    def _diagnostic_rejected(metrics: Mapping[str, Any] | None) -> bool:
        if not isinstance(metrics, Mapping):
            return False
        acceptance = metrics.get("diagnostic_acceptance")
        return (
            metrics.get("_execution_status") == "success"
            and isinstance(acceptance, Mapping)
            and acceptance.get("accepted") is False
        )


__all__ = [
    "DiagnosticsExpert",
    "HARMONIC_MARKERS",
    "UNBALANCE_MARKERS",
    "HIF_MARKERS",
]
