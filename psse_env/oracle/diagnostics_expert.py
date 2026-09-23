"""Routing expert for the specialized diagnostic tools.

Routes harmonic, three-phase-unbalance, and high-impedance-fault
investigation from observable signals only: which telemetry channels exist on the active state
(``available_evidence``), which anomaly signatures are unresolved, and what
earlier diagnostics in this episode already produced.  Privileged fault
flags and oracle hints are deliberately ignored here: holding the policy
observation fixed must hold the production target action fixed too.

The intended escalation ladders are:

- harmonic: ``get_harmonic_context`` -> ``run_hse_from_path``;
- three-phase unbalance: ``run_wls`` -> ``get_three_phase_context`` ->
  ``run_three_phase_nlm_from_path`` -> an observable
  non-HIF unbalance classification (recorded by the provider);
- HIF: ``run_three_phase_nlm_from_path`` (line-level localization) ->
  ``estimate_hif_location_magnitude_multiscan_from_path`` when a persistent
  scan window exists, else ``estimate_hif_location_magnitude_from_path``.
"""

from __future__ import annotations

import json
import math
from typing import Any, Mapping, Sequence

from psse_env.evidence_profile import is_scada_only

from psse_env.actions import (
    ANOMALY_FAMILY_MARKERS,
    ASK_FOR_MORE_EVIDENCE,
    CORRECT_MEASUREMENTS,
    DIAGNOSTIC_TOOLS,
    ESTIMATE_HIF_FROM_PATH,
    ESTIMATE_HIF_MULTISCAN_FROM_PATH,
    GET_HARMONIC_CONTEXT,
    GET_MEASUREMENT_CONTEXT,
    GET_PARAMETER_CONTEXT,
    GET_THREE_PHASE_CONTEXT,
    GET_TOPOLOGY_CONTEXT,
    HIF_DIAGNOSTICS_EXHAUSTED_REQUEST,
    HIF_CONDITIONING_UNAVAILABLE_REQUEST,
    RUN_HSE_FROM_PATH,
    RUN_THREE_PHASE_NLM_FROM_PATH,
    RUN_WLS,
    current_gnn_screen,
    safe_normalize_action,
    harmonic_screening_pending,
    successful_current_wls,
    three_phase_acquisition_pending,
    three_phase_context_available,
    three_phase_screening_pending,
    unexplained_signatures,
    waveform_anomaly_signatures,
)
from psse_env.oracle.expert_types import (
    ExpertActionProposal,
    matching_evidence_codes,
    policy_state_view,
    state_value,
)
from psse_env.oracle.hif_continuation import (
    accepted_hif_explanation, current_hif_conditioning,
    hif_conditioned_closure_ready, hif_meter_route_ready,
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

    def hif_continuation_proposals(self, state: Any) -> list[ExpertActionProposal]:
        """Check residuals after every accepted fit, irrespective of true family."""
        state = policy_state_view(state)
        if is_scada_only(state):
            return []
        active = state_value(state, "active_state_id")
        if not active or state_value(state, "has_open_candidate") or not accepted_hif_explanation(state):
            return []
        conditioning = current_hif_conditioning(state)
        if conditioning is None:
            return [self._proposal(RUN_WLS, {"state_id": active}, confidence=1.0,
                evidence=["accepted_hif_requires_current_conditioned_residual_check"])]
        if conditioning["status"] == "unavailable":
            return [self._proposal(ASK_FOR_MORE_EVIDENCE,
                {"state_id": active, "request": HIF_CONDITIONING_UNAVAILABLE_REQUEST}, confidence=1.0,
                evidence=["hif_conditioning_unavailable", "physical_fault_still_present", "operator_handoff_required"])]
        if not hif_meter_route_ready(state):
            return []
        if hif_conditioned_closure_ready(state) and not state_value(state, "accepted_corrections", []):
            return []
        contexts = state_value(state, "fresh_context_evidence") or {}
        measurement = contexts.get("measurement") or {}
        if (not state_value(state, "has_fresh_measurement_context")
            or str(measurement.get("state_id") or "") != str(active)
            or measurement.get("state_hash") != conditioning["state_hash"]):
            return [self._proposal(GET_MEASUREMENT_CONTEXT, {"state_id": active}, confidence=1.0,
                evidence=["hif_effect_accounted", "conditional_meter_investigation_required"])]
        candidates = set(conditioning["remaining_meter_candidate_indices"])
        proposals = []
        for raw in measurement.get("supported_corrections") or []:
            if not isinstance(raw, Mapping):
                continue
            action = safe_normalize_action(raw)
            indices = action["arguments"].get("suspect_group")
            if (action["tool"] != CORRECT_MEASUREMENTS
                or str(action["arguments"].get("state_id") or "") != str(active)
                or not isinstance(indices, (list, tuple)) or not indices
                or any(type(index) is not int for index in indices)
                or len(set(indices)) != len(indices) or not set(indices) <= candidates):
                continue
            proposals.append(self._proposal(CORRECT_MEASUREMENTS, action["arguments"], confidence=.99,
                evidence=["hif_effect_accounted", "same_state_conditional_meter_candidate"]))
        return proposals

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
        if is_scada_only(state):
            return []
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
        if harmonic_signal:
            if GET_HARMONIC_CONTEXT not in completed:
                proposals.append(
                    self._proposal(
                        GET_HARMONIC_CONTEXT,
                        {"state_id": active_id},
                        confidence=0.86,
                        evidence=[
                            "harmonic_measurements_requested",
                            *harmonic_codes,
                        ],
                    )
                )
            elif (
                RUN_HSE_FROM_PATH not in completed
                and "harmonic_measurements" in available
            ):
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
        phase_contexts = state_value(state, "fresh_context_evidence") or {}
        phase_context = phase_contexts.get("three_phase") or {}
        phase_acquired = three_phase_context_available(phase_contexts, active_id)
        phase_attempted = (
            str(phase_context.get("state_id") or "") == str(active_id)
            and phase_context.get("request_attempted") is True
        )
        if unbalance_signal and not hif_signal:
            if not successful_current_wls(state, history):
                return [self._proposal(
                    RUN_WLS, {"state_id": active_id}, confidence=0.95,
                    evidence=["unbalance_requires_current_wls_baseline"],
                )]
            if not phase_attempted:
                return [self._proposal(
                    GET_THREE_PHASE_CONTEXT, {"state_id": active_id}, confidence=0.95,
                    evidence=["three_phase_measurements_requested", *unbalance_codes],
                )]
        current_channel_available = "three_phase_branch_currents" in available
        nlm_channel_available = bool(
            "nlm_diagnostic" in available
            or current_channel_available
            or (unbalance_signal and "three_phase_voltages" in available)
        )
        nlm_metrics = completed.get(RUN_THREE_PHASE_NLM_FROM_PATH)
        hif_branch = self._nlm_top_branch(nlm_metrics)
        hif_phase = self._nlm_suspected_phase(nlm_metrics)
        nlm_attempted = (
            str(phase_context.get("state_id") or "") == str(active_id)
            and phase_context.get("nlm_attempted") is True
        )
        # A new acquisition retires the previous scan's NLM outcome. The
        # durable ledger is authoritative even when older successful/rejected
        # diagnostics remain in the visible history window.
        nlm_completed = (
            nlm_attempted if phase_attempted
            else RUN_THREE_PHASE_NLM_FROM_PATH in completed
        )
        if (hif_signal or (unbalance_signal and phase_acquired)) and nlm_channel_available and (
            not nlm_completed
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
                            else (
                                "three_phase_branch_current_telemetry_available"
                                if current_channel_available
                                else "three_phase_voltage_telemetry_available"
                            )
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
            # An observable per-phase differential current names the faulted
            # phase; passing it bounds the estimator search to that phase.
            if hif_phase is not None:
                follow_up_arguments["candidate_phase"] = hif_phase
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

    def gnn_balanced_screening_proposals(
        self, state: Any, history: Sequence[Mapping[str, Any]] | None = None,
    ) -> list[ExpertActionProposal]:
        """Inspect each balanced-error context once after a bound anomaly score.

        Family scores only order read-only requests. They do not name a faulty
        asset or make a correction admissible. A failed/untrained phase head is
        not a negative phase screen, and existing waveform findings retain their
        diagnostic route.
        """
        del history
        state = policy_state_view(state)
        active_id = str(state_value(state, "active_state_id") or "")
        if not active_id or state_value(state, "has_open_candidate"):
            return []
        strict = is_scada_only(state)
        if not strict and waveform_anomaly_signatures(state_value(state, "unresolved_signatures", [])):
            return []
        contexts = state_value(state, "fresh_context_evidence") or {}
        if not isinstance(contexts, Mapping):
            return []
        screen = current_gnn_screen(state)
        if not ((strict and (screen.get("phase_trigger") is True or screen.get("anomaly_trigger") is True))
            or (not strict and screen.get("phase_trigger") is False and screen.get("anomaly_trigger") is True)):
            return []
        state_hash = screen["state_hash"]
        tried: set[str] = set()
        for signature in state_value(state, "tried_action_signatures", []) or []:
            tool, separator, encoded = str(signature).partition(":")
            if not separator:
                continue
            try:
                arguments = json.loads(encoded)
            except (TypeError, ValueError):
                continue
            if isinstance(arguments, Mapping) and str(arguments.get("state_id") or "") == active_id:
                tried.add(tool)
        routes = [
            ("measurement", GET_MEASUREMENT_CONTEXT),
            ("parameter", GET_PARAMETER_CONTEXT),
            ("topology", GET_TOPOLOGY_CONTEXT),
        ]
        scores = screen.get("family_scores") or {}
        scores = scores if isinstance(scores, Mapping) else {}
        disabled = screen.get("disabled_family_heads") or []

        def ranking(route: tuple[str, str]) -> float:
            family = route[0]
            score = scores.get(family)
            if family in disabled or isinstance(score, bool) or not isinstance(score, (float, int)):
                return -1.0
            return float(score) if math.isfinite(score) and 0 <= score <= 1 else -1.0

        for family, tool in sorted(routes, key=ranking, reverse=True):
            context = contexts.get(family) or {}
            if isinstance(context, Mapping) and (
                str(context.get("state_id") or "") == active_id
                and context.get("state_hash") == state_hash
            ):
                continue
            if tool in tried:
                continue
            return [self._proposal(
                tool, {"state_id": active_id}, confidence=0.95,
                evidence=["gnn_anomaly_screen_positive", "balanced_context_investigation_requested"],
            )]
        return []

    def harmonic_screening_proposals(
        self, state: Any, history: Sequence[Mapping[str, Any]] | None = None,
    ) -> list[ExpertActionProposal]:
        """Request additional measurements after WLS, without a family hint."""
        state = policy_state_view(state)
        if is_scada_only(state):
            return []
        active_id = state_value(state, "active_state_id")
        if not active_id or state_value(state, "has_open_candidate"):
            return []
        if not harmonic_screening_pending(
            unresolved=state_value(state, "unresolved_signatures", []),
            tried_action_signatures=state_value(state, "tried_action_signatures", []),
            active_state_id=active_id,
            context_evidence=state_value(state, "fresh_context_evidence"),
            evidence_profile=state_value(state, "evidence_profile"),
        ):
            return []
        if GET_HARMONIC_CONTEXT in self._completed_diagnostics(
            history or [], active_state_id=str(active_id)
        ):
            return []
        return [self._proposal(
            GET_HARMONIC_CONTEXT, {"state_id": active_id}, confidence=0.95,
            evidence=["fundamental_anomaly_detected", "request_spectral_evidence_before_correction"],
        )]

    def three_phase_screening_proposals(
        self,
        state: Any,
        history: Sequence[Mapping[str, Any]] | None = None,
    ) -> list[ExpertActionProposal]:
        """Acquire phase measurements, then screen an unflagged WLS anomaly."""
        state = policy_state_view(state)
        if is_scada_only(state):
            return []
        active_id = state_value(state, "active_state_id")
        if not active_id or state_value(state, "has_open_candidate"):
            return []
        raw_unresolved = state_value(state, "unresolved_signatures", []) or []
        if waveform_anomaly_signatures(raw_unresolved):
            return []
        unresolved = unexplained_signatures(
            raw_unresolved, state_value(state, "explained_anomalies", [])
        )
        fundamental = [str(item) for item in unresolved if str(item).startswith("wls_")]
        if not fundamental:
            return []
        contexts = state_value(state, "fresh_context_evidence")
        if three_phase_acquisition_pending(
            unresolved=unresolved,
            tried_action_signatures=state_value(state, "tried_action_signatures", []),
            active_state_id=active_id,
            context_evidence=contexts,
            evidence_profile=state_value(state, "evidence_profile"),
        ):
            tool = GET_THREE_PHASE_CONTEXT if successful_current_wls(state, history) else RUN_WLS
            return [self._proposal(
                tool, {"state_id": active_id}, confidence=0.95,
                evidence=["fundamental_anomaly_detected", "request_three_phase_evidence_before_correction"],
            )]
        available = {str(item) for item in state_value(state, "available_evidence", []) or []}
        if not three_phase_screening_pending(
            unresolved=unresolved, available_evidence=available,
            tried_action_signatures=state_value(state, "tried_action_signatures", []),
            active_state_id=active_id, context_evidence=contexts,
            evidence_profile=state_value(state, "evidence_profile"),
        ):
            return []
        if not successful_current_wls(state, history):
            return [self._proposal(
                RUN_WLS, {"state_id": active_id}, confidence=0.95,
                evidence=["three_phase_screening_requires_current_wls_baseline"],
            )]
        completed = self._completed_diagnostics(
            history or [], active_state_id=str(active_id)
        )
        if not isinstance(contexts, Mapping) and RUN_THREE_PHASE_NLM_FROM_PATH in completed:
            return []
        return [
            self._proposal(
                RUN_THREE_PHASE_NLM_FROM_PATH,
                {"state_id": active_id},
                confidence=0.95,
                evidence=[
                    "fundamental_anomaly_detected",
                    "three_phase_telemetry_available",
                    "three_phase_screening_before_correction",
                    *fundamental[:3],
                ],
            )
        ]

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
                if (
                    tool in {GET_HARMONIC_CONTEXT, RUN_HSE_FROM_PATH,
                             GET_THREE_PHASE_CONTEXT, RUN_THREE_PHASE_NLM_FROM_PATH}
                    and status == "failure" and error_code == "missing_precondition"
                ):
                    continue
                observed = dict(metrics) if isinstance(metrics, Mapping) else {}
                observed["_execution_status"] = status
                if error_code is not None:
                    observed["_error_code"] = str(error_code)
                completed[tool] = observed
        return completed

    @staticmethod
    def _nlm_suspected_phase(metrics: Mapping[str, Any] | None) -> str | None:
        """Faulted phase reported by terminal-current NLM output, if any."""
        if not isinstance(metrics, Mapping):
            return None
        summary = metrics.get("nlm_summary")
        if not isinstance(summary, Mapping):
            return None
        phase = summary.get("suspected_phase")
        if phase is None:
            return None
        text = str(phase).strip().upper()
        return text if text in {"A", "B", "C"} else None

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
