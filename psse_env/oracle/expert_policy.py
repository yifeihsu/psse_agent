from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from psse_env.actions import (
    ASK_FOR_MORE_EVIDENCE,
    CORRECT_MEASUREMENTS,
    CORRECT_PARAMETERS,
    CORRECT_TOPOLOGY,
    CORRECTION_TOOLS,
    GET_MEASUREMENT_CONTEXT,
    GET_PARAMETER_CONTEXT,
    GET_TOPOLOGY_CONTEXT,
    RECOVERY_BUDGET_EXHAUSTED_REQUEST,
    RECOVERY_OPTIONS_EXHAUSTED_REQUEST,
    RUN_WLS,
    action_signature,
    safe_normalize_action,
)
from psse_env.oracle.candidate_quality import CandidateQualityOracle
from psse_env.oracle.diagnostics_expert import DiagnosticsExpert
from psse_env.oracle.expert_types import ExpertActionProposal
from psse_env.oracle.measurement_expert import MeasurementExpert
from psse_env.oracle.parameter_expert import ParameterExpert
from psse_env.oracle.process_validity import ProcessValidityOracle
from psse_env.oracle.recovery_expert import RecoveryExpert
from psse_env.oracle.termination_expert import TerminationExpert
from psse_env.oracle.topology_expert import TopologyExpert
from psse_env.state_store import OracleState, PolicyObservation, policy_safe_copy


@dataclass(frozen=True)
class _ExpertContext:
    policy_state: PolicyObservation | Mapping[str, Any]
    history: list[Mapping[str, Any]]
    oracle_hints: list[Mapping[str, Any]]
    oracle_fault_families: frozenset[str]


class ExpertPolicyOracle:
    """Recovery-aware orchestrator for the domain-specific expert modules.

    Policy-visible state and privileged oracle data are separated at entry.
    Sub-experts receive the policy observation plus narrowly routed action hints
    and family-presence flags; privileged payloads are never copied into that
    observation or returned as policy features.
    """

    def __init__(
        self,
        *,
        process_oracle: ProcessValidityOracle | None = None,
        candidate_oracle: CandidateQualityOracle | None = None,
        measurement_expert: MeasurementExpert | None = None,
        parameter_expert: ParameterExpert | None = None,
        topology_expert: TopologyExpert | None = None,
        diagnostics_expert: DiagnosticsExpert | None = None,
        recovery_expert: RecoveryExpert | None = None,
        termination_expert: TerminationExpert | None = None,
    ) -> None:
        self.process_oracle = process_oracle or ProcessValidityOracle()
        # Kept as a public collaborator for callers that use the expert policy
        # and candidate assessment through one object.
        self.candidate_oracle = candidate_oracle or CandidateQualityOracle()
        self.measurement_expert = measurement_expert or MeasurementExpert()
        self.parameter_expert = parameter_expert or ParameterExpert()
        self.topology_expert = topology_expert or TopologyExpert()
        self.diagnostics_expert = diagnostics_expert or DiagnosticsExpert()
        self.recovery_expert = recovery_expert or RecoveryExpert(self.process_oracle)
        self.termination_expert = termination_expert or TerminationExpert(
            anomaly_threshold=self.process_oracle.anomaly_threshold
        )

    def next_actions(
        self,
        state: OracleState | PolicyObservation | Mapping[str, Any],
        history: list[Mapping[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """Return ranked admissible actions for an oracle or policy-like state."""
        return [proposal.as_action() for proposal in self.next_action_proposals(state, history)]

    def next_action_proposals(
        self,
        state: OracleState | PolicyObservation | Mapping[str, Any],
        history: list[Mapping[str, Any]] | None = None,
    ) -> list[ExpertActionProposal]:
        """Return evidence-bearing proposals in the required control-flow order."""
        context = self._context(state, history)
        policy = context.policy_state
        seen_signatures = self._seen_action_signatures(policy, context.history)
        blocked_correction_tools = self._structurally_blocked_correction_tools(
            policy, context.history
        )

        recovery = self.recovery_expert.repair_actions(policy, context.history)
        if recovery:
            return self._rank_and_filter(
                recovery,
                policy,
                seen_signatures=seen_signatures,
                blocked_correction_tools=blocked_correction_tools,
                mandatory=True,
            )

        verification = self.termination_expert.verification_actions(policy)
        if verification:
            return self._rank_and_filter(
                verification,
                policy,
                seen_signatures=seen_signatures,
                blocked_correction_tools=blocked_correction_tools,
                mandatory=True,
            )

        disposition = self.termination_expert.candidate_disposition_actions(policy)
        if disposition:
            return self._rank_and_filter(
                disposition,
                policy,
                seen_signatures=seen_signatures,
                blocked_correction_tools=blocked_correction_tools,
                mandatory=True,
            )

        terminal = self.termination_expert.propose(policy, context.history)
        if terminal:
            return self._rank_and_filter(
                terminal,
                policy,
                seen_signatures=seen_signatures,
                blocked_correction_tools=blocked_correction_tools,
                mandatory=True,
            )

        # Hidden fault-family flags and action hints may rank later diagnosis,
        # but they must never choose a different label for an otherwise
        # indistinguishable initial learner observation.  Establish a common
        # observable baseline before consulting domain-specific privileged
        # routing information.
        if self._needs_observable_baseline(policy, context.history):
            active_id = self._get(policy, "active_state_id")
            baseline = [
                ExpertActionProposal(
                    action={"tool": RUN_WLS, "arguments": {"state_id": active_id}},
                    source_expert="diagnostic_baseline",
                    confidence=1.0,
                    evidence_codes=[
                        "observable_baseline_required",
                        "hidden_family_routing_deferred",
                    ],
                    admissible=active_id is not None,
                    estimated_immediate_risk=0.0,
                )
            ]
            return self._rank_and_filter(
                baseline,
                policy,
                seen_signatures=seen_signatures,
                blocked_correction_tools=blocked_correction_tools,
                mandatory=True,
            )

        budget_handoff = self._recovery_budget_proposals(policy, context.history)
        if budget_handoff:
            return self._rank_and_filter(
                budget_handoff,
                policy,
                seen_signatures=seen_signatures,
                blocked_correction_tools=blocked_correction_tools,
                mandatory=False,
            )

        proposals = (
            self.measurement_expert.propose(
                policy,
                context.history,
                oracle_hints=context.oracle_hints,
                oracle_fault_present="measurement" in context.oracle_fault_families,
            )
            + self.parameter_expert.propose(
                policy,
                context.history,
                oracle_hints=context.oracle_hints,
                oracle_fault_present="parameter" in context.oracle_fault_families,
            )
            + self.topology_expert.propose(
                policy,
                context.history,
                oracle_hints=context.oracle_hints,
                oracle_fault_present="topology" in context.oracle_fault_families,
            )
            + self.diagnostics_expert.propose(
                policy,
                context.history,
                oracle_hints=context.oracle_hints,
                harmonic_fault_present="harmonic" in context.oracle_fault_families,
                hif_fault_present="hif" in context.oracle_fault_families,
            )
        )
        ranked = self._rank_and_filter(
            proposals,
            policy,
            seen_signatures=seen_signatures,
            blocked_correction_tools=blocked_correction_tools,
            mandatory=False,
        )
        if ranked:
            return ranked
        escalation = self._recovery_exhaustion_proposals(policy, context.history)
        return self._rank_and_filter(
            escalation,
            policy,
            seen_signatures=seen_signatures,
            blocked_correction_tools=blocked_correction_tools,
            mandatory=False,
        )

    def enumerate_admissible_actions(
        self,
        state: OracleState | PolicyObservation | Mapping[str, Any],
        history: list[Mapping[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        return self.next_actions(state, history)

    def enumerate_top_actions(
        self,
        state: OracleState | PolicyObservation | Mapping[str, Any],
        top_l: int = 8,
        history: list[Mapping[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        return self.next_actions(state, history)[: max(int(top_l), 0)]

    def label_transition(
        self,
        *,
        state: OracleState | PolicyObservation | Mapping[str, Any],
        action: Mapping[str, Any] | str,
        tool_output: Mapping[str, Any],
        next_state: OracleState | PolicyObservation | Mapping[str, Any],
        history: list[Mapping[str, Any]] | None = None,
        store: Any | None = None,
        hidden_truth: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Label the executed transition without querying next-state actions.

        The collector owns that second query after it has appended the current
        transition to history.  This avoids producing labels from stale
        history while preserving immediate process-repair actions.
        """
        del history, hidden_truth
        current_policy = self._context(state, None).policy_state
        next_policy = self._context(next_state, None).policy_state
        process_label = self.process_oracle.check(current_policy, action, store=store)
        disposition = self._candidate_disposition(next_policy, tool_output)
        normalized = safe_normalize_action(action)
        if normalized["tool"] in {RUN_WLS, "verify_candidate"} and self._get(
            next_policy, "has_verified_candidate", False
        ):
            disposition = disposition or "INCONCLUSIVE"

        return {
            **process_label,
            "execution_status": tool_output.get("execution_status", "unknown"),
            "candidate_disposition": disposition,
            "progress_class": self._progress_class(tool_output, disposition),
            # Only immediate repairs belong here.  The collector fills normal
            # next actions after updating the transition history.
            "valid_next_actions": list(process_label.get("valid_next_actions") or []),
        }

    def _context(
        self,
        state: OracleState | PolicyObservation | Mapping[str, Any],
        history: list[Mapping[str, Any]] | None,
    ) -> _ExpertContext:
        hints: list[Mapping[str, Any]] = []
        fault_families: set[str] = set()

        if isinstance(state, OracleState):
            policy: PolicyObservation | Mapping[str, Any] = state.policy_observation.as_dict()
            # Candidate quality is privileged supervision.  It is added only
            # to this private expert view, never to PolicyObservation.
            policy["candidate_disposition"] = state.candidate_disposition
            policy["candidate_lifecycle"] = state.candidate_lifecycle
            policy["candidate_assessment"] = dict(state.candidate_assessment)
            hints = [dict(item) for item in state.oracle_action_hints if isinstance(item, Mapping)]
            if state.true_measurement_errors:
                fault_families.add("measurement")
            if state.true_parameter_errors:
                fault_families.add("parameter")
            if state.true_topology_errors:
                fault_families.add("topology")
        elif isinstance(state, PolicyObservation):
            policy = state.as_dict()
        elif isinstance(state, Mapping):
            nested = state.get("policy_observation")
            if isinstance(nested, PolicyObservation):
                policy = nested.as_dict()
            elif isinstance(nested, Mapping):
                policy = policy_safe_copy(dict(nested))
            else:
                # A flat mapping is treated as a legacy policy observation.
                # Privileged keys are stripped and never interpreted as hints.
                policy = policy_safe_copy(dict(state))

            if nested is not None:
                # Serialized OracleState compatibility.  Flat mappings remain
                # policy-only and have privileged keys stripped above.
                policy["candidate_disposition"] = state.get("candidate_disposition")
                policy["candidate_lifecycle"] = state.get(
                    "candidate_lifecycle", policy.get("candidate_lifecycle")
                )
                assessment = state.get("candidate_assessment")
                if isinstance(assessment, Mapping):
                    policy["candidate_assessment"] = dict(assessment)
                raw_hints = state.get("oracle_action_hints") or []
                hints = [dict(item) for item in raw_hints if isinstance(item, Mapping)]
                if state.get("true_measurement_errors"):
                    fault_families.add("measurement")
                if state.get("true_parameter_errors"):
                    fault_families.add("parameter")
                if state.get("true_topology_errors"):
                    fault_families.add("topology")
        else:
            raise TypeError(f"state must be OracleState, PolicyObservation, or mapping, got {type(state).__name__}")

        # Older callers exposed only the two verified/unverified booleans.  A
        # candidate in either state is necessarily open for process-gate
        # purposes, so restore that implied lifecycle flag on our private copy.
        if isinstance(policy, Mapping):
            policy = dict(policy)
            if policy.get("has_unverified_candidate") or policy.get("has_verified_candidate"):
                policy["has_open_candidate"] = True
        if history is None:
            raw_history = self._get(policy, "history_window", []) or []
            history_items = [dict(item) for item in raw_history if isinstance(item, Mapping)]
        else:
            history_items = [dict(item) for item in history if isinstance(item, Mapping)]
        if isinstance(policy, Mapping):
            observable_corrections = self._observable_supported_corrections(
                policy, history_items
            )
            if observable_corrections:
                # Context evidence remains usable after a rejected candidate
                # is rolled back to the exact same active state.  The latest
                # tool output is then the rollback rather than the context
                # payload, so discarding history here used to strand every
                # untried provider-supported alternative.  Only context
                # outputs bound to a still-fresh active state are recovered.
                # They remain the sole source of correction targets; private
                # scenario hints are used only when no observable provider
                # contract exists.
                hints = observable_corrections
        return _ExpertContext(
            policy_state=policy,
            history=history_items,
            oracle_hints=hints,
            oracle_fault_families=frozenset(fault_families),
        )

    @classmethod
    def _observable_supported_corrections(
        cls,
        policy: Mapping[str, Any],
        history: Sequence[Mapping[str, Any]] = (),
    ) -> list[dict[str, Any]]:
        active_id = policy.get("active_state_id")
        if active_id is None:
            return []
        context_contracts = {
            GET_MEASUREMENT_CONTEXT: ("measurement", CORRECT_MEASUREMENTS),
            GET_PARAMETER_CONTEXT: ("parameter", CORRECT_PARAMETERS),
            GET_TOPOLOGY_CONTEXT: ("topology", CORRECT_TOPOLOGY),
        }

        # The explicit history is authoritative.  The synthetic last-output
        # event supports direct callers whose compact policy observation does
        # not carry a history window.
        events = list(history)
        last_tool = policy.get("last_tool")
        last_output = policy.get("last_tool_output")
        if last_tool in context_contracts and isinstance(last_output, Mapping):
            events.append(
                {
                    "action": {
                        "tool": last_tool,
                        "arguments": {"state_id": active_id},
                    },
                    "tool_output": last_output,
                }
            )

        actions: list[dict[str, Any]] = []
        emitted: set[str] = set()
        latest_context_seen: set[str] = set()
        for event in reversed(events):
            event_action = event.get("action") or event.get("executed_action")
            normalized_event = safe_normalize_action(event_action or {})
            context_tool = normalized_event["tool"]
            contract = context_contracts.get(context_tool)
            if contract is None or context_tool in latest_context_seen:
                continue
            family, correction_tool = contract
            if not (
                policy.get(f"has_fresh_{family}_context", False)
                and str(policy.get(f"{family}_context_state_id")) == str(active_id)
            ):
                continue
            requested = normalized_event["arguments"].get("state_id")
            if requested is not None and str(requested) != str(active_id):
                continue
            latest_context_seen.add(context_tool)
            output = event.get("tool_output")
            if not isinstance(output, Mapping) or output.get("execution_status") == "failure":
                continue
            metrics = output.get("tool_metrics")
            if not isinstance(metrics, Mapping):
                metrics = output.get("observable_metrics")
            if not isinstance(metrics, Mapping):
                continue
            raw_actions = metrics.get("supported_corrections")
            if not isinstance(raw_actions, (list, tuple)):
                continue
            for raw_action in raw_actions:
                if not isinstance(raw_action, Mapping):
                    continue
                normalized = safe_normalize_action(raw_action)
                if normalized["tool"] != correction_tool:
                    continue
                target_state = normalized["arguments"].get("state_id")
                if target_state is not None and str(target_state) != str(active_id):
                    continue
                signature = cls._signature(normalized)
                if signature is None or signature in emitted:
                    continue
                emitted.add(signature)
                actions.append(normalized)
        return actions

    def _rank_and_filter(
        self,
        proposals: Sequence[ExpertActionProposal],
        policy: PolicyObservation | Mapping[str, Any],
        *,
        seen_signatures: set[str],
        blocked_correction_tools: set[str],
        mandatory: bool,
    ) -> list[ExpertActionProposal]:
        assessed: list[tuple[int, ExpertActionProposal, str]] = []
        for index, proposal in enumerate(proposals):
            normalized = safe_normalize_action(proposal.action)
            if normalized["tool"] in blocked_correction_tools:
                continue
            signature = self._signature(proposal.action)
            equivalent_signatures = self._equivalent_signatures(proposal.action)
            if signature is None or equivalent_signatures & seen_signatures:
                continue
            validity = self.process_oracle.check(policy, proposal.action)
            checked = proposal.with_admissibility(
                proposal.admissible and bool(validity.get("process_valid"))
            )
            if checked.admissible:
                assessed.append((index, checked, signature))

        # Mandatory lifecycle actions are allowed to repeat only when the state
        # still requires them.  Domain proposals never repeat a tried/rejected
        # signature.
        if not assessed and mandatory and proposals:
            for index, proposal in enumerate(proposals):
                normalized = safe_normalize_action(proposal.action)
                if normalized["tool"] in blocked_correction_tools:
                    continue
                signature = self._signature(proposal.action)
                if signature is None:
                    continue
                validity = self.process_oracle.check(policy, proposal.action)
                checked = proposal.with_admissibility(
                    proposal.admissible and bool(validity.get("process_valid"))
                )
                if checked.admissible:
                    assessed.append((index, checked, signature))

        source_priority = {
            "recovery_expert": 0,
            "termination_expert": 1,
            "diagnostic_baseline": 2,
            "parameter_expert": 3,
            "topology_expert": 4,
            "measurement_expert": 5,
            "diagnostics_expert": 6,
        }
        assessed.sort(
            key=lambda item: (
                -item[1].confidence,
                item[1].estimated_immediate_risk,
                source_priority.get(item[1].source_expert, 99),
                item[0],
            )
        )
        result: list[ExpertActionProposal] = []
        emitted: set[str] = set()
        for _, proposal, signature in assessed:
            if signature in emitted:
                continue
            emitted.add(signature)
            result.append(proposal)
        return result

    @classmethod
    def _needs_observable_baseline(
        cls,
        policy: PolicyObservation | Mapping[str, Any],
        history: Sequence[Mapping[str, Any]],
    ) -> bool:
        if cls._get(policy, "has_open_candidate", False):
            return False
        if cls._get(policy, "last_tool") is not None:
            return False
        if history:
            return False
        if cls._get(policy, "remaining_anomaly_score") is not None:
            return False
        if cls._get(policy, "unresolved_signatures", []) or cls._get(
            policy, "accepted_corrections", []
        ) or cls._get(policy, "rejected_hypotheses", []):
            return False
        if any(
            cls._get(policy, f"has_fresh_{family}_context", False)
            for family in ("measurement", "parameter", "topology")
        ):
            return False
        return True

    def _recovery_exhaustion_proposals(
        self,
        policy: PolicyObservation | Mapping[str, Any],
        history: Sequence[Mapping[str, Any]],
    ) -> list[ExpertActionProposal]:
        """Request an operator handoff after observable autonomous options end.

        This is not a resolution label.  It is reachable only after a
        successful WLS solve on the current state plus at least one observable
        investigation step, and only when every ordinary expert proposal has
        already been filtered out.  The environment re-audits the full history
        and provider response before treating the request as terminal.
        """
        if self._get(policy, "has_open_candidate", False):
            return []
        active_id = self._get(policy, "active_state_id")
        if active_id is None or self._get(policy, "no_material_anomaly_remaining", False):
            return []
        score = self._get(policy, "remaining_anomaly_score")
        try:
            score_unresolved = (
                score is not None
                and float(score) >= float(self.process_oracle.anomaly_threshold)
            )
        except (TypeError, ValueError):
            score_unresolved = False
        if not (
            score_unresolved
            or bool(self._get(policy, "unresolved_signatures", []) or [])
        ):
            return []

        successful_current_wls = False
        investigation_seen = False
        investigation_tools = {
            GET_MEASUREMENT_CONTEXT,
            GET_PARAMETER_CONTEXT,
            GET_TOPOLOGY_CONTEXT,
            *CORRECTION_TOOLS,
        }
        for event in history:
            if not isinstance(event, Mapping):
                continue
            event_action = safe_normalize_action(
                event.get("action") or event.get("executed_action") or {}
            )
            requested = event_action["arguments"].get("state_id")
            output = event.get("tool_output")
            success = (
                isinstance(output, Mapping)
                and output.get("execution_status") == "success"
            )
            if (
                event_action["tool"] == RUN_WLS
                and success
                and requested is not None
                and str(requested) == str(active_id)
            ):
                successful_current_wls = True
            if (
                event_action["tool"] in investigation_tools
                and success
                and (requested is None or str(requested) == str(active_id))
            ):
                investigation_seen = True
        if not (successful_current_wls and investigation_seen):
            return []
        return [
            ExpertActionProposal(
                action={
                    "tool": ASK_FOR_MORE_EVIDENCE,
                    "arguments": {
                        "state_id": active_id,
                        "request": RECOVERY_OPTIONS_EXHAUSTED_REQUEST,
                    },
                },
                source_expert="recovery_expert",
                confidence=1.0,
                evidence_codes=[
                    "observable_recovery_options_exhausted",
                    "unresolved_anomaly_requires_operator_handoff",
                ],
                admissible=True,
                estimated_immediate_risk=0.0,
            )
        ]

    def _recovery_budget_proposals(
        self,
        policy: PolicyObservation | Mapping[str, Any],
        history: Sequence[Mapping[str, Any]],
    ) -> list[ExpertActionProposal]:
        """Hand off before opening a lifecycle the remaining budget cannot close."""
        if self._get(policy, "has_open_candidate", False):
            return []
        active_id = self._get(policy, "active_state_id")
        if active_id is None or self._get(
            policy, "no_material_anomaly_remaining", False
        ):
            return []
        try:
            remaining_budget = int(self._get(policy, "remaining_budget", 0) or 0)
        except (TypeError, ValueError):
            return []
        # A new hypothesis needs correction -> verification -> disposition,
        # followed by either finalization or an explicit handoff.
        if not 0 < remaining_budget < 4:
            return []
        if not (
            self._get(policy, "unresolved_signatures", [])
            or self._get(policy, "remaining_anomaly_score") is not None
        ):
            return []

        successful_current_wls = False
        investigation_seen = False
        investigation_tools = {
            GET_MEASUREMENT_CONTEXT,
            GET_PARAMETER_CONTEXT,
            GET_TOPOLOGY_CONTEXT,
            *CORRECTION_TOOLS,
        }
        for event in history:
            if not isinstance(event, Mapping):
                continue
            event_action = safe_normalize_action(
                event.get("action") or event.get("executed_action") or {}
            )
            requested = event_action["arguments"].get("state_id")
            output = event.get("tool_output")
            success = (
                isinstance(output, Mapping)
                and output.get("execution_status") == "success"
            )
            if (
                event_action["tool"] == RUN_WLS
                and success
                and requested is not None
                and str(requested) == str(active_id)
            ):
                successful_current_wls = True
            if (
                event_action["tool"] in investigation_tools
                and success
                and (requested is None or str(requested) == str(active_id))
            ):
                investigation_seen = True
        if not (successful_current_wls and investigation_seen):
            return []
        return [
            ExpertActionProposal(
                action={
                    "tool": ASK_FOR_MORE_EVIDENCE,
                    "arguments": {
                        "state_id": active_id,
                        "request": RECOVERY_BUDGET_EXHAUSTED_REQUEST,
                    },
                },
                source_expert="recovery_expert",
                confidence=1.0,
                evidence_codes=[
                    "autonomous_recovery_budget_exhausted",
                    "operator_handoff_required",
                ],
                admissible=True,
                estimated_immediate_risk=0.0,
            )
        ]

    def _seen_action_signatures(
        self,
        policy: PolicyObservation | Mapping[str, Any],
        history: Sequence[Mapping[str, Any]],
    ) -> set[str]:
        active_id = self._get(policy, "active_state_id")
        signatures = {
            str(signature)
            for signature in (self._get(policy, "tried_action_signatures", []) or [])
            if signature
        }
        for signature in list(signatures):
            semantic = (
                self._semantic_signature_from_text(signature)
                if self._signature_applies_to_state(signature, active_id)
                else None
            )
            if semantic:
                signatures.add(semantic)
        for field in ("rejected_hypotheses", "accepted_corrections"):
            for remembered in self._get(policy, field, []) or []:
                if not isinstance(remembered, Mapping):
                    continue
                if remembered.get("action_signature"):
                    text_signature = str(remembered["action_signature"])
                    signatures.add(text_signature)
                    semantic = (
                        self._semantic_signature_from_text(text_signature)
                        if self._signature_applies_to_state(text_signature, active_id)
                        else None
                    )
                    if semantic:
                        signatures.add(semantic)
                source_action = remembered.get("source_action")
                if source_action:
                    normalized = safe_normalize_action(source_action)
                    exact = self._signature(normalized)
                    if exact is not None:
                        signatures.add(exact)
                    requested = normalized["arguments"].get("state_id")
                    if (
                        normalized["tool"] not in CORRECTION_TOOLS
                        or requested is None
                        or str(requested) == str(active_id)
                    ):
                        semantic = self._semantic_signature_from_text(exact or "")
                        if semantic:
                            signatures.add(semantic)
        for item in history:
            action = item.get("action") or item.get("executed_action")
            if action:
                normalized = safe_normalize_action(action)
                exact = self._signature(normalized)
                if exact is not None:
                    signatures.add(exact)
                requested = normalized["arguments"].get("state_id")
                if (
                    normalized["tool"] not in CORRECTION_TOOLS
                    or requested is None
                    or str(requested) == str(active_id)
                ):
                    semantic = self._semantic_signature_from_text(exact or "")
                    if semantic:
                        signatures.add(semantic)
        return signatures

    @classmethod
    def _structurally_blocked_correction_tools(
        cls,
        policy: PolicyObservation | Mapping[str, Any],
        history: Sequence[Mapping[str, Any]],
    ) -> set[str]:
        """Return correction families unavailable on the current active state.

        Some executor failures describe the state contract, not a bad target.
        In particular, ``parameter_scans_missing`` means *every* proposed line
        will fail until the active state changes.  Treating it like a
        target-specific rejection caused the expert to exhaust the same invalid
        operation over every ranked line.  This gate consumes only observable
        action/output history and is scoped to the state on which it occurred.
        """

        active_id = cls._get(policy, "active_state_id")
        if active_id is None:
            return set()
        family_wide_failures = {
            CORRECT_PARAMETERS: {"parameter_scans_missing"},
            CORRECT_TOPOLOGY: {"topology_correction_unsupported"},
        }
        events = list(history)
        last_tool = cls._get(policy, "last_tool")
        last_output = cls._get(policy, "last_tool_output", {})
        if last_tool in family_wide_failures and isinstance(last_output, Mapping):
            events.append(
                {
                    "action": {
                        "tool": last_tool,
                        "arguments": {"state_id": active_id},
                    },
                    "tool_output": last_output,
                }
            )
        blocked: set[str] = set()
        for event in events:
            action = safe_normalize_action(
                event.get("action") or event.get("executed_action") or {}
            )
            tool = action["tool"]
            if tool not in family_wide_failures:
                continue
            requested = action["arguments"].get("state_id")
            if requested is not None and str(requested) != str(active_id):
                continue
            output = event.get("tool_output")
            if not isinstance(output, Mapping):
                continue
            error_code = str(output.get("error_code") or "")
            if error_code in family_wide_failures[tool]:
                blocked.add(tool)
        return blocked

    @staticmethod
    def _signature(action: Any) -> str | None:
        try:
            return action_signature(action)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _equivalent_signatures(cls, action: Any) -> set[str]:
        signature = cls._signature(action)
        if signature is None:
            return set()
        signatures = {signature}
        semantic = cls._semantic_signature_from_text(signature)
        if semantic:
            signatures.add(semantic)
        return signatures

    @staticmethod
    def _semantic_signature_from_text(signature: str) -> str | None:
        try:
            tool, raw_arguments = signature.split(":", 1)
            if tool not in CORRECTION_TOOLS:
                return None
            arguments = json.loads(raw_arguments)
            if not isinstance(arguments, Mapping):
                return None
            semantic_arguments = dict(arguments)
            semantic_arguments.pop("state_id", None)
            return f"{tool}:{json.dumps(semantic_arguments, sort_keys=True, separators=(',', ':'))}"
        except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
            return None

    @staticmethod
    def _signature_applies_to_state(signature: str, active_id: Any) -> bool:
        """Whether a state-free correction identity may be reused here."""
        try:
            tool, raw_arguments = signature.split(":", 1)
            if tool not in CORRECTION_TOOLS:
                return True
            arguments = json.loads(raw_arguments)
            if not isinstance(arguments, Mapping):
                return False
            requested = arguments.get("state_id")
            return requested is None or str(requested) == str(active_id)
        except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
            return False

    @staticmethod
    def _get(state: Any, key: str, default: Any = None) -> Any:
        getter = getattr(state, "get", None)
        return getter(key, default) if callable(getter) else getattr(state, key, default)

    @classmethod
    def _candidate_disposition(cls, next_state: Any, tool_output: Mapping[str, Any]) -> Any:
        disposition = cls._get(next_state, "candidate_disposition") or tool_output.get(
            "candidate_disposition"
        )
        metrics = tool_output.get("tool_metrics")
        if not disposition and isinstance(metrics, Mapping):
            disposition = metrics.get("candidate_disposition")
            assessment = metrics.get("candidate_assessment")
            if not disposition and isinstance(assessment, Mapping):
                disposition = assessment.get("disposition")
            elif not disposition and assessment is not None:
                disposition = getattr(assessment, "disposition", None)
        return getattr(disposition, "value", disposition)

    @staticmethod
    def _progress_class(tool_output: Mapping[str, Any], disposition: Any) -> str | None:
        metrics = tool_output.get("tool_metrics")
        assessment: Any = metrics.get("candidate_assessment") if isinstance(metrics, Mapping) else None
        if isinstance(assessment, Mapping) and assessment.get("progress_class"):
            return str(assessment["progress_class"])
        if assessment is not None and getattr(assessment, "progress_class", None):
            return str(assessment.progress_class)
        if disposition == "REJECT":
            return "no_target_progress"
        if disposition == "ACCEPT_PARTIAL":
            return "target_progress_remaining_faults"
        if disposition == "ACCEPT_FINAL":
            return "resolved"
        if disposition == "INCONCLUSIVE":
            return "mixed_or_weak"
        if isinstance(metrics, Mapping) and metrics.get("progress_class") is not None:
            return str(metrics["progress_class"])
        value = tool_output.get("progress_class")
        return str(value) if value is not None else None
