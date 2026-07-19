from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from psse_env.actions import CORRECTION_TOOLS, RUN_WLS, action_signature, safe_normalize_action
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

        recovery = self.recovery_expert.repair_actions(policy, context.history)
        if recovery:
            return self._rank_and_filter(recovery, policy, seen_signatures=seen_signatures, mandatory=True)

        verification = self.termination_expert.verification_actions(policy)
        if verification:
            return self._rank_and_filter(verification, policy, seen_signatures=seen_signatures, mandatory=True)

        disposition = self.termination_expert.candidate_disposition_actions(policy)
        if disposition:
            return self._rank_and_filter(disposition, policy, seen_signatures=seen_signatures, mandatory=True)

        terminal = self.termination_expert.propose(policy, context.history)
        if terminal:
            return self._rank_and_filter(terminal, policy, seen_signatures=seen_signatures, mandatory=True)

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
                mandatory=True,
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
        return self._rank_and_filter(
            proposals,
            policy,
            seen_signatures=seen_signatures,
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
            hidden = state.hidden_truth if isinstance(state.hidden_truth, Mapping) else {}
            if hidden.get("true_harmonic_errors"):
                fault_families.add("harmonic")
            if hidden.get("true_hif_errors"):
                fault_families.add("hif")
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
                hidden = state.get("hidden_truth")
                hidden = hidden if isinstance(hidden, Mapping) else {}
                if state.get("true_harmonic_errors") or hidden.get("true_harmonic_errors"):
                    fault_families.add("harmonic")
                if state.get("true_hif_errors") or hidden.get("true_hif_errors"):
                    fault_families.add("hif")
        else:
            raise TypeError(f"state must be OracleState, PolicyObservation, or mapping, got {type(state).__name__}")

        # Older callers exposed only the two verified/unverified booleans.  A
        # candidate in either state is necessarily open for process-gate
        # purposes, so restore that implied lifecycle flag on our private copy.
        if isinstance(policy, Mapping):
            policy = dict(policy)
            if policy.get("has_unverified_candidate") or policy.get("has_verified_candidate"):
                policy["has_open_candidate"] = True
            observable_corrections = self._observable_supported_corrections(policy)
            if observable_corrections:
                # Once an observable context provider supplies exact bounded
                # corrections, it is the sole source of correction targets.
                # Private scenario hints remain available for synthetic flows
                # that have no observable provider contract.
                hints = observable_corrections

        if history is None:
            raw_history = self._get(policy, "history_window", []) or []
            history_items = [dict(item) for item in raw_history if isinstance(item, Mapping)]
        else:
            history_items = [dict(item) for item in history if isinstance(item, Mapping)]
        return _ExpertContext(
            policy_state=policy,
            history=history_items,
            oracle_hints=hints,
            oracle_fault_families=frozenset(fault_families),
        )

    @staticmethod
    def _observable_supported_corrections(
        policy: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        output = policy.get("last_tool_output")
        if not isinstance(output, Mapping):
            return []
        metrics = output.get("tool_metrics")
        if not isinstance(metrics, Mapping):
            metrics = output.get("observable_metrics")
        if not isinstance(metrics, Mapping):
            return []
        raw_actions = metrics.get("supported_corrections")
        if not isinstance(raw_actions, (list, tuple)):
            return []
        actions: list[dict[str, Any]] = []
        for raw_action in raw_actions:
            if not isinstance(raw_action, Mapping):
                continue
            normalized = safe_normalize_action(raw_action)
            if normalized["tool"] in CORRECTION_TOOLS:
                actions.append(normalized)
        return actions

    def _rank_and_filter(
        self,
        proposals: Sequence[ExpertActionProposal],
        policy: PolicyObservation | Mapping[str, Any],
        *,
        seen_signatures: set[str],
        mandatory: bool,
    ) -> list[ExpertActionProposal]:
        assessed: list[tuple[int, ExpertActionProposal, str]] = []
        for index, proposal in enumerate(proposals):
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

    def _seen_action_signatures(
        self,
        policy: PolicyObservation | Mapping[str, Any],
        history: Sequence[Mapping[str, Any]],
    ) -> set[str]:
        signatures = {
            str(signature)
            for signature in (self._get(policy, "tried_action_signatures", []) or [])
            if signature
        }
        for signature in list(signatures):
            semantic = self._semantic_signature_from_text(signature)
            if semantic:
                signatures.add(semantic)
        for field in ("rejected_hypotheses", "accepted_corrections"):
            for remembered in self._get(policy, field, []) or []:
                if not isinstance(remembered, Mapping):
                    continue
                if remembered.get("action_signature"):
                    text_signature = str(remembered["action_signature"])
                    signatures.add(text_signature)
                    semantic = self._semantic_signature_from_text(text_signature)
                    if semantic:
                        signatures.add(semantic)
                source_action = remembered.get("source_action")
                if source_action:
                    signatures.update(self._equivalent_signatures(source_action))
        for item in history:
            action = item.get("action") or item.get("executed_action")
            if action:
                signatures.update(self._equivalent_signatures(action))
        return signatures

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
