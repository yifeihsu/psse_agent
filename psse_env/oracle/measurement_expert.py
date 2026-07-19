from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from psse_env.actions import (
    CORRECT_MEASUREMENTS,
    CORRECT_PARAMETERS,
    CORRECT_TOPOLOGY,
    GET_MEASUREMENT_CONTEXT,
    RUN_WLS,
)
from psse_env.oracle.expert_types import (
    ExpertActionProposal,
    dominance_confidence,
    history_action_tool,
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
        history = list(history or [])
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
        unresolved = state_value(state, "unresolved_signatures", [])
        measurement_codes = matching_evidence_codes(
            unresolved,
            "measurement",
            "bad_data",
            "large_residual",
            "meter",
            "residual_outlier",
        )
        branch_codes = matching_evidence_codes(
            unresolved, "wls_branch_multiplier"
        )
        branch_dominant = bool(
            matching_evidence_codes(unresolved, "wls_branch_multiplier_dominant")
        )
        measurement_dominant = bool(
            matching_evidence_codes(measurement_codes, "dominant")
        )
        branch_rejected = self._rejected_branch_hypothesis(
            state, CORRECT_PARAMETERS
        ) and self._rejected_branch_hypothesis(state, CORRECT_TOPOLOGY)
        partial_branch_rows = self._accepted_partial_branch_rows(state)
        colocated_post_branch_indices = self._colocated_post_branch_measurement_indices(
            state,
            history,
            active_id=active_id,
            accepted_branch_rows=partial_branch_rows,
        )
        measurement_signal = (
            bool(measurement_codes)
            and not (
                branch_dominant and not measurement_dominant and not branch_rejected
            )
        )
        partial_measurement = self._accepted_partial_measurement(state)
        # After a partial measurement commit, any still-current branch evidence
        # must be resolved (or explicitly exhausted) before another measurement
        # correction may use that model's residuals.  Otherwise the second
        # correction can mask a real parameter/topology fault while passing WLS.
        measurement_hint_allowed = measurement_signal and not (
            partial_measurement and branch_codes and not measurement_dominant
        )
        proposals: list[ExpertActionProposal] = []

        for action in normalized_hint_actions(
            list(oracle_hints or ()),
            allowed_tools={CORRECT_MEASUREMENTS},
            active_state_id=active_id,
        ):
            if action["tool"] == CORRECT_MEASUREMENTS and not measurement_hint_allowed:
                continue
            if (
                action["tool"] == CORRECT_MEASUREMENTS
                and branch_codes
                and not measurement_dominant
                and self._measurement_target_indices(action)
                & colocated_post_branch_indices
            ):
                # A direct flow residual on an already repaired branch is not
                # independent meter evidence.  Suppress only the action that
                # touches that co-located target; a separate residual elsewhere
                # remains a valid sequential recovery candidate.
                continue
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

        # A measurement correction can always drive the residuals of a wrong
        # model to zero, so it is the one route that masks branch faults
        # instead of being rejected by verification.  While branch-multiplier
        # evidence dominates the solve and no measurement signature is itself
        # dominant, the measurement route stands down and lets the branch
        # families resolve first; the next solve re-evaluates dominance.
        # Escape hatch: only after BOTH branch families have had a hypothesis
        # rejected by verification does branch-multiplier dominance stop
        # justifying suppression — otherwise an episode can stall with every
        # family standing down.  A single wrong-line rejection must not open
        # the measurement route: branch dominance (lambda clearly above the
        # residuals) empirically only occurs on true branch faults, where a
        # measurement correction would mask the fault and still be accepted.
        # A global safety requirement for measurement context is not itself
        # evidence of a measurement fault.  Context routing must be supported
        # by an observable signature (or private teacher supervision that the
        # production collector independently audits against that signature).
        # A committed measurement correction is also observable evidence that
        # this family made progress.  If the episode remains nonterminal, the
        # residual structure must be recomputed on the newly active state
        # before a different correction family is considered.  In particular,
        # never reuse the parent state's context after an ACCEPT_PARTIAL
        # commit: that context is stale even when its old findings still appear
        # in bounded history.
        partial_refresh = bool(
            partial_measurement
            and (not branch_codes or measurement_dominant)
        )
        if (measurement_signal or partial_refresh) and not has_context and active_id:
            evidence = ["measurement_context_missing"]
            if measurement_signal:
                evidence.extend(["measurement_anomaly_evidence", *measurement_codes])
            if partial_refresh:
                evidence.extend(
                    [
                        "measurement_correction_accepted_partial",
                        "fresh_post_commit_context_required",
                    ]
                )
            proposals.append(
                ExpertActionProposal(
                    action={"tool": GET_MEASUREMENT_CONTEXT, "arguments": {"state_id": active_id}},
                    source_expert=self.source_expert,
                    confidence=(
                        1.0
                        if partial_refresh
                        else dominance_confidence(0.87, measurement_codes)
                    ),
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

    @staticmethod
    def _rejected_branch_hypothesis(state: Any, tool: str) -> bool:
        family = {CORRECT_PARAMETERS: "parameter", CORRECT_TOPOLOGY: "topology"}[tool]
        for item in state_value(state, "rejected_hypotheses", []) or []:
            if history_action_tool(item) == tool:
                return True
            if isinstance(item, Mapping):
                item_family = (
                    item.get("family") or item.get("action_family") or item.get("error_family")
                )
                if str(item_family).lower() == family:
                    return True
                if f"{tool}:" in str(item.get("action_signature") or ""):
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
            # The commit itself is observable.  An accepted measurement
            # correction in a state that is not ready for finalization is
            # operationally partial; no privileged disposition is needed.
            if not state_value(state, "no_material_anomaly_remaining", False):
                return True
        return False

    @staticmethod
    def _accepted_partial_branch_rows(state: Any) -> set[int]:
        """Zero-based branch rows committed while an anomaly remains."""
        rows: set[int] = set()
        if state_value(state, "no_material_anomaly_remaining", False):
            return rows
        for item in state_value(state, "accepted_corrections", []) or []:
            tool = history_action_tool(item)
            if tool not in {CORRECT_PARAMETERS, CORRECT_TOPOLOGY}:
                if not isinstance(item, Mapping) or str(
                    item.get("family") or item.get("action_family") or ""
                ).lower() not in {"parameter", "topology"}:
                    continue
            action = item.get("source_action") if isinstance(item, Mapping) else None
            action = action if isinstance(action, Mapping) else item
            arguments = action.get("arguments") if isinstance(action, Mapping) else None
            arguments = arguments if isinstance(arguments, Mapping) else {}
            try:
                if arguments.get("branch_row0") is not None:
                    rows.add(int(arguments["branch_row0"]))
                elif arguments.get("line_index1") is not None:
                    rows.add(int(arguments["line_index1"]) - 1)
                elif arguments.get("line_index") is not None:
                    rows.add(int(arguments["line_index"]) - 1)
            except (TypeError, ValueError):
                continue
        return rows

    @staticmethod
    def _colocated_post_branch_measurement_indices(
        state: Any,
        history: Sequence[Mapping[str, Any]],
        *,
        active_id: Any,
        accepted_branch_rows: set[int],
    ) -> set[int]:
        """Return direct-flow measurement targets co-located with repaired rows."""
        if not accepted_branch_rows or active_id is None:
            return set()
        signature_rows: set[int] = set()
        for signature in state_value(state, "unresolved_signatures", []) or []:
            if "wls_branch_multiplier" not in str(signature):
                continue
            match = re.search(r"(?:^|\s)line=(\d+)(?:\s|$)", str(signature))
            if match:
                signature_rows.add(int(match.group(1)) - 1)
        candidate_rows = accepted_branch_rows & signature_rows
        if not candidate_rows:
            return set()

        findings: Sequence[Any] = ()
        for event in reversed(history):
            if not isinstance(event, Mapping):
                continue
            action = event.get("action") or event.get("executed_action") or {}
            if history_action_tool(action) != GET_MEASUREMENT_CONTEXT:
                continue
            arguments = action.get("arguments") if isinstance(action, Mapping) else None
            arguments = arguments if isinstance(arguments, Mapping) else {}
            requested = arguments.get("state_id")
            if requested is not None and str(requested) != str(active_id):
                continue
            output = event.get("tool_output")
            if not isinstance(output, Mapping) or output.get("execution_status") != "success":
                continue
            metrics = output.get("tool_metrics")
            if not isinstance(metrics, Mapping):
                continue
            if str(metrics.get("state_id")) != str(active_id):
                continue
            raw_findings = metrics.get("measurement_findings")
            if isinstance(raw_findings, (list, tuple)):
                findings = raw_findings
                break
        colocated: set[int] = set()
        for item in findings:
            if not isinstance(item, Mapping):
                continue
            if str(item.get("channel") or "") not in {"Pf", "Qf", "Pt", "Qt"}:
                continue
            index0 = item.get("index0")
            channel_offset = item.get("channel_offset")
            if (
                not isinstance(index0, int)
                or isinstance(index0, bool)
                or index0 < 0
                or not isinstance(channel_offset, int)
                or isinstance(channel_offset, bool)
                or channel_offset < 0
            ):
                continue
            if channel_offset in candidate_rows:
                colocated.add(index0)
        return colocated

    @staticmethod
    def _measurement_target_indices(action: Mapping[str, Any]) -> set[int]:
        arguments = action.get("arguments")
        arguments = arguments if isinstance(arguments, Mapping) else {}
        group = arguments.get("suspect_group")
        if isinstance(group, (list, tuple)):
            values = group
        else:
            updates = arguments.get("measurement_updates")
            values = updates.keys() if isinstance(updates, Mapping) else ()
        indices: set[int] = set()
        for value in values:
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                indices.add(value)
        return indices
