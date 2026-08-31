from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from psse_env.actions import (
    CORRECT_MEASUREMENTS,
    CORRECT_PARAMETERS,
    CORRECT_TOPOLOGY,
    GET_MEASUREMENT_CONTEXT,
    GET_PARAMETER_CONTEXT,
    GET_TOPOLOGY_CONTEXT,
    RUN_WLS,
    action_signature,
    safe_normalize_action,
)
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
from psse_env.oracle.measurement_recovery_evidence import (
    accepted_measurement_indices,
    eligible_joint_measurement_targets,
    measurement_target_indices,
    verified_terminal_measurement_closure_action,
)


class MeasurementExpert:
    """Propose measurement-family diagnosis and correction actions."""

    source_expert = "measurement_expert"
    _TOOLS = {GET_MEASUREMENT_CONTEXT, CORRECT_MEASUREMENTS}
    _COUPLED_REFINEMENT_MAX_ANOMALY_RATIO = 1.10
    _COUPLED_REFINEMENT_MIN_REMAINING_BUDGET = 8
    _REJECTED_TARGET_JOINT_MIN_REMAINING_BUDGET = 4

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
        homogeneous_multi_residual_signal = self._homogeneous_residual_channel(
            measurement_codes
        )
        branch_routes_exhausted = self._branch_recovery_routes_exhausted(
            state, history, active_id=active_id
        )
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
                branch_dominant
                and not measurement_dominant
                and not branch_routes_exhausted
            )
        )
        partial_measurement = self._accepted_partial_measurement(state)
        accepted_measurement_indices = self._accepted_measurement_indices(state)
        accepted_target_refinement = self._has_fresh_accepted_target_refinement(
            state, history, active_id=active_id
        )
        terminal_closure_action = self._fresh_terminal_closure_action(
            state,
            active_id=active_id,
            accepted_targets=accepted_measurement_indices,
        )
        flagged_residual_indices = self._residual_outlier_indices(unresolved)
        # After a partial measurement commit, any still-current branch evidence
        # must be resolved (or explicitly exhausted) before another measurement
        # correction may use that model's residuals.  Otherwise the second
        # correction can mask a real parameter/topology fault while passing WLS.
        measurement_hint_allowed = measurement_signal and not (
            partial_measurement
            and branch_codes
            and not branch_routes_exhausted
        )
        proposals: list[ExpertActionProposal] = []
        normalized_measurement_hints = normalized_hint_actions(
            list(oracle_hints or ()),
            allowed_tools={CORRECT_MEASUREMENTS},
            active_state_id=active_id,
        )

        for action in normalized_measurement_hints:
            if action["tool"] == CORRECT_MEASUREMENTS:
                target_indices = self._measurement_target_indices(action)
                is_observable_refinement = bool(
                    accepted_target_refinement
                    and target_indices
                    and target_indices == accepted_measurement_indices
                )
                is_observable_terminal_closure = bool(
                    branch_routes_exhausted
                    and terminal_closure_action is not None
                    and action_signature(action)
                    == action_signature(terminal_closure_action)
                    and accepted_measurement_indices
                    and accepted_measurement_indices < target_indices
                    and len(target_indices - accepted_measurement_indices) == 1
                    # The one new closure member must itself carry a current
                    # residual-outlier signature.  A provider closure group can
                    # otherwise fold in an unflagged healthy meter purely
                    # because editing it resolves the global statistic -- a
                    # masking commit (measured on held-out root
                    # r0_680cc8de358a, healthy index 64).
                    and target_indices - accepted_measurement_indices
                    <= flagged_residual_indices
                )
                # Provider-declared multi-target fallbacks are executable
                # context contracts, not immediate labels.  They become an
                # expert proposal only after the same-state rejected-candidate
                # proof below establishes the exact bounded union.  Immediate
                # grouped routes are limited to a provider-flagged refinement
                # of already accepted targets or the separately preverified
                # terminal closure above.
                if len(target_indices) >= 2 and not (
                    is_observable_refinement or is_observable_terminal_closure
                ):
                    continue
                if (
                    not measurement_hint_allowed
                    and not (
                        is_observable_refinement
                        or is_observable_terminal_closure
                    )
                ):
                    continue
            if (
                action["tool"] == CORRECT_MEASUREMENTS
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
                    confidence=(
                        0.999
                        if is_observable_terminal_closure
                        else 0.995
                        if is_observable_refinement
                        else 0.98
                    ),
                    evidence_codes=(
                        [
                            "provider_verified_terminal_measurement_closure",
                            "singleton_target_accepted",
                            "branch_routes_exhausted",
                            "same_state_measurement_context",
                        ]
                        if is_observable_terminal_closure
                        else
                        [
                            "observable_accepted_target_refinement",
                            "same_state_measurement_context",
                        ]
                        if is_observable_refinement
                        else ["oracle_action_hint", "measurement_family"]
                    ),
                    admissible=(
                        action["tool"] != CORRECT_MEASUREMENTS
                        or not requires_context
                        or has_context
                    ),
                    estimated_immediate_risk=0.12 if action["tool"] == CORRECT_MEASUREMENTS else 0.02,
                )
            )

        # Two individually verified singleton corrections can each fix their
        # local residual while missing the 20% partial-progress floor because
        # their errors are coupled.  Once both candidates have been rolled
        # back, retry only those same provider-supported targets as a joint
        # transaction, jointly re-estimating any previously accepted targets.
        # This does not admit an unsupported meter: each new target must have
        # target-local success, positive global progress, complete physical
        # safety evidence, and still-current same-state context support, while
        # every other target was already transactionally accepted.  The normal
        # candidate-quality gate and offline exact-target audit remain
        # authoritative for the joint candidate.
        joint_rejected_targets = self._joint_locally_fixed_rejected_targets(
            state,
            history,
            active_id=active_id,
            supported_actions=normalized_measurement_hints,
            accepted_indices=accepted_measurement_indices,
        )
        joint_retry_targets = sorted(
            accepted_measurement_indices | set(joint_rejected_targets)
        )
        if (
            measurement_hint_allowed
            and has_context
            and joint_rejected_targets
            and not (set(joint_retry_targets) & colocated_post_branch_indices)
        ):
            proposals.append(
                ExpertActionProposal(
                    action={
                        "tool": CORRECT_MEASUREMENTS,
                        "arguments": {
                            "state_id": active_id,
                            "suspect_group": joint_retry_targets,
                        },
                    },
                    source_expert=self.source_expert,
                    confidence=0.997,
                    evidence_codes=[
                        "locally_fixed_singletons_rejected",
                        "physical_safety_verified_per_target",
                        "previously_accepted_targets_jointly_reestimated",
                        "same_state_joint_retry",
                    ],
                    admissible=True,
                    estimated_immediate_risk=0.10,
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
        near_threshold_refinement = self._near_threshold_refinement_needed(
            state, accepted_measurement_indices
        )
        # Every accepted partial measurement invalidates the parent model's
        # residual and branch inventories.  The fresh measurement context also
        # carries same-state parameter/topology screening, so refresh it first
        # even when branch evidence currently dominates; a surviving branch
        # correction will then be routed from that bundled contract.
        partial_refresh = bool(partial_measurement)
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
            if near_threshold_refinement:
                evidence.append("near_threshold_accepted_target_refinement")
            proposals.append(
                ExpertActionProposal(
                    action={"tool": GET_MEASUREMENT_CONTEXT, "arguments": {"state_id": active_id}},
                    source_expert=self.source_expert,
                    confidence=(
                        1.0
                        if partial_refresh
                        else max(
                            dominance_confidence(0.87, measurement_codes),
                            0.89 if homogeneous_multi_residual_signal else 0.0,
                        )
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
    def _homogeneous_residual_channel(codes: Sequence[Any]) -> bool:
        """Return true for at least three residual findings on one channel."""
        channels: list[str] = []
        for code in codes:
            text = str(code)
            marker = "channel="
            if marker not in text:
                continue
            channel = text.rsplit(marker, 1)[1].split(maxsplit=1)[0].strip()
            if channel:
                channels.append(channel)
        return len(channels) >= 3 and len(set(channels)) == 1

    @classmethod
    def _near_threshold_refinement_needed(
        cls, state: Any, accepted_indices: set[int]
    ) -> bool:
        if len(accepted_indices) < 2:
            return False
        try:
            score = float(state_value(state, "remaining_anomaly_score", 0.0))
            budget = int(state_value(state, "remaining_budget", 0))
        except (TypeError, ValueError, OverflowError):
            return False
        return bool(
            1.0 <= score <= cls._COUPLED_REFINEMENT_MAX_ANOMALY_RATIO
            and budget >= cls._COUPLED_REFINEMENT_MIN_REMAINING_BUDGET
        )

    @classmethod
    def _joint_locally_fixed_rejected_targets(
        cls,
        state: Any,
        history: Sequence[Mapping[str, Any]],
        *,
        active_id: Any,
        supported_actions: Sequence[Mapping[str, Any]],
        accepted_indices: set[int],
    ) -> list[int]:
        """Return an evidence-closed rejected target set for one bounded retry."""
        return eligible_joint_measurement_targets(
            state,
            history,
            active_id=active_id,
            supported_actions=supported_actions,
            accepted_indices=accepted_indices,
            min_remaining_budget=cls._REJECTED_TARGET_JOINT_MIN_REMAINING_BUDGET,
        )

    @staticmethod
    def _rejected_branch_hypothesis(state: Any, tool: str) -> bool:
        family = {CORRECT_PARAMETERS: "parameter", CORRECT_TOPOLOGY: "topology"}[tool]
        active_id = state_value(state, "active_state_id")
        for item in state_value(state, "rejected_hypotheses", []) or []:
            if not recovery_record_applies_to_state(item, active_id):
                continue
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

    @classmethod
    def _branch_recovery_routes_exhausted(
        cls,
        state: Any,
        history: Sequence[Mapping[str, Any]],
        *,
        active_id: Any,
    ) -> bool:
        """Return true only after both observable branch routes are closed.

        A partial meter repair deliberately stands down while a current WLS
        solve still carries branch evidence: correcting another residual can
        otherwise mask a real parameter or topology fault.  A branch context
        that explicitly returns no candidates is nevertheless just as closed
        as a context whose complete candidate inventory has been tried and
        rejected.  Treating the former as perpetually open stranded pure
        multi-measurement episodes after harmless WLS branch cross-signals.

        This predicate consumes only state-bound provider inventories and
        rejected-candidate records already visible to the policy.  One
        rejection is not enough when the same context exposes another branch
        candidate.
        """

        if active_id is None:
            return False
        routes = (
            ("parameter", GET_PARAMETER_CONTEXT, CORRECT_PARAMETERS),
            ("topology", GET_TOPOLOGY_CONTEXT, CORRECT_TOPOLOGY),
        )
        return all(
            cls._branch_recovery_route_exhausted(
                state,
                history,
                active_id=active_id,
                family=family,
                context_tool=context_tool,
                correction_tool=correction_tool,
            )
            for family, context_tool, correction_tool in routes
        )

    @classmethod
    def _branch_recovery_route_exhausted(
        cls,
        state: Any,
        history: Sequence[Mapping[str, Any]],
        *,
        active_id: Any,
        family: str,
        context_tool: str,
        correction_tool: str,
    ) -> bool:
        if not (
            state_value(state, f"has_fresh_{family}_context", False)
            and str(state_value(state, f"{family}_context_state_id") or "")
            == str(active_id)
        ):
            return False

        supported: Sequence[Any] | None = None
        route_status_present = False
        route_status: str | None = None
        fresh = state_value(state, "fresh_context_evidence", {})
        if isinstance(fresh, Mapping):
            evidence = fresh.get(family)
            if (
                isinstance(evidence, Mapping)
                and str(evidence.get("state_id") or "") == str(active_id)
                and isinstance(evidence.get("supported_corrections"), (list, tuple))
            ):
                supported = evidence["supported_corrections"]
                if "route_status" in evidence:
                    route_status_present = True
                    raw_status = evidence.get("route_status")
                    route_status = (
                        str(raw_status) if raw_status is not None else None
                    )

        # Compact legacy observations may have fresh flags but omit the
        # durable evidence mapping.  Recover only the latest successful,
        # same-state provider contract from bounded observable history.
        if supported is None:
            for event in reversed(history):
                if not isinstance(event, Mapping):
                    continue
                action = safe_normalize_action(
                    event.get("action") or event.get("executed_action") or {}
                )
                if action["tool"] != context_tool:
                    continue
                requested = action["arguments"].get("state_id")
                if requested is not None and str(requested) != str(active_id):
                    continue
                output = event.get("tool_output")
                if (
                    not isinstance(output, Mapping)
                    or output.get("execution_status") != "success"
                ):
                    return False
                metrics = output.get("tool_metrics")
                if not isinstance(metrics, Mapping):
                    return False
                if str(metrics.get("state_id") or "") != str(active_id):
                    return False
                inventory = metrics.get("supported_corrections")
                if not isinstance(inventory, (list, tuple)):
                    return False
                supported = inventory
                if "route_status" in metrics:
                    route_status_present = True
                    raw_status = metrics.get("route_status")
                    route_status = (
                        str(raw_status) if raw_status is not None else None
                    )
                break
        if supported is None:
            return False

        supported_signatures: set[str] = set()
        for raw_action in supported:
            if not isinstance(raw_action, Mapping):
                return False
            normalized = safe_normalize_action(raw_action)
            if normalized["tool"] != correction_tool:
                return False
            requested = normalized["arguments"].get("state_id")
            if requested is None or str(requested) != str(active_id):
                return False
            try:
                supported_signatures.add(action_signature(normalized))
            except (TypeError, ValueError):
                return False

        # An empty inventory closes the route when the provider explicitly
        # completed its screen with nothing executable — a negative diagnostic
        # or an inconclusive screen that advertises no correction.  Neither
        # offers an action the expert could try, so leaving the route "open"
        # can never be resolved by acting on it; on multi-measurement states
        # the branch cross-signals it waits on are themselves caused by the
        # remaining meter errors, deadlocking recovery.  State binding still
        # forces a fresh same-state re-screen after every commit, so a branch
        # fault that becomes observably dominant later reopens the route.
        # Only a legacy empty contract without a route_status stays open.
        if not supported_signatures:
            return route_status in ("complete_negative", "unavailable_or_inconclusive")
        if route_status_present and route_status != "actionable":
            return False

        rejected_signatures: set[str] = set()
        for item in state_value(state, "rejected_hypotheses", []) or []:
            if not recovery_record_applies_to_state(item, active_id):
                continue
            action = item.get("source_action") or item.get("action") or {}
            if history_action_tool(action) != correction_tool:
                continue
            try:
                rejected_signatures.add(
                    action_signature(safe_normalize_action(action))
                )
            except (TypeError, ValueError):
                continue
        return supported_signatures <= rejected_signatures

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

    @classmethod
    def _accepted_measurement_indices(cls, state: Any) -> set[int]:
        return accepted_measurement_indices(state)

    @staticmethod
    def _has_fresh_accepted_target_refinement(
        state: Any,
        history: Sequence[Mapping[str, Any]],
        *,
        active_id: Any,
    ) -> bool:
        """Recognize the provider's same-state observable joint-refinement gate."""
        if active_id is None:
            return False
        fresh_context_evidence = state_value(state, "fresh_context_evidence", {})
        if isinstance(fresh_context_evidence, Mapping):
            measurement_evidence = fresh_context_evidence.get("measurement")
            if (
                isinstance(measurement_evidence, Mapping)
                and str(measurement_evidence.get("state_id") or "")
                == str(active_id)
                and measurement_evidence.get("accepted_target_refinement") is True
            ):
                return True
        for event in reversed(history):
            if not isinstance(event, Mapping):
                continue
            action = event.get("action") or event.get("executed_action") or {}
            if history_action_tool(action) != GET_MEASUREMENT_CONTEXT:
                continue
            arguments = action.get("arguments") if isinstance(action, Mapping) else None
            arguments = arguments if isinstance(arguments, Mapping) else {}
            if arguments.get("state_id") is not None and str(
                arguments["state_id"]
            ) != str(active_id):
                continue
            output = event.get("tool_output")
            if not isinstance(output, Mapping) or output.get("execution_status") != "success":
                return False
            metrics = output.get("tool_metrics")
            if not isinstance(metrics, Mapping):
                return False
            if str(metrics.get("state_id")) != str(active_id):
                return False
            return metrics.get("accepted_target_refinement") is True
        return False

    _RESIDUAL_OUTLIER_INDEX_RE = re.compile(
        r"^wls_residual_outlier\S*\s+index=(\d+)\b"
    )

    @classmethod
    def _residual_outlier_indices(cls, unresolved: Any) -> set[int]:
        """Measurement indices carrying a current residual-outlier signature."""
        indices: set[int] = set()
        for raw in unresolved or []:
            match = cls._RESIDUAL_OUTLIER_INDEX_RE.match(str(raw))
            if match:
                indices.add(int(match.group(1)))
        return indices

    @staticmethod
    def _fresh_terminal_closure_action(
        state: Any,
        *,
        active_id: Any,
        accepted_targets: set[int],
    ) -> dict[str, Any] | None:
        """Read one atomically validated same-state terminal meter action."""

        if active_id is None:
            return None
        fresh = state_value(state, "fresh_context_evidence", {})
        if isinstance(fresh, Mapping):
            evidence = fresh.get("measurement")
            if isinstance(evidence, Mapping):
                return verified_terminal_measurement_closure_action(
                    evidence,
                    active_id=active_id,
                    accepted_targets=accepted_targets,
                )
        return None

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
        # Once a branch repair has been accepted, direct-flow residuals on
        # that same row are not independent meter evidence.  They often arise
        # from bounded parameter-estimation error even after the branch
        # multiplier signature itself disappears, so requiring that stale
        # signature let a measurement correction mask the repaired branch.
        candidate_rows = accepted_branch_rows

        findings: Sequence[Any] = ()
        fresh_context_evidence = state_value(state, "fresh_context_evidence", {})
        if isinstance(fresh_context_evidence, Mapping):
            measurement_evidence = fresh_context_evidence.get("measurement")
            if (
                isinstance(measurement_evidence, Mapping)
                and str(measurement_evidence.get("state_id") or "")
                == str(active_id)
            ):
                raw_findings = measurement_evidence.get("measurement_findings")
                if isinstance(raw_findings, (list, tuple)):
                    findings = raw_findings
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
        return measurement_target_indices(action)
