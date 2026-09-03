from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping

from psse_env.actions import CORRECT_MEASUREMENTS, CORRECT_PARAMETERS, CORRECT_TOPOLOGY
from psse_env.private_target_matching import (
    action_targets_private_fault,
    canonical_branch_target,
    correction_family,
    correction_matches_private_fault,
    matched_private_fault_indices,
)


class CandidateDisposition(str, Enum):
    ACCEPT_FINAL = "ACCEPT_FINAL"
    ACCEPT_PARTIAL = "ACCEPT_PARTIAL"
    REJECT = "REJECT"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass(frozen=True)
class CandidateAssessment:
    disposition: CandidateDisposition
    progress_class: str
    target_progress: float | None = None
    global_progress: float | None = None
    collateral_damage: bool = False
    remaining_true_fault_count: int | None = None
    remaining_suspect_count: int | None = None
    belief_update: dict[str, str] = field(default_factory=dict)
    unresolved_signatures: list[str] = field(default_factory=list)
    rationale_codes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["disposition"] = self.disposition.value
        return payload

    @property
    def remaining_fault_count(self) -> int | None:
        """Deprecated compatibility view; never serialized as observable evidence."""
        return (
            self.remaining_true_fault_count
            if self.remaining_true_fault_count is not None
            else self.remaining_suspect_count
        )


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class CandidateQualityOracle:
    """Assess a verified branch using privileged truth or observable evidence.

    Hidden truth is optional.  When it is absent, unknown quantities remain
    unknown; they are never silently interpreted as zero remaining faults.
    """

    def __init__(
        self,
        *,
        min_target_progress: float = 0.05,
        min_partial_global_progress: float = 0.20,
        min_branch_partial_global_progress: float = 0.30,
        min_branch_target_progress: float = 0.80,
        min_branch_global_progress: float = 0.50,
        max_branch_target_threshold_ratio: float = 1.25,
        min_topology_structural_global_progress: float = 0.95,
        max_new_violations: int = 0,
        coupled_measurement_partial: bool = True,
        accepted_channel_measurement_partial: bool = True,
        mode: str = "auto",
        case_differ: Any = None,
        case_loader: Any = None,
    ) -> None:
        self.min_target_progress = float(min_target_progress)
        self.min_partial_global_progress = float(min_partial_global_progress)
        self.min_branch_partial_global_progress = float(
            min_branch_partial_global_progress
        )
        self.min_branch_target_progress = float(min_branch_target_progress)
        self.min_branch_global_progress = float(min_branch_global_progress)
        self.max_branch_target_threshold_ratio = float(
            max_branch_target_threshold_ratio
        )
        self.min_topology_structural_global_progress = float(
            min_topology_structural_global_progress
        )
        self.max_new_violations = int(max_new_violations)
        # V2-B: allow a singleton measurement correction inside a coherent
        # same-channel residual cluster to pass a halved global-progress
        # floor.  With several interacting meter errors, one correct fix
        # cannot clear the shared statistic alone; requiring the full floor
        # rejects physically correct repairs (short-horizon credit
        # assignment).  The cluster and channel evidence come from the
        # observable unresolved-signature ledger, never hidden truth.
        self.coupled_measurement_partial = bool(coupled_measurement_partial)
        # V2-D: a lone remaining error has no same-channel companion left, so
        # the cluster route above cannot apply to the last error of a chain.
        # The accepted-channel route lets a rank-1, currently-flagged
        # singleton whose channel matches >=2 already-accepted measurement
        # corrections use the same halved floor, with branch routes closed.
        # Kept independently ablatable from the cluster route.
        self.accepted_channel_measurement_partial = bool(
            accepted_channel_measurement_partial
        )
        if mode not in {"auto", "synthetic", "deployment"}:
            raise ValueError("mode must be 'auto', 'synthetic', or 'deployment'.")
        self.mode = mode
        # Path-valued cases cannot be structurally compared in memory.  A
        # deployment integration may inject a differ returning
        # {"comparable": bool, "changed_branch_rows": {row0: [column, ...]}}
        # for two case paths; without one, path-case corrections are treated
        # as unverifiable collateral damage (fail closed).
        self.case_differ = case_differ
        # Private truth retirement additionally needs the target branch values,
        # not only a structural diff.  The loader remains oracle-only and is
        # never reachable from PolicyObservation.
        self.case_loader = case_loader

    def label_candidate(
        self,
        *,
        parent_state: Mapping[str, Any],
        source_action: Mapping[str, Any],
        candidate_state: Mapping[str, Any],
        verification_output: Mapping[str, Any],
        hidden_truth: Mapping[str, Any] | None = None,
    ) -> CandidateAssessment:
        """Return the authoritative four-way candidate assessment."""
        parent = dict(parent_state)
        action = dict(source_action)
        candidate = dict(candidate_state)
        verification = dict(verification_output)
        if self.mode == "synthetic" and hidden_truth is None:
            raise ValueError("synthetic candidate assessment requires hidden_truth.")
        truth = {} if self.mode == "deployment" else dict(hidden_truth or {})
        synthetic_truth = self.mode != "deployment" and bool(truth)
        candidate_meta = candidate.get("metadata") if isinstance(candidate.get("metadata"), Mapping) else {}

        target_progress = _optional_float(verification.get("target_progress"))
        global_progress = _optional_float(verification.get("global_progress"))
        target_metric_value = _optional_float(verification.get("target_metric_value"))
        target_metric_threshold = _optional_float(
            verification.get("target_metric_threshold")
        )
        target_fixed = self._target_fixed(
            action, verification, truth, candidate_meta, parent, candidate
        )
        action_family = self._action_family(action)
        partial_global_progress_floor = (
            self.min_branch_partial_global_progress
            if action_family in {"parameter", "topology"}
            else self.min_partial_global_progress
        )
        branch_target_materially_improved = bool(
            not synthetic_truth
            and target_fixed is False
            and action_family in {"parameter", "topology"}
            and target_progress is not None
            and target_progress >= self.min_branch_target_progress
            and target_metric_value is not None
            and target_metric_threshold is not None
            and target_metric_threshold > 0.0
            and target_metric_value
            <= self.max_branch_target_threshold_ratio * target_metric_threshold
            and global_progress is not None
            and global_progress >= self.min_branch_global_progress
        )
        topology_multiplier = _optional_float(
            verification.get("topology_target_branch_multiplier")
        )
        topology_multiplier_threshold = _optional_float(
            verification.get("topology_target_branch_multiplier_threshold")
        )
        topology_structural_target_ambiguous = bool(
            not synthetic_truth
            and action_family == "topology"
            and target_fixed is True
            and verification.get("topology_target_status_matches_requested") is True
            and topology_multiplier is not None
            and topology_multiplier_threshold is not None
            and topology_multiplier_threshold > 0.0
            and topology_multiplier
            > self.max_branch_target_threshold_ratio * topology_multiplier_threshold
            and (
                global_progress is None
                or global_progress < self.min_topology_structural_global_progress
            )
        )
        remaining_true_faults = self._remaining_true_faults(
            action, target_fixed, verification, truth, candidate_meta, parent, candidate
        )
        remaining_suspects = self._remaining_suspects(verification)
        global_resolved = self._global_resolved(verification)
        if global_resolved is None and synthetic_truth:
            global_resolved = (
                remaining_true_faults == 0
                if remaining_true_faults is not None
                else None
            )
        # A parameter repair whose candidate solve passes the global chi-square
        # test, leaves no other meter or branch suspect, and whose only
        # remaining threshold crossing is the corrected branch's own
        # multiplier sitting marginally above the per-branch cutoff.  The
        # chi-square test is the authoritative goodness-of-fit statistic; one
        # normalized multiplier among every branch exceeding a 3-sigma cutoff
        # by a few percent has a material false-rejection rate on a clean
        # solve, and refusing such a repair sends the expert on to healthy
        # lines that fit the same data worse (frozen root r0_8c0755fce51c
        # after the 2026-09-03 Jacobian repair: true line 7 rejected at 3.13
        # with the solve at 79 against 130, then healthy lines 4 and 3
        # committed).  The tolerance band is the one already used for
        # marginal partial branch progress.
        solve_resolved = self._solve_resolved(verification)
        branch_target_marginal_on_clean_solve = bool(
            not synthetic_truth
            and target_fixed is False
            and action_family == "parameter"
            and solve_resolved is True
            and self._remaining_suspects(verification) == 0
            and target_metric_value is not None
            and target_metric_threshold is not None
            and target_metric_threshold > 0.0
            and target_metric_value
            <= self.max_branch_target_threshold_ratio * target_metric_threshold
        )
        collateral_damage = self._collateral_damage(
            action, verification, truth, candidate_meta, parent, candidate
        )
        physical_ok = self._physical_ok(verification)
        unresolved = self._unresolved_signatures(verification, truth)
        rationale: list[str] = []

        if collateral_damage:
            disposition = CandidateDisposition.REJECT
            progress_class = "healthy_component_corruption"
            rationale.append("collateral_damage_detected")
        elif physical_ok is False:
            disposition = CandidateDisposition.REJECT
            progress_class = "physical_regression"
            rationale.append("physical_constraints_failed")
        elif physical_ok is None and not synthetic_truth:
            disposition = CandidateDisposition.INCONCLUSIVE
            progress_class = "physical_status_unknown"
            rationale.append("physical_constraint_evidence_missing")
        elif (
            not synthetic_truth
            and action_family == "measurement"
            and verification.get("sequential_cross_family_measurement") is True
            and verification.get("measurement_evidence_dominant") is not True
            and verification.get("measurement_target_branch_colocated") is None
        ):
            disposition = CandidateDisposition.INCONCLUSIVE
            progress_class = "measurement_target_locality_unknown"
            rationale.append("measurement_target_locality_missing")
        elif (
            not synthetic_truth
            and action_family == "measurement"
            and verification.get("sequential_cross_family_measurement") is True
            and verification.get("measurement_evidence_dominant") is not True
            and verification.get("measurement_target_branch_colocated") is True
        ):
            disposition = CandidateDisposition.REJECT
            progress_class = "ambiguous_cross_family_measurement_cleanup"
            rationale.append("independent_measurement_evidence_missing")
        elif topology_structural_target_ambiguous:
            # Exact status equality proves that the requested topology edit
            # landed, but it does not by itself identify the correct outage.
            # When the same row still carries a non-marginal multiplier, only
            # an exceptional global reduction supports retaining it as a
            # sequential partial repair.  This rejects healthy-line outages
            # that merely absorb part of a mixed-error residual pattern.
            disposition = CandidateDisposition.REJECT
            progress_class = "ambiguous_structural_topology_target"
            rationale.extend(
                [
                    "topology_status_applied",
                    "topology_multiplier_not_cleared",
                    "topology_global_progress_below_structural_threshold",
                ]
            )
        elif (
            target_fixed is True
            and global_resolved is False
            and global_progress is not None
            and global_progress < partial_global_progress_floor
        ):
            halved_floor_measurement_singleton = bool(
                not synthetic_truth
                and action_family == "measurement"
                and self._singleton_measurement_action(action)
                and global_progress >= 0.5 * partial_global_progress_floor
            )
            cluster_size = _optional_float(
                verification.get("measurement_target_cluster_size")
            )
            accepted_count = _optional_float(
                verification.get("accepted_measurement_target_count")
            )
            coupled_cluster_route = bool(
                self.coupled_measurement_partial
                and halved_floor_measurement_singleton
                and cluster_size is not None
                and cluster_size >= 2
            )
            accepted_channel_route = bool(
                self.accepted_channel_measurement_partial
                and halved_floor_measurement_singleton
                and verification.get("measurement_target_channel") is not None
                and accepted_count is not None
                and accepted_count >= 2
                and verification.get("accepted_measurement_shared_channel")
                is not None
                and str(verification["accepted_measurement_shared_channel"])
                == str(verification["measurement_target_channel"])
                and verification.get("measurement_target_rank_one") is True
                and verification.get("measurement_branch_routes_closed") is True
            )
            if coupled_cluster_route:
                disposition = CandidateDisposition.ACCEPT_PARTIAL
                progress_class = "coupled_measurement_partial"
                rationale.extend(
                    [
                        "target_fixed",
                        "coupled_same_channel_residual_cluster",
                        "coupled_partial_floor_half",
                        "global_anomaly_remains",
                    ]
                )
            elif accepted_channel_route:
                disposition = CandidateDisposition.ACCEPT_PARTIAL
                progress_class = "accepted_channel_measurement_partial"
                rationale.extend(
                    [
                        "target_fixed",
                        "accepted_channel_coherent",
                        "residual_rank_one",
                        "branch_routes_closed",
                        "coupled_partial_floor_half",
                        "global_anomaly_remains",
                    ]
                )
            else:
                disposition = CandidateDisposition.REJECT
                progress_class = "insufficient_global_progress"
                rationale.append("partial_global_progress_below_threshold")
        elif target_fixed is True and synthetic_truth:
            if remaining_true_faults == 0 and global_resolved is True:
                disposition = CandidateDisposition.ACCEPT_FINAL
                progress_class = "resolved"
                rationale.extend(
                    ["target_fixed", "no_remaining_true_faults", "globally_resolved"]
                )
            elif remaining_true_faults is not None and remaining_true_faults > 0:
                disposition = CandidateDisposition.ACCEPT_PARTIAL
                progress_class = "target_progress_remaining_true_faults"
                rationale.extend(["target_fixed", "remaining_true_faults_present"])
            else:
                disposition = CandidateDisposition.INCONCLUSIVE
                progress_class = "target_fixed_remaining_truth_unknown"
                rationale.extend(["target_fixed", "remaining_true_fault_count_unknown"])
        elif target_fixed is True and global_resolved is True:
            disposition = CandidateDisposition.ACCEPT_FINAL
            progress_class = "observable_resolved"
            rationale.extend(["target_fixed", "global_test_resolved"])
        elif target_fixed is True and global_resolved is False:
            disposition = CandidateDisposition.ACCEPT_PARTIAL
            progress_class = "observable_progress_global_anomaly_remains"
            rationale.extend(["target_fixed", "global_anomaly_remains"])
        elif target_fixed is True:
            disposition = CandidateDisposition.INCONCLUSIVE
            progress_class = "target_fixed_global_status_unknown"
            rationale.extend(["target_fixed", "global_resolution_not_established"])
        elif branch_target_marginal_on_clean_solve:
            disposition = CandidateDisposition.ACCEPT_FINAL
            progress_class = "observable_resolved_marginal_target"
            rationale.extend(
                [
                    "global_test_resolved",
                    "no_remaining_suspects",
                    "target_marginally_above_threshold",
                ]
            )
        elif branch_target_materially_improved and global_resolved is False:
            # The exact context-supported branch target improved materially
            # and is now only marginally above its local cutoff.  Retain the
            # repair as partial progress, but never call it final while the
            # explicit target test remains false.
            disposition = CandidateDisposition.ACCEPT_PARTIAL
            progress_class = "observable_branch_target_progress"
            rationale.extend(
                [
                    "target_local_progress",
                    "target_marginally_above_threshold",
                    "global_anomaly_remains",
                ]
            )
        elif target_fixed is False:
            disposition = CandidateDisposition.REJECT
            progress_class = "no_target_progress"
            rationale.append("target_not_fixed")
        else:
            observable = self._observable_progress(verification)
            if observable["target_regressed"] or observable["new_violations"]:
                disposition = CandidateDisposition.REJECT
                progress_class = "no_target_progress"
                rationale.append("observable_regression")
            elif (
                observable["target_improved"]
                and observable["global_resolved"] is True
            ):
                disposition = CandidateDisposition.ACCEPT_FINAL
                progress_class = "observable_resolved"
                rationale.extend(
                    ["observable_target_progress", "global_test_resolved"]
                )
            elif (
                observable["target_improved"]
                and observable["global_resolved"] is False
                and (global_progress is None or global_progress >= 0.0)
            ):
                disposition = CandidateDisposition.ACCEPT_PARTIAL
                progress_class = "observable_progress_global_anomaly_remains"
                rationale.extend(["observable_target_progress", "global_anomaly_remains"])
            elif observable["target_improved"]:
                disposition = CandidateDisposition.INCONCLUSIVE
                progress_class = "observable_progress_global_status_unknown"
                rationale.append("global_resolution_not_established")
            elif observable["mixed_or_weak"]:
                disposition = CandidateDisposition.INCONCLUSIVE
                progress_class = "mixed_or_weak"
                rationale.append("insufficient_observable_evidence")
            else:
                disposition = CandidateDisposition.REJECT
                progress_class = "no_target_progress"
                rationale.append("no_observable_progress")

        if parent.get("state_id") and candidate.get("parent_state_id") not in {None, parent.get("state_id")}:
            disposition = CandidateDisposition.REJECT
            progress_class = "invalid_provenance"
            rationale.append("candidate_parent_mismatch")
        family = self._action_family(action)
        belief_update = {family: progress_class} if family else {}

        return CandidateAssessment(
            disposition=disposition,
            progress_class=progress_class,
            target_progress=target_progress,
            global_progress=global_progress,
            collateral_damage=collateral_damage,
            remaining_true_fault_count=remaining_true_faults,
            remaining_suspect_count=remaining_suspects,
            belief_update=belief_update,
            unresolved_signatures=unresolved,
            rationale_codes=rationale,
        )

    def _target_fixed(
        self,
        action: Mapping[str, Any],
        verification: Mapping[str, Any],
        truth: Mapping[str, Any],
        candidate_meta: Mapping[str, Any],
        parent_state: Mapping[str, Any],
        candidate_state: Mapping[str, Any],
    ) -> bool | None:
        if not truth.get("truth_complete"):
            explicit = _first_present(
                truth.get("target_fixed"),
                verification.get("target_fixed"),
            )
            if explicit is not None:
                return bool(explicit)
            return None
        if self._physical_state_changed(parent_state, candidate_state) is False:
            return False
        family = self._action_family(action)
        if family is None:
            return None
        faults = truth.get(f"true_{family}_errors") or []
        if not faults:
            return False
        matches = [
            self._correction_matches_fault(
                action,
                fault,
                truth=truth,
                parent_state=parent_state,
                candidate_state=candidate_state,
            )
            for fault in faults
        ]
        if any(match is True for match in matches):
            return True
        return None if any(match is None for match in matches) else False

    def _remaining_true_faults(
        self,
        action: Mapping[str, Any],
        target_fixed: bool | None,
        verification: Mapping[str, Any],
        truth: Mapping[str, Any],
        candidate_meta: Mapping[str, Any],
        parent_state: Mapping[str, Any],
        candidate_state: Mapping[str, Any],
    ) -> int | None:
        # Synthetic truth is authoritative. Observable verifier suspect counts
        # may disagree, but cannot erase injected physical faults.
        if truth.get("truth_complete"):
            pre_action = _first_present(
                truth.get("remaining_true_fault_count"),
                truth.get("remaining_true_faults"),
            )
            matched = self._matched_fault_count(
                action,
                truth,
                parent_state=parent_state,
                candidate_state=candidate_state,
            ) if target_fixed else 0
            if pre_action is not None:
                count = len(pre_action) if isinstance(pre_action, (list, tuple, set)) else int(pre_action)
                return max(count - matched, 0)
            total = sum(len(truth.get(key) or []) for key in (
                "true_measurement_errors",
                "true_parameter_errors",
                "true_topology_errors",
            ))
            return max(total - matched, 0) if target_fixed else total

        explicit = _first_present(
            truth.get("remaining_true_fault_count"),
            truth.get("remaining_true_faults"),
            # Backward compatibility for older synthetic fixtures only. In
            # deployment mode hidden truth was already discarded above.
            truth.get("remaining_fault_count"),
        )
        if explicit is not None:
            return len(explicit) if isinstance(explicit, (list, tuple, set)) else int(explicit)

        return None

    @staticmethod
    def _remaining_suspects(verification: Mapping[str, Any]) -> int | None:
        value = verification.get("remaining_suspect_count")
        if value is None and verification.get("evidence_source") is not None:
            # Transitional compatibility for an external deployment provider
            # that has not yet adopted the renamed observable field. This
            # value is still treated only as suspect evidence.
            value = verification.get("remaining_fault_count")
        if value is None:
            return None
        return len(value) if isinstance(value, (list, tuple, set)) else int(value)

    def _collateral_damage(
        self,
        action: Mapping[str, Any],
        verification: Mapping[str, Any],
        truth: Mapping[str, Any],
        candidate_meta: Mapping[str, Any],
        parent_state: Mapping[str, Any],
        candidate_state: Mapping[str, Any],
    ) -> bool:
        explicit = _first_present(
            verification.get("healthy_component_modified"),
            verification.get("collateral_damage"),
        )
        if explicit is None and not truth.get("truth_complete"):
            explicit = truth.get("healthy_component_modified")
        if explicit is not None and bool(explicit):
            return True
        if self._structural_collateral(
            action,
            parent_state,
            candidate_state,
            allow_multi_measurement=bool(truth.get("truth_complete")),
        ):
            return True
        global_progress = _optional_float(verification.get("global_progress"))
        if global_progress is not None and global_progress < -self.min_target_progress:
            return True
        if truth.get("truth_complete"):
            if self._known_fault_regression(truth, parent_state, candidate_state):
                return True
            family = self._action_family(action)
            if family:
                faults = list(truth.get(f"true_{family}_errors") or [])
                if not faults:
                    return True
                targets = self._action_targets(action)
                fault_targets: set[str] = set().union(
                    *(self._fault_targets(fault) for fault in faults)
                )
                if family == "measurement":
                    if targets and not targets.issubset(fault_targets):
                        return True
                elif not any(
                    self._action_fault_targets_match(action, fault, parent_state)
                    for fault in faults
                ):
                    return True
                parent_case = parent_state.get("case")
                candidate_case = candidate_state.get("case")
                parent_measurements = parent_state.get("measurements")
                candidate_measurements = candidate_state.get("measurements")
                if family == "measurement":
                    if parent_case != candidate_case:
                        return True
                    if isinstance(parent_measurements, list) and isinstance(candidate_measurements, list):
                        changed = {
                            str(index)
                            for index, (before, after) in enumerate(
                                zip(parent_measurements, candidate_measurements)
                            )
                            if before != after
                        }
                        if len(parent_measurements) != len(candidate_measurements):
                            return True
                        if not changed.issubset(fault_targets):
                            return True
                elif parent_measurements != candidate_measurements:
                    # Parameter/topology corrections must not smuggle a
                    # measurement mutation into the same candidate.
                    return True
                elif family in {"parameter", "topology"}:
                    if isinstance(parent_case, Mapping) and isinstance(candidate_case, Mapping):
                        parent_nonbranch = {
                            key: value for key, value in parent_case.items() if key != "branch"
                        }
                        candidate_nonbranch = {
                            key: value for key, value in candidate_case.items() if key != "branch"
                        }
                        if parent_nonbranch != candidate_nonbranch:
                            return True
                    parent_rows = parent_case.get("branch") if isinstance(parent_case, Mapping) else None
                    candidate_rows = candidate_case.get("branch") if isinstance(candidate_case, Mapping) else None
                    if isinstance(parent_rows, list) and isinstance(candidate_rows, list):
                        if len(parent_rows) != len(candidate_rows):
                            return True
                        changed_rows = [
                            (before, after)
                            for before, after in zip(parent_rows, candidate_rows)
                            if before != after
                        ]
                        if len(changed_rows) > 1:
                            return True
                        if changed_rows and all(isinstance(row, Mapping) for pair in changed_rows for row in pair):
                            before, after = changed_rows[0]
                            changed_fields = {
                                str(key)
                                for key in set(before) | set(after)
                                if before.get(key) != after.get(key)
                            }
                            args = action.get("arguments") if isinstance(action.get("arguments"), Mapping) else {}
                            if family == "parameter":
                                allowed_fields = {str(args.get("parameter") or args.get("field") or "x")}
                            else:
                                allowed_fields = {str(args.get("status_field") or "status"), "br_status"}
                            if not changed_fields.issubset(allowed_fields):
                                return True
        return False

    @classmethod
    def _known_fault_regression(
        cls,
        truth: Mapping[str, Any],
        parent_state: Mapping[str, Any],
        candidate_state: Mapping[str, Any],
    ) -> bool:
        clean_measurements = truth.get("clean_measurements")
        parent_measurements = parent_state.get("measurements")
        candidate_measurements = candidate_state.get("measurements")
        for fault in truth.get("true_measurement_errors") or []:
            if not isinstance(fault, Mapping):
                continue
            index = _first_present(fault.get("index"), fault.get("index0"), fault.get("measurement_index"))
            expected = _first_present(
                fault.get("clean"), fault.get("clean_value"), fault.get("true_value")
            )
            try:
                numeric_index = int(index)
                if expected is None and isinstance(clean_measurements, (list, tuple)):
                    expected = clean_measurements[numeric_index]
                before = parent_measurements[numeric_index]
                after = candidate_measurements[numeric_index]
            except (TypeError, ValueError, IndexError):
                continue
            if expected is not None and cls._distance(after, expected) > cls._distance(before, expected) + 1e-12:
                return True

        clean_case = truth.get("clean_case")
        for family, tool in (("parameter", CORRECT_PARAMETERS), ("topology", CORRECT_TOPOLOGY)):
            for fault in truth.get(f"true_{family}_errors") or []:
                if not isinstance(fault, Mapping):
                    continue
                parent_row = cls._branch_row(parent_state.get("case"), fault)
                candidate_row = cls._branch_row(candidate_state.get("case"), fault)
                clean_row = cls._branch_row(clean_case, fault)
                if not isinstance(parent_row, Mapping) or not isinstance(candidate_row, Mapping):
                    continue
                if tool == CORRECT_PARAMETERS:
                    field = str(fault.get("parameter") or fault.get("field") or "x")
                    expected = _first_present(
                        fault.get("clean"),
                        fault.get("clean_value"),
                        fault.get("true_value"),
                        clean_row.get(field) if isinstance(clean_row, Mapping) else None,
                    )
                else:
                    field = str(
                        fault.get("status_field")
                        or ("br_status" if "br_status" in parent_row else "status")
                    )
                    expected = _first_present(
                        fault.get("expected_status"),
                        fault.get("clean"),
                        fault.get("true_value"),
                        clean_row.get(field) if isinstance(clean_row, Mapping) else None,
                    )
                if expected is None or field not in parent_row or field not in candidate_row:
                    continue
                if cls._distance(candidate_row[field], expected) > cls._distance(parent_row[field], expected) + 1e-12:
                    return True
        return False

    @staticmethod
    def _distance(value: Any, expected: Any) -> float:
        try:
            return abs(float(value) - float(expected))
        except (TypeError, ValueError):
            return 0.0 if value == expected else 1.0

    _PATH_CASE_FAMILY_COLUMNS = {
        # MATPOWER branch-matrix columns a family's correction may touch.
        "parameter": {2, 3, 4},  # BR_R, BR_X, BR_B
        "topology": {10},  # BR_STATUS
    }

    def _path_case_collateral(
        self,
        action: Mapping[str, Any],
        family: str,
        parent_case: Any,
        candidate_case: Any,
    ) -> bool:
        """Structurally audit a path-valued case correction via the differ."""
        if not isinstance(parent_case, str) or not isinstance(candidate_case, str):
            return True
        if parent_case == candidate_case or self.case_differ is None:
            return True
        try:
            diff = self.case_differ(parent_case, candidate_case)
        except Exception:
            return True
        if not isinstance(diff, Mapping) or not diff.get("comparable"):
            return True
        if diff.get("bus_changed") or diff.get("gen_changed") or diff.get("base_mva_changed"):
            return True
        changed_rows = diff.get("changed_branch_rows")
        if not isinstance(changed_rows, Mapping) or len(changed_rows) != 1:
            return True
        args = action.get("arguments") if isinstance(action.get("arguments"), Mapping) else {}
        target_row0: int | None = None
        if args.get("branch_row0") is not None:
            target_row0 = int(args["branch_row0"])
        elif args.get("line_index1") is not None:
            target_row0 = int(args["line_index1"]) - 1
        elif args.get("line_index") is not None:
            target_row0 = int(args["line_index"]) - 1
        ((changed_row, changed_columns),) = changed_rows.items()
        if target_row0 is None or int(changed_row) != target_row0:
            return True
        allowed = self._PATH_CASE_FAMILY_COLUMNS.get(family, set())
        return not {int(column) for column in changed_columns}.issubset(allowed)

    def _structural_collateral(
        self,
        action: Mapping[str, Any],
        parent_state: Mapping[str, Any],
        candidate_state: Mapping[str, Any],
        *,
        allow_multi_measurement: bool = False,
    ) -> bool:
        family = CandidateQualityOracle._action_family(action)
        parent_case = parent_state.get("case")
        candidate_case = candidate_state.get("case")
        parent_measurements = parent_state.get("measurements")
        candidate_measurements = candidate_state.get("measurements")
        if family == "measurement":
            if (
                "case" not in parent_state
                and "case" not in candidate_state
                and "measurements" not in parent_state
                and "measurements" not in candidate_state
            ):
                return False
            if parent_case != candidate_case:
                return True
            if not isinstance(parent_measurements, list) or not isinstance(candidate_measurements, list):
                return True
            if len(parent_measurements) != len(candidate_measurements):
                return True
            changed_indices = {
                index
                for index, (before, after) in enumerate(zip(parent_measurements, candidate_measurements))
                if before != after
            }
            args = action.get("arguments") if isinstance(action.get("arguments"), Mapping) else {}
            updates = args.get("measurement_updates")
            suspect_group = args.get("suspect_group")
            if isinstance(updates, Mapping):
                try:
                    declared_indices = {int(index) for index in updates}
                except (TypeError, ValueError):
                    return True
                if not changed_indices or changed_indices != declared_indices:
                    return True
                return not allow_multi_measurement and len(changed_indices) != 1
            if isinstance(suspect_group, (list, tuple)) and suspect_group:
                # Executor-hydrated grouped correction: the action declares the
                # suspect set and the executor computes the values, so touching
                # any measurement outside the declared group is collateral
                # damage while correcting a subset of it is not.
                try:
                    declared_indices = {int(index) for index in suspect_group}
                except (TypeError, ValueError):
                    return True
                return not changed_indices or not changed_indices.issubset(declared_indices)
            return True
        if family not in {"parameter", "topology"}:
            return False
        if parent_measurements != candidate_measurements:
            return True
        if isinstance(parent_case, str) or isinstance(candidate_case, str):
            return self._path_case_collateral(action, family, parent_case, candidate_case)
        if not isinstance(parent_case, Mapping) or not isinstance(candidate_case, Mapping):
            return True
        parent_nonbranch = {key: value for key, value in parent_case.items() if key != "branch"}
        candidate_nonbranch = {key: value for key, value in candidate_case.items() if key != "branch"}
        if parent_nonbranch != candidate_nonbranch:
            return True
        parent_rows = parent_case.get("branch")
        candidate_rows = candidate_case.get("branch")
        if not isinstance(parent_rows, list) or not isinstance(candidate_rows, list):
            return True
        if len(parent_rows) != len(candidate_rows):
            return True
        changed_rows = [
            (index, before, after)
            for index, (before, after) in enumerate(zip(parent_rows, candidate_rows))
            if before != after
        ]
        if len(changed_rows) != 1:
            return True
        if not all(
            isinstance(row, Mapping)
            for _, before, after in changed_rows
            for row in (before, after)
        ):
            return True
        changed_index, before, after = changed_rows[0]
        changed_fields = {
            str(key)
            for key in set(before) | set(after)
            if before.get(key) != after.get(key)
        }
        args = action.get("arguments") if isinstance(action.get("arguments"), Mapping) else {}
        declared_index = self._branch_index(parent_case, args)
        if declared_index is None or declared_index != changed_index:
            return True
        if family == "parameter":
            allowed_field = str(args.get("parameter") or args.get("field") or "x")
        else:
            allowed_field = self._topology_field(args, before)
        return changed_fields != {allowed_field}

    def _physical_ok(self, verification: Mapping[str, Any]) -> bool | None:
        explicit = verification.get("physical_constraints_ok")
        convergence_values = [
            verification.get(key)
            # Generic/WLS ``converged`` only proves optimizer completion; it
            # must never stand in for physical feasibility.
            for key in ("power_flow_converged", "topology_feasible")
            if verification.get(key) is not None
        ]
        if explicit is False or any(value is False for value in convergence_values):
            return False
        violations = _first_present(
            verification.get("physical_bound_violations"),
            verification.get("new_constraint_violations"),
            verification.get("new_violations"),
        )
        if violations is not None:
            count = len(violations) if isinstance(violations, list) else int(violations or 0)
            if count > self.max_new_violations:
                return False
        if explicit is True:
            return True
        if any(value is True for value in convergence_values) and violations is not None:
            return True
        return None

    def _solve_resolved(self, verification: Mapping[str, Any]) -> bool | None:
        """Did the candidate's own solve pass the global anomaly test?

        Distinct from :meth:`_global_resolved`: the deployment WLS provider
        folds the per-target test into ``globally_resolved`` (resolved *and*
        target fixed), so a clean solve with a marginal target multiplier
        reports ``globally_resolved=False`` while ``post_action_resolved`` and
        the statistic itself say the anomaly is gone.
        """
        explicit = _first_present(
            verification.get("post_action_resolved"),
            verification.get("no_material_anomaly_remaining"),
        )
        if explicit is not None:
            return bool(explicit)
        score = verification.get("remaining_anomaly_score")
        threshold = verification.get("anomaly_threshold", verification.get("chi_square_threshold"))
        if score is not None and threshold is not None:
            return float(score) < float(threshold)
        return None

    def _global_resolved(self, verification: Mapping[str, Any]) -> bool | None:
        explicit = _first_present(
            verification.get("globally_resolved"),
            verification.get("post_action_resolved"),
        )
        if explicit is not None:
            return bool(explicit)
        score = verification.get("remaining_anomaly_score")
        threshold = verification.get("anomaly_threshold", verification.get("chi_square_threshold"))
        if score is not None and threshold is not None:
            return float(score) < float(threshold)
        return None

    def _observable_progress(self, verification: Mapping[str, Any]) -> dict[str, bool]:
        target_progress = float(verification.get("target_progress", 0.0) or 0.0)
        global_progress = float(verification.get("global_progress", 0.0) or 0.0)
        violations = verification.get("new_large_residuals", verification.get("new_violations", 0))
        violation_count = len(violations) if isinstance(violations, list) else int(violations or 0)
        global_resolved = self._global_resolved(verification)
        target_improved = target_progress >= self.min_target_progress
        return {
            "target_improved": target_improved,
            "target_regressed": target_progress < -self.min_target_progress,
            "global_resolved": global_resolved,
            "new_violations": violation_count > self.max_new_violations,
            "mixed_or_weak": (
                abs(target_progress) < self.min_target_progress
                or global_progress < -self.min_target_progress
                or violation_count > self.max_new_violations
            ),
        }

    @staticmethod
    def _singleton_measurement_action(action: Mapping[str, Any]) -> bool:
        arguments = (
            action.get("arguments")
            if isinstance(action.get("arguments"), Mapping)
            else {}
        )
        group = arguments.get("suspect_group")
        if isinstance(group, (list, tuple)):
            return len(group) == 1
        updates = arguments.get("measurement_updates")
        if isinstance(updates, Mapping):
            return len(updates) == 1
        return False

    @staticmethod
    def _action_family(action: Mapping[str, Any]) -> str | None:
        return correction_family(action)

    @staticmethod
    def _action_targets(action: Mapping[str, Any]) -> set[str]:
        args = action.get("arguments") if isinstance(action.get("arguments"), Mapping) else {}
        targets: set[str] = set()
        for key in (
            "target", "meter", "measurement_id", "measurement_index", "index", "index0",
            "branch_id", "line_index", "line_index1", "branch_row0", "cb_name",
        ):
            if args.get(key) is not None:
                targets.add(str(args[key]))
        updates = args.get("measurement_updates")
        if isinstance(updates, Mapping):
            targets.update(str(key) for key in updates)
        group = args.get("suspect_group")
        if isinstance(group, (list, tuple, set)):
            targets.update(str(value) for value in group)
        return targets

    @staticmethod
    def _fault_targets(fault: Any) -> set[str]:
        if not isinstance(fault, Mapping):
            return {str(fault)}
        result: set[str] = set()
        for key in (
            "target", "meter", "measurement_id", "measurement_index", "index", "index0",
            "branch_id", "line_index", "line_index1", "branch_row0", "cb_name",
        ):
            if fault.get(key) is not None:
                result.add(str(fault[key]))
        return result

    def _matched_fault_count(
        self,
        action: Mapping[str, Any],
        truth: Mapping[str, Any],
        *,
        parent_state: Mapping[str, Any],
        candidate_state: Mapping[str, Any],
    ) -> int:
        return len(
            self.matched_fault_indices(
                action,
                truth,
                parent_state=parent_state,
                candidate_state=candidate_state,
            )
        )

    def matched_fault_indices(
        self,
        action: Mapping[str, Any],
        truth: Mapping[str, Any],
        *,
        parent_state: Mapping[str, Any],
        candidate_state: Mapping[str, Any],
    ) -> list[int]:
        """Return exactly the synthetic faults fixed by the physical candidate."""
        return matched_private_fault_indices(
            action,
            truth,
            parent_state=parent_state,
            candidate_state=candidate_state,
            case_loader=self.case_loader,
        )

    def _correction_matches_fault(
        self,
        action: Mapping[str, Any],
        fault: Any,
        *,
        truth: Mapping[str, Any],
        parent_state: Mapping[str, Any],
        candidate_state: Mapping[str, Any],
    ) -> bool | None:
        if not isinstance(fault, Mapping):
            return None
        return correction_matches_private_fault(
            action,
            fault,
            truth=truth,
            parent_state=parent_state,
            candidate_state=candidate_state,
            case_loader=self.case_loader,
        )

    @staticmethod
    def _physical_state_changed(
        parent_state: Mapping[str, Any], candidate_state: Mapping[str, Any]
    ) -> bool | None:
        parent_hash = parent_state.get("state_hash")
        candidate_hash = candidate_state.get("state_hash")
        if parent_hash is not None and candidate_hash is not None:
            return str(parent_hash) != str(candidate_hash)
        comparable = False
        for key in ("case", "measurements"):
            if key in parent_state and key in candidate_state:
                comparable = True
                if parent_state.get(key) != candidate_state.get(key):
                    return True
        return False if comparable else None

    def _action_fault_targets_match(
        self,
        action: Mapping[str, Any],
        fault: Any,
        state: Mapping[str, Any],
    ) -> bool:
        if not isinstance(fault, Mapping):
            return False
        return action_targets_private_fault(
            action,
            fault,
            parent_state=state,
            case_loader=self.case_loader,
        )

    @staticmethod
    def _branch_index(case: Any, descriptor: Mapping[str, Any]) -> int | None:
        if not isinstance(case, Mapping) or not isinstance(case.get("branch"), list):
            return None
        rows = case["branch"]
        target = canonical_branch_target(descriptor)
        if target is not None and target[0] == "branch_row0":
            index = int(target[1])
            return index if 0 <= index < len(rows) else None
        reference = _first_present(descriptor.get("branch_id"), descriptor.get("cb_name"))
        if reference is not None:
            for index, row in enumerate(rows):
                if not isinstance(row, Mapping):
                    continue
                if any(
                    row.get(key) is not None and str(row.get(key)) == str(reference)
                    for key in ("branch_id", "id", "name", "cb_name")
                ):
                    return index
        # Compatibility for pre-contract synthetic fixtures that used the
        # impossible one-based value zero.  Every positive ``line_index`` is
        # handled above using the reviewed one-based production contract.
        if descriptor.get("line_index") == 0:
            return 0 if rows else None
        return None

    @classmethod
    def _branch_row(cls, case: Any, descriptor: Mapping[str, Any]) -> Mapping[str, Any] | None:
        index = cls._branch_index(case, descriptor)
        if index is None or not isinstance(case, Mapping):
            return None
        rows = case.get("branch")
        return rows[index] if isinstance(rows, list) and isinstance(rows[index], Mapping) else None

    @staticmethod
    def _topology_field(arguments: Mapping[str, Any], row: Any) -> str:
        if arguments.get("status_field") is not None:
            return str(arguments["status_field"])
        if isinstance(row, Mapping) and "br_status" in row:
            return "br_status"
        return "status"

    @staticmethod
    def _unresolved_signatures(
        verification: Mapping[str, Any], truth: Mapping[str, Any]
    ) -> list[str]:
        raw = _first_present(verification.get("unresolved_signatures"), truth.get("unresolved_signatures"), [])
        return [str(item) for item in raw] if isinstance(raw, (list, tuple, set)) else [str(raw)]
