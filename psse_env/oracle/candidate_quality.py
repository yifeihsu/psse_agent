from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import math
from typing import Any, Mapping

from psse_env.actions import CORRECT_MEASUREMENTS, CORRECT_PARAMETERS, CORRECT_TOPOLOGY


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
    remaining_fault_count: int | None = None
    belief_update: dict[str, str] = field(default_factory=dict)
    unresolved_signatures: list[str] = field(default_factory=list)
    rationale_codes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["disposition"] = self.disposition.value
        return payload


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
        max_new_violations: int = 0,
        mode: str = "auto",
    ) -> None:
        self.min_target_progress = float(min_target_progress)
        self.max_new_violations = int(max_new_violations)
        if mode not in {"auto", "synthetic", "deployment"}:
            raise ValueError("mode must be 'auto', 'synthetic', or 'deployment'.")
        self.mode = mode

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
        candidate_meta = candidate.get("metadata") if isinstance(candidate.get("metadata"), Mapping) else {}

        target_progress = _optional_float(verification.get("target_progress"))
        global_progress = _optional_float(verification.get("global_progress"))
        target_fixed = self._target_fixed(
            action, verification, truth, candidate_meta, parent, candidate
        )
        remaining_faults = self._remaining_faults(
            action, target_fixed, verification, truth, candidate_meta, parent, candidate
        )
        global_resolved = self._global_resolved(verification, remaining_faults)
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
        elif not physical_ok:
            disposition = CandidateDisposition.REJECT
            progress_class = "physical_regression"
            rationale.append("physical_constraints_failed")
        elif target_fixed is True and remaining_faults == 0 and global_resolved:
            disposition = CandidateDisposition.ACCEPT_FINAL
            progress_class = "resolved"
            rationale.extend(["target_fixed", "no_remaining_faults", "globally_resolved"])
        elif target_fixed is True and remaining_faults is not None and remaining_faults > 0:
            disposition = CandidateDisposition.ACCEPT_PARTIAL
            progress_class = "target_progress_remaining_faults"
            rationale.extend(["target_fixed", "remaining_faults_present"])
        elif target_fixed is True and remaining_faults is None:
            disposition = CandidateDisposition.INCONCLUSIVE
            progress_class = "target_fixed_remaining_unknown"
            rationale.extend(["target_fixed", "remaining_fault_count_unknown"])
        elif target_fixed is True:
            disposition = CandidateDisposition.INCONCLUSIVE
            progress_class = "target_fixed_global_status_unknown"
            rationale.extend(["target_fixed", "global_resolution_not_established"])
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
                and observable["global_resolved"]
                and remaining_faults == 0
            ):
                disposition = CandidateDisposition.ACCEPT_FINAL
                progress_class = "observable_resolved"
                rationale.extend(
                    ["observable_target_progress", "globally_resolved", "no_remaining_faults"]
                )
            elif (
                observable["target_improved"]
                and remaining_faults is not None
                and remaining_faults > 0
                and (global_progress is None or global_progress >= 0.0)
            ):
                disposition = CandidateDisposition.ACCEPT_PARTIAL
                progress_class = "observable_progress_remaining_faults"
                rationale.extend(["observable_target_progress", "remaining_faults_present"])
            elif observable["target_improved"] and observable["global_resolved"]:
                disposition = CandidateDisposition.INCONCLUSIVE
                progress_class = "observable_resolution_remaining_unknown"
                rationale.extend(
                    ["observable_target_progress", "globally_resolved", "remaining_fault_count_unknown"]
                )
            elif observable["target_improved"]:
                disposition = CandidateDisposition.INCONCLUSIVE
                progress_class = "observable_progress_truth_unknown"
                rationale.append("remaining_fault_count_unknown")
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
            remaining_fault_count=remaining_faults,
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
        targets = self._action_targets(action)
        relevant = (
            list(faults)
            if not targets and len(faults) == 1
            else [
                fault
                for fault in faults
                if self._action_fault_targets_match(action, fault, parent_state)
            ]
        )
        if not relevant:
            return False
        matches = [
            self._correction_matches_fault(
                action,
                fault,
                truth=truth,
                parent_state=parent_state,
                candidate_state=candidate_state,
            )
            for fault in relevant
        ]
        if any(match is True for match in matches):
            return True
        return None if any(match is None for match in matches) else False

    def _remaining_faults(
        self,
        action: Mapping[str, Any],
        target_fixed: bool | None,
        verification: Mapping[str, Any],
        truth: Mapping[str, Any],
        candidate_meta: Mapping[str, Any],
        parent_state: Mapping[str, Any],
        candidate_state: Mapping[str, Any],
    ) -> int | None:
        # Complete synthetic truth is authoritative.  Observable verifier
        # counts may disagree, but cannot erase injected faults.
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

        # Deployment verifier counts are post-action observable evidence.
        post_action = _first_present(
            verification.get("remaining_fault_count"),
            verification.get("remaining_faults"),
            truth.get("remaining_fault_count"),
        )
        if post_action is not None:
            return len(post_action) if isinstance(post_action, (list, tuple, set)) else int(post_action)

        return None

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
            if not isinstance(updates, Mapping):
                return True
            try:
                declared_indices = {int(index) for index in updates}
            except (TypeError, ValueError):
                return True
            if not changed_indices or changed_indices != declared_indices:
                return True
            return not allow_multi_measurement and len(changed_indices) != 1
        if family not in {"parameter", "topology"}:
            return False
        if parent_measurements != candidate_measurements:
            return True
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

    def _physical_ok(self, verification: Mapping[str, Any]) -> bool:
        if verification.get("power_flow_converged") is False or verification.get("topology_feasible") is False:
            return False
        violations = verification.get("physical_bound_violations", verification.get("new_violations", 0))
        count = len(violations) if isinstance(violations, list) else int(violations or 0)
        return count <= self.max_new_violations

    def _global_resolved(self, verification: Mapping[str, Any], remaining_faults: int | None) -> bool:
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
        return remaining_faults == 0

    def _observable_progress(self, verification: Mapping[str, Any]) -> dict[str, bool]:
        target_progress = float(verification.get("target_progress", 0.0) or 0.0)
        global_progress = float(verification.get("global_progress", 0.0) or 0.0)
        violations = verification.get("new_large_residuals", verification.get("new_violations", 0))
        violation_count = len(violations) if isinstance(violations, list) else int(violations or 0)
        global_resolved = self._global_resolved(verification, None)
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
    def _action_family(action: Mapping[str, Any]) -> str | None:
        return {
            CORRECT_MEASUREMENTS: "measurement",
            CORRECT_PARAMETERS: "parameter",
            CORRECT_TOPOLOGY: "topology",
        }.get(action.get("tool"))

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
        family = self._action_family(action)
        if family is None:
            return []
        faults = list(truth.get(f"true_{family}_errors") or [])
        targets = self._action_targets(action)
        return [
            index
            for index, fault in enumerate(faults)
            if (
                not targets and len(faults) == 1
                or self._action_fault_targets_match(action, fault, parent_state)
            )
            and self._correction_matches_fault(
                action,
                fault,
                truth=truth,
                parent_state=parent_state,
                candidate_state=candidate_state,
            ) is True
        ]

    @classmethod
    def _correction_matches_fault(
        cls,
        action: Mapping[str, Any],
        fault: Any,
        *,
        truth: Mapping[str, Any],
        parent_state: Mapping[str, Any],
        candidate_state: Mapping[str, Any],
    ) -> bool | None:
        changed = cls._physical_state_changed(parent_state, candidate_state)
        if changed is False:
            return False
        if not isinstance(fault, Mapping):
            return None
        expected = _first_present(
            fault.get("clean"),
            fault.get("clean_value"),
            fault.get("true_value"),
            fault.get("expected_value"),
            fault.get("correct_value"),
            fault.get("expected_status"),
        )
        args = action.get("arguments") if isinstance(action.get("arguments"), Mapping) else {}
        actual = None
        target: Any = None
        updates = args.get("measurement_updates")
        if isinstance(updates, Mapping):
            fault_targets = cls._fault_targets(fault)
            for key, value in updates.items():
                if str(key) in fault_targets:
                    target = key
                    actual = value
                    break
        if target is None:
            target = _first_present(
                fault.get("index"),
                fault.get("index0"),
                fault.get("measurement_index"),
                fault.get("line_index"),
                fault.get("line_index1"),
                fault.get("branch_row0"),
                fault.get("branch_id"),
                fault.get("cb_name"),
            )
        if expected is None and action.get("tool") == CORRECT_MEASUREMENTS:
            clean_measurements = truth.get("clean_measurements")
            try:
                if isinstance(clean_measurements, (list, tuple)) and target is not None:
                    expected = clean_measurements[int(target)]
            except (TypeError, ValueError, IndexError):
                expected = None
        if expected is None and action.get("tool") == CORRECT_PARAMETERS:
            clean_values = truth.get("clean_parameter_values")
            if isinstance(clean_values, Mapping) and target is not None:
                expected = clean_values.get(target, clean_values.get(str(target)))
            elif isinstance(clean_values, (list, tuple)) and target is not None:
                try:
                    expected = clean_values[int(target)]
                except (TypeError, ValueError, IndexError):
                    expected = None
        parameter_field = str(args.get("parameter") or args.get("field") or "x")
        topology_field: str | None = None
        if action.get("tool") == CORRECT_TOPOLOGY:
            parent_row = cls._branch_row(parent_state.get("case"), args)
            topology_field = cls._topology_field(args, parent_row)
            fault_field = fault.get("status_field")
            if fault_field is not None and str(fault_field) != topology_field:
                return False
        if expected is None and action.get("tool") in {CORRECT_PARAMETERS, CORRECT_TOPOLOGY}:
            clean_row = cls._branch_row(truth.get("clean_case"), fault)
            if isinstance(clean_row, Mapping):
                if action.get("tool") == CORRECT_PARAMETERS:
                    expected = clean_row.get(parameter_field)
                else:
                    expected = clean_row.get(str(fault.get("status_field") or topology_field))
        if actual is None:
            actual = _first_present(
                args.get("value"),
                args.get("corrected_value"),
                args.get("new_value"),
                args.get("multiplier"),
                args.get("status"),
            )
        if action.get("tool") == CORRECT_MEASUREMENTS:
            measurements = candidate_state.get("measurements")
            try:
                if isinstance(measurements, (list, tuple)) and target is not None:
                    actual = measurements[int(target)]
            except (TypeError, ValueError, IndexError):
                actual = None
        elif action.get("tool") in {CORRECT_PARAMETERS, CORRECT_TOPOLOGY}:
            candidate_row = cls._branch_row(candidate_state.get("case"), args)
            if isinstance(candidate_row, Mapping):
                if action.get("tool") == CORRECT_PARAMETERS:
                    actual = candidate_row.get(parameter_field)
                else:
                    actual = candidate_row.get(str(topology_field))
        if expected is None or actual is None:
            return None
        try:
            return math.isclose(float(actual), float(expected), rel_tol=1e-6, abs_tol=1e-9)
        except (TypeError, ValueError):
            return actual == expected

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

    @classmethod
    def _action_fault_targets_match(
        cls,
        action: Mapping[str, Any],
        fault: Any,
        state: Mapping[str, Any],
    ) -> bool:
        if not isinstance(fault, Mapping):
            return bool(cls._action_targets(action) & cls._fault_targets(fault))
        family = cls._action_family(action)
        if family == "measurement":
            return bool(cls._action_targets(action) & cls._fault_targets(fault))
        args = action.get("arguments") if isinstance(action.get("arguments"), Mapping) else {}
        if family == "parameter":
            action_field = str(args.get("parameter") or args.get("field") or "x")
            fault_field = str(fault.get("parameter") or fault.get("field") or "x")
            if action_field != fault_field:
                return False
        elif family == "topology" and fault.get("status_field") is not None:
            row = cls._branch_row(state.get("case"), args)
            if cls._topology_field(args, row) != str(fault.get("status_field")):
                return False
        case = state.get("case")
        action_index = cls._branch_index(case, args)
        fault_index = cls._branch_index(case, fault)
        if action_index is not None and fault_index is not None:
            return action_index == fault_index
        return bool(cls._action_targets(action) & cls._fault_targets(fault))

    @staticmethod
    def _branch_index(case: Any, descriptor: Mapping[str, Any]) -> int | None:
        if not isinstance(case, Mapping) or not isinstance(case.get("branch"), list):
            return None
        rows = case["branch"]
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
        if descriptor.get("branch_row0") is not None:
            raw_index = descriptor.get("branch_row0")
            try:
                index = int(raw_index)
            except (TypeError, ValueError):
                return None
        elif descriptor.get("line_index1") is not None:
            raw_index = descriptor.get("line_index1")
            try:
                index = int(raw_index) - 1
            except (TypeError, ValueError):
                return None
        else:
            raw_index = _first_present(descriptor.get("line_index"), reference)
            try:
                index = int(raw_index)
            except (TypeError, ValueError):
                return None
            if not 0 <= index < len(rows) and 1 <= index <= len(rows):
                index -= 1
        return index if 0 <= index < len(rows) else None

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
