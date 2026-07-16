"""Deterministic, observable process-verifier rules.

Rules are intentionally conservative around finalization.  An
``ACCEPT_FINAL`` decision requires explicit observable evidence that the
remaining anomaly is resolved; missing evidence never implies success.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from psse_env.oracle.process_validity import ProcessValidityOracle

from .features import (
    ACTION_TOOLS,
    CONTEXT_TOOLS,
    CORRECTION_TOOLS,
    VERIFICATION_TOOLS,
    TransitionInput,
    extract_transition_features,
    normalize_transition,
    observable_verification_metrics,
)


DISPOSITIONS = ("REJECT", "INCONCLUSIVE", "ACCEPT_PARTIAL", "ACCEPT_FINAL")


@dataclass(frozen=True)
class RuleConfig:
    """Thresholds for the deployment-visible rule verifier."""

    min_target_progress: float = 0.05
    max_new_violations: int = 0
    anomaly_threshold: float = 1.0
    modification_scale: float = 1.0
    default_reject_remaining_steps: float = 3.0
    default_inconclusive_remaining_steps: float = 2.0

    def __post_init__(self) -> None:
        finite_nonnegative = (
            "min_target_progress",
            "max_new_violations",
            "default_reject_remaining_steps",
            "default_inconclusive_remaining_steps",
        )
        for name in finite_nonnegative:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative.")
            object.__setattr__(self, name, int(value) if name == "max_new_violations" else value)
        for name in ("anomaly_threshold", "modification_scale"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class VerifierDecision:
    """Structured process-verifier output specified by the roadmap."""

    process_valid: bool
    candidate_disposition: str
    progress_class: str
    collateral_damage_probability: float
    terminal_success_probability: float
    estimated_remaining_steps: float
    valid_next_action_types: list[str] = field(default_factory=list)
    rationale_codes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["collateral_damage_probability"] = _probability(
            result["collateral_damage_probability"]
        )
        result["terminal_success_probability"] = _probability(
            result["terminal_success_probability"]
        )
        result["estimated_remaining_steps"] = max(
            float(result["estimated_remaining_steps"]), 0.0
        )
        return result


def _probability(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return min(max(number, 0.0), 1.0)


def _float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _count(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return len(value)
    if isinstance(value, (list, tuple, set, frozenset)):
        return len(value)
    number = _float(value)
    return max(int(number), 0) if number is not None else None


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1", "success", "resolved"}:
            return True
        if lowered in {"false", "no", "0", "failure", "unresolved"}:
            return False
        return None
    return bool(value)


def _has_open_candidate(state: Mapping[str, Any]) -> bool:
    return bool(
        state.get("has_open_candidate")
        or state.get("has_unverified_candidate")
        or state.get("has_verified_candidate")
    )


def _context_is_fresh(state: Mapping[str, Any], family: str) -> bool:
    if not state.get(f"has_fresh_{family}_context"):
        return False
    context_id = state.get(f"{family}_context_state_id")
    active_id = state.get("active_state_id")
    return context_id is not None and active_id is not None and str(context_id) == str(active_id)


def _same_identifier(left: Any, right: Any) -> bool:
    """Compare optional state identifiers without conflating ``None`` and text."""

    if left is None or right is None:
        return left is right
    return str(left) == str(right)


def _identifier_episode(state_id: Any) -> str | None:
    """Extract the episode namespace from canonical ``<episode>:sN`` IDs."""

    if state_id is None:
        return None
    text = str(state_id)
    prefix, marker, serial = text.rpartition(":s")
    return prefix if marker and prefix and serial.isdigit() else None


def _invalid_payload_shape(tool: str, args: Mapping[str, Any]) -> bool:
    """Reject malformed payload containers not covered by the process gate."""

    nested = args.get("modification")
    if nested is not None and not isinstance(nested, Mapping):
        return True
    payload = dict(nested) if isinstance(nested, Mapping) else dict(args)
    payload.pop("state_id", None)
    payload.pop("candidate_state_id", None)

    if "case" in payload and payload.get("case") is None:
        return True
    if "case_updates" in payload and (
        not isinstance(payload.get("case_updates"), Mapping) or not payload.get("case_updates")
    ):
        return True
    if "measurements" in payload and not isinstance(payload.get("measurements"), (list, tuple)):
        return True

    updates = payload.get("measurement_updates")
    if updates is not None:
        if isinstance(updates, Mapping):
            if not updates:
                return True
            try:
                if any(int(index) < 0 for index in updates):
                    return True
            except (TypeError, ValueError, OverflowError):
                return True
        elif isinstance(updates, (list, tuple)):
            if not updates:
                return True
            for item in updates:
                if not isinstance(item, Mapping) or "value" not in item:
                    return True
                index = item.get("index", item.get("index0"))
                try:
                    if index is None or int(index) < 0:
                        return True
                except (TypeError, ValueError, OverflowError):
                    return True
        else:
            return True

    if tool == "correct_parameters":
        multiplier = payload.get("multiplier")
        if multiplier is not None and _float(multiplier) is None:
            return True
    return False


class RuleBasedVerifier:
    """Conservative verifier based only on structured observable evidence."""

    def __init__(self, config: RuleConfig | None = None, **config_overrides: Any) -> None:
        if config is not None and config_overrides:
            raise ValueError("Pass either config or keyword overrides, not both.")
        self.config = config or RuleConfig(**config_overrides)
        self.process_oracle = ProcessValidityOracle(
            anomaly_threshold=self.config.anomaly_threshold
        )

    def verify(
        self,
        transition: Mapping[str, Any] | None = None,
        **transition_parts: Any,
    ) -> dict[str, Any]:
        item = normalize_transition(transition, **transition_parts)
        features = extract_transition_features(item.as_dict())
        process_valid, process_reason = self.check_process(item)
        disposition, disposition_reasons = self._candidate_disposition(
            item, features, process_valid
        )
        reasons = ([process_reason] if process_reason else []) + disposition_reasons
        progress_class = self._progress_class(disposition, features)
        collateral_probability = self._collateral_probability(disposition, features)
        terminal_probability = self._terminal_probability(disposition, features)
        remaining_steps = self._estimated_remaining_steps(disposition, features)
        next_actions = self.valid_next_action_types(
            item,
            disposition=disposition,
            process_valid=process_valid,
            process_reason=process_reason,
        )
        return VerifierDecision(
            process_valid=process_valid,
            candidate_disposition=disposition,
            progress_class=progress_class,
            collateral_damage_probability=collateral_probability,
            terminal_success_probability=terminal_probability,
            estimated_remaining_steps=remaining_steps,
            valid_next_action_types=next_actions,
            rationale_codes=list(dict.fromkeys(reason for reason in reasons if reason)),
        ).as_dict()

    predict = verify

    @staticmethod
    def _transition_identity_error(item: TransitionInput) -> str | None:
        """Return a fail-closed reason for impossible state-ID transitions."""

        parent = item.parent_state_summary
        successor = item.candidate_state_summary
        output = item.tool_output
        output_metrics = output.get("tool_metrics")
        output_metrics = output_metrics if isinstance(output_metrics, Mapping) else {}
        tool = str(item.action.get("tool", "__invalid_action__"))
        status = output.get("execution_status")

        parent_episode = parent.get("episode_id")
        successor_episode = successor.get("episode_id")
        if (
            parent_episode is not None
            and successor_episode is not None
            and str(parent_episode) != str(successor_episode)
        ):
            return "cross_episode_transition"
        parent_namespace = _identifier_episode(parent.get("active_state_id"))
        successor_namespace = _identifier_episode(successor.get("active_state_id"))
        if (
            parent_namespace is not None
            and successor_namespace is not None
            and parent_namespace != successor_namespace
        ):
            return "cross_episode_transition"
        expected_episode = (
            parent_episode
            if parent_episode is not None
            else successor_episode
            if successor_episode is not None
            else parent_namespace
            if parent_namespace is not None
            else successor_namespace
        )
        if expected_episode is not None:
            identifiers = (
                parent.get("active_state_id"),
                parent.get("candidate_state_id"),
                parent.get("candidate_parent_id"),
                successor.get("active_state_id"),
                successor.get("candidate_state_id"),
                successor.get("candidate_parent_id"),
                output.get("active_state_id"),
                output.get("candidate_state_id"),
                output_metrics.get("candidate_state_id"),
                output_metrics.get("parent_state_id"),
            )
            for state_id in identifiers:
                namespace = _identifier_episode(state_id)
                if namespace is not None and namespace != str(expected_episode):
                    return "state_id_crosses_episode_boundary"
            output_episode = output_metrics.get("episode_id")
            if output_episode is not None and str(output_episode) != str(expected_episode):
                return "output_episode_id_mismatch"

        for state in (parent, successor):
            candidate_id = state.get("candidate_state_id")
            candidate_parent_id = state.get("candidate_parent_id")
            active_id = state.get("active_state_id")
            if candidate_id is None and candidate_parent_id is not None:
                return "dangling_candidate_parent_id"
            if (
                candidate_id is not None
                and candidate_parent_id is not None
                and active_id is not None
                and not _same_identifier(candidate_parent_id, active_id)
            ):
                return "candidate_parent_mismatch"

        if "active_state_id" in output and "active_state_id" in successor:
            if not _same_identifier(output.get("active_state_id"), successor.get("active_state_id")):
                return "output_active_state_id_mismatch"
        if "candidate_state_id" in output and "candidate_state_id" in successor:
            if not _same_identifier(
                output.get("candidate_state_id"), successor.get("candidate_state_id")
            ):
                return "output_candidate_state_id_mismatch"
        metrics_candidate = output_metrics.get("candidate_state_id")
        if metrics_candidate is not None:
            expected_candidate = successor.get("candidate_state_id")
            if expected_candidate is None:
                expected_candidate = output.get("candidate_state_id")
            if expected_candidate is not None and not _same_identifier(
                metrics_candidate, expected_candidate
            ):
                return "output_provenance_candidate_state_id_mismatch"
        metrics_parent = output_metrics.get("parent_state_id")
        if metrics_parent is not None:
            expected_parent = successor.get("candidate_parent_id")
            if expected_parent is None and tool in CORRECTION_TOOLS:
                expected_parent = parent.get("active_state_id")
            if expected_parent is not None and not _same_identifier(
                metrics_parent, expected_parent
            ):
                return "output_provenance_parent_state_id_mismatch"

        parent_active = parent.get("active_state_id")
        parent_candidate = parent.get("candidate_state_id")
        successor_active = successor.get("active_state_id")
        successor_candidate = successor.get("candidate_state_id")
        output_active = output.get("active_state_id")
        output_candidate = output.get("candidate_state_id")

        if status == "failure":
            comparisons = (
                (parent_active, successor_active, "failed_transition_changed_active_state"),
                (parent_active, output_active, "failed_output_changed_active_state"),
            )
            for expected, observed, reason in comparisons:
                if expected is not None and observed is not None and not _same_identifier(expected, observed):
                    return reason
            if "candidate_state_id" in successor and not _same_identifier(
                parent_candidate, successor_candidate
            ):
                return "failed_transition_changed_candidate_state"
            if "candidate_state_id" in output and not _same_identifier(
                parent_candidate, output_candidate
            ):
                return "failed_output_changed_candidate_state"
            return None

        if status != "success":
            return None
        if tool in CORRECTION_TOOLS:
            if successor_candidate is None and output_candidate is None:
                return "successful_correction_missing_candidate_state"
            if successor_candidate is not None:
                successor_parent = successor.get("candidate_parent_id")
                if successor_parent is None:
                    return "successful_correction_missing_candidate_parent"
                if parent_active is not None and not _same_identifier(
                    successor_parent, parent_active
                ):
                    return "candidate_parent_mismatch"
            if (
                parent_active is not None
                and successor_active is not None
                and not _same_identifier(parent_active, successor_active)
            ):
                return "correction_changed_active_state"
            if (
                parent_active is not None
                and output_active is not None
                and not _same_identifier(parent_active, output_active)
            ):
                return "correction_output_changed_active_state"
        elif tool == "commit_state":
            if (
                parent_candidate is not None
                and successor_active is not None
                and not _same_identifier(parent_candidate, successor_active)
            ):
                return "commit_active_state_id_mismatch"
            if successor_candidate is not None or output_candidate is not None:
                return "commit_left_candidate_open"
        else:
            if (
                parent_active is not None
                and successor_active is not None
                and not _same_identifier(parent_active, successor_active)
            ):
                return "unexpected_active_state_change"
            if (
                parent_active is not None
                and output_active is not None
                and not _same_identifier(parent_active, output_active)
            ):
                return "unexpected_output_active_state_change"
            if tool == "rollback_state" and (
                successor_candidate is not None or output_candidate is not None
            ):
                return "rollback_left_candidate_open"
            if tool != "rollback_state":
                if "candidate_state_id" in successor and not _same_identifier(
                    parent_candidate, successor_candidate
                ):
                    return "unexpected_candidate_state_change"
                if "candidate_state_id" in output and not _same_identifier(
                    parent_candidate, output_candidate
                ):
                    return "unexpected_output_candidate_state_change"
        return None

    def check_process(self, transition: Mapping[str, Any] | TransitionInput) -> tuple[bool, str | None]:
        item = transition if isinstance(transition, TransitionInput) else normalize_transition(transition)
        parent = item.parent_state_summary
        output = item.tool_output
        tool = str(item.action.get("tool", "__invalid_action__"))
        args = item.action.get("arguments")
        args = args if isinstance(args, Mapping) else {}
        has_open = _has_open_candidate(parent)
        active_id = parent.get("active_state_id")
        candidate_id = parent.get("candidate_state_id")
        reason: str | None = None

        identity_error = self._transition_identity_error(item)
        if identity_error is not None:
            return False, identity_error

        if tool not in ACTION_TOOLS:
            reason = "unknown_or_malformed_action"
        elif tool in VERIFICATION_TOOLS:
            expected_id = candidate_id if has_open else active_id
            state_id = args.get("state_id") or expected_id
            if not state_id:
                reason = "verification_missing_state_id"
            elif tool == "verify_candidate" and not has_open:
                reason = "verify_without_candidate"
            elif parent.get("has_verified_candidate"):
                reason = "candidate_already_verified"
            elif expected_id is not None and str(state_id) != str(expected_id):
                reason = "verification_state_reference_mismatch"
        elif tool == "rollback_state":
            requested = args.get("candidate_state_id") or candidate_id
            if candidate_id is None:
                reason = "rollback_without_candidate_state"
            elif not parent.get("has_verified_candidate"):
                reason = "rollback_without_verified_candidate"
            elif parent.get("candidate_committed"):
                reason = "rollback_after_commit"
            elif candidate_id is not None and str(requested) != str(candidate_id):
                reason = "rollback_candidate_reference_mismatch"
            elif (
                parent.get("candidate_disposition") is not None
                and parent.get("candidate_disposition") not in {"REJECT", "INCONCLUSIVE"}
            ):
                reason = "rollback_accepted_candidate"
        elif tool == "commit_state":
            requested = args.get("candidate_state_id") or candidate_id
            if candidate_id is None:
                reason = "commit_without_candidate_state"
            elif not parent.get("has_verified_candidate"):
                reason = "commit_without_verified_candidate"
            elif candidate_id is not None and str(requested) != str(candidate_id):
                reason = "commit_candidate_reference_mismatch"
            elif (
                parent.get("candidate_disposition") is not None
                and parent.get("candidate_disposition") not in {"ACCEPT_PARTIAL", "ACCEPT_FINAL"}
            ):
                reason = "commit_rejected_or_inconclusive_candidate"
        elif tool in CORRECTION_TOOLS:
            requested = args.get("state_id") or active_id
            if has_open:
                reason = "correction_with_open_candidate"
            elif active_id is not None and str(requested) != str(active_id):
                reason = "correction_state_reference_mismatch"
            elif tool == "correct_parameters" and not _context_is_fresh(parent, "parameter"):
                reason = "parameter_correction_without_parameter_context"
            elif tool == "correct_topology" and not _context_is_fresh(parent, "topology"):
                reason = "topology_correction_without_topology_context"
            elif (
                tool == "correct_measurements"
                and parent.get("requires_measurement_context")
                and not _context_is_fresh(parent, "measurement")
            ):
                reason = "measurement_correction_without_measurement_context"
            else:
                authoritative = self.process_oracle.check(parent, item.action, store=None)
                if authoritative.get("error_code") == "schema_error":
                    reason = (
                        "empty_correction_payload"
                        if authoritative.get("error_detail") == "empty_correction_payload"
                        else "invalid_correction_payload"
                    )
                elif _invalid_payload_shape(tool, args):
                    reason = "invalid_correction_payload"
        elif tool in CONTEXT_TOOLS:
            requested = args.get("state_id") or active_id
            if has_open:
                reason = "context_request_with_open_candidate"
            elif active_id is not None and str(requested) != str(active_id):
                reason = "context_state_reference_mismatch"
        elif tool == "finalize_diagnosis":
            if has_open:
                reason = "finalize_with_open_candidate"
            elif not self._observably_resolved(item):
                reason = "finalize_without_observable_resolution"
        elif tool in {"ask_for_more_evidence", "run_alternative_test"}:
            if has_open and parent.get("candidate_disposition") != "INCONCLUSIVE":
                reason = "evidence_action_requires_inconclusive_candidate"
            else:
                expected_id = candidate_id if has_open else active_id
                state_id = args.get("state_id") or expected_id
                if expected_id is not None and str(state_id) != str(expected_id):
                    reason = "evidence_state_reference_mismatch"

        # Structured environment errors are evidence that a nominally legal
        # action violated a transaction precondition.  Solver failures, on the
        # other hand, do not make the attempted process step illegal.
        error = str(output.get("error_code") or output.get("error") or "")
        error_detail = str(output.get("error_detail") or "")
        process_error_codes = {
            "argument_decode_error",
            "candidate_lifecycle_violation",
            "episode_already_terminal",
            "invalid_action",
            "json_parse_error",
            "missing_precondition",
            "policy_exception",
            "schema_error",
            "state_reference_mismatch",
            "terminal_condition_not_met",
            "unknown_state_id",
            "unknown_tool",
        }
        process_error_markers = (
            "unknown_tool",
            "unknown_state",
            "missing_state",
            "open_candidate",
            "without_",
            "after_commit",
            "already_terminal",
            "schema",
            "parse_error",
        )
        if reason is None and output.get("execution_status") == "failure" and (
            error in process_error_codes
            or any(marker in error or marker in error_detail for marker in process_error_markers)
        ):
            reason = error or "structured_process_failure"
        return reason is None, reason

    def _candidate_disposition(
        self,
        item: TransitionInput,
        features: Mapping[str, float],
        process_valid: bool,
    ) -> tuple[str, list[str]]:
        tool = str(item.action.get("tool"))
        if not process_valid:
            return "REJECT", ["invalid_process_transition"]

        if tool in CORRECTION_TOOLS:
            return "INCONCLUSIVE", ["candidate_requires_verification"]
        if tool == "rollback_state":
            return "REJECT", ["candidate_rolled_back"]
        if tool == "commit_state":
            inherited = item.parent_state_summary.get("candidate_disposition")
            if inherited in {"ACCEPT_PARTIAL", "ACCEPT_FINAL"}:
                return str(inherited), ["accepted_candidate_committed"]
            if item.tool_output.get("execution_status") == "success":
                return "INCONCLUSIVE", ["accepted_commit_disposition_not_observable"]
            return "REJECT", ["commit_failed_or_unaccepted"]

        violations = int(features.get("new_violations_count", 0.0))
        physical_failure = (
            features.get("power_flow_converged_known") == 1.0
            and features.get("power_flow_converged") == 0.0
        ) or (
            features.get("topology_feasible_known") == 1.0
            and features.get("topology_feasible") == 0.0
        )
        if physical_failure:
            return "REJECT", ["physical_feasibility_failure"]
        if violations > self.config.max_new_violations:
            return "REJECT", ["new_observable_violations"]
        if item.tool_output.get("execution_status") == "failure":
            return "INCONCLUSIVE", ["tool_execution_failed"]

        if tool == "finalize_diagnosis":
            if self._observably_resolved(item):
                return "ACCEPT_FINAL", ["successful_observable_finalization"]
            return "REJECT", ["false_finalization"]

        if tool not in VERIFICATION_TOOLS:
            return "INCONCLUSIVE", ["non_candidate_assessment_step"]

        # Running WLS on the active state is evidence gathering, not candidate
        # acceptance.  Candidate flags are the observable discriminator.
        candidate_after = item.candidate_state_summary
        verifying_candidate = bool(
            item.parent_state_summary.get("candidate_state_id")
            or candidate_after.get("candidate_state_id")
            or candidate_after.get("has_verified_candidate")
        )
        if not verifying_candidate:
            return "INCONCLUSIVE", ["active_state_verification"]

        target_known = features.get("target_progress_known") == 1.0
        target_progress = float(features.get("target_progress", 0.0))
        if target_known and target_progress <= 0.0:
            return "REJECT", ["no_target_progress"]
        if not target_known:
            return "INCONCLUSIVE", ["target_progress_unknown"]
        if target_progress < self.config.min_target_progress:
            return "INCONCLUSIVE", ["weak_target_progress"]

        if features.get("anomaly_resolved_known") == 1.0:
            if features.get("anomaly_resolved") == 1.0:
                return "ACCEPT_FINAL", ["observable_global_resolution"]
            return "ACCEPT_PARTIAL", ["target_progress_with_remaining_anomaly"]
        return "INCONCLUSIVE", ["global_resolution_unknown"]

    def _observably_resolved(self, item: TransitionInput) -> bool:
        metrics = observable_verification_metrics(item)
        power_ok = _optional_bool(metrics.get("power_flow_converged"))
        topology_ok = _optional_bool(metrics.get("topology_feasible"))
        violations = _count(metrics.get("new_violations"))
        if power_ok is False or topology_ok is False:
            return False
        if violations is not None and violations > self.config.max_new_violations:
            return False

        declared_resolved = _optional_bool(metrics.get("post_action_resolved"))
        score = _float(metrics.get("remaining_anomaly_score"))
        threshold = _float(metrics.get("anomaly_threshold"))
        if score is not None and threshold is not None:
            score_resolved = score < threshold
            # Any contradiction resolves toward the safer non-final result.
            return score_resolved and declared_resolved is not False
        if score is not None or threshold is not None:
            return False
        if declared_resolved is not None:
            return declared_resolved
        return item.candidate_state_summary.get("no_material_anomaly_remaining") is True

    def valid_next_action_types(
        self,
        transition: Mapping[str, Any] | TransitionInput,
        *,
        disposition: str,
        process_valid: bool = True,
        process_reason: str | None = None,
    ) -> list[str]:
        item = transition if isinstance(transition, TransitionInput) else normalize_transition(transition)
        parent = item.parent_state_summary
        candidate = item.candidate_state_summary
        # A present post-transition summary is authoritative even when it
        # explicitly closes the candidate with candidate_state_id=None.
        state = candidate if candidate else parent
        candidate_id = state.get("candidate_state_id")

        if not process_valid:
            repairs = {
                "parameter_correction_without_parameter_context": ["get_parameter_context"],
                "topology_correction_without_topology_context": ["get_topology_context"],
                "measurement_correction_without_measurement_context": ["get_measurement_context"],
                "finalize_without_observable_resolution": ["run_wls"],
                "verification_missing_state_id": ["run_wls"],
            }
            if process_reason in repairs:
                return repairs[process_reason]
            if candidate_id:
                if state.get("has_unverified_candidate"):
                    return ["run_wls"]
                return ["rollback_state"]
            return ["run_wls"]

        if state.get("has_unverified_candidate"):
            return ["run_wls"]
        if state.get("has_verified_candidate") or candidate_id:
            if disposition in {"ACCEPT_PARTIAL", "ACCEPT_FINAL"}:
                return ["commit_state"]
            if disposition == "REJECT":
                return ["rollback_state"]
            return ["rollback_state", "ask_for_more_evidence", "run_alternative_test"]
        if disposition == "REJECT":
            return ["run_wls", "ask_for_more_evidence", "run_alternative_test"]
        if disposition == "ACCEPT_FINAL" or self._observably_resolved(item):
            return ["finalize_diagnosis"]
        return ["run_wls", "ask_for_more_evidence", "run_alternative_test"]

    def _progress_class(self, disposition: str, features: Mapping[str, float]) -> str:
        if disposition == "ACCEPT_FINAL":
            return "resolved"
        if disposition == "ACCEPT_PARTIAL":
            return "target_progress_remaining_faults"
        if disposition == "REJECT":
            if features.get("new_violations_count", 0.0) > 0.0:
                return "collateral_damage"
            return "no_target_progress"
        return "mixed_or_weak"

    def _collateral_probability(self, disposition: str, features: Mapping[str, float]) -> float:
        if (
            features.get("power_flow_converged_known") == 1.0
            and features.get("power_flow_converged") == 0.0
        ) or (
            features.get("topology_feasible_known") == 1.0
            and features.get("topology_feasible") == 0.0
        ):
            return 0.99
        violations = features.get("new_violations_count", 0.0)
        if violations > self.config.max_new_violations:
            return min(0.75 + 0.05 * violations, 0.99)
        magnitude = max(features.get("modification_magnitude", 0.0), 0.0)
        scale = max(self.config.modification_scale, 1e-9)
        magnitude_risk = 0.25 * magnitude / (magnitude + scale)
        baseline = 0.30 if disposition == "REJECT" else 0.03
        return _probability(baseline + magnitude_risk)

    def _terminal_probability(self, disposition: str, features: Mapping[str, float]) -> float:
        if disposition == "ACCEPT_FINAL":
            return 0.98
        if disposition == "REJECT":
            return 0.02
        if disposition == "ACCEPT_PARTIAL":
            return 0.15
        if features.get("anomaly_resolved_known") == 1.0:
            return 0.65 if features.get("anomaly_resolved") == 1.0 else 0.10
        return 0.20

    def _estimated_remaining_steps(self, disposition: str, features: Mapping[str, float]) -> float:
        if disposition == "ACCEPT_FINAL":
            return 0.0
        if disposition == "REJECT":
            estimate = self.config.default_reject_remaining_steps
        elif disposition == "ACCEPT_PARTIAL":
            estimate = 1.5
        else:
            estimate = self.config.default_inconclusive_remaining_steps

        score = features.get("anomaly_score", 0.0)
        margin = features.get("anomaly_margin", 0.0)
        if features.get("anomaly_score_known") == 1.0 and score > 0.0:
            threshold = score + margin
            if threshold > 0.0:
                estimate += min(max(score / threshold - 1.0, 0.0), 4.0)
        budget = features.get("remaining_budget", 0.0)
        return min(estimate, budget) if budget > 0.0 else estimate


def verify_transition(
    transition: Mapping[str, Any] | None = None,
    *,
    config: RuleConfig | None = None,
    **transition_parts: Any,
) -> dict[str, Any]:
    """Functional entry point for one deterministic transition decision."""

    return RuleBasedVerifier(config=config).verify(transition, **transition_parts)


DeterministicProcessVerifier = RuleBasedVerifier


__all__ = [
    "DISPOSITIONS",
    "DeterministicProcessVerifier",
    "RuleBasedVerifier",
    "RuleConfig",
    "VerifierDecision",
    "verify_transition",
]
