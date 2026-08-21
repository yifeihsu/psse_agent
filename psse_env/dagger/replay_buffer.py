from __future__ import annotations

import copy
import hashlib
import math
import random
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping

from psse_env.dagger.round1_view_policy import (
    ROUND1_THREE_SOURCE_VIEW_POLICY,
    ROUND1_VIEW_POLICY_CONTRACT,
    round1_view_policy_digest,
    validate_round1_view_policy,
)


DEFAULT_REPLAY_WEIGHTS: dict[str, float] = {
    "clean_successful": 0.30,
    "rejected_candidate_recovery": 0.25,
    "accepted_partial_continuation": 0.20,
    "accepted_partial_commit": 0.10,
    "accepted_final_commit": 0.10,
    "invalid_precondition_recovery": 0.10,
    "terminal_resolved": 0.07,
    "terminal_operator_escalation": 0.03,
    "loop_repetition": 0.05,
}

DEFAULT_TRAINING_TOOL_CATEGORY_WEIGHTS: dict[str, float] = {
    "baseline_diagnostics": 0.20,
    "context_acquisition": 0.15,
    "corrections": 0.20,
    "verification_lifecycle": 0.20,
    "terminal_or_handoff": 0.10,
    "specialized_diagnostics": 0.15,
}

DEFAULT_MINIMUM_TOOL_CATEGORY_NATURAL_ROWS = 16
DEFAULT_MINIMUM_TOOL_CATEGORY_DISTINCT_ROOTS = 10

# These are release-support floors, not replay weights.  Rows repeated from one
# physical scenario must never make a critical DAgger-1 recovery behavior look
# independently supported.  The first-run floors are intentionally explicit so
# collection and final aggregate ingestion can recompute the same contract.
DAGGER1_TARGETED_STATE_CELL_MINIMUM_DISTINCT_ROOTS: dict[str, int] = {
    "multi_measurement_cardinality_2": 5,
    "multi_measurement_cardinality_4": 5,
    "multi_measurement_cardinality_5": 5,
    "parameter_route_actionable": 5,
    "parameter_route_complete_negative": 5,
    "parameter_route_unavailable": 5,
    "parameter_near_1_2_strict_rank": 5,
    "sequential_measurement_first": 5,
    "sequential_parameter_first": 5,
    "partial_success_retention": 5,
}

DAGGER1_RECOVERY_STRATUM_MINIMUM_DISTINCT_ROOTS: dict[str, int] = {
    # Central known-failure strata receive the stronger ten-root floor.
    "multi_measurement_safe_handoff": 10,
    "post_failure_no_candidate": 10,
    "sequential_measurement_parameter_recovery": 10,
    "unsupported_correction_recovery": 10,
    # The remaining predeclared D1 recovery strata still require independent
    # support rather than a single repeated physical trajectory.
    "premature_commit_recovery": 5,
    "premature_escalation_recovery": 5,
}
#: Strata whose occurrence depends on the learner making a specific mistake.
#: The complete 477-episode DAgger-1 schedule produced three roots for each
#: against a floor of ten, while every incidence-independent stratum scaled with
#: episode count.  Holding an absolute on-policy floor here penalises a learner
#: for improving, so natural support is reported rather than gated, and the
#: recovery competence it was meant to guarantee is carried by the probe and
#: combined floors instead.
DAGGER1_INCIDENCE_DEPENDENT_RECOVERY_STRATA = frozenset(
    {"post_failure_no_candidate", "unsupported_correction_recovery"}
)
#: Natural on-policy floors: every stratum except the incidence-dependent pair.
DAGGER1_NATURAL_RECOVERY_STRATUM_MINIMUM_DISTINCT_ROOTS: dict[str, int] = {
    stratum: floor
    for stratum, floor in DAGGER1_RECOVERY_STRATUM_MINIMUM_DISTINCT_ROOTS.items()
    if stratum not in DAGGER1_INCIDENCE_DEPENDENT_RECOVERY_STRATA
}
#: Probe and combined floors govern only the incidence-dependent pair.
DAGGER1_INCIDENCE_DEPENDENT_MINIMUM_DISTINCT_ROOTS: dict[str, int] = {
    stratum: DAGGER1_RECOVERY_STRATUM_MINIMUM_DISTINCT_ROOTS[stratum]
    for stratum in sorted(DAGGER1_INCIDENCE_DEPENDENT_RECOVERY_STRATA)
}


def _state_class(row: Mapping[str, Any]) -> str:
    labels = row.get("labels")
    label_class = labels.get("state_class") if isinstance(labels, Mapping) else None
    return str(row.get("state_class") or label_class or "clean_successful")


def _root_key(row: Mapping[str, Any], index: int) -> str:
    for field in ("physical_root_fingerprint", "root_scenario_id", "scenario_id"):
        value = row.get(field)
        if value not in (None, ""):
            return str(value)
    # Rows without root provenance must not all collapse into one artificial
    # root and starve the sampler.  The exporter/auditor can separately require
    # physical fingerprints for production datasets.
    return f"__unidentified_row_{index}"


def _physical_root(row: Mapping[str, Any]) -> str | None:
    """Return only an explicit physical root; never synthesize coverage."""
    value = row.get("physical_root_fingerprint")
    if value in (None, ""):
        return None
    return str(value)


def _production_label_eligibility(row: Mapping[str, Any]) -> bool | None:
    """Resolve release eligibility conservatively across supported locations."""
    values: list[Any] = []
    for source in (row, row.get("labels"), row.get("metadata")):
        if not isinstance(source, Mapping):
            continue
        if "production_label_eligible" in source:
            values.append(source.get("production_label_eligible"))
        nested_labels = source.get("labels")
        if (
            isinstance(nested_labels, Mapping)
            and "production_label_eligible" in nested_labels
        ):
            values.append(nested_labels.get("production_label_eligible"))
    if any(value is False for value in values):
        return False
    if any(value is True for value in values):
        return True
    return None


def _iteration(row: Mapping[str, Any]) -> int | None:
    try:
        value = int(row.get("iteration"))
    except (TypeError, ValueError, OverflowError):
        return None
    return value


def _allocate_counts(size: int, weights: Mapping[str, float]) -> dict[str, int]:
    total_weight = sum(weights.values())
    exact = {key: size * weight / total_weight for key, weight in weights.items()}
    counts = {key: int(value) for key, value in exact.items()}
    remainder = size - sum(counts.values())
    order = sorted(
        exact,
        key=lambda key: (exact[key] - counts[key], weights[key], key),
        reverse=True,
    )
    for key in order[:remainder]:
        counts[key] += 1
    return counts


def _capacity_aware_counts(
    *,
    size: int,
    weights: Mapping[str, float],
    capacities: Mapping[str, int],
) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
    """Return the closest weighted targets that fit marginal source capacity.

    The unconstrained allocation is retained in the report.  Values whose
    desired count exceeds their duplicate- and root-limited capacity are
    clipped, then the displaced count is deterministically redistributed to
    values with spare capacity. Release gates may still impose independent
    source-support floors and an achieved-deviation bound against these
    adjusted targets.
    """
    desired = _allocate_counts(size, weights)
    if set(desired) != set(capacities):
        raise ValueError("capacity-aware targets require one capacity per value")
    effective = {
        value: min(int(target), int(capacities[value]))
        for value, target in desired.items()
    }
    remaining = int(size) - sum(effective.values())
    while remaining:
        candidates = [
            value
            for value in sorted(effective)
            if effective[value] < int(capacities[value])
        ]
        if not candidates:
            raise ValueError(
                "capacity-aware targets cannot satisfy the requested training view"
            )
        # Fill the value furthest below its weighted fair share.  The value
        # name is the deterministic final tie-break.
        chosen = min(
            candidates,
            key=lambda value: (
                effective[value] / float(weights[value]),
                value,
            ),
        )
        effective[chosen] += 1
        remaining -= 1

    adjustments: dict[str, dict[str, int]] = {}
    for value in sorted(desired):
        if effective[value] == desired[value]:
            continue
        adjustments[value] = {
            "unconstrained_target": int(desired[value]),
            "capacity_adjusted_target": int(effective[value]),
            "maximum_achievable": int(capacities[value]),
            "necessary_reduction": max(
                int(desired[value]) - int(capacities[value]), 0
            ),
            "redistributed_increase": max(
                int(effective[value]) - int(desired[value]), 0
            ),
        }
    return effective, adjustments


def _minimum_feasible_root_cap(
    *, size: int, root_capacities: Mapping[str, int]
) -> int:
    if not root_capacities:
        raise ValueError("Replay sampling has no eligible physical roots.")
    if sum(root_capacities.values()) < size:
        raise ValueError(
            "Replay sample size exceeds the duplicate-limited source capacity; "
            "reduce size or raise max_duplicate_count explicitly."
        )
    lower = max(1, math.ceil(size / len(root_capacities)))
    upper = max(root_capacities.values())
    for cap in range(lower, upper + 1):
        if sum(min(cap, capacity) for capacity in root_capacities.values()) >= size:
            return cap
    raise ValueError("Unable to derive a feasible physical-root replay cap.")


def _summary(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    materialized = list(rows)
    state_classes: Counter[str] = Counter()
    iterations: Counter[str] = Counter()
    executed_by: Counter[str] = Counter()
    state_visited_by: Counter[str] = Counter()
    roots: Counter[str] = Counter()
    for index, row in enumerate(materialized):
        state_classes[_state_class(row)] += 1
        iteration = _iteration(row)
        iterations[str(iteration) if iteration is not None else "unknown"] += 1
        executed_by[str(row.get("executed_by") or "unknown")] += 1
        state_visited_by[str(row.get("state_visited_by") or "unknown")] += 1
        roots[_root_key(row, index)] += 1
    return {
        "rows": len(materialized),
        "state_class": dict(sorted(state_classes.items())),
        "iteration": dict(sorted(iterations.items())),
        "executed_by": dict(sorted(executed_by.items())),
        "state_visited_by": dict(sorted(state_visited_by.items())),
        "physical_root": dict(sorted(roots.items())),
    }


class BalancedReplayBuffer:
    """Sample a bounded, auditable view of the full DAgger aggregate.

    State-class weights remain the primary target.  Unknown classes fail closed
    by default, source-example duplication is bounded, physical-root
    contribution is capped at the smallest feasible value, and a configurable
    quota reserves capacity for learner-controlled states from the latest
    available DAgger iteration.  The sample report exposes any target shortfall.
    """

    def __init__(
        self,
        examples: Iterable[Mapping[str, Any]] | None = None,
        *,
        class_weights: Mapping[str, float] | None = None,
        unknown_class_policy: str = "error",
        unknown_class_weight: float = 0.05,
        max_duplicate_count: int = 2,
        max_rows_per_root: int | None = None,
        late_iteration_model_fraction: float = 0.25,
        require_late_iteration_model_quota: bool = False,
    ) -> None:
        self.class_weights = dict(class_weights or DEFAULT_REPLAY_WEIGHTS)
        if any(weight < 0 for weight in self.class_weights.values()) or sum(
            self.class_weights.values()
        ) <= 0:
            raise ValueError("Replay weights must be non-negative and contain positive mass.")
        if unknown_class_policy not in {"error", "fallback"}:
            raise ValueError("unknown_class_policy must be 'error' or 'fallback'.")
        if unknown_class_weight <= 0:
            raise ValueError("unknown_class_weight must be positive.")
        if int(max_duplicate_count) != max_duplicate_count or max_duplicate_count < 1:
            raise ValueError("max_duplicate_count must be a positive integer.")
        if max_rows_per_root is not None and (
            int(max_rows_per_root) != max_rows_per_root or max_rows_per_root < 1
        ):
            raise ValueError("max_rows_per_root must be a positive integer when provided.")
        if not 0.0 <= float(late_iteration_model_fraction) <= 1.0:
            raise ValueError("late_iteration_model_fraction must lie in [0, 1].")
        self.unknown_class_policy = unknown_class_policy
        self.unknown_class_weight = float(unknown_class_weight)
        self.max_duplicate_count = int(max_duplicate_count)
        self.max_rows_per_root = (
            int(max_rows_per_root) if max_rows_per_root is not None else None
        )
        self.late_iteration_model_fraction = float(late_iteration_model_fraction)
        self.require_late_iteration_model_quota = bool(
            require_late_iteration_model_quota
        )
        self._examples: list[dict[str, Any]] = [dict(row) for row in (examples or [])]
        self.last_sample_report: dict[str, Any] | None = None

    def add(self, example: Mapping[str, Any]) -> None:
        self._examples.append(dict(example))

    def extend(self, examples: Iterable[Mapping[str, Any]]) -> None:
        self._examples.extend(dict(row) for row in examples)

    def all_examples(self) -> list[dict[str, Any]]:
        return list(self._examples)

    def sample(self, size: int, *, rng: random.Random | None = None) -> list[dict[str, Any]]:
        if size < 0:
            raise ValueError("sample size must be non-negative")
        if size == 0:
            self.last_sample_report = {
                "requested_size": 0,
                "returned_size": 0,
                "before": _summary(self._examples),
                "after": _summary([]),
            }
            return []
        if not self._examples:
            raise ValueError("cannot sample an empty replay buffer")

        generator = rng or random.Random()
        buckets: dict[str, list[int]] = defaultdict(list)
        classes: dict[int, str] = {}
        roots: dict[int, str] = {}
        for index, row in enumerate(self._examples):
            state_class = _state_class(row)
            classes[index] = state_class
            roots[index] = _root_key(row, index)
            buckets[state_class].append(index)

        unknown_classes = sorted(set(buckets) - set(self.class_weights))
        if unknown_classes and self.unknown_class_policy == "error":
            raise ValueError(
                "Unknown replay state classes: " + ", ".join(unknown_classes)
            )
        effective_weights = dict(self.class_weights)
        if self.unknown_class_policy == "fallback":
            for state_class in unknown_classes:
                effective_weights[state_class] = self.unknown_class_weight
        available_weights = {
            state_class: weight
            for state_class, weight in effective_weights.items()
            if weight > 0 and buckets.get(state_class)
        }
        if not available_weights:
            raise ValueError("No replay classes with positive sampling mass are available.")

        eligible_indices = [
            index for state_class in available_weights for index in buckets[state_class]
        ]
        root_capacities: Counter[str] = Counter(
            {
                root: count * self.max_duplicate_count
                for root, count in Counter(roots[index] for index in eligible_indices).items()
            }
        )
        root_cap = self.max_rows_per_root
        if root_cap is None:
            root_cap = _minimum_feasible_root_cap(
                size=size, root_capacities=root_capacities
            )
        elif sum(min(root_cap, capacity) for capacity in root_capacities.values()) < size:
            raise ValueError(
                "max_rows_per_root is too small for the requested replay sample size."
            )

        target_counts = _allocate_counts(size, available_weights)
        remaining_counts = dict(target_counts)
        occurrence_counts: Counter[int] = Counter()
        selected_root_counts: Counter[str] = Counter()
        selected_indices: list[int] = []

        def choose_index(candidates: Iterable[int]) -> int | None:
            materialized = [
                index
                for index in candidates
                if occurrence_counts[index] < self.max_duplicate_count
                and selected_root_counts[roots[index]] < root_cap
            ]
            if not materialized:
                return None
            min_occurrences = min(occurrence_counts[index] for index in materialized)
            materialized = [
                index
                for index in materialized
                if occurrence_counts[index] == min_occurrences
            ]
            min_root_count = min(selected_root_counts[roots[index]] for index in materialized)
            materialized = [
                index
                for index in materialized
                if selected_root_counts[roots[index]] == min_root_count
            ]
            return generator.choice(materialized)

        def select(index: int) -> None:
            selected_indices.append(index)
            occurrence_counts[index] += 1
            selected_root_counts[roots[index]] += 1
            state_class = classes[index]
            if remaining_counts.get(state_class, 0) > 0:
                remaining_counts[state_class] -= 1

        numeric_iterations = [
            value
            for index in eligible_indices
            if (value := _iteration(self._examples[index])) is not None
        ]
        latest_iteration = max(numeric_iterations) if numeric_iterations else None
        late_candidates = [
            index
            for index in eligible_indices
            if latest_iteration is not None
            and _iteration(self._examples[index]) == latest_iteration
            and str(
                self._examples[index].get("state_visited_by")
                or self._examples[index].get("executed_by")
                or ""
            ).lower()
            in {"model", "learner"}
        ]
        late_candidate_set = set(late_candidates)
        late_requested = (
            math.ceil(size * self.late_iteration_model_fraction)
            if latest_iteration is not None and latest_iteration > 0
            else 0
        )
        late_selected = 0
        while late_selected < late_requested:
            candidate = choose_index(
                index
                for index in late_candidates
                if remaining_counts.get(classes[index], 0) > 0
            )
            if candidate is None:
                break
            select(candidate)
            late_selected += 1

        # Fill the requested state-class targets, prioritizing constrained
        # classes first.  If a cap makes one target infeasible, the final pass
        # redistributes its mass instead of duplicating a rare row indefinitely.
        class_order = sorted(
            available_weights,
            key=lambda state_class: (
                len({roots[index] for index in buckets[state_class]}),
                len(buckets[state_class]),
                state_class,
            ),
        )
        for state_class in class_order:
            while remaining_counts[state_class] > 0:
                candidate = choose_index(buckets[state_class])
                if candidate is None:
                    break
                select(candidate)

        while len(selected_indices) < size:
            candidate = choose_index(eligible_indices)
            if candidate is None:
                raise ValueError(
                    "Replay constraints cannot satisfy the requested sample size; "
                    "reduce size or relax an explicit root/duplicate cap."
                )
            select(candidate)

        generator.shuffle(selected_indices)
        result = [dict(self._examples[index]) for index in selected_indices]
        sampled_class_counts = Counter(classes[index] for index in selected_indices)
        target_shortfalls = {
            state_class: max(target - sampled_class_counts.get(state_class, 0), 0)
            for state_class, target in target_counts.items()
        }
        target_excesses = {
            state_class: max(sampled_class_counts.get(state_class, 0) - target, 0)
            for state_class, target in target_counts.items()
        }
        late_selected_count = sum(
            1 for index in selected_indices if index in late_candidate_set
        )
        late_quota_applicable = latest_iteration is not None and latest_iteration > 0
        self.last_sample_report = {
            "requested_size": size,
            "returned_size": len(result),
            "before": _summary(self._examples),
            "after": _summary(result),
            "target_state_class_counts": dict(sorted(target_counts.items())),
            "sampled_state_class_counts": dict(sorted(sampled_class_counts.items())),
            "state_class_target_shortfalls": dict(sorted(target_shortfalls.items())),
            "state_class_target_excesses": dict(sorted(target_excesses.items())),
            "unknown_classes": unknown_classes,
            "unknown_class_policy": self.unknown_class_policy,
            "max_duplicate_count": self.max_duplicate_count,
            "duplicate_occurrences": sum(
                max(count - 1, 0) for count in occurrence_counts.values()
            ),
            "max_rows_per_root": root_cap,
            "latest_iteration": latest_iteration,
            "late_iteration_model_fraction": self.late_iteration_model_fraction,
            "require_late_iteration_model_quota": (
                self.require_late_iteration_model_quota
            ),
            "late_iteration_model_requested": late_requested,
            "late_iteration_model_available": len(late_candidates),
            "late_iteration_model_selected": late_selected_count,
            "late_iteration_model_quota_applicable": late_quota_applicable,
            "late_iteration_model_shortfall": (
                max(late_requested - late_selected_count, 0)
                if late_quota_applicable
                else 0
            ),
            "late_iteration_model_quota_met": (
                not late_quota_applicable or late_selected_count >= late_requested
            ),
        }
        if (
            self.require_late_iteration_model_quota
            and late_quota_applicable
            and late_selected_count < late_requested
        ):
            raise ValueError(
                "Latest-iteration learner-state replay quota was not met: "
                f"selected {late_selected_count} of {late_requested}."
            )
        return result

    def sample_report(self) -> dict[str, Any] | None:
        return copy.deepcopy(self.last_sample_report)

    def class_counts(self) -> dict[str, int]:
        return dict(Counter(_state_class(row) for row in self._examples))


def balanced_replay_sample(
    examples: Iterable[Mapping[str, Any]],
    size: int,
    *,
    seed: int = 0,
    class_weights: Mapping[str, float] | None = None,
    unknown_class_policy: str = "error",
    unknown_class_weight: float = 0.05,
    max_duplicate_count: int = 2,
    max_rows_per_root: int | None = None,
    late_iteration_model_fraction: float = 0.25,
    require_late_iteration_model_quota: bool = False,
) -> list[dict[str, Any]]:
    return BalancedReplayBuffer(
        examples,
        class_weights=class_weights,
        unknown_class_policy=unknown_class_policy,
        unknown_class_weight=unknown_class_weight,
        max_duplicate_count=max_duplicate_count,
        max_rows_per_root=max_rows_per_root,
        late_iteration_model_fraction=late_iteration_model_fraction,
        require_late_iteration_model_quota=require_late_iteration_model_quota,
    ).sample(size, rng=random.Random(seed))


def _nonmodel_value(row: Mapping[str, Any], field: str) -> Any:
    if row.get(field) is not None:
        return row.get(field)
    metadata = row.get("metadata")
    if isinstance(metadata, Mapping):
        if metadata.get(field) is not None:
            return metadata.get(field)
        labels = metadata.get("labels")
        if isinstance(labels, Mapping) and labels.get(field) is not None:
            return labels.get(field)
    labels = row.get("labels")
    if isinstance(labels, Mapping):
        return labels.get(field)
    return None


def _target_tool(row: Mapping[str, Any]) -> str | None:
    """Return the exact tool that the chat exporter would supervise.

    Native collector rows normally carry ``preferred_action``.  Compatibility
    rows may instead expose the first admissible action.  A row with neither
    target is not an SFT example and must never acquire a synthetic
    ``unknown``/``specialized_diagnostics`` label inside the balancer.
    """
    action = row.get("preferred_action")
    if action is None:
        valid = row.get("valid_next_actions")
        if isinstance(valid, list) and valid:
            action = valid[0]
    if action is not None:
        if not isinstance(action, Mapping) or not action.get("tool"):
            raise ValueError(
                "training-view target must be a mapping with a non-empty tool: "
                f"{row.get('example_id')!r}"
            )
        return str(action["tool"])
    return None


def _normalize_dagger1_root_floors(
    value: Mapping[str, int], *, name: str
) -> dict[str, int]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{name} must be a non-empty mapping")
    normalized: dict[str, int] = {}
    for raw_key, raw_floor in value.items():
        key = str(raw_key).strip()
        if not key:
            raise ValueError(f"{name} keys must be non-empty")
        if (
            isinstance(raw_floor, bool)
            or not isinstance(raw_floor, int)
            or raw_floor <= 0
        ):
            raise ValueError(f"{name} values must be positive integers")
        normalized[key] = int(raw_floor)
    return dict(sorted(normalized.items()))


def dagger1_targeted_state_cells(row: Mapping[str, Any]) -> frozenset[str]:
    """Classify the first-run D1 root-support cells from non-secret fields.

    The classifier intentionally consumes only public grouping metadata, the
    policy observation, and the already-fixed teacher target.  Physical-root
    identity is used only by the outer audit to count independent support and
    is never added to the model observation.
    """

    cells: set[str] = set()
    family = str(_nonmodel_value(row, "scenario_family") or "")
    try:
        cardinality = int(_nonmodel_value(row, "error_cardinality") or 0)
    except (TypeError, ValueError, OverflowError):
        cardinality = 0
    if (
        family == "multi_measurement"
        and cardinality in {2, 4, 5}
        and _nonmodel_value(row, "parameter_scans_available") is False
    ):
        cells.add(f"multi_measurement_cardinality_{cardinality}")

    observation = row.get("policy_observation") or row.get("state_summary") or {}
    observation = observation if isinstance(observation, Mapping) else {}
    evidence_by_family = observation.get("fresh_context_evidence")
    parameter = (
        evidence_by_family.get("parameter")
        if isinstance(evidence_by_family, Mapping)
        else None
    )
    if isinstance(parameter, Mapping):
        route = str(parameter.get("route_status") or "")
        if route == "actionable":
            cells.add("parameter_route_actionable")
        elif route == "complete_negative":
            cells.add("parameter_route_complete_negative")
        elif route.startswith("unavailable"):
            cells.add("parameter_route_unavailable")
        try:
            ratio = float(parameter.get("parameter_ranking_dominance_ratio"))
        except (TypeError, ValueError, OverflowError):
            ratio = float("nan")
        if math.isfinite(ratio) and 1.0 < ratio < 1.2:
            cells.add("parameter_near_1_2_strict_rank")

    prior_families: set[str] = set()
    history = observation.get("history_window")
    if isinstance(history, (list, tuple)):
        for event in history:
            action = event.get("action") if isinstance(event, Mapping) else None
            tool = (
                str(action.get("tool") or "")
                if isinstance(action, Mapping)
                else ""
            )
            if "measurement" in tool:
                prior_families.add("measurement")
            elif "parameter" in tool:
                prior_families.add("parameter")
    target_tool = _target_tool(row) or ""
    if family == "measurement+parameter":
        if "measurement" in prior_families and "parameter" in target_tool:
            cells.add("sequential_measurement_first")
        if "parameter" in prior_families and "measurement" in target_tool:
            cells.add("sequential_parameter_first")
    if observation.get("accepted_corrections") and observation.get(
        "no_material_anomaly_remaining"
    ) is not True:
        cells.add("partial_success_retention")
    return frozenset(cells)


def _dagger1_recovery_stratum(row: Mapping[str, Any]) -> str | None:
    value = _nonmodel_value(row, "recovery_stratum")
    if value in (None, ""):
        return None
    return str(value)


def audit_dagger1_independent_root_support(
    examples: Iterable[Mapping[str, Any]],
    *,
    targeted_state_cell_minimum_distinct_roots: Mapping[str, int] = (
        DAGGER1_TARGETED_STATE_CELL_MINIMUM_DISTINCT_ROOTS
    ),
    recovery_stratum_minimum_distinct_roots: Mapping[str, int] = (
        DAGGER1_RECOVERY_STRATUM_MINIMUM_DISTINCT_ROOTS
    ),
) -> dict[str, Any]:
    """Require independent physical support for D1 cells and strata.

    This audit is suitable both immediately after collection and at final
    aggregate ingestion.  It never trusts row counts as a proxy for physical
    diversity: only explicit ``physical_root_fingerprint`` values count.
    """

    cell_floors = _normalize_dagger1_root_floors(
        targeted_state_cell_minimum_distinct_roots,
        name="targeted_state_cell_minimum_distinct_roots",
    )
    stratum_floors = _normalize_dagger1_root_floors(
        recovery_stratum_minimum_distinct_roots,
        name="recovery_stratum_minimum_distinct_roots",
    )
    rows = list(examples)
    cell_rows: Counter[str] = Counter()
    stratum_rows: Counter[str] = Counter()
    cell_roots: dict[str, set[str]] = defaultdict(set)
    stratum_roots: dict[str, set[str]] = defaultdict(set)
    cell_missing_roots: Counter[str] = Counter()
    stratum_missing_roots: Counter[str] = Counter()

    for row in rows:
        root_value = _physical_root(row)
        root = str(root_value).strip() if root_value is not None else None
        if not root:
            root = None
        for cell in dagger1_targeted_state_cells(row):
            cell_rows[cell] += 1
            if root is None:
                cell_missing_roots[cell] += 1
            else:
                cell_roots[cell].add(root)
        stratum = _dagger1_recovery_stratum(row)
        if stratum is None:
            continue
        stratum_rows[stratum] += 1
        if root is None:
            stratum_missing_roots[stratum] += 1
        else:
            stratum_roots[stratum].add(root)

    def support_report(
        *,
        floors: Mapping[str, int],
        row_counts: Mapping[str, int],
        roots: Mapping[str, set[str]],
        missing_roots: Mapping[str, int],
    ) -> dict[str, dict[str, int | bool]]:
        result: dict[str, dict[str, int | bool]] = {}
        for name in sorted(set(row_counts) | set(floors)):
            observed_roots = len(roots.get(name, set()))
            minimum_roots = int(floors.get(name, 0))
            missing_count = int(missing_roots.get(name, 0))
            shortfall = max(minimum_roots - observed_roots, 0)
            required = name in floors
            result[name] = {
                "target_bearing_rows": int(row_counts.get(name, 0)),
                "distinct_physical_roots": observed_roots,
                "rows_missing_physical_root": missing_count,
                "minimum_distinct_physical_roots": minimum_roots,
                "root_shortfall": shortfall,
                "required_for_release": required,
                "passed": (not required) or (
                    shortfall == 0 and missing_count == 0
                ),
            }
        return result

    cell_support = support_report(
        floors=cell_floors,
        row_counts=cell_rows,
        roots=cell_roots,
        missing_roots=cell_missing_roots,
    )
    stratum_support = support_report(
        floors=stratum_floors,
        row_counts=stratum_rows,
        roots=stratum_roots,
        missing_roots=stratum_missing_roots,
    )
    cell_shortfalls = {
        name: details
        for name, details in cell_support.items()
        if details["required_for_release"] and not details["passed"]
    }
    stratum_shortfalls = {
        name: details
        for name, details in stratum_support.items()
        if details["required_for_release"] and not details["passed"]
    }
    return {
        "contract": "dagger1_independent_physical_root_support_v1",
        "total_rows": len(rows),
        "targeted_state_cell_minimum_distinct_roots": cell_floors,
        "recovery_stratum_minimum_distinct_roots": stratum_floors,
        "targeted_state_cells": cell_support,
        "recovery_strata": stratum_support,
        "targeted_state_cell_shortfalls": cell_shortfalls,
        "recovery_stratum_shortfalls": stratum_shortfalls,
        "targeted_state_cells_passed": not cell_shortfalls,
        "recovery_strata_passed": not stratum_shortfalls,
        "passed": not cell_shortfalls and not stratum_shortfalls,
    }


DAGGER1_TRAINING_SUPPORT_CONTRACT = "dagger1_natural_probe_combined_support_v1"


def audit_dagger1_training_support(
    natural_rows: Iterable[Mapping[str, Any]],
    probe_rows: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Report natural, probe, and combined recovery support separately.

    Three questions are deliberately kept apart, because one number cannot
    answer them:

    * how often the current learner naturally reaches each recovery state;
    * whether the auxiliary probe suite supplies independent supervision for the
      two incidence-dependent strata;
    * whether the training corpus as a whole clears the competence floor.

    Natural support for the incidence-dependent pair is reported, never gated:
    an improved learner makes fewer mistakes and would otherwise fail release
    for getting better.  Every other floor stays a natural on-policy floor, so
    probe rows cannot paper over a genuine coverage gap.
    """

    natural = list(natural_rows)
    probes = list(probe_rows)

    natural_support = audit_dagger1_independent_root_support(
        natural,
        recovery_stratum_minimum_distinct_roots=(
            DAGGER1_NATURAL_RECOVERY_STRATUM_MINIMUM_DISTINCT_ROOTS
        ),
    )
    def strata_only(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
        """Support for the incidence-dependent pair alone.

        Targeted-state cells are a property of the natural corpus: an auxiliary
        probe is not meant to cover them, so judging the probe or combined
        report against cell floors would fail for the wrong reason.
        """

        full = audit_dagger1_independent_root_support(
            rows,
            recovery_stratum_minimum_distinct_roots=(
                DAGGER1_INCIDENCE_DEPENDENT_MINIMUM_DISTINCT_ROOTS
            ),
        )
        strata = {
            name: entry
            for name, entry in full["recovery_strata"].items()
            if name in DAGGER1_INCIDENCE_DEPENDENT_RECOVERY_STRATA
        }
        shortfalls = {
            name: entry
            for name, entry in strata.items()
            if entry["required_for_release"] and not entry["passed"]
        }
        return {
            "contract": full["contract"],
            "total_rows": full["total_rows"],
            "recovery_stratum_minimum_distinct_roots": (
                DAGGER1_INCIDENCE_DEPENDENT_MINIMUM_DISTINCT_ROOTS
            ),
            "recovery_strata": strata,
            "recovery_stratum_shortfalls": shortfalls,
            "passed": not shortfalls,
        }

    probe_support = strata_only(probes)
    combined_support = strata_only(natural + probes)

    # Report natural support for the ungated pair explicitly, so a reader never
    # has to infer it from the absence of a floor.
    natural_incidence = {
        stratum: {
            "distinct_physical_roots": int(
                natural_support["recovery_strata"]
                .get(stratum, {})
                .get("distinct_physical_roots", 0)
            ),
            "gated": False,
        }
        for stratum in sorted(DAGGER1_INCIDENCE_DEPENDENT_RECOVERY_STRATA)
    }
    passed = bool(
        natural_support.get("passed")
        and probe_support.get("passed")
        and combined_support.get("passed")
    )
    return {
        "contract": DAGGER1_TRAINING_SUPPORT_CONTRACT,
        "natural_on_policy_support": natural_support,
        "observable_probe_support": probe_support,
        "combined_training_support": combined_support,
        "incidence_dependent_recovery_strata": sorted(
            DAGGER1_INCIDENCE_DEPENDENT_RECOVERY_STRATA
        ),
        "natural_incidence_report_only": natural_incidence,
        "natural_rows": len(natural),
        "probe_rows": len(probes),
        "passed": passed,
    }


def dagger1_probe_replay_quota(
    probe_rows: Iterable[Mapping[str, Any]],
    *,
    total_size: int,
    probe_share: float,
    max_duplicate_count: int = 2,
    max_rows_per_root: int = 8,
) -> dict[str, Any]:
    """Size the probe replay bucket without drawing on the natural D1 share.

    Probe rows occupy their own quota so an auxiliary row can never displace a
    natural on-policy row: the D0/D1 split is computed over the remaining
    ``1 - probe_share`` of the view.

    ``probe_share`` has no default on purpose.  Its value is a preregistration
    decision, not something this function may pick.
    """

    rows = list(probe_rows)
    if not rows:
        raise ValueError("probe replay quota requires at least one probe row")
    if int(total_size) < 1:
        raise ValueError("total_size must be a positive integer")
    if not 0.0 < float(probe_share) < 1.0:
        raise ValueError("probe_share must satisfy 0 < share < 1")

    # Same duplicate/root rule the D0 and D1 sources use: a root contributes at
    # most ``max_rows_per_root`` rows, and each distinct example at most
    # ``max_duplicate_count`` copies.
    duplicate_cap = int(max_duplicate_count)
    root_cap = int(max_rows_per_root)
    examples_by_root: dict[str, set[str]] = defaultdict(set)
    malformed = 0
    for row in rows:
        root = _physical_root(row)
        example_id = str(row.get("example_id") or "").strip()
        if root is None or not str(root).strip() or not example_id:
            malformed += 1
            continue
        examples_by_root[str(root).strip()].add(example_id)
    capacity_by_root = {
        root: min(root_cap, len(example_ids) * duplicate_cap)
        for root, example_ids in sorted(examples_by_root.items())
    }
    requested = int(round(float(probe_share) * int(total_size)))
    available = sum(capacity_by_root.values())
    shortfall = max(requested - available, 0)
    return {
        "contract": "dagger1_probe_replay_quota_v1",
        "probe_share": float(probe_share),
        "total_size": int(total_size),
        "requested_probe_rows": requested,
        "available_probe_rows": available,
        "probe_capacity_shortfall": shortfall,
        "distinct_physical_roots": len(examples_by_root),
        "malformed_probe_rows": malformed,
        "capacity_by_physical_root": capacity_by_root,
        "max_duplicate_count": duplicate_cap,
        "max_rows_per_root": root_cap,
        # What the D0/D1 split may still divide between them.
        "natural_share_remaining": round(1.0 - float(probe_share), 6),
        "natural_rows_remaining": int(total_size) - requested,
        "passed": shortfall == 0 and malformed == 0,
    }


def _dagger1_source_capacity(
    rows: Iterable[Mapping[str, Any]],
    *,
    max_duplicate_count: int,
    max_rows_per_root: int,
) -> dict[str, Any]:
    """Compute one source's identity-safe duplicate/root-limited capacity."""

    materialized = list(rows)
    examples_by_root: dict[str, set[str]] = defaultdict(set)
    roots_by_example: dict[str, set[str]] = defaultdict(set)
    missing_root_rows = 0
    missing_example_id_rows = 0
    for row in materialized:
        root = _physical_root(row)
        example_id = str(row.get("example_id") or "").strip()
        if root is None or not str(root).strip():
            missing_root_rows += 1
            continue
        if not example_id:
            missing_example_id_rows += 1
            continue
        normalized_root = str(root).strip()
        examples_by_root[normalized_root].add(example_id)
        roots_by_example[example_id].add(normalized_root)
    cross_root_example_ids = sorted(
        example_id
        for example_id, roots in roots_by_example.items()
        if len(roots) > 1
    )
    capacity_by_root = {
        root: min(
            int(max_rows_per_root),
            len(example_ids) * int(max_duplicate_count),
        )
        for root, example_ids in sorted(examples_by_root.items())
    }
    return {
        "natural_rows": len(materialized),
        "distinct_examples": len(roots_by_example),
        "distinct_physical_roots": len(examples_by_root),
        "missing_physical_root_rows": missing_root_rows,
        "missing_example_id_rows": missing_example_id_rows,
        "example_ids_spanning_multiple_roots": cross_root_example_ids,
        "capacity_by_physical_root": capacity_by_root,
        "maximum_replay_rows": sum(capacity_by_root.values()),
        "passed": not (
            missing_root_rows
            or missing_example_id_rows
            or cross_root_example_ids
        ),
    }


def dagger1_replay_capacity_report(
    d0_rows: Iterable[Mapping[str, Any]],
    d1_rows: Iterable[Mapping[str, Any]],
    *,
    size: int | None = None,
    d1_share: float = 0.25,
    minimum_d1_share: float = 0.20,
    maximum_d1_share: float = 0.30,
    max_duplicate_count: int = 2,
    max_rows_per_root: int = 8,
) -> dict[str, Any]:
    """Report duplicate/root-limited D0+D1 replay capacity before sampling.

    This is a marginal capacity bound: semantic balancing may impose stricter
    limits later, but the sampler can never exceed this report.  Reporting the
    largest feasible total makes an undersized D1 collection actionable before
    an expensive Round-1 launch.
    """

    d0 = list(d0_rows)
    d1 = list(d1_rows)
    if not d0 or not d1:
        raise ValueError("D0 and D1 capacity inputs must both be non-empty")
    if not 0.0 < float(minimum_d1_share) <= float(maximum_d1_share) < 1.0:
        raise ValueError("D1 share band must satisfy 0 < minimum <= maximum < 1")
    if not float(minimum_d1_share) <= float(d1_share) <= float(maximum_d1_share):
        raise ValueError("configured D1 share is outside the allowed share band")
    if (
        isinstance(max_duplicate_count, bool)
        or int(max_duplicate_count) != max_duplicate_count
        or int(max_duplicate_count) < 1
    ):
        raise ValueError("max_duplicate_count must be a positive integer")
    if (
        isinstance(max_rows_per_root, bool)
        or int(max_rows_per_root) != max_rows_per_root
        or int(max_rows_per_root) < 1
    ):
        raise ValueError("max_rows_per_root must be a positive integer")

    duplicate_cap = int(max_duplicate_count)
    root_cap = int(max_rows_per_root)

    d0_capacity = _dagger1_source_capacity(
        d0,
        max_duplicate_count=duplicate_cap,
        max_rows_per_root=root_cap,
    )
    d1_capacity = _dagger1_source_capacity(
        d1,
        max_duplicate_count=duplicate_cap,
        max_rows_per_root=root_cap,
    )
    requested_size = len(d0) + len(d1) if size is None else size
    if (
        isinstance(requested_size, bool)
        or not isinstance(requested_size, int)
        or requested_size < 2
    ):
        raise ValueError("size must be an integer of at least two")

    def allocation(total: int) -> tuple[int, int, float]:
        d1_count = int(math.floor(total * float(d1_share) + 0.5))
        d0_count = total - d1_count
        return d0_count, d1_count, d1_count / total

    maximum_total_bound = int(d0_capacity["maximum_replay_rows"]) + int(
        d1_capacity["maximum_replay_rows"]
    )
    largest_total = 0
    largest_allocation = (0, 0, 0.0)
    for candidate in range(2, maximum_total_bound + 1):
        d0_count, d1_count, observed_share = allocation(candidate)
        if (
            d0_count >= 1
            and d1_count >= 1
            and float(minimum_d1_share)
            <= observed_share
            <= float(maximum_d1_share)
            and d0_count <= int(d0_capacity["maximum_replay_rows"])
            and d1_count <= int(d1_capacity["maximum_replay_rows"])
        ):
            largest_total = candidate
            largest_allocation = (d0_count, d1_count, observed_share)

    requested_d0, requested_d1, requested_share = allocation(requested_size)
    requested_passed = bool(
        d0_capacity["passed"]
        and d1_capacity["passed"]
        and requested_d0 >= 1
        and requested_d1 >= 1
        and float(minimum_d1_share)
        <= requested_share
        <= float(maximum_d1_share)
        and requested_d0 <= int(d0_capacity["maximum_replay_rows"])
        and requested_d1 <= int(d1_capacity["maximum_replay_rows"])
    )
    return {
        "schema_version": 1,
        "contract": "dagger1_duplicate_and_root_limited_capacity_v1",
        "configured_d1_share": float(d1_share),
        "minimum_d1_share": float(minimum_d1_share),
        "maximum_d1_share": float(maximum_d1_share),
        "max_duplicate_count": duplicate_cap,
        "max_rows_per_root": root_cap,
        "sources": {"d0_bc0": d0_capacity, "d1_recovery": d1_capacity},
        "requested": {
            "total_rows": requested_size,
            "d0_bc0_rows": requested_d0,
            "d1_recovery_rows": requested_d1,
            "observed_d1_share": requested_share,
            "d0_capacity_shortfall": max(
                requested_d0 - int(d0_capacity["maximum_replay_rows"]), 0
            ),
            "d1_capacity_shortfall": max(
                requested_d1 - int(d1_capacity["maximum_replay_rows"]), 0
            ),
            "passed": requested_passed,
        },
        "largest_feasible": {
            "total_rows": largest_total,
            "d0_bc0_rows": largest_allocation[0],
            "d1_recovery_rows": largest_allocation[1],
            "observed_d1_share": largest_allocation[2],
        },
        "passed": requested_passed,
    }


DAGGER1_ROUND1_SOURCE_CAPACITY_CONTRACT = (
    "dagger1_round1_three_source_source_capacity_v1"
)


def dagger1_round1_source_capacity_report(
    d0_rows: Iterable[Mapping[str, Any]],
    natural_d1_rows: Iterable[Mapping[str, Any]],
    *,
    policy: Mapping[str, Any] = ROUND1_THREE_SOURCE_VIEW_POLICY,
) -> dict[str, Any]:
    """Audit the strict collector's sources against the frozen allocation.

    The strict collector owns D0 and natural D1 rows, so it can prove their
    duplicate/root-limited capacity directly.  It does not own the recovery
    probe artifact.  Probe capacity and the cap shared by natural D1 and probe
    rows therefore remain explicit downstream obligations rather than being
    guessed from a source-length-derived two-source share.

    Exact policy counts are authoritative.  In particular, this function must
    never re-derive 1,317/525/38 from a rounded share or from the number of
    selected natural rows available at a collection checkpoint.
    """

    frozen_policy = copy.deepcopy(dict(policy))
    policy_validation = validate_round1_view_policy(frozen_policy)
    if frozen_policy.get("contract") != ROUND1_VIEW_POLICY_CONTRACT:
        raise ValueError("Round-1 source capacity policy contract is not approved")
    if frozen_policy.get("schema_version") != 1:
        raise ValueError("Round-1 source capacity policy schema is not approved")

    allocation = frozen_policy.get("allocation")
    allocation = allocation if isinstance(allocation, Mapping) else {}
    d0_required = int(allocation.get("d0_bc0_rows") or 0)
    natural_d1_required = int(allocation.get("natural_d1_rows") or 0)
    probe_required = int(
        allocation.get("observable_recovery_probe_rows") or 0
    )
    total_required = int(frozen_policy.get("total_rows") or 0)
    natural_total = d0_required + natural_d1_required

    caps = frozen_policy.get("global_caps")
    if not isinstance(caps, Mapping):
        raise ValueError("Round-1 source capacity policy is missing global caps")
    duplicate_cap = caps.get("max_duplicate_count")
    root_cap = caps.get("max_rows_per_root")
    if caps.get("applies_across_sources") is not True:
        raise ValueError("Round-1 shared-source capacity policy is not enabled")

    if (
        isinstance(duplicate_cap, bool)
        or not isinstance(duplicate_cap, int)
        or duplicate_cap < 1
    ):
        raise ValueError("Round-1 max_duplicate_count must be a positive integer")
    if (
        isinstance(root_cap, bool)
        or not isinstance(root_cap, int)
        or root_cap < 1
    ):
        raise ValueError("Round-1 max_rows_per_root must be a positive integer")
    d0 = list(d0_rows)
    natural_d1 = list(natural_d1_rows)
    if not d0 or not natural_d1:
        raise ValueError("D0 and natural D1 capacity inputs must both be non-empty")
    d0_source = _dagger1_source_capacity(
        d0,
        max_duplicate_count=int(duplicate_cap),
        max_rows_per_root=int(root_cap),
    )
    natural_d1_source = _dagger1_source_capacity(
        natural_d1,
        max_duplicate_count=int(duplicate_cap),
        max_rows_per_root=int(root_cap),
    )

    def bind_source(
        source: Mapping[str, Any] | None,
        *,
        required_rows: int,
    ) -> dict[str, Any]:
        report = copy.deepcopy(dict(source or {}))
        available = int(report.get("maximum_replay_rows") or 0)
        identity_passed = report.get("passed") is True
        shortfall = max(required_rows - available, 0)
        report.update(
            {
                "identity_contract_passed": identity_passed,
                "required_rows": required_rows,
                "capacity_shortfall": shortfall,
                "capacity_margin": max(available - required_rows, 0),
                "passed": bool(identity_passed and shortfall == 0),
            }
        )
        return report

    d0_capacity = bind_source(d0_source, required_rows=d0_required)
    natural_d1_capacity = bind_source(
        natural_d1_source,
        required_rows=natural_d1_required,
    )
    bucket = frozen_policy.get("probe_bucket")
    bucket = bucket if isinstance(bucket, Mapping) else {}
    strata = frozen_policy.get("incidence_dependent_recovery_strata")
    strata = list(strata) if isinstance(strata, (list, tuple)) else []
    unique_probe_source_rows = int(
        bucket.get("distinct_roots_retained_per_stratum") or 0
    ) * len(strata)
    theoretical_probe_capacity = unique_probe_source_rows * int(duplicate_cap)
    passed = bool(
        policy_validation.get("passed") is True
        and d0_capacity.get("passed") is True
        and natural_d1_capacity.get("passed") is True
    )
    return {
        "schema_version": 1,
        "contract": DAGGER1_ROUND1_SOURCE_CAPACITY_CONTRACT,
        "scope": "strict_collection_d0_and_natural_d1_sources",
        "policy_contract": ROUND1_VIEW_POLICY_CONTRACT,
        "policy_digest": round1_view_policy_digest(frozen_policy),
        "policy_validation": policy_validation,
        "required_allocation": {
            "total_rows": total_required,
            "natural_rows": natural_total,
            "d0_bc0_rows": d0_required,
            "natural_d1_rows": natural_d1_required,
            "observable_recovery_probe_rows": probe_required,
        },
        "global_caps": {
            "max_duplicate_count": int(duplicate_cap),
            "max_rows_per_root": int(root_cap),
            "applies_across_sources": True,
        },
        "sources": {
            "d0_bc0": d0_capacity,
            "natural_d1": natural_d1_capacity,
        },
        "probe_policy_arithmetic": {
            "applicable_at_strict_collection": False,
            "unique_source_rows_required": unique_probe_source_rows,
            "requested_rows": probe_required,
            "theoretical_upper_bound_replay_rows": theoretical_probe_capacity,
            "theoretical_margin_rows": max(
                theoretical_probe_capacity - probe_required, 0
            ),
            "status": "non_evidentiary_policy_arithmetic_only",
        },
        "probe_artifact_capacity": {
            "applicable_at_strict_collection": False,
            "required_rows": probe_required,
            "status": "deferred_until_probe_artifact_is_validated",
            "verification_stage": "probe_suite_then_three_source_aggregate",
        },
        "combined_natural_probe_root_capacity": {
            "applicable_at_strict_collection": False,
            "status": "deferred_until_probe_rows_exist",
            "verification_stage": "three_source_aggregate",
        },
        "passed": passed,
    }


def _tool_category(tool: str) -> str:
    if tool in {"run_wls", "wls_from_path"}:
        return "baseline_diagnostics"
    if tool.startswith("get_") and tool.endswith("_context"):
        return "context_acquisition"
    if tool.startswith("correct_"):
        return "corrections"
    if tool in {"verify_candidate", "commit_state", "rollback_state"}:
        return "verification_lifecycle"
    if tool in {"finalize_diagnosis", "ask_for_more_evidence"}:
        return "terminal_or_handoff"
    return "specialized_diagnostics"


def _known_cost_margin(row: Mapping[str, Any]) -> float | None:
    for source in (row, row.get("metadata"), row.get("labels")):
        if not isinstance(source, Mapping):
            continue
        value = source.get("cost_margin")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def _target_tool_unique_root_support(
    rows: Iterable[Mapping[str, Any]],
    *,
    minimum_distinct_roots: Mapping[str, int],
) -> tuple[dict[str, dict[str, int | bool]], dict[str, dict[str, int]]]:
    """Report exact-action support without letting duplicates inflate roots."""
    materialized = list(rows)
    row_counts: Counter[str] = Counter()
    roots: dict[str, set[str]] = defaultdict(set)
    rows_missing_physical_root: Counter[str] = Counter()
    for row in materialized:
        tool = _target_tool(row)
        if tool is None:
            continue
        row_counts[tool] += 1
        root = _physical_root(row)
        if root is None:
            rows_missing_physical_root[tool] += 1
        else:
            roots[tool].add(root)

    support: dict[str, dict[str, int | bool]] = {}
    shortfalls: dict[str, dict[str, int]] = {}
    for tool in sorted(set(row_counts) | set(minimum_distinct_roots)):
        observed_roots = len(roots[tool])
        required_roots = int(minimum_distinct_roots.get(tool, 0))
        root_shortfall = max(required_roots - observed_roots, 0)
        support[tool] = {
            "target_bearing_rows": int(row_counts[tool]),
            "distinct_physical_roots": observed_roots,
            "rows_missing_physical_root": int(
                rows_missing_physical_root[tool]
            ),
            "minimum_distinct_physical_roots": required_roots,
            "root_shortfall": root_shortfall,
            "required_for_release": tool in minimum_distinct_roots,
            "passed": root_shortfall == 0,
        }
        if root_shortfall:
            shortfalls[tool] = {
                "target_bearing_rows": int(row_counts[tool]),
                "distinct_physical_roots": observed_roots,
                "rows_missing_physical_root": int(
                    rows_missing_physical_root[tool]
                ),
                "minimum_distinct_physical_roots": required_roots,
                "root_shortfall": root_shortfall,
            }
    return support, shortfalls


def _target_tool_joint_unique_root_support(
    rows: Iterable[Mapping[str, Any]],
    *,
    dimension: str,
    minimum_distinct_roots: Mapping[str, Mapping[str, int]],
) -> tuple[
    dict[str, dict[str, dict[str, int | bool]]],
    dict[str, dict[str, dict[str, int]]],
]:
    """Report exact tool-by-context support using explicit physical roots."""
    if dimension not in {"state_class", "scenario_family"}:
        raise ValueError(f"unsupported target-tool joint dimension: {dimension}")
    row_counts: Counter[tuple[str, str]] = Counter()
    roots: dict[tuple[str, str], set[str]] = defaultdict(set)
    rows_missing_physical_root: Counter[tuple[str, str]] = Counter()
    for row in rows:
        tool = _target_tool(row)
        if tool is None:
            continue
        if dimension == "state_class":
            value = _state_class(row)
        else:
            value = str(_nonmodel_value(row, dimension) or "unknown")
        cell = (tool, value)
        row_counts[cell] += 1
        root = _physical_root(row)
        if root is None:
            rows_missing_physical_root[cell] += 1
        else:
            roots[cell].add(root)

    required_cells = {
        (tool, value)
        for tool, values in minimum_distinct_roots.items()
        for value in values
    }
    support: dict[str, dict[str, dict[str, int | bool]]] = {}
    shortfalls: dict[str, dict[str, dict[str, int]]] = {}
    for tool, value in sorted(set(row_counts) | required_cells):
        observed_roots = len(roots[(tool, value)])
        required_roots = int(
            minimum_distinct_roots.get(tool, {}).get(value, 0)
        )
        root_shortfall = max(required_roots - observed_roots, 0)
        details: dict[str, int | bool] = {
            "target_bearing_rows": int(row_counts[(tool, value)]),
            "distinct_physical_roots": observed_roots,
            "rows_missing_physical_root": int(
                rows_missing_physical_root[(tool, value)]
            ),
            "minimum_distinct_physical_roots": required_roots,
            "root_shortfall": root_shortfall,
            "required_for_release": (tool, value) in required_cells,
            "passed": root_shortfall == 0,
        }
        support.setdefault(tool, {})[value] = details
        if root_shortfall:
            shortfalls.setdefault(tool, {})[value] = {
                key: int(details[key])
                for key in (
                    "target_bearing_rows",
                    "distinct_physical_roots",
                    "rows_missing_physical_root",
                    "minimum_distinct_physical_roots",
                    "root_shortfall",
                )
            }
    return support, shortfalls


def _normalize_joint_root_floors(
    value: Mapping[str, Mapping[str, int]] | None,
    *,
    name: str,
) -> dict[str, dict[str, int]]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a nested mapping")
    normalized: dict[str, dict[str, int]] = {}
    for raw_tool, raw_values in value.items():
        tool = str(raw_tool).strip()
        if not tool:
            raise ValueError(f"{name} tool keys must be non-empty")
        if not isinstance(raw_values, Mapping) or not raw_values:
            raise ValueError(f"{name}[{tool!r}] must be a non-empty mapping")
        for raw_dimension_value, raw_floor in raw_values.items():
            dimension_value = str(raw_dimension_value).strip()
            if not dimension_value:
                raise ValueError(
                    f"{name}[{tool!r}] dimension keys must be non-empty"
                )
            if (
                isinstance(raw_floor, bool)
                or int(raw_floor) != raw_floor
                or int(raw_floor) <= 0
            ):
                raise ValueError(f"{name} values must be positive integers")
            normalized.setdefault(tool, {})[dimension_value] = int(raw_floor)
    return {
        tool: dict(sorted(values.items()))
        for tool, values in sorted(normalized.items())
    }


def _normalize_same_root_prerequisite_rules(
    value: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    """Normalize target-tool prerequisites used by the training-view gate.

    A correction target is useful only when the same physical root also keeps
    the observable context decision that made the correction admissible.  The
    rule is deliberately expressed over non-model metadata (physical root and
    scenario family); it never adds a feature to the learner observation.
    """

    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("same_root_prerequisite_rules must be a mapping")
    normalized: dict[str, dict[str, Any]] = {}
    for raw_target, raw_rule in value.items():
        target = str(raw_target).strip()
        if not target:
            raise ValueError(
                "same_root_prerequisite_rules target tools must be non-empty"
            )
        if not isinstance(raw_rule, Mapping):
            raise ValueError(
                f"same_root_prerequisite_rules[{target!r}] must be a mapping"
            )
        raw_options_by_family = raw_rule.get("prerequisite_options_by_family")
        if raw_options_by_family is None:
            prerequisite = str(raw_rule.get("prerequisite_tool") or "").strip()
            if not prerequisite or prerequisite == target:
                raise ValueError(
                    f"same_root_prerequisite_rules[{target!r}] requires a "
                    "distinct non-empty prerequisite_tool"
                )
            raw_families = raw_rule.get("scenario_families")
            if (
                not isinstance(raw_families, (list, tuple, set, frozenset))
                or not raw_families
            ):
                raise ValueError(
                    f"same_root_prerequisite_rules[{target!r}] requires "
                    "non-empty scenario_families"
                )
            raw_options_by_family = {
                str(family).strip(): [{"tool": prerequisite}]
                for family in raw_families
                if str(family).strip()
            }
        if not isinstance(raw_options_by_family, Mapping) or not raw_options_by_family:
            raise ValueError(
                f"same_root_prerequisite_rules[{target!r}] requires non-empty "
                "prerequisite_options_by_family"
            )
        options_by_family: dict[str, list[dict[str, Any]]] = {}
        for raw_family, raw_options in raw_options_by_family.items():
            family = str(raw_family).strip()
            if not family or not isinstance(raw_options, (list, tuple)) or not raw_options:
                raise ValueError(
                    f"same_root_prerequisite_rules[{target!r}] has invalid "
                    f"options for family {raw_family!r}"
                )
            options: list[dict[str, Any]] = []
            for raw_option in raw_options:
                if not isinstance(raw_option, Mapping):
                    raise ValueError(
                        f"same_root_prerequisite_rules[{target!r}] prerequisite "
                        "options must be mappings"
                    )
                tool = str(raw_option.get("tool") or "").strip()
                if not tool or tool == target:
                    raise ValueError(
                        f"same_root_prerequisite_rules[{target!r}] option tool "
                        "must be non-empty and distinct from the target"
                    )
                evidence_path = str(raw_option.get("evidence_path") or "").strip()
                evidence_contract = str(
                    raw_option.get("evidence_contract") or ""
                ).strip()
                if evidence_contract and not evidence_path:
                    raise ValueError(
                        f"same_root_prerequisite_rules[{target!r}] evidence "
                        "contracts require an evidence_path"
                    )
                if evidence_contract not in {
                    "",
                    "bound_supported_parameter_inventory_v1",
                }:
                    raise ValueError(
                        f"same_root_prerequisite_rules[{target!r}] has unknown "
                        f"evidence contract {evidence_contract!r}"
                    )
                option: dict[str, Any] = {"tool": tool}
                if evidence_path:
                    option["evidence_path"] = evidence_path
                if evidence_contract:
                    option["evidence_contract"] = evidence_contract
                options.append(option)
            options_by_family[family] = options
        normalized[target] = {
            "prerequisite_options_by_family": dict(
                sorted(options_by_family.items())
            )
        }
    return dict(sorted(normalized.items()))


def _nested_mapping_value(row: Mapping[str, Any], path: str) -> Any:
    value: Any = row
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def _row_satisfies_prerequisite_option(
    row: Mapping[str, Any], option: Mapping[str, Any]
) -> bool:
    if _target_tool(row) != str(option["tool"]):
        return False
    evidence_path = str(option.get("evidence_path") or "")
    if not evidence_path:
        return True
    evidence = _nested_mapping_value(row, evidence_path)
    contract = str(option.get("evidence_contract") or "")
    if not contract:
        return evidence is not None
    if contract == "bound_supported_parameter_inventory_v1":
        if not isinstance(evidence, Mapping):
            return False
        supported = evidence.get("supported_corrections")
        return bool(
            evidence.get("context_tool") == "get_parameter_context"
            and evidence.get("route_status") == "actionable"
            and str(evidence.get("state_id") or "").strip()
            and str(evidence.get("state_hash") or "").strip()
            and isinstance(supported, (list, tuple))
            and supported
            and all(
                isinstance(action, Mapping)
                and str(action.get("tool") or "") == "correct_parameters"
                for action in supported
            )
        )
    return False


def _same_root_prerequisite_support(
    rows: Iterable[Mapping[str, Any]],
    *,
    rules: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, dict[str, dict[str, Any]]]]:
    """Audit paired target/prerequisite support without counting duplicates."""

    materialized = list(rows)
    support: dict[str, dict[str, dict[str, Any]]] = {}
    shortfalls: dict[str, dict[str, dict[str, Any]]] = {}
    for target_tool, rule in sorted(rules.items()):
        for family, options in rule["prerequisite_options_by_family"].items():
            target_roots: set[str] = set()
            prerequisite_roots: set[str] = set()
            target_rows = 0
            prerequisite_rows = 0
            option_rows: Counter[str] = Counter()
            option_roots: dict[str, set[str]] = defaultdict(set)
            target_rows_missing_root = 0
            for row in materialized:
                if str(_nonmodel_value(row, "scenario_family") or "unknown") != family:
                    continue
                tool = _target_tool(row)
                root = _physical_root(row)
                if tool == target_tool:
                    target_rows += 1
                    if root is None:
                        target_rows_missing_root += 1
                    else:
                        target_roots.add(root)
                for option in options:
                    if not _row_satisfies_prerequisite_option(row, option):
                        continue
                    option_id = str(option["tool"])
                    if option.get("evidence_contract"):
                        option_id += ":" + str(option["evidence_contract"])
                    prerequisite_rows += 1
                    option_rows[option_id] += 1
                    if root is not None:
                        prerequisite_roots.add(root)
                        option_roots[option_id].add(root)
                    break
            paired_roots = target_roots & prerequisite_roots
            missing_roots = sorted(target_roots - prerequisite_roots)
            details: dict[str, Any] = {
                "prerequisite_options": copy.deepcopy(options),
                "prerequisite_tools": sorted(
                    {str(option["tool"]) for option in options}
                ),
                "target_bearing_rows": target_rows,
                "target_distinct_physical_roots": len(target_roots),
                "target_rows_missing_physical_root": target_rows_missing_root,
                "prerequisite_bearing_rows": prerequisite_rows,
                "prerequisite_distinct_physical_roots": len(prerequisite_roots),
                "prerequisite_option_bearing_rows": dict(sorted(option_rows.items())),
                "prerequisite_option_distinct_physical_roots": {
                    option_id: len(roots)
                    for option_id, roots in sorted(option_roots.items())
                },
                "paired_distinct_physical_roots": len(paired_roots),
                "unpaired_target_distinct_physical_roots": len(missing_roots),
                "unpaired_target_physical_roots": missing_roots,
                "passed": not missing_roots and target_rows_missing_root == 0,
            }
            support.setdefault(target_tool, {})[family] = details
            if not details["passed"]:
                shortfalls.setdefault(target_tool, {})[family] = copy.deepcopy(
                    details
                )
    return support, shortfalls


def _training_view_summary(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    materialized = list(rows)
    axes = {
        "state_class": Counter(),
        "target_tool": Counter(),
        "tool_category": Counter(),
        "scenario_family": Counter(),
        "error_cardinality": Counter(),
        "terminal_outcome": Counter(),
        "physical_root": Counter(),
    }
    target_tool_roots: dict[str, set[str]] = defaultdict(set)
    target_tool_rows_missing_physical_root: Counter[str] = Counter()
    for index, row in enumerate(materialized):
        tool = _target_tool(row)
        if tool is None:
            raise ValueError(
                "training-view summary received a targetless row: "
                f"{row.get('example_id', index)!r}"
            )
        axes["state_class"][str(_state_class(row))] += 1
        axes["target_tool"][tool] += 1
        axes["tool_category"][_tool_category(tool)] += 1
        axes["scenario_family"][str(_nonmodel_value(row, "scenario_family") or "unknown")] += 1
        axes["error_cardinality"][str(_nonmodel_value(row, "error_cardinality") or 0)] += 1
        axes["terminal_outcome"][str(
            _nonmodel_value(row, "episode_terminal_outcome")
            or _nonmodel_value(row, "terminal_outcome")
            or "unknown"
        )] += 1
        axes["physical_root"][_root_key(row, index)] += 1
        physical_root = _physical_root(row)
        if physical_root is None:
            target_tool_rows_missing_physical_root[tool] += 1
        else:
            target_tool_roots[tool].add(physical_root)
    return {
        "rows": len(materialized),
        **{axis: dict(sorted(counts.items())) for axis, counts in axes.items()},
        "target_tool_distinct_physical_roots": {
            tool: len(roots)
            for tool, roots in sorted(target_tool_roots.items())
        },
        "target_tool_rows_missing_physical_root": dict(
            sorted(target_tool_rows_missing_physical_root.items())
        ),
    }


def build_balanced_training_view(
    rows: Iterable[Mapping[str, Any]],
    *,
    size: int | None = None,
    seed: int = 0,
    tool_category_weights: Mapping[str, float] | None = None,
    max_duplicate_count: int = 2,
    max_rows_per_root: int | None = None,
    low_cost_margin_threshold: float = 0.05,
    maximum_tool_category_target_deviation: float = 0.10,
    minimum_tool_category_natural_rows: int = (
        DEFAULT_MINIMUM_TOOL_CATEGORY_NATURAL_ROWS
    ),
    minimum_tool_category_distinct_roots: int = (
        DEFAULT_MINIMUM_TOOL_CATEGORY_DISTINCT_ROOTS
    ),
    target_tool_minimum_distinct_roots: Mapping[str, int] | None = None,
    target_tool_state_class_minimum_distinct_roots: (
        Mapping[str, Mapping[str, int]] | None
    ) = None,
    target_tool_scenario_family_minimum_distinct_roots: (
        Mapping[str, Mapping[str, int]] | None
    ) = None,
    root_group_minimum_distinct_roots: Mapping[str, int] | None = None,
    example_root_group_memberships: (
        Mapping[str, Iterable[str]] | None
    ) = None,
    same_root_prerequisite_rules: Mapping[str, Mapping[str, Any]] | None = None,
    require_production_label_eligible: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build a deterministic multi-axis balanced SFT training view.

    The immutable aggregate remains untouched. Known low-margin rows are
    excluded from single-label SFT, while state class, target tool/category,
    scenario family, error cardinality, terminal outcome, and physical-root
    contribution jointly drive the greedy deficit sampler. Every axis uses
    deterministic capacity-aware targets. Tool-category targets retain a
    strict achieved-deviation gate after capacity adjustment, and configured
    nonzero categories must independently satisfy natural target-bearing row
    and distinct-root support floors. Configured exact target tools and
    tool-by-state/family cells and caller-defined per-example root groups must
    meet independent physical-root floors in both the eligible natural source
    and the returned view; duplicated placements never increase that support.
    Configured correction
    prerequisites must be retained on the same physical root in both the
    natural source and returned view. Explicitly production-ineligible rows are
    always excluded.
    """
    source = [dict(row) for row in rows]
    if not source:
        raise ValueError("cannot build a training view from an empty aggregate")
    if max_duplicate_count < 1:
        raise ValueError("max_duplicate_count must be positive")
    if not 0.0 <= float(maximum_tool_category_target_deviation) <= 1.0:
        raise ValueError(
            "maximum_tool_category_target_deviation must be between 0 and 1"
        )
    for value, name in (
        (
            minimum_tool_category_natural_rows,
            "minimum_tool_category_natural_rows",
        ),
        (
            minimum_tool_category_distinct_roots,
            "minimum_tool_category_distinct_roots",
        ),
    ):
        if isinstance(value, bool) or int(value) != value or int(value) < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if not isinstance(require_production_label_eligible, bool):
        raise ValueError("require_production_label_eligible must be boolean")
    if (
        target_tool_minimum_distinct_roots is not None
        and not isinstance(target_tool_minimum_distinct_roots, Mapping)
    ):
        raise ValueError(
            "target_tool_minimum_distinct_roots must be a mapping"
        )
    configured_target_tool_root_floors: dict[str, int] = {}
    for raw_tool, raw_floor in (
        {} if target_tool_minimum_distinct_roots is None
        else target_tool_minimum_distinct_roots
    ).items():
        tool = str(raw_tool).strip()
        if not tool:
            raise ValueError(
                "target_tool_minimum_distinct_roots keys must be non-empty"
            )
        if (
            isinstance(raw_floor, bool)
            or int(raw_floor) != raw_floor
            or int(raw_floor) <= 0
        ):
            raise ValueError(
                "target_tool_minimum_distinct_roots values must be positive "
                "integers"
            )
        configured_target_tool_root_floors[tool] = int(raw_floor)
    configured_target_tool_state_class_root_floors = (
        _normalize_joint_root_floors(
            target_tool_state_class_minimum_distinct_roots,
            name="target_tool_state_class_minimum_distinct_roots",
        )
    )
    configured_target_tool_scenario_family_root_floors = (
        _normalize_joint_root_floors(
            target_tool_scenario_family_minimum_distinct_roots,
            name="target_tool_scenario_family_minimum_distinct_roots",
        )
    )
    if root_group_minimum_distinct_roots is None:
        configured_root_group_floors: dict[str, int] = {}
    else:
        configured_root_group_floors = _normalize_dagger1_root_floors(
            root_group_minimum_distinct_roots,
            name="root_group_minimum_distinct_roots",
        )
    if (
        example_root_group_memberships is not None
        and not isinstance(example_root_group_memberships, Mapping)
    ):
        raise ValueError("example_root_group_memberships must be a mapping")
    configured_example_root_groups: dict[str, frozenset[str]] = {}
    for raw_example_id, raw_groups in (
        {}
        if example_root_group_memberships is None
        else example_root_group_memberships
    ).items():
        example_id = str(raw_example_id).strip()
        if not example_id:
            raise ValueError(
                "example_root_group_memberships keys must be non-empty"
            )
        if isinstance(raw_groups, (str, bytes)) or not isinstance(
            raw_groups, Iterable
        ):
            raise ValueError(
                "example_root_group_memberships values must be iterables of "
                "group names"
            )
        groups = frozenset(str(group).strip() for group in raw_groups)
        if any(not group for group in groups):
            raise ValueError("root-group names must be non-empty")
        unknown_groups = sorted(groups - set(configured_root_group_floors))
        if unknown_groups:
            raise ValueError(
                "example_root_group_memberships references groups without "
                "configured floors: " + ", ".join(unknown_groups)
            )
        configured_example_root_groups[example_id] = groups
    if configured_root_group_floors and example_root_group_memberships is None:
        raise ValueError(
            "root_group_minimum_distinct_roots requires "
            "example_root_group_memberships"
        )
    configured_same_root_prerequisite_rules = (
        _normalize_same_root_prerequisite_rules(same_root_prerequisite_rules)
    )

    explicitly_ineligible_rows = [
        row for row in source if _production_label_eligibility(row) is False
    ]
    explicitly_eligible_rows = [
        row for row in source if _production_label_eligibility(row) is True
    ]
    missing_eligibility_rows = [
        row
        for row in source
        if _production_label_eligibility(row) is None
    ]
    production_source = [
        row
        for row in source
        if _production_label_eligibility(row) is not False
        and (
            not require_production_label_eligible
            or _production_label_eligibility(row) is True
        )
    ]
    if not production_source:
        raise ValueError(
            "no production-label-eligible rows remain for the training view"
        )
    targetless_rows = [
        row for row in production_source if _target_tool(row) is None
    ]
    target_bearing_source = [
        row for row in production_source if _target_tool(row) is not None
    ]
    excluded_low_margin = [
        row
        for row in target_bearing_source
        if (margin := _known_cost_margin(row)) is not None
        and margin <= float(low_cost_margin_threshold)
    ]
    eligible = [
        row for row in target_bearing_source if row not in excluded_low_margin
    ]
    if not eligible:
        if targetless_rows and not target_bearing_source:
            raise ValueError("all training rows are targetless")
        raise ValueError(
            "all target-bearing training rows were excluded by the cost-margin gate"
        )
    root_group_memberships_by_index: dict[int, frozenset[str]] = {}
    if configured_root_group_floors:
        eligible_example_ids: set[str] = set()
        for index, row in enumerate(eligible):
            example_id = str(row.get("example_id") or "").strip()
            if not example_id:
                raise ValueError(
                    "root-group reservation requires every eligible row to "
                    "have an example_id"
                )
            if example_id in eligible_example_ids:
                raise ValueError(
                    "root-group reservation requires unique eligible example_id "
                    f"values; duplicate {example_id!r}"
                )
            eligible_example_ids.add(example_id)
            root_group_memberships_by_index[index] = (
                configured_example_root_groups.get(example_id, frozenset())
            )
    else:
        root_group_memberships_by_index = {
            index: frozenset() for index in range(len(eligible))
        }
    (
        natural_target_tool_unique_root_support,
        natural_target_tool_unique_root_shortfalls,
    ) = _target_tool_unique_root_support(
        eligible,
        minimum_distinct_roots=configured_target_tool_root_floors,
    )
    (
        natural_target_tool_state_class_unique_root_support,
        natural_target_tool_state_class_unique_root_shortfalls,
    ) = _target_tool_joint_unique_root_support(
        eligible,
        dimension="state_class",
        minimum_distinct_roots=(
            configured_target_tool_state_class_root_floors
        ),
    )
    (
        natural_target_tool_scenario_family_unique_root_support,
        natural_target_tool_scenario_family_unique_root_shortfalls,
    ) = _target_tool_joint_unique_root_support(
        eligible,
        dimension="scenario_family",
        minimum_distinct_roots=(
            configured_target_tool_scenario_family_root_floors
        ),
    )
    (
        natural_same_root_prerequisite_support,
        natural_same_root_prerequisite_shortfalls,
    ) = _same_root_prerequisite_support(
        eligible,
        rules=configured_same_root_prerequisite_rules,
    )
    requested_size = len(eligible) if size is None else int(size)
    if requested_size <= 0:
        raise ValueError("training-view size must be positive")
    if requested_size > len(eligible) * int(max_duplicate_count):
        raise ValueError("training-view size exceeds duplicate-limited capacity")

    category_weights = dict(
        DEFAULT_TRAINING_TOOL_CATEGORY_WEIGHTS
        if tool_category_weights is None
        else tool_category_weights
    )
    if any(value < 0 for value in category_weights.values()) or not any(
        value > 0 for value in category_weights.values()
    ):
        raise ValueError("tool-category weights must contain positive mass")
    configured_category_weights = {
        category: float(weight)
        for category, weight in sorted(category_weights.items())
        if float(weight) > 0.0
    }

    axis_values: dict[int, dict[str, str]] = {}
    roots: dict[int, str] = {}
    explicit_physical_roots: dict[int, str | None] = {}
    for index, row in enumerate(eligible):
        tool = _target_tool(row)
        state_class = _state_class(row)
        if state_class not in DEFAULT_REPLAY_WEIGHTS:
            raise ValueError(f"Unknown training-view state class: {state_class}")
        axis_values[index] = {
            "state_class": state_class,
            "target_tool": tool,
            "tool_category": _tool_category(tool),
            "scenario_family": str(_nonmodel_value(row, "scenario_family") or "unknown"),
            "error_cardinality": str(_nonmodel_value(row, "error_cardinality") or 0),
            "terminal_outcome": str(
                _nonmodel_value(row, "episode_terminal_outcome")
                or _nonmodel_value(row, "terminal_outcome")
                or "unknown"
            ),
        }
        roots[index] = _root_key(row, index)
        explicit_physical_roots[index] = _physical_root(row)

    def root_group_support(
        selected_indices: Iterable[int],
    ) -> tuple[
        dict[str, dict[str, int | bool]],
        dict[str, dict[str, int]],
    ]:
        row_counts: Counter[str] = Counter()
        group_roots: dict[str, set[str]] = defaultdict(set)
        missing_root_rows: Counter[str] = Counter()
        for selected_index in selected_indices:
            for group in root_group_memberships_by_index[selected_index]:
                row_counts[group] += 1
                root = explicit_physical_roots[selected_index]
                if root is None:
                    missing_root_rows[group] += 1
                else:
                    group_roots[group].add(root)
        support: dict[str, dict[str, int | bool]] = {}
        shortfalls: dict[str, dict[str, int]] = {}
        for group, floor in configured_root_group_floors.items():
            observed_roots = len(group_roots[group])
            missing_rows = int(missing_root_rows[group])
            root_shortfall = max(int(floor) - observed_roots, 0)
            passed = root_shortfall == 0 and missing_rows == 0
            support[group] = {
                "target_bearing_rows": int(row_counts[group]),
                "distinct_physical_roots": observed_roots,
                "rows_missing_physical_root": missing_rows,
                "minimum_distinct_physical_roots": int(floor),
                "root_shortfall": root_shortfall,
                "passed": passed,
            }
            if not passed:
                shortfalls[group] = {
                    "target_bearing_rows": int(row_counts[group]),
                    "distinct_physical_roots": observed_roots,
                    "rows_missing_physical_root": missing_rows,
                    "minimum_distinct_physical_roots": int(floor),
                    "root_shortfall": root_shortfall,
                }
        return support, shortfalls

    (
        natural_root_group_unique_root_support,
        natural_root_group_unique_root_shortfalls,
    ) = root_group_support(range(len(eligible)))

    prerequisite_key_by_target_index: dict[int, tuple[str, str, str] | None] = {}
    prerequisite_keys_by_index: dict[int, set[tuple[str, str, str]]] = {
        index: set() for index in range(len(eligible))
    }
    prerequisite_indices_by_key: dict[tuple[str, str, str], list[int]] = (
        defaultdict(list)
    )
    for target_tool, rule in configured_same_root_prerequisite_rules.items():
        for family, options in rule["prerequisite_options_by_family"].items():
            for index, values in axis_values.items():
                if values["scenario_family"] != family:
                    continue
                root = explicit_physical_roots[index]
                if values["target_tool"] == target_tool:
                    prerequisite_key_by_target_index[index] = (
                        (target_tool, family, root) if root is not None else None
                    )
                if root is not None and any(
                    _row_satisfies_prerequisite_option(eligible[index], option)
                    for option in options
                ):
                    key = (target_tool, family, root)
                    prerequisite_keys_by_index[index].add(key)
                    prerequisite_indices_by_key[key].append(index)

    available_categories = {
        values["tool_category"] for values in axis_values.values()
    }
    missing_weight = sorted(available_categories - set(category_weights))
    if missing_weight:
        raise ValueError(
            "Missing tool-category training weights: " + ", ".join(missing_weight)
        )
    zero_mass_categories = sorted(
        category
        for category in available_categories
        if float(category_weights[category]) <= 0.0
    )
    if zero_mass_categories:
        raise ValueError(
            "Available tool categories must receive positive training-weight mass: "
            + ", ".join(zero_mass_categories)
        )

    natural_category_rows: Counter[str] = Counter()
    natural_category_roots: dict[str, set[str]] = defaultdict(set)
    for index, row in enumerate(target_bearing_source):
        tool = _target_tool(row)
        if tool is None:  # Defensive: target-bearing rows were filtered above.
            continue
        category = _tool_category(tool)
        natural_category_rows[category] += 1
        natural_category_roots[category].add(_root_key(row, index))
    tool_category_natural_support: dict[str, dict[str, int | bool]] = {}
    tool_category_natural_support_shortfalls: dict[
        str, dict[str, int]
    ] = {}
    for category in configured_category_weights:
        natural_rows = int(natural_category_rows[category])
        distinct_roots = len(natural_category_roots[category])
        row_shortfall = max(
            int(minimum_tool_category_natural_rows) - natural_rows, 0
        )
        root_shortfall = max(
            int(minimum_tool_category_distinct_roots) - distinct_roots, 0
        )
        tool_category_natural_support[category] = {
            "natural_target_bearing_rows": natural_rows,
            "distinct_roots": distinct_roots,
            "minimum_natural_target_bearing_rows": int(
                minimum_tool_category_natural_rows
            ),
            "minimum_distinct_roots": int(
                minimum_tool_category_distinct_roots
            ),
            "row_shortfall": row_shortfall,
            "root_shortfall": root_shortfall,
            "passed": row_shortfall == 0 and root_shortfall == 0,
        }
        if row_shortfall or root_shortfall:
            tool_category_natural_support_shortfalls[category] = {
                "natural_target_bearing_rows": natural_rows,
                "distinct_roots": distinct_roots,
                "row_shortfall": row_shortfall,
                "root_shortfall": root_shortfall,
            }
    root_capacities = Counter(roots.values())
    root_capacities = Counter(
        {root: count * int(max_duplicate_count) for root, count in root_capacities.items()}
    )
    root_cap = max_rows_per_root
    if root_cap is None:
        root_cap = _minimum_feasible_root_cap(
            size=requested_size, root_capacities=root_capacities
        )
    if sum(min(int(root_cap), count) for count in root_capacities.values()) < requested_size:
        raise ValueError("max_rows_per_root cannot satisfy the requested training view")

    secondary_axes = (
        "state_class",
        "target_tool",
        "scenario_family",
        "error_cardinality",
        "terminal_outcome",
    )
    axis_capacities: dict[str, dict[str, int]] = {}
    for axis in ("tool_category", *secondary_axes):
        capacities: dict[str, int] = {}
        values = (
            sorted(configured_category_weights)
            if axis == "tool_category"
            else sorted({item[axis] for item in axis_values.values()})
        )
        for value in values:
            capacity_by_root: Counter[str] = Counter(
                roots[index]
                for index, values in axis_values.items()
                if values[axis] == value
            )
            capacities[value] = min(
                requested_size,
                sum(
                    min(
                        int(root_cap),
                        int(count) * int(max_duplicate_count),
                    )
                    for count in capacity_by_root.values()
                ),
            )
        axis_capacities[axis] = capacities

    unconstrained_category_targets = _allocate_counts(
        requested_size, configured_category_weights
    )
    adjusted_category_targets, category_adjustments = _capacity_aware_counts(
        size=requested_size,
        weights=configured_category_weights,
        capacities=axis_capacities["tool_category"],
    )
    unconstrained_target_counts: dict[str, dict[str, int]] = {
        "tool_category": unconstrained_category_targets
    }
    target_counts: dict[str, dict[str, int]] = {
        "tool_category": adjusted_category_targets
    }
    capacity_adjustments: dict[str, dict[str, dict[str, int]]] = {}
    if category_adjustments:
        capacity_adjustments["tool_category"] = category_adjustments
    for axis in secondary_axes:
        values = sorted(axis_capacities[axis])
        weights = {value: 1.0 for value in values}
        unconstrained_target_counts[axis] = _allocate_counts(
            requested_size, weights
        )
        adjusted, adjustments = _capacity_aware_counts(
            size=requested_size,
            weights=weights,
            capacities=axis_capacities[axis],
        )
        target_counts[axis] = adjusted
        if adjustments:
            capacity_adjustments[axis] = adjustments

    necessary_feasibility_shortfalls: dict[str, dict[str, dict[str, int]]] = {}
    for axis, targets in target_counts.items():
        axis_shortfalls: dict[str, dict[str, int]] = {}
        for value, target in targets.items():
            maximum_achievable = axis_capacities[axis][value]
            shortfall = max(int(target) - int(maximum_achievable), 0)
            if shortfall:
                axis_shortfalls[value] = {
                    "target": int(target),
                    "maximum_achievable": int(maximum_achievable),
                    "shortfall": int(shortfall),
                }
        if axis_shortfalls:
            necessary_feasibility_shortfalls[axis] = dict(
                sorted(axis_shortfalls.items())
            )

    selected: list[int] = []
    occurrences: Counter[int] = Counter()
    root_counts: Counter[str] = Counter()
    observed: dict[str, Counter[str]] = {
        axis: Counter() for axis in target_counts
    }
    tie_break = {
        index: int.from_bytes(
            hashlib.sha256(
                f"{seed}:{eligible[index].get('example_id', index)}".encode("utf-8")
            ).digest()[:8],
            "big",
        )
        for index in range(len(eligible))
    }
    requirement_floors: dict[tuple[str, str, str], int] = {
        ("target_tool", tool, ""): floor
        for tool, floor in configured_target_tool_root_floors.items()
    }
    requirement_floors.update(
        {
            ("state_class", tool, state_class): floor
            for tool, state_classes in (
                configured_target_tool_state_class_root_floors.items()
            )
            for state_class, floor in state_classes.items()
        }
    )
    requirement_floors.update(
        {
            ("scenario_family", tool, family): floor
            for tool, families in (
                configured_target_tool_scenario_family_root_floors.items()
            )
            for family, floor in families.items()
        }
    )
    requirement_floors.update(
        {
            ("root_group", group, ""): floor
            for group, floor in configured_root_group_floors.items()
        }
    )

    def matches_requirement(
        index: int,
        requirement: tuple[str, str, str],
    ) -> bool:
        axis, tool, value = requirement
        if axis == "root_group":
            return tool in root_group_memberships_by_index[index]
        values = axis_values[index]
        return values["target_tool"] == tool and (
            axis == "target_tool" or values[axis] == value
        )

    matching_requirements: dict[int, set[tuple[str, str, str]]] = {
        index: {
            requirement
            for requirement in requirement_floors
            if matches_requirement(index, requirement)
        }
        for index in range(len(eligible))
    }
    natural_requirement_roots: dict[
        tuple[str, str, str], set[str]
    ] = {
        requirement: {
            root
            for index, root in explicit_physical_roots.items()
            if root is not None
            and requirement in matching_requirements[index]
        }
        for requirement in requirement_floors
    }
    reservation_targets = {
        requirement: min(
            int(floor),
            len(natural_requirement_roots[requirement]),
        )
        for requirement, floor in requirement_floors.items()
    }
    reservation_requirement_roots: dict[
        tuple[str, str, str], set[str]
    ] = {
        requirement: set() for requirement in requirement_floors
    }
    reserved_indices: list[int] = []
    selected_prerequisite_keys: set[tuple[str, str, str]] = set()

    def candidate_score(index: int) -> tuple[float, int]:
        values = axis_values[index]
        secondary_score = 0.0
        for axis in secondary_axes:
            targets = target_counts[axis]
            value = values[axis]
            target = max(int(targets.get(value, 0)), 1)
            deficit = target - observed[axis][value]
            secondary_score += deficit / target
        secondary_score -= occurrences[index] * 0.5
        secondary_score -= root_counts[roots[index]] / max(int(root_cap), 1)
        return secondary_score, -tie_break[index]

    def selection_bundle(index: int) -> tuple[int, ...] | None:
        """Return prerequisite-first indices for one selectable target.

        The first selected correction on a governed root consumes a context
        row in the same atomic choice.  Once that prerequisite is present,
        correction duplicates may be selected normally.  This makes it
        impossible for oversampling to amplify a correction before the model
        has any same-root context target.
        """

        if occurrences[index] >= int(max_duplicate_count):
            return None
        if root_counts[roots[index]] >= int(root_cap):
            return None
        if index not in prerequisite_key_by_target_index:
            return (index,)
        key = prerequisite_key_by_target_index[index]
        if key is None:
            return None
        if key in selected_prerequisite_keys:
            return (index,)
        prerequisite_candidates = [
            prerequisite_index
            for prerequisite_index in prerequisite_indices_by_key.get(key, [])
            if occurrences[prerequisite_index] < int(max_duplicate_count)
            and root_counts[roots[prerequisite_index]] < int(root_cap)
        ]
        if not prerequisite_candidates:
            return None
        prerequisite_index = max(prerequisite_candidates, key=candidate_score)
        bundle = (prerequisite_index, index)
        added_by_root = Counter(roots[item] for item in bundle)
        if any(
            root_counts[root] + additional > int(root_cap)
            for root, additional in added_by_root.items()
        ):
            return None
        if len(selected) + len(bundle) > requested_size:
            return None
        return bundle

    def remaining_category_capacity(category: str) -> int:
        capacity_by_root: Counter[str] = Counter()
        for index, values in axis_values.items():
            if values["tool_category"] != category:
                continue
            if selection_bundle(index) is None:
                continue
            remaining_occurrences = int(max_duplicate_count) - occurrences[index]
            if remaining_occurrences > 0:
                capacity_by_root[roots[index]] += remaining_occurrences
        return sum(
            min(
                max(int(root_cap) - root_counts[root], 0),
                available,
            )
            for root, available in capacity_by_root.items()
        )

    def select_index(index: int, *, reserved: bool) -> None:
        selected.append(index)
        occurrences[index] += 1
        root_counts[roots[index]] += 1
        for axis, value in axis_values[index].items():
            observed[axis][value] += 1
        explicit_root = explicit_physical_roots[index]
        if explicit_root is not None:
            for requirement in matching_requirements[index]:
                reservation_requirement_roots[requirement].add(
                    explicit_root
                )
        selected_prerequisite_keys.update(prerequisite_keys_by_index[index])
        if reserved:
            reserved_indices.append(index)

    def select_bundle(index: int, *, reserved: bool) -> bool:
        bundle = selection_bundle(index)
        if bundle is None:
            return False
        for bundled_index in bundle:
            select_index(bundled_index, reserved=reserved)
        return True

    def reservation_blocking_pressure(index: int) -> int:
        """Count unmet requirements that would lose a necessary root slot."""
        root = explicit_physical_roots[index]
        if root is None or root_counts[roots[index]] + 1 < int(root_cap):
            return 0
        blocked = 0
        for requirement, target in reservation_targets.items():
            selected_roots = reservation_requirement_roots[requirement]
            remaining_need = int(target) - len(selected_roots)
            if remaining_need <= 0 or root in selected_roots:
                continue
            if requirement in matching_requirements[index]:
                # Selecting this row consumes the slot but also satisfies this
                # requirement on the root.
                continue
            available_roots = {
                candidate_root
                for candidate_root in natural_requirement_roots[requirement]
                if candidate_root not in selected_roots
                and root_counts[candidate_root] < int(root_cap)
            }
            if root in available_roots and len(available_roots) <= remaining_need:
                blocked += 1
        return blocked

    requirement_order = sorted(
        requirement_floors,
        key=lambda requirement: (
            len(natural_requirement_roots[requirement])
            < int(requirement_floors[requirement]),
            max(
                len(natural_requirement_roots[requirement])
                - int(reservation_targets[requirement]),
                0,
            ),
            len(natural_requirement_roots[requirement]),
            requirement,
        ),
    )
    for requirement in requirement_order:
        target = int(reservation_targets[requirement])
        while (
            len(reservation_requirement_roots[requirement]) < target
            and len(selected) < requested_size
        ):
            candidates = [
                index
                for index in range(len(eligible))
                if requirement in matching_requirements[index]
                and explicit_physical_roots[index] is not None
                and explicit_physical_roots[index]
                not in reservation_requirement_roots[requirement]
                and occurrences[index] < int(max_duplicate_count)
                and root_counts[roots[index]] < int(root_cap)
                and selection_bundle(index) is not None
            ]
            if not candidates:
                break

            def reservation_score(
                index: int,
            ) -> tuple[int, int, int, float, int]:
                root = explicit_physical_roots[index]
                newly_covered = [
                    candidate_requirement
                    for candidate_requirement in matching_requirements[index]
                    if root
                    not in reservation_requirement_roots[
                        candidate_requirement
                    ]
                    and len(
                        reservation_requirement_roots[
                            candidate_requirement
                        ]
                    )
                    < int(reservation_targets[candidate_requirement])
                ]
                feasible_coverage = sum(
                    len(natural_requirement_roots[item])
                    >= int(requirement_floors[item])
                    for item in newly_covered
                )
                balance_score, deterministic_tie_break = candidate_score(
                    index
                )
                return (
                    -reservation_blocking_pressure(index),
                    feasible_coverage,
                    len(newly_covered),
                    balance_score,
                    deterministic_tie_break,
                )

            chosen = max(candidates, key=reservation_score)
            if not select_bundle(chosen, reserved=True):
                raise ValueError(
                    "same-root prerequisite bundle became infeasible during "
                    "requirement reservation"
                )

    reserved_requirement_roots = {
        requirement: set(roots_for_requirement)
        for requirement, roots_for_requirement in (
            reservation_requirement_roots.items()
        )
    }

    while len(selected) < requested_size:
        category_options: list[tuple[str, int, int]] = []
        for category, target in target_counts["tool_category"].items():
            remaining_target = int(target) - observed["tool_category"][category]
            if remaining_target <= 0:
                continue
            remaining_capacity = remaining_category_capacity(category)
            if remaining_capacity > 0:
                category_options.append(
                    (category, remaining_target, remaining_capacity)
                )
        if category_options:
            chosen_category, _, _ = max(
                category_options,
                key=lambda item: (
                    item[1] / item[2],
                    item[1]
                    / max(int(target_counts["tool_category"][item[0]]), 1),
                    item[0],
                ),
            )
            candidates = [
                index
                for index, values in axis_values.items()
                if values["tool_category"] == chosen_category
                and occurrences[index] < int(max_duplicate_count)
                and root_counts[roots[index]] < int(root_cap)
                and selection_bundle(index) is not None
            ]
        else:
            # A marginally capacity-aware target can still become jointly
            # infeasible after physical-root caps interact across categories.
            # Finish the requested view deterministically and let the explicit
            # adjusted-target deviation gate fail closed in the report.
            candidates = [
                index
                for index in range(len(eligible))
                if occurrences[index] < int(max_duplicate_count)
                and root_counts[roots[index]] < int(root_cap)
                and selection_bundle(index) is not None
            ]
        if not candidates:
            raise ValueError("training-view constraints exhausted before reaching requested size")
        chosen = max(candidates, key=candidate_score)
        if not select_bundle(chosen, reserved=False):
            raise ValueError(
                "same-root prerequisite bundle became infeasible during balancing"
            )

    # Stable hash ordering makes the persisted view byte-reproducible without
    # retaining curriculum-like blocks from greedy selection.
    selected.sort(
        key=lambda index: hashlib.sha256(
            f"view:{seed}:{eligible[index].get('example_id', index)}:{occurrences[index]}".encode(
                "utf-8"
            )
        ).hexdigest()
    )
    view = [copy.deepcopy(eligible[index]) for index in selected]
    (
        training_view_root_group_unique_root_support,
        training_view_root_group_unique_root_shortfalls,
    ) = root_group_support(selected)
    (
        training_view_target_tool_unique_root_support,
        training_view_target_tool_unique_root_shortfalls,
    ) = _target_tool_unique_root_support(
        view,
        minimum_distinct_roots=configured_target_tool_root_floors,
    )
    (
        training_view_target_tool_state_class_unique_root_support,
        training_view_target_tool_state_class_unique_root_shortfalls,
    ) = _target_tool_joint_unique_root_support(
        view,
        dimension="state_class",
        minimum_distinct_roots=(
            configured_target_tool_state_class_root_floors
        ),
    )
    (
        training_view_target_tool_scenario_family_unique_root_support,
        training_view_target_tool_scenario_family_unique_root_shortfalls,
    ) = _target_tool_joint_unique_root_support(
        view,
        dimension="scenario_family",
        minimum_distinct_roots=(
            configured_target_tool_scenario_family_root_floors
        ),
    )
    (
        training_view_same_root_prerequisite_support,
        training_view_same_root_prerequisite_shortfalls,
    ) = _same_root_prerequisite_support(
        view,
        rules=configured_same_root_prerequisite_rules,
    )
    achieved_counts = {
        axis: dict(sorted(counts.items())) for axis, counts in observed.items()
    }
    target_deviation: dict[str, dict[str, Any]] = {}
    for axis, targets in target_counts.items():
        achieved = observed[axis]
        value_deviations: dict[str, dict[str, float | int]] = {}
        total_shortfall = 0
        total_excess = 0
        for value in sorted(set(targets) | set(achieved)):
            target = int(targets.get(value, 0))
            actual = int(achieved.get(value, 0))
            shortfall = max(target - actual, 0)
            excess = max(actual - target, 0)
            absolute_deviation = abs(actual - target)
            relative_deviation = absolute_deviation / max(target, 1)
            total_shortfall += shortfall
            total_excess += excess
            value_deviations[value] = {
                "target": target,
                "achieved": actual,
                "shortfall": shortfall,
                "excess": excess,
                "absolute_deviation": absolute_deviation,
                # A target of zero uses a denominator of one, so an unexpected
                # selected row remains finite JSON and fails a 10% release gate.
                "relative_deviation": relative_deviation,
            }
        total_absolute_deviation = total_shortfall + total_excess
        target_deviation[axis] = {
            "values": value_deviations,
            "target_total": sum(int(value) for value in targets.values()),
            "achieved_total": sum(int(value) for value in achieved.values()),
            "total_shortfall": total_shortfall,
            "total_excess": total_excess,
            "total_absolute_deviation": total_absolute_deviation,
            "relative_deviation": total_absolute_deviation
            / max(sum(int(value) for value in targets.values()), 1),
            "maximum_value_relative_deviation": max(
                (
                    float(details["relative_deviation"])
                    for details in value_deviations.values()
                ),
                default=0.0,
            ),
        }
    achieved_tool_category_target_deviation = float(
        target_deviation["tool_category"]["maximum_value_relative_deviation"]
    )
    feasibility_shortfall_total = sum(
        details["shortfall"]
        for axis_shortfalls in necessary_feasibility_shortfalls.values()
        for details in axis_shortfalls.values()
    )
    capacity_adjustment_total = sum(
        details["necessary_reduction"]
        for axis_adjustments in capacity_adjustments.values()
        for details in axis_adjustments.values()
    )
    passed = (
        not tool_category_natural_support_shortfalls
        and not natural_target_tool_unique_root_shortfalls
        and not training_view_target_tool_unique_root_shortfalls
        and not natural_target_tool_state_class_unique_root_shortfalls
        and not training_view_target_tool_state_class_unique_root_shortfalls
        and not natural_target_tool_scenario_family_unique_root_shortfalls
        and not training_view_target_tool_scenario_family_unique_root_shortfalls
        and not natural_root_group_unique_root_shortfalls
        and not training_view_root_group_unique_root_shortfalls
        and not natural_same_root_prerequisite_shortfalls
        and not training_view_same_root_prerequisite_shortfalls
        and feasibility_shortfall_total == 0
        and achieved_tool_category_target_deviation
        <= float(maximum_tool_category_target_deviation)
    )
    reservation_requirement_report = []
    for requirement in sorted(requirement_floors):
        axis, tool, value = requirement
        configured_floor = int(requirement_floors[requirement])
        natural_roots = len(natural_requirement_roots[requirement])
        reserved_roots = len(reserved_requirement_roots[requirement])
        selected_roots = len(reservation_requirement_roots[requirement])
        reservation_requirement_report.append(
            {
                "axis": axis,
                **(
                    {"root_group": tool}
                    if axis == "root_group"
                    else {
                        "target_tool": tool,
                        **({"value": value} if value else {}),
                    }
                ),
                "minimum_distinct_physical_roots": configured_floor,
                "natural_distinct_physical_roots": natural_roots,
                "natural_support_feasible": natural_roots >= configured_floor,
                "reservation_target_distinct_physical_roots": int(
                    reservation_targets[requirement]
                ),
                "reserved_distinct_physical_roots": reserved_roots,
                "reservation_shortfall": max(
                    int(reservation_targets[requirement]) - reserved_roots,
                    0,
                ),
                "selected_distinct_physical_roots": selected_roots,
                "selected_root_shortfall": max(
                    configured_floor - selected_roots,
                    0,
                ),
            }
        )
    if configured_same_root_prerequisite_rules and configured_root_group_floors:
        reservation_policy = (
            "constrained_first_with_same_root_prerequisites_and_root_groups_v3"
        )
    elif configured_same_root_prerequisite_rules:
        reservation_policy = "constrained_first_with_same_root_prerequisites_v2"
    elif configured_root_group_floors:
        reservation_policy = "constrained_first_generic_root_group_preselection_v2"
    else:
        reservation_policy = (
            "constrained_first_distinct_physical_root_preselection_v1"
        )
    report = {
        "seed": int(seed),
        "requested_size": requested_size,
        "returned_size": len(view),
        "input_rows": len(source),
        "training_view_candidate_input_rows": len(production_source),
        "explicitly_production_eligible_input_rows": len(
            explicitly_eligible_rows
        ),
        "explicitly_production_ineligible_input_rows": len(
            explicitly_ineligible_rows
        ),
        "missing_production_label_eligibility_input_rows": len(
            missing_eligibility_rows
        ),
        "require_production_label_eligible": require_production_label_eligible,
        "target_bearing_input_rows": len(target_bearing_source),
        "excluded_targetless_rows": len(targetless_rows),
        "low_cost_margin_threshold": float(low_cost_margin_threshold),
        "excluded_low_margin_rows": len(excluded_low_margin),
        "max_duplicate_count": int(max_duplicate_count),
        "duplicate_occurrences": sum(max(count - 1, 0) for count in occurrences.values()),
        "max_rows_per_root": int(root_cap),
        "target_contract": {
            "size_policy": "requested_full_size_with_bounded_replacement",
            "strict_target_axes": [
                axis
                for axis, enabled in (
                    (
                        "target_tool_distinct_physical_roots",
                        configured_target_tool_root_floors,
                    ),
                    (
                        "target_tool_x_state_class_distinct_physical_roots",
                        configured_target_tool_state_class_root_floors,
                    ),
                    (
                        "target_tool_x_scenario_family_distinct_physical_roots",
                        configured_target_tool_scenario_family_root_floors,
                    ),
                    (
                        "same_root_target_prerequisites",
                        configured_same_root_prerequisite_rules,
                    ),
                    (
                        "root_group_distinct_physical_roots",
                        configured_root_group_floors,
                    ),
                )
                if enabled
            ],
            "deviation_gated_target_axes": ["tool_category"],
            "capacity_aware_target_axes": ["tool_category", *secondary_axes],
            "capacity_aware_policy": "weighted_then_clip_and_redistribute_v1",
            "requirement_aware_reservation_policy": reservation_policy,
            "tool_category_natural_support_floor": {
                "minimum_natural_target_bearing_rows": int(
                    minimum_tool_category_natural_rows
                ),
                "minimum_distinct_roots": int(
                    minimum_tool_category_distinct_roots
                ),
            },
            "target_tool_minimum_distinct_physical_roots": dict(
                sorted(configured_target_tool_root_floors.items())
            ),
            "target_tool_state_class_minimum_distinct_physical_roots": (
                configured_target_tool_state_class_root_floors
            ),
            "target_tool_scenario_family_minimum_distinct_physical_roots": (
                configured_target_tool_scenario_family_root_floors
            ),
            "same_root_prerequisite_rules": copy.deepcopy(
                configured_same_root_prerequisite_rules
            ),
            "root_group_minimum_distinct_physical_roots": dict(
                configured_root_group_floors
            ),
            "production_label_eligibility_policy": (
                "explicit_true_required"
                if require_production_label_eligible
                else "explicit_false_excluded"
            ),
        },
        "requirement_aware_reservation": {
            "policy": reservation_policy,
            "reserved_rows": len(reserved_indices),
            "feasible_requirements_satisfied_by_reservation": all(
                item["reservation_shortfall"] == 0
                for item in reservation_requirement_report
                if item["natural_support_feasible"]
            ),
            "requirements": reservation_requirement_report,
        },
        "configured_tool_category_weights": configured_category_weights,
        "tool_category_natural_support": tool_category_natural_support,
        "tool_category_natural_support_shortfalls": (
            tool_category_natural_support_shortfalls
        ),
        "tool_category_natural_support_passed": (
            not tool_category_natural_support_shortfalls
        ),
        "target_tool_unique_root_support": {
            "eligible_natural_source": (
                natural_target_tool_unique_root_support
            ),
            "training_view": training_view_target_tool_unique_root_support,
        },
        "target_tool_unique_root_shortfalls": {
            "eligible_natural_source": (
                natural_target_tool_unique_root_shortfalls
            ),
            "training_view": (
                training_view_target_tool_unique_root_shortfalls
            ),
        },
        "target_tool_unique_root_support_passed": (
            not natural_target_tool_unique_root_shortfalls
            and not training_view_target_tool_unique_root_shortfalls
        ),
        "root_group_unique_root_support": {
            "eligible_natural_source": natural_root_group_unique_root_support,
            "training_view": training_view_root_group_unique_root_support,
        },
        "root_group_unique_root_shortfalls": {
            "eligible_natural_source": natural_root_group_unique_root_shortfalls,
            "training_view": training_view_root_group_unique_root_shortfalls,
        },
        "root_group_unique_root_support_passed": (
            not natural_root_group_unique_root_shortfalls
            and not training_view_root_group_unique_root_shortfalls
        ),
        "same_root_prerequisite_support": {
            "eligible_natural_source": natural_same_root_prerequisite_support,
            "training_view": training_view_same_root_prerequisite_support,
        },
        "same_root_prerequisite_shortfalls": {
            "eligible_natural_source": natural_same_root_prerequisite_shortfalls,
            "training_view": training_view_same_root_prerequisite_shortfalls,
        },
        "same_root_prerequisite_support_passed": (
            not natural_same_root_prerequisite_shortfalls
            and not training_view_same_root_prerequisite_shortfalls
        ),
        "critical_joint_unique_root_support": {
            "target_tool_x_state_class": {
                "eligible_natural_source": (
                    natural_target_tool_state_class_unique_root_support
                ),
                "training_view": (
                    training_view_target_tool_state_class_unique_root_support
                ),
            },
            "target_tool_x_scenario_family": {
                "eligible_natural_source": (
                    natural_target_tool_scenario_family_unique_root_support
                ),
                "training_view": (
                    training_view_target_tool_scenario_family_unique_root_support
                ),
            },
        },
        "critical_joint_unique_root_shortfalls": {
            "target_tool_x_state_class": {
                "eligible_natural_source": (
                    natural_target_tool_state_class_unique_root_shortfalls
                ),
                "training_view": (
                    training_view_target_tool_state_class_unique_root_shortfalls
                ),
            },
            "target_tool_x_scenario_family": {
                "eligible_natural_source": (
                    natural_target_tool_scenario_family_unique_root_shortfalls
                ),
                "training_view": (
                    training_view_target_tool_scenario_family_unique_root_shortfalls
                ),
            },
        },
        "critical_joint_unique_root_support_passed": (
            not natural_target_tool_state_class_unique_root_shortfalls
            and not training_view_target_tool_state_class_unique_root_shortfalls
            and not natural_target_tool_scenario_family_unique_root_shortfalls
            and not training_view_target_tool_scenario_family_unique_root_shortfalls
        ),
        "unconstrained_target_counts": {
            axis: dict(sorted(counts.items()))
            for axis, counts in unconstrained_target_counts.items()
        },
        "target_counts": {
            axis: dict(sorted(counts.items())) for axis, counts in target_counts.items()
        },
        "capacity_adjustments": capacity_adjustments,
        "capacity_adjustment_total": capacity_adjustment_total,
        "achieved_counts": achieved_counts,
        "target_deviation": target_deviation,
        "necessary_feasibility_shortfalls": necessary_feasibility_shortfalls,
        "necessary_feasibility_shortfall_total": feasibility_shortfall_total,
        "maximum_tool_category_target_deviation": float(
            maximum_tool_category_target_deviation
        ),
        "achieved_tool_category_target_deviation": (
            achieved_tool_category_target_deviation
        ),
        "passed": passed,
        "release_ready": passed,
        "before": _training_view_summary(target_bearing_source),
        "after": _training_view_summary(view),
    }
    return view, report


def build_dagger1_training_view(
    d0_rows: Iterable[Mapping[str, Any]],
    d1_rows: Iterable[Mapping[str, Any]],
    *,
    size: int | None = None,
    seed: int = 0,
    d1_share: float = 0.25,
    minimum_d1_share: float = 0.20,
    maximum_d1_share: float = 0.30,
    max_duplicate_count: int = 2,
    max_rows_per_root: int = 8,
    d0_training_view_kwargs: Mapping[str, Any] | None = None,
    d1_training_view_kwargs: Mapping[str, Any] | None = None,
    d1_targeted_state_cell_minimum_distinct_roots: Mapping[str, int] = (
        DAGGER1_TARGETED_STATE_CELL_MINIMUM_DISTINCT_ROOTS
    ),
    d1_recovery_stratum_minimum_distinct_roots: Mapping[str, int] = (
        DAGGER1_RECOVERY_STRATUM_MINIMUM_DISTINCT_ROOTS
    ),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build and gate a deterministic production D0 union D1 view.

    The two sources are sampled independently to make the learner-recovery
    allocation exact, then deterministically interleaved.  Physical roots must
    be disjoint across sources, so enforcing the same per-root and per-example
    caps within each source also enforces them globally.  Before sampling, D1
    must independently satisfy the reviewed physical-root floors for every
    targeted recovery cell and required recovery stratum; replay duplication
    can therefore never manufacture that support.
    """
    d0 = [dict(row) for row in d0_rows]
    d1 = [dict(row) for row in d1_rows]
    if not d0 or not d1:
        raise ValueError("D0 and D1 must both contain at least one row")
    if not 0.0 < float(minimum_d1_share) <= float(maximum_d1_share) < 1.0:
        raise ValueError("D1 share band must satisfy 0 < minimum <= maximum < 1")
    if not float(minimum_d1_share) <= float(d1_share) <= float(maximum_d1_share):
        raise ValueError("configured D1 share is outside the allowed share band")
    if int(max_duplicate_count) != max_duplicate_count or int(max_duplicate_count) < 1:
        raise ValueError("max_duplicate_count must be a positive integer")
    if int(max_rows_per_root) != max_rows_per_root or int(max_rows_per_root) < 1:
        raise ValueError("max_rows_per_root must be a positive integer")

    source_roots: dict[str, set[str]] = {"d0": set(), "d1": set()}
    for source_name, rows in (("d0", d0), ("d1", d1)):
        for index, row in enumerate(rows):
            if row.get("production_label_eligible") is not True:
                raise ValueError(
                    f"{source_name.upper()} row {index} is not explicitly "
                    "production-label eligible"
                )
            root = _physical_root(row)
            if root is None:
                raise ValueError(
                    f"{source_name.upper()} row {index} lacks a physical root"
                )
            source_roots[source_name].add(root)
            if not str(row.get("example_id") or "").strip():
                raise ValueError(
                    f"{source_name.upper()} row {index} lacks an example_id"
                )
    overlap = sorted(source_roots["d0"] & source_roots["d1"])
    if overlap:
        raise ValueError(
            "D0 and D1 physical roots must be disjoint: " + ", ".join(overlap)
        )

    for index, row in enumerate(d1):
        labels = row.get("labels") if isinstance(row.get("labels"), Mapping) else {}
        try:
            beta = float(row.get("collection_beta", labels.get("collection_beta")))
            iteration = int(row.get("iteration"))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"D1 row {index} lacks a valid iteration/beta contract"
            ) from exc
        role = row.get("collection_role", labels.get("collection_role"))
        if role != "training" or not 0.25 <= beta <= 0.5 or iteration < 1:
            raise ValueError(
                f"D1 row {index} is not from an approved mixed-policy "
                "training collection"
            )
        if row.get("state_origin") != "learner_policy":
            raise ValueError(f"D1 row {index} is not learner-visited")
        if row.get("recovery_label_contract") != "observable_rank_one_learner_state_v1":
            raise ValueError(f"D1 row {index} lacks the recovery label contract")
        if row.get("collection_training_eligible") is not True:
            raise ValueError(
                f"D1 row {index} is not covered by a passing collection gate"
            )

    normalized_d1_cell_floors = _normalize_dagger1_root_floors(
        d1_targeted_state_cell_minimum_distinct_roots,
        name="d1_targeted_state_cell_minimum_distinct_roots",
    )
    normalized_d1_stratum_floors = _normalize_dagger1_root_floors(
        d1_recovery_stratum_minimum_distinct_roots,
        name="d1_recovery_stratum_minimum_distinct_roots",
    )
    d1_independent_root_support = audit_dagger1_independent_root_support(
        d1,
        targeted_state_cell_minimum_distinct_roots=(
            normalized_d1_cell_floors
        ),
        recovery_stratum_minimum_distinct_roots=(
            normalized_d1_stratum_floors
        ),
    )
    if d1_independent_root_support.get("passed") is not True:
        raise ValueError(
            "D1 independent physical-root support failed: "
            "targeted_cells="
            f"{d1_independent_root_support.get('targeted_state_cell_shortfalls')}, "
            "recovery_strata="
            f"{d1_independent_root_support.get('recovery_stratum_shortfalls')}"
        )
    d1_root_group_floors = {
        **{
            f"targeted_state_cell:{cell}": floor
            for cell, floor in normalized_d1_cell_floors.items()
        },
        **{
            f"recovery_stratum:{stratum}": floor
            for stratum, floor in normalized_d1_stratum_floors.items()
        },
    }
    d1_example_root_groups: dict[str, list[str]] = {}
    for row in d1:
        groups = {
            f"targeted_state_cell:{cell}"
            for cell in dagger1_targeted_state_cells(row)
            if cell in normalized_d1_cell_floors
        }
        stratum = _dagger1_recovery_stratum(row)
        if stratum in normalized_d1_stratum_floors:
            groups.add(f"recovery_stratum:{stratum}")
        d1_example_root_groups[str(row["example_id"])] = sorted(groups)

    total_size = len(d0) + len(d1) if size is None else int(size)
    if isinstance(size, bool) or total_size < 2 or (size is not None and total_size != size):
        raise ValueError("size must be an integer of at least two")
    d1_size = int(math.floor(total_size * float(d1_share) + 0.5))
    d0_size = total_size - d1_size
    if d0_size < 1 or d1_size < 1:
        raise ValueError("D0/D1 allocation must retain both sources")
    minimum_single_group_rows = max(d1_root_group_floors.values())
    if d1_size < minimum_single_group_rows:
        raise ValueError(
            "D1 allocation cannot preserve configured independent-root floors: "
            f"allocated_rows={d1_size}, minimum_single_group_floor="
            f"{minimum_single_group_rows}"
        )
    allocated_share = d1_size / total_size
    if not float(minimum_d1_share) <= allocated_share <= float(maximum_d1_share):
        raise ValueError(
            "integer D1 allocation falls outside the allowed share band; "
            "increase the requested size"
        )
    replay_capacity = dagger1_replay_capacity_report(
        d0,
        d1,
        size=total_size,
        d1_share=d1_share,
        minimum_d1_share=minimum_d1_share,
        maximum_d1_share=maximum_d1_share,
        max_duplicate_count=max_duplicate_count,
        max_rows_per_root=max_rows_per_root,
    )
    if replay_capacity.get("passed") is not True:
        largest = replay_capacity.get("largest_feasible") or {}
        requested = replay_capacity.get("requested") or {}
        raise ValueError(
            "requested D0/D1 training view exceeds duplicate/root-limited "
            "capacity: "
            f"requested={requested}, largest_feasible={largest}"
        )

    blocked_kwargs = {
        "size",
        "seed",
        "max_duplicate_count",
        "max_rows_per_root",
        "require_production_label_eligible",
        "root_group_minimum_distinct_roots",
        "example_root_group_memberships",
    }

    def source_view(
        rows: list[dict[str, Any]],
        *,
        source_name: str,
        requested_size: int,
        kwargs: Mapping[str, Any] | None,
        source_seed: int,
        root_group_floors: Mapping[str, int] | None = None,
        example_root_groups: Mapping[str, Iterable[str]] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        configured = dict(kwargs or {})
        forbidden = sorted(blocked_kwargs & set(configured))
        if forbidden:
            raise ValueError(
                f"{source_name}_training_view_kwargs cannot override: "
                + ", ".join(forbidden)
            )
        builder_kwargs = {
            "minimum_tool_category_natural_rows": 0,
            "minimum_tool_category_distinct_roots": 0,
            **configured,
        }
        if root_group_floors is not None:
            builder_kwargs["root_group_minimum_distinct_roots"] = (
                root_group_floors
            )
            builder_kwargs["example_root_group_memberships"] = (
                example_root_groups or {}
            )
        view, report = build_balanced_training_view(
            rows,
            size=requested_size,
            seed=source_seed,
            max_duplicate_count=int(max_duplicate_count),
            max_rows_per_root=int(max_rows_per_root),
            require_production_label_eligible=True,
            **builder_kwargs,
        )
        for row in view:
            row["replay_source"] = source_name
        return view, report

    d0_view, d0_report = source_view(
        d0,
        source_name="d0_bc0",
        requested_size=d0_size,
        kwargs=d0_training_view_kwargs,
        source_seed=int(seed),
    )
    d1_view, d1_report = source_view(
        d1,
        source_name="d1_recovery",
        requested_size=d1_size,
        kwargs=d1_training_view_kwargs,
        source_seed=int(seed) + 1,
        root_group_floors=d1_root_group_floors,
        example_root_groups=d1_example_root_groups,
    )
    sampled_d1_independent_root_support = audit_dagger1_independent_root_support(
        d1_view,
        targeted_state_cell_minimum_distinct_roots=normalized_d1_cell_floors,
        recovery_stratum_minimum_distinct_roots=normalized_d1_stratum_floors,
    )
    if sampled_d1_independent_root_support.get("passed") is not True:
        raise ValueError(
            "sampled D1 training view cannot preserve independent physical-root "
            "support within its allocated size: "
            f"allocated_rows={d1_size}, targeted_cells="
            f"{sampled_d1_independent_root_support.get('targeted_state_cell_shortfalls')}, "
            "recovery_strata="
            f"{sampled_d1_independent_root_support.get('recovery_stratum_shortfalls')}"
        )
    view = d0_view + d1_view
    random.Random(int(seed)).shuffle(view)

    root_counts = Counter(_physical_root(row) for row in view)
    example_counts = Counter(str(row["example_id"]) for row in view)
    root_violations = {
        str(root): count
        for root, count in sorted(root_counts.items(), key=lambda item: str(item[0]))
        if count > int(max_rows_per_root)
    }
    duplicate_violations = {
        example_id: count
        for example_id, count in sorted(example_counts.items())
        if count > int(max_duplicate_count)
    }
    observed_d1 = sum(row.get("replay_source") == "d1_recovery" for row in view)
    observed_share = observed_d1 / len(view)
    share_passed = (
        observed_d1 == d1_size
        and float(minimum_d1_share) <= observed_share <= float(maximum_d1_share)
    )
    passed = bool(
        d0_report.get("release_ready")
        and d1_report.get("release_ready")
        and share_passed
        and not root_violations
        and not duplicate_violations
        and sampled_d1_independent_root_support.get("passed") is True
        and all(row.get("production_label_eligible") is True for row in view)
    )
    report = {
        "schema_version": 1,
        "builder_contract": "deterministic_d0_d1_balanced_union_v1",
        "seed": int(seed),
        "requested_size": total_size,
        "returned_size": len(view),
        "source_allocation": {
            "d0_bc0_rows": len(view) - observed_d1,
            "d1_recovery_rows": observed_d1,
            "configured_d1_share": float(d1_share),
            "observed_d1_share": observed_share,
            "minimum_d1_share": float(minimum_d1_share),
            "maximum_d1_share": float(maximum_d1_share),
            "passed": share_passed,
        },
        "physical_root_contract": {
            "sources_disjoint": True,
            "max_rows_per_root": int(max_rows_per_root),
            "violations": root_violations,
            "passed": not root_violations,
        },
        "duplicate_contract": {
            "max_duplicate_count": int(max_duplicate_count),
            "violations": duplicate_violations,
            "passed": not duplicate_violations,
        },
        "production_label_eligibility_passed": all(
            row.get("production_label_eligible") is True for row in view
        ),
        "replay_capacity": replay_capacity,
        "d1_independent_root_support": {
            "eligible_natural_source": d1_independent_root_support,
            "sampled_training_view": sampled_d1_independent_root_support,
            "passed": bool(
                d1_independent_root_support.get("passed") is True
                and sampled_d1_independent_root_support.get("passed") is True
            ),
        },
        "d0_training_view": d0_report,
        "d1_training_view": d1_report,
        "passed": passed,
        "release_ready": passed,
    }
    return view, report
