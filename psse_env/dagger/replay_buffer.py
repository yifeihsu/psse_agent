from __future__ import annotations

import copy
import hashlib
import math
import random
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping


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
    values with spare capacity.  This is appropriate for secondary balancing
    axes, where equal representation is a sampling objective rather than a
    release requirement.
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


def _target_tool(row: Mapping[str, Any]) -> str:
    action = row.get("preferred_action")
    if isinstance(action, Mapping) and action.get("tool"):
        return str(action["tool"])
    messages = row.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, Mapping) or message.get("role") != "assistant":
                continue
            calls = message.get("tool_calls")
            if isinstance(calls, list) and calls:
                function = calls[0].get("function") if isinstance(calls[0], Mapping) else None
                if isinstance(function, Mapping) and function.get("name"):
                    return str(function["name"])
    return "unknown"


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
    for index, row in enumerate(materialized):
        tool = _target_tool(row)
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
    return {
        "rows": len(materialized),
        **{axis: dict(sorted(counts.items())) for axis, counts in axes.items()},
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
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build a deterministic multi-axis balanced SFT training view.

    The immutable aggregate remains untouched. Known low-margin rows are
    excluded from single-label SFT, while state class, target tool/category,
    scenario family, error cardinality, terminal outcome, and physical-root
    contribution jointly drive the greedy deficit sampler. Tool-category
    proportions are strict release targets. The other axes use deterministic
    capacity-aware equalization, so a rare state cannot make a full-size view
    mathematically impossible merely because uniform representation is not a
    declared release requirement.
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
    excluded_low_margin = [
        row
        for row in source
        if (margin := _known_cost_margin(row)) is not None
        and margin <= float(low_cost_margin_threshold)
    ]
    eligible = [row for row in source if row not in excluded_low_margin]
    if not eligible:
        raise ValueError("all training rows were excluded by the cost-margin gate")
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

    axis_values: dict[int, dict[str, str]] = {}
    roots: dict[int, str] = {}
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
        for value in sorted({item[axis] for item in axis_values.values()}):
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

    strict_category_weights = {
        category: category_weights[category]
        for category in sorted(available_categories)
    }
    strict_category_targets = _allocate_counts(
        requested_size, strict_category_weights
    )
    unconstrained_target_counts: dict[str, dict[str, int]] = {
        "tool_category": dict(strict_category_targets)
    }
    target_counts: dict[str, dict[str, int]] = {
        "tool_category": dict(strict_category_targets)
    }
    capacity_adjustments: dict[str, dict[str, dict[str, int]]] = {}
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

    def candidate_score(index: int) -> tuple[float, int]:
        values = axis_values[index]
        score = 0.0
        for axis, targets in target_counts.items():
            value = values[axis]
            target = max(int(targets.get(value, 0)), 1)
            deficit = target - observed[axis][value]
            multiplier = 4.0 if axis == "tool_category" else 1.0
            score += multiplier * deficit / target
        score -= occurrences[index] * 0.5
        score -= root_counts[roots[index]] / max(int(root_cap), 1)
        return score, -tie_break[index]

    while len(selected) < requested_size:
        candidates = [
            index
            for index in range(len(eligible))
            if occurrences[index] < int(max_duplicate_count)
            and root_counts[roots[index]] < int(root_cap)
        ]
        if not candidates:
            raise ValueError("training-view constraints exhausted before reaching requested size")
        chosen = max(candidates, key=candidate_score)
        selected.append(chosen)
        occurrences[chosen] += 1
        root_counts[roots[chosen]] += 1
        for axis, value in axis_values[chosen].items():
            observed[axis][value] += 1

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
        feasibility_shortfall_total == 0
        and achieved_tool_category_target_deviation
        <= float(maximum_tool_category_target_deviation)
    )
    report = {
        "seed": int(seed),
        "requested_size": requested_size,
        "returned_size": len(view),
        "low_cost_margin_threshold": float(low_cost_margin_threshold),
        "excluded_low_margin_rows": len(excluded_low_margin),
        "max_duplicate_count": int(max_duplicate_count),
        "duplicate_occurrences": sum(max(count - 1, 0) for count in occurrences.values()),
        "max_rows_per_root": int(root_cap),
        "target_contract": {
            "size_policy": "requested_full_size_with_bounded_replacement",
            "strict_target_axes": ["tool_category"],
            "capacity_aware_target_axes": list(secondary_axes),
            "capacity_aware_policy": "uniform_then_clip_and_redistribute_v1",
        },
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
        "before": _training_view_summary(source),
        "after": _training_view_summary(view),
    }
    return view, report
