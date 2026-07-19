from __future__ import annotations

import copy
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
    "terminal_decision": 0.10,
    "loop_repetition": 0.05,
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
