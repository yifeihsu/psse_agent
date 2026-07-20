"""Preflight audits for DAgger chat/tool SFT exports.

The checks in this module are intentionally independent from the privileged
teacher.  They answer whether a deterministic policy input has a realizable
semantic label after controller identifiers have been canonicalized.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from typing import Any, Iterable, Mapping

from psse_env.actions import INVALID_ACTION, safe_normalize_action
from psse_env.dagger.dataset_builder import (
    ACTION_STATE_REFERENCE_KEYS,
    HISTORY_KEYS,
    alias_model_visible_state,
    build_hash_alias_bindings,
    build_state_alias_bindings,
    find_model_identifier_leaks,
    prepare_model_policy_observation,
    validate_policy_payload,
    validate_tool_schemas,
)


def _stable_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Audit payload is not finite JSON: {exc}") from exc


def _episode_aliases(value: Any) -> dict[str, str]:
    identifiers: list[str] = []

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                if str(key) == "episode_id" and child is not None:
                    identifiers.append(str(child))
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
    return {f"episode{index}" if index else "episode": identifier for index, identifier in enumerate(dict.fromkeys(identifiers))}


def canonicalize_state_identifiers(value: Any, *related_values: Any) -> Any:
    """Canonicalize ephemeral controller IDs while preserving their equality."""
    bindings = build_state_alias_bindings(value, *related_values)
    episodes: dict[str, str] = {}
    for payload in (value, *related_values):
        for alias, identifier in _episode_aliases(payload).items():
            if identifier not in episodes.values():
                candidate = alias
                suffix = 1
                while candidate in episodes:
                    candidate = f"episode{suffix}"
                    suffix += 1
                episodes[candidate] = identifier
    return alias_model_visible_state(
        value,
        bindings,
        episode_aliases=episodes,
        hash_aliases=build_hash_alias_bindings(value, *related_values),
    )


def canonical_policy_observation(
    observation: Mapping[str, Any],
    *,
    history: Iterable[Mapping[str, Any]] | None = None,
    already_model_visible: bool = False,
) -> dict[str, Any]:
    """Return the identifier-invariant semantic policy input used for hashing."""
    if already_model_visible:
        canonical = canonicalize_state_identifiers(
            {
                str(key): value
                for key, value in observation.items()
                if str(key)
                not in {
                    "semantic_field_provenance",
                    "policy_field_provenance",
                    "policy_provenance",
                    "provenance",
                }
            }
        )
    else:
        canonical, _ = prepare_model_policy_observation(observation, history=history)
    if not isinstance(canonical, Mapping):  # defensive; input is typed as a mapping
        raise ValueError("Canonical policy observation must remain a mapping.")
    validate_policy_payload(canonical)
    return dict(canonical)


def policy_observation_hash(
    observation: Mapping[str, Any],
    *,
    history: Iterable[Mapping[str, Any]] | None = None,
) -> str:
    """Hash an observation after removing episode/state identifier shortcuts."""
    canonical = canonical_policy_observation(observation, history=history)
    return hashlib.sha256(_stable_json(canonical).encode("utf-8")).hexdigest()


def canonical_semantic_action(
    action: Mapping[str, Any] | str,
    observation: Mapping[str, Any],
    *,
    history: Iterable[Mapping[str, Any]] | None = None,
    already_model_visible: bool = False,
) -> dict[str, Any]:
    """Canonicalize only controller IDs; retain semantic tool arguments."""
    normalized = safe_normalize_action(action)
    if normalized["tool"] == INVALID_ACTION:
        raise ValueError(f"Teacher action is malformed: {normalized['arguments']}")
    if already_model_visible:
        prepared_observation = dict(observation)
        alias_metadata = {
            "state_aliases": {
                str(value): str(value)
                for key, value in prepared_observation.items()
                if str(key) in {"active_state_id", "candidate_state_id"}
                and value is not None
            },
            "episode_aliases": {},
            "hash_aliases": {},
        }
    else:
        prepared_observation, alias_metadata = prepare_model_policy_observation(
            observation,
            history=history,
        )
    canonical = alias_model_visible_state(
        normalized,
        alias_metadata["state_aliases"],
        episode_aliases=alias_metadata["episode_aliases"],
        hash_aliases=alias_metadata["hash_aliases"],
    )
    if not isinstance(canonical, Mapping):
        raise ValueError("Canonical action must remain a mapping.")
    canonical_dict = dict(canonical)
    for key in ACTION_STATE_REFERENCE_KEYS:
        reference = canonical_dict["arguments"].get(key)
        if reference is not None and str(reference) not in alias_metadata["state_aliases"]:
            raise ValueError(f"Teacher action references unbound controller state for {key}: {reference}")
    return canonical_dict


def semantic_action_key(
    action: Mapping[str, Any] | str,
    observation: Mapping[str, Any],
) -> str:
    return _stable_json(canonical_semantic_action(action, observation))


def admissible_semantic_action_count(example: Mapping[str, Any]) -> int:
    """Count distinct admissible labels after controller-ID canonicalization.

    Raw oracle proposal lists can contain actions that differ only by ephemeral
    state identifiers.  Those are one learning target, not multiple competing
    labels.  Conversely, genuinely different tools or arguments must not be
    collapsed merely because the rank-one teacher selected one of them.
    """
    valid_actions = example.get("valid_next_actions")
    if not isinstance(valid_actions, list) or not valid_actions:
        return int(_example_target(example) is not None)
    observation = _example_observation(example)
    if observation is None:
        return 0
    already_model_visible = not any(
        isinstance(example.get(key), Mapping)
        for key in ("policy_observation", "state_summary")
    )
    raw_history = None if already_model_visible else example.get("history_window")
    history = (
        list(raw_history)
        if raw_history is not None
        and not isinstance(raw_history, (str, bytes, Mapping))
        else None
    )
    distinct_actions: set[str] = set()
    for action in valid_actions:
        try:
            semantic_action = canonical_semantic_action(
                action,
                observation,
                history=history,
                already_model_visible=already_model_visible,
            )
        except ValueError:
            continue
        if semantic_action["tool"] != INVALID_ACTION:
            distinct_actions.add(_stable_json(semantic_action))
    return len(distinct_actions) or int(_example_target(example) is not None)


def _chat_observation(example: Mapping[str, Any]) -> Mapping[str, Any] | None:
    messages = example.get("messages")
    if not isinstance(messages, list):
        return None
    for message in messages:
        if not isinstance(message, Mapping) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            try:
                content = json.loads(content)
            except json.JSONDecodeError:
                return None
        if isinstance(content, Mapping):
            state = content.get("state", content)
            return state if isinstance(state, Mapping) else None
    return None


def _example_observation(example: Mapping[str, Any]) -> Mapping[str, Any] | None:
    for key in ("policy_observation", "state_summary"):
        value = example.get(key)
        if isinstance(value, Mapping):
            return value
    return _chat_observation(example)


def _chat_target(example: Mapping[str, Any]) -> Mapping[str, Any] | None:
    messages = example.get("messages")
    if not isinstance(messages, list):
        return None
    for message in messages:
        if not isinstance(message, Mapping) or message.get("role") != "assistant":
            continue
        calls = message.get("tool_calls")
        if isinstance(calls, list) and calls and isinstance(calls[0], Mapping):
            function = calls[0].get("function")
            if isinstance(function, Mapping):
                return {
                    "tool": function.get("name"),
                    "arguments": function.get("arguments", {}),
                }
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            try:
                decoded = json.loads(content)
            except json.JSONDecodeError:
                return None
            return decoded if isinstance(decoded, Mapping) else None
    return None


def _example_target(example: Mapping[str, Any]) -> Mapping[str, Any] | str | None:
    target = example.get("preferred_action")
    if isinstance(target, (Mapping, str)):
        return target
    valid = example.get("valid_next_actions")
    if isinstance(valid, list) and valid and isinstance(valid[0], (Mapping, str)):
        return valid[0]
    return _chat_target(example)


def audit_teacher_realizability(
    examples: Iterable[Mapping[str, Any]],
    *,
    conflict_tolerance: float = 0.0,
    max_conflict_details: int = 100,
) -> dict[str, Any]:
    """Report identical policy inputs with conflicting semantic teacher labels.

    ``conflict_rate`` is the fraction of labeled rows that belong to an
    observation equivalence class with more than one semantic preferred action.
    This row-weighted definition makes the launch tolerance explicit and avoids
    hiding a frequent conflict behind many singleton observations.
    """
    if not 0.0 <= float(conflict_tolerance) <= 1.0:
        raise ValueError("conflict_tolerance must be between 0 and 1.")
    if max_conflict_details < 0:
        raise ValueError("max_conflict_details must be nonnegative.")

    grouped: dict[str, dict[str, Any]] = {}
    total = 0
    unlabeled = 0
    invalid: list[dict[str, Any]] = []
    for index, example in enumerate(examples):
        total += 1
        example_id = str(example.get("example_id", f"row_{index}"))
        observation = _example_observation(example)
        target = _example_target(example)
        already_model_visible = not any(
            isinstance(example.get(key), Mapping)
            for key in ("policy_observation", "state_summary")
        )
        raw_history = None if already_model_visible else example.get("history_window")
        history = (
            list(raw_history)
            if raw_history is not None
            and not isinstance(raw_history, (str, bytes, Mapping))
            else None
        )
        if observation is None or target is None:
            unlabeled += 1
            invalid.append(
                {
                    "example_id": example_id,
                    "reason": "missing_policy_observation" if observation is None else "missing_preferred_action",
                }
            )
            continue
        try:
            observation_canonical = canonical_policy_observation(
                observation,
                history=history,
                already_model_visible=already_model_visible,
            )
            observation_digest = hashlib.sha256(
                _stable_json(observation_canonical).encode("utf-8")
            ).hexdigest()
            action_canonical = canonical_semantic_action(
                target,
                observation,
                history=history,
                already_model_visible=already_model_visible,
            )
            action_key = _stable_json(action_canonical)
        except ValueError as exc:
            unlabeled += 1
            invalid.append({"example_id": example_id, "reason": str(exc)})
            continue

        bucket = grouped.setdefault(
            observation_digest,
            {
                "observation_hash": observation_digest,
                "canonical_observation": observation_canonical,
                "rows": [],
                "actions": defaultdict(list),
            },
        )
        bucket["rows"].append(example_id)
        bucket["actions"][action_key].append(example_id)

    conflicts: list[dict[str, Any]] = []
    conflicting_rows = 0
    for digest, bucket in sorted(grouped.items()):
        actions = bucket["actions"]
        if len(actions) <= 1:
            continue
        conflicting_rows += len(bucket["rows"])
        conflicts.append(
            {
                "observation_hash": digest,
                "example_ids": list(bucket["rows"]),
                "semantic_actions": [json.loads(key) for key in sorted(actions)],
                "action_examples": {
                    key: list(action_ids) for key, action_ids in sorted(actions.items())
                },
            }
        )

    labeled = total - unlabeled
    conflict_rate = float(conflicting_rows / labeled) if labeled else 0.0
    return {
        "total_examples": total,
        "labeled_examples": labeled,
        "unlabeled_or_invalid_examples": unlabeled,
        "unique_observations": len(grouped),
        "conflict_observations": len(conflicts),
        "conflicting_examples": conflicting_rows,
        "conflict_rate": conflict_rate,
        "conflict_tolerance": float(conflict_tolerance),
        "passed": not invalid and conflict_rate <= float(conflict_tolerance),
        "conflicts": conflicts[:max_conflict_details],
        "conflict_details_truncated": max(0, len(conflicts) - max_conflict_details),
        "invalid_examples": invalid,
    }


_APPROX_NUMERIC_MARKERS = (
    "residual",
    "lagrange",
    "lambda",
    "chi_square",
    "objective",
    "anomaly_score",
    "progress",
    "threshold",
    "ratio",
    "suspect_count",
)
_APPROX_STRUCTURAL_KEYS = frozenset(
    {
        "candidate_lifecycle",
        "candidate_status",
        "has_open_candidate",
        "has_unverified_candidate",
        "has_verified_candidate",
        "has_fresh_measurement_context",
        "has_fresh_parameter_context",
        "has_fresh_topology_context",
        "last_tool",
        "last_tool_status",
        "execution_status",
        "process_valid",
        "state_mutated",
        "tool",
    }
)


def _normalized_signature_text(value: Any) -> str:
    text = str(value).strip().lower()
    # Provider signatures may append indices or magnitudes. Keep the semantic
    # family while allowing nearby continuous states to share an audit bucket.
    return re.sub(r"[-+]?\d+(?:\.\d+)?", "#", text)


def _scaled_numeric(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("Approximate audit encountered a non-finite feature.")
    if abs(value) <= 20.0:
        return value
    return math.copysign(20.0 + math.log1p(abs(value) - 20.0), value)


def _approximate_features(
    observation: Mapping[str, Any],
) -> tuple[dict[str, float], dict[str, Any]]:
    numeric: dict[str, float] = {}
    structural: dict[str, Any] = {}

    def visit(value: Any, path: str, key: str = "") -> None:
        if isinstance(value, Mapping):
            # Explicit threshold ratios make WLS magnitudes comparable across
            # operating points and covariance/noise settings.
            ratio_pairs = (
                ("chi_square_statistic", "chi_square_threshold"),
                ("remaining_anomaly_score", "anomaly_threshold"),
                ("global_residual_sum", "global_residual_threshold"),
            )
            for numerator_key, denominator_key in ratio_pairs:
                numerator = value.get(numerator_key)
                denominator = value.get(denominator_key)
                if (
                    isinstance(numerator, (int, float))
                    and not isinstance(numerator, bool)
                    and isinstance(denominator, (int, float))
                    and not isinstance(denominator, bool)
                    and float(denominator) != 0.0
                ):
                    numeric[f"{path}.{numerator_key}_to_{denominator_key}"] = _scaled_numeric(
                        float(numerator) / float(denominator)
                    )
            for child_key, child in value.items():
                child_key_text = str(child_key)
                child_path = f"{path}.{child_key_text}"
                if child_key_text in _APPROX_STRUCTURAL_KEYS and isinstance(
                    child, (str, bool)
                ):
                    structural[child_path] = child
                if child_key_text == "unresolved_signatures" and isinstance(child, list):
                    structural[child_path] = sorted(
                        _normalized_signature_text(item) for item in child
                    )
                if child_key_text in {"top_residuals", "top_lagrange"} and isinstance(
                    child, list
                ):
                    ranks: list[tuple[Any, ...]] = []
                    for item in child:
                        if not isinstance(item, Mapping):
                            continue
                        if child_key_text == "top_residuals":
                            ranks.append((item.get("channel"), item.get("index0")))
                        else:
                            ranks.append(
                                (
                                    item.get("line_row0"),
                                    item.get("from_bus"),
                                    item.get("to_bus"),
                                    item.get("terminal"),
                                )
                            )
                    structural[f"{child_path}.rank_pattern"] = ranks
                if child_key_text == "top_hif_groups" and isinstance(child, list):
                    # NLM branch localization is the observable categorical
                    # target for the downstream HIF estimator.  Treating all
                    # localized branches as the same approximate state creates
                    # a false teacher conflict even though the model prompt
                    # explicitly contains the ranked branch row.
                    structural[f"{child_path}.rank_pattern"] = [
                        (
                            item.get("branch_row0"),
                            item.get("dss_element"),
                            item.get("phase"),
                        )
                        for item in child
                        if isinstance(item, Mapping)
                    ]
                visit(child, child_path, child_key_text)
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]", key)
        elif (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and any(marker in key.lower() for marker in _APPROX_NUMERIC_MARKERS)
        ):
            numeric[path] = _scaled_numeric(float(value))

    visit(observation, "state")
    return numeric, structural


def _quantized_feature_key(
    numeric: Mapping[str, float],
    structural: Mapping[str, Any],
    *,
    quantization_bin: float,
) -> str:
    quantized = {
        key: int(round(value / quantization_bin)) for key, value in numeric.items()
    }
    return _stable_json({"numeric": quantized, "structural": structural})


def _feature_distance(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    keys = set(left) | set(right)
    if not keys:
        return 0.0
    squared = 0.0
    for key in keys:
        if key not in left or key not in right:
            squared += 4.0
        else:
            squared += (float(left[key]) - float(right[key])) ** 2
    return math.sqrt(squared / len(keys))


def example_cost_margin(example: Mapping[str, Any]) -> float | None:
    sources = [example]
    for key in ("metadata", "labels"):
        nested = example.get(key)
        if isinstance(nested, Mapping):
            sources.append(nested)
    for source in sources:
        value = source.get("cost_margin")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        costs = source.get("action_costs")
        if isinstance(costs, list):
            values = sorted(
                float(item["q_cost"])
                for item in costs
                if isinstance(item, Mapping)
                and isinstance(item.get("q_cost"), (int, float))
            )
            if len(values) >= 2:
                return values[1] - values[0]
    return None


def _example_requires_cost_margin(example: Mapping[str, Any]) -> bool:
    sources = [example]
    for key in ("metadata", "labels"):
        nested = example.get(key)
        if isinstance(nested, Mapping):
            sources.append(nested)
    for source in sources:
        value = source.get("admissible_semantic_action_count")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return int(value) > 1
    # Native collector rows carry the admissible semantic set directly.  Do
    # not let the release gate silently treat those rows as single-action just
    # because they have not yet passed through the chat exporter, where the
    # explicit count is added to non-model metadata.
    return admissible_semantic_action_count(example) > 1


def audit_approximate_teacher_realizability(
    examples: Iterable[Mapping[str, Any]],
    *,
    quantization_bin: float = 0.25,
    conflict_tolerance: float = 0.05,
    nearest_neighbor_tolerance: float = 0.10,
    perturbation_radius: float = 0.25,
    low_cost_margin_threshold: float = 0.05,
    require_cost_margins: bool = False,
    require_cost_margins_for_multi_action: bool = False,
    minimum_nearest_neighbor_comparisons: int = 0,
    minimum_nearest_neighbor_coverage: float = 0.0,
    local_perturbation_tolerance: float | None = None,
    minimum_local_perturbation_comparisons: int = 0,
    minimum_local_perturbation_coverage: float = 0.0,
    max_conflict_details: int = 100,
) -> dict[str, Any]:
    """Audit label stability for nearby continuous WLS observations.

    This complements, rather than replaces, the exact-hash audit. It combines
    provider-normalized residuals/threshold ratios, rank patterns, quantized
    buckets, nearest-neighbor disagreement, conditional action entropy, local
    perturbation stability, and available expert cost margins.
    """
    if quantization_bin <= 0 or perturbation_radius < 0:
        raise ValueError("quantization_bin must be positive and perturbation_radius nonnegative.")
    for value, name in (
        (conflict_tolerance, "conflict_tolerance"),
        (nearest_neighbor_tolerance, "nearest_neighbor_tolerance"),
        (minimum_nearest_neighbor_coverage, "minimum_nearest_neighbor_coverage"),
        (minimum_local_perturbation_coverage, "minimum_local_perturbation_coverage"),
    ):
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1.")
    if local_perturbation_tolerance is not None and not 0.0 <= float(
        local_perturbation_tolerance
    ) <= 1.0:
        raise ValueError("local_perturbation_tolerance must be between 0 and 1.")
    if minimum_nearest_neighbor_comparisons < 0 or minimum_local_perturbation_comparisons < 0:
        raise ValueError("minimum comparison counts must be nonnegative.")

    records: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    margins: list[float] = []
    required_margin_examples = 0
    required_margin_present = 0
    for index, example in enumerate(examples):
        example_id = str(example.get("example_id", f"row_{index}"))
        observation = _example_observation(example)
        target = _example_target(example)
        if observation is None or target is None:
            invalid.append({"example_id": example_id, "reason": "missing observation or target"})
            continue
        already_model_visible = not any(
            isinstance(example.get(key), Mapping)
            for key in ("policy_observation", "state_summary")
        )
        raw_history = None if already_model_visible else example.get("history_window")
        history = (
            list(raw_history)
            if raw_history is not None
            and not isinstance(raw_history, (str, bytes, Mapping))
            else None
        )
        try:
            canonical = canonical_policy_observation(
                observation,
                history=history,
                already_model_visible=already_model_visible,
            )
            action = canonical_semantic_action(
                target,
                observation,
                history=history,
                already_model_visible=already_model_visible,
            )
            numeric, structural = _approximate_features(canonical)
        except ValueError as exc:
            invalid.append({"example_id": example_id, "reason": str(exc)})
            continue
        margin = example_cost_margin(example)
        margin_required = _example_requires_cost_margin(example)
        required_margin_examples += int(margin_required)
        if margin is not None and math.isfinite(margin):
            margins.append(margin)
            required_margin_present += int(margin_required)
        records.append(
            {
                "example_id": example_id,
                "action_key": _stable_json(action),
                "numeric": numeric,
                "structural": structural,
                "bucket": _quantized_feature_key(
                    numeric, structural, quantization_bin=quantization_bin
                ),
            }
        )

    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    structures: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        buckets[record["bucket"]].append(record)
        structures[_stable_json(record["structural"])].append(record)

    conflicting_rows = 0
    entropy_weighted = 0.0
    conflicts: list[dict[str, Any]] = []
    for bucket_key, bucket in buckets.items():
        action_counts: dict[str, int] = defaultdict(int)
        for record in bucket:
            action_counts[record["action_key"]] += 1
        bucket_entropy = -sum(
            (count / len(bucket)) * math.log2(count / len(bucket))
            for count in action_counts.values()
        )
        entropy_weighted += len(bucket) * bucket_entropy
        if len(action_counts) > 1:
            conflicting_rows += len(bucket)
            conflicts.append(
                {
                    "feature_hash": hashlib.sha256(bucket_key.encode("utf-8")).hexdigest(),
                    "example_ids": [record["example_id"] for record in bucket],
                    "semantic_actions": [json.loads(key) for key in sorted(action_counts)],
                    "conditional_action_entropy_bits": bucket_entropy,
                }
            )

    nearest_total = 0
    nearest_disagreements = 0
    local_total = 0
    local_disagreements = 0
    nearest_distances: list[float] = []
    for structural_group in structures.values():
        if len(structural_group) < 2:
            continue
        for index, record in enumerate(structural_group):
            candidates = [
                (
                    _feature_distance(record["numeric"], other["numeric"]),
                    other,
                )
                for other_index, other in enumerate(structural_group)
                if other_index != index
            ]
            distance, neighbor = min(
                candidates, key=lambda item: (item[0], item[1]["example_id"])
            )
            nearest_total += 1
            nearest_distances.append(distance)
            disagrees = record["action_key"] != neighbor["action_key"]
            nearest_disagreements += int(disagrees)
            if distance <= perturbation_radius:
                local_total += 1
                local_disagreements += int(disagrees)

    labeled = len(records)
    conflict_rate = conflicting_rows / labeled if labeled else 0.0
    nearest_rate = nearest_disagreements / nearest_total if nearest_total else 0.0
    local_rate = local_disagreements / local_total if local_total else 0.0
    low_margin = sum(margin <= low_cost_margin_threshold for margin in margins)
    margin_coverage = len(margins) / labeled if labeled else 0.0
    nearest_coverage = nearest_total / labeled if labeled else 0.0
    local_coverage = local_total / labeled if labeled else 0.0
    passed = (
        not invalid
        and conflict_rate <= conflict_tolerance
        and nearest_rate <= nearest_neighbor_tolerance
        and nearest_total >= int(minimum_nearest_neighbor_comparisons)
        and nearest_coverage >= float(minimum_nearest_neighbor_coverage)
        and local_total >= int(minimum_local_perturbation_comparisons)
        and local_coverage >= float(minimum_local_perturbation_coverage)
        and (
            local_perturbation_tolerance is None
            or local_rate <= float(local_perturbation_tolerance)
        )
        and (not require_cost_margins or len(margins) == labeled)
        and (
            not require_cost_margins_for_multi_action
            or required_margin_present == required_margin_examples
        )
    )
    return {
        "total_examples": len(records) + len(invalid),
        "labeled_examples": labeled,
        "quantization_bin": quantization_bin,
        "quantized_bucket_count": len(buckets),
        "conflict_buckets": len(conflicts),
        "conflicting_examples": conflicting_rows,
        "approximate_conflict_rate": conflict_rate,
        "conflict_tolerance": conflict_tolerance,
        "conditional_action_entropy_bits": entropy_weighted / labeled if labeled else 0.0,
        "nearest_neighbor_compared_examples": nearest_total,
        "nearest_neighbor_comparison_coverage": nearest_coverage,
        "minimum_nearest_neighbor_comparisons": int(
            minimum_nearest_neighbor_comparisons
        ),
        "minimum_nearest_neighbor_coverage": float(
            minimum_nearest_neighbor_coverage
        ),
        "nearest_neighbor_action_disagreement_rate": nearest_rate,
        "nearest_neighbor_tolerance": nearest_neighbor_tolerance,
        "mean_nearest_neighbor_distance": (
            sum(nearest_distances) / len(nearest_distances) if nearest_distances else None
        ),
        "perturbation_radius": perturbation_radius,
        "local_perturbation_compared_examples": local_total,
        "local_perturbation_comparison_coverage": local_coverage,
        "local_perturbation_action_disagreement_rate": local_rate,
        "local_perturbation_tolerance": local_perturbation_tolerance,
        "minimum_local_perturbation_comparisons": int(
            minimum_local_perturbation_comparisons
        ),
        "minimum_local_perturbation_coverage": float(
            minimum_local_perturbation_coverage
        ),
        "cost_margin_examples": len(margins),
        "cost_margin_coverage": margin_coverage,
        "low_cost_margin_threshold": low_cost_margin_threshold,
        "low_cost_margin_rate": low_margin / len(margins) if margins else None,
        "cost_margins_required": require_cost_margins,
        "multi_action_cost_margins_required": require_cost_margins_for_multi_action,
        "multi_action_examples": required_margin_examples,
        "multi_action_cost_margin_examples": required_margin_present,
        "multi_action_cost_margin_coverage": (
            required_margin_present / required_margin_examples
            if required_margin_examples
            else 1.0
        ),
        "normalization": "provider-normalized residuals, threshold ratios, and rank patterns",
        "passed": passed,
        "conflicts": conflicts[:max_conflict_details],
        "conflict_details_truncated": max(0, len(conflicts) - max_conflict_details),
        "invalid_examples": invalid,
    }


def count_model_history_locations(value: Any) -> int:
    """Count history containers recursively in a model-visible payload."""
    count = 0
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key) in HISTORY_KEYS:
                count += 1
            count += count_model_history_locations(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            count += count_model_history_locations(item)
    return count


def audit_chat_sft_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Check portable row structure before expensive tokenizer/model audits."""
    errors: list[dict[str, Any]] = []
    total = 0
    for index, row in enumerate(rows):
        total += 1
        row_id = str(row.get("example_id", f"row_{index}"))
        row_errors: list[str] = []
        tools = row.get("tools")
        try:
            if not isinstance(tools, list):
                raise ValueError("row-level tools must be a list")
            validate_tool_schemas(tools)
        except ValueError as exc:
            row_errors.append(str(exc))

        messages = row.get("messages")
        user_payload: Any = None
        assistant_calls: Any = None
        if not isinstance(messages, list):
            row_errors.append("messages must be a list")
        else:
            for message in messages:
                if not isinstance(message, Mapping):
                    continue
                if message.get("role") == "user":
                    content = message.get("content")
                    try:
                        user_payload = json.loads(content) if isinstance(content, str) else content
                    except json.JSONDecodeError:
                        row_errors.append("user message is not valid JSON")
                if message.get("role") == "assistant":
                    assistant_calls = message.get("tool_calls")
        if user_payload is None:
            row_errors.append("missing user policy payload")
        else:
            provenance = None
            metadata = row.get("metadata")
            if isinstance(metadata, Mapping):
                candidate_provenance = metadata.get("semantic_field_provenance")
                if isinstance(candidate_provenance, Mapping):
                    provenance = candidate_provenance
            try:
                validate_policy_payload(
                    user_payload,
                    provenance=provenance,
                    require_derived_provenance=True,
                )
            except ValueError as exc:
                row_errors.append(str(exc))
            leaks = find_model_identifier_leaks(user_payload)
            if leaks:
                row_errors.append(
                    "model-visible opaque controller identifiers at " + ", ".join(leaks)
                )
            locations = count_model_history_locations(user_payload)
            if locations != 1:
                row_errors.append(f"expected exactly one history location, found {locations}")

        if not isinstance(assistant_calls, list) or len(assistant_calls) != 1:
            row_errors.append("assistant must contain exactly one native tool call")
        else:
            function = assistant_calls[0].get("function") if isinstance(assistant_calls[0], Mapping) else None
            arguments = function.get("arguments") if isinstance(function, Mapping) else None
            if not isinstance(arguments, Mapping):
                row_errors.append("assistant function.arguments must be a dictionary")
        if row_errors:
            errors.append({"example_id": row_id, "errors": row_errors})
    return {
        "total_rows": total,
        "invalid_rows": len(errors),
        "passed": not errors,
        "errors": errors,
    }


__all__ = [
    "admissible_semantic_action_count",
    "audit_approximate_teacher_realizability",
    "audit_chat_sft_rows",
    "audit_teacher_realizability",
    "canonical_policy_observation",
    "canonical_semantic_action",
    "canonicalize_state_identifiers",
    "count_model_history_locations",
    "example_cost_margin",
    "policy_observation_hash",
    "semantic_action_key",
]
