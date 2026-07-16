"""Preflight audits for DAgger chat/tool SFT exports.

The checks in this module are intentionally independent from the privileged
teacher.  They answer whether a deterministic policy input has a realizable
semantic label after controller identifiers have been canonicalized.
"""

from __future__ import annotations

import hashlib
import json
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
    "audit_chat_sft_rows",
    "audit_teacher_realizability",
    "canonical_policy_observation",
    "canonical_semantic_action",
    "canonicalize_state_identifiers",
    "count_model_history_locations",
    "policy_observation_hash",
    "semantic_action_key",
]
