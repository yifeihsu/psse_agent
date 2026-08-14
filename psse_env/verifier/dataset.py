"""Dataset utilities for complete-transition process supervision."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .features import (
    extract_transition_features,
    normalize_action,
    normalize_transition,
    observable_copy,
    observable_verification_metrics,
    summarize_history,
)


LABEL_FIELDS = (
    "process_valid",
    "candidate_disposition",
    "progress_class",
    "collateral_damage",
    "collateral_damage_probability",
    "terminal_success",
    "terminal_success_probability",
    "estimated_remaining_steps",
    "valid_next_action_types",
    "valid_next_actions",
)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _action_types(actions: Any) -> list[str]:
    if not isinstance(actions, Sequence) or isinstance(actions, (str, bytes, bytearray)):
        return []
    result: list[str] = []
    for action in actions:
        if isinstance(action, str):
            tool = action
        else:
            tool = normalize_action(action).get("tool")
        if tool and tool not in result:
            result.append(str(tool))
    return result


def extract_transition_labels(row: Mapping[str, Any]) -> dict[str, Any]:
    """Collect labels without placing them in the model input or features."""

    tool_output = _mapping(row.get("tool_output"))
    tool_metrics = _mapping(tool_output.get("tool_metrics"))
    candidate_assessment = _mapping(tool_metrics.get("candidate_assessment"))
    candidate_state = _mapping(
        row.get("candidate_state_summary") or row.get("next_state_summary") or row.get("next_state")
    )
    sources = (
        _mapping(row.get("labels")),
        _mapping(row.get("transition_label")),
        _mapping(row.get("target")),
        candidate_assessment,
        tool_metrics,
        tool_output,
        candidate_state,
        dict(row),
    )
    labels: dict[str, Any] = {}
    for field in LABEL_FIELDS:
        for source in sources:
            if field in source and source[field] is not None:
                labels[field] = source[field]
                break

    if "candidate_disposition" not in labels and candidate_assessment.get("disposition") is not None:
        labels["candidate_disposition"] = candidate_assessment["disposition"]

    if "valid_next_action_types" not in labels:
        actions = labels.get("valid_next_actions") or row.get("valid_next_actions")
        action_types = _action_types(actions)
        if action_types:
            labels["valid_next_action_types"] = action_types
    labels.pop("valid_next_actions", None)
    return labels


def transition_from_dagger_example(example: Mapping[str, Any]) -> dict[str, Any]:
    """Convert one collector row to the six-field verifier input schema."""

    item = normalize_transition(example)
    return {
        "parent_state_summary": observable_copy(item.parent_state_summary),
        "action": observable_copy(item.action),
        "tool_output": observable_copy(item.tool_output),
        "candidate_state_summary": observable_copy(item.candidate_state_summary),
        "verification_metrics": observable_verification_metrics(item),
        "history_summary": observable_copy(summarize_history(item.history_summary)),
    }


def build_verifier_example(
    transition: Mapping[str, Any],
    *,
    labels: Mapping[str, Any] | None = None,
    include_features: bool = True,
    example_id: str | None = None,
) -> dict[str, Any]:
    """Build one serializable, leakage-separated verifier example."""

    normalized = transition_from_dagger_example(transition)
    row: dict[str, Any] = {
        "example_id": example_id or transition.get("example_id"),
        "scenario_id": transition.get("scenario_id"),
        "root_scenario_id": transition.get("root_scenario_id") or transition.get("scenario_id"),
        "episode_id": transition.get("episode_id"),
        "branch_id": transition.get("branch_id"),
        **normalized,
        "labels": dict(labels) if labels is not None else extract_transition_labels(transition),
    }
    if include_features:
        row["features"] = extract_transition_features(normalized)
    return row


def build_verifier_dataset(
    transitions: Iterable[Mapping[str, Any]],
    *,
    include_features: bool = True,
    require_disposition_label: bool = False,
) -> list[dict[str, Any]]:
    """Build a deterministic list of complete-transition examples."""

    result: list[dict[str, Any]] = []
    for index, transition in enumerate(transitions):
        base_id = str(transition.get("example_id") or f"verifier_transition_{index}")
        expanded: list[tuple[str, Mapping[str, Any]]] = [(base_id, transition)]
        for suffix, field in (("injection", "injection_transition"), ("verification", "verification_transition")):
            nested = transition.get(field)
            if not isinstance(nested, Mapping):
                continue
            materialized = {
                "scenario_id": transition.get("scenario_id"),
                "root_scenario_id": transition.get("root_scenario_id") or transition.get("scenario_id"),
                "episode_id": transition.get("episode_id"),
                "branch_id": transition.get("branch_id") or transition.get("branch_family"),
                **dict(nested),
            }
            expanded.append((f"{base_id}_{suffix}", materialized))
        for example_id, materialized in expanded:
            row = build_verifier_example(
                materialized,
                include_features=include_features,
                example_id=example_id,
            )
            if require_disposition_label and not row["labels"].get("candidate_disposition"):
                continue
            result.append(row)
    return result


build_transition_dataset = build_verifier_dataset


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, default=str) + "\n")


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if not isinstance(value, Mapping):
                raise ValueError(f"JSONL line {line_number} is not an object.")
            rows.append(dict(value))
    return rows


def _stable_group_score(group: Any, seed: int) -> int:
    payload = f"{seed}:{group}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest(), 16)


def split_dataset(
    rows: Iterable[Mapping[str, Any]],
    *,
    validation_fraction: float = 0.2,
    seed: int = 0,
    group_key: str | None = "root_scenario_id",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deterministically split rows, keeping scenarios together by default."""

    if not 0.0 <= validation_fraction <= 1.0:
        raise ValueError("validation_fraction must be in [0, 1].")
    materialized = [dict(row) for row in rows]
    if not materialized or validation_fraction == 0.0:
        return materialized, []
    if validation_fraction == 1.0:
        return [], materialized

    if group_key is None:
        shuffled = list(materialized)
        random.Random(seed).shuffle(shuffled)
        validation_size = max(1, round(len(shuffled) * validation_fraction))
        validation = shuffled[:validation_size]
        training = shuffled[validation_size:]
        return training, validation

    grouped: dict[str, list[dict[str, Any]]] = {}
    group_order: list[str] = []
    for index, row in enumerate(materialized):
        raw_group = row.get(group_key)
        if raw_group is None and group_key == "root_scenario_id":
            raw_group = row.get("scenario_id")
        group = str(raw_group) if raw_group is not None else f"__row_{index}"
        if group not in grouped:
            grouped[group] = []
            group_order.append(group)
        grouped[group].append(row)

    ordered_groups = sorted(group_order, key=lambda value: (_stable_group_score(value, seed), value))
    target = max(1, round(len(materialized) * validation_fraction))
    validation_groups: set[str] = set()
    count = 0
    for group in ordered_groups:
        if count >= target and validation_groups:
            break
        validation_groups.add(group)
        count += len(grouped[group])

    validation = [row for group in group_order if group in validation_groups for row in grouped[group]]
    training = [row for group in group_order if group not in validation_groups for row in grouped[group]]
    return training, validation


def dataset_statistics(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    total = 0
    labeled = 0
    dispositions: dict[str, int] = {}
    process_valid: dict[str, int] = {"true": 0, "false": 0, "unknown": 0}
    for row in rows:
        total += 1
        labels = _mapping(row.get("labels")) or extract_transition_labels(row)
        disposition = labels.get("candidate_disposition")
        if disposition is not None:
            labeled += 1
            key = str(getattr(disposition, "value", disposition))
            dispositions[key] = dispositions.get(key, 0) + 1
        value = labels.get("process_valid")
        if isinstance(value, str):
            lowered = value.strip().lower()
            parsed = True if lowered in {"true", "1", "yes"} else False if lowered in {"false", "0", "no"} else None
        else:
            parsed = None if value is None else bool(value)
        key = "unknown" if parsed is None else str(parsed).lower()
        process_valid[key] += 1
    return {
        "total_examples": total,
        "labeled_disposition_examples": labeled,
        "by_candidate_disposition": dispositions,
        "by_process_valid": process_valid,
    }


__all__ = [
    "LABEL_FIELDS",
    "build_transition_dataset",
    "build_verifier_dataset",
    "build_verifier_example",
    "dataset_statistics",
    "extract_transition_labels",
    "load_jsonl",
    "split_dataset",
    "transition_from_dagger_example",
    "write_jsonl",
]
