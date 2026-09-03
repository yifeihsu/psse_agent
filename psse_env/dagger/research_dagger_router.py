from __future__ import annotations

"""Small observable-state DAgger router for research-only closed-loop trials.

The Gemma repair checkpoints changed their LoRA weights without changing any of
the 27 greedy recovery actions.  This module provides a deliberately small
alternative policy: a multinomial linear router trained on the already
collected learner-visited states.  It predicts only the next canonical tool.
Arguments are then bound from policy-visible state aliases and advertised
``supported_corrections``; no scenario truth or offline audit fields are read at
inference time.

The exported model is plain JSON.  Runtime inference uses only the Python
standard library, so a failed GPU experiment can be diagnosed with a CPU-only
preflight and closed-loop replay.
"""

import argparse
import copy
import hashlib
import json
import math
import os
import tempfile
import threading
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from psse_env.dagger.policy_adapter import LocalAliasPolicyAdapter
from psse_env.dagger.protocol_bridge import (
    CANONICAL_TO_INTERNAL_TOOL,
    internal_to_canonical_action,
)
from psse_env.dagger.release_factories import _observable_candidate_disposition


ROUTER_CONTRACT = "research_dagger_observable_router_v1"
ROUTER_REPORT_CONTRACT = "research_dagger_observable_router_report_v1"
FEATURE_CONTRACT = "observable_dagger_router_semantic_features_v2"
DEFAULT_SEED = 3407
DEFAULT_C = 1.0

_RECOVERY_REQUIRED_TOOLS = frozenset(
    {
        "wls_from_path",
        "rollback_state",
        "correct_parameters_from_path",
        "get_measurement_context",
        "get_topology_context",
    }
)
_CONTEXT_TOOLS = frozenset(
    {
        "get_measurement_context",
        "get_parameter_context",
        "get_topology_context",
        "get_harmonic_context",
    }
)
_SIMPLE_ACTIVE_TOOLS = frozenset(
    {
        "run_hse_from_path",
        "run_three_phase_nlm_from_path",
    }
)
_CORRECTION_TOOLS = frozenset(
    {
        "correct_measurements_from_path",
        "correct_parameters_from_path",
        "correct_topology_from_path",
    }
)
_INTERNAL_TO_CANONICAL_FEATURE_TOOL = {
    "run_wls": "wls_from_path",
    "correct_measurements": "correct_measurements_from_path",
    "correct_parameters": "correct_parameters_from_path",
    "correct_topology": "correct_topology_from_path",
}
_TRIED_TOOL_NAMES = (
    "run_wls",
    "get_measurement_context",
    "get_parameter_context",
    "get_topology_context",
    "get_harmonic_context",
    "correct_measurements",
    "correct_parameters",
    "correct_topology",
    "commit_state",
    "rollback_state",
    "finalize_diagnosis",
    "ask_for_more_evidence",
)
_BOOLEAN_STATE_FIELDS = (
    "has_open_candidate",
    "has_unverified_candidate",
    "has_verified_candidate",
    "has_fresh_measurement_context",
    "has_fresh_parameter_context",
    "has_fresh_topology_context",
    "no_material_anomaly_remaining",
    "requires_measurement_context",
    "candidate_committed",
)
_VERIFICATION_FIELDS = (
    "globally_resolved",
    "post_action_resolved",
    "physical_constraints_ok",
    "target_test_passed",
    "converged",
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    except Exception:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def _canonical_tool_name(tool: Any) -> str:
    normalized = str(tool or "").strip()
    return _INTERNAL_TO_CANONICAL_FEATURE_TOOL.get(normalized, normalized)


def _categorical(features: dict[str, float], name: str, value: Any) -> None:
    features[f"{name}={value}"] = 1.0


def _numeric(features: dict[str, float], name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        value = 0
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"non-finite observable router feature: {name}")
    features[name] = normalized


def observable_router_features(observation: Mapping[str, Any]) -> dict[str, float]:
    """Extract only stable, policy-visible control semantics.

    Deliberately excluded are physical-root IDs, scenario IDs, injected-fault
    labels, private audit values, raw residual magnitudes, and root-specific
    numeric fingerprints.  This keeps the router focused on lifecycle,
    evidence availability, tool history, and observable action support.
    """

    if not isinstance(observation, Mapping):
        raise TypeError("observable DAgger router requires a state mapping")
    state = dict(observation)
    last_output = state.get("last_tool_output")
    last_output = dict(last_output) if isinstance(last_output, Mapping) else {}
    verification = state.get("last_verification")
    verification = dict(verification) if isinstance(verification, Mapping) else {}
    history = state.get("history_window")
    history = list(history) if isinstance(history, Sequence) and not isinstance(
        history, (str, bytes, bytearray)
    ) else []
    tried = state.get("tried_action_signatures")
    tried = list(tried) if isinstance(tried, Sequence) and not isinstance(
        tried, (str, bytes, bytearray)
    ) else []
    unresolved = state.get("unresolved_signatures")
    unresolved = list(unresolved) if isinstance(unresolved, Sequence) and not isinstance(
        unresolved, (str, bytes, bytearray)
    ) else []

    features: dict[str, float] = {}
    for field in (
        "candidate_lifecycle",
        "candidate_status",
        "last_tool",
        "last_tool_status",
    ):
        value = state.get(field)
        if field == "last_tool":
            value = _canonical_tool_name(value)
        _categorical(features, field, value)
    _categorical(features, "last_error", last_output.get("error_code"))
    _categorical(features, "last_execution", last_output.get("execution_status"))
    valid_next = last_output.get("valid_next_actions")
    if isinstance(valid_next, Sequence) and not isinstance(
        valid_next, (str, bytes, bytearray)
    ):
        for action in valid_next:
            if isinstance(action, Mapping):
                _categorical(
                    features,
                    "last_valid_next_tool",
                    _canonical_tool_name(action.get("tool")),
                )
    observable_metrics = last_output.get("observable_metrics")
    observable_metrics = (
        dict(observable_metrics) if isinstance(observable_metrics, Mapping) else {}
    )
    for field in (
        "globally_resolved",
        "post_action_resolved",
        "physical_constraints_ok",
    ):
        _categorical(
            features,
            f"last_metric_{field}",
            observable_metrics.get(field),
        )
    for field in _BOOLEAN_STATE_FIELDS:
        _categorical(features, field, state.get(field))

    _numeric(features, "history_n", len(history))
    _numeric(features, "tried_n", len(tried))
    _numeric(features, "unresolved_n", len(unresolved))

    evidence = state.get("available_evidence")
    if isinstance(evidence, Sequence) and not isinstance(
        evidence, (str, bytes, bytearray)
    ):
        for item in evidence:
            _categorical(features, "evidence", str(item))

    bounded_history = history[-6:]
    for offset, event in enumerate(bounded_history, start=-len(bounded_history)):
        event = dict(event) if isinstance(event, Mapping) else {}
        action = event.get("action")
        action = dict(action) if isinstance(action, Mapping) else {}
        outcome = event.get("outcome")
        if not isinstance(outcome, Mapping):
            outcome = event.get("tool_output")
        outcome = dict(outcome) if isinstance(outcome, Mapping) else {}
        tool = event.get("tool") or action.get("tool")
        _categorical(
            features,
            f"h{offset}.tool",
            _canonical_tool_name(tool),
        )
        _categorical(
            features,
            f"h{offset}.status",
            outcome.get("execution_status"),
        )
        _categorical(features, f"h{offset}.err", outcome.get("error_code"))

    string_tries = [str(value) for value in tried]
    for tool in _TRIED_TOOL_NAMES:
        count = sum(value.startswith(f"{tool}:") for value in string_tries)
        _numeric(features, f"try_{_canonical_tool_name(tool)}", count)

    contexts = state.get("fresh_context_evidence")
    contexts = dict(contexts) if isinstance(contexts, Mapping) else {}
    for context_name, raw_context in sorted(contexts.items(), key=lambda item: str(item[0])):
        context = dict(raw_context) if isinstance(raw_context, Mapping) else {}
        prefix = f"ctx_{context_name}"
        _categorical(features, f"{prefix}_route", context.get("route_status"))
        supported = context.get("supported_corrections")
        supported = list(supported) if isinstance(supported, Sequence) and not isinstance(
            supported, (str, bytes, bytearray)
        ) else []
        _numeric(features, f"{prefix}_supported_n", len(supported))
        for action in supported:
            if isinstance(action, Mapping):
                _categorical(
                    features,
                    f"{prefix}_supports",
                    _canonical_tool_name(action.get("tool")),
                )

    for field in _VERIFICATION_FIELDS:
        _categorical(features, f"verification_{field}", verification.get(field))
    sufficiency = verification.get("evidence_sufficiency")
    sufficiency = dict(sufficiency) if isinstance(sufficiency, Mapping) else {}
    _categorical(
        features,
        "verification_sufficient",
        sufficiency.get("sufficient"),
    )
    return features


def _row_state(row: Mapping[str, Any]) -> dict[str, Any]:
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise ValueError("training row lacks canonical messages")
    users = [message for message in messages if message.get("role") == "user"]
    if len(users) != 1 or not isinstance(users[0].get("content"), str):
        raise ValueError("training row must contain one canonical user message")
    payload = json.loads(users[0]["content"])
    if not isinstance(payload, Mapping) or not isinstance(payload.get("state"), Mapping):
        raise ValueError("canonical user message does not contain a state mapping")
    return copy.deepcopy(dict(payload["state"]))


def _row_prompt_key(row: Mapping[str, Any]) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise ValueError("training row lacks canonical messages")
    visible = [
        {"role": message.get("role"), "content": message.get("content")}
        for message in messages
        if message.get("role") in {"system", "user"}
    ]
    return _stable_sha256(visible)


def _row_target(row: Mapping[str, Any]) -> dict[str, Any]:
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise ValueError("training row lacks canonical messages")
    assistants = [message for message in messages if message.get("role") == "assistant"]
    if len(assistants) != 1:
        raise ValueError("training row must contain one assistant target")
    calls = assistants[0].get("tool_calls")
    if not isinstance(calls, list) or len(calls) != 1:
        raise ValueError("training row must contain one assistant tool call")
    function = calls[0].get("function")
    if not isinstance(function, Mapping):
        raise ValueError("assistant target lacks a function mapping")
    tool = str(function.get("name") or "").strip()
    arguments = function.get("arguments")
    if not tool or not isinstance(arguments, Mapping):
        raise ValueError("assistant target tool or arguments are invalid")
    return {"tool": tool, "arguments": copy.deepcopy(dict(arguments))}


def _row_root(row: Mapping[str, Any]) -> str:
    root = str(row.get("physical_root_fingerprint") or "").strip()
    if not root:
        raise ValueError("selected router row lacks a physical_root_fingerprint")
    return root


def _collect_protected_roots(path: Path) -> set[str]:
    values: list[Any]
    if path.suffix.lower() == ".jsonl":
        values = _read_jsonl(path)
    else:
        values = [json.loads(path.read_text(encoding="utf-8"))]
    roots: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            root = value.get("physical_root_fingerprint")
            if isinstance(root, str) and root.strip():
                roots.add(root.strip())
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    for value in values:
        visit(value)
    if not roots:
        raise ValueError(f"protected source contains no physical roots: {path}")
    return roots


@dataclass(frozen=True)
class LinearRouter:
    classes: tuple[str, ...]
    feature_names: tuple[str, ...]
    coefficients: tuple[tuple[float, ...], ...]
    intercepts: tuple[float, ...]

    def predict(self, observation: Mapping[str, Any]) -> tuple[str, float, dict[str, float]]:
        values = observable_router_features(observation)
        scores: list[float] = []
        for row, intercept in zip(self.coefficients, self.intercepts, strict=True):
            score = intercept
            for index, name in enumerate(self.feature_names):
                value = values.get(name)
                if value:
                    score += row[index] * value
            if not math.isfinite(score):
                raise ValueError("observable router produced a non-finite score")
            scores.append(score)
        winner = max(range(len(scores)), key=lambda index: (scores[index], -index))
        ordered = sorted(scores, reverse=True)
        margin = ordered[0] - ordered[1] if len(ordered) > 1 else math.inf
        return (
            self.classes[winner],
            float(margin),
            {tool: float(score) for tool, score in zip(self.classes, scores, strict=True)},
        )


def _validate_artifact(value: Mapping[str, Any]) -> LinearRouter:
    if value.get("contract") != ROUTER_CONTRACT:
        raise ValueError("observable router artifact contract mismatch")
    if value.get("feature_contract") != FEATURE_CONTRACT:
        raise ValueError("observable router feature contract mismatch")
    classes = value.get("classes")
    feature_names = value.get("feature_names")
    coefficients = value.get("coefficients")
    intercepts = value.get("intercepts")
    if not isinstance(classes, list) or len(classes) < 2:
        raise ValueError("observable router artifact has too few classes")
    if classes != sorted(set(str(item) for item in classes)):
        raise ValueError("observable router classes must be unique and sorted")
    if not isinstance(feature_names, list) or not feature_names:
        raise ValueError("observable router artifact has no features")
    if feature_names != sorted(set(str(item) for item in feature_names)):
        raise ValueError("observable router features must be unique and sorted")
    if not isinstance(coefficients, list) or len(coefficients) != len(classes):
        raise ValueError("observable router coefficient row count is invalid")
    if not isinstance(intercepts, list) or len(intercepts) != len(classes):
        raise ValueError("observable router intercept count is invalid")
    normalized_rows: list[tuple[float, ...]] = []
    for row in coefficients:
        if not isinstance(row, list) or len(row) != len(feature_names):
            raise ValueError("observable router coefficient width is invalid")
        normalized = tuple(float(item) for item in row)
        if not all(math.isfinite(item) for item in normalized):
            raise ValueError("observable router coefficients must be finite")
        normalized_rows.append(normalized)
    normalized_intercepts = tuple(float(item) for item in intercepts)
    if not all(math.isfinite(item) for item in normalized_intercepts):
        raise ValueError("observable router intercepts must be finite")
    return LinearRouter(
        classes=tuple(str(item) for item in classes),
        feature_names=tuple(str(item) for item in feature_names),
        coefficients=tuple(normalized_rows),
        intercepts=normalized_intercepts,
    )


def load_router_artifact(path: Path, revision: str) -> LinearRouter:
    resolved = path.expanduser()
    if not resolved.is_absolute():
        raise ValueError("observable router model_id must be an absolute path")
    resolved = resolved.resolve(strict=True)
    normalized_revision = str(revision).strip().lower()
    if len(normalized_revision) != 64 or any(
        character not in "0123456789abcdef" for character in normalized_revision
    ):
        raise ValueError("observable router revision must be a lowercase SHA-256 digest")
    if _file_sha256(resolved) != normalized_revision:
        raise ValueError("observable router artifact digest does not match model_revision")
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("observable router artifact is not a JSON object")
    model = _validate_artifact(value)
    if _file_sha256(resolved) != normalized_revision:
        raise ValueError("observable router artifact changed while it was loaded")
    return model


def _state_reference(observation: Mapping[str, Any], field: str, fallback: str) -> str:
    value = str(observation.get(field) or "").strip()
    return value or fallback


def _safe_request(observation: Mapping[str, Any], reason: str) -> dict[str, Any]:
    active = _state_reference(observation, "active_state_id", "active")
    candidate = str(observation.get("candidate_state_id") or "").strip()
    case_path = candidate if observation.get("has_open_candidate") and candidate else active
    arguments: dict[str, Any] = {"case_path": case_path}
    if not candidate:
        # This is the controller's accepted terminal handoff request.  Arbitrary
        # diagnostic prose is rejected and previously created avoidable loops.
        arguments["request"] = "operator_escalation:recovery_options_exhausted"
    return {"tool": "ask_for_more_evidence", "arguments": arguments}


def _visible_supported_actions(observation: Mapping[str, Any]) -> Iterable[dict[str, Any]]:
    contexts = observation.get("fresh_context_evidence")
    contexts = dict(contexts) if isinstance(contexts, Mapping) else {}
    for raw_context in contexts.values():
        context = dict(raw_context) if isinstance(raw_context, Mapping) else {}
        actions = context.get("supported_corrections")
        if isinstance(actions, Sequence) and not isinstance(
            actions, (str, bytes, bytearray)
        ):
            for action in actions:
                if isinstance(action, Mapping):
                    yield copy.deepcopy(dict(action))
    last_output = observation.get("last_tool_output")
    last_output = dict(last_output) if isinstance(last_output, Mapping) else {}
    actions = last_output.get("valid_next_actions")
    if isinstance(actions, Sequence) and not isinstance(actions, (str, bytes, bytearray)):
        for action in actions:
            if isinstance(action, Mapping):
                yield copy.deepcopy(dict(action))


def bind_observable_router_action(
    predicted_tool: str,
    observation: Mapping[str, Any],
) -> tuple[dict[str, Any], str | None]:
    """Bind a predicted canonical tool using only the visible observation."""

    tool = str(predicted_tool).strip()
    active = _state_reference(observation, "active_state_id", "active")
    candidate = str(observation.get("candidate_state_id") or "").strip()
    lifecycle = str(observation.get("candidate_lifecycle") or "")
    verification = observation.get("last_verification")
    verification = dict(verification) if isinstance(verification, Mapping) else {}
    evidence_sufficiency = verification.get("evidence_sufficiency")
    evidence_sufficiency = (
        dict(evidence_sufficiency)
        if isinstance(evidence_sufficiency, Mapping)
        else {}
    )

    last_output = observation.get("last_tool_output")
    last_output = dict(last_output) if isinstance(last_output, Mapping) else {}
    advertised = last_output.get("valid_next_actions")
    advertised = list(advertised) if isinstance(advertised, Sequence) and not isinstance(
        advertised, (str, bytes, bytearray)
    ) else []
    advertised_tools = {
        str(action.get("tool"))
        for action in advertised
        if isinstance(action, Mapping) and action.get("tool")
    }

    advertised_commit_bound = False
    for advertised_action in advertised:
        if not isinstance(advertised_action, Mapping):
            continue
        if str(advertised_action.get("tool") or "") != "commit_state":
            continue
        arguments = advertised_action.get("arguments")
        arguments = dict(arguments) if isinstance(arguments, Mapping) else {}
        advertised_target = str(
            arguments.get("candidate_state_id") or arguments.get("case_path") or ""
        ).strip()
        if candidate and advertised_target == candidate:
            advertised_commit_bound = True
            break

    # The release expert consumes the unsummarized controller history, whereas
    # this router receives the equivalent model-visible history contract.  Map
    # only those visible fields back into the shape expected by the canonical
    # disposition helper so there is one closure rule, not a second threshold
    # implementation that can drift from it.
    visible_history: list[dict[str, Any]] = []
    history_window = observation.get("history_window")
    if isinstance(history_window, Sequence) and not isinstance(
        history_window, (str, bytes, bytearray)
    ):
        for event in history_window:
            if not isinstance(event, Mapping):
                continue
            if isinstance(event.get("action"), Mapping) and isinstance(
                event.get("tool_output"), Mapping
            ):
                visible_history.append(copy.deepcopy(dict(event)))
                continue
            event_tool = str(event.get("tool") or "").strip()
            if not event_tool:
                continue
            event_arguments = event.get("arguments")
            outcome = event.get("outcome")
            tool_output = dict(outcome) if isinstance(outcome, Mapping) else {}
            observable_metrics = event.get("observable_metrics")
            if isinstance(observable_metrics, Mapping):
                tool_output.update(dict(observable_metrics))
            visible_history.append(
                {
                    "action": {
                        "tool": CANONICAL_TO_INTERNAL_TOOL.get(event_tool, event_tool),
                        "arguments": (
                            copy.deepcopy(dict(event_arguments))
                            if isinstance(event_arguments, Mapping)
                            else {}
                        ),
                    },
                    "tool_output": tool_output,
                }
            )

    canonical_disposition = _observable_candidate_disposition(
        observation, visible_history
    )
    observably_accepted = False

    # Candidate disposition is a lifecycle invariant, not a learned guess.  A
    # rich WLS result used to lose these booleans during canonical compaction;
    # now that the learner can see them, reuse the release expert's exact
    # closure before considering the router's less certain class score.
    if lifecycle == "VERIFIED_CANDIDATE" and candidate:
        explicit_final_closure = any(
            verification.get(field) is True
            for field in (
                "globally_resolved",
                "post_action_resolved",
                "target_fixed",
                "target_test_passed",
            )
        )
        observably_accepted = (
            verification.get("physical_constraints_ok") is not False
            and evidence_sufficiency.get("sufficient") is not False
            and (
                canonical_disposition == "commit"
                or (
                    canonical_disposition in {None, "inconclusive"}
                    and explicit_final_closure
                )
                or advertised_commit_bound
            )
        )
        if observably_accepted:
            return {
                "tool": "commit_state",
                "arguments": {"case_path": candidate},
            }, "verified_candidate_commit_closure"
        if predicted_tool == "commit_state":
            # A classifier must never turn a visibly unresolved candidate into
            # a commit.  Roll back when the evidence is decisively negative;
            # otherwise ask for more evidence.
            if canonical_disposition == "rollback" or (
                canonical_disposition in {None, "inconclusive"}
                and verification.get("globally_resolved") is False
            ):
                return {
                    "tool": "rollback_state",
                    "arguments": {"case_path": candidate},
                }, "unsafe_commit_redirected_to_rollback"
            return _safe_request(observation, "commit_evidence_inconclusive"), (
                "commit_evidence_inconclusive"
            )

    if (
        lifecycle == "NO_CANDIDATE"
        and observation.get("has_open_candidate") is not True
        and observation.get("no_material_anomaly_remaining") is True
    ):
        return {"tool": "finalize_diagnosis", "arguments": {}}, (
            "observable_terminal_closure"
        )

    if tool == "wls_from_path":
        target = candidate if lifecycle == "OPEN_UNVERIFIED_CANDIDATE" and candidate else active
        return {"tool": tool, "arguments": {"case_path": target}}, None
    if tool in _CONTEXT_TOOLS:
        return {"tool": tool, "arguments": {"case_path": active}}, None
    if tool in _SIMPLE_ACTIVE_TOOLS:
        return {"tool": tool, "arguments": {"case_path": active}}, None
    if tool == "ask_for_more_evidence":
        target = candidate if observation.get("has_open_candidate") and candidate else active
        arguments: dict[str, Any] = {"case_path": target}
        if not candidate:
            arguments["request"] = "operator_escalation:recovery_options_exhausted"
        return {"tool": tool, "arguments": arguments}, None
    if tool == "rollback_state":
        if (
            candidate
            and observation.get("has_verified_candidate") is True
            and (
                canonical_disposition == "rollback"
                or (
                    canonical_disposition in {None, "inconclusive"}
                    and verification.get("globally_resolved") is False
                )
            )
        ):
            # The accepted-candidate closure above has already consumed every
            # affirmative signal (target closure, progress, or advertised
            # commit).  A remaining verified candidate with explicit global
            # failure is therefore the observable rollback case.  This also
            # keeps compatibility with older compact traces that retained the
            # global decision but accidentally dropped the richer disposition
            # fields.
            return {"tool": tool, "arguments": {"case_path": candidate}}, None
        return _safe_request(observation, "rollback_not_observably_rejected"), (
            "rollback_not_observably_rejected"
        )
    if tool == "commit_state":
        if candidate and observation.get("has_verified_candidate") is True and observably_accepted:
            return {"tool": tool, "arguments": {"case_path": candidate}}, None
        return _safe_request(observation, "commit_not_observably_accepted"), (
            "commit_not_observably_accepted"
        )
    if tool == "finalize_diagnosis":
        if (
            observation.get("no_material_anomaly_remaining") is True
            and observation.get("has_open_candidate") is not True
        ):
            return {"tool": tool, "arguments": {}}, None
        return _safe_request(observation, "finalize_not_observably_resolved"), (
            "finalize_not_observably_resolved"
        )
    if tool in _CORRECTION_TOOLS:
        expected_internal = CANONICAL_TO_INTERNAL_TOOL.get(tool)
        if observation.get("has_open_candidate") is True:
            return _safe_request(observation, "correction_with_open_candidate"), (
                "correction_with_open_candidate"
            )
        for internal_action in _visible_supported_actions(observation):
            if internal_action.get("tool") != expected_internal:
                continue
            try:
                return internal_to_canonical_action(internal_action), None
            except (TypeError, ValueError):
                continue
        return _safe_request(observation, "correction_not_visibly_supported"), (
            "correction_not_visibly_supported"
        )

    # The two HIF estimators require arguments not reliably recoverable from
    # the compact generic observation.  Falling back is safer than inventing a
    # branch or scan-window identifier.  Their existing D0 supervision remains
    # in the router so confidence on adjacent state classes is not distorted.
    return _safe_request(observation, f"unbound_tool:{tool}"), f"unbound_tool:{tool}"


class _CanonicalObservableRouterPolicy:
    def __init__(self, model: LinearRouter) -> None:
        self._model = model
        self._last_action_metrics: dict[str, Any] = {}

    @property
    def last_action_metrics(self) -> dict[str, Any]:
        return copy.deepcopy(self._last_action_metrics)

    def act(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        predicted_tool, margin, scores = self._model.predict(observation)
        action, guard = bind_observable_router_action(predicted_tool, observation)
        self._last_action_metrics = {
            "router_contract": ROUTER_CONTRACT,
            "feature_contract": FEATURE_CONTRACT,
            "predicted_tool": predicted_tool,
            "selected_tool": action["tool"],
            "score_margin": margin,
            "guard_fallback": guard,
            "feature_count": len(observable_router_features(observation)),
            "class_scores": scores,
        }
        return action


class ResearchDaggerRouterPolicy:
    """Root-disjoint learner-trace router with canonical alias binding."""

    def __init__(self, model: LinearRouter, *, model_id: str, model_revision: str) -> None:
        self._canonical = _CanonicalObservableRouterPolicy(model)
        self._adapter = LocalAliasPolicyAdapter(self._canonical, protocol="canonical")
        self._model_id = model_id
        self._model_revision = model_revision

    @property
    def release_policy_identity(self) -> dict[str, str | None]:
        return {
            "explicit_policy_identity": None,
            "model_id": self._model_id,
            "model_revision": self._model_revision,
        }

    @property
    def last_action_metrics(self) -> dict[str, Any]:
        return self._canonical.last_action_metrics

    def act(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        return self._adapter.act(copy.deepcopy(dict(observation)))


_MODEL_CACHE: dict[tuple[str, str], LinearRouter] = {}
_MODEL_CACHE_LOCK = threading.Lock()


def research_dagger_router_policy_factory(
    *,
    model_id: str | None = None,
    model_revision: str | None = None,
    seed: int | None = None,
    rng: Any | None = None,
) -> ResearchDaggerRouterPolicy:
    del seed, rng
    normalized_id = str(model_id or "").strip()
    normalized_revision = str(model_revision or "").strip().lower()
    if not normalized_id or not normalized_revision:
        raise ValueError("observable router requires model_id and model_revision")
    path = Path(normalized_id).expanduser()
    if not path.is_absolute():
        raise ValueError("observable router model_id must be an absolute path")
    resolved = str(path.resolve(strict=True))
    key = (resolved, normalized_revision)
    with _MODEL_CACHE_LOCK:
        model = _MODEL_CACHE.get(key)
        if model is None:
            model = load_router_artifact(Path(resolved), normalized_revision)
            _MODEL_CACHE[key] = model
    return ResearchDaggerRouterPolicy(
        model,
        model_id=resolved,
        model_revision=normalized_revision,
    )


def _deduplicate_training_rows(
    rows: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    by_prompt: dict[str, dict[str, Any]] = {}
    conflicts = 0
    for raw_row in rows:
        row = copy.deepcopy(dict(raw_row))
        key = _row_prompt_key(row)
        previous = by_prompt.get(key)
        if previous is not None and _row_target(previous) != _row_target(row):
            conflicts += 1
            raise ValueError(
                "same visible router prompt has conflicting expert targets"
            )
        by_prompt[key] = row
    return [by_prompt[key] for key in sorted(by_prompt)], conflicts


def _fit_linear_router(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    inverse_regularization: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        import sklearn
        from sklearn.feature_extraction import DictVectorizer
        from sklearn.linear_model import LogisticRegression
    except Exception as exc:  # pragma: no cover - optional research dependency.
        raise RuntimeError(f"scikit-learn is required to fit the router: {exc}") from exc

    if not rows:
        raise ValueError("observable router training view is empty")
    states = [_row_state(row) for row in rows]
    targets = [_row_target(row)["tool"] for row in rows]
    classes = sorted(set(targets))
    if len(classes) < 3:
        raise ValueError("observable router requires at least three target tools")
    missing = sorted(_RECOVERY_REQUIRED_TOOLS - set(classes))
    if missing:
        raise ValueError(f"observable router training view misses recovery tools: {missing}")
    if inverse_regularization <= 0 or not math.isfinite(inverse_regularization):
        raise ValueError("router inverse regularization must be finite and positive")

    vectorizer = DictVectorizer(sparse=True, sort=True)
    matrix = vectorizer.fit_transform(observable_router_features(state) for state in states)
    classifier = LogisticRegression(
        C=float(inverse_regularization),
        class_weight="balanced",
        max_iter=5000,
        random_state=int(seed),
        solver="lbfgs",
    )
    classifier.fit(matrix, targets)
    if max(int(value) for value in classifier.n_iter_) >= int(classifier.max_iter):
        raise RuntimeError("observable router logistic fit did not converge")
    learned_classes = [str(value) for value in classifier.classes_.tolist()]
    if learned_classes != classes:
        raise RuntimeError("observable router class order is not deterministic")
    feature_names = [str(value) for value in vectorizer.get_feature_names_out().tolist()]
    coefficients = [[float(value) for value in row] for row in classifier.coef_.tolist()]
    intercepts = [float(value) for value in classifier.intercept_.tolist()]
    if len(coefficients) != len(classes):
        raise RuntimeError("unexpected binary observable router coefficient layout")

    model = {
        "contract": ROUTER_CONTRACT,
        "artifact_type": "research_only_observable_dagger_action_router",
        "research_only": True,
        "release_eligible": False,
        "release_ineligibility_reasons": [
            "research prototype distilled from learner-trace supervision",
            "full preregistered multi-seed study remains outstanding",
        ],
        "feature_contract": FEATURE_CONTRACT,
        "classes": classes,
        "feature_names": feature_names,
        "coefficients": coefficients,
        "intercepts": intercepts,
        "fit": {
            "algorithm": "multinomial_logistic_regression",
            "inverse_regularization": float(inverse_regularization),
            "class_weight": "balanced",
            "max_iter": 5000,
            "seed": int(seed),
            "sklearn_version": str(sklearn.__version__),
        },
    }
    validated = _validate_artifact(model)
    train_predictions = [validated.predict(state)[0] for state in states]
    training_correct = sum(
        predicted == expected for predicted, expected in zip(train_predictions, targets, strict=True)
    )
    diagnostics = {
        "row_count": len(rows),
        "class_distribution": dict(sorted(Counter(targets).items())),
        "feature_count": len(feature_names),
        "training_tool_match_count": training_correct,
        "training_tool_match_rate": training_correct / len(rows),
        "iterations": [int(value) for value in classifier.n_iter_.tolist()],
    }
    return model, diagnostics


def _evaluate_validation_rows(
    model: LinearRouter,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    from psse_env.dagger.preliminary_tool_gate import evaluate_generation

    results: list[dict[str, Any]] = []
    for row in rows:
        state = _row_state(row)
        predicted_tool, margin, scores = model.predict(state)
        action, guard = bind_observable_router_action(predicted_tool, state)
        generated_text = json.dumps(
            {"name": action["tool"], "arguments": action["arguments"]},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        metrics = {
            "predicted_tool": predicted_tool,
            "selected_tool": action["tool"],
            "score_margin": margin,
            "guard_fallback": guard,
            "class_scores": scores,
        }
        result = evaluate_generation(row, generated_text, action_metrics=metrics)
        results.append(result)

    count = len(results)
    truth_count = lambda field: sum(result.get(field) is True for result in results)
    overall = {
        "row_count": count,
        "schema_valid_count": truth_count("schema_valid"),
        "state_bound_count": truth_count("state_bound"),
        "target_tool_match_count": truth_count("target_tool_match"),
        "exact_target_match_count": truth_count("exact_target_match"),
    }
    for field in (
        "schema_valid",
        "state_bound",
        "target_tool_match",
        "exact_target_match",
    ):
        overall[f"{field}_rate"] = (
            truth_count(field) / count if count else 0.0
        )
    by_tool: dict[str, list[Mapping[str, Any]]] = {}
    for result in results:
        expected = result.get("expected_action")
        tool = str(expected.get("tool")) if isinstance(expected, Mapping) else "<missing>"
        by_tool.setdefault(tool, []).append(result)
    per_tool = {
        tool: {
            "expected_count": len(tool_rows),
            "tool_match_count": sum(
                row.get("target_tool_match") is True for row in tool_rows
            ),
            "exact_match_count": sum(
                row.get("exact_target_match") is True for row in tool_rows
            ),
        }
        for tool, tool_rows in sorted(by_tool.items())
    }
    return {"overall": overall, "per_expected_tool": per_tool, "results": results}


def fit_router_artifact(
    *,
    anchor_path: Path,
    trace_path: Path | Sequence[Path],
    artifact_output: Path,
    report_output: Path,
    validation_path: Path | None = None,
    protected_paths: Sequence[Path] = (),
    seed: int = DEFAULT_SEED,
    inverse_regularization: float = DEFAULT_C,
) -> dict[str, Any]:
    anchor = anchor_path.expanduser().resolve(strict=True)
    raw_trace_paths = (
        [trace_path]
        if isinstance(trace_path, Path)
        else list(trace_path)
    )
    if not raw_trace_paths:
        raise ValueError("at least one learner-trace source is required")
    traces = [path.expanduser().resolve(strict=True) for path in raw_trace_paths]
    if len(set(traces)) != len(traces):
        raise ValueError("learner-trace source paths must be unique")
    validation = (
        validation_path.expanduser().resolve(strict=True)
        if validation_path is not None
        else None
    )
    protected = [path.expanduser().resolve(strict=True) for path in protected_paths]
    if int(seed) != DEFAULT_SEED:
        raise ValueError(f"research router seed is frozen at {DEFAULT_SEED}")

    anchor_rows = _read_jsonl(anchor)
    trace_sources = [(path, _read_jsonl(path)) for path in traces]
    trace_all = [row for _path, rows in trace_sources for row in rows]
    trace_rows = [
        row
        for row in trace_all
        if row.get("dataset_mode") == "research_dagger_learner_trace"
        and row.get("state_origin") == "learner_policy"
        and row.get("state_visited_by") == "learner_policy"
    ]
    if not trace_rows:
        raise ValueError("trace source has no learner-visited research rows")
    selected_rows, conflict_count = _deduplicate_training_rows(
        [*anchor_rows, *trace_rows]
    )
    training_roots = {_row_root(row) for row in selected_rows}

    protected_roots: set[str] = set()
    protected_bindings: list[dict[str, Any]] = []
    for path in protected:
        roots = _collect_protected_roots(path)
        protected_roots.update(roots)
        protected_bindings.append(
            {
                "path": str(path),
                "sha256": _file_sha256(path),
                "physical_root_count": len(roots),
                "physical_roots": sorted(roots),
            }
        )
    if validation is not None:
        validation_rows = _read_jsonl(validation)
        validation_roots = {_row_root(row) for row in validation_rows}
        protected_roots.update(validation_roots)
    else:
        validation_rows = []
        validation_roots = set()
    overlap = sorted(training_roots & protected_roots)
    if overlap:
        raise ValueError(f"router training roots overlap protected roots: {overlap}")

    artifact, training_diagnostics = _fit_linear_router(
        selected_rows,
        seed=int(seed),
        inverse_regularization=float(inverse_regularization),
    )
    artifact["training_binding"] = {
        "anchor_sha256": _file_sha256(anchor),
        "trace_sha256": [
            {"path": str(path), "sha256": _file_sha256(path)}
            for path in traces
        ],
        "selected_prompt_digest": _stable_sha256(
            [_row_prompt_key(row) for row in selected_rows]
        ),
        "selected_row_count": len(selected_rows),
        "physical_root_count": len(training_roots),
    }
    _validate_artifact(artifact)
    artifact_path = artifact_output.expanduser().resolve()
    report_path = report_output.expanduser().resolve()
    _write_json_atomic(artifact_path, artifact)
    artifact_sha256 = _file_sha256(artifact_path)
    loaded = load_router_artifact(artifact_path, artifact_sha256)

    validation_evaluation = (
        _evaluate_validation_rows(loaded, validation_rows)
        if validation_rows
        else None
    )
    report: dict[str, Any] = {
        "contract": ROUTER_REPORT_CONTRACT,
        "artifact_type": "research_only_observable_dagger_router_fit_report",
        "research_only": True,
        "artifact_path": str(artifact_path),
        "artifact_sha256": artifact_sha256,
        "sources": {
            "anchor": {
                "path": str(anchor),
                "sha256": _file_sha256(anchor),
                "row_count": len(anchor_rows),
            },
            "trace": [
                {
                    "path": str(path),
                    "sha256": _file_sha256(path),
                    "row_count": len(rows),
                    "selected_learner_row_count": sum(
                        row.get("dataset_mode") == "research_dagger_learner_trace"
                        and row.get("state_origin") == "learner_policy"
                        and row.get("state_visited_by") == "learner_policy"
                        for row in rows
                    ),
                }
                for path, rows in trace_sources
            ],
            "validation": (
                {
                    "path": str(validation),
                    "sha256": _file_sha256(validation),
                    "row_count": len(validation_rows),
                    "physical_root_count": len(validation_roots),
                    "physical_roots": sorted(validation_roots),
                }
                if validation is not None
                else None
            ),
            "protected": protected_bindings,
        },
        "training": {
            **training_diagnostics,
            "raw_row_count": len(anchor_rows) + len(trace_rows),
            "deduplicated_row_count": len(selected_rows),
            "conflicting_prompt_count": conflict_count,
            "physical_root_count": len(training_roots),
            "physical_roots": sorted(training_roots),
            "protected_root_overlap": overlap,
        },
        "validation": validation_evaluation,
    }
    _write_json_atomic(report_path, report)
    return report


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit a CPU observable-state router from saved DAgger traces"
    )
    parser.add_argument("--anchor", required=True, type=Path)
    parser.add_argument(
        "--trace",
        required=True,
        action="append",
        type=Path,
        help="learner-trace JSONL; repeat to aggregate DAgger iterations",
    )
    parser.add_argument("--validation", type=Path)
    parser.add_argument("--protected", action="append", default=[], type=Path)
    parser.add_argument("--artifact-output", required=True, type=Path)
    parser.add_argument("--report-output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--inverse-regularization", type=float, default=DEFAULT_C)
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    report = fit_router_artifact(
        anchor_path=args.anchor,
        trace_path=args.trace,
        validation_path=args.validation,
        protected_paths=args.protected,
        artifact_output=args.artifact_output,
        report_output=args.report_output,
        seed=args.seed,
        inverse_regularization=args.inverse_regularization,
    )
    validation = report.get("validation")
    overall = validation.get("overall") if isinstance(validation, Mapping) else None
    print(
        json.dumps(
            {
                "artifact_path": report["artifact_path"],
                "artifact_sha256": report["artifact_sha256"],
                "training": report["training"],
                "validation_overall": overall,
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()


__all__ = [
    "FEATURE_CONTRACT",
    "ROUTER_CONTRACT",
    "ROUTER_REPORT_CONTRACT",
    "LinearRouter",
    "ResearchDaggerRouterPolicy",
    "bind_observable_router_action",
    "fit_router_artifact",
    "load_router_artifact",
    "observable_router_features",
    "research_dagger_router_policy_factory",
]
