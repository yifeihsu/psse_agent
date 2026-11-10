"""How much of an episode's tool history the policy is shown (workplan of 2026-11-10, stage 1).

``bounded_history_v1`` is the mode of every recorded cell: the evaluator hands
the policy the last four transitions, and the model-visible view keeps of each
its action, its outcome and a compacted summary of what it returned, without
the per-bus and per-branch tables of a solve, inside a budget of eight events
and 4,096 characters (``psse_env.dagger.evaluator.STUDY_POLICY_HISTORY_WINDOW``,
``psse_env.dagger.dataset_builder.summarize_history``).  A tool result is shown
in full once, as the last tool output, and leaves the prompt four actions
later; the tables leave it after one.

``complete_history_v1`` keeps every transition of the episode.  The evaluator
adds the whole observable transition stream to the policy's observation under
:data:`OBSERVABLE_HISTORY_KEY` (the bounded window stays where it was, so the
evaluator's own objective metrics and a rule expert read what they always
read), and the model-visible view renders every event with the evidence the
model was shown when that event's output was the latest one
(``dataset_builder.prepare_complete_history_observation``).  Nothing the
tools did not return is added: a solve's tables are still the provider's
bounded tables.

The mode is carried by the observation itself, so a renderer cannot be set to
one mode while the evaluator runs the other: the adapter renders the complete
view exactly when the observation holds the complete record, the system
prompt gains its paragraph exactly when the model-visible view carries
:data:`HISTORY_RECORD_KEY`, and the RL update re-renders the stored
model-visible view as it is.

The stream is the evaluator's ``history``: policy-safe copies of each executed
action and its tool output with the active and candidate state at the time.
Offline truth, verdicts and audit assessments are stripped from it by
``policy_safe_copy`` before it is ever appended, as for the bounded window.
"""
from __future__ import annotations

import copy
from typing import Any, Mapping, Sequence

BOUNDED_HISTORY = "bounded_history_v1"
COMPLETE_HISTORY = "complete_history_v1"
OBSERVATION_MODES = (BOUNDED_HISTORY, COMPLETE_HISTORY)
#: Where the evaluator puts the episode's transition stream in the policy's observation (complete mode only):
#: ``{"contract": COMPLETE_HISTORY, "events": [transition, ...]}``.
OBSERVABLE_HISTORY_KEY = "observable_history"
#: The marker in the model-visible view of a complete-history observation: ``{"contract": ..., "events": n}``.
HISTORY_RECORD_KEY = "history_record"
#: The model-visible list of the candidates tried so far (complete mode only).
CANDIDATE_EXPERIMENTS_KEY = "candidate_experiments"


def validate_observation_mode(mode: Any) -> str:
    if mode not in OBSERVATION_MODES:
        raise ValueError(f"observation_mode must be one of {OBSERVATION_MODES}, got {mode!r}")
    return str(mode)


def complete_history_record(observation: Mapping[str, Any] | Any) -> Mapping[str, Any] | None:
    """The complete-history record of a policy observation (the evaluator's), else ``None``."""

    getter = getattr(observation, "get", None)
    record = getter(OBSERVABLE_HISTORY_KEY) if callable(getter) else None
    return record if isinstance(record, Mapping) and record.get("contract") == COMPLETE_HISTORY else None


def observable_history_of(observation: Mapping[str, Any] | Any) -> list[dict[str, Any]] | None:
    """The transition stream a complete-history observation carries; ``None`` in the bounded mode.

    A complete-history observation stored without its events (the evaluator's
    trace keeps only their number) raises: rendering it from the bounded
    window would give a prompt the policy was never shown.
    """

    record = complete_history_record(observation)
    if record is None:
        return None
    events = record.get("events")
    if not isinstance(events, list) or any(not isinstance(item, Mapping) for item in events):
        raise ValueError(
            "the observation was recorded under complete_history_v1 without its events; rebuild them from the "
            "episode's trace (observable_history_from_trace) and pass them as the history"
        )
    return [dict(item) for item in events]


def with_observable_history(observation: Mapping[str, Any], history: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """``observation`` with the whole transition stream attached (a copy; the stream is copied too)."""

    bound = dict(observation)
    bound[OBSERVABLE_HISTORY_KEY] = {"contract": COMPLETE_HISTORY, "events": [copy.deepcopy(dict(item)) for item in history]}
    return bound


def stored_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    """What a trace keeps of a policy observation: the complete record shrinks to its event count.

    The events of step ``t`` are the transitions of the steps before it,
    which the trace holds once each; keeping them again inside every step's
    observation would make an episode's record grow with the square of its
    length.  :func:`observable_history_from_trace` gives them back.
    """

    record = complete_history_record(observation)
    if record is None:
        return dict(observation)
    stored = dict(observation)
    events = record.get("events")
    stored[OBSERVABLE_HISTORY_KEY] = {"contract": COMPLETE_HISTORY,
                                      "event_count": len(events) if isinstance(events, list) else record.get("event_count")}
    return stored


def observable_history_from_trace(trace: Sequence[Mapping[str, Any]], before_step: int | None = None) -> list[dict[str, Any]]:
    """The transition stream the policy was shown at trace position ``before_step`` (all of it when ``None``).

    Rebuilt from the evaluator's trace rows: the active and candidate state
    before the action (``state_before``), the action and the policy-safe tool
    output, which are the fields of the evaluator's own ``history`` entries.
    """

    rows = list(trace) if before_step is None else list(trace)[: int(before_step)]
    events: list[dict[str, Any]] = []
    for row in rows:
        before = row.get("state_before") if isinstance(row.get("state_before"), Mapping) else {}
        observation = row.get("policy_observation") if isinstance(row.get("policy_observation"), Mapping) else {}
        events.append({
            "state_id": before.get("active_state_id", observation.get("active_state_id")),
            "candidate_state_id": before.get("candidate_state_id", observation.get("candidate_state_id")),
            "action": copy.deepcopy(row.get("action")),
            "tool_output": copy.deepcopy(row.get("policy_tool_output") if isinstance(row.get("policy_tool_output"), Mapping) else {}),
        })
    return events


def restored_observation(trace: Sequence[Mapping[str, Any]], position: int) -> dict[str, Any] | None:
    """The policy observation of trace row ``position`` as the policy was shown it, events included."""

    row = trace[position]
    observation = row.get("policy_observation")
    if not isinstance(observation, Mapping):
        return None
    record = observation.get(OBSERVABLE_HISTORY_KEY)
    if not (isinstance(record, Mapping) and record.get("contract") == COMPLETE_HISTORY):
        return copy.deepcopy(dict(observation))
    events = observable_history_from_trace(trace, position)
    expected = record.get("event_count")
    if expected is not None and int(expected) != len(events):
        raise ValueError(f"trace row {position} records {expected} events, the rows before it give {len(events)}")
    return with_observable_history(copy.deepcopy(dict(observation)), events)


__all__ = [
    "BOUNDED_HISTORY",
    "CANDIDATE_EXPERIMENTS_KEY",
    "COMPLETE_HISTORY",
    "HISTORY_RECORD_KEY",
    "OBSERVABLE_HISTORY_KEY",
    "OBSERVATION_MODES",
    "complete_history_record",
    "observable_history_from_trace",
    "observable_history_of",
    "restored_observation",
    "stored_observation",
    "validate_observation_mode",
    "with_observable_history",
]
