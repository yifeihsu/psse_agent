"""Where a model's episodes stop, what is still wrong there, and what a truth-built continuation would do next.

    python -m research.minimal_env.stopping_states --suite SUITE.json --episodes EVAL_DIR --split dev \\
        --cases REFERENCE_OR_SEQUENCES.json --output-dir OUT [--workers N] [--limit N]

Reads the episodes ``research.minimal_env.evaluate`` stored for a suite (``EVAL_DIR/episodes/<case>.json``) and
replays each one's actions in ``MinimalDiagnosisEnv`` under the settings the evaluation ran with.  The environment
is deterministic, so the replay reaches the states the model read; a replay whose results or score differ from the
stored ones is reported and left out of everything below.

For every state an action was chosen on, the truth says what is wrong there (``scoring.physical_comparison``):
which injected errors are not repaired and which healthy objects are outside tolerance.  Each such state also
gives the evidence-probe rows of ``research.minimal_env.evidence_probe``: the features of what the model could
read there, with the substation table computed whether or not the model had inspected it.

For an episode that ends with the case unrepaired, the **stopping state** is the state its last action was chosen
on (a ``finish``, or the last action before the budget ran out).  There the record adds:

* each error left, and whether a correction of its object is in effect (corrected but not within tolerance), was
  undone, or was never made;
* a **truth-built continuation** from that state: the corrections the truth names for the errors left, tried in
  every order (at most ``MAX_ORDERS``) from the current version and, when none of those repairs the case, from
  the other saved versions (an ``undo`` there first).  The first base version with a repairing order gives the
  continuation; the first actions of its repairing orders are the actions ``score_actions`` asks the model about;
* the observation itself, the prompt the model read (``scoring_inputs.jsonl``).

The search is the truth's, unconstrained by what the evidence shows, like the truth-built sequences themselves.
Nothing here reaches a prompt or a training row.

Outputs under ``OUT``: ``episodes.jsonl`` (one record per episode), ``scoring_inputs.jsonl`` (one record per
stopping state of an unrepaired episode), ``probe_<split>.npz`` (the evidence-probe rows) and ``summary.json``.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import multiprocessing
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from psse_env.minimal.actions import (  # noqa: E402
    CORRECT_MEASUREMENTS,
    CORRECT_PARAMETERS,
    CORRECT_TOPOLOGY,
    FINISH,
    INSPECT,
    OUTCOME_RESOLVED,
    OUTCOME_UNRESOLVED,
    UNDO,
    VIEW_BREAKERS,
)
from psse_env.minimal.env import DEFAULT_MAX_ACTIONS, DEFAULT_TABLE_VERSIONS, MinimalDiagnosisEnv  # noqa: E402
from psse_env.minimal.kernel import NumericalKernel  # noqa: E402
from psse_env.minimal.scoring import audit_scenario, physical_comparison, score_env  # noqa: E402
from psse_env.minimal.truth import orders  # noqa: E402
from research.minimal_env import evidence_probe  # noqa: E402

STOPPING_CONTRACT = "minimal_stopping_states_v1"
SCORING_INPUT_CONTRACT = "minimal_scoring_inputs_v1"
#: Orders of the corrections left tried at one base version, as ``truth.build_success`` tries them.
MAX_ORDERS = 6
#: Saved versions other than the current one a continuation may start from (after an ``undo``).
MAX_OTHER_BASES = 3
#: The continuation's own budget: room for an undo and every correction left, whatever the episode had used.
CONTINUATION_ACTIONS = 64
#: What a stopping state's errors left are, by where the model's corrections stand on their objects.
CORRECTED_NOT_REPAIRED = "corrected, not within tolerance"
UNDONE = "corrected, then undone"
NEVER_CORRECTED = "never corrected"
#: The score fields a replay must reproduce.
REPRODUCED_FIELDS = ("solved", "declared", "estimator_passes")


# ------------------------------------------------------------------ objects and corrections


def error_object(error: Mapping[str, Any]) -> tuple[str, str]:
    """An error record of ``physical_comparison`` as ``(kind, object)``: a meter index, a line number or a breaker name."""
    kind = str(error["kind"])
    if kind == "meter":
        return "meter", str(int(error["meter"]))
    if kind == "breaker":
        return "breaker", str(error["breaker"])
    return ("line" if kind == "line" else "line_status"), str(int(error["line"]))


def named_objects(record: Mapping[str, Any]) -> set[tuple[str, str]]:
    """The objects a modification record names in its call, and the lines and meters its changes touched."""
    arguments = record.get("arguments") or {}
    tool = record.get("tool")
    out: set[tuple[str, str]] = set()
    if tool == CORRECT_MEASUREMENTS:
        out |= {("meter", str(int(index))) for index in arguments.get("meters") or []}
    elif tool == CORRECT_PARAMETERS:
        out.add(("line", str(int(arguments["line"]))))
    elif tool == CORRECT_TOPOLOGY:
        if arguments.get("breaker") is not None:
            out.add(("breaker", str(arguments["breaker"])))
        if arguments.get("line") is not None:
            out.add(("line_status", str(int(arguments["line"]))))
    for change in record.get("changes") or []:
        if change.get("object") == "line" and change.get("field") in ("r", "x"):
            out.add(("line", str(int(change["line"]))))
        elif change.get("object") == "meter":
            out.add(("meter", str(int(change["meter"]))))
        elif change.get("object") == "breaker":
            out.add(("breaker", str(change["breaker"])))
    return out


def error_standing(error: Mapping[str, Any], modifications: Sequence[Mapping[str, Any]]) -> str:
    """Where the model's corrections stand on an unrepaired error's object (``modifications`` carry ``in_effect``)."""
    target = error_object(error)
    in_effect = any(target in named_objects(record) for record in modifications if record.get("in_effect"))
    if in_effect:
        return CORRECTED_NOT_REPAIRED
    if any(target in named_objects(record) for record in modifications):
        return UNDONE
    return NEVER_CORRECTED


def corrections_left(scenario: Mapping[str, Any], physical: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The truth's corrections for the errors ``physical`` leaves unrepaired: breakers, lines, then the bad meters
    as one group, in the form of the minimal interface (``truth.truth_corrections`` restricted to what is left)."""
    left = [error for error in physical["errors"] if not error["repaired"]]
    closed = {str(item["cb_name"]): bool(item.get("true_cb_closed"))
              for item in scenario.get("true_topology_errors") or [] if item.get("cb_name")}
    expected = {}
    for item in scenario.get("true_topology_errors") or []:
        if item.get("cb_name"):
            continue
        line = int(item["line_index1"]) if item.get("line_index1") is not None else int(item["branch_row0"]) + 1
        expected[line] = float(item.get("expected_status", 1)) > 0
    actions: list[dict[str, Any]] = []
    for error in sorted((error for error in left if error["kind"] == "breaker"), key=lambda error: str(error["breaker"])):
        name = str(error["breaker"])
        actions.append({"tool": CORRECT_TOPOLOGY, "arguments": {"breaker": name, "status": "closed" if closed.get(name) else "open"}})
    for error in sorted((error for error in left if error["kind"] == "line_status"), key=lambda error: int(error["line"])):
        line = int(error["line"])
        actions.append({"tool": CORRECT_TOPOLOGY, "arguments": {"line": line, "status": "in" if expected.get(line, True) else "out"}})
    for error in sorted((error for error in left if error["kind"] == "line"), key=lambda error: int(error["line"])):
        actions.append({"tool": CORRECT_PARAMETERS, "arguments": {"line": int(error["line"])}})
    meters = sorted(int(error["meter"]) for error in left if error["kind"] == "meter")
    if meters:
        actions.append({"tool": CORRECT_MEASUREMENTS, "arguments": {"meters": meters}})
    return actions


def _signature(action: Mapping[str, Any]) -> str:
    return json.dumps({"tool": action["tool"], "arguments": action.get("arguments") or {}}, sort_keys=True)


# ------------------------------------------------------------------ replay


def stored_actions(record: Mapping[str, Any]) -> list[Any]:
    """The calls of a stored episode as the environment took them: the validated call, or the model's words."""
    out = []
    for step in record.get("steps") or []:
        action = step.get("action")
        if step.get("raw_output") is not None and (not isinstance(action, Mapping) or not action.get("tool")):
            out.append(step["raw_output"])
        else:
            out.append(copy.deepcopy(action))
    return out


def replay_mismatches(record: Mapping[str, Any], results: Sequence[Mapping[str, Any]], score: Mapping[str, Any]) -> list[str]:
    """Where a replay differs from the stored episode: a step's status or error, or the score."""
    found: list[str] = []
    steps = record.get("steps") or []
    if len(results) != len(steps):
        found.append(f"{len(results)} steps replayed of {len(steps)}")
    for stored, result in zip(steps, results):
        before = stored.get("result") or {}
        for key in ("status", "error_code", "to_version", "modification"):
            if before.get(key) != result.get(key):
                found.append(f"step {stored.get('step')}: {key} {before.get(key)!r} -> {result.get(key)!r}")
    stored_score = record.get("score") or {}
    for key in REPRODUCED_FIELDS:
        if stored_score.get(key) != score.get(key):
            found.append(f"score {key} {stored_score.get(key)!r} -> {score.get(key)!r}")
    for key in ("true_errors", "repaired", "damaged"):
        if (stored_score.get("physical") or {}).get(key) != (score.get("physical") or {}).get(key):
            found.append(f"physical {key} {(stored_score.get('physical') or {}).get(key)!r} -> {(score.get('physical') or {}).get(key)!r}")
    return found


def _run(env: MinimalDiagnosisEnv, root: Mapping[str, Any], actions: Sequence[Any]) -> list[dict[str, Any]]:
    env.reset(root)
    results = []
    for action in actions:
        if env.done:
            break
        _observation, result = env.step(copy.deepcopy(action))
        results.append(result)
    return results


# ------------------------------------------------------------------ the continuation


def _bases(env: MinimalDiagnosisEnv, scenario: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The saved versions of the stopping state with what the truth says of each, the current one first."""
    store = env.store
    assert store is not None
    current = store.current
    out = []
    for version in range(len(store)):
        physical = physical_comparison(scenario, store.state(version))
        out.append({"version": version, "current": version == current, "physical": physical,
                    "left": int(physical["true_errors"]) - int(physical["repaired"]), "damaged": int(physical["damaged"])})
    first = [item for item in out if item["current"]]
    others = sorted((item for item in out if not item["current"]), key=lambda item: (item["damaged"], item["left"], -item["version"]))
    return first + others[:MAX_OTHER_BASES]


def _undo_forms(env: MinimalDiagnosisEnv, base: int) -> list[dict[str, Any]]:
    """The calls that return to ``base``: by its number and, where it is the one before the latest correction in
    effect, the bare ``undo`` too."""
    store = env.store
    assert store is not None
    forms = [{"tool": UNDO, "arguments": {"version": int(base)}}]
    if store.version().parent == int(base):
        forms.insert(0, {"tool": UNDO, "arguments": {}})
    return forms


def continuation(root: Mapping[str, Any], prefix: Sequence[Any], *, kernel: NumericalKernel,
                 table_versions: int | None) -> dict[str, Any]:
    """The truth-built continuation from the state ``prefix`` reaches (see the module docstring)."""
    scenario = audit_scenario(root)
    env = MinimalDiagnosisEnv(kernel, max_actions=CONTINUATION_ACTIONS, table_versions=table_versions)
    _run(env, root, prefix)
    breakers_shown = _breakers_shown(env.observation())
    bases = _bases(env, scenario)
    tried: list[dict[str, Any]] = []
    chosen: dict[str, Any] | None = None
    for base in bases:
        actions_left = corrections_left(scenario, base["physical"])
        if not actions_left and base["damaged"]:
            continue
        undo = [] if base["current"] else [_undo_forms(env, base["version"])[-1]]
        repairing: list[list[dict[str, Any]]] = []
        for order in (orders(actions_left, MAX_ORDERS) if actions_left else [[]]):
            sequence = undo + order
            run_env = MinimalDiagnosisEnv(kernel, max_actions=CONTINUATION_ACTIONS, table_versions=table_versions)
            results = _run(run_env, root, list(prefix) + sequence)
            physical = physical_comparison(scenario, run_env.final_state())
            ok = all(result.get("status") == "ok" for result in results[len(prefix):])
            tried.append({"base": base["version"], "sequence": sequence, "solved": bool(physical["solved"]), "tool_errors": not ok})
            if physical["solved"]:
                repairing.append(sequence)
        if repairing:
            chosen = {"base": base["version"], "undo_needed": not base["current"], "orders": repairing}
            break
    out: dict[str, Any] = {"repairs": chosen is not None, "runs": len(tried), "tried": tried, "breakers_shown": breakers_shown,
                           "bases": [{key: item[key] for key in ("version", "current", "left", "damaged")} for item in bases]}
    if chosen is None:
        out["first_actions"] = []
        return out
    first: list[dict[str, Any]] = []
    for sequence in chosen["orders"]:
        if not sequence:
            first.append({"tool": FINISH, "arguments": {"outcome": OUTCOME_RESOLVED}})
            continue
        head = sequence[0]
        if head["tool"] == UNDO:
            first.extend(_undo_forms(env, int(head["arguments"]["version"])))
            continue
        first.append(head)
        if head["tool"] == CORRECT_TOPOLOGY and "breaker" in head["arguments"] and not breakers_shown:
            # A truth-built sequence inspects the breakers before a breaker correction on a version that does not
            # show them; both calls are a step on the way.
            first.append({"tool": INSPECT, "arguments": {"view": VIEW_BREAKERS}})
    unique = list({_signature(action): action for action in first}.values())
    out.update({"base": chosen["base"], "undo_needed": chosen["undo_needed"], "orders": chosen["orders"],
                "first_actions": unique, "shortest": min(len(sequence) for sequence in chosen["orders"])})
    return out


def _breakers_shown(observation: Mapping[str, Any]) -> bool:
    current = int(observation["current_version"])
    return any(int(entry["version"]) == current and "breakers" in (entry.get("substation") or {})
               for entry in observation["versions"])


# ------------------------------------------------------------------ one episode


def analyse(root: Mapping[str, Any], record: Mapping[str, Any], *, split: str, kernel: NumericalKernel,
            probe_kernel: NumericalKernel, max_actions: int = DEFAULT_MAX_ACTIONS,
            table_versions: int | None = DEFAULT_TABLE_VERSIONS, with_continuation: bool = True) -> dict[str, Any]:
    """Replay one stored episode; its record, its stopping state's scoring input and its probe rows."""
    started = time.perf_counter()
    scenario = audit_scenario(root)
    case = str(record["scenario_id"])
    env = MinimalDiagnosisEnv(kernel, max_actions=max_actions, table_versions=table_versions)
    observation = env.reset(root)
    actions = stored_actions(record)
    results: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    states: list[dict[str, Any]] = []
    for index, action in enumerate(actions):
        if env.done:
            break
        state = env.final_state()
        physical = physical_comparison(scenario, state)
        substation = probe_kernel.substation(state)
        rows = evidence_probe.state_rows(observation, substation if substation.get("ok") else None, scenario, physical)
        states.append({"index": index, "physical": physical, "rows": rows})
        observations.append(observation)
        observation, result = env.step(copy.deepcopy(action))
        results.append(result)
    score = score_env(root, env)
    mismatches = replay_mismatches(record, results, score)
    stored_score = record.get("score") or {}
    out: dict[str, Any] = {
        "contract": STOPPING_CONTRACT, "case": case, "split": split, "class": record.get("compound_class") or score.get("class"),
        "solved": bool(stored_score.get("solved")), "declared": stored_score.get("declared"), "outcome": record.get("outcome"),
        "estimator_passes_at_end": bool(stored_score.get("estimator_passes")),
        "actions": [(action.get("tool") if isinstance(action, Mapping) else "(text)") for action in actions],
        "reproduced": not mismatches, "mismatches": mismatches[:8],
    }
    scoring_input = None
    if not mismatches and not out["solved"] and states:
        stop = states[-1]
        physical = stop["physical"]
        modifications = observations[-1]["modifications"]
        left = [error for error in physical["errors"] if not error["repaired"]]
        standings = [error_standing(error, modifications) for error in left]
        current_estimate = next(entry for entry in observations[-1]["versions"]
                                if int(entry["version"]) == int(observations[-1]["current_version"]))["estimate"]
        from psse_env.minimal.scoring import estimator_passes

        out["stop"] = {
            "index": stop["index"], "action": actions[stop["index"]],
            "estimator_passes": estimator_passes(current_estimate),
            "errors_left": [{"kind": error_object(error)[0], "object": error_object(error)[1], "standing": standing}
                            for error, standing in zip(left, standings)],
            "damage": [{key: value for key, value in item.items() if key in ("kind", "meter", "line")} for item in physical["damage"]],
            "breakers_shown": _breakers_shown(observations[-1]),
            "actions_used": int(observations[-1]["budget"]["used"]),
        }
        if with_continuation:
            out["continuation"] = continuation(root, actions[: stop["index"]], kernel=kernel, table_versions=table_versions)
            own = actions[stop["index"]]
            to_score = [{"role": "own", "action": own}]
            for action in out["continuation"]["first_actions"]:
                to_score.append({"role": "repairing", "action": action})
            if isinstance(own, Mapping) and own.get("tool") == FINISH:
                other = OUTCOME_UNRESOLVED if (own.get("arguments") or {}).get("outcome") == OUTCOME_RESOLVED else OUTCOME_RESOLVED
                to_score.append({"role": "other_finish", "action": {"tool": FINISH, "arguments": {"outcome": other}}})
            seen: set[str] = set()
            unique = []
            for item in to_score:
                if not isinstance(item["action"], Mapping):
                    continue
                key = _signature(item["action"])
                if key not in seen:
                    seen.add(key)
                    unique.append(item)
            scoring_input = {"contract": SCORING_INPUT_CONTRACT, "case": case, "split": split, "class": out["class"],
                             "step": stop["index"] + 1, "point": "stop", "observation": observations[-1], "actions": unique}
    probe = evidence_probe.pack_states(
        case, split, [state["rows"] for state in states],
        stop_index=(states[-1]["index"] if states else -1), episode_solved=out["solved"], reproduced=not mismatches)
    out["seconds"] = round(time.perf_counter() - started, 2)
    return {"record": out, "scoring_input": scoring_input, "probe": probe}


# ------------------------------------------------------------------ the first step off the truth's path

#: What the first step off the truth's path did.
OFF_HEALTHY = "corrects a healthy object"
OFF_STATUS = "sets a faulted breaker or line to a wrong status"
OFF_UNDO = "undoes with nothing damaged"
OFF_FINISH = "finishes with an error left"


def off_path(action: Any, physical: Mapping[str, Any], scenario: Mapping[str, Any]) -> str | None:
    """Why ``action`` at a state with the truth's ``physical`` is off the truth's path, or ``None`` when it is on it.

    On the path: an ``inspect``; a correction that names only objects with an unrepaired error (a breaker or a
    line status set to its true value); an ``undo`` where a healthy object is damaged; ``finish`` on a repaired
    state.  The order of the corrections is not judged: a step on an unrepaired object is a step on the path.
    """
    if not isinstance(action, Mapping):
        return OFF_HEALTHY
    tool = action.get("tool")
    arguments = action.get("arguments") or {}
    if tool == INSPECT:
        return None
    if tool == FINISH:
        return None if physical["solved"] else OFF_FINISH
    if tool == UNDO:
        return None if physical["damage"] else OFF_UNDO
    left = {error_object(error) for error in physical["errors"] if not error["repaired"]}
    if tool == CORRECT_MEASUREMENTS:
        return None if all(("meter", str(int(index))) in left for index in arguments.get("meters") or []) else OFF_HEALTHY
    if tool == CORRECT_PARAMETERS:
        return None if ("line", str(int(arguments["line"]))) in left else OFF_HEALTHY
    if tool == CORRECT_TOPOLOGY:
        wanted = {_signature(item) for item in corrections_left(scenario, physical) if item["tool"] == CORRECT_TOPOLOGY}
        if _signature(action) in wanted:
            return None
        key = ("breaker", str(arguments["breaker"])) if arguments.get("breaker") is not None else ("line_status", str(arguments.get("line")))
        return OFF_STATUS if key in left else OFF_HEALTHY
    return OFF_HEALTHY


def divergence(root: Mapping[str, Any], record: Mapping[str, Any], *, split: str, kernel: NumericalKernel,
               max_actions: int = DEFAULT_MAX_ACTIONS, table_versions: int | None = DEFAULT_TABLE_VERSIONS) -> dict[str, Any]:
    """The first state of an unrepaired episode where its action leaves the truth's path (``off_path``), the
    truth-built continuation from there, and the scoring input of that state when it is not the stopping state."""
    started = time.perf_counter()
    scenario = audit_scenario(root)
    env = MinimalDiagnosisEnv(kernel, max_actions=max_actions, table_versions=table_versions)
    observation = env.reset(root)
    actions = stored_actions(record)
    found: dict[str, Any] | None = None
    refused = 0
    for index, action in enumerate(actions):
        if env.done:
            break
        physical = physical_comparison(scenario, env.final_state())
        stored = (record["steps"][index].get("result") or {})
        why = off_path(action, physical, scenario) if stored.get("status") == "ok" else None
        refused += int(stored.get("status") != "ok")
        if why is not None:
            found = {"index": index, "action": action, "kind": why, "observation": observation,
                     "errors_left": int(physical["true_errors"]) - int(physical["repaired"]), "damaged": int(physical["damaged"]),
                     "breakers_shown": _breakers_shown(observation), "actions_used": int(observation["budget"]["used"])}
            break
        observation, _result = env.step(copy.deepcopy(action))
    case = str(record["scenario_id"])
    out: dict[str, Any] = {"contract": STOPPING_CONTRACT, "case": case, "split": split,
                           "class": record.get("compound_class"), "refused_calls_before": refused}
    scoring_input = None
    if found is None:
        out["divergence"] = None
    else:
        stop_index = len(actions) - 1
        out["divergence"] = {key: value for key, value in found.items() if key != "observation"}
        out["divergence"]["is_stop"] = found["index"] == stop_index
        if not out["divergence"]["is_stop"]:
            way = continuation(root, actions[: found["index"]], kernel=kernel, table_versions=table_versions)
            out["continuation"] = way
            to_score = [{"role": "own", "action": found["action"]}] + [{"role": "repairing", "action": action}
                                                                        for action in way["first_actions"]]
            seen: set[str] = set()
            unique = []
            for item in to_score:
                if isinstance(item["action"], Mapping) and _signature(item["action"]) not in seen:
                    seen.add(_signature(item["action"]))
                    unique.append(item)
            scoring_input = {"contract": SCORING_INPUT_CONTRACT, "case": case, "split": split, "class": out["class"],
                             "step": found["index"] + 1, "point": "divergence", "observation": found["observation"],
                             "actions": unique}
    out["seconds"] = round(time.perf_counter() - started, 2)
    return {"record": out, "scoring_input": scoring_input}


# ------------------------------------------------------------------ the pool

_WORKER: dict[str, Any] = {}


def _init_worker(max_actions: int, table_versions: int | None) -> None:
    _WORKER["kernel"] = NumericalKernel()
    _WORKER["probe_kernel"] = NumericalKernel()
    _WORKER["max_actions"] = int(max_actions)
    _WORKER["table_versions"] = table_versions


def _work(task: tuple[Mapping[str, Any], Mapping[str, Any], str, str]) -> dict[str, Any]:
    root, record, split, mode = task
    try:
        if mode == "divergence":
            return {**divergence(root, record, split=split, kernel=_WORKER["kernel"], max_actions=_WORKER["max_actions"],
                                 table_versions=_WORKER["table_versions"]), "probe": None}
        return analyse(root, record, split=split, kernel=_WORKER["kernel"], probe_kernel=_WORKER["probe_kernel"],
                       max_actions=_WORKER["max_actions"], table_versions=_WORKER["table_versions"])
    except Exception as exc:  # one case's failure is reported, never the whole pass's
        import traceback

        return {"record": {"contract": STOPPING_CONTRACT, "case": str(record.get("scenario_id")), "split": split,
                           "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-2000:]},
                "scoring_input": None, "probe": None}


def load_roots(path: Path, wanted: set[str]) -> dict[str, dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        payload = payload.get("scenarios") or payload.get("roots") or []
    return {str(root["execution"]["scenario_id"]): root for root in payload if str(root["execution"]["scenario_id"]) in wanted}


def load_episodes(directory: Path) -> list[dict[str, Any]]:
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted((Path(directory) / "episodes").glob("*.json"))]


def case_filter(path: Path | None) -> set[str] | None:
    """The cases a reference file (``{case: {"truth": bool}}``, kept where the truth repairs) or any file of
    ``truth_sequences.case_ids`` names; ``None`` keeps every case."""
    if path is None:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, Mapping) and payload and all(isinstance(value, Mapping) and "truth" in value for value in payload.values()):
        return {str(case) for case, value in payload.items() if value.get("truth")}
    from research.minimal_env.truth_sequences import case_ids

    return case_ids(path)


def summarize(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Counts over the replayed episodes: how the unrepaired ones stop, and whether the truth goes on from there."""
    totals: Counter = Counter()
    endings: Counter = Counter()
    standings: Counter = Counter()
    kinds: Counter = Counter()
    continuations: Counter = Counter()
    by_class: dict[str, Counter] = {}
    for record in records:
        totals["episodes"] += 1
        if record.get("error"):
            totals["errors"] += 1
            continue
        if not record["reproduced"]:
            totals["not_reproduced"] += 1
            continue
        totals["reproduced"] += 1
        if record["solved"]:
            totals["solved"] += 1
            continue
        stop = record.get("stop")
        if stop is None:
            continue
        totals["unrepaired"] += 1
        left = stop["errors_left"]
        ending = (str(record["declared"] or "out of actions"), "estimate passes" if stop["estimator_passes"] else "estimate fails",
                  f"errors left {len(left)}", "damage" if stop["damage"] else "no damage")
        endings[ending] += 1
        for error in left:
            standings[error["standing"]] += 1
            kinds[f"{error['kind']}: {error['standing']}"] += 1
        primary = (NEVER_CORRECTED if any(error["standing"] == NEVER_CORRECTED for error in left) else
                   UNDONE if any(error["standing"] == UNDONE for error in left) else
                   CORRECTED_NOT_REPAIRED if left else "damage only")
        totals[f"primary: {primary}"] += 1
        cell = by_class.setdefault(str(record.get("class")), Counter())
        cell["unrepaired"] += 1
        cell[f"primary: {primary}"] += 1
        item = record.get("continuation")
        if item is not None:
            if not item["repairs"]:
                continuations["no continuation repairs"] += 1
            elif item["undo_needed"]:
                continuations["repairs after an undo"] += 1
            else:
                continuations["repairs from the stopping state"] += 1
            continuations["runs"] += int(item["runs"])
    return {"totals": dict(sorted(totals.items())), "endings": {", ".join(key): value for key, value in endings.most_common()},
            "errors_left_by_standing": dict(standings.most_common()), "errors_left_by_kind": dict(sorted(kinds.items())),
            "continuations": dict(continuations), "by_class": {name: dict(cell) for name, cell in sorted(by_class.items())}}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--episodes", required=True, type=Path, help="an evaluation folder of research.minimal_env.evaluate")
    parser.add_argument("--split", required=True)
    parser.add_argument("--cases", type=Path, default=None,
                        help="keep these cases: a reference file (cases the truth repairs) or a file of truth_sequences")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-actions", type=int, default=DEFAULT_MAX_ACTIONS)
    parser.add_argument("--table-versions", type=int, default=DEFAULT_TABLE_VERSIONS)
    parser.add_argument("--divergence-from", type=Path, default=None,
                        help="episodes_<split>.jsonl of an earlier call: find the first step off the truth's path of each "
                             "unrepaired episode it reproduced, instead of the whole replay")
    args = parser.parse_args(argv)

    started = time.time()
    wanted = case_filter(args.cases)
    if args.divergence_from is not None:
        replayed = [json.loads(line) for line in Path(args.divergence_from).read_text(encoding="utf-8").splitlines() if line.strip()]
        unrepaired = {str(item["case"]) for item in replayed if item.get("reproduced") and not item.get("solved") and item.get("stop")}
        wanted = unrepaired if wanted is None else wanted & unrepaired
    episodes = [record for record in load_episodes(args.episodes) if wanted is None or str(record["scenario_id"]) in wanted]
    if args.limit is not None:
        episodes = episodes[: int(args.limit)]
    roots = load_roots(args.suite, {str(record["scenario_id"]) for record in episodes})
    absent = [str(record["scenario_id"]) for record in episodes if str(record["scenario_id"]) not in roots]
    if absent:
        raise SystemExit(f"{len(absent)} episodes have no case in {args.suite} (first: {absent[0]})")
    print(f"[stopping-states] {len(episodes)} episodes of {args.split}, suite read in {time.time() - started:.0f} s", flush=True)
    table_versions = int(args.table_versions) or None
    mode = "divergence" if args.divergence_from is not None else "replay"
    tasks = [(roots[str(record["scenario_id"])], record, args.split, mode) for record in episodes]
    del roots
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    probes: list[dict[str, Any]] = []
    # The divergence pass writes beside the replay's files, never over them.
    suffix = "_divergence" if mode == "divergence" else ""
    with (args.output_dir / f"episodes_{args.split}{suffix}.jsonl").open("w", encoding="utf-8", newline="\n") as stream, \
            (args.output_dir / f"scoring_inputs_{args.split}{suffix}.jsonl").open("w", encoding="utf-8", newline="\n") as inputs:
        def take(result: Mapping[str, Any]) -> None:
            records.append(result["record"])
            stream.write(json.dumps(result["record"], sort_keys=True) + "\n")
            if result["scoring_input"] is not None:
                inputs.write(json.dumps(result["scoring_input"], sort_keys=True) + "\n")
            if result["probe"] is not None:
                probes.append(result["probe"])
            if len(records) % 50 == 0 or len(records) == len(tasks):
                print(f"[stopping-states] {len(records)}/{len(tasks)} episodes, {time.time() - started:.0f} s", flush=True)

        if int(args.workers) <= 1:
            _init_worker(args.max_actions, table_versions)
            for task in tasks:
                take(_work(task))
        else:
            context = multiprocessing.get_context("spawn")
            with context.Pool(int(args.workers), initializer=_init_worker, initargs=(args.max_actions, table_versions)) as pool:
                for result in pool.imap_unordered(_work, tasks, chunksize=1):
                    take(result)
    if mode == "divergence":
        summary = {"contract": STOPPING_CONTRACT, "split": args.split, "mode": mode, "seconds": round(time.time() - started, 1),
                   **summarize_divergence(records)}
        (args.output_dir / f"summary_{args.split}{suffix}.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n",
                                                                           encoding="utf-8")
        print(json.dumps(summary, sort_keys=True))
        return 0
    evidence_probe.save_packed(probes, args.output_dir / f"probe_{args.split}.npz")
    summary = {"contract": STOPPING_CONTRACT, "split": args.split, "suite": str(args.suite), "episodes": str(args.episodes),
               "cases": str(args.cases) if args.cases else None, "seconds": round(time.time() - started, 1),
               **summarize(records)}
    (args.output_dir / f"summary_{args.split}.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("totals", "continuations")}, sort_keys=True))
    return 0


def summarize_divergence(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Where the unrepaired episodes first leave the truth's path, and what the step there was."""
    totals: Counter = Counter()
    kinds: Counter = Counter()
    tools: Counter = Counter()
    for record in records:
        totals["episodes"] += 1
        if record.get("error"):
            totals["errors"] += 1
            continue
        found = record.get("divergence")
        if found is None:
            totals["never off the path"] += 1
            continue
        totals["at the stopping state" if found["is_stop"] else "before the stopping state"] += 1
        kinds[found["kind"]] += 1
        action = found["action"]
        tools[(action.get("tool") if isinstance(action, Mapping) else "(text)") + (", breakers shown" if found["breakers_shown"] else "")] += 1
        totals[f"at action {found['index'] + 1}"] += 1
        way = record.get("continuation")
        if way is not None:
            totals["the way on from there needs an undo" if way.get("undo_needed") else "the way on from there needs no undo"
                   if way.get("repairs") else "no way on found from there"] += 1
    return {"totals": dict(sorted(totals.items())), "kinds": dict(kinds.most_common()), "tools": dict(tools.most_common())}


if __name__ == "__main__":
    raise SystemExit(main())
