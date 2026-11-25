"""Which errors does the model itself hold plausible where an episode stops?  The exact top of its next-call distribution.

    python -m research.minimal_env.candidate_sets build --scoring-inputs scoring_inputs_dev.jsonl \\
        --episodes episodes_dev.jsonl --output candidate_inputs_dev.jsonl
    python -m research.minimal_env.candidate_sets score --adapter ADAPTER --inputs candidate_inputs_dev.jsonl \\
        --output-dir OUT [--shard S/N] [--top 5] [--max-passes 96]
    python -m research.minimal_env.candidate_sets read --scores OUT [--reference REFERENCE.json] \\
        [--probe-report probe_report.json] [--output summary.json]

The question (asked 2026-11-25): where an episode stops with an error left, could the agent report a *set* of
plausible errors with the model's own probabilities, and would the true error be in it?  This reads the answer off
the model without changing it: at the state an unrepaired episode's last action was chosen on, the probability the
model gives every correction it could write there, and whether the objects still wrong are among the one, three or
five most probable.

**The candidates** come from the observation alone (``candidates``): one ``correct_parameters`` call per line the
prompt lists, one ``correct_measurements`` call per meter it lists, both statuses of every breaker it names, and
beside the corrections the two ``finish`` calls, ``undo`` and the two ``inspect`` views.  Nothing of the truth
decides what is a candidate.  A breaker whose table the prompt does not show cannot be named and is no candidate:
such an error left counts as not covered, and is counted apart.

**The probabilities are exact.**  A call's probability is the product over its tokens, the chance that one sample
writes exactly that call (as in ``score_actions``).  A state lists 150 to 300 candidates and a forward pass of the
prompt is seconds, so the calls are not scored one by one.  They are laid out as a tree of tokens; one forward pass
over a call gives the model's next-token distribution at every token of it, which is the exact probability of
every branch that leaves its path.  A branch's probability bounds every call under it, so the tree is searched
best first (``CallSearch``): the most probable call is found, then the next, and the search stops when the wanted
number of distinct objects is found and nothing unscored can exceed them.  Typically a few dozen passes.

**The truth** (the errors left at the state, from ``stopping_states``) rides along in an input beside the
observation and never reaches a prompt.  It does two things: the calls on the objects still wrong are always scored
(so their probability is exact wherever they stand), and the reader counts coverage.  It changes no probability.

``read`` reports, over the stopping states with an error left: how often an object still wrong is among the 1, 3
and 5 most probable correction objects, and how often all of them are; the same over the states where every object
still wrong could be named; the kinds of error against the three correction tools; the probability mass on
corrections and the share of it on the objects still wrong; and, with the evidence probe's report, the model's set
against the probe's reading of the same evidence.
"""
from __future__ import annotations

import argparse
import heapq
import itertools
import json
import math
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from psse_env.minimal.actions import (  # noqa: E402
    BREAKER_STATUSES,
    CORRECT_MEASUREMENTS,
    CORRECT_PARAMETERS,
    CORRECT_TOPOLOGY,
    CORRECTION_TOOLS,
    FINISH,
    INSPECT,
    LINE_STATUSES,
    OUTCOME_RESOLVED,
    OUTCOME_UNRESOLVED,
    UNDO,
    VIEW_BREAKERS,
    VIEW_ESTIMATE,
)

INPUT_CONTRACT = "minimal_candidate_inputs_v1"
SCORES_CONTRACT = "minimal_candidate_scores_v1"
SUMMARY_CONTRACT = "minimal_candidate_sets_v1"
#: The set sizes coverage is read at; the search finds the largest.
SET_SIZES = (1, 3, 5)
DEFAULT_TOP = max(SET_SIZES)
#: Forward passes one state may take before the search gives up and says so.
DEFAULT_MAX_PASSES = 96
CORRECTION = "correction"
#: A log-probability may come back a hair above zero from a half-precision softmax; more than this is an error.
_TOLERANCE = 1e-3


# ------------------------------------------------------------------ the candidates


def signature(action: Mapping[str, Any]) -> str:
    return json.dumps({"tool": action["tool"], "arguments": action.get("arguments") or {}}, sort_keys=True)


def call_object(action: Mapping[str, Any]) -> list[str] | None:
    """The object a correction call names as ``[kind, id]``; ``None`` for any other call.

    A ``correct_measurements`` call that names several meters is one object of kind ``meters``.
    """
    tool = action.get("tool")
    arguments = action.get("arguments") or {}
    if tool == CORRECT_PARAMETERS:
        return ["line", str(int(arguments["line"]))]
    if tool == CORRECT_TOPOLOGY:
        if arguments.get("breaker") is not None:
            return ["breaker", str(arguments["breaker"])]
        return ["line_status", str(int(arguments["line"]))]
    if tool == CORRECT_MEASUREMENTS:
        meters = sorted(int(index) for index in arguments.get("meters") or [])
        return ["meter", str(meters[0])] if len(meters) == 1 else ["meters", ",".join(str(index) for index in meters)]
    return None


def call_kind(action: Mapping[str, Any]) -> str:
    return CORRECTION if action.get("tool") in CORRECTION_TOOLS else str(action.get("tool"))


def shown_objects(observation: Mapping[str, Any]) -> dict[str, list[Any]]:
    """The lines, meters and breakers an observation names: the tables of the versions it shows, and the breakers
    its own earlier calls named."""
    lines: set[int] = set()
    meters: set[int] = set()
    breakers: set[str] = set()
    for entry in observation.get("versions") or []:
        estimate = entry.get("estimate") or {}
        lines |= {int(row["line"]) for row in estimate.get("lines") or []}
        meters |= {int(row["index"]) for row in estimate.get("meters") or []}
        breakers |= {str(row["cb"]) for row in (entry.get("substation") or {}).get("breakers") or []}
    named = [record.get("arguments") or {} for record in observation.get("modifications") or []]
    named += [(event.get("action") or {}).get("arguments") or {} for event in observation.get("history") or []
              if isinstance(event.get("action"), Mapping)]
    for arguments in named:
        if isinstance(arguments.get("breaker"), str) and arguments["breaker"]:
            breakers.add(arguments["breaker"])
        for name in arguments.get("flip") or []:
            if isinstance(name, str) and name:
                breakers.add(name)
    return {"lines": sorted(lines), "meters": sorted(meters), "breakers": sorted(breakers)}


def candidates(observation: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Every call the read-out asks about at a state, from what its observation shows and nothing else."""
    shown = shown_objects(observation)
    breaker_model = bool(observation.get("breaker_model"))
    current = int(observation["current_version"])
    actions: list[dict[str, Any]] = [
        {"tool": FINISH, "arguments": {"outcome": OUTCOME_RESOLVED}},
        {"tool": FINISH, "arguments": {"outcome": OUTCOME_UNRESOLVED}},
        {"tool": UNDO, "arguments": {}},
    ]
    versions = sorted(int(entry["version"]) for entry in observation.get("versions") or [])
    actions += [{"tool": UNDO, "arguments": {"version": version}} for version in versions if version != current]
    actions.append({"tool": INSPECT, "arguments": {"view": VIEW_ESTIMATE}})
    if breaker_model:
        actions.append({"tool": INSPECT, "arguments": {"view": VIEW_BREAKERS}})
    actions += [{"tool": CORRECT_PARAMETERS, "arguments": {"line": line}} for line in shown["lines"]]
    if breaker_model:
        actions += [{"tool": CORRECT_TOPOLOGY, "arguments": {"breaker": name, "status": status}}
                    for name in shown["breakers"] for status in BREAKER_STATUSES]
    else:
        actions += [{"tool": CORRECT_TOPOLOGY, "arguments": {"line": line, "status": status}}
                    for line in shown["lines"] for status in LINE_STATUSES]
    actions += [{"tool": CORRECT_MEASUREMENTS, "arguments": {"meters": [meter]}} for meter in shown["meters"]]
    return [{"action": action, "kind": call_kind(action), "object": call_object(action)} for action in actions]


# ------------------------------------------------------------------ the search


class _Node:
    __slots__ = ("parent", "children", "logp", "expanded", "leaf", "members")

    def __init__(self, parent: "_Node | None" = None) -> None:
        self.parent = parent
        self.children: dict[int, _Node] = {}
        #: The log-probability of the tokens down to this node; ``None`` until a pass through the parent gave it.
        self.logp: float | None = None
        self.expanded = False
        self.leaf: int | None = None
        self.members: list[int] = []


class CallSearch:
    """The calls of one state as a tree of tokens, scored by forward passes.

    ``sequences[i]`` are call ``i``'s completion tokens; no call may be the beginning of another.
    ``forward(sequence, wanted)`` runs the model on the prompt followed by ``sequence`` and returns, for every
    position ``k`` of it, the log-probabilities of the tokens ``wanted[k]`` there (given the prompt and the first
    ``k`` tokens).  One pass over a call gives that call's exact log-probability and the exact log-probability
    of every branch that leaves its path.
    """

    def __init__(self, sequences: Sequence[Sequence[int]], forward: Callable[[Sequence[int], Sequence[Sequence[int]]], Sequence[Sequence[float]]]) -> None:
        self.sequences = [[int(token) for token in sequence] for sequence in sequences]
        self.forward = forward
        self.root = _Node()
        self.root.logp = 0.0
        self.leaves: list[_Node] = []
        self.passes = 0
        self._ticket = itertools.count()
        #: Branches whose probability is known, most probable first; the search starts at the root.
        self._heap: list[tuple[float, int, _Node]] = [(0.0, next(self._ticket), self.root)]
        for index, sequence in enumerate(self.sequences):
            if not sequence:
                raise ValueError(f"call {index} has no tokens")
            node = self.root
            node.members.append(index)
            for token in sequence:
                if node.leaf is not None:
                    raise ValueError(f"call {node.leaf} is the beginning of call {index}")
                child = node.children.get(token)
                if child is None:
                    child = node.children[token] = _Node(node)
                node = child
                node.members.append(index)
            if node.leaf is not None:
                raise ValueError(f"calls {node.leaf} and {index} have the same tokens")
            if node.children:
                raise ValueError(f"call {index} is the beginning of call {node.members[0]}")
            node.leaf = index
            self.leaves.append(node)

    def score(self, index: int) -> float:
        """One forward pass over call ``index``; returns its log-probability."""
        path = [self.root]
        for token in self.sequences[index]:
            path.append(path[-1].children[token])
        wanted = [list(node.children) for node in path[:-1]]
        rows = self.forward(self.sequences[index], wanted)
        self.passes += 1
        if len(rows) != len(wanted):
            raise RuntimeError(f"{len(rows)} positions returned for a call of {len(wanted)} tokens")
        for node, tokens, values in zip(path[:-1], wanted, rows):
            if len(values) != len(tokens):
                raise RuntimeError(f"{len(values)} log-probabilities returned for {len(tokens)} tokens")
            if node.expanded:
                # The first pass through a node set its branches; a later pass reads the same numbers again.
                continue
            assert node.logp is not None
            for token, value in zip(tokens, values):
                value = float(value)
                if not math.isfinite(value) or value > _TOLERANCE:
                    raise RuntimeError(f"not a log-probability: {value!r}")
                child = node.children[token]
                child.logp = node.logp + min(value, 0.0)
                heapq.heappush(self._heap, (-child.logp, next(self._ticket), child))
            node.expanded = True
        found = self.leaves[index].logp
        assert found is not None
        return found

    def logprob(self, index: int) -> float | None:
        """Call ``index``'s exact log-probability, or ``None`` while no pass gave it."""
        return self.leaves[index].logp

    def bound(self, index: int) -> float:
        """An upper bound on call ``index``'s log-probability: that of the deepest part of it a pass has reached."""
        node = self.leaves[index]
        while node.logp is None:
            assert node.parent is not None
            node = node.parent
        return node.logp

    def top(self, *, counted: Iterable[int], group: Callable[[int], Any] | None = None, top: int = DEFAULT_TOP,
            max_passes: int = DEFAULT_MAX_PASSES) -> dict[str, Any]:
        """Score calls best first until ``top`` distinct groups of the ``counted`` calls are found.

        Returns ``order`` (the counted calls in the order found, most probable first), ``groups`` (how many
        distinct groups that is) and ``complete`` (``False`` when ``max_passes`` ended the search first).  The
        first ``groups`` groups of ``order`` are the most probable ones exactly: every call not scored lies under
        a branch that is no more probable than the last call found.
        """
        counted_set = {int(index) for index in counted}
        order: list[int] = []
        found: set[Any] = set()
        complete = True
        while self._heap and len(found) < int(top):
            _negative, _ticket, node = heapq.heappop(self._heap)
            if node.leaf is not None:
                if node.leaf in counted_set:
                    order.append(node.leaf)
                    found.add(group(node.leaf) if group is not None else node.leaf)
                continue
            if node.expanded:
                continue
            pending = [index for index in node.members if index in counted_set and self.leaves[index].logp is None]
            if not pending:
                continue
            if self.passes >= int(max_passes):
                # Put it back: what is left unscored is bounded by this branch.
                heapq.heappush(self._heap, (_negative, _ticket, node))
                complete = False
                break
            self.score(pending[0])
        return {"order": order, "groups": len(found), "complete": complete}

    def label_logprobs(self, labels: Sequence[Any]) -> dict[Any, float | None]:
        """Per label, the log-probability that the call starts as only calls of that label do.

        A label's branches are the highest ones under which every call carries it (for a tool: the tokens of its
        name); their probabilities add.  ``None`` where a pass has not reached one of them yet.
        """
        found: dict[Any, list[float | None]] = defaultdict(list)
        stack = [self.root]
        while stack:
            node = stack.pop()
            kinds = {labels[index] for index in node.members}
            if len(kinds) == 1:
                found[next(iter(kinds))].append(node.logp)
            else:
                stack.extend(node.children.values())
        out: dict[Any, float | None] = {}
        for label, values in found.items():
            if any(value is None for value in values):
                out[label] = None
            else:
                peak = max(values)  # type: ignore[type-var]
                out[label] = peak + math.log(sum(math.exp(value - peak) for value in values))  # type: ignore[operator]
        return out


# ------------------------------------------------------------------ one state


def state_key(item: Mapping[str, Any]) -> str:
    return f"{item['split']}_{item['case']}_{int(item['step'])}"


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name) or "state"


def score_state(item: Mapping[str, Any], *, view: Callable[[Mapping[str, Any]], Mapping[str, Any]],
                render: Callable[[Mapping[str, Any]], str], prompt_ids: Callable[[str], tuple[Sequence[int], int]],
                render_target: Callable[[Mapping[str, Any], Mapping[str, Any]], Sequence[int]],
                positions: Callable[[Sequence[int], Sequence[int], Sequence[Sequence[int]]], Sequence[Sequence[float]]],
                tokens: Callable[[Sequence[int]], Sequence[str]] | None = None, top: int = DEFAULT_TOP,
                max_passes: int = DEFAULT_MAX_PASSES, cache: dict[str, list[int]] | None = None) -> dict[str, Any]:
    """One state: the exact probability of its most probable corrections, of every other call, and of the calls on
    the objects the truth says are still wrong.

    ``cache`` keeps a call's tokens from state to state (they do not depend on the prompt: a call starts at a
    special token).  It is checked at every state: the calls that are always scored are rendered afresh, and a
    cached call whose fresh render differs is an error.
    """
    raw = item["observation"]
    observation = view(raw)
    ids, original = prompt_ids(render(observation))
    ids = [int(value) for value in ids]
    truth = dict(item.get("truth") or {})
    left = {(str(error["kind"]), str(error["object"])) for error in truth.get("errors_left") or []}

    entries: list[dict[str, Any]] = []
    by_signature: dict[str, dict[str, Any]] = {}
    for candidate in candidates(raw):
        entry = {**candidate, "roles": [], "listed": True}
        by_signature[signature(candidate["action"])] = entry
        entries.append(entry)
    for extra in item.get("actions") or []:
        action = extra.get("action")
        if not isinstance(action, Mapping) or not action.get("tool"):
            continue
        entry = by_signature.get(signature(action))
        if entry is None:
            # The model's own call or a step of the truth-built continuation that is no listed candidate (a group
            # of meters, an object the prompt does not show): scored, never counted as a candidate.
            entry = {"action": {"tool": str(action["tool"]), "arguments": dict(action.get("arguments") or {})},
                     "kind": call_kind(action), "object": call_object(action), "roles": [], "listed": False}
            by_signature[signature(action)] = entry
            entries.append(entry)
        if str(extra.get("role")) not in entry["roles"]:
            entry["roles"].append(str(extra.get("role")))

    def always(entry: Mapping[str, Any]) -> bool:
        return (entry["kind"] != CORRECTION or bool(entry["roles"])
                or (entry["object"] is not None and tuple(entry["object"]) in left))

    cache_key = "1:" if raw.get("breaker_model") else "0:"
    usable: list[dict[str, Any]] = []
    unrenderable: list[dict[str, Any]] = []
    rendered_fresh = 0
    render_started = time.perf_counter()
    for entry in entries:
        key = cache_key + signature(entry["action"])
        cached = cache.get(key) if cache is not None else None
        if cached is None or always(entry):
            try:
                fresh = [int(value) for value in render_target(observation, entry["action"])]
            except ValueError as exc:
                unrenderable.append({"action": entry["action"], "roles": entry["roles"], "listed": entry["listed"],
                                     "unrenderable": f"{type(exc).__name__}: {exc}"[:300]})
                continue
            rendered_fresh += 1
            if cached is not None and cached != fresh:
                raise RuntimeError(f"the tokens of {signature(entry['action'])} differ from one state to another")
            if cache is not None:
                cache[key] = fresh
            cached = fresh
        usable.append({**entry, "ids": list(cached)})
    render_seconds = time.perf_counter() - render_started

    search = CallSearch([entry["ids"] for entry in usable], lambda sequence, wanted: positions(ids, sequence, wanted))
    for index, entry in enumerate(usable):
        if always(entry) and search.logprob(index) is None:
            search.score(index)
    required_passes = search.passes
    counted = [index for index, entry in enumerate(usable) if entry["kind"] == CORRECTION and entry["listed"]]
    found = search.top(counted=counted, group=lambda index: tuple(usable[index]["object"]), top=top, max_passes=max_passes)
    tool_logprob = search.label_logprobs([entry["action"]["tool"] for entry in usable])

    calls: list[dict[str, Any]] = []
    unscored = [index for index in counted if search.logprob(index) is None]
    for index, entry in enumerate(usable):
        value = search.logprob(index)
        if value is None:
            continue
        record = {"action": entry["action"], "kind": entry["kind"], "object": entry["object"], "roles": entry["roles"],
                  "listed": entry["listed"], "tokens": len(entry["ids"]), "logprob": round(value, 6)}
        if tokens is not None:
            record["token_text"] = list(tokens(entry["ids"]))
        calls.append(record)
    calls.sort(key=lambda record: -record["logprob"])
    shown = shown_objects(raw)
    return {
        "contract": SCORES_CONTRACT, "case": item["case"], "split": item["split"], "class": item.get("class"),
        "step": int(item["step"]), "point": str(item.get("point") or "stop"),
        "prompt_tokens": len(ids), "prompt_cut": int(original) != len(ids),
        "shown": {key: len(value) for key, value in shown.items()},
        "candidates": {"listed": sum(1 for entry in usable if entry["listed"]), "corrections": len(counted),
                       "unrenderable": len(unrenderable)},
        "passes": search.passes, "passes_always": required_passes, "top": int(top), "complete": bool(found["complete"]),
        "exact_objects": int(found["groups"]),
        "unscored": {"corrections": len(unscored),
                     "highest_bound": round(max(search.bound(index) for index in unscored), 6) if unscored else None},
        "tool_logprob": {str(tool): (None if value is None else round(value, 6)) for tool, value in sorted(tool_logprob.items())},
        "calls": calls, "unrenderable": unrenderable, "truth": truth,
        "rendered_fresh": rendered_fresh, "render_seconds": round(render_seconds, 3),
    }


def score_all(items: Sequence[Mapping[str, Any]], *, directory: Path, shard: tuple[int, int] | None,
              scorer: Callable[[Mapping[str, Any]], dict[str, Any]],
              progress: Callable[[Mapping[str, Any]], None] | None = None) -> int:
    """Score every state of the shard that has no file yet.  Returns how many were scored."""
    pending = [item for index, item in enumerate(items)
               if (shard is None or index % shard[1] == shard[0]) and not (directory / f"{_safe(state_key(item))}.json").is_file()]
    for number, item in enumerate(pending, start=1):
        started = time.perf_counter()
        result = scorer(item)
        result["seconds"] = round(time.perf_counter() - started, 3)
        _write_json(directory / f"{_safe(state_key(item))}.json", result)
        if progress is not None:
            progress({"state": state_key(item), "done": number, "of": len(pending), "seconds": result["seconds"],
                      "passes": result["passes"], "complete": result["complete"], "prompt_tokens": result["prompt_tokens"]})
    return len(pending)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


# ------------------------------------------------------------------ the inputs


def build_inputs(scoring_inputs: Iterable[Mapping[str, Any]], episodes: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The stopping states of ``stopping_states`` with what the truth says is left at each, one input per state.

    ``scoring_inputs`` are the lines of ``scoring_inputs_<split>.jsonl`` (the observation a last action was chosen
    on) and ``episodes`` those of ``episodes_<split>.jsonl`` (the stopping state's errors left, by object).
    """
    stops = {(str(record["split"]), str(record["case"])): record for record in episodes if record.get("stop")}
    out: list[dict[str, Any]] = []
    for item in scoring_inputs:
        if str(item.get("point") or "stop") != "stop":
            raise ValueError(f"case {item.get('case')}: only stopping states carry the errors left by object")
        record = stops.get((str(item["split"]), str(item["case"])))
        if record is None:
            raise ValueError(f"case {item['case']} of {item['split']} has no stopping state in the episode records")
        stop = record["stop"]
        if int(stop["index"]) + 1 != int(item["step"]):
            raise ValueError(f"case {item['case']}: the scoring input is step {item['step']}, the stopping state step {int(stop['index']) + 1}")
        way = record.get("continuation") or {}
        truth = {
            "errors_left": [{"kind": str(error["kind"]), "object": str(error["object"]), "standing": error.get("standing")}
                            for error in stop.get("errors_left") or []],
            "damaged": len(stop.get("damage") or []),
            "declared": record.get("declared"), "outcome": record.get("outcome"),
            "estimator_passes": bool(stop.get("estimator_passes")),
            "continuation_repairs": bool(way.get("repairs")), "undo_needed": bool(way.get("undo_needed")),
        }
        out.append({"contract": INPUT_CONTRACT, "case": str(item["case"]), "split": str(item["split"]), "class": item.get("class"),
                    "step": int(item["step"]), "point": "stop", "observation": item["observation"],
                    "actions": list(item.get("actions") or []), "truth": truth})
    return out


def read_inputs(paths: Sequence[Path]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("contract") != INPUT_CONTRACT:
                raise ValueError(f"{path}: not an input of research.minimal_env.candidate_sets build")
            key = state_key(item)
            if key in seen:
                raise ValueError(f"{path}: state {key} appears twice")
            seen.add(key)
            items.append(item)
    return items


def read_results(directory: Path) -> list[dict[str, Any]]:
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(Path(directory).glob("*.json"))]


# ------------------------------------------------------------------ reading the scores


def _probability(value: float | None) -> float | None:
    return None if value is None else math.exp(min(float(value), 0.0))


def state_reading(result: Mapping[str, Any]) -> dict[str, Any]:
    """What one scored state says: where the objects still wrong stand among the model's correction candidates."""
    calls = list(result["calls"])
    truth = result.get("truth") or {}
    left = [(str(error["kind"]), str(error["object"]), error.get("standing")) for error in truth.get("errors_left") or []]
    highest = (result.get("unscored") or {}).get("highest_bound")

    ranking: list[tuple[str, str]] = []
    best: dict[tuple[str, str], float] = {}
    for call in sorted((call for call in calls if call["kind"] == CORRECTION and call["listed"]), key=lambda call: -call["logprob"]):
        key = (str(call["object"][0]), str(call["object"][1]))
        if key not in best:
            best[key] = float(call["logprob"])
            ranking.append(key)
    place = {key: number for number, key in enumerate(ranking, start=1)}

    def in_set(position: int | None, logprob: float | None, size: int) -> bool | None:
        """Whether a call or object at ``position`` among the scored ones is among the ``size`` most probable of
        all: ``True`` and ``False`` are certain, ``None`` is not decided (the search stopped at its limit first).

        Something unscored can only stand above a scored one, never below it: a position outside the set is
        outside for certain, and one inside is certain when nothing unscored can exceed it.  Every call the search
        found in order passes that test: a branch left unscored is never more probable than a call found before it.
        """
        if position is None or position > size:
            return False
        if highest is None or (logprob is not None and logprob >= float(highest) - 1e-6):
            return True
        return None

    tools = {tool: _probability(value) for tool, value in (result.get("tool_logprob") or {}).items()}
    known = [tools.get(tool) for tool in CORRECTION_TOOLS if tool in tools]
    mass = sum(known) if known and all(value is not None for value in known) else None  # type: ignore[arg-type]

    errors: list[dict[str, Any]] = []
    for kind, item, standing in left:
        key = (kind, item)
        logprob = best.get(key)
        errors.append({"kind": kind, "object": item, "standing": standing, "listed": key in place,
                       "place": place.get(key), "logprob": logprob,
                       "in_set": {str(size): in_set(place.get(key), logprob, size) for size in SET_SIZES},
                       "share_of_corrections": (math.exp(logprob) / mass if logprob is not None and mass else None)})
    covered: dict[str, dict[str, bool | None]] = {}
    for size in SET_SIZES:
        verdicts = [error["in_set"][str(size)] for error in errors]
        covered[str(size)] = {
            "any": None if not errors else True if any(verdict is True for verdict in verdicts)
            else False if all(verdict is False for verdict in verdicts) else None,
            "all": None if not errors else True if all(verdict is True for verdict in verdicts)
            else False if any(verdict is False for verdict in verdicts) else None,
        }

    order = sorted((tool for tool in CORRECTION_TOOLS if tools.get(tool) is not None), key=lambda tool: -tools[tool])  # type: ignore[operator]
    tool_of = {"line": CORRECT_PARAMETERS, "meter": CORRECT_MEASUREMENTS, "meters": CORRECT_MEASUREMENTS,
               "breaker": CORRECT_TOPOLOGY, "line_status": CORRECT_TOPOLOGY}
    wanted_tools = {tool_of[kind] for kind, _item, _standing in left if kind in tool_of}
    kinds: dict[str, dict[str, bool | None]] = {}
    for size in (1, 2):
        kinds[str(size)] = ({"any": bool(wanted_tools & set(order[:size])), "all": wanted_tools <= set(order[:size])}
                            if wanted_tools and len(order) == len(CORRECTION_TOOLS) else {"any": None, "all": None})

    def call_probability(tool: str, arguments: Mapping[str, Any]) -> float | None:
        wanted = signature({"tool": tool, "arguments": arguments})
        return next((math.exp(call["logprob"]) for call in calls if signature(call["action"]) == wanted), None)

    own = next((call for call in calls if "own" in call["roles"]), None)
    wrong = {(kind, item) for kind, item, _standing in left}
    going_on = sorted((call for call in calls if call["listed"] and call["kind"] != FINISH), key=lambda call: -call["logprob"])
    has_repairing = any("repairing" in call["roles"] for call in calls)
    next_step: dict[str, bool | None] = {}
    for size in SET_SIZES:
        verdicts = [in_set(position, call["logprob"], size)
                    for position, call in enumerate(going_on, start=1) if "repairing" in call["roles"]]
        next_step[str(size)] = (None if not has_repairing else True if any(verdict is True for verdict in verdicts)
                                else False if all(verdict is False for verdict in verdicts) else None)
    truth_mass = sum(math.exp(call["logprob"]) for call in calls if call["kind"] == CORRECTION and (
        "repairing" in call["roles"] or (call["object"] is not None and (str(call["object"][0]), str(call["object"][1])) in wrong)))
    top_object = ranking[0] if ranking else None
    return {
        "key": f"{result['split']}:{result['case']}", "group": f"{result['split']}:{result.get('point') or 'stop'}",
        "case": str(result["case"]), "class": result.get("class"), "complete": bool(result.get("complete")),
        "passes": int(result.get("passes") or 0), "errors": errors, "errors_left": len(errors),
        "listed_left": sum(1 for error in errors if error["listed"]),
        "covered": covered, "kinds": kinds, "next_step": next_step, "has_repairing": has_repairing,
        "own": None if own is None else math.exp(own["logprob"]), "own_tool": None if own is None else own["action"]["tool"],
        "own_outcome": None if own is None else (own["action"].get("arguments") or {}).get("outcome"),
        "finish_resolved": call_probability(FINISH, {"outcome": OUTCOME_RESOLVED}),
        "finish_unresolved": call_probability(FINISH, {"outcome": OUTCOME_UNRESOLVED}),
        "inspect_breakers": call_probability(INSPECT, {"view": VIEW_BREAKERS}),
        "tools": tools, "correction_mass": mass,
        "mass_on_objects_still_wrong": truth_mass,
        "share_on_objects_still_wrong": (truth_mass / mass) if mass else None,
        "top_object": None if top_object is None else list(top_object),
        "top_share": (math.exp(best[top_object]) / mass) if top_object is not None and mass else None,
        "top_is_wrong_object": None if top_object is None or not errors else top_object in wrong,
        "declared": truth.get("declared"), "damaged": int(truth.get("damaged") or 0),
        "estimator_passes": bool(truth.get("estimator_passes")),
    }


def _quantiles(values: Sequence[float]) -> dict[str, float] | None:
    values = [float(value) for value in values if value is not None]
    if not values:
        return None
    ordered = sorted(values)

    def at(share: float) -> float:
        return ordered[min(len(ordered) - 1, int(share * len(ordered)))]

    return {"p10": at(0.10), "p25": at(0.25), "median": at(0.5), "p75": at(0.75), "p90": at(0.90), "states": len(ordered)}


def _tally(flags: Iterable[bool | None]) -> dict[str, int]:
    """Counts of a three-valued reading: yes, no, and not decided (the search stopped at its limit first)."""
    counts = Counter("yes" if flag is True else "no" if flag is False else "undecided" for flag in flags)
    return {"yes": int(counts["yes"]), "no": int(counts["no"]), "undecided": int(counts["undecided"])}


def _coverage(readings: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {str(size): {"an object still wrong is in the set": _tally(reading["covered"][str(size)]["any"] for reading in readings),
                        "every object still wrong is in the set": _tally(reading["covered"][str(size)]["all"] for reading in readings)}
            for size in SET_SIZES}


def _cell(readings: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The counts of one group of stopping states."""
    with_error = [reading for reading in readings if reading["errors_left"] > 0]
    nameable = [reading for reading in with_error if reading["listed_left"] == reading["errors_left"]]
    unnameable = [reading for reading in with_error if reading["listed_left"] == 0]
    by_kind: dict[str, Counter] = defaultdict(Counter)
    by_standing: dict[str, Counter] = defaultdict(Counter)
    for reading in with_error:
        for error in reading["errors"]:
            for cell in (by_kind[str(error["kind"])], by_standing[str(error["standing"])]):
                cell["errors"] += 1
                cell["listed"] += int(error["listed"])
                for size in SET_SIZES:
                    cell[f"in the {size} most probable"] += int(error["in_set"][str(size)] is True)
    confident = Counter()
    for reading in with_error:
        share = reading["top_share"]
        if share is None or reading["top_is_wrong_object"] is None:
            continue
        band = "under 0.5" if share < 0.5 else "0.5 to 0.9" if share < 0.9 else "0.9 and over"
        confident[f"{band}: states"] += 1
        confident[f"{band}: the most probable object is one still wrong"] += int(bool(reading["top_is_wrong_object"]))
    return {
        "states": len(readings), "states_with_an_error_left": len(with_error),
        "every_object_still_wrong_can_be_named": len(nameable), "no_object_still_wrong_can_be_named": len(unnameable),
        "search_stopped_at_its_limit": sum(1 for reading in readings if not reading["complete"]),
        "passes": _quantiles([reading["passes"] for reading in readings]),
        "coverage": _coverage(with_error),
        "coverage_where_every_object_can_be_named": _coverage(nameable),
        "kind_of_error_against_the_correction_tools": {
            str(size): {"a tool of an error left is in the set": _tally(reading["kinds"][str(size)]["any"] for reading in with_error),
                        "every such tool is in the set": _tally(reading["kinds"][str(size)]["all"] for reading in with_error)}
            for size in (1, 2)},
        "a_step_of_the_truth_built_continuation_among_the_listed_calls_that_go_on": {
            str(size): _tally(reading["next_step"][str(size)] for reading in readings if reading["has_repairing"])
            for size in SET_SIZES},
        "errors_left_by_kind": {name: dict(cell) for name, cell in sorted(by_kind.items())},
        "errors_left_by_standing": {name: dict(cell) for name, cell in sorted(by_standing.items())},
        "probability": {
            "own action": _quantiles([reading["own"] for reading in readings]),
            "finish resolved": _quantiles([reading["finish_resolved"] for reading in readings]),
            "finish unresolved": _quantiles([reading["finish_unresolved"] for reading in readings]),
            "inspect breakers": _quantiles([reading["inspect_breakers"] for reading in readings]),
            "any correction": _quantiles([reading["correction_mass"] for reading in readings]),
            "a correction of an object still wrong": _quantiles([reading["mass_on_objects_still_wrong"] for reading in with_error]),
            "share of the correction mass on objects still wrong": _quantiles([reading["share_on_objects_still_wrong"] for reading in with_error]),
            "share of the correction mass on the most probable object": _quantiles([reading["top_share"] for reading in with_error]),
        },
        "most_probable_object_by_its_share_of_the_correction_mass": dict(sorted(confident.items())),
    }


def probe_places(report: Mapping[str, Any], reading: str = "candidate_inspect") -> dict[str, int]:
    """Per stopping state (``split:case``), the place the evidence probe gives the best-placed object still wrong."""
    out: dict[str, int] = {}
    for key, entry in (report.get("stopping_states") or {}).items():
        found = entry.get(reading)
        if isinstance(found, Mapping) and found.get("place") is not None:
            out[str(key)] = int(found["place"])
    return out


def summarize(results: Iterable[Mapping[str, Any]], *, reference: Mapping[str, Mapping[str, Any]] | None = None,
              probe: Mapping[str, int] | None = None) -> dict[str, Any]:
    """The read-out over scored states, per split, and within a split by reference group and by ending."""
    from research.minimal_env.repair_breakdown import REFERENCE_GROUPS, reference_group

    readings = [state_reading(result) for result in results]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for reading in readings:
        grouped[reading["group"]].append(reading)
    by_split: dict[str, Any] = {}
    for name, items in sorted(grouped.items()):
        cell = _cell(items)
        endings: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for reading in items:
            ending = ("finish " + str(reading["own_outcome"])) if reading["own_tool"] == FINISH else "another call (out of actions)"
            endings[ending].append(reading)
        cell["by_own_last_action"] = {ending: _cell(group) for ending, group in sorted(endings.items())}
        if reference is not None:
            groups: dict[str, list[dict[str, Any]]] = {group: [] for group in REFERENCE_GROUPS}
            for reading in items:
                known = reference.get(reading["case"])
                if known is not None:
                    groups[reference_group(known)].append(reading)
            cell["by_reference"] = {group: _cell(members) for group, members in groups.items()}
        if probe is not None:
            cross: dict[str, Counter] = {str(size): Counter() for size in SET_SIZES}
            joined = 0
            for reading in items:
                place = probe.get(reading["key"])
                if place is None or reading["errors_left"] == 0:
                    continue
                joined += 1
                for size in SET_SIZES:
                    model = reading["covered"][str(size)]["any"]
                    evidence = place <= size
                    cross[str(size)][("evidence and model" if evidence and model else "evidence only" if evidence and model is False
                                      else "model only" if model and not evidence else "neither" if model is False else "model undecided")] += 1
            cell["against_the_evidence_probe"] = {"states": joined, **{size: dict(sorted(counts.items())) for size, counts in cross.items()}}
        by_split[name] = cell
    return {"contract": SUMMARY_CONTRACT, "set_sizes": list(SET_SIZES), "by_split": by_split}


def _short(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3g}"


def lines_of(summary: Mapping[str, Any]) -> list[str]:
    """The summary's head as text: coverage per split, in the words of the tables."""
    out: list[str] = []
    for name, cell in summary["by_split"].items():
        out.append(f"{name}: {cell['states']} states, {cell['states_with_an_error_left']} with an error left "
                   f"({cell['every_object_still_wrong_can_be_named']} where every object still wrong can be named, "
                   f"{cell['no_object_still_wrong_can_be_named']} where none can); search stopped at its limit in "
                   f"{cell['search_stopped_at_its_limit']}")
        for label, key in (("all states with an error left", "coverage"),
                           ("states where every object can be named", "coverage_where_every_object_can_be_named")):
            for size in SET_SIZES:
                found = cell[key][str(size)]
                one, every = found["an object still wrong is in the set"], found["every object still wrong is in the set"]
                total = one["yes"] + one["no"] + one["undecided"]
                out.append(f"  set of {size}, {label}: an object still wrong in {one['yes']} of {total}"
                           f"{'' if not one['undecided'] else ' (' + str(one['undecided']) + ' undecided)'}, "
                           f"every one in {every['yes']} of {total}")
        median = (cell["probability"]["own action"] or {}).get("median")
        mass = (cell["probability"]["any correction"] or {}).get("median")
        share = (cell["probability"]["share of the correction mass on objects still wrong"] or {}).get("median")
        out.append(f"  median probability: own action {_short(median)}, any correction {_short(mass)}, share of the "
                   f"correction mass on objects still wrong {_short(share)}")
    return out


# ------------------------------------------------------------------ the model's side


def position_logprobs(model: Any, row: Any, *, use_logits_to_keep: bool = True, temperature: float = 1.0) -> Any:
    """The log-probability of every vocabulary token at each completion position of ``row`` (shape ``[n, V]``).

    ``psse_env.rl.update.completion_logprobs`` with its last step left out: the same inputs, the same forward
    pass and the same slice of the logits, then the whole log-softmax row instead of the completion token's entry.
    """
    import torch

    from psse_env.rl.update import _model_device, _row_tensors

    device = _model_device(model)
    inputs = _row_tensors(row, device, torch)
    n = row.completion_tokens
    if n <= 0:
        raise ValueError("row has no completion tokens")
    outputs = None
    if use_logits_to_keep:
        try:
            outputs = model(**inputs, logits_to_keep=n + 1, use_cache=False)
        except TypeError:
            outputs = None
    if outputs is None:
        outputs = model(**inputs, use_cache=False)
    logits = outputs.logits
    if logits.shape[1] > n + 1:
        logits = logits[:, -(n + 1):, :]
    predicting = logits[0, :-1, :].float() / float(temperature)
    return torch.log_softmax(predicting, dim=-1)


def model_scorer(adapter: str, output_dir: Path, *, model_choice: str, max_length: int, max_input_tokens: int,
                 table_versions: int, max_actions: int, temperature: float, top: int,
                 max_passes: int) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
    """The model the RL update loads, with the policy's renderers: a scorer of one state."""
    import torch

    from psse_env.rl.rows import build_step_row
    from research.minimal_env.score_actions import model_parts

    parts = model_parts(adapter, output_dir, model_choice=model_choice, max_length=max_length, max_input_tokens=max_input_tokens,
                        table_versions=table_versions, max_actions=max_actions, temperature=temperature, label="candidate-sets")

    def positions(prompt: Sequence[int], completion: Sequence[int], wanted: Sequence[Sequence[int]]) -> list[list[float]]:
        row = build_step_row(root="score", sample_index=0, step=0, advantage=0.0, prompt_ids=prompt,
                             completion_ids=completion, max_length=int(parts.config.max_length),
                             required_side_inputs=parts.side_inputs)
        if row.prompt_truncated:
            raise RuntimeError("a scored prompt does not fit max_length")
        with torch.no_grad():
            table = position_logprobs(parts.model, row, use_logits_to_keep=True, temperature=float(parts.config.temperature))
            out = []
            for position, tokens_wanted in enumerate(wanted):
                if not tokens_wanted:
                    out.append([])
                    continue
                index = torch.tensor([int(token) for token in tokens_wanted], dtype=torch.long, device=table.device)
                out.append([float(value) for value in table[position].index_select(0, index).cpu().tolist()])
        return out

    cache: dict[str, list[int]] = {}

    def scorer(item: Mapping[str, Any]) -> dict[str, Any]:
        return score_state(item, view=parts.view, render=parts.render, prompt_ids=parts.prompt_ids,
                           render_target=parts.render_target, positions=positions, tokens=parts.tokens, top=top,
                           max_passes=max_passes, cache=cache)

    return scorer


# ------------------------------------------------------------------ the commands


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def _build(args: argparse.Namespace) -> int:
    items = build_inputs(_jsonl(args.scoring_inputs), _jsonl(args.episodes))
    if args.cases is not None:
        from research.minimal_env.stopping_states import case_filter

        kept = case_filter(args.cases) or set()
        items = [item for item in items if item["case"] in kept]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as stream:
        for item in items:
            stream.write(json.dumps(item, sort_keys=True) + "\n")
    listed = [len(candidates(item["observation"])) for item in items]
    left = Counter(len(item["truth"]["errors_left"]) for item in items)
    print(json.dumps({"states": len(items), "candidates": {"min": min(listed, default=0), "max": max(listed, default=0),
                                                           "total": sum(listed)},
                      "states_by_errors_left": {str(count): left[count] for count in sorted(left)},
                      "bytes": args.output.stat().st_size}, sort_keys=True))
    return 0


def _score(args: argparse.Namespace) -> int:
    from research.minimal_env.score_actions import parse_shard

    shard = parse_shard(args.shard)
    items = read_inputs(args.inputs)
    directory = args.output_dir / "states"
    started = time.time()
    pending = [item for index, item in enumerate(items) if (shard is None or index % shard[1] == shard[0])
               and not (directory / f"{_safe(state_key(item))}.json").is_file()]
    print(f"[candidate-sets] {len(items)} states in the inputs, {len(pending)} left for this call", flush=True)
    if pending:
        scorer = model_scorer(str(args.adapter), args.output_dir, model_choice=args.model_choice, max_length=args.max_length,
                              max_input_tokens=args.max_input_tokens, table_versions=args.table_versions,
                              max_actions=args.max_actions, temperature=args.temperature, top=args.top,
                              max_passes=args.max_passes)

        def progress(record: Mapping[str, Any]) -> None:
            print(f"[candidate-sets{'' if shard is None else f' shard {shard[0]}/{shard[1]}'}] {json.dumps(dict(record), sort_keys=True)}",
                  flush=True)

        score_all(items, directory=directory, shard=shard, scorer=scorer, progress=progress)
    if shard is not None:
        print(json.dumps({"shard": list(shard), "seconds": round(time.time() - started, 1)}))
        return 0
    results = read_results(directory)
    missing = len(items) - len(results)
    if missing:
        raise SystemExit(f"{missing} states have no scores after the run")
    summary = {**summarize(results), "adapter": str(args.adapter), "inputs": [str(path) for path in args.inputs],
               "settings": {"temperature": args.temperature, "max_input_tokens": args.max_input_tokens, "precision": "bf16",
                            "top": args.top, "max_passes": args.max_passes}}
    _write_json(args.output_dir / "summary.json", summary)
    print("\n".join(lines_of(summary)))
    return 0


def _read(args: argparse.Namespace) -> int:
    directory = args.scores / "states" if (args.scores / "states").is_dir() else args.scores
    results = read_results(directory)
    if not results:
        raise SystemExit(f"{directory} holds no scored state")
    reference = json.loads(args.reference.read_text(encoding="utf-8")) if args.reference is not None else None
    probe = probe_places(json.loads(args.probe_report.read_text(encoding="utf-8")), args.probe_reading) if args.probe_report is not None else None
    summary = summarize(results, reference=reference, probe=probe)
    if args.output is not None:
        _write_json(args.output, summary)
    print("\n".join(lines_of(summary)))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="join the stopping states with the errors the truth says are left")
    build.add_argument("--scoring-inputs", required=True, type=Path, help="scoring_inputs_<split>.jsonl of stopping_states")
    build.add_argument("--episodes", required=True, type=Path, help="episodes_<split>.jsonl of stopping_states")
    build.add_argument("--cases", type=Path, default=None, help="keep these cases (a reference file or a file of case ids)")
    build.add_argument("--output", required=True, type=Path)
    build.set_defaults(run=_build)
    score = commands.add_parser("score", help="score the candidates of every state with a model")
    score.add_argument("--adapter", required=True, type=Path)
    score.add_argument("--inputs", required=True, type=Path, action="append")
    score.add_argument("--output-dir", required=True, type=Path)
    score.add_argument("--shard", default=None)
    score.add_argument("--top", type=int, default=DEFAULT_TOP, help="distinct correction objects to find exactly")
    score.add_argument("--max-passes", type=int, default=DEFAULT_MAX_PASSES, help="forward passes one state may take")
    score.add_argument("--model-choice", default="12b")
    score.add_argument("--max-length", type=int, default=32768)
    score.add_argument("--max-input-tokens", type=int, default=32768)
    score.add_argument("--max-actions", type=int, default=16)
    score.add_argument("--table-versions", type=int, default=2)
    score.add_argument("--temperature", type=float, default=1.0)
    score.set_defaults(run=_score)
    read = commands.add_parser("read", help="summarize scored states")
    read.add_argument("--scores", required=True, type=Path, help="the output folder of score, or its states folder")
    read.add_argument("--reference", type=Path, default=None)
    read.add_argument("--probe-report", type=Path, default=None, help="probe_report.json of evidence_probe")
    read.add_argument("--probe-reading", default="candidate_inspect", choices=("candidate_inspect", "candidate_shown"))
    read.add_argument("--output", type=Path, default=None)
    read.set_defaults(run=_read)
    args = parser.parse_args(argv)
    return int(args.run(args))


if __name__ == "__main__":
    raise SystemExit(main())
