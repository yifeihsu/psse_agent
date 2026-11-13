"""Truth-built corrective sequences in the open-rules environment, and their training rows.

    python -m research.compound_errors.truth_sequences \\
        --suite output/compound_rl_20261106/train_ieee14/suite.json \\
        --output-dir output/compound_rl_20261106/truth_rows_open_rules_ieee14 \\
        --reference output/compound_rl_20261106/truth_probe_train_ieee14/results.jsonl --workers 20

The user's direction of 2026-11-12: where the rule expert fails, build the
corrective sequence from the known error configuration directly, not through
the controller's rules or the acceptance check, and train from it.  The rules
in question are an environment option since the same day (``open_rules``,
``psse_env.actions.OPEN_RULES``); this module runs a scripted policy in that
environment, with nothing wrapped, and exports each episode as chat rows in
the student's prompt format.

**The sequence.**  The hidden truth names the corrections: every bad meter as
one group, every faulted line, every wrong breaker
(``decision_audit.truth_correction_actions``).  The policy issues the first
solve; then, per correction, the family's context call unless a fresh one is
on the active state, the correction, the solve on the candidate, and the
commit whatever the acceptance check said; then the closing measurement
context and the operator handoff.  A correction whose tool fails is tried once
more after the others.  The correction tools compute their own values and the
truth audit scores the final case, as everywhere else.

**The order.**  Breakers, lines, meters first (a meter corrected before its
line's parameter is replaced from the wrong model); then the other
permutations, until one passes the truth audit (``--max-orders``).  A passing
sequence with a failed correction is run once more without the targets whose
tool never succeeded and with the ones that needed a second attempt moved to
the end, and the tidier run is kept when it passes too.  A root no order
solves keeps its first order's run: the correction tools' limit, reported as
a failure.

**This is a privileged behaviour policy**, never a deployable one: it reads
the truth.  Its rows say what the right action was, not that the evidence in
the prompt shows it.  Every row therefore carries, beside the cloning label,
what a later selection needs (``row["cloning"]["truth_sequence"]``): the kind
of step, the lifted rules the action needed (as the environment reports them,
``TransactionalPSSEEnv.last_open_rules_used``), the action's status, the
acceptance check's verdict on a commit, whether the episode passes the truth
audit and, with ``--reference``, whether the rule expert solves the root.
Nothing of this reaches the messages.

**The evidence.**  A correction row also says whether the state in its own
user turn names the target (``target_evidence``, read from the row, never
from the truth): ``listed`` when the fresh context of the target's family
lists it among its supported corrections (a meter group ``listed one by one``
when the context lists its meters singly); ``named`` when it is not listed
but the fresh evidence or a signature names it (a ranking or finding entry, a
residual or multiplier signature, the solve's hypothesis line, a
flow-at-noise line, a breaker finding; for a breaker ``line named`` when only
its line is); ``absent`` when nothing shown names it.  Named means the
identifier is in the prompt, not that the evidence is enough to choose it; an
absent target can only be guessed.

**The environment** is the student's (``STUDENT_ENVIRONMENT_OPTIONS``) with
every rule lifted (``--open-rules`` names fewer).  Rows are rendered as the
cloning rows were (``cloning_rows.episode_chat_rows``: bounded history, the
run-time system prompt and tool list of the row's own observation), so the
prompt names the lifted rules.  Only training roots belong here: a held-out
case must never become a training row.

Outputs: ``rows.jsonl`` (one chat row per policy step), ``episodes.jsonl``
(the compact episode record with the order, the kinds and the accounting) and
``manifest.json`` (settings and counts).
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import re
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from psse_env.actions import (  # noqa: E402
    ASK_FOR_MORE_EVIDENCE,
    COMMIT_STATE,
    CORRECT_MEASUREMENTS,
    CORRECT_PARAMETERS,
    CORRECT_TOPOLOGY,
    GET_MEASUREMENT_CONTEXT,
    OPEN_RULES,
    RECOVERY_OPTIONS_EXHAUSTED_REQUEST,
    ROLLBACK_STATE,
    RUN_WLS,
    normalize_open_rules,
)
from psse_env.oracle.expert_variants import compound_study_expert_variant  # noqa: E402
from research.compound_errors import cloning_rows, rollout  # noqa: E402
from research.rl import decision_audit as audit  # noqa: E402
from research.rl.decision_audit import parse_run_options  # noqa: E402

SEQUENCE_CONTRACT = "compound_truth_sequence_rows_v1"
BEHAVIOUR = "truth_built_sequence"
#: The student's environment, as the improvement series names it (``research/hpc/rl_improvement_20261110/configs``).
STUDENT_ENVIRONMENT_OPTIONS = {"lifecycle_choice": True, "detailed_evidence": True, "pair_refit_adjacent": True,
                               "flow_at_noise_cue": True, "route_comparison": False}
#: First order tried.  On the held-out cases a meter corrected before its line's parameter is replaced by an estimate
#: from the wrong model, and the line-first order solved 10 of the 14 order-dependent cases.
FAMILY_ORDER = {CORRECT_TOPOLOGY: 0, CORRECT_PARAMETERS: 1, CORRECT_MEASUREMENTS: 2}

KIND_FIRST_SOLVE, KIND_CONTEXT, KIND_CORRECTION, KIND_VERIFY = "first_solve", "context", "correction", "verify"
KIND_COMMIT, KIND_ROLLBACK, KIND_CLOSING_CONTEXT, KIND_HANDOFF = "commit", "rollback", "closing_context", "handoff"
KINDS = (KIND_FIRST_SOLVE, KIND_CONTEXT, KIND_CORRECTION, KIND_VERIFY, KIND_COMMIT, KIND_ROLLBACK, KIND_CLOSING_CONTEXT,
         KIND_HANDOFF)
#: The middle part of a row's example id, so these rows and the cloning rows of the same cases can sit in one file.
ROW_TAG = "truth"
EVIDENCE_LISTED, EVIDENCE_LISTED_SINGLY, EVIDENCE_NAMED = "listed", "listed one by one", "named"
EVIDENCE_PARTLY_NAMED, EVIDENCE_LINE_NAMED, EVIDENCE_ABSENT = "partly named", "line named", "absent"
_STATE_ARGUMENTS = ("state_id", "candidate_state_id")
_LINE_KEYS = frozenset({"line_index1", "line_index", "parameter_line", "topology_line"})
_METER_KEYS = frozenset({"index0", "measurement_index"})
_LINE_SIGNATURE = re.compile(r"\bline=(\d+)")
_METER_SIGNATURE = re.compile(r"\bindex=(\d+)")


def strip(action: Mapping[str, Any]) -> dict[str, Any]:
    """An action without the state it was bound to: what the sequence does, whatever the episode's state ids."""
    return {"tool": action.get("tool"),
            "arguments": {key: value for key, value in (action.get("arguments") or {}).items() if key not in _STATE_ARGUMENTS}}


def signature(action: Mapping[str, Any]) -> str:
    return json.dumps(strip(action), sort_keys=True)


def canonical_order(corrections: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Breakers, lines, meters; inside a family as the truth lists them."""
    return sorted((copy.deepcopy(dict(action)) for action in corrections), key=lambda action: FAMILY_ORDER.get(str(action["tool"]), 9))


def orders(corrections: Sequence[Mapping[str, Any]], limit: int) -> list[list[dict[str, Any]]]:
    """The first order, then every other permutation, at most ``limit`` in all."""
    first = canonical_order(corrections)
    out: list[list[dict[str, Any]]] = []
    for permutation in itertools.permutations(range(len(first))):
        out.append([copy.deepcopy(first[index]) for index in permutation])
        if len(out) >= max(1, int(limit)):
            break
    return out


class TruthSequencePolicy:
    """The truth's corrections in the given order, each verified and kept, then the closing context and the handoff."""

    def __init__(self, corrections: Sequence[Mapping[str, Any]], *, identity_source: Any = None) -> None:
        self.pending = [copy.deepcopy(dict(action)) for action in corrections]
        self.retry: list[dict[str, Any]] = []
        self.kinds: list[str] = []
        #: Corrections whose tool failed, and candidates whose commit was refused twice.
        self.failed: list[dict[str, Any]] = []
        self._identity_source = identity_source
        self._retried: set[str] = set()
        self._started = False
        self._closing = False
        self._context_requests: set[tuple[str, str]] = set()
        self._verify_attempts: Counter = Counter()
        self._commit_attempts: Counter = Counter()
        self._rollback_attempts: Counter = Counter()
        self._last: tuple[str, dict[str, Any]] | None = None
        self._last_correction: dict[str, Any] | None = None

    @property
    def release_policy_identity(self) -> Any:
        return getattr(self._identity_source, "release_policy_identity", None)

    def _issue(self, kind: str, action: dict[str, Any]) -> dict[str, Any]:
        self._last = (kind, action)
        self.kinds.append(kind)
        return action

    def _handoff(self, active: Any) -> dict[str, Any]:
        return self._issue(KIND_HANDOFF, {"tool": ASK_FOR_MORE_EVIDENCE,
                                          "arguments": {"state_id": active, "request": RECOVERY_OPTIONS_EXHAUSTED_REQUEST}})

    def act(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        active = observation.get("active_state_id")
        candidate = observation.get("candidate_state_id")
        open_candidate = bool(observation.get("has_open_candidate") or candidate)
        output = observation.get("last_tool_output") if isinstance(observation.get("last_tool_output"), Mapping) else {}
        if self._last is not None and self._last[0] == KIND_CORRECTION and not open_candidate:
            # The correction opened no candidate: its tool failed.  One more attempt, after the others.
            action = self._last[1]
            self.failed.append({"action": strip(action), "error_code": output.get("error_code"),
                                "error_detail": output.get("error_detail")})
            key = signature(action)
            if key not in self._retried:
                self._retried.add(key)
                self.retry.append(copy.deepcopy(action))
        if not self._started:
            self._started = True
            return self._issue(KIND_FIRST_SOLVE, {"tool": RUN_WLS, "arguments": {"state_id": active}})
        if open_candidate:
            if observation.get("has_unverified_candidate") and self._verify_attempts[candidate] < 2:
                self._verify_attempts[candidate] += 1
                return self._issue(KIND_VERIFY, {"tool": RUN_WLS, "arguments": {"state_id": candidate}})
            if self._commit_attempts[candidate] >= 2:
                # A commit refused twice (a rule this run did not lift): withdraw the candidate and go on.
                if self._rollback_attempts[candidate] >= 2:
                    return self._handoff(active)
                self._rollback_attempts[candidate] += 1
                if self._rollback_attempts[candidate] == 1:
                    self.failed.append({"action": self._last_correction, "error_code": f"commit_refused:{output.get('error_code')}",
                                        "error_detail": output.get("error_detail")})
                return self._issue(KIND_ROLLBACK, {"tool": ROLLBACK_STATE, "arguments": {"candidate_state_id": candidate}})
            self._commit_attempts[candidate] += 1
            return self._issue(KIND_COMMIT, {"tool": COMMIT_STATE, "arguments": {"candidate_state_id": candidate}})
        queue = self.pending if self.pending else self.retry
        if queue:
            tool = str(queue[0]["tool"])
            family = audit.FAMILY_OF_CORRECTION[tool]
            fresh = (bool(observation.get(f"has_fresh_{family}_context"))
                     and str(observation.get(f"{family}_context_state_id")) == str(active))
            if not fresh and (active, family) not in self._context_requests:
                self._context_requests.add((active, family))
                return self._issue(KIND_CONTEXT, {"tool": audit.CONTEXT_TOOL_OF[tool], "arguments": {"state_id": active}})
            action = copy.deepcopy(queue.pop(0))
            action["arguments"] = dict(action.get("arguments") or {}, state_id=active)
            self._last_correction = strip(action)
            return self._issue(KIND_CORRECTION, action)
        if not self._closing:
            self._closing = True
            return self._issue(KIND_CLOSING_CONTEXT, {"tool": GET_MEASUREMENT_CONTEXT, "arguments": {"state_id": active}})
        return self._handoff(active)


# --------------------------------------------------------------------------- one root


def run_order(harness: Any, root: Mapping[str, Any], order: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], TruthSequencePolicy]:
    policy = TruthSequencePolicy(order, identity_source=harness.expert_policy())
    return harness.run(root, policy), policy


def run_summary(order: Sequence[Mapping[str, Any]], episode: Mapping[str, Any], policy: TruthSequencePolicy) -> dict[str, Any]:
    return {"order": [strip(action) for action in order], "success": bool(episode.get("truth_audited_task_success")),
            "reasons": audit.truth_audit_reasons(episode), "steps": len(audit.policy_steps(episode.get("trace") or [])),
            "failed_corrections": len(policy.failed), "terminal_outcome": episode.get("terminal_outcome")}


def tidied_order(order: Sequence[Mapping[str, Any]], episode: Mapping[str, Any], policy: TruthSequencePolicy) -> list[dict[str, Any]] | None:
    """The order without the targets whose tool never succeeded, the ones that needed a second attempt last; None when nothing changes."""
    failed = {signature(entry["action"]) for entry in policy.failed if isinstance(entry.get("action"), Mapping)}
    if not failed:
        return None
    steps = audit.policy_steps(episode.get("trace") or [])
    succeeded = {signature(step.get("action") or {}) for kind, step in zip(policy.kinds, steps)
                 if kind == KIND_CORRECTION and step.get("execution_status") == "success"}
    late = failed & succeeded
    tidy = ([copy.deepcopy(dict(action)) for action in order if signature(action) not in failed]
            + [copy.deepcopy(dict(action)) for action in order if signature(action) in late])
    return tidy if [signature(a) for a in tidy] != [signature(a) for a in order] else None


def build_sequence(root: Mapping[str, Any], harness: Any, *, max_orders: int) -> dict[str, Any]:
    """The root's sequence: the first order that passes the truth audit (tidied when it can be), else the first order's run."""
    corrections = audit.truth_correction_actions(root, "S")
    candidates = orders(corrections, max_orders)
    runs: list[dict[str, Any]] = []
    kept: tuple[dict[str, Any], TruthSequencePolicy, list[dict[str, Any]]] | None = None
    for order in candidates:
        episode, policy = run_order(harness, root, order)
        runs.append(run_summary(order, episode, policy))
        if kept is None or runs[-1]["success"]:
            kept = (episode, policy, order)
        if runs[-1]["success"]:
            break
    assert kept is not None
    episode, policy, order = kept
    tidied = False
    if bool(episode.get("truth_audited_task_success")):
        tidy = tidied_order(order, episode, policy)
        if tidy is not None:
            again, again_policy = run_order(harness, root, tidy)
            runs.append(dict(run_summary(tidy, again, again_policy), tidied=True))
            if runs[-1]["success"] and len(again_policy.failed) < len(policy.failed):
                episode, policy, order, tidied = again, again_policy, tidy, True
    return {"episode": episode, "policy": policy, "order": order, "runs": runs, "tidied": tidied,
            "truth_corrections": [strip(action) for action in canonical_order(corrections)],
            "orders_tried": sum(1 for run in runs if not run.get("tidied")),
            "orders_available": math.factorial(len(corrections))}


def _named(node: Any, lines: set[int], meters: set[int], breakers: set[str], key: str | None = None) -> None:
    """The lines, meters and breakers a piece of the shown state names."""
    if isinstance(node, Mapping):
        for name, value in node.items():
            if isinstance(value, int) and not isinstance(value, bool):
                if name == "line_row0":
                    lines.add(value + 1)
                elif name in _LINE_KEYS:
                    lines.add(value)
                elif name in _METER_KEYS:
                    meters.add(value)
            _named(value, lines, meters, breakers, str(name))
    elif isinstance(node, (list, tuple)):
        for value in node:
            if isinstance(value, int) and not isinstance(value, bool) and key is not None:
                if key.endswith("_lines"):
                    lines.add(value)
                elif key == "suspect_group":
                    meters.add(value)
            _named(value, lines, meters, breakers, key)
    elif isinstance(node, str):
        if node.startswith("CB_"):
            breakers.add(node)
        lines.update(int(match.group(1)) for match in _LINE_SIGNATURE.finditer(node))
        meters.update(int(match.group(1)) for match in _METER_SIGNATURE.finditer(node))


def target_evidence(state: Mapping[str, Any], action: Mapping[str, Any]) -> str | None:
    """Whether the state a correction row shows names the correction's target (module docstring, "The evidence").

    Read from the row's own user turn, so it says what the student can see; never from the truth.
    """
    tool = str(action.get("tool"))
    if tool not in audit.FAMILY_OF_CORRECTION:
        return None
    arguments = action.get("arguments") or {}
    evidence = state.get("fresh_context_evidence") if isinstance(state.get("fresh_context_evidence"), Mapping) else {}
    context = evidence.get(audit.FAMILY_OF_CORRECTION[tool])
    supported = [item.get("arguments") or {} for item in (context.get("supported_corrections") or [] if isinstance(context, Mapping) else [])
                 if isinstance(item, Mapping)]
    lines: set[int] = set()
    meters: set[int] = set()
    breakers: set[str] = set()
    _named(evidence, lines, meters, breakers)
    _named(list(state.get("unresolved_signatures") or []), lines, meters, breakers)
    if tool == CORRECT_PARAMETERS:
        line = arguments.get("line_index")
        if any(item.get("line_index") == line for item in supported):
            return EVIDENCE_LISTED
        return EVIDENCE_NAMED if line in lines else EVIDENCE_ABSENT
    if tool == CORRECT_MEASUREMENTS:
        group = sorted(int(index) for index in arguments.get("suspect_group") or [])
        listed = [sorted(int(index) for index in item.get("suspect_group") or []) for item in supported]
        if group in listed:
            return EVIDENCE_LISTED
        if group and all([index] in listed for index in group):
            return EVIDENCE_LISTED_SINGLY
        if group and all(index in meters for index in group):
            return EVIDENCE_NAMED
        return EVIDENCE_PARTLY_NAMED if any(index in meters for index in group) else EVIDENCE_ABSENT
    breaker = arguments.get("cb_name")
    if breaker is None:
        row = arguments.get("branch_row0")
        if any(item.get("branch_row0") == row for item in supported):
            return EVIDENCE_LISTED
        return EVIDENCE_NAMED if isinstance(row, int) and row + 1 in lines else EVIDENCE_ABSENT
    if any(str(item.get("cb_name")) == str(breaker) for item in supported):
        return EVIDENCE_LISTED
    if str(breaker) in breakers:
        return EVIDENCE_NAMED
    return EVIDENCE_LINE_NAMED if arguments.get("line_index") in lines else EVIDENCE_ABSENT


def shown_state(row: Mapping[str, Any]) -> Mapping[str, Any]:
    """The state a row's prompt shows: the payload of its user turn."""
    for message in row.get("messages") or []:
        if isinstance(message, Mapping) and message.get("role") == "user":
            state = json.loads(str(message.get("content"))).get("state")
            if isinstance(state, Mapping):
                return state
    return {}


def label_rows(rows: Sequence[dict[str, Any]], episode: Mapping[str, Any], kinds: Sequence[str], *, passes: bool,
               rule_expert_solves: bool | None) -> None:
    """What a later selection needs, on every row; outside the messages."""
    steps = audit.policy_steps(episode.get("trace") or [])
    for row in rows:
        index = int(row["cloning"]["step"])
        step = steps[index]
        action = step.get("action") or {}
        tool = str(action.get("tool"))
        kind = kinds[index] if index < len(kinds) else None
        row["cloning"]["truth_sequence"] = {
            "contract": SEQUENCE_CONTRACT,
            "kind": kind,
            "open_rules_used": [str(rule) for rule in step.get("open_rules_used") or []],
            "execution_status": step.get("execution_status"),
            "error_code": step.get("error_code"),
            "verdict": step.get("candidate_disposition_offline") if tool in (COMMIT_STATE, ROLLBACK_STATE) else None,
            "target_in_evidence": target_evidence(shown_state(row), action) if kind == KIND_CORRECTION else None,
            "passes_truth_audit": bool(passes),
            "rule_expert_solves": rule_expert_solves,
        }


def sequence_record(root: Mapping[str, Any], built: Mapping[str, Any], *, rule_expert_solves: bool | None) -> dict[str, Any]:
    """The compact episode record with the order, the kind of every step and the accounting of the lifted rules."""
    episode, policy = built["episode"], built["policy"]
    record = rollout.compact_episode(episode, root)
    steps = audit.policy_steps(episode.get("trace") or [])
    for entry, kind in zip(record["trace"], policy.kinds):
        # The rollout's trace already names the lifted rules a step needed (``rollout.slim_trace``).
        entry["kind"] = kind
    observation = (steps[0].get("policy_observation") or {}) if steps else {}
    record.update(
        sequence_contract=SEQUENCE_CONTRACT,
        behaviour=BEHAVIOUR,
        passes_truth_audit=record["success"],
        rule_expert_solves=rule_expert_solves,
        truth_corrections=built["truth_corrections"],
        order=[strip(action) for action in built["order"]],
        orders_tried=built["orders_tried"],
        orders_available=built["orders_available"],
        runs=built["runs"],
        tidied=built["tidied"],
        kinds=list(policy.kinds),
        failed_corrections=list(policy.failed),
        keep_accounting=dict(episode.get("keep_accounting") or {}),
        open_rule_accounting=dict(episode.get("open_rule_accounting") or {}),
        open_rules=list((observation.get("open_rules") or {}).get("rules") or []),
        false_rollback_count=int(episode.get("false_rollback_count") or 0),
        # The fields the selection of the cloning pass reads (``cloning_select.read_episodes``): nothing is declined here.
        declines=0,
        gate_decisions=[],
    )
    return record


# --------------------------------------------------------------------------- the sweep

_WORKER: dict[str, Any] = {}


def _init_worker(settings: Mapping[str, Any], max_orders: int, export_rows: bool) -> None:
    audit._init_worker(settings)
    _WORKER.update(max_orders=int(max_orders), export_rows=bool(export_rows))


def _run_root(payload: Mapping[str, Any]) -> dict[str, Any]:
    root = payload["root"]
    scenario_id = str(root["execution"]["scenario_id"])
    reference = payload.get("rule_expert_solves")
    started = time.perf_counter()
    try:
        built = build_sequence(root, audit._HARNESS, max_orders=_WORKER["max_orders"])
        record = sequence_record(root, built, rule_expert_solves=reference)
        if _WORKER["export_rows"]:
            rows, refused = cloning_rows.episode_chat_rows(built["episode"], root, behaviour=BEHAVIOUR, tag=ROW_TAG)
            label_rows(rows, built["episode"], built["policy"].kinds, passes=record["success"], rule_expert_solves=reference)
            for row in rows:
                # The episode record keeps what the row's prompt names of its target, for the counts.
                tier = row["cloning"]["truth_sequence"]["target_in_evidence"]
                if tier is not None:
                    record["trace"][int(row["cloning"]["step"])]["target_in_evidence"] = tier
            # The parent writes the rows and drops them from the episode record.
            record["_rows"] = rows
            record["rows_exported"] = len(rows)
            record["rows_refused"] = refused
    except Exception as exc:  # keep the sweep going; the record names the failure
        compound = (root.get("audit") or {}).get("compound") or {}
        record = {"scenario_id": scenario_id, "class": compound.get("class"), "success": False, "passes_truth_audit": False,
                  "error": f"{type(exc).__name__}: {exc}", "steps": 0, "corrections": [], "tools": [], "trace": [], "kinds": [],
                  "rule_expert_solves": reference, "sequence_contract": SEQUENCE_CONTRACT}
    record["seconds"] = round(time.perf_counter() - started, 1)
    return record


def read_reference(path: Path | None) -> dict[str, bool]:
    """Whether the rule expert solves each root: compact rollout records, or ``{"id", "episode"}`` lines holding one."""
    solved: dict[str, bool] = {}
    if path is None:
        return solved
    with Path(path).open(encoding="utf-8") as lines:
        for line in lines:
            if not line.strip():
                continue
            record = json.loads(line)
            episode = record.get("episode") if isinstance(record.get("episode"), Mapping) else record
            scenario_id = record.get("id", episode.get("scenario_id"))
            if scenario_id is not None and episode.get("success") is not None:
                solved[str(scenario_id)] = bool(episode["success"])
    return solved


def row_counts(counts: Counter, record: Mapping[str, Any]) -> None:
    """Step counts of one episode record (every policy step is one row unless the exporter refused it)."""
    passes = bool(record.get("success"))
    for entry in record.get("trace") or []:
        kind = str(entry.get("kind"))
        counts["steps"] += 1
        counts[f"kind:{kind}"] += 1
        counts[f"status:{entry.get('status')}"] += 1
        counts["steps_in_passing_episodes"] += int(passes)
        rules = entry.get("open_rules_used") or []
        if rules:
            counts["steps_needing_a_lifted_rule"] += 1
            counts["steps_needing_a_lifted_rule_in_passing_episodes"] += int(passes)
        for rule in rules:
            counts[f"rule:{rule}"] += 1
            counts[f"rule_in_passing_episodes:{rule}"] += int(passes)
        if entry.get("target_in_evidence") is not None:
            lifted = "needing a lifted rule" if rules else "legal with the rules closed"
            counts[f"correction target {entry['target_in_evidence']}, {lifted}"] += 1
            counts[f"correction target {entry['target_in_evidence']}, {lifted}, in passing episodes"] += int(passes)


def summarize(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    records = list(records)
    counts: Counter = Counter()
    refused: Counter = Counter()
    by_class: dict[str, Counter] = {}
    by_reference: dict[str, Counter] = {}
    for record in records:
        row_counts(counts, record)
        passes = bool(record.get("success"))
        entry = by_class.setdefault(str(record.get("class")), Counter())
        entry["roots"] += 1
        entry["pass"] += int(passes)
        reference = record.get("rule_expert_solves")
        name = "unknown" if reference is None else ("rule expert solves" if reference else "rule expert fails")
        side = by_reference.setdefault(name, Counter())
        side["roots"] += 1
        side["pass"] += int(passes)
        side["steps"] += int(record.get("steps") or 0)
        for reason, count in (record.get("rows_refused") or {}).items():
            refused[str(reason)] += int(count)
    return {
        "roots": len(records),
        "errors": sum(bool(record.get("error")) for record in records),
        "pass": sum(bool(record.get("success")) for record in records),
        "tidied": sum(bool(record.get("tidied")) for record in records),
        "episodes_needing_a_lifted_rule": sum(bool((record.get("open_rule_accounting") or {}).get("counts")) for record in records),
        "kept_commits": sum(int((record.get("keep_accounting") or {}).get("kept") or 0) for record in records),
        "false_commits": sum(int(record.get("false_commit_count") or 0) for record in records),
        "closed_by_handoff": sum(record.get("terminal_outcome") == "operator_escalation" for record in records),
        "orders_tried": dict(sorted(Counter(int(record.get("orders_tried") or 0) for record in records).items())),
        "audit_reasons_of_failures": dict(Counter(reason for record in records if not record.get("success")
                                                  for reason in record.get("audit_reasons") or []).most_common()),
        "by_class": {name: dict(entry) for name, entry in sorted(by_class.items())},
        "by_rule_expert_outcome": {name: dict(entry) for name, entry in sorted(by_reference.items())},
        "steps": {key: counts[key] for key in sorted(counts)},
        "rows_refused": dict(refused.most_common()),
    }


def _git_head() -> dict[str, Any]:
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout
        return {"head": head, "uncommitted_paths": len([line for line in dirty.splitlines() if line.strip()])}
    except Exception as exc:  # the record is a convenience, never a gate
        return {"head": None, "error": f"{type(exc).__name__}: {exc}"}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", required=True, type=Path, help="training roots only")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--reference", type=Path, default=None,
                        help="the rule expert's episodes on the same roots (rollout records), for the label rule_expert_solves")
    parser.add_argument("--open-rules", default="all", help="the rules lifted: all, or rule names joined by +")
    parser.add_argument("--max-orders", type=int, default=6)
    parser.add_argument("--no-rows", action="store_true", help="build the sequences only")
    parser.add_argument("--roots", nargs="*", default=None, help="scenario ids to run")
    parser.add_argument("--per-class", type=int, default=0, help="first N roots of every class")
    parser.add_argument("--system", default=None)
    parser.add_argument("--profile", default="scada_only")
    parser.add_argument("--expert", default=None, help="expert variant whose provider options the environment takes")
    parser.add_argument("--progress-basis", default=None)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--env-option", action="append", default=None, metavar="KEY=VALUE",
                        help="added to the student's environment options")
    parser.add_argument("--expert-option", action="append", default=None, metavar="KEY=VALUE")
    args = parser.parse_args(argv)

    lifted = normalize_open_rules(args.open_rules)
    if not lifted:
        raise SystemExit("--open-rules names no rule; the sequences need the open-rules environment")
    roots = rollout.load_suite(args.suite)
    if args.roots:
        wanted = set(args.roots)
        roots = [root for root in roots if str(root["execution"]["scenario_id"]) in wanted]
    if args.per_class:
        taken: Counter = Counter()
        kept = []
        for root in roots:
            name = ((root.get("audit") or {}).get("compound") or {}).get("class")
            if taken[name] < args.per_class:
                taken[name] += 1
                kept.append(root)
        roots = kept
    system = rollout.suite_system(roots, args.system)
    args.progress_basis = rollout.system_progress_basis(system, args.progress_basis)
    args.expert = args.expert or compound_study_expert_variant(system)
    env_options = dict(STUDENT_ENVIRONMENT_OPTIONS)
    env_options.update(parse_run_options(args.env_option))
    env_options["open_rules"] = "+".join(lifted)
    settings = dict(profile=args.profile, expert_variant=args.expert, triage_model=None, progress_basis=args.progress_basis,
                    max_steps=args.max_steps, seed=args.seed, reward=None, env_options=env_options,
                    expert_options=parse_run_options(args.expert_option))
    reference = read_reference(args.reference)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    export_rows = not args.no_rows
    print(f"[truth sequences] {len(roots)} roots, rules lifted {list(lifted)}, rows {export_rows}, workers {args.workers}", flush=True)
    started = time.perf_counter()
    payloads = [{"root": root, "rule_expert_solves": reference.get(str(root["execution"]["scenario_id"]))} for root in roots]
    del roots
    initargs = (settings, args.max_orders, export_rows)
    records: list[dict[str, Any]] = []
    row_total = 0
    row_stream = (out / "rows.jsonl").open("w", encoding="utf-8", newline="\n") if export_rows else None

    def take(record: dict[str, Any], stream: Any) -> None:
        nonlocal row_total
        rows = record.pop("_rows", None) or []
        if row_stream is not None:
            for row in rows:
                row_stream.write(json.dumps(row, sort_keys=True, default=str) + "\n")
            row_total += len(rows)
        stream.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        stream.flush()
        # The summary reads the slim record; the observations were never in it.
        records.append(record)
        if len(records) % 25 == 0 or len(records) == len(payloads):
            print(json.dumps({"done": len(records), "pass": sum(bool(r.get("success")) for r in records), "rows": row_total,
                              "errors": sum(bool(r.get("error")) for r in records),
                              "elapsed": round(time.perf_counter() - started, 1)}), flush=True)

    with (out / "episodes.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        if args.workers > 1:
            with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker, initargs=initargs) as pool:
                for record in pool.map(_run_root, payloads, chunksize=1):
                    take(record, stream)
        else:
            _init_worker(*initargs)
            for payload in payloads:
                take(_run_root(payload), stream)
    if row_stream is not None:
        row_stream.close()
    manifest = dict(contract=SEQUENCE_CONTRACT, behaviour=BEHAVIOUR, suite=str(args.suite), reference=str(args.reference) if args.reference else None,
                    system=system, profile=args.profile, expert=args.expert, progress_basis=args.progress_basis, max_steps=args.max_steps,
                    seed=args.seed, max_orders=args.max_orders, open_rules=list(lifted), all_rules_lifted=tuple(lifted) == tuple(OPEN_RULES),
                    env_options=settings["env_options"], expert_options=settings["expert_options"], rows=row_total,
                    rows_format={"exporter": "cloning_rows.episode_chat_rows", "protocol": "canonical", "history": "bounded (8 events, 4096 characters)"},
                    code=_git_head(), workers=args.workers, total_seconds=round(time.perf_counter() - started, 1),
                    summary=summarize(records))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in ("rows", "total_seconds")} | {key: manifest["summary"][key] for key in
                     ("roots", "errors", "pass", "tidied", "kept_commits", "false_commits", "closed_by_handoff", "rows_refused")}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
