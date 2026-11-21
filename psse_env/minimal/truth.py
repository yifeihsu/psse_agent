"""Demonstrations built from the truth, executed through the minimal environment.

These are privileged policies: they read the scenario's audit part to know
which breaker, line and meters are wrong.  They are used to make training
data and reference numbers, never to act at run time.  Everything a
demonstration records is what the environment actually returned when the
action ran here; nothing is edited out of an older trace.

**Success paths** (``build_success``).  The truth names the corrections: each
wrong breaker, each faulted line, the bad meters as one group.  For a case
whose errors do not interact the sequence is direct: breakers, then lines,
then meters, one action each, then ``finish``.  Only when that order does not
end in a state both audits call repaired are other orders tried
(``max_orders``), and a correction whose tool fails is tried once more after
the others.  A case no order repairs keeps its first order's run and ends
``finish unresolved``: a limit of the correction tools, reported as a failure.

An ``inspect`` of the breakers is inserted where the demonstrated decision
needs evidence the state does not show by itself: before a breaker
correction on a version whose breakers have not been inspected, and, with
``opening_inspection``, once at the start of every case with a breaker
model, so that looking at the breakers is not a decision the truth made for
the student.

**Look rules** (``looks``).  Under that first rule (``needed``) a
demonstration looks again after a breaker correction only when another
breaker correction follows, so in the first build the look after the first
of two breakers was the truth's decision (300 of 300 two-breaker cases, none
of the 1,000 others), and the supervised model, which cannot see that
decision's reason, mostly did not look (the stopping-state diagnostic of
2026-11-20).  ``every_version`` looks before every action on a version whose
breakers have not been looked at, ``finish`` included: a demonstration then
acts only on versions whose breaker table it has seen, and looking is never
a decision the truth makes.  ``after_breakers`` is the cheaper middle way:
``needed``, and a look after every breaker correction and before ``finish``.
A look yields to the action budget (``skipped_looks`` counts it).

**Refusal paths** (``build_refusal``).  The true corrections up to the first
breaker correction (none when the case has no breaker error), then, on the
version the student has not looked at, the call the supervised model most
often had refused there: a fit of the healthy line with the largest
multiplier (``plausible_errors``).  Kept only when the tool refuses it; the
refused call is context, never a target, and the rows begin with the step
after it, under the look rule.

**Breakers-last paths** (``build_breakers_last``).  For a case with a
breaker error and another one, the same true corrections with the breakers
last: the state where the lines or meters are corrected and a breaker is
still wrong, which the supervised model reaches by correcting what it sees
first, is then in the demonstrations, with the look that shows the breaker.
A line fitted on the wrong topology is fitted wrong, so once the breakers
are right the true objects still unrepaired are corrected once more.  Kept
only when the path repairs the case by both audits.

**Recovery paths** (``build_recovery``).  A demonstration that never errs
cannot teach undoing.  A recovery path replays the first ``after`` true
corrections, applies one correction to a healthy component, and then shows
the way back: ``undo``, the remaining true corrections, ``finish``.  Only
the steps from the ``undo`` on are training targets; the erroneous action is
context, never a target.  ``plausible_errors`` proposes the healthy component
a reader of the evidence would most easily mistake: the healthy meter with
the largest normalized residual, the healthy line with the largest
multiplier, the healthy breaker with the largest score.  That choice
simulates a student's mistake; recovery paths from a trained model's own
failed episodes can replace it once such episodes exist.

**Stopping.**  Every path ends with ``finish`` on the state the truth says
is repaired (or not), so the data holds as many stop decisions as cases, and
a recovery path with ``after`` equal to the number of true corrections is a
wrong correction made after everything was already right.
"""
from __future__ import annotations

import copy
import itertools
import math
import random
from typing import Any, Mapping, Sequence

from psse_env.minimal.actions import (
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
from psse_env.minimal.env import MinimalDiagnosisEnv
from psse_env.minimal.scoring import audit_scenario, physical_comparison, score_env

DEMONSTRATION_CONTRACT = "minimal_truth_demonstration_v1"
FAMILY_ORDER = {CORRECT_TOPOLOGY: 0, CORRECT_PARAMETERS: 1, CORRECT_MEASUREMENTS: 2}
KIND_INSPECT, KIND_CORRECTION, KIND_UNDO, KIND_FINISH = "inspect", "correction", "undo", "finish"
KIND_PREFIX, KIND_ERROR, KIND_REFUSED = "prefix", "injected_error", "refused_call"
#: Step kinds that are training targets.  The replayed prefix repeats the success path, the injected error is wrong
#: and the refused call is the student's mistake the path answers.
TARGET_KINDS = (KIND_INSPECT, KIND_CORRECTION, KIND_UNDO, KIND_FINISH)
REPRESENTABLE_FLIPS = frozenset({"dangling_line_terminal", "reconnect_line_terminal", "bus_split", "merge"})
#: The look rules (module docstring, "Look rules").  ``needed`` is the first build's and the default.
LOOK_NEEDED, LOOK_AFTER_BREAKERS, LOOK_EVERY_VERSION = "needed", "after_breakers", "every_version"
LOOK_RULES = (LOOK_NEEDED, LOOK_AFTER_BREAKERS, LOOK_EVERY_VERSION)
INSPECT_BREAKERS = {"tool": INSPECT, "arguments": {"view": VIEW_BREAKERS}}


def truth_corrections(root: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The corrections the truth names, as actions of the minimal interface, breakers first."""
    scenario = audit_scenario(root)
    from psse_env.dagger.release_audit import _not_applicable_parameter_targets

    isolated = {int(target[1]) + 1 for target in _not_applicable_parameter_targets(scenario) if target[0] == "branch_row0"}
    actions: list[dict[str, Any]] = []
    for item in sorted(scenario.get("true_topology_errors") or [], key=lambda row: str(row.get("cb_name") or row.get("line_index1"))):
        if item.get("cb_name"):
            actions.append({"tool": CORRECT_TOPOLOGY, "arguments": {
                "breaker": str(item["cb_name"]), "status": "closed" if item.get("true_cb_closed") else "open"}})
        else:
            line = int(item["line_index1"]) if item.get("line_index1") is not None else int(item["branch_row0"]) + 1
            actions.append({"tool": CORRECT_TOPOLOGY, "arguments": {
                "line": line, "status": "in" if float(item.get("expected_status", 1)) > 0 else "out"}})
    for item in sorted(scenario.get("true_parameter_errors") or [], key=lambda row: int(row["line_index1"])):
        if int(item["line_index1"]) not in isolated:  # an isolated line carries no current: nothing can estimate it
            actions.append({"tool": CORRECT_PARAMETERS, "arguments": {"line": int(item["line_index1"])}})
    meters = sorted(int(item["index"]) for item in scenario.get("true_measurement_errors") or [] if item.get("index") is not None)
    if meters:
        actions.append({"tool": CORRECT_MEASUREMENTS, "arguments": {"meters": meters}})
    return actions


def orders(corrections: Sequence[Mapping[str, Any]], limit: int) -> list[list[dict[str, Any]]]:
    """Breakers, lines, meters; then every other order, at most ``limit`` in all."""
    first = sorted((copy.deepcopy(dict(action)) for action in corrections), key=lambda action: FAMILY_ORDER[action["tool"]])
    out: list[list[dict[str, Any]]] = []
    for permutation in itertools.permutations(range(len(first))):
        out.append([copy.deepcopy(first[index]) for index in permutation])
        if len(out) >= max(1, int(limit)):
            break
    return out


def _signature(action: Mapping[str, Any]) -> str:
    return f"{action['tool']}:{sorted((action.get('arguments') or {}).items())}"


def _breakers_shown(observation: Mapping[str, Any]) -> bool:
    current = int(observation["current_version"])
    return any(int(entry["version"]) == current and "substation" in entry for entry in observation["versions"])


def _is_breaker_correction(action: Mapping[str, Any] | None) -> bool:
    return bool(action) and action.get("tool") == CORRECT_TOPOLOGY and "breaker" in (action.get("arguments") or {})


def _check_looks(looks: str) -> str:
    if looks not in LOOK_RULES:
        raise ValueError(f"looks must be one of {LOOK_RULES}, got {looks!r}")
    return looks


class _Run:
    """One scripted run: steps with the observation each action was chosen on."""

    def __init__(self, env: MinimalDiagnosisEnv, root: Mapping[str, Any], *, looks: str = LOOK_NEEDED) -> None:
        self.env = env
        self.root = root
        self.looks = _check_looks(looks)
        self.scenario = audit_scenario(root)
        self.observation = env.reset(root)
        self.steps: list[dict[str, Any]] = []
        #: The last accepted correction or ``undo``: the change that made the current version.
        self.last_change: dict[str, Any] | None = None
        self.skipped_looks = 0

    def act(self, action: Mapping[str, Any], kind: str) -> dict[str, Any]:
        before = self.observation
        self.observation, result = self.env.step(copy.deepcopy(dict(action)))
        self.steps.append({"step": self.env.steps, "kind": kind, "action": self.env.history()[-1]["action"],
                           "result": result, "observation": before})
        if result.get("status") == "ok" and action.get("tool") in (CORRECT_TOPOLOGY, CORRECT_PARAMETERS, CORRECT_MEASUREMENTS, UNDO):
            self.last_change = copy.deepcopy(dict(action))
        return result

    def solved(self) -> bool:
        return bool(physical_comparison(self.scenario, self.env.final_state())["solved"])

    def room(self, reserve: int = 1) -> bool:
        """Whether an action can still be taken with ``reserve`` actions left for what must follow."""
        return not self.env.done and self.env.max_actions - self.env.steps > reserve


def _look(run: _Run, before: Mapping[str, Any], kind: str) -> bool:
    """Inspect the breakers before the action ``before`` where the run's look rule asks for it; whether it looked.

    Under ``needed`` the caller's loop holds the budget check, as in the first build; under the other rules a look
    needs room for itself, the action and, before a correction, the ``finish`` after it.
    """
    if not run.env.breakers or run.env.done or _breakers_shown(run.observation):
        return False
    if run.looks == LOOK_EVERY_VERSION:
        due = True
    else:
        due = _is_breaker_correction(before)
        if run.looks == LOOK_AFTER_BREAKERS:
            due = due or before.get("tool") == FINISH or _is_breaker_correction(run.last_change)
    if not due:
        return False
    if run.looks != LOOK_NEEDED and not run.room(reserve=1 if before.get("tool") == FINISH else 2):
        run.skipped_looks += 1
        return False
    run.act(INSPECT_BREAKERS, kind)
    return True


def _apply_corrections(run: _Run, queue: Sequence[Mapping[str, Any]], *, kind: str, inspect_kind: str,
                       retry: bool = True) -> list[dict[str, Any]]:
    """Run the corrections in order, inspecting the breakers first where the look rule asks for it.

    Returns the corrections whose tool failed, with the error.  A failed
    correction is tried once more after the others.
    """
    pending = [copy.deepcopy(dict(action)) for action in queue]
    again: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    tried: set[str] = set()
    while (pending or again) and run.room():
        source = pending if pending else again
        action = source[0]
        if _look(run, action, inspect_kind):
            continue
        source.pop(0)
        result = run.act(action, kind)
        if result.get("status") != "ok":
            failed.append({"action": action, "error_code": result.get("error_code"), "error_detail": result.get("error_detail")})
            if retry and _signature(action) not in tried:
                tried.add(_signature(action))
                again.append(action)
    return failed


def _finish(run: _Run) -> None:
    if run.env.done:
        return
    _look(run, {"tool": FINISH, "arguments": {}}, KIND_INSPECT)
    if not run.env.done:
        run.act({"tool": FINISH, "arguments": {"outcome": OUTCOME_RESOLVED if run.solved() else OUTCOME_UNRESOLVED}}, KIND_FINISH)


def _record(run: _Run, *, demonstration: str, order: Sequence[Mapping[str, Any]], failed: Sequence[Mapping[str, Any]],
            **extra: Any) -> dict[str, Any]:
    score = score_env(run.root, run.env)
    return {"contract": DEMONSTRATION_CONTRACT, "demonstration": demonstration, "scenario_id": run.env.scenario_id,
            "class": score.get("class"), "order": [copy.deepcopy(dict(action)) for action in order],
            "failed_corrections": [copy.deepcopy(dict(item)) for item in failed], "steps": run.steps, "score": score,
            "solved": bool(score["solved"]), "looks": run.looks, "skipped_looks": run.skipped_looks, **extra}


def run_order(env: MinimalDiagnosisEnv, root: Mapping[str, Any], order: Sequence[Mapping[str, Any]], *,
              opening_inspection: bool = True, looks: str = LOOK_NEEDED, demonstration: str = "success") -> dict[str, Any]:
    """One success-path run in the given order."""
    run = _Run(env, root, looks=looks)
    if opening_inspection and env.breakers:
        run.act(INSPECT_BREAKERS, KIND_INSPECT)
    failed = _apply_corrections(run, order, kind=KIND_CORRECTION, inspect_kind=KIND_INSPECT)
    _finish(run)
    return _record(run, demonstration=demonstration, order=order, failed=failed)


def _tidied(order: Sequence[Mapping[str, Any]], built: Mapping[str, Any]) -> list[dict[str, Any]] | None:
    """The order without the targets whose tool never worked, the ones that needed a second attempt last."""
    failed = {_signature(item["action"]) for item in built["failed_corrections"]}
    if not failed:
        return None
    worked = {_signature(step["action"]) for step in built["steps"]
              if step["kind"] == KIND_CORRECTION and step["result"].get("status") == "ok"}
    tidy = ([copy.deepcopy(dict(action)) for action in order if _signature(action) not in failed]
            + [copy.deepcopy(dict(action)) for action in order if _signature(action) in failed & worked])
    return tidy if [_signature(a) for a in tidy] != [_signature(a) for a in order] else None


def build_success(root: Mapping[str, Any], env: MinimalDiagnosisEnv, *, max_orders: int = 6,
                  opening_inspection: bool = True, looks: str = LOOK_NEEDED) -> dict[str, Any]:
    """The case's success path: the first order that ends repaired, else the first order's run."""
    _check_looks(looks)
    corrections = truth_corrections(root)
    kept: dict[str, Any] | None = None
    runs: list[dict[str, Any]] = []

    def rank(built: Mapping[str, Any]) -> int:
        """2: repaired by both audits; 1: by the object-level one alone; 0: not repaired."""
        return int(bool(built["solved"])) + int(bool(built["solved"]) and bool(built["score"]["strict"]["solved"]))

    def note(built: Mapping[str, Any], **extra: Any) -> None:
        runs.append({"order": built["order"], "solved": built["solved"], "strict_solved": bool(built["score"]["strict"]["solved"]),
                     "actions": len(built["steps"]), "failed_corrections": len(built["failed_corrections"]), **extra})

    for order in orders(corrections, max_orders):
        built = run_order(env, root, order, opening_inspection=opening_inspection, looks=looks)
        note(built)
        if kept is None or rank(built) > rank(kept):
            kept = built
        if rank(built) == 2:
            break
    assert kept is not None
    tidied = False
    if kept["solved"]:
        tidy = _tidied(kept["order"], kept)
        if tidy is not None:
            again = run_order(env, root, tidy, opening_inspection=opening_inspection, looks=looks)
            note(again, tidied=True)
            if rank(again) >= rank(kept) and len(again["failed_corrections"]) < len(kept["failed_corrections"]):
                kept, tidied = again, True
    return {**kept, "truth_corrections": corrections, "runs": runs, "tidied": tidied,
            "orders_tried": sum(1 for item in runs if not item.get("tidied")),
            "orders_available": math.factorial(len(corrections))}


def _current(observation: Mapping[str, Any]) -> Mapping[str, Any]:
    current = int(observation["current_version"])
    return next(entry for entry in observation["versions"] if int(entry["version"]) == current)


def plausible_errors(root: Mapping[str, Any], observation: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Corrections on healthy components that the shown evidence makes easiest to mistake, one per family."""
    scenario = audit_scenario(root)
    true_meters = {int(item["index"]) for item in scenario.get("true_measurement_errors") or [] if item.get("index") is not None}
    true_lines = {int(item["line_index1"]) for item in scenario.get("true_parameter_errors") or []}
    true_lines |= {int(item["line_index1"]) for item in scenario.get("true_topology_errors") or [] if item.get("line_index1") is not None}
    true_breakers = {str(item["cb_name"]) for item in scenario.get("true_topology_errors") or [] if item.get("cb_name")}
    entry = _current(observation)
    estimate = entry["estimate"]
    out: list[dict[str, Any]] = []
    meters = [row for row in estimate.get("meters") or [] if row.get("residual") is not None and int(row["index"]) not in true_meters]
    if meters:
        worst = max(meters, key=lambda row: abs(float(row["residual"])))
        out.append({"tool": CORRECT_MEASUREMENTS, "arguments": {"meters": [int(worst["index"])]}})
    lines = [row for row in estimate.get("lines") or [] if int(row["line"]) not in true_lines and not row.get("out")
             and row.get("lr") is not None and row.get("lx") is not None]
    if lines:
        worst = max(lines, key=lambda row: max(abs(float(row["lr"])), abs(float(row["lx"]))))
        out.append({"tool": CORRECT_PARAMETERS, "arguments": {"line": int(worst["line"])}})
    breakers = [row for row in (entry.get("substation") or {}).get("breakers") or []
                if str(row["cb"]) not in true_breakers and row.get("flip_effect") in REPRESENTABLE_FLIPS
                and row.get("score") is not None and int(row.get("line") or 0) not in true_lines]
    if breakers:
        worst = max(breakers, key=lambda row: abs(float(row["score"])))
        out.append({"tool": CORRECT_TOPOLOGY, "arguments": {
            "breaker": str(worst["cb"]), "status": "open" if str(worst["status"]) == "closed" else "closed"}})
    return out


def build_recovery(root: Mapping[str, Any], env: MinimalDiagnosisEnv, order: Sequence[Mapping[str, Any]], *,
                   after: int, family: str | None = None, error: Mapping[str, Any] | None = None,
                   opening_inspection: bool = True, looks: str = LOOK_NEEDED) -> dict[str, Any] | None:
    """A recovery path: ``after`` true corrections, one correction on a healthy component, then the way back.

    ``error`` names the erroneous correction; otherwise ``plausible_errors``
    proposes one, of ``family`` when given (a tool name).  Returns ``None``
    when no error could be staged: the prefix did not run cleanly, nothing
    healthy stood out, or the erroneous tool call failed and so left nothing
    to undo.  Under a look rule other than ``needed`` the error is made on a
    version whose breakers were looked at, and the erroneous version is
    looked at (a target) before the ``undo`` where the rule asks for it.
    """
    run = _Run(env, root, looks=looks)
    if opening_inspection and env.breakers:
        run.act(INSPECT_BREAKERS, KIND_PREFIX)
    prefix, rest = list(order[: int(after)]), list(order[int(after):])
    if _apply_corrections(run, prefix, kind=KIND_PREFIX, inspect_kind=KIND_PREFIX, retry=False) or run.env.done:
        return None
    breaker_error = (error is None and family == CORRECT_TOPOLOGY) or (
        error is not None and error["tool"] == CORRECT_TOPOLOGY and "breaker" in error["arguments"])
    looked_before = (looks == LOOK_EVERY_VERSION
                     or (looks == LOOK_AFTER_BREAKERS and _is_breaker_correction(run.last_change)))
    if (breaker_error or looked_before) and env.breakers and not _breakers_shown(run.observation):
        # The mistake is made by a reader of the breaker table, so the table is on the state it is made on.
        run.act(INSPECT_BREAKERS, KIND_PREFIX)
    if error is None:
        candidates = [item for item in plausible_errors(root, run.observation) if family in (None, item["tool"])]
        if not candidates:
            return None
        error = candidates[0]
    reserve = 2 + len(rest) if looks == LOOK_NEEDED else 2 * len(rest) + 3
    if not run.room(reserve=reserve) or run.act(error, KIND_ERROR).get("status") != "ok":
        return None
    undo = {"tool": UNDO, "arguments": {}}
    _look(run, undo, KIND_INSPECT)
    run.act(undo, KIND_UNDO)
    failed = _apply_corrections(run, rest, kind=KIND_CORRECTION, inspect_kind=KIND_INSPECT)
    _finish(run)
    return _record(run, demonstration="recovery", order=order, failed=failed, after=int(after),
                   injected_error=copy.deepcopy(dict(error)))


def build_refusal(root: Mapping[str, Any], env: MinimalDiagnosisEnv, order: Sequence[Mapping[str, Any]], *,
                  error: Mapping[str, Any] | None = None, opening_inspection: bool = True,
                  looks: str = LOOK_NEEDED) -> dict[str, Any] | None:
    """A refusal path: the true corrections up to the first breaker correction, a call the tool refuses, the rest.

    ``error`` names the refused call; otherwise it is the fit of the healthy
    line ``plausible_errors`` proposes on the version reached (module
    docstring, "Refusal paths").  Returns ``None`` when nothing could be
    staged: the prefix did not run cleanly, no healthy line stood out, the
    budget is too short, or the tool carried the call out (a correction made
    on a healthy component is a recovery path's matter).
    """
    run = _Run(env, root, looks=looks)
    if opening_inspection and env.breakers:
        run.act(INSPECT_BREAKERS, KIND_PREFIX)
    first = next((index for index, action in enumerate(order) if _is_breaker_correction(action)), None)
    after = 0 if first is None else first + 1
    prefix, rest = list(order[:after]), list(order[after:])
    # The prefix looks only where its rule asks before each correction: after its last one the student has not looked.
    if _apply_corrections(run, prefix, kind=KIND_PREFIX, inspect_kind=KIND_PREFIX, retry=False) or run.env.done:
        return None
    if error is None:
        candidates = [item for item in plausible_errors(root, run.observation) if item["tool"] == CORRECT_PARAMETERS]
        if not candidates:
            return None
        error = candidates[0]
    if not run.room(reserve=2 * len(rest) + 2):
        return None
    result = run.act(error, KIND_REFUSED)
    if result.get("status") == "ok":
        return None
    failed = _apply_corrections(run, rest, kind=KIND_CORRECTION, inspect_kind=KIND_INSPECT)
    _finish(run)
    return _record(run, demonstration="refusal", order=order, failed=failed, after=after,
                   refused_call=copy.deepcopy(dict(error)),
                   refusal={"error_code": result.get("error_code"), "error_detail": str(result.get("error_detail") or "")[:200]})


def breakers_last(order: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The same corrections with the breaker corrections moved to the end, each family's order kept."""
    return ([copy.deepcopy(dict(action)) for action in order if not _is_breaker_correction(action)]
            + [copy.deepcopy(dict(action)) for action in order if _is_breaker_correction(action)])


def _refits(run: _Run) -> list[dict[str, Any]]:
    """The corrections of the true lines and meters the current state still leaves unrepaired, to run once more."""
    left = [error for error in physical_comparison(run.scenario, run.env.final_state())["errors"] if not error["repaired"]]
    out = [{"tool": CORRECT_PARAMETERS, "arguments": {"line": int(error["line"])}}
           for error in sorted((error for error in left if error["kind"] == "line"), key=lambda error: int(error["line"]))]
    meters = sorted(int(error["meter"]) for error in left if error["kind"] == "meter")
    if meters:
        out.append({"tool": CORRECT_MEASUREMENTS, "arguments": {"meters": meters}})
    return out


def build_breakers_last(root: Mapping[str, Any], env: MinimalDiagnosisEnv, order: Sequence[Mapping[str, Any]], *,
                        opening_inspection: bool = True, looks: str = LOOK_NEEDED) -> dict[str, Any] | None:
    """A breakers-last path (module docstring), or ``None`` when the case lacks a breaker error or another one, the
    order already has its breakers last, or the path does not repair the case by both audits.

    A line fitted, or a meter repaired, while a breaker is still wrong is corrected on the wrong topology; once the
    breakers are right, the corrections of the true objects still unrepaired run once more (``refits``), as the way on
    from the state the supervised model reaches.
    """
    if not any(_is_breaker_correction(action) for action in order) or all(_is_breaker_correction(action) for action in order):
        return None
    alternative = breakers_last(order)
    if [_signature(action) for action in alternative] == [_signature(action) for action in order]:
        return None
    run = _Run(env, root, looks=looks)
    if opening_inspection and env.breakers:
        run.act(INSPECT_BREAKERS, KIND_INSPECT)
    failed = _apply_corrections(run, alternative, kind=KIND_CORRECTION, inspect_kind=KIND_INSPECT)
    refits = _refits(run) if not run.env.done else []
    if refits:
        failed += _apply_corrections(run, refits, kind=KIND_CORRECTION, inspect_kind=KIND_INSPECT, retry=False)
    _finish(run)
    built = _record(run, demonstration="breakers_last", order=alternative, failed=failed, refits=refits)
    if not (built["solved"] and built["score"]["strict"]["solved"]):
        return None
    return built


def recovery_plan(root: Mapping[str, Any], order: Sequence[Mapping[str, Any]], *, seed: int, count: int = 1) -> list[dict[str, Any]]:
    """Where and in which family to stage the errors of a case's recovery paths, drawn from the case's own seed."""
    scenario_id = str((root.get("execution") or {}).get("scenario_id") or "")
    rng = random.Random(f"{int(seed)}:{scenario_id}")
    families = [CORRECT_MEASUREMENTS, CORRECT_PARAMETERS, CORRECT_TOPOLOGY]
    plan = []
    for _ in range(int(count)):
        plan.append({"after": rng.randint(0, len(order)), "family": rng.choice(families)})
    return plan


__all__ = [
    "DEMONSTRATION_CONTRACT", "KIND_CORRECTION", "KIND_ERROR", "KIND_FINISH", "KIND_INSPECT", "KIND_PREFIX", "KIND_REFUSED",
    "KIND_UNDO", "LOOK_AFTER_BREAKERS", "LOOK_EVERY_VERSION", "LOOK_NEEDED", "LOOK_RULES", "TARGET_KINDS",
    "breakers_last", "build_breakers_last", "build_recovery", "build_refusal", "build_success", "orders",
    "plausible_errors", "recovery_plan", "run_order", "truth_corrections",
]
