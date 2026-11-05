"""Detour ledger: what an episode spent beyond its protocol floor, and on what.

The closed-loop evaluator reports cost totals (actions, WLS calls) and protocol
violations (invalid actions, false commits, loops), but it never names a
detour: its loop detector resets on every lifecycle advance, so a
correct-verify-reject-rollback cycle is "progress", and its invalid-action
counter mixes the agent's mistakes with the corrector's own failures.  This
script reads the per-episode records the studies already write and attributes
every action above the protocol floor to a named cause.

Definitions
-----------
protocol floor
    ``2 + 4 x atoms``: one opening WLS solve and one handoff, plus for every
    error atom a context fetch, the correction, its verifying solve and the
    commit.  This is the rule expert's own minimum at one correction per atom
    (``docs/rl_hard_cases_plan_20261013.md`` section 3.1).
slack
    actions minus the floor.
hypothesis
    one correction attempt, executed or refused.
rejected cycle
    an executed correction whose lifecycle step was a rollback: the correction,
    the solves in between and the rollback are wasted.  A correction the
    episode never disposed of (the budget ended with the candidate open) is an
    ``undisposed`` cycle and is wasted too.
refused correction
    a correction whose execution failed before any verification: one wasted
    action.  Refusals are split by error code into ``tool_refused`` (the
    executor could not do it: a corrector that does not converge, a dispatch
    error) and ``policy_invalid`` (the gate refused an action the agent should
    not have proposed: unsupported target, lifecycle violation, missing
    precondition).  The same split applies to every other failed action.
fruitless route
    a context fetch after which no correction of that context's family was
    committed before the next context fetch or the end of the episode.  The
    fetch itself is wasted; the corrections inside the segment are accounted
    for by their own cycles, so nothing is counted twice.
comparison
    ``compare_parameter_candidates`` and ``compare_routes`` are exploration,
    not waste: they are counted, and a comparison whose best line was then
    committed is ``productive``.
wasted actions
    the size of the union of the wasted step sets above.
harm, interface
    an accepted correction on a wrong target is harm (counted, not waste); a
    true target rejected by the verifier or refused by the executor is an
    interface limit.  Both need the ``target_is_true`` flags of the compound
    rollout record; without them the truth columns are left empty.

Inputs
------
``episodes``: one JSON object per line in either of the formats the repository
writes: the compound rollout's compact record (``research/compound_errors/
rollout.py``: ``tools``, ``trace``, ``corrections`` with ``target_is_true``) or
the closed-loop evaluator's ``EpisodeEvaluation`` record (``trace`` steps with
``action``, ``execution_status``, ``error_code``,
``candidate_disposition_offline``).  The DAgger cells' compact
``episodes.jsonl`` (``research/hpc/full_pipeline_20260907/evaluate_suite.py``)
carries no trace; such rows get cost columns only.

``summary``: a rollout ``summary.json`` (``by_class`` with the verdict ledger
``tool:true|wrong:verdict``), from which a class-level approximation is built
with the assumption that a rejected candidate cost three actions (correction,
verifying solve, rollback) and a refused one cost one.

Usage
-----
    python research/compound_errors/detour_ledger.py episodes \\
        --episodes output/compound_errors_ieee118_20261028/expert_baseline_compare/episodes.jsonl \\
        --manifest output/compound_errors_ieee118_20261028/manifest.json \\
        --output-dir output/compound_errors_ieee118_20261028/detour_ledger

    python research/compound_errors/detour_ledger.py summary \\
        --summary name=path/to/summary.json [--summary name=...] \\
        --manifest docs/compound_errors_ieee118_20261028/ieee118_suite_manifest.json \\
        --output-dir output/detour_ledger_summaries

The script imports nothing from ``psse_env``; the tool names below mirror
``psse_env/actions.py`` so it runs anywhere the episode files are.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

LEDGER_CONTRACT = "detour_ledger_v1"

RUN_WLS = "run_wls"
VERIFY_CANDIDATE = "verify_candidate"
COMMIT_STATE = "commit_state"
ROLLBACK_STATE = "rollback_state"
FINALIZE_DIAGNOSIS = "finalize_diagnosis"
ASK_FOR_MORE_EVIDENCE = "ask_for_more_evidence"
REVERT_STATE = "revert_state"
INVALID_ACTION = "__invalid_action__"
COMPARE_PARAMETER_CANDIDATES = "compare_parameter_candidates"
COMPARE_ROUTES = "compare_routes"

CONTEXT_FAMILY = {
    "get_measurement_context": "measurement",
    "get_parameter_context": "parameter",
    "get_topology_context": "topology",
    "get_harmonic_context": "harmonic",
    "get_three_phase_context": "three_phase",
}
CORRECTION_FAMILY = {
    "correct_measurements": "measurement",
    "correct_parameters": "parameter",
    "correct_topology": "topology",
}
CORRECTION_TOOLS = frozenset(CORRECTION_FAMILY)
CONTEXT_TOOLS = frozenset(CONTEXT_FAMILY)
COMPARISON_TOOLS = frozenset({COMPARE_PARAMETER_CANDIDATES, COMPARE_ROUTES})
LIFECYCLE_TOOLS = frozenset({COMMIT_STATE, ROLLBACK_STATE})
ACCEPTED_VERDICTS = frozenset({"ACCEPT_FINAL", "ACCEPT_PARTIAL"})
BUDGET_EXHAUSTED_REQUEST = "operator_escalation:recovery_budget_exhausted"

#: Error codes of a failed action that the executor, not the agent, is
#: responsible for.  Everything else is the gate refusing a proposal.
TOOL_REFUSED_CODES = frozenset({
    "dispatch_error",
    "executor_failure",
    "correction_execution_failure",
    "verification_solver_failure",
    "solver_failure",
    "solver_error",
    "wls_failure",
    "evidence_failure",
    "comparison_unavailable",
    "hif_estimation_failure",
    "hif_multiscan_failure",
    "injected_transient_tool_failure",
    "topology_correction_telemetry_incomplete",
    "topology_correction_input_error",
    "topology_context_input_error",
    "topology_reference_unavailable",
})
TOOL_REFUSED_SUFFIXES = ("_correction_failure",)


# --------------------------------------------------------------------------- normalization


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def refusal_kind(error_code: Any, tool: str | None = None, extra_tool_codes: Iterable[str] = ()) -> str:
    """``tool_refused`` or ``policy_invalid`` for a failed action."""
    if tool == INVALID_ACTION:
        return "policy_invalid"
    code = str(error_code or "").strip()
    if code in TOOL_REFUSED_CODES or code in set(extra_tool_codes) or code.endswith(TOOL_REFUSED_SUFFIXES):
        return "tool_refused"
    return "policy_invalid"


def normalize_steps(episode: Mapping[str, Any]) -> list[dict[str, Any]] | None:
    """The policy steps of either record format as ``{tool, status, error_code, disposition, arguments}``; None without a trace."""
    trace = episode.get("trace")
    if not isinstance(trace, list) or not trace:
        tools = episode.get("tools")
        if isinstance(tools, list) and tools:
            # A tool list without statuses: assume every step executed.
            return [{"tool": str(t), "status": "success", "error_code": None, "disposition": None, "arguments": {}} for t in tools]
        return None
    out: list[dict[str, Any]] = []
    for step in trace:
        if not isinstance(step, Mapping) or step.get("intervention"):
            continue
        if isinstance(step.get("action"), Mapping):  # evaluator record
            action = step["action"]
            tool = str(action.get("tool"))
            arguments = {k: v for k, v in (action.get("arguments") or {}).items() if k != "state_id"}
            status = step.get("execution_status")
            disposition = step.get("candidate_disposition_offline")
        else:  # compound slim trace
            tool = str(step.get("tool"))
            arguments = dict(step.get("arguments") or {})
            status = step.get("status", step.get("execution_status"))
            disposition = step.get("disposition")
        out.append({"tool": tool, "status": status, "error_code": step.get("error_code"), "disposition": disposition,
                    "arguments": arguments, "comparison": step.get("comparison")})
    return out


def truth_flags(episode: Mapping[str, Any]) -> dict[int, bool]:
    """``step index -> target_is_true`` from the compound record's corrections, when present."""
    flags: dict[int, bool] = {}
    for record in episode.get("corrections") or []:
        if isinstance(record, Mapping) and isinstance(record.get("target_is_true"), bool) and _as_int(record.get("step")) is not None:
            flags[int(record["step"])] = bool(record["target_is_true"])
    return flags


def episode_identity(episode: Mapping[str, Any]) -> dict[str, Any]:
    success = episode.get("success")
    if success is None:
        success = episode.get("truth_audited_task_success")
    group = episode.get("class") or episode.get("family") or episode.get("stratum") or "all"
    escalation = episode.get("escalation_request")
    return {
        "id": str(episode.get("scenario_id") or episode.get("episode_key") or episode.get("physical_root") or ""),
        "group": str(group),
        "success": bool(success),
        "terminal": episode.get("terminal"),
        "budget_exhausted": bool(escalation == BUDGET_EXHAUSTED_REQUEST or episode.get("terminal") is False),
        "cardinality": _as_int(episode.get("error_cardinality", episode.get("cardinality"))),
    }


# --------------------------------------------------------------------------- the ledger of one episode


def episode_ledger(episode: Mapping[str, Any], *, atoms: int | None = None,
                   extra_tool_codes: Iterable[str] = ()) -> dict[str, Any]:
    """Attribute one episode's actions.  ``atoms`` overrides the record's own cardinality for the floor."""
    identity = episode_identity(episode)
    steps = normalize_steps(episode)
    atom_count = atoms if atoms is not None else identity["cardinality"]
    row: dict[str, Any] = {
        **identity,
        "atoms": atom_count,
        "trace_available": steps is not None,
        "actions": int(episode.get("steps") or (len(steps) if steps else 0) or 0),
        "wls_calls": _as_int(episode.get("wls_calls")),
        "invalid_action_count": int(episode.get("invalid_action_count") or 0),
        "loop_detected": bool(episode.get("loop_detected")),
    }
    row["floor"] = (2 + 4 * atom_count) if atom_count is not None else None
    row["slack"] = (row["actions"] - row["floor"]) if row["floor"] is not None else None
    if steps is None:
        return row
    flags = truth_flags(episode)
    n = len(steps)
    wasted: set[int] = set()
    counts: Counter[str] = Counter()
    refused_codes: Counter[str] = Counter()
    failed_other_codes: Counter[str] = Counter()
    cycles: list[dict[str, Any]] = []

    # Correction cycles.
    i = 0
    while i < n:
        step = steps[i]
        if step["tool"] in CORRECTION_TOOLS:
            truth = flags.get(i)
            if step["status"] != "success":
                kind = refusal_kind(step["error_code"], step["tool"], extra_tool_codes)
                counts["hypotheses"] += 1
                counts[f"refused_{kind}"] += 1
                counts["refused"] += 1
                if truth is True:
                    counts["refused_true"] += 1
                elif truth is False:
                    counts["refused_wrong"] += 1
                refused_codes[str(step["error_code"])] += 1
                wasted.add(i)
                i += 1
                continue
            members = [i]
            lifecycle = None
            verdict = None
            j = i + 1
            while j < n:
                tool_j = steps[j]["tool"]
                if tool_j in CORRECTION_TOOLS:
                    break
                members.append(j)
                if tool_j in LIFECYCLE_TOOLS and steps[j]["status"] == "success":
                    lifecycle = tool_j
                    verdict = steps[j]["disposition"]
                    break
                if tool_j == FINALIZE_DIAGNOSIS or tool_j == ASK_FOR_MORE_EVIDENCE:
                    members.pop()
                    break
                j += 1
            counts["hypotheses"] += 1
            if lifecycle == COMMIT_STATE:
                counts["commits"] += 1
                counts["accepted"] += 1
                if truth is True:
                    counts["accepted_true"] += 1
                elif truth is False:
                    counts["accepted_wrong"] += 1
                outcome = "accepted"
            elif lifecycle == ROLLBACK_STATE:
                counts["rollbacks"] += 1
                counts["rejected"] += 1
                if truth is True:
                    counts["rejected_true"] += 1
                elif truth is False:
                    counts["rejected_wrong"] += 1
                wasted.update(members)
                counts["wasted_rejected_cycles"] += len(members)
                outcome = "rejected"
            else:
                counts["undisposed"] += 1
                wasted.update(members)
                counts["wasted_undisposed"] += len(members)
                outcome = "undisposed"
            cycles.append({"start": i, "end": members[-1], "family": CORRECTION_FAMILY[step["tool"]], "outcome": outcome,
                           "verdict": verdict, "target_is_true": truth, "length": len(members)})
            i = members[-1] + 1 if members[-1] > i else i + 1
            continue
        i += 1

    committed_by_cycle = {(c["start"], c["end"]): c for c in cycles if c["outcome"] == "accepted"}

    # Other failed actions (not corrections).
    for idx, step in enumerate(steps):
        if step["tool"] in CORRECTION_TOOLS:
            continue
        if step["status"] != "success" or step["tool"] == INVALID_ACTION:
            kind = refusal_kind(step["error_code"], step["tool"], extra_tool_codes)
            counts[f"failed_other_{kind}"] += 1
            counts["failed_other"] += 1
            failed_other_codes[str(step["error_code"] or step["tool"])] += 1
            wasted.add(idx)

    # Routes: a context fetch and the segment up to the next fetch.
    fetch_indices = [idx for idx, s in enumerate(steps) if s["tool"] in CONTEXT_TOOLS and s["status"] == "success"]
    for k, idx in enumerate(fetch_indices):
        end = fetch_indices[k + 1] if k + 1 < len(fetch_indices) else n
        family = CONTEXT_FAMILY[steps[idx]["tool"]]
        counts["routes"] += 1
        productive = any(c["family"] == family and idx < c["start"] < end for c in committed_by_cycle.values())
        if not productive:
            counts["fruitless_routes"] += 1
            counts[f"fruitless_routes_{family}"] += 1
            wasted.add(idx)
            counts["wasted_fruitless_fetches"] += 1

    # Comparisons: exploration; productive when a compared line was committed next.
    for idx, step in enumerate(steps):
        if step["tool"] not in COMPARISON_TOOLS or step["status"] != "success":
            continue
        counts["comparisons"] += 1
        best = None
        if isinstance(step.get("comparison"), Mapping):
            best = step["comparison"].get("best_line")
        productive = False
        for c in cycles:
            if c["start"] <= idx or c["family"] != "parameter":
                continue
            target = steps[c["start"]]["arguments"].get("line_index")
            if c["outcome"] == "accepted" and (best is None or target == best):
                productive = True
            break
        if productive:
            counts["comparisons_productive"] += 1

    # Consecutive solves on an unchanged state.
    for idx in range(1, n):
        if steps[idx]["tool"] == RUN_WLS and steps[idx - 1]["tool"] == RUN_WLS and steps[idx]["status"] == "success":
            counts["repeated_solves"] += 1

    # Revert accounting (item 7 step 6), when present.
    counts["reverts"] = sum(1 for s in steps if s["tool"] == REVERT_STATE and s["status"] == "success")

    row.update({key: counts.get(key, 0) for key in LEDGER_COUNT_KEYS})
    row["wasted"] = len(wasted)
    row["wasted_share_of_actions"] = (len(wasted) / row["actions"]) if row["actions"] else None
    row["wasted_share_of_slack"] = (len(wasted) / row["slack"]) if row["slack"] else None
    row["hypotheses_per_accepted"] = (counts["hypotheses"] / counts["accepted"]) if counts["accepted"] else None
    row["refused_codes"] = dict(refused_codes)
    row["failed_other_codes"] = dict(failed_other_codes)
    row["cycles"] = cycles
    return row


LEDGER_COUNT_KEYS = (
    "hypotheses", "accepted", "accepted_true", "accepted_wrong", "rejected", "rejected_true", "rejected_wrong",
    "refused", "refused_true", "refused_wrong", "refused_tool_refused", "refused_policy_invalid", "undisposed",
    "commits", "rollbacks", "failed_other", "failed_other_tool_refused", "failed_other_policy_invalid",
    "routes", "fruitless_routes", "fruitless_routes_measurement", "fruitless_routes_parameter",
    "fruitless_routes_topology", "comparisons", "comparisons_productive", "repeated_solves", "reverts",
    "wasted_rejected_cycles", "wasted_undisposed", "wasted_fruitless_fetches",
)
MEAN_KEYS = ("actions", "floor", "slack", "wasted", "hypotheses", "accepted", "rejected", "refused",
             "refused_tool_refused", "refused_policy_invalid", "undisposed", "rollbacks", "routes", "fruitless_routes",
             "comparisons", "comparisons_productive", "failed_other", "repeated_solves", "wasted_rejected_cycles",
             "wasted_undisposed", "wasted_fruitless_fetches")
TRUTH_KEYS = ("accepted_true", "accepted_wrong", "rejected_true", "rejected_wrong", "refused_true", "refused_wrong")


# --------------------------------------------------------------------------- aggregation


def _mean(values: Sequence[float | None]) -> float | None:
    present = [float(v) for v in values if v is not None]
    return (sum(present) / len(present)) if present else None


def aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Means per group x outcome, with the totals every mean is read from."""
    out: dict[str, Any] = {"groups": {}, "all": {}}

    def block(subset: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        traced = [r for r in subset if r.get("trace_available")]
        entry: dict[str, Any] = {"n": len(subset), "n_traced": len(traced), "solved": sum(1 for r in subset if r["success"]),
                                 "budget_exhausted": sum(1 for r in subset if r.get("budget_exhausted"))}
        for key in MEAN_KEYS:
            source = subset if key in ("actions", "floor", "slack") else traced
            entry[f"mean_{key}"] = _mean([r.get(key) for r in source])
            entry[f"total_{key}"] = sum(float(r.get(key) or 0) for r in source if r.get(key) is not None)
        total_actions = entry["total_actions"]
        total_slack = entry["total_slack"]
        entry["wasted_share_of_actions"] = (entry["total_wasted"] / total_actions) if total_actions else None
        entry["wasted_share_of_slack"] = (entry["total_wasted"] / total_slack) if total_slack else None
        accepted = entry["total_accepted"]
        entry["hypotheses_per_accepted"] = (entry["total_hypotheses"] / accepted) if accepted else None
        for key in TRUTH_KEYS:
            entry[f"total_{key}"] = sum(int(r.get(key) or 0) for r in traced)
        entry["refused_codes"] = dict(sum((Counter(r.get("refused_codes") or {}) for r in traced), Counter()))
        entry["failed_other_codes"] = dict(sum((Counter(r.get("failed_other_codes") or {}) for r in traced), Counter()))
        return entry

    by_group: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[row["group"]].append(row)
    for group, subset in sorted(by_group.items()):
        out["groups"][group] = {
            "all": block(subset),
            "solved": block([r for r in subset if r["success"]]),
            "failed": block([r for r in subset if not r["success"]]),
        }
    out["all"] = {
        "all": block(rows),
        "solved": block([r for r in rows if r["success"]]),
        "failed": block([r for r in rows if not r["success"]]),
    }
    return out


def _fmt(value: Any, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def markdown(aggregated: Mapping[str, Any], title: str) -> str:
    lines = [f"# Detour ledger: {title}", "",
             "Means per episode. Floor = 2 + 4 x atoms. Wasted = rejected cycles + undisposed cycles + refused corrections "
             "+ other failed actions + fruitless context fetches (union of steps).", "",
             "| group | outcome | n | actions | floor | slack | wasted | wasted / slack | hypotheses per accepted | "
             "rejected | refused (tool / policy) | rollbacks | fruitless routes / routes | comparisons (productive) | "
             "accepted wrong | rejected or refused true |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    groups = list(aggregated["groups"].items()) + [("all", aggregated["all"])]
    for group, outcomes in groups:
        for outcome in ("all", "solved", "failed"):
            e = outcomes[outcome]
            if not e["n"]:
                continue
            lines.append(
                f"| {group} | {outcome} | {e['n']} | {_fmt(e['mean_actions'])} | {_fmt(e['mean_floor'])} | "
                f"{_fmt(e['mean_slack'])} | {_fmt(e['mean_wasted'])} | {_fmt(e['wasted_share_of_slack'], 2)} | "
                f"{_fmt(e['hypotheses_per_accepted'], 2)} | {_fmt(e['mean_rejected'], 2)} | "
                f"{_fmt(e['mean_refused_tool_refused'], 2)} / {_fmt(e['mean_refused_policy_invalid'], 2)} | "
                f"{_fmt(e['mean_rollbacks'], 2)} | {_fmt(e['mean_fruitless_routes'], 2)} / {_fmt(e['mean_routes'], 2)} | "
                f"{_fmt(e['mean_comparisons'], 2)} ({_fmt(e['mean_comparisons_productive'], 2)}) | "
                f"{e['total_accepted_wrong']} | {e['total_rejected_true'] + e['total_refused_true']} |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- class atoms from a manifest


def atoms_by_class(manifest: Mapping[str, Any] | None) -> dict[str, float]:
    """Error atoms per compound class from a suite manifest's ``classes`` (a range counts at its midpoint)."""
    out: dict[str, float] = {}
    for spec in (manifest or {}).get("classes") or []:
        if not isinstance(spec, Mapping) or not spec.get("name"):
            continue
        total = 0.0
        for key in ("parameters", "meters", "breakers"):
            value = spec.get(key, 0)
            if isinstance(value, (list, tuple)) and len(value) == 2:
                total += (float(value[0]) + float(value[1])) / 2.0
            elif isinstance(value, (int, float)):
                total += float(value)
        out[str(spec["name"])] = total
    return out


def parse_atoms(items: Sequence[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for item in items:
        name, _, value = item.partition("=")
        out[name.strip()] = float(value)
    return out


# --------------------------------------------------------------------------- class-level approximation from summary.json


def _parse_verdict_key(key: str) -> tuple[str, str, str] | None:
    parts = key.split(":")
    if len(parts) != 3:
        return None
    return parts[0], parts[1], parts[2]


def summary_ledger(summary: Mapping[str, Any], *, atoms: Mapping[str, float],
                   rejected_cost: float = 3.0, refused_cost: float = 1.0, undisposed_cost: float = 2.0) -> dict[str, Any]:
    """Approximate ledger per class from a rollout summary's verdict counts."""
    out: dict[str, Any] = {"contract": LEDGER_CONTRACT, "mode": "summary", "expert": summary.get("expert"),
                           "expert_options": summary.get("expert_options"), "max_steps": summary.get("max_steps"),
                           "system": summary.get("system"), "assumptions": {"rejected_cost": rejected_cost,
                                                                             "refused_cost": refused_cost,
                                                                             "undisposed_cost": undisposed_cost},
                           "classes": {}}
    totals: Counter[str] = Counter()
    total_slack = 0.0
    total_actions = 0.0
    for name, entry in (summary.get("by_class") or {}).items():
        if not isinstance(entry, Mapping):
            continue
        n = int(entry.get("n") or 0)
        counts: Counter[str] = Counter()
        for key, count in (entry.get("verdicts") or {}).items():
            parsed = _parse_verdict_key(str(key))
            if parsed is None:
                continue
            tool, truth, verdict = parsed
            if tool not in CORRECTION_TOOLS:
                continue
            count = int(count)
            counts["hypotheses"] += count
            if verdict in ACCEPTED_VERDICTS:
                counts["accepted"] += count
                counts["accepted_true" if truth == "true" else "accepted_wrong"] += count
            elif verdict == "refused":
                counts["refused"] += count
                counts["refused_true" if truth == "true" else "refused_wrong"] += count
            elif verdict == "undisposed":
                counts["undisposed"] += count
            else:  # REJECT, INCONCLUSIVE
                counts["rejected"] += count
                counts["rejected_true" if truth == "true" else "rejected_wrong"] += count
        mean_steps = float(entry.get("mean_steps") or 0.0)
        atom_count = atoms.get(name)
        floor = (2.0 + 4.0 * atom_count) if atom_count is not None else None
        slack = (mean_steps - floor) if floor is not None else None
        est_wasted = rejected_cost * counts["rejected"] + refused_cost * counts["refused"] + undisposed_cost * counts["undisposed"]
        row = {
            "n": n, "solved": int(entry.get("success") or 0), "mean_actions": mean_steps, "atoms": atom_count,
            "floor": floor, "slack": slack, "mean_wls_calls": entry.get("mean_wls_calls"),
            "comparisons_per_root": entry.get("mean_comparisons"),
            **{k: counts[k] for k in ("hypotheses", "accepted", "accepted_true", "accepted_wrong", "rejected",
                                      "rejected_true", "rejected_wrong", "refused", "refused_true", "refused_wrong",
                                      "undisposed")},
            "hypotheses_per_accepted": (counts["hypotheses"] / counts["accepted"]) if counts["accepted"] else None,
            "est_wasted_actions": est_wasted,
            "est_wasted_per_root": (est_wasted / n) if n else None,
            "est_wasted_share_of_slack": (est_wasted / (slack * n)) if slack and n else None,
            "est_wasted_share_of_actions": (est_wasted / (mean_steps * n)) if mean_steps and n else None,
            "invalid_action_episodes": entry.get("invalid_action_episodes"),
            "loop_episodes": entry.get("loop_episodes"),
            "false_commit_episodes": entry.get("false_commit_episodes"),
        }
        out["classes"][name] = row
        for k in ("hypotheses", "accepted", "accepted_true", "accepted_wrong", "rejected", "rejected_true",
                  "rejected_wrong", "refused", "refused_true", "refused_wrong", "undisposed"):
            totals[k] += counts[k]
        totals["n"] += n
        totals["solved"] += int(entry.get("success") or 0)
        totals["est_wasted_actions"] += est_wasted
        total_actions += mean_steps * n
        if slack is not None:
            total_slack += slack * n
    n_all = totals["n"]
    out["all"] = {
        **{k: totals[k] for k in totals},
        "mean_actions": (total_actions / n_all) if n_all else None,
        "est_wasted_per_root": (totals["est_wasted_actions"] / n_all) if n_all else None,
        "est_wasted_share_of_slack": (totals["est_wasted_actions"] / total_slack) if total_slack else None,
        "est_wasted_share_of_actions": (totals["est_wasted_actions"] / total_actions) if total_actions else None,
        "hypotheses_per_accepted": (totals["hypotheses"] / totals["accepted"]) if totals["accepted"] else None,
        "mean_slack": (total_slack / n_all) if n_all else None,
    }
    return out


def summary_markdown(ledgers: Mapping[str, Mapping[str, Any]]) -> str:
    lines = ["# Detour ledger, class-level approximation from rollout summaries", "",
             "Assumes a rejected candidate cost 3 actions (correction, verifying solve, rollback), a refused one 1, "
             "an undisposed one 2. Floor = 2 + 4 x atoms (control_meter at 2 atoms, the midpoint of 1 to 3).", "",
             "| run | class | n | solved | actions | floor | slack | est. wasted | wasted / slack | hypotheses per accepted "
             "| rejected (true / wrong) | refused (true / wrong) | accepted wrong | comparisons / root |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for run, ledger in ledgers.items():
        for name, row in list(ledger["classes"].items()) + [("all", ledger["all"])]:
            if name == "all":
                lines.append(
                    f"| {run} | **all** | {row['n']} | {row['solved']} | {_fmt(row['mean_actions'])} | | "
                    f"{_fmt(row['mean_slack'])} | {_fmt(row['est_wasted_per_root'])} | "
                    f"{_fmt(row['est_wasted_share_of_slack'], 2)} | {_fmt(row['hypotheses_per_accepted'], 2)} | "
                    f"{row['rejected']} ({row['rejected_true']} / {row['rejected_wrong']}) | "
                    f"{row['refused']} ({row['refused_true']} / {row['refused_wrong']}) | {row['accepted_wrong']} | |")
            else:
                lines.append(
                    f"| {run} | {name} | {row['n']} | {row['solved']} | {_fmt(row['mean_actions'])} | {_fmt(row['floor'])} | "
                    f"{_fmt(row['slack'])} | {_fmt(row['est_wasted_per_root'])} | {_fmt(row['est_wasted_share_of_slack'], 2)} | "
                    f"{_fmt(row['hypotheses_per_accepted'], 2)} | {row['rejected']} ({row['rejected_true']} / {row['rejected_wrong']}) | "
                    f"{row['refused']} ({row['refused_true']} / {row['refused_wrong']}) | {row['accepted_wrong']} | "
                    f"{_fmt(row['comparisons_per_root'], 2)} |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- I/O


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run_episodes(args: argparse.Namespace) -> int:
    episodes = read_jsonl(args.episodes)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8")) if args.manifest else None
    atoms = atoms_by_class(manifest)
    atoms.update(parse_atoms(args.atoms))
    rows = []
    for episode in episodes:
        group = str(episode.get("class") or episode.get("family") or "")
        atom_count = atoms.get(group)
        rows.append(episode_ledger(episode, atoms=int(round(atom_count)) if atom_count is not None else None,
                                   extra_tool_codes=args.tool_refused_code))
    aggregated = aggregate(rows)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    with (out / "ledger.jsonl").open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    payload = {"contract": LEDGER_CONTRACT, "mode": "episodes", "episodes": str(args.episodes),
               "manifest": str(args.manifest) if args.manifest else None, "atoms_by_class": atoms,
               "rows": len(rows), "aggregate": aggregated}
    (out / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    text = markdown(aggregated, args.title or str(args.episodes))
    (out / "summary.md").write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def run_summary(args: argparse.Namespace) -> int:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8")) if args.manifest else None
    atoms = atoms_by_class(manifest)
    atoms.update(parse_atoms(args.atoms))
    ledgers: dict[str, dict[str, Any]] = {}
    for item in args.summary:
        name, _, path = item.partition("=")
        if not path:
            path, name = name, Path(name).stem
        summary = json.loads(Path(path).read_text(encoding="utf-8"))
        ledgers[name] = summary_ledger(summary, atoms=atoms, rejected_cost=args.rejected_cost,
                                      refused_cost=args.refused_cost, undisposed_cost=args.undisposed_cost)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary_ledgers.json").write_text(json.dumps({"contract": LEDGER_CONTRACT, "atoms_by_class": atoms,
                                                           "ledgers": ledgers}, indent=2, sort_keys=True) + "\n",
                                               encoding="utf-8")
    text = summary_markdown(ledgers)
    (out / "summary_ledgers.md").write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)
    ep = sub.add_parser("episodes", help="per-episode ledger from episodes.jsonl")
    ep.add_argument("--episodes", type=Path, required=True)
    ep.add_argument("--manifest", type=Path, default=None, help="suite manifest.json; its classes give the atoms per class")
    ep.add_argument("--atoms", action="append", default=[], help="CLASS=ATOMS override, repeatable")
    ep.add_argument("--tool-refused-code", action="append", default=[], help="extra error codes to count as tool refusals")
    ep.add_argument("--title", default=None)
    ep.add_argument("--output-dir", type=Path, required=True)
    ep.set_defaults(func=run_episodes)
    sm = sub.add_parser("summary", help="class-level approximation from rollout summary.json files")
    sm.add_argument("--summary", action="append", required=True, help="NAME=path/to/summary.json, repeatable")
    sm.add_argument("--manifest", type=Path, default=None)
    sm.add_argument("--atoms", action="append", default=[])
    sm.add_argument("--rejected-cost", type=float, default=3.0)
    sm.add_argument("--refused-cost", type=float, default=1.0)
    sm.add_argument("--undisposed-cost", type=float, default=2.0)
    sm.add_argument("--output-dir", type=Path, required=True)
    sm.set_defaults(func=run_summary)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
