"""Roll the rule expert out on a compound suite in parallel and summarize per class.

    python -m research.compound_errors.rollout --suite output/compound_errors_20261027/suite.json \\
        --output-dir output/compound_errors_20261027/expert_baseline_compare \\
        --profile scada_only --expert baseline_compare --workers 14

Each worker holds the decision audit's harness (the research environment of
``scripts/run_dagger_research.py`` under one evidence profile, the named
expert variant through the policy-observation boundary, the paired-evaluation
contract, the 40-action budget) and rolls the expert out once per root.  The
episode record keeps what the study reads: the truth-audited outcome and its
basis, the workflow counters, a slim trace, and every correction with its
verification metrics, its verdict (the offline disposition the lifecycle step
carried) and whether its target is a true one.  The summary gives per class
the success, the mean actions and WLS calls, the verdict distribution of the
corrections and the failure reasons.

``--system`` (2026-10-28) names the suite's registered system; by default it
is read from the roots, which must all belong to it.  A system other than
IEEE 14 runs with the progress basis its research cells use,
``chi_square_excess`` (the verifier's partial-progress floors were set on IEEE
14's 95 degrees of freedom; on IEEE 118's 863 the whole-statistic basis makes
them several times stricter), unless ``--progress-basis`` names one:

    python -m research.compound_errors.rollout --suite output/compound_errors_ieee118_20261028/suite.json \\
        --output-dir output/compound_errors_ieee118_20261028/expert_baseline_compare --system case118 --workers 28
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from psse_env.actions import (  # noqa: E402
    ASK_FOR_MORE_EVIDENCE,
    CORRECT_MEASUREMENTS,
    CORRECT_PARAMETERS,
    CORRECT_TOPOLOGY,
    GET_PARAMETER_CONTEXT,
    GET_TOPOLOGY_CONTEXT,
    RUN_WLS,
)
from research.rl import decision_audit as audit  # noqa: E402
from research.rl.decision_audit import parse_run_options  # noqa: E402

ROLLOUT_CONTRACT = "compound_expert_rollout_v1"
COMPARE_TOOL = "compare_parameter_candidates"
VERDICTS = ("ACCEPT_FINAL", "ACCEPT_PARTIAL", "REJECT", "INCONCLUSIVE", "refused", "undisposed")
_WLS_KEYS = ("chi_square_statistic", "chi_square_threshold", "chi_square_ratio", "globally_resolved", "global_progress",
             "target_progress", "remaining_suspect_count", "max_normalized_residual", "chi_square_dof", "flow_at_noise_lines")
#: The progress basis of a system's research cells: IEEE 14 keeps the environment's default.
SYSTEM_PROGRESS_BASIS = {"case14": None}
LARGE_SYSTEM_PROGRESS_BASIS = "chi_square_excess"


def suite_system(roots: Sequence[Mapping[str, Any]], system: str | None = None) -> str:
    """The registered system of every root (``grouping.case_id``); a named system must match all of them."""
    from psse_env.systems import resolve_system

    found = {str((root.get("grouping") or {}).get("case_id") or "case14") for root in roots}
    cases = {resolve_system(case).case_id for case in found}
    if system is not None:
        named = resolve_system(system).case_id
        if cases - {named}:
            raise SystemExit(f"suite roots belong to {sorted(cases)}, not {named}")
        return named
    if len(cases) > 1:
        raise SystemExit(f"suite mixes systems {sorted(cases)}; name one with --system")
    return next(iter(cases)) if cases else "case14"


def system_progress_basis(system: str, progress_basis: str | None) -> str | None:
    """The named basis, else the system's (None keeps the environment's default)."""
    if progress_basis:
        return progress_basis
    return SYSTEM_PROGRESS_BASIS.get(system, LARGE_SYSTEM_PROGRESS_BASIS)


def truth_of(root: Mapping[str, Any]) -> dict[str, Any]:
    truth = (root.get("audit") or {}).get("truth") or {}
    return {
        "meters": {int(item["index"]) for item in truth.get("true_measurement_errors") or [] if item.get("index") is not None},
        "lines": {int(item["line_index1"]) for item in truth.get("true_parameter_errors") or []},
        "breakers": {str(item["cb_name"]) for item in truth.get("true_topology_errors") or [] if item.get("cb_name")},
        "dangling_lines": {int(item["line_index1"]) for item in truth.get("true_topology_errors") or [] if item.get("line_index1")},
    }


def target_is_true(record: Mapping[str, Any], truth: Mapping[str, Any]) -> bool | None:
    arguments = record.get("arguments") or {}
    tool = record.get("tool")
    if tool == CORRECT_MEASUREMENTS:
        group = arguments.get("suspect_group")
        if group is None:
            index = arguments.get("index", arguments.get("measurement_index"))
            group = [index] if index is not None else []
        try:
            targets = {int(i) for i in group}
        except (TypeError, ValueError):
            return None
        return bool(targets) and targets <= set(truth["meters"])
    if tool == CORRECT_PARAMETERS:
        line = arguments.get("line_index")
        return int(line) in truth["lines"] if isinstance(line, int) else None
    if tool == CORRECT_TOPOLOGY:
        if arguments.get("cb_name") is not None:
            return str(arguments["cb_name"]) in truth["breakers"]
        line = arguments.get("line_index")
        return int(line) in truth["dangling_lines"] if isinstance(line, int) else None
    return None


def slim_trace(steps: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The policy steps without their observations: tool, target, status, verdict and the solve's metrics."""
    out = []
    for index, step in enumerate(steps):
        action = step.get("action") or {}
        tool = str(action.get("tool"))
        arguments = {k: v for k, v in (action.get("arguments") or {}).items() if k != "state_id"}
        record: dict[str, Any] = {"step": index, "tool": tool, "arguments": arguments,
                                  "status": step.get("execution_status"), "error_code": step.get("error_code"),
                                  "disposition": step.get("candidate_disposition_offline")}
        output = step.get("policy_tool_output") if isinstance(step.get("policy_tool_output"), Mapping) else {}
        metrics = output.get("tool_metrics") if isinstance(output.get("tool_metrics"), Mapping) else {}
        if tool == RUN_WLS:
            record["metrics"] = {key: metrics.get(key) for key in _WLS_KEYS if key in metrics}
            record["signatures"] = [str(s) for s in output.get("unresolved_signatures") or []][:6]
        elif tool == GET_PARAMETER_CONTEXT:
            record["ranking"] = {
                "lines": [int(item["line_index1"]) for item in metrics.get("parameter_ranking_distinct_lines") or []][:5],
                "dominance_ratio": metrics.get("parameter_ranking_dominance_ratio"),
                "dominant": metrics.get("parameter_ranking_dominant"),
                "route_status": metrics.get("route_status"),
                "supported": [a.get("arguments", {}).get("line_index") for a in metrics.get("supported_corrections") or []],
            }
        elif tool == GET_TOPOLOGY_CONTEXT:
            findings = metrics.get("breaker_findings") or []
            record["ranking"] = {
                "breakers": [{"cb": f.get("cb_name"), "rank": f.get("rank"), "clean": f.get("flip_explains_substation_measurements"),
                              "progress": f.get("gse_progress_after_flip"), "effect": f.get("bus_branch_effect")} for f in findings[:5]],
                "route_status": metrics.get("route_status"),
                "supported": [a.get("arguments", {}).get("cb_name") or a.get("arguments", {}).get("line_index")
                              for a in metrics.get("supported_corrections") or []],
            }
        elif tool == COMPARE_TOOL:
            record["comparison"] = {key: metrics.get(key) for key in ("best_line", "ranked_lines", "verdicts", "order") if key in metrics}
        out.append(record)
    return out


def compact_episode(episode: Mapping[str, Any], root: Mapping[str, Any]) -> dict[str, Any]:
    steps = audit.policy_steps(episode.get("trace") or [])
    tools = [str((step.get("action") or {}).get("tool")) for step in steps]
    truth = truth_of(root)
    corrections = audit.correction_outcomes(steps)
    for record in corrections:
        record["target_is_true"] = target_is_true(record, truth)
    compound = (root.get("audit") or {}).get("compound") or {}
    return {
        "contract": ROLLOUT_CONTRACT,
        "scenario_id": str(root["execution"]["scenario_id"]),
        "class": compound.get("class"),
        "family": str(root.get("grouping", {}).get("scenario_family")),
        "stratum": str((root.get("audit") or {}).get("parameter_ranking", {}).get("stratum")),
        "error_cardinality": root.get("grouping", {}).get("error_cardinality"),
        "success": bool(episode.get("truth_audited_task_success")),
        "basis": audit._basis(episode),
        "audit_reasons": audit.truth_audit_reasons(episode),
        "terminal": bool(episode.get("terminal")),
        "terminal_outcome": episode.get("terminal_outcome"),
        "steps": len(tools),
        "wls_calls": int(episode.get("wls_calls") or 0),
        "invalid_action_count": int(episode.get("invalid_action_count") or 0),
        "false_commit_count": int(episode.get("false_commit_count") or 0),
        "loop_detected": bool(episode.get("loop_detected")),
        "healthy_components_preserved": episode.get("healthy_components_preserved"),
        "comparisons": tools.count(COMPARE_TOOL),
        "escalation_request": next((str((s.get("action") or {}).get("arguments", {}).get("request"))
                                    for s in reversed(steps) if str((s.get("action") or {}).get("tool")) == ASK_FOR_MORE_EVIDENCE
                                    and s.get("execution_status") == "success"), None),
        "abstentions": handoff_abstentions(steps),
        "tools": tools,
        "corrections": corrections,
        "trace": slim_trace(steps),
        "seconds": episode.get("seconds"),
    }


def handoff_abstentions(steps: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The abstention statements of the episode's successful handoff, as the environment recorded them (stage 2 item 5)."""
    for step in reversed(steps):
        action = step.get("action") or {}
        if str(action.get("tool")) != ASK_FOR_MORE_EVIDENCE or step.get("execution_status") != "success":
            continue
        output = step.get("policy_tool_output") or step.get("tool_output") or step.get("outcome") or {}
        metrics = output.get("tool_metrics") if isinstance(output, Mapping) else None
        recorded = metrics.get("abstentions") if isinstance(metrics, Mapping) else None
        if isinstance(recorded, (list, tuple)):
            return [dict(item) for item in recorded if isinstance(item, Mapping)]
        declared = (action.get("arguments") or {}).get("abstentions")
        if isinstance(declared, (list, tuple)):
            return [dict(item) for item in declared if isinstance(item, Mapping)]
        return []
    return []


def _run_root(payload: Mapping[str, Any]) -> dict[str, Any]:
    harness = audit._HARNESS
    root = payload["root"]
    started = time.perf_counter()
    try:
        episode = harness.run(root, harness.expert_policy())
        record = compact_episode(episode, root)
    except Exception as exc:  # keep the sweep going; the record names the failure
        compound = (root.get("audit") or {}).get("compound") or {}
        record = {"contract": ROLLOUT_CONTRACT, "scenario_id": str(root["execution"]["scenario_id"]), "class": compound.get("class"),
                  "family": str(root.get("grouping", {}).get("scenario_family")), "success": False, "error": f"{type(exc).__name__}: {exc}",
                  "steps": 0, "wls_calls": 0, "corrections": [], "tools": [], "trace": []}
    record["seconds"] = round(time.perf_counter() - started, 1)
    return record


def summarize(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_class: dict[str, dict[str, Any]] = {}
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record.get("class"))].append(record)
    for name, rows in sorted(groups.items()):
        counter: Counter = Counter()
        verdicts: Counter = Counter()
        reasons: Counter = Counter()
        basis: Counter = Counter()
        outcomes: Counter = Counter()
        for row in rows:
            counter["n"] += 1
            counter["success"] += bool(row.get("success"))
            counter["errors"] += bool(row.get("error"))
            counter["steps"] += int(row.get("steps") or 0)
            counter["wls_calls"] += int(row.get("wls_calls") or 0)
            counter["comparisons"] += int(row.get("comparisons") or 0)
            counter["invalid_action_episodes"] += int(int(row.get("invalid_action_count") or 0) > 0)
            counter["false_commit_episodes"] += int(int(row.get("false_commit_count") or 0) > 0)
            counter["loop_episodes"] += int(bool(row.get("loop_detected")))
            counter["healthy_touched"] += int(row.get("healthy_components_preserved") is False)
            declared = row.get("abstentions") or []
            counter["abstention_episodes"] += int(bool(declared))
            counter["abstentions_unsupported"] += sum(1 for item in declared if item.get("supported") is False)
            outcomes[str(row.get("terminal_outcome") or ("nonterminal" if not row.get("terminal") else "terminal"))] += 1
            basis[str(row.get("basis"))] += 1
            if not row.get("success"):
                for reason in row.get("audit_reasons") or []:
                    reasons[reason] += 1
            for correction in row.get("corrections") or []:
                truthful = correction.get("target_is_true")
                side = "true" if truthful else "wrong" if truthful is False else "unknown"
                if correction.get("execution_status") != "success":
                    verdict = "refused"
                else:
                    verdict = str(correction.get("verdict") or "undisposed")
                verdicts[f"{correction['tool']}:{side}:{verdict}"] += 1
                counter[f"corrections_{side}"] += 1
        entry = dict(counter)
        entry["success_rate"] = counter["success"] / max(1, counter["n"])
        entry["mean_steps"] = counter["steps"] / max(1, counter["n"])
        entry["mean_wls_calls"] = counter["wls_calls"] / max(1, counter["n"])
        entry["mean_comparisons"] = counter["comparisons"] / max(1, counter["n"])
        entry["verdicts"] = dict(sorted(verdicts.items()))
        entry["failure_reasons"] = dict(reasons.most_common())
        entry["basis"] = dict(basis)
        entry["terminal_outcomes"] = dict(outcomes)
        by_class[name] = entry
    totals = Counter()
    for entry in by_class.values():
        for key in ("n", "success", "errors", "steps", "wls_calls", "comparisons", "invalid_action_episodes",
                    "false_commit_episodes", "loop_episodes", "abstention_episodes", "abstentions_unsupported"):
            totals[key] += int(entry.get(key, 0) or 0)
    return {"by_class": by_class, "totals": dict(totals)}


def verdict_cells(entry: Mapping[str, Any]) -> dict[str, int]:
    """True-target corrections by verdict, and wrong-target commits, summed over tools."""
    cells = Counter()
    for key, count in (entry.get("verdicts") or {}).items():
        tool, side, verdict = key.split(":")
        if side == "true":
            cells[verdict] += count
        elif side == "wrong" and verdict in {"ACCEPT_FINAL", "ACCEPT_PARTIAL"}:
            cells["wrong_accepted"] += count
        elif side == "wrong":
            cells["wrong_rejected"] += count
    return dict(cells)


def markdown(summary: Mapping[str, Any], *, title: str) -> str:
    lines = [f"# {title}", "",
             "| class | n | solved | rate | mean actions | mean WLS | comparisons | true corrections final / partial / rejected / refused | wrong accepted / rejected | handoff | budget | false-commit ep. | abstained (unsupported) |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, entry in summary["by_class"].items():
        cells = verdict_cells(entry)
        outcomes = entry.get("terminal_outcomes") or {}
        lines.append(
            f"| {name} | {entry['n']} | {entry['success']} | {100 * entry['success_rate']:.0f}% | {entry['mean_steps']:.1f} | "
            f"{entry['mean_wls_calls']:.2f} | {entry['comparisons']} | {cells.get('ACCEPT_FINAL', 0)} / {cells.get('ACCEPT_PARTIAL', 0)} / "
            f"{cells.get('REJECT', 0)} / {cells.get('refused', 0)} | {cells.get('wrong_accepted', 0)} / {cells.get('wrong_rejected', 0)} | "
            f"{outcomes.get('operator_escalation', 0)} | {outcomes.get('nonterminal', 0)} | {entry.get('false_commit_episodes', 0)} | "
            f"{entry.get('abstention_episodes', 0)} ({entry.get('abstentions_unsupported', 0)}) |"
        )
    totals = summary["totals"]
    lines.append(f"| all | {totals['n']} | {totals['success']} | {100 * totals['success'] / max(1, totals['n']):.0f}% | "
                 f"{totals['steps'] / max(1, totals['n']):.1f} | {totals['wls_calls'] / max(1, totals['n']):.2f} | {totals['comparisons']} | | | | | | "
                 f"{totals.get('abstention_episodes', 0)} ({totals.get('abstentions_unsupported', 0)}) |")
    return "\n".join(lines)


def load_suite(path: Path) -> list[dict[str, Any]]:
    suite = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(suite, Mapping):
        suite = suite.get("scenarios") or suite.get("roots") or []
    return list(suite)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--profile", default="scada_only")
    parser.add_argument("--expert", default="baseline_compare")
    parser.add_argument("--progress-basis", default=None)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--classes", nargs="*", default=None)
    parser.add_argument("--roots", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--per-class", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--env-option", action="append", default=None, metavar="KEY=VALUE",
                        help="research environment option, e.g. pair_refit_adjacent=1 (stage 2 item 3)")
    parser.add_argument("--expert-option", action="append", default=None, metavar="KEY=VALUE",
                        help="expert option on top of the variant, e.g. abstain_isolated_lines=1 (stage 2 item 5)")
    parser.add_argument("--system", default=None, help="the suite's registered system (default: read from the roots)")
    args = parser.parse_args(argv)

    roots = load_suite(args.suite)
    system = suite_system(roots, args.system)
    args.progress_basis = system_progress_basis(system, args.progress_basis)
    if args.classes:
        wanted = set(args.classes)
        roots = [r for r in roots if ((r.get("audit") or {}).get("compound") or {}).get("class") in wanted]
    if args.roots:
        wanted_roots = set(args.roots)
        roots = [r for r in roots if str(r["execution"]["scenario_id"]) in wanted_roots]
    if args.per_class:
        taken: Counter = Counter()
        kept = []
        for root in roots:
            name = ((root.get("audit") or {}).get("compound") or {}).get("class")
            if taken[name] < args.per_class:
                taken[name] += 1
                kept.append(root)
        roots = kept
    if args.limit:
        roots = roots[: args.limit]
    settings = dict(profile=args.profile, expert_variant=args.expert, triage_model=None, progress_basis=args.progress_basis,
                    max_steps=args.max_steps, seed=args.seed, reward=None,
                    env_options=parse_run_options(args.env_option), expert_options=parse_run_options(args.expert_option))
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    print(f"[rollout] {len(roots)} {system} roots from {args.suite} under {args.profile} ({args.expert}), "
          f"progress basis {args.progress_basis or 'default'}, workers {args.workers}", flush=True)
    started = time.perf_counter()
    payloads = [{"root": root} for root in roots]
    records: list[dict[str, Any]] = []
    stream = (out / "episodes.jsonl").open("w", encoding="utf-8")
    try:
        if args.workers > 1:
            with ProcessPoolExecutor(max_workers=args.workers, initializer=audit._init_worker, initargs=(settings,)) as pool:
                iterator = pool.map(_run_root, payloads, chunksize=1)
                for record in iterator:
                    records.append(record)
                    stream.write(json.dumps(record, sort_keys=True, default=str) + "\n")
                    stream.flush()
                    if len(records) % 25 == 0 or len(records) == len(payloads):
                        solved = sum(bool(r.get("success")) for r in records)
                        print(json.dumps({"done": len(records), "solved": solved, "elapsed": round(time.perf_counter() - started, 1)}), flush=True)
        else:
            audit._init_worker(settings)
            for payload in payloads:
                record = _run_root(payload)
                records.append(record)
                stream.write(json.dumps(record, sort_keys=True, default=str) + "\n")
                stream.flush()
                print(json.dumps({"root": record["scenario_id"], "class": record.get("class"), "success": record.get("success"),
                                  "steps": record.get("steps"), "seconds": record.get("seconds"), "error": record.get("error")}), flush=True)
    finally:
        stream.close()
    summary = summarize(records)
    summary.update(contract=ROLLOUT_CONTRACT, suite=str(args.suite), system=system, profile=args.profile, expert=args.expert,
                   progress_basis=args.progress_basis, max_steps=args.max_steps, seed=args.seed, roots=len(roots),
                   env_options=settings["env_options"], expert_options=settings["expert_options"],
                   workers=args.workers, total_seconds=round(time.perf_counter() - started, 1))
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    text = markdown(summary, title=f"Expert {args.expert} under {args.profile} on {len(roots)} compound roots")
    (out / "summary.md").write_text(text + "\n", encoding="utf-8")
    print(text, flush=True)
    print(json.dumps({"solved": summary["totals"]["success"], "roots": summary["totals"]["n"], "seconds": summary["total_seconds"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
