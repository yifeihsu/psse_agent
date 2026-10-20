"""End-to-end check of the bounded candidate comparison on the parameter-bearing hard roots (2026-10-19).

    python -m research.rl.candidate_comparison_check --suite output/rl_hard_draw_20261015/hard_training_screen.json \\
        --expert ledger_ranked_compare --output-dir output/decision_audit_20261019/compare_train96 --workers 4
    python -m research.rl.candidate_comparison_check ... --expert ledger_ranked --output-dir .../control_train96
    python -m research.rl.candidate_comparison_check --summarize output/decision_audit_20261019/compare_train96 \\
        --baseline output/decision_audit_20261019/control_train96

Rolls the observable rule expert of one variant out on every root of the
parameter-bearing strata (misranked, ambiguous, dominant; with and without a
meter error) through the audit harness (``research.rl.decision_audit._Harness``:
the research environment under ``suspicion_gated_diagnostics`` and the paired
evaluation contract) and records per root the truth-audited outcome, the
charged steps, the comparisons taken (their lines, verdicts and best line),
the first parameter line corrected and the context's top-ranked line.  With a
baseline directory the summary pairs the roots and lists every change.
"""
from __future__ import annotations

import argparse
import copy
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

from psse_env.actions import COMPARE_PARAMETER_CANDIDATES, CORRECT_PARAMETERS  # noqa: E402
from research.rl.decision_audit import _Harness, outcome_of, policy_steps, select_roots  # noqa: E402

CHECK_CONTRACT = "candidate_comparison_check_v1"
PARAMETER_STRATA = (
    "parameter:misranked", "parameter:ambiguous", "parameter:dominant",
    "measurement+parameter:misranked", "measurement+parameter:ambiguous", "measurement+parameter:dominant",
)

_HARNESS: _Harness | None = None


def _init_worker(settings: Mapping[str, Any]) -> None:
    global _HARNESS
    _HARNESS = _Harness(**settings)


def comparisons_of(trace: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Every comparison the policy took, with its verdict summary."""
    rows: list[dict[str, Any]] = []
    for index, step in enumerate(trace):
        action = step.get("action") or {}
        if str(action.get("tool")) != COMPARE_PARAMETER_CANDIDATES:
            continue
        output = step.get("policy_tool_output") or {}
        metrics = output.get("tool_metrics") or {}
        rows.append({
            "step": index,
            "state_id": (action.get("arguments") or {}).get("state_id"),
            "line_indices": list((action.get("arguments") or {}).get("line_indices") or []),
            "execution_status": output.get("execution_status"),
            "error_code": output.get("error_code"),
            "best_line_index": metrics.get("best_line_index"),
            "ranked_line_indices": list(metrics.get("ranked_line_indices") or []),
            "verdicts": [
                {key: candidate.get(key) for key in (
                    "line_index", "verification_verdict", "global_progress", "chi_square_statistic", "execution_status", "error_code",
                )}
                for candidate in metrics.get("candidates") or []
            ],
            "internal_wls_calls": metrics.get("internal_wls_calls"),
        })
    return rows


def first_parameter_correction(trace: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    for index, step in enumerate(trace):
        action = step.get("action") or {}
        if str(action.get("tool")) == CORRECT_PARAMETERS:
            observation = step.get("policy_observation") or {}
            parameter = ((observation.get("fresh_context_evidence") or {}).get("parameter") or {})
            ranked = parameter.get("parameter_ranking_distinct_lines") or []
            top = ranked[0].get("line_index1") if ranked and isinstance(ranked[0], Mapping) else None
            return {
                "step": index,
                "line_index": (action.get("arguments") or {}).get("line_index"),
                "context_top_line": top,
                "dominant": parameter.get("parameter_ranking_dominant"),
                "dominance_ratio": parameter.get("parameter_ranking_dominance_ratio"),
                "execution_status": step.get("execution_status"),
            }
    return None


def check_root(root: Mapping[str, Any], harness: _Harness) -> dict[str, Any]:
    from psse_env.rl.rollout import scenario_family, scenario_root
    from research.rl.hard_draw_report import root_stratum

    started = time.perf_counter()
    episode = harness.run(root, harness.expert_policy())
    trace = episode.get("trace") or []
    steps = policy_steps(trace)
    outcome = outcome_of(episode)
    outcome["reward"] = harness.reward(episode, scenario_family(root))
    true_lines = sorted(
        int(item["branch_row0"]) + 1
        for item in (((root.get("audit") or {}).get("truth") or {}).get("true_parameter_errors") or [])
        if isinstance(item, Mapping) and isinstance(item.get("branch_row0"), int)
    )
    return {
        "contract": CHECK_CONTRACT,
        "root": scenario_root(root),
        "family": scenario_family(root),
        "stratum": root_stratum(root),
        "true_lines": true_lines,
        "outcome": outcome,
        "comparisons": comparisons_of(steps),
        "first_parameter_correction": first_parameter_correction(steps),
        "actions": [copy.deepcopy(dict(step.get("action") or {})) for step in steps],
        "seconds": round(time.perf_counter() - started, 1),
    }


def _check_in_worker(root: Mapping[str, Any]) -> dict[str, Any]:
    assert _HARNESS is not None
    try:
        return check_root(root, _HARNESS)
    except Exception as exc:  # keep the sweep going; the row records the failure
        from psse_env.rl.rollout import scenario_root
        from research.rl.hard_draw_report import root_stratum

        return {"contract": CHECK_CONTRACT, "root": scenario_root(root), "stratum": root_stratum(root),
                "error": f"{type(exc).__name__}: {exc}", "outcome": None, "comparisons": []}


def load_rows(directory: Path) -> dict[str, dict[str, Any]]:
    path = directory / "roots.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {str(row["root"]): row for row in rows}


def summarize(rows: Mapping[str, Mapping[str, Any]], baseline: Mapping[str, Mapping[str, Any]] | None) -> dict[str, Any]:
    by_stratum: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "roots": 0, "solved": 0, "steps": 0, "comparisons": 0, "roots_with_comparison": 0,
        "comparison_changed_first_line": 0, "comparison_best_is_true": 0, "comparison_best_known": 0,
        "errors": 0, "baseline_solved": 0, "baseline_steps": 0, "paired": 0,
    })
    changes: list[dict[str, Any]] = []
    for root_id, row in sorted(rows.items()):
        entry = by_stratum[str(row.get("stratum") or "?")]
        entry["roots"] += 1
        if row.get("error") or not row.get("outcome"):
            entry["errors"] += 1
            continue
        outcome = row["outcome"]
        entry["solved"] += bool(outcome["success"])
        entry["steps"] += int(outcome["steps"])
        comparisons = row.get("comparisons") or []
        entry["comparisons"] += len(comparisons)
        entry["roots_with_comparison"] += bool(comparisons)
        first = row.get("first_parameter_correction")
        if comparisons and first and first.get("context_top_line") is not None and first.get("line_index") != first.get("context_top_line"):
            entry["comparison_changed_first_line"] += 1
        true_lines = set(row.get("true_lines") or [])
        for comparison in comparisons:
            if comparison.get("best_line_index") is not None and true_lines:
                entry["comparison_best_known"] += 1
                entry["comparison_best_is_true"] += int(comparison["best_line_index"] in true_lines)
        if baseline is not None and root_id in baseline and baseline[root_id].get("outcome"):
            base = baseline[root_id]["outcome"]
            entry["paired"] += 1
            entry["baseline_solved"] += bool(base["success"])
            entry["baseline_steps"] += int(base["steps"])
            if bool(base["success"]) != bool(outcome["success"]) or abs(int(base["steps"]) - int(outcome["steps"])) >= 2:
                changes.append({
                    "root": root_id, "stratum": row.get("stratum"), "true_lines": sorted(true_lines),
                    "baseline": {"success": base["success"], "steps": base["steps"], "healthy_preserved": base.get("healthy_preserved")},
                    "variant": {"success": outcome["success"], "steps": outcome["steps"], "healthy_preserved": outcome.get("healthy_preserved")},
                    "comparisons": [{k: c.get(k) for k in ("line_indices", "ranked_line_indices", "best_line_index")} for c in comparisons],
                    "first_parameter_correction": first,
                })
    totals = Counter()
    for entry in by_stratum.values():
        for key, value in entry.items():
            totals[key] += value
    return {"contract": CHECK_CONTRACT, "strata": dict(sorted(by_stratum.items())), "totals": dict(totals), "changes": changes}


def markdown(summary: Mapping[str, Any], *, variant: str, baseline_name: str | None) -> str:
    lines = [f"Variant `{variant}`" + (f" against baseline `{baseline_name}`" if baseline_name else "") + ".", ""]
    header = "| stratum | roots | solved | mean steps | roots compared | comparisons | best = true line | first line changed |"
    if baseline_name:
        header += " baseline solved | baseline mean steps |"
    lines.append(header)
    lines.append("| --- " * (header.count("|") - 1) + "|")
    for stratum, e in list(summary["strata"].items()) + [("all", summary["totals"])]:
        n = max(e["roots"] - e["errors"], 1)
        row = (f"| {stratum} | {e['roots']} | {e['solved']} | {e['steps'] / n:.1f} | {e['roots_with_comparison']} | {e['comparisons']} | "
               f"{e['comparison_best_is_true']}/{e['comparison_best_known']} | {e['comparison_changed_first_line']} |")
        if baseline_name:
            paired = max(e["paired"], 1)
            row += f" {e['baseline_solved']} | {e['baseline_steps'] / paired:.1f} |"
        lines.append(row)
    if summary.get("changes"):
        lines += ["", "Roots whose outcome or step count (by 2 or more) changed:", ""]
        for change in summary["changes"]:
            b, v = change["baseline"], change["variant"]
            lines.append(f"- `{change['root']}` ({change['stratum']}, true line {change['true_lines']}): baseline "
                         f"{'solved' if b['success'] else 'failed'} in {b['steps']} -> variant {'solved' if v['success'] else 'failed'} "
                         f"in {v['steps']}; comparisons {change['comparisons']}; first correction {change['first_parameter_correction']}")
    elif baseline_name:
        lines += ["", "No root changed outcome or step count by 2 or more."]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--summarize", default=None, help="an existing output directory to summarize (no rollouts)")
    parser.add_argument("--baseline", default=None, help="another check's output directory to pair against")
    parser.add_argument("--profile", default="suspicion_gated_diagnostics")
    parser.add_argument("--expert", default="ledger_ranked_compare",
                        choices=("baseline", "ledger", "ledger_ranked", "ledger_ranked_compare"))
    parser.add_argument("--triage-model", default=None)
    parser.add_argument("--progress-basis", default=None)
    parser.add_argument("--strata", nargs="*", default=list(PARAMETER_STRATA))
    parser.add_argument("--roots", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)

    if args.summarize:
        out = Path(args.summarize)
        variant = json.loads((out / "settings.json").read_text(encoding="utf-8")).get("expert_variant", "?") if (out / "settings.json").exists() else "?"
    else:
        if not args.suite or not args.output_dir:
            parser.error("--suite and --output-dir are required unless --summarize is given")
        suite = json.loads(Path(args.suite).read_text(encoding="utf-8"))
        if isinstance(suite, Mapping):
            suite = suite.get("scenarios") or suite.get("roots") or []
        roots = select_roots(suite, strata=args.strata, per_stratum=None, tags=None, prefer_tag=None,
                             limit=args.limit, roots=args.roots)
        settings = dict(profile=args.profile, expert_variant=args.expert, triage_model=args.triage_model,
                        progress_basis=args.progress_basis, max_steps=args.max_steps, seed=args.seed, reward=None)
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "settings.json").write_text(json.dumps({**settings, "suite": args.suite, "strata": args.strata}, indent=2), encoding="utf-8")
        variant = args.expert
        print(f"[check] {len(roots)} roots from {args.suite}; profile {args.profile}, expert {args.expert}, workers {args.workers}", flush=True)
        started = time.perf_counter()
        results: list[dict[str, Any]] = []
        with (out / "roots.jsonl").open("w", encoding="utf-8") as stream:
            if args.workers > 1:
                with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker, initargs=(settings,)) as pool:
                    iterator = pool.map(_check_in_worker, roots, chunksize=1)
                    for result in iterator:
                        results.append(result)
                        stream.write(json.dumps(result, default=str) + "\n")
                        stream.flush()
                        _progress(result, len(results), started)
            else:
                harness = _Harness(**settings)
                for root in roots:
                    result = check_root(root, harness)
                    results.append(result)
                    stream.write(json.dumps(result, default=str) + "\n")
                    stream.flush()
                    _progress(result, len(results), started)
    rows = load_rows(out)
    baseline = load_rows(Path(args.baseline)) if args.baseline else None
    summary = summarize(rows, baseline)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    text = markdown(summary, variant=variant, baseline_name=args.baseline)
    (out / "summary.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


def _progress(result: Mapping[str, Any], done: int, started: float) -> None:
    outcome = result.get("outcome") or {}
    print(json.dumps({
        "root": result.get("root"), "stratum": result.get("stratum"), "success": outcome.get("success"),
        "steps": outcome.get("steps"), "comparisons": len(result.get("comparisons") or []),
        "error": result.get("error"), "done": done, "elapsed": round(time.perf_counter() - started, 1),
    }), flush=True)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
