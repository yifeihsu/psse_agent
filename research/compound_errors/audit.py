"""Decision audit of a compound suite: every failed root and a sample of solved roots per class.

    python -m research.compound_errors.audit --suite output/compound_errors_20261027/suite.json \\
        --episodes output/compound_errors_20261027/expert_baseline_compare/episodes.jsonl \\
        --output-dir output/compound_errors_20261027/audit --profile scada_only --expert baseline_compare \\
        --solved-per-class 10 --workers 14

Runs ``research.rl.decision_audit.audit_root`` (the reference trajectory, up
to ``--max-states`` decision states with ``--max-alternatives`` menu
alternatives each, and the truth-branch probe that applies every true
correction, breakers by name) on the roots the rollout failed and on the
first ``--solved-per-class`` solved roots of each class, in parallel.  The
summary bins the roots as the plan defines (rule_best, legal_alternative,
interface_limit, evidence_limit) per class and keeps, for every truth probe,
the audit reasons and the verdicts of the applied corrections, which is what
separates an interface limit from an evidence limit in the study note.
``--system`` and the progress basis follow ``research.compound_errors.rollout``.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.rl import decision_audit as audit  # noqa: E402

AUDIT_CONTRACT = "compound_decision_audit_v1"


def class_of(root: Mapping[str, Any]) -> str:
    return str(((root.get("audit") or {}).get("compound") or {}).get("class"))


def select_roots(suite: Sequence[Mapping[str, Any]], episodes: Mapping[str, Mapping[str, Any]], *,
                 solved_per_class: int, failed_per_class: int | None, seed: int,
                 classes: Sequence[str] | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Every failed root (capped per class when asked) and a seeded sample of solved roots per class."""
    by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for root in suite:
        name = class_of(root)
        if classes and name not in set(classes):
            continue
        by_class[name].append(root)
    chosen: list[dict[str, Any]] = []
    report: dict[str, Any] = {}
    for name, roots in sorted(by_class.items()):
        failed = [r for r in roots if not bool((episodes.get(str(r["execution"]["scenario_id"])) or {}).get("success"))]
        solved = [r for r in roots if r not in failed]
        rng = random.Random(f"{seed}:{name}")
        rng.shuffle(failed)
        rng.shuffle(solved)
        take_failed = failed if failed_per_class is None else failed[:failed_per_class]
        take_solved = solved[:solved_per_class]
        chosen.extend(take_failed)
        chosen.extend(take_solved)
        report[name] = {"roots": len(roots), "failed": len(failed), "audited_failed": len(take_failed), "audited_solved": len(take_solved)}
    return chosen, report


def truth_probe_summary(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Per class: how the truth-branch probe fared and why it fell short."""
    out: dict[str, dict[str, Any]] = {}
    for result in results:
        name = str(result.get("stratum") or "").removeprefix("compound:")
        entry = out.setdefault(name, {"probes": 0, "solved": 0, "reasons": Counter(), "refused": Counter(), "rejected_true": Counter(),
                                      "roots": []})
        for state in result.get("decision_states") or []:
            truth = state.get("truth")
            if not isinstance(truth, Mapping):
                continue
            entry["probes"] += 1
            success = bool((truth.get("outcome") or {}).get("success"))
            entry["solved"] += int(success)
            for reason in truth.get("audit_reasons") or []:
                entry["reasons"][reason] += 1
            for correction in truth.get("corrections") or []:
                if correction.get("execution_status") != "success":
                    entry["refused"][f"{correction['tool']}:{correction.get('error_code')}"] += 1
                elif correction.get("verdict") == "REJECT":
                    entry["rejected_true"][str(correction["tool"])] += 1
            entry["roots"].append({"root": result.get("root"), "reference_success": bool((result.get("reference") or {}).get("success")),
                                   "truth_success": success, "bin": result.get("bin"), "unapplied": len(truth.get("unapplied") or []),
                                   "reasons": truth.get("audit_reasons") or [],
                                   "corrections": [(c["tool"], c.get("execution_status"), c.get("error_code"), c.get("verdict"))
                                                   for c in truth.get("corrections") or []]})
            break  # one truth probe per root
    for entry in out.values():
        entry["reasons"] = dict(entry["reasons"].most_common())
        entry["refused"] = dict(entry["refused"].most_common())
        entry["rejected_true"] = dict(entry["rejected_true"].most_common())
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--episodes", required=True, type=Path, help="the rollout's episodes.jsonl (which roots the expert failed)")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--profile", default="scada_only")
    parser.add_argument("--expert", default="baseline_compare")
    parser.add_argument("--progress-basis", default=None)
    parser.add_argument("--solved-per-class", type=int, default=10)
    parser.add_argument("--failed-per-class", type=int, default=None, help="cap on audited failed roots per class (default: all)")
    parser.add_argument("--classes", nargs="*", default=None)
    parser.add_argument("--max-states", type=int, default=4)
    parser.add_argument("--max-alternatives", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--sample-seed", type=int, default=20261027)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--system", default=None, help="the suite's registered system (default: read from the roots)")
    parser.add_argument("--env-option", action="append", default=None, metavar="KEY=VALUE",
                        help="research environment option of the rollout, e.g. pair_refit_adjacent=1")
    args = parser.parse_args(argv)

    from research.compound_errors.rollout import suite_system, system_progress_basis

    suite = json.loads(args.suite.read_text(encoding="utf-8"))
    if isinstance(suite, Mapping):
        suite = suite.get("scenarios") or suite.get("roots") or []
    system = suite_system(suite, args.system)
    args.progress_basis = system_progress_basis(system, args.progress_basis)
    episodes = {}
    for line in args.episodes.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            episodes[str(record["scenario_id"])] = record
    roots, selection = select_roots(suite, episodes, solved_per_class=args.solved_per_class, failed_per_class=args.failed_per_class,
                                    seed=args.sample_seed, classes=args.classes)
    settings = dict(profile=args.profile, expert_variant=args.expert, triage_model=None, progress_basis=args.progress_basis,
                    max_steps=args.max_steps, seed=args.seed, reward=None,
                    env_options=audit.parse_run_options(args.env_option))
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    print(f"[audit] {len(roots)} roots ({sum(v['audited_failed'] for v in selection.values())} failed, "
          f"{sum(v['audited_solved'] for v in selection.values())} solved) under {args.profile} ({args.expert}); "
          f"{args.max_states} states x {args.max_alternatives} alternatives, workers {args.workers}", flush=True)
    started = time.perf_counter()
    payloads = [{"root": root, "max_states": args.max_states, "max_alternatives": args.max_alternatives, "with_truth": True} for root in roots]
    results: list[dict[str, Any]] = []
    stream = (out / "roots.jsonl").open("w", encoding="utf-8")

    def emit(result: Mapping[str, Any]) -> None:
        results.append(dict(result))
        stream.write(json.dumps(result, sort_keys=True, default=str) + "\n")
        stream.flush()
        print(json.dumps({"root": result.get("root"), "stratum": result.get("stratum"), "bin": result.get("bin"),
                          "reference_success": (result.get("reference") or {}).get("success"),
                          "states": len(result.get("decision_states") or []), "episodes": result.get("episodes_run"),
                          "seconds": result.get("seconds"), "error": result.get("error"), "done": len(results),
                          "elapsed": round(time.perf_counter() - started, 1)}), flush=True)

    try:
        if args.workers > 1:
            with ProcessPoolExecutor(max_workers=args.workers, initializer=audit._init_worker, initargs=(settings,)) as pool:
                for result in pool.map(audit._audit_root_in_worker, payloads, chunksize=1):
                    emit(result)
        else:
            audit._init_worker(settings)
            for payload in payloads:
                emit(audit._audit_root_in_worker(payload))
    finally:
        stream.close()
    summary = audit.summarize(results)
    summary.update(contract=AUDIT_CONTRACT, suite=str(args.suite), episodes=str(args.episodes), system=system,
                   progress_basis=args.progress_basis, env_options=settings["env_options"], profile=args.profile,
                   expert=args.expert, max_states=args.max_states, max_alternatives=args.max_alternatives,
                   max_steps=args.max_steps, seed=args.seed, selection=selection,
                   truth_probe=truth_probe_summary(results), total_seconds=round(time.perf_counter() - started, 1))
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    (out / "summary.md").write_text(audit.markdown(summary) + "\n", encoding="utf-8")
    with (out / "states.jsonl").open("w", encoding="utf-8") as handle:
        for result in results:
            for state in result.get("decision_states") or []:
                handle.write(json.dumps({"root": result["root"], "stratum": result.get("stratum"), "reference": result.get("reference"),
                                         **state}, sort_keys=True, default=str) + "\n")
    print(audit.markdown(summary), flush=True)
    print(json.dumps({"bins": summary["bins"], "total_seconds": summary["total_seconds"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
