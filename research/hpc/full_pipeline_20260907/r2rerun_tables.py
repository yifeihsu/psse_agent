#!/usr/bin/env python3
"""Round-2 rerun of the conventional-only cell (2026-10-29; note section 9).

    r2rerun_tables.py --cell-dir PIPE [--new-suffix ""] [--old-suffix _pool_20261023]
                      [--traces N] [--output OUT.json]

Reads the rerun's R2 (``out/r2`` and the evaluations ``submit_r2_rerun_20261029.sh``
writes) beside the recorded pool-recipe R2 (the same records set aside under
``--old-suffix``) and the expert, BC0 and R1 records, which are not re-run:

* every set and horizon: the first draw at 12, 16, 20 and 40 actions, the
  second draw at 12, 20 and 40, the two draws pooled at 12, 20 and 40, the
  validation and hard-draw roots at 40.  Per policy: successes, safety-clean
  successes, violation episodes (invalid action / false commit / loop),
  episodes at the budget, mean actions and WLS calls, per family, per error
  cohort (the multi-error roots and the single-error and clean roots, with
  mean actions) and per parameter-ranking stratum (with the comparisons used);
* exact McNemar tests (``followup_tables.mcnemar``) of the expert, BC0 and R1
  against the new R2 and of the new R2 against the recorded one, and Holm's
  correction over the 18 pre-stated pooled tests of section 8.6 (BC0 vs R1,
  BC0 vs expert, R1 vs R2 on success and safety-clean success at 12, 20 and
  40 actions) and over the 18 tests of the expert, BC0 and R1 against R2;
* the new R2's failures and violation episodes with the other policies'
  outcome on the same root, and the first ``--traces`` failures' compact
  traces (tool, arguments, refusal, verifier disposition, J and the largest
  normalized residual after each estimator call);
* r2e's re-score of R1 against R1's recorded evaluation, root by root;
* the training pass: mixture report, collection and training receipts, the
  trainer's log and the GPU of every training attempt (``logs/gpu-train-r2-*``).

``--new-suffix`` and ``--old-suffix`` name the two R2 record sets; a dry run
over the recorded records alone passes the same suffix twice.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

HERE = Path(__file__).resolve().parent
POLICIES = ("expert", "bc0", "r1", "r2", "r2_pool")
MULTI_ERROR = ("multi_measurement", "measurement+parameter", "measurement+topology")
FAMILIES = ("measurement", "multi_measurement", "parameter", "topology", "measurement+parameter",
            "measurement+topology", "no_error")
PRESTATED = (("bc0", "r1"), ("bc0", "expert"), ("r1", "r2"))
AGAINST_R2 = (("expert", "r2"), ("bc0", "r2"), ("r1", "r2"))
FIELDS = ("success", "invalid", "false_commit", "loop", "violation", "clean", "terminal_outcome", "steps",
          "wls_calls", "comparisons")


def _load(name: str):
    spec = importlib.util.spec_from_file_location(Path(name).stem, HERE / name)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


tables = _load("followup_tables.py")


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sources(cell: Path, new: str, old: str) -> dict[str, dict[str, Any]]:
    """set name -> rows, horizon and policy -> evaluation file."""
    out = cell / "out"
    ev = out / "extra" / "evaluations"
    test1 = out / "r1" / "collection" / "development_scenarios.json"
    test2 = out / "extra" / "test_draw2" / "suite" / "development.json"
    r2_new, r2_old = out / f"r2{new}", out / f"r2{old}"
    result: dict[str, dict[str, Any]] = {
        "draw1@40": {"rows": test1, "horizon": 40, "files": {
            "expert": out / "r1" / "collection" / "evaluation" / "expert_eval.json",
            "bc0": out / "r1" / "collection" / "evaluation" / "bc0_eval.json",
            "r1": out / "r1" / "collection" / "evaluation" / "r1_eval.json",
            "r2": r2_new / "collection" / "evaluation" / "r1_eval.json",
            "r2_pool": r2_old / "collection" / "evaluation" / "r1_eval.json"}},
        "draw2@40": {"rows": test2, "horizon": 40, "files": {
            "expert": ev / "test2_40" / "expert_eval.json", "bc0": ev / "test2_40" / "bc0_eval.json",
            "r1": ev / "test2_40" / "r1_eval.json", "r2": ev / f"test2_40_r2{new}" / "bc0_eval.json",
            "r2_pool": ev / f"test2_40_r2{old}" / "bc0_eval.json"}},
    }
    for draw, rows, prefix, budgets in (("draw1", test1, "test_budget", (12, 16, 20)), ("draw2", test2, "test2_budget", (12, 20))):
        for b in budgets:
            result[f"{draw}@{b}"] = {"rows": rows, "horizon": b, "files": {
                "expert": ev / f"{prefix}{b}" / "expert_eval.json", "bc0": ev / f"{prefix}{b}" / "bc0_eval.json",
                "r1": ev / f"{prefix}{b}" / "r1_eval.json", "r2": ev / f"{prefix}{b}_r2{new}" / "bc0_eval.json",
                "r2_pool": ev / f"{prefix}{b}_r2{old}" / "bc0_eval.json"}}
    for suite in ("validation", "hard"):
        result[f"{suite}@40"] = {"rows": out / "extra" / "suites" / f"{suite}.json", "horizon": 40, "files": {
            "expert": ev / f"{suite}_bc0_r1" / "expert_eval.json", "bc0": ev / f"{suite}_bc0_r1" / "bc0_eval.json",
            "r1": ev / f"{suite}_bc0_r1" / "r1_eval.json", "r2": ev / f"{suite}_r2{new}" / "bc0_eval.json",
            "r2_pool": ev / f"{suite}_r2_control{old}" / "bc0_eval.json"}}
    return result


def load(cell: Path, new: str, old: str) -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
    """set -> policy -> root -> record (followup_tables.records, with the scenario id)."""
    loaded: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for name, spec in sources(cell, new, old).items():
        rows = _read(spec["rows"])
        roots = tables.describe_roots(rows)
        scenario = {tables._root(row): str((row.get("grouping") or {}).get("root_scenario_id") or "") for row in rows}
        loaded[name] = {}
        for policy, path in spec["files"].items():
            if path.is_file():
                records = tables.records(path, roots)
                for root, item in records.items():
                    item["scenario_id"] = scenario.get(root, "")
                loaded[name][policy] = records
    for horizon in (12, 20, 40):
        first, second = loaded.get(f"draw1@{horizon}", {}), loaded.get(f"draw2@{horizon}", {})
        pooled = {}
        for policy in POLICIES:
            if policy in first and policy in second:
                pooled[policy] = {**first[policy], **second[policy]}
        loaded[f"pooled@{horizon}"] = pooled
    return loaded


def summary(records: Mapping[str, Mapping[str, Any]], horizon: int) -> dict[str, Any]:
    n = len(records)
    items = list(records.values())
    def block(selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        k = len(selected)
        return {"episodes": k, "successes": sum(r["success"] for r in selected),
                "clean_successes": sum(r["clean"] for r in selected),
                "violation_episodes": sum(r["violation"] for r in selected),
                "invalid": sum(r["invalid"] for r in selected), "false_commit": sum(r["false_commit"] for r in selected),
                "loop": sum(r["loop"] for r in selected),
                "mean_actions": round(sum(r["steps"] for r in selected) / k, 3) if k else None,
                "mean_wls_calls": round(sum(r["wls_calls"] for r in selected) / k, 3) if k else None,
                "comparisons": sum(r["comparisons"] for r in selected)}
    result = block(items)
    result["at_budget"] = sum(r["steps"] >= horizon for r in items)
    result["by_family"] = {f: block([r for r in items if r.get("family") == f]) for f in FAMILIES
                           if any(r.get("family") == f for r in items)}
    result["by_cohort"] = {"multi_error": block([r for r in items if r.get("family") in MULTI_ERROR]),
                           "single_or_clean": block([r for r in items if r.get("family") not in MULTI_ERROR])}
    strata = defaultdict(list)
    for r in items:
        if r.get("family") in ("parameter", "measurement+parameter"):
            strata[f"{r['family']}:{r.get('stratum')}"].append(r)
            strata[f"all:{r.get('stratum')}"].append(r)
    result["by_stratum"] = {k: block(v) for k, v in sorted(strata.items())}
    result["episodes"] = n
    return result


def paired(a: Mapping[str, Mapping[str, Any]], b: Mapping[str, Mapping[str, Any]], field: str) -> dict[str, Any]:
    common = sorted(set(a) & set(b))
    only_a = sum(1 for r in common if not a[r][field] and b[r][field])
    only_b = sum(1 for r in common if a[r][field] and not b[r][field])
    return {"paired_roots": len(common), "fails_only_first": only_a, "fails_only_second": only_b,
            "p": tables.mcnemar(only_a, only_b)}


def holm(tests: Mapping[str, float], alpha: float = 0.05) -> dict[str, Any]:
    ordered = sorted(tests.items(), key=lambda kv: kv[1])
    m = len(ordered)
    result: dict[str, Any] = {}
    running = 0.0
    stopped = False
    for rank, (name, p) in enumerate(ordered, start=1):
        running = max(running, min(1.0, (m - rank + 1) * p))
        rejected = not stopped and p <= alpha / (m - rank + 1)
        stopped = stopped or not rejected
        result[name] = {"p": p, "rank": rank, "threshold": alpha / (m - rank + 1), "holm_p": running,
                        "rejected": rejected}
    return result


def failures(loaded: Mapping[str, Any], set_name: str) -> list[dict[str, Any]]:
    block = loaded.get(set_name, {})
    new = block.get("r2", {})
    rows = []
    for root, item in sorted(new.items(), key=lambda kv: (kv[1].get("family") or "", kv[1].get("scenario_id") or "")):
        if item["success"] and not item["violation"]:
            continue
        others = {p: (block[p][root]["success"], block[p][root]["violation"], block[p][root]["steps"])
                  for p in ("expert", "bc0", "r1", "r2_pool") if p in block and root in block[p]}
        rows.append({"root": root, "scenario_id": item.get("scenario_id"), "family": item.get("family"),
                     "stratum": item.get("stratum"), "success": item["success"], "invalid": item["invalid"],
                     "false_commit": item["false_commit"], "loop": item["loop"], "steps": item["steps"],
                     "terminal_outcome": item.get("terminal_outcome"), "others": others})
    return rows


def compact_traces(path: Path, scenario_ids: Iterable[str], limit: int) -> dict[str, list[str]]:
    wanted = list(dict.fromkeys(scenario_ids))[:limit]
    if not wanted or not path.is_file():
        return {}
    episodes = {str(e.get("scenario_id")): e for e in tables._episodes(_read(path)) if str(e.get("scenario_id")) in wanted}
    result = {}
    for sid in wanted:
        episode = episodes.get(sid)
        if episode is None:
            continue
        audit = (episode.get("audit") or {}).get("accepted_target_audit") or {}
        lines = [f"true {json.dumps(audit.get('true_targets'))} accepted {json.dumps(audit.get('accepted_targets'))}"]
        for step in episode.get("trace") or []:
            action = step.get("action") or {}
            arguments = dict(action.get("arguments") or {})
            arguments.pop("state_id", None)
            output = step.get("policy_tool_output")
            metrics = output.get("tool_metrics", {}) if isinstance(output, Mapping) else {}
            extra = ""
            if action.get("tool") in ("run_wls", "verify_candidate"):
                extra = f" J={metrics.get('chi_square_statistic')} rN={metrics.get('max_normalized_residual')}"
            lines.append(f"{step.get('step')} {action.get('tool')} {json.dumps(arguments)[:140]} "
                         f"{step.get('execution_status')} err={step.get('error_code')} "
                         f"disp={step.get('candidate_disposition_offline')}{extra}")
        result[sid] = lines
    return result


def rescore(cell: Path, new: str, old: str) -> dict[str, Any]:
    """r2e's re-score of R1 (its first slot) against R1's recorded round-1 evaluation, root by root."""
    out = cell / "out"
    roots = tables.describe_roots(_read(out / "r1" / "collection" / "development_scenarios.json"))
    recorded = tables.records(out / "r1" / "collection" / "evaluation" / "r1_eval.json", roots)
    result = {}
    for label, suffix in (("rerun_r2e", new), ("recorded_r2e", old)):
        path = out / f"r2{suffix}" / "collection" / "evaluation" / "bc0_eval.json"
        if not path.is_file():
            continue
        rescored = tables.records(path, roots)
        common = sorted(set(recorded) & set(rescored))
        differ = [r for r in common if any(recorded[r][f] != rescored[r][f] for f in FIELDS)]
        result[label] = {"paired": len(common), "differing_roots": len(differ),
                         "successes": [sum(recorded[r]["success"] for r in common), sum(rescored[r]["success"] for r in common)],
                         "differences": [{"root": r, "family": recorded[r].get("family"),
                                          "recorded": {f: recorded[r][f] for f in FIELDS},
                                          "rescored": {f: rescored[r][f] for f in FIELDS}} for r in differ[:12]]}
    return result


def training(cell: Path, new: str) -> dict[str, Any]:
    round_dir = cell / "out" / f"r2{new}"
    result: dict[str, Any] = {}
    for name in ("collection.done", "training.done", "collection/mixture_report.json", "collection/collection_metrics.json",
                 "training/research_run.json"):
        path = round_dir / name
        if path.is_file():
            payload = _read(path)
            if name == "collection/mixture_report.json":
                conflicts = payload.get("label_conflicts") or {}
                payload = {**{k: v for k, v in payload.items() if k != "label_conflicts"},
                           "label_conflicts": {k: v for k, v in conflicts.items() if k != "conflicts"},
                           "label_conflict_examples": (conflicts.get("conflicts") or [])[:4]}
            elif isinstance(payload, Mapping):
                payload = {k: v for k, v in payload.items() if not isinstance(v, (list, dict)) or k in (
                    "training_families", "mixture_label_conflicts", "prior_onpolicy_rounds", "label_yield")}
            result[name] = payload
    states = sorted(round_dir.glob("training/**/trainer_state.json"), key=lambda p: p.stat().st_mtime)
    if states:
        state = _read(states[-1])
        history = state.get("log_history") or []
        result["trainer_state"] = {"file": str(states[-1].relative_to(cell)), "global_step": state.get("global_step"),
                                   "max_steps": state.get("max_steps"),
                                   "first": history[:2], "last": history[-4:],
                                   "eval": [h for h in history if "eval_loss" in h][-6:]}
    gpus = {}
    for path in sorted((cell / "logs").glob("gpu-train-r2-*.csv")):
        gpus[path.name] = path.read_text(encoding="utf-8").strip()
    result["gpu_train_logs"] = gpus
    return result


def markdown(result: Mapping[str, Any]) -> str:
    lines = []
    for name, block in result["sets"].items():
        lines.append(f"### {name}: successes (safety-clean); violations (invalid/false commit/loop); mean actions; at budget")
        for policy, s in block["summary"].items():
            lines.append(f"- {policy}: {s['successes']}/{s['episodes']} ({s['clean_successes']}); {s['violation_episodes']} "
                         f"({s['invalid']}/{s['false_commit']}/{s['loop']}); {s['mean_actions']}; {s['at_budget']}; "
                         "cohorts " + "; ".join(f"{k} {v['successes']}/{v['episodes']} ({v['clean_successes']}) {v['mean_actions']}"
                                               for k, v in s["by_cohort"].items()))
        for test, value in block["tests"].items():
            lines.append(f"  - {test}: {value['fails_only_first']}/{value['fails_only_second']} (p = {value['p']:.3g})")
    for family, entries in result["holm"].items():
        lines.append(f"### Holm over {family}")
        for name, value in sorted(entries.items(), key=lambda kv: kv[1]["rank"]):
            lines.append(f"- {name}: p = {value['p']:.3g}, Holm p = {value['holm_p']:.3g}, rejected {value['rejected']}")
    return "\n".join(lines) + "\n"


def build(cell: Path, new: str, old: str, traces: int) -> dict[str, Any]:
    loaded = load(cell, new, old)
    result: dict[str, Any] = {"sets": {}, "holm": {}, "failures": {}, "traces": {}}
    for name, block in loaded.items():
        horizon = int(name.split("@")[1])
        tests = {}
        for a, b in PRESTATED + AGAINST_R2[:2] + (("r2_pool", "r2"),):
            if a in block and b in block:
                for field in ("success", "clean", "violation"):
                    if field == "violation":
                        tests[f"{a}_vs_{b}:{field}"] = paired({r: {"v": not x["violation"]} for r, x in block[a].items()},
                                                             {r: {"v": not x["violation"]} for r, x in block[b].items()}, "v")
                    else:
                        tests[f"{a}_vs_{b}:{field}"] = paired(block[a], block[b], field)
        result["sets"][name] = {"summary": {p: summary(block[p], horizon) for p in POLICIES if p in block}, "tests": tests}
    for family, pairs in (("prestated_18", PRESTATED), ("against_r2_18", AGAINST_R2)):
        family_tests = {}
        for horizon in (12, 20, 40):
            tests = result["sets"].get(f"pooled@{horizon}", {}).get("tests", {})
            for a, b in pairs:
                for field in ("success", "clean"):
                    key = f"{a}_vs_{b}:{field}"
                    if key in tests:
                        family_tests[f"{key}@{horizon}"] = tests[key]["p"]
        if family_tests:
            result["holm"][family] = holm(family_tests)
    files = sources(cell, new, old)
    for name in ("draw1@40", "draw2@40", "validation@40", "hard@40", "draw1@12", "draw1@16", "draw1@20", "draw2@12", "draw2@20"):
        rows = failures(loaded, name)
        result["failures"][name] = rows
        if name.endswith("@40") and traces:
            result["traces"][name] = compact_traces(files[name]["files"]["r2"], [r["scenario_id"] for r in rows if not r["success"]] +
                                                    [r["scenario_id"] for r in rows if r["success"]], traces)
    result["r1_rescore"] = rescore(cell, new, old)
    result["training"] = training(cell, new)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cell-dir", required=True, type=Path)
    parser.add_argument("--new-suffix", default="")
    parser.add_argument("--old-suffix", default="_pool_20261023")
    parser.add_argument("--traces", type=int, default=12)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    result = build(args.cell_dir, args.new_suffix, args.old_suffix, args.traces)
    text = markdown(result)
    print(text, end="")
    if args.output is not None:
        args.output.write_text(json.dumps(result, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
        args.output.with_suffix(".md").write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
