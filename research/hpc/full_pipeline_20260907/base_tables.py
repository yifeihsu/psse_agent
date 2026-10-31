#!/usr/bin/env python3
"""The prompted, untuned base model on the conventional-only cell's test draws (2026-10-31; note section 11).

    base_tables.py --cell-dir PIPE [--label base_prompted] [--traces N] [--output OUT.json]

Reads the base model's seven evaluations (``submit_base_prompted_20261031.sh``:
the first draw at 12, 16, 20 and 40 actions, the second at 12, 20 and 40)
beside the expert, BC0, R1, Table I's R2 (section 9's, set aside under
``_bc0init_20261030``) and the R1-start R2 (section 10), none of them re-run:

* per set, and pooled over the two draws at 12, 20 and 40 actions: successes,
  safety-clean successes, violation episodes (invalid action / false commit /
  loop), episodes at the budget, mean actions, per family and error cohort
  (``r2rerun_tables.summary``);
* exact McNemar tests (``followup_tables.mcnemar``) of BC0, the expert, R1 and
  R2 against the base model on success, safety-clean success and violation
  episodes; none of them is among the 18 pre-stated tests of section 8.6;
* how the base model's episodes go, per set: terminal outcomes, how they end
  (a terminal state, the action budget, an evaluator circuit breaker),
  refused or invalid actions by tool and error code, the outputs that did not
  parse into one registered tool call (by message, quoted names and numbers
  masked), the tools it calls, its first action, and the compact traces of its
  first ``--traces`` failed or violating episodes at 40 actions;
* one record per set, policy and root (``<output>_episodes.jsonl``).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

HERE = Path(__file__).resolve().parent
TEST_SETS = ("draw1@40", "draw2@40", "draw1@12", "draw1@16", "draw1@20", "draw2@12", "draw2@20")
POLICIES = ("expert", "bc0", "r1", "r2", "r2_r1start", "base")
#: The base model is compared with these (Table I's columns); r2 is section 9's R2.
AGAINST = ("bc0", "expert", "r1", "r2")
CIRCUIT_BREAKERS = ("evaluation_repeated_nonadvancing_failure", "evaluation_repeated_nonadvancing_diagnostic",
                    "evaluation_specialized_tool_budget_exhausted")
INVALID_ACTION = "__invalid_action__"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(Path(name).stem, HERE / name)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


rerun = _load("r2rerun_tables.py")
tables = rerun.tables


def base_files(cell: Path, label: str) -> dict[str, Path]:
    """Set name -> the base model's evaluation file (its one policy sits in the evaluator's first slot)."""
    ev = cell / "out" / "extra" / "evaluations"
    names = {"draw1@40": f"test_40_{label}", "draw2@40": f"test2_40_{label}"}
    names.update({f"draw1@{b}": f"test_budget{b}_{label}" for b in (12, 16, 20)})
    names.update({f"draw2@{b}": f"test2_budget{b}_{label}" for b in (12, 20)})
    return {name: ev / directory / "bc0_eval.json" for name, directory in names.items()}


def load(cell: Path, label: str) -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
    """set -> policy -> root -> record, for the test sets and the pooled draws."""
    specs = rerun.sources(cell, "", "_pool_20261023", "_bc0init_20261030")
    base = base_files(cell, label)
    loaded: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for name in TEST_SETS:
        spec = specs[name]
        rows = rerun._read(spec["rows"])
        roots = tables.describe_roots(rows)
        scenario = {tables._root(row): str((row.get("grouping") or {}).get("root_scenario_id") or "") for row in rows}
        files = {"expert": spec["files"]["expert"], "bc0": spec["files"]["bc0"], "r1": spec["files"]["r1"],
                 "r2": spec["files"]["r2_bc0init"], "r2_r1start": spec["files"]["r2"], "base": base[name]}
        loaded[name] = {}
        for policy, path in files.items():
            if path.is_file():
                records = tables.records(path, roots)
                for root, item in records.items():
                    item["scenario_id"] = scenario.get(root, "")
                loaded[name][policy] = records
    for horizon in (12, 20, 40):
        first, second = loaded[f"draw1@{horizon}"], loaded[f"draw2@{horizon}"]
        loaded[f"pooled@{horizon}"] = {p: {**first[p], **second[p]} for p in POLICIES if p in first and p in second}
    return loaded


def against_base(block: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """BC0, the expert, R1 and R2 against the base model: roots only the first / only the base fails."""
    tests: dict[str, Any] = {}
    if "base" not in block:
        return tests
    for other in AGAINST:
        if other not in block:
            continue
        for field in ("success", "clean"):
            tests[f"{other}_vs_base:{field}"] = rerun.paired(block[other], block["base"], field)
        tests[f"{other}_vs_base:violation"] = rerun.paired({r: {"v": not x["violation"]} for r, x in block[other].items()},
                                                          {r: {"v": not x["violation"]} for r, x in block["base"].items()}, "v")
    return tests


def mask(text: str) -> str:
    """A refusal or parse message with its quoted names and numbers masked, so equal kinds group."""
    text = re.sub(r"'[^']*'|\"[^\"]*\"", "<q>", str(text))
    text = re.sub(r"\d+(\.\d+)?", "<n>", text)
    return text[:160]


def episode_modes(path: Path, horizon: int, traces: int) -> dict[str, Any]:
    """How the base model's episodes go: endings, refusals, unparsed outputs, tools, first actions."""
    episodes = tables._episodes(rerun._read(path))
    endings, outcomes, refusals, unparsed, tools, first, failed_ids = (Counter() for _ in range(7))
    examples: dict[str, str] = {}
    for episode in episodes:
        trace = episode.get("trace") or []
        last = trace[-1] if trace else {}
        if last.get("error_code") in CIRCUIT_BREAKERS:
            endings[f"circuit breaker {last.get('error_code')}"] += 1
        elif episode.get("terminal"):
            endings["terminal state"] += 1
        elif len(trace) >= horizon:
            endings["action budget"] += 1
        else:
            endings["other"] += 1
        outcomes[str(episode.get("terminal_outcome"))] += 1
        if trace:
            first[str((trace[0].get("action") or {}).get("tool"))] += 1
        for step in trace:
            action = step.get("action") or {}
            tool = str(action.get("tool"))
            tools[tool] += 1
            if step.get("execution_status") != "success":
                refusals[f"{tool}: {step.get('error_code')}"] += 1
            if tool == INVALID_ACTION:
                detail = str((action.get("arguments") or {}).get("error_detail") or "")
                kind = mask(detail)
                unparsed[kind] += 1
                examples.setdefault(kind, detail[:300])
        if not episode.get("truth_audited_task_success") or int(episode.get("invalid_action_count") or 0) \
                or int(episode.get("false_commit_count") or 0) or episode.get("loop_detected"):
            failed_ids[str(episode.get("scenario_id"))] += 1
    result = {
        "episodes": len(episodes),
        "endings": dict(endings.most_common()),
        "terminal_outcomes": dict(outcomes.most_common()),
        "first_action": dict(first.most_common()),
        "tool_calls": dict(tools.most_common()),
        "refused_or_invalid_steps": dict(refusals.most_common()),
        "unparsed_outputs": dict(unparsed.most_common()),
        "unparsed_examples": examples,
        "steps": sum(len(e.get("trace") or []) for e in episodes),
    }
    if traces:
        result["traces"] = rerun.compact_traces(path, list(failed_ids), traces)
    return result


def build(cell: Path, label: str, traces: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    loaded = load(cell, label)
    result: dict[str, Any] = {"sets": {}, "modes": {}, "label": label}
    for name, block in loaded.items():
        horizon = int(name.split("@")[1])
        result["sets"][name] = {"summary": {p: rerun.summary(block[p], horizon) for p in POLICIES if p in block},
                                "tests": against_base(block)}
    for name, path in base_files(cell, label).items():
        if path.is_file():
            horizon = int(name.split("@")[1])
            result["modes"][name] = episode_modes(path, horizon, traces if horizon == 40 else 0)
    records = [{"set": name, "policy": policy, "root": root, **item}
               for name in TEST_SETS for policy, block in loaded[name].items() for root, item in sorted(block.items())]
    return result, records


def markdown(result: Mapping[str, Any]) -> str:
    lines = []
    for name, block in result["sets"].items():
        lines.append(f"### {name}: successes (safety-clean); violations (invalid/false commit/loop); mean actions; at budget")
        for policy, s in block["summary"].items():
            lines.append(f"- {policy}: {s['successes']}/{s['episodes']} ({s['clean_successes']}); {s['violation_episodes']} "
                         f"({s['invalid']}/{s['false_commit']}/{s['loop']}); {s['mean_actions']}; {s['at_budget']}; "
                         "cohorts " + "; ".join(f"{k} {v['successes']}/{v['episodes']} ({v['clean_successes']})"
                                               for k, v in s["by_cohort"].items()))
        for test, value in block["tests"].items():
            lines.append(f"  - {test}: {value['fails_only_first']}/{value['fails_only_second']} (p = {value['p']:.3g}; "
                         "not pre-stated)")
    for name, modes in result["modes"].items():
        lines.append(f"### base model, {name}: {modes['episodes']} episodes, {modes['steps']} actions")
        for key in ("endings", "terminal_outcomes", "first_action", "tool_calls", "refused_or_invalid_steps", "unparsed_outputs"):
            lines.append(f"- {key}: {json.dumps(modes[key], sort_keys=False)}")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cell-dir", required=True, type=Path)
    parser.add_argument("--label", default="base_prompted")
    parser.add_argument("--traces", type=int, default=16)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    result, records = build(args.cell_dir, args.label, args.traces)
    text = markdown(result)
    print(text, end="")
    if args.output is not None:
        args.output.write_text(json.dumps(result, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
        args.output.with_suffix(".md").write_text(text, encoding="utf-8")
        args.output.with_name(args.output.stem + "_episodes.jsonl").write_text(
            "".join(json.dumps(record, sort_keys=True, default=str) + "\n" for record in records), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
