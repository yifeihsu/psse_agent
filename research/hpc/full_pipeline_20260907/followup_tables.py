#!/usr/bin/env python3
"""Tables and paired tests for the conventional-only cell's follow-ups (2026-10-23, 2026-10-27).

    followup_tables.py --cell-dir PIPE [--extra-policy LABEL ...] [--output OUT.json]

Joins every policy's per-root outcome on each suite by physical root:

* ``test``: the 192 test roots.  Expert, BC0 and R1 from the round-1
  evaluation, R2 from the round-2 evaluation (its candidate slot), the
  matched cloning control from ``out/extra/evaluations/test_control``.
* ``hard`` and ``validation``: ``out/extra/evaluations/{suite}_bc0_r1``
  (BC0, R1, expert) and ``{suite}_r2_control`` (R2, control).
* ``--extra-policy LABEL`` (2026-10-27): an adapter evaluated alone on the
  three suites by ``submit_extras_20261027.sh`` (``{suite}_LABEL``, its
  record in the evaluator's first adapter slot).
* ``--r2-evaluation NAME`` and ``--control-evaluation NAME`` (2026-10-29):
  read R2 on the hard and validation suites from ``{suite}_NAME``'s first
  slot and the control from ``{suite}_NAME``'s second slot, for the round-2
  rerun, whose R2 is evaluated alone (``{suite}_r2``) while the recorded
  ``{suite}_r2_control`` is set aside.

For each suite and policy: successes per family and per parameter-ranking
stratum (and per mimic variant on the hard draw), safety-clean successes
(solved with no invalid action, false commit or loop), episodes with an
invalid action, a false commit, a loop, and with any of the three ("process
violations"), mean actions and WLS calls; and for every policy against R1
and against BC0, the exact two-sided McNemar test on the paired roots for
success, for safety-clean success and for a process violation.  The round
selection applies the rule fixed in the cell note (section 6) to BC0, R1
and R2 on the validation roots.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from math import comb
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

POLICIES = ("expert", "bc0", "r1", "r2", "control")
ITERATES = ("bc0", "r1", "r2")
#: Policies every other one is paired against (the DAgger result, the cloning baseline).
REFERENCES = ("r1", "bc0")


def mcnemar(only_a: int, only_b: int) -> float:
    """Exact two-sided McNemar p from the two discordant counts."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def _episodes(payload: Any) -> list[Mapping[str, Any]]:
    if isinstance(payload, Mapping):
        if isinstance(payload.get("episodes"), list):
            return [e for e in payload["episodes"] if isinstance(e, Mapping)]
        for value in payload.values():
            found = _episodes(value)
            if found:
                return found
    return []


def _root(row: Mapping[str, Any]) -> str:
    for holder in (row, row.get("grouping") or {}, row.get("metadata") or {}):
        if isinstance(holder, Mapping) and holder.get("physical_root_fingerprint"):
            return str(holder["physical_root_fingerprint"])
    return ""


def describe_roots(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    described = {}
    for row in rows:
        audit = row.get("audit") or {}
        ranking = audit.get("parameter_ranking") if isinstance(audit.get("parameter_ranking"), Mapping) else {}
        mimic = audit.get("mimic") if isinstance(audit.get("mimic"), Mapping) else {}
        described[_root(row)] = {
            "family": str((row.get("grouping") or {}).get("scenario_family") or ""),
            "stratum": ranking.get("stratum"),
            "mimic": mimic.get("variant"),
        }
    return described


def records(eval_path: Path, roots: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """root -> compact outcome of one policy's evaluation file."""
    payload = json.loads(eval_path.read_text(encoding="utf-8"))
    result = {}
    for episode in _episodes(payload):
        root = str(episode.get("physical_root") or "")
        info = roots.get(root, {})
        invalid = int(episode.get("invalid_action_count") or 0) > 0
        false_commit = int(episode.get("false_commit_count") or 0) > 0
        loop = bool(episode.get("loop_detected"))
        success = bool(episode.get("truth_audited_task_success"))
        violation = invalid or false_commit or loop
        result[root] = {
            **info,
            "success": success,
            "invalid": invalid,
            "false_commit": false_commit,
            "loop": loop,
            "violation": violation,
            # Safety-clean success: solved with no process violation (2026-10-27).
            "clean": success and not violation,
            "terminal": bool(episode.get("terminal")),
            "terminal_outcome": episode.get("terminal_outcome"),
            "steps": int(episode.get("steps") or 0),
            "wls_calls": int(episode.get("wls_calls") or 0),
            "comparisons": int((episode.get("tool_counts") or {}).get("compare_parameter_candidates") or 0),
        }
    return result


def summarize_policy(outcomes: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    n = len(outcomes)
    by_family: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    by_stratum: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    by_mimic: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for item in outcomes.values():
        for table, key in ((by_family, item.get("family")), (by_stratum, item.get("stratum")), (by_mimic, item.get("mimic"))):
            if key is not None:
                table[str(key)][0] += int(item["success"])
                table[str(key)][1] += 1
                table[str(key)][2] += int(item["clean"])
    count = lambda field: sum(1 for item in outcomes.values() if item[field])
    cells = lambda table: {k: {"successes": v[0], "episodes": v[1], "clean_successes": v[2]} for k, v in sorted(table.items())}
    return {
        "episodes": n,
        "successes": count("success"),
        "clean_successes": count("clean"),
        "invalid_action_episodes": count("invalid"),
        "false_commit_episodes": count("false_commit"),
        "loop_episodes": count("loop"),
        "violation_episodes": count("violation"),
        "mean_actions": sum(item["steps"] for item in outcomes.values()) / n if n else None,
        "mean_wls_calls": sum(item["wls_calls"] for item in outcomes.values()) / n if n else None,
        "comparisons": sum(item["comparisons"] for item in outcomes.values()),
        "by_family": cells(by_family),
        "by_stratum": cells(by_stratum),
        "by_mimic": cells(by_mimic),
    }


def paired(a: Mapping[str, Mapping[str, Any]], b: Mapping[str, Mapping[str, Any]], field: str) -> dict[str, Any]:
    """Discordant roots on ``field`` (True counts against success, for violations)."""
    shared = sorted(set(a) & set(b))
    only_a = sum(1 for root in shared if a[root][field] and not b[root][field])
    only_b = sum(1 for root in shared if b[root][field] and not a[root][field])
    return {"paired_roots": len(shared), "only_first": only_a, "only_second": only_b, "p": mcnemar(only_a, only_b)}


def _failures(outcomes: Mapping[str, Mapping[str, Any]], field: str) -> dict[str, dict[str, bool]]:
    return {root: {"f": not item[field]} for root, item in outcomes.items()}


def paired_tests(outcomes: Mapping[str, Mapping[str, Mapping[str, Any]]], reference: str) -> dict[str, Any]:
    """Every other policy against ``reference``: success, safety-clean success, process violations."""
    tests: dict[str, Any] = {}
    if reference not in outcomes:
        return tests
    for policy in outcomes:
        if policy == reference:
            continue
        tests[f"{policy}_vs_{reference}"] = {
            "success_failures": paired(_failures(outcomes[policy], "success"), _failures(outcomes[reference], "success"), "f"),
            "clean_success_failures": paired(_failures(outcomes[policy], "clean"), _failures(outcomes[reference], "clean"), "f"),
            "violations": paired(outcomes[policy], outcomes[reference], "violation"),
        }
    return tests


def select_round(validation: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """The cell note's rule: most successes, then fewest violation episodes, then fewest mean actions."""
    ranked = sorted(
        (name for name in ITERATES if name in validation),
        key=lambda name: (-validation[name]["successes"], validation[name]["violation_episodes"],
                          validation[name]["mean_actions"]),
    )
    return {"rule": "most successes; then fewest process-violation episodes; then fewest mean actions",
            "order": ranked, "selected": ranked[0] if ranked else None}


def suite_sources(cell: Path, extra: Sequence[str] = (), *, r2_evaluation: str = "r2_control",
                  control_evaluation: str = "r2_control") -> dict[str, dict[str, tuple[Path, Path]]]:
    """suite -> policy -> (evaluation file, suite rows)."""
    out = cell / "out"
    extra_dir = out / "extra" / "evaluations"
    test_rows = out / "r1" / "collection" / "development_scenarios.json"
    sources = {
        "test": {
            "expert": (out / "r1" / "collection" / "evaluation" / "expert_eval.json", test_rows),
            "bc0": (out / "r1" / "collection" / "evaluation" / "bc0_eval.json", test_rows),
            "r1": (out / "r1" / "collection" / "evaluation" / "r1_eval.json", test_rows),
            "r2": (out / "r2" / "collection" / "evaluation" / "r1_eval.json", test_rows),
            "control": (extra_dir / "test_control" / "bc0_eval.json", test_rows),
        },
    }
    for suite in ("hard", "validation"):
        rows = out / "extra" / "suites" / f"{suite}.json"
        sources[suite] = {
            "expert": (extra_dir / f"{suite}_bc0_r1" / "expert_eval.json", rows),
            "bc0": (extra_dir / f"{suite}_bc0_r1" / "bc0_eval.json", rows),
            "r1": (extra_dir / f"{suite}_bc0_r1" / "r1_eval.json", rows),
            "r2": (extra_dir / f"{suite}_{r2_evaluation}" / "bc0_eval.json", rows),
            "control": (extra_dir / f"{suite}_{control_evaluation}" / "r1_eval.json", rows),
        }
    for label in extra:
        if label in POLICIES or not label:
            raise ValueError(f"extra policy label {label!r} is reserved or empty")
        for suite, rows in (("test", test_rows), ("hard", out / "extra" / "suites" / "hard.json"),
                            ("validation", out / "extra" / "suites" / "validation.json")):
            sources[suite][label] = (extra_dir / f"{suite}_{label}" / "bc0_eval.json", rows)
    return sources


def build(cell: Path, extra: Sequence[str] = (), *, r2_evaluation: str = "r2_control",
          control_evaluation: str = "r2_control") -> dict[str, Any]:
    result: dict[str, Any] = {}
    sources = suite_sources(cell, extra, r2_evaluation=r2_evaluation, control_evaluation=control_evaluation)
    for suite, policies in sources.items():
        outcomes: dict[str, dict[str, dict[str, Any]]] = {}
        for policy, (eval_path, rows_path) in policies.items():
            if eval_path.is_file() and rows_path.is_file():
                roots = describe_roots(json.loads(rows_path.read_text(encoding="utf-8")))
                outcomes[policy] = records(eval_path, roots)
        summaries = {policy: summarize_policy(items) for policy, items in outcomes.items()}
        result[suite] = {
            "policies": summaries,
            "paired_against_r1": paired_tests(outcomes, "r1"),
            "paired_against_bc0": paired_tests(outcomes, "bc0"),
        }
    if "validation" in result:
        result["round_selection"] = select_round(result["validation"]["policies"])
    return result


def _paired_line(name: str, tests: Mapping[str, Any]) -> str:
    first, reference = name.split("_vs_")
    s, c, v = tests["success_failures"], tests["clean_success_failures"], tests["violations"]
    return (f"- {name}: failures only {first} {s['only_first']} / only {reference} {s['only_second']} "
            f"(p = {s['p']:.3g}); clean-success failures only {first} {c['only_first']} / only {reference} "
            f"{c['only_second']} (p = {c['p']:.3g}); violations only {first} {v['only_first']} / only {reference} "
            f"{v['only_second']} (p = {v['p']:.3g}); {s['paired_roots']} paired roots")


def markdown(result: Mapping[str, Any]) -> str:
    lines = []
    for suite in ("test", "hard", "validation"):
        block = result.get(suite)
        if not block or not block["policies"]:
            continue
        names = [p for p in POLICIES if p in block["policies"]]
        names += sorted(p for p in block["policies"] if p not in POLICIES)
        lines.append(f"### {suite}")
        lines.append("| | " + " | ".join(names) + " |")
        lines.append("| --- |" + " --- |" * len(names))
        families = sorted({f for p in names for f in block["policies"][p]["by_family"]})
        for family in families:
            cells = []
            for p in names:
                cell = block["policies"][p]["by_family"].get(family)
                cells.append(f"{cell['successes']}/{cell['episodes']} ({cell['clean_successes']})" if cell else "-")
            lines.append(f"| {family} (clean) | " + " | ".join(cells) + " |")
        for key, label in (("successes", "**successes**"), ("clean_successes", "safety-clean successes"),
                           ("violation_episodes", "process-violation episodes"),
                           ("invalid_action_episodes", "invalid-action episodes"),
                           ("false_commit_episodes", "false-commit episodes"), ("loop_episodes", "loop episodes"),
                           ("mean_actions", "mean actions"), ("mean_wls_calls", "mean WLS calls"),
                           ("comparisons", "comparisons")):
            values = [block["policies"][p][key] for p in names]
            text = [f"{v:.2f}" if isinstance(v, float) else str(v) for v in values]
            lines.append(f"| {label} | " + " | ".join(text) + " |")
        for reference in REFERENCES:
            for name, tests in block.get(f"paired_against_{reference}", {}).items():
                lines.append(_paired_line(name, tests))
        lines.append("")
    if "round_selection" in result:
        selection = result["round_selection"]
        lines.append(f"Round selection ({selection['rule']}): order {selection['order']}, selected {selection['selected']}.")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cell-dir", required=True, type=Path)
    parser.add_argument("--extra-policy", action="append", default=[], help="label of an adapter evaluated on the three suites")
    parser.add_argument("--r2-evaluation", default="r2_control",
                        help="R2 on the hard and validation suites from {suite}_NAME's first adapter slot")
    parser.add_argument("--control-evaluation", default="r2_control",
                        help="the control on the hard and validation suites from {suite}_NAME's second adapter slot")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    result = build(args.cell_dir, args.extra_policy, r2_evaluation=args.r2_evaluation,
                   control_evaluation=args.control_evaluation)
    text = markdown(result)
    print(text, end="")
    if args.output is not None:
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        args.output.with_suffix(".md").write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
