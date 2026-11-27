"""The model's set against the evidence probe by kind of error left, and the breakers view at the stops.

    python research/hpc/minimal_candidates_20261125/readings/probe_by_kind.py \\
        output/minimal_candidates_20261125/scores_steps17/states research/hpc/minimal_sft_20261115/dev300_reference.json \\
        output/minimal_candidates_20261125/stops_steps17/probe_report.json

Run from the repository root.  The probe has two readings: `candidate_inspect` (the substation table of the current
state, one `inspect breakers` away) and `candidate_shown` (that table only where the prompt shows it).

A reading script of the cell, not a tool with tests: on the arm `base` (with the probe report of 2026-11-20) it
reproduces the README's table of the model's set against the probe.
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from research.minimal_env import candidate_sets  # noqa: E402
from research.minimal_env.repair_breakdown import reference_group  # noqa: E402

SIZES = candidate_sets.SET_SIZES


def q(values):
    cell = candidate_sets._quantiles(values)
    if cell is None:
        return "n/a"
    return " / ".join(f"{cell[key]:.3g}" for key in ("p10", "p25", "median", "p75", "p90")) + f" ({cell['states']})"


def main() -> int:
    results = candidate_sets.read_results(Path(sys.argv[1]))
    reference = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    report = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
    places = {name: candidate_sets.probe_places(report, name) for name in ("candidate_inspect", "candidate_shown")}
    by_case = {result["case"]: result for result in results}
    readings = {case: candidate_sets.state_reading(result) for case, result in by_case.items()}
    with_error = {case: r for case, r in readings.items() if r["errors_left"]}
    solvable = {case: r for case, r in with_error.items() if reference_group(reference[case]) != "neither repairs"}
    joined = {case: r for case, r in solvable.items() if r["key"] in places["candidate_inspect"]}
    print(f"stops {len(readings)}, with an error left {len(with_error)}, solvable with an error left {len(solvable)}, "
          f"joined with the probe {len(joined)} (inspect), {sum(1 for r in solvable.values() if r['key'] in places['candidate_shown'])} (shown)")
    missing = [case for case in solvable if case not in joined]
    if missing:
        print("  solvable stops with an error left that the probe report does not hold:", missing)

    def has_breaker(r):
        return any(error["kind"] == "breaker" for error in r["errors"])

    groups = (("all", list(joined.values())), ("a breaker still wrong", [r for r in joined.values() if has_breaker(r)]),
              ("lines or meters only", [r for r in joined.values() if not has_breaker(r)]))
    for label, chosen in groups:
        print(f"  {label}: {len(chosen)} stops")
        for name in ("candidate_inspect", "candidate_shown"):
            print(f"    probe, {name}: " + " / ".join(str(sum(1 for r in chosen if places[name].get(r['key'], 10**9) <= size)) for size in SIZES))
        print("    the model's set: " + " / ".join(str(sum(1 for r in chosen if r['covered'][str(size)]['any'] is True)) for size in SIZES)
              + "; undecided " + " / ".join(str(sum(1 for r in chosen if r['covered'][str(size)]['any'] is None)) for size in SIZES))
    for size in SIZES:
        cells = Counter()
        for r in joined.values():
            probe = places["candidate_inspect"][r["key"]] <= size
            model = r["covered"][str(size)]["any"] is True
            cells["both" if probe and model else "probe only" if probe else "model only" if model else "neither"] += 1
        print(f"  with the breakers view, set of {size}: {dict(cells)}")

    print("\n=== the breakers view at the stops")
    shown = {case: by_case[case]["shown"]["breakers"] for case in readings}
    for label, chosen in (("all stops", list(readings)), ("stops with a breaker still wrong", [c for c, r in readings.items() if has_breaker(r)]),
                          ("  of those, solvable", [c for c, r in readings.items() if has_breaker(r) and reference_group(reference[c]) != "neither repairs"]),
                          ("stops with lines or meters only", [c for c, r in with_error.items() if not has_breaker(r)])):
        tables = Counter("the prompt names a breaker" if shown[c] else "the prompt names none" for c in chosen)
        values = [readings[c]["inspect_breakers"] for c in chosen]
        print(f"  {label}: {len(chosen)} stops; {dict(tables)}; inspect breakers {q(values)}; over 0.01 at {sum(1 for v in values if v is not None and v > 0.01)}, "
              f"over 0.1 at {sum(1 for v in values if v is not None and v > 0.1)}")
        for table in ("the prompt names a breaker", "the prompt names none"):
            part = [readings[c]["inspect_breakers"] for c in chosen if ("the prompt names a breaker" if shown[c] else "the prompt names none") == table]
            if part:
                print(f"      {table}: inspect breakers {q(part)}; over 0.01 at {sum(1 for v in part if v > 0.01)}")
    breakers = [(case, error) for case, r in readings.items() for error in r["errors"] if error["kind"] == "breaker"]
    print(f"  breaker errors left: {len(breakers)}; nameable {sum(1 for _, e in breakers if e['listed'])}; "
          f"at a stop whose prompt names a breaker {sum(1 for c, _ in breakers if shown[c])}; standings {dict(Counter(e['standing'] for _, e in breakers))}")
    scored = []
    for case, error in breakers:
        calls = [call for call in by_case[case]["calls"] if call["object"] is not None and (str(call["object"][0]), str(call["object"][1])) == ("breaker", error["object"])]
        if calls:
            scored.append(max(math.exp(call["logprob"]) for call in calls))
    print(f"  the call on a breaker still wrong, scored for {len(scored)}: {q(scored)}; max {max(scored) if scored else float('nan'):.3g}")
    ends = Counter()
    for case, r in readings.items():
        if has_breaker(r):
            ends[("finish " + str(r["own_outcome"])) if r["own_tool"] == "finish" else "out of actions"] += 1
    print(f"  endings at the stops with a breaker still wrong: {dict(ends)}")
    classes = Counter(readings[c]["class"] for c, r in readings.items() if has_breaker(r))
    print(f"  classes there: {dict(classes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
