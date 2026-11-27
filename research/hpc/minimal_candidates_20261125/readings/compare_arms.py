"""Two models' candidate read-outs side by side, and call by call at the stopping states they share.

    python research/hpc/minimal_candidates_20261125/readings/compare_arms.py \\
        output/minimal_candidates_20261125/scores_base/states output/minimal_candidates_20261125/scores_steps17/states \\
        output/minimal_candidates_20261125/candidate_inputs_dev_base.jsonl \\
        output/minimal_candidates_20261125/candidate_inputs_dev_steps17.jsonl \\
        research/hpc/minimal_sft_20261115/dev300_reference.json base steps17

Run from the repository root.  A pair is a case whose episode stops at the same observation under both models (the
inputs' `observation` equal): the prompt and the candidate calls are the same there and only the model differs.

A reading script of the cell, not a tool with tests: it was checked by giving it the arm `base` twice (every
difference zero) and by its first block reproducing the tables `candidate_sets read` prints.  Quantiles are the
reader's own (`candidate_sets._quantiles`).  Two tools exactly as probable are reported as equal.
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from research.minimal_env import candidate_sets  # noqa: E402
from research.minimal_env.repair_breakdown import reference_group  # noqa: E402

SIZES = candidate_sets.SET_SIZES
TOOL_OF = {"line": "correct_parameters", "breaker": "correct_topology", "meter": "correct_measurements"}


def load_inputs(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return {row["case"]: row for row in rows}


def observation_key(row: dict) -> str:
    return hashlib.sha256(json.dumps(row["observation"], sort_keys=True).encode()).hexdigest()


def quantile(values, q):
    values = sorted(values)
    if not values:
        return float("nan")
    position = q * (len(values) - 1)
    low, high = math.floor(position), math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def fmt(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:.3g}"


def coverage(readings, key="covered"):
    out = []
    for size in SIZES:
        chosen = [r for r in readings if r["errors_left"]]
        any_ = sum(1 for r in chosen if r[key][str(size)]["any"] is True)
        all_ = sum(1 for r in chosen if r[key][str(size)]["all"] is True)
        undecided = sum(1 for r in chosen if r[key][str(size)]["any"] is None)
        out.append((any_, all_, undecided))
    return out


def coverage_line(readings):
    chosen = [r for r in readings if r["errors_left"]]
    parts = coverage(readings)
    return (f"{len(chosen)} with an error left: any {' / '.join(str(p[0]) for p in parts)}, every {' / '.join(str(p[1]) for p in parts)}"
            + (f", undecided {' / '.join(str(p[2]) for p in parts)}" if any(p[2] for p in parts) else ""))


def kinds_line(readings):
    counts, hits = Counter(), {size: Counter() for size in SIZES}
    nameable = Counter()
    for r in readings:
        for error in r["errors"]:
            counts[error["kind"]] += 1
            nameable[error["kind"]] += 1 if error["listed"] else 0
            for size in SIZES:
                hits[size][error["kind"]] += 1 if error["in_set"][str(size)] is True else 0
    return "; ".join(f"{kind} {counts[kind]} ({nameable[kind]} nameable) {' / '.join(str(hits[size][kind]) for size in SIZES)}"
                     for kind in ("line", "breaker", "meter") if counts[kind])


def standing_line(readings):
    counts, hits = Counter(), {size: Counter() for size in SIZES}
    for r in readings:
        for error in r["errors"]:
            counts[error["standing"]] += 1
            for size in SIZES:
                hits[size][error["standing"]] += 1 if error["in_set"][str(size)] is True else 0
    return "; ".join(f"{standing} {counts[standing]}: {' / '.join(str(hits[size][standing]) for size in SIZES)}" for standing in sorted(counts))


QUANTITIES = (
    ("own last action", lambda r: r["own"]),
    ("finish resolved", lambda r: r["finish_resolved"]),
    ("finish unresolved", lambda r: r["finish_unresolved"]),
    ("inspect breakers", lambda r: r["inspect_breakers"]),
    ("any correction", lambda r: r["correction_mass"]),
    ("a correction of an object still wrong", lambda r: r["mass_on_objects_still_wrong"]),
    ("share of the correction mass on objects still wrong", lambda r: r["share_on_objects_still_wrong"]),
    ("share on the most probable object", lambda r: r["top_share"]),
    ("correct_parameters", lambda r: r["tools"].get("correct_parameters")),
    ("correct_topology", lambda r: r["tools"].get("correct_topology")),
    ("correct_measurements", lambda r: r["tools"].get("correct_measurements")),
    ("undo", lambda r: r["tools"].get("undo")),
)


WITH_ERROR_ONLY = {"a correction of an object still wrong", "share of the correction mass on objects still wrong",
                   "share on the most probable object"}


def quantiles_block(name, readings):
    """The reader's own quantiles (`_quantiles`) over the reader's own populations (`_cell`)."""
    print(f"  {name}: {len(readings)} stops")
    for label, pick in QUANTITIES:
        chosen = [r for r in readings if r["errors_left"]] if label in WITH_ERROR_ONLY else readings
        print(f"    {label}: " + reader_quantiles([pick(r) for r in chosen]))


def top_tool(result):
    """The correction tool(s) with the most probability, and whether that is decided.

    Two tools whose log-probabilities are equal are both returned.  Where no topology call is scored that tool is
    only bounded by what the scored calls leave; the leader is decided when it exceeds the bound.
    """
    known = {tool: value for tool, value in result["tool_logprob"].items()
             if tool in TOOL_OF.values() and value is not None}
    best = max(known.values())
    leaders = sorted(tool for tool in known if abs(known[tool] - best) < 1e-9)
    held = sum(math.exp(call["logprob"]) for call in result["calls"])
    rest = max(0.0, 1.0 - held)
    decided = len(known) == 3 or math.exp(best) > rest
    return leaders, decided


def tool_table(results):
    table = {}
    verdicts = Counter()
    for result in results:
        left = result["truth"].get("errors_left") or []
        if not left:
            continue
        kinds = "+".join(sorted({error["kind"] for error in left}))
        leaders, decided = top_tool(result)
        label = " = ".join(tool.replace("correct_", "") for tool in leaders) + ("" if decided else " (not decided)")
        table.setdefault(kinds, Counter())[label] += 1
        wanted = {TOOL_OF[error["kind"]] for error in left}
        where = "all three measured" if sum(1 for tool in TOOL_OF.values() if result["tool_logprob"].get(tool) is not None) == 3 else "topology bounded"
        verdicts[(where, "not decided" if not decided else "equal" if len(leaders) > 1 else "a tool of an error left" if leaders[0] in wanted else "another tool")] += 1
    return ("; ".join(f"{kinds} ({sum(counts.values())}): {dict(counts)}" for kinds, counts in sorted(table.items()))
            + " || " + "; ".join(f"{where}, {verdict}: {count}" for (where, verdict), count in sorted(verdicts.items())))


def reader_quantiles(values):
    cell = candidate_sets._quantiles([value for value in values if value is not None])
    if cell is None:
        return "n/a"
    return " / ".join(fmt(cell[key]) for key in ("p10", "p25", "median", "p75", "p90")) + f" ({cell['states']})"


def reader_median(values):
    cell = candidate_sets._quantiles([value for value in values if value is not None])
    return None if cell is None else cell["median"]


def main() -> int:
    states_a, states_b, inputs_a, inputs_b, reference_path = (Path(value) for value in sys.argv[1:6])
    name_a, name_b = (sys.argv[6], sys.argv[7]) if len(sys.argv) > 7 else ("A", "B")
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    results_a = {r["case"]: r for r in candidate_sets.read_results(states_a)}
    results_b = {r["case"]: r for r in candidate_sets.read_results(states_b)}
    rows_a, rows_b = load_inputs(inputs_a), load_inputs(inputs_b)
    read_a = {case: candidate_sets.state_reading(result) for case, result in results_a.items()}
    read_b = {case: candidate_sets.state_reading(result) for case, result in results_b.items()}
    print(f"{name_a}: {len(results_a)} scored stops of {len(rows_a)} inputs; {name_b}: {len(results_b)} scored stops of {len(rows_b)} inputs")

    both = sorted(set(results_a) & set(results_b))
    same = [case for case in both if observation_key(rows_a[case]) == observation_key(rows_b[case])]
    other = [case for case in both if case not in same]
    only_a = sorted(set(results_a) - set(results_b))
    only_b = sorted(set(results_b) - set(results_a))
    print(f"cases scored under both {len(both)}: the same state {len(same)}, another state {len(other)}; only {name_a} {len(only_a)}, only {name_b} {len(only_b)}")

    print("\n=== each model over all its stops")
    for name, readings, results in ((name_a, read_a, results_a), (name_b, read_b, results_b)):
        chosen = list(readings.values())
        print(f"  {name}: {coverage_line(chosen)}")
        for group in ("rule expert solves", "only the truth-built sequence repairs", "neither repairs"):
            members = [r for r in chosen if reference_group(reference[r["case"]]) == group]
            print(f"      {group}: {coverage_line(members)}")
        endings = Counter(("finish " + str(r["own_outcome"])) if r["own_tool"] == "finish" else "out of actions" for r in chosen)
        print(f"      endings {dict(endings)}; no error left {sum(1 for r in chosen if not r['errors_left'])}")
        print(f"      errors left by kind: {kinds_line(chosen)}")
        print(f"      by standing: {standing_line(chosen)}")
        print(f"      most probable object: {dict(Counter(r['top_object'][0] for r in chosen if r['top_object']))}")
        print(f"      kind of error left -> most probable correction tool: {tool_table(results.values())}")
        runs = list(results.values())
        print(f"      passes {min(r['passes'] for r in runs)}-{max(r['passes'] for r in runs)} (median {statistics.median(r['passes'] for r in runs)}), "
              f"sum {sum(r['passes'] for r in runs)}; seconds median {statistics.median(r['seconds'] for r in runs):.0f}, max {max(r['seconds'] for r in runs):.0f}; "
              f"prompt tokens {min(r['prompt_tokens'] for r in runs)}-{max(r['prompt_tokens'] for r in runs)}; complete {sum(1 for r in runs if r['complete'])}; "
              f"exact objects min {min(r['exact_objects'] for r in runs)}; cut {sum(1 for r in runs if r['prompt_cut'])}; "
              f"unrenderable {sum(len(r['unrenderable']) for r in runs)}; candidates {sum(r['candidates']['listed'] for r in runs)}; "
              f"held median {statistics.median(sum(math.exp(c['logprob']) for c in r['calls']) for r in runs):.4f}, "
              f"min {min(sum(math.exp(c['logprob']) for c in r['calls']) for r in runs):.3f}, max {max(sum(math.exp(c['logprob']) for c in r['calls']) for r in runs):.5f}")
    print("\n=== probabilities at the stop, 10th / 25th / median / 75th / 90th")
    quantiles_block(name_a, list(read_a.values()))
    quantiles_block(name_b, list(read_b.values()))
    for name, readings in ((name_a, read_a), (name_b, read_b)):
        for ending in ("resolved", "unresolved"):
            chosen = [r for r in readings.values() if r["own_tool"] == "finish" and r["own_outcome"] == ending]
            if chosen:
                print(f"  {name}, ended finish {ending} ({len(chosen)}): that call median {fmt(reader_median([r['own'] for r in chosen]))}, "
                      f"any correction median {fmt(reader_median([r['correction_mass'] for r in chosen]))}, "
                      f"inspect breakers median {fmt(reader_median([r['inspect_breakers'] for r in chosen]))}")
        confident = Counter()
        for r in readings.values():
            if not r["errors_left"] or r["top_share"] is None or r["top_is_wrong_object"] is None:
                continue
            band = "under 0.5" if r["top_share"] < 0.5 else "0.5 to 0.9" if r["top_share"] < 0.9 else "0.9 and over"
            confident[band + ": stops"] += 1
            confident[band + ": the most probable object is one still wrong"] += int(bool(r["top_is_wrong_object"]))
        print(f"  {name}, the most probable object by its share of the correction mass: {dict(sorted(confident.items()))}")
        steps = [r for r in readings.values() if r["has_repairing"]]
        print(f"  {name}, a first step of the truth-built continuation among the 1 / 3 / 5 most probable going-on calls: "
              + " / ".join(str(sum(1 for r in steps if r["next_step"][str(size)] is True)) for size in SIZES) + f" of {len(steps)}")

    print(f"\n=== the {len(same)} stops at the same state")
    pairs = [(read_a[case], read_b[case], results_a[case], results_b[case]) for case in same]
    with_error = [pair for pair in pairs if pair[0]["errors_left"]]
    print(f"with an error left: {len(with_error)}; prompt tokens equal at {sum(1 for a, b, ra, rb in pairs if ra['prompt_tokens'] == rb['prompt_tokens'])} of {len(pairs)}; "
          f"candidates equal at {sum(1 for a, b, ra, rb in pairs if ra['candidates'] == rb['candidates'])}")
    print(f"the same last action at {sum(1 for a, b, ra, rb in pairs if (a['own_tool'], a['own_outcome']) == (b['own_tool'], b['own_outcome']))}; "
          f"{name_a} endings {dict(Counter(str(a['own_tool']) + ' ' + str(a['own_outcome']) for a, b, ra, rb in pairs))}; "
          f"{name_b} endings {dict(Counter(str(b['own_tool']) + ' ' + str(b['own_outcome']) for a, b, ra, rb in pairs))}")
    print(f"groups: {dict(Counter(reference_group(reference[a['case']]) for a, b, ra, rb in pairs))}")
    print(f"  {name_a}: {coverage_line([a for a, b, ra, rb in pairs])}")
    print(f"  {name_b}: {coverage_line([b for a, b, ra, rb in pairs])}")
    for size in SIZES:
        moved = Counter((a["covered"][str(size)]["any"], b["covered"][str(size)]["any"]) for a, b, ra, rb in with_error)
        print(f"    set of {size}, an object still wrong in the set ({name_a}, {name_b}): {dict(moved)}")
    print(f"  errors left by kind, {name_a}: {kinds_line([a for a, b, ra, rb in pairs])}")
    print(f"  errors left by kind, {name_b}: {kinds_line([b for a, b, ra, rb in pairs])}")
    print(f"  tools, {name_a}: {tool_table([ra for a, b, ra, rb in pairs])}")
    print(f"  tools, {name_b}: {tool_table([rb for a, b, ra, rb in pairs])}")
    print(f"  quantity: median {name_a} -> median {name_b}; higher under {name_b} / lower / equal; median ratio {name_b} over {name_a}")
    for label, pick in QUANTITIES:
        chosen = [(pick(a), pick(b)) for a, b, ra, rb in pairs if pick(a) is not None and pick(b) is not None]
        if not chosen:
            continue
        up = sum(1 for x, y in chosen if y > x * 1.0000001)
        down = sum(1 for x, y in chosen if y < x * 0.9999999)
        ratios = [y / x for x, y in chosen if x > 0]
        print(f"    {label} ({len(chosen)}): {fmt(reader_median([x for x, y in chosen]))} -> {fmt(reader_median([y for x, y in chosen]))}; "
              f"{up} / {down} / {len(chosen) - up - down}; ratio {fmt(statistics.median(ratios)) if ratios else 'n/a'}")
    print("  the call on an object still wrong, log-probability under the two models (both scored)")
    by_kind = {}
    for a, b, ra, rb in pairs:
        for error_a, error_b in zip(a["errors"], b["errors"]):
            assert (error_a["kind"], error_a["object"]) == (error_b["kind"], error_b["object"]), (a["case"], error_a, error_b)
            if error_a["logprob"] is None or error_b["logprob"] is None:
                continue
            by_kind.setdefault(error_a["kind"] + ", " + error_a["standing"], []).append((error_a["logprob"], error_b["logprob"], error_a["place"], error_b["place"]))
    for kind, values in sorted(by_kind.items()):
        differences = [y - x for x, y, _, _ in values]
        places_a = [p for _, _, p, _ in values if p is not None]
        places_b = [p for _, _, _, p in values if p is not None]
        print(f"    {kind} ({len(values)}): median {fmt(statistics.median(x for x, _, _, _ in values))} -> {fmt(statistics.median(y for _, y, _, _ in values))}, "
              f"difference median {fmt(statistics.median(differences))} (up {sum(1 for d in differences if d > 0.05)}, down {sum(1 for d in differences if d < -0.05)}); "
              f"place median {fmt(statistics.median(places_a)) if places_a else 'n/a'} ({len(places_a)}) -> {fmt(statistics.median(places_b)) if places_b else 'n/a'} ({len(places_b)})")
    print("  calls scored under both models at a pair: the log-probabilities' difference")
    differences = []
    for a, b, ra, rb in pairs:
        calls_b = {json.dumps(call["action"], sort_keys=True): call["logprob"] for call in rb["calls"]}
        for call in ra["calls"]:
            key = json.dumps(call["action"], sort_keys=True)
            if key in calls_b:
                differences.append((call["kind"], calls_b[key] - call["logprob"], call["logprob"]))
    for kind in sorted({d[0] for d in differences}):
        chosen = [d[1] for d in differences if d[0] == kind]
        print(f"    {kind} ({len(chosen)} calls): median difference {fmt(statistics.median(chosen))}, 10th {fmt(quantile(chosen, 0.1))}, 90th {fmt(quantile(chosen, 0.9))}, "
              f"|difference| median {fmt(statistics.median(abs(c) for c in chosen))}")

    print(f"\n=== the {len(other)} cases both leave unsolved at another state")
    print(f"  {name_a}: {coverage_line([read_a[case] for case in other])}; kinds {kinds_line([read_a[case] for case in other])}")
    print(f"  {name_b}: {coverage_line([read_b[case] for case in other])}; kinds {kinds_line([read_b[case] for case in other])}")
    print(f"\n=== the {len(only_a)} stops only {name_a} has (the cases {name_b} solves)")
    chosen = [read_a[case] for case in only_a]
    print(f"  {name_a}: {coverage_line(chosen)}; kinds {kinds_line(chosen)}")
    print(f"  groups {dict(Counter(reference_group(reference[case]) for case in only_a))}; classes {dict(Counter(read_a[case]['class'] for case in only_a))}")
    print(f"  tools: {tool_table([results_a[case] for case in only_a])}")
    quantiles_block(name_a + " there", chosen)
    print(f"\n=== the {len(only_b)} stops only {name_b} has (the cases {name_a} solves)")
    chosen = [read_b[case] for case in only_b]
    print(f"  {name_b}: {coverage_line(chosen)}; kinds {kinds_line(chosen)}")
    print(f"  groups {dict(Counter(reference_group(reference[case]) for case in only_b))}; classes {dict(Counter(read_b[case]['class'] for case in only_b))}")
    print(f"  tools: {tool_table([results_b[case] for case in only_b])}")
    quantiles_block(name_b + " there", chosen)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
