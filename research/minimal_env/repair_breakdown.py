"""How far the episodes of an evaluation got: every error repaired, some of them, none; and what they damaged.

    python -m research.minimal_env.repair_breakdown EVAL [EVAL ...] [--names A,B,...] \\
        [--reference REFERENCE.json] [--output OUT.json]

``solved`` (``psse_env.minimal.scoring``) asks for every injected error within tolerance and no healthy object
outside it, and the reward follows it: an episode that repairs two of three errors counts and earns the same as one
that repairs none.  Every scored episode already carries the record this module reads (``score.physical``: each
injected error with whether it ended repaired, and every healthy object that ended outside tolerance).  Nothing is
re-scored here and nothing is credited: ``solved`` stays the result, and this is the split of what lies under it.

An episode falls in one of four groups:

* ``solved``;
* every error repaired, but a healthy object damaged;
* some errors repaired (a partial repair, with or without damage);
* no error repaired.

Beside the split: the injected errors repaired by kind (breaker, line, meter), how the unsolved episodes ended
(``resolved`` declared, ``unresolved`` declared, out of actions) with the estimator's verdict at the end, how many
errors they left, and the partial repairs that ended honestly (``unresolved`` declared, nothing damaged).  With a
reference file (``{case: {"expert": bool, "truth": bool}}``) the same split is given for the cases the rule expert
solves, the ones only a truth-built sequence repairs, and the ones neither repairs.

``research.minimal_env.evaluate`` puts ``breakdown`` into every summary it writes (``summary["repairs"]``).  The
command reads evaluations that are already stored: each ``EVAL`` is an evaluation folder (``EVAL/episodes/*.json``),
a folder with an ``episodes.jsonl`` (the rule expert's replay, the truth-built sequences), or such a file.  With
two or more it prints them side by side and compares each with the first, case by case.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from math import comb
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

BREAKDOWN_CONTRACT = "minimal_repair_breakdown_v1"
SOLVED = "solved"
DAMAGE_ONLY = "every error repaired, a healthy object damaged"
PARTIAL = "some errors repaired"
NOTHING = "no error repaired"
GROUPS = (SOLVED, DAMAGE_ONLY, PARTIAL, NOTHING)
RESOLVED = "declared resolved"
UNRESOLVED = "declared unresolved"
OUT_OF_ACTIONS = "out of actions"
UNDECLARED = "ended without a declaration"
ENDINGS = (RESOLVED, UNRESOLVED, OUT_OF_ACTIONS, UNDECLARED)
HONEST_PARTIAL = "some errors repaired, nothing damaged, unresolved declared"
REFERENCE_GROUPS = ("rule expert solves", "only the truth-built sequence repairs", "neither repairs")
#: The kinds an error record of ``scoring.physical_comparison`` carries, in the order they are printed.
KINDS = ("breaker", "line_status", "line", "meter")


def repair_group(score: Mapping[str, Any]) -> str:
    """Which of the four groups a scored episode falls in."""
    physical = score["physical"]
    if score.get("solved"):
        return SOLVED
    errors, repaired = int(physical.get("true_errors") or 0), int(physical.get("repaired") or 0)
    if repaired >= errors:
        return DAMAGE_ONLY
    return PARTIAL if repaired > 0 else NOTHING


def ending(record: Mapping[str, Any]) -> str:
    """How an episode ended: what it declared, or why it declared nothing."""
    declared = (record.get("score") or {}).get("declared")
    if declared == "resolved":
        return RESOLVED
    if declared == "unresolved":
        return UNRESOLVED
    return OUT_OF_ACTIONS if record.get("outcome") == "budget_exhausted" else UNDECLARED


def case_counts(record: Mapping[str, Any]) -> dict[str, Any]:
    """One episode's numbers: errors injected, repaired, healthy objects damaged, and its group."""
    score = record["score"]
    physical = score["physical"]
    return {"errors": int(physical.get("true_errors") or 0), "repaired": int(physical.get("repaired") or 0),
            "damaged": int(physical.get("damaged") or 0), "group": repair_group(score)}


def _cell(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups: Counter = Counter()
    injected: Counter = Counter()
    repaired: Counter = Counter()
    damage: Counter = Counter()
    shares: list[float] = []
    errors = fixed = damaged_objects = damaged_episodes = 0
    for record in records:
        physical = record["score"]["physical"]
        counts = case_counts(record)
        groups[counts["group"]] += 1
        errors += counts["errors"]
        fixed += counts["repaired"]
        damaged_objects += counts["damaged"]
        damaged_episodes += int(counts["damaged"] > 0)
        if counts["errors"]:
            shares.append(counts["repaired"] / counts["errors"])
        for item in physical.get("errors") or []:
            injected[str(item.get("kind"))] += 1
            repaired[str(item.get("kind"))] += int(bool(item.get("repaired")))
        for item in physical.get("damage") or []:
            damage[str(item.get("kind"))] += 1
    kinds = [kind for kind in KINDS if kind in injected] + sorted(set(injected) - set(KINDS))
    return {
        "cases": len(records),
        "episodes": {name: int(groups[name]) for name in GROUPS},
        "errors": {"injected": errors, "repaired": fixed,
                   "by_kind": {kind: {"injected": int(injected[kind]), "repaired": int(repaired[kind])} for kind in kinds}},
        "share_of_errors_repaired": round(fixed / errors, 4) if errors else None,
        "mean_share_repaired_per_case": round(statistics.mean(shares), 4) if shares else None,
        "damage": {"episodes": damaged_episodes, "objects": damaged_objects, "by_kind": dict(sorted(damage.items()))},
    }


def _unsolved(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_ending: dict[str, Counter] = {name: Counter() for name in ENDINGS}
    left: Counter = Counter()
    left_by_kind: Counter = Counter()
    honest = total = 0
    for record in records:
        score = record["score"]
        physical = score["physical"]
        counts = case_counts(record)
        if counts["group"] == SOLVED:
            continue
        total += 1
        how = ending(record)
        cell = by_ending[how]
        cell["episodes"] += 1
        cell[counts["group"]] += 1
        cell["with a damaged healthy object"] += int(counts["damaged"] > 0)
        cell["estimator passes at the end"] += int(bool(score.get("estimator_passes")))
        left[max(counts["errors"] - counts["repaired"], 0)] += 1
        for item in physical.get("errors") or []:
            if not item.get("repaired"):
                left_by_kind[str(item.get("kind"))] += 1
        honest += int(counts["group"] == PARTIAL and counts["damaged"] == 0 and how == UNRESOLVED)
    keys = ("episodes", DAMAGE_ONLY, PARTIAL, NOTHING, "with a damaged healthy object", "estimator passes at the end")
    return {
        "episodes": total,
        "by_ending": {name: {key: int(cell[key]) for key in keys} for name, cell in by_ending.items()},
        "errors_left": {str(count): int(left[count]) for count in sorted(left)},
        "errors_left_by_kind": {kind: int(left_by_kind[kind]) for kind in
                                [kind for kind in KINDS if kind in left_by_kind] + sorted(set(left_by_kind) - set(KINDS))},
        HONEST_PARTIAL: honest,
    }


def reference_group(known: Mapping[str, Any]) -> str:
    return REFERENCE_GROUPS[0] if known.get("expert") else REFERENCE_GROUPS[1] if known.get("truth") else REFERENCE_GROUPS[2]


def breakdown(records: Sequence[Mapping[str, Any]], reference: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """The split of ``records`` (stored episodes of ``research.minimal_env.evaluate``, each with its ``score``)."""
    out: dict[str, Any] = {"contract": BREAKDOWN_CONTRACT, **_cell(records), "unsolved": _unsolved(records)}
    by_class: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        by_class.setdefault(str(record.get("compound_class")), []).append(record)
    out["by_class"] = {name: _cell(items) for name, items in sorted(by_class.items())}
    if reference is not None:
        grouped: dict[str, list[Mapping[str, Any]]] = {name: [] for name in REFERENCE_GROUPS}
        for record in records:
            known = reference.get(str(record.get("scenario_id")))
            if known is not None:
                grouped[reference_group(known)].append(record)
        out["by_reference"] = {name: {**_cell(items), "unsolved": _unsolved(items)} for name, items in grouped.items()}
    return out


# ------------------------------------------------------------------ two evaluations, case by case


def mcnemar(only_first: int, only_second: int) -> float:
    """Exact two-sided McNemar p from the two discordant counts."""
    total = only_first + only_second
    if total == 0:
        return 1.0
    low = min(only_first, only_second)
    return min(1.0, 2 * sum(comb(total, index) for index in range(low + 1)) / 2 ** total)


def paired(first: Sequence[Mapping[str, Any]], other: Sequence[Mapping[str, Any]],
           cases: Iterable[str] | None = None) -> dict[str, Any]:
    """``other`` against ``first`` on the cases both hold (of ``cases`` when given): solved gained and lost, and the
    cases where more or fewer of the injected errors ended repaired."""
    before = {str(record["scenario_id"]): record for record in first}
    after = {str(record["scenario_id"]): record for record in other}
    shared = sorted(set(before) & set(after) if cases is None else set(before) & set(after) & {str(case) for case in cases})
    counts: Counter = Counter()
    for case in shared:
        was, now = case_counts(before[case]), case_counts(after[case])
        counts["gained"] += int(now["group"] == SOLVED and was["group"] != SOLVED)
        counts["lost"] += int(was["group"] == SOLVED and now["group"] != SOLVED)
        counts["more errors repaired"] += int(now["repaired"] > was["repaired"])
        counts["fewer errors repaired"] += int(now["repaired"] < was["repaired"])
        counts["errors repaired, change"] += now["repaired"] - was["repaired"]
        counts["healthy objects damaged, change"] += now["damaged"] - was["damaged"]
    return {"cases": len(shared), "solved gained": int(counts["gained"]), "solved lost": int(counts["lost"]),
            "p": round(mcnemar(counts["gained"], counts["lost"]), 4),
            "more errors repaired": int(counts["more errors repaired"]),
            "fewer errors repaired": int(counts["fewer errors repaired"]),
            "errors repaired, change": int(counts["errors repaired, change"]),
            "healthy objects damaged, change": int(counts["healthy objects damaged, change"])}


# ------------------------------------------------------------------ reading stored evaluations


def normalize(record: Mapping[str, Any]) -> dict[str, Any] | None:
    """A stored record as an evaluation episode, or ``None`` for one that is not a case's result.

    Three writers are read: ``evaluate`` (the episode itself), ``expert_baseline`` (the episode under ``episode``)
    and ``build_demonstrations`` (one record per path: the success path is the case's result, a recovery path is
    left out).
    """
    if isinstance(record.get("episode"), Mapping):
        inner = dict(record["episode"])
        inner.setdefault("compound_class", record.get("compound_class"))
        record = inner
    if "demonstration" in record and record.get("demonstration") != "success":
        return None
    if not isinstance(record.get("score"), Mapping) or "physical" not in record["score"]:
        raise ValueError(f"a record of case {record.get('scenario_id')!r} carries no score")
    out = dict(record)
    out["scenario_id"] = str(record.get("scenario_id") or record["score"].get("scenario_id") or "")
    out["compound_class"] = record.get("compound_class") or record.get("class") or record["score"].get("class")
    return out


def read_records(path: Path) -> list[dict[str, Any]]:
    """The episodes of an evaluation folder, of a folder with an ``episodes.jsonl``, or of such a file, by case."""
    path = Path(path)
    raw: list[Mapping[str, Any]] = []
    if path.is_dir() and (path / "episodes").is_dir():
        raw = [json.loads(item.read_text(encoding="utf-8")) for item in sorted((path / "episodes").glob("*.json"))]
    else:
        source = path / "episodes.jsonl" if path.is_dir() else path
        if not source.is_file():
            raise FileNotFoundError(f"{path} holds no episodes folder and no episodes.jsonl")
        raw = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = [item for item in (normalize(record) for record in raw) if item is not None]
    seen: set[str] = set()
    for record in records:
        if record["scenario_id"] in seen:
            raise ValueError(f"{path}: case {record['scenario_id']} appears twice")
        seen.add(record["scenario_id"])
    if not records:
        raise ValueError(f"{path} holds no episode")
    return sorted(records, key=lambda record: record["scenario_id"])


def _of(part: int, whole: int) -> str:
    return f"{part} of {whole}"


def table(names: Sequence[str], breakdowns: Sequence[Mapping[str, Any]], key: str | None = None) -> str:
    """The breakdowns side by side as a Markdown table; ``key`` names a reference group to print instead of the whole."""
    cells = [item["by_reference"][key] if key else item for item in breakdowns]
    rows: list[tuple[str, list[str]]] = [("Cases", [str(cell["cases"]) for cell in cells])]
    labels = {SOLVED: "Solved", DAMAGE_ONLY: "Every error repaired, a healthy object damaged", PARTIAL: "Some errors repaired",
              NOTHING: "No error repaired"}
    for name in GROUPS:
        rows.append((labels[name], [str(cell["episodes"][name]) for cell in cells]))
    rows.append(("Errors repaired", [_of(cell["errors"]["repaired"], cell["errors"]["injected"]) for cell in cells]))
    kinds = [kind for kind in KINDS if any(kind in cell["errors"]["by_kind"] for cell in cells)]
    for kind in kinds:
        rows.append((f"  {kind.replace('_', ' ')}s", [
            _of(cell["errors"]["by_kind"].get(kind, {}).get("repaired", 0), cell["errors"]["by_kind"].get(kind, {}).get("injected", 0))
            for cell in cells]))
    rows.append(("Mean share of a case's errors repaired", [
        "n/a" if cell["mean_share_repaired_per_case"] is None else f"{cell['mean_share_repaired_per_case']:.3f}" for cell in cells]))
    rows.append(("Healthy objects damaged (episodes)", [f"{cell['damage']['objects']} ({cell['damage']['episodes']})" for cell in cells]))
    rows.append(("Unsolved episodes", [str(cell["unsolved"]["episodes"]) for cell in cells]))
    for name in ENDINGS:
        if any(cell["unsolved"]["by_ending"][name]["episodes"] for cell in cells):
            rows.append((f"  {name} (estimator passes at the end)", [
                f"{cell['unsolved']['by_ending'][name]['episodes']} ({cell['unsolved']['by_ending'][name]['estimator passes at the end']})"
                for cell in cells]))
    rows.append((f"  {HONEST_PARTIAL}", [str(cell["unsolved"][HONEST_PARTIAL]) for cell in cells]))
    counts = sorted({int(count) for cell in cells for count in cell["unsolved"]["errors_left"]})
    for count in counts:
        rows.append((f"  unsolved with {count} error{'' if count == 1 else 's'} left",
                     [str(cell["unsolved"]["errors_left"].get(str(count), 0)) for cell in cells]))
    lines = ["| | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    lines += [f"| {label} | " + " | ".join(values) + " |" for label, values in rows]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("evaluations", nargs="+", type=Path)
    parser.add_argument("--names", default=None, help="comma-separated column names, one per evaluation")
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None, help="write the breakdowns and the comparisons here as JSON")
    args = parser.parse_args(argv)
    names = [name.strip() for name in args.names.split(",")] if args.names else [path.name or str(path) for path in args.evaluations]
    if len(names) != len(args.evaluations):
        raise SystemExit(f"{len(names)} names for {len(args.evaluations)} evaluations")
    reference = json.loads(args.reference.read_text(encoding="utf-8")) if args.reference is not None else None
    stored = [read_records(path) for path in args.evaluations]
    breakdowns = [breakdown(records, reference) for records in stored]
    print(table(names, breakdowns))
    payload: dict[str, Any] = {"contract": BREAKDOWN_CONTRACT, "evaluations": {
        name: {"path": str(path), **item} for name, path, item in zip(names, args.evaluations, breakdowns)}}
    if reference is not None:
        for group in REFERENCE_GROUPS:
            print(f"\n{group}\n")
            print(table(names, breakdowns, group))
    if len(stored) > 1:
        solvable = None if reference is None else [case for case, known in reference.items() if known.get("truth") or known.get("expert")]
        payload["against_the_first"] = {}
        print(f"\nAgainst {names[0]}, case by case\n")
        for name, records in zip(names[1:], stored[1:]):
            entry = {"all cases": paired(stored[0], records)}
            if solvable is not None:
                entry["cases a truth-built sequence or the rule expert repairs"] = paired(stored[0], records, solvable)
            payload["against_the_first"][name] = entry
            for scope, item in entry.items():
                p_value = f"p = {item['p']:.2f}" if item["p"] >= 0.01 else "p < 0.01"
                print(f"{name}, {scope} ({item['cases']}): solved gained {item['solved gained']}, lost {item['solved lost']} "
                      f"({p_value}); more errors repaired in {item['more errors repaired']} cases, fewer in "
                      f"{item['fewer errors repaired']}; errors repaired {item['errors repaired, change']:+d}, healthy objects "
                      f"damaged {item['healthy objects damaged, change']:+d}")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
