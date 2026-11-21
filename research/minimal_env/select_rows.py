"""Training and validation views of the minimal environment's demonstration rows, split by case.

    python -m research.minimal_env.select_rows \\
        --demonstrations output/minimal_env_20261115/demonstrations_train_ieee14 \\
        --output-dir output/minimal_env_20261115/selection_ieee14

Reads ``rows.jsonl`` and ``episodes.jsonl`` of ``build_demonstrations``.

**Cases.**  Every case is taken.  A case whose success path ends repaired by both audits
gives all its rows and those of its recovery paths that end repaired.  The cases the rule
expert solves and the ones it does not are taken alike: the supervised pass has to hold
the ordinary cases as well as the hard ones.

A case the correction tools cannot repair gives the steps of its sequence too: each names
a true target, which is the right decision whatever the tool then made of it.  Its last
step, ``finish unresolved``, is kept only when the final estimate does not pass: there
the prompt shows why the case is not resolved.  Where the final estimate passes, the
prompt cannot tell the state from a repaired one, and a row saying ``unresolved`` on it
would teach a guess, so that row is left out.  ``--solved-only`` leaves such cases out
entirely.

**Split.**  Whole cases, never rows: a seeded share of every class is held out
(``--validation-share``), so no case has rows on both sides.  ``--exclude-roots`` names
cases kept apart for another use; they appear in neither view.  The validation view is a
seeded sample of the held-out cases' rows (``--validation-rows``), balanced over the step
kinds so that the rare ones (``undo``) are seen.

**Options of the second build** (2026-11-21).  ``--skip-demonstrations`` leaves out the
paths of the named kinds (``recovery``, ``refusal``, ``breakers_last``) without a rebuild.
``--repeat-breaker NAME=K`` writes every training row whose target corrects breaker NAME K
times in all (validation untouched): the supervised model of the first build wrote
``CB_2R4_R5`` for ``CB_2R4_2R5`` in 245 of 263 calls, a copy defect of one name that its 36
training targets did not prevent.

Outputs: ``train.jsonl``, ``validation.jsonl``, ``selection_cases.json`` and
``selection_report.json``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SELECTION_CONTRACT = "minimal_row_selection_v1"


def _jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def choose_cases(episodes: Sequence[Mapping[str, Any]], *, keep_unsolved: bool = True,
                 excluded: frozenset[str] | set[str] = frozenset(),
                 skip: frozenset[str] | set[str] = frozenset()) -> dict[str, dict[str, Any]]:
    """Per taken case: its class, whether it is repaired and the tags of the sequences whose rows are taken."""
    cases: dict[str, dict[str, Any]] = {}
    for record in episodes:
        scenario = str(record["scenario_id"])
        if scenario in excluded or str(record["demonstration"]) in skip:
            continue
        repaired = bool(record["solved"]) and bool(record["score"]["strict"]["solved"])
        entry = cases.setdefault(scenario, {"class": record.get("compound_class"), "solved": False, "tags": []})
        if record["demonstration"] == "success":
            entry["solved"] = repaired
            if repaired or keep_unsolved:
                entry["tags"].append(str(record["tag"]))
        elif repaired:
            entry["tags"].append(str(record["tag"]))
    return {scenario: entry for scenario, entry in cases.items() if entry["solved"] or (keep_unsolved and entry["tags"])}


def target_breaker(row: Mapping[str, Any]) -> str | None:
    """The breaker a row's target call corrects, if it is a breaker correction."""
    for call in row["messages"][-1].get("tool_calls") or []:
        function = call.get("function") or {}
        arguments = function.get("arguments") or {}
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        if function.get("name") == "correct_topology" and arguments.get("breaker"):
            return str(arguments["breaker"])
    return None


def parse_repeats(values: Sequence[str] | None) -> dict[str, int]:
    out: dict[str, int] = {}
    for value in values or []:
        name, _, count = str(value).partition("=")
        if not name or not count.isdigit() or int(count) < 1:
            raise ValueError(f"--repeat-breaker wants NAME=K with K >= 1, got {value!r}")
        out[name] = int(count)
    return out


def split_cases(cases: Mapping[str, Mapping[str, Any]], *, share: float, seed: int) -> set[str]:
    """The held-out cases: the same seeded share of every class, at least one of a class that has four or more."""
    by_class: dict[str, list[str]] = defaultdict(list)
    for scenario, entry in cases.items():
        by_class[str(entry["class"])].append(scenario)
    held: set[str] = set()
    for name, members in sorted(by_class.items()):
        members = sorted(members)
        random.Random(f"{int(seed)}:{name}").shuffle(members)
        count = round(float(share) * len(members))
        if share > 0 and count == 0 and len(members) >= 4:
            count = 1
        held.update(members[:count])
    return held


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--demonstrations", required=True, type=Path, help="an output directory of build_demonstrations")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--validation-share", type=float, default=0.06)
    parser.add_argument("--validation-rows", type=int, default=256)
    parser.add_argument("--exclude-roots", type=Path, default=None, help='JSON list of scenario ids, or {"roots": [...]}')
    parser.add_argument("--solved-only", action="store_true", help="leave out the cases the correction tools cannot repair")
    parser.add_argument("--seed", type=int, default=20261115)
    parser.add_argument("--skip-demonstrations", default="", help="comma-separated demonstration kinds to leave out")
    parser.add_argument("--repeat-breaker", action="append", default=[], metavar="NAME=K",
                        help="write the training rows whose target corrects breaker NAME K times in all")
    args = parser.parse_args(argv)
    skip = {item.strip() for item in str(args.skip_demonstrations).split(",") if item.strip()}
    if "success" in skip:
        raise SystemExit("the success paths cannot be left out")
    repeats = parse_repeats(args.repeat_breaker)

    excluded: set[str] = set()
    if args.exclude_roots is not None:
        payload = json.loads(args.exclude_roots.read_text(encoding="utf-8"))
        excluded = {str(item) for item in (payload.get("roots") if isinstance(payload, Mapping) else payload) or []}
    episodes = list(_jsonl(args.demonstrations / "episodes.jsonl"))
    cases = choose_cases(episodes, keep_unsolved=not args.solved_only, excluded=excluded, skip=skip)
    held = split_cases(cases, share=args.validation_share, seed=args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    counts: dict[str, Counter] = {"train": Counter(), "validation_pool": Counter(), "dropped": Counter(), "repeated": Counter()}
    chars: list[int] = []
    pool: dict[str, list[dict[str, Any]]] = defaultdict(list)
    train_path = args.output_dir / "train.jsonl"
    rows_seen = 0
    with train_path.open("w", encoding="utf-8", newline="\n") as train:
        for row in _jsonl(args.demonstrations / "rows.jsonl"):
            rows_seen += 1
            scenario = str(row["scenario_id"])
            entry = cases.get(scenario)
            tag = str(row["example_id"]).split(":")[1]
            if entry is None or tag not in entry["tags"]:
                continue
            labels = row["minimal"]
            if not labels["sequence_solved"] and labels["kind"] == "finish" and labels.get("estimator_passes_at_end", True):
                counts["dropped"]["unresolved_finish_on_a_passing_estimate"] += 1
                continue  # the prompt cannot tell this state from a repaired one
            row["selection"] = {"contract": SELECTION_CONTRACT, "split": "validation" if scenario in held else "train",
                                "class": entry["class"]}
            key = f"{labels['demonstration']}:{labels['kind']}"
            if scenario in held:
                pool[key].append(row)
                counts["validation_pool"][key] += 1
            else:
                copies = repeats.get(target_breaker(row) or "", 1)
                for copy_index in range(copies):
                    written = row if copy_index == 0 else {**row, "example_id": f"{row['example_id']}:repeat{copy_index}"}
                    train.write(json.dumps(written, sort_keys=True, allow_nan=False) + "\n")
                    counts["train"][key] += 1
                    counts["train"][f"tool:{labels['tool']}"] += 1
                    chars.append(sum(len(str(message.get("content") or "")) for message in row["messages"][:2]))
                if copies > 1:
                    counts["repeated"][str(target_breaker(row))] += copies - 1
    # The validation view: the kinds in turn, so a rare kind is not crowded out by the common ones.
    rng = random.Random(int(args.seed))
    for rows in pool.values():
        rng.shuffle(rows)
    validation: list[dict[str, Any]] = []
    kinds = sorted(pool)
    while len(validation) < int(args.validation_rows) and any(pool[kind] for kind in kinds):
        for kind in kinds:
            if pool[kind] and len(validation) < int(args.validation_rows):
                validation.append(pool[kind].pop())
    validation_path = args.output_dir / "validation.jsonl"
    with validation_path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in validation:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")

    by_class: dict[str, Counter] = defaultdict(Counter)
    for scenario, entry in cases.items():
        by_class[str(entry["class"])]["validation_cases" if scenario in held else "train_cases"] += 1
    ordered = sorted(chars)
    report = {
        "contract": SELECTION_CONTRACT, "demonstrations": str(args.demonstrations.resolve()),
        "settings": {"validation_share": args.validation_share, "validation_rows": args.validation_rows,
                     "solved_only": bool(args.solved_only), "seed": args.seed, "excluded_roots": len(excluded),
                     **({"skip_demonstrations": sorted(skip)} if skip else {}),
                     **({"repeat_breaker": repeats} if repeats else {})},
        "rows_read": rows_seen,
        "rows_dropped": dict(sorted(counts["dropped"].items())),
        "cases": {"taken": len(cases), "train": len(cases) - len(held), "validation": len(held),
                  "repaired": sum(1 for entry in cases.values() if entry["solved"]),
                  "beyond_the_tools": sum(1 for entry in cases.values() if not entry["solved"]),
                  "with_a_recovery_path": sum(1 for entry in cases.values() if len(entry["tags"]) > 1),
                  "paths_by_kind": dict(sorted(Counter(
                      tag.rstrip("0123456789") for entry in cases.values() for tag in entry["tags"]).items()))},
        **({"repeated_rows": dict(sorted(counts["repeated"].items()))} if counts["repeated"] else {}),
        "train_rows": sum(value for key, value in counts["train"].items() if not key.startswith("tool:")),
        "train_rows_by_kind": {key: value for key, value in sorted(counts["train"].items()) if not key.startswith("tool:")},
        "train_rows_by_tool": {key[5:]: value for key, value in sorted(counts["train"].items()) if key.startswith("tool:")},
        "validation_rows": len(validation),
        "validation_rows_by_kind": dict(sorted(Counter(
            f"{row['minimal']['demonstration']}:{row['minimal']['kind']}" for row in validation).items())),
        "validation_pool_rows": sum(counts["validation_pool"].values()),
        "train_prompt_characters": ({"median": ordered[len(ordered) // 2], "p95": ordered[int(0.95 * len(ordered))],
                                     "max": ordered[-1]} if ordered else None),
        "by_class": {name: dict(sorted(cell.items())) for name, cell in sorted(by_class.items())},
        "sha256": {"train": _sha256(train_path), "validation": _sha256(validation_path)},
    }
    (args.output_dir / "selection_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                                                          encoding="utf-8", newline="\n")
    (args.output_dir / "selection_cases.json").write_text(json.dumps(
        {"train": sorted(set(cases) - held), "validation": sorted(held)}, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({key: report[key] for key in ("cases", "train_rows", "validation_rows", "train_rows_by_kind")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
