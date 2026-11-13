"""Training selection for a supervised pass in the open-rules environment (2026-11-12).

    python -m research.compound_errors.open_rules_selection \\
        --truth-rows T/rows.jsonl --truth-episodes T/episodes.jsonl \\
        --demonstration-rows D/rows.jsonl --demonstration-episodes D/episodes.jsonl \\
        --demonstration-cases s0_cases.json --output-dir out/

Two behaviour policies, both rendered in the open-rules prompt, one per case:

* the truth-built sequence (``truth_sequences``): the true corrections in the order that works, each kept.  It is
  the only source that solves the cases the rule expert cannot, and it never tries a wrong target, so alone it
  teaches "keep everything";
* the demonstration of the cloning pass (``commit_reader_loop run --decider audit --export-rows`` with
  ``--env-option open_rules=1``): the rule expert, with the truth audit declining the accepted candidates on healthy
  components.  It holds the other side: wrong targets tried, verified and rolled back.

**Cases.**  A case takes the truth-built rows when its sequence passes the truth audit and the demonstration fails
it (``--truth-roots demonstration-fails``, the default), or already when the rule expert alone fails it
(``--truth-roots rule-expert-fails``: the cases the demonstration solves by declining then lose their decline rows).
Every such case is taken.  The demonstration's cases are the ones named in ``--demonstration-cases`` (the cases of
the cloning pass the start adapter was trained on, with their split), less those the truth-built rows took.  A case
never has both policies' choices: they part at the first choice of a target, where the same prompt would carry two
labels.  One kind of demonstration row survives in a named case the truth-built rows took: its rollbacks of
candidates on healthy components (declines of accepted ones, rollbacks of rejected ones).  They sit on states the
sequence never visits, their label does not depend on how the episode ended, and the start adapter was trained on
them (``--no-demonstration-contrast`` leaves them out).

**Rows.**  Inside a chosen case every row is kept, except:

* a truth-built correction whose target the row's own prompt does not name (``target_in_evidence`` absent): it can
  only be guessed;
* a demonstration rollback of a rejected or inconclusive candidate on a truly faulted component: with the commit
  rule lifted, keeping it is the right step, so the rollback is the wrong label here;
* a failed action that repeats an identical failed action on the same active state, and the tools named in
  ``--thin`` kept with the given probability by a hash of the row's id (the demonstration with the cloning pass's
  ids and seed, so its kept rows are the same ones).  Lifecycle and correction rows are never thinned.

**Reserved roots.**  ``--reserved-roots`` names training roots kept apart for another use (the RL series' fork
probe draws its diagnostic validation and decision panel from them, and no arm of the series trains on them).  A
truth-built case that is reserved and was not a case of the cloning pass is left out, so the new pass sees no
reserved root the start adapter's pass did not see.

**Split.**  A case of ``--demonstration-cases`` keeps its split, also when the truth-built rows take it; of the
other truth-built cases a seeded share is held out whole.  The validation view holds, from the held-out cases, every
contrast row (declines, rollbacks of rejected candidates, rejected candidates kept, corrections that needed a lifted
rule), a seeded sample of the other lifecycle rows up to half of ``--validation-rows``, and a seeded sample of the
rest for the other half.

Every written row carries ``cloning_selection``: source, class, split and what kind of contrast row it is.  Outputs:
``selection.train_view.jsonl``, ``selection.validation.jsonl``, ``selection_cases.json``, ``selection_report.json``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SELECT_CONTRACT = "compound_open_rules_selection_v1"
SOURCE_TRUTH, SOURCE_DEMONSTRATION = "truth_built_sequence", "demonstration"
COMMIT, ROLLBACK = "commit_state", "rollback_state"
LIFECYCLE_TOOLS = (COMMIT, ROLLBACK)
CORRECTION_TOOLS = frozenset({"correct_measurements", "correct_parameters", "correct_topology"})
NEVER_THINNED = frozenset({COMMIT, ROLLBACK}) | CORRECTION_TOOLS
REJECTED = frozenset({"REJECT", "INCONCLUSIVE"})
TRUTH_ROOTS = ("demonstration-fails", "rule-expert-fails")
#: Why a row of a chosen case is left out.
LEFT_ABSENT, LEFT_TRUE_ROLLBACK = "target absent from the prompt", "true target rejected and rolled back"
LEFT_REPEAT, LEFT_THINNED, LEFT_FAILED = "repeated failed action", "thinned", "failed action of a truth-built sequence"


def stream(path: Path) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as lines:
        for line in lines:
            if line.strip():
                yield json.loads(line)


def keep_probability(row_id: str, seed: int) -> float:
    """A number in [0, 1) fixed by the row's id and the seed (the cloning pass's own thinning hash)."""
    digest = hashlib.sha256(f"{seed}:{row_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2 ** 64


def repeated_failures(trace: Sequence[Mapping[str, Any]]) -> set[int]:
    """Steps whose action failed after an identical action had already failed on the same active state."""
    out: set[int] = set()
    seen: set[tuple[str, str, int]] = set()
    state = 0
    for step in trace:
        tool = str(step.get("tool"))
        if tool == COMMIT and step.get("status") == "success":
            state += 1
        if step.get("status") != "success":
            signature = (tool, json.dumps(step.get("arguments"), sort_keys=True), state)
            if signature in seen:
                out.add(int(step["step"]))
            seen.add(signature)
    return out


def lifecycle_of_corrections(record: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    """The lifecycle step that disposed of each successful correction: {that step: the correction's record}."""
    trace = list(record.get("trace") or [])
    out: dict[int, Mapping[str, Any]] = {}
    for correction in record.get("corrections") or []:
        if correction.get("execution_status") != "success":
            continue
        for entry in trace[int(correction["step"]) + 1:]:
            tool = str(entry.get("tool"))
            if tool in CORRECTION_TOOLS:
                break
            if tool in LIFECYCLE_TOOLS:
                out[int(entry["step"])] = correction
                break
    return out


def read_cases(path: Path, *, source: str) -> dict[str, dict[str, Any]]:
    """What the selection reads of each episode record of one source."""
    cases: dict[str, dict[str, Any]] = {}
    for record in stream(path):
        trace = list(record.get("trace") or [])
        disposed = lifecycle_of_corrections(record)
        declines = {int(d["step"]) for d in record.get("gate_decisions") or [] if d.get("declined") and d.get("step") is not None}
        steps: dict[int, dict[str, Any]] = {}
        for entry in trace:
            if entry.get("step") is None:
                continue
            index = int(entry["step"])
            tool = str(entry.get("tool"))
            correction = disposed.get(index)
            verdict = str(correction.get("verdict")) if correction is not None else None
            true_target = correction.get("target_is_true") if correction is not None else None
            steps[index] = {
                "tool": tool,
                "status": entry.get("status"),
                "kind": entry.get("kind"),
                "open_rules_used": [str(rule) for rule in entry.get("open_rules_used") or []],
                "target_in_evidence": entry.get("target_in_evidence"),
                "verdict": verdict,
                "target_is_true": true_target,
                "decline": tool == ROLLBACK and index in declines,
                "rejected_rollback": tool == ROLLBACK and verdict in REJECTED,
                "kept_rejected": tool == COMMIT and verdict in REJECTED,
            }
        cases[str(record["scenario_id"])] = {
            "source": source,
            "class": str(record.get("class")),
            "success": bool(record.get("success")),
            "rule_expert_solves": record.get("rule_expert_solves"),
            "steps": steps,
            "repeats": repeated_failures(trace),
        }
    return cases


def assign(truth: Mapping[str, Mapping[str, Any]], demonstration: Mapping[str, Mapping[str, Any]],
           named: Mapping[str, Sequence[str]], *, truth_roots: str, validation: float, seed: int,
           demonstration_contrast: bool = True, reserved: frozenset[str] = frozenset()) -> dict[str, dict[str, Any]]:
    """{case: {"source", "split", "class"}}: one source per case.

    A truth-built case that is also a named demonstration case carries ``demonstration_contrast``: the
    demonstration's rollbacks of candidates on healthy components are kept beside the sequence's rows.  A
    ``reserved`` case is taken only where the cloning pass had it (``named``).
    """
    if truth_roots not in TRUTH_ROOTS:
        raise ValueError(f"truth_roots must be one of {TRUTH_ROOTS}, got {truth_roots!r}")
    split_of = {str(case): "train" for case in named.get("train") or []}
    split_of.update({str(case): "validation" for case in named.get("validation") or []})
    missing = sorted(case for case in split_of if case not in demonstration)
    if missing:
        raise ValueError(f"{len(missing)} named demonstration cases have no episode record, e.g. {missing[:3]}")

    def takes_truth(case: str) -> bool:
        entry = truth[case]
        if not entry["success"]:
            return False
        if truth_roots == "rule-expert-fails":
            if entry["rule_expert_solves"] is None:
                raise ValueError(f"{case}: the truth-built episodes do not say whether the rule expert solves the case")
            return entry["rule_expert_solves"] is False
        if case not in demonstration:
            raise ValueError(f"{case}: no demonstration episode to compare with")
        return not demonstration[case]["success"]

    chosen: dict[str, dict[str, Any]] = {}
    fresh: list[str] = []
    for case in sorted(truth):
        if case in reserved and case not in split_of:
            continue
        if takes_truth(case):
            chosen[case] = {"source": SOURCE_TRUTH, "split": split_of.get(case), "class": truth[case]["class"]}
            if case not in split_of:
                fresh.append(case)
            elif demonstration_contrast:
                chosen[case]["demonstration_contrast"] = True
    rng = random.Random(f"{seed}:truth")
    rng.shuffle(fresh)
    held = set(fresh[: int(round(float(validation) * len(fresh)))])
    for case in fresh:
        chosen[case]["split"] = "validation" if case in held else "train"
    for case, split in sorted(split_of.items()):
        if case not in chosen:
            chosen[case] = {"source": SOURCE_DEMONSTRATION, "split": split, "class": demonstration[case]["class"]}
    return chosen


def read_reserved(path: Path | None) -> frozenset[str]:
    """The reserved roots of a file: ``{"roots": [...]}`` (the RL series' reserved_roots.json) or a list of ids."""
    if path is None:
        return frozenset()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    roots = payload.get("roots") if isinstance(payload, Mapping) else payload
    if not isinstance(roots, list):
        raise ValueError(f"{path}: no list of roots")
    return frozenset(str(item if not isinstance(item, Mapping) else item.get("scenario_id") or item.get("id")) for item in roots)


def row_id(source: str, case: str, step: int) -> str:
    """The id the thinning hashes: the cloning pass's own for the demonstration, so its kept rows are the same ones."""
    return f"{case}:{step}" if source == SOURCE_DEMONSTRATION else f"{case}:truth:{step}"


def left_out(source: str, case: str, step: int, entry: Mapping[str, Any], info: Mapping[str, Any], *, thin: Mapping[str, float],
             seed: int, keep_absent: bool, keep_true_rollbacks: bool) -> str | None:
    """Why this row of a chosen case is left out, or None when it is kept."""
    tool = str(entry["tool"])
    if source == SOURCE_TRUTH:
        if entry["status"] != "success":
            return LEFT_FAILED
        if entry.get("kind") == "correction" and entry.get("target_in_evidence") == "absent" and not keep_absent:
            return LEFT_ABSENT
    else:
        if step in info["repeats"]:
            return LEFT_REPEAT
        if entry["rejected_rollback"] and entry["target_is_true"] is True and not keep_true_rollbacks:
            return LEFT_TRUE_ROLLBACK
    if tool in NEVER_THINNED or tool not in thin:
        return None
    return None if keep_probability(row_id(source, case, step), seed) < float(thin[tool]) else LEFT_THINNED


def contrast_kind(source: str, entry: Mapping[str, Any]) -> str | None:
    """What a row shows of the keep-or-roll-back choice and of the lifted rules; None for the other rows."""
    if entry["decline"]:
        return "accepted candidate on a healthy component, rolled back"
    if entry["rejected_rollback"]:
        return ("rejected candidate on a healthy component, rolled back" if entry["target_is_true"] is False
                else "rejected candidate on a faulted component, rolled back")
    if entry["kept_rejected"]:
        return "rejected candidate on a faulted component, kept"
    if source == SOURCE_TRUTH and entry.get("kind") == "correction" and entry["open_rules_used"]:
        return "correction that needed a lifted rule"
    if source == SOURCE_TRUTH and entry.get("kind") == "handoff" and entry["open_rules_used"]:
        return "handoff that needed a lifted rule"
    return None


def select_rows(path: Path, source: str, cases: Mapping[str, Mapping[str, Any]], chosen: Mapping[str, Mapping[str, Any]], *,
                thin: Mapping[str, float], seed: int, keep_absent: bool, keep_true_rollbacks: bool, train_stream: Any,
                validation_pool: list[dict[str, Any]], counts: Counter, left: Counter, system: str = "ieee14") -> None:
    """Stream one source's rows; write the kept training rows, collect the kept rows of the held-out cases."""
    for row in stream(path):
        case = str(row.get("scenario_id"))
        if case not in chosen:
            continue
        # A row of the case's own source, or a demonstration row in a truth-built case that keeps the
        # demonstration's rollbacks of candidates on healthy components.
        contrast_only = chosen[case]["source"] != source
        if contrast_only and not (source == SOURCE_DEMONSTRATION and chosen[case].get("demonstration_contrast")):
            continue
        info = cases[case]
        label = row.get("cloning") if isinstance(row.get("cloning"), Mapping) else {}
        step = int(label["step"])
        entry = info["steps"][step]
        if str(label.get("tool")) != entry["tool"]:
            raise ValueError(f"{row.get('example_id')}: the row's tool {label.get('tool')!r} is not the episode's {entry['tool']!r}")
        split = chosen[case]["split"]
        if contrast_only:
            if not (entry["decline"] or (entry["rejected_rollback"] and entry["target_is_true"] is False)):
                continue
        else:
            reason = left_out(source, case, step, entry, info, thin=thin, seed=seed, keep_absent=keep_absent,
                              keep_true_rollbacks=keep_true_rollbacks)
            if reason is not None:
                left[f"{source} | {split} | {reason}"] += 1
                continue
        contrast = contrast_kind(source, entry)
        row["cloning_selection"] = {
            # ``system`` as the cloning pass's selection names it (its smoke and outcome tools group rows by it).
            "contract": SELECT_CONTRACT, "system": system, "source": source, "case_source": chosen[case]["source"],
            "contrast_only": contrast_only,
            "class": info["class"], "split": split, "case_solved_by_source": info["success"], "decline_row": bool(entry["decline"]),
            "contrast": contrast, "verdict": entry["verdict"], "open_rules_used": list(entry["open_rules_used"]),
        }
        action = "decline" if entry["decline"] else entry["tool"]
        counts[f"rows | {source} | {split}"] += 1
        if contrast_only:
            counts[f"demonstration rollbacks kept in truth-built cases | {split}"] += 1
        counts[f"action | {source} | {split} | {action}"] += 1
        if contrast is not None:
            counts[f"contrast | {split} | {contrast}"] += 1
        for rule in entry["open_rules_used"]:
            counts[f"lifted rule | {source} | {split} | {rule}"] += 1
        if split == "train":
            train_stream.write(json.dumps(row, sort_keys=True) + "\n")
        else:
            validation_pool.append(row)


def validation_view(pool: Sequence[dict[str, Any]], *, rows: int, seed: int) -> list[dict[str, Any]]:
    """Every contrast row of the held-out cases, other lifecycle rows up to half the budget, the rest for the other half."""
    rng = random.Random(f"{seed}:validation")
    ordered = sorted(pool, key=lambda row: str(row["example_id"]))
    contrast = [row for row in ordered if row["cloning_selection"]["contrast"] is not None]
    lifecycle = [row for row in ordered if row["cloning_selection"]["contrast"] is None
                 and str(row["cloning"]["tool"]) in LIFECYCLE_TOOLS]
    others = [row for row in ordered if row["cloning_selection"]["contrast"] is None
              and str(row["cloning"]["tool"]) not in LIFECYCLE_TOOLS]
    rng.shuffle(lifecycle)
    rng.shuffle(others)
    half = int(rows) // 2
    view = contrast + lifecycle[: max(0, half - len(contrast))]
    view += others[: max(0, int(rows) - len(view))]
    return sorted(view, key=lambda row: str(row["example_id"]))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--truth-rows", required=True, type=Path)
    parser.add_argument("--truth-episodes", required=True, type=Path)
    parser.add_argument("--demonstration-rows", required=True, type=Path)
    parser.add_argument("--demonstration-episodes", required=True, type=Path)
    parser.add_argument("--demonstration-cases", required=True, type=Path, help='JSON {"train": [cases], "validation": [cases]}')
    parser.add_argument("--truth-roots", choices=TRUTH_ROOTS, default=TRUTH_ROOTS[0])
    parser.add_argument("--thin", action="append", default=None, metavar="TOOL=FRACTION", help="keep this share of a tool's rows")
    parser.add_argument("--keep-absent-targets", action="store_true", help="keep truth-built corrections whose target the prompt does not name")
    parser.add_argument("--keep-true-rejected-rollbacks", action="store_true",
                        help="keep the demonstration's rollbacks of rejected candidates on faulted components")
    parser.add_argument("--no-demonstration-contrast", action="store_true",
                        help="a named case the truth-built rows took keeps none of the demonstration's rows")
    parser.add_argument("--reserved-roots", type=Path, default=None,
                        help='JSON of training roots kept apart ({"roots": [ids]} or a list): taken only where the cloning pass had them')
    parser.add_argument("--validation-cases", type=float, default=0.06, help="share of the truth-built cases without a named split held out")
    parser.add_argument("--validation-rows", type=int, default=192)
    parser.add_argument("--seed", type=int, default=20261106)
    parser.add_argument("--system", default="ieee14")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    thin: dict[str, float] = {}
    for item in args.thin or []:
        tool, _separator, value = item.partition("=")
        if tool.strip() in NEVER_THINNED:
            raise SystemExit(f"{tool} rows are never thinned")
        thin[tool.strip()] = float(value)
    truth = read_cases(args.truth_episodes, source=SOURCE_TRUTH)
    demonstration = read_cases(args.demonstration_episodes, source=SOURCE_DEMONSTRATION)
    named = json.loads(args.demonstration_cases.read_text(encoding="utf-8"))
    reserved = read_reserved(args.reserved_roots)
    chosen = assign(truth, demonstration, named, truth_roots=args.truth_roots, validation=args.validation_cases, seed=args.seed,
                    demonstration_contrast=not args.no_demonstration_contrast, reserved=reserved)
    named_cases = {str(case) for case in (named.get("train") or [])} | {str(case) for case in (named.get("validation") or [])}
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    counts: Counter = Counter()
    left: Counter = Counter()
    pool: list[dict[str, Any]] = []
    with (out / "selection.train_view.jsonl").open("w", encoding="utf-8", newline="\n") as train_stream:
        for source, rows_path, cases in ((SOURCE_DEMONSTRATION, args.demonstration_rows, demonstration),
                                         (SOURCE_TRUTH, args.truth_rows, truth)):
            select_rows(rows_path, source, cases, chosen, thin=thin, seed=args.seed, keep_absent=args.keep_absent_targets,
                        keep_true_rollbacks=args.keep_true_rejected_rollbacks, train_stream=train_stream,
                        validation_pool=pool, counts=counts, left=left, system=args.system)
    view = validation_view(pool, rows=args.validation_rows, seed=args.seed)
    with (out / "selection.validation.jsonl").open("w", encoding="utf-8", newline="\n") as validation_stream:
        for row in view:
            validation_stream.write(json.dumps(row, sort_keys=True) + "\n")
    for row in view:
        selection = row["cloning_selection"]
        counts[f"validation view | {selection['source']}"] += 1
        if selection["contrast"] is not None:
            counts[f"validation view | contrast | {selection['contrast']}"] += 1
    by_case: dict[str, Counter] = defaultdict(Counter)
    classes: dict[str, Counter] = defaultdict(Counter)
    for case, entry in chosen.items():
        by_case[f"{entry['source']} | {entry['split']}"]["cases"] += 1
        info = (truth if entry["source"] == SOURCE_TRUTH else demonstration)[case]
        by_case[f"{entry['source']} | {entry['split']}"]["solved by their source"] += int(info["success"])
        by_case[f"{entry['source']} | {entry['split']}"]["with a decline"] += int(any(step["decline"] for step in info["steps"].values()))
        by_case[f"{entry['source']} | {entry['split']}"]["keeping the demonstration's contrast rows"] += int(bool(entry.get("demonstration_contrast")))
        classes[entry["class"]][f"{entry['source']} | {entry['split']}"] += 1
    (out / "selection_cases.json").write_text(json.dumps(chosen, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    report = {
        "contract": SELECT_CONTRACT, "system": args.system, "seed": args.seed, "truth_roots": args.truth_roots, "thin": thin,
        "keep_absent_targets": args.keep_absent_targets, "keep_true_rejected_rollbacks": args.keep_true_rejected_rollbacks,
        "demonstration_contrast": not args.no_demonstration_contrast,
        "reserved_roots": {
            "file": str(args.reserved_roots) if args.reserved_roots else None, "roots": len(reserved),
            "in_the_selection": sum(1 for case in chosen if case in reserved),
            "in_the_selection_and_the_cloning_pass": sum(1 for case in chosen if case in reserved and case in named_cases),
            "left_out": sorted(case for case in reserved if case not in chosen and case not in named_cases and case in truth
                               and truth[case]["success"] and (case not in demonstration or not demonstration[case]["success"])),
        },
        "validation_cases": args.validation_cases, "validation_rows": args.validation_rows,
        "inputs": {"truth_rows": str(args.truth_rows), "truth_episodes": str(args.truth_episodes),
                   "demonstration_rows": str(args.demonstration_rows), "demonstration_episodes": str(args.demonstration_episodes),
                   "demonstration_cases": str(args.demonstration_cases)},
        "cases": {name: dict(entry) for name, entry in sorted(by_case.items())},
        "cases_by_class": {name: dict(sorted(entry.items())) for name, entry in sorted(classes.items())},
        "rows_train": counts[f"rows | {SOURCE_TRUTH} | train"] + counts[f"rows | {SOURCE_DEMONSTRATION} | train"],
        "rows_held_out": len(pool), "rows_validation_view": len(view),
        "counts": {key: counts[key] for key in sorted(counts)},
        "rows_left_out": {key: left[key] for key in sorted(left)},
    }
    (out / "selection_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("cases", "rows_train", "rows_held_out", "rows_validation_view", "rows_left_out")},
                     indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
