"""The baseline register: every finished evaluation a comparison may rest on (workplan of 2026-11-10, ticket 2).

    python -m research.rl.baseline_register add --register research/hpc/rl_improvement_20261110/baseline_register.json \\
        --name arm_b/S1 --run $WORK/out/arm_b_complete --evaluation iteration_000 --config $WORK/configs/arm_b_complete.json
    python -m research.rl.baseline_register show --register research/hpc/rl_improvement_20261110/baseline_register.json
    python -m research.rl.baseline_register compare --register ... --first arm_b/S1 --second arm_a/S1

The workplan asks that every newly completed checkpoint evaluation be
appended with its checkpoint, evaluator, precision, observation mode and
configuration before the next run, because a score quoted without them has
been compared with scores taken another way (the cloning cell's 4-bit arms
against this driver's bf16 evaluations).  An entry holds

* what was evaluated: the policy (an adapter, the rule expert, the audit
  deciding), the run and iteration it belongs to;
* how: the suite and its digest, the action limit and budgets, the
  precision, the observation mode, the prompt limit, the source commit;
* the result: solved, clean, within each budget, steps, the lifecycle
  counts, the process failures;
* the digests of the evaluation's own files, so the entry can be checked
  against them later.

``compare`` names what two entries differ in beside their policy.  Two
entries on another suite, action limit or precision are not comparable; an
observation mode or prompt limit that differs is the stated treatment of a
matched pair and is reported as such.  An entry is never overwritten
silently: adding a name that exists with other digests is refused unless
``--replace`` says so.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

REGISTER_CONTRACT = "rl_baseline_register_v1"
#: What must agree for two entries to be compared at all.
COMPARABILITY_FIELDS = ("suite_sha256", "roots", "max_steps", "precision")
#: What may differ between matched arms, and is then the treatment.
TREATMENT_FIELDS = ("observation_mode", "max_input_tokens")


def file_sha256(path: Path) -> str | None:
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def empty_register() -> dict[str, Any]:
    return {"contract": REGISTER_CONTRACT, "entries": {}}


def load(path: Path) -> dict[str, Any]:
    if not Path(path).is_file():
        return empty_register()
    register = json.loads(Path(path).read_text(encoding="utf-8"))
    if register.get("contract") != REGISTER_CONTRACT or not isinstance(register.get("entries"), Mapping):
        raise ValueError(f"{path} is not a baseline register ({REGISTER_CONTRACT})")
    return register


def save(register: Mapping[str, Any], path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(register, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def results_from_summary(summary: Mapping[str, Any], *, clean: int | None = None, healthy_component_corrupted: int | None = None) -> dict[str, Any]:
    """The register's result block from a driver evaluation summary (``evaluation.json``)."""

    lifecycle = summary.get("lifecycle") if isinstance(summary.get("lifecycle"), Mapping) else {}
    return {
        "roots": summary.get("roots"),
        "solved": summary.get("truth_audited_task_success"),
        "clean": clean if clean is not None else summary.get("clean_success"),
        "solved_within_actions": dict(summary.get("solved_within_actions") or {}),
        "mean_steps": summary.get("mean_steps"),
        "mean_reward": summary.get("mean_reward"),
        "healthy_component_corrupted": (healthy_component_corrupted if healthy_component_corrupted is not None
                                        else summary.get("healthy_component_corrupted")),
        "lifecycle": {key: lifecycle.get(key) for key in ("decisions", "declines", "true_target_declined", "wrong_target_committed",
                                                         "wrong_target_declined")},
        "invalid_actions": summary.get("invalid_action_count"),
        "false_commits": summary.get("false_commit_count"),
        "loop_episodes": summary.get("loop_episodes"),
        "prompt_limit_refusals": summary.get("prompt_limit_refusals"),
    }


def entry_from_evaluation(run: Path, evaluation: str, *, name: str, config: Mapping[str, Any] | None = None,
                          source_commit: str | None = None, suite_sha256: str | None = None, note: str | None = None) -> dict[str, Any]:
    """An entry read from a run directory's ``<evaluation>/evaluation.json`` and episode records.

    ``evaluation`` is ``baseline`` or ``iteration_NNN``.  ``config`` is the
    run's config, for what the summary does not carry (the suite, the
    budgets); the summary's own record wins where both have a field.
    """

    from research.rl.scorecard import load_policy, summarize_policy

    directory = Path(run) / evaluation
    summary_path, episodes_path = directory / "evaluation.json", directory / "evaluation_episodes.jsonl"
    if not summary_path.is_file() or not episodes_path.is_file():
        raise FileNotFoundError(f"{directory} holds no finished evaluation (evaluation.json and evaluation_episodes.jsonl)")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))["summary"]
    if summary.get("skipped"):
        raise ValueError(f"{directory} was not evaluated: {summary['skipped']}")
    config = dict(config or {})
    card = summarize_policy(load_policy(episodes_path))
    limits = summary.get("prompt_limits") if isinstance(summary.get("prompt_limits"), Mapping) else {}
    suite = config.get("development_suite")
    if suite_sha256 is None and suite:
        suite_sha256 = file_sha256(Path(suite))
    return {
        "name": name,
        "kind": "learner",
        "policy": {"adapter": summary.get("adapter"), "baseline": bool(summary.get("baseline")), "iteration": summary.get("iteration")},
        "run": str(run),
        "series": dict(summary.get("series") or config.get("series") or {}),
        "how": {
            "suite": suite, "suite_sha256": suite_sha256, "roots": summary.get("roots"),
            "max_steps": summary.get("max_steps", config.get("max_steps")),
            "budgets": sorted(int(budget) for budget in (summary.get("solved_within_actions") or {})),
            "precision": summary.get("precision") or ("4bit" if config.get("load_in_4bit") else "bf16" if config else None),
            "observation_mode": summary.get("observation_mode") or config.get("observation_mode") or "bounded_history_v1",
            "max_input_tokens": limits.get("max_input_tokens", config.get("max_input_tokens")),
            "decoding": "greedy",
            "evaluator": "psse_env.rl.train greedy evaluation (paired evaluation contract)",
            "source_commit": source_commit,
        },
        "results": results_from_summary(summary, clean=card["clean"], healthy_component_corrupted=card["healthy_component_corrupted_episodes"]),
        "files": {"evaluation_sha256": file_sha256(summary_path), "episodes_sha256": file_sha256(episodes_path)},
        "evaluated_at": datetime.fromtimestamp(summary_path.stat().st_mtime, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "note": note,
    }


def add(register: dict[str, Any], entry: Mapping[str, Any], *, replace: bool = False) -> dict[str, Any]:
    """Put ``entry`` in the register; an entry of the same name with other files is refused unless ``replace``."""

    name = str(entry["name"])
    existing = register["entries"].get(name)
    if existing is not None and not replace:
        if existing.get("files") == entry.get("files") and existing.get("results") == entry.get("results"):
            return register
        raise ValueError(f"the register already has {name!r} with other files or results; --replace overwrites it")
    register["entries"][name] = dict(entry)
    return register


def compare(first: Mapping[str, Any], second: Mapping[str, Any]) -> dict[str, Any]:
    """What two entries differ in beside the policy: what forbids the comparison, and what is the treatment."""

    how_a, how_b = first.get("how") or {}, second.get("how") or {}
    blocking = {field: [how_a.get(field), how_b.get(field)] for field in COMPARABILITY_FIELDS
                if how_a.get(field) is not None and how_b.get(field) is not None and how_a.get(field) != how_b.get(field)}
    unknown = sorted(field for field in COMPARABILITY_FIELDS if how_a.get(field) is None or how_b.get(field) is None)
    treatment = {field: [how_a.get(field), how_b.get(field)] for field in TREATMENT_FIELDS if how_a.get(field) != how_b.get(field)}
    return {"first": first.get("name"), "second": second.get("name"), "comparable": not blocking, "not_comparable_because": blocking,
            "not_recorded": unknown, "differs_as_treatment": treatment}


def markdown(register: Mapping[str, Any], *, title: str = "Baseline register") -> str:
    def cell(value: Any) -> str:
        if value is None:
            return "n/a"
        if isinstance(value, float):
            return f"{value:.2f}"
        return str(value)

    lines = [f"# {title}", "",
             "| Entry | Kind | Solved | Clean | Within 16 / 20 | Steps | Declines (true / wrong) | Wrong kept | Observation | Precision | "
             "Suite | Evaluated |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name in sorted(register["entries"]):
        entry = register["entries"][name]
        how, results = entry.get("how") or {}, entry.get("results") or {}
        lifecycle = results.get("lifecycle") or {}
        within = results.get("solved_within_actions") or {}
        suite = how.get("suite_sha256")
        lines.append(
            f"| {name} | {entry.get('kind')} | {cell(results.get('solved'))} of {cell(results.get('roots'))} | {cell(results.get('clean'))} | "
            f"{cell(within.get('16'))} / {cell(within.get('20'))} | {cell(results.get('mean_steps'))} | "
            f"{cell(lifecycle.get('declines'))} ({cell(lifecycle.get('true_target_declined'))} / {cell(lifecycle.get('wrong_target_declined'))}) | "
            f"{cell(lifecycle.get('wrong_target_committed'))} | {cell(how.get('observation_mode'))} | {cell(how.get('precision'))} | "
            f"{suite[:8] if suite else 'n/a'} | {cell(entry.get('evaluated_at'))} |")
    notes = [(name, entry.get("note")) for name, entry in sorted(register["entries"].items()) if entry.get("note")]
    if notes:
        lines += ["", "Notes:", ""] + [f"- **{name}**: {note}" for name, note in notes]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    adder = commands.add_parser("add", help="append a finished evaluation of a run")
    adder.add_argument("--register", required=True, type=Path)
    adder.add_argument("--name", required=True)
    adder.add_argument("--run", required=True, type=Path, help="the run's output directory")
    adder.add_argument("--evaluation", required=True, help="baseline or iteration_NNN")
    adder.add_argument("--config", default=None, type=Path, help="the run's config (suite, budgets, limits)")
    adder.add_argument("--source-commit", default=None)
    adder.add_argument("--note", default=None)
    adder.add_argument("--replace", action="store_true")
    shower = commands.add_parser("show", help="the register as a table")
    shower.add_argument("--register", required=True, type=Path)
    comparer = commands.add_parser("compare", help="whether two entries can be compared, and what they differ in")
    comparer.add_argument("--register", required=True, type=Path)
    comparer.add_argument("--first", required=True)
    comparer.add_argument("--second", required=True)
    args = parser.parse_args(argv)
    register = load(args.register)
    if args.command == "add":
        config = json.loads(args.config.read_text(encoding="utf-8")) if args.config else None
        entry = entry_from_evaluation(args.run, args.evaluation, name=args.name, config=config, source_commit=args.source_commit,
                                      note=args.note)
        save(add(register, entry, replace=args.replace), args.register)
        print(json.dumps(entry, sort_keys=True))
        return 0
    if args.command == "show":
        print(markdown(register), flush=True)
        return 0
    missing = [name for name in (args.first, args.second) if name not in register["entries"]]
    if missing:
        raise SystemExit(f"not in the register: {missing}")
    result = compare(register["entries"][args.first], register["entries"][args.second])
    print(json.dumps(result, indent=1, sort_keys=True))
    return 0 if result["comparable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
