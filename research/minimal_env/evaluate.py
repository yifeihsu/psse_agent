"""A trained adapter, evaluated greedily on a suite in the minimal environment.

    python -m research.minimal_env.evaluate --suite SUITE.json --adapter ADAPTER_DIR --output-dir OUT \\
        [--reference REFERENCE.json] [--shard S/N] [--cases CASES.json]

The model reads the observation of ``psse_env.minimal.observation`` and answers with one
tool call per step (``psse_env.minimal.policy``); the environment executes it; the
finished episode is scored against the truth (``psse_env.minimal.scoring``).  Greedy
decoding, bf16 unless ``--load-in-4bit``.

Each case's episode is written to ``OUT/episodes/<case>.json`` as soon as it ends, so a
preempted job loses at most the case it was on and a later start skips what is there.
``--shard S/N`` evaluates every N-th case starting at S and writes no summary: several
processes can share a card, each with its own shard.  A call without ``--shard`` runs
whatever is still missing and writes ``OUT/evaluation.json``.

``--reference`` names a JSON file ``{case: {"expert": bool, "truth": bool}}``: whether the
rule expert's recorded episode and the truth-built sequence repair the case.  With it the
summary splits the cases the way the review of 2026-11-15 reads a result: the ones the
rule expert solves, the ones only the truth-built sequence repairs, and the rest.

``RESEARCH_MAX_INPUT_TOKENS`` must hold the prompts (about 9,000 tokens at the median,
14,000 and more late in a long episode); a prompt over it ends the episode undeclared
(``prompt_limit``), which the summary counts.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from psse_env.minimal.env import DEFAULT_MAX_ACTIONS, DEFAULT_TABLE_VERSIONS, MinimalDiagnosisEnv  # noqa: E402
from psse_env.minimal.episode import OUTCOME_PROMPT_LIMIT, compact, run_episode  # noqa: E402
from research.minimal_env import repair_breakdown  # noqa: E402

EVALUATION_CONTRACT = "minimal_model_evaluation_v1"
REFERENCE_GROUPS = repair_breakdown.REFERENCE_GROUPS


def load_suite(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        payload = payload.get("scenarios") or payload.get("roots") or []
    return [dict(item) for item in payload]


def case_id(root: Mapping[str, Any]) -> str:
    return str((root.get("execution") or {}).get("scenario_id") or "")


def class_of(root: Mapping[str, Any]) -> str:
    return str(((root.get("audit") or {}).get("compound") or {}).get("class") or (root.get("grouping") or {}).get("scenario_family") or "")


def keep_cases(roots: Sequence[Mapping[str, Any]], path: Path) -> list[dict[str, Any]]:
    """The cases of ``roots`` that the file at ``path`` names, in the suite's order; a named case the suite lacks is refused."""
    from research.minimal_env.truth_sequences import case_ids

    wanted = case_ids(path)
    kept = [dict(root) for root in roots if case_id(root) in wanted]
    absent = wanted - {case_id(root) for root in kept}
    if absent:
        raise ValueError(f"{len(absent)} cases of {path} are not in the suite (first: {sorted(absent)[0]})")
    return kept


def parse_shard(value: str | None) -> tuple[int, int] | None:
    if not value:
        return None
    match = re.fullmatch(r"(\d+)/(\d+)", value.strip())
    if not match or int(match.group(2)) < 1 or int(match.group(1)) >= int(match.group(2)):
        raise ValueError(f"--shard must be S/N with 0 <= S < N, got {value!r}")
    return int(match.group(1)), int(match.group(2))


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name) or "case"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def evaluate_cases(roots: Sequence[Mapping[str, Any]], *, policy_factory: Callable[[], Any],
                   env_factory: Callable[[], MinimalDiagnosisEnv], episodes_dir: Path,
                   shard: tuple[int, int] | None = None,
                   progress: Callable[[Mapping[str, Any]], None] | None = None) -> int:
    """Run every case of the shard that has no episode file yet.  Returns how many were run.

    The policy is built only when a case is left to run, so a start that
    finds everything done loads no model.
    """
    episodes_dir.mkdir(parents=True, exist_ok=True)
    pending = [root for index, root in enumerate(roots)
               if (shard is None or index % shard[1] == shard[0])
               and not (episodes_dir / f"{_safe_name(case_id(root))}.json").is_file()]
    if not pending:
        return 0
    policy = policy_factory()
    env = env_factory()
    for number, root in enumerate(pending, start=1):
        episode = compact(run_episode(env, root, policy, keep_observations=False))
        episode["compound_class"] = class_of(root)
        _write_json(episodes_dir / f"{_safe_name(case_id(root))}.json", episode)
        if progress is not None:
            score = episode["score"]
            progress({"case": case_id(root), "done": number, "of": len(pending), "solved": bool(score["solved"]),
                      "declared": score["declared"], "actions": score["actions"], "seconds": episode["seconds"]["episode"]})
    return len(pending)


def read_episodes(roots: Sequence[Mapping[str, Any]], episodes_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """The stored episodes in the suite's order, and the cases that have none yet."""
    records, missing = [], []
    for root in roots:
        path = episodes_dir / f"{_safe_name(case_id(root))}.json"
        if path.is_file():
            records.append(json.loads(path.read_text(encoding="utf-8")))
        else:
            missing.append(case_id(root))
    return records, missing


def _percentile(values: Sequence[float], share: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(share * len(ordered)))]


def summarize(records: Sequence[Mapping[str, Any]], reference: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Counts over the evaluated episodes.  ``solved`` is the object-level audit, ``strict_solved`` the old one.

    ``repairs`` is what lies under ``solved`` (``research.minimal_env.repair_breakdown``): the episodes that
    repaired every error, some of them or none, the errors repaired by kind, and how the unsolved episodes ended.
    """
    totals: Counter = Counter()
    by_class: dict[str, Counter] = defaultdict(Counter)
    by_reference: dict[str, Counter] = {name: Counter() for name in REFERENCE_GROUPS}
    tools: Counter = Counter()
    prompt_tokens: list[float] = []
    rewards: list[float] = []
    actions: list[int] = []
    for record in records:
        score = record["score"]
        solved, strict = bool(score["solved"]), bool(score["strict"]["solved"])
        declared = score.get("declared")
        totals["cases"] += 1
        totals["solved"] += int(solved)
        totals["strict_solved"] += int(strict)
        totals["solved_and_declared_resolved"] += int(bool(score["success"]))
        totals["solved_but_not_declared"] += int(solved and declared != "resolved")
        totals["false_resolved"] += int(declared == "resolved" and not solved)
        totals["declared_resolved"] += int(declared == "resolved")
        totals["declared_unresolved"] += int(declared == "unresolved")
        totals["unresolved_declared_on_an_unsolved_case"] += int(declared == "unresolved" and not solved)
        totals["budget_exhausted"] += int(record.get("outcome") == "budget_exhausted")
        totals["prompt_limit"] += int(record.get("outcome") == OUTCOME_PROMPT_LIMIT)
        totals["estimator_passes_at_end"] += int(bool(score.get("estimator_passes")))
        totals["episodes_with_a_damaged_healthy_object"] += int(score["physical"]["damaged"] > 0)
        totals["damaged_healthy_objects"] += int(score["physical"]["damaged"])
        totals["true_errors"] += int(score["physical"]["true_errors"])
        totals["true_errors_repaired"] += int(score["physical"]["repaired"])
        totals["tool_errors"] += int(score.get("tool_errors") or 0)
        totals["episodes_with_a_tool_error"] += int((score.get("tool_errors") or 0) > 0)
        totals["corrections_in_effect"] += int(score.get("corrections_in_effect") or 0)
        for key in ("wls_solves", "substation_solves", "parameter_fits", "measurement_repairs"):
            totals[key] += int((score.get("cost") or {}).get(key) or 0)
        actions.append(int(score.get("actions") or 0))
        rewards.append(float((record.get("reward") or {}).get("total") or 0.0))
        for step in record.get("steps") or []:
            action = step.get("action") or {}
            tools[str(action.get("tool"))] += 1
            if (step.get("result") or {}).get("status") not in ("ok", None):
                totals[f"error:{(step.get('result') or {}).get('error_code')}"] += 1
            metrics = step.get("policy_metrics") or {}
            if metrics.get("original_prompt_tokens"):
                prompt_tokens.append(float(metrics["original_prompt_tokens"]))
            totals["steps_cut_at_max_new_tokens"] += int(bool(metrics.get("hit_max_new_tokens")))
        cell = by_class[str(record.get("compound_class"))]
        cell["cases"] += 1
        cell["solved"] += int(solved)
        cell["strict_solved"] += int(strict)
        if reference is not None:
            known = reference.get(str(record.get("scenario_id")))
            if known is not None:
                group = REFERENCE_GROUPS[0] if known.get("expert") else REFERENCE_GROUPS[1] if known.get("truth") else REFERENCE_GROUPS[2]
                by_reference[group]["cases"] += 1
                by_reference[group]["solved"] += int(solved)
                by_reference[group]["strict_solved"] += int(strict)
    summary: dict[str, Any] = {
        "totals": dict(sorted(totals.items())),
        "mean_actions": round(statistics.mean(actions), 2) if actions else None,
        "mean_reward": round(statistics.mean(rewards), 4) if rewards else None,
        "actions_by_tool": dict(sorted(tools.items())),
        "prompt_tokens": {"median": _percentile(prompt_tokens, 0.5), "p95": _percentile(prompt_tokens, 0.95),
                          "max": max(prompt_tokens) if prompt_tokens else None, "steps": len(prompt_tokens)},
        "by_class": {name: dict(sorted(cell.items())) for name, cell in sorted(by_class.items())},
        "repairs": repair_breakdown.breakdown(records, reference),
    }
    if reference is not None:
        summary["by_reference"] = {name: dict(sorted(cell.items())) for name, cell in by_reference.items()}
    return summary


def evaluation_payload(records: Sequence[Mapping[str, Any]], *, suite: str, adapter: str, settings: Mapping[str, Any],
                       reference: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """What ``evaluation.json`` holds: the settings, the summary and one line per case.

    A case's line carries how far its episode got (errors injected, repaired, healthy objects damaged), so two
    evaluations can be compared case by case from their ``evaluation.json`` alone.
    """
    return {
        "contract": EVALUATION_CONTRACT, "suite": str(suite), "adapter": str(adapter), "settings": dict(settings),
        "summary": summarize(records, reference),
        "episodes": [{"scenario_id": record["scenario_id"], "class": record.get("compound_class"),
                      "solved": bool(record["score"]["solved"]), "strict_solved": bool(record["score"]["strict"]["solved"]),
                      "declared": record["score"].get("declared"), "outcome": record.get("outcome"),
                      "actions": [str((step.get("action") or {}).get("tool")) for step in record.get("steps") or []],
                      "reward": (record.get("reward") or {}).get("total"),
                      **{key: value for key, value in repair_breakdown.case_counts(record).items() if key != "group"}}
                     for record in records],
    }


def model_policy_factory(adapter: str, *, model_choice: str, load_in_4bit: bool) -> Callable[[], Any]:
    def factory() -> Any:
        from psse_env.minimal.policy import minimal_policy_factory
        from psse_env.research_models import get_research_model_spec

        spec = get_research_model_spec(model_choice)
        return minimal_policy_factory(adapter, base_model=spec.model_id, base_revision=spec.revision,
                                      load_in_4bit=load_in_4bit, local_files_only=True, architecture=spec.architecture)

    return factory


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-choice", default="12b")
    parser.add_argument("--load-in-4bit", action="store_true", help="evaluate in 4-bit (default: bf16)")
    parser.add_argument("--max-actions", type=int, default=DEFAULT_MAX_ACTIONS)
    parser.add_argument("--table-versions", type=int, default=DEFAULT_TABLE_VERSIONS,
                        help="versions whose tables the prompt shows (0: every version); must be what the adapter was trained on")
    parser.add_argument("--reference", type=Path, default=None)
    parser.add_argument("--shard", default=None, help="S/N: evaluate every N-th case starting at S, write no summary")
    parser.add_argument("--limit", type=int, default=None, help="the first N cases only")
    parser.add_argument("--cases", type=Path, default=None,
                        help="evaluate these cases of the suite only: a JSON list of case ids, an object keyed by them, or a "
                             "file of research.minimal_env.truth_sequences")
    args = parser.parse_args(argv)

    shard = parse_shard(args.shard)
    roots = load_suite(args.suite)
    if args.cases is not None:
        roots = keep_cases(roots, args.cases)
    if args.limit is not None:
        roots = roots[: int(args.limit)]
    if not roots:
        raise SystemExit("no case to evaluate")
    table_versions = int(args.table_versions) or None
    episodes_dir = args.output_dir / "episodes"
    started = time.time()

    def progress(record: Mapping[str, Any]) -> None:
        print(f"[evaluate{'' if shard is None else f' shard {shard[0]}/{shard[1]}'}] {json.dumps(dict(record), sort_keys=True)}", flush=True)

    ran = evaluate_cases(
        roots,
        policy_factory=model_policy_factory(str(args.adapter), model_choice=args.model_choice, load_in_4bit=args.load_in_4bit),
        env_factory=lambda: MinimalDiagnosisEnv(max_actions=int(args.max_actions), table_versions=table_versions),
        episodes_dir=episodes_dir, shard=shard, progress=progress,
    )
    if shard is not None:
        print(json.dumps({"shard": list(shard), "ran": ran, "seconds": round(time.time() - started, 1)}))
        return 0
    records, missing = read_episodes(roots, episodes_dir)
    if missing:
        raise SystemExit(f"{len(missing)} cases have no episode after the run: {missing[:5]}")
    reference = json.loads(args.reference.read_text(encoding="utf-8")) if args.reference is not None else None
    payload = evaluation_payload(
        records, suite=str(args.suite), adapter=str(args.adapter), reference=reference,
        settings={"model_choice": args.model_choice, "load_in_4bit": bool(args.load_in_4bit), "decoding": "greedy",
                  "max_actions": int(args.max_actions), "table_versions": table_versions,
                  "max_input_tokens": os.environ.get("RESEARCH_MAX_INPUT_TOKENS"),
                  "max_new_tokens": os.environ.get("RESEARCH_MAX_NEW_TOKENS")})
    summary = payload["summary"]
    _write_json(args.output_dir / "evaluation.json", payload)
    print(json.dumps({**summary["totals"], "mean_actions": summary["mean_actions"], "mean_reward": summary["mean_reward"],
                      "repairs": summary["repairs"]["episodes"],
                      **({"by_reference": summary["by_reference"]} if reference is not None else {})}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
