"""How many cases does one forced step repair?  The one-step forks of an adapter's greedy episodes, with no update.

    python -m research.minimal_env.fork_probe --suite SUITE.json --adapter ADAPTER_DIR --output-dir OUT \\
        [--cases CASES.json] [--unrepaired-in EVALUATION_DIR] [--sample N] [--argument-samples K] [--shard S/N]

For every case the adapter's greedy episode is run, and at every state of it every other tool is forced once: the
model writes the arguments, continues greedily, and the branch is scored like any episode
(``psse_env.minimal.forks``).  The summary counts the cases the greedy episode leaves unrepaired and, of them,
the ones some single forced step repairs.

What the count is: a labelled upper bound on what one round of the RL plan's stage B can teach (the update trains
the better branch's action; whether training makes the model take it is the run's question).  It is a training-time
measurement on recorded cases with the truth as the scorer, not a way to diagnose a case: an evaluated episode
never branches.

``--cases`` keeps the named cases of the suite (a JSON list of case ids, an object keyed by them, or a file of
``research.minimal_env.truth_sequences``).  ``--unrepaired-in`` names a folder ``research.minimal_env.evaluate`` (or
the RL driver) wrote for this adapter and keeps the cases it left unrepaired; with ``--reference`` (``{case:
{"expert": bool, "truth": bool}}``) only those a truth-built sequence repairs.  ``--sample N`` keeps a seeded
sample of N of what is left, in the suite's order.  ``--argument-samples K`` adds K branches per tool whose forced
call is sampled at temperature 1 (other objects of one tool).

One file per case under ``OUT/cases``, so a preempted job loses the case it was on; ``--shard S/N`` forks every
N-th case from S and writes no summary; a call without ``--shard`` forks what is left and writes ``OUT/summary.json``.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import types
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from psse_env.minimal.env import DEFAULT_MAX_ACTIONS, DEFAULT_TABLE_VERSIONS, MinimalDiagnosisEnv  # noqa: E402
from psse_env.minimal.forks import STATUS_RAN, ForkSettings, ForkStore, fork_roots, summarize  # noqa: E402
from psse_env.rl.rollout import scenario_root  # noqa: E402
from research.minimal_env import evaluate  # noqa: E402

PROBE_CONTRACT = "minimal_fork_probe_v1"


def unrepaired_cases(folder: Path, reference: Mapping[str, Mapping[str, Any]] | None = None) -> set[str]:
    """The cases an evaluation left unrepaired; with a reference, those of them a truth-built sequence repairs."""
    path = Path(folder) / "evaluation.json"
    if not path.is_file():
        raise ValueError(f"{folder} holds no finished evaluation (evaluation.json)")
    episodes = json.loads(path.read_text(encoding="utf-8")).get("episodes") or []
    left = {str(entry["scenario_id"]) for entry in episodes if not entry.get("solved")}
    if reference is not None:
        left = {case for case in left if (reference.get(case) or {}).get("truth")}
    return left


def select_cases(roots: Sequence[Mapping[str, Any]], *, cases: Path | None = None, unrepaired_in: Path | None = None,
                 reference: Mapping[str, Mapping[str, Any]] | None = None, sample: int | None = None,
                 seed: int = 0) -> list[dict[str, Any]]:
    """The cases to fork, in the suite's order."""
    kept = [dict(root) for root in roots]
    if cases is not None:
        kept = evaluate.keep_cases(kept, cases)
    if unrepaired_in is not None:
        left = unrepaired_cases(unrepaired_in, reference)
        kept = [root for root in kept if scenario_root(root) in left]
    if sample is not None:
        if sample < 1:
            raise ValueError("--sample must be positive")
        if sample < len(kept):
            chosen = set(random.Random(int(seed)).sample(range(len(kept)), int(sample)))
            kept = [root for index, root in enumerate(kept) if index in chosen]
    return kept


def greedy_policy_factory(adapter: str, *, model_choice: str, load_in_4bit: bool) -> Callable[[], Any]:
    return evaluate.model_policy_factory(adapter, model_choice=model_choice, load_in_4bit=load_in_4bit)


def sampling_policy_factory(adapter: str, *, model_choice: str, load_in_4bit: bool) -> Callable[[int], Any]:
    """``factory(seed)``: the adapter sampling at temperature 1 with no cut, seeded; the model is the greedy policy's."""

    def factory(seed: int) -> Any:
        from psse_env.dagger.research_policy_factory import GenerationOptions
        from psse_env.minimal.policy import minimal_policy_factory
        from psse_env.research_models import get_research_model_spec

        spec = get_research_model_spec(model_choice)
        return minimal_policy_factory(
            adapter, base_model=spec.model_id, base_revision=spec.revision, load_in_4bit=load_in_4bit, local_files_only=True,
            architecture=spec.architecture,
            generation=GenerationOptions(do_sample=True, temperature=1.0, top_p=1.0, top_k=None, seed=int(seed)))

    return factory


def prefix_check_factory(adapter: str, *, model_choice: str) -> Callable[[Any], Any]:
    """``factory(policy)``: the check that a forced call's given tokens start the call's own rendering (the driver's)."""

    def factory(policy: Any) -> Any:
        from psse_env.research_models import get_research_model_spec
        from research.minimal_env import rl_train

        spec = get_research_model_spec(model_choice)
        config = types.SimpleNamespace(base_model=str(spec.model_id), base_revision=str(spec.revision), architecture=str(spec.architecture))
        return rl_train.fork_prefix_check(config, adapter, policy)

    return factory


def case_line(payload: Mapping[str, Any]) -> dict[str, Any]:
    """One case of the summary: what the greedy episode did and the first forced step that repairs it, if any."""
    record = payload["record"]
    own = record["own"]
    repairing = sorted((branch for branch in record["branches"]
                        if branch["status"] == STATUS_RAN and branch.get("executed") and branch.get("solved")),
                       key=lambda branch: (branch["state"], -float(branch["return"])))
    line: dict[str, Any] = {
        "scenario_id": record["scenario_id"], "class": record.get("class"), "own_solved": bool(own["solved"]),
        "own_declared": own.get("declared"), "own_actions": [str((step.get("action") or {}).get("tool")) if isinstance(step.get("action"), Mapping)
                                                             else "unreadable" for step in own["steps"]],
        "states": record["states"], "branches": len(record["branches"]),
        "repaired_by_one_forced_step": bool(repairing),
        "better": [{"state": item["state"], "tool": item["tool"], "action": item["action"], "margin": item["margin"],
                    "gain": item["gain"], "solved": item["solved"], "arguments": item.get("arguments")}
                   for item in record.get("better") or []],
    }
    if repairing:
        first = repairing[0]
        line["first_repairing_branch"] = {"state": first["state"], "tool": first["tool"], "action": first["action"],
                                          "arguments": first.get("arguments"), "at_the_last_state": first["state"] == record["states"] - 1,
                                          "continuation": [str((action or {}).get("tool")) if isinstance(action, Mapping) else "unreadable"
                                                           for action in first.get("continuation") or []]}
    return line


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, sort_keys=True, allow_nan=False, indent=1)
        stream.write("\n")
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-choice", default="12b")
    parser.add_argument("--load-in-4bit", action="store_true", help="run in 4-bit (default: bf16)")
    parser.add_argument("--max-actions", type=int, default=DEFAULT_MAX_ACTIONS)
    parser.add_argument("--table-versions", type=int, default=DEFAULT_TABLE_VERSIONS,
                        help="versions whose tables the prompt shows (0: every version); must be what the adapter was trained on")
    parser.add_argument("--cases", type=Path, default=None, help="fork these cases of the suite only")
    parser.add_argument("--unrepaired-in", type=Path, default=None,
                        help="a finished evaluation of this adapter: fork the cases it left unrepaired only")
    parser.add_argument("--reference", type=Path, default=None,
                        help="with --unrepaired-in: keep the cases a truth-built sequence repairs only")
    parser.add_argument("--sample", type=int, default=None, help="a seeded sample of N of the selected cases")
    parser.add_argument("--seed", type=int, default=20261122)
    parser.add_argument("--argument-samples", type=int, default=0,
                        help="further branches per tool whose forced call is sampled at temperature 1")
    parser.add_argument("--shard", default=None, help="S/N: fork every N-th case starting at S, write no summary")
    args = parser.parse_args(argv)

    shard = evaluate.parse_shard(args.shard)
    reference = json.loads(args.reference.read_text(encoding="utf-8")) if args.reference is not None else None
    if reference is not None and args.unrepaired_in is None:
        raise SystemExit("--reference narrows --unrepaired-in: name the evaluation too")
    roots = select_cases(evaluate.load_suite(args.suite), cases=args.cases, unrepaired_in=args.unrepaired_in, reference=reference,
                         sample=args.sample, seed=args.seed)
    if not roots:
        raise SystemExit("no case to fork")
    settings = ForkSettings(enabled=True, argument_samples=int(args.argument_samples))
    settings.validate()
    table_versions = int(args.table_versions) or None
    env = MinimalDiagnosisEnv(max_actions=int(args.max_actions), table_versions=table_versions)
    store = ForkStore(args.output_dir)
    started = time.time()

    def progress(record: Mapping[str, Any]) -> None:
        print(f"[fork-probe{'' if shard is None else f' shard {shard[0]}/{shard[1]}'}] {json.dumps(dict(record), sort_keys=True)}", flush=True)

    mine = [root for index, root in enumerate(roots) if shard is None or index % shard[1] == shard[0]]
    options = dict(model_choice=args.model_choice, load_in_4bit=bool(args.load_in_4bit))
    ran = fork_roots(
        mine, store=store, env_factory=lambda: env, policy_factory=greedy_policy_factory(str(args.adapter), **options), settings=settings,
        argument_policy_factory=sampling_policy_factory(str(args.adapter), **options) if settings.argument_samples else None,
        base_seed=int(args.seed), canonical_factory=prefix_check_factory(str(args.adapter), model_choice=args.model_choice),
        progress=progress)
    if shard is not None:
        print(json.dumps({"shard": list(shard), "forked": ran, "seconds": round(time.time() - started, 1)}))
        return 0
    payloads = []
    for root in roots:
        case = scenario_root(root)
        if not store.has(case):
            raise SystemExit(f"{case} has no fork file after the run")
        payloads.append(store.read(case))
    records = [payload["record"] for payload in payloads]
    summary = summarize(records)
    _write_json(args.output_dir / "summary.json", {
        "contract": PROBE_CONTRACT, "suite": str(args.suite), "adapter": str(args.adapter),
        "settings": {"model_choice": args.model_choice, "load_in_4bit": bool(args.load_in_4bit), "decoding": "greedy",
                     "max_actions": int(args.max_actions), "table_versions": table_versions,
                     "argument_samples": int(args.argument_samples), "seed": int(args.seed),
                     "max_input_tokens": os.environ.get("RESEARCH_MAX_INPUT_TOKENS"),
                     "max_new_tokens": os.environ.get("RESEARCH_MAX_NEW_TOKENS")},
        "selection": {"cases": str(args.cases) if args.cases else None,
                      "unrepaired_in": str(args.unrepaired_in) if args.unrepaired_in else None,
                      "reference": str(args.reference) if args.reference else None, "sample": args.sample, "selected": len(roots)},
        "summary": summary,
        "cases": [case_line(payload) for payload in payloads],
    })
    totals = summary["totals"]
    print(json.dumps({"cases": totals.get("cases", 0), "own_episode_unrepaired": totals.get("own_episode_unrepaired", 0),
                      "unrepaired_cases_one_forced_step_repairs": totals.get("unrepaired_cases_one_forced_step_repairs", 0),
                      "branches": totals.get("branches", 0), "forced_prefixes_checked": totals.get("forced_prefixes_checked", 0),
                      "forced_prefixes_canonical": totals.get("forced_prefixes_canonical", 0), "seconds": summary["seconds"]},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
