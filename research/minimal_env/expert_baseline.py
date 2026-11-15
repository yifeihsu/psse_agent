"""The rule expert's recorded episodes, replayed through the minimal interface.

    python -m research.minimal_env.expert_baseline \\
        --suite <suite.json> --episodes <rule expert episodes.jsonl> \\
        --output-dir output/minimal_env_20261115/expert_dev300 --workers 20

``--episodes`` is a rollout record of the rule expert in the old environment
(``research.compound_errors.rollout``: one line per case with its compact
trace and its truth-audited outcome).  Each episode is translated step for
step (``psse_env.minimal.expert``) and replayed here.  Two readings:

* **the same tools through either interface**: the replay's strict audit must
  agree with the recorded outcome case by case; a disagreement is listed with
  both sets of reasons;
* **what the interface costs**: the expert's actions in the old interface
  against the new one, with the solves and fits both actually ran.

Outputs: ``episodes.jsonl`` (one replay per case) and ``summary.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from psse_env.minimal.env import MinimalDiagnosisEnv  # noqa: E402
from psse_env.minimal.episode import compact, run_episode  # noqa: E402
from psse_env.minimal.expert import TranslatedExpertPolicy, Untranslatable, translate_trace  # noqa: E402
from research.minimal_env.build_demonstrations import class_of, load_suite  # noqa: E402

BASELINE_CONTRACT = "minimal_expert_replay_v1"
_ENV: MinimalDiagnosisEnv | None = None


def _init_worker(max_actions: int) -> None:
    global _ENV
    _ENV = MinimalDiagnosisEnv(max_actions=int(max_actions))


def _run(payload: Mapping[str, Any]) -> dict[str, Any]:
    assert _ENV is not None
    root, recorded = payload["root"], payload["recorded"]
    base = {"scenario_id": recorded["scenario_id"], "compound_class": class_of(root),
            "recorded": {"success": bool(recorded.get("success")), "actions": int(recorded.get("steps") or len(recorded["trace"])),
                         "wls_calls": int(recorded.get("wls_calls") or 0), "audit_reasons": recorded.get("audit_reasons") or []}}
    try:
        actions = translate_trace(recorded["trace"], breakers=True)
        _ENV.reset(root)
        if not _ENV.breakers:
            actions = translate_trace(recorded["trace"], breakers=False)
    except Untranslatable as error:
        return {**base, "untranslatable": str(error)}
    episode = compact(run_episode(_ENV, root, TranslatedExpertPolicy(actions), keep_observations=False))
    return {**base, "translated_actions": len(actions), "episode": episode}


def summarize(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    totals: Counter = Counter()
    by_class: dict[str, Counter] = defaultdict(Counter)
    disagreements: list[dict[str, Any]] = []
    old_actions: list[int] = []
    new_actions: list[int] = []
    for record in records:
        cell = by_class[str(record["compound_class"])]
        totals["cases"] += 1
        cell["cases"] += 1
        totals["recorded_solved"] += int(record["recorded"]["success"])
        cell["recorded_solved"] += int(record["recorded"]["success"])
        if "untranslatable" in record:
            totals["untranslatable"] += 1
            continue
        score = record["episode"]["score"]
        strict, solved = bool(score["strict"]["solved"]), bool(score["solved"])
        totals["replay_strict_solved"] += int(strict)
        totals["replay_solved"] += int(solved)
        totals["replay_success_declared"] += int(bool(score["success"]))
        totals["replay_false_resolved"] += int(score["declared"] == "resolved" and not solved)
        totals["replay_healthy_damaged"] += int(score["physical"]["damaged"] > 0)
        totals["replay_budget_exhausted"] += int(record["episode"]["outcome"] == "budget_exhausted")
        cell["replay_strict_solved"] += int(strict)
        cell["replay_solved"] += int(solved)
        if strict != record["recorded"]["success"]:
            disagreements.append({"scenario_id": record["scenario_id"], "class": record["compound_class"],
                                  "recorded": record["recorded"]["success"], "recorded_reasons": record["recorded"]["audit_reasons"],
                                  "replay_strict": strict, "replay_reasons": score["strict"]["reasons"]})
        old_actions.append(record["recorded"]["actions"])
        new_actions.append(int(score["actions"]))
        totals["recorded_wls_calls"] += record["recorded"]["wls_calls"]
        for key in ("wls_solves", "substation_solves", "parameter_fits", "measurement_repairs"):
            totals[f"replay_{key}"] += int(score["cost"].get(key) or 0)
    replayed = len(old_actions)
    return {
        "contract": BASELINE_CONTRACT,
        "totals": dict(sorted(totals.items())),
        "outcome_agreement": {"replayed": replayed, "agree": replayed - len(disagreements), "disagree": disagreements},
        "mean_actions": {"old_interface": round(sum(old_actions) / replayed, 2) if replayed else None,
                         "minimal_interface": round(sum(new_actions) / replayed, 2) if replayed else None},
        "by_class": {name: dict(sorted(cell.items())) for name, cell in sorted(by_class.items())},
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--episodes", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-actions", type=int, default=40,
                        help="the old environment's budget, so that no translated episode is cut short")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    recorded = {}
    with args.episodes.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                record = json.loads(line)
                recorded[str(record["scenario_id"])] = record
    payloads = [{"root": root, "recorded": recorded[str(root["execution"]["scenario_id"])]}
                for root in load_suite(args.suite) if str(root["execution"]["scenario_id"]) in recorded]
    if args.limit is not None:
        payloads = payloads[: int(args.limit)]
    if not payloads:
        raise SystemExit("no case of the suite has a recorded episode")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    records: list[dict[str, Any]] = []
    with (args.output_dir / "episodes.jsonl").open("w", encoding="utf-8", newline="\n") as out, \
            ProcessPoolExecutor(max_workers=max(1, int(args.workers)), initializer=_init_worker, initargs=(args.max_actions,)) as pool:
        for done, record in enumerate(pool.map(_run, payloads, chunksize=1), start=1):
            out.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            records.append(record)
            if done % 50 == 0 or done == len(payloads):
                print(f"[expert replay] {done}/{len(payloads)} cases, {time.time() - started:.0f} s", flush=True)
    summary = summarize(records)
    summary.update(suite=str(args.suite.resolve()), episodes=str(args.episodes.resolve()), max_actions=int(args.max_actions),
                   seconds=round(time.time() - started, 1))
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8",
                                                  newline="\n")
    print(json.dumps({**summary["totals"], "agree": summary["outcome_agreement"]["agree"],
                      "mean_actions": summary["mean_actions"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
