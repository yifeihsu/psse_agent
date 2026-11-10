"""Seed the series' baseline register from a read of the first arm on the cluster.

    python -m research.hpc.rl_improvement_20261110.seed_register

``first_arm_read.json`` is that read as it came back (a read-only script on
torch, 2026-11-10T14:38Z): the first arm's manifest, the summary and file
digests of each of its evaluations, and the rule expert's and the audit's
records on the same 300 cases.  This script turns it into register entries,
so no figure in the register was typed by hand.  Evaluations of the new
series are appended on the cluster with ``research.rl.baseline_register add``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.rl.baseline_register import add, empty_register, results_from_summary, save  # noqa: E402

CELL = Path(__file__).resolve().parent
READ_FILE = CELL / "first_arm_read.json"
REGISTER_FILE = CELL / "baseline_register.json"
FIRST_ARM_RUN = "/scratch/yx3882/compound_rl_20261107/out/ieee14_v1"
DEVELOPMENT_SUITE = "/scratch/yx3882/compound_rl_20261107/data/suites/ieee14_dev300.json"
#: The first arm's evaluations by the name the project uses for the checkpoint.
CHECKPOINTS = {"baseline": "S0", "iteration_000": "S1", "iteration_001": "S2", "iteration_002": "S3", "iteration_003": "S4"}
#: The cell of 2026-11-07 ran every phase with RESEARCH_MAX_INPUT_TOKENS=32768 (its rl.sbatch).
FIRST_ARM_INPUT_LIMIT = 32768


def learner_entry(name: str, evaluation: str, record: Mapping[str, Any], read: Mapping[str, Any]) -> dict[str, Any]:
    summary = record["summary"]
    return {
        "name": name,
        "kind": "learner",
        "policy": {"adapter": summary.get("adapter"), "baseline": bool(summary.get("baseline")), "iteration": summary.get("iteration")},
        "run": FIRST_ARM_RUN,
        "series": {"name": "compound_rl_20261107", "arm": "crl-ieee14"},
        "how": {
            "suite": DEVELOPMENT_SUITE, "suite_sha256": read["suites_sha256"]["ieee14_dev300.json"], "roots": summary.get("roots"),
            "max_steps": 40, "budgets": sorted(int(budget) for budget in summary.get("solved_within_actions") or {}),
            "precision": "bf16", "observation_mode": "bounded_history_v1", "max_input_tokens": FIRST_ARM_INPUT_LIMIT,
            "decoding": "greedy", "evaluator": "psse_env.rl.train greedy evaluation (paired evaluation contract)",
            "source_commit": read["source"],
        },
        "results": results_from_summary(summary, clean=record["clean"], healthy_component_corrupted=record["healthy_component_corrupted"]),
        "files": {"evaluation_sha256": record["sha256"], "episodes_sha256": record["episodes_sha256"]},
        "evaluated_at": record["mtime_utc"],
        "note": ("the cloned start adapter under the first arm's loader" if evaluation == "baseline" else
                 "the first arm: every update was two steps of a new optimizer over halves of the batch"),
    }


def map_entry(name: str, kind: str, record: Mapping[str, Any], read: Mapping[str, Any], *, path: str, note: str) -> dict[str, Any]:
    return {
        "name": name,
        "kind": kind,
        "policy": {"record": path},
        "run": None,
        "series": {},
        "how": {
            "suite": DEVELOPMENT_SUITE, "suite_sha256": read["suites_sha256"]["ieee14_dev300.json"], "roots": record["on_dev300"],
            "max_steps": 40, "budgets": [], "precision": None, "observation_mode": None, "max_input_tokens": None,
            "decoding": "rule policy", "evaluator": "compound study rollout, read as a map of root to result", "source_commit": None,
        },
        "results": {"roots": record["on_dev300"], "solved": record["solved"], "clean": record["clean"], "solved_within_actions": {},
                    "mean_steps": None, "mean_reward": None, "healthy_component_corrupted": None,
                    "lifecycle": {"decisions": None, "declines": None, "true_target_declined": None,
                                  "wrong_target_committed": record["wrong_kept"], "wrong_target_declined": None},
                    "invalid_actions": None, "false_commits": None, "loop_episodes": None, "prompt_limit_refusals": None,
                    "solved_on_all_cases": record["solved_all"], "all_cases": record["entries"]},
        "files": {"evaluation_sha256": record["sha256"], "episodes_sha256": None},
        "evaluated_at": None,
        "note": note,
    }


def build(read: Mapping[str, Any]) -> dict[str, Any]:
    register = empty_register()
    register["seeded_from"] = {"file": READ_FILE.name, "read_at": read["read_at"], "source_commit_on_torch": read["source"],
                               "first_arm_manifest": read["manifest"]}
    for evaluation, record in read["evaluations"].items():
        add(register, learner_entry(f"first_arm/{CHECKPOINTS[evaluation]}", evaluation, record, read))
    add(register, map_entry("rule_expert/route_teacher", "expert", read["teacher_map.json"], read,
                            path="/scratch/yx3882/compound_cloning_20261106/data/teacher_map.json",
                            note="the rule expert with its route comparison on: the project's reference. In the learner's "
                                 "environment (route comparison off) the same rules solve 184 of the 300 "
                                 "(docs/rl_beyond_teacher_plan_20261105.md, point 47); that run is not in this register"))
    add(register, map_entry("audit_deciding", "audit", read["audit_map.json"], read,
                            path="/scratch/yx3882/compound_cloning_20261106/data/audit_map.json",
                            note="the rule expert with the offline audit deciding every lifecycle choice: a restricted diagnostic "
                                 "reference, not a bound on what a policy can reach"))
    return register


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--read", default=READ_FILE, type=Path)
    parser.add_argument("--output", default=REGISTER_FILE, type=Path)
    args = parser.parse_args(argv)
    register = build(json.loads(args.read.read_text(encoding="utf-8")))
    save(register, args.output)
    print(json.dumps({name: {"solved": entry["results"]["solved"], "clean": entry["results"]["clean"]}
                      for name, entry in register["entries"].items()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
