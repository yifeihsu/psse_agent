"""Audit saved rollouts without sampling or updating a model.

python -m research.rl.signal_report --store <iteration>/trajectories --output <report.json>
Optional --suite supplies offline stratum labels; --greedy-evaluation compares
to a frozen adapter evaluated on those *same training roots*. No audit labels
or report fields are passed back to the policy.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from psse_env.rl.advantages import group_advantages
from psse_env.rl.reward import RewardWeights, episode_reward, load_expert_reference
from psse_env.rl.rollout import Trajectory, TrajectoryStore, distinct_trajectories
from research.rl.hard_draw_report import root_stratum

CORRECTIONS = {"correct_parameters", "correct_measurements", "correct_topology"}
REFERENCE_KEYS = {"state_id", "candidate_state_id", "parent_state_id"}


def semantic_actions(trajectory: Trajectory) -> tuple[str, ...]:
    """Ignore JSON formatting and episode-local state names, retain targets."""
    return tuple(json.dumps({"tool": step.tool, "arguments": {
        key: value for key, value in (step.action.get("arguments") or {}).items()
        if key not in REFERENCE_KEYS}}, sort_keys=True) for step in trajectory.steps)


def first_correction(trajectory: Trajectory) -> str | None:
    for step, signature in zip(trajectory.steps, semantic_actions(trajectory)):
        if step.tool in CORRECTIONS:
            return signature
    return None


def group_signal(group: Sequence[Trajectory], *, weights: RewardWeights | None = None,
                 reference_calls: int | None = None, greedy: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if not group:
        raise ValueError("empty rollout group")
    rewards = [episode_reward(t.evaluation, family=t.family, expert_reference_calls=reference_calls,
                              weights=weights).as_dict() if weights is not None else t.reward for t in group]
    totals = [float(r["total"]) for r in rewards]
    successes = [bool(t.evaluation.get("truth_audited_task_success")) for t in group]
    credits = [float(r.get("credit", 0.0)) for r in rewards]
    decisive = [ok and r.get("basis") == "counterfactual_resolution" and credit == 1.0
                for ok, r, credit in zip(successes, rewards, credits)]
    advantages = group_advantages(totals)
    penalty_signatures = {tuple(r.get(key) for key in (
        "false_commits", "false_finalizations", "healthy_component_corrupted", "loop_detected", "invalid_actions"))
        for r in rewards}
    safety_signatures = {tuple(r.get(key) for key in (
        "false_commits", "false_finalizations", "healthy_component_corrupted")) for r in rewards}
    outcome_signatures = {(ok, r.get("basis"), credit) for ok, r, credit in zip(successes, rewards, credits)}
    semantic = {semantic_actions(t) for t in group}
    corrections = {value for t in group if (value := first_correction(t)) is not None}
    comparison: dict[str, Any] = {"available": greedy is not None}
    if greedy is not None:
        baseline_reward = greedy.get("reward") or {}
        # Reweighting a compact evaluation lacks the full acquisition trace.
        # Do not compare a newly weighted sample to an old-weight baseline.
        if weights is None and baseline_reward.get("total") is not None:
            comparison["better_reward_sample"] = max(totals) > float(baseline_reward["total"]) + 1e-6
        comparison["solves_greedy_failure"] = any(successes) and not greedy.get("truth_audited_task_success", False)
        if baseline_reward.get("credit") is not None:
            comparison["higher_credit_sample"] = max(credits) > float(baseline_reward["credit"])
    return {
        "root": group[0].root, "family": group[0].family, "samples": len(group),
        "fully_paired": all(t.pairing_ok for t in group),
        "token_distinct": distinct_trajectories(group), "semantic_distinct": len(semantic),
        "first_correction_targets": len(corrections),
        "reward_spread": max(totals) - min(totals), "reward_std": advantages.std,
        "informative": advantages.informative,
        "mixed_success": len(set(successes)) > 1, "mixed_decisive": len(set(decisive)) > 1,
        "mixed_credit": len(set(credits)) > 1,
        "mixed_safety": len(safety_signatures) > 1,
        "diagnostic_or_safety_signal": len(outcome_signatures) > 1 or len(safety_signatures) > 1,
        "cost_only_signal": advantages.informative and len(outcome_signatures) == 1 and len(penalty_signatures) == 1,
        "all_fail": not any(successes), "all_success": all(successes),
        "all_fail_zero_signal": not any(successes) and not advantages.informative,
        "invalid_action_samples": sum(int(r.get("invalid_actions") or 0) > 0 for r in rewards),
        "rewards": totals, "credits": credits,
        "advantages": {
            "grpo_group": list(advantages.advantages),
            "grpo_none": list(group_advantages(totals, scale="none").advantages),
            "rloo_none": list(group_advantages(totals, method="rloo", scale="none").advantages),
        },
        "greedy_comparison": comparison,
    }


def summarize(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    count = len(records)
    flags = ("informative", "mixed_success", "mixed_decisive", "mixed_credit", "mixed_safety", "diagnostic_or_safety_signal", "cost_only_signal",
             "all_fail", "all_success", "all_fail_zero_signal", "fully_paired")
    result: dict[str, Any] = {"groups": count, **{key: sum(bool(r[key]) for r in records) for key in flags}}
    result["semantic_diverse_groups"] = sum(r["semantic_distinct"] > 1 for r in records)
    result["first_correction_diverse_groups"] = sum(r["first_correction_targets"] > 1 for r in records)
    result["informative_fraction"] = result["informative"] / count if count else None
    result["mixed_decisive_fraction"] = result["mixed_decisive"] / count if count else None
    result["greedy_comparisons"] = sum(r["greedy_comparison"]["available"] for r in records)
    result["groups_solving_greedy_failure"] = sum(bool(r["greedy_comparison"].get("solves_greedy_failure")) for r in records)
    return result


def analyze_store(store: TrajectoryStore, *, strata: Mapping[str, str] | None = None,
                  weights: RewardWeights | None = None, expert_reference: Mapping[str, int] | None = None,
                  greedy: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    records = []
    by_stratum = defaultdict(list)
    for root, group in store.iter_groups():
        record = group_signal(group, weights=weights, reference_calls=(expert_reference or {}).get(root),
                              greedy=(greedy or {}).get(root))
        record["stratum"] = (strata or {}).get(root, record["family"])
        records.append(record)
        by_stratum[record["stratum"]].append(record)
    return {"contract": "rl_saved_signal_audit_v1", "rescored": weights is not None,
            "summary": summarize(records), "by_stratum": {key: summarize(value) for key, value in sorted(by_stratum.items())},
            "by_root": records}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", required=True)
    parser.add_argument("--suite")
    parser.add_argument("--reward-config", help="Full RL config or reward weights JSON; analysis only")
    parser.add_argument("--expert-reference")
    parser.add_argument("--greedy-evaluation", help="evaluation_episodes.jsonl on the same training roots")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if not (Path(args.store) / "groups").is_dir():
        raise SystemExit("trajectory groups directory does not exist")
    strata = {}
    if args.suite:
        suite = json.loads(Path(args.suite).read_text())
        if isinstance(suite, dict):
            suite = suite.get("scenarios") or suite.get("rows") or []
        strata = {str(row["execution"]["scenario_id"]): root_stratum(row) for row in suite}
    weights = None
    if args.reward_config:
        payload = json.loads(Path(args.reward_config).read_text())
        weights = RewardWeights.from_mapping(payload.get("reward", payload))
        if weights.tool_regret_cost and not args.expert_reference:
            raise SystemExit("rescoring tool regret requires --expert-reference")
    greedy = {}
    if args.greedy_evaluation:
        path = Path(args.greedy_evaluation)
        if path.suffix == ".jsonl":
            episodes = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        else:
            payload = json.loads(path.read_text())
            episodes = payload.get("episodes", []) if isinstance(payload, dict) else payload
        if not episodes:
            raise SystemExit("greedy evaluation has no episode records; use evaluation_episodes.jsonl")
        greedy = {str(row["scenario_id"]): row for row in episodes}
    reference = load_expert_reference(args.expert_reference)
    store = TrajectoryStore(args.store)
    if weights is not None and weights.tool_regret_cost and set(store.roots()) - set(reference):
        raise SystemExit("expert reference is missing roots needed to rescore tool regret")
    report = analyze_store(store, strata=strata, weights=weights, expert_reference=reference, greedy=greedy)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(report["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
