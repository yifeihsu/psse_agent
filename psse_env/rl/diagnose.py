"""Where the learning signal of one iteration's stored rollouts is.

    python -m psse_env.rl.diagnose --iteration out/hard_v1/iteration_000
    python -m psse_env.rl.diagnose --store out/hard_v1/iteration_000/trajectories --output report.json

A group-relative update learns only from root groups whose sampled episodes
scored differently.  For every stored group this reports whether the samples
differ at all, whether their rewards differ, what the difference is made of
(a success flip; a penalty such as a false commit, a loop or an invalid
action; or only the call count), at which step the samples first part, and
how peaked the sampling was.  Beside the advantages the configured scale
gives it puts the unscaled ones, because a group standardized by its own
standard deviation gives a one-call difference the same weight as a success
flip.  Nothing here changes the store.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from psse_env.rl.advantages import ADVANTAGE_METHODS, SCALE_MODES, group_advantages
from psse_env.rl.rollout import Trajectory, TrajectoryStore, reward_spread

DIAGNOSE_CONTRACT = "rl_rollout_diagnosis_v1"
#: Why a group's rewards differ, in the order the reasons are assigned: a
#: group with a success flip is counted as such even when its call counts
#: differ too; ``cost_only`` groups differ in calls, regret or acquisitions
#: alone.
SPREAD_REASONS = ("none", "success_flip", "penalty", "cost_only")
PENALTY_KEYS = ("false_commits", "false_finalizations", "healthy_component_corrupted", "loop_detected", "invalid_actions")
PEAKED_PROBABILITY = 0.99
Groups = Sequence[tuple[str, Sequence[Trajectory]]]


def spread_reason(group: Sequence[Trajectory], *, epsilon: float = 1e-9) -> str:
    """Why the group's rewards differ: a success flip first, then a penalty, then cost alone."""

    if reward_spread(group) <= epsilon:
        return "none"
    credits = {round(float(trajectory.reward.get("credit") or 0.0), 9) for trajectory in group}
    successes = {bool(trajectory.evaluation.get("truth_audited_task_success")) for trajectory in group}
    if len(credits) > 1 or len(successes) > 1:
        return "success_flip"
    for key in PENALTY_KEYS:
        if len({json.dumps(trajectory.reward.get(key), sort_keys=True) for trajectory in group}) > 1:
            return "penalty"
    return "cost_only"


def _token_sequences(trajectory: Trajectory) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(step.completion_token_ids) for step in trajectory.steps)


def divergence(group: Sequence[Trajectory]) -> dict[str, Any]:
    """Where the samples part: the first step whose tokens differ from the group's most common trajectory."""

    sequences = [_token_sequences(trajectory) for trajectory in group]
    counts = Counter(sequences)
    modal, modal_count = counts.most_common(1)[0]
    steps: list[int] = []
    for sequence in sequences:
        if sequence == modal:
            continue
        index = next((i for i, (a, b) in enumerate(zip(sequence, modal)) if a != b), min(len(sequence), len(modal)))
        steps.append(int(index))
    shared = [step / len(modal) for step in steps] if modal else []
    return {
        "distinct_trajectories": len(counts),
        "modal_count": int(modal_count),
        "modal_steps": len(modal),
        "first_divergence_step": min(steps) if steps else None,
        "divergence_steps": steps,
        "shared_prefix_fraction": (sum(shared) / len(shared)) if shared else None,
    }


def _min_probabilities(trajectory: Trajectory) -> list[float]:
    return [min(step.token_probabilities) for step in trajectory.steps if step.token_probabilities]


def group_report(root: str, group: Sequence[Trajectory], *, method: str = "grpo", scale: str = "group") -> dict[str, Any]:
    rewards = [trajectory.total_reward for trajectory in group]
    scaled = group_advantages(rewards, method=method, scale=scale, root=root)
    unscaled = group_advantages(rewards, method=method, scale="none", root=root)
    minima = [value for trajectory in group for value in _min_probabilities(trajectory)]
    return {
        "root": root,
        "family": group[0].family if group else "unknown",
        "samples": len(group),
        "rewards": rewards,
        "reward_spread": reward_spread(group),
        "reason": spread_reason(group),
        "successes": sum(int(bool(trajectory.evaluation.get("truth_audited_task_success"))) for trajectory in group),
        "calls": [int(trajectory.reward.get("calls") or 0) for trajectory in group],
        "unpaired": sum(int(not trajectory.pairing_ok) for trajectory in group),
        "advantages": list(scaled.advantages),
        "advantages_unscaled": list(unscaled.advantages),
        "sampled_steps": len(minima),
        "steps_all_tokens_peaked": sum(1 for value in minima if value >= PEAKED_PROBABILITY),
        **divergence(group),
    }


def _quantile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]


def diagnose_groups(groups: Groups, *, method: str = "grpo", scale: str = "group") -> dict[str, Any]:
    """The iteration's signal: how many groups carry it, of what kind, and how the advantage scale weighs them."""

    if method not in ADVANTAGE_METHODS or scale not in SCALE_MODES:
        raise ValueError(f"method must be one of {ADVANTAGE_METHODS} and scale one of {SCALE_MODES}")
    reports = [group_report(root, group, method=method, scale=scale) for root, group in groups]
    by_reason = Counter(report["reason"] for report in reports)
    by_family: dict[str, dict[str, int]] = {}
    for report in reports:
        entry = by_family.setdefault(
            str(report["family"]), {"groups": 0, "informative": 0, "success_flip": 0, "episodes": 0, "successes": 0}
        )
        entry["groups"] += 1
        entry["informative"] += int(report["reason"] != "none")
        entry["success_flip"] += int(report["reason"] == "success_flip")
        entry["episodes"] += int(report["samples"])
        entry["successes"] += int(report["successes"])
    comparison: dict[str, Any] = {}
    for reason in SPREAD_REASONS[1:]:
        rows = [report for report in reports if report["reason"] == reason]
        if not rows:
            continue
        tokens = sum(len(report["advantages"]) for report in rows)
        comparison[reason] = {
            "groups": len(rows),
            "mean_reward_spread": sum(report["reward_spread"] for report in rows) / len(rows),
            "mean_abs_advantage_scaled": sum(abs(a) for report in rows for a in report["advantages"]) / tokens,
            "mean_abs_advantage_unscaled": sum(abs(a) for report in rows for a in report["advantages_unscaled"]) / tokens,
        }
    minima = [value for _root, group in groups for trajectory in group for value in _min_probabilities(trajectory)]
    first_steps = [report["first_divergence_step"] for report in reports if report["first_divergence_step"] is not None]
    shared = [report["shared_prefix_fraction"] for report in reports if report["shared_prefix_fraction"] is not None]
    episodes = sum(report["samples"] for report in reports)
    successes = sum(report["successes"] for report in reports)
    informative = sum(count for reason, count in by_reason.items() if reason != "none")
    return {
        "contract": DIAGNOSE_CONTRACT,
        "method": method,
        "scale": scale,
        "groups": len(reports),
        "episodes": episodes,
        "success_rate": (successes / episodes) if episodes else None,
        "groups_with_distinct_samples": sum(1 for report in reports if report["distinct_trajectories"] > 1),
        "informative_groups": informative,
        "informative_fraction": (informative / len(reports)) if reports else None,
        "groups_by_spread_reason": {reason: int(by_reason.get(reason, 0)) for reason in SPREAD_REASONS},
        "unpaired_trajectories": sum(report["unpaired"] for report in reports),
        "advantage_scale_comparison": comparison,
        "divergence": {
            "groups": len(first_steps),
            "first_step_histogram": {str(step): count for step, count in sorted(Counter(first_steps).items())},
            "mean_shared_prefix_fraction": (sum(shared) / len(shared)) if shared else None,
        },
        "sampling": {
            "sampled_steps": len(minima),
            "fraction_of_steps_with_every_token_above_0_99": (
                sum(1 for value in minima if value >= PEAKED_PROBABILITY) / len(minima)
            ) if minima else None,
            "min_token_probability_quantiles": {
                "p10": _quantile(minima, 0.10),
                "p50": _quantile(minima, 0.50),
                "p90": _quantile(minima, 0.90),
            },
        },
        "by_family": dict(sorted(by_family.items())),
        "groups_detail": reports,
    }


def diagnose_store(store: TrajectoryStore, *, method: str = "grpo", scale: str = "group") -> dict[str, Any]:
    return diagnose_groups(list(store.iter_groups()), method=method, scale=scale)


def format_summary(report: Mapping[str, Any]) -> str:
    reasons = report["groups_by_spread_reason"]
    sampling = report["sampling"]
    lines = [
        f"groups {report['groups']}, episodes {report['episodes']}, success rate "
        f"{report['success_rate'] if report['success_rate'] is None else round(report['success_rate'], 3)}",
        f"groups with distinct samples {report['groups_with_distinct_samples']}, with a reward spread "
        f"{report['informative_groups']} (success flip {reasons['success_flip']}, penalty {reasons['penalty']}, "
        f"cost only {reasons['cost_only']}); unpaired trajectories {report['unpaired_trajectories']}",
    ]
    for reason, entry in report["advantage_scale_comparison"].items():
        lines.append(
            f"  {reason}: {entry['groups']} groups, mean reward spread {entry['mean_reward_spread']:.3f}, "
            f"mean |advantage| scaled {entry['mean_abs_advantage_scaled']:.3f} / unscaled {entry['mean_abs_advantage_unscaled']:.3f}"
        )
    peaked = sampling["fraction_of_steps_with_every_token_above_0_99"]
    lines.append(
        f"sampled steps {sampling['sampled_steps']}, every token above 0.99 in "
        f"{'n/a' if peaked is None else f'{100 * peaked:.1f}%'}; first divergence step histogram "
        f"{report['divergence']['first_step_histogram']}"
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--iteration", help="an iteration directory of a run (holds trajectories/)")
    source.add_argument("--store", help="a trajectory store directory (holds groups/)")
    parser.add_argument("--method", choices=ADVANTAGE_METHODS, default="grpo")
    parser.add_argument("--scale", choices=SCALE_MODES, default="group")
    parser.add_argument("--output", default=None, help="write the full report (with every group) as JSON here")
    args = parser.parse_args(argv)
    directory = Path(args.store) if args.store else Path(args.iteration) / "trajectories"
    if not (directory / "groups").is_dir():
        raise SystemExit(f"no trajectory store under {directory}")
    report = diagnose_store(TrajectoryStore(directory), method=args.method, scale=args.scale)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(format_summary(report))
    return 0


__all__ = [
    "DIAGNOSE_CONTRACT",
    "PEAKED_PROBABILITY",
    "PENALTY_KEYS",
    "SPREAD_REASONS",
    "diagnose_groups",
    "diagnose_store",
    "divergence",
    "format_summary",
    "group_report",
    "spread_reason",
]


if __name__ == "__main__":
    raise SystemExit(main())
