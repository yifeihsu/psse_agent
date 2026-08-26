"""Roll a policy through held-out scenarios and report behaviour.

This is the measurement that makes a DAgger iteration meaningful: run the same
scenarios under the round-0 policy and the round-1 policy and compare.  It
reports what happened -- how many episodes reached a terminal state, how often
the policy emitted an unusable action -- and makes no release claim.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from psse_env.dagger.protocol_bridge import canonical_to_internal_action
from psse_env.dagger.release_factories import production_environment_factory

from .collect import load_scenarios
from .model import load_policy

DEFAULT_MAX_STEPS = 24


def _scenario_execution(scenario: dict[str, Any]) -> dict[str, Any]:
    """Return the runtime half of a composed scenario envelope."""

    execution = scenario.get("execution")
    if isinstance(execution, dict):
        runtime = dict(execution)
        audit = scenario.get("audit")
        truth = audit.get("truth") if isinstance(audit, dict) else None
        if isinstance(truth, dict):
            clean_state = truth.get("clean_state")
            if isinstance(clean_state, dict):
                for nested, flat in (
                    ("case", "clean_case"),
                    ("measurements", "clean_measurements"),
                ):
                    if nested in clean_state:
                        runtime.setdefault(flat, clean_state[nested])
            for key, value in truth.items():
                if key in {"truth_complete", "clean_state"}:
                    continue
                runtime.setdefault(str(key), value)
        return runtime
    return dict(scenario)


def run_episode(
    env: Any, policy: Any, scenario: dict[str, Any], *, max_steps: int
) -> dict[str, Any]:
    """Run one episode and record how it ended."""

    env.reset(_scenario_execution(scenario))
    history: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    invalid_actions = 0
    steps = 0
    terminal_outcome: str | None = None

    first_error: str | None = None
    for _ in range(max_steps):
        observation = env.get_policy_observation(history)
        # The environment returns a PolicyObservation object; the policy
        # adapter requires a plain mapping.  Passing the object through made
        # every episode fail at step 0 and score as 65 invalid actions.
        if hasattr(observation, "as_dict"):
            observation = observation.as_dict()
        try:
            action = policy.act(observation)
        except Exception as exc:  # noqa: BLE001 - an unusable generation is a result
            invalid_actions += 1
            if first_error is None:
                first_error = f"{type(exc).__name__}: {exc}"
            break
        steps += 1
        try:
            executed = canonical_to_internal_action(action)
        except Exception:  # noqa: BLE001
            executed = action
        try:
            next_state, tool_output = env.step(executed)
        except Exception:  # noqa: BLE001
            invalid_actions += 1
            break
        if isinstance(tool_output, dict):
            if tool_output.get("execution_status") == "failure":
                invalid_actions += 1
            # The environment reports termination inside tool_metrics; the
            # top-level key is kept as a fallback for older outputs.  Missing
            # this ran every episode to the step horizon and counted each
            # post-terminal action as a failure.
            metrics = tool_output.get("tool_metrics")
            outcome = (
                metrics.get("terminal_outcome")
                if isinstance(metrics, dict)
                else None
            ) or tool_output.get("terminal_outcome")
            if outcome:
                terminal_outcome = str(outcome)
        arguments = executed.get("arguments") if isinstance(executed, dict) else None
        rendered_arguments = json.dumps(arguments, sort_keys=True, default=str)
        trace.append(
            {
                "tool": executed.get("tool") if isinstance(executed, dict) else None,
                "arguments": (
                    rendered_arguments
                    if len(rendered_arguments) <= 300
                    else rendered_arguments[:300] + "..."
                ),
                "status": (
                    tool_output.get("execution_status")
                    if isinstance(tool_output, dict)
                    else None
                ),
                "error_code": (
                    tool_output.get("error_code")
                    if isinstance(tool_output, dict)
                    else None
                ),
            }
        )
        history.append({"action": executed, "tool_output": tool_output})
        if terminal_outcome is not None:
            break
        if getattr(env, "terminal", False):
            terminal_outcome = str(
                getattr(env, "terminal_outcome", None) or "environment_terminal"
            )
            break
        if isinstance(next_state, dict) and next_state.get("done"):
            terminal_outcome = terminal_outcome or "environment_terminal"
            break

    return {
        "scenario_id": scenario.get("scenario_id")
        or (scenario.get("grouping") or {}).get("scenario_id"),
        "steps": steps,
        "invalid_actions": invalid_actions,
        "first_error": first_error,
        "actions": trace,
        "terminal_outcome": terminal_outcome,
        "horizon_truncated": terminal_outcome is None,
    }


def evaluate(
    *,
    scenarios_path: str | Path,
    adapter_path: str | Path | None,
    label: str,
    max_steps: int = DEFAULT_MAX_STEPS,
    model_id: str | None = None,
    revision: str | None = None,
    guards: bool = False,
) -> dict[str, Any]:
    scenarios = load_scenarios(scenarios_path)
    policy = load_policy(
        model_id=model_id,
        revision=revision,
        adapter_path=adapter_path,
        guarded=guards,
    )
    env = production_environment_factory()

    episodes = []
    for scenario in scenarios:
        episode = run_episode(env, policy, scenario, max_steps=max_steps)
        if hasattr(policy, "pop_events"):
            episode["guard_events"] = policy.pop_events()
        episodes.append(episode)
    outcomes: Counter[str] = Counter(
        str(episode["terminal_outcome"] or "horizon_truncated")
        for episode in episodes
    )
    total_steps = sum(int(episode["steps"]) for episode in episodes)
    total_invalid = sum(int(episode["invalid_actions"]) for episode in episodes)
    terminated = sum(
        1 for episode in episodes if not episode["horizon_truncated"]
    )
    return {
        "label": label,
        "adapter": str(adapter_path) if adapter_path else None,
        "episodes": len(episodes),
        "episodes_terminated": terminated,
        "episodes_horizon_truncated": len(episodes) - terminated,
        "total_steps": total_steps,
        "invalid_actions": total_invalid,
        "invalid_action_rate": (
            total_invalid / total_steps if total_steps else None
        ),
        "terminal_outcomes": dict(outcomes.most_common()),
        "guards_enabled": guards,
        "guard_interventions": sum(
            len(episode.get("guard_events") or []) for episode in episodes
        ),
        "per_episode": episodes,
        "release_evidence": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--label", default="policy")
    parser.add_argument("--output")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--model-id")
    parser.add_argument("--revision")
    parser.add_argument(
        "--guards",
        action="store_true",
        help="Enable silence-retry and stale-reference inference guards.",
    )
    args = parser.parse_args(argv)

    report = evaluate(
        scenarios_path=args.scenarios,
        adapter_path=args.adapter,
        label=args.label,
        max_steps=args.max_steps,
        model_id=args.model_id,
        revision=args.revision,
        guards=args.guards,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
