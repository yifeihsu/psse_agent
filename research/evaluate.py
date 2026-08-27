"""Roll a policy through held-out scenarios and report behaviour.

This is the measurement that makes a DAgger iteration meaningful: run the same
scenarios under the round-0 policy and the round-1 policy and compare.  It
reports what happened -- how many episodes reached a terminal state, how often
the policy emitted an unusable action -- and makes no release claim.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from psse_env.dagger.protocol_bridge import canonical_to_internal_action
from psse_env.dagger.release_factories import production_environment_factory

from .collect import load_scenarios
from .model import load_policy
from .train import file_sha256

DEFAULT_MAX_STEPS = 24
EVALUATION_SCHEMA_VERSION = 2


def _scenario_execution(scenario: dict[str, Any]) -> dict[str, Any]:
    """Return the runtime half of a composed scenario envelope."""

    execution = scenario.get("execution")
    if isinstance(execution, dict):
        runtime = copy.deepcopy(dict(execution))
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
                        runtime.setdefault(flat, copy.deepcopy(clean_state[nested]))
            for key, value in truth.items():
                if key in {"truth_complete", "clean_state"}:
                    continue
                runtime.setdefault(str(key), copy.deepcopy(value))
        return runtime
    return copy.deepcopy(dict(scenario))


def _scenario_identity(scenario: dict[str, Any]) -> dict[str, str | None]:
    grouping = scenario.get("grouping")
    grouping = grouping if isinstance(grouping, dict) else {}
    root = scenario.get("root_scenario_id") or grouping.get("root_scenario_id")
    scenario_id = scenario.get("scenario_id") or grouping.get("scenario_id") or root
    fingerprint = scenario.get("physical_root_fingerprint") or grouping.get(
        "physical_root_fingerprint"
    )
    return {
        "scenario_id": str(scenario_id) if scenario_id is not None else None,
        "root_scenario_id": str(root) if root is not None else None,
        "physical_root_fingerprint": (
            str(fingerprint) if fingerprint is not None else None
        ),
    }


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
    termination_reason: str | None = None

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
            termination_reason = "policy_generation_abort"
            break
        steps += 1
        try:
            executed = canonical_to_internal_action(action)
        except Exception:  # noqa: BLE001
            executed = action
        try:
            next_state, tool_output = env.step(executed)
        except Exception as exc:  # noqa: BLE001
            invalid_actions += 1
            if first_error is None:
                first_error = f"env.step {type(exc).__name__}: {exc}"
            termination_reason = "environment_step_exception"
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
                termination_reason = "terminal_outcome"
        arguments = executed.get("arguments") if isinstance(executed, dict) else None
        trace.append(
            {
                "tool": executed.get("tool") if isinstance(executed, dict) else None,
                # Full structured arguments are required for deterministic
                # physical replay.  Truncation makes an evaluation impossible
                # to audit and is therefore never allowed in schema v2.
                "arguments": copy.deepcopy(arguments),
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
            termination_reason = "environment_terminal"
            break
        if isinstance(next_state, dict) and next_state.get("done"):
            terminal_outcome = terminal_outcome or "environment_terminal"
            termination_reason = "state_done"
            break

    if termination_reason is None:
        termination_reason = (
            "terminal_outcome" if terminal_outcome is not None else "step_horizon"
        )
    identity = _scenario_identity(scenario)
    return {
        **identity,
        "steps": steps,
        "invalid_actions": invalid_actions,
        "first_error": first_error,
        "actions": trace,
        "terminal_outcome": terminal_outcome,
        "termination_reason": termination_reason,
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
    scenarios_sha256 = file_sha256(scenarios_path)
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
    if file_sha256(scenarios_path) != scenarios_sha256:
        raise RuntimeError("evaluation scenarios changed while the run was in progress")
    return {
        "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
        "label": label,
        "adapter": str(adapter_path) if adapter_path else None,
        "episodes": len(episodes),
        "scenarios_path": str(scenarios_path),
        "scenarios_sha256": scenarios_sha256,
        "model_id": model_id,
        "model_revision": revision,
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
