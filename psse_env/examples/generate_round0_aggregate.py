"""Generate the recovery-balanced round-0 DAgger aggregate on real physics.

Round 0 is the offline pass of the DAgger loop: every executed action comes
from the expert oracle (beta = 1.0), so no trained policy is required.  The
recovery classes that pure expert rollouts never visit are forced through the
counterfactual generator, which executes plausible learner mistakes in
isolated environment branches and records the expert's recovery at every
reached state.

Each episode additionally passes a truth-side audit: an episode whose
accepted corrections do not line up with the scenario's injected faults
(e.g. a measurement correction that masks a topology error against the stale
model) is quarantined out of the export rather than taught as a clean path.

Usage:
    python -m psse_env.examples.generate_round0_aggregate --output-dir OUT \
        [--scale 2] [--protocol canonical] [--seed 20260719]
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from psse_env.actions import (
    COMMIT_STATE,
    CORRECT_MEASUREMENTS,
    CORRECT_PARAMETERS,
    CORRECT_TOPOLOGY,
    RUN_WLS,
)
from psse_env.dagger.counterfactual_generator import CounterfactualGenerator
from psse_env.dagger.dataset_builder import examples_to_chat_sft, write_jsonl
from psse_env.dagger.error_injectors import plausible_wrong_actions
from psse_env.dagger.rollout_collector import (
    DaggerRolloutCollector,
    audit_target_aware_state_classes,
)
from psse_env.dagger.sft_audit import audit_chat_sft_rows, audit_teacher_realizability
from psse_env.dagger.splits import grouped_scenario_split
from psse_env.oracle import ExpertPolicyOracle
from psse_env.providers import MatpowerDeploymentProviders
from psse_env.providers.scenario_generator import Round0ScenarioGenerator
from psse_env.transactional_env import TransactionalPSSEEnv

DEFAULT_SEED = 20260719

# Root-scenario counts per family at --scale 1.
DEFAULT_PLAN: dict[str, int] = {
    "no_error": 4,
    "measurement": 6,
    "multi_measurement": 4,
    "parameter": 6,
    "topology": 6,
    "harmonic": 4,
    "hif": 3,
    "measurement+parameter": 3,
    "measurement+topology": 3,
    "measurement+hif": 2,
}


class ObservableBaselinePolicy:
    """Scripted round-0 stand-in; its proposals are logged, never trained on."""

    def act(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "tool": RUN_WLS,
            "arguments": {"state_id": observation["active_state_id"]},
        }


class CommitDisciplinedExpertOracle:
    """Keep uniform state diversification away from belief-corrupting actions.

    The collector samples uniformly over every expert proposal at beta=1.0.
    Non-preferred corrections and commits can pass verification against a
    wrong model (masking) and corrupt the rest of the episode; deliberate
    wrong actions belong to the bounded counterfactual injector instead.  The
    rank-1 proposal (the supervision label) is always kept; lower-ranked
    corrections/commits are dropped from the sampling pool.
    """

    _MUTATING_TOOLS = {
        CORRECT_MEASUREMENTS,
        CORRECT_PARAMETERS,
        CORRECT_TOPOLOGY,
        COMMIT_STATE,
    }

    def __init__(self, inner: ExpertPolicyOracle) -> None:
        self._inner = inner

    def next_actions(self, state: Any, history: Any) -> list[dict[str, Any]]:
        actions = list(self._inner.next_actions(state, history))
        if len(actions) <= 1:
            return actions
        return [actions[0]] + [
            action
            for action in actions[1:]
            if str(action.get("tool")) not in self._MUTATING_TOOLS
        ]

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def build_environment(args: argparse.Namespace) -> tuple[TransactionalPSSEEnv, ExpertPolicyOracle]:
    providers = MatpowerDeploymentProviders(
        chi2_alpha=args.chi2_alpha,
        hif_alpha_grid_size=args.hif_alpha_grid,
        hif_r_grid_size=args.hif_r_grid,
        hif_max_scans=args.hif_max_scans,
    )
    env = TransactionalPSSEEnv(
        **providers.env_kwargs(),
        production_dataset_mode=True,
        max_steps=args.max_steps,
        history_window=4,
    )
    oracle = ExpertPolicyOracle(process_oracle=env.process_oracle)
    return env, oracle


# ------------------------------------------------------------- episode audit


def _accepted_corrections(final_state: Mapping[str, Any]) -> list[dict[str, Any]]:
    corrections = []
    for item in final_state.get("accepted_corrections") or []:
        if isinstance(item, Mapping):
            corrections.append(dict(item))
    return corrections


def _correction_action(item: Mapping[str, Any]) -> Mapping[str, Any]:
    action = item.get("source_action") or item.get("action") or item
    return action if isinstance(action, Mapping) else {}


def _target_line(arguments: Mapping[str, Any]) -> int | None:
    for key, offset in (("branch_row0", 1), ("line_index1", 0), ("line_index", 0)):
        if arguments.get(key) is not None:
            return int(arguments[key]) + offset
    return None


def audit_episode_against_truth(
    scenario: Mapping[str, Any],
    final_state: Mapping[str, Any],
    *,
    terminal: bool,
) -> dict[str, Any]:
    """Truth-side episode audit for round-0 quarantining.

    Never used as model supervision: it gates which episodes are exported,
    exactly like the existing target-aware collector audits.
    """
    truth_measurement = {
        int(fault["index"])
        for fault in scenario.get("true_measurement_errors") or []
        if fault.get("index") is not None
    }
    truth_lines = {
        int(fault["line_index1"])
        for key in ("true_parameter_errors", "true_topology_errors")
        for fault in scenario.get(key) or []
        if fault.get("line_index1") is not None
    }
    problems: list[str] = []
    for item in _accepted_corrections(final_state):
        action = _correction_action(item)
        tool = str(action.get("tool") or "")
        arguments = action.get("arguments")
        arguments = arguments if isinstance(arguments, Mapping) else {}
        if tool == CORRECT_MEASUREMENTS:
            group = arguments.get("suspect_group") or list(
                (arguments.get("measurement_updates") or {}).keys()
            )
            indices = {int(index) for index in group} if group else set()
            if not truth_measurement:
                problems.append(f"accepted_measurement_correction_without_truth:{sorted(indices)}")
            elif indices and not indices & truth_measurement:
                problems.append(
                    f"measurement_correction_misses_truth:{sorted(indices)}"
                )
        elif tool in {CORRECT_PARAMETERS, CORRECT_TOPOLOGY}:
            line = _target_line(arguments)
            if line is not None and line not in truth_lines:
                problems.append(f"{tool}_on_healthy_line:{line}")
    return {
        "scenario_id": str(scenario.get("scenario_id")),
        "scenario_family": str(scenario.get("scenario_family", "unknown")),
        "terminal": bool(terminal),
        "problems": problems,
        "quarantined": bool(problems),
    }


# ---------------------------------------------------------------- collection


def _bounded_injected_actions(
    env: TransactionalPSSEEnv,
    oracle: ExpertPolicyOracle,
    scenario: Mapping[str, Any],
    *,
    limit: int,
    rng: random.Random,
) -> list[Any]:
    """One injected mistake per family, capped at ``limit``, deterministic."""
    env.reset(scenario)
    root_observation = env.get_policy_observation([])
    root_oracle = env.get_oracle_state([])
    expert_actions = list(oracle.next_actions(root_oracle, []))
    expert_actions.extend(CounterfactualGenerator._truth_correction_actions(root_oracle))
    physical = env.store.get_state(root_observation.active_state_id)
    injected = plausible_wrong_actions(
        root_observation.as_dict(), expert_actions, physical_state=physical
    )
    by_family: dict[str, list[Any]] = {}
    for item in injected:
        by_family.setdefault(item.family, []).append(item)
    picked = [rng.choice(items) for _, items in sorted(by_family.items())]
    rng.shuffle(picked)
    return picked[: max(int(limit), 0)]


def collect_round0(
    args: argparse.Namespace,
    scenarios: list[dict[str, Any]],
) -> dict[str, Any]:
    env, oracle = build_environment(args)
    collector_rng = random.Random(args.seed)
    counterfactual_rng = random.Random(args.seed + 1)
    episode_rows: list[dict[str, Any]] = []
    recovery_rows: list[dict[str, Any]] = []
    episode_audits: list[dict[str, Any]] = []
    quarantined_rows = 0

    for scenario in scenarios:
        collector = DaggerRolloutCollector(
            env=env,
            policy=ObservableBaselinePolicy(),
            expert_oracle=CommitDisciplinedExpertOracle(oracle),
            rng=collector_rng,
        )
        rows = collector.collect_iteration(
            scenarios=[scenario],
            iteration=0,
            beta=1.0,
            max_steps=args.max_steps,
        )
        audit = audit_episode_against_truth(
            scenario, env.current_state(), terminal=env.is_terminal()
        )
        episode_audits.append(audit)
        if audit["quarantined"]:
            quarantined_rows += len(rows)
        else:
            episode_rows.extend(rows)

        if args.counterfactuals_per_scenario > 0:
            generator = CounterfactualGenerator(env=env, expert_oracle=oracle)
            injected = _bounded_injected_actions(
                env,
                oracle,
                scenario,
                limit=args.counterfactuals_per_scenario,
                rng=counterfactual_rng,
            )
            recovery_rows.extend(
                generator.generate_from_current(
                    injected,
                    root_scenario_id=str(scenario["scenario_id"]),
                )
            )

    return {
        "episode_rows": episode_rows,
        "recovery_rows": recovery_rows,
        "episode_audits": episode_audits,
        "quarantined_rows": quarantined_rows,
    }


# ------------------------------------------------------------------- reports


def _distribution(rows: Iterable[Mapping[str, Any]], key: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        value = row.get(key)
        if value is None and isinstance(row.get("labels"), Mapping):
            value = row["labels"].get(key)
        counts[str(value)] += 1
    return dict(sorted(counts.items()))


def _family_by_root(scenarios: Iterable[Mapping[str, Any]]) -> dict[str, str]:
    return {
        str(scenario["scenario_id"]): str(scenario.get("scenario_family", "unknown"))
        for scenario in scenarios
    }


def _family_class_matrix(
    rows: Iterable[Mapping[str, Any]], family_by_root: Mapping[str, str]
) -> dict[str, dict[str, int]]:
    matrix: dict[str, Counter[str]] = {}
    for row in rows:
        family = family_by_root.get(str(row.get("root_scenario_id")), "unknown")
        state_class = row.get("state_class")
        if state_class is None and isinstance(row.get("labels"), Mapping):
            state_class = row["labels"].get("state_class")
        matrix.setdefault(family, Counter())[str(state_class)] += 1
    return {family: dict(sorted(counts.items())) for family, counts in sorted(matrix.items())}


def generate(args: argparse.Namespace) -> dict[str, Any]:
    plan = {family: count * args.scale for family, count in DEFAULT_PLAN.items()}
    if args.plan:
        plan = json.loads(Path(args.plan).read_text()) if Path(args.plan).is_file() else json.loads(args.plan)
    generator = Round0ScenarioGenerator(seed=args.seed, hif_max_scans=args.hif_max_scans)
    scenarios = generator.build(plan)
    if not scenarios:
        raise RuntimeError("Scenario generator produced no scenarios for the plan.")

    collected = collect_round0(args, scenarios)
    all_rows = collected["episode_rows"] + collected["recovery_rows"]
    if not all_rows:
        raise RuntimeError("Round-0 collection produced no rows.")

    splits = grouped_scenario_split(
        all_rows,
        train_fraction=0.75,
        validation_fraction=0.15,
        seed=args.seed,
    )
    exported = {
        name: examples_to_chat_sft(rows, protocol=args.protocol)
        for name, rows in splits.items()
        if rows
    }
    all_exported = [row for rows in exported.values() for row in rows]
    chat_audit = audit_chat_sft_rows(all_exported)
    realizability = audit_teacher_realizability(all_exported, conflict_tolerance=0.0)
    class_audit = audit_target_aware_state_classes(collected["episode_rows"])
    if not chat_audit["passed"]:
        raise RuntimeError(f"Chat-row audit failed: {chat_audit['errors'][:3]}")
    if not realizability["passed"]:
        raise RuntimeError(
            f"Teacher realizability audit failed: conflict_rate={realizability['conflict_rate']}"
        )
    if not class_audit["passed"]:
        raise RuntimeError(f"Target-aware class audit failed: {class_audit['mismatches'][:3]}")

    family_by_root = _family_by_root(scenarios)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "aggregate.raw.jsonl", all_rows)
    for name, rows in exported.items():
        write_jsonl(output_dir / f"aggregate.{name}.jsonl", rows)
    manifest = {
        "scenario_manifest": generator.manifest,
        "family_by_root": family_by_root,
        "episode_audits": collected["episode_audits"],
    }
    (output_dir / "aggregate.manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    report = {
        "seed": args.seed,
        "protocol": args.protocol,
        "plan": plan,
        "scenario_report": generator.report(),
        "episode_rows": len(collected["episode_rows"]),
        "recovery_rows": len(collected["recovery_rows"]),
        "quarantined_rows": collected["quarantined_rows"],
        "quarantined_episodes": [
            audit for audit in collected["episode_audits"] if audit["quarantined"]
        ],
        "nonterminal_episodes": [
            audit["scenario_id"]
            for audit in collected["episode_audits"]
            if not audit["terminal"]
        ],
        "split_rows": {name: len(rows) for name, rows in exported.items()},
        "state_class_distribution": _distribution(all_rows, "state_class"),
        "family_state_class_matrix": _family_class_matrix(all_rows, family_by_root),
        "native_chat_audit": chat_audit,
        "teacher_realizability": realizability,
        "target_aware_class_audit": class_audit,
    }
    preferred: Counter[str] = Counter()
    for row in collected["episode_rows"]:
        action = row.get("preferred_action")
        preferred[str(action.get("tool") if isinstance(action, Mapping) else None)] += 1
    report["preferred_action_distribution"] = dict(sorted(preferred.items()))
    (output_dir / "aggregate.preflight.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the recovery-balanced round-0 DAgger aggregate."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "round0_aggregate",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--scale", type=int, default=1, help="Multiply the default plan.")
    parser.add_argument("--plan", type=str, default=None, help="JSON plan or path to one.")
    parser.add_argument("--protocol", choices=("controller", "canonical"), default="canonical")
    parser.add_argument("--max-steps", type=int, default=18)
    parser.add_argument("--counterfactuals-per-scenario", type=int, default=3)
    parser.add_argument("--chi2-alpha", type=float, default=0.01)
    parser.add_argument("--hif-alpha-grid", type=int, default=5)
    parser.add_argument("--hif-r-grid", type=int, default=7)
    parser.add_argument("--hif-max-scans", type=int, default=3)
    args = parser.parse_args()
    report = generate(args)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "episode_rows": report["episode_rows"],
                "recovery_rows": report["recovery_rows"],
                "quarantined_rows": report["quarantined_rows"],
                "split_rows": report["split_rows"],
                "state_classes": report["state_class_distribution"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
