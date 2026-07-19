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
from typing import Any, Iterable, Mapping, Sequence

import psse_env.dagger.counterfactual_generator as counterfactual_generator_module
import psse_env.dagger.dataset_builder as dataset_builder_module
import psse_env.dagger.protocol_bridge as protocol_bridge_module
import psse_env.dagger.rollout_collector as rollout_collector_module
import psse_env.dagger.sft_audit as sft_audit_module
import psse_env.dagger.splits as splits_module
import psse_env.oracle as oracle_module
import psse_env.providers.matpower as matpower_provider_module
import psse_env.providers.scenario_generator as scenario_generator_module
import psse_env.transactional_env as transactional_env_module
from psse_env.actions import (
    CORRECT_MEASUREMENTS,
    CORRECT_PARAMETERS,
    CORRECT_TOPOLOGY,
    RUN_WLS,
)
from psse_env.dagger.counterfactual_generator import CounterfactualGenerator
from psse_env.dagger.dataset_builder import (
    TOOL_JSON_SCHEMAS,
    examples_to_chat_sft,
    write_jsonl,
)
from psse_env.dagger.error_injectors import plausible_wrong_actions
from psse_env.dagger.rollout_collector import (
    DaggerRolloutCollector,
    audit_target_aware_state_classes,
)
from psse_env.dagger.sft_audit import (
    audit_approximate_teacher_realizability,
    audit_chat_sft_rows,
    audit_teacher_realizability,
)
from psse_env.dagger.splits import (
    audit_physical_split_disjointness,
    grouped_scenario_split,
    physical_root_fingerprint,
)
from psse_env.dagger.protocol_bridge import unified_tool_schemas
from psse_env.oracle import ExpertPolicyOracle
from psse_env.providers import MatpowerDeploymentProviders
from psse_env.providers.scenario_generator import Round0ScenarioGenerator
from psse_env.sft.provenance import file_sha256, git_source_state, stable_json_sha256
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
    terminal_outcome: str | None = None,
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
        "terminal_outcome": str(terminal_outcome) if terminal_outcome else None,
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
            expert_oracle=oracle,
            rng=collector_rng,
        )
        rows = collector.collect_iteration(
            scenarios=[scenario],
            iteration=0,
            beta=1.0,
            max_steps=args.max_steps,
        )
        audit = audit_episode_against_truth(
            scenario,
            env.current_state(),
            terminal=env.is_terminal(),
            terminal_outcome=env.terminal_outcome,
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
                    physical_root_fingerprint=scenario.get(
                        "physical_root_fingerprint"
                    ),
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


def _collection_release_failures(
    *,
    plan: Mapping[str, int],
    scenarios: Iterable[Mapping[str, Any]],
    nonterminal_episodes: Sequence[str],
    quarantined_episodes: Sequence[Mapping[str, Any]],
    unknown_terminal_outcome_episodes: Sequence[str] = (),
) -> list[str]:
    """Fail release provenance when the requested expert suite is incomplete."""
    failures: list[str] = []
    if nonterminal_episodes:
        failures.append(
            f"{len(nonterminal_episodes)} expert episode(s) did not reach a terminal decision"
        )
    if quarantined_episodes:
        failures.append(
            f"{len(quarantined_episodes)} episode(s) failed the truth-side correction audit"
        )
    if unknown_terminal_outcome_episodes:
        failures.append(
            f"{len(unknown_terminal_outcome_episodes)} terminal episode(s) lacked "
            "an explicit resolved/operator_escalation outcome"
        )
    built_by_family = Counter(
        str(scenario.get("scenario_family") or "unknown") for scenario in scenarios
    )
    plan_shortfalls = {
        family: int(required) - int(built_by_family.get(family, 0))
        for family, required in plan.items()
        if int(built_by_family.get(family, 0)) < int(required)
    }
    if plan_shortfalls:
        failures.append(
            "scenario generator did not fulfill the requested plan: "
            + json.dumps(plan_shortfalls, sort_keys=True)
        )
    return failures


def _terminal_scenario_matrix(
    episode_audits: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Summarize whether every collected expert root terminates, by family."""
    grouped: dict[str, dict[str, Any]] = {}
    for audit in episode_audits:
        family = str(audit.get("scenario_family") or "unknown")
        entry = grouped.setdefault(
            family,
            {
                "episodes": 0,
                "terminal_episodes": 0,
                "nonterminal_episode_ids": [],
                "quarantined_episode_ids": [],
                "resolved_episode_ids": [],
                "operator_escalation_episode_ids": [],
                "unknown_terminal_outcome_episode_ids": [],
                "terminal_outcome_counts": Counter(),
            },
        )
        entry["episodes"] += 1
        scenario_id = str(audit.get("scenario_id") or "unknown")
        if audit.get("terminal") is True:
            entry["terminal_episodes"] += 1
            outcome = str(audit.get("terminal_outcome") or "unknown")
            entry["terminal_outcome_counts"][outcome] += 1
            if outcome == "resolved":
                entry["resolved_episode_ids"].append(scenario_id)
            elif outcome == "operator_escalation":
                entry["operator_escalation_episode_ids"].append(scenario_id)
            else:
                entry["unknown_terminal_outcome_episode_ids"].append(scenario_id)
        else:
            entry["nonterminal_episode_ids"].append(scenario_id)
        if audit.get("quarantined") is True:
            entry["quarantined_episode_ids"].append(scenario_id)
    for entry in grouped.values():
        entry["terminal_rate"] = (
            entry["terminal_episodes"] / entry["episodes"]
            if entry["episodes"]
            else 0.0
        )
        entry["resolution_rate"] = (
            len(entry["resolved_episode_ids"]) / entry["episodes"]
            if entry["episodes"]
            else 0.0
        )
        entry["terminal_outcome_counts"] = dict(
            sorted(entry["terminal_outcome_counts"].items())
        )
        entry["release_terminal_coverage"] = bool(
            entry["episodes"]
            and entry["terminal_episodes"] == entry["episodes"]
            and not entry["quarantined_episode_ids"]
            and not entry["unknown_terminal_outcome_episode_ids"]
        )
        entry["release_resolution_coverage"] = bool(
            entry["episodes"]
            and len(entry["resolved_episode_ids"]) == entry["episodes"]
            and not entry["quarantined_episode_ids"]
        )
    return {family: grouped[family] for family in sorted(grouped)}


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


def _generation_descriptor(
    args: argparse.Namespace,
    plan: Mapping[str, int],
) -> dict[str, Any]:
    """Bind a generated aggregate to its source, schemas, and collection plan."""
    repo_root = Path(__file__).resolve().parents[2]
    source_files = (
        Path(__file__),
        Path(dataset_builder_module.__file__),
        Path(protocol_bridge_module.__file__),
        Path(rollout_collector_module.__file__),
        Path(counterfactual_generator_module.__file__),
        Path(sft_audit_module.__file__),
        Path(splits_module.__file__),
        Path(scenario_generator_module.__file__),
        Path(matpower_provider_module.__file__),
        Path(transactional_env_module.__file__),
        Path(oracle_module.__file__),
    )
    schemas = (
        unified_tool_schemas()
        if args.protocol == "canonical"
        else TOOL_JSON_SCHEMAS
    )
    return {
        "generation_provenance_version": 1,
        "source_state": git_source_state(repo_root),
        "protocol": args.protocol,
        "schema_registry_hash": stable_json_sha256(schemas),
        "generator_hashes": {
            str(path.resolve().relative_to(repo_root)): file_sha256(path)
            for path in source_files
        },
        "generation_config": {
            "seed": args.seed,
            "plan": dict(sorted(plan.items())),
            "max_steps": args.max_steps,
            "counterfactuals_per_scenario": args.counterfactuals_per_scenario,
            "chi2_alpha": args.chi2_alpha,
            "hif_alpha_grid": args.hif_alpha_grid,
            "hif_r_grid": args.hif_r_grid,
            "hif_max_scans": args.hif_max_scans,
        },
    }


def generate(args: argparse.Namespace) -> dict[str, Any]:
    plan = {family: count * args.scale for family, count in DEFAULT_PLAN.items()}
    if args.plan:
        plan = json.loads(Path(args.plan).read_text()) if Path(args.plan).is_file() else json.loads(args.plan)
    generation_descriptor = _generation_descriptor(args, plan)
    generation_provenance_id = stable_json_sha256(generation_descriptor)
    generator = Round0ScenarioGenerator(seed=args.seed, hif_max_scans=args.hif_max_scans)
    scenarios = generator.build(plan)
    if not scenarios:
        raise RuntimeError("Scenario generator produced no scenarios for the plan.")

    physical_by_root = {}
    for scenario in scenarios:
        fingerprint = physical_root_fingerprint(scenario)
        scenario["physical_root_fingerprint"] = fingerprint
        physical_by_root[str(scenario["root_scenario_id"])] = fingerprint

    # Assign the physical-root split before any DAgger or counterfactual
    # descendants are generated. Every later row inherits this immutable
    # assignment; it is never independently hashed by scenario/branch ID.
    root_split_rows = [
        {
            "root_scenario_id": str(scenario["root_scenario_id"]),
            "physical_root_fingerprint": scenario["physical_root_fingerprint"],
        }
        for scenario in scenarios
    ]
    root_splits = grouped_scenario_split(
        root_split_rows,
        train_fraction=0.75,
        validation_fraction=0.15,
        seed=args.seed,
    )
    split_by_root = {
        str(row["root_scenario_id"]): split_name
        for split_name, rows in root_splits.items()
        for row in rows
    }
    for scenario in scenarios:
        scenario["dataset_split"] = split_by_root[str(scenario["root_scenario_id"])]

    collected = collect_round0(args, scenarios)
    all_rows = collected["episode_rows"] + collected["recovery_rows"]
    if not all_rows:
        raise RuntimeError("Round-0 collection produced no rows.")

    for row in all_rows:
        root_id = str(row.get("root_scenario_id", row.get("scenario_id")))
        try:
            expected_fingerprint = physical_by_root[root_id]
        except KeyError as exc:
            raise RuntimeError(
                f"Collected row references unknown physical root {root_id!r}."
            ) from exc
        if row.get("physical_root_fingerprint") != expected_fingerprint:
            raise RuntimeError(
                f"Collected row {row.get('example_id')!r} did not inherit its "
                "physical-root fingerprint."
            )
        row["dataset_split"] = split_by_root[root_id]
        row["generation_provenance_id"] = generation_provenance_id
        if row.get("production_label_eligible") is not False:
            row["production_label_eligible"] = True
            row.setdefault("dataset_source", "dagger_rollout")

    eligible_rows = [
        row for row in all_rows if row.get("production_label_eligible") is True
    ]
    auxiliary_rows = [
        row for row in all_rows if row.get("production_label_eligible") is False
    ]
    if not eligible_rows:
        raise RuntimeError("Round-0 collection produced no production-label-eligible rows.")
    splits = {
        split_name: [
            row for row in eligible_rows if row.get("dataset_split") == split_name
        ]
        for split_name in ("train", "validation", "test")
    }
    physical_split_audit = audit_physical_split_disjointness(splits)
    if not physical_split_audit["passed"]:
        raise RuntimeError(
            f"Physical-root split audit failed: {physical_split_audit}"
        )
    exported = {
        name: examples_to_chat_sft(rows, protocol=args.protocol)
        for name, rows in splits.items()
        if rows
    }
    all_exported = [row for rows in exported.values() for row in rows]
    chat_audit = audit_chat_sft_rows(all_exported)
    realizability = audit_teacher_realizability(all_exported, conflict_tolerance=0.0)
    approximate_realizability = audit_approximate_teacher_realizability(
        all_exported,
        quantization_bin=0.25,
        conflict_tolerance=0.05,
        nearest_neighbor_tolerance=0.10,
        perturbation_radius=0.25,
        require_cost_margins=False,
    )
    class_audit = audit_target_aware_state_classes(collected["episode_rows"])
    if not chat_audit["passed"]:
        raise RuntimeError(f"Chat-row audit failed: {chat_audit['errors'][:3]}")
    if not realizability["passed"]:
        raise RuntimeError(
            f"Teacher realizability audit failed: conflict_rate={realizability['conflict_rate']}"
        )
    if not approximate_realizability["passed"]:
        raise RuntimeError(
            "Approximate teacher realizability audit failed: "
            f"conflict_rate={approximate_realizability['approximate_conflict_rate']}, "
            "nearest_neighbor_disagreement_rate="
            f"{approximate_realizability['nearest_neighbor_action_disagreement_rate']}"
        )
    if not class_audit["passed"]:
        raise RuntimeError(f"Target-aware class audit failed: {class_audit['mismatches'][:3]}")

    family_by_root = _family_by_root(scenarios)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_paths = [output_dir / "aggregate.raw.jsonl"]
    write_jsonl(dataset_paths[0], eligible_rows)
    if auxiliary_rows:
        auxiliary_path = output_dir / "aggregate.auxiliary_counterfactual.raw.jsonl"
        write_jsonl(auxiliary_path, auxiliary_rows)
        dataset_paths.append(auxiliary_path)
    for name, rows in exported.items():
        split_path = output_dir / f"aggregate.{name}.jsonl"
        write_jsonl(split_path, rows)
        dataset_paths.append(split_path)
    manifest = {
        "scenario_manifest": [
            {
                **entry,
                "physical_root_fingerprint": physical_by_root[str(entry["scenario_id"])],
                "dataset_split": split_by_root[str(entry["scenario_id"])],
            }
            for entry in generator.manifest
        ],
        "family_by_root": family_by_root,
        "episode_audits": collected["episode_audits"],
    }
    (output_dir / "aggregate.manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    nonterminal_episodes = [
        audit["scenario_id"]
        for audit in collected["episode_audits"]
        if not audit["terminal"]
    ]
    quarantined_episodes = [
        audit for audit in collected["episode_audits"] if audit["quarantined"]
    ]
    operator_escalation_episodes = [
        audit["scenario_id"]
        for audit in collected["episode_audits"]
        if audit.get("terminal_outcome") == "operator_escalation"
    ]
    unknown_terminal_outcome_episodes = [
        audit["scenario_id"]
        for audit in collected["episode_audits"]
        if audit.get("terminal") is True
        and audit.get("terminal_outcome") not in {"resolved", "operator_escalation"}
    ]
    terminal_scenario_matrix = _terminal_scenario_matrix(
        collected["episode_audits"]
    )
    report = {
        "seed": args.seed,
        "protocol": args.protocol,
        "plan": plan,
        "scenario_report": generator.report(),
        "episode_rows": len(collected["episode_rows"]),
        "recovery_rows": len(collected["recovery_rows"]),
        "production_label_eligible_rows": len(eligible_rows),
        "auxiliary_ineligible_rows": len(auxiliary_rows),
        "quarantined_rows": collected["quarantined_rows"],
        "quarantined_episodes": quarantined_episodes,
        "nonterminal_episodes": nonterminal_episodes,
        "operator_escalation_episodes": operator_escalation_episodes,
        "unknown_terminal_outcome_episodes": unknown_terminal_outcome_episodes,
        "terminal_scenario_matrix": terminal_scenario_matrix,
        "split_rows": {name: len(rows) for name, rows in exported.items()},
        "state_class_distribution": _distribution(eligible_rows, "state_class"),
        "auxiliary_state_class_distribution": _distribution(
            auxiliary_rows, "state_class"
        ),
        "family_state_class_matrix": _family_class_matrix(
            eligible_rows, family_by_root
        ),
        "native_chat_audit": chat_audit,
        "teacher_realizability": realizability,
        "approximate_teacher_realizability": approximate_realizability,
        "physical_split_audit": physical_split_audit,
        "target_aware_class_audit": class_audit,
        "generation_provenance": {
            "generation_provenance_id": generation_provenance_id,
            "source_commit": generation_descriptor["source_state"].get(
                "source_commit"
            ),
            "source_worktree_dirty": generation_descriptor["source_state"].get(
                "source_worktree_dirty"
            ),
            "schema_registry_hash": generation_descriptor[
                "schema_registry_hash"
            ],
        },
    }
    preferred: Counter[str] = Counter()
    for row in collected["episode_rows"]:
        action = row.get("preferred_action")
        preferred[str(action.get("tool") if isinstance(action, Mapping) else None)] += 1
    report["preferred_action_distribution"] = dict(sorted(preferred.items()))
    release_failures: list[str] = []
    if generation_descriptor["source_state"].get("release_eligible_source") is not True:
        release_failures.append("source worktree was not clean at generation time")
    if args.protocol != "canonical":
        release_failures.append("model-visible protocol was not canonical")
    release_failures.extend(
        _collection_release_failures(
            plan=plan,
            scenarios=scenarios,
            nonterminal_episodes=nonterminal_episodes,
            quarantined_episodes=quarantined_episodes,
            unknown_terminal_outcome_episodes=unknown_terminal_outcome_episodes,
        )
    )
    if set(exported) != {"train", "validation", "test"}:
        release_failures.append("one or more required train/validation/test splits were empty")
    observed_schema_hashes = {
        stable_json_sha256(row.get("tools")) for row in all_exported
    }
    if observed_schema_hashes != {generation_descriptor["schema_registry_hash"]}:
        release_failures.append("exported rows did not match the generated schema registry")
    generation_provenance = {
        **generation_descriptor,
        "generation_descriptor": generation_descriptor,
        "generation_provenance_id": generation_provenance_id,
        "dataset_hashes": {
            path.name: file_sha256(path) for path in sorted(dataset_paths)
        },
        "release_eligible": not release_failures,
        "release_failures": release_failures,
    }
    (output_dir / "aggregate.generation_provenance.json").write_text(
        json.dumps(generation_provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report["generation_provenance"].update(
        {
            "release_eligible": not release_failures,
            "release_failures": release_failures,
            "path": str(output_dir / "aggregate.generation_provenance.json"),
        }
    )
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
    parser.add_argument("--max-steps", type=int, default=24)
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
