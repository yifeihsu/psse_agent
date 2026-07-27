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
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import psse_env.dagger.counterfactual_generator as counterfactual_generator_module
import psse_env.dagger.dataset_builder as dataset_builder_module
import psse_env.dagger.evaluation_gate as evaluation_gate_module
import psse_env.dagger.evaluator as evaluator_module
import psse_env.dagger.protocol_bridge as protocol_bridge_module
import psse_env.dagger.replay_buffer as replay_buffer_module
import psse_env.dagger.release_audit as release_audit_module
import psse_env.dagger.rollout_collector as rollout_collector_module
import psse_env.dagger.sft_audit as sft_audit_module
import psse_env.dagger.splits as splits_module
import psse_env.dagger.suite_builder as suite_builder_module
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
from psse_env.dagger.evaluation_gate import load_evaluation_policy
from psse_env.dagger.evaluator import (
    load_evaluation_suites,
    validate_release_scenario_suites,
)
from psse_env.dagger.protocol_bridge import unified_tool_schemas
from psse_env.dagger.rollout_collector import (
    DaggerRolloutCollector,
    audit_target_aware_state_classes,
)
from psse_env.dagger.replay_buffer import (
    DEFAULT_MINIMUM_TOOL_CATEGORY_DISTINCT_ROOTS,
    DEFAULT_MINIMUM_TOOL_CATEGORY_NATURAL_ROWS,
    DEFAULT_TRAINING_TOOL_CATEGORY_WEIGHTS,
    build_balanced_training_view,
)
from psse_env.dagger.release_audit import (
    audit_episode_against_truth as strict_audit_episode_against_truth,
)
from psse_env.dagger.sft_audit import (
    admissible_semantic_action_count,
    audit_approximate_teacher_realizability,
    audit_chat_sft_rows,
    audit_teacher_realizability,
    example_cost_margin,
)
from psse_env.dagger.splits import (
    audit_physical_split_disjointness,
    audit_stratified_split_coverage,
    physical_root_fingerprint,
    stratified_grouped_scenario_split,
)
from psse_env.dagger.suite_builder import validate_builder_environment
from psse_env.oracle import ExpertPolicyOracle
from psse_env.providers import MatpowerDeploymentProviders
from psse_env.providers.scenario_generator import Round0ScenarioGenerator
from psse_env.sft.provenance import file_sha256, git_source_state, stable_json_sha256
from psse_env.transactional_env import TransactionalPSSEEnv

DEFAULT_SEED = 20260719
BC0_AGGREGATE_SOURCE_PARTITION = "train"
DEFAULT_EVALUATION_SUITE_PATH = (
    Path(__file__).resolve().parents[1]
    / "dagger"
    / "suites"
    / "bc0_eval_suite_v1.json"
)
DEFAULT_EVALUATION_POLICY_PATH = evaluation_gate_module.DEFAULT_POLICY_PATH.resolve()

# Root-scenario counts per family at --scale 1.
DEFAULT_PLAN: dict[str, int] = {
    "no_error": 4,
    "measurement": 8,
    "multi_measurement": 20,
    "parameter": 8,
    "topology": 8,
    "harmonic": 4,
    # The clean checkout carries 17 independent tracked HIF roots.  HIF has a
    # separately reported handoff allowance until additional localization
    # roots are checked in.
    "hif": 17,
    "measurement+parameter": 22,
    "measurement+topology": 22,
    "measurement+hif": 2,
}

# Every positive-count release family has an explicit outcome contract.  The
# directly recoverable and deterministic diagnostic families must resolve all
# of their small default suites; the mixed parameter/topology suites permit at
# most one safe handoff in twenty.  Pure multi-measurement and HIF-bearing
# families keep an explicit handoff allowance until their observable evidence
# contracts can safely authorize autonomous continuation, and that allowance
# is reported rather than counted as recovery.
BC0_FAMILY_RELEASE_POLICY: dict[str, dict[str, float | int]] = {
    "no_error": {
        "minimum_physical_roots": 4,
        "minimum_resolution_rate": 1.0,
        "maximum_operator_escalation_rate": 0.0,
    },
    "measurement": {
        "minimum_physical_roots": 8,
        "minimum_resolution_rate": 1.0,
        "maximum_operator_escalation_rate": 0.0,
    },
    "multi_measurement": {
        "minimum_physical_roots": 20,
        "minimum_resolution_rate": 0.0,
        "maximum_operator_escalation_rate": 1.0,
    },
    "parameter": {
        "minimum_physical_roots": 8,
        "minimum_resolution_rate": 1.0,
        "maximum_operator_escalation_rate": 0.0,
    },
    "topology": {
        "minimum_physical_roots": 8,
        "minimum_resolution_rate": 1.0,
        "maximum_operator_escalation_rate": 0.0,
    },
    "harmonic": {
        "minimum_physical_roots": 4,
        "minimum_resolution_rate": 1.0,
        "maximum_operator_escalation_rate": 0.0,
    },
    # Explicit HIF-family handoff allowance: safe escalation is retained as a
    # reported outcome and never contributes to the resolution numerator.
    "hif": {
        "minimum_physical_roots": 17,
        "minimum_resolution_rate": 0.0,
        "maximum_operator_escalation_rate": 1.0,
    },
    "measurement+parameter": {
        "minimum_physical_roots": 22,
        "minimum_resolution_rate": 0.95,
        "maximum_operator_escalation_rate": 0.05,
    },
    "measurement+topology": {
        "minimum_physical_roots": 22,
        "minimum_resolution_rate": 0.95,
        "maximum_operator_escalation_rate": 0.05,
    },
    # The two-root composition remains an observability pilot.  Like pure HIF,
    # it may hand off safely but may not report that handoff as resolution.
    "measurement+hif": {
        "minimum_physical_roots": 2,
        "minimum_resolution_rate": 0.0,
        "maximum_operator_escalation_rate": 1.0,
    },
}

APPROXIMATE_REALIZABILITY_RELEASE_KWARGS: dict[str, Any] = {
    "quantization_bin": 0.25,
    "conflict_tolerance": 0.05,
    "nearest_neighbor_tolerance": 0.10,
    "perturbation_radius": 0.25,
    "local_perturbation_tolerance": 0.10,
    "minimum_nearest_neighbor_comparisons": 20,
    "minimum_nearest_neighbor_coverage": 0.10,
    "minimum_local_perturbation_comparisons": 10,
    "minimum_local_perturbation_coverage": 0.05,
    "require_cost_margins_for_multi_action": True,
}

# The global natural-population and balanced-view audits retain the full
# comparison-count and coverage gates above.  Family-level neighbor stability
# is additionally binding once the stratum contains at least this many
# independent physical roots.  Smaller families remain subject to every
# safety and margin check, plus the separate terminality gates below, but are
# reported as statistically
# underpowered rather than being forced to satisfy an impossible 20-pair
# floor.  State-class partitions are diagnostic projections: the class alone
# does not preserve target-specific rank structure, so their conflict,
# disagreement, validity, and margin checks remain binding while pair-count
# coverage is reported but not used as a second global gate.
BC0_FAMILY_NEIGHBOR_GATE_MINIMUM_ROOTS = 20
BC0_STATE_CLASS_NEIGHBOR_GATE_POLICY = "diagnostic_only"

# These are deliberately narrower than the full controller registry.  BC0 does
# not promise that every protocol tool is an SFT target, but it must supervise
# context acquisition for all three correctable physical modes, the matching
# correction actions, and transactional rollback on independent roots. Ten
# roots matches the existing category-support diversity floor.
BC0_CRITICAL_TARGET_TOOL_MINIMUM_DISTINCT_ROOTS: dict[str, int] = {
    "get_measurement_context": 10,
    "get_parameter_context": 10,
    "get_topology_context": 10,
    "correct_measurements": 10,
    "correct_parameters": 10,
    "correct_topology": 10,
    "rollback_state": 10,
}

# An action is only useful recovery supervision in the right lifecycle state.
# Context/correction targets after a partial acceptance keep the controller
# progressing through compound errors; clean-success targets cannot substitute
# for those cells. Likewise, rollback outside rejected-candidate recovery
# cannot satisfy the transactional recovery contract.
BC0_CRITICAL_TARGET_TOOL_STATE_CLASS_MINIMUM_DISTINCT_ROOTS: dict[
    str, dict[str, int]
] = {
    "commit_state": {
        "accepted_final_commit": 10,
        "accepted_partial_commit": 10,
    },
    "correct_measurements": {"accepted_partial_continuation": 5},
    "correct_parameters": {"accepted_partial_continuation": 5},
    "correct_topology": {"accepted_partial_continuation": 5},
    "get_measurement_context": {"accepted_partial_continuation": 10},
    "get_parameter_context": {"accepted_partial_continuation": 5},
    "get_topology_context": {"accepted_partial_continuation": 5},
    "rollback_state": {"rejected_candidate_recovery": 10},
}

# Corrections must also cover the families in which each error mode occurs;
# aggregate exact-action counts cannot substitute for these cells. Floors are
# bounded by the default physical-family plans: five for eight-root pure
# families, ten for 20/22-root multi/mixed families, and two for the two-root
# measurement+HIF observability pilot. HIF and multi-measurement handoffs are
# likewise kept as distinct observable regimes.
BC0_CRITICAL_TARGET_TOOL_SCENARIO_FAMILY_MINIMUM_DISTINCT_ROOTS: dict[
    str, dict[str, int]
] = {
    "ask_for_more_evidence": {
        "hif": 5,
        "measurement+hif": 2,
        "multi_measurement": 10,
    },
    "correct_measurements": {
        "measurement": 5,
        "measurement+hif": 2,
        "measurement+parameter": 10,
        "measurement+topology": 10,
        "multi_measurement": 10,
    },
    "correct_parameters": {
        "measurement+parameter": 10,
        "parameter": 5,
    },
    "correct_topology": {
        "measurement+topology": 10,
        "topology": 5,
    },
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
    active_physical_state: Mapping[str, Any] | None = None,
    remaining_truth: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Truth-side episode audit for round-0 quarantining.

    Never used as model supervision: it gates which episodes are exported,
    exactly like the existing target-aware collector audits.
    """
    return strict_audit_episode_against_truth(
        scenario,
        final_state,
        terminal=terminal,
        terminal_outcome=terminal_outcome,
        active_physical_state=active_physical_state,
        remaining_truth=remaining_truth,
        case_loader=matpower_provider_module._load_python_case,
    )


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


def _truth_free_execution_scenario(
    scenario: Mapping[str, Any],
) -> dict[str, Any]:
    """Remove offline audit truth before the observable teacher is executed."""
    execution = copy.deepcopy(dict(scenario))
    for key in list(execution):
        if (
            str(key).startswith("true_")
            or str(key).startswith("clean_")
            or key in {"hidden_truth", "oracle_action_hints"}
        ):
            execution.pop(key, None)
    return execution


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
        execution_scenario = _truth_free_execution_scenario(scenario)
        collector = DaggerRolloutCollector(
            env=env,
            policy=ObservableBaselinePolicy(),
            expert_oracle=oracle,
            rng=collector_rng,
        )
        rows = collector.collect_iteration(
            scenarios=[execution_scenario],
            iteration=0,
            beta=1.0,
            max_steps=args.max_steps,
        )
        for row in rows:
            row["episode_terminal_outcome"] = env.terminal_outcome
            labels = row.get("labels")
            if isinstance(labels, dict):
                labels["episode_terminal_outcome"] = env.terminal_outcome
        final_state = env.current_state()
        active_physical_state = env.store.get_state(
            str(final_state["active_state_id"])
        )
        audit = audit_episode_against_truth(
            scenario,
            final_state,
            terminal=env.is_terminal(),
            terminal_outcome=env.terminal_outcome,
            active_physical_state=active_physical_state,
            remaining_truth=None,
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


def _family_resolution_release_failures(
    terminal_matrix: Mapping[str, Mapping[str, Any]],
    *,
    policy: Mapping[str, Mapping[str, float | int]] = BC0_FAMILY_RELEASE_POLICY,
    plan: Mapping[str, int] | None = None,
) -> list[str]:
    failures: list[str] = []
    planned_families = (
        sorted(policy)
        if plan is None
        else sorted(family for family, count in plan.items() if int(count) > 0)
    )
    missing_policy = sorted(set(planned_families) - set(policy))
    if missing_policy:
        failures.append(
            "positive-count planned families lack BC0 release policy: "
            + ", ".join(missing_policy)
        )
    for family in planned_families:
        requirements = policy.get(family)
        if requirements is None:
            continue
        entry = terminal_matrix.get(family) or {}
        distinct_roots = int(entry.get("distinct_physical_roots") or 0)
        resolution_rate = float(entry.get("resolution_rate") or 0.0)
        escalation_rate = float(entry.get("operator_escalation_rate") or 0.0)
        missing_root_ids = list(
            entry.get("missing_physical_root_episode_ids") or []
        )
        minimum_roots = int(requirements["minimum_physical_roots"])
        minimum_resolution = float(requirements["minimum_resolution_rate"])
        maximum_escalation = float(requirements["maximum_operator_escalation_rate"])
        if distinct_roots < minimum_roots:
            failures.append(
                f"{family}: {distinct_roots} distinct physical roots < required "
                f"{minimum_roots}"
            )
        if missing_root_ids:
            failures.append(
                f"{family}: {len(missing_root_ids)} episode(s) lack an explicit "
                "physical_root_fingerprint"
            )
        if resolution_rate + 1e-12 < minimum_resolution:
            failures.append(
                f"{family}: resolution rate {resolution_rate:.3f} < required "
                f"{minimum_resolution:.3f}"
            )
        if escalation_rate - 1e-12 > maximum_escalation:
            failures.append(
                f"{family}: operator-escalation rate {escalation_rate:.3f} > allowed "
                f"{maximum_escalation:.3f}"
            )
    return failures


def _terminal_scenario_matrix(
    episode_audits: Iterable[Mapping[str, Any]],
    *,
    policy: Mapping[str, Mapping[str, float | int]] = BC0_FAMILY_RELEASE_POLICY,
) -> dict[str, dict[str, Any]]:
    """Summarize terminality and policy coverage with one vote per physical root.

    Episode-level identifiers remain in the report for debugging, but release
    rates and root floors are calculated only from explicit
    ``physical_root_fingerprint`` values. Repeated evaluation of the same root
    therefore cannot inflate either the resolution numerator or denominator.
    Conflicting duplicate outcomes fail terminal coverage conservatively.
    """
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
                "claimed_resolved_episode_ids": [],
                "resolved_episode_ids": [],
                "operator_escalation_episode_ids": [],
                "unknown_terminal_outcome_episode_ids": [],
                "episode_terminal_outcome_counts": Counter(),
                "missing_physical_root_episode_ids": [],
                "_audits_by_physical_root": {},
            },
        )
        entry["episodes"] += 1
        scenario_id = str(audit.get("scenario_id") or "unknown")
        fingerprint = audit.get("physical_root_fingerprint")
        if not isinstance(fingerprint, str) or not fingerprint.strip():
            entry["missing_physical_root_episode_ids"].append(scenario_id)
        else:
            entry["_audits_by_physical_root"].setdefault(
                fingerprint.strip(), []
            ).append((scenario_id, audit))
        if audit.get("terminal") is True:
            entry["terminal_episodes"] += 1
            outcome = str(audit.get("terminal_outcome") or "unknown")
            entry["episode_terminal_outcome_counts"][outcome] += 1
            if outcome == "resolved":
                entry["claimed_resolved_episode_ids"].append(scenario_id)
                if audit.get("quarantined") is not True:
                    entry["resolved_episode_ids"].append(scenario_id)
            elif outcome == "operator_escalation":
                entry["operator_escalation_episode_ids"].append(scenario_id)
            else:
                entry["unknown_terminal_outcome_episode_ids"].append(scenario_id)
        else:
            entry["nonterminal_episode_ids"].append(scenario_id)
        if audit.get("quarantined") is True:
            entry["quarantined_episode_ids"].append(scenario_id)
    for family, entry in grouped.items():
        audits_by_root = entry.pop("_audits_by_physical_root")
        entry["physical_root_fingerprints"] = sorted(audits_by_root)
        entry["distinct_physical_roots"] = len(audits_by_root)
        entry["duplicate_physical_root_fingerprints"] = {
            fingerprint: [scenario_id for scenario_id, _ in records]
            for fingerprint, records in sorted(audits_by_root.items())
            if len(records) > 1
        }
        terminal_roots: list[str] = []
        nonterminal_roots: list[str] = []
        quarantined_roots: list[str] = []
        claimed_resolved_roots: list[str] = []
        resolved_roots: list[str] = []
        escalation_roots: list[str] = []
        unknown_outcome_roots: list[str] = []
        conflicting_outcome_roots: list[str] = []
        root_outcome_counts: Counter[str] = Counter()
        for fingerprint, records in sorted(audits_by_root.items()):
            root_terminal = all(audit.get("terminal") is True for _, audit in records)
            root_quarantined = any(
                audit.get("quarantined") is True for _, audit in records
            )
            if root_quarantined:
                quarantined_roots.append(fingerprint)
            if not root_terminal:
                nonterminal_roots.append(fingerprint)
                continue
            terminal_roots.append(fingerprint)
            outcomes = {
                str(audit.get("terminal_outcome") or "unknown")
                for _, audit in records
            }
            if outcomes == {"resolved"}:
                claimed_resolved_roots.append(fingerprint)
                root_outcome_counts["resolved"] += 1
                if not root_quarantined:
                    resolved_roots.append(fingerprint)
            elif outcomes == {"operator_escalation"}:
                escalation_roots.append(fingerprint)
                root_outcome_counts["operator_escalation"] += 1
            elif len(outcomes) == 1:
                unknown_outcome_roots.append(fingerprint)
                root_outcome_counts[next(iter(outcomes))] += 1
            else:
                conflicting_outcome_roots.append(fingerprint)
                unknown_outcome_roots.append(fingerprint)
                root_outcome_counts["conflicting"] += 1

        entry["terminal_physical_root_fingerprints"] = terminal_roots
        entry["nonterminal_physical_root_fingerprints"] = nonterminal_roots
        entry["quarantined_physical_root_fingerprints"] = quarantined_roots
        entry["claimed_resolved_physical_root_fingerprints"] = (
            claimed_resolved_roots
        )
        entry["resolved_physical_root_fingerprints"] = resolved_roots
        entry["operator_escalation_physical_root_fingerprints"] = escalation_roots
        entry["unknown_terminal_outcome_physical_root_fingerprints"] = (
            unknown_outcome_roots
        )
        entry["conflicting_terminal_outcome_physical_root_fingerprints"] = (
            conflicting_outcome_roots
        )
        distinct_roots = entry["distinct_physical_roots"]
        entry["terminal_rate"] = (
            len(terminal_roots) / distinct_roots
            if distinct_roots
            else 0.0
        )
        entry["resolution_rate"] = (
            len(resolved_roots) / distinct_roots
            if distinct_roots
            else 0.0
        )
        entry["claimed_resolution_rate"] = (
            len(claimed_resolved_roots) / distinct_roots
            if distinct_roots
            else 0.0
        )
        entry["audit_verified_resolution_rate"] = entry["resolution_rate"]
        entry["operator_escalation_rate"] = (
            len(escalation_roots) / distinct_roots
            if distinct_roots
            else 0.0
        )
        entry["episode_terminal_outcome_counts"] = dict(
            sorted(entry["episode_terminal_outcome_counts"].items())
        )
        entry["terminal_outcome_counts"] = dict(
            sorted(root_outcome_counts.items())
        )
        entry["release_terminal_coverage"] = bool(
            distinct_roots
            and len(terminal_roots) == distinct_roots
            and not entry["missing_physical_root_episode_ids"]
            and not quarantined_roots
            and not unknown_outcome_roots
        )
        requirements = policy.get(family)
        minimum_roots = (
            int(requirements["minimum_physical_roots"])
            if requirements is not None
            else 1
        )
        minimum_resolution = (
            float(requirements["minimum_resolution_rate"])
            if requirements is not None
            else 1.0
        )
        maximum_escalation = (
            float(requirements["maximum_operator_escalation_rate"])
            if requirements is not None
            else 0.0
        )
        entry["release_resolution_coverage"] = bool(
            entry["release_terminal_coverage"]
            and distinct_roots >= minimum_roots
            and entry["resolution_rate"] + 1e-12 >= minimum_resolution
            and entry["operator_escalation_rate"] - 1e-12
            <= maximum_escalation
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


def _row_metadata_value(row: Mapping[str, Any], field: str) -> Any:
    if row.get(field) is not None:
        return row.get(field)
    metadata = row.get("metadata")
    if isinstance(metadata, Mapping):
        if metadata.get(field) is not None:
            return metadata.get(field)
        labels = metadata.get("labels")
        if isinstance(labels, Mapping):
            return labels.get(field)
    return None


def _apply_single_label_eligibility(row: dict[str, Any]) -> int:
    """Route unranked ambiguous teacher states out of production SFT."""
    labels = row.get("labels")
    semantic_action_count = admissible_semantic_action_count(row)
    row["admissible_semantic_action_count"] = semantic_action_count
    if isinstance(labels, dict):
        labels["admissible_semantic_action_count"] = semantic_action_count
    if (
        row.get("production_label_eligible") is not False
        and semantic_action_count > 1
        and example_cost_margin(row) is None
    ):
        row["production_label_eligible"] = False
        row["dataset_source"] = "dagger_unranked_multi_action_auxiliary"
        row["production_ineligibility_reason"] = (
            "multiple_semantic_actions_without_cost_margin"
        )
        if isinstance(labels, dict):
            labels["production_label_eligible"] = False
            labels["production_ineligibility_reason"] = (
                "multiple_semantic_actions_without_cost_margin"
            )
    return semantic_action_count


def _stratified_approximate_realizability(
    rows: Sequence[Mapping[str, Any]],
    field: str,
    *,
    required_values: Iterable[str] = (),
    neighbor_gate_minimum_distinct_roots: int | None = None,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        value = _row_metadata_value(row, field)
        grouped.setdefault(str(value if value is not None else "unknown"), []).append(row)
    for value in required_values:
        grouped.setdefault(str(value), [])
    reports: dict[str, dict[str, Any]] = {}
    for value, group in sorted(grouped.items()):
        report = audit_approximate_teacher_realizability(
            group,
            **APPROXIMATE_REALIZABILITY_RELEASE_KWARGS,
        )
        fingerprints = {
            str(fingerprint)
            for row in group
            if (
                fingerprint := _row_metadata_value(
                    row, "physical_root_fingerprint"
                )
            )
            not in (None, "")
        }
        missing_fingerprints = sum(
            _row_metadata_value(row, "physical_root_fingerprint")
            in (None, "")
            for row in group
        )
        nearest_tolerance = float(
            report.get("nearest_neighbor_tolerance", 0.0)
        )
        local_tolerance = report.get("local_perturbation_tolerance")
        safety_passed = bool(
            int(report.get("labeled_examples", 0)) > 0
            and not report.get("invalid_examples")
            and missing_fingerprints == 0
            and float(report.get("approximate_conflict_rate", 1.0))
            <= float(report.get("conflict_tolerance", 0.0))
            and float(
                report.get("nearest_neighbor_action_disagreement_rate", 1.0)
            )
            <= nearest_tolerance
            and (
                local_tolerance is None
                or float(
                    report.get(
                        "local_perturbation_action_disagreement_rate", 1.0
                    )
                )
                <= float(local_tolerance)
            )
            and float(report.get("multi_action_cost_margin_coverage", 0.0))
            >= 1.0
        )
        distinct_roots = len(fingerprints)
        neighbor_applicable = bool(
            neighbor_gate_minimum_distinct_roots is not None
            and distinct_roots >= int(neighbor_gate_minimum_distinct_roots)
        )
        if not safety_passed:
            gate_status = "failed_safety"
            release_gate_passed = False
        elif neighbor_gate_minimum_distinct_roots is None:
            gate_status = "safety_passed_neighbor_diagnostic_only"
            release_gate_passed = True
        elif not neighbor_applicable:
            gate_status = "safety_passed_neighbor_underpowered"
            release_gate_passed = True
        elif report.get("passed") is True:
            gate_status = "passed"
            release_gate_passed = True
        else:
            gate_status = "failed_neighbor_stability"
            release_gate_passed = False
        report.update(
            {
                "distinct_physical_roots": distinct_roots,
                "missing_physical_root_rows": missing_fingerprints,
                "stratified_safety_passed": safety_passed,
                "neighbor_stability_gate_applicable": neighbor_applicable,
                "neighbor_stability_minimum_distinct_roots": (
                    int(neighbor_gate_minimum_distinct_roots)
                    if neighbor_gate_minimum_distinct_roots is not None
                    else None
                ),
                "release_gate_status": gate_status,
                "release_gate_passed": release_gate_passed,
            }
        )
        reports[value] = report
    return reports


def _stratified_realizability_release_failures(
    reports: Mapping[str, Mapping[str, Any]], *, dimension: str
) -> list[str]:
    failures: list[str] = []
    for stratum, report in sorted(reports.items()):
        passed = report.get("release_gate_passed")
        if passed is None:
            passed = report.get("passed")
        if passed is True:
            continue
        failures.append(
            "approximate teacher realizability failed for "
            f"{dimension}={stratum}: examples={report.get('labeled_examples', 0)}, "
            f"roots={report.get('distinct_physical_roots', 'unknown')}, "
            f"status={report.get('release_gate_status', 'legacy_failed')}, "
            "nearest="
            f"{report.get('nearest_neighbor_compared_examples', 0)}/"
            f"{report.get('nearest_neighbor_comparison_coverage', 0.0):.3f}, "
            "local="
            f"{report.get('local_perturbation_compared_examples', 0)}/"
            f"{report.get('local_perturbation_comparison_coverage', 0.0):.3f}, "
            "margin_coverage="
            f"{report.get('multi_action_cost_margin_coverage', 0.0):.3f}"
        )
    return failures


def _generation_descriptor(
    args: argparse.Namespace,
    plan: Mapping[str, int],
    *,
    builder_environment: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind a generated aggregate to its source, schemas, and collection plan."""
    if builder_environment is None:
        builder_environment = validate_builder_environment()
    repo_root = Path(__file__).resolve().parents[2]
    source_files = (
        Path(__file__),
        Path(dataset_builder_module.__file__),
        Path(evaluation_gate_module.__file__),
        Path(evaluator_module.__file__),
        Path(protocol_bridge_module.__file__),
        Path(rollout_collector_module.__file__),
        Path(replay_buffer_module.__file__),
        Path(release_audit_module.__file__),
        Path(counterfactual_generator_module.__file__),
        Path(sft_audit_module.__file__),
        Path(splits_module.__file__),
        Path(suite_builder_module.__file__),
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
    input_corpora = _input_corpus_bindings(args, plan, repo_root=repo_root)
    parameter_artifacts = _tracked_parameter_artifact_binding(repo_root)
    evaluation_holdout = _evaluation_suite_binding(args, repo_root=repo_root)
    evaluation_policy = _evaluation_policy_binding(
        args,
        evaluation_holdout=evaluation_holdout,
        repo_root=repo_root,
    )
    return {
        "generation_provenance_version": 1,
        "builder_environment": copy.deepcopy(dict(builder_environment)),
        "source_state": git_source_state(repo_root),
        "protocol": args.protocol,
        "schema_registry_hash": stable_json_sha256(schemas),
        "generator_hashes": {
            str(path.resolve().relative_to(repo_root)): file_sha256(path)
            for path in source_files
        },
        "input_corpora": input_corpora,
        "input_artifacts": {"parameter_cases": parameter_artifacts},
        "evaluation_holdout": evaluation_holdout,
        "evaluation_policy": evaluation_policy,
        "generation_config": {
            "seed": args.seed,
            "source_partition": BC0_AGGREGATE_SOURCE_PARTITION,
            "plan": dict(sorted(plan.items())),
            "max_steps": args.max_steps,
            "counterfactuals_per_scenario": args.counterfactuals_per_scenario,
            "chi2_alpha": args.chi2_alpha,
            "hif_alpha_grid": args.hif_alpha_grid,
            "hif_r_grid": args.hif_r_grid,
            "hif_max_scans": args.hif_max_scans,
            "family_release_policy": BC0_FAMILY_RELEASE_POLICY,
            "critical_split_minimums": {"validation": 5, "test": 5},
            "stratified_approximate_realizability": {
                "global_population_gate": "full_release_thresholds",
                "family_neighbor_gate_minimum_distinct_roots": (
                    BC0_FAMILY_NEIGHBOR_GATE_MINIMUM_ROOTS
                ),
                "underpowered_family_status": "reported_not_applicable",
                "state_class_neighbor_gate": (
                    BC0_STATE_CLASS_NEIGHBOR_GATE_POLICY
                ),
                "always_binding_checks": [
                    "row_validity",
                    "physical_root_presence",
                    "quantized_conflict_rate",
                    "observed_neighbor_disagreement",
                    "observed_local_disagreement",
                    "multi_action_cost_margin_coverage",
                ],
            },
            "training_view": {
                "size_policy": (
                    "natural_target_bearing_train_row_count_with_bounded_replacement"
                ),
                "strict_target_axes": [
                    "target_tool_distinct_physical_roots",
                    "target_tool_x_state_class_distinct_physical_roots",
                    "target_tool_x_scenario_family_distinct_physical_roots",
                ],
                "deviation_gated_target_axes": ["tool_category"],
                "capacity_aware_target_axes": [
                    "tool_category",
                    "state_class",
                    "target_tool",
                    "scenario_family",
                    "error_cardinality",
                    "terminal_outcome",
                ],
                "capacity_aware_policy": (
                    "weighted_then_clip_and_redistribute_v1"
                ),
                "requirement_aware_reservation_policy": (
                    "constrained_first_distinct_physical_root_preselection_v1"
                ),
                "configured_tool_category_weights": dict(
                    DEFAULT_TRAINING_TOOL_CATEGORY_WEIGHTS
                ),
                "tool_category_natural_support_floor": {
                    "minimum_natural_target_bearing_rows": (
                        DEFAULT_MINIMUM_TOOL_CATEGORY_NATURAL_ROWS
                    ),
                    "minimum_distinct_roots": (
                        DEFAULT_MINIMUM_TOOL_CATEGORY_DISTINCT_ROOTS
                    ),
                },
                "target_tool_minimum_distinct_physical_roots": dict(
                    BC0_CRITICAL_TARGET_TOOL_MINIMUM_DISTINCT_ROOTS
                ),
                "target_tool_state_class_minimum_distinct_physical_roots": (
                    copy.deepcopy(
                        BC0_CRITICAL_TARGET_TOOL_STATE_CLASS_MINIMUM_DISTINCT_ROOTS
                    )
                ),
                "target_tool_scenario_family_minimum_distinct_physical_roots": (
                    copy.deepcopy(
                        BC0_CRITICAL_TARGET_TOOL_SCENARIO_FAMILY_MINIMUM_DISTINCT_ROOTS
                    )
                ),
                "production_label_eligibility_policy": (
                    "explicit_true_required"
                ),
                "max_duplicate_count": 2,
                "low_cost_margin_threshold": 0.05,
                "maximum_tool_category_target_deviation": 0.10,
            },
        },
    }


def _configured_input_corpora(
    args: argparse.Namespace,
    plan: Mapping[str, int],
) -> dict[str, Path]:
    """Return the exact corpus files that the round-0 generator may read.

    Release generation uses the independently generated multiscan HIF corpus;
    the compact curated fallback is reserved for the frozen evaluation
    holdout.  Its QA metadata is also bound below so a release cannot silently
    substitute different physics-validation evidence.
    """
    configured: dict[str, Path] = {
        "measurement_corpus": Path(
            getattr(args, "measurement_corpus", None)
            or scenario_generator_module.DEFAULT_CORPUS_PATH
        ),
    }
    if any(int(plan.get(family, 0)) > 0 for family in ("hif", "measurement+hif")):
        hif_corpora = getattr(args, "hif_corpus", None)
        selected_hif_corpora = (
            tuple(Path(path) for path in hif_corpora)
            if hif_corpora
            else tuple(
                Path(path)
                for path in scenario_generator_module.DEFAULT_RELEASE_HIF_SAMPLE_PATHS
            )
        )
        for index, path in enumerate(selected_hif_corpora):
            configured[f"hif_corpus_{index}"] = path
        if not hif_corpora:
            for index, path in enumerate(
                scenario_generator_module.DEFAULT_RELEASE_HIF_QUALITY_PATHS
            ):
                configured[f"hif_quality_{index}"] = Path(path)
    if any(
        int(plan.get(family, 0)) > 0
        for family in ("three_phase_unbalance", "telemetry_no_disturbance")
    ):
        configured["three_phase_unbalance_corpus"] = Path(
            getattr(args, "imbalance_corpus", None)
            or scenario_generator_module.DEFAULT_IMBALANCE_SAMPLE_PATH
        )
    return configured


def _assert_training_view_export_integrity(
    selected_rows: Sequence[Mapping[str, Any]],
    exported_rows: Sequence[Mapping[str, Any]],
) -> None:
    """Fail closed if export drops or substitutes a balanced-view row."""
    selected_ids = [str(row.get("example_id")) for row in selected_rows]
    exported_ids = [str(row.get("example_id")) for row in exported_rows]
    if len(selected_rows) != len(exported_rows):
        raise RuntimeError(
            "Balanced training-view export cardinality mismatch: "
            f"selected={len(selected_rows)}, exported={len(exported_rows)}. "
            "Every selected row must carry an exportable expert target."
        )
    if Counter(selected_ids) != Counter(exported_ids):
        raise RuntimeError(
            "Balanced training-view export identity mismatch: exported rows did "
            "not preserve the selected example-id multiset."
        )


def _git_tracks_file(repo_root: Path, path: Path) -> bool:
    try:
        relative = path.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return False
    completed = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", relative.as_posix()],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0


def _input_corpus_bindings(
    args: argparse.Namespace,
    plan: Mapping[str, int],
    *,
    repo_root: Path,
) -> dict[str, dict[str, Any]]:
    """Content-address every corpus and record its release trust boundary."""
    bindings: dict[str, dict[str, Any]] = {}
    for name, configured_path in sorted(_configured_input_corpora(args, plan).items()):
        path = configured_path.resolve()
        try:
            path_label = path.relative_to(repo_root.resolve()).as_posix()
        except ValueError:
            path_label = str(path)
        exists = path.is_file()
        bindings[name] = {
            "path": path_label,
            "sha256": file_sha256(path) if exists else None,
            "exists": exists,
            "git_tracked": exists and _git_tracks_file(repo_root, path),
        }
    return bindings


def _input_corpus_release_failures(
    generation_descriptor: Mapping[str, Any],
) -> list[str]:
    failures: list[str] = []
    bindings = generation_descriptor.get("input_corpora")
    if not isinstance(bindings, Mapping) or not bindings:
        return ["generation descriptor has no input corpus bindings"]
    for name, raw_binding in sorted(bindings.items()):
        binding = raw_binding if isinstance(raw_binding, Mapping) else {}
        path = binding.get("path")
        if binding.get("exists") is not True or not binding.get("sha256"):
            failures.append(f"input corpus is missing or unhashed: {name} ({path})")
        if binding.get("git_tracked") is not True:
            failures.append(f"input corpus is not repository-tracked: {name} ({path})")
    return failures


def _path_label(path: Path, *, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _duplicate_counts(values: Sequence[str]) -> dict[str, int]:
    return {
        value: count
        for value, count in sorted(Counter(values).items())
        if count > 1
    }


def _uses_current_physical_fingerprint(value: str) -> bool:
    prefix = f"physical_v{splits_module.PHYSICAL_FINGERPRINT_VERSION}_"
    digest = value[len(prefix) :] if value.startswith(prefix) else ""
    return len(digest) == 64 and all(
        character in "0123456789abcdef" for character in digest
    )


def _evaluation_suite_semantic_identity(
    path: Path,
    *,
    repo_root: Path,
) -> dict[str, Any]:
    """Load, validate, and content-address the frozen release holdout."""

    resolved = path.resolve()
    exists = resolved.is_file()
    identity: dict[str, Any] = {
        "path": _path_label(resolved, repo_root=repo_root),
        "file_sha256": file_sha256(resolved) if exists else None,
        "exists": exists,
        "git_tracked": exists and _git_tracks_file(repo_root, resolved),
        "schema_valid": False,
        "validation_error": None,
        "suite_names": [],
        "suite_count": 0,
        "episode_count": 0,
        "physical_root_count": 0,
        "physical_root_set_sha256": None,
        "scenario_id_count": 0,
        "scenario_id_set_sha256": None,
        "duplicate_physical_roots": {},
        "duplicate_scenario_ids": {},
        "missing_physical_root_entries": [],
        "invalid_physical_root_version_entries": [],
        "physical_fingerprint_version": splits_module.PHYSICAL_FINGERPRINT_VERSION,
        "missing_scenario_id_entries": [],
    }
    if not exists:
        identity["validation_error"] = "evaluation suite file is missing"
        return identity

    try:
        suites = validate_release_scenario_suites(load_evaluation_suites(resolved))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        identity["validation_error"] = str(exc)
        return identity

    physical_roots: list[str] = []
    scenario_ids: list[str] = []
    missing_roots: list[str] = []
    invalid_root_versions: list[str] = []
    missing_ids: list[str] = []
    episode_count = 0
    for suite_name in sorted(suites):
        for index, scenario in enumerate(suites[suite_name]):
            episode_count += 1
            label = f"{suite_name}[{index}]"
            grouping = scenario.get("grouping")
            grouping = grouping if isinstance(grouping, Mapping) else {}
            root = grouping.get("physical_root_fingerprint")
            if not isinstance(root, str) or not root.strip():
                missing_roots.append(label)
            else:
                normalized_root = root.strip()
                physical_roots.append(normalized_root)
                if not _uses_current_physical_fingerprint(normalized_root):
                    invalid_root_versions.append(label)
            execution = scenario.get("execution")
            execution = execution if isinstance(execution, Mapping) else {}
            scenario_id = execution.get("scenario_id")
            if not isinstance(scenario_id, str) or not scenario_id.strip():
                missing_ids.append(label)
            else:
                scenario_ids.append(scenario_id.strip())

    root_set = sorted(set(physical_roots))
    scenario_id_set = sorted(set(scenario_ids))
    identity.update(
        {
            "schema_valid": True,
            "suite_names": sorted(suites),
            "suite_count": len(suites),
            "episode_count": episode_count,
            "physical_root_count": len(root_set),
            "physical_root_set_sha256": stable_json_sha256(root_set),
            "scenario_id_count": len(scenario_id_set),
            "scenario_id_set_sha256": stable_json_sha256(scenario_id_set),
            "duplicate_physical_roots": _duplicate_counts(physical_roots),
            "duplicate_scenario_ids": _duplicate_counts(scenario_ids),
            "missing_physical_root_entries": missing_roots,
            "invalid_physical_root_version_entries": invalid_root_versions,
            "missing_scenario_id_entries": missing_ids,
        }
    )
    return identity


def _evaluation_suite_binding(
    args: argparse.Namespace,
    *,
    repo_root: Path,
) -> dict[str, Any]:
    configured = Path(
        getattr(args, "evaluation_suite", None) or DEFAULT_EVALUATION_SUITE_PATH
    )
    return _evaluation_suite_semantic_identity(configured, repo_root=repo_root)


def _evaluation_policy_semantic_identity(
    path: Path,
    *,
    evaluation_holdout: Mapping[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    """Bind the holdout to the exact suite approved by the release policy."""

    resolved = path.resolve()
    exists = resolved.is_file()
    identity: dict[str, Any] = {
        "path": _path_label(resolved, repo_root=repo_root),
        "file_sha256": file_sha256(resolved) if exists else None,
        "exists": exists,
        "git_tracked": exists and _git_tracks_file(repo_root, resolved),
        "schema_valid": False,
        "validation_error": None,
        "policy_id": None,
        "matches_packaged_policy": False,
        "suite_policy_status": None,
        "approved_suite_sha256": None,
        "approved_manifest_root_set_sha256": None,
        "suite_file_sha256_matches_approval": False,
        "suite_root_set_sha256_matches_approval": False,
        "approval_passed": False,
        "failures": [],
    }
    if not exists:
        identity["validation_error"] = "evaluation policy file is missing"
    else:
        try:
            policy = load_evaluation_policy(resolved)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            identity["validation_error"] = str(exc)
        else:
            suite_policy = policy["suite_policy"]
            approved_manifest = suite_policy.get("approved_suite_manifest")
            approved_manifest = (
                approved_manifest if isinstance(approved_manifest, Mapping) else {}
            )
            approved_suite_sha256 = suite_policy.get("approved_suite_sha256")
            approved_root_sha256 = approved_manifest.get("root_set_sha256")
            file_matches = bool(
                approved_suite_sha256
                and evaluation_holdout.get("file_sha256")
                == approved_suite_sha256
            )
            root_matches = bool(
                approved_root_sha256
                and evaluation_holdout.get("physical_root_set_sha256")
                == approved_root_sha256
            )
            identity.update(
                {
                    "schema_valid": True,
                    "policy_id": policy.get("policy_id"),
                    "matches_packaged_policy": bool(
                        policy.get("policy_id")
                        == evaluation_gate_module.DEFAULT_POLICY_ID
                        and identity["file_sha256"]
                        == file_sha256(DEFAULT_EVALUATION_POLICY_PATH)
                    ),
                    "suite_policy_status": suite_policy.get("status"),
                    "approved_suite_sha256": approved_suite_sha256,
                    "approved_manifest_root_set_sha256": approved_root_sha256,
                    "suite_file_sha256_matches_approval": file_matches,
                    "suite_root_set_sha256_matches_approval": root_matches,
                }
            )

    failures: list[str] = []
    if identity["exists"] is not True:
        failures.append("evaluation policy is missing")
    if identity["git_tracked"] is not True:
        failures.append("evaluation policy is not repository-tracked")
    if identity["schema_valid"] is not True:
        failures.append(
            "evaluation policy schema is invalid: "
            + str(identity["validation_error"] or "unknown validation error")
        )
    if identity["matches_packaged_policy"] is not True:
        failures.append("evaluation policy does not match the packaged BC0 policy")
    if identity["suite_policy_status"] != "pinned":
        failures.append("evaluation policy suite status is not pinned")
    if identity["suite_file_sha256_matches_approval"] is not True:
        failures.append(
            "evaluation suite file SHA-256 does not match the policy approval"
        )
    if identity["suite_root_set_sha256_matches_approval"] is not True:
        failures.append(
            "evaluation suite root-set SHA-256 does not match the policy approval"
        )
    identity["failures"] = failures
    identity["approval_passed"] = not failures
    return identity


def _evaluation_policy_binding(
    args: argparse.Namespace,
    *,
    evaluation_holdout: Mapping[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    configured = Path(
        getattr(args, "evaluation_policy", None) or DEFAULT_EVALUATION_POLICY_PATH
    )
    return _evaluation_policy_semantic_identity(
        configured,
        evaluation_holdout=evaluation_holdout,
        repo_root=repo_root,
    )


def _holdout_disjointness_report(
    scenarios: Sequence[Mapping[str, Any]],
    *,
    evaluation_holdout: Mapping[str, Any],
    evaluation_policy: Mapping[str, Any],
    evaluation_suite_path: Path,
    evaluation_policy_path: Path,
    repo_root: Path,
    raise_on_overlap: bool = True,
) -> dict[str, Any]:
    """Audit the aggregate roots against the exact bound evaluation suite."""

    current = _evaluation_suite_semantic_identity(
        evaluation_suite_path, repo_root=repo_root
    )
    current_policy = _evaluation_policy_semantic_identity(
        evaluation_policy_path,
        evaluation_holdout=current,
        repo_root=repo_root,
    )
    aggregate_roots: list[str] = []
    aggregate_ids: list[str] = []
    missing_aggregate_roots: list[str] = []
    invalid_aggregate_root_versions: list[str] = []
    missing_aggregate_ids: list[str] = []
    for index, scenario in enumerate(scenarios):
        label = str(
            scenario.get("scenario_id")
            or scenario.get("root_scenario_id")
            or f"scenario[{index}]"
        )
        root = scenario.get("physical_root_fingerprint")
        if not isinstance(root, str) or not root.strip():
            missing_aggregate_roots.append(label)
        else:
            normalized_root = root.strip()
            aggregate_roots.append(normalized_root)
            if not _uses_current_physical_fingerprint(normalized_root):
                invalid_aggregate_root_versions.append(label)
        scenario_id = scenario.get("scenario_id")
        if not isinstance(scenario_id, str) or not scenario_id.strip():
            missing_aggregate_ids.append(label)
        else:
            aggregate_ids.append(scenario_id.strip())

    aggregate_root_set = sorted(set(aggregate_roots))
    aggregate_id_set = sorted(set(aggregate_ids))
    holdout_roots = set()
    holdout_ids = set()
    if current.get("schema_valid") is True:
        suites = validate_release_scenario_suites(
            load_evaluation_suites(evaluation_suite_path)
        )
        for rows in suites.values():
            for scenario in rows:
                grouping = scenario.get("grouping")
                grouping = grouping if isinstance(grouping, Mapping) else {}
                root = grouping.get("physical_root_fingerprint")
                if isinstance(root, str) and root.strip():
                    holdout_roots.add(root.strip())
                execution = scenario.get("execution")
                execution = execution if isinstance(execution, Mapping) else {}
                scenario_id = execution.get("scenario_id")
                if isinstance(scenario_id, str) and scenario_id.strip():
                    holdout_ids.add(scenario_id.strip())

    root_intersection = sorted(set(aggregate_root_set) & holdout_roots)
    id_intersection = sorted(set(aggregate_id_set) & holdout_ids)
    binding_fields = (
        "path",
        "file_sha256",
        "physical_root_count",
        "physical_root_set_sha256",
        "physical_fingerprint_version",
        "scenario_id_count",
        "scenario_id_set_sha256",
    )
    binding_mismatches = [
        field
        for field in binding_fields
        if evaluation_holdout.get(field) != current.get(field)
    ]
    policy_binding_fields = (
        "path",
        "file_sha256",
        "policy_id",
        "matches_packaged_policy",
        "suite_policy_status",
        "approved_suite_sha256",
        "approved_manifest_root_set_sha256",
    )
    policy_binding_mismatches = [
        field
        for field in policy_binding_fields
        if evaluation_policy.get(field) != current_policy.get(field)
    ]
    failures: list[str] = []
    if current.get("exists") is not True:
        failures.append("evaluation suite is missing")
    if current.get("git_tracked") is not True:
        failures.append("evaluation suite is not repository-tracked")
    if current.get("schema_valid") is not True:
        failures.append(
            "evaluation suite schema is invalid: "
            + str(current.get("validation_error") or "unknown validation error")
        )
    if current.get("missing_physical_root_entries"):
        failures.append("evaluation suite contains entries without physical roots")
    if current.get("invalid_physical_root_version_entries"):
        failures.append(
            "evaluation suite contains physical roots from an obsolete or invalid "
            "fingerprint version"
        )
    if current.get("missing_scenario_id_entries"):
        failures.append("evaluation suite contains entries without scenario IDs")
    if current.get("duplicate_physical_roots"):
        failures.append("evaluation suite contains duplicate physical roots")
    if current.get("duplicate_scenario_ids"):
        failures.append("evaluation suite contains duplicate scenario IDs")
    if binding_mismatches:
        failures.append(
            "evaluation suite no longer matches its generation binding: "
            + ", ".join(binding_mismatches)
        )
    current_policy_failures = current_policy.get("failures")
    if isinstance(current_policy_failures, Sequence) and not isinstance(
        current_policy_failures, (str, bytes)
    ):
        failures.extend(str(failure) for failure in current_policy_failures)
    else:
        failures.append("evaluation policy approval report is invalid")
    if policy_binding_mismatches:
        failures.append(
            "evaluation policy no longer matches its generation binding: "
            + ", ".join(policy_binding_mismatches)
        )
    if missing_aggregate_roots:
        failures.append("aggregate contains scenarios without physical roots")
    if invalid_aggregate_root_versions:
        failures.append(
            "aggregate contains physical roots from an obsolete or invalid "
            "fingerprint version"
        )
    if missing_aggregate_ids:
        failures.append("aggregate contains scenarios without scenario IDs")
    aggregate_root_duplicates = _duplicate_counts(aggregate_roots)
    aggregate_id_duplicates = _duplicate_counts(aggregate_ids)
    if aggregate_root_duplicates:
        failures.append("aggregate contains duplicate physical roots")
    if aggregate_id_duplicates:
        failures.append("aggregate contains duplicate scenario IDs")
    if root_intersection:
        failures.append("aggregate and evaluation suite share physical roots")
    if id_intersection:
        failures.append("aggregate and evaluation suite share scenario IDs")

    report = {
        "holdout_schema_version": 1,
        "passed": not failures,
        "failures": failures,
        "evaluation_suite": current,
        "binding_mismatches": binding_mismatches,
        "evaluation_policy": current_policy,
        "policy_binding_mismatches": policy_binding_mismatches,
        "aggregate": {
            "scenario_count": len(scenarios),
            "physical_root_count": len(aggregate_root_set),
            "physical_root_set_sha256": stable_json_sha256(aggregate_root_set),
            "scenario_id_count": len(aggregate_id_set),
            "scenario_id_set_sha256": stable_json_sha256(aggregate_id_set),
            "duplicate_physical_roots": aggregate_root_duplicates,
            "duplicate_scenario_ids": aggregate_id_duplicates,
            "missing_physical_root_entries": missing_aggregate_roots,
            "invalid_physical_root_version_entries": invalid_aggregate_root_versions,
            "missing_scenario_id_entries": missing_aggregate_ids,
        },
        "physical_root_intersection_count": len(root_intersection),
        "physical_root_intersection": root_intersection,
        "scenario_id_intersection_count": len(id_intersection),
        "scenario_id_intersection": id_intersection,
    }
    if raise_on_overlap and (root_intersection or id_intersection):
        raise RuntimeError(
            "Frozen evaluation holdout overlaps the round-0 aggregate: "
            f"physical_roots={len(root_intersection)}, "
            f"scenario_ids={len(id_intersection)}"
        )
    return report


def _holdout_release_failures(report: Mapping[str, Any]) -> list[str]:
    if report.get("passed") is True:
        return []
    failures = report.get("failures")
    if not isinstance(failures, Sequence) or isinstance(failures, (str, bytes)):
        return ["evaluation holdout disjointness report is invalid"]
    return [f"evaluation holdout gate failed: {failure}" for failure in failures]


def _has_symlink_component(repo_root: Path, path: Path) -> bool:
    relative = path.absolute().relative_to(repo_root.absolute())
    current = repo_root.absolute()
    for component in relative.parts:
        current /= component
        if current.is_symlink():
            return True
    return False


def _tracked_parameter_artifact_binding(repo_root: Path) -> dict[str, Any]:
    """Bind the clean tracked parameter-case tree used by corpus basenames."""

    root = (
        Path(scenario_generator_module.DEFAULT_BALANCED_ARTIFACT_DIR)
        / "cases_parameter_error"
    ).absolute()
    relative_root = root.relative_to(repo_root.absolute()).as_posix()
    try:
        tracked = subprocess.run(
            ["git", "ls-files", "-z", "--", relative_root],
            cwd=repo_root,
            check=True,
            capture_output=True,
        ).stdout
        dirty = subprocess.run(
            ["git", "diff", "--name-only", "-z", "HEAD", "--", relative_root],
            cwd=repo_root,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", b"")
        message = detail.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            "parameter artifact provenance requires an intact Git checkout"
            + (f": {message}" if message else "")
        ) from exc
    dirty_paths = sorted(
        item.decode("utf-8") for item in dirty.split(b"\0") if item
    )
    if dirty_paths:
        raise RuntimeError(
            "parameter artifact inputs differ from HEAD: " + ", ".join(dirty_paths)
        )
    relative_files = sorted(
        item.decode("utf-8") for item in tracked.split(b"\0") if item
    )
    if not relative_files:
        raise RuntimeError("no tracked parameter case artifacts are available")
    paths = [repo_root / relative for relative in relative_files]
    invalid = [
        relative
        for relative, path in zip(relative_files, paths)
        if not path.is_file() or _has_symlink_component(repo_root, path)
    ]
    if invalid:
        raise RuntimeError(
            "parameter artifact inputs are missing or symlinked: "
            + ", ".join(invalid)
        )
    file_hashes = {
        relative: file_sha256(path)
        for relative, path in zip(relative_files, paths)
    }
    return {
        "root": relative_root,
        "file_count": len(file_hashes),
        "files": file_hashes,
        "tree_sha256": stable_json_sha256(file_hashes),
    }


def _input_artifact_release_failures(
    generation_descriptor: Mapping[str, Any],
) -> list[str]:
    artifacts = generation_descriptor.get("input_artifacts")
    parameter = artifacts.get("parameter_cases") if isinstance(artifacts, Mapping) else None
    if not isinstance(parameter, Mapping):
        return ["generation descriptor has no parameter artifact binding"]
    files = parameter.get("files")
    if not isinstance(files, Mapping) or not files:
        return ["parameter artifact binding contains no files"]
    failures: list[str] = []
    if parameter.get("file_count") != len(files):
        failures.append("parameter artifact file count is inconsistent")
    if parameter.get("tree_sha256") != stable_json_sha256(dict(files)):
        failures.append("parameter artifact tree hash is inconsistent")
    return failures


def generate(args: argparse.Namespace) -> dict[str, Any]:
    # Aggregate topology roots and their physical-v3 fingerprints depend on
    # the same numerical stack as the frozen evaluation suite.  Validate it
    # before reading the plan or any corpus so a clean but incompatible host
    # cannot emit release-eligible evidence with different physical roots.
    builder_environment = validate_builder_environment()
    plan = {family: count * args.scale for family, count in DEFAULT_PLAN.items()}
    if args.plan:
        plan = json.loads(Path(args.plan).read_text()) if Path(args.plan).is_file() else json.loads(args.plan)
    generation_descriptor = _generation_descriptor(
        args,
        plan,
        builder_environment=builder_environment,
    )
    generation_provenance_id = stable_json_sha256(generation_descriptor)
    repo_root = Path(__file__).resolve().parents[2]
    configured_corpora = _configured_input_corpora(args, plan)
    hif_corpus_paths = [
        path
        for name, path in sorted(configured_corpora.items())
        if name.startswith("hif_corpus_")
    ]
    generator = Round0ScenarioGenerator(
        corpus_path=configured_corpora["measurement_corpus"],
        hif_sample_paths=hif_corpus_paths,
        imbalance_sample_path=configured_corpora.get(
            "three_phase_unbalance_corpus",
            scenario_generator_module.DEFAULT_IMBALANCE_SAMPLE_PATH,
        ),
        balanced_artifact_dir=scenario_generator_module.DEFAULT_BALANCED_ARTIFACT_DIR,
        artifact_allowlist=[
            repo_root / relative
            for relative in generation_descriptor["input_artifacts"]["parameter_cases"]["files"]
        ],
        seed=args.seed,
        source_partition=BC0_AGGREGATE_SOURCE_PARTITION,
        chi2_alpha=args.chi2_alpha,
        hif_max_scans=args.hif_max_scans,
    )
    scenarios = generator.build(plan)
    if not scenarios:
        raise RuntimeError("Scenario generator produced no scenarios for the plan.")

    physical_by_root = {}
    scenario_by_root: dict[str, dict[str, Any]] = {}
    for scenario in scenarios:
        fingerprint = physical_root_fingerprint(scenario)
        scenario["physical_root_fingerprint"] = fingerprint
        root_id = str(scenario["root_scenario_id"])
        physical_by_root[root_id] = fingerprint
        scenario_by_root[root_id] = scenario

    evaluation_suite_path = Path(
        getattr(args, "evaluation_suite", None) or DEFAULT_EVALUATION_SUITE_PATH
    )
    evaluation_policy_path = Path(
        getattr(args, "evaluation_policy", None) or DEFAULT_EVALUATION_POLICY_PATH
    )
    holdout_disjointness = _holdout_disjointness_report(
        scenarios,
        evaluation_holdout=generation_descriptor["evaluation_holdout"],
        evaluation_policy=generation_descriptor["evaluation_policy"],
        evaluation_suite_path=evaluation_suite_path,
        evaluation_policy_path=evaluation_policy_path,
        repo_root=repo_root,
    )

    # Assign the physical-root split before any DAgger or counterfactual
    # descendants are generated. Every later row inherits this immutable
    # assignment; it is never independently hashed by scenario/branch ID.
    root_split_rows = [
        {
            "root_scenario_id": str(scenario["root_scenario_id"]),
            "physical_root_fingerprint": scenario["physical_root_fingerprint"],
            "case_id": str(scenario.get("network_case") or scenario.get("case")),
            "error_family_combination": str(scenario["scenario_family"]),
            "error_cardinality": int(scenario["error_cardinality"]),
            "source_tier": str(scenario["source_tier"]),
        }
        for scenario in scenarios
    ]
    split_floor = {"validation": 5, "test": 5}
    critical_families = sorted(
        family
        for family, requirements in BC0_FAMILY_RELEASE_POLICY.items()
        if int(plan.get(family, 0)) > 0
        and int(requirements["minimum_physical_roots"])
        >= sum(split_floor.values())
    )
    available_roots = Counter(
        str(scenario["scenario_family"]) for scenario in scenarios
    )
    enforceable_critical_families = [
        family
        for family in critical_families
        if available_roots[family] >= sum(split_floor.values())
    ]
    root_splits = stratified_grouped_scenario_split(
        root_split_rows,
        train_fraction=0.75,
        validation_fraction=0.15,
        seed=args.seed,
        critical_families=enforceable_critical_families,
        minimum_roots_per_critical_family=(
            split_floor if enforceable_critical_families else None
        ),
    )
    split_coverage = audit_stratified_split_coverage(
        root_splits,
        critical_families=critical_families,
        minimum_roots_per_critical_family=split_floor,
    )
    split_by_root = {
        str(row["root_scenario_id"]): split_name
        for split_name, rows in root_splits.items()
        for row in rows
    }
    for scenario in scenarios:
        scenario["dataset_split"] = split_by_root[str(scenario["root_scenario_id"])]

    collected = collect_round0(args, scenarios)
    audit_by_root = {
        str(audit["scenario_id"]): audit for audit in collected["episode_audits"]
    }
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
        scenario = scenario_by_root[root_id]
        row["scenario_family"] = scenario.get("scenario_family")
        row["error_cardinality"] = scenario.get("error_cardinality")
        row["network_case"] = scenario.get("network_case")
        row["source_tier"] = scenario.get("source_tier")
        if root_id in audit_by_root:
            row["episode_terminal_outcome"] = audit_by_root[root_id].get(
                "terminal_outcome"
            )
        labels = row.get("labels")
        if isinstance(labels, dict):
            for field in (
                "scenario_family",
                "error_cardinality",
                "network_case",
                "source_tier",
                "episode_terminal_outcome",
            ):
                labels[field] = row.get(field)
        row["dataset_split"] = split_by_root[root_id]
        row["generation_provenance_id"] = generation_provenance_id
        # A rank-one label is not release-grade supervision when the teacher
        # exposed multiple genuinely distinct admissible actions but supplied
        # no Q-cost ordering/margin. Keep the trace for ranking/auxiliary work,
        # but fail closed out of single-label production SFT.
        _apply_single_label_eligibility(row)
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
    train_view_rows, training_view_report = build_balanced_training_view(
        splits["train"],
        seed=args.seed,
        max_duplicate_count=2,
        low_cost_margin_threshold=0.05,
        maximum_tool_category_target_deviation=0.10,
        minimum_tool_category_natural_rows=(
            DEFAULT_MINIMUM_TOOL_CATEGORY_NATURAL_ROWS
        ),
        minimum_tool_category_distinct_roots=(
            DEFAULT_MINIMUM_TOOL_CATEGORY_DISTINCT_ROOTS
        ),
        target_tool_minimum_distinct_roots=(
            BC0_CRITICAL_TARGET_TOOL_MINIMUM_DISTINCT_ROOTS
        ),
        target_tool_state_class_minimum_distinct_roots=(
            BC0_CRITICAL_TARGET_TOOL_STATE_CLASS_MINIMUM_DISTINCT_ROOTS
        ),
        target_tool_scenario_family_minimum_distinct_roots=(
            BC0_CRITICAL_TARGET_TOOL_SCENARIO_FAMILY_MINIMUM_DISTINCT_ROOTS
        ),
        require_production_label_eligible=True,
    )
    exported_train_view = examples_to_chat_sft(
        train_view_rows, protocol=args.protocol
    )
    _assert_training_view_export_integrity(
        train_view_rows, exported_train_view
    )
    exported = {
        "train_view": exported_train_view,
        **{
            name: examples_to_chat_sft(rows, protocol=args.protocol)
            for name, rows in splits.items()
            if name in {"validation", "test"} and rows
        },
    }
    all_exported = [row for rows in exported.values() for row in rows]
    natural_exported_by_split = {
        name: examples_to_chat_sft(rows, protocol=args.protocol)
        for name, rows in splits.items()
    }
    natural_exported = [
        row for rows in natural_exported_by_split.values() for row in rows
    ]
    # Release realizability is measured on the immutable natural population.
    # The duplicate-capable balanced training view is checked separately so it
    # cannot inflate comparison counts or hide omitted natural train rows.
    chat_audit = audit_chat_sft_rows(natural_exported)
    training_view_chat_audit = audit_chat_sft_rows(exported["train_view"])
    realizability = audit_teacher_realizability(
        natural_exported, conflict_tolerance=0.0
    )
    training_view_realizability = audit_teacher_realizability(
        exported["train_view"], conflict_tolerance=0.0
    )
    approximate_realizability = audit_approximate_teacher_realizability(
        natural_exported,
        **APPROXIMATE_REALIZABILITY_RELEASE_KWARGS,
    )
    training_view_approximate_realizability = (
        audit_approximate_teacher_realizability(
            exported["train_view"],
            **APPROXIMATE_REALIZABILITY_RELEASE_KWARGS,
        )
    )
    approximate_by_family = _stratified_approximate_realizability(
        natural_exported,
        "scenario_family",
        required_values=(family for family, count in plan.items() if count > 0),
        neighbor_gate_minimum_distinct_roots=(
            BC0_FAMILY_NEIGHBOR_GATE_MINIMUM_ROOTS
        ),
    )
    approximate_by_stage = _stratified_approximate_realizability(
        natural_exported,
        "state_class",
        neighbor_gate_minimum_distinct_roots=None,
    )
    class_audit = audit_target_aware_state_classes(collected["episode_rows"])
    if not chat_audit["passed"]:
        raise RuntimeError(f"Chat-row audit failed: {chat_audit['errors'][:3]}")
    if not training_view_chat_audit["passed"]:
        raise RuntimeError(
            "Balanced training-view chat audit failed: "
            f"{training_view_chat_audit['errors'][:3]}"
        )
    if not realizability["passed"]:
        raise RuntimeError(
            f"Teacher realizability audit failed: conflict_rate={realizability['conflict_rate']}"
        )
    if not training_view_realizability["passed"]:
        raise RuntimeError(
            "Balanced training-view exact realizability failed: conflict_rate="
            f"{training_view_realizability['conflict_rate']}"
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
        "family_release_policy": BC0_FAMILY_RELEASE_POLICY,
        "split_rows": {name: len(rows) for name, rows in exported.items()},
        "natural_split_rows": {name: len(rows) for name, rows in splits.items()},
        "training_view": training_view_report,
        "state_class_distribution": _distribution(eligible_rows, "state_class"),
        "auxiliary_state_class_distribution": _distribution(
            auxiliary_rows, "state_class"
        ),
        "family_state_class_matrix": _family_class_matrix(
            eligible_rows, family_by_root
        ),
        "native_chat_audit": chat_audit,
        "training_view_chat_audit": training_view_chat_audit,
        "teacher_realizability": realizability,
        "training_view_teacher_realizability": training_view_realizability,
        "approximate_teacher_realizability": approximate_realizability,
        "training_view_approximate_teacher_realizability": (
            training_view_approximate_realizability
        ),
        "approximate_teacher_realizability_by_family": approximate_by_family,
        "approximate_teacher_realizability_by_state_class": approximate_by_stage,
        "physical_split_audit": physical_split_audit,
        "stratified_split_coverage": split_coverage,
        "target_aware_class_audit": class_audit,
        "holdout_disjointness": holdout_disjointness,
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
    release_failures.extend(_input_corpus_release_failures(generation_descriptor))
    release_failures.extend(_input_artifact_release_failures(generation_descriptor))
    release_failures.extend(_holdout_release_failures(holdout_disjointness))
    if generation_descriptor["source_state"].get("release_eligible_source") is not True:
        release_failures.append("source worktree was not clean at generation time")
    if args.protocol != "canonical":
        release_failures.append("model-visible protocol was not canonical")
    if not split_coverage["passed"]:
        release_failures.append(
            "stratified split coverage failed: "
            + json.dumps(
                split_coverage.get("critical_family_shortfalls")
                or split_coverage.get("coverage_deficits")
                or split_coverage,
                sort_keys=True,
            )
        )
    if training_view_report.get("release_ready") is not True:
        release_failures.append(
            "balanced training-view target gate failed: category_support_shortfalls="
            f"{training_view_report.get('tool_category_natural_support_shortfalls')}, "
            "exact_action_root_shortfalls="
            f"{training_view_report.get('target_tool_unique_root_shortfalls')}, "
            "critical_joint_root_shortfalls="
            f"{training_view_report.get('critical_joint_unique_root_shortfalls')}, "
            "feasibility_shortfall="
            f"{training_view_report.get('necessary_feasibility_shortfall_total')}, "
            "tool-category deviation="
            f"{training_view_report.get('achieved_tool_category_target_deviation')}"
        )
    if not approximate_realizability["passed"]:
        release_failures.append(
            "release-grade approximate teacher realizability failed: "
            f"nearest comparisons={approximate_realizability['nearest_neighbor_compared_examples']}, "
            f"local disagreement={approximate_realizability['local_perturbation_action_disagreement_rate']}, "
            "multi-action margin coverage="
            f"{approximate_realizability['multi_action_cost_margin_coverage']}"
        )
    if not training_view_approximate_realizability["passed"]:
        release_failures.append(
            "balanced training-view approximate teacher realizability failed: "
            "nearest comparisons="
            f"{training_view_approximate_realizability['nearest_neighbor_compared_examples']}, "
            "local disagreement="
            f"{training_view_approximate_realizability['local_perturbation_action_disagreement_rate']}"
        )
    release_failures.extend(
        _stratified_realizability_release_failures(
            approximate_by_family, dimension="scenario_family"
        )
    )
    release_failures.extend(
        _stratified_realizability_release_failures(
            approximate_by_stage, dimension="state_class"
        )
    )
    release_failures.extend(
        _collection_release_failures(
            plan=plan,
            scenarios=scenarios,
            nonterminal_episodes=nonterminal_episodes,
            quarantined_episodes=quarantined_episodes,
            unknown_terminal_outcome_episodes=unknown_terminal_outcome_episodes,
        )
    )
    release_failures.extend(
        _family_resolution_release_failures(
            terminal_scenario_matrix,
            plan=plan,
        )
    )
    if set(exported) != {"train_view", "validation", "test"}:
        release_failures.append(
            "one or more required train-view/validation/test splits were empty"
        )
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
        "holdout_disjointness": holdout_disjointness,
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
        default=Path(__file__).resolve().parents[2]
        / "data"
        / "round0_aggregate_release",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--scale", type=int, default=1, help="Multiply the default plan.")
    parser.add_argument("--plan", type=str, default=None, help="JSON plan or path to one.")
    parser.add_argument("--protocol", choices=("controller", "canonical"), default="canonical")
    parser.add_argument("--max-steps", type=int, default=24)
    parser.add_argument("--counterfactuals-per-scenario", type=int, default=3)
    parser.add_argument(
        "--evaluation-suite",
        type=Path,
        default=DEFAULT_EVALUATION_SUITE_PATH,
        help=(
            "Frozen schema-v1 evaluation suite reserved as a physical-root "
            "and scenario-ID holdout."
        ),
    )
    parser.add_argument(
        "--evaluation-policy",
        type=Path,
        default=DEFAULT_EVALUATION_POLICY_PATH,
        help=(
            "Pinned BC0 evaluation policy whose approved suite SHA-256 must "
            "match --evaluation-suite."
        ),
    )
    parser.add_argument(
        "--chi2-alpha",
        type=float,
        default=scenario_generator_module.DEFAULT_CHI2_ALPHA,
    )
    parser.add_argument("--hif-alpha-grid", type=int, default=5)
    parser.add_argument("--hif-r-grid", type=int, default=7)
    parser.add_argument("--hif-max-scans", type=int, default=3)
    parser.add_argument(
        "--measurement-corpus",
        type=Path,
        default=scenario_generator_module.DEFAULT_CORPUS_PATH,
        help="Measurement/parameter/harmonic JSONL corpus.",
    )
    parser.add_argument(
        "--hif-corpus",
        type=Path,
        action="append",
        default=None,
        help=(
            "HIF JSONL corpus; repeat for multiple files. The release default "
            "is the independently generated multiscan training corpus."
        ),
    )
    parser.add_argument(
        "--imbalance-corpus",
        type=Path,
        default=scenario_generator_module.DEFAULT_IMBALANCE_SAMPLE_PATH,
        help="Three-phase-unbalance JSONL corpus.",
    )
    args = parser.parse_args()
    report = generate(args)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "release_eligible": report["generation_provenance"].get(
                    "release_eligible"
                ),
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
    if report["generation_provenance"].get("release_eligible") is not True:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
