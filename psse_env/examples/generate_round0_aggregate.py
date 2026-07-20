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
import psse_env.dagger.replay_buffer as replay_buffer_module
import psse_env.dagger.release_audit as release_audit_module
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
from psse_env.dagger.replay_buffer import build_balanced_training_view
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
    "multi_measurement": 20,
    "parameter": 6,
    "topology": 6,
    "harmonic": 4,
    # The clean checkout carries 17 independent tracked HIF roots.  HIF has a
    # separately reported handoff allowance until additional localization
    # roots are checked in.
    "hif": 17,
    "measurement+parameter": 20,
    "measurement+topology": 20,
    "measurement+hif": 2,
    "three_phase_unbalance": 20,
    "telemetry_no_disturbance": 20,
}

# Every positive-count release family has an explicit outcome contract.  The
# directly recoverable and deterministic diagnostic families must resolve all
# of their small default suites; the larger mixed/unbalance suites permit at
# most one safe handoff in twenty.  HIF-bearing families keep an explicit
# handoff allowance until a sufficiently observable localization corpus is
# available, and that allowance is reported rather than counted as recovery.
BC0_FAMILY_RELEASE_POLICY: dict[str, dict[str, float | int]] = {
    "no_error": {
        "minimum_physical_roots": 4,
        "minimum_resolution_rate": 1.0,
        "maximum_operator_escalation_rate": 0.0,
    },
    "measurement": {
        "minimum_physical_roots": 6,
        "minimum_resolution_rate": 1.0,
        "maximum_operator_escalation_rate": 0.0,
    },
    "multi_measurement": {
        "minimum_physical_roots": 20,
        "minimum_resolution_rate": 0.95,
        "maximum_operator_escalation_rate": 0.05,
    },
    "parameter": {
        "minimum_physical_roots": 6,
        "minimum_resolution_rate": 1.0,
        "maximum_operator_escalation_rate": 0.0,
    },
    "topology": {
        "minimum_physical_roots": 6,
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
        "minimum_physical_roots": 20,
        "minimum_resolution_rate": 0.95,
        "maximum_operator_escalation_rate": 0.05,
    },
    "measurement+topology": {
        "minimum_physical_roots": 20,
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
    "three_phase_unbalance": {
        "minimum_physical_roots": 20,
        "minimum_resolution_rate": 0.95,
        "maximum_operator_escalation_rate": 0.05,
    },
    "telemetry_no_disturbance": {
        "minimum_physical_roots": 20,
        "minimum_resolution_rate": 1.0,
        "maximum_operator_escalation_rate": 0.0,
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
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        value = _row_metadata_value(row, field)
        grouped.setdefault(str(value if value is not None else "unknown"), []).append(row)
    for value in required_values:
        grouped.setdefault(str(value), [])
    return {
        value: audit_approximate_teacher_realizability(
            group,
            **APPROXIMATE_REALIZABILITY_RELEASE_KWARGS,
        )
        for value, group in sorted(grouped.items())
    }


def _stratified_realizability_release_failures(
    reports: Mapping[str, Mapping[str, Any]], *, dimension: str
) -> list[str]:
    failures: list[str] = []
    for stratum, report in sorted(reports.items()):
        if report.get("passed") is True:
            continue
        failures.append(
            "approximate teacher realizability failed for "
            f"{dimension}={stratum}: examples={report.get('labeled_examples', 0)}, "
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
) -> dict[str, Any]:
    """Bind a generated aggregate to its source, schemas, and collection plan."""
    repo_root = Path(__file__).resolve().parents[2]
    source_files = (
        Path(__file__),
        Path(dataset_builder_module.__file__),
        Path(protocol_bridge_module.__file__),
        Path(rollout_collector_module.__file__),
        Path(replay_buffer_module.__file__),
        Path(release_audit_module.__file__),
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
            "family_release_policy": BC0_FAMILY_RELEASE_POLICY,
            "critical_split_minimums": {"validation": 5, "test": 5},
            "training_view": {
                "size_policy": "natural_train_row_count_with_bounded_replacement",
                "strict_target_axes": ["tool_category"],
                "capacity_aware_target_axes": [
                    "state_class",
                    "target_tool",
                    "scenario_family",
                    "error_cardinality",
                    "terminal_outcome",
                ],
                "capacity_aware_policy": "uniform_then_clip_and_redistribute_v1",
                "max_duplicate_count": 2,
                "low_cost_margin_threshold": 0.05,
                "maximum_tool_category_target_deviation": 0.10,
            },
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
    scenario_by_root: dict[str, dict[str, Any]] = {}
    for scenario in scenarios:
        fingerprint = physical_root_fingerprint(scenario)
        scenario["physical_root_fingerprint"] = fingerprint
        root_id = str(scenario["root_scenario_id"])
        physical_by_root[root_id] = fingerprint
        scenario_by_root[root_id] = scenario

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
        size=len(splits["train"]),
        seed=args.seed,
        max_duplicate_count=2,
        low_cost_margin_threshold=0.05,
        maximum_tool_category_target_deviation=0.10,
    )
    exported = {
        "train_view": examples_to_chat_sft(
            train_view_rows, protocol=args.protocol
        ),
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
    )
    approximate_by_stage = _stratified_approximate_realizability(
        natural_exported, "state_class"
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
            "balanced training-view target gate failed: feasibility_shortfall="
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
