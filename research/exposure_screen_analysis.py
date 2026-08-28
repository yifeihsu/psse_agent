"""Analyze the frozen occupancy/exposure screen from physical-audit reports.

The analysis is deliberately CPU-only.  It consumes the replay-complete
``physical_audit.full.json`` files, proves that every policy is aligned on the
same 61 faulted and four no-error physical roots, and only then computes the
screening metrics and paired exact McNemar comparisons.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


EXPECTED_EPISODES = 65
EXPECTED_FAULTED = 61
EXPECTED_NO_ERROR = 4
EXPECTED_SCENARIOS_SHA256 = (
    "16fff49570a77b66001adcd95a39277e32395fec868d2fd8f618e61d863530af"
)
CURRENT_REPORTS = frozenset({"A", "B", "C", "D"})
HISTORICAL_REPORTS = frozenset(
    {
        "historical_e2b_selective",
        "historical_e2b_full",
        "historical_12b_selective",
        "historical_12b_full",
    }
)
HEX64 = re.compile(r"[0-9a-f]{64}")

EXACT_PHYSICAL_RECOVERY = "exact_physical_recovery"
PARTIAL_RECOVERY = "partial_recovery"
FALSE_INTERVENTION = "false_intervention"
NO_PHYSICAL_PROGRESS = "no_physical_progress"
GENERATION_ABORT = "not_assessable_generation_abort"
LOOP_BEFORE_STABLE_FINAL_STATE = "loop_before_stable_final_state"
OUTCOME_CLASSES = frozenset(
    {
        EXACT_PHYSICAL_RECOVERY,
        PARTIAL_RECOVERY,
        FALSE_INTERVENTION,
        NO_PHYSICAL_PROGRESS,
        GENERATION_ABORT,
        LOOP_BEFORE_STABLE_FINAL_STATE,
    }
)
FINAL_ACTIVE_STATE_CLASSES = frozenset(
    {
        EXACT_PHYSICAL_RECOVERY,
        PARTIAL_RECOVERY,
        FALSE_INTERVENTION,
        NO_PHYSICAL_PROGRESS,
    }
)

PAIR_SPECS = (
    ("C", "historical_e2b_selective"),
    ("C", "A"),
    ("C", "historical_12b_selective"),
    ("A", "historical_e2b_selective"),
    ("B", "historical_12b_selective"),
    ("B", "historical_12b_full"),
    ("D", "historical_12b_selective"),
)

REPORT_CONTRACTS = {
    "A": {
        "evaluation_key": "occupancy-cell-A",
        "report_sha256": "a93bd79aeb265edd8a7a487216b13b5a22ede93f38820d8ac01fab2cd67390fe",
        "source_evaluation_sha256": "475917fb219a89cb82ad0279e838833db80526a91f7a2415f93a3a536981062d",
        "source_scenario_binding": "sha256_and_per_episode_identity",
    },
    "B": {
        "evaluation_key": "occupancy-cell-B",
        "report_sha256": "b250d031731be7906cf0df7b7969a38ad252f4e36c2a53c09d400b3c6cbe4bd5",
        "source_evaluation_sha256": "73a49e0056c3ee7eb6d8f36908e617d3db0597977d865608fcfa4f3c56b7feae",
        "source_scenario_binding": "sha256_and_per_episode_identity",
    },
    "C": {
        "evaluation_key": "occupancy-cell-C",
        "report_sha256": "8c44ab5ade8e8e679f783473b385c5aa8146e8a2693768488278d746d2404536",
        "source_evaluation_sha256": "04755b67434064346eb1bc22b20bc3378b910f252591f9d758ea11cc32f83815",
        "source_scenario_binding": "sha256_and_per_episode_identity",
    },
    "D": {
        "evaluation_key": "occupancy-cell-D",
        "report_sha256": "38cf901dfa6e1024b000ea10335c66b8c28d4ed7186642ef0afb5bc6b533c06a",
        "source_evaluation_sha256": "330e1c97bf7dfb9965e65abcb18b28a8ffaaea285ddd6887cbfbe46299d1f774",
        "source_scenario_binding": "sha256_and_per_episode_identity",
    },
    "historical_e2b_selective": {
        "evaluation_key": "e2b_selective",
        "report_sha256": "ea89434d24bdf4d9df36b71889d0c9d5e25bf301b805bfd58688b2680aae5ea8",
        "source_evaluation_sha256": "d34bf62f1aee3b49d5c2f71331d1547312ba0f87882161564e00415d918be992",
        "source_scenario_binding": "ordered_replay_without_source_suite_hash",
    },
    "historical_e2b_full": {
        "evaluation_key": "e2b_full_occupancy",
        "report_sha256": "ea89434d24bdf4d9df36b71889d0c9d5e25bf301b805bfd58688b2680aae5ea8",
        "source_evaluation_sha256": "9ca26350f29a9fefc1b087dafdec2aac11b3d08a8e030ce67e303fa7e2da28c7",
        "source_scenario_binding": "ordered_replay_without_source_suite_hash",
    },
    "historical_12b_selective": {
        "evaluation_key": "12b_selective",
        "report_sha256": "ea89434d24bdf4d9df36b71889d0c9d5e25bf301b805bfd58688b2680aae5ea8",
        "source_evaluation_sha256": "94ac8b35089025e84a2ea5f329d6897720f983bc33035eb0f797ab7ae2b59395",
        "source_scenario_binding": "ordered_replay_without_source_suite_hash",
    },
    "historical_12b_full": {
        "evaluation_key": "12b_full_occupancy",
        "report_sha256": "ea89434d24bdf4d9df36b71889d0c9d5e25bf301b805bfd58688b2680aae5ea8",
        "source_evaluation_sha256": "594f54567bde7a7bb308d520a0766d3c02b48222f1fb33adb5f9798a8efcbbb9",
        "source_scenario_binding": "ordered_replay_without_source_suite_hash",
    },
}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _root_id(episode: Mapping[str, Any]) -> str:
    value = episode.get("root_scenario_id") or episode.get("scenario_id")
    if not isinstance(value, str) or not value:
        raise ValueError("physical-audit episode lacks root scenario identity")
    return value


def _nonnegative_int(row: Mapping[str, Any], key: str) -> int:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"physical-audit episode has invalid {key}: {value!r}")
    return value


def _faulted(episode: Mapping[str, Any]) -> bool:
    return _nonnegative_int(episode, "initial_true_error_count") > 0


def _exact(episode: Mapping[str, Any]) -> bool:
    return _faulted(episode) and episode.get("final_active_state_class") == (
        EXACT_PHYSICAL_RECOVERY
    )


def _stable_exact(episode: Mapping[str, Any]) -> bool:
    return (
        _exact(episode)
        and not bool(episode.get("generation_abort"))
        and not bool(episode.get("loop_before_stable_final_state"))
    )


def _final_false_intervention(episode: Mapping[str, Any]) -> bool:
    return _faulted(episode) and episode.get("final_active_state_class") == (
        FALSE_INTERVENTION
    )


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _same_number(first: Any, second: Any) -> bool:
    if first is None or second is None:
        return first is None and second is None
    if (
        isinstance(first, bool)
        or isinstance(second, bool)
        or not isinstance(first, (int, float))
        or not isinstance(second, (int, float))
    ):
        return False
    return math.isfinite(float(first)) and math.isfinite(float(second)) and math.isclose(
        float(first), float(second), rel_tol=0.0, abs_tol=1e-15
    )


def _recompute_metrics(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    assessable = [row for row in episodes if row.get("physical_assessable") is True]
    faulted = [row for row in episodes if _faulted(row)]
    no_error = [row for row in assessable if not _faulted(row)]
    committed_events = sum(
        _nonnegative_int(row, "committed_correction_target_event_count")
        for row in assessable
    )
    unique_committed = sum(
        _nonnegative_int(row, "unique_committed_correction_target_count")
        for row in assessable
    )
    true_committed = sum(
        _nonnegative_int(row, "true_committed_correction_count")
        for row in assessable
    )
    true_errors = sum(
        _nonnegative_int(row, "initial_true_error_count") for row in assessable
    )
    corrected = sum(
        _nonnegative_int(row, "true_errors_corrected") for row in assessable
    )
    excess = [
        _nonnegative_int(row, "excess_steps_relative_to_expert")
        for row in episodes
    ]
    six_way = Counter(str(row.get("episode_outcome_class")) for row in episodes)
    final_active = Counter(str(row.get("final_active_state_class")) for row in episodes)
    by_family: dict[str, dict[str, int]] = {}
    for row in episodes:
        family = row.get("scenario_family")
        if not isinstance(family, str) or not family:
            raise ValueError("physical-audit episode lacks scenario_family")
        record = by_family.setdefault(
            family,
            {"episodes": 0, "faulted": 0, "final_exact": 0, "stable_exact": 0},
        )
        record["episodes"] += 1
        if _faulted(row):
            record["faulted"] += 1
            record["final_exact"] += int(_exact(row))
            record["stable_exact"] += int(_stable_exact(row))
    return {
        "faulted_episodes": len(faulted),
        "no_error_episodes": len(no_error),
        "final_exact_recovery": sum(_exact(row) for row in episodes),
        "stable_exact_recovery": sum(_stable_exact(row) for row in episodes),
        "event_level_correction_precision": _ratio(true_committed, committed_events),
        "event_level_correction_precision_numerator_true_committed_targets": (
            true_committed
        ),
        "event_level_correction_precision_denominator_all_committed_target_events": (
            committed_events
        ),
        "unique_target_precision": _ratio(true_committed, unique_committed),
        "unique_target_precision_numerator_true_committed_targets": true_committed,
        "unique_target_precision_denominator_unique_committed_targets": (
            unique_committed
        ),
        "correction_recall": _ratio(corrected, true_errors),
        "correction_recall_numerator_true_errors_corrected": corrected,
        "correction_recall_denominator_true_errors_assessable": true_errors,
        "faulted_final_active_false_interventions": sum(
            _final_false_intervention(row) for row in episodes
        ),
        "no_error_false_interventions": sum(
            row.get("final_active_state_class") == FALSE_INTERVENTION
            for row in no_error
        ),
        "no_error_clean_preservation": sum(
            row.get("final_active_state_class") == EXACT_PHYSICAL_RECOVERY
            for row in no_error
        ),
        "six_way_outcome_counts": dict(sorted(six_way.items())),
        "final_active_state_counts": dict(sorted(final_active.items())),
        "partial_recovery": sum(
            row.get("final_active_state_class") == PARTIAL_RECOVERY
            for row in faulted
        ),
        "generation_aborts": sum(bool(row.get("generation_abort")) for row in faulted),
        "loops": sum(
            bool(row.get("loop_before_stable_final_state")) for row in faulted
        ),
        "family_exact_recovery": {
            key: {
                "exact": value["final_exact"],
                "stable_exact": value["stable_exact"],
                "faulted": value["faulted"],
                "episodes": value["episodes"],
            }
            for key, value in sorted(by_family.items())
            if value["faulted"] > 0
        },
        "family_all": dict(sorted(by_family.items())),
        "mean_excess_steps": sum(excess) / len(excess) if excess else None,
        "mean_excess_steps_episode_count": len(excess),
    }


def parse_report_spec(value: str) -> tuple[str, Path, str]:
    try:
        name, source = value.split("=", 1)
        path_text, evaluation = source.rsplit("::", 1)
    except ValueError as exc:
        raise ValueError("report must be NAME=PATH::EVALUATION_KEY") from exc
    if not name or not path_text or not evaluation:
        raise ValueError("report must contain non-empty name, path, and evaluation key")
    return name, Path(path_text), evaluation


def load_report(value: str) -> dict[str, Any]:
    name, path, evaluation_key = parse_report_spec(value)
    payload = json.loads(path.read_text(encoding="utf-8"))
    evaluations = payload.get("evaluations")
    if not isinstance(evaluations, Mapping) or evaluation_key not in evaluations:
        raise ValueError(f"{path} has no evaluation {evaluation_key!r}")
    evaluation = evaluations[evaluation_key]
    if not isinstance(evaluation, Mapping):
        raise ValueError(f"{path} evaluation {evaluation_key!r} is not an object")
    episodes = evaluation.get("per_episode")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError(f"{name} is not a replay-complete physical-audit report")
    roots = [_root_id(row) for row in episodes]
    return {
        "name": name,
        "path": str(path),
        "sha256": file_sha256(path),
        "evaluation_key": evaluation_key,
        "evaluation": evaluation,
        "episodes": episodes,
        "by_root": dict(zip(roots, episodes, strict=True)),
        "audit_schema_version": payload.get("audit_schema_version"),
        "evaluation_audit_schema_version": evaluation.get("audit_schema_version"),
        "scenarios_sha256": payload.get("scenarios_sha256"),
        "release_evidence": payload.get("release_evidence"),
        "evaluation_release_evidence": evaluation.get("release_evidence"),
        "source_evaluation": evaluation.get("source_evaluation"),
        "source_scenario_binding": evaluation.get("source_scenario_binding"),
        "source_scenarios_sha256": evaluation.get("source_scenarios_sha256"),
        "source_release_evidence": evaluation.get("source_release_evidence"),
    }


def quality_checks(reports: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    per_report: dict[str, dict[str, Any]] = {}
    reference_roots: set[str] | None = None
    reference_fingerprints: dict[str, str] | None = None
    reference_fault_mask: dict[str, bool] | None = None
    reference_scenarios_sha256: str | None = None
    cross_root_sets = True
    cross_fingerprints = True
    cross_fault_masks = True
    cross_scenario_hashes = True
    for name, report in reports.items():
        episodes = report["episodes"]
        roots = [_root_id(row) for row in episodes]
        faulted = [row for row in episodes if _faulted(row)]
        no_error = [row for row in episodes if not _faulted(row)]
        fingerprints = {
            _root_id(row): str(row.get("physical_root_fingerprint") or "")
            for row in episodes
        }
        fault_mask = {_root_id(row): _faulted(row) for row in episodes}
        summary = report["evaluation"].get("summary") or {}
        if not isinstance(summary, Mapping):
            raise ValueError(f"{name} physical-audit summary is not an object")
        recomputed = _recompute_metrics(episodes)
        source_evaluation = report.get("source_evaluation")
        source_evaluation = (
            source_evaluation if isinstance(source_evaluation, Mapping) else {}
        )
        contract = REPORT_CONTRACTS.get(name)
        if contract is None:
            raise ValueError(f"unknown report provenance class: {name}")
        expected_binding = contract["source_scenario_binding"]
        strong_binding = name in CURRENT_REPORTS
        summary_family = report["evaluation"].get("by_scenario_family")
        summary_family = summary_family if isinstance(summary_family, Mapping) else {}
        recomputed_family = recomputed["family_all"]
        family_consistent = set(summary_family) == set(recomputed_family) and all(
            isinstance(summary_family.get(family), Mapping)
            and summary_family[family].get("episodes") == values["episodes"]
            and summary_family[family].get("faulted_episode_count")
            == values["faulted"]
            and summary_family[family].get("exact_episode_recovery_count")
            == values["final_exact"]
            and summary_family[family].get(
                "stable_terminal_exact_episode_recovery_count"
            )
            == values["stable_exact"]
            for family, values in recomputed_family.items()
        )
        checks = {
            "evaluation_label_expected": report.get("evaluation_key")
            == contract["evaluation_key"],
            "audit_report_sha256_expected": report.get("sha256")
            == contract["report_sha256"],
            "audit_schema_v2": report.get("audit_schema_version") == 2
            and report.get("evaluation_audit_schema_version") == 2,
            "research_only": report.get("release_evidence") is False
            and report.get("evaluation_release_evidence") is False
            and report.get("source_release_evidence") is False,
            "scenario_sha256": report.get("scenarios_sha256")
            == EXPECTED_SCENARIOS_SHA256,
            "source_evaluation_path_present": isinstance(
                source_evaluation.get("path"), str
            )
            and bool(source_evaluation.get("path")),
            "source_evaluation_sha256_present": isinstance(
                source_evaluation.get("sha256"), str
            )
            and HEX64.fullmatch(str(source_evaluation.get("sha256"))) is not None,
            "source_evaluation_sha256_expected": source_evaluation.get("sha256")
            == contract["source_evaluation_sha256"],
            "source_binding_expected": report.get("source_scenario_binding")
            == expected_binding,
            "source_suite_binding_expected": (
                report.get("source_scenarios_sha256")
                == EXPECTED_SCENARIOS_SHA256
                if strong_binding
                else report.get("source_scenarios_sha256") is None
            ),
            "episode_count": len(episodes) == EXPECTED_EPISODES,
            "unique_root_ids": len(set(roots)) == len(roots),
            "faulted_count": len(faulted) == EXPECTED_FAULTED,
            "no_error_count": len(no_error) == EXPECTED_NO_ERROR,
            "physical_assessment_complete": all(
                row.get("physical_assessable") is True for row in episodes
            ),
            "replay_consistent": all(
                row.get("replay_matches_record") is True for row in episodes
            ),
            "classified": all(
                row.get("episode_outcome_class") in OUTCOME_CLASSES
                and row.get("final_active_state_class")
                in FINAL_ACTIVE_STATE_CLASSES
                for row in episodes
            ),
            "physical_fingerprints_present": all(fingerprints.values()),
            "summary_exact_consistent": summary.get("exact_episode_recovery_count")
            == recomputed["final_exact_recovery"],
            "summary_stable_exact_consistent": summary.get(
                "stable_terminal_exact_episode_recovery_count"
            )
            == recomputed["stable_exact_recovery"],
            "summary_event_precision_support_consistent": summary.get(
                "correction_precision_numerator_true_committed_targets"
            )
            == recomputed[
                "event_level_correction_precision_numerator_true_committed_targets"
            ]
            and summary.get(
                "correction_precision_denominator_all_committed_target_events"
            )
            == recomputed[
                "event_level_correction_precision_denominator_all_committed_target_events"
            ]
            and _same_number(
                summary.get("correction_precision"),
                recomputed["event_level_correction_precision"],
            ),
            "summary_unique_precision_support_consistent": summary.get(
                "unique_target_correction_precision_numerator_true_committed_targets"
            )
            == recomputed[
                "unique_target_precision_numerator_true_committed_targets"
            ]
            and summary.get(
                "unique_target_correction_precision_denominator_unique_committed_targets"
            )
            == recomputed[
                "unique_target_precision_denominator_unique_committed_targets"
            ]
            and _same_number(
                summary.get("unique_target_correction_precision"),
                recomputed["unique_target_precision"],
            ),
            "summary_recall_support_consistent": summary.get(
                "correction_recall_numerator_true_errors_corrected"
            )
            == recomputed["correction_recall_numerator_true_errors_corrected"]
            and summary.get("correction_recall_denominator_true_errors_assessable")
            == recomputed[
                "correction_recall_denominator_true_errors_assessable"
            ]
            and _same_number(
                summary.get("correction_recall"),
                recomputed["correction_recall"],
            ),
            "summary_six_way_consistent": summary.get("six_way_outcome_counts")
            == recomputed["six_way_outcome_counts"],
            "summary_family_consistent": family_consistent,
            "summary_mean_excess_consistent": summary.get(
                "excess_steps_relative_to_expert_episode_count"
            )
            == recomputed["mean_excess_steps_episode_count"]
            and _same_number(
                summary.get("mean_excess_steps_relative_to_expert"),
                recomputed["mean_excess_steps"],
            ),
        }
        per_report[name] = {
            "checks": checks,
            "provenance": {
                "class": (
                    "strong_hash_and_per_episode_identity"
                    if strong_binding
                    else "legacy_ordered_replay_only"
                ),
                "scenario_sha256": report.get("scenarios_sha256"),
                "source_scenario_binding": report.get("source_scenario_binding"),
                "source_scenarios_sha256": report.get("source_scenarios_sha256"),
                "source_evaluation": dict(source_evaluation),
            },
        }
        root_set = set(roots)
        if reference_roots is None:
            reference_roots = root_set
            reference_fingerprints = fingerprints
            reference_fault_mask = fault_mask
            reference_scenarios_sha256 = str(report.get("scenarios_sha256") or "")
        else:
            cross_root_sets &= root_set == reference_roots
            cross_fingerprints &= fingerprints == reference_fingerprints
            cross_fault_masks &= fault_mask == reference_fault_mask
            cross_scenario_hashes &= (
                report.get("scenarios_sha256") == reference_scenarios_sha256
            )
    all_local = all(
        all(record["checks"].values()) for record in per_report.values()
    )
    return {
        "passed": all_local
        and cross_root_sets
        and cross_fingerprints
        and cross_fault_masks
        and cross_scenario_hashes,
        "per_report": per_report,
        "cross_report_same_root_set": cross_root_sets,
        "cross_report_same_physical_fingerprints": cross_fingerprints,
        "cross_report_same_fault_mask": cross_fault_masks,
        "cross_report_same_scenario_sha256": cross_scenario_hashes,
        "scenario_sha256": EXPECTED_SCENARIOS_SHA256,
        "historical_provenance_limit": (
            "Historical source evaluations lack suite hashes and per-episode "
            "identities; their replay-complete audits are aligned by immutable "
            "ordered replay. Paired historical comparisons inherit this weaker "
            "lineage and must not be described as identity-bound source evaluations."
        ),
    }


def checkpoint_metrics(report: Mapping[str, Any]) -> dict[str, Any]:
    return _recompute_metrics(report["episodes"])


def exact_mcnemar_p(policy_1_only: int, policy_2_only: int) -> float:
    discordant = policy_1_only + policy_2_only
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index)
        for index in range(min(policy_1_only, policy_2_only) + 1)
    )
    return min(1.0, 2.0 * tail / (2**discordant))


def paired_table(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    predicate: Callable[[Mapping[str, Any]], bool],
) -> dict[str, Any]:
    first_rows = first["by_root"]
    second_rows = second["by_root"]
    faulted_roots = sorted(root for root, row in first_rows.items() if _faulted(row))
    if set(faulted_roots) != {
        root for root, row in second_rows.items() if _faulted(row)
    }:
        raise ValueError("paired reports do not share the faulted-root set")
    both = first_only = second_only = neither = 0
    for root in faulted_roots:
        a = predicate(first_rows[root])
        b = predicate(second_rows[root])
        if a and b:
            both += 1
        elif a:
            first_only += 1
        elif b:
            second_only += 1
        else:
            neither += 1
    return {
        "both_positive": both,
        "policy_1_only": first_only,
        "policy_2_only": second_only,
        "neither_positive": neither,
        "discordant": first_only + second_only,
        "exact_mcnemar_two_sided_p": exact_mcnemar_p(first_only, second_only),
    }


def pairwise_comparisons(reports: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    predicates = {
        "final_exact_recovery": _exact,
        "stable_exact_recovery": _stable_exact,
        "final_false_intervention": _final_false_intervention,
    }
    output: dict[str, Any] = {}
    for first, second in PAIR_SPECS:
        if first not in reports or second not in reports:
            raise ValueError(f"missing report required for {first} versus {second}")
        output[f"{first}_vs_{second}"] = {
            "policy_1": first,
            "policy_2": second,
            "metrics": {
                name: paired_table(reports[first], reports[second], predicate)
                for name, predicate in predicates.items()
            },
        }
    return output


def gain_decomposition(
    current: Mapping[str, Any], historical: Mapping[str, Any]
) -> dict[str, Any]:
    current_rows = current["by_root"]
    historical_rows = historical["by_root"]
    roots = sorted(root for root, row in current_rows.items() if _faulted(row))
    rescued = [
        root
        for root in roots
        if _exact(current_rows[root]) and not _exact(historical_rows[root])
    ]
    regressed = [
        root
        for root in roots
        if _exact(historical_rows[root]) and not _exact(current_rows[root])
    ]
    rescued_families = Counter(
        str(current_rows[root].get("scenario_family")) for root in rescued
    )
    committed_deltas = [
        int(current_rows[root].get("committed_correction_target_event_count") or 0)
        - int(historical_rows[root].get("committed_correction_target_event_count") or 0)
        for root in rescued
    ]
    return {
        "historical_exact": sum(_exact(row) for row in historical_rows.values()),
        "current_exact": sum(_exact(row) for row in current_rows.values()),
        "interpretation": (
            "Descriptive, non-additive episode transitions; categories can overlap "
            "and do not identify a causal mechanism."
        ),
        "net_exact_gain": len(rescued) - len(regressed),
        "rescued_roots": rescued,
        "regressed_roots": regressed,
        "rescued_by_family": dict(sorted(rescued_families.items())),
        "rescued_multi_measurement": sum(
            current_rows[root].get("scenario_family") == "multi-measurement"
            for root in rescued
        ),
        "rescued_topology_containing": sum(
            "topology" in str(current_rows[root].get("scenario_family"))
            for root in rescued
        ),
        "rescued_from_generation_abort": sum(
            bool(historical_rows[root].get("generation_abort")) for root in rescued
        ),
        "rescued_from_loop": sum(
            bool(historical_rows[root].get("loop_before_stable_final_state"))
            for root in rescued
        ),
        "rescued_from_final_false_intervention": sum(
            _final_false_intervention(historical_rows[root]) for root in rescued
        ),
        "rescued_with_more_committed_target_events": sum(
            delta > 0 for delta in committed_deltas
        ),
        "rescued_committed_target_event_delta": sum(committed_deltas),
    }


def selection_gate(metrics: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    arms = {name: metrics[name] for name in ("A", "B", "C", "D")}
    rows = {
        name: {
            "preserves_all_four_no_error": value["no_error_clean_preservation"] == 4
            and value["no_error_false_interventions"] == 0,
            "zero_faulted_false_interventions": value[
                "faulted_final_active_false_interventions"
            ]
            == 0,
            "stable_exact_recovery": value["stable_exact_recovery"],
        }
        for name, value in arms.items()
    }
    eligible = [
        name
        for name, row in rows.items()
        if row["preserves_all_four_no_error"]
        and row["zero_faulted_false_interventions"]
    ]
    leader = max(eligible, key=lambda name: rows[name]["stable_exact_recovery"], default=None)
    return {
        "per_arm": rows,
        "strictly_eligible_arms": eligible,
        "provisional_leader": leader,
        "rule": "preserve no-error; require zero faulted final-state false interventions; maximize stable exact",
    }


def analyze(report_specs: Sequence[str]) -> dict[str, Any]:
    loaded = [load_report(value) for value in report_specs]
    if len({item["name"] for item in loaded}) != len(loaded):
        raise ValueError("report names must be unique")
    reports = {item["name"]: item for item in loaded}
    required = set(REPORT_CONTRACTS)
    missing = sorted(required - reports.keys())
    if missing:
        raise ValueError(f"missing required reports: {', '.join(missing)}")
    quality = quality_checks(reports)
    if not quality["passed"]:
        raise ValueError("physical-audit data-quality gate failed")
    metrics = {name: checkpoint_metrics(report) for name, report in reports.items()}
    return {
        "schema": "research_exposure_screen_analysis_v1",
        "release_evidence": False,
        "quality": quality,
        "inputs": {
            name: {
                key: report[key]
                for key in (
                    "path",
                    "sha256",
                    "evaluation_key",
                    "scenarios_sha256",
                    "source_evaluation",
                    "source_scenario_binding",
                    "source_scenarios_sha256",
                )
            }
            for name, report in reports.items()
        },
        "checkpoint_metrics": metrics,
        "selection_gate": selection_gate(metrics),
        "pairwise_comparisons": pairwise_comparisons(reports),
        "pairwise_inference_note": (
            "All exact McNemar p-values are two-sided, unadjusted, and exploratory; "
            "21 correlated tests are reported without multiplicity correction."
        ),
        "c_vs_historical_e2b_full_gain": gain_decomposition(
            reports["C"], reports["historical_e2b_full"]
        ),
    }


def _fmt_rate(value: Any) -> str:
    return "n/a" if value is None else f"{100 * float(value):.1f}%"


def _fmt_supported_rate(value: Any, numerator: int, denominator: int) -> str:
    return f"{numerator}/{denominator} ({_fmt_rate(value)})"


def render_markdown(result: Mapping[str, Any]) -> str:
    quality = result["quality"]
    lines = [
        "# Frozen A-D occupancy/exposure screen",
        "",
        "Data-quality gate: **passed** (65 aligned physical roots: 61 faulted + 4 no-error).",
        f"Scenario SHA-256: `{quality['scenario_sha256']}`.",
        "",
        f"Provenance limitation: {quality['historical_provenance_limit']}",
        "",
        "| Policy | Exact | Stable exact | Event precision | Unique precision | Recall | Faulted FI | Partial | Aborts | Loops | Mean excess steps |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    metrics = result["checkpoint_metrics"]
    for name in ("A", "B", "C", "D"):
        row = metrics[name]
        lines.append(
            f"| {name} | {row['final_exact_recovery']}/61 | "
            f"{row['stable_exact_recovery']}/61 | "
            f"{_fmt_supported_rate(row['event_level_correction_precision'], row['event_level_correction_precision_numerator_true_committed_targets'], row['event_level_correction_precision_denominator_all_committed_target_events'])} | "
            f"{_fmt_supported_rate(row['unique_target_precision'], row['unique_target_precision_numerator_true_committed_targets'], row['unique_target_precision_denominator_unique_committed_targets'])} | "
            f"{_fmt_supported_rate(row['correction_recall'], row['correction_recall_numerator_true_errors_corrected'], row['correction_recall_denominator_true_errors_assessable'])} | "
            f"{row['faulted_final_active_false_interventions']} | "
            f"{row['partial_recovery']} | {row['generation_aborts']} | "
            f"{row['loops']} | {float(row['mean_excess_steps']):.2f} |"
        )
    lines.extend(
        [
            "",
            "## Input provenance",
            "",
            "| Policy | Audit SHA-256 | Source evaluation SHA-256 | Binding |",
            "| --- | --- | --- | --- |",
        ]
    )
    report_order = (
        "A",
        "B",
        "C",
        "D",
        "historical_e2b_selective",
        "historical_e2b_full",
        "historical_12b_selective",
        "historical_12b_full",
    )
    for name in report_order:
        source = result["inputs"][name]["source_evaluation"]
        binding = result["inputs"][name]["source_scenario_binding"]
        lines.append(
            f"| {name} | `{result['inputs'][name]['sha256']}` | "
            f"`{source['sha256']}` | `{binding}` |"
        )
    lines.extend(
        [
            "",
            "## No-error preservation",
            "",
            "| Policy | Clean preserved | No-error false interventions |",
            "| --- | ---: | ---: |",
        ]
    )
    for name in ("A", "B", "C", "D"):
        row = metrics[name]
        lines.append(
            f"| {name} | {row['no_error_clean_preservation']}/4 | "
            f"{row['no_error_false_interventions']} |"
        )
    lines.extend(
        [
            "",
            "## Six-way trajectory outcomes (all 65 episodes)",
            "",
            "| Policy | Exact | Partial | False intervention | No progress | Abort | Loop |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name in ("A", "B", "C", "D"):
        counts = metrics[name]["six_way_outcome_counts"]
        lines.append(
            f"| {name} | {counts.get(EXACT_PHYSICAL_RECOVERY, 0)} | "
            f"{counts.get(PARTIAL_RECOVERY, 0)} | "
            f"{counts.get(FALSE_INTERVENTION, 0)} | "
            f"{counts.get(NO_PHYSICAL_PROGRESS, 0)} | "
            f"{counts.get(GENERATION_ABORT, 0)} | "
            f"{counts.get(LOOP_BEFORE_STABLE_FINAL_STATE, 0)} |"
        )
    lines.extend(
        [
            "",
            "## Family-level final exact recovery (faulted episodes)",
            "",
            "| Policy | Family | Exact | Stable exact |",
            "| --- | --- | ---: | ---: |",
        ]
    )
    for name in ("A", "B", "C", "D"):
        for family, family_row in metrics[name]["family_exact_recovery"].items():
            lines.append(
                f"| {name} | {family} | {family_row['exact']}/{family_row['faulted']} | "
                f"{family_row['stable_exact']}/{family_row['faulted']} |"
            )
    lines.extend(["", "## Paired exact McNemar comparisons", ""])
    lines.append(result["pairwise_inference_note"])
    lines.append("")
    lines.extend(
        [
            "| Pair | Metric | Both | Policy 1 only | Policy 2 only | Neither | Exact p (unadjusted) |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for pair_name, pair in result["pairwise_comparisons"].items():
        for metric_name, table in pair["metrics"].items():
            lines.append(
                f"| {pair_name} | {metric_name} | {table['both_positive']} | "
                f"{table['policy_1_only']} | {table['policy_2_only']} | "
                f"{table['neither_positive']} | "
                f"{table['exact_mcnemar_two_sided_p']:.4f} |"
            )
    gain = result["c_vs_historical_e2b_full_gain"]
    lines.extend(
        [
            "",
            "## C versus historical low-exposure E2B full occupancy",
            "",
            gain["interpretation"],
            "",
            f"- Exact recovery: {gain['historical_exact']}/61 -> {gain['current_exact']}/61.",
            f"- Rescued roots: {len(gain['rescued_roots'])}; regressed roots: {len(gain['regressed_roots'])}; net gain: {gain['net_exact_gain']}.",
            f"- Rescued by family: `{json.dumps(gain['rescued_by_family'], sort_keys=True)}`.",
            f"- Rescued from abort: {gain['rescued_from_generation_abort']}; loop: {gain['rescued_from_loop']}; false intervention: {gain['rescued_from_final_false_intervention']}.",
            f"- Rescued with more committed target events: {gain['rescued_with_more_committed_target_events']}; summed event delta: {gain['rescued_committed_target_event_delta']}.",
            "",
            "## Selection gate",
            "",
            f"Strictly eligible A-D arms: `{result['selection_gate']['strictly_eligible_arms']}`.",
            f"Provisional leader under the stated hard rule: `{result['selection_gate']['provisional_leader']}`.",
            "",
            "This is research evidence, not release evidence.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        action="append",
        required=True,
        help="NAME=PATH::EVALUATION_KEY; repeat for every checkpoint",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown-output")
    args = parser.parse_args(argv)
    result = analyze(args.report)
    rendered = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    Path(args.output).write_text(rendered, encoding="utf-8")
    if args.markdown_output:
        Path(args.markdown_output).write_text(render_markdown(result), encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
