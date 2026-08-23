"""Build the preregistered, policy-visible DAgger recovery-stress suite.

The suite is derived from the protected 30-root DAgger development pool, but
uses deterministic states that are not present in the ordinary development
rollout.  Every stress cell contains ten independent physical roots, and the
entire parent pool is already disjoint from D0, D1 collection, the frozen
evaluation suite, and the observable recovery-probe training roots.

No target label is stored in the policy-visible scenario.  The builder runs
the real production environment and requires the canonical observable expert
to independently classify and select the expected recovery action from the
first post-intervention PolicyObservation.
"""

from __future__ import annotations

import argparse
import copy
import inspect
import json
import os
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from psse_env.actions import (
    ASK_FOR_MORE_EVIDENCE,
    COMMIT_STATE,
    CORRECT_MEASUREMENTS,
    GET_MEASUREMENT_CONTEXT,
    RECOVERY_OPTIONS_EXHAUSTED_REQUEST,
    RUN_WLS,
)
from psse_env.dagger.evaluator import (
    RECOVERY_STRESS_SUITES,
    RECOVERY_STRESS_SUITE_TO_STRATUM,
    evaluate_rollout_suites,
    fingerprint_evaluation_suites,
    validate_release_scenario_suites,
)
from psse_env.dagger.release_factories import (
    EXPERT_POLICY_IDENTITY,
    deterministic_case_loader,
    observable_expert_policy_factory,
    production_environment_factory,
)
from psse_env.dagger.root_sets import (
    physical_roots_from_artifact,
    root_set_digest,
)
from psse_env.dagger.suite_builder import _partial_setup_actions
from psse_env.sft.provenance import (
    file_sha256,
    git_source_state,
    stable_json_sha256,
    validate_aggregate_manifest_binding,
)


RECOVERY_STRESS_CONTRACT = "dagger1_root_disjoint_recovery_stress_v1"
RECOVERY_STRESS_ARTIFACT_TYPE = "dagger1_recovery_stress_suite"
RECOVERY_STRESS_SPLIT = "dagger_recovery_stress"
RECOVERY_STRESS_ROOTS_PER_STRATUM = 10
RECOVERY_STRESS_DISTINCT_ROOT_COUNT = 20
RECOVERY_STRESS_EPISODE_COUNT = (
    len(RECOVERY_STRESS_SUITES) * RECOVERY_STRESS_ROOTS_PER_STRATUM
)
RECOVERY_STRESS_EVALUATOR_SEED = 20260723

_MULTI_MEASUREMENT_SUITES = (
    "recovery_post_failure_no_candidate",
    "recovery_unsupported_correction",
    "recovery_premature_commit",
    "recovery_premature_escalation",
)
_MIXED_SUITES = (
    "recovery_rejected_candidate_rollback",
    "recovery_safe_continuation_after_partial_success",
    "recovery_measurement_parameter_sequential_handoff",
)

CandidateValidator = Callable[[str, Mapping[str, Any]], Mapping[str, Any]]


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must contain one JSON object")
    return dict(payload)


def _bound_hash(
    manifest: Mapping[str, Any],
    *,
    field: str,
    path: Path,
    label: str,
) -> str:
    expected = manifest.get(field)
    actual = file_sha256(path)
    if expected != actual:
        raise RuntimeError(
            f"{label} hash binding failed: field={field}, "
            f"expected={expected!r}, actual={actual}"
        )
    return actual


def _development_rows(path: Path) -> list[dict[str, Any]]:
    payload = _load_json_object(path, label="development holdout")
    if set(payload) != {"dagger1_development"}:
        raise ValueError(
            "development holdout must contain exactly dagger1_development"
        )
    rows = payload["dagger1_development"]
    if not isinstance(rows, list) or len(rows) != 30:
        raise ValueError("development holdout must contain exactly 30 rows")
    if not all(isinstance(row, Mapping) for row in rows):
        raise ValueError("development holdout contains a non-object row")
    return [copy.deepcopy(dict(row)) for row in rows]


def _scenario_root(row: Mapping[str, Any]) -> str:
    grouping = row.get("grouping")
    grouping = grouping if isinstance(grouping, Mapping) else {}
    root = str(grouping.get("physical_root_fingerprint") or "").strip()
    if not root:
        raise ValueError("stress candidate has no physical root fingerprint")
    return root


def _scenario_family(row: Mapping[str, Any]) -> str:
    grouping = row.get("grouping")
    grouping = grouping if isinstance(grouping, Mapping) else {}
    return str(grouping.get("scenario_family") or "").strip()


def _measurement_error(row: Mapping[str, Any]) -> tuple[int, float, float]:
    audit = row.get("audit")
    audit = audit if isinstance(audit, Mapping) else {}
    truth = audit.get("truth")
    truth = truth if isinstance(truth, Mapping) else {}
    errors = truth.get("true_measurement_errors")
    if not isinstance(errors, list) or not errors or not isinstance(errors[0], Mapping):
        raise ValueError("mixed stress candidate has no audited measurement error")
    first = errors[0]
    index = first.get("index")
    clean = first.get("clean")
    execution = row.get("execution")
    execution = execution if isinstance(execution, Mapping) else {}
    measurements = execution.get("measurements")
    if (
        isinstance(index, bool)
        or not isinstance(index, int)
        or index < 0
        or not isinstance(measurements, list)
        or index >= len(measurements)
        or isinstance(clean, bool)
        or not isinstance(clean, (int, float))
    ):
        raise ValueError("mixed stress candidate measurement audit is malformed")
    observed = measurements[index]
    if isinstance(observed, bool) or not isinstance(observed, (int, float)):
        raise ValueError("mixed stress candidate observed measurement is malformed")
    return index, float(clean), float(observed)


def _failed_action_intervention(suite: str) -> dict[str, Any]:
    if suite == "recovery_post_failure_no_candidate":
        action = {"tool": RUN_WLS, "arguments": {"state_id": "$active"}}
        error_code = "injected_transient_tool_failure"
    elif suite == "recovery_unsupported_correction":
        action = {
            "tool": CORRECT_MEASUREMENTS,
            "arguments": {"state_id": "$active", "suspect_group": [0]},
        }
        error_code = "correction_not_supported_by_current_context"
    elif suite == "recovery_premature_commit":
        action = {
            "tool": COMMIT_STATE,
            "arguments": {
                "candidate_state_id": "stress-missing-candidate"
            },
        }
        error_code = "candidate_lifecycle_violation"
    elif suite == "recovery_premature_escalation":
        action = {
            "tool": ASK_FOR_MORE_EVIDENCE,
            "arguments": {
                "state_id": "$active",
                "request": RECOVERY_OPTIONS_EXHAUSTED_REQUEST,
            },
        }
        error_code = "operator_escalation_precondition_not_met"
    else:
        raise ValueError(f"unsupported failed-action stress suite: {suite}")
    return {
        "intervention_schema_version": 1,
        "kind": "failed_policy_action",
        "action": action,
        "error_code": error_code,
    }


def _intervention_for_suite(
    suite: str,
    row: Mapping[str, Any],
) -> dict[str, Any]:
    if suite in _MULTI_MEASUREMENT_SUITES:
        return _failed_action_intervention(suite)
    if suite == "recovery_rejected_candidate_rollback":
        index, _, observed = _measurement_error(row)
        return {
            "intervention_schema_version": 1,
            "kind": "open_rejected_candidate",
            "setup_actions": [
                {
                    "tool": GET_MEASUREMENT_CONTEXT,
                    "arguments": {"state_id": "$active"},
                },
                {
                    "tool": CORRECT_MEASUREMENTS,
                    "arguments": {
                        "state_id": "$active",
                        "measurement_updates": {str(index): observed + 10.0},
                    },
                },
                {"tool": RUN_WLS, "arguments": {"state_id": "$candidate"}},
            ],
            "required_disposition": "REJECT",
        }
    if suite in {
        "recovery_safe_continuation_after_partial_success",
        "recovery_measurement_parameter_sequential_handoff",
    }:
        setup_actions = _partial_setup_actions(row)
        kind = "committed_partial_correction"
        if suite == "recovery_measurement_parameter_sequential_handoff":
            kind = "committed_partial_correction_with_observable_bridge"
            setup_actions.append(
                {
                    "tool": GET_MEASUREMENT_CONTEXT,
                    "arguments": {"state_id": "$active"},
                }
            )
        return {
            "intervention_schema_version": 1,
            "kind": kind,
            "setup_actions": setup_actions,
            "retention_required": True,
        }
    raise ValueError(f"unsupported recovery-stress suite: {suite}")


def _stress_row(suite: str, source: Mapping[str, Any]) -> dict[str, Any]:
    row = copy.deepcopy(dict(source))
    audit = row.get("audit")
    grouping = row.get("grouping")
    if not isinstance(audit, Mapping) or not isinstance(grouping, Mapping):
        raise ValueError("stress candidate is not a partitioned scenario")
    row["audit"] = copy.deepcopy(dict(audit))
    row["grouping"] = copy.deepcopy(dict(grouping))
    row["audit"]["evaluation_intervention"] = _intervention_for_suite(
        suite, row
    )
    row["grouping"]["split"] = RECOVERY_STRESS_SPLIT
    return row


def validate_recovery_stress_candidate(
    suite: str,
    scenario: Mapping[str, Any],
) -> dict[str, Any]:
    """Execute one real intervention and prove its first objective action."""

    result = evaluate_rollout_suites(
        {suite: [copy.deepcopy(dict(scenario))]},
        env_factory=production_environment_factory,
        policy_factory=observable_expert_policy_factory,
        case_loader=deterministic_case_loader,
        max_steps=1,
        seed=RECOVERY_STRESS_EVALUATOR_SEED,
        required_suites=(suite,),
        minimum_suites=1,
        minimum_episodes_per_suite=1,
        minimum_roots_per_suite=1,
        require_release_environment=True,
        expected_policy_identity={
            "explicit_policy_identity": EXPERT_POLICY_IDENTITY,
            "model_id": None,
            "model_revision": None,
        },
        require_policy_identity=True,
    )
    episodes = result.suite_metrics.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != 1:
        raise RuntimeError("stress candidate evaluation did not return one episode")
    episode = episodes[0]
    intervention = episode.get("evaluation_intervention")
    intervention = intervention if isinstance(intervention, Mapping) else {}
    pre_steps = intervention.get("pre_policy_step_count")
    trace = episode.get("trace")
    if (
        isinstance(pre_steps, bool)
        or not isinstance(pre_steps, int)
        or not isinstance(trace, list)
        or pre_steps < 1
        or pre_steps >= len(trace)
    ):
        raise RuntimeError("stress candidate has invalid pre-policy trace evidence")
    first_policy = trace[pre_steps]
    assessment = first_policy.get("objective_action_assessment")
    assessment = assessment if isinstance(assessment, Mapping) else {}
    expected_stratum = RECOVERY_STRESS_SUITE_TO_STRATUM[suite]
    expected_action = assessment.get("expected_action")
    policy_action = first_policy.get("action")
    checks = {
        "evidence_available": assessment.get("evidence_available") is True,
        "no_policy_leakage": assessment.get("policy_payload_leakage_paths") == [],
        "exact_stratum": assessment.get("recovery_stratum") == expected_stratum,
        "expert_matches_objective": expected_action == policy_action,
        "no_evaluator_error": episode.get("evaluator_error") is None,
        "intervention_applied": intervention.get("applied") is True,
    }
    failures = sorted(name for name, passed in checks.items() if not passed)
    if failures:
        raise RuntimeError(
            "stress candidate objective validation failed: " + ", ".join(failures)
        )
    return {
        "passed": True,
        "suite": suite,
        "recovery_stratum": expected_stratum,
        "physical_root_fingerprint": _scenario_root(scenario),
        "pre_policy_step_count": pre_steps,
        "expected_action_tool": (
            expected_action.get("tool")
            if isinstance(expected_action, Mapping)
            else None
        ),
        "expected_action_sha256": stable_json_sha256(expected_action),
    }


def build_recovery_stress_payload(
    development_rows: Sequence[Mapping[str, Any]],
    *,
    validator: CandidateValidator = validate_recovery_stress_candidate,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Select ten physically validated roots for every recovery cell."""

    if not callable(validator):
        raise TypeError("validator must be callable")
    roots = [_scenario_root(row) for row in development_rows]
    if len(roots) != len(set(roots)):
        raise ValueError("development parent repeats a physical root")
    by_family: dict[str, list[Mapping[str, Any]]] = {
        "multi_measurement": [],
        "measurement+parameter": [],
    }
    for row in development_rows:
        family = _scenario_family(row)
        if family in by_family:
            by_family[family].append(row)
    for family in by_family:
        by_family[family].sort(key=_scenario_root)

    selected: dict[str, list[Mapping[str, Any]]] = {}
    validation_records: list[dict[str, Any]] = []
    rejected_records: list[dict[str, Any]] = []
    for family, suites in (
        ("multi_measurement", _MULTI_MEASUREMENT_SUITES),
        ("measurement+parameter", _MIXED_SUITES),
    ):
        accepted: list[Mapping[str, Any]] = []
        for source in by_family[family]:
            root = _scenario_root(source)
            root_records: list[dict[str, Any]] = []
            try:
                for suite in suites:
                    row = _stress_row(suite, source)
                    record = dict(validator(suite, row))
                    if record.get("passed") is not True:
                        raise RuntimeError("validator did not return passed=true")
                    root_records.append(record)
            except Exception as exc:
                rejected_records.append(
                    {
                        "physical_root_fingerprint": root,
                        "family": family,
                        "reason": type(exc).__name__,
                        "detail": str(exc),
                    }
                )
                continue
            accepted.append(source)
            validation_records.extend(root_records)
            if len(accepted) == RECOVERY_STRESS_ROOTS_PER_STRATUM:
                break
        if len(accepted) != RECOVERY_STRESS_ROOTS_PER_STRATUM:
            family_rejections = [
                row for row in rejected_records if row["family"] == family
            ]
            rejection_summary = dict(
                sorted(Counter(row["reason"] for row in family_rejections).items())
            )
            raise RuntimeError(
                f"insufficient validated {family} stress roots: "
                f"required={RECOVERY_STRESS_ROOTS_PER_STRATUM}, "
                f"observed={len(accepted)}, rejection_summary="
                f"{json.dumps(rejection_summary, sort_keys=True)}, examples="
                f"{json.dumps(family_rejections[:3], sort_keys=True)}"
            )
        selected[family] = accepted

    suites_payload = {name: [] for name in RECOVERY_STRESS_SUITES}
    for suite in _MULTI_MEASUREMENT_SUITES:
        suites_payload[suite] = [
            _stress_row(suite, row) for row in selected["multi_measurement"]
        ]
    for suite in _MIXED_SUITES:
        suites_payload[suite] = [
            _stress_row(suite, row)
            for row in selected["measurement+parameter"]
        ]
    for suite in suites_payload:
        suites_payload[suite].sort(key=_scenario_root)

    validate_release_scenario_suites(suites_payload)
    fingerprint = fingerprint_evaluation_suites(
        suites_payload,
        seed=RECOVERY_STRESS_EVALUATOR_SEED,
        required_suites=RECOVERY_STRESS_SUITES,
        minimum_suites=len(RECOVERY_STRESS_SUITES),
        minimum_episodes_per_suite=RECOVERY_STRESS_ROOTS_PER_STRATUM,
        minimum_roots_per_suite={
            suite: RECOVERY_STRESS_ROOTS_PER_STRATUM
            for suite in RECOVERY_STRESS_SUITES
        },
    )
    selected_roots = {
        _scenario_root(row)
        for rows in selected.values()
        for row in rows
    }
    if len(selected_roots) != RECOVERY_STRESS_DISTINCT_ROOT_COUNT:
        raise RuntimeError("stress family allocations are not root-disjoint")
    return suites_payload, {
        "validation_records": validation_records,
        "rejected_candidates": rejected_records,
        "selected_roots_by_family": {
            family: sorted(_scenario_root(row) for row in rows)
            for family, rows in sorted(selected.items())
        },
        "fingerprint": fingerprint,
    }


def _source_bindings(repo_root: Path) -> dict[str, str]:
    paths = {
        Path(__file__).resolve(),
        Path(inspect.getsourcefile(evaluate_rollout_suites) or "").resolve(),
        Path(inspect.getsourcefile(production_environment_factory) or "").resolve(),
        Path(inspect.getsourcefile(_partial_setup_actions) or "").resolve(),
        Path(inspect.getsourcefile(physical_roots_from_artifact) or "").resolve(),
    }
    if any(not path.is_file() for path in paths):
        raise RuntimeError("recovery-stress source closure is not inspectable")
    return {
        str(path.relative_to(repo_root.resolve())).replace(os.sep, "/"): file_sha256(path)
        for path in sorted(paths)
    }


def build_dagger1_recovery_stress(
    *,
    development_holdout: Path,
    development_holdout_manifest: Path,
    development_holdout_generator_report: Path,
    d0_aggregate_dir: Path,
    d1_training_scenarios: Path,
    d1_training_manifest: Path,
    recovery_probes: Path,
    recovery_probe_manifest: Path,
    frozen_suite: Path,
    output: Path,
    validator: CandidateValidator = validate_recovery_stress_candidate,
) -> dict[str, Any]:
    """Build and publish one immutable recovery-stress suite and manifest."""

    repo_root = Path(__file__).resolve().parents[2]
    source_state = git_source_state(repo_root)
    if source_state.get("release_eligible_source") is not True:
        raise RuntimeError(
            "recovery-stress generation requires a clean committed source tree"
        )
    manifest_output = output.with_suffix(output.suffix + ".manifest.json")
    if output.exists() or manifest_output.exists():
        raise FileExistsError("recovery-stress output or manifest already exists")

    development_manifest = _load_json_object(
        development_holdout_manifest,
        label="development holdout manifest",
    )
    _bound_hash(
        development_manifest,
        field="output_sha256",
        path=development_holdout,
        label="development holdout",
    )
    _bound_hash(
        development_manifest,
        field="generator_report_sha256",
        path=development_holdout_generator_report,
        label="development generator report",
    )
    d1_manifest = _load_json_object(
        d1_training_manifest,
        label="D1 training manifest",
    )
    _bound_hash(
        d1_manifest,
        field="output_sha256",
        path=d1_training_scenarios,
        label="D1 training scenarios",
    )
    probe_manifest = _load_json_object(
        recovery_probe_manifest,
        label="recovery probe manifest",
    )
    probe_rows_binding = probe_manifest.get("probe_rows")
    if not isinstance(probe_rows_binding, Mapping):
        raise RuntimeError("recovery probe manifest has no probe_rows binding")
    _bound_hash(
        probe_rows_binding,
        field="sha256",
        path=recovery_probes,
        label="recovery probe rows",
    )
    if probe_manifest.get("passed") is not True:
        raise RuntimeError("recovery probe manifest is not passing")

    d0_provenance_path = d0_aggregate_dir / "aggregate.generation_provenance.json"
    d0_provenance = _load_json_object(
        d0_provenance_path,
        label="D0 generation provenance",
    )
    aggregate_binding = validate_aggregate_manifest_binding(
        d0_provenance,
        aggregate_dir=d0_aggregate_dir,
    )
    if aggregate_binding.get("passed") is not True:
        raise RuntimeError(
            "D0 aggregate manifest binding failed: "
            + "; ".join(aggregate_binding.get("failures", []))
        )

    development_rows = _development_rows(development_holdout)
    suites, selection = build_recovery_stress_payload(
        development_rows,
        validator=validator,
    )
    parent_roots = physical_roots_from_artifact(development_holdout)
    stress_roots = {
        _scenario_root(row)
        for rows in suites.values()
        for row in rows
    }
    protected_roots = {
        "d0": physical_roots_from_artifact(
            d0_aggregate_dir / "aggregate.raw.jsonl"
        ),
        "d1_training": physical_roots_from_artifact(d1_training_scenarios),
        "recovery_probes": physical_roots_from_artifact(recovery_probes),
        "frozen_evaluation": physical_roots_from_artifact(frozen_suite),
    }
    if not stress_roots <= parent_roots:
        raise RuntimeError("stress roots are not contained by the development parent")
    protected_overlap = {
        name: sorted(stress_roots & roots)
        for name, roots in sorted(protected_roots.items())
    }
    if any(protected_overlap.values()):
        raise RuntimeError(
            "recovery-stress roots overlap a protected source: "
            + json.dumps(protected_overlap, sort_keys=True)
        )
    if len(stress_roots) != RECOVERY_STRESS_DISTINCT_ROOT_COUNT:
        raise RuntimeError("recovery-stress suite has the wrong distinct-root count")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(suites, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    input_bindings = {
        "development_holdout": file_sha256(development_holdout),
        "development_holdout_manifest": file_sha256(
            development_holdout_manifest
        ),
        "development_holdout_generator_report": file_sha256(
            development_holdout_generator_report
        ),
        "d0_raw": file_sha256(d0_aggregate_dir / "aggregate.raw.jsonl"),
        "d0_generation_provenance": file_sha256(d0_provenance_path),
        "d0_manifest": file_sha256(d0_aggregate_dir / "aggregate.manifest.json"),
        "d1_training_scenarios": file_sha256(d1_training_scenarios),
        "d1_training_manifest": file_sha256(d1_training_manifest),
        "recovery_probes": file_sha256(recovery_probes),
        "recovery_probe_manifest": file_sha256(recovery_probe_manifest),
        "frozen_suite": file_sha256(frozen_suite),
    }
    rows_by_suite = {
        name: len(rows) for name, rows in sorted(suites.items())
    }
    roots_by_suite = {
        name: len({_scenario_root(row) for row in rows})
        for name, rows in sorted(suites.items())
    }
    manifest = {
        "schema_version": 1,
        "scenario_schema_version": 1,
        "artifact_type": RECOVERY_STRESS_ARTIFACT_TYPE,
        "contract": RECOVERY_STRESS_CONTRACT,
        "source_state": source_state,
        "source_bindings": _source_bindings(repo_root),
        "suite_format": "evaluation_suite_mapping_v1",
        "suite_names": list(RECOVERY_STRESS_SUITES),
        "split": RECOVERY_STRESS_SPLIT,
        "evaluator_seed": RECOVERY_STRESS_EVALUATOR_SEED,
        "rows": sum(rows_by_suite.values()),
        "rows_by_suite": rows_by_suite,
        "distinct_physical_roots": len(stress_roots),
        "distinct_roots_by_suite": roots_by_suite,
        "minimum_distinct_roots_per_stratum": (
            RECOVERY_STRESS_ROOTS_PER_STRATUM
        ),
        "development_parent_root_count": len(parent_roots),
        "development_parent_subset": sorted(stress_roots) == sorted(
            stress_roots & parent_roots
        ),
        "training_eligible": False,
        "model_selection_eligible": False,
        "recovery_test_evidence_eligible": True,
        "natural_coverage_eligible": False,
        "probe_training_root_overlap": protected_overlap["recovery_probes"],
        "protected_root_overlap": protected_overlap,
        "root_set_sha256": {
            "stress": root_set_digest(stress_roots),
            "development_parent": root_set_digest(parent_roots),
            **{
                name: root_set_digest(roots)
                for name, roots in sorted(protected_roots.items())
            },
        },
        "input_bindings": input_bindings,
        "input_bindings_sha256": stable_json_sha256(input_bindings),
        "selection": selection,
        "output_sha256": file_sha256(output),
    }
    with manifest_output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build the seven-cell, root-disjoint DAgger recovery-stress suite"
        )
    )
    parser.add_argument("--development-holdout", type=Path, required=True)
    parser.add_argument(
        "--development-holdout-manifest", type=Path, required=True
    )
    parser.add_argument(
        "--development-holdout-generator-report", type=Path, required=True
    )
    parser.add_argument("--d0-aggregate-dir", type=Path, required=True)
    parser.add_argument("--d1-training-scenarios", type=Path, required=True)
    parser.add_argument("--d1-training-manifest", type=Path, required=True)
    parser.add_argument("--recovery-probes", type=Path, required=True)
    parser.add_argument("--recovery-probe-manifest", type=Path, required=True)
    parser.add_argument("--frozen-suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    manifest = build_dagger1_recovery_stress(
        development_holdout=args.development_holdout,
        development_holdout_manifest=args.development_holdout_manifest,
        development_holdout_generator_report=(
            args.development_holdout_generator_report
        ),
        d0_aggregate_dir=args.d0_aggregate_dir,
        d1_training_scenarios=args.d1_training_scenarios,
        d1_training_manifest=args.d1_training_manifest,
        recovery_probes=args.recovery_probes,
        recovery_probe_manifest=args.recovery_probe_manifest,
        frozen_suite=args.frozen_suite,
        output=args.output,
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
