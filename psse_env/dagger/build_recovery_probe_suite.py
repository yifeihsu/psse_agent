"""Build the observable recovery-probe suite as a separate auxiliary source.

The probe suite is published beside the natural DAgger-1 corpus, never inside
it: its own JSONL, its own manifest, its own provenance, and its own replay
quota.  Nothing here may satisfy a natural on-policy floor.

Probe roots are drawn only from scenarios whose physical root is outside the
frozen evaluation suite, the development holdout, and the D0 aggregate.  Sharing
a root with the *natural* DAgger corpus is permitted and reported: a probe
deliberately visits a state the learner did not reach, so it leaks no evaluation
answer, but the overlap belongs in the record.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import psse_env.dagger.offline_teacher_target_audit as offline_audit_module
import psse_env.dagger.recovery_probes as recovery_probes_module
import psse_env.dagger.release_factories as release_factories_module
import psse_env.dagger.rollout_collector as rollout_collector_module
from psse_env.dagger.collect_dagger1 import (
    DEFAULT_EVALUATION_POLICY,
    DEFAULT_FORBIDDEN_SUITE,
    _file_sha256,
    _write_fsynced_jsonl,
    frozen_physical_roots,
    validate_d0_provenance_binding,
    validate_development_holdout_binding,
    validate_scenario_builder_manifest,
    validate_training_scenarios,
    validate_training_source_report,
)
from psse_env.dagger.recovery_probes import (
    RECOVERY_PROBE_CONTRACT,
    RECOVERY_PROBE_COLLECTION_ROLE,
    RECOVERY_PROBE_DATASET_SOURCE,
    RECOVERY_PROBE_ROOT_QUOTAS,
    RECOVERY_PROBE_STATE_ORIGIN,
    audit_recovery_probe_support,
    generate_recovery_probes,
    recovery_probe_manifest,
)
from psse_env.dagger.release_factories import (
    production_environment_factory,
    select_observable_expert_actions,
)
from psse_env.dagger.round1_view_policy import (
    ROUND1_THREE_SOURCE_VIEW_POLICY,
    round1_view_policy_digest,
)
from psse_env.dagger.root_sets import (
    physical_roots_from_artifact,
    root_set_digest,
)
from psse_env.dagger.rollout_collector import classify_state_example
from psse_env.oracle.expert_policy import ExpertPolicyOracle
from psse_env.sft.provenance import (
    AGGREGATE_MANIFEST_FILENAME,
    file_sha256,
    git_source_state,
    stable_json_sha256,
)

PROBE_SUITE_ARTIFACT_TYPE = "dagger1_observable_recovery_probe_suite"
PROBE_GENERATOR_IDENTITY = "observable_recovery_probe_generator_v2"
PROBE_PROVENANCE_CONTRACT = "dagger1_recovery_probe_provenance_v1"
PROBE_BINDING_VALIDATION_CONTRACT = "dagger1_recovery_probe_binding_v1"


def _load_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return dict(payload)


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"{path}:{number} must contain a JSON object")
            rows.append(dict(value))
    if not rows:
        raise ValueError(f"{path} contains no probe rows")
    return rows


def _relative_source_hash(path: Path, *, repo_root: Path) -> tuple[str, str]:
    resolved = Path(path).resolve()
    return str(resolved.relative_to(repo_root.resolve())).replace("\\", "/"), file_sha256(resolved)


def _factory_identities() -> dict[str, dict[str, str]]:
    bindings: dict[str, tuple[str, Any]] = {
        "environment": (
            "psse_env.dagger.release_factories:production_environment_factory",
            production_environment_factory,
        ),
        "expert_oracle": (
            "psse_env.oracle.expert_policy:ExpertPolicyOracle",
            ExpertPolicyOracle,
        ),
        "expert_selector": (
            "psse_env.dagger.release_factories:select_observable_expert_actions",
            select_observable_expert_actions,
        ),
    }
    result: dict[str, dict[str, str]] = {}
    for role, (import_spec, value) in bindings.items():
        source = inspect.getsourcefile(value)
        if source is None:
            raise RuntimeError(f"cannot identify source for probe factory {role}")
        result[role] = {
            "import_spec": import_spec,
            "source_sha256": file_sha256(source),
        }
    return result


def envelope_roots(path: Path) -> set[str]:
    """Physical roots named by a scenario list or a suite mapping.

    Delegates to the shared reader.  The real development holdout is a suite
    mapping, not the envelope list an earlier local reader assumed, so this
    stage raised before any disjointness check could run.
    """

    return physical_roots_from_artifact(path)


def aggregate_roots(directory: Path) -> set[str]:
    """Physical roots present in a D0 aggregate's raw row file."""

    return physical_roots_from_artifact(Path(directory) / "aggregate.raw.jsonl")


def _validated_probe_inputs(
    *,
    scenarios_path: Path,
    scenario_manifest_path: Path,
    scenario_generator_report_path: Path,
    development_holdout: Path,
    development_holdout_manifest: Path,
    development_holdout_generator_report: Path,
    d0_aggregate_dir: Path,
    forbidden_suite: Path,
    evaluation_policy: Path,
    reviewed_source_commit: str,
) -> dict[str, Any]:
    """Validate every mutable prerequisite and return content-bound facts."""

    repo_root = Path(__file__).resolve().parents[2]
    source_state = git_source_state(repo_root)
    if source_state.get("release_eligible_source") is not True:
        raise RuntimeError("recovery probes require a clean committed source tree")
    reviewed = str(reviewed_source_commit).strip().lower()
    if not reviewed or source_state.get("source_commit") != reviewed:
        raise RuntimeError(
            "recovery probe reviewed source commit differs from current clean HEAD"
        )

    paths = (
        scenarios_path,
        scenario_manifest_path,
        scenario_generator_report_path,
        development_holdout,
        development_holdout_manifest,
        development_holdout_generator_report,
        forbidden_suite,
        evaluation_policy,
    )
    for path in paths:
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    d0_raw_path = Path(d0_aggregate_dir) / "aggregate.raw.jsonl"
    d0_provenance_path = (
        Path(d0_aggregate_dir) / "aggregate.generation_provenance.json"
    )
    d0_manifest_path = Path(d0_aggregate_dir) / AGGREGATE_MANIFEST_FILENAME
    for path in (d0_raw_path, d0_provenance_path, d0_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    d0_provenance = _load_mapping(d0_provenance_path)
    validate_d0_provenance_binding(
        d0_provenance,
        raw_path=d0_raw_path,
        source_state=source_state,
    )
    scenario_report = _load_mapping(scenario_generator_report_path)
    validate_training_source_report(scenario_report)
    scenario_manifest = _load_mapping(scenario_manifest_path)
    scenarios_value = json.loads(Path(scenarios_path).read_text(encoding="utf-8"))
    if not isinstance(scenarios_value, list) or not scenarios_value:
        raise ValueError("probe scenarios must be a non-empty envelope list")
    if not all(isinstance(row, Mapping) for row in scenarios_value):
        raise ValueError("every probe scenario must be a JSON object")
    scenarios = [dict(row) for row in scenarios_value]
    validate_scenario_builder_manifest(
        scenario_manifest,
        scenarios=scenarios,
        input_path=Path(scenarios_path),
        generator_report_path=Path(scenario_generator_report_path),
        source_state=source_state,
        d0_raw_path=d0_raw_path,
        d0_provenance_path=d0_provenance_path,
        d0_manifest_path=d0_manifest_path,
        forbidden_suite_path=Path(forbidden_suite),
        evaluation_policy_path=Path(evaluation_policy),
    )
    development_roots = validate_development_holdout_binding(
        Path(development_holdout),
        Path(development_holdout_manifest),
        generator_report_path=Path(development_holdout_generator_report),
        source_state=source_state,
        scenario_input_path=Path(scenarios_path),
        scenario_manifest_path=Path(scenario_manifest_path),
        d0_raw_path=d0_raw_path,
        d0_provenance_path=d0_provenance_path,
        d0_manifest_path=d0_manifest_path,
        forbidden_suite_path=Path(forbidden_suite),
        evaluation_policy_path=Path(evaluation_policy),
        require_model_selection_eligible=True,
    )
    frozen = set(frozen_physical_roots(forbidden_suite))
    d0 = aggregate_roots(d0_aggregate_dir)
    validate_training_scenarios(
        scenarios,
        forbidden_roots=frozenset(frozen | d0 | set(development_roots)),
    )
    policy = _load_mapping(evaluation_policy)
    suite_policy = policy.get("suite_policy")
    suite_policy = suite_policy if isinstance(suite_policy, Mapping) else {}
    if (
        suite_policy.get("status") != "pinned"
        or suite_policy.get("approved_suite_sha256")
        != file_sha256(forbidden_suite)
    ):
        raise ValueError("probe frozen evaluation suite is not pinned/current")

    candidate_roots = envelope_roots(scenarios_path)
    input_paths = {
        "scenarios": Path(scenarios_path),
        "scenario_manifest": Path(scenario_manifest_path),
        "scenario_generator_report": Path(scenario_generator_report_path),
        "d0_raw": d0_raw_path,
        "d0_generation_provenance": d0_provenance_path,
        "d0_manifest": d0_manifest_path,
        "development_holdout": Path(development_holdout),
        "development_holdout_manifest": Path(development_holdout_manifest),
        "development_holdout_generator_report": Path(
            development_holdout_generator_report
        ),
        "frozen_evaluation_suite": Path(forbidden_suite),
        "evaluation_policy": Path(evaluation_policy),
    }
    return {
        "repo_root": repo_root,
        "source_state": source_state,
        "scenarios": scenarios,
        "candidate_roots": candidate_roots,
        "d0_roots": d0,
        "development_roots": set(development_roots),
        "frozen_roots": frozen,
        "d0_generation_provenance_id": d0_provenance.get(
            "generation_provenance_id"
        ),
        "input_artifacts": {
            name: {
                "path": str(path.resolve()),
                "sha256": file_sha256(path),
            }
            for name, path in sorted(input_paths.items())
        },
    }


def _normalized_probe_rows_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    """Bind row semantics without creating a provenance-ID hash cycle."""

    normalized = [
        {
            key: value
            for key, value in dict(row).items()
            if key != "generation_provenance_id"
        }
        for row in rows
    ]
    return stable_json_sha256(normalized)


def _probe_generation_descriptor(
    *,
    facts: Mapping[str, Any],
    probe_roots: set[str],
    probe_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    repo_root = Path(facts["repo_root"])
    source_files = (
        Path(__file__),
        Path(recovery_probes_module.__file__),
        Path(release_factories_module.__file__),
        Path(rollout_collector_module.__file__),
        Path(offline_audit_module.__file__),
    )
    root_sets = {
        "candidate_scenarios": set(facts["candidate_roots"]),
        "d0": set(facts["d0_roots"]),
        "development": set(facts["development_roots"]),
        "frozen_evaluation": set(facts["frozen_roots"]),
        "probe": set(probe_roots),
    }
    generator_hashes = dict(
        _relative_source_hash(path, repo_root=repo_root) for path in source_files
    )
    return {
        "contract": PROBE_PROVENANCE_CONTRACT,
        "schema_version": 1,
        "generator_identity": PROBE_GENERATOR_IDENTITY,
        "source_state": facts["source_state"],
        "round1_view_policy": {
            "contract": ROUND1_THREE_SOURCE_VIEW_POLICY["contract"],
            "digest": round1_view_policy_digest(),
        },
        "root_quotas": dict(sorted(RECOVERY_PROBE_ROOT_QUOTAS.items())),
        "generator_hashes": generator_hashes,
        "factory_identities": _factory_identities(),
        "input_artifacts": dict(facts["input_artifacts"]),
        "d0_generation_provenance_id": facts["d0_generation_provenance_id"],
        "normalized_probe_rows": {
            "row_count": len(probe_rows),
            "sha256": _normalized_probe_rows_sha256(probe_rows),
        },
        "root_sets": {
            name: {
                "count": len(roots),
                "root_set_sha256": root_set_digest(roots),
            }
            for name, roots in sorted(root_sets.items())
        },
    }


def _serialized_probe_rows_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    """Hash the exact JSONL bytes written by ``_write_fsynced_jsonl``."""

    payload = "".join(
        json.dumps(
            dict(row),
            sort_keys=True,
            default=str,
            allow_nan=False,
        )
        + os.linesep
        for row in rows
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _candidate_scenario_summary(
    facts: Mapping[str, Any],
) -> tuple[set[str], int, set[str], int]:
    """Recompute candidate and eligible roots from the parsed scenario bytes."""

    protected = (
        set(facts["d0_roots"])
        | set(facts["development_roots"])
        | set(facts["frozen_roots"])
    )
    candidate_roots: set[str] = set()
    eligible_roots: set[str] = set()
    eligible_scenarios = 0
    for envelope in facts["scenarios"]:
        grouping = envelope.get("grouping")
        grouping = grouping if isinstance(grouping, Mapping) else envelope
        root = str(grouping.get("physical_root_fingerprint") or "").strip()
        if not root:
            raise ValueError("validated probe scenario lacks a physical root")
        candidate_roots.add(root)
        if root not in protected:
            eligible_scenarios += 1
            eligible_roots.add(root)
    return (
        candidate_roots,
        eligible_scenarios,
        eligible_roots,
        len(candidate_roots - eligible_roots),
    )


def _scenario_row_bindings(facts: Mapping[str, Any]) -> set[tuple[Any, ...]]:
    """Public grouping identities that a generated probe row may claim."""

    bindings: set[tuple[Any, ...]] = set()
    for envelope in facts["scenarios"]:
        grouping = envelope.get("grouping")
        grouping = grouping if isinstance(grouping, Mapping) else envelope
        bindings.add(
            (
                str(grouping.get("physical_root_fingerprint") or "").strip(),
                grouping.get("scenario_id"),
                str(grouping.get("scenario_family") or ""),
                int(grouping.get("error_cardinality") or 0),
            )
        )
    return bindings


def _rank_one_actions_from_row(
    observation: Mapping[str, Any],
    preferred_action: Mapping[str, Any] | str,
    proof: Mapping[str, Any],
) -> list[Mapping[str, Any] | str] | None:
    """Recover the ranked action list that the observable proof commits to."""

    action_count = proof.get("expert_action_count")
    if not isinstance(action_count, int) or isinstance(action_count, bool):
        return None
    evidence_by_family = observation.get("fresh_context_evidence")
    evidence = (
        evidence_by_family.get("parameter")
        if isinstance(evidence_by_family, Mapping)
        else None
    )
    supported = evidence.get("supported_corrections") if isinstance(evidence, Mapping) else None
    if isinstance(supported, (list, tuple)) and supported:
        if len(supported) != action_count or not all(
            isinstance(action, (Mapping, str)) for action in supported
        ):
            return None
        return list(supported)
    if action_count == 1:
        return [preferred_action]
    return None


def _generation_report_is_valid(
    report: Any,
    *,
    rows: Sequence[Mapping[str, Any]],
    support: Mapping[str, Any],
    facts: Mapping[str, Any],
) -> bool:
    if not isinstance(report, Mapping):
        return False
    expected_fields = {
        "contract",
        "scenarios_considered",
        "root_quotas",
        "roots_admitted",
        "quota_met",
        "skipped",
        "attempts",
        "probe_support",
        "passed",
    }
    if set(report) != expected_fields:
        return False

    roots_by_stratum = {
        stratum: {
            str(row.get("physical_root_fingerprint") or "").strip()
            for row in rows
            if row.get("recovery_stratum") == stratum
        }
        for stratum in RECOVERY_PROBE_ROOT_QUOTAS
    }
    actual_counts = {
        stratum: len(roots) for stratum, roots in sorted(roots_by_stratum.items())
    }
    quota_met = {
        stratum: actual_counts[stratum] >= quota
        for stratum, quota in sorted(RECOVERY_PROBE_ROOT_QUOTAS.items())
    }
    _, eligible_scenarios, _, _ = _candidate_scenario_summary(facts)
    skipped = report.get("skipped")
    if (
        not isinstance(skipped, Mapping)
        or any(not isinstance(key, str) for key in skipped)
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in skipped.values()
        )
    ):
        return False

    attempts = report.get("attempts")
    if not isinstance(attempts, list):
        return False
    admitted_pairs: set[tuple[str, str]] = set()
    candidate_roots = set(facts["candidate_roots"])
    for attempt in attempts:
        if not isinstance(attempt, Mapping) or set(attempt) != {
            "physical_root_fingerprint",
            "expected_stratum",
            "actual_stratum",
            "admitted",
        }:
            return False
        root = str(attempt.get("physical_root_fingerprint") or "").strip()
        expected = str(attempt.get("expected_stratum") or "")
        actual = attempt.get("actual_stratum")
        if (
            root not in candidate_roots
            or expected not in RECOVERY_PROBE_ROOT_QUOTAS
            or (actual is not None and not isinstance(actual, str))
            or not isinstance(attempt.get("admitted"), bool)
            or attempt.get("admitted") is not (actual == expected)
        ):
            return False
        if attempt["admitted"]:
            admitted_pairs.add((expected, root))
    row_pairs = {
        (
            str(row.get("recovery_stratum") or ""),
            str(row.get("physical_root_fingerprint") or "").strip(),
        )
        for row in rows
    }
    if not row_pairs <= admitted_pairs:
        return False

    expected_passed = bool(support.get("passed") and all(quota_met.values()))
    return bool(
        report.get("contract") == RECOVERY_PROBE_CONTRACT
        and report.get("scenarios_considered") == eligible_scenarios
        and report.get("root_quotas")
        == dict(sorted(RECOVERY_PROBE_ROOT_QUOTAS.items()))
        and report.get("roots_admitted") == actual_counts
        and report.get("quota_met") == quota_met
        and report.get("probe_support") == support
        and report.get("passed") is expected_passed
    )


def _expected_probe_manifest(
    *,
    rows: Sequence[Mapping[str, Any]],
    facts: Mapping[str, Any],
    descriptor: Mapping[str, Any],
    provenance_id: str,
    generation_report: Mapping[str, Any],
    rows_name: str,
    rows_sha256: str,
) -> dict[str, Any]:
    candidate_roots, eligible_count, eligible_roots, excluded_count = (
        _candidate_scenario_summary(facts)
    )
    protected_d0 = set(facts["d0_roots"])
    development = set(facts["development_roots"])
    frozen = set(facts["frozen_roots"])
    probe_roots = {
        str(row.get("physical_root_fingerprint") or "").strip() for row in rows
    }
    probe_roots.discard("")
    expected = recovery_probe_manifest(
        rows,
        generator_identity=PROBE_GENERATOR_IDENTITY,
        source_commit=str(facts["source_state"]["source_commit"]),
        natural_roots=sorted(candidate_roots),
        development_roots=sorted(development),
        frozen_evaluation_roots=sorted(frozen),
        d0_roots=sorted(protected_d0),
    )
    disjointness = dict(expected["root_disjointness"])
    containment = disjointness.pop("natural_dagger_overlap")
    disjointness["candidate_scenario_containment"] = containment
    disjointness["passed"] = bool(
        not disjointness["development_holdout_overlap"]
        and not disjointness["frozen_evaluation_overlap"]
        and not disjointness["d0_overlap"]
        and set(containment) == probe_roots
    )
    expected["root_disjointness"] = disjointness
    expected.update(
        {
            "artifact_type": PROBE_SUITE_ARTIFACT_TYPE,
            "schema_version": 1,
            "source_state": facts["source_state"],
            "generation_descriptor": dict(descriptor),
            "generation_provenance_id": provenance_id,
            "generation_report": dict(generation_report),
            "candidate_roots_considered": eligible_count,
            "candidate_roots_excluded": excluded_count,
            "passed": bool(
                expected["probe_support"]["passed"]
                and disjointness["passed"]
                and generation_report.get("passed") is True
            ),
            "probe_rows": {
                "relative_path": rows_name,
                "row_count": len(rows),
                "sha256": rows_sha256,
            },
        }
    )
    # ``eligible_roots`` is intentionally evaluated above even when all inputs
    # are already disjoint: it makes excluded-root counting derive from the
    # parsed scenario bytes rather than a mutable path read.
    del eligible_roots
    return expected


def _validate_recovery_probe_payload(
    *,
    rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    facts: Mapping[str, Any],
    rows_name: str,
    rows_sha256: str,
    require_release_eligible: bool = True,
) -> dict[str, Any]:
    """Validate probe semantics and every manifest claim from parsed values."""

    roots = {
        str(row.get("physical_root_fingerprint") or "").strip() for row in rows
    }
    roots.discard("")
    expected_descriptor = _probe_generation_descriptor(
        facts=facts,
        probe_roots=roots,
        probe_rows=rows,
    )
    expected_id = stable_json_sha256(expected_descriptor)
    support = audit_recovery_probe_support(rows)
    marker_failures: list[str] = []
    evidence_failures: list[str] = []
    source_row_bindings = _scenario_row_bindings(facts)
    roots_by_stratum: dict[str, set[str]] = {
        stratum: set() for stratum in RECOVERY_PROBE_ROOT_QUOTAS
    }
    for index, row in enumerate(rows):
        example = str(row.get("example_id") or index)
        root = str(row.get("physical_root_fingerprint") or "").strip()
        stratum = str(row.get("recovery_stratum") or "")
        if stratum in roots_by_stratum and root:
            roots_by_stratum[stratum].add(root)
        exact_markers = bool(
            row.get("collector_contract") == RECOVERY_PROBE_CONTRACT
            and row.get("state_origin") == RECOVERY_PROBE_STATE_ORIGIN
            and row.get("dataset_source") == RECOVERY_PROBE_DATASET_SOURCE
            and row.get("replay_source") == RECOVERY_PROBE_DATASET_SOURCE
            and row.get("collection_role") == RECOVERY_PROBE_COLLECTION_ROLE
            and row.get("state_visited_by") == RECOVERY_PROBE_STATE_ORIGIN
            and row.get("dataset_mode") == "production"
            and row.get("auxiliary_training_eligible") is True
            and row.get("production_label_eligible") is False
            and row.get("natural_on_policy_support_eligible") is False
            and row.get("training_decision_evidence_verified") is True
            and isinstance(row.get("probe_intervention"), Mapping)
            and bool(row.get("probe_intervention"))
            and row.get("generation_provenance_id") == expected_id
        )
        if not exact_markers:
            marker_failures.append(example)

        observation = row.get("policy_observation")
        preferred = row.get("preferred_action")
        state_class = row.get("state_class")
        family = row.get("scenario_family")
        cardinality = row.get("error_cardinality")
        stored_verification = row.get("probe_stratum_verification")
        stored_rank_one = row.get("observable_rank_one_target_proof")
        stored_offline = row.get("offline_teacher_target_audit")
        evidence_valid = bool(
            isinstance(observation, Mapping)
            and isinstance(preferred, (Mapping, str))
            and isinstance(state_class, str)
            and bool(state_class)
            and isinstance(family, str)
            and bool(family)
            and isinstance(cardinality, int)
            and not isinstance(cardinality, bool)
            and stratum in RECOVERY_PROBE_ROOT_QUOTAS
            and isinstance(stored_verification, Mapping)
            and isinstance(stored_rank_one, Mapping)
            and isinstance(stored_offline, Mapping)
            and (
                root,
                row.get("scenario_id"),
                family,
                cardinality,
            )
            in source_row_bindings
        )
        if evidence_valid:
            try:
                recomputed_state_class = classify_state_example(
                    observation,
                    preferred_action=preferred,
                )
                recomputed_verification = recovery_probes_module.verify_probe_stratum(
                    observation,
                    preferred_action=preferred,
                    state_class=recomputed_state_class,
                    scenario_family=family,
                    error_cardinality=cardinality,
                    expected_stratum=stratum,
                )
                rank_actions = _rank_one_actions_from_row(
                    observation,
                    preferred,
                    stored_rank_one,
                )
                recomputed_rank_one = (
                    rollout_collector_module.observable_rank_one_target_proof(
                        observation,
                        preferred_action=preferred,
                        expert_actions=rank_actions,
                    )
                    if rank_actions is not None
                    else None
                )
                validated_offline = (
                    offline_audit_module.validate_offline_teacher_target_audit_metadata(
                        stored_offline,
                        require_passed=True,
                    )
                )
                evidence_valid = bool(
                    state_class == recomputed_state_class
                    and dict(stored_verification) == recomputed_verification
                    and recomputed_verification.get("passed") is True
                    and recomputed_rank_one is not None
                    and dict(stored_rank_one) == recomputed_rank_one
                    and recomputed_rank_one.get("passed") is True
                    and dict(stored_offline) == validated_offline
                )
            except (KeyError, TypeError, ValueError, OverflowError):
                evidence_valid = False
        if not evidence_valid:
            evidence_failures.append(example)

    quota_counts = {
        stratum: len(roots) for stratum, roots in sorted(roots_by_stratum.items())
    }
    exact_quotas = quota_counts == dict(sorted(RECOVERY_PROBE_ROOT_QUOTAS.items()))
    candidate_roots, _, _, _ = _candidate_scenario_summary(facts)
    protected = (
        set(facts["d0_roots"])
        | set(facts["development_roots"])
        | set(facts["frozen_roots"])
    )
    generation_report = manifest.get("generation_report")
    generation_report_valid = _generation_report_is_valid(
        generation_report,
        rows=rows,
        support=support,
        facts=facts,
    )
    expected_manifest = _expected_probe_manifest(
        rows=rows,
        facts=facts,
        descriptor=expected_descriptor,
        provenance_id=expected_id,
        generation_report=(
            generation_report if isinstance(generation_report, Mapping) else {}
        ),
        rows_name=rows_name,
        rows_sha256=rows_sha256,
    )
    checks = {
        "source_candidate_roots": set(facts["candidate_roots"])
        == candidate_roots,
        "generation_descriptor": manifest.get("generation_descriptor")
        == expected_descriptor,
        "generation_provenance_id": manifest.get("generation_provenance_id")
        == expected_id,
        "row_markers": not marker_failures,
        "row_evidence": not evidence_failures,
        "candidate_containment": roots <= candidate_roots
        and (bool(roots) or not require_release_eligible),
        "protected_disjointness": not bool(roots & protected),
        "generation_report": generation_report_valid,
        "manifest_claims": dict(manifest) == expected_manifest,
    }
    if require_release_eligible:
        checks.update(
            {
                "support": support.get("passed") is True,
                "root_quotas": exact_quotas,
                "generation_report_passed": isinstance(generation_report, Mapping)
                and generation_report.get("passed") is True,
                "manifest_passed": manifest.get("passed") is True,
            }
        )
    failures = sorted(name for name, passed in checks.items() if not passed)
    if failures:
        raise ValueError(
            "recovery probe suite binding failed: " + ", ".join(failures)
        )
    return {
        "contract": PROBE_BINDING_VALIDATION_CONTRACT,
        "passed": True,
        "generation_provenance_id": expected_id,
        "probe_rows": len(rows),
        "probe_roots": len(roots),
        "probe_root_set_sha256": root_set_digest(roots),
        "rows_sha256": rows_sha256,
        "view_policy_digest": round1_view_policy_digest(),
        "support": support,
    }


def build_recovery_probe_suite(
    *,
    scenarios_path: Path,
    scenario_manifest_path: Path,
    scenario_generator_report_path: Path,
    output: Path,
    manifest_path: Path,
    reviewed_source_commit: str,
    forbidden_suite: Path,
    evaluation_policy: Path,
    development_holdout: Path,
    development_holdout_manifest: Path,
    development_holdout_generator_report: Path,
    d0_aggregate_dir: Path,
) -> dict[str, Any]:
    """Generate, verify, and atomically publish one probe suite."""

    output = Path(output)
    manifest_path = Path(manifest_path)
    for path in (output, manifest_path):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing to overwrite probe artifact: {path}")

    facts = _validated_probe_inputs(
        scenarios_path=Path(scenarios_path),
        scenario_manifest_path=Path(scenario_manifest_path),
        scenario_generator_report_path=Path(scenario_generator_report_path),
        development_holdout=Path(development_holdout),
        development_holdout_manifest=Path(development_holdout_manifest),
        development_holdout_generator_report=Path(
            development_holdout_generator_report
        ),
        d0_aggregate_dir=Path(d0_aggregate_dir),
        forbidden_suite=Path(forbidden_suite),
        evaluation_policy=Path(evaluation_policy),
        reviewed_source_commit=reviewed_source_commit,
    )
    payload = list(facts["scenarios"])
    frozen = set(facts["frozen_roots"])
    development = set(facts["development_roots"])
    d0 = set(facts["d0_roots"])
    excluded = frozen | development | d0

    candidate_roots: set[str] = set()
    eligible: list[Mapping[str, Any]] = []
    for envelope in payload:
        grouping = envelope.get("grouping")
        grouping = grouping if isinstance(grouping, Mapping) else envelope
        root = str(grouping.get("physical_root_fingerprint") or "").strip()
        if not root:
            raise ValueError("validated probe scenario lacks a physical root")
        candidate_roots.add(root)
        if root not in excluded:
            eligible.append(envelope)
    if not eligible:
        raise ValueError(
            "every candidate probe root is excluded by the frozen suite, "
            "development holdout, or D0 aggregate"
        )

    env = production_environment_factory()
    oracle = ExpertPolicyOracle(process_oracle=env.process_oracle)
    rows, report = generate_recovery_probes(
        eligible,
        env=env,
        expert_oracle=oracle,
        state_class_for=lambda observation, preferred: classify_state_example(
            observation, preferred_action=preferred
        ),
        quotas=dict(RECOVERY_PROBE_ROOT_QUOTAS),
    )

    probe_roots = {
        str(row.get("physical_root_fingerprint") or "").strip() for row in rows
    }
    probe_roots.discard("")
    descriptor = _probe_generation_descriptor(
        facts=facts,
        probe_roots=probe_roots,
        probe_rows=rows,
    )
    provenance_id = stable_json_sha256(descriptor)
    for row in rows:
        row["generation_provenance_id"] = provenance_id
    rows_sha256 = _serialized_probe_rows_sha256(rows)
    manifest = _expected_probe_manifest(
        rows=rows,
        facts=facts,
        descriptor=descriptor,
        provenance_id=provenance_id,
        generation_report=report,
        rows_name=output.name,
        rows_sha256=rows_sha256,
    )

    # Revalidate every mutable prerequisite after the potentially long probe
    # generation.  Comparing the parsed scenarios as well as the descriptor
    # closes the window where paths could be replaced with new, internally
    # valid artifacts while rows were generated from earlier in-memory bytes.
    refreshed_facts = _validated_probe_inputs(
        scenarios_path=Path(scenarios_path),
        scenario_manifest_path=Path(scenario_manifest_path),
        scenario_generator_report_path=Path(scenario_generator_report_path),
        development_holdout=Path(development_holdout),
        development_holdout_manifest=Path(development_holdout_manifest),
        development_holdout_generator_report=Path(
            development_holdout_generator_report
        ),
        d0_aggregate_dir=Path(d0_aggregate_dir),
        forbidden_suite=Path(forbidden_suite),
        evaluation_policy=Path(evaluation_policy),
        reviewed_source_commit=reviewed_source_commit,
    )
    refreshed_descriptor = _probe_generation_descriptor(
        facts=refreshed_facts,
        probe_roots=probe_roots,
        probe_rows=rows,
    )
    if (
        refreshed_facts["scenarios"] != facts["scenarios"]
        or refreshed_descriptor != descriptor
    ):
        raise RuntimeError(
            "recovery probe inputs changed while the suite was being generated"
        )
    _validate_recovery_probe_payload(
        rows=rows,
        manifest=manifest,
        facts=refreshed_facts,
        rows_name=output.name,
        rows_sha256=rows_sha256,
        require_release_eligible=False,
    )

    # Publish rows first so a manifest never names a file that does not exist.
    # Any byte-level write failure leaves only unclaimed row evidence; nothing
    # is written and then deleted to manufacture success.
    _write_fsynced_jsonl(output, rows)
    if _file_sha256(output) != rows_sha256:
        raise RuntimeError("published probe rows differ from the validated payload")
    with manifest_path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return manifest


def validate_recovery_probe_suite_binding(
    *,
    rows_path: Path,
    manifest_path: Path,
    scenarios_path: Path,
    scenario_manifest_path: Path,
    scenario_generator_report_path: Path,
    development_holdout: Path,
    development_holdout_manifest: Path,
    development_holdout_generator_report: Path,
    d0_aggregate_dir: Path,
    forbidden_suite: Path,
    evaluation_policy: Path,
    reviewed_source_commit: str,
) -> dict[str, Any]:
    """Recompute the complete probe binding from explicit immutable inputs."""

    facts = _validated_probe_inputs(
        scenarios_path=Path(scenarios_path),
        scenario_manifest_path=Path(scenario_manifest_path),
        scenario_generator_report_path=Path(scenario_generator_report_path),
        development_holdout=Path(development_holdout),
        development_holdout_manifest=Path(development_holdout_manifest),
        development_holdout_generator_report=Path(
            development_holdout_generator_report
        ),
        d0_aggregate_dir=Path(d0_aggregate_dir),
        forbidden_suite=Path(forbidden_suite),
        evaluation_policy=Path(evaluation_policy),
        reviewed_source_commit=reviewed_source_commit,
    )
    rows = _load_rows(rows_path)
    manifest = _load_mapping(manifest_path)
    report = _validate_recovery_probe_payload(
        rows=rows,
        manifest=manifest,
        facts=facts,
        rows_name=Path(rows_path).name,
        rows_sha256=file_sha256(rows_path),
    )
    return {**report, "manifest_sha256": file_sha256(manifest_path)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build the observable recovery-probe auxiliary suite. Exits 0 when "
            "the manifest passes, 20 when it does not, so orchestration can "
            "branch without parsing stdout."
        )
    )
    parser.add_argument("--scenarios", type=Path, required=True)
    parser.add_argument("--scenario-manifest", type=Path, required=True)
    parser.add_argument("--scenario-generator-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reviewed-source-commit", required=True)
    parser.add_argument(
        "--forbidden-suite", type=Path, default=DEFAULT_FORBIDDEN_SUITE
    )
    parser.add_argument(
        "--evaluation-policy", type=Path, default=DEFAULT_EVALUATION_POLICY
    )
    parser.add_argument("--development-holdout", type=Path, required=True)
    parser.add_argument(
        "--development-holdout-manifest", type=Path, required=True
    )
    parser.add_argument(
        "--development-holdout-generator-report", type=Path, required=True
    )
    parser.add_argument("--d0-aggregate-dir", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)

    manifest = build_recovery_probe_suite(
        scenarios_path=args.scenarios,
        scenario_manifest_path=args.scenario_manifest,
        scenario_generator_report_path=args.scenario_generator_report,
        output=args.output,
        manifest_path=args.manifest,
        reviewed_source_commit=args.reviewed_source_commit,
        forbidden_suite=args.forbidden_suite,
        evaluation_policy=args.evaluation_policy,
        development_holdout=args.development_holdout,
        development_holdout_manifest=args.development_holdout_manifest,
        development_holdout_generator_report=(
            args.development_holdout_generator_report
        ),
        d0_aggregate_dir=args.d0_aggregate_dir,
    )
    support = manifest["probe_support"]["probe_strata"]
    summary = " ".join(
        f"{stratum}={entry['distinct_physical_roots']}/"
        f"{entry['minimum_distinct_physical_roots']}"
        for stratum, entry in sorted(support.items())
    )
    print(
        f"[{'PROBE_SUITE_OK' if manifest['passed'] else 'PROBE_SUITE_SHORT'}] "
        f"{summary} roots={manifest['distinct_physical_roots']} "
        f"contract={RECOVERY_PROBE_CONTRACT}"
    )
    print(json.dumps(manifest["probe_support"], sort_keys=True))
    return 0 if manifest["passed"] else 20


if __name__ == "__main__":
    raise SystemExit(main())
