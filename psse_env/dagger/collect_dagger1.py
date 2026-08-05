from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import math
import random
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from psse_env.dagger.dataset_builder import write_jsonl
from psse_env.dagger.release_factories import (
    BC0_PARAMETER_RANKING_DOMINANCE_THRESHOLD,
)
from psse_env.dagger.rollout_collector import (
    DAGGER1_OBSERVABLE_RECOVERY_SUPERVISION,
    RECOMMENDED_DAGGER1_RECOVERY_STRATA,
    DaggerRolloutCollector,
    audit_dagger1_recovery_labels,
    audit_target_aware_state_classes,
)
from psse_env.oracle.expert_policy import ExpertPolicyOracle
from psse_env.providers.matpower import PARAMETER_RANKING_CONTRACT
from psse_env.sft.provenance import file_sha256, git_source_state


DEFAULT_FORBIDDEN_SUITE = (
    Path(__file__).resolve().parent / "suites" / "bc0_eval_suite_v1.json"
)
DEFAULT_EVALUATION_POLICY = (
    Path(__file__).resolve().parent / "bc0_evaluation_policy.json"
)
DEFAULT_ENV_FACTORY_SPEC = (
    "psse_env.dagger.release_factories:production_environment_factory"
)
DEFAULT_POLICY_FACTORY_SPEC = (
    "psse_env.dagger.release_factories:gemma_release_policy_factory"
)
_IMMUTABLE_REVISION = re.compile(r"(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})")
DEFAULT_TARGET_MIN_ROWS = 300
DEFAULT_TARGET_MAX_ROWS = 600
DAGGER1_SCENARIO_BUILDER_CONTRACT = (
    "fresh_train_partition_dagger1_scenarios_v2"
)
_TRUTH_KEYS = frozenset(
    {
        "clean_case",
        "clean_measurements",
        "clean_parameter_values",
        "hidden_truth",
        "oracle_action_hints",
        "suggested_actions",
        "true_measurement_errors",
        "true_parameter_errors",
        "true_topology_errors",
    }
)


def _forbidden_collection_paths(value: Any, prefix: str = "$") -> list[str]:
    paths: list[str] = []
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key)
            path = f"{prefix}.{key}"
            if (
                key == "audit"
                or key in _TRUTH_KEYS
                or key.startswith("true_")
                or key.startswith("clean_")
                or key.startswith("remaining_true_")
            ):
                paths.append(path)
                continue
            paths.extend(_forbidden_collection_paths(item, path))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            paths.extend(_forbidden_collection_paths(item, f"{prefix}[{index}]"))
    return paths


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_or_jsonl(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        payload: Any = [
            json.loads(line)
            for line in text.splitlines()
            if line.strip()
        ]
    else:
        payload = json.loads(text)
    if not isinstance(payload, list) or not payload:
        raise ValueError("DAgger-1 input must be a non-empty JSON/JSONL list")
    if not all(isinstance(row, Mapping) for row in payload):
        raise ValueError("every DAgger-1 scenario must be a JSON object")
    return [dict(row) for row in payload]


def frozen_physical_roots(path: Path) -> frozenset[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("forbidden evaluation suite must be a suite mapping")
    roots: set[str] = set()
    for rows in payload.values():
        if not isinstance(rows, list):
            raise ValueError("forbidden evaluation suite values must be lists")
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError("forbidden evaluation suite rows must be objects")
            grouping = row.get("grouping")
            if not isinstance(grouping, Mapping):
                raise ValueError("forbidden suite row is missing grouping")
            root = str(grouping.get("physical_root_fingerprint") or "").strip()
            if not root:
                raise ValueError("forbidden suite row is missing a physical root")
            roots.add(root)
    if not roots:
        raise ValueError("forbidden evaluation suite has no physical roots")
    return frozenset(roots)


def validate_training_scenarios(
    scenarios: Sequence[Mapping[str, Any]],
    *,
    forbidden_roots: frozenset[str],
) -> None:
    if not forbidden_roots:
        raise ValueError("forbidden physical-root holdout must be non-empty")
    for index, scenario in enumerate(scenarios):
        is_envelope = all(
            key in scenario for key in ("execution", "audit", "grouping")
        )
        if is_envelope:
            if scenario.get("scenario_schema_version") != 1:
                raise ValueError(
                    "DAgger-1 scenario_schema_version must equal 1"
                )
            execution = scenario.get("execution")
            audit = scenario.get("audit")
            grouping = scenario.get("grouping")
            if not all(
                isinstance(value, Mapping)
                for value in (execution, audit, grouping)
            ):
                raise ValueError("DAgger-1 execution/audit/grouping envelope is malformed")
            unexpected = sorted(
                set(scenario)
                - {"scenario_schema_version", "execution", "audit", "grouping"}
            )
            if unexpected:
                raise ValueError(
                    "DAgger-1 envelope has unexpected top-level keys: "
                    + ", ".join(unexpected)
                )
            leaked = sorted(
                _forbidden_collection_paths(execution)
                + _forbidden_collection_paths(grouping)
            )
            if leaked:
                raise ValueError(
                    "DAgger-1 execution/grouping is not truth-free: "
                    + ", ".join(leaked)
                )
            unexpected_audit = sorted(
                set(audit)
                - {"truth", "evaluation_intervention", "release_audit"}
            )
            truth = audit.get("truth")
            if unexpected_audit or not isinstance(truth, Mapping):
                raise ValueError(
                    "DAgger-1 audit must contain only private truth plus "
                    "optional quarantined evaluation metadata"
                )
            for key in ("evaluation_intervention", "release_audit"):
                if key in audit and not isinstance(audit.get(key), Mapping):
                    raise ValueError(
                        f"DAgger-1 private audit field {key!r} must be a mapping"
                    )
            invalid_truth_keys = sorted(
                str(key)
                for key in truth
                if not (
                    str(key) == "truth_complete"
                    or str(key).startswith("true_")
                    or str(key).startswith("clean_")
                    or str(key).startswith("remaining_true_")
                )
            )
            if invalid_truth_keys or truth.get("truth_complete") is not True:
                raise ValueError(
                    "DAgger-1 private audit truth contract is invalid: "
                    + ", ".join(invalid_truth_keys)
                )
            scenario_id = str(execution.get("scenario_id") or index)
            scenario_metadata = grouping
        else:
            if "audit" in scenario:
                raise ValueError(
                    "flat DAgger-1 scenarios may not contain audit; use the "
                    "versioned execution/audit/grouping envelope ($.audit)"
                )
            scenario_id = str(
                scenario.get("scenario_id") or scenario.get("id") or index
            )
            scenario_metadata = scenario
            forbidden_paths = sorted(_forbidden_collection_paths(scenario))
            if forbidden_paths:
                raise ValueError(
                    f"DAgger-1 scenario {scenario_id!r} is not truth-free: "
                    + ", ".join(forbidden_paths)
                )
        split = str(
            scenario_metadata.get("dataset_split")
            or scenario_metadata.get("split")
            or ""
        ).strip()
        if split not in {"train", "dagger_train"}:
            raise ValueError(
                f"DAgger-1 scenario {scenario_id!r} must declare train or "
                f"dagger_train split, got {split or 'missing'}"
            )
        root = str(
            scenario_metadata.get("physical_root_fingerprint") or ""
        ).strip()
        if not root:
            raise ValueError(
                f"DAgger-1 scenario {scenario_id!r} lacks a physical root"
            )
        if root in forbidden_roots:
            raise ValueError(
                f"DAgger-1 scenario {scenario_id!r} overlaps a protected "
                f"D0/evaluation root {root}"
            )


def validate_training_source_report(report: Mapping[str, Any]) -> None:
    """Require evidence that fresh roots came from the train source partition."""
    source_partition = report.get("source_partition")
    if not isinstance(source_partition, Mapping):
        raise ValueError("scenario-generator report lacks source_partition")
    if (
        source_partition.get("enabled") is not True
        or source_partition.get("selected") != "train"
    ):
        raise ValueError(
            "DAgger-1 scenarios must come from "
            "Round0ScenarioGenerator(source_partition='train')"
        )
    admission = report.get("parameter_ranking_admission")
    if not isinstance(admission, Mapping) or (
        admission.get("contract") != PARAMETER_RANKING_CONTRACT
        or admission.get("enforced") is not True
        or admission.get("threshold")
        != BC0_PARAMETER_RANKING_DOMINANCE_THRESHOLD
    ):
        raise ValueError(
            "DAgger-1 scenario report lacks the reviewed parameter-ranking "
            "admission threshold"
        )


def validate_scenario_builder_manifest(
    manifest: Mapping[str, Any],
    *,
    scenarios: Sequence[Mapping[str, Any]],
    input_path: Path,
    generator_report_path: Path,
    source_state: Mapping[str, Any],
    d0_raw_path: Path,
    d0_provenance_path: Path,
    forbidden_suite_path: Path,
    evaluation_policy_path: Path,
) -> None:
    """Bind the collected scenarios to the reviewed fresh-root builder."""

    manifest_source = manifest.get("source_state")
    manifest_source = (
        manifest_source if isinstance(manifest_source, Mapping) else {}
    )
    plan = manifest.get("plan")
    selected_counts = manifest.get("selected_count_by_family")
    if not isinstance(plan, Mapping) or not isinstance(selected_counts, Mapping):
        raise ValueError("scenario-builder manifest lacks exact family counts")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in (*plan.values(), *selected_counts.values())
    ):
        raise ValueError("scenario-builder family counts are invalid")
    try:
        normalized_plan = {str(key): int(value) for key, value in plan.items()}
        normalized_selected = {
            str(key): int(value) for key, value in selected_counts.items()
        }
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("scenario-builder family counts are invalid") from exc
    actual_counts: Counter[str] = Counter()
    actual_roots: set[str] = set()
    for scenario in scenarios:
        grouping = scenario.get("grouping")
        grouping = grouping if isinstance(grouping, Mapping) else {}
        family = str(grouping.get("scenario_family") or "").strip()
        root = str(grouping.get("physical_root_fingerprint") or "").strip()
        if family:
            actual_counts[family] += 1
        if root:
            actual_roots.add(root)
    checks = {
        "schema_version": manifest.get("schema_version") == 1,
        "builder_contract": (
            manifest.get("builder_contract")
            == DAGGER1_SCENARIO_BUILDER_CONTRACT
        ),
        "release_evidence_eligible": (
            manifest.get("release_evidence_eligible") is False
        ),
        "input_sha256": manifest.get("output_sha256") == _file_sha256(input_path),
        "generator_report_sha256": (
            manifest.get("generator_report_sha256")
            == _file_sha256(generator_report_path)
        ),
        "source_commit": (
            manifest_source.get("release_eligible_source") is True
            and manifest_source.get("source_commit")
            == source_state.get("source_commit")
        ),
        "source_partition": manifest.get("source_partition") == "train",
        "parameter_threshold": (
            manifest.get("parameter_ranking_dominance_threshold")
            == BC0_PARAMETER_RANKING_DOMINANCE_THRESHOLD
        ),
        "exact_family_counts": normalized_selected == normalized_plan,
        "scenario_count": manifest.get("scenario_count") == sum(normalized_plan.values()),
        "physical_root_count": (
            manifest.get("physical_root_count") == sum(normalized_plan.values())
        ),
        "actual_family_counts": dict(actual_counts) == normalized_plan,
        "actual_scenario_count": len(scenarios) == sum(normalized_plan.values()),
        "actual_unique_roots": len(actual_roots) == len(scenarios),
        "protected_root_overlap": manifest.get("protected_root_overlap") == [],
        "d0_raw_sha256": manifest.get("d0_raw_sha256") == _file_sha256(d0_raw_path),
        "d0_provenance_sha256": (
            manifest.get("d0_generation_provenance_sha256")
            == _file_sha256(d0_provenance_path)
        ),
        "forbidden_suite_sha256": (
            manifest.get("frozen_suite_sha256")
            == _file_sha256(forbidden_suite_path)
        ),
        "evaluation_policy_sha256": (
            manifest.get("evaluation_policy_sha256")
            == _file_sha256(evaluation_policy_path)
        ),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(
            "scenario-builder manifest binding failed: " + ", ".join(failed)
        )


def validate_collection_output_paths(
    *,
    output: Path,
    all_output: Path | None,
    protected_paths: Sequence[Path],
) -> Path:
    """Fail before collection if an output can overwrite evidence or inputs."""

    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    outputs = [output, manifest_path]
    if all_output is not None:
        outputs.append(all_output)
    resolved_outputs = [path.resolve() for path in outputs]
    if len(set(resolved_outputs)) != len(resolved_outputs):
        raise ValueError("DAgger-1 output paths must be mutually distinct")
    protected = {path.resolve() for path in protected_paths}
    collisions = sorted(
        str(path)
        for path, resolved in zip(outputs, resolved_outputs)
        if resolved in protected
    )
    if collisions:
        raise ValueError(
            "DAgger-1 outputs alias protected input/evidence paths: "
            + ", ".join(collisions)
        )
    existing = sorted(str(path) for path in outputs if path.exists())
    if existing:
        raise FileExistsError(
            "DAgger-1 refuses to overwrite existing outputs: "
            + ", ".join(existing)
        )
    return manifest_path


def validate_export_rows_truth_free(
    rows: Sequence[Mapping[str, Any]],
) -> None:
    """Reject any dynamic oracle-truth leak before writing either D1 output."""

    violations: list[str] = []
    for index, row in enumerate(rows):
        violations.extend(
            f"$[{index}]{path[1:]}"
            for path in _forbidden_collection_paths(row)
        )
    if violations:
        raise RuntimeError(
            "DAgger-1 export rows contain private oracle truth: "
            + ", ".join(sorted(violations)[:20])
        )


def _load_callable(spec: str, *, field: str) -> Callable[..., Any]:
    module_name, separator, attribute_path = str(spec).strip().partition(":")
    if not separator or not module_name or not attribute_path:
        raise ValueError(f"{field} must use MODULE:ATTRIBUTE syntax")
    value: Any = importlib.import_module(module_name)
    for part in attribute_path.split("."):
        value = getattr(value, part)
    if not callable(value):
        raise TypeError(f"{field} must resolve to a callable")
    return value


def _call_factory(
    factory: Callable[..., Any],
    *,
    seed: int,
    model_id: str | None = None,
    model_revision: str | None = None,
) -> Any:
    try:
        parameters = inspect.signature(factory).parameters
    except (TypeError, ValueError):
        parameters = {}
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    kwargs: dict[str, Any] = {}
    if "seed" in parameters or accepts_kwargs:
        kwargs["seed"] = seed
    if "rng" in parameters or accepts_kwargs:
        kwargs["rng"] = random.Random(seed)
    if model_id is not None and ("model_id" in parameters or accepts_kwargs):
        kwargs["model_id"] = model_id
    if model_revision is not None and (
        "model_revision" in parameters or accepts_kwargs
    ):
        kwargs["model_revision"] = model_revision
    return factory(**kwargs)


def _callable_source_binding(value: Callable[..., Any], spec: str) -> dict[str, str]:
    source = inspect.getsourcefile(value)
    if source is None:
        raise ValueError(f"factory {spec} has no inspectable source file")
    path = Path(source).resolve()
    return {"import_spec": spec, "source_sha256": file_sha256(path)}


def validate_collection_pass(*, collection_pass: str, beta: float) -> dict[str, Any]:
    """Validate the review-prescribed diagnostic or mixed-policy beta."""
    pass_name = str(collection_pass).strip()
    beta_value = float(beta)
    if pass_name == "diagnostic":
        passed = beta_value == 0.0
        expected = {"minimum": 0.0, "maximum": 0.0}
    elif pass_name == "training":
        passed = 0.25 <= beta_value <= 0.5
        expected = {"minimum": 0.25, "maximum": 0.5}
    else:
        raise ValueError("collection_pass must be diagnostic or training")
    report = {
        "collection_pass": pass_name,
        "observed_beta": beta_value,
        "expected_beta": expected,
        "passed": passed,
    }
    if not passed:
        raise ValueError(
            f"{pass_name} DAgger-1 pass requires beta in "
            f"[{expected['minimum']}, {expected['maximum']}], got {beta_value}"
        )
    return report


def recommended_collection_gate(
    rows: Sequence[Mapping[str, Any]],
    *,
    target_min_rows: int = DEFAULT_TARGET_MIN_ROWS,
    target_max_rows: int = DEFAULT_TARGET_MAX_ROWS,
) -> dict[str, Any]:
    """Report the recommended first-round row and recovery-strata coverage."""
    minimum = int(target_min_rows)
    maximum = int(target_max_rows)
    if minimum < 1 or maximum < minimum:
        raise ValueError("target row range must satisfy 1 <= minimum <= maximum")
    observed = len(rows)
    strata = Counter(str(row.get("recovery_stratum") or "unclassified") for row in rows)
    missing = sorted(RECOMMENDED_DAGGER1_RECOVERY_STRATA - set(strata))
    row_count_passed = minimum <= observed <= maximum
    strata_passed = not missing
    return {
        "recommended_row_target": {
            "minimum": minimum,
            "maximum": maximum,
            "observed": observed,
            "passed": row_count_passed,
        },
        "recommended_recovery_strata": sorted(
            RECOMMENDED_DAGGER1_RECOVERY_STRATA
        ),
        "observed_recovery_strata": dict(sorted(strata.items())),
        "missing_recommended_recovery_strata": missing,
        "recovery_strata_passed": strata_passed,
        "passed": row_count_passed and strata_passed,
    }


def targeted_state_coverage(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Audit independent-root support for the review's first D1 state cells."""
    cells: dict[str, set[str]] = {
        **{f"multi_measurement_cardinality_{value}": set() for value in (2, 4, 5)},
        "parameter_route_actionable": set(),
        "parameter_route_complete_negative": set(),
        "parameter_route_unavailable": set(),
        "parameter_near_1_2_strict_rank": set(),
        "sequential_measurement_first": set(),
        "sequential_parameter_first": set(),
        "partial_success_retention": set(),
    }
    for row in rows:
        root = str(row.get("physical_root_fingerprint") or "").strip()
        if not root:
            continue
        family = str(row.get("scenario_family") or "")
        try:
            cardinality = int(row.get("error_cardinality") or 0)
        except (TypeError, ValueError, OverflowError):
            cardinality = 0
        if (
            family == "multi_measurement"
            and cardinality in {2, 4, 5}
            and row.get("parameter_scans_available") is False
        ):
            cells[f"multi_measurement_cardinality_{cardinality}"].add(root)
        observation = row.get("policy_observation")
        observation = observation if isinstance(observation, Mapping) else {}
        evidence_by_family = observation.get("fresh_context_evidence")
        parameter = (
            evidence_by_family.get("parameter")
            if isinstance(evidence_by_family, Mapping)
            else None
        )
        if isinstance(parameter, Mapping):
            route = str(parameter.get("route_status") or "")
            if route == "actionable":
                cells["parameter_route_actionable"].add(root)
            elif route == "complete_negative":
                cells["parameter_route_complete_negative"].add(root)
            elif route.startswith("unavailable"):
                cells["parameter_route_unavailable"].add(root)
            try:
                ratio = float(parameter.get("parameter_ranking_dominance_ratio"))
            except (TypeError, ValueError, OverflowError):
                ratio = float("nan")
            if math.isfinite(ratio) and 1.0 < ratio < 1.2:
                cells["parameter_near_1_2_strict_rank"].add(root)
        history = observation.get("history_window")
        prior_families: set[str] = set()
        if isinstance(history, list):
            for event in history:
                action = event.get("action") if isinstance(event, Mapping) else None
                tool = str(action.get("tool") or "") if isinstance(action, Mapping) else ""
                if "measurement" in tool:
                    prior_families.add("measurement")
                elif "parameter" in tool:
                    prior_families.add("parameter")
        target = row.get("preferred_action")
        target_tool = str(target.get("tool") or "") if isinstance(target, Mapping) else ""
        if family == "measurement+parameter":
            if "measurement" in prior_families and "parameter" in target_tool:
                cells["sequential_measurement_first"].add(root)
            if "parameter" in prior_families and "measurement" in target_tool:
                cells["sequential_parameter_first"].add(root)
        if observation.get("accepted_corrections") and not observation.get(
            "no_material_anomaly_remaining"
        ):
            cells["partial_success_retention"].add(root)
    counts = {name: len(roots) for name, roots in sorted(cells.items())}
    missing = sorted(name for name, count in counts.items() if count < 1)
    return {
        "minimum_distinct_physical_roots_per_cell": 1,
        "distinct_physical_roots_by_cell": counts,
        "missing_cells": missing,
        "passed": not missing,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Collect production-safe DAgger-1 recovery labels on non-evaluation "
            "training roots."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--d0-aggregate-dir",
        type=Path,
        required=True,
        help="Current-commit D0 aggregate whose complete physical-root set is forbidden",
    )
    parser.add_argument(
        "--scenario-generator-report",
        type=Path,
        required=True,
        help=(
            "Round0ScenarioGenerator report proving source_partition='train'"
        ),
    )
    parser.add_argument(
        "--scenario-manifest",
        type=Path,
        required=True,
        help="Fresh-root builder manifest binding input and generator report",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help=(
            "Training-eligible recovery rows for the training pass, or "
            "explicitly training-ineligible recovery rows for diagnostic"
        ),
    )
    parser.add_argument(
        "--all-output",
        type=Path,
        help="Optional audit JSONL containing eligible and ineligible visited states",
    )
    parser.add_argument(
        "--forbidden-suite",
        type=Path,
        default=DEFAULT_FORBIDDEN_SUITE,
        help="Frozen evaluation suite whose physical roots are forbidden",
    )
    parser.add_argument(
        "--env-factory",
        default=DEFAULT_ENV_FACTORY_SPEC,
    )
    parser.add_argument(
        "--policy-factory",
        default=DEFAULT_POLICY_FACTORY_SPEC,
    )
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--iteration", type=int, default=1)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument(
        "--collection-pass",
        choices=("diagnostic", "training"),
        default="diagnostic",
        help="Diagnostic requires beta=0; training requires beta in [0.25, 0.5]",
    )
    parser.add_argument("--target-min-rows", type=int, default=DEFAULT_TARGET_MIN_ROWS)
    parser.add_argument("--target-max-rows", type=int, default=DEFAULT_TARGET_MAX_ROWS)
    parser.add_argument(
        "--require-recommended-target",
        action="store_true",
        help="Fail unless eligible rows meet the recommended row and stratum target",
    )
    parser.add_argument("--max-steps", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260719)
    args = parser.parse_args(list(argv) if argv is not None else None)

    if _IMMUTABLE_REVISION.fullmatch(args.model_revision) is None:
        parser.error("--model-revision must be a 40- or 64-character hex digest")
    if args.iteration < 1 or not 0.0 <= args.beta < 1.0:
        parser.error("DAgger-1 requires --iteration >= 1 and 0 <= --beta < 1")
    if args.max_steps != 24:
        parser.error("production DAgger-1 collection requires --max-steps 24")
    try:
        beta_contract = validate_collection_pass(
            collection_pass=args.collection_pass,
            beta=args.beta,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.collection_pass == "diagnostic" and args.require_recommended_target:
        parser.error(
            "--require-recommended-target is valid only for the training pass"
        )
    if args.env_factory != DEFAULT_ENV_FACTORY_SPEC:
        parser.error("DAgger-1 requires the reviewed production environment factory")
    if args.policy_factory != DEFAULT_POLICY_FACTORY_SPEC:
        parser.error("DAgger-1 requires the reviewed Gemma release policy factory")
    repo_root = Path(__file__).resolve().parents[2]
    source_state = git_source_state(repo_root)
    if source_state.get("release_eligible_source") is not True:
        raise RuntimeError("DAgger-1 collection requires a clean committed source tree")
    scenarios = _load_json_or_jsonl(args.input)
    source_report = json.loads(
        args.scenario_generator_report.read_text(encoding="utf-8")
    )
    if not isinstance(source_report, Mapping):
        parser.error("--scenario-generator-report must contain a JSON object")
    validate_training_source_report(source_report)
    scenario_manifest = json.loads(
        args.scenario_manifest.read_text(encoding="utf-8")
    )
    if not isinstance(scenario_manifest, Mapping):
        parser.error("--scenario-manifest must contain a JSON object")
    frozen_roots = frozen_physical_roots(args.forbidden_suite)
    suite_sha256 = _file_sha256(args.forbidden_suite)
    policy_payload = json.loads(DEFAULT_EVALUATION_POLICY.read_text(encoding="utf-8"))
    suite_policy = (
        policy_payload.get("suite_policy")
        if isinstance(policy_payload, Mapping)
        else None
    )
    if not isinstance(suite_policy, Mapping) or (
        suite_policy.get("status") != "pinned"
        or suite_policy.get("approved_suite_sha256") != suite_sha256
    ):
        raise RuntimeError(
            "DAgger-1 forbidden suite does not match the pinned evaluation policy"
        )
    d0_raw_path = args.d0_aggregate_dir / "aggregate.raw.jsonl"
    d0_provenance_path = (
        args.d0_aggregate_dir / "aggregate.generation_provenance.json"
    )
    output_manifest_path = validate_collection_output_paths(
        output=args.output,
        all_output=args.all_output,
        protected_paths=(
            args.input,
            args.scenario_generator_report,
            args.scenario_manifest,
            args.forbidden_suite,
            DEFAULT_EVALUATION_POLICY,
            d0_raw_path,
            d0_provenance_path,
        ),
    )
    if not d0_raw_path.is_file() or not d0_provenance_path.is_file():
        raise FileNotFoundError(
            "--d0-aggregate-dir must contain aggregate.raw.jsonl and "
            "aggregate.generation_provenance.json"
        )
    d0_rows = _load_json_or_jsonl(d0_raw_path)
    d0_roots = frozenset(
        str(row.get("physical_root_fingerprint") or "").strip()
        for row in d0_rows
        if str(row.get("physical_root_fingerprint") or "").strip()
    )
    if not d0_roots:
        raise RuntimeError("D0 aggregate has no physical roots")
    d0_provenance = json.loads(d0_provenance_path.read_text(encoding="utf-8"))
    d0_descriptor = (
        d0_provenance.get("generation_descriptor")
        if isinstance(d0_provenance, Mapping)
        else None
    )
    d0_source = (
        d0_descriptor.get("source_state")
        if isinstance(d0_descriptor, Mapping)
        else None
    )
    if not (
        isinstance(d0_provenance, Mapping)
        and d0_provenance.get("release_eligible") is True
        and isinstance(d0_source, Mapping)
        and d0_source.get("source_commit") == source_state.get("source_commit")
    ):
        raise RuntimeError("D0 aggregate is not release eligible for current source")
    d0_hashes = d0_provenance.get("dataset_hashes")
    d0_hashes = d0_hashes if isinstance(d0_hashes, Mapping) else {}
    if d0_hashes.get(d0_raw_path.name) != _file_sha256(d0_raw_path):
        raise RuntimeError("D0 aggregate raw bytes do not match provenance")
    validate_scenario_builder_manifest(
        scenario_manifest,
        scenarios=scenarios,
        input_path=args.input,
        generator_report_path=args.scenario_generator_report,
        source_state=source_state,
        d0_raw_path=d0_raw_path,
        d0_provenance_path=d0_provenance_path,
        forbidden_suite_path=args.forbidden_suite,
        evaluation_policy_path=DEFAULT_EVALUATION_POLICY,
    )
    forbidden_roots = frozen_roots | d0_roots
    validate_training_scenarios(scenarios, forbidden_roots=forbidden_roots)

    env_factory = _load_callable(args.env_factory, field="env_factory")
    policy_factory = _load_callable(args.policy_factory, field="policy_factory")
    env_factory_binding = _callable_source_binding(env_factory, args.env_factory)
    policy_factory_binding = _callable_source_binding(
        policy_factory, args.policy_factory
    )
    expert_source = inspect.getsourcefile(ExpertPolicyOracle)
    if expert_source is None:
        raise RuntimeError("ExpertPolicyOracle source is not inspectable")
    env = _call_factory(env_factory, seed=args.seed)
    if getattr(env, "production_dataset_mode", None) is not True:
        raise RuntimeError("DAgger-1 environment is not in production dataset mode")
    policy = _call_factory(
        policy_factory,
        seed=args.seed,
        model_id=args.model_id,
        model_revision=args.model_revision,
    )
    expert = ExpertPolicyOracle(
        process_oracle=env.process_oracle,
        candidate_oracle=env.candidate_quality_oracle,
    )
    rows = DaggerRolloutCollector(
        env=env,
        policy=policy,
        expert_oracle=expert,
        rng=random.Random(args.seed),
        supervision_policy=DAGGER1_OBSERVABLE_RECOVERY_SUPERVISION,
        forbidden_physical_roots=forbidden_roots,
    ).collect_iteration(
        scenarios=scenarios,
        iteration=args.iteration,
        beta=args.beta,
        max_steps=args.max_steps,
        collection_role=args.collection_pass,
    )
    validate_export_rows_truth_free(rows)
    class_audit = audit_target_aware_state_classes(rows)
    if not class_audit["passed"]:
        raise RuntimeError(f"DAgger-1 class audit failed: {class_audit}")
    recovery_audit = audit_dagger1_recovery_labels(rows)
    if not recovery_audit["passed"]:
        raise RuntimeError(
            f"DAgger-1 recovery-label audit failed: {recovery_audit}"
        )
    training_eligible = [
        row for row in rows if row.get("production_label_eligible") is True
    ]
    targeted_coverage = targeted_state_coverage(training_eligible)
    if args.collection_pass == "training":
        output_rows = training_eligible
        if not output_rows:
            raise RuntimeError(
                "DAgger-1 reached no rank-one learner recovery states; "
                "refusing to write an empty production dataset"
            )
        collection_gate: dict[str, Any] = recommended_collection_gate(
            output_rows,
            target_min_rows=args.target_min_rows,
            target_max_rows=args.target_max_rows,
        )
        if args.require_recommended_target and not (
            collection_gate["passed"] and targeted_coverage["passed"]
        ):
            raise RuntimeError(
                "DAgger-1 recommended collection target failed: "
                f"collection={collection_gate}, coverage={targeted_coverage}"
            )
    else:
        if training_eligible:
            raise RuntimeError(
                "diagnostic beta=0 collection unexpectedly produced "
                "training-eligible rows"
            )
        output_rows = [
            row
            for row in rows
            if row.get("state_origin") == "learner_policy"
            and row.get("recovery_stratum") is not None
        ]
        if not output_rows:
            raise RuntimeError(
                "DAgger-1 diagnostic reached no learner recovery states"
            )
        collection_gate = {
            "applicable": False,
            "reason": "diagnostic_beta_zero_output_is_training_ineligible",
            "passed": False,
        }

    artifact_training_eligible = bool(
        args.collection_pass == "training"
        and beta_contract["passed"]
        and class_audit["passed"]
        and recovery_audit["passed"]
        and collection_gate["passed"]
        and targeted_coverage["passed"]
    )
    for row in output_rows:
        row["collection_training_eligible"] = artifact_training_eligible
        labels = row.get("labels")
        if isinstance(labels, dict):
            labels["collection_training_eligible"] = artifact_training_eligible

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output, output_rows)
    if args.all_output is not None:
        args.all_output.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(args.all_output, rows)
    manifest = {
        "schema_version": 1,
        "release_evidence_eligible": False,
        "training_eligible": artifact_training_eligible,
        "output_sha256": _file_sha256(args.output),
        "collector_contract": DAGGER1_OBSERVABLE_RECOVERY_SUPERVISION,
        "recovery_label_contract": "observable_rank_one_learner_state_v1",
        "input": str(args.input),
        "input_sha256": _file_sha256(args.input),
        "scenario_generator_report": str(args.scenario_generator_report),
        "scenario_generator_report_sha256": _file_sha256(
            args.scenario_generator_report
        ),
        "scenario_manifest": str(args.scenario_manifest),
        "scenario_manifest_sha256": _file_sha256(args.scenario_manifest),
        "scenario_builder_contract": DAGGER1_SCENARIO_BUILDER_CONTRACT,
        "source_partition": "train",
        "source_state": source_state,
        "factory_identities": {
            "environment": env_factory_binding,
            "learner_policy": policy_factory_binding,
            "expert_oracle": {
                "class": "psse_env.oracle.expert_policy:ExpertPolicyOracle",
                "source_sha256": file_sha256(expert_source),
            },
        },
        "release_environment_contract": {
            "parameter_ranking_dominance_threshold": (
                BC0_PARAMETER_RANKING_DOMINANCE_THRESHOLD
            ),
            "production_dataset_mode": True,
            "max_steps": args.max_steps,
        },
        "forbidden_suite": str(args.forbidden_suite),
        "forbidden_suite_sha256": _file_sha256(args.forbidden_suite),
        "evaluation_policy": str(DEFAULT_EVALUATION_POLICY),
        "evaluation_policy_sha256": _file_sha256(DEFAULT_EVALUATION_POLICY),
        "forbidden_physical_root_count": len(forbidden_roots),
        "frozen_evaluation_root_count": len(frozen_roots),
        "d0_physical_root_count": len(d0_roots),
        "d0_aggregate_dir": str(args.d0_aggregate_dir),
        "d0_raw_sha256": _file_sha256(d0_raw_path),
        "d0_generation_provenance_sha256": _file_sha256(d0_provenance_path),
        "model_id": args.model_id,
        "model_revision": args.model_revision.lower(),
        "iteration": args.iteration,
        "beta": args.beta,
        "beta_contract": beta_contract,
        "collection_pass": args.collection_pass,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "visited_rows": len(rows),
        "output_rows": len(output_rows),
        "production_eligible_recovery_rows": len(training_eligible),
        "diagnostic_training_ineligible_recovery_rows": (
            len(output_rows) if args.collection_pass == "diagnostic" else 0
        ),
        "eligible_recovery_strata": dict(
            sorted(
                Counter(
                    str(row.get("recovery_stratum"))
                    for row in training_eligible
                ).items()
            )
        ),
        "eligible_physical_root_count": len(
            {
                str(row.get("physical_root_fingerprint"))
                for row in training_eligible
                if row.get("physical_root_fingerprint")
            }
        ),
        "recommended_collection_gate": collection_gate,
        "targeted_state_coverage": targeted_coverage,
        "class_audit": class_audit,
        "recovery_label_audit": recovery_audit,
    }
    output_manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({**manifest, "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
