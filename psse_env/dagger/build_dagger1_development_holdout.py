"""Build a fresh, diagnostic-only DAgger-1 development holdout.

The development roots are generated from the same physical ``train`` source
partition as DAgger-1 collection, but they are an independent exclusion set:
they may be used for closed-loop model selection and must never enter SFT or
count as promotion/release evidence.
"""

from __future__ import annotations

import argparse
import inspect
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from psse_env.dagger.dataset_builder import load_jsonl
from psse_env.dagger.suite_builder import partition_release_scenario_v1
from psse_env.providers.scenario_generator import Round0ScenarioGenerator
from psse_env.sft.provenance import (
    AGGREGATE_MANIFEST_FILENAME,
    file_sha256,
    git_source_state,
    stable_json_sha256,
    validate_aggregate_manifest_binding,
)


DAGGER1_DEVELOPMENT_HOLDOUT_CONTRACT = (
    "fresh_train_partition_dagger1_development_holdout_v3"
)
DAGGER1_DEVELOPMENT_SUITE_NAME = "dagger1_development"
DAGGER1_DEVELOPMENT_SPLIT = "dagger_development"
DAGGER1_DEVELOPMENT_PARAMETER_RANKING_THRESHOLD = 1.0
DAGGER1_TRAINING_SCENARIO_BUILDER_CONTRACT = (
    "fresh_train_partition_dagger1_scenarios_v4"
)
DEFAULT_DAGGER1_DEVELOPMENT_MULTI_MEASUREMENT_CANDIDATE_REQUEST = 176
DEFAULT_DAGGER1_DEVELOPMENT_MULTI_MEASUREMENT_CARDINALITY_QUOTA = {
    2: 3,
    3: 3,
    4: 3,
    5: 3,
}
EXPECTED_FROZEN_PHYSICAL_ROOT_COUNT = 115
DEFAULT_FORBIDDEN_SUITE = (
    Path(__file__).resolve().parent / "suites" / "bc0_eval_suite_v1.json"
)
DEFAULT_EVALUATION_POLICY = (
    Path(__file__).resolve().parent / "bc0_evaluation_policy.json"
)
DEFAULT_DAGGER1_DEVELOPMENT_ROOT_PLAN = {
    "measurement+parameter": 12,
    "multi_measurement": 12,
    "parameter": 6,
}
APPROVED_DAGGER1_DEVELOPMENT_ROOT_COUNT = sum(
    DEFAULT_DAGGER1_DEVELOPMENT_ROOT_PLAN.values()
)
REQUIRED_POST_EVALUATION_RECOVERY_STRATA = (
    "multi_measurement_safe_handoff",
    "post_failure_no_candidate",
    "premature_commit_recovery",
    "premature_escalation_recovery",
    "sequential_measurement_parameter_recovery",
    "unsupported_correction_recovery",
)


def _load_plan(value: str | None) -> dict[str, int]:
    if value is None:
        payload: Any = DEFAULT_DAGGER1_DEVELOPMENT_ROOT_PLAN
    else:
        path = Path(value)
        payload = json.loads(path.read_text()) if path.is_file() else json.loads(value)
    if not isinstance(payload, Mapping) or not payload:
        raise ValueError("development holdout plan must be a non-empty mapping")
    plan: dict[str, int] = {}
    for family, count in payload.items():
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("development holdout counts must be positive integers")
        plan[str(family)] = count
    return dict(sorted(plan.items()))


def _diagnostic_model_selection_eligible(
    plan: Mapping[str, int],
    *,
    physical_root_count: int,
) -> bool:
    """Return whether a holdout satisfies the reviewed 30-root plan."""

    return (
        dict(sorted(plan.items()))
        == dict(sorted(DEFAULT_DAGGER1_DEVELOPMENT_ROOT_PLAN.items()))
        and physical_root_count == APPROVED_DAGGER1_DEVELOPMENT_ROOT_COUNT
    )


def _load_scenario_list(path: Path, *, label: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    text = path.read_text(encoding="utf-8")
    payload: Any
    if path.suffix.lower() == ".jsonl":
        payload = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        payload = json.loads(text)
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"{label} must be a non-empty JSON/JSONL list")
    if not all(isinstance(row, Mapping) for row in payload):
        raise ValueError(f"{label} must contain only scenario objects")
    return [dict(row) for row in payload]


def _scenario_root(row: Mapping[str, Any], *, label: str) -> str:
    if row.get("scenario_schema_version") != 1:
        raise ValueError(f"{label} is not a schema-v1 scenario")
    execution = row.get("execution")
    audit = row.get("audit")
    grouping = row.get("grouping")
    if not all(isinstance(item, Mapping) for item in (execution, audit, grouping)):
        raise ValueError(f"{label} has a malformed execution/audit/grouping envelope")
    root = str(grouping.get("physical_root_fingerprint") or "").strip()
    if not root:
        raise ValueError(f"{label} has no physical root fingerprint")
    return root


def _frozen_physical_roots(path: Path) -> frozenset[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or not payload:
        raise ValueError("frozen evaluation suite must be a non-empty mapping")
    roots: set[str] = set()
    for suite, rows in payload.items():
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"frozen suite {suite!r} must be a non-empty list")
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise ValueError(f"frozen suite {suite!r}[{index}] is not an object")
            roots.add(_scenario_root(row, label=f"frozen suite {suite!r}[{index}]"))
    if len(roots) != EXPECTED_FROZEN_PHYSICAL_ROOT_COUNT:
        raise RuntimeError(
            "frozen evaluation root count changed: "
            f"observed {len(roots)}, required {EXPECTED_FROZEN_PHYSICAL_ROOT_COUNT}"
        )
    return frozenset(roots)


def _require_policy_pinned_frozen_suite(path: Path) -> str:
    """Return the suite digest only when it matches the repository policy."""

    if not path.is_file():
        raise FileNotFoundError(f"frozen evaluation suite is missing: {path}")
    if not DEFAULT_EVALUATION_POLICY.is_file():
        raise FileNotFoundError(
            f"evaluation policy is missing: {DEFAULT_EVALUATION_POLICY}"
        )
    policy = json.loads(DEFAULT_EVALUATION_POLICY.read_text(encoding="utf-8"))
    suite_policy = policy.get("suite_policy") if isinstance(policy, Mapping) else None
    suite_sha256 = file_sha256(path)
    if not (
        isinstance(suite_policy, Mapping)
        and suite_policy.get("status") == "pinned"
        and suite_policy.get("approved_suite_sha256") == suite_sha256
    ):
        raise RuntimeError(
            "development holdout frozen suite does not match the pinned "
            "evaluation policy"
        )
    return suite_sha256


def _require_current_d0(
    d0_aggregate_dir: Path,
    *,
    source_state: Mapping[str, Any],
) -> tuple[Path, Path, Path, set[str]]:
    raw_path = d0_aggregate_dir / "aggregate.raw.jsonl"
    provenance_path = d0_aggregate_dir / "aggregate.generation_provenance.json"
    manifest_path = d0_aggregate_dir / AGGREGATE_MANIFEST_FILENAME
    if not (
        raw_path.is_file()
        and provenance_path.is_file()
        and manifest_path.is_file()
    ):
        raise FileNotFoundError(
            "D0 aggregate raw rows/provenance/manifest are missing"
        )
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    descriptor = (
        provenance.get("generation_descriptor")
        if isinstance(provenance, Mapping)
        else None
    )
    d0_source = descriptor.get("source_state") if isinstance(descriptor, Mapping) else None
    dataset_hashes = provenance.get("dataset_hashes") if isinstance(provenance, Mapping) else None
    if not (
        isinstance(provenance, Mapping)
        and provenance.get("release_eligible") is True
        and isinstance(descriptor, Mapping)
        and provenance.get("generation_provenance_id")
        == stable_json_sha256(descriptor)
        and isinstance(d0_source, Mapping)
        and d0_source.get("release_eligible_source") is True
        and d0_source.get("source_commit") == source_state.get("source_commit")
        and isinstance(dataset_hashes, Mapping)
        and dataset_hashes.get(raw_path.name) == file_sha256(raw_path)
    ):
        raise RuntimeError("D0 aggregate is not byte-bound to the clean current source")
    manifest_binding = validate_aggregate_manifest_binding(
        provenance,
        aggregate_dir=d0_aggregate_dir,
    )
    if manifest_binding["passed"] is not True:
        raise RuntimeError(
            "D0 aggregate manifest is not byte-bound to provenance: "
            + "; ".join(manifest_binding["failures"])
        )
    rows = load_jsonl(raw_path)
    roots = {
        str(row.get("physical_root_fingerprint") or "").strip()
        for row in rows
        if str(row.get("physical_root_fingerprint") or "").strip()
    }
    if not roots:
        raise ValueError("D0 aggregate has no physical roots")
    return raw_path, provenance_path, manifest_path, roots


def _require_current_training_boundary(
    scenarios_path: Path,
    manifest_path: Path,
    *,
    source_state: Mapping[str, Any],
    d0_raw_path: Path,
    d0_provenance_path: Path,
    frozen_suite_path: Path,
) -> tuple[list[dict[str, Any]], set[str], dict[str, tuple[str, ...]]]:
    scenarios = _load_scenario_list(
        scenarios_path,
        label="D1 training scenarios",
    )
    if not manifest_path.is_file():
        raise FileNotFoundError(f"D1 training scenario manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_source = manifest.get("source_state") if isinstance(manifest, Mapping) else None
    roots = {
        _scenario_root(row, label=f"D1 training scenario[{index}]")
        for index, row in enumerate(scenarios)
    }
    if len(roots) != len(scenarios):
        raise ValueError("D1 training scenarios must have globally unique physical roots")
    reserved_payload = (
        manifest.get("development_reserved_roots_by_family")
        if isinstance(manifest, Mapping)
        else None
    )
    reserved_hashes = (
        manifest.get("development_reserved_root_set_sha256_by_family")
        if isinstance(manifest, Mapping)
        else None
    )
    reserved_counts = (
        manifest.get("withheld_for_development_count_by_family")
        if isinstance(manifest, Mapping)
        else None
    )
    if not all(
        isinstance(value, Mapping)
        for value in (reserved_payload, reserved_hashes, reserved_counts)
    ):
        raise RuntimeError(
            "D1 training scenario manifest lacks the reviewed development "
            "root reservation"
        )
    development_reserved_roots: dict[str, tuple[str, ...]] = {}
    for family, raw_roots in reserved_payload.items():
        if not (
            isinstance(raw_roots, Sequence)
            and not isinstance(raw_roots, (str, bytes))
        ):
            raise RuntimeError(
                "D1 training development root reservation is malformed"
            )
        family_roots = tuple(sorted(str(root).strip() for root in raw_roots))
        if (
            any(not root for root in family_roots)
            or len(set(family_roots)) != len(family_roots)
            or reserved_counts.get(family) != len(family_roots)
            or reserved_hashes.get(family)
            != stable_json_sha256(list(family_roots))
        ):
            raise RuntimeError(
                "D1 training development root reservation is not count/hash bound"
            )
        development_reserved_roots[str(family)] = family_roots
    all_reserved_roots = {
        root
        for family_roots in development_reserved_roots.values()
        for root in family_roots
    }
    if len(all_reserved_roots) != sum(
        len(family_roots)
        for family_roots in development_reserved_roots.values()
    ):
        raise RuntimeError(
            "D1 training development root reservation repeats roots across families"
        )
    if all_reserved_roots & roots:
        raise RuntimeError(
            "D1 training scenarios consumed development-reserved roots"
        )
    checks = {
        "schema_version": isinstance(manifest, Mapping)
        and manifest.get("schema_version") == 1,
        "builder_contract": isinstance(manifest, Mapping)
        and manifest.get("builder_contract")
        == DAGGER1_TRAINING_SCENARIO_BUILDER_CONTRACT,
        "release_evidence_eligible": isinstance(manifest, Mapping)
        and manifest.get("release_evidence_eligible") is False,
        "source_partition": isinstance(manifest, Mapping)
        and manifest.get("source_partition") == "train",
        "parameter_threshold": isinstance(manifest, Mapping)
        and manifest.get("parameter_ranking_dominance_threshold")
        == DAGGER1_DEVELOPMENT_PARAMETER_RANKING_THRESHOLD,
        "source_commit": isinstance(manifest_source, Mapping)
        and manifest_source.get("release_eligible_source") is True
        and manifest_source.get("source_commit") == source_state.get("source_commit"),
        "scenario_bytes": isinstance(manifest, Mapping)
        and manifest.get("output_sha256") == file_sha256(scenarios_path),
        "scenario_count": isinstance(manifest, Mapping)
        and manifest.get("scenario_count") == len(scenarios),
        "physical_root_count": isinstance(manifest, Mapping)
        and manifest.get("physical_root_count") == len(roots),
        "protected_root_overlap": isinstance(manifest, Mapping)
        and manifest.get("protected_root_overlap") == [],
        "d0_raw": isinstance(manifest, Mapping)
        and manifest.get("d0_raw_sha256") == file_sha256(d0_raw_path),
        "d0_provenance": isinstance(manifest, Mapping)
        and manifest.get("d0_generation_provenance_sha256")
        == file_sha256(d0_provenance_path),
        "frozen_suite": isinstance(manifest, Mapping)
        and manifest.get("frozen_suite_sha256") == file_sha256(frozen_suite_path),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise RuntimeError(
            "D1 training scenario boundary is not current/byte-bound: "
            + ", ".join(failed)
        )
    return scenarios, roots, development_reserved_roots


def _source_bindings(repo_root: Path) -> dict[str, str]:
    sources = [
        Path(__file__),
        Path(inspect.getsourcefile(Round0ScenarioGenerator) or ""),
        Path(inspect.getsourcefile(partition_release_scenario_v1) or ""),
    ]
    if any(not path.is_file() for path in sources):
        raise RuntimeError("development holdout source bindings are not inspectable")
    return {
        str(path.resolve().relative_to(repo_root.resolve())): file_sha256(path)
        for path in sorted({path.resolve() for path in sources})
    }


def build_dagger1_development_holdout(
    *,
    d0_aggregate_dir: Path,
    d1_training_scenarios: Path,
    d1_training_manifest: Path,
    output: Path,
    generator_report_path: Path,
    seed: int,
    plan: Mapping[str, int],
    candidate_multiplier: int = 4,
    forbidden_suite_path: Path = DEFAULT_FORBIDDEN_SUITE,
) -> dict[str, Any]:
    """Generate an independent D1 development suite and provenance manifest."""

    repo_root = Path(__file__).resolve().parents[2]
    source_state = git_source_state(repo_root)
    if source_state.get("release_eligible_source") is not True:
        raise RuntimeError(
            "DAgger-1 development holdout generation requires a clean source tree"
        )
    if (
        isinstance(candidate_multiplier, bool)
        or not isinstance(candidate_multiplier, int)
        or candidate_multiplier < 1
    ):
        raise ValueError("candidate_multiplier must be a positive integer")
    normalized_plan = _load_plan(json.dumps(dict(plan)))
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    builder_paths = (output, generator_report_path, manifest_path)
    if len({path.resolve() for path in builder_paths}) != len(builder_paths):
        raise ValueError("development output/report/manifest paths must be distinct")
    if any(path.exists() for path in builder_paths):
        raise FileExistsError("development output/report/manifest already exists")

    (
        d0_raw_path,
        d0_provenance_path,
        d0_manifest_path,
        d0_roots,
    ) = _require_current_d0(
        d0_aggregate_dir,
        source_state=source_state,
    )
    frozen_suite_sha256 = _require_policy_pinned_frozen_suite(
        forbidden_suite_path
    )
    frozen_roots = set(_frozen_physical_roots(forbidden_suite_path))
    _, training_roots, training_development_reserved_roots = (
        _require_current_training_boundary(
            d1_training_scenarios,
            d1_training_manifest,
            source_state=source_state,
            d0_raw_path=d0_raw_path,
            d0_provenance_path=d0_provenance_path,
            frozen_suite_path=forbidden_suite_path,
        )
    )
    pairwise_input_overlap = {
        "d0_frozen": sorted(d0_roots & frozen_roots),
        "d0_d1_training": sorted(d0_roots & training_roots),
        "frozen_d1_training": sorted(frozen_roots & training_roots),
    }
    if any(pairwise_input_overlap.values()):
        raise RuntimeError(
            "existing D0/frozen/D1-training boundaries overlap: "
            + json.dumps(pairwise_input_overlap, sort_keys=True)
        )
    all_training_development_reserved_roots = {
        root
        for family_roots in training_development_reserved_roots.values()
        for root in family_roots
    }
    reserved_boundary_overlap = {
        "d0": sorted(all_training_development_reserved_roots & d0_roots),
        "frozen": sorted(
            all_training_development_reserved_roots & frozen_roots
        ),
        "d1_training": sorted(
            all_training_development_reserved_roots & training_roots
        ),
    }
    if any(reserved_boundary_overlap.values()):
        raise RuntimeError(
            "training-manifest development reservation overlaps a protected "
            "boundary: "
            + json.dumps(reserved_boundary_overlap, sort_keys=True)
        )
    protected_roots = d0_roots | frozen_roots | training_roots

    generator = Round0ScenarioGenerator(
        seed=int(seed),
        source_partition="train",
        parameter_ranking_dominance_threshold=(
            DAGGER1_DEVELOPMENT_PARAMETER_RANKING_THRESHOLD
        ),
    )
    default_plan = normalized_plan == DEFAULT_DAGGER1_DEVELOPMENT_ROOT_PLAN
    candidate_plan = {
        family: count * candidate_multiplier
        for family, count in sorted(normalized_plan.items())
    }
    if default_plan:
        candidate_plan["multi_measurement"] = (
            DEFAULT_DAGGER1_DEVELOPMENT_MULTI_MEASUREMENT_CANDIDATE_REQUEST
        )
    candidates = [
        partition_release_scenario_v1(
            scenario,
            split=DAGGER1_DEVELOPMENT_SPLIT,
        )
        for scenario in generator.build(candidate_plan)
    ]
    candidates_by_family: dict[str, list[dict[str, Any]]] = {
        family: [] for family in normalized_plan
    }
    filtered_protected: set[str] = set()
    filtered_multi_measurement_with_scans: set[str] = set()
    for index, envelope in enumerate(candidates):
        root = _scenario_root(envelope, label=f"development candidate[{index}]")
        grouping = envelope["grouping"]
        family = str(grouping.get("scenario_family") or "")
        if family not in candidates_by_family:
            continue
        metadata = envelope["execution"].get("metadata")
        metadata = metadata if isinstance(metadata, Mapping) else {}
        parameter_scans = metadata.get("parameter_scans")
        parameter_scans = parameter_scans if isinstance(parameter_scans, Mapping) else {}
        z_scans = parameter_scans.get("z_scans")
        scans_available = bool(
            isinstance(z_scans, Sequence)
            and not isinstance(z_scans, (str, bytes))
            and len(z_scans) > 0
        )
        # This is a builder-only selection predicate.  The partition helper
        # has already produced and validated the strict release-evaluation
        # envelope, so do not mutate its grouping schema with this derived
        # inventory field after validation.
        if root in protected_roots:
            filtered_protected.add(root)
            continue
        if family == "multi_measurement" and scans_available:
            filtered_multi_measurement_with_scans.add(root)
            continue
        candidates_by_family[family].append(envelope)

    selected: list[dict[str, Any]] = []
    selected_roots: set[str] = set()
    selected_counts: Counter[str] = Counter()
    for family, count in sorted(normalized_plan.items()):
        ordered = sorted(
            candidates_by_family[family],
            key=lambda row: (
                str(row["grouping"]["physical_root_fingerprint"]),
                str(row["execution"].get("scenario_id") or ""),
            ),
        )
        if family == "multi_measurement" and default_plan:
            reserved_multi_roots = set(
                training_development_reserved_roots.get(
                    "multi_measurement", ()
                )
            )
            if len(reserved_multi_roots) != count:
                raise RuntimeError(
                    "training manifest does not reserve the exact reviewed "
                    f"{count}-root multi-measurement development allocation"
                )
            candidate_by_root: dict[str, dict[str, Any]] = {}
            repeated_candidate_roots: set[str] = set()
            for envelope in ordered:
                root = str(
                    envelope["grouping"]["physical_root_fingerprint"]
                )
                if root in candidate_by_root:
                    repeated_candidate_roots.add(root)
                candidate_by_root[root] = envelope
            if repeated_candidate_roots:
                raise RuntimeError(
                    "development multi-measurement candidate pool repeats "
                    f"physical roots: {sorted(repeated_candidate_roots)[:8]}"
                )
            missing_reserved_roots = sorted(
                reserved_multi_roots - set(candidate_by_root)
            )
            if missing_reserved_roots:
                raise RuntimeError(
                    "full multi-measurement development enumeration does not "
                    "reconstruct the training-manifest reservation: "
                    f"{missing_reserved_roots[:8]}"
                )
            ordered = [
                candidate_by_root[root] for root in sorted(reserved_multi_roots)
            ]
        for envelope in ordered:
            root = str(envelope["grouping"]["physical_root_fingerprint"])
            if root in selected_roots:
                continue
            selected.append(envelope)
            selected_roots.add(root)
            selected_counts[family] += 1
            if selected_counts[family] == count:
                break
    shortfalls = {
        family: {"requested": count, "selected": selected_counts[family]}
        for family, count in sorted(normalized_plan.items())
        if selected_counts[family] != count
    }
    if shortfalls:
        raise RuntimeError(
            "fresh development candidate pool did not satisfy the exact plan: "
            + json.dumps(shortfalls, sort_keys=True)
        )
    selected_multi = [
        row
        for row in selected
        if row["grouping"]["scenario_family"] == "multi_measurement"
    ]
    selected_multi_cardinality = Counter(
        str(row["grouping"].get("error_cardinality"))
        for row in selected_multi
    )
    if default_plan and selected_multi_cardinality != Counter(
        {
            str(key): value
            for key, value in (
                DEFAULT_DAGGER1_DEVELOPMENT_MULTI_MEASUREMENT_CARDINALITY_QUOTA.items()
            )
        }
    ):
        raise RuntimeError(
            "reserved multi-measurement development roots do not satisfy the "
            "reviewed error-cardinality quota"
        )
    selected.sort(
        key=lambda row: (
            str(row["grouping"]["scenario_family"]),
            str(row["grouping"]["physical_root_fingerprint"]),
        )
    )
    output_payload = {DAGGER1_DEVELOPMENT_SUITE_NAME: selected}
    # The holdout is consumed directly by the diagnostic release evaluator.
    # Enforce that integration boundary before any bytes are published instead
    # of relying on the earlier per-row partition validation followed by local
    # selection logic.
    from psse_env.dagger.evaluator import validate_release_scenario_suites

    validate_release_scenario_suites(
        output_payload,
        allow_diagnostic_development=True,
    )
    report = generator.report()
    source_partition = report.get("source_partition")
    parameter_admission = report.get("parameter_ranking_admission")
    if not (
        isinstance(source_partition, Mapping)
        and source_partition.get("enabled") is True
        and source_partition.get("selected") == "train"
        and isinstance(parameter_admission, Mapping)
        and parameter_admission.get("enforced") is True
        and parameter_admission.get("threshold")
        == DAGGER1_DEVELOPMENT_PARAMETER_RANKING_THRESHOLD
    ):
        raise RuntimeError(
            "scenario generator did not attest the reviewed train/threshold contract"
        )
    development_overlap = {
        "d0": sorted(selected_roots & d0_roots),
        "frozen": sorted(selected_roots & frozen_roots),
        "d1_training": sorted(selected_roots & training_roots),
    }
    if any(development_overlap.values()):
        raise RuntimeError(
            "development roots overlap a protected boundary: "
            + json.dumps(development_overlap, sort_keys=True)
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    generator_report_path.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(output_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    generator_report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    fresh_candidate_inventory = {}
    for family, family_candidates in sorted(candidates_by_family.items()):
        family_roots = sorted(
            str(row["grouping"]["physical_root_fingerprint"])
            for row in family_candidates
        )
        cardinality = Counter(
            str(row["grouping"].get("error_cardinality"))
            for row in family_candidates
        )
        fresh_candidate_inventory[family] = {
            "physical_root_count": len(set(family_roots)),
            "error_cardinality": dict(
                sorted(cardinality.items(), key=lambda item: int(item[0]))
            ),
            "physical_root_set_sha256": stable_json_sha256(
                sorted(set(family_roots))
            ),
        }
    selected_multi_roots = sorted(
        str(row["grouping"]["physical_root_fingerprint"])
        for row in selected_multi
    )
    reserved_multi_roots = list(
        training_development_reserved_roots.get("multi_measurement", ())
    )
    manifest = {
        "schema_version": 1,
        "scenario_schema_version": 1,
        "artifact_type": "dagger1_development_holdout_suite",
        "builder_contract": DAGGER1_DEVELOPMENT_HOLDOUT_CONTRACT,
        "source_state": source_state,
        "source_bindings": _source_bindings(repo_root),
        "suite_name": DAGGER1_DEVELOPMENT_SUITE_NAME,
        "suite_format": "evaluation_suite_mapping_v1",
        "split": DAGGER1_DEVELOPMENT_SPLIT,
        "source_partition": "train",
        "parameter_ranking_dominance_threshold": (
            DAGGER1_DEVELOPMENT_PARAMETER_RANKING_THRESHOLD
        ),
        "seed": int(seed),
        "plan": normalized_plan,
        "candidate_multiplier": int(candidate_multiplier),
        "candidate_request_plan": candidate_plan,
        "candidate_plan": candidate_plan,
        "candidate_count": len(candidates),
        "fresh_candidate_inventory": fresh_candidate_inventory,
        "selected_count_by_family": dict(sorted(selected_counts.items())),
        "selected_multi_measurement_cardinality_inventory": dict(
            sorted(
                selected_multi_cardinality.items(),
                key=lambda item: int(item[0]),
            )
        ),
        "training_development_reserved_roots_by_family": {
            family: list(roots)
            for family, roots in sorted(
                training_development_reserved_roots.items()
            )
        },
        "training_development_reserved_multi_measurement_root_set_sha256": (
            stable_json_sha256(reserved_multi_roots)
        ),
        "selected_multi_measurement_root_set_sha256": stable_json_sha256(
            selected_multi_roots
        ),
        "selected_multi_measurement_matches_training_reservation": (
            selected_multi_roots == reserved_multi_roots
        ),
        "scenario_count": len(selected),
        "physical_root_count": len(selected_roots),
        "filtered_protected_root_count": len(filtered_protected),
        "filtered_multi_measurement_with_parameter_scans_root_count": len(
            filtered_multi_measurement_with_scans
        ),
        "training_eligible": False,
        "training_collection_eligible": False,
        "release_evidence_eligible": False,
        "promotion_evidence_eligible": False,
        "diagnostic_closed_loop_model_selection_eligible": (
            _diagnostic_model_selection_eligible(
                normalized_plan,
                physical_root_count=len(selected_roots),
            )
        ),
        "recovery_stratum_qualified_model_selection_eligible": False,
        "intended_use": "dagger1_closed_loop_development_model_selection_only",
        "required_post_evaluation_recovery_strata": list(
            REQUIRED_POST_EVALUATION_RECOVERY_STRATA
        ),
        "recovery_strata_coverage_requires_closed_loop_evaluation": True,
        "recovery_strata_qualification_status": (
            "pending_teacher_opportunity_trace_instrumentation"
        ),
        "root_counts": {
            "d0": len(d0_roots),
            "frozen": len(frozen_roots),
            "d1_training": len(training_roots),
            "development": len(selected_roots),
        },
        "root_set_sha256": {
            "d0": stable_json_sha256(sorted(d0_roots)),
            "frozen": stable_json_sha256(sorted(frozen_roots)),
            "d1_training": stable_json_sha256(sorted(training_roots)),
            "development": stable_json_sha256(sorted(selected_roots)),
        },
        "pairwise_input_overlap": pairwise_input_overlap,
        "training_development_reserved_boundary_overlap": (
            reserved_boundary_overlap
        ),
        "development_protected_overlap": development_overlap,
        "output_sha256": file_sha256(output),
        "generator_report_sha256": file_sha256(generator_report_path),
        "d0_raw_sha256": file_sha256(d0_raw_path),
        "d0_generation_provenance_sha256": file_sha256(d0_provenance_path),
        "d0_manifest_sha256": file_sha256(d0_manifest_path),
        "d1_training_scenarios_sha256": file_sha256(d1_training_scenarios),
        "d1_training_manifest_sha256": file_sha256(d1_training_manifest),
        "frozen_suite_sha256": frozen_suite_sha256,
        "evaluation_policy_sha256": file_sha256(DEFAULT_EVALUATION_POLICY),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a fresh 30-root DAgger-1 development suite that is "
            "ineligible for training and release evidence"
        )
    )
    parser.add_argument("--d0-aggregate-dir", type=Path, required=True)
    parser.add_argument("--d1-training-scenarios", type=Path, required=True)
    parser.add_argument("--d1-training-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generator-report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--candidate-multiplier", type=int, default=4)
    parser.add_argument(
        "--plan",
        help="JSON object/path; defaults to 12 mixed, 12 multi-measurement, 6 parameter roots",
    )
    parser.add_argument(
        "--forbidden-suite",
        type=Path,
        default=DEFAULT_FORBIDDEN_SUITE,
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    manifest = build_dagger1_development_holdout(
        d0_aggregate_dir=args.d0_aggregate_dir,
        d1_training_scenarios=args.d1_training_scenarios,
        d1_training_manifest=args.d1_training_manifest,
        output=args.output,
        generator_report_path=args.generator_report,
        seed=args.seed,
        plan=_load_plan(args.plan),
        candidate_multiplier=args.candidate_multiplier,
        forbidden_suite_path=args.forbidden_suite,
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
