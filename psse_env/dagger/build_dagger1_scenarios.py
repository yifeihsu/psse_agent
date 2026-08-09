from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from psse_env.dagger.collect_dagger1 import (
    DAGGER1_BASE_RESERVE_PLAN,
    DAGGER1_COLLECTION_SCHEDULE_CONTRACT,
    DAGGER1_MAXIMUM_ROLLOUT_REPLICAS_BY_FAMILY,
    DAGGER1_PREDECESSOR_SOURCE_COMMIT,
    DAGGER1_PREDECESSOR_TRAINING_ROOT_SET_SHA256,
    DAGGER1_RESERVE_FAMILY_PRIORITY,
    DAGGER1_SCENARIO_BUILDER_CONTRACT,
    DAGGER1_TOPUP_RESERVE_PLAN,
    DAGGER1_TOPUP_SUBCOHORT,
    DEFAULT_EVALUATION_POLICY,
    DEFAULT_FORBIDDEN_SUITE,
    frozen_physical_roots,
    validate_training_scenarios,
)
from psse_env.dagger.dataset_builder import load_jsonl
from psse_env.dagger.release_factories import (
    BC0_PARAMETER_RANKING_DOMINANCE_THRESHOLD,
)
from psse_env.dagger.suite_builder import partition_release_scenario_v1
from psse_env.providers.scenario_generator import Round0ScenarioGenerator
from psse_env.sft.provenance import (
    file_sha256,
    git_source_state,
    stable_json_sha256,
)


DEFAULT_DAGGER1_ROOT_PLAN = {
    # This is the reviewed primary collection cohort.  The finite reserve
    # cohort and the independent 12/12/6 development holdout are declared
    # separately below so neither can silently change this primary balance.
    "measurement+parameter": 48,
    "multi_measurement": 48,
    "parameter": 24,
}

# The default builder deliberately asks for more train-partition candidates than
# it publishes.  This creates a finite reserve collection pool while leaving a
# named, root-disjoint multi-measurement allocation for development.
DEFAULT_DAGGER1_CANDIDATE_REQUEST_PLAN = {
    "measurement+parameter": 108,
    "multi_measurement": 176,
    "parameter": 48,
}
DEFAULT_DAGGER1_MULTI_MEASUREMENT_PRIMARY_QUOTA = {
    2: 16,
    3: 6,
    4: 10,
    5: 16,
}
DEFAULT_DAGGER1_MULTI_MEASUREMENT_DEVELOPMENT_QUOTA = {
    2: 3,
    3: 3,
    4: 3,
    5: 3,
}
DEFAULT_DAGGER1_TRAINING_POOL_PLAN = {
    "measurement+parameter": 108,
    "multi_measurement": 79,
    "parameter": 24,
}


def _scenario_sort_key(envelope: Mapping[str, Any]) -> tuple[str, str]:
    grouping = envelope["grouping"]
    execution = envelope["execution"]
    return (
        str(grouping["physical_root_fingerprint"]),
        str(execution.get("scenario_id") or ""),
    )


def _error_cardinality(envelope: Mapping[str, Any]) -> int:
    grouping = envelope["grouping"]
    cardinality = grouping.get("error_cardinality")
    if (
        isinstance(cardinality, bool)
        or not isinstance(cardinality, int)
        or cardinality <= 0
    ):
        raise RuntimeError("candidate scenario has invalid error_cardinality")
    return cardinality


def _cardinality_inventory(
    envelopes: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    inventory: dict[str, int] = {}
    for envelope in envelopes:
        key = str(_error_cardinality(envelope))
        inventory[key] = inventory.get(key, 0) + 1
    return dict(sorted(inventory.items(), key=lambda item: int(item[0])))


def _root_set_sha256(envelopes: Sequence[Mapping[str, Any]]) -> str:
    roots = sorted(
        str(envelope["grouping"]["physical_root_fingerprint"])
        for envelope in envelopes
    )
    return stable_json_sha256(roots)


def _load_plan(value: str | None) -> dict[str, int]:
    if value is None:
        payload: Any = DEFAULT_DAGGER1_ROOT_PLAN
    else:
        path = Path(value)
        payload = json.loads(path.read_text()) if path.is_file() else json.loads(value)
    if not isinstance(payload, Mapping) or not payload:
        raise ValueError("DAgger-1 scenario plan must be a non-empty mapping")
    plan: dict[str, int] = {}
    for family, count in payload.items():
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("DAgger-1 scenario plan counts must be positive integers")
        plan[str(family)] = count
    return dict(sorted(plan.items()))


def build_dagger1_scenarios(
    *,
    d0_aggregate_dir: Path,
    output: Path,
    generator_report_path: Path,
    seed: int,
    plan: Mapping[str, int],
    candidate_multiplier: int = 2,
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    source_state = git_source_state(repo_root)
    if source_state.get("release_eligible_source") is not True:
        raise RuntimeError("DAgger-1 scenario generation requires a clean source tree")
    if (
        isinstance(candidate_multiplier, bool)
        or not isinstance(candidate_multiplier, int)
        or candidate_multiplier < 1
    ):
        raise ValueError("candidate_multiplier must be a positive integer")
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    builder_paths = (output, generator_report_path, manifest_path)
    if len({path.resolve() for path in builder_paths}) != len(builder_paths):
        raise ValueError("DAgger-1 scenario output/report/manifest must be distinct")
    if output.exists() or generator_report_path.exists() or manifest_path.exists():
        raise FileExistsError(
            "DAgger-1 scenario output/report/manifest already exists"
        )
    d0_raw_path = d0_aggregate_dir / "aggregate.raw.jsonl"
    d0_provenance_path = d0_aggregate_dir / "aggregate.generation_provenance.json"
    if not d0_raw_path.is_file() or not d0_provenance_path.is_file():
        raise FileNotFoundError("D0 aggregate raw rows/provenance are missing")
    d0_provenance = json.loads(d0_provenance_path.read_text(encoding="utf-8"))
    descriptor = (
        d0_provenance.get("generation_descriptor")
        if isinstance(d0_provenance, Mapping)
        else None
    )
    d0_source = (
        descriptor.get("source_state")
        if isinstance(descriptor, Mapping)
        else None
    )
    if not isinstance(descriptor, Mapping):
        raise RuntimeError("D0 aggregate generation descriptor is missing")
    expected_provenance_id = stable_json_sha256(descriptor)
    if d0_provenance.get("generation_provenance_id") != expected_provenance_id:
        raise RuntimeError(
            "D0 aggregate generation provenance ID does not match descriptor"
        )
    if not (
        isinstance(d0_source, Mapping)
        and d0_source.get("release_eligible_source") is True
    ):
        raise RuntimeError("D0 aggregate source state is not release eligible")
    if not (
        isinstance(d0_provenance, Mapping)
        and d0_provenance.get("release_eligible") is True
        and d0_source.get("source_commit") == source_state.get("source_commit")
    ):
        raise RuntimeError("D0 aggregate is not release eligible for current source")
    d0_hashes = d0_provenance.get("dataset_hashes")
    d0_hashes = d0_hashes if isinstance(d0_hashes, Mapping) else {}
    if d0_hashes.get(d0_raw_path.name) != file_sha256(d0_raw_path):
        raise RuntimeError("D0 aggregate raw bytes do not match provenance")
    d0_rows = load_jsonl(d0_raw_path)
    d0_roots = {
        str(row.get("physical_root_fingerprint"))
        for row in d0_rows
        if row.get("physical_root_fingerprint")
    }
    frozen_roots = set(frozen_physical_roots(DEFAULT_FORBIDDEN_SUITE))
    protected_roots = d0_roots | frozen_roots

    generator = Round0ScenarioGenerator(
        seed=int(seed),
        source_partition="train",
        parameter_ranking_dominance_threshold=(
            BC0_PARAMETER_RANKING_DOMINANCE_THRESHOLD
        ),
    )
    normalized_plan = {
        str(family): int(count) for family, count in sorted(plan.items())
    }
    default_pool = normalized_plan == DEFAULT_DAGGER1_ROOT_PLAN
    if default_pool and candidate_multiplier != 2:
        raise ValueError(
            "the reviewed default DAgger-1 plan uses the explicit "
            "108/176/48 candidate request; candidate_multiplier must remain 2"
        )
    candidate_plan = (
        dict(DEFAULT_DAGGER1_CANDIDATE_REQUEST_PLAN)
        if default_pool
        else {
            family: int(count) * int(candidate_multiplier)
            for family, count in normalized_plan.items()
        }
    )
    raw_scenarios = generator.build(candidate_plan)
    candidate_envelopes = [
        partition_release_scenario_v1(scenario, split="dagger_train")
        for scenario in raw_scenarios
    ]
    candidates_by_family: dict[str, list[dict[str, Any]]] = {
        family: [] for family in normalized_plan
    }
    filtered_protected_roots: set[str] = set()
    filtered_parameter_scan_roots: set[str] = set()
    for envelope in candidate_envelopes:
        grouping = envelope["grouping"]
        family = str(grouping["scenario_family"])
        if family not in candidates_by_family:
            continue
        root = str(grouping["physical_root_fingerprint"])
        execution_metadata = envelope["execution"].get("metadata")
        execution_metadata = (
            execution_metadata if isinstance(execution_metadata, Mapping) else {}
        )
        parameter_scans = execution_metadata.get("parameter_scans")
        parameter_scans = (
            parameter_scans if isinstance(parameter_scans, Mapping) else {}
        )
        z_scans = parameter_scans.get("z_scans")
        parameter_scans_available = bool(
            isinstance(z_scans, Sequence)
            and not isinstance(z_scans, (str, bytes))
            and len(z_scans) > 0
        )
        grouping["parameter_scans_available"] = parameter_scans_available
        if root in protected_roots:
            filtered_protected_roots.add(root)
            continue
        if family == "multi_measurement" and parameter_scans_available:
            filtered_parameter_scan_roots.add(root)
            continue
        candidates_by_family[family].append(envelope)

    # Treat physical-root collisions as a builder error rather than silently
    # allowing the candidate ordering to decide which family owns a root.
    fresh_candidates_by_family: dict[str, list[dict[str, Any]]] = {}
    candidate_root_family: dict[str, str] = {}
    for family in sorted(normalized_plan):
        family_candidates: list[dict[str, Any]] = []
        family_roots: set[str] = set()
        for envelope in sorted(
            candidates_by_family.get(family, []), key=_scenario_sort_key
        ):
            root = str(envelope["grouping"]["physical_root_fingerprint"])
            if root in family_roots:
                raise RuntimeError(
                    f"DAgger-1 candidate pool repeats physical root {root!r}"
                )
            other_family = candidate_root_family.get(root)
            if other_family is not None:
                raise RuntimeError(
                    "DAgger-1 candidate physical root crosses families: "
                    f"{root!r} ({other_family!r}, {family!r})"
                )
            family_roots.add(root)
            candidate_root_family[root] = family
            family_candidates.append(envelope)
        fresh_candidates_by_family[family] = family_candidates

    primary_by_family: dict[str, list[dict[str, Any]]] = {
        family: [] for family in normalized_plan
    }
    reserve_by_family: dict[str, list[dict[str, Any]]] = {
        family: [] for family in normalized_plan
    }
    topup_by_family: dict[str, list[dict[str, Any]]] = {
        family: [] for family in normalized_plan
    }
    development_reserved_by_family: dict[str, list[dict[str, Any]]] = {
        family: [] for family in normalized_plan
    }
    if default_pool:
        for family in ("measurement+parameter", "parameter"):
            requested = normalized_plan[family]
            primary_by_family[family] = fresh_candidates_by_family[family][
                :requested
            ]

        multi_candidates = fresh_candidates_by_family["multi_measurement"]
        multi_by_cardinality: dict[int, list[dict[str, Any]]] = {}
        for envelope in multi_candidates:
            cardinality = _error_cardinality(envelope)
            multi_by_cardinality.setdefault(cardinality, []).append(envelope)
        for cardinality in multi_by_cardinality:
            multi_by_cardinality[cardinality].sort(key=_scenario_sort_key)

        multi_primary: list[dict[str, Any]] = []
        multi_development: list[dict[str, Any]] = []
        multi_reserve_candidates: list[dict[str, Any]] = []
        quota_shortfalls: dict[str, dict[str, int]] = {}
        supported_cardinalities = set(
            DEFAULT_DAGGER1_MULTI_MEASUREMENT_PRIMARY_QUOTA
        )
        unexpected_cardinalities = sorted(
            set(multi_by_cardinality) - supported_cardinalities
        )
        if unexpected_cardinalities:
            raise RuntimeError(
                "default DAgger-1 multi-measurement pool has unsupported "
                f"cardinalities: {unexpected_cardinalities}"
            )
        for cardinality, primary_quota in sorted(
            DEFAULT_DAGGER1_MULTI_MEASUREMENT_PRIMARY_QUOTA.items()
        ):
            candidates = multi_by_cardinality.get(cardinality, [])
            development_quota = (
                DEFAULT_DAGGER1_MULTI_MEASUREMENT_DEVELOPMENT_QUOTA[
                    cardinality
                ]
            )
            required = primary_quota + development_quota
            if len(candidates) < required:
                quota_shortfalls[str(cardinality)] = {
                    "required": required,
                    "available": len(candidates),
                }
                continue
            multi_primary.extend(candidates[:primary_quota])
            multi_development.extend(
                candidates[primary_quota : primary_quota + development_quota]
            )
            multi_reserve_candidates.extend(
                candidates[primary_quota + development_quota :]
            )
        if quota_shortfalls:
            raise RuntimeError(
                "DAgger-1 fresh multi-measurement inventory cannot satisfy "
                "primary plus development quotas: "
                + json.dumps(quota_shortfalls, sort_keys=True)
            )
        primary_by_family["multi_measurement"] = sorted(
            multi_primary, key=_scenario_sort_key
        )
        development_reserved_by_family["multi_measurement"] = sorted(
            multi_development, key=_scenario_sort_key
        )
        multi_reserve_count = (
            DEFAULT_DAGGER1_TRAINING_POOL_PLAN["multi_measurement"]
            - normalized_plan["multi_measurement"]
        )
        reserve_by_family["multi_measurement"] = sorted(
            multi_reserve_candidates, key=_scenario_sort_key
        )[:multi_reserve_count]

        mixed_reserve_count = (
            DEFAULT_DAGGER1_TRAINING_POOL_PLAN["measurement+parameter"]
            - normalized_plan["measurement+parameter"]
        )
        reserve_by_family["measurement+parameter"] = (
            fresh_candidates_by_family["measurement+parameter"][
                normalized_plan["measurement+parameter"] :
                normalized_plan["measurement+parameter"]
                + mixed_reserve_count
            ]
        )
        topup_start = (
            normalized_plan["measurement+parameter"]
            + DAGGER1_BASE_RESERVE_PLAN["measurement+parameter"]
        )
        topup_stop = (
            topup_start
            + DAGGER1_TOPUP_RESERVE_PLAN["measurement+parameter"]
        )
        topup_by_family["measurement+parameter"] = (
            fresh_candidates_by_family["measurement+parameter"][
                topup_start:topup_stop
            ]
        )
    else:
        for family, requested_count in normalized_plan.items():
            primary_by_family[family] = fresh_candidates_by_family[family][
                :requested_count
            ]

    primary_counts = {
        family: len(primary_by_family[family]) for family in normalized_plan
    }
    primary_shortfalls = {
        family: {
            "requested": normalized_plan[family],
            "selected": primary_counts[family],
        }
        for family in normalized_plan
        if primary_counts[family] != normalized_plan[family]
    }
    if primary_shortfalls:
        raise RuntimeError(
            "DAgger-1 fresh-root candidate pool did not satisfy the exact "
            "primary plan: " + json.dumps(primary_shortfalls, sort_keys=True)
        )

    reserve_counts = {
        family: len(reserve_by_family[family]) for family in normalized_plan
    }
    topup_counts = {
        family: len(topup_by_family[family]) for family in normalized_plan
    }
    if default_pool and topup_counts != DAGGER1_TOPUP_RESERVE_PLAN:
        raise RuntimeError(
            "DAgger-1 fresh-root top-up does not satisfy the reviewed plan: "
            + json.dumps(topup_counts, sort_keys=True)
        )
    if default_pool:
        pool_shortfalls = {
            family: {
                "requested": DEFAULT_DAGGER1_TRAINING_POOL_PLAN[family],
                "selected": primary_counts[family] + reserve_counts[family],
            }
            for family in normalized_plan
            if primary_counts[family] + reserve_counts[family]
            != DEFAULT_DAGGER1_TRAINING_POOL_PLAN[family]
        }
        if pool_shortfalls:
            raise RuntimeError(
                "DAgger-1 fresh-root candidate pool did not satisfy the exact "
                "primary plus reserve plan: "
                + json.dumps(pool_shortfalls, sort_keys=True)
            )

    primary = sorted(
        (
            envelope
            for family in sorted(primary_by_family)
            for envelope in primary_by_family[family]
        ),
        key=lambda row: (
            str(row["grouping"]["scenario_family"]),
            *_scenario_sort_key(row),
        ),
    )
    reserve = sorted(
        (
            envelope
            for family in reserve_by_family
            for envelope in reserve_by_family[family]
        ),
        key=lambda row: (
            DAGGER1_RESERVE_FAMILY_PRIORITY.index(
                str(row["grouping"]["scenario_family"])
            ),
            *_scenario_sort_key(row),
        ),
    )
    envelopes = primary + reserve
    topup_roots = {
        str(row["grouping"]["physical_root_fingerprint"])
        for rows in topup_by_family.values()
        for row in rows
    }
    for collection_order, envelope in enumerate(envelopes):
        grouping = envelope["grouping"]
        family = str(grouping["scenario_family"])
        cohort = "primary" if collection_order < len(primary) else "reserve"
        grouping["collection_cohort"] = cohort
        grouping["collection_subcohort"] = (
            DAGGER1_TOPUP_SUBCOHORT
            if str(grouping["physical_root_fingerprint"]) in topup_roots
            else ("primary" if cohort == "primary" else "base_reserve")
        )
        grouping["collection_priority"] = (
            0
            if cohort == "primary"
            else DAGGER1_RESERVE_FAMILY_PRIORITY.index(family) + 1
        )
        grouping["collection_order"] = collection_order

    selected_counts = {
        family: primary_counts[family] + reserve_counts[family]
        for family in normalized_plan
    }
    selected_roots = {
        str(row["grouping"]["physical_root_fingerprint"])
        for row in envelopes
    }
    development_reserved_roots = {
        str(row["grouping"]["physical_root_fingerprint"])
        for rows in development_reserved_by_family.values()
        for row in rows
    }
    if selected_roots & development_reserved_roots:
        raise RuntimeError(
            "DAgger-1 development-reserved roots entered the training pool"
        )
    topup_development_overlap = sorted(
        topup_roots & development_reserved_roots
    )
    if topup_development_overlap:
        raise RuntimeError(
            "DAgger-1 fresh-root top-up consumed development-reserved roots"
        )
    base_training = [
        row
        for row in envelopes
        if row["grouping"]["collection_subcohort"]
        != DAGGER1_TOPUP_SUBCOHORT
    ]
    predecessor_root_hash = _root_set_sha256(base_training)
    if (
        default_pool
        and predecessor_root_hash
        != DAGGER1_PREDECESSOR_TRAINING_ROOT_SET_SHA256
    ):
        raise RuntimeError(
            "DAgger-1 base training roots no longer reproduce the frozen "
            f"{DAGGER1_PREDECESSOR_SOURCE_COMMIT[:12]} predecessor allocation"
        )
    predecessor_roots = {
        str(row["grouping"]["physical_root_fingerprint"])
        for row in base_training
    }
    topup_predecessor_overlap = sorted(topup_roots & predecessor_roots)
    if topup_predecessor_overlap:
        raise RuntimeError(
            "DAgger-1 fresh-root top-up reuses predecessor training roots"
        )
    generated_roots = {
        str(row["grouping"]["physical_root_fingerprint"]) for row in envelopes
    }
    overlap = sorted(generated_roots & protected_roots)
    if overlap:
        raise RuntimeError(
            "fresh DAgger-1 generation reused protected roots; choose a new "
            f"seed/plan: {overlap[:8]}"
        )
    validate_training_scenarios(
        envelopes, forbidden_roots=frozenset(protected_roots)
    )
    report = generator.report()
    source_partition = report.get("source_partition")
    if not (
        isinstance(source_partition, Mapping)
        and source_partition.get("enabled") is True
        and source_partition.get("selected") == "train"
    ):
        raise RuntimeError("scenario generator did not attest train partition")
    output.parent.mkdir(parents=True, exist_ok=True)
    generator_report_path.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(envelopes, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    generator_report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    fresh_candidate_inventory = {
        family: {
            "physical_root_count": len(fresh_candidates_by_family[family]),
            "error_cardinality": _cardinality_inventory(
                fresh_candidates_by_family[family]
            ),
            "physical_root_set_sha256": _root_set_sha256(
                fresh_candidates_by_family[family]
            ),
        }
        for family in normalized_plan
    }
    development_reserved_roots_by_family = {
        family: sorted(
            str(row["grouping"]["physical_root_fingerprint"])
            for row in development_reserved_by_family[family]
        )
        for family in normalized_plan
    }
    development_reserved_root_set_sha256_by_family = {
        family: stable_json_sha256(roots)
        for family, roots in development_reserved_roots_by_family.items()
    }
    topup_roots_by_family = {
        family: sorted(
            str(row["grouping"]["physical_root_fingerprint"])
            for row in topup_by_family[family]
        )
        for family in normalized_plan
    }
    topup_root_set_sha256_by_family = {
        family: stable_json_sha256(roots)
        for family, roots in topup_roots_by_family.items()
    }
    selected_roots_by_family = {
        family: {
            str(row["grouping"]["physical_root_fingerprint"])
            for row in primary_by_family[family] + reserve_by_family[family]
        }
        for family in normalized_plan
    }
    unused_fresh_candidate_count_by_family = {
        family: len(
            {
                str(row["grouping"]["physical_root_fingerprint"])
                for row in fresh_candidates_by_family[family]
            }
            - selected_roots_by_family[family]
            - set(development_reserved_roots_by_family[family])
        )
        for family in normalized_plan
    }
    collection_schedule = {
        "contract": DAGGER1_COLLECTION_SCHEDULE_CONTRACT,
        "cohort_order": ["primary", "reserve"],
        "reserve_family_priority": list(
            DAGGER1_RESERVE_FAMILY_PRIORITY
        ),
        "priority_field": "grouping.collection_priority",
        "order_field": "grouping.collection_order",
        "subcohort_field": "grouping.collection_subcohort",
        "reserve_subcohort_order": ["base_reserve", DAGGER1_TOPUP_SUBCOHORT],
        "maximum_rollout_replicas_by_family": dict(
            DAGGER1_MAXIMUM_ROLLOUT_REPLICAS_BY_FAMILY
        ),
    }
    manifest = {
        "schema_version": 1,
        "builder_contract": DAGGER1_SCENARIO_BUILDER_CONTRACT,
        "source_state": source_state,
        "seed": int(seed),
        # ``plan`` is retained as the v2-compatible name for the primary plan.
        "plan": normalized_plan,
        "primary_plan": normalized_plan,
        "primary_count_by_family": primary_counts,
        "primary_multi_measurement_cardinality_quota": {
            str(key): value
            for key, value in sorted(
                DEFAULT_DAGGER1_MULTI_MEASUREMENT_PRIMARY_QUOTA.items()
            )
        }
        if default_pool
        else {},
        "primary_multi_measurement_cardinality_count": (
            _cardinality_inventory(primary_by_family["multi_measurement"])
            if "multi_measurement" in primary_by_family
            else {}
        ),
        "reserve_count_by_family": reserve_counts,
        "base_reserve_plan": (
            dict(DAGGER1_BASE_RESERVE_PLAN) if default_pool else reserve_counts
        ),
        "topup_reserve_plan": (
            dict(DAGGER1_TOPUP_RESERVE_PLAN) if default_pool else topup_counts
        ),
        "topup_reserve_count_by_family": topup_counts,
        "topup_reserve_roots_by_family": topup_roots_by_family,
        "topup_reserve_root_set_sha256_by_family": (
            topup_root_set_sha256_by_family
        ),
        "topup_reserve_physical_root_set_sha256": stable_json_sha256(
            sorted(topup_roots)
        ),
        "predecessor_source_commit": DAGGER1_PREDECESSOR_SOURCE_COMMIT,
        "predecessor_training_root_count": len(predecessor_roots),
        "predecessor_training_root_set_sha256": predecessor_root_hash,
        "topup_predecessor_overlap": topup_predecessor_overlap,
        "topup_development_reserved_overlap": topup_development_overlap,
        "reserve_multi_measurement_cardinality_inventory": (
            _cardinality_inventory(reserve_by_family["multi_measurement"])
            if "multi_measurement" in reserve_by_family
            else {}
        ),
        "candidate_multiplier": int(candidate_multiplier),
        "candidate_request_plan": candidate_plan,
        "candidate_plan": candidate_plan,
        "candidate_count": len(candidate_envelopes),
        "fresh_candidate_count": sum(
            len(rows) for rows in fresh_candidates_by_family.values()
        ),
        "fresh_candidate_inventory": fresh_candidate_inventory,
        "selected_count_by_family": selected_counts,
        "training_pool_plan": (
            dict(DEFAULT_DAGGER1_TRAINING_POOL_PLAN)
            if default_pool
            else selected_counts
        ),
        "unused_fresh_candidate_count_by_family": (
            unused_fresh_candidate_count_by_family
        ),
        "development_reserved_roots_by_family": (
            development_reserved_roots_by_family
        ),
        "development_reserved_root_set_sha256_by_family": (
            development_reserved_root_set_sha256_by_family
        ),
        "withheld_for_development_count_by_family": {
            family: len(roots)
            for family, roots in development_reserved_roots_by_family.items()
        },
        "withheld_for_development_multi_measurement_cardinality_inventory": (
            _cardinality_inventory(
                development_reserved_by_family.get("multi_measurement", [])
            )
        ),
        "collection_schedule": collection_schedule,
        "primary_physical_root_set_sha256": _root_set_sha256(primary),
        "reserve_physical_root_set_sha256": _root_set_sha256(reserve),
        "training_physical_root_set_sha256": _root_set_sha256(envelopes),
        "filtered_protected_root_count": len(filtered_protected_roots),
        "filtered_multi_measurement_with_parameter_scans_root_count": len(
            filtered_parameter_scan_roots
        ),
        "scenario_count": len(envelopes),
        "physical_root_count": len(generated_roots),
        "source_partition": "train",
        "parameter_ranking_dominance_threshold": (
            BC0_PARAMETER_RANKING_DOMINANCE_THRESHOLD
        ),
        "d0_root_count": len(d0_roots),
        "frozen_root_count": len(frozen_roots),
        "protected_root_overlap": [],
        "output_sha256": file_sha256(output),
        "generator_report_sha256": file_sha256(generator_report_path),
        "d0_raw_sha256": file_sha256(d0_raw_path),
        "d0_generation_provenance_sha256": file_sha256(d0_provenance_path),
        "frozen_suite_sha256": file_sha256(DEFAULT_FORBIDDEN_SUITE),
        "evaluation_policy_sha256": file_sha256(DEFAULT_EVALUATION_POLICY),
        "release_evidence_eligible": False,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate fresh train-partition DAgger-1 scenario envelopes"
    )
    parser.add_argument("--d0-aggregate-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generator-report", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument(
        "--candidate-multiplier",
        type=int,
        default=2,
        help=(
            "Scale custom plans before protected-root filtering; the reviewed "
            "default uses explicit 108/176/48 requests and requires value 2"
        ),
    )
    parser.add_argument(
        "--plan",
        help="JSON object/path; defaults to the targeted DAgger-1 root plan",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    manifest = build_dagger1_scenarios(
        d0_aggregate_dir=args.d0_aggregate_dir,
        output=args.output,
        generator_report_path=args.generator_report,
        seed=args.seed,
        plan=_load_plan(args.plan),
        candidate_multiplier=args.candidate_multiplier,
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
