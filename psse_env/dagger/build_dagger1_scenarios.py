from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from psse_env.dagger.collect_dagger1 import (
    DAGGER1_SCENARIO_BUILDER_CONTRACT,
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
from psse_env.sft.provenance import file_sha256, git_source_state


DEFAULT_DAGGER1_ROOT_PLAN = {
    "measurement+parameter": 60,
    "multi_measurement": 60,
    "parameter": 30,
}


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
    if not (
        isinstance(d0_provenance, Mapping)
        and d0_provenance.get("release_eligible") is True
        and isinstance(d0_source, Mapping)
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
    candidate_plan = {
        family: int(count) * int(candidate_multiplier)
        for family, count in sorted(plan.items())
    }
    raw_scenarios = generator.build(candidate_plan)
    candidate_envelopes = [
        partition_release_scenario_v1(scenario, split="dagger_train")
        for scenario in raw_scenarios
    ]
    candidates_by_family: dict[str, list[dict[str, Any]]] = {
        family: [] for family in plan
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

    envelopes: list[dict[str, Any]] = []
    selected_roots: set[str] = set()
    selected_counts: dict[str, int] = {}
    for family, requested_count in sorted(plan.items()):
        selected: list[dict[str, Any]] = []
        candidates = sorted(
            candidates_by_family.get(family, []),
            key=lambda row: (
                str(row["grouping"]["physical_root_fingerprint"]),
                str(row["execution"].get("scenario_id") or ""),
            ),
        )
        for envelope in candidates:
            root = str(envelope["grouping"]["physical_root_fingerprint"])
            if root in selected_roots:
                continue
            selected.append(envelope)
            selected_roots.add(root)
            if len(selected) == requested_count:
                break
        selected_counts[family] = len(selected)
        envelopes.extend(selected)
    shortfalls = {
        family: {
            "requested": int(plan[family]),
            "selected": int(selected_counts.get(family, 0)),
        }
        for family in sorted(plan)
        if selected_counts.get(family, 0) != int(plan[family])
    }
    if shortfalls:
        raise RuntimeError(
            "DAgger-1 fresh-root candidate pool did not satisfy the exact plan: "
            + json.dumps(shortfalls, sort_keys=True)
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
    manifest = {
        "schema_version": 1,
        "builder_contract": DAGGER1_SCENARIO_BUILDER_CONTRACT,
        "source_state": source_state,
        "seed": int(seed),
        "plan": dict(sorted(plan.items())),
        "candidate_multiplier": int(candidate_multiplier),
        "candidate_plan": candidate_plan,
        "candidate_count": len(candidate_envelopes),
        "selected_count_by_family": selected_counts,
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
        help="Deterministically over-generate before protected-root filtering",
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
