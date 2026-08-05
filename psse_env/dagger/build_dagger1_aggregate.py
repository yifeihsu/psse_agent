from __future__ import annotations

import argparse
import importlib
import inspect
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import psse_env.dagger.dataset_builder as dataset_builder_module
import psse_env.dagger.replay_buffer as replay_buffer_module
from psse_env.dagger.dataset_builder import examples_to_chat_sft, load_jsonl, write_jsonl
from psse_env.dagger.collect_dagger1 import (
    DAGGER1_SCENARIO_BUILDER_CONTRACT,
    DEFAULT_ENV_FACTORY_SPEC,
    DEFAULT_EVALUATION_POLICY,
    DEFAULT_FORBIDDEN_SUITE,
    DEFAULT_POLICY_FACTORY_SPEC,
    frozen_physical_roots,
    validate_export_rows_truth_free,
)
from psse_env.dagger.release_factories import (
    BC0_PARAMETER_RANKING_DOMINANCE_THRESHOLD,
)
from psse_env.dagger.replay_buffer import (
    DEFAULT_MINIMUM_TOOL_CATEGORY_DISTINCT_ROOTS,
    DEFAULT_MINIMUM_TOOL_CATEGORY_NATURAL_ROWS,
    build_dagger1_training_view,
)
from psse_env.examples.generate_round0_aggregate import (
    BC0_CRITICAL_TARGET_TOOL_MINIMUM_DISTINCT_ROOTS,
    BC0_CRITICAL_TARGET_TOOL_SCENARIO_FAMILY_MINIMUM_DISTINCT_ROOTS,
    BC0_CRITICAL_TARGET_TOOL_STATE_CLASS_MINIMUM_DISTINCT_ROOTS,
    BC0_SAME_ROOT_PREREQUISITE_RULES,
)
from psse_env.sft.provenance import (
    file_sha256,
    git_source_state,
    stable_json_sha256,
    tool_schema_hashes,
)
from psse_env.oracle.expert_policy import ExpertPolicyOracle


def _load_mapping(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return dict(value)


def _verify_recorded_hash(
    *, provenance: Mapping[str, Any], path: Path, label: str
) -> None:
    hashes = provenance.get("dataset_hashes")
    hashes = hashes if isinstance(hashes, Mapping) else {}
    expected = hashes.get(path.name)
    if not expected or expected != file_sha256(path):
        raise ValueError(f"{label} hash does not match its D0 provenance")


def _bind_generation_id(row: Mapping[str, Any], provenance_id: str) -> dict[str, Any]:
    bound = dict(row)
    bound["generation_provenance_id"] = provenance_id
    metadata = bound.get("metadata")
    if isinstance(metadata, Mapping):
        bound["metadata"] = {**metadata, "generation_provenance_id": provenance_id}
    return bound


def _source_hash_for_import_spec(spec: str) -> str:
    module_name, separator, attribute_path = spec.partition(":")
    if not separator:
        raise ValueError(f"invalid import spec {spec!r}")
    value: Any = importlib.import_module(module_name)
    for part in attribute_path.split("."):
        value = getattr(value, part)
    source = inspect.getsourcefile(value)
    if source is None:
        raise ValueError(f"no source file for {spec}")
    return file_sha256(source)


def build_round1_aggregate(
    *,
    d0_aggregate_dir: Path,
    d1_path: Path,
    d1_manifest_path: Path,
    output_dir: Path,
    seed: int,
    size: int | None,
    d1_share: float,
    minimum_d1_share: float,
    maximum_d1_share: float,
    max_duplicate_count: int,
    max_rows_per_root: int,
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    source_state = git_source_state(repo_root)
    if source_state.get("release_eligible_source") is not True:
        raise RuntimeError("Round-1 aggregate requires a clean committed source tree")

    d0_raw_path = d0_aggregate_dir / "aggregate.raw.jsonl"
    validation_path = d0_aggregate_dir / "aggregate.validation.jsonl"
    test_path = d0_aggregate_dir / "aggregate.test.jsonl"
    d0_provenance_path = d0_aggregate_dir / "aggregate.generation_provenance.json"
    for path in (d0_raw_path, validation_path, test_path, d0_provenance_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    d0_provenance = _load_mapping(d0_provenance_path)
    if d0_provenance.get("release_eligible") is not True:
        raise ValueError("D0 aggregate generation provenance is not release eligible")
    d0_descriptor = d0_provenance.get("generation_descriptor")
    d0_descriptor = d0_descriptor if isinstance(d0_descriptor, Mapping) else {}
    d0_source = d0_descriptor.get("source_state")
    d0_source = d0_source if isinstance(d0_source, Mapping) else {}
    if d0_source.get("source_commit") != source_state.get("source_commit"):
        raise ValueError("D0 aggregate source commit does not match current source")
    for path, label in (
        (d0_raw_path, "D0 raw aggregate"),
        (validation_path, "D0 validation split"),
        (test_path, "D0 test split"),
    ):
        _verify_recorded_hash(provenance=d0_provenance, path=path, label=label)

    d1_manifest = _load_mapping(d1_manifest_path)
    if d1_manifest.get("release_evidence_eligible") is not False:
        raise ValueError("D1 manifest must explicitly reject promotion-evidence use")
    if d1_manifest.get("training_eligible") is not True:
        raise ValueError("D1 collection manifest is not training eligible")
    if d1_manifest.get("output_sha256") != file_sha256(d1_path):
        raise ValueError("D1 rows do not match the collection manifest hash")
    if (
        d1_manifest.get("scenario_builder_contract")
        != DAGGER1_SCENARIO_BUILDER_CONTRACT
    ):
        raise ValueError("D1 collection lacks the reviewed scenario-builder binding")
    scenario_manifest_value = str(
        d1_manifest.get("scenario_manifest") or ""
    ).strip()
    if not scenario_manifest_value:
        raise ValueError("D1 collection does not identify its scenario manifest")
    scenario_manifest_path = Path(scenario_manifest_value)
    if not scenario_manifest_path.is_absolute():
        scenario_manifest_path = repo_root / scenario_manifest_path
    if (
        not scenario_manifest_path.is_file()
        or d1_manifest.get("scenario_manifest_sha256")
        != file_sha256(scenario_manifest_path)
    ):
        raise ValueError("D1 scenario-builder manifest hash is unavailable or changed")
    d1_source = d1_manifest.get("source_state")
    d1_source = d1_source if isinstance(d1_source, Mapping) else {}
    if (
        d1_source.get("release_eligible_source") is not True
        or d1_source.get("source_commit") != source_state.get("source_commit")
    ):
        raise ValueError("D1 collection source does not match current clean source")
    identities = d1_manifest.get("factory_identities")
    identities = identities if isinstance(identities, Mapping) else {}
    expected_factory_bindings = {
        "environment": (
            DEFAULT_ENV_FACTORY_SPEC,
            _source_hash_for_import_spec(DEFAULT_ENV_FACTORY_SPEC),
        ),
        "learner_policy": (
            DEFAULT_POLICY_FACTORY_SPEC,
            _source_hash_for_import_spec(DEFAULT_POLICY_FACTORY_SPEC),
        ),
    }
    for role, (spec, source_hash) in expected_factory_bindings.items():
        binding = identities.get(role)
        binding = binding if isinstance(binding, Mapping) else {}
        if (
            binding.get("import_spec") != spec
            or binding.get("source_sha256") != source_hash
        ):
            raise ValueError(f"D1 {role} factory identity does not match source")
    expert_binding = identities.get("expert_oracle")
    expert_binding = expert_binding if isinstance(expert_binding, Mapping) else {}
    expert_source = inspect.getsourcefile(ExpertPolicyOracle)
    if (
        expert_source is None
        or expert_binding.get("source_sha256") != file_sha256(expert_source)
    ):
        raise ValueError("D1 expert oracle identity does not match source")
    release_contract = d1_manifest.get("release_environment_contract")
    release_contract = (
        release_contract if isinstance(release_contract, Mapping) else {}
    )
    if (
        release_contract.get("parameter_ranking_dominance_threshold")
        != BC0_PARAMETER_RANKING_DOMINANCE_THRESHOLD
        or release_contract.get("production_dataset_mode") is not True
        or release_contract.get("max_steps") != 24
    ):
        raise ValueError("D1 release environment contract is not approved")
    evaluation_policy = _load_mapping(DEFAULT_EVALUATION_POLICY)
    suite_policy = evaluation_policy.get("suite_policy")
    suite_policy = suite_policy if isinstance(suite_policy, Mapping) else {}
    current_suite_hash = file_sha256(DEFAULT_FORBIDDEN_SUITE)
    if (
        suite_policy.get("status") != "pinned"
        or suite_policy.get("approved_suite_sha256") != current_suite_hash
        or d1_manifest.get("forbidden_suite_sha256") != current_suite_hash
        or d1_manifest.get("evaluation_policy_sha256")
        != file_sha256(DEFAULT_EVALUATION_POLICY)
    ):
        raise ValueError("D1 frozen-suite holdout binding is not pinned/current")

    raw_d0 = load_jsonl(d0_raw_path)
    d0_train = [
        row
        for row in raw_d0
        if row.get("dataset_split") == "train"
        and row.get("production_label_eligible") is True
    ]
    d1 = load_jsonl(d1_path)
    # Treat the collection manifest as an integrity binding, not as authority
    # for safety properties that are cheap to recompute.  A forged manifest
    # must not be able to reintroduce oracle truth or a frozen evaluation root
    # at the final D0+D1 ingestion boundary.
    validate_export_rows_truth_free(d1)
    all_d0_roots = {
        str(row.get("physical_root_fingerprint"))
        for row in raw_d0
        if row.get("physical_root_fingerprint")
    }
    d1_roots = {
        str(row.get("physical_root_fingerprint"))
        for row in d1
        if row.get("physical_root_fingerprint")
    }
    leaked = sorted(all_d0_roots & d1_roots)
    if leaked:
        raise ValueError(
            "D1 roots overlap the D0 aggregate: " + ", ".join(leaked)
        )
    frozen_roots = frozen_physical_roots(DEFAULT_FORBIDDEN_SUITE)
    frozen_leaked = sorted(frozen_roots & d1_roots)
    if frozen_leaked:
        raise ValueError(
            "D1 roots overlap the frozen evaluation suite: "
            + ", ".join(frozen_leaked)
        )

    raw_view, training_view_report = build_dagger1_training_view(
        d0_train,
        d1,
        size=size,
        seed=seed,
        d1_share=d1_share,
        minimum_d1_share=minimum_d1_share,
        maximum_d1_share=maximum_d1_share,
        max_duplicate_count=max_duplicate_count,
        max_rows_per_root=max_rows_per_root,
        d0_training_view_kwargs={
            "minimum_tool_category_natural_rows": (
                DEFAULT_MINIMUM_TOOL_CATEGORY_NATURAL_ROWS
            ),
            "minimum_tool_category_distinct_roots": (
                DEFAULT_MINIMUM_TOOL_CATEGORY_DISTINCT_ROOTS
            ),
            "target_tool_minimum_distinct_roots": (
                BC0_CRITICAL_TARGET_TOOL_MINIMUM_DISTINCT_ROOTS
            ),
            "target_tool_state_class_minimum_distinct_roots": (
                BC0_CRITICAL_TARGET_TOOL_STATE_CLASS_MINIMUM_DISTINCT_ROOTS
            ),
            "target_tool_scenario_family_minimum_distinct_roots": (
                BC0_CRITICAL_TARGET_TOOL_SCENARIO_FAMILY_MINIMUM_DISTINCT_ROOTS
            ),
            "same_root_prerequisite_rules": BC0_SAME_ROOT_PREREQUISITE_RULES,
        },
    )
    if training_view_report.get("release_ready") is not True:
        raise RuntimeError(
            f"D0 union D1 training-view gate failed: {training_view_report}"
        )

    validation_rows = load_jsonl(validation_path)
    test_rows = load_jsonl(test_path)
    tentative_train = examples_to_chat_sft(raw_view, protocol="canonical")
    schema_hashes = tool_schema_hashes(
        [*tentative_train, *validation_rows, *test_rows]
    )
    if len(schema_hashes) != 1:
        raise ValueError(f"D0/D1 tool schema mismatch: {schema_hashes}")

    source_files = (
        Path(__file__),
        Path(dataset_builder_module.__file__),
        Path(replay_buffer_module.__file__),
    )
    generation_descriptor = {
        "generation_provenance_version": 1,
        "builder_contract": "deterministic_d0_d1_balanced_union_v1",
        "source_state": source_state,
        "protocol": "canonical",
        "schema_registry_hash": schema_hashes[0],
        "generator_hashes": {
            str(path.resolve().relative_to(repo_root)): file_sha256(path)
            for path in source_files
        },
        "input_artifacts": {
            "d0_generation_provenance_id": d0_provenance.get(
                "generation_provenance_id"
            ),
            "d0_generation_provenance_sha256": file_sha256(d0_provenance_path),
            "d0_raw_sha256": file_sha256(d0_raw_path),
            "d0_validation_sha256": file_sha256(validation_path),
            "d0_test_sha256": file_sha256(test_path),
            "d1_rows_sha256": file_sha256(d1_path),
            "d1_manifest_sha256": file_sha256(d1_manifest_path),
        },
        "training_view_report_sha256": stable_json_sha256(training_view_report),
        "generation_config": {
            "seed": int(seed),
            "requested_size": size,
            "d1_share": float(d1_share),
            "minimum_d1_share": float(minimum_d1_share),
            "maximum_d1_share": float(maximum_d1_share),
            "max_duplicate_count": int(max_duplicate_count),
            "max_rows_per_root": int(max_rows_per_root),
        },
    }
    provenance_id = stable_json_sha256(generation_descriptor)
    for row in raw_view:
        row["generation_provenance_id"] = provenance_id
    train_rows = examples_to_chat_sft(raw_view, protocol="canonical")
    validation_rows = [
        _bind_generation_id(row, provenance_id) for row in validation_rows
    ]
    test_rows = [_bind_generation_id(row, provenance_id) for row in test_rows]

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "aggregate.train_view.raw.jsonl": output_dir
        / "aggregate.train_view.raw.jsonl",
        "aggregate.train_view.jsonl": output_dir / "aggregate.train_view.jsonl",
        "aggregate.validation.jsonl": output_dir / "aggregate.validation.jsonl",
        "aggregate.test.jsonl": output_dir / "aggregate.test.jsonl",
    }
    occupied = [str(path) for path in output_paths.values() if path.exists()]
    if occupied:
        raise FileExistsError(
            "Round-1 output files already exist: " + ", ".join(occupied)
        )
    write_jsonl(output_paths["aggregate.train_view.raw.jsonl"], raw_view)
    write_jsonl(output_paths["aggregate.train_view.jsonl"], train_rows)
    write_jsonl(output_paths["aggregate.validation.jsonl"], validation_rows)
    write_jsonl(output_paths["aggregate.test.jsonl"], test_rows)

    dataset_hashes = {
        name: file_sha256(path) for name, path in sorted(output_paths.items())
    }
    release_checks = {
        "current_clean_source": source_state.get("release_eligible_source") is True,
        "d0_current_source": d0_source.get("source_commit")
        == source_state.get("source_commit"),
        "d1_current_source": d1_source.get("source_commit")
        == source_state.get("source_commit"),
        "d1_training_eligible": d1_manifest.get("training_eligible") is True,
        "training_view_release_ready": training_view_report.get("release_ready")
        is True,
        "source_mix_passed": training_view_report.get("source_allocation", {}).get(
            "passed"
        )
        is True,
    }
    release_eligible = all(release_checks.values())
    if not release_eligible:
        raise RuntimeError(f"Round-1 aggregate release checks failed: {release_checks}")
    provenance = {
        **generation_descriptor,
        "generation_descriptor": generation_descriptor,
        "generation_provenance_id": provenance_id,
        "dataset_hashes": dataset_hashes,
        "release_checks": release_checks,
        "release_eligible": release_eligible,
        "release_failures": [],
    }
    provenance_path = output_dir / "aggregate.generation_provenance.json"
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    preflight = {
        "release_eligible": release_eligible,
        "release_checks": release_checks,
        "generation_provenance_id": provenance_id,
        "training_view": training_view_report,
        "split_rows": {
            "train_view": len(train_rows),
            "validation": len(validation_rows),
            "test": len(test_rows),
        },
        "d1_collection_manifest": d1_manifest,
    }
    preflight_path = output_dir / "aggregate.preflight.json"
    preflight_path.write_text(
        json.dumps(preflight, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    checksum_paths = [*output_paths.values(), provenance_path, preflight_path]
    (output_dir / "SHA256SUMS").write_text(
        "".join(
            f"{file_sha256(path)}  {path.name}\n"
            for path in sorted(checksum_paths)
        ),
        encoding="utf-8",
    )
    return preflight


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a provenance-valid D0 union D1 round-1 SFT aggregate"
    )
    parser.add_argument("--d0-aggregate-dir", type=Path, required=True)
    parser.add_argument("--d1", type=Path, required=True)
    parser.add_argument("--d1-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--size", type=int)
    parser.add_argument("--d1-share", type=float, default=0.25)
    parser.add_argument("--minimum-d1-share", type=float, default=0.20)
    parser.add_argument("--maximum-d1-share", type=float, default=0.30)
    parser.add_argument("--max-duplicate-count", type=int, default=2)
    parser.add_argument("--max-rows-per-root", type=int, default=8)
    args = parser.parse_args(list(argv) if argv is not None else None)
    d1_manifest = args.d1_manifest or args.d1.with_suffix(
        args.d1.suffix + ".manifest.json"
    )
    report = build_round1_aggregate(
        d0_aggregate_dir=args.d0_aggregate_dir,
        d1_path=args.d1,
        d1_manifest_path=d1_manifest,
        output_dir=args.output_dir,
        seed=args.seed,
        size=args.size,
        d1_share=args.d1_share,
        minimum_d1_share=args.minimum_d1_share,
        maximum_d1_share=args.maximum_d1_share,
        max_duplicate_count=args.max_duplicate_count,
        max_rows_per_root=args.max_rows_per_root,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
