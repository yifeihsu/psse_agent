from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path

import pytest

import psse_env.dagger.preliminary_dataset as preliminary
from psse_env.sft.provenance import file_sha256, stable_json_sha256


_COMMIT = "1" * 40
_GENERATION_ID = "2" * 64


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _candidate(root: str, index: int, *, stratum: str = "loop_escape") -> dict[str, object]:
    return {
        "example_id": f"d1-{index}",
        "physical_root_fingerprint": root,
        "production_label_eligible": True,
        "collection_role": "training",
        "state_origin": "learner_policy",
        "recovery_stratum": stratum,
        "scenario_family": "measurement",
        "source_tier": "fixture",
    }


def _failure_evidence(
    candidate_path: Path, candidate_count: int
) -> dict[str, object]:
    passed = {"passed": True}
    return {
        "artifact_type": preliminary.STRICT_FAILURE_ARTIFACT_TYPE,
        "artifact_schema_version": preliminary.STRICT_FAILURE_SCHEMA_VERSION,
        "collection_outcome": "strict_gate_failed",
        "collection_pass": "training",
        "strict_gate_requested": True,
        "strict_gate_evaluated": True,
        "strict_gate_passed": False,
        "analysis_only": False,
        "diagnostic_only": True,
        "production_outputs_published": False,
        "release_evidence_eligible": False,
        "training_eligible": False,
        "round1_aggregate_eligible": False,
        "failed_gate_names": [preliminary.EXPECTED_FAILED_GATE],
        "round1_replay_capacity": {"passed": False},
        "collection_stopping_report": {
            "contract": preliminary.DAGGER1_COLLECTION_SCHEDULE_CONTRACT,
            "workflow_terminal": True,
            "collection_pass": "training",
            "training_eligible": True,
            "passed": False,
            "planned_episode_count": 4,
            "executed_episode_count": 4,
            "planned_batch_ids": ["primary", "reserve"],
            "executed_batch_ids": ["primary", "reserve"],
            "unexecuted_batch_ids": [],
            "terminal_failure": None,
        },
        "class_audit": passed,
        "recovery_label_audit": passed,
        "independent_root_support": passed,
        "targeted_state_coverage": passed,
        "deterministic_collection_selection": passed,
        "recommended_collection_gate": passed,
        "rollout_disposition_matrix": passed,
        "offline_teacher_target_quarantine_summary": {
            "passed": True,
            "zero_truth_audit_quarantine": True,
            "quarantined_rows": 0,
            "invalid_or_missing_audit_rows": 0,
            "candidate_rows": candidate_count,
            "passed_rows": candidate_count,
        },
        "diagnostic_artifacts": {
            "candidate_recovery_rows": {
                "relative_path": "diagnostic.candidate_recovery_rows.jsonl",
                "sha256": file_sha256(candidate_path),
                "row_count": candidate_count,
            }
        },
        "candidate_recovery_row_count": candidate_count,
        "source_state": {
            "source_commit": _COMMIT,
            "release_eligible_source": True,
            "source_worktree_dirty": False,
        },
    }


def test_failure_bundle_rejects_tamper_and_any_second_failed_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate_path = tmp_path / "diagnostic.candidate_recovery_rows.jsonl"
    rows = [_candidate("root-a", 0), _candidate("root-b", 1)]
    _write_jsonl(candidate_path, rows)
    evidence = _failure_evidence(candidate_path, len(rows))
    monkeypatch.setattr(
        preliminary,
        "summarize_dagger1_offline_teacher_target_quarantine",
        lambda _rows: {
            "passed": True,
            "quarantined_rows": 0,
            "candidate_rows": len(rows),
        },
    )
    monkeypatch.setattr(
        preliminary,
        "audit_target_aware_state_classes",
        lambda _rows: {"passed": True},
    )
    report = preliminary.validate_strict_failure_bundle(
        evidence, candidate_path=candidate_path, candidate_rows=rows
    )
    assert report["passed"] is True

    second_failure = copy.deepcopy(evidence)
    second_failure["failed_gate_names"] = [
        preliminary.EXPECTED_FAILED_GATE,
        "label_audit",
    ]
    with pytest.raises(ValueError, match="sole failed gate"):
        preliminary.validate_strict_failure_bundle(
            second_failure, candidate_path=candidate_path, candidate_rows=rows
        )

    with candidate_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_candidate("root-c", 2)) + "\n")
    with pytest.raises(ValueError, match="hash/count"):
        preliminary.validate_strict_failure_bundle(
            evidence, candidate_path=candidate_path, candidate_rows=rows
        )


def test_d0_train_validation_hashes_are_bound_to_generation_provenance(
    tmp_path: Path,
) -> None:
    train_path = tmp_path / "aggregate.train_view.jsonl"
    validation_path = tmp_path / "aggregate.validation.jsonl"
    train_rows = [
        {
            "example_id": "d0-train",
            "physical_root_fingerprint": "d0-train-root",
            "generation_provenance_id": _GENERATION_ID,
        }
    ]
    validation_rows = [
        {
            "example_id": "d0-validation",
            "physical_root_fingerprint": "d0-validation-root",
            "generation_provenance_id": _GENERATION_ID,
        }
    ]
    _write_jsonl(train_path, train_rows)
    _write_jsonl(validation_path, validation_rows)
    provenance = {
        "generation_provenance_id": _GENERATION_ID,
        "release_eligible": True,
        "release_failures": [],
        "source_state": {
            "source_commit": _COMMIT,
            "release_eligible_source": True,
            "source_worktree_dirty": False,
        },
        "dataset_hashes": {
            "aggregate.train_view.jsonl": file_sha256(train_path),
            "aggregate.validation.jsonl": file_sha256(validation_path),
            "aggregate.raw.jsonl": "3" * 64,
            "aggregate.manifest.json": "4" * 64,
        },
    }
    provenance_path = tmp_path / "aggregate.generation_provenance.json"
    _write_json(provenance_path, provenance)
    evidence = {
        "d0_generation_provenance_sha256": file_sha256(provenance_path),
        "d0_raw_sha256": "3" * 64,
        "d0_manifest_sha256": "4" * 64,
    }
    assert preliminary.validate_d0_generation_binding(
        evidence,
        provenance_path=provenance_path,
        provenance=provenance,
        train_path=train_path,
        train_rows=train_rows,
        validation_path=validation_path,
        validation_rows=validation_rows,
    )["passed"] is True

    with train_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(train_rows[0]) + "\n")
    with pytest.raises(ValueError, match="provenance hash binding"):
        preliminary.validate_d0_generation_binding(
            evidence,
            provenance_path=provenance_path,
            provenance=provenance,
            train_path=train_path,
            train_rows=train_rows,
            validation_path=validation_path,
            validation_rows=validation_rows,
        )


def test_floor_aware_root_reservation_preserves_scarce_target_cell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    floor = preliminary.DAGGER1_TARGETED_STATE_CELL_MINIMUM_DISTINCT_ROOTS[
        "parameter_route_complete_negative"
    ]
    monkeypatch.setattr(
        preliminary,
        "_gated_selection_floors",
        lambda: {"targeted_state_cell:parameter_route_complete_negative": floor},
    )
    rows: list[dict[str, object]] = []
    for index in range(6):
        row = _candidate(f"scarce-{index}", index, stratum="rare")
        row["policy_observation"] = {
            "fresh_context_evidence": {
                "parameter": {"route_status": "complete_negative"}
            }
        }
        rows.append(row)
    rows.extend(
        _candidate(f"neutral-{index}", 10 + index, stratum="common")
        for index in range(4)
    )
    first, first_report = preliminary.reserve_evaluation_roots(
        rows, root_count=2, minimum_remaining_rows=4, seed=17
    )
    second, second_report = preliminary.reserve_evaluation_roots(
        list(reversed(rows)), root_count=2, minimum_remaining_rows=4, seed=17
    )
    assert first == second
    assert first_report == second_report
    assert (
        first_report["remaining_roots_by_gated_group"][
            "targeted_state_cell:parameter_route_complete_negative"
        ]
        >= floor
    )


def _minimal_build_inputs(tmp_path: Path) -> dict[str, Path]:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    candidates = [_candidate(f"d1-root-{index}", index) for index in range(8)]
    d0_train = [
        {
            "example_id": "d0-train",
            "physical_root_fingerprint": "d0-train-root",
            "generation_provenance_id": _GENERATION_ID,
            "production_label_eligible": True,
        }
    ]
    d0_validation = [
        {
            "example_id": "d0-validation",
            "physical_root_fingerprint": "d0-validation-root",
            "generation_provenance_id": _GENERATION_ID,
            "production_label_eligible": True,
        }
    ]
    candidate_path = inputs / "diagnostic.candidate_recovery_rows.jsonl"
    train_path = inputs / "aggregate.train_view.jsonl"
    validation_path = inputs / "aggregate.validation.jsonl"
    _write_jsonl(candidate_path, candidates)
    _write_jsonl(train_path, d0_train)
    _write_jsonl(validation_path, d0_validation)
    provenance_path = inputs / "aggregate.generation_provenance.json"
    _write_json(provenance_path, {"source_state": {"source_commit": _COMMIT}})
    failure_path = inputs / "failure_evidence.json"
    _write_json(
        failure_path,
        {
            "source_state": {"source_commit": _COMMIT},
            "d0_generation_provenance_sha256": file_sha256(provenance_path),
        },
    )
    return {
        "failure": failure_path,
        "candidates": candidate_path,
        "provenance": provenance_path,
        "train": train_path,
        "validation": validation_path,
    }


def _patch_build_admission(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        preliminary,
        "validate_strict_failure_bundle",
        lambda *_args, **_kwargs: {"passed": True},
    )
    monkeypatch.setattr(
        preliminary,
        "validate_d0_generation_binding",
        lambda *_args, **_kwargs: {"passed": True, "source_commit": _COMMIT},
    )
    monkeypatch.setattr(preliminary, "_gated_selection_floors", lambda: {})
    monkeypatch.setattr(
        preliminary,
        "_builder_source_attestation",
        lambda _root: {
            "contract": preliminary.PRELIMINARY_BUILDER_SOURCE_CONTRACT,
            "source_commit": _COMMIT,
            "tracked_files_match_head": True,
            "tracked_files": {
                "psse_env/dagger/preliminary_dataset.py": {
                    "git_blob_oid": "3" * 40,
                    "sha256": "4" * 64,
                    "size_bytes": 100,
                },
                "scripts/build_preliminary_dagger_dataset.py": {
                    "git_blob_oid": "5" * 40,
                    "sha256": "6" * 64,
                    "size_bytes": 50,
                },
            },
        },
    )

    def select(rows: list[dict[str, object]], **kwargs: object):
        count = int(kwargs["target_min_rows"])
        return [dict(row) for row in rows[:count]], {
            "contract": preliminary.DAGGER1_COLLECTION_SELECTION_CONTRACT,
            "passed": True,
            "selected_rows": count,
        }

    monkeypatch.setattr(preliminary, "select_dagger1_collection_rows", select)

    def export(rows: list[dict[str, object]]) -> list[dict[str, object]]:
        result = []
        for source in rows:
            row = copy.deepcopy(source)
            row["production_label_eligible"] = False
            row["auxiliary_training_eligible"] = True
            row["preliminary_release_eligible"] = False
            result.append(row)
        return result

    monkeypatch.setattr(preliminary, "_export_preliminary_d1", export)


def test_build_is_deterministic_atomic_nonrelease_and_root_disjoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _minimal_build_inputs(tmp_path)
    _patch_build_admission(monkeypatch)
    receipts = []
    for name in ("first", "second"):
        receipts.append(
            preliminary.build_preliminary_dagger_dataset(
                failure_evidence_path=paths["failure"],
                candidate_rows_path=paths["candidates"],
                d0_generation_provenance_path=paths["provenance"],
                d0_train_path=paths["train"],
                d0_validation_path=paths["validation"],
                output_dir=tmp_path / name,
                evaluation_root_count=2,
                validation_root_count=1,
                d1_training_row_count=3,
                selection_seed=23,
                repo_root=Path(__file__).resolve().parents[2],
            )
        )
    assert receipts[0] == receipts[1]
    assert receipts[0]["release_eligible"] is False
    assert receipts[0]["artifact_type"] == preliminary.PRELIMINARY_ARTIFACT_TYPE
    assert receipts[0]["audits"]["root_disjointness"]["passed"] is True
    composition = receipts[0]["audits"]["content_composition"]
    assert composition["passed"] is True
    assert (
        composition["d1_validation_plus_test_row_multiset_sha256"]
        == composition["d1_eval_combined_row_multiset_sha256"]
    )
    assert (
        composition["bc0_plus_d1_train_row_multiset_sha256"]
        == composition["mixed_train_row_multiset_sha256"]
    )
    assert all(
        not overlap
        for overlap in receipts[0]["audits"]["root_disjointness"][
            "required_pairwise_overlaps"
        ].values()
    )
    for filename in [*preliminary.OUTPUT_FILENAMES.values(), preliminary.RECEIPT_FILENAME]:
        assert (tmp_path / "first" / filename).read_bytes() == (
            tmp_path / "second" / filename
        ).read_bytes()
    assert not list((tmp_path / "first").glob("*.tmp"))


def test_build_rejects_bc0_d1_root_overlap_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _minimal_build_inputs(tmp_path)
    candidates = [
        json.loads(line)
        for line in paths["candidates"].read_text(encoding="utf-8").splitlines()
    ]
    candidates[0]["physical_root_fingerprint"] = "d0-train-root"
    _write_jsonl(paths["candidates"], candidates)
    _patch_build_admission(monkeypatch)
    output = tmp_path / "overlap-output"
    with pytest.raises(ValueError, match="overlap BC0"):
        preliminary.build_preliminary_dagger_dataset(
            failure_evidence_path=paths["failure"],
            candidate_rows_path=paths["candidates"],
            d0_generation_provenance_path=paths["provenance"],
            d0_train_path=paths["train"],
            d0_validation_path=paths["validation"],
            output_dir=output,
            evaluation_root_count=2,
            validation_root_count=1,
            d1_training_row_count=3,
        )
    assert not output.exists()


def test_root_set_hash_uses_sorted_unique_roots() -> None:
    expected = stable_json_sha256(["a", "b"])
    assert preliminary._root_set_sha256(["b", "a", "b"]) == expected


def test_builder_source_attestation_rejects_untracked_and_modified_sources(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    module = repo / "psse_env" / "dagger" / "preliminary_dataset.py"
    wrapper = repo / "scripts" / "build_preliminary_dagger_dataset.py"
    module.parent.mkdir(parents=True)
    wrapper.parent.mkdir(parents=True)
    module.write_bytes(b"VALUE = 1\n")
    wrapper.write_bytes(b"print('build')\n")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "preliminary@example.invalid"],
        cwd=repo,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Preliminary Test"],
        cwd=repo,
        check=True,
    )
    subprocess.run(["git", "add", module.relative_to(repo)], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "module only"], cwd=repo, check=True)

    with pytest.raises(ValueError, match="not tracked"):
        preliminary._builder_source_attestation(repo)

    subprocess.run(["git", "add", wrapper.relative_to(repo)], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "add wrapper"], cwd=repo, check=True)
    attestation = preliminary._builder_source_attestation(repo)
    assert attestation["tracked_files_match_head"] is True
    assert attestation["source_commit"] == subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    wrapper.write_bytes(b"print('changed')\n")
    with pytest.raises(ValueError, match="differs from HEAD"):
        preliminary._builder_source_attestation(repo)
