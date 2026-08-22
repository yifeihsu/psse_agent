"""Executable tests for evaluator-produced preregistered study evidence."""

from __future__ import annotations

import copy
import hashlib
import io
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from psse_env.dagger import evaluator
from psse_env.dagger.build_dagger1_development_holdout import (
    DAGGER1_DEVELOPMENT_HOLDOUT_CONTRACT,
    DAGGER1_DEVELOPMENT_PARAMETER_RANKING_THRESHOLD,
    DAGGER1_DEVELOPMENT_SPLIT,
    DAGGER1_DEVELOPMENT_SUITE_NAME,
    DEFAULT_DAGGER1_DEVELOPMENT_ROOT_PLAN,
    _source_bindings,
)
from psse_env.dagger.evaluator import build_study_evaluation_binding
from psse_env.dagger.study_manifest import (
    DEFAULT_STUDY_MANIFEST,
    EXPECTED_DEVELOPMENT_EVALUATION_CONTRACT_SHA256,
    PINNED_BASE_MODEL_ID,
    PINNED_BASE_MODEL_REVISION,
    load_study_manifest,
)
from psse_env.dagger.study_metrics import StudyEvidenceError, extract_artifact_metrics
from psse_env.dagger.test_evaluator import _release_partitioned_resolved_scenario
from psse_env.dagger.test_study_manifest import _checkpoint_artifact
from psse_env.providers.matpower import PARAMETER_RANKING_CONTRACT
from psse_env.sft.provenance import file_sha256, stable_json_sha256


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_COMMIT = "b" * 40
CLEAN_SOURCE = {
    "source_commit": SOURCE_COMMIT,
    "source_worktree_dirty": False,
    "tracked_diff_hash": hashlib.sha256(b"").hexdigest(),
    "untracked_source_files": [],
    "release_eligible_source": True,
}


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _development_rows() -> list[dict[str, Any]]:
    families = (
        ["measurement+parameter"] * 12
        + ["multi_measurement"] * 12
        + ["parameter"] * 6
    )
    rows: list[dict[str, Any]] = []
    for index, family in enumerate(families):
        row = _release_partitioned_resolved_scenario()
        scenario_id = f"development-{index:02d}"
        physical_root = f"development-root-{index:02d}"
        row["execution"]["scenario_id"] = scenario_id
        row["grouping"].update(
            {
                "root_scenario_id": scenario_id,
                "physical_root_fingerprint": physical_root,
                "scenario_family": family,
                "error_cardinality": (
                    2 + (index % 4) if family == "multi_measurement" else 2
                ),
                "split": DAGGER1_DEVELOPMENT_SPLIT,
                "source_tier": "test",
            }
        )
        rows.append(row)
    return rows


def _write_development_inputs(
    root: Path,
) -> tuple[Path, Path, Path, dict[str, Any]]:
    study_manifest = load_study_manifest()
    rows = _development_rows()
    suite_path = root / "dagger1-development.json"
    generator_report_path = root / "dagger1-development.generator.json"
    holdout_manifest_path = root / "dagger1-development.manifest.json"
    _write_json(suite_path, {DAGGER1_DEVELOPMENT_SUITE_NAME: rows})
    generator_report = {
        "seed": 20260721,
        "source_partition": {"enabled": True, "selected": "train"},
        "parameter_ranking_admission": {
            "contract": PARAMETER_RANKING_CONTRACT,
            "enforced": True,
            "threshold": DAGGER1_DEVELOPMENT_PARAMETER_RANKING_THRESHOLD,
        },
    }
    _write_json(generator_report_path, generator_report)
    roots = sorted(
        row["grouping"]["physical_root_fingerprint"] for row in rows
    )
    selected_count_by_family = {
        family: sum(
            row["grouping"]["scenario_family"] == family for row in rows
        )
        for family in sorted(DEFAULT_DAGGER1_DEVELOPMENT_ROOT_PLAN)
    }
    frozen = study_manifest["bindings"]["evaluation"]
    holdout_manifest = {
        "schema_version": 1,
        "scenario_schema_version": 1,
        "artifact_type": "dagger1_development_holdout_suite",
        "builder_contract": DAGGER1_DEVELOPMENT_HOLDOUT_CONTRACT,
        "source_state": copy.deepcopy(CLEAN_SOURCE),
        "source_bindings": _source_bindings(REPO_ROOT),
        "suite_name": DAGGER1_DEVELOPMENT_SUITE_NAME,
        "suite_format": "evaluation_suite_mapping_v1",
        "split": DAGGER1_DEVELOPMENT_SPLIT,
        "source_partition": "train",
        "parameter_ranking_dominance_threshold": (
            DAGGER1_DEVELOPMENT_PARAMETER_RANKING_THRESHOLD
        ),
        "seed": 20260721,
        "plan": dict(sorted(DEFAULT_DAGGER1_DEVELOPMENT_ROOT_PLAN.items())),
        "selected_count_by_family": selected_count_by_family,
        "scenario_count": len(rows),
        "physical_root_count": len(roots),
        "training_eligible": False,
        "training_collection_eligible": False,
        "release_evidence_eligible": False,
        "promotion_evidence_eligible": False,
        "diagnostic_closed_loop_model_selection_eligible": True,
        "root_counts": {"development": len(roots)},
        "root_set_sha256": {"development": stable_json_sha256(roots)},
        "pairwise_input_overlap": {
            "d0_frozen": [],
            "d0_d1_training": [],
            "frozen_d1_training": [],
        },
        "training_development_reserved_boundary_overlap": {
            "d0": [],
            "frozen": [],
            "d1_training": [],
        },
        "development_protected_overlap": {
            "d0": [],
            "frozen": [],
            "d1_training": [],
        },
        "output_sha256": file_sha256(suite_path),
        "generator_report_sha256": file_sha256(generator_report_path),
        "frozen_suite_sha256": frozen["suite_sha256"],
        "evaluation_policy_sha256": frozen["policy_sha256"],
    }
    _write_json(holdout_manifest_path, holdout_manifest)
    return suite_path, holdout_manifest_path, generator_report_path, study_manifest


def _base_development_binding_kwargs(root: Path) -> dict[str, Any]:
    suite, holdout_manifest, generator_report, _ = _write_development_inputs(root)
    return {
        "study_manifest_path": DEFAULT_STUDY_MANIFEST,
        "variant_id": "base",
        "reviewed_source_commit": SOURCE_COMMIT,
        "model_id": PINNED_BASE_MODEL_ID,
        "model_revision": PINNED_BASE_MODEL_REVISION,
        "input_suite_path": suite,
        "diagnostic_only": True,
        "evaluator_seed": 20260721,
        "max_steps": 24,
        "required_suites": (DAGGER1_DEVELOPMENT_SUITE_NAME,),
        "minimum_suites": 1,
        "minimum_episodes_per_suite": 1,
        "minimum_roots_per_suite": 30,
        "protocol": "canonical",
        "development_holdout_manifest_path": holdout_manifest,
        "development_holdout_generator_report_path": generator_report,
    }


def _rehash_artifact(payload: dict[str, Any]) -> dict[str, Any]:
    forged = copy.deepcopy(payload)
    forged.pop("content_sha256", None)
    forged["content_sha256"] = stable_json_sha256(forged)
    return forged


def test_development_suite_schema_exception_is_explicitly_gated() -> None:
    suite = {DAGGER1_DEVELOPMENT_SUITE_NAME: [_development_rows()[0]]}
    with pytest.raises(ValueError, match="unsupported evaluation suite"):
        evaluator.validate_release_scenario_suites(suite)
    validated = evaluator.validate_release_scenario_suites(
        suite,
        allow_diagnostic_development=True,
    )
    assert set(validated) == {DAGGER1_DEVELOPMENT_SUITE_NAME}


def test_cli_emits_real_bound_development_artifact_consumed_by_metrics(
    tmp_path: Path,
) -> None:
    suite, holdout_manifest, generator_report, study_manifest = (
        _write_development_inputs(tmp_path)
    )
    output = tmp_path / "base-development-evaluation.json"
    arguments = [
        "--input",
        str(suite),
        "--output",
        str(output),
        "--env-factory",
        "psse_env.dagger.test_evaluator:_ReleaseScriptEnv",
        "--policy-factory",
        "psse_env.dagger.test_evaluator:_cli_policy_factory",
        "--case-loader",
        "psse_env.dagger.test_evaluator:_cli_case_loader",
        "--model-id",
        PINNED_BASE_MODEL_ID,
        "--model-revision",
        PINNED_BASE_MODEL_REVISION,
        "--required-suite",
        DAGGER1_DEVELOPMENT_SUITE_NAME,
        "--minimum-suites",
        "1",
        "--minimum-episodes-per-suite",
        "1",
        "--minimum-roots-per-suite",
        "30",
        "--seed",
        "20260721",
        "--max-steps",
        "24",
        "--diagnostic-only",
        "--study-manifest",
        str(DEFAULT_STUDY_MANIFEST),
        "--study-variant",
        "base",
        "--reviewed-source-commit",
        SOURCE_COMMIT,
        "--development-holdout-manifest",
        str(holdout_manifest),
        "--development-holdout-generator-report",
        str(generator_report),
    ]
    with mock.patch.object(evaluator, "git_source_state", return_value=CLEAN_SOURCE):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            assert evaluator.main(arguments) == 0
    artifact = json.loads(output.read_text(encoding="utf-8"))
    run = extract_artifact_metrics(
        output,
        variant_id="base",
        study_seed=None,
        evaluation_scope="development_holdout",
        study_manifest=study_manifest,
        expected_source_commit=SOURCE_COMMIT,
    )
    assert run["evaluation_scope"] == "development_holdout"
    assert run["evaluator_seed"] == 20260721
    assert run["max_steps"] == 24
    assert len(run["root_records"]) == 30
    assert artifact["checkpoint_receipt_id"] is None
    assert artifact["checkpoint_adapter_tree_sha256"] is None
    assert artifact["training_seed"] is None
    assert artifact["development_evaluation_contract_sha256"] == (
        EXPECTED_DEVELOPMENT_EVALUATION_CONTRACT_SHA256
    )
    assert all(
        episode["trace"][0]["error_code"] == "schema_error"
        for episode in artifact["evaluation"]["suite_metrics"]["episodes"]
    )

    # Study evidence is write-once even when a rerun would reproduce it.
    with mock.patch.object(evaluator, "git_source_state", return_value=CLEAN_SOURCE):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            with pytest.raises(FileExistsError):
                evaluator.main(arguments)

    for mutate in (
        lambda value: value.update(
            {"development_evaluation_contract_sha256": "a" * 64}
        ),
        lambda value: value["evaluation"]["suite_metrics"]["configuration"].update(
            {"seed": 20260722}
        ),
        lambda value: value["evaluation"]["suite_metrics"]["configuration"].update(
            {"required_suites": ["forged-development"]}
        ),
        lambda value: value["evaluation"]["suite_metrics"]["configuration"].update(
            {"minimum_roots_per_suite": 29}
        ),
        lambda value: value["provenance"]["protocol_registry"].update(
            {"protocol": "controller"}
        ),
    ):
        forged = copy.deepcopy(artifact)
        mutate(forged)
        forged = _rehash_artifact(forged)
        with pytest.raises(StudyEvidenceError):
            extract_artifact_metrics(
                forged,
                variant_id="base",
                study_seed=None,
                evaluation_scope="development_holdout",
                study_manifest=study_manifest,
                expected_source_commit=SOURCE_COMMIT,
            )

    with mock.patch.object(evaluator, "git_source_state", return_value=CLEAN_SOURCE):
        binding = build_study_evaluation_binding(
            study_manifest_path=DEFAULT_STUDY_MANIFEST,
            variant_id="base",
            reviewed_source_commit=SOURCE_COMMIT,
            model_id=PINNED_BASE_MODEL_ID,
            model_revision=PINNED_BASE_MODEL_REVISION,
            input_suite_path=suite,
            diagnostic_only=True,
            evaluator_seed=20260721,
            max_steps=24,
            required_suites=(DAGGER1_DEVELOPMENT_SUITE_NAME,),
            minimum_suites=1,
            minimum_episodes_per_suite=1,
            minimum_roots_per_suite=30,
            protocol="canonical",
            development_holdout_manifest_path=holdout_manifest,
            development_holdout_generator_report_path=generator_report,
        )
    result = evaluator.EvaluationResult(
        score=artifact["evaluation"]["score"],
        metrics=evaluator.RecoveryMetrics(**artifact["evaluation"]["metrics"]),
        suite_metrics=artifact["evaluation"]["suite_metrics"],
    )
    mismatched_suite_metrics = copy.deepcopy(result.suite_metrics)
    mismatched_suite_metrics["configuration"]["suite_content_sha256"] = "0" * 64
    mismatched_result = evaluator.EvaluationResult(
        score=result.score,
        metrics=result.metrics,
        suite_metrics=mismatched_suite_metrics,
    )
    with pytest.raises(ValueError, match="not produced from the bound suite"):
        evaluator.write_evaluation_artifact(
            mismatched_result,
            tmp_path / "forged-mismatched-result.json",
            provenance=artifact["provenance"],
            diagnostic_only=True,
            study_binding=binding,
            study_manifest_path=DEFAULT_STUDY_MANIFEST,
        )
    changed_suite = json.loads(suite.read_text(encoding="utf-8"))
    changed_suite[DAGGER1_DEVELOPMENT_SUITE_NAME][0]["execution"][
        "measurements"
    ][0] = 123.0
    _write_json(suite, changed_suite)
    with pytest.raises(ValueError, match="input bytes changed after binding"):
        evaluator.write_evaluation_artifact(
            result,
            tmp_path / "forged-swapped-input.json",
            provenance=artifact["provenance"],
            diagnostic_only=True,
            study_binding=binding,
            study_manifest_path=DEFAULT_STUDY_MANIFEST,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("evaluator_seed", 20260722),
        ("max_steps", 23),
        ("required_suites", ("forged-development",)),
        ("minimum_suites", 2),
        ("minimum_episodes_per_suite", 2),
        ("minimum_roots_per_suite", 29),
        ("protocol", "controller"),
    ),
)
def test_development_builder_rejects_every_contract_mismatch(
    tmp_path: Path,
    field: str,
    value: Any,
) -> None:
    kwargs = _base_development_binding_kwargs(tmp_path)
    kwargs[field] = value
    with mock.patch.object(evaluator, "git_source_state", return_value=CLEAN_SOURCE):
        with pytest.raises(ValueError, match="preregistered contract"):
            build_study_evaluation_binding(**kwargs)


def _write_bc0_receipt(
    root: Path,
) -> tuple[dict[str, Any], Path, Path, dict[str, Any]]:
    study_manifest = load_study_manifest()
    adapter = root / "bc0-seed3407" / "lora"
    adapter.mkdir(parents=True)
    receipt = _checkpoint_artifact(
        study_manifest,
        variant_id="bc0",
        seed=3407,
    )
    receipt["adapter_path"] = str(adapter.resolve())
    receipt.pop("checkpoint_receipt_id")
    receipt["checkpoint_receipt_id"] = stable_json_sha256(receipt)
    receipt_path = adapter.parent / "checkpoint_receipt.json"
    _write_json(receipt_path, receipt)
    return receipt, receipt_path, adapter.resolve(), study_manifest


def _bc0_binding_kwargs(
    receipt_path: Path,
    adapter: Path,
    study_manifest: dict[str, Any],
) -> dict[str, Any]:
    frozen = study_manifest["bindings"]["evaluation"]
    return {
        "study_manifest_path": DEFAULT_STUDY_MANIFEST,
        "variant_id": "bc0",
        "reviewed_source_commit": SOURCE_COMMIT,
        "model_id": str(adapter),
        "model_revision": "f" * 64,
        "input_suite_path": REPO_ROOT / frozen["suite_path"],
        "diagnostic_only": False,
        "evaluator_seed": frozen["evaluator_seed"],
        "max_steps": frozen["max_steps"],
        "required_suites": ("standard_success",),
        "minimum_suites": 1,
        "minimum_episodes_per_suite": 1,
        "minimum_roots_per_suite": 1,
        "protocol": "canonical",
        "training_seed": 3407,
        "checkpoint_receipt_path": receipt_path,
    }


def test_trained_binding_requires_exact_receipt_seed_variant_source_and_live_tree(
    tmp_path: Path,
) -> None:
    receipt, receipt_path, adapter, study_manifest = _write_bc0_receipt(tmp_path)
    kwargs = _bc0_binding_kwargs(receipt_path, adapter, study_manifest)

    def inspection(*_: Any, **__: Any) -> dict[str, Any]:
        return {"path": str(adapter), "tree_sha256": "f" * 64}

    with mock.patch.object(evaluator, "git_source_state", return_value=CLEAN_SOURCE), mock.patch(
        "psse_env.dagger.release_factories.inspect_release_checkpoint",
        side_effect=inspection,
    ):
        binding = build_study_evaluation_binding(**kwargs)
    assert binding["training_seed"] == 3407
    assert binding["checkpoint_receipt_id"] == receipt["checkpoint_receipt_id"]
    assert binding["checkpoint_adapter_tree_sha256"] == "f" * 64
    assert binding["variant_id"] == "bc0"

    mismatches = (
        {"training_seed": 3408},
        {"variant_id": "natural_dagger"},
    )
    for mismatch in mismatches:
        forged_kwargs = {**kwargs, **mismatch}
        with mock.patch.object(
            evaluator, "git_source_state", return_value=CLEAN_SOURCE
        ), mock.patch(
            "psse_env.dagger.release_factories.inspect_release_checkpoint",
            side_effect=inspection,
        ):
            with pytest.raises(ValueError):
                build_study_evaluation_binding(**forged_kwargs)

    forged_receipt = copy.deepcopy(receipt)
    forged_receipt["reviewed_source_commit"] = "c" * 40
    forged_receipt.pop("checkpoint_receipt_id")
    forged_receipt["checkpoint_receipt_id"] = stable_json_sha256(forged_receipt)
    forged_path = tmp_path / "forged-source-receipt.json"
    _write_json(forged_path, forged_receipt)
    with mock.patch.object(evaluator, "git_source_state", return_value=CLEAN_SOURCE):
        with pytest.raises(ValueError):
            build_study_evaluation_binding(
                **{**kwargs, "checkpoint_receipt_path": forged_path}
            )

    invalid_id = copy.deepcopy(receipt)
    invalid_id["checkpoint_receipt_id"] = "0" * 64
    invalid_id_path = tmp_path / "forged-id-receipt.json"
    _write_json(invalid_id_path, invalid_id)
    with mock.patch.object(evaluator, "git_source_state", return_value=CLEAN_SOURCE):
        with pytest.raises(ValueError):
            build_study_evaluation_binding(
                **{**kwargs, "checkpoint_receipt_path": invalid_id_path}
            )

    with mock.patch.object(evaluator, "git_source_state", return_value=CLEAN_SOURCE), mock.patch(
        "psse_env.dagger.release_factories.inspect_release_checkpoint",
        return_value={"path": str(adapter), "tree_sha256": "e" * 64},
    ):
        with pytest.raises(ValueError, match="must be identical"):
            build_study_evaluation_binding(**kwargs)

    other_adapter = tmp_path / "other-adapter"
    other_adapter.mkdir()
    with mock.patch.object(evaluator, "git_source_state", return_value=CLEAN_SOURCE), mock.patch(
        "psse_env.dagger.release_factories.inspect_release_checkpoint",
        return_value={"path": str(other_adapter.resolve()), "tree_sha256": "f" * 64},
    ):
        with pytest.raises(ValueError, match="adapter_path differs"):
            build_study_evaluation_binding(**kwargs)
