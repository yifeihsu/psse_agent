"""Durable checkpoint provenance for every preregistered training run."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from psse_env.dagger.study_manifest import (
    build_production_d1_quarantine_binding,
    build_training_protocol_binding,
    canonical_production_d1_quarantine_binding,
    canonical_training_rng_attestation,
    load_study_manifest,
)
from psse_env.sft.gates import GateError
from psse_env.sft.provenance import stable_json_sha256
from psse_env.sft.training import (
    LoraSettings,
    TrainerSettings,
    _prepare_checkpoint_receipt_binding,
    _reject_existing_checkpoint_outputs,
    _validated_parent_checkpoint_receipt,
    run_lora_training,
    _write_base_snapshot_attestation,
    _write_checkpoint_receipt,
)


MODEL_REVISION = "8a796db4df380b178065ed910849477ff0e99c87"
SOURCE_COMMIT = "a" * 40
PARENT_REVISION = "b" * 64
ADAPTER_REVISION = "d" * 64
ACCELERATOR = {
    "device_count": 1,
    "bf16_supported": True,
    "torch_cuda_version": "12.8",
    "required_accelerator_class": None,
    "required_accelerator_class_matched": True,
    "devices": [
        {
            "index": 0,
            "name": "NVIDIA H200",
            "total_memory_bytes": 141 * 1024**3,
            "compute_capability": [9, 0],
            "accelerator_class": "h200",
        }
    ],
}
QUARANTINE_SUMMARY = {
    "contract": "dagger1_offline_teacher_target_quarantine_summary_v1",
    "candidate_definition": {},
    "total_rows": 25,
    "candidate_rows": 25,
    "non_candidate_rows": 0,
    "passed_rows": 25,
    "quarantined_rows": 0,
    "invalid_or_missing_audit_rows": 0,
    "quarantined_by_action_class": {},
    "quarantined_by_reason_code": {},
    "quarantined_example_ids": [],
    "zero_truth_audit_quarantine": True,
    "passed": True,
}
ROUND1_DESCRIPTOR = {
    "builder_contract": "deterministic_d0_d1_probe_balanced_union_v2",
    "source_state": {
        "source_commit": SOURCE_COMMIT,
        "release_eligible_source": True,
    },
    "audit_report_sha256": {
        "d1_offline_teacher_target_quarantine_summary": stable_json_sha256(
            QUARANTINE_SUMMARY
        )
    },
}
VIEW_PROVENANCE = stable_json_sha256(ROUND1_DESCRIPTOR)


def _generation() -> dict:
    return {
        "passed": True,
        "failures": [],
        "release_eligible": True,
        "generation_provenance_id": VIEW_PROVENANCE,
        "source_commit": SOURCE_COMMIT,
    }


def _write_bc0_parent_receipt(
    tmp_path: Path,
    *,
    seed: int,
    source_commit: str = SOURCE_COMMIT,
    adapter_revision: str = PARENT_REVISION,
) -> tuple[Path, Path, dict]:
    study = load_study_manifest()
    output = tmp_path / "bc0"
    adapter = output / "lora"
    adapter.mkdir(parents=True)
    protocol = build_training_protocol_binding(study, variant_id="bc0")
    payload = {
        "artifact_schema_version": 1,
        "artifact_role": "checkpoint",
        "variant_id": "bc0",
        "study_manifest_sha256": study["manifest_sha256"],
        "reviewed_source_commit": source_commit,
        "base_model_id": "unsloth/gemma-4-31B-it",
        "base_model_revision": MODEL_REVISION,
        "base_snapshot_attestation_sha256": "9" * 64,
        "training_seed": seed,
        "training_view_provenance_id": "8" * 64,
        "training_protocol": protocol,
        "training_configuration": protocol["configuration"],
        "training_rng_attestation": canonical_training_rng_attestation(
            variant_id="bc0",
            training_seed=seed,
        ),
        "parent_checkpoint_receipt_id": None,
        "training_sources": ["d0_bc0"],
        "round1_view": None,
        "production_d1_quarantine_binding": (
            canonical_production_d1_quarantine_binding("bc0")
        ),
        "training_dataset_sha256": {
            "train": "1" * 64,
            "validation": "2" * 64,
        },
        "parent_model_revision": MODEL_REVISION,
        "adapter_path": str(adapter.resolve()),
        "adapter_tree_sha256": adapter_revision,
        "runtime_accelerator_attestation": ACCELERATOR,
    }
    payload["checkpoint_receipt_id"] = stable_json_sha256(payload)
    receipt = output / "checkpoint_receipt.json"
    receipt.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return adapter, receipt, payload


def _round1_source_report(
    train: Path, validation: Path, *, variant_id: str
) -> dict:
    def rows(path: Path) -> list[dict]:
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    return {
        "passed": True,
        "generation_provenance_id": VIEW_PROVENANCE,
        "canonical_dataset_content_sha256": {
            "train": stable_json_sha256(rows(train)),
            "validation": stable_json_sha256(rows(validation)),
        },
        "production_d1_quarantine_binding": (
            build_production_d1_quarantine_binding(
                variant_id=variant_id,
                generation_provenance_id=VIEW_PROVENANCE,
                generation_descriptor=ROUND1_DESCRIPTOR,
                summary=QUARANTINE_SUMMARY,
                audit_report_sha256=stable_json_sha256(QUARANTINE_SUMMARY),
            )
        ),
    }


def test_prepare_binding_authenticates_variant_source_and_seed_parent(
    tmp_path: Path,
) -> None:
    parent_adapter, parent_receipt, parent_payload = _write_bc0_parent_receipt(
        tmp_path,
        seed=3408,
    )
    train = tmp_path / "aggregate.train.jsonl"
    validation = tmp_path / "aggregate.validation.jsonl"
    train.write_text(
        "\n".join(
            json.dumps({"replay_source": source})
            for source in ("d0_bc0", "natural_dagger1")
        )
        + "\n",
        encoding="utf-8",
    )
    validation.write_text("{}\n", encoding="utf-8")
    settings = TrainerSettings(
        revision=MODEL_REVISION,
        output_dir=str(tmp_path / "output"),
        max_length=6144,
        learning_rate=0.00003,
        epochs=1.0,
        load_in_4bit=True,
        required_processor_loader="AutoProcessor",
        initial_adapter_path=str(parent_adapter),
        initial_adapter_revision=PARENT_REVISION,
        parent_checkpoint_receipt_path=str(parent_receipt),
        round1_provenance_path=str(tmp_path / "provenance.json"),
        round1_preflight_path=str(tmp_path / "preflight.json"),
        reviewed_source_commit=SOURCE_COMMIT,
        round1_view="natural-only",
        study_variant="natural_dagger",
        seed=3408,
    )
    with (
        mock.patch(
            "psse_env.sft.training.validate_generation_provenance",
            return_value=_generation(),
        ),
        mock.patch(
            "psse_env.sft.training.git_source_state",
            return_value={
                "source_commit": SOURCE_COMMIT,
                "release_eligible_source": True,
            },
        ),
        mock.patch(
            "psse_env.sft.round1_source_gate.validate_round1_source_mix_gate",
            return_value=_round1_source_report(
                train, validation, variant_id="natural_dagger"
            ),
        ),
    ):
        binding = _prepare_checkpoint_receipt_binding(
            train_file=train,
            validation_file=validation,
            settings=settings,
        )

    assert binding["variant_id"] == "natural_dagger"
    assert binding["training_sources"] == ["d0_bc0", "natural_dagger1"]
    assert binding["parent_model_revision"] == PARENT_REVISION
    assert binding["parent_checkpoint_receipt_id"] == parent_payload[
        "checkpoint_receipt_id"
    ]
    assert binding["training_view_provenance_id"] == VIEW_PROVENANCE
    assert binding["reviewed_source_commit"] == SOURCE_COMMIT
    assert binding["round1_view"] == "natural-only"
    assert binding["production_d1_quarantine_binding"][
        "audit_report_sha256"
    ] == stable_json_sha256(QUARANTINE_SUMMARY)
    assert binding["production_d1_quarantine_binding"]["quarantined_rows"] == 0
    assert set(binding["dataset_sha256"]) == {"train", "validation"}


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("forged_receipt_id", "does not match its payload"),
        ("wrong_seed", "paired seed"),
        ("wrong_source", "source commit"),
        ("wrong_manifest", "immutable study manifest"),
        ("wrong_variant", "variant_id"),
        ("wrong_tree", "adapter tree differs"),
        ("wrong_path", "adapter path differs"),
    ],
)
def test_parent_checkpoint_receipt_rejects_forged_or_mismatched_identity(
    tmp_path: Path,
    mutation: str,
    match: str,
) -> None:
    adapter, receipt, payload = _write_bc0_parent_receipt(
        tmp_path,
        seed=3408,
    )
    other_adapter = tmp_path / "other" / "lora"
    other_adapter.mkdir(parents=True)
    if mutation == "forged_receipt_id":
        payload["checkpoint_receipt_id"] = "f" * 64
    elif mutation == "wrong_seed":
        payload["training_seed"] = 3409
    elif mutation == "wrong_source":
        payload["reviewed_source_commit"] = "c" * 40
    elif mutation == "wrong_manifest":
        payload["study_manifest_sha256"] = "d" * 64
    elif mutation == "wrong_variant":
        payload["variant_id"] = "natural_dagger"
    elif mutation == "wrong_tree":
        payload["adapter_tree_sha256"] = "c" * 64
    elif mutation == "wrong_path":
        payload["adapter_path"] = str(other_adapter.resolve())
    if mutation != "forged_receipt_id":
        payload.pop("checkpoint_receipt_id")
        payload["checkpoint_receipt_id"] = stable_json_sha256(payload)
    receipt.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    settings = TrainerSettings(
        revision=MODEL_REVISION,
        initial_adapter_path=str(adapter),
        initial_adapter_revision=PARENT_REVISION,
        parent_checkpoint_receipt_path=str(receipt),
        seed=3408,
    )

    with pytest.raises(GateError, match=match):
        _validated_parent_checkpoint_receipt(
            settings=settings,
            study=load_study_manifest(),
            variant_id="natural_dagger",
            source_commit=SOURCE_COMMIT,
        )


def test_parent_checkpoint_receipt_is_required_and_must_be_canonical_sibling(
    tmp_path: Path,
) -> None:
    adapter, receipt, _payload = _write_bc0_parent_receipt(tmp_path, seed=3408)
    common = {
        "revision": MODEL_REVISION,
        "initial_adapter_path": str(adapter),
        "initial_adapter_revision": PARENT_REVISION,
        "seed": 3408,
    }
    with pytest.raises(GateError, match="requires the same-seed BC0 parent"):
        _validated_parent_checkpoint_receipt(
            settings=TrainerSettings(**common),
            study=load_study_manifest(),
            variant_id="natural_dagger",
            source_commit=SOURCE_COMMIT,
        )

    copied = tmp_path / "copied-receipt.json"
    copied.write_bytes(receipt.read_bytes())
    with pytest.raises(GateError, match="canonical sibling"):
        _validated_parent_checkpoint_receipt(
            settings=TrainerSettings(
                **common,
                parent_checkpoint_receipt_path=str(copied),
            ),
            study=load_study_manifest(),
            variant_id="natural_dagger",
            source_commit=SOURCE_COMMIT,
        )


def test_prepare_binding_rejects_variant_source_contamination(tmp_path: Path) -> None:
    train = tmp_path / "aggregate.train.jsonl"
    validation = tmp_path / "aggregate.validation.jsonl"
    train.write_text(
        json.dumps({"replay_source": "observable_recovery_probe"}) + "\n",
        encoding="utf-8",
    )
    validation.write_text("{}\n", encoding="utf-8")
    settings = TrainerSettings(
        revision=MODEL_REVISION,
        output_dir=str(tmp_path / "output"),
        study_variant="bc0",
        seed=3407,
    )
    with (
        mock.patch(
            "psse_env.sft.training.validate_generation_provenance",
            return_value=_generation(),
        ),
        mock.patch(
            "psse_env.sft.training.git_source_state",
            return_value={
                "source_commit": SOURCE_COMMIT,
                "release_eligible_source": True,
            },
        ),
        pytest.raises(GateError, match="requires training sources"),
    ):
        _prepare_checkpoint_receipt_binding(
            train_file=train,
            validation_file=validation,
            settings=settings,
        )


@pytest.mark.parametrize(
    ("setting_updates", "lora", "label"),
    [
        ({"max_length": 4096}, LoraSettings(), "max_length"),
        (
            {"required_processor_loader": None},
            LoraSettings(),
            "processor loader",
        ),
        ({"trust_remote_code": True}, LoraSettings(), "processor trust"),
        ({"load_in_4bit": False}, LoraSettings(), "4-bit"),
        ({"bf16": False, "fp16": True}, LoraSettings(), "dtype"),
        ({"batch_size": 2}, LoraSettings(), "batch"),
        (
            {"gradient_accumulation_steps": 8},
            LoraSettings(),
            "gradient accumulation",
        ),
        ({"learning_rate": 0.0002}, LoraSettings(), "learning rate"),
        ({"epochs": 1.0}, LoraSettings(), "epochs"),
        ({"max_steps": 10}, LoraSettings(), "max steps"),
        ({"optimizer": "adamw_hf"}, LoraSettings(), "optimizer"),
        ({"lr_scheduler_type": "cosine"}, LoraSettings(), "scheduler"),
        ({}, LoraSettings(rank=8), "LoRA rank"),
    ],
)
def test_prepare_binding_rejects_every_material_protocol_drift(
    tmp_path: Path,
    setting_updates: dict,
    lora: LoraSettings,
    label: str,
) -> None:
    train = tmp_path / "aggregate.train.jsonl"
    validation = tmp_path / "aggregate.validation.jsonl"
    train.write_text(
        json.dumps({"replay_source": "d0_bc0"}) + "\n",
        encoding="utf-8",
    )
    validation.write_text("{}\n", encoding="utf-8")
    values = {
        "revision": MODEL_REVISION,
        "output_dir": str(tmp_path / "output"),
        "max_length": 6144,
        "batch_size": 1,
        "gradient_accumulation_steps": 4,
        "learning_rate": 0.0001,
        "epochs": 2.0,
        "max_steps": -1,
        "optimizer": "adamw_torch",
        "lr_scheduler_type": "linear",
        "bf16": True,
        "fp16": False,
        "load_in_4bit": True,
        "required_processor_loader": "AutoProcessor",
        "study_variant": "bc0",
        "seed": 3407,
    }
    values.update(setting_updates)
    settings = TrainerSettings(**values)
    with (
        mock.patch(
            "psse_env.sft.training.validate_generation_provenance",
            return_value=_generation(),
        ),
        mock.patch(
            "psse_env.sft.training.git_source_state",
            return_value={
                "source_commit": SOURCE_COMMIT,
                "release_eligible_source": True,
            },
        ),
        pytest.raises(GateError, match="immutable training protocol"),
    ):
        _prepare_checkpoint_receipt_binding(
            train_file=train,
            validation_file=validation,
            settings=settings,
            lora=lora,
        )


def test_prepare_binding_infers_full_variant_from_exact_sources(
    tmp_path: Path,
) -> None:
    parent_adapter, parent_receipt, parent_payload = _write_bc0_parent_receipt(
        tmp_path,
        seed=3408,
    )
    train = tmp_path / "aggregate.train.jsonl"
    validation = tmp_path / "aggregate.validation.jsonl"
    train.write_text(
        "\n".join(
            json.dumps({"replay_source": source})
            for source in (
                "d0_bc0",
                "natural_dagger1",
                "observable_recovery_probe",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    validation.write_text("{}\n", encoding="utf-8")
    settings = TrainerSettings(
        revision=MODEL_REVISION,
        output_dir=str(tmp_path / "output"),
        max_length=6144,
        learning_rate=0.00003,
        epochs=1.0,
        load_in_4bit=True,
        required_processor_loader="AutoProcessor",
        initial_adapter_path=str(parent_adapter),
        initial_adapter_revision=PARENT_REVISION,
        parent_checkpoint_receipt_path=str(parent_receipt),
        round1_provenance_path=str(tmp_path / "provenance.json"),
        round1_preflight_path=str(tmp_path / "preflight.json"),
        reviewed_source_commit=SOURCE_COMMIT,
        round1_view="full",
        seed=3408,
    )
    with (
        mock.patch(
            "psse_env.sft.training.validate_generation_provenance",
            return_value=_generation(),
        ),
        mock.patch(
            "psse_env.sft.training.git_source_state",
            return_value={
                "source_commit": SOURCE_COMMIT,
                "release_eligible_source": True,
            },
        ),
        mock.patch(
            "psse_env.sft.round1_source_gate.validate_round1_source_mix_gate",
            return_value=_round1_source_report(
                train, validation, variant_id="natural_dagger_probes"
            ),
        ),
    ):
        binding = _prepare_checkpoint_receipt_binding(
            train_file=train,
            validation_file=validation,
            settings=settings,
        )

    assert binding["variant_id"] == "natural_dagger_probes"
    assert binding["parent_checkpoint_receipt_id"] == parent_payload[
        "checkpoint_receipt_id"
    ]
    assert binding["round1_view"] == "full"
    assert binding["production_d1_quarantine_binding"]["candidate_rows"] == 25


def test_checkpoint_receipt_is_write_once_and_binds_final_tree(tmp_path: Path) -> None:
    output = tmp_path / "output"
    adapter = output / "lora"
    adapter.mkdir(parents=True)
    study = load_study_manifest()
    settings = TrainerSettings(
        revision=MODEL_REVISION,
        output_dir=str(output),
        max_length=6144,
        epochs=2.0,
        load_in_4bit=True,
        required_processor_loader="AutoProcessor",
        seed=3409,
        study_variant="bc0",
    )
    protocol = build_training_protocol_binding(study, variant_id="bc0")
    binding = {
        "variant_id": "bc0",
        "study_manifest": study,
        "study_manifest_sha256": study["manifest_sha256"],
        "reviewed_source_commit": SOURCE_COMMIT,
        "training_view_provenance_id": VIEW_PROVENANCE,
        "training_sources": ["d0_bc0"],
        "round1_view": None,
        "production_d1_quarantine_binding": (
            canonical_production_d1_quarantine_binding("bc0")
        ),
        "training_protocol": protocol,
        "parent_checkpoint_receipt_id": None,
        "parent_model_revision": MODEL_REVISION,
        "dataset_sha256": {"train": "e" * 64, "validation": "f" * 64},
    }
    inspection = {
        "path": str(adapter.resolve()),
        "tree_sha256": ADAPTER_REVISION,
        "file_count": 2,
        "total_bytes": 1234,
    }
    base = {
        "model_id": "unsloth/gemma-4-31B-it",
        "model_revision": MODEL_REVISION,
    }
    base_path = _write_base_snapshot_attestation(
        settings=settings,
        snapshot_attestation=base,
    )
    with (
        mock.patch(
            "psse_env.dagger.release_factories.inspect_release_checkpoint",
            return_value=inspection,
        ),
        mock.patch("psse_env.sft.training._fsync_directory") as directory_fsync,
    ):
        receipt = _write_checkpoint_receipt(
            settings=settings,
            binding=binding,
            adapter_dir=adapter,
            base_snapshot_attestation=base,
            base_snapshot_attestation_path=base_path,
            runtime_accelerator_attestation=ACCELERATOR,
            training_rng_attestation=canonical_training_rng_attestation(
                variant_id="bc0",
                training_seed=3409,
            ),
        )
        directory_fsync.assert_called_once_with(output)
        with pytest.raises(GateError, match="refusing replacement"):
            _write_checkpoint_receipt(
                settings=settings,
                binding=binding,
                adapter_dir=adapter,
                base_snapshot_attestation=base,
                base_snapshot_attestation_path=base_path,
                runtime_accelerator_attestation=ACCELERATOR,
                training_rng_attestation=canonical_training_rng_attestation(
                    variant_id="bc0",
                    training_seed=3409,
                ),
            )

    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert receipt.parent == output
    assert adapter not in receipt.parents
    assert list(output.glob(".checkpoint_receipt.*.tmp")) == []
    receipt_id = payload.pop("checkpoint_receipt_id")
    assert receipt_id == stable_json_sha256(payload)
    assert payload["artifact_role"] == "checkpoint"
    assert payload["variant_id"] == "bc0"
    assert payload["training_seed"] == 3409
    assert payload["training_view_provenance_id"] == VIEW_PROVENANCE
    assert payload["production_d1_quarantine_binding"] == (
        canonical_production_d1_quarantine_binding("bc0")
    )
    assert payload["adapter_tree_sha256"] == ADAPTER_REVISION
    assert payload["parent_model_revision"] == MODEL_REVISION
    assert payload["parent_checkpoint_receipt_id"] is None
    assert payload["training_protocol"] == protocol
    assert payload["training_rng_attestation"] == (
        canonical_training_rng_attestation(
            variant_id="bc0",
            training_seed=3409,
        )
    )
    assert payload["base_snapshot_attestation_sha256"] == (
        hashlib.sha256(base_path.read_bytes()).hexdigest()
    )
    assert payload["runtime_accelerator_attestation"]["devices"][0][
        "accelerator_class"
    ] == "h200"


@pytest.mark.parametrize(
    "conflict_name",
    [
        "lora",
        "base_snapshot_attestation.json",
        "initial_adapter_attestation.json",
        "checkpoint_receipt.json",
        "checkpoint-25",
        ".checkpoint_receipt.stale.tmp",
        ".base_snapshot_attestation.stale.tmp",
        ".initial_adapter_attestation.stale.tmp",
        ".lora.stale.tmp",
    ],
)
def test_checkpoint_preflight_rejects_partial_or_staging_outputs(
    tmp_path: Path,
    conflict_name: str,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    conflict = output / conflict_name
    if conflict_name == "lora" or conflict_name.startswith("checkpoint-"):
        conflict.mkdir()
    else:
        conflict.write_text("partial", encoding="utf-8")
    settings = TrainerSettings(
        revision=MODEL_REVISION,
        output_dir=str(output),
        study_variant="bc0",
        seed=3407,
    )

    with pytest.raises(GateError, match="not clean"):
        _reject_existing_checkpoint_outputs(settings)


def test_checkpoint_preflight_rejects_dangling_lora_symlink(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    link = output / "lora"
    try:
        os.symlink(output / "missing-checkpoint", link, target_is_directory=True)
    except OSError as exc:  # pragma: no cover - depends on Windows policy.
        pytest.skip(f"symbolic links are unavailable: {exc}")
    assert not link.exists()
    assert os.path.lexists(link)
    settings = TrainerSettings(
        revision=MODEL_REVISION,
        output_dir=str(output),
        study_variant="bc0",
        seed=3407,
    )

    with pytest.raises(GateError, match="not clean"):
        _reject_existing_checkpoint_outputs(settings)


def test_base_snapshot_attestation_is_atomic_and_write_once(
    tmp_path: Path,
) -> None:
    settings = TrainerSettings(
        revision=MODEL_REVISION,
        output_dir=str(tmp_path / "output"),
        study_variant="bc0",
        seed=3407,
    )
    snapshot = {
        "model_id": "unsloth/gemma-4-31B-it",
        "model_revision": MODEL_REVISION,
    }
    path = _write_base_snapshot_attestation(
        settings=settings,
        snapshot_attestation=snapshot,
    )
    with pytest.raises(GateError, match="refusing replacement"):
        _write_base_snapshot_attestation(
            settings=settings,
            snapshot_attestation=snapshot,
        )
    assert json.loads(path.read_text(encoding="utf-8")) == snapshot
    assert list(path.parent.glob(".base_snapshot_attestation.*.tmp")) == []


def test_training_requeue_conflict_fails_before_processor_or_model(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    (output / "lora").mkdir(parents=True)
    settings = TrainerSettings(
        revision=MODEL_REVISION,
        output_dir=str(output),
        study_variant="bc0",
        seed=3407,
    )
    with (
        mock.patch("psse_env.sft.training._prepare_pilot") as prepare_pilot,
        mock.patch("psse_env.sft.training._load_model") as load_model,
        pytest.raises(GateError, match="not clean"),
    ):
        run_lora_training(
            train_file=tmp_path / "missing-train.jsonl",
            validation_file=tmp_path / "missing-validation.jsonl",
            settings=settings,
        )
    prepare_pilot.assert_not_called()
    load_model.assert_not_called()


def test_training_input_replacement_after_receipt_binding_fails_before_pilot(
    tmp_path: Path,
) -> None:
    train_file = tmp_path / "train.jsonl"
    validation_file = tmp_path / "validation.jsonl"
    train_file.write_text('{"row": "original"}\n', encoding="utf-8")
    validation_file.write_text('{"row": "validation"}\n', encoding="utf-8")
    settings = TrainerSettings(
        revision=MODEL_REVISION,
        output_dir=str(tmp_path / "output"),
        study_variant="bc0",
        seed=3407,
    )

    def replace_authenticated_input(**_kwargs):
        train_file.write_text('{"row": "replacement"}\n', encoding="utf-8")
        return {"variant_id": "bc0"}

    with (
        mock.patch(
            "psse_env.sft.training._prepare_checkpoint_receipt_binding",
            side_effect=replace_authenticated_input,
        ),
        mock.patch("psse_env.sft.training._prepare_pilot") as prepare_pilot,
        mock.patch("psse_env.sft.training._load_model") as load_model,
        pytest.raises(GateError, match="Training dataset bytes changed"),
    ):
        run_lora_training(
            train_file=train_file,
            validation_file=validation_file,
            settings=settings,
        )
    prepare_pilot.assert_not_called()
    load_model.assert_not_called()


def test_training_flow_publishes_receipt_after_final_save(tmp_path: Path) -> None:
    events: list[str] = []

    class FakeModel:
        def save_pretrained(self, path: str) -> None:
            Path(path).mkdir(parents=True, exist_ok=True)
            events.append("model_saved")

    class FakeProcessor:
        def save_pretrained(self, path: str) -> None:
            events.append("processor_saved")

    class FakeTrainer:
        def __init__(
            self,
            model,
            args,
            train_dataset,
            eval_dataset,
            data_collator,
            processing_class=None,
        ) -> None:
            del args, train_dataset, eval_dataset, data_collator, processing_class
            self.model = model
            self.data_collator = None

        def train(self):
            events.append("trained")
            return SimpleNamespace(metrics={"train_loss": 1.0})

    processor = FakeProcessor()
    model = FakeModel()
    example = SimpleNamespace(expected_tool_call={"name": "run_wls"})
    binding = {"variant_id": "bc0"}
    snapshot = {
        "model_id": "unsloth/gemma-4-31B-it",
        "model_revision": MODEL_REVISION,
        "snapshot_path": "/verified/base",
    }
    settings = TrainerSettings(
        revision=MODEL_REVISION,
        output_dir=str(tmp_path / "output"),
        seed=3407,
        study_variant="bc0",
    )
    train_file = tmp_path / "train.jsonl"
    validation_file = tmp_path / "validation.jsonl"
    train_file.write_text('{"row": "train"}\n', encoding="utf-8")
    validation_file.write_text('{"row": "validation"}\n', encoding="utf-8")

    def write_base(**kwargs):
        assert events[-2:] == ["model_saved", "processor_saved"]
        events.append("base_attestation_written")
        path = tmp_path / "output" / "base_snapshot_attestation.json"
        path.write_text("{}\n", encoding="utf-8")
        return path

    def write_receipt(**kwargs):
        assert events[-1] == "base_attestation_written"
        assert kwargs["binding"] is binding
        assert kwargs["training_rng_attestation"] == (
            canonical_training_rng_attestation(
                variant_id="bc0",
                training_seed=3407,
            )
        )
        events.append("receipt_written")
        return tmp_path / "output" / "checkpoint_receipt.json"

    def seed_rng(seed: int):
        events.append("rng_seeded")
        assert seed == 3407
        return seeded

    def load_model(_settings):
        events.append("model_loaded")
        return model, snapshot

    def attach_adapter(*_args, **_kwargs):
        events.append("adapter_attached")
        return model, None

    def prepare_receipt_binding(**_kwargs):
        events.append("binding_validated")
        return binding

    fake_trl = SimpleNamespace(SFTTrainer=FakeTrainer)
    seeded = canonical_training_rng_attestation(
        variant_id="bc0",
        training_seed=3407,
    )["pre_model_construction"]
    with (
        mock.patch.dict(sys.modules, {"trl": fake_trl}),
        mock.patch(
            "psse_env.sft.training._prepare_pilot",
            return_value=(processor, [example], [example]),
        ) as prepare_pilot,
        mock.patch(
            "psse_env.sft.training._prepare_checkpoint_receipt_binding",
            side_effect=prepare_receipt_binding,
        ) as prepare_binding,
        mock.patch(
            "psse_env.sft.training._attest_training_accelerator",
            return_value=ACCELERATOR,
        ),
        mock.patch(
            "psse_env.sft.training._seed_training_rngs",
            side_effect=seed_rng,
        ) as seed_rngs,
        mock.patch("psse_env.sft.training._inspect_initial_adapter", return_value=None),
        mock.patch("psse_env.sft.training._load_model", side_effect=load_model),
        mock.patch(
            "psse_env.sft.training.infer_required_side_input_names",
            return_value=(),
        ),
        mock.patch(
            "psse_env.sft.training.ensure_required_side_inputs",
            side_effect=lambda examples, _required: examples,
        ),
        mock.patch(
            "psse_env.sft.training._attach_trainable_adapter",
            side_effect=attach_adapter,
        ),
        mock.patch(
            "psse_env.sft.training._snapshot_trainable_parameters",
            return_value={},
        ),
        mock.patch("psse_env.sft.training.run_training_smoke"),
        mock.patch(
            "psse_env.sft.training._restore_trainable_parameters",
            return_value={"performed": True},
        ),
        mock.patch("psse_env.sft.training.build_trl_config", return_value=object()),
        mock.patch("psse_env.sft.training._records_dataset", return_value=[]),
        mock.patch("psse_env.sft.training.AssistantOnlyCollator", return_value=object()),
        mock.patch("psse_env.sft.training.run_generation_tool_call_smoke"),
        mock.patch("psse_env.sft.training._normalize_adapter_base_reference"),
        mock.patch(
            "psse_env.sft.training._write_base_snapshot_attestation",
            side_effect=write_base,
        ),
        mock.patch(
            "psse_env.sft.training._write_checkpoint_receipt",
            side_effect=write_receipt,
        ) as receipt_writer,
    ):
        result = run_lora_training(
            train_file=train_file,
            validation_file=validation_file,
            settings=settings,
        )

    assert result.metrics == {"train_loss": 1.0}
    assert events == [
        "binding_validated",
        "rng_seeded",
        "model_loaded",
        "rng_seeded",
        "adapter_attached",
        "trained",
        "model_saved",
        "processor_saved",
        "base_attestation_written",
        "receipt_written",
    ]
    prepare_binding.assert_called_once()
    assert (
        prepare_binding.call_args.kwargs["input_snapshot"]
        is prepare_pilot.call_args.kwargs["input_snapshot"]
    )
    assert seed_rngs.call_args_list == [mock.call(3407), mock.call(3407)]
    receipt_writer.assert_called_once()
