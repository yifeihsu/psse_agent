"""Focused contracts for the non-release E2B preliminary pipeline."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest

from psse_env.dagger.preliminary_receipt import (
    BUILDER_SOURCE_ATTESTATION_CONTRACT,
    DATASET_ARTIFACT_TYPE,
    DATASET_RECEIPT_CONTRACT,
    EXPECTED_SPLITS,
    PINNED_MODEL_NAME,
    PINNED_MODEL_REVISION,
    PreliminaryReceiptError,
    ensure_preliminary_stage_plan,
    validate_preliminary_dataset_receipt,
    validate_preliminary_resume_checkpoint,
    validate_preliminary_stage_receipt,
    write_preliminary_stage_receipt,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = REPO_ROOT / "submit_preliminary_dagger_e2b.sh"
DOC = Path(__file__).with_name("PRELIMINARY_DAGGER_E2B.md")


def _stable_hash(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_dataset(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    split_roots = {
        "bc0_train": ["bc0-train"],
        "bc0_validation": ["bc0-validation"],
        "d1_train": ["d1-train"],
        "d1_validation": ["d1-validation"],
        "d1_test": ["d1-test"],
        "d1_eval_combined": ["d1-validation", "d1-test"],
        "mixed_train": ["bc0-train", "d1-train"],
    }
    outputs: dict[str, object] = {}
    splits: dict[str, object] = {}
    for split, filename in EXPECTED_SPLITS.items():
        rows = []
        for physical_root in split_roots[split]:
            row = {
                "physical_root_fingerprint": physical_root,
                "production_label_eligible": not (
                    split.startswith("d1_")
                    or (split == "mixed_train" and physical_root.startswith("d1-"))
                ),
                "messages": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "state"},
                    {"role": "assistant", "content": "action"},
                ],
            }
            rows.append(row)
        data = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
        path = root / filename
        path.write_text(data, encoding="utf-8", newline="\n")
        roots = sorted(split_roots[split])
        identity = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "row_count": len(rows),
            "physical_root_count": len(roots),
            "physical_root_set_sha256": _stable_hash(roots),
        }
        outputs[filename] = dict(identity)
        splits[split] = {
            "filename": filename,
            **identity,
            "physical_roots": roots,
            "distributions": {},
        }
    receipt = {
        "contract": DATASET_RECEIPT_CONTRACT,
        "artifact_type": DATASET_ARTIFACT_TYPE,
        "release_eligible": False,
        "release_ineligibility_reasons": ["diagnostic legacy rows"],
        "output_dir": "/untrusted/ignored/path",
        "source_commits": {
            "d0_generation": "a" * 40,
            "diagnostic_collection": "b" * 40,
            "builder": "c" * 40,
        },
        "builder_source_attestation": {
            "contract": BUILDER_SOURCE_ATTESTATION_CONTRACT,
            "source_commit": "c" * 40,
            "tracked_files_match_head": True,
            "tracked_files": {
                "psse_env/dagger/preliminary_dataset.py": {
                    "git_blob_oid": "d" * 40,
                    "sha256": "e" * 64,
                    "size_bytes": 1,
                },
                "scripts/build_preliminary_dagger_dataset.py": {
                    "git_blob_oid": "f" * 40,
                    "sha256": "0" * 64,
                    "size_bytes": 1,
                },
            },
        },
        "inputs": {},
        "outputs": outputs,
        "splits": splits,
        "audits": {
            "strict_failure_bundle_admission": {"passed": True},
            "d0_generation_binding": {"passed": True},
            "evaluation_root_reservation": {"passed": True},
            "evaluation_root_partition": {"passed": True},
            "canonical_d1_training_selection": {"passed": True},
            "root_disjointness": {
                "passed": True,
                "required_pairwise_overlaps": {
                    "bc0_train__bc0_validation": [],
                    "d1_train__d1_validation": [],
                    "d1_train__d1_test": [],
                    "d1_validation__d1_test": [],
                },
            },
            "content_composition": {
                "passed": True,
                "d1_validation_plus_test_row_multiset_sha256": _stable_hash(
                    sorted(
                        _stable_hash(row)
                        for split in ("d1_validation", "d1_test")
                        for row in [
                            json.loads(
                                (root / EXPECTED_SPLITS[split]).read_text(
                                    encoding="utf-8"
                                )
                            )
                        ]
                    )
                ),
                "d1_eval_combined_row_multiset_sha256": _stable_hash(
                    sorted(
                        _stable_hash(json.loads(line))
                        for line in (
                            root / EXPECTED_SPLITS["d1_eval_combined"]
                        ).read_text(encoding="utf-8").splitlines()
                        if line
                    )
                ),
                "bc0_plus_d1_train_row_multiset_sha256": _stable_hash(
                    sorted(
                        _stable_hash(row)
                        for split in ("bc0_train", "d1_train")
                        for row in [
                            json.loads(
                                (root / EXPECTED_SPLITS[split]).read_text(
                                    encoding="utf-8"
                                )
                            )
                        ]
                    )
                ),
                "mixed_train_row_multiset_sha256": _stable_hash(
                    sorted(
                        _stable_hash(json.loads(line))
                        for line in (
                            root / EXPECTED_SPLITS["mixed_train"]
                        ).read_text(encoding="utf-8").splitlines()
                        if line
                    )
                ),
            },
        },
        "parameters": {},
    }
    receipt_path = root / "preliminary.dataset_receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return receipt_path


def _write_hardware_attestation(root: Path) -> Path:
    path = root / "preliminary_hardware_attestation.json"
    payload = {
        "contract": "preliminary_dagger_hardware_attestation_v1",
        "artifact_type": "preliminary_dagger_nonrelease_hardware_attestation",
        "release_eligible": False,
        "accelerator_class": "l40s",
        "device_count": 1,
        "bf16_supported": True,
        "torch_cuda_version": "12.8",
        "required_accelerator_class": "l40s",
        "required_accelerator_class_matched": True,
        "devices": [
            {
                "index": 0,
                "name": "NVIDIA L40S",
                "total_memory_bytes": 46_068 * 1024**2,
                "compute_capability": [8, 9],
                "accelerator_class": "l40s",
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_run_config_from_plan(plan_path: Path, *, override: dict[str, object] | None = None) -> Path:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    training = plan["training_arguments"]
    initial_adapter = plan["initial_adapter"]
    if initial_adapter:
        initial_config = json.loads(
            (Path(initial_adapter) / "adapter_config.json").read_text(encoding="utf-8")
        )
        base_model = initial_config["base_model_name_or_path"]
    else:
        base_model = PINNED_MODEL_NAME
    payload: dict[str, object] = {
        "requested_model_name": PINNED_MODEL_NAME,
        "requested_model_revision": PINNED_MODEL_REVISION,
        "adapter_base_model_name_or_path": base_model,
        "init_adapter": initial_adapter or "",
        "output_dir": plan["output_dir"],
        "train_file": plan["train_file"],
        "valid_file": plan["validation_file"],
        "dataset_summary": {
            "raw_train_conversations": plan["optimizer_visible_train_rows"],
            "raw_validation_conversations": plan["evaluated_validation_rows"],
            "processed_train_samples": plan["optimizer_visible_train_rows"],
            "repeated_train_samples": plan["optimizer_visible_train_rows"],
            "processed_validation_samples": plan["evaluated_validation_rows"],
        },
        "sft_args": {
            "max_seq_length": training["max_seq_length"],
            "dataset_num_proc": training["dataset_num_proc"],
            "per_device_train_batch_size": training["batch_size"],
            "per_device_eval_batch_size": training["per_device_eval_batch_size"],
            "gradient_accumulation_steps": training[
                "gradient_accumulation_steps"
            ],
            "warmup_steps": training["warmup_steps"],
            "max_steps": training["max_steps"],
            "num_train_epochs": training["num_train_epochs"],
            "learning_rate": training["learning_rate"],
            "logging_steps": training["logging_steps"],
            "save_steps": training["save_steps"],
            "eval_steps": training["eval_steps"],
            "save_total_limit": training["save_total_limit"],
            "weight_decay": training["weight_decay"],
            "lr_scheduler_type": training["lr_scheduler_type"],
            "dataloader_num_workers": training["dataloader_workers"],
            "drop_too_long_targets": training["drop_too_long_targets"],
            "load_in_4bit": training["load_in_4bit"],
            "load_in_16bit": training["load_in_16bit"],
            "lora_r": training["lora_r"],
            "lora_alpha": training["lora_alpha"],
            "lora_dropout": training["lora_dropout"],
            "lora_target_scope": training["lora_target_scope"],
            "report_to": training["report_to"],
            "run_name": training["run_name"],
            "seed": training["training_seed"],
        },
        "prompt_args": {
            key: value
            for key, value in plan["prompt_arguments"].items()
            if key != "sanity_check_samples"
        },
    }
    if override:
        payload.update(override)
    path = Path(plan["output_dir"]) / "run_config.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_training_source_repo(root: Path) -> Path:
    if (root / ".git").is_dir():
        return root
    for relative in (
        "gemma_adapter_loader.py",
        "gpt_oss_power_sft_revised_v3.py",
        "hif_search_limits.py",
        "psse_env/dagger/preliminary_receipt.py",
        "psse_env/sft/preliminary_adapter.py",
        "psse_env/sft/preliminary_hardware.py",
        "psse_env/sft/release_hardware.py",
        "submit_preliminary_dagger_e2b.sh",
        "trace_protocol.py",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"test source for {relative}\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(
        ["git", "config", "user.email", "preliminary-test@example.invalid"],
        cwd=root,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Preliminary Test"],
        cwd=root,
        check=True,
    )
    subprocess.run(
        ["git", "config", "core.autocrlf", "false"], cwd=root, check=True
    )
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "test training sources"],
        cwd=root,
        check=True,
    )
    return root


def _make_stage_plan(
    dataset_receipt: Path,
    output: Path,
    *,
    stage: str,
    max_train_rows: int | None = None,
) -> tuple[Path, Path, Path, Path | None]:
    output.mkdir(parents=True, exist_ok=True)
    dataset_dir = dataset_receipt.parent
    source_repo = _write_training_source_repo(dataset_dir.parent / "source_repo")
    train_split = "bc0_train" if stage == "bc0" else "mixed_train"
    train = dataset_dir / EXPECTED_SPLITS[train_split]
    validation = dataset_dir / EXPECTED_SPLITS["d1_validation"]
    initial_adapter: Path | None = None
    if stage == "dagger":
        snapshot = dataset_dir / "snapshot" / PINNED_MODEL_REVISION
        snapshot.mkdir(parents=True)
        (snapshot / "config.json").write_text("{}\n", encoding="utf-8")
        initial_adapter = dataset_dir / "pinned_init_adapter"
        initial_adapter.mkdir()
        (initial_adapter / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": str(snapshot.resolve())}),
            encoding="utf-8",
        )
    plan = ensure_preliminary_stage_plan(
        stage=stage,
        dataset_receipt=dataset_receipt,
        train_file=train,
        validation_file=validation,
        output_dir=output,
        repo_root=source_repo,
        initial_adapter=initial_adapter,
        training_seed=3407,
        max_train_rows=(
            max_train_rows
            if max_train_rows is not None
            else (256 if stage == "bc0" else 512)
        ),
        max_valid_rows=128,
        max_steps=8,
        max_seq_length=4096,
        batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=0.0001,
        lora_r=16,
        lora_alpha=16,
        save_steps=8,
        eval_steps=8,
        save_total_limit=3,
        dataloader_workers=4,
        report_to="none",
    )
    return plan, train, validation, initial_adapter


def _write_completed_stage(
    dataset_receipt: Path,
    output: Path,
    *,
    stage: str,
) -> tuple[Path, Path, Path, Path]:
    plan, train, validation, initial_adapter = _make_stage_plan(
        dataset_receipt, output, stage=stage
    )
    _write_run_config_from_plan(plan)
    checkpoint = output / "checkpoint-8"
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text(
        json.dumps(
            {
                "global_step": 8,
                "log_history": [
                    {"eval_loss": 1.25, "epoch": 1.0, "step": 8}
                ],
            }
        ),
        encoding="utf-8",
    )
    adapter = output / "lora"
    adapter.mkdir()
    if stage == "dagger":
        assert initial_adapter is not None
        base_model = json.loads(
            (initial_adapter / "adapter_config.json").read_text(encoding="utf-8")
        )["base_model_name_or_path"]
    else:
        base_model = PINNED_MODEL_NAME
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": base_model}), encoding="utf-8"
    )
    hardware = _write_hardware_attestation(output)
    return plan, train, validation, hardware


def test_dataset_receipt_rehashes_all_splits_and_ignores_output_dir(
    tmp_path: Path,
) -> None:
    receipt = _write_dataset(tmp_path)
    result = validate_preliminary_dataset_receipt(receipt)
    assert result["release_eligible"] is False
    assert set(result["paths"]) == set(EXPECTED_SPLITS)
    assert all(Path(path).parent == tmp_path for path in result["paths"].values())


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("contract", "production_receipt_v1", "contract"),
        ("artifact_type", "release_dataset", "artifact_type"),
        ("release_eligible", True, "release_eligible=false"),
    ),
)
def test_dataset_receipt_rejects_release_or_wrong_contract(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    receipt_path = _write_dataset(tmp_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt[field] = value
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(PreliminaryReceiptError, match=message):
        validate_preliminary_dataset_receipt(receipt_path)


def test_dataset_receipt_rejects_tampered_bytes(tmp_path: Path) -> None:
    receipt = _write_dataset(tmp_path)
    path = tmp_path / EXPECTED_SPLITS["d1_train"]
    path.write_text(path.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    with pytest.raises(PreliminaryReceiptError, match="SHA-256"):
        validate_preliminary_dataset_receipt(receipt)


def test_dataset_receipt_rejects_forged_heldout_root_overlap(tmp_path: Path) -> None:
    receipt_path = _write_dataset(tmp_path)
    validation_path = tmp_path / EXPECTED_SPLITS["d1_validation"]
    row = json.loads(validation_path.read_text(encoding="utf-8"))
    row["physical_root_fingerprint"] = "d1-test"
    validation_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(validation_path.read_bytes()).hexdigest()
    root_hash = _stable_hash(["d1-test"])
    split = receipt["splits"]["d1_validation"]
    split.update(
        {
            "sha256": digest,
            "physical_roots": ["d1-test"],
            "physical_root_set_sha256": root_hash,
        }
    )
    receipt["outputs"][EXPECTED_SPLITS["d1_validation"]].update(
        {"sha256": digest, "physical_root_set_sha256": root_hash}
    )
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(PreliminaryReceiptError, match="physical roots overlap"):
        validate_preliminary_dataset_receipt(receipt_path)


def test_dataset_receipt_rejects_same_root_content_substitution(tmp_path: Path) -> None:
    receipt_path = _write_dataset(tmp_path)
    mixed_path = tmp_path / EXPECTED_SPLITS["mixed_train"]
    rows = [json.loads(line) for line in mixed_path.read_text(encoding="utf-8").splitlines()]
    rows[1]["messages"][2]["content"] = "substituted-target"
    mixed_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(mixed_path.read_bytes()).hexdigest()
    receipt["splits"]["mixed_train"]["sha256"] = digest
    receipt["outputs"][EXPECTED_SPLITS["mixed_train"]]["sha256"] = digest
    actual_multiset = _stable_hash(sorted(_stable_hash(row) for row in rows))
    receipt["audits"]["content_composition"][
        "mixed_train_row_multiset_sha256"
    ] = actual_multiset
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(PreliminaryReceiptError, match="rows must exactly equal"):
        validate_preliminary_dataset_receipt(receipt_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("tracked_files_match_head", False, "tracked_files_match_head"),
        ("source_commit", "9" * 40, "must equal"),
    ),
)
def test_dataset_receipt_rejects_invalid_builder_source_attestation(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    receipt_path = _write_dataset(tmp_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["builder_source_attestation"][field] = value
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(PreliminaryReceiptError, match=message):
        validate_preliminary_dataset_receipt(receipt_path)


def test_stage_plan_refuses_foreign_checkpoint_without_binding(
    tmp_path: Path,
) -> None:
    dataset_receipt = _write_dataset(tmp_path / "dataset")
    output = tmp_path / "output"
    stale = output / "checkpoint-8"
    stale.mkdir(parents=True)
    (stale / "trainer_state.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(PreliminaryReceiptError, match="without an immutable stage plan"):
        _make_stage_plan(dataset_receipt, output, stage="bc0")


def test_stage_plan_rejects_dirty_training_source(tmp_path: Path) -> None:
    dataset_receipt = _write_dataset(tmp_path / "dataset")
    source_repo = _write_training_source_repo(tmp_path / "source_repo")
    source = source_repo / "psse_env" / "sft" / "preliminary_hardware.py"
    source.write_text("changed after commit\n", encoding="utf-8")
    with pytest.raises(PreliminaryReceiptError, match="differs from HEAD"):
        _make_stage_plan(dataset_receipt, tmp_path / "output", stage="bc0")


def test_dagger_stage_plan_requires_d1_in_optimizer_visible_prefix(
    tmp_path: Path,
) -> None:
    dataset_receipt = _write_dataset(tmp_path / "dataset")
    with pytest.raises(PreliminaryReceiptError, match="at least one D1 row/root"):
        _make_stage_plan(
            dataset_receipt,
            tmp_path / "output",
            stage="dagger",
            max_train_rows=1,
        )


def test_resume_rejects_run_config_that_differs_from_stage_plan(
    tmp_path: Path,
) -> None:
    dataset_receipt = _write_dataset(tmp_path / "dataset")
    output = tmp_path / "output"
    plan, _, _, _ = _make_stage_plan(dataset_receipt, output, stage="bc0")
    _write_run_config_from_plan(plan, override={"requested_model_revision": "0" * 40})
    checkpoint = output / "checkpoint-8"
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": 8}), encoding="utf-8"
    )
    with pytest.raises(PreliminaryReceiptError, match="model_revision"):
        validate_preliminary_resume_checkpoint(
            stage_plan=plan, checkpoint=checkpoint
        )


def test_stage_receipt_requires_finite_final_step_eval_loss(tmp_path: Path) -> None:
    dataset_receipt = _write_dataset(tmp_path / "dataset")
    output = tmp_path / "output"
    plan, train, validation, hardware = _write_completed_stage(
        dataset_receipt, output, stage="bc0"
    )
    state_path = output / "checkpoint-8" / "trainer_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["log_history"][0]["step"] = 7
    state_path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(PreliminaryReceiptError, match="final max step"):
        write_preliminary_stage_receipt(
            stage="bc0",
            dataset_receipt=dataset_receipt,
            train_file=train,
            validation_file=validation,
            output_dir=output,
            hardware_attestation=hardware,
            stage_plan=plan,
        )


def test_stage_receipt_rejects_preprocessing_that_drops_bound_train_rows(
    tmp_path: Path,
) -> None:
    dataset_receipt = _write_dataset(tmp_path / "dataset")
    output = tmp_path / "output"
    plan, train, validation, hardware = _write_completed_stage(
        dataset_receipt, output, stage="dagger"
    )
    config_path = output / "run_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["dataset_summary"]["processed_train_samples"] = 0
    config["dataset_summary"]["repeated_train_samples"] = 0
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(PreliminaryReceiptError, match="processed train rows"):
        write_preliminary_stage_receipt(
            stage="dagger",
            dataset_receipt=dataset_receipt,
            train_file=train,
            validation_file=validation,
            output_dir=output,
            hardware_attestation=hardware,
            stage_plan=plan,
        )


def test_stage_plan_rejects_bc0_validation_instead_of_common_d1_validation(
    tmp_path: Path,
) -> None:
    dataset_receipt = _write_dataset(tmp_path / "dataset")
    output = tmp_path / "output"
    output.mkdir()
    source_repo = _write_training_source_repo(tmp_path / "source_repo")
    with pytest.raises(PreliminaryReceiptError, match="held-out D1 validation"):
        ensure_preliminary_stage_plan(
            stage="bc0",
            dataset_receipt=dataset_receipt,
            train_file=tmp_path / "dataset" / EXPECTED_SPLITS["bc0_train"],
            validation_file=(
                tmp_path / "dataset" / EXPECTED_SPLITS["bc0_validation"]
            ),
            output_dir=output,
            repo_root=source_repo,
            initial_adapter=None,
            training_seed=3407,
            max_train_rows=256,
            max_valid_rows=128,
            max_steps=8,
            max_seq_length=4096,
            batch_size=2,
            gradient_accumulation_steps=8,
            learning_rate=0.0001,
            lora_r=16,
            lora_alpha=16,
            save_steps=8,
            eval_steps=8,
            save_total_limit=3,
            dataloader_workers=4,
            report_to="none",
        )


def test_stage_receipt_is_write_once_and_binds_adapter_tree(tmp_path: Path) -> None:
    dataset_receipt = _write_dataset(tmp_path / "dataset")
    output = tmp_path / "output"
    plan, train, validation, hardware = _write_completed_stage(
        dataset_receipt, output, stage="bc0"
    )
    receipt_path = write_preliminary_stage_receipt(
        stage="bc0",
        dataset_receipt=dataset_receipt,
        train_file=train,
        validation_file=validation,
        output_dir=output,
        hardware_attestation=hardware,
        stage_plan=plan,
    )
    receipt_mode = receipt_path.stat().st_mode & 0o777
    if os.name == "nt":
        assert receipt_mode & 0o222 == 0
    else:
        assert receipt_mode == 0o400
    result = validate_preliminary_stage_receipt(
        stage="bc0",
        dataset_receipt=dataset_receipt,
        train_file=train,
        validation_file=validation,
        output_dir=output,
        hardware_attestation=hardware,
        stage_plan=plan,
    )
    assert result["release_eligible"] is False
    with pytest.raises(PreliminaryReceiptError, match="already exists"):
        write_preliminary_stage_receipt(
            stage="bc0",
            dataset_receipt=dataset_receipt,
            train_file=train,
            validation_file=validation,
            output_dir=output,
            hardware_attestation=hardware,
            stage_plan=plan,
        )


def test_stage_receipt_rejects_post_completion_adapter_tampering(
    tmp_path: Path,
) -> None:
    dataset_receipt = _write_dataset(tmp_path / "dataset")
    output = tmp_path / "output"
    plan, train, validation, hardware = _write_completed_stage(
        dataset_receipt, output, stage="dagger"
    )
    write_preliminary_stage_receipt(
        stage="dagger",
        dataset_receipt=dataset_receipt,
        train_file=train,
        validation_file=validation,
        output_dir=output,
        hardware_attestation=hardware,
        stage_plan=plan,
    )
    config = output / "lora" / "adapter_config.json"
    config.write_text(
        '{"base_model_name_or_path": "tampered"}\n', encoding="utf-8"
    )
    with pytest.raises(PreliminaryReceiptError, match="exact pinned"):
        validate_preliminary_stage_receipt(
            stage="dagger",
            dataset_receipt=dataset_receipt,
            train_file=train,
            validation_file=validation,
            output_dir=output,
            hardware_attestation=hardware,
            stage_plan=plan,
        )


def test_launcher_is_pinned_bounded_hardware_attested_and_nonrelease() -> None:
    launcher = LAUNCHER.read_text(encoding="utf-8")
    for contract in (
        'readonly MODEL_NAME="unsloth/gemma-4-E2B-it"',
        'readonly MODEL_REVISION="f0c5915f17ad6c66dbeb577fb06ff8925bf8d7ae"',
        '#SBATCH --constraint="h200|h100|rtx6000|l40s"',
        "psse_env.sft.preliminary_hardware",
        "auto|h100|h200|rtx6000|l40s",
        'if [[ "$ACTUAL_ACCELERATOR_CLASS" == "l40s" ]]',
        "PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-2}",
        "GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-8}",
        "BC0_MAX_STEPS=${BC0_MAX_STEPS:-24}",
        "DAGGER_MAX_STEPS=${DAGGER_MAX_STEPS:-40}",
        "bounded_uint BC0_MAX_STEPS \"$BC0_MAX_STEPS\" 1 64",
        "bounded_uint DAGGER_MAX_STEPS \"$DAGGER_MAX_STEPS\" 1 128",
        "preliminary.dataset_receipt.json",
        "preliminary.bc0_train.jsonl",
        "preliminary.mixed_train.jsonl",
        "preliminary.d1_validation.jsonl",
        "preliminary.d1_test.jsonl",
        "psse_env.sft.preliminary_adapter",
        "pinned_bc0_init_adapter",
    ):
        assert contract in launcher
    assert "unsloth/gemma-4-31B-it" not in launcher
    assert "submit_dagger_sft_round0.sh" not in launcher


def test_launcher_passes_recovery_compatible_prompt_contract_to_both_stages() -> None:
    launcher = LAUNCHER.read_text(encoding="utf-8")
    for argument in (
        "--preserve-system-text",
        "--no-phase-gated-prompt",
        "--sanity-check-samples 0",
    ):
        assert argument in launcher
        assert launcher.count(argument) == 1
    assert launcher.count('"${COMMON_ARGS[@]}"') == 2
    assert '--init-adapter "$PINNED_INIT_ADAPTER"' in launcher
    assert '--train-file "$MIXED_TRAIN_FILE"' in launcher
    assert launcher.count('--valid-file "$D1_VALIDATION_FILE"') == 2
    # One immutable-plan binding plus one argument in each Trainer invocation.
    assert launcher.count('--max-valid-rows "$D1_MAX_VALID_ROWS"') == 3
    assert '--valid-file "$BC0_VALIDATION_FILE"' not in launcher
    assert '--train-file "$D1_TEST_FILE"' not in launcher
    assert '--valid-file "$D1_TEST_FILE"' not in launcher


def test_launcher_has_write_once_stage_receipts_and_explicit_checkpoint_resume() -> None:
    launcher = LAUNCHER.read_text(encoding="utf-8")
    for contract in (
        "preliminary_stage_receipt.json",
        "preliminary_hardware_attestation.json",
        "stage-check",
        "stage-write",
        "stage-plan",
        "resume-check",
        "preliminary_stage_plan.json",
        '--repo-root "$REPO_ROOT"',
        "latest_checkpoint",
        "unreceipted lora tree",
        'resume_args+=(--resume-from-checkpoint "$resume_checkpoint")',
        "scontrol requeue",
        'if kill -0 "$TRAIN_PID" 2>/dev/null; then',
    ):
        assert contract in launcher
    assert "--resume-from-checkpoint auto" not in launcher


def test_launcher_parses_and_doc_keeps_claim_boundary() -> None:
    subprocess.run(
        ["bash", "-n"],
        cwd=REPO_ROOT,
        input=LAUNCHER.read_bytes().replace(b"\r\n", b"\n"),
        check=True,
        capture_output=True,
    )
    documentation = DOC.read_text(encoding="utf-8")
    for contract in (
        "never release eligible",
        "optimizer-visible",
        "closed-loop",
        "RTX Pro 6000",
        "NVIDIA L40S",
        "effective",
        "ALLOW_DOWNLOAD=1",
    ):
        assert contract in documentation
