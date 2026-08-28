from __future__ import annotations

import json

import pytest

from research.train import (
    ExposureCountingCollator,
    TRAINING_STEP_EXPOSURE_KEY,
    TrainingStepExposureCounter,
    canonical_json_sha256,
    finalize_restart_checkpoint,
    pop_training_step_exposure,
    sampled_batches_after_updates,
    sampled_rows_after_updates,
    summarize_prepared_rows,
    validate_restart_checkpoint,
    validate_sampled_exposure,
)


def test_summarize_prepared_rows_counts_only_nonmasked_targets() -> None:
    summary = summarize_prepared_rows(
        [
            {"labels": [-100, -100, 10, 11]},
            {"labels": [-100, 20, 21, 22]},
        ]
    )

    assert summary == {
        "rows": 2,
        "supervised_tokens": 5,
        "mean_supervised_tokens_per_row": 2.5,
        "min_supervised_tokens_per_row": 2,
        "max_supervised_tokens_per_row": 3,
    }


def test_summarize_prepared_rows_handles_empty_input() -> None:
    assert summarize_prepared_rows([]) == {
        "rows": 0,
        "supervised_tokens": 0,
        "mean_supervised_tokens_per_row": None,
        "min_supervised_tokens_per_row": None,
        "max_supervised_tokens_per_row": None,
    }


def test_exposure_collator_counts_sampled_rows_and_nonmasked_tokens() -> None:
    class Processor:
        pad_token_id = 0

    collator = ExposureCountingCollator(Processor())
    batch = collator(
        [
            {
                "input_ids": [1, 2, 3],
                "attention_mask": [1, 1, 1],
                "labels": [-100, 2, 3],
                "_exposure_split": "train",
            },
            {
                "input_ids": [4, 5],
                "attention_mask": [1, 1],
                "labels": [-100, 5],
                "_exposure_split": "train",
            },
        ]
    )

    assert tuple(batch["input_ids"].shape) == (2, 3)
    assert collator.summary("train") == {
        "batches": 1,
        "rows": 2,
        "input_tokens": 5,
        "supervised_tokens": 3,
    }
    assert collator.summary("validation")["rows"] == 0


def test_exposure_collator_keeps_training_metadata_out_of_validation() -> None:
    class Processor:
        pad_token_id = 0

    collator = ExposureCountingCollator(Processor())
    batch = collator(
        [
            {
                "input_ids": [1, 2],
                "attention_mask": [1, 1],
                "labels": [-100, 2],
                "_exposure_split": "validation",
            }
        ]
    )

    assert TRAINING_STEP_EXPOSURE_KEY not in batch
    assert collator.summary("validation")["rows"] == 1


def test_training_step_exposure_counts_only_committed_batches() -> None:
    class Processor:
        pad_token_id = 0

    counter = TrainingStepExposureCounter()
    collator = ExposureCountingCollator(Processor())
    first_batch = collator(
        [
            {
                "input_ids": [1, 2, 3],
                "attention_mask": [1, 1, 1],
                "labels": [-100, 2, 3],
                "_exposure_split": "train",
            },
            {
                "input_ids": [4, 5],
                "attention_mask": [1, 1],
                "labels": [-100, 5],
                "_exposure_split": "train",
            },
        ]
    )
    prefetched_batch = collator(
        [
            {
                "input_ids": [7, 8, 9],
                "attention_mask": [1, 1, 1],
                "labels": [-100, 8, 9],
                "_exposure_split": "train",
            }
        ]
    )

    first = pop_training_step_exposure(first_batch)
    counter.add(first)

    assert TRAINING_STEP_EXPOSURE_KEY not in first_batch
    assert prefetched_batch[TRAINING_STEP_EXPOSURE_KEY]["rows"] == 1
    assert counter.summary() == {
        "batches": 1,
        "rows": 2,
        "input_tokens": 5,
        "supervised_tokens": 3,
    }


def test_exposure_counters_restore_persisted_restart_ledger() -> None:
    class Processor:
        pad_token_id = 0

    initial = {
        "batches": 8,
        "rows": 8,
        "input_tokens": 80,
        "supervised_tokens": 24,
    }
    counter = TrainingStepExposureCounter(initial)
    collator = ExposureCountingCollator(
        Processor(), {"train": initial, "validation": {**initial, "rows": 2}}
    )

    counter.add(
        {"batches": 1, "rows": 1, "input_tokens": 10, "supervised_tokens": 3}
    )

    assert counter.summary() == {
        "batches": 9,
        "rows": 9,
        "input_tokens": 90,
        "supervised_tokens": 27,
    }
    assert collator.summary("train") == initial
    assert collator.summary("validation")["rows"] == 2


def test_sampled_row_schedule_accounts_for_short_epoch_tail() -> None:
    values = [
        sampled_rows_after_updates(
            train_rows=10,
            updates=step,
            batch_size=1,
            gradient_accumulation_steps=4,
        )
        for step in range(7)
    ]

    assert values == [0, 4, 8, 10, 14, 18, 20]
    assert [
        sampled_batches_after_updates(
            train_rows=10,
            updates=step,
            batch_size=1,
            gradient_accumulation_steps=4,
        )
        for step in range(7)
    ] == [0, 4, 8, 10, 14, 18, 20]

    validate_sampled_exposure(
        {"batches": 18, "rows": 18},
        train_rows=10,
        updates=5,
        batch_size=1,
        gradient_accumulation_steps=4,
    )
    with pytest.raises(ValueError, match="disagrees"):
        validate_sampled_exposure(
            {"batches": 18, "rows": 17},
            train_rows=10,
            updates=5,
            batch_size=1,
            gradient_accumulation_steps=4,
        )


def test_completed_restart_checkpoint_is_bound_and_tamper_evident(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint-25"
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": 25}), encoding="utf-8"
    )
    for name in (
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
        "adapter_config.json",
        "adapter_model.safetensors",
    ):
        (checkpoint / name).write_bytes(name.encode())
    binding = {"schema": "test", "arm": "E2B-selective"}
    binding_sha256 = canonical_json_sha256(binding)
    ledger = {
        "schema": "research_exposure_checkpoint_v2",
        "global_step": 25,
        "run_binding": binding,
        "run_binding_sha256": binding_sha256,
        "training_step_train_exposure": {
            "batches": 100,
            "rows": 100,
            "input_tokens": 1000,
            "supervised_tokens": 300,
        },
        "collated_exposure": {},
    }

    finalize_restart_checkpoint(checkpoint, ledger)
    assert validate_restart_checkpoint(
        checkpoint, expected_run_binding_sha256=binding_sha256
    ) == ledger

    (checkpoint / "optimizer.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="file verification failed"):
        validate_restart_checkpoint(checkpoint)
