from __future__ import annotations

from research.train import ExposureCountingCollator, summarize_prepared_rows


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
