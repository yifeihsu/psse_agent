from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from research.exposure import summarize_dataset


def test_summarize_dataset_records_digest_failures_and_target_tokens(
    tmp_path: Path,
) -> None:
    source = tmp_path / "rows.jsonl"
    source.write_text(
        json.dumps({"example_id": "one"}) + "\n"
        + json.dumps({"example_id": "two"}) + "\n",
        encoding="utf-8",
    )
    prepared = [
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1], "labels": [-100, 2, 3]}
    ]
    with patch(
        "research.exposure.prepare_rows",
        return_value=(prepared, ["train[1]: rejected"]),
    ):
        report = summarize_dataset(
            source,
            object(),
            max_length=8192,
            label="train",
        )

    assert report["input_rows"] == 2
    assert report["prepared"]["rows"] == 1
    assert report["prepared"]["supervised_tokens"] == 2
    assert report["preparation_failure_count"] == 1
    assert report["preparation_failures"] == ["train[1]: rejected"]
    assert len(report["sha256"]) == 64
