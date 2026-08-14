"""Pretokenized assistant-only SFT collator."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .gates import GateError, PreparedExample


def resolve_pad_token_id(processor: Any) -> int:
    for candidate in (processor, getattr(processor, "tokenizer", None)):
        if candidate is None:
            continue
        pad = getattr(candidate, "pad_token_id", None)
        if pad is not None:
            return int(pad)
        eos = getattr(candidate, "eos_token_id", None)
        if eos is not None:
            return int(eos)
    raise GateError("Processor/tokenizer exposes neither pad_token_id nor eos_token_id.")


class AssistantOnlyCollator:
    """Pad pretokenized rows while preserving ``-100`` outside the target."""

    def __init__(self, processor: Any) -> None:
        self.pad_token_id = resolve_pad_token_id(processor)

    def __call__(self, features: Sequence[PreparedExample | Mapping[str, Any]]) -> dict[str, Any]:
        try:
            import torch
        except Exception as exc:  # pragma: no cover - depends on training environment.
            raise GateError(f"torch is required for collation: {exc}") from exc
        if not features:
            raise GateError("AssistantOnlyCollator received an empty batch.")

        rows = [feature.model_record() if isinstance(feature, PreparedExample) else dict(feature) for feature in features]
        maximum = max(len(row["input_ids"]) for row in rows)
        keys = {"input_ids", "attention_mask", "labels"}
        for row in rows:
            keys.update(key for key in row if key.endswith("token_type_ids"))
        batches: dict[str, list[list[int]]] = {key: [] for key in keys}
        for row in rows:
            input_ids = list(row["input_ids"])
            attention = list(row.get("attention_mask", [1] * len(input_ids)))
            labels = list(row["labels"])
            if len(input_ids) != len(attention) or len(input_ids) != len(labels):
                raise GateError("Collator row has unaligned input_ids, attention_mask, or labels.")
            padding = maximum - len(input_ids)
            batches["input_ids"].append(input_ids + [self.pad_token_id] * padding)
            batches["attention_mask"].append(attention + [0] * padding)
            batches["labels"].append(labels + [-100] * padding)
            for key in keys - {"input_ids", "attention_mask", "labels"}:
                values = list(row.get(key, [0] * len(input_ids)))
                if len(values) != len(input_ids):
                    raise GateError(f"Collator row has unaligned {key}.")
                batches[key].append(values + [0] * padding)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batches.items()}
