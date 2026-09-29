"""Gemma 4 tool-call SFT: row validation, prompt rendering and research LoRA training.

Transformers, PEFT, TRL, datasets and torch stay runtime-only dependencies, so
dataset validation and the fake-processor tests import without loading a model.
"""

from .gates import (
    DatasetGateReport,
    GateError,
    LengthAudit,
    PreparedExample,
    audit_dataset,
    load_exact_processor,
    load_jsonl,
    parse_tool_call,
    prepare_example,
    validate_current_tool_registry,
)
from .training import LoraSettings, TrainerSettings, validate_training_seed

__all__ = [
    "DatasetGateReport",
    "GateError",
    "LengthAudit",
    "LoraSettings",
    "PreparedExample",
    "TrainerSettings",
    "audit_dataset",
    "load_exact_processor",
    "load_jsonl",
    "parse_tool_call",
    "prepare_example",
    "validate_training_seed",
    "validate_current_tool_registry",
]
