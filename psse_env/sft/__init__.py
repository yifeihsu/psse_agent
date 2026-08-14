"""Fail-closed Gemma tool-SFT preparation and training gates.

The package deliberately keeps Transformers, PEFT, TRL, datasets, and torch as
runtime-only dependencies.  Dataset validation and fake-processor tests can be
imported without loading a model or importing the root training script.
"""

from .gates import (
    DatasetGateReport,
    GateError,
    GroupedPilotReport,
    LengthAudit,
    PreparedExample,
    audit_dataset,
    load_exact_processor,
    load_jsonl,
    parse_tool_call,
    prepare_example,
    validate_grouped_pilot,
    validate_current_tool_registry,
)
from .provenance import validate_release_gate_report
from .training import LoraSettings, TrainerSettings

__all__ = [
    "DatasetGateReport",
    "GateError",
    "GroupedPilotReport",
    "LengthAudit",
    "LoraSettings",
    "PreparedExample",
    "TrainerSettings",
    "audit_dataset",
    "load_exact_processor",
    "load_jsonl",
    "parse_tool_call",
    "prepare_example",
    "validate_grouped_pilot",
    "validate_release_gate_report",
    "validate_current_tool_registry",
]
