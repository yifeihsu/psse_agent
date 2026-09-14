"""Opt-in canonical protocol for raw-section logical topology experiments.

The legacy branch-row correction schema remains unchanged. This explicitly
versioned extension encodes a joint candidate in one action and one transaction.
"""
from __future__ import annotations

import copy
from psse_env.actions import CORRECT_TOPOLOGY, safe_normalize_action
from logical_topology.atomic import validate_atomic_arguments
from . import protocol_bridge as legacy

CONTRACT = "ieee57_logical_protocol_v1"
CORRECT_LOGICAL_TOPOLOGY = "correct_logical_topology_from_context"
ALLOWED_INTERNAL = {"run_wls", "verify_candidate", "get_topology_context", "correct_topology",
                    "commit_state", "rollback_state", "finalize_diagnosis", "ask_for_more_evidence"}


def internal_to_canonical_action(action):
    normalized = safe_normalize_action(action)
    if normalized["tool"] not in ALLOWED_INTERNAL:
        raise ValueError("unsupported tool in pure logical protocol")
    if normalized["tool"] != CORRECT_TOPOLOGY:
        return legacy.internal_to_canonical_action(normalized)
    arguments = copy.deepcopy(normalized["arguments"])
    validate_atomic_arguments(arguments)
    arguments["case_path"] = arguments.pop("state_id")
    return {"tool": CORRECT_LOGICAL_TOPOLOGY, "arguments": arguments}


def canonical_to_internal_action(action):
    normalized = safe_normalize_action(action)
    allowed_canonical = {legacy.INTERNAL_TO_CANONICAL_TOOL[tool] for tool in ALLOWED_INTERNAL if tool != CORRECT_TOPOLOGY}
    allowed_canonical.add(CORRECT_LOGICAL_TOPOLOGY)
    if normalized["tool"] not in allowed_canonical:
        raise ValueError("tool is outside the explicit logical canonical registry")
    if normalized["tool"] == CORRECT_LOGICAL_TOPOLOGY:
        arguments = copy.deepcopy(normalized["arguments"])
        validate_atomic_arguments(arguments, state_key="case_path")
        arguments["state_id"] = arguments.pop("case_path")
        return {"tool": CORRECT_TOPOLOGY, "arguments": arguments}
    result = legacy.canonical_to_internal_action(normalized)
    if result["tool"] not in ALLOWED_INTERNAL or result["tool"] == CORRECT_TOPOLOGY:
        raise ValueError("unsupported tool in pure logical protocol")
    return result


def logical_tool_schemas():
    allowed = {legacy.INTERNAL_TO_CANONICAL_TOOL[tool] for tool in ALLOWED_INTERNAL if tool != CORRECT_TOPOLOGY}
    schemas = [copy.deepcopy(row) for row in legacy.unified_tool_schemas() if row["function"]["name"] in allowed]
    schemas.append({"type": "function", "function": {
        "name": CORRECT_LOGICAL_TOPOLOGY,
        "description": "Create one atomic topology candidate. Submit exactly all desired logical switch statuses and the candidate/certificate hash from the fresh context. Verification and commit are separate actions.",
        "parameters": {"type": "object", "additionalProperties": False,
            "properties": {"case_path": {"type": "string"}, "candidate_id": {"type": "string"},
                "certificate_hash": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                "desired_statuses": {"type": "object", "minProperties": 1,
                    "additionalProperties": {"type": "integer", "enum": [0, 1]}}},
            "required": ["case_path", "candidate_id", "certificate_hash", "desired_statuses"]}}})
    return schemas
