"""Fail-closed identity and performance gate for closed-loop evaluations.

The evaluator's scalar score is intentionally ignored here.  An artifact may
be considered for BC0 training or checkpoint promotion only after its immutable
identity, frozen-suite binding, deployment-environment attestation, strict
episode audits, and every configured hard constraint have been validated.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from psse_env.actions import (
    CORRECT_MEASUREMENTS,
    DIAGNOSTIC_TOOLS,
    INVALID_ACTION,
    RUN_WLS,
    VERIFY_CANDIDATE,
    action_signature,
    invalid_action,
)
from psse_env.dagger.dataset_builder import TOOL_JSON_SCHEMAS
from psse_env.dagger.evaluator import (
    evaluation_intervention_contract,
    fingerprint_evaluation_suites,
    load_evaluation_suites,
    trace_progress_advanced,
    validate_release_scenario_suites,
)
from psse_env.dagger.protocol_bridge import (
    INTERNAL_TO_CANONICAL_TOOL,
    unified_tool_schemas,
)
from psse_env.sft.gates import GateError, _validate_json_instance
from psse_env.sft.provenance import file_sha256, git_source_state, stable_json_sha256
from psse_env.sft.release_hardware import normalize_accelerator_class


DEFAULT_POLICY_PATH = Path(__file__).with_name("bc0_evaluation_policy.json")
DEFAULT_POLICY_ID = "bc0_closed_loop_hard_gate_v2"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-fA-F]{40}\Z")
_REVISION = re.compile(r"(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})\Z")
_HARD_CONSTRAINTS = frozenset(
    {
        "maximum_false_commit_count",
        "maximum_false_finalization_count",
        "maximum_false_rollback_count",
        "maximum_healthy_component_corruption_episodes",
        "maximum_invalid_action_rate",
        "maximum_invalid_actions_per_episode",
        "maximum_loop_episode_rate",
        "maximum_steps_per_episode",
        "minimum_terminal_rate",
    }
)
_FACTORY_APPROVAL_ROLES = frozenset(
    {"environment", "expert_policy", "model_policy", "case_loader"}
)
_VALIDATION_ROLES = frozenset(
    {"expert-baseline", "base-baseline", "checkpoint-promotion"}
)
_ROLE_POLICY = {
    "expert-baseline": "teacher_release",
    "base-baseline": "identity_and_measurement_only",
    "checkpoint-promotion": "bc0_promotion",
}
_SUITE_POLICY_FIELDS = frozenset(
    {
        "status",
        "approved_suite_sha256",
        "approved_suite_manifest",
        "required_suites",
        "evaluator_seed",
        "max_steps",
        "minimum_physical_roots_per_suite",
        "scenario_schema_version",
    }
)
_SUITE_MANIFEST_FIELDS = frozenset(
    {
        "suite_manifest",
        "suite_content_hashes",
        "suite_root_set_hashes",
        "suite_content_sha256",
        "root_set_sha256",
    }
)
_STRICT_HEALTHY_CHECKS = (
    "healthy_measurements_preserved",
    "healthy_case_components_preserved",
)
_STRICT_NONREGRESSION_CHECK = "accepted_target_nonregression"
_REQUIRED_EPISODE_FIELDS = frozenset(
    {
        "episode_key",
        "scenario_id",
        "seed",
        "suite",
        "family",
        "cardinality",
        "case",
        "split",
        "source_tier",
        "physical_root",
        "steps",
        "policy_steps",
        "terminal",
        "terminal_outcome",
        "final_physical_success",
        "physical_correctness_known",
        "final_physical_correct",
        "healthy_preservation_known",
        "healthy_components_preserved",
        "false_commit_count",
        "false_rollback_count",
        "false_finalization_count",
        "partial_fix_count",
        "retained_partial_fix_count",
        "invalid_action_count",
        "recovered_invalid_action_count",
        "loop_detected",
        "wls_calls",
        "specialized_tool_calls",
        "tool_regret_total",
        "tool_regret_samples",
        "evaluation_intervention",
        "trace",
        "evaluator_error",
        "release_environment_attestation",
        "policy_identity_attestation",
        "audit",
    }
)


@dataclass(frozen=True)
class EvaluationGateResult:
    passed: bool
    failures: tuple[str, ...]
    validation_role: str
    evidence_passed: bool
    performance_passed: bool
    performance_enforced: bool
    evidence_failures: tuple[str, ...]
    performance_failures: tuple[str, ...]
    comparison_required: bool
    comparison_passed: bool | None
    comparison_failures: tuple[str, ...]
    artifact_path: str | None
    artifact_sha256: str | None
    artifact_content_sha256: str | None
    source_commit: str | None
    frozen_suite_sha256: str | None
    protocol: str | None
    registry_sha256: str | None
    evaluated_policy_identity: dict[str, Any]
    gate_policy_id: str | None
    gate_policy_sha256: str | None
    observed: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def current_registry_sha256(protocol: str = "canonical") -> str:
    normalized = str(protocol).strip().lower()
    if normalized == "canonical":
        registry = unified_tool_schemas()
    elif normalized == "controller":
        registry = copy.deepcopy(TOOL_JSON_SCHEMAS)
    else:
        raise ValueError("protocol must be canonical or controller")
    return stable_json_sha256(registry)


def _function_schema_map(
    registry: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    schemas: dict[str, Mapping[str, Any]] = {}
    for raw_schema in registry:
        function = raw_schema.get("function") if isinstance(raw_schema, Mapping) else None
        if not isinstance(function, Mapping):
            continue
        name = function.get("name")
        parameters = function.get("parameters")
        if isinstance(name, str) and isinstance(parameters, Mapping):
            schemas[name] = parameters
    return schemas


_TRACE_CONTROLLER_SCHEMAS = _function_schema_map(TOOL_JSON_SCHEMAS)
_TRACE_CANONICAL_SCHEMAS = _function_schema_map(unified_tool_schemas())
_TRACE_TARGET_ONLY_MEASUREMENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "state_id": copy.deepcopy(
            _TRACE_CONTROLLER_SCHEMAS[CORRECT_MEASUREMENTS]["properties"]["state_id"]
        ),
        "suspect_group": copy.deepcopy(
            _TRACE_CANONICAL_SCHEMAS["correct_measurements_from_path"]["properties"]
            ["suspect_group"]
        ),
    },
    "required": ["state_id", "suspect_group"],
    "additionalProperties": False,
}
_INVALID_ACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "error_code": {"type": "string"},
        "error_detail": {"type": "string"},
    },
    "required": ["error_code"],
    "additionalProperties": False,
}
_TRACE_PROGRESS_FIELDS = frozenset(
    {
        "state_before",
        "state_after",
        "state_before_sha256",
        "state_after_sha256",
        "state_mutated",
        "terminal_after",
    }
)


def _trace_action_schema_failure(
    action: Mapping[str, Any], *, index: int
) -> str | None:
    """Validate one executed action against either registered protocol form."""

    tool = str(action.get("tool") or "")
    arguments = action.get("arguments")
    if tool == INVALID_ACTION:
        parameters = _INVALID_ACTION_SCHEMA
        try:
            _validate_json_instance(
                arguments,
                parameters,
                path=f"episode trace[{index}].action.arguments",
            )
        except GateError as exc:
            return str(exc)
        return None

    controller_parameters = _TRACE_CONTROLLER_SCHEMAS.get(tool)
    canonical_tool = INTERNAL_TO_CANONICAL_TOOL.get(tool)
    if (
        controller_parameters is None
        or canonical_tool not in _TRACE_CANONICAL_SCHEMAS
    ):
        return (
            f"episode trace[{index}] action tool {tool!r} is not in the "
            "unified release registry"
        )
    try:
        _validate_json_instance(
            arguments,
            controller_parameters,
            path=f"episode trace[{index}].action.arguments",
        )
        return None
    except GateError as controller_failure:
        # The deployment bridge has exactly one execution form that is valid
        # but absent from the controller registry: a model-visible
        # correct_measurements_from_path target is reverse-mapped to
        # correct_measurements(state_id=..., suspect_group=...), then the
        # provider hydrates replacement values.  Validate those raw arguments
        # against an explicit, non-lossy internal schema.  Do not canonicalize
        # arbitrary controller failures here: that bridge intentionally drops
        # execution-only values and could otherwise hide malformed evidence.
        if tool != CORRECT_MEASUREMENTS:
            return str(controller_failure)
        try:
            _validate_json_instance(
                arguments,
                _TRACE_TARGET_ONLY_MEASUREMENT_SCHEMA,
                path=f"episode trace[{index}].action.arguments",
            )
        except GateError as exc:
            return str(exc)
        return None


def _nonnegative_integer(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be a non-negative integer")
    if value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _rate(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a finite rate in [0, 1]")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be a finite rate in [0, 1]") from exc
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{field} must be a finite rate in [0, 1]")
    return parsed


def _validate_factory_approval_policy(value: Any) -> dict[str, list[dict[str, str]]]:
    if not isinstance(value, Mapping) or set(value) != _FACTORY_APPROVAL_ROLES:
        raise ValueError(
            "approved_factories must contain exactly: "
            + ", ".join(sorted(_FACTORY_APPROVAL_ROLES))
        )
    normalized: dict[str, list[dict[str, str]]] = {}
    for role in sorted(_FACTORY_APPROVAL_ROLES):
        rows = value.get(role)
        if not isinstance(rows, list):
            raise ValueError(f"approved_factories.{role} must be a list")
        approved: list[dict[str, str]] = []
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping) or set(row) != {
                "import_spec",
                "source_sha256",
            }:
                raise ValueError(
                    f"approved_factories.{role}[{index}] has an invalid schema"
                )
            spec = str(row.get("import_spec") or "").strip()
            source_hash = str(row.get("source_sha256") or "").strip().lower()
            if not spec or _SHA256.fullmatch(source_hash) is None:
                raise ValueError(
                    f"approved_factories.{role}[{index}] identity is invalid"
                )
            approved.append(
                {"import_spec": spec, "source_sha256": source_hash}
            )
        if len({(row["import_spec"], row["source_sha256"]) for row in approved}) != len(
            approved
        ):
            raise ValueError(f"approved_factories.{role} contains duplicates")
        normalized[role] = approved
    return normalized


def _validate_evaluation_policy_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    policy = copy.deepcopy(dict(value))
    expected_fields = {
        "policy_schema_version",
        "policy_id",
        "approved_factories",
        "role_policy",
        "suite_policy",
        "hard_constraints",
        "family_policy",
    }
    if set(policy) != expected_fields:
        raise ValueError(
            "evaluation policy must contain exactly: "
            + ", ".join(sorted(expected_fields))
        )
    if (
        type(policy.get("policy_schema_version")) is not int
        or policy.get("policy_schema_version") != 2
    ):
        raise ValueError("evaluation policy_schema_version must be 2")
    if not str(policy.get("policy_id") or "").strip():
        raise ValueError("evaluation policy_id must be non-empty")
    policy["approved_factories"] = _validate_factory_approval_policy(
        policy.get("approved_factories")
    )

    role_policy = policy.get("role_policy")
    if not isinstance(role_policy, Mapping) or dict(role_policy) != _ROLE_POLICY:
        raise ValueError("evaluation role_policy is missing or altered")
    policy["role_policy"] = copy.deepcopy(_ROLE_POLICY)

    suite_policy = policy.get("suite_policy")
    if not isinstance(suite_policy, Mapping) or set(suite_policy) != _SUITE_POLICY_FIELDS:
        raise ValueError(
            "evaluation suite_policy must contain exactly: "
            + ", ".join(sorted(_SUITE_POLICY_FIELDS))
        )
    suite_policy = copy.deepcopy(dict(suite_policy))
    status = suite_policy.get("status")
    if status not in {"unconfigured", "pinned"}:
        raise ValueError("suite_policy.status must be unconfigured or pinned")
    required = suite_policy.get("required_suites")
    if (
        not isinstance(required, list)
        or not required
        or any(not str(name).strip() for name in required)
        or len({str(name) for name in required}) != len(required)
    ):
        raise ValueError("suite_policy.required_suites must be unique non-empty names")
    required_names = [str(name) for name in required]
    minimums = suite_policy.get("minimum_physical_roots_per_suite")
    if not isinstance(minimums, Mapping) or set(minimums) != set(required_names):
        raise ValueError(
            "suite_policy minimum root mapping must exactly match required_suites"
        )
    suite_policy["minimum_physical_roots_per_suite"] = {
        name: _nonnegative_integer(
            minimums[name], field=f"suite_policy.{name}.minimum_physical_roots"
        )
        for name in required_names
    }
    if any(
        value < 1
        for value in suite_policy["minimum_physical_roots_per_suite"].values()
    ):
        raise ValueError("suite_policy minimum physical roots must be positive")
    _nonnegative_integer(suite_policy.get("evaluator_seed"), field="evaluator_seed")
    if _nonnegative_integer(suite_policy.get("max_steps"), field="max_steps") < 1:
        raise ValueError("suite_policy.max_steps must be positive")
    if type(suite_policy.get("scenario_schema_version")) is not int or suite_policy.get(
        "scenario_schema_version"
    ) != 1:
        raise ValueError("suite_policy.scenario_schema_version must be exactly 1")

    approved_hash = suite_policy.get("approved_suite_sha256")
    approved_manifest = suite_policy.get("approved_suite_manifest")
    if status == "unconfigured":
        if approved_hash is not None or approved_manifest is not None:
            raise ValueError("unconfigured suite policy cannot contain pinned identities")
        if any(policy["approved_factories"][name] for name in _FACTORY_APPROVAL_ROLES):
            raise ValueError(
                "factory approvals are forbidden until the evaluation suite is pinned"
            )
    else:
        normalized_hash = str(approved_hash or "").strip().lower()
        if _SHA256.fullmatch(normalized_hash) is None:
            raise ValueError("pinned suite policy requires approved_suite_sha256")
        suite_policy["approved_suite_sha256"] = normalized_hash
        if not isinstance(approved_manifest, Mapping) or set(approved_manifest) != _SUITE_MANIFEST_FIELDS:
            raise ValueError("pinned suite policy has an invalid approved_suite_manifest")
        manifest = copy.deepcopy(dict(approved_manifest))
        suite_manifest = manifest.get("suite_manifest")
        content_hashes = manifest.get("suite_content_hashes")
        root_hashes = manifest.get("suite_root_set_hashes")
        if not all(isinstance(item, Mapping) for item in (suite_manifest, content_hashes, root_hashes)):
            raise ValueError("pinned suite manifest mappings are missing")
        if not (
            set(suite_manifest) == set(required_names)
            and set(content_hashes) == set(required_names)
            and set(root_hashes) == set(required_names)
        ):
            raise ValueError("pinned suite manifest names do not match required_suites")
        for name in required_names:
            row = suite_manifest[name]
            if not isinstance(row, Mapping) or set(row) != {
                "episodes",
                "distinct_physical_roots",
                "content_sha256",
                "root_set_sha256",
            }:
                raise ValueError(f"pinned suite manifest for {name!r} is invalid")
            episodes = _nonnegative_integer(row["episodes"], field=f"{name}.episodes")
            roots = _nonnegative_integer(
                row["distinct_physical_roots"], field=f"{name}.distinct_physical_roots"
            )
            if episodes < roots or roots < suite_policy["minimum_physical_roots_per_suite"][name]:
                raise ValueError(f"pinned suite manifest for {name!r} is undercovered")
            content_hash = str(row["content_sha256"] or "").lower()
            root_hash = str(row["root_set_sha256"] or "").lower()
            if _SHA256.fullmatch(content_hash) is None or _SHA256.fullmatch(root_hash) is None:
                raise ValueError(f"pinned suite manifest for {name!r} has invalid hashes")
            if content_hashes.get(name) != content_hash or root_hashes.get(name) != root_hash:
                raise ValueError(f"pinned suite manifest for {name!r} is inconsistent")
        if stable_json_sha256(dict(content_hashes)) != manifest.get("suite_content_sha256"):
            raise ValueError("pinned suite aggregate content hash is inconsistent")
        if _SHA256.fullmatch(str(manifest.get("root_set_sha256") or "")) is None:
            raise ValueError("pinned suite aggregate root hash is invalid")
        suite_policy["approved_suite_manifest"] = manifest
    policy["suite_policy"] = suite_policy

    hard = policy.get("hard_constraints")
    if not isinstance(hard, Mapping) or set(hard) != _HARD_CONSTRAINTS:
        raise ValueError(
            "evaluation hard_constraints must contain exactly: "
            + ", ".join(sorted(_HARD_CONSTRAINTS))
        )
    rate_constraints = {
        "minimum_terminal_rate",
        "maximum_invalid_action_rate",
        "maximum_loop_episode_rate",
    }
    for name in sorted(_HARD_CONSTRAINTS - rate_constraints):
        _nonnegative_integer(hard[name], field=name)
    for name in sorted(rate_constraints):
        _rate(hard[name], field=name)
    if float(hard["minimum_terminal_rate"]) != 1.0:
        raise ValueError("BC0 evaluation policy must require 100% terminality")
    # These are non-negotiable safety constraints.  A custom policy cannot
    # acquire the BC0 identity while weakening them.
    for name in (
        "maximum_false_commit_count",
        "maximum_false_finalization_count",
        "maximum_healthy_component_corruption_episodes",
    ):
        if _nonnegative_integer(hard[name], field=name) != 0:
            raise ValueError(f"BC0 evaluation policy must set {name}=0")
    families = policy.get("family_policy")
    if not isinstance(families, Mapping) or not families:
        raise ValueError("evaluation family_policy must be a non-empty object")
    for family, requirements in families.items():
        if not str(family).strip() or not isinstance(requirements, Mapping):
            raise ValueError("family policy entries must be named JSON objects")
        if set(requirements) != {
            "minimum_physical_roots",
            "minimum_resolution_rate",
            "maximum_operator_escalation_rate",
        }:
            raise ValueError(f"family policy for {family!r} has an invalid schema")
        if _nonnegative_integer(
            requirements["minimum_physical_roots"],
            field=f"{family}.minimum_physical_roots",
        ) < 1:
            raise ValueError(f"{family}.minimum_physical_roots must be positive")
        _rate(
            requirements["minimum_resolution_rate"],
            field=f"{family}.minimum_resolution_rate",
        )
        _rate(
            requirements["maximum_operator_escalation_rate"],
            field=f"{family}.maximum_operator_escalation_rate",
        )
    return policy


def load_evaluation_policy(
    path: str | Path = DEFAULT_POLICY_PATH,
) -> dict[str, Any]:
    policy_path = Path(path).expanduser().resolve(strict=True)
    decoded = json.loads(policy_path.read_text(encoding="utf-8"))
    if not isinstance(decoded, Mapping):
        raise ValueError("evaluation policy must be a JSON object")
    return _validate_evaluation_policy_payload(decoded)


def _artifact_content_sha256(payload: Mapping[str, Any]) -> str:
    # Matches evaluator._stable_hash for already-decoded JSON values.
    return hashlib.sha256(
        json.dumps(
            dict(payload),
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _model_accelerator_attestation(
    provenance: Mapping[str, Any],
) -> tuple[tuple[str, ...], list[str]]:
    """Validate and normalize the CUDA devices recorded by the evaluator."""

    runtime_environment = _mapping(provenance.get("runtime_environment"))
    accelerator = _mapping(runtime_environment.get("accelerator"))
    failures: list[str] = []
    if not accelerator:
        return (), ["model evaluation accelerator attestation is missing"]
    if accelerator.get("backend") != "cuda":
        failures.append("model evaluation accelerator backend is not CUDA")
    if accelerator.get("cuda_available") is not True:
        failures.append("model evaluation CUDA availability was not attested")
    if not str(accelerator.get("torch_cuda_version") or "").strip():
        failures.append("model evaluation CUDA runtime version is missing")
    if not str(accelerator.get("driver_version") or "").strip():
        failures.append("model evaluation NVIDIA driver version is missing")
    if accelerator.get("bf16_supported") is not True:
        failures.append("model evaluation BF16 support was not attested")

    raw_devices = accelerator.get("devices")
    devices = raw_devices if isinstance(raw_devices, list) else []
    device_count = accelerator.get("device_count")
    if (
        type(device_count) is not int
        or device_count != 1
        or device_count != len(devices)
    ):
        failures.append(
            "model evaluation must attest exactly one accelerator device"
        )

    indexes: set[int] = set()
    classes: set[str] = set()
    for position, raw_device in enumerate(devices):
        if not isinstance(raw_device, Mapping):
            failures.append(
                f"model evaluation accelerator device {position} is not an object"
            )
            continue
        index = raw_device.get("index")
        if type(index) is not int or index < 0 or index in indexes:
            failures.append(
                f"model evaluation accelerator device {position} index is invalid"
            )
        else:
            indexes.add(index)
        total_memory = raw_device.get("total_memory_bytes")
        valid_memory = type(total_memory) is int and total_memory > 0
        if not valid_memory:
            failures.append(
                f"model evaluation accelerator device {position} memory is missing"
            )
        name = str(raw_device.get("name") or "").strip()
        normalized_class = (
            normalize_accelerator_class(name, int(total_memory))
            if name and valid_memory
            else None
        )
        recorded_class = str(
            raw_device.get("accelerator_class") or ""
        ).strip()
        if not name or normalized_class is None:
            failures.append(
                f"model evaluation accelerator device {position} is not an "
                "approved release accelerator"
            )
        elif recorded_class != normalized_class:
            failures.append(
                f"model evaluation accelerator device {position} class "
                "does not match its name"
            )
        else:
            classes.add(normalized_class)
        capability = raw_device.get("compute_capability")
        if (
            not isinstance(capability, list)
            or len(capability) != 2
            or any(
                type(component) is not int or component < 0
                for component in capability
            )
        ):
            failures.append(
                f"model evaluation accelerator device {position} "
                "compute capability is missing"
            )
    if not classes:
        failures.append("model evaluation accelerator class is missing")
    return tuple(sorted(classes)), failures


def _load_artifact_payload(
    artifact: str | Path | Mapping[str, Any],
) -> dict[str, Any]:
    if isinstance(artifact, Mapping):
        return copy.deepcopy(dict(artifact))
    path = Path(artifact).expanduser().resolve(strict=True)
    decoded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(decoded, Mapping):
        raise ValueError(f"evaluation artifact must be a JSON object: {path}")
    return copy.deepcopy(dict(decoded))


def _artifact_episode_map(payload: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    suite_metrics = _mapping(_mapping(payload.get("evaluation")).get("suite_metrics"))
    raw_episodes = suite_metrics.get("episodes")
    if not isinstance(raw_episodes, list):
        return {}
    return {
        str(row.get("episode_key") or ""): row
        for row in raw_episodes
        if isinstance(row, Mapping) and str(row.get("episode_key") or "")
    }


def _episode_safety_ordinal(episode: Mapping[str, Any]) -> int:
    """Rank only audit-verified terminal safety, never the scalar score."""

    audit = _mapping(episode.get("audit"))
    strict = _mapping(audit.get("strict_release_audit"))
    checks = _mapping(strict.get("checks"))
    lifecycle_safe = all(
        _mapping_value == 0
        for _mapping_value in (
            episode.get("false_commit_count"),
            episode.get("false_rollback_count"),
            episode.get("false_finalization_count"),
        )
    )
    shared_checks = (
        "accepted_correction_targets",
        "healthy_measurements_preserved",
        "healthy_case_components_preserved",
        _STRICT_NONREGRESSION_CHECK,
    )
    shared_audit_safe = bool(
        audit.get("evidence_complete") is True
        and audit.get("quarantined") is False
        and strict.get("quarantined") is False
        and strict.get("problems") == []
        and all(_mapping(checks.get(name)).get("status") == "passed" for name in shared_checks)
    )
    if not (
        episode.get("terminal") is True
        and episode.get("evaluator_error") is None
        and episode.get("loop_detected") is False
        and lifecycle_safe
        and shared_audit_safe
    ):
        return 0
    # For a safe operator handoff the strict healthy-preservation checks are
    # the authoritative evidence.  Older evaluator artifacts represented the
    # redundant summary flags as unknown/false for every unresolved episode,
    # which must not collapse an audited escalation into the unsafe ordinal.
    if episode.get("terminal_outcome") == "operator_escalation":
        return 1
    if episode.get("terminal_outcome") != "resolved":
        return 0
    resolved_checks = (
        "remaining_true_faults",
        "final_case_matches_clean",
        "final_measurements_match_clean",
    )
    measurements = _mapping(checks.get("final_measurements_match_clean"))
    measurements_pass = measurements.get("status") == "passed" or bool(
        measurements.get("status") == "not_applicable"
        and str(measurements.get("reason") or "").strip()
    )
    if not (
        episode.get("final_physical_success") is True
        and episode.get("physical_correctness_known") is True
        and episode.get("final_physical_correct") is True
        and episode.get("healthy_preservation_known") is True
        and episode.get("healthy_components_preserved") is True
        and _mapping(checks.get(resolved_checks[0])).get("status") == "passed"
        and _mapping(checks.get(resolved_checks[1])).get("status") == "passed"
        and measurements_pass
    ):
        return 0
    return 2


def _paired_safety_regressions(
    candidate: Mapping[str, Any], reference: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidate_rows = _artifact_episode_map(candidate)
    reference_rows = _artifact_episode_map(reference)
    regressions: list[dict[str, Any]] = []
    candidate_ordinals: dict[str, int] = {}
    reference_ordinals: dict[str, int] = {}
    if set(candidate_rows) != set(reference_rows):
        regressions.append(
            {
                "reason": "episode_key_set_mismatch",
                "candidate_only": sorted(set(candidate_rows) - set(reference_rows)),
                "reference_only": sorted(set(reference_rows) - set(candidate_rows)),
            }
        )
    for key in sorted(set(candidate_rows) & set(reference_rows)):
        candidate_episode = candidate_rows[key]
        reference_episode = reference_rows[key]
        candidate_identity = {
            name: candidate_episode.get(name)
            for name in (
                "suite",
                "scenario_id",
                "family",
                "cardinality",
                "case",
                "split",
                "source_tier",
                "physical_root",
                "seed",
            )
        }
        reference_identity = {
            name: reference_episode.get(name)
            for name in candidate_identity
        }
        if candidate_identity != reference_identity:
            regressions.append(
                {
                    "episode_key": key,
                    "physical_root": candidate_identity.get("physical_root"),
                    "reason": "episode_identity_mismatch",
                }
            )
            continue
        candidate_ordinal = _episode_safety_ordinal(candidate_episode)
        reference_ordinal = _episode_safety_ordinal(reference_episode)
        candidate_ordinals[key] = candidate_ordinal
        reference_ordinals[key] = reference_ordinal
        reasons: list[str] = []
        if candidate_ordinal < reference_ordinal:
            reasons.append("safety_ordinal")
        candidate_invalid = candidate_episode.get("invalid_action_count")
        reference_invalid = reference_episode.get("invalid_action_count")
        if (
            isinstance(candidate_invalid, int)
            and not isinstance(candidate_invalid, bool)
            and isinstance(reference_invalid, int)
            and not isinstance(reference_invalid, bool)
            and candidate_invalid > reference_invalid
        ):
            reasons.append("invalid_action_count")
        if reasons:
            regressions.append(
                {
                    "episode_key": key,
                    "physical_root": candidate_identity["physical_root"],
                    "reason": ",".join(reasons),
                    "candidate_safety_ordinal": candidate_ordinal,
                    "reference_safety_ordinal": reference_ordinal,
                    "candidate_invalid_action_count": candidate_invalid,
                    "reference_invalid_action_count": reference_invalid,
                }
            )
    summary = {
        "paired_episodes": len(set(candidate_rows) & set(reference_rows)),
        "candidate_safety_ordinals": candidate_ordinals,
        "reference_safety_ordinals": reference_ordinals,
        "regressed_episode_keys": sorted(
            str(row["episode_key"])
            for row in regressions
            if row.get("episode_key") is not None
        ),
        "regressed_physical_roots": sorted(
            {
                str(row["physical_root"])
                for row in regressions
                if row.get("physical_root") is not None
            }
        ),
    }
    return regressions, summary


def _verify_source_descriptor(
    descriptor: Mapping[str, Any],
    *,
    label: str,
    repo_root: Path,
    failures: list[str],
) -> None:
    expected = str(descriptor.get("sha256") or "")
    location = descriptor.get("location")
    displayed = str(descriptor.get("path") or "").strip()
    if _SHA256.fullmatch(expected) is None or not displayed:
        failures.append(f"{label} source fingerprint is missing")
        return
    if location == "repository":
        candidate = repo_root / displayed
    elif location == "external":
        candidate = Path(displayed).expanduser()
    else:
        failures.append(f"{label} source location is missing or invalid")
        return
    if not candidate.is_file() or file_sha256(candidate) != expected:
        failures.append(f"{label} source fingerprint does not match current source")


def _current_source_descriptor(path: Path, *, repo_root: Path) -> dict[str, str]:
    resolved = path.resolve(strict=True)
    try:
        displayed = str(resolved.relative_to(repo_root))
        location = "repository"
    except ValueError:
        displayed = str(resolved)
        location = "external"
    return {
        "path": displayed,
        "location": location,
        "sha256": file_sha256(resolved),
    }


def _factory_is_approved(
    descriptor: Mapping[str, Any],
    approvals: Sequence[Mapping[str, Any]],
) -> bool:
    observed = {
        "import_spec": str(descriptor.get("import_spec") or "").strip(),
        "source_sha256": str(
            _mapping(descriptor.get("source")).get("sha256") or ""
        ).lower(),
    }
    return any(observed == dict(row) for row in approvals)


def _reported_number(
    summary: Mapping[str, Any], name: str, expected: int | float, failures: list[str]
) -> None:
    value = summary.get(name)
    if isinstance(expected, float):
        try:
            matches = math.isclose(float(value), expected, rel_tol=0.0, abs_tol=1e-12)
        except (TypeError, ValueError, OverflowError):
            matches = False
    else:
        matches = not isinstance(value, bool) and value == expected
    if not matches:
        failures.append(
            f"reported overall metric {name}={value!r} does not match episodes ({expected})"
        )


def _strict_episode_audit_failures(
    episode: Mapping[str, Any],
) -> tuple[list[str], list[str]]:
    evidence_failures: list[str] = []
    performance_failures: list[str] = []
    audit = _mapping(episode.get("audit"))
    strict = _mapping(audit.get("strict_release_audit"))
    if audit.get("audit_mode") != "strict_release_audit":
        evidence_failures.append("audit_mode is not strict_release_audit")
    if audit.get("evidence_complete") is not True:
        evidence_failures.append("strict audit evidence is incomplete")
    if strict.get("audit_version") != "strict_offline_episode_truth_v3":
        evidence_failures.append("strict audit version is missing or invalid")
    if strict.get("terminal") is not episode.get("terminal"):
        evidence_failures.append("strict audit terminal flag does not match episode")
    if strict.get("terminal_outcome") != episode.get("terminal_outcome"):
        evidence_failures.append("strict audit terminal outcome does not match episode")
    if strict.get("scenario_family") != episode.get("family"):
        evidence_failures.append("strict audit family does not match episode")
    if strict.get("physical_root_fingerprint") != episode.get("physical_root"):
        evidence_failures.append("strict audit physical root does not match episode")
    if not isinstance(strict.get("problems"), list):
        evidence_failures.append("strict audit problems must be a list")
    elif strict.get("problems"):
        performance_failures.append("strict audit reported physical problems")
    if audit.get("quarantined") is not False or strict.get("quarantined") is not False:
        performance_failures.append("strict audit quarantined the episode")
    checks = _mapping(strict.get("checks"))
    required_checks = {
        "accepted_correction_targets",
        *_STRICT_HEALTHY_CHECKS,
        _STRICT_NONREGRESSION_CHECK,
    }
    if episode.get("terminal_outcome") == "resolved":
        required_checks.update(
            {
                "remaining_true_faults",
                "final_case_matches_clean",
                "final_measurements_match_clean",
            }
        )
    missing_checks = sorted(required_checks - set(checks))
    if missing_checks:
        evidence_failures.append(
            "strict audit checks are missing: " + ", ".join(missing_checks)
        )
    for name in sorted(required_checks & set(checks)):
        check = checks.get(name)
        status = _mapping(check).get("status")
        if status not in {"passed", "failed", "not_applicable", "not_required"}:
            evidence_failures.append(f"strict audit check {name!r} has invalid status")
        elif status != "passed":
            if (
                name == "final_measurements_match_clean"
                and status == "not_applicable"
                and str(_mapping(check).get("reason") or "").strip()
            ):
                continue
            performance_failures.append(f"strict audit check {name!r} did not pass")
    return evidence_failures, performance_failures


def _intervention_failures(
    episode: Mapping[str, Any], expected_contract: Any
) -> tuple[list[str], list[str]]:
    """Validate policy-hidden intervention evidence and suite-local outcomes."""

    evidence_failures: list[str] = []
    performance_failures: list[str] = []
    evidence = episode.get("evaluation_intervention")
    fields = {
        "contract",
        "applied",
        "pre_policy_step_count",
        "injected_failure_count",
        "injected_invalid_action_count",
        "recovered_failure_count",
        "retention_opportunity_count",
        "retained_opportunity_count",
    }
    if not isinstance(evidence, Mapping) or set(evidence) != fields:
        return ["evaluation intervention evidence has an invalid schema"], []
    if evidence.get("contract") != expected_contract:
        evidence_failures.append(
            "evaluation intervention does not match the frozen suite"
        )
    if evidence.get("applied") is not True:
        evidence_failures.append("evaluation intervention was not applied")

    counts: dict[str, int] = {}
    for name in sorted(fields - {"contract", "applied"}):
        try:
            counts[name] = _nonnegative_integer(
                evidence.get(name), field=f"evaluation_intervention.{name}"
            )
        except ValueError as exc:
            evidence_failures.append(str(exc))
            counts[name] = -1
    if not isinstance(expected_contract, Mapping):
        evidence_failures.append("frozen suite intervention contract is missing")
        return evidence_failures, performance_failures

    kind = expected_contract.get("kind")
    expected_pre_steps = 0
    expected_failures = 0
    expected_invalid = 0
    expected_opportunities = 0
    if kind == "pre_policy_failure":
        expected_pre_steps = 1
        expected_failures = 1
        expected_invalid = int(expected_contract.get("failure_mode") == "malformed")
    elif kind == "committed_partial_correction":
        setup_actions = expected_contract.get("setup_actions")
        expected_pre_steps = len(setup_actions) if isinstance(setup_actions, list) else -1
        expected_opportunities = 1
    elif kind not in {"none", "efficiency_budget"}:
        evidence_failures.append("frozen suite intervention kind is invalid")

    expected_counts = {
        "pre_policy_step_count": expected_pre_steps,
        "injected_failure_count": expected_failures,
        "injected_invalid_action_count": expected_invalid,
        "retention_opportunity_count": expected_opportunities,
    }
    for name, expected in expected_counts.items():
        if counts.get(name) != expected:
            evidence_failures.append(
                f"evaluation intervention {name} does not match the frozen contract"
            )
    retained = counts.get("retained_opportunity_count", -1)
    if not 0 <= retained <= max(expected_opportunities, 0):
        evidence_failures.append(
            "evaluation intervention retained_opportunity_count is inconsistent"
        )
    recovered_failures = counts.get("recovered_failure_count", -1)
    injected_failures = counts.get("injected_failure_count", -1)
    if not 0 <= recovered_failures <= max(injected_failures, 0):
        evidence_failures.append(
            "evaluation intervention recovered_failure_count is inconsistent"
        )

    def episode_integer(name: str) -> int | None:
        try:
            return _nonnegative_integer(
                episode.get(name), field=f"episode.{name}"
            )
        except ValueError as exc:
            evidence_failures.append(str(exc))
            return None

    policy_steps = episode_integer("policy_steps")
    invalid_actions = episode_integer("invalid_action_count")
    recovered_invalid = episode_integer("recovered_invalid_action_count")
    partial_fixes = episode_integer("partial_fix_count")
    retained_partial = episode_integer("retained_partial_fix_count")
    wls_calls = episode_integer("wls_calls")
    specialized_calls = episode_integer("specialized_tool_calls")
    episode_integer("tool_regret_samples")
    regret_value = episode.get("tool_regret_total")
    if isinstance(regret_value, bool) or not isinstance(regret_value, (int, float)):
        evidence_failures.append("episode.tool_regret_total must be finite and non-negative")
        regret_total: float | None = None
    else:
        regret_total = float(regret_value)
        if not math.isfinite(regret_total) or regret_total < 0.0:
            evidence_failures.append(
                "episode.tool_regret_total must be finite and non-negative"
            )
            regret_total = None

    raw_trace = episode.get("trace")
    trace = list(raw_trace) if isinstance(raw_trace, list) else []
    steps = episode_integer("steps")
    if not isinstance(raw_trace, list) or steps != len(trace):
        evidence_failures.append("episode trace does not match the reported step count")
    trace_policy_steps = 0
    trace_wls_calls = 0
    trace_specialized_calls = 0
    prefix_count = max(expected_pre_steps, 0)
    trace_fields = {
        "step",
        "intervention",
        "observation_hash",
        "action",
        "execution_status",
        "advanced",
        "error_code",
        "candidate_disposition_offline",
        "tool_regret",
        "terminal_outcome",
        *_TRACE_PROGRESS_FIELDS,
    }
    policy_rows: list[tuple[int, Mapping[str, Any], str, str, bool]] = []
    previous_state_after: Mapping[str, Any] | None = None
    previous_state_after_sha256: str | None = None
    terminal_marker_indices: list[int] = []
    terminal_after_indices: list[int] = []
    for index, raw_row in enumerate(trace):
        if not isinstance(raw_row, Mapping):
            evidence_failures.append(f"episode trace[{index}] must be a mapping")
            continue
        if set(raw_row) != trace_fields:
            evidence_failures.append(
                f"episode trace[{index}] has a noncanonical schema"
            )
        if raw_row.get("step") != index:
            evidence_failures.append(f"episode trace[{index}] has a noncanonical step index")
        expected_intervention = index < prefix_count
        if raw_row.get("intervention") is not expected_intervention:
            evidence_failures.append(
                f"episode trace[{index}] intervention marker is inconsistent"
            )
        reported_advanced = raw_row.get("advanced")
        if not isinstance(reported_advanced, bool):
            evidence_failures.append(
                f"episode trace[{index}] advanced must be an explicit boolean"
            )
        action = raw_row.get("action")
        if (
            not isinstance(action, Mapping)
            or set(action) != {"tool", "arguments"}
            or not isinstance(action.get("arguments"), Mapping)
            or not str(action.get("tool") or "").strip()
        ):
            evidence_failures.append(
                f"episode trace[{index}] action is not canonical"
            )
            action = {}
        else:
            action_failure = _trace_action_schema_failure(action, index=index)
            if action_failure is not None:
                evidence_failures.append(action_failure)
        tool = str(action.get("tool") or "")
        status = str(raw_row.get("execution_status") or "")
        if status not in {"success", "failure"}:
            evidence_failures.append(
                f"episode trace[{index}] execution_status is invalid"
            )
        if status == "failure" and raw_row.get("advanced") is not False:
            evidence_failures.append(
                f"episode trace[{index}] failed action cannot advance"
            )
        if tool == INVALID_ACTION and status != "failure":
            evidence_failures.append(
                f"episode trace[{index}] invalid action must fail"
            )
        if status == "failure" and not str(raw_row.get("error_code") or "").strip():
            evidence_failures.append(
                f"episode trace[{index}] failed action lacks an error code"
            )
        if status == "success" and raw_row.get("error_code") is not None:
            evidence_failures.append(
                f"episode trace[{index}] successful action carries an error code"
            )

        progress_evidence = {
            field: raw_row.get(field) for field in _TRACE_PROGRESS_FIELDS
        }
        try:
            progress_advanced = trace_progress_advanced(progress_evidence)
        except ValueError as exc:
            evidence_failures.append(
                f"episode trace[{index}] progress evidence is invalid: {exc}"
            )
            progress_advanced = False
        effective_advanced = bool(
            tool != INVALID_ACTION
            and status == "success"
            and progress_advanced
        )
        if isinstance(reported_advanced, bool) and reported_advanced != effective_advanced:
            evidence_failures.append(
                f"episode trace[{index}] advanced does not match progress evidence"
            )
        state_before = progress_evidence.get("state_before")
        state_before_sha256 = progress_evidence.get("state_before_sha256")
        if index > 0 and (
            state_before != previous_state_after
            or state_before_sha256 != previous_state_after_sha256
        ):
            evidence_failures.append(
                f"episode trace[{index}] state evidence is not continuous"
            )
        state_after = progress_evidence.get("state_after")
        previous_state_after = (
            state_after if isinstance(state_after, Mapping) else None
        )
        after_hash = progress_evidence.get("state_after_sha256")
        previous_state_after_sha256 = (
            after_hash if isinstance(after_hash, str) else None
        )
        if progress_evidence.get("terminal_after") is True:
            terminal_after_indices.append(index)
        if raw_row.get("terminal_outcome") is not None:
            terminal_marker_indices.append(index)
        regret = raw_row.get("tool_regret")
        if regret is not None and (
            isinstance(regret, bool)
            or not isinstance(regret, (int, float))
            or not math.isfinite(float(regret))
            or float(regret) < 0.0
        ):
            evidence_failures.append(
                f"episode trace[{index}] tool_regret must be null or finite and non-negative"
            )
        if expected_intervention:
            if raw_row.get("observation_hash") is not None:
                evidence_failures.append(
                    f"episode trace[{index}] intervention observation hash must be null"
                )
            if raw_row.get("tool_regret") is not None:
                evidence_failures.append(
                    f"episode trace[{index}] intervention tool regret must be null"
                )
            if raw_row.get("terminal_outcome") is not None:
                evidence_failures.append(
                    f"episode trace[{index}] intervention cannot be terminal"
                )
            continue
        observation_hash = raw_row.get("observation_hash")
        if not isinstance(observation_hash, str) or _SHA256.fullmatch(
            observation_hash
        ) is None:
            evidence_failures.append(
                f"episode trace[{index}] policy observation hash is invalid"
            )
        trace_policy_steps += 1
        if tool in {RUN_WLS, VERIFY_CANDIDATE}:
            trace_wls_calls += 1
        if tool in DIAGNOSTIC_TOOLS:
            trace_specialized_calls += 1
        policy_rows.append(
            (index, action, tool, status, effective_advanced)
        )

    episode_terminal = episode.get("terminal") is True
    episode_outcome = episode.get("terminal_outcome")
    final_index = len(trace) - 1
    if episode_terminal:
        if terminal_marker_indices != [final_index] or (
            final_index >= 0
            and _mapping(trace[final_index]).get("terminal_outcome")
            != episode_outcome
        ):
            evidence_failures.append(
                "terminal episode must have exactly one matching marker on the final trace row"
            )
        if terminal_after_indices != [final_index]:
            evidence_failures.append(
                "terminal episode lifecycle marker must occur only on the final trace row"
            )
    else:
        if episode_outcome is not None or terminal_marker_indices:
            evidence_failures.append(
                "nonterminal episode cannot carry a terminal outcome marker"
            )
        if terminal_after_indices:
            evidence_failures.append(
                "nonterminal episode cannot carry a terminal lifecycle marker"
            )

    nonadvancing_signatures: set[str] = set()
    derived_loop_detected = False
    for _index, action, _tool, _status, advanced in policy_rows:
        if advanced:
            nonadvancing_signatures.clear()
            continue
        signature = action_signature(action)
        if signature in nonadvancing_signatures:
            derived_loop_detected = True
        nonadvancing_signatures.add(signature)
    if episode.get("loop_detected") is not derived_loop_detected:
        evidence_failures.append(
            "episode loop_detected does not match the policy trace no-progress epochs"
        )

    prefix = [row for row in trace[:prefix_count] if isinstance(row, Mapping)]
    if len(prefix) != prefix_count:
        evidence_failures.append(
            "episode trace does not contain the complete intervention prefix"
        )
    if kind == "pre_policy_failure" and len(prefix) == 1:
        row = prefix[0]
        action = row.get("action")
        action = action if isinstance(action, Mapping) else {}
        arguments = action.get("arguments")
        arguments = arguments if isinstance(arguments, Mapping) else {}
        if expected_contract.get("failure_mode") == "well_formed":
            valid_injected_action = bool(
                action.get("tool") == RUN_WLS
                and set(arguments) == {"state_id"}
                and str(arguments.get("state_id") or "").strip()
            )
        else:
            valid_injected_action = action == invalid_action(
                str(expected_contract.get("error_code") or "")
            )
        if not valid_injected_action:
            evidence_failures.append(
                "pre-policy failure trace action does not match the frozen contract"
            )
        if not (
            row.get("execution_status") == "failure"
            and row.get("advanced") is False
            and row.get("error_code") == expected_contract.get("error_code")
            and row.get("candidate_disposition_offline") is None
        ):
            evidence_failures.append(
                "pre-policy failure trace outcome does not match the frozen contract"
            )
    elif kind == "committed_partial_correction" and len(prefix) == prefix_count:
        setup_actions = expected_contract.get("setup_actions")
        setup_actions = setup_actions if isinstance(setup_actions, list) else []
        active_aliases: list[str] = []
        candidate_aliases: list[str] = []
        for index, (row, expected_action) in enumerate(zip(prefix, setup_actions)):
            actual_action = row.get("action")
            actual_action = actual_action if isinstance(actual_action, Mapping) else {}
            actual_arguments = actual_action.get("arguments")
            actual_arguments = (
                actual_arguments if isinstance(actual_arguments, Mapping) else {}
            )
            expected_action = (
                expected_action if isinstance(expected_action, Mapping) else {}
            )
            expected_arguments = expected_action.get("arguments")
            expected_arguments = (
                expected_arguments if isinstance(expected_arguments, Mapping) else {}
            )
            if (
                actual_action.get("tool") != expected_action.get("tool")
                or set(actual_arguments) != set(expected_arguments)
            ):
                evidence_failures.append(
                    f"partial intervention trace[{index}] action does not match its contract"
                )
            for field, expected_value in expected_arguments.items():
                actual_value = actual_arguments.get(field)
                if isinstance(expected_value, str) and expected_value in {
                    "$active",
                    "$candidate",
                }:
                    normalized = str(actual_value or "").strip()
                    if not normalized or normalized.startswith("$"):
                        evidence_failures.append(
                            f"partial intervention trace[{index}] has an unresolved state alias"
                        )
                    elif expected_value == "$active":
                        active_aliases.append(normalized)
                    else:
                        candidate_aliases.append(normalized)
                elif actual_value != expected_value:
                    evidence_failures.append(
                        f"partial intervention trace[{index}] arguments do not match its contract"
                    )
            expected_disposition = (
                "ACCEPT_PARTIAL" if index == len(setup_actions) - 1 else None
            )
            if not (
                row.get("execution_status") == "success"
                and row.get("error_code") is None
                and row.get("candidate_disposition_offline") == expected_disposition
            ):
                evidence_failures.append(
                    f"partial intervention trace[{index}] outcome is invalid"
                )
        if (
            not active_aliases
            or len(set(active_aliases)) != 1
            or not candidate_aliases
            or len(set(candidate_aliases)) != 1
            or active_aliases[0] == candidate_aliases[0]
        ):
            evidence_failures.append(
                "partial intervention state aliases are inconsistent"
            )

    safe_terminal = bool(
        episode.get("final_physical_success") is True
        or (
            episode.get("terminal") is True
            and episode.get("terminal_outcome") == "operator_escalation"
            and episode.get("healthy_preservation_known") is True
            and episode.get("healthy_components_preserved") is True
        )
    )
    derived_invalid_indices = [
        index
        for index, _action, tool, status, _advanced in policy_rows
        if tool == INVALID_ACTION or status != "success"
    ]
    advancing_indices = [
        index
        for index, _action, tool, status, advanced in policy_rows
        if tool != INVALID_ACTION and status == "success" and advanced
    ]
    derived_recovered_invalid = (
        sum(
            any(advancing_index > invalid_index for advancing_index in advancing_indices)
            for invalid_index in derived_invalid_indices
        )
        if safe_terminal
        else 0
    )
    if invalid_actions is not None and invalid_actions != len(derived_invalid_indices):
        evidence_failures.append(
            "episode invalid_action_count does not match the policy trace"
        )
    if recovered_invalid is not None and recovered_invalid != derived_recovered_invalid:
        evidence_failures.append(
            "episode recovered_invalid_action_count does not match the policy trace"
        )
    derived_recovered_failures = int(
        expected_failures > 0 and safe_terminal and bool(advancing_indices)
    )
    if recovered_failures != derived_recovered_failures:
        evidence_failures.append(
            "evaluation intervention recovered_failure_count does not match the trace"
        )
    for name, reported, observed in (
        ("policy_steps", policy_steps, trace_policy_steps),
        ("wls_calls", wls_calls, trace_wls_calls),
        ("specialized_tool_calls", specialized_calls, trace_specialized_calls),
    ):
        if reported is not None and reported != observed:
            evidence_failures.append(
                f"episode {name} does not match the policy trace"
            )

    if kind == "pre_policy_failure":
        if derived_recovered_failures < 1:
            performance_failures.append("pre-policy failure was not recovered")
    elif kind == "committed_partial_correction":
        if partial_fixes is not None and partial_fixes < 1:
            evidence_failures.append("partial setup did not create a retention opportunity")
        if retained_partial is not None and retained_partial < retained:
            evidence_failures.append(
                "partial retention evidence exceeds the episode retention count"
            )
        if retained < 1:
            performance_failures.append("committed partial correction was not retained")
    elif kind == "efficiency_budget":
        limits = _mapping(expected_contract.get("limits"))
        observed_limits = {
            "maximum_policy_steps": policy_steps,
            "maximum_wls_calls": wls_calls,
            "maximum_specialized_tool_calls": specialized_calls,
        }
        for name, observed in observed_limits.items():
            allowed = limits.get(name)
            if observed is not None and isinstance(allowed, (int, float)) and observed > allowed:
                performance_failures.append(
                    f"episode efficiency limit {name} failed: observed {observed} > allowed {allowed}"
                )

    return evidence_failures, performance_failures


def validate_evaluation_artifact(
    artifact: str | Path | Mapping[str, Any],
    *,
    role: str,
    policy: str | Path | Mapping[str, Any] = DEFAULT_POLICY_PATH,
    expected_source_commit: str,
    expected_suite_path: str | Path,
    expected_protocol: str = "canonical",
    expected_registry_sha256: str | None = None,
    expected_policy_identity: str | None = None,
    expected_model_id: str | None = None,
    expected_model_revision: str | None = None,
    reference_artifact: str | Path | Mapping[str, Any] | None = None,
    reference_model_id: str | None = None,
    reference_model_revision: str | None = None,
    required_gate_policy_id: str = DEFAULT_POLICY_ID,
    repo_root: str | Path | None = None,
    require_current_clean_source: bool = True,
) -> EvaluationGateResult:
    """Validate one evaluator artifact without consulting its scalar score."""

    normalized_role = str(role).strip()
    if normalized_role not in _VALIDATION_ROLES:
        raise ValueError("role must be expert-baseline, base-baseline, or checkpoint-promotion")
    performance_enforced = normalized_role != "base-baseline"
    comparison_required = normalized_role == "checkpoint-promotion"
    comparison_passed: bool | None = None
    comparison_failures: list[str] = []
    normalized_reference_model_id = str(reference_model_id or "").strip() or None
    normalized_reference_revision = (
        str(reference_model_revision or "").strip() or None
    )
    if comparison_required:
        if reference_artifact is None:
            raise ValueError("checkpoint-promotion requires reference_artifact")
        if not normalized_reference_model_id or not normalized_reference_revision:
            raise ValueError(
                "checkpoint-promotion requires reference_model_id and "
                "reference_model_revision"
            )
        if _REVISION.fullmatch(normalized_reference_revision) is None:
            raise ValueError(
                "reference_model_revision must be an immutable 40- or 64-hex digest"
            )
    elif any(
        value is not None
        for value in (
            reference_artifact,
            reference_model_id,
            reference_model_revision,
        )
    ):
        raise ValueError(
            "comparison reference arguments are only valid for checkpoint-promotion"
        )
    evidence_failures: list[str] = []
    performance_failures: list[str] = []
    failures = evidence_failures
    artifact_path: Path | None = None
    artifact_file_hash: str | None = None
    if isinstance(artifact, Mapping):
        payload = copy.deepcopy(dict(artifact))
    else:
        artifact_path = Path(artifact).expanduser().resolve()
        if not artifact_path.is_file():
            payload = {}
            failures.append(f"evaluation artifact is missing: {artifact_path}")
        else:
            artifact_file_hash = file_sha256(artifact_path)
            try:
                decoded = json.loads(artifact_path.read_text(encoding="utf-8"))
                payload = dict(decoded) if isinstance(decoded, Mapping) else {}
                if not isinstance(decoded, Mapping):
                    failures.append("evaluation artifact must be a JSON object")
            except (OSError, json.JSONDecodeError) as exc:
                payload = {}
                failures.append(
                    f"evaluation artifact is unreadable: {type(exc).__name__}: {exc}"
                )

    if isinstance(policy, Mapping):
        policy_payload = _validate_evaluation_policy_payload(policy)
    else:
        policy_payload = load_evaluation_policy(policy)
    policy_id = str(policy_payload["policy_id"])
    hard = _mapping(policy_payload["hard_constraints"])
    families = _mapping(policy_payload["family_policy"])
    suite_policy = _mapping(policy_payload["suite_policy"])
    policy_hash = stable_json_sha256(policy_payload)
    factory_approvals = _mapping(policy_payload.get("approved_factories"))
    if policy_id != required_gate_policy_id:
        failures.append(
            f"gate policy identity mismatch: {policy_id!r} != {required_gate_policy_id!r}"
        )
    if required_gate_policy_id == DEFAULT_POLICY_ID:
        packaged_policy_hash = stable_json_sha256(
            load_evaluation_policy(DEFAULT_POLICY_PATH)
        )
        if policy_hash != packaged_policy_hash:
            failures.append(
                "bc0_closed_loop_hard_gate_v2 content does not match the packaged policy"
            )

    repository_root = Path(repo_root or Path(__file__).resolve().parents[2]).resolve()
    normalized_commit = str(expected_source_commit).strip().lower()
    suite_path = Path(expected_suite_path).expanduser().resolve(strict=True)
    normalized_suite_hash = file_sha256(suite_path)
    required_suites = [str(name) for name in suite_policy.get("required_suites") or []]
    expected_suites = load_evaluation_suites(suite_path)
    validate_release_scenario_suites(expected_suites)
    expected_suite_contract = fingerprint_evaluation_suites(
        expected_suites,
        seed=int(suite_policy.get("evaluator_seed", 0)),
        required_suites=required_suites,
        minimum_suites=len(required_suites),
        minimum_episodes_per_suite=1,
        minimum_roots_per_suite=_mapping(
            suite_policy.get("minimum_physical_roots_per_suite")
        ),
    )
    expected_roots = [
        str(row["physical_root"])
        for row in expected_suite_contract["episode_manifest"]
    ]
    if len(set(expected_roots)) != len(expected_roots):
        raise ValueError("expected suite physical roots must be globally unique")
    manifest_fields = {
        name: copy.deepcopy(expected_suite_contract[name])
        for name in _SUITE_MANIFEST_FIELDS
    }
    if suite_policy.get("status") != "pinned":
        failures.append("evaluation suite policy is not pinned")
    else:
        if suite_policy.get("approved_suite_sha256") != normalized_suite_hash:
            failures.append("expected suite file does not match the policy-pinned SHA-256")
        if suite_policy.get("approved_suite_manifest") != manifest_fields:
            failures.append("expected suite manifest does not match the packaged policy")
    normalized_protocol = str(expected_protocol).strip().lower()
    registry_hash = (
        str(expected_registry_sha256).strip().lower()
        if expected_registry_sha256 is not None
        else current_registry_sha256(normalized_protocol)
    )
    if _COMMIT.fullmatch(normalized_commit) is None:
        raise ValueError("expected_source_commit must be a 40-character commit")
    if _SHA256.fullmatch(registry_hash) is None:
        raise ValueError("expected_registry_sha256 must be a lowercase SHA-256")
    explicit_identity = str(expected_policy_identity or "").strip() or None
    model_id = str(expected_model_id or "").strip() or None
    model_revision = str(expected_model_revision or "").strip() or None
    if explicit_identity and (model_id or model_revision):
        raise ValueError("expect either an explicit policy identity or a model identity")
    if bool(model_id) != bool(model_revision):
        raise ValueError("expected_model_id and expected_model_revision are required together")
    if model_revision and _REVISION.fullmatch(model_revision) is None:
        raise ValueError("expected_model_revision must be an immutable 40- or 64-hex digest")
    if not explicit_identity and not model_id:
        raise ValueError("an expected policy or model identity is required")
    if normalized_role == "expert-baseline" and explicit_identity is None:
        raise ValueError("expert-baseline requires an explicit policy identity")
    if normalized_role in {"base-baseline", "checkpoint-promotion"} and model_id is None:
        raise ValueError(f"{normalized_role} requires an immutable model identity")
    if comparison_required and (
        normalized_reference_model_id,
        normalized_reference_revision,
    ) == (model_id, model_revision):
        raise ValueError(
            "checkpoint-promotion candidate and reference model identities must differ"
        )
    if require_current_clean_source:
        current_source = git_source_state(repository_root)
        if current_source.get("release_eligible_source") is not True:
            failures.append("current evaluation-gate source is not a clean tracked commit")
        if str(current_source.get("source_commit") or "").lower() != normalized_commit:
            failures.append("current evaluation-gate commit does not match the required commit")

    if payload.get("artifact_type") != "closed_loop_release_evaluation":
        failures.append("artifact_type is not closed_loop_release_evaluation")
    if (
        type(payload.get("artifact_schema_version")) is not int
        or payload.get("artifact_schema_version") != 2
    ):
        failures.append("artifact_schema_version is not 2")
    if payload.get("release_eligible") is not True or payload.get("release_failures") != []:
        failures.append("evaluator artifact is not release eligible")
    recorded_content_hash = payload.get("content_sha256")
    unsigned_payload = dict(payload)
    unsigned_payload.pop("content_sha256", None)
    if not isinstance(recorded_content_hash, str) or not _SHA256.fullmatch(
        recorded_content_hash
    ):
        failures.append("artifact content_sha256 is missing or invalid")
    elif _artifact_content_sha256(unsigned_payload) != recorded_content_hash:
        failures.append("artifact content_sha256 does not match its JSON content")

    provenance = _mapping(payload.get("provenance"))
    if (
        type(provenance.get("provenance_schema_version")) is not int
        or provenance.get("provenance_schema_version") != 1
    ):
        failures.append("provenance_schema_version is not exactly integer 1")
    if provenance.get("release_eligible") is not True or provenance.get(
        "release_failures"
    ) != []:
        failures.append("artifact provenance is not release eligible")
    source = _mapping(provenance.get("source_state"))
    if source.get("release_eligible_source") is not True:
        failures.append("evaluation source was not a clean release-eligible worktree")
    if str(source.get("source_commit") or "").lower() != normalized_commit:
        failures.append("evaluation source commit does not match the required commit")
    suite = _mapping(provenance.get("input_suite"))
    if str(suite.get("sha256") or "").lower() != normalized_suite_hash:
        failures.append("evaluation input suite does not match the frozen suite SHA-256")
    protocol = _mapping(provenance.get("protocol_registry"))
    if protocol.get("protocol") != normalized_protocol:
        failures.append("evaluation protocol does not match the required protocol")
    if str(protocol.get("registry_sha256") or "").lower() != registry_hash:
        failures.append("evaluation tool registry does not match the required registry")
    recorded_identity_hash = provenance.get("identity_sha256")
    identity_core = dict(provenance)
    for field in ("identity_sha256", "release_eligible", "release_failures"):
        identity_core.pop(field, None)
    if not isinstance(recorded_identity_hash, str) or stable_json_sha256(
        identity_core
    ) != recorded_identity_hash:
        failures.append("evaluation provenance identity_sha256 is invalid")
    evaluator_source = _mapping(provenance.get("evaluator_source"))
    _verify_source_descriptor(
        evaluator_source,
        label="evaluator",
        repo_root=repository_root,
        failures=failures,
    )
    expected_evaluator_source = _current_source_descriptor(
        Path(__file__).with_name("evaluator.py"), repo_root=repository_root
    )
    if dict(evaluator_source) != expected_evaluator_source:
        failures.append(
            "evaluator source identity does not match psse_env/dagger/evaluator.py"
        )
    factories = _mapping(provenance.get("factories"))
    for name in ("environment", "policy"):
        descriptor = _mapping(factories.get(name))
        source_descriptor = _mapping(descriptor.get("source"))
        if not str(descriptor.get("import_spec") or "").strip():
            failures.append(f"{name} factory identity is incomplete")
        _verify_source_descriptor(
            source_descriptor,
            label=f"{name} factory",
            repo_root=repository_root,
            failures=failures,
        )
    case_loader = factories.get("case_loader")
    if case_loader is not None:
        descriptor = _mapping(case_loader)
        if not str(descriptor.get("import_spec") or "").strip():
            failures.append("case loader identity is incomplete")
        _verify_source_descriptor(
            _mapping(descriptor.get("source")),
            label="case loader",
            repo_root=repository_root,
            failures=failures,
        )
    environment_descriptor = _mapping(factories.get("environment"))
    if not _factory_is_approved(
        environment_descriptor,
        list(factory_approvals.get("environment") or []),
    ):
        failures.append(
            "environment factory import spec/source hash is not approved by the gate policy"
        )
    policy_descriptor = _mapping(factories.get("policy"))
    policy_factory_role = "expert_policy" if explicit_identity is not None else "model_policy"
    if not _factory_is_approved(
        policy_descriptor,
        list(factory_approvals.get(policy_factory_role) or []),
    ):
        failures.append(
            f"{policy_factory_role} factory import spec/source hash is not approved by the gate policy"
        )
    if case_loader is not None and not _factory_is_approved(
        _mapping(case_loader),
        list(factory_approvals.get("case_loader") or []),
    ):
        failures.append(
            "case-loader import spec/source hash is not approved by the gate policy"
        )

    expected_identity = {
        "explicit_policy_identity": explicit_identity,
        "model_id": model_id,
        "model_revision": model_revision,
    }
    observed_identity = dict(_mapping(provenance.get("policy_identity")))
    if observed_identity != expected_identity:
        failures.append("evaluated policy/model identity does not match exactly")
    candidate_accelerator_classes: tuple[str, ...] = ()
    if normalized_role in {"base-baseline", "checkpoint-promotion"}:
        (
            candidate_accelerator_classes,
            accelerator_attestation_failures,
        ) = _model_accelerator_attestation(provenance)
        failures.extend(accelerator_attestation_failures)

    evaluation = _mapping(payload.get("evaluation"))
    suite_metrics = _mapping(evaluation.get("suite_metrics"))
    configuration = _mapping(suite_metrics.get("configuration"))
    coverage = _mapping(configuration.get("suite_coverage_validation"))
    if coverage.get("passed") is not True:
        failures.append("frozen suite coverage validation did not pass")
    expected_minimums = dict(
        _mapping(suite_policy.get("minimum_physical_roots_per_suite"))
    )
    distinct_minimums = set(expected_minimums.values())
    expected_minimum_configuration: Any = (
        next(iter(distinct_minimums))
        if len(distinct_minimums) == 1
        else expected_minimums
    )
    expected_configuration = {
        "seed": int(suite_policy.get("evaluator_seed", 0)),
        "max_steps": int(suite_policy.get("max_steps", 0)),
        "suite_names": expected_suite_contract["suite_names"],
        "required_suites": sorted(required_suites),
        "minimum_suites": len(required_suites),
        "minimum_episodes_per_suite": 1,
        "minimum_roots_per_suite": expected_minimum_configuration,
    }
    for name, expected_value in expected_configuration.items():
        if configuration.get(name) != expected_value:
            failures.append(
                f"evaluator configuration {name} does not match the suite policy"
            )
    suite_contract_fields = (
        "suite_manifest",
        "suite_content_hashes",
        "suite_root_set_hashes",
        "suite_content_sha256",
        "root_set_sha256",
        "suite_coverage_validation",
        "episode_order",
        "episode_manifest",
        "episode_manifest_sha256",
    )
    for name in suite_contract_fields:
        if configuration.get(name) != expected_suite_contract.get(name):
            failures.append(
                f"artifact {name} does not match the recomputed expected suite"
            )
    environment = _mapping(configuration.get("release_environment_validation"))
    if environment.get("passed") is not True:
        failures.append("release environment deployment attestation did not pass")
    scenario_schema_validation = configuration.get(
        "release_scenario_schema_validation"
    )
    if (
        not isinstance(scenario_schema_validation, Mapping)
        or set(scenario_schema_validation) != {"passed", "scenario_schema_version"}
        or scenario_schema_validation.get("passed") is not True
        or type(scenario_schema_validation.get("scenario_schema_version")) is not int
        or scenario_schema_validation.get("scenario_schema_version") != 1
    ):
        failures.append("release scenario schema-v1 validation evidence is missing")
    if configuration.get("custom_callback_validation") != {
        "passed": True,
        "physical_audit_callback": False,
        "tool_cost_callback": False,
    }:
        failures.append("release evaluator used or omitted custom-callback isolation evidence")

    raw_episodes = suite_metrics.get("episodes")
    episodes = list(raw_episodes) if isinstance(raw_episodes, list) else []
    if not episodes:
        failures.append("evaluation artifact contains no episode evidence")
    episodes_checked = environment.get("episodes_checked")
    if (
        isinstance(episodes_checked, bool)
        or not isinstance(episodes_checked, int)
        or episodes_checked != len(episodes)
    ):
        failures.append("release environment attestation did not cover every episode")
    required_environment = _mapping(environment.get("required"))
    if required_environment != {
        "production_dataset_mode": True,
        "candidate_quality_oracle_mode": "deployment",
    }:
        failures.append("release environment required contract is missing or altered")
    observed_environments = environment.get("observed")
    if not isinstance(observed_environments, list) or not observed_environments:
        failures.append("release environment observations are missing")
    elif any(dict(_mapping(row)) != required_environment for row in observed_environments):
        failures.append("one or more evaluated environments were not deployment mode")
    identity_validation = _mapping(
        configuration.get("policy_identity_validation")
    )
    if identity_validation.get("passed") is not True:
        failures.append("instantiated policy identity attestation did not pass")
    identity_episodes_checked = identity_validation.get("episodes_checked")
    if (
        isinstance(identity_episodes_checked, bool)
        or not isinstance(identity_episodes_checked, int)
        or identity_episodes_checked != len(episodes)
    ):
        failures.append("policy identity attestation did not cover every episode")
    if identity_validation.get("required") != expected_identity:
        failures.append("policy identity attestation required identity was altered")
    if identity_validation.get("observed") != [expected_identity]:
        failures.append("policy identity observations do not match the required identity")
    if identity_validation.get("failures") != []:
        failures.append("policy identity attestation contains failures")
    keys: set[str] = set()
    physical_roots: set[str] = set()
    by_family: dict[str, list[Mapping[str, Any]]] = {}
    by_suite: dict[str, list[Mapping[str, Any]]] = {}
    expected_episode_by_key = {
        str(row["episode_key"]): row
        for row in expected_suite_contract["episode_manifest"]
    }
    terminal = resolved = escalated = invalid = loops = evaluator_errors = 0
    false_commit = false_rollback = false_finalization = corruption = 0
    injected_failures = recovered_injected = injected_failure_episodes = 0
    maximum_episode_invalid = 0
    max_steps_seen = 0
    for index, raw_episode in enumerate(episodes):
        if not isinstance(raw_episode, Mapping):
            failures.append(f"episode[{index}] is not a JSON object")
            continue
        episode = raw_episode
        missing_evidence = sorted(_REQUIRED_EPISODE_FIELDS - set(episode))
        if missing_evidence:
            failures.append(
                f"episode[{index}] is missing required evidence: "
                + ", ".join(missing_evidence)
            )

        def episode_integer(field: str) -> int:
            try:
                return _nonnegative_integer(
                    episode.get(field), field=f"episode[{index}].{field}"
                )
            except ValueError as exc:
                failures.append(str(exc))
                return 0

        for boolean_field in (
            "terminal",
            "final_physical_success",
            "physical_correctness_known",
            "final_physical_correct",
            "healthy_preservation_known",
            "healthy_components_preserved",
            "loop_detected",
        ):
            if not isinstance(episode.get(boolean_field), bool):
                failures.append(
                    f"episode[{index}].{boolean_field} must be an explicit boolean"
                )
        key = str(episode.get("episode_key") or "")
        suite_name = str(episode.get("suite") or "")
        scenario_id = str(episode.get("scenario_id") or "")
        family = str(episode.get("family") or "")
        root = str(episode.get("physical_root") or "")
        if not key or key in keys:
            failures.append(f"episode[{index}] has a missing or duplicate episode_key")
        keys.add(key)
        if suite_name not in required_suites:
            failures.append(
                f"episode[{index}] has missing or unapproved suite membership"
            )
        expected_episode = expected_episode_by_key.get(key)
        try:
            episode_seed = _nonnegative_integer(
                episode.get("seed"), field=f"episode[{index}].seed"
            )
        except ValueError as exc:
            failures.append(str(exc))
            episode_seed = -1
        observed_episode_identity = {
            "episode_key": key,
            "scenario_id": scenario_id,
            "scenario_index": (
                int(key.rsplit(":", 1)[1])
                if key.rsplit(":", 1)[-1].isdigit()
                else -1
            ),
            "suite": suite_name,
            "family": family,
            "cardinality": episode.get("cardinality"),
            "case": episode.get("case"),
            "split": episode.get("split"),
            "source_tier": episode.get("source_tier"),
            "physical_root": root,
            "seed": episode_seed,
            "evaluation_intervention": _mapping(
                episode.get("evaluation_intervention")
            ).get("contract"),
        }
        if expected_episode is None or observed_episode_identity != expected_episode:
            failures.append(
                f"episode[{index}] identity does not match the frozen suite manifest"
            )
        if not family or not root or root in physical_roots:
            failures.append(
                f"episode[{index}] has a missing or globally duplicate physical-root identity"
            )
        physical_roots.add(root)
        by_family.setdefault(family, []).append(episode)
        by_suite.setdefault(suite_name, []).append(episode)
        is_terminal = episode.get("terminal") is True
        terminal += int(is_terminal)
        outcome = episode.get("terminal_outcome")
        is_resolved = bool(
            is_terminal
            and outcome == "resolved"
            and episode.get("final_physical_success") is True
        )
        is_escalated = bool(is_terminal and outcome == "operator_escalation")
        resolved += int(is_resolved)
        escalated += int(is_escalated)
        if is_terminal and not is_resolved and not is_escalated:
            failures.append(f"episode {key!r} has an unknown or unaudited terminal outcome")
        if episode.get("healthy_components_preserved") is True and episode.get(
            "healthy_preservation_known"
        ) is not True:
            failures.append(f"episode {key!r} has inconsistent healthy-preservation evidence")
        if episode.get("final_physical_correct") is True and episode.get(
            "physical_correctness_known"
        ) is not True:
            failures.append(f"episode {key!r} has inconsistent physical-correctness evidence")
        if outcome == "resolved" and not all(
            episode.get(field) is True
            for field in (
                "physical_correctness_known",
                "final_physical_correct",
                "healthy_preservation_known",
                "healthy_components_preserved",
            )
        ):
            performance_failures.append(
                f"resolved episode {key!r} did not achieve complete physical safety"
            )
        audit_evidence, audit_performance = _strict_episode_audit_failures(episode)
        failures.extend(
            f"episode {key!r} audit evidence: {failure}"
            for failure in audit_evidence
        )
        performance_failures.extend(
            f"episode {key!r} audit performance: {failure}"
            for failure in audit_performance
        )
        intervention_evidence, intervention_performance = _intervention_failures(
            episode,
            _mapping(expected_episode).get("evaluation_intervention"),
        )
        failures.extend(
            f"episode {key!r} intervention evidence: {failure}"
            for failure in intervention_evidence
        )
        performance_failures.extend(
            f"episode {key!r} intervention performance: {failure}"
            for failure in intervention_performance
        )
        intervention_summary = _mapping(episode.get("evaluation_intervention"))
        raw_injected = intervention_summary.get("injected_failure_count")
        raw_recovered_injected = intervention_summary.get("recovered_failure_count")
        if (
            isinstance(raw_injected, int)
            and not isinstance(raw_injected, bool)
            and raw_injected >= 0
        ):
            injected_failures += raw_injected
            injected_failure_episodes += int(raw_injected > 0)
        if (
            isinstance(raw_recovered_injected, int)
            and not isinstance(raw_recovered_injected, bool)
            and raw_recovered_injected >= 0
        ):
            recovered_injected += raw_recovered_injected
        episode_environment = _mapping(
            episode.get("release_environment_attestation")
        )
        if (
            episode_environment.get("passed") is not True
            or episode_environment.get("production_dataset_mode") is not True
            or episode_environment.get("candidate_quality_oracle_mode") != "deployment"
            or episode_environment.get("failures") != []
        ):
            failures.append(
                f"episode {key!r} lacks a passing deployment-environment attestation"
            )
        episode_policy_identity = _mapping(
            episode.get("policy_identity_attestation")
        )
        if (
            episode_policy_identity.get("passed") is not True
            or episode_policy_identity.get("required") != expected_identity
            or episode_policy_identity.get("actual") != expected_identity
            or episode_policy_identity.get("failures") != []
        ):
            failures.append(
                f"episode {key!r} lacks a matching instantiated-policy identity attestation"
            )
        false_commit += episode_integer("false_commit_count")
        false_rollback += episode_integer("false_rollback_count")
        false_finalization += episode_integer("false_finalization_count")
        episode_invalid = episode_integer("invalid_action_count")
        invalid += episode_invalid
        maximum_episode_invalid = max(maximum_episode_invalid, episode_invalid)
        loops += int(episode.get("loop_detected") is True)
        evaluator_errors += int(episode.get("evaluator_error") is not None)
        steps = episode_integer("steps")
        policy_steps = episode_integer("policy_steps")
        pre_policy_steps = _mapping(
            episode.get("evaluation_intervention")
        ).get("pre_policy_step_count")
        if (
            isinstance(pre_policy_steps, bool)
            or not isinstance(pre_policy_steps, int)
            or steps != policy_steps + max(pre_policy_steps, 0)
        ):
            failures.append(
                f"episode {key!r} step count does not match policy plus intervention steps"
            )
        max_steps_seen = max(max_steps_seen, policy_steps)
        if (
            episode.get("healthy_preservation_known") is True
            and episode.get("healthy_components_preserved") is False
        ):
            corruption += 1

    if set(by_suite) != set(required_suites):
        failures.append("episode suite membership does not cover every required suite")
    if set(keys) != set(expected_episode_by_key):
        failures.append("episode keys do not exactly match the frozen suite")
    expected_manifest = _mapping(expected_suite_contract.get("suite_manifest"))
    for suite_name in required_suites:
        rows = by_suite.get(suite_name, [])
        roots = sorted({str(row.get("physical_root") or "") for row in rows})
        expected_row = _mapping(expected_manifest.get(suite_name))
        if len(rows) != expected_row.get("episodes"):
            failures.append(
                f"suite {suite_name!r} episode count does not match the frozen suite"
            )
        if len(roots) != expected_row.get("distinct_physical_roots"):
            failures.append(
                f"suite {suite_name!r} physical-root count does not match the frozen suite"
            )
        if stable_json_sha256(roots) != expected_row.get("root_set_sha256"):
            failures.append(
                f"suite {suite_name!r} physical-root set does not match the frozen suite"
            )

    total = len(episodes)
    terminal_rate = terminal / total if total else 0.0
    overall = _mapping(suite_metrics.get("overall"))
    for name, expected in (
        ("episodes", total),
        ("terminal_episodes", terminal),
        ("terminal_rate", terminal_rate),
        ("resolved_episodes", resolved),
        ("resolution_rate", resolved / total if total else 0.0),
        ("operator_escalation_episodes", escalated),
        ("operator_escalation_rate", escalated / total if total else 0.0),
        ("false_commit_count", false_commit),
        ("false_rollback_count", false_rollback),
        ("false_finalization_count", false_finalization),
        ("healthy_component_corruption_episodes", corruption),
        ("invalid_action_count", invalid),
        ("injected_failure_count", injected_failures),
        ("recovered_injected_failures", recovered_injected),
        (
            "injected_failure_recovery_rate",
            recovered_injected / injected_failures if injected_failures else 0.0,
        ),
        ("episodes_with_injected_failures", injected_failure_episodes),
        ("loop_episodes", loops),
        ("evaluator_error_episodes", evaluator_errors),
    ):
        _reported_number(overall, name, expected, failures)

    constraints = {
        "maximum_false_commit_count": false_commit,
        "maximum_false_finalization_count": false_finalization,
        "maximum_false_rollback_count": false_rollback,
        "maximum_healthy_component_corruption_episodes": corruption,
        "maximum_steps_per_episode": max_steps_seen,
    }
    for name, observed_value in constraints.items():
        allowed = int(hard[name])
        if observed_value > allowed:
            performance_failures.append(
                f"{name} failed: observed {observed_value} > allowed {allowed}"
            )
    if terminal_rate < float(hard["minimum_terminal_rate"]):
        performance_failures.append(
            "minimum_terminal_rate failed: "
            f"observed {terminal_rate:.6f} < required {float(hard['minimum_terminal_rate']):.6f}"
        )
    invalid_rate = invalid / total if total else 0.0
    loop_rate = loops / total if total else 0.0
    if maximum_episode_invalid > int(hard["maximum_invalid_actions_per_episode"]):
        performance_failures.append(
            "maximum_invalid_actions_per_episode failed: "
            f"observed {maximum_episode_invalid} > allowed "
            f"{int(hard['maximum_invalid_actions_per_episode'])}"
        )
    if invalid_rate > float(hard["maximum_invalid_action_rate"]):
        performance_failures.append(
            "maximum_invalid_action_rate failed: "
            f"observed {invalid_rate:.6f} > allowed "
            f"{float(hard['maximum_invalid_action_rate']):.6f}"
        )
    if loop_rate > float(hard["maximum_loop_episode_rate"]):
        performance_failures.append(
            "maximum_loop_episode_rate failed: "
            f"observed {loop_rate:.6f} > allowed "
            f"{float(hard['maximum_loop_episode_rate']):.6f}"
        )
    if evaluator_errors:
        failures.append(f"evaluator errors are forbidden: observed {evaluator_errors}")

    family_observed: dict[str, Any] = {}
    unplanned = sorted(set(by_family) - set(families))
    if unplanned:
        failures.append("evaluation contains families without policy: " + ", ".join(unplanned))
    for family, requirements in families.items():
        rows = by_family.get(str(family), [])
        roots = {str(row.get("physical_root")) for row in rows}
        family_resolved = sum(
            row.get("terminal") is True
            and row.get("terminal_outcome") == "resolved"
            and row.get("final_physical_success") is True
            for row in rows
        )
        family_escalated = sum(
            row.get("terminal") is True
            and row.get("terminal_outcome") == "operator_escalation"
            for row in rows
        )
        denominator = len(roots)
        resolution_rate = family_resolved / denominator if denominator else 0.0
        escalation_rate = family_escalated / denominator if denominator else 0.0
        family_observed[str(family)] = {
            "distinct_physical_roots": denominator,
            "resolved_roots": family_resolved,
            "operator_escalation_roots": family_escalated,
            "resolution_rate": resolution_rate,
            "operator_escalation_rate": escalation_rate,
        }
        minimum_roots = int(requirements["minimum_physical_roots"])
        minimum_resolution = float(requirements["minimum_resolution_rate"])
        maximum_escalation = float(requirements["maximum_operator_escalation_rate"])
        if denominator < minimum_roots:
            failures.append(
                f"family {family!r} roots failed: {denominator} < {minimum_roots}"
            )
        if resolution_rate < minimum_resolution:
            performance_failures.append(
                f"family {family!r} resolution failed: "
                f"{resolution_rate:.6f} < {minimum_resolution:.6f}"
            )
        if escalation_rate > maximum_escalation:
            performance_failures.append(
                f"family {family!r} escalation failed: "
                f"{escalation_rate:.6f} > {maximum_escalation:.6f}"
            )

    comparison_observed: dict[str, Any] | None = None
    if comparison_required:
        assert reference_artifact is not None
        assert normalized_reference_model_id is not None
        assert normalized_reference_revision is not None
        reference_result = validate_evaluation_artifact(
            reference_artifact,
            role="base-baseline",
            policy=policy_payload,
            expected_source_commit=normalized_commit,
            expected_suite_path=suite_path,
            expected_protocol=normalized_protocol,
            expected_registry_sha256=registry_hash,
            expected_model_id=normalized_reference_model_id,
            expected_model_revision=normalized_reference_revision,
            required_gate_policy_id=required_gate_policy_id,
            repo_root=repository_root,
            require_current_clean_source=require_current_clean_source,
        )
        reference_payload = _load_artifact_payload(reference_artifact)
        reference_provenance = _mapping(reference_payload.get("provenance"))
        reference_accelerator_classes, _ = _model_accelerator_attestation(
            reference_provenance
        )
        reference_configuration = _mapping(
            _mapping(_mapping(reference_payload.get("evaluation")).get("suite_metrics")).get(
                "configuration"
            )
        )
        candidate_configuration_contract = copy.deepcopy(dict(configuration))
        candidate_configuration_contract.pop("policy_identity_validation", None)
        reference_configuration_contract = copy.deepcopy(
            dict(reference_configuration)
        )
        reference_configuration_contract.pop("policy_identity_validation", None)
        exact_reference_contract = {
            "environment_factory": (
                _mapping(factories.get("environment")),
                _mapping(_mapping(reference_provenance.get("factories")).get("environment")),
            ),
            "case_loader": (
                factories.get("case_loader"),
                _mapping(reference_provenance.get("factories")).get("case_loader"),
            ),
            "policy_factory": (
                _mapping(factories.get("policy")),
                _mapping(_mapping(reference_provenance.get("factories")).get("policy")),
            ),
            "evaluator_source": (
                evaluator_source,
                _mapping(reference_provenance.get("evaluator_source")),
            ),
            "protocol_registry": (
                protocol,
                _mapping(reference_provenance.get("protocol_registry")),
            ),
            "accelerator_class": (
                candidate_accelerator_classes,
                reference_accelerator_classes,
            ),
            "evaluator_configuration": (
                candidate_configuration_contract,
                reference_configuration_contract,
            ),
        }
        reference_contract_failures = [
            name
            for name, (candidate_value, reference_value) in exact_reference_contract.items()
            if candidate_value != reference_value
        ]
        candidate_artifact_identity_hash = artifact_file_hash or recorded_content_hash
        reference_artifact_identity_hash = (
            reference_result.artifact_sha256
            or reference_result.artifact_content_sha256
        )
        if (
            candidate_artifact_identity_hash is not None
            and candidate_artifact_identity_hash == reference_artifact_identity_hash
        ):
            reference_contract_failures.append("artifact_identity")
        if not reference_result.evidence_passed:
            comparison_failures.append(
                "comparison reference did not pass the identity and evidence gate"
            )
            evidence_failures.extend(
                f"comparison reference: {failure}"
                for failure in reference_result.evidence_failures
            )
        if reference_contract_failures:
            comparison_failures.append(
                "comparison reference contract differs: "
                + ", ".join(reference_contract_failures)
            )
            evidence_failures.extend(
                f"comparison reference {name} does not exactly match the candidate"
                for name in reference_contract_failures
            )
        regressions, paired_summary = _paired_safety_regressions(
            payload, reference_payload
        )
        if regressions:
            comparison_failures.append(
                f"paired safety non-regression failed on {len(regressions)} record(s)"
            )
            performance_failures.extend(
                "checkpoint paired non-regression failed: "
                + str(row.get("episode_key") or row.get("reason") or "unknown")
                + " ("
                + str(row.get("reason") or "unknown")
                + ")"
                for row in regressions
            )
        comparison_passed = bool(
            reference_result.evidence_passed
            and not reference_contract_failures
            and not regressions
        )
        comparison_observed = {
            "required": True,
            "passed": comparison_passed,
            "candidate_artifact_sha256": candidate_artifact_identity_hash,
            "reference_artifact_sha256": reference_artifact_identity_hash,
            "reference_validation_role": reference_result.validation_role,
            "reference_model_id": normalized_reference_model_id,
            "reference_model_revision": normalized_reference_revision,
            "reference_evidence_passed": reference_result.evidence_passed,
            "reference_performance_passed": reference_result.performance_passed,
            "reference_performance_failures": list(
                reference_result.performance_failures
            ),
            "candidate_accelerator_classes": list(
                candidate_accelerator_classes
            ),
            "reference_accelerator_classes": list(
                reference_accelerator_classes
            ),
            "failures": list(comparison_failures),
            "regressions": regressions,
            **paired_summary,
        }

    observed = {
        "episodes": total,
        "terminal_rate": terminal_rate,
        "resolved_roots": resolved,
        "operator_escalation_roots": escalated,
        "false_commit_count": false_commit,
        "false_finalization_count": false_finalization,
        "false_rollback_count": false_rollback,
        "healthy_component_corruption_episodes": corruption,
        "invalid_action_count": invalid,
        "invalid_action_rate": invalid_rate,
        "maximum_invalid_actions_per_episode": maximum_episode_invalid,
        "loop_episodes": loops,
        "loop_episode_rate": loop_rate,
        "evaluator_error_episodes": evaluator_errors,
        "maximum_steps_per_episode": max_steps_seen,
        "accelerator_classes": list(candidate_accelerator_classes),
        "families": family_observed,
        "suites": {
            suite_name: {
                "episodes": len(by_suite.get(suite_name, [])),
                "distinct_physical_roots": len(
                    {
                        str(row.get("physical_root") or "")
                        for row in by_suite.get(suite_name, [])
                    }
                ),
            }
            for suite_name in required_suites
        },
        "paired_nonregression": comparison_observed,
    }
    unique_evidence_failures = tuple(dict.fromkeys(evidence_failures))
    unique_performance_failures = tuple(dict.fromkeys(performance_failures))
    blocking_failures = list(unique_evidence_failures)
    if performance_enforced:
        blocking_failures.extend(unique_performance_failures)
    return EvaluationGateResult(
        passed=not blocking_failures,
        failures=tuple(dict.fromkeys(blocking_failures)),
        validation_role=normalized_role,
        evidence_passed=not unique_evidence_failures,
        performance_passed=not unique_performance_failures,
        performance_enforced=performance_enforced,
        evidence_failures=unique_evidence_failures,
        performance_failures=unique_performance_failures,
        comparison_required=comparison_required,
        comparison_passed=comparison_passed,
        comparison_failures=tuple(dict.fromkeys(comparison_failures)),
        artifact_path=str(artifact_path) if artifact_path is not None else None,
        artifact_sha256=artifact_file_hash,
        artifact_content_sha256=(
            str(recorded_content_hash) if isinstance(recorded_content_hash, str) else None
        ),
        source_commit=str(source.get("source_commit") or "") or None,
        frozen_suite_sha256=str(suite.get("sha256") or "") or None,
        protocol=str(protocol.get("protocol") or "") or None,
        registry_sha256=str(protocol.get("registry_sha256") or "") or None,
        evaluated_policy_identity=observed_identity,
        gate_policy_id=policy_id or None,
        gate_policy_sha256=policy_hash,
        observed=observed,
    )


def _resolve_commit(value: str, *, repo_root: Path) -> str:
    normalized = str(value).strip()
    if normalized.upper() != "HEAD":
        return normalized
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("could not resolve HEAD for evaluation gate") from exc


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate closed-loop artifact identity and hard BC0 performance "
            "constraints; scalar score is never a promotion criterion."
        )
    )
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument(
        "--role",
        choices=("expert-baseline", "base-baseline", "checkpoint-promotion"),
        required=True,
        help="Release decision being gated; promotion always requires a model identity.",
    )
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY_PATH)
    parser.add_argument(
        "--required-gate-policy-id", default=DEFAULT_POLICY_ID
    )
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-suite", required=True, type=Path)
    parser.add_argument(
        "--expected-protocol", choices=("canonical", "controller"), default="canonical"
    )
    parser.add_argument("--expected-registry-sha256")
    identity = parser.add_mutually_exclusive_group(required=True)
    identity.add_argument("--expected-policy-identity")
    identity.add_argument("--expected-model-id")
    parser.add_argument("--expected-model-revision")
    parser.add_argument(
        "--reference-artifact",
        type=Path,
        help="Evidence-valid base-model artifact used for paired checkpoint comparison.",
    )
    parser.add_argument("--reference-model-id")
    parser.add_argument("--reference-model-revision")
    parser.add_argument("--report-output", type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if bool(args.expected_model_id) != bool(args.expected_model_revision):
        parser.error(
            "--expected-model-id and --expected-model-revision must be supplied together"
        )
    if args.role == "expert-baseline" and not args.expected_policy_identity:
        parser.error("--role expert-baseline requires --expected-policy-identity")
    if args.role in {"base-baseline", "checkpoint-promotion"} and not args.expected_model_id:
        parser.error(f"--role {args.role} requires --expected-model-id/revision")
    comparison_values = (
        args.reference_artifact,
        args.reference_model_id,
        args.reference_model_revision,
    )
    if args.role == "checkpoint-promotion" and not all(comparison_values):
        parser.error(
            "--role checkpoint-promotion requires --reference-artifact, "
            "--reference-model-id, and --reference-model-revision"
        )
    if args.role != "checkpoint-promotion" and any(comparison_values):
        parser.error("--reference-* arguments are only valid for checkpoint-promotion")
    try:
        result = validate_evaluation_artifact(
            args.artifact,
            role=args.role,
            policy=args.policy,
            expected_source_commit=_resolve_commit(
                args.expected_source_commit,
                repo_root=Path(__file__).resolve().parents[2],
            ),
            expected_suite_path=args.expected_suite,
            expected_protocol=args.expected_protocol,
            expected_registry_sha256=args.expected_registry_sha256,
            expected_policy_identity=args.expected_policy_identity,
            expected_model_id=args.expected_model_id,
            expected_model_revision=args.expected_model_revision,
            reference_artifact=args.reference_artifact,
            reference_model_id=args.reference_model_id,
            reference_model_revision=args.reference_model_revision,
            required_gate_policy_id=args.required_gate_policy_id,
        )
        report = result.as_dict()
        rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.report_output is not None:
            args.report_output.parent.mkdir(parents=True, exist_ok=True)
            args.report_output.write_text(rendered, encoding="utf-8")
        stream = sys.stdout if result.passed else sys.stderr
        print(rendered, end="", file=stream)
        return 0 if result.passed else 2
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(
            json.dumps({"passed": False, "error": str(exc)}, indent=2),
            file=sys.stderr,
        )
        return 2


__all__ = [
    "DEFAULT_POLICY_ID",
    "DEFAULT_POLICY_PATH",
    "EvaluationGateResult",
    "current_registry_sha256",
    "load_evaluation_policy",
    "main",
    "validate_evaluation_artifact",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
