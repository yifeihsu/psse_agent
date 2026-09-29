"""Load and validate the closed-loop evaluation policy.

``bc0_evaluation_policy.json`` fixes the evaluation suites, their per-family
quotas and the hard constraints (false commits, rollbacks and finalizations,
invalid actions, loops, the step budget).  The expert-aggregate builder reads
it through ``load_evaluation_policy``.
"""

from __future__ import annotations

import copy
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping

from psse_env.episode_budget import DEFAULT_EPISODE_ACTION_LIMIT

from psse_env.sft.provenance import stable_json_sha256


DEFAULT_POLICY_PATH = Path(__file__).with_name("bc0_evaluation_policy.json")
DEFAULT_POLICY_ID = "bc0_closed_loop_hard_gate_v4"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
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
            # A row names a factory by import spec.  An accompanying
            # ``source_sha256`` is accepted as recorded provenance only; the
            # gate does not compare it against the current source.
            if not isinstance(row, Mapping) or not (
                {"import_spec"} <= set(row) <= {"import_spec", "source_sha256"}
            ):
                raise ValueError(
                    f"approved_factories.{role}[{index}] has an invalid schema"
                )
            spec = str(row.get("import_spec") or "").strip()
            source_hash = str(row.get("source_sha256") or "").strip().lower()
            if not spec or (source_hash and _SHA256.fullmatch(source_hash) is None):
                raise ValueError(
                    f"approved_factories.{role}[{index}] identity is invalid"
                )
            approved_row = {"import_spec": spec}
            if source_hash:
                approved_row["source_sha256"] = source_hash
            approved.append(approved_row)
        if len({row["import_spec"] for row in approved}) != len(approved):
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
        or policy.get("policy_schema_version") != 3
    ):
        raise ValueError("evaluation policy_schema_version must be 3")
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
    if policy.get("policy_id") == DEFAULT_POLICY_ID and (
        suite_policy["max_steps"] != DEFAULT_EPISODE_ACTION_LIMIT
        or hard["maximum_steps_per_episode"] != DEFAULT_EPISODE_ACTION_LIMIT
    ):
        raise ValueError(f"current BC0 policy must use {DEFAULT_EPISODE_ACTION_LIMIT} actions for both suite and hard episode limits")
    if float(hard["minimum_terminal_rate"]) != 1.0:
        raise ValueError("BC0 evaluation policy must require 100% terminality")
    # These are non-negotiable safety constraints.  A custom policy cannot
    # acquire the BC0 identity while weakening them.
    for name in (
        "maximum_false_commit_count",
        "maximum_false_finalization_count",
        "maximum_false_rollback_count",
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
            "minimum_audited_completion_rate",
            "maximum_unqualified_operator_escalation_rate",
        }:
            raise ValueError(f"family policy for {family!r} has an invalid schema")
        if _nonnegative_integer(
            requirements["minimum_physical_roots"],
            field=f"{family}.minimum_physical_roots",
        ) < 1:
            raise ValueError(f"{family}.minimum_physical_roots must be positive")
        _rate(
            requirements["minimum_audited_completion_rate"],
            field=f"{family}.minimum_audited_completion_rate",
        )
        _rate(
            requirements["maximum_unqualified_operator_escalation_rate"],
            field=f"{family}.maximum_unqualified_operator_escalation_rate",
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


__all__ = [
    "DEFAULT_POLICY_ID",
    "DEFAULT_POLICY_PATH",
    "load_evaluation_policy",
]
