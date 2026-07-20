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

from psse_env.dagger.dataset_builder import TOOL_JSON_SCHEMAS
from psse_env.dagger.protocol_bridge import unified_tool_schemas
from psse_env.sft.provenance import file_sha256, git_source_state, stable_json_sha256


DEFAULT_POLICY_PATH = Path(__file__).with_name("bc0_evaluation_policy.json")
DEFAULT_POLICY_ID = "bc0_closed_loop_hard_gate_v1"
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
_STRICT_HEALTHY_CHECKS = (
    "healthy_measurements_preserved",
    "healthy_case_components_preserved",
)
_STRICT_NONREGRESSION_CHECK = "accepted_target_nonregression"
_REQUIRED_EPISODE_FIELDS = frozenset(
    {
        "episode_key",
        "family",
        "physical_root",
        "steps",
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
        "invalid_action_count",
        "loop_detected",
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


def _nonnegative_integer(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a non-negative integer")
    try:
        parsed = int(value)
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be a non-negative integer") from exc
    if parsed < 0 or not math.isfinite(numeric) or numeric != parsed:
        raise ValueError(f"{field} must be a non-negative integer")
    return parsed


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


def load_evaluation_policy(
    path: str | Path = DEFAULT_POLICY_PATH,
) -> dict[str, Any]:
    policy_path = Path(path).expanduser().resolve(strict=True)
    decoded = json.loads(policy_path.read_text(encoding="utf-8"))
    if not isinstance(decoded, Mapping):
        raise ValueError("evaluation policy must be a JSON object")
    policy = copy.deepcopy(dict(decoded))
    if policy.get("policy_schema_version") != 1:
        raise ValueError("evaluation policy_schema_version must be 1")
    if not str(policy.get("policy_id") or "").strip():
        raise ValueError("evaluation policy_id must be non-empty")
    policy["approved_factories"] = _validate_factory_approval_policy(
        policy.get("approved_factories")
    )
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


def _strict_episode_audit_passed(episode: Mapping[str, Any]) -> bool:
    audit = _mapping(episode.get("audit"))
    strict = _mapping(audit.get("strict_release_audit"))
    if audit.get("audit_mode") != "strict_release_audit":
        return False
    if audit.get("quarantined") is not False or strict.get("quarantined") is not False:
        return False
    if strict.get("audit_version") != "strict_offline_episode_truth_v3":
        return False
    if strict.get("terminal") is not episode.get("terminal"):
        return False
    if strict.get("terminal_outcome") != episode.get("terminal_outcome"):
        return False
    if strict.get("scenario_family") != episode.get("family"):
        return False
    if strict.get("physical_root_fingerprint") != episode.get("physical_root"):
        return False
    if strict.get("problems") != []:
        return False
    checks = _mapping(strict.get("checks"))
    if _mapping(checks.get("accepted_correction_targets")).get("status") != "passed":
        return False
    if not all(
        _mapping(checks.get(name)).get("status") == "passed"
        for name in _STRICT_HEALTHY_CHECKS
    ):
        return False
    if _mapping(checks.get(_STRICT_NONREGRESSION_CHECK)).get("status") != "passed":
        return False
    if episode.get("terminal_outcome") == "resolved":
        if _mapping(checks.get("remaining_true_faults")).get("status") != "passed":
            return False
        if _mapping(checks.get("final_case_matches_clean")).get("status") != "passed":
            return False
        final_measurements = _mapping(checks.get("final_measurements_match_clean"))
        measurement_status = final_measurements.get("status")
        if measurement_status == "not_applicable":
            if not str(final_measurements.get("reason") or "").strip():
                return False
        elif measurement_status != "passed":
            return False
    return True


def validate_evaluation_artifact(
    artifact: str | Path | Mapping[str, Any],
    *,
    policy: str | Path | Mapping[str, Any] = DEFAULT_POLICY_PATH,
    expected_source_commit: str,
    expected_suite_sha256: str,
    expected_protocol: str = "canonical",
    expected_registry_sha256: str | None = None,
    expected_policy_identity: str | None = None,
    expected_model_id: str | None = None,
    expected_model_revision: str | None = None,
    required_gate_policy_id: str = DEFAULT_POLICY_ID,
    repo_root: str | Path | None = None,
    require_current_clean_source: bool = True,
) -> EvaluationGateResult:
    """Validate one evaluator artifact without consulting its scalar score."""

    failures: list[str] = []
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
        policy_payload = copy.deepcopy(dict(policy))
        # Reuse the strict loader validation by applying the same checks below.
        temporary_policy = policy_payload
        if temporary_policy.get("policy_schema_version") != 1:
            raise ValueError("evaluation policy_schema_version must be 1")
        temporary_policy["approved_factories"] = _validate_factory_approval_policy(
            temporary_policy.get("approved_factories")
        )
        # Mapping policies are primarily for library/tests; canonicalize and
        # validate through a compact local mirror of the file-backed contract.
        policy_id = str(temporary_policy.get("policy_id") or "").strip()
        hard = _mapping(temporary_policy.get("hard_constraints"))
        families = _mapping(temporary_policy.get("family_policy"))
        if not policy_id or set(hard) != _HARD_CONSTRAINTS or not families:
            raise ValueError("evaluation policy mapping has an invalid schema")
        rate_constraints = {
            "minimum_terminal_rate",
            "maximum_invalid_action_rate",
            "maximum_loop_episode_rate",
        }
        for name in _HARD_CONSTRAINTS - rate_constraints:
            _nonnegative_integer(hard[name], field=name)
        for name in rate_constraints:
            _rate(hard[name], field=name)
        if float(hard["minimum_terminal_rate"]) != 1.0:
            raise ValueError("BC0 evaluation policy must require 100% terminality")
        for name in (
            "maximum_false_commit_count",
            "maximum_false_finalization_count",
            "maximum_healthy_component_corruption_episodes",
        ):
            if _nonnegative_integer(hard[name], field=name) != 0:
                raise ValueError(f"BC0 evaluation policy must set {name}=0")
        for family, requirements in families.items():
            requirements = _mapping(requirements)
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
            _rate(requirements["minimum_resolution_rate"], field="minimum_resolution_rate")
            _rate(
                requirements["maximum_operator_escalation_rate"],
                field="maximum_operator_escalation_rate",
            )
    else:
        policy_payload = load_evaluation_policy(policy)
        policy_id = str(policy_payload["policy_id"])
        hard = _mapping(policy_payload["hard_constraints"])
        families = _mapping(policy_payload["family_policy"])
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
                "bc0_closed_loop_hard_gate_v1 content does not match the packaged policy"
            )

    root = Path(repo_root or Path(__file__).resolve().parents[2]).resolve()
    normalized_commit = str(expected_source_commit).strip().lower()
    normalized_suite_hash = str(expected_suite_sha256).strip().lower()
    normalized_protocol = str(expected_protocol).strip().lower()
    registry_hash = (
        str(expected_registry_sha256).strip().lower()
        if expected_registry_sha256 is not None
        else current_registry_sha256(normalized_protocol)
    )
    if _COMMIT.fullmatch(normalized_commit) is None:
        raise ValueError("expected_source_commit must be a 40-character commit")
    if _SHA256.fullmatch(normalized_suite_hash) is None:
        raise ValueError("expected_suite_sha256 must be a lowercase SHA-256")
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
    if require_current_clean_source:
        current_source = git_source_state(root)
        if current_source.get("release_eligible_source") is not True:
            failures.append("current evaluation-gate source is not a clean tracked commit")
        if str(current_source.get("source_commit") or "").lower() != normalized_commit:
            failures.append("current evaluation-gate commit does not match the required commit")

    if payload.get("artifact_type") != "closed_loop_release_evaluation":
        failures.append("artifact_type is not closed_loop_release_evaluation")
    if payload.get("artifact_schema_version") != 2:
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
        repo_root=root,
        failures=failures,
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
            repo_root=root,
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
            repo_root=root,
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

    evaluation = _mapping(payload.get("evaluation"))
    suite_metrics = _mapping(evaluation.get("suite_metrics"))
    configuration = _mapping(suite_metrics.get("configuration"))
    coverage = _mapping(configuration.get("suite_coverage_validation"))
    if coverage.get("passed") is not True:
        failures.append("frozen suite coverage validation did not pass")
    content_hashes = _mapping(configuration.get("suite_content_hashes"))
    if not content_hashes or _artifact_content_sha256(dict(content_hashes)) != configuration.get(
        "suite_content_sha256"
    ):
        failures.append("suite content manifest identity is missing or inconsistent")
    environment = _mapping(configuration.get("release_environment_validation"))
    if environment.get("passed") is not True:
        failures.append("release environment deployment attestation did not pass")

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
    terminal = resolved = escalated = invalid = loops = evaluator_errors = 0
    false_commit = false_rollback = false_finalization = corruption = 0
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
        family = str(episode.get("family") or "")
        root = str(episode.get("physical_root") or "")
        if not key or key in keys:
            failures.append(f"episode[{index}] has a missing or duplicate episode_key")
        keys.add(key)
        if not family or not root or root in physical_roots:
            failures.append(
                f"episode[{index}] has a missing or globally duplicate physical-root identity"
            )
        physical_roots.add(root)
        by_family.setdefault(family, []).append(episode)
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
            failures.append(f"resolved episode {key!r} lacks complete physical safety evidence")
        if not _strict_episode_audit_passed(episode):
            failures.append(f"episode {key!r} did not pass the strict release audit")
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
        false_commit += _nonnegative_integer(
            episode.get("false_commit_count", 0), field=f"episode[{index}].false_commit_count"
        )
        false_rollback += _nonnegative_integer(
            episode.get("false_rollback_count", 0), field=f"episode[{index}].false_rollback_count"
        )
        false_finalization += _nonnegative_integer(
            episode.get("false_finalization_count", 0),
            field=f"episode[{index}].false_finalization_count",
        )
        episode_invalid = _nonnegative_integer(
            episode.get("invalid_action_count", 0), field=f"episode[{index}].invalid_action_count"
        )
        invalid += episode_invalid
        maximum_episode_invalid = max(maximum_episode_invalid, episode_invalid)
        loops += int(episode.get("loop_detected") is True)
        evaluator_errors += int(episode.get("evaluator_error") is not None)
        steps = _nonnegative_integer(episode.get("steps", 0), field=f"episode[{index}].steps")
        max_steps_seen = max(max_steps_seen, steps)
        if (
            episode.get("healthy_preservation_known") is True
            and episode.get("healthy_components_preserved") is False
        ):
            corruption += 1

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
            failures.append(f"{name} failed: observed {observed_value} > allowed {allowed}")
    if terminal_rate < float(hard["minimum_terminal_rate"]):
        failures.append(
            "minimum_terminal_rate failed: "
            f"observed {terminal_rate:.6f} < required {float(hard['minimum_terminal_rate']):.6f}"
        )
    invalid_rate = invalid / total if total else 0.0
    loop_rate = loops / total if total else 0.0
    if maximum_episode_invalid > int(hard["maximum_invalid_actions_per_episode"]):
        failures.append(
            "maximum_invalid_actions_per_episode failed: "
            f"observed {maximum_episode_invalid} > allowed "
            f"{int(hard['maximum_invalid_actions_per_episode'])}"
        )
    if invalid_rate > float(hard["maximum_invalid_action_rate"]):
        failures.append(
            "maximum_invalid_action_rate failed: "
            f"observed {invalid_rate:.6f} > allowed "
            f"{float(hard['maximum_invalid_action_rate']):.6f}"
        )
    if loop_rate > float(hard["maximum_loop_episode_rate"]):
        failures.append(
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
            failures.append(
                f"family {family!r} resolution failed: "
                f"{resolution_rate:.6f} < {minimum_resolution:.6f}"
            )
        if escalation_rate > maximum_escalation:
            failures.append(
                f"family {family!r} escalation failed: "
                f"{escalation_rate:.6f} > {maximum_escalation:.6f}"
            )

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
        "families": family_observed,
    }
    return EvaluationGateResult(
        passed=not failures,
        failures=tuple(dict.fromkeys(failures)),
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
    suite = parser.add_mutually_exclusive_group(required=True)
    suite.add_argument("--expected-suite", type=Path)
    suite.add_argument("--expected-suite-sha256")
    parser.add_argument(
        "--expected-protocol", choices=("canonical", "controller"), default="canonical"
    )
    parser.add_argument("--expected-registry-sha256")
    identity = parser.add_mutually_exclusive_group(required=True)
    identity.add_argument("--expected-policy-identity")
    identity.add_argument("--expected-model-id")
    parser.add_argument("--expected-model-revision")
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
    try:
        expected_suite_hash = (
            file_sha256(args.expected_suite)
            if args.expected_suite is not None
            else str(args.expected_suite_sha256)
        )
        result = validate_evaluation_artifact(
            args.artifact,
            policy=args.policy,
            expected_source_commit=_resolve_commit(
                args.expected_source_commit,
                repo_root=Path(__file__).resolve().parents[2],
            ),
            expected_suite_sha256=expected_suite_hash,
            expected_protocol=args.expected_protocol,
            expected_registry_sha256=args.expected_registry_sha256,
            expected_policy_identity=args.expected_policy_identity,
            expected_model_id=args.expected_model_id,
            expected_model_revision=args.expected_model_revision,
            required_gate_policy_id=args.required_gate_policy_id,
        )
        report = {**result.as_dict(), "validation_role": args.role}
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
