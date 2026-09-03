from __future__ import annotations

import copy
import math
import re
import types
from collections.abc import Callable, Iterable
from typing import Any, Mapping

from .actions import (
    ANOMALY_FAMILY_MARKERS,
    ASK_FOR_MORE_EVIDENCE,
    DIAGNOSTIC_TOOLS,
    COMMIT_STATE,
    CONTEXT_TOOLS,
    CORRECT_MEASUREMENTS,
    CORRECT_PARAMETERS,
    CORRECT_TOPOLOGY,
    CORRECTION_TOOLS,
    ESTIMATE_HIF_FROM_PATH,
    ESTIMATE_HIF_MULTISCAN_FROM_PATH,
    FINALIZE_DIAGNOSIS,
    GET_HARMONIC_CONTEXT,
    GET_MEASUREMENT_CONTEXT,
    GET_PARAMETER_CONTEXT,
    GET_TOPOLOGY_CONTEXT,
    HIF_DIAGNOSTICS_EXHAUSTED_REQUEST,
    INVALID_ACTION,
    POST_CORRECTION_CONFIRMATION_SIGNATURE,
    RECOVERY_BUDGET_EXHAUSTED_REQUEST,
    RECOVERY_OPTIONS_EXHAUSTED_REQUEST,
    ROLLBACK_STATE,
    RUN_ALTERNATIVE_TEST,
    RUN_HSE_FROM_PATH,
    RUN_THREE_PHASE_NLM_FROM_PATH,
    RUN_WLS,
    VERIFY_CANDIDATE,
    action_signature,
    safe_normalize_action,
    terminal_explanation_signatures,
    unexplained_signatures,
)
from .oracle.candidate_quality import CandidateAssessment, CandidateDisposition, CandidateQualityOracle
from .oracle.expert_types import (
    matching_evidence_codes,
    recovery_record_applies_to_state,
)
from .oracle.measurement_recovery_evidence import (
    accepted_measurement_indices,
    eligible_joint_measurement_targets,
    measurement_target_indices,
    verified_terminal_measurement_closure_action,
)
from .oracle.process_validity import ProcessValidityOracle
from .private_target_matching import correction_family
from .state_store import (
    CandidateLifecycle,
    FORBIDDEN_POLICY_KEYS,
    SYNTHETIC_TERMINAL_COMPATIBILITY_KEY,
    OracleState,
    PolicyObservation,
    PowerSystemStateStore,
    StateStoreError,
    policy_safe_copy,
)


def _mutable_collaborator_reference(value: Any, seen: set[int] | None = None) -> bool:
    """Whether sharing ``value`` could leak branch-side mutations to root."""
    if value is None or isinstance(value, (bool, int, float, complex, str, bytes, range)):
        return False
    seen = set() if seen is None else seen
    if isinstance(value, tuple):
        return any(_mutable_collaborator_reference(item, seen) for item in value)
    if isinstance(value, frozenset):
        return any(_mutable_collaborator_reference(item, seen) for item in value)
    if isinstance(value, (types.ModuleType, type, types.BuiltinFunctionType)):
        return False
    if isinstance(value, types.FunctionType):
        return _function_captures_mutable_state(value, seen)
    return True


def _function_captures_mutable_state(
    function: types.FunctionType, seen: set[int] | None = None
) -> bool:
    seen = set() if seen is None else seen
    if id(function) in seen:
        return False
    seen.add(id(function))
    defaults = tuple(function.__defaults__ or ())
    keyword_defaults = tuple((function.__kwdefaults__ or {}).values())
    if any(_mutable_collaborator_reference(value, seen) for value in (*defaults, *keyword_defaults)):
        return True
    for cell in function.__closure__ or ():
        try:
            value = cell.cell_contents
        except ValueError:
            continue
        if _mutable_collaborator_reference(value, seen):
            return True
    for name in function.__code__.co_names:
        if name not in function.__globals__:
            continue
        if _mutable_collaborator_reference(function.__globals__[name], seen):
            return True
    return False


def _clone_collaborator(value: Any, label: str) -> Any:
    """Deep-copy a branch hook, sharing only functions proven stateless."""
    if value is None:
        return None
    if isinstance(value, types.FunctionType):
        if _function_captures_mutable_state(value):
            raise StateStoreError(
                f"Cannot isolate branch collaborator {label!r}; use a deepcopyable callable object."
            )
        return value
    try:
        cloned = copy.deepcopy(value)
    except Exception as exc:
        raise StateStoreError(
            f"Cannot isolate branch collaborator {label!r}: {type(exc).__name__}."
        ) from exc
    if cloned is value and _mutable_collaborator_reference(value):
        raise StateStoreError(
            f"Branch collaborator {label!r} was not copied; provide an isolated clone."
        )
    return cloned


_PRODUCTION_DECISION_EVIDENCE_KEYS = frozenset(
    {
        "wls_objective",
        "objective",
        "chi_square_statistic",
        "residual_norm",
        "max_normalized_residual",
        "normalized_residuals",
        "target_fixed",
        "target_progress",
        "target_metric_kind",
        "target_metric_value",
        "target_metric_threshold",
        "parent_target_metric_value",
        "global_progress",
        "sequential_cross_family_measurement",
        "measurement_evidence_dominant",
        "measurement_target_branch_colocated",
        "independent_measurement_target",
        "remaining_anomaly_score",
        "remaining_suspect_count",
        # Transitional external-provider compatibility. This legacy field is
        # interpreted only as observable suspect evidence in deployment mode.
        "remaining_fault_count",
        "globally_resolved",
        "post_action_resolved",
        "physical_constraints_ok",
        "power_flow_converged",
        "topology_feasible",
        "new_constraint_violations",
        "healthy_component_modified",
        "collateral_damage",
        "converged",
    }
)

_SEMANTIC_POLICY_FIELDS = (
    "unresolved_signatures",
    "remaining_anomaly_score",
    "no_material_anomaly_remaining",
)


def _provider_kind(provider: Any) -> str:
    for owner in (provider, getattr(provider, "__self__", None)):
        if owner is None:
            continue
        value = getattr(owner, "provider_kind", None)
        if value is not None:
            return str(value).strip().lower()
        if bool(getattr(owner, "is_deterministic_mock", False)):
            return "deterministic_mock"
    return "configured"


def _provider_production_approved(provider: Any) -> bool:
    return any(
        bool(getattr(owner, "production_dataset_approved", False))
        for owner in (provider, getattr(provider, "__self__", None))
        if owner is not None
    )


def _provider_label(provider: Any) -> str:
    owner = getattr(provider, "__self__", None)
    target = owner if owner is not None else provider
    name = getattr(target, "__qualname__", None) or getattr(target, "__name__", None)
    if name is None:
        name = type(target).__name__
    module = getattr(target, "__module__", None) or getattr(type(target), "__module__", "")
    return f"{module}.{name}".strip(".")


def _observable_provenance_source(source: Any) -> bool:
    normalized = str(source or "").strip().lower()
    if any(token in normalized for token in ("hidden", "oracle", "truth", "synthetic")):
        return False
    return normalized.startswith(
        (
            "observable",
            "deployment",
            "sensor",
            "wls",
            "context_provider",
            "configured_provider",
            "controller_default",
            "operator",
            "real",
            "production",
        )
    )


_BRANCH_ROUTE_STATUSES = frozenset(
    {"actionable", "complete_negative", "unavailable_or_inconclusive"}
)


def _branch_route_contract_valid(
    route_status: Any,
    supported: list[dict[str, Any]],
) -> bool:
    if not isinstance(route_status, str) or route_status not in _BRANCH_ROUTE_STATUSES:
        return False
    return bool(supported) == (route_status == "actionable")


def _terminal_closure_branch_screening_valid(
    screening: Any,
    *,
    state_id: str,
    state_hash: str,
) -> bool:
    if not isinstance(screening, Mapping) or set(screening) != {
        "parameter",
        "topology",
    }:
        return False
    contracts = {
        "parameter": GET_PARAMETER_CONTEXT,
        "topology": GET_TOPOLOGY_CONTEXT,
    }
    for family, context_tool in contracts.items():
        evidence = screening.get(family)
        if not isinstance(evidence, Mapping):
            return False
        inventory = evidence.get("supported_corrections")
        if (
            evidence.get("context_tool") != context_tool
            or str(evidence.get("state_id") or "") != state_id
            or str(evidence.get("state_hash") or "") != state_hash
            or not _observable_provenance_source(evidence.get("evidence_source"))
            or not isinstance(inventory, (list, tuple))
            or not _branch_route_contract_valid(
                evidence.get("route_status"), list(inventory)
            )
            or evidence.get("route_status") != "complete_negative"
        ):
            return False
    return True


def _semantic_correction_signature(action: Mapping[str, Any] | str) -> str | None:
    """Canonical bounded correction identity independent of active state ID."""
    normalized = safe_normalize_action(action)
    tool = normalized["tool"]
    if tool not in CORRECTION_TOOLS:
        return None
    arguments = dict(normalized["arguments"])
    arguments.pop("state_id", None)
    arguments.pop("candidate_state_id", None)
    if any(key in arguments for key in ("case", "case_updates", "measurements")):
        return None

    if tool == CORRECT_MEASUREMENTS:
        group = arguments.get("suspect_group")
        updates = arguments.get("measurement_updates")
        group_indices: set[int] | None = None
        update_indices: set[int] | None = None
        if group is not None:
            if not isinstance(group, (list, tuple)) or not group:
                return None
            group_indices = set()
            for index in group:
                if (
                    not isinstance(index, int)
                    or isinstance(index, bool)
                    or index < 0
                ):
                    return None
                group_indices.add(index)
            if len(group_indices) != len(group):
                return None
        if updates is not None:
            if not isinstance(updates, Mapping) or not updates:
                return None
            update_indices = set()
            for index in updates:
                if (
                    not isinstance(index, int)
                    or isinstance(index, bool)
                    or index < 0
                ):
                    return None
                update_indices.add(index)
        if group_indices is None and update_indices is None:
            return None
        if (
            group_indices is not None
            and update_indices is not None
            and group_indices != update_indices
        ):
            return None
        if group_indices is not None:
            arguments["suspect_group"] = sorted(group_indices)
    else:
        branch_keys = [
            key
            for key in ("branch_row0", "line_index1", "line_index", "branch_id", "cb_name")
            if arguments.get(key) is not None
        ]
        if len(branch_keys) != 1:
            return None
        branch_key = branch_keys[0]
        branch_value = arguments[branch_key]
        if branch_key in {"branch_row0", "line_index1", "line_index"}:
            if not isinstance(branch_value, int) or isinstance(branch_value, bool):
                return None
            row0 = branch_value if branch_key == "branch_row0" else branch_value - 1
            if row0 < 0:
                return None
            canonical_target = ("branch_row0", row0)
        else:
            if not isinstance(branch_value, str) or not branch_value.strip():
                return None
            canonical_target = (branch_key, branch_value.strip())
        for key in ("branch_row0", "line_index1", "line_index", "branch_id", "cb_name"):
            arguments.pop(key, None)
        arguments[canonical_target[0]] = canonical_target[1]

        if tool == CORRECT_TOPOLOGY:
            statuses = [
                arguments[key]
                for key in ("status", "expected_status")
                if arguments.get(key) is not None
            ]
            if not statuses:
                return None
            canonical_statuses: set[int] = set()
            for status in statuses:
                if isinstance(status, bool):
                    canonical_statuses.add(int(status))
                elif isinstance(status, int) and status in {0, 1}:
                    canonical_statuses.add(status)
                else:
                    return None
            if len(canonical_statuses) != 1:
                return None
            arguments.pop("expected_status", None)
            arguments["status"] = canonical_statuses.pop()
    try:
        return action_signature({"tool": tool, "arguments": arguments})
    except (TypeError, ValueError):
        return None


class TransactionalPSSEEnv:
    """Recovery-aware transactional controller around PSSE macro-actions."""

    def __init__(
        self,
        *,
        store: PowerSystemStateStore | None = None,
        wls_runner: Callable[[dict[str, Any]], Mapping[str, Any]] | None = None,
        process_oracle: ProcessValidityOracle | None = None,
        candidate_quality_oracle: CandidateQualityOracle | None = None,
        context_providers: Mapping[str, Callable[[dict[str, Any]], Mapping[str, Any]]] | None = None,
        evidence_providers: Mapping[str, Callable[[dict[str, Any]], Mapping[str, Any]]] | None = None,
        correction_executors: Mapping[
            str, Callable[[dict[str, Any], dict[str, Any]], Mapping[str, Any]]
        ]
        | None = None,
        production_dataset_mode: bool = False,
        approved_deterministic_providers: Iterable[str] | None = None,
        max_steps: int = 24,
        history_window: int = 4,
    ) -> None:
        self.store = store or PowerSystemStateStore()
        self.wls_runner = wls_runner
        self.process_oracle = process_oracle or ProcessValidityOracle()
        self.candidate_quality_oracle = candidate_quality_oracle or CandidateQualityOracle()
        self.context_providers = dict(context_providers or {})
        self.evidence_providers = dict(evidence_providers or {})
        self.correction_executors = dict(correction_executors or {})
        self.production_dataset_mode = bool(production_dataset_mode)
        self.approved_deterministic_providers = {
            str(name) for name in (approved_deterministic_providers or ())
        }
        self.max_steps = int(max_steps)
        self.history_window = int(history_window)
        self.current_candidate_id: str | None = None
        self.context_flags: dict[str, Any] = {}
        self.history: list[dict[str, Any]] = []
        self.terminal = False
        self.terminal_outcome: str | None = None
        self._episode_counter = 0
        self._oracle_payload: dict[str, Any] = {}
        self._audited_evaluation_setup_correction = False
        if self.production_dataset_mode:
            self.validate_production_configuration()

    def validate_production_configuration(self) -> dict[str, Any]:
        """Fail closed unless every production data dependency is configured.

        Deployment adapters must declare ``provider_kind`` as ``real``,
        ``deployment``, or ``production``.  Providers explicitly marked
        ``provider_kind='mock'``/``'deterministic'`` (or
        ``is_deterministic_mock=True``) must instead be named in
        ``approved_deterministic_providers`` or carry
        ``production_dataset_approved=True``.  This permits reviewed,
        deterministic pilot adapters without silently accepting placeholders.
        """
        required: dict[str, Any] = {
            RUN_WLS: self.wls_runner,
            **{tool: self.context_providers.get(tool) for tool in sorted(CONTEXT_TOOLS)},
            **{tool: self.correction_executors.get(tool) for tool in sorted(CORRECTION_TOOLS)},
        }
        missing = sorted(name for name, provider in required.items() if not callable(provider))
        if missing:
            raise ValueError(
                "production_dataset_mode requires configured providers: "
                + ", ".join(missing)
            )

        unapproved: list[str] = []
        unclassified: list[str] = []
        provider_status: dict[str, dict[str, Any]] = {}
        for name, provider in required.items():
            kind = _provider_kind(provider)
            deterministic = kind in {
                "mock",
                "synthetic",
                "deterministic",
                "deterministic_mock",
                "test",
            }
            approved = (
                name in self.approved_deterministic_providers
                or _provider_production_approved(provider)
            )
            if deterministic and not approved:
                unapproved.append(name)
            elif not deterministic and kind not in {"real", "deployment", "production"}:
                unclassified.append(name)
            provider_status[name] = {
                "provider": _provider_label(provider),
                "provider_kind": kind,
                "approved_deterministic": bool(approved),
            }
        if unapproved:
            raise ValueError(
                "production_dataset_mode has deterministic/mock providers without explicit approval: "
                + ", ".join(sorted(unapproved))
            )
        if unclassified:
            raise ValueError(
                "production_dataset_mode providers must declare provider_kind as real/deployment/production "
                "or be explicitly approved deterministic providers: "
                + ", ".join(sorted(unclassified))
            )
        return {"production_dataset_mode": True, "providers": provider_status}

    def reset(self, scenario: Mapping[str, Any]) -> dict[str, Any]:
        scenario_id = str(scenario.get("scenario_id", scenario.get("id", "scenario")))
        base_episode_id = scenario.get("episode_id", scenario_id)
        episode_id = f"{base_episode_id}_episode{self._episode_counter}"
        self._episode_counter += 1

        raw_metadata = dict(scenario.get("metadata") or {})
        raw_semantic_provenance = scenario.get(
            "semantic_field_provenance",
            raw_metadata.get("semantic_field_provenance", {}),
        )
        if raw_semantic_provenance is None:
            raw_semantic_provenance = {}
        if not isinstance(raw_semantic_provenance, Mapping):
            raise ValueError("semantic_field_provenance must be a mapping.")
        hidden_input = scenario.get("hidden_truth")
        if self.production_dataset_mode and isinstance(hidden_input, Mapping):
            leaked = sorted(set(hidden_input) & set(_SEMANTIC_POLICY_FIELDS))
            if leaked:
                raise ValueError(
                    "Production policy semantic fields cannot be initialized from hidden_truth: "
                    + ", ".join(leaked)
                )

        semantic_provenance: dict[str, str] = {}
        for field in _SEMANTIC_POLICY_FIELDS:
            supplied = field in scenario or field in raw_metadata
            source = raw_semantic_provenance.get(field)
            if self.production_dataset_mode and supplied:
                if source is None:
                    raise ValueError(
                        f"Production policy field {field!r} requires semantic_field_provenance."
                    )
                if not _observable_provenance_source(source):
                    raise ValueError(
                        f"Production policy field {field!r} has non-observable provenance: {source!r}."
                    )
            semantic_provenance[field] = str(
                source
                if source is not None
                else ("synthetic_scenario_input" if supplied else "controller_default")
            )
        oracle_payload = dict(scenario.get("hidden_truth") or {})
        truth_keys = {
            "clean_case",
            "clean_measurements",
            "clean_parameter_values",
            "true_measurement_errors",
            "true_parameter_errors",
            "true_topology_errors",
            "remaining_true_faults",
            "remaining_true_fault_count",
            "remaining_fault_count",
        }
        truth_was_supplied = bool(scenario.get("hidden_truth"))
        for key in truth_keys:
            if key in scenario:
                oracle_payload[key] = copy.deepcopy(scenario[key])
                truth_was_supplied = True
            elif key in raw_metadata:
                oracle_payload[key] = copy.deepcopy(raw_metadata[key])
                truth_was_supplied = True
        private_release_audit = scenario.get("release_audit")
        if private_release_audit is not None:
            if not isinstance(private_release_audit, Mapping):
                raise ValueError("release_audit must be a private mapping.")
            oracle_payload["release_audit"] = copy.deepcopy(
                dict(private_release_audit)
            )
        if (
            "remaining_fault_count" in oracle_payload
            and "remaining_true_fault_count" not in oracle_payload
        ):
            oracle_payload["remaining_true_fault_count"] = oracle_payload.pop("remaining_fault_count")
        oracle_payload["truth_complete"] = bool(oracle_payload.get("truth_complete", truth_was_supplied))
        if oracle_payload["truth_complete"]:
            family_keys = (
                "true_measurement_errors",
                "true_parameter_errors",
                "true_topology_errors",
            )
            all_faults = [
                copy.deepcopy(fault)
                for key in family_keys
                for fault in list(oracle_payload.get(key) or [])
            ]
            explicit_remaining = "remaining_true_faults" in oracle_payload
            if explicit_remaining:
                remaining_faults = copy.deepcopy(list(oracle_payload.get("remaining_true_faults") or []))
                unmatched = [fault for fault in remaining_faults if fault not in all_faults]
                if all_faults and unmatched:
                    raise ValueError(
                        "remaining_true_faults must be a subset of the supplied true_*_errors."
                    )
                if all_faults:
                    for key in family_keys:
                        oracle_payload[key] = [
                            copy.deepcopy(fault)
                            for fault in list(oracle_payload.get(key) or [])
                            if fault in remaining_faults
                        ]
                    remaining_faults = [
                        copy.deepcopy(fault)
                        for key in family_keys
                        for fault in list(oracle_payload.get(key) or [])
                    ]
            else:
                remaining_faults = all_faults
                supplied_count = oracle_payload.get("remaining_true_fault_count")
                if supplied_count is not None and all_faults and int(supplied_count) != len(all_faults):
                    raise ValueError(
                        "A subset remaining_true_fault_count requires explicit remaining_true_faults."
                    )
            oracle_payload["remaining_true_faults"] = remaining_faults
            if explicit_remaining or all_faults or oracle_payload.get("remaining_true_fault_count") is None:
                oracle_payload["remaining_true_fault_count"] = len(remaining_faults)
        hints = scenario.get("oracle_action_hints")
        if hints is None:
            hints = scenario.get("suggested_actions", raw_metadata.get("suggested_actions", []))
        oracle_payload["oracle_action_hints"] = copy.deepcopy(list(hints or []))
        self._oracle_payload = oracle_payload

        metadata = {key: copy.deepcopy(value) for key, value in raw_metadata.items() if key not in FORBIDDEN_POLICY_KEYS}
        metadata.setdefault("scenario_id", scenario_id)
        root_id = self.store.create_root(
            case=scenario.get("case", scenario.get("case_path")),
            measurements=scenario.get("measurements", scenario.get("z_obs", [])),
            metadata=metadata,
            episode_id=episode_id,
            created_at_step=0,
        )
        self.current_candidate_id = None
        self.context_flags = {
            "remaining_anomaly_score": raw_metadata.get(
                "remaining_anomaly_score", scenario.get("remaining_anomaly_score")
            ),
            "no_material_anomaly_remaining": bool(
                raw_metadata.get("no_material_anomaly_remaining", scenario.get("no_material_anomaly_remaining", False))
            ),
            # Production correction labels must be backed by observable
            # measurement context.  Make this a controller-wide invariant,
            # not a scenario-supplied bit that could encode the hidden family.
            "requires_measurement_context": True
            if self.production_dataset_mode
            else bool(
                raw_metadata.get(
                    "requires_measurement_context",
                    scenario.get("requires_measurement_context", False),
                )
            ),
            "unresolved_signatures": list(
                raw_metadata.get("unresolved_signatures", scenario.get("unresolved_signatures", [])) or []
            ),
            "accepted_corrections": [],
            "rejected_hypotheses": [],
            "tried_action_signatures": [],
            "semantic_field_provenance": semantic_provenance,
        }
        self.history = []
        self.terminal = False
        self.terminal_outcome = None
        return self.current_state()

    def current_state(self) -> dict[str, Any]:
        return self.store.decision_summary(
            candidate_state_id=self.current_candidate_id,
            remaining_budget=max(self.max_steps - len(self.history), 0),
            context_flags=self.context_flags,
        )

    _EVIDENCE_CHANNEL_KEYS = (
        "harmonic_measurements",
        "parameter_scans",
        "hif_scan_window",
        "nlm_diagnostic",
        "hif_runtime",
        "three_phase_voltages",
        "three_phase_branch_currents",
    )

    def _observable_evidence_channels(self) -> list[str]:
        """Telemetry channels present on the active state's metadata.

        Which data streams exist (harmonic scans, HIF scan windows, repeated
        parameter scans) is deployment-observable operator knowledge; the
        channel *contents* stay out of the policy observation.
        """
        try:
            payload = self.store.get_state(str(self.store.active_state_id))
        except Exception:
            return []
        metadata = payload.get("metadata")
        if not isinstance(metadata, Mapping):
            return []
        return [key for key in self._EVIDENCE_CHANNEL_KEYS if metadata.get(key)]

    def get_policy_observation(
        self,
        history: list[Mapping[str, Any]] | None = None,
        *,
        history_window: int | None = None,
    ) -> PolicyObservation:
        summary = self.current_state()
        window_size = self.history_window if history_window is None else int(history_window)
        history_source = self.history if history is None else history
        window = policy_safe_copy(list(history_source)[-window_size:])
        has_unverified_candidate = bool(summary.get("has_unverified_candidate"))
        has_verified_candidate = bool(summary.get("has_verified_candidate"))
        if has_unverified_candidate:
            policy_candidate_status = "unverified"
            policy_candidate_lifecycle = CandidateLifecycle.OPEN_UNVERIFIED_CANDIDATE.value
        elif has_verified_candidate:
            policy_candidate_status = "verified"
            policy_candidate_lifecycle = CandidateLifecycle.VERIFIED_CANDIDATE.value
        else:
            policy_candidate_status = None
            policy_candidate_lifecycle = CandidateLifecycle.NO_CANDIDATE.value
        return PolicyObservation(
            active_state_id=str(summary["active_state_id"]),
            candidate_state_id=summary.get("candidate_state_id"),
            candidate_status=policy_candidate_status,
            last_tool=summary.get("last_tool"),
            last_tool_status=summary.get("last_tool_status"),
            last_tool_output=policy_safe_copy(summary.get("last_tool_output") or {}),
            last_verification=policy_safe_copy(summary.get("last_verification") or {}),
            accepted_corrections=policy_safe_copy(summary.get("accepted_corrections") or []),
            explained_anomalies=policy_safe_copy(summary.get("explained_anomalies") or []),
            rejected_hypotheses=policy_safe_copy(summary.get("rejected_hypotheses") or []),
            unresolved_signatures=list(summary.get("unresolved_signatures") or []),
            tried_action_signatures=list(summary.get("tried_action_signatures") or []),
            remaining_budget=int(summary.get("remaining_budget") or 0),
            history_window=window,
            episode_id=summary.get("episode_id"),
            candidate_parent_id=summary.get("candidate_parent_id"),
            candidate_lifecycle=policy_candidate_lifecycle,
            candidate_committed=bool(summary.get("candidate_committed")),
            has_open_candidate=bool(summary.get("has_open_candidate")),
            has_unverified_candidate=has_unverified_candidate,
            has_verified_candidate=has_verified_candidate,
            remaining_anomaly_score=summary.get("remaining_anomaly_score"),
            no_material_anomaly_remaining=bool(summary.get("no_material_anomaly_remaining")),
            has_fresh_measurement_context=bool(summary.get("has_fresh_measurement_context")),
            has_fresh_parameter_context=bool(summary.get("has_fresh_parameter_context")),
            has_fresh_topology_context=bool(summary.get("has_fresh_topology_context")),
            measurement_context_state_id=summary.get("measurement_context_state_id"),
            parameter_context_state_id=summary.get("parameter_context_state_id"),
            topology_context_state_id=summary.get("topology_context_state_id"),
            fresh_context_evidence=policy_safe_copy(
                dict(summary.get("fresh_context_evidence") or {})
            ),
            requires_measurement_context=bool(summary.get("requires_measurement_context")),
            available_evidence=self._observable_evidence_channels(),
            semantic_field_provenance=policy_safe_copy(
                dict(summary.get("semantic_field_provenance") or {})
            ),
        )

    def get_oracle_state(self, history: list[Mapping[str, Any]] | None = None) -> OracleState:
        payload = self._oracle_payload
        summary = self.current_state()
        candidate_payload: Mapping[str, Any] = {}
        candidate_id = summary.get("candidate_state_id")
        if candidate_id and self.store.exists(str(candidate_id)):
            candidate_payload = self.store.get_state(str(candidate_id))
        return OracleState(
            policy_observation=self.get_policy_observation(history),
            clean_case=copy.deepcopy(payload.get("clean_case")),
            clean_measurements=copy.deepcopy(payload.get("clean_measurements")),
            true_measurement_errors=copy.deepcopy(list(payload.get("true_measurement_errors") or [])),
            true_parameter_errors=copy.deepcopy(list(payload.get("true_parameter_errors") or [])),
            true_topology_errors=copy.deepcopy(list(payload.get("true_topology_errors") or [])),
            remaining_true_faults=copy.deepcopy(list(payload.get("remaining_true_faults") or [])),
            oracle_action_hints=copy.deepcopy(list(payload.get("oracle_action_hints") or [])),
            hidden_truth={
                key: copy.deepcopy(value)
                for key, value in payload.items()
                if key != "oracle_action_hints"
            },
            candidate_disposition=summary.get("candidate_disposition"),
            candidate_lifecycle=str(summary.get("candidate_lifecycle")),
            candidate_assessment=copy.deepcopy(candidate_payload.get("candidate_assessment") or {}),
        )

    def encode_observation(
        self,
        state: Mapping[str, Any] | None = None,
        history: list[Mapping[str, Any]] | None = None,
        *,
        history_window: int = 4,
    ) -> dict[str, Any]:
        """Compatibility wrapper; new code should call get_policy_observation()."""
        observation = self.get_policy_observation(history, history_window=history_window).as_dict()
        return {"state_summary": observation, "history_window": observation["history_window"]}

    def apply_audited_evaluation_setup_correction(
        self,
        action: Mapping[str, Any] | str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Apply one private frozen-suite partial-state setup correction.

        Historical partial-success interventions contain the exact physical
        measurement update needed to construct a committed partial repair.
        That audit-only update intentionally is not a model-visible supported
        target.  Keep the exception narrower than normal execution: only a
        full measurement update is accepted, nesting is forbidden, and all
        ordinary lifecycle, state-reference, context-freshness, executor, and
        mutation checks still run through :meth:`step`.
        """

        normalized = safe_normalize_action(action)
        arguments = normalized["arguments"]
        updates = arguments.get("measurement_updates")
        if (
            normalized["tool"] != CORRECT_MEASUREMENTS
            or not isinstance(updates, Mapping)
            or not updates
            or arguments.get("suspect_group") is not None
        ):
            raise ValueError(
                "audited evaluation setup accepts only a full measurement update"
            )
        if self._audited_evaluation_setup_correction:
            raise RuntimeError("nested audited evaluation setup is forbidden")
        self._audited_evaluation_setup_correction = True
        try:
            return self.step(normalized)
        finally:
            self._audited_evaluation_setup_correction = False

    def step(self, action: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        normalized = safe_normalize_action(action)
        before_hash = self.store.episode_hash()
        before_active_id = self.store.active_state_id
        before_candidate_id = self.current_candidate_id

        if self.terminal:
            output = self.record_noop_failure(
                action=normalized,
                error_code="episode_already_terminal",
                error_detail="episode_already_terminal",
                valid_next_actions=[],
            )
            self._record_transition(
                normalized,
                output,
                source_state_id=before_active_id,
                source_candidate_id=before_candidate_id,
            )
            return self.current_state(), output

        validity_state = self.current_state()
        # Production corrections must be chosen from the exact same-state
        # provider inventory even when a caller supplies a custom process
        # oracle rather than the release factory's hydrated configuration.
        validity_state["require_context_supported_corrections"] = bool(
            self.production_dataset_mode
        )
        validity_state["audited_evaluation_setup_correction"] = bool(
            self._audited_evaluation_setup_correction
        )
        # Synthetic pilot mode may use a private terminal fixture.  Deployment
        # training must never consume that oracle-only bit as a legality
        # bypass; its finalization labels require public evidence below.
        if not self.production_dataset_mode:
            validity_state["oracle_terminal_eligible"] = bool(
                self._oracle_payload.get("oracle_terminal_eligible", False)
            )
            validity_state[SYNTHETIC_TERMINAL_COMPATIBILITY_KEY] = True
        validity = self.process_oracle.check(validity_state, normalized, store=self.store)
        if not validity["process_valid"]:
            output = self.record_noop_failure(
                action=normalized,
                error_code=str(validity.get("error_code") or "invalid_action"),
                error_detail=validity.get("error_detail"),
                valid_next_actions=validity.get("valid_next_actions") or [],
            )
        else:
            try:
                output = self.dispatch_valid_action(normalized)
            except Exception as exc:  # tool/provider failures are collectable learner states
                safe_detail = type(exc).__name__
                output = self.record_noop_failure(
                    action=normalized,
                    error_code="dispatch_error",
                    error_detail=safe_detail,
                    valid_next_actions=self.process_oracle.repair_actions(
                        self.current_state(), "state_reference_mismatch", safe_detail
                    ),
                )

        if output["execution_status"] == "failure":
            # A failed action must be a no-op on the store, with one narrow
            # exception: a candidate whose verification solve itself fails is
            # recorded as verified-REJECT (the verdict is store metadata; the
            # active and candidate states are untouched), otherwise the open
            # candidate would have no legal rollback path.
            verified_reject_recorded = bool(
                normalized["tool"] in {RUN_WLS, VERIFY_CANDIDATE}
                and output.get("state_mutated")
                and before_candidate_id is not None
                and self.current_candidate_id == before_candidate_id
                and self.store.get_state(str(before_candidate_id)).get("candidate_disposition")
                == "REJECT"
            )
            if before_hash != self.store.episode_hash() and not verified_reject_recorded:
                raise RuntimeError("Failed action mutated the transactional state store.")
            if before_active_id != self.store.active_state_id or before_candidate_id != self.current_candidate_id:
                raise RuntimeError("Failed action changed active or candidate state.")
        self._record_transition(
            normalized,
            output,
            source_state_id=before_active_id,
            source_candidate_id=before_candidate_id,
        )
        return self.current_state(), output

    def record_noop_failure(
        self,
        *,
        action: Mapping[str, Any],
        error_code: str,
        error_detail: str | None,
        valid_next_actions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return self._standard_output(
            execution_status="failure",
            error_code=error_code,
            error_detail=error_detail,
            state_mutated=False,
            valid_next_actions=valid_next_actions,
        )

    def dispatch_valid_action(self, action: dict[str, Any]) -> dict[str, Any]:
        tool = action["tool"]
        args = action["arguments"]
        if tool in CORRECTION_TOOLS:
            return self._step_correction(action)
        if tool in {RUN_WLS, VERIFY_CANDIDATE}:
            return self._step_wls(action)
        if tool in CONTEXT_TOOLS:
            return self._step_context(action)
        if tool == COMMIT_STATE:
            candidate_id = str(args.get("candidate_state_id") or self.current_candidate_id)
            if self.production_dataset_mode:
                evidence = self.candidate_decision_evidence(candidate_id)
                if not evidence["sufficient"]:
                    return self.record_noop_failure(
                        action=action,
                        error_code="insufficient_observable_evidence",
                        error_detail=",".join(evidence["missing"]),
                        valid_next_actions=[],
                    )
            candidate_payload = self.store.get_state(candidate_id)
            next_oracle_payload = self._truth_after_commit(candidate_payload)
            accepted_record = self._accepted_candidate_record(candidate_payload)
            assessment = copy.deepcopy(candidate_payload.get("candidate_assessment") or {})
            accept_final = (
                candidate_payload.get("candidate_disposition")
                == CandidateDisposition.ACCEPT_FINAL.value
            )
            committed_id = self.store.commit(candidate_id)
            self.context_flags.setdefault("accepted_corrections", []).append(accepted_record)
            self._oracle_payload = next_oracle_payload
            self._persist_observable_semantics(
                candidate_payload.get("verification_output") or {},
                source=str(
                    (candidate_payload.get("verification_output") or {}).get("evidence_source")
                    or "controller_default_after_commit"
                ),
                replace_missing=True,
            )
            self._invalidate_context_flags()
            self.current_candidate_id = None
            if self.production_dataset_mode:
                # ``ACCEPT_FINAL`` is a candidate-quality disposition, not a
                # privileged certificate for the whole episode.  Do not mint
                # or retain the synthetic process bypass in deployment mode.
                self._oracle_payload.pop("oracle_terminal_eligible", None)
            elif accept_final:
                self._oracle_payload["oracle_terminal_eligible"] = True
            return self._standard_output(
                execution_status="success",
                state_mutated=True,
                active_state_id=committed_id,
                tool_metrics={"candidate_assessment": assessment},
            )
        if tool == ROLLBACK_STATE:
            candidate_id = str(args.get("candidate_state_id") or self.current_candidate_id)
            if self.production_dataset_mode:
                evidence = self.candidate_decision_evidence(candidate_id)
                if not evidence["sufficient"]:
                    return self.record_noop_failure(
                        action=action,
                        error_code="insufficient_observable_evidence",
                        error_detail=",".join(evidence["missing"]),
                        valid_next_actions=[],
                    )
            candidate_payload = self.store.get_state(candidate_id)
            parent_hash = self.store.state_hash(str(candidate_payload["parent_state_id"]))
            parent_id = self.store.rollback(candidate_id)
            if self.store.state_hash(parent_id) != parent_hash:
                raise StateStoreError("Rollback did not restore the exact parent hash.")
            self._remember_rejected_hypothesis(candidate_payload)
            self.current_candidate_id = None
            return self._standard_output(
                execution_status="success",
                state_mutated=True,
                active_state_id=parent_id,
                tool_metrics={"rolled_back_state_id": candidate_id, "restored_parent_hash": parent_hash},
            )
        if tool == FINALIZE_DIAGNOSIS:
            self.terminal = True
            self.terminal_outcome = "resolved"
            return self._standard_output(
                execution_status="success",
                state_mutated=False,
                tool_metrics={
                    "finalized": True,
                    "terminal_outcome": self.terminal_outcome,
                },
            )
        if tool in {ASK_FOR_MORE_EVIDENCE, RUN_ALTERNATIVE_TEST} or tool in DIAGNOSTIC_TOOLS:
            provider = self.evidence_providers.get(tool)
            target_id = str(args.get("state_id") or self.current_candidate_id or self.store.active_state_id)
            provider_state = self.store.get_state(target_id)
            provider_state["policy_observation"] = self.get_policy_observation().as_dict()
            provider_state["evidence_request"] = args.get("request")
            if provider is None and self.production_dataset_mode:
                return self.record_noop_failure(
                    action=action,
                    error_code="production_provider_missing",
                    error_detail=tool,
                    valid_next_actions=[],
                )
            if provider is None:
                metrics = {
                    "evidence_requested": tool,
                    "evidence_source": "synthetic_placeholder",
                }
            elif tool in DIAGNOSTIC_TOOLS:
                # Specialized diagnostics receive the full action so bounded
                # arguments (candidate branch, phase, grid options) reach the
                # underlying estimator.
                metrics = dict(provider(copy.deepcopy(provider_state), copy.deepcopy(action)))
            else:
                metrics = dict(provider(copy.deepcopy(provider_state)))
            status = str(metrics.pop("execution_status", "success"))
            if status != "success":
                return self._standard_output(
                    execution_status="failure",
                    error_code=str(metrics.pop("error_code", "evidence_provider_failure")),
                    error_detail=metrics.pop("error_detail", None),
                    state_mutated=False,
                    tool_metrics=metrics,
                )
            if provider is not None:
                metrics.setdefault(
                    "evidence_source", f"configured_provider:{_provider_label(provider)}"
                )
                metrics.setdefault("state_id", target_id)
                metrics.setdefault("state_hash", provider_state["state_hash"])
            evidence_bound = (
                str(metrics.get("state_id")) == target_id
                and str(metrics.get("state_hash")) == str(provider_state["state_hash"])
            )
            if self.production_dataset_mode and (
                not evidence_bound or not self._provider_metrics_are_substantive(metrics)
            ):
                return self.record_noop_failure(
                    action=action,
                    error_code="insufficient_observable_evidence",
                    error_detail=(
                        f"{tool}_provider_evidence_unbound"
                        if not evidence_bound
                        else f"{tool}_provider_returned_no_evidence"
                    ),
                    valid_next_actions=[],
                )
            if tool in DIAGNOSTIC_TOOLS:
                self._record_anomaly_explanation(tool, target_id, metrics)
            if (
                tool == ASK_FOR_MORE_EVIDENCE
                and args.get("request")
                in {
                    HIF_DIAGNOSTICS_EXHAUSTED_REQUEST,
                    RECOVERY_BUDGET_EXHAUSTED_REQUEST,
                    RECOVERY_OPTIONS_EXHAUSTED_REQUEST,
                }
            ):
                escalation_audit = self._operator_escalation_audit(
                    action, provider_metrics=metrics
                )
                if not escalation_audit["sufficient"]:
                    return self.record_noop_failure(
                        action=action,
                        error_code="operator_escalation_precondition_not_met",
                        error_detail=",".join(escalation_audit["missing"]),
                        valid_next_actions=[],
                    )
                self.terminal = True
                self.terminal_outcome = "operator_escalation"
                metrics["operator_escalation_audit"] = escalation_audit["ledger"]
                metrics["terminal_outcome"] = self.terminal_outcome
            return self._standard_output(execution_status="success", state_mutated=False, tool_metrics=metrics)
        return self.record_noop_failure(
            action=action,
            error_code="unknown_tool",
            error_detail=tool,
            valid_next_actions=[],
        )

    def is_terminal(self, state: Mapping[str, Any] | None = None) -> bool:
        return self.terminal

    def candidate_decision_evidence(self, candidate_state_id: str | None = None) -> dict[str, Any]:
        """Audit observable evidence supporting a production commit/rollback target."""
        candidate_id = str(candidate_state_id or self.current_candidate_id or "")
        missing: list[str] = []
        if not candidate_id or not self.store.exists(candidate_id):
            return {
                "sufficient": False,
                "candidate_state_id": candidate_id or None,
                "missing": ["candidate_state_missing"],
                "evidence_keys": [],
            }
        candidate = self.store.get_state(candidate_id)
        verification = candidate.get("verification_output")
        if not isinstance(verification, Mapping):
            missing.append("successful_verification_missing")
            verification = {}
        elif verification.get("execution_status", "success") != "success":
            missing.append("successful_verification_missing")
        if str(verification.get("state_id")) != candidate_id:
            missing.append("verification_state_id_unbound")
        if str(verification.get("state_hash")) != str(candidate.get("state_hash")):
            missing.append("verification_state_hash_unbound")
        evidence_keys = sorted(
            key
            for key in _PRODUCTION_DECISION_EVIDENCE_KEYS
            if key in verification and verification.get(key) is not None
        )
        if not evidence_keys:
            missing.append("decision_metrics_missing")
        source = str(verification.get("evidence_source") or "")
        if not source:
            missing.append("evidence_source_missing")
        elif any(token in source.lower() for token in ("placeholder", "fallback")):
            missing.append("placeholder_evidence_prohibited")
        elif not _observable_provenance_source(source):
            missing.append("non_observable_evidence_source")
        source_tool = safe_normalize_action(candidate.get("source_action") or {})[
            "tool"
        ]
        partial_global_progress_floor = (
            self.candidate_quality_oracle.min_branch_partial_global_progress
            if source_tool in {CORRECT_PARAMETERS, CORRECT_TOPOLOGY}
            else self.candidate_quality_oracle.min_partial_global_progress
        )
        missing.extend(
            self._target_decision_evidence_missing(
                verification,
                str(candidate.get("candidate_disposition") or ""),
                min_partial_global_progress=partial_global_progress_floor,
                min_topology_structural_global_progress=(
                    self.candidate_quality_oracle.min_topology_structural_global_progress
                ),
                max_branch_target_threshold_ratio=(
                    self.candidate_quality_oracle.max_branch_target_threshold_ratio
                ),
            )
        )
        return {
            "sufficient": not missing,
            "candidate_state_id": candidate_id,
            "missing": list(dict.fromkeys(missing)),
            "evidence_keys": evidence_keys,
            "evidence_source": source or None,
        }

    def assert_training_decision_evidence(self, action: Mapping[str, Any] | str) -> None:
        """Reject production teacher decisions unsupported by observable evidence."""
        normalized = safe_normalize_action(action)
        if not self.production_dataset_mode:
            return
        tool = normalized["tool"]
        state = self.current_state()
        if tool in CONTEXT_TOOLS:
            family = tool.removeprefix("get_").removesuffix("_context")
            markers = {
                "measurement": ("measurement", "bad_data", "meter", "residual"),
                "parameter": (
                    "parameter",
                    "impedance",
                    "reactance",
                    "resistance",
                    "admittance",
                ),
                "topology": ("topology", "breaker", "switch", "line_status"),
            }[family]
            signature_text = " ".join(
                str(item).lower() for item in state.get("unresolved_signatures") or []
            )
            active_id = state.get("active_state_id")
            current_rejection = any(
                recovery_record_applies_to_state(item, active_id)
                for item in state.get("rejected_hypotheses") or []
            )
            observable_recovery_route = family == "parameter" and bool(
                current_rejection or state.get("accepted_corrections")
            )
            if family == "measurement" and not state.get(
                "no_material_anomaly_remaining"
            ):
                observable_recovery_route = any(
                    safe_normalize_action(
                        item.get("source_action") or item.get("action") or {}
                    )["tool"]
                    == CORRECT_MEASUREMENTS
                    for item in state.get("accepted_corrections") or []
                    if isinstance(item, Mapping)
                )
            if not any(marker in signature_text for marker in markers) and not observable_recovery_route:
                raise ValueError(
                    f"Production training row for {tool} lacks observable {family} evidence."
                )
            return
        if tool in CORRECTION_TOOLS:
            family = tool.removeprefix("correct_")
            context_family = "measurement" if family == "measurements" else (
                "parameter" if family == "parameters" else "topology"
            )
            if not (
                state.get(f"has_fresh_{context_family}_context")
                and str(state.get(f"{context_family}_context_state_id"))
                == str(state.get("active_state_id"))
            ):
                raise ValueError(
                    f"Production training row for {tool} lacks fresh observable "
                    f"{context_family} context."
                )
            supported_signatures = self._supported_correction_signatures(context_family)
            target_signature = action_signature(normalized)
            if target_signature not in supported_signatures:
                raise ValueError(
                    f"Production training row for {tool} is not supported by the latest "
                    f"observable {context_family} context."
                )
            return
        if (
            tool == ASK_FOR_MORE_EVIDENCE
            and normalized["arguments"].get("request")
            in {
                HIF_DIAGNOSTICS_EXHAUSTED_REQUEST,
                RECOVERY_BUDGET_EXHAUSTED_REQUEST,
                RECOVERY_OPTIONS_EXHAUSTED_REQUEST,
            }
        ):
            audit = self._operator_escalation_audit(normalized)
            if not audit["sufficient"]:
                raise ValueError(
                    "Production operator-escalation label lacks exhausted, "
                    "same-state observable HIF evidence: " + ", ".join(audit["missing"])
                )
            return
        if tool in DIAGNOSTIC_TOOLS:
            self._assert_specialized_diagnostic_evidence(normalized, state)
            return
        if tool == FINALIZE_DIAGNOSIS:
            score = state.get("remaining_anomaly_score")
            try:
                score_resolved = (
                    score is not None
                    and float(score) < float(self.process_oracle.anomaly_threshold)
                )
            except (TypeError, ValueError):
                score_resolved = False
            accepted_corrections = state.get("accepted_corrections") or []
            if accepted_corrections:
                raise ValueError(
                    "Production training row for finalize_diagnosis lacks an "
                    "independent post-correction release certificate."
                )
            terminal_field = None
            terminal_field = (
                "no_material_anomaly_remaining"
                if state.get("no_material_anomaly_remaining")
                else "remaining_anomaly_score"
                if score_resolved
                else None
            )
            provenance = state.get("semantic_field_provenance") or {}
            source = provenance.get(terminal_field) if terminal_field else None
            if terminal_field is None or not _observable_provenance_source(source):
                signatures = terminal_explanation_signatures(
                    state.get("unresolved_signatures") or []
                )
                records = [
                    record
                    for record in (state.get("explained_anomalies") or [])
                    if isinstance(record, Mapping) and record.get("explained_signatures")
                ]
                anomalies_explained = (
                    bool(signatures)
                    and not unexplained_signatures(signatures, records)
                    and all(
                        _observable_provenance_source(record.get("evidence_source"))
                        for record in records
                    )
                )
                if anomalies_explained:
                    return
                raise ValueError(
                    "Production training row for finalize_diagnosis lacks observable "
                    "terminal evidence."
                )
            return
        if tool not in {COMMIT_STATE, ROLLBACK_STATE}:
            return
        candidate_id = normalized["arguments"].get("candidate_state_id") or self.current_candidate_id
        report = self.candidate_decision_evidence(
            str(candidate_id) if candidate_id is not None else None
        )
        if not report["sufficient"]:
            raise ValueError(
                f"Production training row for {normalized['tool']} lacks observable evidence: "
                + ", ".join(report["missing"])
            )

    def _assert_specialized_diagnostic_evidence(
        self,
        action: Mapping[str, Any],
        state: Mapping[str, Any],
    ) -> None:
        """Fail closed when a production diagnostic target is not observable."""
        tool = str(action.get("tool") or "")
        unresolved = state.get("unresolved_signatures") or []
        harmonic_codes = matching_evidence_codes(
            unresolved, *ANOMALY_FAMILY_MARKERS["harmonic"]
        )
        unbalance_codes = matching_evidence_codes(
            unresolved, *ANOMALY_FAMILY_MARKERS["three_phase_unbalance"]
        )
        hif_codes = matching_evidence_codes(
            unresolved, *ANOMALY_FAMILY_MARKERS["hif"]
        )
        provenance = state.get("semantic_field_provenance")
        provenance = provenance if isinstance(provenance, Mapping) else {}
        signature_source = provenance.get("unresolved_signatures")
        if not _observable_provenance_source(signature_source):
            raise ValueError(
                f"Production training row for {tool} lacks observable signature provenance."
            )

        available = set(self._observable_evidence_channels())
        if tool in {GET_HARMONIC_CONTEXT, RUN_HSE_FROM_PATH}:
            if not harmonic_codes:
                raise ValueError(
                    f"Production training row for {tool} lacks an observable harmonic signature."
                )
            if "harmonic_measurements" not in available:
                raise ValueError(
                    f"Production training row for {tool} lacks harmonic_measurements telemetry."
                )
            if tool == RUN_HSE_FROM_PATH and self._latest_successful_tool_metrics(
                GET_HARMONIC_CONTEXT
            ) is None:
                raise ValueError(
                    "Production training row for run_hse_from_path lacks successful "
                    "observable harmonic context."
                )
            return

        if tool == RUN_THREE_PHASE_NLM_FROM_PATH:
            if not hif_codes and not unbalance_codes:
                raise ValueError(
                    "Production training row for run_three_phase_nlm_from_path lacks "
                    "an observable HIF or three-phase-unbalance signature."
                )
            # Per-phase branch-current telemetry localizes both HIF lines and
            # unbalance sources directly, so it satisfies the channel gate on
            # its own; a stored NLM diagnostic remains sufficient as before.
            required_channels = (
                {"nlm_diagnostic", "three_phase_branch_currents"}
                if hif_codes
                else {"nlm_diagnostic", "three_phase_voltages", "three_phase_branch_currents"}
            )
            if not (available & required_channels):
                raise ValueError(
                    "Production training row for run_three_phase_nlm_from_path lacks "
                    "required observable three-phase telemetry."
                )
            return

        if tool in {ESTIMATE_HIF_FROM_PATH, ESTIMATE_HIF_MULTISCAN_FROM_PATH}:
            if not hif_codes:
                raise ValueError(
                    f"Production training row for {tool} lacks an observable HIF signature."
                )
            if not (available & {"nlm_diagnostic", "three_phase_branch_currents"}):
                raise ValueError(
                    f"Production training row for {tool} lacks nlm_diagnostic or "
                    "three_phase_branch_currents telemetry."
                )
            if (
                tool == ESTIMATE_HIF_MULTISCAN_FROM_PATH
                and "hif_scan_window" not in available
            ):
                raise ValueError(
                    "Production training row for the multi-scan HIF estimator lacks "
                    "hif_scan_window telemetry."
                )
            nlm_metrics = self._latest_successful_tool_metrics(
                RUN_THREE_PHASE_NLM_FROM_PATH
            )
            summary = (
                nlm_metrics.get("nlm_summary")
                if isinstance(nlm_metrics, Mapping)
                else None
            )
            groups = summary.get("top_hif_groups") if isinstance(summary, Mapping) else None
            supported_rows = {
                int(group["branch_row0"])
                for group in (groups or [])
                if isinstance(group, Mapping) and group.get("branch_row0") is not None
            }
            target = action.get("arguments")
            target = target.get("candidate_branch_row0") if isinstance(target, Mapping) else None
            try:
                target_row = int(target)
            except (TypeError, ValueError):
                target_row = None
            if target_row is None or target_row not in supported_rows:
                raise ValueError(
                    f"Production training row for {tool} targets a branch not supported "
                    "by the latest observable NLM output."
                )
            return

        raise ValueError(f"Unsupported production diagnostic action: {tool}")

    def _operator_escalation_audit(
        self,
        action: Mapping[str, Any],
        *,
        provider_metrics: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Audit the explicit HIF evidence-exhaustion handoff contract.

        Terminal escalation is intentionally separate from anomaly resolution.
        The configured provider may report its evidence inventory, but this
        method independently checks the environment's full same-state history,
        including the fail-closed acceptance decision from every applicable HIF
        estimator.
        """
        normalized = safe_normalize_action(action)
        arguments = normalized["arguments"]
        summary = self.current_state()
        active_id = str(summary.get("active_state_id") or "")
        target_id = str(arguments.get("state_id") or active_id)
        missing: list[str] = []
        if normalized["tool"] != ASK_FOR_MORE_EVIDENCE:
            missing.append("escalation_tool_invalid")
        request = arguments.get("request")
        supported_requests = {
            HIF_DIAGNOSTICS_EXHAUSTED_REQUEST,
            RECOVERY_BUDGET_EXHAUSTED_REQUEST,
            RECOVERY_OPTIONS_EXHAUSTED_REQUEST,
        }
        if request not in supported_requests:
            missing.append("escalation_request_invalid")
        if summary.get("has_open_candidate"):
            missing.append("open_candidate_present")
        if not callable(self.evidence_providers.get(ASK_FOR_MORE_EVIDENCE)):
            missing.append("operator_escalation_provider_missing")
        if not active_id or target_id != active_id:
            missing.append("escalation_state_not_active")
        try:
            active_payload = self.store.get_state(active_id)
            active_hash = str(active_payload.get("state_hash") or "")
        except Exception:
            active_hash = ""
            missing.append("active_state_missing")

        provenance = summary.get("semantic_field_provenance")
        provenance = provenance if isinstance(provenance, Mapping) else {}
        if not _observable_provenance_source(provenance.get("unresolved_signatures")):
            missing.append("anomaly_signature_provenance_not_observable")
        unresolved = unexplained_signatures(
            summary.get("unresolved_signatures") or [],
            summary.get("explained_anomalies") or [],
        )
        try:
            remaining_budget = int(summary.get("remaining_budget") or 0)
        except (TypeError, ValueError):
            remaining_budget = 0
        post_correction_budget_deferral = bool(
            request == RECOVERY_BUDGET_EXHAUSTED_REQUEST
            and remaining_budget == 1
            and summary.get("accepted_corrections")
            and POST_CORRECTION_CONFIRMATION_SIGNATURE in unresolved
        )
        hif_signatures = matching_evidence_codes(
            unresolved, *ANOMALY_FAMILY_MARKERS["hif"]
        )
        if request == HIF_DIAGNOSTICS_EXHAUSTED_REQUEST and not hif_signatures:
            missing.append("unexplained_hif_signature_missing")

        score = summary.get("remaining_anomaly_score")
        try:
            score_unresolved = (
                score is not None
                and float(score) >= float(self.process_oracle.anomaly_threshold)
            )
        except (TypeError, ValueError):
            score_unresolved = False
        post_correction_confirmation_handoff = bool(
            request == RECOVERY_OPTIONS_EXHAUSTED_REQUEST
            and summary.get("accepted_corrections")
            and POST_CORRECTION_CONFIRMATION_SIGNATURE in unresolved
            and not terminal_explanation_signatures(unresolved)
            and not score_unresolved
        )
        if request in {
            RECOVERY_OPTIONS_EXHAUSTED_REQUEST,
            RECOVERY_BUDGET_EXHAUSTED_REQUEST,
        } and not (
            unresolved or score_unresolved
        ):
            missing.append("unresolved_observable_anomaly_missing")

        available = set(self._observable_evidence_channels())
        required_estimators = [ESTIMATE_HIF_FROM_PATH]
        if "hif_scan_window" in available:
            required_estimators.insert(0, ESTIMATE_HIF_MULTISCAN_FROM_PATH)

        def latest_success(tool: str) -> tuple[Mapping[str, Any], Mapping[str, Any]] | None:
            for event in reversed(self.history):
                if not isinstance(event, Mapping):
                    continue
                event_action = safe_normalize_action(event.get("action") or {})
                if event_action["tool"] != tool:
                    continue
                requested = event_action["arguments"].get("state_id")
                if requested is not None and str(requested) != active_id:
                    continue
                output = event.get("tool_output")
                if not isinstance(output, Mapping) or output.get("execution_status") != "success":
                    continue
                metrics = output.get("tool_metrics")
                if not isinstance(metrics, Mapping):
                    continue
                return event_action, metrics
            return None

        def bound_observable_metrics(tool: str) -> tuple[Mapping[str, Any], Mapping[str, Any]] | None:
            event = latest_success(tool)
            if event is None:
                missing.append(f"{tool}_successful_evidence_missing")
                return None
            event_action, metrics = event
            if str(metrics.get("state_id")) != active_id:
                missing.append(f"{tool}_state_id_unbound")
            if str(metrics.get("state_hash")) != active_hash:
                missing.append(f"{tool}_state_hash_unbound")
            if not _observable_provenance_source(metrics.get("evidence_source")):
                missing.append(f"{tool}_source_not_observable")
            return event_action, metrics

        rejected_estimators: list[str] = []
        investigation_tools: list[str] = []
        supported_recovery_targets: list[str] = []
        exhausted_recovery_targets: set[str] = set()
        safety_blocked_recovery_targets: set[str] = set()
        conditional_measurement_targets: dict[str, set[int]] = {}
        eligible_joint_measurement_signature: str | None = None
        outstanding_recovery_targets: list[str] = []
        missing_required_contexts: list[str] = []
        if request == HIF_DIAGNOSTICS_EXHAUSTED_REQUEST:
            nlm_event = bound_observable_metrics(RUN_THREE_PHASE_NLM_FROM_PATH)
            supported_rows: set[int] = set()
            if nlm_event is not None:
                summary_payload = nlm_event[1].get("nlm_summary")
                groups = (
                    summary_payload.get("top_hif_groups")
                    if isinstance(summary_payload, Mapping)
                    else None
                )
                for group in groups or []:
                    if isinstance(group, Mapping) and group.get("branch_row0") is not None:
                        try:
                            supported_rows.add(int(group["branch_row0"]))
                        except (TypeError, ValueError):
                            continue
                if not supported_rows:
                    missing.append("nlm_hif_branch_missing")

            for tool in required_estimators:
                estimator_event = bound_observable_metrics(tool)
                if estimator_event is None:
                    continue
                estimator_action, estimator_metrics = estimator_event
                acceptance = estimator_metrics.get("diagnostic_acceptance")
                if not (
                    isinstance(acceptance, Mapping)
                    and acceptance.get("accepted") is False
                ):
                    missing.append(f"{tool}_not_explicitly_rejected")
                else:
                    rejected_estimators.append(tool)
                target_row = estimator_action["arguments"].get("candidate_branch_row0")
                try:
                    target_row = int(target_row)
                except (TypeError, ValueError):
                    target_row = None
                if target_row is None or target_row not in supported_rows:
                    missing.append(f"{tool}_target_not_nlm_supported")
        elif request in {
            RECOVERY_OPTIONS_EXHAUSTED_REQUEST,
            RECOVERY_BUDGET_EXHAUSTED_REQUEST,
        }:
            bound_observable_metrics(RUN_WLS)
            measurement_signal = bool(
                matching_evidence_codes(
                    unresolved,
                    "measurement",
                    "bad_data",
                    "large_residual",
                    "meter",
                    "residual_outlier",
                )
            )
            parameter_signal = bool(
                matching_evidence_codes(
                    unresolved,
                    "parameter",
                    "impedance",
                    "reactance",
                    "resistance",
                    "admittance",
                    "multiplier",
                )
            )
            topology_signal = bool(
                matching_evidence_codes(
                    unresolved,
                    "topology",
                    "breaker",
                    "switch",
                    "line_status",
                    "connectivity",
                    "islanding",
                )
            )
            measurement_dominant = bool(
                matching_evidence_codes(
                    unresolved, "wls_residual_outlier_dominant"
                )
            )
            branch_dominant = bool(
                matching_evidence_codes(
                    unresolved, "wls_branch_multiplier_dominant"
                )
            )
            required_contexts: set[str] = set()
            if measurement_signal and not (
                branch_dominant and not measurement_dominant
            ):
                required_contexts.add(GET_MEASUREMENT_CONTEXT)
            if parameter_signal and not (
                measurement_dominant and not branch_dominant
            ):
                required_contexts.add(GET_PARAMETER_CONTEXT)
            if topology_signal and not (
                measurement_dominant and not branch_dominant
            ):
                required_contexts.add(GET_TOPOLOGY_CONTEXT)

            context_contracts = {
                "measurement": (GET_MEASUREMENT_CONTEXT, CORRECT_MEASUREMENTS),
                "parameter": (GET_PARAMETER_CONTEXT, CORRECT_PARAMETERS),
                "topology": (GET_TOPOLOGY_CONTEXT, CORRECT_TOPOLOGY),
            }
            for context_family, (
                context_tool,
                correction_tool,
            ) in context_contracts.items():
                bound_event: tuple[Mapping[str, Any], Mapping[str, Any]] | None = None
                for event in reversed(self.history):
                    if not isinstance(event, Mapping):
                        continue
                    event_action = safe_normalize_action(event.get("action") or {})
                    if event_action["tool"] != context_tool:
                        continue
                    requested = event_action["arguments"].get("state_id")
                    if requested is not None and str(requested) != active_id:
                        continue
                    output = event.get("tool_output")
                    if not isinstance(output, Mapping) or output.get("execution_status") != "success":
                        continue
                    metrics = output.get("tool_metrics")
                    if not isinstance(metrics, Mapping):
                        continue
                    if str(metrics.get("state_id")) != active_id:
                        continue
                    if str(metrics.get("state_hash")) != active_hash:
                        continue
                    if not _observable_provenance_source(metrics.get("evidence_source")):
                        continue
                    bound_event = event_action, metrics
                    break
                # A post-commit measurement investigation can atomically
                # bundle parameter and topology route contracts.  Preserve a
                # literal same-family context event as the authoritative audit
                # record; only use the bundle when that direct event is absent.
                # Cross-check the accepted durable ledger and revalidate the
                # raw history contract so stale or tampered evidence fails
                # closed.
                if bound_event is None and context_family in {"parameter", "topology"}:
                    fresh_contexts = summary.get("fresh_context_evidence")
                    durable_evidence = (
                        fresh_contexts.get(context_family)
                        if isinstance(fresh_contexts, Mapping)
                        else None
                    )
                    for event in reversed(self.history):
                        if not isinstance(event, Mapping):
                            continue
                        event_action = safe_normalize_action(event.get("action") or {})
                        if event_action["tool"] != GET_MEASUREMENT_CONTEXT:
                            continue
                        requested = event_action["arguments"].get("state_id")
                        if requested is not None and str(requested) != active_id:
                            continue
                        output = event.get("tool_output")
                        if (
                            not isinstance(output, Mapping)
                            or output.get("execution_status") != "success"
                        ):
                            continue
                        metrics = output.get("tool_metrics")
                        bundled = (
                            metrics.get("branch_route_screening")
                            if isinstance(metrics, Mapping)
                            else None
                        )
                        raw_evidence = (
                            bundled.get(context_family)
                            if isinstance(bundled, Mapping)
                            else None
                        )
                        raw_inventory = (
                            raw_evidence.get("supported_corrections")
                            if isinstance(raw_evidence, Mapping)
                            else None
                        )
                        if not (
                            summary.get(f"has_fresh_{context_family}_context")
                            is True
                            and str(
                                summary.get(
                                    f"{context_family}_context_state_id"
                                )
                                or ""
                            )
                            == active_id
                            and isinstance(durable_evidence, Mapping)
                            and isinstance(raw_evidence, Mapping)
                            and raw_evidence.get("context_tool") == context_tool
                            and str(raw_evidence.get("state_id") or "") == active_id
                            and str(raw_evidence.get("state_hash") or "")
                            == active_hash
                            and _observable_provenance_source(
                                raw_evidence.get("evidence_source")
                            )
                            and isinstance(raw_inventory, (list, tuple))
                            and _branch_route_contract_valid(
                                raw_evidence.get("route_status"), raw_inventory
                            )
                            and str(durable_evidence.get("state_id") or "")
                            == active_id
                            and str(durable_evidence.get("state_hash") or "")
                            == active_hash
                            and durable_evidence.get("evidence_source")
                            == raw_evidence.get("evidence_source")
                            and durable_evidence.get("route_status")
                            == raw_evidence.get("route_status")
                            and durable_evidence.get("supported_corrections")
                            == raw_inventory
                        ):
                            continue
                        bound_event = (
                            {
                                "tool": context_tool,
                                "arguments": {"state_id": active_id},
                            },
                            raw_evidence,
                        )
                        break
                if bound_event is None:
                    # A provider that observably reported insufficient
                    # evidence for this exact active state is a complete
                    # investigation outcome contributing zero recovery
                    # targets; requiring a successful context here would make
                    # a safe handoff unreachable in states the provider
                    # cannot describe.  Only the provider-returned-no-evidence
                    # detail qualifies: an unbound provider response is an
                    # integrity failure, not exhaustion.
                    exhausted_by_provider = False
                    for event in reversed(self.history):
                        if not isinstance(event, Mapping):
                            continue
                        event_action = safe_normalize_action(event.get("action") or {})
                        if event_action["tool"] != context_tool:
                            continue
                        requested = event_action["arguments"].get("state_id")
                        if requested is None or str(requested) != active_id:
                            continue
                        output = event.get("tool_output")
                        if not isinstance(output, Mapping):
                            continue
                        if (
                            output.get("execution_status") == "failure"
                            and str(output.get("error_code") or "")
                            == "insufficient_observable_evidence"
                            and str(output.get("error_detail") or "")
                            == f"{context_tool}_provider_returned_no_evidence"
                        ):
                            exhausted_by_provider = True
                        break
                    if exhausted_by_provider:
                        investigation_tools.append(context_tool)
                        continue
                    if context_tool in required_contexts:
                        missing_required_contexts.append(context_tool)
                    continue
                investigation_tools.append(context_tool)
                raw_supported = bound_event[1].get("supported_corrections")
                if not isinstance(raw_supported, (list, tuple)):
                    missing.append(f"{context_tool}_supported_corrections_missing")
                    continue
                validated_supported: list[dict[str, Any]] = []
                for raw_action in raw_supported:
                    if not isinstance(raw_action, Mapping):
                        missing.append(f"{context_tool}_supported_correction_malformed")
                        continue
                    normalized_supported = safe_normalize_action(raw_action)
                    if normalized_supported["tool"] != correction_tool:
                        missing.append(f"{context_tool}_supported_correction_tool_mismatch")
                        continue
                    supported_state = normalized_supported["arguments"].get("state_id")
                    if supported_state is None or str(supported_state) != active_id:
                        missing.append(f"{context_tool}_supported_correction_state_unbound")
                        continue
                    signature = _semantic_correction_signature(normalized_supported)
                    if signature is None:
                        missing.append(f"{context_tool}_supported_correction_malformed")
                        continue
                    validated_supported.append(normalized_supported)
                    supported_recovery_targets.append(signature)
                if context_tool == GET_MEASUREMENT_CONTEXT:
                    accepted_indices = accepted_measurement_indices(summary)
                    eligible_new_targets = eligible_joint_measurement_targets(
                        summary,
                        self.history,
                        active_id=active_id,
                        supported_actions=validated_supported,
                        accepted_indices=accepted_indices,
                    )
                    eligible_targets = accepted_indices | set(eligible_new_targets)
                    if len(eligible_targets) >= 2:
                        eligible_joint_measurement_signature = (
                            _semantic_correction_signature(
                                {
                                    "tool": CORRECT_MEASUREMENTS,
                                    "arguments": {
                                        "state_id": active_id,
                                        "suspect_group": sorted(eligible_targets),
                                    },
                                }
                            )
                        )
                    refinement_ready = (
                        bound_event[1].get("accepted_target_refinement") is True
                    )
                    for supported_action in validated_supported:
                        targets = measurement_target_indices(supported_action)
                        if len(targets) < 2:
                            continue
                        is_immediate_refinement = bool(
                            refinement_ready
                            and accepted_indices
                            and targets == accepted_indices
                        )
                        if is_immediate_refinement:
                            continue
                        conditional_signature = _semantic_correction_signature(
                            supported_action
                        )
                        if conditional_signature is not None:
                            conditional_measurement_targets[
                                conditional_signature
                            ] = targets - accepted_indices

            for record in summary.get("rejected_hypotheses") or []:
                if not isinstance(record, Mapping):
                    continue
                if str(record.get("candidate_parent_id") or "") != active_id:
                    continue
                rejected_action = safe_normalize_action(
                    record.get("source_action") or {}
                )
                if (
                    str(rejected_action["arguments"].get("state_id") or "")
                    != active_id
                ):
                    continue
                signature = _semantic_correction_signature(rejected_action)
                if signature is not None:
                    exhausted_recovery_targets.add(signature)

            family_wide_failures = {
                CORRECT_PARAMETERS: {
                    "parameter_scans_missing",
                    "correction_route_not_actionable",
                },
                CORRECT_TOPOLOGY: {
                    "topology_correction_unsupported",
                    "correction_route_not_actionable",
                },
            }
            process_failure_codes = {
                "schema_error",
                "unknown_tool",
                "candidate_lifecycle_violation",
                "unknown_state_id",
                "state_reference_mismatch",
                "missing_precondition",
                "post_correction_confirmation_required",
            }
            exhausted_families: set[str] = set()
            for event in self.history:
                if not isinstance(event, Mapping):
                    continue
                event_action = safe_normalize_action(event.get("action") or {})
                if event_action["tool"] not in CORRECTION_TOOLS:
                    continue
                requested = event_action["arguments"].get("state_id")
                if requested is None or str(requested) != active_id:
                    continue
                output = event.get("tool_output")
                if not isinstance(output, Mapping) or output.get("execution_status") != "failure":
                    continue
                error_code = str(output.get("error_code") or "")
                if error_code in process_failure_codes:
                    continue
                signature = _semantic_correction_signature(event_action)
                if signature is not None:
                    exhausted_recovery_targets.add(signature)
                if error_code in family_wide_failures.get(event_action["tool"], set()):
                    exhausted_families.add(event_action["tool"])

            supported_recovery_targets = list(
                dict.fromkeys(supported_recovery_targets)
            )
            supported_recovery_target_set = set(supported_recovery_targets)
            for signature, new_targets in conditional_measurement_targets.items():
                if signature == eligible_joint_measurement_signature:
                    continue
                singleton_signatures = {
                    _semantic_correction_signature(
                        {
                            "tool": CORRECT_MEASUREMENTS,
                            "arguments": {
                                "state_id": active_id,
                                "suspect_group": [target],
                            },
                        }
                    )
                    for target in new_targets
                }
                # A conditional group is unavailable only after every new
                # constituent singleton is both provider-supported and
                # exhausted.  Before then the ordinary singleton inventory
                # keeps a handoff fail-closed.  If the shared evidence gate
                # admits the exact group, it remains outstanding until tried.
                if (
                    singleton_signatures
                    and None not in singleton_signatures
                    and singleton_signatures <= supported_recovery_target_set
                    and singleton_signatures <= exhausted_recovery_targets
                ):
                    safety_blocked_recovery_targets.add(signature)
            accepted_measurement_partial = any(
                safe_normalize_action(
                    item.get("source_action") or item.get("action") or {}
                )["tool"]
                == CORRECT_MEASUREMENTS
                for item in summary.get("accepted_corrections") or []
                if isinstance(item, Mapping)
            )
            fresh_evidence = summary.get("fresh_context_evidence")
            unavailable_branch_routes: set[str] = set()
            if isinstance(fresh_evidence, Mapping):
                for branch_family in ("parameter", "topology"):
                    branch_evidence = fresh_evidence.get(branch_family)
                    if (
                        isinstance(branch_evidence, Mapping)
                        and branch_evidence.get("route_status")
                        == "unavailable_or_inconclusive"
                    ):
                        unavailable_branch_routes.add(branch_family)
            if (
                accepted_measurement_partial
                and (
                    unavailable_branch_routes
                    or (parameter_signal and not measurement_dominant)
                )
            ):
                # A residual correction can mask the still-observable branch
                # anomaly after a partial measurement commit.  These provider
                # targets remain visible in the ledger but are deliberately
                # unavailable to autonomous recovery until branch evidence is
                # resolved; an operator handoff is the safe terminal action.
                safety_blocked_recovery_targets.update(
                    signature
                    for signature in supported_recovery_targets
                    if signature.startswith(f"{CORRECT_MEASUREMENTS}:")
                )
            if post_correction_confirmation_handoff:
                # Confirmation-provider suggestions are retained for operator
                # review, but the synthetic controller obligation is not a
                # license for another autonomous correction.
                safety_blocked_recovery_targets.update(
                    supported_recovery_targets
                )
            outstanding_recovery_targets = [
                signature
                for signature in supported_recovery_targets
                if signature not in exhausted_recovery_targets
                and signature not in safety_blocked_recovery_targets
                and signature.split(":", 1)[0] not in exhausted_families
            ]
            if not investigation_tools and not post_correction_budget_deferral:
                missing.append("same_state_investigation_evidence_missing")
            if request == RECOVERY_OPTIONS_EXHAUSTED_REQUEST:
                if (
                    missing_required_contexts
                    and not post_correction_confirmation_handoff
                ):
                    missing.append("required_recovery_contexts_missing")
                if (
                    outstanding_recovery_targets
                    and not post_correction_confirmation_handoff
                ):
                    missing.append("same_state_supported_corrections_unexhausted")
            else:
                if not 0 < remaining_budget < 4:
                    missing.append("autonomous_recovery_budget_not_exhausted")

        if provider_metrics is not None:
            if str(provider_metrics.get("state_id")) != active_id:
                missing.append("escalation_provider_state_id_unbound")
            if str(provider_metrics.get("state_hash")) != active_hash:
                missing.append("escalation_provider_state_hash_unbound")
            if not _observable_provenance_source(provider_metrics.get("evidence_source")):
                missing.append("escalation_provider_source_not_observable")
            if request == RECOVERY_BUDGET_EXHAUSTED_REQUEST:
                if provider_metrics.get("autonomous_budget_available") is not False:
                    missing.append("autonomous_budget_not_explicitly_exhausted")
                if provider_metrics.get("additional_evidence_available") is not True:
                    missing.append("remaining_options_not_acknowledged")
            elif provider_metrics.get("additional_evidence_available") is not False:
                missing.append("additional_evidence_not_explicitly_exhausted")
            if provider_metrics.get("operator_review_required") is not True:
                missing.append("operator_review_not_required")
            if provider_metrics.get("request") != request:
                missing.append("escalation_provider_request_mismatch")

        additional_evidence_available = (
            True
            if request == RECOVERY_BUDGET_EXHAUSTED_REQUEST
            else bool(missing_required_contexts or outstanding_recovery_targets)
        )
        ledger = {
            "request": request,
            "active_state_id": active_id,
            "active_state_hash": active_hash,
            "family": (
                "hif"
                if request == HIF_DIAGNOSTICS_EXHAUSTED_REQUEST
                else "recovery_budget"
                if request == RECOVERY_BUDGET_EXHAUSTED_REQUEST
                else "mixed_or_unresolved"
            ),
            "unexplained_signature_count": len(unresolved),
            "required_estimators": (
                required_estimators
                if request == HIF_DIAGNOSTICS_EXHAUSTED_REQUEST
                else []
            ),
            "rejected_estimators": rejected_estimators,
            "investigation_tools": investigation_tools,
            "supported_recovery_target_count": len(supported_recovery_targets),
            "exhausted_recovery_target_count": len(
                set(supported_recovery_targets) & exhausted_recovery_targets
            ),
            "outstanding_recovery_targets": outstanding_recovery_targets,
            "safety_blocked_recovery_targets": sorted(
                safety_blocked_recovery_targets
            ),
            "missing_required_contexts": missing_required_contexts,
            "additional_evidence_available": additional_evidence_available,
            "autonomous_budget_available": (
                False if request == RECOVERY_BUDGET_EXHAUSTED_REQUEST else None
            ),
            "post_correction_confirmation_deferred": (
                post_correction_budget_deferral
            ),
            "post_correction_confirmation_handoff": (
                post_correction_confirmation_handoff
            ),
            "operator_review_required": True,
        }
        return {
            "sufficient": not missing,
            "missing": list(dict.fromkeys(missing)),
            "ledger": ledger,
        }

    def _latest_successful_tool_metrics(self, tool: str) -> Mapping[str, Any] | None:
        for event in reversed(self.history):
            if not isinstance(event, Mapping):
                continue
            if safe_normalize_action(event.get("action") or {})["tool"] != tool:
                continue
            output = event.get("tool_output")
            if not isinstance(output, Mapping) or output.get("execution_status") != "success":
                return None
            metrics = output.get("tool_metrics")
            return metrics if isinstance(metrics, Mapping) else None
        return None

    def _latest_successful_wls_score(self, state_id: str) -> float | None:
        """Return the latest observable anomaly score bound to ``state_id``."""
        if not state_id:
            return None
        for event in reversed(self.history):
            if not isinstance(event, Mapping):
                continue
            event_action = safe_normalize_action(event.get("action") or {})
            if event_action["tool"] not in {RUN_WLS, VERIFY_CANDIDATE}:
                continue
            if str(event_action["arguments"].get("state_id") or "") != state_id:
                continue
            output = event.get("tool_output")
            if not isinstance(output, Mapping) or output.get("execution_status") != "success":
                return None
            metrics = output.get("tool_metrics")
            if not isinstance(metrics, Mapping):
                return None
            value = metrics.get("remaining_anomaly_score")
            try:
                return None if value is None else float(value)
            except (TypeError, ValueError):
                return None
        return None

    def _parent_target_metric_value(
        self, state_id: str, source_action: Mapping[str, Any]
    ) -> float | None:
        """Return context evidence for the exact correction target.

        Continuous target progress must compare the candidate against the
        same target that a fresh, state-bound provider context offered.  A
        global WLS improvement cannot substitute for this local baseline.
        """
        if not state_id:
            return None
        normalized = safe_normalize_action(source_action)
        tool = normalized["tool"]
        context_tool = {
            CORRECT_MEASUREMENTS: GET_MEASUREMENT_CONTEXT,
            CORRECT_PARAMETERS: GET_PARAMETER_CONTEXT,
            CORRECT_TOPOLOGY: GET_TOPOLOGY_CONTEXT,
        }.get(tool)
        if context_tool is None:
            return None
        try:
            expected_hash = self.store.state_hash(state_id)
            expected_signature = action_signature(normalized)
        except (StateStoreError, TypeError, ValueError):
            return None

        for event in reversed(self.history):
            if not isinstance(event, Mapping):
                continue
            event_action = safe_normalize_action(event.get("action") or {})
            if event_action["tool"] != context_tool:
                continue
            requested = event_action["arguments"].get("state_id")
            if requested is not None and str(requested) != state_id:
                continue
            output = event.get("tool_output")
            if not isinstance(output, Mapping) or output.get("execution_status") != "success":
                return None
            metrics = output.get("tool_metrics")
            if not isinstance(metrics, Mapping):
                return None
            if str(metrics.get("state_id")) != state_id:
                return None
            if str(metrics.get("state_hash")) != str(expected_hash):
                return None
            if not _observable_provenance_source(metrics.get("evidence_source")):
                return None
            supported = metrics.get("supported_corrections")
            if not isinstance(supported, (list, tuple)):
                return None
            supported_signatures: set[str] = set()
            for item in supported:
                if not isinstance(item, Mapping):
                    return None
                normalized_item = safe_normalize_action(item)
                if normalized_item["tool"] == INVALID_ACTION:
                    return None
                try:
                    supported_signatures.add(action_signature(normalized_item))
                except (TypeError, ValueError):
                    return None
            if expected_signature not in supported_signatures:
                return None

            arguments = normalized["arguments"]
            if tool == CORRECT_MEASUREMENTS:
                group = arguments.get("suspect_group")
                updates = arguments.get("measurement_updates")
                if isinstance(group, (list, tuple)):
                    raw_target_indices = group
                elif isinstance(updates, Mapping):
                    raw_target_indices = updates.keys()
                else:
                    return None
                target_indices: set[int] = set()
                for index in raw_target_indices:
                    if (
                        not isinstance(index, int)
                        or isinstance(index, bool)
                        or index < 0
                    ):
                        return None
                    target_indices.add(index)
                if not target_indices:
                    return None
                findings = metrics.get("measurement_findings")
                if not isinstance(findings, (list, tuple)):
                    return None
                values: list[float] = []
                for item in findings:
                    if not isinstance(item, Mapping):
                        return None
                    index0 = item.get("index0")
                    if (
                        not isinstance(index0, int)
                        or isinstance(index0, bool)
                        or index0 < 0
                        or item.get("value") is None
                    ):
                        return None
                    try:
                        value = abs(float(item["value"]))
                    except (TypeError, ValueError, OverflowError):
                        return None
                    if not math.isfinite(value):
                        return None
                    if index0 in target_indices:
                        values.append(value)
            else:
                raw_targets = [
                    (key, arguments[key])
                    for key in ("branch_row0", "line_index1", "line_index")
                    if arguments.get(key) is not None
                ]
                if len(raw_targets) != 1:
                    return None
                target_key, raw_target = raw_targets[0]
                if not isinstance(raw_target, int) or isinstance(raw_target, bool):
                    return None
                target_row0 = raw_target if target_key == "branch_row0" else raw_target - 1
                if target_row0 < 0:
                    return None
                findings_key = (
                    "parameter_findings"
                    if tool == CORRECT_PARAMETERS
                    else "topology_findings"
                )
                findings = metrics.get(findings_key)
                if not isinstance(findings, (list, tuple)):
                    return None
                values = []
                for item in findings:
                    if not isinstance(item, Mapping):
                        return None
                    row0 = item.get("line_row0")
                    if (
                        not isinstance(row0, int)
                        or isinstance(row0, bool)
                        or row0 < 0
                        or item.get("value") is None
                    ):
                        return None
                    try:
                        value = abs(float(item["value"]))
                    except (TypeError, ValueError, OverflowError):
                        return None
                    if not math.isfinite(value):
                        return None
                    if row0 == target_row0:
                        values.append(value)
            return max(values) if values else None
        return None

    def _accepted_branch_rows(self) -> set[int]:
        rows: set[int] = set()
        for item in self.context_flags.get("accepted_corrections") or []:
            if not isinstance(item, Mapping):
                continue
            action = safe_normalize_action(
                item.get("source_action") or item.get("action") or {}
            )
            if action["tool"] not in {CORRECT_PARAMETERS, CORRECT_TOPOLOGY}:
                continue
            arguments = action["arguments"]
            try:
                if arguments.get("branch_row0") is not None:
                    row0 = int(arguments["branch_row0"])
                elif arguments.get("line_index1") is not None:
                    row0 = int(arguments["line_index1"]) - 1
                elif arguments.get("line_index") is not None:
                    row0 = int(arguments["line_index"]) - 1
                else:
                    continue
            except (TypeError, ValueError, OverflowError):
                continue
            if row0 >= 0:
                rows.add(row0)
        return rows

    _RESIDUAL_OUTLIER_SIGNATURE_RE = re.compile(
        r"^wls_residual_outlier\S*\s+index=(\d+)\s+channel=(\S+)"
    )

    def _measurement_target_cluster(
        self, source_action: Mapping[str, Any]
    ) -> tuple[str, int] | None:
        """Observable same-channel residual cluster containing the target.

        Returns ``(channel, member_count)`` when every index in the action's
        suspect group carries a current residual-outlier signature and all of
        them share one measurement channel; ``member_count`` is how many
        current residual outliers flag that channel.  Derived only from the
        active state's public unresolved-signature ledger.
        """
        arguments = safe_normalize_action(source_action)["arguments"]
        group = arguments.get("suspect_group")
        if not isinstance(group, (list, tuple)) or not group:
            return None
        try:
            targets = {int(index) for index in group}
        except (TypeError, ValueError):
            return None
        channel_by_index: dict[int, str] = {}
        channel_counts: dict[str, int] = {}
        for raw in self.context_flags.get("unresolved_signatures") or []:
            match = self._RESIDUAL_OUTLIER_SIGNATURE_RE.match(str(raw))
            if not match:
                continue
            index, channel = int(match.group(1)), match.group(2)
            channel_by_index[index] = channel
            channel_counts[channel] = channel_counts.get(channel, 0) + 1
        channels = {channel_by_index.get(target) for target in targets}
        if None in channels or len(channels) != 1:
            return None
        channel = channels.pop()
        return channel, channel_counts.get(channel, 0)

    def _accepted_measurement_channel_evidence(self) -> tuple[int, str | None]:
        """Count accepted measurement targets and their single shared channel.

        Channels come from each accepted candidate's stored verification
        metrics, which recorded the target's channel from the public
        signature ledger current at its own verification.  The shared
        channel is None when there are no acceptances, any channel is
        unknown, or the accepted channels are mixed.
        """
        targets: set[int] = set()
        channels: set[str] = set()
        unknown = False
        for record in self.context_flags.get("accepted_corrections") or []:
            if not isinstance(record, Mapping):
                continue
            action = safe_normalize_action(record.get("source_action") or {})
            if action["tool"] != CORRECT_MEASUREMENTS:
                continue
            try:
                targets.update(
                    int(index)
                    for index in action["arguments"].get("suspect_group") or []
                )
            except (TypeError, ValueError):
                unknown = True
            channel = None
            candidate_id = str(record.get("candidate_state_id") or "")
            if candidate_id and self.store.exists(candidate_id):
                verification = self.store.get_state(candidate_id).get(
                    "verification_output"
                )
                if isinstance(verification, Mapping):
                    channel = verification.get("measurement_target_channel")
            if channel is None:
                unknown = True
            else:
                channels.add(str(channel))
        shared = channels.pop() if not unknown and len(channels) == 1 else None
        return len(targets), shared

    def _measurement_target_rank_one(
        self, source_action: Mapping[str, Any]
    ) -> bool | None:
        """Whether the singleton target is the top-ranked residual outlier.

        Residual-outlier signatures enter the public ledger in descending
        normalized-residual order, so the first one is the rank-1 finding.
        Returns None for grouped actions or when no residual signature is
        current.
        """
        arguments = safe_normalize_action(source_action)["arguments"]
        group = arguments.get("suspect_group")
        if not isinstance(group, (list, tuple)) or len(group) != 1:
            return None
        try:
            target = int(group[0])
        except (TypeError, ValueError):
            return None
        for raw in self.context_flags.get("unresolved_signatures") or []:
            match = self._RESIDUAL_OUTLIER_SIGNATURE_RE.match(str(raw))
            if match:
                return int(match.group(1)) == target
        return None

    def _measurement_branch_routes_closed(self) -> bool:
        """Both branch routes screened on the active state with nothing to try.

        A conservative observable proxy for the expert's route-exhaustion
        predicate: fresh parameter and topology evidence bound to the active
        state, each with an empty supported inventory and an explicit
        route status.
        """
        fresh = self.context_flags.get("fresh_context_evidence")
        if not isinstance(fresh, Mapping):
            return False
        active_id = str(self.store.active_state_id)
        for family in ("parameter", "topology"):
            evidence = fresh.get(family)
            if not isinstance(evidence, Mapping):
                return False
            if str(evidence.get("state_id") or "") != active_id:
                return False
            supported = evidence.get("supported_corrections")
            if not isinstance(supported, (list, tuple)) or supported:
                return False
            if evidence.get("route_status") not in (
                "complete_negative",
                "unavailable_or_inconclusive",
            ):
                return False
        return True

    def _measurement_target_branch_colocated(
        self, state_id: str, source_action: Mapping[str, Any]
    ) -> bool | None:
        """Bind a measurement target to direct-flow evidence on repaired rows.

        ``None`` means the provider context was absent or malformed, so a
        nondominant sequential measurement candidate cannot be accepted.
        """
        accepted_rows = self._accepted_branch_rows()
        if not accepted_rows:
            return False
        signature_rows: set[int] = set()
        for signature in self.context_flags.get("unresolved_signatures") or []:
            if "wls_branch_multiplier" not in str(signature):
                continue
            match = re.search(r"(?:^|\s)line=(\d+)(?:\s|$)", str(signature))
            if match:
                signature_rows.add(int(match.group(1)) - 1)
        candidate_rows = accepted_rows & signature_rows
        if not candidate_rows:
            return False

        normalized = safe_normalize_action(source_action)
        if normalized["tool"] != CORRECT_MEASUREMENTS:
            return None
        arguments = normalized["arguments"]
        group = arguments.get("suspect_group")
        updates = arguments.get("measurement_updates")
        raw_targets = (
            group
            if isinstance(group, (list, tuple))
            else updates.keys()
            if isinstance(updates, Mapping)
            else ()
        )
        target_indices: set[int] = set()
        for raw_index in raw_targets:
            if (
                not isinstance(raw_index, int)
                or isinstance(raw_index, bool)
                or raw_index < 0
            ):
                return None
            target_indices.add(raw_index)
        if not target_indices:
            return None

        try:
            expected_hash = self.store.state_hash(state_id)
            expected_signature = action_signature(normalized)
        except (StateStoreError, TypeError, ValueError):
            return None
        for event in reversed(self.history):
            if not isinstance(event, Mapping):
                continue
            event_action = safe_normalize_action(event.get("action") or {})
            if event_action["tool"] != GET_MEASUREMENT_CONTEXT:
                continue
            requested = event_action["arguments"].get("state_id")
            if requested is not None and str(requested) != state_id:
                continue
            output = event.get("tool_output")
            if not isinstance(output, Mapping) or output.get("execution_status") != "success":
                return None
            metrics = output.get("tool_metrics")
            if not isinstance(metrics, Mapping):
                return None
            if str(metrics.get("state_id")) != state_id:
                return None
            if str(metrics.get("state_hash")) != str(expected_hash):
                return None
            if not _observable_provenance_source(metrics.get("evidence_source")):
                return None
            supported = metrics.get("supported_corrections")
            if not isinstance(supported, (list, tuple)):
                return None
            signatures: set[str] = set()
            for raw_action in supported:
                if not isinstance(raw_action, Mapping):
                    return None
                try:
                    signatures.add(action_signature(raw_action))
                except (TypeError, ValueError):
                    return None
            if expected_signature not in signatures:
                return None
            findings = metrics.get("measurement_findings")
            if not isinstance(findings, (list, tuple)):
                return None
            for item in findings:
                if not isinstance(item, Mapping):
                    return None
                if str(item.get("channel") or "") not in {"Pf", "Qf", "Pt", "Qt"}:
                    continue
                index0 = item.get("index0")
                offset = item.get("channel_offset")
                if (
                    not isinstance(index0, int)
                    or isinstance(index0, bool)
                    or index0 < 0
                    or not isinstance(offset, int)
                    or isinstance(offset, bool)
                    or offset < 0
                ):
                    return None
                if index0 in target_indices and offset in candidate_rows:
                    return True
            return False
        return None

    def _supported_correction_signatures(self, context_family: str) -> set[str]:
        context_tool = f"get_{context_family}_context"
        correction_tool = {
            "measurement": CORRECT_MEASUREMENTS,
            "parameter": CORRECT_PARAMETERS,
            "topology": CORRECT_TOPOLOGY,
        }[context_family]
        state = self.current_state()
        active_id = str(state.get("active_state_id") or "")

        def bound_signatures(evidence: Mapping[str, Any]) -> set[str]:
            if str(evidence.get("state_id") or "") != active_id:
                return set()
            if (
                context_family in {"parameter", "topology"}
                and "route_status" in evidence
                and evidence.get("route_status") != "actionable"
            ):
                return set()
            raw_actions = evidence.get("supported_corrections")
            if not isinstance(raw_actions, (list, tuple)):
                return set()
            signatures: set[str] = set()
            for raw_action in raw_actions:
                if not isinstance(raw_action, Mapping):
                    return set()
                normalized = safe_normalize_action(raw_action)
                if normalized["tool"] != correction_tool:
                    return set()
                target_state = normalized["arguments"].get("state_id")
                if target_state is None or str(target_state) != active_id:
                    return set()
                signatures.add(action_signature(normalized))
            return signatures

        if (
            state.get(f"has_fresh_{context_family}_context")
            and str(state.get(f"{context_family}_context_state_id") or "")
            == active_id
        ):
            fresh = state.get("fresh_context_evidence")
            evidence = (
                fresh.get(context_family)
                if isinstance(fresh, Mapping)
                else None
            )
            if isinstance(evidence, Mapping):
                # This ledger includes branch inventories bundled by a fresh
                # measurement context, and remains authoritative after a
                # rejected candidate rolls back to the same active state.
                return bound_signatures(evidence)

        # Legacy fallback for observations created before the durable context
        # ledger existed.  The newest matching context is authoritative even
        # when its inventory is empty.
        for event in reversed(self.history):
            if not isinstance(event, Mapping):
                continue
            event_action = event.get("action")
            normalized_event = safe_normalize_action(event_action or {})
            if normalized_event["tool"] != context_tool:
                continue
            requested = normalized_event["arguments"].get("state_id")
            if requested is not None and str(requested) != active_id:
                continue
            output = event.get("tool_output")
            if (
                not isinstance(output, Mapping)
                or output.get("execution_status") != "success"
            ):
                return set()
            metrics = output.get("tool_metrics")
            if not isinstance(metrics, Mapping):
                return set()
            return bound_signatures(metrics)
        return set()

    def enumerate_available_actions(self, state: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
        summary = dict(state or self.current_state())
        if summary.get("has_unverified_candidate"):
            return [{"tool": RUN_WLS, "arguments": {"state_id": summary["candidate_state_id"]}}]
        if summary.get("has_verified_candidate"):
            disposition = summary.get("candidate_disposition")
            if disposition in {"ACCEPT_PARTIAL", "ACCEPT_FINAL"}:
                return [{"tool": COMMIT_STATE, "arguments": {"candidate_state_id": summary["candidate_state_id"]}}]
            if disposition == "INCONCLUSIVE":
                return [
                    {"tool": ASK_FOR_MORE_EVIDENCE, "arguments": {"state_id": summary["candidate_state_id"]}},
                    {"tool": ROLLBACK_STATE, "arguments": {"candidate_state_id": summary["candidate_state_id"]}},
                ]
            return [{"tool": ROLLBACK_STATE, "arguments": {"candidate_state_id": summary["candidate_state_id"]}}]
        active_id = summary["active_state_id"]
        actions = [
            {"tool": RUN_WLS, "arguments": {"state_id": active_id}},
            {"tool": "get_measurement_context", "arguments": {"state_id": active_id}},
            {"tool": "get_parameter_context", "arguments": {"state_id": active_id}},
            {"tool": "get_topology_context", "arguments": {"state_id": active_id}},
            {"tool": FINALIZE_DIAGNOSIS, "arguments": {}},
        ]
        return [
            action for action in actions
            if self.process_oracle.check(summary, action, store=self.store)["process_valid"]
        ]

    def clone(self) -> TransactionalPSSEEnv:
        """Return an isolated branch suitable for counterfactual evaluation."""
        branch = copy.copy(self)
        branch.store = copy.deepcopy(self.store)
        branch.context_flags = copy.deepcopy(self.context_flags)
        branch.history = copy.deepcopy(self.history)
        branch._oracle_payload = copy.deepcopy(self._oracle_payload)
        # Counterfactual execution must not mutate stateful solver/oracle
        # collaborators owned by the live rollout environment.
        branch.wls_runner = _clone_collaborator(self.wls_runner, "wls_runner")
        branch.process_oracle = _clone_collaborator(self.process_oracle, "process_oracle")
        branch.candidate_quality_oracle = _clone_collaborator(
            self.candidate_quality_oracle, "candidate_quality_oracle"
        )
        branch.context_providers = {
            name: _clone_collaborator(provider, f"context_providers[{name}]")
            for name, provider in self.context_providers.items()
        }
        branch.evidence_providers = {
            name: _clone_collaborator(provider, f"evidence_providers[{name}]")
            for name, provider in self.evidence_providers.items()
        }
        branch.correction_executors = {
            name: _clone_collaborator(provider, f"correction_executors[{name}]")
            for name, provider in self.correction_executors.items()
        }
        branch.approved_deterministic_providers = set(self.approved_deterministic_providers)
        return branch

    def clone_from(self, state_id: str | None = None) -> TransactionalPSSEEnv:
        if state_id is not None and str(state_id) not in {
            str(self.store.active_state_id),
            str(self.current_candidate_id),
        }:
            raise StateStoreError("clone_from only accepts the active or current candidate state.")
        return self.clone()

    def _step_correction(self, action: dict[str, Any]) -> dict[str, Any]:
        args = action["arguments"]
        parent_state_id = str(args.get("state_id") or self.store.active_state_id)
        parent_payload = self.store.get_state(parent_state_id)
        executor = self.correction_executors.get(action["tool"])
        executor_metrics: dict[str, Any] = {}
        if executor is not None:
            raw_result = executor(copy.deepcopy(parent_payload), copy.deepcopy(action))
            if not isinstance(raw_result, Mapping):
                raise TypeError("Correction executor must return a mapping.")
            result = dict(copy.deepcopy(raw_result))
            status = str(result.pop("execution_status", "success"))
            if status != "success":
                error_code = str(
                    result.pop("error_code", "correction_execution_failure")
                )
                error_detail = result.pop("error_detail", None)
                return self._standard_output(
                    execution_status="failure",
                    error_code=error_code,
                    error_detail=error_detail,
                    state_mutated=False,
                    tool_metrics=policy_safe_copy(result),
                    valid_next_actions=self.process_oracle.repair_actions(
                        self.current_state(), error_code, error_detail
                    ),
                )
            result.pop("error_code", None)
            result.pop("error_detail", None)
            candidate_state = result.pop("candidate_state", None)
            modification = result.pop("modification", None)
            if modification is not None and not isinstance(modification, Mapping):
                raise TypeError("Correction executor modification must be a mapping.")
            if modification is None and isinstance(candidate_state, Mapping):
                modification = {
                    key: copy.deepcopy(candidate_state[key])
                    for key in ("case", "measurements", "metadata_updates")
                    if key in candidate_state
                }
            if modification is None:
                physical_keys = {
                    "case",
                    "case_updates",
                    "measurements",
                    "measurement_updates",
                    "metadata_updates",
                    "line_index",
                    "line_index1",
                    "branch_row0",
                    "branch_id",
                    "cb_name",
                    "parameter",
                    "field",
                    "value",
                    "corrected_value",
                    "new_value",
                    "multiplier",
                    "status",
                    "expected_status",
                    "status_field",
                }
                modification = {
                    key: copy.deepcopy(value)
                    for key, value in result.items()
                    if key in physical_keys
                }
                result = {key: value for key, value in result.items() if key not in physical_keys}
            modification = dict(modification or {})
            if not modification:
                raise StateStoreError(
                    "Correction executor returned no physical candidate modification."
                )
            executor_metrics = policy_safe_copy(result)
            executor_metrics.setdefault("evidence_source", f"configured_provider:{_provider_label(executor)}")
        else:
            modification = args.get("modification")
            if not isinstance(modification, Mapping):
                modification = {
                    key: value
                    for key, value in args.items()
                    if key not in {"state_id", "candidate_state_id"}
                }
            modification = dict(modification)
        modification.setdefault("modification_signature", action_signature(action))
        candidate_id = self.store.clone_candidate(
            parent_state_id,
            modification,
            action,
            created_at_step=len(self.history) + 1,
        )
        self.current_candidate_id = candidate_id
        provenance = self.store.candidate_provenance(candidate_id)
        return self._standard_output(
            execution_status="success",
            state_mutated=True,
            candidate_state_id=candidate_id,
            tool_metrics={**provenance, **executor_metrics},
        )

    def _step_wls(self, action: dict[str, Any]) -> dict[str, Any]:
        args = action["arguments"]
        state_id = str(args.get("state_id") or self.current_candidate_id or self.store.active_state_id)
        state_payload = self.store.get_state(state_id)
        if self.wls_runner is None:
            metrics: dict[str, Any] = {
                "state_id": state_id,
                "state_hash": state_payload["state_hash"],
                "wls_objective": state_payload.get("metadata", {}).get("wls_objective", 0.0),
                "remaining_anomaly_score": state_payload.get("metadata", {}).get("remaining_anomaly_score"),
            }
        else:
            runner_payload = copy.deepcopy(state_payload)
            # Deployment WLS runners merge previously recorded sensor-sourced
            # signatures with their own solve-derived ones, so they receive the
            # same deployment-safe observation the other providers already get.
            runner_payload["policy_observation"] = self.get_policy_observation().as_dict()
            metrics = dict(self.wls_runner(runner_payload))
        runner_status = metrics.pop("execution_status", "success")
        if runner_status != "success":
            error_code = str(metrics.pop("error_code", "wls_failure"))
            error_detail = metrics.pop("error_detail", None)
            # A candidate whose verification solve itself fails (for example a
            # topology hypothesis that islands part of the network and makes
            # the estimator singular) can never produce acceptance evidence.
            # The solver failure is observable rejection evidence, so the
            # candidate is marked verified-REJECT; leaving it unverified would
            # deadlock the episode with no legal rollback.
            if (
                self.current_candidate_id is not None
                and state_id == str(self.current_candidate_id)
                and not state_payload.get("verified")
            ):
                rejection_metrics = {
                    "state_id": state_id,
                    "state_hash": state_payload["state_hash"],
                    "evidence_source": str(
                        metrics.get("evidence_source")
                        or (
                            f"configured_provider:{_provider_label(self.wls_runner)}"
                            if self.wls_runner is not None
                            else "controller_default"
                        )
                    ),
                    "verification_error_code": error_code,
                    "converged": False,
                    "power_flow_converged": False,
                    "physical_constraints_ok": False,
                }
                assessment = {
                    "disposition": "REJECT",
                    "progress_class": "verification_solver_failure",
                    "collateral_damage": True,
                    "rationale_codes": ["verification_solver_failure", error_code],
                }
                self.store.mark_verified(
                    state_id, rejection_metrics, "REJECT", assessment
                )
                return self._standard_output(
                    execution_status="failure",
                    error_code=error_code,
                    error_detail=error_detail,
                    state_mutated=True,
                    candidate_state_id=self.current_candidate_id,
                    tool_metrics={
                        **metrics,
                        **rejection_metrics,
                        "candidate_disposition": "REJECT",
                        "candidate_assessment": assessment,
                    },
                    valid_next_actions=[
                        {
                            "tool": ROLLBACK_STATE,
                            "arguments": {
                                "candidate_state_id": str(self.current_candidate_id)
                            },
                        }
                    ],
                )
            return self._standard_output(
                execution_status="failure",
                error_code=error_code,
                error_detail=error_detail,
                state_mutated=False,
                tool_metrics=metrics,
                valid_next_actions=self.process_oracle.repair_actions(
                    self.current_state(), "state_reference_mismatch", "wls_failure"
                ),
            )
        metrics.setdefault("state_id", state_id)
        metrics.setdefault("state_hash", state_payload["state_hash"])
        if self.wls_runner is not None:
            metrics.setdefault(
                "evidence_source", f"configured_provider:{_provider_label(self.wls_runner)}"
            )
        if self.production_dataset_mode:
            evidence_keys = sorted(
                key
                for key in _PRODUCTION_DECISION_EVIDENCE_KEYS
                if key in metrics and metrics.get(key) is not None
            )
            evidence_source = str(metrics.get("evidence_source") or "")
            missing: list[str] = []
            if str(metrics.get("state_id")) != state_id:
                missing.append("verification_state_id_unbound")
            if str(metrics.get("state_hash")) != str(state_payload["state_hash"]):
                missing.append("verification_state_hash_unbound")
            if not evidence_keys:
                missing.append("decision_metrics_missing")
            if not evidence_source:
                missing.append("evidence_source_missing")
            elif any(token in evidence_source.lower() for token in ("placeholder", "fallback")):
                missing.append("placeholder_evidence_prohibited")
            elif not _observable_provenance_source(evidence_source):
                missing.append("non_observable_evidence_source")
            if missing:
                return self._standard_output(
                    execution_status="failure",
                    error_code="insufficient_observable_evidence",
                    error_detail=",".join(missing),
                    state_mutated=False,
                    tool_metrics={
                        "state_id": state_id,
                        "state_hash": state_payload["state_hash"],
                        "evidence_source": evidence_source or None,
                        "evidence_keys": evidence_keys,
                    },
                )
            metrics["evidence_sufficiency"] = {
                "sufficient": True,
                "evidence_keys": evidence_keys,
                "evidence_source": evidence_source,
            }

        assessment: CandidateAssessment | None = None
        if state_id == self.current_candidate_id:
            parent_id = state_payload.get("parent_state_id")
            parent_payload = self.store.get_state(str(parent_id)) if parent_id else {}
            source_action = state_payload.get("source_action") or action
            parent_score = self._latest_successful_wls_score(str(parent_id or ""))
            candidate_score = metrics.get("remaining_anomaly_score")
            try:
                if parent_score is not None and candidate_score is not None:
                    denominator = max(abs(float(parent_score)), 1e-12)
                    metrics.setdefault(
                        "global_progress",
                        (float(parent_score) - float(candidate_score)) / denominator,
                    )
                    metrics.setdefault("parent_anomaly_score", float(parent_score))
            except (TypeError, ValueError, OverflowError):
                pass
            parent_target_metric = self._parent_target_metric_value(
                str(parent_id or ""), source_action
            )
            candidate_target_metric = metrics.get("target_metric_value")
            target_metric_threshold = metrics.get("target_metric_threshold")
            try:
                if parent_target_metric is not None and candidate_target_metric is not None:
                    parent_target_metric = float(parent_target_metric)
                    candidate_target_metric = float(candidate_target_metric)
                    target_metric_threshold = float(target_metric_threshold or 0.0)
                    denominator = max(
                        abs(parent_target_metric),
                        abs(target_metric_threshold),
                        1e-12,
                    )
                    metrics["parent_target_metric_value"] = parent_target_metric
                    metrics["target_progress"] = (
                        parent_target_metric - candidate_target_metric
                    ) / denominator
            except (TypeError, ValueError, OverflowError):
                pass
            source_tool = safe_normalize_action(source_action)["tool"]
            if source_tool == CORRECT_MEASUREMENTS:
                if self._accepted_branch_rows():
                    parent_signatures = [
                        str(item)
                        for item in self.context_flags.get("unresolved_signatures") or []
                    ]
                    metrics["sequential_cross_family_measurement"] = True
                    metrics["measurement_evidence_dominant"] = bool(
                        matching_evidence_codes(
                            parent_signatures, "wls_residual_outlier_dominant"
                        )
                    )
                    target_colocated = self._measurement_target_branch_colocated(
                        str(parent_id or ""), source_action
                    )
                    if target_colocated is not None:
                        metrics["measurement_target_branch_colocated"] = bool(
                            target_colocated
                        )
                        metrics["independent_measurement_target"] = not bool(
                            target_colocated
                        )
                cluster = self._measurement_target_cluster(source_action)
                if cluster is not None:
                    metrics["measurement_target_channel"] = cluster[0]
                    metrics["measurement_target_cluster_size"] = cluster[1]
                accepted_count, shared_channel = (
                    self._accepted_measurement_channel_evidence()
                )
                metrics["accepted_measurement_target_count"] = accepted_count
                if shared_channel is not None:
                    metrics["accepted_measurement_shared_channel"] = shared_channel
                rank_one = self._measurement_target_rank_one(source_action)
                if rank_one is not None:
                    metrics["measurement_target_rank_one"] = rank_one
                metrics["measurement_branch_routes_closed"] = (
                    self._measurement_branch_routes_closed()
                )
            truth = self.get_oracle_state().truth_dict() if self._oracle_payload.get("truth_complete") else None
            assessment = self.candidate_quality_oracle.label_candidate(
                parent_state=parent_payload,
                source_action=source_action,
                candidate_state=state_payload,
                verification_output=metrics,
                hidden_truth=truth,
            )
            if self.production_dataset_mode:
                partial_global_progress_floor = (
                    self.candidate_quality_oracle.min_branch_partial_global_progress
                    if source_tool in {CORRECT_PARAMETERS, CORRECT_TOPOLOGY}
                    else self.candidate_quality_oracle.min_partial_global_progress
                )
                decision_missing = self._target_decision_evidence_missing(
                    metrics,
                    assessment.disposition.value,
                    min_partial_global_progress=partial_global_progress_floor,
                    min_topology_structural_global_progress=(
                        self.candidate_quality_oracle.min_topology_structural_global_progress
                    ),
                    max_branch_target_threshold_ratio=(
                        self.candidate_quality_oracle.max_branch_target_threshold_ratio
                    ),
                )
                if decision_missing:
                    return self._standard_output(
                        execution_status="failure",
                        error_code="insufficient_observable_evidence",
                        error_detail=",".join(decision_missing),
                        state_mutated=False,
                        candidate_state_id=self.current_candidate_id,
                        tool_metrics={
                            "state_id": state_id,
                            "state_hash": state_payload["state_hash"],
                            "evidence_source": metrics.get("evidence_source"),
                            "evidence_keys": sorted(
                                key
                                for key in _PRODUCTION_DECISION_EVIDENCE_KEYS
                                if key in metrics and metrics.get(key) is not None
                            ),
                        },
                    )
            self.store.mark_verified(
                state_id,
                metrics,
                assessment.disposition.value,
                assessment.as_dict(),
            )
            metrics["candidate_disposition"] = assessment.disposition.value
            metrics["candidate_assessment"] = assessment.as_dict()
        else:
            self._persist_observable_semantics(
                metrics,
                source=str(metrics.get("evidence_source") or "synthetic_placeholder"),
                replace_missing=False,
            )
        return self._standard_output(
            execution_status="success",
            state_mutated=assessment is not None,
            candidate_state_id=self.current_candidate_id,
            tool_metrics=metrics,
        )

    def _step_context(self, action: dict[str, Any]) -> dict[str, Any]:
        tool = action["tool"]
        provider = self.context_providers.get(tool)
        state = self.current_state()
        active_payload = self.store.get_state(str(state["active_state_id"]))
        active_payload["policy_observation"] = self.get_policy_observation().as_dict()
        metrics = (
            dict(provider(copy.deepcopy(active_payload)))
            if provider
            else {"context_tool": tool, "evidence_source": "synthetic_placeholder"}
        )
        status = metrics.pop("execution_status", "success")
        if status != "success":
            return self._standard_output(
                execution_status="failure",
                error_code=str(metrics.pop("error_code", "context_failure")),
                error_detail=metrics.pop("error_detail", None),
                state_mutated=False,
                tool_metrics=metrics,
            )
        if provider is not None:
            metrics.setdefault(
                "evidence_source", f"configured_provider:{_provider_label(provider)}"
            )
            metrics.setdefault("state_id", str(state["active_state_id"]))
            metrics.setdefault("state_hash", active_payload["state_hash"])
        context_bound = (
            str(metrics.get("state_id")) == str(state["active_state_id"])
            and str(metrics.get("state_hash")) == str(active_payload["state_hash"])
        )
        if self.production_dataset_mode and (
            not context_bound or not self._provider_metrics_are_substantive(metrics)
        ):
            return self._standard_output(
                execution_status="failure",
                error_code="insufficient_observable_evidence",
                error_detail=(
                    f"{tool}_provider_evidence_unbound"
                    if not context_bound
                    else f"{tool}_provider_returned_no_evidence"
                ),
                state_mutated=False,
                tool_metrics={"evidence_source": metrics.get("evidence_source")},
            )
        context_state_id = str(action["arguments"].get("state_id") or state["active_state_id"])
        family = tool.removeprefix("get_").removesuffix("_context")
        correction_tool = {
            "measurement": CORRECT_MEASUREMENTS,
            "parameter": CORRECT_PARAMETERS,
            "topology": CORRECT_TOPOLOGY,
        }[family]
        supported: list[dict[str, Any]] = []
        raw_supported = metrics.get("supported_corrections")
        invalid_supported_contract = False
        if isinstance(raw_supported, (list, tuple)):
            for item in raw_supported:
                if not isinstance(item, Mapping):
                    invalid_supported_contract = True
                    continue
                normalized_supported = safe_normalize_action(item)
                if normalized_supported["tool"] != correction_tool:
                    invalid_supported_contract = True
                    continue
                target_state_id = normalized_supported["arguments"].get("state_id")
                if (
                    target_state_id is None
                    or str(target_state_id) != context_state_id
                ):
                    invalid_supported_contract = True
                    continue
                supported.append(normalized_supported)
        elif raw_supported is not None:
            invalid_supported_contract = True
        if invalid_supported_contract:
            return self._standard_output(
                execution_status="failure",
                error_code="insufficient_observable_evidence",
                error_detail=f"{tool}_supported_correction_contract_invalid",
                state_mutated=False,
                tool_metrics={
                    "state_id": context_state_id,
                    "state_hash": active_payload["state_hash"],
                    "evidence_source": metrics.get("evidence_source"),
                },
            )
        route_status_present = "route_status" in metrics
        route_contract_required = family in {"parameter", "topology"} and (
            self.production_dataset_mode or route_status_present
        )
        if route_contract_required and not _branch_route_contract_valid(
            metrics.get("route_status"), supported
        ):
            return self._standard_output(
                execution_status="failure",
                error_code="insufficient_observable_evidence",
                error_detail=f"{tool}_route_contract_invalid",
                state_mutated=False,
                tool_metrics={
                    "state_id": context_state_id,
                    "state_hash": active_payload["state_hash"],
                    "evidence_source": metrics.get("evidence_source"),
                },
            )
        raw_terminal_targets = metrics.get(
            "verified_terminal_measurement_closure_targets"
        )
        raw_terminal_evidence = metrics.get(
            "verified_terminal_measurement_closure_evidence"
        )
        terminal_targets_claimed = (
            bool(raw_terminal_targets)
            if isinstance(raw_terminal_targets, (list, tuple))
            else raw_terminal_targets is not None
        )
        terminal_closure_claimed = family == "measurement" and (
            terminal_targets_claimed
            or (
                isinstance(raw_terminal_evidence, Mapping)
                and raw_terminal_evidence.get("eligible") is True
            )
        )
        accepted_targets = accepted_measurement_indices(
            self.get_policy_observation()
        )
        if terminal_closure_claimed and (
            verified_terminal_measurement_closure_action(
                metrics,
                active_id=context_state_id,
                active_state_hash=active_payload["state_hash"],
                accepted_targets=accepted_targets,
            )
            is None
            or not _terminal_closure_branch_screening_valid(
                metrics.get("branch_route_screening"),
                state_id=context_state_id,
                state_hash=str(active_payload["state_hash"]),
            )
        ):
            return self._standard_output(
                execution_status="failure",
                error_code="insufficient_observable_evidence",
                error_detail=f"{tool}_terminal_closure_contract_invalid",
                state_mutated=False,
                tool_metrics={
                    "state_id": context_state_id,
                    "state_hash": active_payload["state_hash"],
                    "evidence_source": metrics.get("evidence_source"),
                },
            )
        self.context_flags[f"has_fresh_{family}_context"] = True
        self.context_flags[f"{family}_context_state_id"] = context_state_id
        context_evidence = dict(self.context_flags.get("fresh_context_evidence") or {})
        durable_metrics = {
            "context_tool": tool,
            "context_binding": "direct_context",
            "state_id": context_state_id,
            "state_hash": active_payload["state_hash"],
            "evidence_source": metrics.get("evidence_source"),
            "supported_corrections": supported,
        }
        for key in (
            "accepted_target_refinement",
            "verified_terminal_measurement_closure_targets",
            "verified_terminal_measurement_closure_evidence",
            "physical_vm_joint_targets",
            "measurement_findings",
            "parameter_findings",
            "parameter_ranking_contract",
            "parameter_ranking_distinct_lines",
            "parameter_ranking_top_abs_lambda",
            "parameter_ranking_runner_up_abs_lambda",
            "parameter_ranking_dominance_ratio",
            "parameter_ranking_dominance_threshold",
            "parameter_ranking_singleton",
            "parameter_ranking_dominant",
            "topology_findings",
            "topology_candidate_screening",
            "route_status",
            "route_status_reason",
        ):
            if key in metrics:
                durable_metrics[key] = policy_safe_copy(metrics[key])
        context_evidence[family] = policy_safe_copy(durable_metrics)
        # A post-commit measurement context may bundle the two independently
        # observable branch inventories for this exact active state.  Accept
        # only fully bound, correctly typed provider contracts; an omitted or
        # malformed family remains fresh=False so recovery fails closed and
        # the dedicated context action is still required.
        bundled = metrics.get("branch_route_screening")
        if family == "measurement" and isinstance(bundled, Mapping):
            bundled_contracts = {
                "parameter": (GET_PARAMETER_CONTEXT, CORRECT_PARAMETERS),
                "topology": (GET_TOPOLOGY_CONTEXT, CORRECT_TOPOLOGY),
            }
            for bundled_family, (bundled_tool, bundled_correction) in (
                bundled_contracts.items()
            ):
                raw_evidence = bundled.get(bundled_family)
                if not isinstance(raw_evidence, Mapping):
                    continue
                if (
                    raw_evidence.get("context_tool") != bundled_tool
                    or str(raw_evidence.get("state_id") or "") != context_state_id
                    or str(raw_evidence.get("state_hash") or "")
                    != str(active_payload["state_hash"])
                    or not _observable_provenance_source(
                        str(raw_evidence.get("evidence_source") or "")
                    )
                ):
                    continue
                raw_inventory = raw_evidence.get("supported_corrections")
                if not isinstance(raw_inventory, (list, tuple)):
                    continue
                bundled_supported: list[dict[str, Any]] = []
                invalid_bundled_contract = False
                for item in raw_inventory:
                    if not isinstance(item, Mapping):
                        invalid_bundled_contract = True
                        continue
                    normalized = safe_normalize_action(item)
                    if (
                        normalized["tool"] != bundled_correction
                        or str(normalized["arguments"].get("state_id") or "")
                        != context_state_id
                    ):
                        invalid_bundled_contract = True
                        continue
                    bundled_supported.append(normalized)
                if invalid_bundled_contract:
                    continue
                route_status = raw_evidence.get("route_status")
                if not _branch_route_contract_valid(
                    route_status, bundled_supported
                ):
                    continue
                self.context_flags[f"has_fresh_{bundled_family}_context"] = True
                self.context_flags[f"{bundled_family}_context_state_id"] = (
                    context_state_id
                )
                bundled_durable = {
                    "context_tool": bundled_tool,
                    "context_binding": (
                        f"branch_route_screening.{bundled_family}"
                    ),
                    "bundled_by_context_tool": tool,
                    "state_id": context_state_id,
                    "state_hash": active_payload["state_hash"],
                    "evidence_source": raw_evidence.get("evidence_source"),
                    "supported_corrections": bundled_supported,
                    "route_status": str(route_status),
                    "route_status_reason": raw_evidence.get(
                        "route_status_reason"
                    ),
                }
                for key in (
                    "parameter_findings",
                    "parameter_ranking_contract",
                    "parameter_ranking_distinct_lines",
                    "parameter_ranking_top_abs_lambda",
                    "parameter_ranking_runner_up_abs_lambda",
                    "parameter_ranking_dominance_ratio",
                    "parameter_ranking_dominance_threshold",
                    "parameter_ranking_singleton",
                    "parameter_ranking_dominant",
                    "topology_findings",
                    "topology_candidate_screening",
                ):
                    if key in raw_evidence:
                        bundled_durable[key] = policy_safe_copy(raw_evidence[key])
                context_evidence[bundled_family] = policy_safe_copy(
                    bundled_durable
                )
        self.context_flags["fresh_context_evidence"] = context_evidence
        self._persist_observable_semantics(
            metrics,
            source=f"context_provider:{tool}",
            replace_missing=False,
        )
        return self._standard_output(execution_status="success", state_mutated=False, tool_metrics=metrics)

    def _standard_output(
        self,
        *,
        execution_status: str,
        state_mutated: bool,
        error_code: str | None = None,
        error_detail: str | None = None,
        active_state_id: str | None = None,
        candidate_state_id: str | None = None,
        tool_metrics: Mapping[str, Any] | None = None,
        valid_next_actions: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        return {
            "execution_status": execution_status,
            "error_code": error_code,
            "error_detail": error_detail,
            "state_mutated": bool(state_mutated),
            "active_state_id": active_state_id or self.store.active_state_id,
            "candidate_state_id": candidate_state_id if candidate_state_id is not None else self.current_candidate_id,
            "tool_metrics": policy_safe_copy(dict(tool_metrics or {})),
            "valid_next_actions": copy.deepcopy(list(valid_next_actions or [])),
        }

    def _record_transition(
        self,
        action: dict[str, Any],
        output: dict[str, Any],
        *,
        source_state_id: str | None = None,
        source_candidate_id: str | None = None,
    ) -> None:
        self.context_flags["last_tool"] = action["tool"]
        self.context_flags["last_tool_status"] = output["execution_status"]
        self.context_flags["last_tool_output"] = policy_safe_copy(output)
        self.context_flags.setdefault("tried_action_signatures", []).append(action_signature(action))
        self.history.append(
            {
                "state_id": source_state_id or self.store.active_state_id,
                "candidate_state_id": source_candidate_id,
                "action": policy_safe_copy(action),
                "tool_output": policy_safe_copy(output),
            }
        )

    def _record_anomaly_explanation(
        self, tool: str, target_id: str, metrics: Mapping[str, Any]
    ) -> None:
        """Record a diagnostic finding that accounts for anomaly signatures.

        Providers declare the explanation (family plus finding detail); the
        environment binds it to the unresolved signatures whose observable
        markers match that family.  Fully explained signatures satisfy the
        terminal condition without a physical correction.
        """
        explanation = metrics.get("anomaly_explanation")
        if not isinstance(explanation, Mapping):
            return
        family = str(explanation.get("family") or "")
        markers = ANOMALY_FAMILY_MARKERS.get(family)
        if not markers:
            return
        unresolved = list(self.current_state().get("unresolved_signatures") or [])
        explained = matching_evidence_codes(unresolved, *markers)
        record = policy_safe_copy(
            {
                "tool": tool,
                "state_id": target_id,
                "family": family,
                "kind": explanation.get("kind"),
                "detail": explanation.get("detail") or {},
                "evidence_source": metrics.get("evidence_source"),
                "explained_signatures": explained,
            }
        )
        self.context_flags.setdefault("explained_anomalies", []).append(record)

    def _invalidate_context_flags(self) -> None:
        for family in ("measurement", "parameter", "topology"):
            self.context_flags[f"has_fresh_{family}_context"] = False
            self.context_flags[f"{family}_context_state_id"] = None
        self.context_flags["fresh_context_evidence"] = {}
        # Rejected candidates are evidence about alternatives to the old
        # active parent.  They remain durable across a same-state rollback, but
        # become stale as soon as another candidate is committed.
        self.context_flags["rejected_hypotheses"] = []

    def _set_semantic_provenance(self, field: str, source: str) -> None:
        if field not in _SEMANTIC_POLICY_FIELDS:
            raise ValueError(f"Unknown semantic policy field: {field}")
        self.context_flags.setdefault("semantic_field_provenance", {})[field] = str(source)

    def _persist_observable_semantics(
        self,
        metrics: Mapping[str, Any],
        *,
        source: str,
        replace_missing: bool,
    ) -> None:
        if metrics.get("unresolved_signatures") is not None:
            raw_signatures = metrics.get("unresolved_signatures")
            if isinstance(raw_signatures, (list, tuple, set, frozenset)):
                signatures = [str(value) for value in raw_signatures]
            else:
                signatures = [str(raw_signatures)] if raw_signatures else []
            self.context_flags["unresolved_signatures"] = signatures
            self._set_semantic_provenance("unresolved_signatures", source)
        elif replace_missing:
            self.context_flags["unresolved_signatures"] = []
            self._set_semantic_provenance(
                "unresolved_signatures", "controller_default_after_commit"
            )

        score = metrics.get("remaining_anomaly_score")
        if score is not None:
            self.context_flags["remaining_anomaly_score"] = score
            self._set_semantic_provenance("remaining_anomaly_score", source)
        elif replace_missing:
            self.context_flags["remaining_anomaly_score"] = None
            self._set_semantic_provenance(
                "remaining_anomaly_score", "controller_default_after_commit"
            )

        explicit_resolution = metrics.get("no_material_anomaly_remaining")
        if explicit_resolution is None:
            explicit_resolution = metrics.get("post_action_resolved")
        if explicit_resolution is None:
            threshold = metrics.get("anomaly_threshold", metrics.get("chi_square_threshold"))
            if score is not None and threshold is not None:
                try:
                    explicit_resolution = float(score) < float(threshold)
                except (TypeError, ValueError):
                    explicit_resolution = None
        if explicit_resolution is not None:
            self.context_flags["no_material_anomaly_remaining"] = bool(explicit_resolution)
            self._set_semantic_provenance("no_material_anomaly_remaining", source)
        elif replace_missing:
            self.context_flags["no_material_anomaly_remaining"] = False
            self._set_semantic_provenance(
                "no_material_anomaly_remaining", "controller_default_after_commit"
            )

        if not (
            self.production_dataset_mode
            and self.context_flags.get("accepted_corrections")
        ):
            return
        try:
            statistically_quiescent = bool(
                self.context_flags.get("no_material_anomaly_remaining")
                or (
                    self.context_flags.get("remaining_anomaly_score") is not None
                    and float(self.context_flags["remaining_anomaly_score"])
                    < float(self.process_oracle.anomaly_threshold)
                )
            )
        except (TypeError, ValueError, OverflowError):
            statistically_quiescent = bool(
                self.context_flags.get("no_material_anomaly_remaining")
            )
        if not statistically_quiescent:
            return

        # Candidate verification and active-state release finality are
        # intentionally separate.  This obligation is derived entirely from
        # public controller state (an accepted correction plus quiescent WLS),
        # never from the remaining hidden fault count.
        signatures = list(self.context_flags.get("unresolved_signatures") or [])
        if POST_CORRECTION_CONFIRMATION_SIGNATURE not in signatures:
            signatures.append(POST_CORRECTION_CONFIRMATION_SIGNATURE)
        confirmation_source = (
            "controller_default:post_correction_resolution_confirmation_required"
        )
        self.context_flags["unresolved_signatures"] = signatures
        self._set_semantic_provenance(
            "unresolved_signatures", confirmation_source
        )
        self.context_flags["no_material_anomaly_remaining"] = False
        self._set_semantic_provenance(
            "no_material_anomaly_remaining", confirmation_source
        )

    @staticmethod
    def _provider_metrics_are_substantive(metrics: Mapping[str, Any]) -> bool:
        source = str(metrics.get("evidence_source") or "")
        if not _observable_provenance_source(source):
            return False
        excluded = {
            "execution_status",
            "error_code",
            "error_detail",
            "evidence_source",
            "state_id",
            "state_hash",
            "context_tool",
            "evidence_requested",
        }
        return any(key not in excluded and value is not None for key, value in metrics.items())

    @staticmethod
    def _target_decision_evidence_missing(
        metrics: Mapping[str, Any],
        disposition: str,
        *,
        min_partial_global_progress: float = 0.20,
        min_topology_structural_global_progress: float = 0.95,
        max_branch_target_threshold_ratio: float = 1.25,
    ) -> list[str]:
        """Require observable metrics that distinguish the proposed disposition."""
        disposition = str(disposition or "")

        def number(key: str) -> float | None:
            value = metrics.get(key)
            try:
                return None if value is None else float(value)
            except (TypeError, ValueError):
                return None

        target_progress = number("target_progress")
        global_progress = number("global_progress")
        topology_multiplier = number("topology_target_branch_multiplier")
        topology_multiplier_threshold = number(
            "topology_target_branch_multiplier_threshold"
        )
        score = number("remaining_anomaly_score")
        threshold = number("anomaly_threshold")
        if threshold is None:
            threshold = number("chi_square_threshold")
        resolved = any(
            metrics.get(key) is True
            for key in ("globally_resolved", "post_action_resolved", "no_material_anomaly_remaining")
        ) or (
            score is not None and threshold is not None and score < threshold
        )
        anomaly_remains = any(
            metrics.get(key) is False
            for key in (
                "globally_resolved",
                "post_action_resolved",
                "no_material_anomaly_remaining",
            )
        ) or (
            score is not None and threshold is not None and score >= threshold
        )
        physical_ok = metrics.get("physical_constraints_ok") is True or (
            any(
                metrics.get(key) is True
                # Generic ``converged`` is commonly emitted by WLS/state-
                # estimation providers.  It proves optimizer completion, not
                # AC power-flow feasibility or constraint safety.
                for key in ("power_flow_converged", "topology_feasible")
            )
            and number("new_constraint_violations") == 0
        )

        if (
            disposition
            in {
                CandidateDisposition.ACCEPT_FINAL.value,
                CandidateDisposition.ACCEPT_PARTIAL.value,
            }
            and metrics.get("sequential_cross_family_measurement") is True
            and metrics.get("measurement_evidence_dominant") is not True
            and metrics.get("measurement_target_branch_colocated") is not False
        ):
            return ["independent_measurement_target_evidence_missing"]

        if disposition == CandidateDisposition.ACCEPT_FINAL.value:
            if not physical_ok:
                return ["physical_constraint_evidence_missing"]
            if not resolved:
                return ["final_resolution_evidence_missing"]
        elif disposition == CandidateDisposition.ACCEPT_PARTIAL.value:
            progress = metrics.get("target_fixed") is True or (
                target_progress is not None and target_progress > 0
            )
            if not physical_ok:
                return ["physical_constraint_evidence_missing"]
            if not anomaly_remains or not progress:
                return ["partial_progress_evidence_missing"]
        elif disposition == CandidateDisposition.REJECT.value:
            violations = number("new_constraint_violations")
            rejected = (
                metrics.get("target_fixed") is False
                or metrics.get("healthy_component_modified") is True
                or metrics.get("collateral_damage") is True
                or metrics.get("physical_constraints_ok") is False
                or metrics.get("converged") is False
                or (
                    metrics.get("sequential_cross_family_measurement") is True
                    and metrics.get("measurement_evidence_dominant") is not True
                    and metrics.get("measurement_target_branch_colocated") is True
                )
                or (target_progress is not None and target_progress <= 0)
                or (global_progress is not None and global_progress < 0)
                or (
                    metrics.get("target_fixed") is True
                    and global_progress is not None
                    and global_progress < float(min_partial_global_progress)
                )
                or (
                    metrics.get("target_fixed") is True
                    and metrics.get("topology_target_status_matches_requested") is True
                    and topology_multiplier is not None
                    and topology_multiplier_threshold is not None
                    and topology_multiplier_threshold > 0.0
                    and topology_multiplier
                    > float(max_branch_target_threshold_ratio)
                    * topology_multiplier_threshold
                    and global_progress is not None
                    and global_progress
                    < float(min_topology_structural_global_progress)
                )
                or (violations is not None and violations > 0)
            )
            if not rejected:
                return ["rejection_evidence_missing"]
        return []

    @staticmethod
    def _accepted_candidate_record(candidate_payload: Mapping[str, Any]) -> dict[str, Any]:
        return policy_safe_copy(
            {
                "candidate_state_id": candidate_payload.get("state_id"),
                "candidate_parent_id": candidate_payload.get("parent_state_id"),
                "source_action": candidate_payload.get("source_action"),
            }
        )

    def _remember_rejected_hypothesis(self, candidate_payload: Mapping[str, Any]) -> None:
        source_action = candidate_payload.get("source_action") or {}
        raw_verification = candidate_payload.get("verification_output")
        verification_summary: dict[str, Any] = {}
        if isinstance(raw_verification, Mapping):
            for key in (
                "state_id",
                "evidence_source",
                "target_metric_value",
                "target_metric_threshold",
                "target_progress",
                "global_progress",
                "globally_resolved",
                "physical_constraints_ok",
                "physical_bound_violations",
            ):
                if key in raw_verification:
                    verification_summary[key] = policy_safe_copy(raw_verification[key])
        self.context_flags.setdefault("rejected_hypotheses", []).append(
            policy_safe_copy(
                {
                    "candidate_state_id": candidate_payload.get("state_id"),
                    "candidate_parent_id": candidate_payload.get("parent_state_id"),
                    "source_action": source_action,
                    "action_signature": action_signature(source_action) if source_action else None,
                    "verification_summary": verification_summary,
                }
            )
        )

    def _truth_after_commit(self, candidate_payload: Mapping[str, Any]) -> dict[str, Any]:
        next_payload = copy.deepcopy(self._oracle_payload)
        if not next_payload.get("truth_complete"):
            return next_payload
        source_action = candidate_payload.get("source_action") or {}
        family = correction_family(source_action)
        if family:
            key = f"true_{family}_errors"
            faults = list(next_payload.get(key) or [])
            if faults:
                parent_id = candidate_payload.get("parent_state_id")
                parent_payload = self.store.get_state(str(parent_id)) if parent_id else {}
                truth = copy.deepcopy(next_payload)
                truth["truth_complete"] = True
                matcher = getattr(
                    self.candidate_quality_oracle,
                    "matched_fault_indices",
                    None,
                )
                if not callable(matcher):
                    matcher = CandidateQualityOracle(
                        mode="synthetic"
                    ).matched_fault_indices
                matched_indices = matcher(
                    source_action,
                    truth,
                    parent_state=parent_payload,
                    candidate_state=candidate_payload,
                )
                if matched_indices:
                    faults = [fault for index, fault in enumerate(faults) if index not in set(matched_indices)]
                next_payload[key] = faults
        remaining = []
        for key in ("true_measurement_errors", "true_parameter_errors", "true_topology_errors"):
            remaining.extend(copy.deepcopy(list(next_payload.get(key) or [])))
        next_payload["remaining_true_faults"] = remaining
        next_payload["remaining_true_fault_count"] = len(remaining)
        next_payload.pop("remaining_fault_count", None)
        return next_payload

    def failure_report(self, state: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return {"terminal": self.terminal, "state": dict(state or self.current_state()), "history": list(self.history)}

    def final_report(self, state: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return {"terminal": True, "state": dict(state or self.current_state()), "history": list(self.history)}
