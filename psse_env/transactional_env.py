from __future__ import annotations

import copy
import types
from collections.abc import Callable, Iterable
from typing import Any, Mapping

from .actions import (
    ASK_FOR_MORE_EVIDENCE,
    COMMIT_STATE,
    CONTEXT_TOOLS,
    CORRECTION_TOOLS,
    FINALIZE_DIAGNOSIS,
    ROLLBACK_STATE,
    RUN_ALTERNATIVE_TEST,
    RUN_WLS,
    VERIFY_CANDIDATE,
    action_signature,
    safe_normalize_action,
)
from .oracle.candidate_quality import CandidateAssessment, CandidateDisposition, CandidateQualityOracle
from .oracle.process_validity import ProcessValidityOracle
from .state_store import (
    CandidateLifecycle,
    FORBIDDEN_POLICY_KEYS,
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
        "global_progress",
        "remaining_anomaly_score",
        "remaining_fault_count",
        "globally_resolved",
        "post_action_resolved",
        "physical_constraints_ok",
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
        max_steps: int = 12,
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
        self._episode_counter = 0
        self._oracle_payload: dict[str, Any] = {}
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
            "unresolved_signatures",
        }
        truth_was_supplied = bool(scenario.get("hidden_truth"))
        for key in truth_keys:
            if key in scenario:
                oracle_payload[key] = copy.deepcopy(scenario[key])
                truth_was_supplied = True
            elif key in raw_metadata:
                oracle_payload[key] = copy.deepcopy(raw_metadata[key])
                truth_was_supplied = True
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
        return self.current_state()

    def current_state(self) -> dict[str, Any]:
        return self.store.decision_summary(
            candidate_state_id=self.current_candidate_id,
            remaining_budget=max(self.max_steps - len(self.history), 0),
            context_flags=self.context_flags,
        )

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
            requires_measurement_context=bool(summary.get("requires_measurement_context")),
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
        # Synthetic terminal eligibility remains an oracle-side process fact;
        # it is deliberately absent from PolicyObservation.
        validity_state["oracle_terminal_eligible"] = bool(
            self._oracle_payload.get("oracle_terminal_eligible", False)
        )
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
            if before_hash != self.store.episode_hash():
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
            if accept_final:
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
            return self._standard_output(
                execution_status="success", state_mutated=False, tool_metrics={"finalized": True}
            )
        if tool in {ASK_FOR_MORE_EVIDENCE, RUN_ALTERNATIVE_TEST}:
            provider = self.evidence_providers.get(tool)
            target_id = str(args.get("state_id") or self.current_candidate_id or self.store.active_state_id)
            provider_state = self.store.get_state(target_id)
            provider_state["policy_observation"] = self.get_policy_observation().as_dict()
            if provider is None and self.production_dataset_mode:
                return self.record_noop_failure(
                    action=action,
                    error_code="production_provider_missing",
                    error_detail=tool,
                    valid_next_actions=[],
                )
            metrics = dict(provider(copy.deepcopy(provider_state))) if provider else {
                "evidence_requested": tool,
                "evidence_source": "synthetic_placeholder",
            }
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
        missing.extend(
            self._target_decision_evidence_missing(
                verification,
                str(candidate.get("candidate_disposition") or ""),
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
            observable_recovery_route = family == "parameter" and bool(
                state.get("rejected_hypotheses") or state.get("accepted_corrections")
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
        if tool == FINALIZE_DIAGNOSIS:
            score = state.get("remaining_anomaly_score")
            try:
                score_resolved = (
                    score is not None
                    and float(score) < float(self.process_oracle.anomaly_threshold)
                )
            except (TypeError, ValueError):
                score_resolved = False
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

    def _supported_correction_signatures(self, context_family: str) -> set[str]:
        context_tool = f"get_{context_family}_context"
        for event in reversed(self.history):
            if not isinstance(event, Mapping):
                continue
            event_action = event.get("action")
            if safe_normalize_action(event_action or {})["tool"] != context_tool:
                continue
            output = event.get("tool_output")
            if not isinstance(output, Mapping):
                return set()
            metrics = output.get("tool_metrics")
            if not isinstance(metrics, Mapping):
                return set()
            raw_actions = metrics.get("supported_corrections")
            if not isinstance(raw_actions, (list, tuple)):
                return set()
            signatures: set[str] = set()
            for raw_action in raw_actions:
                if not isinstance(raw_action, Mapping):
                    continue
                normalized = safe_normalize_action(raw_action)
                if normalized["tool"] in CORRECTION_TOOLS:
                    signatures.add(action_signature(normalized))
            return signatures
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
                return self._standard_output(
                    execution_status="failure",
                    error_code=str(result.pop("error_code", "correction_execution_failure")),
                    error_detail=result.pop("error_detail", None),
                    state_mutated=False,
                    tool_metrics=policy_safe_copy(result),
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
            metrics = dict(self.wls_runner(copy.deepcopy(state_payload)))
        runner_status = metrics.pop("execution_status", "success")
        if runner_status != "success":
            return self._standard_output(
                execution_status="failure",
                error_code=str(metrics.pop("error_code", "wls_failure")),
                error_detail=metrics.pop("error_detail", None),
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
            truth = self.get_oracle_state().truth_dict() if self._oracle_payload.get("truth_complete") else None
            assessment = self.candidate_quality_oracle.label_candidate(
                parent_state=parent_payload,
                source_action=source_action,
                candidate_state=state_payload,
                verification_output=metrics,
                hidden_truth=truth,
            )
            if self.production_dataset_mode:
                decision_missing = self._target_decision_evidence_missing(
                    metrics, assessment.disposition.value
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
        self.context_flags[f"has_fresh_{family}_context"] = True
        self.context_flags[f"{family}_context_state_id"] = context_state_id
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

    def _invalidate_context_flags(self) -> None:
        for family in ("measurement", "parameter", "topology"):
            self.context_flags[f"has_fresh_{family}_context"] = False
            self.context_flags[f"{family}_context_state_id"] = None

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
        metrics: Mapping[str, Any], disposition: str
    ) -> list[str]:
        """Require observable metrics that distinguish the proposed disposition."""
        disposition = str(disposition or "")

        def number(key: str) -> float | None:
            value = metrics.get(key)
            try:
                return None if value is None else float(value)
            except (TypeError, ValueError):
                return None

        remaining = number("remaining_fault_count")
        target_progress = number("target_progress")
        global_progress = number("global_progress")
        score = number("remaining_anomaly_score")
        threshold = number("anomaly_threshold")
        if threshold is None:
            threshold = number("chi_square_threshold")
        resolved = any(
            metrics.get(key) is True
            for key in ("globally_resolved", "post_action_resolved", "no_material_anomaly_remaining")
        ) or (remaining == 0) or (
            score is not None and threshold is not None and score < threshold
        )

        if disposition == CandidateDisposition.ACCEPT_FINAL.value:
            if not resolved:
                return ["final_resolution_evidence_missing"]
        elif disposition == CandidateDisposition.ACCEPT_PARTIAL.value:
            progress = metrics.get("target_fixed") is True or (
                target_progress is not None and target_progress > 0
            )
            if remaining is None or remaining <= 0 or not progress:
                return ["partial_progress_evidence_missing"]
        elif disposition == CandidateDisposition.REJECT.value:
            violations = number("new_constraint_violations")
            rejected = (
                metrics.get("target_fixed") is False
                or metrics.get("healthy_component_modified") is True
                or metrics.get("collateral_damage") is True
                or metrics.get("physical_constraints_ok") is False
                or metrics.get("converged") is False
                or (target_progress is not None and target_progress <= 0)
                or (global_progress is not None and global_progress < 0)
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
        self.context_flags.setdefault("rejected_hypotheses", []).append(
            policy_safe_copy(
                {
                    "candidate_state_id": candidate_payload.get("state_id"),
                    "candidate_parent_id": candidate_payload.get("parent_state_id"),
                    "source_action": source_action,
                    "action_signature": action_signature(source_action) if source_action else None,
                }
            )
        )

    def _truth_after_commit(self, candidate_payload: Mapping[str, Any]) -> dict[str, Any]:
        next_payload = copy.deepcopy(self._oracle_payload)
        if not next_payload.get("truth_complete"):
            return next_payload
        source_action = candidate_payload.get("source_action") or {}
        family = {
            "correct_measurements": "measurement",
            "correct_parameters": "parameter",
            "correct_topology": "topology",
        }.get(source_action.get("tool"))
        if family:
            key = f"true_{family}_errors"
            faults = list(next_payload.get(key) or [])
            if faults:
                parent_id = candidate_payload.get("parent_state_id")
                parent_payload = self.store.get_state(str(parent_id)) if parent_id else {}
                truth = copy.deepcopy(next_payload)
                truth["truth_complete"] = True
                matcher = CandidateQualityOracle(mode="synthetic")
                matched_indices = matcher.matched_fault_indices(
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
