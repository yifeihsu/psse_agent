from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping

from .actions import action_signature


SYNTHETIC_TERMINAL_COMPATIBILITY_KEY = "_synthetic_terminal_compatibility"


FORBIDDEN_POLICY_KEYS = frozenset(
    {
        "suggested_actions",
        "oracle_action_hints",
        "true_measurement_errors",
        "true_parameter_errors",
        "true_topology_errors",
        "true_error_locations",
        "clean_case",
        "clean_measurements",
        "clean_parameter_values",
        "remaining_true_faults",
        "remaining_true_fault_count",
        "remaining_fault_count",
        "remaining_faults",
        "oracle_cost_to_go",
        "candidate_hidden_truth_labels",
        "candidate_disposition",
        "candidate_assessment",
        "progress_class",
        "target_fixed",
        "healthy_component_modified",
        "release_audit",
        "oracle_terminal_eligible",
        SYNTHETIC_TERMINAL_COMPATIBILITY_KEY,
        "hidden_truth",
    }
)


class StateStoreError(RuntimeError):
    """Raised when a transactional state operation is invalid."""


class CandidateLifecycle(str, Enum):
    NO_CANDIDATE = "NO_CANDIDATE"
    OPEN_UNVERIFIED_CANDIDATE = "OPEN_UNVERIFIED_CANDIDATE"
    # Policy-visible lifecycle state.  The accepted/rejected/inconclusive
    # variants below are oracle/store-only because synthetic truth may be the
    # sole reason they differ.
    VERIFIED_CANDIDATE = "VERIFIED_CANDIDATE"
    VERIFIED_ACCEPTED_CANDIDATE = "VERIFIED_ACCEPTED_CANDIDATE"
    VERIFIED_REJECTED_CANDIDATE = "VERIFIED_REJECTED_CANDIDATE"
    VERIFIED_INCONCLUSIVE_CANDIDATE = "VERIFIED_INCONCLUSIVE_CANDIDATE"


def _json_clone(value: Any) -> Any:
    return copy.deepcopy(value)


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def policy_safe_copy(value: Any) -> Any:
    """Recursively remove oracle-only keys from an observable payload."""
    if isinstance(value, Mapping):
        return {
            str(key): policy_safe_copy(item)
            for key, item in value.items()
            if str(key) not in FORBIDDEN_POLICY_KEYS
        }
    if isinstance(value, list):
        return [policy_safe_copy(item) for item in value]
    if isinstance(value, tuple):
        return [policy_safe_copy(item) for item in value]
    return _json_clone(value)


def find_forbidden_policy_paths(value: Any, prefix: str = "$") -> list[str]:
    """Return JSON-style paths containing oracle-only policy keys."""
    paths: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            path = f"{prefix}.{key_text}"
            if key_text in FORBIDDEN_POLICY_KEYS:
                paths.append(path)
            paths.extend(find_forbidden_policy_paths(item, path))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            paths.extend(find_forbidden_policy_paths(item, f"{prefix}[{index}]"))
    return paths


@dataclass(frozen=True)
class PolicyObservation:
    active_state_id: str
    candidate_state_id: str | None = None
    candidate_status: str | None = None
    last_tool: str | None = None
    last_tool_status: str | None = None
    last_tool_output: dict[str, Any] = field(default_factory=dict)
    last_verification: dict[str, Any] = field(default_factory=dict)
    accepted_corrections: list[dict[str, Any]] = field(default_factory=list)
    # Diagnostic findings that account for observable anomaly signatures
    # without a physical correction (localized harmonic source, estimated
    # HIF).  Fully explained signatures satisfy the terminal condition.
    explained_anomalies: list[dict[str, Any]] = field(default_factory=list)
    rejected_hypotheses: list[dict[str, Any]] = field(default_factory=list)
    unresolved_signatures: list[str] = field(default_factory=list)
    tried_action_signatures: list[str] = field(default_factory=list)
    remaining_budget: int = 0
    history_window: list[dict[str, Any]] = field(default_factory=list)
    episode_id: str | None = None
    candidate_parent_id: str | None = None
    candidate_lifecycle: str = CandidateLifecycle.NO_CANDIDATE.value
    candidate_committed: bool = False
    has_open_candidate: bool = False
    has_unverified_candidate: bool = False
    has_verified_candidate: bool = False
    remaining_anomaly_score: float | None = None
    no_material_anomaly_remaining: bool = False
    has_fresh_measurement_context: bool = False
    has_fresh_parameter_context: bool = False
    has_fresh_topology_context: bool = False
    measurement_context_state_id: str | None = None
    parameter_context_state_id: str | None = None
    topology_context_state_id: str | None = None
    # Compact provider evidence from context calls bound to the current active
    # state.  Context freshness can outlive the bounded history window after a
    # rejected candidate is rolled back, so retain the already-observed action
    # inventory and the small set of fields needed to evaluate conditional
    # retries.  The environment clears this ledger whenever the active state
    # changes.
    fresh_context_evidence: dict[str, dict[str, Any]] = field(default_factory=dict)
    requires_measurement_context: bool = False
    # Observable telemetry channels attached to the active state (harmonic
    # scans, HIF scan windows, repeated parameter scans, ...).  Knowing which
    # data streams exist is deployment-visible operator knowledge, not hidden
    # truth; it is the routing signal for specialized diagnostics.
    available_evidence: list[str] = field(default_factory=list)
    # Per-field origin for semantic summaries that could otherwise become
    # privileged shortcuts.  Production environments use stable values such
    # as controller_default, wls_runner:<adapter>,
    # context_provider:<tool>, observable_input, or deployment_sensor.
    semantic_field_provenance: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        forbidden = find_forbidden_policy_paths(payload)
        if forbidden:
            raise ValueError(f"PolicyObservation contains privileged fields: {', '.join(forbidden)}")
        return _json_clone(payload)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


@dataclass(frozen=True)
class OracleState:
    policy_observation: PolicyObservation
    clean_case: Any | None = None
    clean_measurements: list[Any] | None = None
    true_measurement_errors: list[Any] = field(default_factory=list)
    true_parameter_errors: list[Any] = field(default_factory=list)
    true_topology_errors: list[Any] = field(default_factory=list)
    remaining_true_faults: list[Any] = field(default_factory=list)
    oracle_action_hints: list[dict[str, Any]] = field(default_factory=list)
    hidden_truth: dict[str, Any] = field(default_factory=dict)
    candidate_disposition: str | None = None
    candidate_lifecycle: str = CandidateLifecycle.NO_CANDIDATE.value
    candidate_assessment: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy_observation": self.policy_observation.as_dict(),
            "clean_case": _json_clone(self.clean_case),
            "clean_measurements": _json_clone(self.clean_measurements),
            "true_measurement_errors": _json_clone(self.true_measurement_errors),
            "true_parameter_errors": _json_clone(self.true_parameter_errors),
            "true_topology_errors": _json_clone(self.true_topology_errors),
            "remaining_true_faults": _json_clone(self.remaining_true_faults),
            "oracle_action_hints": _json_clone(self.oracle_action_hints),
            "hidden_truth": _json_clone(self.hidden_truth),
            "candidate_disposition": self.candidate_disposition,
            "candidate_lifecycle": self.candidate_lifecycle,
            "candidate_assessment": _json_clone(self.candidate_assessment),
        }

    def truth_dict(self) -> dict[str, Any]:
        truth = dict(_json_clone(self.hidden_truth))
        truth.setdefault("clean_case", _json_clone(self.clean_case))
        truth.setdefault("clean_measurements", _json_clone(self.clean_measurements))
        truth.setdefault("true_measurement_errors", _json_clone(self.true_measurement_errors))
        truth.setdefault("true_parameter_errors", _json_clone(self.true_parameter_errors))
        truth.setdefault("true_topology_errors", _json_clone(self.true_topology_errors))
        if self.remaining_true_faults:
            truth.setdefault("remaining_true_faults", _json_clone(self.remaining_true_faults))
        return truth

    def get(self, key: str, default: Any = None) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        return self.policy_observation.get(key, default)

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        return self.policy_observation[key]


def _state_content_hash(case: Any, measurements: Any, metadata: Mapping[str, Any]) -> str:
    physical_metadata = {
        key: value
        for key, value in metadata.items()
        if key not in {"transactional_modifications", "scenario_id"}
    }
    payload = {"case": case, "measurements": measurements, "metadata": physical_metadata}
    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()


def _merge_mapping(base: Any, updates: Mapping[str, Any]) -> Any:
    if not isinstance(base, Mapping):
        return _json_clone(updates)
    merged = dict(_json_clone(base))
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _merge_mapping(merged[key], value)
        else:
            merged[key] = _json_clone(value)
    return merged


def _apply_measurement_updates(measurements: Any, updates: Any) -> Any:
    updated = _json_clone(measurements)
    if not isinstance(updated, list):
        return updated
    if isinstance(updates, Mapping):
        iterator = updates.items()
    elif isinstance(updates, list):
        iterator = []
        for item in updates:
            if not isinstance(item, Mapping):
                raise ValueError("measurement_updates list entries must be mappings.")
            iterator.append((item.get("index", item.get("index0")), item.get("value")))
    else:
        raise ValueError("measurement_updates must be a mapping or list.")
    for raw_index, value in iterator:
        index = int(raw_index)
        if index < 0 or index >= len(updated):
            raise IndexError(f"measurement update index {index} outside [0, {len(updated) - 1}]")
        updated[index] = _json_clone(value)
    return updated


def _branch_rows(case: Any) -> list[Any] | None:
    if isinstance(case, Mapping) and isinstance(case.get("branch"), list):
        return case["branch"]
    return None


def _resolve_branch_index(rows: list[Any], modification: Mapping[str, Any]) -> int | None:
    if modification.get("branch_row0") is not None:
        index = int(modification["branch_row0"])
        return index if 0 <= index < len(rows) else None
    if modification.get("line_index1") is not None:
        index = int(modification["line_index1"]) - 1
        return index if 0 <= index < len(rows) else None
    reference = modification.get("branch_id", modification.get("cb_name"))
    if reference is not None:
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                continue
            for key in ("branch_id", "id", "name", "cb_name"):
                if row.get(key) is not None and str(row[key]) == str(reference):
                    return index
        try:
            numeric = int(reference)
        except (TypeError, ValueError):
            return None
        if 0 <= numeric < len(rows):
            return numeric
        if 1 <= numeric <= len(rows):
            return numeric - 1
        return None
    if modification.get("line_index") is not None:
        index = int(modification["line_index"])
        if 0 <= index < len(rows):
            return index
        if 1 <= index <= len(rows):
            return index - 1
    return None


def _apply_parameter_update(case: Any, modification: Mapping[str, Any]) -> Any:
    updated = _json_clone(case)
    rows = _branch_rows(updated)
    if rows is None:
        return updated
    index = _resolve_branch_index(rows, modification)
    if index is None or not isinstance(rows[index], Mapping):
        return updated
    row = dict(rows[index])
    parameter = str(modification.get("parameter") or modification.get("field") or "x")
    value = _first_parameter_value(modification)
    if value is not None:
        row[parameter] = _json_clone(value)
    elif modification.get("multiplier") is not None and row.get(parameter) is not None:
        row[parameter] = float(row[parameter]) * float(modification["multiplier"])
    rows[index] = row
    return updated


def _first_parameter_value(modification: Mapping[str, Any]) -> Any:
    for key in ("value", "corrected_value", "new_value"):
        if modification.get(key) is not None:
            return modification[key]
    return None


def _apply_topology_update(case: Any, modification: Mapping[str, Any]) -> Any:
    updated = _json_clone(case)
    rows = _branch_rows(updated)
    if rows is None:
        return updated
    index = _resolve_branch_index(rows, modification)
    status = modification.get("status", modification.get("expected_status"))
    if index is None or status is None or not isinstance(rows[index], Mapping):
        return updated
    row = dict(rows[index])
    status_field = str(modification.get("status_field") or ("br_status" if "br_status" in row else "status"))
    row[status_field] = _json_clone(status)
    rows[index] = row
    return updated


def apply_modification(
    *, case: Any, measurements: Any, metadata: Mapping[str, Any], modification: Mapping[str, Any]
) -> tuple[Any, Any, dict[str, Any]]:
    """Apply a generic transactional modification without mutating the parent payload."""
    new_case = _json_clone(case)
    new_measurements = _json_clone(measurements)
    new_metadata = dict(_json_clone(metadata))
    if "case" in modification:
        new_case = _json_clone(modification["case"])
    if "case_updates" in modification:
        new_case = _merge_mapping(new_case, modification["case_updates"])
    if "measurements" in modification:
        new_measurements = _json_clone(modification["measurements"])
    if "measurement_updates" in modification:
        new_measurements = _apply_measurement_updates(new_measurements, modification["measurement_updates"])
    if "metadata_updates" in modification:
        new_metadata = _merge_mapping(new_metadata, modification["metadata_updates"])
    if any(
        key in modification
        for key in ("line_index", "line_index1", "branch_row0", "branch_id", "cb_name")
    ) and any(
        key in modification for key in ("value", "corrected_value", "new_value", "multiplier")
    ):
        new_case = _apply_parameter_update(new_case, modification)
    if any(
        key in modification
        for key in ("branch_id", "cb_name", "line_index", "line_index1", "branch_row0")
    ) and any(
        key in modification for key in ("status", "expected_status")
    ):
        new_case = _apply_topology_update(new_case, modification)
    new_metadata.setdefault("transactional_modifications", []).append(_json_clone(dict(modification)))
    return new_case, new_measurements, new_metadata


@dataclass
class PowerSystemState:
    episode_id: str
    state_id: str
    case: Any
    measurements: Any
    metadata: dict[str, Any] = field(default_factory=dict)
    parent_state_id: str | None = None
    status: str = "candidate"
    source_action: dict[str, Any] | None = None
    modification: dict[str, Any] | None = None
    verification_output: dict[str, Any] | None = None
    candidate_disposition: str | None = None
    candidate_assessment: dict[str, Any] | None = None
    active: bool = False
    depth: int = 0
    created_at_step: int = 0

    @property
    def state_hash(self) -> str:
        return _state_content_hash(self.case, self.measurements, self.metadata)

    @property
    def candidate_status(self) -> str:
        if self.status == "candidate":
            if self.candidate_disposition:
                return f"verified_{self.candidate_disposition.lower()}"
            return "unverified"
        return self.status

    @property
    def candidate_lifecycle(self) -> CandidateLifecycle:
        if self.status != "candidate":
            return CandidateLifecycle.NO_CANDIDATE
        if not self.candidate_disposition:
            return CandidateLifecycle.OPEN_UNVERIFIED_CANDIDATE
        if self.candidate_disposition in {"ACCEPT_PARTIAL", "ACCEPT_FINAL"}:
            return CandidateLifecycle.VERIFIED_ACCEPTED_CANDIDATE
        if self.candidate_disposition == "REJECT":
            return CandidateLifecycle.VERIFIED_REJECTED_CANDIDATE
        return CandidateLifecycle.VERIFIED_INCONCLUSIVE_CANDIDATE

    def as_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "state_id": self.state_id,
            "parent_state_id": self.parent_state_id,
            "case": _json_clone(self.case),
            "measurements": _json_clone(self.measurements),
            "metadata": _json_clone(self.metadata),
            "status": self.status,
            "candidate_status": self.candidate_status,
            "candidate_lifecycle": self.candidate_lifecycle.value,
            "source_action": _json_clone(self.source_action),
            "modification": _json_clone(self.modification),
            "verification_output": _json_clone(self.verification_output),
            "candidate_disposition": self.candidate_disposition,
            "candidate_assessment": _json_clone(self.candidate_assessment),
            "active": self.active,
            "state_hash": self.state_hash,
            "depth": self.depth,
            "created_at_step": self.created_at_step,
        }


class PowerSystemStateStore:
    """Episode-namespaced in-memory transactional state store."""

    def __init__(self) -> None:
        self._states: dict[str, PowerSystemState] = {}
        self._episode_counters: dict[str, int] = {}
        self._episode_serial = 0
        self._current_episode_id: str | None = None
        self._active_state_id: str | None = None

    @property
    def active_state_id(self) -> str | None:
        return self._active_state_id

    @property
    def current_episode_id(self) -> str | None:
        return self._current_episode_id

    def clear(self) -> None:
        self._states.clear()
        self._episode_counters.clear()
        self._episode_serial = 0
        self._current_episode_id = None
        self._active_state_id = None

    @staticmethod
    def normalize_episode_id(episode_id: Any) -> str:
        text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(episode_id)).strip("_")
        return text or "episode"

    def _next_id(self) -> str:
        if self._current_episode_id is None:
            raise StateStoreError("No current episode exists.")
        counter = self._episode_counters[self._current_episode_id]
        self._episode_counters[self._current_episode_id] = counter + 1
        return f"{self._current_episode_id}:s{counter}"

    def create_root(
        self,
        case: Any,
        measurements: Any,
        metadata: Mapping[str, Any] | None = None,
        *,
        episode_id: str | None = None,
        created_at_step: int = 0,
    ) -> str:
        if episode_id is None:
            episode_id = f"episode_{self._episode_serial}"
            self._episode_serial += 1
        episode_id = self.normalize_episode_id(episode_id)
        if episode_id in self._episode_counters:
            raise StateStoreError(f"Episode already exists: {episode_id}")
        if self._active_state_id and self._active_state_id in self._states:
            active = self._states[self._active_state_id]
            active.active = False
            if active.status == "active":
                active.status = "inactive"
        self._current_episode_id = episode_id
        self._episode_counters[episode_id] = 0
        state_id = self._next_id()
        self._states[state_id] = PowerSystemState(
            episode_id=episode_id,
            state_id=state_id,
            case=_json_clone(case),
            measurements=_json_clone(measurements),
            metadata=dict(_json_clone(metadata or {})),
            status="active",
            active=True,
            depth=0,
            created_at_step=int(created_at_step),
        )
        self._active_state_id = state_id
        return state_id

    def clone_candidate(
        self,
        parent_state_id: str,
        modification: Mapping[str, Any],
        source_action: Mapping[str, Any],
        *,
        created_at_step: int = 0,
    ) -> str:
        parent = self._require_state(parent_state_id)
        if parent.state_id != self._active_state_id or parent.status != "active":
            raise StateStoreError(f"Candidate parent must be the active state: {parent_state_id}.")
        new_case, new_measurements, new_metadata = apply_modification(
            case=parent.case,
            measurements=parent.measurements,
            metadata=parent.metadata,
            modification=modification,
        )
        if new_case == parent.case and new_measurements == parent.measurements:
            raise StateStoreError("Correction produced no physical case or measurement change.")
        if _state_content_hash(new_case, new_measurements, new_metadata) == parent.state_hash:
            raise StateStoreError("Correction produced no physical state change.")
        state_id = self._next_id()
        self._states[state_id] = PowerSystemState(
            episode_id=parent.episode_id,
            state_id=state_id,
            case=new_case,
            measurements=new_measurements,
            metadata=new_metadata,
            parent_state_id=parent_state_id,
            status="candidate",
            source_action=dict(_json_clone(source_action)),
            modification=dict(_json_clone(modification)),
            depth=parent.depth + 1,
            created_at_step=int(created_at_step),
        )
        return state_id

    def get_state(self, state_id: str) -> dict[str, Any]:
        return self._require_state(state_id).as_dict()

    def get_state_for_audit(self, state_id: str) -> dict[str, Any]:
        try:
            return self._states[state_id].as_dict()
        except KeyError as exc:
            raise StateStoreError(f"Unknown state_id: {state_id}") from exc

    def exists(self, state_id: str | None) -> bool:
        if not state_id or state_id not in self._states:
            return False
        return self._states[state_id].episode_id == self._current_episode_id

    def state_hash(self, state_id: str) -> str:
        return self._require_state(state_id).state_hash

    def episode_hash(self) -> str:
        payload = [
            state.as_dict()
            for state in self._states.values()
            if state.episode_id == self._current_episode_id
        ]
        return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()

    def mark_verified(
        self,
        candidate_state_id: str,
        verification_output: Mapping[str, Any],
        candidate_disposition: Any,
        candidate_assessment: Mapping[str, Any] | None = None,
    ) -> str:
        candidate = self._require_state(candidate_state_id)
        if candidate.status != "candidate" or candidate.candidate_disposition is not None:
            raise StateStoreError(f"State {candidate_state_id} is not an unverified candidate.")
        value = getattr(candidate_disposition, "value", candidate_disposition)
        value = str(value)
        if value not in {"ACCEPT_FINAL", "ACCEPT_PARTIAL", "REJECT", "INCONCLUSIVE"}:
            raise StateStoreError(f"Unknown candidate disposition: {value}")
        candidate.verification_output = dict(_json_clone(verification_output))
        candidate.candidate_disposition = value
        candidate.candidate_assessment = dict(_json_clone(candidate_assessment or {}))
        return candidate_state_id

    def commit(self, candidate_state_id: str) -> str:
        candidate = self._require_state(candidate_state_id)
        if candidate.state_id == self._active_state_id or candidate.status != "candidate":
            raise StateStoreError(f"Only the current open candidate can be committed: {candidate_state_id}.")
        if candidate.parent_state_id != self._active_state_id:
            raise StateStoreError(f"Candidate {candidate_state_id} does not belong to the active state.")
        if candidate.candidate_disposition not in {"ACCEPT_PARTIAL", "ACCEPT_FINAL"}:
            raise StateStoreError(
                f"Candidate {candidate_state_id} cannot be committed with disposition "
                f"{candidate.candidate_disposition!r}."
            )
        active = self._require_state(str(self._active_state_id))
        active.active = False
        active.status = "inactive"
        candidate.status = "active"
        candidate.active = True
        self._active_state_id = candidate_state_id
        return candidate_state_id

    def rollback(self, candidate_state_id: str) -> str:
        candidate = self._require_state(candidate_state_id)
        if candidate.status != "candidate" or not candidate.parent_state_id:
            raise StateStoreError(f"Only the current open candidate can be rolled back: {candidate_state_id}.")
        if candidate.parent_state_id != self._active_state_id:
            raise StateStoreError(f"Candidate {candidate_state_id} does not belong to the active state.")
        if candidate.candidate_disposition not in {"REJECT", "INCONCLUSIVE"}:
            raise StateStoreError(
                f"Candidate {candidate_state_id} cannot be rolled back with disposition "
                f"{candidate.candidate_disposition!r}."
            )
        candidate.status = "rolled_back"
        candidate.active = False
        return candidate.parent_state_id

    def lineage(self, state_id: str) -> list[str]:
        lineage: list[str] = []
        current: str | None = state_id
        episode_id = self._require_state(state_id).episode_id
        while current is not None:
            state = self._require_state(current)
            if state.episode_id != episode_id:
                raise StateStoreError("State lineage crossed an episode boundary.")
            lineage.append(current)
            current = state.parent_state_id
        return list(reversed(lineage))

    def candidate_provenance(self, candidate_state_id: str) -> dict[str, Any]:
        candidate = self._require_state(candidate_state_id)
        if not candidate.parent_state_id:
            raise StateStoreError(f"State {candidate_state_id} is not a candidate.")
        parent = self._require_state(candidate.parent_state_id)
        if parent.episode_id != candidate.episode_id:
            raise StateStoreError("Candidate parent belongs to a different episode.")
        signature = None
        if isinstance(candidate.modification, Mapping):
            signature = candidate.modification.get("modification_signature") or candidate.modification.get("signature")
        if signature is None and candidate.source_action:
            signature = action_signature(candidate.source_action)
        return {
            "execution_status": "success",
            "episode_id": candidate.episode_id,
            "parent_state_id": parent.state_id,
            "candidate_state_id": candidate.state_id,
            "state_hash_before": parent.state_hash,
            "state_hash_after": candidate.state_hash,
            "modification_signature": signature,
            "depth": candidate.depth,
            "created_at_step": candidate.created_at_step,
        }

    def decision_summary(
        self,
        *,
        candidate_state_id: str | None = None,
        remaining_budget: int | None = None,
        context_flags: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self._active_state_id is None:
            raise StateStoreError("No active state exists.")
        active = self._require_state(self._active_state_id)
        candidate = self._require_state(candidate_state_id) if candidate_state_id else None
        if candidate and candidate.parent_state_id != active.state_id:
            raise StateStoreError("Current candidate does not belong to the active state.")
        verification = candidate.verification_output if candidate else None
        source_action = candidate.source_action if candidate else None
        flags = dict(context_flags or {})
        remaining_anomaly_score = flags.get("remaining_anomaly_score")
        if remaining_anomaly_score is None and isinstance(verification, Mapping):
            remaining_anomaly_score = verification.get("remaining_anomaly_score")
        semantic_field_provenance = dict(flags.get("semantic_field_provenance") or {})
        if (
            isinstance(verification, Mapping)
            and verification.get("remaining_anomaly_score") is not None
        ):
            semantic_field_provenance["remaining_anomaly_score"] = str(
                verification.get("evidence_source") or "observable_candidate_verification"
            )

        def fresh_context(family: str) -> bool:
            state_id = flags.get(f"{family}_context_state_id")
            return bool(flags.get(f"has_fresh_{family}_context")) and str(state_id) == active.state_id

        lifecycle = candidate.candidate_lifecycle if candidate else CandidateLifecycle.NO_CANDIDATE
        return {
            "episode_id": active.episode_id,
            "active_state_id": active.state_id,
            "candidate_state_id": candidate.state_id if candidate else None,
            "candidate_parent_id": candidate.parent_state_id if candidate else None,
            "candidate_status": candidate.candidate_status if candidate else None,
            "candidate_lifecycle": lifecycle.value,
            "candidate_disposition": candidate.candidate_disposition if candidate else None,
            "candidate_committed": bool(candidate and candidate.status == "active"),
            "has_open_candidate": bool(candidate and candidate.status == "candidate"),
            "has_unverified_candidate": bool(candidate and lifecycle == CandidateLifecycle.OPEN_UNVERIFIED_CANDIDATE),
            "has_verified_candidate": bool(
                candidate
                and lifecycle
                in {
                    CandidateLifecycle.VERIFIED_ACCEPTED_CANDIDATE,
                    CandidateLifecycle.VERIFIED_REJECTED_CANDIDATE,
                    CandidateLifecycle.VERIFIED_INCONCLUSIVE_CANDIDATE,
                }
            ),
            "last_tool": flags.get("last_tool")
            or (source_action.get("tool") if isinstance(source_action, Mapping) else None),
            "last_tool_status": flags.get("last_tool_status"),
            "last_tool_output": policy_safe_copy(flags.get("last_tool_output") or {}),
            "last_verification": policy_safe_copy(verification or {}),
            "accepted_corrections": policy_safe_copy(list(flags.get("accepted_corrections", []))),
            "explained_anomalies": policy_safe_copy(list(flags.get("explained_anomalies", []))),
            "rejected_hypotheses": policy_safe_copy(list(flags.get("rejected_hypotheses", []))),
            "tried_action_signatures": list(flags.get("tried_action_signatures", [])),
            "unresolved_signatures": list(flags.get("unresolved_signatures", [])),
            "remaining_budget": remaining_budget,
            "remaining_anomaly_score": remaining_anomaly_score,
            "no_material_anomaly_remaining": bool(flags.get("no_material_anomaly_remaining", False)),
            "has_fresh_measurement_context": fresh_context("measurement"),
            "has_fresh_parameter_context": fresh_context("parameter"),
            "has_fresh_topology_context": fresh_context("topology"),
            "measurement_context_state_id": flags.get("measurement_context_state_id"),
            "parameter_context_state_id": flags.get("parameter_context_state_id"),
            "topology_context_state_id": flags.get("topology_context_state_id"),
            "fresh_context_evidence": policy_safe_copy(
                dict(flags.get("fresh_context_evidence") or {})
            ),
            "requires_measurement_context": bool(flags.get("requires_measurement_context", False)),
            "semantic_field_provenance": policy_safe_copy(semantic_field_provenance),
        }

    def _require_state(self, state_id: str) -> PowerSystemState:
        try:
            state = self._states[state_id]
        except KeyError as exc:
            raise StateStoreError(f"Unknown state_id: {state_id}") from exc
        if state.episode_id != self._current_episode_id:
            raise StateStoreError(f"State {state_id} does not belong to the current episode.")
        return state
