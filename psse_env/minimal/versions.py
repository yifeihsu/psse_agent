"""The saved states of one episode.

Every correction saves the state it starts from and adds a new one, so any
earlier state can be restored.  A state is the operator's case (a MATPOWER case
path), the measurement vector and the observable metadata the numerical tools
read (noise declaration, repeated scans, substation telemetry, reported breaker
statuses).  Nothing here judges a state: there is no candidate, no verdict and
no commit.  ``restore`` moves the current pointer to a saved state; the states
it leaves stay in the list and their numbers are never reused.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from mcp_server.matpower_server import _load_python_case
from psse_env.state_store import _state_content_hash, apply_modification

BR_R, BR_X, BR_STATUS = 2, 3, 10
#: Metadata the tools write for their own bookkeeping; a change of these alone is not a change of the state.
_BOOKKEEPING = ("transactional_modifications", "last_parameter_correction", "last_topology_correction")


class VersionError(RuntimeError):
    """A request the saved states cannot serve (an unknown version, a change that changes nothing)."""


@dataclass
class Version:
    index: int
    parent: int | None
    case: str
    measurements: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)
    #: The step whose correction created this state (``None`` for the first one).
    created_by_step: int | None = None

    @property
    def state_hash(self) -> str:
        return _state_content_hash(self.case, self.measurements, self.metadata)


class VersionStore:
    """The states of one episode, the first one at index 0."""

    def __init__(self, case: Any, measurements: Any, metadata: Mapping[str, Any] | None = None) -> None:
        if isinstance(case, Mapping):
            case = case.get("case_path")
        if not isinstance(case, str) or not case:
            raise VersionError("the case must be a MATPOWER case path")
        self._versions: list[Version] = [
            Version(index=0, parent=None, case=case, measurements=[float(value) for value in measurements],
                    metadata=copy.deepcopy(dict(metadata or {})))
        ]
        self.current = 0

    def __len__(self) -> int:
        return len(self._versions)

    def exists(self, index: Any) -> bool:
        return isinstance(index, int) and not isinstance(index, bool) and 0 <= index < len(self._versions)

    def version(self, index: int | None = None) -> Version:
        index = self.current if index is None else index
        if not self.exists(index):
            raise VersionError(f"unknown version {index!r}")
        return self._versions[index]

    def state(self, index: int | None = None) -> dict[str, Any]:
        """The state in the form the numerical tools take; a copy, so a tool cannot change what is saved."""
        version = self.version(index)
        return {
            "case": version.case,
            "measurements": list(version.measurements),
            "metadata": copy.deepcopy(version.metadata),
            "state_id": f"v{version.index}",
            "state_hash": version.state_hash,
        }

    def lineage(self, index: int | None = None) -> list[int]:
        """The versions from the first one to ``index``, oldest first."""
        chain: list[int] = []
        cursor: int | None = self.version(index).index
        while cursor is not None:
            chain.append(cursor)
            cursor = self._versions[cursor].parent
        return chain[::-1]

    def apply(self, modification: Mapping[str, Any], *, step: int) -> int:
        """Save a new state: the current one with ``modification`` applied.  The current state is untouched."""
        parent = self.version()
        case, measurements, metadata = apply_modification(
            case=parent.case, measurements=parent.measurements, metadata=parent.metadata, modification=modification,
        )
        if isinstance(case, Mapping):
            case = case.get("case_path")
        if case == parent.case and measurements == parent.measurements and _physical(metadata) == _physical(parent.metadata):
            raise VersionError("the correction changed nothing")
        child = Version(index=len(self._versions), parent=parent.index, case=str(case),
                        measurements=[float(value) for value in measurements], metadata=dict(metadata),
                        created_by_step=int(step))
        self._versions.append(child)
        self.current = child.index
        return child.index

    def restore(self, to: int) -> list[int]:
        """Make a saved state current again; the versions that leave the current lineage, oldest first.

        Any saved state can be restored, an earlier one of the current
        lineage (the usual undo) or one left behind by an earlier undo.
        """
        if not self.exists(to):
            raise VersionError(f"unknown version {to!r}")
        if int(to) == self.current:
            raise VersionError(f"version {to} is already the current state")
        kept = set(self.lineage(int(to)))
        left = [index for index in self.lineage() if index not in kept]
        self.current = int(to)
        return left


def _physical(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metadata.items() if key not in _BOOKKEEPING}


def _number(value: Any) -> float:
    return float(round(float(value), 6))


def state_changes(before: Mapping[str, Any], after: Mapping[str, Any]) -> list[dict[str, Any]]:
    """What differs between two states, read from the states themselves and not from a tool's own report.

    One record per changed object: a line's series R or X or its service
    status, a reported breaker status, a measurement.  A change that re-renders
    the operator model (a breaker flip that splits or merges buses) is one
    record with the bus and measurement counts, because rows no longer
    correspond one to one.
    """
    changes: list[dict[str, Any]] = []
    labels_before = dict((before.get("metadata") or {}).get("reported_breaker_status") or {})
    labels_after = dict((after.get("metadata") or {}).get("reported_breaker_status") or {})
    for name in sorted(set(labels_before) | set(labels_after)):
        if str(labels_before.get(name)) != str(labels_after.get(name)):
            changes.append({"object": "breaker", "breaker": str(name),
                            "before": labels_before.get(name), "after": labels_after.get(name)})
    old_case = _load_python_case(str(before["case"]))
    new_case = _load_python_case(str(after["case"]))
    old_branch = np.asarray(old_case["branch"], dtype=float)
    new_branch = np.asarray(new_case["branch"], dtype=float)
    old_z = [float(value) for value in before["measurements"]]
    new_z = [float(value) for value in after["measurements"]]
    rendered = (
        np.asarray(old_case["bus"]).shape[0] != np.asarray(new_case["bus"]).shape[0]
        or old_branch.shape[0] != new_branch.shape[0] or len(old_z) != len(new_z)
    )
    if rendered:
        changes.append({
            "object": "operator_model",
            "buses_before": int(np.asarray(old_case["bus"]).shape[0]), "buses_after": int(np.asarray(new_case["bus"]).shape[0]),
            "measurements_before": len(old_z), "measurements_after": len(new_z),
        })
    if old_branch.shape[0] == new_branch.shape[0]:
        for row in range(old_branch.shape[0]):
            for column, name in ((BR_R, "r"), (BR_X, "x")):
                if float(old_branch[row, column]) != float(new_branch[row, column]):
                    changes.append({"object": "line", "line": row + 1, "field": name,
                                    "before": _number(old_branch[row, column]), "after": _number(new_branch[row, column])})
            if old_branch.shape[1] > BR_STATUS and new_branch.shape[1] > BR_STATUS:
                if float(old_branch[row, BR_STATUS]) != float(new_branch[row, BR_STATUS]):
                    changes.append({"object": "line", "line": row + 1, "field": "status",
                                    "before": "in" if old_branch[row, BR_STATUS] > 0 else "out",
                                    "after": "in" if new_branch[row, BR_STATUS] > 0 else "out"})
    if not rendered:
        for index, (old, new) in enumerate(zip(old_z, new_z)):
            if old != new:
                changes.append({"object": "meter", "meter": index, "before": _number(old), "after": _number(new)})
    return changes


__all__ = ["Version", "VersionError", "VersionStore", "state_changes"]
