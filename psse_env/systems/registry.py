"""Canonical IEEE case assets, identities, and the existing WLS observation contract.

This registry does not enable detailed topology or three-phase physics for IEEE
57 or IEEE 118. Its covariance is deliberately fixed to the deployed balanced WLS solver.
Cases are loaded from repository assets, never the installed PYPOWER version.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


_REPO_ROOT = Path(__file__).resolve().parents[2]
_CASE57_SOURCE_SHA256 = "97802230f6b4dafd484e97f061c69542359648ab7ab5d51949368e68f4c975f6"
_CASE57_BASE_CASE_HASH = "417701198ec205ae9cf8502b365664c1adb5a265894a03ffa4ddba95d540beca"
_CASE118_SOURCE_SHA256 = "90c28f0d55324a6f11b6371c3fd8424b3c1bc18830699deaec207d7d72e50c25"
_NOISE_MODEL = "balanced_wls_vm001_power01_v1"
_BALANCED_FAMILIES = (
    "no_error", "measurement", "multi_measurement", "parameter", "measurement+parameter"
)
_CASE14_FAMILIES = (
    "no_error", "measurement", "parameter", "topology", "harmonic", "hif",
    "multi_measurement", "measurement+parameter", "measurement+topology",
    "measurement+hif", "three_phase_unbalance", "telemetry_no_disturbance",
)


def _read_case(path: Path) -> dict[str, Any]:
    """Read the fixed numeric case format, retaining OPF cost data as well."""
    text = path.read_text(encoding="utf-8")
    scalar = re.search(r"mpc\.baseMVA\s*=\s*([^;]+);", text)
    version = re.search(r"mpc\.version\s*=\s*'([^']+)';", text)
    if scalar is None or version is None:
        raise ValueError(f"Canonical case lacks version/baseMVA: {path}")
    case: dict[str, Any] = {
        "version": version.group(1), "baseMVA": float(scalar.group(1))
    }
    for name in ("bus", "gen", "branch", "gencost"):
        match = re.search(rf"mpc\.{name}\s*=\s*\[(.*?)\];", text, re.DOTALL)
        if match is None:
            raise ValueError(f"Canonical case lacks {name}: {path}")
        matrix_text = re.sub(r"%[^\n]*", "", match.group(1))
        rows = [
            [float(value) for value in row.replace(",", " ").split()]
            for row in matrix_text.split(";") if row.strip()
        ]
        matrix = np.asarray(rows, dtype=float)
        if matrix.ndim != 2 or not matrix.size or not np.isfinite(matrix).all():
            raise ValueError(f"Invalid canonical {name} matrix: {path}")
        case[name] = matrix
    return case


def _case_hash(case: Mapping[str, Any]) -> str:
    """Semantic hash is independent of asset newlines and machine paths."""
    content = {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in case.items()
    }
    encoded = json.dumps(content, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class BusAsset:
    """Stable bus identity with external number and canonical matrix row."""

    asset_id: str
    row0: int
    external_bus: int


@dataclass(frozen=True)
class BranchAsset:
    """A canonical branch row; parallel circuits remain separate assets.

    ``index1`` is the existing correction-tool index. ``circuit_ordinal`` is
    one-based within the unordered external endpoint pair in canonical row
    order, while ``from_bus``/``to_bus`` retain the source orientation.
    """

    asset_id: str
    row0: int
    index1: int
    from_bus: int
    to_bus: int
    circuit_ordinal: int
    tap: float
    status: int


@dataclass(frozen=True)
class SystemSpec:
    case_id: str
    case_path: str
    nb: int
    nl: int
    nz: int
    state_count: int
    base_case_hash: str
    supported_families: tuple[str, ...]
    buses: tuple[BusAsset, ...]
    branches: tuple[BranchAsset, ...]
    source_provenance: Mapping[str, str]
    _asset_path: Path = field(repr=False)

    @property
    def external_bus_to_row0(self) -> Mapping[int, int]:
        return MappingProxyType({bus.external_bus: bus.row0 for bus in self.buses})

    @property
    def row0_to_external_bus(self) -> tuple[int, ...]:
        return tuple(bus.external_bus for bus in self.buses)

    @property
    def eligible_parameter_rows0(self) -> tuple[int, ...]:
        """Existing parameter generator targets active lines without taps."""
        return tuple(branch.row0 for branch in self.branches if branch.status > 0 and branch.tap == 0.0)

    def load_case(self) -> dict[str, Any]:
        """Return fresh mutable matrices; reject an asset changed after resolution."""
        case = _read_case(self._asset_path)
        if _case_hash(case) != self.base_case_hash:
            raise ValueError(f"Canonical {self.case_id} asset changed after system resolution")
        return case

    def measurement_sigma(self) -> np.ndarray:
        """The deployed WLS covariance: .001 pu Vm and .01 pu power."""
        sigma = np.full(self.nz, 0.01, dtype=float)
        sigma[:self.nb] = 0.001
        return sigma

    def to_manifest(self) -> dict[str, Any]:
        """Return independent JSON-ready provenance and observation metadata."""
        ranges = (
            ("Vm", 0, self.nb), ("Pinj", self.nb, 2 * self.nb),
            ("Qinj", 2 * self.nb, 3 * self.nb),
            ("Pf", 3 * self.nb, 3 * self.nb + self.nl),
            ("Qf", 3 * self.nb + self.nl, 3 * self.nb + 2 * self.nl),
            ("Pt", 3 * self.nb + 2 * self.nl, 3 * self.nb + 3 * self.nl),
            ("Qt", 3 * self.nb + 3 * self.nl, self.nz),
        )
        return {
            "schema": "balanced_system_spec_v1",
            "case_id": self.case_id, "case_path": self.case_path,
            "base_case_hash": self.base_case_hash,
            "base_case_hash_contract": "sha256_sorted_json_case_matrices_v1",
            "source_provenance": dict(self.source_provenance),
            "nb": self.nb, "nl": self.nl, "nz": self.nz,
            "state_count": self.state_count,
            "residual_degrees_of_freedom": self.nz - self.state_count,
            "supported_families": list(self.supported_families),
            "measurement_contract": {
                "layout": {name: [start, stop] for name, start, stop in ranges},
                "indexing": "zero_based_half_open",
                "units": "per_unit_on_case_baseMVA",
                "covariance_model": _NOISE_MODEL,
                "sigma_vm": 0.001, "sigma_power": 0.01,
                "injection_sign": "generation_minus_load",
                "branch_flow_sign": "injection_into_branch_at_named_terminal",
                "capabilities": {
                    "balanced_snapshot": True,
                    "detailed_topology": self.case_id == "case14",
                    "three_phase": self.case_id == "case14",
                },
            },
            "buses": [asdict(bus) for bus in self.buses],
            "branches": [asdict(branch) for branch in self.branches],
            "eligible_parameter_rows0": list(self.eligible_parameter_rows0),
        }


@lru_cache(maxsize=3)
def _resolve_canonical(case_id: str) -> SystemSpec:
    asset_path = _REPO_ROOT / "mcp_server" / f"{case_id}.m"
    case = _read_case(asset_path)
    if case_id == "case57" and _case_hash(case) != _CASE57_BASE_CASE_HASH:
        raise ValueError("Canonical case57 matrices do not match the pinned PYPOWER 5.1.19 source")
    bus, branch = case["bus"], case["branch"]
    external = bus[:, 0].astype(int)
    if not np.array_equal(bus[:, 0], external) or len(set(external)) != len(external):
        raise ValueError(f"Invalid external bus numbering for {case_id}")
    if np.count_nonzero(bus[:, 1] == 3) != 1:
        raise ValueError(f"{case_id} must have one reference bus")
    buses = tuple(
        BusAsset(f"{case_id}:bus:{number}", row0, int(number))
        for row0, number in enumerate(external)
    )
    circuits: dict[tuple[int, int], int] = {}
    branches = []
    for row0, row in enumerate(branch):
        from_bus, to_bus = int(row[0]), int(row[1])
        if from_bus not in external or to_bus not in external:
            raise ValueError(f"Unknown branch endpoint in {case_id} row {row0}")
        pair = tuple(sorted((from_bus, to_bus)))
        circuits[pair] = circuits.get(pair, 0) + 1
        branches.append(BranchAsset(
            asset_id=f"{case_id}:branch:{row0 + 1}", row0=row0, index1=row0 + 1,
            from_bus=from_bus, to_bus=to_bus, circuit_ordinal=circuits[pair],
            tap=float(row[8]), status=int(row[10]),
        ))
    provenance = {"asset_path": f"mcp_server/{case_id}.m"}
    source_sha256 = {"case57": _CASE57_SOURCE_SHA256, "case118": _CASE118_SOURCE_SHA256}
    if case_id in source_sha256:
        provenance.update({
            "source": "PYPOWER", "source_version": "5.1.19",
            "source_file": f"pypower/{case_id}.py", "source_sha256": source_sha256[case_id],
            "source_url": f"https://github.com/rwl/PYPOWER/blob/v5.1.19/pypower/{case_id}.py",
        })
    else:
        provenance["source"] = "existing_repository_case14_asset"
    nb, nl = len(bus), len(branch)
    return SystemSpec(
        case_id=case_id, case_path=case_id, nb=nb, nl=nl, nz=3 * nb + 4 * nl,
        state_count=2 * nb - 1, base_case_hash=_case_hash(case),
        supported_families=_CASE14_FAMILIES if case_id == "case14" else _BALANCED_FAMILIES,
        buses=buses, branches=tuple(branches),
        source_provenance=MappingProxyType(provenance), _asset_path=asset_path,
    )


def resolve_system(system: str = "case14", *, covariance_model: str = _NOISE_MODEL) -> SystemSpec:
    """Resolve a supported named system without altering legacy IEEE 14 defaults.

    Unsupported noise models fail explicitly: declaring a new covariance here
    cannot change the fixed covariance used by deployment WLS.
    """
    if covariance_model != _NOISE_MODEL:
        raise ValueError(f"Unsupported covariance model: {covariance_model!r}; expected {_NOISE_MODEL!r}")
    if not isinstance(system, str):
        raise TypeError("system must be a registered case name")
    aliases = {
        "14": "case14", "ieee14": "case14", "case14": "case14", "case14.m": "case14",
        "57": "case57", "ieee57": "case57", "case57": "case57", "case57.m": "case57",
        "118": "case118", "ieee118": "case118", "case118": "case118", "case118.m": "case118",
    }
    try:
        canonical = aliases[system.strip().lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported system {system!r}; choose case14, case57 or case118") from exc
    return _resolve_canonical(canonical)
