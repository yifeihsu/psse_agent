"""Round-0 scenario synthesis for the recovery-balanced DAgger aggregate.

Builds ``TransactionalPSSEEnv`` scenarios on the real MATPOWER/OpenDSS stack
from three physically consistent sources:

- the production measurement corpus (``data/measurements_5class_merged.jsonl``)
  for no-error, gross-measurement, parameter (with multi-scan telemetry), and
  harmonic snapshots;
- the OpenDSS HIF sample sets (``artifacts/measurements/hif_multiscan_*``)
  for high-impedance-fault snapshots with real NLM diagnostics and persistent
  scan windows;
- direct power-flow synthesis (PYPOWER + the Lagrangian measurement model)
  for topology errors, because the corpus topology rows are node-breaker CB
  events that a case14 branch-status correction cannot represent.

Multi-error compositions overlay gross sensor offsets on any base snapshot:
a gross measurement error is additive on ``z``, so the combined vector stays
physically consistent with the base scenario's system state.

Scenario identifiers are deliberately opaque (``r0_<digest>``): the scenario
id reaches policy-visible metadata, so family names must never appear in it.
The family lives only in the top-level ``scenario_family`` key (ignored by the
environment) and in the generator's ``manifest``.

Every scenario is validated against the same WLS stack the environment uses:
correction families must present a clearly detectable anomaly on the
agent-visible model and must return below the chi-square threshold on the
corrected configuration; explanation families (harmonic, HIF) must solve.
Rows that fail validation are skipped and recorded in ``skipped``.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from mcp_server.matpower_server import (  # noqa: E402  (repo-root package)
    _load_python_case,
    _wls_json,
)
from tools.lagrangian_correct_port import make_ybus  # noqa: E402
from trace_protocol import chi2_threshold  # noqa: E402

from psse_env.providers.matpower import (
    _render_matpower_case,
    measurement_index_map,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CORPUS_PATH = _REPO_ROOT / "data" / "measurements_5class_merged.jsonl"
DEFAULT_HIF_SAMPLE_PATHS = (
    _REPO_ROOT
    / "artifacts"
    / "measurements"
    / "hif_multiscan_benchmark_fixed_diverse_17x20_20260714"
    / "samples.jsonl",
    _REPO_ROOT
    / "artifacts"
    / "measurements"
    / "hif_multiscan_benchmark_fixed_identical_17x20_20260714"
    / "samples.jsonl",
)
DEFAULT_BALANCED_ARTIFACT_DIR = (
    _REPO_ROOT / "artifacts" / "measurements" / "out_measurements_balanced"
)

NB, NL = 14, 20
NZ = 3 * NB + 4 * NL
_STATE_COUNT = 2 * NB - 1

BASE_FAMILIES = (
    "no_error",
    "measurement",
    "parameter",
    "topology",
    "harmonic",
    "hif",
)
COMPOSED_FAMILIES = (
    "multi_measurement",
    "measurement+parameter",
    "measurement+topology",
    "measurement+hif",
)
SCENARIO_FAMILIES = BASE_FAMILIES + COMPOSED_FAMILIES

_SNAPSHOT_PROVENANCE = "deployment_sensor:scada_snapshot"
_POWER_QUALITY_PROVENANCE = "deployment_sensor:power_quality"
_WAVEFORM_PROVENANCE = "deployment_sensor:waveform_capture"

HARMONIC_SIGNATURE = "harmonic_distortion_detected"
HIF_SIGNATURE = "hif_suspected_zero_sequence"


class ScenarioRejected(RuntimeError):
    """A source row cannot become a physically validated scenario."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


def build_measurement_vector(ppc: Mapping[str, Any]) -> np.ndarray:
    """Evaluate the WLS measurement function ``h(x)`` at the case's stored state.

    Uses the same admittance construction as the Lagrangian WLS port, so a
    vector built from a solved power flow is exactly consistent with the
    estimator's model.
    """
    bus = np.asarray(ppc["bus"], dtype=float).copy()
    branch = np.asarray(ppc["branch"], dtype=float).copy()
    base_mva = float(ppc["baseMVA"])
    branch[:, 0] -= 1.0
    branch[:, 1] -= 1.0
    ybus, yf, yt = make_ybus(base_mva, bus, branch)
    voltage = bus[:, 7] * np.exp(1j * np.pi / 180.0 * bus[:, 8])
    injection = voltage * np.conj(ybus @ voltage)
    from_bus = branch[:, 0].astype(int)
    to_bus = branch[:, 1].astype(int)
    s_from = voltage[from_bus] * np.conj(yf @ voltage)
    s_to = voltage[to_bus] * np.conj(yt @ voltage)
    return np.r_[
        np.abs(voltage),
        injection.real,
        injection.imag,
        s_from.real,
        s_from.imag,
        s_to.real,
        s_to.imag,
    ]


class Round0ScenarioGenerator:
    """Deterministic, validated scenario source for round-0 collection."""

    def __init__(
        self,
        *,
        corpus_path: str | Path | None = None,
        hif_sample_paths: Sequence[str | Path] | None = None,
        balanced_artifact_dir: str | Path | None = None,
        derived_case_dir: str | Path | None = None,
        seed: int = 20260719,
        validate: bool = True,
        chi2_alpha: float = 0.01,
        anomaly_margin: float = 1.25,
        topology_noise_scale: float = 0.8,
        hif_max_scans: int = 8,
        noise_profile_rows: int = 200,
    ) -> None:
        self.corpus_path = Path(corpus_path or DEFAULT_CORPUS_PATH)
        self.hif_sample_paths = [
            Path(path) for path in (hif_sample_paths or DEFAULT_HIF_SAMPLE_PATHS)
        ]
        self.balanced_artifact_dir = Path(
            balanced_artifact_dir or DEFAULT_BALANCED_ARTIFACT_DIR
        )
        self.derived_case_dir = Path(
            derived_case_dir
            or os.path.join(tempfile.gettempdir(), "psse_round0_cases")
        )
        self.seed = int(seed)
        self.validate = bool(validate)
        self.chi2_alpha = float(chi2_alpha)
        self.anomaly_margin = float(anomaly_margin)
        self.topology_noise_scale = float(topology_noise_scale)
        self.hif_max_scans = int(hif_max_scans)
        self.noise_profile_rows = int(noise_profile_rows)

        self.manifest: list[dict[str, Any]] = []
        self.skipped: list[dict[str, Any]] = []
        self._rng = np.random.default_rng(self.seed)
        self._corpus_by_class: dict[str, list[dict[str, Any]]] | None = None
        self._hif_samples: list[dict[str, Any]] | None = None
        self._noise_std: np.ndarray | None = None
        self._case14: dict[str, Any] | None = None
        self._chi2_cache: dict[tuple[str, str], float] = {}

    # ------------------------------------------------------------ data access

    def _corpus(self) -> dict[str, list[dict[str, Any]]]:
        if self._corpus_by_class is None:
            grouped: dict[str, list[dict[str, Any]]] = {}
            with open(self.corpus_path, encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    grouped.setdefault(str(row.get("scenario")), []).append(row)
            self._corpus_by_class = grouped
        return self._corpus_by_class

    def _hif_rows(self) -> list[dict[str, Any]]:
        if self._hif_samples is None:
            rows: list[dict[str, Any]] = []
            for path in self.hif_sample_paths:
                if not path.is_file():
                    continue
                with open(path, encoding="utf-8") as handle:
                    rows.extend(json.loads(line) for line in handle)
            self._hif_samples = rows
        return self._hif_samples

    def noise_profile(self) -> np.ndarray:
        """Per-index measurement noise std estimated from no-error corpus rows."""
        if self._noise_std is None:
            diffs = [
                np.asarray(row["z_obs"], dtype=float)
                - np.asarray(row["z_true"], dtype=float)
                for row in self._corpus().get("no_error", [])[: self.noise_profile_rows]
            ]
            if not diffs:
                raise RuntimeError(
                    f"corpus {self.corpus_path} carries no no_error rows for the noise profile"
                )
            self._noise_std = np.stack(diffs).std(axis=0)
        return self._noise_std

    def _clean_case14(self) -> dict[str, Any]:
        if self._case14 is None:
            self._case14 = _load_python_case("case14")
        return self._case14

    # ------------------------------------------------------------- validation

    def _chi2_statistic(self, case: str, z: Sequence[float]) -> float:
        key = (case, hashlib.sha256(np.asarray(z, dtype=float).tobytes()).hexdigest())
        if key not in self._chi2_cache:
            payload = _wls_json(case, [float(value) for value in z])
            if not payload.get("success"):
                raise ScenarioRejected("wls_failure", str(payload.get("error")))
            self._chi2_cache[key] = float(payload.get("global_residual_sum") or 0.0)
        return self._chi2_cache[key]

    @property
    def chi2_limit(self) -> float:
        dof = max(1, NZ - _STATE_COUNT)
        return float(chi2_threshold(dof, self.chi2_alpha))

    def _require_anomalous(self, case: str, z: Sequence[float], family: str) -> None:
        if not self.validate:
            return
        statistic = self._chi2_statistic(case, z)
        if statistic <= self.anomaly_margin * self.chi2_limit:
            raise ScenarioRejected(
                "anomaly_not_detectable",
                f"{family}: chi2 {statistic:.1f} <= {self.anomaly_margin:.2f} x {self.chi2_limit:.1f}",
            )

    def _require_clean(self, case: str, z: Sequence[float], family: str) -> None:
        if not self.validate:
            return
        statistic = self._chi2_statistic(case, z)
        if statistic >= self.chi2_limit:
            raise ScenarioRejected(
                "corrected_configuration_still_anomalous",
                f"{family}: chi2 {statistic:.1f} >= {self.chi2_limit:.1f}",
            )

    # -------------------------------------------------------------- utilities

    def _scenario_id(self, *parts: Any) -> str:
        digest = hashlib.sha256(
            json.dumps([self.seed, *[str(part) for part in parts]]).encode("utf-8")
        ).hexdigest()[:12]
        return f"r0_{digest}"

    def _derived_case(self, ppc: Mapping[str, Any], tag: str) -> str:
        text = _render_matpower_case(ppc, f"derived_{tag}")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
        os.makedirs(self.derived_case_dir, exist_ok=True)
        path = self.derived_case_dir / f"{tag}_{digest}.m"
        if not path.is_file():
            path.write_text(text, encoding="utf-8")
        return str(path)

    def _local_artifact(self, referenced_path: str, subdir: str) -> Path:
        basename = os.path.basename(str(referenced_path).replace("\\", "/"))
        local = self.balanced_artifact_dir / subdir / basename
        if not local.is_file():
            raise ScenarioRejected("artifact_missing", str(local))
        return local

    @staticmethod
    def _base_scenario(
        scenario_id: str,
        *,
        case: str,
        measurements: Sequence[float],
        family: str,
    ) -> dict[str, Any]:
        return {
            "scenario_id": scenario_id,
            "root_scenario_id": scenario_id,
            "scenario_family": family,
            "case": case,
            "measurements": [float(value) for value in measurements],
            "semantic_field_provenance": {"measurements": _SNAPSHOT_PROVENANCE},
            "metadata": {},
        }

    # --------------------------------------------------------- base families

    def _no_error_scenario(self, row: Mapping[str, Any], index: int) -> dict[str, Any]:
        scenario = self._base_scenario(
            self._scenario_id("no_error", row.get("id"), index),
            case="case14",
            measurements=row["z_obs"],
            family="no_error",
        )
        self._require_clean("case14", row["z_obs"], "no_error")
        scenario["clean_case"] = "case14"
        scenario["clean_measurements"] = [float(value) for value in row["z_true"]]
        scenario["true_measurement_errors"] = []
        scenario["true_parameter_errors"] = []
        scenario["true_topology_errors"] = []
        return scenario

    def _measurement_scenario(
        self, row: Mapping[str, Any], index: int, *, family: str = "measurement"
    ) -> dict[str, Any]:
        label = dict(row.get("label") or {})
        if label.get("indices") is not None:
            error_indices = [int(value) for value in label["indices"]]
        elif label.get("index") is not None:
            error_indices = [int(label["index"])]
        else:
            raise ScenarioRejected("label_missing_index", str(label))
        z_obs = [float(value) for value in row["z_obs"]]
        z_true = [float(value) for value in row["z_true"]]
        if any(not 0 <= i < len(z_obs) for i in error_indices):
            raise ScenarioRejected("label_index_out_of_range", str(error_indices))
        self._require_anomalous("case14", z_obs, family)
        scenario = self._base_scenario(
            self._scenario_id(family, row.get("id"), index),
            case="case14",
            measurements=z_obs,
            family=family,
        )
        scenario["clean_case"] = "case14"
        scenario["clean_measurements"] = z_true
        scenario["true_measurement_errors"] = [
            {
                "index": error_index,
                "channel": label.get("channel"),
                "observed": z_obs[error_index],
                "clean": z_true[error_index],
            }
            for error_index in error_indices
        ]
        return scenario

    def _parameter_scenario(self, row: Mapping[str, Any], index: int) -> dict[str, Any]:
        label = dict(row.get("label") or {})
        line_row0 = label.get("line_row")
        if line_row0 is None:
            raise ScenarioRejected("label_missing_line_row", str(label))
        line_row0 = int(line_row0)
        if not 0 <= line_row0 < NL:
            raise ScenarioRejected("label_line_row_out_of_range", str(line_row0))
        if not row.get("z_scans") or not row.get("initial_states"):
            raise ScenarioRejected("parameter_scans_missing", str(row.get("id")))
        true_case = self._local_artifact(
            row.get("parameter_error_case_path") or "", "cases_parameter_error"
        )
        z_obs = [float(value) for value in row["z_obs"]]
        # The physical line changed; the agent's model database (case14) is
        # stale.  The measurements must therefore be anomalous under case14 and
        # consistent under the changed-parameter case the corpus generated.
        self._require_anomalous("case14", z_obs, "parameter")
        self._require_clean(str(true_case), z_obs, "parameter")
        true_ppc = _load_python_case(str(true_case))
        scenario = self._base_scenario(
            self._scenario_id("parameter", row.get("id"), index),
            case="case14",
            measurements=z_obs,
            family="parameter",
        )
        scenario["clean_case"] = str(true_case)
        scenario["clean_measurements"] = [float(value) for value in row["z_true"]]
        scenario["true_parameter_errors"] = [
            {
                "branch_row0": line_row0,
                "line_index1": line_row0 + 1,
                "parameter": "rx",
                "clean_r": float(true_ppc["branch"][line_row0][2]),
                "clean_x": float(true_ppc["branch"][line_row0][3]),
                "r_factor": label.get("r_factor"),
                "x_factor": label.get("x_factor"),
                "from_bus": label.get("from_bus"),
                "to_bus": label.get("to_bus"),
            }
        ]
        scenario["metadata"]["parameter_scans"] = {
            "z_scans": [[float(v) for v in scan] for scan in row["z_scans"]],
            "initial_states": [
                [float(v) for v in scan] for scan in row["initial_states"]
            ],
        }
        return scenario

    def _safe_outage_rows(self) -> list[int]:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components

        branch = np.asarray(self._clean_case14()["branch"], dtype=float)
        safe: list[int] = []
        for row0 in range(branch.shape[0]):
            keep = [i for i in range(branch.shape[0]) if i != row0]
            from_bus = branch[keep, 0].astype(int) - 1
            to_bus = branch[keep, 1].astype(int) - 1
            adjacency = coo_matrix(
                (np.ones(len(keep)), (from_bus, to_bus)), shape=(NB, NB)
            )
            components, _ = connected_components(adjacency, directed=False)
            if components == 1:
                safe.append(row0)
        return safe

    def _topology_scenario(self, index: int) -> dict[str, Any]:
        from pypower.api import case14 as pypower_case14
        from pypower.api import ppoption, runpf

        safe_rows = self._safe_outage_rows()
        row0 = int(self._rng.choice(safe_rows))
        load_scale = float(self._rng.uniform(0.85, 1.15))
        truth = pypower_case14()
        truth["bus"][:, 2] *= load_scale
        truth["bus"][:, 3] *= load_scale
        slack_rows = truth["bus"][:, 1] == 3
        slack_buses = set(truth["bus"][slack_rows, 0].astype(int))
        for gen_row in truth["gen"]:
            if int(gen_row[0]) not in slack_buses:
                gen_row[1] *= load_scale
        truth["branch"][row0, 10] = 0
        solution, converged = runpf(truth, ppoption(VERBOSE=0, OUT_ALL=0))
        if not converged:
            raise ScenarioRejected("power_flow_diverged", f"outage row0={row0}")
        z_true = build_measurement_vector(solution)
        noise = self._rng.normal(0.0, self.noise_profile()) * self.topology_noise_scale
        z_obs = (z_true + noise).tolist()

        corrected_ppc = copy.deepcopy(self._clean_case14())
        corrected_ppc["branch"][row0][10] = 0.0
        corrected_case = self._derived_case(corrected_ppc, f"r0_topo_l{row0 + 1}s0")
        self._require_anomalous("case14", z_obs, "topology")
        self._require_clean(corrected_case, z_obs, "topology")

        branch = self._clean_case14()["branch"]
        scenario = self._base_scenario(
            self._scenario_id("topology", row0, load_scale, index),
            case="case14",
            measurements=z_obs,
            family="topology",
        )
        scenario["clean_case"] = corrected_case
        scenario["clean_measurements"] = [float(value) for value in z_true]
        scenario["true_topology_errors"] = [
            {
                "branch_row0": row0,
                "line_index1": row0 + 1,
                "expected_status": 0,
                "from_bus": int(branch[row0][0]),
                "to_bus": int(branch[row0][1]),
                "load_scale": load_scale,
            }
        ]
        return scenario

    def _harmonic_scenario(self, row: Mapping[str, Any], index: int) -> dict[str, Any]:
        label = dict(row.get("label") or {})
        harmonic_measurements = row.get("harmonic_measurements")
        if not harmonic_measurements:
            raise ScenarioRejected("harmonic_measurements_missing", str(row.get("id")))
        z_obs = [float(value) for value in row["z_obs"]]
        # Harmonic distortion corrupts the fundamental snapshot too (corpus
        # rows sit far above the chi-square threshold).  No correction can fix
        # that vector; the episode terminates through the explained-anomaly
        # route once HSE localizes the source, so only solvability is required.
        if self.validate:
            self._chi2_statistic("case14", z_obs)
        scenario = self._base_scenario(
            self._scenario_id("harmonic", row.get("id"), index),
            case="case14",
            measurements=z_obs,
            family="harmonic",
        )
        scenario["clean_case"] = "case14"
        scenario["clean_measurements"] = [float(value) for value in row["z_true"]]
        scenario["unresolved_signatures"] = [HARMONIC_SIGNATURE]
        scenario["semantic_field_provenance"]["unresolved_signatures"] = (
            _POWER_QUALITY_PROVENANCE
        )
        scenario["metadata"]["harmonic_measurements"] = [
            dict(item) for item in harmonic_measurements
        ]
        if row.get("harmonic_orders"):
            scenario["metadata"]["harmonic_orders"] = [
                int(order) for order in row["harmonic_orders"]
            ]
        scenario["hidden_truth"] = {
            "true_harmonic_errors": [
                {
                    "bus_1based": label.get("source_bus"),
                    "thd_target": label.get("thd_target"),
                }
            ]
        }
        return scenario

    def _hif_scenario(self, row: Mapping[str, Any], index: int) -> dict[str, Any]:
        label = dict(row.get("label") or {})
        diagnostic = row.get("nlm_diagnostic")
        scans = row.get("scans")
        if not isinstance(diagnostic, Mapping) or not diagnostic.get("success"):
            raise ScenarioRejected("nlm_diagnostic_missing", str(row.get("id")))
        if not scans:
            raise ScenarioRejected("hif_scans_missing", str(row.get("id")))
        z_obs = [float(value) for value in row["z_obs"]]
        if self.validate:
            self._chi2_statistic("case14", z_obs)  # must solve; may be subtle
        if len(scans) > self.hif_max_scans:
            picks = np.linspace(0, len(scans) - 1, self.hif_max_scans).round().astype(int)
            scans = [scans[int(i)] for i in dict.fromkeys(picks.tolist())]
        scenario = self._base_scenario(
            self._scenario_id("hif", row.get("id"), index),
            case="case14",
            measurements=z_obs,
            family="hif",
        )
        scenario["clean_case"] = "case14"
        scenario["clean_measurements"] = [float(value) for value in row["z_true"]]
        scenario["unresolved_signatures"] = [HIF_SIGNATURE]
        scenario["semantic_field_provenance"]["unresolved_signatures"] = (
            _WAVEFORM_PROVENANCE
        )
        scenario["metadata"]["nlm_diagnostic"] = copy.deepcopy(dict(diagnostic))
        scenario["metadata"]["hif_runtime"] = {
            "z_obs": z_obs,
            "three_phase_voltages": copy.deepcopy(row.get("three_phase_voltages")),
            "load_scale": float((row.get("op_point") or {}).get("load_scale", 1.0)),
        }
        scenario["metadata"]["hif_scan_window"] = {
            "scan_window_path": str(row.get("id") or scenario["scenario_id"]),
            "scans": copy.deepcopy(list(scans)),
            "sigma_z": copy.deepcopy(row.get("sigma_z")),
        }
        scenario["hidden_truth"] = {"true_hif_errors": [copy.deepcopy(label)]}
        return scenario

    # ---------------------------------------------------------- compositions

    def _overlay_indices(self, scenario: Mapping[str, Any], count: int) -> list[int]:
        """Indices eligible for a gross-offset overlay on this scenario."""
        index_map = measurement_index_map(NB, NL)
        blocked: set[int] = set()
        for fault in scenario.get("true_measurement_errors") or []:
            if fault.get("index") is not None:
                blocked.add(int(fault["index"]))
        for fault in scenario.get("true_topology_errors") or []:
            row0 = fault.get("branch_row0")
            if row0 is None:
                continue
            for channel in ("Pf", "Qf", "Pt", "Qt"):
                blocked.add(index_map[channel].start + int(row0))
        eligible = [
            candidate
            for channel in ("Pinj", "Qinj", "Pf", "Qf", "Pt", "Qt")
            for candidate in range(index_map[channel].start, index_map[channel].stop)
            if candidate not in blocked
        ]
        picked = self._rng.choice(len(eligible), size=count, replace=False)
        return sorted(int(eligible[int(i)]) for i in picked)

    def _compose_measurement(
        self,
        scenario: dict[str, Any],
        *,
        offsets: int,
        family: str,
        index: int,
    ) -> dict[str, Any]:
        composed = copy.deepcopy(scenario)
        composed["scenario_family"] = family
        composed["scenario_id"] = self._scenario_id(family, scenario["scenario_id"], index)
        composed["root_scenario_id"] = composed["scenario_id"]
        measurements = [float(value) for value in composed["measurements"]]
        errors = list(composed.get("true_measurement_errors") or [])
        for overlay_index in self._overlay_indices(composed, offsets):
            magnitude = float(self._rng.uniform(0.10, 0.30))
            sign = 1.0 if self._rng.random() < 0.5 else -1.0
            clean_value = measurements[overlay_index]
            measurements[overlay_index] = clean_value + sign * magnitude
            errors.append(
                {
                    "index": overlay_index,
                    "observed": measurements[overlay_index],
                    "clean": clean_value,
                }
            )
        composed["measurements"] = measurements
        composed["true_measurement_errors"] = errors
        self._require_anomalous("case14", measurements, family)
        return composed

    # ---------------------------------------------------------------- driver

    def build(self, plan: Mapping[str, int]) -> list[dict[str, Any]]:
        """Build scenarios per the family->count plan, skipping invalid rows."""
        unknown = sorted(set(plan) - set(SCENARIO_FAMILIES))
        if unknown:
            raise ValueError(f"Unknown scenario families: {unknown}")
        scenarios: list[dict[str, Any]] = []
        for family in SCENARIO_FAMILIES:
            count = int(plan.get(family, 0))
            if count <= 0:
                continue
            scenarios.extend(self._build_family(family, count))
        return scenarios

    def _build_family(self, family: str, count: int) -> list[dict[str, Any]]:
        built: list[dict[str, Any]] = []
        attempts = 0

        def record(scenario: dict[str, Any], source: str) -> None:
            self.manifest.append(
                {
                    "scenario_id": scenario["scenario_id"],
                    "scenario_family": family,
                    "source": source,
                }
            )
            built.append(scenario)

        if family == "topology":
            while len(built) < count and attempts < count * 8:
                attempts += 1
                try:
                    record(self._topology_scenario(attempts), "synthesized_pypower")
                except ScenarioRejected as rejection:
                    self._record_skip(family, f"synthesized_{attempts}", rejection)
            return built

        if family == "measurement+topology":
            while len(built) < count and attempts < count * 8:
                attempts += 1
                try:
                    base = self._topology_scenario(1000 + attempts)
                    record(
                        self._compose_measurement(
                            base, offsets=1, family=family, index=attempts
                        ),
                        "synthesized_pypower",
                    )
                except ScenarioRejected as rejection:
                    self._record_skip(family, f"synthesized_{attempts}", rejection)
            return built

        source_rows, builder = self._family_source(family)
        order = self._rng.permutation(len(source_rows))
        for position in order:
            if len(built) >= count:
                break
            row = source_rows[int(position)]
            source = str(row.get("id") or position)
            attempts += 1
            try:
                record(builder(row, attempts), source)
            except ScenarioRejected as rejection:
                self._record_skip(family, source, rejection)
        return built

    @staticmethod
    def _measurement_subtype(row: Mapping[str, Any]) -> str:
        return str((row.get("label") or {}).get("subtype") or "")

    def _family_source(self, family: str):
        corpus = self._corpus()
        if family == "no_error":
            return corpus.get("no_error", []), self._no_error_scenario
        if family == "measurement":
            rows = [
                row
                for row in corpus.get("measurement_error", [])
                if self._measurement_subtype(row) != "multi_gross_outliers"
            ]
            return rows, self._measurement_scenario
        if family == "multi_measurement":
            rows = [
                row
                for row in corpus.get("measurement_error", [])
                if self._measurement_subtype(row) == "multi_gross_outliers"
            ]

            def multi_builder(row: Mapping[str, Any], index: int) -> dict[str, Any]:
                return self._measurement_scenario(row, index, family="multi_measurement")

            return rows, multi_builder
        if family == "parameter":
            return corpus.get("parameter_error", []), self._parameter_scenario
        if family == "harmonic":
            return corpus.get("harmonic_anomaly", []), self._harmonic_scenario
        if family == "hif":
            return self._hif_rows(), self._hif_scenario
        if family == "measurement+parameter":

            def parameter_builder(row: Mapping[str, Any], index: int) -> dict[str, Any]:
                base = self._parameter_scenario(row, index)
                return self._compose_measurement(
                    base, offsets=1, family=family, index=index
                )

            return corpus.get("parameter_error", []), parameter_builder
        if family == "measurement+hif":

            def hif_builder(row: Mapping[str, Any], index: int) -> dict[str, Any]:
                base = self._hif_scenario(row, index)
                return self._compose_measurement(
                    base, offsets=1, family=family, index=index
                )

            return self._hif_rows(), hif_builder
        raise ValueError(f"Unknown scenario family: {family}")

    def _record_skip(
        self, family: str, source: str, rejection: ScenarioRejected
    ) -> None:
        self.skipped.append(
            {
                "scenario_family": family,
                "source": source,
                "reason": rejection.reason,
                "detail": str(rejection),
            }
        )

    def report(self) -> dict[str, Any]:
        family_counts: dict[str, int] = {}
        for entry in self.manifest:
            family_counts[entry["scenario_family"]] = (
                family_counts.get(entry["scenario_family"], 0) + 1
            )
        skip_reasons: dict[str, int] = {}
        for entry in self.skipped:
            skip_reasons[entry["reason"]] = skip_reasons.get(entry["reason"], 0) + 1
        return {
            "seed": self.seed,
            "chi2_alpha": self.chi2_alpha,
            "chi2_limit": self.chi2_limit,
            "built_by_family": dict(sorted(family_counts.items())),
            "skipped_by_reason": dict(sorted(skip_reasons.items())),
            "skipped": list(self.skipped),
        }


__all__ = [
    "Round0ScenarioGenerator",
    "ScenarioRejected",
    "SCENARIO_FAMILIES",
    "BASE_FAMILIES",
    "COMPOSED_FAMILIES",
    "build_measurement_vector",
    "DEFAULT_CORPUS_PATH",
    "DEFAULT_HIF_SAMPLE_PATHS",
    "DEFAULT_BALANCED_ARTIFACT_DIR",
]
