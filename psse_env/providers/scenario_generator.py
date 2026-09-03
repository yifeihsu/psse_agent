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
import math
import os
import tempfile
from decimal import Decimal, ROUND_HALF_EVEN, localcontext
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from mcp_server.matpower_server import (  # noqa: E402  (repo-root package)
    _load_python_case,
    _param_correction_json,
    _wls_json,
)
from tools.lagrangian_correct_port import make_ybus  # noqa: E402
from trace_protocol import chi2_threshold  # noqa: E402
from three_phase_nlm.branch_current_analysis import (  # noqa: E402
    BRANCH_CURRENT_CHANNEL,
    BRANCH_CURRENT_SIGMA_KEY,
    DEFAULT_UNBALANCE_VUF_THRESHOLD,
    balanced_branch_current_control,
    branch_current_rows_to_phasors,
    line_differential_null_test,
    unbalance_source_localization,
    voltage_unbalance_factors,
)

from psse_env.actions import (
    CORRECT_MEASUREMENTS,
    CORRECT_TOPOLOGY,
    GET_MEASUREMENT_CONTEXT,
    GET_TOPOLOGY_CONTEXT,
    normalize_action,
)
from psse_env.oracle.candidate_quality import CandidateQualityOracle
from psse_env.providers.matpower import (
    MatpowerDeploymentProviders,
    PARAMETER_RANKING_CONTRACT,
    PARAMETER_RANKING_DOMINANCE_THRESHOLD,
    _render_matpower_case,
    measurement_index_map,
    observable_parameter_initial_states,
    parameter_ranking_contract_is_dominant,
)
from psse_env.state_store import _state_content_hash, apply_modification

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHI2_ALPHA = 0.01

SYNTHESIZED_MEASUREMENT_CANONICALIZATION_CONTRACT = (
    "bc0_synthesized_measurement_decimal12_half_even_v1"
)
_SYNTHESIZED_MEASUREMENT_QUANTUM = Decimal("1e-12")

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
DEFAULT_HIF_FALLBACK_SAMPLE_PATHS = (
    _REPO_ROOT
    / "artifacts"
    / "measurements"
    / "out_measurements_hif_representative_20260705_curated17"
    / "samples.jsonl",
)
# BC0 reserves the compact curated fallback above for the frozen evaluation
# holdout. The release aggregate trains on this independently generated,
# physics-validated multiscan population so no HIF physical root crosses the
# train/evaluation boundary.
DEFAULT_RELEASE_HIF_SAMPLE_PATHS = (
    _REPO_ROOT
    / "artifacts"
    / "measurements"
    / "hif_multiscan_benchmark_fixed_diverse_17x20_20260714"
    / "samples.jsonl",
)
DEFAULT_RELEASE_HIF_QUALITY_PATHS = (
    DEFAULT_RELEASE_HIF_SAMPLE_PATHS[0].with_name("meta.json"),
    DEFAULT_RELEASE_HIF_SAMPLE_PATHS[0].with_name("quality_report.json"),
)
# The 2026-07 imbalance corpus carries an unlabeled second unbalance at bus 3
# (the checked-in OpenDSS load file splits bus 3 unevenly and the old generator
# only scaled it; see docs/branch_current_telemetry_20260903.md).  It is kept
# as a legacy reference only and is never a generator default.
LEGACY_IMBALANCE_SAMPLE_PATH = (
    _REPO_ROOT
    / "artifacts"
    / "measurements"
    / "out_measurements_imbalance"
    / "samples.jsonl"
)
DEFAULT_IMBALANCE_SAMPLE_PATH = (
    _REPO_ROOT
    / "artifacts"
    / "measurements"
    / "out_measurements_imbalance_currents_20260903"
    / "samples.jsonl"
)
CURRENT_TELEMETRY_IMBALANCE_SAMPLE_PATH = DEFAULT_IMBALANCE_SAMPLE_PATH
# Per-phase branch-current HIF corpora.  The 85-window pool (seed 20260904,
# five windows per eligible line, strict-physics QA passed) is the research
# DAgger training/development source; the 17-window corpus is the validated
# reference from the telemetry revision.  Neither is a BC0 release input: the
# frozen release paths above stay voltage-only.
CURRENT_TELEMETRY_HIF_SAMPLE_PATHS = (
    _REPO_ROOT
    / "artifacts"
    / "measurements"
    / "hif_multiscan_currents_train_85x10_20260903"
    / "samples.jsonl",
    _REPO_ROOT
    / "artifacts"
    / "measurements"
    / "hif_multiscan_currents_17x10_20260903"
    / "samples.jsonl",
)
DEFAULT_BALANCED_ARTIFACT_DIR = (
    _REPO_ROOT / "artifacts" / "measurements" / "out_measurements_balanced"
)

NB, NL = 14, 20
NZ = 3 * NB + 4 * NL
_STATE_COUNT = 2 * NB - 1
_PARAMETER_RANKING_METRIC_KEYS = (
    "parameter_ranking_contract",
    "parameter_ranking_distinct_lines",
    "parameter_ranking_top_abs_lambda",
    "parameter_ranking_runner_up_abs_lambda",
    "parameter_ranking_dominance_ratio",
    "parameter_ranking_dominance_threshold",
    "parameter_ranking_singleton",
    "parameter_ranking_dominant",
)

BASE_FAMILIES = (
    "no_error",
    "measurement",
    "parameter",
    "topology",
    "harmonic",
    "hif",
    "three_phase_unbalance",
    "telemetry_no_disturbance",
)
COMPOSED_FAMILIES = (
    "multi_measurement",
    "measurement+parameter",
    "measurement+topology",
    "measurement+hif",
)
# Preserve the established seeded order for existing families; new diagnostic
# controls are appended so adding their source corpus cannot perturb earlier
# mixed-error selections.
SCENARIO_FAMILIES = (
    "no_error",
    "measurement",
    "parameter",
    "topology",
    "harmonic",
    "hif",
    *COMPOSED_FAMILIES,
    "three_phase_unbalance",
    "telemetry_no_disturbance",
)

_SNAPSHOT_PROVENANCE = "deployment_sensor:scada_snapshot"
_POWER_QUALITY_PROVENANCE = "deployment_sensor:power_quality"
_WAVEFORM_PROVENANCE = "deployment_sensor:waveform_capture"

HARMONIC_SIGNATURE = "harmonic_distortion_detected"
HIF_SIGNATURE = "hif_suspected_zero_sequence"
# Unbalance sensor signatures are policy-visible text, so each one is emitted
# only when the row's telemetry actually shows it: the VUF flag when the
# largest bus VUF clears the shared deployment gate, and the current-spread
# flag when the branch-current channel exposes a noise-significant unbalance
# source with a quiet line-differential null.
UNBALANCE_SIGNATURE = "three_phase_unbalance vuf_threshold_exceeded"
UNBALANCE_CURRENT_SIGNATURE = "three_phase_unbalance phase_current_spread_detected"

# The tabular measurement corpus is shared by both the round-0 aggregate and
# the frozen evaluation-suite builder.  Assign its physical source rows before
# seeded sampling so changing a corpus row id (or copying the same physical row
# under a second id) cannot move that realization across the release boundary.
_SOURCE_PARTITION_FAMILIES = (
    "no_error",
    "measurement",
    "multi_measurement",
    "parameter",
    "harmonic",
    "measurement+parameter",
)
_SOURCE_PARTITION_CORPUS_SCENARIOS = frozenset(
    {"no_error", "measurement_error", "parameter_error", "harmonic_anomaly"}
)
_SOURCE_PARTITION_MODULUS = 5
_SOURCE_PARTITION_EVALUATION_BUCKETS = frozenset({0})
_SOURCE_PARTITION_ALGORITHM = "sha256_physical_content_modulo_v1"
_SOURCE_NONPHYSICAL_TOP_LEVEL_KEYS = frozenset(
    {
        "id",
        "example_id",
        "scenario_id",
        "root_scenario_id",
    }
)

_EXPLANATION_ONLY_RELEASE_AUDIT = {
    "explanation_only_contract": "explanation_only_diagnostic_localization_v1",
    "not_applicable": {
        "final_measurements_match_clean": (
            "The diagnostic resolves an explanation-only waveform anomaly; "
            "it does not rewrite the fundamental measurement snapshot."
        )
    }
}


class ScenarioRejected(RuntimeError):
    """A source row cannot become a physically validated scenario."""

    def __init__(
        self,
        reason: str,
        detail: str = "",
        *,
        metrics: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.metrics = dict(metrics or {})


def _canonicalize_synthesized_measurement_vector(
    values: Sequence[float],
) -> list[float]:
    """Project admitted PYPOWER telemetry onto its release decimal lattice.

    PYPOWER and its linear-algebra dependencies can differ by a few final
    binary64 bits across otherwise approved hosts.  Release-suite bytes and
    physical-root fingerprints are pinned exactly, so persist synthesized
    topology telemetry on a lattice far finer than any recovery tolerance.
    """

    canonical: list[float] = []
    # A binary64 value can have 309 integer digits. Quantizing any finite
    # binary64 value to twelve fractional places therefore needs at most 321
    # significant decimal digits.
    with localcontext() as context:
        context.prec = 400
        for index, value in enumerate(values):
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError(
                    "synthesized measurement must be finite: "
                    f"index={index}, value={numeric!r}"
                )
            quantized = Decimal.from_float(numeric).quantize(
                _SYNTHESIZED_MEASUREMENT_QUANTUM,
                rounding=ROUND_HALF_EVEN,
            )
            normalized = float(quantized)
            canonical.append(0.0 if normalized == 0.0 else normalized)
    return canonical


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
        imbalance_sample_path: str | Path | None = None,
        balanced_artifact_dir: str | Path | None = None,
        artifact_allowlist: Sequence[str | Path] | None = None,
        derived_case_dir: str | Path | None = None,
        seed: int = 20260719,
        validate: bool = True,
        chi2_alpha: float = DEFAULT_CHI2_ALPHA,
        anomaly_margin: float = 1.25,
        topology_noise_scale: float = 0.8,
        hif_max_scans: int = 8,
        noise_profile_rows: int = 200,
        source_partition: str | None = None,
        parameter_ranking_dominance_threshold: float | None = None,
        unbalance_vuf_threshold: float = DEFAULT_UNBALANCE_VUF_THRESHOLD,
    ) -> None:
        if source_partition not in (None, "train", "evaluation"):
            raise ValueError(
                "source_partition must be None, 'train', or 'evaluation'"
            )
        self.corpus_path = Path(corpus_path or DEFAULT_CORPUS_PATH)
        uses_default_hif_paths = hif_sample_paths is None
        self.hif_sample_paths = [
            Path(path)
            for path in (
                DEFAULT_HIF_SAMPLE_PATHS
                if uses_default_hif_paths
                else hif_sample_paths
            )
        ]
        self.hif_fallback_sample_paths = (
            [Path(path) for path in DEFAULT_HIF_FALLBACK_SAMPLE_PATHS]
            if uses_default_hif_paths
            else []
        )
        self.imbalance_sample_path = Path(
            imbalance_sample_path or DEFAULT_IMBALANCE_SAMPLE_PATH
        )
        self.balanced_artifact_dir = Path(
            balanced_artifact_dir or DEFAULT_BALANCED_ARTIFACT_DIR
        )
        self.artifact_allowlist = (
            {Path(path).absolute() for path in artifact_allowlist}
            if artifact_allowlist is not None
            else None
        )
        self.consumed_artifacts: set[Path] = set()
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
        self.source_partition = source_partition
        if not (0.0 < float(unbalance_vuf_threshold) < 1.0):
            raise ValueError("unbalance_vuf_threshold must be a fraction in (0, 1)")
        self.unbalance_vuf_threshold = float(unbalance_vuf_threshold)
        # The frozen evaluation suite deliberately preserves its previously
        # approved physical roots, including hard/ambiguous parameter cases.
        # Dominance is a single-label *training admission* requirement, not a
        # reason to rewrite a pinned holdout.  The evaluation builder therefore
        # uses the legacy rank-one inventory threshold while train/unspecified
        # generation enforces the release threshold below.
        self._enforce_parameter_ranking_dominance = (
            source_partition != "evaluation"
        )
        default_parameter_threshold = (
            1.0
            if source_partition == "evaluation"
            else PARAMETER_RANKING_DOMINANCE_THRESHOLD
        )
        self.parameter_ranking_dominance_threshold = float(
            default_parameter_threshold
            if parameter_ranking_dominance_threshold is None
            else parameter_ranking_dominance_threshold
        )
        if (
            not math.isfinite(self.parameter_ranking_dominance_threshold)
            or self.parameter_ranking_dominance_threshold < 1.0
        ):
            raise ValueError(
                "parameter_ranking_dominance_threshold must be finite and >= 1.0"
            )
        self._source_partition_metadata: dict[str, Any] = {
            "selected": source_partition,
            "enabled": source_partition is not None,
            "algorithm": (
                _SOURCE_PARTITION_ALGORITHM
                if source_partition is not None
                else "disabled"
            ),
            "eligible_families": list(_SOURCE_PARTITION_FAMILIES),
            "modulus": (
                _SOURCE_PARTITION_MODULUS
                if source_partition is not None
                else None
            ),
            "evaluation_buckets": (
                sorted(_SOURCE_PARTITION_EVALUATION_BUCKETS)
                if source_partition is not None
                else []
            ),
            "corpus_loaded": False,
            "rows_total_by_corpus_scenario": {},
            "rows_selected_by_corpus_scenario": {},
            "physical_groups_total_by_corpus_scenario": {},
            "physical_groups_selected_by_corpus_scenario": {},
        }
        self._parameter_gate_provider = MatpowerDeploymentProviders(
            chi2_alpha=self.chi2_alpha,
            derived_case_dir=str(self.derived_case_dir),
            parameter_ranking_dominance_threshold=(
                self.parameter_ranking_dominance_threshold
            ),
        )
        self._parameter_gate_candidate_oracle = CandidateQualityOracle(
            mode="deployment"
        )
        self._parameter_gate_results: dict[str, dict[str, Any]] = {}

        self.manifest: list[dict[str, Any]] = []
        self.skipped: list[dict[str, Any]] = []
        self._rng = np.random.default_rng(self.seed)
        self._corpus_by_class: dict[str, list[dict[str, Any]]] | None = None
        self._hif_samples: list[dict[str, Any]] | None = None
        self._imbalance_samples: list[dict[str, Any]] | None = None
        self._hif_order_population_size: int | None = None
        self._noise_std: np.ndarray | None = None
        self._case14: dict[str, Any] | None = None
        self._chi2_cache: dict[tuple[str, str], float] = {}

    # ------------------------------------------------------------ data access

    @staticmethod
    def _source_physical_digest(row: Mapping[str, Any]) -> str:
        """Return a stable digest of raw physical content, excluding aliases.

        The measurements, operating point, injected-error label, repeated
        scans, and any diagnostic telemetry remain in the payload.  Row ids
        and staging paths do not: they identify packaging rather than a
        physical realization.  The shared corpus already carries the physical
        parameter realization in its telemetry and label, so a renamed copy of
        the corresponding case file must not cross the partition boundary.
        """

        physical = {
            str(key): value
            for key, value in row.items()
            if str(key) not in _SOURCE_NONPHYSICAL_TOP_LEVEL_KEYS
            and not str(key).endswith(("_path", "_dir"))
        }
        encoded = json.dumps(
            {
                "partition_schema_version": 1,
                "physical_source": physical,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def _row_source_partition(cls, row: Mapping[str, Any]) -> str:
        return cls._digest_source_partition(cls._source_physical_digest(row))

    @staticmethod
    def _digest_source_partition(digest: str) -> str:
        raw_digest = bytes.fromhex(digest)
        bucket = (
            int.from_bytes(raw_digest[:8], byteorder="big")
            % _SOURCE_PARTITION_MODULUS
        )
        if bucket in _SOURCE_PARTITION_EVALUATION_BUCKETS:
            return "evaluation"
        return "train"

    def _corpus(self) -> dict[str, list[dict[str, Any]]]:
        if self._corpus_by_class is None:
            grouped: dict[str, list[dict[str, Any]]] = {}
            total_rows: dict[str, int] = {}
            selected_rows: dict[str, int] = {}
            total_groups: dict[str, set[str]] = {}
            selected_groups: dict[str, set[str]] = {}
            with open(self.corpus_path, encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    corpus_scenario = str(row.get("scenario"))
                    if corpus_scenario in _SOURCE_PARTITION_CORPUS_SCENARIOS:
                        total_rows[corpus_scenario] = (
                            total_rows.get(corpus_scenario, 0) + 1
                        )
                        selected_rows.setdefault(corpus_scenario, 0)
                        if self.source_partition is not None:
                            digest = self._source_physical_digest(row)
                            total_groups.setdefault(corpus_scenario, set()).add(digest)
                            selected_groups.setdefault(corpus_scenario, set())
                            assigned_partition = self._digest_source_partition(digest)
                            if assigned_partition != self.source_partition:
                                continue
                            selected_groups.setdefault(corpus_scenario, set()).add(
                                digest
                            )
                        selected_rows[corpus_scenario] += 1
                    grouped.setdefault(corpus_scenario, []).append(row)
            self._source_partition_metadata.update(
                {
                    "corpus_loaded": True,
                    "rows_total_by_corpus_scenario": dict(sorted(total_rows.items())),
                    "rows_selected_by_corpus_scenario": dict(
                        sorted(selected_rows.items())
                    ),
                    "physical_groups_total_by_corpus_scenario": {
                        key: len(value) for key, value in sorted(total_groups.items())
                    },
                    "physical_groups_selected_by_corpus_scenario": {
                        key: len(value)
                        for key, value in sorted(selected_groups.items())
                    },
                }
            )
            self._corpus_by_class = grouped
        return self._corpus_by_class

    def _hif_rows(self) -> list[dict[str, Any]]:
        if self._hif_samples is None:
            rows: list[dict[str, Any]] = []
            used_fallback = False
            for path in self.hif_sample_paths:
                if not path.is_file():
                    continue
                with open(path, encoding="utf-8") as handle:
                    rows.extend(json.loads(line) for line in handle)
            if not rows:
                used_fallback = True
                for path in self.hif_fallback_sample_paths:
                    if not path.is_file():
                        continue
                    with open(path, encoding="utf-8") as handle:
                        rows.extend(json.loads(line) for line in handle)
            self._hif_samples = [self._normalize_hif_row(row) for row in rows]
            # Both primary corpora cover the same 17 HIF branches.  Preserve
            # their combined ordering population when the single tracked
            # fallback corpus is used, so later families see the same seeded
            # random stream without duplicating physical fallback scenarios.
            source_sets = len(self.hif_sample_paths) if used_fallback else 1
            self._hif_order_population_size = len(self._hif_samples) * source_sets
        return self._hif_samples

    def _imbalance_rows(self) -> list[dict[str, Any]]:
        if self._imbalance_samples is None:
            rows: list[dict[str, Any]] = []
            if self.imbalance_sample_path.is_file():
                with open(self.imbalance_sample_path, encoding="utf-8") as handle:
                    rows.extend(json.loads(line) for line in handle)
            self._imbalance_samples = rows
        return self._imbalance_samples

    @staticmethod
    def _normalize_hif_row(row: Mapping[str, Any]) -> dict[str, Any]:
        """Promote a legacy HIF snapshot to the current scan-window schema.

        The tracked representative HIF corpus predates persistent multiscan
        windows.  It remains a physically generated snapshot with NLM and
        three-phase evidence, so a clean checkout can safely expose it as a
        one-scan window when the larger benchmark artifacts are unavailable.
        """
        normalized = copy.deepcopy(dict(row))
        if normalized.get("scans"):
            return normalized
        required = ("z_obs", "z_true", "three_phase_voltages")
        if any(not normalized.get(key) for key in required):
            return normalized
        topology_id = str(normalized.get("topology_id") or "ieee14_base")
        promoted_scan = {
            "scan_index": 0,
            "z_clean": copy.deepcopy(normalized["z_true"]),
            "z_obs": copy.deepcopy(normalized["z_obs"]),
            "three_phase_voltages": copy.deepcopy(
                normalized["three_phase_voltages"]
            ),
            "op_point": copy.deepcopy(normalized.get("op_point") or {}),
            "topology_id": topology_id,
        }
        if normalized.get(BRANCH_CURRENT_CHANNEL):
            promoted_scan[BRANCH_CURRENT_CHANNEL] = copy.deepcopy(
                normalized[BRANCH_CURRENT_CHANNEL]
            )
            if normalized.get(BRANCH_CURRENT_SIGMA_KEY) is not None:
                promoted_scan[BRANCH_CURRENT_SIGMA_KEY] = normalized[BRANCH_CURRENT_SIGMA_KEY]
        normalized["scans"] = [promoted_scan]
        normalized["scan_count"] = 1
        normalized["topology_id"] = topology_id
        normalized["window_metadata"] = {
            "source_kind": "tracked_single_scan_fallback",
        }
        return normalized

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

    def _require_parameter_correction_realizable(
        self,
        *,
        line_row0: int,
        clean_r: float,
        clean_x: float,
        z_scans: Sequence[Sequence[float]],
        measurements: Sequence[float],
        final_case_abs_tolerance: float,
    ) -> dict[str, Any] | None:
        """Truth-side gate for the deployed multi-scan parameter corrector.

        Release scenarios must be recoverable by the same correction entry
        point that the transactional environment invokes.  The estimator is
        evaluated only while constructing/auditing scenarios; its output and
        comparison with the clean target are never attached to an accepted
        scenario or exposed to the online policy.
        """
        if not self.validate:
            return None
        line_index1 = int(line_row0) + 1
        normalized_scans = [
            [float(value) for value in scan] for scan in z_scans
        ]
        initial_states = observable_parameter_initial_states(
            self._clean_case14(), normalized_scans
        )
        payload = _param_correction_json(
            "case14",
            line_index1,
            normalized_scans,
            initial_states,
        )
        success = bool(payload.get("success"))
        corrected = payload.get("corrected_params") or []
        metrics: dict[str, Any] = {
            "line_index1": line_index1,
            "solver_success": success,
            "final_case_abs_tolerance": float(final_case_abs_tolerance),
        }
        if not success or len(corrected) < 2:
            error = payload.get("error")
            if error:
                metrics["solver_error"] = str(error)
            raise ScenarioRejected(
                "parameter_correction_unrealizable",
                f"line {line_index1}: configured multi-scan corrector did not converge",
                metrics=metrics,
            )

        try:
            corrected_r = float(corrected[0])
            corrected_x = float(corrected[1])
        except (TypeError, ValueError, OverflowError) as exc:
            metrics["solver_error"] = f"invalid corrected parameters: {type(exc).__name__}"
            raise ScenarioRejected(
                "parameter_correction_unrealizable",
                f"line {line_index1}: configured multi-scan corrector returned invalid parameters",
                metrics=metrics,
            ) from exc
        if not np.isfinite(corrected_r) or not np.isfinite(corrected_x):
            metrics["solver_error"] = "non-finite corrected parameters"
            raise ScenarioRejected(
                "parameter_correction_unrealizable",
                f"line {line_index1}: configured multi-scan corrector returned non-finite parameters",
                metrics=metrics,
            )

        r_error = abs(corrected_r - float(clean_r))
        x_error = abs(corrected_x - float(clean_x))
        r_limit = float(final_case_abs_tolerance) + 1e-9 * abs(float(clean_r))
        x_limit = float(final_case_abs_tolerance) + 1e-9 * abs(float(clean_x))
        metrics.update(
            {
                "corrected_r": corrected_r,
                "corrected_x": corrected_x,
                "clean_r": float(clean_r),
                "clean_x": float(clean_x),
                "r_abs_error": r_error,
                "x_abs_error": x_error,
                "max_abs_error": max(r_error, x_error),
            }
        )
        if r_error > r_limit or x_error > x_limit:
            raise ScenarioRejected(
                "parameter_correction_outside_release_tolerance",
                (
                    f"line {line_index1}: max abs error {max(r_error, x_error):.6g} "
                    f"> declared final-case tolerance {final_case_abs_tolerance:.6g}"
                ),
                metrics=metrics,
            )

        corrected_case = copy.deepcopy(self._clean_case14())
        corrected_case["branch"][line_row0][2] = corrected_r
        corrected_case["branch"][line_row0][3] = corrected_x
        corrected_case_path = self._derived_case(
            corrected_case, f"parameter_gate_l{line_index1}"
        )
        candidate_metrics = self._parameter_gate_provider.run_wls(
            {
                "state_id": f"offline_parameter_gate:l{line_index1}",
                "status": "candidate",
                "source_action": {
                    "tool": "correct_parameters",
                    "arguments": {
                        "state_id": "offline_parameter_gate:parent",
                        "line_index": line_index1,
                    },
                },
                "case": corrected_case_path,
                "measurements": [float(value) for value in measurements],
                "policy_observation": {},
            }
        )
        if candidate_metrics.get("execution_status") == "failure":
            metrics["verification_error_code"] = candidate_metrics.get("error_code")
            if candidate_metrics.get("error_detail"):
                metrics["verification_error_detail"] = str(
                    candidate_metrics["error_detail"]
                )
            raise ScenarioRejected(
                "parameter_correction_verification_failed",
                f"line {line_index1}: post-correction WLS failed",
                metrics=metrics,
            )
        metrics.update(
            {
                "target_fixed": candidate_metrics.get("target_fixed"),
                "target_metric_value": candidate_metrics.get("target_metric_value"),
                "target_metric_threshold": candidate_metrics.get(
                    "target_metric_threshold"
                ),
                "chi_square_statistic": candidate_metrics.get(
                    "chi_square_statistic"
                ),
                "chi_square_threshold": candidate_metrics.get(
                    "chi_square_threshold"
                ),
                "post_action_resolved": candidate_metrics.get(
                    "post_action_resolved"
                ),
                "globally_resolved": candidate_metrics.get("globally_resolved"),
                "physical_constraints_ok": candidate_metrics.get(
                    "physical_constraints_ok"
                ),
                "physical_evidence_scope": candidate_metrics.get(
                    "physical_evidence_scope"
                ),
            }
        )
        if not (
            candidate_metrics.get("target_fixed") is True
            and candidate_metrics.get("post_action_resolved") is True
            and candidate_metrics.get("globally_resolved") is True
            and candidate_metrics.get("physical_constraints_ok") is True
        ):
            raise ScenarioRejected(
                "parameter_correction_candidate_unresolved",
                (
                    f"line {line_index1}: configured candidate criteria failed "
                    f"(target_fixed={candidate_metrics.get('target_fixed')}, "
                    f"globally_clean={candidate_metrics.get('post_action_resolved')}, "
                    "physical_ok="
                    f"{candidate_metrics.get('physical_constraints_ok')})"
                ),
                metrics=metrics,
            )

        # Directly applying the truth-selected correction above proves that a
        # root *can* be repaired, but release eligibility also requires the
        # deployed observable route to select that target.  A healthy line can
        # absorb enough of a confounded residual pattern to pass the partial
        # candidate gate before the true line is tried.  The ensuing cleanup
        # may then mask the still-present physical fault.  Exercise the exact
        # production parameter-context provider on the stale root and require
        # its first supported correction to name the declared line.  This is a
        # truth-side construction gate only: neither the expected target nor
        # this comparison is attached to the scenario seen by the policy.
        context_state_id = f"offline_parameter_context:l{line_index1}"
        context_metrics = self._parameter_gate_provider.get_parameter_context(
            {
                "state_id": context_state_id,
                "case": "case14",
                "measurements": [float(value) for value in measurements],
                "metadata": {
                    "parameter_scans": {
                        "z_scans": copy.deepcopy(normalized_scans),
                    }
                },
                "policy_observation": {},
            }
        )
        if context_metrics.get("execution_status") == "failure":
            metrics["parameter_context_error_code"] = context_metrics.get(
                "error_code"
            )
            if context_metrics.get("error_detail"):
                metrics["parameter_context_error_detail"] = str(
                    context_metrics["error_detail"]
                )
            raise ScenarioRejected(
                "parameter_context_target_unavailable",
                f"line {line_index1}: deployed parameter context failed",
                metrics=metrics,
            )

        parameter_ranking = {
            key: copy.deepcopy(context_metrics.get(key))
            for key in _PARAMETER_RANKING_METRIC_KEYS
        }
        metrics["parameter_context_ranking"] = parameter_ranking
        if (
            self._enforce_parameter_ranking_dominance
            and not parameter_ranking_contract_is_dominant(
                context_metrics,
                expected_threshold=self.parameter_ranking_dominance_threshold,
            )
        ):
            raise ScenarioRejected(
                "parameter_context_target_not_dominant",
                (
                    f"line {line_index1}: deployed parameter context did not "
                    f"satisfy {PARAMETER_RANKING_CONTRACT} at dominance "
                    f"threshold {self.parameter_ranking_dominance_threshold}"
                ),
                metrics=metrics,
            )

        supported = context_metrics.get("supported_corrections")
        supported_lines: list[int] = []
        if isinstance(supported, (list, tuple)):
            for raw_action in supported:
                if not isinstance(raw_action, Mapping):
                    continue
                if str(raw_action.get("tool") or "") != "correct_parameters":
                    continue
                arguments = raw_action.get("arguments")
                if not isinstance(arguments, Mapping):
                    continue
                raw_targets = [
                    (key, arguments[key])
                    for key in ("branch_row0", "line_index1", "line_index")
                    if arguments.get(key) is not None
                ]
                if len(raw_targets) != 1:
                    continue
                target_key, raw_target = raw_targets[0]
                if not isinstance(raw_target, int) or isinstance(raw_target, bool):
                    continue
                supported_line = (
                    int(raw_target) + 1
                    if target_key == "branch_row0"
                    else int(raw_target)
                )
                if supported_line <= 0 or supported_line in supported_lines:
                    continue
                supported_lines.append(supported_line)

        metrics["parameter_context_supported_line_indices1"] = supported_lines
        if not supported_lines:
            raise ScenarioRejected(
                "parameter_context_target_unavailable",
                (
                    f"line {line_index1}: deployed parameter context exposed "
                    "no valid correction target"
                ),
                metrics=metrics,
            )
        first_supported_line = supported_lines[0]
        metrics["parameter_context_first_line_index1"] = first_supported_line
        if first_supported_line != line_index1:
            raise ScenarioRejected(
                "parameter_context_target_ambiguous",
                (
                    f"line {line_index1}: deployed parameter context ranked "
                    f"line {first_supported_line} first"
                ),
                metrics=metrics,
            )
        return {
            "corrected_case_path": corrected_case_path,
            "corrected_r": corrected_r,
            "corrected_x": corrected_x,
            "base_candidate_metrics": {
                key: candidate_metrics.get(key)
                for key in (
                    "target_fixed",
                    "target_metric_value",
                    "target_metric_threshold",
                    "chi_square_statistic",
                    "chi_square_threshold",
                    "post_action_resolved",
                    "globally_resolved",
                    "physical_constraints_ok",
                    "physical_evidence_scope",
                )
            },
        }

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
        local = (self.balanced_artifact_dir / subdir / basename).absolute()
        if self.artifact_allowlist is not None and local not in self.artifact_allowlist:
            raise ScenarioRejected("artifact_missing", str(local))
        if not local.is_file():
            raise ScenarioRejected("artifact_missing", str(local))
        self.consumed_artifacts.add(local)
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
            # This is the stable network identity used for grouped/stratified
            # release reporting. ``case`` may later become a content-addressed
            # derived artifact after a parameter/topology correction and must
            # not fragment one IEEE-14 population into one pseudo-case per root.
            "network_case": "case14",
            "case": case,
            "measurements": [float(value) for value in measurements],
            "semantic_field_provenance": {"measurements": _SNAPSHOT_PROVENANCE},
            "metadata": {},
        }

    def _declare_measurement_recovery_tolerance(
        self,
        scenario: dict[str, Any],
        indices: Iterable[int],
    ) -> None:
        """Persist a three-sigma bound for estimator-derived replacements.

        The correction executor estimates a bad meter's clean value from the
        observable WLS state; it is not handed the corpus reference.  The
        estimator and that tracked reference contribute independent sensor
        uncertainty, so their difference has standard deviation
        ``sqrt(sigma_estimator^2 + sigma_reference^2)``.  Under the equal-noise
        tracked-corpus contract this is ``sqrt(2) * sigma``.  Target identity
        and every healthy channel remain exact, separately enforced checks in
        the strict truth audit.
        """
        valid = [
            int(index)
            for index in indices
            if 0 <= int(index) < len(self.noise_profile())
        ]
        if not valid:
            return
        measurement_abs = max(
            1e-6,
            3.0
            * float(np.sqrt(2.0))
            * max(float(self.noise_profile()[index]) for index in valid),
        )
        release_audit = scenario.setdefault("release_audit", {})
        tolerances = release_audit.setdefault("tolerances", {})
        tolerances["measurement_abs"] = max(
            float(tolerances.get("measurement_abs", 0.0)), measurement_abs
        )
        release_audit["measurement_tolerance_basis"] = (
            "three_sigma_independent_estimator_and_reference_noise"
        )

    def _require_sequential_measurement_observability(
        self,
        *,
        clean_measurements: Sequence[float],
        faults: Sequence[Mapping[str, Any]],
        family: str,
    ) -> None:
        """Require every declared meter fault to remain detectable on its own.

        A truth-free policy cannot recover a small co-injected fault if fixing a
        larger peer makes the global WLS gate clean.  Such a root can terminate
        only by silently leaving truth behind, so it is not a valid sequential
        recovery scenario even when the original combined vector is anomalous.
        """
        if not self.validate or len(faults) <= 1:
            return
        for fault in faults:
            index = fault.get("index")
            observed = fault.get("observed")
            if index is None or observed is None:
                raise ScenarioRejected(
                    "sequential_fault_truth_incomplete", f"{family}: {fault}"
                )
            index = int(index)
            probe = [float(value) for value in clean_measurements]
            probe[index] = float(observed)
            statistic = self._chi2_statistic("case14", probe)
            if statistic < self.chi2_limit:
                raise ScenarioRejected(
                    "sequential_fault_not_individually_detectable",
                    f"{family}: index {index} chi2 {statistic:.1f} < "
                    f"{self.chi2_limit:.1f}",
                )

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
        # A healthy sensor snapshot includes ordinary measurement noise.  The
        # release target is therefore the observed vector, not a noiseless
        # latent vector that no transactional action can reproduce.
        scenario["clean_measurements"] = [float(value) for value in row["z_obs"]]
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
        clean_measurements = list(z_obs)
        for error_index in error_indices:
            clean_measurements[error_index] = z_true[error_index]
        # A release scenario must be solvable by fixing exactly the declared
        # bad meters while preserving every healthy observed channel. Some raw
        # corpus rows retain enough ordinary/noise outliers on undeclared
        # channels that even this truth-restored vector fails the global gate;
        # those roots are intrinsically non-terminal and are skipped rather
        # than encouraging broad healthy-channel rewrites.
        self._require_clean("case14", clean_measurements, family)
        scenario["clean_measurements"] = clean_measurements
        scenario["true_measurement_errors"] = [
            {
                "index": error_index,
                "channel": label.get("channel"),
                "observed": z_obs[error_index],
                "clean": z_true[error_index],
            }
            for error_index in error_indices
        ]
        self._require_sequential_measurement_observability(
            clean_measurements=clean_measurements,
            faults=scenario["true_measurement_errors"],
            family=family,
        )
        self._declare_measurement_recovery_tolerance(scenario, error_indices)
        return scenario

    def _parameter_scenario(self, row: Mapping[str, Any], index: int) -> dict[str, Any]:
        label = dict(row.get("label") or {})
        line_row0 = label.get("line_row")
        if line_row0 is None:
            raise ScenarioRejected("label_missing_line_row", str(label))
        line_row0 = int(line_row0)
        if not 0 <= line_row0 < NL:
            raise ScenarioRejected("label_line_row_out_of_range", str(line_row0))
        if not row.get("z_scans"):
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
        clean_r = float(true_ppc["branch"][line_row0][2])
        clean_x = float(true_ppc["branch"][line_row0][3])
        # The multi-scan inverse problem identifies line impedance only within
        # a bounded numerical/measurement uncertainty.  Keep healthy branches
        # on the audit's tight default and declare a separate final-target
        # allowance of 10% of the larger recovered R/X scale (with a 0.02 pu
        # floor for short lines).  The offline realizability gate below uses
        # this exact release tolerance; it does not add its estimator result to
        # the scenario that the online policy receives.
        parameter_scale = max(abs(clean_r), abs(clean_x))
        final_case_abs_tolerance = max(0.02, 0.10 * parameter_scale)
        gate_result = self._require_parameter_correction_realizable(
            line_row0=line_row0,
            clean_r=clean_r,
            clean_x=clean_x,
            z_scans=row["z_scans"],
            measurements=z_obs,
            final_case_abs_tolerance=final_case_abs_tolerance,
        )
        scenario = self._base_scenario(
            self._scenario_id("parameter", row.get("id"), index),
            case="case14",
            measurements=z_obs,
            family="parameter",
        )
        scenario["clean_case"] = str(true_case)
        scenario["clean_measurements"] = list(z_obs)
        scenario["true_parameter_errors"] = [
            {
                "branch_row0": line_row0,
                "line_index1": line_row0 + 1,
                "parameter": "rx",
                "clean_r": clean_r,
                "clean_x": clean_x,
                "r_factor": label.get("r_factor"),
                "x_factor": label.get("x_factor"),
                "from_bus": label.get("from_bus"),
                "to_bus": label.get("to_bus"),
            }
        ]
        scenario["release_audit"] = {
            "tolerances": {
                "final_case_abs": final_case_abs_tolerance,
            },
            "tolerance_basis": "multi_scan_parameter_estimator_v2",
        }
        scenario["metadata"]["parameter_scans"] = {
            "z_scans": [[float(v) for v in scan] for scan in row["z_scans"]],
            "initial_state_strategy": "observed_vm_plus_configured_case_angles_v1",
        }
        if gate_result is not None:
            self._parameter_gate_results[scenario["scenario_id"]] = gate_result
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
        from pypower.idx_bus import BUS_I, VMAX, VMIN
        from pypower.idx_gen import GEN_BUS, VG

        safe_rows = self._safe_outage_rows()
        row0 = int(self._rng.choice(safe_rows))
        load_scale = float(self._rng.uniform(0.85, 1.15))
        truth = pypower_case14()
        truth["bus"][:, 2] *= load_scale
        truth["bus"][:, 3] *= load_scale
        # PYPOWER's canonical case14 ships generator voltage setpoints of 1.07
        # and 1.09 pu at buses whose declared VMAX is 1.06 pu.  That is usable
        # as a power-flow demo but not as a physically admissible verification
        # fixture.  Keep synthesized telemetry inside the case's own declared
        # voltage limits before solving instead of teaching the verifier to
        # tolerate an actual limit violation.
        voltage_bounds = {
            int(row[BUS_I]): (float(row[VMIN]), float(row[VMAX]))
            for row in truth["bus"]
        }
        for gen_row in truth["gen"]:
            vmin, vmax = voltage_bounds[int(gen_row[GEN_BUS])]
            gen_row[VG] = min(max(float(gen_row[VG]), vmin), vmax)
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
        z_obs = _canonicalize_synthesized_measurement_vector(z_obs)
        # The emitted vector, not only the native solver vector, must satisfy
        # the same physics contract before it is persisted or fingerprinted.
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
        scenario["clean_measurements"] = list(z_obs)
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
        scenario["release_audit"] = copy.deepcopy(
            _EXPLANATION_ONLY_RELEASE_AUDIT
        )
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
        # The hidden clean current copy is QA replay data, never runtime telemetry.
        clean_current_key = f"{BRANCH_CURRENT_CHANNEL}_clean"
        runtime_scans = [
            {key: value for key, value in dict(scan).items() if key != clean_current_key}
            for scan in scans
        ]
        scenario["metadata"]["hif_scan_window"] = {
            "scan_window_path": str(row.get("id") or scenario["scenario_id"]),
            "scans": copy.deepcopy(runtime_scans),
            "sigma_z": copy.deepcopy(row.get("sigma_z")),
            "window_metadata": copy.deepcopy(row.get("window_metadata") or {}),
        }
        branch_currents = row.get(BRANCH_CURRENT_CHANNEL)
        if not branch_currents and runtime_scans:
            branch_currents = runtime_scans[0].get(BRANCH_CURRENT_CHANNEL)
        if branch_currents and branch_current_rows_to_phasors(branch_currents):
            current_sigma = row.get(BRANCH_CURRENT_SIGMA_KEY)
            if current_sigma is None and runtime_scans:
                current_sigma = runtime_scans[0].get(BRANCH_CURRENT_SIGMA_KEY)
            scenario["metadata"][BRANCH_CURRENT_CHANNEL] = copy.deepcopy(list(branch_currents))
            scenario["metadata"]["hif_runtime"][BRANCH_CURRENT_CHANNEL] = copy.deepcopy(
                list(branch_currents)
            )
            if current_sigma is not None:
                scenario["metadata"][BRANCH_CURRENT_SIGMA_KEY] = float(current_sigma)
                scenario["metadata"]["hif_runtime"][BRANCH_CURRENT_SIGMA_KEY] = float(
                    current_sigma
                )
                scenario["metadata"]["hif_scan_window"][BRANCH_CURRENT_SIGMA_KEY] = float(
                    current_sigma
                )
        scenario["hidden_truth"] = {"true_hif_errors": [copy.deepcopy(label)]}
        scenario["release_audit"] = copy.deepcopy(
            _EXPLANATION_ONLY_RELEASE_AUDIT
        )
        return scenario

    @staticmethod
    def _balanced_voltage_control(
        voltages: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        """Create a balanced telemetry null at the same bus voltage scale.

        This is non-model control metadata derived from tracked three-phase
        phasors.  It preserves each bus's mean line-neutral magnitude and
        phase-A angle while enforcing an exact 120-degree phase separation.
        """
        balanced: list[dict[str, Any]] = []
        for item in voltages:
            magnitudes = item.get("vln_pu")
            angles = item.get("ang_deg")
            if not isinstance(magnitudes, Sequence) or len(magnitudes) != 3:
                continue
            if not isinstance(angles, Sequence) or len(angles) != 3:
                continue
            magnitude = float(sum(float(value) for value in magnitudes) / 3.0)
            phase_a = float(angles[0])
            balanced.append(
                {
                    **copy.deepcopy(dict(item)),
                    "vln_pu": [magnitude, magnitude, magnitude],
                    "ang_deg": [phase_a, phase_a - 120.0, phase_a + 120.0],
                }
            )
        return balanced

    def _observable_unbalance_signatures(self, row: Mapping[str, Any]) -> list[str]:
        """Sensor signatures an operator's monitors would raise on this row.

        The signature text reaches the policy, so it must describe what the
        telemetry shows rather than what the label says.  The VUF flag needs
        the largest bus VUF to clear the same gate the deployment provider
        applies; the current-spread flag needs the branch-current channel to
        expose a noise-significant unbalance source and a quiet
        line-differential null (a line carrying fault current is an HIF
        signature, not an unbalance one).  An empty result means the labeled
        unbalance is unobservable and the row must be rejected.
        """
        voltages = row.get("three_phase_voltages")
        signatures: list[str] = []
        factors = voltage_unbalance_factors(voltages)
        if factors and float(factors[0]["vuf"]) >= self.unbalance_vuf_threshold:
            signatures.append(UNBALANCE_SIGNATURE)
        currents = row.get(BRANCH_CURRENT_CHANNEL)
        if currents and branch_current_rows_to_phasors(currents):
            raw_sigma = row.get(BRANCH_CURRENT_SIGMA_KEY)
            sigma = float(raw_sigma) if raw_sigma is not None else None
            localization = unbalance_source_localization(
                voltages, currents, top_k=1, sigma_pu=sigma
            )
            null_test = line_differential_null_test(voltages, currents, sigma_pu=sigma)
            hif_like = bool(
                isinstance(null_test, Mapping)
                and null_test.get("hif_like_differential_present")
            )
            if (
                localization is not None
                and bool(localization.get("significant"))
                and not hif_like
            ):
                signatures.append(UNBALANCE_CURRENT_SIGNATURE)
        return signatures

    def _unbalance_scenario(
        self, row: Mapping[str, Any], index: int
    ) -> dict[str, Any]:
        label = copy.deepcopy(dict(row.get("label") or {}))
        voltages = row.get("three_phase_voltages")
        if not isinstance(voltages, Sequence) or not voltages:
            raise ScenarioRejected("three_phase_voltages_missing", str(row.get("id")))
        signatures = self._observable_unbalance_signatures(row)
        if not signatures:
            raise ScenarioRejected("unbalance_not_observable", str(row.get("id")))
        z_obs = [float(value) for value in row["z_obs"]]
        if self.validate:
            self._chi2_statistic("case14", z_obs)
        scenario = self._base_scenario(
            self._scenario_id("three_phase_unbalance", row.get("id"), index),
            case="case14",
            measurements=z_obs,
            family="three_phase_unbalance",
        )
        scenario["clean_case"] = "case14"
        scenario["clean_measurements"] = [float(value) for value in row["z_true"]]
        scenario["unresolved_signatures"] = signatures
        scenario["semantic_field_provenance"]["unresolved_signatures"] = (
            _WAVEFORM_PROVENANCE
        )
        scenario["metadata"]["three_phase_voltages"] = copy.deepcopy(list(voltages))
        branch_currents = row.get(BRANCH_CURRENT_CHANNEL)
        if branch_currents and branch_current_rows_to_phasors(branch_currents):
            scenario["metadata"][BRANCH_CURRENT_CHANNEL] = copy.deepcopy(list(branch_currents))
            if row.get(BRANCH_CURRENT_SIGMA_KEY) is not None:
                scenario["metadata"][BRANCH_CURRENT_SIGMA_KEY] = float(
                    row[BRANCH_CURRENT_SIGMA_KEY]
                )
        scenario["hidden_truth"] = {"true_unbalance_errors": [label]}
        scenario["release_audit"] = copy.deepcopy(
            _EXPLANATION_ONLY_RELEASE_AUDIT
        )
        return scenario

    def _telemetry_no_disturbance_scenario(
        self, row: Mapping[str, Any], index: int
    ) -> dict[str, Any]:
        voltages = row.get("three_phase_voltages")
        if not isinstance(voltages, Sequence) or not voltages:
            raise ScenarioRejected("three_phase_voltages_missing", str(row.get("id")))
        balanced = self._balanced_voltage_control(voltages)
        if not balanced:
            raise ScenarioRejected("balanced_telemetry_control_invalid", str(row.get("id")))
        measurements = [float(value) for value in row["z_true"]]
        self._require_clean("case14", measurements, "telemetry_no_disturbance")
        scenario = self._base_scenario(
            self._scenario_id("telemetry_no_disturbance", row.get("id"), index),
            case="case14",
            measurements=measurements,
            family="telemetry_no_disturbance",
        )
        scenario["clean_case"] = "case14"
        scenario["clean_measurements"] = list(measurements)
        scenario["metadata"]["three_phase_voltages"] = balanced
        branch_currents = row.get(BRANCH_CURRENT_CHANNEL)
        if branch_currents and branch_current_rows_to_phasors(branch_currents):
            balanced_currents = balanced_branch_current_control(list(branch_currents))
            if not balanced_currents:
                raise ScenarioRejected(
                    "balanced_current_control_invalid", str(row.get("id"))
                )
            scenario["metadata"][BRANCH_CURRENT_CHANNEL] = balanced_currents
            if row.get(BRANCH_CURRENT_SIGMA_KEY) is not None:
                scenario["metadata"][BRANCH_CURRENT_SIGMA_KEY] = float(
                    row[BRANCH_CURRENT_SIGMA_KEY]
                )
        scenario["hidden_truth"] = {
            "true_unbalance_errors": [],
            "control_kind": "telemetry_present_no_disturbance",
        }
        return scenario

    # ---------------------------------------------------------- compositions

    def _overlay_indices(self, scenario: Mapping[str, Any], count: int) -> list[int]:
        """Indices eligible for a gross-offset overlay on this scenario."""
        index_map = measurement_index_map(NB, NL)
        blocked: set[int] = set()
        for fault in scenario.get("true_measurement_errors") or []:
            if fault.get("index") is not None:
                blocked.add(int(fault["index"]))
        for fault in [
            *(scenario.get("true_parameter_errors") or []),
            *(scenario.get("true_topology_errors") or []),
        ]:
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

    def _require_mixed_parameter_recovery_realizable(
        self,
        scenario: Mapping[str, Any],
        *,
        base_scenario_id: str,
    ) -> None:
        """Validate the observable two-stage route for a mixed parameter root.

        This is an offline scenario-selection check.  It executes the same WLS
        contexts and measurement corrector used online, while reusing the
        configured multi-scan result already validated for the base parameter
        root.  Truth is consulted only to reject a root whose rank-one
        observable correction targets the wrong component; no route decision,
        estimator output, or comparison metric is persisted on accepted
        scenarios.
        """
        if not self.validate:
            return
        gate_result = self._parameter_gate_results.get(str(base_scenario_id))
        parameter_faults = scenario.get("true_parameter_errors") or []
        measurement_faults = scenario.get("true_measurement_errors") or []
        if not gate_result or len(parameter_faults) != 1 or len(measurement_faults) != 1:
            raise ScenarioRejected(
                "mixed_parameter_recovery_gate_incomplete",
                "one cached parameter correction and one measurement target are required",
                metrics={
                    "base_scenario_id": str(base_scenario_id),
                    "parameter_fault_count": len(parameter_faults),
                    "measurement_fault_count": len(measurement_faults),
                    "cached_parameter_result": bool(gate_result),
                },
            )

        expected_line = int(parameter_faults[0]["line_index1"])
        expected_measurement = int(measurement_faults[0]["index"])
        metrics: dict[str, Any] = {
            "expected_parameter_line_index1": expected_line,
            "expected_measurement_index0": expected_measurement,
            "stages": [],
        }

        def reject(reason: str, detail: str, **extra: Any) -> None:
            metrics.update(extra)
            raise ScenarioRejected(reason, detail, metrics=metrics)

        current_case = str(scenario.get("case") or "case14")
        current_measurements = [float(value) for value in scenario["measurements"]]
        accepted: list[dict[str, Any]] = []

        def state(stage: str) -> dict[str, Any]:
            return {
                "state_id": f"offline_mixed_parameter_gate:{stage}",
                "case": current_case,
                "measurements": current_measurements,
                # The production parameter context only advertises a
                # correction when policy-observable repeated scans are
                # present and structurally usable.  Preserve that exact
                # metadata in this offline release-admissibility probe; the
                # prior probe silently omitted it, so every mixed parameter
                # root was rejected as context-empty even though the online
                # environment would expose the scans.
                "metadata": copy.deepcopy(scenario.get("metadata") or {}),
                "policy_observation": {
                    "accepted_corrections": copy.deepcopy(accepted),
                },
            }

        parent_metrics = self._parameter_gate_provider.run_wls(state("root"))
        if parent_metrics.get("execution_status") == "failure":
            reject(
                "mixed_parameter_recovery_wls_failed",
                "root WLS failed during offline route validation",
                failed_stage="root_wls",
                error_code=parent_metrics.get("error_code"),
            )
        signatures = [
            str(item) for item in parent_metrics.get("unresolved_signatures") or []
        ]
        measurement_dominant = any(
            item.startswith("wls_residual_outlier_dominant") for item in signatures
        )
        branch_dominant = any(
            item.startswith("wls_branch_multiplier_dominant") for item in signatures
        )
        order = (
            ("measurement", "parameter")
            if measurement_dominant and not branch_dominant
            else ("parameter", "measurement")
        )
        metrics["observable_stage_order"] = list(order)

        for stage_index, family in enumerate(order):
            stage_metrics: dict[str, Any] = {
                "stage": stage_index + 1,
                "family": family,
            }
            if family == "measurement":
                context = self._parameter_gate_provider.get_measurement_context(
                    state(f"measurement_context_{stage_index}")
                )
                if context.get("execution_status") == "failure":
                    reject(
                        "mixed_parameter_recovery_context_failed",
                        "measurement context failed during offline route validation",
                        failed_stage="measurement_context",
                        error_code=context.get("error_code"),
                    )
                supported = context.get("supported_corrections") or []
                if not supported:
                    reject(
                        "mixed_parameter_recovery_context_empty",
                        "measurement context produced no correction",
                        failed_stage="measurement_context",
                    )
                action = copy.deepcopy(dict(supported[0]))
                group = action.get("arguments", {}).get("suspect_group") or []
                observed_targets = sorted(int(value) for value in group)
                stage_metrics["rank_one_target_indices0"] = observed_targets
                if observed_targets != [expected_measurement]:
                    metrics["stages"].append(stage_metrics)
                    reject(
                        "mixed_parameter_recovery_target_mismatch",
                        (
                            "rank-one measurement correction does not match the "
                            "declared mixed-error target"
                        ),
                        failed_stage="measurement_context",
                    )
                correction = self._parameter_gate_provider.correct_measurements(
                    state(f"measurement_correction_{stage_index}"), action
                )
                if correction.get("execution_status") == "failure":
                    metrics["stages"].append(stage_metrics)
                    reject(
                        "mixed_parameter_recovery_correction_failed",
                        "rank-one measurement correction failed",
                        failed_stage="measurement_correction",
                        error_code=correction.get("error_code"),
                    )
                modification = correction.get("modification") or {}
                updates = modification.get("measurement_updates") or {}
                if not updates:
                    metrics["stages"].append(stage_metrics)
                    reject(
                        "mixed_parameter_recovery_correction_failed",
                        "rank-one measurement correction returned no updates",
                        failed_stage="measurement_correction",
                    )
                candidate_measurements = list(current_measurements)
                for index, value in updates.items():
                    candidate_measurements[int(index)] = float(value)
                candidate_case = current_case
                progress_floor = float(
                    self._parameter_gate_candidate_oracle.min_partial_global_progress
                )
            else:
                context = self._parameter_gate_provider.get_parameter_context(
                    state(f"parameter_context_{stage_index}")
                )
                if context.get("execution_status") == "failure":
                    reject(
                        "mixed_parameter_recovery_context_failed",
                        "parameter context failed during offline route validation",
                        failed_stage="parameter_context",
                        error_code=context.get("error_code"),
                    )
                stage_metrics["parameter_ranking"] = {
                    key: copy.deepcopy(context.get(key))
                    for key in _PARAMETER_RANKING_METRIC_KEYS
                }
                if (
                    self._enforce_parameter_ranking_dominance
                    and not parameter_ranking_contract_is_dominant(
                        context,
                        expected_threshold=(
                            self.parameter_ranking_dominance_threshold
                        ),
                    )
                ):
                    metrics["stages"].append(stage_metrics)
                    reject(
                        "mixed_parameter_recovery_context_not_dominant",
                        (
                            "parameter context did not satisfy the declared "
                            "observable distinct-line dominance contract"
                        ),
                        failed_stage="parameter_context",
                    )
                supported = context.get("supported_corrections") or []
                if not supported:
                    reject(
                        "mixed_parameter_recovery_context_empty",
                        "parameter context produced no correction",
                        failed_stage="parameter_context",
                    )
                action = copy.deepcopy(dict(supported[0]))
                observed_line = int(action.get("arguments", {}).get("line_index", -1))
                stage_metrics["rank_one_line_index1"] = observed_line
                if observed_line != expected_line:
                    metrics["stages"].append(stage_metrics)
                    reject(
                        "mixed_parameter_recovery_target_mismatch",
                        (
                            "rank-one parameter correction does not match the "
                            "declared mixed-error target"
                        ),
                        failed_stage="parameter_context",
                    )
                candidate_case = str(gate_result["corrected_case_path"])
                candidate_measurements = list(current_measurements)
                progress_floor = float(
                    self._parameter_gate_candidate_oracle.min_branch_partial_global_progress
                )

            candidate_state = {
                "state_id": f"offline_mixed_parameter_gate:candidate_{stage_index}",
                "status": "candidate",
                "source_action": action,
                "case": candidate_case,
                "measurements": candidate_measurements,
                "policy_observation": {
                    "accepted_corrections": copy.deepcopy(accepted),
                },
            }
            candidate_metrics = self._parameter_gate_provider.run_wls(candidate_state)
            if candidate_metrics.get("execution_status") == "failure":
                metrics["stages"].append(stage_metrics)
                reject(
                    "mixed_parameter_recovery_wls_failed",
                    f"{family} candidate WLS failed",
                    failed_stage=f"{family}_candidate_wls",
                    error_code=candidate_metrics.get("error_code"),
                )
            try:
                parent_score = float(parent_metrics["remaining_anomaly_score"])
                candidate_score = float(candidate_metrics["remaining_anomaly_score"])
                global_progress = (parent_score - candidate_score) / max(
                    abs(parent_score), 1e-12
                )
            except (KeyError, TypeError, ValueError, OverflowError):
                global_progress = None
            stage_metrics.update(
                {
                    "target_fixed": candidate_metrics.get("target_fixed"),
                    "target_metric_value": candidate_metrics.get(
                        "target_metric_value"
                    ),
                    "target_metric_threshold": candidate_metrics.get(
                        "target_metric_threshold"
                    ),
                    "post_action_resolved": candidate_metrics.get(
                        "post_action_resolved"
                    ),
                    "physical_constraints_ok": candidate_metrics.get(
                        "physical_constraints_ok"
                    ),
                    "global_progress": global_progress,
                    "partial_progress_floor": progress_floor,
                }
            )
            metrics["stages"].append(stage_metrics)
            partial_progress = (
                global_progress is not None and global_progress >= progress_floor
            )
            if not (
                candidate_metrics.get("target_fixed") is True
                and candidate_metrics.get("physical_constraints_ok") is True
                and (
                    candidate_metrics.get("post_action_resolved") is True
                    or partial_progress
                )
            ):
                reject(
                    "mixed_parameter_recovery_candidate_unrealizable",
                    f"{family} candidate fails observable acceptance criteria",
                    failed_stage=f"{family}_candidate_acceptance",
                )

            current_case = candidate_case
            current_measurements = candidate_measurements
            accepted.append({"source_action": copy.deepcopy(action)})
            parent_metrics = candidate_metrics

        if not (
            parent_metrics.get("target_fixed") is True
            and parent_metrics.get("post_action_resolved") is True
            and parent_metrics.get("physical_constraints_ok") is True
        ):
            reject(
                "mixed_parameter_recovery_not_terminal",
                "two-stage configured recovery did not reach a clean terminal candidate",
                failed_stage="terminal_verification",
            )

    def _require_mixed_topology_recovery_realizable(
        self,
        scenario: Mapping[str, Any],
    ) -> None:
        """Validate the observable two-stage route for a mixed-topology root.

        The deployed route can be measurement-first or topology-first.  The
        root topology inventory chooses the order: an actionable correction
        runs first, while a complete-negative inventory permits the
        measurement fallback.  After that partial measurement correction, a
        refreshed measurement context must bundle an actionable topology
        route.  Clean truth is consulted only to reject a wrong target or a
        final meter estimate outside the already-declared strict-audit
        tolerance; it is never an input to either correction.
        """
        if not self.validate:
            return

        metrics: dict[str, Any] = {"stages": []}

        def reject(reason: str, detail: str, **extra: Any) -> None:
            metrics.update(extra)
            raise ScenarioRejected(reason, detail, metrics=metrics)

        def require_bound_context(
            context: Any,
            parent_state: Mapping[str, Any],
            *,
            context_tool: str,
            failed_stage: str,
        ) -> Mapping[str, Any]:
            if not isinstance(context, Mapping):
                reject(
                    "mixed_topology_recovery_context_failed",
                    f"{failed_stage} returned malformed evidence",
                    failed_stage=failed_stage,
                )
            if context.get("execution_status") == "failure":
                reject(
                    "mixed_topology_recovery_context_failed",
                    f"{failed_stage} failed",
                    failed_stage=failed_stage,
                    error_code=context.get("error_code"),
                )
            parent_state_id = parent_state.get("state_id")
            if (
                context.get("context_tool") != context_tool
                or context.get("state_id") != parent_state_id
            ):
                reject(
                    "mixed_topology_recovery_context_unbound",
                    (
                        f"{failed_stage} is not bound to the requested "
                        "state and context tool"
                    ),
                    failed_stage=failed_stage,
                    expected_context_tool=context_tool,
                    expected_state_id=parent_state_id,
                    observed_context_tool=context.get("context_tool"),
                    observed_state_id=context.get("state_id"),
                )
            parent_hash = parent_state.get("state_hash")
            context_hash = context.get("state_hash")
            if (
                parent_hash is not None or context_hash is not None
            ) and context_hash != parent_hash:
                reject(
                    "mixed_topology_recovery_context_unbound",
                    f"{failed_stage} carries a stale state hash",
                    failed_stage=failed_stage,
                    expected_state_hash=parent_hash,
                    observed_state_hash=context_hash,
                )
            return context

        def require_bound_action(
            action: Any,
            parent_state: Mapping[str, Any],
            *,
            expected_tool: str,
            failed_stage: str,
        ) -> dict[str, Any]:
            if not isinstance(action, Mapping) or set(action) != {
                "tool",
                "arguments",
            }:
                reject(
                    "mixed_topology_recovery_action_schema_invalid",
                    (
                        f"{failed_stage} action must contain exactly tool "
                        "and arguments"
                    ),
                    failed_stage=failed_stage,
                )
            raw_arguments = action.get("arguments")
            if not isinstance(raw_arguments, Mapping):
                reject(
                    "mixed_topology_recovery_action_schema_invalid",
                    f"{failed_stage} arguments must be a mapping",
                    failed_stage=failed_stage,
                )
            expected_argument_keys = (
                {"state_id", "suspect_group"}
                if expected_tool == CORRECT_MEASUREMENTS
                else {"state_id", "line_index", "status"}
            )
            observed_argument_keys = set(raw_arguments)
            if observed_argument_keys != expected_argument_keys:
                reject(
                    "mixed_topology_recovery_action_schema_invalid",
                    (
                        f"{failed_stage} correction arguments do not match "
                        "the production context-action schema"
                    ),
                    failed_stage=failed_stage,
                    expected_argument_keys=sorted(expected_argument_keys),
                    observed_argument_keys=sorted(
                        str(key) for key in observed_argument_keys
                    ),
                )
            try:
                normalized = normalize_action(action)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                reject(
                    "mixed_topology_recovery_action_schema_invalid",
                    f"{failed_stage} action normalization failed",
                    failed_stage=failed_stage,
                    evidence_error=f"{type(exc).__name__}: {exc}",
                )
            if normalized["tool"] != expected_tool:
                reject(
                    "mixed_topology_recovery_action_schema_invalid",
                    f"{failed_stage} selected the wrong correction tool",
                    failed_stage=failed_stage,
                    expected_tool=expected_tool,
                    observed_tool=normalized["tool"],
                )
            arguments = normalized["arguments"]
            if arguments.get("state_id") != parent_state.get("state_id"):
                reject(
                    "mixed_topology_recovery_action_unbound",
                    (
                        f"{failed_stage} correction is not bound to the "
                        "requested parent state"
                    ),
                    failed_stage=failed_stage,
                    expected_state_id=parent_state.get("state_id"),
                    observed_state_id=arguments.get("state_id"),
                )
            if expected_tool == CORRECT_MEASUREMENTS:
                group = arguments.get("suspect_group")
                if (
                    not isinstance(group, (list, tuple))
                    or not group
                    or any(
                        not isinstance(index, int)
                        or isinstance(index, bool)
                        or index < 0
                        for index in group
                    )
                    or len(set(group)) != len(group)
                ):
                    reject(
                        "mixed_topology_recovery_action_schema_invalid",
                        (
                            f"{failed_stage} suspect_group must contain "
                            "unique non-negative integer indices"
                        ),
                        failed_stage=failed_stage,
                    )
            else:
                line_index = arguments.get("line_index")
                status = arguments.get("status")
                if (
                    not isinstance(line_index, int)
                    or isinstance(line_index, bool)
                    or line_index < 1
                    or not isinstance(status, int)
                    or isinstance(status, bool)
                    or status not in {0, 1}
                ):
                    reject(
                        "mixed_topology_recovery_action_schema_invalid",
                        (
                            f"{failed_stage} topology target or status "
                            "is malformed"
                        ),
                        failed_stage=failed_stage,
                    )
            return normalized

        try:
            topology_faults = list(
                scenario.get("true_topology_errors") or []
            )
            measurement_faults = list(
                scenario.get("true_measurement_errors") or []
            )
            if len(topology_faults) != 1 or len(measurement_faults) != 1:
                raise ValueError(
                    "one topology target and one measurement target are required"
                )
            topology_fault = topology_faults[0]
            measurement_fault = measurement_faults[0]
            if not isinstance(topology_fault, Mapping) or not isinstance(
                measurement_fault, Mapping
            ):
                raise TypeError("truth targets must be mappings")

            if topology_fault.get("branch_row0") is not None:
                expected_branch_row0 = int(topology_fault["branch_row0"])
            else:
                expected_branch_row0 = (
                    int(topology_fault["line_index1"]) - 1
                )
            expected_status = int(
                topology_fault.get("expected_status", 0)
            )
            if measurement_fault.get("index") is not None:
                expected_measurement = int(measurement_fault["index"])
            else:
                expected_measurement = int(measurement_fault["index0"])
            clean_value = float(measurement_fault["clean"])

            current_case = scenario["case"]
            current_measurements = [
                float(value) for value in scenario["measurements"]
            ]
            clean_measurements = [
                float(value) for value in scenario["clean_measurements"]
            ]
            metadata = scenario.get("metadata")
            current_metadata = copy.deepcopy(
                dict(metadata) if isinstance(metadata, Mapping) else {}
            )
            release_audit = scenario.get("release_audit")
            release_audit = (
                release_audit
                if isinstance(release_audit, Mapping)
                else {}
            )
            tolerances = release_audit.get("tolerances")
            tolerances = (
                tolerances if isinstance(tolerances, Mapping) else {}
            )
            measurement_abs = float(tolerances["measurement_abs"])
            measurement_rel = float(
                tolerances.get("measurement_rel", 1e-6)
            )
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            reject(
                "mixed_topology_recovery_gate_incomplete",
                "mixed-topology recovery evidence is missing or malformed",
                evidence_error=f"{type(exc).__name__}: {exc}",
            )

        if (
            not 0 <= expected_branch_row0 < NL
            or expected_status not in {0, 1}
            or not 0 <= expected_measurement < len(current_measurements)
            or len(clean_measurements) != len(current_measurements)
            or not all(
                math.isfinite(value)
                for value in (
                    clean_value,
                    measurement_abs,
                    measurement_rel,
                )
            )
            or measurement_abs < 0.0
            or measurement_rel < 0.0
        ):
            reject(
                "mixed_topology_recovery_gate_incomplete",
                "mixed-topology targets or tolerances are outside the state",
            )
        if not math.isclose(
            clean_measurements[expected_measurement],
            clean_value,
            abs_tol=measurement_abs,
            rel_tol=measurement_rel,
        ):
            reject(
                "mixed_topology_recovery_gate_incomplete",
                "declared meter truth conflicts with the clean vector",
            )

        accepted: list[dict[str, Any]] = []

        def state(stage: str) -> dict[str, Any]:
            state_payload = {
                "state_id": f"offline_mixed_topology_gate:{stage}",
                "case": current_case,
                "measurements": list(current_measurements),
                "metadata": copy.deepcopy(current_metadata),
                "policy_observation": {
                    "accepted_corrections": copy.deepcopy(accepted),
                },
            }
            state_payload["state_hash"] = _state_content_hash(
                state_payload["case"],
                state_payload["measurements"],
                state_payload["metadata"],
            )
            return state_payload

        root_state = state("root")
        parent_metrics = self._parameter_gate_provider.run_wls(root_state)
        if not isinstance(parent_metrics, Mapping):
            reject(
                "mixed_topology_recovery_wls_failed",
                "root WLS returned malformed evidence",
                failed_stage="root_wls",
            )
        if parent_metrics.get("execution_status") == "failure":
            reject(
                "mixed_topology_recovery_wls_failed",
                "root WLS failed during offline route validation",
                failed_stage="root_wls",
                error_code=parent_metrics.get("error_code"),
            )

        root_topology_context = require_bound_context(
            self._parameter_gate_provider.get_topology_context(root_state),
            root_state,
            context_tool=GET_TOPOLOGY_CONTEXT,
            failed_stage="root_topology_context",
        )
        root_topology_supported = [
            copy.deepcopy(dict(action))
            for action in root_topology_context.get(
                "supported_corrections"
            )
            or []
            if isinstance(action, Mapping)
            and str(action.get("tool") or "") == CORRECT_TOPOLOGY
        ]
        root_topology_route = str(
            root_topology_context.get("route_status") or ""
        )
        metrics["root_topology_route_status"] = root_topology_route
        if (
            root_topology_route == "actionable"
            and root_topology_supported
        ):
            order = ("topology", "measurement")
        elif (
            root_topology_route == "complete_negative"
            and not root_topology_supported
        ):
            order = ("measurement", "topology")
        else:
            reject(
                "mixed_topology_recovery_context_inconclusive",
                (
                    "root topology route is neither actionable nor a "
                    "complete-negative fallback"
                ),
                failed_stage="root_topology_context",
            )
        metrics["observable_stage_order"] = list(order)

        for stage_index, family in enumerate(order):
            parent_state = state(f"{family}_context_{stage_index}")
            stage_metrics: dict[str, Any] = {
                "stage": stage_index + 1,
                "family": family,
            }

            if family == "measurement":
                context = require_bound_context(
                    self._parameter_gate_provider.get_measurement_context(
                        parent_state
                    ),
                    parent_state,
                    context_tool=GET_MEASUREMENT_CONTEXT,
                    failed_stage="measurement_context",
                )
                supported = [
                    copy.deepcopy(dict(action))
                    for action in context.get("supported_corrections") or []
                    if isinstance(action, Mapping)
                    and str(action.get("tool") or "")
                    == CORRECT_MEASUREMENTS
                ]
                if not supported:
                    reject(
                        "mixed_topology_recovery_context_empty",
                        "measurement context produced no correction",
                        failed_stage="measurement_context",
                    )
                action = require_bound_action(
                    supported[0],
                    parent_state,
                    expected_tool=CORRECT_MEASUREMENTS,
                    failed_stage="measurement_context",
                )
                arguments = action.get("arguments")
                arguments = (
                    arguments if isinstance(arguments, Mapping) else {}
                )
                group = arguments.get("suspect_group")
                try:
                    targets = sorted(int(value) for value in group)
                except (TypeError, ValueError, OverflowError) as exc:
                    reject(
                        "mixed_topology_recovery_target_mismatch",
                        "measurement context returned a malformed target",
                        failed_stage="measurement_context",
                        evidence_error=f"{type(exc).__name__}: {exc}",
                    )
                stage_metrics["rank_one_target_indices0"] = targets
                if targets != [expected_measurement]:
                    reject(
                        "mixed_topology_recovery_target_mismatch",
                        "measurement correction does not match declared truth",
                        failed_stage="measurement_context",
                        expected_measurement_index0=expected_measurement,
                    )
                correction = (
                    self._parameter_gate_provider.correct_measurements(
                        parent_state, action
                    )
                )
                partial_floor = float(
                    self._parameter_gate_candidate_oracle
                    .min_partial_global_progress
                )
            else:
                if stage_index == 0:
                    parent_state = root_state
                    context = root_topology_context
                    stage_metrics["route_source"] = (
                        "root_topology_context"
                    )
                else:
                    parent_state = state(
                        "post_measurement_route_context"
                    )
                    refreshed_measurement_context = require_bound_context(
                        self._parameter_gate_provider
                        .get_measurement_context(parent_state),
                        parent_state,
                        context_tool=GET_MEASUREMENT_CONTEXT,
                        failed_stage="post_measurement_route_context",
                    )
                    branch_routes = (
                        refreshed_measurement_context.get(
                            "branch_route_screening"
                        )
                    )
                    if not isinstance(branch_routes, Mapping):
                        reject(
                            "mixed_topology_recovery_context_inconclusive",
                            (
                                "post-measurement context did not bundle "
                                "branch route screening"
                            ),
                            failed_stage=(
                                "post_measurement_route_context"
                            ),
                        )
                    context = branch_routes.get("topology")
                    if not isinstance(context, Mapping):
                        reject(
                            "mixed_topology_recovery_context_inconclusive",
                            (
                                "post-measurement context did not bundle "
                                "a topology inventory"
                            ),
                            failed_stage=(
                                "post_measurement_route_context"
                            ),
                        )
                    context = require_bound_context(
                        context,
                        parent_state,
                        context_tool=GET_TOPOLOGY_CONTEXT,
                        failed_stage=(
                            "post_measurement_topology_context"
                        ),
                    )
                    if context.get("route_status") != "actionable":
                        reject(
                            "mixed_topology_recovery_context_inconclusive",
                            (
                                "post-measurement topology route is not "
                                "actionable"
                            ),
                            failed_stage=(
                                "post_measurement_route_context"
                            ),
                            topology_route_status=context.get(
                                "route_status"
                            ),
                        )
                    stage_metrics["route_source"] = (
                        "measurement_context.branch_route_screening.topology"
                    )
                supported = [
                    copy.deepcopy(dict(action))
                    for action in context.get("supported_corrections") or []
                    if isinstance(action, Mapping)
                    and str(action.get("tool") or "")
                    == CORRECT_TOPOLOGY
                ]
                if not supported:
                    reject(
                        "mixed_topology_recovery_context_empty",
                        "topology context produced no correction",
                        failed_stage="topology_context",
                    )
                action = require_bound_action(
                    supported[0],
                    parent_state,
                    expected_tool=CORRECT_TOPOLOGY,
                    failed_stage="topology_context",
                )
                arguments = action.get("arguments")
                arguments = (
                    arguments if isinstance(arguments, Mapping) else {}
                )
                try:
                    if arguments.get("branch_row0") is not None:
                        target_row0 = int(arguments["branch_row0"])
                    elif arguments.get("line_index1") is not None:
                        target_row0 = int(arguments["line_index1"]) - 1
                    else:
                        target_row0 = int(arguments["line_index"]) - 1
                    target_status = arguments.get(
                        "status", arguments.get("expected_status")
                    )
                    if (
                        arguments.get("desired_status") is not None
                        and target_status is None
                    ):
                        target_status = int(
                            bool(arguments["desired_status"])
                        )
                    target_status = int(target_status)
                except (
                    KeyError,
                    TypeError,
                    ValueError,
                    OverflowError,
                ) as exc:
                    reject(
                        "mixed_topology_recovery_target_mismatch",
                        "topology context returned a malformed target",
                        failed_stage="topology_context",
                        evidence_error=f"{type(exc).__name__}: {exc}",
                    )
                stage_metrics["rank_one_topology_target"] = {
                    "branch_row0": target_row0,
                    "status": target_status,
                }
                if (
                    target_row0 != expected_branch_row0
                    or target_status != expected_status
                ):
                    reject(
                        "mixed_topology_recovery_target_mismatch",
                        "topology correction does not match declared truth",
                        failed_stage="topology_context",
                        expected_branch_row0=expected_branch_row0,
                        expected_status=expected_status,
                    )
                correction = (
                    self._parameter_gate_provider.correct_topology(
                        parent_state, action
                    )
                )
                partial_floor = float(
                    self._parameter_gate_candidate_oracle
                    .min_branch_partial_global_progress
                )

            if correction.get("execution_status") == "failure":
                reject(
                    "mixed_topology_recovery_correction_failed",
                    f"{family} correction failed",
                    failed_stage=f"{family}_correction",
                    error_code=correction.get("error_code"),
                )
            modification = correction.get("modification")
            if not isinstance(modification, Mapping):
                reject(
                    "mixed_topology_recovery_correction_failed",
                    f"{family} correction returned no modification",
                    failed_stage=f"{family}_correction",
                )
            if family == "measurement":
                updates = modification.get("measurement_updates")
                try:
                    normalized_updates = {
                        int(index): float(value)
                        for index, value in updates.items()
                    }
                except (
                    AttributeError,
                    TypeError,
                    ValueError,
                    OverflowError,
                ) as exc:
                    reject(
                        "mixed_topology_recovery_correction_failed",
                        "measurement correction returned malformed updates",
                        failed_stage="measurement_correction",
                        evidence_error=f"{type(exc).__name__}: {exc}",
                    )
                if (
                    set(normalized_updates) != {expected_measurement}
                    or not math.isfinite(
                        normalized_updates.get(
                            expected_measurement, math.nan
                        )
                    )
                ):
                    reject(
                        "mixed_topology_recovery_target_mismatch",
                        "measurement correction modified another target",
                        failed_stage="measurement_correction",
                    )
            elif not modification.get("case"):
                reject(
                    "mixed_topology_recovery_correction_failed",
                    "topology correction returned no candidate case",
                    failed_stage="topology_correction",
                )

            try:
                (
                    candidate_case,
                    candidate_measurements,
                    candidate_metadata,
                ) = apply_modification(
                    case=current_case,
                    measurements=current_measurements,
                    metadata=current_metadata,
                    modification=modification,
                )
            except Exception as exc:
                reject(
                    "mixed_topology_recovery_correction_failed",
                    f"{family} modification could not be applied",
                    failed_stage=f"{family}_correction",
                    evidence_error=f"{type(exc).__name__}: {exc}",
                )
            candidate_state = {
                "state_id": (
                    "offline_mixed_topology_gate:"
                    f"candidate_{stage_index}"
                ),
                "status": "candidate",
                "source_action": copy.deepcopy(action),
                "case": candidate_case,
                "measurements": candidate_measurements,
                "metadata": candidate_metadata,
                "policy_observation": {
                    "accepted_corrections": copy.deepcopy(accepted),
                },
            }
            candidate_metrics = (
                self._parameter_gate_provider.run_wls(candidate_state)
            )
            if candidate_metrics.get("execution_status") == "failure":
                reject(
                    "mixed_topology_recovery_wls_failed",
                    f"{family} candidate WLS failed",
                    failed_stage=f"{family}_candidate_wls",
                    error_code=candidate_metrics.get("error_code"),
                )
            try:
                parent_score = float(
                    parent_metrics["remaining_anomaly_score"]
                )
                candidate_score = float(
                    candidate_metrics["remaining_anomaly_score"]
                )
                global_progress = (
                    parent_score - candidate_score
                ) / max(abs(parent_score), 1e-12)
            except (
                KeyError,
                TypeError,
                ValueError,
                OverflowError,
            ):
                global_progress = None
            stage_metrics.update(
                {
                    "target_fixed": candidate_metrics.get(
                        "target_fixed"
                    ),
                    "post_action_resolved": candidate_metrics.get(
                        "post_action_resolved"
                    ),
                    "physical_constraints_ok": candidate_metrics.get(
                        "physical_constraints_ok"
                    ),
                    "global_progress": global_progress,
                    "partial_progress_floor": partial_floor,
                }
            )
            metrics["stages"].append(stage_metrics)
            if not (
                candidate_metrics.get("target_fixed") is True
                and candidate_metrics.get("physical_constraints_ok") is True
                and (
                    candidate_metrics.get("post_action_resolved") is True
                    or (
                        global_progress is not None
                        and global_progress >= partial_floor
                    )
                )
            ):
                reject(
                    "mixed_topology_recovery_candidate_unrealizable",
                    f"{family} candidate fails observable acceptance criteria",
                    failed_stage=f"{family}_candidate_acceptance",
                )

            current_case = candidate_case
            current_measurements = [
                float(value) for value in candidate_measurements
            ]
            current_metadata = copy.deepcopy(candidate_metadata)
            accepted.append({"source_action": copy.deepcopy(action)})
            parent_metrics = candidate_metrics

        if not (
            parent_metrics.get("target_fixed") is True
            and parent_metrics.get("post_action_resolved") is True
            and parent_metrics.get("physical_constraints_ok") is True
        ):
            reject(
                "mixed_topology_recovery_candidate_unrealizable",
                "two-stage recovery did not reach a clean final candidate",
                failed_stage="terminal_verification",
            )

        final_value = current_measurements[expected_measurement]
        final_distance = abs(final_value - clean_value)
        within_tolerance = math.isclose(
            final_value,
            clean_value,
            abs_tol=measurement_abs,
            rel_tol=measurement_rel,
        )
        metrics["measurement_truth_check"] = {
            "index0": expected_measurement,
            "final_distance": final_distance,
            "measurement_abs": measurement_abs,
            "measurement_rel": measurement_rel,
            "within_tolerance": within_tolerance,
        }
        if not within_tolerance:
            reject(
                "mixed_topology_recovery_outside_truth_tolerance",
                (
                    "production measurement estimate remains outside the "
                    "declared strict-audit tolerance"
                ),
                failed_stage="measurement_truth_check",
            )

    def _compose_measurement(
        self,
        scenario: dict[str, Any],
        *,
        offsets: int,
        family: str,
        index: int,
    ) -> dict[str, Any]:
        base_scenario_id = str(scenario["scenario_id"])
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
        self._declare_measurement_recovery_tolerance(
            composed,
            [int(item["index"]) for item in errors if item.get("index") is not None],
        )
        self._require_anomalous("case14", measurements, family)
        corrected_case = str(composed.get("clean_case") or "case14")
        if corrected_case != "case14":
            # In branch+measurement recovery the branch family may correctly
            # resolve first. The independent bad meter must still be observable
            # under that repaired model; otherwise exact sequential recovery is
            # order-dependent and a truth-free policy can finalize too early.
            self._require_anomalous(corrected_case, measurements, family)
        if family == "measurement+topology":
            self._require_mixed_topology_recovery_realizable(composed)
        if family == "measurement+parameter":
            self._require_mixed_parameter_recovery_realizable(
                composed,
                base_scenario_id=base_scenario_id,
            )
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
            hidden = scenario.get("hidden_truth")
            hidden = hidden if isinstance(hidden, Mapping) else {}
            truth_keys = (
                "true_measurement_errors",
                "true_parameter_errors",
                "true_topology_errors",
            )
            diagnostic_truth_keys = (
                "true_harmonic_errors",
                "true_hif_errors",
                "true_unbalance_errors",
            )
            error_cardinality = sum(
                len(scenario.get(key) or []) for key in truth_keys
            ) + sum(len(hidden.get(key) or []) for key in diagnostic_truth_keys)
            if family == "topology" or "topology" in family:
                source_tier = "physics_synthesized"
            elif family == "telemetry_no_disturbance":
                source_tier = "derived_negative_control"
            elif family in {"harmonic", "hif", "three_phase_unbalance", "measurement+hif"}:
                source_tier = "tracked_diagnostic_corpus"
            elif "+" in family or family == "multi_measurement":
                source_tier = "tracked_composed_corpus"
            else:
                source_tier = "tracked_measurement_corpus"
            scenario["network_case"] = str(
                scenario.get("network_case") or "case14"
            )
            scenario["error_cardinality"] = int(error_cardinality)
            scenario["source_tier"] = source_tier
            self.manifest.append(
                {
                    "scenario_id": scenario["scenario_id"],
                    "scenario_family": family,
                    "source": source,
                    "network_case": scenario["network_case"],
                    "error_cardinality": scenario["error_cardinality"],
                    "source_tier": scenario["source_tier"],
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
        order_population = len(source_rows)
        if family in {"hif", "measurement+hif"}:
            order_population = max(
                order_population,
                int(self._hif_order_population_size or 0),
            )
        raw_order = self._rng.permutation(order_population)
        order = list(
            dict.fromkeys(
                int(position) % len(source_rows) for position in raw_order
            )
        ) if source_rows else []
        for position in order:
            if len(built) >= count:
                break
            row = source_rows[position]
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
        if family == "three_phase_unbalance":
            return self._imbalance_rows(), self._unbalance_scenario
        if family == "telemetry_no_disturbance":
            return self._imbalance_rows(), self._telemetry_no_disturbance_scenario
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
        record = {
            "scenario_family": family,
            "source": source,
            "reason": rejection.reason,
            "detail": str(rejection),
        }
        if rejection.metrics:
            record["metrics"] = copy.deepcopy(rejection.metrics)
        self.skipped.append(record)

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
            "source_partition": copy.deepcopy(self._source_partition_metadata),
            "parameter_ranking_admission": {
                "contract": PARAMETER_RANKING_CONTRACT,
                "enforced": self._enforce_parameter_ranking_dominance,
                "threshold": self.parameter_ranking_dominance_threshold,
            },
            "synthesized_measurement_canonicalization": {
                "contract": SYNTHESIZED_MEASUREMENT_CANONICALIZATION_CONTRACT,
                "scope": "pypower_topology_z_obs",
                "decimal_quantum": "1e-12",
                "rounding": "half_even",
                "application": "post_admission_pre_scenario_materialization",
                "canonical_vector_revalidated": True,
                "maximum_absolute_projection": "5e-13",
            },
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
    "DEFAULT_HIF_FALLBACK_SAMPLE_PATHS",
    "DEFAULT_RELEASE_HIF_SAMPLE_PATHS",
    "DEFAULT_RELEASE_HIF_QUALITY_PATHS",
    "DEFAULT_IMBALANCE_SAMPLE_PATH",
    "LEGACY_IMBALANCE_SAMPLE_PATH",
    "CURRENT_TELEMETRY_HIF_SAMPLE_PATHS",
    "CURRENT_TELEMETRY_IMBALANCE_SAMPLE_PATH",
    "UNBALANCE_SIGNATURE",
    "UNBALANCE_CURRENT_SIGNATURE",
    "DEFAULT_BALANCED_ARTIFACT_DIR",
    "DEFAULT_CHI2_ALPHA",
    "SYNTHESIZED_MEASUREMENT_CANONICALIZATION_CONTRACT",
]
