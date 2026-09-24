"""Round-0 scenario synthesis for the recovery-balanced DAgger aggregate.

Builds ``TransactionalPSSEEnv`` scenarios on the real MATPOWER/OpenDSS stack
from three physically consistent sources:

- the production measurement corpus (``data/measurements_5class_merged.jsonl``)
  for no-error, gross-measurement, parameter (with multi-scan telemetry), and
  harmonic snapshots;
- the OpenDSS HIF sample sets (``artifacts/measurements/hif_multiscan_*``)
  for high-impedance-fault snapshots with real NLM diagnostics and persistent
  scan windows;
- direct synthesis on AC-OPF operating points for topology errors and
  harmonic distortion.  Topology roots are single breaker-status errors in the
  full IEEE-14 node/breaker model (``Transmission/ieee14_full_topology.py``)
  whose effect the operator's branch-status correction can represent; the
  corpus topology rows are pocket-model CB events without that guarantee.
  Harmonic roots re-run the legacy harmonic synthesis on a load-scaled OPF
  point because every tracked harmonic row sits at unit load on planning
  voltages.  Both families draw the load scale from the corpus range and use
  the corpus solver and noise, so no family is identifiable by its dispatch,
  load level or noise floor.

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

Fresh balanced corpora can select a registered system explicitly. IEEE 57
currently supports clean, measurement, parameter and their measurement
compositions. ``admission_mode='physical'`` preserves physically validated
development roots without conditioning on WLS detection or teacher success;
the default ``recoverable`` retains the existing IEEE 14 admission behavior.
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
from psse_env.evidence_profile import (
    DEFAULT_EVIDENCE_PROFILE, AUXILIARY_EVIDENCE_PROFILE,
    allows_diagnostic_tools, is_scada_only, is_strict_boundary,
    validate_evidence_profile, sanitize_execution_for_profile,
)

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
from psse_env.systems import resolve_system

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHI2_ALPHA = 0.01
# Local test paired with the global chi-square test: an anomaly is present
# when either fires.  Four sigma was chosen on fresh clean windows of both
# networks (3 sigma flags 63% of healthy IEEE 57 windows and 30% of IEEE 14
# ones, 3.5 sigma 22% and 6%, 4 sigma 5% and 2%) while every ten-sigma meter
# fault is caught at any of the three; see the pipeline README.
DEFAULT_NORMALIZED_RESIDUAL_THRESHOLD = 4.0

# The tabular corpus families were synthesized on AC-OPF operating points with
# loads scaled uniformly over this range (Transmission/generate_measurements.py
# DEFAULTS).  Synthesized families draw from the same range and solver so no
# family is identifiable by its dispatch or load level.
SYNTHESIZED_LOAD_SCALE_RANGE = (0.80, 1.25)
# Breaker-error classes the topology family can sample; catalogue categories
# such as ``merge_10_14`` map onto ``merge``.
TOPOLOGY_EFFECTS = frozenset({"dangling_line_terminal", "bus_split", "merge"})


def _breaker_effect_class(category: str) -> str:
    return "merge" if str(category).startswith("merge") else str(category)
# Legacy harmonic synthesis controls (Transmission/generate_measurements.py
# HARMONIC_DEFAULT_CANDIDATES and make_harmonic_anomaly_record).
HARMONIC_SOURCE_CANDIDATES = (2, 3, 4, 5, 9, 10, 11, 12, 13, 14)
HARMONIC_THD_RANGE = (0.10, 0.20)

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
# Physical-ohm corpora are opt-in; historical telemetry paths retain replay identity.
# PMU phasor precision of the wls_gated_diagnostics study (2026-09-23): the HIF corpora are
# regenerated with three-phase voltage and branch-current phasor noise of 1e-4 pu per
# rectangular component (the 2026-09-19/21/23 corpora used 5e-3 / 1e-3), and both the HIF and
# the unbalance corpora take OPF-driven operating points (dispatch, setpoints and source voltage
# from the AC-OPF at the window's load scale, as the pypower families already do).  The
# phasor sigma leaves the SCADA draws unchanged; the OPF dispatch does not, so the detectable
# counts of the 20260923opf corpora were only known after the regeneration and are resolved
# from the artifact directory names below rather than hard-coded (the 2026-09-23 regeneration
# admitted 27 + 8 + 77 + 19 HIF windows, the same ids as 20260923b, and 162 unbalance windows).
PMU_PHASOR_SIGMA_PU = 1e-4
PHYSICAL_HIF_CORPUS_TAG = "20260923opf"
_PHYSICAL_MEASUREMENTS_DIR = _REPO_ROOT / "artifacts" / "measurements"


def resolve_tagged_corpus_path(stem: str, tag: str = PHYSICAL_HIF_CORPUS_TAG, *,
                               suffix: str = "x10", root: Path = _PHYSICAL_MEASUREMENTS_DIR) -> Path:
    """``samples.jsonl`` of ``<stem>_<N><suffix>_<tag>`` with ``N`` read from the directory name.

    Detectable subsets carry their admitted window count in the name, which is
    unknown until the corpus is regenerated.  Exactly one matching directory
    resolves to its samples file; none resolves to a placeholder path with
    ``PENDING`` in place of the count (``Path.is_file()`` is False, so the
    generator's source lookups skip it and the pipeline's existence checks
    fail loudly); several matches are ambiguous and raise.
    """
    pattern = f"{stem}_*{suffix}_{tag}"
    matches = sorted(candidate for candidate in root.glob(pattern) if (candidate / "samples.jsonl").is_file())
    if len(matches) > 1:
        raise RuntimeError(f"ambiguous corpus directories for {pattern!r}: {[m.name for m in matches]}")
    if matches:
        return matches[0] / "samples.jsonl"
    return root / f"{stem}_PENDING{suffix}_{tag}" / "samples.jsonl"


# The 20260923opf corpora (regenerated 2026-09-23, docs/opf_operating_points_20260923.md):
# HIF corpora at PMU sigma 1e-4 with OPF-driven operating points on the regulated, tightly
# converged OpenDSS model, then the discovered-mode detectable subsets (27 + 8 + 77 + 19
# windows, tracked).  The resolver still returns a PENDING placeholder if a subset directory
# is missing from a checkout; pass explicit hif_sample_paths to run on the committed
# 20260923b corpora (5e-3 / 1e-3 phasor sigma, model-default dispatch).
PHYSICAL_HIF_SAMPLE_PATHS = tuple(resolve_tagged_corpus_path(stem) for stem in (
    # Detectable subsets (discovered-mode WLS admission, margin 1.25) of the physical 69 kV
    # 100-1000 ohm corpora; the full corpora carry the same names without "_detectable"
    # (84 + 21 + 252 + 63 windows).
    "hif_physical69_main_train_detectable", "hif_physical69_main_valid_detectable",
    "hif_physical69_main_train_extra_detectable", "hif_physical69_main_valid_extra_detectable"))
# The 2026-09-23b detectable subsets (regulated reactive limits, OpenDSS tolerance 1e-8,
# 5e-3 / 1e-3 phasor sigma, model-default dispatch): 27 + 8 + 77 + 19 windows, committed.
PHYSICAL_HIF_SAMPLE_PATHS_20260923B = tuple(_PHYSICAL_MEASUREMENTS_DIR / name / "samples.jsonl" for name in (
    "hif_physical69_main_train_detectable_27x10_20260923b", "hif_physical69_main_valid_detectable_8x10_20260923b",
    "hif_physical69_main_train_extra_detectable_77x10_20260923b", "hif_physical69_main_valid_extra_detectable_19x10_20260923b"))
# The 2026-09-21 detectable subsets (5e-3 / 1e-3 phasor sigma, pre-fix reactive limits) that
# the 2026-09-21 cell ran on; kept for replay identity of that cell only.
PHYSICAL_HIF_SAMPLE_PATHS_20260921 = tuple(_PHYSICAL_MEASUREMENTS_DIR / name / "samples.jsonl" for name in (
    "hif_physical69_main_train_detectable_25x10_20260921", "hif_physical69_main_valid_detectable_7x10_20260921",
    "hif_physical69_main_train_extra_detectable_69x10_20260921", "hif_physical69_main_valid_extra_detectable_17x10_20260921"))
# The detection-limit and sweep corpora follow the tag; their window counts are fixed by the
# recipe (21 and 336), so the names are known.  Like their 20260923b predecessors they are
# generated artifacts that are not tracked in git.
PHYSICAL_HIF_DETECTION_LIMIT_SAMPLE_PATH = _PHYSICAL_MEASUREMENTS_DIR / f"hif_physical69_detection_limit_21x10_{PHYSICAL_HIF_CORPUS_TAG}" / "samples.jsonl"
PHYSICAL_HIF_SWEEP_SAMPLE_PATH = _PHYSICAL_MEASUREMENTS_DIR / f"hif_physical_sweep_eval_336x10_{PHYSICAL_HIF_CORPUS_TAG}" / "samples.jsonl"
# Unbalance corpus under the WLS shunt convention (ybus), phase-A Vm, physical telemetry bases;
# 440 windows + 60 balanced controls, seed 20260925.  Regenerated 2026-09-23 (20260923opf) with
# OPF-driven operating points (phasor sigmas unchanged): 162 windows admitted, 157 of them in
# common with 20260923b; the admitted count is resolved from the directory name.  The
# committed 20260923b subset (160 windows) is kept beside it.
PHYSICAL_IMBALANCE_SAMPLE_PATH = resolve_tagged_corpus_path("out_measurements_imbalance_currents_ybus_detectable", suffix="")
PHYSICAL_IMBALANCE_SAMPLE_PATH_20260923B = _PHYSICAL_MEASUREMENTS_DIR / "out_measurements_imbalance_currents_ybus_detectable_160_20260923b" / "samples.jsonl"
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
# How a waveform family's root announces itself.  ``flagged`` seeds the
# sensor signature at reset, as if a power-quality monitor or relay had raised
# it.  ``discovered`` withholds it: the operator starts from the
# positive-sequence snapshot and the balanced model alone, the first
# observable evidence is the WLS anomaly.  Attributing it requires additional
# measured telemetry; positive-sequence SCADA alone does not identify an
# unbalance or harmonic source. Every waveform family defaults to discovery;
# auxiliary telemetry requires an explicitly selected evidence profile.
WAVEFORM_SIGNATURE_MODES = ("flagged", "discovered")
DEFAULT_WAVEFORM_SIGNATURE_MODE = {
    "harmonic": "discovered",
    "three_phase_unbalance": "discovered",
    "hif": "discovered",
}

# The tabular measurement corpus is shared by both the round-0 aggregate and
# the frozen evaluation-suite builder.  Assign its physical source rows before
# seeded sampling so changing a corpus row id (or copying the same physical row
# under a second id) cannot move that realization across the release boundary.
_SOURCE_PARTITION_FAMILIES = (
    "no_error",
    "measurement",
    "multi_measurement",
    "parameter",
    "measurement+parameter",
)
_SOURCE_PARTITION_CORPUS_SCENARIOS = frozenset(
    {"no_error", "measurement_error", "parameter_error"}
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


def _canonicalize_telemetry(telemetry: Mapping[str, Any]) -> dict[str, Any]:
    """Project every substation reading onto the release decimal lattice."""

    canonical = copy.deepcopy(dict(telemetry))
    for key in ("node_vm", "node_pinj", "node_qinj", "cb_p", "cb_q"):
        values = canonical.get(key)
        if isinstance(values, Mapping) and values:
            names = list(values)
            quantized = _canonicalize_synthesized_measurement_vector(
                [float(values[name]) for name in names]
            )
            canonical[key] = dict(zip(names, quantized))
    for key in ("branch_pf", "branch_qf", "branch_pt", "branch_qt"):
        values = canonical.get(key)
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
            canonical[key] = _canonicalize_synthesized_measurement_vector(
                [float(value) for value in values]
            )
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
        system: str = "case14",
        admission_mode: str = "recoverable",
        corpus_path: str | Path | None = None,
        hif_sample_paths: Sequence[str | Path] | None = None,
        imbalance_sample_path: str | Path | None = None,
        balanced_artifact_dir: str | Path | None = None,
        artifact_allowlist: Sequence[str | Path] | None = None,
        derived_case_dir: str | Path | None = None,
        seed: int = 20260719,
        validate: bool = True,
        chi2_alpha: float = DEFAULT_CHI2_ALPHA,
        normalized_residual_threshold: float | None = None,
        anomaly_margin: float = 1.25,
        topology_noise_scale: float = 1.0,
        enforce_topology_ranking: bool = True,
        topology_effects: Sequence[str] = ("dangling_line_terminal", "bus_split"),
        require_branch_dominant_topology: bool = True,
        hif_max_scans: int = 8,
        noise_profile_rows: int = 200,
        source_partition: str | None = None,
        parameter_ranking_dominance_threshold: float | None = None,
        enforce_parameter_ranking_dominance: bool | None = None,
        parameter_target_rank_allowance: int | None = None,
        unbalance_vuf_threshold: float = DEFAULT_UNBALANCE_VUF_THRESHOLD,
        waveform_signature_mode: Mapping[str, str] | None = None,
        min_measurement_error_sigma: float | None = None,
        evidence_profile: str = DEFAULT_EVIDENCE_PROFILE,
    ) -> None:
        self.evidence_profile = validate_evidence_profile(evidence_profile)
        self.system = resolve_system(system)
        self.case_path = self.system.case_path
        self.nb, self.nl = self.system.nb, self.system.nl
        self.nz, self.state_count = self.system.nz, self.system.state_count
        if admission_mode not in {"recoverable", "physical"}:
            raise ValueError("admission_mode must be 'recoverable' or 'physical'")
        self.admission_mode = admission_mode
        if self.system.case_id != "case14":
            if corpus_path is None or balanced_artifact_dir is None:
                raise ValueError("non-IEEE14 systems require an explicit fresh corpus and artifact directory")
            if hif_sample_paths or imbalance_sample_path or waveform_signature_mode:
                raise ValueError("waveform/three-phase sources are unsupported for this system")
            hif_sample_paths = []
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
        # None keeps the historical chi-square-only admission; with a
        # threshold the admission tests mirror the environment's rule:
        # anomalous when either test fires, clean only when neither does.
        if normalized_residual_threshold is not None and (
            isinstance(normalized_residual_threshold, bool)
            or not math.isfinite(float(normalized_residual_threshold))
            or float(normalized_residual_threshold) <= 0.0
        ):
            raise ValueError("normalized_residual_threshold must be positive or None")
        self.normalized_residual_threshold = (
            None if normalized_residual_threshold is None else float(normalized_residual_threshold)
        )
        self.anomaly_margin = float(anomaly_margin)
        self.topology_noise_scale = float(topology_noise_scale)
        # A topology root is admitted only if the node/breaker estimator ranks
        # the true breaker first on the reported statuses, so the substation
        # investigation the operator performs can identify it.
        self.enforce_topology_ranking = bool(enforce_topology_ranking)
        # Breaker-error classes offered as topology roots.  An isolated line
        # terminal keeps the 14-bus operator layout; a bus split re-renders the
        # operator model with one more bus and a merge with one fewer once the
        # breaker is corrected.  "merge" is supported but not a default: on the
        # operator's 14-bus model the 10/14 merge is residual-dominant (it looks
        # like a bad meter), so the branch-dominance admission below rejects
        # every merge draw; sampling merges needs that gate off and accepts a
        # measurement-first deployed route.
        self.topology_effects = tuple(str(effect) for effect in topology_effects)
        self.require_branch_dominant_topology = bool(require_branch_dominant_topology)
        unknown_effects = set(self.topology_effects) - TOPOLOGY_EFFECTS
        if unknown_effects or not self.topology_effects:
            raise ValueError(
                "topology_effects must be a non-empty subset of "
                f"{sorted(TOPOLOGY_EFFECTS)}, got {sorted(unknown_effects)}"
            )
        self.hif_max_scans = int(hif_max_scans)
        self.noise_profile_rows = int(noise_profile_rows)
        self.source_partition = source_partition
        if not (0.0 < float(unbalance_vuf_threshold) < 1.0):
            raise ValueError("unbalance_vuf_threshold must be a fraction in (0, 1)")
        self.unbalance_vuf_threshold = float(unbalance_vuf_threshold)
        # Gross-error floor for injected meter faults, in multiples of the
        # per-index noise profile.  The tracked corpus injects errors whose
        # magnitude is centred near ten sigma but reaches down to five, and a
        # five-to-seven sigma voltage bias among larger peers is exactly what
        # the residual test loses once the peers are repaired.  With a floor,
        # a corpus error below it is rescaled (same sign, random 1.0-1.5x the
        # floor) before admission, and composed overlays are lifted the same
        # way; every existing admission check still runs on the result.
        self.min_measurement_error_sigma = (
            None if min_measurement_error_sigma is None else float(min_measurement_error_sigma)
        )
        if self.min_measurement_error_sigma is not None and self.min_measurement_error_sigma <= 0:
            raise ValueError("min_measurement_error_sigma must be positive")
        modes = dict(DEFAULT_WAVEFORM_SIGNATURE_MODE)
        for family, mode in dict(waveform_signature_mode or {}).items():
            if family not in modes:
                raise ValueError(f"unknown waveform signature family: {family!r}")
            if mode not in WAVEFORM_SIGNATURE_MODES:
                raise ValueError(
                    f"waveform signature mode for {family!r} must be one of {WAVEFORM_SIGNATURE_MODES}"
                )
            modes[family] = str(mode)
        self.waveform_signature_mode = modes
        # Every strict-boundary profile (scada_only, wls_gated_diagnostics)
        # withholds the sensor flag: the first observable evidence is the
        # balanced WLS alarm.  Seeded signatures are historical reproduction.
        if is_strict_boundary(self.evidence_profile) and any(mode == "flagged" for mode in modes.values()):
            raise ValueError("flagged waveform signatures require evidence_profile='auxiliary_diagnostics'")
        # The frozen evaluation suite deliberately preserves its previously
        # approved physical roots, including hard/ambiguous parameter cases.
        # Dominance is a single-label *training admission* requirement, not a
        # reason to rewrite a pinned holdout.  By default the evaluation
        # partition therefore keeps the legacy rank-one inventory (gate off,
        # threshold 1.0) while train/unspecified generation enforces the
        # release threshold below.  A builder whose release contract gates
        # parameter routes at evaluation time (the BC0 suite builder since the
        # 2026-09-03 re-freeze) passes ``enforce_parameter_ranking_dominance``
        # explicitly so frozen roots are admitted under the same rule the
        # release teacher evaluates them with.
        self._enforce_parameter_ranking_dominance = (
            source_partition != "evaluation"
            if enforce_parameter_ranking_dominance is None
            else bool(enforce_parameter_ranking_dominance)
        )
        # ``None`` keeps the release rule that the deployed parameter context
        # must rank the true line first.  A positive allowance admits a root
        # whose true line ranks anywhere within that many candidates, which is
        # how a development suite drawn at the detection threshold keeps the
        # adjacent-line ambiguity the network really has; the recorded rank
        # stratifies the results afterwards.
        self._parameter_target_rank_allowance = (
            None if parameter_target_rank_allowance is None else int(parameter_target_rank_allowance)
        )
        if (
            self._parameter_target_rank_allowance is not None
            and self._parameter_target_rank_allowance < 1
        ):
            raise ValueError("parameter_target_rank_allowance must be positive")
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
            evidence_profile=self.evidence_profile,
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
        self._fresh_corpus = False
        self._hif_samples: list[dict[str, Any]] | None = None
        self._imbalance_samples: list[dict[str, Any]] | None = None
        self._hif_order_population_size: int | None = None
        self._noise_std: np.ndarray | None = None
        self._base_case: dict[str, Any] | None = None
        self._full_topology = None
        self._full_topology_fingerprint: str | None = None
        self._breaker_errors: list[dict[str, Any]] | None = None
        self._wls_cache: dict[tuple[str, str], tuple[float, float]] = {}

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

        # Fresh corpora retain a parent operating-window identity across noise
        # variants and overlays. Legacy corpora retain their exact old hashes.
        if row.get("source_realization_id"):
            return hashlib.sha256(json.dumps({
                "network_case": row.get("network_case"),
                "base_case_hash": row.get("base_case_hash"),
                "source_realization_id": row["source_realization_id"],
            }, sort_keys=True).encode("utf-8")).hexdigest()
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

    def _validate_source_row(self, row: Mapping[str, Any]) -> None:
        """Fail on mixed system/layout inputs before any family selection."""
        declared = row.get("network_case")
        fresh = declared is not None or self.system.case_id != "case14"
        if not fresh and self.admission_mode == "recoverable":
            return  # Preserve the existing IEEE14 corpus and admission contract.
        if fresh and declared != self.system.case_id:
            raise ValueError(f"corpus system mismatch: expected {self.system.case_id}, got {declared!r}")
        if fresh and row.get("base_case_hash") != self.system.base_case_hash:
            raise ValueError("corpus base-case hash does not match the selected system")
        if fresh and row.get("sigmas") != {"vm": 0.001, "inj": 0.01, "flow": 0.01}:
            raise ValueError("corpus covariance differs from the deployed WLS noise model")
        self._fresh_corpus = self._fresh_corpus or fresh
        if fresh or self.admission_mode == "physical":
            validation = row.get("physical_validation") or {}
            if validation.get("passed") is not True:
                raise ValueError("fresh/physical corpus rows require passed physical validation")
            if not str(row.get("source_realization_id") or "").strip():
                raise ValueError("fresh corpus row is missing parent realization identity")
        for key in ("z_obs", "z_true"):
            values = np.asarray(row.get(key), dtype=float)
            if values.shape != (self.nz,) or not np.isfinite(values).all():
                raise ValueError(f"{key} must contain {self.nz} finite measurements for {self.system.case_id}")
        if row.get("z_scans") is not None:
            scans = np.asarray(row["z_scans"], dtype=float)
            if scans.ndim != 2 or scans.shape[1] != self.nz or not np.isfinite(scans).all():
                raise ValueError("parameter scan layout does not match the selected system")

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
                    self._validate_source_row(row)
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

    @staticmethod
    def _waveform_source_rows(path: Path) -> list[dict[str, Any]]:
        """Keep historical noise evidence private until scenario preparation."""
        metadata_path = path.with_name("meta.json")
        source_meta = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else None
        rows = []
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("scenario") == "no_error":
                    continue
                row["_source_noise_metadata"] = copy.deepcopy(source_meta)
                rows.append(row)
        return rows

    def _aligned_waveform_row(self, row: Mapping[str, Any], family: str) -> dict[str, Any]:
        from three_phase_nlm.measurement_noise import align_legacy_waveform_row
        try:
            aligned = align_legacy_waveform_row(
                row, family, self._rng, legacy_metadata=row.get("_source_noise_metadata")
            )
            contract = aligned.get("noise_contract")
            if not isinstance(contract, Mapping) or contract.get("schema") != "generated_sensor_noise_v1":
                raise ValueError("Weighting sigmas alone do not establish applied noise; regenerate with an aligned generator or provide supported source metadata")
            return aligned
        except (TypeError, ValueError, KeyError) as exc:
            raise ScenarioRejected("waveform_noise_contract_invalid", f"{row.get('id')}: {exc}") from exc

    @staticmethod
    def _observable_waveform_scan(scan: Mapping[str, Any]) -> dict[str, Any]:
        """Allowlist measurements/configuration; clean copies and labels stay offline."""
        from three_phase_nlm.hif_operating_point import canonicalize_ieee14_operating_point
        allowed = ("scan_index", "z_obs", "three_phase_voltages", BRANCH_CURRENT_CHANNEL,
                   BRANCH_CURRENT_SIGMA_KEY, "sigma_z", "three_phase_sigma", "topology_id",
                   "noise_contract", "time_tag", "measurement_convention")
        observed = {key: copy.deepcopy(scan[key]) for key in allowed if key in scan}
        if "op_point" in scan:
            observed["op_point"] = canonicalize_ieee14_operating_point(scan["op_point"])
        return observed

    def _hif_rows(self) -> list[dict[str, Any]]:
        if self._hif_samples is None:
            rows: list[dict[str, Any]] = []
            used_fallback = False
            for path in self.hif_sample_paths:
                if not path.is_file():
                    continue
                rows.extend(self._waveform_source_rows(path))
            if not rows:
                used_fallback = True
                for path in self.hif_fallback_sample_paths:
                    if not path.is_file():
                        continue
                    rows.extend(self._waveform_source_rows(path))
            self._hif_samples = [self._normalize_hif_row(row) for row in rows]
            # Legacy primary corpora cover 17 lines; physical corpora use 16
            # same-voltage lines (7 in the main 69 kV population). Preserve
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
                rows.extend(self._waveform_source_rows(self.imbalance_sample_path))
            self._imbalance_samples = rows
        return self._imbalance_samples

    @staticmethod
    def _normalize_hif_row(row: Mapping[str, Any]) -> dict[str, Any]:
        """Promote a legacy HIF snapshot to the current scan-window schema.

        Promotion changes the shape only. The later noise-contract check still
        requires evidence for every channel before admitting legacy telemetry.
        """
        normalized = copy.deepcopy(dict(row))
        if normalized.get("scans"):
            return normalized
        required = ("z_obs", "three_phase_voltages")
        if any(not normalized.get(key) for key in required):
            return normalized
        topology_id = str(normalized.get("topology_id") or "ieee14_base")
        promoted_scan = {
            "scan_index": 0,
            "z_obs": copy.deepcopy(normalized["z_obs"]),
            "three_phase_voltages": copy.deepcopy(
                normalized["three_phase_voltages"]
            ),
            "op_point": copy.deepcopy(normalized.get("op_point") or {}),
            "topology_id": topology_id,
        }
        # z_true may be a balanced counterfactual, not this HIF's sensor mean.
        for key in ("z_clean", "three_phase_voltages_clean", "sigma_z", "three_phase_sigma", "noise_contract", "measurement_convention"):
            if key in normalized:
                promoted_scan[key] = copy.deepcopy(normalized[key])
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
        """Declared SCADA sigma shared by generation, error floors and WLS."""
        return self.system.measurement_sigma()

    def _floored_measurement_error(
        self, index: int, observed: float, clean: float, *, sigma_z=None
    ) -> tuple[float, bool]:
        """Lift a meter error below the sigma floor; return (observed, lifted)."""

        floor = self.min_measurement_error_sigma
        if floor is None:
            return float(observed), False
        sigma = float((self.noise_profile() if sigma_z is None else sigma_z)[int(index)])
        delta = float(observed) - float(clean)
        if abs(delta) >= floor * sigma:
            return float(observed), False
        sign = 1.0 if delta > 0 else -1.0 if delta < 0 else (1.0 if self._rng.random() < 0.5 else -1.0)
        magnitude = floor * sigma * float(self._rng.uniform(1.0, 1.5))
        return float(clean) + sign * magnitude, True

    def _clean_case(self) -> dict[str, Any]:
        if self._base_case is None:
            self._base_case = _load_python_case(self.case_path)
        return self._base_case

    # ------------------------------------------------------------- validation

    def _wls_detection(self, case: str, z: Sequence[float], *, sigma_z=None, exact_measurement_indices=None) -> tuple[float, float]:
        """Chi-square statistic and maximum absolute normalized residual of one solve."""
        noise_key = (None if sigma_z is None else tuple(float(v) for v in sigma_z), tuple(exact_measurement_indices or []))
        key = (case, hashlib.sha256(np.asarray(z, dtype=float).tobytes()).hexdigest(), noise_key)
        if key not in self._wls_cache:
            options = {}
            if sigma_z is not None:
                options["measurement_sigma"] = list(sigma_z)
            if exact_measurement_indices:
                options["exact_measurement_indices"] = list(exact_measurement_indices)
            payload = _wls_json(case, [float(value) for value in z], **options)
            if not payload.get("success"):
                raise ScenarioRejected("wls_failure", str(payload.get("error")))
            residuals = np.asarray(payload.get("r") or [], dtype=float)
            maximum = float(np.max(np.abs(residuals))) if residuals.size else 0.0
            self._wls_cache[key] = (float(payload.get("global_residual_sum") or 0.0), maximum)
            if not hasattr(self, "_wls_dof_cache"):
                self._wls_dof_cache = {}
            self._wls_dof_cache[key] = int(payload.get("dof", self.nz - self.state_count))
        return self._wls_cache[key]

    def _cached_chi2_limit(self, case: str, z: Sequence[float], *, sigma_z=None, exact_measurement_indices=None) -> float:
        noise_key = (None if sigma_z is None else tuple(float(v) for v in sigma_z), tuple(exact_measurement_indices or []))
        key = (case, hashlib.sha256(np.asarray(z, dtype=float).tobytes()).hexdigest(), noise_key)
        dof = getattr(self, "_wls_dof_cache", {}).get(key, self.nz - self.state_count)
        return float(chi2_threshold(max(1, int(dof)), self.chi2_alpha))

    def _chi2_statistic(self, case: str, z: Sequence[float], **noise_options) -> float:
        return self._wls_detection(case, z, **noise_options)[0]

    def _max_normalized_residual(self, case: str, z: Sequence[float], **noise_options) -> float:
        return self._wls_detection(case, z, **noise_options)[1]

    def _residual_alarm(self, case: str, z: Sequence[float], *, margin: float = 1.0, **noise_options) -> bool:
        """The local test at ``margin`` times the threshold; False when disabled."""
        if self.normalized_residual_threshold is None:
            return False
        return self._max_normalized_residual(case, z, **noise_options) >= margin * self.normalized_residual_threshold

    def _detection_detail(self, case: str, z: Sequence[float], **noise_options) -> str:
        statistic, maximum = self._wls_detection(case, z, **noise_options)
        detail = f"chi2 {statistic:.1f} vs {self._cached_chi2_limit(case, z, **noise_options):.1f}"
        if self.normalized_residual_threshold is not None:
            detail += f", max|r| {maximum:.2f} vs {self.normalized_residual_threshold:.2f}"
        return detail

    @property
    def chi2_limit(self) -> float:
        dof = max(1, self.nz - self.state_count)
        return float(chi2_threshold(dof, self.chi2_alpha))

    def _require_anomalous(self, case: str, z: Sequence[float], family: str, **noise_options) -> None:
        """Admit only faults either test detects with the anomaly margin to spare."""
        if not self.validate or self.admission_mode == "physical":
            return
        statistic = self._chi2_statistic(case, z, **noise_options)
        chi_detectable = statistic > self.anomaly_margin * self._cached_chi2_limit(case, z, **noise_options)
        if chi_detectable or self._residual_alarm(case, z, margin=self.anomaly_margin, **noise_options):
            return
        raise ScenarioRejected(
            "anomaly_not_detectable",
            f"{family}: {self._detection_detail(case, z, **noise_options)} with margin {self.anomaly_margin:.2f}",
        )

    def _require_clean(self, case: str, z: Sequence[float], family: str, **noise_options) -> None:
        """A corrected configuration must clear both tests, as the environment requires."""
        if not self.validate or self.admission_mode == "physical":
            return
        statistic = self._chi2_statistic(case, z, **noise_options)
        if statistic >= self._cached_chi2_limit(case, z, **noise_options) or self._residual_alarm(case, z, **noise_options):
            raise ScenarioRejected(
                "corrected_configuration_still_anomalous",
                f"{family}: {self._detection_detail(case, z, **noise_options)}",
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
        sigma_z: Sequence[float] | None = None,
        sigma_z_scans: Sequence[float] | None = None,
    ) -> dict[str, Any] | None:
        """Truth-side gate for the deployed multi-scan parameter corrector.

        Release scenarios must be recoverable by the same correction entry
        point that the transactional environment invokes.  The estimator is
        evaluated only while constructing/auditing scenarios; its output and
        comparison with the clean target are never attached to an accepted
        scenario or exposed to the online policy.
        """
        if not self.validate or self.admission_mode == "physical":
            return None
        line_index1 = int(line_row0) + 1
        normalized_scans = [
            [float(value) for value in scan] for scan in z_scans
        ]
        initial_states = observable_parameter_initial_states(
            self._clean_case(), normalized_scans
        )
        payload = _param_correction_json(
            self.case_path,
            line_index1,
            normalized_scans,
            initial_states,
            **({"R_variances_full": np.square(sigma_z_scans).tolist()} if sigma_z_scans is not None else {}),
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

        corrected_case = copy.deepcopy(self._clean_case())
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
                "metadata": {"sigma_z": list(sigma_z)} if sigma_z is not None else {},
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
                "case": self.case_path,
                "measurements": [float(value) for value in measurements],
                "metadata": {
                    **({"sigma_z": list(sigma_z)} if sigma_z is not None else {}),
                    "parameter_scans": {
                        "z_scans": copy.deepcopy(normalized_scans),
                        **({"sigma_z": list(sigma_z_scans)} if sigma_z_scans is not None else {}),
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
        true_line_rank = (
            supported_lines.index(line_index1) + 1
            if line_index1 in supported_lines
            else None
        )
        metrics["parameter_context_true_line_rank"] = true_line_rank
        allowance = self._parameter_target_rank_allowance
        admitted_by_rank = (
            true_line_rank is not None and true_line_rank <= allowance
            if allowance is not None
            else first_supported_line == line_index1
        )
        if not admitted_by_rank:
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
            "parameter_context_ranking": copy.deepcopy(parameter_ranking),
            "parameter_context_true_line_rank": true_line_rank,
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
        if self.system.case_id != "case14":
            parts = (self.system.case_id, *parts)
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

    def _base_scenario(
        self,
        scenario_id: str,
        *,
        case: str,
        measurements: Sequence[float],
        family: str,
        sigma_z: Sequence[float] | None = None,
    ) -> dict[str, Any]:
        return {
            "scenario_id": scenario_id,
            "root_scenario_id": scenario_id,
            "scenario_family": family,
            # This is the stable network identity used for grouped/stratified
            # release reporting. ``case`` may later become a content-addressed
            # derived artifact after a parameter/topology correction and must
            # not fragment one IEEE-14 population into one pseudo-case per root.
            "network_case": self.case_path,
            "case": case,
            "measurements": [float(value) for value in measurements],
            "semantic_field_provenance": {"measurements": _SNAPSHOT_PROVENANCE},
            "metadata": {"sigma_z": list(sigma_z) if sigma_z is not None else self.noise_profile().tolist(),
                         "evidence_profile": self.evidence_profile},
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
        sigma = list((scenario.get("metadata") or {}).get("sigma_z", self.noise_profile()))
        valid = [
            int(index)
            for index in indices
            if 0 <= int(index) < len(sigma) and float(sigma[int(index)]) > 0
        ]
        if not valid:
            return
        measurement_abs = max(
            1e-6,
            3.0
            * float(np.sqrt(2.0))
            * max(float(sigma[index]) for index in valid),
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
        sigma_z: Sequence[float] | None = None,
    ) -> None:
        """Require every declared meter fault to remain detectable on its own.

        A truth-free policy cannot recover a small co-injected fault if fixing a
        larger peer makes the global WLS gate clean.  Such a root can terminate
        only by silently leaving truth behind, so it is not a valid sequential
        recovery scenario even when the original combined vector is anomalous.
        """
        if not self.validate or self.admission_mode == "physical" or len(faults) <= 1:
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
            statistic = self._chi2_statistic(self.case_path, probe, sigma_z=sigma_z)
            if statistic < self._cached_chi2_limit(self.case_path, probe, sigma_z=sigma_z) and not self._residual_alarm(self.case_path, probe, sigma_z=sigma_z):
                raise ScenarioRejected(
                    "sequential_fault_not_individually_detectable",
                    f"{family}: index {index} {self._detection_detail(self.case_path, probe)}",
                )

    # --------------------------------------------------------- base families

    def _no_error_scenario(self, row: Mapping[str, Any], index: int) -> dict[str, Any]:
        scenario = self._base_scenario(
            self._scenario_id("no_error", row.get("id"), index),
            case=self.case_path,
            measurements=row["z_obs"],
            family="no_error",
            sigma_z=row.get("sigma_z"),
        )
        self._require_clean(self.case_path, row["z_obs"], "no_error", sigma_z=row.get("sigma_z"))
        scenario["clean_case"] = self.case_path
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
        if any(not 0 <= i for i in error_indices) or any(i >= len(z_obs) for i in error_indices):
            raise ScenarioRejected("label_index_out_of_range", str(error_indices))
        lifted_indices: list[int] = []
        for error_index in error_indices:
            z_obs[error_index], lifted = self._floored_measurement_error(
                error_index, z_obs[error_index], z_true[error_index], sigma_z=row.get("sigma_z")
            )
            if lifted:
                lifted_indices.append(error_index)
        self._require_anomalous(self.case_path, z_obs, family, sigma_z=row.get("sigma_z"))
        scenario = self._base_scenario(
            self._scenario_id(family, row.get("id"), index),
            case=self.case_path,
            measurements=z_obs,
            family=family,
            sigma_z=row.get("sigma_z"),
        )
        scenario["clean_case"] = self.case_path
        clean_measurements = list(z_obs)
        for error_index in error_indices:
            clean_measurements[error_index] = z_true[error_index]
        # A release scenario must be solvable by fixing exactly the declared
        # bad meters while preserving every healthy observed channel. Some raw
        # corpus rows retain enough ordinary/noise outliers on undeclared
        # channels that even this truth-restored vector fails the global gate;
        # those roots are intrinsically non-terminal and are skipped rather
        # than encouraging broad healthy-channel rewrites.
        self._require_clean(self.case_path, clean_measurements, family, sigma_z=row.get("sigma_z"))
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
        if self.min_measurement_error_sigma is not None:
            sigma = row.get("sigma_z", self.noise_profile())
            scenario["measurement_error_floor"] = {
                "sigma_multiple": self.min_measurement_error_sigma,
                "lifted_indices": lifted_indices,
                "error_sigma_multiples": {
                    str(error_index): abs(z_obs[error_index] - z_true[error_index])
                    / float(sigma[error_index])
                    for error_index in error_indices
                },
            }
        self._require_sequential_measurement_observability(
            clean_measurements=clean_measurements,
            faults=scenario["true_measurement_errors"],
            family=family,
            sigma_z=row.get("sigma_z"),
        )
        self._declare_measurement_recovery_tolerance(scenario, error_indices)
        return scenario

    def _parameter_scenario(self, row: Mapping[str, Any], index: int) -> dict[str, Any]:
        label = dict(row.get("label") or {})
        line_row0 = label.get("line_row")
        if line_row0 is None:
            raise ScenarioRejected("label_missing_line_row", str(label))
        line_row0 = int(line_row0)
        if not 0 <= line_row0 < self.nl:
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
        self._require_anomalous(self.case_path, z_obs, "parameter", sigma_z=row.get("sigma_z"))
        self._require_clean(str(true_case), z_obs, "parameter", sigma_z=row.get("sigma_z"))
        true_ppc = _load_python_case(str(true_case))
        if self.system.case_id != "case14":
            base = self._clean_case()
            actual_branch = np.asarray(true_ppc["branch"], dtype=float)
            expected_branch = np.asarray(base["branch"], dtype=float).copy()
            if (np.asarray(true_ppc["bus"]).shape != np.asarray(base["bus"]).shape
                or actual_branch.shape != expected_branch.shape
                or not np.array_equal(np.asarray(true_ppc["bus"])[:, 0], np.asarray(base["bus"])[:, 0])
                or float(true_ppc["baseMVA"]) != float(base["baseMVA"])):
                raise ValueError("parameter artifact does not match the selected base system")
            expected_branch[line_row0, 2:4] = actual_branch[line_row0, 2:4]
            if not np.allclose(actual_branch, expected_branch, rtol=1e-10, atol=1e-12):
                raise ValueError("parameter artifact changed an undeclared branch or topology field")
            static_bus_columns = [0, 1, 4, 5, 6, 9, 10, 11, 12]
            actual_bus = np.asarray(true_ppc["bus"], dtype=float)
            expected_bus = np.asarray(base["bus"], dtype=float)
            actual_gen = np.asarray(true_ppc["gen"], dtype=float)
            expected_gen = np.asarray(base["gen"], dtype=float)
            static_gen_columns = [i for i in range(expected_gen.shape[1]) if i not in {1, 2, 5}]
            if (not np.allclose(actual_bus[:, static_bus_columns], expected_bus[:, static_bus_columns], rtol=1e-10, atol=1e-12)
                or actual_gen.shape != expected_gen.shape
                or not np.allclose(actual_gen[:, static_gen_columns], expected_gen[:, static_gen_columns], rtol=1e-10, atol=1e-12)):
                raise ValueError("parameter artifact changed undeclared bus or generator configuration")
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
            sigma_z=row.get("sigma_z"),
            sigma_z_scans=row.get("sigma_z_scans", row.get("sigma_z")),
        )
        scenario = self._base_scenario(
            self._scenario_id("parameter", row.get("id"), index),
            case=self.case_path,
            measurements=z_obs,
            family="parameter",
            sigma_z=row.get("sigma_z"),
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
            "sigma_z": list(row.get("sigma_z_scans", row.get("sigma_z", self.noise_profile()))),
            "initial_state_strategy": "observed_vm_plus_configured_case_angles_v1",
        }
        # The ranking the deployed parameter context produced on this root
        # is part of the root's identity for research reporting: a suite
        # drawn at the detection threshold records which roots sit inside
        # the dominance band, so results can be stratified by ambiguity.
        ranking = (
            gate_result.get("parameter_context_ranking")
            if isinstance(gate_result, Mapping)
            else None
        )
        if isinstance(ranking, Mapping):
            scenario["parameter_ranking"] = {
                **{
                    key: copy.deepcopy(ranking.get(key))
                    for key in (
                        "parameter_ranking_dominance_ratio",
                        "parameter_ranking_top_abs_lambda",
                        "parameter_ranking_runner_up_abs_lambda",
                        "parameter_ranking_singleton",
                    )
                },
                "true_line_rank": gate_result.get("parameter_context_true_line_rank"),
                "generation_threshold": self.parameter_ranking_dominance_threshold,
                "enforced": bool(self._enforce_parameter_ranking_dominance),
                "rank_allowance": self._parameter_target_rank_allowance,
            }
        if gate_result is not None:
            self._parameter_gate_results[scenario["scenario_id"]] = gate_result
        return scenario

    # ------------------------------------------------------- synthesized physics

    @staticmethod
    def _solve_ac_opf(ppc: Mapping[str, Any]) -> dict[str, Any] | None:
        """AC-OPF operating point: the solver every tabular corpus family was built on."""
        from pypower.api import ppoption, runopf

        result = runopf(copy.deepcopy(ppc), ppoption(VERBOSE=0, OUT_ALL=0))
        return result if result.get("success") else None

    @staticmethod
    def _scaled_case14(load_scale: float) -> dict[str, Any]:
        from pypower.api import case14 as pypower_case14

        ppc = pypower_case14()
        ppc["bus"][:, 2] *= load_scale
        ppc["bus"][:, 3] *= load_scale
        return ppc

    def _draw_load_scale(self) -> float:
        low, high = SYNTHESIZED_LOAD_SCALE_RANGE
        return float(self._rng.uniform(low, high))

    def _full_topology_model(self):
        if self._full_topology is None:
            from Transmission.ieee14_full_topology import build_full_topology

            self._full_topology = build_full_topology()
            self._full_topology_fingerprint = self._full_topology.fingerprint()
        return self._full_topology

    def _require_branch_dominant_wls_evidence(self, z_obs: Sequence[float], cb_name: str,
                                            *, sigma_z=None, exact_measurement_indices=None) -> None:
        """Admit a topology root only when the operator's WLS evidence routes to a branch.

        The deployed solve tags its signatures by the classical discrimination
        between the largest normalized residual and the largest normalized branch
        multiplier (``MatpowerDeploymentProviders.run_wls``).  A breaker error whose
        operator-model anomaly is weak enough for a single residual to dominate
        sends the expert down the measurement route first, where it would commit
        a false meter correction before the topology route repairs the switch.
        Such draws are rejected, exactly as parameter roots are admitted only when
        their ranking is dominant.
        """
        if not self.validate or not self.require_branch_dominant_topology:
            return
        options = {}
        if sigma_z is not None:
            options["measurement_sigma"] = list(sigma_z)
        if exact_measurement_indices:
            options["exact_measurement_indices"] = list(exact_measurement_indices)
        payload = _wls_json(self.case_path, [float(value) for value in z_obs], **options)
        if not payload.get("success"):
            raise ScenarioRejected("wls_failure", str(payload.get("error")))
        max_residual = max((abs(float(v)) for v in payload.get("r") or []), default=0.0)
        max_multiplier = max((abs(float(v)) for v in payload.get("lambdaN") or []), default=0.0)
        if not max_multiplier > 1.2 * max_residual:
            raise ScenarioRejected(
                "topology_root_not_branch_dominant",
                f"{cb_name}: max |lambda| {max_multiplier:.1f} <= 1.2 x max |r| {max_residual:.1f}",
            )

    def _breaker_error_catalogue(self) -> list[dict[str, Any]]:
        """Single breaker-status errors of the sampled classes."""
        if self._breaker_errors is None:
            from Transmission.ieee14_full_measurements import single_flip_catalogue

            self._breaker_errors = [
                {**entry, "effect": _breaker_effect_class(entry["category"])}
                for entry in single_flip_catalogue(self._full_topology_model())
                if _breaker_effect_class(entry["category"]) in self.topology_effects
            ]
            if not self._breaker_errors:
                raise RuntimeError(
                    "the full node/breaker model offers no switch errors of "
                    f"{self.topology_effects}"
                )
        return self._breaker_errors

    def _topology_scenario(
        self, index: int, effects: Sequence[str] | None = None
    ) -> dict[str, Any]:
        """One breaker-status error in the full IEEE-14 node/breaker model.

        The physical truth is a switch whose real state differs from the reported
        schematic-normal state.  Three classes are sampled (``topology_effects``):
        a switch that isolates exactly one line terminal, which the operator's
        topology processor renders as that line out of service; a switch that
        splits a bus into two energized sections, rendered as one more bus; and a
        switch that joins two buses (the shared 10/14 yard), rendered as one bus
        fewer.  The fix is a breaker-level ``correct_topology`` naming the switch.
        Islanded bays stay out: they are equipment outages the operator's WLS
        cannot see.  A merge is marginal for that WLS at light load, so the
        anomaly gate below rejects the draws it does not flag.

        Telemetry is one physical solution: an AC OPF on the true contracted
        topology at a corpus-range load scale fixes the dispatch, and a power flow
        of the 65-node network with breakers as tiny impedances supplies both the
        substation measurements (node voltages, unit and load meters, terminal and
        breaker flows) and, through the fixed meter identity, the operator's
        vector.  The scenario is admitted only if the operator's bus-branch WLS
        flags it, the operator case rendered from the true statuses verifies
        clean, and the node/breaker estimator on the reported statuses flags it,
        is clean on the true statuses, and ranks the true breaker first.
        """
        from Transmission.ieee14_full_gse import gse_topology_nlm
        from Transmission.ieee14_full_measurements import (
            MODEL_ID,
            flipped_case,
            main_section_nodes,
        )
        from Transmission.ieee14_full_substation import (
            add_telemetry_noise,
            operator_model_from_map,
            operator_noise_for_layout,
            operator_vector_for_layout,
            solve_node_breaker,
            status_labels,
            substation_telemetry,
        )

        model = self._full_topology_model()
        allowed = set(effects or self.topology_effects)
        errors = [e for e in self._breaker_error_catalogue() if e["effect"] in allowed]
        if not errors:
            raise ScenarioRejected("topology_effects_unavailable", str(sorted(allowed)))
        error = errors[int(self._rng.integers(len(errors)))]
        cb_name = str(error["cb_name"])
        true_closed = bool(error["true_closed"])
        category = str(error["effect"])
        load_scale = self._draw_load_scale()
        reference = self._scaled_case14(load_scale)
        truth_case, info, removed = flipped_case(
            reference, {cb_name: true_closed}, model=model
        )
        if removed["dead_buses"]:
            raise ScenarioRejected(
                "topology_flip_islands_equipment", f"{cb_name}: {removed}"
            )
        dispatch = self._solve_ac_opf(truth_case)
        if dispatch is None:
            raise ScenarioRejected(
                "opf_diverged", f"{cb_name} load_scale={load_scale:.3f}"
            )
        node_to_bus = info["node_to_bus"]
        physical, physical_info, physical_removed = solve_node_breaker(
            model, reference, {cb_name: true_closed}, dispatch, node_to_bus
        )
        if physical is None:
            raise ScenarioRejected(
                "node_breaker_power_flow_diverged",
                f"{cb_name} load_scale={load_scale:.3f}",
            )
        if physical_removed["dead_buses"]:
            raise ScenarioRejected(
                "topology_flip_islands_equipment", f"{cb_name}: {physical_removed}"
            )
        telemetry = substation_telemetry(
            physical, model, reference, physical_info, physical_removed
        )
        telemetry = add_telemetry_noise(
            telemetry, self._rng, scale=self.topology_noise_scale
        )
        telemetry = _canonicalize_telemetry(telemetry)
        meter_nodes = main_section_nodes(model, node_to_bus, [])
        # The operator's layout: the schematic-normal map rendered by the same
        # topology processor the correction executor uses.
        clean_case14 = self._clean_case()
        _, normal_layout = operator_model_from_map(clean_case14, model, {}, meter_nodes)
        z_obs = _canonicalize_synthesized_measurement_vector(
            operator_vector_for_layout(telemetry, normal_layout).tolist()
        )
        operator_noise = operator_noise_for_layout(telemetry, normal_layout)

        def noise_options(contract):
            covariance = np.asarray(contract["measurement_covariance"], dtype=float)
            if not np.array_equal(covariance, np.diag(np.diag(covariance))):
                raise ScenarioRejected("correlated_operator_noise_unsupported", cb_name)
            return {"sigma_z": contract["measurement_sigma"],
                    "exact_measurement_indices": contract["structural_zero_indices"]}

        source_noise_options = noise_options(operator_noise)

        if category == "dangling_line_terminal":
            row0 = int(error["equivalent_branch_row0"])
            corrected_ppc = copy.deepcopy(clean_case14)
            corrected_ppc["branch"][row0][10] = 0.0
            corrected_case = self._derived_case(corrected_ppc, f"r0_topo_l{row0 + 1}s0")
            clean_measurements = list(z_obs)
            corrected_operator_noise = operator_noise
        else:
            corrected_ppc, corrected_layout = operator_model_from_map(
                clean_case14, model, {cb_name: true_closed}, meter_nodes
            )
            corrected_case = self._derived_case(
                corrected_ppc, f"r0_topo_{cb_name.lower()}_s{int(true_closed)}"
            )
            clean_measurements = _canonicalize_synthesized_measurement_vector(
                operator_vector_for_layout(telemetry, corrected_layout).tolist()
            )
            corrected_operator_noise = operator_noise_for_layout(telemetry, corrected_layout)
        self._require_anomalous(self.case_path, z_obs, "topology", **source_noise_options)
        self._require_clean(corrected_case, clean_measurements, "topology", **noise_options(corrected_operator_noise))
        self._require_branch_dominant_wls_evidence(z_obs, cb_name, **source_noise_options)

        reported_labels = status_labels(model)
        reported_estimate = gse_topology_nlm(model, reference, {}, telemetry)
        if not reported_estimate["success"]:
            raise ScenarioRejected("node_breaker_estimate_diverged", cb_name)
        gse_limit = float(
            chi2_threshold(max(1, int(reported_estimate["dof"])), self.chi2_alpha)
        )
        if reported_estimate["chi_square"] <= self.anomaly_margin * gse_limit:
            raise ScenarioRejected(
                "breaker_error_not_detectable_at_substation",
                f"{cb_name}: chi2 {reported_estimate['chi_square']:.1f} <= "
                f"{self.anomaly_margin:.2f} x {gse_limit:.1f}",
            )
        true_estimate = gse_topology_nlm(
            model, reference, {cb_name: true_closed}, telemetry
        )
        true_limit = (
            float(chi2_threshold(max(1, int(true_estimate["dof"])), self.chi2_alpha))
            if true_estimate["success"]
            else 0.0
        )
        if not true_estimate["success"] or true_estimate["chi_square"] >= true_limit:
            raise ScenarioRejected(
                "true_breaker_status_not_clean_at_substation",
                f"{cb_name}: chi2 {true_estimate['chi_square']:.1f} >= {true_limit:.1f}",
            )
        order = [item["cb_name"] for item in reported_estimate["ranking"]]
        true_rank = order.index(cb_name) + 1
        if self.enforce_topology_ranking and true_rank != 1:
            raise ScenarioRejected(
                "breaker_not_top_ranked",
                f"{cb_name} ranked {true_rank} behind {order[0]}",
            )

        branch = clean_case14["branch"]
        scenario = self._base_scenario(
            self._scenario_id("topology", cb_name, load_scale, index),
            case=self.case_path,
            measurements=z_obs,
            family="topology",
            sigma_z=operator_noise["measurement_sigma"],
        )
        scenario["clean_case"] = corrected_case
        scenario["clean_measurements"] = list(clean_measurements)
        scenario["metadata"]["substation_telemetry"] = telemetry
        scenario["metadata"]["reported_breaker_status"] = reported_labels
        scenario["metadata"]["operator_voltage_meter_nodes"] = {
            str(bus): node for bus, node in meter_nodes.items()
        }
        scenario["metadata"]["operator_layout"] = normal_layout
        scenario["metadata"]["operator_noise"] = operator_noise
        scenario["metadata"]["structural_zero_indices"] = operator_noise["structural_zero_indices"]
        scenario["metadata"]["topology_model_id"] = MODEL_ID
        scenario["metadata"]["topology_model_fingerprint"] = self._full_topology_fingerprint
        # Ranking statistics only: the breaker names stay in the hidden truth.
        scenario["topology_ranking"] = {
            "true_breaker_rank": true_rank,
            "top_score": float(reported_estimate["ranking"][0]["score"]),
            "runner_up_score": (
                float(reported_estimate["ranking"][1]["score"]) if len(order) > 1 else None
            ),
            "gse_chi_square": float(reported_estimate["chi_square"]),
            "gse_threshold": gse_limit,
            "enforced": bool(self.enforce_topology_ranking),
        }
        truth = {
            "expected_status": int(true_closed),
            "load_scale": load_scale,
            "operating_point": "ac_opf",
            "cb_name": cb_name,
            "cb_yard": str(error["yard"]),
            "reported_cb_closed": bool(error["reported_closed"]),
            "true_cb_closed": true_closed,
            "physical_effect": category,
            "topology_model_id": MODEL_ID,
            "topology_model_fingerprint": self._full_topology_fingerprint,
        }
        if category == "dangling_line_terminal":
            truth.update(
                {
                    "branch_row0": row0,
                    "line_index1": row0 + 1,
                    "from_bus": int(branch[row0][0]),
                    "to_bus": int(branch[row0][1]),
                }
            )
        else:
            truth["affected_planning_buses"] = [
                int(bus) for bus in (error.get("affected_planning_buses") or [])
            ]
            truth["operator_bus_count_after_fix"] = int(corrected_layout["bus_count"])
        scenario["true_topology_errors"] = [truth]
        return scenario

    def _synthesized_harmonic_row(self, index: int) -> dict[str, Any]:
        """A corpus-shaped harmonic snapshot on an AC-OPF operating point.

        The tracked harmonic rows were all synthesized at unit load on the stored
        case14 planning voltages, which made load level and dispatch a family tell.
        This draws the load scale from the corpus range, solves the fundamental
        with the same OPF the corpus families used, and runs the legacy harmonic
        synthesis (source buses, THD range, spectrum, transducer, noise) on that
        operating point.
        """
        from Transmission.generate_hse_traces import build_trace

        load_scale = self._draw_load_scale()
        solution = self._solve_ac_opf(self._scaled_case14(load_scale))
        if solution is None:
            raise ScenarioRejected("opf_diverged", f"harmonic load_scale={load_scale:.3f}")
        source_bus = int(self._rng.choice(HARMONIC_SOURCE_CANDIDATES))
        thd_target = float(self._rng.uniform(*HARMONIC_THD_RANGE))
        trace = build_trace(
            source_bus,
            thd_target,
            int(self._rng.integers(2**31 - 1)),
            bus=solution["bus"],
            branch=solution["branch"],
        )
        harmonic_measurements: list[dict[str, Any]] = []
        orders: set[int] = set()
        for order_text, phasors in trace["harmonic_phasors"].items():
            order = int(order_text)
            orders.add(order)
            for item in phasors:
                real, imag = item["V_complex_noisy"]
                harmonic_measurements.append(
                    {
                        "bus": int(item["bus_1based"]),
                        "h": order,
                        "V_real": float(real),
                        "V_imag": float(imag),
                        "sigma": float(item["sigma"]),
                        "sigma_semantics": item.get("sigma_semantics", "per_component"),
                    }
                )
        return {
            "id": f"synthesized_harmonic_{index}",
            "scenario": "harmonic_anomaly",
            "z_true": _canonicalize_synthesized_measurement_vector(trace["z_scada_true"]),
            "z_obs": _canonicalize_synthesized_measurement_vector(trace["z_scada_meas"]),
            "sigma_z": list(trace["sigma_z"]),
            "noise_contract": copy.deepcopy(trace["noise_contract"]),
            "harmonic_measurements": harmonic_measurements,
            "harmonic_orders": sorted(orders),
            "label": {
                "error_type": "harmonic_anomaly",
                "source_bus": source_bus,
                "thd_target": thd_target,
                "actual_thd": float(trace["actual_thd"]),
                "load_scale": load_scale,
                "operating_point": "ac_opf",
            },
            "op_point": {"load_scale": load_scale},
        }

    def _harmonic_scenario(self, row: Mapping[str, Any], index: int) -> dict[str, Any]:
        label = dict(row.get("label") or {})
        harmonic_measurements = row.get("harmonic_measurements")
        # Profiles that can request spectra after a WLS alarm need the
        # spectrum the ground truth generated; scada_only strips it anyway.
        if not harmonic_measurements and allows_diagnostic_tools(self.evidence_profile):
            raise ScenarioRejected("harmonic_measurements_missing", str(row.get("id")))
        harmonic_measurements = copy.deepcopy(harmonic_measurements or [])
        for item in harmonic_measurements:
            # Legacy generated harmonic rows stored complex RMS sigma. Fresh
            # rows name rectangular component sigma explicitly.
            if "sigma_semantics" not in item:
                item["sigma_complex_rms"] = float(item["sigma"])
                item["sigma"] = float(item["sigma"]) / math.sqrt(2.0)
                item["sigma_semantics"] = "per_component"
                item["noise_alignment"] = "legacy_generated_complex_rms_to_component"
        z_obs = [float(value) for value in row["z_obs"]]
        mode = self.waveform_signature_mode["harmonic"]
        # These synthetic corpus rows also perturb the positive-sequence
        # snapshot.  Discovery requires a WLS anomaly to justify requesting
        # additional measurements; that anomaly alone is not harmonic evidence.
        # The legacy monitor-flagged mode only requires WLS solvability.
        if mode == "discovered":
            self._require_anomalous(self.case_path, z_obs, "harmonic", sigma_z=row.get("sigma_z"))
        elif self.validate:
            self._chi2_statistic(self.case_path, z_obs, sigma_z=row.get("sigma_z"))
        scenario = self._base_scenario(
            self._scenario_id("harmonic", row.get("id"), index),
            case=self.case_path,
            measurements=z_obs,
            family="harmonic",
            sigma_z=row.get("sigma_z"),
        )
        scenario["clean_case"] = self.case_path
        scenario["clean_measurements"] = [float(value) for value in row["z_true"]]
        if mode == "flagged":
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
        harmonic_truth: dict[str, Any] = {
            "bus_1based": label.get("source_bus"),
            "thd_target": label.get("thd_target"),
        }
        for key in ("actual_thd", "load_scale", "operating_point"):
            if label.get(key) is not None:
                harmonic_truth[key] = label[key]
        scenario["hidden_truth"] = {"true_harmonic_errors": [harmonic_truth]}
        scenario["release_audit"] = {
            **copy.deepcopy(_EXPLANATION_ONLY_RELEASE_AUDIT),
            "signature_mode": mode,
            "sensor_signatures_withheld": (
                [HARMONIC_SIGNATURE] if mode == "discovered" else []
            ),
        }
        return scenario

    def _hif_scenario(self, row: Mapping[str, Any], index: int) -> dict[str, Any]:
        row = self._aligned_waveform_row(row, "hif")
        label = dict(row.get("label") or {})
        diagnostic = row.get("nlm_diagnostic")
        scans = row.get("scans")
        # Only the historical auxiliary profile hands the corpus's cached NLM
        # diagnosis to the runtime; strict profiles recompute from the
        # phasors, so admission does not depend on it.
        if self.evidence_profile == AUXILIARY_EVIDENCE_PROFILE and (not isinstance(diagnostic, Mapping) or not diagnostic.get("success")):
            raise ScenarioRejected("nlm_diagnostic_missing", str(row.get("id")))
        if allows_diagnostic_tools(self.evidence_profile) and not scans:
            raise ScenarioRejected("hif_scans_missing", str(row.get("id")))
        diagnostic = diagnostic if isinstance(diagnostic, Mapping) else {}
        scans = list(scans or [])
        z_obs = [float(value) for value in row["z_obs"]]
        if self.validate:
            self._chi2_statistic(self.case_path, z_obs, sigma_z=row["sigma_z"])  # must solve; may be subtle
        if len(scans) > self.hif_max_scans:
            picks = np.linspace(0, len(scans) - 1, self.hif_max_scans).round().astype(int)
            scans = [scans[int(i)] for i in dict.fromkeys(picks.tolist())]
        scenario = self._base_scenario(
            self._scenario_id("hif", row.get("id"), index),
            case=self.case_path,
            measurements=z_obs,
            family="hif",
            sigma_z=row["sigma_z"],
        )
        scenario["clean_case"] = self.case_path
        scenario["clean_measurements"] = [float(value) for value in row["z_true"]]
        hif_mode = self.waveform_signature_mode["hif"]
        if hif_mode == "discovered":
            self._require_anomalous(self.case_path, z_obs, "hif", sigma_z=row["sigma_z"])
        else:
            scenario["unresolved_signatures"] = [HIF_SIGNATURE]
            scenario["semantic_field_provenance"]["unresolved_signatures"] = (
                _WAVEFORM_PROVENANCE
            )
        if is_scada_only(self.evidence_profile):
            if row.get("noise_contract") is not None:
                scenario["metadata"]["noise_contract"] = copy.deepcopy(row["noise_contract"])
            if row.get("measurement_convention") is not None:
                scenario["metadata"]["measurement_convention"] = copy.deepcopy(row["measurement_convention"])
            scenario["hidden_truth"] = {"true_hif_errors": [copy.deepcopy(label)]}
            scenario["release_audit"] = {**copy.deepcopy(_EXPLANATION_ONLY_RELEASE_AUDIT),
                "signature_mode": hif_mode, "sensor_signatures_withheld": [HIF_SIGNATURE]}
            return scenario
        strict = is_strict_boundary(self.evidence_profile)
        if not strict:
            # Historical reproduction only: the corpus's cached diagnosis.
            scenario["metadata"]["nlm_diagnostic"] = {
                key: copy.deepcopy(diagnostic[key])
                for key in ("success", "converged", "method", "backend", "top_hif_groups") if key in diagnostic
            }
        else:
            # wls_gated_diagnostics: the three-phase PMU phasors the ground
            # truth generated for this window, at the corpus's declared
            # per-component sigma, as a top-level channel like the unbalance
            # roots carry.  No precomputed diagnosis and no clean copies.
            voltages = row.get("three_phase_voltages")
            if not voltages and scans:
                voltages = scans[0].get("three_phase_voltages")
            if not isinstance(voltages, Sequence) or not voltages:
                raise ScenarioRejected("three_phase_voltages_missing", str(row.get("id")))
            scenario["metadata"]["three_phase_voltages"] = copy.deepcopy(list(voltages))
        scenario["metadata"]["three_phase_sigma"] = float(row["three_phase_sigma"])
        if row.get("noise_contract") is not None:
            scenario["metadata"]["noise_contract"] = copy.deepcopy(row["noise_contract"])
        scenario["metadata"]["hif_runtime"] = {
            "z_obs": z_obs,
            "scan_index": row.get("scan_index", scans[0].get("scan_index")),
            "op_point": copy.deepcopy(row.get("op_point") or scans[0].get("op_point")),
            "sigma_z": copy.deepcopy(row["sigma_z"]),
            "three_phase_sigma": float(row["three_phase_sigma"]),
            "three_phase_voltages": copy.deepcopy(row.get("three_phase_voltages")),
            "load_scale": float((row.get("op_point") or {}).get("load_scale", 1.0)),
        }
        runtime_scans = [self._observable_waveform_scan(scan) for scan in scans]
        window_meta = row.get("window_metadata") or {}
        scenario["metadata"]["hif_scan_window"] = {
            # Strict profiles key the window by the opaque root id: the corpus
            # row id names the family, and nothing consumes it as a path when
            # the scans are supplied inline.
            "scan_window_path": scenario["scenario_id"] if strict else str(row.get("id") or scenario["scenario_id"]),
            "scans": copy.deepcopy(runtime_scans),
            "sigma_z": copy.deepcopy(row["sigma_z"]),
            "three_phase_sigma": float(row["three_phase_sigma"]),
        }
        if not strict:
            # Simulator provenance of the window (source kind, operating-point
            # mode) is historical-reproduction metadata; no provider reads it.
            scenario["metadata"]["hif_scan_window"]["window_metadata"] = {
                key: copy.deepcopy(window_meta[key]) for key in ("source_kind", "operating_point_mode")
                if key in window_meta
            }
        if row.get("measurement_convention") is not None:
            declaration = copy.deepcopy(row["measurement_convention"])
            scenario["metadata"]["measurement_convention"] = declaration
            scenario["metadata"]["hif_runtime"]["measurement_convention"] = copy.deepcopy(declaration)
            scenario["metadata"]["hif_scan_window"]["measurement_convention"] = copy.deepcopy(declaration)
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
        scenario["release_audit"] = {
            **copy.deepcopy(_EXPLANATION_ONLY_RELEASE_AUDIT),
            "signature_mode": hif_mode,
            "sensor_signatures_withheld": [HIF_SIGNATURE] if hif_mode == "discovered" else [],
        }
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
        row = self._aligned_waveform_row(row, "three_phase_unbalance")
        label = copy.deepcopy(dict(row.get("label") or {}))
        voltages = row.get("three_phase_voltages")
        # A profile whose diagnostics can be requested after the WLS alarm
        # needs the phasors and an unbalance the telemetry actually shows;
        # scada_only never sees them, so admission there is WLS-only.
        diagnostics_available = allows_diagnostic_tools(self.evidence_profile)
        if diagnostics_available and (not isinstance(voltages, Sequence) or not voltages):
            raise ScenarioRejected("three_phase_voltages_missing", str(row.get("id")))
        signatures = self._observable_unbalance_signatures(row) if diagnostics_available else []
        if diagnostics_available and not signatures:
            raise ScenarioRejected("unbalance_not_observable", str(row.get("id")))
        z_obs = [float(value) for value in row["z_obs"]]
        mode = self.waveform_signature_mode["three_phase_unbalance"]
        if mode == "discovered":
            # The operator starts from the positive-sequence snapshot alone,
            # so the unbalance must at least register as a WLS anomaly.
            self._require_anomalous(self.case_path, z_obs, "three_phase_unbalance", sigma_z=row["sigma_z"])
        elif self.validate:
            self._chi2_statistic(self.case_path, z_obs, sigma_z=row["sigma_z"])
        scenario = self._base_scenario(
            self._scenario_id("three_phase_unbalance", row.get("id"), index),
            case=self.case_path,
            measurements=z_obs,
            family="three_phase_unbalance",
            sigma_z=row["sigma_z"],
        )
        scenario["clean_case"] = self.case_path
        scenario["clean_measurements"] = [float(value) for value in row["z_true"]]
        if mode == "flagged":
            scenario["unresolved_signatures"] = signatures
            scenario["semantic_field_provenance"]["unresolved_signatures"] = (
                _WAVEFORM_PROVENANCE
            )
        if diagnostics_available:
            scenario["metadata"]["three_phase_voltages"] = copy.deepcopy(list(voltages))
            scenario["metadata"]["three_phase_sigma"] = float(row["three_phase_sigma"])
        if row.get("noise_contract") is not None:
            scenario["metadata"]["noise_contract"] = copy.deepcopy(row["noise_contract"])
        branch_currents = row.get(BRANCH_CURRENT_CHANNEL)
        if branch_currents and branch_current_rows_to_phasors(branch_currents):
            scenario["metadata"][BRANCH_CURRENT_CHANNEL] = copy.deepcopy(list(branch_currents))
            if row.get(BRANCH_CURRENT_SIGMA_KEY) is not None:
                scenario["metadata"][BRANCH_CURRENT_SIGMA_KEY] = float(
                    row[BRANCH_CURRENT_SIGMA_KEY]
                )
        scenario["hidden_truth"] = {"true_unbalance_errors": [label]}
        scenario["release_audit"] = {
            **copy.deepcopy(_EXPLANATION_ONLY_RELEASE_AUDIT),
            "signature_mode": mode,
            "sensor_signatures_withheld": list(signatures) if mode == "discovered" else [],
        }
        return scenario

    def _telemetry_no_disturbance_scenario(
        self, row: Mapping[str, Any], index: int
    ) -> dict[str, Any]:
        from three_phase_nlm.hif_operating_point import canonicalize_ieee14_operating_point
        from three_phase_nlm.hif_parameter_estimator import _resolve_model_dir, _simulate_base
        from three_phase_nlm.measurement_noise import (
            add_scada_noise, add_voltage_phasor_noise, generated_noise_contract,
        )
        from three_phase_nlm.branch_current_analysis import add_branch_current_noise

        row = self._aligned_waveform_row(row, "hif" if row.get("scans") else "three_phase_unbalance")
        sigma_z = row["sigma_z"]
        voltage_sigma = float(row["three_phase_sigma"])
        current_sigma = float(row[BRANCH_CURRENT_SIGMA_KEY])
        op_point = canonicalize_ieee14_operating_point(row.get("op_point") or {})
        try:
            balanced_model = _simulate_base(_resolve_model_dir(None, self.case_path), op_point=op_point)
        except Exception as exc:
            raise ScenarioRejected("balanced_telemetry_replay_failed", str(exc)) from exc
        # SCADA retains the canonical balanced reference used by this control;
        # auxiliary means come from a fresh balanced OpenDSS solve. Never project
        # already noisy currents/voltages and then declare the original variance.
        clean_measurements = [float(value) for value in row["z_true"]]
        measurements = add_scada_noise(clean_measurements, self._rng, sigma_z)
        balanced = add_voltage_phasor_noise(balanced_model["three_phase_voltages"], self._rng, voltage_sigma)
        balanced_currents = add_branch_current_noise(balanced_model[BRANCH_CURRENT_CHANNEL], self._rng, current_sigma)
        self._require_clean(self.case_path, measurements, "telemetry_no_disturbance", sigma_z=sigma_z)
        scenario = self._base_scenario(
            self._scenario_id("telemetry_no_disturbance", row.get("id"), index),
            case=self.case_path,
            measurements=measurements,
            family="telemetry_no_disturbance",
            sigma_z=sigma_z,
        )
        scenario["clean_case"] = self.case_path
        scenario["clean_measurements"] = clean_measurements
        scenario["metadata"]["three_phase_voltages"] = balanced
        scenario["metadata"]["three_phase_sigma"] = voltage_sigma
        scenario["metadata"][BRANCH_CURRENT_CHANNEL] = balanced_currents
        scenario["metadata"][BRANCH_CURRENT_SIGMA_KEY] = current_sigma
        scenario["metadata"]["noise_contract"] = generated_noise_contract(
            sigma_z, noise_scale=float((row.get("noise_contract") or {}).get("noise_scale", 1.0)),
            three_phase_sigma=voltage_sigma, branch_current_sigma_pu=current_sigma,
        )
        scenario["metadata"]["telemetry_control_semantics"] = {
            "scada_mean": "canonical_balanced_PYPOWER_reference",
            "auxiliary_means": "fresh_balanced_OpenDSS_solve_at_source_operating_point",
            "noise": "independent_Gaussian_draws_added_once_to_each_deterministic_sensor_mean",
            "cross_simulator_operating_point_identity": "not_asserted",
        }
        scenario["hidden_truth"] = {
            "true_unbalance_errors": [],
            "control_kind": "telemetry_present_no_disturbance",
        }
        return scenario

    # ---------------------------------------------------------- compositions

    def _overlay_indices(self, scenario: Mapping[str, Any], count: int) -> list[int]:
        """Indices eligible for a gross-offset overlay on this scenario."""
        index_map = measurement_index_map(self.nb, self.nl)
        blocked: set[int] = set()
        blocked.update(int(i) for i in (scenario.get("metadata") or {}).get("structural_zero_indices", []))
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
        if not self.validate or self.admission_mode == "physical":
            return None
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

        current_case = str(scenario.get("case") or self.case_path)
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
                supported_lines = [
                    int(dict(item).get("arguments", {}).get("line_index", -1))
                    for item in supported
                ]
                observed_line = supported_lines[0]
                stage_metrics["rank_one_line_index1"] = observed_line
                true_line_rank = (
                    supported_lines.index(expected_line) + 1
                    if expected_line in supported_lines
                    else None
                )
                stage_metrics["true_line_rank"] = true_line_rank
                allowance = self._parameter_target_rank_allowance
                admitted_by_rank = (
                    true_line_rank is not None and true_line_rank <= allowance
                    if allowance is not None
                    else observed_line == expected_line
                )
                if not admitted_by_rank:
                    metrics["stages"].append(stage_metrics)
                    reject(
                        "mixed_parameter_recovery_target_mismatch",
                        (
                            "rank-one parameter correction does not match the "
                            "declared mixed-error target"
                        ),
                        failed_stage="parameter_context",
                    )
                # The offline route is validated on the true line's correction;
                # a rank allowance only decides admission.
                action = copy.deepcopy(dict(supported[true_line_rank - 1]))
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
        return metrics

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
            # A breaker-level topology correction names the switch beside the
            # bus-branch row it affects; both spellings are production schema.
            if (
                expected_tool == CORRECT_TOPOLOGY
                and observed_argument_keys == expected_argument_keys | {"cb_name"}
            ):
                expected_argument_keys = observed_argument_keys
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
            not 0 <= expected_branch_row0 < self.nl
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
        metadata = composed.get("metadata") or {}
        sigma_z = list(metadata.get("sigma_z", self.noise_profile()))
        errors = list(composed.get("true_measurement_errors") or [])
        for overlay_index in self._overlay_indices(composed, offsets):
            magnitude = float(self._rng.uniform(0.10, 0.30))
            if self.min_measurement_error_sigma is not None:
                floor = self.min_measurement_error_sigma * float(
                    sigma_z[overlay_index]
                )
                magnitude = max(magnitude, floor * float(self._rng.uniform(1.0, 1.5)))
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
        # The active snapshot and its diagnostic copy are one acquisition.
        # Historical scans stay independent; only the matching current scan
        # receives the same corruption, preventing a hidden uncorrupted copy.
        runtime = metadata.get("hif_runtime")
        if isinstance(runtime, dict):
            runtime["z_obs"] = list(measurements)
            window = metadata.get("hif_scan_window") or {}
            for scan in window.get("scans", []):
                if scan.get("z_obs") == scenario["measurements"]:
                    scan["z_obs"] = list(measurements)
        composed["true_measurement_errors"] = errors
        if family == "measurement+hif":
            audit = composed.setdefault("release_audit", {})
            audit.get("not_applicable", {}).pop("final_measurements_match_clean", None)
            audit["measurement_reference_kind"] = "hif_present_meter_recovery_acquisition_v1"
        self._declare_measurement_recovery_tolerance(
            composed,
            [int(item["index"]) for item in errors if item.get("index") is not None],
        )
        noise_options = {"sigma_z": sigma_z, "exact_measurement_indices": metadata.get("structural_zero_indices", [])}
        self._require_anomalous(self.case_path, measurements, family, **noise_options)
        corrected_case = str(composed.get("clean_case") or self.case_path)
        if corrected_case != self.case_path:
            # In branch+measurement recovery the branch family may correctly
            # resolve first. The independent bad meter must still be observable
            # under that repaired model; otherwise exact sequential recovery is
            # order-dependent and a truth-free policy can finalize too early.
            self._require_anomalous(corrected_case, measurements, family, **noise_options)
        if family == "measurement+topology":
            self._require_mixed_topology_recovery_realizable(composed)
        if family == "measurement+parameter":
            mixed_metrics = self._require_mixed_parameter_recovery_realizable(
                composed,
                base_scenario_id=base_scenario_id,
            )
            # The deployed ranking on the composed root (with the bad meter in
            # place) is what a diagnosis faces, so it replaces the base root's.
            parameter_stage = next(
                (
                    stage
                    for stage in (mixed_metrics or {}).get("stages", [])
                    if isinstance(stage, Mapping) and "parameter_ranking" in stage
                ),
                None,
            )
            if parameter_stage is not None:
                ranking = parameter_stage["parameter_ranking"]
                composed["parameter_ranking"] = {
                    **{
                        key: copy.deepcopy(ranking.get(key))
                        for key in (
                            "parameter_ranking_dominance_ratio",
                            "parameter_ranking_top_abs_lambda",
                            "parameter_ranking_runner_up_abs_lambda",
                            "parameter_ranking_singleton",
                        )
                    },
                    "true_line_rank": parameter_stage.get("true_line_rank"),
                    "generation_threshold": self.parameter_ranking_dominance_threshold,
                    "enforced": bool(self._enforce_parameter_ranking_dominance),
                    "rank_allowance": self._parameter_target_rank_allowance,
                }
        return composed

    # ---------------------------------------------------------------- driver

    def build(self, plan: Mapping[str, int]) -> list[dict[str, Any]]:
        """Build scenarios per the family->count plan, skipping invalid rows."""
        unknown = sorted(set(plan) - set(SCENARIO_FAMILIES))
        if unknown:
            raise ValueError(f"Unknown scenario families: {unknown}")
        supported = set(self.system.supported_families)
        if self.admission_mode == "physical":
            supported &= {"no_error", "measurement", "multi_measurement", "parameter", "measurement+parameter"}
        unsupported = sorted(family for family, count in plan.items() if int(count) > 0 and family not in supported)
        if unsupported:
            raise ValueError(f"Unsupported families for {self.system.case_id}/{self.admission_mode}: {unsupported}")
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
            if family == "harmonic" or family == "topology" or "topology" in family:
                source_tier = "physics_synthesized"
            elif family == "telemetry_no_disturbance":
                source_tier = "derived_negative_control"
            elif family in {"hif", "three_phase_unbalance", "measurement+hif"}:
                source_tier = "tracked_diagnostic_corpus"
            elif "+" in family or family == "multi_measurement":
                source_tier = "tracked_composed_corpus"
            else:
                source_tier = "tracked_measurement_corpus"
            scenario["network_case"] = str(
                scenario.get("network_case") or self.case_path
            )
            scenario["error_cardinality"] = int(error_cardinality)
            scenario["source_tier"] = source_tier
            if scenario.get("source_realization_id"):
                scenario["source_tier"] = "physics_synthesized_balanced"
            if is_strict_boundary(self.evidence_profile):
                # scada_only keeps balanced SCADA and its declarations only;
                # wls_gated_diagnostics also keeps the auxiliary streams the
                # ground truth generated (gated at runtime behind the WLS
                # alarm) but no seeded signature, hint, label or precomputed
                # diagnosis.  Keep generator-only truth/grouping fields for
                # the later execution/audit partition; replace only runtime
                # fields.
                strict = sanitize_execution_for_profile(scenario, self.evidence_profile)
                for key in ("metadata", "semantic_field_provenance", "unresolved_signatures",
                            "remaining_anomaly_score", "no_material_anomaly_remaining", "requires_measurement_context"):
                    if key in strict:
                        scenario[key] = strict[key]
                    else:
                        scenario.pop(key, None)
                for key in ("family_hint", "correction_hint", "expected_actions", "hint"):
                    scenario.pop(key, None)
            self.manifest.append(
                {
                    "scenario_id": scenario["scenario_id"],
                    "scenario_family": family,
                    "source": source,
                    "network_case": scenario["network_case"],
                    "error_cardinality": scenario["error_cardinality"],
                    "source_tier": scenario["source_tier"],
                    "evidence_profile": self.evidence_profile,
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
                    base = self._topology_scenario(
                        1000 + attempts, effects=("dangling_line_terminal",)
                    )
                    record(
                        self._compose_measurement(
                            base, offsets=1, family=family, index=attempts
                        ),
                        "synthesized_pypower",
                    )
                except ScenarioRejected as rejection:
                    self._record_skip(family, f"synthesized_{attempts}", rejection)
            return built

        if family == "harmonic":
            while len(built) < count and attempts < count * 8:
                attempts += 1
                try:
                    record(
                        self._harmonic_scenario(
                            self._synthesized_harmonic_row(attempts), attempts
                        ),
                        "synthesized_pypower_hse",
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
                scenario = builder(row, attempts)
                if row.get("source_realization_id"):
                    scenario["source_realization_id"] = row["source_realization_id"]
                    scenario["base_case_version"] = self.system.base_case_hash
                    scenario["scenario_admission_mode"] = self.admission_mode
                record(scenario, source)
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
            "system": self.system.to_manifest(),
            "admission_mode": self.admission_mode,
            "evidence_profile": self.evidence_profile,
            "waveform_signature_mode": dict(self.waveform_signature_mode),
            "chi2_alpha": self.chi2_alpha,
            "chi2_limit": self.chi2_limit,
            "normalized_residual_threshold": self.normalized_residual_threshold,
            "anomaly_detection_rule": (
                "chi_square_or_normalized_residual"
                if self.normalized_residual_threshold is not None
                else "chi_square_only"
            ),
            "source_partition": copy.deepcopy(self._source_partition_metadata),
            "parameter_ranking_admission": {
                "contract": PARAMETER_RANKING_CONTRACT,
                "enforced": self._enforce_parameter_ranking_dominance and self.validate and self.admission_mode == "recoverable",
                "threshold": self.parameter_ranking_dominance_threshold,
            },
            "measurement_error_floor_sigma": self.min_measurement_error_sigma,
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
    "PHYSICAL_HIF_SAMPLE_PATHS",
    "PHYSICAL_HIF_SAMPLE_PATHS_20260923B",
    "PHYSICAL_HIF_SAMPLE_PATHS_20260921",
    "PHYSICAL_HIF_CORPUS_TAG",
    "PHYSICAL_IMBALANCE_SAMPLE_PATH_20260923B",
    "PMU_PHASOR_SIGMA_PU",
    "resolve_tagged_corpus_path",
    "PHYSICAL_HIF_DETECTION_LIMIT_SAMPLE_PATH",
    "PHYSICAL_HIF_SWEEP_SAMPLE_PATH",
    "PHYSICAL_IMBALANCE_SAMPLE_PATH",
    "CURRENT_TELEMETRY_IMBALANCE_SAMPLE_PATH",
    "UNBALANCE_SIGNATURE",
    "UNBALANCE_CURRENT_SIGNATURE",
    "WAVEFORM_SIGNATURE_MODES",
    "DEFAULT_WAVEFORM_SIGNATURE_MODE",
    "DEFAULT_BALANCED_ARTIFACT_DIR",
    "DEFAULT_CHI2_ALPHA",
    "SYNTHESIZED_MEASUREMENT_CANONICALIZATION_CONTRACT",
]
