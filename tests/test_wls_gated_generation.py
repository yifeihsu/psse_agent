"""Generation under the wls_gated_diagnostics evidence contract (2026-09-23).

Detection uses only balanced SCADA and its WLS: no fault flag, seeded signature,
hint or precomputed diagnosis reaches the agent on any family.  A root carries
the auxiliary measurement streams the ground truth generated for it (PMU phasors
on HIF and unbalance roots, spectra on harmonic roots, breaker telemetry on
topology roots, repeated scans on HIF and parameter roots); the environment
releases them only after a current WLS alarm.  These tests cover the generator,
the suite partition, the runner's provenance guards and the corpus constants.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from psse_env.dagger.suite_builder import partition_release_scenario_v1
from psse_env.evidence_profile import (
    DEFAULT_EVIDENCE_PROFILE, PRECOMPUTED_DIAGNOSIS_FIELDS, WLS_GATED_PROFILE,
)
from psse_env.providers.scenario_generator import (
    PHYSICAL_HIF_CORPUS_TAG, PHYSICAL_HIF_DETECTION_LIMIT_SAMPLE_PATH, PHYSICAL_HIF_SAMPLE_PATHS,
    PHYSICAL_HIF_SAMPLE_PATHS_20260923B, PHYSICAL_HIF_SWEEP_SAMPLE_PATH, PHYSICAL_IMBALANCE_SAMPLE_PATH,
    PHYSICAL_IMBALANCE_SAMPLE_PATH_20260923B, PMU_PHASOR_SIGMA_PU, Round0ScenarioGenerator,
    resolve_tagged_corpus_path,
)
from scripts.run_dagger_research import (
    build_research_mixture, parser, resolve_scenario_sources, validate_training_evidence_profile,
)

# The committed 2026-09-23b subsets (regulated limits, tolerance 1e-8, phasor sigma
# 5e-3 / 1e-3) stand in for the pending 20260923opf corpora as generation sources.
HIF_CORPUS = PHYSICAL_HIF_SAMPLE_PATHS_20260923B[1]
IMBALANCE_CORPUS = PHYSICAL_IMBALANCE_SAMPLE_PATH_20260923B
WAVEFORM_FAMILIES = ("hif", "harmonic", "three_phase_unbalance")
PLAN = {
    "no_error": 1, "measurement": 1, "multi_measurement": 1, "parameter": 1, "topology": 1,
    "harmonic": 1, "hif": 1, "measurement+parameter": 1, "measurement+topology": 1,
    "measurement+hif": 1, "three_phase_unbalance": 1, "telemetry_no_disturbance": 1,
}
# Truth, labels and precomputed diagnoses that must not appear anywhere in a
# root's execution metadata under the strict profile.
PRIVATE_MARKERS = (
    *sorted(PRECOMPUTED_DIAGNOSIS_FIELDS), "detected_top1", "detected_top3", "z_clean",
    "three_phase_voltages_clean", "three_phase_branch_currents_clean", "window_metadata",
    '"label"', "hidden_truth", "true_hif_errors", "true_unbalance_errors", "true_topology_errors",
    "true_parameter_errors", "true_measurement_errors", "true_harmonic_errors", "source_bus",
    "target_bus", "resistance_class", "r_hif_pu", "true_cb_closed", "initial_states",
    "family_hint", "correction_hint",
)


def _metadata_json(scenario) -> str:
    return json.dumps(scenario["metadata"], sort_keys=True)


@pytest.fixture(scope="module")
def roots():
    if not (HIF_CORPUS.is_file() and IMBALANCE_CORPUS.is_file()):
        pytest.skip("committed 20260923b HIF/unbalance corpora are not checked out")
    generator = Round0ScenarioGenerator(
        seed=20260923, hif_sample_paths=[HIF_CORPUS], imbalance_sample_path=IMBALANCE_CORPUS,
        hif_max_scans=3, normalized_residual_threshold=4.0,
    )
    assert generator.evidence_profile == WLS_GATED_PROFILE
    built = generator.build(PLAN)
    by_family = {}
    for scenario in built:
        by_family.setdefault(scenario["scenario_family"], scenario)
    missing = sorted(set(PLAN) - set(by_family))
    assert not missing, f"no admitted root for {missing}: {generator.report()['skipped_by_reason']}"
    return generator, by_family


# ------------------------------------------------------------------ defaults


def test_the_gated_profile_is_the_default_everywhere():
    assert DEFAULT_EVIDENCE_PROFILE == WLS_GATED_PROFILE == "wls_gated_diagnostics"
    generator = Round0ScenarioGenerator(validate=False)
    assert generator.evidence_profile == WLS_GATED_PROFILE
    assert set(generator.waveform_signature_mode.values()) == {"discovered"}
    assert generator.report()["evidence_profile"] == WLS_GATED_PROFILE
    base = generator._base_scenario("root", case="case14", measurements=[0.0] * generator.nz, family="no_error")
    assert base["metadata"]["evidence_profile"] == WLS_GATED_PROFILE
    args = parser().parse_args(["--d0-raw", "raw", "--d0-train", "train", "--adapter-path", "adapter", "--output-dir", "out"])
    assert args.evidence_profile == WLS_GATED_PROFILE and args.hif_signature_mode == "discovered"
    assert resolve_scenario_sources(plan_families={"measurement"})["evidence_profile"] == WLS_GATED_PROFILE


@pytest.mark.parametrize("family", WAVEFORM_FAMILIES)
def test_flagged_signatures_are_rejected_under_every_strict_profile(family):
    for profile in ("wls_gated_diagnostics", "scada_only"):
        with pytest.raises(ValueError, match="auxiliary"):
            Round0ScenarioGenerator(evidence_profile=profile, waveform_signature_mode={family: "flagged"})
    with pytest.raises(ValueError, match="auxiliary"):
        Round0ScenarioGenerator(waveform_signature_mode={family: "flagged"})
    with pytest.raises(ValueError, match="auxiliary"):
        resolve_scenario_sources(plan_families={family}, signature_modes={family: "flagged"})
    legacy = Round0ScenarioGenerator(evidence_profile="auxiliary_diagnostics", waveform_signature_mode={family: "flagged"})
    assert legacy.waveform_signature_mode[family] == "flagged"


# ------------------------------------------------------- per-family contents


def test_every_root_records_the_profile_and_carries_no_flag_hint_or_diagnosis(roots):
    generator, by_family = roots
    for family, scenario in by_family.items():
        metadata = scenario["metadata"]
        assert metadata["evidence_profile"] == WLS_GATED_PROFILE, family
        assert "unresolved_signatures" not in scenario, family
        assert "unresolved_signatures" not in scenario.get("semantic_field_provenance", {}), family
        for key in ("family_hint", "correction_hint", "expected_actions", "hint"):
            assert key not in scenario, (family, key)
        encoded = _metadata_json(scenario)
        for marker in PRIVATE_MARKERS:
            assert marker not in encoded, (family, marker)
        # Truth stays on the private side for the offline audit.
        assert scenario.get("hidden_truth") is not None or any(
            scenario.get(key) for key in ("true_measurement_errors", "true_parameter_errors", "true_topology_errors")
        ) or family == "no_error", family
    assert {entry["scenario_family"] for entry in generator.manifest} == set(PLAN)
    assert all(entry["evidence_profile"] == WLS_GATED_PROFILE for entry in generator.manifest)
    assert generator.report()["evidence_profile"] == WLS_GATED_PROFILE


@pytest.mark.parametrize("family", ["no_error", "measurement", "multi_measurement"])
def test_balanced_corpus_roots_carry_scada_declarations_only(roots, family):
    _, by_family = roots
    metadata = by_family[family]["metadata"]
    assert set(metadata) == {"sigma_z", "evidence_profile"}
    assert len(metadata["sigma_z"]) == len(by_family[family]["measurements"])


@pytest.mark.parametrize("family", ["parameter", "measurement+parameter"])
def test_parameter_roots_carry_repeated_scada_scans_without_truth_states(roots, family):
    _, by_family = roots
    scans = by_family[family]["metadata"]["parameter_scans"]
    assert set(scans) <= {"z_scans", "sigma_z", "scan_indices", "time_tags", "scan_index", "time_tag"}
    assert len(scans["z_scans"]) >= 2 and all(len(scan) == len(scans["sigma_z"]) for scan in scans["z_scans"])
    assert "initial_state" not in _metadata_json(by_family[family])


@pytest.mark.parametrize("family", ["topology", "measurement+topology"])
def test_topology_roots_carry_breaker_telemetry_but_not_the_true_status(roots, family):
    _, by_family = roots
    scenario = by_family[family]
    metadata = scenario["metadata"]
    for key in ("substation_telemetry", "reported_breaker_status", "operator_noise", "structural_zero_indices", "operator_layout"):
        assert key in metadata, (family, key)
    assert scenario["true_topology_errors"], family
    truth = scenario["true_topology_errors"][0]
    # The reported map shows the operator's (wrong) belief, never the truth flag.
    assert "true_cb_closed" not in _metadata_json(scenario)
    assert truth["cb_name"] in metadata["reported_breaker_status"]


def test_harmonic_root_carries_the_spectrum_but_not_the_source_bus(roots):
    _, by_family = roots
    scenario = by_family["harmonic"]
    spectrum = scenario["metadata"]["harmonic_measurements"]
    assert spectrum and {"bus", "h", "V_real", "V_imag", "sigma"} <= set(spectrum[0])
    assert scenario["metadata"]["harmonic_orders"]
    assert scenario["hidden_truth"]["true_harmonic_errors"][0]["bus_1based"] in range(1, 15)
    assert "source_bus" not in _metadata_json(scenario)


@pytest.mark.parametrize("family", ["hif", "measurement+hif"])
def test_hif_roots_carry_pmu_phasors_scan_window_and_runtime_without_the_cached_diagnosis(roots, family):
    _, by_family = roots
    scenario = by_family[family]
    metadata = scenario["metadata"]
    rows = [json.loads(line) for line in HIF_CORPUS.read_text(encoding="utf-8").splitlines() if line.strip()]
    declared_sigma = {float(row["three_phase_sigma"]) for row in rows if row["scenario"] == "high_impedance_fault"}
    declared_current_sigma = {float(row["branch_current_sigma_pu"]) for row in rows if row["scenario"] == "high_impedance_fault"}
    assert len(declared_sigma) == 1 and len(declared_current_sigma) == 1
    assert metadata["three_phase_sigma"] == declared_sigma.pop()
    assert metadata["branch_current_sigma_pu"] == declared_current_sigma.pop()
    voltages = metadata["three_phase_voltages"]
    currents = metadata["three_phase_branch_currents"]
    assert len(voltages) == 14 and {"bus", "vln_pu", "ang_deg"} <= set(voltages[0])
    assert len(currents) == 20 and {"branch_row0", "i_from_pu", "i_to_pu"} <= set(currents[0])
    window = metadata["hif_scan_window"]
    assert 2 <= len(window["scans"]) <= 3
    # The window is keyed by the opaque root id (the composed root re-derives
    # its own scenario_id from the HIF base), never by the corpus row id.
    assert window["scan_window_path"].startswith("r0_") and "hif" not in window["scan_window_path"]
    if family == "hif":
        assert window["scan_window_path"] == scenario["scenario_id"]
    assert "window_metadata" not in window
    for scan in window["scans"]:
        assert {"scan_index", "z_obs", "three_phase_voltages", "three_phase_branch_currents", "op_point"} <= set(scan)
        assert "z_clean" not in scan
    runtime = metadata["hif_runtime"]
    assert runtime["three_phase_voltages"] and runtime["three_phase_branch_currents"]
    assert runtime["three_phase_sigma"] == metadata["three_phase_sigma"]
    assert runtime["op_point"]["load_scale"] == runtime["load_scale"]
    assert metadata["measurement_convention"]["shunt_convention"] == "ybus"
    assert set(metadata["noise_contract"]["channels"]) == {"scada", "three_phase_voltages", "three_phase_branch_currents"}
    assert "nlm_diagnostic" not in metadata and "faulted_model_dir" not in metadata
    assert scenario["hidden_truth"]["true_hif_errors"][0]["branch_row0"] in range(20)
    assert scenario["release_audit"]["signature_mode"] == "discovered"
    assert scenario["release_audit"]["sensor_signatures_withheld"] == ["hif_suspected_zero_sequence"]


def test_unbalance_root_and_its_balanced_control_carry_the_same_pmu_channels(roots):
    _, by_family = roots
    unbalance = by_family["three_phase_unbalance"]
    control = by_family["telemetry_no_disturbance"]
    for scenario in (unbalance, control):
        metadata = scenario["metadata"]
        assert len(metadata["three_phase_voltages"]) == 14
        assert len(metadata["three_phase_branch_currents"]) == 20
        assert metadata["three_phase_sigma"] > 0 and metadata["branch_current_sigma_pu"] > 0
        assert {"scada", "three_phase_voltages", "three_phase_branch_currents"} <= set(metadata["noise_contract"]["channels"])
        assert "target_bus" not in _metadata_json(scenario) and "op_point" not in metadata
    assert unbalance["metadata"]["three_phase_sigma"] == control["metadata"]["three_phase_sigma"]
    assert unbalance["hidden_truth"]["true_unbalance_errors"]
    assert control["hidden_truth"] == {"true_unbalance_errors": [], "control_kind": "telemetry_present_no_disturbance"}
    assert unbalance["release_audit"]["signature_mode"] == "discovered"
    assert unbalance["release_audit"]["sensor_signatures_withheld"]


@pytest.mark.parametrize("family", ["hif", "three_phase_unbalance", "topology", "harmonic", "parameter"])
def test_release_partition_keeps_the_streams_and_moves_truth_to_the_audit(roots, family):
    _, by_family = roots
    envelope = partition_release_scenario_v1(deepcopy(by_family[family]))
    execution = envelope["execution"]
    assert execution["evidence_profile"] == WLS_GATED_PROFILE
    assert execution["metadata"]["evidence_profile"] == WLS_GATED_PROFILE
    assert "unresolved_signatures" not in execution
    encoded = json.dumps(execution, sort_keys=True)
    for marker in PRIVATE_MARKERS:
        assert marker not in encoded, (family, marker)
    kept = {
        "hif": ("three_phase_voltages", "three_phase_branch_currents", "hif_scan_window", "hif_runtime", "noise_contract"),
        "three_phase_unbalance": ("three_phase_voltages", "three_phase_branch_currents", "noise_contract"),
        "topology": ("substation_telemetry", "reported_breaker_status", "operator_noise", "structural_zero_indices"),
        "harmonic": ("harmonic_measurements", "harmonic_orders"),
        "parameter": ("parameter_scans",),
    }[family]
    for key in kept:
        assert key in execution["metadata"], (family, key)
    assert envelope["audit"]["truth"]


def test_partition_strips_seeded_signatures_and_cached_diagnoses_from_gated_scenarios():
    scenario = {
        "scenario_id": "r0_gated", "scenario_family": "hif", "error_cardinality": 1, "source_tier": "test_fixture",
        "case": "case14", "measurements": [1.0, 2.0], "clean_case": "case14", "clean_measurements": [1.0, 2.0],
        "unresolved_signatures": ["hif_suspected_zero_sequence"],
        "semantic_field_provenance": {"measurements": "deployment_sensor:scada_snapshot",
                                      "unresolved_signatures": "deployment_sensor:waveform_capture"},
        "hidden_truth": {"true_hif_errors": [{"branch_row0": 3}]},
        "metadata": {
            "evidence_profile": WLS_GATED_PROFILE, "sigma_z": [0.001, 0.01],
            "three_phase_voltages": [{"bus": "b1", "vln_pu": [1, 1, 1], "ang_deg": [0, -120, 120]}],
            "three_phase_sigma": 1e-4, "label": {"branch_row0": 3},
            "nlm_diagnostic": {"success": True, "top_hif_groups": [{"branch_row0": 3}], "detected_top1": True},
            "faulted_model_dir": "/tmp/faulted", "op_point": {"load_scale": 1.0},
            "hif_scan_window": {"scan_window_path": "ieee14_hif_000001", "scans": [
                {"scan_index": 0, "z_obs": [1.0, 2.0], "z_clean": [1.0, 2.0], "op_point": {"load_scale": 1.0}}]},
        },
    }
    envelope = partition_release_scenario_v1(scenario)
    execution = envelope["execution"]
    metadata = execution["metadata"]
    assert "unresolved_signatures" not in execution
    assert metadata["three_phase_voltages"] and metadata["three_phase_sigma"] == 1e-4
    assert metadata["hif_scan_window"]["scans"][0]["z_obs"] == [1.0, 2.0]
    for key in ("label", "nlm_diagnostic", "faulted_model_dir", "op_point"):
        assert key not in metadata, key
    assert "z_clean" not in metadata["hif_scan_window"]["scans"][0]
    assert envelope["audit"]["truth"]["true_hif_errors"] == [{"branch_row0": 3}]


# ------------------------------------------------------------ runner provenance


def test_runner_rejects_undeclared_and_foreign_rows_under_the_gated_profile():
    gated = {"example_id": "g", "physical_root_fingerprint": "r", "metadata": {"evidence_profile": WLS_GATED_PROFILE},
             "messages": [{"role": "user", "content": "observed"}]}
    validate_training_evidence_profile([gated])
    for row in ({"example_id": "old"}, {"example_id": "s", "metadata": {"evidence_profile": "scada_only"}},
                {"example_id": "a", "evidence_profile": "auxiliary_diagnostics"},
                {"example_id": "m", "evidence_profile": WLS_GATED_PROFILE,
                 "policy_observation": {"evidence_profile": "scada_only"}}):
        with pytest.raises(ValueError, match="do not relabel"):
            validate_training_evidence_profile([row])
    mixed, report = build_research_mixture([gated], [dict(gated, example_id="g1", physical_root_fingerprint="r1")],
                                           d1_share=0.5, d1_cap=None, seed=1)
    assert report["evidence_profile"] == WLS_GATED_PROFILE and len(mixed) == 2
    with pytest.raises(ValueError, match="evidence profile"):
        build_research_mixture([gated], [{"example_id": "s", "metadata": {"evidence_profile": "scada_only"}}],
                               d1_share=0.5, d1_cap=None, seed=1)


# ------------------------------------------------------------------- corpora


def test_corpus_constants_follow_the_pending_opf_tag_and_the_pmu_precision():
    assert PHYSICAL_HIF_CORPUS_TAG == "20260923opf"
    assert PMU_PHASOR_SIGMA_PU == 1e-4
    assert len(PHYSICAL_HIF_SAMPLE_PATHS) == 4
    stems = ("hif_physical69_main_train_detectable", "hif_physical69_main_valid_detectable",
             "hif_physical69_main_train_extra_detectable", "hif_physical69_main_valid_extra_detectable")
    for path, stem in zip(PHYSICAL_HIF_SAMPLE_PATHS, stems):
        assert path.name == "samples.jsonl"
        assert path.parent.name.startswith(f"{stem}_") and path.parent.name.endswith(f"x10_{PHYSICAL_HIF_CORPUS_TAG}")
    assert PHYSICAL_HIF_DETECTION_LIMIT_SAMPLE_PATH.parent.name == f"hif_physical69_detection_limit_21x10_{PHYSICAL_HIF_CORPUS_TAG}"
    assert PHYSICAL_HIF_SWEEP_SAMPLE_PATH.parent.name == f"hif_physical_sweep_eval_336x10_{PHYSICAL_HIF_CORPUS_TAG}"
    imbalance = PHYSICAL_IMBALANCE_SAMPLE_PATH.parent.name
    assert imbalance.startswith("out_measurements_imbalance_currents_ybus_detectable_") and imbalance.endswith(f"_{PHYSICAL_HIF_CORPUS_TAG}")
    # The committed 20260923b subsets remain addressable for replay identity.
    assert [p.parent.name for p in PHYSICAL_HIF_SAMPLE_PATHS_20260923B] == [
        "hif_physical69_main_train_detectable_27x10_20260923b", "hif_physical69_main_valid_detectable_8x10_20260923b",
        "hif_physical69_main_train_extra_detectable_77x10_20260923b", "hif_physical69_main_valid_extra_detectable_19x10_20260923b"]
    assert PHYSICAL_IMBALANCE_SAMPLE_PATH_20260923B.parent.name == "out_measurements_imbalance_currents_ybus_detectable_160_20260923b"


def test_tagged_corpus_resolver_reads_the_admitted_count_from_the_directory_name(tmp_path: Path):
    pending = resolve_tagged_corpus_path("hif_physical69_main_train_detectable", "t", root=tmp_path)
    assert pending.parent.name == "hif_physical69_main_train_detectable_PENDINGx10_t" and not pending.is_file()
    (tmp_path / "hif_physical69_main_train_detectable_31x10_t").mkdir()
    (tmp_path / "hif_physical69_main_train_detectable_31x10_t" / "samples.jsonl").write_text("{}\n", encoding="utf-8")
    resolved = resolve_tagged_corpus_path("hif_physical69_main_train_detectable", "t", root=tmp_path)
    assert resolved.parent.name == "hif_physical69_main_train_detectable_31x10_t" and resolved.is_file()
    # Another tag or a directory without samples never matches.
    (tmp_path / "hif_physical69_main_train_detectable_9x10_other").mkdir()
    (tmp_path / "hif_physical69_main_train_detectable_12x10_t").mkdir()
    assert resolve_tagged_corpus_path("hif_physical69_main_train_detectable", "t", root=tmp_path) == resolved
    (tmp_path / "hif_physical69_main_train_detectable_12x10_t" / "samples.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="ambiguous"):
        resolve_tagged_corpus_path("hif_physical69_main_train_detectable", "t", root=tmp_path)
    unbalance = resolve_tagged_corpus_path("out_measurements_imbalance_currents_ybus_detectable", "t", suffix="", root=tmp_path)
    assert unbalance.parent.name == "out_measurements_imbalance_currents_ybus_detectable_PENDING_t"


def test_regenerated_hif_corpora_declare_the_pmu_phasor_sigma():
    """Runs against the 20260923opf corpora once they exist; skipped while pending."""
    present = [path for path in PHYSICAL_HIF_SAMPLE_PATHS if path.is_file()]
    if not present:
        pytest.skip(f"{PHYSICAL_HIF_CORPUS_TAG} HIF corpora are pending regeneration")
    for path in present:
        meta = json.loads(path.with_name("meta.json").read_text(encoding="utf-8"))
        assert meta["three_phase_sigma"] == pytest.approx(PMU_PHASOR_SIGMA_PU, rel=1e-9), path.parent.name
        assert meta["branch_current_sigma_pu"] == pytest.approx(PMU_PHASOR_SIGMA_PU, rel=1e-9), path.parent.name
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row["scenario"] != "high_impedance_fault":
                continue
            assert row["three_phase_sigma"] == pytest.approx(PMU_PHASOR_SIGMA_PU, rel=1e-9)
            assert row["branch_current_sigma_pu"] == pytest.approx(PMU_PHASOR_SIGMA_PU, rel=1e-9)
            for scan in row["scans"]:
                assert scan["three_phase_sigma"] == pytest.approx(PMU_PHASOR_SIGMA_PU, rel=1e-9)
                assert scan["branch_current_sigma_pu"] == pytest.approx(PMU_PHASOR_SIGMA_PU, rel=1e-9)
            break
