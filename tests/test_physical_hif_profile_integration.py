"""Fresh physical-profile generation, WLS, replay and expert-prefix integration."""
from copy import deepcopy
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

# Load the native DSS extension before the other numerical runtimes on Windows.
pytest.importorskip("opendssdirect")

from research.gnn_screen.dataset import load_manifest
from research.gnn_screen.feature_schema import EDGE_FEATURES, EDGE_DIM, GLOBAL_DIM, NODE_DIM
from research.gnn_screen.graph_builder import build_graph
from research.gnn_screen.practical_corpus import SLOTS, generate_corpus, physical_hif_slots, reviewed_slots
from research.gnn_screen.wls_features import build_wls_features, configured_case
from research.reviewed_observable_context import build_observable_context
from three_phase_model.voltage_bases import (
    IEEE14_NOMINAL_KV, IEEE14_VOLTAGE_BASE_PROFILE_ID, hif_resistance_class,
)


PROFILE = "ieee14_physical_hif_v1"


@pytest.fixture(scope="module")
def physical_corpus(tmp_path_factory):
    directory = tmp_path_factory.mktemp("physical_hif_integration")
    core = directory / "baseline" / "core"
    report = generate_corpus(core, parents_by_split={"train": 1, "calibration": 1, "test": 1},
        seed=742021, noise_replicates=1, healthy_calibration_replicates=2,
        attempt_cap=2, scenario_profile=PROFILE, noise_profile="baseline", stage="full")
    paths = [core / "manifest.jsonl", core / "boundary_manifest.jsonl",
             *[Path(value["path"]) for value in report["auxiliary_manifests"].values()]]
    rows = [row for path in paths if path.read_text().strip() for row in load_manifest(path)]
    return directory, core, report, rows


def test_curriculum_and_evaluation_resistances_do_not_share_training_slots():
    for stage, weights in (("early", [2, 2, 1]), ("full", [1, 1, 1])):
        train = physical_hif_slots(stage=stage, split="train")
        hif = [options for _, families, options in train if families == ("hif",)]
        assert [sum(row["hif_ohm_band"] == band for row in hif) for band in range(3)] == weights
        assert all(row["voltage_kv"] == 69 for row in hif)
        assert not any("resistance_sweep" in options for _, _, options in train)
        evaluation = [options for _, _, options in physical_hif_slots(stage=stage, split="test")
                      if options.get("resistance_sweep")]
        assert {(row["voltage_kv"], row["resistance_ohm"]) for row in evaluation} == {
            (kv, resistance) for kv in (69, 13.8) for resistance in (50, 100, 200, 500, 1000, 2000, 5000)}
        assert all(row["destination"] == "hif_resistance_evaluation" for row in evaluation)
        # Detection-limit cohort: evaluation-only, 69 kV, named band, own destination.
        assert not any(options.get("hif_ohm_band") == "detection_limit" for _, _, options in train)
        assert not any(options.get("destination") == "hif_detection_limit_evaluation" for _, _, options in train)
        for split in ("validation", "test"):
            held_out = physical_hif_slots(stage=stage, split=split)
            detection = [(name, families, options) for name, families, options in held_out
                         if options.get("destination") == "hif_detection_limit_evaluation"]
            assert [name for name, _, _ in detection] == ["hif_69kv_detection_limit_0", "hif_69kv_detection_limit_1"]
            assert all(families == ("hif",) for _, families, _ in detection)
            assert all(options == {"hif_ohm_band": "detection_limit", "voltage_kv": 69.0,
                                   "destination": "hif_detection_limit_evaluation"} for _, _, options in detection)
            # Under the physical profile no slot carries the legacy pu resistance_range option.
            assert not any("resistance_range" in options for _, _, options in held_out)
        assert not any("resistance_range" in options for _, _, options in train)
    # The shared legacy slot table and the reviewed profile keep the pu option unchanged.
    assert dict((name, options) for name, _, options in SLOTS)["measurement_hif"]["resistance_range"] == (5., 40.)
    assert any("resistance_range" in options for _, _, options in reviewed_slots(stage="full", split="test"))
    assert physical_hif_slots(stage="full", split="calibration") == []
    # Standalone historical corpus generation remains opt-in/default compatible.
    assert inspect.signature(generate_corpus).parameters["scenario_profile"].default == "legacy_v1"
    assert [options["hif_band"] for _, families, options in reviewed_slots(stage="full", split="train")
            if families == ("hif",)] == [0, 1, 2]


def test_real_main_and_calibration_corpora_preserve_units_split_and_local_bases(physical_corpus):
    _, core, report, rows = physical_corpus
    assert report["schema"] == "practical_physical_wls_screen_corpus_v3"
    assert report["scenario_profile"] == PROFILE
    assert report["voltage_profile"] == IEEE14_VOLTAGE_BASE_PROFILE_ID
    assert report["parents_by_split"] == {"train": 1, "calibration": 1, "test": 1}
    assert report["healthy_max_balanced_equation_error_pu"] < 1e-6
    main = load_manifest(core / "manifest.jsonl")
    calibration = [row for row in main if row["split"] == "calibration"]
    assert len(calibration) == 1 and calibration[0]["families"] == []
    assert calibration[0]["noise_replicates"] == 2
    pure_hif = [row for row in rows if row["families"] == ["hif"] and row["split"] == "train"]
    assert pure_hif
    for row in rows:
        meta = row["offline_metadata"]
        assert meta["scenario_profile"] == PROFILE
        case = row["case"]
        assert {int(bus[0]): bus[9] for bus in case["bus"]} == IEEE14_NOMINAL_KV
        assert np.asarray(row["measurement_sigma"]).shape == (122,)
        if "hif" not in row["families"]:
            continue
        hif, units = meta["settings"]["hif"], meta["settings"]["hif_units"]
        assert "resistance_ohm" in hif and "resistance_pu" not in hif
        f, t = map(int, np.asarray(case["branch"])[hif["branch_row0"], :2])
        assert IEEE14_NOMINAL_KV[f] == IEEE14_NOMINAL_KV[t] == units["kv_ll"]
        assert units["resistance_ohm"] == hif["resistance_ohm"]
        assert units["impedance_base_ohm"] == pytest.approx(units["kv_ll"]**2 / case["baseMVA"])
        assert units["resistance_pu"] == pytest.approx(hif["resistance_ohm"] / units["impedance_base_ohm"])
        assert .25 <= hif["alpha"] <= .75 and hif["phase"] in (1, 2, 3)
        assert meta["settings"]["hif_resistance_class"] == hif_resistance_class(hif["resistance_ohm"])
        if meta["cohort"] == "hif_detection_limit_evaluation":
            assert 1000 <= hif["resistance_ohm"] < 5000 and units["kv_ll"] == 69
            assert row["split"] != "train"
            assert meta["settings"]["hif_population"] == "69kv_detection_limit_1000_to_5000_ohm"
            assert meta["settings"]["hif_resistance_class"] == "extreme_weak_hif"
            assert 21 <= units["resistance_pu"] <= 105.1
        elif meta["cohort"] != "hif_resistance_evaluation":
            assert 100 <= hif["resistance_ohm"] <= 1000 and units["kv_ll"] == 69
            assert meta["settings"]["hif_population"] == "69kv_main_100_to_1000_ohm"
            assert meta["settings"]["hif_resistance_class"] in {
                "moderately_high_resistance", "representative_hif", "weak_hif"}
    # The detection-limit cohort never appears in the training split anywhere.
    assert not [row for row in rows if row["offline_metadata"]["cohort"] == "hif_detection_limit_evaluation"
                and row["split"] == "train"]
    assert not [row for row in main if row["offline_metadata"]["cohort"] == "hif_detection_limit_evaluation"]
    ledger = [json.loads(line) for line in (core / "proposal_ledger.jsonl").read_text().splitlines()]
    assert not [event for event in ledger if event.get("outcome") == "simulation_failure"]


def test_real_evaluation_resistance_sweep_retains_low_voltage_and_extreme_cases(physical_corpus):
    _, core, report, _ = physical_corpus
    details = report["auxiliary_manifests"]["hif_resistance_evaluation"]
    assert not details["training_eligible"]
    sweep = load_manifest(details["path"])
    assert len(sweep) == 14
    actual = set()
    for row in sweep:
        meta = row["offline_metadata"]
        assert row["split"] == "test" and not meta["training_eligible"]
        assert row["families"] == ["hif"]
        assert meta["settings"]["hif_population"] == "voltage_stratified_resistance_sweep"
        actual.add((meta["settings"]["hif_units"]["kv_ll"], meta["settings"]["hif"]["resistance_ohm"]))
    assert actual == {(kv, resistance) for kv in (69, 13.8)
                      for resistance in (50, 100, 200, 500, 1000, 2000, 5000)}
    assert not (core / "weak_hif_evaluation_manifest.jsonl").exists()
    # 2026-09-19 detection-limit cohort: separate manifest, evaluation-only, 69 kV, 1000-5000 ohm.
    detection = report["auxiliary_manifests"]["hif_detection_limit_evaluation"]
    assert not detection["training_eligible"]
    assert Path(detection["path"]).name == "hif_detection_limit_evaluation_manifest.jsonl"
    assert detection["rows_by_split"]["train"] == 0 and detection["rows_by_split"]["test"] == 2
    cohort = load_manifest(detection["path"])
    assert len(cohort) == 2
    for row in cohort:
        meta = row["offline_metadata"]
        assert row["split"] == "test" and not meta["training_eligible"]
        assert row["families"] == ["hif"] and meta["cohort"] == "hif_detection_limit_evaluation"
        assert row["severity"] == "hif_detection_limit_evaluation"
        assert meta["settings"]["hif_population"] == "69kv_detection_limit_1000_to_5000_ohm"
        assert meta["settings"]["hif_resistance_class"] == "extreme_weak_hif"
        assert 1000 <= meta["settings"]["hif"]["resistance_ohm"] < 5000
        assert meta["settings"]["hif_units"]["kv_ll"] == 69
        assert not meta["scenario_policy"]["admitted_main"]
    assert report["profiles"]["hif_detection_limit"].startswith("Evaluation-only 69 kV detection-limit cohort")
    assert "extreme_weak_hif [1000, 5000) ohm" in report["profiles"]["hif_resistance_classification"]


def test_exported_transformers_and_checkpoint_feature_dimensions(physical_corpus):
    _, core, _, _ = physical_corpus
    controls = [row for row in load_manifest(core / "manifest.jsonl") if not row["families"]]
    assert {row["split"] for row in controls} == {"train", "calibration", "test"}
    for row in controls:
        model = core / "parents" / row["parent_id"] / "model"
        manifest = json.loads((model / "build_manifest.json").read_text())
        registry = json.loads((model / "asset_registry.json").read_text())
        assert manifest["voltage_profile"] == IEEE14_VOLTAGE_BASE_PROFILE_ID
        assert manifest["line_count"] == 16 and manifest["transformer_count"] == 4
        assert registry["branches"][13]["dss_element"].startswith("Transformer.")
        assert registry["branches"][14]["dss_element"].startswith("Line.")
        graph = build_graph(row["case"], row["z"], measurement_sigma=row["measurement_sigma"])
        assert (NODE_DIM, EDGE_DIM, GLOBAL_DIM) == (27, 40, 4)
        assert graph["x"].shape == (14, 27) and graph["edge_attr"].shape == (40, 40)
        assert graph["u"].shape == (4,)
        is_xfmr = graph["edge_attr"][0::2, EDGE_FEATURES.index("is_transformer")]
        assert np.flatnonzero(is_xfmr).tolist() == [7, 8, 9, 13]
        assert np.count_nonzero(graph["edge_attr"][0::2, EDGE_FEATURES.index("is_line")]) == 16
        assert graph["metadata"]["schema_version"] == "wls_screen_v1"


def test_known_voltage_bases_leave_balanced_wls_per_unit_fit_invariant(physical_corpus):
    _, core, _, _ = physical_corpus
    row = next(row for row in load_manifest(core / "manifest.jsonl") if not row["families"])
    specified = configured_case(row["case"])
    unspecified = deepcopy(specified)
    unspecified["bus"][:, 9] = 0
    noisy = np.asarray(row["z"]) + np.random.default_rng(823).normal(size=122) * row["measurement_sigma"]
    specified_wls = build_wls_features(specified, noisy, measurement_sigma=row["measurement_sigma"])
    unspecified_wls = build_wls_features(unspecified, noisy, measurement_sigma=row["measurement_sigma"])
    for field in ("fitted", "signed_normalized_residual", "variance", "theta_est_rad", "vm_est_pu"):
        np.testing.assert_array_equal(specified_wls[field], unspecified_wls[field])
    assert specified_wls["wls_objective"] == unspecified_wls["wls_objective"]
    a = build_graph(specified, noisy, measurement_sigma=row["measurement_sigma"])
    b = build_graph(unspecified, noisy, measurement_sigma=row["measurement_sigma"])
    np.testing.assert_array_equal(a["x"], b["x"])
    np.testing.assert_array_equal(a["u"], b["u"])
    difference = np.argwhere(a["edge_attr"] != b["edge_attr"])
    assert set(map(tuple, difference)) == {(edge, column) for edge in (26, 27) for column in (35, 36)}


def test_observable_context_replays_ohmic_hif_with_matching_mean_and_no_private_inputs(physical_corpus):
    _, core, _, rows = physical_corpus
    source = next(row for row in rows if row["split"] == "train" and row["families"] == ["hif"])
    metadata, receipt = build_observable_context(core / "manifest.jsonl", source, noise_seed=9871)
    assert receipt["maximum_clean_scada_error"] < 1e-8
    np.testing.assert_allclose(receipt["replayed_clean_scada"], source["z"], rtol=0, atol=1e-8)
    assert metadata["sigma_z"] == source["measurement_sigma"]
    assert len(metadata["three_phase_voltages"]) == 14
    assert len(metadata["three_phase_branch_currents"]) == 20
    encoded = json.dumps(metadata).lower()
    assert all(term not in encoded for term in ("resistance_ohm", "resistance_pu", "j_exact", "physical_audit", "replayed_clean"))
    assert metadata["noise_contract"]["scada_noise_drawn_here"] is False


def test_real_expert_prefix_admission_preserves_profile_and_frozen_measurements(physical_corpus):
    from research.filter_reviewed_training import filter_training_manifest, materialize_training_windows

    directory, core, _, _ = physical_corpus
    raw = [json.loads(line) for line in (core / "manifest.jsonl").read_text().splitlines()]
    source = next(row for row in raw if row["split"] == "train" and row["families"] == ["measurement"])
    selected = core / "single_measurement_admission.jsonl"
    selected.write_text(json.dumps(source) + "\n", encoding="utf-8")
    expected = materialize_training_windows(load_manifest(selected)[0])[0]
    report = filter_training_manifest(selected, directory / "real_admission")
    assert report["training_windows_kept"] == report["expert_prefix_targets"] == 1
    assert report["training_windows_excluded"] == 0
    assert report["complete_episode_success_claimed"] is False
    accepted = load_manifest(report["outputs"]["training_manifest"])[0]
    assert accepted["offline_metadata"]["scenario_profile"] == PROFILE
    assert accepted["measurement_kind"] == "observed" and "noise_seed" not in accepted
    np.testing.assert_array_equal(accepted["z"], expected["z"])
    admission = accepted["offline_metadata"]["training_admission"]
    assert admission["eligible"] and admission["wls_alarm"]
    assert admission["action_executed"] and admission["expert_valid"] and admission["execution_success"]
    assert admission["scope"] == "executed_expert_prefix_only"
    assert not admission["complete_diagnosis_or_repair_verified"]
    ledger = json.loads(Path(report["outputs"]["admission_ledger"]).read_text().splitlines()[0])
    assert ledger["expert_probe"]["initial_wls"]["success"]
    assert ledger["expert_probe"]["fault_actionable"]
    assert ledger["context_generation"]["maximum_clean_scada_error"] < 1e-8
    assert ledger["runtime_model"]["private_fault_model_used"] is False
    assert Path(report["outputs"]["expert_chat_sft"]).read_text().strip()


def test_raw_physical_profile_training_requires_executed_observed_admission(physical_corpus):
    from research.gnn_screen.train import validate_reviewed_training_admission

    _, core, _, _ = physical_corpus
    original = (core / "manifest.jsonl").read_bytes()
    with pytest.raises(ValueError, match="fixed observed replica"):
        validate_reviewed_training_admission(core / "manifest.jsonl")
    assert (core / "manifest.jsonl").read_bytes() == original
