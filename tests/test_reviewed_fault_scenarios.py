"""Independent physical/noise/adapter checks for the reviewed scenario bundle."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest
from pypower.api import case14, ppoption, runpf
from threadpoolctl import threadpool_limits

from psse_env.fault_profiles import measurement_sigma
from research import reviewed_fault_scenarios as reviewed
from research.gnn_screen.dataset import (
    MEASUREMENT_CONVENTION, content_hash, load_manifest, prepare_corpus, write_json,
)
from research.gnn_screen.graph_builder import build_graph
from Transmission.generate_measurements import compute_measurements_pu


@pytest.fixture(autouse=True)
def one_native_thread():
    with threadpool_limits(limits=1):
        yield


@pytest.fixture
def core(tmp_path):
    """Small legitimate fixed-mean manifest, independent of practical sampling."""
    root = tmp_path / "core"
    solved, ok = runpf(case14(), ppoption(VERBOSE=0, OUT_ALL=0, PF_TOL=1e-11))
    assert ok
    mean = compute_measurements_pu(solved)
    sigma = measurement_sigma(14, 20)
    rows = []
    for parent, split, seed in (("parent_train", "train", 813), ("parent_calibration", "calibration", 814)):
        case_path = f"parents/{parent}/configured.json"
        write_json(root / case_path, solved)
        write_json(root / f"parents/{parent}/model/source_case.json", solved)
        write_json(root / f"parents/{parent}/physics_audit.json", {"fixture": "independent solved power flow"})
        rows.append({"parent_id": parent, "window_id": f"{parent}:healthy", "split": split,
            "case": case_path, "z": mean.tolist(), "families": [], "severity": "healthy",
            "measurement_kind": "noiseless_mean", "measurement_convention": MEASUREMENT_CONVENTION,
            "measurement_sigma": "measurement_sigma.json", "noise_replicates": 2, "noise_seed": seed,
            "offline_metadata": {"physical_audit_path": f"parents/{parent}/physics_audit.json",
                "observable_strength": {"J_exact": 0., "source_only": True}, "private_audit_sentinel": "never_in_graph"}})
    fault = deepcopy(rows[0])
    fault_mean = mean.copy()
    fault_mean[45] += .13
    write_json(root / "fault_mean.json", fault_mean)
    fault.update(window_id="parent_train:fixed_meter", z="fault_mean.json", families=["measurement"], severity="fixed_13sigma")
    rows.append(fault)
    write_json(root / "measurement_sigma.json", sigma)
    reviewed.write_jsonl(root / "manifest.jsonl", rows)
    return root, solved, mean, rows


def _capture_real_graphs(manifest):
    calls = []

    def spy(case, measurements, **kwargs):
        assert set(kwargs) == {"measurement_sigma", "solver_settings"}
        calls.append((np.array(measurements, copy=True), np.array(kwargs["measurement_sigma"], copy=True)))
        return build_graph(case, measurements, **kwargs)

    corpus = prepare_corpus(manifest, graph_builder=spy)
    assert not corpus.invalid
    for sample in corpus.samples:
        metadata = json.dumps(sample.graph["metadata"])
        assert "never_in_graph" not in metadata
        assert "offline_metadata" not in metadata
        assert "physical_audit" not in metadata
        assert "physical_severity" not in metadata
        assert "families" not in metadata
    return calls, corpus


def test_harmonic_zero_thd_matches_independent_balanced_mean_and_noise_is_single_draw(core, tmp_path):
    root, _case, expected, _rows = core
    report = reviewed.build_harmonic_companion(root, tmp_path / "harmonics", seed=41)
    rows = reviewed.read_jsonl(Path(report["path"]))
    assert len(rows) == 24
    healthy = [row for row in rows if row["cohort"] == "healthy"]
    assert len(healthy) == 12
    for row in healthy:
        np.testing.assert_allclose(row["offline_audit"]["z_exact"], expected, atol=1e-10, rtol=0)
        assert row["offline_audit"]["observable_strength"]["success"]
        assert row["offline_audit"]["observable_strength"]["J_exact"] < 1e-5
    paired = {}
    for row in rows:
        sigma = measurement_sigma(14, 20, row["noise_profile"])
        mean = np.array(row["offline_audit"]["z_exact"])
        draw = np.random.default_rng(row["offline_audit"]["noise_seed"]).standard_normal(122)
        np.testing.assert_array_equal(row["sigma_z"], sigma)
        np.testing.assert_allclose(row["measurements"], mean + draw * sigma, atol=1e-15, rtol=0)
        assert row["noise_contract"]["draw_count"] == 1
        assert row["harmonic_phasors"]
        for phasors in row["harmonic_phasors"].values():
            for item in phasors:
                assert "V_complex_true" not in item
                assert "V_complex_noisy" in item
                assert item["sigma_semantics"] == "per_component"
        np.testing.assert_array_equal(row["noise_contract"]["applied_sigma"], sigma)
        np.testing.assert_array_equal(row["noise_contract"]["estimator_sigma"], sigma)
        # The only accuracy-dependent random term is sigma times the same draw.
        key = row["window_id"].rsplit(":", 1)[0]
        if key in paired:
            assert row["offline_audit"]["noise_seed"] == paired[key]["offline_audit"]["noise_seed"]
            np.testing.assert_array_equal(mean, paired[key]["offline_audit"]["z_exact"])
        paired[key] = row
    assert {row["cohort"] for row in rows if row["split"] == "calibration"} == {"healthy"}


def test_accuracy_views_freeze_physics_labels_parent_splits_and_seed_with_real_loader(core, tmp_path):
    root, _case, _mean, _rows = core
    before = (root / "manifest.jsonl").read_bytes()
    reports = reviewed.build_accuracy_views(root, tmp_path / "views")
    baseline = load_manifest(root / "manifest.jsonl")
    base_calls, _ = _capture_real_graphs(root / "manifest.jsonl")
    for profile in ("accuracy_005", "accuracy_002"):
        path = tmp_path / "views" / profile / "manifest.jsonl"
        rows = load_manifest(path)
        assert len(rows) == len(baseline)
        for original, alternate in zip(baseline, rows):
            for key in ("parent_id", "window_id", "families", "split", "severity", "noise_seed", "noise_replicates", "measurement_kind"):
                assert original[key] == alternate[key]
            np.testing.assert_array_equal(original["z"], alternate["z"])
            assert content_hash(original["case"]) == content_hash(alternate["case"])
            np.testing.assert_array_equal(alternate["measurement_sigma"], measurement_sigma(14, 20, profile))
            audit = alternate["offline_metadata"]
            assert audit["source_observable_strength"] == original["offline_metadata"]["observable_strength"]
            assert not audit["accuracy_view"]["cohort_membership_reselected"]
            assert audit["accuracy_view"]["physical_mean_sha256"] == content_hash(alternate["z"])
            assert (path.parent / audit["physical_audit_path"]).is_file()
        alternate_calls, _ = _capture_real_graphs(path)
        assert len(base_calls) == len(alternate_calls) == 6
        for index, ((baseline_z, baseline_sigma), (alternate_z, alternate_sigma)) in enumerate(zip(base_calls, alternate_calls)):
            exact = np.asarray(baseline[index // 2]["z"])
            np.testing.assert_allclose((baseline_z - exact) / baseline_sigma,
                                       (alternate_z - exact) / alternate_sigma, atol=1e-12, rtol=1e-12)
        assert reports[profile]["physical_population_frozen"]
    assert (root / "manifest.jsonl").read_bytes() == before


def test_accuracy_views_reject_observed_rows_instead_of_drawing_noise_twice(core, tmp_path):
    root, _case, _mean, rows = core
    rows[0]["measurement_kind"] = "observed"
    reviewed.write_jsonl(root / "manifest.jsonl", rows)
    with pytest.raises(ValueError, match="observed data must never be re-noised"):
        reviewed.build_accuracy_views(root, tmp_path / "bad_views", noise_profiles=("accuracy_005",))


def test_ct_pt_companion_couples_only_target_pq_and_loads_without_audit_leak(core, tmp_path):
    root, _case, mean, _rows = core
    report = reviewed.build_measurement_chain_companion(root, tmp_path / "chains", seed=57)
    manifest = Path(report["path"])
    rows = load_manifest(manifest)
    assert len(rows) == 4
    assert not report["included_in_main_training_manifest"]
    for row in rows:
        assert row["measurement_kind"] == "noiseless_mean"
        assert row["split"] == "train"
        audit = row["offline_metadata"]["measurement_chain"]
        p_index, q_index = audit["indices0"]
        gain = audit.get("ct_gain", 1.)
        angle = -audit.get("ct_angle_rad", 0.)
        expected = gain * np.exp(1j * angle) * complex(mean[p_index], mean[q_index])
        z = np.asarray(row["z"])
        assert complex(z[p_index], z[q_index]) == pytest.approx(expected)
        keep = np.ones(122, dtype=bool)
        keep[[p_index, q_index]] = False
        np.testing.assert_array_equal(z[keep], mean[keep])
        assert not audit["field_calibrated"]
        np.testing.assert_array_equal(row["measurement_sigma"], measurement_sigma(14, 20))
    calls, corpus = _capture_real_graphs(manifest)
    assert len(corpus.samples) == len(calls) == 8
    for index, row in enumerate(rows):
        seed = int(content_hash({"seed": row["noise_seed"], "parent": row["parent_id"], "window": row["window_id"]})[:16], 16)
        rng = np.random.default_rng(seed)
        for replicate in range(2):
            actual, sigma = calls[index * 2 + replicate]
            expected = np.asarray(row["z"]) + rng.normal(size=122) * sigma
            np.testing.assert_array_equal(actual, expected)


def test_attribution_four_arms_preserve_mean_algebra_labels_and_actual_common_noise(core, tmp_path):
    root, case, healthy_mean, rows = core
    faulty_case = deepcopy(case)
    faulty_case["branch"][0, 2] *= 2.
    solved, ok = runpf(faulty_case, ppoption(VERBOSE=0, OUT_ALL=0, PF_TOL=1e-11))
    assert ok
    physical_mean = compute_measurements_pu(solved)
    physical = deepcopy(rows[0])
    physical.update(window_id="parent_train:physical_parameter", families=["parameter"], z=physical_mean.tolist())
    mixed = deepcopy(physical)
    mixed_mean = physical_mean.copy()
    mixed_mean[45] += .13
    mixed.update(window_id="parent_train:physical_parameter_plus_meter", families=["parameter", "measurement"], z=mixed_mean.tolist())
    mixed["offline_metadata"]["component_core_audit"] = {"source_window_id": physical["window_id"]}
    reviewed.write_jsonl(root / "manifest.jsonl", rows + [physical, mixed])
    report = reviewed.build_attribution_pairs(root, tmp_path / "attribution")
    loaded = load_manifest(report["path"])
    assert len(loaded) == report["rows"] == 4
    by_role = {row["offline_metadata"]["attribution_role"]: row for row in loaded}
    assert set(by_role) == {"healthy", "physical_only", "meter_only", "mixed"}
    assert by_role["healthy"]["families"] == []
    assert by_role["physical_only"]["families"] == ["parameter"]
    assert by_role["meter_only"]["families"] == ["measurement"]
    assert by_role["mixed"]["families"] == ["parameter", "measurement"]
    np.testing.assert_array_equal(by_role["healthy"]["z"], healthy_mean)
    np.testing.assert_array_equal(by_role["physical_only"]["z"], physical_mean)
    np.testing.assert_array_equal(by_role["mixed"]["z"], mixed_mean)
    np.testing.assert_allclose(np.array(by_role["mixed"]["z"]) - by_role["physical_only"]["z"],
                               np.array(by_role["meter_only"]["z"]) - by_role["healthy"]["z"], atol=1e-15, rtol=0)
    assert len({row["parent_id"] for row in loaded}) == 1
    assert len({row["noise_seed"] for row in loaded}) == 1
    assert len({row["window_id"] for row in loaded}) == 4
    assert {row["split"] for row in loaded} == {"train"}
    calls, _ = _capture_real_graphs(Path(report["path"]))
    assert len(calls) == 8
    common = None
    for index, row in enumerate(loaded):
        draws = np.array([(calls[index * 2 + rep][0] - row["z"]) / calls[index * 2 + rep][1] for rep in range(2)])
        if common is None:
            common = draws
        else:
            np.testing.assert_allclose(draws, common, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("problem", ["observed", "seed", "replicates", "dimension"])
def test_paired_noise_groups_reject_invalid_or_inconsistent_observations(core, problem):
    root, _case, mean, rows = core
    first = deepcopy(rows[0])
    second = deepcopy(first)
    first["noise_group_id"] = second["noise_group_id"] = "paired_probe"
    second["window_id"] += ":second"
    if problem == "observed":
        second["measurement_kind"] = "observed"
        second["noise_replicates"] = 1
        second.pop("noise_seed")
    elif problem == "seed":
        second["noise_seed"] += 1
    elif problem == "replicates":
        second["noise_replicates"] += 1
    else:
        second["z"] = mean[:-1].tolist()
        second["measurement_sigma"] = measurement_sigma(14, 20)[:-1].tolist()
    reviewed.write_jsonl(root / "invalid_pair_manifest.jsonl", [first, second])
    with pytest.raises(ValueError, match="noiseless_mean|paired noise group disagrees"):
        prepare_corpus(root / "invalid_pair_manifest.jsonl")


def test_node_breaker_real_three_case_smoke_keeps_fixed_meters_and_projected_covariance(tmp_path):
    from Transmission.ieee14_full_substation import operator_vector_for_layout

    report = reviewed.build_node_breaker_companion(tmp_path / "node_breaker", seed=20260917,
        parents_by_split={"train": 1}, attempts=12)
    rows = reviewed.read_jsonl(Path(report["path"]))
    assert report["coverage_shortfalls"] == 0
    assert len(rows) == 9
    assert {row["offline_audit"]["physical_severity"]["effect"] for row in rows} == {
        "healthy", "dangling_line_terminal", "bus_split"}
    identities = rows[0]["measurement_ids"]
    assert len(identities) == len(set(identities)) == 122
    grouped = {}
    for row in rows:
        audit = row["offline_audit"]
        assert row["measurement_ids"] == identities
        assert row["structural_zero_indices"] == [20, 34]
        sigma = np.asarray(row["sigma_z"])
        covariance = np.asarray(row["measurement_covariance"])
        np.testing.assert_allclose(covariance, np.diag(sigma**2), atol=1e-16, rtol=1e-12)
        assert sigma[20] == sigma[34] == 0
        assert row["measurements"][20] == row["measurements"][34] == 0
        assert audit["physical_solve_success"]
        assert not audit["admission_uses_wls_alarm"]
        assert audit["low_signal_events_retained"]
        np.testing.assert_allclose(operator_vector_for_layout(audit["source_telemetry"], audit["operator_layout"]),
                                   audit["z_exact"], atol=1e-14, rtol=0)
        np.testing.assert_array_equal(operator_vector_for_layout(row["substation_telemetry"], row["operator_layout"]),
                                      row["measurements"])
        assert row["substation_telemetry"]["measurement_kind"] == "observed"
        assert row["substation_telemetry"]["noise_draw_count"] == 1
        expected_power = measurement_sigma(14, 20, row["noise_profile"])[14]
        assert sigma[16] == pytest.approx(np.sqrt(2) * expected_power)
        assert sigma[30] == pytest.approx(np.sqrt(2) * expected_power)
        effect = audit["physical_severity"]["effect"]
        if effect == "healthy":
            assert audit["observable_strength"]["success"]
            # Finite closed-breaker impedance is omitted by the contracted
            # operator model: record its small nonzero mean discrepancy.
            assert 0 < audit["observable_strength"]["J_exact"] < (2. if row["noise_profile"] == "accuracy_002" else 1.)
        if effect in grouped:
            previous = grouped[effect]
            assert previous["offline_audit"]["noise_seed"] == audit["noise_seed"]
            np.testing.assert_array_equal(previous["offline_audit"]["z_exact"], audit["z_exact"])
            active = sigma > 0
            np.testing.assert_allclose(
                (np.asarray(row["measurements"])[active] - np.asarray(audit["z_exact"])[active]) / sigma[active],
                (np.asarray(previous["measurements"])[active] - np.asarray(previous["offline_audit"]["z_exact"])[active]) / np.asarray(previous["sigma_z"])[active],
                atol=1e-12, rtol=1e-12)
        grouped[effect] = row


def test_bundle_and_cli_forward_curriculum_stage_using_current_core_api(monkeypatch, tmp_path):
    calls = []
    # The thin CLI test needs no OpenDSS import or simulation; the real physical
    # helper smoke above covers this module's node/breaker adapter separately.
    fake = types.ModuleType("research.gnn_screen.practical_corpus")
    def generate(_output_dir, *, parents_by_split, seed, noise_replicates,
                 healthy_calibration_replicates, attempt_cap, scenario_profile, stage, noise_profile):
        calls.append({"stage": stage, "scenario_profile": scenario_profile, "noise_profile": noise_profile})
        return {}
    fake.generate_corpus = generate
    monkeypatch.setitem(sys.modules, "research.gnn_screen.practical_corpus", fake)
    monkeypatch.setattr(reviewed, "build_harmonic_companion", lambda *args, **kwargs: {})
    monkeypatch.setattr(reviewed, "build_measurement_chain_companion", lambda *args, **kwargs: {})
    monkeypatch.setattr(reviewed, "build_attribution_pairs", lambda *args, **kwargs: {})
    monkeypatch.setattr(reviewed, "build_energy_balanced_training_view", lambda *args, **kwargs: {})
    monkeypatch.setattr(reviewed, "build_accuracy_views", lambda *args, **kwargs: {})
    def admit(bundle):
        calls.append({"training_admission": str(bundle)})
        return {"profiles": {"baseline": {"outputs": {"filtered_manifest": "qualified.jsonl"}}}}
    monkeypatch.setattr(reviewed, "admit_training", admit)
    reviewed.main(["--output-dir", str(tmp_path / "bundle"), "--curriculum-stage", "full", "--skip-node-breaker"])
    assert calls[0]["scenario_profile"] == "ieee14_physical_hif_v1"
    assert calls[0]["stage"] == "full"
    assert calls[0]["noise_profile"] == "baseline"
    assert "training_admission" in calls[1]


def test_energy_balanced_view_equalizes_available_bins_without_altering_source_or_file_means(core, tmp_path):
    """Sampling-only fixture: bins are supplied offline inputs, not fit claims."""
    from collections import Counter

    root, _case, mean, original_rows = core
    source_rows = deepcopy(original_rows[:2])  # One training and one calibration control.
    write_json(root / "healthy_mean.json", mean)
    source_rows[0]["z"] = "healthy_mean.json"
    for bin_name, family, count in (("below_1", "hif", 4), ("1_to_9", "measurement", 2),
                                     ("9_to_25", "parameter", 3)):
        for ordinal in range(count):
            row = deepcopy(original_rows[0])
            window = f"parent_train:{bin_name}:{ordinal}"
            exact = mean.copy()
            exact[45] += .001 * (ordinal + 1)
            mean_path = f"energy_means/{bin_name}_{ordinal}.json"
            write_json(root / mean_path, exact)
            row.update(window_id=window, z=mean_path, families=[family])
            row["offline_metadata"]["noiseless_wls_audit"] = {
                "residual_visible_energy_bin": bin_name, "sampling_unit_fixture": True,
            }
            source_rows.append(row)
    unavailable = deepcopy(original_rows[0])
    unavailable.update(window_id="parent_train:unavailable", families=["topology"])
    unavailable["offline_metadata"]["noiseless_wls_audit"] = {
        "success": False, "residual_visible_energy_bin": "unavailable",
    }
    source_rows.append(unavailable)
    heldout = deepcopy(original_rows[1])
    heldout.update(window_id="parent_calibration:heldout_fault", families=["topology"])
    heldout["offline_metadata"]["noiseless_wls_audit"] = {"residual_visible_energy_bin": "above_25"}
    source_rows.append(heldout)
    reviewed.write_jsonl(root / "manifest.jsonl", source_rows)
    before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*.json*")}

    report = reviewed.build_energy_balanced_training_view(root, tmp_path / "energy_view", seed=155)
    selected = load_manifest(report["path"])
    assert report["rows"] == len(selected) == 7
    assert report["source_fault_counts_by_bin"] == {"below_1": 4, "1_to_9": 2, "9_to_25": 3, "above_25": 0}
    assert report["selected_faults_per_available_bin"] == 2
    assert report["missing_bins"] == ["above_25"]
    assert not report["all_four_bins_populated"]
    assert report["healthy_rows_retained"] == 1
    assert report["unavailable_energy_rows_retained_in_full_source"] == 1
    assert report["selected_family_counts"] == {"hif": 2, "measurement": 2, "parameter": 2}
    assert not report["main_population_modified"]
    assert not report["automatic_training_enabled"]
    assert {row["split"] for row in selected} == {"train"}
    chosen_bins = Counter(row["offline_metadata"]["noiseless_wls_audit"]["residual_visible_energy_bin"]
                          for row in selected if row["families"])
    assert chosen_bins == {"below_1": 2, "1_to_9": 2, "9_to_25": 2}
    sources = {row["window_id"]: row for row in source_rows}
    for row in selected:
        source = sources[row["window_id"]]
        expected_mean = json.loads((root / source["z"]).read_text()) if isinstance(source["z"], str) else source["z"]
        np.testing.assert_array_equal(row["z"], expected_mean)
        np.testing.assert_array_equal(row["measurement_sigma"], measurement_sigma(14, 20))
        assert row["families"] == source["families"]
        assert row["parent_id"] == source["parent_id"]
        assert row["noise_seed"] == source["noise_seed"]
        assert row["noise_replicates"] == source["noise_replicates"]
        assert row["measurement_kind"] == source["measurement_kind"] == "noiseless_mean"
        assert (Path(report["path"]).parent / row["offline_metadata"]["physical_audit_path"]).is_file()
    assert before == {path.relative_to(root): path.read_bytes() for path in root.rglob("*.json*")}
