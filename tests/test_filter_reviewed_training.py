"""Training selection uses fixed noisy observations; no neural model is trained."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from mcp_server.matpower_server import _load_python_case
from research import filter_reviewed_training as filtering
from research.gnn_screen.dataset import content_hash, jsonable, load_manifest, prepare_corpus
from research.gnn_screen.feature_schema import MEASUREMENT_CONVENTION
from research.gnn_screen.graph_builder import build_graph
from research.gnn_screen.train import validate_reviewed_training_admission
from research.gnn_screen.wls_features import configured_case, default_measurement_sigma, state_measurements_and_jacobian


def _sensor_fixture():
    full = _load_python_case("case14")
    case = configured_case(full)
    mean, _ = state_measurements_and_jacobian(case, np.deg2rad(full["bus"][:, 8]), full["bus"][:, 7])
    sigma = default_measurement_sigma(14, 20)
    return jsonable(case), mean, sigma


def _row(case, z, sigma, *, name, families, parent="parent-train", split="train", **extra):
    return {"case": copy.deepcopy(case), "z": np.asarray(z).tolist(), "measurement_sigma": sigma.tolist(),
        "parent_id": parent, "window_id": f"{parent}:{name}", "split": split, "families": families,
        "severity": "unit_fixture", "measurement_kind": "observed", "noise_replicates": 1,
        "measurement_convention": MEASUREMENT_CONVENTION,
        "offline_metadata": {"scenario_profile": "reviewed_v1", "private_label_sentinel": "never_expert_input"},
        **extra}


def _write(path, rows):
    path.write_text("".join(json.dumps(jsonable(row)) + "\n" for row in rows))
    return path


def test_materializer_matches_actual_dataset_stream_and_observed_rows_never_redraw(tmp_path):
    case, mean, sigma = _sensor_fixture()
    source = _row(case, mean, sigma, name="mean", families=[], measurement_kind="noiseless_mean",
                  noise_replicates=3, noise_seed=42, noise_group_id="paired-source")
    path = _write(tmp_path / "source.jsonl", [source])
    first = []
    def capture(case, observed, **kwargs):
        first.append(observed.copy())
        return build_graph(case, observed, **kwargs)
    prepare_corpus(path, graph_builder=capture)
    windows = filtering.materialize_training_windows(load_manifest(path)[0])
    assert len(windows) == len(first) == 3
    for index, window in enumerate(windows):
        np.testing.assert_array_equal(window["z"], first[index])
        assert window["measurement_kind"] == "observed" and window["noise_replicates"] == 1
        assert "noise_seed" not in window and "noise_group_id" not in window
        lineage = window["offline_metadata"]["admission_source"]
        assert lineage["source_noise_group_id"] == "paired-source" and lineage["replicate"] == index
    fixed_path = _write(tmp_path / "fixed.jsonl", windows)
    second = []
    def capture_fixed(case, observed, **kwargs):
        second.append(observed.copy())
        return build_graph(case, observed, **kwargs)
    prepare_corpus(fixed_path, graph_builder=capture_fixed)
    for before, after in zip(first, second):
        np.testing.assert_array_equal(before, after)


def test_real_wls_training_filter_preserves_heldout_and_excludes_quiet_faults(tmp_path, monkeypatch):
    case, mean, sigma = _sensor_fixture()
    bad = mean.copy(); bad[20] += .25
    healthy = _row(case, mean, sigma, name="healthy", families=[])
    meter = _row(case, bad, sigma, name="meter", families=["measurement"])
    quiet = _row(case, mean, sigma, name="quiet-physical", families=["hif"])
    heldout = _row(case, mean, sigma, name="weak", families=["hif"], parent="parent-test", split="test",
                    measurement_kind="noiseless_mean", noise_seed=17, noise_replicates=4, noise_group_id="weak-eval")
    source = _write(tmp_path / "source.jsonl", [healthy, meter, quiet, heldout])
    source_bytes = source.read_bytes()
    calls = []
    def context_builder(path, row, **kwargs):
        return {"sigma_z": row["measurement_sigma"]}, {"unit_context": True}
    def probe(model, observed, std, *, observable_metadata, max_actions):
        calls.append((copy.deepcopy(model), list(observed), copy.deepcopy(observable_metadata)))
        assert set(observable_metadata) == {"sigma_z"} and max_actions == 40
        assert not ({"families", "offline_metadata", "label", "z_clean"} & set(model))
        detection = filtering.wls_detection(model, observed, std)
        alarm = bool(detection["alarm"])
        action = {"tool": "get_measurement_context" if alarm else "finalize_diagnosis", "arguments": {"state_id": "active"}}
        event = {"process_evidence_valid": True, "action_executed": True, "execution_success": True,
                 "preferred_action": action, "ordered_actions": [action], "policy_observation": {"active_state_id": "active"}}
        return {"initial_wls": detection, "fault_actionable": alarm, "healthy_completion_valid": not alarm,
                "actionable_event": event if alarm else None, "events": [event]}
    # Admission logic is independent of renderer schemas; actual canonical export
    # is exercised separately below using its real converter.
    monkeypatch.setattr(filtering, "_export_action", lambda event, fixed: (
        {"example_id": fixed["window_id"], "preferred_action": event["preferred_action"]},
        {"messages": [{"role": "assistant", "content": "unit converter stub"}]}))
    report = filtering.filter_training_manifest(source, tmp_path / "filtered", probe=probe, context_builder=context_builder)
    assert report["training_windows_kept"] == 2 and report["training_windows_excluded"] == 1
    assert report["held_out_source_windows_unchanged"] == 1 and len(calls) == 2
    assert not report["complete_episode_success_claimed"] and not report["training_performed"]
    loaded = load_manifest(report["outputs"]["filtered_manifest"])
    actual_test = next(row for row in loaded if row["split"] == "test")
    for key in ("case", "z", "measurement_sigma", "parent_id", "window_id", "families", "measurement_kind", "noise_seed", "noise_group_id", "noise_replicates"):
        assert actual_test[key] == heldout[key]
    admission = validate_reviewed_training_admission(report["outputs"]["filtered_manifest"])
    assert admission["reviewed_fault_windows"] == admission["reviewed_healthy_windows"] == 1
    exclusions = load_manifest(report["outputs"]["excluded_training_manifest"])
    assert exclusions[0]["families"] == ["hif"] and exclusions[0]["split"] == "train"
    assert exclusions[0]["offline_metadata"]["training_exclusion"]["reason"] == "wls_quiet_fault"
    assert source.read_bytes() == source_bytes


def test_mixed_global_alarm_does_not_hide_wls_quiet_physical_component():
    case, mean, sigma = _sensor_fixture()
    healthy = _row(case, mean, sigma, name="healthy", families=[], measurement_kind="noiseless_mean")
    physical = _row(case, mean, sigma, name="weak-hif", families=["hif"], measurement_kind="noiseless_mean")
    mixed_z = mean.copy(); mixed_z[20] += .25
    mixed = _row(case, mixed_z, sigma, name="mixed", families=["measurement", "hif"], measurement_kind="noiseless_mean")
    mixed["offline_metadata"]["component_core_audit"] = {"source_window_id": physical["window_id"]}
    fixed = copy.deepcopy(mixed); fixed["measurement_kind"] = "observed"
    assert filtering.wls_detection(case, mixed_z, sigma)["alarm"]
    result = filtering.mixed_component_checks(mixed, fixed, {physical["window_id"]: physical}, {healthy["parent_id"]: healthy})
    assert result["meter_only"]["alarm"] and not result["physical_only"]["alarm"]
    assert not result["passed"]


def test_mixed_counterfactuals_keep_identical_selected_noise(monkeypatch):
    case, mean, sigma = _sensor_fixture()
    physical_z = mean.copy(); physical_z[5] += .05
    mixed_z = physical_z.copy(); mixed_z[20] += .25
    noise = np.linspace(-.001, .001, len(mean))
    healthy = _row(case, mean, sigma, name="healthy", families=[], measurement_kind="noiseless_mean")
    physical = _row(case, physical_z, sigma, name="physical", families=["parameter"], measurement_kind="noiseless_mean")
    mixed = _row(case, mixed_z, sigma, name="mixed", families=["measurement", "parameter"], measurement_kind="noiseless_mean")
    mixed["offline_metadata"]["component_core_audit"] = {"source_window_id": physical["window_id"]}
    fixed = copy.deepcopy(mixed); fixed["z"] = (mixed_z + noise).tolist()
    seen = []
    monkeypatch.setattr(filtering, "wls_detection", lambda case, z, sigma: seen.append(np.asarray(z)) or {"success": True, "alarm": True})
    result = filtering.mixed_component_checks(mixed, fixed, {physical["window_id"]: physical}, {healthy["parent_id"]: healthy})
    assert result["passed"] and result["same_noise_as_mixed"]
    np.testing.assert_allclose(seen[0] - physical_z, noise, atol=1e-15)
    np.testing.assert_allclose(seen[1] - mean - (mixed_z - physical_z), noise, atol=1e-15)


def test_canonical_prefix_export_retains_parent_identity_and_scope():
    from psse_env.dagger.test_sft_export import _example
    source = _example()
    event = {"policy_observation": source["policy_observation"], "preferred_action": source["preferred_action"],
             "ordered_actions": [source["preferred_action"]]}
    case, mean, sigma = _sensor_fixture()
    fixed = _row(case, mean, sigma, name="noise0", families=[], parent="physical-parent-a")
    raw, chat = filtering._export_action(event, fixed)
    assert chat["root_scenario_id"] == fixed["parent_id"]
    assert chat["scenario_id"] == fixed["window_id"]
    call = chat["messages"][-1]["tool_calls"][0]["function"]
    assert call["name"] == "wls_from_path" and isinstance(call["arguments"], dict)
    assert chat["metadata"]["scope"] == "executed_expert_prefix_only"
    assert not chat["metadata"]["complete_repair_verified"]
    assert chat["metadata"]["measurement_sha256"] == content_hash(fixed["z"])


def test_filter_rejects_drifting_paired_noise_groups_before_output(tmp_path):
    case, mean, sigma = _sensor_fixture()
    one = _row(case, mean, sigma, name="arm1", families=[], measurement_kind="noiseless_mean",
               noise_seed=41, noise_group_id="same-pair", noise_replicates=2)
    two = copy.deepcopy(one)
    two.update(window_id="parent-train:arm2", noise_seed=42)
    source = _write(tmp_path / "group.jsonl", [one, two])
    with pytest.raises(ValueError, match="paired noise group"):
        filtering.filter_training_manifest(source, tmp_path / "unwritten", probe=lambda *a, **k: None,
                                            context_builder=lambda *a, **k: None)
    assert not (tmp_path / "unwritten").exists()


def test_mixed_counterfactual_does_not_add_noise_to_already_observed_reference():
    case, mean, sigma = _sensor_fixture()
    healthy = _row(case, mean, sigma, name="healthy", families=[], measurement_kind="noiseless_mean")
    physical = _row(case, mean, sigma, name="physical", families=["parameter"], measurement_kind="observed")
    mixed = _row(case, mean, sigma, name="mixed", families=["parameter", "measurement"], measurement_kind="noiseless_mean")
    mixed["offline_metadata"]["component_core_audit"] = {"source_window_id": physical["window_id"]}
    result = filtering.mixed_component_checks(mixed, mixed, {physical["window_id"]: physical}, {healthy["parent_id"]: healthy})
    assert not result["passed"] and result["reason"] == "mixed_counterfactual_requires_noiseless_references"


@pytest.fixture(scope="module")
def companion_bundle(tmp_path_factory):
    """Real small PF/OPF companions, independent of the published pilot files."""
    from pypower.api import case14, ppoption, runpf
    from threadpoolctl import threadpool_limits
    from Transmission.generate_measurements import compute_measurements_pu
    from research import reviewed_fault_scenarios as reviewed
    from research.gnn_screen.dataset import write_json
    bundle = tmp_path_factory.mktemp("companion_admission")
    core = bundle / "baseline" / "core"
    with threadpool_limits(limits=1):
        full, success = runpf(case14(), ppoption(VERBOSE=0, OUT_ALL=0, PF_TOL=1e-11))
        assert success
        mean, sigma = compute_measurements_pu(full), default_measurement_sigma(14, 20)
        rows = []
        for split in ("train", "calibration"):
            parent = f"companion-{split}"
            case_path = f"parents/{parent}/configured.json"
            audit_path = f"parents/{parent}/physical_audit.json"
            write_json(core / case_path, configured_case(full))
            write_json(core / f"parents/{parent}/model/source_case.json", full)
            write_json(core / audit_path, {"fixture": "public balanced reference"})
            row = _row(configured_case(full), mean, sigma, name="healthy", families=[], parent=parent, split=split,
                       measurement_kind="noiseless_mean", noise_seed=78, noise_replicates=1)
            row["case"] = case_path
            row["offline_metadata"]["physical_audit_path"] = audit_path
            rows.append(row)
        _write(core / "manifest.jsonl", rows)
        harmonic = reviewed.build_harmonic_companion(core, bundle, seed=813, noise_profiles=("baseline",))
        breaker = reviewed.build_node_breaker_companion(bundle, seed=814,
            parents_by_split={"train": 1, "calibration": 1}, noise_profiles=("baseline",), attempts=4)
    assert harmonic["rows"] == 8 and breaker["coverage_shortfalls"] == 0
    return bundle


def test_companion_filter_uses_only_observed_sensors_and_preserves_heldout(companion_bundle, tmp_path):
    from Transmission.ieee14_full_topology import build_full_topology
    from Transmission.ieee14_full_substation import status_labels
    from psse_env.dagger.test_sft_export import _example
    from research.gnn_screen.feature_schema import FAMILY_NAMES
    source_rows = {name: filtering.read_jsonl(companion_bundle / f"{name}_scenarios.jsonl")
                   for name in ("harmonic", "node_breaker")}
    original_bytes = {name: (companion_bundle / f"{name}_scenarios.jsonl").read_bytes() for name in source_rows}
    by_measurement = {content_hash(row["measurements"]): row for rows in source_rows.values() for row in rows}
    seen = []
    def probe(case, observed, sigma, *, observable_metadata, max_actions):
        source = by_measurement[content_hash(observed)]
        assert source["split"] == "train" and max_actions == 40
        assert "offline_audit" not in observable_metadata and "families" not in observable_metadata
        detection = filtering.wls_detection(case, observed, sigma,
            exact_rows=observable_metadata.get("structural_zero_indices", []))
        if source["schema"].startswith("harmonic"):
            sensors = observable_metadata["harmonic_measurements"]
            expected = {(int(h), int(sensor["bus_1based"])): sensor for h, values in source["harmonic_phasors"].items() for sensor in values}
            for sensor in sensors:
                reference = expected[sensor["h"], sensor["bus"]]
                assert [sensor["V_real"], sensor["V_imag"]] == reference["V_complex_noisy"]
                assert sensor["sigma"] == reference["sigma"]
                assert "V_complex_true" not in sensor and "source_bus" not in sensor
        else:
            zeros = source["structural_zero_indices"]
            assert zeros and all(sigma[index] == 0 for index in zeros)
            assert observable_metadata["structural_zero_indices"] == zeros
            assert observable_metadata["reported_breaker_status"] == status_labels(build_full_topology())
            assert observable_metadata["substation_telemetry"] == source["substation_telemetry"]
            fault = source["offline_audit"]["physical_severity"]["breaker"]
            if fault:
                expected = "closed" if fault["reported_closed"] else "open"
                assert observable_metadata["reported_breaker_status"][fault["cb_name"]] == expected
                assert fault["reported_closed"] != fault["true_closed"]
            assert "true_breaker_status" not in observable_metadata
        seen.append(source["schema"])
        example = _example()
        event = {"policy_observation": example["policy_observation"], "preferred_action": example["preferred_action"],
                 "ordered_actions": [example["preferred_action"]], "process_evidence_valid": True,
                 "action_executed": True, "execution_success": True}
        return {"initial_wls": detection, "fault_actionable": bool(detection["alarm"]),
                "healthy_completion_valid": not detection["alarm"], "actionable_event": event, "events": [event]}
    out = tmp_path / "filtered_companions"
    reports = filtering.filter_companion_training(companion_bundle, out, probe=probe)
    assert "harmonic" not in FAMILY_NAMES and seen
    for name, original in source_rows.items():
        held = filtering.read_jsonl(out / name / "held_out_scenarios.jsonl")
        assert held == [row for row in original if row["split"] != "train"]
        assert (companion_bundle / f"{name}_scenarios.jsonl").read_bytes() == original_bytes[name]
        report = reports[name]
        assert report["not_a_gnn_five_family_manifest"]
        assert report["training_rows_kept"] + report["training_rows_excluded"] == sum(row["split"] == "train" for row in original)
        for chat in filtering.read_jsonl(out / name / "expert_prefixes.chat_sft.jsonl"):
            assert chat["metadata"]["scope"] == "executed_expert_prefix_only"
            assert not chat["metadata"]["complete_repair_verified"]
    assert any(name.startswith("harmonic") for name in seen) and any("node_breaker" in name for name in seen)


def test_breaker_detection_keeps_exact_rows_and_matches_independent_constrained_wls(companion_bundle):
    from tools.lagrangian_port import lagrangian_m_singlephase_details
    row = filtering.read_jsonl(companion_bundle / "node_breaker_scenarios.jsonl")[0]
    exact = row["structural_zero_indices"]
    assert exact and all(row["sigma_z"][index] == 0 for index in exact)
    case = copy.deepcopy(row["case"])
    for field in ("bus", "branch", "gen"):
        case[field] = np.asarray(case[field], dtype=float)
    case["bus"][:, 7], case["bus"][:, 8] = 1., 0.
    independent = lagrangian_m_singlephase_details(np.asarray(row["measurements"]), case, 0, case["bus"],
        measurement_sigma=np.asarray(row["sigma_z"]), exact_measurement_indices=exact, max_it=30, tol=1e-8)
    actual = filtering.wls_detection(row["case"], row["measurements"], row["sigma_z"], exact_rows=exact)
    assert actual["success"] and independent["success"]
    assert actual["J"] == pytest.approx(independent["wls_objective"], rel=1e-12)
    assert actual["dof"] == independent["dof"]
    assert set(independent["exact_measurement_indices"].tolist()) == set(exact)
    # The unconstrained positive-diagonal path must not silently epsilon-fill.
    assert not filtering.wls_detection(row["case"], row["measurements"], row["sigma_z"])["success"]


def test_real_observable_expert_safely_finalizes_noisy_breaker_control(companion_bundle):
    from Transmission.ieee14_full_topology import build_full_topology
    from Transmission.ieee14_full_substation import operator_noise_for_layout, status_labels
    from research.reviewed_expert_admission import probe_expert_action
    row = next(row for row in filtering.read_jsonl(companion_bundle / "node_breaker_scenarios.jsonl")
               if row["split"] == "train" and not row["families"])
    metadata = {key: copy.deepcopy(row[key]) for key in (
        "substation_telemetry", "operator_layout", "structural_zero_indices", "measurement_ids", "sigma_z")}
    metadata.update(reported_breaker_status=status_labels(build_full_topology()),
        topology_model_id=row["substation_telemetry"]["model_id"],
        topology_model_fingerprint=row["substation_telemetry"]["model_fingerprint"],
        operator_noise=operator_noise_for_layout(row["substation_telemetry"], row["operator_layout"]))
    before = copy.deepcopy(metadata)
    receipt = probe_expert_action(row["case"], row["measurements"], row["sigma_z"],
                                 observable_metadata=metadata, max_actions=40)
    assert receipt["initial_wls"]["success"] and not receipt["initial_wls"]["alarm"]
    assert receipt["healthy_completion_valid"] and not receipt["fault_actionable"]
    assert receipt["events"][-1]["preferred_action"]["tool"] == "finalize_diagnosis"
    assert not receipt["full_repair_validated"] and metadata == before
