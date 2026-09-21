"""Physical practical-cohort contracts, independently of any learned score."""
from __future__ import annotations

import copy
import json

import numpy as np
import pytest

pytest.importorskip("opendssdirect")
from research.gnn_screen.dataset import content_hash, load_manifest
from research.gnn_screen.practical_corpus import (
    MAIN_MIN_DISTANCE, SLOTS, audit_constant_pq, cohort_decision, generate_corpus, meter_overlay,
    perturb_physical_case, physical_hif_slots, reviewed_slots,
)
from research.gnn_screen.scenario_policy import paired_visibility
from research.gnn_screen.wls_features import default_measurement_sigma
from psse_env.systems import resolve_system


@pytest.fixture(scope="module")
def practical(tmp_path_factory):
    path = tmp_path_factory.mktemp("practical_physics") / "corpus"
    report = generate_corpus(path, parents_by_split={s: 1 for s in ("train", "validation", "calibration", "test")},
        seed=2026091823, noise_replicates=2, healthy_calibration_replicates=3,
        healthy_replicates_by_split={"validation": 4, "test": 5}, attempt_cap=16)
    return path, report, load_manifest(path / "manifest.jsonl"), load_manifest(path / "boundary_manifest.jsonl")


def test_core_labels_grouping_identical_reported_model_and_audited_physics(practical):
    path, report, rows, boundary = practical
    assert report["intended_main_slots_per_noncalibration_parent"] == 18
    assert not report["admission_uses_wls_alarm"]
    assert not report["admission_uses_noisy_or_learned_scores"]
    assert report["healthy_max_balanced_equation_error_pu"] < 1e-8
    assert len({r["parent_id"] for r in rows}) == 4
    for parent in {r["parent_id"] for r in rows}:
        group = [r for r in rows + boundary if r["parent_id"] == parent]
        assert len({r["split"] for r in group}) == 1
        assert len({content_hash(r["case"]) for r in group}) == 1
        assert len({r["offline_metadata"]["reported_case_hash"] for r in group}) == 1
    for row in rows:
        meta = row["offline_metadata"]
        assert meta["cohort"] == "main"
        assert meta["scenario_policy"]["admitted_main"]
        assert len(row["z"]) == 122 and len(row["case"]["bus"]) == 14 and len(row["case"]["branch"]) == 20
        physics = json.loads((path / meta["physical_audit_path"]).read_text())
        assert physics["diagnostics"]["load_pq_audit"]["passed"]
        assert physics["external_kcl_max_mismatch_pu"] < 1e-7
        assert physics["diagnostics"]["minimum_phase_voltage_pu"] > 0
        if row["families"]:
            assert meta["paired_visibility"]["mean_separation_d"] >= MAIN_MIN_DISTANCE
            assert meta["scenario_policy"]["physical_cohort"] == "core"
        if "unbalance" in row["families"]:
            assert meta["maximum_voltage_negative_positive_ratio"] >= .01
        if "hif" in row["families"]:
            assert meta["diagnostic_currents"]["hif_differential_current_sigma"] >= 6
        if row["split"] == "calibration":
            assert not row["families"] and row["noise_replicates"] == 3


def test_parameter_and_status_errors_change_physics_not_reported_graph(practical):
    path, _, rows, _ = practical
    for row in rows:
        if not set(row["families"]) & {"parameter", "topology"}:
            continue
        meta = row["offline_metadata"]
        physics = json.loads((path / meta["physical_audit_path"]).read_text())
        perturbation = physics["physical_perturbation"]
        actual = json.loads((path / physics["actual_physical_model_path"] / "source_case.json").read_text())
        index = perturbation["branch_row0"]
        np.testing.assert_array_equal(np.asarray(actual["branch"])[:, :2], np.asarray(row["case"]["branch"])[:, :2])
        assert physics["variant_source_opf"]["success"]
        if "parameter" in row["families"]:
            for name, factor in perturbation["physical_factors"].items():
                assert .1 <= factor <= .5 or 2 <= factor <= 5
                column = {"R": 2, "X": 3}[name]
                assert actual["branch"][index][column] / row["case"]["branch"][index][column] == pytest.approx(factor)
        else:
            assert actual["branch"][index][10] != row["case"]["branch"][index][10]
            assert perturbation["physical_graph_connected"]
            assert perturbation["scope"] == "branch_status_only_not_full_node_breaker_topology"


def test_mixed_meter_cannot_promote_subthreshold_physical_component(practical):
    path, _, rows, _ = practical
    by_window = {row["window_id"]: row for row in rows}
    mixed = [row for row in rows if len(row["families"]) > 1]
    assert mixed
    sigma = default_measurement_sigma(14, 20)
    for row in mixed:
        component = row["offline_metadata"]["component_core_audit"]
        assert component["individually_accepted_before_meter_overlay"]
        assert component["scenario_policy"]["admitted_main"]
        assert component["paired_visibility"]["mean_separation_d"] >= MAIN_MIN_DISTANCE
        source = by_window[component["source_window_id"]]
        assert len(source["families"]) == 1 and source["families"][0] in row["families"]
        expected = np.asarray(source["z"]).copy()
        meter = row["offline_metadata"]["settings"]["measurement"]
        indices = meter["channel_indices0"]
        expected[indices] += sigma[indices] * np.asarray(meter["sigma_multiples"])
        np.testing.assert_allclose(row["z"], expected, atol=1e-12, rtol=0)


def test_boundary_keeps_fault_labels_and_sft_hif_challenge(practical):
    path, _, main, boundary = practical
    assert boundary and all(row["families"] for row in boundary)
    assert not any(row["split"] == "calibration" for row in boundary)
    challenges = [r for r in boundary if r["offline_metadata"]["cohort"] == "hif_sft_challenge"]
    assert len(challenges) == 3
    for row in challenges:
        assert row["families"] == ["hif"]
        assert 20 <= row["offline_metadata"]["settings"]["hif"]["resistance_pu"] <= 200
    ledger = [json.loads(line) for line in (path / "proposal_ledger.jsonl").read_text().splitlines()]
    below = [row for row in ledger if row.get("outcome") == "valid_positive_below_main_criteria"]
    assert below and all(row["families"] for row in below)
    assert all(row["scenario_policy"]["rejection_reasons"] for row in below)
    assert not set(r["window_id"] for r in main) & set(r["window_id"] for r in boundary)


def test_legacy_and_reviewed_slot_tables_keep_pu_options_only_physical_drops_them():
    # The shared SLOTS tuple is the legacy_v1 contract and is unchanged (pu ranges intact).
    legacy = {name: options for name, _, options in SLOTS}
    assert legacy["hif_5to10"] == {"resistance_range": (5., 10.)}
    assert legacy["measurement_hif"] == {"resistance_range": (5., 40.), "meter_count": 1}
    assert len(SLOTS) == 17
    # reviewed_v1 still forwards the mixed slot's legacy option untouched.
    reviewed = {name: options for name, _, options in reviewed_slots(stage="full", split="test")}
    assert reviewed["measurement_hif"] == {"resistance_range": (5., 40.), "meter_count": 1}
    assert reviewed["hif_weak_evaluation"]["hif_band"] == "weak"
    # Only the physical profile strips the dead pu option and adds the detection-limit cohort.
    physical = {name: options for name, _, options in physical_hif_slots(stage="full", split="test")}
    assert physical["measurement_hif"] == {"meter_count": 1}
    assert not any("resistance_range" in options for options in physical.values())
    assert not any("hif_band" in options for options in physical.values())
    assert physical["hif_69kv_detection_limit_0"] == physical["hif_69kv_detection_limit_1"] == {
        "hif_ohm_band": "detection_limit", "voltage_kv": 69.0, "destination": "hif_detection_limit_evaluation"}
    assert "hif_69kv_detection_limit_0" not in {name for name, _, _ in physical_hif_slots(stage="full", split="train")}


def test_meter_magnitude_count_and_same_channel_contract():
    z = np.zeros(122)
    rng = np.random.default_rng(44)
    for count in (1, 2, 3, 4, 5, None):
        changed, audit = meter_overlay(z, rng, count=count)
        assert 1 <= audit["meter_count"] <= 5
        assert np.count_nonzero(changed) == audit["meter_count"]
        assert 10 <= audit["minimum_absolute_sigma_multiple"] <= audit["maximum_absolute_sigma_multiple"] <= 15
        assert len(set(audit["channel_indices0"])) == audit["meter_count"]


def test_constant_pq_audit_rejects_voltage_envelope_fallback():
    registry = {"loads": [{"element": "Load.test", "bus": 2, "phase": 1, "kw": 1000., "kvar": 200.}]}
    telemetry = {"load_powers": [{"element": "Load.test", "power_into_element_pu": {"real": .01, "imag": .002}}]}
    assert audit_constant_pq(telemetry, registry, 100)["passed"]
    telemetry["load_powers"][0]["power_into_element_pu"]["real"] *= .5
    with pytest.raises(RuntimeError, match="voltage-envelope fallback"):
        audit_constant_pq(telemetry, registry, 100)


def test_admission_is_independent_of_wls_alarm_or_fit_statistics():
    visibility = paired_visibility(np.zeros(122), np.r_[6., np.zeros(121)], np.ones(122))
    physics = {"diagnostics": {"maximum_voltage_negative_positive_ratio": .02}}
    accepted = cohort_decision(["unbalance"], physics, visibility)
    assert accepted["admitted_main"]
    for alarm, objective in ((False, 0.), (True, 1e9)):
        altered = copy.deepcopy(physics)
        altered.update(wls_alarm=alarm, wls_objective=objective)
        assert cohort_decision(["unbalance"], altered, visibility) == accepted


def test_physical_mutation_does_not_modify_parent_and_reports_actual_over_reference():
    parent = resolve_system("case14").load_case()
    before = content_hash(parent)
    for components in (("R",), ("X",), ("R", "X")):
        actual, audit = perturb_physical_case(parent, np.random.default_rng(41), family="parameter", components=components)
        assert content_hash(parent) == before
        for key, factor in audit["physical_factors"].items():
            assert audit["actual_values"][key] / audit["reported_values"][key] == pytest.approx(factor)
    with pytest.raises(ValueError):
        perturb_physical_case(parent, np.random.default_rng(0), family="parameter", components=("Z",))


def test_x_only_parameter_errors_exclude_transformers_and_zero_r_lines():
    parent = resolve_system("case14").load_case()
    rng = np.random.default_rng(71)
    for _ in range(50):
        _, audit = perturb_physical_case(parent, rng, family="parameter", components=("X",))
        row = parent["branch"][audit["branch_row0"]]
        assert row[10] and row[8] == 0 and row[2] > 1e-9 and row[3] > 1e-9
    parent["branch"][parent["branch"][:, 8] == 0, 10] = 0
    with pytest.raises(ValueError, match="eligible active line"):
        perturb_physical_case(parent, rng, family="parameter", components=("X",))
