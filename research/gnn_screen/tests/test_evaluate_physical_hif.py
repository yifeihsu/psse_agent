"""Frozen-sweep replay identity, pairing, availability and type-only sensitivity."""
from copy import deepcopy

import numpy as np
import pytest

from research.gnn_screen.evaluate_physical_hif import (
    DETECTORS, attach_paired_controls, legacy_type_graph, numeric_hash,
    reconstruct_observation, saved_wls_fields, summarize,
)


@pytest.mark.parametrize("power_sigma", [.01, .005, .002])
def test_reconstructs_exact_saved_noise_without_any_new_random_draw(power_sigma, monkeypatch):
    mean = np.array([1.043, -.173, .829], dtype=np.float64)
    unit = np.array([.3486, -1.5387, .0042], dtype=np.float64)
    sigma = np.array([.001, power_sigma, power_sigma])
    expected = mean + unit * sigma
    row = {"case_id": "fault", "noise_group_id": "shared-group", "noise_profile": "selected",
           "observed_sha256": numeric_hash(expected)}
    physical = {"fault": {"physical_success": True, "mean_measurement_vector": mean.tolist()}}
    groups = {"shared-group": {"unit_noise": unit.tolist()}}

    def forbidden_rng(*args, **kwargs):
        raise AssertionError("Frozen sweep replay must not redraw noise")

    monkeypatch.setattr(np.random, "default_rng", forbidden_rng)
    actual, actual_sigma = reconstruct_observation(row, physical, groups, {"selected": sigma.tolist()})
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual_sigma, sigma)
    assert numeric_hash(actual) == row["observed_sha256"]


@pytest.mark.parametrize("changed", ["mean", "noise", "sigma", "hash"])
def test_reconstruction_rejects_any_changed_observation_component(changed):
    row = {"case_id": "fault", "noise_group_id": "g", "noise_profile": "p",
           "observed_sha256": numeric_hash([1.01, 2.02])}
    physical = {"fault": {"physical_success": True, "mean_measurement_vector": [1., 2.]}}
    groups = {"g": {"unit_noise": [1., 2.]}}
    sigmas = {"p": [.01, .01]}
    if changed == "mean":
        physical["fault"]["mean_measurement_vector"][0] += .1
    elif changed == "noise":
        groups["g"]["unit_noise"][0] += 1
    elif changed == "sigma":
        sigmas["p"][0] = .02
    else:
        row["observed_sha256"] = "incorrect"
    with pytest.raises(ValueError, match="reconstruction mismatch"):
        reconstruct_observation(row, physical, groups, sigmas)


def test_failed_physical_case_never_produces_neural_observation():
    with pytest.raises(ValueError, match="failed physical case"):
        reconstruct_observation({"case_id": "failed"}, {"failed": {"physical_success": False}}, {}, {})


@pytest.mark.parametrize("matched_threshold,expected", [(1.0, True), (1.05, False), (1.1, False)])
def test_saved_successful_wls_stays_available_independently_of_gnn(matched_threshold, expected):
    saved = {"success": True, "J": 50., "chi_square_threshold": 100., "max_normalized_residual": 4.2,
             "chi_square_alarm": False, "normalized_residual_alarm": True, "alarm": True}
    fields = saved_wls_fields(saved, {"matched_wls_threshold": matched_threshold}, local_threshold=4.)
    assert fields == {"wls_chi_square": False, "wls_normalized": True, "wls_dual": True,
                      "wls_source_calibrated": expected}
    # Source-calibrated comparison is strict >, even when the ordinary WLS
    # comparator independently alarms using its inclusive protection thresholds.
    unavailable_gnn = _row("hif", "g", decisions=None, phase_score=None)
    unavailable_gnn.update(fields)
    summary = summarize([unavailable_gnn], ["kind"])[0]
    assert summary["gnn_unavailable"] == 1
    assert summary["gnn_phase_available"] == 0
    assert summary["wls_dual_available"] == summary["wls_dual_count"] == 1


def test_saved_failed_wls_does_not_invent_negative_decisions():
    fields = saved_wls_fields({"success": False}, {}, local_threshold=4.)
    assert fields == {"wls_chi_square": None, "wls_normalized": None, "wls_dual": None,
                      "wls_source_calibrated": None}


def _row(kind, group, *, case_id=None, decisions=False, profile="baseline", phase_score=.2):
    row = {"kind": kind, "case_id": case_id or kind, "parent_id": "parent0",
           "noise_group_id": group, "noise_profile": profile, "phase_score": phase_score}
    row.update({name: decisions for name in DETECTORS})
    return row


def test_pairs_many_fault_strengths_to_one_healthy_control_without_duplication():
    healthy = _row("healthy", "group0", decisions=True)
    no_fault = _row("no_fault", "group0", decisions=False)
    fault1 = _row("hif", "group0", case_id="fault-100ohm", decisions=True)
    fault2 = _row("hif", "group0", case_id="fault-1000ohm", decisions=False)
    rows = [healthy, no_fault, fault1, fault2]
    identities = list(map(id, rows))
    attach_paired_controls(rows)
    assert list(map(id, rows)) == identities
    assert len([row for row in rows if row["kind"] == "healthy"]) == 1
    for detector in DETECTORS:
        assert fault1[f"{detector}_new_vs_healthy"] is False
        assert fault2[f"{detector}_new_vs_healthy"] is False
    assert not any(key.endswith("_new_vs_healthy") for key in healthy)
    assert not any(key.endswith("_new_vs_healthy") for key in no_fault)


def test_control_pairing_distinguishes_noise_profiles_and_rejects_duplicate_keys():
    rows = [_row("healthy", "g", decisions=True, profile="baseline"),
            _row("healthy", "g", decisions=False, profile="accurate"),
            _row("hif", "g", decisions=True, profile="accurate")]
    attach_paired_controls(rows)
    assert rows[-1]["gnn_phase_new_vs_healthy"] is True
    with pytest.raises(ValueError, match="Duplicate healthy"):
        attach_paired_controls([rows[0], deepcopy(rows[0])])
    with pytest.raises(KeyError):
        attach_paired_controls([_row("hif", "missing")])


@pytest.mark.parametrize("healthy,fault,new,lost", [
    (False, True, True, False), (True, False, False, True),
    (True, True, False, False), (False, False, False, False),
    (None, False, None, None), (False, None, None, None), (None, None, None, None),
])
def test_paired_lost_alarms_preserve_direction_and_availability(healthy, fault, new, lost):
    control = _row("healthy", "g", decisions=healthy)
    positive = _row("hif", "g", decisions=fault, phase_score=None if fault is None else .5)
    attach_paired_controls([control, positive])
    summary = summarize([positive], ["kind"])[0]
    for detector in DETECTORS:
        assert positive[f"{detector}_new_vs_healthy"] is new
        assert positive[f"{detector}_lost_vs_healthy"] is lost
        assert summary[f"{detector}_paired_available"] == int(new is not None)
        assert summary[f"{detector}_new_vs_healthy_count"] == int(new is True)
        assert summary[f"{detector}_lost_vs_healthy_count"] == int(lost is True)


def test_summary_tracks_each_detector_availability_and_paired_new_alarms():
    controls = [_row("healthy", "g0", decisions=False),
                _row("healthy", "g1", decisions=True),
                _row("healthy", "g2", decisions=False)]
    faults = [_row("hif", "g0", case_id="f0", decisions=False, phase_score=.8),
              _row("hif", "g1", case_id="f1", decisions=True, phase_score=.2),
              _row("hif", "g2", case_id="f2", decisions=True, phase_score=None)]
    faults[0].update(gnn_phase=True, legacy_type_gnn_phase=True)
    faults[1].update(gnn_phase=False, legacy_type_gnn_phase=True)
    faults[2].update(gnn_phase=None, legacy_type_gnn_phase=None)
    attach_paired_controls(controls + faults)
    summary = summarize(faults, ["kind"])[0]
    assert summary["observations"] == summary["physical_cases"] == summary["noise_groups"] == 3
    assert summary["operating_parents"] == 1
    assert summary["gnn_unavailable"] == 1
    assert summary["gnn_phase_available"] == 2
    assert summary["gnn_phase_count"] == 1
    assert summary["gnn_phase_rate"] == .5
    # An unavailable GNN is not silently converted to a negative prediction or
    # used to remove independently available WLS decisions from their denominator.
    assert summary["wls_dual_available"] == 3
    assert summary["wls_dual_count"] == 2
    assert summary["wls_dual_rate"] == pytest.approx(2 / 3)
    assert summary["gnn_phase_new_vs_healthy_count"] == 1
    assert summary["gnn_phase_lost_vs_healthy_count"] == 1
    assert summary["gnn_phase_paired_available"] == 2
    assert summary["wls_dual_new_vs_healthy_count"] == 1
    assert summary["wls_dual_lost_vs_healthy_count"] == 0
    assert summary["wls_dual_paired_available"] == 3
    assert summary["gnn_additional_to_wls"] == 1
    assert summary["wls_misses_with_gnn_available"] == 1
    assert summary["legacy_type_decision_flips"] == 1


def test_missing_legacy_prediction_is_not_counted_as_a_decision_flip():
    row = _row("hif", "g", decisions=False)
    row.update(gnn_phase=True, legacy_type_gnn_phase=None)
    summary = summarize([row], ["kind"])[0]
    assert summary["legacy_type_decision_flips"] == 0


def test_repeated_healthy_case_is_counted_as_windows_and_unique_physical_case():
    rows = [_row("healthy", "g0", decisions=False), _row("healthy", "g1", decisions=True)]
    summary = summarize(rows, ["kind", "noise_profile"])[0]
    assert summary["observations"] == summary["noise_groups"] == 2
    assert summary["physical_cases"] == summary["operating_parents"] == 1
    assert summary["gnn_phase_rate"] == .5


def test_legacy_type_sensitivity_changes_only_line_transformer_bits():
    from mcp_server.matpower_server import _load_python_case
    from research.gnn_screen.graph_builder import build_graph
    from research.gnn_screen.wls_features import configured_case, state_measurements_and_jacobian
    from three_phase_model.voltage_bases import apply_ieee14_voltage_bases

    case = apply_ieee14_voltage_bases(_load_python_case("case14"))
    observed, _ = state_measurements_and_jacobian(configured_case(case),
                                                 np.deg2rad(case["bus"][:, 8]), case["bus"][:, 7])
    graph = build_graph(case, observed)
    original = deepcopy(graph)
    legacy = legacy_type_graph(graph, case)
    changes = np.argwhere(legacy["edge_attr"] != graph["edge_attr"])
    assert changes.tolist() == [[26, 35], [26, 36], [27, 35], [27, 36]]
    np.testing.assert_array_equal(graph["edge_attr"][[26, 27], 35:37], [[0, 1], [0, 1]])
    np.testing.assert_array_equal(legacy["edge_attr"][[26, 27], 35:37], [[1, 0], [1, 0]])
    for field in ("x", "u", "edge_index", "edge_pair"):
        np.testing.assert_array_equal(legacy[field], original[field])
    for field in ("x", "u", "edge_index", "edge_pair", "edge_attr"):
        np.testing.assert_array_equal(graph[field], original[field])
    assert legacy["metadata"] == original["metadata"]
    assert case["bus"][6, 9] == 13.8 and case["bus"][7, 9] == 18.0
