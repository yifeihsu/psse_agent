from copy import deepcopy

import numpy as np
import pytest

from scripts import verify_hif_continuation as probe


def fixture():
    values = np.zeros(122)
    values[14], values[26] = 2., .4
    aux = {"three_phase_voltages": [{"bus": "b1", "phasor": [1., 0.]}],
           "three_phase_branch_currents": [{"branch_row0": 0, "phasor": [.1, .2]}]}
    scans = [{"scan_index": index, "z_obs": values.tolist(), "op_point": {"load_scale": 1.},
              **deepcopy(aux)} for index in (0, 1)]
    scans[1]["three_phase_voltages"][0]["phasor"][0] = 1.1
    execution = {"scenario_id": "fixture", "case": "case14", "measurements": values.tolist(),
        "metadata": {"sigma_z": [.01]*122, "hif_runtime": {"z_obs": values.tolist(), **aux},
                     "hif_scan_window": {"scans": scans}}}
    return {"execution": execution, "grouping": {"scenario_family": "measurement+hif", "error_cardinality": 2},
        "audit": {"truth": {"clean_measurements": [0.]*122,
            "true_measurement_errors": [{"index": 26, "clean": .2, "observed": .4}]},
            "release_audit": {"tolerances": {"measurement_abs": .0424264}}}}


def test_meter_audit_uses_noisy_event_present_reference_not_event_free_clean_vector():
    envelope = fixture()
    final = list(envelope["execution"]["measurements"])
    final[26] = .201
    audit = probe.offline_meter_audit(envelope, {"final_measurements": final})
    assert audit["meter_recovery"] and audit["exact_write_support"]
    assert audit["target_absolute_errors_pu"]["26"] == pytest.approx(.001)
    assert audit["off_target_write_indices"] == []


def test_overlap_probe_preserves_same_channel_error_and_only_rebinds_current_scada():
    envelope = fixture()
    execution = envelope["execution"]
    before = deepcopy(execution)
    center = np.asarray(execution["measurements"]).copy()
    center[14], center[26] = 2.001, .2
    effect = np.zeros(122)
    effect[14] = .15
    prediction = {"predicted_hif_measurements": center.tolist(), "measurement_effect": effect.tolist(),
                  "prediction_lower": center.tolist(), "prediction_upper": center.tolist()}
    changed, evidence = probe.overlap_probe(execution, prediction)
    assert execution == before
    assert evidence["target_index"] == 14 and evidence["material_overlap"]
    assert evidence["same_channel_error_retained"] and evidence["candidate_contains_injected_index"]
    assert evidence["conditional_decision"]["candidate_indices"] == [14, 26]
    assert changed["metadata"]["hif_scan_window"]["scans"][1] == before["metadata"]["hif_scan_window"]["scans"][1]
    assert changed["metadata"]["hif_scan_window"]["scans"][0]["z_obs"] == changed["measurements"]
    assert changed["metadata"]["hif_runtime"]["z_obs"] == changed["measurements"]


def test_paired_control_restores_only_overlay_and_keeps_independent_history_and_auxiliary_noise():
    envelope = fixture()
    before = deepcopy(envelope)
    control = probe.paired_hif_control(envelope)
    assert envelope == before
    assert control["execution"]["measurements"][26] == .2
    assert control["execution"]["measurements"][14] == 2.
    assert control["execution"]["metadata"]["hif_runtime"]["three_phase_voltages"] == before["execution"]["metadata"]["hif_runtime"]["three_phase_voltages"]
    assert control["execution"]["metadata"]["hif_scan_window"]["scans"][1] == before["execution"]["metadata"]["hif_scan_window"]["scans"][1]
    assert control["audit"]["truth"]["true_measurement_errors"] == []
    probe.assert_observable(control["execution"])


@pytest.mark.parametrize("key", ["audit", "truth", "z_clean", "three_phase_voltages_clean", "true_hif_errors"])
def test_runtime_rejects_private_fields_at_any_depth(key):
    with pytest.raises(ValueError, match="Private runtime input"):
        probe.assert_observable({"metadata": {"window": [{key: []}]}})


def test_fit_cache_is_bound_to_observations_and_supports_case_alias(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "case_fingerprint", lambda path: "semantic-case14-hash")
    model_identity = ["model_v1"]
    monkeypatch.setattr(probe, "model_fingerprint", lambda path: model_identity[0])

    class Provider:
        calls = 0

        def _memoized_hif_multiscan(self, **kwargs):
            self.calls += 1
            return {"success": True, "selected_scan_indices": [1]}

    provider = Provider()
    cache = probe.ObservableFitCache(tmp_path)
    cache.attach(provider)
    kwargs = {"case_path": "case14", "scans": [{"scan_index": 1, "z_obs": [1.]}]}
    provider._memoized_hif_multiscan(**kwargs)
    provider._memoized_hif_multiscan(**kwargs)
    assert provider.calls == 1 and cache.calls[-1]["origin"] == "cached_fresh_observable_fit"
    kwargs["scans"][0]["z_obs"][0] = 2.
    provider._memoized_hif_multiscan(**kwargs)
    assert provider.calls == 2
    model_identity[0] = "changed_model_v2"
    provider._memoized_hif_multiscan(**kwargs)
    assert provider.calls == 3
