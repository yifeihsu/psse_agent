"""Reviewed physical cohorts stay distinct from their observable graph inputs."""
from __future__ import annotations

import copy
import json

import numpy as np
import pytest

pytest.importorskip("opendssdirect")
from psse_env.fault_profiles import measurement_sigma
from research.gnn_screen.dataset import load_manifest, prepare_corpus, write_json
from research.gnn_screen.practical_corpus import (
    REVIEWED_AUXILIARY_MANIFESTS, generate_corpus, meter_overlay,
    mixed_meter_overlay, noiseless_wls_audit, perturb_physical_case, reviewed_slots,
)
from psse_env.systems import resolve_system


@pytest.fixture(scope="module")
def reviewed(tmp_path_factory):
    out = tmp_path_factory.mktemp("reviewed_practical") / "corpus"
    report = generate_corpus(out, parents_by_split={"train": 1, "validation": 1},
        seed=742019, noise_replicates=2, healthy_calibration_replicates=2,
        attempt_cap=4, scenario_profile="reviewed_v1", noise_profile="accuracy_005", stage="early")
    return out, report


def test_curriculum_slots_and_heldout_sensitivity_are_explicit():
    early = reviewed_slots(stage="early", split="train")
    full = reviewed_slots(stage="full", split="train")
    assert [sum(options.get("hif_band") == band for _, _, options in early) for band in range(3)] == [2, 2, 1]
    assert [sum(options.get("hif_band") == band for _, _, options in full) for band in range(3)] == [1, 1, 1]
    assert not any(options.get("destination") for _, _, options in early)
    targets = {options["vuf_target"] for _, _, options in reviewed_slots(stage="full", split="validation") if "vuf_target" in options}
    assert targets == {"below_1pct", "1_to_2pct", "2_to_3pct", "above_3pct"}
    assert reviewed_slots(stage="early", split="calibration") == []


def test_noise_profiles_scale_both_meter_error_and_estimator_covariance():
    for name, expected_power in (("baseline", .01), ("accuracy_005", .005), ("accuracy_002", .002)):
        sigma = measurement_sigma(14, 20, noise_profile=name)
        assert sigma[0] == .001 and sigma[14] == expected_power
        changed, audit = meter_overlay(np.zeros(122), np.random.default_rng(46), count=3, measurement_sigma=sigma)
        indices = audit["channel_indices0"]
        np.testing.assert_allclose(changed[indices] / sigma[indices], audit["sigma_multiples"])


def test_moderate_parameter_samples_are_not_forced_into_gross_bands():
    case = resolve_system("case14").load_case()
    for i in range(8):
        _, audit = perturb_physical_case(case, np.random.default_rng(i), family="parameter",
                                         scenario_profile="reviewed_v1", parameter_cohort="moderate")
        for factor in audit["physical_factors"].values():
            assert .7 <= factor <= .95 or 1.05 <= factor <= 1.3


def test_failed_exact_wls_is_unavailable_not_zero_signal():
    result = noiseless_wls_audit({}, np.zeros(122), np.ones(122))
    assert not result["success"] and result["J_exact"] is None
    assert result["residual_visible_energy_bin"] == "unavailable" and result["error"]
    assert not result["used_for_admission"]


def test_reviewed_manifests_enforce_achieved_strata_and_keep_every_positive(reviewed):
    out, report = reviewed
    assert report["schema"] == "practical_physical_wls_screen_corpus_v2"
    assert report["scenario_profile"] == "reviewed_v1" and report["curriculum_stage"] == "early"
    assert report["noise_profile"] == "accuracy_005" and not report["exact_wls_audit_used_for_admission"]
    all_rows = []
    for filename in ("manifest.jsonl", "boundary_manifest.jsonl", *REVIEWED_AUXILIARY_MANIFESTS.values()):
        rows = load_manifest(out / filename) if (out / filename).read_text().strip() else []
        all_rows.extend(rows)
        for row in rows:
            np.testing.assert_array_equal(row["measurement_sigma"], measurement_sigma(14, 20, "accuracy_005"))
            meta = row["offline_metadata"]
            assert meta["scenario_profile"] == "reviewed_v1"
            assert meta["noiseless_wls_audit"]["used_for_admission"] is False
            if filename in REVIEWED_AUXILIARY_MANIFESTS.values():
                assert row["split"] in {"validation", "test"} and not meta["training_eligible"]
            if filename == "manifest.jsonl" and "unbalance" in row["families"]:
                assert meta["vuf_stratum"] == meta["requested_vuf_stratum"]
                assert meta["vuf_stratum"] in {"1_to_2pct", "2_to_3pct"}
            if "unbalance" in row["families"]:
                assert meta["actual_source_bus"] is not None and len(meta["source_phase_powers_system_pu"]) == 3
            if "hif" in row["families"]:
                assert meta["physical_hif"]["fault_power_w_in_normalized_model"] > 0
                assert meta["physical_hif"]["resistance_to_line_impedance_ratio"] > 0
            if "measurement" in row["families"] and len(row["families"]) > 1:
                meter = meta["settings"]["measurement"]
                assert 14 <= meter["channel_indices0"][0] < 122
                assert .1 <= abs(meter["additive_offsets_pu"][0]) <= .3
                physics = json.loads((out / meta["physical_audit_path"]).read_text())
                assert physics["mixed_counterfactuals"]["source_component_window_id"] == meta["component_core_audit"]["source_window_id"]
    weak = load_manifest(out / REVIEWED_AUXILIARY_MANIFESTS["weak_hif_evaluation"])
    assert weak and all(20 <= row["offline_metadata"]["settings"]["hif"]["resistance_pu"] <= 200 for row in weak)
    moderate = load_manifest(out / REVIEWED_AUXILIARY_MANIFESTS["parameter_sensitivity"])
    assert moderate and all(row["offline_metadata"]["settings"]["parameter_cohort"] == "moderate" for row in moderate)
    ledger = [json.loads(line) for line in (out / "proposal_ledger.jsonl").read_text().splitlines()]
    valid = [event for event in ledger if event.get("event") == "proposal" and event.get("physics_valid")]
    positive = [row for row in all_rows if row["families"]]
    assert len(valid) == len(positive)
    assert all(event["boundary_snapshot_retained"] for event in valid if event.get("outcome") == "valid_positive_below_main_criteria")
    assert not any(event.get("scenario_policy", {}).get("admission_uses_J_exact") for event in valid)


def test_reviewed_mixed_offsets_stay_absolute_across_accuracy_profiles():
    outputs = []
    for name in ("baseline", "accuracy_005", "accuracy_002"):
        sigma = measurement_sigma(14, 20, name)
        changed, audit = mixed_meter_overlay(np.zeros(122), np.random.default_rng(72), sigma)
        outputs.append(changed)
        index = audit["channel_indices0"][0]
        assert index >= 14 and 0.1 <= abs(changed[index]) <= .3
        assert changed[index] / sigma[index] == pytest.approx(audit["sigma_multiples"][0])
    np.testing.assert_array_equal(outputs[0], outputs[1])
    np.testing.assert_array_equal(outputs[0], outputs[2])


def test_offline_exact_energy_and_physics_cannot_change_graph_features(reviewed, tmp_path):
    out, _ = reviewed
    source = load_manifest(out / "manifest.jsonl")[0]
    source["noise_replicates"] = 1
    first_path, second_path = tmp_path / "first.jsonl", tmp_path / "second.jsonl"
    # load_manifest returns resolved numeric model/noise inputs. Its derived
    # labels field is intentionally excluded from the manifest input schema.
    source.pop("labels", None)
    first_path.write_text(json.dumps(source) + "\n")
    modified = copy.deepcopy(source)
    modified["offline_metadata"]["noiseless_wls_audit"] = {"J_exact": 1e99, "residual_visible_energy_bin": "forged"}
    modified["offline_metadata"]["physical_hif"] = {"fault_power_w_in_normalized_model": -1e99}
    second_path.write_text(json.dumps(modified) + "\n")
    a, b = prepare_corpus(first_path), prepare_corpus(second_path)
    assert len(a.samples) == len(b.samples) == 1
    for field in ("x", "edge_attr", "u", "edge_index"):
        np.testing.assert_array_equal(a.samples[0].graph[field], b.samples[0].graph[field])
