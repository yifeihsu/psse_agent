from __future__ import annotations

from copy import deepcopy
import json

import numpy as np
import pytest

from psse_env.fault_profiles import (
    get_fault_profile, hif_resistance, hif_resistance_ohm, measurement_chain_error, measurement_sigma,
    parameter_factor, signal_energy_stratum, vuf_stratum,
)
from three_phase_model.voltage_bases import (
    HIF_DETECTION_LIMIT_BAND_OHM, hif_resistance_class, hif_resistance_classification_table,
)


def test_profile_is_versioned_json_safe_and_detached():
    profile = get_fault_profile()
    assert profile["profile_id"] == "reviewed_v1"
    assert not profile["field_accuracy_claim"]
    assert profile["hif"]["stage_weights"] == {"early": [2, 2, 1], "full": [1, 1, 1]}
    assert profile["hif"]["weak_band_pu"] == [20, 200]
    assert profile["hif"]["location_fraction"] == [.25, .75]
    assert profile["harmonic"] == {"sensitivity_thd_fraction": [.01, .05], "stress_thd_fraction": [.1, .2]}
    json.dumps(profile, allow_nan=False)
    profile["hif"]["bands_pu"][0][0] = -1
    profile["noise_profiles"]["baseline"]["sigma_pq"] = 99
    assert get_fault_profile()["hif"]["bands_pu"][0] == [5, 10]
    assert measurement_sigma(14, 20)[-1] == .01
    with pytest.raises(ValueError, match="Unknown fault profile"):
        get_fault_profile("missing_version")


@pytest.mark.parametrize("nb,nl,count", [(14, 20, 122), (57, 80, 491), (1, 0, 3)])
def test_noise_profiles_keep_voltage_weights_and_scale_covariance_squared(nb, nl, count):
    baseline = measurement_sigma(nb, nl)
    moderate = measurement_sigma(nb, nl, "accuracy_005")
    accurate = measurement_sigma(nb, nl, "accuracy_002")
    assert baseline.shape == moderate.shape == accurate.shape == (count,)
    np.testing.assert_array_equal(baseline[:nb], np.full(nb, .001))
    np.testing.assert_array_equal(moderate[:nb], baseline[:nb])
    np.testing.assert_array_equal(accurate[:nb], baseline[:nb])
    np.testing.assert_allclose(moderate[nb:]**2 / baseline[nb:]**2, .25)
    np.testing.assert_allclose(accurate[nb:]**2 / baseline[nb:]**2, .04)
    assert np.all(baseline > 0)


@pytest.mark.parametrize("value,label", [
    (0, "below_1pct"), (np.nextafter(.01, 0), "below_1pct"),
    (.01, "1_to_2pct"), (np.nextafter(.02, 0), "1_to_2pct"),
    (.02, "2_to_3pct"), (.03, "2_to_3pct"),
    (np.nextafter(.03, np.inf), "above_3pct"),
])
def test_vuf_boundaries(value, label):
    assert vuf_stratum(value) == label


@pytest.mark.parametrize("value,label", [
    (0, "below_1"), (np.nextafter(1., 0), "below_1"),
    (1, "1_to_9"), (np.nextafter(9., 0), "1_to_9"),
    (9, "9_to_25"), (25, "9_to_25"),
    (np.nextafter(25., np.inf), "above_25"),
])
def test_signal_energy_boundaries(value, label):
    assert signal_energy_stratum(value) == label


@pytest.mark.parametrize("cohort,bands", [
    ("gross", ((.1, .5), (2., 5.))), ("moderate", ((.7, .95), (1.05, 1.3))),
])
def test_parameter_factors_remain_in_declared_nonunit_bands_and_are_reproducible(cohort, bands):
    first, second = np.random.default_rng(1337), np.random.default_rng(1337)
    values = [parameter_factor(first, cohort) for _ in range(1000)]
    assert values == [parameter_factor(second, cohort) for _ in range(1000)]
    assert all(any(low <= value <= high for low, high in bands) for value in values)
    assert any(value < 1 for value in values) and any(value > 1 for value in values)


@pytest.mark.parametrize("stage,expected", [("early", [.4, .4, .2]), ("full", [1/3, 1/3, 1/3])])
def test_hif_curriculum_uses_requested_band_weights(stage, expected):
    rng = np.random.default_rng(212)
    values = np.array([hif_resistance(rng, stage=stage) for _ in range(6000)])
    assert np.min(values) >= 5 and np.max(values) < 40
    frequencies = np.histogram(values, bins=[5, 10, 20, 40])[0] / len(values)
    np.testing.assert_allclose(frequencies, expected, atol=.025, rtol=0)


@pytest.mark.parametrize("band,bounds", [(0, (5, 10)), (1, (10, 20)), (2, (20, 40)),
    ([5, 10], (5, 10)), ((20, 40), (20, 40)), ("weak", (20, 200)), ([20, 200], (20, 200))])
def test_explicit_hif_bands_are_reproducible_and_independent_of_curriculum_weights(band, bounds):
    a, b = np.random.default_rng(18), np.random.default_rng(18)
    early = [hif_resistance(a, band=band, stage="early") for _ in range(50)]
    full = [hif_resistance(b, band=band, stage="full") for _ in range(50)]
    assert early == full
    assert all(bounds[0] <= value <= bounds[1] for value in full)


def test_measurement_chain_coherently_mixes_p_and_q_with_correct_ct_pt_angle_sign():
    p, q = measurement_chain_error(2., 0., pt_angle_rad=np.pi/2)
    assert p == pytest.approx(0, abs=1e-15) and q == pytest.approx(2)
    p, q = measurement_chain_error(2., 0., ct_angle_rad=np.pi/2)
    assert p == pytest.approx(0, abs=1e-15) and q == pytest.approx(-2)
    original = 1.2 + .4j
    p, q = measurement_chain_error(original.real, original.imag,
        ct_gain=1.01, pt_gain=.98, ct_angle_rad=.02, pt_angle_rad=-.01)
    expected = 1.01 * .98 * np.exp(-.03j) * original
    assert p + 1j*q == pytest.approx(expected)
    assert abs(p + 1j*q) == pytest.approx(1.01 * .98 * abs(original))


def test_measurement_chain_identity_and_equal_phase_errors_add_no_noise_or_mutation():
    p, q = np.array([1., -2., 0.]), np.array([.4, .5, -1.])
    before_p, before_q = p.copy(), q.copy()
    output_p, output_q = measurement_chain_error(p, q, ct_angle_rad=.7, pt_angle_rad=.7)
    np.testing.assert_array_equal(output_p, p)
    np.testing.assert_array_equal(output_q, q)
    output_p[0] = 99
    np.testing.assert_array_equal(p, before_p)
    np.testing.assert_array_equal(q, before_q)


@pytest.mark.parametrize("bad", [-.001, float("nan"), float("inf"), True, "0.1", [0.1]])
def test_strata_reject_invalid_values(bad):
    with pytest.raises(ValueError):
        vuf_stratum(bad)
    with pytest.raises(ValueError):
        signal_energy_stratum(bad)


@pytest.mark.parametrize("args", [(0, 20), (14, -1), (True, 20), (14, 2.5), (14, 20, "unknown")])
def test_noise_profile_rejects_invalid_layouts_or_names(args):
    with pytest.raises(ValueError):
        measurement_sigma(*args)


@pytest.mark.parametrize("kwargs", [{"band": -1}, {"band": 3}, {"band": True}, {"band": [1, 2]},
    {"band": [5, 10, 20]}, {"band": [True, 10]}, {"stage": "unknown"}])
def test_hif_rejects_unknown_ranges_without_consuming_rng(kwargs):
    rng = np.random.default_rng(1)
    before = deepcopy(rng.bit_generator.state)
    with pytest.raises(ValueError):
        hif_resistance(rng, **kwargs)
    assert rng.bit_generator.state == before


def test_samplers_reject_unknown_cohorts_and_non_generator_rng():
    rng = np.random.default_rng(1)
    with pytest.raises(ValueError):
        parameter_factor(rng, "unknown")
    with pytest.raises(ValueError):
        parameter_factor(None)
    with pytest.raises(ValueError):
        hif_resistance(None)


@pytest.mark.parametrize("p,q,kwargs", [
    ([1, 2], [1], {}), (True, 1, {}), (1+0j, 1, {}), ([], [], {}),
    (float("nan"), 1, {}), (1, 1, {"ct_gain": 0}), (1, 1, {"pt_gain": -1}),
    (1, 1, {"ct_angle_rad": float("inf")}), (1, 1, {"ct_gain": 1e308, "pt_gain": 1e308}),
])
def test_measurement_chain_rejects_invalid_or_nonfinite_inputs(p, q, kwargs):
    with pytest.raises(ValueError):
        measurement_chain_error(p, q, **kwargs)


def test_physical_profile_inherits_other_families_and_keeps_voltage_strata_explicit():
    legacy = get_fault_profile()
    profile = get_fault_profile("ieee14_physical_hif_v1")
    assert profile["profile_id"] == "ieee14_physical_hif_v1"
    assert profile["parent_profile_id"] == "reviewed_v1"
    assert profile["voltage_profile"] == "ieee14_nominal_69_13p8_18kv_v1"
    assert not profile["voltage_base_profile"]["universal_ieee14_variant_claim"]
    assert not profile["field_accuracy_claim"]
    for family in ("noise_profiles", "parameter", "unbalance", "harmonic", "measurement_chain", "signal_energy"):
        assert profile[family] == legacy[family]
    hif = profile["hif"]
    assert hif["resistance_input_unit"] == "ohm"
    assert hif["bands_ohm"] == [[100, 200], [200, 500], [500, 1000]]
    assert hif["main_range_ohm"] == [100, 1000]
    assert hif["evaluation_sweep_ohm"] == [50, 100, 200, 500, 1000, 2000]
    assert hif["extreme_evaluation_ohm"] == [5000]
    assert hif["main_voltage_stratum_kv_ll"] == 69
    assert hif["evaluation_voltage_strata_kv_ll"] == [69, 13.8]
    assert hif["location_fraction"] == [.25, .75]
    assert hif["stage_weights"] == {"early": [2, 2, 1], "full": [1, 1, 1]}
    assert "bands_pu" not in hif and "weak_band_pu" not in hif
    # 2026-09-19: opt-in detection-limit cohort, classification table, pu labels.
    assert hif["detection_limit_band_ohm"] == [1000, 5000] == list(HIF_DETECTION_LIMIT_BAND_OHM)
    assert hif["detection_limit_band_ohm"] not in hif["bands_ohm"]
    assert hif["detection_limit_voltage_stratum_kv_ll"] == 69
    assert hif["resistance_classification_ohm"] == hif_resistance_classification_table()
    assert [row["name"] for row in hif["resistance_classification_ohm"]] == [
        "low_resistance_fault", "moderately_resistive", "moderately_high_resistance", "representative_hif",
        "weak_hif", "extreme_weak_hif", "near_open_circuit"]
    assert [(row["lower"], row["upper"]) for row in hif["resistance_classification_ohm"]] == [
        (0, 50), (50, 100), (100, 200), (200, 500), (500, 1000), (1000, 5000), (5000, None)]
    pu = hif["pu_equivalents_69kv"]
    assert pu["impedance_base_ohm"] == pytest.approx(47.61)
    assert pu["main_range_pu"] == pytest.approx([100 / 47.61, 1000 / 47.61])
    assert pu["main_range_pu"] == pytest.approx([2.1004, 21.004], abs=1e-3)
    assert pu["detection_limit_band_pu"] == pytest.approx([21.004, 105.02], abs=1e-2)
    assert pu["evaluation_sweep_pu"] == pytest.approx([r / 47.61 for r in (50, 100, 200, 500, 1000, 2000)])
    assert pu["extreme_evaluation_pu"] == pytest.approx([5000 / 47.61])
    assert pu["bands_pu"] == pytest.approx(np.asarray([[lo / 47.61, hi / 47.61] for lo, hi in hif["bands_ohm"]]))
    assert "47.6 kOhm" in pu["note"] and "must not be described as a 1000 ohm HIF" in pu["note"]
    json.dumps(profile, allow_nan=False)
    hif["bands_ohm"][0][0] = -1
    hif["resistance_classification_ohm"][0]["name"] = "mutated"
    profile["parameter"]["gross_factor_bands"][0][0] = -1
    assert get_fault_profile("ieee14_physical_hif_v1")["hif"]["bands_ohm"][0] == [100, 200]
    assert get_fault_profile("ieee14_physical_hif_v1")["hif"]["resistance_classification_ohm"][0]["name"] == "low_resistance_fault"
    assert get_fault_profile() == legacy
    assert "detection_limit_band_ohm" not in legacy["hif"] and "resistance_classification_ohm" not in legacy["hif"]


@pytest.mark.parametrize("stage,expected", [("early", [.4, .4, .2]), ("full", [1/3, 1/3, 1/3])])
def test_physical_hif_curriculum_only_samples_main_ohm_bands(stage, expected):
    rng = np.random.default_rng(212)
    values = np.array([hif_resistance_ohm(rng, stage=stage) for _ in range(6000)])
    assert np.min(values) >= 100 and np.max(values) < 1000
    frequencies = np.histogram(values, bins=[100, 200, 500, 1000])[0] / len(values)
    np.testing.assert_allclose(frequencies, expected, atol=.025, rtol=0)


@pytest.mark.parametrize("band,bounds", [
    (0, (100, 200)), (1, (200, 500)), (2, (500, 1000)),
    ("100_200", (100, 200)), ("200_500", (200, 500)), ("500_1000", (500, 1000)),
    ([100, 200], (100, 200)), ((200, 500), (200, 500)), (np.array([500, 1000]), (500, 1000)),
    ("detection_limit", (1000, 5000)), ([1000, 5000], (1000, 5000)), ((1000.0, 5000.0), (1000, 5000)),
    (np.array([1000, 5000]), (1000, 5000)),
])
def test_physical_explicit_hif_bands_are_reproducible_without_stage_weighting(band, bounds):
    first, second = np.random.default_rng(18), np.random.default_rng(18)
    early = [hif_resistance_ohm(first, band, "early") for _ in range(30)]
    full = [hif_resistance_ohm(second, band, "full") for _ in range(30)]
    assert early == full
    assert all(bounds[0] <= value < bounds[1] for value in full)


def test_physical_detection_limit_band_is_opt_in_and_classified_extreme():
    rng = np.random.default_rng(2026)
    draws = [hif_resistance_ohm(rng, band="detection_limit") for _ in range(2000)]
    assert min(draws) >= 1000 and max(draws) < 5000
    assert {hif_resistance_class(value) for value in draws} == {"extreme_weak_hif"}
    # Integer indices never reach the detection-limit band (it is not in bands_ohm).
    with pytest.raises(ValueError, match="0, 1, or 2"):
        hif_resistance_ohm(rng, band=3)
    # The curriculum (band=None) never draws the detection-limit cohort at either stage.
    for stage in ("early", "full"):
        curriculum = [hif_resistance_ohm(rng, stage=stage) for _ in range(6000)]
        assert max(curriculum) < 1000
        assert {hif_resistance_class(value) for value in curriculum} == {
            "moderately_high_resistance", "representative_hif", "weak_hif"}
    # The reviewed pu sampler does not learn the physical band name.
    with pytest.raises(ValueError, match="Unknown HIF band"):
        hif_resistance(rng, band="detection_limit")


@pytest.mark.parametrize("kwargs", [
    {"band": -1}, {"band": 3}, {"band": True}, {"band": "weak"}, {"band": "5_10"},
    {"band": [50, 2000]}, {"band": [100, 200, 500]}, {"band": [True, 200]},
    {"stage": "unknown"}, {"band": 5000},
])
def test_physical_sampler_rejects_unknown_or_eval_ranges_without_rng_consumption(kwargs):
    rng = np.random.default_rng(1)
    before = deepcopy(rng.bit_generator.state)
    with pytest.raises(ValueError):
        hif_resistance_ohm(rng, **kwargs)
    assert rng.bit_generator.state == before


def test_physical_sampler_rejects_non_generator_and_keeps_legacy_seed_stream():
    with pytest.raises(ValueError):
        hif_resistance_ohm(None)
    actual, expected = np.random.default_rng(917), np.random.default_rng(917)
    bands = [[5., 10.], [10., 20.], [20., 40.]]
    for _ in range(30):
        bounds = bands[int(expected.choice(3, p=np.array([2., 2., 1.]) / 5))]
        assert hif_resistance(actual, stage="early") == float(expected.uniform(*bounds))
