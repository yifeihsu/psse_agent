"""Scientific separation of physical relevance and observable signal strength."""
import copy

import numpy as np
import pytest
from scipy.special import ndtri

from research.gnn_screen.scenario_policy import classify_scenario, paired_visibility


@pytest.mark.parametrize("distance,band", [(0, "below2"), (1.9999, "below2"), (2, "boundary2to4"),
                                          (3.9999, "boundary2to4"), (4, "visible_ge4")])
def test_gaussian_visibility_boundaries_and_oracle(distance, band):
    result = paired_visibility([0, 0], [distance, 0], [1, 1])
    assert result["mean_separation_d"] == distance
    assert result["rawband"] == band
    if distance == 0:
        assert result["oracle_recall"] == pytest.approx(.01)
    if distance == 4:
        assert result["oracle_recall"] == pytest.approx(.952900506156048)
    assert result["physical_importance_inferred"] is False
    assert "unknown operating parents" in result["oracle_limitation"]


def test_gaussian_oracle_and_joint_change_of_units_are_consistent():
    target_distance = ndtri(.99) + ndtri(.95)
    result = paired_visibility([.7, .1], [.7 + target_distance * .02, .1], [.02, .003])
    assert result["oracle_recall"] == pytest.approx(.95)
    changed_units = paired_visibility([700, 100], [( .7 + target_distance * .02) * 1000, 100], [20, 3])
    assert changed_units["mean_separation_d"] == pytest.approx(result["mean_separation_d"])


def test_whitened_projection_distinguishes_absorbed_and_residual_signal():
    # First channel lies on the balanced state manifold; second does not.
    result = paired_visibility([0, 0], [20, 6], [2, 3], H=[[2], [0]])
    assert result["mean_separation_d_squared"] == pytest.approx(104)
    assert result["local_absorbed_state_energy"] == pytest.approx(100)
    assert result["local_projected_residual_energy"] == pytest.approx(4)
    assert result["local_jacobian_rank"] == 1
    assert result["local_residual_degrees_of_freedom"] == 1
    absorbed = paired_visibility([0, 0], [20, 0], [2, 3], H=[[2], [0]])
    assert absorbed["mean_separation_d"] == 10
    assert absorbed["local_projected_residual_energy"] == pytest.approx(0)


def test_projection_invariant_to_whitened_rotation_state_basis_and_rank_deficiency():
    rng = np.random.default_rng(841)
    jacobian = rng.normal(size=(8, 3))
    shift = rng.normal(size=8)
    rotation, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    before = paired_visibility(np.zeros(8), shift, np.ones(8), H=jacobian)
    after = paired_visibility(np.zeros(8), rotation @ shift, np.ones(8), H=rotation @ jacobian)
    duplicate = paired_visibility(np.zeros(8), shift, np.ones(8), H=np.column_stack([jacobian, jacobian[:, 0] * 7]))
    for result in (after, duplicate):
        assert result["local_jacobian_rank"] == 3
        assert result["mean_separation_d"] == pytest.approx(before["mean_separation_d"])
        assert result["local_projected_residual_energy"] == pytest.approx(before["local_projected_residual_energy"])


def test_differing_configured_case_cannot_be_declared_indistinguishable_from_z_alone():
    result = paired_visibility([1, 2], [1, 2], [.1, .1], same_configured_model=False)
    assert result["mean_separation_d"] is None and result["oracle_recall"] is None
    assert result["rawband"] == "not_applicable_configured_model_discrepancy"
    assert result["model_discrepancy_visibility"]["measurement_only_comparison"]["mean_separation_d"] == 0
    policy = classify_scenario(["parameter"], parameter_physical_factors={"R": .2}, visibility=result)
    assert policy["physical_cohort"] == "core"
    assert policy["balanced_visible_ge4"] is None


@pytest.mark.parametrize("factor", [.1, .5, 2, 5])
def test_parameter_uses_physical_factors_on_both_sft_range_boundaries(factor):
    for components in ({"R": factor}, {"X": factor}, {"R": factor, "X": factor}):
        result = classify_scenario(["parameter"], parameter_physical_factors=components)
        assert result["physical_cohort"] == "core"
        assert result["family_rules"]["parameter"]["factor_convention"] == "physical_actual_over_reference"


@pytest.mark.parametrize("factor,cohort", [(.8, "boundary"), (1, "out_of_scope"), (.05, "out_of_scope"), (8, "out_of_scope")])
def test_parameter_ranges_are_profile_membership_not_reciprocal_configured_errors(factor, cohort):
    assert classify_scenario(["parameter"], parameter_physical_factors={"R": factor})["physical_cohort"] == cohort
    with pytest.raises(ValueError, match="R and/or X"):
        classify_scenario(["parameter"], parameter_physical_factors={"configured_multiplier": factor})


@pytest.mark.parametrize("strength,cohort", [(0, "out_of_scope"), (9.99, "boundary"), (10, "core"),
                                           (15, "core"), (15.01, "out_of_scope")])
def test_measurement_sft_strengths(strength, cohort):
    assert classify_scenario(["measurement"], measurement_sigma_multiple=strength)["physical_cohort"] == cohort


@pytest.mark.parametrize("vuf,cohort,stratum", [(.00999, "boundary", "below1_percent"),
    (.01, "core", "main_1_to_2_percent"), (.01999, "core", "main_1_to_2_percent"),
    (.02, "core", "strong_ge2_percent")])
def test_unbalance_is_actual_sequence_ratio(vuf, cohort, stratum):
    policy = classify_scenario(["unbalance"], max_vuf=vuf)
    assert policy["physical_cohort"] == cohort
    assert policy["family_rules"]["unbalance"]["physical_stratum"] == stratum
    assert "not_NEMA" in policy["family_rules"]["unbalance"]["metric"]


def test_hif_remains_positive_below_balanced_visibility_and_never_selects_on_wls_alarm():
    visibility = paired_visibility([0, 0], [.01, 0], [1, 1])
    visibility["wls_alarm"] = False
    quiet = classify_scenario(["hif"], hif_injected=True, hif_phase_current_sigma=10, visibility=visibility)
    alarming_visibility = copy.deepcopy(visibility)
    alarming_visibility.update(wls_alarm=True, rawband="visible_ge4")
    alarming = classify_scenario(["hif"], hif_injected=True, hif_phase_current_sigma=10, visibility=alarming_visibility)
    assert quiet["physical_cohort"] == alarming["physical_cohort"] == "core"
    assert quiet["balanced_visible_ge4"] is False and alarming["balanced_visible_ge4"] is True
    assert quiet["selection_uses_wls_alarm"] is False
    boundary = classify_scenario(["hif"], hif_injected=True, hif_phase_current_sigma=9.99)
    assert boundary["physical_cohort"] == "boundary"
    assert boundary["family_rules"]["hif"]["physical_importance_preserved"] is True
    assert "does not mean harmless" in boundary["family_rules"]["hif"]["interpretation"]


def test_healthy_controls_and_mixed_cases_preserve_family_scope():
    assert classify_scenario([], visibility={"rawband": "visible_ge4"})["physical_cohort"] == "control"
    mixed = classify_scenario(["hif", "unbalance"], hif_injected=True, hif_phase_current_sigma=12, max_vuf=.005)
    assert mixed["physical_cohort"] == "boundary"
    assert mixed["family_rules"]["hif"]["cohort"] == "core"
    assert mixed["family_rules"]["unbalance"]["cohort"] == "boundary"
    assert classify_scenario(["topology"], topology_changed_connectivity=False)["physical_cohort"] == "out_of_scope"


@pytest.mark.parametrize("strength,cohort", [(5.999, "boundary"), (6, "core"), (20, "core")])
def test_preferred_differential_current_sft_gate_is_six_sigma(strength, cohort):
    policy = classify_scenario(["hif"], hif_injected=True, hif_differential_current_sigma=strength)
    assert policy["physical_cohort"] == cohort
    rule = policy["family_rules"]["hif"]
    assert rule["diagnostic_criterion"] == "two_terminal_differential_current_ge6_sigma"
    assert rule["diagnostic_threshold_sigma"] == 6
    assert rule["physical_importance_preserved"] is True


def test_differential_current_takes_precedence_and_reports_legacy_fallback():
    main = classify_scenario(["hif"], hif_injected=True, hif_differential_current_sigma=6,
                             hif_phase_current_sigma=1)
    boundary = classify_scenario(["hif"], hif_injected=True, hif_differential_current_sigma=5,
                                 hif_phase_current_sigma=100)
    legacy = classify_scenario(["hif"], hif_injected=True, hif_phase_current_sigma=10)
    assert main["physical_cohort"] == "core"
    assert boundary["physical_cohort"] == "boundary"
    assert boundary["family_rules"]["hif"]["diagnostic_significance_sigma"] == 5
    assert legacy["physical_cohort"] == "core"
    assert legacy["family_rules"]["hif"]["diagnostic_criterion"] == "legacy_phase_current_ge10_sigma"


@pytest.mark.parametrize("kwargs", [dict(z_healthy=[0], z_fault=[1], sigma=[0]),
    dict(z_healthy=[0], z_fault=[1, 2], sigma=[1]), dict(z_healthy=[0], z_fault=[np.nan], sigma=[1]),
    dict(z_healthy=[0], z_fault=[1], sigma=[1], H=[[1], [2]])])
def test_visibility_rejects_undefined_or_mismatched_noise_model(kwargs):
    with pytest.raises(ValueError):
        paired_visibility(**kwargs)
