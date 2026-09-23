import math
import pytest
from three_phase_nlm import hif_units as u
from three_phase_nlm.dss_hif_injector import hif_ohms_from_pu


def test_physical_line_eligibility_and_conversion():
    assert len(u.PHYSICAL_ELIGIBLE_HIF_BRANCHES) == 16
    assert 13 not in u.PHYSICAL_ELIGIBLE_HIF_BRANCHES
    assert len(u.eligible_rows_for_stratum("69kv")) == 7
    assert len(u.eligible_rows_for_stratum("13p8kv")) == 9
    record = u.hif_resistance_record(branch_row0=2, resistance_ohm=500)
    assert record['r_hif_pu'] == pytest.approx(500 / 47.61)
    assert record['r_hif_model_ohm'] == pytest.approx(500 / 47.61 * .01)
    assert record['nominal_fault_current_a'] == pytest.approx(69000 / math.sqrt(3) / 500)
    assert record['resistance_class'] == 'weak_hif'
    assert u.hif_resistance_record(branch_row0=10, resistance_ohm=500)['r_hif_pu'] == pytest.approx(500/1.9044)
    assert u.resolve_line_kv_ll(13)['cross_voltage_branch']
    with pytest.raises(ValueError):
        hif_ohms_from_pu(10, kv_ll=69)


def test_legacy_and_physical_label_roundtrip():
    old = dict(r_hif_pu=124.17, r_hif_ohm=1.2417, kv_ln=.577, branch_row0=11)
    assert u.label_model_ohm(old) == 1.2417
    assert u.label_physical_ohm(old) == pytest.approx(124.17*1.9044)
    new = u.hif_resistance_record(branch_row0=2, resistance_ohm=500)
    assert u.label_physical_ohm(new) == 500
    assert u.label_model_ohm(new) == pytest.approx(500/47.61*.01)
    assert u.label_local_kv_ll(new) == 69


def test_search_boxes_are_local_and_mixed_units_fail():
    box = u.resolve_resistance_search_box(branch_row0=0)
    assert (box['r_hif_pu_min'], box['r_hif_pu_max']) == pytest.approx((50/47.61, 5000/47.61))
    assert u.resolve_resistance_search_box(branch_row0=0, r_hif_pu_min=5, r_hif_pu_max=1000)['box_source'] == 'explicit_pu'
    assert u.resolve_resistance_search_box(branch_row0=10, r_hif_ohm_min=100, r_hif_ohm_max=500)['r_hif_pu_max'] == pytest.approx(500/1.9044)
    assert u.resolve_resistance_search_box(branch_row0=0, default_ohm=None, default_pu=(5,1000))['r_hif_pu_min'] == 5
    with pytest.raises(ValueError):
        u.resolve_resistance_search_box(branch_row0=0, r_hif_pu_min=5, r_hif_pu_max=1000, r_hif_ohm_min=50, r_hif_ohm_max=5000)


@pytest.mark.parametrize('ohm,category', [(49,'low_resistance_fault'),(50,'moderately_resistive'),(100,'moderately_high_resistance'),(200,'representative_hif'),(500,'weak_hif'),(1000,'extreme_weak_hif'),(5000,'near_open_circuit')])
def test_class_boundaries(ohm, category):
    assert u.hif_resistance_class(ohm) == category
