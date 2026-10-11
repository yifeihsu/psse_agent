"""The admission rules of psse_env.wls_admission and the generator option that selects them."""
from __future__ import annotations

import pytest
from scipy.stats import chi2

from psse_env import wls_admission as wa

THR14, THR118 = chi2.ppf(0.99, 95), chi2.ppf(0.99, 863)


def test_chi_square_margin_rule_is_the_legacy_arithmetic():
    assert wa.admitted(1.25 * THR14 + 1e-6, 95, 0.0) is True
    assert wa.admitted(1.25 * THR14, 95, 4.99) is False        # strict on the chi-square side, 5 sigma on the residual side
    assert wa.admitted(0.0, 95, 5.0) is True
    assert wa.admitted(0.0, 95, 5.0, residual_threshold=None) is False
    assert wa.admitted(float("nan"), 95, 5.0) is False


def test_excess_energy_rule_admits_the_same_error_on_both_networks():
    # The same error: excess 80 and a 4.2-sigma channel.  Under the legacy rule it is admitted on
    # IEEE 14 (excess needed 67.5) and rejected on IEEE 118 (excess needed 340, no 5-sigma channel).
    assert wa.admitted(95 + 80.0, 95, 4.2) is True
    assert wa.admitted(863 + 80.0, 863, 4.2) is False
    assert wa.admitted(95 + 80.0, 95, 4.2, rule="excess_energy") is True
    assert wa.admitted(863 + 80.0, 863, 4.2, rule="excess_energy") is True
    # Below the floor only a 5-sigma channel admits.
    assert wa.admitted(863 + 60.0, 863, 4.9, rule="excess_energy") is False
    assert wa.admitted(863 + 60.0, 863, 5.0, rule="excess_energy") is True


def test_excess_energy_rule_still_requires_the_operator_alarm():
    # Excess 80 without any 4-sigma channel: the IEEE 14 chi-square alarm fires (80 > 35), the IEEE 118 one does not (80 < 99.6).
    assert wa.operator_alarm(95 + 80.0, 95, 3.5) is True
    assert wa.operator_alarm(863 + 80.0, 863, 3.5) is False
    assert wa.admitted(95 + 80.0, 95, 3.5, rule="excess_energy") is True
    assert wa.admitted(863 + 80.0, 863, 3.5, rule="excess_energy") is False
    with pytest.raises(ValueError):
        wa.admitted(1.0, 95, 1.0, rule="something_else")
    assert "floor 70.0" in wa.describe(900.0, 863, 4.2, rule="excess_energy")


def test_generator_selects_the_rule_and_reports_it(monkeypatch):
    from psse_env.providers.scenario_generator import Round0ScenarioGenerator, ScenarioRejected

    legacy = Round0ScenarioGenerator(seed=20261011, normalized_residual_threshold=4.0)
    energy = Round0ScenarioGenerator(seed=20261011, normalized_residual_threshold=4.0, admission_rule="excess_energy")
    for generator in (legacy, energy):
        monkeypatch.setattr(generator, "_wls_detection", lambda case, z, **kw: (863 + 80.0, 4.2))
        monkeypatch.setattr(generator, "_wls_dof", lambda case, z, **kw: 863)
        monkeypatch.setattr(generator, "_detection_detail", lambda case, z, **kw: "chi2 943.0 vs 962.6, max|r| 4.20 vs 4.00")
    with pytest.raises(ScenarioRejected, match="anomaly_not_detectable"):
        legacy._require_anomalous("case118", [0.0], "hif")
    energy._require_anomalous("case118", [0.0], "hif")
    monkeypatch.setattr(energy, "_wls_detection", lambda case, z, **kw: (863 + 60.0, 4.2))
    with pytest.raises(ScenarioRejected, match="floor 70.0"):
        energy._require_anomalous("case118", [0.0], "hif")
    report = energy.report()
    assert report["admission_rule"] == "excess_energy" and report["admission_excess_floor"] == 70.0
    assert legacy.report()["admission_rule"] == "chi_square_margin" and legacy.report()["anomaly_margin"] == 1.25
    with pytest.raises(ValueError, match="admission_rule"):
        Round0ScenarioGenerator(seed=1, admission_rule="energy")
    with pytest.raises(ValueError, match="admission_excess_floor"):
        Round0ScenarioGenerator(seed=1, admission_excess_floor=0.0)
