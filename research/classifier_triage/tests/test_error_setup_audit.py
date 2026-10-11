"""The error-setup audit's helpers: the load-unbalance factor and the alarm signature."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import chi2

from research.classifier_triage import error_setup_audit as audit
from research.classifier_triage import features


def test_load_unbalance_factor_is_zero_when_balanced_and_one_for_a_single_phase():
    assert audit.load_unbalance_factor({"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}) == pytest.approx(0.0, abs=1e-12)
    assert audit.load_unbalance_factor([1.0, 0.0, 0.0]) == pytest.approx(1.0)
    assert audit.load_unbalance_factor([0.5, 0.5, 0.0]) == pytest.approx(0.5)
    # the factor is a ratio: scaling the fractions does not change it
    assert audit.load_unbalance_factor([0.6, 0.3, 0.1]) == pytest.approx(audit.load_unbalance_factor([6, 3, 1]))


def test_signature_counts_channels_assets_and_the_admitting_test():
    net = features.network("case14")
    nb, nl = int(net["nb"]), int(net["nl"])
    dof = 3 * nb + 4 * nl - (2 * nb - 1)
    residual = np.zeros(3 * nb + 4 * nl)
    residual[3 * nb + 2] = 6.0          # Pf of branch 2
    residual[3 * nb + 2 * nl + 2] = -4.0  # Pt of the same branch
    residual[nb + 5] = 3.5               # Pinj of bus 5 (not zero-injection)
    payload = {"signed_normalized_residual": residual, "lambda_normalized": np.zeros(2 * nl), "objective": float(dof + 50.0), "dof": dof}
    sig = audit.signature("case14", payload)
    assert sig["channels_gt3"] == 3 and sig["assets_gt3"] == 2 and sig["max_residual"] == 6.0
    assert sig["top2_share"] == pytest.approx((36 + 16) / (36 + 16 + 12.25))
    assert sig["vm_share_gt3"] == 0.0 and sig["reactive_share_gt3"] == 0.0
    assert sig["excess_objective"] == pytest.approx(50.0)
    threshold = chi2.ppf(0.99, dof)
    assert sig["residual_admitted"] == 1.0 and sig["residual_alarm"] == 1.0
    assert sig["chi_admitted"] == float(dof + 50.0 > 1.25 * threshold) and sig["chi_alarm"] == float(dof + 50.0 >= threshold)
