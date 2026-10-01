"""The learned ranker (steps 4 and 5): policy-visible features, the JSON model, the acquisition deferral."""
from __future__ import annotations

import json
import math

import pytest

from psse_env.actions import CORRECT_MEASUREMENTS, CORRECT_PARAMETERS, GET_MEASUREMENT_CONTEXT, GET_PARAMETER_CONTEXT, GET_THREE_PHASE_CONTEXT
from psse_env.oracle.expert_policy import ExpertPolicyOracle
from psse_env.oracle.expert_variants import current_expert_variant, expert_variant_options
from psse_env.oracle.learned_ranker import (
    DEFAULT_RANKER_MODEL, DEFERRAL_EVIDENCE, FEATURE_SET, RANKER_CONTRACT, LearnedRanker, acquisition_deferral,
    policy_visible_features,
)
from psse_env.oracle.process_validity import ProcessValidityOracle

ACTIVE = "s0"


def _screen(*, suspected: bool, accepted, scores, best, ranked, voltage=()):
    kinds = {"hif": suspected, "voltage_meter": bool(voltage), "unexplained": False}
    report = {
        "status": "valid", "suspected": suspected, "outcome": ">".join(a["class"] for a in accepted) or ("hif" if suspected else None),
        "explained": not suspected and bool(accepted), "unexplained": False, "accepted_hypotheses": list(accepted),
        "voltage_meter_channels": list(voltage), "phasor_suspicion": kinds,
        "rounds": [{"winner": max(scores, key=scores.get), "clean_after_removal": None, "set_aside_channels": [],
                    "scores": scores, "best": best, "ranked": ranked}],
    }
    if suspected:
        report.update(branch_row0=best["hif"]["branch_row0"], line_index1=best["hif"]["branch_row0"] + 1,
                      alpha_from_from_bus=best["hif"].get("alpha_from_from_bus", 0.5), hif_variant="shunt")
    return report


def _policy(report, *, signatures=(), rejected=(), bus_count=14, phase=None):
    contexts = {"wls": {"state_id": ACTIVE, "state_hash": "h", "successful": True, "chi_square_alarm": True,
                        "normalized_residual_alarm": True, "max_normalized_residual": 9.5, "anomaly_breadth": 0.05,
                        "bus_count": bus_count, "hif_screen": report}}
    if phase is not None:
        contexts["three_phase"] = {"state_id": ACTIVE, "state_hash": "h", "request_attempted": True, **phase}
    return {
        "evidence_profile": "suspicion_gated_diagnostics", "active_state_id": ACTIVE, "candidate_state_id": None,
        "has_open_candidate": False, "unresolved_signatures": list(signatures), "remaining_anomaly_score": 4.2,
        "tried_action_signatures": [], "available_evidence": [], "fresh_context_evidence": contexts,
        "rejected_hypotheses": list(rejected), "accepted_corrections": [],
    }


MIMIC_SCREEN = _screen(
    suspected=True, accepted=[],
    scores={"meter": 30.0, "parameter": 5.0, "topology": -3.0, "hif": 38.0},
    best={"meter": {"channel_index0": 47, "J": 120.0}, "parameter": {"branch_row0": 4, "parameter": "X", "J": 150.0},
          "topology": {"branch_row0": 4, "J": 160.0}, "hif": {"branch_row0": 4, "alpha_from_from_bus": 0.5, "J": 110.0}},
    ranked={"meter": [{"channel_index0": 47, "J": 120.0}, {"channel_index0": 87, "J": 125.0}],
            "parameter": [{"branch_row0": 4, "parameter": "X", "J": 150.0}], "topology": [{"branch_row0": 4, "J": 160.0}],
            "hif": [{"branch_row0": 4, "J": 110.0}, {"branch_row0": 5, "J": 140.0}]},
)
MIMIC_SIGNATURES = ("wls_residual_outlier_dominant index=47 channel=Pf", "wls_residual_outlier_dominant index=87 channel=Pt",
                    "wls_branch_multiplier line_status_or_parameter line=5", "wls_hif_suspected line=5")


def _model(*, needs_aux_logit: float, threshold: float = 0.1) -> LearnedRanker:
    """A one-feature logistic model whose probability is fixed by its intercept (the feature has zero weight)."""
    names = sorted(policy_visible_features(_policy(MIMIC_SCREEN, signatures=MIMIC_SIGNATURES), branch_count=20))
    zeros = [0.0] * len(names)
    spec = {"type": "logistic", "mean": zeros, "scale": [1.0] * len(names), "coef": zeros, "intercept": needs_aux_logit,
            "platt": {"coef": None, "intercept": None}, "thresholds": {"at_rule_recall": threshold}}
    return LearnedRanker({"contract": RANKER_CONTRACT, "feature_set": FEATURE_SET, "features": names,
                          "system": {"case_id": "case14", "bus_count": 14, "branch_count": 20},
                          "targets": {"needs_aux": spec}})


def test_policy_visible_features_read_the_ledger_signatures_and_screen():
    features = policy_visible_features(_policy(MIMIC_SCREEN, signatures=MIMIC_SIGNATURES), branch_count=20)
    assert features["screen_valid"] == 1.0 and features["suspected"] == 1.0 and features["explained"] == 0.0
    assert features["n_residual_sigs"] == 2.0 and features["meas_dominant"] == 1.0 and features["branch_sig"] == 1.0
    assert features["sig_Pf"] == 1.0 and features["sig_Pt"] == 1.0 and features["sig1_Pf"] == 1.0
    # Channels 47 (Pf) and 87 (Pt) are the two ends of branch row 5 on IEEE 14 (3*14 = 42, 42 + 2*20 = 82).
    assert features["flow_pair_sig"] == 1.0
    assert policy_visible_features(_policy(MIMIC_SCREEN, signatures=MIMIC_SIGNATURES))["flow_pair_sig"] == 0.0
    assert features["w1_hif"] == 1.0 and features["s1_margin"] == pytest.approx(8.0)
    assert features["j1_hif_vs_meter_log"] == pytest.approx(math.log(111.0 / 121.0))
    assert features["hif_rank_gap_log"] == pytest.approx(math.log(141.0 / 111.0))
    assert features["max_rn_log1p"] == pytest.approx(math.log1p(9.5)) and features["anomaly_score_log"] == pytest.approx(math.log(4.2))
    # Without a current screen every screen feature is at its missing value.
    quiet = policy_visible_features({"active_state_id": ACTIVE, "fresh_context_evidence": {"wls": {"successful": False}}})
    assert quiet["screen_valid"] == 0.0 and quiet["s1_hif"] == -50.0 and quiet["n_residual_sigs"] == 0.0


def test_model_contract_probability_and_network_guard():
    ranker = _model(needs_aux_logit=-3.0)
    policy = _policy(MIMIC_SCREEN, signatures=MIMIC_SIGNATURES)
    assert ranker.probability(policy, "needs_aux") == pytest.approx(1.0 / (1.0 + math.exp(3.0)))
    assert ranker.probability(policy, "has_hif") is None  # no such target in this model
    assert ranker.probability(_policy(MIMIC_SCREEN, bus_count=57), "needs_aux") is None  # another network
    assert ranker.threshold("needs_aux") == 0.1 and ranker.threshold("needs_aux", "at_rule_false_rate") is None
    with pytest.raises(ValueError):
        LearnedRanker({"contract": "other", "feature_set": FEATURE_SET, "features": ["x"], "targets": {}})


def test_gradient_boosted_export_is_evaluated_like_sklearn():
    sklearn = pytest.importorskip("sklearn")
    from sklearn.ensemble import HistGradientBoostingClassifier
    import numpy as np

    from research.hypothesis_ranking.ranker import _hgb_spec

    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 3))
    y = (x[:, 0] + 0.5 * x[:, 1] ** 2 > 0.3).astype(int)
    model = HistGradientBoostingClassifier(max_iter=25, max_leaf_nodes=7, random_state=0).fit(x, y)
    spec = _hgb_spec(model)
    spec.update(platt={"coef": None, "intercept": None}, thresholds={})
    payload = {"contract": RANKER_CONTRACT, "feature_set": FEATURE_SET, "features": ["a", "b", "c"],
               "system": {"case_id": "toy", "bus_count": 14, "branch_count": 20}, "targets": {"needs_aux": spec}}
    ranker = LearnedRanker(payload)
    from psse_env.oracle.learned_ranker import _hgb_raw_prediction, _sigmoid

    expected = model.predict_proba(x[:50])[:, 1]
    got = [_sigmoid(_hgb_raw_prediction([float(v) for v in row], spec)) for row in x[:50]]
    assert max(abs(e - g) for e, g in zip(expected, got)) < 1e-9
    assert ranker.targets["needs_aux"]["type"] == "hgb"


def test_deferral_conditions():
    policy = _policy(MIMIC_SCREEN, signatures=MIMIC_SIGNATURES)
    low = _model(needs_aux_logit=-5.0)   # p ~ 0.007 < 0.1: an auxiliary stream is unlikely
    high = _model(needs_aux_logit=0.0)   # p = 0.5 >= 0.1: acquire at once
    deferral = acquisition_deferral(policy, low)
    assert deferral is not None and deferral["family"] == "measurement" and deferral["target"] == 47
    assert deferral["suspicion"] == "hif" and deferral["probability"] < deferral["threshold"]
    assert acquisition_deferral(policy, high) is None
    assert acquisition_deferral(policy, None) is None
    # Spent once per state: a rejected candidate on this state ends the deferral.
    rejected = {"candidate_state_id": "c1", "candidate_parent_id": ACTIVE,
                "source_action": {"tool": CORRECT_MEASUREMENTS, "arguments": {"state_id": ACTIVE, "suspect_group": [47]}}}
    assert acquisition_deferral(_policy(MIMIC_SCREEN, signatures=MIMIC_SIGNATURES, rejected=[rejected]), low) is None
    # Phasors already requested on this state: the acquisition stands.
    assert acquisition_deferral(_policy(MIMIC_SCREEN, signatures=MIMIC_SIGNATURES, phase={"nlm_attempted": False}), low) is None
    # Another network: no model, no deferral.
    assert acquisition_deferral(_policy(MIMIC_SCREEN, signatures=MIMIC_SIGNATURES, bus_count=57), low) is None
    # D3: a voltage-channel leading target waits for the phasors whatever the model says.
    voltage_screen = _screen(
        suspected=False, accepted=[{"class": "meter", "channel_index0": 3}], voltage=[3],
        scores={"meter": 40.0, "parameter": 2.0, "topology": -1.0, "hif": -4.0},
        best={"meter": {"channel_index0": 3, "J": 90.0}, "parameter": {"branch_row0": 1, "parameter": "R", "J": 140.0},
              "topology": {"branch_row0": 2, "J": 150.0}, "hif": {"branch_row0": 1, "alpha_from_from_bus": 0.4, "J": 145.0}},
        ranked={"meter": [{"channel_index0": 3, "J": 90.0}], "parameter": [], "topology": [], "hif": []},
    )
    assert acquisition_deferral(_policy(voltage_screen), low) is None
    # No suspicion at all: nothing to defer.
    explained = _screen(
        suspected=False, accepted=[{"class": "meter", "channel_index0": 47}],
        scores={"meter": 40.0, "parameter": 2.0, "topology": -1.0, "hif": -4.0},
        best={"meter": {"channel_index0": 47, "J": 90.0}, "parameter": {"branch_row0": 1, "parameter": "R", "J": 140.0},
              "topology": {"branch_row0": 2, "J": 150.0}, "hif": {"branch_row0": 1, "alpha_from_from_bus": 0.4, "J": 145.0}},
        ranked={"meter": [{"channel_index0": 47, "J": 90.0}], "parameter": [], "topology": [], "hif": []},
    )
    assert acquisition_deferral(_policy(explained), low) is None


def test_c5_the_screen_suspicion_no_longer_blocks_balanced_corrections():
    oracle = ProcessValidityOracle()
    policy = _policy(MIMIC_SCREEN, signatures=MIMIC_SIGNATURES)
    edit = {"tool": CORRECT_MEASUREMENTS, "arguments": {"state_id": ACTIVE, "suspect_group": [47]}}
    verdict = oracle.check(policy, edit)
    assert verdict["error_detail"] != "measurement_correction_route_not_actionable"
    assert not (verdict["error_code"] == "correction_route_not_actionable" and "waveform" in str(verdict["error_detail"]))
    # A sensor-reported waveform signature still closes the routes.
    unbalance = _policy(MIMIC_SCREEN, signatures=[*MIMIC_SIGNATURES, "three_phase_unbalance_detected bus=4"])
    blocked = oracle.check(unbalance, edit)
    assert blocked["error_code"] == "correction_route_not_actionable"
    # Once phasors were requested on the state, the suspicion is theirs to settle and blocks again.
    requested = _policy(MIMIC_SCREEN, signatures=MIMIC_SIGNATURES, phase={"nlm_attempted": False})
    tested = oracle.check(requested, edit)
    assert tested["error_code"] == "correction_route_not_actionable", tested
    assert tested["error_detail"] == "measurement_fundamental_route_blocked_by_waveform_anomaly", tested


def test_the_ranked_expert_tries_the_balanced_hypothesis_first_then_acquires():
    low = _model(needs_aux_logit=-5.0)
    policy = _policy(MIMIC_SCREEN, signatures=MIMIC_SIGNATURES)
    ranked = ExpertPolicyOracle(hypothesis_ledger=True, learned_ranker=low)
    ledger_only = ExpertPolicyOracle(hypothesis_ledger=True)
    first_ranked = ranked.next_action_proposals(policy)
    first_ledger = ledger_only.next_action_proposals(policy)
    assert first_ledger and first_ledger[0].action["tool"] == GET_THREE_PHASE_CONTEXT
    assert first_ranked and first_ranked[0].action["tool"] != GET_THREE_PHASE_CONTEXT
    assert first_ranked[0].action["tool"] in {GET_MEASUREMENT_CONTEXT, CORRECT_MEASUREMENTS}
    assert any(code.startswith(DEFERRAL_EVIDENCE) for code in first_ranked[0].evidence_codes)
    # After a rejected candidate on this state the acquisition follows unchanged.
    rejected = {"candidate_state_id": "c1", "candidate_parent_id": ACTIVE,
                "source_action": {"tool": CORRECT_MEASUREMENTS, "arguments": {"state_id": ACTIVE, "suspect_group": [47]}},
                "verification_summary": {"global_progress": 0.01}}
    after = ranked.next_action_proposals(_policy(MIMIC_SCREEN, signatures=MIMIC_SIGNATURES, rejected=[rejected]))
    assert after and after[0].action["tool"] == GET_THREE_PHASE_CONTEXT
    # A confident model acquires at once, like the ledger expert.
    confident = ExpertPolicyOracle(hypothesis_ledger=True, learned_ranker=_model(needs_aux_logit=3.0))
    assert confident.next_action_proposals(policy)[0].action["tool"] == GET_THREE_PHASE_CONTEXT
    with pytest.raises(ValueError):
        ExpertPolicyOracle(learned_ranker=low)


def test_expert_variants_and_the_tracked_model(monkeypatch):
    monkeypatch.delenv("PSSE_EXPERT_VARIANT", raising=False)
    assert current_expert_variant() == "baseline" and expert_variant_options() == {}
    monkeypatch.setenv("PSSE_EXPERT_VARIANT", "ledger")
    assert expert_variant_options() == {"hypothesis_ledger": True}
    monkeypatch.setenv("PSSE_EXPERT_VARIANT", "ledger_ranked")
    options = expert_variant_options()
    assert options["hypothesis_ledger"] is True and options["learned_ranker"] == str(DEFAULT_RANKER_MODEL)
    monkeypatch.setenv("PSSE_EXPERT_VARIANT", "nonsense")
    with pytest.raises(ValueError):
        current_expert_variant()
    if not DEFAULT_RANKER_MODEL.is_file():
        pytest.skip("tracked ranker export not present")
    ranker = LearnedRanker.from_json(DEFAULT_RANKER_MODEL)
    assert ranker.bus_count == 14 and ranker.branch_count == 20 and "needs_aux" in ranker.targets
    payload = json.loads(DEFAULT_RANKER_MODEL.read_text(encoding="utf-8"))
    assert payload["targets"]["needs_aux"]["thresholds"]["at_rule_recall"] > 0
    probability = ranker.probability(_policy(MIMIC_SCREEN, signatures=MIMIC_SIGNATURES), "needs_aux")
    assert probability is not None and 0.0 <= probability <= 1.0
