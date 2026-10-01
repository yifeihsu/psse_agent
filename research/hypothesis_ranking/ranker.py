"""Step 4: a learned ranker on the balanced evidence, against the physics screen (hypothesis-ranking plan).

Trains gradient-boosted and logistic classifiers on the step-1 states as
re-analyzed by the step-2 screen (``build_dataset --reanalyze``): every
feature is a number the policy sees (the WLS ledger, the compact screen
report, the class tests), never truth.  Splits are by physical parent, as the
dataset recorded them: train for fitting, calibration for thresholds and
probability calibration, test for every number reported.  Parent-bootstrap
intervals accompany each comparison.

The questions, as step 3 narrowed them:

1. Acquisition: can a learned ``needs_phasors`` score reach the roots that
   need phase-resolved measurements (HIF, unbalance) with fewer requests on
   roots that do not, than the shipped rule (HIF won, voltage-meter pick, or
   unexplained)?  Compared at the rule's false-acquisition rate and at the
   rule's recall.
2. HIF against the same-sign flow-meter pair: at the screen's HIF recall, how
   many mimics does a learned HIF score flag?
3. Unbalance against a voltage-meter error among roots with a voltage-meter
   pick: is there any separation on balanced evidence (AUC)?
4. Adjacent-line ambiguity: among parameter roots whose true line is one of
   the multiplier ranking's top two, does a pairwise model choose better than
   "take the top line"?
5. Transfer: the IEEE-14 rankers applied unchanged to the IEEE 57 study roots.

    python -m research.hypothesis_ranking.ranker --dataset-dir output/hypothesis_ranking_20260930/ieee14_v3 --transfer-dir output/hypothesis_ranking_20260930/ieee57_v3 --output-dir output/hypothesis_ranking_20260930/ranker

Step 5 adds the deployable form.  ``--feature-set policy`` computes the
features the expert can compute at runtime from the policy observation alone
(``psse_env.oracle.learned_ranker.policy_visible_features`` on a pseudo
observation built from the record: the WLS ledger fields, the ``wls_``
signatures the provider would mint, the compact screen report), and
``--export-model PATH`` writes the models of ``needs_aux``, ``has_hif`` and
``has_unbalance`` (the gradient-boosted trees by default, ``--export-model-type
logistic`` for the standardized logistic models) with their Platt scaling and
the operating-point thresholds as JSON for ``LearnedRanker``; the export is
checked against sklearn's own predictions before it is written:

    python -m research.hypothesis_ranking.ranker --dataset-dir output/hypothesis_ranking_20260930/ieee14_v3 --transfer-dir output/hypothesis_ranking_20260930/ieee57_v3 --output-dir output/hypothesis_ranking_20260930/ranker_policy --feature-set policy --export-model psse_env/oracle/models/learned_ranker_ieee14_20261001.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from psse_env.oracle.learned_ranker import FEATURE_SET, RANKER_CONTRACT, policy_visible_features  # noqa: E402
from psse_env.providers.hif_screen import compact_screen_report  # noqa: E402
from research.hypothesis_ranking.features import json_safe  # noqa: E402

CLASSES = ("meter", "parameter", "topology", "hif")
CHANNELS = ("Vm", "P", "Q", "Pf", "Qf", "Pt", "Qt")
TARGETS = ("has_measurement", "has_parameter", "has_topology", "has_hif", "has_unbalance", "has_harmonic",
           "needs_phasors", "needs_spectra", "needs_aux")
NEEDS_NO_PHASORS = {"measurement", "multi_measurement", "parameter", "measurement+parameter", "topology",
                    "measurement+topology", "healthy_window", "mimic_flow_pair_same_sign", "mimic_flow_pair_opposite_sign"}
BOOTSTRAP_DRAWS = 1000
#: Offline channel blocks (features.CHANNEL_BLOCKS) as the WLS ledger names them.
LEDGER_BLOCKS = {"Vm": "Vm", "P": "Pinj", "Q": "Qinj", "Pf": "Pf", "Qf": "Qf", "Pt": "Pt", "Qt": "Qt"}
#: The provider's signature budget (MatpowerDeploymentProviders top_k / residual_threshold / lambda_threshold).
SIGNATURE_TOP_K = 5
SIGNATURE_MIN_ABS = 3.0
EXPORT_TARGETS = ("needs_aux", "has_hif", "has_unbalance")


# ------------------------------------------------------------------ records


def _read(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def alarmed(record: Mapping[str, Any]) -> bool:
    return bool(((record.get("analysis") or {}).get("wls") or {}).get("alarm"))


def labels(record: Mapping[str, Any]) -> dict[str, int]:
    truth = record.get("truth") or {}
    has = {
        "has_measurement": int(bool(truth.get("measurement"))),
        "has_parameter": int(bool(truth.get("parameter"))),
        "has_topology": int(bool(truth.get("topology"))),
        "has_hif": int(bool(truth.get("hif"))),
        "has_unbalance": int(bool(truth.get("unbalance"))),
        "has_harmonic": int(bool(truth.get("harmonic"))),
    }
    has["needs_phasors"] = int(has["has_hif"] or has["has_unbalance"])
    has["needs_spectra"] = int(has["has_harmonic"])
    # Under C2 a harmonic root reaches its spectra only through balanced phasors,
    # so the acquisition target counts every root that needs an auxiliary stream.
    has["needs_aux"] = int(has["has_hif"] or has["has_unbalance"] or has["has_harmonic"])
    return has


def _log(x: float, floor: float = 1e-6) -> float:
    return float(math.log(max(float(x), floor)))


def features(record: Mapping[str, Any]) -> dict[str, float]:
    """Fixed-length numeric view of one state's observable evidence."""
    analysis = record.get("analysis") or {}
    wls = analysis.get("wls") or {}
    screen = analysis.get("screen") or {}
    f: dict[str, float] = {}
    max_rn = float(wls.get("max_normalized_residual") or 0.0)
    max_lambda = float(wls.get("max_abs_branch_multiplier") or 0.0)
    f["chi_ratio_log"] = _log(wls.get("chi_square_ratio") or 1e-6)
    f["max_rn_log1p"] = math.log1p(max_rn)
    f["cnt_gt3"] = float(wls.get("count_abs_residual_gt3") or 0)
    f["cnt_gt4"] = float(wls.get("count_abs_residual_gt4") or 0)
    f["max_lambda_log1p"] = math.log1p(max_lambda)
    f["meas_dominant"] = float(bool(wls.get("measurement_dominant")))
    f["branch_dominant"] = float(bool(wls.get("branch_dominant")))
    ratio = wls.get("branch_ranking_dominance_ratio")
    f["dominance_ratio"] = float(min(ratio, 10.0)) if isinstance(ratio, (int, float)) else 10.0
    f["rn_over_lambda_log"] = _log((max_rn + 1e-6) / (max_lambda + 1e-6))
    top = sorted(wls.get("top_residuals") or [], key=lambda item: -abs(float(item.get("value") or 0.0)))
    for i in range(6):
        item = top[i] if i < len(top) else None
        value = float(item["value"]) if item else 0.0
        f[f"r{i}_log1p"] = math.log1p(abs(value))
        f[f"r{i}_sign"] = float(np.sign(value))
    counts = Counter(str(item.get("channel")) for item in top[:6])
    for channel in CHANNELS:
        f[f"top6_{channel}"] = float(counts.get(channel, 0))
        f[f"top1_{channel}"] = float(bool(top) and str(top[0].get("channel")) == channel)
    f["vm_frac_top3"] = sum(1 for item in top[:3] if item.get("channel") == "Vm") / 3.0
    lambdas = wls.get("top_branch_multipliers") or []
    for i in range(6):
        f[f"l{i}_log1p"] = math.log1p(float(lambdas[i]["value"])) if i < len(lambdas) else 0.0
    f["lambda_top_gap_log"] = (
        _log((float(lambdas[0]["value"]) + 1e-6) / (float(lambdas[1]["value"]) + 1e-6)) if len(lambdas) > 1 else 3.0
    )
    # Two flow meters of one branch among the top residuals (the HIF mimic).
    by_asset: dict[tuple[int, str], dict[str, float]] = defaultdict(dict)
    for item in top[:6]:
        channel = str(item.get("channel"))
        if channel in ("Pf", "Pt", "Qf", "Qt"):
            by_asset[(int(item.get("asset_row0", -1)), channel[0])][channel[1]] = float(item["value"])
    pair = next((ends for ends in by_asset.values() if "f" in ends and "t" in ends), None)
    f["flow_pair"] = float(pair is not None)
    f["flow_pair_same_sign"] = float(pair is not None and np.sign(pair["f"]) == np.sign(pair["t"]))
    valid = screen.get("status") == "valid"
    f["screen_valid"] = float(valid)
    rounds = [r for r in (screen.get("rounds") or []) if isinstance(r, Mapping) and r.get("scores")]
    first = rounds[0] if rounds else {}
    scores = first.get("scores") or {}
    for name in CLASSES:
        f[f"s1_{name}"] = float(scores.get(name, -50.0))
    ordered = sorted(float(v) for v in scores.values())
    f["s1_margin"] = (ordered[-1] - ordered[-2]) if len(ordered) > 1 else 0.0
    f["s1_hif_minus_meter"] = float(scores.get("hif", -50.0)) - float(scores.get("meter", -50.0))
    winner = str(first.get("winner") or "")
    for name in CLASSES:
        f[f"w1_{name}"] = float(winner == name)
    accepted = [str(item.get("class")) for item in (screen.get("accepted_hypotheses") or []) if isinstance(item, Mapping)]
    f["n_accepted"] = float(len(accepted))
    for position in (0, 1):
        for name in CLASSES:
            f[f"acc{position}_{name}"] = float(len(accepted) > position and accepted[position] == name)
    f["explained"] = float(bool(screen.get("explained")))
    f["unexplained"] = float(bool(screen.get("unexplained")))
    f["suspected"] = float(bool(screen.get("suspected")))
    f["vm_channels_n"] = float(len(screen.get("voltage_meter_channels") or []))
    f["n_rounds"] = float(len(rounds))
    tests = first.get("class_tests") or {}
    classes = tests.get("classes") or {}
    for name in CLASSES:
        entry = classes.get(name) or {}
        refit = entry.get("J_refit")
        threshold = entry.get("chi2_threshold_after")
        f[f"ct1_{name}_jratio_log"] = _log((float(refit) + 1.0) / (float(threshold) + 1.0)) if refit is not None and threshold else 3.0
        f[f"ct1_{name}_explains"] = float(bool(entry.get("explains_alarm")))
        f[f"ct1_{name}_maxrn_log1p"] = math.log1p(float(entry.get("max_normalized_residual_after") or 50.0))
    f["ct1_any_explains"] = float(bool(tests.get("any_class_explains_alarm")))
    f["ct1_winner_explains"] = float(bool(tests.get("winner_explains_alarm")))
    plus_one = (tests.get("extras") or {}).get("winner_plus_one_meter") or {}
    f["ct1_plus_one_explains"] = float(bool(plus_one.get("explains_alarm")))
    final = screen.get("final") or {}
    f["final_alarm"] = float(bool(final.get("alarm")))
    f["final_jratio_log"] = _log((float(final.get("J0") or 0.0) + 1.0) / (float(final.get("chi2_threshold") or 1.0) + 1.0))
    f["final_maxrn_log1p"] = math.log1p(float(final.get("max_normalized_residual") or 0.0))
    best = first.get("best") or {}
    hif_best = best.get("hif") or {}
    f["hif_alpha"] = float(hif_best.get("alpha") or hif_best.get("alpha_grid") or 0.0)
    f["hif_G_log1p"] = math.log1p(max(float(hif_best.get("shunt_conductance_pu") or 0.0), 0.0))
    for name in CLASSES:
        ranked = (best.get(name) or {}).get("ranked") or []
        if len(ranked) > 1 and ranked[0].get("J") is not None and ranked[1].get("J") is not None:
            f[f"{name}_rank_gap_log"] = _log((float(ranked[1]["J"]) + 1.0) / (float(ranked[0]["J"]) + 1.0))
        else:
            f[f"{name}_rank_gap_log"] = 0.0
    return f


def observation_from_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """The policy observation the record's state would carry on its first WLS.

    Built from the offline analysis the way the provider and the environment
    build the real one: the WLS ledger fields, the ``wls_`` residual
    signatures (top five normalized residuals at or above 3.0, dominance tag
    from the residual-versus-multiplier rule), one branch-multiplier
    signature when any multiplier reaches 3.0, the residual breadth and its
    dominant block, the remaining anomaly score, the bus count, and the
    compact screen report (the same compaction the provider applies).
    """
    analysis = record.get("analysis") or {}
    wls = analysis.get("wls") or {}
    screen = analysis.get("screen") or {}
    nb, nl = int(analysis.get("nb") or 0), int(analysis.get("nl") or 0)
    nz = 3 * nb + 4 * nl
    top = sorted(wls.get("top_residuals") or [], key=lambda item: -abs(float(item.get("value") or 0.0)))
    max_rn = float(wls.get("max_normalized_residual") or 0.0)
    max_lambda = float(wls.get("max_abs_branch_multiplier") or 0.0)
    measurement_dominant = bool(wls.get("measurement_dominant"))
    branch_dominant = bool(wls.get("branch_dominant"))
    signatures: list[str] = []
    residual_tag = "wls_residual_outlier_dominant" if measurement_dominant else "wls_residual_outlier"
    for item in top[:SIGNATURE_TOP_K]:
        if abs(float(item["value"])) < SIGNATURE_MIN_ABS:
            break
        signatures.append(f"{residual_tag} index={int(item['index'])} channel={LEDGER_BLOCKS[str(item['channel'])]}")
    if max_lambda >= SIGNATURE_MIN_ABS:
        branch_tag = ("wls_branch_multiplier_dominant line_status_or_parameter" if branch_dominant
                      else "wls_branch_multiplier line_status_or_parameter")
        rows = [int(item["branch_row0"]) for item in (wls.get("top_branch_multipliers") or [])
                if float(item.get("value") or 0.0) >= SIGNATURE_MIN_ABS][:SIGNATURE_TOP_K]
        signatures.extend(f"{branch_tag} line={row + 1}" for row in rows)
    compact = compact_screen_report(screen) if screen.get("status") else None
    if compact is not None and screen.get("status") == "valid":
        if compact["suspected"]:
            signatures.append(f"wls_hif_suspected line={compact['line_index1']}")
        if compact["voltage_meter_channels"]:
            signatures.append("wls_voltage_meter_suspected channels=" + ",".join(str(c) for c in compact["voltage_meter_channels"]))
        if compact["unexplained"]:
            signatures.append("wls_unexplained_balanced_discrepancy")
    ledger = {
        "state_id": "s0", "state_hash": "offline", "successful": bool(wls.get("success", True)),
        "evidence_source": "offline_study", "anomalous": bool(signatures),
        "chi_square_alarm": bool(wls.get("chi_square_alarm")),
        "normalized_residual_alarm": bool(wls.get("normalized_residual_alarm")),
        "normalized_residual_threshold": wls.get("normalized_residual_threshold"),
        "max_normalized_residual": max_rn,
        "anomaly_breadth": (int(wls.get("count_abs_residual_gt3") or 0) / nz) if nz else 0.0,
        "dominant_residual_block": LEDGER_BLOCKS[str(top[0]["channel"])] if top else None,
        "bus_count": nb,
    }
    if compact is not None:
        ledger["hif_screen"] = compact
    return {
        "evidence_profile": "suspicion_gated_diagnostics", "active_state_id": "s0", "candidate_state_id": None,
        "has_open_candidate": False, "unresolved_signatures": signatures,
        "remaining_anomaly_score": wls.get("remaining_anomaly_score"),
        "fresh_context_evidence": {"wls": ledger}, "rejected_hypotheses": [], "accepted_corrections": [],
        "tried_action_signatures": [], "available_evidence": [],
    }


def policy_features(record: Mapping[str, Any]) -> dict[str, float]:
    """The runtime feature view of one record (what ``LearnedRanker`` computes in the expert)."""
    nl = int((record.get("analysis") or {}).get("nl") or 0) or None
    return policy_visible_features(observation_from_record(record), branch_count=nl)


def rule_v3(record: Mapping[str, Any]) -> int:
    screen = (record.get("analysis") or {}).get("screen") or {}
    if screen.get("status") != "valid":
        return 0
    kinds = screen.get("phasor_suspicion") or {}
    return int(bool(screen.get("suspected") or kinds.get("unexplained") or kinds.get("voltage_meter")
                    or screen.get("voltage_meter_channels")))


def rule_hif(record: Mapping[str, Any]) -> int:
    screen = (record.get("analysis") or {}).get("screen") or {}
    return int(screen.get("status") == "valid" and bool(screen.get("suspected")))


def voltage_meter_pick(record: Mapping[str, Any]) -> bool:
    screen = (record.get("analysis") or {}).get("screen") or {}
    return bool(screen.get("status") == "valid" and screen.get("voltage_meter_channels"))


# ------------------------------------------------------------------ dataset


def load_dataset(dataset_dir: Path, *, include_children: bool, feature_set: str = "offline") -> list[dict[str, Any]]:
    """Alarmed, valid-screen states with their features, labels, split and parent."""
    extract = features if feature_set == "offline" else policy_features
    rows: list[dict[str, Any]] = []
    sources = [("roots.jsonl", "root"), ("healthy_alarms.jsonl", "healthy"), ("mimic.jsonl", "mimic")]
    if include_children:
        sources.append(("children.jsonl", "child"))
    for name, kind in sources:
        for record in _read(dataset_dir / name):
            if not alarmed(record):
                continue
            screen = (record.get("analysis") or {}).get("screen") or {}
            if screen.get("status") != "valid":
                continue
            rows.append({
                "id": record.get("root_id"), "kind": kind, "family": record.get("family"),
                "parent": record.get("parent_id"), "split": record.get("split") or "test",
                "features": extract(record), "labels": labels(record),
                "rule_v3": rule_v3(record), "rule_hif": rule_hif(record), "vm_pick": voltage_meter_pick(record),
                "needs_no_phasors_family": record.get("family") in NEEDS_NO_PHASORS,
                "record": record,
            })
    return rows


def matrix(rows: Sequence[Mapping[str, Any]], names: Sequence[str]) -> np.ndarray:
    return np.asarray([[float(row["features"].get(name, 0.0)) for name in names] for row in rows], dtype=float)


# ------------------------------------------------------------------ models


def fit_models(x_train: np.ndarray, y_train: np.ndarray, seed: int) -> dict[str, Any]:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    models: dict[str, Any] = {}
    if len(np.unique(y_train)) < 2:
        return models
    gbm = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=10,
                                         l2_regularization=1.0, random_state=seed)
    gbm.fit(x_train, y_train)
    models["gbm"] = gbm
    logit = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))
    logit.fit(x_train, y_train)
    models["logit"] = logit
    return models


class PlattScaler:
    """Platt scaling fitted on the calibration split (logistic in the logit of the model probability)."""

    def __init__(self, coef: float | None, intercept: float | None) -> None:
        self.coef = coef
        self.intercept = intercept

    def __call__(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=float)
        if self.coef is None:
            return scores
        z = np.log(np.clip(scores, 1e-6, 1 - 1e-6) / np.clip(1 - scores, 1e-6, 1))
        return 1.0 / (1.0 + np.exp(-(self.coef * z + self.intercept)))


def platt(scores_cal: np.ndarray, y_cal: np.ndarray) -> PlattScaler:
    from sklearn.linear_model import LogisticRegression

    if len(np.unique(y_cal)) < 2:
        return PlattScaler(None, None)
    z = np.log(np.clip(scores_cal, 1e-6, 1 - 1e-6) / np.clip(1 - scores_cal, 1e-6, 1))
    model = LogisticRegression(C=1e6, max_iter=1000).fit(z.reshape(-1, 1), y_cal)
    return PlattScaler(float(model.coef_[0][0]), float(model.intercept_[0]))


def _logistic_spec(pipeline: Any) -> dict[str, Any]:
    scaler, logit = pipeline[0], pipeline[-1]
    scale = np.where(np.asarray(scaler.scale_, dtype=float) > 0, np.asarray(scaler.scale_, dtype=float), 1.0)
    return {"type": "logistic", "mean": [float(v) for v in scaler.mean_], "scale": [float(v) for v in scale],
            "coef": [float(v) for v in logit.coef_[0]], "intercept": float(logit.intercept_[0])}


def _hgb_spec(model: Any) -> dict[str, Any]:
    """sklearn's HistGradientBoostingClassifier as plain node lists (binary, numerical splits only)."""
    trees: list[list[dict[str, Any]]] = []
    for predictors in model._predictors:
        nodes = predictors[0].nodes
        if nodes["is_categorical"].any():
            raise ValueError("categorical splits are not exportable")
        trees.append([
            {"leaf": bool(node["is_leaf"]), "value": float(node["value"]), "feature": int(node["feature_idx"]),
             "threshold": float(node["num_threshold"]), "left": int(node["left"]), "right": int(node["right"]),
             "missing_left": bool(node["missing_go_to_left"])}
            for node in nodes
        ])
    return {"type": "hgb", "baseline": float(np.ravel(model._baseline_prediction)[0]), "trees": trees}


def export_models(path: Path, *, model_type: str, names: Sequence[str], models: Mapping[str, Mapping[str, Any]],
                  calibrators: Mapping[str, PlattScaler], thresholds: Mapping[str, Mapping[str, float]],
                  test_metrics: Mapping[str, Any], system: Mapping[str, Any], provenance: Mapping[str, Any],
                  check_x: np.ndarray) -> dict[str, Any]:
    """Write the per-target models as the JSON ``LearnedRanker`` reads, after checking the export reproduces sklearn."""
    from psse_env.oracle.learned_ranker import LearnedRanker

    targets: dict[str, Any] = {}
    fitted_key = "gbm" if model_type == "hgb" else "logit"
    for target in EXPORT_TARGETS:
        fitted = (models.get(target) or {}).get(fitted_key)
        if fitted is None:
            continue
        spec = _hgb_spec(fitted) if model_type == "hgb" else _logistic_spec(fitted)
        calibrator = calibrators.get(target) or PlattScaler(None, None)
        spec.update(platt={"coef": calibrator.coef, "intercept": calibrator.intercept},
                    thresholds=dict(thresholds.get(target) or {}), test=json_safe(test_metrics.get(target) or {}))
        targets[target] = spec
    payload = {"contract": RANKER_CONTRACT, "feature_set": FEATURE_SET, "features": list(names), "system": dict(system),
               "targets": targets, "provenance": json_safe(provenance)}
    ranker = LearnedRanker(payload, source="export-check")
    for target, spec in targets.items():
        fitted = models[target][fitted_key]
        calibrator = calibrators.get(target) or PlattScaler(None, None)
        expected = calibrator(fitted.predict_proba(check_x)[:, 1])
        got = np.asarray([_runtime_probability(ranker, spec, row) for row in check_x], dtype=float)
        worst = float(np.max(np.abs(expected - got))) if len(check_x) else 0.0
        if worst > 1e-6:
            raise RuntimeError(f"exported {model_type} model for {target} deviates from sklearn by {worst:.3g}")
        spec["export_check_max_abs_difference"] = worst
    path.parent.mkdir(parents=True, exist_ok=True)
    # Compact: the gradient-boosted export is a few megabytes of node lists.
    path.write_text(json.dumps(json_safe(payload), separators=(",", ":"), sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _runtime_probability(ranker: Any, spec: Mapping[str, Any], x: np.ndarray) -> float:
    """The runtime probability for one feature vector (bypassing the observation step)."""
    from psse_env.oracle import learned_ranker as runtime

    values = [float(v) for v in x]
    if spec.get("type") == "hgb":
        logit = runtime._hgb_raw_prediction(values, spec)
    else:
        logit = float(spec.get("intercept") or 0.0)
        for value, mean, scale, coefficient in zip(values, spec["mean"], spec["scale"], spec["coef"]):
            logit += float(coefficient) * ((value - float(mean)) / float(scale) if float(scale) else 0.0)
    probability = runtime._sigmoid(logit)
    platt = spec.get("platt") or {}
    if platt.get("coef") is not None:
        probability = runtime._sigmoid(float(platt["coef"]) * runtime._logit(probability) + float(platt.get("intercept") or 0.0))
    return float(probability)


def expected_calibration_error(probabilities: np.ndarray, y: np.ndarray, bins: int = 10) -> tuple[float, list[dict[str, float]]]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    table = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (probabilities >= low) & (probabilities < high if high < 1.0 else probabilities <= high)
        if not mask.any():
            continue
        confidence = float(probabilities[mask].mean())
        accuracy = float(y[mask].mean())
        total += mask.mean() * abs(confidence - accuracy)
        table.append({"bin_low": float(low), "bin_high": float(high), "n": int(mask.sum()), "mean_probability": confidence,
                      "positive_rate": accuracy})
    return float(total), table


def auc(scores: np.ndarray, y: np.ndarray) -> float | None:
    from sklearn.metrics import roc_auc_score

    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, scores))


# ------------------------------------------------------------- bootstrap


def parent_bootstrap(parents: Sequence[str], statistic: Callable[[np.ndarray], float | None], *, draws: int = BOOTSTRAP_DRAWS,
                     seed: int = 0) -> dict[str, float | None]:
    """Percentile interval of ``statistic(indices)`` under resampling of whole parents."""
    rng = np.random.default_rng(seed)
    groups: dict[str, list[int]] = defaultdict(list)
    for index, parent in enumerate(parents):
        groups[str(parent)].append(index)
    keys = list(groups)
    point = statistic(np.arange(len(parents)))
    values = []
    for _ in range(draws):
        chosen = rng.choice(len(keys), size=len(keys), replace=True)
        indices = np.concatenate([np.asarray(groups[keys[k]]) for k in chosen])
        value = statistic(indices)
        if value is not None and math.isfinite(value):
            values.append(value)
    if not values:
        return {"point": point, "low": None, "high": None}
    return {"point": point, "low": float(np.percentile(values, 2.5)), "high": float(np.percentile(values, 97.5))}


# ----------------------------------------------------------------- studies


def threshold_at_fpr(scores_neg: np.ndarray, fpr: float) -> float:
    """Score threshold whose false-positive rate on ``scores_neg`` is at most ``fpr``."""
    if scores_neg.size == 0:
        return 0.5
    if fpr <= 0:
        return float(np.max(scores_neg)) + 1e-9
    return float(np.quantile(scores_neg, 1.0 - fpr, method="higher"))


def threshold_at_recall(scores_pos: np.ndarray, recall: float) -> float:
    if scores_pos.size == 0:
        return 0.5
    return float(np.quantile(scores_pos, 1.0 - recall, method="lower"))


def acquisition_study(rows_cal: list, rows_test: list, scores_cal: np.ndarray, scores_test: np.ndarray, seed: int) -> dict[str, Any]:
    """Learned needs_phasors against rule v3, matched on the rule's false rate and on its recall."""
    y_cal = np.asarray([r["labels"]["needs_aux"] for r in rows_cal])
    neg_cal = np.asarray([r["needs_no_phasors_family"] for r in rows_cal])
    y_test = np.asarray([r["labels"]["needs_aux"] for r in rows_test])
    neg_test = np.asarray([r["needs_no_phasors_family"] for r in rows_test])
    rule_cal = np.asarray([r["rule_v3"] for r in rows_cal])
    rule_test = np.asarray([r["rule_v3"] for r in rows_test])
    parents = [r["parent"] for r in rows_test]
    # Thresholds chosen on the calibration split: the rule's false rate there, and its recall there.
    rule_fpr_cal = float(rule_cal[neg_cal].mean()) if neg_cal.any() else 0.0
    rule_recall_cal = float(rule_cal[y_cal == 1].mean()) if (y_cal == 1).any() else 0.0
    t_fpr = threshold_at_fpr(scores_cal[neg_cal], rule_fpr_cal)
    t_recall = threshold_at_recall(scores_cal[y_cal == 1], rule_recall_cal)

    def recall_at(indices, scores, threshold):
        positives = indices[y_test[indices] == 1]
        return float((scores[positives] >= threshold).mean()) if positives.size else None

    def fpr_at(indices, scores, threshold):
        negatives = indices[neg_test[indices]]
        return float((scores[negatives] >= threshold).mean()) if negatives.size else None

    def rule_recall(indices):
        positives = indices[y_test[indices] == 1]
        return float(rule_test[positives].mean()) if positives.size else None

    def rule_fpr(indices):
        negatives = indices[neg_test[indices]]
        return float(rule_test[negatives].mean()) if negatives.size else None

    result = {
        "rule_recall": parent_bootstrap(parents, rule_recall, seed=seed),
        "rule_false_rate": parent_bootstrap(parents, rule_fpr, seed=seed),
        "learned_recall_at_rule_false_rate": parent_bootstrap(parents, lambda i: recall_at(i, scores_test, t_fpr), seed=seed),
        "learned_false_rate_at_rule_false_rate": parent_bootstrap(parents, lambda i: fpr_at(i, scores_test, t_fpr), seed=seed),
        "learned_false_rate_at_rule_recall": parent_bootstrap(parents, lambda i: fpr_at(i, scores_test, t_recall), seed=seed),
        "learned_recall_at_rule_recall": parent_bootstrap(parents, lambda i: recall_at(i, scores_test, t_recall), seed=seed),
        "recall_difference_at_rule_false_rate": parent_bootstrap(
            parents, lambda i: (recall_at(i, scores_test, t_fpr) or 0.0) - (rule_recall(i) or 0.0), seed=seed),
        "false_rate_difference_at_rule_recall": parent_bootstrap(
            parents, lambda i: (fpr_at(i, scores_test, t_recall) or 0.0) - (rule_fpr(i) or 0.0), seed=seed),
        "auc": parent_bootstrap(parents, lambda i: auc(scores_test[i], y_test[i]), seed=seed),
        "thresholds": {"at_rule_false_rate": t_fpr, "at_rule_recall": t_recall,
                       "rule_false_rate_calibration": rule_fpr_cal, "rule_recall_calibration": rule_recall_cal},
        "n_test_positive": int((y_test == 1).sum()), "n_test_no_phasor_roots": int(neg_test.sum()),
    }
    by_family: dict[str, dict[str, Any]] = {}
    for family in sorted({r["family"] for r in rows_test}):
        mask = np.asarray([r["family"] == family for r in rows_test])
        by_family[family] = {"n": int(mask.sum()), "rule_positive": int(rule_test[mask].sum()),
                             "learned_positive_at_rule_false_rate": int((scores_test[mask] >= t_fpr).sum()),
                             "learned_positive_at_rule_recall": int((scores_test[mask] >= t_recall).sum())}
    result["by_family"] = by_family
    return result


def hif_mimic_study(rows_cal: list, rows_test: list, scores_cal: np.ndarray, scores_test: np.ndarray, seed: int) -> dict[str, Any]:
    """Learned HIF score against the screen's HIF flag, matched on the screen's HIF recall."""
    def subset(rows, scores):
        mask = np.asarray([r["labels"]["has_hif"] == 1 or r["family"].startswith("mimic") for r in rows])
        return [r for r, m in zip(rows, mask) if m], scores[mask]

    cal_rows, cal_scores = subset(rows_cal, scores_cal)
    test_rows, test_scores = subset(rows_test, scores_test)
    y_cal = np.asarray([r["labels"]["has_hif"] for r in cal_rows])
    y_test = np.asarray([r["labels"]["has_hif"] for r in test_rows])
    rule_cal = np.asarray([r["rule_hif"] for r in cal_rows])
    rule_test = np.asarray([r["rule_hif"] for r in test_rows])
    same_sign = np.asarray([r["family"] == "mimic_flow_pair_same_sign" for r in test_rows])
    parents = [r["parent"] for r in test_rows]
    rule_recall_cal = float(rule_cal[y_cal == 1].mean()) if (y_cal == 1).any() else 1.0
    threshold = threshold_at_recall(cal_scores[y_cal == 1], rule_recall_cal) if (y_cal == 1).any() else 0.5

    def recall(indices, scores_or_rule):
        positives = indices[y_test[indices] == 1]
        return float(scores_or_rule[positives].mean()) if positives.size else None

    def mimic_rate(indices, flags):
        mimics = indices[same_sign[indices]]
        return float(flags[mimics].mean()) if mimics.size else None

    learned_flags = (test_scores >= threshold).astype(float)
    return {
        "screen_hif_recall": parent_bootstrap(parents, lambda i: recall(i, rule_test.astype(float)), seed=seed),
        "learned_hif_recall_at_threshold": parent_bootstrap(parents, lambda i: recall(i, learned_flags), seed=seed),
        "screen_same_sign_mimic_flag_rate": parent_bootstrap(parents, lambda i: mimic_rate(i, rule_test.astype(float)), seed=seed),
        "learned_same_sign_mimic_flag_rate": parent_bootstrap(parents, lambda i: mimic_rate(i, learned_flags), seed=seed),
        "mimic_rate_difference": parent_bootstrap(
            parents, lambda i: (mimic_rate(i, learned_flags) or 0.0) - (mimic_rate(i, rule_test.astype(float)) or 0.0), seed=seed),
        "auc_hif_vs_mimics": parent_bootstrap(parents, lambda i: auc(test_scores[i], y_test[i]), seed=seed),
        "threshold": threshold, "n_test_hif": int((y_test == 1).sum()), "n_test_same_sign_mimics": int(same_sign.sum()),
        "n_test_opposite_sign_mimics": int(sum(1 for r in test_rows if r["family"] == "mimic_flow_pair_opposite_sign")),
    }


def unbalance_vs_voltage_meter_study(rows_test: list, scores_test: np.ndarray, seed: int) -> dict[str, Any]:
    """Among test roots with a voltage-meter pick: can the learned unbalance score tell the two apart?"""
    families = {"three_phase_unbalance", "measurement", "multi_measurement", "healthy_window",
                "mimic_flow_pair_same_sign", "mimic_flow_pair_opposite_sign"}
    mask = np.asarray([r["vm_pick"] and r["family"] in families for r in rows_test])
    rows = [r for r, m in zip(rows_test, mask) if m]
    scores = scores_test[mask]
    y = np.asarray([r["labels"]["has_unbalance"] for r in rows])
    parents = [r["parent"] for r in rows]
    families = Counter(r["family"] for r in rows)
    if not rows or len(np.unique(y)) < 2:
        return {"n": len(rows), "families": dict(families), "auc": None}
    # Acquisitions saved on non-unbalance roots at 95% unbalance recall (threshold from this subset: descriptive only).
    threshold = threshold_at_recall(scores[y == 1], 0.95)

    def saved(indices):
        negatives = indices[y[indices] == 0]
        return float((scores[negatives] < threshold).mean()) if negatives.size else None

    return {
        "n": len(rows), "families": dict(families), "n_unbalance": int(y.sum()),
        "auc": parent_bootstrap(parents, lambda i: auc(scores[i], y[i]), seed=seed),
        "non_unbalance_acquisitions_avoided_at_95pct_recall": parent_bootstrap(parents, saved, seed=seed),
        "note": "threshold set on the test subset itself; descriptive, not a deployable operating point",
    }


def adjacent_line_study(rows_train: list, rows_test: list, seed: int) -> dict[str, Any]:
    """Pairwise choice between the multiplier ranking's top two lines on parameter roots."""
    from mcp_server.matpower_server import _load_python_case
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    case = _load_python_case("case14")
    branch = np.asarray(case["branch"], dtype=float)
    endpoints = {k: (int(branch[k, 0]), int(branch[k, 1])) for k in range(branch.shape[0])}

    def pair_rows(rows):
        out = []
        for row in rows:
            if row["family"] not in ("parameter", "measurement+parameter"):
                continue
            record = row["record"]
            truth = record["truth"].get("parameter") or []
            if len(truth) != 1:
                continue
            true_row = int(truth[0]["branch_row0"])
            wls = record["analysis"]["wls"]
            ranking = wls.get("branch_multiplier_ranking") or []
            if len(ranking) < 2 or true_row not in ranking[:2]:
                continue
            top, second = int(ranking[0]), int(ranking[1])
            lambdas = {int(item["branch_row0"]): float(item["value"]) for item in wls.get("top_branch_multipliers") or []}
            screen = record["analysis"].get("screen") or {}
            first = next((r for r in screen.get("rounds") or [] if r.get("scores")), {})
            ranked = ((first.get("best") or {}).get("parameter") or {}).get("ranked") or []
            screen_j = {int(item["branch_row0"]): float(item["J"]) for item in ranked if item.get("J") is not None}
            screen_top = int(ranked[0]["branch_row0"]) if ranked else -1
            shared_bus = len(set(endpoints[top]) & set(endpoints[second])) > 0
            features_pair = [
                _log((lambdas.get(top, 1e-6) + 1e-6) / (lambdas.get(second, 1e-6) + 1e-6)),
                float(min(wls.get("branch_ranking_dominance_ratio") or 10.0, 10.0)),
                _log((screen_j.get(second, 1e3) + 1.0) / (screen_j.get(top, 1e3) + 1.0)),
                float(screen_top == top), float(screen_top == second), float(shared_bus),
                float(bool(wls.get("measurement_dominant"))), math.log1p(float(wls.get("max_normalized_residual") or 0.0)),
            ]
            out.append({"x": features_pair, "y": int(true_row == top), "parent": row["parent"], "family": row["family"],
                        "screen_pick_true": int(screen_top == true_row), "screen_in_pair": int(screen_top in (top, second))})
        return out

    train_pairs, test_pairs = pair_rows(rows_train), pair_rows(rows_test)
    if len(train_pairs) < 20 or not test_pairs:
        return {"n_train": len(train_pairs), "n_test": len(test_pairs), "note": "too few pairs"}
    x_train = np.asarray([p["x"] for p in train_pairs]); y_train = np.asarray([p["y"] for p in train_pairs])
    x_test = np.asarray([p["x"] for p in test_pairs]); y_test = np.asarray([p["y"] for p in test_pairs])
    model = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000)).fit(x_train, y_train)
    prob_top = model.predict_proba(x_test)[:, 1]
    parents = [p["parent"] for p in test_pairs]
    picks_model = (prob_top >= 0.5).astype(int)
    screen_pick = np.asarray([p["screen_pick_true"] for p in test_pairs])

    def accuracy(indices, picks):
        return float((picks[indices] == y_test[indices]).mean()) if indices.size else None

    return {
        "n_train": len(train_pairs), "n_test": len(test_pairs),
        "test_top_is_true_rate": float(y_test.mean()),
        "accuracy_take_top": parent_bootstrap(parents, lambda i: accuracy(i, np.ones_like(y_test)), seed=seed),
        "accuracy_screen_pick": parent_bootstrap(parents, lambda i: float(screen_pick[i].mean()) if i.size else None, seed=seed),
        "accuracy_pairwise_model": parent_bootstrap(parents, lambda i: accuracy(i, picks_model), seed=seed),
        "model_minus_take_top": parent_bootstrap(
            parents, lambda i: (accuracy(i, picks_model) or 0.0) - (accuracy(i, np.ones_like(y_test)) or 0.0), seed=seed),
        "coefficients": dict(zip(["lambda_gap_log", "dominance_ratio", "screen_j_gap_log", "screen_top_is_top",
                                  "screen_top_is_second", "shared_bus", "measurement_dominant", "max_rn_log1p"],
                                 [float(c) for c in model[-1].coef_[0]])),
    }


def transfer_study(transfer_dir: Path, names: Sequence[str], models: Mapping[str, Mapping[str, Any]],
                   calibrators: Mapping[str, Callable], thresholds: Mapping[str, float], seed: int) -> dict[str, Any]:
    rows = load_dataset(transfer_dir, include_children=False)
    if not rows:
        return {"n": 0}
    x = matrix(rows, names)
    parents = [r["parent"] for r in rows]
    out: dict[str, Any] = {"n": len(rows), "families": dict(Counter(r["family"] for r in rows))}
    for target in ("needs_aux", "has_hif"):
        model = (models.get(target) or {}).get("gbm")
        if model is None:
            continue
        scores = calibrators[target](model.predict_proba(x)[:, 1])
        y = np.asarray([r["labels"][target] for r in rows])
        rule = np.asarray([r["rule_v3"] if target == "needs_aux" else r["rule_hif"] for r in rows])
        threshold = thresholds[target]
        neg = np.asarray([r["needs_no_phasors_family"] for r in rows]) if target == "needs_aux" else (y == 0)

        def recall(indices, flags):
            positives = indices[y[indices] == 1]
            return float(flags[positives].mean()) if positives.size else None

        def false_rate(indices, flags):
            negatives = indices[neg[indices]]
            return float(flags[negatives].mean()) if negatives.size else None

        learned = (scores >= threshold).astype(float)
        out[target] = {
            "auc": parent_bootstrap(parents, lambda i: auc(scores[i], y[i]), seed=seed),
            "rule_recall": parent_bootstrap(parents, lambda i: recall(i, rule.astype(float)), seed=seed),
            "learned_recall_ieee14_threshold": parent_bootstrap(parents, lambda i: recall(i, learned), seed=seed),
            "rule_false_rate": parent_bootstrap(parents, lambda i: false_rate(i, rule.astype(float)), seed=seed),
            "learned_false_rate_ieee14_threshold": parent_bootstrap(parents, lambda i: false_rate(i, learned), seed=seed),
            "n_positive": int(y.sum()),
        }
    return out


# -------------------------------------------------------------------- main


def _git_commit() -> str | None:
    import subprocess

    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return None


def _fmt(interval: Mapping[str, Any] | None) -> str:
    if not isinstance(interval, Mapping) or interval.get("point") is None:
        return "n/a"
    point = interval["point"]
    if interval.get("low") is None:
        return f"{100 * point:.1f}%"
    return f"{100 * point:.1f}% [{100 * interval['low']:.1f}, {100 * interval['high']:.1f}]"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--transfer-dir", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--no-children", action="store_true", help="do not add the truth-corrected child states to training")
    parser.add_argument("--feature-set", choices=("offline", "policy"), default="offline",
                        help="offline: the step-4 study features (full screen report); policy: what the expert computes at runtime")
    parser.add_argument("--export-model", default=None,
                        help="write the needs_aux/has_hif/has_unbalance models as JSON for psse_env.oracle.learned_ranker (policy feature set)")
    parser.add_argument("--export-model-type", choices=("hgb", "logistic"), default="hgb",
                        help="which fitted model the export carries: the gradient-boosted trees (default) or the standardized logistic model")
    args = parser.parse_args(argv)
    if args.export_model and args.feature_set != "policy":
        parser.error("--export-model needs --feature-set policy: the runtime computes the policy-visible features only")
    started = time.perf_counter()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = load_dataset(Path(args.dataset_dir), include_children=not args.no_children, feature_set=args.feature_set)
    names = sorted(rows[0]["features"])
    by_split = {split: [r for r in rows if r["split"] == split] for split in ("train", "calibration", "test")}
    # Children are training augmentation only: every evaluation reads roots, healthy alarms and mimics.
    eval_cal = [r for r in by_split["calibration"] if r["kind"] != "child"]
    eval_test = [r for r in by_split["test"] if r["kind"] != "child"]
    x_train = matrix(by_split["train"], names)
    x_cal = matrix(eval_cal, names)
    x_test = matrix(eval_test, names)
    print(f"[ranker] rows train {len(by_split['train'])} (children {sum(r['kind'] == 'child' for r in by_split['train'])}), "
          f"calibration {len(eval_cal)}, test {len(eval_test)}, features {len(names)} ({args.feature_set})", flush=True)

    models: dict[str, dict[str, Any]] = {}
    calibrators: dict[str, Callable] = {}
    target_metrics: dict[str, Any] = {}
    scores_test: dict[str, np.ndarray] = {}
    scores_cal: dict[str, np.ndarray] = {}
    # The deployable logistic models beside the gradient-boosted study models.
    logit_calibrators: dict[str, PlattScaler] = {}
    logit_scores_test: dict[str, np.ndarray] = {}
    logit_scores_cal: dict[str, np.ndarray] = {}
    for target in TARGETS:
        y_train = np.asarray([r["labels"][target] for r in by_split["train"]])
        y_cal = np.asarray([r["labels"][target] for r in eval_cal])
        y_test = np.asarray([r["labels"][target] for r in eval_test])
        fitted = fit_models(x_train, y_train, args.seed)
        models[target] = fitted
        entry: dict[str, Any] = {"n_train_positive": int(y_train.sum()), "n_test_positive": int(y_test.sum())}
        parents_test = [r["parent"] for r in eval_test]
        for name, model in fitted.items():
            raw_cal = model.predict_proba(x_cal)[:, 1]
            raw_test = model.predict_proba(x_test)[:, 1]
            calibrate = platt(raw_cal, y_cal)
            prob_test = calibrate(raw_test)
            ece, bins = expected_calibration_error(prob_test, y_test)
            entry[name] = {
                "auc": parent_bootstrap(parents_test, lambda i, s=raw_test, y=y_test: auc(s[i], y[i]), seed=args.seed),
                "ece_calibrated": ece, "reliability": bins,
            }
            if name == "gbm":
                calibrators[target] = calibrate
                scores_test[target] = prob_test
                scores_cal[target] = calibrate(raw_cal)
            else:
                logit_calibrators[target] = calibrate
                logit_scores_test[target] = prob_test
                logit_scores_cal[target] = calibrate(raw_cal)
        target_metrics[target] = entry

    acquisition = acquisition_study(eval_cal, eval_test, scores_cal["needs_aux"], scores_test["needs_aux"], args.seed)
    mimic = hif_mimic_study(eval_cal, eval_test, scores_cal["has_hif"], scores_test["has_hif"], args.seed)
    unbalance = unbalance_vs_voltage_meter_study(eval_test, scores_test["has_unbalance"], args.seed)
    adjacent = adjacent_line_study([r for r in by_split["train"] if r["kind"] == "root"], eval_test, args.seed)
    transfer = {}
    if args.transfer_dir:
        transfer = transfer_study(Path(args.transfer_dir), names, models, calibrators,
                                  {"needs_aux": acquisition["thresholds"]["at_rule_false_rate"], "has_hif": mimic["threshold"]},
                                  args.seed)
    # The deployed logistic models at the same operating points.
    deployed: dict[str, Any] = {}
    if "needs_aux" in logit_scores_test:
        deployed["acquisition"] = acquisition_study(eval_cal, eval_test, logit_scores_cal["needs_aux"], logit_scores_test["needs_aux"], args.seed)
    if "has_hif" in logit_scores_test:
        deployed["hif_vs_mimic"] = hif_mimic_study(eval_cal, eval_test, logit_scores_cal["has_hif"], logit_scores_test["has_hif"], args.seed)
    if "has_unbalance" in logit_scores_test:
        deployed["unbalance_vs_voltage_meter"] = unbalance_vs_voltage_meter_study(eval_test, logit_scores_test["has_unbalance"], args.seed)
    exported = None
    if args.export_model:
        first_valid = next((r["record"] for r in rows if (r["record"].get("analysis") or {}).get("nb")), None)
        analysis = (first_valid or {}).get("analysis") or {}
        case_name = str((first_valid or {}).get("case") or "")
        system = {"case_id": Path(case_name).stem if case_name else None, "bus_count": analysis.get("nb"), "branch_count": analysis.get("nl")}
        # The export carries the studied model's own operating points: the
        # gradient-boosted studies above, or the logistic ones just computed.
        source_acq = acquisition if args.export_model_type == "hgb" else deployed["acquisition"]
        source_mimic = mimic if args.export_model_type == "hgb" else deployed["hif_vs_mimic"]
        source_unbalance = unbalance if args.export_model_type == "hgb" else deployed["unbalance_vs_voltage_meter"]
        model_key = "gbm" if args.export_model_type == "hgb" else "logit"
        thresholds = {
            "needs_aux": {"at_rule_recall": source_acq["thresholds"]["at_rule_recall"],
                          "at_rule_false_rate": source_acq["thresholds"]["at_rule_false_rate"]},
            "has_hif": {"at_screen_recall": source_mimic["threshold"]},
            "has_unbalance": {},
        }
        test_metrics = {
            "needs_aux": {key: source_acq[key] for key in (
                "rule_recall", "rule_false_rate", "learned_recall_at_rule_recall", "learned_false_rate_at_rule_recall",
                "learned_recall_at_rule_false_rate", "learned_false_rate_at_rule_false_rate", "auc", "n_test_positive",
                "n_test_no_phasor_roots", "by_family")},
            "has_hif": {key: source_mimic[key] for key in (
                "screen_hif_recall", "learned_hif_recall_at_threshold", "screen_same_sign_mimic_flag_rate",
                "learned_same_sign_mimic_flag_rate", "auc_hif_vs_mimics", "n_test_hif", "n_test_same_sign_mimics")},
            "has_unbalance": {"auc": (target_metrics.get("has_unbalance") or {}).get(model_key, {}).get("auc"),
                              "unbalance_vs_voltage_meter_auc": source_unbalance.get("auc")},
        }
        provenance = {"dataset_dir": str(args.dataset_dir), "seed": args.seed, "feature_set": args.feature_set,
                      "rows": {k: len(v) for k, v in by_split.items()}, "eval_rows": {"calibration": len(eval_cal), "test": len(eval_test)},
                      "trained_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "git_commit": _git_commit(),
                      "model": ("sklearn HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, "
                                "min_samples_leaf=10, l2_regularization=1.0), Platt on calibration" if args.export_model_type == "hgb"
                                else "sklearn StandardScaler + LogisticRegression(C=0.5), Platt on calibration")}
        exported = export_models(Path(args.export_model), model_type=args.export_model_type, names=names, models=models,
                                 calibrators=(calibrators if args.export_model_type == "hgb" else logit_calibrators),
                                 thresholds=thresholds, test_metrics=test_metrics, system=system, provenance=provenance,
                                 check_x=x_test)
        print(f"[ranker] exported {args.export_model_type} {sorted(exported['targets'])} to {args.export_model}", flush=True)

    importances: dict[str, list[tuple[str, float]]] = {}
    from sklearn.inspection import permutation_importance

    for target in ("needs_aux", "has_hif", "has_unbalance"):
        model = models[target].get("gbm")
        if model is None:
            continue
        y_test = np.asarray([r["labels"][target] for r in eval_test])
        if len(np.unique(y_test)) < 2:
            continue
        result = permutation_importance(model, x_test, y_test, scoring="roc_auc", n_repeats=5, random_state=args.seed)
        order = np.argsort(-result.importances_mean)[:12]
        importances[target] = [(names[i], float(result.importances_mean[i])) for i in order]

    metrics = {
        "dataset_dir": str(args.dataset_dir), "transfer_dir": args.transfer_dir, "seed": args.seed,
        "rows": {k: len(v) for k, v in by_split.items()}, "eval_rows": {"calibration": len(eval_cal), "test": len(eval_test)},
        "features": names, "targets": target_metrics, "acquisition": acquisition, "hif_vs_mimic": mimic,
        "unbalance_vs_voltage_meter": unbalance, "adjacent_line": adjacent, "transfer_ieee57": transfer,
        "permutation_importance": importances, "seconds": time.perf_counter() - started,
        "feature_set": args.feature_set, "deployed_logistic": deployed,
        "export_model": (str(args.export_model) if args.export_model else None),
    }
    (out / "metrics.json").write_text(json.dumps(json_safe(metrics), indent=2, sort_keys=True), encoding="utf-8")
    with (out / "test_scores.jsonl").open("w", encoding="utf-8") as stream:
        for index, row in enumerate(eval_test):
            stream.write(json.dumps(json_safe({
                "id": row["id"], "family": row["family"], "parent": row["parent"], "labels": row["labels"],
                "rule_v3": row["rule_v3"], "rule_hif": row["rule_hif"],
                "scores": {target: float(scores_test[target][index]) for target in scores_test},
                "logit_scores": {target: float(logit_scores_test[target][index]) for target in logit_scores_test},
            }), sort_keys=True) + "\n")

    lines = [f"# Step 4: learned ranker against the physics screen ({args.feature_set} features)\n",
             f"Dataset `{args.dataset_dir}`; rows train {len(by_split['train'])} (incl. {sum(r['kind'] == 'child' for r in by_split['train'])} child states), "
             f"calibration {len(eval_cal)}, test {len(eval_test)}; {len(names)} {args.feature_set} features; parent-bootstrap 95% intervals in brackets.\n",
             "### Per-target discrimination on the test split (AUC; GBM, logistic)\n",
             "| target | test positives | GBM AUC | logistic AUC | GBM ECE after Platt |", "|---|---|---|---|---|"]
    for target, entry in target_metrics.items():
        gbm = entry.get("gbm") or {}
        logit = entry.get("logit") or {}
        lines.append(f"| {target} | {entry['n_test_positive']} | {_fmt(gbm.get('auc'))} | {_fmt(logit.get('auc'))} | "
                     f"{gbm.get('ece_calibrated', float('nan')):.3f} |")
    a = acquisition
    lines += ["", "### 1. Acquisition: learned needs_aux (HIF, unbalance or harmonic) against rule v3 (HIF won, voltage-meter pick, or unexplained)\n",
              f"Test roots: {a['n_test_positive']} need an auxiliary stream, {a['n_test_no_phasor_roots']} need none. Thresholds set on the calibration split.\n",
              "| quantity | rule v3 | learned |", "|---|---|---|",
              f"| recall on roots that need an auxiliary stream | {_fmt(a['rule_recall'])} | {_fmt(a['learned_recall_at_rule_false_rate'])} (at the rule's false rate) |",
              f"| acquisitions on roots that need none | {_fmt(a['rule_false_rate'])} | {_fmt(a['learned_false_rate_at_rule_recall'])} (at the rule's recall) |",
              f"| recall difference at the rule's false rate | | {_fmt(a['recall_difference_at_rule_false_rate'])} |",
              f"| false-rate difference at the rule's recall | | {_fmt(a['false_rate_difference_at_rule_recall'])} |",
              f"| AUC | | {_fmt(a['auc'])} |", "",
              "| family | test roots | rule positive | learned positive at rule false rate | learned positive at rule recall |", "|---|---|---|---|---|"]
    for family, entry in a["by_family"].items():
        lines.append(f"| {family} | {entry['n']} | {entry['rule_positive']} | {entry['learned_positive_at_rule_false_rate']} | {entry['learned_positive_at_rule_recall']} |")
    m = mimic
    lines += ["", "### 2. HIF against the same-sign flow-meter pair\n",
              f"Test: {m['n_test_hif']} HIF roots, {m['n_test_same_sign_mimics']} same-sign and {m['n_test_opposite_sign_mimics']} opposite-sign mimics.\n",
              "| quantity | screen flag | learned score at matched recall |", "|---|---|---|",
              f"| HIF recall | {_fmt(m['screen_hif_recall'])} | {_fmt(m['learned_hif_recall_at_threshold'])} |",
              f"| same-sign mimics flagged | {_fmt(m['screen_same_sign_mimic_flag_rate'])} | {_fmt(m['learned_same_sign_mimic_flag_rate'])} |",
              f"| difference in mimics flagged | | {_fmt(m['mimic_rate_difference'])} |",
              f"| AUC, HIF roots against mimics | | {_fmt(m['auc_hif_vs_mimics'])} |"]
    u = unbalance
    lines += ["", "### 3. Unbalance against a voltage-meter error, among test roots with a voltage-meter pick\n",
              f"{u.get('n', 0)} roots ({u.get('families')}), {u.get('n_unbalance', 0)} unbalance. AUC {_fmt(u.get('auc'))}; "
              f"non-unbalance acquisitions avoided at 95% unbalance recall {_fmt(u.get('non_unbalance_acquisitions_avoided_at_95pct_recall'))} ({u.get('note', '')}).\n"]
    d = adjacent
    if "accuracy_take_top" in d:
        lines += ["### 4. Adjacent-line choice between the multiplier ranking's top two (parameter roots)\n",
                  f"Pairs: {d['n_train']} train, {d['n_test']} test; the top line is true on {100 * d['test_top_is_true_rate']:.1f}% of test pairs.\n",
                  "| chooser | accuracy |", "|---|---|",
                  f"| take the top line | {_fmt(d['accuracy_take_top'])} |",
                  f"| the screen's top line | {_fmt(d['accuracy_screen_pick'])} |",
                  f"| pairwise logistic model | {_fmt(d['accuracy_pairwise_model'])} |",
                  f"| model minus take-top | {_fmt(d['model_minus_take_top'])} |", ""]
    else:
        lines += ["### 4. Adjacent-line choice\n", f"{d}\n"]
    if transfer.get("n"):
        lines += ["### 5. Transfer to the IEEE 57 study roots (IEEE-14 models and thresholds unchanged)\n",
                  f"{transfer['n']} alarmed roots ({transfer['families']}).\n",
                  "| target | positives | AUC | rule recall | learned recall | rule false rate | learned false rate |", "|---|---|---|---|---|---|---|"]
        for target in ("needs_aux", "has_hif"):
            t = transfer.get(target) or {}
            if t:
                lines.append(f"| {target} | {t['n_positive']} | {_fmt(t['auc'])} | {_fmt(t['rule_recall'])} | {_fmt(t['learned_recall_ieee14_threshold'])} | "
                             f"{_fmt(t['rule_false_rate'])} | {_fmt(t['learned_false_rate_ieee14_threshold'])} |")
    if deployed:
        da, dm, du = deployed.get("acquisition") or {}, deployed.get("hif_vs_mimic") or {}, deployed.get("unbalance_vs_voltage_meter") or {}
        lines += ["### 6. Deployed logistic models (what LearnedRanker computes at runtime)\n",
                  "| quantity | rule / screen | logistic |", "|---|---|---|"]
        if da:
            lines += [f"| acquisition recall at the rule's false rate | {_fmt(da['rule_recall'])} | {_fmt(da['learned_recall_at_rule_false_rate'])} |",
                      f"| unnecessary acquisitions at the rule's recall | {_fmt(da['rule_false_rate'])} | {_fmt(da['learned_false_rate_at_rule_recall'])} |",
                      f"| recall at the rule's recall threshold (deferral operating point) | {_fmt(da['rule_recall'])} | {_fmt(da['learned_recall_at_rule_recall'])} |",
                      f"| needs_aux AUC | | {_fmt(da['auc'])} |",
                      f"| needs_aux thresholds (at rule recall / at rule false rate) | | {da['thresholds']['at_rule_recall']:.4f} / {da['thresholds']['at_rule_false_rate']:.4f} |"]
        if dm:
            lines += [f"| HIF recall at the screen's recall | {_fmt(dm['screen_hif_recall'])} | {_fmt(dm['learned_hif_recall_at_threshold'])} |",
                      f"| same-sign mimics flagged | {_fmt(dm['screen_same_sign_mimic_flag_rate'])} | {_fmt(dm['learned_same_sign_mimic_flag_rate'])} |",
                      f"| AUC, HIF roots against mimics | | {_fmt(dm['auc_hif_vs_mimics'])} |"]
        if du:
            lines += [f"| unbalance against voltage-meter pick, AUC | | {_fmt(du.get('auc'))} |"]
        if da:
            lines += ["", "| family | test roots | rule positive | logistic positive at rule false rate | logistic positive at rule recall |", "|---|---|---|---|---|"]
            for family, entry in da["by_family"].items():
                lines.append(f"| {family} | {entry['n']} | {entry['rule_positive']} | {entry['learned_positive_at_rule_false_rate']} | {entry['learned_positive_at_rule_recall']} |")
        if exported is not None:
            lines.append(f"\nExported `{args.export_model}` ({args.export_model_type}: {', '.join(sorted(exported['targets']))}; "
                         "the runtime evaluation reproduces sklearn's probabilities on the test split to 1e-6).")
        lines.append("")
    lines += ["", "### Permutation importance (test AUC drop, top 12)\n"]
    for target, items in importances.items():
        lines.append(f"- **{target}**: " + ", ".join(f"{name} {value:.3f}" for name, value in items))
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
