"""Frozen-threshold test evaluation with physical-parent bootstrap intervals."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .calibrate import validate_heldout_parents
from .dataset import FAMILY_NAMES, file_sha256, prepare_corpus, write_json
from .train import load_trained_model, predict_samples


def family_ranking_metrics(rows, family_index):
    """Threshold-free discrimination of labeled families, with ties preserved."""
    from scipy.stats import rankdata
    labeled = [r for r in rows if r["labels"]["family_mask"][family_index]]
    labels = np.asarray([r["labels"]["family"][family_index] for r in labeled], dtype=np.float64)
    scores = np.asarray([r["family_scores"][family_index] for r in labeled], dtype=np.float64)
    positives, negatives = int(labels.sum()), int(len(labels) - labels.sum())
    if not positives or not negatives:
        return {"labeled_count": len(labeled), "positive_count": positives, "roc_auc": None, "average_precision": None}
    ranks = rankdata(scores, method="average")
    auc = (ranks[labels == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives)
    order = np.argsort(-scores, kind="stable")
    sorted_scores, sorted_labels = scores[order], labels[order]
    cumulative_positive = np.cumsum(sorted_labels)
    tie_end = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(scores) - 1]
    recall = cumulative_positive[tie_end] / positives
    precision = cumulative_positive[tie_end] / (tie_end + 1)
    average_precision = np.sum(np.diff(np.r_[0.0, recall]) * precision)
    return {"labeled_count": len(labeled), "positive_count": positives,
            "roc_auc": float(auc), "average_precision": float(average_precision)}


def grouped_rate(rows, values, *, bootstrap_replicates=1000, seed=2026):
    """Window-weighted rate, resampling entire independent physical parents."""
    if not rows:
        return {"count": 0, "parents": 0, "rate": None, "parent_bootstrap_ci95": None}
    groups = defaultdict(list)
    for row, value in zip(rows, values):
        groups[row["parent_id"]].append(float(value))
    parent_totals = np.asarray([[sum(v), len(v)] for v in groups.values()], dtype=np.float64)
    interval = None
    if len(groups) >= 2 and bootstrap_replicates > 0:
        rng = np.random.default_rng(seed)
        rates = []
        for _ in range(bootstrap_replicates):
            totals = parent_totals[rng.integers(len(groups), size=len(groups))].sum(axis=0)
            rates.append(totals[0] / totals[1])
        interval = np.quantile(rates, [0.025, 0.975]).tolist()
    return {"count": len(rows), "parents": len(groups), "rate": float(np.mean(values)),
            "parent_bootstrap_ci95": interval}


def summarize(predictions, calibration, trained_mask, *, bootstrap_replicates=1000, seed=2026):
    threshold = calibration["phase_threshold"]
    anomaly_threshold = calibration["anomaly_threshold"]
    def rate(rows, decision):
        return grouped_rate(rows, [decision(r) for r in rows], bootstrap_replicates=bootstrap_replicates, seed=seed)
    trigger = lambda r: r["phase_score"] > threshold
    healthy = [r for r in predictions if r["labels"]["anomaly_mask"] and not r["labels"]["anomaly"]]
    phase = [r for r in predictions if r["labels"]["phase_mask"] and r["labels"]["phase"]]
    nonphase = [r for r in predictions if r["labels"]["phase_mask"] and not r["labels"]["phase"] and r["labels"]["anomaly"]]
    anomaly = [r for r in predictions if r["labels"]["anomaly_mask"] and r["labels"]["anomaly"]]
    known_wls = all(r["wls_alarm"] is not None for r in predictions)
    by_family, by_severity = {}, {}
    for i, name in enumerate(FAMILY_NAMES):
        rows = [r for r in predictions if r["labels"]["family_mask"][i] and r["labels"]["family"][i]]
        by_family[name] = {"head_trained": bool(trained_mask[i]), "phase_trigger": rate(rows, trigger),
            "mean_family_score_positive": float(np.mean([r["family_scores"][i] for r in rows])) if rows and trained_mask[i] else None,
            "family_score_discrimination": family_ranking_metrics(predictions, i) if trained_mask[i] else None}
        for severity in sorted({r["severity"] for r in rows}):
            subset = [r for r in rows if r["severity"] == severity]
            by_severity[f"{name}/{severity}"] = rate(subset, trigger)
    # Offline audit strata may describe visibility/severity, never model inputs.
    offline_strata = {}
    for key in ("residual_visible_energy_bin", "affected_phase", "phase_localizable"):
        values = sorted({str(r["offline_metadata"][key]) for r in phase if key in r["offline_metadata"]})
        offline_strata[key] = {value: rate([r for r in phase if str(r["offline_metadata"].get(key)) == value], trigger) for value in values}
    result = {
        "valid_test_windows": len(predictions), "phase_recall": rate(phase, trigger),
        "healthy_false_trigger_rate": rate(healthy, trigger),
        "nonphase_fault_trigger_rate": rate(nonphase, trigger),
        "general_anomaly_recall": rate(anomaly, lambda r: r["anomaly_score"] > anomaly_threshold),
        "healthy_anomaly_trigger_rate": rate(healthy, lambda r: r["anomaly_score"] > anomaly_threshold),
        "phase_acquisition_fraction": rate(predictions, trigger),
        "by_family": by_family, "by_family_and_severity": by_severity, "offline_phase_strata": offline_strata,
        "wls_comparator_available": known_wls,
        "uncertainty_method": "percentile bootstrap of physical parents, keeping every paired window together",
        "acquisition_cost_note": "Acquisition fractions are screening requests, not measured downstream diagnostic cost or recovery.",
    }
    if known_wls:
        result.update({
            "wls_phase_recall": rate(phase, lambda r: bool(r["wls_alarm"])),
            "wls_healthy_false_trigger_rate": rate(healthy, lambda r: bool(r["wls_alarm"])),
            "union_phase_recall": rate(phase, lambda r: trigger(r) or bool(r["wls_alarm"])),
            "union_healthy_false_trigger_rate": rate(healthy, lambda r: trigger(r) or bool(r["wls_alarm"])),
            "union_acquisition_fraction": rate(predictions, lambda r: trigger(r) or bool(r["wls_alarm"])),
            "paired_phase_recall_gain_over_wls": rate(phase, lambda r: float(trigger(r)) - float(bool(r["wls_alarm"]))),
        })
    return result


def evaluate(manifest, checkpoint_path, calibration_path, output, *, cache_dir=None,
             device="cpu", allow_new_parents=False, bootstrap_replicates=1000, seed=2026):
    model, scaler, checkpoint = load_trained_model(checkpoint_path, device)
    calibration = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
    if calibration["checkpoint_sha256"] != file_sha256(checkpoint_path) or calibration["model_id"] != checkpoint["model_id"]:
        raise ValueError("calibration belongs to a different frozen checkpoint")
    if calibration.get("threshold_comparison") != ">":
        raise ValueError("unsupported threshold comparison")
    corpus = prepare_corpus(manifest, cache_dir=cache_dir,
                            split_seed=checkpoint["training_config"]["split_seed"])
    parents = validate_heldout_parents(corpus.parent_splits, checkpoint, "test", allow_new_parents=allow_new_parents)
    if set(parents) & set(calibration["calibration_parent_ids"]):
        raise ValueError("test parents overlap independent calibration parents")
    comparator = calibration["wls_comparator"]
    predictions = predict_samples(model, scaler, corpus.split("test"), device=device,
                                  wls_chi2_alpha=comparator["chi2_alpha"],
                                  wls_normalized_threshold=comparator["normalized_residual_threshold"])
    if not predictions:
        raise ValueError("no valid test graphs")
    result = summarize(predictions, calibration, checkpoint["trained_family_mask"], bootstrap_replicates=bootstrap_replicates, seed=seed)
    invalid = [row for row in corpus.invalid if row["split"] == "test"]
    phase_invalid = sum(bool(r["labels"]["phase_mask"] and r["labels"]["phase"]) for r in invalid)
    valid_phase = [r for r in predictions if r["labels"]["phase_mask"] and r["labels"]["phase"]]
    triggered_phase = sum(r["phase_score"] > calibration["phase_threshold"] for r in valid_phase)
    result.update({"model_id": checkpoint["model_id"], "checkpoint_sha256": calibration["checkpoint_sha256"],
        "test_parent_ids": parents, "calibration": calibration, "invalid_test_windows": invalid,
        "phase_positive_unavailable_count": phase_invalid,
        "phase_trigger_recall_counting_unavailable_as_untriggered": (triggered_phase / (len(valid_phase) + phase_invalid)
                                                                     if len(valid_phase) + phase_invalid else None),
        "screen_availability": len(predictions) / (len(predictions) + len(invalid)),
        "invalid_screen_policy": "unavailable, never a negative prediction; independent acquisition fallback remains required",
        "transfer_study": ("new_parent_healthy_recalibrated" if calibration["calibration_kind"] == "new_parent_healthy_recalibration"
                           else "frozen_source_model_scaler_threshold_on_new_parents"
                           if any(parent not in checkpoint["parent_splits"] for parent in parents) else "source_parent_heldout"),
        "network_transfer_claim": "none; new parent IDs alone do not establish a different network",
        "predictions": [{**r, "family_scores": [score if trained else None for score, trained in zip(r["family_scores"], checkpoint["trained_family_mask"])]} for r in predictions]})
    write_json(output, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dir")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--allow-new-parents", action="store_true")
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)
    result = evaluate(args.manifest, args.checkpoint, args.calibration, args.output,
        cache_dir=args.cache_dir, device=args.device, allow_new_parents=args.allow_new_parents,
        bootstrap_replicates=args.bootstrap_replicates, seed=args.seed)
    print(f"evaluated {result['valid_test_windows']} valid windows; see {args.output}")


if __name__ == "__main__":
    main()
