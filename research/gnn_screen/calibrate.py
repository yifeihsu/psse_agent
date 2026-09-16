"""Healthy-reference thresholds for a selected, frozen WLS screen checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .dataset import FAMILY_NAMES, file_sha256, prepare_corpus, write_json
from .train import empirical_threshold, load_trained_model, predict_samples


def validate_heldout_parents(parent_splits, checkpoint, split, *, allow_new_parents=False):
    """Enforce frozen source splits; explicitly support disjoint target corpora."""
    original = checkpoint["parent_splits"]
    selected = {parent for parent, name in parent_splits.items() if name == split}
    if not selected:
        raise ValueError(f"no parents declared for {split}")
    for parent in selected:
        if parent in original and original[parent] != split:
            raise ValueError(f"parent {parent!r} was assigned {original[parent]}, not {split}")
        if parent not in original and not allow_new_parents:
            raise ValueError("new parent corpus requires explicit --allow-new-parents transfer evaluation")
    return sorted(selected)


def calibrate(manifest, checkpoint_path, output, *, cache_dir=None, device="cpu",
              false_trigger_rate=0.01, allow_new_parents=False,
              wls_chi2_alpha=0.05, wls_normalized_threshold=4.0, workers=1):
    model, scaler, checkpoint = load_trained_model(checkpoint_path, device)
    corpus = prepare_corpus(manifest, cache_dir=cache_dir,
                            split_seed=checkpoint["training_config"]["split_seed"], workers=workers)
    parent_ids = validate_heldout_parents(corpus.parent_splits, checkpoint, "calibration",
                                         allow_new_parents=allow_new_parents)
    predictions = predict_samples(model, scaler, corpus.split("calibration"), device=device,
                                  wls_chi2_alpha=wls_chi2_alpha, wls_normalized_threshold=wls_normalized_threshold)
    healthy = [r for r in predictions if r["labels"]["anomaly_mask"] and not r["labels"]["anomaly"]]
    if not healthy:
        raise ValueError("healthy calibration requires all five represented family labels known negative")
    phase_threshold = empirical_threshold([row["phase_score"] for row in healthy], false_trigger_rate)
    anomaly_threshold = empirical_threshold([row["anomaly_score"] for row in healthy], false_trigger_rate)
    both_known = all(row["wls_alarm"] is not None for row in healthy)
    family_thresholds = {
        name: empirical_threshold([row["family_scores"][i] for row in healthy], false_trigger_rate)
        for i, name in enumerate(FAMILY_NAMES) if checkpoint["trained_family_mask"][i]
    }
    wls_scores = [row.get("wls_score") for row in healthy]
    matched_wls_threshold = (empirical_threshold(wls_scores, false_trigger_rate)
                             if all(value is not None for value in wls_scores) else None)
    report = {
        "model_id": checkpoint["model_id"], "checkpoint_sha256": file_sha256(checkpoint_path),
        "phase_threshold": phase_threshold, "anomaly_threshold": anomaly_threshold,
        "family_thresholds": family_thresholds,
        "family_threshold_policy": "Separate healthy-reference quantiles; no family-wise false-trigger guarantee",
        "matched_wls_threshold": matched_wls_threshold,
        "matched_wls_score": "max(J/chi2_threshold, max_abs_normalized_residual/local_threshold); omit local term in chi-square-only mode",
        "threshold_comparison": ">", "quantile": 1 - false_trigger_rate,
        "requested_healthy_false_trigger_rate": false_trigger_rate,
        "healthy_count": len(healthy), "healthy_parent_count": len({r["parent_id"] for r in healthy}),
        "calibration_parent_ids": parent_ids, "feature_schema_version": checkpoint["feature_schema_version"],
        "measurement_convention": checkpoint["measurement_convention"],
        "calibration_kind": ("new_parent_healthy_recalibration"
                             if any(parent not in checkpoint["parent_splits"] for parent in parent_ids)
                             else "source_heldout_healthy"),
        "score_semantics": "uncalibrated sigmoid scores; thresholds are empirical healthy-reference quantiles",
        "calibration_healthy_phase_trigger_rate": float(np.mean([r["phase_score"] > phase_threshold for r in healthy])),
        "calibration_healthy_anomaly_trigger_rate": float(np.mean([r["anomaly_score"] > anomaly_threshold for r in healthy])),
        "calibration_healthy_wls_union_trigger_rate": float(np.mean([bool(r["wls_alarm"]) or r["phase_score"] > phase_threshold for r in healthy])) if both_known else None,
        "union_threshold_jointly_calibrated": False,
        "wls_comparator": {"chi2_alpha": wls_chi2_alpha, "normalized_residual_threshold": wls_normalized_threshold,
                           "rule": ("J >= chi2 threshold" if wls_normalized_threshold is None else
                                    "J >= chi2 threshold OR max(abs(rN)) >= normalized residual threshold")},
        "population_note": "Empirical calibration only; independent parent-held-out testing is required. The WLS union has its own measured rate.",
        "small_calibration_warning": len(healthy) < 10000,
        "invalid_calibration_graphs": [row for row in corpus.invalid if row["split"] == "calibration"],
    }
    write_json(output, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dir")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--false-trigger-rate", type=float, default=0.01)
    parser.add_argument("--allow-new-parents", action="store_true")
    parser.add_argument("--wls-chi2-alpha", type=float, default=0.05)
    parser.add_argument("--wls-normalized-threshold", type=float, default=4.0)
    parser.add_argument("--wls-chi-square-only", action="store_true",
                        help="disable the optional local normalized-residual gate in the WLS comparator")
    args = parser.parse_args(argv)
    import torch
    torch.set_num_threads(args.torch_threads)
    result = calibrate(args.manifest, args.checkpoint, args.output, cache_dir=args.cache_dir,
        device=args.device, false_trigger_rate=args.false_trigger_rate, allow_new_parents=args.allow_new_parents,
        workers=args.workers,
        wls_chi2_alpha=args.wls_chi2_alpha,
        wls_normalized_threshold=None if args.wls_chi_square_only else args.wls_normalized_threshold)
    print(f"phase_threshold={result['phase_threshold']:.8g}; healthy windows={result['healthy_count']}")


if __name__ == "__main__":
    main()
