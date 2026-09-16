"""Offline supervised screen training, selected only on parent-held-out validation."""
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .dataset import (FAMILY_NAMES, FEATURE_SCHEMA_VERSION, MEASUREMENT_CONVENTION,
                      Sample, content_hash, prepare_corpus, trained_family_mask, write_json)
from .graph_builder import FeatureScaler
from .losses import screen_loss
from .model import WLSScreenGNN, collate_graphs

DEFAULT_CONFIG = {
    "model": {"hidden_dim": 128, "layers": 3, "dropout": 0.10},
    "training": {"learning_rate": 0.0003, "weight_decay": 0.0001,
                 "batch_size_graphs": 64, "max_epochs": 100,
                 "early_stopping_patience": 12, "gradient_clip_norm": 1.0,
                 "training_seeds": [0, 1, 2, 3, 4], "split_seed": 2026,
                 "healthy_false_trigger_rate": 0.01, "balanced_sampling": True},
}


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config = copy.deepcopy(DEFAULT_CONFIG)
    if path is not None:
        import yaml
        incoming = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if set(incoming) - {"model", "training", "calibration", "evaluation"}:
            raise ValueError("unknown configuration section")
        for section, values in incoming.items():
            if section in config:
                unknown = set(values) - set(config[section])
                if unknown:
                    raise ValueError(f"unknown {section} options: {sorted(unknown)}")
                config[section].update(values)
            else:
                config[section] = values
    training = config["training"]
    if not 0 < training["healthy_false_trigger_rate"] < 1:
        raise ValueError("healthy_false_trigger_rate must be between zero and one")
    if isinstance(training["training_seeds"], int):
        training["training_seeds"] = list(range(training["training_seeds"]))
    if not training["training_seeds"] or any(training[key] < 1 for key in ("max_epochs", "batch_size_graphs", "early_stopping_patience")):
        raise ValueError("training requires positive epochs, batch size, patience, and at least one seed")
    return config


def empirical_threshold(scores, false_trigger_rate: float) -> float:
    """Upper empirical healthy quantile; deployment uses strict score > threshold."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.size == 0 or not np.all(np.isfinite(scores)):
        raise ValueError("threshold calibration requires finite healthy scores")
    if not 0 < false_trigger_rate < 1:
        raise ValueError("false_trigger_rate must be between zero and one")
    return float(np.quantile(scores, 1.0 - false_trigger_rate, method="higher"))


def batch_targets(samples: list[Sample], family_mask, device) -> dict[str, torch.Tensor]:
    targets = {name: torch.tensor([sample.labels[name] for sample in samples], dtype=torch.float32, device=device)
               for name in ("phase", "phase_mask", "anomaly", "anomaly_mask", "family", "family_mask")}
    targets["family_mask"] *= torch.tensor(family_mask, dtype=torch.float32, device=device)
    return targets


class _PreparedGraphs:
    """Validate once, then gather equal-size graphs directly on the device.

    This does not assume equal connectivity. Each graph retains its own edges
    and physical-branch IDs. Mixed-size datasets use the general collator.
    """
    def __init__(self, graphs, device):
        self.graphs, self.device = graphs, device
        sizes = {(len(g["x"]), len(g["edge_attr"])) for g in graphs}
        self.arrays = None
        if len(sizes) == 1:
            for graph in graphs:
                collate_graphs([graph])
            self.nodes, self.edges = next(iter(sizes))
            self.branches = self.edges // 2
            self.arrays = {name: torch.as_tensor(np.stack([g[name] for g in graphs]),
                dtype=torch.long if name in ("edge_index", "edge_pair") else torch.float32, device=device)
                for name in ("x", "edge_index", "edge_attr", "u", "edge_pair")}

    def batch(self, indices):
        if self.arrays is None:
            return collate_graphs([self.graphs[i] for i in indices], device=self.device)
        count = len(indices)
        ids = torch.as_tensor(indices, dtype=torch.long, device=self.device)
        groups = torch.arange(count, device=self.device)
        return {
            "x": self.arrays["x"][ids].reshape(-1, 27),
            "edge_attr": self.arrays["edge_attr"][ids].reshape(-1, 40),
            "u": self.arrays["u"][ids].reshape(-1, 4),
            "edge_index": (self.arrays["edge_index"][ids] + groups[:, None, None] * self.nodes).permute(1, 0, 2).reshape(2, -1),
            "edge_pair": (self.arrays["edge_pair"][ids] + groups[:, None] * self.branches).reshape(-1),
            "node_batch": groups.repeat_interleave(self.nodes),
            "branch_batch": groups.repeat_interleave(self.branches),
        }


def predict_samples(model, scaler, samples: list[Sample], *, batch_size: int = 64,
                    device: str = "cpu", wls_chi2_alpha: float = 0.05,
                    wls_normalized_threshold: float | None = 4.0,
                    prepared: _PreparedGraphs | None = None) -> list[dict[str, Any]]:
    from scipy.stats import chi2
    if not 0 < wls_chi2_alpha < 1 or (wls_normalized_threshold is not None and
                                    (not np.isfinite(wls_normalized_threshold) or wls_normalized_threshold <= 0)):
        raise ValueError("invalid WLS comparator thresholds")
    model.eval()
    results = []
    chi_thresholds = {dof: float(chi2.ppf(1 - wls_chi2_alpha, dof))
                      for dof in {s.graph.get("metadata", {}).get("dof") for s in samples} if dof}
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            portion = samples[start:start + batch_size]
            batch = (prepared.batch(range(start, start + len(portion))) if prepared else
                     collate_graphs([scaler.transform(s.graph) for s in portion], device=device))
            outputs = model(batch)
            phase = torch.sigmoid(outputs["phase_screen_logit"]).cpu().numpy().reshape(-1)
            anomaly = torch.sigmoid(outputs["anomaly_logit"]).cpu().numpy().reshape(-1)
            families = torch.sigmoid(outputs["family_logits"]).cpu().numpy()
            for i, sample in enumerate(portion):
                meta = sample.graph.get("metadata", {})
                raw_globals = np.asarray(sample.graph["u"]).reshape(-1)
                dof = meta.get("dof")
                local_alarm = wls_normalized_threshold is not None and raw_globals[1] >= wls_normalized_threshold
                alarm = (bool(raw_globals[0] * dof >= chi_thresholds[dof]
                              or local_alarm) if dof else None)
                wls_score = float(raw_globals[0] * dof / chi_thresholds[dof]) if dof else None
                if wls_score is not None and wls_normalized_threshold is not None:
                    wls_score = max(wls_score, float(raw_globals[1] / wls_normalized_threshold))
                results.append({"parent_id": sample.parent_id, "window_id": sample.window_id,
                    "severity": sample.severity, "split": sample.split, "labels": sample.labels,
                    "phase_score": float(phase[i]), "anomaly_score": float(anomaly[i]),
                    "family_scores": families[i].tolist(), "wls_alarm": alarm, "wls_score": wls_score,
                    "offline_metadata": sample.offline_metadata})
    return results


def validation_operating_point(predictions, false_trigger_rate: float) -> dict[str, Any]:
    healthy = [r for r in predictions if r["labels"]["anomaly_mask"] and not r["labels"]["anomaly"]]
    phase = [r for r in predictions if r["labels"]["phase_mask"] and r["labels"]["phase"]]
    nonphase = [r for r in predictions if r["labels"]["phase_mask"] and not r["labels"]["phase"] and r["labels"]["anomaly"]]
    if not healthy or not phase:
        raise ValueError("validation requires independently grouped healthy and physical phase-positive samples")
    threshold = empirical_threshold([r["phase_score"] for r in healthy], false_trigger_rate)
    strata = {}
    for idx, family in enumerate(FAMILY_NAMES):
        rows = [r for r in predictions if r["labels"]["family_mask"][idx] and r["labels"]["family"][idx]]
        for severity in sorted({r["severity"] for r in rows}):
            selected = [r for r in rows if r["severity"] == severity]
            strata[f"{family}/{severity}"] = {"count": len(selected),
                "phase_trigger_rate": float(np.mean([r["phase_score"] > threshold for r in selected]))}
    return {"phase_threshold": threshold, "by_family_and_severity": strata,
            "phase_recall": float(np.mean([r["phase_score"] > threshold for r in phase])),
            "healthy_trigger_rate": float(np.mean([r["phase_score"] > threshold for r in healthy])),
            "nonphase_trigger_rate": float(np.mean([r["phase_score"] > threshold for r in nonphase])) if nonphase else 0.0}


def load_checkpoint(path: str | Path, device: str = "cpu") -> dict[str, Any]:
    # This artifact contains tensors plus simple Python metadata, never modules.
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        raise ValueError("unsupported GNN checkpoint feature schema")
    if tuple(checkpoint.get("family_names", ())) != FAMILY_NAMES:
        raise ValueError("checkpoint family order mismatch")
    if checkpoint.get("measurement_convention") != MEASUREMENT_CONVENTION:
        raise ValueError("checkpoint measurement convention mismatch")
    return checkpoint


def load_trained_model(path: str | Path, device: str = "cpu"):
    checkpoint = load_checkpoint(path, device)
    model = WLSScreenGNN(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    scaler = FeatureScaler.from_dict(checkpoint["scaler"])
    return model, scaler, checkpoint


def train(manifest: str | Path, output_dir: str | Path, *, config: dict | None = None,
          cache_dir: str | Path | None = None, device: str = "cpu", workers: int = 1,
          progress: bool = False, resume: bool = False) -> dict[str, Any]:
    config = copy.deepcopy(config or DEFAULT_CONFIG)
    settings = config["training"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus = prepare_corpus(manifest, cache_dir=cache_dir, split_seed=settings["split_seed"], workers=workers, progress=progress)
    training, validation = corpus.split("train"), corpus.split("validation")
    if not training or not validation:
        raise ValueError("training and validation splits must contain valid WLS graphs")
    for head in ("phase", "anomaly"):
        known = {s.labels[head] for s in training if s.labels[head + "_mask"]}
        if known != {0.0, 1.0}:
            raise ValueError(f"{head} head requires both known positive and known negative training labels")
    family_mask = trained_family_mask(training)
    scaler = FeatureScaler().fit([s.graph for s in training], split="train")
    scaled = [scaler.transform(s.graph) for s in training]
    prepared_training = _PreparedGraphs(scaled, device)
    prepared_validation = _PreparedGraphs([scaler.transform(s.graph) for s in validation], device)
    all_targets = batch_targets(training, family_mask, device)
    strata = [str((tuple(s.labels["family"]), tuple(s.labels["family_mask"]), s.severity)) for s in training]
    counts = Counter(strata)
    sample_weights = torch.tensor([1.0 / counts[key] for key in strata], dtype=torch.double)
    global_best, history, seed_results = None, [], []
    run_id = content_hash({"config": config, "scaler": scaler.to_dict(), "samples": [
        {"graph": s.graph["metadata"]["cache_fingerprint"], "labels": s.labels, "parent": s.parent_id,
         "split": s.split, "severity": s.severity} for s in corpus.samples], "invalid": corpus.invalid})
    restored = None
    if resume:
        restored = torch.load(output_dir / "latest.pt", map_location="cpu", weights_only=True)
        if restored["run_id"] != run_id:
            raise ValueError("resume requires the same physical corpus, labels and training configuration")
        global_best, history, seed_results = restored["global_best"], restored["history"], restored["seed_results"]
    started = time.monotonic()
    for seed_index, seed in enumerate(settings["training_seeds"]):
        if restored and seed_index < restored["seed_index"]:
            continue
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))
        model = WLSScreenGNN(**config["model"]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
        generator = torch.Generator().manual_seed(int(seed))
        best_key, patience = None, 0
        first_epoch = 1
        if restored and seed_index == restored["seed_index"]:
            model.load_state_dict(restored["model_state"])
            optimizer.load_state_dict(restored["optimizer_state"])
            generator.set_state(restored["sampling_rng"])
            torch.set_rng_state(restored["torch_rng"])
            if torch.cuda.is_available() and restored["cuda_rng"] is not None:
                torch.cuda.set_rng_state_all(restored["cuda_rng"])
            best_key, patience, seed_best = tuple(restored["best_key"]), restored["patience"], restored["seed_best"]
            first_epoch = restored["epoch"] + 1
            if patience >= settings["early_stopping_patience"]:
                first_epoch = settings["max_epochs"] + 1
        for epoch in range(first_epoch, settings["max_epochs"] + 1):
            epoch_started = time.monotonic()
            model.train()
            order = (torch.multinomial(sample_weights, len(training), replacement=True, generator=generator).tolist()
                     if settings["balanced_sampling"] else torch.randperm(len(training), generator=generator).tolist())
            loss_total, seen = 0.0, 0
            for start in range(0, len(order), settings["batch_size_graphs"]):
                indices = order[start:start + settings["batch_size_graphs"]]
                batch = prepared_training.batch(indices)
                selection = torch.as_tensor(indices, dtype=torch.long, device=device)
                targets = {name: values[selection] for name, values in all_targets.items()}
                optimizer.zero_grad(set_to_none=True)
                loss = screen_loss(model(batch), targets)["loss"]
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("nonfinite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip_norm"])
                optimizer.step()
                loss_total += float(loss.detach().cpu()) * len(indices)
                seen += len(indices)
            predictions = predict_samples(model, scaler, validation, batch_size=settings["batch_size_graphs"], device=device,
                                          prepared=prepared_validation)
            operating = validation_operating_point(predictions, settings["healthy_false_trigger_rate"])
            # Tie-breaking also uses validation, never the calibration/test set.
            val_losses = []
            with torch.no_grad():
                for start in range(0, len(validation), settings["batch_size_graphs"]):
                    chunk = validation[start:start + settings["batch_size_graphs"]]
                    outputs = model(prepared_validation.batch(range(start, start + len(chunk))))
                    val_losses.extend([float(screen_loss(outputs, batch_targets(chunk, family_mask, device))["loss"].cpu())] * len(chunk))
            val_loss = float(np.mean(val_losses))
            key = (operating["phase_recall"], -operating["nonphase_trigger_rate"], -val_loss)
            history.append({"seed": int(seed), "epoch": epoch, "training_loss": loss_total / seen,
                            "validation_loss": val_loss, "epoch_seconds": time.monotonic() - epoch_started, **operating})
            improved = best_key is None or key > best_key
            if improved:
                best_key, patience = key, 0
                seed_best = {"seed": int(seed), "epoch": epoch, "selection_key": list(key),
                             "checkpoint": str((output_dir / f"seed_{seed}" / "checkpoint.pt").resolve()), **operating}
            else:
                patience += 1
            if improved:
                state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
                buffer = io.BytesIO()
                torch.save(state, buffer)
                model_id = hashlib.sha256(buffer.getvalue() + content_hash(scaler.to_dict()).encode()).hexdigest()
                checkpoint = {"model_config": config["model"], "model_state": state,
                    "scaler": scaler.to_dict(), "trained_family_mask": family_mask,
                    "trained_heads": {"phase": True, "anomaly": True}, "family_names": list(FAMILY_NAMES),
                    "model_id": model_id, "parent_splits": corpus.parent_splits,
                    "selected_seed": int(seed), "selected_epoch": epoch, "selection_key": list(key),
                    "validation_operating_point": operating, "training_config": settings,
                    "feature_schema_version": FEATURE_SCHEMA_VERSION, "measurement_convention": MEASUREMENT_CONVENTION,
                    "score_semantics": "uncalibrated sigmoid scores; thresholds require independent calibration"}
                seed_dir = output_dir / f"seed_{seed}"
                seed_dir.mkdir(exist_ok=True)
                torch.save(checkpoint, seed_dir / "checkpoint.pt")
                if global_best is None or key > tuple(global_best["selection_key"]):
                    torch.save(checkpoint, output_dir / "checkpoint.pt")
                    global_best = {"selection_key": list(key), "seed": int(seed), "epoch": epoch, "model_id": model_id}
            latest = {"run_id": run_id, "seed_index": seed_index, "epoch": epoch,
                "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                "sampling_rng": generator.get_state(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "best_key": list(best_key), "patience": patience, "seed_best": seed_best,
                "global_best": global_best, "history": history, "seed_results": seed_results}
            temporary = output_dir / "latest.tmp.pt"
            torch.save(latest, temporary)
            temporary.replace(output_dir / "latest.pt")
            write_json(output_dir / "training_progress.json", {"status": "training", "latest_epoch": history[-1],
                "selected": global_best, "completed_seeds": seed_results, "elapsed_this_process_s": time.monotonic() - started,
                "checkpoint": str((output_dir / "checkpoint.pt").resolve())})
            if progress:
                print(f"seed={seed} epoch={epoch} loss={loss_total/seen:.6f} val_loss={val_loss:.6f} "
                      f"phase_recall={operating['phase_recall']:.4f} nonphase_trigger={operating['nonphase_trigger_rate']:.4f} "
                      f"patience={patience} seconds={time.monotonic()-epoch_started:.1f}", flush=True)
            if patience >= settings["early_stopping_patience"]:
                break
        seed_results.append(seed_best)
    report = {"selected": global_best, "seeds": seed_results, "history": history,
              "parent_splits": corpus.parent_splits, "invalid_graphs": corpus.invalid,
              "valid_graph_counts": dict(Counter(s.split for s in corpus.samples)),
              "trained_family_mask": dict(zip(FAMILY_NAMES, family_mask)),
              "checkpoint": str((output_dir / "checkpoint.pt").resolve()),
              "run_id": run_id, "runtime": {"device": str(device), "torch_version": str(torch.__version__),
                  "elapsed_this_process_s": time.monotonic() - started, "resume": resume},
              "status": "trained_uncalibrated", "independent_test_evaluated": False}
    write_json(output_dir / "training_report.json", report)
    write_json(output_dir / "training_progress.json", {"status": "trained_uncalibrated", "selected": global_best,
               "completed_seeds": seed_results, "checkpoint": report["checkpoint"]})
    return report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config")
    parser.add_argument("--cache-dir")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    torch.set_num_threads(args.torch_threads)
    result = train(args.manifest, args.output_dir, config=load_config(args.config), cache_dir=args.cache_dir, device=args.device,
                   workers=args.workers, progress=True, resume=args.resume)
    print(result["checkpoint"])


if __name__ == "__main__":
    main()
