"""Train the triage GNN on the study's train split and score every evaluation row.

    python -m research.classifier_triage.train_gnn --output-dir output/classifier_triage_20261004/gnn_residual --view residual

Training rows are the train split with its truth-corrected children; a
tenth of the train parents is held out for early stopping, so the
calibration split stays untouched for thresholds.  Each seed trains one
model; the reported score is the mean probability over seeds.  ``--probe``
adds the background probe rows to the scored set; with
``--augment-probe`` the probe rows of train parents also enter training
(the background-decorrelation variant) and only the others are scored.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.classifier_triage import data, features  # noqa: E402

VALIDATION_FRACTION = 0.10
DEFAULTS = {"hidden_dim": 128, "layers": 3, "dropout": 0.1, "learning_rate": 3e-4, "weight_decay": 1e-4,
            "batch_size": 64, "max_epochs": 150, "patience": 15, "family_weight": 0.5, "first_weight": 0.5,
            "clip_norm": 1.0}
IGNORE = -100


def targets(row: Mapping[str, Any]) -> tuple[float, list[float], int]:
    """``needs_aux``, the six family labels and the first-family class of one row, from truth."""
    from research.classifier_triage.model import FAMILY_HEADS, FIRST_CLASSES

    truth = (row["record"].get("truth") or {})
    present = {name for name in FAMILY_HEADS if truth.get(name)}
    first = row["triage"].get("first")
    return (float(row["triage"]["needs_aux"]), [float(name in present) for name in FAMILY_HEADS],
            FIRST_CLASSES.index(first) if first in FIRST_CLASSES else IGNORE)


def in_validation(parent: str, seed: int = 0) -> bool:
    digest = hashlib.sha256(f"triage-validation:{seed}:{parent}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF < VALIDATION_FRACTION


def graphs_for(rows: Sequence[Mapping[str, Any]], view: str) -> list[dict[str, np.ndarray]]:
    return [features.build_graph(str(row["record"]["case"]), row["payload"], view) for row in rows]


def _tensorize(graphs, rows, device):
    import torch

    from research.classifier_triage.model import collate

    aux, family, first = zip(*(targets(row) for row in rows)) if rows else ((), (), ())
    return {"graphs": graphs, "aux": torch.tensor(aux, dtype=torch.float32, device=device),
            "family": torch.tensor(family, dtype=torch.float32, device=device),
            "first": torch.tensor(first, dtype=torch.long, device=device), "collate": collate}


def _predict(model, graphs, device, batch_size: int = 256) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch

    from research.classifier_triage.model import collate

    model.eval()
    aux, family, first = [], [], []
    with torch.no_grad():
        for start in range(0, len(graphs), batch_size):
            out = model(collate(graphs[start:start + batch_size], device))
            aux.append(torch.sigmoid(out["needs_aux"]).cpu().numpy())
            family.append(torch.sigmoid(out["family"]).cpu().numpy())
            first.append(torch.softmax(out["first"], dim=-1).cpu().numpy())
    if not aux:
        return np.zeros(0), np.zeros((0, 6)), np.zeros((0, 3))
    return np.concatenate(aux), np.concatenate(family), np.concatenate(first)


def train_one(train_graphs, train_rows, val_graphs, val_rows, view: str, seed: int, device: str,
              settings: Mapping[str, Any], log=print):
    """One seed: AdamW with early stopping on the validation loss; returns the best model."""
    import torch
    from torch.nn import functional as F

    from research.classifier_triage.model import TriageGNN, collate

    torch.manual_seed(seed)
    np.random.seed(seed)
    node_dim, edge_dim, global_dim = features.dims(view)
    model = TriageGNN(node_dim, edge_dim, global_dim, hidden_dim=settings["hidden_dim"], layers=settings["layers"],
                      dropout=settings["dropout"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    train = _tensorize(train_graphs, train_rows, device)
    val = _tensorize(val_graphs, val_rows, device)
    generator = torch.Generator().manual_seed(seed)

    def loss_of(out, aux, family, first):
        loss = (F.binary_cross_entropy_with_logits(out["needs_aux"], aux)
                + settings["family_weight"] * F.binary_cross_entropy_with_logits(out["family"], family))
        if bool((first != IGNORE).any()):
            loss = loss + settings["first_weight"] * F.cross_entropy(out["first"], first, ignore_index=IGNORE)
        return loss

    best_loss, best_state, stale, best_epoch = float("inf"), None, 0, 0
    for epoch in range(1, int(settings["max_epochs"]) + 1):
        model.train()
        order = torch.randperm(len(train_graphs), generator=generator).tolist()
        for start in range(0, len(order), int(settings["batch_size"])):
            indices = order[start:start + int(settings["batch_size"])]
            batch = collate([train_graphs[i] for i in indices], device)
            loss = loss_of(model(batch), train["aux"][indices], train["family"][indices], train["first"][indices])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings["clip_norm"])
            optimizer.step()
        model.eval()
        with torch.no_grad():
            total = 0.0
            for start in range(0, len(val_graphs), 256):
                indices = list(range(start, min(start + 256, len(val_graphs))))
                out = model(collate([val_graphs[i] for i in indices], device))
                total += float(loss_of(out, val["aux"][indices], val["family"][indices], val["first"][indices])) * len(indices)
            val_loss = total / max(len(val_graphs), 1)
        if val_loss < best_loss - 1e-5:
            best_loss, stale, best_epoch = val_loss, 0, epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= int(settings["patience"]):
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    log(f"[gnn] view {view} seed {seed}: best epoch {best_epoch}, validation loss {best_loss:.4f}")
    return model, {"seed": seed, "best_epoch": best_epoch, "validation_loss": best_loss}


def train_and_score(rows: Sequence[dict[str, Any]], extra: Sequence[dict[str, Any]] = (), *, view: str = "residual",
                    seeds: Sequence[int] = (0, 1, 2, 3, 4), device: str | None = None, augment: Sequence[dict[str, Any]] = (),
                    settings: Mapping[str, Any] | None = None, checkpoint_dir: Path | None = None, log=print) -> dict[str, Any]:
    """Train one model per seed and score calibration, test and ``extra`` rows.

    ``augment`` rows join the training set (they must not appear in ``extra``).
    Returns ids, per-seed probabilities and their mean for ``needs_aux`` and the six families.
    """
    import torch

    settings = {**DEFAULTS, **(settings or {})}
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    split = data.split_rows(rows)
    fit_rows = list(split["train"]) + list(augment)
    train_rows = [r for r in fit_rows if not in_validation(str(r["parent"]))]
    val_rows = [r for r in fit_rows if in_validation(str(r["parent"]))]
    scored = list(split["calibration"]) + list(split["test"]) + list(extra)
    started = time.perf_counter()
    train_graphs, val_graphs, scored_graphs = (graphs_for(group, view) for group in (train_rows, val_rows, scored))
    log(f"[gnn] view {view}: train {len(train_rows)} validation {len(val_rows)} scored {len(scored)} "
        f"(graphs built in {time.perf_counter() - started:.1f} s) on {device}")
    aux_runs, family_runs, first_runs, history = [], [], [], []
    for seed in seeds:
        model, info = train_one(train_graphs, train_rows, val_graphs, val_rows, view, int(seed), device, settings, log)
        aux, family, first = _predict(model, scored_graphs, device)
        aux_runs.append(aux)
        family_runs.append(family)
        first_runs.append(first)
        history.append(info)
        if checkpoint_dir is not None:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "config": model.config, "view": view, "seed": int(seed),
                        "settings": dict(settings)}, checkpoint_dir / f"seed_{int(seed)}.pt")
    # Cost of one decision: feature build plus a forward pass, one graph, CPU.
    probe_row = scored[0]
    cpu_model = model.to("cpu")
    started = time.perf_counter()
    repeats = 50
    for _ in range(repeats):
        graph = features.build_graph(str(probe_row["record"]["case"]), probe_row["payload"], view)
        _predict(cpu_model, [graph], "cpu")
    seconds = (time.perf_counter() - started) / repeats
    return {
        "view": view, "ids": [str(r["id"]) for r in scored], "seeds": [int(s) for s in seeds], "history": history,
        "needs_aux_by_seed": np.stack(aux_runs, axis=1), "family_by_seed": np.stack(family_runs, axis=1),
        "needs_aux": np.mean(np.stack(aux_runs, axis=1), axis=1), "family": np.mean(np.stack(family_runs, axis=1), axis=1),
        "first": np.mean(np.stack(first_runs, axis=1), axis=1),
        "seconds_per_decision_cpu": seconds, "train_rows": len(train_rows), "validation_rows": len(val_rows),
        "parameters": int(sum(p.numel() for p in model.parameters())), "settings": dict(settings),
    }


def score_with_checkpoints(checkpoint_dir: Path, rows: Sequence[Mapping[str, Any]], view: str,
                           device: str = "cpu") -> dict[str, np.ndarray]:
    """Mean probabilities of the saved per-seed models on ``rows`` (no training)."""
    import torch

    from research.classifier_triage.model import TriageGNN

    graphs = graphs_for(rows, view)
    runs = []
    for path in sorted(Path(checkpoint_dir).glob("seed_*.pt")):
        saved = torch.load(path, map_location=device, weights_only=False)
        if saved.get("view") != view:
            raise ValueError(f"{path} was trained on view {saved.get('view')!r}, not {view!r}")
        config = saved["config"]
        model = TriageGNN(config["node_dim"], config["edge_dim"], config["global_dim"], hidden_dim=config["hidden_dim"],
                          layers=config["layers"], dropout=config["dropout"]).to(device)
        model.load_state_dict(saved["state_dict"])
        runs.append(_predict(model, graphs, device))
    if not runs:
        raise FileNotFoundError(f"no seed checkpoints under {checkpoint_dir}")
    return {"needs_aux": np.mean([r[0] for r in runs], axis=0), "family": np.mean([r[1] for r in runs], axis=0),
            "first": np.mean([r[2] for r in runs], axis=0)}


def save_scores(result: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: (value.tolist() if isinstance(value, np.ndarray) else value) for key, value in result.items()}
    path.write_text(json.dumps(payload), encoding="utf-8")


def load_scores(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("needs_aux_by_seed", "family_by_seed", "needs_aux", "family", "first"):
        payload[key] = np.asarray(payload[key], dtype=float)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", default=str(data.DEFAULT_DATASET))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--view", choices=features.VIEWS, default="residual")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--probe", help="probe records (jsonl) to score, from benchmark --build-probe")
    parser.add_argument("--augment-probe", action="store_true", help="train on the probe rows of train parents")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device")
    args = parser.parse_args(argv)
    out = Path(args.output_dir)
    rows = data.load_rows(Path(args.dataset_dir), workers=args.workers)
    extra: list[dict[str, Any]] = []
    augment: list[dict[str, Any]] = []
    if args.probe:
        with Path(args.probe).open(encoding="utf-8") as stream:
            probe = data.probe_rows(json.loads(line) for line in stream if line.strip())
        probe = data.attach_payloads(probe, out / "probe_payloads.pkl", workers=args.workers)
        if args.augment_probe:
            augment = [r for r in probe if r["split"] == "train"]
            extra = [r for r in probe if r["split"] != "train"]
        else:
            extra = probe
    result = train_and_score(rows, extra, view=args.view, seeds=range(args.seeds), device=args.device, augment=augment,
                             checkpoint_dir=out / "checkpoints")
    save_scores(result, out / "scores.json")
    print(f"[gnn] wrote {out / 'scores.json'}: {len(result['ids'])} rows, {result['parameters']} parameters, "
          f"{result['seconds_per_decision_cpu'] * 1000:.1f} ms per decision on CPU")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
