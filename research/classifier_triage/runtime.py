"""The triage classifier at runtime: a frozen GNN ensemble that reads one balanced WLS solve.

    python -m research.classifier_triage.runtime export --checkpoints output/.../gnn_residual/checkpoints \\
        --benchmark output/.../benchmark.json --output psse_env/oracle/models/triage_gnn_ieee14_20261004

Under ``classifier_gated_diagnostics`` the WLS provider loads the exported
directory once and, after every solve, writes a ``triage`` report on the WLS
ledger entry: the mean request probability of the seeds against the
calibrated threshold (decision G1) and the balanced family to investigate
first.  The report reads nothing but the solve and the operator's current
case; it never sees truth.  The six family heads are not reported
(2026-10-07): on IEEE 14 the waveform families are told apart from a bad
voltage meter on balanced data only through simulator artifacts, so their
scores stay out of the policy's view; ``TriageClassifier.scores`` still
returns them for offline study.

``export`` turns the benchmark's per-seed checkpoints into the runtime
directory: ``runtime.json`` (contract, view, threshold and where it came
from, family names, checkpoint files, model id) beside the checkpoints
stored in half precision.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.classifier_triage import features  # noqa: E402
from research.classifier_triage.model import FAMILY_HEADS, FIRST_CLASSES  # noqa: E402

RUNTIME_CONTRACT = "triage_gnn_runtime_v1"
METHOD = "triage_gnn"
DEFAULT_TRIAGE_CLASSIFIER = REPO_ROOT / "psse_env" / "oracle" / "models" / "triage_gnn_ieee14_20261004"


def payload_from_wls(payload: Mapping[str, Any]) -> dict[str, Any]:
    """The feature payload of a provider WLS payload (``_wls_json``): signed residuals, multipliers, objective, dof."""
    magnitude = np.asarray(payload.get("r") or [], dtype=float)
    raw = np.asarray(payload.get("raw_residual") or [], dtype=float)
    if raw.shape != magnitude.shape:
        raise ValueError("WLS payload lacks aligned normalized and raw residuals")
    return {
        "signed_normalized_residual": np.sign(raw) * magnitude,
        "lambda_normalized": np.asarray(payload.get("lambdaN") or [], dtype=float),
        "objective": float(payload.get("global_residual_sum") or 0.0),
        "dof": int(payload.get("dof") or 1),
    }


class TriageClassifier:
    """A frozen ensemble with its calibrated threshold (see the module docstring)."""

    method = METHOD

    def __init__(self, models: Sequence[Any], manifest: Mapping[str, Any]) -> None:
        if not models:
            raise ValueError("a triage classifier needs at least one model")
        if tuple(manifest.get("families") or ()) != tuple(FAMILY_HEADS) or tuple(manifest.get("first_classes") or ()) != tuple(FIRST_CLASSES):
            raise ValueError("runtime manifest does not match the model's family heads")
        if manifest.get("view") != "residual":
            raise ValueError("the runtime classifier supports the residual view only")
        threshold = float(manifest["request_threshold"])
        if not 0.0 < threshold < 1.0:
            raise ValueError("request_threshold must lie strictly between 0 and 1")
        self.models = list(models)
        self.manifest = dict(manifest)
        self.view = str(manifest["view"])
        self.threshold = threshold
        self.model_id = str(manifest["model_id"])

    def scores(self, case: Mapping[str, Any] | str, feature_payload: Mapping[str, Any]) -> dict[str, Any]:
        """Mean probabilities of the seeds on one solve."""
        import torch

        from research.classifier_triage.model import collate

        graph = features.build_graph(case, feature_payload, self.view)
        batch = collate([graph], "cpu")
        aux, family, first = [], [], []
        with torch.no_grad():
            for model in self.models:
                out = model(batch)
                aux.append(torch.sigmoid(out["needs_aux"]).reshape(-1)[0].item())
                family.append(torch.sigmoid(out["family"]).reshape(-1).numpy())
                first.append(torch.softmax(out["first"], dim=-1).reshape(-1).numpy())
        return {"needs_aux": float(np.mean(aux)), "family": np.mean(family, axis=0), "first": np.mean(first, axis=0)}

    def report(self, case: Mapping[str, Any] | str, wls_payload: Mapping[str, Any]) -> dict[str, Any]:
        """The policy-visible triage report of one WLS solve."""
        scores = self.scores(case, payload_from_wls(wls_payload))
        if not (np.isfinite(scores["needs_aux"]) and np.all(np.isfinite(scores["family"])) and np.all(np.isfinite(scores["first"]))):
            return {"method": METHOD, "model_id": self.model_id, "status": "unavailable",
                    "reason": "nonfinite scores", "request_admitted": False}
        return {
            "method": METHOD,
            "model_id": self.model_id,
            "status": "valid",
            "request_score": round(scores["needs_aux"], 4),
            "request_threshold": round(self.threshold, 4),
            "request_admitted": bool(scores["needs_aux"] >= self.threshold),
            "first_family": FIRST_CLASSES[int(np.argmax(scores["first"]))],
            "first_family_scores": {name: round(float(v), 3) for name, v in zip(FIRST_CLASSES, scores["first"])},
            "score_interpretation": "mean of the seeds' probabilities",
        }


def _model_id(paths: Sequence[Path], threshold: float) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    digest.update(f"{threshold:.6f}".encode())
    return f"{METHOD}:{digest.hexdigest()[:16]}"


@lru_cache(maxsize=4)
def _load(directory: str, stamp: tuple) -> TriageClassifier:
    import torch

    from research.classifier_triage.model import TriageGNN

    root = Path(directory)
    manifest = json.loads((root / "runtime.json").read_text(encoding="utf-8"))
    if manifest.get("contract") != RUNTIME_CONTRACT:
        raise ValueError(f"{root} is not a {RUNTIME_CONTRACT} directory")
    models = []
    for name in manifest["checkpoints"]:
        saved = torch.load(root / name, map_location="cpu", weights_only=False)
        if saved.get("view") != manifest["view"]:
            raise ValueError(f"{name} was trained on view {saved.get('view')!r}, not {manifest['view']!r}")
        config = saved["config"]
        model = TriageGNN(config["node_dim"], config["edge_dim"], config["global_dim"], hidden_dim=config["hidden_dim"],
                          layers=config["layers"], dropout=config["dropout"])
        model.load_state_dict({key: value.float() for key, value in saved["state_dict"].items()})
        models.append(model.eval())
    return TriageClassifier(models, manifest)


def load_triage_classifier(directory: str | Path) -> TriageClassifier:
    """The classifier exported under ``directory`` (cached by path and file stamps)."""
    root = Path(directory).resolve()
    if not (root / "runtime.json").is_file():
        raise FileNotFoundError(f"no runtime.json under {root}")
    stamp = tuple(sorted((item.name, item.stat().st_size, item.stat().st_mtime_ns) for item in root.iterdir() if item.is_file()))
    return _load(str(root), stamp)


def export_runtime(checkpoint_dir: Path, benchmark_json: Path, output: Path, *, classifier_name: str = "gnn_residual") -> dict[str, Any]:
    """Write the runtime directory from the benchmark's per-seed checkpoints and its calibrated threshold."""
    import torch

    benchmark = json.loads(Path(benchmark_json).read_text(encoding="utf-8"))
    result = next(item for item in benchmark["results"] if item["name"] == classifier_name)
    thresholds = result["request"]["thresholds"]
    threshold = float(thresholds["at_rule_recall"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    names = []
    view = None
    for path in sorted(Path(checkpoint_dir).glob("seed_*.pt")):
        saved = torch.load(path, map_location="cpu", weights_only=False)
        view = saved["view"] if view is None else view
        if saved["view"] != view:
            raise ValueError("checkpoints of mixed views")
        torch.save({"state_dict": {key: value.half() for key, value in saved["state_dict"].items()}, "config": saved["config"],
                    "view": saved["view"], "seed": saved["seed"], "settings": saved.get("settings")}, output / path.name)
        names.append(path.name)
    if not names:
        raise FileNotFoundError(f"no seed checkpoints under {checkpoint_dir}")
    manifest = {
        "contract": RUNTIME_CONTRACT,
        "method": METHOD,
        "model_id": _model_id([output / name for name in names], threshold),
        "view": view,
        "request_threshold": threshold,
        "threshold_rule": ("the screen rule's recall on the calibration split "
                           f"({thresholds['rule_recall_calibration']:.4f}); mean of the seeds' request probabilities"),
        "families": list(FAMILY_HEADS),
        "first_classes": list(FIRST_CLASSES),
        "checkpoints": names,
        "source": {"benchmark": str(benchmark_json), "classifier": classifier_name, "checkpoints": str(checkpoint_dir),
                   "dataset": benchmark["meta"].get("dataset_dir"), "train_rows": result.get("train_rows"),
                   "parameters": result.get("parameters"), "test_auc": result["request"]["auc"]["point"]},
        "storage": "state_dict in float16, cast to float32 when loaded",
    }
    (output / "runtime.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export", help="write a runtime directory from benchmark checkpoints")
    export.add_argument("--checkpoints", required=True)
    export.add_argument("--benchmark", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--classifier", default="gnn_residual")
    args = parser.parse_args(argv)
    manifest = export_runtime(Path(args.checkpoints), Path(args.benchmark), Path(args.output), classifier_name=args.classifier)
    print(json.dumps({k: manifest[k] for k in ("model_id", "view", "request_threshold", "checkpoints")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
