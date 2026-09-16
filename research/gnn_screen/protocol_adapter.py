"""Read-only, snapshot-bound inference from a trained and calibrated screen.

Family outputs are hypotheses, never correction or diagnosis certificates.
No state metadata or phase/harmonic telemetry is passed to the graph builder.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .feature_schema import FAMILY_NAMES
from .graph_builder import build_graph
from .wls_features import ScreenInputError


def snapshot_binding(case: Mapping[str, Any], z: Any) -> dict[str, str]:
    """Content bindings use only configured network and permitted measurements."""
    configured = {
        "baseMVA": float(case["baseMVA"]),
        # Bus load/dispatch and state guesses are deliberately absent.
        "bus": np.asarray(case["bus"], dtype=float)[:, [0, 1, 4, 5]].tolist(),
        "branch": np.asarray(case["branch"], dtype=float)[:, :13].tolist(),
    }
    encode = lambda value: json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
    return {
        "configured_model_hash": hashlib.sha256(encode(configured)).hexdigest(),
        "measurement_window_hash": hashlib.sha256(encode(np.asarray(z, dtype=float).tolist())).hexdigest(),
    }


def unavailable_report(status: str, reason: str, **binding: Any) -> dict[str, Any]:
    return {
        "screen_status": status,
        "phase_trigger": None,
        "anomaly_trigger": None,
        "family_scores": {},
        "recommendation": "screen_unavailable_retain_independent_coverage",
        "reason": reason,
        "read_only": True,
        "score_interpretation": "uncalibrated_sigmoid_scores_not_probabilities",
        **binding,
    }


class FrozenScreen:
    """Frozen model with an independent, checkpoint-bound threshold policy."""

    def __init__(self, model: Any, scaler: Any, checkpoint: Mapping[str, Any], calibration: Mapping[str, Any]):
        if tuple(checkpoint.get("family_names", ())) != tuple(FAMILY_NAMES):
            raise ValueError("Checkpoint family schema does not match the five-family screen")
        if not checkpoint.get("model_id") or calibration.get("model_id") != checkpoint["model_id"]:
            raise ValueError("Calibration must be bound to the selected model")
        if calibration.get("threshold_comparison") != ">":
            raise ValueError("Calibration must specify strict '>' threshold comparison")
        self.trained_heads = dict(checkpoint.get("trained_heads", {}))
        self.family_mask = list(checkpoint.get("trained_family_mask", ()))
        if len(self.family_mask) != len(FAMILY_NAMES):
            raise ValueError("Checkpoint must declare every trained family head")
        for head in ("phase", "anomaly"):
            threshold = calibration.get(f"{head}_threshold")
            if self.trained_heads.get(head) and (
                threshold is None or not np.isfinite(threshold) or not 0 <= float(threshold) <= 1
            ):
                raise ValueError(f"A trained {head} head requires a finite calibrated threshold")
        self.model = model.eval()
        self.scaler = scaler
        self.model_id = str(checkpoint["model_id"])
        self.calibration = dict(calibration)

    def screen(self, case: Mapping[str, Any], z: Any, *, wls_details: Mapping[str, Any] | None = None,
               state_id: str | None = None, state_hash: str | None = None) -> dict[str, Any]:
        binding: dict[str, Any] = {"model_id": self.model_id}
        if state_id is not None:
            binding["state_id"] = str(state_id)
        if state_hash is not None:
            binding["state_hash"] = str(state_hash)
        try:
            binding.update(snapshot_binding(case, z))
            graph = build_graph(case, z, wls_details=wls_details, scaler=self.scaler)
            binding["covariance_hash"] = graph["metadata"]["covariance_hash"]
            binding["solver_settings"] = graph["metadata"]["solver_settings"]
        except ScreenInputError as exc:
            return unavailable_report(exc.status, str(exc), **binding)
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            return unavailable_report("unsupported_input", str(exc), **binding)
        import torch
        from .model import collate_graphs

        with torch.no_grad():
            device = next(self.model.parameters()).device
            logits = self.model(collate_graphs([graph], device=device))
            scores = {name: torch.sigmoid(value).detach().cpu().numpy() for name, value in logits.items()}
        if any(not np.all(np.isfinite(value)) for value in scores.values()):
            return unavailable_report("model_failure", "Model emitted nonfinite scores", **binding)
        phase = float(scores["phase_screen_logit"].reshape(-1)[0]) if self.trained_heads.get("phase") else None
        anomaly = float(scores["anomaly_logit"].reshape(-1)[0]) if self.trained_heads.get("anomaly") else None
        phase_trigger = phase > self.calibration["phase_threshold"] if phase is not None else None
        anomaly_trigger = anomaly > self.calibration["anomaly_threshold"] if anomaly is not None else None
        recommendation = (
            "acquire_three_phase_context" if phase_trigger else
            "continue_balanced_investigation" if anomaly_trigger else
            "no_anomaly_detected_from_available_wls_snapshot_retain_independent_coverage"
            if phase_trigger is False and anomaly_trigger is False else
            "screen_incomplete_retain_independent_coverage"
        )
        return {
            # Put the operational evidence first for compact model views.
            "screen_status": "valid",
            "phase_trigger": phase_trigger,
            "anomaly_trigger": anomaly_trigger,
            "family_scores": {name: float(scores["family_logits"][0, i])
                              for i, name in enumerate(FAMILY_NAMES) if self.family_mask[i]},
            "recommendation": recommendation,
            "phase_score": phase,
            "anomaly_score": anomaly,
            "thresholds": {"phase": self.calibration.get("phase_threshold"),
                           "anomaly": self.calibration.get("anomaly_threshold"), "comparison": ">"},
            "read_only": True,
            "score_interpretation": "uncalibrated_sigmoid_scores_not_probabilities",
            "unsupported_families": ["harmonic"],
            "disabled_family_heads": [name for i, name in enumerate(FAMILY_NAMES) if not self.family_mask[i]],
            **binding,
        }


@lru_cache(maxsize=4)
def _load_screen(checkpoint_path: str, calibration_path: str, checkpoint_stat: tuple, calibration_stat: tuple) -> FrozenScreen:
    from .train import load_trained_model

    calibration = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
    digest = hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest()
    if calibration.get("checkpoint_sha256") != digest:
        raise ValueError("Calibration checkpoint fingerprint mismatch")
    model, scaler, checkpoint = load_trained_model(checkpoint_path, device="cpu")
    return FrozenScreen(model, scaler, checkpoint, calibration)


def load_screen(checkpoint_path: str | Path, calibration_path: str | Path) -> FrozenScreen:
    paths = [Path(checkpoint_path).resolve(), Path(calibration_path).resolve()]
    stats = [(p.stat().st_mtime_ns, p.stat().st_size) for p in paths]
    return _load_screen(str(paths[0]), str(paths[1]), *stats)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--case", required=True, help="Configured MATPOWER case path")
    parser.add_argument("--measurements", required=True, help="JSON array in Vm/Pinj/Qinj/Pf/Qf/Pt/Qt order")
    parser.add_argument("--output")
    args = parser.parse_args()
    from mcp_server.matpower_server import _load_python_case

    report = load_screen(args.checkpoint, args.calibration).screen(
        _load_python_case(args.case), json.loads(Path(args.measurements).read_text(encoding="utf-8")))
    text = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
