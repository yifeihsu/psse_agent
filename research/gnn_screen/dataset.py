"""Explicit, parent-grouped WLS-only manifests and cached noisy graph construction.

Labels and experimental metadata never enter ``build_graph``. A manifest row
must identify the actual shared physical operating parent, before fault/noise
expansion. Scenario names and noise seeds are deliberately not fallback IDs.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import sqlite3
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .feature_schema import (FAMILY_NAMES, MEASUREMENT_CONVENTION,
                             SCHEMA_VERSION as FEATURE_SCHEMA_VERSION)

SPLITS = ("train", "validation", "calibration", "test")
ROW_KEYS = {
    "case", "z", "parent_id", "families", "severity", "split", "window_id",
    "measurement_sigma", "solver_settings", "measurement_convention",
    "measurement_kind", "noise_replicates", "noise_seed", "noise_group_id", "offline_metadata",
}


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def content_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(jsonable(value), sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _load_value(value: Any, base: Path) -> Any:
    return json.loads((base / value).read_text(encoding="utf-8-sig")) if isinstance(value, str) else value


def make_labels(families: list[str] | dict[str, Any]) -> dict[str, Any]:
    """Lists are exhaustive labels; dictionaries permit explicit unknowns."""
    if isinstance(families, list):
        if len(set(families)) != len(families) or set(families) - set(FAMILY_NAMES):
            raise ValueError("families must contain unique supported family names; harmonic is not a v1 label")
        values = [float(name in families) for name in FAMILY_NAMES]
        masks = [1.0] * len(FAMILY_NAMES)
    elif isinstance(families, dict):
        if set(families) - set(FAMILY_NAMES):
            raise ValueError("unsupported family label")
        if any(value is not None and value not in (0, 1, False, True) for value in families.values()):
            raise ValueError("family labels must be 0, 1, or null")
        values = [float(families.get(name) or 0) for name in FAMILY_NAMES]
        masks = [float(families.get(name) is not None) for name in FAMILY_NAMES]
    else:
        raise ValueError("families must be an exhaustive list or a partial label mapping")
    phase = float(any(values[:2]))
    anomaly = float(any(values))
    return {
        "family": values, "family_mask": masks,
        "phase": phase, "phase_mask": float(bool(phase) or all(masks[:2])),
        "anomaly": anomaly, "anomaly_mask": float(bool(anomaly) or all(masks)),
    }


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path).resolve()
    records = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or set(row) - ROW_KEYS:
            raise ValueError(f"manifest line {lineno}: unexpected fields; use offline_metadata for audit-only labels")
        required = {"case", "z", "parent_id", "families", "measurement_convention"}
        if required - row.keys():
            raise ValueError(f"manifest line {lineno}: missing {sorted(required - row.keys())}")
        if not isinstance(row["parent_id"], str) or not row["parent_id"].strip():
            raise ValueError("parent_id must explicitly name the physical operating parent")
        if row["measurement_convention"] != MEASUREMENT_CONVENTION:
            raise ValueError("corrected phase-A Vm / total-three-phase P/Q shunt convention must be explicitly declared")
        row = copy.deepcopy(row)
        row["case"] = _load_value(row["case"], path.parent)
        row["z"] = _load_value(row["z"], path.parent)
        if not isinstance(row["case"], dict) or not {"baseMVA", "bus", "branch"} <= row["case"].keys():
            raise ValueError("case must be an explicit configured MATPOWER case dictionary or JSON path")
        if "measurement_sigma" in row:
            row["measurement_sigma"] = _load_value(row["measurement_sigma"], path.parent)
        row["labels"] = make_labels(row["families"])
        row["severity"] = str(row.get("severity", "unspecified"))
        row["window_id"] = str(row.get("window_id", content_hash(row["z"])))
        if "noise_group_id" in row and (
            not isinstance(row["noise_group_id"], str) or not row["noise_group_id"].strip()
        ):
            raise ValueError("noise_group_id must be a nonempty string when supplied")
        if row.get("split") not in (None, *SPLITS):
            raise ValueError("split must be train, validation, calibration, or test")
        records.append(row)
    if not records:
        raise ValueError("manifest contains no records")
    return records


def assign_parent_splits(records: list[dict[str, Any]], *, seed: int = 2026,
                         fractions: tuple[float, ...] = (0.70, 0.10, 0.10, 0.10)) -> dict[str, str]:
    explicit = [record.get("split") is not None for record in records]
    if any(explicit) and not all(explicit):
        raise ValueError("supply splits for every row or leave every split unset")
    mapping: dict[str, str] = {}
    if all(explicit):
        for record in records:
            parent, split = record["parent_id"], record["split"]
            if parent in mapping and mapping[parent] != split:
                raise ValueError(f"physical parent leakage across splits: {parent}")
            mapping[parent] = split
    else:
        parents = sorted({record["parent_id"] for record in records})
        if len(parents) < 4:
            raise ValueError("at least four independent operating parents are needed for four automatic splits")
        if len(fractions) != 4 or any(v <= 0 for v in fractions) or not np.isclose(sum(fractions), 1):
            raise ValueError("four positive split fractions must sum to one")
        np.random.default_rng(seed).shuffle(parents)
        # Reserve one parent per split, then allocate the rest proportionally.
        counts = np.ones(4, dtype=int)
        remainder = np.asarray(fractions) * (len(parents) - 4)
        counts += np.floor(remainder).astype(int)
        for idx in np.argsort(-(remainder - np.floor(remainder)))[:len(parents) - int(counts.sum())]:
            counts[idx] += 1
        offset = 0
        for split, count in zip(SPLITS, counts):
            mapping.update({parent: split for parent in parents[offset:offset + count]})
            offset += count
    for record in records:
        record["split"] = mapping[record["parent_id"]]
    return mapping


@dataclass
class Sample:
    graph: dict[str, Any]
    labels: dict[str, Any]
    parent_id: str
    split: str
    severity: str
    window_id: str
    offline_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Corpus:
    samples: list[Sample]
    invalid: list[dict[str, Any]]
    parent_splits: dict[str, str]

    def split(self, name: str) -> list[Sample]:
        return [sample for sample in self.samples if sample.split == name]


def _worker_init() -> None:
    # A WLS solve has tiny dense matrices; many BLAS threads per process only
    # oversubscribe the machine. Keep one native thread in each worker.
    from threadpoolctl import threadpool_limits
    global _worker_thread_limit
    _worker_thread_limit = threadpool_limits(limits=1)


def _build_task(task, builder=None):
    from .graph_builder import build_graph
    from .wls_features import ScreenInputError
    case, observed, sigma, settings = task
    try:
        return {"graph": (builder or build_graph)(case, observed, measurement_sigma=sigma, solver_settings=settings)}
    except ScreenInputError as exc:
        return {"invalid": {"screen_status": getattr(exc, "status", "unavailable"), "reason": str(exc)}}


def _cache_code_hash() -> str:
    """Numerical implementation changes invalidate previously built graphs."""
    here = Path(__file__).resolve().parent
    paths = [here / name for name in ("feature_schema.py", "graph_builder.py", "wls_features.py")]
    paths.extend(here.parents[1] / "tools" / name for name in ("lagrangian_port.py", "branch_param_jacobian.py"))
    return content_hash({str(path.name): file_sha256(path) for path in paths})


def _encode_cache(result) -> bytes:
    stream = io.BytesIO()
    if "graph" in result:
        graph = result["graph"]
        arrays = {name: graph[name] for name in ("x", "edge_index", "edge_attr", "u", "edge_pair")}
        metadata = {"metadata": graph.get("metadata", {})}
    else:
        arrays, metadata = {}, result
    np.savez(stream, **arrays, description=np.frombuffer(json.dumps(jsonable(metadata), allow_nan=False).encode(), dtype=np.uint8))
    return stream.getvalue()


def _decode_cache(payload: bytes):
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        metadata = json.loads(archive["description"].tobytes())
        if "invalid" in metadata:
            return metadata
        return {"graph": {**{name: archive[name] for name in ("x", "edge_index", "edge_attr", "u", "edge_pair")}, **metadata}}


def prepare_corpus(manifest: str | Path, *, cache_dir: str | Path | None = None,
                   split_seed: int = 2026, graph_builder=None, workers: int = 1,
                   progress: bool = False) -> Corpus:
    """Build deterministic noisy graphs, optionally in bounded CPU workers.

    One parent process owns the compact SQLite cache. Payloads are NumPy arrays
    and JSON (never pickle), keyed by observations, covariance, solver and code.
    Labels remain in manifest rows, outside graph construction and cache keys.
    """
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
        raise ValueError("workers must be a positive integer")
    if graph_builder is not None and workers != 1:
        raise ValueError("custom graph_builder requires workers=1")
    records = load_manifest(manifest)
    parent_splits = assign_parent_splits(records, seed=split_seed)
    cache = Path(cache_dir) if cache_dir else None
    if cache:
        cache.mkdir(parents=True, exist_ok=True)
    database = sqlite3.connect(cache / "graphs.sqlite3") if cache else None
    if database:
        database.execute("CREATE TABLE IF NOT EXISTS graphs (fingerprint TEXT PRIMARY KEY, payload BLOB NOT NULL)")
    samples, invalid, pending = [], [], []
    sample_order = {}
    code_hash = _cache_code_hash()
    started, hits, completed = time.monotonic(), 0, 0

    def collect(row, window, fingerprint, result, *, store=False):
        nonlocal completed
        if "graph" in result:
            graph = result["graph"]
            graph["metadata"] = {**graph.get("metadata", {}), "cache_fingerprint": fingerprint,
                                 "measurement_window": window}
            samples.append(Sample(graph, copy.deepcopy(row["labels"]), row["parent_id"], row["split"],
                                  row["severity"], window, row.get("offline_metadata", {})))
        else:
            invalid.append({"parent_id": row["parent_id"], "split": row["split"], "window_id": window,
                            **result["invalid"], "labels": row["labels"], "severity": row["severity"]})
        if store and database:
            database.execute("INSERT OR REPLACE INTO graphs VALUES (?, ?)", (fingerprint, _encode_cache(result)))
        completed += 1
        if completed % 100 == 0:
            if database:
                database.commit()
            if progress:
                print(f"graphs={completed} cached={hits} invalid={len(invalid)} elapsed_s={time.monotonic()-started:.1f}", flush=True)

    noise_groups = {}
    for row in records:
        case, z = row["case"], np.asarray(row["z"], dtype=np.float64)
        if z.ndim != 1 or not np.all(np.isfinite(z)):
            raise ValueError("z must be a finite 1D vector in the declared measurement order")
        nb, nl = len(case["bus"]), len(case["branch"])
        sigma = np.asarray(row.get("measurement_sigma", [0.001] * nb + [0.01] * (2 * nb + 4 * nl)), dtype=np.float64)
        if sigma.shape != z.shape or not np.all(np.isfinite(sigma)) or np.any(sigma <= 0):
            raise ValueError("measurement_sigma must contain one finite positive standard deviation per z channel")
        reps = row.get("noise_replicates", 1)
        if not isinstance(reps, int) or isinstance(reps, bool) or reps < 1:
            raise ValueError("noise_replicates must be a positive integer")
        kind = row.get("measurement_kind", "observed")
        if kind not in ("observed", "noiseless_mean"):
            raise ValueError("measurement_kind must be observed or noiseless_mean")
        if kind == "observed" and (reps != 1 or "noise_seed" in row or "noise_group_id" in row):
            raise ValueError("fresh noise requires explicitly declared noiseless_mean measurements")
        if kind == "noiseless_mean" and ("measurement_sigma" not in row or "noise_seed" not in row):
            raise ValueError("noise generation requires explicit measurement_sigma and noise_seed")
        # An explicit within-parent group permits paired diagnostic arms with
        # distinct sample identities but the same standardized Gaussian draw.
        # With no group the historical per-window stream is exactly preserved.
        noise_identity = row.get("noise_group_id", row["window_id"])
        if "noise_group_id" in row:
            group_key = (row["parent_id"], noise_identity)
            group_contract = (row["noise_seed"], reps, z.size)
            if group_key in noise_groups and noise_groups[group_key] != group_contract:
                raise ValueError("paired noise group disagrees on seed, replicate count, or sensor dimension")
            noise_groups[group_key] = group_contract
        rng_seed = int(content_hash({"seed": row.get("noise_seed", 0), "parent": row["parent_id"], "window": noise_identity})[:16], 16)
        rng = np.random.default_rng(rng_seed)
        settings = {"max_it": 30, "tol": 1e-8, **row.get("solver_settings", {})}
        if set(settings) != {"max_it", "tol"}:
            raise ValueError("unsupported solver_settings; only max_it and tol are accepted")
        for replicate in range(reps):
            observed = z + rng.normal(size=z.shape) * sigma if kind == "noiseless_mean" else z.copy()
            window = row["window_id"] + (f":noise{replicate}" if kind == "noiseless_mean" else "")
            sample_order[(row["parent_id"], window)] = len(sample_order)
            fingerprint = content_hash({"schema": FEATURE_SCHEMA_VERSION, "code_hash": code_hash, "window": window,
                "case": case, "measurements": observed, "covariance_diagonal": sigma ** 2,
                "solver_settings": settings, "measurement_convention": MEASUREMENT_CONVENTION})
            cached = database.execute("SELECT payload FROM graphs WHERE fingerprint=?", (fingerprint,)).fetchone() if database else None
            if cached:
                hits += 1
                collect(row, window, fingerprint, _decode_cache(cached[0]))
            else:
                pending.append((row, window, fingerprint, (case, observed, sigma, settings)))
    if progress:
        print(f"corpus cache_hits={hits} pending={len(pending)} workers={workers}", flush=True)
    try:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            with (ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) if workers > 1 and pending else nullcontext()) as pool:
                results = (pool.map(_build_task, (item[3] for item in pending), chunksize=8) if pool else
                           (_build_task(item[3], graph_builder) for item in pending))
                for (row, window, fingerprint, _), result in zip(pending, results):
                    collect(row, window, fingerprint, result, store=True)
    finally:
        if database:
            database.commit()
            database.close()
    # Cache hits and newly built rows retain the same manifest/window order.
    samples.sort(key=lambda sample: sample_order[(sample.parent_id, sample.window_id)])
    invalid.sort(key=lambda row: sample_order[(row["parent_id"], row["window_id"])])
    if progress:
        print(f"corpus complete valid={len(samples)} invalid={len(invalid)} elapsed_s={time.monotonic()-started:.1f}", flush=True)
    return Corpus(samples, invalid, parent_splits)


def trained_family_mask(samples: list[Sample]) -> list[bool]:
    return [any(s.labels["family_mask"][i] and s.labels["family"][i] == 1 for s in samples)
            and any(s.labels["family_mask"][i] and s.labels["family"][i] == 0 for s in samples)
            for i in range(len(FAMILY_NAMES))]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    corpus = prepare_corpus(args.manifest, cache_dir=args.cache_dir, split_seed=args.split_seed, workers=args.workers, progress=True)
    write_json(args.report, {"valid_graphs": dict(Counter(s.split for s in corpus.samples)),
                            "invalid": corpus.invalid, "parent_splits": corpus.parent_splits,
                            "trained_family_mask": dict(zip(FAMILY_NAMES, trained_family_mask(corpus.split("train"))))})


if __name__ == "__main__":
    main()
