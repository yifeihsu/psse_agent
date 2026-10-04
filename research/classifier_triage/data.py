"""Benchmark data: study rows, truth-derived triage labels, operator WLS payloads, and the background probes.

Rows and splits are those of the step-4 ranker study
(``research.hypothesis_ranking.ranker.load_dataset``): alarmed states with a
valid screen, split by physical parent; truth-corrected children are training
augmentation only.  Every row gains

* ``triage``: the truth-derived labels a classifier is trained on and scored
  against (decision T1 of the plan): ``needs_aux`` (an HIF, an unbalance or a
  harmonic source is present, so phase-resolved measurements are needed), the
  set of balanced families present, and for mixed roots the family whose
  removal lowers the WLS objective most (the oracle-greedy first family);
* ``payload``: the operator's balanced WLS solve of the state (normalized
  residuals with their signs, normalized Lagrange multipliers, the estimated
  state, the declared sigmas and exact rows), the only evidence a triage
  classifier may read.

The background probes test what a classifier keys on.  Every waveform root of
the study comes from the OpenDSS corpora and every balanced root from the OPF
corpus, so a flexible classifier can separate "needs phasors" by the
simulator background instead of the event.  A probe row is a healthy window
(the same-operating-point OpenDSS reference of a corpus row, or a clean OPF
corpus row, with fresh sensor noise) that does not alarm, plus one biased
meter that makes it alarm: truth is a single bad meter, no auxiliary stream
needed.  A classifier that requests phasors more often on the OpenDSS probe
than on the OPF probe reads the background.  Three kinds (``PROBE_KINDS``):
a power meter, a voltage meter at the generator's magnitudes, and a voltage
meter below them.  The voltage kinds ask a second question: unbalance and
harmonics show on the phase-A voltage channels, and every bad voltage meter
of the training population is at least ten sigma, so a classifier may have
learned "a moderate voltage residual means a waveform event".
"""
from __future__ import annotations

import json
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BALANCED_FAMILIES = ("measurement", "parameter", "topology")
WAVEFORM_FAMILIES = ("hif", "unbalance", "harmonic")
DEFAULT_DATASET = REPO_ROOT / "output" / "hypothesis_ranking_20260930" / "ieee14_v3"
PROBE_SEED = 20261004
#: Probe kinds: which channel block is biased and by how many declared sigmas.  The generator's meter
#: families use at least ten sigma, so the small voltage kind is outside the training population.
PROBE_KINDS = {
    "meter": ("power", (10.0, 20.0)),
    "vm_large": ("voltage", (10.0, 20.0)),
    "vm_small": ("voltage", (4.5, 9.0)),
}


# ----------------------------------------------------------------- WLS payload

def wls_payload(record: Mapping[str, Any]) -> dict[str, Any] | None:
    """The operator's balanced WLS solve of one record, as arrays; ``None`` when it does not converge."""
    from mcp_server.matpower_server import _wls_json
    from psse_env.noise_contract import resolve_state_measurement_noise

    z = [float(value) for value in record["measurements"]]
    noise = resolve_state_measurement_noise({"metadata": record.get("metadata") or {}}, len(z))
    exact = [int(i) for i in noise["exact_measurement_indices"]]
    payload = _wls_json(str(record["case"]), z, measurement_sigma=noise["measurement_sigma"],
                        exact_measurement_indices=exact or None)
    if not payload.get("success"):
        return None
    raw = np.asarray(payload["raw_residual"], dtype=float)
    magnitude = np.abs(np.asarray(payload["r"], dtype=float))
    sigma = noise["measurement_sigma"]
    return {
        "signed_normalized_residual": np.sign(raw) * magnitude,
        "lambda_normalized": np.asarray(payload.get("lambdaN") or [], dtype=float),
        "theta": np.asarray(payload["theta_est_rad"], dtype=float),
        "vm": np.asarray(payload["vm_est_pu"], dtype=float),
        "raw_residual": raw,
        "z": np.asarray(z, dtype=float),
        "sigma": (np.asarray(sigma, dtype=float) if sigma is not None else None),
        "exact": exact,
        "objective": float(payload["global_residual_sum"]),
        "dof": int(payload["dof"]),
    }


def _payload_job(item: tuple[str, dict[str, Any]]) -> tuple[str, dict[str, Any] | None]:
    key, record = item
    try:
        return key, wls_payload(record)
    except Exception:  # an unloadable case or a solver error: the row is dropped and counted
        return key, None


def attach_payloads(rows: Sequence[dict[str, Any]], cache: Path | None = None, *, workers: int = 8) -> list[dict[str, Any]]:
    """Rows with ``payload`` attached (cached by row id); rows whose WLS fails are dropped."""
    cached: dict[str, Any] = {}
    if cache is not None and cache.is_file():
        with cache.open("rb") as stream:
            cached = pickle.load(stream)
    missing = [(str(row["id"]), {k: row["record"][k] for k in ("case", "measurements", "metadata")})
               for row in rows if str(row["id"]) not in cached]
    if missing:
        if workers > 1 and len(missing) > 16:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(_payload_job, missing, chunksize=16))
        else:
            results = [_payload_job(item) for item in missing]
        cached.update(dict(results))
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            with cache.open("wb") as stream:
                pickle.dump(cached, stream, protocol=pickle.HIGHEST_PROTOCOL)
    kept = []
    for row in rows:
        payload = cached.get(str(row["id"]))
        if payload is None:
            continue
        row["payload"] = payload
        kept.append(row)
    return kept


# ----------------------------------------------------------------------- labels

def _objective(record: Mapping[str, Any]) -> float | None:
    value = ((record.get("analysis") or {}).get("wls") or {}).get("chi_square_statistic")
    return float(value) if value is not None else None


def triage_labels(record: Mapping[str, Any], children: Sequence[Mapping[str, Any]] = ()) -> dict[str, Any]:
    """Truth-derived triage of one state (decision T1).

    ``needs_aux`` is 1 when the truth holds an HIF, an unbalance or a harmonic
    source.  ``families`` lists the balanced families present.  ``first`` is
    the family to investigate first: the only one, or on a mixed root the one
    whose truth-corrected child lowers the WLS objective most (the meter
    overlay against the branch error); ``None`` when no balanced error exists.
    """
    truth = record.get("truth") or {}
    present = [name for name in BALANCED_FAMILIES if truth.get(name)]
    waveform = [name for name in WAVEFORM_FAMILIES if truth.get(name)]
    first: str | None = present[0] if len(present) == 1 else None
    if len(present) > 1:
        base = _objective(record)
        by_kind = {str(child.get("child_kind")): _objective(child) for child in children}
        without_meter = by_kind.get("remove_meter_overlay")
        branch = next(name for name in present if name != "measurement")
        if base is not None and without_meter is not None:
            meter_effect = base - without_meter
            fixed = by_kind.get("fix_parameter")
            # Removing the branch error: measured directly when its child exists,
            # else what is left once the meter overlay is gone.
            branch_effect = (base - fixed) if fixed is not None else without_meter - float(
                ((record.get("analysis") or {}).get("wls") or {}).get("chi_square_dof") or 0.0)
            first = "measurement" if meter_effect >= branch_effect else branch
        else:
            first = branch
    return {"needs_aux": int(bool(waveform)), "families": present, "waveform": waveform, "first": first}


def load_rows(dataset_dir: Path = DEFAULT_DATASET, *, payload_cache: Path | None = None, workers: int = 8) -> list[dict[str, Any]]:
    """The step-4 study rows with triage labels and WLS payloads."""
    from research.hypothesis_ranking import ranker

    dataset_dir = Path(dataset_dir)
    rows = ranker.load_dataset(dataset_dir, include_children=True, feature_set="offline")
    children: dict[str, list[dict[str, Any]]] = {}
    for child in ranker._read(dataset_dir / "children.jsonl"):
        children.setdefault(str(child.get("child_of")), []).append(child)
    for row in rows:
        row["triage"] = triage_labels(row["record"], children.get(str(row["id"]), ()))
    cache = payload_cache if payload_cache is not None else dataset_dir.parent / "classifier_triage" / "payloads_ieee14_v3.pkl"
    return attach_payloads(rows, cache, workers=workers)


def split_rows(rows: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Train (with children), calibration and test (roots, healthy alarms and mimics only)."""
    return {
        "train": [r for r in rows if r["split"] == "train"],
        "calibration": [r for r in rows if r["split"] == "calibration" and r["kind"] != "child"],
        "test": [r for r in rows if r["split"] == "test" and r["kind"] != "child"],
    }


# ----------------------------------------------------------------------- probes

def _corpus_rows(path: Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def healthy_references(dataset_dir: Path = DEFAULT_DATASET) -> list[dict[str, Any]]:
    """Noise-free healthy measurement vectors: OpenDSS references of the waveform corpora and clean OPF rows."""
    from psse_env.providers.scenario_generator import PHYSICAL_HIF_SAMPLE_PATHS, PHYSICAL_IMBALANCE_SAMPLE_PATH
    from research.hypothesis_ranking.build_dataset import _split, build_generator

    manifest = json.loads((Path(dataset_dir) / "manifest.json").read_text(encoding="utf-8"))
    seed = int(manifest["seed"])
    references: list[dict[str, Any]] = []
    for path, prefix in [(p, "hif_window") for p in PHYSICAL_HIF_SAMPLE_PATHS] + [(PHYSICAL_IMBALANCE_SAMPLE_PATH, "unbalance_window")]:
        if not Path(path).is_file():
            continue
        for row in _corpus_rows(Path(path)):
            if not row.get("z_true") or not row.get("sigma_z"):
                continue
            parent = f"{prefix}:{row['id']}"
            references.append({"background": "opendss", "parent": parent, "split": _split(parent, seed),
                               "z_true": [float(v) for v in row["z_true"]], "sigma": [float(v) for v in row["sigma_z"]]})
    generator = build_generator(seed)
    default_sigma = [float(v) for v in generator.noise_profile().tolist()]
    for row in generator._corpus().get("no_error", []):
        clean = row.get("z_true") or row.get("z_clean")
        if not clean:
            continue
        parent = f"corpus:{row['id']}"
        references.append({"background": "opf", "parent": parent, "split": _split(parent, seed),
                           "z_true": [float(v) for v in clean],
                           "sigma": [float(v) for v in (row.get("sigma_z") or default_sigma)]})
    return references


def _probe_job(task: tuple[dict[str, Any], int, str, str]) -> dict[str, Any] | None:
    """One probe row: healthy window that does not alarm, one biased meter that does."""
    from research.hypothesis_ranking.features import analyze_state

    reference, seed, case, kind = task
    block, sigma_range = PROBE_KINDS[kind]
    rng = np.random.default_rng(seed)
    sigma = np.asarray(reference["sigma"], dtype=float)
    z = np.asarray(reference["z_true"], dtype=float) + rng.normal(0.0, 1.0, sigma.size) * sigma
    metadata = {"sigma_z": sigma.tolist()}
    healthy = analyze_state({"case": case, "measurements": z.tolist(), "metadata": metadata}, run_screen=False)
    if not healthy["wls"].get("success") or healthy["wls"]["alarm"]:
        return None  # the background alone alarms: not a clean base for the probe
    nb = int(healthy["nb"])
    channel = int(rng.integers(0, nb)) if block == "voltage" else int(rng.integers(nb, sigma.size))
    bias = float(rng.uniform(*sigma_range)) * float(sigma[channel]) * float(rng.choice((-1.0, 1.0)))
    clean_value = float(z[channel])
    z[channel] = clean_value + bias
    record = {
        "root_id": f"probe_{kind}_{reference['background']}:{reference['parent']}:{seed}",
        "family": f"probe_{kind}_{reference['background']}",
        "probe_kind": kind, "background": reference["background"], "bias_sigmas": abs(bias) / float(sigma[channel]),
        "parent_id": reference["parent"], "case": case, "measurements": z.tolist(), "metadata": metadata,
        "split": reference["split"],
        "truth": {"measurement": [{"index": channel, "observed": float(z[channel]), "clean": clean_value}],
                  "parameter": [], "topology": [], "hif": [], "unbalance": [], "harmonic": []},
    }
    record["analysis"] = analyze_state(record, run_screen=True)
    if not record["analysis"]["wls"].get("alarm"):
        return None
    return record


def build_probe(dataset_dir: Path = DEFAULT_DATASET, *, replicas: int = 2, limit_per_background: int = 600,
                workers: int = 8, case: str = "case14", seed: int = PROBE_SEED, kind: str = "meter") -> list[dict[str, Any]]:
    """Alarmed bad-meter rows of one kind on healthy OpenDSS and OPF backgrounds (see the module docstring)."""
    references = healthy_references(dataset_dir)
    offset = sorted(PROBE_KINDS).index(kind) * 1_000_003
    rng = np.random.default_rng(seed + offset)
    tasks: list[tuple[dict[str, Any], int, str, str]] = []
    for background in ("opendss", "opf"):
        pool = [r for r in references if r["background"] == background]
        order = rng.permutation(len(pool))
        budget = limit_per_background
        for replica in range(replicas):
            for position in order:
                if budget <= 0:
                    break
                tasks.append((pool[int(position)], int(seed + offset + 7919 * replica + int(position)), case, kind))
                budget -= 1
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool_executor:
            built = list(pool_executor.map(_probe_job, tasks, chunksize=8))
    else:
        built = [_probe_job(task) for task in tasks]
    return [record for record in built if record is not None]


def probe_rows(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Probe records in the row format of ``load_rows`` (features, labels, rule flags, triage)."""
    from research.hypothesis_ranking import ranker

    rows = []
    for record in records:
        screen = (record.get("analysis") or {}).get("screen") or {}
        if screen.get("status") != "valid":
            continue
        rows.append({
            "id": record["root_id"], "kind": "probe", "family": record["family"], "parent": record["parent_id"],
            "split": record.get("split") or "test", "features": ranker.features(record), "labels": ranker.labels(record),
            "rule_v3": ranker.rule_v3(record), "rule_hif": ranker.rule_hif(record),
            "vm_pick": ranker.voltage_meter_pick(record), "needs_no_phasors_family": True,
            "record": dict(record), "triage": triage_labels(record),
            "probe_kind": str(record.get("probe_kind") or "meter"),
            "background": str(record.get("background") or record["family"].rsplit("_", 1)[-1]),
        })
    return rows
