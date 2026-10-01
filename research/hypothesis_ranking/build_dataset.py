"""Step 1 dataset: balanced WLS ledgers and screen reports over every family, with offline truth.

Roots are built with the round-0 generator's own family builders (the same
admission the DAgger suites use: WLS alarm with margin 1.25, development-style
parameter ranking at threshold 1.0 with rank allowance 2, meter errors lifted
to ten sigma) but called row by row so every root records the physical parent
it came from: the corpus row or HIF/unbalance window, or its own synthesized
operating point.  Splits are assigned by parent, never by noise replica, and a
window's HIF root and its measurement+HIF overlay share one parent.

Besides the roots the builder writes:

* ``children.jsonl``: truth-corrected child states of the mixed and
  multi-meter roots (the overlay meter restored, the largest meter restored,
  the branch parameter restored), labelled with what remains;
* ``healthy_alarms.jsonl``: the clean corpus windows whose noise alone alarms
  the WLS, the population an acquisition rule would fire on for nothing;
* ``mimic.jsonl``: purpose-built two-flow-meter roots (both meters of one
  line biased ten to fifteen sigma), the known HIF mimic.

Usage (from the repository root):

    python -m research.hypothesis_ranking.build_dataset --output-dir output/hypothesis_ranking_20260930/ieee14 --workers 16
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from psse_env.evidence_profile import SUSPICION_GATED_PROFILE  # noqa: E402
from psse_env.providers.hif_screen import default_hif_lines  # noqa: E402
from psse_env.providers.scenario_generator import (  # noqa: E402
    PHYSICAL_HIF_SAMPLE_PATHS, PHYSICAL_IMBALANCE_SAMPLE_PATH, Round0ScenarioGenerator, ScenarioRejected,
)
from research.hypothesis_ranking.features import analyze_state, flow_channel_index, json_safe  # noqa: E402

DEFAULT_PLAN: dict[str, int] = {
    "no_error": 250,
    "measurement": 250,
    "multi_measurement": 250,
    "parameter": 250,
    "measurement+parameter": 250,
    "topology": 250,
    "measurement+topology": 250,
    "harmonic": 250,
    "hif": 200,
    "measurement+hif": 200,
    "three_phase_unbalance": 200,
}
SPLIT_FRACTIONS = (("train", 0.60), ("calibration", 0.20), ("test", 0.20))
_METADATA_KEYS = ("sigma_z", "structural_zero_indices", "operator_noise")


# --------------------------------------------------------------------- helpers


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return None


def _scalars(mapping: Mapping[str, Any] | None, *, keep_lists: Iterable[str] = ()) -> dict[str, Any]:
    """Scalar fields of a truth label (plus named short lists), so records stay small."""
    keep = set(keep_lists)
    out: dict[str, Any] = {}
    for key, value in (mapping or {}).items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            out[str(key)] = value
        elif key in keep:
            out[str(key)] = json_safe(value)
    return out


def _truth(scenario: Mapping[str, Any]) -> dict[str, Any]:
    hidden = scenario.get("hidden_truth") if isinstance(scenario.get("hidden_truth"), Mapping) else {}
    unbalance = []
    for label in hidden.get("true_unbalance_errors") or []:
        item = _scalars(label)
        split = label.get("load_split") if isinstance(label, Mapping) else None
        if isinstance(split, Mapping):
            item["load_split_fractions"] = json_safe(split.get("fractions"))
        unbalance.append(item)
    return {
        "measurement": [
            {"index": int(m["index"]), "observed": float(m["observed"]), "clean": float(m["clean"]),
             "channel": m.get("channel")}
            for m in scenario.get("true_measurement_errors") or []
        ],
        "parameter": [_scalars(p) for p in scenario.get("true_parameter_errors") or []],
        "topology": [_scalars(t, keep_lists=("affected_planning_buses",)) for t in scenario.get("true_topology_errors") or []],
        "hif": [_scalars(h) for h in hidden.get("true_hif_errors") or []],
        "unbalance": unbalance,
        "harmonic": [_scalars(h) for h in hidden.get("true_harmonic_errors") or []],
    }


def _root_record(scenario: Mapping[str, Any], *, family: str, parent_id: str, source_id: str) -> dict[str, Any]:
    metadata = scenario.get("metadata") or {}
    return {
        "root_id": str(scenario["scenario_id"]),
        "family": family,
        "parent_id": parent_id,
        "source_id": source_id,
        "case": str(scenario["case"]),
        "clean_case": (str(scenario["clean_case"]) if scenario.get("clean_case") else None),
        "measurements": [float(value) for value in scenario["measurements"]],
        "metadata": {key: json_safe(metadata[key]) for key in _METADATA_KEYS if key in metadata},
        "truth": _truth(scenario),
    }


def _split(parent_id: str, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{parent_id}".encode("utf-8")).hexdigest()
    point = int(digest[:8], 16) / 0xFFFFFFFF
    edge = 0.0
    for name, fraction in SPLIT_FRACTIONS:
        edge += fraction
        if point < edge:
            return name
    return SPLIT_FRACTIONS[-1][0]


def _children(root: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Truth-corrected child states, labelled with what remains."""
    family = root["family"]
    truth = root["truth"]
    meters = list(truth["measurement"])
    sigma = (root.get("metadata") or {}).get("sigma_z")
    kids: list[dict[str, Any]] = []

    def restored(indices: set[int]) -> list[float]:
        z = list(root["measurements"])
        for meter in meters:
            if meter["index"] in indices:
                z[meter["index"]] = float(meter["clean"])
        return z

    def child(kind: str, *, measurements=None, case=None, remaining: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "root_id": f"{root['root_id']}::{kind}",
            "child_of": root["root_id"],
            "child_kind": kind,
            "family": family,
            "parent_id": root["parent_id"],
            "source_id": root["source_id"],
            "case": str(case or root["case"]),
            "clean_case": root.get("clean_case"),
            "measurements": [float(v) for v in (measurements if measurements is not None else root["measurements"])],
            "metadata": copy.deepcopy(root["metadata"]),
            "truth": {key: list(remaining.get(key, [])) for key in truth},
        }

    physical_only = {key: value for key, value in truth.items() if key != "measurement"}
    if family in {"measurement+parameter", "measurement+topology", "measurement+hif"} and meters:
        kids.append(child("remove_meter_overlay", measurements=restored({m["index"] for m in meters}),
                          remaining=physical_only))
    if family == "measurement+parameter" and root.get("clean_case"):
        kids.append(child("fix_parameter", case=root["clean_case"], remaining={"measurement": meters}))
    if family == "multi_measurement" and len(meters) >= 2 and sigma:
        order = sorted(meters, key=lambda m: -abs(m["observed"] - m["clean"]) / float(sigma[m["index"]]))
        kids.append(child("remove_largest_meter", measurements=restored({order[0]["index"]}),
                          remaining={"measurement": order[1:]}))
        if len(meters) >= 3:
            kids.append(child("remove_all_but_smallest", measurements=restored({m["index"] for m in order[:-1]}),
                              remaining={"measurement": order[-1:]}))
    if family == "measurement" and meters:
        kids.append(child("remove_meter", measurements=restored({m["index"] for m in meters}), remaining={}))
    if family == "parameter" and root.get("clean_case"):
        kids.append(child("fix_parameter", case=root["clean_case"], remaining={}))
    return kids


# ------------------------------------------------------------------ generation


class RootBuilder:
    """Row-by-row use of the generator's family builders with known parents."""

    def __init__(self, generator: Round0ScenarioGenerator, seed: int) -> None:
        self.generator = generator
        self.order_rng = np.random.default_rng(seed + 101)
        self.skips: dict[str, dict[str, int]] = {}
        self.roots: list[dict[str, Any]] = []

    def _skip(self, family: str, rejection: Exception) -> None:
        reason = getattr(rejection, "reason", None) or getattr(rejection, "args", ["?"])[0]
        bucket = self.skips.setdefault(family, {})
        bucket[str(reason)] = bucket.get(str(reason), 0) + 1

    def _permute(self, rows: Sequence[Mapping[str, Any]]) -> list[int]:
        return [int(i) for i in self.order_rng.permutation(len(rows))]

    def corpus_rows(self, scenario: str) -> list[dict[str, Any]]:
        return list(self.generator._corpus().get(scenario, []))

    def build_no_error(self, count: int) -> None:
        rows = self.corpus_rows("no_error")
        built = 0
        for position in self._permute(rows):
            if built >= count:
                break
            row = rows[position]
            try:
                scenario = self.generator._no_error_scenario(row, position)
            except ScenarioRejected as rejection:
                self._skip("no_error", rejection)
                continue
            self.roots.append(_root_record(scenario, family="no_error", parent_id=f"corpus:{row['id']}", source_id=str(row["id"])))
            built += 1

    def build_measurement(self, family: str, count: int) -> None:
        subtype = "multi_gross_outliers"
        rows = [
            row for row in self.corpus_rows("measurement_error")
            if (self.generator._measurement_subtype(row) == subtype) == (family == "multi_measurement")
        ]
        built = 0
        for position in self._permute(rows):
            if built >= count:
                break
            row = rows[position]
            try:
                scenario = self.generator._measurement_scenario(row, position, family=family)
            except ScenarioRejected as rejection:
                self._skip(family, rejection)
                continue
            self.roots.append(_root_record(scenario, family=family, parent_id=f"corpus:{row['id']}", source_id=str(row["id"])))
            built += 1

    def build_parameter(self, count: int, composed: int) -> None:
        rows = self.corpus_rows("parameter_error")
        built = built_composed = 0
        for position in self._permute(rows):
            if built >= count and built_composed >= composed:
                break
            row = rows[position]
            try:
                base = self.generator._parameter_scenario(row, position)
            except ScenarioRejected as rejection:
                self._skip("parameter", rejection)
                continue
            parent = f"corpus:{row['id']}"
            if built < count:
                self.roots.append(_root_record(base, family="parameter", parent_id=parent, source_id=str(row["id"])))
                built += 1
            if built_composed < composed:
                try:
                    mixed = self.generator._compose_measurement(base, offsets=1, family="measurement+parameter", index=position)
                except ScenarioRejected as rejection:
                    self._skip("measurement+parameter", rejection)
                    continue
                self.roots.append(_root_record(mixed, family="measurement+parameter", parent_id=parent, source_id=str(row["id"])))
                built_composed += 1

    def build_hif(self, count: int, composed: int) -> None:
        rows = [row for row in self.generator._hif_rows() if (row.get("label") or {}).get("error_type") != "no_error"]
        built = built_composed = 0
        for position in self._permute(rows):
            if built >= count and built_composed >= composed:
                break
            row = rows[position]
            try:
                base = self.generator._hif_scenario(row, position)
            except ScenarioRejected as rejection:
                self._skip("hif", rejection)
                continue
            parent = f"hif_window:{row['id']}"
            if built < count:
                self.roots.append(_root_record(base, family="hif", parent_id=parent, source_id=str(row["id"])))
                built += 1
            if built_composed < composed:
                try:
                    mixed = self.generator._compose_measurement(base, offsets=1, family="measurement+hif", index=position)
                except ScenarioRejected as rejection:
                    self._skip("measurement+hif", rejection)
                    continue
                self.roots.append(_root_record(mixed, family="measurement+hif", parent_id=parent, source_id=str(row["id"])))
                built_composed += 1

    def build_unbalance(self, count: int) -> None:
        rows = [row for row in self.generator._imbalance_rows() if (row.get("label") or {}).get("error_type") != "no_error"]
        built = 0
        for position in self._permute(rows):
            if built >= count:
                break
            row = rows[position]
            try:
                scenario = self.generator._unbalance_scenario(row, position)
            except ScenarioRejected as rejection:
                self._skip("three_phase_unbalance", rejection)
                continue
            self.roots.append(_root_record(scenario, family="three_phase_unbalance", parent_id=f"unbalance_window:{row['id']}", source_id=str(row["id"])))
            built += 1

    def build_topology(self, family: str, count: int) -> None:
        built = attempts = 0
        offset = 0 if family == "topology" else 100000
        while built < count and attempts < count * 8:
            attempts += 1
            index = offset + attempts
            try:
                if family == "topology":
                    scenario = self.generator._topology_scenario(index)
                else:
                    base = self.generator._topology_scenario(index, effects=("dangling_line_terminal",))
                    scenario = self.generator._compose_measurement(base, offsets=1, family=family, index=index)
            except ScenarioRejected as rejection:
                self._skip(family, rejection)
                continue
            parent = f"synthesized_topology:{scenario['scenario_id']}"
            self.roots.append(_root_record(scenario, family=family, parent_id=parent, source_id=f"synthesized_{index}"))
            built += 1

    def build_harmonic(self, count: int) -> None:
        built = attempts = 0
        while built < count and attempts < count * 8:
            attempts += 1
            try:
                scenario = self.generator._harmonic_scenario(self.generator._synthesized_harmonic_row(attempts), attempts)
            except ScenarioRejected as rejection:
                self._skip("harmonic", rejection)
                continue
            parent = f"synthesized_harmonic:{scenario['scenario_id']}"
            self.roots.append(_root_record(scenario, family="harmonic", parent_id=parent, source_id=f"synthesized_{attempts}"))
            built += 1


def healthy_windows(generator: Round0ScenarioGenerator) -> list[dict[str, Any]]:
    """Every clean corpus window (tabular corpus and the HIF corpora's controls)."""
    records: list[dict[str, Any]] = []
    for row in generator._corpus().get("no_error", []):
        records.append({
            "root_id": f"healthy:{row['id']}", "family": "healthy_window", "parent_id": f"corpus:{row['id']}",
            "source_id": str(row["id"]), "source": "tabular_corpus", "case": generator.case_path,
            "measurements": [float(v) for v in row["z_obs"]],
            "metadata": {"sigma_z": list(row["sigma_z"])} if row.get("sigma_z") else {},
            "truth": {"measurement": [], "parameter": [], "topology": [], "hif": [], "unbalance": [], "harmonic": []},
        })
    for row in generator._hif_rows():
        if (row.get("label") or {}).get("error_type") != "no_error" or not row.get("z_obs"):
            continue
        records.append({
            "root_id": f"healthy:{row['id']}", "family": "healthy_window", "parent_id": f"hif_window:{row['id']}",
            "source_id": str(row["id"]), "source": "hif_corpus_control", "case": generator.case_path,
            "measurements": [float(v) for v in row["z_obs"]],
            "metadata": {"sigma_z": list(row["sigma_z"])} if row.get("sigma_z") else {},
            "truth": {"measurement": [], "parameter": [], "topology": [], "hif": [], "unbalance": [], "harmonic": []},
        })
    return records


def mimic_roots(generator: Round0ScenarioGenerator, count_per_variant: int, seed: int) -> list[dict[str, Any]]:
    """Two biased flow meters at the two ends of one candidate line, from clean windows."""
    from mcp_server.matpower_server import _load_python_case

    case = _load_python_case(generator.case_path)
    nb, nl = int(np.asarray(case["bus"]).shape[0]), int(np.asarray(case["branch"]).shape[0])
    lines = default_hif_lines(case)
    rng = np.random.default_rng(seed + 202)
    rows = generator._corpus().get("no_error", [])
    order = rng.permutation(len(rows))
    records: list[dict[str, Any]] = []
    variants = (("mimic_flow_pair_same_sign", (1.0, 1.0)), ("mimic_flow_pair_opposite_sign", (1.0, -1.0)))
    for variant_index, (family, signs) in enumerate(variants):
        built = 0
        for position in order[variant_index::2]:
            if built >= count_per_variant:
                break
            row = rows[int(position)]
            sigma = list(row.get("sigma_z") or generator.noise_profile().tolist())
            z = [float(v) for v in row["z_obs"]]
            line = int(lines[built % len(lines)])
            i_from = flow_channel_index("Pf", line, nb, nl)
            i_to = flow_channel_index("Pt", line, nb, nl)
            errors = []
            for index, sign in ((i_from, signs[0]), (i_to, signs[1])):
                magnitude = float(rng.uniform(10.0, 15.0)) * float(sigma[index])
                clean = z[index]
                z[index] = clean + sign * magnitude
                errors.append({"index": index, "observed": z[index], "clean": clean, "channel": None})
            records.append({
                "root_id": f"{family}:{row['id']}:{line}", "family": family, "parent_id": f"corpus:{row['id']}",
                "source_id": str(row["id"]), "case": generator.case_path, "measurements": z,
                "metadata": {"sigma_z": sigma},
                "truth": {"measurement": errors, "parameter": [], "topology": [], "hif": [], "unbalance": [], "harmonic": [],
                          "mimic": {"branch_row0": line, "signs": list(signs)}},
            })
            built += 1
    return records


# ---------------------------------------------------------------- analysis run


def _analyze(record: Mapping[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    out = dict(record)
    try:
        out["analysis"] = analyze_state(record)
    except Exception as exc:  # keep the row; the report counts failures
        out["analysis"] = {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
    out["analysis_seconds"] = time.perf_counter() - started
    return out


def _run(records: Sequence[Mapping[str, Any]], workers: int, label: str) -> list[dict[str, Any]]:
    if not records:
        return []
    started = time.perf_counter()
    if workers <= 1:
        results = [_analyze(record) for record in records]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_analyze, records, chunksize=4))
    print(f"[analysis] {label}: {len(records)} states in {time.perf_counter() - started:.1f} s", flush=True)
    return results


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(json_safe(row), sort_keys=True) + "\n")
            count += 1
    return count


def build_generator(seed: int) -> Round0ScenarioGenerator:
    return Round0ScenarioGenerator(
        seed=seed,
        hif_sample_paths=list(PHYSICAL_HIF_SAMPLE_PATHS),
        imbalance_sample_path=PHYSICAL_IMBALANCE_SAMPLE_PATH,
        normalized_residual_threshold=4.0,
        evidence_profile=SUSPICION_GATED_PROFILE,
        hif_max_scans=3,
        min_measurement_error_sigma=10.0,
        topology_effects=("dangling_line_terminal", "bus_split"),
        parameter_ranking_dominance_threshold=1.0,
        parameter_target_rank_allowance=2,
    )


def reanalyze(source: Path, out: Path, workers: int) -> int:
    """Re-run the analysis on a saved dataset (same roots, children, healthy pool and mimics).

    The screen's decision rules change between study steps; this keeps the
    generated states and parent split fixed so the two screens are compared
    on identical inputs.  The healthy pool is re-run from the saved alarmed
    windows only (a WLS alarm does not depend on the screen).
    """
    out.mkdir(parents=True, exist_ok=True)

    def load(name: str) -> list[dict[str, Any]]:
        path = source / name
        if not path.is_file():
            return []
        rows = []
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                row.pop("analysis", None)
                row.pop("analysis_seconds", None)
                rows.append(row)
        return rows

    started = time.perf_counter()
    counts = {}
    for name, label in (("roots.jsonl", "roots"), ("children.jsonl", "children"),
                        ("healthy_alarms.jsonl", "healthy alarms"), ("mimic.jsonl", "mimic")):
        rows = load(name)
        analyzed = _run(rows, workers, label)
        counts[label] = _write_jsonl(out / name, analyzed)
    manifest = {}
    if (source / "manifest.json").is_file():
        manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    manifest.update({
        "reanalyzed_from": str(source), "reanalysis_git_commit": _git_commit(),
        "reanalysis_created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "reanalysis_counts": counts, "reanalysis_seconds": time.perf_counter() - started,
    })
    manifest.setdefault("counts", {})["healthy_alarms"] = counts.get("healthy alarms", 0)
    (out / "manifest.json").write_text(json.dumps(json_safe(manifest), indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(counts, indent=2), flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--workers", type=int, default=max(1, min(16, (os.cpu_count() or 2) - 2)))
    parser.add_argument("--plan", type=json.loads, default=None, help="JSON family->count; defaults to DEFAULT_PLAN")
    parser.add_argument("--mimic-per-variant", type=int, default=100)
    parser.add_argument("--no-healthy", action="store_true", help="skip the healthy-window alarm pool")
    parser.add_argument("--reanalyze", default=None,
                        help="re-run WLS and the screen on the states saved under this dataset directory instead of generating")
    args = parser.parse_args(argv)
    if args.reanalyze:
        return reanalyze(Path(args.reanalyze), Path(args.output_dir), args.workers)

    plan = dict(DEFAULT_PLAN)
    if args.plan:
        plan.update({str(k): int(v) for k, v in args.plan.items()})
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    generator = build_generator(args.seed)
    builder = RootBuilder(generator, args.seed)
    timings: dict[str, float] = {}

    def timed(name: str, fn, *fn_args) -> None:
        t0 = time.perf_counter()
        fn(*fn_args)
        timings[name] = time.perf_counter() - t0
        print(f"[generate] {name}: {timings[name]:.1f} s, roots so far {len(builder.roots)}", flush=True)

    timed("no_error", builder.build_no_error, plan.get("no_error", 0))
    timed("measurement", builder.build_measurement, "measurement", plan.get("measurement", 0))
    timed("multi_measurement", builder.build_measurement, "multi_measurement", plan.get("multi_measurement", 0))
    timed("parameter", builder.build_parameter, plan.get("parameter", 0), plan.get("measurement+parameter", 0))
    timed("hif", builder.build_hif, plan.get("hif", 0), plan.get("measurement+hif", 0))
    timed("three_phase_unbalance", builder.build_unbalance, plan.get("three_phase_unbalance", 0))
    timed("harmonic", builder.build_harmonic, plan.get("harmonic", 0))
    timed("topology", builder.build_topology, "topology", plan.get("topology", 0))
    timed("measurement+topology", builder.build_topology, "measurement+topology", plan.get("measurement+topology", 0))

    roots = builder.roots
    for root in roots:
        root["split"] = _split(root["parent_id"], args.seed)
    children = [child for root in roots for child in _children(root)]
    for child in children:
        child["split"] = _split(child["parent_id"], args.seed)
    healthy = [] if args.no_healthy else healthy_windows(generator)
    mimic = mimic_roots(generator, args.mimic_per_variant, args.seed)
    for record in (*healthy, *mimic):
        record["split"] = _split(record["parent_id"], args.seed)
    print(f"[generate] roots {len(roots)}, children {len(children)}, healthy windows {len(healthy)}, mimic {len(mimic)}", flush=True)

    analyzed_roots = _run(roots, args.workers, "roots")
    analyzed_children = _run(children, args.workers, "children")
    analyzed_healthy = _run(healthy, args.workers, "healthy windows")
    analyzed_mimic = _run(mimic, args.workers, "mimic")

    counts = {
        "roots": _write_jsonl(out / "roots.jsonl", analyzed_roots),
        "children": _write_jsonl(out / "children.jsonl", analyzed_children),
        "healthy_windows_total": len(analyzed_healthy),
        "healthy_alarms": _write_jsonl(
            out / "healthy_alarms.jsonl",
            [row for row in analyzed_healthy if (row.get("analysis") or {}).get("wls", {}).get("alarm")],
        ),
        "mimic": _write_jsonl(out / "mimic.jsonl", analyzed_mimic),
    }
    by_family: dict[str, int] = {}
    by_split: dict[str, int] = {}
    for root in roots:
        by_family[root["family"]] = by_family.get(root["family"], 0) + 1
        by_split[root["split"]] = by_split.get(root["split"], 0) + 1
    manifest = {
        "schema": "hypothesis_ranking_step1_v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": _git_commit(),
        "seed": args.seed,
        "plan": plan,
        "generator": {
            "evidence_profile": SUSPICION_GATED_PROFILE, "normalized_residual_threshold": 4.0, "chi2_alpha": 0.01,
            "anomaly_margin": generator.anomaly_margin, "min_measurement_error_sigma": 10.0,
            "parameter_ranking_dominance_threshold": 1.0, "parameter_target_rank_allowance": 2,
            "topology_effects": ["dangling_line_terminal", "bus_split"],
            "hif_sample_paths": [str(p) for p in PHYSICAL_HIF_SAMPLE_PATHS],
            "imbalance_sample_path": str(PHYSICAL_IMBALANCE_SAMPLE_PATH),
        },
        "split_fractions": dict(SPLIT_FRACTIONS),
        "split_unit": "physical parent (corpus row or waveform window; synthesized roots are their own parent)",
        "counts": counts, "roots_by_family": by_family, "roots_by_split": by_split,
        "skips": builder.skips, "generation_seconds": timings,
        "total_seconds": time.perf_counter() - started,
        "workers": args.workers,
        "files": {
            "roots": "roots.jsonl", "children": "children.jsonl", "healthy_alarms": "healthy_alarms.jsonl",
            "mimic": "mimic.jsonl",
        },
    }
    (out / "manifest.json").write_text(json.dumps(json_safe(manifest), indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"counts": counts, "roots_by_family": by_family, "skips": builder.skips}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
