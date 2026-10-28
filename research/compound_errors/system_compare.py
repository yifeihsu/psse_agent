"""Compare two systems' compound-error suites class by class at matched effect ratios (IEEE 14 against IEEE 118, 2026-10-28).

    python -m research.compound_errors.system_compare \\
        --suite case14=<ieee14 study>/suite.json --suite case118=output/compound_errors_ieee118_20261028/suite.json \\
        --rollout case14:deployed=<ieee14 study>/stage2_item2/expert_baseline_compare/episodes.jsonl \\
        --rollout case14:pair_refit=<ieee14 study>/stage2_item3/expert_baseline_compare/episodes.jsonl \\
        --rollout case118:deployed=output/compound_errors_ieee118_20261028/expert_baseline_compare/episodes.jsonl \\
        --rollout case118:pair_refit=output/compound_errors_ieee118_20261028/expert_baseline_compare_pair_refit/episodes.jsonl \\
        --output-dir output/compound_errors_ieee118_20261028/compare --workers 26

Only the classes that need no breaker are read.  Per root (one worker each),
with the deployed solve on the root's own case, sigma and exact rows:

* every atom's effect ratio on its own, the noiseless chi-square over the
  alarm threshold's excess (a line from the suite's solo-effect record; a meter
  as the symmetric difference ``(J(c + d) + J(c - d)) / 2 - J(c)`` of its offset
  ``d`` on the true model's clean snapshot ``c``, which cancels the noise cross
  term; the bus-branch suite's own healthy-network record is kept beside it);
* the branch-parameter multipliers on the reported model: how many lines lie
  within the comparison band (1.5) of the top one, where the true lines rank,
  the pair's footprint overlap (``psse_env.providers.compound_pairs``) and the
  larger endpoint degree of each true line;
* the truth probe of a silent second atom: each atom corrected exactly (a
  line's true impedance in the model, a meter's clean reading), and whether
  the solve still alarms on what remains;
* the open-terminal cue ``flow_at_noise_lines`` of the deployed ``run_wls`` on
  the initial state (every one is a false fire: no root here carries a
  breaker atom).

The random-pair geometry of each system (hops between every pair of the
lines the sampler perturbs, and their footprint overlap at the unit-load OPF
under the registry's noise model) and the rollouts (solved, actions, the
first committed true correction's verdict) complete the tables.  A root is
*matched* when every atom's ratio is at least ``--matched-floor`` (1.5, the
IEEE 118 suite's admission floor).
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CONTRACT = "compound_system_compare_v1"
CLASSES = ("rx2_adjacent", "rx2_distant", "meter_rx2", "meter_on_rx", "control_rx", "control_meter")
CLASS_LABELS = {
    "rx2_adjacent": "two R/X lines, adjacent",
    "rx2_distant": "two R/X lines, distant",
    "meter_rx2": "meter + two R/X",
    "meter_on_rx": "meter on the faulted line + R/X",
    "control_rx": "control: one R/X line",
    "control_meter": "control: 1 to 3 meters",
}
ALPHA = 0.01
RESIDUAL_THRESHOLD = 4.0
RATIO_BINS = ((0.0, 1.5), (1.5, 5.0), (5.0, 20.0), (20.0, math.inf))
DEGREE_BINS = ((1, 3), (4, 5), (6, 8), (9, 99))


# --------------------------------------------------------------------------- the deployed solve


@lru_cache(maxsize=4)
def _dof_of_threshold(threshold: float) -> int:
    from scipy.stats import chi2

    for dof in range(1, 5000):
        if abs(float(chi2.ppf(1.0 - ALPHA, dof)) - threshold) < 1e-6 * max(1.0, threshold):
            return dof
    raise ValueError(f"no dof matches the threshold {threshold}")


def threshold_excess(dof: int) -> float:
    from research.classifier_triage.effect_sampler import threshold_excess as excess

    return float(excess(dof, ALPHA))


def _ppc(case: Any) -> dict[str, Any]:
    from mcp_server.matpower_server import _load_python_case

    return _load_python_case(str(case)) if isinstance(case, str) else copy.deepcopy(dict(case))


def solve(ppc: Mapping[str, Any], z: Sequence[float], sigma: Sequence[float], exact: Sequence[int]) -> dict[str, Any] | None:
    """The deployed WLS (``tools.lagrangian_port``) with the alarm rule at margin one."""
    from scipy.stats import chi2
    from tools.lagrangian_port import lagrangian_m_singlephase_details

    options: dict[str, Any] = {"measurement_sigma": np.asarray(sigma, dtype=float)}
    if exact:
        options["exact_measurement_indices"] = [int(i) for i in exact]
    try:
        details = lagrangian_m_singlephase_details(z=np.asarray(z, dtype=float), result=copy.deepcopy(dict(ppc)), ind=0,
                                                   bus_data=np.asarray(ppc["bus"], dtype=float), **options)
    except (ValueError, np.linalg.LinAlgError, RuntimeError):
        return None
    if not details.get("success"):
        return None
    objective, dof = float(details["wls_objective"]), int(details["dof"])
    maximum = float(np.max(details["r_norm"])) if np.asarray(details["r_norm"]).size else 0.0
    threshold = float(chi2.ppf(1.0 - ALPHA, max(1, dof)))
    return {"objective": objective, "dof": dof, "threshold": threshold, "max_normalized_residual": maximum,
            "alarmed": bool(objective >= threshold or maximum >= RESIDUAL_THRESHOLD)}


def with_line(ppc: Mapping[str, Any], row0: int, r: float, x: float) -> dict[str, Any]:
    out = copy.deepcopy(dict(ppc))
    out["branch"] = np.array(out["branch"], dtype=float, copy=True)
    out["branch"][int(row0), 2] = float(r)
    out["branch"][int(row0), 3] = float(x)
    return out


# --------------------------------------------------------------------------- one root


def _atoms(root: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [dict(a) for a in ((root.get("audit") or {}).get("compound") or {}).get("atoms") or []]


def probe_root(root: Mapping[str, Any]) -> dict[str, Any]:
    from psse_env.providers import compound_pairs
    from psse_env.providers.matpower import MatpowerDeploymentProviders

    started = time.perf_counter()
    execution = root["execution"]
    audit = root.get("audit") or {}
    compound = audit.get("compound") or {}
    truth = audit.get("truth") or {}
    metadata = execution.get("metadata") or {}
    system = str((root.get("grouping") or {}).get("case_id") or "case14")
    sigma = [float(v) for v in metadata.get("sigma_z") or []]
    exact = [int(i) for i in metadata.get("structural_zero_indices") or []]
    z_obs = [float(v) for v in execution["measurements"]]
    clean = [float(v) for v in truth.get("clean_measurements") or []]
    reported = _ppc(execution["case"])
    true_case = _ppc(truth["clean_case"]) if truth.get("clean_case") else None
    atoms = _atoms(root)
    lines = [a for a in atoms if a["kind"] == "parameter"]
    meters = [a for a in atoms if a["kind"] == "meter"]
    solo = {int(e["atom"]): e for e in compound.get("solo_effects") or []}
    out: dict[str, Any] = {"scenario_id": execution["scenario_id"], "system": system, "class": compound.get("class"),
                           "physics": compound.get("physics", "node_breaker"),
                           "lines": [int(a["line_index1"]) for a in lines], "meters": [int(a["index"]) for a in meters]}

    # Every atom's effect ratio on its own.
    ratios: list[dict[str, Any]] = []
    for position, atom in enumerate(lines):
        record = solo.get(position) or {}
        if record.get("effect_ratio") is not None:
            ratio = float(record["effect_ratio"])
        elif record.get("chi_square") is not None and record.get("chi_square_threshold"):
            dof = _dof_of_threshold(float(record["chi_square_threshold"]))
            ratio = float(record["chi_square"]) / threshold_excess(dof)
        else:
            ratio = None
        ratios.append({"kind": "parameter", "line": int(atom["line_index1"]), "ratio": ratio, "basis": "solo_record"})
    base = solve(true_case, clean, sigma, exact) if (meters and true_case is not None and clean) else None
    for position, atom in enumerate(meters, start=len(lines)):
        estimate = None
        if base is not None:
            index, offset = int(atom["index"]), float(atom["offset"])
            plus, minus = list(clean), list(clean)
            plus[index] += offset
            minus[index] -= offset
            up, down = solve(true_case, plus, sigma, exact), solve(true_case, minus, sigma, exact)
            if up is not None and down is not None:
                estimate = (0.5 * (up["objective"] + down["objective"]) - base["objective"]) / threshold_excess(base["dof"])
        record = solo.get(position) or {}
        ratios.append({"kind": "meter", "index": int(atom["index"]), "ratio": estimate, "basis": "symmetric_difference",
                       "healthy_network_ratio": record.get("effect_ratio"), "sigma_multiple": atom.get("sigma_multiple")})
    out["atom_ratios"] = ratios
    known = [r["ratio"] for r in ratios if r["ratio"] is not None]
    out["min_atom_ratio"] = min(known) if len(known) == len(ratios) and known else None

    # The multipliers on the reported model.
    branch = np.asarray(reported["branch"], dtype=float)
    degree = compound_pairs.bus_degree(branch)
    out["line_degrees"] = {str(a["line_index1"]): compound_pairs.line_degree(branch, a["branch_row0"], degree) for a in lines}
    if lines:
        view = compound_pairs.multiplier_view(reported, z_obs, sigma=sigma, exact_measurement_indices=exact)
        if view is not None:
            band = compound_pairs.band_view(compound_pairs.line_scores(view["lambdaN"]), rows_of_interest=[a["branch_row0"] for a in lines])
            out["band"] = {key: band[key] for key in ("band_count", "band_lines", "top_lines", "line_ranks", "top_score")}
            if len(lines) == 2:
                out["overlap"] = compound_pairs.pair_overlap(view["ea"], lines[0]["branch_row0"], lines[1]["branch_row0"])
        if len(lines) == 2:
            out["hops"] = compound_pairs.line_hops(branch, lines[0]["branch_row0"], lines[1]["branch_row0"])

    # A silent second atom: each atom corrected exactly, does what remains alarm?
    if len(atoms) >= 2:
        silent: dict[str, Any] = {}
        for atom in lines:
            fixed = with_line(reported, atom["branch_row0"], atom["clean_r"], atom["clean_x"])
            result = solve(fixed, z_obs, sigma, exact)
            silent[f"line:{atom['line_index1']}"] = None if result is None else not result["alarmed"]
        for atom in meters:
            if not clean:
                continue
            restored = list(z_obs)
            restored[int(atom["index"])] = clean[int(atom["index"])]
            result = solve(reported, restored, sigma, exact)
            silent[f"meter:{atom['index']}"] = None if result is None else not result["alarmed"]
        out["silent_after_exact_fix"] = silent

    # The open-terminal cue at the first solve of the deployed run_wls.
    providers = MatpowerDeploymentProviders(evidence_profile="scada_only", normalized_residual_threshold=RESIDUAL_THRESHOLD)
    state = {"state_id": "compare:s0", "status": "active", "case": copy.deepcopy(execution["case"]),
             "measurements": list(z_obs), "metadata": copy.deepcopy(metadata)}
    try:
        metrics = providers.run_wls(state)
        fired = [int(v) for v in metrics.get("flow_at_noise_lines") or []]
        out["cue"] = {"fired": fired, "true_lines_fired": [v for v in fired if v in out["lines"]],
                      "globally_resolved": metrics.get("globally_resolved"), "chi_square_ratio": metrics.get("chi_square_ratio")}
    except Exception as exc:  # the record names the failure
        out["cue"] = {"error": f"{type(exc).__name__}: {exc}"}
    out["seconds"] = round(time.perf_counter() - started, 1)
    return out


def _probe_safe(payload: tuple[str, Mapping[str, Any]]) -> dict[str, Any]:
    """One root of the suite labelled ``label`` (two suites of one system keep apart)."""
    label, root = payload
    try:
        record = probe_root(root)
    except Exception as exc:
        record = {"scenario_id": root["execution"]["scenario_id"], "error": f"{type(exc).__name__}: {exc}",
                  "class": ((root.get("audit") or {}).get("compound") or {}).get("class")}
    record["suite"] = label
    return record


# --------------------------------------------------------------------------- random-pair geometry


def pair_geometry(system: str) -> dict[str, Any]:
    """Hops and footprint overlap of every pair of the lines the sampler perturbs, at the unit-load OPF."""
    from psse_env.providers import compound_pairs
    from psse_env.providers.scenario_generator import Round0ScenarioGenerator, build_measurement_vector
    from psse_env.systems import resolve_system

    spec = resolve_system(system)
    ppc = _ppc(spec.case_path)
    branch = np.asarray(ppc["branch"], dtype=float)
    lines = [k for k in range(len(branch)) if branch[k, 8] == 0.0 and abs(branch[k, 2]) > 1e-9 and abs(branch[k, 3]) > 1e-9
             and branch[k, 10] > 0.0]
    table = compound_pairs.bus_distances(branch)
    degree = compound_pairs.bus_degree(branch)
    scaled = Round0ScenarioGenerator._scaled_case14(1.0) if spec.case_id == "case14" else spec.load_case()
    solved = Round0ScenarioGenerator._solve_ac_opf(scaled)
    view = None
    flows_at_noise = None
    if solved is not None:
        z = build_measurement_vector(solved)
        sigma = spec.measurement_sigma()
        view = compound_pairs.multiplier_view(ppc, z, sigma=sigma)
        nb, nl = spec.nb, spec.nl
        pf, pt = z[3 * nb:3 * nb + nl], z[3 * nb + 2 * nl:3 * nb + 3 * nl]
        flows_at_noise = int(sum(1 for k in lines if max(abs(pf[k]), abs(pt[k])) < 3.0 * 0.01))
    hops: Counter = Counter()
    overlaps: dict[str, list[float]] = defaultdict(list)
    for i, a in enumerate(lines):
        for b in lines[i + 1:]:
            h = compound_pairs.line_hops(branch, a, b, table)
            hops[h] += 1
            if view is not None:
                value = compound_pairs.pair_overlap(view["ea"], a, b)["overlap"]
                if value is not None:
                    overlaps["0" if h == 0 else "1" if h == 1 else "2" if h == 2 else "3+"].append(float(value))
    total = sum(hops.values())
    degrees = [degree[b] for b in degree]
    return {
        "system": spec.case_id, "branches": int(len(branch)), "lines": len(lines), "pairs": total,
        "hops": {str(k): v for k, v in sorted(hops.items())},
        "share_bus": hops[0] / total, "within_one_hop": (hops[0] + hops[1]) / total,
        "at_least_two_hops": sum(v for k, v in hops.items() if k >= 2) / total,
        "at_least_three_hops": sum(v for k, v in hops.items() if k >= 3) / total,
        "max_hops": max(hops), "median_hops": float(np.median([k for k, v in hops.items() for _ in range(v)])),
        "bus_degree": {"mean": float(np.mean(degrees)), "max": int(max(degrees)), "buses_degree_6_plus": int(sum(1 for d in degrees if d >= 6))},
        "lines_at_noise_unit_load": flows_at_noise,
        "overlap_by_hops": {key: {"n": len(values), "median": float(np.median(values)), "p90": float(np.percentile(values, 90)),
                                  "share_above_0.5": float(np.mean(np.asarray(values) > 0.5))}
                            for key, values in sorted(overlaps.items())},
    }


# --------------------------------------------------------------------------- the rollouts


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def first_true_fix(record: Mapping[str, Any], atoms: int) -> dict[str, Any] | None:
    """The first committed correction of a true target: its verdict and whether the solve was then clean with atoms left."""
    corrected: set[str] = set()
    for correction in record.get("corrections") or []:
        if correction.get("lifecycle") != "commit_state" or correction.get("verdict") not in {"ACCEPT_FINAL", "ACCEPT_PARTIAL"}:
            continue
        if not correction.get("target_is_true"):
            return {"first_commit_true": False}
        arguments = correction.get("arguments") or {}
        if correction.get("tool") == "correct_measurements":
            group = arguments.get("suspect_group") or ([arguments.get("index")] if arguments.get("index") is not None else [])
            corrected.update(f"meter:{int(i)}" for i in group)
        else:
            corrected.add(f"line:{arguments.get('line_index')}")
        resolved = bool((correction.get("verification") or {}).get("globally_resolved"))
        return {"first_commit_true": True, "verdict": correction.get("verdict"), "resolved": resolved,
                "atoms_left": max(0, atoms - len(corrected)), "silent_with_atoms_left": resolved and atoms - len(corrected) > 0}
    return None


def ratio_bin(value: float | None) -> str:
    if value is None:
        return "unknown"
    for low, high in RATIO_BINS:
        if low <= value < high:
            return f"{low:g}-{high:g}" if math.isfinite(high) else f"{low:g}+"
    return "unknown"


def degree_bin(value: int | None) -> str:
    if value is None:
        return "unknown"
    for low, high in DEGREE_BINS:
        if low <= value <= high:
            return f"{low}-{high}" if high < 99 else f"{low}+"
    return "unknown"


# --------------------------------------------------------------------------- summaries


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def summarize(probes: Sequence[Mapping[str, Any]], suites: Mapping[str, Sequence[Mapping[str, Any]]],
              rollouts: Mapping[str, Mapping[str, Mapping[str, Mapping[str, Any]]]], *, matched_floor: float) -> dict[str, Any]:
    by_suite: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for p in probes:
        if not p.get("error"):
            by_suite[str(p.get("suite"))][str(p["scenario_id"])] = p
    out: dict[str, Any] = {"matched_floor": matched_floor, "systems": {}}
    for system, roots in suites.items():
        entry: dict[str, Any] = {"classes": {}}
        for name in CLASSES:
            rows = [r for r in roots if ((r.get("audit") or {}).get("compound") or {}).get("class") == name]
            ids = [str(r["execution"]["scenario_id"]) for r in rows]
            info = [by_suite[system].get(i) for i in ids]
            matched = [i for i, p in zip(ids, info) if p and p.get("min_atom_ratio") is not None and p["min_atom_ratio"] >= matched_floor]
            cell: dict[str, Any] = {"n": len(rows), "matched": len(matched), "probe_errors": sum(1 for p in info if p is None)}
            ratios = [p["min_atom_ratio"] for p in info if p and p.get("min_atom_ratio") is not None]
            cell["min_atom_ratio_median"] = float(np.median(ratios)) if ratios else None
            cell["ratio_bins"] = dict(Counter(ratio_bin(p.get("min_atom_ratio")) for p in info if p))
            pairs = [p for p in info if p and p.get("hops") is not None]
            if pairs:
                cell["share_bus"] = _rate(sum(1 for p in pairs if p["hops"] == 0), len(pairs))
                cell["hops_median"] = float(np.median([p["hops"] for p in pairs]))
                overlaps = [p["overlap"]["overlap"] for p in pairs if (p.get("overlap") or {}).get("overlap") is not None]
                cell["overlap_median"] = float(np.median(overlaps)) if overlaps else None
                cell["overlap_p90"] = float(np.percentile(overlaps, 90)) if overlaps else None
            multi = [p for p in info if p and p.get("silent_after_exact_fix")]
            if multi:
                # some single exact fix leaves what remains silent
                cell["silent_after_one_exact_fix"] = sum(1 for p in multi if any(v for v in p["silent_after_exact_fix"].values()))
                cell["silent_after_one_exact_fix_matched"] = sum(
                    1 for p in multi if p["scenario_id"] in set(matched) and any(v for v in p["silent_after_exact_fix"].values()))
                cell["multi_atom_roots"] = len(multi)
                cell["multi_atom_roots_matched"] = sum(1 for p in multi if p["scenario_id"] in set(matched))
            banded = [p for p in info if p and p.get("band")]
            if banded:
                cell["band_count_mean"] = float(np.mean([p["band"]["band_count"] for p in banded]))
                cell["band_two_or_more"] = sum(1 for p in banded if p["band"]["band_count"] >= 2)
                cell["top_line_true"] = sum(1 for p in banded if p["band"]["top_lines"] and p["band"]["top_lines"][0] in p["lines"])
                cell["banded_roots"] = len(banded)
            cue = [p for p in info if p and isinstance(p.get("cue"), Mapping) and "fired" in p["cue"]]
            cell["cue_roots"] = len(cue)
            cell["cue_fired"] = sum(1 for p in cue if p["cue"]["fired"])
            cell["cue_fired_on_true_line"] = sum(1 for p in cue if p["cue"]["true_lines_fired"])
            comparison_wanted = 0
            ranked = 0
            for root in rows:
                ranking = ((root.get("audit") or {}).get("compound") or {}).get("parameter_ranking")
                if isinstance(ranking, Mapping):
                    ranked += 1
                    ratio = ranking.get("parameter_ranking_dominance_ratio")
                    dominant = ranking.get("parameter_ranking_dominant") is True
                    comparison_wanted += int((not dominant) or (ratio is not None and float(ratio) < 1.5))
            cell["deployed_ranked"] = ranked
            cell["deployed_comparison_wanted"] = comparison_wanted
            for arm, episodes in (rollouts.get(system) or {}).items():
                records = [episodes.get(i) for i in ids]
                present = [r for r in records if r is not None]
                solved = {str(r["scenario_id"]) for r in present if r.get("success")}
                arm_cell: dict[str, Any] = {
                    "n": len(present), "solved": len(solved),
                    "mean_actions": float(np.mean([r.get("steps") or 0 for r in present])) if present else None,
                    "mean_wls_calls": float(np.mean([r.get("wls_calls") or 0 for r in present])) if present else None,
                    "comparisons": sum(int(r.get("comparisons") or 0) for r in present),
                    "matched_solved": sum(1 for i in matched if i in solved),
                    "by_ratio_bin": {},
                    "errors": sum(1 for r in present if r.get("error")),
                }
                bins: dict[str, list[int]] = defaultdict(lambda: [0, 0])
                for i, p in zip(ids, info):
                    if p is None or i not in episodes:
                        continue
                    key = ratio_bin(p.get("min_atom_ratio"))
                    bins[key][0] += 1
                    bins[key][1] += int(i in solved)
                arm_cell["by_ratio_bin"] = {k: {"n": v[0], "solved": v[1]} for k, v in sorted(bins.items())}
                first = []
                for i, r, p in zip(ids, records, info):
                    if r is None or p is None:
                        continue
                    atoms = len(p.get("lines") or []) + len(p.get("meters") or [])
                    if atoms >= 2:
                        view = first_true_fix(r, atoms)
                        if view is not None:
                            first.append({**view, "matched": i in set(matched)})
                if first:
                    arm_cell["first_commit"] = {
                        "roots": len(first),
                        "true": sum(1 for f in first if f.get("first_commit_true")),
                        "final_with_atoms_left": sum(1 for f in first if f.get("silent_with_atoms_left")),
                        "final_with_atoms_left_matched": sum(1 for f in first if f.get("silent_with_atoms_left") and f["matched"]),
                        "roots_matched": sum(1 for f in first if f["matched"]),
                    }
                reasons = Counter(reason for r in present if not r.get("success") for reason in r.get("audit_reasons") or [])
                arm_cell["failure_reasons"] = dict(reasons.most_common(6))
                cell[arm] = arm_cell
            entry["classes"][name] = cell
        # ranking ambiguity against the degree of the true lines' buses
        degrees: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for root in roots:
            p = by_suite[system].get(str(root["execution"]["scenario_id"]))
            if p and p.get("band") and p.get("line_degrees"):
                degrees[degree_bin(max(int(v) for v in p["line_degrees"].values()))].append(p)
        entry["band_by_degree"] = {
            key: {"roots": len(rows), "band_count_mean": float(np.mean([p["band"]["band_count"] for p in rows])),
                  "band_two_or_more": sum(1 for p in rows if p["band"]["band_count"] >= 2),
                  "top_line_true": sum(1 for p in rows if p["band"]["top_lines"] and p["band"]["top_lines"][0] in p["lines"])}
            for key, rows in sorted(degrees.items())
        }
        meters = [r for p in by_suite[system].values() for r in p.get("atom_ratios") or []
                  if r["kind"] == "meter" and r.get("ratio") is not None and r.get("healthy_network_ratio") is not None]
        if meters:
            gaps = [abs(r["ratio"] - r["healthy_network_ratio"]) / max(r["healthy_network_ratio"], 1e-9) for r in meters]
            entry["meter_ratio_definitions"] = {"meters": len(meters), "median_relative_gap": float(np.median(gaps)),
                                                "p90_relative_gap": float(np.percentile(gaps, 90))}
        out["systems"][system] = entry
    return out


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _pct(numerator: int | None, denominator: int | None) -> str:
    if not denominator:
        return "n/a"
    return f"{numerator} of {denominator} ({100.0 * (numerator or 0) / denominator:.0f}%)"


def markdown(report: Mapping[str, Any]) -> str:
    systems = list(report["summary"]["systems"])
    lines = [f"# Compound errors across systems ({', '.join(systems)})", ""]
    lines += ["## Random-pair geometry", "", "| system | lines | pairs | share a bus | within one hop | >= 3 hops | median / max hops | max bus degree | overlap median at 0 / 1 / 2 / 3+ hops |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for system, geometry in report["geometry"].items():
        overlap = geometry.get("overlap_by_hops") or {}
        lines.append(f"| {system} | {geometry['lines']} | {geometry['pairs']} | {100 * geometry['share_bus']:.1f}% | "
                     f"{100 * geometry['within_one_hop']:.1f}% | {100 * geometry['at_least_three_hops']:.1f}% | "
                     f"{geometry['median_hops']:g} / {geometry['max_hops']} | {geometry['bus_degree']['max']} | "
                     + " / ".join(_fmt((overlap.get(k) or {}).get("median"), 3) for k in ("0", "1", "2", "3+")) + " |")
    lines += ["", "## Per class", ""]
    header = ("| class | system | n | matched (every atom >= floor) | solved, deployed | solved, pair refit | matched solved, deployed / pair refit | "
              "mean actions, deployed | silent after one exact fix (matched) | first true fix final with atoms left (deployed) | cue fired |")
    lines += [header, "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name in CLASSES:
        for system in systems:
            cell = report["summary"]["systems"][system]["classes"].get(name) or {}
            deployed, refit = cell.get("deployed") or {}, cell.get("pair_refit") or {}
            first = deployed.get("first_commit") or {}
            silent = (f"{cell.get('silent_after_one_exact_fix')} of {cell.get('multi_atom_roots')} "
                      f"({cell.get('silent_after_one_exact_fix_matched')} of {cell.get('multi_atom_roots_matched')})"
                      if cell.get("multi_atom_roots") else "n/a")
            lines.append(
                f"| {CLASS_LABELS[name]} | {system} | {cell.get('n')} | {cell.get('matched')} | {_pct(deployed.get('solved'), deployed.get('n'))} | "
                f"{_pct(refit.get('solved'), refit.get('n'))} | {deployed.get('matched_solved', 'n/a')} / {refit.get('matched_solved', 'n/a')} of {cell.get('matched')} | "
                f"{_fmt(deployed.get('mean_actions'), 1)} | {silent} | "
                f"{(str(first.get('final_with_atoms_left')) + ' of ' + str(first.get('true'))) if first else 'n/a'} | {cell.get('cue_fired')} of {cell.get('cue_roots')} |")
    lines += ["", "## Ranking ambiguity against the true lines' bus degree", "", "| system | degree | roots | mean lines within the band | band holds two or more | top line is true |",
              "| --- | --- | --- | --- | --- | --- |"]
    for system in systems:
        for key, cell in (report["summary"]["systems"][system].get("band_by_degree") or {}).items():
            lines.append(f"| {system} | {key} | {cell['roots']} | {cell['band_count_mean']:.2f} | {cell['band_two_or_more']} | {cell['top_line_true']} |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- main


def load_suite(path: Path, classes: Sequence[str]) -> list[dict[str, Any]]:
    suite = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(suite, Mapping):
        suite = suite.get("scenarios") or suite.get("roots") or []
    return [r for r in suite if ((r.get("audit") or {}).get("compound") or {}).get("class") in set(classes)]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", action="append", required=True, metavar="SYSTEM=PATH")
    parser.add_argument("--rollout", action="append", default=[], metavar="SYSTEM:ARM=EPISODES")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--classes", nargs="*", default=list(CLASSES))
    parser.add_argument("--matched-floor", type=float, default=1.5)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="roots per class and system (a smoke run)")
    parser.add_argument("--probes", type=Path, default=None, help="reuse a probes.jsonl of an earlier run (same suites)")
    parser.add_argument("--probes-only", action="store_true", help="write the probes and stop")
    args = parser.parse_args(argv)

    started = time.perf_counter()
    suites: dict[str, list[dict[str, Any]]] = {}
    for item in args.suite:
        system, _, path = item.partition("=")
        roots = load_suite(Path(path), args.classes)
        if args.limit:
            taken: Counter = Counter()
            kept = []
            for root in roots:
                name = ((root.get("audit") or {}).get("compound") or {}).get("class")
                if taken[name] < args.limit:
                    taken[name] += 1
                    kept.append(root)
            roots = kept
        suites[system] = roots
        print(f"[compare] {system}: {len(roots)} roots from {path}", flush=True)
    rollouts: dict[str, dict[str, dict[str, dict[str, Any]]]] = defaultdict(dict)
    for item in args.rollout:
        key, _, path = item.partition("=")
        system, _, arm = key.partition(":")
        rollouts[system][arm] = {str(r["scenario_id"]): r for r in load_jsonl(Path(path))}
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    roots = [(label, root) for label in suites for root in suites[label]]
    probes: list[dict[str, Any]] = []
    if args.probes is not None:
        # the probes of an earlier run of the same suites (they read no rollout)
        wanted = {(label, str(root["execution"]["scenario_id"])) for label, root in roots}
        probes = [p for p in load_jsonl(args.probes) if (str(p.get("suite")), str(p["scenario_id"])) in wanted]
        missing = len(wanted) - len({(str(p.get("suite")), str(p["scenario_id"])) for p in probes})
        if missing:
            raise SystemExit(f"{missing} roots have no probe in {args.probes}")
    else:
        with (out / "probes.jsonl").open("w", encoding="utf-8") as stream:
            if args.workers > 1:
                with ProcessPoolExecutor(max_workers=args.workers) as pool:
                    for record in pool.map(_probe_safe, roots, chunksize=2):
                        probes.append(record)
                        stream.write(json.dumps(record, sort_keys=True, default=str) + "\n")
                        if len(probes) % 100 == 0:
                            print(json.dumps({"probed": len(probes), "elapsed": round(time.perf_counter() - started, 1)}), flush=True)
            else:
                for payload in roots:
                    record = _probe_safe(payload)
                    probes.append(record)
                    stream.write(json.dumps(record, sort_keys=True, default=str) + "\n")
    if args.probes_only:
        print(json.dumps({"probes": len(probes), "errors": sum(1 for p in probes if p.get("error")),
                          "seconds": round(time.perf_counter() - started, 1)}), flush=True)
        return 0
    geometry = {label: pair_geometry(str((rows[0].get("grouping") or {}).get("case_id") or "case14"))
                for label, rows in suites.items() if rows}
    summary = summarize(probes, suites, rollouts, matched_floor=args.matched_floor)
    report = {"contract": CONTRACT, "suites": dict(item.partition("=")[::2] for item in args.suite), "rollouts": args.rollout,
              "geometry": geometry, "summary": summary, "probe_errors": [p for p in probes if p.get("error")][:20],
              "total_seconds": round(time.perf_counter() - started, 1)}
    (out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    text = markdown(report)
    (out / "report.md").write_text(text + "\n", encoding="utf-8")
    print(text, flush=True)
    print(json.dumps({"probes": len(probes), "errors": len(report["probe_errors"]), "seconds": report["total_seconds"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
