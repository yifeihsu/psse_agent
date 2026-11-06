"""Tables of the sparse multi-error design note from the separability probe's outputs (2026-10-28).

    python -m research.compound_errors.sparse_report --probe output/sparse_multi_error_probe_20261028 \\
        --variant output/sparse_multi_error_probe_20261028/lambda4 --output output/sparse_multi_error_probe_20261028/report.md

``--probe`` is the run at the deployed flag levels (normalized residual 4, line multiplier 3), ``--variant``
the run with the line flag at 4.  Every number in ``docs/sparse_multi_error_design_20261028.md`` is a cell of
one of these tables or a key of a run's ``summary.json``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

SYSTEMS = ["ieee14", "ieee118"]   # narrowed to the systems a run holds (main)


def _load(path: Path) -> dict[str, Any]:
    return json.loads((path / "summary.json").read_text(encoding="utf-8"))


def _records(path: Path, system: str, name: str) -> list[dict[str, Any]]:
    f = path / f"combos_{system}_{name}.jsonl"
    return [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line.strip()] if f.exists() else []


def _pct(x: Any) -> str:
    return "n/a" if x is None else f"{100 * float(x):.0f}%"


def _num(x: Any, digits: int = 2) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, (list, tuple)):
        return " / ".join(_num(v, digits) for v in x)
    return f"{float(x):.{digits}f}"


def _table(head: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    out = ["| " + " | ".join(head) + " |", "| " + " | ".join("---" for _ in head) + " |"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def systems_table(s: Mapping[str, Any], v: Mapping[str, Any]) -> str:
    rows = []
    for name in SYSTEMS:
        c = s["systems"][name]["curves"]
        nv = v["systems"][name]["curves"]["noise_only"]
        n = c["noise_only"]
        rows.append([name, c["nb"], c["nl"], c["m"], c["dof"], _num(c["threshold"], 1), _num(c["excess"], 1),
                     c["bus_distance_max"], f'{c["degree_max"]} / {_num(c["degree_mean"], 1)}',
                     _pct(n["global_false_alarm"]), _pct(n["global_chi_false_alarm"]), _pct(n["snapshots_with_flag"]),
                     _pct(nv["snapshots_with_flag"]), _num(n["clusters_mean"]), _pct(n["noise_cluster_local_false_alarm"]),
                     _pct(c["satterthwaite_nominal_false_alarm"]["median"]),
                     f'{_num(c["redundancy"]["min"])} ({c["redundancy"]["below_0.3"]} below 0.3)'])
    return _table(["system", "buses", "lines", "meters m", "dof", "chi-square threshold", "threshold excess",
                   "diameter (hops)", "degree max / mean", "noise-only global alarm", "... chi-square part",
                   "noise-only snapshots with a flag (lambda 3)", "... (lambda 4)", "noise clusters per snapshot (lambda 3)",
                   "local alarm on a noise cluster", "local alarm, unselected 1-hop set", "min redundancy 1-h_ii"], rows)


def footprint_table(s: Mapping[str, Any], kinds=("meter", "parameter", "outage"),
                    ratios=("0.3", "0.6", "1.0", "1.5", "3.0", "10.0", "20.0")) -> str:
    rows = []
    for kind in kinds:
        for ratio in ratios:
            cells = [kind, ratio]
            for name in SYSTEMS:
                f = s["systems"][name]["footprints"].get(f"{kind}@{ratio}")
                if f is None:
                    cells += ["n/a"] * 5
                    continue
                cells += [f'{_num(f["n_buses"][1], 0)} [{_num(f["n_buses"][2], 0)}]',
                          _num(f["radius"][2], 0) if f["radius"] else "n/a",
                          f'{_pct(f["global_alarm"])} ({_pct(f["global_chi_alarm"])})', _pct(f["local_alarm"]),
                          f'{_num(f["local_dof"][1], 0)} / {_num(f["local_ratio"][1], 1)}']
            rows.append(cells)
    head = ["kind", "global effect ratio"]
    for name in SYSTEMS:
        head += [f"{name}: footprint buses median [p90]", "radius p90", "global alarm (chi-square part)",
                 "local alarm", "local dof / local effect ratio (medians)"]
    return _table(head, rows)


def decay_table(s: Mapping[str, Any]) -> str:
    rows = []
    for kind in ("line_line", "line_meter", "meter_meter"):
        for d in ("0", "1", "2", "3", "4", "5", "6"):
            cells = [kind.replace("_", "-"), d if d != "6" else "6+"]
            for name in SYSTEMS:
                e = s["systems"][name]["curves"]["correlation_decay"][kind].get(d)
                cells += ["n/a"] * 3 if e is None else [_num(e["median"], 3), _num(e["p99"], 3), _num(e["max"], 3)]
            rows.append(cells)
    head = ["hypotheses", "bus distance"]
    for name in SYSTEMS:
        head += [f"{name}: median abs corr", "p99", "max"]
    return _table(head, rows)


SEP_SETS = ("pairs_random", "triples_random", "pairs_meter_line", "pairs_ratio_1.5", "pairs_ratio_3.0", "pairs_ratio_10.0")


def separability_table(s: Mapping[str, Any], v: Mapping[str, Any]) -> str:
    rows = []
    for set_name in SEP_SETS:
        for name in SYSTEMS:
            x = s["systems"][name]["sets"].get(set_name)
            y = v["systems"][name]["sets"].get(set_name)
            if not x or not x.get("n"):
                continue
            cp = x.get("criterion_parts", {})
            sens = x.get("separable_sensitivity", {})
            rows.append([set_name, name, x["n"], _pct(x.get("separable")), _pct(cp.get("S1_all_visible")),
                         _pct(cp.get("S2_distinct_and_pure")), _pct(cp.get("S3_masking_keep_0.9")),
                         _pct(x.get("noisy_partition_recovered")),
                         f'{_pct(sens.get("link0"))} / {_pct(sens.get("link2"))}' if sens else "n/a",
                         f'{_pct(y.get("separable"))} / {_pct(y.get("noisy_partition_recovered"))}' if y else "n/a",
                         _pct(x.get("atoms_isolated"))])
    return _table(["set", "system", "n", "separable", "S1 all visible", "S2 distinct and pure", "S3 masking >= 0.9",
                   "noisy partition recovered", "separable at link 0 / link 2", "lambda 4: separable / noisy",
                   "atoms isolated"], rows)


def distance_table(s: Mapping[str, Any]) -> str:
    rows = []
    for d in ("0", "1", "2", "3", "4", "5", "6"):
        cells = [d if d != "6" else "6+"]
        for name in SYSTEMS:
            e = s["systems"][name]["sets"]["pairs_by_distance"]["by_distance"].get(d)
            if e is None:
                cells += ["n/a"] * 6
                continue
            cells += [e["n"], _pct(e["separable"]), _num(e["overlap"][0]),
                      _num(e["own_corr"][2] if e["own_corr"] else None),
                      _num(e["cross_block_max"][2] if e["cross_block_max"] else None),
                      _pct(e["noisy_partition_recovered"])]
        rows.append(cells)
    head = ["atom distance (hops)"]
    for name in SYSTEMS:
        head += [f"{name}: n", "separable", "footprint overlap median", "own-hypothesis corr p90",
                 "cross-cluster block max p90", "noisy partition recovered"]
    return _table(head, rows)


def masking_table(s: Mapping[str, Any], v: Mapping[str, Any]) -> str:
    rows = []
    for set_name in ("pairs_random", "triples_random", "pairs_ratio_1.5", "variable_density_2.0", "variable_density_4.0"):
        for name in SYSTEMS:
            x = s["systems"][name]["sets"].get(set_name)
            if not x or not x.get("n"):
                continue
            rows.append([set_name, name, _num(x.get("c_own_alarm_alone", {}).get("mean")),
                         _pct(x.get("c_unflagged_with_others_share")), _num(x.get("c_after_others_fixed_alarm")),
                         _pct(x.get("c_after_others_fixed_silent_share")), x.get("d_global_silent_draws"),
                         _pct(x.get("d_local_flags_when_global_silent")),
                         _pct(x.get("d_oracle_local_flags_when_global_silent"))])
    return _table(["set", "system", "P(alarm), atom alone", "atoms unflagged with the others present",
                   "P(alarm) after the others are fixed", "atoms silent after the others are fixed (P < 0.5)",
                   "silent draws", "cluster-local chi-square flags them", "... on the noiseless cluster"], rows)


VARIABLE = tuple(f"variable_density_{d}{suf}" for d in ("1.0", "2.0", "4.0") for suf in ("", "_natural_outages"))


def variable_table(s: Mapping[str, Any], path: Path) -> str:
    rows = []
    for set_name in VARIABLE:
        for name in SYSTEMS:
            x = s["systems"][name]["sets"].get(set_name)
            if not x or not x.get("n"):
                continue
            peel = x.get("peel") or {}
            rk = x.get("line_atom_rank") or {}
            nc = x.get("n_clusters_noiseless") or {}
            clusters = sum(int(k) * v for k, v in nc.items()) / max(1, sum(nc.values()))
            rows.append([set_name.replace("variable_density_", "").replace("_natural_outages", " (natural outages)"),
                         name, x["n"], _num(x.get("atoms_per_root"), 0), _num(clusters, 1), _pct(x.get("separable")),
                         f'{_pct(x.get("atoms_isolated"))} / {_pct(x.get("atoms_isolated_noisy"))}',
                         _num(peel.get("rounds_over_atoms", [None, None, None])[1]), _pct(peel.get("fixed_alone_share")),
                         _pct(peel.get("never_flagged_share")),
                         f'{_pct(rk.get("global_top1"))} / {_pct(rk.get("cluster_top1"))}'])
    return _table(["errors per 100 branches", "system", "roots", "atoms per root p10 / p50 / p90", "clusters per root",
                   "roots separable", "atoms isolated noiseless / noisy", "peel rounds per atom (median)",
                   "atoms corrected alone while peeling", "atoms never flagged", "line atom first: global / in cluster"],
                  rows)


RULES = (("floor_global_j", "floor, J basis"), ("floor_global_excess", "floor, excess basis"),
         ("floor_cluster_local", "floor, cluster-local"), ("deployed_rule", "deployed (final or J floor)"),
         ("excess_rule", "final or excess floor"), ("cluster_resolved", "cluster resolved"),
         ("resolved_and_best", "resolved and best in the cluster"))


def acceptance_table(s: Mapping[str, Any], sets=("pairs_random", "triples_random", "variable_density_2.0",
                                                 "variable_density_4.0", "variable_density_1.0")) -> str:
    rows = []
    for set_name in sets:
        for name in SYSTEMS:
            x = s["systems"][name]["sets"].get(set_name)
            if not x or "progress" not in x:
                continue
            for split in ("alone", "merged"):
                p = x["progress"].get(split) or {}
                if "true" not in p:
                    continue
                cells = [set_name, name, split, p["true"]["n"]]
                for key, _ in RULES:
                    cells.append(" / ".join(_pct(p[c][key]) if c in p else "n/a" for c in ("true", "wrong_line", "wrong_meter")))
                rows.append(cells)
    return _table(["set", "system", "atom's cluster", "corrections"] + [f"{label}: true / wrong line / wrong meter"
                                                                        for _, label in RULES], rows)


def rank_table(s: Mapping[str, Any], path: Path) -> str:
    rows = []
    for set_name in ("pairs_random", "triples_random", "variable_density_2.0", "variable_density_4.0"):
        for name in SYSTEMS:
            recs = _records(path, name, set_name)
            pairs = [(g, c) for r in recs for g, c in zip(r["own_line_rank_global"], r["own_line_rank_cluster"]) if g is not None]
            if not pairs:
                continue
            def share(fn):
                return _pct(np.mean([fn(g, c) for g, c in pairs]))
            rows.append([set_name, name, len(pairs)] + [share(lambda g, c, k=k: g <= k) for k in (1, 3, 5)]
                        + [share(lambda g, c, k=k: c is not None and c <= k) for k in (1, 3, 5)])
    return _table(["set", "system", "line atoms", "global rank 1", "global top 3", "global top 5",
                   "cluster rank 1", "cluster top 3", "cluster top 5"], rows)


def suite_table(s: Mapping[str, Any]) -> str:
    rows = []
    for cls, e in sorted(s.get("suite14_natural", {}).items()):
        if not e.get("evaluated"):
            rows.append([cls, e["n"], 0] + ["skipped: bus splits"] + [""] * 10)
            continue
        rows.append([cls, e["n"], e["evaluated"], _num(e.get("root_linear_vs_nonlinear_ratio"), 3),
                     _num(e.get("linear_vs_nonlinear_fixed_state_ratio"), 3),
                     _num(e.get("linear_vs_recorded_solo_ratio")),
                     f'{e.get("recorded_structural_atom_below_chi")} / {e.get("linear_structural_atom_below_chi")}',
                     _pct(e.get("separable")), _pct(e.get("S2_distinct_and_pure")), _pct(e.get("S3_masking")),
                     _pct(e.get("noisy_partition_recovered")), _pct(e.get("atom_silent_after_fix_share")),
                     _pct(e.get("local_flags_when_global_silent")), _pct(e.get("oracle_local_flags_when_global_silent"))])
    return _table(["class", "roots", "evaluated", "root: linear / nonlinear WLS on its noiseless vector p10 / p50 / p90",
                   "atom alone: linear / nonlinear WLS at the root's true state",
                   "linear / recorded solo chi-square p10 / p50 / p90",
                   "roots with a structural atom below the alarm alone: recorded / linear", "separable",
                   "S2 distinct and pure", "S3 masking >= 0.9", "noisy partition recovered",
                   "atoms silent after the others are fixed", "cluster-local flags them", "... noiseless cluster"], rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--probe", required=True, type=Path)
    parser.add_argument("--variant", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    s, v = _load(args.probe), _load(args.variant)
    SYSTEMS[:] = [n for n in SYSTEMS if n in s["systems"] and n in v["systems"]]
    parts = [
        "# Sparse multi-error separability probe: tables (2026-10-28)",
        f"Probe: `{args.probe.as_posix()}` (line flag {s['config']['lambda_flag']}); variant: `{args.variant.as_posix()}` "
        f"(line flag {v['config']['lambda_flag']}).  Seed {s['seed']}, {s['config']['states']} operating points per system, "
        f"{s['config']['draws']} noise draws per combination.",
        "## T1. The two systems", systems_table(s, v),
        "## T2. Footprints of one atom at a fixed global effect ratio", footprint_table(s),
        "## T3. Correlation of the normalized multipliers against bus distance", decay_table(s),
        "## T4. Separability of pairs and triples at matched effect ratios", separability_table(s, v),
        "## T5. Pairs by distance", distance_table(s),
        "## T6. An atom's own effect: alone, with the others, after they are fixed", masking_table(s, v),
        "## T7. Variable-count draws", variable_table(s, args.probe),
        "## T8. What the verifier's rules would keep", acceptance_table(s),
        "## T9. Where a line atom's line ranks: the global list against its cluster", rank_table(s, args.probe),
        "## T10. The IEEE 14 compound suite at its natural magnitudes", suite_table(s),
    ]
    args.output.write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
