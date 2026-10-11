"""Are the error setups of the IEEE 14 and IEEE 118 triage sets the same distribution?

    python -m research.classifier_triage.error_setup_audit --output-dir output/classifier_triage_20261011/error_setup_audit

For every family present on both networks (the IEEE 14 study rows and the
IEEE 118 set of ``ieee118_dataset``), three layers are compared:

* **physical error parameters** as drawn: HIF resistance in ohms and in per
  unit of the local base, split, phase, fault current; the unbalance's phase
  fractions as a load-unbalance factor (negative- over positive-sequence
  current share of the split load) and the negative-sequence load it
  injects; harmonic THD and source load; meter error in sigmas and channel
  type; parameter factors;
* **WLS-visible signature**: the excess objective J - dof (the residual
  energy the error injects), log(J / threshold), max |r|, channels above 3
  sigma, assets they touch, the top-2 share of their energy, the voltage and
  reactive shares, the largest branch multiplier, and which test admitted
  the root (chi-square at the margin, max |r| at the margin, or both);
* **set composition**: rows per family, positive share, healthy-alarm
  mechanism.

Each numeric feature gets 10/50/90 % quantiles on both networks and the
two-sample Kolmogorov-Smirnov statistic between them (0 = same distribution,
1 = disjoint).  A matched-strength check then restricts both networks' HIF
roots to the per-unit resistance range they share and both networks'
unbalances to the same negative-sequence-load band, and compares the
signatures again: what remains is the network's effect, not the recipe's.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.stats import chi2, ks_2samp

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.classifier_triage import data, features  # noqa: E402
from research.classifier_triage.transfer_ieee118 import load_ieee118_rows  # noqa: E402

CHI2_ALPHA, RESIDUAL_THRESHOLD, MARGIN = 0.01, 4.0, 1.25
FAMILIES = ("hif", "measurement+hif", "three_phase_unbalance", "harmonic", "measurement", "multi_measurement", "parameter",
            "measurement+parameter", "healthy_window", "mimic_flow_pair_same_sign", "mimic_flow_pair_opposite_sign")
ROOT_FAMILIES = FAMILIES[:8]
SIGNATURE_KEYS = ("excess_objective", "log_chi_square_ratio", "max_residual", "channels_gt3", "assets_gt3", "top2_share",
                  "vm_share_gt3", "reactive_share_gt3", "max_multiplier")
MATCHED_HIF_PU = (2.1, 5.3)
MATCHED_UNBALANCE_NEGSEQ_MW = (5.0, 30.0)
A = np.exp(2j * np.pi / 3)


# ---------------------------------------------------------------- signatures

def signature(case: str, payload: Mapping[str, Any]) -> dict[str, float]:
    net = features.network(case)
    nb, nl = int(net["nb"]), int(net["nl"])
    residual = np.abs(features.neutral_residuals(payload, net))
    dof = int(payload["dof"])
    objective = float(payload["objective"])
    threshold = float(chi2.ppf(1.0 - CHI2_ALPHA, max(dof, 1)))
    hot = np.flatnonzero(residual > 3.0)
    energy = np.sort(residual[hot] ** 2)[::-1]
    block = np.searchsorted(np.cumsum([nb, nb, nb, nl, nl, nl, nl]), hot, side="right")
    assets = {("bus", int(i % nb)) if i < 3 * nb else ("branch", int((i - 3 * nb) % nl)) for i in hot}
    raw_lambdas = payload.get("lambda_normalized")
    lambdas = np.abs(np.asarray([] if raw_lambdas is None else raw_lambdas, dtype=float))
    max_r = float(residual.max())
    return {
        "excess_objective": objective - dof, "log_chi_square_ratio": float(np.log(max(objective, 1e-9) / threshold)),
        "max_residual": max_r, "channels_gt3": float(hot.size), "assets_gt3": float(len(assets)),
        "top2_share": float(energy[:2].sum() / energy.sum()) if energy.size else 0.0,
        "vm_share_gt3": float(np.mean(block == 0)) if hot.size else 0.0,
        "reactive_share_gt3": float(np.mean(np.isin(block, (2, 4, 6)))) if hot.size else 0.0,
        "max_multiplier": float(lambdas.max()) if lambdas.size else 0.0,
        "chi_admitted": float(objective > MARGIN * threshold), "residual_admitted": float(max_r >= MARGIN * RESIDUAL_THRESHOLD),
        "chi_alarm": float(objective >= threshold), "residual_alarm": float(max_r >= RESIDUAL_THRESHOLD),
    }


def load_unbalance_factor(fractions: Mapping[str, float] | Sequence[float]) -> float:
    """|I2| / |I1| of a load split over the phases at balanced voltage, from its fractions."""
    fa, fb, fc = ([float(fractions[k]) for k in ("a", "b", "c")] if isinstance(fractions, Mapping) else [float(v) for v in fractions])
    return float(abs(fa + A * fb + A * A * fc) / (fa + fb + fc))


# ------------------------------------------------------------------ physical

def _case_loads(case: str) -> dict[int, float]:
    from mcp_server.matpower_server import _load_python_case

    bus = np.asarray(_load_python_case(case)["bus"], dtype=float)
    return {int(row[0]): float(row[2]) for row in bus}


def _ieee14_waveform_rows() -> dict[str, dict[str, Any]]:
    """The IEEE 14 corpus rows by id (for the operating point and the split bus's load)."""
    from psse_env.providers.scenario_generator import PHYSICAL_HIF_SAMPLE_PATHS, PHYSICAL_IMBALANCE_SAMPLE_PATH

    rows: dict[str, dict[str, Any]] = {}
    for path in [*PHYSICAL_HIF_SAMPLE_PATHS, PHYSICAL_IMBALANCE_SAMPLE_PATH]:
        if not Path(path).is_file():
            continue
        with Path(path).open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                rows[str(row["id"])] = {"op_point": row.get("op_point"), "label": row.get("label")}
    return rows


def physical(row: Mapping[str, Any], network: str, corpus14: Mapping[str, Mapping[str, Any]], loads: Mapping[str, Mapping[int, float]]) -> dict[str, float]:
    record = row["record"]
    truth = record.get("truth") or {}
    out: dict[str, float] = {}
    sigma = np.asarray((record.get("metadata") or {}).get("sigma_z") or [], dtype=float)
    if truth.get("hif"):
        hif = truth["hif"][0]
        r_ohm = float(hif.get("resistance_ohm") or hif.get("r_hif_ohm") or np.nan)
        out.update(hif_r_ohm=r_ohm, hif_r_pu=float(hif.get("r_hif_pu") or np.nan), hif_kv=float(hif.get("kv_ll") or np.nan),
                   hif_split=float(hif.get("split_ratio") if hif.get("split_ratio") is not None else hif.get("alpha", np.nan)),
                   hif_fault_current_pu=float(hif.get("fault_current_pu") or hif.get("nominal_fault_current_pu") or np.nan))
        phase = hif.get("phase")
        out["hif_phase"] = float("ABC".index(phase) + 1) if isinstance(phase, str) else float(phase)
    if truth.get("unbalance"):
        unb = truth["unbalance"][0]
        fractions = unb.get("load_split_fractions")
        bus = int(unb.get("bus") or unb.get("unbalance_bus"))
        luf = load_unbalance_factor(fractions)
        if network == "ieee14":
            source = corpus14.get(str(record.get("source_id")), {})
            scale = float((source.get("op_point") or {}).get("load_scale") or 1.0)
            total_mw = loads["case14"].get(bus, np.nan) * scale
        else:
            total_mw = float(unb.get("total_kw", np.nan)) / 1000.0
        out.update(unb_luf=luf, unb_total_mw=total_mw, unb_negseq_mw=luf * total_mw,
                   unb_max_dev=float(max(abs(v - 1 / 3) for v in (fractions.values() if isinstance(fractions, Mapping) else fractions))))
        if unb.get("vuf_at_bus") is not None:
            out["unb_vuf"] = float(unb["vuf_at_bus"])
    if truth.get("harmonic"):
        har = truth["harmonic"][0]
        bus = int(har.get("bus_1based"))
        case = "case14" if network == "ieee14" else "case118"
        out.update(har_thd=float(har.get("actual_thd") or har.get("thd_target")),
                   har_source_mw=loads[case].get(bus, np.nan) * float(har.get("load_scale") or 1.0))
    if truth.get("measurement") and sigma.size:
        multiples = [abs(float(m["observed"]) - float(m["clean"])) / float(sigma[int(m["index"])]) for m in truth["measurement"]]
        out.update(meter_sigmas_max=max(multiples), meter_count=float(len(multiples)))
        net = features.network("case14" if network == "ieee14" else "case118")
        nb = int(net["nb"])
        out["meter_voltage_share"] = float(np.mean([int(m["index"]) < nb for m in truth["measurement"]]))
    if truth.get("parameter"):
        par = truth["parameter"][0]
        r_f, x_f = float(par.get("r_factor") or 1.0), float(par.get("x_factor") or 1.0)
        out.update(par_log_r_factor=float(np.log(r_f)), par_log_x_factor=float(np.log(x_f)),
                   par_abs_log_factor=float(max(abs(np.log(r_f)), abs(np.log(x_f)))), par_both=float(r_f != 1.0 and x_f != 1.0))
    return out


# ------------------------------------------------------------------- tables

def _quantiles(values: Sequence[float]) -> list[float] | None:
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    return np.quantile(arr, [0.1, 0.5, 0.9]).round(3).tolist() if arr.size else None


def compare(a: Sequence[float], b: Sequence[float]) -> dict[str, Any]:
    arr_a = np.asarray([v for v in a if v is not None and np.isfinite(v)], dtype=float)
    arr_b = np.asarray([v for v in b if v is not None and np.isfinite(v)], dtype=float)
    out = {"ieee14": _quantiles(arr_a), "ieee118": _quantiles(arr_b), "n14": int(arr_a.size), "n118": int(arr_b.size)}
    if arr_a.size and arr_b.size:
        ks = ks_2samp(arr_a, arr_b)
        out["ks"] = float(ks.statistic)
        out["ks_p"] = float(ks.pvalue)
    return out


def family_tables(rows: Mapping[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Per family: every numeric feature compared across networks, plus admission mechanism shares."""
    out: dict[str, Any] = {}
    for family in FAMILIES:
        sub = {net: [r for r in rows[net] if r["family"] == family] for net in ("ieee14", "ieee118")}
        if not sub["ieee14"] or not sub["ieee118"]:
            continue
        keys = sorted({k for net in sub for r in sub[net] for k in (*r["signature"], *r["physical"])}
                      - {"chi_admitted", "residual_admitted", "chi_alarm", "residual_alarm"})
        entry: dict[str, Any] = {"rows": {net: len(sub[net]) for net in sub}, "features": {}}
        for key in keys:
            values = {net: [r["signature"].get(key, r["physical"].get(key)) for r in sub[net]] for net in sub}
            if all(v is None for v in values["ieee14"]) or all(v is None for v in values["ieee118"]):
                continue
            entry["features"][key] = compare(values["ieee14"], values["ieee118"])
        mechanism = {}
        for net in sub:
            margin_keys = ("chi_admitted", "residual_admitted") if family in ROOT_FAMILIES else ("chi_alarm", "residual_alarm")
            chi = np.asarray([r["signature"][margin_keys[0]] for r in sub[net]], dtype=bool)
            res = np.asarray([r["signature"][margin_keys[1]] for r in sub[net]], dtype=bool)
            mechanism[net] = {"chi_square_only": float(np.mean(chi & ~res)), "residual_only": float(np.mean(res & ~chi)),
                              "both": float(np.mean(chi & res)), "neither": float(np.mean(~chi & ~res))}
        entry["admission_mechanism"] = mechanism
        if family in ("hif", "measurement+hif"):
            for net in sub:
                entry.setdefault("hif_kv_share", {})[net] = dict(Counter(f"{r['physical'].get('hif_kv'):g} kV" for r in sub[net]))
                entry.setdefault("hif_phase_share", {})[net] = dict(Counter(int(r["physical"].get("hif_phase", 0)) for r in sub[net]))
        if family in ("measurement", "multi_measurement"):
            for net in sub:
                channels = Counter()
                nb = int(features.network("case14" if net == "ieee14" else "case118")["nb"])
                nl = int(features.network("case14" if net == "ieee14" else "case118")["nl"])
                for r in sub[net]:
                    for m in r["record"]["truth"]["measurement"]:
                        i = int(m["index"])
                        channels["Vm" if i < nb else "Pinj" if i < 2 * nb else "Qinj" if i < 3 * nb else ("Pf", "Qf", "Pt", "Qt")[(i - 3 * nb) // nl]] += 1
                total = sum(channels.values())
                entry.setdefault("meter_channel_share", {})[net] = {k: round(v / total, 3) for k, v in sorted(channels.items())}
        if family in ("parameter", "measurement+parameter"):
            for net in sub:
                entry.setdefault("parameter_kind_share", {})[net] = dict(Counter(str(r["record"]["truth"]["parameter"][0].get("parameter")) for r in sub[net]))
        out[family] = entry
    return out


def matched_strength(rows: Mapping[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Signatures after restricting both networks to the same physical strength band."""
    out: dict[str, Any] = {}
    selections: dict[str, tuple[str, Callable[[Mapping[str, float]], bool], str]] = {
        "hif": ("hif", lambda p: MATCHED_HIF_PU[0] <= p.get("hif_r_pu", -1) <= MATCHED_HIF_PU[1],
                f"HIF resistance {MATCHED_HIF_PU[0]}-{MATCHED_HIF_PU[1]} pu of the local base (138 kV lines only on IEEE 118)"),
        "three_phase_unbalance": ("three_phase_unbalance",
                                  lambda p: MATCHED_UNBALANCE_NEGSEQ_MW[0] <= p.get("unb_negseq_mw", -1) <= MATCHED_UNBALANCE_NEGSEQ_MW[1],
                                  f"negative-sequence load {MATCHED_UNBALANCE_NEGSEQ_MW[0]}-{MATCHED_UNBALANCE_NEGSEQ_MW[1]} MW"),
        "harmonic": ("harmonic", lambda p: 0.12 <= p.get("har_thd", -1) <= 0.18 and 5.0 <= p.get("har_source_mw", -1) <= 60.0,
                     "source THD 0.12-0.18 and source load 5-60 MW"),
        "measurement": ("measurement", lambda p: 11.0 <= p.get("meter_sigmas_max", -1) <= 14.0 and p.get("meter_voltage_share", 1) == 0.0,
                        "one bad power meter of 11-14 sigma"),
    }
    for name, (family, keep, label) in selections.items():
        sub = {net: [r for r in rows[net] if r["family"] == family and keep(r["physical"])] for net in ("ieee14", "ieee118")}
        if not sub["ieee14"] or not sub["ieee118"]:
            continue
        entry: dict[str, Any] = {"selection": label, "rows": {net: len(sub[net]) for net in sub}, "features": {}}
        for key in (*SIGNATURE_KEYS, "hif_r_pu", "hif_fault_current_pu", "unb_negseq_mw", "unb_luf", "har_thd", "meter_sigmas_max"):
            values = {net: [r["signature"].get(key, r["physical"].get(key)) for r in sub[net]] for net in sub}
            if all(v is None for v in values["ieee14"]) or all(v is None for v in values["ieee118"]):
                continue
            entry["features"][key] = compare(values["ieee14"], values["ieee118"])
        out[name] = entry
    return out


def composition(rows: Mapping[str, list[dict[str, Any]]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for net, items in rows.items():
        families = Counter(r["family"] for r in items)
        positives = sum(r["triage"]["needs_aux"] for r in items)
        healthy = [r for r in items if r["family"] == "healthy_window"]
        mechanism = Counter()
        for r in healthy:
            s = r["signature"]
            mechanism["chi_square_only" if s["chi_alarm"] and not s["residual_alarm"] else "residual_only" if s["residual_alarm"] and not s["chi_alarm"] else "both"] += 1
        out[net] = {"rows": len(items), "positives": int(positives), "positive_share": float(positives / len(items)),
                    "families": dict(sorted(families.items())),
                    "healthy_alarm_mechanism": {k: v / max(len(healthy), 1) for k, v in mechanism.items()},
                    "healthy_max_residual_q50": float(np.median([r["signature"]["max_residual"] for r in healthy])) if healthy else None}
    return out


# ---------------------------------------------------------------------- main

def collect(dataset118: Path, workers: int) -> dict[str, list[dict[str, Any]]]:
    corpus14 = _ieee14_waveform_rows()
    loads = {"case14": _case_loads("case14"), "case118": _case_loads("case118")}
    rows: dict[str, list[dict[str, Any]]] = {"ieee14": [], "ieee118": []}
    for row in data.load_rows(workers=workers):
        if row["kind"] == "child" or row["family"] not in FAMILIES:
            continue
        row["signature"] = signature("case14", row["payload"])
        row["physical"] = physical(row, "ieee14", corpus14, loads)
        rows["ieee14"].append(row)
    for row in load_ieee118_rows(dataset118):
        if row["family"] not in FAMILIES:
            continue
        row["signature"] = signature("case118", row["payload"])
        row["physical"] = physical(row, "ieee118", corpus14, loads)
        rows["ieee118"].append(row)
    return rows


def _fmt(q: list[float] | None) -> str:
    return "n/a" if q is None else f"{q[0]:g} / {q[1]:g} / {q[2]:g}"


def write_report(summary: Mapping[str, Any], path: Path) -> None:
    lines = ["# Error setup audit: IEEE 14 study rows against the IEEE 118 set\n",
             "Quantiles are 10 / 50 / 90 %. KS is the two-sample Kolmogorov-Smirnov statistic between the networks "
             "(0 same, 1 disjoint); the KS flag marks 0.5 and above.\n"]
    comp = summary["composition"]
    lines += ["## Composition\n", "| | IEEE 14 | IEEE 118 |", "| --- | --- | --- |",
              f"| rows | {comp['ieee14']['rows']} | {comp['ieee118']['rows']} |",
              f"| rows that need phasors | {comp['ieee14']['positives']} ({100 * comp['ieee14']['positive_share']:.0f}%) | "
              f"{comp['ieee118']['positives']} ({100 * comp['ieee118']['positive_share']:.0f}%) |",
              f"| healthy alarms by max abs r only | {100 * comp['ieee14']['healthy_alarm_mechanism'].get('residual_only', 0):.0f}% | "
              f"{100 * comp['ieee118']['healthy_alarm_mechanism'].get('residual_only', 0):.0f}% |",
              f"| healthy alarms, median max abs r | {comp['ieee14']['healthy_max_residual_q50']:.2f} | {comp['ieee118']['healthy_max_residual_q50']:.2f} |",
              "", "| family | IEEE 14 | IEEE 118 |", "| --- | --- | --- |"]
    for family in sorted(set(comp["ieee14"]["families"]) | set(comp["ieee118"]["families"])):
        lines.append(f"| {family} | {comp['ieee14']['families'].get(family, 0)} | {comp['ieee118']['families'].get(family, 0)} |")
    for family, entry in summary["families"].items():
        lines += ["", f"## {family} ({entry['rows']['ieee14']} IEEE 14 rows, {entry['rows']['ieee118']} IEEE 118 rows)\n",
                  "| feature | IEEE 14 | IEEE 118 | KS | |", "| --- | --- | --- | --- | --- |"]
        for key, cell in sorted(entry["features"].items(), key=lambda item: -(item[1].get("ks") or 0)):
            ks = cell.get("ks")
            lines.append(f"| {key} | {_fmt(cell['ieee14'])} | {_fmt(cell['ieee118'])} | {ks:.2f} | {'**differs**' if ks is not None and ks >= 0.5 else ''} |"
                         if ks is not None else f"| {key} | {_fmt(cell['ieee14'])} | {_fmt(cell['ieee118'])} | | |")
        mech = entry["admission_mechanism"]
        lines += ["", "| admitted by | IEEE 14 | IEEE 118 |", "| --- | --- | --- |"]
        for key in ("chi_square_only", "residual_only", "both", "neither"):
            lines.append(f"| {key} | {100 * mech['ieee14'][key]:.0f}% | {100 * mech['ieee118'][key]:.0f}% |")
        for extra in ("hif_kv_share", "hif_phase_share", "meter_channel_share", "parameter_kind_share"):
            if extra in entry:
                lines += ["", f"{extra}: `{json.dumps(entry[extra])}`"]
    lines += ["", "## Matched strength: the same physical error on both networks\n"]
    for name, entry in summary["matched_strength"].items():
        lines += [f"### {name}: {entry['selection']} ({entry['rows']['ieee14']} IEEE 14 rows, {entry['rows']['ieee118']} IEEE 118 rows)\n",
                  "| feature | IEEE 14 | IEEE 118 | KS | |", "| --- | --- | --- | --- | --- |"]
        for key, cell in entry["features"].items():
            ks = cell.get("ks")
            lines.append(f"| {key} | {_fmt(cell['ieee14'])} | {_fmt(cell['ieee118'])} | {ks:.2f} | {'**differs**' if ks is not None and ks >= 0.5 else ''} |"
                         if ks is not None else f"| {key} | {_fmt(cell['ieee14'])} | {_fmt(cell['ieee118'])} | | |")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", default=str(REPO_ROOT / "output" / "classifier_triage_20261007" / "ieee118"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = collect(Path(args.dataset_dir), args.workers)
    summary = {"composition": composition(rows), "families": family_tables(rows), "matched_strength": matched_strength(rows)}
    (out / "error_setup_audit.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    write_report(summary, out / "error_setup_audit.md")
    print((out / "error_setup_audit.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
