#!/usr/bin/env python3
"""Statistics of drawn test suites, for the paper's description of the test sets (2026-10-29).

    suite_stats.py --suite LABEL=SUITE.json [--suite LABEL=SUITE.json ...] [--output OUT.json]

For each suite and family: roots and error cardinalities; the injected meter
errors (errors per root, channels, magnitude |observed - clean| in pu and in
multiples of the channel's sigma); the line R/X errors (r and x factors, the
parameter-ranking stratum, dominance ratio and true-line rank); the
breaker-status errors (physical effect, breaker, load scale, operating
point); and the auxiliary data each root carries (parameter scans, substation
telemetry).  Everything is read from what the suite builder stores under
``audit.truth``, ``audit.parameter_ranking`` and ``execution.metadata``; a
field absent from a row is counted as absent.  With two or more suites the
markdown puts their families side by side, so a second draw can be compared
with the first.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

FAMILY_ORDER = ("measurement", "multi_measurement", "parameter", "topology",
                "measurement+parameter", "measurement+topology", "no_error")
#: The IEEE 14 SCADA vector of the suites (122 values): bus voltage magnitudes, bus injections,
#: then the 20 branches' from-end and to-end flows; used to name a meter error whose truth record
#: carries only its index (the mixed families).
CHANNELS_122 = (("Vm", 0, 14), ("Pinj", 14, 28), ("Qinj", 28, 42), ("Pf", 42, 62), ("Qf", 62, 82),
                ("Pt", 82, 102), ("Qt", 102, 122))


def channel_of(error: Mapping[str, Any], measurements: Sequence[Any]) -> str:
    if error.get("channel"):
        return str(error["channel"])
    index = error.get("index")
    if isinstance(index, int) and len(measurements) == 122:
        for name, low, high in CHANNELS_122:
            if low <= index < high:
                return name
    return "?"


def _family(row: Mapping[str, Any]) -> str:
    return str((row.get("grouping") or {}).get("scenario_family") or row.get("scenario_family") or "")


def _find_key(node: Any, key: str, predicate=None, depth: int = 0):
    """Depth-first search for ``key`` in nested mappings (first match wins)."""
    if depth > 6:
        return None
    if isinstance(node, Mapping):
        if key in node and (predicate is None or predicate(node[key])):
            return node[key]
        for value in node.values():
            found = _find_key(value, key, predicate, depth + 1)
            if found is not None:
                return found
    elif isinstance(node, list):
        for value in node:
            found = _find_key(value, key, predicate, depth + 1)
            if found is not None:
                return found
    return None


def _spread(values: Sequence[float]) -> dict[str, float | int | None]:
    values = [float(v) for v in values if v is not None]
    if not values:
        return {"n": 0, "min": None, "median": None, "max": None}
    return {"n": len(values), "min": min(values), "median": statistics.median(values), "max": max(values)}


def describe_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    by_family: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_family[_family(row)].append(row)
    result: dict[str, Any] = {"roots": len(rows), "families": {}}
    for family in sorted(by_family, key=lambda f: FAMILY_ORDER.index(f) if f in FAMILY_ORDER else 99):
        members = by_family[family]
        cardinality: Counter = Counter()
        meters_per_root: Counter = Counter()
        channels: Counter = Counter()
        meter_pu: list[float] = []
        meter_sigma: list[float] = []
        r_factors: list[float] = []
        x_factors: list[float] = []
        strata: Counter = Counter()
        dominance: list[float] = []
        ranks: Counter = Counter()
        effects: Counter = Counter()
        breakers: Counter = Counter()
        load_scales: list[float] = []
        operating_points: Counter = Counter()
        scans_present = telemetry_present = 0
        scans_per_root: list[int] = []
        for row in members:
            grouping = row.get("grouping") or {}
            cardinality[str(grouping.get("error_cardinality"))] += 1
            audit = row.get("audit") or {}
            truth = audit.get("truth") or {}
            execution = row.get("execution") or {}
            meta = execution.get("metadata") or {}
            measurements = execution.get("measurements") or []
            sigma = _find_key(meta, "sigma_z", lambda v: isinstance(v, list) and len(v) == len(measurements))
            errors = truth.get("true_measurement_errors") or []
            meters_per_root[str(len(errors))] += 1
            for error in errors:
                channels[channel_of(error, measurements)] += 1
                try:
                    magnitude = abs(float(error["observed"]) - float(error["clean"]))
                except (KeyError, TypeError, ValueError):
                    continue
                meter_pu.append(magnitude)
                index = error.get("index")
                if sigma is not None and isinstance(index, int) and 0 <= index < len(sigma) and float(sigma[index]) > 0:
                    meter_sigma.append(magnitude / float(sigma[index]))
            for error in truth.get("true_parameter_errors") or []:
                if error.get("r_factor") is not None:
                    r_factors.append(float(error["r_factor"]))
                if error.get("x_factor") is not None:
                    x_factors.append(float(error["x_factor"]))
            ranking = audit.get("parameter_ranking") if isinstance(audit.get("parameter_ranking"), Mapping) else {}
            if family in ("parameter", "measurement+parameter"):
                strata[str(ranking.get("stratum"))] += 1
                if ranking.get("dominance_ratio") is not None:
                    dominance.append(float(ranking["dominance_ratio"]))
                ranks[str(ranking.get("true_line_rank"))] += 1
            for error in truth.get("true_topology_errors") or []:
                effects[str(error.get("physical_effect"))] += 1
                breakers[str(error.get("cb_name"))] += 1
                if error.get("load_scale") is not None:
                    load_scales.append(float(error["load_scale"]))
                operating_points[str(error.get("operating_point"))] += 1
            scans = meta.get("parameter_scans")
            if scans:
                scans_present += 1
                if isinstance(scans, Mapping):
                    lists = [v for v in scans.values() if isinstance(v, list) and v and isinstance(v[0], (list, Mapping))]
                    if lists:
                        scans_per_root.append(max(len(v) for v in lists))
            if meta.get("operator_layout") is not None or meta.get("substation_telemetry") is not None or meta.get("node_breaker") is not None:
                telemetry_present += 1
        result["families"][family] = {
            "roots": len(members),
            "error_cardinality": dict(sorted(cardinality.items())),
            "meter_errors_per_root": dict(sorted(meters_per_root.items())),
            "meter_channels": dict(sorted(channels.items())),
            "meter_error_pu": _spread(meter_pu),
            "meter_error_sigma": _spread(meter_sigma),
            "r_factor": _spread(r_factors),
            "x_factor": _spread(x_factors),
            "r_factor_unchanged": sum(1 for v in r_factors if abs(v - 1.0) < 1e-9),
            "x_factor_unchanged": sum(1 for v in x_factors if abs(v - 1.0) < 1e-9),
            "strata": dict(sorted(strata.items())),
            "dominance_ratio": _spread(dominance),
            "true_line_rank": dict(sorted(ranks.items())),
            "topology_effects": dict(sorted(effects.items())),
            "breakers": dict(sorted(breakers.items())),
            "load_scale": _spread(load_scales),
            "operating_points": dict(sorted(operating_points.items())),
            "roots_with_parameter_scans": scans_present,
            "scans_per_root": _spread(scans_per_root),
            "roots_with_substation_telemetry": telemetry_present,
        }
    return result


def _fmt_spread(block: Mapping[str, Any], digits: int = 2) -> str:
    if not block or not block.get("n"):
        return "-"
    return f"{block['min']:.{digits}f} / {block['median']:.{digits}f} / {block['max']:.{digits}f} (n={block['n']})"


def markdown(result: Mapping[str, Any]) -> str:
    labels = list(result)
    families = []
    for label in labels:
        for family in result[label]["families"]:
            if family not in families:
                families.append(family)
    lines = []
    for family in families:
        lines.append(f"### {family}")
        lines.append("| | " + " | ".join(labels) + " |")
        lines.append("| --- |" + " --- |" * len(labels))
        rows = (
            ("roots", lambda b: str(b["roots"])),
            ("error cardinality", lambda b: json.dumps(b["error_cardinality"])),
            ("meter errors per root", lambda b: json.dumps(b["meter_errors_per_root"])),
            ("meter channels", lambda b: json.dumps(b["meter_channels"])),
            ("meter error, pu (min / median / max)", lambda b: _fmt_spread(b["meter_error_pu"], 3)),
            ("meter error, sigma (min / median / max)", lambda b: _fmt_spread(b["meter_error_sigma"], 1)),
            ("R factor (min / median / max)", lambda b: _fmt_spread(b["r_factor"])),
            ("X factor (min / median / max)", lambda b: _fmt_spread(b["x_factor"])),
            ("R unchanged / X unchanged", lambda b: f"{b['r_factor_unchanged']} / {b['x_factor_unchanged']}"),
            ("ranking strata", lambda b: json.dumps(b["strata"])),
            ("dominance ratio (min / median / max)", lambda b: _fmt_spread(b["dominance_ratio"])),
            ("true line rank", lambda b: json.dumps(b["true_line_rank"])),
            ("topology effects", lambda b: json.dumps(b["topology_effects"])),
            ("load scale (min / median / max)", lambda b: _fmt_spread(b["load_scale"])),
            ("operating points", lambda b: json.dumps(b["operating_points"])),
            ("roots with parameter scans; scans per root", lambda b: f"{b['roots_with_parameter_scans']}; {_fmt_spread(b['scans_per_root'], 0)}"),
            ("roots with substation telemetry", lambda b: str(b["roots_with_substation_telemetry"])),
        )
        for name, fmt in rows:
            cells = []
            for label in labels:
                block = result[label]["families"].get(family)
                cells.append(fmt(block) if block else "-")
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", action="append", required=True, help="LABEL=SUITE.json")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    result: dict[str, Any] = {}
    for value in args.suite:
        label, sep, path = str(value).partition("=")
        if not sep or not label or not path:
            raise SystemExit(f"--suite takes LABEL=SUITE.json, got {value!r}")
        rows = json.loads(Path(path).read_text(encoding="utf-8"))
        result[label] = {"path": str(path), **describe_rows(rows)}
    text = markdown(result)
    print(text, end="")
    if args.output is not None:
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        args.output.with_suffix(".md").write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
