"""Step 1 identifiability report from the offline hypothesis dataset.

Reads the JSONL files written by ``build_dataset`` and answers, per family:
what the balanced screen's final explanation is against the truth; how often
``unexplained`` (no single balanced cause clears the alarm) fires on the
families that have no balanced route, and how often it fires wrongly on
single-cause roots and alarmed healthy windows; how well the screen's own
target rankings recover the true meter, branch or line against the plain WLS
residual and multiplier rankings; whether the two-round outcome names both
components of a mixed root; what the screen says on truth-corrected child
states; and the HIF mimic rate of two biased flow meters.

    python -m research.hypothesis_ranking.report --dataset-dir output/hypothesis_ranking_20260930/ieee14
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SINGLE_CAUSE = ("measurement", "parameter", "topology", "hif")
NO_ROUTE = ("three_phase_unbalance", "harmonic")
MIXED = ("measurement+parameter", "measurement+topology", "measurement+hif")
FAMILY_ORDER = (
    "no_error", "measurement", "multi_measurement", "parameter", "measurement+parameter", "topology",
    "measurement+topology", "harmonic", "hif", "measurement+hif", "three_phase_unbalance",
)
CLASSES = ("meter", "parameter", "topology", "hif", "unexplained")
NEEDS_NO_PHASORS = ("measurement", "multi_measurement", "parameter", "measurement+parameter", "topology",
                    "measurement+topology", "healthy_window", "mimic_flow_pair_same_sign", "mimic_flow_pair_opposite_sign")


def _read(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100.0 * value:.1f}%"


def screen_of(row: Mapping[str, Any]) -> Mapping[str, Any] | None:
    analysis = row.get("analysis") or {}
    screen = analysis.get("screen")
    return screen if isinstance(screen, Mapping) else None


def alarmed(row: Mapping[str, Any]) -> bool:
    return bool(((row.get("analysis") or {}).get("wls") or {}).get("alarm"))


def final_class(row: Mapping[str, Any]) -> str:
    """The screen's explanation: its first accepted class, ``hif``, ``unexplained``, or why there is none.

    A v2 screen names every accepted hypothesis in ``outcome``; the first one
    is the primary explanation (``parameter>meter`` reads as a parameter root
    with a meter beside it).  A v1 report has no ``explained`` flag and keeps
    its last-round winner.
    """
    if not alarmed(row):
        return "no_alarm"
    screen = screen_of(row)
    if screen is None:
        return "screen_missing"
    if screen.get("status") != "valid":
        return f"screen_{screen.get('status')}"
    if screen.get("suspected"):
        return "hif"
    if screen.get("unexplained"):
        return "unexplained"
    outcome = str(screen.get("outcome") or "")
    if "explained" in screen:
        return outcome.split(">")[0] if outcome else "none"
    final = screen.get("final") or {}
    if final.get("explained_by_meter_removal"):
        return "meter"
    return str(final.get("winner") or outcome or "none").split(">")[-1]


def outcome_of(row: Mapping[str, Any]) -> str:
    screen = screen_of(row) or {}
    return str(screen.get("outcome") or "none") if alarmed(row) else "no_alarm"


def winner_class(row: Mapping[str, Any]) -> str:
    """The production winner (last round), ignoring the alarm test."""
    if not alarmed(row):
        return "no_alarm"
    screen = screen_of(row)
    if screen is None or screen.get("status") != "valid":
        return "invalid"
    outcome = screen.get("outcome")
    return str(outcome).split(">")[-1] if outcome else "none"


UNEXPLAINED_VARIANTS = ("production", "first_round_single_cause", "with_one_more_meter")


def unexplained_variant(row: Mapping[str, Any], variant: str = "single_cause") -> bool:
    """``unexplained`` under one definition: single balanced causes only, or also the
    offline joint R/X refit, the winner plus one more meter, or either."""
    screen = screen_of(row)
    if not screen or screen.get("status") != "valid":
        return False
    variants = screen.get("unexplained_variants")
    if isinstance(variants, Mapping) and variant in variants:
        return bool(variants[variant])
    return bool(screen.get("unexplained"))


def suspicion_v2(row: Mapping[str, Any], variant: str = "production") -> bool:
    """Acquisition rule of option A: HIF suspected or unexplained (under ``variant``)."""
    screen = screen_of(row)
    return bool(screen and screen.get("status") == "valid" and (screen.get("suspected") or unexplained_variant(row, variant)))


def voltage_meter_suspicion(row: Mapping[str, Any]) -> bool:
    screen = screen_of(row) or {}
    kinds = screen.get("phasor_suspicion") if isinstance(screen.get("phasor_suspicion"), Mapping) else {}
    return bool(screen.get("status") == "valid" and (kinds.get("voltage_meter") or screen.get("voltage_meter_channels")))


def suspicion_v3(row: Mapping[str, Any]) -> bool:
    """The 2026-09-30 rule: HIF suspected, unexplained, or a voltage-meter hypothesis (D3)."""
    return suspicion_v2(row) or voltage_meter_suspicion(row)


def _rank_of(target: int, ranked: Iterable[Mapping[str, Any]], key: str) -> int | None:
    for position, item in enumerate(ranked, start=1):
        if int(item.get(key, -1)) == int(target):
            return position
    return None


def _final_round(screen: Mapping[str, Any]) -> Mapping[str, Any]:
    rounds = [r for r in screen.get("rounds") or [] if r.get("winner") is not None]
    return rounds[-1] if rounds else {}


def _confusion(rows: Sequence[Mapping[str, Any]], classifier) -> dict[str, dict[str, int]]:
    table: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        table[row["family"]][classifier(row)] += 1
    return {family: dict(counter) for family, counter in table.items()}


def _table(title: str, header: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = [f"### {title}", "", "| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines) + "\n"


def target_recall(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Screen-ranked versus WLS-ranked recovery of the true meter, branch or line."""
    out: dict[str, Any] = {}
    # single meter roots: true index in the meter ranking / the residual ranking
    meters = [r for r in rows if r["family"] == "measurement" and alarmed(r) and screen_of(r) and screen_of(r).get("status") == "valid"]
    hits = Counter()
    for row in meters:
        truth = row["truth"]["measurement"][0]["index"]
        screen = screen_of(row)
        first = (screen.get("rounds") or [{}])[0]
        ranked = ((first.get("best") or {}).get("meter") or {}).get("ranked") or []
        rank = _rank_of(truth, ranked, "channel_index0")
        residual_rank = _rank_of(truth, row["analysis"]["wls"]["top_residuals"], "index")
        hits["screen_top1"] += rank == 1
        hits["screen_top3"] += rank is not None and rank <= 3
        hits["residual_top1"] += residual_rank == 1
        hits["residual_top3"] += residual_rank is not None and residual_rank <= 3
    out["measurement"] = {"n": len(meters), **{k: _rate(v, len(meters)) for k, v in hits.items()}}
    # parameter roots: true branch in the parameter ranking / the multiplier ranking, by stratum
    params = [r for r in rows if r["family"] == "parameter" and alarmed(r) and screen_of(r) and screen_of(r).get("status") == "valid"]
    strata: dict[str, Counter] = defaultdict(Counter)
    for row in params:
        truth = int(row["truth"]["parameter"][0]["branch_row0"])
        wls = row["analysis"]["wls"]
        ranking = wls.get("branch_multiplier_ranking") or []
        ratio = wls.get("branch_ranking_dominance_ratio")
        if ranking and ranking[0] == truth:
            stratum = "dominant" if (ratio is None or ratio >= 1.2) else "ambiguous"
        else:
            stratum = "misranked"
        screen = screen_of(row)
        best = (_final_round(screen).get("best") or {}).get("parameter") or {}
        rank = _rank_of(truth, best.get("ranked") or [], "branch_row0")
        lambda_rank = (ranking.index(truth) + 1) if truth in ranking else None
        counter = strata[stratum]
        counter["n"] += 1
        counter["screen_top1"] += rank == 1
        counter["screen_top2"] += rank is not None and rank <= 2
        counter["lambda_top1"] += lambda_rank == 1
        counter["lambda_top2"] += lambda_rank is not None and lambda_rank <= 2
        counter["screen_class_parameter"] += final_class(row) == "parameter"
    out["parameter_by_stratum"] = {
        stratum: {"n": c["n"], **{k: _rate(c[k], c["n"]) for k in ("screen_top1", "screen_top2", "lambda_top1", "lambda_top2", "screen_class_parameter")}}
        for stratum, c in strata.items()
    }
    # topology roots (dangling terminal only carry a line): true line in the outage ranking
    topo = [r for r in rows if r["family"] == "topology" and alarmed(r) and screen_of(r) and screen_of(r).get("status") == "valid"]
    counter = Counter()
    for row in topo:
        truth = row["truth"]["topology"][0] if row["truth"]["topology"] else {}
        if truth.get("branch_row0") is None:
            counter["bus_split_or_no_line"] += 1
            continue
        counter["n_line"] += 1
        screen = screen_of(row)
        best = (_final_round(screen).get("best") or {}).get("topology") or {}
        rank = _rank_of(int(truth["branch_row0"]), best.get("ranked") or [], "branch_row0")
        counter["screen_top1"] += rank == 1
        counter["screen_top2"] += rank is not None and rank <= 2
        counter["screen_class_topology"] += final_class(row) == "topology"
    out["topology"] = {"n": len(topo), "bus_split_or_no_line": counter["bus_split_or_no_line"], "n_line": counter["n_line"],
                       **{k: _rate(counter[k], counter["n_line"]) for k in ("screen_top1", "screen_top2", "screen_class_topology")}}
    # hif roots: true line is the flagged line
    hifs = [r for r in rows if r["family"] == "hif" and alarmed(r) and screen_of(r) and screen_of(r).get("status") == "valid"]
    counter = Counter()
    for row in hifs:
        truth = int(row["truth"]["hif"][0]["branch_row0"])
        screen = screen_of(row)
        counter["suspected"] += bool(screen.get("suspected"))
        counter["suspected_right_line"] += bool(screen.get("suspected")) and int(screen.get("branch_row0", -1)) == truth
        best = (_final_round(screen).get("best") or {}).get("hif") or {}
        rank = _rank_of(truth, best.get("ranked") or [], "branch_row0")
        counter["hif_rank_top1"] += rank == 1
        counter["hif_rank_top2"] += rank is not None and rank <= 2
    out["hif"] = {"n": len(hifs), **{k: _rate(counter[k], len(hifs)) for k in ("suspected", "suspected_right_line", "hif_rank_top1", "hif_rank_top2")}}
    return out


def mixed_coverage(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Does the two-round outcome name both components of a mixed root?"""
    out: dict[str, Any] = {}
    for family in MIXED:
        subset = [r for r in rows if r["family"] == family and alarmed(r) and screen_of(r) and screen_of(r).get("status") == "valid"]
        counter = Counter()
        for row in subset:
            screen = screen_of(row)
            meter_truth = {m["index"] for m in row["truth"]["measurement"]}
            physical_key = {"measurement+parameter": "parameter", "measurement+topology": "topology", "measurement+hif": "hif"}[family]
            physical_truth = row["truth"][physical_key][0] if row["truth"][physical_key] else {}
            physical_line = physical_truth.get("branch_row0")
            meter_named = False
            physical_named = False
            for round_ in screen.get("rounds") or []:
                winner = round_.get("winner")
                best = (round_.get("best") or {}).get(winner or "", {})
                if winner == "meter" and int(best.get("channel_index0", -1)) in meter_truth:
                    meter_named = True
                if winner == physical_key and physical_line is not None and int(best.get("branch_row0", -2)) == int(physical_line):
                    physical_named = True
                if winner == physical_key and physical_line is None and physical_key == "topology":
                    physical_named = True  # bus split: the class is right, no line to match
            key = {(True, True): "both_named", (True, False): "meter_only", (False, True): "physical_only", (False, False): "neither"}[(meter_named, physical_named)]
            counter[key] += 1
            counter["unexplained"] += final_class(row) == "unexplained"
            counter["final_" + final_class(row)] += 1
        out[family] = {"n": len(subset), **{k: v for k, v in counter.items()}}
    return out


def children_summary(children: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    groups: dict[tuple[str, str], list] = defaultdict(list)
    for row in children:
        groups[(row["family"], row["child_kind"])].append(row)
    for (family, kind), subset in sorted(groups.items()):
        counter = Counter()
        for row in subset:
            counter["n"] += 1
            counter["alarmed"] += alarmed(row)
            counter["final_" + final_class(row)] += 1
            remaining = row["truth"]
            if remaining.get("parameter") and alarmed(row):
                truth = int(remaining["parameter"][0]["branch_row0"])
                screen = screen_of(row) or {}
                best = (_final_round(screen).get("best") or {}).get("parameter") or {}
                counter["parameter_named"] += final_class(row) == "parameter" and int(best.get("branch_row0", -1)) == truth
            if remaining.get("measurement") and alarmed(row):
                truth_set = {m["index"] for m in remaining["measurement"]}
                screen = screen_of(row) or {}
                first = (screen.get("rounds") or [{}])[0]
                best = (first.get("best") or {}).get("meter") or {}
                counter["meter_named"] += final_class(row) == "meter" and int(best.get("channel_index0", -1)) in truth_set
            if remaining.get("hif") and alarmed(row):
                screen = screen_of(row) or {}
                counter["hif_suspected_right_line"] += bool(screen.get("suspected")) and int(screen.get("branch_row0", -1)) == int(remaining["hif"][0]["branch_row0"])
        out[f"{family}::{kind}"] = dict(counter)
    return out


def build_report(dataset_dir: Path) -> tuple[dict[str, Any], str]:
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8")) if (dataset_dir / "manifest.json").is_file() else {}
    roots = _read(dataset_dir / "roots.jsonl")
    children = _read(dataset_dir / "children.jsonl")
    healthy = _read(dataset_dir / "healthy_alarms.jsonl")
    mimic = _read(dataset_dir / "mimic.jsonl")
    families = [f for f in FAMILY_ORDER if any(r["family"] == f for r in roots)]

    failures = Counter()
    for row in (*roots, *children, *healthy, *mimic):
        analysis = row.get("analysis") or {}
        if analysis.get("error"):
            failures["analysis_error"] += 1
        elif (analysis.get("wls") or {}).get("success") is False:
            failures["wls_failure"] += 1
        elif alarmed(row) and (screen_of(row) or {}).get("status") not in (None, "valid"):
            failures["screen_" + str((screen_of(row) or {}).get("status"))] += 1

    alarm_rate = {f: _rate(sum(alarmed(r) for r in roots if r["family"] == f), sum(r["family"] == f for r in roots)) for f in families}
    confusion_final = _confusion(roots, final_class)
    confusion_winner = _confusion(roots, winner_class)

    def unexplained_rate(family_rows: Sequence[Mapping[str, Any]]) -> tuple[int, int]:
        alarmed_rows = [r for r in family_rows if alarmed(r) and (screen_of(r) or {}).get("status") == "valid"]
        return sum(final_class(r) == "unexplained" for r in alarmed_rows), len(alarmed_rows)

    capture = {}
    for family in families:
        hit, n = unexplained_rate([r for r in roots if r["family"] == family])
        capture[family] = {"unexplained": hit, "alarmed_valid": n, "rate": _rate(hit, n)}
    healthy_hit, healthy_n = unexplained_rate(healthy)
    capture["healthy_window_alarmed"] = {"unexplained": healthy_hit, "alarmed_valid": healthy_n, "rate": _rate(healthy_hit, healthy_n)}
    no_route_hit = sum(capture[f]["unexplained"] for f in NO_ROUTE if f in capture)
    no_route_n = sum(capture[f]["alarmed_valid"] for f in NO_ROUTE if f in capture)
    single_hit = sum(capture[f]["unexplained"] for f in SINGLE_CAUSE if f in capture)
    single_n = sum(capture[f]["alarmed_valid"] for f in SINGLE_CAUSE if f in capture)
    decision = {
        "no_route_capture": _rate(no_route_hit, no_route_n),
        "single_cause_false_rate": _rate(single_hit, single_n),
        "healthy_alarm_false_rate": capture["healthy_window_alarmed"]["rate"],
        "rule": "capture >= 0.80 on unbalance+harmonic and false rate < 0.05 on single-cause roots",
        "option_a_viable": bool(no_route_n and single_n and (no_route_hit / no_route_n) >= 0.80 and (single_hit / single_n) < 0.05),
    }
    # The same rates under every unexplained definition (offline extra hypotheses).
    variants: dict[str, Any] = {}
    for variant in UNEXPLAINED_VARIANTS:
        per_family = {}
        for family in [*families]:
            subset = [r for r in roots if r["family"] == family and alarmed(r) and (screen_of(r) or {}).get("status") == "valid"]
            per_family[family] = {"alarmed_valid": len(subset), "unexplained": sum(unexplained_variant(r, variant) for r in subset)}
            per_family[family]["rate"] = _rate(per_family[family]["unexplained"], len(subset))
        healthy_subset = [r for r in healthy if alarmed(r) and (screen_of(r) or {}).get("status") == "valid"]
        per_family["healthy_window_alarmed"] = {"alarmed_valid": len(healthy_subset),
                                                "unexplained": sum(unexplained_variant(r, variant) for r in healthy_subset)}
        per_family["healthy_window_alarmed"]["rate"] = _rate(per_family["healthy_window_alarmed"]["unexplained"], len(healthy_subset))
        nr_hit = sum(per_family[f]["unexplained"] for f in NO_ROUTE if f in per_family)
        nr_n = sum(per_family[f]["alarmed_valid"] for f in NO_ROUTE if f in per_family)
        sc_hit = sum(per_family[f]["unexplained"] for f in SINGLE_CAUSE if f in per_family)
        sc_n = sum(per_family[f]["alarmed_valid"] for f in SINGLE_CAUSE if f in per_family)
        variants[variant] = {
            "per_family": per_family, "no_route_capture": _rate(nr_hit, nr_n), "single_cause_false_rate": _rate(sc_hit, sc_n),
            "healthy_alarm_false_rate": per_family["healthy_window_alarmed"]["rate"],
            "option_a_viable": bool(nr_n and sc_n and (nr_hit / nr_n) >= 0.80 and (sc_hit / sc_n) < 0.05),
        }
    decision["by_variant"] = variants

    acquisition = {}

    def acquisition_row(subset):
        return {
            "n_alarmed": len(subset),
            "hif_suspected": sum(bool((screen_of(r) or {}).get("suspected")) for r in subset),
            "suspicion_v2": sum(suspicion_v2(r) for r in subset),
            "voltage_meter": sum(voltage_meter_suspicion(r) for r in subset),
            "suspicion_v3": sum(suspicion_v3(r) for r in subset),
        }

    for family in families:
        acquisition[family] = acquisition_row([r for r in roots if r["family"] == family and alarmed(r)])
    acquisition["healthy_window_alarmed"] = acquisition_row(list(healthy))
    for entry in acquisition.values():
        entry["rate_v1"] = _rate(entry["hif_suspected"], entry["n_alarmed"])
        entry["rate_v2"] = _rate(entry["suspicion_v2"], entry["n_alarmed"])
        entry["rate_v3"] = _rate(entry["suspicion_v3"], entry["n_alarmed"])
    unnecessary_v2 = sum(acquisition[f]["suspicion_v2"] for f in acquisition if f in NEEDS_NO_PHASORS)
    unnecessary_v3 = sum(acquisition[f]["suspicion_v3"] for f in acquisition if f in NEEDS_NO_PHASORS)
    unnecessary_v1 = sum(acquisition[f]["hif_suspected"] for f in acquisition if f in NEEDS_NO_PHASORS)
    unnecessary_n = sum(acquisition[f]["n_alarmed"] for f in acquisition if f in NEEDS_NO_PHASORS)

    mimic_summary = {}
    for family in ("mimic_flow_pair_same_sign", "mimic_flow_pair_opposite_sign"):
        subset = [r for r in mimic if r["family"] == family]
        alarmed_rows = [r for r in subset if alarmed(r) and (screen_of(r) or {}).get("status") == "valid"]
        right_line = sum(bool((screen_of(r) or {}).get("suspected")) and int((screen_of(r) or {}).get("branch_row0", -1)) == int(r["truth"]["mimic"]["branch_row0"]) for r in alarmed_rows)
        mimic_summary[family] = {
            "n": len(subset), "alarmed": len(alarmed_rows),
            "hif_suspected": sum(bool((screen_of(r) or {}).get("suspected")) for r in alarmed_rows),
            "hif_suspected_on_that_line": right_line,
            "unexplained": sum(final_class(r) == "unexplained" for r in alarmed_rows),
            "final_classes": dict(Counter(final_class(r) for r in alarmed_rows)),
        }
        mimic_summary[family]["hif_rate"] = _rate(mimic_summary[family]["hif_suspected"], len(alarmed_rows))

    timing = {
        "screen_seconds_mean": (sum((r.get("analysis") or {}).get("screen_seconds", 0.0) for r in roots if (r.get("analysis") or {}).get("screen_seconds"))
                                / max(1, sum(1 for r in roots if (r.get("analysis") or {}).get("screen_seconds")))),
        "wls_seconds_mean": sum((r.get("analysis") or {}).get("wls_seconds", 0.0) for r in roots) / max(1, len(roots)),
    }
    metrics = {
        "dataset_dir": str(dataset_dir), "manifest": {k: manifest.get(k) for k in ("created_utc", "git_commit", "seed", "counts", "roots_by_family", "roots_by_split", "skips")},
        "analysis_failures": dict(failures), "alarm_rate": alarm_rate,
        "confusion_final": confusion_final, "confusion_winner": confusion_winner,
        "unexplained_capture": capture, "decision": decision, "acquisition": acquisition,
        "unnecessary_acquisitions": {"n_alarmed": unnecessary_n, "v1_hif_only": unnecessary_v1, "v2_hif_or_unexplained": unnecessary_v2,
                                     "v3_hif_unexplained_or_voltage_meter": unnecessary_v3,
                                     "rate_v1": _rate(unnecessary_v1, unnecessary_n), "rate_v2": _rate(unnecessary_v2, unnecessary_n),
                                     "rate_v3": _rate(unnecessary_v3, unnecessary_n)},
        "target_recall": target_recall(roots), "mixed_coverage": mixed_coverage(roots),
        "children": children_summary(children), "mimic": mimic_summary, "timing": timing,
    }

    # ------------------------------------------------------------- markdown
    parts = [f"# Hypothesis-ranking step 1: identifiability of the balanced evidence\n",
             f"Dataset `{dataset_dir}` (commit {manifest.get('git_commit')}, seed {manifest.get('seed')}, built {manifest.get('created_utc')}). "
             f"Roots {len(roots)}, children {len(children)}, alarmed healthy windows {len(healthy)} of {manifest.get('counts', {}).get('healthy_windows_total')}, mimic roots {len(mimic)}. "
             f"Analysis failures: {dict(failures) or 'none'}. Mean screen time {timing['screen_seconds_mean']:.2f} s.\n",
             "Definitions. *Final class*: the screen's last-round winner when some single balanced cause clears the alarm rule "
             "(chi-square at the class's remaining degrees of freedom, alpha 0.01, and max normalized residual below 4), `meter` when the "
             "second-round base solve is clean after a meter is set aside, and `unexplained` when the base solve alarms and no class clears it. "
             "*Winner*: the production winner regardless of the alarm test. Suspicion v1 is the shipped rule (HIF class wins); v2 is option A (HIF wins or unexplained).\n"]
    class_columns = [*CLASSES, "no_alarm"]
    rows_md = []
    for family in families:
        counts = confusion_final.get(family, {})
        n = sum(r["family"] == family for r in roots)
        other = sum(v for k, v in counts.items() if k not in class_columns)
        rows_md.append([family, n, _pct(alarm_rate[family]), *[counts.get(c, 0) for c in class_columns], other])
    parts.append(_table("Final explanation by family (roots)", ["family", "roots", "alarm rate", *class_columns, "other"], rows_md))
    rows_md = []
    for family in families:
        counts = confusion_winner.get(family, {})
        rows_md.append([family, *[counts.get(c, 0) for c in ("meter", "parameter", "topology", "hif", "no_alarm")]])
    parts.append(_table("Last-round winner by family (alarm test ignored)", ["family", "meter", "parameter", "topology", "hif", "no_alarm"], rows_md))
    outcome_counts: dict[str, Counter] = defaultdict(Counter)
    for row in roots:
        outcome_counts[row["family"]][outcome_of(row)] += 1
    rows_md = [[family, ", ".join(f"{k} {v}" for k, v in outcome_counts[family].most_common(6))] for family in families]
    parts.append(_table("Accepted-hypothesis sequences by family (top 6)", ["family", "outcomes"], rows_md))
    variant_counts: dict[str, Counter] = defaultdict(Counter)
    for row in roots:
        screen = screen_of(row) or {}
        if screen.get("suspected"):
            variant_counts[row["family"]][str(screen.get("hif_variant") or "shunt")] += 1
    if variant_counts:
        rows_md = [[family, dict(variant_counts[family])] for family in families if variant_counts.get(family)]
        parts.append(_table("HIF suspicions by screen variant", ["family", "variants"], rows_md))
    rows_md = [[f, capture[f]["alarmed_valid"], capture[f]["unexplained"], _pct(capture[f]["rate"])] for f in [*families, "healthy_window_alarmed"] if f in capture]
    parts.append(_table("Unexplained rate (alarmed roots with a valid screen)", ["family", "alarmed", "unexplained", "rate"], rows_md))
    parts.append(f"**Decision rule** ({decision['rule']}): capture on unbalance+harmonic {_pct(decision['no_route_capture'])}, "
                 f"false rate on single-cause roots {_pct(decision['single_cause_false_rate'])}, on alarmed healthy windows {_pct(decision['healthy_alarm_false_rate'])}. "
                 f"Option A viable by this rule: **{decision['option_a_viable']}**.\n")
    variant_families = [*families, "healthy_window_alarmed"]
    rows_md = []
    for variant, summary in decision["by_variant"].items():
        rows_md.append([variant, *[_pct(summary["per_family"][f]["rate"]) if f in summary["per_family"] else "n/a" for f in variant_families],
                        _pct(summary["no_route_capture"]), _pct(summary["single_cause_false_rate"]), summary["option_a_viable"]])
    parts.append(_table("Unexplained rate under each definition (offline extra hypotheses added to the single-cause classes)",
                        ["definition", *variant_families, "no-route capture", "single-cause false", "viable"], rows_md))
    rows_md = [[f, acquisition[f]["n_alarmed"], acquisition[f]["hif_suspected"], _pct(acquisition[f]["rate_v1"]),
                acquisition[f]["suspicion_v2"], _pct(acquisition[f]["rate_v2"]), acquisition[f]["voltage_meter"],
                acquisition[f]["suspicion_v3"], _pct(acquisition[f]["rate_v3"])]
               for f in acquisition]
    parts.append(_table("Phasor acquisitions the three rules would make",
                        ["family", "alarmed", "v1 (HIF wins)", "v1 rate", "v2 (HIF or unexplained)", "v2 rate",
                         "voltage-meter picks", "v3 (v2 or voltage meter)", "v3 rate"], rows_md))
    ua = metrics["unnecessary_acquisitions"]
    parts.append(f"Acquisitions on families whose truth needs no phasors: v1 {ua['v1_hif_only']} of {ua['n_alarmed']} ({_pct(ua['rate_v1'])}), "
                 f"v2 {ua['v2_hif_or_unexplained']} ({_pct(ua['rate_v2'])}), v3 {ua['v3_hif_unexplained_or_voltage_meter']} ({_pct(ua['rate_v3'])}).\n")
    tr = metrics["target_recall"]
    m = tr["measurement"]
    rows_md = [["measurement (single meter)", m["n"], _pct(m.get("screen_top1")), _pct(m.get("screen_top3")), _pct(m.get("residual_top1")), _pct(m.get("residual_top3"))]]
    parts.append(_table("True meter in the ranking", ["roots", "n", "screen top-1", "screen top-3", "residual top-1", "residual top-3"], rows_md))
    rows_md = [[s, c["n"], _pct(c["screen_top1"]), _pct(c["screen_top2"]), _pct(c["lambda_top1"]), _pct(c["lambda_top2"]), _pct(c["screen_class_parameter"])]
               for s, c in sorted(tr["parameter_by_stratum"].items())]
    parts.append(_table("True branch in the ranking (parameter roots, by multiplier stratum)", ["stratum", "n", "screen top-1", "screen top-2", "lambda top-1", "lambda top-2", "final class parameter"], rows_md))
    t = tr["topology"]
    parts.append(f"Topology roots: {t['n']} alarmed, {t['n_line']} carry an isolated line ({t['bus_split_or_no_line']} bus splits): true line screen top-1 {_pct(t.get('screen_top1'))}, top-2 {_pct(t.get('screen_top2'))}, final class topology {_pct(t.get('screen_class_topology'))}.\n")
    h = tr["hif"]
    parts.append(f"HIF roots: {h['n']} alarmed; suspected {_pct(h.get('suspected'))}, suspected on the true line {_pct(h.get('suspected_right_line'))}, true line ranked first in the HIF class {_pct(h.get('hif_rank_top1'))}, top-2 {_pct(h.get('hif_rank_top2'))}.\n")
    rows_md = [[f, c["n"], c.get("both_named", 0), c.get("meter_only", 0), c.get("physical_only", 0), c.get("neither", 0), c.get("unexplained", 0)] for f, c in metrics["mixed_coverage"].items()]
    parts.append(_table("Mixed roots: components named by the two-round outcome", ["family", "n", "both", "meter only", "physical only", "neither", "unexplained"], rows_md))
    rows_md = []
    for key, c in metrics["children"].items():
        finals = ", ".join(f"{k[6:]} {v}" for k, v in sorted(c.items()) if k.startswith("final_"))
        rows_md.append([key, c.get("n", 0), c.get("alarmed", 0), finals, c.get("meter_named", ""), c.get("parameter_named", ""), c.get("hif_suspected_right_line", "")])
    parts.append(_table("Truth-corrected child states", ["root family :: correction", "n", "still alarmed", "final classes", "meter named", "branch named", "HIF line named"], rows_md))
    rows_md = [[f, c["n"], c["alarmed"], c["hif_suspected"], c["hif_suspected_on_that_line"], _pct(c["hif_rate"]), c["unexplained"], c["final_classes"]] for f, c in mimic_summary.items()]
    parts.append(_table("Two biased flow meters on one line (HIF mimic)", ["variant", "n", "alarmed", "HIF suspected", "on that line", "HIF rate", "unexplained", "final classes"], rows_md))
    if manifest.get("skips"):
        parts.append("### Generator skips\n\n```json\n" + json.dumps(manifest["skips"], indent=2, sort_keys=True) + "\n```\n")
    return metrics, "\n".join(parts)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", required=True)
    args = parser.parse_args(argv)
    dataset_dir = Path(args.dataset_dir)
    metrics, markdown = build_report(dataset_dir)
    (dataset_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True, default=str), encoding="utf-8")
    (dataset_dir / "report.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
