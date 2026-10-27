"""Tables of the compound-error study note from the suite, the rollout, the audit and the identifiability probes.

    python -m research.compound_errors.report --suite-dir output/compound_errors_20261027 \\
        --rollout expert_baseline_compare [--audit audit] [--identifiability identifiability] --output report

Per class: the suite's composition (strata, solo-detectable atoms, meters on
faulted elements), the expert's success (on every root and on the roots whose
structural atoms each alarm alone), its episode length and solver cost, the
verdicts of its corrections, the audit bins and the truth probe, the failure
reasons.  Two post-hoc analyses of the rollout records: the verifier's
partial-progress floors against the number of faults still open when a true
correction is verified (the scaled-floor rule), and the order in which the
expert attacks parameter-plus-breaker roots (which family it corrects first
and what that costs).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from psse_env.providers.scenario_generator import DEFAULT_COMPOUND_CLASSES  # noqa: E402

CLASS_ORDER = [spec.name for spec in DEFAULT_COMPOUND_CLASSES]
CLASS_LABELS = {
    "rx2_adjacent": "two R/X lines, adjacent",
    "rx2_distant": "two R/X lines, distant",
    "rx_cb_same_line": "R/X + breaker, same line",
    "rx_cb_shared_bus": "R/X + breaker, shared bus",
    "rx_cb_distant": "R/X + breaker, distant",
    "meter_rx2": "meter + two R/X",
    "meter_on_rx": "meter on the faulted line + R/X",
    "rx_cb_meter": "R/X + breaker + meter",
    "cb2_dangling": "two breakers, both dangling",
    "cb2_mixed": "two breakers, dangling + split",
    "cb2_split": "two breakers, both splits",
    "control_rx": "control: one R/X line",
    "control_cb": "control: one breaker",
    "control_meter": "control: 1-3 meters",
}
BRANCH_FLOOR = 0.30
METER_FLOOR = 0.20
SOLO_DETECTABLE_RATIO = 1.0


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def pct(numerator: float, denominator: float) -> str:
    return f"{100.0 * numerator / denominator:.0f}%" if denominator else "n/a"


def fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


# --------------------------------------------------------------------------- suite view


def suite_view(suite: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for root in suite:
        compound = root["audit"]["compound"]
        name = compound["class"]
        entry = out.setdefault(name, {"n": 0, "strata": Counter(), "solo_detectable": 0, "with_solo": 0, "meters_on_fault": 0,
                                      "relations": Counter(), "cardinality": Counter(), "alarm_ratio": [], "atoms_below": 0})
        entry["n"] += 1
        entry["strata"][root["audit"]["parameter_ranking"]["stratum"]] += 1
        effects = compound.get("solo_effects") or []
        if effects:
            entry["with_solo"] += 1
            if all((e.get("chi_square_ratio") or 0.0) >= SOLO_DETECTABLE_RATIO for e in effects):
                entry["solo_detectable"] += 1
            entry["atoms_below"] += sum(1 for e in effects if (e.get("chi_square_ratio") or 0.0) < SOLO_DETECTABLE_RATIO)
        entry["meters_on_fault"] += int(compound.get("meters_on_faulted_elements") or 0)
        for relation in compound.get("relations") or []:
            entry["relations"][relation["relation"]] += 1
        entry["cardinality"][int(root["grouping"]["error_cardinality"])] += 1
        entry["alarm_ratio"].append(float(compound["detection"]["reported_model"]["chi_square_ratio"] or 0.0))
    for entry in out.values():
        ratios = sorted(entry.pop("alarm_ratio"))
        entry["alarm_ratio_median"] = ratios[len(ratios) // 2] if ratios else None
        entry["strata"] = dict(entry["strata"])
        entry["relations"] = dict(entry["relations"])
        entry["cardinality"] = dict(sorted(entry["cardinality"].items()))
    return out


def solo_detectable_ids(suite: Sequence[Mapping[str, Any]]) -> set[str]:
    ids = set()
    for root in suite:
        effects = root["audit"]["compound"].get("solo_effects") or []
        if not effects or all((e.get("chi_square_ratio") or 0.0) >= SOLO_DETECTABLE_RATIO for e in effects):
            ids.add(str(root["execution"]["scenario_id"]))
    return ids


# --------------------------------------------------------------------------- rollout views


def expert_view(episodes: Sequence[Mapping[str, Any]], detectable: set[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for episode in episodes:
        name = str(episode.get("class"))
        entry = out.setdefault(name, {"n": 0, "solved": 0, "n_detectable": 0, "solved_detectable": 0, "steps": 0, "wls": 0,
                                      "comparisons": 0, "handoff": 0, "budget": 0, "false_commit_episodes": 0, "invalid_episodes": 0,
                                      "loops": 0, "errors": 0, "verdicts": Counter(), "reasons": Counter(), "basis": Counter()})
        entry["n"] += 1
        solved = bool(episode.get("success"))
        entry["solved"] += solved
        if episode["scenario_id"] in detectable:
            entry["n_detectable"] += 1
            entry["solved_detectable"] += solved
        entry["steps"] += int(episode.get("steps") or 0)
        entry["wls"] += int(episode.get("wls_calls") or 0)
        entry["comparisons"] += int(episode.get("comparisons") or 0)
        entry["handoff"] += episode.get("terminal_outcome") == "operator_escalation"
        entry["budget"] += not episode.get("terminal") and not episode.get("error")
        entry["false_commit_episodes"] += int(episode.get("false_commit_count") or 0) > 0
        entry["invalid_episodes"] += int(episode.get("invalid_action_count") or 0) > 0
        entry["loops"] += bool(episode.get("loop_detected"))
        entry["errors"] += bool(episode.get("error"))
        entry["basis"][str(episode.get("basis"))] += 1
        if not solved:
            for reason in episode.get("audit_reasons") or []:
                entry["reasons"][reason] += 1
        for correction in episode.get("corrections") or []:
            truthful = correction.get("target_is_true")
            side = "true" if truthful else "wrong" if truthful is False else "unknown"
            verdict = "refused" if correction.get("execution_status") != "success" else str(correction.get("verdict") or "undisposed")
            entry["verdicts"][(side, verdict)] += 1
            if correction.get("execution_status") != "success":
                entry.setdefault("refusals", Counter())[f"{correction['tool']}:{side}:{correction.get('error_code')}"] += 1
    for entry in out.values():
        n = max(1, entry["n"])
        entry["mean_steps"] = entry["steps"] / n
        entry["mean_wls"] = entry["wls"] / n
        entry["verdicts"] = {f"{side}:{verdict}": count for (side, verdict), count in sorted(entry["verdicts"].items())}
        entry["reasons"] = dict(entry["reasons"].most_common(6))
        entry["basis"] = dict(entry["basis"])
        entry["refusals"] = dict(entry.get("refusals", Counter()).most_common(6))
    return out


def floor_analysis(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """True-target corrections rejected below the partial floor, by open faults, and the scaled-floor rule."""
    rows: list[dict[str, Any]] = []
    for episode in episodes:
        cardinality = int(episode.get("error_cardinality") or 0)
        accepted = 0
        for correction in episode.get("corrections") or []:
            verification = correction.get("verification") or {}
            progress = verification.get("global_progress")
            if correction.get("execution_status") != "success" or progress is None:
                continue
            open_faults = max(1, cardinality - accepted)
            family = "measurement" if correction["tool"] == "correct_measurements" else "branch"
            floor = METER_FLOOR if family == "measurement" else BRANCH_FLOOR
            rows.append({
                "class": episode.get("class"), "tool": correction["tool"], "family": family, "true": correction.get("target_is_true"),
                "verdict": correction.get("verdict"), "global_progress": float(progress),
                "target_progress": verification.get("target_progress"), "resolved": verification.get("globally_resolved"),
                "open_faults": open_faults, "floor": floor, "scaled_floor": floor / open_faults,
            })
            if correction.get("verdict") in {"ACCEPT_FINAL", "ACCEPT_PARTIAL"}:
                accepted += 1
    by_open: dict[int, Counter] = defaultdict(Counter)
    for row in rows:
        key = int(row["open_faults"])
        if row["true"]:
            by_open[key]["true_verified"] += 1
            by_open[key]["true_rejected"] += row["verdict"] == "REJECT"
            below = row["global_progress"] < row["floor"] and row["resolved"] is not True
            by_open[key]["true_rejected_below_floor"] += (row["verdict"] == "REJECT" and below)
            by_open[key]["true_rejected_below_floor_would_pass_scaled"] += (
                row["verdict"] == "REJECT" and below and row["global_progress"] >= row["scaled_floor"]
            )
        elif row["true"] is False:
            by_open[key]["wrong_verified"] += 1
            by_open[key]["wrong_rejected"] += row["verdict"] == "REJECT"
            by_open[key]["wrong_accepted"] += row["verdict"] in {"ACCEPT_FINAL", "ACCEPT_PARTIAL"}
            by_open[key]["wrong_rejected_would_pass_scaled"] += (
                row["verdict"] == "REJECT" and row["resolved"] is not True and row["global_progress"] >= row["scaled_floor"]
                and (row["target_progress"] or 0.0) >= 0.5
            )
    progress_true = defaultdict(list)
    for row in rows:
        if row["true"]:
            progress_true[int(row["open_faults"])].append(row["global_progress"])
    quantiles = {}
    for key, values in progress_true.items():
        values = sorted(values)
        quantiles[key] = {"n": len(values), "p10": values[len(values) // 10], "median": values[len(values) // 2],
                          "p90": values[min(len(values) - 1, (9 * len(values)) // 10)]}
    return {"rows": len(rows), "by_open_faults": {str(k): dict(v) for k, v in sorted(by_open.items())},
            "true_progress_quantiles": {str(k): v for k, v in sorted(quantiles.items())}}


def ordering_analysis(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Parameter-plus-breaker roots: which family the expert corrects first, and the outcome by that choice."""
    out: dict[str, Any] = {}
    for episode in episodes:
        name = str(episode.get("class"))
        if not (name.startswith("rx_cb") or name.startswith("cb2") or name == "control_cb"):
            continue
        entry = out.setdefault(name, {"n": 0, "first": Counter(), "solved_by_first": Counter(), "parameter_before_breaker": 0,
                                      "parameter_attempts_before_breaker": 0, "parameter_refused_before_breaker": 0,
                                      "wrong_parameter_accepted": 0, "breaker_corrections": Counter(), "solved": 0})
        entry["n"] += 1
        entry["solved"] += bool(episode.get("success"))
        corrections = [c for c in episode.get("corrections") or []]
        first = next((c["tool"] for c in corrections), None)
        label = {"correct_parameters": "parameter", "correct_topology": "breaker", "correct_measurements": "meter", None: "none"}.get(first, first)
        entry["first"][label] += 1
        entry["solved_by_first"][label] += bool(episode.get("success"))
        breaker_index = next((i for i, c in enumerate(corrections) if c["tool"] == "correct_topology"), None)
        before = corrections[:breaker_index] if breaker_index is not None else corrections
        parameter_before = [c for c in before if c["tool"] == "correct_parameters"]
        entry["parameter_before_breaker"] += bool(parameter_before)
        entry["parameter_attempts_before_breaker"] += len(parameter_before)
        entry["parameter_refused_before_breaker"] += sum(1 for c in parameter_before if c.get("execution_status") != "success")
        entry["wrong_parameter_accepted"] += sum(1 for c in corrections if c["tool"] == "correct_parameters" and c.get("target_is_true") is False
                                                 and c.get("verdict") in {"ACCEPT_FINAL", "ACCEPT_PARTIAL"})
        for c in corrections:
            if c["tool"] == "correct_topology":
                verdict = "refused" if c.get("execution_status") != "success" else str(c.get("verdict") or "undisposed")
                entry["breaker_corrections"][f"{'true' if c.get('target_is_true') else 'wrong'}:{verdict}"] += 1
    for entry in out.values():
        entry["first"] = dict(entry["first"])
        entry["solved_by_first"] = dict(entry["solved_by_first"])
        entry["breaker_corrections"] = dict(sorted(entry["breaker_corrections"].items()))
    return out


# --------------------------------------------------------------------------- audit view


def audit_view(summary: Mapping[str, Any] | None, roots: Sequence[Mapping[str, Any]] | None) -> dict[str, dict[str, Any]]:
    if not summary:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for stratum, entry in (summary.get("strata") or {}).items():
        name = str(stratum).removeprefix("compound:")
        out[name] = {"roots": entry.get("roots"), "expert_solves": entry.get("reference_success"), "bins": dict(entry.get("bins") or {}),
                     "decision_states": entry.get("decision_states"), "alternatives": entry.get("alternatives"),
                     "alternatives_better": entry.get("alternatives_better"), "truth_tried": entry.get("truth_tried"),
                     "truth_success": entry.get("truth_success"), "errors": entry.get("errors")}
    for name, probe in (summary.get("truth_probe") or {}).items():
        out.setdefault(name, {})["truth_probe"] = {k: probe.get(k) for k in ("probes", "solved", "reasons", "refused", "rejected_true")}
    if roots:
        failed_bins: dict[str, Counter] = defaultdict(Counter)
        for result in roots:
            name = str(result.get("stratum") or "").removeprefix("compound:")
            if not (result.get("reference") or {}).get("success"):
                failed_bins[name][str(result.get("bin"))] += 1
        for name, bins in failed_bins.items():
            out.setdefault(name, {})["failed_root_bins"] = dict(bins)
    return out


# --------------------------------------------------------------------------- markdown


def markdown(report: Mapping[str, Any]) -> str:
    lines: list[str] = []
    suite = report["suite"]
    expert = report["expert"]
    audit = report.get("audit") or {}
    classes = [name for name in CLASS_ORDER if name in suite]

    lines += ["## Suite composition", "",
              "| class | n | cardinality | strata (dominant / ambiguous / misranked / unranked) | atoms each alarm alone | meters on a faulted element | alarm ratio (median) |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for name in classes:
        s = suite[name]
        strata = s["strata"]
        strata_text = " / ".join(str(strata.get(k, 0)) for k in ("dominant", "ambiguous", "misranked", "unranked")) if any(
            k in strata for k in ("dominant", "ambiguous", "misranked", "unranked")) else "n/a"
        solo = f"{s['solo_detectable']} of {s['with_solo']}" if s["with_solo"] else "n/a"
        lines.append(f"| {CLASS_LABELS.get(name, name)} | {s['n']} | {', '.join(f'{k}: {v}' for k, v in s['cardinality'].items())} | {strata_text} | "
                     f"{solo} | {s['meters_on_fault']} | {fmt(s['alarm_ratio_median'], 1)} |")
    lines += ["", "## Expert per class", "",
              "| class | n | solved | solved, atoms each alarm alone | mean actions | mean WLS | comparisons | true corrections final / partial / rejected / refused | wrong accepted | handoff | budget | false-commit ep. |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name in classes:
        e = expert.get(name)
        if not e:
            continue
        v = e["verdicts"]
        wrong_accepted = v.get("wrong:ACCEPT_FINAL", 0) + v.get("wrong:ACCEPT_PARTIAL", 0)
        lines.append(f"| {CLASS_LABELS.get(name, name)} | {e['n']} | {e['solved']} ({pct(e['solved'], e['n'])}) | "
                     f"{e['solved_detectable']} of {e['n_detectable']} ({pct(e['solved_detectable'], e['n_detectable'])}) | {e['mean_steps']:.1f} | "
                     f"{e['mean_wls']:.2f} | {e['comparisons']} | {v.get('true:ACCEPT_FINAL', 0)} / {v.get('true:ACCEPT_PARTIAL', 0)} / "
                     f"{v.get('true:REJECT', 0)} / {v.get('true:refused', 0)} | {wrong_accepted} | {e['handoff']} | {e['budget']} | {e['false_commit_episodes']} |")
    totals = {k: sum(e[k] for e in expert.values()) for k in ("n", "solved", "n_detectable", "solved_detectable", "steps", "wls", "comparisons")}
    lines.append(f"| all | {totals['n']} | {totals['solved']} ({pct(totals['solved'], totals['n'])}) | "
                 f"{totals['solved_detectable']} of {totals['n_detectable']} ({pct(totals['solved_detectable'], totals['n_detectable'])}) | "
                 f"{totals['steps'] / max(1, totals['n']):.1f} | {totals['wls'] / max(1, totals['n']):.2f} | {totals['comparisons']} | | | | | |")
    lines += ["", "## Failure reasons (truth audit) per class", ""]
    for name in classes:
        e = expert.get(name)
        if not e or not e["reasons"]:
            continue
        lines.append(f"- {CLASS_LABELS.get(name, name)} ({e['n'] - e['solved']} failed): " + "; ".join(f"{k} {v}" for k, v in e["reasons"].items()))
    lines += ["", "## Refused corrections (executor or process gate) per class", ""]
    for name in classes:
        e = expert.get(name)
        if not e or not e.get("refusals"):
            continue
        lines.append(f"- {CLASS_LABELS.get(name, name)}: " + "; ".join(f"{k} {v}" for k, v in e["refusals"].items()))
    if audit:
        lines += ["", "## Decision audit per class", "",
                  "| class | audited | expert solves | rule_best | legal_alternative | interface_limit | evidence_limit | failed roots: bins | truth probe solves | probe reasons |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for name in classes:
            a = audit.get(name)
            if not a:
                continue
            bins = a.get("bins") or {}
            probe = a.get("truth_probe") or {}
            failed = a.get("failed_root_bins") or {}
            reasons = "; ".join(f"{k} {v}" for k, v in list((probe.get("reasons") or {}).items())[:3])
            lines.append(f"| {CLASS_LABELS.get(name, name)} | {a.get('roots')} | {a.get('expert_solves')} | {bins.get('rule_best', 0)} | "
                         f"{bins.get('legal_alternative', 0)} | {bins.get('interface_limit', 0)} | {bins.get('evidence_limit', 0)} | "
                         f"{', '.join(f'{k} {v}' for k, v in sorted(failed.items()))} | {probe.get('solved')}/{probe.get('probes')} | {reasons} |")
    floors = report.get("floors") or {}
    if floors:
        lines += ["", "## Partial-progress floors against open faults", "",
                  "| open faults when verified | true corrections verified | rejected | rejected below the floor | of those, pass floor/k | true global progress p10 / median / p90 | wrong corrections verified | wrong accepted | wrong rejected that floor/k would admit |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for key, entry in floors["by_open_faults"].items():
            q = floors["true_progress_quantiles"].get(key) or {}
            lines.append(f"| {key} | {entry.get('true_verified', 0)} | {entry.get('true_rejected', 0)} | {entry.get('true_rejected_below_floor', 0)} | "
                         f"{entry.get('true_rejected_below_floor_would_pass_scaled', 0)} | {fmt(q.get('p10'))} / {fmt(q.get('median'))} / {fmt(q.get('p90'))} | "
                         f"{entry.get('wrong_verified', 0)} | {entry.get('wrong_accepted', 0)} | {entry.get('wrong_rejected_would_pass_scaled', 0)} |")
    ordering = report.get("ordering") or {}
    if ordering:
        lines += ["", "## First correction on breaker-bearing roots", "",
                  "| class | n | solved | first correction: parameter / breaker / meter / none | solved when parameter first / breaker first | roots with a parameter attempt before the breaker | parameter attempts (refused) before the breaker | wrong parameter corrections accepted | true breaker corrections final / partial / rejected / refused |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for name in classes:
            o = ordering.get(name)
            if not o:
                continue
            f, s = o["first"], o["solved_by_first"]
            b = o["breaker_corrections"]
            lines.append(f"| {CLASS_LABELS.get(name, name)} | {o['n']} | {o['solved']} | {f.get('parameter', 0)} / {f.get('breaker', 0)} / {f.get('meter', 0)} / {f.get('none', 0)} | "
                         f"{s.get('parameter', 0)} of {f.get('parameter', 0)} / {s.get('breaker', 0)} of {f.get('breaker', 0)} | {o['parameter_before_breaker']} | "
                         f"{o['parameter_attempts_before_breaker']} ({o['parameter_refused_before_breaker']}) | {o['wrong_parameter_accepted']} | "
                         f"{b.get('true:ACCEPT_FINAL', 0)} / {b.get('true:ACCEPT_PARTIAL', 0)} / {b.get('true:REJECT', 0)} / {b.get('true:refused', 0)} |")
    identifiability = report.get("identifiability") or {}
    if identifiability:
        lines += ["", "## Identifiability probes", "", "```json", json.dumps(identifiability, indent=1, default=str)[:12000], "```"]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite-dir", required=True, type=Path)
    parser.add_argument("--rollout", default="expert_baseline_compare", help="rollout directory under the suite dir")
    parser.add_argument("--audit", default=None, help="audit directory under the suite dir")
    parser.add_argument("--identifiability", default=None, help="identifiability directory under the suite dir")
    parser.add_argument("--output", default="report", help="output directory under the suite dir")
    args = parser.parse_args(argv)

    suite = json.loads((args.suite_dir / "suite.json").read_text(encoding="utf-8"))
    episodes = load_jsonl(args.suite_dir / args.rollout / "episodes.jsonl")
    detectable = solo_detectable_ids(suite)
    report: dict[str, Any] = {
        "suite": suite_view(suite),
        "expert": expert_view(episodes, detectable),
        "floors": floor_analysis(episodes),
        "ordering": ordering_analysis(episodes),
        "solo_detectable_ratio": SOLO_DETECTABLE_RATIO,
    }
    if args.audit:
        audit_dir = args.suite_dir / args.audit
        summary = json.loads((audit_dir / "summary.json").read_text(encoding="utf-8")) if (audit_dir / "summary.json").is_file() else None
        roots = load_jsonl(audit_dir / "roots.jsonl") if (audit_dir / "roots.jsonl").is_file() else None
        report["audit"] = audit_view(summary, roots)
    if args.identifiability:
        path = args.suite_dir / args.identifiability / "summary.json"
        if path.is_file():
            report["identifiability"] = json.loads(path.read_text(encoding="utf-8"))
    out = args.suite_dir / args.output
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    text = markdown(report)
    (out / "report.md").write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
