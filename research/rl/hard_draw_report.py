"""Per-stratum report of a hard draw: composition, WLS effect, identifiability tags and expert ceilings.

    python -m research.rl.hard_draw_report --suite <development.json> [--tags <identifiability/tags.jsonl>] \\
        [--expert NAME=<expert_e2e episodes.jsonl> ...] --output <report.json>

A root's stratum is its parameter-ranking stratum for the parameter families
(``parameter:misranked`` and so on) and its family otherwise.  The effect
ratio is the operator WLS's ``(J - dof) / threshold excess`` on the root's
measurements, the same scale as research.classifier_triage.effect_sampler.
The plan's decision rule (docs/rl_hard_cases_plan_20261013.md, step 1) reads
the best expert arm's overall success: above 95% the draw is not hard enough.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PARAMETER_FAMILIES = {"parameter", "measurement+parameter"}


def root_stratum(root: Mapping[str, Any]) -> str:
    mimic = (root.get("audit") or {}).get("mimic")
    if isinstance(mimic, Mapping) and mimic.get("variant"):
        return f"mimic:{str(mimic['variant']).removeprefix('mimic_')}"
    family = str(root["grouping"]["scenario_family"])
    if family in PARAMETER_FAMILIES:
        ranking = (root.get("audit") or {}).get("parameter_ranking") or {}
        return f"{family}:{ranking.get('stratum') or 'unknown'}"
    return family


def effect_ratio_of(root: Mapping[str, Any]) -> float | None:
    from research.classifier_triage.data import wls_payload
    from research.classifier_triage.effect_sampler import effect_ratio

    execution = root["execution"]
    try:
        payload = wls_payload({"case": execution["case"], "measurements": execution["measurements"],
                               "metadata": execution.get("metadata") or {}})
    except Exception:
        return None
    if payload is None:
        return None
    return float(effect_ratio(payload["objective"], payload["dof"]))


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def build_report(suite: Sequence[Mapping[str, Any]], *, tags: Sequence[Mapping[str, Any]] = (),
                 experts: Mapping[str, Sequence[Mapping[str, Any]]] | None = None, with_effect: bool = True) -> dict[str, Any]:
    experts = experts or {}
    tag_by_root = {str(t["scenario_id"]): t for t in tags}
    expert_by_root = {name: {str(e["scenario_id"]): e for e in episodes} for name, episodes in experts.items()}
    strata: dict[str, dict[str, Any]] = defaultdict(lambda: {"roots": 0, "effects": [], "tags": defaultdict(int),
                                                             "distinguishable": 0, "experts": defaultdict(lambda: {"success": 0, "steps": []})})
    for root in suite:
        root_id = str(root["execution"]["scenario_id"])
        entry = strata[root_stratum(root)]
        entry["roots"] += 1
        if with_effect:
            ratio = effect_ratio_of(root)
            if ratio is not None:
                entry["effects"].append(ratio)
        tag = tag_by_root.get(root_id)
        if tag is not None:
            entry["tags"][str(tag["tag"])] += 1
            entry["distinguishable"] += int(bool(tag.get("distinguishable_by_final_verdict")) and tag["tag"] == "not_separable")
        for name, by_root in expert_by_root.items():
            episode = by_root.get(root_id)
            if episode is None:
                continue
            expert = entry["experts"][name]
            expert["success"] += int(bool(episode.get("truth_audited_task_success")))
            expert["steps"].append(int(episode.get("steps") or 0))
    table: dict[str, Any] = {}
    totals: dict[str, dict[str, int]] = defaultdict(lambda: {"success": 0, "roots": 0, "steps": 0})
    for stratum in sorted(strata):
        entry = strata[stratum]
        effects = entry["effects"]
        row = {
            "roots": entry["roots"],
            "effect_ratio_median": statistics.median(effects) if effects else None,
            "effect_ratio_min": min(effects) if effects else None,
            "tags": dict(sorted(entry["tags"].items())),
            "not_separable_but_distinguishable_by_final_verdict": entry["distinguishable"],
            "experts": {},
        }
        for name, expert in entry["experts"].items():
            count = len(expert["steps"])
            row["experts"][name] = {"success": expert["success"], "roots": count,
                                    "mean_steps": (sum(expert["steps"]) / count) if count else None}
            totals[name]["success"] += expert["success"]
            totals[name]["roots"] += count
            totals[name]["steps"] += sum(expert["steps"])
        table[stratum] = row
    overall = {name: {"success": t["success"], "roots": t["roots"],
                      "success_rate": (t["success"] / t["roots"]) if t["roots"] else None,
                      "mean_steps": (t["steps"] / t["roots"]) if t["roots"] else None} for name, t in totals.items()}
    best = max((v["success_rate"] for v in overall.values() if v["success_rate"] is not None), default=None)
    tag_totals: dict[str, int] = defaultdict(int)
    for tag in tags:
        tag_totals[str(tag["tag"])] += 1
    balanced = sum(v for k, v in tag_totals.items() if k != "no_balanced_target")
    return {
        "roots": len(suite),
        "strata": table,
        "experts": overall,
        "tags": dict(sorted(tag_totals.items())),
        "separable_fraction_of_balanced_roots": (tag_totals.get("separable_by_verification", 0) / balanced) if balanced else None,
        "decision": (None if best is None else
                     "not hard enough (best expert above 95%)" if best > 0.95 else
                     "hard enough (best expert 60-95%)" if best >= 0.60 else
                     "very hard (best expert below 60%): check the unlearnable share"),
    }


def markdown(report: Mapping[str, Any]) -> str:
    names = sorted(report["experts"])
    header = "| stratum | roots | effect ratio (median) | tags | " + " | ".join(f"{n} success" for n in names) + " |"
    lines = [header, "| " + " | ".join(["---"] * (4 + len(names))) + " |"]
    for stratum, row in report["strata"].items():
        effect = row["effect_ratio_median"]
        tags = ", ".join(f"{k} {v}" for k, v in row["tags"].items()) or "-"
        cells = [f"{row['experts'].get(n, {}).get('success', '-')}/{row['experts'].get(n, {}).get('roots', '-')}" for n in names]
        lines.append(f"| {stratum} | {row['roots']} | {'-' if effect is None else f'{effect:.1f}'} | {tags} | " + " | ".join(cells) + " |")
    totals = [f"{report['experts'][n]['success']}/{report['experts'][n]['roots']}" for n in names]
    lines.append(f"| all | {report['roots']} | | | " + " | ".join(totals) + " |")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--tags", default=None)
    parser.add_argument("--expert", action="append", default=[], help="NAME=path/to/episodes.jsonl")
    parser.add_argument("--no-effect", action="store_true", help="skip the WLS effect ratio of every root")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    suite = json.loads(Path(args.suite).read_text(encoding="utf-8"))
    tags = load_jsonl(args.tags) if args.tags else []
    experts = {}
    for item in args.expert:
        name, _, path = item.partition("=")
        experts[name] = load_jsonl(path)
    report = build_report(suite, tags=tags, experts=experts, with_effect=not args.no_effect)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output.with_suffix(".md").write_text(markdown(report) + "\n", encoding="utf-8")
    print(markdown(report))
    print(json.dumps({"experts": report["experts"], "tags": report["tags"],
                      "separable_fraction_of_balanced_roots": report["separable_fraction_of_balanced_roots"],
                      "decision": report["decision"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
