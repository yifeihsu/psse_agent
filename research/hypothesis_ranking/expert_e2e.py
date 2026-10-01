"""Step 2 end-to-end check: the rule-based expert on every family under the suspicion-gated contract.

Builds a few roots per family with the round-0 generator (the DAgger suites'
admission), rolls the research expert out on them with the paired-evaluation
contract, and summarizes per family what step 2 cares about: truth-audited
task success and its basis (counterfactual resolution, bounded branch
handoff, unexplained-discrepancy handoff), the terminal outcome, how many
episodes acquired phasors or spectra and after how many steps, false commits
and healthy components touched, and the mean episode length.

    python -m research.hypothesis_ranking.expert_e2e --output-dir output/hypothesis_ranking_20260930/expert_e2e --per-family 4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.run_dagger_research as research  # noqa: E402
from psse_env.actions import (  # noqa: E402
    GET_HARMONIC_CONTEXT, GET_THREE_PHASE_CONTEXT, RUN_HSE_FROM_PATH, RUN_THREE_PHASE_NLM_FROM_PATH,
)
from psse_env.dagger.evaluator import evaluate_rollout_suites  # noqa: E402
from psse_env.dagger.release_factories import deterministic_case_loader  # noqa: E402
from psse_env.dagger.suite_builder import partition_release_scenario_v1  # noqa: E402
from psse_env.evidence_profile import SUSPICION_GATED_PROFILE  # noqa: E402
from research.hypothesis_ranking.build_dataset import build_generator  # noqa: E402
from research.hypothesis_ranking.features import json_safe  # noqa: E402

FAMILIES = (
    "no_error", "measurement", "multi_measurement", "parameter", "measurement+parameter", "topology",
    "measurement+topology", "harmonic", "hif", "measurement+hif", "three_phase_unbalance", "telemetry_no_disturbance",
)
NEEDS_PHASORS = {"hif", "measurement+hif", "three_phase_unbalance"}
NEEDS_SPECTRA = {"harmonic"}


def _basis(episode: Mapping[str, Any]) -> str:
    audit = episode.get("audit") if isinstance(episode.get("audit"), Mapping) else {}
    assessment = audit.get("truth_audited_task_assessment") if isinstance(audit, Mapping) else None
    if isinstance(assessment, Mapping) and assessment.get("basis"):
        return str(assessment["basis"])
    for key in ("truth_audited_task_assessment", "task_success_assessment"):
        value = episode.get(key)
        if isinstance(value, Mapping) and value.get("basis"):
            return str(value["basis"])
    return "none" if not episode.get("truth_audited_task_success") else "unknown"


def _escalation_request(episode: Mapping[str, Any]) -> str | None:
    trace = episode.get("trace") or []
    for step in reversed(trace):
        action = step.get("action") or {}
        if action.get("tool") == "ask_for_more_evidence":
            output = step.get("tool_output") or step.get("outcome") or {}
            if isinstance(output, Mapping) and output.get("execution_status") == "success":
                return str((action.get("arguments") or {}).get("request"))
    return None


def summarize(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_family: dict[str, dict[str, Any]] = {}
    groups: dict[str, list] = defaultdict(list)
    for episode in episodes:
        groups[str(episode.get("family"))].append(episode)
    for family, rows in sorted(groups.items()):
        counter: Counter = Counter()
        steps_total = 0
        phasor_steps: list[int] = []
        for episode in rows:
            tools = [str((step.get("action") or {}).get("tool")) for step in episode.get("trace") or []]
            counter["n"] += 1
            counter["truth_audited_success"] += bool(episode.get("truth_audited_task_success"))
            counter["terminal"] += bool(episode.get("terminal"))
            counter[f"outcome_{episode.get('terminal_outcome') or 'nonterminal'}"] += 1
            counter[f"basis_{_basis(episode)}"] += 1
            request = _escalation_request(episode)
            if request:
                counter[f"request_{request.split(':', 1)[-1]}"] += 1
            if GET_THREE_PHASE_CONTEXT in tools:
                counter["phasors_acquired"] += 1
                phasor_steps.append(tools.index(GET_THREE_PHASE_CONTEXT) + 1)
            counter["nlm_run"] += RUN_THREE_PHASE_NLM_FROM_PATH in tools
            counter["spectra_acquired"] += GET_HARMONIC_CONTEXT in tools
            counter["hse_run"] += RUN_HSE_FROM_PATH in tools
            counter["false_commits"] += int(episode.get("false_commit_count") or 0)
            counter["healthy_touched"] += 0 if episode.get("healthy_components_preserved", True) else 1
            steps_total += len(tools)
        entry = dict(counter)
        entry["mean_steps"] = steps_total / max(1, len(rows))
        entry["mean_phasor_step"] = (sum(phasor_steps) / len(phasor_steps)) if phasor_steps else None
        entry["unnecessary_phasors"] = entry.get("phasors_acquired", 0) if family not in NEEDS_PHASORS else 0
        entry["unnecessary_spectra"] = entry.get("spectra_acquired", 0) if family not in NEEDS_SPECTRA else 0
        by_family[family] = entry
    totals = Counter()
    for entry in by_family.values():
        for key in ("n", "truth_audited_success", "phasors_acquired", "spectra_acquired", "unnecessary_phasors",
                    "unnecessary_spectra", "false_commits", "healthy_touched"):
            totals[key] += int(entry.get(key, 0) or 0)
    return {"by_family": by_family, "totals": dict(totals)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--per-family", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--families", nargs="*", default=list(FAMILIES))
    parser.add_argument("--max-steps", type=int, default=research.RESEARCH_EPISODE_BUDGET)
    args = parser.parse_args(argv)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    generator = build_generator(args.seed)
    built = generator.build({family: int(args.per_family) for family in args.families})
    roots = [partition_release_scenario_v1(row, split="dagger_train") for row in built]
    print(f"[e2e] {len(roots)} roots built in {time.perf_counter() - started:.1f} s", flush=True)
    research.RESEARCH_ENVIRONMENT_OPTIONS["evidence_profile"] = SUSPICION_GATED_PROFILE
    research.RESEARCH_ENVIRONMENT_OPTIONS["normalized_residual_threshold"] = 4.0
    factory = research.resolve_environment_factory("research", SUSPICION_GATED_PROFILE)

    def expert(**_kwargs):
        return research.research_expert_policy(factory)

    result = evaluate_rollout_suites(
        {"standard_success": roots}, env_factory=factory, policy_factory=expert,
        max_steps=int(args.max_steps), seed=args.seed, case_loader=deterministic_case_loader,
        **research.PAIRED_EVALUATION_CONTRACT,
    ).as_dict()
    episodes = result["suite_metrics"]["episodes"]
    summary = summarize(episodes)
    summary.update(seed=args.seed, per_family=args.per_family, roots=len(roots), max_steps=int(args.max_steps),
                   evidence_profile=SUSPICION_GATED_PROFILE, total_seconds=time.perf_counter() - started)
    (out / "summary.json").write_text(json.dumps(json_safe(summary), indent=2, sort_keys=True), encoding="utf-8")
    with (out / "episodes.jsonl").open("w", encoding="utf-8") as stream:
        for episode in episodes:
            compact = {key: episode.get(key) for key in (
                "scenario_id", "family", "terminal", "terminal_outcome", "truth_audited_task_success",
                "false_commit_count", "healthy_components_preserved", "steps",
            )}
            compact["tools"] = [str((step.get("action") or {}).get("tool")) for step in episode.get("trace") or []]
            compact["escalation_request"] = _escalation_request(episode)
            compact["basis"] = _basis(episode)
            stream.write(json.dumps(json_safe(compact), sort_keys=True) + "\n")
    lines = ["# Expert end-to-end check under suspicion_gated_diagnostics\n",
             f"{len(roots)} roots ({args.per_family} per family, seed {args.seed}), budget {args.max_steps} steps.\n",
             "| family | n | success | outcomes | bases | phasors | NLM | spectra | HSE | mean steps | false commits |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for family, entry in summary["by_family"].items():
        outcomes = ", ".join(f"{k[8:]} {v}" for k, v in sorted(entry.items()) if k.startswith("outcome_"))
        bases = ", ".join(f"{k[6:]} {v}" for k, v in sorted(entry.items()) if k.startswith("basis_"))
        lines.append(f"| {family} | {entry['n']} | {entry['truth_audited_success']} | {outcomes} | {bases} | "
                     f"{entry.get('phasors_acquired', 0)} | {entry.get('nlm_run', 0)} | {entry.get('spectra_acquired', 0)} | "
                     f"{entry.get('hse_run', 0)} | {entry['mean_steps']:.1f} | {entry.get('false_commits', 0)} |")
    totals = summary["totals"]
    lines.append(f"\nTotals: success {totals['truth_audited_success']} of {totals['n']}; phasors acquired {totals['phasors_acquired']} "
                 f"(unnecessary {totals['unnecessary_phasors']}); spectra acquired {totals['spectra_acquired']} "
                 f"(unnecessary {totals['unnecessary_spectra']}); false commits {totals['false_commits']}; "
                 f"healthy components touched {totals['healthy_touched']}.\n")
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
