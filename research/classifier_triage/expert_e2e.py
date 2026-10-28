"""Stage 1 check: the rule expert end to end on the development roots under a triage source.

    python -m research.classifier_triage.expert_e2e --suite output/classifier_triage_20261004/suites/development.json \\
        --profile classifier_gated_diagnostics --output-dir output/classifier_triage_20261004/e2e/classifier_baseline

Rolls the research expert out on a saved suite of partitioned release roots
(the pipeline's ``development.json``) under one evidence profile, through the
paired-evaluation contract, and summarizes per family what the triage
decision changes: truth-audited success, phasor and spectra acquisitions
(and the unnecessary ones), false commits, episode length, and under the
classifier profile the first triage report of each episode (its request
score, whether the request was admitted, the first family) so recall and
unneeded requests of the gate in closed loop sit beside the offline figures.

``--profile scada_only`` with ``--expert baseline_compare`` is the preflight of
the conventional-only cell (2026-10-22): no auxiliary tool and no triage
report; the table then also carries the workflow metrics of that cell's
summary (invalid-action and loop episodes, WLS calls, bounded comparisons).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.run_dagger_research as research  # noqa: E402
from psse_env.actions import GET_HARMONIC_CONTEXT, GET_THREE_PHASE_CONTEXT  # noqa: E402
from psse_env.dagger.evaluator import evaluate_rollout_suites  # noqa: E402
from psse_env.dagger.release_factories import deterministic_case_loader  # noqa: E402
from psse_env.transactional_env import DEFAULT_PROGRESS_BASIS, PROGRESS_BASES  # noqa: E402
from psse_env.evidence_profile import (  # noqa: E402
    CLASSIFIER_GATED_PROFILE, SCADA_ONLY_PROFILE, SUSPICION_GATED_PROFILE, WLS_GATED_PROFILE, validate_evidence_profile,
)
from psse_env.oracle.expert_variants import EXPERT_VARIANTS  # noqa: E402
from research.hypothesis_ranking.expert_e2e import NEEDS_PHASORS, NEEDS_SPECTRA, _basis, _escalation_request, summarize  # noqa: E402
from research.hypothesis_ranking.features import json_safe  # noqa: E402
from research.rl.decision_audit import parse_run_options  # noqa: E402

PROFILES = (SCADA_ONLY_PROFILE, WLS_GATED_PROFILE, SUSPICION_GATED_PROFILE, CLASSIFIER_GATED_PROFILE)
COMPARE_TOOL = "compare_parameter_candidates"


def workflow_summary(episodes: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per family: invalid-action, false-commit and loop episodes, WLS calls and comparisons."""
    by_family: dict[str, Counter] = {}
    for episode in episodes:
        counter = by_family.setdefault(str(episode.get("family")), Counter())
        counter["n"] += 1
        counter["invalid_action_episodes"] += int(int(episode.get("invalid_action_count") or 0) > 0)
        counter["false_commit_episodes"] += int(int(episode.get("false_commit_count") or 0) > 0)
        counter["loop_episodes"] += int(bool(episode.get("loop_detected")))
        counter["wls_calls"] += int(episode.get("wls_calls") or 0)
        counter["comparisons"] += int((episode.get("tool_counts") or {}).get(COMPARE_TOOL) or 0)
    table = {}
    for family, counter in sorted(by_family.items()):
        entry = dict(counter)
        entry["mean_wls_calls"] = counter["wls_calls"] / max(1, counter["n"])
        table[family] = entry
    return table


def first_triage(episode: Mapping[str, Any]) -> dict[str, Any] | None:
    """The triage report on the first solved state of the episode, as the policy saw it."""
    for step in episode.get("trace") or []:
        observation = step.get("policy_observation") or {}
        wls = ((observation.get("fresh_context_evidence") or {}).get("wls") or {}) if isinstance(observation, Mapping) else {}
        report = wls.get("triage")
        if isinstance(report, Mapping):
            return {key: report.get(key) for key in ("status", "request_score", "request_threshold", "request_admitted", "first_family")}
    return None


def triage_summary(episodes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The gate in closed loop: first-report admissions by family, recall and unneeded requests."""
    by_family: dict[str, Counter] = {}
    for episode in episodes:
        family = str(episode.get("family"))
        counter = by_family.setdefault(family, Counter())
        counter["n"] += 1
        report = first_triage(episode)
        if report is None:
            counter["no_report"] += 1
            continue
        counter["admitted_first"] += bool(report.get("request_admitted"))
        tools = [str((step.get("action") or {}).get("tool")) for step in episode.get("trace") or []]
        counter["phasors_acquired"] += GET_THREE_PHASE_CONTEXT in tools
        if report.get("first_family"):
            counter[f"first_{report['first_family']}"] += 1
    needed = [f for f in by_family if f in NEEDS_PHASORS]
    no_need = [f for f in by_family if f not in NEEDS_PHASORS and f not in NEEDS_SPECTRA]
    positives = sum(by_family[f]["n"] for f in needed)
    admitted = sum(by_family[f]["admitted_first"] for f in needed)
    negatives = sum(by_family[f]["n"] for f in no_need)
    unneeded = sum(by_family[f]["admitted_first"] for f in no_need)
    return {
        "by_family": {family: dict(counter) for family, counter in sorted(by_family.items())},
        "first_report_recall": (admitted / positives) if positives else None,
        "first_report_unneeded": (unneeded / negatives) if negatives else None,
        "roots_needing_phasors": positives, "roots_needing_none": negatives,
    }


def first_hypothesis_confidence(steps: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """The three-category confidence of the episode's first successful WLS (item 7 step 6), when the solve carried one."""
    for step in steps:
        if str((step.get("action") or {}).get("tool")) != "run_wls" or step.get("execution_status") != "success":
            continue
        output = step.get("policy_tool_output") if isinstance(step.get("policy_tool_output"), Mapping) else {}
        metrics = output.get("tool_metrics") if isinstance(output.get("tool_metrics"), Mapping) else {}
        record = metrics.get("hypothesis_confidence")
        return dict(record) if isinstance(record, Mapping) else None
    return None


def handoff_abstentions(episode: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The abstention statements of the episode's successful handoff, as the environment recorded them (compound study, stage 2 item 5)."""
    for step in reversed(episode.get("trace") or []):
        action = step.get("action") or {}
        if str(action.get("tool")) != "ask_for_more_evidence":
            continue
        output = step.get("policy_tool_output") or step.get("tool_output") or step.get("outcome") or {}
        status = step.get("execution_status")
        if status is None and isinstance(output, Mapping):
            status = output.get("execution_status")
        if status != "success":
            continue
        metrics = output.get("tool_metrics") if isinstance(output, Mapping) else None
        recorded = metrics.get("abstentions") if isinstance(metrics, Mapping) else None
        if isinstance(recorded, (list, tuple)):
            return [dict(item) for item in recorded if isinstance(item, Mapping)]
        declared = (action.get("arguments") or {}).get("abstentions")
        return [dict(item) for item in declared if isinstance(item, Mapping)] if isinstance(declared, (list, tuple)) else []
    return []


def load_suite(path: Path, families: Sequence[str] | None, limit: int | None, per_family: int | None = None) -> list[dict[str, Any]]:
    roots = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(roots, Mapping):
        roots = roots.get("scenarios") or roots.get("roots") or []
    if families:
        wanted = set(families)
        roots = [root for root in roots if str(root.get("grouping", {}).get("scenario_family")) in wanted]
    if per_family:
        taken: Counter = Counter()
        sampled = []
        for root in roots:
            family = str(root.get("grouping", {}).get("scenario_family"))
            if taken[family] < per_family:
                taken[family] += 1
                sampled.append(root)
        roots = sampled
    if limit:
        roots = roots[:limit]
    return list(roots)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", required=True, help="partitioned release roots (the pipeline's development.json)")
    parser.add_argument("--profile", choices=PROFILES, default=CLASSIFIER_GATED_PROFILE)
    parser.add_argument("--expert", choices=EXPERT_VARIANTS, default="baseline")
    parser.add_argument("--triage-model", default=None, help="runtime directory of the triage classifier (default: the tracked model)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--families", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None, help="roots to roll out, in suite order")
    parser.add_argument("--per-family", type=int, default=None, help="at most this many roots of each family")
    parser.add_argument("--max-steps", type=int, default=research.RESEARCH_EPISODE_BUDGET)
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--progress-basis", choices=PROGRESS_BASES, default=DEFAULT_PROGRESS_BASIS,
                        help="basis of a candidate's global progress in the environment (psse_env.transactional_env)")
    parser.add_argument("--env-option", action="append", default=None, metavar="KEY=VALUE",
                        help="research environment option (compound-error study, stage 2)")
    parser.add_argument("--expert-option", action="append", default=None, metavar="KEY=VALUE",
                        help="expert option on top of the variant (compound-error study, stage 2)")
    args = parser.parse_args(argv)
    started = time.perf_counter()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    profile = validate_evidence_profile(args.profile)
    roots = load_suite(Path(args.suite), args.families, args.limit, args.per_family)
    print(f"[e2e] {len(roots)} roots from {args.suite} under {profile} ({args.expert} expert)", flush=True)

    research.RESEARCH_ENVIRONMENT_OPTIONS["evidence_profile"] = profile
    research.RESEARCH_ENVIRONMENT_OPTIONS["normalized_residual_threshold"] = 4.0
    research.RESEARCH_ENVIRONMENT_OPTIONS["triage_classifier"] = args.triage_model
    research.RESEARCH_ENVIRONMENT_OPTIONS["progress_basis"] = args.progress_basis
    env_options = parse_run_options(args.env_option)
    expert_options = parse_run_options(args.expert_option)
    for key, value in env_options.items():
        research.RESEARCH_ENVIRONMENT_OPTIONS[key] = value
    for key, value in expert_options.items():
        research.RESEARCH_EXPERT_OPTIONS[key] = value
    research.RESEARCH_EXPERT_OPTIONS["variant"] = args.expert
    factory = research.resolve_environment_factory("research", profile)

    def expert(**_kwargs):
        return research.research_expert_policy(factory)

    result = evaluate_rollout_suites(
        {"standard_success": roots}, env_factory=factory, policy_factory=expert,
        max_steps=int(args.max_steps), seed=args.seed, case_loader=deterministic_case_loader,
        **research.PAIRED_EVALUATION_CONTRACT,
    ).as_dict()
    episodes = result["suite_metrics"]["episodes"]
    family_of = {str(root["execution"]["scenario_id"]): str(root.get("grouping", {}).get("scenario_family")) for root in roots}
    for episode in episodes:
        if not episode.get("family") or episode.get("family") == "None":
            episode["family"] = family_of.get(str(episode.get("scenario_id")), str(episode.get("family")))
    summary = summarize(episodes)
    summary["triage"] = triage_summary(episodes)
    summary["workflow"] = workflow_summary(episodes)
    summary.update(profile=profile, expert=args.expert, triage_model=args.triage_model, progress_basis=args.progress_basis,
                   env_options=env_options, expert_options=expert_options,
                   suite=str(args.suite), roots=len(roots),
                   seed=args.seed, max_steps=int(args.max_steps), total_seconds=time.perf_counter() - started)
    (out / "summary.json").write_text(json.dumps(json_safe(summary), indent=2, sort_keys=True), encoding="utf-8")
    with (out / "episodes.jsonl").open("w", encoding="utf-8") as stream:
        for episode in episodes:
            compact = {key: episode.get(key) for key in (
                "scenario_id", "family", "physical_root", "terminal", "terminal_outcome", "truth_audited_task_success",
                "false_commit_count", "invalid_action_count", "loop_detected", "wls_calls",
                "healthy_components_preserved", "steps")}
            compact["tools"] = [str((step.get("action") or {}).get("tool")) for step in episode.get("trace") or []]
            compact["escalation_request"] = _escalation_request(episode)
            compact["basis"] = _basis(episode)
            compact["first_triage"] = first_triage(episode)
            compact["abstentions"] = handoff_abstentions(episode)
            compact["revert_accounting"] = dict(episode.get("revert_accounting") or {})
            compact["first_hypothesis_confidence"] = first_hypothesis_confidence(
                [step for step in episode.get("trace") or [] if not step.get("intervention")])
            compact["corrections"] = [
                {"tool": str((step.get("action") or {}).get("tool")),
                 "arguments": {k: v for k, v in ((step.get("action") or {}).get("arguments") or {}).items() if k != "state_id"},
                 "status": str(((step.get("tool_output") or step.get("outcome") or {}) or {}).get("execution_status"))}
                for step in episode.get("trace") or []
                if str((step.get("action") or {}).get("tool")) in ("correct_measurements", "correct_parameters", "correct_topology")]
            stream.write(json.dumps(json_safe(compact), sort_keys=True) + "\n")
    lines = [f"# Expert end to end under {profile} ({args.expert} expert, {len(roots)} roots)\n",
             "| family | n | success | phasors | unnecessary phasors | spectra | false commits | mean steps | admitted at first WLS "
             "| invalid-action episodes | loop episodes | mean WLS calls | comparisons |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for family, entry in sorted(summary["by_family"].items()):
        triage = summary["triage"]["by_family"].get(family, {})
        workflow = summary["workflow"].get(family, {})
        lines.append(f"| {family} | {entry['n']} | {entry.get('truth_audited_success', 0)} | {entry.get('phasors_acquired', 0)} | "
                     f"{entry.get('unnecessary_phasors', 0)} | {entry.get('spectra_acquired', 0)} | {entry.get('false_commits', 0)} | "
                     f"{entry['mean_steps']:.1f} | {triage.get('admitted_first', 'n/a')} | "
                     f"{workflow.get('invalid_action_episodes', 0)} | {workflow.get('loop_episodes', 0)} | "
                     f"{workflow.get('mean_wls_calls', 0.0):.1f} | {workflow.get('comparisons', 0)} |")
    totals = summary["totals"]
    lines += ["", f"Totals: success {totals['truth_audited_success']}/{totals['n']}, phasors {totals['phasors_acquired']} "
                  f"(unnecessary {totals['unnecessary_phasors']}), spectra {totals['spectra_acquired']} (unnecessary {totals['unnecessary_spectra']}), "
                  f"false commits {totals['false_commits']}, healthy touched {totals['healthy_touched']}."]
    triage = summary["triage"]
    if triage["first_report_recall"] is not None:
        lines.append(f"First triage report: recall {100 * triage['first_report_recall']:.1f}% on {triage['roots_needing_phasors']} roots "
                     f"that need phasors, unneeded {100 * (triage['first_report_unneeded'] or 0):.1f}% on {triage['roots_needing_none']} that need none.")
    lines.append(f"Wall time {summary['total_seconds'] / 60:.1f} min.")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
