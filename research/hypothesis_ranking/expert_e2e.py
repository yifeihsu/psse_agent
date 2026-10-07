"""Step 2 end-to-end check: the rule-based expert on every family under the suspicion-gated contract.

Builds a few roots per family with the round-0 generator (the DAgger suites'
admission), rolls the research expert out on them with the paired-evaluation
contract, and summarizes per family what step 2 cares about: truth-audited
task success and its basis (counterfactual resolution, bounded branch
handoff, unexplained-discrepancy handoff), the terminal outcome, how many
episodes acquired phasors or spectra and after how many steps, false commits
and healthy components touched, and the mean episode length.

    python -m research.hypothesis_ranking.expert_e2e --output-dir output/hypothesis_ranking_20260930/expert_e2e --per-family 4

Step 5 arms: ``--expert ledger_ranked`` is the ledger expert with the learned
ranker's acquisition deferral (``--ranker-model`` points at another export);
``--mimic-per-variant N`` adds N same-sign and N opposite-sign flow-meter
pair roots (two biased flow meters at the ends of one candidate HIF line,
built from clean corpus windows as two-meter roots and reported as their own
families), cached beside ``--roots-file``; every episode records whether a
balanced correction was tried on a state whose screen flagged an HIF before
the phasors were requested (``deferred_acquisition``).
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
MIMIC_FAMILIES = {"mimic_flow_pair_same_sign": (1.0, 1.0), "mimic_flow_pair_opposite_sign": (1.0, -1.0)}
EXPERTS = ("baseline", "ledger", "ledger_ranked")


def build_mimic_roots(seed: int, count_per_variant: int) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Flow-meter pair mimics as release roots: two biased flow meters at the ends of one candidate HIF line.

    Built from clean corpus windows through the generator's own two-meter
    admission (``_measurement_scenario`` with both indices declared, the
    truth-restored vector required clean, sequential observability, the
    recovery tolerance declared, true-state phasors and clean spectra
    attached), so the environment and the truth audit see an ordinary
    two-meter root; the harness reports them under their mimic family.
    Returns the partitioned roots and the scenario-id -> mimic family map.
    """
    import numpy as np

    from mcp_server.matpower_server import _load_python_case
    from psse_env.providers.hif_screen import default_hif_lines
    from research.hypothesis_ranking.features import flow_channel_index

    generator = build_generator(seed + 7)
    case = _load_python_case(generator.case_path)
    nb, nl = int(np.asarray(case["bus"]).shape[0]), int(np.asarray(case["branch"]).shape[0])
    lines = default_hif_lines(case)
    rng = np.random.default_rng(seed + 202)
    clean_rows = generator._corpus().get("no_error", [])
    order = rng.permutation(len(clean_rows))
    roots: list[dict[str, Any]] = []
    families: dict[str, str] = {}
    for variant_index, (family, signs) in enumerate(MIMIC_FAMILIES.items()):
        built = 0
        for position in order[variant_index::2]:
            if built >= count_per_variant:
                break
            row = clean_rows[int(position)]
            sigma = list(row.get("sigma_z") or generator.noise_profile().tolist())
            z_true = [float(v) for v in row["z_obs"]]
            z = list(z_true)
            line = int(lines[built % len(lines)])
            indices = [flow_channel_index("Pf", line, nb, nl), flow_channel_index("Pt", line, nb, nl)]
            for index, sign in zip(indices, signs):
                z[index] = z_true[index] + sign * float(rng.uniform(10.0, 15.0)) * float(sigma[index])
            synthetic = {"id": f"{family}:{row['id']}:{line}", "z_obs": z, "z_true": z_true, "sigma_z": sigma,
                         "label": {"indices": indices, "channel": None, "error_type": "measurement_error",
                                   "subtype": "multi_gross_outliers"}}
            original_source = generator._family_source
            generator._family_source = lambda name, _rows=[synthetic], _builder=None: (  # noqa: E731
                _rows, lambda r, i: generator._measurement_scenario(r, i, family="multi_measurement"))
            try:
                scenarios = generator._build_family("multi_measurement", 1)
            finally:
                generator._family_source = original_source
            if not scenarios:
                continue
            scenario = scenarios[0]
            scenario["mimic"] = {"family": family, "branch_row0": line, "signs": list(signs)}
            root = partition_release_scenario_v1(scenario, split="dagger_train")
            families[str(root["execution"]["scenario_id"])] = family
            roots.append(root)
            built += 1
    return roots, families


def _deferred_acquisition(episode: Mapping[str, Any]) -> bool:
    """A balanced correction tried on a state whose screen flagged an HIF before any phasor request."""
    for step in episode.get("trace") or []:
        action = step.get("action") or {}
        tool = str(action.get("tool"))
        if tool == GET_THREE_PHASE_CONTEXT:
            return False
        if tool not in ("correct_measurements", "correct_parameters", "correct_topology"):
            continue
        observation = step.get("policy_observation") or {}
        wls = ((observation.get("fresh_context_evidence") or {}).get("wls") or {}) if isinstance(observation, Mapping) else {}
        screen = wls.get("hif_screen") or {}
        if screen.get("suspected") and not screen.get("refuted_by_phase_measurements"):
            return True
    return False


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
    """The request of the episode's last successful ``ask_for_more_evidence``.

    Evaluator trace steps carry the status at the top level and the output as
    ``policy_tool_output``; older traces nested both under ``tool_output``.
    """
    trace = episode.get("trace") or []
    for step in reversed(trace):
        action = step.get("action") or {}
        if action.get("tool") == "ask_for_more_evidence":
            output = step.get("policy_tool_output") or step.get("tool_output") or step.get("outcome") or {}
            status = step.get("execution_status")
            if status is None and isinstance(output, Mapping):
                status = output.get("execution_status")
            if status == "success":
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
            counter["deferred_acquisition"] += int(_deferred_acquisition(episode))
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
                    "unnecessary_spectra", "false_commits", "healthy_touched", "deferred_acquisition"):
            totals[key] += int(entry.get(key, 0) or 0)
    return {"by_family": by_family, "totals": dict(totals)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--per-family", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--families", nargs="*", default=list(FAMILIES))
    parser.add_argument("--max-steps", type=int, default=research.RESEARCH_EPISODE_BUDGET)
    parser.add_argument("--plan", type=json.loads, default=None,
                        help="JSON family->count; overrides --per-family/--families (e.g. the development plan)")
    parser.add_argument("--expert", choices=EXPERTS, default="baseline",
                        help="the current expert, the expert with the step-3 hypothesis ledger, or the ledger with the learned ranker")
    parser.add_argument("--ranker-model", default=None, help="learned ranker export for --expert ledger_ranked (default: the tracked model)")
    parser.add_argument("--roots-file", default=None,
                        help="JSON file of partitioned roots: loaded when it exists, written after generation otherwise")
    parser.add_argument("--mimic-per-variant", type=int, default=0,
                        help="add this many same-sign and opposite-sign flow-meter pair roots (cached beside --roots-file)")
    args = parser.parse_args(argv)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    plan = dict(args.plan) if args.plan else {family: int(args.per_family) for family in args.families}
    research.RESEARCH_EXPERT_OPTIONS["variant"] = args.expert
    research.RESEARCH_EXPERT_OPTIONS["learned_ranker"] = args.ranker_model
    roots_file = Path(args.roots_file) if args.roots_file else None
    if roots_file is not None and roots_file.is_file():
        roots = json.loads(roots_file.read_text(encoding="utf-8"))
        print(f"[e2e] {len(roots)} roots loaded from {roots_file}", flush=True)
    else:
        generator = build_generator(args.seed)
        built = generator.build({str(k): int(v) for k, v in plan.items()})
        roots = [partition_release_scenario_v1(row, split="dagger_train") for row in built]
        if roots_file is not None:
            roots_file.parent.mkdir(parents=True, exist_ok=True)
            roots_file.write_text(json.dumps(json_safe(roots)), encoding="utf-8")
            print(f"[e2e] roots saved to {roots_file}", flush=True)
    mimic_families: dict[str, str] = {}
    if args.mimic_per_variant > 0:
        mimic_file = roots_file.with_name(roots_file.stem + f"_mimic{args.mimic_per_variant}.json") if roots_file is not None else None
        if mimic_file is not None and mimic_file.is_file():
            cached = json.loads(mimic_file.read_text(encoding="utf-8"))
            mimic_roots, mimic_families = cached["roots"], cached["families"]
            print(f"[e2e] {len(mimic_roots)} mimic roots loaded from {mimic_file}", flush=True)
        else:
            mimic_roots, mimic_families = build_mimic_roots(args.seed, int(args.mimic_per_variant))
            if mimic_file is not None:
                mimic_file.write_text(json.dumps(json_safe({"roots": mimic_roots, "families": mimic_families})), encoding="utf-8")
                print(f"[e2e] mimic roots saved to {mimic_file}", flush=True)
        roots = [*roots, *mimic_roots]
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
    for episode in episodes:
        mimic = mimic_families.get(str(episode.get("scenario_id")))
        if mimic:
            episode["family"] = mimic
    by_id = {str(root["execution"]["scenario_id"]): root for root in roots}
    summary = summarize(episodes)
    summary.update(seed=args.seed, per_family=args.per_family, plan=plan, expert=args.expert, roots=len(roots),
                   max_steps=int(args.max_steps), evidence_profile=SUSPICION_GATED_PROFILE,
                   total_seconds=time.perf_counter() - started)
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
            compact["deferred_acquisition"] = _deferred_acquisition(episode)
            # The corrections attempted, in order, with their targets, and the
            # root's truth targets: the paired comparison reads which family
            # and target each arm tried first.
            compact["corrections"] = [
                {"tool": str((step.get("action") or {}).get("tool")),
                 "arguments": {k: v for k, v in ((step.get("action") or {}).get("arguments") or {}).items() if k != "state_id"},
                 "status": str(((step.get("tool_output") or step.get("outcome") or {}) or {}).get("execution_status"))}
                for step in episode.get("trace") or []
                if str((step.get("action") or {}).get("tool")) in ("correct_measurements", "correct_parameters", "correct_topology")
            ]
            root = by_id.get(str(episode.get("scenario_id")))
            truth = (root or {}).get("audit", {}).get("truth", {}) if isinstance(root, Mapping) else {}
            compact["truth"] = {
                "measurement_indices": [m.get("index") for m in truth.get("true_measurement_errors") or [] if isinstance(m, Mapping)],
                "parameter_lines": [p.get("line_index1") for p in truth.get("true_parameter_errors") or [] if isinstance(p, Mapping)],
                "topology_lines": [t.get("line_index1") for t in truth.get("true_topology_errors") or [] if isinstance(t, Mapping)],
            }
            stream.write(json.dumps(json_safe(compact), sort_keys=True) + "\n")
    lines = [f"# Expert end-to-end check under suspicion_gated_diagnostics ({args.expert} expert)\n",
             f"{len(roots)} roots (plan {json.dumps(plan, sort_keys=True)}, seed {args.seed}), budget {args.max_steps} steps.\n",
             "| family | n | success | outcomes | bases | phasors | NLM | spectra | HSE | deferred | mean steps | false commits |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for family, entry in summary["by_family"].items():
        outcomes = ", ".join(f"{k[8:]} {v}" for k, v in sorted(entry.items()) if k.startswith("outcome_"))
        bases = ", ".join(f"{k[6:]} {v}" for k, v in sorted(entry.items()) if k.startswith("basis_"))
        lines.append(f"| {family} | {entry['n']} | {entry['truth_audited_success']} | {outcomes} | {bases} | "
                     f"{entry.get('phasors_acquired', 0)} | {entry.get('nlm_run', 0)} | {entry.get('spectra_acquired', 0)} | "
                     f"{entry.get('hse_run', 0)} | {entry.get('deferred_acquisition', 0)} | {entry['mean_steps']:.1f} | {entry.get('false_commits', 0)} |")
    totals = summary["totals"]
    lines.append(f"\nTotals: success {totals['truth_audited_success']} of {totals['n']}; phasors acquired {totals['phasors_acquired']} "
                 f"(unnecessary {totals['unnecessary_phasors']}); spectra acquired {totals['spectra_acquired']} "
                 f"(unnecessary {totals['unnecessary_spectra']}); deferred acquisitions {totals.get('deferred_acquisition', 0)}; "
                 f"false commits {totals['false_commits']}; healthy components touched {totals['healthy_touched']}.\n")
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
