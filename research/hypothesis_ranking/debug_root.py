"""Replay one root with the research expert and print the full trace with tool errors (step 3 debugging).

    python -m research.hypothesis_ranking.debug_root --roots-file output/.../step3_roots.json --scenario-id r0_... [--expert ledger] [--screen]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.run_dagger_research as research  # noqa: E402
from psse_env.evidence_profile import SUSPICION_GATED_PROFILE  # noqa: E402
from research.hypothesis_ranking.features import analyze_state, json_safe  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--roots-file", required=True)
    parser.add_argument("--scenario-id", required=True)
    parser.add_argument("--expert", choices=("baseline", "ledger"), default="baseline")
    parser.add_argument("--max-steps", type=int, default=research.RESEARCH_EPISODE_BUDGET)
    parser.add_argument("--screen", action="store_true", help="also print the balanced screen's report on the root")
    args = parser.parse_args(argv)
    roots = json.loads(Path(args.roots_file).read_text(encoding="utf-8"))
    root = next(r for r in roots if str(r["execution"]["scenario_id"]) == args.scenario_id)
    print("family:", root["grouping"]["scenario_family"], "truth:", json.dumps(json_safe(root["audit"]["truth"]))[:600])
    if args.screen:
        execution = root["execution"]
        analysis = analyze_state({"case": execution["case"], "measurements": execution["measurements"],
                                  "metadata": execution.get("metadata") or {}})
        screen = analysis.get("screen") or {}
        print("screen:", json.dumps(json_safe({k: screen.get(k) for k in (
            "outcome", "suspected", "explained", "unexplained", "accepted_hypotheses", "voltage_meter_channels")})))
        for r in screen.get("rounds") or []:
            print("  round winner", r.get("winner"), "scores", {k: round(v, 1) for k, v in (r.get("scores") or {}).items()},
                  "best", {k: {kk: vv for kk, vv in (v or {}).items() if kk in ("branch_row0", "channel_index0", "parameter", "J")} for k, v in (r.get("best") or {}).items()})
    research.RESEARCH_ENVIRONMENT_OPTIONS["evidence_profile"] = SUSPICION_GATED_PROFILE
    research.RESEARCH_ENVIRONMENT_OPTIONS["normalized_residual_threshold"] = 4.0
    research.RESEARCH_EXPERT_OPTIONS["hypothesis_ledger"] = args.expert == "ledger"
    factory = research.resolve_environment_factory("research", SUSPICION_GATED_PROFILE)
    env = factory()
    policy = research.research_expert_policy(factory)
    env.reset(root["execution"])
    for step in range(int(args.max_steps)):
        observation = env.get_policy_observation().as_dict()
        action = policy.act(observation)
        _, output = env.step(action)
        status = output.get("execution_status")
        line = f"{step + 1:2d} {action['tool']} {json.dumps({k: v for k, v in (action.get('arguments') or {}).items() if k != 'state_id'})} -> {status}"
        if status != "success":
            line += f" error={output.get('error_code')} detail={output.get('error_detail')}"
        metrics = output.get("tool_metrics") or {}
        if action["tool"] == "run_wls" and status == "success":
            screen = metrics.get("hif_screen") or {}
            line += f" | alarm={metrics.get('chi_square_alarm') or metrics.get('normalized_residual_alarm')} outcome={screen.get('outcome')} unexplained={screen.get('unexplained')} vm={screen.get('voltage_meter_channels')}"
        if action["tool"] == "ask_for_more_evidence":
            line += f" | audit={json.dumps(json_safe(metrics.get('operator_escalation_audit') or {}))[:300]}"
        print(line)
        if env.is_terminal():
            print("terminal:", env.terminal_outcome)
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
