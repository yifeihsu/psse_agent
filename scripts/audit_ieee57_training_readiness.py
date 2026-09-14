"""Reproduce the P0 failure audit from retained local pilot artifacts.

The private branch labels below are used for retrospective physical auditing
only. All policy probes receive execution data or stored policy observations.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcp_server.matpower_server import _load_python_case
from psse_env.dagger.ieee57_runtime import (
    ieee57_environment_factory, ieee57_runtime_manifest, validate_ieee57_wls_metrics,
)
from psse_env.dagger.release_factories import (
    observable_expert_policy_factory, EXPERT_POLICY_IDENTITY,
)
from psse_env.providers.scenario_generator import build_measurement_vector
from scripts.validate_balanced_transfer import summarize_episode, evaluate_scenarios


def digest(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "location_kind": "local_file", "publicly_retrievable": False}


def build_audit(pilot: Path) -> dict:
    paths = {name: pilot / name for name in (
        "expert_validation.json", "scenarios.json", "generator_rows.json", "corpus/samples.jsonl"
    )}
    report = json.loads(paths["expert_validation.json"].read_text(encoding="utf-8"))
    scenarios = json.loads(paths["scenarios.json"].read_text(encoding="utf-8"))
    generated = json.loads(paths["generator_rows.json"].read_text(encoding="utf-8"))
    samples = {row["id"]: row for row in (
        json.loads(line) for line in paths["corpus/samples.jsonl"].read_text(encoding="utf-8").splitlines()
    )}
    rows = {row["execution"]["scenario_id"]: row for row in scenarios}
    generators = {row["scenario_id"]: row for row in generated}
    episodes = report["evaluation"]["suite_metrics"]["episodes"]
    result = {
        "contract": "ieee57_training_readiness_failure_audit_v1",
        "scope": "Development retrospective plus observable probes; not a training corpus or independent test",
        "sources": {name: digest(path) for name, path in paths.items()},
        "historical_trace_archive": {
            "repository_path": "research/ieee57/evidence/training_readiness_20260912.json.gz",
            "uncompressed_sha256": digest(paths["expert_validation.json"])["sha256"],
            "format": "gzip of original expert_validation.json bytes",
        },
        "historical_scenarios_archive": {
            "repository_path": "research/ieee57/evidence/training_readiness_scenarios_20260912.json.gz",
            "uncompressed_sha256": digest(paths["scenarios.json"])["sha256"],
            "format": "gzip of original scenarios.json bytes; test replays receive execution partition only",
        },
        "runtime": ieee57_runtime_manifest(),
        "source_episodes_retained": len(episodes),
        "episode_summaries": [summarize_episode(episode) for episode in episodes],
        "parameter_failures": [], "invalid_actions": [],
    }
    policy = observable_expert_policy_factory(policy_identity=EXPERT_POLICY_IDENTITY)
    for episode in episodes:
        scenario = rows[episode["scenario_id"]]
        if episode["family"] == "parameter" and episode["false_finalization_count"]:
            source = samples[generators[episode["scenario_id"]]["source_realization_id"]]
            truth = scenario["audit"]["truth"]
            error = truth["true_parameter_errors"][0]
            physical_path = pilot / "corpus/physical_cases" / Path(source["physical_case_path"]).name
            physical = _load_python_case(str(physical_path))
            configured = _load_python_case(scenario["execution"]["case"])
            index = int(error["branch_row0"])
            calculated = build_measurement_vector(physical)
            # Counterfactual diagnostic acquisition only: no labels or true
            # branch indices are supplied to the provider or policy.
            env = ieee57_environment_factory()
            env.reset(copy.deepcopy(scenario["execution"]))
            sid = env.store.active_state_id
            _, wls = env.step({"tool": "run_wls", "arguments": {"state_id": sid}})
            validate_ieee57_wls_metrics(wls["tool_metrics"])
            _, context = env.step({"tool": "get_parameter_context", "arguments": {"state_id": sid}})
            solved = env.wls_runner.__self__._solve(env.store.get_state(sid))
            result["parameter_failures"].append({
                "scenario_id": episode["scenario_id"],
                "physical_construction": {
                    "source_realization_id": source["id"], "branch_row0": index,
                    "line_index1": index + 1, "endpoints": physical["branch"][index, :2].tolist(),
                    "reported_rx": configured["branch"][index, 2:4].tolist(),
                    "true_rx": physical["branch"][index, 2:4].tolist(),
                    "stored_physical_validation": source["physical_validation"],
                    "operating_point": source["op_point"], "sensor_sigmas": source["sigmas"],
                    "physical_case": digest(physical_path),
                    "recomputed_hx_max_error": float(np.max(np.abs(calculated - source["z_true"]))),
                    "observed_vector_matches_source": scenario["execution"]["measurements"] == source["z_obs"],
                    "audit_clean_vector_matches_noisy_source": truth["clean_measurements"] == source["z_obs"],
                    "audit_clean_vector_semantics": "For parameter-only faults, clean_measurements retains the same sensor noise without measurement corruption; z_true is the separate noiseless physical vector.",
                },
                "initial_detection": summarize_episode(episode)["initial_wls"],
                "available_diagnosis": {
                    "historical_parameter_context_requested": False,
                    "historical_parameter_optimizer_called": False,
                    "observable_counterfactual_context": context,
                    "maximum_abs_branch_multiplier": float(np.max(np.abs(solved["payload"]["lambdaN"]))),
                    "supported_candidate_optimizer_called": False,
                    "correction_feasibility": "No provider-supported candidate; no hidden-truth-directed optimizer probe",
                },
                "termination": {
                    "action": episode["trace"][-1]["action"],
                    "policy_observation": episode["trace"][-1]["policy_observation"],
                    "outcome": episode["terminal_outcome"],
                    "false_finalization_count": episode["false_finalization_count"],
                    "remaining_true_fault_count": episode["audit"].get("remaining_true_fault_count"),
                    "interpretation": "Both detector tests pass and the available parameter context has no candidate. This is an undetected fault under the declared evidence and routing contract, not a demonstrated optimizer failure. Resolved supervision is quarantined; no truth-informed replacement is emitted.",
                },
            })
        for transition in episode["trace"]:
            if transition["execution_status"] == "success":
                continue
            step = transition["step"]
            result["invalid_actions"].append({
                "scenario_id": episode["scenario_id"], "step": step,
                "historical_policy_observation": transition["policy_observation"],
                "historical_action": transition["action"],
                "rejection": transition["policy_tool_output"],
                "following_recovery": [{"step": t["step"], "action": t["action"],
                    "execution_status": t["execution_status"], "error_code": t.get("error_code"),
                    "terminal_outcome": t.get("terminal_outcome")}
                    for t in episode["trace"] if t["step"] > step],
                "current_policy_action_on_exact_observation": policy.act(transition["policy_observation"]),
                "diagnosis": "Post-branch refinement eligibility ignored commit ordering. The meter had already been estimated after the branch repair, and its target NR was near zero. A repeat advertised correction produced no change. This was an execution-invalid no-op, not an accepted physical correction.",
                "repair": "Require the most recent target measurement correction to precede a subsequent branch repair, using ordered model-visible accepted corrections only. The parameter and residual alarms remain unchanged.",
            })
    # Re-run affected mixed episodes through the real protocol and offline
    # audit. Inputs are retained and output is separate from historical traces.
    affected = {row["scenario_id"] for row in result["invalid_actions"]}
    if affected:
        replay = evaluate_scenarios([rows[sid] for sid in sorted(affected)], seed=report["seed"])
        replay_path = pilot.parent / "ieee57_training_p0_20260913" / "mixed_replay.json"
        replay_path.parent.mkdir(parents=True, exist_ok=True)
        replay_path.write_text(json.dumps(replay, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        result["mixed_protocol_replay"] = {"source": digest(replay_path), "episodes": replay["episodes"]}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, default=ROOT / "output/ieee57_training_readiness_20260912")
    parser.add_argument("--output", type=Path, default=ROOT / "research/ieee57/training_readiness_failure_audit_20260912.json")
    args = parser.parse_args()
    result = build_audit(args.pilot)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "parameter_failures": len(result["parameter_failures"]),
                      "invalid_actions": len(result["invalid_actions"])}))


if __name__ == "__main__":
    main()
