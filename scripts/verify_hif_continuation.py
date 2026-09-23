"""Frozen-input HIF/meter continuation through the real provider and expert.

The environment receives only each frozen execution envelope. Hidden truth is
read after execution for a separate audit. Fits are fresh or explicitly cached
from the same observable inputs; historical rounded R1 summaries are not used.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import subprocess
import time

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ.setdefault("PSSE_HIF_WORKERS", "1")
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np
from threadpoolctl import threadpool_limits

from psse_env.actions import ESTIMATE_HIF_MULTISCAN_FROM_PATH
from psse_env.dagger.dataset_builder import validate_policy_payload
from psse_env.dagger.release_factories import (
    BC0_CHI2_ALPHA, BC0_PARAMETER_RANKING_DOMINANCE_THRESHOLD, select_observable_expert_actions,
    BC0_HIF_ALPHA_GRID_SIZE, BC0_HIF_R_GRID_SIZE, BC0_HIF_MAX_SCANS,
)
from psse_env.dagger.release_audit import audit_truth_audited_task_success
from mcp_server.matpower_server import _load_python_case
from psse_env.oracle import ExpertPolicyOracle
from psse_env.providers.matpower import MatpowerDeploymentProviders
from psse_env.providers.hif_continuation import current_scan, conditioned_prediction, case_fingerprint, model_fingerprint
from psse_env.transactional_env import TransactionalPSSEEnv
from three_phase_nlm.conditioned_meter_recovery import diagnose_conditioned_meter_errors

FIT_SOURCES = ("three_phase_nlm/hif_multiscan_estimator.py", "three_phase_nlm/hif_parameter_estimator.py",
    "three_phase_nlm/dss_hif_injector.py", "three_phase_nlm/hif_operating_point.py",
    "three_phase_nlm/branch_current_analysis.py", "three_phase_nlm/hif_units.py",
    "IEEE_14_OpenDSS/export_measurement_series.py", "IEEE_14_OpenDSS/measurement_convention.py")
PRIVATE_KEYS = {"audit", "truth", "hidden_truth", "label", "labels", "hif_label", "clean_measurements",
    "true_measurement_errors", "true_hif_errors", "true_parameter_errors", "true_topology_errors", "z_clean", "z_true"}


def jsonable(value):
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def digest(value):
    return hashlib.sha256(json.dumps(jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), indent=2, sort_keys=True, allow_nan=False)+"\n", encoding="utf-8")


def assert_observable(value, path="execution"):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in PRIVATE_KEYS or key.endswith("_clean"):
                raise ValueError(f"Private runtime input: {path}.{key}")
            assert_observable(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for item in value:
            assert_observable(item, path)


class ObservableFitCache:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.calls = []
        self.sources = {name: hashlib.sha256((REPO/name).read_bytes()).hexdigest() for name in FIT_SOURCES}

    def attach(self, provider):
        original = provider._memoized_hif_multiscan

        def fitted(**kwargs):
            assert_observable(kwargs, "fit_arguments")
            binding = {"arguments": kwargs, "fit_sources_sha256": self.sources,
                       "case_sha256": case_fingerprint(kwargs["case_path"]),
                       "pristine_model_sha256": model_fingerprint(kwargs.get("pristine_model_dir"))}
            key = digest(binding)
            path = self.directory/f"{key}.json"
            started = time.perf_counter()
            if path.exists():
                stored = json.loads(path.read_text())
                if stored["binding"] != jsonable(binding):
                    raise ValueError("Cached fit binding mismatch")
                payload = stored["payload"]
                origin = "cached_fresh_observable_fit"
            else:
                payload = original(**kwargs)
                write_json(path, {"binding": binding, "payload": payload,
                    "source_role": "fresh_real_multiscan_estimator_from_observable_history",
                    "wall_seconds": time.perf_counter()-started})
                origin = "fresh_real_observable_fit"
            self.calls.append({"cache_key": key, "path": str(path), "origin": origin,
                "wall_seconds": time.perf_counter()-started,
                "supplied_scan_indices": [s["scan_index"] for s in kwargs["scans"]],
                "selected_scan_indices": payload.get("selected_scan_indices"),
                "success": payload.get("success", False)})
            return deepcopy(payload)

        provider._memoized_hif_multiscan = fitted


def make_environment(cache, *, alpha_grid_size=BC0_HIF_ALPHA_GRID_SIZE, r_grid_size=BC0_HIF_R_GRID_SIZE,
                     max_scans=BC0_HIF_MAX_SCANS, max_steps=40, normalized_residual_threshold=None):
    provider = MatpowerDeploymentProviders(chi2_alpha=BC0_CHI2_ALPHA, normalized_residual_threshold=normalized_residual_threshold,
        evidence_profile="auxiliary_diagnostics",
        parameter_ranking_dominance_threshold=BC0_PARAMETER_RANKING_DOMINANCE_THRESHOLD,
        branch_first_partial=False, hif_alpha_grid_size=alpha_grid_size, hif_r_grid_size=r_grid_size,
        hif_max_scans=max_scans)
    cache.attach(provider)
    env = TransactionalPSSEEnv(**provider.env_kwargs(), production_dataset_mode=True,
                              max_steps=max_steps, history_window=4)
    return env, provider


def runtime_state(env, provider, history):
    state = env.store.get_state(env.store.active_state_id)
    state["policy_observation"] = env.get_policy_observation(history).as_dict()
    state["case"] = provider._case_path(state)
    return state


def run_episode(execution, cache, *, fit_only=False, **settings):
    execution = deepcopy(execution)
    assert_observable(execution)
    target = current_scan(execution["metadata"])
    first_call = len(cache.calls)
    env, provider = make_environment(cache, **settings)
    candidate_quality_events = []
    original_label = env.candidate_quality_oracle.label_candidate

    def audited_label(**kwargs):
        if kwargs.get("hidden_truth") is not None:
            raise ValueError("Runtime candidate assessment unexpectedly received hidden truth")
        assessment = original_label(**kwargs)
        candidate_quality_events.append({
            **{key: deepcopy(kwargs[key]) for key in
               ("parent_state", "source_action", "candidate_state", "verification_output")},
            "assessment": assessment.as_dict(), "hidden_truth_supplied": False})
        return assessment

    env.candidate_quality_oracle.label_candidate = audited_label
    env.reset(execution)
    expert = ExpertPolicyOracle(process_oracle=env.process_oracle, candidate_oracle=env.candidate_quality_oracle)
    history, events, error = [], [], None
    started = time.perf_counter()
    for step in range(settings.get("max_steps", 40)):
        observation = env.get_policy_observation(history).as_dict()
        validate_policy_payload(observation)
        selection = select_observable_expert_actions(policy_observation=observation, expert_oracle=expert)
        action = deepcopy(selection.preferred_action)
        if not action:
            error = "observable_expert_returned_no_action"
            break
        before = env.store.get_state(env.store.active_state_id)
        try:
            env.assert_training_decision_evidence(action)
            _, output = env.step(action)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            events.append({"step": step, "action": action, "execution_error": error})
            break
        after = env.store.get_state(env.store.active_state_id)
        events.append({"step": step, "action": action, "policy_observation": observation,
            "tool_output": output, "selection_basis": selection.selection_basis,
            "active_measurement_writes": np.flatnonzero(np.asarray(before["measurements"]) != np.asarray(after["measurements"])).tolist()})
        history.append({"action": action, "tool_output": output})
        print(f"{execution['scenario_id']} step={step} {action['tool']} {output.get('execution_status')}", flush=True)
        if env.terminal or fit_only and action["tool"] == ESTIMATE_HIF_MULTISCAN_FROM_PATH:
            break
    else:
        error = "episode_action_budget_exhausted"
    state = runtime_state(env, provider, history)
    final_observation = state["policy_observation"]
    try:
        prediction = conditioned_prediction(state, {})
        prediction_error = None
    except Exception as exc:
        prediction, prediction_error = None, f"{type(exc).__name__}: {exc}"
    calls = deepcopy(cache.calls[first_call:])
    if any(target["scan_index"] in call["supplied_scan_indices"] for call in calls):
        raise ValueError("Current acquisition leaked into HIF fitting inputs")
    return {"scenario_id": execution["scenario_id"], "input_execution_sha256": digest(execution),
        "source_role": "fresh_real_provider_and_observable_expert_on_frozen_execution",
        "current_scan_binding": target, "fit_calls": calls, "events": events,
        "candidate_quality_events": candidate_quality_events,
        "terminal": env.terminal, "terminal_outcome": env.terminal_outcome, "error": error,
        "final_measurements": state["measurements"], "final_case": state["case"],
        "final_policy_observation": final_observation, "prediction": prediction,
        "prediction_error": prediction_error, "wall_seconds": time.perf_counter()-started,
        "fit_only": fit_only, "runtime_truth_supplied": False}


def strict_offline_audit(envelope, result):
    scenario = {**deepcopy(envelope["execution"]), **deepcopy(envelope["grouping"]),
                **deepcopy(envelope["audit"]["truth"]), "release_audit": deepcopy(envelope["audit"]["release_audit"])}
    return audit_truth_audited_task_success(scenario, result["final_policy_observation"],
        actual_terminal=result["terminal"], actual_terminal_outcome=result["terminal_outcome"],
        active_physical_state={"case": result["final_case"], "measurements": result["final_measurements"]},
        case_loader=_load_python_case, evaluator_error=result["error"])


def offline_meter_audit(envelope, result, *, extra_index=None):
    """Hidden truth enters ONLY after run_episode returns its final state."""
    original = np.asarray(envelope["execution"]["measurements"])
    proposed = np.asarray(result["final_measurements"])
    truth = envelope["audit"]["truth"]
    targets = {int(row["index"]) for row in truth.get("true_measurement_errors", [])}
    if extra_index is not None:
        targets.add(int(extra_index))
    # The stored clean_measurements may be event-free. Restore only the known
    # meter overlays to obtain the noisy HIF-present acquisition reference.
    reference = original.copy()
    for fault in truth.get("true_measurement_errors", []):
        reference[int(fault["index"])] = float(fault["clean"])
    absolute = float(envelope["audit"]["release_audit"].get("tolerances", {}).get("measurement_abs", 0.))
    changed = set(np.flatnonzero(proposed != original).tolist())
    errors = {str(index): abs(float(proposed[index]-reference[index])) for index in sorted(targets)}
    return {"changed_indices": sorted(changed), "truth_target_indices": sorted(targets),
        "off_target_write_indices": sorted(changed-targets), "exact_write_support": changed == targets,
        "target_absolute_errors_pu": errors, "frozen_measurement_abs_tolerance": absolute,
        "reference_basis": "initial_acquisition_with_each_meter_overlay_restored_to_recorded_preoverlay_value",
        "meter_recovery": bool(changed == targets and all(value <= absolute for value in errors.values())),
        "truth_used_in_runtime": False, "physical_hif_removal_claimed": False}


def overlap_probe(execution, prediction, *, bias_sigma=10.):
    """Choose an overlapping channel from predicted effects, never injected truth."""
    active = np.asarray(execution["measurements"], dtype=float)
    sigma = np.asarray(execution["metadata"]["sigma_z"], dtype=float)
    effect = np.asarray(prediction["measurement_effect"], dtype=float)
    center = np.asarray(prediction["predicted_hif_measurements"], dtype=float)
    index = int(np.argmax(np.abs(effect)/sigma))
    sign = 1. if active[index] >= center[index] else -1.
    mixed = active.copy()
    mixed[index] += sign*bias_sigma*sigma[index]
    delta = (mixed-effect)-(active-effect)
    decision = diagnose_conditioned_meter_errors(mixed, center, sigma,
        prediction_lower=prediction["prediction_lower"], prediction_upper=prediction["prediction_upper"],
        event_effect=effect, detection_sigma=5., max_envelope_width_sigma=2.)
    changed = deepcopy(execution)
    changed["scenario_id"] += "_controlled_overlap"
    changed["measurements"] = mixed.tolist()
    scan_index = current_scan(changed["metadata"])["scan_index"]
    changed["metadata"]["hif_runtime"]["z_obs"] = mixed.tolist()
    for scan in changed["metadata"]["hif_scan_window"]["scans"]:
        if scan["scan_index"] == scan_index:
            scan["z_obs"] = mixed.tolist()
    evidence = {"target_index": index, "selection_basis": "largest_absolute_predicted_HIF_effect_in_sensor_sigma",
        "predicted_effect_sigma": float(abs(effect[index])/sigma[index]),
        "material_overlap": bool(abs(effect[index]) >= sigma[index]),
        "injected_sigma": sign*bias_sigma, "injected_delta_pu": float(mixed[index]-active[index]),
        "compensated_delta_pu": float(delta[index]),
        "same_channel_error_retained": bool(np.allclose(delta, mixed-active, atol=1e-14, rtol=0)),
        "candidate_contains_injected_index": index in decision["candidate_indices"],
        "conditional_decision": decision,
        "prediction_and_fit_held_fixed": True, "truth_used_to_choose_target": False}
    return changed, evidence


def paired_hif_control(envelope):
    """Construct a dependent pure-HIF control; truth never enters its runtime."""
    control = deepcopy(envelope)
    execution = control["execution"]
    measurements = list(execution["measurements"])
    for fault in control["audit"]["truth"]["true_measurement_errors"]:
        measurements[int(fault["index"])] = float(fault["clean"])
    execution["scenario_id"] += "_paired_hif_control"
    execution["measurements"] = measurements
    scan_index = current_scan(execution["metadata"])["scan_index"]
    execution["metadata"]["hif_runtime"]["z_obs"] = list(measurements)
    for scan in execution["metadata"]["hif_scan_window"]["scans"]:
        if scan["scan_index"] == scan_index:
            scan["z_obs"] = list(measurements)
    control["audit"]["truth"]["true_measurement_errors"] = []
    control["grouping"]["scenario_family"] = "hif"
    control["grouping"]["error_cardinality"] = 1
    return control


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-scenarios", type=Path, default=REPO/"output/hif_continuation_fix_20260922/frozen_mixed_scenarios.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fit-cache-dir", type=Path, default=REPO/"output/hif_continuation_fix_20260922/fresh_fit_cache")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-scans", type=int, default=BC0_HIF_MAX_SCANS)
    parser.add_argument("--alpha-grid-size", type=int, default=BC0_HIF_ALPHA_GRID_SIZE)
    parser.add_argument("--r-grid-size", type=int, default=BC0_HIF_R_GRID_SIZE)
    parser.add_argument("--normalized-residual-threshold", type=float, default=None)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--fit-only", action="store_true")
    parser.add_argument("--controlled-overlap", action="store_true")
    parser.add_argument("--paired-hif-controls", action="store_true")
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError("Use a new output directory; prior probes are preserved")
    args.output_dir.mkdir(parents=True)
    rows = json.loads(args.frozen_scenarios.read_text(encoding="utf-8-sig"))
    rows = rows[:args.limit] if args.limit else rows
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(frozen_source_sha256=hashlib.sha256(args.frozen_scenarios.read_bytes()).hexdigest(),
        evidence_profile="auxiliary_diagnostics",
        chi_square_alpha=BC0_CHI2_ALPHA, normalized_residual_threshold=args.normalized_residual_threshold, conditional_meter_sigma=5.,
        max_prediction_envelope_width_sigma=2., runtime_truth_source="none; execution envelope only")
    implementation = {"scripts/verify_hif_continuation.py", "psse_env/providers/matpower.py",
        "psse_env/providers/hif_continuation.py", "psse_env/transactional_env.py",
        "psse_env/dagger/release_audit.py", "psse_env/dagger/release_factories.py", "mcp_server/matpower_server.py"}
    implementation.update(str(path.relative_to(REPO)).replace('\\', '/') for path in (REPO/"psse_env/oracle").glob("*.py")
                          if not path.name.startswith("test_"))
    config["runtime_source_sha256"] = {name: hashlib.sha256((REPO/name).read_bytes()).hexdigest()
                                       for name in sorted(implementation)}
    config["git_head"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    config["pristine_model_sha256"] = model_fingerprint(None)
    write_json(args.output_dir/"configuration.json", config)
    cache = ObservableFitCache(args.fit_cache_dir)
    results = []
    settings = {key: getattr(args, key) for key in ("max_scans", "alpha_grid_size", "r_grid_size", "max_steps", "normalized_residual_threshold")}
    with threadpool_limits(limits=1):
        for envelope in rows:
            execution = envelope["execution"]
            result = run_episode(execution, cache, fit_only=args.fit_only, **settings)
            result["offline_meter_audit"] = offline_meter_audit(envelope, result)
            result["strict_task_audit"] = strict_offline_audit(envelope, result)
            write_json(args.output_dir/f"{execution['scenario_id']}.json", result)
            if args.controlled_overlap and result["prediction"] is not None and not args.fit_only:
                variant, overlap = overlap_probe(execution, result["prediction"])
                second = run_episode(variant, cache, **settings)
                audit_envelope = deepcopy(envelope)
                audit_envelope["execution"] = variant
                audit_truth = audit_envelope["audit"]["truth"]
                index = overlap["target_index"]
                if index not in {int(row["index"]) for row in audit_truth["true_measurement_errors"]}:
                    audit_truth["true_measurement_errors"].append({"index": index,
                        "clean": execution["measurements"][index], "observed": variant["measurements"][index]})
                second["offline_meter_audit"] = offline_meter_audit(audit_envelope, second, extra_index=overlap["target_index"])
                second["strict_task_audit"] = strict_offline_audit(audit_envelope, second)
                second["controlled_overlap"] = overlap
                baseline_keys = {item["cache_key"] for item in result["fit_calls"]}
                variant_keys = {item["cache_key"] for item in second["fit_calls"]}
                overlap["actual_runtime_fit_cache_reused"] = bool(baseline_keys and variant_keys == baseline_keys
                    and all(item["origin"] == "cached_fresh_observable_fit" for item in second["fit_calls"]))
                write_json(args.output_dir/f"{variant['scenario_id']}.json", second)
                result["controlled_overlap_result"] = {"terminal_outcome": second["terminal_outcome"],
                    "offline_meter_audit": second["offline_meter_audit"], "probe": overlap,
                    "strict_task_audit": second["strict_task_audit"], "fit_calls": second["fit_calls"]}
            if args.paired_hif_controls and not args.fit_only:
                control = paired_hif_control(envelope)
                checked = run_episode(control["execution"], cache, **settings)
                checked["offline_meter_audit"] = offline_meter_audit(control, checked)
                checked["strict_task_audit"] = strict_offline_audit(control, checked)
                checked["control_construction"] = {"truth_used_only_for_experimental_overlay_removal": True,
                    "original_sensor_noise_retained": True, "independent_population_false_alarm_claim": False,
                    "dependent_parent": execution["scenario_id"]}
                write_json(args.output_dir/f"{control['execution']['scenario_id']}.json", checked)
                result["paired_hif_control_result"] = {"terminal_outcome": checked["terminal_outcome"],
                    "offline_meter_audit": checked["offline_meter_audit"], "strict_task_audit": checked["strict_task_audit"],
                    "fit_calls": checked["fit_calls"]}
            results.append(result)
            write_json(args.output_dir/"summary.json", {"attempted_roots": len(results), "requested_roots": len(rows),
                "completed": len(results) == len(rows),
                "meter_recovered_roots": sum(r["offline_meter_audit"]["meter_recovery"] for r in results),
                "strict_task_passed_roots": sum(r["strict_task_audit"]["status"] == "passed" for r in results),
                "off_target_writes": sum(len(r["offline_meter_audit"]["off_target_write_indices"]) for r in results),
                "fit_only": args.fit_only, "roots": [{"scenario_id": r["scenario_id"], "terminal": r["terminal"],
                    "terminal_outcome": r["terminal_outcome"], "steps": len(r["events"]), "error": r["error"],
                    "meter_audit": r["offline_meter_audit"], "fit_calls": r["fit_calls"],
                    "strict_task_audit": r["strict_task_audit"],
                    "controlled_overlap": r.get("controlled_overlap_result"),
                    "paired_hif_control": r.get("paired_hif_control_result")} for r in results]})
    write_json(args.output_dir/"source_stability.json", {"runtime_source_sha256_before": config["runtime_source_sha256"],
        "runtime_source_sha256_after": {name: hashlib.sha256((REPO/name).read_bytes()).hexdigest()
                                       for name in config["runtime_source_sha256"]},
        "pristine_model_sha256_after": model_fingerprint(None)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
