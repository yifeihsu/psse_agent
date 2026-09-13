"""Reproducible pure-logical protocol smoke pilot, not independent SFT data.

python -m psse_env.dagger.logical_pilot --output output/ieee57_logical_adapter_pilot
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from pypower.api import ppoption, runpf
from threadpoolctl import threadpool_limits
from logical_topology.inventory import build_inventory, process_topology
from logical_topology.measurements import build_measurement_inventory, expected_measurements, sample_measurements
from logical_topology.provider import _native
from psse_env.systems import resolve_system
from .logical_adapter import logical_environment_factory, logical_scenario, run_logical_episode


def run_pilot(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    repository = Path(__file__).resolve().parents[2]
    sources = [*sorted((repository/"logical_topology").glob("*.py")),
               Path(__file__).resolve(), Path(__file__).with_name("logical_adapter.py").resolve(),
               Path(__file__).with_name("logical_protocol.py").resolve(),
               repository/"psse_env"/"transactional_env.py", repository/"psse_env"/"state_store.py"]
    hashes = {str(path.relative_to(repository)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    case = resolve_system("case57").load_case()
    inventory = build_inventory("case57")
    sensors = build_measurement_inventory(inventory, "direct")
    branch = inventory["branches"][0]["device_id"]
    outage = inventory["branches"][18]["device_id"]
    coupler = next(row["device_id"] for row in inventory["couplers"] if row["base_bus"] == 4)

    def write(name, value):
        path = output/name
        raw = (json.dumps(_native(value), sort_keys=True, indent=2, allow_nan=False)+"\n").encode()
        path.write_bytes(raw)
        return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}

    inventory_ref = write("inventory.json", inventory)
    sensor_ref = write("sensors.json", sensors)
    write("canonical_case.json", case)
    worlds, rows = {}, []
    for family in ("healthy", "branch_inclusion", "branch_exclusion", "split", "merge", "pair", "truncated_pair"):
        true = dict(inventory["normal_statuses"])
        if family == "merge": true[coupler] = 0
        if family == "branch_exclusion": true[outage] = 0
        parent = hashlib.sha256(json.dumps(true, sort_keys=True).encode()).hexdigest()
        if parent not in worlds:
            physical, success = runpf(process_topology(case, inventory, true)["case"], ppoption(VERBOSE=0, OUT_ALL=0, PF_TOL=1e-10))
            if not success:
                raise RuntimeError(f"physical fixture failed power flow: {family}")
            expected = expected_measurements(case, inventory, true, physical, sensors)
            worlds[parent] = sample_measurements(expected, sensors, noise=False)
        reported = copy.deepcopy(true)
        if family in ("branch_inclusion", "pair", "truncated_pair"): reported[branch] = 0
        if family == "branch_exclusion": reported[outage] = 1
        if family in ("split", "pair", "truncated_pair"): reported[coupler] = 0
        if family == "merge": reported[coupler] = 1
        options = {}
        if family in ("pair", "truncated_pair"):
            options.update(include_pairs=True, pair_devices=[branch, coupler], max_pairs=0 if family == "truncated_pair" else 1)
        env = logical_environment_factory(derived_case_dir=str(output/"derived"), scan_options=options)
        scenario = logical_scenario(env.logical_providers, case=case, inventory=inventory_ref,
            reported_statuses=reported, sensors=sensor_ref, observations=worlds[parent],
            scenario_id=f"ieee57_logical_adapter_{family}", parent_id=parent, true_statuses=true)
        scenario_ref = write(f"{family}_scenario.json", scenario)
        trajectory = run_logical_episode(env, scenario)
        trace_ref = write(f"{family}_trajectory.json", trajectory)
        rows.append({"family": family, "scenario": scenario_ref, "trajectory": trace_ref,
                     "private_audit": trajectory["private_audit"],
                     "actions": [row["action"]["tool"] for row in trajectory["steps"]],
                     "all_actions_succeeded": all(row["output"]["execution_status"] == "success" for row in trajectory["steps"]),
                     "scan_options": options})
    summary = {"contract": "ieee57_logical_adapter_smoke_pilot_v1", "rows": rows,
        "implementation_sha256": hashes,
        "source_files_unchanged_during_run": all(hashlib.sha256(path.read_bytes()).hexdigest() == hashes[str(path.relative_to(repository)).replace("\\", "/")] for path in sources),
        "physical_parent_count": len(worlds), "raw_sensor_count": len(sensors["records"]),
        "population": "noiseless protocol regression fixtures; not independent train/validation/test populations",
        "search_scope": "all 93 single flips; pair fixtures additionally declare exactly one branch+coupler pair",
        "strict_resolved_count": sum(row["private_audit"]["strict_resolved"] for row in rows),
        "inconclusive_count": sum(row["private_audit"]["terminal_outcome"] == "inconclusive" for row in rows),
        "all_actions_succeeded": all(row["all_actions_succeeded"] for row in rows)}
    write("summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        result = run_pilot(args.output)
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))
