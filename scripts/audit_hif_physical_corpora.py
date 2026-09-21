"""Read-only physical-corpus units, fixed-threshold WLS and admission audit."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys

for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[name] = "1"
import opendssdirect  # Load native DSS before the model/provider dependency tree.
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from psse_env.providers.matpower import MatpowerDeploymentProviders
from psse_env.providers.scenario_generator import Round0ScenarioGenerator, ScenarioRejected
from three_phase_nlm.hif_parameter_estimator import _simulate_base, _resolve_model_dir
from three_phase_nlm.hif_units import label_model_ohm, label_physical_ohm, label_local_kv_ll


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    provider = MatpowerDeploymentProviders(chi2_alpha=.01, normalized_residual_threshold=4.)
    model = _resolve_model_dir(None, "case14")
    cells, cohorts = defaultdict(list), {}
    failures = []

    def fit(z, sigma, identity):
        try:
            solved = provider._solve({"state_id": identity, "case": "case14", "measurements": z,
                                     "metadata": {"sigma_z": sigma}, "policy_observation": {}})
            metrics = provider._wls_detection_metrics(solved)
            if not solved["payload"]["success"]:
                raise ValueError("WLS did not converge")
            return {"success": True, "alarm": not metrics["no_material_anomaly_remaining"], **metrics}
        except Exception as exc:
            failures.append({"identity": identity, "error": str(exc)})
            return {"success": False, "alarm": None, "error": str(exc)}

    with (args.out / "wls_observations.jsonl").open("w") as observations:
        for directory in args.corpus_dirs:
            path = directory / "samples.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            generator = Round0ScenarioGenerator(seed=20260920, hif_sample_paths=[path],
                chi2_alpha=.01, normalized_residual_threshold=4., hif_max_scans=10,
                waveform_signature_mode={"hif": "discovered"})
            counts, classes, bands, detectability, rejections = Counter(), Counter(), Counter(), Counter(), Counter()
            admission = []
            for ordinal, row in enumerate(rows):
                identity = f"{directory.name}:{ordinal}"
                sigma = row["sigma_z"]
                np.testing.assert_allclose(sigma, [.001]*14+[.01]*108, rtol=0, atol=1e-15)
                if row["scenario"] == "no_error":
                    healthy = fit(row["z_obs"], sigma, identity)
                    counts["healthy_rows"] += 1
                    counts["healthy_alarms"] += healthy["alarm"] is True
                    observations.write(json.dumps({"cohort": directory.name, "kind": "healthy", "id": row["id"], "wls": healthy})+'\n')
                    continue
                label = row["label"]
                resistance, kv = label_physical_ohm(label), label_local_kv_ll(label)
                assert label["resistance_units"] == "ohm_local_base"
                assert np.isclose(resistance, label["r_hif_pu"]*kv**2/100)
                assert np.isclose(label_model_ohm(label), label["r_hif_pu"]*.01)
                counts["hif_windows"] += 1
                classes[label["resistance_class"]] += 1
                band = "100-200" if 100 <= resistance < 200 else "200-500" if 200 <= resistance < 500 else "500-1000" if 500 <= resistance <= 1000 else "outside_main_bands"
                bands[band] += 1
                detectability[str(label.get("expected_detectability"))] += 1
                top = [entry["branch_row0"] for entry in row["nlm_diagnostic"].get("top_hif_groups", [])]
                entry = {"id": row["id"], "cohort": directory.name, "kv": kv, "ohm": resistance,
                         "nlm_top1": bool(top and top[0] == label["branch_row0"]),
                         "nlm_top3": label["branch_row0"] in top[:3], "scan_alarms": 0, "scan_count": len(row["scans"])}
                for scan in row["scans"]:
                    assert scan["measurement_convention"]["shunt_convention"] == "ybus"
                    assert scan["sigma_z"] == sigma
                    result = fit(scan["z_obs"], sigma, f"{identity}:{scan['scan_index']}")
                    entry["scan_alarms"] += result["alarm"] is True
                    record = {"cohort": directory.name, "kind": "hif", "id": row["id"],
                              "scan_index": scan["scan_index"], "kv": kv, "ohm": resistance, "wls": result}
                    if scan["scan_index"] == 0:
                        entry["first_scan_alarm"] = result["alarm"] is True
                        # Same operating point and exact same noise, with HIF absent.
                        baseline = _simulate_base(model, op_point=scan["op_point"], shunt_convention="ybus")
                        paired_z = np.asarray(baseline["z"]) + np.asarray(scan["z_obs"]) - np.asarray(scan["z_clean"])
                        null = fit(paired_z.tolist(), sigma, identity+":paired_healthy")
                        entry["paired_healthy_alarm"] = null["alarm"] is True
                        entry["new_first_scan_alarm"] = entry["first_scan_alarm"] and null["alarm"] is False
                        record["paired_healthy_wls"] = null
                    observations.write(json.dumps(record, allow_nan=False)+'\n')
                counts["first_scan_alarms"] += entry["first_scan_alarm"]
                counts["paired_healthy_alarms"] += entry["paired_healthy_alarm"]
                counts["new_first_scan_alarms"] += entry["new_first_scan_alarm"]
                counts["nlm_top1"] += entry["nlm_top1"]
                counts["nlm_top3"] += entry["nlm_top3"]
                if "sweep_eval" in directory.name:
                    cells[kv, resistance].append(entry)
                try:
                    scenario = generator._hif_scenario(row, ordinal)
                    assert scenario["metadata"]["measurement_convention"] == row["measurement_convention"]
                    counts["scenario_admitted"] += 1
                    admission.append({"id": row["id"], "admitted": True})
                except ScenarioRejected as exc:
                    rejections[exc.reason] += 1
                    admission.append({"id": row["id"], "admitted": False, "reason": exc.reason})
                if counts["hif_windows"] % 20 == 0:
                    observations.flush()
                    print(json.dumps({"cohort": directory.name, "windows_audited": counts["hif_windows"]}), flush=True)
            cohorts[directory.name] = {"counts": dict(counts), "resistance_class_counts": dict(classes),
                "main_band_counts": dict(bands), "expected_detectability_counts": dict(detectability), "admission_rejections": dict(rejections),
                "admission_rows": admission, "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            write_json(args.out/'cohorts_progress.json', cohorts)
    summary = {"cohorts": cohorts, "failures": failures, "complete": True, "success": not failures,
        "settings": {"chi_square_alpha": .01, "normalized_residual_threshold": 4., "sigma_vm": .001,
            "sigma_power": .01, "admission_mode": "recoverable/discovered", "admission_anomaly_margin": 1.25},
        "scope": "Fixed synthetic corpus, reference-scan and all-scan WLS; paired healthy first scan; scenario admission is not expert success or training completion.",
        "sweep": [{"kv": kv, "ohm": ohm, "windows": len(items),
            **{key: sum(row[key] for row in items) for key in ('nlm_top1','nlm_top3','first_scan_alarm','paired_healthy_alarm','new_first_scan_alarm','scan_alarms','scan_count')}}
            for (kv,ohm),items in sorted(cells.items())]}
    write_json(args.out/'summary.json', summary)
    print(json.dumps({"success": summary['success'], "cohorts": {key:value['counts'] for key,value in cohorts.items()}}), flush=True)


if __name__ == '__main__':
    main()
