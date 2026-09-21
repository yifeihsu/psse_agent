import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_unbalance_generator_uses_wls_convention_phase_a_vm_and_paired_reference(tmp_path):
    output = tmp_path / "imb"
    subprocess.run([sys.executable, str(ROOT / "Transmission/generate_measurements_imbalance.py"), "--out", str(output),
                    "--n-imbalance", "2", "--n-no-error", "1", "--seed", "5"], check=True, cwd=ROOT,
                   capture_output=True, text=True)
    meta = json.loads((output / "meta.json").read_text())
    assert meta["measurement_convention"]["shunt_convention"] == "ybus"
    assert meta["operator_vm_channel"]["semantics"] == "phase_a_line_to_neutral_voltage_magnitude_pu"
    assert meta["telemetry_base_semantics"] == "physical_local_bases"
    assert meta["imbalance"]["balanced_reference"] == "opendss_same_operating_point"
    rows = [json.loads(line) for line in (output / "samples.jsonl").read_text().splitlines()]
    faults = [row for row in rows if row["scenario"] == "three_phase_imbalance"]
    assert len(faults) == 2 and len(rows) == 3
    control = next(row for row in rows if row["scenario"] == "no_error")
    assert control["measurement_convention"]["shunt_convention"] == "ybus"
    for row in faults:
        assert row["measurement_convention"]["shunt_convention"] == "ybus"
        assert row["balanced_reference"] == "opendss_same_operating_point"
        assert row["z_true_semantics"].startswith("balanced_same_operating_point")
        # The bus-9 capacitor no longer separates the exported injection from the balanced reference.
        assert abs(row["z_clean"][36] - row["z_true"][36]) < 0.01
        gap = max(abs(a - b) for a, b in zip(row["z_true"], row["z_clean"]))
        opf_gap = max(abs(a - b) for a, b in zip(row["z_reference_opf"], row["z_clean"]))
        assert gap < 0.15 and opf_gap > gap
        assert row["three_phase_voltages"][0]["kvbase_ln"] == 69 / math.sqrt(3)
        assert row["three_phase_voltages"][5]["kvbase_ln"] == 13.8 / math.sqrt(3)
        assert row["three_phase_branch_currents"][0]["ibase_from_a"] == 100e6 / 3 / (69e3 / math.sqrt(3))
        assert "noise_contract" in row and len(row["z_obs"]) == 122
