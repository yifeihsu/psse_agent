import json
import math
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
DISPATCH_KEYS = ("generator_dispatch_kw", "voltage_setpoints_pu", "source_voltage_pu")


@pytest.mark.parametrize("dispatch_mode", ["opf", "case14"])
def test_unbalance_generator_uses_wls_convention_phase_a_vm_and_paired_reference(tmp_path, dispatch_mode):
    output = tmp_path / f"imb_{dispatch_mode}"
    subprocess.run([sys.executable, str(ROOT / "Transmission/generate_measurements_imbalance.py"), "--out", str(output),
                    "--n-imbalance", "2", "--n-no-error", "1", "--seed", "5", "--dispatch-mode", dispatch_mode],
                   check=True, cwd=ROOT, capture_output=True, text=True)
    meta = json.loads((output / "meta.json").read_text())
    assert meta["measurement_convention"]["shunt_convention"] == "ybus"
    assert meta["operator_vm_channel"]["semantics"] == "phase_a_line_to_neutral_voltage_magnitude_pu"
    assert meta["telemetry_base_semantics"] == "physical_local_bases"
    assert meta["imbalance"]["balanced_reference"] == "opendss_same_operating_point"
    assert meta["imbalance"]["dispatch_mode"] == dispatch_mode
    assert meta["imbalance"]["dispatch"]["mode"] == dispatch_mode
    rows = [json.loads(line) for line in (output / "samples.jsonl").read_text().splitlines()]
    faults = [row for row in rows if row["scenario"] == "three_phase_imbalance"]
    assert len(faults) == 2 and len(rows) == 3
    control = next(row for row in rows if row["scenario"] == "no_error")
    assert control["measurement_convention"]["shunt_convention"] == "ybus"
    for row in faults:
        assert row["measurement_convention"]["shunt_convention"] == "ybus"
        assert row["balanced_reference"] == "opendss_same_operating_point"
        assert row["z_true_semantics"].startswith("balanced_same_operating_point")
        assert row["dispatch"]["mode"] == dispatch_mode
        # The bus-9 capacitor no longer separates the exported injection from the balanced reference.
        assert abs(row["z_clean"][36] - row["z_true"][36]) < 0.01
        gap = max(abs(a - b) for a, b in zip(row["z_true"], row["z_clean"]))
        opf_gap = max(abs(a - b) for a, b in zip(row["z_reference_opf"], row["z_true"]))
        assert gap < 0.15
        if dispatch_mode == "opf":
            # The OPF's dispatch is the one applied to both OpenDSS solves, so the pypower
            # OPF vector and the balanced OpenDSS reference are the same operating point;
            # what remains is the Vsource's finite short-circuit impedance (about 2e-4 pu
            # in Q, 0.02 SCADA sigma), and the row op_point carries the applied dispatch.
            assert opf_gap < 5e-4
            assert all(key in row["op_point"] for key in DISPATCH_KEYS)
            assert set(row["op_point"]["generator_dispatch_kw"]) == {"b2", "b3", "b6", "b8"}
        else:
            # The checked-in model dispatch (bus 2 at 40 MW, condensers at 3/6/8) differs
            # from the pypower OPF at the same load, so the OPF vector is farther from the
            # unbalanced sensor mean than the paired OpenDSS reference is.
            assert opf_gap > gap
            assert not any(key in row["op_point"] for key in DISPATCH_KEYS)
        assert row["three_phase_voltages"][0]["kvbase_ln"] == 69 / math.sqrt(3)
        assert row["three_phase_voltages"][5]["kvbase_ln"] == 13.8 / math.sqrt(3)
        assert row["three_phase_branch_currents"][0]["ibase_from_a"] == 100e6 / 3 / (69e3 / math.sqrt(3))
        assert "noise_contract" in row and len(row["z_obs"]) == 122
