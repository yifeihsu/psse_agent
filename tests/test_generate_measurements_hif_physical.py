import json
import math
from pathlib import Path
import subprocess
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
DISPATCH_KEYS=("generator_dispatch_kw","voltage_setpoints_pu","source_voltage_pu")


@pytest.mark.parametrize('units,dispatch_mode', [('ohm','opf'),('ohm','case14'),('pu','opf')])
def test_generated_units_conventions_and_injector_replay(tmp_path, units, dispatch_mode):
    output=tmp_path/f"{units}_{dispatch_mode}"
    flags=['--r-hif-ohm-sweep','500','--voltage-stratum','69kv'] if units=='ohm' else ['--r-hif-pu-min','20','--r-hif-pu-max','200']
    subprocess.run([sys.executable,str(ROOT/'Transmission/generate_measurements_hif_ieee14.py'),'--out',str(output),'--n-hif','2','--n-no-error','1','--seed','7','--scans-per-window','2','--resistance-units',units,'--dispatch-mode',dispatch_mode,*flags],check=True,cwd=ROOT,capture_output=True,text=True)
    rows=[json.loads(line) for line in (output/'samples.jsonl').read_text().splitlines()]
    meta=json.loads((output/'meta.json').read_text())
    assert meta['hif']['dispatch']['mode']==dispatch_mode
    assert meta['hif']['scan_window']['dispatch_mode']==dispatch_mode
    faults=[row for row in rows if row['scenario']=='high_impedance_fault']
    assert len(faults)==2 and len(rows)==3
    for row in faults:
        label=row['label'];assert row['nlm_diagnostic']['success']
        assert row['dispatch']['mode']==dispatch_mode
        assert row['window_metadata']['dispatch_mode']==dispatch_mode
        assert label['kv_ln']==pytest.approx(1/math.sqrt(3))
        if units=='ohm':
            assert label['r_hif_ohm']==500
            assert label['r_hif_pu']==pytest.approx(500/47.61)
            assert label['r_hif_model_ohm']==pytest.approx(500/47.61*.01)
            assert label['resistance_units']=='ohm_local_base'
            assert label['resistance_class']=='weak_hif'
            assert label['expected_detectability']=='weak'
            assert meta['hif']['eligible_branch_row0']==list(range(7))
            assert row['three_phase_voltages'][0]['kvbase_ln']==pytest.approx(69/math.sqrt(3))
            assert row['three_phase_voltages'][5]['kvbase_ln']==pytest.approx(13.8/math.sqrt(3))
            assert abs(row['z_clean'][36]-row['z_true'][36])<.03
            # z_true is the paired balanced OpenDSS solve at scan 0's operating point:
            # it differs from the sensor mean only by the 500 ohm fault itself.
            assert row['balanced_reference']=='opendss_same_operating_point'
            assert row['z_true_semantics'].startswith('balanced_same_operating_point')
            gap=max(abs(a-b) for a,b in zip(row['z_true'],row['z_clean']))
            opf_gap=max(abs(a-b) for a,b in zip(row['z_reference_opf'],row['z_true']))
            assert gap<.05
            if dispatch_mode=='opf':
                # The OPF's dispatch is the one applied to both OpenDSS solves, so the pypower
                # OPF vector and the balanced OpenDSS reference are the same operating point;
                # what remains is the Vsource's finite short-circuit impedance (about 2e-4 pu),
                # and the row op_point carries the applied dispatch.
                assert opf_gap<5e-4
                assert row['dispatch']['solver'] and 'slack_pg_kw' in row['dispatch']
                assert all(key in row['op_point'] for key in DISPATCH_KEYS)
                assert set(row['op_point']['generator_dispatch_kw'])=={'b2','b3','b6','b8'}
                # The OPF dispatches the units at buses 3/6/8 well above the model's 1 kW condensers.
                assert max(row['op_point']['generator_dispatch_kw'][b] for b in ('b3','b6','b8'))>100
            else:
                # The checked-in model dispatch (bus 2 at 40 MW, condensers at 3/6/8) differs
                # from the pypower OPF at the same load, so the OPF vector is farther from the
                # paired OpenDSS reference than the HIF sensor mean is.  The canonical op_point
                # still lists all four units, with the condensers at their model 1 kW values.
                assert opf_gap>gap
                assert row['dispatch']['solver'] is None
                assert max(row['op_point']['generator_dispatch_kw'][b] for b in ('b3','b6','b8'))<10
        else:
            assert label['r_hif_ohm']==pytest.approx(label['r_hif_pu']*.01)
            assert label['resistance_units']=='pu_legacy_normalized_model'
            assert row['balanced_reference']=='pypower_opf' and row['z_true']==row['z_reference_opf']
        for scan in row['scans']:
            assert scan['measurement_convention']['shunt_convention']==('ybus' if units=='ohm' else 'legacy_injection')
    subprocess.run([sys.executable,str(ROOT/'scripts/validate_hif_multiscan_dataset.py'),str(output/'samples.jsonl'),'--meta',str(output/'meta.json'),'--strict-physics'],check=True,cwd=ROOT,capture_output=True,text=True)
