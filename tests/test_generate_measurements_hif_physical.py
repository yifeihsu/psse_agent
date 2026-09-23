import json
import math
from pathlib import Path
import subprocess
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('units', ['ohm','pu'])
def test_generated_units_conventions_and_injector_replay(tmp_path, units):
    output=tmp_path/units
    flags=['--r-hif-ohm-sweep','500','--voltage-stratum','69kv'] if units=='ohm' else ['--r-hif-pu-min','20','--r-hif-pu-max','200']
    subprocess.run([sys.executable,str(ROOT/'Transmission/generate_measurements_hif_ieee14.py'),'--out',str(output),'--n-hif','2','--n-no-error','1','--seed','7','--scans-per-window','2','--resistance-units',units,*flags],check=True,cwd=ROOT,capture_output=True,text=True)
    rows=[json.loads(line) for line in (output/'samples.jsonl').read_text().splitlines()]
    meta=json.loads((output/'meta.json').read_text())
    faults=[row for row in rows if row['scenario']=='high_impedance_fault']
    assert len(faults)==2 and len(rows)==3
    for row in faults:
        label=row['label'];assert row['nlm_diagnostic']['success']
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
            # it differs from the sensor mean only by the 500 ohm fault itself, while
            # the pypower OPF vector (kept as z_reference_opf) carries a different dispatch.
            assert row['balanced_reference']=='opendss_same_operating_point'
            assert row['z_true_semantics'].startswith('balanced_same_operating_point')
            gap=max(abs(a-b) for a,b in zip(row['z_true'],row['z_clean']))
            opf_gap=max(abs(a-b) for a,b in zip(row['z_reference_opf'],row['z_clean']))
            assert gap<.05 and opf_gap>gap
        else:
            assert label['r_hif_ohm']==pytest.approx(label['r_hif_pu']*.01)
            assert label['resistance_units']=='pu_legacy_normalized_model'
            assert row['balanced_reference']=='pypower_opf' and row['z_true']==row['z_reference_opf']
        for scan in row['scans']:
            assert scan['measurement_convention']['shunt_convention']==('ybus' if units=='ohm' else 'legacy_injection')
    subprocess.run([sys.executable,str(ROOT/'scripts/validate_hif_multiscan_dataset.py'),str(output/'samples.jsonl'),'--meta',str(output/'meta.json'),'--strict-physics'],check=True,cwd=ROOT,capture_output=True,text=True)
