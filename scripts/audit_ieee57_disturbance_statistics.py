"""Independently cross-tabulate complete IEEE 57 disturbance run artifacts.

Reads final raw results as a stream; never imports or changes the experiment
implementation. Unknown/failed WLS results stay separate from negative alarms.
"""
from __future__ import annotations

import argparse
import collections
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def audit(output_dir: str | Path) -> dict:
    root=Path(output_dir).resolve(strict=True)
    paths=sorted(root.glob('*/results.json'))
    assert len(paths)==4 and (root/'run_receipt.json').exists(), 'Full run has not completed'
    receipt=json.loads((root/'run_receipt.json').read_text())
    assert receipt['all_sources_unchanged_during_run'] is True

    def iter_rows(path):
        decoder=json.JSONDecoder()
        with path.open(encoding='utf-8') as handle:
            buffer=''
            started=False
            eof=False
            while True:
                buffer=buffer.lstrip()
                if not started:
                    if not buffer:
                        buffer=handle.read(1024*1024)
                        if not buffer: raise ValueError('empty results')
                        continue
                    if buffer[0]!='[': raise ValueError('results must be an array')
                    buffer=buffer[1:]
                    started=True
                buffer=buffer.lstrip(' \t\r\n,')
                if buffer.startswith(']'): return
                try:
                    row,end=decoder.raw_decode(buffer)
                except json.JSONDecodeError:
                    chunk=handle.read(1024*1024)
                    if not chunk: raise
                    buffer+=chunk
                    continue
                buffer=buffer[end:]
                yield row

    families=collections.Counter()
    measured=collections.Counter()
    models=collections.Counter()
    roots=collections.defaultdict(set)
    wls=collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    phase=collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    cross=collections.defaultdict(collections.Counter)
    misses=collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
    miss_bus=collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    strength=collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    healthy_model=collections.defaultdict(collections.Counter)
    failure_ids=[]
    row_ids=set()

    def correct_phase(row,diagnostic):
        truth=row.get('truth',{})
        if row['family']=='hif':
            candidate=diagnostic.get('hif_candidate') or {}
            return (diagnostic.get('classification')=='hif_like_branch_mismatch'
                and not diagnostic.get('ambiguous')
                and candidate.get('branch_row0')==truth.get('branch_row0')
                and candidate.get('phase')==truth.get('phase'))
        if row['family']=='unbalance':
            candidate=diagnostic.get('unbalance_candidate') or {}
            return (diagnostic.get('classification')=='load_unbalance'
                and not diagnostic.get('ambiguous')
                and candidate.get('bus')==truth.get('bus'))
        return False

    for path in paths:
        count=0
        for row in iter_rows(path):
            count+=1
            assert row['scenario_id'] not in row_ids
            row_ids.add(row['scenario_id'])
            family=row['family']
            model=row['model_id']
            families[family]+=1
            if row.get('observations_path'):
                assert (root/row['observations_path']).is_file(), 'Recorded observations are missing'
                measured[family]+=1
            models[model]+=1
            roots[family].add(row.get('physical_root_fingerprint') or row['physical_root'])
            if row.get('execution_failure'): failure_ids.append(row['scenario_id'])
            for mode in ('exact','noisy'):
                result=row.get('wls',{}).get(mode,{})
                c=wls[family][mode]
                c['total']+=1
                known=result.get('converged') is True and isinstance(result.get('alarm'),bool)
                if not known:
                    c['unknown_or_failed']+=1
                    continue
                c['known']+=1
                chi=result.get('chi_square_alarm') is True
                nr=result.get('normalized_residual_alarm') is True
                c['chi_square_alarm']+=int(chi)
                c['normalized_residual_alarm']+=int(nr)
                c['both_alarms']+=int(chi and nr)
                c['nr_only']+=int(nr and not chi)
                c['chi_square_only']+=int(chi and not nr)
                c['either_alarm']+=int(chi or nr)
                c['neither_alarm']+=int(not(chi or nr))
                assert result['alarm']==(chi or nr)
                if family=='healthy' and mode=='noisy':
                    healthy_model[model]['noisy_rows']+=1
                    healthy_model[model]['wls_chi_square_false_positive']+=int(chi)
                    healthy_model[model]['wls_nr_false_positive']+=int(nr)
                    healthy_model[model]['wls_nr_only_false_positive']+=int(nr and not chi)
                    healthy_model[model]['wls_union_false_positive']+=int(chi or nr)
            for mode in ('exact','noisy_nominal','noisy_precision_sensitivity'):
                diagnostic=row.get('phase_diagnostics',{}).get(mode,{})
                c=phase[family][mode]
                c['total']+=1
                c['anomaly_detected']+=int(diagnostic.get('anomaly_detected') is True)
                c['ambiguous']+=int(diagnostic.get('ambiguous') is True)
                c['correct_localization']+=int(correct_phase(row,diagnostic))
                c['classification:'+str(diagnostic.get('classification','unavailable'))]+=1
                if family=='healthy':
                    healthy_model[model]['phase_false_positive:'+mode]+=int(diagnostic.get('anomaly_detected') is True)
                if family in ('hif','unbalance'):
                    axis=('Rpu='+str(row['truth']['resistance_pu'])) if family=='hif' else ('delta='+str(row['truth']['delta']))
                    strength[family][axis]['total']+=int(mode=='noisy_nominal')
                    strength[family][axis]['correct:'+mode]+=int(correct_phase(row,diagnostic))
                if family=='unbalance' and not correct_phase(row,diagnostic):
                    delta=str(row['truth']['delta'])
                    bus=row['truth']['bus']
                    misses[model][mode][delta].append(bus)
                    miss_bus[mode][delta][str(bus)]+=1
            if family in ('hif','unbalance'):
                diag=row.get('phase_diagnostics',{}).get('noisy_nominal',{})
                w=row.get('wls',{}).get('noisy',{})
                correct=correct_phase(row,diag)
                c=cross[family]
                c['total']+=1
                c['correct_phase_localization']+=int(correct)
                c['correct_phase_but_wls_no_alarm']+=int(correct and w.get('alarm') is False)
                c['correct_phase_and_wls_alarm']+=int(correct and w.get('alarm') is True)
                c['correct_phase_and_wls_unknown']+=int(correct and w.get('alarm') is None)
                c['phase_not_correct_but_wls_alarm']+=int(not correct and w.get('alarm') is True)
        assert count==373,(path,count)

    def ordinary(value):
        if isinstance(value,dict): return {str(k):ordinary(v) for k,v in value.items()}
        if isinstance(value,set): return sorted(value)
        if isinstance(value,list): return [ordinary(v) for v in value]
        return value

    result=ordinary({
        'contract':'independent_raw_results_wls_phase_crosscheck_v1',
        'audited_at_utc':datetime.now(timezone.utc).isoformat(),
        'run_complete':True,
        'fault_attempt_count':families['hif']+families['unbalance'],
        'fault_measurement_count':measured['hif']+measured['unbalance'],
        'measurement_counts':measured,
        'auditor_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'all_runtime_sources_unchanged':True,
        'definitions':{
           'correct_hif':'classification hif_like_branch_mismatch, not ambiguous, matching branch row and phase',
           'correct_unbalance':'classification load_unbalance, not ambiguous, matching bus',
           'wls_known':'converged true and alarm is boolean',
           'healthy_replicates':'100 instrument realizations per physical healthy model, 400 rows over 4 physical roots',
           'exact_phase_threshold':'no added random noise but uses nominal measurement-uncertainty configuration',
        },
        'family_counts':families,
        'model_counts':models,
        'physical_root_counts':{k:len(v) for k,v in roots.items()},
        'execution_failure_ids':failure_ids,
        'wls':wls,
        'phase':phase,
        'nominal_phase_wls_cross':cross,
        'healthy_by_model':healthy_model,
        'unbalance_not_correct_buses_by_model_mode_delta':misses,
        'unbalance_not_correct_bus_frequency_by_mode_delta':miss_bus,
        'strength':strength,
    })
    out=root/'independent_wls_phase_audit.json'
    assert not out.exists(), 'Independent audit already exists'
    out.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('family_counts','physical_root_counts','execution_failure_ids','wls','nominal_phase_wls_cross','healthy_by_model','strength')},indent=2))
    print('audit_path',out)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    audit(args.output_dir)


if __name__ == "__main__":
    main()
