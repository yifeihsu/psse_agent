#!/usr/bin/env python3
"""Standalone structural/electrical audit. Not the production WLS/DAgger run.

The optional AC checker uses SciPy's root solver for the standard polar AC PF
residual. It does not enforce generator Q/dispatch/voltage limits, run OPF, or
silently discard unsupplied islands. It is intentionally independent of pp.
"""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
import csv
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from Transmission.ieee14_full_topology import (build_full_topology,
    topology_to_matpower,single_flip_audit,write_matpower_case,MODEL_ID)


def admittance(case):
    bus=np.asarray(case['bus'],float); branch=np.asarray(case['branch'],float)
    n=len(bus); lookup={int(row[0]):i for i,row in enumerate(bus)}
    y=np.zeros((n,n),complex); yf=np.zeros((len(branch),n),complex); yt=yf.copy()
    for k,r in enumerate(branch):
        if not r[10]:
            continue
        f,t=lookup[int(r[0])],lookup[int(r[1])]
        z=complex(r[2],r[3])
        if z==0:
            raise ValueError('Zero impedance electrical branch')
        ys=1/z; bc=1j*r[4]/2
        tap=(r[8] if r[8]!=0 else 1)*np.exp(1j*np.deg2rad(r[9]))
        ff=(ys+bc)/abs(tap)**2; tt=ys+bc
        ft=-ys/np.conj(tap); tf=-ys/tap
        yf[k,f]+=ff; yf[k,t]+=ft; yt[k,f]+=tf; yt[k,t]+=tt
        y[f,f]+=ff; y[t,t]+=tt; y[f,t]+=ft; y[t,f]+=tf
    y[np.diag_indices(n)]+=(bus[:,4]+1j*bus[:,5])/float(case['baseMVA'])
    return y,yf,yt


def solve_ac(case):
    from scipy.optimize import root
    bus=np.asarray(case['bus'],float); gen=np.asarray(case['gen'],float)
    branch=np.asarray(case['branch'],float); n=len(bus)
    lookup={int(row[0]):i for i,row in enumerate(bus)}
    refs=np.where(bus[:,1]==3)[0]; pq=np.where(bus[:,1]==1)[0]
    pvpq=np.where(np.isin(bus[:,1],[1,2]))[0]
    active=set(np.where(bus[:,1]!=4)[0]); graph={i:set() for i in active}
    for r in branch:
        if r[10]>0:
            f,t=lookup[int(r[0])],lookup[int(r[1])]
            if f in active and t in active:
                graph[f].add(t); graph[t].add(f)
    seen=set(refs); todo=list(refs)
    while todo:
        i=todo.pop()
        for j in graph.get(i,()):
            if j not in seen:
                seen.add(j); todo.append(j)
    if active-seen:
        raise ValueError(f'Island without reference: buses {[int(bus[i,0]) for i in sorted(active-seen)]}')
    y,yf,yt=admittance(case)
    sp=-(bus[:,2]+1j*bus[:,3])/float(case['baseMVA'])
    for r in gen:
        if r[7]>0:
            sp[lookup[int(r[0])]]+=complex(r[1],r[2])/float(case['baseMVA'])
    vm=bus[:,7].copy(); va=np.deg2rad(bus[:,8])
    def unpack(x):
        a=va.copy(); v=vm.copy(); a[pvpq]=x[:len(pvpq)]; v[pq]=x[len(pvpq):]
        return v*np.exp(1j*a)
    def residual(x):
        v=unpack(x); mis=v*np.conj(y@v)-sp
        return np.r_[mis[pvpq].real,mis[pq].imag]
    x0=np.r_[va[pvpq],vm[pq]]
    sol=root(residual,x0,method='hybr',options={'xtol':1e-10,'maxfev':5000})
    err=float(np.max(np.abs(residual(sol.x))))
    v=unpack(sol.x)
    if err>1e-8 or np.any(abs(v)<.1):
        raise RuntimeError(f'AC residual {err:.3g}; {sol.message}')
    f=np.array([lookup[int(r[0])] for r in branch]); t=np.array([lookup[int(r[1])] for r in branch])
    sf=v[f]*np.conj(yf@v); st=v[t]*np.conj(yt@v)
    return {'V':v,'Sf':sf,'St':st,'residual_pu':err,'solver_success_flag':bool(sol.success)}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case-json',type=Path,default=ROOT/'models/ieee14_full_reference_case.json')
    p.add_argument('--out',type=Path,default=ROOT/'validation')
    p.add_argument('--validate-ac',action='store_true')
    p.add_argument('--export-single-flips',action='store_true')
    args=p.parse_args(argv); args.out.mkdir(parents=True,exist_ok=True)
    case=json.loads(args.case_json.read_text()); m=build_full_topology()
    normal,info=topology_to_matpower(case,model=m)
    arrays={k:bool(np.array_equal(np.asarray(case[k]),normal[k])) for k in ('bus','gen','branch','gencost')}
    report={'model_id':MODEL_ID,'model_fingerprint':m.fingerprint(),
       'connectivity_nodes':len(m.nodes),'switches':len(m.breakers),
       'normal_closed':sum(c.closed for c in m.breakers),
       'normal_open':sum(not c.closed for c in m.breakers),
       'normal_topological_buses':len(normal['bus']),'physical_branches':len(normal['branch']),
       'generator_rows':len(normal['gen']),'reference_matrices_exactly_equal':arrays,
       'yards':dict(Counter(c.yard for c in m.breakers)),
       'scope':'standalone topology/model audit; not production WLS/DAgger or operational feasibility certification',
       'unexecuted':['pandapower adapter integration','MATLAB/MATPOWER runpf','production WLS/expert/DAgger'],
       'q_limits_enforced':False}
    audit=single_flip_audit(m)
    report['single_flip_partition_effects']=dict(Counter(r['effect'] for r in audit))
    report['single_flip_terminal_partition_changes']=sum(r['terminal_partition_changed'] for r in audit)
    y0,_,_=admittance(case); y1,_,_=admittance(normal)
    report['max_normal_ybus_difference']=float(np.max(abs(y0-y1)))
    write_matpower_case(normal,args.out/'case14_full_normal.m','case14_full_normal')
    report['all_single_flip_branch_parameters_preserved']=True
    report['all_single_flip_generator_nonbus_columns_preserved']=True
    report['all_single_flip_load_shunt_totals_preserved']=True
    for rec in audit:
        changed,meta=topology_to_matpower(case,{rec['cb_name']:rec['flipped_closed']},model=m)
        assert np.array_equal(changed['branch'][:,2:],np.asarray(case['branch'])[:,2:])
        assert np.array_equal(changed['gen'][:,1:],np.asarray(case['gen'])[:,1:])
        assert np.allclose(changed['bus'][:,2:6].sum(axis=0),np.asarray(case['bus'])[:,2:6].sum(axis=0),atol=1e-12,rtol=0)
        if args.export_single_flips:
            name='case_'+rec['cb_name']
            write_matpower_case(changed,args.out/'single_flip_cases'/f'{name}.m',name)
        if args.validate_ac:
            try:
                s=solve_ac(changed)
                rec.update(ac_result='solved',ac_residual_pu=s['residual_pu'],ac_error='')
            except (ValueError,RuntimeError) as ex:
                rec.update(ac_result='unsupplied_island' if 'Island' in str(ex) else 'not_converged',
                           ac_residual_pu=None,ac_error=str(ex))
    if args.validate_ac:
        original=solve_ac(case); expanded=solve_ac(normal)
        report['normal_ac']={'reference_residual_pu':original['residual_pu'],
          'full_residual_pu':expanded['residual_pu'],
          'max_voltage_difference_pu':float(np.max(abs(original['V']-expanded['V']))),
          'max_branch_power_difference_pu':float(max(np.max(abs(original['Sf']-expanded['Sf'])),
                                                   np.max(abs(original['St']-expanded['St']))))}
        report['single_flip_ac_results']=dict(Counter(r['ac_result'] for r in audit))
    with (args.out/'single_flip_audit.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(audit[0]));writer.writeheader();writer.writerows(audit)
    (args.out/'single_flip_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    (args.out/'normal_node_to_bus.json').write_text(json.dumps(info,indent=2)+'\n')
    (args.out/'validation_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    if not all(arrays.values()):
        raise SystemExit('Normal-state reference matrix equivalence failed')

if __name__=='__main__':
    main()
