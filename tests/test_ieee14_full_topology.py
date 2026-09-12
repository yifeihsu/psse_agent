"""Executable regression tests for the additive full-schematic model."""
import json
from pathlib import Path
import sys
from copy import deepcopy
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from Transmission.ieee14_full_topology import (build_full_topology,parse_status,
    single_flip_audit,topology_to_matpower,BRANCH_PAIRS,write_matpower_case,
    build_nb_ieee14_full)
from scripts.validate_ieee14_full_topology import admittance,solve_ac

@pytest.fixture
def case():
    return json.loads((ROOT/'models/ieee14_full_reference_case.json').read_text())

def test_inventory():
    m=build_full_topology()
    assert (len(m.nodes),len(m.breakers),sum(c.closed for c in m.breakers))==(65,73,53)
    assert len(m.terminals)==40
    assert len({n.planning_bus for n in m.nodes.values()})==14

def test_normal_partition():
    m=build_full_topology(); mapping=m.node_to_bus()
    assert len(m.components())==14
    assert all(mapping[n]==meta.planning_bus for n,meta in m.nodes.items())

def test_no_fictional_7_8_breakers():
    m=build_full_topology()
    assert not any(c.yard in {'7','8'} for c in m.breakers)
    assert m.terminals[7,4]==m.terminals[7,8]==m.terminals[7,9]=='7STAR'
    assert m.terminals[8,7]=='8B'

def test_shared_yard_has_two_normal_buses():
    m=build_full_topology(); mapping=m.node_to_bus()
    assert len([c for c in m.breakers if c.yard=='10_14'])==9
    assert mapping['10B1']!=mapping['14B1']
    assert (10,14) not in m.terminals

@pytest.mark.parametrize('name', ['CB_Y1014_14B_10N1','CB_Y1014_14N2_10B','CB_Y1014_I14_I10'])
def test_shared_yard_merge(name):
    m=build_full_topology(); mapping=m.node_to_bus({name:True})
    assert len(m.components({name:True}))==13
    assert mapping['10B1']==mapping['14B1']

def test_legacy_1_2_3_switch_ids_preserved():
    m=build_full_topology(); ids={c.name for c in m.breakers}
    old=['CB_1_B1_N1','CB_1_N1_N2','CB_1_N2_B2','CB_1_B1_N3','CB_1_N3_N4','CB_1_N4_B2',
         'CB_2R1_2R2','CB_2R2_2R3','CB_2R3_2R4','CB_2R4_2R5','CB_2R5_2R1',
         'CB_3_L32_B1','CB_3_L32_B2','CB_3_L34_B1','CB_3_L34_B2']
    assert set(old)<=ids

@pytest.mark.parametrize('value,expected',[(True,True),(False,False),(0,False),(1,True),
                         ('open',False),('closed',True),(' OPEN ',False)])
def test_status_parsing(value,expected):
    assert parse_status(value) is expected

@pytest.mark.parametrize('value',['false','yes',2,None,0.5])
def test_invalid_status_rejected(value):
    with pytest.raises(ValueError): parse_status(value)

def test_unknown_cb_rejected():
    with pytest.raises(ValueError): build_full_topology().states({'CB_typo':False})

def test_normal_exact_electrical_equivalence(case):
    out,_=topology_to_matpower(case)
    for key in ['bus','gen','branch','gencost']:
        np.testing.assert_array_equal(out[key],case[key])
    np.testing.assert_array_equal(admittance(out)[0],admittance(case)[0])

def test_equipment_not_split(case):
    out,_=topology_to_matpower(case)
    assert len(out['gen'])==len(case['gen'])==5
    assert build_full_topology().equipment['gen'][3]=='3B1'
    assert build_full_topology().equipment['load'][3]=='3B2'

def test_case_input_not_mutated(case):
    old=deepcopy(case)
    topology_to_matpower(case,{'CB_6_B1_B2':False})
    assert case==old

def test_parameterized_operating_point_preserved(case):
    case=deepcopy(case)
    for r in case['bus']: r[2]*=1.07; r[3]*=.91
    case['branch'][7][8]=1.031
    case['branch'][8][9]=2.3
    case['branch'][12][10]=0
    case['gen'][2][3]=57
    out,_=topology_to_matpower(case)
    np.testing.assert_array_equal(out['branch'],case['branch'])
    np.testing.assert_array_equal(out['gen'],case['gen'])
    np.testing.assert_array_equal(out['bus'],case['bus'])

def test_branch_row_order_preserved(case):
    case=deepcopy(case);case['branch']=list(reversed(case['branch']))
    out,_=topology_to_matpower(case)
    np.testing.assert_array_equal(out['branch'],case['branch'])

def test_wrong_reference_rejected(case):
    case=deepcopy(case);case['branch'][0][1]=14
    with pytest.raises(ValueError):topology_to_matpower(case)

def test_switch_noops_not_topology_errors():
    from collections import Counter
    rows=single_flip_audit()
    assert Counter(r['effect'] for r in rows)=={'split':45,'merge':3,'equivalent':25}
    assert sum(r['terminal_partition_changed'] for r in rows)==48

@pytest.mark.parametrize('row',single_flip_audit(),ids=lambda r:r['cb_name'])
def test_all_single_flips_preserve_equipment(case,row):
    out,info=topology_to_matpower(case,{row['cb_name']:row['flipped_closed']})
    assert len(out['bus'])==row['topological_buses']
    assert out['branch'].shape==(20,13)
    np.testing.assert_array_equal(out['branch'][:,2:],np.asarray(case['branch'])[:,2:])
    np.testing.assert_array_equal(out['gen'][:,1:],np.asarray(case['gen'])[:,1:])
    np.testing.assert_allclose(out['bus'][:,2:6].sum(axis=0),np.asarray(case['bus'])[:,2:6].sum(axis=0),atol=1e-12,rtol=0)
    assert len(info['node_to_bus'])==65

def test_open_end_line_charging_retained(case):
    # Isolate the 1--5 terminal at SS5 by opening both of its bus connections.
    out,info=topology_to_matpower(case,{'CB_5_L51_B1':False,'CB_5_L51_B2':False})
    idx=1 # original 1--5 row, charging b=.0492
    assert out['branch'][idx,10]==1
    assert out['branch'][idx,4]==case['branch'][idx][4]==.0492
    terminal=info['node_to_bus']['5|L51']
    assert terminal not in info['inactive_empty_busbars']
    assert out['bus'][terminal-1,1]==1

def test_empty_busbar_retained_but_inactive(case):
    out,info=topology_to_matpower(case,{'CB_5_L51_B2':False,'CB_5_T56_B2':False})
    b=info['node_to_bus']['5B2']
    assert b in info['inactive_empty_busbars']
    assert out['bus'][b-1,1]==4

def test_ac_reference_equivalence(case):
    out,_=topology_to_matpower(case)
    a,b=solve_ac(case),solve_ac(out)
    assert a['residual_pu']<1e-9 and b['residual_pu']<1e-9
    np.testing.assert_array_equal(a['V'],b['V'])
    np.testing.assert_array_equal(a['Sf'],b['Sf'])

def test_no_silent_load_shedding(case):
    out,_=topology_to_matpower(case,{'CB_5_I_B1':False})
    with pytest.raises(ValueError,match='Island without reference'):solve_ac(out)
    assert np.isclose(out['bus'][:,2].sum(),259.)

def test_text_matpower_export(case,tmp_path):
    out,_=topology_to_matpower(case)
    path=tmp_path/'case_full.m';write_matpower_case(out,path,'case_full')
    text=path.read_text()
    assert text.startswith('function mpc = case_full')
    assert all(f'mpc.{key} = [' in text for key in ['bus','gen','branch','gencost'])

def test_optional_pandapower_adapter():
    pp=pytest.importorskip('pandapower',reason='Optional pp integration dependency is not installed in this sandbox')
    import pandapower.networks as pn
    ref=pn.case14()
    net,sec,cb,li,tr=build_nb_ieee14_full(reference_net=ref)
    assert len(net.bus)==65 and len(net.switch)==73
    assert len(net.gen)==len(ref.gen) and len(net.ext_grid)==len(ref.ext_grid)
    for table,ignore in [('line',{'from_bus','to_bus','name'}),('trafo',{'hv_bus','lv_bus','name'}),
                          ('gen',{'bus'}),('load',{'bus'}),('shunt',{'bus'}),('ext_grid',{'bus'})]:
        for col in ref[table].columns:
            if col not in ignore:
                assert net[table][col].equals(ref[table][col]),(table,col)
    pp.runpp(ref,init='flat',calculate_voltage_angles=True)
    pp.runpp(net,init='flat',calculate_voltage_angles=True)
    m=build_full_topology()
    for node,data in m.nodes.items():
        np.testing.assert_allclose(net.res_bus.at[sec[node],'vm_pu'],
            ref.res_bus.iloc[data.planning_bus-1]['vm_pu'],atol=1e-8,rtol=0)
