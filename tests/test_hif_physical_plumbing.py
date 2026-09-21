import json
from pathlib import Path
import pytest
from psse_env.providers import matpower as providers
from mcp_server import matpower_server as server
from three_phase_nlm.hif_multiscan_estimator import _parse_scans
from three_phase_nlm.hif_parameter_estimator import _resolve_search_configuration, _physical_hif_magnitudes
from trace_protocol import summarize_hif_parameter_estimate_payload, hydrate_tool_arguments


@pytest.mark.parametrize('mode,arguments', [('physical_ohm', {}), ('physical_ohm', {'r_hif_pu_min':5,'r_hif_pu_max':1000}), ('physical_ohm', {'r_hif_ohm_min':100,'r_hif_ohm_max':500}), ('legacy_pu', {})])
@pytest.mark.parametrize('multiscan', [False, True])
def test_provider_forwards_only_explicit_units_and_observation_convention(monkeypatch, mode, arguments, multiscan):
    provider = providers.MatpowerDeploymentProviders(hif_resistance_search=mode)
    calls = []
    fake = lambda **kwargs: calls.append(kwargs) or {'success':False, 'error':'test stop after forwarding'}
    monkeypatch.setattr(providers, '_estimate_hif_location_magnitude_logic', fake)
    monkeypatch.setattr(provider, '_memoized_hif_multiscan', fake)
    declaration = {'measurement_convention':{'shunt_convention':'ybus'}}
    state = {'state_id':'s0','case':'case14','measurements':[1.0]*122,'metadata':{'hif_runtime':declaration,'hif_scan_window':{**declaration,'scans':[{'z_obs':[1.0]*122}]}}}
    method = provider.estimate_hif_multiscan if multiscan else provider.estimate_hif
    method(state, {'arguments':{'candidate_branch_row0':2, **arguments}})
    assert len(calls) == 1
    actual = calls[0]
    assert actual['resistance_search'] == mode
    assert actual['shunt_convention'] == 'ybus'
    for key in ('r_hif_pu_min','r_hif_pu_max','r_hif_ohm_min','r_hif_ohm_max'):
        assert actual[key] == arguments.get(key)


@pytest.mark.parametrize('row,base', [(2,47.61),(10,1.9044),(13,1.9044)])
def test_estimator_search_and_physical_power_contract(row, base):
    kwargs = dict(candidate_branch_row0=row, r_hif_pu_min=None,r_hif_pu_max=None,r_hif_ohm_min=None,r_hif_ohm_max=None,kv_ll=None)
    box = _resolve_search_configuration(**kwargs, resistance_search='physical_ohm')
    assert (box['r_hif_pu_min'],box['r_hif_pu_max']) == pytest.approx((50/base,5000/base))
    legacy = _resolve_search_configuration(**kwargs,resistance_search='legacy_pu')
    assert (legacy['r_hif_pu_min'],legacy['r_hif_pu_max']) == (5,1000)
    value = _physical_hif_magnitudes(r_hif_pu=10,r_hif_model_ohm=.1,fault_v_model_volts=500,box=box)
    assert value['r_hif_ohm'] == pytest.approx(10*base)
    assert value['p_hif_kw'] == 500**2/.1/1000
    assert value['i_hif_amp'] == pytest.approx(500*box['kv_ll']/(10*base))


def test_scan_convention_resolution_and_mixed_window_rejection(tmp_path):
    scan = {'z_obs':[1]*122,'measurement_convention':'ybus'}
    parsed,_ = _parse_scans(scans=[scan],scan_window_path=None)
    assert parsed[0].shunt_convention == 'ybus'
    with pytest.raises(ValueError,match='shunt_convention'):
        _parse_scans(scans=[scan,{'z_obs':[1]*122}],scan_window_path=None)
    path=tmp_path/'window.json';path.write_text(json.dumps({'measurement_convention':'ybus','scans':[{'z_obs':[1]*122}]}))
    assert _parse_scans(scans=None,scan_window_path=path)[0][0].shunt_convention == 'ybus'


def test_summary_and_hydration_preserve_physical_units():
    payload={'success':True,'estimated':{'r_hif_ohm':500,'r_hif_model_ohm':.105,'local_kv_ll':69,'resistance_class':'weak_hif','resistance_basis':'local_line_kv_ll'},'search':{'r_hif_ohm_min':50,'r_hif_ohm_max':5000,'box_source':'default_ohm','shunt_convention':'ybus'}}
    summary=summarize_hif_parameter_estimate_payload(payload)
    assert summary['estimated']['r_hif_model_ohm'] == .105
    assert summary['search'] == payload['search']
    hydrated,_=hydrate_tool_arguments('estimate_hif_location_magnitude_from_path',{},[],{'hif_context':{'measurement_convention':'ybus'}})
    assert hydrated['shunt_convention']=='ybus'
