import pytest
from IEEE_14_OpenDSS.measurement_convention import measurement_convention_payload, resolve_shunt_convention, validate_shunt_convention


def test_payload_and_resolution():
    declared = {'measurement_convention': measurement_convention_payload('ybus')}
    assert declared['measurement_convention']['consistent_with_operator_wls']
    assert resolve_shunt_convention(None, declared) == 'ybus'
    assert resolve_shunt_convention('legacy_injection', declared) == 'legacy_injection'
    assert resolve_shunt_convention(None, {'measurement_convention':'legacy_injection'}, declared) == 'legacy_injection'
    assert resolve_shunt_convention(None, {}) == 'legacy_injection'
    with pytest.raises(ValueError):
        validate_shunt_convention('unknown')
