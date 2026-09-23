from copy import deepcopy

import pytest

from psse_env.dagger.evaluator import privileged_execution_paths
from psse_env.dagger.suite_builder import partition_release_scenario_v1
from three_phase_nlm.measurement_noise import generated_noise_contract


def _scenario():
    sigma = [.001] * 14 + [.01] * 108
    return {
        "scenario_id": "noise-release-envelope-test",
        "scenario_family": "no_error", "case": "case14", "network_case": "case14",
        "measurements": [1.0] * 14 + [0.0] * 108,
        "error_cardinality": 0, "source_tier": "test_fixture",
        "metadata": {
            "sigma_z": sigma,
            "three_phase_sigma": .005,
            "branch_current_sigma_pu": .001,
            "structural_zero_indices": [20],
            "operator_noise": {"sigma_z": sigma, "structural_zero_indices": [20]},
            "noise_contract": generated_noise_contract(
                sigma, noise_scale=1.0, three_phase_sigma=.005, branch_current_sigma_pu=.001),
            "telemetry_control_semantics": {"scada_mean": "balanced_model"},
        },
        "clean_case": "case14", "clean_measurements": [0.0] * 122,
        "true_measurement_errors": [],
    }


def test_observable_noise_metadata_survives_partition_without_truth_leakage():
    source = _scenario()
    before = deepcopy(source)
    envelope = partition_release_scenario_v1(source, split="development")
    assert envelope["execution"]["metadata"] == source["metadata"]
    assert not privileged_execution_paths(envelope["execution"])
    assert "clean_measurements" in envelope["audit"]["truth"]
    assert source == before


def test_only_known_old_noise_contract_description_is_renamed_recursively():
    source = _scenario()
    contract = source["metadata"]["noise_contract"]
    contract["clean_fields_role"] = contract.pop("reference_fields_role")
    source["metadata"]["hif_scan_window"] = {"scans": [{"noise_contract": deepcopy(contract)}]}
    before = deepcopy(source)
    envelope = partition_release_scenario_v1(source)
    metadata = envelope["execution"]["metadata"]
    for actual in (metadata["noise_contract"], metadata["hif_scan_window"]["scans"][0]["noise_contract"]):
        assert "clean_fields_role" not in actual
        assert actual["reference_fields_role"] == "noiseless_reference_for_offline_audit_only"
    assert source == before


@pytest.mark.parametrize("key,value", [
    ("clean_measurements", [0.0] * 122),
    ("clean_fields_role", {"values": [0.0] * 122}),
    ("true_fault", {"branch": 3}),
    ("z_true", [0.0] * 122),
])
def test_noise_metadata_does_not_allow_actual_privileged_fields(key, value):
    source = _scenario()
    source["metadata"]["noise_contract"][key] = value
    with pytest.raises(ValueError, match="audit-only"):
        partition_release_scenario_v1(source)


def test_unknown_metadata_fields_still_fail_closed():
    source = _scenario()
    source["metadata"]["arbitrary_new_field"] = {"value": 1}
    with pytest.raises(ValueError, match="unsupported fields"):
        partition_release_scenario_v1(source)
