"""Node/breaker generalized state estimation with normalized Lagrange multipliers."""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pypower.api import case14, ppoption, runopf  # noqa: E402

from Transmission.ieee14_full_gse import gse_topology_nlm, rank_breakers, screen_breaker_flips  # noqa: E402
from Transmission.ieee14_full_measurements import (  # noqa: E402
    flipped_case,
    main_section_nodes,
    operator_measurements,
    single_flip_catalogue,
)
from Transmission.ieee14_full_substation import (  # noqa: E402
    add_telemetry_noise,
    operator_vector_from_telemetry,
    solve_node_breaker,
    substation_telemetry,
)
from Transmission.ieee14_full_topology import build_full_topology  # noqa: E402
from trace_protocol import chi2_threshold  # noqa: E402

OPT = ppoption(VERBOSE=0, OUT_ALL=0)


@pytest.fixture(scope="module")
def model():
    return build_full_topology()


@pytest.fixture(scope="module")
def reference():
    return case14()


def _physical_truth(model, reference, status_map):
    truth_case, info, removed = flipped_case(reference, status_map, model=model)
    assert not removed["dead_buses"]
    dispatch = runopf(deepcopy(truth_case), OPT)
    assert dispatch["success"]
    solution, node_info, node_removed = solve_node_breaker(
        model, reference, status_map, dispatch, info["node_to_bus"]
    )
    assert solution is not None
    telemetry = substation_telemetry(solution, model, reference, node_info, node_removed)
    return dispatch, info, telemetry


def test_operator_vector_reads_the_same_meters_as_the_telemetry(model, reference):
    dispatch, info, telemetry = _physical_truth(model, reference, {})
    meters = main_section_nodes(model, info["node_to_bus"], [])
    from_telemetry = operator_vector_from_telemetry(telemetry, model, meters)
    from_ideal_opf = operator_measurements(
        dispatch, reference, model, info["node_to_bus"], [], meters
    )
    # The breaker impedances are tiny, so the physical solution stays within a
    # fraction of one sigma of the ideal contracted dispatch on every channel.
    assert np.max(np.abs(from_telemetry - from_ideal_opf)) < 2e-3
    assert telemetry["node_vm"]["6|I"] > 0
    assert set(telemetry["node_pinj"]) == set(telemetry["injection_metered_nodes"])
    assert len(telemetry["cb_p"]) == 73


def test_normal_state_is_clean_and_loop_constraints_are_unidentifiable(model, reference):
    _, _, telemetry = _physical_truth(model, reference, {})
    noisy = add_telemetry_noise(telemetry, np.random.default_rng(5))
    estimate = gse_topology_nlm(model, reference, {}, noisy)
    assert estimate["success"]
    assert estimate["chi_square"] < chi2_threshold(estimate["dof"], 0.01)
    # Two double-connected bays (yards 5 and 11) close loops in the
    # closed-breaker graph; the loop-closing breakers' two constraints each are
    # implied by the others and carry no multiplier.
    assert set(estimate["loop_dependent_breakers"]) == {"CB_5_T56_B2", "CB_11_L1110_B2"}
    assert estimate["n_dependent_constraints"] == 4
    for name in estimate["loop_dependent_breakers"]:
        assert all(value == 0.0 for value in estimate["breaker_multipliers"][name].values())
    assert estimate["ranking"][0]["score"] < 5.0
    assert len(estimate["ranking"]) == 73


@pytest.mark.parametrize("cb_name", ["CB_6_L613_B1", "CB_1_B1_N1", "CB_9_N4_B2"])
def test_wrong_reported_status_is_ranked_first_and_confirmed_by_flip(model, reference, cb_name):
    catalogue = {e["cb_name"]: e for e in single_flip_catalogue(model)}
    error = catalogue[cb_name]
    assert error["category"] == "dangling_line_terminal"
    status_map = {cb_name: error["true_closed"]}
    _, _, telemetry = _physical_truth(model, reference, status_map)
    noisy = add_telemetry_noise(telemetry, np.random.default_rng(11))
    reported = gse_topology_nlm(model, reference, {}, noisy)
    assert reported["success"]
    assert reported["chi_square"] > 10 * chi2_threshold(reported["dof"], 0.01)
    assert reported["ranking"][0]["cb_name"] == cb_name
    flips = screen_breaker_flips(
        model, reference, {}, noisy, [item["cb_name"] for item in reported["ranking"][:4]]
    )
    by_name = {item["cb_name"]: item for item in flips}
    assert by_name[cb_name]["chi_square"] < chi2_threshold(by_name[cb_name]["dof"], 0.01)
    others = [item for item in flips if item["cb_name"] != cb_name]
    assert all(item["chi_square"] > chi2_threshold(item["dof"], 0.01) for item in others)
    truth = gse_topology_nlm(model, reference, status_map, noisy)
    assert truth["success"] and truth["chi_square"] < chi2_threshold(truth["dof"], 0.01)


def test_rank_breakers_breaks_series_ties_with_the_second_constraint():
    multipliers = {
        "chain_a": {"theta": 189.4, "vm": 50.0},
        "chain_b": {"theta": 189.41, "vm": 104.6},
        "chain_c": {"theta": -189.4, "vm": 88.7},
        "quiet": {"p": 1.2, "q": 0.3},
    }
    assert rank_breakers(multipliers) == ["chain_b", "chain_c", "chain_a", "quiet"]
