"""Operator-model rendering of a reported breaker map and the layout projection."""
from copy import deepcopy
from pathlib import Path
import os
import sys
import tempfile

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pypower.api import case14, ppoption, runopf  # noqa: E402

from Transmission.ieee14_full_measurements import (  # noqa: E402
    flipped_case,
    main_section_nodes,
    single_flip_catalogue,
)
from Transmission.ieee14_full_substation import (  # noqa: E402
    add_telemetry_noise,
    operator_model_from_map,
    operator_vector_for_layout,
    operator_vector_from_telemetry,
    solve_node_breaker,
    substation_telemetry,
)
from Transmission.ieee14_full_topology import build_full_topology  # noqa: E402
from mcp_server.matpower_server import _load_python_case, _wls_json  # noqa: E402
from psse_env.providers.matpower import _render_matpower_case  # noqa: E402
from trace_protocol import chi2_threshold  # noqa: E402

OPT = ppoption(VERBOSE=0, OUT_ALL=0)


@pytest.fixture(scope="module")
def model():
    return build_full_topology()


@pytest.fixture(scope="module")
def meters(model):
    _, info, _ = flipped_case(case14(), {}, model=model)
    return main_section_nodes(model, info["node_to_bus"], [])


def _telemetry(model, status_map, seed):
    reference = case14()
    truth_case, info, removed = flipped_case(reference, status_map, model=model)
    assert not removed["dead_buses"]
    dispatch = runopf(deepcopy(truth_case), OPT)
    assert dispatch["success"]
    solution, node_info, node_removed = solve_node_breaker(
        model, reference, status_map, dispatch, info["node_to_bus"]
    )
    assert solution is not None
    telemetry = substation_telemetry(solution, model, reference, node_info, node_removed)
    return add_telemetry_noise(telemetry, np.random.default_rng(seed))


def test_normal_map_renders_the_reference_case_and_projection(model, meters):
    clean = _load_python_case("case14")
    rendered, layout = operator_model_from_map(clean, model, {}, meters)
    assert layout["bus_count"] == 14
    np.testing.assert_allclose(rendered["bus"], clean["bus"])
    np.testing.assert_allclose(rendered["gen"], clean["gen"])
    np.testing.assert_allclose(rendered["branch"], clean["branch"])
    assert layout["sections"]["6"]["meter_node"] == meters[6]
    assert not layout["dropped_sections"]
    telemetry = _telemetry(model, {}, 3)
    np.testing.assert_allclose(
        operator_vector_for_layout(telemetry, layout),
        operator_vector_from_telemetry(telemetry, model, meters),
    )


def test_dangling_terminal_renders_as_the_line_out_of_service(model, meters):
    clean = _load_python_case("case14")
    catalogue = {e["cb_name"]: e for e in single_flip_catalogue(model)}
    error = catalogue["CB_6_L613_B1"]
    rendered, layout = operator_model_from_map(
        clean, model, {"CB_6_L613_B1": error["true_closed"]}, meters
    )
    row0 = int(error["equivalent_branch_row0"])
    assert layout["bus_count"] == 14
    assert layout["dangling_rows"] == [row0]
    assert rendered["branch"][row0][10] == 0.0
    expected = deepcopy(clean)
    expected["branch"][row0][10] = 0.0
    np.testing.assert_allclose(rendered["branch"], expected["branch"])
    np.testing.assert_allclose(rendered["bus"], expected["bus"])


@pytest.mark.parametrize("cb_name", ["CB_6_B1_B2", "CB_4_N2_B2", "CB_9_N1_N2"])
def test_bus_split_renders_one_more_bus_and_verifies_clean(model, meters, cb_name):
    clean = _load_python_case("case14")
    catalogue = {e["cb_name"]: e for e in single_flip_catalogue(model)}
    error = catalogue[cb_name]
    assert error["category"] == "bus_split"
    status_map = {cb_name: error["true_closed"]}
    rendered, layout = operator_model_from_map(clean, model, status_map, meters)
    assert layout["bus_count"] == 15
    split_bus = error["affected_planning_buses"][0]
    # The planning bus keeps its number; the new section is bus 15.
    assert layout["main_section_by_bus"][str(split_bus)] == split_bus
    assert split_bus in layout["sections"]["15"]["planning_buses"]
    assert all(rendered["branch"][k][10] == 1.0 for k in range(20))
    telemetry = _telemetry(model, status_map, 5)
    z = operator_vector_for_layout(telemetry, layout)
    assert len(z) == 3 * 15 + 4 * 20
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "rendered.m")
        Path(path).write_text(_render_matpower_case(rendered, "derived_test"), encoding="utf-8")
        payload = _wls_json(path, z.tolist())
    assert payload["success"]
    dof = len(z) - (2 * 15 - 1)
    assert payload["global_residual_sum"] < chi2_threshold(dof, 0.01)
    # The same telemetry is anomalous on the reported 14-bus model.
    normal_layout = operator_model_from_map(clean, model, {}, meters)[1]
    reported = _wls_json("case14", operator_vector_for_layout(telemetry, normal_layout).tolist())
    assert reported["global_residual_sum"] > 3 * chi2_threshold(122 - 27, 0.01)


def test_merge_renders_one_bus_fewer_and_verifies_clean(model, meters):
    clean = _load_python_case("case14")
    status_map = {"CB_Y1014_14B_10N1": True}
    rendered, layout = operator_model_from_map(clean, model, status_map, meters)
    assert layout["bus_count"] == 13
    merged = layout["main_section_by_bus"]["10"]
    assert layout["main_section_by_bus"]["14"] == merged
    assert layout["sections"][str(merged)]["planning_buses"] == [10, 14]
    # Bus 14's load joins bus 10's row and its two lines now land on that bus.
    assert rendered["bus"][merged - 1][2] == pytest.approx(clean["bus"][9][2] + clean["bus"][13][2])
    for k in range(20):
        assert rendered["branch"][k][10] == 1.0
    telemetry = _telemetry(model, status_map, 8)
    z = operator_vector_for_layout(telemetry, layout)
    assert len(z) == 3 * 13 + 4 * 20
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "rendered.m")
        Path(path).write_text(_render_matpower_case(rendered, "derived_test"), encoding="utf-8")
        payload = _wls_json(path, z.tolist())
    assert payload["success"]
    assert payload["global_residual_sum"] < chi2_threshold(len(z) - (2 * 13 - 1), 0.01)
