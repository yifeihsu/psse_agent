"""Regression tests for the fixed-identity SCADA mapping of the full IEEE-14 model."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pypower.api import case14, makeYbus, ppoption, runpf  # noqa: E402
from pypower.idx_bus import PD, QD  # noqa: E402

from Transmission.ieee14_full_measurements import (  # noqa: E402
    NZ,
    dangling_terminal_errors,
    flipped_case,
    main_section_nodes,
    operator_measurements,
    single_flip_catalogue,
)
from Transmission.ieee14_full_topology import build_full_topology  # noqa: E402

OPT = ppoption(VERBOSE=0, OUT_ALL=0)


def h_of_x(ppc):
    """Pipeline measurement function at the stored state (shunts inside Ybus)."""
    bus = np.asarray(ppc["bus"], float).copy()
    branch = np.asarray(ppc["branch"], float).copy()
    # makeYbus wants internal (0-based, consecutive) bus numbering.
    lookup = {int(r[0]): i for i, r in enumerate(bus)}
    bus[:, 0] = np.arange(len(bus))
    f = np.array([lookup[int(r[0])] for r in branch])
    t = np.array([lookup[int(r[1])] for r in branch])
    branch[:, 0] = f
    branch[:, 1] = t
    ybus, yf, yt = makeYbus(float(ppc["baseMVA"]), bus, branch)
    v = bus[:, 7] * np.exp(1j * np.deg2rad(bus[:, 8]))
    inj = v * np.conj(ybus @ v)
    sf = v[f] * np.conj(yf @ v)
    st = v[t] * np.conj(yt @ v)
    return np.r_[np.abs(v), inj.real, inj.imag, sf.real, sf.imag, st.real, st.imag]


@pytest.fixture(scope="module")
def model():
    return build_full_topology()


@pytest.fixture(scope="module")
def catalogue(model):
    return single_flip_catalogue(model)


def solve_flip(model, status_map, load_scale=1.0):
    ref = case14()
    ref["bus"][:, PD] *= load_scale
    ref["bus"][:, QD] *= load_scale
    case, info, removed = flipped_case(ref, status_map, model=model)
    sol, ok = runpf(deepcopy(case), OPT)
    assert ok
    return ref, sol, info, removed


def test_catalogue_counts(catalogue):
    counts = Counter(entry["category"] for entry in catalogue)
    assert counts == {
        "dangling_line_terminal": 26,
        "equivalent": 25,
        "bus_split": 10,
        "unsupplied_island": 9,
        "merge_10_14": 3,
    }
    assert sum(entry["partition_changed"] for entry in catalogue) == 48


def test_dangling_errors_are_reported_closed_truly_open(model):
    errors = dangling_terminal_errors(model)
    assert len(errors) == 26
    for entry in errors:
        assert entry["reported_closed"] is True and entry["true_closed"] is False
        assert 0 <= entry["equivalent_branch_row0"] < 20
        assert entry["minor_section"]["equipment"] == []
        assert len(entry["minor_section"]["terminal_rows"]) == 1


@pytest.mark.parametrize("load_scale", [0.8, 1.0, 1.25])
def test_normal_state_matches_h_x(model, load_scale):
    ref, sol, info, removed = solve_flip(model, {}, load_scale)
    assert removed["dead_buses"] == []
    nodes = main_section_nodes(model, info["node_to_bus"], [])
    z = operator_measurements(sol, ref, model, info["node_to_bus"], [], nodes)
    assert z.shape == (NZ,)
    assert np.max(np.abs(z - h_of_x(sol))) < 1e-9


def test_equivalent_flips_reproduce_normal_measurements(model, catalogue):
    ref, sol, info, _ = solve_flip(model, {})
    z_normal = operator_measurements(
        sol, ref, model, info["node_to_bus"], [], main_section_nodes(model, info["node_to_bus"], [])
    )
    for entry in catalogue:
        if entry["category"] != "equivalent":
            continue
        ref_f, sol_f, info_f, removed = solve_flip(model, {entry["cb_name"]: entry["true_closed"]})
        assert removed["dead_buses"] == []
        z = operator_measurements(
            sol_f, ref_f, model, info_f["node_to_bus"], [],
            main_section_nodes(model, info_f["node_to_bus"], []),
        )
        assert np.allclose(z, z_normal, atol=1e-9), entry["cb_name"]


def test_dangling_flips_keep_every_section_energized(model):
    for entry in dangling_terminal_errors(model):
        _, _, _, removed = solve_flip(model, {entry["cb_name"]: entry["true_closed"]})
        assert removed["dead_buses"] == [], entry["cb_name"]
        assert removed["shed_p_mw"] == 0.0


def test_main_section_meter_avoids_dangling_side(model):
    # Opening CB_1_B1_N1 leaves {1B1, 1N3} hanging on line 1-5; the meter must
    # stay with the slack section even though the busbar node is on the other side.
    _, _, info, _ = solve_flip(model, {"CB_1_B1_N1": False})
    nodes = main_section_nodes(model, info["node_to_bus"], [])
    assert info["node_to_bus"][nodes[1]] == info["node_to_bus"]["1N1"]
    assert info["node_to_bus"][nodes[1]] != info["node_to_bus"]["1N3"]


def test_islanded_load_bay_reads_zero(model):
    # CB_5_I_B1 cuts the bus-5 load bay off: load shed, dead bay reads 0 pu on the
    # anchor placement, while the main-section meter still reads a live voltage.
    ref, sol, info, removed = solve_flip(model, {"CB_5_I_B1": False})
    assert removed["shed_p_mw"] == pytest.approx(7.6)
    dead = removed["dead_buses"]
    assert dead
    z_anchor = operator_measurements(sol, ref, model, info["node_to_bus"], dead, dict(model.anchors))
    z_main = operator_measurements(
        sol, ref, model, info["node_to_bus"], dead, main_section_nodes(model, info["node_to_bus"], dead)
    )
    assert z_anchor[4] == 0.0
    assert z_main[4] > 0.9
    assert z_main[14 + 4] == 0.0  # Pinj[5]: no generator, load shed
