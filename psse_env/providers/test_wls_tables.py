"""The bus and branch tables of a WLS solve, and how the model view carries them."""
from __future__ import annotations

import json

import numpy as np
import pytest

from mcp_server.matpower_server import _load_python_case
from psse_env.providers.wls_tables import wls_tables, zero_injection_rows


@pytest.fixture(scope="module")
def case14():
    return _load_python_case("case14")


def _solve(case, **loud):
    """A quiet solve with a few loud channels: ``loud`` maps channel names to (offset, value)."""
    nb, nl = case["bus"].shape[0], case["branch"].shape[0]
    rng = np.random.default_rng(7)
    residual = rng.normal(0.0, 0.5, 3 * nb + 4 * nl)
    lam = rng.normal(0.0, 0.5, 2 * nl)
    starts = {"Vm": 0, "Pinj": nb, "Qinj": 2 * nb, "Pf": 3 * nb, "Qf": 3 * nb + nl, "Pt": 3 * nb + 2 * nl, "Qt": 3 * nb + 3 * nl}
    for channel, (offset, value) in loud.items():
        if channel in ("lr", "lx"):
            lam[2 * offset + (channel == "lx")] = value
        else:
            residual[starts[channel] + offset] = value
    return residual, lam


def test_zero_injection_rule_matches_the_graph_features(case14):
    from research.classifier_triage import features

    assert zero_injection_rows(case14) == sorted(int(i) for i in np.flatnonzero(features.network("case14")["zero_injection"]))
    assert zero_injection_rows(case14) == [6]  # bus 7 of IEEE 14


def test_tables_hold_the_alarm_neighbourhood_in_magnitude_order(case14):
    # An HIF-like pattern on line 9 (bus 4 to bus 9): both P flows positive, the X multiplier large.
    residual, lam = _solve(case14, Pf=(8, 20.0), Pt=(8, 6.5), lx=(8, -12.9), Pinj=(3, 4.3))
    tables = wls_tables(residual, lam, case14)
    branch = tables["branch_table"]
    assert branch[0]["line"] == 9 and (branch[0]["from"], branch[0]["to"]) == (4, 9)
    assert branch[0]["pf"] == 20.0 and branch[0]["pt"] == 6.5 and branch[0]["lx"] == -12.9 and branch[0]["xfmr"] is True
    magnitudes = [max(abs(row[k]) for k in ("pf", "qf", "pt", "qt", "lr", "lx")) for row in branch]
    assert magnitudes == sorted(magnitudes, reverse=True)
    buses = {row["bus"] for row in tables["bus_table"]}
    assert {4, 9} <= buses                      # the ends of the loud line
    assert tables["bus_table"][0]["bus"] == 4 and tables["bus_table"][0]["p"] == 4.3
    lines = {row["line"] for row in branch}
    assert all(row["line"] in lines for row in branch if 4 in (row["from"], row["to"]))  # branches at bus 4 are listed
    assert tables["omitted"] == {"buses": 0, "branches": 0}


def test_zero_injection_bus_shows_no_injection_residual_whatever_the_solve_says(case14):
    residual, lam = _solve(case14, Pinj=(6, 5.0), Vm=(6, 3.0))
    row = next(item for item in wls_tables(residual, lam, case14)["bus_table"] if item["bus"] == 7)
    assert row["p"] is None and row["q"] is None and row["vm"] == 3.0
    # Its loud injection does not even select it: only its voltage did.
    residual, lam = _solve(case14, Pinj=(6, 5.0))
    assert all(item["bus"] != 7 for item in wls_tables(residual, lam, case14)["bus_table"])


def test_caps_keep_the_largest_rows_and_count_the_rest(case14):
    nb, nl = case14["bus"].shape[0], case14["branch"].shape[0]
    residual = np.full(3 * nb + 4 * nl, 3.0)
    tables = wls_tables(residual, np.zeros(2 * nl), case14, max_buses=3, max_branches=4)
    assert len(tables["bus_table"]) == 3 and len(tables["branch_table"]) == 4
    assert tables["omitted"] == {"buses": nb - 3, "branches": nl - 4}
    assert len(json.dumps(tables, separators=(",", ":"))) < 900


def test_quiet_solve_gives_empty_tables(case14):
    nb, nl = case14["bus"].shape[0], case14["branch"].shape[0]
    tables = wls_tables(np.zeros(3 * nb + 4 * nl), np.zeros(2 * nl), case14)
    assert tables == {"bus_table": [], "branch_table": [], "omitted": {"buses": 0, "branches": 0}}
    with pytest.raises(ValueError):
        wls_tables(np.zeros(5), np.zeros(2 * nl), case14)


def test_model_view_keeps_the_tables_in_the_last_output_and_drops_them_from_history(case14):
    from psse_env.dagger.dataset_builder import _compact_last_tool_output, summarize_history

    nb, nl = case14["bus"].shape[0], case14["branch"].shape[0]
    tables = wls_tables(np.full(3 * nb + 4 * nl, 3.0), np.zeros(2 * nl), case14)
    assert len(tables["branch_table"]) == 12 and len(tables["bus_table"]) == 10
    summary = {"success": True, "top_residuals": [], "top_lagrange": [], **tables}
    output = {"execution_status": "success", "tool_metrics": {"wls_summary": summary, "chi_square_alarm": True}}
    compact = _compact_last_tool_output(output)["observable_metrics"]["wls_summary"]
    assert len(compact["branch_table"]) == 12 and "_omitted_items" not in json.dumps(compact)
    assert set(compact["branch_table"][0]) == set(tables["branch_table"][0])   # no field dropped from a row
    history = summarize_history([{"action": {"tool": "run_wls", "arguments": {"state_id": "s0"}}, "tool_output": output}],
                                max_events=8, max_chars=4096)
    event_summary = history[0]["observable_metrics"]["wls_summary"]
    assert "branch_table" not in event_summary and "bus_table" not in event_summary and "omitted" not in event_summary
    assert event_summary["success"] is True
