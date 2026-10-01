"""Offline study fields and v2 outputs of the balanced HIF screen (hypothesis-ranking plan, steps 1 and 2).

``screen_hif`` reports, per compared round, whether each class's best refit
clears the alarm rule (``class_tests``), and on the whole run the accepted
hypotheses, the production ``unexplained`` and ``voltage_meter_channels``
outputs, and ``phasor_suspicion``.  The study fields ride on the full report
only: the provider's compact policy-visible report copies a fixed key set.
"""
from __future__ import annotations

import json

import pytest

from mcp_server.matpower_server import _load_python_case
from psse_env.providers.hif_screen import default_hif_lines, screen_hif
from psse_env.providers.scenario_generator import PHYSICAL_HIF_SAMPLE_PATHS
from tools.lagrangian_port import _copy_result_to_internal

CASE = _load_python_case("case14")
PPC = _copy_result_to_internal(CASE)
LINES = default_hif_lines(CASE)
DECISION_KEYS = ("status", "suspected", "outcome", "branch_row0", "alpha", "shunt_conductance_pu", "meter_set_aside_index0")
VARIANTS = ("production", "first_round_single_cause", "with_one_more_meter")


def _hif_row() -> dict:
    path = PHYSICAL_HIF_SAMPLE_PATHS[0]
    if not path.is_file():
        pytest.skip("tracked 20260923opf HIF corpus is not checked out")
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if (row.get("label") or {}).get("error_type") != "no_error":
            return row
    pytest.skip("no HIF row in the corpus")


def _screen(z, sigma):
    return screen_hif(PPC["baseMVA"], PPC["bus"], PPC["branch"], z, sigma, (), lines=LINES)


def test_an_explained_hif_carries_the_study_fields_and_is_not_unexplained():
    row = _hif_row()
    report = _screen(row["z_obs"], row["sigma_z"])
    assert report["suspected"] is True and report["unexplained"] is False
    assert report["phasor_suspicion"] == {"hif": True, "voltage_meter": False, "unexplained": False}
    assert report["accepted_hypotheses"] == [] and report["outcome"] == "hif"
    final = report["final"]
    assert final["alarm"] is True and final["winner"] == "hif"
    assert set(final["classes"]) == {"meter", "parameter", "topology", "hif"}
    assert final["classes"]["hif"]["explains_alarm"] is True
    assert final["winner_explains_alarm"] is True and final["any_class_explains_alarm"] is True
    assert final["score_margin"] > 0.0
    assert set(report["unexplained_variants"]) == set(VARIANTS)
    assert not any(report["unexplained_variants"].values())
    for name in ("meter", "parameter", "topology", "hif"):
        ranked = report["rounds"][-1]["best"][name]["ranked"]
        assert ranked and ranked[0]["J"] == pytest.approx(min(item["J"] for item in ranked))
        assert [item["J"] for item in ranked] == sorted(item["J"] for item in ranked)


def test_three_gross_meters_leave_the_alarm_unexplained():
    row = _hif_row()
    # Three gross errors on injection channels of a window whose HIF is the
    # only other cause: three rounds set three meters aside (the HIF class is
    # compared only while at most one is set aside) and the HIF remains.
    z = list(row["z_obs"])
    for index, sigmas in ((28, 30.0), (31, -28.0), (33, 26.0)):
        z[index] += sigmas * float(row["sigma_z"][index])
    report = _screen(z, row["sigma_z"])
    assert report["status"] == "valid"
    assert report["final"]["alarm"] is True
    assert report["suspected"] is False
    assert report["unexplained"] is True
    assert report["unexplained_variants"]["production"] is True
    assert [h["class"] for h in report["accepted_hypotheses"]] == ["meter", "meter", "meter"]
    assert report["rounds"][-1]["hif_compared"] is False
    assert set(report["final"]["extras"]) == {"winner_plus_one_meter"}


def test_a_single_gross_meter_is_explained_by_its_removal():
    row = _hif_row()
    z = list(row["z_true"])
    z[52] += 20.0 * float(row["sigma_z"][52])
    report = _screen(z, row["sigma_z"])
    assert report["status"] == "valid"
    assert report["outcome"] == "meter"
    assert report["explained"] is True and report["unexplained"] is False
    assert report["final"]["explained_by_meter_removal"] is True
    assert report["voltage_meter_channels"] == []
    assert report["phasor_suspicion"] == {"hif": False, "voltage_meter": False, "unexplained": False}


def test_a_gross_voltage_meter_names_a_voltage_meter_suspicion():
    row = _hif_row()
    z = list(row["z_true"])
    z[4] += 20.0 * float(row["sigma_z"][4])  # Vm at bus row 4
    report = _screen(z, row["sigma_z"])
    assert report["outcome"] == "meter" and report["explained"] is True
    assert report["voltage_meter_channels"] == [4]
    assert report["phasor_suspicion"] == {"hif": False, "voltage_meter": True, "unexplained": False}


def test_study_fields_do_not_reach_the_compact_policy_report():
    from psse_env.providers.matpower import MatpowerDeploymentProviders

    row = _hif_row()
    report = _screen(row["z_obs"], row["sigma_z"])
    for key in DECISION_KEYS:
        assert key in report
    source = MatpowerDeploymentProviders._hif_screen.__code__.co_consts
    strings = {c for c in source if isinstance(c, str)}
    for key in ("final", "unexplained_variants", "class_tests"):
        assert key not in strings
