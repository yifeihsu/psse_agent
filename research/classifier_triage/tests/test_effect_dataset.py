"""Effect-targeted dataset rows (research.classifier_triage.ieee118_dataset under a band, 2026-10-12)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from research.classifier_triage import ieee118_dataset as ds
from research.classifier_triage.effect_sampler import effect_ratio

REPO = Path(__file__).resolve().parents[3]
HIF_CORPUS = REPO / "artifacts/measurements/hif_physical69_main_valid_detectable_8x10_20260923opf/samples.jsonl"
BAND = (1.5, 20.0)


def _hif_row() -> dict:
    with HIF_CORPUS.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("scenario") == "high_impedance_fault":
                return row
    raise AssertionError("no HIF row")


@pytest.fixture(autouse=True)
def _reset_module_state():
    yield
    ds.set_system("case118")
    ds.set_effect_band(None)


def test_a_corpus_hif_row_becomes_an_in_band_root_with_its_meter_overlay_and_corpus_row():
    if not HIF_CORPUS.is_file():
        pytest.skip("IEEE 14 HIF corpus not present")
    task = {"system": "case14", "effect_band": list(BAND), "kind": "hif", "index": 3, "seed": 7, "row": _hif_row(), "healthy_replicas": 1}
    out = ds.corpus_waveform_job(task)
    assert out["attempt"]["outcome"] == "admitted", out["attempt"]
    families = [root["family"] for root, _ in out["roots"]]
    assert families[0] == "hif"
    root, payload = out["roots"][0]
    ratio = effect_ratio(payload["objective"], payload["dof"])
    assert BAND[0] * 0.85 <= ratio <= BAND[1] * 1.15
    assert root["case"] == "case14" and root["effect"]["sampler"] == "effect_targeted_v1"
    assert root["truth"]["hif"][0]["drawn"]["r_hif_ohm"] > 0 and "r_hif_ohm" in root["truth"]["hif"][0]
    if "measurement+hif" in families:
        mixed, mixed_payload = out["roots"][families.index("measurement+hif")]
        assert mixed["overlay_effect"]["sampler"] == "effect_targeted_v1"
        assert effect_ratio(mixed_payload["objective"], mixed_payload["dof"]) > ratio
    corpus_row = out["corpus"][0]
    assert corpus_row["effect"]["target"] == pytest.approx(root["effect"]["target"])
    assert len(corpus_row["three_phase_voltages"]) == 14 and corpus_row["scan_count"] == 1
    assert root["root_id"] in out["phasors"]


def test_a_harmonic_root_is_scaled_into_the_band_and_keeps_the_drawn_thd():
    pytest.importorskip("pypower")
    out = ds.harmonic_job({"system": "case14", "effect_band": list(BAND), "kind": "harmonic", "index": 2, "seed": 7})
    assert out["attempt"]["outcome"] == "admitted", out["attempt"]
    root, payload = out["roots"][0]
    ratio = effect_ratio(payload["objective"], payload["dof"])
    assert BAND[0] * 0.85 <= ratio <= BAND[1] * 1.15
    harmonic = root["truth"]["harmonic"][0]
    assert harmonic["thd_target"] == pytest.approx(harmonic["drawn_thd_target"] * np.sqrt(root["effect"]["scale"]))


def test_the_generator_scales_its_harmonic_roots_under_a_band():
    pytest.importorskip("pypower")
    from psse_env.providers.scenario_generator import Round0ScenarioGenerator
    from research.classifier_triage.effect_sampler import wls_solver

    generator = Round0ScenarioGenerator(system="case14", seed=3, effect_band=BAND)
    row = generator._synthesized_harmonic_row(0)
    objective, dof = wls_solver("case14", row["sigma_z"])(np.asarray(row["z_obs"], dtype=float))
    ratio = effect_ratio(objective, dof)
    assert BAND[0] * 0.85 <= ratio <= BAND[1] * 1.15
    assert row["label"]["effect"]["sampler"] == "effect_targeted_v1"
    assert row["label"]["thd_target"] == pytest.approx(row["label"]["drawn_thd_target"] * np.sqrt(row["label"]["effect"]["scale"]))
    plain = Round0ScenarioGenerator(system="case14", seed=3)
    assert "effect" not in plain._synthesized_harmonic_row(0)["label"]
