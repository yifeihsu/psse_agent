"""IEEE 118 waveform support (2026-10-11): the system's line table, the registry's families and the runtime thresholds."""
from __future__ import annotations

import json

import numpy as np
import pytest

from mcp_server.matpower_server import _load_python_case
from psse_env.systems import resolve_system
from three_phase_nlm.branch_current_analysis import case14_line_parameters, line_parameters_from_case


def test_line_table_of_case118_names_every_line_with_its_voltage_base():
    table = line_parameters_from_case(_load_python_case("case118"))
    assert len(table) == 186
    lines = [row0 for row0, item in table.items() if item["is_line"]]
    assert len(lines) == 177  # the nine tapped transformers are not lines
    assert table[0]["dss_element"] == "Line.1-2" and table[0]["kv_ll"] == 138.0
    assert {item["kv_ll"] for item in table.values()} == {138.0, 345.0}  # the 161 kV bus 87 is never a from-bus
    assert all(set(item) >= {"r", "x", "b", "from_bus", "to_bus", "is_line", "dss_element", "kv_ll"} for item in table.values())


def test_line_table_of_case14_matches_the_legacy_table_up_to_names_and_bases():
    legacy, generic = case14_line_parameters(), line_parameters_from_case(_load_python_case("case14"))
    for row0, item in legacy.items():
        for key in ("r", "x", "b", "from_bus", "to_bus", "is_line"):
            assert generic[row0][key] == pytest.approx(item[key]) if key in ("r", "x", "b") else generic[row0][key] == item[key]
    assert generic[0]["kv_ll"] is None  # the pypower case carries no base kV; the legacy resolver keeps the IEEE 14 bases


def test_registry_lists_the_waveform_families_for_case118_only():
    assert {"hif", "measurement+hif", "three_phase_unbalance", "harmonic"} <= set(resolve_system("case118").supported_families)
    assert "topology" not in resolve_system("case118").supported_families
    assert "hif" not in resolve_system("case57").supported_families


def test_runtime_picks_the_threshold_of_the_scored_network(tmp_path):
    torch = pytest.importorskip("torch")
    from research.classifier_triage import features, runtime
    from research.classifier_triage.model import TriageGNN

    node_dim, edge_dim, global_dim = features.dims("residual_v2")
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    torch.manual_seed(0)
    model = TriageGNN(node_dim, edge_dim, global_dim, hidden_dim=16, layers=1, dropout=0.0)
    torch.save({"state_dict": model.state_dict(), "config": model.config, "view": "residual_v2", "seed": 0, "settings": {}},
               checkpoints / "seed_0.pt")
    manifest = runtime.export_checkpoints(checkpoints, tmp_path / "runtime", threshold=0.2, threshold_rule="test",
                                          source={}, thresholds={"case14": 0.2, "case118": 0.6})
    assert manifest["request_thresholds"] == {"case14": 0.2, "case118": 0.6}
    classifier = runtime.load_triage_classifier(tmp_path / "runtime")
    assert classifier.threshold_for("case118") == 0.6 and classifier.threshold_for("case14") == 0.2
    assert classifier.threshold_for(_load_python_case("case118")) == 0.6
    assert classifier.threshold_for(_load_python_case("case14")) == 0.2
    assert classifier.threshold_for("case57") == 0.2  # the fallback
    net = features.network("case118")
    nb, nl = int(net["nb"]), int(net["nl"])
    rng = np.random.default_rng(1)
    report = classifier.report(_load_python_case("case118"), {
        "success": True, "r": np.abs(rng.normal(size=3 * nb + 4 * nl)).tolist(), "raw_residual": rng.normal(size=3 * nb + 4 * nl).tolist(),
        "lambdaN": rng.normal(size=2 * nl).tolist(), "global_residual_sum": 900.0, "dof": 3 * nb + 4 * nl - (2 * nb - 1)})
    assert report["request_threshold"] == 0.6
    (tmp_path / "runtime" / "runtime.json").write_text(json.dumps({**manifest, "request_thresholds": {"case14": 1.5}}), encoding="utf-8")
    with pytest.raises(ValueError, match="request_thresholds"):
        runtime.load_triage_classifier(tmp_path / "runtime")


def test_research_runner_accepts_ieee118_waveform_corpora_and_refuses_them_elsewhere(tmp_path):
    import scripts.run_dagger_research as research

    corpus = tmp_path / "samples.jsonl"
    corpus.write_text("{}\n", encoding="utf-8")
    hif = tmp_path / "hif_samples.jsonl"
    hif.write_text("{}\n", encoding="utf-8")
    sources = research.resolve_scenario_sources(
        plan_families={"hif", "measurement"}, system="case118", measurement_corpus=corpus, balanced_artifact_dir=tmp_path,
        hif_sample_paths=[hif], imbalance_sample_path=None, admission_mode="recoverable",
        evidence_profile="classifier_gated_diagnostics", signature_modes={"hif": "discovered"},
    )
    assert sources["system"] == "case118" and sources["hif_sample_paths"] == [str(hif.resolve())]
    with pytest.raises(ValueError, match="three-phase route"):
        research.resolve_scenario_sources(
            plan_families={"measurement"}, system="case57", measurement_corpus=corpus, balanced_artifact_dir=tmp_path,
            hif_sample_paths=[hif], imbalance_sample_path=None, admission_mode="recoverable",
            evidence_profile="classifier_gated_diagnostics",
        )
    with pytest.raises(ValueError, match="does not support families"):
        research.resolve_scenario_sources(
            plan_families={"topology"}, system="case118", measurement_corpus=corpus, balanced_artifact_dir=tmp_path,
            admission_mode="recoverable", evidence_profile="classifier_gated_diagnostics",
        )
