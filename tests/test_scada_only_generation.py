"""Scientific evidence-profile boundaries for current research generation/replay."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from psse_env.dagger.suite_builder import partition_release_scenario_v1
from psse_env.providers.scenario_generator import Round0ScenarioGenerator, ScenarioRejected
from scripts.run_dagger_research import (
    build_research_mixture, parser, refresh_d0_training_view,
    resolve_scenario_sources, validate_training_evidence_profile,
)


def _source(generator):
    fixture = Path(__file__).resolve().parents[1] / "psse_env/providers/fixtures/case14_z.json"
    z = json.loads(fixture.read_text(encoding="utf-8"))["z_obs"]
    return {"id": "source_without_auxiliary_sensors", "z_obs": z, "z_true": z.copy(),
        "sigma_z": generator.noise_profile().tolist(), "label": {"source_bus": 7},
        "nlm_diagnostic": {"success": False, "private_marker": "never-visible"}}


def test_default_discovery_cannot_seed_waveform_alarms():
    # The research default moved to wls_gated_diagnostics on 2026-09-23; the
    # scada_only invariants below are unchanged for the explicit profile.
    assert Round0ScenarioGenerator().evidence_profile == "wls_gated_diagnostics"
    generator = Round0ScenarioGenerator(evidence_profile="scada_only")
    assert generator.evidence_profile == "scada_only"
    assert set(generator.waveform_signature_mode.values()) == {"discovered"}
    for family in ("hif", "harmonic", "three_phase_unbalance"):
        for profile in ("scada_only", "wls_gated_diagnostics"):
            with pytest.raises(ValueError, match="auxiliary"):
                Round0ScenarioGenerator(evidence_profile=profile, waveform_signature_mode={family: "flagged"})
        with pytest.raises(ValueError, match="auxiliary"):
            Round0ScenarioGenerator(waveform_signature_mode={family: "flagged"})
    legacy = Round0ScenarioGenerator(evidence_profile="auxiliary_diagnostics", waveform_signature_mode={"hif": "flagged"})
    assert legacy.waveform_signature_mode["hif"] == "flagged"


@pytest.mark.parametrize("family,method", [("hif", "_hif_scenario"), ("harmonic", "_harmonic_scenario"), ("three_phase_unbalance", "_unbalance_scenario")])
def test_strict_waveform_sources_require_wls_alarm_not_auxiliary_diagnosis(family, method):
    generator = Round0ScenarioGenerator(evidence_profile="scada_only")
    source = _source(generator)
    with patch.object(generator, "_aligned_waveform_row", side_effect=lambda row, _: deepcopy(row)), \
         patch.object(generator, "_chi2_statistic", return_value=1000), \
         patch.object(generator, "_require_anomalous") as alarm:
        scenario = getattr(generator, method)(source, 1)
    alarm.assert_called_once()
    assert scenario["measurements"] == source["z_obs"]
    scenario.update(error_cardinality=1, source_tier="test_fixture")
    envelope = partition_release_scenario_v1(scenario)
    encoded = json.dumps(envelope["execution"])
    for forbidden in ("nlm_diagnostic", "harmonic_measurements", "three_phase_voltages", "hif_scan_window", "hif_runtime", "never-visible", "op_point"):
        assert forbidden not in encoded
    assert envelope["execution"]["metadata"]["sigma_z"] == source["sigma_z"]
    assert envelope["audit"]["truth"]
    with patch.object(generator, "_aligned_waveform_row", side_effect=lambda row, _: deepcopy(row)), \
         patch.object(generator, "_chi2_statistic", return_value=1), \
         patch.object(generator, "_require_anomalous", side_effect=ScenarioRejected("quiet_wls", "no alarm")):
        with pytest.raises(ScenarioRejected, match="no alarm"):
            getattr(generator, method)(source, 1)


def test_public_build_sanitizes_runtime_but_keeps_offline_truth_and_scada_history():
    generator = Round0ScenarioGenerator(validate=False, evidence_profile="scada_only")
    source = _source(generator)
    raw = generator._base_scenario("opaque-root", case="case14", measurements=source["z_obs"], family="hif", sigma_z=source["sigma_z"])
    raw["hidden_truth"] = {"true_hif_errors": [{"source_bus": 7}]}
    raw["clean_case"] = "case14"
    raw["clean_measurements"] = source["z_true"]
    raw["metadata"].update({"three_phase_voltages": [1, 2, 3], "op_point": {"load_scale": 1.0},
        "parameter_scans": {"z_scans": [source["z_obs"]], "sigma_z": source["sigma_z"], "scan_indices": [4],
            "initial_states": [{"va": [0.5]}], "op_point": {"true_load": 2}}})
    raw["unresolved_signatures"] = ["hif_suspected_zero_sequence"]
    with patch.object(generator, "_family_source", return_value=([source], lambda *_: deepcopy(raw))):
        built = generator.build({"hif": 1})
    assert len(built) == 1
    assert built[0]["measurements"] == source["z_obs"]
    assert built[0]["hidden_truth"] == raw["hidden_truth"]
    assert "unresolved_signatures" not in built[0]
    assert set(built[0]["metadata"]["parameter_scans"]) == {"z_scans", "sigma_z", "scan_indices"}
    assert "three_phase_voltages" not in built[0]["metadata"]
    assert generator.report()["evidence_profile"] == "scada_only"


@pytest.mark.parametrize("row", [{"example_id": "old"}, {"evidence_profile": "auxiliary_diagnostics"},
    {"evidence_profile": "scada_only", "policy_observation": {"evidence_profile": "auxiliary_diagnostics"}}])
def test_strict_replay_rejects_undeclared_auxiliary_and_conflicting_profiles(row):
    with pytest.raises(ValueError, match="do not relabel"):
        validate_training_evidence_profile([row])


def test_legacy_reproduction_is_explicit_and_never_relabels_source_rows():
    row = {"example_id": "old"}
    validate_training_evidence_profile([row], evidence_profile="auxiliary_diagnostics")
    assert row == {"example_id": "old"}
    with patch("scripts.run_dagger_research.examples_to_chat_sft") as export:
        with pytest.raises(ValueError, match="D0 raw"):
            refresh_d0_training_view([row], [row])
        export.assert_not_called()


def test_mixture_checks_both_pools_before_sampling_and_preserves_observations():
    d0 = {"example_id": "d0", "physical_root_fingerprint": "r0", "metadata": {"evidence_profile": "scada_only"}, "messages": [{"role": "user", "content": "observed SCADA"}]}
    d1 = {"example_id": "d1", "physical_root_fingerprint": "r1", "evidence_profile": "scada_only", "messages": [{"role": "user", "content": "another observation"}]}
    mixed, report = build_research_mixture([d0], [d1], d1_share=.5, d1_cap=None, seed=3, evidence_profile="scada_only")
    assert report["evidence_profile"] == "scada_only"
    assert {r["example_id"]: r["messages"] for r in mixed} == {"d0": d0["messages"], "d1": d1["messages"]}
    for first, second in (([{}], [d1]), ([d0], [{}])):
        with pytest.raises(ValueError, match="evidence profile"):
            build_research_mixture(first, second, d1_share=1, d1_cap=None, seed=3, evidence_profile="scada_only")
    # scada_only rows are not wls_gated rows: the default profile refuses them.
    with pytest.raises(ValueError, match="evidence profile"):
        build_research_mixture([d0], [d1], d1_share=.5, d1_cap=None, seed=3)


def test_cli_and_source_descriptor_declare_strict_default():
    args = parser().parse_args(["--d0-raw", "raw", "--d0-train", "train", "--adapter-path", "adapter", "--output-dir", "out"])
    assert args.evidence_profile == "wls_gated_diagnostics"
    assert args.hif_signature_mode == "discovered"
    assert resolve_scenario_sources(plan_families={"measurement"})["evidence_profile"] == "wls_gated_diagnostics"
    assert resolve_scenario_sources(plan_families={"measurement"}, evidence_profile="scada_only")["evidence_profile"] == "scada_only"
    for profile in ("scada_only", "wls_gated_diagnostics"):
        with pytest.raises(ValueError, match="auxiliary"):
            resolve_scenario_sources(plan_families={"hif"}, signature_modes={"hif": "flagged"}, evidence_profile=profile)
    with pytest.raises(ValueError, match="auxiliary"):
        resolve_scenario_sources(plan_families={"hif"}, signature_modes={"hif": "flagged"})


def test_hpc_templates_guard_reuse_and_record_selected_profile():
    root = Path(__file__).resolve().parents[1] / "research/hpc/full_pipeline_20260907"
    env = (root / "pipeline.env").read_text(encoding="utf-8")
    assert 'EVIDENCE_PROFILE=${EVIDENCE_PROFILE:-wls_gated_diagnostics}' in env
    assert 'case "$EVIDENCE_PROFILE" in scada_only|wls_gated_diagnostics|auxiliary_diagnostics)' in env
    assert 'declared = declared or "auxiliary_diagnostics"' in env
    assert 'assert_stage_evidence_profile "$PREVIOUS_PIPE/out/$receipt"' in env
    assert env.index('assert_stage_evidence_profile "$PREVIOUS_PIPE/out/$receipt"') < env.index('ln -s "$PREVIOUS_PIPE/out/$subdir"')
    for name in ("stage_d0.sbatch", "stage_bc0.sbatch"):
        assert '"evidence_profile": sys.argv[' in (root / name).read_text(encoding="utf-8")
    assert "validate_training_evidence_profile" in (root / "stage_bc0.sbatch").read_text(encoding="utf-8")
