"""suspicion_gated_diagnostics: phasors follow a balanced HIF suspicion, and every root has them.

The contract (2026-09-27): the agent sees balanced SCADA and its WLS; the
balanced HIF screen runs on each WLS alarm, and only its suspicion admits the
phase-resolved PMU phasors and the HIF diagnostics on them.  Because the screen
can fire on a root of any family, every root carries phasors from its true
state at one PMU sigma, so a request never answers by availability.
"""
from __future__ import annotations

import copy
import json

import pytest

import scripts.run_dagger_research as research
from mcp_server.matpower_server import _load_python_case
from psse_env.actions import (
    ESTIMATE_HIF_FROM_PATH, ESTIMATE_HIF_MULTISCAN_FROM_PATH, GET_HARMONIC_CONTEXT, GET_THREE_PHASE_CONTEXT,
    RUN_HSE_FROM_PATH, RUN_THREE_PHASE_NLM_FROM_PATH, RUN_WLS,
)
from psse_env.dagger.evaluator import evaluate_rollout_suites
from psse_env.dagger.release_factories import deterministic_case_loader
from psse_env.dagger.suite_builder import partition_release_scenario_v1
from psse_env.evidence_profile import (
    STRICT_BOUNDARY_PROFILES, SUSPICION_GATED_PROFILE, WLS_GATED_PROFILE, disabled_tools, is_wls_gated,
    required_suspicion, requires_wls_alarm_for_diagnostics, sanitize_gated_metadata,
)
from psse_env.oracle.process_validity import ProcessValidityOracle
from psse_env.providers.balanced_phasors import balanced_phasor_rows, true_balanced_state
from psse_env.providers.scenario_generator import (
    PHYSICAL_HIF_SAMPLE_PATHS, PHYSICAL_IMBALANCE_SAMPLE_PATH, PMU_PHASOR_SIGMA_PU, Round0ScenarioGenerator,
    build_measurement_vector,
)
from psse_env.providers.suspicion_gated import zero_sequence_line_screen
from three_phase_nlm.branch_current_analysis import line_differential_null_test

PROFILE = SUSPICION_GATED_PROFILE
SEED = 20260927


def _generator(profile: str = PROFILE, seed: int = SEED) -> Round0ScenarioGenerator:
    if not PHYSICAL_HIF_SAMPLE_PATHS[0].is_file():
        pytest.skip("tracked 20260923opf HIF corpora are not checked out")
    return Round0ScenarioGenerator(
        seed=seed, hif_sample_paths=list(PHYSICAL_HIF_SAMPLE_PATHS),
        imbalance_sample_path=PHYSICAL_IMBALANCE_SAMPLE_PATH, normalized_residual_threshold=4.0,
        evidence_profile=profile, hif_max_scans=3,
    )


@pytest.fixture(scope="module")
def roots():
    built = _generator().build({"hif": 2, "measurement+hif": 1, "measurement": 2, "parameter": 1, "topology": 1})
    return [partition_release_scenario_v1(row, split="dagger_train") for row in built]


def _environment():
    research.RESEARCH_ENVIRONMENT_OPTIONS["evidence_profile"] = PROFILE
    research.RESEARCH_ENVIRONMENT_OPTIONS["normalized_residual_threshold"] = 4.0
    return research.resolve_environment_factory("research", PROFILE)


# --------------------------------------------------------------------- profile


def test_the_profile_is_strict_alarm_gated_and_adds_suspicions():
    assert PROFILE in STRICT_BOUNDARY_PROFILES
    assert is_wls_gated(PROFILE) and requires_wls_alarm_for_diagnostics(PROFILE)
    for tool in (GET_THREE_PHASE_CONTEXT, RUN_THREE_PHASE_NLM_FROM_PATH, ESTIMATE_HIF_FROM_PATH):
        assert required_suspicion(PROFILE, tool) == "hif"
        assert required_suspicion(WLS_GATED_PROFILE, tool) is None
    for tool in (GET_HARMONIC_CONTEXT, RUN_HSE_FROM_PATH):
        assert required_suspicion(PROFILE, tool) == "harmonic"
    # The multi-scan estimator needs a window only HIF roots carry.
    assert ESTIMATE_HIF_MULTISCAN_FROM_PATH in disabled_tools(PROFILE)
    assert ESTIMATE_HIF_MULTISCAN_FROM_PATH not in disabled_tools(WLS_GATED_PROFILE)


def test_the_sanitizer_drops_the_opendss_acquisition_blocks():
    metadata = {"sigma_z": [0.01], "three_phase_voltages": [{"bus": "b1"}], "three_phase_sigma": 1e-4,
                "hif_runtime": {"op_point": {"load_scale": 1.1}}, "hif_scan_window": {"scans": []}}
    gated = sanitize_gated_metadata(metadata, WLS_GATED_PROFILE)
    screened = sanitize_gated_metadata(metadata, PROFILE)
    assert "hif_runtime" in gated and "hif_scan_window" in gated
    assert "hif_runtime" not in screened and "hif_scan_window" not in screened
    assert screened["three_phase_voltages"] == metadata["three_phase_voltages"]
    assert screened["evidence_profile"] == PROFILE


def test_the_process_oracle_needs_the_suspicion_on_top_of_the_alarm():
    oracle = ProcessValidityOracle()

    def state(screen):
        ledger = {"state_id": "s0", "state_hash": "hash0", "successful": True, "chi_square_alarm": True,
                  "normalized_residual_alarm": False}
        if screen is not None:
            ledger["hif_screen"] = screen
        return {"evidence_profile": PROFILE, "active_state_id": "s0", "candidate_state_id": None,
                "has_open_candidate": False, "unresolved_signatures": [], "available_evidence": [],
                "tried_action_signatures": [], "fresh_context_evidence": {"wls": ledger}}

    action = {"tool": GET_THREE_PHASE_CONTEXT, "arguments": {"state_id": "s0"}}
    for screen in (None, {"status": "valid", "suspected": False}, {"status": "screen_error", "suspected": True}):
        check = oracle.check(state(screen), action)
        assert check["error_code"] == "diagnostics_require_hif_suspicion", screen
        assert check["valid_next_actions"] == [{"tool": RUN_WLS, "arguments": {"state_id": "s0"}}]
    assert oracle.check(state({"status": "valid", "suspected": True}), action)["process_valid"]
    spectra = oracle.check(state({"status": "valid", "suspected": True}),
                           {"tool": GET_HARMONIC_CONTEXT, "arguments": {"state_id": "s0"}})
    assert spectra["error_code"] == "diagnostics_require_harmonic_suspicion"


# --------------------------------------------------------------------- phasors


def test_true_state_rows_follow_the_exporter_format_and_are_balanced():
    case = _load_python_case("case14")
    z = build_measurement_vector(case).tolist()
    vm, va = true_balanced_state(case, z)
    voltages, currents = balanced_phasor_rows(case, vm, va)
    assert len(voltages) == 14 and len(currents) == 20
    assert set(voltages[0]) == {"bus", "kvbase_ln", "vln_pu", "ang_deg"}
    assert set(currents[0]) == {"branch", "branch_row0", "from_bus", "to_bus", "ibase_from_a", "ibase_to_a",
                                "i_from_pu", "ang_from_deg", "i_to_pu", "ang_to_deg"}
    assert voltages[0]["bus"] == "b1" and abs(voltages[0]["kvbase_ln"] - 69.0 / 3 ** 0.5) < 1e-9
    assert currents[7]["branch"] == "Transformer.4-7"
    assert abs(currents[7]["ibase_to_a"] / currents[7]["ibase_from_a"] - 69.0 / 13.8) < 1e-9
    for row in voltages:
        assert max(row["vln_pu"]) - min(row["vln_pu"]) < 1e-12
    null = line_differential_null_test(voltages, currents, sigma_pu=PMU_PHASOR_SIGMA_PU)
    assert null["max_line_differential_pu"] < 1e-9


def test_the_zero_sequence_screen_ignores_a_balanced_model_error():
    case = _load_python_case("case14")
    vm, va = true_balanced_state(case, build_measurement_vector(case).tolist())
    voltages, currents = balanced_phasor_rows(case, vm, va)
    # A line the operator models in service that is out in the field: its
    # terminal currents are zero, so the modeled charging current remains as
    # a balanced differential on all three phases.
    outage = copy.deepcopy(currents)
    for key in ("i_from_pu", "i_to_pu"):
        outage[0][key] = [0.0, 0.0, 0.0]
    per_phase = line_differential_null_test(voltages, outage, sigma_pu=PMU_PHASOR_SIGMA_PU)
    assert per_phase["hif_like_differential_present"] is True
    screen = zero_sequence_line_screen(voltages, outage, sigma_pu=PMU_PHASOR_SIGMA_PU)
    assert screen["hif_like"] is False
    assert screen["max_zero_sequence_pu"] < 1e-9


def test_every_family_carries_phasors_at_one_sigma_and_scada_is_unchanged():
    plan = {family: 1 for family in (
        "no_error", "measurement", "multi_measurement", "parameter", "topology", "harmonic", "hif",
        "measurement+hif", "three_phase_unbalance", "telemetry_no_disturbance",
    )}
    screened = _generator(PROFILE, 7).build(plan)
    gated = {row["scenario_id"]: row for row in _generator(WLS_GATED_PROFILE, 7).build(plan)}
    assert {row["scenario_family"] for row in screened} == set(plan)
    for row in screened:
        metadata = partition_release_scenario_v1(row, split="dagger_train")["execution"]["metadata"]
        assert len(metadata["three_phase_voltages"]) == 14, row["scenario_family"]
        assert len(metadata["three_phase_branch_currents"]) == 20, row["scenario_family"]
        assert metadata["three_phase_sigma"] == PMU_PHASOR_SIGMA_PU
        assert metadata["branch_current_sigma_pu"] == PMU_PHASOR_SIGMA_PU
        assert "hif_runtime" not in metadata and "hif_scan_window" not in metadata
        screen = zero_sequence_line_screen(metadata["three_phase_voltages"], metadata["three_phase_branch_currents"],
                                           sigma_pu=PMU_PHASOR_SIGMA_PU)
        assert screen["hif_like"] is (row["scenario_family"] in {"hif", "measurement+hif"}), row["scenario_family"]
        # Phasors come from their own random stream: SCADA draws are those of
        # the WLS-gated build with the same seed.
        assert row["measurements"] == gated[row["scenario_id"]]["measurements"], row["scenario_family"]


# ------------------------------------------------------------------ environment


def test_phasors_are_admitted_only_on_a_suspicion(roots):
    factory = _environment()
    meter = next(root for root in roots if root["grouping"]["scenario_family"] == "measurement")
    hif = next(root for root in roots if root["grouping"]["scenario_family"] == "hif")
    for scenario, suspected in ((meter, False), (hif, True)):
        env = factory()
        state = env.reset(scenario["execution"])
        active = state["active_state_id"]
        _, wls = env.step({"tool": RUN_WLS, "arguments": {"state_id": active}})
        metrics = wls["tool_metrics"]
        assert metrics["chi_square_alarm"] or metrics["normalized_residual_alarm"]
        assert metrics["hif_screen"]["suspected"] is suspected
        signature = any(str(item).startswith("wls_hif_suspected") for item in metrics["unresolved_signatures"])
        assert signature is suspected
        _, phasors = env.step({"tool": GET_THREE_PHASE_CONTEXT, "arguments": {"state_id": active}})
        if suspected:
            assert phasors["execution_status"] == "success", phasors
            assert phasors["tool_metrics"]["three_phase_context_status"] == "available"
        else:
            assert phasors["error_code"] == "diagnostics_require_hif_suspicion"
        _, spectra = env.step({"tool": GET_HARMONIC_CONTEXT, "arguments": {"state_id": active}})
        assert spectra["error_code"] == "diagnostics_require_harmonic_suspicion"


def test_the_expert_requests_phasors_only_on_a_suspicion_and_solves(roots):
    factory = _environment()

    def expert(**_kwargs):
        return research.research_expert_policy(factory)

    result = evaluate_rollout_suites(
        {"standard_success": roots}, env_factory=factory, policy_factory=expert,
        max_steps=research.RESEARCH_EPISODE_BUDGET, seed=4, case_loader=deterministic_case_loader,
        **research.PAIRED_EVALUATION_CONTRACT,
    ).as_dict()
    auxiliary = {GET_THREE_PHASE_CONTEXT, RUN_THREE_PHASE_NLM_FROM_PATH, ESTIMATE_HIF_FROM_PATH,
                 GET_HARMONIC_CONTEXT, RUN_HSE_FROM_PATH}
    for episode in result["suite_metrics"]["episodes"]:
        family = episode["family"]
        tools = [step["action"]["tool"] for step in episode["trace"]]
        assert episode["truth_audited_task_success"], (family, tools)
        if family == "hif":
            assert tools[:5] == [RUN_WLS, GET_THREE_PHASE_CONTEXT, RUN_THREE_PHASE_NLM_FROM_PATH,
                                 ESTIMATE_HIF_FROM_PATH, RUN_WLS], tools
        elif family == "measurement+hif":
            assert tools[1] == GET_THREE_PHASE_CONTEXT and "correct_measurements" in tools, tools
        else:
            assert not auxiliary & set(tools), (family, tools)
    assert json.dumps(result["suite_metrics"]["episodes"][0]["trace"][0]["policy_observation"]).count("hif_runtime") == 0
