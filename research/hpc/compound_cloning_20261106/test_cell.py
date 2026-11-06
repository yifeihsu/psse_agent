"""What can be checked of the cloning cell without the cluster: no rollout, no model, no GPU."""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

CELL = Path(__file__).resolve().parent
PIPELINE = CELL.parent / "full_pipeline_20260907"
GPU_STAGES = ("stage_clone_train.sbatch", "stage_eval.sbatch")
SHELL_FILES = GPU_STAGES + ("cell.env", "eval_suites.env", "submit_cell.sh", "deploy_from_windows.sh", "remote_prepare.sh", "status_cell.sh")


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


env_report = _load(CELL / "env_report.py")


@pytest.fixture
def research(monkeypatch):
    """The runner's module options, restored afterwards: evaluate_suite.py and the factory write them."""
    import scripts.run_dagger_research as module

    environment = dict(module.RESEARCH_ENVIRONMENT_OPTIONS)
    expert = dict(module.RESEARCH_EXPERT_OPTIONS)
    monkeypatch.setenv("PSSE_EXPERT_VARIANT", env_report.EXPERT_VARIANT)
    yield module
    module.RESEARCH_ENVIRONMENT_OPTIONS.clear()
    module.RESEARCH_ENVIRONMENT_OPTIONS.update(environment)
    module.RESEARCH_EXPERT_OPTIONS.clear()
    module.RESEARCH_EXPERT_OPTIONS.update(expert)


@pytest.mark.parametrize("name", SHELL_FILES)
def test_scripts_are_lf_only_and_name_no_row_count(name: str) -> None:
    text = (CELL / name).read_bytes().decode("utf-8")
    assert "\r" not in text
    # The selection is re-made: the step count comes from the row file.
    assert not re.search(r"\b(3,?740|2,?297|2,?430|14,?960|9,?185|9,?700)\b", text)


def test_default_suite_list_is_ieee14_and_the_conventional_test() -> None:
    text = (CELL / "eval_suites.env").read_text(encoding="utf-8")
    assert re.findall(r"^eval_suite (\S+)", text, flags=re.MULTILINE) == ["ieee14_dev", "conventional_test"]
    # IEEE 118 is one uncommented line away, and no stage or script names any suite.
    assert re.findall(r"^# eval_suite (\S+)", text, flags=re.MULTILINE) == ["ieee118_draw1", "ieee118_draw2"]
    for name in GPU_STAGES + ("submit_cell.sh", "cell.env"):
        stage = (CELL / name).read_text(encoding="utf-8")
        assert "ieee14_dev" not in stage and "ieee118_draw" not in stage and "compound_errors_" not in stage, name
    assert "ieee118" not in re.search(r"^UPLOAD_SUITES=.*$", (CELL / "deploy_from_windows.sh").read_text(encoding="utf-8"), flags=re.MULTILINE).group(0)


@pytest.mark.parametrize("name", GPU_STAGES)
def test_gpu_stages_follow_the_cluster_routing_rules(name: str) -> None:
    text = (CELL / name).read_text(encoding="utf-8")
    header = [line for line in text.splitlines() if line.startswith("#SBATCH")]
    for line in ("--account=torch_pr_627_general", "--constraint=a100|h100|h200|rtx6000", "--comment=preemption=yes;requeue=true",
                 "--requeue", "--gres=gpu:1", "--time=12:00:00"):
        assert f"#SBATCH {line}" in header
    assert not any(line.startswith("#SBATCH --partition") for line in header)
    assert 'source "$WORK/source/research/hpc/compound_cloning_20261106/cell.env"' in text and "cell_environment" in text
    assert "another_link_running" in text  # links of a chain never share an output directory


def test_training_cadence_is_derived_from_the_file_and_respects_the_slot() -> None:
    env = (CELL / "cell.env").read_text(encoding="utf-8")
    save = int(re.search(r"^CLONE_SAVE_STEPS=(\d+)$", env, flags=re.MULTILINE).group(1))
    evaluate = int(re.search(r"^CLONE_EVAL_STEPS=(\d+)$", env, flags=re.MULTILINE).group(1))
    assert save <= 32 and evaluate % save == 0
    stage = (CELL / "stage_clone_train.sbatch").read_text(encoding="utf-8")
    assert '--save-steps "$SAVE_STEPS"' in stage and '--eval-steps "$EVAL_STEPS"' in stage
    assert '--max-steps "$TRAIN_MAX_STEPS"' in stage and "print(math.ceil(counts[0] / 4), counts[0])" in stage
    # A dated run name would fail the finalization of a pass that crosses midnight.
    assert "date" not in re.search(r"--run-name .*", stage).group(0)


def test_training_renders_the_run_time_tool_list_unless_told_otherwise() -> None:
    from psse_env.actions import lifecycle_choice_record
    from psse_env.dagger.dataset_builder import LIFECYCLE_CHOICE_ROLLBACK_DESCRIPTION, tool_schemas_for_observation
    from psse_env.dagger.preliminary_e2b_eval import canonical_prompt_tool_schemas
    from psse_env.sft.research_rows import normalize_research_rows

    state = {"evidence_profile": "scada_only", "lifecycle_choice": lifecycle_choice_record()}
    row = {"example_id": "r", "metadata": {"protocol": "canonical"}, "tools": [],
           "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": json.dumps({"state": state})},
                        {"role": "assistant", "content": ""}]}
    stock, _report = normalize_research_rows([row], source_label="probe")
    rendered, report = _load(CELL / "train_runtime_tools.py").normalize_with_runtime_tools([row], source_label="probe")
    def rollback(tools: list) -> str:
        return next(tool for tool in tools if tool["function"]["name"] == "rollback_state")["function"]["description"]
    # What this file exists for: the stock normalization renders something else than the policy does.
    assert len(stock[0]["tools"]) > len(rendered[0]["tools"]) and rollback(stock[0]["tools"]) != LIFECYCLE_CHOICE_ROLLBACK_DESCRIPTION
    assert rendered[0]["tools"] == tool_schemas_for_observation(canonical_prompt_tool_schemas(), state)
    assert rollback(rendered[0]["tools"]) == LIFECYCLE_CHOICE_ROLLBACK_DESCRIPTION
    assert report["rendered_tools"]["rows_by_tool_count"] == {str(len(rendered[0]["tools"])): 1}
    assert re.search(r"^CLONE_TOOLS=runtime$", (CELL / "cell.env").read_text(encoding="utf-8"), flags=re.MULTILINE)


def test_cell_env_exports_the_teacher_the_report_names() -> None:
    assert f"export PSSE_EXPERT_VARIANT={env_report.EXPERT_VARIANT}" in (CELL / "cell.env").read_text(encoding="utf-8")


def test_report_follows_the_system() -> None:
    small = env_report.build_report("case14")
    large = env_report.build_report("case118")
    named = {"lifecycle_choice": 1, "detailed_evidence": 1, "pair_refit_adjacent": 1, "flow_at_noise_cue": 1, "route_comparison": 0,
             "normalized_residual_threshold": 4.0, "triage_classifier": None, "evidence_profile": "scada_only"}
    for report in (small, large):
        assert named.items() <= report["environment_options"].items()
        assert report["research_profile"]["evidence_profile"] == "scada_only"
        assert report["research_profile"]["hif_search_profile"] == "research"
        assert report["model_selection"]["model_id"] == "google/gemma-4-12B-it"
    assert "progress_basis" not in small["environment_options"]
    assert large["environment_options"]["progress_basis"] == "chi_square_excess"
    assert small["research_profile"]["multi_scan_rcond"] == "dense_svd"
    assert large["research_profile"]["multi_scan_rcond"] == "sparse_symmetric"
    with pytest.raises(SystemExit):
        env_report.build_report("case57")


@pytest.mark.parametrize("system", ("case14", "case118"))
def test_environment_built_from_the_report_is_the_rows_environment(system: str, research, monkeypatch) -> None:
    assert env_report.environment_flags(env_report.build_report(system)) == env_report.expected_flags(system)
    # The route teacher's variant would turn the route comparison on; the report's explicit 0 holds.
    monkeypatch.setenv("PSSE_EXPERT_VARIANT", "baseline_compare_guarded_routes")
    assert env_report.environment_flags(env_report.build_report(system)) == env_report.expected_flags(system)
    # And under the cell's own variant no teacher supplies the cue: only the report's explicit 1 does.
    monkeypatch.setenv("PSSE_EXPERT_VARIANT", env_report.EXPERT_VARIANT)
    without_cue = env_report.build_report(system)
    del without_cue["environment_options"]["flow_at_noise_cue"]
    research.RESEARCH_ENVIRONMENT_OPTIONS.pop("flow_at_noise_cue", None)
    assert env_report.environment_flags(without_cue)["flow_at_noise_cue"] is False


def test_evaluate_suite_accepts_the_report_unchanged(tmp_path: Path, research) -> None:
    env_report.write_report(tmp_path / "env", "case118")
    suite = tmp_path / "suite.json"
    suite.write_text(json.dumps([{"grouping": {"physical_root_fingerprint": "a", "scenario_family": "compound"}}]), encoding="utf-8")
    evaluate = _load(PIPELINE / "evaluate_suite.py")
    # Everything before the adapter path passed: the report, the teacher check, the options, the suite.
    with pytest.raises(FileNotFoundError):
        evaluate.main(["--cell-dir", str(tmp_path / "env" / "case118"), "--suite", str(suite), "--output-dir", str(tmp_path / "out"),
                       "--name", "x", "--policy", f"before={tmp_path / 'no_adapter'}", "--seed", "1"])
    assert research.RESEARCH_ENVIRONMENT_OPTIONS["progress_basis"] == "chi_square_excess"
    assert research.RESEARCH_ENVIRONMENT_OPTIONS["route_comparison"] == 0


def _root() -> dict:
    return {"execution": {"scenario_id": "s1"},
            "grouping": {"scenario_family": "compound", "case_id": "case14", "physical_root_fingerprint": "p1"},
            "audit": {"compound": {"class": "rx_cb_meter"}, "parameter_ranking": {"stratum": "dominant"},
                      "truth": {"true_parameter_errors": [{"line_index1": 7}], "true_topology_errors": [{"cb_name": "CB_1"}],
                                "true_measurement_errors": []}}}


def _step(tool: str, disposition: str | None = None, **arguments) -> dict:
    return {"action": {"tool": tool, "arguments": arguments}, "execution_status": "success", "candidate_disposition_offline": disposition}


def _episode() -> dict:
    trace = [
        _step("correct_parameters", line_index=3), _step("run_wls"), _step("rollback_state", "ACCEPT_PARTIAL"),  # a wrong target declined
        _step("correct_parameters", line_index=7), _step("run_wls"), _step("rollback_state", "ACCEPT_PARTIAL"),  # a true target declined
        _step("correct_parameters", line_index=7), _step("run_wls"), _step("commit_state", "ACCEPT_PARTIAL"),  # a true target kept
        _step("correct_parameters", line_index=3), _step("run_wls"), _step("commit_state", "ACCEPT_FINAL"),  # a wrong target kept
        _step("correct_parameters", line_index=4), _step("run_wls"), _step("rollback_state", "REJECT"),  # a rejection is no decline
        _step("ask_for_more_evidence", request="operator_escalation:post_correction_confirmation"),
    ]
    return {"scenario_id": "s1", "physical_root": "p1", "trace": trace, "truth_audited_task_success": True, "terminal": True,
            "terminal_outcome": "operator_escalation", "policy_steps": len(trace),
            "decline_accounting": {"declines": 2, "events": [{"step": 2, "disposition": "ACCEPT_PARTIAL"}, {"step": 5, "disposition": "ACCEPT_PARTIAL"}]}}


def test_outcomes_keep_the_lifecycle_and_join_the_class(tmp_path: Path) -> None:
    outcomes = _load(CELL / "episode_outcomes.py")
    suite = tmp_path / "suite.json"
    suite.write_text(json.dumps([_root()]), encoding="utf-8")
    run = tmp_path / "run"
    run.mkdir()
    (run / "policies.json").write_text(json.dumps({"policies": {"bc0": {"label": "after", "adapter": "/x", "tree_sha256": "0"}}, "seed": 1}), encoding="utf-8")
    (run / "bc0_eval.json").write_text(json.dumps({"suite_metrics": {"episodes": [_episode()]}}), encoding="utf-8")
    summary = outcomes.process(suite, run)["policies"]["after"]
    record = json.loads((run / "outcomes_after.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert record["scenario_id"] == "s1" and record["class"] == "rx_cb_meter" and record["group"] == "breaker"
    assert record["decline_accounting"]["declines"] == 2 and record["declines"] == 2
    overall = summary["overall"]
    assert overall["solved"] == 1 and overall["wrong_corrections_kept"] == 1 and overall["true_corrections_kept"] == 1
    assert overall["true_target_declined"] == 1 and overall["wrong_target_declined"] == 1
    assert overall["episodes_decline_count_differs"] == 0 and overall["errors"] == 0
    assert summary["by_group"]["breaker"]["episodes"] == 1 and summary["by_class"]["rx_cb_meter"]["episodes"] == 1
    # The study's paired comparison reads these records as they are.  That script is not part of every
    # checkout, so the rest of this test runs only where it is present.
    compare = pytest.importorskip("research.compound_errors.commit_reader_loop").compare

    report, _text = compare(run / "outcomes_after.jsonl", [("after", run / "outcomes_after.jsonl")])
    assert report["runs"]["after"]["declined_wrong"] == 1 and report["runs"]["after"]["declined_true"] == 1
    assert report["runs"]["after"]["wrong_kept"] == 1


def test_smoke_rows_start_with_the_longest_and_cover_every_group() -> None:
    smoke = _load(CELL / "smoke_inputs.py")
    def line(index: int, system: str, tool: str, size: int, decline: bool = False) -> str:
        return json.dumps({"example_id": str(index), "messages": [{"role": "user", "content": "x" * size}],
                           "cloning": {"tool": tool}, "cloning_selection": {"system": system, "decline_row": decline}})
    lines = [line(0, "ieee14", "run_wls", 10), line(1, "ieee14", "run_wls", 20), line(2, "ieee14", "rollback_state", 10, True),
             line(3, "ieee118", "commit_state", 500), line(4, "ieee118", "run_wls", 10), line(5, "ieee14", "commit_state", 10)]
    chosen = smoke.choose_rows(lines, 4, 7)
    assert chosen[0] == 3 and len(set(chosen)) == 4 and chosen == smoke.choose_rows(lines, 4, 7)
    groups = {smoke._row_group(json.loads(lines[index])) for index in chosen}
    assert ("ieee14", "decline") in groups and ("ieee118", "run_wls") in groups
    assert len(smoke.choose_rows(lines, 50, 7)) == len(lines)


def test_smoke_suite_slice_covers_the_classes() -> None:
    smoke = _load(CELL / "smoke_inputs.py")
    roots = [{"id": i, "audit": {"compound": {"class": "a" if i < 3 else "b"}}} for i in range(5)]
    assert [root["id"] for root in smoke.choose_roots(roots, 3)] == [0, 3, 1]
    assert len(smoke.choose_roots([{"grouping": {"scenario_family": "topology"}}], 8)) == 1
