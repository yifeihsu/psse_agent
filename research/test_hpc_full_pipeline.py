from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

from psse_env.systems import resolve_system

CELL = Path(__file__).resolve().parent / "hpc" / "full_pipeline_20260907"
GPU_STAGES = (
    "stage_bc0.sbatch", "stage_collect.sbatch", "stage_train.sbatch", "stage_eval.sbatch",
    "stage_zeroshot.sbatch", "stage_control_train.sbatch", "stage_eval_suite.sbatch",
)
CPU_STAGES = ("stage_d0.sbatch",)
# Follow-up stages of a finished cell (2026-10-23): CPU work outside stage 0.
EXTRA_CPU_STAGES = ("stage_extra_cpu.sbatch", "stage_test_draw.sbatch")
SHELL_FILES = ("prerequisites.sh", "submit_pipeline.sh", "status_pipeline.sh", "deploy_remote.sh",
               "submit_extras_20261023.sh", "submit_extras_20261027.sh", "submit_test_draw2_20261029.sh")
PYTHON_FILES = ("build_suite.py", "summarize.py")


def _load(name: str):
    path = CELL / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", GPU_STAGES)
def test_gpu_stages_follow_the_cluster_routing_rules(name: str) -> None:
    text = (CELL / name).read_text(encoding="utf-8")
    header = [line for line in text.splitlines() if line.startswith("#SBATCH")]
    assert "#SBATCH --account=torch_pr_627_general" in header
    assert "#SBATCH --constraint=a100|h100|h200|rtx6000" in header
    assert "#SBATCH --comment=preemption=yes;requeue=true" in header
    assert "#SBATCH --requeue" in header
    assert "#SBATCH --gres=gpu:1" in header
    assert not any(line.startswith("#SBATCH --partition") for line in header)
    assert 'source "$PIPE/pipeline.env"' in text
    assert "pipeline_environment" in text
    # Collection and evaluation run the parallel OpenDSS search beside the
    # policy GPU; training stages need no more than the default.
    expected_cpus = (
        16 if name in ("stage_collect.sbatch", "stage_eval.sbatch", "stage_zeroshot.sbatch", "stage_eval_suite.sbatch")
        else 8
    )
    assert f"#SBATCH --cpus-per-task={expected_cpus}" in header


@pytest.mark.parametrize("name", EXTRA_CPU_STAGES)
def test_extra_cpu_stages_request_no_gpu_and_are_deployed(name: str) -> None:
    text = (CELL / name).read_text(encoding="utf-8")
    header = [line for line in text.splitlines() if line.startswith("#SBATCH")]
    assert "#SBATCH --account=torch_pr_627_general" in header
    assert not any("gpu" in line for line in header)
    assert any(line.startswith("#SBATCH --partition=") for line in header)
    assert "\r" not in text
    deploy = (CELL / "deploy_remote.sh").read_text(encoding="utf-8")
    for deployed in (name, "stage_control_train.sbatch", "stage_eval_suite.sbatch", "submit_extras_20261023.sh",
                     "submit_extras_20261027.sh", "stage_test_draw.sbatch", "submit_test_draw2_20261029.sh"):
        assert deployed in deploy, deployed


def test_control_mixture_replaces_each_collected_row_by_a_fresh_row_of_its_family() -> None:
    builder = _load("build_control_mixture.py")
    def row(example_id: str, family: str, root: str) -> dict:
        return {"example_id": example_id, "scenario_family": family, "physical_root_fingerprint": root}
    mixture = [row("d0a", "parameter", "p0"), row("d1a", "parameter", "c1"), row("d1b", "topology", "c2"),
               row("d0b", "topology", "p1"), row("d1c", "parameter", "c3")]
    fresh = [row(f"f{i}", family, f"n{i}") for i, family in enumerate(["parameter", "parameter", "parameter", "topology"])]
    result, report = builder.replace_collected_rows(mixture, {"d1a", "d1b", "d1c"}, fresh, seed=7)
    assert len(result) == len(mixture)
    assert [r["scenario_family"] for r in result] == [r["scenario_family"] for r in mixture]
    assert result[0]["example_id"] == "d0a" and result[3]["example_id"] == "d0b"  # D0 rows keep their places
    assert {result[1]["example_id"], result[4]["example_id"]} <= {"f0", "f1", "f2"} and result[2]["example_id"] == "f3"
    assert report["replaced_by_family"] == {"parameter": 2, "topology": 1}
    again, _ = builder.replace_collected_rows(mixture, {"d1a", "d1b", "d1c"}, fresh, seed=7)
    assert again == result  # seeded
    with pytest.raises(ValueError, match="short per family"):
        builder.replace_collected_rows(mixture, {"d1a", "d1b", "d1c"}, fresh[:1] + fresh[3:], seed=7)
    with pytest.raises(ValueError, match="kept D0 rows"):  # a fresh root that is a kept D0 root
        builder.replace_collected_rows(mixture, {"d1a", "d1b", "d1c"},
                                       [row("g0", "parameter", "p0"), row("g1", "parameter", "p1"), row("g2", "topology", "n9")], seed=7)


def test_eval_suites_keep_conventional_roots_outside_every_excluded_set() -> None:
    suites = _load("build_eval_suites.py")
    def row(root: str, family: str) -> dict:
        return {"grouping": {"physical_root_fingerprint": root, "scenario_family": family}}
    rows = [row("a", "parameter"), row("b", "harmonic"), row("c", "topology"), row("d", "measurement")]
    kept, report = suites.select(rows, {"d0": {"c"}, "suite:development": {"c", "d"}},
                                 lambda r: r["grouping"]["physical_root_fingerprint"])
    assert [r["grouping"]["physical_root_fingerprint"] for r in kept] == ["a"]
    assert report["dropped"] == {"c": ["d0", "suite:development"], "d": ["suite:development"]}
    assert report["other_families"] == {"harmonic": 1}


def test_followup_tables_pair_by_root_and_apply_the_selection_rule(tmp_path: Path) -> None:
    tables = _load("followup_tables.py")
    assert tables.mcnemar(19, 0) == pytest.approx(2 * 0.5 ** 19) and tables.mcnemar(0, 0) == 1.0
    rows = [{"grouping": {"physical_root_fingerprint": f"root{i}", "scenario_family": "parameter" if i < 2 else "measurement"},
             "audit": {"parameter_ranking": {"stratum": "misranked" if i == 0 else "dominant"}}} for i in range(4)]
    def episode(i: int, ok: bool, invalid: int = 0) -> dict:
        return {"physical_root": f"root{i}", "truth_audited_task_success": ok, "invalid_action_count": invalid,
                "false_commit_count": 0, "loop_detected": False, "steps": 7, "wls_calls": 2, "tool_counts": {}}
    def write(path: Path, episodes: list) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"suite_metrics": {"episodes": episodes}}), encoding="utf-8")
    out = tmp_path / "out"
    (out / "r1" / "collection").mkdir(parents=True)
    (out / "r1" / "collection" / "development_scenarios.json").write_text(json.dumps(rows), encoding="utf-8")
    (out / "extra" / "suites").mkdir(parents=True)
    for suite in ("hard", "validation"):
        (out / "extra" / "suites" / f"{suite}.json").write_text(json.dumps(rows), encoding="utf-8")
    ok = [episode(i, True) for i in range(4)]
    write(out / "r1" / "collection" / "evaluation" / "expert_eval.json", ok)
    write(out / "r1" / "collection" / "evaluation" / "bc0_eval.json", [episode(0, False, 2), episode(1, True, 1)] + ok[2:])
    write(out / "r1" / "collection" / "evaluation" / "r1_eval.json", ok)
    write(out / "r2" / "collection" / "evaluation" / "r1_eval.json", [episode(0, True, 1)] + ok[1:])
    write(out / "extra" / "evaluations" / "test_control" / "bc0_eval.json", [episode(0, False, 1)] + ok[1:])
    for suite in ("hard", "validation"):
        write(out / "extra" / "evaluations" / f"{suite}_bc0_r1" / "expert_eval.json", ok)
        write(out / "extra" / "evaluations" / f"{suite}_bc0_r1" / "bc0_eval.json", [episode(0, False)] + ok[1:])
        write(out / "extra" / "evaluations" / f"{suite}_bc0_r1" / "r1_eval.json", ok)
        write(out / "extra" / "evaluations" / f"{suite}_r2_control" / "bc0_eval.json", [episode(1, True, 1)] + ok[:1] + ok[2:])
        write(out / "extra" / "evaluations" / f"{suite}_r2_control" / "r1_eval.json", ok)
    result = tables.build(tmp_path)
    test = result["test"]
    assert test["policies"]["bc0"]["successes"] == 3 and test["policies"]["bc0"]["violation_episodes"] == 2
    assert test["policies"]["bc0"]["by_stratum"]["misranked"] == {"successes": 0, "episodes": 1, "clean_successes": 0}
    assert test["paired_against_r1"]["bc0_vs_r1"]["violations"]["only_first"] == 2
    assert test["paired_against_r1"]["control_vs_r1"]["success_failures"]["only_first"] == 1
    # Safety-clean success (2026-10-27): BC0 fails root0 and solves root1 with an invalid action, so 2 of its 4 are clean.
    assert test["policies"]["bc0"]["clean_successes"] == 2 and test["policies"]["r1"]["clean_successes"] == 4
    assert test["policies"]["bc0"]["by_family"]["parameter"] == {"successes": 1, "episodes": 2, "clean_successes": 0}
    assert test["paired_against_bc0"]["r1_vs_bc0"]["clean_success_failures"] == {
        "paired_roots": 4, "only_first": 0, "only_second": 2, "p": 0.5}
    assert test["paired_against_r1"]["bc0_vs_r1"]["clean_success_failures"]["only_first"] == 2
    assert "safety-clean successes" in tables.markdown(result) and "r1_vs_bc0" in tables.markdown(result)
    # An extra adapter evaluated alone on the three suites is joined by its label.
    write(out / "extra" / "evaluations" / "test_bcx" / "bc0_eval.json", [episode(0, False)] + ok[1:])
    extra = tables.build(tmp_path, ["bcx"])
    assert extra["test"]["policies"]["bcx"]["successes"] == 3 and "bcx_vs_bc0" in extra["test"]["paired_against_bc0"]
    assert "hard" in extra and "bcx" not in extra["hard"]["policies"]
    with pytest.raises(ValueError):
        tables.suite_sources(tmp_path, ["bc0"])
    # R1 and R2 tie on successes (4 each); R1 has fewer violations, so R1 is selected.
    assert result["round_selection"]["selected"] == "r1" and result["round_selection"]["order"] == ["r1", "r2", "bc0"]
    assert "### test" in tables.markdown(result)


def test_suite_evaluation_takes_one_or_two_labelled_adapters() -> None:
    evaluate = _load("evaluate_suite.py")
    assert evaluate.parse_policies(["bc0=/x/a", "r1=/x/b"]) == [("bc0", Path("/x/a")), ("r1", Path("/x/b"))]
    for bad in (["a"], ["a=/x", "a=/y"], ["expert=/x"], ["a=/x", "b=/y", "c=/z"], []):
        with pytest.raises(ValueError):
            evaluate.parse_policies(bad)


def test_budget_tables_join_budgets_by_root_and_compare_with_the_approximation(tmp_path: Path) -> None:
    budget = _load("budget_tables.py")
    rows = [{"grouping": {"physical_root_fingerprint": f"root{i}",
                          "scenario_family": "multi_measurement" if i < 2 else "measurement"}, "audit": {}} for i in range(4)]

    def episode(i: int, ok: bool, steps: int, invalid: int = 0, terminal: bool = True) -> dict:
        return {"physical_root": f"root{i}", "truth_audited_task_success": ok, "invalid_action_count": invalid,
                "false_commit_count": 0, "loop_detected": False, "steps": steps, "wls_calls": 2, "tool_counts": {},
                "terminal": terminal}

    def write(path: Path, payload: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def evaluation(*episodes: dict) -> dict:
        return {"suite_metrics": {"episodes": list(episodes)}}

    out = tmp_path / "out"
    collection = out / "r1" / "collection"
    write(collection / "development_scenarios.json", rows)
    write(collection / "evaluation" / "comparison.json", {"max_steps": 40})
    # At 40 actions: the expert solves root0 in 20 and root1 in 10; BC0 solves root0 in 14 with an invalid action and
    # runs root1 to the budget; R1 solves everything.
    write(collection / "evaluation" / "expert_eval.json",
          evaluation(episode(0, True, 20), episode(1, True, 10), episode(2, True, 7), episode(3, True, 7)))
    write(collection / "evaluation" / "bc0_eval.json",
          evaluation(episode(0, True, 14, 1), episode(1, False, 40, 0, False), episode(2, True, 7), episode(3, True, 7)))
    write(collection / "evaluation" / "r1_eval.json",
          evaluation(episode(0, True, 18), episode(1, True, 10), episode(2, True, 7), episode(3, True, 7)))
    write(out / "bc0.done", {"adapter_tree_sha256": "a" * 64})
    write(out / "r1" / "training.done", {"candidate_tree_sha256": "b" * 64})
    write(out / "r2" / "training.done", {"candidate_tree_sha256": "c" * 64})
    # R2 at 40 is the candidate slot of the round-2 evaluation: it loses root1 in 40 actions.
    write(out / "r2" / "collection" / "evaluation" / "comparison.json", {"max_steps": 40})
    write(out / "r2" / "collection" / "evaluation" / "r1_eval.json",
          evaluation(episode(0, True, 16), episode(1, False, 40, 0, False), episode(2, True, 7), episode(3, True, 7)))
    # At 12 actions: the expert hands root0 off; BC0 runs root0 to the budget and solves root1 in 11 this time (which
    # the approximation from the 40-action episode cannot see); R1 solves root2 with an invalid action.
    evaluated = out / "extra" / "evaluations" / "test_budget12"
    write(evaluated / "policies.json", {"max_steps": 12, "seed": 1, "expert": "baseline_compare", "policies": {
        "bc0": {"label": "bc0", "tree_sha256": "a" * 64}, "r1": {"label": "r1", "tree_sha256": "b" * 64}}})
    write(evaluated / "summary.json", {})
    write(evaluated / "expert_eval.json",
          evaluation(episode(0, False, 9), episode(1, True, 10), episode(2, True, 7), episode(3, True, 7)))
    write(evaluated / "bc0_eval.json",
          evaluation(episode(0, False, 12, 0, False), episode(1, True, 11), episode(2, True, 7), episode(3, True, 7)))
    write(evaluated / "r1_eval.json",
          evaluation(episode(0, False, 12, 0, False), episode(1, True, 10), episode(2, True, 7, 1), episode(3, True, 7)))
    # R2 alone at 12 actions, in the first adapter slot of its own evaluation.
    r2_dir = out / "extra" / "evaluations" / "test_budget12_r2"
    write(r2_dir / "policies.json", {"max_steps": 12, "seed": 1, "expert": None,
                                     "policies": {"bc0": {"label": "r2", "tree_sha256": "c" * 64}}})
    write(r2_dir / "summary.json", {})
    write(r2_dir / "bc0_eval.json",
          evaluation(episode(0, False, 12, 0, False), episode(1, False, 12, 1, False), episode(2, True, 7), episode(3, True, 7)))
    result = budget.build(tmp_path, [12, 16])
    assert sorted(result["budgets"]) == ["12", "40"] and result["full_budget"] == 40
    assert result["budgets"]["40"]["policies"]["r2"]["successes"] == 3
    assert result["budgets"]["12"]["policies"]["r2"]["successes"] == 2
    assert result["budgets"]["12"]["policies"]["r2"]["by_cohort"] == {
        "multi_error": {"episodes": 2, "successes": 0, "clean_successes": 0, "violation_episodes": 1},
        "single_or_clean": {"episodes": 2, "successes": 2, "clean_successes": 2, "violation_episodes": 0}}
    assert result["budgets"]["12"]["paired"]["r1_vs_r2"]["success_failures"]["only_second"] == 1
    assert result["budgets"]["12"]["approximation_from_full_budget"]["r2"]["success"]["approximated"] == 2
    assert result["evaluations"]["12_r2"]["adapters_match_receipts"] is True
    at12 = result["budgets"]["12"]
    assert at12["policies"]["expert"]["successes"] == 3 and at12["policies"]["bc0"]["successes"] == 3
    assert at12["policies"]["r1"]["successes"] == 3 and at12["policies"]["r1"]["clean_successes"] == 2
    assert at12["policies"]["bc0"]["budget_exhausted_episodes"] == 1 and at12["policies"]["bc0"]["non_terminal_episodes"] == 1
    assert at12["policies"]["bc0"]["by_family"]["multi_measurement"] == {"successes": 1, "episodes": 2, "clean_successes": 1}
    approximation = at12["approximation_from_full_budget"]
    assert approximation["bc0"]["success"]["approximated"] == 2 and approximation["bc0"]["success"]["measured"] == 3
    assert approximation["bc0"]["success"]["only_measured_roots"] == ["root1"]
    assert approximation["bc0"]["success"]["only_approximated"] == 0
    assert approximation["expert"]["success"]["only_measured"] == 0 and approximation["expert"]["success"]["only_approximated"] == 0
    assert at12["paired"]["bc0_vs_r1"]["success_failures"]["only_first"] == 0
    assert at12["paired"]["r1_vs_expert"]["clean_success_failures"]["only_first"] == 1
    assert result["evaluations"]["12"]["adapters_match_receipts"] is True
    assert result["budgets"]["40"]["approximation_from_full_budget"] == {}
    text = budget.markdown(result)
    assert "### Success under the action budget" in text and "only measured 1" in text
    assert "### Multi-error roots" in text and "r2 vs expert" in text
    write(evaluated / "policies.json", {"max_steps": 16, "policies": {}})
    with pytest.raises(ValueError):
        budget.build(tmp_path, [12])


def test_budget_submit_script_writes_the_horizon_into_its_specs() -> None:
    submit = (CELL / "submit_extras_20261027.sh").read_text(encoding="utf-8")
    assert 'BUDGETS=${BUDGETS:-"12 16 20"}' in submit
    assert "EVAL_MAX_STEPS=%s" in submit and "stage_eval_suite.sbatch" in submit
    assert '"bc0=$BC0" "r1=$R1"' in submit and 'out/r1/round_summary.json' not in submit
    # An adapter alone at every budget (R2, 2026-10-27) and, by default, on the three 40-action suites.
    assert 'test_budget${budget}_$label' in submit and '${ADAPTER_BUDGETS:-0}' in submit and '${ADAPTER_SUITES:-1}' in submit
    stage = (CELL / "stage_eval_suite.sbatch").read_text(encoding="utf-8")
    # The spec is sourced after pipeline.env, so its EVAL_MAX_STEPS overrides the cell's horizon.
    assert stage.index('source "$PIPE/pipeline.env"') < stage.index('source "$EVAL_SPEC"')
    assert '--max-steps "$EVAL_MAX_STEPS"' in stage and '"${EVAL_MAX_STEPS:?}"' in stage


def test_second_test_draw_repeats_the_first_draws_builder_invocation() -> None:
    stage = (CELL / "stage_test_draw.sbatch").read_text(encoding="utf-8")
    first = (CELL / "stage_d0.sbatch").read_text(encoding="utf-8")
    for flag in ('--round-train-plan "$ROUND_TRAIN_PLAN"', '--development-plan "$DEVELOPMENT_PLAN"', '--rounds "$DAGGER_ROUNDS"',
                 '--training-threshold "$SUITE_TRAINING_THRESHOLD"', '--development-threshold "$SUITE_DEVELOPMENT_THRESHOLD"',
                 '--development-rank-allowance "$SUITE_DEVELOPMENT_RANK_ALLOWANCE"',
                 '--min-measurement-error-sigma "$MEASUREMENT_ERROR_MIN_SIGMA"', '--topology-effects "$TOPOLOGY_EFFECTS"',
                 '"${system_args[@]}"', '"${suite_waveform_args[@]}"', '--d0-raw "$D0_DIR/aggregate.raw.jsonl"',
                 '--protected-suite "$BC0_SUITE"'):
        assert flag in stage and flag in first, flag
    assert '--seed "$TEST_DRAW_SEED"' in stage and '--seed "$SEED_SUITE"' not in stage
    assert '--candidate-multiplier "$multiplier"' in stage
    assert 'for multiplier in "$CANDIDATE_MULTIPLIER" "$TEST_DRAW_FALLBACK_MULTIPLIER"' in stage
    assert '--protected-suite "$DRAW/protected_roots.json"' in stage and "collect_reference_roots.py" in stage and "set -f" in stage
    assert '--suite-dir "$DRAW/suite"' in stage and "check_suite_exclusion.py" in stage and "checkable_paths" in stage
    assert "suite_stats.py" in stage and "draw.done" in stage
    overrides = (CELL / "overrides" / "conventional_only_20261022.env").read_text(encoding="utf-8")
    settings = dict(re.findall(r"^([A-Z0-9_]+)=(.*)$", overrides, flags=re.MULTILINE))
    assert settings["TEST_DRAW_NAME"] == "test_draw2" and settings["TEST_DRAW_PREFIX"] == "test2"
    used = {int(settings[k]) for k in ("SEED_SUITE", "SEED_CONTROL", "SEED_D0", "SEED_ROUND")} | {int(settings["SEED_SUITE"]) + 1}
    seed = int(settings["TEST_DRAW_SEED"])
    assert seed not in used and seed + 1 not in used
    assert int(settings["TEST_DRAW_FALLBACK_MULTIPLIER"]) > 3
    references = settings["TEST_DRAW_REFERENCES"]
    for pattern in ("research_full_pipeline_*/out/d0/aggregate.raw.jsonl", "research_full_pipeline_*/out/suite/development.json",
                    "research_full_pipeline_*/out/suite/r*_training.json", "research_full_pipeline_*/out/extra/suites/hard.json",
                    "research_full_pipeline_*/out/extra/suites/validation.json",
                    "research_full_pipeline_*/out/extra/control_d0/aggregate.raw.jsonl", "hard_draw_20261015/hard_*.json",
                    "$EXTRA_VALIDATION_SOURCE", "$EXTRA_HARD_SOURCE"):
        assert pattern in references, pattern
    submit = (CELL / "submit_test_draw2_20261029.sh").read_text(encoding="utf-8")
    assert "stage_test_draw.sbatch" in submit and '--dependency=afterok:"$dependency"' in submit
    assert "SUITE=$DRAW/suite/development.json" in submit and 'BUDGETS=${BUDGETS:-"12 20"}' in submit
    assert '"${PREFIX}_${FULL}"' in submit and '"${PREFIX}_budget${budget}_r2"' in submit and '"r2=$R2"' in submit


def test_reference_roots_collect_every_readable_set(tmp_path: Path) -> None:
    collect = _load("collect_reference_roots.py")

    def write(path: Path, text: str) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    clean = write(tmp_path / "cellA" / "out" / "d0" / "aggregate.raw.jsonl",
                  json.dumps({"physical_root_fingerprint": "physical_a"}) + "\n"
                  + json.dumps({"metadata": {"physical_root_fingerprint": "physical_b"}}) + "\n")
    partial = write(tmp_path / "cellB" / "out" / "d0" / "aggregate.raw.jsonl",
                    json.dumps({"physical_root_fingerprint": "physical_c"}) + "\n" + json.dumps({"example_id": "x"}) + "\n")
    suite = write(tmp_path / "suite.json", json.dumps([{"grouping": {"physical_root_fingerprint": "physical_d"}}]))
    strings = write(tmp_path / "strings.json", json.dumps(["physical_e", "not a root"]))
    aliased = write(tmp_path / "aliased.json", json.dumps([{"scenario_id": "r0_x", "fp": "physical_f"}]))
    report_file = write(tmp_path / "report.json", json.dumps({"roots": 3}))
    own = write(tmp_path / "draw" / "development.json", json.dumps([{"grouping": {"physical_root_fingerprint": "physical_new"}}]))
    patterns = [str(tmp_path / "cell*" / "out" / "d0" / "aggregate.raw.jsonl"), str(suite), str(suite), str(strings),
                str(aliased), str(report_file), str(tmp_path / "missing*.json"), str(tmp_path / "draw" / "*.json")]
    roots, report = collect.collect(patterns, [tmp_path / "draw"])
    assert roots == {"physical_a", "physical_b", "physical_c", "physical_d", "physical_e", "physical_f"}
    assert report["files"][str(aliased)]["roots"] == 1 and report["files"][str(aliased)]["checkable"] is False
    files = report["files"]
    assert files[str(partial)]["rows_without_root"] == 1 and files[str(partial)]["checkable"] is False
    assert files[str(suite)]["roots"] == 1 and files[str(report_file)]["rows"] == 0
    assert report["patterns"][str(tmp_path / "missing*.json")] == 0 and "skipped" in files[str(own)]
    assert sorted(report["checkable_paths"]) == sorted([str(clean), str(suite)])
    output = tmp_path / "out" / "protected_roots.json"
    assert collect.main(patterns + ["--exclude-dir", str(tmp_path / "draw"), "--output", str(output)]) == 0
    written = json.loads(output.read_text(encoding="utf-8"))
    assert [row["physical_root_fingerprint"] for row in written] == [f"physical_{c}" for c in "abcdef"]
    assert json.loads((tmp_path / "out" / "protected_roots.report.json").read_text(encoding="utf-8"))["roots"] == 6
    # check_suite_exclusion reads the written roots as a reference suite.
    assert _load("check_suite_exclusion.py").reference_roots(output) == roots


def test_suite_stats_reads_the_truth_fields() -> None:
    stats = _load("suite_stats.py")
    sigma = [0.01, 0.01, 0.02]
    rows = [
        {"grouping": {"scenario_family": "measurement", "error_cardinality": 1},
         "audit": {"truth": {"true_measurement_errors": [{"channel": "Pt", "index": 2, "clean": 0.1, "observed": 0.4}]},
                   "parameter_ranking": {"stratum": "not_applicable"}},
         "execution": {"measurements": [1.0, 1.0, 1.0], "metadata": {"sigma_z": sigma}}},
        {"grouping": {"scenario_family": "parameter", "error_cardinality": 1},
         "audit": {"truth": {"true_parameter_errors": [{"r_factor": 1.0, "x_factor": 4.5}]},
                   "parameter_ranking": {"stratum": "ambiguous", "dominance_ratio": 1.05, "true_line_rank": 1}},
         "execution": {"measurements": [1.0, 1.0, 1.0],
                       "metadata": {"sigma_z": sigma, "parameter_scans": {"sigma_z": sigma, "z_scans": [[1], [2], [3]]}}}},
        {"grouping": {"scenario_family": "topology", "error_cardinality": 1},
         "audit": {"truth": {"true_topology_errors": [{"physical_effect": "bus_split", "cb_name": "CB_4_N1_N2", "load_scale": 0.9,
                                                      "operating_point": "ac_opf"}]},
                   "parameter_ranking": {"stratum": "not_applicable"}},
         "execution": {"measurements": [1.0, 1.0, 1.0], "metadata": {"substation_telemetry": {"nodes": []}}}},
        {"grouping": {"scenario_family": "measurement+parameter", "error_cardinality": 2},
         "audit": {"truth": {"true_measurement_errors": [{"index": 75, "clean": 0.0, "observed": 0.2}],
                             "true_parameter_errors": [{"r_factor": 0.5, "x_factor": 2.0}]},
                   "parameter_ranking": {"stratum": "dominant", "dominance_ratio": 2.0, "true_line_rank": 1}},
         "execution": {"measurements": [1.0] * 122, "metadata": {"sigma_z": [0.01] * 122}}},
    ]
    described = stats.describe_rows(rows)
    measurement = described["families"]["measurement"]
    assert measurement["meter_error_sigma"]["median"] == pytest.approx(15.0) and measurement["meter_channels"] == {"Pt": 1}
    parameter = described["families"]["parameter"]
    assert parameter["x_factor"]["max"] == 4.5 and parameter["r_factor_unchanged"] == 1 and parameter["strata"] == {"ambiguous": 1}
    assert parameter["roots_with_parameter_scans"] == 1 and parameter["scans_per_root"]["max"] == 3
    topology = described["families"]["topology"]
    assert topology["topology_effects"] == {"bus_split": 1} and topology["roots_with_substation_telemetry"] == 1
    mixed = described["families"]["measurement+parameter"]
    assert mixed["meter_channels"] == {"Qf": 1} and mixed["meter_error_sigma"]["max"] == pytest.approx(20.0)
    assert "### measurement" in stats.markdown({"a": described, "b": described})


def test_pooled_tables_join_two_root_disjoint_draws(tmp_path: Path) -> None:
    pooled = _load("pooled_tables.py")

    def rows(start: int) -> list[dict]:
        return [{"grouping": {"physical_root_fingerprint": f"root{i}",
                              "scenario_family": "multi_measurement" if i % 2 == 0 else "measurement"}, "audit": {}}
                for i in range(start, start + 4)]

    def episode(i: int, ok: bool, invalid: int = 0) -> dict:
        return {"physical_root": f"root{i}", "truth_audited_task_success": ok, "invalid_action_count": invalid,
                "false_commit_count": 0, "loop_detected": False, "steps": 7, "wls_calls": 2, "tool_counts": {},
                "terminal": True}

    def write(path: Path, payload: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def evaluation(start: int, failing: int = -1, invalid_on: int = -1) -> dict:
        return {"suite_metrics": {"episodes": [episode(i, i != failing, int(i == invalid_on)) for i in range(start, start + 4)]}}

    hashes = {"bc0": "a" * 64, "r1": "b" * 64, "r2": "c" * 64}

    def identity(horizon: int, labels: dict) -> dict:
        return {"max_steps": horizon, "seed": 1,
                "policies": {slot: {"label": label, "tree_sha256": hashes[label]} for slot, label in labels.items()}}

    out = tmp_path / "out"
    collection = out / "r1" / "collection"
    write(collection / "development_scenarios.json", rows(0))
    write(collection / "evaluation" / "comparison.json", {"max_steps": 40})
    write(collection / "evaluation" / "expert_eval.json", evaluation(0))
    write(collection / "evaluation" / "bc0_eval.json", evaluation(0, failing=0, invalid_on=2))
    write(collection / "evaluation" / "r1_eval.json", evaluation(0))
    write(out / "r2" / "collection" / "evaluation" / "comparison.json", {"max_steps": 40})
    write(out / "r2" / "collection" / "evaluation" / "r1_eval.json", evaluation(0, failing=2))
    write(out / "bc0.done", {"adapter_tree_sha256": hashes["bc0"]})
    write(out / "r1" / "training.done", {"candidate_tree_sha256": hashes["r1"]})
    write(out / "r2" / "training.done", {"candidate_tree_sha256": hashes["r2"]})
    extra = out / "extra" / "evaluations"
    write(out / "extra" / "test_draw2" / "suite" / "development.json", rows(4))
    for name, start, horizon in (("test2_40", 4, 40), ("test2_budget12", 4, 12), ("test_budget12", 0, 12)):
        write(extra / name / "policies.json", identity(horizon, {"bc0": "bc0", "r1": "r1"}))
        write(extra / name / "summary.json", {})
        write(extra / name / "expert_eval.json", evaluation(start))
        write(extra / name / "bc0_eval.json", evaluation(start, failing=start + 1, invalid_on=start))
        write(extra / name / "r1_eval.json", evaluation(start, failing=start + 1))
        write(extra / f"{name}_r2" / "policies.json", identity(horizon, {"bc0": "r2"}))
        write(extra / f"{name}_r2" / "summary.json", {})
        write(extra / f"{name}_r2" / "bc0_eval.json", evaluation(start))
    result = pooled.build(tmp_path, "test_draw2", "test2", [12])
    assert sorted(result["draws"]) == ["draw1", "draw2", "pooled"]
    at40 = result["draws"]["pooled"]["40"]
    assert at40["policies"]["expert"]["episodes"] == 8
    assert at40["policies"]["bc0"]["successes"] == 6 and at40["policies"]["bc0"]["clean_successes"] == 4
    assert at40["policies"]["r2"]["successes"] == 7
    assert result["draws"]["pooled"]["12"]["policies"]["r1"]["successes"] == 6
    # BC0 fails root0 (draw 1) and root5 (draw 2), R1 fails root5 only.
    assert at40["paired"]["bc0_vs_r1"]["success_failures"] == {"paired_roots": 8, "only_first": 1, "only_second": 0, "p": 1.0}
    assert at40["paired"]["r1_vs_r2"]["success_failures"]["only_first"] == 1
    assert at40["paired"]["r1_vs_r2"]["success_failures"]["only_second"] == 1
    assert result["draws"]["draw2"]["40"]["policies"]["bc0"]["by_cohort"]["multi_error"]["violation_episodes"] == 1
    assert result["draw_differences_at_full_budget"]["bc0"]["successes"] == [3, 3]
    assert result["draw_differences_at_full_budget"]["r2"]["successes"] == [3, 4]
    assert result["identities"]["draw2"]["40:test2_40"]["adapters_match_receipts"] is True
    assert result["identities"]["draw1"]["12:test_budget12_r2"]["adapters_match_receipts"] is True
    episodes = result.pop("episodes")
    assert len(episodes) == 2 * 2 * 4 * 4
    assert {e["draw"] for e in episodes} == {"draw1", "draw2"} and {e["budget"] for e in episodes} == {12, 40}
    record = next(e for e in episodes if e["draw"] == "draw2" and e["policy"] == "bc0" and e["budget"] == 40 and e["physical_root"] == "root4")
    assert record["invalid_action_count"] == 1 and record["family"] == "multi_measurement" and record["success"] is True
    text = pooled.markdown(result)
    assert "### pooled (8 roots)" in text and "Draw 1 against draw 2" in text
    with pytest.raises(ValueError):
        pooled.pool({"a": {40: {"bc0": {"root0": {}}}}, "b": {40: {"bc0": {"root0": {}}}}})


@pytest.mark.parametrize("name", CPU_STAGES)
def test_cpu_stage_requests_no_gpu(name: str) -> None:
    text = (CELL / name).read_text(encoding="utf-8")
    header = [line for line in text.splitlines() if line.startswith("#SBATCH")]
    assert "#SBATCH --account=torch_pr_627_general" in header
    assert not any("gpu" in line for line in header)
    assert any(line.startswith("#SBATCH --partition=") for line in header)
    assert "--research" in text
    assert "build_suite.py" in text


@pytest.mark.parametrize("name", SHELL_FILES + GPU_STAGES + CPU_STAGES + ("pipeline.env",))
def test_scripts_are_lf_only(name: str) -> None:
    assert "\r" not in (CELL / name).read_bytes().decode("utf-8")


def test_pipeline_env_declares_every_setting_the_stages_use() -> None:
    text = (CELL / "pipeline.env").read_text(encoding="utf-8")
    declared = set(re.findall(r"^([A-Z0-9_]+)=", text, flags=re.MULTILINE))
    for name in (
        "PIPE", "SRC", "OUT", "LOGS", "PY", "TEST_PY", "HF_HOME", "MODEL_CHOICE", "MODEL_ID",
        "MODEL_REVISION", "D0_DIR", "SUITE_DIR", "BC0_DIR", "R1_DIR", "R2_DIR", "BC0_SUITE",
        "HIF_CORPUS_TRAIN", "HIF_CORPUS_VALID", "HIF_CORPUS_TRAIN_EXTRA", "HIF_CORPUS_VALID_EXTRA",
        "IMBALANCE_CORPUS", "SEED_D0", "SEED_SUITE",
        "SEED_ROUND", "D0_PLAN", "ROUND_TRAIN_PLAN", "DEVELOPMENT_PLAN", "DAGGER_ROUNDS",
        "CANDIDATE_MULTIPLIER", "D0_COUNTERFACTUALS", "HIF_ALPHA_GRID", "HIF_R_GRID",
        "HIF_MAX_SCANS", "BC0_LEARNING_RATE", "BC0_EPOCHS", "BC0_SAVE_EVAL_STEPS",
        "VALIDATION_ROWS", "ROUND_LEARNING_RATE", "ROUND_SAVE_EVAL_STEPS", "TRAIN_MAX_LENGTH",
        "COLLECTION_BETA", "COLLECTION_MAX_STEPS", "EVAL_MAX_STEPS", "D1_CAP", "D1_SHARE",
        "PREVIOUS_PIPE", "SUITE_TRAINING_THRESHOLD", "SUITE_DEVELOPMENT_THRESHOLD",
        "SUITE_DEVELOPMENT_RANK_ALLOWANCE", "MEASUREMENT_ERROR_MIN_SIGMA",
        "TOPOLOGY_EFFECTS", "SYSTEM", "MEASUREMENT_CORPUS", "BALANCED_ARTIFACT_DIR",
        "ADMISSION_MODE", "NORMALIZED_RESIDUAL_THRESHOLD", "CORPUS_COUNTS", "CORPUS_NUM_SCANS",
        "FROZEN_STUDENT_ADAPTER", "ZEROSHOT_EVAL_NAME", "EVIDENCE_PROFILE", "HIF_SIGNATURE_MODE",
        "PMU_PHASOR_SIGMA", "TRIAGE_CLASSIFIER", "CORPUS_LOAD_SCALE_MIN", "CORPUS_LOAD_SCALE_MAX", "PROGRESS_BASIS",
        "UNVERIFIED_EXPERT_LABEL_POLICY", "ROUND_AGGREGATE", "ROUND_INIT",
    ):
        assert name in declared, name
    assert 'export PSSE_HIF_WORKERS="${SLURM_CPUS_PER_TASK:-8}"' in text
    assert "export PSSE_LOCAL_DIAGNOSTIC_BUILD=1" in text
    assert '--training-scenarios "$TRAINING_SUITE"' in text
    assert '--development-scenarios "$DEVELOPMENT_SUITE"' in text
    assert "--hif-search-profile research" in text
    # The recorded research_profile names the HIF corpora the suites were
    # drawn from: the non-empty HIF_CORPUS_* files of the configured system
    # (2026-10-11: any system with a three-phase route, not IEEE 14 only).
    assert 'for corpus in "${HIF_CORPUS_TRAIN:-}" "${HIF_CORPUS_VALID:-}" "${HIF_CORPUS_TRAIN_EXTRA:-}" "${HIF_CORPUS_VALID_EXTRA:-}"; do' in text
    assert '--hif-sample-paths "${HIF_CORPUS_FILES[@]}"' in text
    assert text.index("waveform_corpus_files()") < text.index('--hif-sample-paths "${HIF_CORPUS_FILES[@]}"')
    assert '--progress-basis "$PROGRESS_BASIS"' in text and "PROGRESS_BASIS=${PROGRESS_BASIS:-chi_square_statistic}" in text
    assert '--triage-classifier "$TRIAGE_CLASSIFIER"' in text
    assert 'UNVERIFIED_EXPERT_LABEL_POLICY=${UNVERIFIED_EXPERT_LABEL_POLICY:-abort}' in text
    assert '== stop ]]; then COLLECTION_ARGS+=(--stop-episode-on-unverified-expert-label); fi' in text


def test_plans_cover_every_family_and_respect_corpus_capacity() -> None:
    text = (CELL / "pipeline.env").read_text(encoding="utf-8")
    def plan(name: str) -> dict[str, int]:
        match = re.search(rf"^{name}='(\{{.*\}})'$", text, flags=re.MULTILINE)
        assert match, name
        return json.loads(match.group(1))
    d0 = plan("D0_PLAN")
    round_plan = plan("ROUND_TRAIN_PLAN")
    development = plan("DEVELOPMENT_PLAN")
    rounds = int(re.search(r"^DAGGER_ROUNDS=(\d+)$", text, flags=re.MULTILINE).group(1))
    families = {
        "no_error", "measurement", "multi_measurement", "parameter", "topology", "harmonic",
        "hif", "measurement+parameter", "measurement+topology", "measurement+hif",
        "three_phase_unbalance", "telemetry_no_disturbance",
    }
    assert set(d0) == set(round_plan) == set(development) == families
    total = {f: d0[f] + rounds * round_plan[f] + development[f] for f in families}
    # 420 raw HIF windows (84 + 252 train, 21 + 63 validation of the physical-ohm
    # corpora) serve hif and measurement+hif separately; only about 30 percent
    # pass discovered-mode admission at margin 1.25 (131 in the 2026-09-23opf
    # OPF-dispatched corpora, the same admitted windows as 20260923b), so the
    # plans are bounded by the admitted pool, not the raw count. 440 unbalance
    # windows serve unbalance and the balanced control (162 admitted at margin
    # 1.25).
    assert total["hif"] <= 131 and total["measurement+hif"] <= 131
    # The 2026-09-23opf unbalance corpus has 440 windows, of which 162 pass
    # discovered-mode admission at margin 1.25; the balanced control draws from
    # the same windows, so both plans are bounded by the admitted pool.
    assert total["three_phase_unbalance"] <= 162 and total["telemetry_no_disturbance"] <= 162
    assert 300 <= sum(total.values()) <= 1000


def test_hif_corpora_are_the_physical_ohm_regeneration() -> None:
    text = (CELL / "pipeline.env").read_text(encoding="utf-8")
    assert re.search(
        r"^HIF_CORPUS_TRAIN=\$SRC/artifacts/measurements/"
        r"hif_physical69_main_train_detectable_27x10_20260923opf/samples\.jsonl$",
        text,
        flags=re.MULTILINE,
    )
    assert re.search(
        r"^HIF_CORPUS_VALID=\$SRC/artifacts/measurements/"
        r"hif_physical69_main_valid_detectable_8x10_20260923opf/samples\.jsonl$",
        text,
        flags=re.MULTILINE,
    )
    # The comment states the ohm bands, the voltage profile, the shunt
    # convention, the OPF dispatch and the PMU precision, and the resulting
    # capacity (the 20260923opf subsets admit the same 131 windows as 20260923b).
    assert "100-200" in text and "200-500" in text and "500-1000 ohm" in text
    assert "ieee14_nominal_69_13p8_18kv_v1" in text
    assert "shunt_convention ybus" in text
    assert "131 HIF" in text and "27 + 8 + 77 + 19" in text
    assert "docs/opf_operating_points_20260923.md" in text
    assert "PMU phasor sigma 1e-4" in text
    assert re.search(
        r"^HIF_CORPUS_TRAIN_EXTRA=\$SRC/artifacts/measurements/"
        r"hif_physical69_main_train_extra_detectable_77x10_20260923opf/samples\.jsonl$",
        text,
        flags=re.MULTILINE,
    )
    assert re.search(
        r"^HIF_CORPUS_VALID_EXTRA=\$SRC/artifacts/measurements/"
        r"hif_physical69_main_valid_extra_detectable_19x10_20260923opf/samples\.jsonl$",
        text,
        flags=re.MULTILINE,
    )
    assert '"${HIF_CORPUS_TRAIN_EXTRA:-}" "${HIF_CORPUS_VALID_EXTRA:-}"' in text  # waveform_corpus_files lists them
    # The unbalance corpus is the 2026-09-23opf OPF-dispatched regeneration of the
    # 2026-09-21 WLS-convention corpus (162 admitted windows) and the collection
    # stage names it so the recorded profile states which rows were used.
    assert re.search(
        r"^IMBALANCE_CORPUS=\$SRC/artifacts/measurements/"
        r"out_measurements_imbalance_currents_ybus_detectable_162_20260923opf/samples\.jsonl$",
        text,
        flags=re.MULTILINE,
    )
    assert "162 unbalance" in text
    assert 'COLLECTION_ARGS+=(--imbalance-sample-path "$IMBALANCE_CORPUS")' in text
    assert "out_measurements_imbalance_currents_20260903" not in text
    assert "docs/ieee14_hif_legacy_reconfiguration_20260919.md" in text
    # No stage input names a superseded generation.
    assert "_20260923b/samples.jsonl" not in text and "_20260923/samples.jsonl" not in text
    assert "_20260921/samples.jsonl" not in text
    # The legacy system-pu corpora are no longer referenced by the cell.
    assert "hif_multiscan_currents_train_85x10_20260903" not in text
    assert "hif_multiscan_currents_17x10_20260903" not in text


def test_build_suite_accepts_and_forwards_the_corpus_flags() -> None:
    build_suite = _load("build_suite.py")
    parser = build_suite.build_parser()
    required = [
        "--source-root", "src", "--d0-raw", "d0.jsonl", "--round-train-plan", "{}",
        "--development-plan", "{}", "--seed", "1", "--output-dir", "out",
    ]
    args = parser.parse_args(required)
    assert args.hif_sample_paths is None and args.imbalance_sample_path is None
    args = parser.parse_args(
        required + ["--hif-sample-paths", "a.jsonl", "b.jsonl", "--imbalance-sample-path", "c.jsonl"]
    )
    assert args.hif_sample_paths == [Path("a.jsonl"), Path("b.jsonl")]
    assert args.imbalance_sample_path == Path("c.jsonl")
    # Stage 0 hands the suite builder the same system arguments as the aggregate (2026-10-11).
    assert parser.parse_args(required + ["--progress-basis", "chi_square_excess"]).progress_basis == "chi_square_excess"
    assert args.progress_basis is None
    source = (CELL / "build_suite.py").read_text(encoding="utf-8")
    call = source[source.index("research.resolve_scenario_sources("):]
    call = call[: call.index(")")]
    assert "hif_sample_paths=args.hif_sample_paths" in call
    assert "imbalance_sample_path=args.imbalance_sample_path" in call


def test_development_stratum_reads_the_recorded_ranking() -> None:
    build_suite = _load("build_suite.py")
    def row(ratio, rank, singleton=False):
        return {"parameter_ranking": {
            "parameter_ranking_dominance_ratio": ratio,
            "true_line_rank": rank,
            "parameter_ranking_singleton": singleton,
        }}
    classify = lambda r, fam="parameter": build_suite.parameter_ranking_stratum(
        r, family=fam, dominance_threshold=1.2
    )["stratum"]
    assert classify(row(1.5, 1)) == "dominant"
    assert classify(row(None, 1, singleton=True)) == "dominant"
    assert classify(row(1.1, 1)) == "ambiguous"
    assert classify(row(1.5, 2)) == "misranked"
    assert classify(row(1.1, 2)) == "misranked"
    assert classify(row(1.5, 1), fam="topology") == "not_applicable"
    assert classify({"audit": row(1.1, 1)}) == "ambiguous"


def test_summary_tables_split_by_stratum_and_carry_the_expert() -> None:
    summarize = _load("summarize.py")
    development = [
        {"grouping": {"scenario_family": "parameter", "physical_root_fingerprint": "a"}, "audit": {"parameter_ranking": {"stratum": "dominant"}}},
        # The recorded fields win over the stored label (a suite built before
        # the misranked stratum existed labelled this root ambiguous).
        {"grouping": {"scenario_family": "parameter", "physical_root_fingerprint": "b"}, "audit": {"parameter_ranking": {"stratum": "ambiguous", "generation": {"parameter_ranking_dominance_ratio": 1.1, "true_line_rank": 1}}}},
        {"grouping": {"scenario_family": "parameter", "physical_root_fingerprint": "c"}, "audit": {"parameter_ranking": {"stratum": "ambiguous", "generation": {"parameter_ranking_dominance_ratio": 1.3, "true_line_rank": 2}}}},
    ]
    families = summarize.family_by_root(development)
    strata = summarize.stratum_by_root(development)
    assert strata == {"a": "dominant", "b": "ambiguous", "c": "misranked"}
    def episode(root, ok, basis=None, wls_calls=2, tool_counts=None):
        # Evaluator episodes carry the success flag at top level as well as
        # inside the audit assessment; it must be counted exactly once.
        return {"physical_root": root, "steps": 3, "truth_audited_task_success": ok, "wls_calls": wls_calls,
                "tool_counts": tool_counts or {}, "audit": {"truth_audited_task_assessment": {"eligible": ok, "basis": basis}}}
    block = summarize.per_family_outcomes(
        {"episodes": [episode("a", True, "counterfactual_resolution"),
                      episode("b", True, "bounded_localization_handoff", wls_calls=3,
                              tool_counts={"compare_parameter_candidates": 1, "run_wls": 3})]},
        families,
        strata,
    )
    # Solver cost per episode: WLS calls and bounded comparisons, as histograms.
    assert block["outcomes"]["parameter"]["wls_calls"] == {"2": 1, "3": 1}
    assert block["outcomes"]["parameter"]["tool_counts.compare_parameter_candidates"] == {"0": 1, "1": 1}
    assert block["success_by_stratum"]["parameter"] == {
        "ambiguous": {"successes": 1, "episodes": 1},
        "dominant": {"successes": 1, "episodes": 1},
    }
    assert block["success_basis"]["parameter"] == {"bounded_localization_handoff": 1, "counterfactual_resolution": 1}
    assert summarize.stratum_table(block) == {
        "ambiguous": {"successes": 1, "episodes": 1},
        "dominant": {"successes": 1, "episodes": 1},
    }
    assert summarize.success_table(block) == {"parameter": {"successes": 2, "episodes": 2}}


def test_suite_exclusion_checks_d0_the_rounds_and_earlier_suites(tmp_path: Path) -> None:
    check = _load("check_suite_exclusion.py")
    def root(name: str) -> dict:
        return {"grouping": {"physical_root_fingerprint": name}}
    (tmp_path / "d0.jsonl").write_text("".join(json.dumps(root(n)) + "\n" for n in ("d1", "d2")), encoding="utf-8")
    suite = tmp_path / "suite"
    suite.mkdir()
    (suite / "manifest.json").write_text(json.dumps({"files": {}}), encoding="utf-8")
    (suite / "r1_training.json").write_text(json.dumps([root("t1"), root("t2")]), encoding="utf-8")
    (suite / "r2_training.json").write_text(json.dumps([root("t3")]), encoding="utf-8")
    (suite / "development.json").write_text(json.dumps([root("v1"), root("v2")]), encoding="utf-8")
    (tmp_path / "old_development.json").write_text(json.dumps([root("o1"), root("o2")]), encoding="utf-8")
    base = ["--d0-raw", str(tmp_path / "d0.jsonl"), "--suite-dir", str(suite)]
    assert check.main(base + ["--reference-suite", str(tmp_path / "old_development.json"),
                              "--reference-suite", str(tmp_path / "d0.jsonl")]) == 0
    # A development root drawn before fails the fresh-draw check.
    (tmp_path / "old_development.json").write_text(json.dumps([root("v2")]), encoding="utf-8")
    assert check.main(base + ["--reference-suite", str(tmp_path / "old_development.json")]) == 1
    # So does a root shared by a training round and the development set.
    (suite / "r2_training.json").write_text(json.dumps([root("v1")]), encoding="utf-8")
    assert check.main(base) == 1


def test_policy_table_reads_both_rounds_and_the_solver_cost() -> None:
    policy_table = _load("policy_table.py")
    def outcomes(successes: int, *, invalid: int = 0, wls: dict | None = None) -> dict:
        block = {
            "truth_audited_task_success": {"True": successes, "False": 2 - successes},
            "invalid_action_count": {"0": 2 - invalid, "3": invalid} if invalid else {"0": 2},
            "false_commit_count": {"0": 2},
            "loop_detected": {"False": 2},
            "steps": {"7": 1, "9": 1},
            "tool_counts.compare_parameter_candidates": {"0": 1, "1": 1},
        }
        if wls is not None:
            block["wls_calls"] = wls
        return {"parameter": block}
    r1 = {"per_family": {
        "expert": {"outcomes": outcomes(2, wls={"3": 2})},
        "bc0": {"outcomes": outcomes(1, invalid=1, wls={"3": 1, "5": 1})},
        "r1": {"outcomes": outcomes(2, wls={"3": 2})},
    }}
    r2 = {"per_family": {"r1": {"outcomes": outcomes(2, wls={"3": 2})}, "r2": {"outcomes": outcomes(2)}}}
    result = policy_table.table(r1, r2, ["parameter", "topology"])
    assert result["policies"]["bc0"]["successes"] == 1 and result["policies"]["bc0"]["invalid_action_episodes"] == 1
    assert result["policies"]["bc0"]["mean_wls_calls"] == 4.0 and result["policies"]["expert"]["mean_actions"] == 8.0
    assert result["policies"]["r2"]["mean_wls_calls"] is None  # a summary without the field
    assert result["policies"]["expert"]["comparisons"] == 1 and result["r1_consistent_across_rounds"]
    text = policy_table.markdown(result)
    assert "| Line R/X error | 2 | 2 | 1 | 2 | 2 |" in text and "| Mean WLS calls per episode | | 3.00 | 4.00 | 3.00 | n/a |" in text


def test_split_rounds_deals_each_family_across_rounds() -> None:
    build_suite = _load("build_suite.py")
    rows = [
        {"grouping": {"scenario_family": "hif", "physical_root_fingerprint": f"h{i}"}} for i in range(5)
    ] + [
        {"grouping": {"scenario_family": "harmonic", "physical_root_fingerprint": f"a{i}"}} for i in range(4)
    ]
    family = lambda row: row["grouping"]["scenario_family"]
    root = lambda row: row["grouping"]["physical_root_fingerprint"]
    dealt = build_suite.split_rounds(rows, rounds=2, family_of=family, root_of=root)
    assert [len(bucket) for bucket in dealt] == [5, 4]
    assert sorted(root(r) for bucket in dealt for r in bucket) == sorted(root(r) for r in rows)
    first = {family(r) for r in dealt[0]}
    second = {family(r) for r in dealt[1]}
    assert first == second == {"hif", "harmonic"}
    assert not {root(r) for r in dealt[0]} & {root(r) for r in dealt[1]}


def test_summary_joins_rounds_and_checks_adapter_consistency(tmp_path: Path) -> None:
    summarize = _load("summarize.py")
    def episode(root: str, ok: bool) -> dict:
        return {
            "physical_root": root,
            "terminal": True,
            "terminal_outcome": "resolved",
            "final_physical_success": ok,
            "steps": 3,
            "false_commit_count": 0,
            "invalid_action_count": 0,
            "loop_detected": False,
            "audit": {"truth_audited_task_assessment": {"truth_audited_task_success": ok}},
        }
    development = [
        {"grouping": {"scenario_family": "hif", "physical_root_fingerprint": "r1"}},
        {"grouping": {"scenario_family": "harmonic", "physical_root_fingerprint": "r2"}},
    ]
    def write_round(name: str, student: list[bool], candidate: list[bool]) -> None:
        collection = tmp_path / name / "collection"
        (collection / "evaluation").mkdir(parents=True, exist_ok=True)
        (collection / "development_scenarios.json").write_text(json.dumps(development), encoding="utf-8")
        (collection / "research_run_report.json").write_text(json.dumps({"research_profile": {}, "collection_metrics": {}, "mixture": {}}), encoding="utf-8")
        (collection / "evaluation" / "comparison.json").write_text(json.dumps({"paired_physical_roots": ["r1", "r2"], "bc0_overall": {}, "r1_overall": {}, "r1_minus_bc0": {}}), encoding="utf-8")
        for label, flags in (("bc0", student), ("r1", candidate)):
            (collection / "evaluation" / f"{label}_eval.json").write_text(
                json.dumps({"episodes": [episode("r1", flags[0]), episode("r2", flags[1])]}), encoding="utf-8"
            )
        summary = summarize.round_summary(tmp_path / name, name)
        (tmp_path / name / "round_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    write_round("r1", [False, False], [True, False])
    write_round("r2", [True, False], [True, True])
    joined = summarize.pipeline_summary(tmp_path)
    assert joined["overall"]["bc0"]["successes"] == 0
    assert joined["overall"]["r1"]["successes"] == 1
    assert joined["overall"]["r2"]["successes"] == 2
    assert joined["adapter_consistency_across_rounds"] == "consistent"
    write_round("r2", [False, False], [True, True])
    joined = summarize.pipeline_summary(tmp_path)
    assert "r1" in joined["adapter_consistency_across_rounds"]


def test_stage_zero_passes_the_system_selection_to_both_draws() -> None:
    text = (CELL / "stage_d0.sbatch").read_text(encoding="utf-8")
    assert 'system_args=(--system "$SYSTEM" --admission-mode "$ADMISSION_MODE"' in text
    assert text.count('"${system_args[@]}"') == 2
    assert '--progress-basis "$PROGRESS_BASIS")' in text
    assert 'label_args+=(--stop-episode-on-unverified-expert-label)' in text and '"${label_args[@]}"' in text
    # The waveform corpora of the configured system (pipeline.env's HIF_CORPUS_FILES), one --hif-corpus each.
    assert 'for corpus in "${HIF_CORPUS_FILES[@]}"; do waveform_args+=(--hif-corpus "$corpus"); done' in text
    assert 'waveform_args+=(--imbalance-corpus "$IMBALANCE_CORPUS")' in text
    # The suite builder reads the same HIF corpora as the aggregate.
    assert 'suite_waveform_args=(--hif-sample-paths "${HIF_CORPUS_FILES[@]}")' in text
    assert 'suite_waveform_args+=(--imbalance-sample-path "$IMBALANCE_CORPUS")' in text
    # The balanced corpus of a non-IEEE 14 system takes the configured load-scale range.
    assert '--load-scale-min "$CORPUS_LOAD_SCALE_MIN" --load-scale-max "$CORPUS_LOAD_SCALE_MAX"' in text
    build_suite_call = text[text.index('"$PY" "$PIPE/build_suite.py"'):]
    build_suite_call = build_suite_call[: build_suite_call.index("suite_console.json")]
    assert '"${suite_waveform_args[@]}"' in build_suite_call
    env = (CELL / "pipeline.env").read_text(encoding="utf-8")
    assert re.search(r"^SYSTEM=case14$", env, flags=re.MULTILINE)
    assert re.search(r"^ADMISSION_MODE=recoverable$", env, flags=re.MULTILINE)
    assert re.search(r"^NORMALIZED_RESIDUAL_THRESHOLD=4.0$", env, flags=re.MULTILINE)
    assert '--normalized-residual-threshold "$NORMALIZED_RESIDUAL_THRESHOLD"' in env
    assert '--normalized-residual-threshold "$NORMALIZED_RESIDUAL_THRESHOLD"' in text


def test_zero_shot_stage_and_frozen_student_wiring() -> None:
    d0 = (CELL / "stage_d0.sbatch").read_text(encoding="utf-8")
    assert "scripts/build_balanced_corpus.py" in d0
    assert 'if [[ "$SYSTEM" != case14 && -z "$MEASUREMENT_CORPUS" ]]; then' in d0
    assert 'ln -sfn "$FROZEN_STUDENT_ADAPTER" "$BC0_DIR/lora"' in d0
    assert "research_full_pipeline_bc0_reused_v1" in d0
    zs = (CELL / "stage_zeroshot.sbatch").read_text(encoding="utf-8")
    assert "--eval-student-only --eval-expert --eval-output-name" in zs
    assert 'source "$PIPE/pipeline.env"' in zs and "collection_args" in zs
    assert "summarize.py" in zs and "--evaluation-name" in zs
    submit = (CELL / "submit_pipeline.sh").read_text(encoding="utf-8")
    assert 'read -ra ORDER <<< "${CHAIN:-d0 bc0 r1c r1t r1e r2c r2t r2e}"' in submit
    assert "stage_zeroshot.sbatch" in submit
    assert "stage_zeroshot.sbatch" in (CELL / "deploy_remote.sh").read_text(encoding="utf-8")
    overrides = (CELL / "overrides" / "ieee57_transfer_20260914.env").read_text(encoding="utf-8")
    assert "\r" not in overrides
    assert re.search(r"^SYSTEM=case57$", overrides, flags=re.MULTILINE)
    for name in ("D0_PLAN", "ROUND_TRAIN_PLAN", "DEVELOPMENT_PLAN"):
        match = re.search(rf"^{name}='(\{{.*\}})'$", overrides, flags=re.MULTILINE)
        assert match, name
        assert set(json.loads(match.group(1))) == {
            "no_error", "measurement", "multi_measurement", "parameter", "measurement+parameter",
        }
    assert re.search(r"^FROZEN_STUDENT_ADAPTER=/scratch/", overrides, flags=re.MULTILINE)
    ieee118 = (CELL / "overrides" / "ieee118_balanced_20260925.env").read_text(encoding="utf-8")
    assert "\r" not in ieee118
    assert re.search(r"^SYSTEM=case118$", ieee118, flags=re.MULTILINE)
    balanced = {"no_error", "measurement", "multi_measurement", "parameter", "measurement+parameter"}
    for name in ("D0_PLAN", "ROUND_TRAIN_PLAN", "DEVELOPMENT_PLAN"):
        match = re.search(rf"^{name}='(\{{.*\}})'$", ieee118, flags=re.MULTILINE)
        assert match, name
        assert set(json.loads(match.group(1))) == balanced
    # 2026-10-11: the agentic transfer cell runs every IEEE 118 family but topology under the
    # classifier-gated profile with the joint gate and the excess progress basis.
    transfer = (CELL / "overrides" / "ieee118_transfer_20261011.env").read_text(encoding="utf-8")
    assert "\r" not in transfer
    assert re.search(r"^SYSTEM=case118$", transfer, flags=re.MULTILINE)
    assert re.search(r"^EVIDENCE_PROFILE=classifier_gated_diagnostics$", transfer, flags=re.MULTILINE)
    assert re.search(r"^PROGRESS_BASIS=chi_square_excess$", transfer, flags=re.MULTILINE)
    assert re.search(r"^UNVERIFIED_EXPERT_LABEL_POLICY=stop$", transfer, flags=re.MULTILINE)
    assert re.search(r"^TRIAGE_CLASSIFIER=\$SRC/psse_env/oracle/models/triage_gnn_joint_20261011$", transfer, flags=re.MULTILINE)
    assert re.search(r"^CORPUS_LOAD_SCALE_MAX=1\.25$", transfer, flags=re.MULTILINE)
    assert re.search(r"^FROZEN_STUDENT_ADAPTER=/scratch/", transfer, flags=re.MULTILINE)
    assert re.search(r"^HIF_CORPUS_TRAIN=\$PIPE/corpora/hif_samples\.jsonl$", transfer, flags=re.MULTILINE)
    assert re.search(r"^IMBALANCE_CORPUS=\$PIPE/corpora/unbalance_samples\.jsonl$", transfer, flags=re.MULTILINE)
    supported = set(resolve_system("case118").supported_families)
    assert supported == balanced | {"harmonic", "hif", "measurement+hif", "three_phase_unbalance"}
    for name in ("D0_PLAN", "ROUND_TRAIN_PLAN", "DEVELOPMENT_PLAN"):
        match = re.search(rf"^{name}='(\{{.*\}})'$", transfer, flags=re.MULTILINE)
        assert match, name
        assert set(json.loads(match.group(1))) == supported


def test_evidence_profile_default_is_suspicion_gated_and_every_guard_accepts_it() -> None:
    """The 2026-09-27 contract: suspicion-gated diagnostics by default (phasors
    only on a balanced HIF suspicion), discovered signatures on every strict
    profile, the PMU sigma of the study declared, every shell whitelist
    accepting both gated profiles, and the corpus paths naming the 20260923opf
    subsets.  The test never requires the corpora to exist."""
    env = (CELL / "pipeline.env").read_text(encoding="utf-8")
    assert "EVIDENCE_PROFILE=${EVIDENCE_PROFILE:-suspicion_gated_diagnostics}" in env
    assert "HIF_SIGNATURE_MODE=${HIF_SIGNATURE_MODE:-discovered}" in env
    assert re.search(r"^PMU_PHASOR_SIGMA=1e-4$", env, flags=re.MULTILINE)
    assert 'case "$EVIDENCE_PROFILE" in scada_only|wls_gated_diagnostics|suspicion_gated_diagnostics|classifier_gated_diagnostics|auxiliary_diagnostics)' in env
    assert '"$EVIDENCE_PROFILE" != auxiliary_diagnostics && "$HIF_SIGNATURE_MODE" != discovered' in env
    assert "docs/wls_gated_evidence_20260923.md" in env
    assert "TODO(20260923opf)" not in env and "TODO" not in env
    for name in ("HIF_CORPUS_TRAIN", "HIF_CORPUS_VALID", "HIF_CORPUS_TRAIN_EXTRA", "HIF_CORPUS_VALID_EXTRA",
                 "IMBALANCE_CORPUS"):
        assert re.search(rf"^{name}=\$SRC/artifacts/measurements/[a-z0-9_]+_20260923opf/samples\.jsonl$", env,
                         flags=re.MULTILINE), name
    deploy = (CELL / "deploy_remote.sh").read_text(encoding="utf-8")
    assert 'case "$EVIDENCE_PROFILE" in scada_only|wls_gated_diagnostics|suspicion_gated_diagnostics|classifier_gated_diagnostics|auxiliary_diagnostics)' in deploy
    prerequisites = (CELL / "prerequisites.sh").read_text(encoding="utf-8")
    assert ('"$EVIDENCE_PROFILE" == wls_gated_diagnostics || "$EVIDENCE_PROFILE" == suspicion_gated_diagnostics '
            '|| "$EVIDENCE_PROFILE" == classifier_gated_diagnostics') in prerequisites
    assert "triage_gnn_ieee14_20261004" in prerequisites
    for key in ("three_phase_sigma", "branch_current_sigma_pu"):
        assert key in prerequisites
    build_suite = _load("build_suite.py")
    args = build_suite.build_parser().parse_args([
        "--source-root", "src", "--d0-raw", "d0.jsonl", "--round-train-plan", "{}",
        "--development-plan", "{}", "--seed", "1", "--output-dir", "out",
    ])
    assert args.evidence_profile == "suspicion_gated_diagnostics"
    classifier_args = build_suite.build_parser().parse_args([
        "--source-root", "src", "--d0-raw", "d0.jsonl", "--round-train-plan", "{}",
        "--development-plan", "{}", "--seed", "1", "--output-dir", "out",
        "--evidence-profile", "classifier_gated_diagnostics",
    ])
    assert classifier_args.evidence_profile == "classifier_gated_diagnostics"
    overrides = (CELL / "overrides" / "classifier_gated_20261007.env").read_text(encoding="utf-8")
    assert "EVIDENCE_PROFILE=classifier_gated_diagnostics" in overrides and "EXPERT_VARIANT=baseline" in overrides
    assert args.hif_signature_mode == "discovered"
    # The stage scripts pass the profile through unchanged and record it.
    for name in ("stage_d0.sbatch", "stage_bc0.sbatch"):
        text = (CELL / name).read_text(encoding="utf-8")
        assert '"$EVIDENCE_PROFILE"' in text and '"evidence_profile": sys.argv[' in text
    readme = (CELL / "README.md").read_text(encoding="utf-8")
    assert "### 2026-09-23 cell: WLS-gated diagnostics" in readme
    assert "EVIDENCE_PROFILE=wls_gated_diagnostics" in readme and "20260923opf" in readme
    assert "pending corpora" not in readme
    for name in ("hif_physical69_main_train_detectable_27x10_20260923opf",
                 "hif_physical69_main_valid_detectable_8x10_20260923opf",
                 "hif_physical69_main_train_extra_detectable_77x10_20260923opf",
                 "hif_physical69_main_valid_extra_detectable_19x10_20260923opf",
                 "out_measurements_imbalance_currents_ybus_detectable_162_20260923opf"):
        assert name in readme, name


def test_rounds_aggregate_cumulatively_and_train_from_bc0_by_default() -> None:
    """The 2026-10-29 round recipe: a later round keeps every earlier round's
    safe rows in its mixture (the DAgger aggregate) and trains from the BC0
    adapter, with the 2026-09-07 recipe (pool sampling, continuing the
    student) still selectable per cell."""
    env = (CELL / "pipeline.env").read_text(encoding="utf-8")
    assert "ROUND_AGGREGATE=${ROUND_AGGREGATE:-cumulative}" in env
    assert "ROUND_INIT=${ROUND_INIT:-bc0}" in env
    assert 'case "$ROUND_AGGREGATE" in cumulative|pool)' in env
    assert 'case "$ROUND_INIT" in bc0|previous)' in env
    assert "docs/dagger_round2_regression_20261029.md" in env
    bindings = env[env.index("round_bindings() {"):env.index("collection_args() {")]
    assert "PRIOR_ONPOLICY_DIRS=()" in bindings
    r2 = bindings[bindings.index("    r2)"):bindings.index("    *) echo")]
    assert 'if [[ "$ROUND_AGGREGATE" == pool ]]; then' in r2
    assert "POOL_RAW=$R2_DIR/pool/raw.jsonl" in r2 and "POOL_TRAIN=$R2_DIR/pool/train.jsonl" in r2
    assert "POOL_RAW=$D0_DIR/aggregate.raw.jsonl" in r2 and "POOL_TRAIN=$D0_DIR/aggregate.train_view.jsonl" in r2
    assert 'PRIOR_ONPOLICY_DIRS=("$R1_DIR/collection")' in r2
    assert "STUDENT=$R1_DIR/training/lora" in r2
    assert 'if [[ "$ROUND_INIT" == bc0 ]]; then' in bindings
    assert "TRAIN_INIT=$BC0_DIR/lora" in bindings and "TRAIN_INIT=$STUDENT" in bindings
    collection = env[env.index("collection_args() {"):]
    assert 'for prior in "${PRIOR_ONPOLICY_DIRS[@]}"; do COLLECTION_ARGS+=(--prior-onpolicy-dir "$prior"); done' in collection
    assert collection.index('(( ${#PRIOR_ONPOLICY_DIRS[@]} ))') < collection.index("--prior-onpolicy-dir")
    collect = (CELL / "stage_collect.sbatch").read_text(encoding="utf-8")
    assert 'if [[ "$ROUND" == r2 && "$ROUND_AGGREGATE" == pool && ! -s "$ROUND_DIR/pool/manifest.json" ]]; then' in collect
    assert '"round_aggregate": sys.argv[4]' in collect and '"$ROUND_AGGREGATE" <<\'PY\'' in collect
    assert '"prior_onpolicy_rounds"' in collect and '"mixture_label_conflicts"' in collect
    assert "research_full_pipeline_collection_done_v2" in collect
    train = (CELL / "stage_train.sbatch").read_text(encoding="utf-8")
    assert '--initial-adapter "$TRAIN_INIT"' in train and '--initial-adapter "$STUDENT"' not in train
    assert '[[ -s "$TRAIN_INIT/adapter_model.safetensors" ]]' in train
    assert '"$STUDENT" "$TRAIN_INIT" "$ROUND_INIT" "$ROUND_AGGREGATE" <<\'PY\'' in train
    assert '"init_adapter_path": sys.argv[7]' in train and '"round_init": sys.argv[8]' in train
    assert '"round_aggregate": sys.argv[9]' in train
    assert "research_full_pipeline_training_done_v2" in train
    # The collector accepts the repeated flag the cell passes.
    import scripts.run_dagger_research as research
    args = research.parser().parse_args([
        "--d0-raw", "raw", "--d0-train", "train", "--adapter-path", "adapter", "--output-dir", "out",
        "--prior-onpolicy-dir", "/cell/out/r1/collection", "--prior-onpolicy-dir", "/cell/out/r2/collection",
    ])
    assert args.prior_onpolicy_dir == [Path("/cell/out/r1/collection"), Path("/cell/out/r2/collection")]
    assert research.parser().parse_args(
        ["--d0-raw", "raw", "--d0-train", "train", "--adapter-path", "adapter", "--output-dir", "out"]
    ).prior_onpolicy_dir == []
    readme = (CELL / "README.md").read_text(encoding="utf-8")
    assert "ROUND_AGGREGATE" in readme and "ROUND_INIT" in readme
    assert "docs/dagger_round2_regression_20261029.md" in readme
