"""Canonical aggregation/seeds and restart safety without loading model weights."""
from copy import deepcopy
import json

import pytest

from scripts import evaluate_revised_expert as runner


def scenario(name):
    return {"scenario_schema_version": 1,
        "execution": {"scenario_id": name, "case": "case14", "measurements": [0.]*122, "metadata": {}},
        "grouping": {"root_scenario_id": name, "physical_root_fingerprint": "root_"+name,
            "scenario_family": "no_error", "error_cardinality": 0, "case_id": "case14",
            "split": "development", "source_tier": "unit_fixture"},
        "audit": {"evaluation_intervention": {"intervention_schema_version": 1, "kind": "none"},
            "truth": {"truth_complete": True, "clean_case": "case14", "clean_measurements": [0.]*122,
            "true_measurement_errors": [], "true_parameter_errors": [], "true_topology_errors": []}}}


def episode(*, suite, scenario, scenario_index, episode_seed):
    sid = scenario["execution"]["scenario_id"]
    success = sid != "b"
    return runner.canonical.EpisodeEvaluation(episode_key=f"{suite}:{sid}:{scenario_index}",
        scenario_id=sid, suite=suite, family="no_error", cardinality=0, case="case14",
        split="development", source_tier="unit_fixture", physical_root="root_"+sid, seed=episode_seed,
        steps=2, policy_steps=2, terminal=True, terminal_outcome="resolved" if success else "operator_escalation",
        final_physical_correct=success, physical_correctness_known=True, final_physical_success=success,
        healthy_components_preserved=True, healthy_preservation_known=True,
        false_commit_count=0, false_rollback_count=0, false_finalization_count=0,
        partial_fix_count=0, retained_partial_fix_count=0, invalid_action_count=0,
        recovered_invalid_action_count=0, loop_detected=False, wls_calls=1,
        specialized_tool_calls=0, tool_counts={"run_wls": 1}, specialized_tool_counts={},
        tool_regret_total=0., tool_regret_samples=0, evaluation_intervention={"kind": "none"},
        truth_audited_task_success=success, truth_audited_task_success_evidence_known=True)


def inputs():
    suites = {runner.SUITE_NAME: [scenario("b"), scenario("a")]}
    manifest = runner.canonical.fingerprint_evaluation_suites(suites, seed=runner.EVALUATOR_SEED,
        required_suites=[runner.SUITE_NAME])
    return suites, manifest


def run(suites, manifest, path, binding="identical-science-inputs"):
    return runner.evaluate_checkpointed(suites, manifest, output_dir=path, run_binding=binding,
        env_factory=lambda: None, policy_factory=lambda: None)


def test_checkpointed_result_equals_canonical_monolithic_result_and_seeds(tmp_path, monkeypatch):
    calls = []

    def fake(self, **kwargs):
        calls.append(kwargs)
        return episode(**kwargs)

    monkeypatch.setattr(runner.canonical.ClosedLoopRolloutEvaluator, "_run_episode", fake)
    suites, manifest = inputs()
    direct = runner.canonical.evaluate_rollout_suites(suites, env_factory=lambda: None, policy_factory=lambda: None,
        max_steps=40, seed=runner.EVALUATOR_SEED, required_suites=[runner.SUITE_NAME],
        case_loader=runner.deterministic_case_loader).as_dict()
    actual, receipt = run(suites, manifest, tmp_path)
    assert actual == direct
    assert receipt == {"executed": 2, "resumed": 0}
    assert [e["scenario_id"] for e in actual["suite_metrics"]["episodes"]] == ["a", "b"]
    assert [e["seed"] for e in actual["suite_metrics"]["episodes"]] == [
        runner.canonical._episode_seed(20260912, "standard_success", name, 0) for name in ("a", "b")]
    assert actual["suite_metrics"]["overall"]["truth_audited_task_success_episodes"] == 1


def test_preemption_preserves_completed_root_and_resume_executes_only_missing(tmp_path, monkeypatch):
    calls = []

    def interrupted(self, **kwargs):
        calls.append(kwargs["scenario"]["execution"]["scenario_id"])
        if calls[-1] == "b":
            raise KeyboardInterrupt("simulated scheduler preemption")
        return episode(**kwargs)

    monkeypatch.setattr(runner.canonical.ClosedLoopRolloutEvaluator, "_run_episode", interrupted)
    suites, manifest = inputs()
    with pytest.raises(KeyboardInterrupt):
        run(suites, manifest, tmp_path)
    assert len(list((tmp_path/"episodes").glob("*.json"))) == 1
    assert not (tmp_path/"completed.json").exists()
    resumed = []

    def finish(self, **kwargs):
        resumed.append(kwargs["scenario"]["execution"]["scenario_id"])
        return episode(**kwargs)

    monkeypatch.setattr(runner.canonical.ClosedLoopRolloutEvaluator, "_run_episode", finish)
    result, counts = run(suites, manifest, tmp_path)
    assert resumed == ["b"] and counts == {"executed": 1, "resumed": 1}
    assert len(result["suite_metrics"]["episodes"]) == 2


@pytest.mark.parametrize("corruption", ["settings", "payload", "source_scenario", "physical_root"])
def test_resume_rejects_changed_or_corrupted_checkpoints(tmp_path, monkeypatch, corruption):
    monkeypatch.setattr(runner.canonical.ClosedLoopRolloutEvaluator, "_run_episode", lambda self, **kwargs: episode(**kwargs))
    suites, manifest = inputs()
    run(suites, manifest, tmp_path)
    binding = "different-config" if corruption == "settings" else "identical-science-inputs"
    if corruption in ("payload", "physical_root"):
        path = sorted((tmp_path/"episodes").glob("*.json"))[0]
        saved = json.loads(path.read_text())
        saved["episode"]["steps" if corruption == "payload" else "physical_root"] = 99 if corruption == "payload" else "wrong-root"
        if corruption == "physical_root":
            saved["episode_sha256"] = runner.digest(saved["episode"])
        runner.atomic_json(path, saved)
    elif corruption == "source_scenario":
        suites[runner.SUITE_NAME][0]["execution"]["measurements"][0] = .1
    with pytest.raises(ValueError):
        run(suites, manifest, tmp_path, binding)


def test_frozen_loader_checks_exact_bytes_count_unique_roots_and_preserves_alias(tmp_path):
    path = tmp_path/"frozen.json"
    rows = [scenario("a"), scenario("b")]
    runner.atomic_json(path, rows)
    suites, manifest = runner.load_frozen_scenarios(path, expected_sha256=runner.file_sha256(path), expected_roots=2)
    assert all(row["execution"]["case"] == "case14" for row in suites[runner.SUITE_NAME])
    assert len(manifest["episode_manifest"]) == 2
    with pytest.raises(ValueError, match="bytes"):
        runner.load_frozen_scenarios(path, expected_sha256="wrong", expected_roots=2)
    with pytest.raises(ValueError, match="exactly"):
        runner.load_frozen_scenarios(path, expected_sha256=runner.file_sha256(path), expected_roots=160)
    rows[1]["grouping"]["physical_root_fingerprint"] = "root_a"
    runner.atomic_json(path, rows)
    with pytest.raises(ValueError):
        runner.load_frozen_scenarios(path, expected_sha256=runner.file_sha256(path), expected_roots=2)


def test_r2_builder_uses_the_existing_research_factory_arguments_without_alias_override():
    calls = []
    options = {"adapter_path": "/existing/r2/lora", "base_model": "google/gemma-4-12B-it",
        "base_revision": "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7", "load_in_4bit": True,
        "trust_remote_code": False, "prompt_profile": "native", "architecture": "gemma4_unified"}

    def loader(*args, **kwargs):
        calls.append((args, kwargs))
        return object()

    lazy = runner.LazyPolicyFactory(runner.policy_builder("r2", options, model_loader=loader))
    assert not calls
    assert lazy(seed=123) is lazy(seed=456)
    assert len(calls) == 1
    assert calls[0][0] == (options["adapter_path"],)
    assert calls[0][1] == {k:options[k] for k in ("base_model","base_revision","load_in_4bit","trust_remote_code","prompt_profile","architecture")} | {"local_files_only": True}


@pytest.mark.parametrize("policy", ["bc0", "r1"])
def test_bc0_and_r1_adapters_use_the_same_research_factory_as_r2(policy):
    calls = []
    options = {"adapter_path": f"/existing/{policy}/lora", "base_model": "google/gemma-4-12B-it",
        "base_revision": "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7", "load_in_4bit": True,
        "trust_remote_code": False, "prompt_profile": "native", "architecture": "gemma4_unified"}

    def loader(*args, **kwargs):
        calls.append((args, kwargs))
        return object()

    lazy = runner.LazyPolicyFactory(runner.policy_builder(policy, options, model_loader=loader))
    lazy(seed=1)
    reference = []
    runner.LazyPolicyFactory(runner.policy_builder("r2", options, model_loader=lambda *a, **k: reference.append((a, k))))(seed=1)
    assert calls == reference
    assert runner.parser().parse_args(["--policy", policy, "--scenarios", "s.json", "--output-dir", "o"]).policy == policy
    with pytest.raises(ValueError, match="policy must be"):
        runner.policy_builder("r3", options, model_loader=loader)


def test_adapter_policies_require_an_adapter_and_the_expert_rejects_one():
    for argv in (["--policy", "bc0"], ["--policy", "r1"], ["--policy", "expert", "--adapter-path", "a"]):
        with pytest.raises(ValueError, match="--adapter-path"):
            runner.main([*argv, "--scenarios", "s.json", "--output-dir", "o"])


def test_fallback_counts_match_the_original_pipeline_report():
    payload = {"suite_metrics": {"overall": {"truth_audited_task_success_episodes": 150}}}
    expected = {"expert": 142, "bc0": 132, "r1": 140, "r2": 141}
    for policy, count in expected.items():
        comparison = runner.compare_baseline(payload, None, policy)
        assert comparison["baseline_task_successes"] == count
        assert comparison["task_success_delta"] == 150 - count


def test_baseline_comparison_requires_identical_roots_and_seeds(tmp_path, monkeypatch):
    monkeypatch.setattr(runner.canonical.ClosedLoopRolloutEvaluator, "_run_episode", lambda self, **kwargs: episode(**kwargs))
    suites, manifest = inputs()
    payload, _ = run(suites, manifest, tmp_path/"run")
    path = tmp_path/"baseline.json"
    runner.atomic_json(path, payload)
    comparison = runner.compare_baseline(payload, path, "expert")
    assert comparison["task_success_delta"] == 0 and comparison["changed_task_outcomes"] == []
    modified = deepcopy(payload)
    modified["suite_metrics"]["episodes"][0]["seed"] += 1
    runner.atomic_json(path, modified)
    with pytest.raises(ValueError, match="roots/seeds"):
        runner.compare_baseline(payload, path, "expert")


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_episode_cannot_be_checkpointed_or_marked_complete(tmp_path, monkeypatch, nonfinite):
    def invalid(self, **kwargs):
        result = episode(**kwargs)
        result.trace.append({"unavailable_numeric_value": nonfinite})
        return result

    monkeypatch.setattr(runner.canonical.ClosedLoopRolloutEvaluator, "_run_episode", invalid)
    suites, manifest = inputs()
    with pytest.raises(ValueError, match="JSON compliant"):
        run(suites, manifest, tmp_path)
    assert list((tmp_path/"episodes").iterdir()) == []
    assert not (tmp_path/"completed.json").exists()
