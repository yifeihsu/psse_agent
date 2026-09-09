from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

CELL = Path(__file__).resolve().parent / "hpc" / "full_pipeline_20260907"
GPU_STAGES = ("stage_bc0.sbatch", "stage_collect.sbatch", "stage_train.sbatch", "stage_eval.sbatch")
CPU_STAGES = ("stage_d0.sbatch",)
SHELL_FILES = ("prerequisites.sh", "submit_pipeline.sh", "status_pipeline.sh", "deploy_remote.sh")
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
    expected_cpus = 16 if name in ("stage_collect.sbatch", "stage_eval.sbatch") else 8
    assert f"#SBATCH --cpus-per-task={expected_cpus}" in header


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
        "HIF_CORPUS_TRAIN", "HIF_CORPUS_VALID", "IMBALANCE_CORPUS", "SEED_D0", "SEED_SUITE",
        "SEED_ROUND", "D0_PLAN", "ROUND_TRAIN_PLAN", "DEVELOPMENT_PLAN", "DAGGER_ROUNDS",
        "CANDIDATE_MULTIPLIER", "D0_COUNTERFACTUALS", "HIF_ALPHA_GRID", "HIF_R_GRID",
        "HIF_MAX_SCANS", "BC0_LEARNING_RATE", "BC0_EPOCHS", "BC0_SAVE_EVAL_STEPS",
        "VALIDATION_ROWS", "ROUND_LEARNING_RATE", "ROUND_SAVE_EVAL_STEPS", "TRAIN_MAX_LENGTH",
        "COLLECTION_BETA", "COLLECTION_MAX_STEPS", "EVAL_MAX_STEPS", "D1_CAP", "D1_SHARE",
        "PREVIOUS_PIPE", "SUITE_TRAINING_THRESHOLD", "SUITE_DEVELOPMENT_THRESHOLD",
        "SUITE_DEVELOPMENT_RANK_ALLOWANCE", "MEASUREMENT_ERROR_MIN_SIGMA",
    ):
        assert name in declared, name
    assert 'export PSSE_HIF_WORKERS="${SLURM_CPUS_PER_TASK:-8}"' in text
    assert "export PSSE_LOCAL_DIAGNOSTIC_BUILD=1" in text
    assert '--training-scenarios "$TRAINING_SUITE"' in text
    assert '--development-scenarios "$DEVELOPMENT_SUITE"' in text
    assert "--hif-search-profile research" in text


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
    # 102 HIF windows serve hif and measurement+hif separately; 220 unbalance
    # rows serve unbalance and the balanced control separately.
    assert total["hif"] <= 102 and total["measurement+hif"] <= 102
    assert total["three_phase_unbalance"] <= 220 and total["telemetry_no_disturbance"] <= 220
    assert 300 <= sum(total.values()) <= 1000


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
    def episode(root, ok, basis=None):
        # Evaluator episodes carry the success flag at top level as well as
        # inside the audit assessment; it must be counted exactly once.
        return {"physical_root": root, "steps": 3, "truth_audited_task_success": ok, "audit": {"truth_audited_task_assessment": {"eligible": ok, "basis": basis}}}
    block = summarize.per_family_outcomes(
        {"episodes": [episode("a", True, "counterfactual_resolution"), episode("b", True, "bounded_localization_handoff")]},
        families,
        strata,
    )
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
