"""CPU-only checks for the 2026-09-03 diagnostic research round scripts."""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

CELL = Path(__file__).resolve().parent / "hpc" / "diagnostic_round_20260903"
SBATCH_FILES = ("diag_collect.sbatch", "diag_train.sbatch", "diag_eval.sbatch")
SHELL_FILES = ("prerequisites.sh", "submit_diag.sh", "status_diag.sh", "deploy_remote.sh", "round.env")


def _load_summarize():
    spec = importlib.util.spec_from_file_location("diag_summarize", CELL / "summarize.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", SBATCH_FILES)
def test_sbatch_headers_follow_the_cluster_routing_rules(name: str) -> None:
    text = (CELL / name).read_text(encoding="utf-8")
    assert "\r" not in text, f"{name} must be LF-only"
    header = [line for line in text.splitlines() if line.startswith("#SBATCH")]
    assert "#SBATCH --account=torch_pr_627_general" in header
    assert "#SBATCH --constraint=a100|h100|h200|rtx6000" in header
    assert "#SBATCH --comment=preemption=yes;requeue=true" in header
    assert "#SBATCH --requeue" in header
    assert "#SBATCH --gres=gpu:1" in header
    assert not any(line.startswith("#SBATCH --partition") for line in header)
    assert "source \"$ROUND/round.env\"" in text
    assert "round_environment" in text


@pytest.mark.parametrize("name", SHELL_FILES + SBATCH_FILES)
def test_scripts_are_lf_only(name: str) -> None:
    assert "\r" not in (CELL / name).read_bytes().decode("utf-8")


def test_round_env_declares_every_setting_the_stages_use() -> None:
    text = (CELL / "round.env").read_text(encoding="utf-8")
    declared = set(re.findall(r"^([A-Z0-9_]+)=", text, flags=re.MULTILINE))
    for name in (
        "ROUND", "SRC", "OUT", "LOGS", "PY", "D0_RAW", "D0_TRAIN", "VALIDATION",
        "PROTECTED_D1", "WARM_START", "BC0_SUITE", "MODEL_CHOICE", "MODEL_ID",
        "MODEL_REVISION", "HF_HOME", "COLLECTION_DIR", "TRAIN_DIR", "CANDIDATE",
        "SEED", "PLAN_PRESET", "COLLECTION_BETA", "COLLECTION_MAX_STEPS",
        "D1_CAP", "D1_SHARE", "EVAL_MAX_STEPS",
    ):
        assert name in declared, name
    assert "--plan-preset \"$PLAN_PRESET\"" in text
    assert "--hif-search-profile auto" in text
    assert "RESEARCH_MAX_INPUT_TOKENS=32768" in text
    assert "PLAN_PRESET=diagnostic" in text
    assert "SEED=20260903" in text


def test_training_uses_the_proven_12b_layout() -> None:
    text = (CELL / "diag_train.sbatch").read_text(encoding="utf-8")
    assert "--batch-size 1" in text
    assert "--gradient-accumulation-steps 4" in text
    assert "--select-best-eval-loss" not in text
    assert "--initial-adapter \"$WARM_START\"" in text


def test_summary_tabulates_outcomes_per_family(tmp_path: Path) -> None:
    summarize = _load_summarize()
    collection = tmp_path / "collection"
    (collection / "evaluation").mkdir(parents=True)
    development = [
        {"grouping": {"physical_root_fingerprint": "root_hif", "scenario_family": "hif"}},
        {"grouping": {"physical_root_fingerprint": "root_unb", "scenario_family": "three_phase_unbalance"}},
    ]
    (collection / "development_scenarios.json").write_text(json.dumps(development), encoding="utf-8")
    (collection / "research_run_report.json").write_text(
        json.dumps({"research_profile": {"plan_preset": "diagnostic"}, "collection_metrics": {"label_yield": 0.5}, "mixture": {"d1_selected": 3}}),
        encoding="utf-8",
    )
    (collection / "evaluation" / "comparison.json").write_text(
        json.dumps({"paired_physical_roots": ["root_hif", "root_unb"], "bc0_overall": {"resolved_episodes": 0.5}, "r1_overall": {"resolved_episodes": 1.0}, "r1_minus_bc0": {"resolved_episodes": 0.5}}),
        encoding="utf-8",
    )
    for label, outcomes in (("bc0", ("resolved", "operator_escalation")), ("r1", ("resolved", "resolved"))):
        payload = {
            "suite_metrics": {"overall": {}},
            "episodes": [
                {
                    "physical_root": root,
                    "family": "unknown-until-joined",
                    "terminal": True,
                    "terminal_outcome": outcome,
                    "steps": 3,
                }
                for root, outcome in zip(("root_hif", "root_unb"), outcomes)
            ],
        }
        (collection / "evaluation" / f"{label}_eval.json").write_text(json.dumps(payload), encoding="utf-8")
    summary = summarize.build_summary(collection_dir=collection, training_done=None, prerequisites=None)
    assert summary["release_evidence"] is False
    assert summary["development_families"] == {"hif": 1, "three_phase_unbalance": 1}
    assert summary["paired_evaluation"]["paired_physical_roots"] == 2
    bc0 = summary["per_family"]["bc0"]
    assert bc0["episodes_per_family"] == {"hif": 1, "three_phase_unbalance": 1}
    assert bc0["outcomes"]["three_phase_unbalance"]["terminal_outcome"] == {"operator_escalation": 1}
    assert summary["per_family"]["r1"]["outcomes"]["three_phase_unbalance"]["terminal_outcome"] == {"resolved": 1}
    assert bc0["unmatched_episodes"] == 0
