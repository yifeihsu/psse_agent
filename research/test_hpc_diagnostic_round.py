"""CPU-only checks for the 2026-09-03 diagnostic research round scripts."""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

CELL = Path(__file__).resolve().parent / "hpc" / "diagnostic_round_20260903"
SBATCH_FILES = ("diag_collect.sbatch", "diag_train.sbatch", "diag_eval.sbatch")
SHELL_FILES = (
    "prerequisites.sh",
    "submit_diag.sh",
    "status_diag.sh",
    "deploy_remote.sh",
    "amend_train_chain.sh",
    "round.env",
)


def _load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, CELL / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_summarize():
    return _load_module("diag_summarize", "summarize.py")


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
    assert "--train-plan \"$TRAIN_PLAN\"" in text
    assert "--development-plan \"$DEVELOPMENT_PLAN\"" in text
    assert "source \"$ROUND/round.overrides.env\"" in text


def test_scale_overrides_keep_the_diagnostic_families() -> None:
    text = (CELL / "overrides" / "scale_20260906.env").read_text(encoding="utf-8")
    assert "\r" not in text
    plans = {}
    for name in ("TRAIN_PLAN", "DEVELOPMENT_PLAN"):
        match = re.search(rf"^{name}='(.*)'$", text, flags=re.MULTILINE)
        assert match, name
        plans[name] = json.loads(match.group(1))
    families = {"hif", "measurement+hif", "three_phase_unbalance", "harmonic", "telemetry_no_disturbance"}
    assert set(plans["TRAIN_PLAN"]) == families
    assert set(plans["DEVELOPMENT_PLAN"]) == families
    # Three times the diagnostic preset, and within the corpora: 102 HIF
    # windows, 220 unbalance rows (also feeding the control), 500 harmonic rows.
    assert plans["TRAIN_PLAN"] == {"hif": 36, "measurement+hif": 18, "three_phase_unbalance": 36, "harmonic": 36, "telemetry_no_disturbance": 18}
    assert plans["DEVELOPMENT_PLAN"] == {"hif": 18, "measurement+hif": 9, "three_phase_unbalance": 18, "harmonic": 18, "telemetry_no_disturbance": 9}
    assert plans["TRAIN_PLAN"]["hif"] + plans["DEVELOPMENT_PLAN"]["hif"] <= 102
    assert "D1_CAP=1000" in text


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
        # The evaluator nests per-suite episode lists under ``metrics``.
        payload = {
            "suite_metrics": {"overall": {}},
            "score": 0.0,
            "metrics": {
                "suites": {
                    "standard_success": {
                        "episodes": [
                            {
                                "physical_root": root,
                                "family": "unknown-until-joined",
                                "terminal": True,
                                "terminal_outcome": outcome,
                                "steps": 3,
                                "audit": {"diagnostics": {"diagnostic_truth_matched": root == "root_hif"}},
                            }
                            for root, outcome in zip(("root_hif", "root_unb"), outcomes)
                        ]
                    }
                }
            },
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
    assert bc0["outcomes"]["hif"]["audit.diagnostic_truth_matched"] == {"True": 1}
    assert bc0["outcomes"]["three_phase_unbalance"]["audit.diagnostic_truth_matched"] == {"False": 1}


def _chat_row(root: str, family: str, source: str) -> dict:
    return {
        "example_id": f"{source}_{root}",
        "physical_root_fingerprint": root,
        "metadata": {"scenario_family": family},
        "messages": [{"role": "user", "content": root}],
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_training_stage_trains_on_the_filtered_mixture() -> None:
    text = (CELL / "diag_train.sbatch").read_text(encoding="utf-8")
    assert "filter_mixture.py" in text
    assert "--train \"$FILTERED\"" in text
    assert "--expected-dropped \"$EXPECTED_STALE_D0_ROWS\"" in text
    env = (CELL / "round.env").read_text(encoding="utf-8")
    assert "STALE_D0_FAMILIES=\"hif,measurement+hif\"" in env
    assert "EXPECTED_STALE_D0_ROWS=91" in env


def test_filter_drops_stale_families_before_sampling_and_keeps_one_to_one(tmp_path: Path) -> None:
    from scripts.run_dagger_research import build_research_mixture

    filter_mixture = _load_module("diag_filter_mixture", "filter_mixture.py")
    d0 = [_chat_row(f"d0_param_{i}", "parameter", "d0") for i in range(6)]
    d0 += [_chat_row(f"d0_hif_{i}", "hif", "d0") for i in range(2)]
    d0 += [_chat_row("d0_mhif_0", "measurement+hif", "d0")]
    d1 = [_chat_row(f"d1_hif_{i}", "hif", "d1") for i in range(3)]
    d0_path, d1_path = tmp_path / "d0.jsonl", tmp_path / "d1.jsonl"
    _write_jsonl(d0_path, d0)
    _write_jsonl(d1_path, d1)
    output, report_path = tmp_path / "filtered.jsonl", tmp_path / "report.json"
    report = filter_mixture.build(
        d0_path=d0_path,
        d1_path=d1_path,
        stale_families=["hif", "measurement+hif"],
        expected_dropped=3,
        d1_share=0.5,
        d1_cap=200,
        seed=7,
        output=output,
        report_path=report_path,
        mixture_builder=build_research_mixture,
    )
    assert report["dropped_by_family"] == {"hif": 2, "measurement+hif": 1}
    assert report["d0_rows_after"] == 6
    assert report["d0_families_after"] == {"parameter": 6}
    assert report["mixture_sources"] == {"d0": 3, "d1": 3}
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 6
    d0_selected = [row for row in rows if row["research_mixture_source"] == "d0"]
    assert all(filter_mixture.row_family(row) == "parameter" for row in d0_selected)
    assert json.loads(report_path.read_text(encoding="utf-8"))["output_sha256"] == report["output_sha256"]
    with pytest.raises(ValueError, match="expected to drop 4"):
        filter_mixture.build(
            d0_path=d0_path,
            d1_path=d1_path,
            stale_families=["hif", "measurement+hif"],
            expected_dropped=4,
            d1_share=0.5,
            d1_cap=200,
            seed=7,
            output=tmp_path / "other.jsonl",
            report_path=tmp_path / "other.json",
            mixture_builder=build_research_mixture,
        )


def test_filter_fails_closed_on_a_row_without_a_family() -> None:
    filter_mixture = _load_module("diag_filter_mixture_2", "filter_mixture.py")
    with pytest.raises(ValueError, match="carries no scenario family"):
        filter_mixture.filter_rows([{"example_id": "x", "metadata": {}}], ["hif"])
    kept, dropped = filter_mixture.filter_rows(
        [{"scenario_family": "topology"}, {"grouping": {"scenario_family": "hif"}}], ["hif"]
    )
    assert len(kept) == 1 and dict(dropped) == {"hif": 1}
