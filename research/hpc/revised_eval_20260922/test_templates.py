"""Resource and failure-gating contracts for the isolated evaluation templates."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


DIRECTORY = Path(__file__).resolve().parent


def _preflight():
    spec = importlib.util.spec_from_file_location("revised_eval_preflight", DIRECTORY / "preflight.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name,cpus,memory,hours", [
    ("preflight.sbatch", 4, "16G", "01:00:00"),
    ("expert.sbatch", 16, "48G", "06:00:00"),
    ("r2.sbatch", 16, "96G", "12:00:00"),
    ("bc0.sbatch", 16, "96G", "12:00:00"),
    ("r1.sbatch", 16, "96G", "12:00:00"),
])
def test_job_resources_and_cluster_routes(name, cpus, memory, hours):
    text = (DIRECTORY / name).read_text()
    header = [line for line in text.splitlines() if line.startswith("#SBATCH")]
    for value in ("--account=torch_pr_627_general", f"--cpus-per-task={cpus}",
                  f"--mem={memory}", f"--time={hours}"):
        assert f"#SBATCH {value}" in header
    if name in ("bc0.sbatch", "r1.sbatch", "r2.sbatch"):
        assert "#SBATCH --constraint=a100|h100|h200|rtx6000" in header
        assert "#SBATCH --gres=gpu:1" in header
        assert "#SBATCH --requeue" in header
        assert "#SBATCH --comment=preemption=yes;requeue=true" in header
        assert not any("--partition" in line for line in header)
    else:
        assert "#SBATCH --partition=cs" in header
        assert not any("--gres" in line for line in header)
    assert "trap revised_eval_cleanup EXIT" in text
    assert text.index("trap revised_eval_cleanup EXIT") < text.index("revised_eval_local_model\n")
    assert "\r" not in (DIRECTORY / name).read_bytes().decode()


def test_frozen_inputs_and_separate_outputs_do_not_change_generation_budgets():
    common = (DIRECTORY / "common.env").read_text()
    assert "OLD_PIPE=/scratch/yx3882/research_full_pipeline_20260921_physical" in common
    assert common.index('source "$OLD_PIPE/pipeline.env"') < common.index("SRC=$PIPE/source") < common.index("pipeline_environment\n")
    assert '"$OLD_PIPE"|"$OLD_PIPE"/*|/)' in common
    assert "FROZEN_SCENARIOS=$OLD_PIPE/out/r1/collection/development_scenarios.json" in common
    assert "FROZEN_R2_ADAPTER=$OLD_PIPE/out/r2/training/lora" in common
    assert "EXPERT_EVAL_DIR=$OUT/expert" in common and "R2_EVAL_DIR=$OUT/r2" in common
    assert "mktemp -d" in common and '"$resolved" == "$OPENDSS_LOCAL"' in common
    assert '"$resolved" == /dev/shm/psse_revised_eval_*' in common
    assert "OPENBLAS_NUM_THREADS=1" in common and "MKL_NUM_THREADS=1" in common
    assert "FROZEN_BC0_ADAPTER=$OLD_PIPE/out/bc0/lora" in common
    assert "FROZEN_R1_ADAPTER=$OLD_PIPE/out/r1/training/lora" in common
    assert "BC0_BASELINE=$OLD_PIPE/out/r1/collection/evaluation/bc0_eval.json" in common
    assert "R1_BASELINE=$OLD_PIPE/out/r1/collection/evaluation/r1_eval.json" in common
    for name, policy in (("expert.sbatch", "expert"), ("r2.sbatch", "r2"),
                         ("bc0.sbatch", "bc0"), ("r1.sbatch", "r1")):
        text = (DIRECTORY / name).read_text()
        assert f"--policy {policy}" in text and "--resume" in text and "--baseline-eval" in text
        assert "RESEARCH_MAX_INPUT_TOKENS=" not in text and "RESEARCH_MAX_NEW_TOKENS=" not in text
    assert "--adapter-path" not in (DIRECTORY / "expert.sbatch").read_text()


def _source(tmp_path, files):
    source = tmp_path / "source"
    for name in files:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixture\n")
    return source


def test_failed_actual_runtime_smoke_stops_before_fallback_or_tests(tmp_path, monkeypatch):
    module = _preflight()
    source = _source(tmp_path, module.TEST_FILES)
    calls = []
    def failed(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=3)
    monkeypatch.setattr(module.subprocess, "run", failed)
    receipt = tmp_path / "out/preflight.json"
    result = module.run_preflight(source, "actual-python", "test-python", receipt)
    assert not result["passed"] and result["test_python"] is None
    assert len(calls) == 1 and calls[0][0] == "actual-python"
    assert result["checks"][0]["name"] == "runtime_imports"
    assert json.loads(receipt.read_text())["passed"] is False


@pytest.mark.parametrize("runtime_has_pytest", [True, False])
def test_actual_runtime_imports_are_required_even_when_pytest_uses_fallback(tmp_path, monkeypatch, runtime_has_pytest):
    module = _preflight()
    source = _source(tmp_path, module.TEST_FILES)
    calls = []
    def completed(command, **kwargs):
        calls.append(command)
        missing = command == ["actual-python", "-c", "import pytest"] and not runtime_has_pytest
        return SimpleNamespace(returncode=1 if missing else 0)
    monkeypatch.setattr(module.subprocess, "run", completed)
    result = module.run_preflight(source, "actual-python", "test-python", tmp_path / "out/preflight.json")
    assert result["passed"]
    expected = "actual-python" if runtime_has_pytest else "test-python"
    assert result["test_python"] == calls[-1][0] == expected
    assert calls[0][0] == calls[1][0] == "actual-python"
    assert calls[1][-1] == "--help"
    compile(calls[-1][2], "<focused-test-command>", "exec")


def test_failing_focused_tests_emit_failed_receipt_and_nonzero_entrypoint(tmp_path, monkeypatch):
    module = _preflight()
    source = _source(tmp_path, module.TEST_FILES)
    def completed(command, **kwargs):
        return SimpleNamespace(returncode=1 if "pytest.main" in command[-1] else 0)
    monkeypatch.setattr(module.subprocess, "run", completed)
    receipt = tmp_path / "out/preflight.json"
    code = module.main(["--source", str(source), "--runtime-python", "actual-python",
                        "--test-python", "test-python", "--receipt", str(receipt)])
    assert code != 0
    result = json.loads(receipt.read_text())
    assert not result["passed"] and result["checks"][-1]["name"] == "focused_tests"
