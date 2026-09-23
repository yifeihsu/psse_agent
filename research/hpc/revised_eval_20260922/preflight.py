"""Import the real runtime, then run focused checks before either evaluation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess


TEST_FILES = (
    "psse_env/providers/test_hif_continuation.py",
    "psse_env/oracle/test_hif_meter_nonregression.py",
    "tests/test_hif_expert_continuation.py",
    "psse_env/dagger/test_acquisition_reference_audit.py",
    "psse_env/dagger/test_hif_policy_serialization.py",
    "psse_env/dagger/test_release_audit.py",
    "psse_env/oracle/test_normalized_residual_gate.py",
    "psse_env/oracle/test_diagnostics_routing.py",
    "psse_env/dagger/test_research_policy_factory.py",
    "tests/test_verify_hif_continuation.py",
    "tests/test_evaluate_revised_expert.py",
    "research/hpc/revised_eval_20260922/test_templates.py",
)
IMPORT_SMOKE = """
import opendssdirect
import json, os, sys, numpy, scipy, torch
from pathlib import Path
from psse_env.providers import hif_continuation
from psse_env.oracle import hif_continuation as controller
from psse_env.dagger import release_audit, release_factories
from psse_env.research_models import get_research_model_spec
from scripts import evaluate_revised_expert
assert Path(sys.executable).resolve() == Path(os.environ['PY']).resolve(), 'Wrong runtime interpreter'
assert Path(hif_continuation.__file__).resolve().is_relative_to(Path(os.environ['SRC']).resolve()), 'Wrong source checkout'
expected = {'RESEARCH_MAX_INPUT_TOKENS': 32768, 'RESEARCH_MAX_NEW_TOKENS': 256,
            'SEED_ROUND': 20260912, 'EPISODE_MAX_STEPS': 40,
            'HIF_ALPHA_GRID': 7, 'HIF_R_GRID': 9, 'HIF_MAX_SCANS': 10}
for name, value in expected.items():
    assert int(os.environ[name]) == value, f'Changed experiment setting: {name}'
assert float(os.environ['NORMALIZED_RESIDUAL_THRESHOLD']) == 4.0
model = get_research_model_spec(os.environ['MODEL_CHOICE'])
assert model.model_id == os.environ['MODEL_ID'] and model.revision == os.environ['MODEL_REVISION']
print(json.dumps({'passed': True, 'runtime_python': sys.executable, 'pid': os.getpid(),
    'model_id': model.model_id, 'model_revision': model.revision,
    'architecture': model.architecture, 'prompt_profile': model.prompt_profile,
    'settings': expected, 'normalized_residual_threshold': 4.0}))
"""


def run_preflight(source: Path, runtime_python: str, test_python: str, receipt: Path) -> dict:
    source, receipt = source.resolve(), receipt.resolve()
    receipt.parent.mkdir(parents=True, exist_ok=True)
    logs = receipt.parent / "preflight_logs"
    logs.mkdir(parents=True, exist_ok=True)
    result = {"contract": "revised_eval_preflight_v1", "passed": False,
        "runtime_python": runtime_python, "test_python": None, "source": str(source),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "checks": [], "tests": list(TEST_FILES)}

    def checked(name, command):
        path = logs / f"{name}.log"
        with path.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(command, cwd=source, stdout=handle, stderr=subprocess.STDOUT)
        result["checks"].append({"name": name, "command": command,
                                 "returncode": completed.returncode, "log": str(path)})
        if completed.returncode:
            raise RuntimeError(f"{name} failed with exit {completed.returncode}; see {path}")

    try:
        checked("runtime_imports", [runtime_python, "-c", IMPORT_SMOKE])
        checked("runner_cli", [runtime_python, "scripts/evaluate_revised_expert.py", "--help"])
        missing = [name for name in TEST_FILES if not (source / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing focused tests: {missing}")
        selected = runtime_python
        probe = subprocess.run([runtime_python, "-c", "import pytest"], cwd=source,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if probe.returncode:
            selected = test_python
            checked("fallback_pytest_import", [selected, "-c", "import pytest"])
        result["test_python"] = selected
        # Preload the native DSS library before pytest's numerical plugins.
        checked("focused_tests", [selected, "-c",
            "import opendssdirect, pytest; raise SystemExit(pytest.main([ '-q', '-p', 'no:cacheprovider', *"
            + repr(list(TEST_FILES)) + "]))"])
        result["passed"] = True
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        result["completed_utc"] = datetime.now(timezone.utc).isoformat()
        receipt.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--runtime-python", required=True)
    parser.add_argument("--test-python", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_preflight(args.source, args.runtime_python, args.test_python, args.receipt)
    print(json.dumps({"passed": result["passed"], "receipt": str(args.receipt), "error": result.get("error")}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
