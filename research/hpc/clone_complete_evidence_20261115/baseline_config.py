"""The baseline configs of the complete-evidence clone cell (2026-11-15): arm A's config in the open environment
with the estimator's output shown whole.

    python -m research.hpc.clone_complete_evidence_20261115.baseline_config           # write configs/baseline_*.json
    python -m research.hpc.clone_complete_evidence_20261115.baseline_config --check   # the files on disk are what this gives

The open-rules clone's cell measured S0 and its clone with the controller's five rules lifted
(``research/hpc/clone_open_rules_20261112/configs/baseline_clone_open.json``: arm A of the RL series with
``open_rules``).  The configs here are that config with one more environment option, ``complete_evidence``, under
which the policy reads every bus, line, meter and breaker of the estimator's output and none of the rule screens'
conclusions, and chooses the lines a comparison fits:

* ``baseline_clone_complete.json``: this cell's cloned adapter on the 300 development cases.  Read against the
  open-rules clone (163 of 300), S0 with the rules lifted (186), the rule expert (184 to 188) and the truth-built
  sequences (247).
* ``baseline_smoke_complete.json``: S0 on eight of the cases (the stage cuts the slice), before the pass has an
  adapter.  S0 was trained on another prompt: the smoke is read for prompt length, memory and refusals, not for
  what it solves.

``stage_baseline.sbatch`` runs the baseline phase of the RL driver from one of them and stops.  The clone's config
is also a complete arm config, should an RL run from the clone be wanted later (a new submission, on the user's word).
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from psse_env.actions import OPEN_RULES  # noqa: E402
from research.hpc.rl_improvement_20261110 import make_configs  # noqa: E402

CELL = Path(__file__).resolve().parent
WORK = "/scratch/yx3882/clone_complete_evidence_20261115"
CONFIG_DIR = CELL / "configs"
MATCHED_ARM = "A"
#: Every rule by name, not ``1``: a rule added to the option later would otherwise be lifted here without a word.
OPEN_RULES_OPTION = "+".join(OPEN_RULES)
SMOKE_SUITE = f"{WORK}/data/suites/ieee14_dev300_smoke8.json"
FILES = {"clone_complete": "baseline_clone_complete.json", "smoke_complete": "baseline_smoke_complete.json"}
CLONED_ADAPTER = f"{WORK}/out/clone/training/lora"
NAME = "clone_complete_evidence_20261115"
#: The open-rules clone's baseline config, which these differ from in the named fields.
OPEN_RULES_CLONE_CONFIG = "clone_open_rules_20261112/configs/baseline_clone_open.json"


def _series(series: dict[str, Any], arm: dict[str, Any], *, label: str, role: str, against: str, differs_in: list[str],
            planned_inputs: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": NAME,
        "arm": label,
        "role": role,
        "config_file": FILES[label],
        "workplan": series["workplan"],
        "matched": {"against": against, "differs_in": list(differs_in)},
        "expert_reference": arm["series"]["expert_reference"],
        "planned_inputs": dict(planned_inputs),
    }


def build() -> dict[str, dict[str, Any]]:
    """The two configs by file name, each loaded through the driver's own validation."""

    from psse_env.rl.train import RLConfig

    series = make_configs.load_series()
    arm = make_configs.arm_config(series, MATCHED_ARM)
    planned = dict(arm["series"]["planned_inputs"])

    clone = copy.deepcopy(arm)
    clone["environment_options"] = {**arm["environment_options"], "open_rules": OPEN_RULES_OPTION, "complete_evidence": 1}
    clone["initial_adapter"] = CLONED_ADAPTER
    clone["output_dir"] = f"{WORK}/out/baseline_clone_complete"
    clone["series"] = _series(series, arm, label="clone_complete", against=OPEN_RULES_CLONE_CONFIG,
                              differs_in=["environment_options", "initial_adapter"],
                              role="the adapter cloned on the complete-evidence selection, on the development cases with the "
                                   "controller's five rules lifted and the estimator's output shown whole",
                              planned_inputs=planned)

    smoke = copy.deepcopy(clone)
    smoke["initial_adapter"] = arm["initial_adapter"]
    smoke["development_suite"] = SMOKE_SUITE
    smoke["output_dir"] = f"{WORK}/out/baseline_smoke_complete"
    smoke["series"] = _series(series, arm, label="smoke_complete", against=FILES["clone_complete"],
                              differs_in=["development_suite", "initial_adapter"],
                              role="smoke: S0 on eight of the development cases in this environment, for prompt length, memory "
                                   "and refusals (S0 was trained on another prompt)",
                              planned_inputs={key: value for key, value in planned.items() if key != "development_suite_sha256"})

    configs = {FILES["clone_complete"]: clone, FILES["smoke_complete"]: smoke}
    for config in configs.values():
        RLConfig(**config).validate()
    return configs


def render(config: dict[str, Any]) -> str:
    return json.dumps(config, indent=2) + "\n"


def check() -> list[str]:
    problems = []
    for name, config in build().items():
        path = CONFIG_DIR / name
        if not path.is_file():
            problems.append(f"{name} is not written")
        elif path.read_bytes() != render(config).encode("utf-8"):
            problems.append(f"{name} is not what arm {MATCHED_ARM} of the series gives (run baseline_config.py)")
    return problems


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="write nothing; fail when a file on disk differs")
    args = parser.parse_args(argv)
    if args.check:
        problems = check()
        for problem in problems:
            print(problem, file=sys.stderr)
        return 1 if problems else 0
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    for name, config in build().items():
        # LF on every platform: the files go to the cluster as they are.
        (CONFIG_DIR / name).write_text(render(config), encoding="utf-8", newline="\n")
        print(CONFIG_DIR / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
