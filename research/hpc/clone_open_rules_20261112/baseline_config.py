"""The baseline configs of the open-rules clone cell (2026-11-12): arm A's config in the open environment.

    python -m research.hpc.clone_open_rules_20261112.baseline_config           # write configs/baseline_*.json
    python -m research.hpc.clone_open_rules_20261112.baseline_config --check   # the files on disk are what this gives

Arm A of the RL series (``research/hpc/rl_improvement_20261110``) begins with a baseline: S0, greedy, bf16, 40
actions, bounded history, on the 300 development cases, with the controller's rules in place.  The configs here are
arm A's with one field changed each, so that every pair of baselines differs in the one thing it asks about:

* ``baseline_s0_open.json``: the controller's five rules lifted (``environment_options.open_rules`` names them
  all).  S0 in the open environment against S0 in arm A's.
* ``baseline_clone_open.json``: the open config with this cell's cloned adapter.  The clone against S0, both open.
* ``baseline_smoke_open.json``: the open config on eight of the cases (the stage cuts the slice), for the smoke.

``stage_baseline.sbatch`` runs the baseline phase of the RL driver from one of them and stops.  Each is also a
complete arm config, should an RL run in the open environment be wanted later (a new submission, on the user's word).
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
WORK = "/scratch/yx3882/clone_open_rules_20261112"
CONFIG_DIR = CELL / "configs"
MATCHED_ARM = "A"
#: Every rule by name, not ``1``: a rule added to the option later would otherwise be lifted here without a word.
OPEN_RULES_OPTION = "+".join(OPEN_RULES)
SMOKE_SUITE = f"{WORK}/data/suites/ieee14_dev300_smoke8.json"
FILES = {"s0_open": "baseline_s0_open.json", "clone_open": "baseline_clone_open.json", "smoke_open": "baseline_smoke_open.json"}
CLONED_ADAPTER = f"{WORK}/out/clone/training/lora"
NAME = "clone_open_rules_20261112"


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
    """The three configs by file name, each loaded through the driver's own validation."""

    from psse_env.rl.train import RLConfig

    series = make_configs.load_series()
    arm = make_configs.arm_config(series, MATCHED_ARM)
    planned = dict(arm["series"]["planned_inputs"])

    s0 = copy.deepcopy(arm)
    s0["environment_options"] = {**arm["environment_options"], "open_rules": OPEN_RULES_OPTION}
    s0["output_dir"] = f"{WORK}/out/baseline_s0_open"
    s0["series"] = _series(series, arm, label="s0_open", against=f"{series['name']} arm {MATCHED_ARM}", differs_in=["environment_options"],
                           role="S0 on the development cases with the controller's five rules lifted (open_rules), arm A's config otherwise",
                           planned_inputs=planned)

    clone = copy.deepcopy(s0)
    clone["initial_adapter"] = CLONED_ADAPTER
    clone["output_dir"] = f"{WORK}/out/baseline_clone_open"
    clone["series"] = _series(series, arm, label="clone_open", against=FILES["s0_open"], differs_in=["initial_adapter"],
                              role="the adapter cloned on the open-rules selection, on the development cases in the open environment",
                              planned_inputs=planned)

    smoke = copy.deepcopy(s0)
    smoke["development_suite"] = SMOKE_SUITE
    smoke["output_dir"] = f"{WORK}/out/baseline_smoke_open"
    smoke["series"] = _series(series, arm, label="smoke_open", against=FILES["s0_open"], differs_in=["development_suite"],
                              role="smoke: S0 on eight of the development cases in the open environment",
                              planned_inputs={key: value for key, value in planned.items() if key != "development_suite_sha256"})

    configs = {FILES["s0_open"]: s0, FILES["clone_open"]: clone, FILES["smoke_open"]: smoke}
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
