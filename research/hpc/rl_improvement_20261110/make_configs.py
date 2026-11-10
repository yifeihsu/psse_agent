"""The series' run configs, written from ``series.json`` (workplan of 2026-11-10, ticket 2).

    python -m research.hpc.rl_improvement_20261110.make_configs           # write configs/*.json
    python -m research.hpc.rl_improvement_20261110.make_configs --check   # the files on disk are what the series gives

``series.json`` holds one base config and, per arm, the fields that differ.
Every run config is the base with its arm's overrides, so two matched arms
differ in exactly the fields the series names for them, and a change to the
base reaches every arm.  The blocks ``update``, ``reward``, ``rows`` and
``environment_options`` are overridden key by key; any other field is
replaced whole.

Each config carries a ``series`` block: the arm, what it is matched with, and
the digests of the inputs the series was planned on (the suites, the expert
reference, the reserved roots).  The digests are a record for the run's
provenance file (``provenance.py``), not a gate: a run on other inputs is
reported, not refused.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CELL = Path(__file__).resolve().parent
SERIES_FILE = CELL / "series.json"
CONFIG_DIR = CELL / "configs"
RESERVED_ROOTS_FILE = CELL / "reserved_roots.json"
#: Blocks an arm overrides key by key; every other field is replaced whole.
MERGED_FIELDS = ("update", "reward", "rows", "environment_options")
#: The arms of the experiment proper, in the order the workplan runs them.
EXPERIMENT_ARMS = ("A", "B", "C", "D")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_series(path: Path = SERIES_FILE) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def planned_inputs(series: Mapping[str, Any], config: Mapping[str, Any], *, reserved_roots_file: Path = RESERVED_ROOTS_FILE) -> dict[str, Any]:
    """The digests of the inputs a config reads, as the series was planned on them."""

    inputs = series["inputs"]
    planned: dict[str, Any] = {"training_suite_sha256": inputs["training_suite"]["sha256"]}
    if config.get("development_suite"):
        planned["development_suite_sha256"] = inputs["development_suite"]["sha256"]
    if config.get("reserved_roots") and Path(reserved_roots_file).is_file():
        planned["reserved_roots_sha256"] = file_sha256(reserved_roots_file)
    planned["expert_reference_sha256"] = inputs["expert_reference"]["sha256"]
    return planned


def arm_config(series: Mapping[str, Any], name: str, *, reserved_roots_file: Path = RESERVED_ROOTS_FILE) -> dict[str, Any]:
    """The run config of one arm: the base, the arm's overrides, its output directory and its ``series`` block."""

    arm = series["arms"][name]
    config = copy.deepcopy(dict(series["base"]))
    unknown = sorted(set(arm["overrides"]) - set(config))
    if unknown:
        raise ValueError(f"arm {name} overrides fields the base does not have: {unknown}")
    for key, value in arm["overrides"].items():
        if key in MERGED_FIELDS and isinstance(value, Mapping):
            config[key] = {**config[key], **copy.deepcopy(dict(value))}
        else:
            config[key] = copy.deepcopy(value)
    config["output_dir"] = f"{series['work_dir']}/{arm['output']}"
    config["series"] = {
        "name": series["name"],
        "arm": name,
        "role": arm["role"],
        "config_file": arm["file"],
        "workplan": series["workplan"],
        "matched": copy.deepcopy(series.get("matched", {}).get(name)),
        # The rule expert's record on the development cases, for the scorecard (the driver does not read it).
        "expert_reference": series["inputs"]["expert_reference"]["path"],
        "planned_inputs": planned_inputs(series, config, reserved_roots_file=reserved_roots_file),
    }
    return config


def build(series: Mapping[str, Any], *, reserved_roots_file: Path = RESERVED_ROOTS_FILE) -> dict[str, dict[str, Any]]:
    """Every arm's config, by file name; each one is loaded through the driver's own validation."""

    from psse_env.rl.train import RLConfig

    configs: dict[str, dict[str, Any]] = {}
    outputs: set[str] = set()
    for name, arm in series["arms"].items():
        config = arm_config(series, name, reserved_roots_file=reserved_roots_file)
        RLConfig(**config).validate()
        if arm["file"] in configs or config["output_dir"] in outputs:
            raise ValueError(f"arm {name} shares its config file or its output directory with another arm")
        configs[arm["file"]] = config
        outputs.add(config["output_dir"])
    return configs


def differing_fields(first: Mapping[str, Any], second: Mapping[str, Any]) -> list[str]:
    """The config fields two arms differ in, beside where they write and what they are called."""

    return sorted(key for key in set(first) | set(second) if key not in ("output_dir", "series") and first.get(key) != second.get(key))


def check_matched(series: Mapping[str, Any], configs: Mapping[str, Mapping[str, Any]]) -> list[str]:
    """What is wrong with the matched pairs: an arm that differs from its control in more, or less, than the series says."""

    by_arm = {config["series"]["arm"]: config for config in configs.values()}
    problems = []
    for name, rule in series.get("matched", {}).items():
        pairs = [(rule["against"], rule["differs_in"])]
        if rule.get("also_against"):
            pairs.append((rule["also_against"], rule["also_differs_in"]))
        for other, expected in pairs:
            found = differing_fields(by_arm[name], by_arm[other])
            if found != sorted(expected):
                problems.append(f"arm {name} differs from arm {other} in {found}, the series says {sorted(expected)}")
    return problems


def render(config: Mapping[str, Any]) -> str:
    return json.dumps(config, indent=2) + "\n"


def write(configs: Mapping[str, Mapping[str, Any]], directory: Path = CONFIG_DIR) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for name, config in configs.items():
        path = directory / name
        # LF on every platform: the files go to the cluster as they are.
        path.write_text(render(config), encoding="utf-8", newline="\n")
        written.append(path)
    return written


def check(configs: Mapping[str, Mapping[str, Any]], directory: Path = CONFIG_DIR) -> list[str]:
    """What differs between the series and the config files on disk."""

    problems = []
    for name, config in configs.items():
        path = directory / name
        if not path.is_file():
            problems.append(f"{name} is not written")
        elif path.read_bytes() != render(config).encode("utf-8"):
            problems.append(f"{name} is not what series.json gives (run make_configs.py)")
    extra = sorted(path.name for path in directory.glob("*.json") if path.name not in configs) if directory.is_dir() else []
    problems.extend(f"{name} is in {directory.name}/ and not in the series" for name in extra)
    return problems


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--series", default=SERIES_FILE, type=Path)
    parser.add_argument("--output-dir", default=CONFIG_DIR, type=Path)
    parser.add_argument("--check", action="store_true", help="write nothing; fail when the files on disk differ from the series")
    args = parser.parse_args(argv)
    series = load_series(args.series)
    configs = build(series)
    problems = check_matched(series, configs)
    if args.check:
        problems.extend(check(configs, args.output_dir))
    else:
        for path in write(configs, args.output_dir):
            print(path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path)
    for problem in problems:
        print(problem, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
