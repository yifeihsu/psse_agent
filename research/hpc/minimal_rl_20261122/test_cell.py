"""The cell of the RL plan of 2026-11-22: its scripts parse and name files that exist, its configs are the focused
run's with the plan's changes and nothing else, the stages drive the driver and the probe as intended, and the
status reader reads what they write.

Run on its own (``python -m pytest research/hpc/minimal_rl_20261122/test_cell.py``): it shares its file name with
the other cells' tests, so a run of the whole repository names it with ``--ignore``.
"""
from __future__ import annotations

import contextlib
import io
import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from research.minimal_env.rl_train import MinimalRLConfig

try:
    import torch
except Exception:  # pragma: no cover - torch is optional for the file checks
    torch = None

CELL = Path(__file__).resolve().parent
REPO = CELL.parents[2]
FOCUS_CELL = CELL.parent / "minimal_rl_20261116"
SCRIPTS = ("cell.env", "stage_rl.sbatch", "stage_fork_probe.sbatch", "submit_cell.sh", "status_cell.sh", "deploy_from_windows.sh")
CONFIGS = ("rl_steps.json", "rl_steps_smoke.json", "rl_forks.json", "rl_forks_smoke.json")
RUNS = ("steps", "steps_smoke", "forks", "forks_smoke")
GPU_STAGES = ("stage_rl.sbatch", "stage_fork_probe.sbatch")
WORK = "/scratch/yx3882/minimal_rl_20261122"
FOCUS_WORK = "/scratch/yx3882/minimal_rl_20261116"
SFT_WORK = "/scratch/yx3882/minimal_sft_20261115"


def _settings(path: Path) -> dict[str, str]:
    """``NAME=value`` lines of a shell settings file (no expansion)."""
    found = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^([A-Z_][A-Z0-9_]*)=(.*)$", line)
        if match:
            found[match.group(1)] = match.group(2)
    return found


def _expanded(path: Path, name: str) -> str:
    """A setting with the earlier settings of its file put in."""
    settings = _settings(path)
    value = settings[name]
    for _ in range(8):
        value = re.sub(r"\$([A-Z_][A-Z0-9_]*)", lambda match: settings.get(match.group(1), match.group(0)), value)
    return value


def _bash(script: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-c", script], cwd=CELL, capture_output=True, text=True, check=False)


def _config(name: str, cell: Path = CELL) -> dict:
    return json.loads((cell / name).read_text(encoding="utf-8"))


def _differences(left: dict, right: dict, prefix: str = "") -> dict[str, tuple]:
    """Every leaf at which two configs differ, by dotted name."""
    found: dict[str, tuple] = {}
    for key in sorted(set(left) | set(right)):
        one, other = left.get(key), right.get(key)
        if isinstance(one, dict) and isinstance(other, dict):
            found.update(_differences(one, other, f"{prefix}{key}."))
        elif one != other:
            found[f"{prefix}{key}"] = (one, other)
    return found


class CellFilesTests(unittest.TestCase):
    def test_every_script_parses_and_holds_no_carriage_return(self) -> None:
        for name in SCRIPTS:
            self.assertNotIn(b"\r", (CELL / name).read_bytes(), name)
            checked = subprocess.run(["bash", "-n", name], cwd=CELL, capture_output=True, text=True, check=False)
            self.assertEqual(checked.returncode, 0, f"{name}: {checked.stderr}")
        for name in CONFIGS:
            self.assertNotIn(b"\r", (CELL / name).read_bytes(), name)
        attributes = (CELL / ".gitattributes").read_text(encoding="utf-8")
        for pattern in ("*.json", "*.sh", "*.sbatch", "*.env"):
            self.assertIn(f"{pattern} text eol=lf", attributes)

    def test_the_scripts_name_this_cell_and_files_that_exist(self) -> None:
        environment = CELL / "cell.env"
        self.assertEqual(_settings(environment)["WORK"], WORK)
        self.assertNotEqual(WORK, _settings(FOCUS_CELL / "cell.env")["WORK"])
        self.assertEqual(_expanded(environment, "CELL"), f"{WORK}/source/research/hpc/{CELL.name}")
        for name in ("stage_rl.sbatch", "stage_fork_probe.sbatch", "submit_cell.sh"):
            text = (CELL / name).read_text(encoding="utf-8")
            self.assertIn(f"WORK={WORK}\n", text, name)
            self.assertIn(f'source "$WORK/source/research/hpc/{CELL.name}/cell.env"', text, name)
        self.assertIn(f"WORK={WORK}\n", (CELL / "status_cell.sh").read_text(encoding="utf-8"))
        # The files the cell reads from its own source are in the repository.
        for name in ("TRUTH_SEQUENCES", "DEV_REFERENCE"):
            relative = _expanded(environment, name).replace(f"{WORK}/source/", "")
            self.assertTrue((REPO / relative).is_file(), relative)
        for stage in GPU_STAGES:
            self.assertIn(f"#SBATCH --output={WORK}/logs/%x-%j.out", (CELL / stage).read_text(encoding="utf-8"))

    def test_the_other_cells_are_read_and_never_written_to(self) -> None:
        environment = CELL / "cell.env"
        self.assertEqual(_expanded(environment, "PASS_ADAPTER"), f"{SFT_WORK}/out/pass_base/training/lora")
        self.assertEqual(_expanded(environment, "DEV_PASS"), f"{SFT_WORK}/out/eval_dev300_base")
        self.assertEqual(_expanded(environment, "TRAIN_PASS"), f"{FOCUS_WORK}/out/train_pass")
        for name in ("stage_rl.sbatch", "stage_fork_probe.sbatch", "submit_cell.sh"):
            for line in (CELL / name).read_text(encoding="utf-8").splitlines():
                for variable in ("$PASS", "$DEV_PASS", "$TRAIN_PASS", "$PASS_ADAPTER"):
                    if variable in line:
                        self.assertNotRegex(line, r"(^|\s)(rm|mv|cp|mkdir|touch|tee|>)\s", f"{name}: {line}")
        # Everything the cell writes is under its own work directory.
        for name in ("OUT", "LOGS"):
            self.assertTrue(_expanded(environment, name).startswith(WORK + "/"), name)

    def test_the_gpu_stages_take_any_large_card_and_may_be_preempted(self) -> None:
        for name in GPU_STAGES:
            text = (CELL / name).read_text(encoding="utf-8")
            self.assertIn("#SBATCH --constraint=a100|h100|h200|rtx6000", text, name)
            self.assertIn("#SBATCH --requeue", text, name)
            self.assertIn("#SBATCH --comment=preemption=yes;requeue=true", text, name)
            self.assertIn("#SBATCH --cpus-per-task=16", text, name)
            self.assertNotIn("--partition", text, name)

    def test_the_stages_drive_the_minimal_environments_driver_and_probe(self) -> None:
        stage = (CELL / "stage_rl.sbatch").read_text(encoding="utf-8")
        self.assertIn('DRIVER=("$PY" -m research.minimal_env.rl_train --config "$CFG")', stage)
        self.assertIn("CFG=$CELL/rl_$RL_RUN.json", stage)
        for run in RUNS:
            self.assertTrue((CELL / f"rl_{run}.json").is_file(), run)
        self.assertIn("steps|steps_smoke|forks|forks_smoke) ;;", stage)
        probe = (CELL / "stage_fork_probe.sbatch").read_text(encoding="utf-8")
        self.assertIn('"$PY" -m research.minimal_env.fork_probe --adapter "$PASS_ADAPTER"', probe)
        self.assertTrue((REPO / "research" / "minimal_env" / "fork_probe.py").is_file())
        source = (REPO / "research" / "minimal_env" / "fork_probe.py").read_text(encoding="utf-8")
        for option in re.findall(r"(--[a-z][a-z-]+)", probe[probe.index("COMMON=("):probe.index('echo "[probe:$PROBE:$PROBE_PART] adapter')]):
            self.assertIn(f'"{option}"', source, option)


class ConfigTests(unittest.TestCase):
    """Stage A is the finished focused run with the plan's four changes; stage B is stage A with the forks."""

    def test_stage_a_differs_from_the_focused_run_in_the_plans_settings_alone(self) -> None:
        focused = _config("rl_focus.json", FOCUS_CELL)
        steps = _config("rl_steps.json")
        self.assertEqual(_differences(focused, steps), {
            "evaluate_every": (None, 3),
            "focus.truth_sequences": (f"{FOCUS_WORK}/source/research/hpc/minimal_rl_20261116/train_truth_sequences.json",
                                      f"{WORK}/source/research/hpc/minimal_rl_20261116/train_truth_sequences.json"),
            "forks": (None, {}),
            "output_dir": (f"{FOCUS_WORK}/out/focus", f"{WORK}/out/steps"),
            "reference": (f"{FOCUS_WORK}/source/research/hpc/minimal_sft_20261115/dev300_reference.json",
                          f"{WORK}/source/research/hpc/minimal_sft_20261115/dev300_reference.json"),
            "seed": (20261118, 20261122),
            "update.fresh_optimizer_steps": (None, 1),
            "update.learning_rate": (1e-05, 3e-05),
            "update.optimizer_steps": (1, 8),
        })
        config = MinimalRLConfig.load(CELL / "rl_steps.json")
        settings = config.update_settings(0)
        # Eight steps a batch by case, one while the optimizer is new, the optimizer carried, no clip.
        self.assertEqual((settings.optimizer_steps, settings.fresh_optimizer_steps, settings.chunk_by, settings.carry_optimizer,
                          settings.clip_ratio, settings.loss_normalization), (8, 1, "group", True, None, "dapo"))
        self.assertEqual([config.evaluated(iteration) for iteration in range(6)], [False, False, True, False, False, True])
        self.assertFalse(config.fork_settings().enabled)
        self.assertEqual(config.focus_settings().demonstration_weight, 1.0)
        # 32 cases an iteration: four a step.
        focus = config.focus_settings()
        self.assertEqual((focus.failed_per_iteration + focus.solved_per_iteration) // settings.optimizer_steps, 4)

    def test_stage_b_is_stage_a_with_the_forks_and_without_the_examples(self) -> None:
        self.assertEqual(_differences(_config("rl_steps.json"), _config("rl_forks.json")), {
            "focus.demonstration_weight": (1.0, 0.0),
            "forks.argument_samples": (None, 0), "forks.enabled": (None, True), "forks.min_gain": (None, 0.1), "forks.weight": (None, 1.0),
            "output_dir": (f"{WORK}/out/steps", f"{WORK}/out/forks"),
        })
        config = MinimalRLConfig.load(CELL / "rl_forks.json")
        self.assertTrue(config.fork_settings().enabled)
        self.assertFalse(config.focus_settings().examples)  # every supervised row of the run comes from a fork

    def test_the_smokes_differ_from_their_runs_in_size_alone(self) -> None:
        sizes = {"baseline_from", "development_limit", "evaluate_every", "focus.failed_per_iteration", "focus.solved_per_iteration",
                 "output_dir", "samples_per_root", "update.optimizer_steps", "update_checkpoint_rows"}
        for run in ("steps", "forks"):
            found = _differences(_config(f"rl_{run}.json"), _config(f"rl_{run}_smoke.json"))
            self.assertEqual(set(found), sizes, run)
            self.assertEqual(found["output_dir"][1], f"{WORK}/out/{run}_smoke")
            self.assertEqual((found["development_limit"][1], found["samples_per_root"][1], found["baseline_from"][1]), (8, 2, None))
        # Stage A's smoke runs two iterations: one step while the optimizer is new, then four, and one evaluation skipped.
        steps = MinimalRLConfig.load(CELL / "rl_steps_smoke.json")
        self.assertEqual((steps.update_settings(1).optimizer_steps, steps.update_settings(0).fresh_optimizer_steps), (4, 1))
        self.assertEqual([steps.evaluated(iteration) for iteration in range(2)], [False, True])
        forks = MinimalRLConfig.load(CELL / "rl_forks_smoke.json")
        self.assertEqual(([forks.evaluated(0)], forks.focus_settings().failed_per_iteration, forks.focus_settings().solved_per_iteration),
                         ([True], 3, 1))

    def test_the_configs_name_the_cells_own_folders_and_the_inputs_read_in_place(self) -> None:
        environment = CELL / "cell.env"
        for name, run in zip(CONFIGS, ("steps", "steps_smoke", "forks", "forks_smoke")):
            config = _config(name)
            MinimalRLConfig.load(CELL / name)
            self.assertEqual(config["output_dir"], f"{_expanded(environment, 'OUT')}/{run}", name)
            self.assertEqual(config["initial_adapter"], _expanded(environment, "PASS_ADAPTER"), name)
            self.assertEqual(config["training_suite"], _expanded(environment, "TRAIN_SUITE"), name)
            self.assertEqual(config["development_suite"], _expanded(environment, "DEV_SUITE"), name)
            self.assertEqual(config["reference"], _expanded(environment, "DEV_REFERENCE"), name)
            self.assertEqual(config["focus"]["evaluation"], _expanded(environment, "TRAIN_PASS"), name)
            self.assertEqual(config["focus"]["truth_sequences"], _expanded(environment, "TRUTH_SEQUENCES"), name)
            self.assertIn(config["baseline_from"], (None, _expanded(environment, "DEV_PASS")), name)
            self.assertEqual((config["max_actions"], config["table_versions"], config["max_input_tokens"]),
                             (int(_settings(environment)["RL_MAX_ACTIONS"]), int(_settings(environment)["RL_TABLE_VERSIONS"]),
                              int(_settings(environment)["RL_MAX_INPUT_TOKENS"])), name)


class SubmitAndDeployTests(unittest.TestCase):
    def test_the_submit_script_names_every_job_and_what_it_runs(self) -> None:
        submit = (CELL / "submit_cell.sh").read_text(encoding="utf-8")
        self.assertIn("--job-name=mrp-steps-smoke --export=ALL,RL_RUN=steps_smoke,RL_TARGET=2", submit)
        self.assertIn("--job-name=mrp-forks-smoke --export=ALL,RL_RUN=forks_smoke,RL_TARGET=1", submit)
        self.assertIn('ids=$(chain "mrp-$name" "$links" "ALL,RL_RUN=$name,RL_TARGET=$target")', submit)
        self.assertIn('run steps "${2:?how many iterations the run should reach in all}"', submit)
        self.assertIn('run forks "${2:?how many iterations the run should reach in all}"', submit)
        self.assertIn("--job-name=mrp-probe-smoke --export=ALL,PROBE=smoke,PROBE_PART=close", submit)
        self.assertIn('--job-name="mrp-probe-$which" --export="ALL,PROBE=$which,PROBE_PART=$part/$parts"', submit)
        self.assertIn('--dependency="afterany:$after" --job-name="mrp-probe-$which-close" --export="ALL,PROBE=$which,PROBE_PART=close"', submit)
        self.assertIn("afterany:$previous", submit)
        self.assertNotIn("--partition", submit)
        self.assertNotIn("--constraint", submit)  # the stages name the cards; nothing is pinned at submission
        for command in ("steps-smoke)", "  steps)", "forks-smoke)", "  forks)", "probe-smoke)", "  probe)"):
            self.assertIn(command, submit)
        # Every job waits for what it reads in place.
        for branch in ("steps-smoke)", "forks-smoke)", "probe-smoke)", "  probe)"):
            body = submit[submit.index(branch):]
            end = re.search(r"^    ;;$", body, flags=re.MULTILINE)  # the branch's own end, not a nested one
            self.assertIn("inputs_ready", body[:end.start()], branch)
        self.assertIn("inputs_ready", submit[submit.index("run() {"):submit.index('case "$COMMAND" in')])
        # A probe is submitted once.
        self.assertIn('[[ ! -s "$OUT/probe_$which/summary.json" ]] || { echo "the probe is done already', submit)

    def test_the_link_counts_cover_the_estimated_hours_an_iteration(self) -> None:
        submit = (CELL / "submit_cell.sh").read_text(encoding="utf-8")
        function = next(line for line in submit.splitlines() if line.startswith("run_links() {"))
        self.assertIn("STEPS_HOURS=7\n", submit)
        self.assertIn("FORKS_HOURS=11\n", submit)
        found = _bash(f"{function}; for n in 1 3 9; do run_links $n 7; done; for n in 1 3 9; do run_links $n 11; done")
        self.assertEqual(found.returncode, 0, found.stderr)
        self.assertEqual(found.stdout.split(), ["2", "3", "7", "2", "4", "10"])  # 11 h of each link, one spare

    def test_the_deploy_script_submits_nothing_and_names_committed_files(self) -> None:
        deploy = (CELL / "deploy_from_windows.sh").read_text(encoding="utf-8")
        self.assertNotIn("sbatch ", deploy.replace("stage_rl.sbatch", "").replace("stage_fork_probe.sbatch", ""))
        listed = deploy[deploy.index("for path in"):deploy.index("; do", deploy.index("for path in"))]
        paths = re.findall(r'"\$CELL_REL/([^"]+)"', listed) + re.findall(r"\s((?:research|psse_env)/[A-Za-z0-9_./]+)", listed)
        self.assertGreaterEqual(len(paths), 18)
        for path in paths:
            target = CELL / path if not path.startswith(("research/", "psse_env/")) else REPO / path
            self.assertTrue(target.is_file(), path)
        for name in CONFIGS:
            self.assertIn(f'"$CELL_REL/{name}"', deploy)
        # A command substitution meant for torch inside an ssh command string is expanded on the WSL side.
        for line in deploy.splitlines():
            if "wsl -- ssh" in line:
                self.assertNotIn("\\$(", line, line)
                self.assertNotIn("$USER", line, line)
        # What the deploy asks of the code is there.
        for needle, path in (("evaluate_every", "research/minimal_env/rl_train.py"), ("fork_settings", "research/minimal_env/rl_train.py"),
                             ("fresh_optimizer_steps", "psse_env/rl/update.py"), ("forced_prefix", "psse_env/dagger/research_policy_factory.py"),
                             ("def act_forced", "psse_env/minimal/policy.py"), ('"--unrepaired-in"', "research/minimal_env/fork_probe.py")):
            self.assertIn(needle, (REPO / path).read_text(encoding="utf-8"), needle)
            self.assertIn(needle, deploy, needle)
        self.assertIn("the source is not moved under them", deploy)  # not under a queued or running job of the cell
        self.assertIn("grep -c '^mrp-'", deploy)


#: A stand-in for the driver, for the stage's control flow alone: it keeps a manifest, records every call and
#: moves the manifest on when a phase is closed (the call without a shard), as the driver does.
STUB_DRIVER = r'''
import json, os, sys
from pathlib import Path

args = sys.argv[1:]
config = json.loads(Path(args[args.index("--config") + 1]).read_text())
run = Path(config["output_dir"])
run.mkdir(parents=True, exist_ok=True)
path = run / "manifest.json"
manifest = json.loads(path.read_text()) if path.is_file() else {"iteration": 0, "phase": "baseline"}
if "--status" in args:
    print(json.dumps(manifest))
    sys.exit(0)
with open(run / "calls.txt", "a") as stream:
    if "--check" in args:
        stream.write("check\n")
        sys.exit(0)
    phase = args[args.index("--phase") + 1]
    shard = args[args.index("--shard") + 1] if "--shard" in args else "close"
    stream.write(f"{manifest['iteration']} {phase} {shard} threads={os.environ.get('OMP_NUM_THREADS')}\n")
if phase != manifest["phase"]:
    sys.exit(f"the run is at {manifest['phase']}, not {phase}")
if shard != "close":
    sys.exit(0)
if phase == "update" and os.environ.get("STUB_FAIL_UPDATE"):
    folder = run / f"iteration_{manifest['iteration']:03d}"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "integrity_failure.json").write_text(json.dumps({"failures": ["a stop condition"]}))
    sys.exit(1)
order = ["baseline", "rollout", "update", "evaluate"]
manifest = ({"iteration": manifest["iteration"] + 1, "phase": "rollout"} if phase == "evaluate"
            else {**manifest, "phase": order[order.index(phase) + 1]})
path.write_text(json.dumps(manifest))
'''


#: A stand-in for the probe, for its stage's control flow: it records every call and, on the call without a shard,
#: writes the summary.
STUB_PROBE = r'''
import json, os, sys
from pathlib import Path

args = sys.argv[1:]
out = Path(args[args.index("--output-dir") + 1])
out.mkdir(parents=True, exist_ok=True)
shard = args[args.index("--shard") + 1] if "--shard" in args else "close"
named = {name: args[args.index(name) + 1] for name in ("--suite", "--adapter", "--cases", "--unrepaired-in", "--reference", "--sample",
                                                       "--argument-samples", "--seed", "--model-choice", "--max-actions",
                                                       "--table-versions") if name in args}
with open(out / "calls.txt", "a") as stream:
    stream.write(json.dumps({"shard": shard, "threads": os.environ.get("OMP_NUM_THREADS"), **named}) + "\n")
if os.environ.get("STUB_FAIL_SHARD") == shard:
    sys.exit(1)
if shard == "close":
    (out / "summary.json").write_text(json.dumps({"summary": {}}))
'''


class StageFlowTests(unittest.TestCase):
    """The stages' shell on a copy of the cell under a temporary work directory, with stubs in the code's place."""

    def setUp(self) -> None:
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        base = Path(self.folder.name)
        self.work = base / "work"
        self.passed, self.dev_pass, self.train_pass = base / "pass_base", base / "eval_dev", base / "train_pass"
        cell = self.work / "source" / "research" / "hpc" / CELL.name
        cell.mkdir(parents=True)
        stub = self.work / "source" / "research" / "minimal_env"
        stub.mkdir(parents=True)
        for package in (stub.parent, stub):
            (package / "__init__.py").write_text("", encoding="utf-8")
        (stub / "rl_train.py").write_text(STUB_DRIVER, encoding="utf-8")
        (stub / "fork_probe.py").write_text(STUB_PROBE, encoding="utf-8")
        environment = (CELL / "cell.env").read_text(encoding="utf-8")
        for old, new in ((f"WORK={WORK}", f"WORK={self.work.as_posix()}"),
                         (f"PASS={SFT_WORK}/out/pass_base", f"PASS={self.passed.as_posix()}"),
                         (f"DEV_PASS={SFT_WORK}/out/eval_dev300_base", f"DEV_PASS={self.dev_pass.as_posix()}"),
                         (f"TRAIN_PASS={FOCUS_WORK}/out/train_pass", f"TRAIN_PASS={self.train_pass.as_posix()}"),
                         (f"PY={_settings(CELL / 'cell.env')['PY']}", f"PY={Path(sys.executable).as_posix()}")):
            self.assertIn(old, environment)
            environment = environment.replace(old, new)
        (cell / "cell.env").write_text(environment, encoding="utf-8", newline="\n")
        for name in GPU_STAGES:
            text = (CELL / name).read_text(encoding="utf-8")
            self.assertEqual(text.count(f"WORK={WORK}\n"), 1, name)
            (cell / name).write_text(text.replace(f"WORK={WORK}\n", f"WORK={self.work.as_posix()}\n"), encoding="utf-8", newline="\n")
        sequences = self.work / "source" / "research" / "hpc" / "minimal_rl_20261116"
        sequences.mkdir(parents=True)
        (sequences / "train_truth_sequences.json").write_text("{}", encoding="utf-8")
        for run in RUNS:
            (cell / f"rl_{run}.json").write_text(json.dumps({"output_dir": (self.work / "out" / run).as_posix()}), encoding="utf-8")
        self.cell = cell
        # No card here: a stand-in that ends at once, so the stage's sampler leaves no process behind.
        self.bin = base / "bin"
        self.bin.mkdir()
        (self.bin / "nvidia-smi").write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8", newline="\n")

    def _stage(self, name: str, **environment: str) -> subprocess.CompletedProcess:
        assignments = " ".join(f'{key}="{value}"' for key, value in environment.items())
        script = (f'chmod +x "{self.bin.as_posix()}/nvidia-smi"; PATH="$(cd "{self.bin.as_posix()}" && pwd):$PATH" {assignments} '
                  f'bash "{(self.cell / name).as_posix()}"')
        return subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=False)

    def _inputs(self, *, adapter: bool = True, train_pass: bool = True, dev_pass: bool = True) -> None:
        (self.passed / "training" / "lora").mkdir(parents=True, exist_ok=True)
        (self.passed / "training.done").write_text("{}", encoding="utf-8")
        if adapter:
            (self.passed / "training" / "lora" / "adapter_model.safetensors").write_bytes(b"weights")
        for wanted, folder in ((train_pass, self.train_pass), (dev_pass, self.dev_pass)):
            if wanted:
                folder.mkdir(parents=True, exist_ok=True)
                (folder / "evaluation.json").write_text("{}", encoding="utf-8")

    def _calls(self, run: str) -> list[str]:
        return (self.work / "out" / run / "calls.txt").read_text(encoding="utf-8").split("\n")[:-1]

    def _probe_calls(self, probe: str) -> list[dict]:
        return [json.loads(line) for line in (self.work / "out" / f"probe_{probe}" / "calls.txt").read_text(encoding="utf-8").splitlines()]

    def test_stage_a_s_smoke_runs_two_iterations_from_the_manifest(self) -> None:
        self._inputs()
        done = self._stage("stage_rl.sbatch", RL_RUN="steps_smoke", RL_TARGET="2", RL_WORKERS="2", SLURM_CPUS_PER_TASK="16")
        self.assertEqual(done.returncode, 0, done.stderr[-2000:])
        self.assertIn("mrp-steps_smoke-complete", done.stdout)
        calls = self._calls("steps_smoke")
        self.assertEqual(calls[0], "check")
        phases = [(line.split()[0], line.split()[1], line.split()[2]) for line in calls[1:]]
        # The baseline once, then both iterations; two shards and the closing call but for the update.
        self.assertEqual([(iteration, phase) for iteration, phase, shard in phases if shard == "close"],
                         [("0", "baseline"), ("0", "rollout"), ("0", "update"), ("0", "evaluate"),
                          ("1", "rollout"), ("1", "update"), ("1", "evaluate")])
        for phase in ("baseline", "rollout", "evaluate"):
            self.assertEqual(sorted({shard for _iteration, name, shard in phases if name == phase}), ["0/2", "1/2", "close"], phase)
        self.assertEqual({line.split()[3] for line in calls[1:] if line.split()[1] == "update"}, {"threads=16"})
        self.assertEqual({line.split()[3] for line in calls[1:] if line.split()[2] in ("0/2", "1/2")}, {"threads=8"})
        again = self._stage("stage_rl.sbatch", RL_RUN="steps_smoke", RL_TARGET="2", RL_WORKERS="1")
        self.assertEqual(again.returncode, 0, again.stderr[-2000:])
        self.assertIn("nothing to do", again.stdout)
        self.assertEqual(self._calls("steps_smoke"), calls)

    def test_a_run_waits_for_what_it_reads_in_place(self) -> None:
        early = self._stage("stage_rl.sbatch", RL_RUN="steps", RL_TARGET="1", RL_WORKERS="1")
        self.assertEqual(early.returncode, 2)
        self.assertIn("no training receipt", early.stderr)
        self._inputs(adapter=False, train_pass=False)
        self.assertIn("no adapter at", self._stage("stage_rl.sbatch", RL_RUN="forks", RL_TARGET="1", RL_WORKERS="1").stderr)
        self._inputs(train_pass=False)
        missing = self._stage("stage_rl.sbatch", RL_RUN="forks", RL_TARGET="1", RL_WORKERS="1")
        self.assertEqual(missing.returncode, 2)
        self.assertIn("no training pass", missing.stderr)
        self.assertFalse((self.work / "out" / "forks").exists())
        for bad in (dict(RL_RUN="focus", RL_TARGET="1"), dict(RL_RUN="steps", RL_TARGET="0")):
            self.assertEqual(self._stage("stage_rl.sbatch", **bad).returncode, 2, str(bad))

    def test_an_update_that_fails_its_checks_stops_the_run_for_every_later_job(self) -> None:
        self._inputs()
        stopped = self._stage("stage_rl.sbatch", RL_RUN="forks_smoke", RL_TARGET="1", RL_WORKERS="1", STUB_FAIL_UPDATE="1")
        self.assertEqual(stopped.returncode, 0, stopped.stderr[-2000:])
        self.assertIn("mrp-forks_smoke-stopped", stopped.stdout)
        calls = self._calls("forks_smoke")
        later = self._stage("stage_rl.sbatch", RL_RUN="forks_smoke", RL_TARGET="1", RL_WORKERS="1")
        self.assertIn("failed an integrity check", later.stdout)
        self.assertEqual(self._calls("forks_smoke"), calls)  # nothing was run

    def test_the_probe_forks_its_parts_and_the_closing_job_writes_the_summary(self) -> None:
        self._inputs()
        part = self._stage("stage_fork_probe.sbatch", PROBE="train", PROBE_PART="1/2", PROBE_WORKERS="2", SLURM_CPUS_PER_TASK="16")
        self.assertEqual(part.returncode, 0, part.stderr[-2000:])
        self.assertIn("mrp-probe-train-part-done", part.stdout)
        calls = self._probe_calls("train")
        self.assertEqual(sorted(call["shard"] for call in calls), ["2/4", "3/4"])  # the second of two parts: two of four shards
        self.assertEqual({call["threads"] for call in calls}, {"8"})
        settings = _settings(CELL / "cell.env")
        for call in calls:
            # A seeded sample of the training cases with a truth-built sequence that the supervised model left unrepaired.
            self.assertEqual((call["--sample"], call["--seed"], call["--unrepaired-in"]),
                             (settings["PROBE_TRAIN_SAMPLE"], settings["PROBE_SEED"], self.train_pass.as_posix()))
            self.assertTrue(call["--cases"].endswith("research/hpc/minimal_rl_20261116/train_truth_sequences.json"))
            # (Git Bash on Windows puts its own root before a /scratch argument; on the cluster the path is as written.)
            self.assertTrue(call["--suite"].endswith(settings["TRAIN_SUITE"]), call["--suite"])
            self.assertNotIn("--reference", call)
            self.assertNotIn("--argument-samples", call)
        self.assertFalse((self.work / "out" / "probe_train" / "summary.json").exists())
        closing = self._stage("stage_fork_probe.sbatch", PROBE="train", PROBE_PART="close")
        self.assertEqual(closing.returncode, 0, closing.stderr[-2000:])
        self.assertIn("mrp-probe-train-complete", closing.stdout)
        self.assertTrue((self.work / "out" / "probe_train" / "summary.json").is_file())
        self.assertEqual(self._probe_calls("train")[-1]["shard"], "close")
        # A later job finds the summary and forks nothing.
        before = len(self._probe_calls("train"))
        self.assertIn("nothing to do", self._stage("stage_fork_probe.sbatch", PROBE="train", PROBE_PART="0/2").stdout)
        self.assertEqual(len(self._probe_calls("train")), before)

    def test_the_development_probe_and_the_smoke_name_the_cases_the_supervised_model_left(self) -> None:
        self._inputs()
        settings = _settings(CELL / "cell.env")
        self.assertEqual(self._stage("stage_fork_probe.sbatch", PROBE="dev", PROBE_PART="close").returncode, 0)
        [dev] = self._probe_calls("dev")
        self.assertTrue(dev["--suite"].endswith(settings["DEV_SUITE"]), dev["--suite"])
        self.assertEqual(dev["--unrepaired-in"], self.dev_pass.as_posix())
        self.assertTrue(dev["--reference"].endswith("research/hpc/minimal_sft_20261115/dev300_reference.json"))
        for absent in ("--sample", "--cases", "--argument-samples"):
            self.assertNotIn(absent, dev)  # every unrepaired development case a truth-built sequence repairs
        self.assertEqual(self._stage("stage_fork_probe.sbatch", PROBE="smoke", PROBE_PART="close").returncode, 0)
        [smoke] = self._probe_calls("smoke")
        self.assertEqual((smoke["--sample"], smoke["--argument-samples"]), (settings["PROBE_SMOKE_SAMPLE"], "1"))
        # One sequential worker runs a part's two shards one after the other; a failed shard fails the job.
        failed = self._stage("stage_fork_probe.sbatch", PROBE="train", PROBE_PART="0/1", PROBE_WORKERS="1", STUB_FAIL_SHARD="1/2")
        self.assertNotEqual(failed.returncode, 0)
        self.assertEqual([call["shard"] for call in self._probe_calls("train")], ["0/2", "1/2"])
        # (The training probe has no summary yet: a probe with one leaves before it reads its part.)
        for bad in (dict(PROBE="all", PROBE_PART="close"), dict(PROBE="train", PROBE_PART="2/2"), dict(PROBE="train", PROBE_PART="first")):
            self.assertEqual(self._stage("stage_fork_probe.sbatch", **bad).returncode, 2, str(bad))
        self._inputs()
        (self.dev_pass / "evaluation.json").unlink()
        self.assertIn("no evaluation", self._stage("stage_fork_probe.sbatch", PROBE="smoke", PROBE_PART="0/1").stderr)


@unittest.skipIf(torch is None, "torch is not installed")
class StatusTests(unittest.TestCase):
    """The status script's readers on what the driver and the probe write (with the stand-ins of the driver's own tests)."""

    def test_the_status_reader_reads_a_run_with_forks_and_a_skipped_evaluation(self) -> None:
        from psse_env.minimal.forks import ForkStore, summarize
        from research.minimal_env import test_rl_train as driver_tests

        if not driver_tests.ROOTS:
            driver_tests.setUpModule()
        script = (CELL / "status_cell.sh").read_text(encoding="utf-8")
        self.assertIn("for run in steps_smoke steps forks_smoke forks; do", script)
        self.assertIn("for probe in smoke dev train; do", script)
        reader = script[script.index("<<'PY'\n") + len("<<'PY'\n"):script.index("\nPY\n")]
        probe_reader = script[script.index("<<'PROBEPY'\n") + len("<<'PROBEPY'\n"):script.index("\nPROBEPY\n")]
        with tempfile.TemporaryDirectory() as folder, driver_tests._stand_ins({"never_repair": True}) as rl:
            config = driver_tests._config(folder, forks={"enabled": True}, rows=driver_tests.FOCUS_ROWS, evaluate_every=2,
                                          update={"learning_rate": 0.01, "carry_optimizer": True, "chunk_by": "group",
                                                  "optimizer_steps": 2, "fresh_optimizer_steps": 1})
            with contextlib.redirect_stdout(io.StringIO()):
                for phase in ("rollout", "update", "evaluate", "rollout"):
                    rl.run_phase(config, phase)
            printed = subprocess.run([sys.executable, "-", config.output_dir], input=reader, capture_output=True, text=True, check=False)
            # A probe folder as research.minimal_env.fork_probe writes it, from the run's own fork records.
            records = [payload["record"] for _case, payload in ForkStore(Path(config.output_dir) / "iteration_000" / "forks").iter()]
            probe = Path(folder) / "probe"
            (probe / "cases").mkdir(parents=True)
            for index in range(len(records)):
                (probe / "cases" / f"{index}.json").write_text("{}", encoding="utf-8")
            before = subprocess.run([sys.executable, "-", str(probe)], input=probe_reader, capture_output=True, text=True, check=False)
            (probe / "summary.json").write_text(json.dumps({"selection": {"selected": len(records)}, "summary": summarize(records)}),
                                                encoding="utf-8")
            after = subprocess.run([sys.executable, "-", str(probe)], input=probe_reader, capture_output=True, text=True, check=False)
        self.assertEqual(printed.returncode, 0, printed.stderr)
        text = printed.stdout
        self.assertIn("at iteration 1, phase update;", text)
        self.assertIn("--- iteration_000", text)
        # Two one-state episodes: five other tools each, and the other outcome of the claim.
        self.assertRegex(text, r"forks: 2 of 2 cases; greedy episode unrepaired 2, one forced step repairs [12] of them; states 2, branches 12 "
                               r'\{"ran": 12\}')
        self.assertRegex(text, r"forks: states with a branch to teach \d \(\d repair an unrepaired case\), by tool \{.*\}, margin \{.*\}; "
                               r"forced prefixes canonical 0 of 0; ")
        self.assertRegex(text, r"rows: rows \d+, policy_rows 0, preservation_rows 2, .*supervised_rows \d, supervised_rows_unrenderable 0")
        self.assertRegex(text, r"fork rows in the update: \d of \d kept branches in \d cases, by tool \{.*\}; whose branch repairs \d "
                               r"\(weight [\d.]+ of [\d.]+\), within one action's cost \d; row weights \{.*\}; not rendered 0, "
                               r"over the input limit 0")
        self.assertRegex(text, r"update: 1 step\(s\) on \d+ rows in .* h; rows per step \[\d+\]")  # one step while the optimizer is new
        self.assertIn("update: mean |log ratio| to the sampler by step [", text)
        self.assertIn("integrity: ok True; failures []", text)
        self.assertIn("evaluation: skipped (the run evaluates every 2 iterations; the next evaluation is of iteration 1)", text)
        tail = text[text.index("--- iteration_001"):]
        self.assertIn("forks: 2 of 2 cases", tail)
        self.assertNotIn("update:", tail)
        self.assertEqual((before.returncode, after.returncode), (0, 0), before.stderr + after.stderr)
        self.assertIn(f"cases forked {len(records)}", before.stdout)
        self.assertIn("summary.json: not yet", before.stdout)
        self.assertRegex(after.stdout, r"selected 2; greedy episode unrepaired 2 of 2; one forced step repairs [12] of them \([12] without "
                                       r"a sampled call\); repaired after a call the environment refused, which changes nothing but the "
                                       r"history: \d")
        self.assertIn("states 2, branches 12", after.stdout)
        self.assertRegex(after.stdout, r"states with a branch to teach \d in \d cases \(\d repair an unrepaired case\); by tool \{.*\}; "
                                       r"margin \(the row weight\) \{.*\}")
        self.assertIn("forced prefixes that start the call's own rendering: 0 of 0 checked", after.stdout)
        self.assertRegex(after.stdout, r"control_meter: unrepaired 1 of 1, one forced step repairs 1")
        self.assertRegex(after.stdout, r"fork time [\d.]+ h in all, [\d.]+ s a case")


if __name__ == "__main__":
    unittest.main()
