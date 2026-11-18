"""The cell of the minimal environment's RL run: its scripts parse and name files that exist, its configs are the
driver's example on the cluster's paths, the smoke differs from the run in size alone, and the status reader reads
what the driver writes.

Run on its own (``python -m pytest research/hpc/minimal_rl_20261116/test_cell.py``): it shares its file name with
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

from psse_env.providers.scenario_generator import DEFAULT_COMPOUND_CLASSES
from research.minimal_env.rl_train import MinimalRLConfig

try:
    import torch
except Exception:  # pragma: no cover - torch is optional for the file checks
    torch = None

CELL = Path(__file__).resolve().parent
REPO = CELL.parents[2]
SFT_CELL = CELL.parent / "minimal_sft_20261115"
SCRIPTS = ("cell.env", "stage_rl.sbatch", "stage_train_pass.sbatch", "submit_cell.sh", "status_cell.sh", "deploy_from_windows.sh")
CONFIGS = ("rl_ieee14.json", "rl_smoke.json", "rl_focus.json", "rl_focus_smoke.json")
RUNS = ("ieee14", "smoke", "focus", "focus_smoke")
GPU_STAGES = ("stage_rl.sbatch", "stage_train_pass.sbatch")
WORK = "/scratch/yx3882/minimal_rl_20261116"
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


def _config(name: str) -> dict:
    return json.loads((CELL / name).read_text(encoding="utf-8"))


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
        self.assertEqual(_settings(CELL / "cell.env")["WORK"], WORK)
        self.assertNotEqual(WORK, _settings(SFT_CELL / "cell.env")["WORK"])
        for name in SCRIPTS:
            text = (CELL / name).read_text(encoding="utf-8")
            for path in re.findall(r'\$CELL/([A-Za-z0-9_./]+)', text):
                if path.endswith((".sbatch", ".py", ".env", ".sh", ".json")):
                    self.assertTrue((CELL / path).is_file(), f"{name} names $CELL/{path}")
            for found in re.findall(r"^WORK=(\S+)", text, flags=re.MULTILINE):
                self.assertEqual(found, WORK, name)
            # No job of this cell takes a name of another cell.
            for prefix in ("msf-", "cec-", "orc-", "rli-"):
                self.assertNotIn(prefix, text, name)
        stage = (CELL / "stage_rl.sbatch").read_text(encoding="utf-8")
        self.assertIn(f"#SBATCH --output={WORK}/logs/%x-%j.out", stage)
        self.assertIn("#SBATCH --job-name=mrl", stage)
        self.assertIn("#SBATCH --account=torch_pr_627_general", stage)
        self.assertIn(f'source "$WORK/source/research/hpc/{CELL.name}/cell.env"', stage)
        self.assertIn("CFG=$CELL/rl_$RL_RUN.json", stage)
        for run in RUNS:
            self.assertTrue((CELL / f"rl_{run}.json").is_file())
            self.assertIn(run, stage[stage.index('case "$RL_RUN" in'):stage.index("esac")])
        for name in GPU_STAGES:
            self.assertIn(f"#SBATCH --output={WORK}/logs/%x-%j.out", (CELL / name).read_text(encoding="utf-8"))

    def test_the_supervised_cell_is_read_and_never_written_to(self) -> None:
        self.assertEqual(_expanded(CELL / "cell.env", "PASS"), f"{SFT_WORK}/out/pass_base")
        for name in SCRIPTS:
            for line in (CELL / name).read_text(encoding="utf-8").splitlines():
                if line.lstrip().startswith("#"):
                    continue
                self.assertIsNone(re.search(r"\b(rm|mkdir|mv|touch|tee|git -C)\b[^|;&]*(\$PASS|\$SOURCE_CELL|minimal_sft_20261115/(out|source|data))", line), f"{name}: {line}")
                self.assertIsNone(re.search(r">>?\s*\"?(\$PASS|\$SOURCE_CELL)", line), f"{name}: {line}")
        stage = (CELL / "stage_rl.sbatch").read_text(encoding="utf-8")
        # The smoke works on a copy: the pass removes old checkpoints as it goes.
        self.assertIn('cp "$origin/adapter_config.json" "$origin/adapter_model.safetensors" "$SMOKE_ADAPTER.part/"', stage)
        deploy = (CELL / "deploy_from_windows.sh").read_text(encoding="utf-8")
        self.assertIn("git clone -q $SOURCE_CELL/source $WORK/source", deploy)  # a source of its own, cloned

    def test_the_gpu_stage_takes_any_large_card_and_may_be_preempted(self) -> None:
        for name in GPU_STAGES:
            stage = (CELL / name).read_text(encoding="utf-8")
            self.assertIn("#SBATCH --constraint=a100|h100|h200|rtx6000", stage, name)
            self.assertIn("#SBATCH --gres=gpu:1", stage, name)
            self.assertIn("#SBATCH --requeue", stage, name)
            self.assertIn("#SBATCH --comment=preemption=yes;requeue=true", stage, name)
            self.assertNotIn("--partition", stage, name)

    def test_the_stage_drives_the_minimal_environments_own_driver(self) -> None:
        stage = (CELL / "stage_rl.sbatch").read_text(encoding="utf-8")
        self.assertIn('DRIVER=("$PY" -m research.minimal_env.rl_train --config "$CFG")', stage)
        self.assertNotIn("psse_env.rl.train", stage)
        self.assertIn('"${DRIVER[@]}" --check', stage)
        self.assertIn('--phase "$phase" --shard "$s/$RL_WORKERS"', stage)
        self.assertIn('if [[ "$phase" != update && "$RL_WORKERS" -gt 1 ]]', stage)  # the update is one process
        # The run does not start before the supervised pass has finished, and stops where an update fails a check.
        self.assertIn('[[ -s "$PASS/training.done" ]] || { echo "no training receipt', stage)
        self.assertIn("integrity_failure.json", stage)
        self.assertIn("if another_link_running; then", stage)
        # The check runs after the cheap exits, so a link that has nothing left loads no suite.
        self.assertLess(stage.index("nothing to do"), stage.index('"${DRIVER[@]}" --check'))

    def test_the_smoke_takes_the_newest_checkpoint_that_was_written_whole(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            training = Path(folder) / "training"
            for step, whole in ((16, True), (128, True), (144, False), (9, True)):
                checkpoint = training / f"checkpoint-{step}"
                checkpoint.mkdir(parents=True)
                (checkpoint / "adapter_model.safetensors").write_bytes(b"0")
                if whole:
                    (checkpoint / "trainer_state.json").write_text("{}", encoding="utf-8")
            found = _bash(f'source ./cell.env; PASS="{Path(folder).as_posix()}"; newest_complete_checkpoint')
            self.assertEqual(found.returncode, 0, found.stderr)
            self.assertEqual(found.stdout.strip(), f"{Path(folder).as_posix()}/training/checkpoint-128")
            empty = _bash(f'source ./cell.env; PASS="{Path(folder).as_posix()}/nowhere"; newest_complete_checkpoint')
            self.assertEqual((empty.returncode, empty.stdout.strip()), (0, ""))


class ConfigTests(unittest.TestCase):
    def test_the_run_is_the_drivers_example_on_the_clusters_paths(self) -> None:
        run = _config("rl_ieee14.json")
        example = json.loads((REPO / "research" / "minimal_env" / "rl_config_example.json").read_text(encoding="utf-8"))
        differing = sorted(key for key in set(run) | set(example) if run.get(key) != example.get(key))
        self.assertEqual(differing, ["baseline_from", "development_suite", "initial_adapter", "output_dir", "reference", "training_suite"])
        for name in CONFIGS:
            MinimalRLConfig.load(CELL / name)
        self.assertEqual(sorted(run["root_sampling"]["counts"]), sorted(item.name for item in DEFAULT_COMPOUND_CLASSES))
        self.assertEqual(sum(run["root_sampling"]["counts"].values()), run["roots_per_iteration"])

    def test_the_smoke_differs_from_the_run_in_size_and_start_alone(self) -> None:
        run, smoke = _config("rl_ieee14.json"), _config("rl_smoke.json")
        self.assertEqual(set(run), set(smoke))
        differing = sorted(key for key in run if run[key] != smoke[key])
        self.assertEqual(differing, ["baseline_from", "development_limit", "initial_adapter", "output_dir", "root_sampling",
                                     "roots_per_iteration", "samples_per_root", "update_checkpoint_rows"])
        self.assertEqual((smoke["roots_per_iteration"], smoke["samples_per_root"], smoke["development_limit"]), (6, 2, 8))
        self.assertGreater(smoke["update"]["beta_kl"], 0.0)  # the smoke loads the reference adapter as the run will

    def test_the_configs_name_the_cells_own_folders_and_the_planned_inputs(self) -> None:
        run, smoke = _config("rl_ieee14.json"), _config("rl_smoke.json")
        # The stage reads the run's folder as $OUT/$RL_RUN.
        self.assertEqual((run["output_dir"], smoke["output_dir"]), (f"{WORK}/out/ieee14", f"{WORK}/out/smoke"))
        self.assertIn("RUN_DIR=$OUT/$RL_RUN", (CELL / "stage_rl.sbatch").read_text(encoding="utf-8"))
        self.assertEqual(run["initial_adapter"], _expanded(CELL / "cell.env", "PASS_ADAPTER"))
        self.assertEqual(smoke["initial_adapter"], _expanded(CELL / "cell.env", "SMOKE_ADAPTER"))
        planned = json.loads((CELL.parent / "rl_improvement_20261110" / "series.json").read_text(encoding="utf-8"))["inputs"]
        for config in (run, smoke):
            self.assertEqual(config["training_suite"], planned["training_suite"]["path"])
            self.assertEqual(config["development_suite"], planned["development_suite"]["path"])
            self.assertEqual(config["reference"], f"{WORK}/source/research/hpc/minimal_sft_20261115/dev300_reference.json")
        self.assertTrue((SFT_CELL / "dev300_reference.json").is_file())
        self.assertEqual(len(json.loads((SFT_CELL / "dev300_reference.json").read_text(encoding="utf-8"))), 300)

    def test_the_supervised_cells_evaluation_can_stand_as_the_baseline(self) -> None:
        """The run takes the supervised cell's evaluation of the same adapter as its baseline: what that evaluation
        was made under is what this cell runs under, or the driver refuses it."""
        run = _config("rl_ieee14.json")
        supervised = _settings(SFT_CELL / "cell.env")
        mine = _settings(CELL / "cell.env")
        evaluation = (SFT_CELL / "stage_eval.sbatch").read_text(encoding="utf-8")
        self.assertIn("EVAL_DIR=$OUT/eval_dev300_$START_NAME", evaluation)
        self.assertIn("ADAPTER=$OUT/pass_$START_NAME/training/lora", evaluation)
        self.assertEqual(run["baseline_from"], f"{SFT_WORK}/out/eval_dev300_base")
        self.assertEqual(run["initial_adapter"], f"{SFT_WORK}/out/pass_base/training/lora")
        self.assertIsNone(_config("rl_smoke.json")["baseline_from"])  # the smoke's adapter has no evaluation to take
        self.assertEqual(run["max_actions"], int(supervised["EVAL_MAX_ACTIONS"]))
        self.assertEqual(run["table_versions"], int(supervised["EVAL_TABLE_VERSIONS"]))
        self.assertEqual(run["model_choice"], supervised["MODEL_CHOICE"])
        self.assertFalse(run["load_in_4bit"])
        self.assertNotIn("--load-in-4bit", evaluation)
        self.assertEqual(run["max_input_tokens"], int(supervised["TRAIN_MAX_LENGTH"]))
        self.assertEqual(int(mine["RL_MAX_INPUT_TOKENS"]), run["max_input_tokens"])
        self.assertIn(f"RESEARCH_MAX_NEW_TOKENS={mine['RL_MAX_NEW_TOKENS']}", (SFT_CELL / "cell.env").read_text(encoding="utf-8"))
        for name in ("PY", "HF_HOME"):
            self.assertEqual(mine[name], supervised[name], name)


class FocusConfigTests(unittest.TestCase):
    """The focused run: the driver's focus example on the cluster's paths, drawn by a pass made under its own settings."""

    def test_the_focused_run_is_the_drivers_focus_example_on_the_clusters_paths(self) -> None:
        run = _config("rl_focus.json")
        example = json.loads((REPO / "research" / "minimal_env" / "rl_focus_config_example.json").read_text(encoding="utf-8"))
        differing = sorted(key for key in set(run) | set(example) if run.get(key) != example.get(key))
        self.assertEqual(differing, ["baseline_from", "development_suite", "focus", "initial_adapter", "output_dir", "reference",
                                     "training_suite"])
        self.assertEqual(sorted(key for key in run["focus"] if run["focus"][key] != example["focus"][key]), ["evaluation", "truth_sequences"])
        # Against the plain run it differs in its draw alone: the same start, baseline, episode, update and row checks.
        plain = _config("rl_ieee14.json")
        self.assertEqual(sorted(key for key in set(run) | set(plain) if run.get(key) != plain.get(key)),
                         ["focus", "output_dir", "root_sampling", "roots_per_iteration", "samples_per_root", "seed"])
        self.assertEqual((run["focus"]["failed_per_iteration"] + run["focus"]["solved_per_iteration"]) * run["samples_per_root"],
                         plain["roots_per_iteration"] * plain["samples_per_root"])  # as many sampled episodes an iteration
        self.assertEqual(run["output_dir"], f"{WORK}/out/focus")

    def test_the_focused_smoke_differs_from_the_focused_run_in_size_alone(self) -> None:
        run, smoke = _config("rl_focus.json"), _config("rl_focus_smoke.json")
        self.assertEqual(set(run), set(smoke))
        self.assertEqual(sorted(key for key in run if run[key] != smoke[key]),
                         ["baseline_from", "development_limit", "focus", "output_dir", "samples_per_root", "update_checkpoint_rows"])
        self.assertEqual(sorted(key for key in run["focus"] if run["focus"][key] != smoke["focus"][key]),
                         ["failed_per_iteration", "solved_per_iteration"])
        self.assertEqual((smoke["focus"]["failed_per_iteration"], smoke["focus"]["solved_per_iteration"], smoke["samples_per_root"],
                          smoke["development_limit"], smoke["output_dir"]), (6, 2, 2, 8, f"{WORK}/out/focus_smoke"))
        self.assertEqual(smoke["initial_adapter"], run["initial_adapter"])  # the supervised model itself, not a checkpoint's copy
        self.assertIsNone(smoke["baseline_from"])  # eight cases: its baseline is run by the smoke
        self.assertGreater(smoke["focus"]["demonstration_weight"], 0.0)

    def test_the_training_pass_is_made_where_and_how_the_focused_run_reads_it(self) -> None:
        run = _config("rl_focus.json")
        settings = _settings(CELL / "cell.env")
        self.assertEqual(run["focus"]["truth_sequences"], _expanded(CELL / "cell.env", "TRUTH_SEQUENCES").replace("$CELL", f"{WORK}/source/research/hpc/{CELL.name}"))
        self.assertEqual(run["focus"]["truth_sequences"], f"{WORK}/source/research/hpc/{CELL.name}/train_truth_sequences.json")
        self.assertEqual(run["focus"]["evaluation"], _expanded(CELL / "cell.env", "TRAIN_PASS"))
        self.assertEqual(run["training_suite"], settings["TRAIN_SUITE"])
        # The driver refuses a pass made under another episode, model or adapter than the run's.
        self.assertEqual((int(settings["RL_MAX_ACTIONS"]), int(settings["RL_TABLE_VERSIONS"]), settings["MODEL_CHOICE"]),
                         (run["max_actions"], run["table_versions"], run["model_choice"]))
        stage = (CELL / "stage_train_pass.sbatch").read_text(encoding="utf-8")
        for needle in ('-m research.minimal_env.evaluate --suite "$TRAIN_SUITE" --adapter "$PASS_ADAPTER" --output-dir "$TRAIN_PASS"',
                       '--cases "$TRUTH_SEQUENCES"', '--max-actions "$RL_MAX_ACTIONS"', '--table-versions "$RL_TABLE_VERSIONS"',
                       '--model-choice "$MODEL_CHOICE"', "cell_environment"):
            self.assertIn(needle, stage)
        self.assertNotIn("--load-in-4bit", stage)
        self.assertNotIn("--limit", stage)
        self.assertEqual(run["initial_adapter"], _expanded(CELL / "cell.env", "PASS_ADAPTER"))
        # The file the configs name is in the cell, with the cases the demonstrations repair.
        payload = json.loads((CELL / "train_truth_sequences.json").read_text(encoding="utf-8"))
        self.assertEqual((payload["contract"], len(payload["cases"])), ("minimal_truth_sequences_v1", 2093))
        self.assertNotIn(b"\r", (CELL / "train_truth_sequences.json").read_bytes())


class SubmitAndDeployTests(unittest.TestCase):
    def test_the_submit_script_names_both_jobs_and_what_they_run(self) -> None:
        submit = (CELL / "submit_cell.sh").read_text(encoding="utf-8")
        self.assertIn("--job-name=mrl-smoke --export=ALL,RL_RUN=smoke,RL_TARGET=1", submit)
        self.assertIn('chain mrl-run "$links" "ALL,RL_RUN=ieee14,RL_TARGET=$target"', submit)
        self.assertIn("afterany:$previous", submit)
        self.assertNotIn("--partition", submit)
        self.assertNotIn("--constraint", submit)  # the stage names the cards; nothing is pinned at submission
        # No run before the supervised pass has finished, and no default for how far it goes.
        self.assertIn('[[ -s "$PASS/training.done" ]] || { echo "no training receipt', submit)
        self.assertIn("target=${2:?how many iterations the run should reach in all}", submit)
        for command in ("smoke)", "run)", "train-pass)", "focus-smoke)", "focus)"):
            self.assertIn(command, submit)

    def test_the_focused_runs_jobs_wait_for_the_training_pass(self) -> None:
        submit = (CELL / "submit_cell.sh").read_text(encoding="utf-8")
        self.assertIn('--job-name=mrl-train-pass --export="ALL,PASS_PART=$part/$parts" "$CELL/stage_train_pass.sbatch"', submit)
        self.assertIn('--dependency="afterany:$after" --job-name=mrl-train-pass-close --export=ALL,PASS_PART=close', submit)
        self.assertIn("--job-name=mrl-focus-smoke --export=ALL,RL_RUN=focus_smoke,RL_TARGET=1", submit)
        self.assertIn('chain mrl-focus "$links" "ALL,RL_RUN=focus,RL_TARGET=$target"', submit)
        self.assertIn("target=${2:?how many iterations the focused run should reach in all}", submit)
        for branch in ("focus-smoke)", "  focus)"):
            body = submit[submit.index(branch):]
            body = body[:body.index(";;")]
            self.assertIn('[[ -s "$TRAIN_PASS/evaluation.json" ]] || { echo "no training pass', body, branch)
            self.assertIn('[[ -s "$PASS/training.done" ]] || { echo "no training receipt', body, branch)
        stage = (CELL / "stage_rl.sbatch").read_text(encoding="utf-8")
        self.assertIn('[[ -s "$TRAIN_PASS/evaluation.json" ]]', stage)
        # The pass is submitted once: not again when its evaluation is there.
        self.assertIn('[[ ! -s "$TRAIN_PASS/evaluation.json" ]] || { echo "the training pass is done already', submit)

    def test_the_link_count_covers_fourteen_hours_an_iteration(self) -> None:
        submit = (CELL / "submit_cell.sh").read_text(encoding="utf-8")
        function = next(line for line in submit.splitlines() if line.startswith("run_links() {"))
        found = _bash(f"{function}; for n in 1 2 3 5; do run_links $n; done")
        self.assertEqual(found.returncode, 0, found.stderr)
        self.assertEqual(found.stdout.split(), ["3", "4", "5", "8"])  # 14 h an iteration, 11 h a link, one spare

    def test_the_deploy_script_submits_nothing_and_names_committed_files(self) -> None:
        deploy = (CELL / "deploy_from_windows.sh").read_text(encoding="utf-8")
        self.assertNotIn("sbatch ", deploy.replace("stage_rl.sbatch", ""))
        listed = deploy[deploy.index("for path in"):deploy.index("; do", deploy.index("for path in"))]
        paths = re.findall(r'"\$CELL_REL/([^"]+)"', listed) + re.findall(r"\s((?:research|psse_env)/[A-Za-z0-9_./]+)", listed)
        self.assertGreaterEqual(len(paths), 15)
        for path in paths:
            target = CELL / path if not path.startswith(("research/", "psse_env/")) else REPO / path
            self.assertTrue(target.is_file(), path)
        # A command substitution meant for torch inside an ssh command string is expanded on the WSL side.
        for line in deploy.splitlines():
            if "wsl -- ssh" in line:
                self.assertNotIn("\\$(", line, line)
                self.assertNotIn("$USER", line, line)
        # What the deploy asks of the driver is there.
        driver = (REPO / "research" / "minimal_env" / "rl_train.py").read_text(encoding="utf-8")
        for needle in ('"--check"', "baseline_from", "development_limit", "demonstration_weight"):
            self.assertIn(needle, driver)
            self.assertIn(needle, deploy)
        self.assertIn('"--cases"', (REPO / "research" / "minimal_env" / "evaluate.py").read_text(encoding="utf-8"))
        self.assertIn('"--cases"', deploy)
        for name in ("stage_train_pass.sbatch", "rl_focus.json", "rl_focus_smoke.json", "train_truth_sequences.json"):
            self.assertIn(f'"$CELL_REL/{name}"', deploy)
        self.assertIn("the source is not moved under them", deploy)  # not under a queued or running job of the cell


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


#: A stand-in for the evaluation script, for the training-pass stage's control flow: it records every call and, on
#: the call without a shard, writes the evaluation.
STUB_EVALUATE = r'''
import json, os, sys
from pathlib import Path

args = sys.argv[1:]
out = Path(args[args.index("--output-dir") + 1])
out.mkdir(parents=True, exist_ok=True)
shard = args[args.index("--shard") + 1] if "--shard" in args else "close"
named = {name: args[args.index(name) + 1] for name in ("--suite", "--adapter", "--cases", "--model-choice", "--max-actions", "--table-versions")}
with open(out / "calls.txt", "a") as stream:
    stream.write(json.dumps({"shard": shard, "threads": os.environ.get("OMP_NUM_THREADS"), **named}) + "\n")
if os.environ.get("STUB_FAIL_SHARD") == shard:
    sys.exit(1)
if shard == "close":
    (out / "evaluation.json").write_text(json.dumps({"summary": {}}))
'''


class StageFlowTests(unittest.TestCase):
    """The stage's shell on a copy of the cell under a temporary work directory, with a stub in the driver's place."""

    def setUp(self) -> None:
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.work = Path(self.folder.name) / "work"
        self.passed = Path(self.folder.name) / "pass_base"
        cell = self.work / "source" / "research" / "hpc" / CELL.name
        cell.mkdir(parents=True)
        stub = self.work / "source" / "research" / "minimal_env"
        stub.mkdir(parents=True)
        for package in (stub.parent, stub):
            (package / "__init__.py").write_text("", encoding="utf-8")
        (stub / "rl_train.py").write_text(STUB_DRIVER, encoding="utf-8")
        (stub / "evaluate.py").write_text(STUB_EVALUATE, encoding="utf-8")
        environment = (CELL / "cell.env").read_text(encoding="utf-8")
        for old, new in ((f"WORK={WORK}", f"WORK={self.work.as_posix()}"), (f"PASS={SFT_WORK}/out/pass_base", f"PASS={self.passed.as_posix()}"),
                         (f"PY={_settings(CELL / 'cell.env')['PY']}", f"PY={Path(sys.executable).as_posix()}")):
            self.assertIn(old, environment)
            environment = environment.replace(old, new)
        (cell / "cell.env").write_text(environment, encoding="utf-8", newline="\n")
        stage = (CELL / "stage_rl.sbatch").read_text(encoding="utf-8")
        self.assertEqual(stage.count(f"WORK={WORK}\n"), 1)
        (cell / "stage_rl.sbatch").write_text(stage.replace(f"WORK={WORK}\n", f"WORK={self.work.as_posix()}\n"), encoding="utf-8", newline="\n")
        train_pass = (CELL / "stage_train_pass.sbatch").read_text(encoding="utf-8")
        self.assertEqual(train_pass.count(f"WORK={WORK}\n"), 1)
        (cell / "stage_train_pass.sbatch").write_text(train_pass.replace(f"WORK={WORK}\n", f"WORK={self.work.as_posix()}\n"),
                                                      encoding="utf-8", newline="\n")
        (cell / "train_truth_sequences.json").write_text("{}", encoding="utf-8")
        for run in RUNS:
            (cell / f"rl_{run}.json").write_text(json.dumps({"output_dir": (self.work / "out" / run).as_posix()}), encoding="utf-8")
        self.stage = cell / "stage_rl.sbatch"
        self.train_pass = cell / "stage_train_pass.sbatch"
        # No card here: a stand-in that ends at once, so the stage's sampler leaves no process behind.
        self.bin = Path(self.folder.name) / "bin"
        self.bin.mkdir()
        (self.bin / "nvidia-smi").write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8", newline="\n")

    def _checkpoint(self, step: int, *, whole: bool = True) -> None:
        folder = self.passed / "training" / f"checkpoint-{step}"
        folder.mkdir(parents=True)
        (folder / "adapter_config.json").write_text("{}", encoding="utf-8")
        (folder / "adapter_model.safetensors").write_bytes(b"weights")
        (folder / "tokenizer.json").write_text("{}", encoding="utf-8")
        if whole:
            (folder / "trainer_state.json").write_text("{}", encoding="utf-8")

    def _stage(self, stage: Path | None = None, **environment: str) -> subprocess.CompletedProcess:
        assignments = " ".join(f'{name}="{value}"' for name, value in environment.items())
        script = (f'chmod +x "{self.bin.as_posix()}/nvidia-smi"; PATH="$(cd "{self.bin.as_posix()}" && pwd):$PATH" {assignments} '
                  f'bash "{(stage or self.stage).as_posix()}"')
        return subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=False)

    def _receipt(self) -> None:
        (self.passed / "training" / "lora").mkdir(parents=True, exist_ok=True)
        (self.passed / "training.done").write_text("{}", encoding="utf-8")

    def _pass_calls(self) -> list[dict]:
        return [json.loads(line) for line in (self.work / "out" / "train_pass" / "calls.txt").read_text(encoding="utf-8").splitlines()]

    def _calls(self, run: str) -> list[str]:
        return (self.work / "out" / run / "calls.txt").read_text(encoding="utf-8").split("\n")[:-1]

    def test_the_smoke_copies_a_whole_checkpoint_and_runs_every_phase_once(self) -> None:
        self._checkpoint(32)
        self._checkpoint(48, whole=False)
        done = self._stage(RL_RUN="smoke", RL_TARGET="1", RL_WORKERS="2", SLURM_CPUS_PER_TASK="16")
        self.assertEqual(done.returncode, 0, done.stderr[-2000:])
        self.assertIn("mrl-smoke-complete", done.stdout)
        adapter = self.work / "data" / "smoke_adapter"
        self.assertEqual(sorted(path.name for path in adapter.iterdir()), ["SOURCE.txt", "adapter_config.json", "adapter_model.safetensors"])
        self.assertIn("checkpoint-32 at ", (adapter / "SOURCE.txt").read_text(encoding="utf-8"))
        calls = self._calls("smoke")
        self.assertEqual(calls[0], "check")
        # Two shards and then the closing call for the baseline, the rollout and the evaluation; the update alone.
        phases = [(line.split()[1], line.split()[2]) for line in calls[1:]]
        self.assertEqual([phase for phase, shard in phases if shard == "close"], ["baseline", "rollout", "update", "evaluate"])
        for phase in ("baseline", "rollout", "evaluate"):
            self.assertEqual(sorted(shard for name, shard in phases if name == phase), ["0/2", "1/2", "close"], phase)
            self.assertEqual({line.split()[3] for line in calls[1:] if line.split()[1] == phase and line.split()[2] != "close"}, {"threads=8"})
        self.assertEqual([line.split()[2:] for line in calls[1:] if line.split()[1] == "update"], [["close", "threads=16"]])
        # A second job finds the target reached, loads nothing and copies nothing again.
        again = self._stage(RL_RUN="smoke", RL_TARGET="1", RL_WORKERS="2")
        self.assertEqual(again.returncode, 0, again.stderr[-2000:])
        self.assertIn("nothing to do", again.stdout)
        self.assertEqual(self._calls("smoke"), calls)
        # A higher target goes on from the manifest: the baseline is not run again.
        further = self._stage(RL_RUN="smoke", RL_TARGET="2", RL_WORKERS="1")
        self.assertEqual(further.returncode, 0, further.stderr[-2000:])
        self.assertEqual([line.split()[:3] for line in self._calls("smoke")[len(calls) + 1:]],
                         [["1", "rollout", "close"], ["1", "update", "close"], ["1", "evaluate", "close"]])

    def test_an_update_that_fails_its_checks_stops_the_run_for_every_later_job(self) -> None:
        self._checkpoint(16)
        stopped = self._stage(RL_RUN="smoke", RL_TARGET="1", RL_WORKERS="1", STUB_FAIL_UPDATE="1")
        self.assertEqual(stopped.returncode, 0, stopped.stderr[-2000:])
        self.assertIn("mrl-smoke-stopped", stopped.stdout)
        self.assertNotIn("mrl-smoke-complete", stopped.stdout)
        calls = self._calls("smoke")
        later = self._stage(RL_RUN="smoke", RL_TARGET="1", RL_WORKERS="1")
        self.assertEqual(later.returncode, 0, later.stderr[-2000:])
        self.assertIn("failed an integrity check", later.stdout)
        self.assertEqual(self._calls("smoke"), calls)  # nothing was run

    def test_the_run_waits_for_the_supervised_pass_and_the_smoke_for_a_checkpoint(self) -> None:
        early = self._stage(RL_RUN="ieee14", RL_TARGET="1", RL_WORKERS="1")
        self.assertEqual(early.returncode, 2)
        self.assertIn("the supervised pass has not finished", early.stderr)
        nothing = self._stage(RL_RUN="smoke", RL_TARGET="1", RL_WORKERS="1")
        self.assertEqual(nothing.returncode, 2)
        self.assertIn("neither an adapter nor a complete checkpoint", nothing.stderr)
        for bad in (dict(RL_RUN="dev", RL_TARGET="1"), dict(RL_RUN="smoke", RL_TARGET="0"), dict(RL_RUN="smoke")):
            self.assertEqual(self._stage(**bad).returncode, 2 if "RL_TARGET" in bad else 1, bad)
        # With the receipt the run starts, from the pass's adapter and with no copy.
        (self.passed / "training" / "lora").mkdir(parents=True)
        (self.passed / "training.done").write_text("{}", encoding="utf-8")
        run = self._stage(RL_RUN="ieee14", RL_TARGET="1", RL_WORKERS="1")
        self.assertEqual(run.returncode, 0, run.stderr[-2000:])
        self.assertIn("mrl-ieee14-complete", run.stdout)
        self.assertFalse((self.work / "data" / "smoke_adapter").exists())

    def test_the_focused_run_waits_for_the_supervised_pass_and_for_the_training_pass(self) -> None:
        for run in ("focus", "focus_smoke"):
            early = self._stage(RL_RUN=run, RL_TARGET="1", RL_WORKERS="1")
            self.assertEqual(early.returncode, 2, run)
            self.assertIn("the supervised pass has not finished", early.stderr, run)
        self._receipt()
        for run in ("focus", "focus_smoke"):
            waiting = self._stage(RL_RUN=run, RL_TARGET="1", RL_WORKERS="1")
            self.assertEqual(waiting.returncode, 2, run)
            self.assertIn("pass over the training cases has not finished", waiting.stderr, run)
            self.assertFalse((self.work / "out" / run).exists(), run)
        (self.work / "out" / "train_pass").mkdir(parents=True)
        (self.work / "out" / "train_pass" / "evaluation.json").write_text("{}", encoding="utf-8")
        for run in ("focus", "focus_smoke"):
            done = self._stage(RL_RUN=run, RL_TARGET="1", RL_WORKERS="2", SLURM_CPUS_PER_TASK="16")
            self.assertEqual(done.returncode, 0, done.stderr[-2000:])
            self.assertIn(f"mrl-{run}-complete", done.stdout)
            calls = self._calls(run)
            self.assertEqual(calls[0], "check")
            self.assertEqual([line.split()[1] for line in calls[1:] if line.split()[2] == "close"], ["baseline", "rollout", "update", "evaluate"])
        self.assertFalse((self.work / "data" / "smoke_adapter").exists())  # the focused smoke starts from the pass's adapter itself

    def test_the_training_pass_runs_its_two_shards_and_the_closing_job_writes_the_evaluation(self) -> None:
        early = self._stage(self.train_pass, PASS_PART="0/4")
        self.assertEqual(early.returncode, 2)
        self.assertIn("the supervised pass has not finished", early.stderr)
        self._receipt()
        for bad in ("4/4", "1", "a/2", "0/0"):
            refused = self._stage(self.train_pass, PASS_PART=bad)
            self.assertEqual(refused.returncode, 2, bad)
        self.assertEqual(self._stage(self.train_pass).returncode, 1)  # no part named
        self.assertFalse((self.work / "out" / "train_pass" / "calls.txt").exists())

        # Part 1 of 4 on a large card: shards 2 and 3 of 8 at once, half the threads each.
        large = self._stage(self.train_pass, PASS_PART="1/4", PASS_WORKERS="2", SLURM_CPUS_PER_TASK="16")
        self.assertEqual(large.returncode, 0, large.stderr[-2000:])
        self.assertIn("mrl-train-pass-part-done", large.stdout)
        calls = self._pass_calls()
        self.assertEqual(sorted(call["shard"] for call in calls), ["2/8", "3/8"])
        self.assertEqual({call["threads"] for call in calls}, {"8"})
        expected = {"--adapter": f"{self.passed.as_posix()}/training/lora",
                    "--cases": f"{self.work.as_posix()}/source/research/hpc/{CELL.name}/train_truth_sequences.json",
                    "--model-choice": "12b", "--max-actions": "16", "--table-versions": "2"}
        for call in calls:
            self.assertEqual({name: call[name] for name in expected}, expected)
            # The cluster's path; Git Bash on Windows puts its own root before an argument that starts with a slash.
            self.assertTrue(call["--suite"].endswith(_settings(CELL / "cell.env")["TRAIN_SUITE"]), call["--suite"])
        # Part 3 of 4 on a smaller card: shards 6 and 7, one after the other, all the threads.
        small = self._stage(self.train_pass, PASS_PART="3/4", PASS_WORKERS="1", SLURM_CPUS_PER_TASK="16")
        self.assertEqual(small.returncode, 0, small.stderr[-2000:])
        self.assertEqual([(call["shard"], call["threads"]) for call in self._pass_calls()[2:]], [("6/8", "16"), ("7/8", "16")])
        self.assertFalse((self.work / "out" / "train_pass" / "evaluation.json").exists())  # a part writes no evaluation
        # A shard that fails fails its job.
        failed = self._stage(self.train_pass, PASS_PART="0/4", PASS_WORKERS="2", STUB_FAIL_SHARD="1/8")
        self.assertEqual(failed.returncode, 1)
        self.assertNotIn("mrl-train-pass-part-done", failed.stdout)

        # The closing job runs the call without a shard, which writes the evaluation; a later job finds it.
        before = len(self._pass_calls())
        closing = self._stage(self.train_pass, PASS_PART="close", SLURM_CPUS_PER_TASK="16")
        self.assertEqual(closing.returncode, 0, closing.stderr[-2000:])
        self.assertIn("mrl-train-pass-complete", closing.stdout)
        self.assertEqual([(call["shard"], call["threads"]) for call in self._pass_calls()[before:]], [("close", "16")])
        self.assertTrue((self.work / "out" / "train_pass" / "evaluation.json").is_file())
        for part in ("2/4", "close"):
            later = self._stage(self.train_pass, PASS_PART=part)
            self.assertEqual(later.returncode, 0, later.stderr[-2000:])
            self.assertIn("nothing to do", later.stdout)
        self.assertEqual(len(self._pass_calls()), before + 1)


@unittest.skipIf(torch is None, "torch is not installed")
class StatusTests(unittest.TestCase):
    """The status script's reader on a run the driver wrote (with the stand-ins of the driver's own tests)."""

    def test_the_status_reader_reads_what_the_driver_writes(self) -> None:
        from research.minimal_env import test_rl_train as driver_tests

        if not driver_tests.ROOTS:
            driver_tests.setUpModule()
        script = (CELL / "status_cell.sh").read_text(encoding="utf-8")
        reader = script[script.index("<<'PY'\n") + len("<<'PY'\n"):script.index("\nPY\n")]
        with tempfile.TemporaryDirectory() as folder, driver_tests._stand_ins({}) as rl:
            config = driver_tests._config(folder, baseline_evaluation=True)
            with contextlib.redirect_stdout(io.StringIO()):
                rl.run_iterations(config, 1)
                rl.run_phase(config, "rollout")
            printed = subprocess.run([sys.executable, "-", config.output_dir], input=reader, capture_output=True, text=True, check=False)
        self.assertEqual(printed.returncode, 0, printed.stderr)
        text = printed.stdout
        solved = sum(driver_tests.REPAIRED)
        self.assertIn("at iteration 1, phase update;", text)
        self.assertIn("baseline: cases 2, solved 0, strict_solved 0, solved_and_declared_resolved 0, false_resolved 2", text)
        self.assertIn("source: evaluated by this run", text)
        self.assertIn("--- iteration_000", text)
        self.assertIn(f"rollout: groups 2, episodes 4, solved {solved},", text)
        self.assertIn(f"groups_with_repaired_and_unrepaired_samples {solved}", text)
        self.assertRegex(text, r"rows: rows \d+, policy_rows \d+, preservation_rows 0, informative_groups 2,")
        self.assertRegex(text, r"update: 1 step\(s\) on \d+ rows in ")
        self.assertIn("rows without a record 0", text)
        self.assertIn('"steps_after": 1', text)
        self.assertIn("integrity: ok True; failures []", text)
        self.assertIn(f"evaluation: cases 2, solved {solved},", text)
        self.assertIn(f"against the baseline: 0 -> {solved} of 2, gained {solved}, lost 0", text)
        # The second iteration's rollout is closed and nothing after it exists yet.
        tail = text[text.index("--- iteration_001"):]
        self.assertIn("rollout: groups 2, episodes 4", tail)
        self.assertNotIn("update:", tail)
        self.assertNotIn("truth-built examples", text)  # a plain run has none

    def test_the_status_reader_reads_a_focused_run_and_the_training_pass(self) -> None:
        from research.minimal_env import test_rl_train as driver_tests

        if not driver_tests.ROOTS:
            driver_tests.setUpModule()
        script = (CELL / "status_cell.sh").read_text(encoding="utf-8")
        reader = script[script.index("<<'PY'\n") + len("<<'PY'\n"):script.index("\nPY\n")]
        pass_reader = script[script.index("<<'PASSPY'\n") + len("<<'PASSPY'\n"):script.index("\nPASSPY\n")]
        self.assertIn("for run in smoke ieee14 focus_smoke focus; do", script)
        with tempfile.TemporaryDirectory() as folder, driver_tests._stand_ins({"never_repair": True}) as rl:
            config = driver_tests._config(folder, focus=driver_tests._focus(folder), rows=driver_tests.FOCUS_ROWS)
            with contextlib.redirect_stdout(io.StringIO()):
                rl.run_phase(config, "rollout")
                rl.run_phase(config, "update")
            printed = subprocess.run([sys.executable, "-", config.output_dir], input=reader, capture_output=True, text=True, check=False)
            # The training pass as research.minimal_env.evaluate writes it: three cases, two repaired.
            passed = Path(folder) / "pass"
            (passed / "episodes").mkdir(parents=True)
            for name in ("a", "b", "c"):
                (passed / "episodes" / f"{name}.json").write_text("{}", encoding="utf-8")
            before = subprocess.run([sys.executable, "-", str(passed)], input=pass_reader, capture_output=True, text=True, check=False)
            (passed / "evaluation.json").write_text(json.dumps({"summary": {
                "totals": {"cases": 3, "solved": 2, "false_resolved": 1}, "mean_actions": 4.5,
                "by_class": {"x": {"cases": 2, "solved": 2}, "y": {"cases": 1, "solved": 0}}}}), encoding="utf-8")
            after = subprocess.run([sys.executable, "-", str(passed)], input=pass_reader, capture_output=True, text=True, check=False)
        self.assertEqual(printed.returncode, 0, printed.stderr)
        text = printed.stdout
        self.assertIn("focus, cases left unrepaired by the starting adapter: 2 drawn of 2; with a repaired sample 0 (and an unrepaired one 0), "
                      "with none 2; episodes repaired 0 of 4", text)
        self.assertIn("truth-built examples: 2 kept for 2 cases without a repaired sample; without a sequence 0, replays that did not repair 0", text)
        self.assertRegex(text, r"rows: rows 7, policy_rows 0, preservation_rows 2, .*supervised_rows 5, supervised_rows_unrenderable 0")
        self.assertRegex(text, r"truth-built examples in the update: 5 rows of 2 examples \(2 kept\), row weights \{.*\}, not above their samples 0, "
                               r"over the input limit 0; supervised loss \[\d")
        self.assertIn("integrity: ok True; failures []", text)
        self.assertEqual((before.returncode, after.returncode), (0, 0), before.stderr + after.stderr)
        self.assertIn("episodes written 3", before.stdout)
        self.assertIn("evaluation.json: not yet", before.stdout)
        self.assertIn("cases 3, repaired 2, left unrepaired 1; false_resolved 1,", after.stdout)
        self.assertIn("x: repaired 2 of 2", after.stdout)
        self.assertIn("y: repaired 0 of 1", after.stdout)


if __name__ == "__main__":
    unittest.main()
