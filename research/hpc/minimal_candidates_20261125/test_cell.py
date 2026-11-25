"""The cell of the candidate read-out: its scripts parse and name this cell, the cells it reads are never written
to, and the stage runs the scorer's shards and its closing call for the arm it is given.

Run on its own (``python -m pytest research/hpc/minimal_candidates_20261125/test_cell.py``): it shares its file name
with the other cells' tests, so a run of the whole repository names it with ``--ignore``.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

CELL = Path(__file__).resolve().parent
REPO = CELL.parents[2]
SCRIPTS = ("cell.env", "stage_candidates.sbatch", "submit_cell.sh", "status_cell.sh", "deploy_from_windows.sh")
WORK = "/scratch/yx3882/minimal_candidates_20261125"
SFT_WORK = "/scratch/yx3882/minimal_sft_20261115"
RL_WORK = "/scratch/yx3882/minimal_rl_20261122"


def _settings(path: Path) -> dict[str, str]:
    found = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^([A-Z_][A-Z0-9_]*)=(.*)$", line)
        if match:
            found[match.group(1)] = match.group(2)
    return found


class CellFilesTests(unittest.TestCase):
    def test_every_script_parses_and_holds_no_carriage_return(self) -> None:
        for name in SCRIPTS:
            self.assertNotIn(b"\r", (CELL / name).read_bytes(), name)
            checked = subprocess.run(["bash", "-n", name], cwd=CELL, capture_output=True, text=True, check=False)
            self.assertEqual(checked.returncode, 0, f"{name}: {checked.stderr}")
        attributes = (CELL / ".gitattributes").read_text(encoding="utf-8")
        for pattern in ("*.json", "*.sh", "*.sbatch", "*.env"):
            self.assertIn(f"{pattern} text eol=lf", attributes)

    def test_the_scripts_name_this_cell_and_its_own_jobs(self) -> None:
        self.assertEqual(_settings(CELL / "cell.env")["WORK"], WORK)
        for name in SCRIPTS:
            text = (CELL / name).read_text(encoding="utf-8")
            for found in re.findall(r"^WORK=(\S+)", text, flags=re.MULTILINE):
                self.assertEqual(found, WORK, name)
            for prefix in ("mrl-", "mrp-", "msf-", "mdg-", "cec-", "orc-", "rli-"):
                self.assertNotIn(f"name={prefix}", text, name)
                self.assertNotIn(f"-n {prefix}", text, name)
            for path in re.findall(r'\$CELL/([A-Za-z0-9_./]+)', text):
                self.assertTrue((CELL / path).is_file(), f"{name} names $CELL/{path}")
        stage = (CELL / "stage_candidates.sbatch").read_text(encoding="utf-8")
        self.assertIn(f"#SBATCH --output={WORK}/logs/%x-%j.out", stage)
        self.assertIn("#SBATCH --job-name=mcs-score", stage)
        self.assertIn("#SBATCH --account=torch_pr_627_general", stage)
        self.assertIn(f'source "$WORK/source/research/hpc/{CELL.name}/cell.env"', stage)
        self.assertIn('SCORE=("$PY" -m research.minimal_env.candidate_sets score --adapter "$ADAPTER" --inputs "$INPUTS"', stage)
        self.assertIn('--top "$CS_TOP" --max-passes "$CS_MAX_PASSES"', stage)
        self.assertTrue((REPO / "research" / "minimal_env" / "candidate_sets.py").is_file())
        submit = (CELL / "submit_cell.sh").read_text(encoding="utf-8")
        self.assertIn('sbatch --parsable --job-name="mcs-$ARM" --export="ALL,ARM=$ARM" "$CELL/stage_candidates.sbatch"', submit)
        self.assertIn('not_queued "mcs-$ARM"', submit)

    def test_the_arms_name_models_that_exist_in_the_cells_they_come_from(self) -> None:
        settings = _settings(CELL / "cell.env")
        self.assertEqual(settings["PASS"], f"{SFT_WORK}/out/pass_base")
        self.assertEqual(settings["STEPS"], f"{RL_WORK}/out/steps")
        self.assertEqual(settings["ARMS"], '"base steps17"')
        environment = (CELL / "cell.env").read_text(encoding="utf-8")
        self.assertIn('base) echo "$PASS/training/lora" ;;', environment)
        self.assertIn('steps17) echo "$STEPS/iteration_002/adapter" ;;', environment)
        self.assertIn('arm_inputs() { echo "$DATA/candidate_inputs_dev_$1.jsonl"; }', environment)
        for name in ("status_cell.sh", "deploy_from_windows.sh"):
            self.assertIn("for arm in base steps17; do", (CELL / name).read_text(encoding="utf-8"), name)
        # The settings the scorer reads are the evaluations' own, and the search's two numbers are the module's defaults.
        from research.minimal_env import candidate_sets

        self.assertEqual(int(settings["CS_TOP"]), candidate_sets.DEFAULT_TOP)
        self.assertEqual(int(settings["CS_MAX_PASSES"]), candidate_sets.DEFAULT_MAX_PASSES)
        self.assertEqual((settings["CS_MAX_ACTIONS"], settings["CS_TABLE_VERSIONS"], settings["CS_MAX_INPUT_TOKENS"]), ("16", "2", "32768"))

    def test_the_cells_it_reads_are_never_written_to(self) -> None:
        for name in SCRIPTS:
            for line in (CELL / name).read_text(encoding="utf-8").splitlines():
                if line.lstrip().startswith("#"):
                    continue
                self.assertIsNone(re.search(
                    r"\b(rm|mkdir|mv|touch|tee|cat >|git -C)\b[^|;&]*(\$PASS|\$STEPS|\$SOURCE_CELL|minimal_sft_20261115|minimal_rl_20261122)", line),
                    f"{name}: {line}")
                self.assertIsNone(re.search(r">>?\s*\"?(\$PASS|\$STEPS|\$SOURCE_CELL)", line), f"{name}: {line}")
        deploy = (CELL / "deploy_from_windows.sh").read_text(encoding="utf-8")
        self.assertIn(f"SOURCE_CELL=${{SOURCE_CELL:-{RL_WORK}}}", deploy)
        self.assertIn("git clone -q $SOURCE_CELL/source $WORK/source", deploy)  # a source of its own, cloned

    def test_the_gpu_stage_takes_any_large_card_and_may_be_preempted(self) -> None:
        stage = (CELL / "stage_candidates.sbatch").read_text(encoding="utf-8")
        self.assertIn("#SBATCH --constraint=a100|h100|h200|rtx6000", stage)
        self.assertIn("#SBATCH --gres=gpu:1", stage)
        self.assertIn("#SBATCH --requeue", stage)
        self.assertIn("#SBATCH --comment=preemption=yes;requeue=true", stage)
        self.assertNotIn("--partition", stage)

    def test_the_deploy_script_submits_nothing_and_uploads_the_inputs_verified(self) -> None:
        deploy = (CELL / "deploy_from_windows.sh").read_text(encoding="utf-8")
        self.assertNotIn("sbatch ", deploy.replace("stage_candidates.sbatch", ""))
        listed = deploy[deploy.index("for path in"):deploy.index("; do", deploy.index("for path in"))]
        paths = re.findall(r'"\$CELL_REL/([^"]+)"', listed) + re.findall(r"\s((?:research|psse_env)/[A-Za-z0-9_./]+)", listed)
        self.assertGreaterEqual(len(paths), 12)
        for path in paths:
            target = CELL / path if not path.startswith(("research/", "psse_env/")) else REPO / path
            self.assertTrue(target.is_file(), path)
        self.assertIn('upload_verified "$INPUTS_LOCAL/candidate_inputs_dev_$arm.jsonl" "$WORK/data/candidate_inputs_dev_$arm.jsonl"', deploy)
        self.assertIn("the source is not moved under them", deploy)
        self.assertIn("grep -c '^mcs-'", deploy)
        # What is sent is the commit: a working tree that differs is named, and the deploy goes on.
        self.assertIn("the deploy sends the commit, not them", deploy)
        self.assertNotIn("commit the sources first", deploy)
        # The commit is the branch's tip or a commit below it (another session's later commit does not block the
        # deploy), and the source ends detached at the commit itself, bundle or not.
        self.assertIn('git merge-base --is-ancestor "$COMMIT" "$BRANCH"', deploy)
        self.assertIn('git diff --name-only "$COMMIT" -- research psse_env scripts tools', deploy)
        self.assertIn("checkout -q --detach $COMMIT", deploy)
        self.assertNotIn("FETCH_HEAD", deploy)
        self.assertEqual(deploy.count("\n  checkout_commit\n"), 2)
        for line in deploy.splitlines():
            if "wsl -- ssh" in line:
                self.assertNotIn("\\$(", line, line)
                self.assertNotIn("$USER", line, line)


#: A stand-in for the scorer, for the stage's control flow: it records every call and, on the call without a
#: shard, writes the summary.
STUB_SCORER = r'''
import json, os, sys
from pathlib import Path

args = sys.argv[1:]
assert args[0] == "score", args
out = Path(args[args.index("--output-dir") + 1])
out.mkdir(parents=True, exist_ok=True)
shard = args[args.index("--shard") + 1] if "--shard" in args else "close"
inputs = [args[index + 1] for index, value in enumerate(args) if value == "--inputs"]
with open(out / "calls.txt", "a") as stream:
    stream.write(json.dumps({"shard": shard, "inputs": inputs, "adapter": args[args.index("--adapter") + 1],
                             "threads": os.environ.get("OMP_NUM_THREADS"), "top": args[args.index("--top") + 1],
                             "passes": args[args.index("--max-passes") + 1],
                             "limit": args[args.index("--max-input-tokens") + 1]}) + "\n")
if shard == "close":
    (out / "summary.json").write_text(json.dumps({"by_split": {}}))
'''


class StageFlowTests(unittest.TestCase):
    """The stage's shell on a copy of the cell under a temporary work directory, with a stub in the scorer's place."""

    def setUp(self) -> None:
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.work = Path(self.folder.name) / "work"
        self.passed = Path(self.folder.name) / "pass_base"
        self.steps = Path(self.folder.name) / "steps"
        cell = self.work / "source" / "research" / "hpc" / CELL.name
        cell.mkdir(parents=True)
        stub = self.work / "source" / "research" / "minimal_env"
        stub.mkdir(parents=True)
        for package in (stub.parent, stub):
            (package / "__init__.py").write_text("", encoding="utf-8")
        (stub / "candidate_sets.py").write_text(STUB_SCORER, encoding="utf-8")
        environment = (CELL / "cell.env").read_text(encoding="utf-8")
        for old, new in ((f"WORK={WORK}", f"WORK={self.work.as_posix()}"), (f"PASS={SFT_WORK}/out/pass_base", f"PASS={self.passed.as_posix()}"),
                         (f"STEPS={RL_WORK}/out/steps", f"STEPS={self.steps.as_posix()}"),
                         (f"PY={_settings(CELL / 'cell.env')['PY']}", f"PY={Path(sys.executable).as_posix()}")):
            self.assertIn(old, environment)
            environment = environment.replace(old, new)
        (cell / "cell.env").write_text(environment, encoding="utf-8", newline="\n")
        stage = (CELL / "stage_candidates.sbatch").read_text(encoding="utf-8")
        self.assertEqual(stage.count(f"WORK={WORK}\n"), 1)
        (cell / "stage_candidates.sbatch").write_text(stage.replace(f"WORK={WORK}\n", f"WORK={self.work.as_posix()}\n"), encoding="utf-8", newline="\n")
        self.stage = cell / "stage_candidates.sbatch"
        self.bin = Path(self.folder.name) / "bin"
        self.bin.mkdir()
        (self.bin / "nvidia-smi").write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8", newline="\n")

    def _ready(self, arm: str = "base", *, adapter: bool = True, inputs: bool = True) -> None:
        if adapter:
            folder = self.passed / "training" / "lora" if arm == "base" else self.steps / "iteration_002" / "adapter"
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "adapter_config.json").write_text("{}", encoding="utf-8")
        if inputs:
            data = self.work / "data"
            data.mkdir(parents=True, exist_ok=True)
            (data / f"candidate_inputs_dev_{arm}.jsonl").write_text("{}\n", encoding="utf-8")

    def _stage(self, **environment: str) -> subprocess.CompletedProcess:
        assignments = " ".join(f'{name}="{value}"' for name, value in environment.items())
        script = (f'chmod +x "{self.bin.as_posix()}/nvidia-smi"; PATH="$(cd "{self.bin.as_posix()}" && pwd):$PATH" {assignments} '
                  f'bash "{self.stage.as_posix()}"')
        return subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=False)

    def _calls(self, arm: str = "base") -> list[dict]:
        path = self.work / "out" / arm / "calls.txt"
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.is_file() else []

    def test_two_workers_run_their_shards_and_the_closing_call_writes_the_summary(self) -> None:
        self._ready()
        done = self._stage(ARM="base", CS_WORKERS="2", SLURM_CPUS_PER_TASK="16")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        self.assertIn("mcs-score-complete", done.stdout)
        calls = self._calls()
        self.assertEqual(sorted(call["shard"] for call in calls[:2]), ["0/2", "1/2"])
        self.assertEqual(calls[-1]["shard"], "close")
        self.assertEqual({call["threads"] for call in calls[:2]}, {"8"})
        self.assertEqual(calls[-1]["threads"], "16")
        for call in calls:
            self.assertEqual([Path(path).name for path in call["inputs"]], ["candidate_inputs_dev_base.jsonl"])
            self.assertTrue(call["adapter"].endswith("pass_base/training/lora"))
            self.assertEqual((call["limit"], call["top"], call["passes"]), ("32768", "5", "96"))
        # A start that finds the summary does nothing.
        again = self._stage(ARM="base", CS_WORKERS="2")
        self.assertEqual(again.returncode, 0, again.stderr)
        self.assertIn("nothing to do", again.stdout)
        self.assertEqual(len(self._calls()), 3)

    def test_another_arm_reads_its_own_adapter_and_inputs_and_writes_beside_the_first(self) -> None:
        self._ready("steps17")
        done = self._stage(ARM="steps17", CS_WORKERS="1")
        self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
        calls = self._calls("steps17")
        self.assertEqual([call["shard"] for call in calls], ["close"])
        self.assertTrue(calls[0]["adapter"].endswith("steps/iteration_002/adapter"))
        self.assertEqual([Path(path).name for path in calls[0]["inputs"]], ["candidate_inputs_dev_steps17.jsonl"])
        self.assertTrue((self.work / "out" / "steps17" / "summary.json").is_file())
        self.assertEqual(self._calls("base"), [])

    def test_the_stage_needs_an_arm_its_adapter_and_its_inputs(self) -> None:
        self._ready()
        unnamed = self._stage(CS_WORKERS="1")
        self.assertNotEqual(unnamed.returncode, 0)
        self.assertIn("the arm to score", unnamed.stderr)
        unknown = self._stage(ARM="other", CS_WORKERS="1")
        self.assertNotEqual(unknown.returncode, 0)
        self.assertIn("unknown arm", unknown.stderr)
        self._ready("steps17", adapter=False)
        refused = self._stage(ARM="steps17", CS_WORKERS="1")
        self.assertEqual(refused.returncode, 2)
        self.assertIn("no adapter", refused.stderr)
        self._ready("steps17", inputs=False)
        (self.work / "data" / "candidate_inputs_dev_steps17.jsonl").unlink()
        refused = self._stage(ARM="steps17", CS_WORKERS="1")
        self.assertEqual(refused.returncode, 2)
        self.assertIn("no inputs", refused.stderr)
        self.assertEqual(self._calls("steps17"), [])


if __name__ == "__main__":
    unittest.main()
