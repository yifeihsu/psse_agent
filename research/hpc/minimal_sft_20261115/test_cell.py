"""The cell of the minimal environment's supervised pass: its scripts parse and name files that exist, the pass is
the earlier cloning passes' but for its rows, its entry and its start, and the manifest refuses rows that are not the
selection's.

Run on its own (``python -m pytest research/hpc/minimal_sft_20261115/test_cell.py``): it shares its file name with
the other cells' tests, so a run of the whole repository names it with ``--ignore``.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

from psse_env.minimal import observation as view
from psse_env.minimal.env import DEFAULT_MAX_ACTIONS, DEFAULT_TABLE_VERSIONS
from psse_env.minimal.rows import MINIMAL_PROTOCOL, ROW_CONTRACT
from research.hpc.minimal_sft_20261115 import rows_manifest
from research.minimal_env.select_rows import SELECTION_CONTRACT

CELL = Path(__file__).resolve().parent
REPO = CELL.parents[2]
BOUNDED_CELL = CELL.parent / "compound_cloning_20261106"
FULL_PIPELINE = CELL.parent / "full_pipeline_20260907"
STAGES = ("stage_row_gate.sbatch", "stage_train.sbatch", "stage_eval.sbatch")
SCRIPTS = ("cell.env", *STAGES, "submit_cell.sh", "status_cell.sh", "deploy_from_windows.sh")


def _settings(path: Path) -> dict[str, str]:
    """``NAME=value`` lines of a shell settings file (no expansion)."""
    found = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^([A-Z_][A-Z0-9_]*)=(.*)$", line)
        if match:
            found[match.group(1)] = match.group(2)
    return found


def _trainer_options(stage: str) -> list[str]:
    """The option lines a stage hands the trainer: between the command and the resume arguments."""
    lines = stage.splitlines()
    start = next(index for index, line in enumerate(lines) if line.strip() == '"${TRAIN_COMMAND[@]}" \\')
    end = next(index for index, line in enumerate(lines) if index > start and line.strip() == '"${resume_args[@]}"')
    return [line.strip().removesuffix("\\").strip() for line in lines[start + 1:end]]


def _bash(script: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-c", script], cwd=CELL, capture_output=True, text=True, check=False)


class CellFilesTests(unittest.TestCase):
    def test_every_script_parses_and_holds_no_carriage_return(self) -> None:
        for name in SCRIPTS:
            self.assertNotIn(b"\r", (CELL / name).read_bytes(), name)
            checked = subprocess.run(["bash", "-n", name], cwd=CELL, capture_output=True, text=True, check=False)
            self.assertEqual(checked.returncode, 0, f"{name}: {checked.stderr}")
        self.assertNotIn(b"\r", (CELL / "dev300_reference.json").read_bytes())
        attributes = (CELL / ".gitattributes").read_text(encoding="utf-8")
        for pattern in ("*.json", "*.sh", "*.sbatch", "*.env"):
            self.assertIn(f"{pattern} text eol=lf", attributes)

    def test_the_scripts_name_this_cell_and_files_that_exist(self) -> None:
        work = _settings(CELL / "cell.env")["WORK"]
        self.assertEqual(work, "/scratch/yx3882/minimal_sft_20261115")
        for name in SCRIPTS:
            text = (CELL / name).read_text(encoding="utf-8")
            for path in re.findall(r'\$CELL/([A-Za-z0-9_./]+)', text):
                if path.endswith((".sbatch", ".py", ".env", ".sh", ".json")):
                    self.assertTrue((CELL / path).is_file(), f"{name} names $CELL/{path}")
            for path in re.findall(r'\$TOOLS/([A-Za-z0-9_./]+)', text):
                self.assertTrue((REPO / "research" / "minimal_env" / path).is_file(), f"{name} names $TOOLS/{path}")
            # Only this cell's work directory is written to.
            for found in re.findall(r"^WORK=(\S+)", text, flags=re.MULTILINE):
                self.assertEqual(found, work, name)
            if name.endswith(".sbatch"):
                self.assertIn(f"#SBATCH --output={work}/logs/%x-%j.out", text, name)
                self.assertRegex(text, r"#SBATCH --job-name=msf-", name)
                self.assertIn(f'source "$WORK/source/research/hpc/{CELL.name}/cell.env"', text, name)
                self.assertIn("#SBATCH --account=torch_pr_627_general", text, name)
            # No job, log or run of this cell takes a name of another cell.
            for prefix in ("cec-", "orc-", "cc-clone"):
                self.assertNotIn(prefix, text, name)
        works = {work, _settings(BOUNDED_CELL / "cell.env")["WORK"],
                 _settings(CELL.parent / "clone_complete_evidence_20261115" / "cell.env")["WORK"]}
        self.assertEqual(len(works), 3)

    def test_gpu_stages_take_any_large_card_and_may_be_preempted(self) -> None:
        for name in ("stage_train.sbatch", "stage_eval.sbatch"):
            text = (CELL / name).read_text(encoding="utf-8")
            self.assertIn("#SBATCH --constraint=a100|h100|h200|rtx6000", text, name)
            self.assertIn("#SBATCH --gres=gpu:1", text, name)
            self.assertIn("#SBATCH --requeue", text, name)
            self.assertIn("#SBATCH --comment=preemption=yes;requeue=true", text, name)
            self.assertNotIn("--partition", text, name)
        gate = (CELL / "stage_row_gate.sbatch").read_text(encoding="utf-8")
        self.assertNotIn("--gres", gate)  # the row gate needs no GPU
        self.assertIn("#SBATCH --partition=cs", gate)

    def test_the_stages_call_the_minimal_environments_own_entries(self) -> None:
        train = (CELL / "stage_train.sbatch").read_text(encoding="utf-8")
        self.assertIn('TRAIN_COMMAND=("$PY" "$TOOLS/train_sft.py")', train)
        self.assertIn('"$PY" "$TOOLS/row_gate.py" rows', train)
        self.assertIn("MINIMAL_PROTOCOL", train)  # every row is checked to be a row of this interface
        self.assertIn("print(math.ceil(counts[0] / 4), counts[0])", train)
        gate = (CELL / "stage_row_gate.sbatch").read_text(encoding="utf-8")
        self.assertIn('"$PY" "$TOOLS/row_gate.py" gate', gate)
        # One gate result serves the pass's every restart, whichever start it has.
        self.assertIn("export MINIMAL_GATE_CACHE=$OUT/row_gate_cache.pkl", gate)
        self.assertIn("export MINIMAL_GATE_CACHE=$OUT/row_gate_cache.pkl", train)
        evaluation = (CELL / "stage_eval.sbatch").read_text(encoding="utf-8")
        self.assertIn('"$PY" -m research.minimal_env.evaluate', evaluation)
        self.assertIn('--reference "$REFERENCE"', evaluation)
        self.assertIn('--shard "$s/$EVAL_WORKERS"', evaluation)
        self.assertNotIn("--load-in-4bit", evaluation)  # bf16, as every baseline of the RL series
        # No stage reaches into the old environment's driver or its trainer wrapper.
        for name in STAGES:
            text = (CELL / name).read_text(encoding="utf-8")
            self.assertNotIn("psse_env.rl.train", text, name)
            self.assertNotIn("train_runtime_tools", text, name)


class StartTests(unittest.TestCase):
    """Where the pass starts decides two trainer settings and where its output goes."""

    def _resolve(self, assignment: str) -> subprocess.CompletedProcess:
        return _bash(f'set -e; source ./cell.env; {assignment} resolve_start; '
                     'printf "%s|%s|%s\\n" "$START_NAME" "$START_LEARNING_RATE" "${START_ARGS[*]}"')

    def test_the_default_is_a_new_lora_on_the_base_model_at_bc0s_rate(self) -> None:
        settings = _settings(CELL / "cell.env")
        self.assertEqual(settings["SFT_START_DEFAULT"], "base")
        default = self._resolve("")
        self.assertEqual(default.returncode, 0, default.stderr)
        self.assertEqual(default.stdout.strip(), "base|1e-4|")
        self.assertEqual(self._resolve("SFT_START=base;").stdout.strip(), "base|1e-4|")
        # The one earlier pass from the base model, and the rate it trained at.
        self.assertEqual(settings["SFT_LEARNING_RATE_BASE"], _settings(FULL_PIPELINE / "pipeline.env")["BC0_LEARNING_RATE"])

    def test_r1_continues_the_adapter_at_the_cloning_passes_rate(self) -> None:
        settings = _settings(CELL / "cell.env")
        found = self._resolve("SFT_START=r1;")
        self.assertEqual(found.stdout.strip(), f"r1|3e-5|--initial-adapter {settings['R1_ADAPTER']}")
        # The adapter S0's pass and both clones started from, as the latest clone cell names it.
        self.assertEqual(settings["R1_ADAPTER"],
                         _settings(CELL.parent / "clone_complete_evidence_20261115" / "cell.env")["START_ADAPTER"])
        self.assertEqual(settings["SFT_LEARNING_RATE_CONTINUE"], _settings(BOUNDED_CELL / "cell.env")["CLONE_LEARNING_RATE"])

    def test_another_start_is_refused(self) -> None:
        refused = self._resolve("SFT_START=s0;")
        self.assertNotEqual(refused.returncode, 0)
        self.assertIn("SFT_START must be r1 or base", refused.stderr)

    def test_the_two_starts_keep_their_output_apart(self) -> None:
        train = (CELL / "stage_train.sbatch").read_text(encoding="utf-8")
        self.assertIn("STAGE_DIR=$OUT/pass_$START_NAME", train)
        self.assertIn("STAGE_DIR=$OUT/smoke_$START_NAME", train)
        evaluation = (CELL / "stage_eval.sbatch").read_text(encoding="utf-8")
        self.assertIn("ADAPTER=$OUT/pass_$START_NAME/training/lora", evaluation)
        self.assertIn("EVAL_DIR=$OUT/eval_dev300_$START_NAME", evaluation)
        self.assertIn('[[ -s "$OUT/pass_$START_NAME/training.done" ]]', evaluation)  # no evaluation of an unfinished pass


class TrainerSettingsTests(unittest.TestCase):
    def test_the_pass_is_the_cloning_pass_but_for_its_start(self) -> None:
        mine, bounded = _settings(CELL / "cell.env"), _settings(BOUNDED_CELL / "cell.env")
        for name, theirs in (("SFT_SAVE_STEPS", "CLONE_SAVE_STEPS"), ("SFT_EVAL_STEPS", "CLONE_EVAL_STEPS"),
                             ("SFT_KEEP_NEWEST", "CLONE_KEEP_NEWEST"), ("SFT_SEED", "CLONE_SEED"),
                             ("TRAIN_MAX_LENGTH", "TRAIN_MAX_LENGTH"), ("MODEL_CHOICE", "MODEL_CHOICE"), ("MODEL_ID", "MODEL_ID"),
                             ("PY", "PY"), ("HF_HOME", "HF_HOME"), ("SMOKE_TRAIN_ROWS", "SMOKE_TRAIN_ROWS"),
                             ("SMOKE_VALIDATION_ROWS", "SMOKE_VALIDATION_ROWS"), ("SMOKE_EVAL_ROOTS", "SMOKE_EVAL_ROOTS")):
            self.assertEqual(mine[name], bounded[theirs], name)
        self.assertNotEqual(mine["SFT_RUN_NAME"], bounded["CLONE_RUN_NAME"])
        options = _trainer_options((CELL / "stage_train.sbatch").read_text(encoding="utf-8"))
        theirs = _trainer_options((BOUNDED_CELL / "stage_clone_train.sbatch").read_text(encoding="utf-8"))
        renamed = {'--initial-adapter "$START_ADAPTER"': '"${START_ARGS[@]}"',
                   '--learning-rate "$CLONE_LEARNING_RATE"': '--learning-rate "$START_LEARNING_RATE"',
                   '--seed "$CLONE_SEED"': '--seed "$SFT_SEED"',
                   '--run-name "$CLONE_RUN_NAME-$MODE"': '--run-name "$SFT_RUN_NAME-$MODE-$START_NAME"'}
        # One option is this cell's own: the loss from the logits of the target's span (the same loss; the smoke of
        # 2026-11-15 ran out of a 95 GB card without it).  It follows the prompt-length option.
        span = "--supervised-logits-only"
        self.assertEqual(options[options.index("--strict-prompt-length") + 1], span)
        self.assertEqual(options.count(span), 1)
        options.remove(span)
        self.assertEqual(options, [renamed.get(option, option) for option in theirs])  # the rest, one for one
        self.assertEqual(len(options), 23)
        self.assertIn("--strict-prompt-length", options)  # a prompt over the limit is refused, never cut
        # The complete-evidence clone's pass, which met the same limit first, passes the same option.
        sibling = _trainer_options((CELL.parent / "clone_complete_evidence_20261115" / "stage_clone_train.sbatch").read_text(encoding="utf-8"))
        self.assertIn(span, sibling)
        from psse_env.sft.research_cli import parser
        self.assertTrue(parser().parse_args(["--train", "t", "--validation", "v", "--output-dir", "o", span]).supervised_logits_only)

    def test_the_evaluation_reads_the_prompt_the_rows_were_built_with(self) -> None:
        settings = _settings(CELL / "cell.env")
        self.assertEqual(int(settings["EVAL_MAX_ACTIONS"]), DEFAULT_MAX_ACTIONS)
        self.assertEqual(int(settings["EVAL_TABLE_VERSIONS"]), DEFAULT_TABLE_VERSIONS)
        environment = (CELL / "cell.env").read_text(encoding="utf-8")
        self.assertIn("export RESEARCH_MAX_INPUT_TOKENS=$TRAIN_MAX_LENGTH RESEARCH_MAX_NEW_TOKENS=256", environment)
        self.assertGreaterEqual(int(settings["TRAIN_MAX_LENGTH"]), 16384)


class SubmitAndDeployTests(unittest.TestCase):
    def test_the_submit_script_names_every_job_and_passes_the_start(self) -> None:
        submit = (CELL / "submit_cell.sh").read_text(encoding="utf-8")
        for job in ("msf-row-gate", '"msf-train-$START_NAME"', '"msf-smoke-$START_NAME"', '"msf-eval-dev300-$START_NAME"', "msf-eval-smoke"):
            self.assertIn(job, submit)
        self.assertIn('"ALL,SFT_MODE=full,SFT_START=$START_NAME"', submit)
        self.assertIn('"ALL,SFT_MODE=smoke,SFT_START=$START_NAME"', submit)
        self.assertIn('"ALL,EVAL_NAME=dev300,SFT_START=$START_NAME"', submit)
        self.assertIn("SFT_START=${2:-$SFT_START_DEFAULT}", submit)
        # The pass waits for the row gate; the evaluation for the pass's last link.
        self.assertIn('first="afterok:$gate"', submit)
        self.assertIn('"${after:+afterok:$after}"', submit)
        self.assertIn("afterany:$previous", submit)
        self.assertNotIn("--partition", submit)
        self.assertNotIn("--constraint", submit)  # the stages name the cards; nothing is pinned at submission
        for command in ("smoke)", "pass)", "eval)", "all)"):
            self.assertIn(command, submit)

    def test_the_link_count_covers_the_pass_at_a_minute_a_step(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            view_file = Path(folder) / "train.jsonl"
            view_file.write_text("x\n" * 14823, encoding="utf-8")
            submit = (CELL / "submit_cell.sh").read_text(encoding="utf-8")
            function = submit[submit.index("train_links() {"):submit.index("PASS_LAST=")]
            found = _bash(f'TRAIN_VIEW="{view_file.as_posix()}"; {function} train_links')
        self.assertEqual(found.returncode, 0, found.stderr)
        self.assertEqual(found.stdout.strip(), "7")  # 3,706 steps at 60 s, 11 h a link, one spare
        self.assertIn("14823 rows, 3706 optimizer steps", found.stderr)

    def test_the_deploy_script_submits_nothing_and_names_committed_files(self) -> None:
        deploy = (CELL / "deploy_from_windows.sh").read_text(encoding="utf-8")
        self.assertNotIn("sbatch ", deploy.replace("stage_row_gate.sbatch", "").replace("stage_train.sbatch", "").replace("stage_eval.sbatch", ""))
        listed = deploy[deploy.index("for path in"):deploy.index("; do", deploy.index("for path in"))]
        paths = re.findall(r'"\$CELL_REL/([^"]+)"', listed) + re.findall(r"\s((?:research|psse_env)/[A-Za-z0-9_./]+)", listed)
        self.assertGreaterEqual(len(paths), 12)
        for path in paths:
            target = CELL / path if not path.startswith(("research/", "psse_env/")) else REPO / path
            self.assertTrue(target.is_file(), path)
        # A command substitution meant for torch inside an ssh command string is expanded on the WSL side.
        for line in deploy.splitlines():
            if "wsl -- ssh" in line:
                self.assertNotIn("\\$(", line, line)
        self.assertIn("SELECTION_DIR=${SELECTION_DIR:-$REPO/output/minimal_env_20261115/selection_ieee14}", deploy)
        self.assertIn("research.hpc.minimal_sft_20261115.rows_manifest", deploy)
        self.assertIn("out/pass_*/training/checkpoint-*", deploy)  # rows are not replaced under a pass that has started

    def test_the_reference_is_the_review_s_split_of_the_300_cases(self) -> None:
        reference = json.loads((CELL / "dev300_reference.json").read_text(encoding="utf-8"))
        self.assertEqual(len(reference), 300)
        expert = sum(1 for entry in reference.values() if entry["expert"])
        truth_only = sum(1 for entry in reference.values() if not entry["expert"] and entry["truth"])
        neither = sum(1 for entry in reference.values() if not entry["expert"] and not entry["truth"])
        self.assertEqual((expert, truth_only, neither), (184, 63, 53))
        self.assertEqual(sum(1 for entry in reference.values() if entry["expert"] and not entry["truth"]), 0)


def _row(case: str, step: int, tool: str = "finish", arguments: dict | None = None) -> dict:
    observation = {"breaker_model": True}
    return {
        "example_id": f"{case}:success:{step:02d}", "scenario_id": case, "physical_root_fingerprint": f"root_{case}",
        "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
                     {"role": "assistant", "content": "", "tool_calls": [
                         {"id": "call_0", "type": "function", "function": {"name": tool, "arguments": arguments or {"outcome": "resolved"}}}]}],
        "tools": view.tools(observation), "metadata": {"protocol": MINIMAL_PROTOCOL, "row_contract": ROW_CONTRACT},
    }


class RowsManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.selection = Path(self.folder.name)
        self.train = [_row("a", 1, "undo", {}), _row("a", 2), _row("b", 1)]
        self.validation = [_row("c", 1)]
        self._write()

    def _write(self, *, report: dict | None = None, newline: str = "\n") -> None:
        digests = {}
        for name, rows in (("train", self.train), ("validation", self.validation)):
            data = "".join(json.dumps(row, sort_keys=True) + newline for row in rows).encode("utf-8")
            (self.selection / f"{name}.jsonl").write_bytes(data)
            digests[name] = hashlib.sha256(data).hexdigest()
        payload = {"contract": SELECTION_CONTRACT, "train_rows": len(self.train), "validation_rows": len(self.validation),
                   "sha256": digests, "cases": {"taken": 3}, "settings": {"seed": 1}}
        payload.update(report or {})
        (self.selection / "selection_report.json").write_text(json.dumps(payload), encoding="utf-8")

    def test_the_manifest_describes_the_two_views(self) -> None:
        manifest = rows_manifest.build(self.selection, commit="c" * 40)
        self.assertEqual(manifest["contract"], rows_manifest.MANIFEST_CONTRACT)
        self.assertEqual((manifest["views"]["train"]["rows"], manifest["views"]["train"]["cases"]), (3, 2))
        self.assertEqual(manifest["views"]["train"]["actions"], {"finish": 2, "undo": 1})
        self.assertEqual((manifest["views"]["validation"]["rows"], manifest["views"]["validation"]["cases"]), (1, 1))
        self.assertEqual(manifest["commit"], "c" * 40)
        self.assertEqual(rows_manifest.main(["--selection-dir", str(self.selection), "--commit", "c" * 40,
                                             "--output", str(self.selection / "manifest.json")]), 0)

    def test_views_that_are_not_the_selections_are_refused(self) -> None:
        cases = {
            "a row count that differs": lambda: self._write(report={"train_rows": 99}),
            "a checksum that differs": lambda: self._write(report={"sha256": {"train": "0" * 64, "validation": "0" * 64}}),
            "another selection's report": lambda: self._write(report={"contract": "open_rules_selection_v1"}),
            "a carriage return": lambda: self._write(newline="\r\n"),
        }
        for label, tamper in cases.items():
            self._write()
            tamper()
            with self.assertRaises(rows_manifest.ManifestError, msg=label):
                rows_manifest.build(self.selection, commit="c" * 40)
        self._write()
        (self.selection / "selection_report.json").unlink()
        with self.assertRaises(rows_manifest.ManifestError):
            rows_manifest.build(self.selection, commit="c" * 40)

    def test_rows_of_another_interface_or_a_shared_case_are_refused(self) -> None:
        old = _row("a", 1)
        old["metadata"]["protocol"] = "canonical"
        registry = _row("a", 1)
        registry["tools"] = registry["tools"][:5]
        unnamed = _row("a", 1)
        unnamed["physical_root_fingerprint"] = ""
        for label, row in (("the old protocol", old), ("not the six tools", registry), ("no case", unnamed)):
            self.train = [row]
            self._write()
            with self.assertRaises(rows_manifest.ManifestError, msg=label):
                rows_manifest.build(self.selection, commit="c" * 40)
        self.train, self.validation = [_row("a", 1)], [_row("a", 2)]
        self._write()
        with self.assertRaises(rows_manifest.ManifestError):
            rows_manifest.build(self.selection, commit="c" * 40)
        self.assertEqual(rows_manifest.main(["--selection-dir", str(self.selection), "--commit", "c" * 40,
                                             "--output", str(self.selection / "manifest.json")]), 2)


if __name__ == "__main__":
    unittest.main()
