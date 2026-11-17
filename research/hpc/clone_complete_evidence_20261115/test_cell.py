"""The complete-evidence clone cell: its pass is S0's pass on the complete-evidence rows, its baseline is the
open-rules clone's with the estimator's output shown whole, and its manifest refuses rows that were not checked for
that view.

Run on its own (``python -m pytest research/hpc/clone_complete_evidence_20261115/test_cell.py``): it shares its file
name with the other cells' tests, so a run of the whole repository names it with ``--ignore``.
"""
from __future__ import annotations

import json
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

from psse_env.actions import COMPLETE_EVIDENCE_CONTRACT, OPEN_RULES, normalize_open_rules
from research.hpc.clone_complete_evidence_20261115 import baseline_config, rows_manifest
from research.hpc.rl_improvement_20261110 import make_configs

CELL = Path(__file__).resolve().parent
BOUNDED_CELL = CELL.parent / "compound_cloning_20261106"
OPEN_RULES_CELL = CELL.parent / "clone_open_rules_20261112"
SCRIPTS = ("cell.env", "stage_row_gate.sbatch", "stage_clone_train.sbatch", "stage_baseline.sbatch", "submit_cell.sh", "status_cell.sh",
           "deploy_from_windows.sh")


def _settings(path: Path) -> dict[str, str]:
    """``NAME=value`` lines of a shell settings file (no expansion)."""

    found = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^([A-Z_][A-Z0-9_]*)=(.*)$", line)
        if match:
            found[match.group(1)] = match.group(2)
    return found


def _expanded(settings: dict[str, str], name: str) -> str:
    """A setting with the ``$NAME`` of earlier settings put in."""

    value = settings[name]
    for _ in range(4):
        value = re.sub(r"\$([A-Z_][A-Z0-9_]*)", lambda match: settings.get(match.group(1), match.group(0)), value)
    return value


def _trainer_options(stage: str) -> list[str]:
    """The option lines a stage hands the trainer: between the command and the resume arguments."""

    lines = stage.splitlines()
    start = next(index for index, line in enumerate(lines) if line.strip() == '"${TRAIN_COMMAND[@]}" \\')
    end = next(index for index, line in enumerate(lines) if index > start and line.strip() == '"${resume_args[@]}"')
    return [line.strip().removesuffix("\\").strip() for line in lines[start + 1:end]]


class CellFilesTests(unittest.TestCase):
    def test_the_pass_is_the_bounded_pass_but_for_its_rows(self) -> None:
        mine, bounded = _settings(CELL / "cell.env"), _settings(BOUNDED_CELL / "cell.env")
        for name in ("CLONE_LEARNING_RATE", "CLONE_SAVE_STEPS", "CLONE_EVAL_STEPS", "CLONE_KEEP_NEWEST", "CLONE_SEED", "CLONE_TOOLS",
                     "TRAIN_MAX_LENGTH", "MODEL_CHOICE", "MODEL_ID", "PY", "HF_HOME", "SMOKE_TRAIN_ROWS", "SMOKE_VALIDATION_ROWS",
                     "SMOKE_EVAL_ROOTS"):
            self.assertEqual(mine[name], bounded[name], name)
        # The start adapter of S0's pass and of the open-rules clone's (the bounded cell's copy of R1): no adapter
        # trained since can read these rows.
        self.assertEqual(mine["START_ADAPTER"], "/scratch/yx3882/compound_cloning_20261106/data/start_adapter")
        self.assertEqual(mine["START_ADAPTER"], _settings(OPEN_RULES_CELL / "cell.env")["START_ADAPTER"])
        self.assertEqual(mine["S0_ADAPTER"], "/scratch/yx3882/compound_cloning_20261106/out/clone/training/lora")
        works = {mine["WORK"], bounded["WORK"], _settings(OPEN_RULES_CELL / "cell.env")["WORK"]}
        self.assertEqual(len(works), 3)
        self.assertNotEqual(mine["CLONE_RUN_NAME"], bounded["CLONE_RUN_NAME"])
        self.assertNotEqual(mine["CLONE_RUN_NAME"], _settings(OPEN_RULES_CELL / "cell.env")["CLONE_RUN_NAME"])
        stage = (CELL / "stage_clone_train.sbatch").read_text(encoding="utf-8")
        bounded_stage = (BOUNDED_CELL / "stage_clone_train.sbatch").read_text(encoding="utf-8")
        options = _trainer_options(stage)
        # The trainer's options, one for one, and one more: the loss from the logits of the target's span alone.  The
        # same loss; without it the 20,390-token row does not fit a 95 GB card (the first smoke, job 23274632).
        self.assertEqual([option for option in options if option != "--supervised-logits-only"], _trainer_options(bounded_stage))
        self.assertEqual(options.count("--supervised-logits-only"), 1)
        self.assertNotIn("--supervised-logits-only", _trainer_options(bounded_stage))
        self.assertEqual(len(options), 24)
        self.assertIn('--initial-adapter "$START_ADAPTER"', options)
        # The bounded cell's entry with the run-time tool lists, and its evidence-profile check of every row.
        self.assertIn('TRAIN_COMMAND=("$PY" "$BOUNDED_CELL/train_runtime_tools.py")', stage)
        self.assertIn('validate_training_evidence_profile(rows, evidence_profile="scada_only", source=path)', stage)
        self.assertIn("print(math.ceil(counts[0] / 4), counts[0])", stage)

    def test_the_scripts_name_this_cell_and_files_that_exist(self) -> None:
        work = _settings(CELL / "cell.env")["WORK"]
        for name in SCRIPTS:
            raw = (CELL / name).read_bytes()
            self.assertNotIn(b"\r", raw, name)
            text = raw.decode("utf-8")
            for path in re.findall(r'\$CELL/([A-Za-z0-9_./]+)', text) + re.findall(r'\$\{CELL\}/([A-Za-z0-9_./]+)', text):
                if path.endswith((".sbatch", ".py", ".env", ".sh", ".json")):
                    self.assertTrue((CELL / path).is_file(), f"{name} names $CELL/{path}")
            for path in re.findall(r'\$BOUNDED_CELL/([A-Za-z0-9_./]+)', text):
                self.assertTrue((BOUNDED_CELL / path).is_file(), f"{name} names $BOUNDED_CELL/{path}")
            # Only this cell's work directory is written to; the others are read in place.
            for found in re.findall(r"^WORK=(\S+)", text, flags=re.MULTILINE):
                self.assertEqual(found, work, name)
            if name.endswith(".sbatch"):
                self.assertIn(f"#SBATCH --output={work}/logs/%x-%j.out", text, name)
                self.assertRegex(text, r"#SBATCH --job-name=cec-", name)
                self.assertIn(f'source "$WORK/source/research/hpc/{CELL.name}/cell.env"', text, name)
            # No job, log or run of this cell takes a name of the open-rules clone's cell.
            self.assertNotIn("orc-", text, name)

    def test_gpu_stages_take_any_large_card_and_may_be_preempted(self) -> None:
        # The pass also takes a B200 (the user's word of 2026-11-17).  The baseline stays off it: with two workers
        # it runs at about 80% of a card, and the B200's low-utilization lines are 70% and 80%.
        cards = {"stage_clone_train.sbatch": "a100|h100|h200|rtx6000|b200", "stage_baseline.sbatch": "a100|h100|h200|rtx6000"}
        for name, constraint in cards.items():
            text = (CELL / name).read_text(encoding="utf-8")
            self.assertEqual(re.findall(r"^#SBATCH --constraint=(\S+)$", text, flags=re.MULTILINE), [constraint], name)
            self.assertIn("#SBATCH --requeue", text, name)
            self.assertIn("#SBATCH --comment=preemption=yes;requeue=true", text, name)
            self.assertNotIn("--partition", text, name)
        gate = (CELL / "stage_row_gate.sbatch").read_text(encoding="utf-8")
        self.assertNotIn("--gres", gate)  # the row gate needs no GPU
        self.assertIn("#SBATCH --partition=cs", gate)

    def test_the_submit_script_names_every_job_and_config(self) -> None:
        submit = (CELL / "submit_cell.sh").read_text(encoding="utf-8")
        baseline = (CELL / "stage_baseline.sbatch").read_text(encoding="utf-8")
        for job in ("cec-row-gate", "cec-train", "cec-smoke", "cec-baseline-clone_complete", "cec-baseline-smoke_complete"):
            self.assertIn(job, submit)
        for command in ("smoke)", "smoke_train)", "smoke_baseline)", "pass)", "clone)", "all)"):
            self.assertIn(f"  {command}", submit)
        for name in ("clone_complete", "smoke_complete"):
            self.assertIn(f"BASELINE_NAME={name}", submit)
            self.assertIn(name, baseline)
        self.assertEqual({path.name for path in (CELL / "configs").glob("*.json")}, set(baseline_config.FILES.values()))
        # The pass waits for the row gate, the clone's baseline for the pass.
        self.assertIn('"afterok:$gate"', submit)
        self.assertIn('"${after:+afterok:$after}"', submit)
        # No RL arm in this cell: an arm from this clone is a later submission.
        self.assertNotIn("stage_arm", submit)
        self.assertFalse((CELL / "stage_arm.sbatch").exists())

    def test_a_baseline_refuses_a_config_or_a_source_without_the_complete_evidence(self) -> None:
        baseline = (CELL / "stage_baseline.sbatch").read_text(encoding="utf-8")
        self.assertIn('[[ "$COMPLETE_EVIDENCE" == 1 ]] ||', baseline)
        self.assertIn('grep -q COMPLETE_EVIDENCE_CONTRACT "$SRC/psse_env/actions.py"', baseline)
        self.assertIn('[[ -s "$OUT/clone/training.done" ]] ||', baseline)
        # Two processes on one card only where it holds them: 69.2 GB at the smoke's peak, so not on an 80 GB card.
        self.assertIn("(( ${gpu_mib:-0} >= 90000 )) && RL_WORKERS=2", baseline)
        deploy = (CELL / "deploy_from_windows.sh").read_text(encoding="utf-8")
        for needed in ('git grep -q COMPLETE_EVIDENCE_CONTRACT "$COMMIT" -- psse_env/actions.py',
                       'git grep -q OPEN_RULE_CORRECTION_OUTSIDE_CONTEXT "$COMMIT" -- psse_env/transactional_env.py',
                       "research.hpc.clone_complete_evidence_20261115.rows_manifest",
                       "complete_evidence_selection_ieee14"):
            self.assertIn(needed, deploy)
        self.assertNotIn("open_rules_selection_ieee14", deploy)

    def test_no_remote_command_substitution_goes_through_the_wsl_shell(self) -> None:
        # ``wsl --`` runs its arguments through a shell of its own: a ``\\$(...)`` meant for torch inside an ssh command
        # string is expanded on the WSL side.  Such commands go in through stdin (a here-document).
        text = (CELL / "deploy_from_windows.sh").read_text(encoding="utf-8")
        for number, line in enumerate(text.splitlines(), start=1):
            if "wsl -- ssh" in line and "<<" not in line:
                self.assertNotIn("\\$(", line, f"deploy_from_windows.sh line {number}")
        self.assertIn('[[ "$(remote_sha "$WORK/data/rows/selection.train_view.jsonl")" == "$(sha256sum "$TRAIN_LOCAL" | cut -d\' \' -f1)" ]]', text)

    def test_the_deploy_script_hands_windows_python_no_posix_path(self) -> None:
        # Windows Python cannot open the /c/... paths Git Bash gives ($REPO, $MAIN, $SELECTION_DIR): a file is named
        # relative to the repo, or through winpath.
        text = (CELL / "deploy_from_windows.sh").read_text(encoding="utf-8")
        for number, line in enumerate(text.splitlines(), start=1):
            if '"$PYTHON"' not in line:
                continue
            bare = line.replace('"$(winpath "$SELECTION_DIR")"', "").replace('"$(winpath "$MANIFEST_LOCAL")"', "")
            for name in ("$REPO", "$MAIN", "$SELECTION_DIR", "$MANIFEST_LOCAL"):
                self.assertNotIn(name, bare, f"deploy_from_windows.sh line {number}")

    def test_the_shell_files_parse(self) -> None:
        try:
            done = subprocess.run(["bash", "--version"], capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
            self.skipTest("no bash here")
        if done.returncode != 0:
            self.skipTest("no usable bash here")
        for name in SCRIPTS:
            if name == "cell.env":
                continue
            result = subprocess.run(["bash", "-n", str(CELL / name)], capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, f"{name}: {result.stderr}")


class BaselineConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.arm_a = make_configs.arm_config(make_configs.load_series(), "A")
        self.configs = baseline_config.build()
        self.clone = self.configs[baseline_config.FILES["clone_complete"]]
        self.smoke = self.configs[baseline_config.FILES["smoke_complete"]]
        self.settings = _settings(CELL / "cell.env")
        self.open_rules_clone = json.loads((OPEN_RULES_CELL / "configs" / "baseline_clone_open.json").read_text(encoding="utf-8"))

    def test_the_configs_on_disk_are_what_the_builder_gives(self) -> None:
        from psse_env.rl.train import RLConfig

        self.assertEqual(baseline_config.check(), [])
        for name in baseline_config.FILES.values():
            path = CELL / "configs" / name
            self.assertNotIn(b"\r", path.read_bytes(), name)
            RLConfig.load(path)

    def test_the_clone_is_the_open_rules_clones_config_with_the_evidence_shown_whole(self) -> None:
        work = self.settings["WORK"]
        self.assertEqual(make_configs.differing_fields(self.clone, self.open_rules_clone), ["environment_options", "initial_adapter"])
        options = dict(self.clone["environment_options"])
        self.assertEqual(options.pop("complete_evidence"), 1)
        self.assertEqual(options, self.open_rules_clone["environment_options"])
        self.assertEqual(normalize_open_rules(options["open_rules"]), OPEN_RULES)
        self.assertEqual(self.clone["initial_adapter"], f"{work}/out/clone/training/lora")
        self.assertEqual(self.clone["initial_adapter"], _expanded(self.settings, "CLONED_ADAPTER"))
        self.assertEqual(self.clone["development_suite"], _expanded(self.settings, "DEVELOPMENT_SUITE"))
        self.assertEqual(self.clone["observation_mode"], "bounded_history_v1")
        self.assertTrue(self.clone["baseline_evaluation"])
        # The longest row is about 20,700 tokens by the local estimate; the driver's prompt limit holds it.
        self.assertGreaterEqual(self.clone["max_input_tokens"], 32768)
        # Arm A's update, rewards and root draws: an RL run from this clone would differ from arm A in these two fields.
        self.assertEqual(make_configs.differing_fields(self.clone, self.arm_a), ["environment_options", "initial_adapter"])

    def test_the_smoke_is_s0_on_eight_cases_in_the_same_environment(self) -> None:
        work = self.settings["WORK"]
        self.assertEqual(make_configs.differing_fields(self.smoke, self.clone), ["development_suite", "initial_adapter"])
        self.assertEqual(self.smoke["environment_options"], self.clone["environment_options"])
        self.assertEqual(self.smoke["initial_adapter"], self.settings["S0_ADAPTER"])
        self.assertEqual(self.smoke["development_suite"], _expanded(self.settings, "SMOKE_SUITE"))
        self.assertNotIn("development_suite_sha256", self.smoke["series"]["planned_inputs"])
        outputs = [config["output_dir"] for config in self.configs.values()]
        self.assertEqual(len(set(outputs)), 2)
        self.assertTrue(all(output.startswith(work + "/out/") for output in outputs))

    def test_the_config_builds_the_environment_the_rows_were_rendered_in(self) -> None:
        import scripts.run_dagger_research as research
        from psse_env.rl.train import RLConfig, configure_research_environment

        snapshot = [(options, dict(options)) for options in (research.RESEARCH_ENVIRONMENT_OPTIONS, research.RESEARCH_EXPERT_OPTIONS)]
        try:
            before = configure_research_environment(RLConfig(**self.open_rules_clone))()
            whole = configure_research_environment(RLConfig(**self.clone))()
        finally:
            for options, original in snapshot:
                options.clear()
                options.update(original)
        self.assertTrue(whole.complete_evidence)
        self.assertTrue(whole.wls_runner.__self__.complete_evidence)
        self.assertFalse(before.complete_evidence)
        self.assertFalse(before.wls_runner.__self__.complete_evidence)
        for env in (before, whole):
            self.assertEqual(tuple(env.open_rules), OPEN_RULES)
            self.assertTrue(env.lifecycle_choice)
            self.assertTrue(env.detailed_evidence)
            self.assertFalse(env.route_comparison)
            self.assertEqual(env.evidence_profile, "scada_only")
            providers = env.wls_runner.__self__
            self.assertTrue(providers.flow_at_noise_cue)
            self.assertTrue(providers.pair_refit_adjacent)


def _selection(folder: Path, *, train: int = 3, validation: int = 2, view: object = "complete", failures=None) -> Path:
    (folder / "checks").mkdir(parents=True, exist_ok=True)
    (folder / rows_manifest.TRAIN_VIEW).write_bytes("".join(json.dumps({"row": i}) + "\n" for i in range(train)).encode("utf-8"))
    (folder / rows_manifest.VALIDATION_VIEW).write_bytes("".join(json.dumps({"row": i}) + "\n" for i in range(validation)).encode("utf-8"))
    report = {"contract": rows_manifest.SELECTION_CONTRACT, "rows_train": train, "rows_validation_view": validation,
              "truth_roots": "demonstration-fails", "seed": 20261106, "system": "ieee14", "thin": {"run_wls": 0.25},
              "reserved_roots": {"roots": 64, "left_out": ["r0_x"]}, "cases": {"demonstration | train": {"cases": 2}}}
    checks = {"counts": {"rows | train": train, "rows passing the message checks | train": train,
                         "rows | validation": validation, "rows passing the message checks | validation": validation},
              "failures": failures or {}, "cases": {"train": 2, "validation": 1}, "contrast_rows": {"train | x": 1},
              "prompt_characters": {"train": {"max": 10}}}
    if view == "complete":
        checks["complete_evidence"] = {"contract": COMPLETE_EVIDENCE_CONTRACT, "line_comparison": "any lines"}
    elif view is not None:
        checks["complete_evidence"] = view
    (folder / rows_manifest.REPORT).write_text(json.dumps(report), encoding="utf-8")
    (folder / rows_manifest.CHECKS).write_text(json.dumps(checks), encoding="utf-8")
    return folder


class RowsManifestTests(unittest.TestCase):
    def test_a_manifest_names_the_views_and_the_view_they_were_checked_for(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = rows_manifest.build(_selection(Path(directory)), commit="a" * 40)
        self.assertEqual(manifest["contract"], rows_manifest.MANIFEST_CONTRACT)
        self.assertEqual(manifest["complete_evidence"]["contract"], COMPLETE_EVIDENCE_CONTRACT)
        self.assertEqual((manifest["views"]["train"]["rows"], manifest["views"]["validation"]["rows"]), (3, 2))
        self.assertEqual(manifest["selection"]["system"], "ieee14")
        self.assertEqual(manifest["source_commit"], "a" * 40)

    def test_rows_checked_for_another_view_are_refused(self) -> None:
        cases = {
            "the open-rules selection's checks": {"view": None},
            "another contract": {"view": {"contract": "something_else_v1"}},
            "a failure of the row checks": {"failures": {"GateError: x": 1}},
        }
        for label, overrides in cases.items():
            with self.subTest(label), tempfile.TemporaryDirectory() as directory:
                folder = _selection(Path(directory), **overrides)
                with self.assertRaises(rows_manifest.ManifestError):
                    rows_manifest.build(folder, commit="a" * 40)
                self.assertEqual(rows_manifest.main(["--selection-dir", str(folder), "--commit", "a" * 40,
                                                     "--output", str(Path(directory) / "manifest.json")]), 2)

    def _output(self, name: str) -> Path | None:
        try:
            common = subprocess.run(["git", "-C", str(CELL), "rev-parse", "--path-format=absolute", "--git-common-dir"],
                                    capture_output=True, text=True, timeout=60).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            common = ""
        folder = Path(common).parent / "output" / "compound_rl_20261106" / name if common else None
        return folder if folder is not None and (folder / rows_manifest.REPORT).is_file() else None

    def test_the_selection_on_disk_passes_when_it_is_here(self) -> None:
        folder = self._output("complete_evidence_selection_ieee14")
        if folder is None or not (folder / rows_manifest.CHECKS).is_file():
            self.skipTest("the selection's output is not on this machine (output/ is not in git)")
        manifest = rows_manifest.build(folder, commit="0" * 40)
        self.assertEqual(manifest["views"]["validation"]["rows"], 192)
        self.assertGreater(manifest["views"]["train"]["rows"], 11000)
        self.assertEqual(manifest["selection"]["truth_roots"], "demonstration-fails")
        self.assertEqual(manifest["complete_evidence"]["contract"], COMPLETE_EVIDENCE_CONTRACT)

    def test_the_open_rules_selection_on_disk_is_refused_when_it_is_here(self) -> None:
        folder = self._output("open_rules_selection_ieee14")
        if folder is None or not (folder / rows_manifest.CHECKS).is_file():
            self.skipTest("the open-rules selection's output is not on this machine (output/ is not in git)")
        with self.assertRaises(rows_manifest.ManifestError):
            rows_manifest.build(folder, commit="0" * 40)


if __name__ == "__main__":
    unittest.main()
