"""The open-rules clone cell: its pass is S0's pass on the open-rules rows, its baselines are arm A's in the open
environment, and its manifest refuses views that are not the ones the selection and the row checks describe.

Run on its own (``python -m pytest research/hpc/clone_open_rules_20261112/test_cell.py``): it shares its file name
with the other cells' tests, so a run of the whole repository names it with ``--ignore``.
"""
from __future__ import annotations

import json
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

from psse_env.actions import OPEN_RULES, normalize_open_rules
from research.hpc.clone_open_rules_20261112 import baseline_config, rows_manifest
from research.hpc.rl_improvement_20261110 import make_configs

CELL = Path(__file__).resolve().parent
BOUNDED_CELL = CELL.parent / "compound_cloning_20261106"
SCRIPTS = ("cell.env", "stage_row_gate.sbatch", "stage_clone_train.sbatch", "stage_baseline.sbatch", "stage_arm.sbatch", "submit_cell.sh",
           "status_cell.sh", "deploy_from_windows.sh")


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
        # The start adapter of S0's pass (its cell's copy of R1), and S0 itself for its baseline.
        self.assertEqual(mine["START_ADAPTER"], "/scratch/yx3882/compound_cloning_20261106/data/start_adapter")
        self.assertEqual(mine["S0_ADAPTER"], "/scratch/yx3882/compound_cloning_20261106/out/clone/training/lora")
        self.assertNotEqual(mine["WORK"], bounded["WORK"])
        self.assertNotEqual(mine["CLONE_RUN_NAME"], bounded["CLONE_RUN_NAME"])
        stage = (CELL / "stage_clone_train.sbatch").read_text(encoding="utf-8")
        bounded_stage = (BOUNDED_CELL / "stage_clone_train.sbatch").read_text(encoding="utf-8")
        options = _trainer_options(stage)
        self.assertEqual(options, _trainer_options(bounded_stage))  # the trainer's options, one for one
        self.assertEqual(len(options), 23)
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
                self.assertRegex(text, r"#SBATCH --job-name=orc-", name)
                self.assertIn(f'source "$WORK/source/research/hpc/{CELL.name}/cell.env"', text, name)

    def test_gpu_stages_take_any_large_card_and_may_be_preempted(self) -> None:
        for name in ("stage_clone_train.sbatch", "stage_baseline.sbatch", "stage_arm.sbatch"):
            text = (CELL / name).read_text(encoding="utf-8")
            self.assertIn("#SBATCH --constraint=a100|h100|h200|rtx6000", text, name)
            self.assertIn("#SBATCH --requeue", text, name)
            self.assertIn("#SBATCH --comment=preemption=yes;requeue=true", text, name)
            self.assertNotIn("--partition", text, name)
        gate = (CELL / "stage_row_gate.sbatch").read_text(encoding="utf-8")
        self.assertNotIn("--gres", gate)  # the row gate needs no GPU
        self.assertIn("#SBATCH --partition=cs", gate)

    def test_the_submit_script_names_every_job_and_config(self) -> None:
        submit = (CELL / "submit_cell.sh").read_text(encoding="utf-8")
        for job in ("orc-row-gate", "orc-train", "orc-smoke", "orc-baseline-s0_open", "orc-baseline-clone_open", "orc-baseline-smoke_open"):
            self.assertIn(job, submit)
        for name in ("s0_open", "clone_open", "smoke_open"):
            self.assertIn(f"BASELINE_NAME={name}", submit)
            self.assertIn(name, (CELL / "stage_baseline.sbatch").read_text(encoding="utf-8"))
        self.assertEqual({path.name for path in (CELL / "configs").glob("*.json")}, set(baseline_config.FILES.values()))
        # The pass waits for the row gate, the clone's baseline for the pass.
        self.assertIn('"afterok:$gate"', submit)
        self.assertIn('"${after:+afterok:$after}"', submit)

    def test_the_arm_continues_the_clones_baseline_and_holds_the_series_stopping_rule(self) -> None:
        submit = (CELL / "submit_cell.sh").read_text(encoding="utf-8")
        stage = (CELL / "stage_arm.sbatch").read_text(encoding="utf-8")
        series = (CELL.parent / "rl_improvement_20261110" / "rl.sbatch").read_text(encoding="utf-8")
        # One config, the clone's baseline config: the arm's run folder is the one that baseline closed in.
        self.assertIn(f"ARM_CONFIG={baseline_config.FILES['clone_open']}", submit)
        self.assertIn(f"ARM_CONFIG=${{ARM_CONFIG:-{baseline_config.FILES['clone_open']}}}", stage)
        self.assertIn('"ALL,ARM_CONFIG=$ARM_CONFIG,ITERATIONS=$target,TARGET_ITERATIONS=$target"', submit)
        self.assertIn("local links=${1:-6} target=${2:-2}", submit)
        self.assertIn("if (( target > 3 )) && [[ -z \"${ALLOW_LONG:-}\" ]]; then", submit)
        for text in (submit, stage):
            self.assertIn("baseline/evaluation.json", text)  # no arm before its paired baseline
            self.assertIn('grep -q "def rejected_choice" "$SRC/psse_env/rl/train.py"', text)  # nor on a source that labels less
        self.assertIn("not_queued orc-arm", submit)
        # The baseline is never run by the arm's job; the other phases are the series' loop, line for line.
        self.assertIn("for phase in rollout continuation update evaluate; do", stage)
        self.assertNotIn("--phase baseline", stage)
        for line in ('"$PY" -m psse_env.rl.train --config "$CFG" --phase "$phase" --shard "$s/$RL_WORKERS" \\',
                     'if ! "$PY" -m psse_env.rl.train --config "$CFG" --phase "$phase"; then',
                     '(( ${gpu_mib:-0} >= 70000 )) && RL_WORKERS=2',
                     'reached_target() { [[ -n "$TARGET_ITERATIONS" ]] && (( $(status_field iteration) >= TARGET_ITERATIONS )); }'):
            self.assertIn(line, stage)
            self.assertIn(line, series)

    def test_no_remote_command_substitution_goes_through_the_wsl_shell(self) -> None:
        # ``wsl --`` runs its arguments through a shell of its own: a ``\\$(...)`` meant for torch inside an ssh command
        # string is expanded on the WSL side.  The row guard compared a checksum taken there, of a file that is not
        # there, and refused a redeploy with identical rows.  Such commands go in through stdin (a here-document).
        text = (CELL / "deploy_from_windows.sh").read_text(encoding="utf-8")
        for number, line in enumerate(text.splitlines(), start=1):
            if "wsl -- ssh" in line and "<<" not in line:
                self.assertNotIn("\\$(", line, f"deploy_from_windows.sh line {number}")
        self.assertIn('[[ "$(remote_sha "$WORK/data/rows/selection.train_view.jsonl")" == "$(sha256sum "$TRAIN_LOCAL" | cut -d\' \' -f1)" ]]', text)

    def test_the_deploy_script_hands_windows_python_no_posix_path(self) -> None:
        # Windows Python cannot open the /c/... paths Git Bash gives ($REPO, $MAIN, $SELECTION_DIR): a file is named
        # relative to the repo, or through winpath.  The arm's check of 2026-11-14 read its config by "$REPO/..." and
        # reported files present that it had not looked at.
        text = (CELL / "deploy_from_windows.sh").read_text(encoding="utf-8")
        for number, line in enumerate(text.splitlines(), start=1):
            if '"$PYTHON"' not in line:
                continue
            bare = line.replace('"$(winpath "$SELECTION_DIR")"', "").replace('"$(winpath "$MANIFEST_LOCAL")"', "")
            for name in ("$REPO", "$MAIN", "$SELECTION_DIR", "$MANIFEST_LOCAL"):
                self.assertNotIn(name, bare, f"deploy_from_windows.sh line {number}")
        self.assertIn('"$CELL_REL/configs/baseline_clone_open.json" "$1"; }', text)
        self.assertIn('[[ "$path" == /* ]] || { echo "the arm\'s config names no $field" >&2; exit 2; }', text)

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
        self.s0 = self.configs[baseline_config.FILES["s0_open"]]
        self.clone = self.configs[baseline_config.FILES["clone_open"]]
        self.smoke = self.configs[baseline_config.FILES["smoke_open"]]
        self.settings = _settings(CELL / "cell.env")

    def test_the_configs_on_disk_are_what_the_builder_gives(self) -> None:
        from psse_env.rl.train import RLConfig

        self.assertEqual(baseline_config.check(), [])
        for name in baseline_config.FILES.values():
            path = CELL / "configs" / name
            self.assertNotIn(b"\r", path.read_bytes(), name)
            RLConfig.load(path)

    def test_s0_open_is_arm_a_with_every_rule_lifted(self) -> None:
        self.assertEqual(make_configs.differing_fields(self.s0, self.arm_a), ["environment_options"])
        options = dict(self.s0["environment_options"])
        self.assertEqual(normalize_open_rules(options.pop("open_rules")), OPEN_RULES)
        self.assertEqual(options, self.arm_a["environment_options"])
        self.assertNotIn("open_rules", self.arm_a["environment_options"])
        # S0 itself, under the bounded history, as arm A's baseline measured it.
        self.assertEqual(self.s0["initial_adapter"], self.settings["S0_ADAPTER"])
        self.assertEqual(self.s0["initial_adapter"], self.arm_a["initial_adapter"])
        self.assertEqual(self.s0["observation_mode"], "bounded_history_v1")
        self.assertTrue(self.s0["baseline_evaluation"])
        self.assertEqual(self.s0["series"]["matched"], {"against": "rl_improvement_20261110 arm A", "differs_in": ["environment_options"]})
        self.assertEqual(self.s0["development_suite"], _expanded(self.settings, "DEVELOPMENT_SUITE"))

    def test_the_clone_and_the_smoke_differ_from_s0_open_in_one_field_each(self) -> None:
        work = self.settings["WORK"]
        self.assertEqual(make_configs.differing_fields(self.clone, self.s0), ["initial_adapter"])
        self.assertEqual(self.clone["initial_adapter"], f"{work}/out/clone/training/lora")
        self.assertEqual(self.clone["initial_adapter"], _expanded(self.settings, "CLONED_ADAPTER"))
        self.assertEqual(make_configs.differing_fields(self.smoke, self.s0), ["development_suite"])
        self.assertEqual(self.smoke["development_suite"], _expanded(self.settings, "SMOKE_SUITE"))
        self.assertNotIn("development_suite_sha256", self.smoke["series"]["planned_inputs"])
        outputs = [config["output_dir"] for config in self.configs.values()]
        self.assertEqual(len(set(outputs)), 3)
        self.assertTrue(all(output.startswith(work + "/out/") for output in outputs))

    def test_the_arms_config_trains_the_choices_the_open_rules_give(self) -> None:
        from psse_env.rl.train import RLConfig, initial_phase, iteration_phases, rejected_choice

        arm = RLConfig(**self.clone)
        arm.validate()
        # Arm A's update, rewards and root draws, from the cloned adapter with the rules lifted: nothing else differs.
        self.assertEqual(make_configs.differing_fields(self.clone, self.arm_a), ["environment_options", "initial_adapter"])
        for field in ("update", "reward", "rows", "root_sampling", "roots_per_iteration", "samples_per_root", "seed", "temperature",
                      "training_suite", "reserved_roots", "observation_mode", "max_steps", "decision_credit"):
            self.assertEqual(self.clone[field], self.arm_a[field], field)
        self.assertEqual((arm.reward["lifecycle_credit"], arm.reward["lifecycle_form"]), (0.25, "supervised"))
        self.assertEqual((initial_phase(arm), iteration_phases(arm)), ("baseline", ("rollout", "update", "evaluate")))
        # The commit rule is lifted, so the rollouts record and label the choice on a rejected candidate; arm A's do not.
        self.assertTrue(rejected_choice(arm))
        self.assertFalse(rejected_choice(RLConfig(**self.arm_a)))
        self.assertTrue(arm.stop_on_integrity_failure)

    def test_the_open_config_builds_an_environment_with_every_rule_lifted(self) -> None:
        import scripts.run_dagger_research as research
        from psse_env.rl.train import RLConfig, configure_research_environment

        snapshot = [(options, dict(options)) for options in (research.RESEARCH_ENVIRONMENT_OPTIONS, research.RESEARCH_EXPERT_OPTIONS)]
        try:
            closed = configure_research_environment(RLConfig(**self.arm_a))()
            opened = configure_research_environment(RLConfig(**self.s0))()
        finally:
            for options, original in snapshot:
                options.clear()
                options.update(original)
        self.assertEqual(tuple(opened.open_rules), OPEN_RULES)
        self.assertEqual(tuple(closed.open_rules), ())
        for env in (closed, opened):
            self.assertTrue(env.lifecycle_choice)
            self.assertTrue(env.detailed_evidence)
            self.assertFalse(env.route_comparison)
            self.assertEqual(env.evidence_profile, "scada_only")
            providers = env.wls_runner.__self__
            self.assertTrue(providers.flow_at_noise_cue)
            self.assertTrue(providers.pair_refit_adjacent)


def _selection(folder: Path, *, train: int = 3, validation: int = 2, report_rows: int | None = None, failures=None,
               carriage_return: bool = False, contract: str = rows_manifest.SELECTION_CONTRACT) -> Path:
    (folder / "checks").mkdir(parents=True, exist_ok=True)
    ending = "\r\n" if carriage_return else "\n"
    (folder / rows_manifest.TRAIN_VIEW).write_bytes("".join(json.dumps({"row": i}) + ending for i in range(train)).encode("utf-8"))
    (folder / rows_manifest.VALIDATION_VIEW).write_bytes("".join(json.dumps({"row": i}) + "\n" for i in range(validation)).encode("utf-8"))
    report = {"contract": contract, "rows_train": train if report_rows is None else report_rows, "rows_validation_view": validation,
              "truth_roots": "demonstration-fails", "seed": 20261106, "system": "ieee14", "thin": {"run_wls": 0.25},
              "reserved_roots": {"roots": 64, "left_out": ["r0_x"]}, "cases": {"demonstration | train": {"cases": 2}}}
    checks = {"counts": {"rows | train": train, "rows passing the message checks | train": train,
                         "rows | validation": validation, "rows passing the message checks | validation": validation},
              "failures": failures or {}, "cases": {"train": 2, "validation": 1}, "contrast_rows": {"train | x": 1},
              "prompt_characters": {"train": {"max": 10}}}
    (folder / rows_manifest.REPORT).write_text(json.dumps(report), encoding="utf-8")
    (folder / rows_manifest.CHECKS).write_text(json.dumps(checks), encoding="utf-8")
    return folder


class RowsManifestTests(unittest.TestCase):
    def test_a_manifest_names_the_views_and_the_selection_they_come_from(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            folder = _selection(Path(directory))
            manifest = rows_manifest.build(folder, commit="a" * 40)
        self.assertEqual(manifest["contract"], rows_manifest.MANIFEST_CONTRACT)
        self.assertEqual((manifest["views"]["train"]["rows"], manifest["views"]["validation"]["rows"]), (3, 2))
        self.assertEqual(len(manifest["views"]["train"]["sha256"]), 64)
        self.assertEqual(manifest["selection"]["reserved_roots"], {"roots": 64, "left_out": ["r0_x"]})
        self.assertEqual(manifest["selection"]["system"], "ieee14")
        self.assertEqual(manifest["source_commit"], "a" * 40)

    def test_views_that_are_not_the_ones_checked_are_refused(self) -> None:
        cases = {
            "a report of other rows": {"report_rows": 4},
            "a failure of the row checks": {"failures": {"GateError: x": 1}},
            "a carriage return": {"carriage_return": True},
            "another selection": {"contract": "compound_cloning_selection_v2"},
        }
        for label, overrides in cases.items():
            with self.subTest(label), tempfile.TemporaryDirectory() as directory:
                folder = _selection(Path(directory), **overrides)
                with self.assertRaises(rows_manifest.ManifestError):
                    rows_manifest.build(folder, commit="a" * 40)
                self.assertEqual(rows_manifest.main(["--selection-dir", str(folder), "--commit", "a" * 40,
                                                     "--output", str(Path(directory) / "manifest.json")]), 2)
        with tempfile.TemporaryDirectory() as directory:
            folder = _selection(Path(directory))
            (folder / rows_manifest.CHECKS).unlink()
            with self.assertRaises(rows_manifest.ManifestError):
                rows_manifest.build(folder, commit="a" * 40)

    def test_the_selection_on_disk_passes_when_it_is_here(self) -> None:
        try:
            common = subprocess.run(["git", "-C", str(CELL), "rev-parse", "--path-format=absolute", "--git-common-dir"],
                                    capture_output=True, text=True, timeout=60).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            common = ""
        folder = Path(common).parent / "output" / "compound_rl_20261106" / "open_rules_selection_ieee14" if common else None
        if folder is None or not (folder / rows_manifest.REPORT).is_file():
            self.skipTest("the selection's output is not on this machine (output/ is not in git)")
        manifest = rows_manifest.build(folder, commit="0" * 40)
        self.assertEqual(manifest["views"]["validation"]["rows"], 192)
        self.assertGreater(manifest["views"]["train"]["rows"], 10000)
        self.assertEqual(manifest["selection"]["truth_roots"], "demonstration-fails")
        self.assertTrue(manifest["selection"]["reserved_roots"]["file"])


if __name__ == "__main__":
    unittest.main()
