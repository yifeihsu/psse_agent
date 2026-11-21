"""The minimal environment on freshly generated compound cases (IEEE 14, node/breaker model).

The cases are built here by the scenario generator, so nothing depends on
data outside the repository.  Four classes: a breaker, a line and a meter
together; two bus splits; meters only; a meter on a faulted line.
"""
from __future__ import annotations

import copy
import json
import unittest

import numpy as np

from mcp_server.matpower_server import _load_python_case
from psse_env.evidence_profile import sanitize_execution_for_profile
from psse_env.minimal import observation as view
from psse_env.minimal import truth
from psse_env.minimal.actions import (
    ERROR_ARGUMENTS,
    ERROR_NO_CHANGE,
    ERROR_UNKNOWN_OBJECT,
    ERROR_UNKNOWN_TOOL,
    ERROR_UNPARSEABLE,
    tool_schemas,
)
from psse_env.minimal.env import CONTRACT, OUTCOME_BUDGET, MinimalDiagnosisEnv
from psse_env.minimal.episode import ScriptedPolicy, run_episode
from psse_env.minimal.expert import TranslatedExpertPolicy, Untranslatable, translate_trace
from psse_env.minimal.kernel import DEFAULT_PROVIDER_OPTIONS, NumericalKernel
from psse_env.minimal.rows import MINIMAL_PROTOCOL, demonstration_rows, validate_row
from psse_env.minimal.scoring import RewardWeights, episode_reward, score_env, truth_targets
from psse_env.providers.matpower import MatpowerDeploymentProviders
from psse_env.providers.scenario_generator import DEFAULT_COMPOUND_CLASSES, Round0ScenarioGenerator
from psse_env.state_store import find_forbidden_policy_paths

GENERATOR_OPTIONS = dict(evidence_profile="scada_only", admission_mode="physical", normalized_residual_threshold=4.0,
                         min_measurement_error_sigma=10.0)
CLASSES = ("rx_cb_meter", "cb2_split", "control_meter", "meter_on_rx")
TOOL_LIMIT = "tool_limit"
ROOTS: dict[str, dict] = {}
KERNEL: NumericalKernel | None = None


def setUpModule() -> None:
    global KERNEL
    from research.compound_errors.build_suite import envelope_of

    generator = Round0ScenarioGenerator(seed=11, **GENERATOR_OPTIONS)
    KERNEL = NumericalKernel()
    for name in CLASSES:
        spec = next(item for item in DEFAULT_COMPOUND_CLASSES if item.name == name)
        # Some cases are beyond the correction tools (a line fit outside tolerance).  The first case the truth
        # repairs stands for its class; the first it cannot is kept as the tool-limit case.
        for row in generator._build_compound_class(spec, 4 if name == "rx_cb_meter" else 1):
            root = envelope_of(row)
            if name != "rx_cb_meter":
                ROOTS[name] = root
            elif truth.build_success(root, MinimalDiagnosisEnv(KERNEL))["solved"]:
                ROOTS.setdefault(name, root)
            else:
                ROOTS.setdefault(TOOL_LIMIT, root)
        assert name in ROOTS, (name, generator.skipped[-3:])


def _env(**options) -> MinimalDiagnosisEnv:
    return MinimalDiagnosisEnv(KERNEL, **options)


def _state(root: dict) -> dict:
    """The first state as the old providers take it."""
    execution = sanitize_execution_for_profile(root["execution"], "scada_only")
    return {"state_id": "episode:s0", "state_hash": "hash0", "status": "active", "case": execution["case"],
            "measurements": list(execution["measurements"]), "metadata": copy.deepcopy(execution["metadata"])}


def _healthy_meter(root: dict) -> int:
    true = set(truth_targets(root)["meters"])
    zero = set(root["execution"]["metadata"].get("structural_zero_indices") or [])
    return next(index for index in range(60, 100) if index not in true and index not in zero)


class KernelMatchesTheOldToolsTests(unittest.TestCase):
    """The same solves and the same tables as the old providers report, with their screens left out."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.old = MatpowerDeploymentProviders(**DEFAULT_PROVIDER_OPTIONS)

    def test_the_estimate_is_the_old_solve(self) -> None:
        for name, root in ROOTS.items():
            state = _state(root)
            estimate = KERNEL.estimate(state)
            old = self.old.run_wls(state)
            self.assertTrue(estimate["converged"], name)
            self.assertEqual(estimate["chi_square"], old["chi_square_statistic"], name)
            self.assertEqual(estimate["chi_square_threshold"], old["chi_square_threshold"], name)
            self.assertEqual(estimate["max_residual"], old["max_normalized_residual"], name)
            self.assertEqual(estimate["buses"], old["wls_summary"]["bus_table"], name)
            flows = {row["line"]: row for row in old["wls_summary"]["branch_table"]}
            for row in estimate["lines"]:
                for key in ("pf", "qf", "pt", "qt", "lr", "lx"):
                    self.assertEqual(row[key], flows[row["line"]][key], (name, row["line"], key))

    def test_the_meter_and_line_tables_are_the_old_contexts_tables(self) -> None:
        for name, root in ROOTS.items():
            state = _state(root)
            estimate = KERNEL.estimate(state)
            meters = self.old.get_measurement_context(state)["meter_table"]
            self.assertEqual([{key: value for key, value in row.items() if key != "sigma"} for row in estimate["meters"]], meters, name)
            lines = self.old.get_parameter_context(state)["line_table"]
            self.assertEqual([{key: row[key] for key in line} for row, line in zip(estimate["lines"], lines)], lines, name)

    def test_the_breaker_table_is_the_old_topology_contexts_table(self) -> None:
        state = _state(ROOTS["rx_cb_meter"])
        substation = KERNEL.substation(state)
        old = self.old.get_topology_context(state)
        self.assertEqual(substation["breakers"], old["breaker_table"])
        self.assertEqual(substation["chi_square"], old["node_breaker_estimate"]["chi_square"])
        self.assertEqual(substation["chi_square_threshold"], old["node_breaker_estimate"]["chi_square_threshold"])
        self.assertEqual(len(substation["breakers"]), 73)
        self.assertEqual(sum("flip_converged" in row for row in substation["breakers"]), self.old.breaker_candidate_count)

    def test_a_named_breaker_is_re_estimated_on_request(self) -> None:
        state = _state(ROOTS["rx_cb_meter"])
        untested = next(row["cb"] for row in KERNEL.substation(state)["breakers"] if "flip_converged" not in row)
        again = {row["cb"]: row for row in KERNEL.substation(state, [untested])["breakers"]}
        self.assertIn("flip_converged", again[untested])
        self.assertEqual(KERNEL.substation(state, ["no_such_breaker"])["error_code"], "unknown_breaker")

    def test_the_old_environment_and_the_new_one_end_in_the_same_state(self) -> None:
        """The truth's corrections through the old environment (every rule lifted) and through the new one."""
        from psse_env.transactional_env import TransactionalPSSEEnv
        from research.compound_errors.truth_sequences import TruthSequencePolicy, canonical_order
        from research.rl.decision_audit import truth_correction_actions

        root = ROOTS["rx_cb_meter"]
        old = TransactionalPSSEEnv(**self.old.env_kwargs(), production_dataset_mode=True, max_steps=40,
                                   lifecycle_choice=True, complete_evidence=True, open_rules=1)
        old.reset(copy.deepcopy(root["execution"]))
        # The same order in both: breaker, line, meters.
        policy = TruthSequencePolicy(canonical_order(truth_correction_actions(root, "S")))
        for _ in range(40):
            if old.terminal:
                break
            old.step(policy.act(old.get_policy_observation().as_dict()))
        self.assertTrue(old.terminal)
        self.assertEqual(policy.failed, [])
        old_final = old.store.get_state(str(old.store.active_state_id))

        env = _env()
        built = truth.run_order(env, root, truth.orders(truth.truth_corrections(root), 1)[0])
        new_final = env.final_state()
        self.assertEqual(new_final["measurements"], [float(value) for value in old_final["measurements"]])
        np.testing.assert_array_equal(_load_python_case(new_final["case"])["branch"], _load_python_case(old_final["case"])["branch"])
        self.assertEqual(new_final["metadata"]["reported_breaker_status"], old_final["metadata"]["reported_breaker_status"])
        # Four corrections' worth of work in the old interface: context, correction, solve, commit each.
        self.assertLess(len(built["steps"]), len(policy.kinds))


class ExecutionContractTests(unittest.TestCase):
    """What a call needs is a valid form and an existing object; nothing about the diagnosis."""

    def test_a_correction_needs_no_context_and_no_alarm(self) -> None:
        root = ROOTS["control_meter"]
        env = _env()
        env.reset(root)
        _, result = env.step({"tool": "correct_measurements", "arguments": {"meters": truth_targets(root)["meters"]}})
        self.assertEqual(result["status"], "ok")
        self.assertTrue(score_env(root, env)["estimator_passes"])
        # The estimator passes and the next correction, on a healthy meter, still runs: no rule closes the tool.
        _, result = env.step({"tool": "correct_measurements", "arguments": {"meters": [_healthy_meter(root)]}})
        self.assertEqual(result["status"], "ok")
        self.assertEqual(env.observation()["current_version"], 2)

    def test_any_existing_object_can_be_corrected(self) -> None:
        root = ROOTS["meter_on_rx"]
        true_line = truth_targets(root)["lines"][0]
        env = _env()
        env.reset(root)
        healthy = next(line for line in range(1, 21) if line != true_line)
        _, result = env.step({"tool": "correct_parameters", "arguments": {"line": healthy}})
        self.assertIn(result["status"], ("ok", "error"))
        if result["status"] == "error":  # the fit itself may fail; that is the tool, never a refusal of the target
            self.assertEqual(result["error_code"], "tool_failure")
            self.assertNotIn("None", str(result["error_detail"]))  # a fit with no reason to give names its code alone

    def test_a_refused_call_changes_nothing_and_still_counts(self) -> None:
        env = _env()
        env.reset(ROOTS["control_meter"])
        before = env.final_state()
        cases = [
            ("not json", ERROR_UNPARSEABLE),
            ({"tool": "commit_state", "arguments": {}}, ERROR_UNKNOWN_TOOL),
            ({"tool": "correct_parameters", "arguments": {"line": "6"}}, ERROR_ARGUMENTS),
            ({"tool": "correct_parameters", "arguments": {"line": 6, "value": 0.1}}, ERROR_ARGUMENTS),
            ({"tool": "correct_parameters", "arguments": {"line": 99}}, ERROR_UNKNOWN_OBJECT),
            ({"tool": "correct_measurements", "arguments": {"meters": [5000]}}, ERROR_UNKNOWN_OBJECT),
            ({"tool": "correct_topology", "arguments": {"breaker": "CB_none", "status": "open"}}, ERROR_UNKNOWN_OBJECT),
            ({"tool": "inspect", "arguments": {"view": "estimate", "version": 7}}, ERROR_UNKNOWN_OBJECT),
            ({"tool": "undo", "arguments": {}}, ERROR_NO_CHANGE),
        ]
        for number, (action, code) in enumerate(cases, start=1):
            observation, result = env.step(action)
            self.assertEqual((result["status"], result["error_code"]), ("error", code), action)
            self.assertEqual(observation["budget"]["used"], number)
        self.assertEqual(env.final_state(), before)
        self.assertEqual(len(env.observation()["versions"]), 1)

    def test_a_breaker_already_at_the_named_status_is_no_change(self) -> None:
        root = ROOTS["rx_cb_meter"]
        env = _env()
        env.reset(root)
        name = truth_targets(root)["breakers"][0]
        reported = root["execution"]["metadata"]["reported_breaker_status"][name]
        _, result = env.step({"tool": "correct_topology", "arguments": {"breaker": name, "status": reported}})
        self.assertEqual((result["status"], result["error_code"]), ("error", ERROR_NO_CHANGE))

    def test_undo_returns_to_any_saved_version(self) -> None:
        root = ROOTS["control_meter"]
        meters = truth_targets(root)["meters"]
        env = _env()
        first = env.reset(root)
        env.step({"tool": "correct_measurements", "arguments": {"meters": meters[:1]}})
        env.step({"tool": "correct_measurements", "arguments": {"meters": meters[1:2]}})
        observation, result = env.step({"tool": "undo", "arguments": {}})
        self.assertEqual((result["from_version"], result["to_version"]), (2, 1))
        self.assertEqual([item["in_effect"] for item in observation["modifications"]], [True, False])
        observation, result = env.step({"tool": "undo", "arguments": {"version": 0}})
        self.assertEqual(env.final_state()["measurements"], env.initial_state()["measurements"])
        self.assertEqual(observation["versions"][0]["estimate"], first["versions"][0]["estimate"])
        # A version left behind is still saved: returning to it needs no recomputation.
        solves = env.cost()["wls_solves"]
        observation, result = env.step({"tool": "undo", "arguments": {"version": 2}})
        self.assertEqual(result["status"], "ok")
        self.assertEqual(env.cost()["wls_solves"], solves)
        self.assertEqual([item["in_effect"] for item in observation["modifications"]], [True, True])

    def test_finish_ends_the_episode_whenever_it_is_called(self) -> None:
        env = _env()
        env.reset(ROOTS["control_meter"])
        _, result = env.step({"tool": "finish", "arguments": {"outcome": "unresolved"}})
        self.assertEqual((result["status"], env.done, env.outcome), ("ok", True, "unresolved"))
        with self.assertRaises(RuntimeError):
            env.step({"tool": "finish", "arguments": {"outcome": "resolved"}})

    def test_the_budget_ends_the_episode_without_a_declaration(self) -> None:
        root = ROOTS["control_meter"]
        env = _env(max_actions=2)
        env.reset(root)
        env.step({"tool": "inspect", "arguments": {"view": "estimate"}})
        self.assertFalse(env.done)
        env.step({"tool": "inspect", "arguments": {"view": "estimate"}})
        self.assertEqual((env.done, env.outcome), (True, OUTCOME_BUDGET))
        self.assertIsNone(score_env(root, env)["declared"])

    def test_a_state_that_does_not_converge_is_kept_and_can_be_undone(self) -> None:
        root = ROOTS["control_meter"]
        kernel = NumericalKernel(KERNEL.providers)
        solve = kernel.estimate
        calls = {"count": 0}

        def estimate(state):
            calls["count"] += 1
            return solve(state) if calls["count"] == 1 else {"converged": False, "error": "singular gain matrix"}

        kernel.estimate = estimate  # type: ignore[method-assign]
        env = MinimalDiagnosisEnv(kernel)
        env.reset(root)
        observation, result = env.step({"tool": "correct_measurements", "arguments": {"meters": truth_targets(root)["meters"]}})
        self.assertEqual(result["status"], "ok")
        self.assertFalse(observation["versions"][1]["estimate"]["converged"])
        self.assertIn("did not converge (singular gain matrix)", view.user_text(observation))
        _, result = env.step({"tool": "undo", "arguments": {}})
        self.assertEqual(result["to_version"], 0)

    def test_the_cost_of_an_episode_is_counted(self) -> None:
        root = ROOTS["rx_cb_meter"]
        env = _env()
        built = truth.run_order(env, root, truth.orders(truth.truth_corrections(root), 1)[0])
        cost = built["score"]["cost"]
        corrections = sum(1 for step in built["steps"] if step["kind"] == truth.KIND_CORRECTION)
        self.assertEqual(cost["actions"], len(built["steps"]))
        self.assertEqual(cost["wls_solves"], 1 + corrections)
        self.assertEqual(cost["substation_solves"], 1 + KERNEL.providers.breaker_candidate_count)
        self.assertEqual((cost["parameter_fits"], cost["measurement_repairs"]), (1, 1))
        self.assertGreater(cost["tool_seconds"], 0.0)


class TruthNeverReachesTheModelTests(unittest.TestCase):
    def test_a_different_truth_on_the_same_visible_data_gives_the_same_prompts(self) -> None:
        root = ROOTS["rx_cb_meter"]
        other = copy.deepcopy(root)
        donor = ROOTS["cb2_split"]
        other["audit"] = copy.deepcopy(donor["audit"])
        other["grouping"] = {**copy.deepcopy(donor["grouping"]), "case_id": root["grouping"]["case_id"]}
        actions = [{"tool": "inspect", "arguments": {"view": "breakers"}}] + truth.truth_corrections(root)
        prompts = []
        for scenario in (root, other):
            env = _env()
            observation = env.reset(scenario)
            seen = [json.dumps(view.messages(observation)), json.dumps(view.tools(observation), sort_keys=True)]
            for action in actions:
                observation, _ = env.step(copy.deepcopy(action))
                seen.append(json.dumps(view.messages(observation)))
            prompts.append(seen)
        self.assertEqual(prompts[0], prompts[1])

    def test_an_execution_part_that_carries_truth_is_refused(self) -> None:
        root = ROOTS["control_meter"]
        leaking = {**copy.deepcopy(root["execution"]), "true_measurement_errors": root["audit"]["truth"]["true_measurement_errors"]}
        with self.assertRaises(ValueError):
            _env().reset(leaking)
        partitioned = copy.deepcopy(root)
        partitioned["execution"]["metadata"]["clean_measurements"] = root["audit"]["truth"]["clean_measurements"]
        with self.assertRaises(ValueError):
            _env().reset(partitioned)

    def test_metadata_outside_the_evidence_profile_never_reaches_a_tool(self) -> None:
        root = ROOTS["control_meter"]
        plain = copy.deepcopy(root["execution"])
        noted = copy.deepcopy(root["execution"])
        noted["metadata"]["operator_note"] = "meters 19, 66 and 104 are bad"
        first, second = _env(), _env()
        self.assertEqual(view.messages(first.reset(plain)), view.messages(second.reset(noted)))
        self.assertNotIn("operator_note", second.final_state()["metadata"])

    def test_an_observation_holds_no_private_key(self) -> None:
        env = _env()
        built = truth.run_order(env, ROOTS["rx_cb_meter"], truth.orders(truth.truth_corrections(ROOTS["rx_cb_meter"]), 1)[0])
        for step in built["steps"]:
            self.assertEqual(find_forbidden_policy_paths(step["observation"]), [])
            text = view.user_text(step["observation"])
            for word in ("ACCEPT", "REJECT", "supported_corrections", "route_status", "valid_next_actions", "true_"):
                self.assertNotIn(word, text)


class ObservationTests(unittest.TestCase):
    def test_every_bus_line_and_meter_is_in_the_prompt(self) -> None:
        env = _env()
        observation = env.reset(ROOTS["control_meter"])
        self.assertEqual(observation["contract"], CONTRACT)
        estimate = observation["versions"][0]["estimate"]
        self.assertEqual((len(estimate["buses"]), len(estimate["lines"]), len(estimate["meters"])), (14, 20, 122))
        text = view.user_text(observation)
        lines = text.split("\n")
        self.assertEqual(lines[0], "Actions used: 0 of 16.")
        self.assertIn("== Version 0 (initial state; current) ==", lines)
        start = lines.index("meters") if "meters" in lines else next(i for i, row in enumerate(lines) if row.startswith("meters"))
        self.assertEqual(len(lines) - start - 2, 122)  # the heading and the column row, then one row per meter
        self.assertNotIn("breakers", lines)

    def test_the_breaker_table_appears_once_inspected(self) -> None:
        env = _env()
        env.reset(ROOTS["rx_cb_meter"])
        observation, result = env.step({"tool": "inspect", "arguments": {"view": "breakers"}})
        self.assertEqual(sorted(result["substation"]), ["chi_square", "chi_square_dof", "chi_square_threshold", "converged", "max_residual"])
        lines = view.user_text(observation).split("\n")
        self.assertEqual(len(lines) - lines.index("breakers") - 2, 73)
        self.assertTrue(lines[lines.index("breakers") - 1].startswith("Substation estimate: converged; chi-square "))

    def test_a_modification_reports_what_changed_and_the_numbers_on_both_sides(self) -> None:
        root = ROOTS["rx_cb_meter"]
        env = _env()
        env.reset(root)
        name = truth_targets(root)["breakers"][0]
        observation, _ = env.step({"tool": "correct_topology", "arguments": {"breaker": name, "status": "open"}})
        record = observation["modifications"][0]
        self.assertEqual((record["from_version"], record["to_version"], record["in_effect"]), (0, 1, True))
        kinds = {(change["object"], change.get("field")) for change in record["changes"]}
        self.assertEqual(kinds, {("breaker", None), ("line", "status")})
        text = view.user_text(observation)
        self.assertIn(f"changed: breaker {name} closed -> open; line", text)
        self.assertIn("estimate before: chi-square ", text)
        self.assertIn("== Version 1 (version 0 with modification 1; current) ==", text)

    def test_a_bus_split_is_reported_as_a_rebuilt_operator_model(self) -> None:
        root = ROOTS["cb2_split"]
        env = _env()
        env.reset(root)
        observation, result = env.step(truth.truth_corrections(root)[0])
        self.assertEqual(result["status"], "ok")
        rebuilt = [change for change in observation["modifications"][0]["changes"] if change["object"] == "operator_model"]
        self.assertEqual((rebuilt[0]["buses_before"], rebuilt[0]["buses_after"]), (14, 15))
        self.assertEqual(observation["versions"][1]["estimate"]["bus_count"], 15)

    def test_the_noise_is_a_legend_with_the_meters_that_differ(self) -> None:
        observation = _env().reset(ROOTS["control_meter"])
        lines = view.user_text(observation).split("\n")
        heading = next(row for row in lines if row.startswith("meters"))
        self.assertTrue(heading.startswith("meters (standard deviation: Vm 0.001; "), heading)
        self.assertIn("index,channel,at,value,residual", lines)
        meters = observation["versions"][0]["estimate"]["meters"]
        odd = sorted(row["index"] for row in meters if row["residual"] is not None and row["channel"] != "Vm" and row["sigma"] != 0.01)
        if odd:  # a bus whose injection is two meters added has a larger declared deviation
            self.assertIn("meters " + ", ".join(map(str, odd)) + " ", heading)

    def test_by_default_the_two_most_recent_versions_show_their_tables(self) -> None:
        root = ROOTS["control_meter"]
        meters = truth_targets(root)["meters"]
        env = _env()
        env.reset(root)
        env.step({"tool": "correct_measurements", "arguments": {"meters": meters[:1]}})
        observation, _ = env.step({"tool": "correct_measurements", "arguments": {"meters": meters[1:2]}})
        shown = ["meters" in entry["estimate"] for entry in observation["versions"]]
        self.assertEqual(shown, [False, True, True])
        text = view.user_text(observation)
        self.assertIn("== Version 0 (initial state) ==\nEstimate: converged; chi-square ", text)
        self.assertIn("(tables not shown)", text)
        # Every version's numbers stay, and so does every modification's before and after.
        self.assertEqual(text.count("estimate before: chi-square "), 2)
        everything = MinimalDiagnosisEnv(KERNEL, table_versions=None)
        everything.reset(root)
        everything.step({"tool": "correct_measurements", "arguments": {"meters": meters[:1]}})
        observation, _ = everything.step({"tool": "correct_measurements", "arguments": {"meters": meters[1:2]}})
        self.assertTrue(all("meters" in entry["estimate"] for entry in observation["versions"]))

    def test_the_table_window_hides_old_tables_and_inspect_brings_them_back(self) -> None:
        root = ROOTS["control_meter"]
        env = _env(table_versions=1)
        env.reset(root)
        observation, _ = env.step({"tool": "correct_measurements", "arguments": {"meters": truth_targets(root)["meters"]}})
        self.assertNotIn("meters", observation["versions"][0]["estimate"])
        self.assertIn("meters", observation["versions"][1]["estimate"])
        self.assertIn("chi_square", observation["versions"][0]["estimate"])  # the numbers of every version stay
        observation, _ = env.step({"tool": "inspect", "arguments": {"view": "estimate", "version": 0}})
        self.assertIn("meters", observation["versions"][0]["estimate"])
        self.assertNotIn("meters", observation["versions"][1]["estimate"])

    def test_the_system_prompt_names_no_rule_of_the_old_controller(self) -> None:
        env = _env()
        prompt = view.system_prompt(env.reset(ROOTS["control_meter"]))
        for word in ("commit", "rollback", "verif", "context", "candidate", "hand", "open rules", "lifted", "must", "never"):
            self.assertNotIn(word, prompt.lower(), word)
        self.assertIn("Tools return computed results; they do not decide whether your diagnosis is correct.", prompt)

    def test_the_tools_are_the_six_actions(self) -> None:
        names = [item["function"]["name"] for item in tool_schemas(breakers=True)]
        self.assertEqual(names, ["inspect", "correct_topology", "correct_parameters", "correct_measurements", "undo", "finish"])
        topology = tool_schemas(breakers=False)[1]["function"]["parameters"]
        self.assertEqual(sorted(topology["properties"]), ["line", "status"])


class ScoringTests(unittest.TestCase):
    def test_a_repaired_case_is_solved_by_both_audits(self) -> None:
        for name in ("rx_cb_meter", "cb2_split", "control_meter"):
            root = ROOTS[name]
            built = truth.build_success(root, _env())
            score = built["score"]
            self.assertTrue(score["solved"], (name, score["physical"]["errors"]))
            self.assertTrue(score["strict"]["solved"], (name, score["strict"]["reasons"]))
            self.assertEqual(score["declared"], "resolved")
            self.assertTrue(score["success"])
            self.assertEqual(score["physical"]["damaged"], 0)

    def test_a_case_beyond_the_tools_ends_unresolved(self) -> None:
        root = ROOTS.get(TOOL_LIMIT)
        if root is None:
            self.skipTest("the generator drew no case beyond the correction tools")
        built = truth.build_success(root, _env())
        self.assertFalse(built["solved"])
        self.assertFalse(built["score"]["strict"]["solved"])
        self.assertEqual(built["steps"][-1]["action"], {"tool": "finish", "arguments": {"outcome": "unresolved"}})
        self.assertEqual(built["score"]["physical"]["damaged"], 0)
        self.assertEqual(episode_reward(built["score"])["outcome"], 0.0)
        self.assertEqual(episode_reward(built["score"])["false_resolved"], 0.0)

    def test_an_untouched_case_is_unsolved_and_undamaged(self) -> None:
        root = ROOTS["rx_cb_meter"]
        episode = run_episode(_env(), root, ScriptedPolicy([], outcome="resolved"))
        score = episode["score"]
        self.assertEqual((score["physical"]["repaired"], score["physical"]["true_errors"], score["physical"]["damaged"]), (0, 3, 0))
        self.assertFalse(score["solved"])
        self.assertFalse(score["strict"]["solved"])
        self.assertEqual(episode["reward"]["false_resolved"], RewardWeights().false_resolved)

    def test_a_correction_kept_on_a_healthy_meter_is_damage(self) -> None:
        root = ROOTS["control_meter"]
        env = _env()
        actions = [{"tool": "correct_measurements", "arguments": {"meters": truth_targets(root)["meters"]}}]
        clean = run_episode(env, root, ScriptedPolicy(copy.deepcopy(actions)))["score"]
        self.assertTrue(clean["solved"])
        healthy = _healthy_meter(root)
        # Corrupt a healthy meter by repairing it against the faulted state, then repair the true ones.
        wrong = [{"tool": "correct_measurements", "arguments": {"meters": [healthy]}}] + actions
        score = run_episode(env, root, ScriptedPolicy(wrong))["score"]
        self.assertFalse(score["strict"]["solved"])
        self.assertIn("accepted_measurement_targets_outside_truth", score["strict"]["reasons"])
        if score["physical"]["damaged"]:
            self.assertEqual([item["meter"] for item in score["physical"]["damage"]], [healthy])
            self.assertFalse(score["solved"])
        # Undone, the same wrong correction leaves no trace in the final state.
        undone = [wrong[0], {"tool": "undo", "arguments": {}}] + actions
        self.assertTrue(run_episode(env, root, ScriptedPolicy(undone))["score"]["solved"])

    def test_the_reward_orders_the_outcomes(self) -> None:
        def reward(solved: bool, declared: str | None, damaged: int = 0, actions: int = 4) -> float:
            return episode_reward({"solved": solved, "declared": declared, "physical": {"damaged": damaged},
                                   "actions": actions, "tool_errors": 0, "cost": {}})["total"]

        solved = reward(True, "resolved")
        unclaimed = reward(True, "unresolved")
        honest = reward(False, "unresolved")
        false_claim = reward(False, "resolved")
        damaged = reward(False, "resolved", damaged=2)
        self.assertGreater(solved, unclaimed)
        self.assertGreater(unclaimed, honest)
        self.assertGreater(honest, false_claim)
        self.assertGreater(false_claim, damaged)
        # Solving with the whole budget still beats stopping at once.
        self.assertGreater(reward(True, "resolved", actions=16), reward(False, "unresolved", actions=1))
        with self.assertRaises(ValueError):
            RewardWeights.from_mapping({"lifecycle_credit": 0.25})


class DemonstrationTests(unittest.TestCase):
    def test_a_success_path_is_one_action_per_correction_then_finish(self) -> None:
        root = ROOTS["rx_cb_meter"]
        built = truth.build_success(root, _env())
        tools = [step["action"]["tool"] for step in built["steps"]]
        self.assertEqual(tools, ["inspect", "correct_topology", "correct_parameters", "correct_measurements", "finish"])
        self.assertEqual(built["steps"][-1]["action"]["arguments"], {"outcome": "resolved"})
        self.assertEqual(built["orders_tried"], 1)
        without = truth.build_success(root, _env(), opening_inspection=False)
        self.assertEqual([step["action"]["tool"] for step in without["steps"]][:2], ["inspect", "correct_topology"])
        meters_only = truth.build_success(ROOTS["control_meter"], _env(), opening_inspection=False)
        self.assertEqual([step["action"]["tool"] for step in meters_only["steps"]], ["correct_measurements", "finish"])

    def test_two_breakers_are_each_inspected_on_their_own_version(self) -> None:
        built = truth.build_success(ROOTS["cb2_split"], _env())
        self.assertEqual([step["action"]["tool"] for step in built["steps"]],
                         ["inspect", "correct_topology", "inspect", "correct_topology", "finish"])

    def test_every_version_acts_only_on_versions_whose_breakers_were_looked_at(self) -> None:
        for name in ("rx_cb_meter", "cb2_split", "control_meter"):
            built = truth.build_success(ROOTS[name], _env(), looks=truth.LOOK_EVERY_VERSION)
            self.assertTrue(built["solved"] and built["score"]["strict"]["solved"], name)
            self.assertEqual(built["skipped_looks"], 0)
            for step in built["steps"]:
                if step["action"]["tool"] != "inspect":
                    self.assertTrue(truth._breakers_shown(step["observation"]), (name, step["step"], step["action"]))
        tools = [step["action"]["tool"] for step in truth.build_success(ROOTS["rx_cb_meter"], _env(), looks=truth.LOOK_EVERY_VERSION)["steps"]]
        self.assertEqual(tools, ["inspect", "correct_topology", "inspect", "correct_parameters", "inspect", "correct_measurements",
                                 "inspect", "finish"])
        # Without the opening inspection the rule still looks at the first version before acting on it.
        meters = truth.build_success(ROOTS["control_meter"], _env(), opening_inspection=False, looks=truth.LOOK_EVERY_VERSION)
        self.assertEqual([step["action"]["tool"] for step in meters["steps"]], ["inspect", "correct_measurements", "inspect", "finish"])
        with self.assertRaises(ValueError):
            truth.build_success(ROOTS["control_meter"], _env(), looks="sometimes")

    def test_after_breakers_looks_after_each_breaker_and_before_finishing(self) -> None:
        tools = [step["action"]["tool"] for step in truth.build_success(ROOTS["rx_cb_meter"], _env(), looks=truth.LOOK_AFTER_BREAKERS)["steps"]]
        self.assertEqual(tools, ["inspect", "correct_topology", "inspect", "correct_parameters", "correct_measurements", "inspect", "finish"])
        tools = [step["action"]["tool"] for step in truth.build_success(ROOTS["cb2_split"], _env(), looks=truth.LOOK_AFTER_BREAKERS)["steps"]]
        self.assertEqual(tools, ["inspect", "correct_topology", "inspect", "correct_topology", "inspect", "finish"])

    def test_the_look_after_a_breaker_correction_is_no_longer_the_truths_decision(self) -> None:
        def after_breaker(looks: str) -> set[bool]:
            seen = set()
            for name in ("rx_cb_meter", "cb2_split"):
                steps = truth.build_success(ROOTS[name], _env(), looks=looks)["steps"]
                for index, step in enumerate(steps[:-1]):
                    if step["action"]["tool"] == "correct_topology":
                        seen.add(steps[index + 1]["action"]["tool"] == "inspect")
            return seen

        # The first build looks after a breaker correction only where another breaker correction follows.
        self.assertEqual(after_breaker(truth.LOOK_NEEDED), {True, False})
        self.assertEqual(after_breaker(truth.LOOK_EVERY_VERSION), {True})
        self.assertEqual(after_breaker(truth.LOOK_AFTER_BREAKERS), {True})

    def test_a_refusal_path_answers_the_refused_call_by_looking(self) -> None:
        root = ROOTS["cb2_split"]
        order = truth.orders(truth.truth_corrections(root), 1)[0]
        refused = {"tool": "correct_parameters", "arguments": {"line": 999}}  # no such line: the tool always refuses it
        built = truth.build_refusal(root, _env(), order, error=refused, looks=truth.LOOK_EVERY_VERSION)
        self.assertIsNotNone(built)
        kinds = [step["kind"] for step in built["steps"]]
        self.assertEqual(kinds, [truth.KIND_PREFIX, truth.KIND_PREFIX, truth.KIND_REFUSED, truth.KIND_INSPECT,
                                 truth.KIND_CORRECTION, truth.KIND_INSPECT, truth.KIND_FINISH])
        self.assertEqual(built["steps"][2]["result"]["error_code"], ERROR_UNKNOWN_OBJECT)
        self.assertTrue(built["solved"] and built["score"]["strict"]["solved"])
        rows = demonstration_rows(root, built, tag="refusal0")
        self.assertEqual([row["minimal"]["kind"] for row in rows], [truth.KIND_INSPECT, truth.KIND_CORRECTION, truth.KIND_INSPECT,
                                                                    truth.KIND_FINISH])
        self.assertIn('correct_parameters {"line":999} -> error', rows[0]["messages"][1]["content"])
        self.assertEqual((rows[0]["minimal"]["refused_call"], rows[0]["minimal"]["looks"]), (refused, truth.LOOK_EVERY_VERSION))
        # A call the tool carries out is a recovery path's matter, not a refusal.
        self.assertIsNone(truth.build_refusal(root, _env(), order, error=order[1], looks=truth.LOOK_EVERY_VERSION))

    def test_a_recovery_path_looks_at_the_erroneous_version_before_undoing_it(self) -> None:
        root = ROOTS["control_meter"]
        order = truth.orders(truth.truth_corrections(root), 1)[0]
        built = truth.build_recovery(root, _env(), order, after=0, family="correct_measurements", looks=truth.LOOK_EVERY_VERSION)
        self.assertIsNotNone(built)
        self.assertEqual([step["kind"] for step in built["steps"]],
                         [truth.KIND_PREFIX, truth.KIND_ERROR, truth.KIND_INSPECT, truth.KIND_UNDO, truth.KIND_CORRECTION,
                          truth.KIND_INSPECT, truth.KIND_FINISH])
        self.assertTrue(built["solved"])

    def test_a_breakers_last_path_corrects_the_breaker_after_the_others(self) -> None:
        cb2 = ROOTS["cb2_split"]
        self.assertIsNone(truth.build_breakers_last(cb2, _env(), truth.orders(truth.truth_corrections(cb2), 1)[0]))
        root = ROOTS["rx_cb_meter"]
        order = truth.orders(truth.truth_corrections(root), 1)[0]
        self.assertEqual([action["tool"] for action in truth.breakers_last(order)],
                         ["correct_parameters", "correct_measurements", "correct_topology"])
        built = truth.build_breakers_last(root, _env(), order, looks=truth.LOOK_EVERY_VERSION)
        self.assertIsNotNone(built)
        tools = [step["action"]["tool"] for step in built["steps"]]
        self.assertEqual(built["demonstration"], "breakers_last")
        breaker = tools.index("correct_topology")
        self.assertLess(tools.index("correct_parameters"), breaker)
        self.assertEqual(tools[breaker - 1], "inspect")
        # The line fitted while the breaker was wrong is fitted wrong; once the breaker is right it is fitted again.
        self.assertEqual([action["tool"] for action in built["refits"]], ["correct_parameters"])
        self.assertGreater(len(tools) - 1 - tools[::-1].index("correct_parameters"), breaker)
        self.assertTrue(built["score"]["strict"]["solved"])

    def test_the_first_builds_rule_writes_the_first_builds_rows(self) -> None:
        built = truth.build_success(ROOTS["rx_cb_meter"], _env())
        self.assertEqual((built["looks"], built["skipped_looks"]), (truth.LOOK_NEEDED, 0))
        self.assertNotIn("looks", demonstration_rows(ROOTS["rx_cb_meter"], built, tag="success")[0]["minimal"])

    def test_a_recovery_path_undoes_the_error_and_finishes_the_case(self) -> None:
        root = ROOTS["control_meter"]
        order = truth.orders(truth.truth_corrections(root), 1)[0]
        built = truth.build_recovery(root, _env(), order, after=0, family="correct_measurements")
        self.assertIsNotNone(built)
        kinds = [step["kind"] for step in built["steps"]]
        self.assertEqual(kinds, [truth.KIND_PREFIX, truth.KIND_ERROR, truth.KIND_UNDO, truth.KIND_CORRECTION, truth.KIND_FINISH])
        self.assertNotIn(built["injected_error"]["arguments"]["meters"][0], truth_targets(root)["meters"])
        self.assertTrue(built["solved"])
        rows = demonstration_rows(root, built, tag="recovery0")
        self.assertEqual([row["minimal"]["kind"] for row in rows], [truth.KIND_UNDO, truth.KIND_CORRECTION, truth.KIND_FINISH])
        # The undo row's prompt shows the erroneous correction as the latest modification, still in effect.
        self.assertIn("2. correct_measurements", rows[0]["messages"][1]["content"])
        self.assertIn("in effect", rows[0]["messages"][1]["content"])

    def test_an_error_after_everything_is_right_is_undone_before_finishing(self) -> None:
        root = ROOTS["control_meter"]
        order = truth.orders(truth.truth_corrections(root), 1)[0]
        built = truth.build_recovery(root, _env(), order, after=len(order), family="correct_measurements")
        self.assertIsNotNone(built)
        self.assertEqual([step["kind"] for step in built["steps"]][-3:], [truth.KIND_ERROR, truth.KIND_UNDO, truth.KIND_FINISH])
        self.assertTrue(built["solved"])

    def test_rows_are_the_runtime_prompt_and_pass_the_trainers_schema_gate(self) -> None:
        root = ROOTS["rx_cb_meter"]
        built = truth.build_success(root, _env())
        rows = demonstration_rows(root, built, tag="success")
        self.assertEqual(len(rows), len(built["steps"]))
        for row, step in zip(rows, built["steps"]):
            validate_row(row)
            self.assertEqual(row["messages"][:2], view.messages(step["observation"]))
            self.assertEqual(row["tools"], view.tools(step["observation"]))
            call = row["messages"][2]["tool_calls"][0]["function"]
            self.assertEqual({"tool": call["name"], "arguments": call["arguments"]}, step["action"])
            self.assertEqual(row["metadata"]["protocol"], MINIMAL_PROTOCOL)
            self.assertTrue(row["physical_root_fingerprint"])
            self.assertNotIn("true_", json.dumps(row["messages"]))


class ExpertTranslationTests(unittest.TestCase):
    def _old(self, tool: str, arguments: dict | None = None, status: str = "success") -> dict:
        return {"tool": tool, "arguments": arguments or {}, "status": status}

    def test_the_old_steps_map_onto_the_six_actions(self) -> None:
        trace = [
            self._old("run_wls"),
            self._old("get_topology_context"),
            self._old("correct_topology", {"cb_name": "CB_9_B1_N3", "status": 0, "line_index": 17}),
            self._old("run_wls"),
            self._old("commit_state"),
            self._old("get_parameter_context"),
            self._old("compare_parameter_candidates", {"line_indices": [6, 7]}),
            self._old("correct_parameters", {"line_index": 6}),
            self._old("run_wls"),
            self._old("rollback_state"),
            self._old("correct_parameters", {"line_index": 7}, status="failure"),
            self._old("correct_measurements", {"suspect_group": [61, 12]}),
            self._old("verify_candidate"),
            self._old("commit_state"),
            self._old("get_measurement_context"),
            self._old("ask_for_more_evidence", {"request": "operator_escalation:recovery_options_exhausted"}),
        ]
        actions = translate_trace(trace, breakers=True)
        self.assertEqual([(action["tool"], action["arguments"]) for action in actions], [
            ("inspect", {"view": "breakers"}),
            ("correct_topology", {"breaker": "CB_9_B1_N3", "status": "open"}),
            ("correct_parameters", {"line": 6}), ("undo", {}),
            ("correct_parameters", {"line": 7}), ("undo", {}),
            ("correct_parameters", {"line": 6}), ("undo", {}),
            ("correct_measurements", {"meters": [12, 61]}),
            ("finish", {}),
        ])
        self.assertEqual([action["tool"] for action in translate_trace(trace, breakers=False)][0], "correct_topology")

    def test_explicit_measurement_values_have_no_equivalent(self) -> None:
        with self.assertRaises(Untranslatable):
            translate_trace([self._old("correct_measurements", {"measurement_updates": {"4": 1.0}})], breakers=True)

    def test_a_candidate_the_old_episode_never_kept_is_undone(self) -> None:
        actions = translate_trace([self._old("correct_parameters", {"line_index": 3}), self._old("run_wls")], breakers=True)
        self.assertEqual([action["tool"] for action in actions], ["correct_parameters", "undo"])

    def test_the_replay_declares_by_the_final_estimate(self) -> None:
        root = ROOTS["control_meter"]
        trace = [self._old("correct_measurements", {"suspect_group": truth_targets(root)["meters"]}), self._old("commit_state"),
                 self._old("ask_for_more_evidence", {"request": "operator_escalation:recovery_options_exhausted"})]
        episode = run_episode(_env(), root, TranslatedExpertPolicy(translate_trace(trace, breakers=True)))
        self.assertEqual(episode["score"]["declared"], "resolved")
        self.assertTrue(episode["score"]["solved"])
        untouched = [self._old("ask_for_more_evidence", {"request": "operator_escalation:recovery_options_exhausted"})]
        episode = run_episode(_env(), root, TranslatedExpertPolicy(translate_trace(untouched, breakers=True)))
        self.assertEqual(episode["score"]["declared"], "unresolved")


class _RecordedModel:
    """A scripted policy that leaves the generation record the research model leaves, for the RL recorder."""

    def __init__(self, actions: list[dict]) -> None:
        self._actions = [copy.deepcopy(action) for action in actions]
        self.last_generation: dict | None = None
        self._calls = 0

    def act(self, observation: dict) -> dict:
        action = self._actions.pop(0) if self._actions else {"tool": "finish", "arguments": {"outcome": "resolved"}}
        self.last_generation = {
            "model_observation": copy.deepcopy(observation), "prompt_tokens": 10, "original_prompt_tokens": 10,
            "completion_token_ids": [1, 2, 3], "text": json.dumps(action), "call_index": self._calls,
            "completion_token_probabilities": [0.5, 0.5, 0.5], "prompt_token_digest": "digest",
        }
        self._calls += 1
        return action


class RolloutTests(unittest.TestCase):
    """Sampled episodes in the record the RL update trains on."""

    def _group(self):
        from psse_env.minimal.rollout import rollout_group

        root = ROOTS["control_meter"]
        repair = [{"tool": "correct_measurements", "arguments": {"meters": truth_targets(root)["meters"]}}]
        scripts = iter([repair, []])  # the first sample repairs the case; the second claims it without acting
        return root, rollout_group(root, samples=2, base_seed=7, env_factory=_env,
                                   policy_factory=lambda seed: _RecordedModel(next(scripts)))

    def test_a_group_is_the_policys_own_episodes_with_the_outcome_reward(self) -> None:
        root, group = self._group()
        self.assertEqual([trajectory.root for trajectory in group], [root["execution"]["scenario_id"]] * 2)
        self.assertTrue(all(trajectory.pairing_ok for trajectory in group))
        self.assertEqual([len(trajectory.steps) for trajectory in group], [2, 1])
        self.assertEqual([trajectory.evaluation["truth_audited_task_success"] for trajectory in group], [True, False])
        self.assertEqual([trajectory.lifecycle for trajectory in group], [[], []])
        solved, claimed = (trajectory.reward for trajectory in group)
        self.assertEqual((solved["outcome"], solved["false_resolved"]), (1.0, 0.0))
        self.assertEqual((claimed["outcome"], claimed["false_resolved"]), (0.0, 0.5))
        self.assertGreater(group[0].total_reward, group[1].total_reward)
        # Each step keeps the observation its prompt is rebuilt from at update time.
        step = group[0].steps[0]
        self.assertEqual(step.model_observation["contract"], CONTRACT)
        self.assertEqual(step.tool, "correct_measurements")

    def test_the_rl_row_builder_takes_the_group_as_it_is(self) -> None:
        from psse_env.rl.advantages import group_advantages
        from psse_env.rl.rows import build_rows

        _, group = self._group()
        advantages = group_advantages([trajectory.total_reward for trajectory in group], method="grpo", scale="none",
                                      root=group[0].root)
        self.assertTrue(advantages.informative)
        rendered: list[str] = []

        def render(observation: dict) -> str:
            rendered.append(view.user_text(observation))
            return rendered[-1]

        rows, report = build_rows([(group, advantages)], render_prompt=render, prompt_token_ids=lambda text: ([1] * 10, 10),
                                  max_length=100000)
        self.assertEqual((report["policy_rows"], len(rows)), (3, 3))
        self.assertEqual((report["prompt_length_mismatches"], report["trajectories_unpaired"], report["lifecycle_rows"]), (0, 0, 0))
        self.assertTrue(rendered[0].startswith("Actions used: 0 of 16."))

    def test_a_policy_whose_prompt_no_longer_fits_ends_the_episode_undeclared(self) -> None:
        from psse_env.minimal.episode import OUTCOME_PROMPT_LIMIT

        class PromptLimitError(RuntimeError):
            pass

        class Refusing:
            def act(self, observation: dict) -> dict:
                raise PromptLimitError("the prompt has 70000 tokens")

        root = ROOTS["control_meter"]
        episode = run_episode(_env(), root, Refusing())
        self.assertEqual(episode["outcome"], OUTCOME_PROMPT_LIMIT)
        self.assertIsNone(episode["score"]["declared"])
        self.assertEqual(episode["score"]["actions"], 0)


class EvaluationTests(unittest.TestCase):
    """The evaluation a cluster cell runs, with a scripted policy in the model's place."""

    def _policy(self, cases: dict) -> object:
        class Scripted:
            """Repairs a case by its own true corrections, read from the test's table by the case it is shown."""

            def __init__(self) -> None:
                self.queue: list[dict] = []
                self.last_action_metrics = {"prompt_tokens": 100, "original_prompt_tokens": 100, "generated_tokens": 9}

            def act(self, observation: dict) -> dict:
                if not observation["history"]:
                    meters = len(observation["versions"][0]["estimate"]["meters"])
                    first = observation["versions"][0]["estimate"]["chi_square"]
                    self.queue = copy.deepcopy(cases[(meters, round(first, 3))])
                return self.queue.pop(0) if self.queue else {"tool": "finish", "arguments": {"outcome": "resolved"}}

        return Scripted()

    def test_cases_are_stored_one_by_one_and_a_later_start_skips_them(self) -> None:
        import tempfile
        from pathlib import Path

        from research.minimal_env import evaluate

        names = ("control_meter", "meter_on_rx", "rx_cb_meter")
        roots = [ROOTS[name] for name in names]
        cases = {}
        for root in roots:
            first = _env().reset(root)["versions"][0]["estimate"]
            cases[(len(first["meters"]), round(first["chi_square"], 3))] = truth.orders(truth.truth_corrections(root), 1)[0]
        built: list[int] = []

        def policy_factory():
            built.append(1)
            return self._policy(cases)

        with tempfile.TemporaryDirectory() as folder:
            episodes = Path(folder) / "episodes"
            ran = evaluate.evaluate_cases(roots, policy_factory=policy_factory, env_factory=_env, episodes_dir=episodes, shard=(0, 2))
            self.assertEqual(ran, 2)  # the cases at positions 0 and 2
            self.assertEqual(len(list(episodes.glob("*.json"))), 2)
            records, missing = evaluate.read_episodes(roots, episodes)
            self.assertEqual(missing, [roots[1]["execution"]["scenario_id"]])
            self.assertEqual(evaluate.evaluate_cases(roots, policy_factory=policy_factory, env_factory=_env, episodes_dir=episodes), 1)
            # Everything is there: nothing runs and no policy is built.
            before = len(built)
            self.assertEqual(evaluate.evaluate_cases(roots, policy_factory=policy_factory, env_factory=_env, episodes_dir=episodes), 0)
            self.assertEqual(len(built), before)
            records, missing = evaluate.read_episodes(roots, episodes)
        self.assertEqual(missing, [])
        self.assertEqual([record["compound_class"] for record in records], list(names))
        self.assertTrue(all("observation" not in step for record in records for step in record["steps"]))
        self.assertEqual(records[0]["steps"][0]["policy_metrics"]["original_prompt_tokens"], 100)
        reference = {record["scenario_id"]: {"expert": index == 0, "truth": True} for index, record in enumerate(records)}
        summary = evaluate.summarize(records, reference)
        solved = sum(1 for record in records if record["score"]["solved"])
        self.assertEqual(summary["totals"]["cases"], 3)
        self.assertEqual(summary["totals"]["solved"], solved)
        self.assertGreaterEqual(solved, 2)  # the three-error case needs its breakers inspected by no rule: it is repaired too
        self.assertEqual(summary["by_reference"]["rule expert solves"]["cases"], 1)
        self.assertEqual(summary["by_reference"]["only the truth-built sequence repairs"]["cases"], 2)

    def test_an_unreadable_call_keeps_the_models_words(self) -> None:
        class Talker:
            def act(self, observation: dict) -> object:
                return "I think line 4 is wrong." if not observation["history"] else {"tool": "finish", "arguments": {"outcome": "unresolved"}}

        episode = run_episode(_env(), ROOTS["control_meter"], Talker())
        first = episode["steps"][0]
        self.assertEqual(first["result"]["error_code"], ERROR_UNPARSEABLE)
        self.assertEqual(first["raw_output"], "I think line 4 is wrong.")
        self.assertEqual(episode["score"]["tool_errors"], 1)


if __name__ == "__main__":
    unittest.main()
