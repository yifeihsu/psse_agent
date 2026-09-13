"""Real IEEE57 raw-section episodes through canonical calls and transactions."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from pypower.api import runpf, ppoption
from threadpoolctl import threadpool_limits
from mcp_server.matpower_server import _load_python_case
from psse_env.providers.matpower import _render_matpower_case

from logical_topology.inventory import build_inventory, process_topology
from logical_topology.measurements import build_measurement_inventory, expected_measurements, sample_measurements
from logical_topology.provider import LogicalTopologyProviders
from psse_env.state_store import find_forbidden_policy_paths
from psse_env.systems import resolve_system
from .logical_adapter import (LogicalCandidateQualityOracle, logical_environment_factory,
    logical_scenario, observable_logical_teacher, run_logical_episode)
from .logical_protocol import canonical_to_internal_action, internal_to_canonical_action, logical_tool_schemas


class LogicalAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.limit = threadpool_limits(limits=1)
        cls.addClassCleanup(cls.limit.restore_original_limits)
        cls.temp = tempfile.TemporaryDirectory(prefix="logical_adapter_")
        cls.addClassCleanup(cls.temp.cleanup)
        cls.case = resolve_system("case57").load_case()
        cls.inventory = build_inventory("case57")
        cls.branch = cls.inventory["branches"][0]["device_id"]
        cls.coupler = next(row["device_id"] for row in cls.inventory["couplers"] if row["base_bus"] == 4)
        cls.sensors = build_measurement_inventory(cls.inventory, "direct")
        cls.worlds = {}

    def scenario(self, *, error="split", scan_options=None, profile=None):
        true = dict(self.inventory["normal_statuses"])
        if error == "merge":
            true[self.coupler] = 0
        sensors = self.sensors if profile is None else build_measurement_inventory(self.inventory, profile)
        cache_key = (error == "merge", profile)
        if cache_key not in self.worlds:
            physical, success = runpf(process_topology(self.case, self.inventory, true)["case"],
                ppoption(VERBOSE=0, OUT_ALL=0, PF_TOL=1e-10))
            self.assertTrue(success)
            expected = expected_measurements(self.case, self.inventory, true, physical, sensors)
            self.worlds[cache_key] = sample_measurements(expected, sensors, noise=False)
        reported = dict(true)
        if error in ("split", "pair"):
            reported[self.coupler] = 0
        if error in ("branch", "pair"):
            reported[self.branch] = 0
        if error == "merge":
            reported[self.coupler] = 1
        options = scan_options if scan_options is not None else {"device_ids": [self.branch, self.coupler],
            "include_pairs": error == "pair"}
        if scan_options is None and error == "pair":
            options["pair_devices"] = [self.branch, self.coupler]
        env = logical_environment_factory(scan_options=options, derived_case_dir=self.temp.name)
        scenario = logical_scenario(env.logical_providers, case=self.case, inventory=self.inventory,
            reported_statuses=reported, sensors=sensors, observations=self.worlds[cache_key],
            scenario_id=f"logical_{error}", parent_id=f"physical_{cache_key}", true_statuses=true)
        return env, scenario

    def ready_candidate(self, error="pair"):
        env, scenario = self.scenario(error=error)
        env.reset(scenario)
        for _ in range(3):
            action = observable_logical_teacher(env.get_policy_observation())
            env.assert_training_decision_evidence(action)
            _, output = env.step_canonical(internal_to_canonical_action(action))
            self.assertEqual(output["execution_status"], "success", output)
        self.assertIsNotNone(env.current_candidate_id)
        return env, scenario

    def test_real_branch_split_merge_pair_and_healthy_episodes(self):
        for error in ("branch", "split", "merge", "pair", "healthy"):
            with self.subTest(error=error):
                env, scenario = self.scenario(error=error)
                result = run_logical_episode(env, scenario)
                self.assertTrue(result["private_audit"]["strict_resolved"], {
                    "audit": result["private_audit"], "steps": [(r["action"]["tool"], r["output"].get("error_code"), r["output"].get("error_detail")) for r in result["steps"]]})
                self.assertTrue(result["private_audit"]["healthy_component_preserved"])
                self.assertTrue(result["private_audit"]["measurements_and_covariance_preserved"])
                self.assertEqual(result["private_audit"]["raw_measurement_count"], 530)
                self.assertEqual(env.get_oracle_state().hidden_truth["remaining_true_fault_count"], 0)
                self.assertFalse(find_forbidden_policy_paths(result["steps"]))
                corrections = [step for step in result["steps"] if step["action"]["tool"] == "correct_logical_topology_from_context"]
                self.assertEqual(len(corrections), 0 if error == "healthy" else 1)
                if error == "pair":
                    self.assertEqual(set(corrections[0]["action"]["arguments"]["desired_statuses"]), {self.branch, self.coupler})
                    self.assertEqual(len(env.current_state()["accepted_corrections"]), 1)

    def test_pair_subset_hash_and_extra_key_tampering_cannot_create_candidate(self):
        env, scenario = self.scenario(error="pair")
        env.reset(scenario)
        for tool in ("run_wls", "get_topology_context"):
            _, result = env.step({"tool": tool, "arguments": {"state_id": env.store.active_state_id}})
            self.assertEqual(result["execution_status"], "success")
        action = observable_logical_teacher(env.get_policy_observation())
        variants = []
        subset = copy.deepcopy(action); subset["arguments"]["desired_statuses"].pop(self.branch); variants.append(subset)
        wrong_hash = copy.deepcopy(action); wrong_hash["arguments"]["certificate_hash"] = "0"*64; variants.append(wrong_hash)
        extra = copy.deepcopy(action); extra["arguments"]["cb_name"] = self.branch; variants.append(extra)
        for changed in variants:
            before = env.store.episode_hash()
            _, result = env.step(changed)
            self.assertEqual(result["execution_status"], "failure")
            self.assertEqual(env.store.episode_hash(), before)
            self.assertIsNone(env.current_candidate_id)
        self.assertEqual(env.logical_providers.correct_topology(env.store.get_state(env.store.active_state_id), subset)["execution_status"], "failure")

    def test_evidence_and_status_tampering_fail_transition_revalidation(self):
        env, _ = self.ready_candidate()
        candidate = env.store.get_state(env.current_candidate_id)
        parent = env.store.get_state(candidate["parent_state_id"])
        env._verify_transition(parent, candidate)
        for field in ("measurement", "covariance", "certificate", "status"):
            with self.subTest(field=field):
                bad = copy.deepcopy(candidate)
                if field == "measurement": bad["measurements"][0] += .01
                if field == "covariance": bad["metadata"]["logical_topology"]["measurement_inventory"]["covariance"][0][0] *= 2
                if field == "certificate": bad["metadata"]["logical_topology"]["last_identification_certificate"]["candidate_id"] = "bad"
                if field == "status": bad["metadata"]["logical_topology"]["current_statuses"][self.branch] = 0
                with self.assertRaises((ValueError, KeyError)):
                    env._verify_transition(parent, bad)

    def test_truncated_pairs_abstain_with_fixed_raw_evidence(self):
        env, scenario = self.scenario(error="pair", scan_options={"device_ids": [self.branch, self.coupler],
            "include_pairs": True, "pair_devices": [self.branch, self.coupler], "max_pairs": 0})
        result = run_logical_episode(env, scenario)
        self.assertEqual(result["private_audit"]["terminal_outcome"], "inconclusive", result)
        self.assertFalse(result["private_audit"]["strict_resolved"])
        self.assertEqual(result["private_audit"]["logical_changes"], {})
        self.assertTrue(result["private_audit"]["measurements_and_covariance_preserved"])

    def test_healthy_but_truncated_scope_abstains(self):
        env, scenario = self.scenario(error="healthy", scan_options={"device_ids": [self.branch, self.coupler],
            "include_pairs": True, "pair_devices": [self.branch, self.coupler], "max_pairs": 0})
        result = run_logical_episode(env, scenario)
        self.assertEqual(result["private_audit"]["terminal_outcome"], "inconclusive")
        self.assertTrue(result["private_audit"]["exact_statuses_recovered"])
        self.assertFalse(result["private_audit"]["strict_resolved"])
        self.assertTrue(all(row["output"]["execution_status"] == "success" for row in result["steps"]))

    def test_private_truth_changes_neither_observable_actions_nor_disposition(self):
        env, scenario = self.scenario(error="split")
        env.reset(scenario)
        clone = env.clone()
        clone._oracle_payload["logical_true_statuses"][self.coupler] = 0
        for _ in range(8):
            left = observable_logical_teacher(env.get_policy_observation())
            right = observable_logical_teacher(clone.get_policy_observation())
            self.assertEqual(left, right)
            _, a = env.step(left); _, b = clone.step(right)
            self.assertEqual(a["execution_status"], b["execution_status"])
            self.assertEqual(env.get_policy_observation().as_dict(), clone.get_policy_observation().as_dict())
            if env.terminal: break
        self.assertTrue(env.private_outcome_audit()["strict_resolved"])
        self.assertFalse(clone.private_outcome_audit()["strict_resolved"])

    def test_protocol_rejects_legacy_targets_and_nonbinary_statuses(self):
        names = {schema["function"]["name"] for schema in logical_tool_schemas()}
        self.assertIn("correct_logical_topology_from_context", names)
        self.assertNotIn("correct_topology_from_path", names)
        with self.assertRaises(ValueError):
            canonical_to_internal_action({"tool": "correct_topology_from_path", "arguments": {"case_path": "x", "line_index1": 1, "desired_status": True}})
        for status in (True, .5, "1", None):
            with self.assertRaises(ValueError):
                internal_to_canonical_action({"tool": "correct_topology", "arguments": {"state_id": "x", "candidate_id": "y", "certificate_hash": "a"*64, "desired_statuses": {self.coupler: status}}})
        with self.assertRaises(NotImplementedError):
            LogicalTopologyProviders().env_kwargs()

    def test_context_requires_explicit_active_wls_and_terminal_rechecks_binding(self):
        env, scenario = self.scenario(error="healthy")
        env.reset(scenario)
        context_action = {"tool": "get_topology_context", "arguments": {"state_id": env.store.active_state_id}}
        before = env.store.episode_hash()
        _, result = env.step(context_action)
        self.assertEqual(result["error_code"], "stale_logical_evidence")
        self.assertEqual(env.store.episode_hash(), before)
        for tool in ("run_wls", "get_topology_context"):
            _, result = env.step({"tool": tool, "arguments": {"state_id": env.store.active_state_id}})
            self.assertEqual(result["execution_status"], "success", result)
        env.logical_providers.scan_options.update(include_pairs=True, max_pairs=0)
        action = observable_logical_teacher(env.get_policy_observation())
        self.assertEqual(action["tool"], "finalize_diagnosis")
        with self.assertRaises(ValueError): env.assert_training_decision_evidence(action)
        _, result = env.step(action)
        self.assertEqual(result["error_code"], "stale_logical_evidence")
        self.assertFalse(env.terminal)

    def test_committed_state_requires_new_wls_before_confirmation_context(self):
        env, _ = self.ready_candidate(error="split")
        for tool, arguments in (("verify_candidate", {"state_id": env.current_candidate_id}),
                                ("commit_state", {"candidate_state_id": env.current_candidate_id})):
            _, result = env.step({"tool": tool, "arguments": arguments})
            self.assertEqual(result["execution_status"], "success", result)
        action = {"tool": "get_topology_context", "arguments": {"state_id": env.store.active_state_id}}
        with self.assertRaises(ValueError): env.assert_training_decision_evidence(action)
        _, result = env.step(action)
        self.assertEqual(result["error_code"], "stale_logical_evidence")

    def test_on_disk_candidate_tamper_is_not_overwritten_by_verification(self):
        env, _ = self.ready_candidate(error="split")
        candidate = env.store.get_state(env.current_candidate_id)
        path = Path(candidate["case"])
        original = path.read_bytes()
        changed_case = _load_python_case(str(path))
        changed_case["branch"][0, 2] *= 1.2
        path.write_text(_render_matpower_case(changed_case, "tampered_candidate"), encoding="utf-8")
        altered = path.read_bytes()
        try:
            parent = env.store.get_state(candidate["parent_state_id"])
            with self.assertRaises(ValueError): env._verify_transition(parent, candidate)
            self.assertEqual(path.read_bytes(), altered)
        finally:
            path.write_bytes(original)

    def test_ambiguous_context_policy_abstains_without_ranking_by_smallest_j(self):
        # Protocol unit: numerical ambiguity is independently exercised by
        # logical_topology/test_runtime.py; this tests the policy boundary.
        observation = {"active_state_id": "x:s0", "has_fresh_topology_context": True,
            "fresh_context_evidence": {"topology": {"state_id": "x:s0",
                "logical_decision": "ambiguous_candidate_set", "scope_complete": True,
                "supported_corrections": [], "plausible_candidate_count": 2}},
            "no_material_anomaly_remaining": False}
        self.assertEqual(observable_logical_teacher(observation)["tool"], "ask_for_more_evidence")


if __name__ == "__main__":
    unittest.main()
