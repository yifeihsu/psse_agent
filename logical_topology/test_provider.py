"""Actual logical provider/state-store chains, without a legacy truth oracle."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from pypower.api import runpf, ppoption
from threadpoolctl import threadpool_limits

from mcp_server.matpower_server import _load_python_case
from psse_env.state_store import PowerSystemStateStore
from psse_env.systems import resolve_system
from .inventory import build_inventory, process_topology
from .measurements import build_measurement_inventory, expected_measurements, sample_measurements
from .provider import LogicalTopologyProviders, _native


class LogicalProviderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.limit = threadpool_limits(limits=1)
        cls.addClassCleanup(cls.limit.restore_original_limits)
        cls.temp = tempfile.TemporaryDirectory(prefix="logical_provider_")
        cls.addClassCleanup(cls.temp.cleanup)
        cls.case = resolve_system("case57").load_case()
        cls.inventory = build_inventory("case57")
        cls.coupler = next(row["device_id"] for row in cls.inventory["couplers"] if row["base_bus"] == 4)

    def setup_state(self, *, profile="direct", changed_parameters=False, pinned=False, retained_outage=False):
        case = copy.deepcopy(self.case)
        if changed_parameters:
            case["branch"][0, 2] *= 1.1
            case["branch"][0, 11:13] = [-123.0, 144.0]
        true_statuses = dict(self.inventory["normal_statuses"])
        if retained_outage:
            true_statuses[self.inventory["branches"][18]["device_id"]] = 0
        physical, success = runpf(process_topology(case, self.inventory, true_statuses)["case"],
                                  ppoption(VERBOSE=0, OUT_ALL=0, PF_TOL=1e-10))
        self.assertTrue(success)
        sensors = build_measurement_inventory(self.inventory, profile)
        truth = expected_measurements(case, self.inventory, true_statuses, physical, sensors)
        observations = sample_measurements(truth, sensors, noise=False)
        reported = dict(true_statuses); reported[self.coupler] = 0
        provider = LogicalTopologyProviders(derived_case_dir=self.temp.name)
        inventory_arg, sensors_arg = self.inventory, sensors
        if pinned:
            def pin(name, value):
                path = Path(self.temp.name)/name
                raw = json.dumps(_native(value), sort_keys=True).encode()
                path.write_bytes(raw)
                return {"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest()}
            inventory_arg = pin("inventory.json", self.inventory)
            sensors_arg = pin("sensors.json", sensors)
        payload = provider.state_payload(case, inventory_arg, reported, sensors_arg, observations)
        if changed_parameters:
            # Simulate a previous parameter correction updating only the current
            # compiled case; the adapter must preserve its canonical row values.
            payload["metadata"]["logical_topology"]["base_case"] = _native(self.case)
        store = PowerSystemStateStore()
        state_id = store.create_root(**payload, episode_id="logical_provider_test")
        return provider, store, store.get_state(state_id), sensors

    def test_real_530_sensor_context_correct_candidate_verify_chain(self):
        provider, store, state, sensors = self.setup_state()
        before = copy.deepcopy(state)
        baseline = provider.run_wls(state)
        self.assertFalse(baseline["no_material_anomaly_remaining"])
        self.assertEqual(baseline["logical_estimation"]["raw_measurement_count"], 530)
        self.assertEqual(baseline["logical_estimation"]["state_dimension"], 139)
        self.assertEqual(baseline["chi_square_dof"], 391)
        context = provider.get_topology_context(state)
        self.assertEqual(context["logical_decision"], "unique_within_declared_scope", context)
        action = context["supported_corrections"][0]
        self.assertEqual(action["arguments"]["cb_name"], self.coupler)
        corrected = provider.correct_topology(state, action)
        self.assertNotIn("measurements", corrected["modification"])
        self.assertNotIn("measurement_updates", corrected["modification"])
        candidate_id = store.clone_candidate(state["state_id"], corrected["modification"], action)
        candidate = store.get_state(candidate_id)
        verification = provider.run_wls(candidate)
        self.assertTrue(verification["no_material_anomaly_remaining"])
        self.assertTrue(verification["target_fixed"])
        self.assertIsNone(verification["physical_constraints_ok"])
        self.assertEqual(candidate["measurements"], before["measurements"])
        self.assertEqual(candidate["metadata"]["logical_topology"]["measurement_inventory"], before["metadata"]["logical_topology"]["measurement_inventory"])
        self.assertEqual(candidate["metadata"]["logical_topology"]["current_statuses"], self.inventory["normal_statuses"])
        self.assertEqual(len(_load_python_case(state["case"])["bus"]), 58)
        self.assertEqual(len(_load_python_case(candidate["case"])["bus"]), 57)
        self.assertEqual(store.get_state(state["state_id"])["measurements"], before["measurements"])
        self.assertEqual(state, before)

    def test_uncertified_and_stale_actions_are_rejected(self):
        provider, _, state, _ = self.setup_state()
        candidate = provider.test_cb(state, self.coupler, 1)["candidate"]
        self.assertTrue(candidate["plausible"])
        action = {"tool": "correct_topology", "arguments": {"state_id": state["state_id"], "cb_name": self.coupler, "status": 1}}
        self.assertEqual(provider.correct_topology(state, action)["execution_status"], "failure")
        context = provider.get_topology_context(state)
        action = context["supported_corrections"][0]
        changed = copy.deepcopy(state)
        changed["measurements"][0] += .01
        # Even a falsely reused state_hash cannot reuse the old evidence proof.
        self.assertEqual(provider.correct_topology(changed, action)["execution_status"], "failure")
        changed_limits = copy.deepcopy(state)
        changed_limits["case"] = _load_python_case(state["case"])
        changed_limits["case"]["branch"][0, 11] = -123.0
        self.assertEqual(provider.correct_topology(changed_limits, action)["execution_status"], "failure")
        wrong = copy.deepcopy(action); wrong["arguments"]["state_id"] = "other:s0"
        self.assertEqual(provider.correct_topology(state, wrong)["execution_status"], "failure")

    def test_preexisting_parameter_edit_survives_coupler_correction(self):
        provider, store, state, _ = self.setup_state(changed_parameters=True, retained_outage=True)
        original = _load_python_case(state["case"])
        context = provider.get_topology_context(state)
        action = context["supported_corrections"][0]
        correction = provider.correct_topology(state, action)
        candidate = store.get_state(store.clone_candidate(state["state_id"], correction["modification"], action))
        expected = self.case["branch"][0, 2]*1.1
        self.assertAlmostEqual(_load_python_case(candidate["case"])["branch"][0, 2], expected)
        self.assertAlmostEqual(candidate["metadata"]["logical_topology"]["base_case"]["branch"][0][2], expected)
        parameter_columns = [*range(2, 10), *range(11, original["branch"].shape[1])]
        np.testing.assert_array_equal(_load_python_case(candidate["case"])["branch"][:, parameter_columns],
                                      original["branch"][:, parameter_columns])
        np.testing.assert_array_equal(np.asarray(candidate["metadata"]["logical_topology"]["base_case"]["branch"])[:, parameter_columns],
                                      original["branch"][:, parameter_columns])
        self.assertEqual(candidate["metadata"]["logical_topology"]["current_statuses"][self.inventory["branches"][18]["device_id"]], 0)
        self.assertEqual(_load_python_case(candidate["case"])["branch"][18, 10], 0)
        self.assertTrue(provider.run_wls(candidate)["no_material_anomaly_remaining"])

    def test_unmapped_current_bus_generator_and_base_edits_are_rejected(self):
        provider, _, state, _ = self.setup_state()
        for matrix, row, column in (("bus", 0, 11), ("bus", 57, 7), ("gen", 0, 6), ("gen", 0, 8)):
            with self.subTest(matrix=matrix, row=row, column=column):
                changed = copy.deepcopy(state)
                changed["case"] = _load_python_case(state["case"])
                changed["case"][matrix][row, column] += .01
                result = provider.run_wls(changed)
                self.assertEqual(result["execution_status"], "failure")
                self.assertIn("canonical base-case metadata", result["error_detail"])
        for field in ("baseMVA", "gencost"):
            with self.subTest(field=field):
                changed = copy.deepcopy(state)
                changed["case"] = _load_python_case(state["case"])
                if field == "baseMVA":
                    changed["case"][field] += 1.0
                else:
                    changed["case"][field] = self.case[field].copy()
                    changed["case"][field][0, -1] += 1.0
                result = provider.run_wls(changed)
                self.assertEqual(result["execution_status"], "failure")
                self.assertIn(f"current {field} differs", result["error_detail"])

    def test_pinned_json_inputs_are_checked_on_every_call(self):
        provider, _, state, _ = self.setup_state(pinned=True)
        self.assertTrue(provider.run_wls(state)["converged"])
        ref = state["metadata"]["logical_topology"]["measurement_inventory"]
        path = Path(ref["path"])
        path.write_bytes(path.read_bytes()+b" ")
        result = provider.run_wls(state)
        self.assertEqual(result["execution_status"], "failure")
        self.assertIn("hash changed", result["error_detail"])

    def test_masked_truth_is_absent_and_legacy_routes_cannot_handle_logical_state(self):
        provider, _, state, sensors = self.setup_state(profile="indirect_even")
        result = provider.run_wls(state)
        self.assertTrue(result["converged"])
        artifact = json.loads(Path(result["evidence_path"]).read_text())
        for index, available in enumerate(sensors["available_mask"]):
            if not available:
                self.assertIsNone(state["measurements"][index])
                self.assertIsNone(artifact["raw_residuals"][index])
        self.assertEqual(provider.correct_measurements(state, {"arguments": {"measurement_updates": {0: 1}}})["execution_status"], "failure")
        self.assertEqual(provider.get_parameter_context(state)["execution_status"], "failure")
        with self.assertRaises(NotImplementedError):
            provider.env_kwargs()
        self.assertIs(provider.provider_hooks()["wls_runner"].__self__, provider)
        nonlogical = copy.deepcopy(state); nonlogical["metadata"].pop("logical_topology")
        self.assertEqual(provider.run_wls(nonlogical)["execution_status"], "failure")
        unredacted = copy.deepcopy(state["measurements"])
        unredacted[sensors["available_mask"].index(False)] = 0.0
        observations = {"values": unredacted, "sensor_ids": [row["sensor_id"] for row in sensors["records"]],
                        "sensor_inventory_hash": sensors["sensor_inventory_hash"]}
        with self.assertRaisesRegex(ValueError, "redacted"):
            provider.state_payload(self.case, self.inventory, self.inventory["normal_statuses"], sensors, observations)


if __name__ == "__main__":
    unittest.main()
