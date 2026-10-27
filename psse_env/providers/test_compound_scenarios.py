"""Compound-error roots (2026-10-27): randomly combined meter, R/X and breaker atoms re-solved on the node/breaker model."""
from __future__ import annotations

import copy
import json
import unittest

import numpy as np

from psse_env.actions import normalize_action
from psse_env.dagger.suite_builder import partition_release_scenario_v1
from psse_env.providers.matpower import MatpowerDeploymentProviders, measurement_index_map
from psse_env.providers.scenario_generator import (
    COMPOUND_FAMILY,
    COMPOUND_SCAN_COUNT,
    DEFAULT_COMPOUND_CLASSES,
    CompoundSpec,
    Round0ScenarioGenerator,
)

GENERATOR_OPTIONS = dict(
    evidence_profile="scada_only",
    admission_mode="physical",
    normalized_residual_threshold=4.0,
    min_measurement_error_sigma=10.0,
)
CLASSES = ("rx2_adjacent", "rx_cb_same_line", "meter_on_rx", "cb2_mixed", "control_cb", "control_meter")


def _spec(name: str) -> CompoundSpec:
    return next(spec for spec in DEFAULT_COMPOUND_CLASSES if spec.name == name)


def _state(scenario: dict, state_id: str = "episode:s0") -> dict:
    return {
        "state_id": state_id,
        "state_hash": "hash0",
        "status": "active",
        "case": scenario["case"],
        "measurements": list(scenario["measurements"]),
        "metadata": copy.deepcopy(scenario["metadata"]),
    }


class CompoundScenarioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.generator = Round0ScenarioGenerator(seed=11, **GENERATOR_OPTIONS)
        cls.roots = {}
        for name in CLASSES:
            built = cls.generator._build_compound_class(_spec(name), 1)
            assert built, (name, cls.generator.skipped[-3:])
            cls.roots[name] = built[0]

    # ----------------------------------------------------------- the draw

    def test_atoms_follow_the_class_specification(self) -> None:
        for name, root in self.roots.items():
            spec = _spec(name)
            compound = root["compound"]
            self.assertEqual(root["scenario_family"], COMPOUND_FAMILY)
            self.assertEqual(compound["class"], name)
            self.assertEqual(len(root["true_parameter_errors"]), spec.parameters, name)
            self.assertEqual(len(root["true_topology_errors"]), spec.breakers, name)
            low, high = spec.meter_range
            self.assertTrue(low <= len(root["true_measurement_errors"]) <= high, name)
            self.assertEqual(root["error_cardinality"], spec.parameters + spec.breakers + len(root["true_measurement_errors"]))
            kinds = [atom["kind"] for atom in compound["atoms"]]
            self.assertEqual(kinds.count("parameter"), spec.parameters)
            self.assertEqual(kinds.count("breaker"), spec.breakers)
            self.assertEqual(kinds.count("meter"), len(root["true_measurement_errors"]))
            for atom in root["true_parameter_errors"]:
                self.assertEqual(atom["parameter"], "rx")
                for factor in (atom["r_factor"], atom["x_factor"]):
                    self.assertTrue(0.1 <= factor <= 0.5 or 2.0 <= factor <= 5.0, factor)
            for atom in root["true_topology_errors"]:
                self.assertIn(atom["physical_effect"], spec.breaker_effects)
                self.assertNotEqual(atom["reported_cb_closed"], atom["true_cb_closed"])
                # The release audit's breaker non-regression reads this for every breaker.
                self.assertEqual(atom["expected_status"], int(atom["true_cb_closed"]))

    def test_placements_are_recorded_as_electrical_distances(self) -> None:
        adjacent = self.roots["rx2_adjacent"]
        (relation,) = adjacent["compound"]["relations"]
        self.assertEqual(relation["relation"], "adjacent")
        self.assertEqual(relation["bus_distance"], 0)
        a, b = adjacent["true_parameter_errors"]
        self.assertTrue({a["from_bus"], a["to_bus"]} & {b["from_bus"], b["to_bus"]})

        same_line = self.roots["rx_cb_same_line"]
        (relation,) = same_line["compound"]["relations"]
        self.assertEqual(relation["relation"], "same_line")
        (line,) = same_line["true_parameter_errors"]
        (breaker,) = same_line["true_topology_errors"]
        self.assertEqual(breaker["physical_effect"], "dangling_line_terminal")
        self.assertEqual(breaker["branch_row0"], line["branch_row0"])

        # The isolated line's impedance is marked not applicable for the release
        # audit (the user's decision of 2026-10-28); the mark names the breaker.
        self.assertEqual(line["not_applicable"], "line_isolated_by_breaker")
        self.assertEqual(line["isolated_by_cb_name"], breaker["cb_name"])
        self.assertEqual(same_line["compound"]["not_applicable_lines"], [line["line_index1"]])
        for name in ("rx2_adjacent", "meter_on_rx"):
            for atom in self.roots[name]["true_parameter_errors"]:
                self.assertNotIn("not_applicable", atom)
            self.assertEqual(self.roots[name]["compound"]["not_applicable_lines"], [])

        mixed = self.roots["cb2_mixed"]
        effects = sorted(atom["physical_effect"] for atom in mixed["true_topology_errors"])
        self.assertEqual(effects, ["bus_split", "dangling_line_terminal"])
        (relation,) = mixed["compound"]["relations"]
        self.assertIn(relation["relation"], {"same_bus", "one_apart", "distant"})

    def test_on_fault_meter_sits_on_the_faulted_line(self) -> None:
        root = self.roots["meter_on_rx"]
        (line,) = root["true_parameter_errors"]
        (meter_atom,) = [atom for atom in root["compound"]["atoms"] if atom["kind"] == "meter"]
        self.assertTrue(meter_atom["on_faulted_element"])
        self.assertEqual(meter_atom["faulted_element_kind"], "parameter_line")
        self.assertIn(meter_atom["channel"], {"Pf", "Qf", "Pt", "Qt"})
        self.assertEqual(meter_atom["branch_row0"], line["branch_row0"])
        self.assertEqual(root["compound"]["meters_on_faulted_elements"], 1)
        (error,) = root["true_measurement_errors"]
        sigma = root["metadata"]["sigma_z"][error["index"]]
        self.assertGreaterEqual(abs(error["observed"] - error["clean"]), 10.0 * sigma)
        self.assertEqual(root["measurements"][error["index"]], error["observed"])

    # -------------------------------------------------- the physical truth

    def test_structural_atoms_change_the_truth_and_are_resolved(self) -> None:
        from mcp_server.matpower_server import _load_python_case

        for name in ("rx2_adjacent", "rx_cb_same_line", "cb2_mixed"):
            root = self.roots[name]
            clean = _load_python_case(root["clean_case"])
            stale = _load_python_case(root["case"])
            for atom in root["true_parameter_errors"]:
                row = int(atom["branch_row0"])
                self.assertAlmostEqual(float(clean["branch"][row][2]), atom["clean_r"], places=9)
                self.assertAlmostEqual(float(clean["branch"][row][3]), atom["clean_x"], places=9)
                self.assertNotAlmostEqual(float(stale["branch"][row][2]), atom["clean_r"], places=6)
            for atom in root["true_topology_errors"]:
                if atom["physical_effect"] == "dangling_line_terminal":
                    self.assertEqual(float(clean["branch"][int(atom["branch_row0"])][10]), 0.0)
                else:
                    self.assertEqual(clean["bus"].shape[0], atom["operator_bus_count_after_fix"])
            telemetry = root["metadata"]["substation_telemetry"]
            self.assertEqual(len(telemetry["node_vm"]), 65)
            self.assertEqual(telemetry["measurement_kind"], "observed")
            # The reported map is the schematic-normal one on every root; the
            # truth differs on exactly the sampled breakers.
            from Transmission.ieee14_full_substation import status_labels

            self.assertEqual(root["metadata"]["reported_breaker_status"], status_labels(self.generator._full_topology_model()))
            for atom in root["true_topology_errors"]:
                reported = root["metadata"]["reported_breaker_status"][atom["cb_name"]]
                self.assertEqual(reported, "closed" if atom["reported_cb_closed"] else "open")

    def test_admission_is_physical_only(self) -> None:
        for name, root in self.roots.items():
            detection = root["compound"]["detection"]
            self.assertTrue(detection["reported_model"]["alarmed"], name)
            self.assertFalse(detection["true_model"]["alarmed"], name)
            options = {"sigma_z": root["metadata"]["sigma_z"], "exact_measurement_indices": root["metadata"]["structural_zero_indices"]}
            alarmed, _ = self.generator._compound_detection(root["case"], root["measurements"], **options)
            self.assertTrue(alarmed, name)
            self.assertEqual(len(root["compound"]["solo_effects"]), len(root["true_parameter_errors"]) + len(root["true_topology_errors"]))
        # No teacher-solvability gate: the generator's parameter-ranking admission is off.
        self.assertFalse(self.generator.report()["parameter_ranking_admission"]["enforced"])

    def test_scans_are_repeated_draws_about_the_operating_point(self) -> None:
        for name, root in self.roots.items():
            scans = root["metadata"]["parameter_scans"]
            z_scans = np.asarray(scans["z_scans"], dtype=float)
            self.assertEqual(z_scans.shape, (COMPOUND_SCAN_COUNT, self.generator.nz), name)
            sigma = np.asarray(scans["sigma_z"], dtype=float)
            self.assertTrue(np.all(sigma > 0.0), name)
            structural = root["metadata"]["structural_zero_indices"]
            self.assertTrue(structural, name)
            self.assertTrue(np.all(z_scans[:, structural] == 0.0), name)
            stochastic = [i for i in range(self.generator.nz) if i not in structural]
            spread = np.abs(z_scans[:, stochastic] - np.asarray(root["clean_measurements"] if len(root["clean_measurements"]) == self.generator.nz else root["measurements"])[stochastic])
            # Each scan is one sensor draw about the same operating point: no
            # scan sits more than a handful of sigmas from the observed vector.
            self.assertLess(float(np.max(spread / sigma[stochastic])), 8.0, name)

    # ------------------------------------------------ the evidence boundary

    def test_policy_visible_metadata_is_uniform_and_carries_no_truth(self) -> None:
        key_sets = {name: tuple(sorted(root["metadata"])) for name, root in self.roots.items()}
        self.assertEqual(len(set(key_sets.values())), 1, key_sets)
        for name, root in self.roots.items():
            envelope = partition_release_scenario_v1(root)
            execution = json.dumps(envelope["execution"], sort_keys=True)
            for forbidden in ("true_", "clean_", "compound", "hidden_truth", "solo_effects", "r_factor"):
                self.assertNotIn(forbidden, execution, (name, forbidden))
            for atom in root["true_parameter_errors"]:
                self.assertNotIn(f"line_index1\": {atom['line_index1']}", execution)
            self.assertIn("parameter_scans", envelope["execution"]["metadata"], name)
            self.assertIn("substation_telemetry", envelope["execution"]["metadata"], name)
            self.assertEqual(envelope["grouping"]["scenario_family"], COMPOUND_FAMILY)
            self.assertEqual(envelope["grouping"]["source_tier"], "physics_synthesized")
            self.assertTrue(envelope["audit"]["truth"]["true_measurement_errors"] is not None)
            self.assertTrue(root["scenario_id"].startswith("r0_"))
            self.assertNotIn(name, root["scenario_id"])

    # ------------------------------------------------------- determinism

    def test_same_seed_reproduces_and_existing_families_are_unchanged(self) -> None:
        # The existing families keep their recoverable admission; the compound
        # family is built last, so their draws are the same with or without it.
        options = {**GENERATOR_OPTIONS, "admission_mode": "recoverable"}
        plan = {"topology": 1, "measurement": 1}
        first = Round0ScenarioGenerator(seed=5, **options)
        second = Round0ScenarioGenerator(seed=5, **options)
        base = first.build(plan)
        with_compound = second.build({**plan, "compound": 1})
        ids = lambda rows: [(row["scenario_family"], row["scenario_id"]) for row in rows]  # noqa: E731
        self.assertEqual(ids(base), ids(with_compound)[: len(base)])
        self.assertEqual(len(with_compound) - len(base), len(DEFAULT_COMPOUND_CLASSES))
        again = Round0ScenarioGenerator(seed=5, **options).build({**plan, "compound": 1})
        self.assertEqual(ids(with_compound), ids(again))
        self.assertEqual([row["measurements"] for row in with_compound], [row["measurements"] for row in again])
        report = second.report()
        self.assertEqual(report["built_by_family"][COMPOUND_FAMILY], len(DEFAULT_COMPOUND_CLASSES))
        self.assertEqual(len(report["compound_classes"]), len(DEFAULT_COMPOUND_CLASSES))
        manifest_classes = [entry["compound_class"] for entry in second.manifest if entry["scenario_family"] == COMPOUND_FAMILY]
        self.assertEqual(manifest_classes, [spec.name for spec in DEFAULT_COMPOUND_CLASSES])

    def test_build_compound_takes_a_class_plan(self) -> None:
        generator = Round0ScenarioGenerator(seed=3, compound_solo_effects=False, **GENERATOR_OPTIONS)
        rows = generator.build_compound({"control_rx": 1})
        self.assertEqual([row["compound"]["class"] for row in rows], ["control_rx"])
        self.assertEqual(rows[0]["compound"]["solo_effects"], [])
        with self.assertRaises(ValueError):
            generator.build_compound({"no_such_class": 1})

    def test_specification_validation(self) -> None:
        with self.assertRaises(ValueError):
            CompoundSpec("bad", meters=4)
        with self.assertRaises(ValueError):
            CompoundSpec("bad", parameters=2, breakers=1)
        with self.assertRaises(ValueError):
            CompoundSpec("bad", parameters=1, parameter_placement="adjacent")
        with self.assertRaises(ValueError):
            CompoundSpec("bad", breakers=1, breaker_placement="same_line")
        with self.assertRaises(ValueError):
            CompoundSpec("bad", meters=1, meter_placement="on_fault")
        with self.assertRaises(ValueError):
            CompoundSpec("bad", breakers=2, breaker_effects=("bus_split",), breaker_pair="mixed")
        with self.assertRaises(ValueError):
            CompoundSpec("bad")
        spec = CompoundSpec.from_mapping({"name": "ok", "meters": [1, 3], "parameters": 1})
        self.assertEqual(spec.meter_range, (1, 3))
        self.assertEqual(CompoundSpec.from_mapping(spec.as_dict()), spec)

    # ------------------------------------------- the deployed executors

    def test_provider_executes_breaker_and_parameter_corrections(self) -> None:
        providers = MatpowerDeploymentProviders(chi2_alpha=0.01, evidence_profile="scada_only")
        root = self.roots["control_cb"]
        (breaker,) = root["true_topology_errors"]
        state = _state(root)
        action = normalize_action({"tool": "correct_topology", "arguments": {
            "state_id": state["state_id"], "cb_name": breaker["cb_name"], "status": int(breaker["true_cb_closed"])}})
        output = providers.correct_topology(state, action)
        self.assertEqual(output.get("execution_status", "success"), "success", output)
        self.assertEqual(output["breaker_effect"], breaker["physical_effect"])

        root = self.roots["rx2_adjacent"]
        line = root["true_parameter_errors"][0]
        state = _state(root)
        action = normalize_action({"tool": "correct_parameters", "arguments": {"state_id": state["state_id"], "line_index": line["line_index1"]}})
        output = providers.correct_parameters(state, action)
        self.assertEqual(output.get("execution_status", "success"), "success", output)
        self.assertEqual(output["line_index"], line["line_index1"])
        self.assertTrue(output["modification"]["case"].endswith(".m"))
        context = providers.get_parameter_context(state)
        self.assertEqual(context.get("execution_status", "success"), "success", context)
        self.assertTrue(context["parameter_scans_available"])
        self.assertEqual(context["parameter_scan_count"], COMPOUND_SCAN_COUNT)


if __name__ == "__main__":
    unittest.main()
