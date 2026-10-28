"""Compound-error roots on the bus-branch model (2026-10-28): the classes that need no breaker, on IEEE 118."""
from __future__ import annotations

import copy
import tempfile
import unittest

import numpy as np

from psse_env.actions import normalize_action
from psse_env.providers import compound_pairs
from psse_env.providers.matpower import MatpowerDeploymentProviders, _load_python_case, measurement_index_map
from psse_env.providers.scenario_generator import (
    BUS_BRANCH_COMPOUND_CLASSES,
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
CLASSES = ("rx2_adjacent", "rx2_distant", "meter_on_rx", "control_meter")
FLOOR = 1.5


def _spec(name: str) -> CompoundSpec:
    return next(spec for spec in BUS_BRANCH_COMPOUND_CLASSES if spec.name == name)


def _state(scenario: dict) -> dict:
    return {
        "state_id": "episode:s0",
        "state_hash": "hash0",
        "status": "active",
        "case": scenario["case"],
        "measurements": list(scenario["measurements"]),
        "metadata": copy.deepcopy(scenario["metadata"]),
    }


class BusBranchCompoundTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory()
        cls.generator = Round0ScenarioGenerator(
            system="case118", compound_only=True, compound_effect_floor=FLOOR, seed=7,
            derived_case_dir=cls.temp.name, **GENERATOR_OPTIONS,
        )
        cls.roots = {}
        for name in CLASSES:
            built = cls.generator._build_compound_class(_spec(name), 1)
            assert built, (name, cls.generator.skipped[-3:])
            cls.roots[name] = built[0]
        cls.branch = np.asarray(_load_python_case("case118")["branch"], dtype=float)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    # ----------------------------------------------------------- configuration

    def test_systems_without_a_node_breaker_model_draw_the_bus_branch_classes(self) -> None:
        self.assertEqual(self.generator.compound_physics, "bus_branch")
        self.assertEqual(self.generator.compound_classes, BUS_BRANCH_COMPOUND_CLASSES)
        self.assertFalse(any(spec.has_breakers for spec in BUS_BRANCH_COMPOUND_CLASSES))
        breaker = next(spec for spec in DEFAULT_COMPOUND_CLASSES if spec.name == "control_cb")
        with self.assertRaisesRegex(ValueError, "node/breaker model"):
            Round0ScenarioGenerator(system="case118", compound_only=True, compound_classes=[breaker], **GENERATOR_OPTIONS)
        with self.assertRaisesRegex(ValueError, "no node/breaker model"):
            Round0ScenarioGenerator(system="case118", compound_only=True, compound_physics="node_breaker", **GENERATOR_OPTIONS)
        # IEEE 14 keeps its node/breaker classes and its physical-only admission.
        ieee14 = Round0ScenarioGenerator(seed=11, **GENERATOR_OPTIONS)
        self.assertEqual(ieee14.compound_physics, "node_breaker")
        self.assertEqual(ieee14.compound_classes, DEFAULT_COMPOUND_CLASSES)
        self.assertIsNone(ieee14.compound_effect_floor)
        with self.assertRaisesRegex(ValueError, "bus_branch admission"):
            Round0ScenarioGenerator(seed=11, compound_effect_floor=FLOOR, **GENERATOR_OPTIONS)

    def test_a_compound_only_generator_reads_no_corpus(self) -> None:
        with self.assertRaisesRegex(ValueError, "explicit fresh corpus"):
            Round0ScenarioGenerator(system="case118", **GENERATOR_OPTIONS)
        with self.assertRaisesRegex(ValueError, "Unsupported families"):
            self.generator.build({"parameter": 1})
        with self.assertRaisesRegex(ValueError, "reads no corpus"):
            self.generator._corpus()
        report = self.generator.report()
        self.assertEqual((report["compound_physics"], report["compound_effect_floor"], report["compound_only"]),
                         ("bus_branch", FLOOR, True))

    def test_distant_hops_is_a_distant_pair_constraint_and_keeps_the_ieee14_record(self) -> None:
        for spec in DEFAULT_COMPOUND_CLASSES:
            self.assertNotIn("distant_hops", spec.as_dict())
            self.assertEqual(CompoundSpec.from_mapping(spec.as_dict()), spec)
        distant = _spec("rx2_distant")
        self.assertEqual(distant.as_dict()["distant_hops"], 3)
        self.assertEqual(CompoundSpec.from_mapping(distant.as_dict()), distant)
        with self.assertRaises(ValueError):
            CompoundSpec("x", parameters=2, parameter_placement="adjacent", distant_hops=3)
        with self.assertRaises(ValueError):
            CompoundSpec("x", parameters=2, parameter_placement="distant", distant_hops=1)

    # ----------------------------------------------------------- the draw

    def test_placements_record_hops_and_the_footprint_overlap(self) -> None:
        table = compound_pairs.bus_distances(self.branch)
        for name in ("rx2_adjacent", "rx2_distant"):
            compound = self.roots[name]["compound"]
            first, second = (atom for atom in compound["atoms"] if atom["kind"] == "parameter")
            hops = compound_pairs.line_hops(self.branch, first["branch_row0"], second["branch_row0"], table)
            (relation,) = compound["relations"]
            self.assertEqual(relation["bus_distance"], hops)
            if name == "rx2_adjacent":
                self.assertEqual(hops, 0)
                self.assertTrue({first["from_bus"], first["to_bus"]} & {second["from_bus"], second["to_bus"]})
            else:
                self.assertGreaterEqual(hops, 3)
            self.assertGreaterEqual(relation["overlap"], 0.0)
            self.assertLessEqual(relation["overlap"], 1.0)
            self.assertLessEqual(abs(relation["x_correlation"]), 1.0 + 1e-9)
            self.assertLessEqual(relation["max_abs_correlation"], relation["overlap"] + 1e-9)
            self.assertEqual(set(compound["line_degrees"]), {str(first["line_index1"]), str(second["line_index1"])})
            self.assertGreaterEqual(compound["multiplier_view"]["band_count"], 1)
        meter = next(atom for atom in self.roots["meter_on_rx"]["compound"]["atoms"] if atom["kind"] == "meter")
        line = next(atom for atom in self.roots["meter_on_rx"]["compound"]["atoms"] if atom["kind"] == "parameter")
        self.assertTrue(meter["on_faulted_element"])
        self.assertEqual(meter["branch_row0"], line["branch_row0"])
        self.assertIn(meter["channel"], {"Pf", "Qf", "Pt", "Qt"})

    def test_every_atom_reaches_the_effect_floor_and_the_admission_is_physical(self) -> None:
        for name, root in self.roots.items():
            compound = root["compound"]
            self.assertEqual((compound["system"], compound["physics"], compound["effect_floor"]), ("case118", "bus_branch", FLOOR))
            effects = compound["solo_effects"]
            self.assertEqual(len(effects), len(compound["atoms"]))
            for effect in effects:
                self.assertGreaterEqual(effect["effect_ratio"], FLOOR, (name, effect))
                self.assertEqual(effect["dof"], 863)
                self.assertIn("alarmed", effect["noisy"])
            self.assertTrue(compound["detection"]["reported_model"]["alarmed"])
            self.assertFalse(compound["detection"]["true_model"]["alarmed"])
            silent = [e["atom"] for e in effects if not e["noisy"]["alarmed"]]
            self.assertEqual(compound["silent_atoms"], silent)
            self.assertEqual(compound["alarms_only_jointly"], len(silent) == len(effects))
        reasons = {entry["reason"] for entry in self.generator.skipped}
        self.assertLessEqual(reasons, {"compound_atom_below_effect_floor", "compound_not_alarmed_on_reported_model",
                                       "compound_true_model_not_clean", "opf_diverged", "compound_operating_bounds_violated",
                                       "compound_telemetry_physics_mismatch"})
        for entry in self.generator.skipped:
            if entry["reason"] == "compound_atom_below_effect_floor":
                self.assertIn("joint_alarm", entry["metrics"])
                self.assertTrue(any(ratio is None or ratio < FLOOR for ratio in entry["metrics"]["solo_effect_ratios"]))

    def test_truth_is_the_true_networks_solution(self) -> None:
        index_map = measurement_index_map(118, 186)
        for name, root in self.roots.items():
            clean = np.asarray(root["clean_measurements"], dtype=float)
            observed = np.asarray(root["measurements"], dtype=float)
            meters = {int(item["index"]) for item in root["true_measurement_errors"]}
            differs = set(np.flatnonzero(np.abs(observed - clean) > 0).tolist())
            self.assertEqual(differs, meters, name)
            corrected = np.asarray(_load_python_case(root["clean_case"])["branch"], dtype=float)
            changed = set(np.flatnonzero(np.any(np.abs(corrected[:, 2:4] - self.branch[:, 2:4]) > 0, axis=1)).tolist())
            self.assertEqual(changed, {int(item["branch_row0"]) for item in root["true_parameter_errors"]}, name)
            for item in root["true_parameter_errors"]:
                self.assertAlmostEqual(corrected[item["branch_row0"], 2], item["clean_r"])
                self.assertAlmostEqual(corrected[item["branch_row0"], 3], item["clean_x"])
            alarmed, _ = self.generator._compound_detection(root["clean_case"], clean.tolist(), sigma_z=root["metadata"]["sigma_z"])
            self.assertFalse(alarmed, name)
            self.assertEqual(len(observed), 3 * 118 + 4 * 186)
            self.assertEqual(index_map["Qt"].stop, len(observed))

    def test_scans_are_repeated_draws_about_the_operating_point(self) -> None:
        for name, root in self.roots.items():
            scans = root["metadata"]["parameter_scans"]
            self.assertEqual(len(scans["z_scans"]), COMPOUND_SCAN_COUNT)
            sigma = np.asarray(scans["sigma_z"], dtype=float)
            clean = np.asarray(root["clean_measurements"], dtype=float)
            for scan in scans["z_scans"]:
                spread = (np.asarray(scan) - clean) / sigma
                # two independent noise draws about the same noiseless point
                self.assertAlmostEqual(float(np.std(spread)), np.sqrt(2.0), delta=0.15)

    def test_policy_visible_metadata_is_uniform_and_carries_no_truth(self) -> None:
        keys = {name: set(root["metadata"]) for name, root in self.roots.items()}
        self.assertEqual(len({frozenset(value) for value in keys.values()}), 1, keys)
        self.assertEqual(next(iter(keys.values())), {"sigma_z", "evidence_profile", "parameter_scans"})
        for root in self.roots.values():
            self.assertEqual(root["scenario_family"], COMPOUND_FAMILY)
            self.assertEqual(root["case"], "case118")
            self.assertEqual(root["network_case"], "case118")
            text = repr(root["metadata"])
            for forbidden in ("true_", "clean_r", "clean_x", "compound", "r_factor"):
                self.assertNotIn(forbidden, text)

    def test_same_seed_reproduces(self) -> None:
        again = Round0ScenarioGenerator(
            system="case118", compound_only=True, compound_effect_floor=FLOOR, seed=7,
            derived_case_dir=self.temp.name, **GENERATOR_OPTIONS,
        )
        built = again._build_compound_class(_spec(CLASSES[0]), 1)
        self.assertEqual(built[0]["scenario_id"], self.roots[CLASSES[0]]["scenario_id"])
        self.assertEqual(built[0]["measurements"], self.roots[CLASSES[0]]["measurements"])

    # ----------------------------------------------------------- the deployed provider

    def test_the_deployed_provider_alarms_ranks_and_corrects_on_a_118_root(self) -> None:
        root = self.roots["rx2_distant"]
        providers = MatpowerDeploymentProviders(evidence_profile="scada_only", normalized_residual_threshold=4.0)
        state = _state(root)
        solve = providers.run_wls(state)
        self.assertEqual(solve.get("execution_status", "success"), "success")
        self.assertFalse(solve["globally_resolved"])
        context = providers.get_parameter_context(state)
        self.assertEqual(context["route_status"], "actionable")
        top = int(context["parameter_ranking_distinct_lines"][0]["line_index1"])
        truth = {int(item["line_index1"]): item for item in root["true_parameter_errors"]}
        if top in truth:
            action = normalize_action({"tool": "correct_parameters", "arguments": {"state_id": "episode:s0", "line_index": top}})
            correction = providers.correct_parameters(state, action)
            self.assertEqual(correction.get("execution_status", "success"), "success", correction.get("error_detail"))


class CompoundPairsTests(unittest.TestCase):
    def test_hops_degree_and_band_on_ieee14(self) -> None:
        branch = np.asarray(_load_python_case("case14")["branch"], dtype=float)
        self.assertEqual(compound_pairs.line_hops(branch, 0, 1), 0)  # 1-2 and 1-5 share bus 1
        self.assertEqual(compound_pairs.bus_degree(branch)[4], 5)
        view = compound_pairs.band_view([3.0, 2.5, 1.9, 0.0], rows_of_interest=[2])
        self.assertEqual((view["band_count"], view["band_lines"], view["line_ranks"]), (2, [1, 2], {"3": 3}))

    def test_overlap_is_a_symmetric_cosine(self) -> None:
        rng = np.random.default_rng(3)
        columns = rng.normal(size=(40, 6))
        ea = columns.T @ columns
        same = compound_pairs.pair_overlap(ea, 1, 1)
        self.assertAlmostEqual(same["overlap"], 1.0, places=9)
        forward, backward = compound_pairs.pair_overlap(ea, 0, 2), compound_pairs.pair_overlap(ea, 2, 0)
        self.assertAlmostEqual(forward["overlap"], backward["overlap"], places=12)
        self.assertLessEqual(forward["max_abs_correlation"], forward["overlap"] + 1e-12)
        orthogonal = np.eye(6)
        self.assertAlmostEqual(compound_pairs.pair_overlap(orthogonal, 0, 1)["overlap"], 0.0, places=12)


if __name__ == "__main__":
    unittest.main()
