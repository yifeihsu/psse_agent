"""Cross-check the canonical case contract at generation/deployment boundaries."""

from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from mcp_server.matpower_server import _load_python_case
from psse_env.systems import resolve_system


class SystemRegistryTests(unittest.TestCase):
    def test_registered_cases_match_existing_deployment_loader(self):
        for name, dimensions in (("case14", (14, 20, 122, 27)), ("case57", (57, 80, 491, 113)),
                                 ("case118", (118, 186, 1098, 235))):
            with self.subTest(system=name):
                spec = resolve_system(name)
                self.assertEqual((spec.nb, spec.nl, spec.nz, spec.state_count), dimensions)
                deployment = _load_python_case(spec.case_path)
                generation = spec.load_case()
                self.assertEqual(generation["baseMVA"], deployment["baseMVA"])
                for key in ("bus", "gen", "branch"):
                    np.testing.assert_array_equal(generation[key], deployment[key])
                self.assertEqual(generation["version"], "2")
                self.assertEqual(generation["gencost"].shape[0], generation["gen"].shape[0])

    def test_ieee57_preserves_parallel_transformers_shunts_and_costs(self):
        spec = resolve_system("case57")
        case = spec.load_case()
        self.assertEqual(spec.base_case_hash, "417701198ec205ae9cf8502b365664c1adb5a265894a03ffa4ddba95d540beca")
        self.assertEqual(case["gen"].shape, (7, 21))
        self.assertEqual(case["gencost"].shape, (7, 7))
        for rows, endpoints in (((18, 19), (4, 18)), ((34, 35), (24, 25))):
            a, b = (spec.branches[row] for row in rows)
            self.assertEqual((a.from_bus, a.to_bus), endpoints)
            self.assertEqual((b.from_bus, b.to_bus), endpoints)
            self.assertEqual((a.circuit_ordinal, b.circuit_ordinal), (1, 2))
            self.assertNotEqual(a.asset_id, b.asset_id)
            self.assertNotEqual(a.tap, 0)
            self.assertNotEqual(b.tap, 0)
            self.assertNotIn(a.row0, spec.eligible_parameter_rows0)
            self.assertNotIn(b.row0, spec.eligible_parameter_rows0)
        self.assertEqual(int(np.count_nonzero(case["bus"][:, 5])), 3)
        self.assertEqual(case["gen"][0, 3], 200)
        self.assertEqual(case["gen"][0, 4], -140)

    def test_ieee118_preserves_parallel_circuits_taps_shunts_and_source_voltages(self):
        spec = resolve_system("case118")
        case = spec.load_case()
        self.assertEqual(spec.source_provenance["source_sha256"],
                         "90c28f0d55324a6f11b6371c3fd8424b3c1bc18830699deaec207d7d72e50c25")
        self.assertEqual(case["gen"].shape, (54, 21))
        self.assertEqual(case["gencost"].shape, (54, 7))
        self.assertEqual(int(np.count_nonzero(case["bus"][:, 1] == 3)), 1)
        self.assertEqual(int(case["bus"][case["bus"][:, 1] == 3, 0][0]), 69)
        parallel = {(branch.from_bus, branch.to_bus) for branch in spec.branches if branch.circuit_ordinal == 2}
        self.assertEqual(parallel, {(42, 49), (49, 54), (56, 59), (49, 66), (77, 80), (89, 90), (89, 92)})
        tapped = [branch.row0 for branch in spec.branches if branch.tap != 0]
        self.assertEqual(tapped, [7, 31, 35, 50, 92, 94, 101, 106, 126])
        self.assertEqual(len(spec.eligible_parameter_rows0), 177)
        self.assertTrue(set(tapped).isdisjoint(spec.eligible_parameter_rows0))
        self.assertEqual(int(np.count_nonzero(case["bus"][:, 5])), 14)
        self.assertEqual(sorted(case["bus"][case["bus"][:, 5] < 0, 0].astype(int).tolist()), [5, 37])
        self.assertEqual(sorted(set(case["bus"][:, 9].tolist())), [138.0, 161.0, 345.0])
        self.assertEqual(spec.to_manifest()["residual_degrees_of_freedom"], 863)

    def test_cases_and_sigma_are_fresh_and_asset_identity_is_immutable(self):
        spec = resolve_system("case57")
        first, second = spec.load_case(), spec.load_case()
        first["bus"][0, 2] = -999.0
        first["gencost"][0, 4] = -999.0
        self.assertNotEqual(first["bus"][0, 2], second["bus"][0, 2])
        self.assertNotEqual(first["gencost"][0, 4], second["gencost"][0, 4])
        sigma = spec.measurement_sigma()
        sigma[:] = 999
        np.testing.assert_array_equal(spec.measurement_sigma()[:57], np.full(57, 0.001))
        np.testing.assert_array_equal(spec.measurement_sigma()[57:], np.full(434, 0.01))
        with self.assertRaises(FrozenInstanceError):
            spec.branches[0].row0 = 99
        with self.assertRaises(TypeError):
            spec.external_bus_to_row0[57] = 0
        self.assertEqual(spec.external_bus_to_row0[57], 56)
        self.assertEqual(spec.row0_to_external_bus[56], 57)
        for branch in spec.branches:
            self.assertEqual(branch.index1, branch.row0 + 1)
        self.assertEqual(len({asset.asset_id for asset in spec.branches}), 80)

    def test_case_mutation_after_resolution_is_detected(self):
        spec = resolve_system("case57")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "case57.m"
            path.write_text(spec._asset_path.read_text().replace("mpc.baseMVA = 100;", "mpc.baseMVA = 101;"))
            changed = replace(spec, _asset_path=path)
            with self.assertRaisesRegex(ValueError, "asset changed"):
                changed.load_case()

    def test_manifest_layout_matches_provider_and_serializes(self):
        from psse_env.providers.matpower import measurement_index_map

        spec = resolve_system("case57")
        manifest = spec.to_manifest()
        json.dumps(manifest, allow_nan=False)
        self.assertEqual(manifest["residual_degrees_of_freedom"], 378)
        self.assertEqual(manifest["supported_families"], [
            "no_error", "measurement", "multi_measurement", "parameter", "measurement+parameter",
        ])
        layout = manifest["measurement_contract"]["layout"]
        for name, slc in measurement_index_map(spec.nb, spec.nl).items():
            self.assertEqual(layout[name], [slc.start, slc.stop])
        self.assertFalse(manifest["measurement_contract"]["capabilities"]["three_phase"])
        self.assertFalse(manifest["measurement_contract"]["capabilities"]["detailed_topology"])
        manifest["branches"][0]["index1"] = 999
        self.assertEqual(spec.to_manifest()["branches"][0]["index1"], 1)

    def test_defaults_aliases_and_unsupported_configuration(self):
        self.assertEqual(resolve_system().case_id, "case14")
        for alias in ("57", "ieee57", "CASE57", "case57.m"):
            self.assertEqual(resolve_system(alias).case_id, "case57")
        for alias in ("118", "ieee118", "CASE118", "case118.m"):
            self.assertEqual(resolve_system(alias).case_id, "case118")
        for unsupported in ("case300", "arbitrary.m", ""):
            with self.assertRaisesRegex(ValueError, "Unsupported system"):
                resolve_system(unsupported)
        with self.assertRaisesRegex(ValueError, "Unsupported covariance"):
            resolve_system("case57", covariance_model="scaled_covariance")
        with self.assertRaises(TypeError):
            resolve_system({"case_id": "case57"})


if __name__ == "__main__":
    unittest.main()
