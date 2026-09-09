"""The gross-error floor on injected meter faults.

With ``min_measurement_error_sigma`` set, a corpus meter error below that
many noise sigmas is lifted (same sign, 1.0 to 1.5 times the floor) before
admission, composed overlays are lifted the same way, and the scenario
records what it did.  Without the option the corpus is used as tracked.
"""

from __future__ import annotations

import unittest

import numpy as np

from psse_env.providers.scenario_generator import Round0ScenarioGenerator


class MeasurementErrorFloorTests(unittest.TestCase):
    def test_floor_lifts_small_corpus_errors_and_records_it(self) -> None:
        generator = Round0ScenarioGenerator(seed=29, min_measurement_error_sigma=10.0)
        sigma = generator.noise_profile()
        scenarios = generator.build({"measurement": 3, "multi_measurement": 2})
        self.assertEqual(len(scenarios), 5)
        lifted_total = 0
        for scenario in scenarios:
            record = scenario["measurement_error_floor"]
            self.assertEqual(record["sigma_multiple"], 10.0)
            lifted_total += len(record["lifted_indices"])
            for fault in scenario["true_measurement_errors"]:
                index = int(fault["index"])
                multiple = abs(fault["observed"] - fault["clean"]) / float(sigma[index])
                self.assertGreaterEqual(multiple, 10.0 - 1e-9, scenario["scenario_id"])
                self.assertAlmostEqual(
                    scenario["measurements"][index], fault["observed"], places=12
                )
                self.assertAlmostEqual(
                    scenario["clean_measurements"][index], fault["clean"], places=12
                )
                if index in record["lifted_indices"]:
                    self.assertLessEqual(multiple, 15.0 + 1e-9)
        # Half of the tracked corpus sits below ten sigma, so a draw of five
        # roots lifts at least one error.
        self.assertGreater(lifted_total, 0)

    def test_floor_keeps_the_original_sign_and_is_off_by_default(self) -> None:
        plain = Round0ScenarioGenerator(seed=29)
        floored = Round0ScenarioGenerator(seed=29, min_measurement_error_sigma=10.0)
        base = plain.build({"measurement": 4})
        lifted = floored.build({"measurement": 4})
        self.assertTrue(all("measurement_error_floor" not in s for s in base))
        by_id = {s["scenario_id"]: s for s in base}
        compared = 0
        for scenario in lifted:
            source = by_id.get(scenario["scenario_id"])
            if source is None:
                continue
            for fault, original in zip(
                scenario["true_measurement_errors"], source["true_measurement_errors"]
            ):
                self.assertEqual(fault["index"], original["index"])
                self.assertEqual(fault["clean"], original["clean"])
                self.assertEqual(
                    np.sign(fault["observed"] - fault["clean"]),
                    np.sign(original["observed"] - original["clean"]),
                )
                self.assertGreaterEqual(
                    abs(fault["observed"] - fault["clean"]),
                    abs(original["observed"] - original["clean"]) - 1e-12,
                )
                compared += 1
        self.assertGreater(compared, 0)

    def test_floor_rejects_nonpositive_values(self) -> None:
        with self.assertRaises(ValueError):
            Round0ScenarioGenerator(seed=1, min_measurement_error_sigma=0.0)

    def test_composed_overlay_respects_the_floor(self) -> None:
        generator = Round0ScenarioGenerator(seed=7, min_measurement_error_sigma=40.0)
        sigma = generator.noise_profile()
        scenarios = generator.build({"measurement+parameter": 1})
        self.assertEqual(len(scenarios), 1)
        for fault in scenarios[0]["true_measurement_errors"]:
            index = int(fault["index"])
            multiple = abs(fault["observed"] - fault["clean"]) / float(sigma[index])
            # The default overlay is 0.10 to 0.30 pu, ten to thirty sigma on a
            # power channel; a forty-sigma floor must lift it.
            self.assertGreaterEqual(multiple, 40.0 - 1e-9)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
