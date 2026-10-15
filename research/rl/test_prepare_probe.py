import unittest

from research.rl.prepare_probe import select_probe


def row(index, stratum):
    audit = {}
    if stratum == "misranked":
        audit["parameter_ranking"] = {"stratum": stratum}
    elif stratum in {"same_sign", "opposite_sign"}:
        audit["mimic"] = {"variant": f"mimic_flow_pair_{stratum}"}
    return {"execution": {"scenario_id": f"r{index}"}, "audit": audit,
            "grouping": {"root_scenario_id": f"r{index}", "physical_root_fingerprint": f"hash{index}",
                         "source_realization_id": f"parent{index}", "scenario_family": "parameter"}}


class PrepareProbeTests(unittest.TestCase):
    def test_balanced_selection_is_frozen_and_does_not_borrow_holdout(self):
        training = [row(i, group) for i, group in enumerate(["misranked", "same_sign", "opposite_sign", "controls"] * 10)]
        selected, manifest = select_probe(training, [row(100, "controls")])
        self.assertEqual(len(selected), 32)
        self.assertTrue(manifest["parent_disjointness_fully_checked"])
        self.assertEqual(select_probe(training, [row(100, "controls")])[0], selected)
        with self.assertRaisesRegex(ValueError, "do not borrow"):
            select_probe(training[:4], [row(100, "controls")])

    def test_distinct_ids_do_not_hide_a_reused_physical_parent(self):
        training = [row(1, "controls")]
        held_out = row(2, "controls")
        held_out["grouping"]["source_realization_id"] = "parent1"
        with self.assertRaisesRegex(ValueError, "overlap"):
            select_probe(training, [held_out], quotas={"controls": 1})

    def test_missing_parent_information_is_reported_as_unchecked(self):
        training = [row(1, "controls")]
        del training[0]["grouping"]["source_realization_id"]
        _, manifest = select_probe(training, [], quotas={"controls": 1})
        self.assertFalse(manifest["parent_disjointness_fully_checked"])


if __name__ == "__main__":
    unittest.main()
