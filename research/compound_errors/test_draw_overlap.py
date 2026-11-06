"""The fresh-draw overlap check (2026-10-29)."""
from __future__ import annotations

import unittest

from research.compound_errors.draw_overlap import overlap, root_key, target_key


def _row(sid: str, cls: str, lines: list[int], meters: list[int], factor: float, load: float) -> dict:
    atoms = [{"kind": "parameter", "line_index1": line, "x_factor": factor} for line in lines]
    atoms += [{"kind": "meter", "index": index, "offset": factor / 10} for index in meters]
    return {"scenario_id": sid, "class": cls, "atoms": atoms, "load_scale": load, "true_lines": lines,
            "true_measurement_indices": meters}


def _manifest(seed: int, class_seeds: list[tuple[str, int]]) -> dict:
    return {"seed": seed, "class_results": [{"class": cls, "seed": s} for cls, s in class_seeds]}


class DrawOverlapTests(unittest.TestCase):
    def test_roots_are_matched_on_every_atom_and_the_load_scale(self) -> None:
        a = _row("r0_a", "meter_rx2", [3, 7], [12], 2.5, 0.91)
        self.assertEqual(root_key(a), root_key({**a, "scenario_id": "other"}))
        self.assertNotEqual(root_key(a), root_key({**a, "load_scale": 0.92}))
        self.assertEqual(target_key(a), target_key(_row("r0_b", "meter_rx2", [7, 3], [12], 1.7, 0.80)))

    def test_a_disjoint_draw_may_repeat_targets_only(self) -> None:
        reference = (_manifest(1, [("rx2_distant", 11), ("control_meter", 12)]),
                     [_row("r0_a", "rx2_distant", [3, 9], [], 2.5, 0.91), _row("r0_b", "control_meter", [], [4], 2.0, 0.85)])
        fresh_rows = [_row("r0_c", "rx2_distant", [3, 9], [], 1.9, 0.88), _row("r0_d", "control_meter", [], [5], 2.0, 0.85)]
        report = overlap((_manifest(2, [("rx2_distant", 21), ("control_meter", 22)]), fresh_rows), reference)
        self.assertTrue(report["disjoint"])
        self.assertEqual(report["classes"]["rx2_distant"],
                         {"n": 1, "shared_seeds": 0, "shared_ids": 0, "identical_roots": 0, "same_targets": 1})
        self.assertEqual(report["all"]["same_targets"], 1)
        # A shared generator seed, a reused id or a repeated root breaks disjointness.
        repeated = overlap((_manifest(2, [("rx2_distant", 11)]), [reference[1][0]]), reference)
        self.assertFalse(repeated["disjoint"])
        self.assertEqual(repeated["classes"]["rx2_distant"],
                         {"n": 1, "shared_seeds": 1, "shared_ids": 1, "identical_roots": 1, "same_targets": 1})


if __name__ == "__main__":
    unittest.main()
