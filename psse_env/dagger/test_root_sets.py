"""One root reader must handle every real artifact shape, and fail closed."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from psse_env.dagger.root_sets import (
    physical_roots_from_artifact,
    root_set_digest,
)

R0 = "physical_v3_" + "a" * 40
R1 = "physical_v3_" + "b" * 40


class RootSetReaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def _write(self, name: str, payload) -> Path:
        path = self.root / name
        if isinstance(payload, str):
            path.write_text(payload, encoding="utf-8")
        else:
            path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_scenario_envelope_list(self):
        path = self._write(
            "scenarios.json",
            [{"grouping": {"physical_root_fingerprint": R0}}],
        )
        self.assertEqual(physical_roots_from_artifact(path), {R0})

    def test_evaluation_suite_mapping(self):
        """The real development holdout is a suite mapping, not a list."""
        path = self._write(
            "development_holdout.json",
            {
                "dagger1_development": [
                    {"grouping": {"physical_root_fingerprint": R0}},
                    {"grouping": {"physical_root_fingerprint": R1}},
                ]
            },
        )
        self.assertEqual(physical_roots_from_artifact(path), {R0, R1})

    def test_aggregate_jsonl(self):
        path = self.root / "aggregate.raw.jsonl"
        path.write_text(
            json.dumps({"physical_root_fingerprint": R0})
            + "\n\n"
            + json.dumps({"physical_root_fingerprint": R1})
            + "\n",
            encoding="utf-8",
        )
        self.assertEqual(physical_roots_from_artifact(path), {R0, R1})

    def test_unversioned_fingerprint_is_rejected(self):
        path = self._write(
            "scenarios.json",
            [{"grouping": {"physical_root_fingerprint": "root-a"}}],
        )
        with self.assertRaises(ValueError) as caught:
            physical_roots_from_artifact(path)
        self.assertIn("versioned", str(caught.exception))

    def test_missing_fingerprint_is_rejected(self):
        path = self._write("scenarios.json", [{"grouping": {}}])
        with self.assertRaises(ValueError):
            physical_roots_from_artifact(path)

    def test_unknown_structure_is_rejected(self):
        with self.assertRaises(ValueError):
            physical_roots_from_artifact(self._write("x.json", 42))

    def test_suite_value_that_is_not_a_row_list_is_rejected(self):
        with self.assertRaises(ValueError):
            physical_roots_from_artifact(self._write("x.json", {"suite": "nope"}))

    def test_empty_artifact_is_rejected_rather_than_returning_an_empty_set(self):
        """An empty set would silently disable a disjointness guard."""
        with self.assertRaises(ValueError):
            physical_roots_from_artifact(self._write("x.json", []))
        with self.assertRaises(ValueError):
            physical_roots_from_artifact(self._write("y.json", {}))

    def test_missing_file_is_rejected(self):
        with self.assertRaises(FileNotFoundError):
            physical_roots_from_artifact(self.root / "absent.json")

    def test_digest_is_order_independent(self):
        self.assertEqual(root_set_digest({R0, R1}), root_set_digest({R1, R0}))
        self.assertNotEqual(root_set_digest({R0}), root_set_digest({R0, R1}))


if __name__ == "__main__":
    unittest.main()
