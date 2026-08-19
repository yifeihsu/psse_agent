"""Probe-suite publication must be atomic, excluded-root aware, and separate."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from psse_env.dagger.build_recovery_probe_suite import (
    aggregate_roots,
    build_recovery_probe_suite,
    envelope_roots,
)


def _envelope(root: str, index: int = 0) -> dict:
    return {
        "grouping": {
            "physical_root_fingerprint": root,
            "scenario_family": "measurement",
            "error_cardinality": 1,
            "scenario_id": f"scenario-{index}",
        },
        "execution": {"scenario_id": f"scenario-{index}"},
    }


class ProbeSuiteRootReaderTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_envelope_roots_reads_grouping_fingerprints(self):
        path = self.root / "scenarios.json"
        path.write_text(
            json.dumps([_envelope("root-a", 0), _envelope("root-b", 1)]),
            encoding="utf-8",
        )
        self.assertEqual(envelope_roots(path), {"root-a", "root-b"})

    def test_envelope_roots_rejects_a_non_list(self):
        path = self.root / "scenarios.json"
        path.write_text(json.dumps({"not": "a list"}), encoding="utf-8")
        with self.assertRaises(ValueError):
            envelope_roots(path)

    def test_aggregate_roots_reads_the_raw_row_file(self):
        directory = self.root / "d0"
        directory.mkdir()
        (directory / "aggregate.raw.jsonl").write_text(
            "\n".join(
                json.dumps({"physical_root_fingerprint": f"d0-root-{i}"})
                for i in range(3)
            )
            + "\n",
            encoding="utf-8",
        )
        self.assertEqual(
            aggregate_roots(directory), {"d0-root-0", "d0-root-1", "d0-root-2"}
        )


class ProbeSuitePublicationTests(unittest.TestCase):
    """These guards fire before the environment is constructed."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.scenarios = self.root / "scenarios.json"
        self.scenarios.write_text(
            json.dumps([_envelope("root-a", 0)]), encoding="utf-8"
        )

    def _build(self, **kwargs):
        return build_recovery_probe_suite(
            scenarios_path=self.scenarios,
            output=self.root / "probes.jsonl",
            manifest_path=self.root / "probes.manifest.json",
            source_commit="a" * 40,
            generator_identity="test",
            quotas={"post_failure_no_candidate": 1},
            **kwargs,
        )

    def test_existing_output_is_never_overwritten(self):
        (self.root / "probes.jsonl").write_text("stale\n", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            self._build()
        # The stale file is left exactly as it was.
        self.assertEqual(
            (self.root / "probes.jsonl").read_text(encoding="utf-8"), "stale\n"
        )

    def test_existing_manifest_is_never_overwritten(self):
        (self.root / "probes.manifest.json").write_text("{}", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            self._build()

    def test_empty_scenario_list_is_refused(self):
        self.scenarios.write_text(json.dumps([]), encoding="utf-8")
        with self.assertRaises(ValueError):
            self._build()

    def test_fully_excluded_root_set_is_refused(self):
        """Every candidate root sits in the development holdout."""
        holdout = self.root / "holdout.json"
        holdout.write_text(json.dumps([_envelope("root-a", 0)]), encoding="utf-8")
        with self.assertRaises(ValueError) as caught:
            self._build(development_holdout=holdout)
        self.assertIn("excluded", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
