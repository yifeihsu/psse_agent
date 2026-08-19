"""Orchestration must not confuse a fail-closed NO-GO with a crash."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from psse_env.dagger.collection_result import (
    ANALYSIS_COMPLETE,
    ANALYSIS_COMPLETE_ARTIFACT_TYPE,
    EXIT_CODES,
    INFRASTRUCTURE_FAILURE,
    STRICT_GO,
    STRICT_NO_GO,
    classify_collection_result,
    format_summary,
)


class CollectionResultClassificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def _production(self, *, rows: bool = True, manifest: bool = True):
        out = self.root / "training_beta025.jsonl"
        man = self.root / "training_beta025.jsonl.manifest.json"
        if rows:
            out.write_text('{"example_id": "row-0"}\n', encoding="utf-8")
        if manifest:
            man.write_text('{"rows": 1}\n', encoding="utf-8")
        return out, man

    def _bundle(self, payload: dict | None = None, *, readable: bool = True) -> Path:
        d = self.root / "training_beta025.failed-collection"
        d.mkdir(parents=True, exist_ok=True)
        text = (
            json.dumps(payload if payload is not None else self._evidence())
            if readable
            else "{not json"
        )
        (d / "failure_evidence.json").write_text(text, encoding="utf-8")
        return d

    @staticmethod
    def _evidence() -> dict:
        return {
            "failed_gate_names": [
                "deterministic_collection_selection",
                "offline_teacher_target_quarantine_summary",
            ],
            "collection_stopping_report": {
                "stopping_reason": "irreversible_truth_audit_quarantine",
                "executed_episode_count": 151,
                "planned_episode_count": 477,
                "terminal_failure": {"quarantined_rows": 2},
            },
        }

    def test_strict_go(self):
        out, man = self._production()
        result = classify_collection_result(
            exit_code=0, production_output=out, production_manifest=man
        )
        self.assertEqual(result["classification"], STRICT_GO)
        self.assertEqual(result["exit_status"], EXIT_CODES[STRICT_GO])
        self.assertTrue(result["production_outputs_present"])

    def test_fail_closed_no_go_is_not_an_infrastructure_failure(self):
        """The exact shape of DAgger-1 round 2: exit 1, bundle, no outputs."""
        bundle = self._bundle()
        result = classify_collection_result(
            exit_code=1,
            production_output=self.root / "training_beta025.jsonl",
            production_manifest=self.root / "training_beta025.jsonl.manifest.json",
            failed_collection_dir=bundle,
        )
        self.assertEqual(result["classification"], STRICT_NO_GO)
        self.assertEqual(result["exit_status"], 20)
        self.assertNotEqual(result["exit_status"], EXIT_CODES[INFRASTRUCTURE_FAILURE])
        self.assertEqual(
            result["failed_gate_names"],
            [
                "deterministic_collection_selection",
                "offline_teacher_target_quarantine_summary",
            ],
        )
        self.assertEqual(result["quarantined_rows"], 2)
        self.assertEqual(result["executed_episode_count"], 151)
        summary = format_summary(result)
        self.assertIn("STRICT_NO_GO", summary)
        self.assertIn("151/477", summary)
        self.assertIn("31.7%", summary)
        self.assertIn("offline_teacher_target_quarantine_summary", summary)

    def test_crash_and_timeout_are_infrastructure_failures(self):
        for exit_code in (137, 139, 2, 255):
            result = classify_collection_result(exit_code=exit_code)
            self.assertEqual(
                result["classification"], INFRASTRUCTURE_FAILURE, msg=str(exit_code)
            )
            self.assertEqual(result["exit_status"], 1)

    def test_exit_one_without_a_bundle_is_infrastructure_failure(self):
        result = classify_collection_result(exit_code=1)
        self.assertEqual(result["classification"], INFRASTRUCTURE_FAILURE)
        self.assertIn("before the gates", result["detail"])

    def test_unreadable_evidence_is_not_reported_as_a_clean_no_go(self):
        bundle = self._bundle(readable=False)
        result = classify_collection_result(
            exit_code=1, failed_collection_dir=bundle
        )
        self.assertEqual(result["classification"], INFRASTRUCTURE_FAILURE)
        self.assertIn("unreadable", result["detail"])

    def test_outputs_and_bundle_together_are_inconsistent(self):
        out, man = self._production()
        bundle = self._bundle()
        result = classify_collection_result(
            exit_code=0,
            production_output=out,
            production_manifest=man,
            failed_collection_dir=bundle,
        )
        self.assertEqual(result["classification"], INFRASTRUCTURE_FAILURE)
        self.assertIn("inconsistent", result["detail"])

    def test_exit_zero_without_manifest_is_not_a_go(self):
        out, _ = self._production(manifest=False)
        result = classify_collection_result(
            exit_code=0,
            production_output=out,
            production_manifest=self.root / "training_beta025.jsonl.manifest.json",
        )
        self.assertEqual(result["classification"], INFRASTRUCTURE_FAILURE)

    def _analysis_bundle(self, *, executed: int = 477, complete: bool = True) -> Path:
        """A well-formed complete-schedule analysis bundle with real digests."""
        d = self.root / "analysis_beta025.failed-collection"
        d.mkdir(parents=True, exist_ok=True)
        rows = d / "diagnostic.all_visited_rows.jsonl"
        rows.write_text('{"example_id": "row-0"}\n', encoding="utf-8")
        digest = hashlib.sha256(rows.read_bytes()).hexdigest()
        evidence = {
            "artifact_type": ANALYSIS_COMPLETE_ARTIFACT_TYPE,
            "collection_outcome": "analysis_only_complete_schedule_exhausted",
            "analysis_only": True,
            "failed_gate_names": ["independent_root_support"],
            "collection_stopping_report": {
                "stopping_reason": "analysis_only_complete_schedule_exhausted",
                "analysis_only": True,
                "executed_episode_count": executed,
                "planned_episode_count": 477,
            },
            "rollout_disposition_matrix": {
                "passed": complete,
                "workflow_disposition_complete": complete,
            },
            "diagnostic_artifacts": {
                "all_visited_rows": {
                    "relative_path": rows.name,
                    "row_count": 1,
                    "sha256": digest,
                }
            },
        }
        (d / "failure_evidence.json").write_text(
            json.dumps(evidence), encoding="utf-8"
        )
        return d

    def test_complete_analysis_run_is_not_a_strict_no_go(self):
        result = classify_collection_result(
            exit_code=1, failed_collection_dir=self._analysis_bundle()
        )
        self.assertEqual(result["classification"], ANALYSIS_COMPLETE)
        self.assertEqual(result["exit_status"], 30)
        self.assertTrue(result["analysis_only"])
        self.assertTrue(result["schedule_complete"])
        self.assertTrue(result["rollout_dispositions_complete"])
        self.assertTrue(result["diagnostic_checksums_valid"])
        # Exit 30 is non-zero so an ``afterok`` aggregate dependency cannot fire.
        self.assertNotEqual(result["exit_status"], EXIT_CODES[STRICT_GO])
        self.assertIn("ANALYSIS-ONLY", format_summary(result))

    def test_incomplete_analysis_run_is_not_reported_as_complete(self):
        result = classify_collection_result(
            exit_code=1,
            failed_collection_dir=self._analysis_bundle(executed=151),
        )
        self.assertEqual(result["classification"], INFRASTRUCTURE_FAILURE)
        self.assertFalse(result["schedule_complete"])
        self.assertIn("did not complete its schedule", result["detail"])

    def test_analysis_bundle_with_corrupt_artifact_is_rejected(self):
        bundle = self._analysis_bundle()
        (bundle / "diagnostic.all_visited_rows.jsonl").write_text(
            '{"example_id": "tampered"}\n', encoding="utf-8"
        )
        result = classify_collection_result(
            exit_code=1, failed_collection_dir=bundle
        )
        self.assertFalse(result["diagnostic_checksums_valid"])
        self.assertEqual(result["classification"], INFRASTRUCTURE_FAILURE)

    def test_strict_no_go_still_classifies_as_no_go(self):
        result = classify_collection_result(
            exit_code=1, failed_collection_dir=self._bundle()
        )
        self.assertEqual(result["classification"], STRICT_NO_GO)
        self.assertEqual(result["exit_status"], 20)
        self.assertFalse(result["analysis_only"])


if __name__ == "__main__":
    unittest.main()
