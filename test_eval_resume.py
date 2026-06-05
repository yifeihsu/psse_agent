import json
import tempfile
import unittest
from pathlib import Path

from eval_sft_agent_gemma_v4 import (
    load_completed_results_for_resume,
    validate_completed_results_for_resume,
)


class EvalResumeTests(unittest.TestCase):
    def test_truncates_malformed_trailing_jsonl_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "eval.jsonl"
            output_path.write_text(
                json.dumps({"sample_index": 0, "family_correct": True}) + "\n{partial",
                encoding="utf-8",
            )

            rows = load_completed_results_for_resume(
                str(output_path),
                truncate_partial_output=True,
            )

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["sample_index"], 0)
            self.assertEqual(output_path.read_text(encoding="utf-8").count("\n"), 1)

    def test_rejects_malformed_trailing_jsonl_without_truncation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "eval.jsonl"
            output_path.write_text('{"sample_index": 0}\n{partial', encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "malformed trailing JSONL"):
                load_completed_results_for_resume(
                    str(output_path),
                    truncate_partial_output=False,
                )

    def test_rejects_non_prefix_ordered_sample_indices(self) -> None:
        completed = [{"sample_index": 0}, {"sample_index": 2}]

        with self.assertRaisesRegex(RuntimeError, "not prefix ordered"):
            validate_completed_results_for_resume(
                completed,
                total_samples=3,
                output_path="eval.jsonl",
            )

    def test_rejects_more_completed_rows_than_requested_samples(self) -> None:
        completed = [{"sample_index": 0}, {"sample_index": 1}]

        with self.assertRaisesRegex(RuntimeError, "already has 2 results"):
            validate_completed_results_for_resume(
                completed,
                total_samples=1,
                output_path="eval.jsonl",
            )


if __name__ == "__main__":
    unittest.main()
