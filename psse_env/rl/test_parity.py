"""Tests of the sampling-versus-training parity diagnostic's pure parts."""

from __future__ import annotations

import math
import unittest

try:
    import torch
except Exception:  # pragma: no cover - torch optional
    torch = None

from psse_env.rl.parity import BUCKETS, CLIP_THRESHOLD, ForcedSequence, bucket_of, gap_statistics, summarize_parity


class ParityTests(unittest.TestCase):
    def test_buckets_partition_the_unit_interval(self) -> None:
        self.assertEqual(bucket_of(0.0), "p<1e-4")
        self.assertEqual(bucket_of(5e-3), "1e-4<=p<1e-2")
        self.assertEqual(bucket_of(0.3), "1e-2<=p<0.5")
        self.assertEqual(bucket_of(0.9), "0.5<=p<0.99")
        self.assertEqual(bucket_of(0.99), "p>=0.99")
        self.assertEqual(bucket_of(1.0), "p>=0.99")
        lows = [low for _name, low, _high in BUCKETS]
        self.assertEqual(lows, sorted(lows))

    def test_gap_statistics_count_the_tokens_beyond_the_clip(self) -> None:
        stats = gap_statistics([-1.0, -2.0, -3.0, float("nan")], [-1.0, -2.5, -3.1, -4.0])
        self.assertEqual(stats["tokens"], 4)
        self.assertEqual(stats["nonfinite"], 1)
        self.assertAlmostEqual(stats["mean_abs"], (0.0 + 0.5 + 0.1) / 3)
        self.assertAlmostEqual(stats["max_abs"], 0.5)
        self.assertEqual(stats["beyond_threshold"], 1)  # 0.5 > ln 1.2 = 0.182
        self.assertAlmostEqual(stats["threshold"], CLIP_THRESHOLD)
        with self.assertRaises(ValueError):
            gap_statistics([1.0], [1.0, 2.0])

    def test_summary_reports_pairs_overall_and_by_bucket_and_lists_the_worst_tokens(self) -> None:
        tokens = [
            {"row": 0, "position": 0, "token_id": 1, "record": math.log(0.995), "forward": math.log(0.994), "training": math.log(0.993),
             "bucket": bucket_of(0.995)},
            {"row": 0, "position": 1, "token_id": 2, "record": math.log(0.6), "forward": math.log(0.59), "training": math.log(0.1),
             "bucket": bucket_of(0.6)},
            {"row": 1, "position": 0, "token_id": 3, "record": math.log(1e-6), "forward": math.log(1e-6), "training": math.log(1e-9),
             "bucket": bucket_of(1e-6)},
        ]
        summary = summarize_parity(tokens, worst=2)
        self.assertEqual(summary["tokens"], 3)
        self.assertEqual(summary["sources"], ["record", "forward", "training"])
        self.assertIn("record_vs_forward", summary["pairs"])
        self.assertIn("forward_vs_training", summary["pairs"])
        self.assertNotIn("record_vs_decode", summary["pairs"])  # no decode values were supplied
        self.assertEqual(summary["buckets"]["0.5<=p<0.99"]["tokens"], 1)
        self.assertAlmostEqual(summary["buckets"]["0.5<=p<0.99"]["record_vs_training"]["max_abs"], math.log(0.6) - math.log(0.1))
        self.assertEqual(summary["buckets"]["p<1e-4"]["record_vs_training"]["beyond_threshold"], 1)
        self.assertEqual(summary["worst_pair"], "record_vs_training")
        self.assertEqual([t["token_id"] for t in summary["worst"]], [3, 2])

    def test_the_probe_selects_the_worst_first_tokens_and_agreeing_controls(self) -> None:
        from psse_env.rl.parity_probe import select_probe_rows

        tokens = [
            {"root": "a", "sample": 0, "step": 1, "position": 0, "record": -0.1, "forward": -20.0},
            {"root": "a", "sample": 0, "step": 1, "position": 1, "record": -0.1, "forward": -30.0},  # not the first token
            {"root": "b", "sample": 1, "step": 2, "position": 0, "record": -1.0, "forward": -3.0},
            {"root": "c", "sample": 0, "step": 3, "position": 0, "record": -0.2, "forward": -0.21},
            {"root": "d", "sample": 2, "step": 4, "position": 0, "record": -0.5, "forward": -0.6},
        ]
        selection = select_probe_rows(tokens, worst=1, controls=1)
        self.assertEqual(selection["bad"], [("a", 0, 1)])
        self.assertEqual(selection["control"], [("c", 0, 3)])
        selection = select_probe_rows(tokens, worst=8, controls=8)
        self.assertEqual(selection["bad"], [("a", 0, 1), ("b", 1, 2)])
        self.assertEqual(selection["control"], [("c", 0, 3), ("d", 2, 4)])

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_the_forced_sequence_processor_leaves_only_the_forced_token(self) -> None:
        processor = ForcedSequence([4, 2])
        scores = torch.zeros(1, 6)
        first = processor(None, scores)
        self.assertEqual(int(first.argmax(dim=-1)), 4)
        self.assertTrue(torch.isinf(first[0, 0]) and first[0, 0] < 0)
        self.assertEqual(float(first[0, 4]), 0.0)
        second = processor(None, scores)
        self.assertEqual(int(second.argmax(dim=-1)), 2)
        with self.assertRaises(RuntimeError):
            processor(None, scores)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
