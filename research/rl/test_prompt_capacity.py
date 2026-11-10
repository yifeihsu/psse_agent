"""The prompt capacity read: its order, where it stops, and what it calls supported (no device needed)."""
from __future__ import annotations

import unittest

from research.rl.prompt_capacity import DeviceMemory, markdown, probe, summarize, synthetic_ids


class _Card:
    """A card of ``total`` MiB on which a step at ``n`` tokens holds the weights and ``per_token[name] * n`` beyond them."""

    WEIGHTS = 24000

    def __init__(self, total: int, per_token: dict[str, float]) -> None:
        self.total, self.per_token, self.peak, self.order = total, per_token, None, []

    def describe(self) -> dict:
        return {"name": "test card", "total_mib": self.total}

    def reset(self) -> None:
        pass

    def read(self) -> dict:
        return {"peak_allocated_mib": self.peak, "peak_reserved_mib": None if self.peak is None else self.peak + 500}

    def step(self, name: str):
        def run(length: int) -> None:
            self.order.append((name, length))
            need = int(self.WEIGHTS + self.per_token[name] * length)
            self.peak = min(need, self.total)
            if need > self.total:
                raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return run


class PromptCapacityTests(unittest.TestCase):
    def test_each_step_is_tried_shortest_first_and_stops_where_the_card_is_full(self) -> None:
        card = _Card(80000, {"generation": 0.5, "update_row": 1.0})
        results = probe([32768, 8192, 49152, 65536, 16384], {"generation": card.step("generation"), "update_row": card.step("update_row")}, card)
        self.assertEqual([entry["tokens"] for entry in results["generation"]], [8192, 16384, 32768, 49152, 65536])
        self.assertTrue(all(entry["fits"] for entry in results["generation"]))
        # The update row holds 49,152 tokens (73,152 MiB) and not 65,536; nothing longer is tried after the failure.
        self.assertEqual([(entry["tokens"], entry["fits"]) for entry in results["update_row"]],
                         [(8192, True), (16384, True), (32768, True), (49152, True), (65536, False)])
        self.assertEqual(card.order[:5], [("generation", n) for n in (8192, 16384, 32768, 49152, 65536)])
        summary = summarize(results, card.describe(), weights_mib=card.WEIGHTS, headroom=0.9)
        self.assertEqual(summary["steps"]["update_row"], {
            "longest_that_fits": 49152, "longest_within_headroom": 32768, "first_that_does_not_fit": 65536,
            "kib_per_token_beyond_the_weights": 1024.0})
        self.assertEqual(summary["steps"]["generation"]["longest_within_headroom"], 65536)
        self.assertAlmostEqual(summary["steps"]["generation"]["kib_per_token_beyond_the_weights"], 512.0)
        # The supported length is the shorter of the two steps, with the headroom; two workers need twice a peak.
        self.assertEqual(summary["supported_prompt_tokens"], 32768)
        self.assertEqual(summary["two_generation_workers_at_once_up_to"], 16384)
        report = {"device": card.describe(), "weights_mib": card.WEIGHTS, "results": results, "summary": summary,
                  "prompt_limits": {"max_input_tokens": 49152, "model_context_window": 262144}}
        text = markdown(report)
        self.assertIn("**Supported prompt length on this card: 32768 tokens**", text)
        self.assertIn("| 65536 | no | 80000 | 80500 |", text)

    def test_an_error_that_is_not_the_devices_memory_is_not_swallowed(self) -> None:
        def broken(length: int) -> None:
            raise ValueError("a bug")

        with self.assertRaisesRegex(ValueError, "a bug"):
            probe([8], {"generation": broken}, DeviceMemory(None))

    def test_without_a_device_the_figures_are_empty_and_nothing_is_claimed(self) -> None:
        memory = DeviceMemory(None)
        results = probe([4, 8], {"generation": lambda length: None}, memory)
        self.assertEqual([(entry["fits"], entry["peak_allocated_mib"]) for entry in results["generation"]], [(True, None), (True, None)])
        summary = summarize(results, memory.describe(), weights_mib=None)
        self.assertEqual((summary["steps"]["generation"]["longest_that_fits"], summary["two_generation_workers_at_once_up_to"]), (8, None))
        self.assertIsNone(summary["steps"]["generation"]["kib_per_token_beyond_the_weights"])

    def test_synthetic_prompts_have_the_asked_length(self) -> None:
        class Tokenizer:
            bos_token_id = 2

            def __call__(self, text, add_special_tokens=False):
                return {"input_ids": [10, 11, 12, 13, 14]}

        for length in (1, 5, 6, 23):
            ids = synthetic_ids(Tokenizer(), length)
            self.assertEqual((len(ids), ids[0]), (length, 2))


if __name__ == "__main__":
    unittest.main()
