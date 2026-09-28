from __future__ import annotations

import copy
import json
import unittest

from psse_env.dagger.dataset_builder import CANONICAL_DAGGER_SYSTEM_PROMPT
from psse_env.dagger.preliminary_e2b_eval import canonical_prompt_tool_schemas
from psse_env.dagger.protocol_bridge import unified_tool_schemas
from psse_env.sft.gates import GateError
from psse_env.sft.research_rows import normalize_research_rows


def _row(index: int, tool: str, *, cohort: str = "d0") -> dict:
    return {
        "example_id": f"{cohort}-{index}",
        "physical_root_fingerprint": f"root-{cohort}-{index}",
        "tools": unified_tool_schemas(),
        "messages": [
            {"role": "system", "content": CANONICAL_DAGGER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {"state": {"history_window": [{"step": value} for value in range(index % 6)]}},
                    sort_keys=True,
                ),
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": tool, "arguments": {}},
                    }
                ],
            },
        ],
        "metadata": {"protocol": "canonical", "state_class": f"class-{index % 3}"},
    }


class ResearchRowNormalizationTests(unittest.TestCase):
    def test_replaces_registry_only_in_deep_copies(self) -> None:
        source = _row(0, "wls_from_path")
        original = copy.deepcopy(source)
        normalized, report = normalize_research_rows([source], source_label="test")

        self.assertEqual(source, original)
        self.assertEqual(normalized[0]["tools"], canonical_prompt_tool_schemas())
        self.assertNotEqual(normalized[0]["tools"], original["tools"])
        self.assertTrue(report["source_registry_replaced"])
        self.assertEqual(report["rows_changed"], 1)
        self.assertFalse(report["strict_release_rows_mutated"])

    def test_normalizes_rows_rendered_under_an_older_registry(self) -> None:
        source = _row(0, "wls_from_path")
        source["tools"] = source["tools"][:-1]
        normalized, report = normalize_research_rows([source], source_label="older")
        self.assertEqual(normalized[0]["tools"], canonical_prompt_tool_schemas())
        self.assertEqual(report["rows_changed"], 1)
        self.assertEqual(len(report["source_registry_digests"]), 1)

    def test_rejects_noncanonical_protocol_before_relabeling_tools(self) -> None:
        source = _row(0, "wls_from_path")
        source["metadata"]["protocol"] = "controller"
        with self.assertRaisesRegex(GateError, "metadata.protocol='canonical'"):
            normalize_research_rows([source], source_label="controller")


if __name__ == "__main__":
    unittest.main()
