from __future__ import annotations

import copy
import unittest

from psse_env.dagger.protocol_bridge import unified_tool_schemas
from psse_env.sft.gates import GateError
from psse_env.sft.research_views import build_research_views


TOOLS = (
    "wls_from_path",
    "get_measurement_context",
    "correct_measurements_from_path",
    "get_parameter_context",
    "correct_parameters_from_path",
    "commit_state",
    "ask_for_more_evidence",
    "finalize_diagnosis",
)


def _row(index: int) -> dict:
    return {
        "example_id": f"example-{index}",
        "physical_root_fingerprint": f"root-{index // 4}",
        "tools": unified_tool_schemas(),
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "{}"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": TOOLS[index % len(TOOLS)],
                            "arguments": {},
                        },
                    }
                ],
            },
        ],
        "metadata": {"protocol": "canonical"},
    }


class ResearchViewTests(unittest.TestCase):
    def test_views_are_deterministic_nested_and_root_disjoint(self) -> None:
        rows = [_row(index) for index in range(240)]
        rows.append(copy.deepcopy(rows[0]))
        first, report = build_research_views(rows)
        second, _ = build_research_views(rows)

        self.assertEqual(first, second)
        self.assertEqual(len(first["smoke_train16"]), 16)
        self.assertEqual(len(first["smoke_validation8"]), 8)
        self.assertEqual(len(first["mini_train128"]), 128)
        self.assertEqual(len(first["mini_validation32"]), 32)
        train_roots = {
            row["physical_root_fingerprint"] for row in first["mini_train128"]
        }
        validation_roots = {
            row["physical_root_fingerprint"]
            for row in first["mini_validation32"]
        }
        self.assertFalse(train_roots & validation_roots)
        self.assertEqual(report["partition"]["overlap"], [])
        self.assertTrue(
            {row["example_id"] for row in first["smoke_train16"]}
            <= {row["example_id"] for row in first["mini_train128"]}
        )

    def test_stale_registry_is_rejected_before_selection(self) -> None:
        rows = [_row(index) for index in range(240)]
        rows[0]["tools"] = rows[0]["tools"][:-1]
        with self.assertRaisesRegex(GateError, "registry is stale"):
            build_research_views(rows)


if __name__ == "__main__":
    unittest.main()
