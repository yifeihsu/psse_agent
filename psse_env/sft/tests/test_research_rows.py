from __future__ import annotations

import copy
import json
import unittest

from psse_env.dagger.dataset_builder import CANONICAL_DAGGER_SYSTEM_PROMPT
from psse_env.dagger.preliminary_e2b_eval import canonical_prompt_tool_schemas
from psse_env.dagger.protocol_bridge import unified_tool_schemas
from psse_env.sft.gates import GateError
from psse_env.sft.research_processor_audit import select_audit_rows
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
    def test_validates_raw_registry_then_replaces_only_deep_copies(self) -> None:
        source = _row(0, "wls_from_path")
        original = copy.deepcopy(source)
        normalized, report = normalize_research_rows([source], source_label="test")

        self.assertEqual(source, original)
        self.assertEqual(normalized[0]["tools"], canonical_prompt_tool_schemas())
        self.assertNotEqual(normalized[0]["tools"], original["tools"])
        self.assertTrue(report["source_registry_validated"])
        self.assertEqual(report["rows_changed"], 1)
        self.assertFalse(report["strict_release_rows_mutated"])

    def test_rejects_stale_source_before_normalization(self) -> None:
        source = _row(0, "wls_from_path")
        source["tools"] = source["tools"][:-1]
        with self.assertRaisesRegex(GateError, "source registry validation failed"):
            normalize_research_rows([source], source_label="stale")

    def test_rejects_noncanonical_protocol_before_relabeling_tools(self) -> None:
        source = _row(0, "wls_from_path")
        source["metadata"]["protocol"] = "controller"
        with self.assertRaisesRegex(GateError, "metadata.protocol='canonical'"):
            normalize_research_rows([source], source_label="controller")


class ResearchProcessorSelectionTests(unittest.TestCase):
    def test_selects_31_unique_d0_and_one_labeled_d1_rollback(self) -> None:
        tool_names = [
            "wls_from_path",
            "get_measurement_context",
            "correct_measurements_from_path",
            "get_parameter_context",
            "correct_parameters_from_path",
            "get_topology_context",
            "correct_topology_from_path",
            "commit_state",
            "ask_for_more_evidence",
            "finalize_diagnosis",
        ]
        d0 = [_row(index, tool_names[index % len(tool_names)]) for index in range(45)]
        # A replica with the same example_id must not consume a second slot.
        d0.append(copy.deepcopy(d0[0]))
        rollback = [_row(index, "rollback_state", cohort="d1") for index in range(4)]

        selected = select_audit_rows(d0, rollback, limit=32)

        cohorts = [row["_research_audit_selection"]["source_cohort"] for row in selected]
        self.assertEqual(cohorts.count("d0"), 31)
        self.assertEqual(cohorts.count("d1_rollback_canary"), 1)
        self.assertEqual(len({row["example_id"] for row in selected}), 32)
        self.assertEqual(selected[-1]["messages"][-1]["tool_calls"][0]["function"]["name"], "rollback_state")

    def test_requires_an_actual_rollback_target(self) -> None:
        d0 = [_row(index, "wls_from_path") for index in range(40)]
        with self.assertRaisesRegex(GateError, "no rollback_state"):
            select_audit_rows(d0, [_row(0, "commit_state", cohort="d1")])

    def test_conflicting_duplicate_example_id_fails_selection(self) -> None:
        d0 = [_row(index, "wls_from_path") for index in range(40)]
        conflict = copy.deepcopy(d0[0])
        conflict["metadata"]["state_class"] = "different"
        d0.append(conflict)
        with self.assertRaisesRegex(GateError, "Conflicting rows"):
            select_audit_rows(
                d0,
                [_row(0, "rollback_state", cohort="d1")],
                limit=32,
            )


if __name__ == "__main__":
    unittest.main()
