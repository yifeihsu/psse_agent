from __future__ import annotations

import unittest

from psse_env.dagger import TOOL_JSON_SCHEMAS
from psse_env.examples.run_exact_tool_probe import (
    build_rows,
    target_arguments_from_registry,
)
from psse_env.sft.gates import validate_tool_schemas


class ExactToolProbeCoverageTests(unittest.TestCase):
    def test_targets_are_schema_derived_and_cover_current_registry(self) -> None:
        targets = target_arguments_from_registry("controller")
        registered = [schema["function"]["name"] for schema in TOOL_JSON_SCHEMAS]
        self.assertEqual([name for name, _ in targets], registered)
        self.assertEqual(len(set(registered)), len(registered))
        self.assertEqual(len(build_rows("controller")), len(registered))

    def test_release_default_is_canonical(self) -> None:
        rows = build_rows()
        self.assertTrue(rows)
        self.assertTrue(
            all(row["metadata"]["protocol"] == "canonical" for row in rows)
        )
        names = {row["messages"][-1]["tool_calls"][0]["function"]["name"] for row in rows}
        self.assertIn("wls_from_path", names)
        self.assertIn("correct_measurements_from_path", names)
        self.assertNotIn("run_wls", names)

    def test_every_generated_target_has_all_required_arguments(self) -> None:
        for index, row in enumerate(build_rows("controller")):
            tools = validate_tool_schemas(row["tools"], row_label=f"row[{index}]")
            schema_by_name = {
                tool["function"]["name"]: tool["function"]["parameters"]
                for tool in tools
            }
            call = row["messages"][-1]["tool_calls"][0]["function"]
            self.assertEqual(
                set(schema_by_name[call["name"]].get("required", [])),
                set(call["arguments"]),
            )


if __name__ == "__main__":
    unittest.main()
