from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from psse_env.dagger import TOOL_JSON_SCHEMAS
from psse_env.sft import audit_dataset, load_exact_processor


PINNED_MODELS = (
    (
        "unsloth/gemma-4-31B-it",
        "8a796db4df380b178065ed910849477ff0e99c87",
    ),
    (
        "unsloth/gemma-4-E2B-it",
        "f0c5915f17ad6c66dbeb577fb06ff8925bf8d7ae",
    ),
)

TARGET_ARGUMENTS: tuple[tuple[str, dict[str, Any]], ...] = (
    ("run_wls", {"state_id": "active"}),
    ("verify_candidate", {"state_id": "candidate"}),
    ("get_measurement_context", {"state_id": "active"}),
    ("get_parameter_context", {"state_id": "active"}),
    ("get_topology_context", {"state_id": "active"}),
    (
        "correct_measurements",
        {"state_id": "active", "measurement_updates": {"0": 1.0}},
    ),
    (
        "correct_parameters",
        {"state_id": "active", "branch_row0": 0, "parameter": "x", "value": 0.1},
    ),
    ("correct_topology", {"state_id": "active", "branch_row0": 0, "status": 1}),
    ("commit_state", {"candidate_state_id": "candidate"}),
    ("rollback_state", {"candidate_state_id": "candidate"}),
    ("finalize_diagnosis", {}),
    (
        "ask_for_more_evidence",
        {"state_id": "active", "request": "fault-family disambiguation"},
    ),
    (
        "run_alternative_test",
        {"state_id": "active", "test_name": "normalized-residual scan"},
    ),
)


def build_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, (name, arguments) in enumerate(TARGET_ARGUMENTS):
        rows.append(
            {
                "example_id": f"exact-tool-probe-{index:02d}-{name}",
                "root_scenario_id": f"probe-{index:02d}",
                "tools": copy.deepcopy(TOOL_JSON_SCHEMAS),
                "messages": [
                    {
                        "role": "system",
                        "content": "Choose exactly one valid PSSE diagnostic tool call.",
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "episode_id": "e",
                                "active_state_id": "active",
                                "candidate_state_id": "candidate",
                                "probe_target": name,
                            },
                            sort_keys=True,
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_0",
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": copy.deepcopy(arguments),
                                },
                            }
                        ],
                    },
                ],
                "metadata": {"state_class": "exact_tool_surface_probe"},
            }
        )
    return rows


def run(*, local_files_only: bool = True, max_length: int = 4096) -> dict[str, Any]:
    rows = build_rows()
    processors: dict[str, Any] = {}
    for model, revision in PINNED_MODELS:
        processor, loader = load_exact_processor(
            model,
            revision,
            local_files_only=local_files_only,
        )
        report = audit_dataset(rows, processor, max_length=max_length)
        report_payload = report.to_dict()
        processors[model] = {
            "revision": revision,
            "processor_loader": loader,
            "passed": report.passed,
            "tool_round_trips": report.tool_round_trips,
            "length_audit": report_payload["length_audit"],
            "failures": report.failures,
        }
        if not report.passed:
            raise RuntimeError(f"Exact tool-surface probe failed for {model}: {report.failures[:3]}")
    return {
        "rows": len(rows),
        "adversarial_episode_id": "e",
        "tool_targets": [name for name, _ in TARGET_ARGUMENTS],
        "processors": processors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run every DAgger macro target through pinned Gemma 4 processors."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent
        / "sft_pilot"
        / "all_tools_exact_processor_probe.json",
    )
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    report = run(
        local_files_only=not args.allow_download,
        max_length=args.max_length,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
