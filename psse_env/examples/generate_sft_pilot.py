from __future__ import annotations

import argparse
import copy
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from psse_env.actions import (
    CORRECT_MEASUREMENTS,
    CORRECT_PARAMETERS,
    CORRECT_TOPOLOGY,
    GET_MEASUREMENT_CONTEXT,
    GET_PARAMETER_CONTEXT,
    GET_TOPOLOGY_CONTEXT,
    RUN_WLS,
)
from psse_env.dagger.dataset_builder import examples_to_chat_sft, write_jsonl
from psse_env.dagger.rollout_collector import (
    DaggerRolloutCollector,
    audit_target_aware_state_classes,
)
from psse_env.dagger.sft_audit import (
    audit_chat_sft_rows,
    audit_teacher_realizability,
)
from psse_env.dagger.splits import grouped_scenario_split
from psse_env.oracle import ExpertPolicyOracle
from psse_env.transactional_env import TransactionalPSSEEnv


SEED = 20260715
DEFAULT_ROOT_SCENARIOS = 15
NOMINAL_MEASUREMENTS = (1.0, 2.0, 3.0)
NOMINAL_BRANCHES = (
    {"branch_id": "L1", "x": 0.10, "status": 1},
    {"branch_id": "L2", "x": 0.20, "status": 1},
)


class ReviewedDeterministicPilotAdapters:
    """Observable deterministic stand-ins approved only for the small pilot.

    These adapters derive their outputs from the physical state presented to
    the provider.  They never inspect the scenario's hidden-truth payload.
    Production integrations should replace them with the real PSSE/WLS and
    correction adapters while retaining the same fail-closed contracts.
    """

    provider_kind = "deterministic"
    production_dataset_approved = True

    @staticmethod
    def _physical_findings(state: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
        measurements = list(state.get("measurements") or [])
        case = state.get("case") if isinstance(state.get("case"), Mapping) else {}
        branches = list(case.get("branch") or []) if isinstance(case, Mapping) else []

        measurement_findings = [
            {
                "measurement_index": index,
                "observed": observed,
                "reference": expected,
                "absolute_residual": abs(float(observed) - expected),
            }
            for index, (observed, expected) in enumerate(
                zip(measurements, NOMINAL_MEASUREMENTS)
            )
            if abs(float(observed) - expected) > 1e-9
        ]
        parameter_findings: list[dict[str, Any]] = []
        topology_findings: list[dict[str, Any]] = []
        for index, expected in enumerate(NOMINAL_BRANCHES):
            if index >= len(branches) or not isinstance(branches[index], Mapping):
                continue
            observed = branches[index]
            if abs(float(observed.get("x", expected["x"])) - float(expected["x"])) > 1e-9:
                parameter_findings.append(
                    {
                        "branch_row0": index,
                        "branch_id": expected["branch_id"],
                        "parameter": "x",
                        "observed": observed.get("x"),
                        "reference": expected["x"],
                    }
                )
            if int(observed.get("status", expected["status"])) != int(expected["status"]):
                topology_findings.append(
                    {
                        "branch_row0": index,
                        "branch_id": expected["branch_id"],
                        "observed_status": observed.get("status"),
                        "reference_status": expected["status"],
                    }
                )
        return {
            "measurement": measurement_findings,
            "parameter": parameter_findings,
            "topology": topology_findings,
        }

    def run_wls(self, state: Mapping[str, Any]) -> dict[str, Any]:
        findings = self._physical_findings(state)
        unresolved = [
            f"measurement_residual:index={item['measurement_index']}"
            for item in findings["measurement"]
        ]
        unresolved.extend(
            f"parameter_reactance:branch={item['branch_id']}"
            for item in findings["parameter"]
        )
        unresolved.extend(
            f"topology_line_status:branch={item['branch_id']}"
            for item in findings["topology"]
        )
        family_scores = {
            family: float(len(items)) for family, items in findings.items()
        }
        anomaly_score = float(sum(family_scores.values()))
        is_candidate = state.get("parent_state_id") is not None
        resolved = anomaly_score < 0.5
        return {
            "wls_objective": anomaly_score,
            "remaining_anomaly_score": anomaly_score,
            "anomaly_threshold": 0.5,
            "max_normalized_residual": max(family_scores.values(), default=0.0),
            "family_residual_scores": family_scores,
            "unresolved_signatures": unresolved,
            "remaining_fault_count": len(unresolved),
            "target_progress": 1.0 if is_candidate and resolved else 0.0,
            "global_progress": 1.0 if is_candidate and resolved else 0.0,
            "globally_resolved": resolved,
            "post_action_resolved": resolved,
            "physical_constraints_ok": True,
            "converged": True,
            "solver": "reviewed_deterministic_pilot_wls_v1",
        }

    def measurement_context(self, state: Mapping[str, Any]) -> dict[str, Any]:
        findings = self._physical_findings(state)["measurement"]
        return {
            "measurement_findings": findings,
            "finding_count": len(findings),
            "context_version": "pilot_measurement_v1",
            "supported_corrections": [
                {
                    "tool": CORRECT_MEASUREMENTS,
                    "arguments": {
                        "state_id": state["state_id"],
                        "measurement_updates": {
                            finding["measurement_index"]: finding["reference"]
                        },
                    },
                }
                for finding in findings
            ],
        }

    def parameter_context(self, state: Mapping[str, Any]) -> dict[str, Any]:
        findings = self._physical_findings(state)["parameter"]
        return {
            "parameter_findings": findings,
            "finding_count": len(findings),
            "context_version": "pilot_parameter_v1",
            "supported_corrections": [
                {
                    "tool": CORRECT_PARAMETERS,
                    "arguments": {
                        "state_id": state["state_id"],
                        "branch_row0": finding["branch_row0"],
                        "parameter": finding["parameter"],
                        "value": finding["reference"],
                    },
                }
                for finding in findings
            ],
        }

    def topology_context(self, state: Mapping[str, Any]) -> dict[str, Any]:
        findings = self._physical_findings(state)["topology"]
        return {
            "topology_findings": findings,
            "finding_count": len(findings),
            "context_version": "pilot_topology_v1",
            "supported_corrections": [
                {
                    "tool": CORRECT_TOPOLOGY,
                    "arguments": {
                        "state_id": state["state_id"],
                        "branch_row0": finding["branch_row0"],
                        "status": finding["reference_status"],
                    },
                }
                for finding in findings
            ],
        }

    def execute_correction(
        self,
        state: Mapping[str, Any],
        action: Mapping[str, Any],
    ) -> dict[str, Any]:
        del state
        arguments = action.get("arguments")
        if not isinstance(arguments, Mapping):
            return {
                "execution_status": "failure",
                "error_code": "invalid_correction_arguments",
            }
        modification = {
            str(key): copy.deepcopy(value)
            for key, value in arguments.items()
            if str(key) not in {"state_id", "candidate_state_id"}
        }
        return {
            "modification": modification,
            "executor_receipt": "reviewed_deterministic_pilot_executor_v1",
        }


class ObservableBaselinePolicy:
    def act(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "tool": RUN_WLS,
            "arguments": {"state_id": observation["active_state_id"]},
        }


def _clean_case() -> dict[str, Any]:
    return {"name": "pilot_case", "branch": copy.deepcopy(list(NOMINAL_BRANCHES))}


def build_scenarios(count: int = DEFAULT_ROOT_SCENARIOS) -> list[dict[str, Any]]:
    if count < 8:
        raise ValueError("The grouped pilot requires at least eight root scenarios.")
    scenarios: list[dict[str, Any]] = []
    for index in range(count):
        family = ("measurement", "parameter", "topology")[index % 3]
        case = _clean_case()
        clean_case = _clean_case()
        measurements = list(NOMINAL_MEASUREMENTS)
        clean_measurements = list(NOMINAL_MEASUREMENTS)
        scenario: dict[str, Any] = {
            "scenario_id": f"pilot_root_{index:03d}",
            "root_scenario_id": f"pilot_root_{index:03d}",
            "case": case,
            "clean_case": clean_case,
            "measurements": measurements,
            "clean_measurements": clean_measurements,
            "true_measurement_errors": [],
            "true_parameter_errors": [],
            "true_topology_errors": [],
        }
        if family == "measurement":
            target = index % len(measurements)
            offset = 0.25 + 0.05 * (index % 4)
            measurements[target] += offset
            scenario["true_measurement_errors"] = [
                {
                    "index": target,
                    "observed": measurements[target],
                    "clean": clean_measurements[target],
                }
            ]
        elif family == "parameter":
            target = index % len(NOMINAL_BRANCHES)
            expected = float(NOMINAL_BRANCHES[target]["x"])
            case["branch"][target]["x"] = expected * (1.5 + 0.1 * (index % 3))
            scenario["true_parameter_errors"] = [
                {
                    "branch_row0": target,
                    "parameter": "x",
                    "clean": expected,
                }
            ]
        else:
            target = index % len(NOMINAL_BRANCHES)
            case["branch"][target]["status"] = 0
            scenario["true_topology_errors"] = [
                {
                    "branch_row0": target,
                    "expected_status": 1,
                }
            ]
        scenarios.append(scenario)
    return scenarios


def _split_ownership(splits: Mapping[str, list[Mapping[str, Any]]]) -> dict[str, str]:
    ownership: dict[str, str] = {}
    for split_name, rows in splits.items():
        for row in rows:
            root = str(row.get("root_scenario_id", row.get("scenario_id")))
            previous = ownership.setdefault(root, split_name)
            if previous != split_name:
                raise RuntimeError(
                    f"root scenario {root!r} appears in both {previous} and {split_name}"
                )
    return ownership


def _preferred_action_distribution(rows: list[Mapping[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        action = row.get("preferred_action")
        tool = action.get("tool") if isinstance(action, Mapping) else None
        counts[str(tool or "unknown")] += 1
    return dict(sorted(counts.items()))


def generate(output_dir: Path, *, root_scenarios: int = DEFAULT_ROOT_SCENARIOS) -> dict[str, Any]:
    adapters = ReviewedDeterministicPilotAdapters()
    env = TransactionalPSSEEnv(
        production_dataset_mode=True,
        wls_runner=adapters.run_wls,
        context_providers={
            GET_MEASUREMENT_CONTEXT: adapters.measurement_context,
            GET_PARAMETER_CONTEXT: adapters.parameter_context,
            GET_TOPOLOGY_CONTEXT: adapters.topology_context,
        },
        correction_executors={
            CORRECT_MEASUREMENTS: adapters.execute_correction,
            CORRECT_PARAMETERS: adapters.execute_correction,
            CORRECT_TOPOLOGY: adapters.execute_correction,
        },
        max_steps=8,
        history_window=4,
    )
    provider_report = env.validate_production_configuration()
    scenarios = build_scenarios(root_scenarios)
    raw_rows = DaggerRolloutCollector(
        env=env,
        policy=ObservableBaselinePolicy(),
        expert_oracle=ExpertPolicyOracle(),
        rng=random.Random(SEED),
    ).collect_iteration(
        scenarios=scenarios,
        iteration=0,
        beta=1.0,
        max_steps=8,
    )
    nonproduction_rows = [
        row.get("example_id")
        for row in raw_rows
        if row.get("dataset_mode") != "production"
        or (row.get("labels") or {}).get("dataset_mode") != "production"
    ]
    if nonproduction_rows:
        raise RuntimeError(
            "Production pilot collector emitted untagged rows: "
            + ", ".join(str(value) for value in nonproduction_rows[:5])
        )
    raw_splits = grouped_scenario_split(
        raw_rows,
        train_fraction=0.75,
        validation_fraction=0.15,
        seed=SEED,
    )
    ownership = _split_ownership(raw_splits)
    if not raw_splits["train"] or not raw_splits["validation"]:
        raise RuntimeError("Deterministic grouped split produced an empty train or validation split.")

    exported = {
        split_name: examples_to_chat_sft(rows)
        for split_name, rows in raw_splits.items()
    }
    train_validation_rows = len(exported["train"]) + len(exported["validation"])
    if not 32 <= train_validation_rows <= 128:
        raise RuntimeError(
            "Corrected pilot must contain 32-128 train+validation rows; "
            f"generated {train_validation_rows}."
        )

    all_exported = [row for split in exported.values() for row in split]
    chat_audit = audit_chat_sft_rows(all_exported)
    realizability = audit_teacher_realizability(all_exported, conflict_tolerance=0.0)
    class_audit = audit_target_aware_state_classes(raw_rows)
    if not chat_audit["passed"]:
        raise RuntimeError(f"Native chat-row audit failed: {chat_audit['errors'][:3]}")
    if not realizability["passed"]:
        raise RuntimeError(
            "Teacher realizability audit failed: "
            f"conflict_rate={realizability['conflict_rate']}"
        )
    if not class_audit["passed"]:
        raise RuntimeError(
            f"Target-aware class audit failed: {class_audit['mismatches'][:3]}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "pilot.raw.jsonl", raw_rows)
    for split_name, rows in exported.items():
        write_jsonl(output_dir / f"pilot.{split_name}.jsonl", rows)

    report = {
        "seed": SEED,
        "production_dataset_mode": True,
        "provider_configuration": provider_report,
        "root_scenarios": root_scenarios,
        "root_ownership": dict(sorted(ownership.items())),
        "raw_rows": len(raw_rows),
        "split_rows": {name: len(rows) for name, rows in exported.items()},
        "train_validation_rows": train_validation_rows,
        "preferred_action_distribution": _preferred_action_distribution(raw_rows),
        "state_class_distribution": class_audit["class_counts"],
        "native_chat_audit": chat_audit,
        "teacher_realizability": realizability,
        "target_aware_class_audit": class_audit,
    }
    (output_dir / "pilot.preflight.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a grouped, production-mode DAgger tool-SFT pilot."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "sft_pilot",
    )
    parser.add_argument("--root-scenarios", type=int, default=DEFAULT_ROOT_SCENARIOS)
    args = parser.parse_args()
    report = generate(args.output_dir, root_scenarios=args.root_scenarios)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "raw_rows": report["raw_rows"],
                "split_rows": report["split_rows"],
                "conflict_rate": report["teacher_realizability"]["conflict_rate"],
                "class_counts": report["state_class_distribution"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
