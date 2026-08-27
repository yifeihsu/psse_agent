"""CPU-only tests for the focused research repair curriculum."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from psse_env.dagger.dataset_builder import CANONICAL_DAGGER_SYSTEM_PROMPT
from psse_env.dagger.offline_teacher_target_audit import (
    OFFLINE_TEACHER_TARGET_AUDIT_CONTRACT,
)
from psse_env.dagger.protocol_bridge import unified_tool_schemas
from scripts import research_dagger_demo


def _row(
    index: int,
    *,
    prefix: str,
    tool: str,
    recovery_stratum: str = "",
    physical_root: str | None = None,
) -> dict[str, Any]:
    case_path = "candidate" if tool == "rollback_state" else "active"
    prefix_marker = sum(
        (ordinal + 1) * ord(character) for ordinal, character in enumerate(prefix)
    )
    state = {
        "active_state_id": "active",
        "candidate_state_id": "candidate",
        "history_window": [],
        "remaining_budget": prefix_marker * 1000 + index,
    }
    return {
        "example_id": f"{prefix}-{tool}-{index}",
        "physical_root_fingerprint": physical_root or f"{prefix}-root-{index}",
        "recovery_stratum": recovery_stratum or None,
        "tools": unified_tool_schemas(),
        "messages": [
            {"role": "system", "content": CANONICAL_DAGGER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {"state": state},
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
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
                            "name": tool,
                            "arguments": {"case_path": case_path},
                        },
                    }
                ],
            },
        ],
        "metadata": {
            "controller": {
                "state_aliases": {
                    "active": f"controller:{prefix}:active:{index}",
                    "candidate": f"controller:{prefix}:candidate:{index}",
                }
            }
        },
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )


def _sources(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    d0 = tmp_path / "d0.jsonl"
    natural_train = tmp_path / "natural-train.jsonl"
    natural_validation = tmp_path / "natural-validation.jsonl"
    probe_donor = tmp_path / "probe-donor.jsonl"
    probe_audit = tmp_path / "probe-audit.jsonl"

    d0_quotas = {
        "get_topology_context": 19,
        "correct_parameters_from_path": 32,
        "wls_from_path": 77,
        "get_measurement_context": 12,
        "ask_for_more_evidence": 6,
        "correct_measurements_from_path": 20,
        "commit_state": 12,
        "correct_topology_from_path": 14,
        "get_parameter_context": 8,
        "finalize_diagnosis": 8,
        "estimate_hif_location_magnitude_from_path": 2,
        "estimate_hif_location_magnitude_multiscan_from_path": 2,
        "get_harmonic_context": 2,
        "run_hse_from_path": 2,
        "run_three_phase_nlm_from_path": 2,
    }
    d0_rows = [
        _row(index, prefix=f"d0-{tool}", tool=tool)
        for tool, count in d0_quotas.items()
        for index in range(count)
    ]

    # These donors exactly realize every frozen natural-D1 category.  The
    # expected-shortfall categories deliberately contain only the expected
    # number of rows, while gettopo contains one donor that must be replayed.
    natural_categories = (
        ("commit", "commit_state", "", 6),
        ("rollback-rejected", "rollback_state", "rejected_candidate_rollback", 18),
        ("wls-escalation", "wls_from_path", "premature_escalation_recovery", 22),
        ("corrmeas", "correct_measurements_from_path", "", 9),
        ("corrparam", "correct_parameters_from_path", "", 19),
        ("wls-invalid", "wls_from_path", "invalid_precondition_repair", 2),
        ("wls-post", "wls_from_path", "post_failure_no_candidate", 5),
        ("ask-loop", "ask_for_more_evidence", "loop_escape", 5),
        ("getmeas-loop", "get_measurement_context", "loop_escape", 3),
        ("rollback-commit", "rollback_state", "premature_commit_recovery", 44),
        ("wls-loop", "wls_from_path", "loop_escape", 29),
        (
            "getmeas-sequential",
            "get_measurement_context",
            "sequential_measurement_parameter_recovery",
            45,
        ),
        (
            "ask-multi",
            "ask_for_more_evidence",
            "multi_measurement_safe_handoff",
            42,
        ),
    )
    natural_train_rows = [
        _row(
            index,
            prefix=f"natural-{name}",
            tool=tool,
            recovery_stratum=stratum,
        )
        for name, tool, stratum, count in natural_categories
        for index in range(count)
    ]
    natural_validation_rows = [
        _row(
            0,
            prefix="natural-gettopo",
            tool="get_topology_context",
        )
    ]
    probe_rows = [
        *[
            _row(
                index,
                prefix="probe-post-failure",
                tool="wls_from_path",
                recovery_stratum="post_failure_no_candidate",
            )
            for index in range(12)
        ],
        *[
            _row(
                index,
                prefix="probe-unsupported",
                tool="get_measurement_context",
                recovery_stratum="unsupported_correction_recovery",
            )
            for index in range(12)
        ],
    ]
    for row in probe_rows:
        row["dataset_source"] = "observable_recovery_probe"
        row["auxiliary_training_eligible"] = True
        row["training_decision_evidence_verified"] = True

    probe_audit_rows = [
        {
            "example_id": row["example_id"],
            "physical_root_fingerprint": row["physical_root_fingerprint"],
            "recovery_stratum": row["recovery_stratum"],
            "auxiliary_training_eligible": True,
            "training_decision_evidence_verified": True,
            "observable_rank_one_target_proof": {"passed": True},
            "offline_teacher_target_audit": {
                "contract": OFFLINE_TEACHER_TARGET_AUDIT_CONTRACT,
                "passed": True,
            },
        }
        for row in probe_rows
    ]
    _write_jsonl(d0, list(reversed(d0_rows)))
    _write_jsonl(natural_train, list(reversed(natural_train_rows)))
    _write_jsonl(natural_validation, natural_validation_rows)
    _write_jsonl(probe_donor, list(reversed(probe_rows)))
    _write_jsonl(probe_audit, list(reversed(probe_audit_rows)))
    return d0, natural_train, natural_validation, probe_donor, probe_audit


def test_repair_curriculum_builds_exact_deterministic_balanced_view(
    tmp_path: Path,
) -> None:
    d0, natural_train, natural_validation, probe_donor, probe_audit = _sources(tmp_path)
    protected = tmp_path / "protected.jsonl"
    _write_jsonl(
        protected,
        [
            _row(
                1,
                prefix="protected",
                tool="wls_from_path",
                physical_root="untouched-root",
            )
        ],
    )
    first = tmp_path / "repair-1.jsonl"
    second = tmp_path / "repair-2.jsonl"

    report = research_dagger_demo.make_repair_curriculum(
        d0,
        [natural_train, natural_validation],
        probe_donor,
        probe_audit,
        first,
        tmp_path / "report-1.json",
        protected_paths=[protected],
        seed=3407,
        require_reference_binding=False,
    )
    research_dagger_demo.make_repair_curriculum(
        d0,
        [natural_train, natural_validation],
        probe_donor,
        probe_audit,
        second,
        tmp_path / "report-2.json",
        protected_paths=[protected],
        seed=3407,
        require_reference_binding=False,
    )

    assert first.read_bytes() == second.read_bytes()
    assert report["passed"] is True
    assert report["research_only"] is True
    assert report["release_eligible"] is False
    assert report["counts"] == {
        "rows": 512,
        "unique_examples": 492,
        "distinct_canonical_prompts": 492,
        "physical_roots": 492,
        "protected_roots": 1,
        "protected_overlap": 0,
        "maximum_example_copies": 2,
        "maximum_canonical_prompt_copies": 2,
        "maximum_d0_rows_per_root": 2,
        "maximum_shared_d1_probe_rows_per_root": 2,
        "d0_candidates_excluded_by_protected_root": 0,
        "natural_candidates_excluded_by_protected_root": 0,
    }
    assert report["source_distribution"] == {
        "d0_bc0": 237,
        "natural_dagger1": 251,
        "observable_recovery_probe": 24,
    }
    assert report["tool_distribution"] == {
        "ask_for_more_evidence": 53,
        "commit_state": 18,
        "correct_measurements_from_path": 29,
        "correct_parameters_from_path": 51,
        "correct_topology_from_path": 14,
        "estimate_hif_location_magnitude_from_path": 2,
        "estimate_hif_location_magnitude_multiscan_from_path": 2,
        "finalize_diagnosis": 8,
        "get_harmonic_context": 2,
        "get_measurement_context": 72,
        "get_parameter_context": 8,
        "get_topology_context": 40,
        "rollback_state": 62,
        "run_hse_from_path": 2,
        "run_three_phase_nlm_from_path": 2,
        "wls_from_path": 147,
    }
    assert report["natural_source_split_distribution"] == {
        natural_train.name: 249,
        natural_validation.name: 2,
    }
    assert report["probe_audit"] == {
        "rows_checked": 24,
        "auxiliary_eligible": 24,
        "training_evidence_verified": 24,
        "rank_one_proof_passed": 24,
        "private_audit_passed": 24,
    }
    assert report["output"]["sha256"] == hashlib.sha256(first.read_bytes()).hexdigest()
    rows = [json.loads(line) for line in first.read_text().splitlines()]
    assert len(rows) == 512
    assert report["contract"] == research_dagger_demo.REPAIR_CURRICULUM_CONTRACT
    assert all(row["tools"] == unified_tool_schemas() for row in rows)
    assert all("research_curriculum_contract" not in row for row in rows)


def test_repair_curriculum_requires_frozen_source_hashes_by_default(
    tmp_path: Path,
) -> None:
    d0, natural_train, natural_validation, probe_donor, probe_audit = _sources(tmp_path)
    output = tmp_path / "repair.jsonl"

    with pytest.raises(ValueError, match="source hashes do not match"):
        research_dagger_demo.make_repair_curriculum(
            d0,
            [natural_train, natural_validation],
            probe_donor,
            probe_audit,
            output,
            tmp_path / "report.json",
            protected_paths=[],
            seed=3407,
        )
    assert not output.exists()


def test_repair_curriculum_rejects_nonfrozen_seed_before_output(
    tmp_path: Path,
) -> None:
    d0, natural_train, natural_validation, probe_donor, probe_audit = _sources(tmp_path)
    output = tmp_path / "repair.jsonl"

    with pytest.raises(ValueError, match="requires seed 3407"):
        research_dagger_demo.make_repair_curriculum(
            d0,
            [natural_train, natural_validation],
            probe_donor,
            probe_audit,
            output,
            tmp_path / "report.json",
            protected_paths=[],
            seed=17,
            require_reference_binding=False,
        )
    assert not output.exists()


def test_repair_curriculum_rejects_protected_root_overlap_before_output(
    tmp_path: Path,
) -> None:
    d0, natural_train, natural_validation, probe_donor, probe_audit = _sources(tmp_path)
    protected = tmp_path / "protected.jsonl"
    protected_row = _row(
        9,
        prefix="protected",
        tool="wls_from_path",
        physical_root="probe-post-failure-root-0",
    )
    _write_jsonl(protected, [protected_row])
    output = tmp_path / "repair.jsonl"

    with pytest.raises(ValueError, match="overlap protected roots"):
        research_dagger_demo.make_repair_curriculum(
            d0,
            [natural_train, natural_validation],
            probe_donor,
            probe_audit,
            output,
            tmp_path / "report.json",
            protected_paths=[protected],
            seed=3407,
            require_reference_binding=False,
        )
    assert not output.exists()


@pytest.mark.parametrize(
    ("defect", "message"),
    [
        ("missing", "lacks its raw audit source"),
        ("auxiliary", "is not auxiliary eligible"),
        ("training", "lacks verified training evidence"),
        ("proof", "lacks a passed rank-one proof"),
        ("private", "lacks a current passed private audit"),
        ("contract", "lacks a current passed private audit"),
        ("root", "donor/audit root mismatch"),
        ("stratum", "donor/audit stratum mismatch"),
    ],
)
def test_repair_curriculum_rejects_unverified_probe_evidence_before_output(
    tmp_path: Path,
    defect: str,
    message: str,
) -> None:
    d0, natural_train, natural_validation, probe_donor, probe_audit = _sources(tmp_path)
    audit_rows = [
        json.loads(line)
        for line in probe_audit.read_text(encoding="utf-8").splitlines()
    ]
    if defect == "missing":
        audit_rows.pop(0)
    elif defect == "auxiliary":
        audit_rows[0]["auxiliary_training_eligible"] = False
    elif defect == "training":
        audit_rows[0]["training_decision_evidence_verified"] = False
    elif defect == "proof":
        audit_rows[0]["observable_rank_one_target_proof"]["passed"] = False
    elif defect == "private":
        audit_rows[0]["offline_teacher_target_audit"]["passed"] = False
    elif defect == "contract":
        audit_rows[0]["offline_teacher_target_audit"]["contract"] = "obsolete"
    elif defect == "root":
        audit_rows[0]["physical_root_fingerprint"] = "different-root"
    else:
        assert defect == "stratum"
        audit_rows[0]["recovery_stratum"] = "different-stratum"
    _write_jsonl(probe_audit, audit_rows)
    output = tmp_path / "repair.jsonl"

    with pytest.raises(ValueError, match=message):
        research_dagger_demo.make_repair_curriculum(
            d0,
            [natural_train, natural_validation],
            probe_donor,
            probe_audit,
            output,
            tmp_path / "report.json",
            protected_paths=[],
            seed=3407,
            require_reference_binding=False,
        )
    assert not output.exists()


def test_repair_curriculum_rejects_insufficient_tool_capacity(
    tmp_path: Path,
) -> None:
    d0, natural_train, natural_validation, probe_donor, probe_audit = _sources(tmp_path)
    rows = [json.loads(line) for line in d0.read_text().splitlines()]
    rows = [
        row
        for row in rows
        if not (
            row["messages"][-1]["tool_calls"][0]["function"]["name"]
            == "get_topology_context"
            and row["example_id"].endswith("-18")
        )
    ]
    _write_jsonl(d0, rows)

    with pytest.raises(ValueError, match="placed 36 of 38"):
        research_dagger_demo.make_repair_curriculum(
            d0,
            [natural_train, natural_validation],
            probe_donor,
            probe_audit,
            tmp_path / "repair.jsonl",
            tmp_path / "report.json",
            protected_paths=[],
            seed=3407,
            require_reference_binding=False,
        )


def test_repair_curriculum_rejects_noncanonical_source_registry(
    tmp_path: Path,
) -> None:
    d0, natural_train, natural_validation, probe_donor, probe_audit = _sources(tmp_path)
    rows = [json.loads(line) for line in d0.read_text().splitlines()]
    rows[0]["tools"] = copy.deepcopy(rows[0]["tools"][:-1])
    _write_jsonl(d0, rows)

    with pytest.raises(ValueError, match="tool registry is not canonical"):
        research_dagger_demo.make_repair_curriculum(
            d0,
            [natural_train, natural_validation],
            probe_donor,
            probe_audit,
            tmp_path / "repair.jsonl",
            tmp_path / "report.json",
            protected_paths=[],
            seed=3407,
            require_reference_binding=False,
        )


def _preflight_report(
    *,
    adapter: str,
    exact: int,
    covered_tools: set[str],
    validation_sha256: str = "v" * 64,
    schema_rate: float = 1.0,
    state_bound_rate: float = 1.0,
) -> dict[str, Any]:
    expected_tools = (
        ("wls_from_path", 10),
        ("rollback_state", 6),
        ("correct_parameters_from_path", 6),
        ("get_topology_context", 5),
    )
    required = {tool for tool, _count in expected_tools}
    if not covered_tools <= required:
        raise ValueError("covered_tools contains an unexpected tool")
    row_count = 27
    if not len(covered_tools) <= exact <= row_count:
        raise ValueError("exact count cannot realize the requested tool coverage")

    row_tools = [tool for tool, count in expected_tools for _index in range(count)]
    tool_indices = {
        tool: [index for index, value in enumerate(row_tools) if value == tool]
        for tool in required
    }
    required_indices = {tool_indices[tool][0] for tool in sorted(covered_tools)}
    exact_indices = required_indices | set(
        index for index in range(row_count) if index not in required_indices
    )
    exact_indices = required_indices | set(
        sorted(exact_indices - required_indices)[: exact - len(required_indices)]
    )

    schema_count = round(row_count * schema_rate)
    state_bound_count = round(row_count * state_bound_rate)
    if not exact <= state_bound_count <= schema_count <= row_count:
        raise ValueError("fixture rates are incompatible with exact matches")

    schema_indices = exact_indices | set(
        index for index in range(row_count) if index not in exact_indices
    )
    schema_indices = exact_indices | set(
        sorted(schema_indices - exact_indices)[: schema_count - exact]
    )
    state_indices = exact_indices | set(
        sorted(schema_indices - exact_indices)[: state_bound_count - exact]
    )

    results: list[dict[str, Any]] = []
    for index, tool in enumerate(row_tools):
        arguments = {"case_path": "candidate" if tool == "rollback_state" else "active"}
        expected_action = {"tool": tool, "arguments": arguments}
        is_exact = index in exact_indices
        generated_action = (
            copy.deepcopy(expected_action)
            if is_exact
            else {
                "tool": "get_measurement_context",
                "arguments": {"case_path": "active"},
            }
        )
        results.append(
            {
                "example_id": f"validation-{index}",
                "physical_root_fingerprint": f"validation-root-{index}",
                "expected_action": expected_action,
                "generated_action": generated_action,
                "bound_internal_action": (
                    copy.deepcopy(generated_action) if index in state_indices else None
                ),
                "schema_valid": index in schema_indices,
                "state_bound": index in state_indices,
                "target_tool_match": is_exact,
                "exact_target_match": is_exact,
                "hit_max_new_tokens": False,
                "action_metrics": {
                    "truncated_input_tokens": 0,
                    "hit_max_new_tokens": False,
                },
                "generated_text_sha256": hashlib.sha256(
                    f"generated-{index}".encode()
                ).hexdigest(),
                "generated_text_preview": f"generated-{index}",
                "generation_error": None,
                "error": None,
            }
        )

    overall = research_dagger_demo._trace_preflight_summary(results)
    per_expected_tool = {}
    for tool, expected_count in expected_tools:
        tool_results = [
            row for row in results if row["expected_action"]["tool"] == tool
        ]
        per_expected_tool[tool] = {
            "expected_count": expected_count,
            **research_dagger_demo._trace_preflight_summary(tool_results),
        }

    behavior_binding = {
        "maximum_input_tokens": 16384,
        "maximum_new_tokens": 512,
        "source_sha256": {"synthetic_runtime": "s" * 64},
    }

    return {
        "contract": research_dagger_demo.TRACE_PREFLIGHT_CONTRACT,
        "artifact_type": "research_only_trace_preflight",
        "research_only": True,
        "diagnostic_only": True,
        "release_eligible": False,
        "release_ineligibility_reasons": ["synthetic test fixture"],
        "selection": {
            "mode": "all_validation_rows",
            "row_count": row_count,
            "example_ids": [row["example_id"] for row in results],
        },
        "adapter_path": adapter,
        "adapter_tree_sha256": hashlib.sha256(adapter.encode()).hexdigest(),
        "validation_file": "synthetic-validation.jsonl",
        "validation_file_sha256": validation_sha256,
        "validation_row_count": row_count,
        "validation_physical_root_count": row_count,
        "validation_physical_roots": [
            row["physical_root_fingerprint"] for row in results
        ],
        "canonical_tool_registry_sha256": "r" * 64,
        "behavior_binding": behavior_binding,
        "behavior_binding_sha256": research_dagger_demo._stable_sha256(
            behavior_binding
        ),
        "overall": overall,
        "per_expected_tool": per_expected_tool,
        "stop_on_zero_exact": False,
        "zero_exact_stop_triggered": False,
        "results": results,
    }


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def test_preflight_decision_prioritizes_required_tool_coverage(
    tmp_path: Path,
) -> None:
    required = [
        "wls_from_path",
        "rollback_state",
        "correct_parameters_from_path",
        "get_topology_context",
    ]
    baseline = tmp_path / "baseline.json"
    broad = tmp_path / "broad.json"
    covered = tmp_path / "covered.json"
    _write_json(
        baseline,
        _preflight_report(adapter="bc0", exact=2, covered_tools=set()),
    )
    _write_json(
        broad,
        _preflight_report(
            adapter="checkpoint-22",
            exact=7,
            covered_tools=set(required[:3]),
        ),
    )
    _write_json(
        covered,
        _preflight_report(
            adapter="checkpoint-33",
            exact=5,
            covered_tools=set(required),
        ),
    )

    report = research_dagger_demo.choose_repair_checkpoint(
        baseline,
        [broad, covered],
        tmp_path / "decision.json",
        required_tools=required,
        minimum_exact=4,
        minimum_schema_rate=0.95,
        minimum_state_bound_rate=0.90,
        require_baseline_improvement=True,
    )

    assert report["passed"] is True
    assert report["decision"] == "evaluate"
    assert report["selected"]["adapter_path"] == "checkpoint-33"
    assert report["selected"]["critical_tool_coverage"] == 4
    assert report["failures"] == []


def test_preflight_decision_reports_all_stop_reasons_and_cli_exits_two(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    output = tmp_path / "decision.json"
    _write_json(
        baseline,
        _preflight_report(adapter="bc0", exact=3, covered_tools=set()),
    )
    _write_json(
        candidate,
        _preflight_report(
            adapter="checkpoint-11",
            exact=2,
            covered_tools={"wls_from_path"},
            schema_rate=0.80,
            state_bound_rate=0.70,
        ),
    )

    return_code = research_dagger_demo.main(
        [
            "preflight-decision",
            "--baseline",
            str(baseline),
            "--candidate",
            str(candidate),
            "--output",
            str(output),
            "--stop-on-fail",
        ]
    )
    capsys.readouterr()

    assert return_code == 2
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["passed"] is False
    assert report["decision"] == "stop_before_closed_loop"
    assert len(report["failures"]) == 5


def test_preflight_decision_rejects_validation_binding_mismatch(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    _write_json(
        baseline,
        _preflight_report(adapter="bc0", exact=0, covered_tools=set()),
    )
    _write_json(
        candidate,
        _preflight_report(
            adapter="checkpoint-11",
            exact=5,
            covered_tools={
                "wls_from_path",
                "rollback_state",
                "correct_parameters_from_path",
                "get_topology_context",
            },
            validation_sha256="different",
        ),
    )

    with pytest.raises(ValueError, match="different validation binding"):
        research_dagger_demo.choose_repair_checkpoint(
            baseline,
            [candidate],
            tmp_path / "decision.json",
            required_tools=[
                "wls_from_path",
                "rollback_state",
                "correct_parameters_from_path",
                "get_topology_context",
            ],
            minimum_exact=4,
            minimum_schema_rate=0.95,
            minimum_state_bound_rate=0.90,
            require_baseline_improvement=True,
        )


def test_preflight_decision_rejects_impossible_aggregate_count(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    _write_json(
        baseline,
        _preflight_report(adapter="bc0", exact=0, covered_tools=set()),
    )
    impossible = _preflight_report(
        adapter="checkpoint-impossible",
        exact=5,
        covered_tools={
            "wls_from_path",
            "rollback_state",
            "correct_parameters_from_path",
            "get_topology_context",
        },
    )
    impossible["overall"]["exact_target_match_count"] = 999
    _write_json(candidate, impossible)

    with pytest.raises(
        ValueError,
        match="exact_target_match_count|reconcile|inconsistent",
    ):
        research_dagger_demo.choose_repair_checkpoint(
            baseline,
            [candidate],
            tmp_path / "decision.json",
            required_tools=[
                "wls_from_path",
                "rollback_state",
                "correct_parameters_from_path",
                "get_topology_context",
            ],
            minimum_exact=4,
            minimum_schema_rate=0.95,
            minimum_state_bound_rate=0.90,
            require_baseline_improvement=True,
        )
