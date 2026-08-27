"""Reproduce the Round-1 evaluation failure-pattern aggregates.

The evaluation report omits usable scenario identifiers.  This script therefore
joins its 65 per-episode rows by list position to the exact ordered scenario file
used by the launcher, records hashes for both inputs, and performs the published
aggregations in an in-memory SQLite table.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any


ARTIFACT_ROOT = Path(__file__).resolve().parents[1]
EVAL_PATH = ARTIFACT_ROOT / "out" / "eval_round1.json"
SCENARIO_PATH = ARTIFACT_ROOT / "source" / "eval_scenarios.json"
OUTPUT_PATH = Path(__file__).with_name("round1_failure_analysis.json")


HEADLINE_SQL = """
SELECT
    COUNT(*) AS total_roots,
    SUM(fault_bearing) AS fault_bearing_roots,
    SUM(CASE WHEN fault_bearing = 1 AND outcome = 'resolved' THEN 1 ELSE 0 END)
        AS fault_bearing_resolved,
    SUM(CASE WHEN outcome = 'resolved' THEN 1 ELSE 0 END) AS terminal_resolved,
    SUM(CASE WHEN outcome = 'operator_escalation' THEN 1 ELSE 0 END)
        AS escalation_roots,
    AVG(CASE WHEN outcome = 'operator_escalation' THEN 1.0 ELSE 0.0 END)
        AS escalation_rate,
    SUM(CASE WHEN outcome = 'horizon' THEN 1 ELSE 0 END) AS horizon_roots,
    AVG(CASE WHEN outcome = 'horizon' THEN 1.0 ELSE 0.0 END) AS horizon_rate,
    SUM(invalid_actions) AS all_invalid_actions,
    SUM(CASE WHEN outcome = 'horizon' THEN invalid_actions ELSE 0 END)
        AS horizon_invalid_actions,
    1.0 * SUM(CASE WHEN outcome = 'horizon' THEN invalid_actions ELSE 0 END)
        / NULLIF(SUM(invalid_actions), 0) AS horizon_invalid_share
FROM joined_eval
""".strip()


OUTCOME_SQL = """
SELECT
    outcome,
    COUNT(*) AS episodes,
    1.0 * COUNT(*) / (SELECT COUNT(*) FROM joined_eval) AS episode_rate,
    SUM(invalid_actions) AS invalid_actions,
    SUM(steps) AS steps,
    SUM(fault_bearing) AS fault_bearing_episodes
FROM joined_eval
GROUP BY outcome
ORDER BY CASE outcome
    WHEN 'resolved' THEN 1
    WHEN 'operator_escalation' THEN 2
    ELSE 3
END
""".strip()


FAMILY_SQL = """
SELECT
    family,
    COUNT(*) AS roots,
    SUM(CASE WHEN outcome = 'resolved' THEN 1 ELSE 0 END) AS resolved,
    SUM(CASE WHEN outcome = 'operator_escalation' THEN 1 ELSE 0 END)
        AS escalations,
    SUM(CASE WHEN outcome = 'horizon' THEN 1 ELSE 0 END) AS horizons,
    AVG(CASE WHEN outcome = 'horizon' THEN 1.0 ELSE 0.0 END) AS horizon_rate,
    SUM(invalid_actions) AS invalid_actions,
    SUM(steps) AS steps,
    1.0 * SUM(invalid_actions) / NULLIF(SUM(steps), 0) AS invalid_rate,
    CAST(AVG(error_cardinality) AS INTEGER) AS fault_cardinality_mode
FROM joined_eval
GROUP BY family
ORDER BY horizon_rate DESC, invalid_actions DESC, family
""".strip()


CATEGORY_SQL = """
SELECT
    category,
    COUNT(*) AS roots,
    SUM(CASE WHEN outcome = 'resolved' THEN 1 ELSE 0 END) AS resolved,
    SUM(CASE WHEN outcome = 'operator_escalation' THEN 1 ELSE 0 END)
        AS escalations,
    SUM(CASE WHEN outcome = 'horizon' THEN 1 ELSE 0 END) AS horizons,
    SUM(invalid_actions) AS invalid_actions,
    SUM(steps) AS steps,
    1.0 * SUM(invalid_actions) / NULLIF(SUM(steps), 0) AS invalid_rate
FROM joined_eval
GROUP BY category
ORDER BY horizons DESC, invalid_actions DESC, category
""".strip()


CARDINALITY_SQL = """
SELECT
    error_cardinality,
    COUNT(*) AS roots,
    SUM(CASE WHEN outcome = 'resolved' THEN 1 ELSE 0 END) AS resolved,
    SUM(CASE WHEN outcome = 'operator_escalation' THEN 1 ELSE 0 END)
        AS escalations,
    SUM(CASE WHEN outcome = 'horizon' THEN 1 ELSE 0 END) AS horizons,
    SUM(invalid_actions) AS invalid_actions,
    SUM(steps) AS steps,
    1.0 * SUM(invalid_actions) / NULLIF(SUM(steps), 0) AS invalid_rate
FROM joined_eval
GROUP BY error_cardinality
ORDER BY error_cardinality
""".strip()


HORIZON_SQL = """
SELECT
    scenario_index,
    root,
    family,
    category,
    error_cardinality,
    measurement_errors,
    parameter_errors,
    topology_errors,
    invalid_actions,
    steps
FROM joined_eval
WHERE outcome = 'horizon'
ORDER BY invalid_actions DESC, scenario_index
""".strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _errors(truth: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = truth.get(key)
    if value is None:
        return []
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return [value] if isinstance(value, dict) else []


def _human_family(value: str) -> str:
    return {
        "measurement+parameter": "Measurement + parameter",
        "measurement+topology": "Measurement + topology",
        "multi_measurement": "Multiple measurements",
        "no_error": "No error",
        "parameter": "Parameter only",
        "topology": "Topology only",
    }.get(value, value)


def _human_category(value: str) -> str:
    return {
        "dagger1_development": "DAgger development",
        "forced_error_recovery": "Forced recovery",
        "partial_success_retention": "Partial-success retention",
        "standard_success": "Standard success",
    }.get(value, value)


def _query(connection: sqlite3.Connection, sql: str) -> list[dict[str, Any]]:
    cursor = connection.execute(sql)
    columns = [description[0] for description in cursor.description]
    return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]


def main() -> None:
    evaluation = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    scenarios = json.loads(SCENARIO_PATH.read_text(encoding="utf-8"))
    episodes = evaluation["per_episode"]
    if len(episodes) != len(scenarios) or len(episodes) != 65:
        raise ValueError("Expected the same ordered 65 rows in evaluation and suite")
    if any(episode.get("scenario_id") is not None for episode in episodes):
        raise ValueError("Positional join assumption changed: scenario_id is no longer null")

    joined: list[dict[str, Any]] = []
    for index, (episode, scenario) in enumerate(zip(episodes, scenarios, strict=True)):
        grouping = scenario.get("grouping") or {}
        execution = scenario.get("execution") or {}
        truth = (scenario.get("audit") or {}).get("truth") or {}
        measurement_errors = _errors(truth, "true_measurement_errors")
        parameter_errors = _errors(truth, "true_parameter_errors")
        topology_errors = _errors(truth, "true_topology_errors")
        error_count = len(measurement_errors) + len(parameter_errors) + len(topology_errors)
        declared_count = int(grouping.get("error_cardinality", error_count))
        if error_count != declared_count:
            raise ValueError(f"Truth cardinality mismatch at scenario index {index}")
        outcome = "horizon" if episode["horizon_truncated"] else episode["terminal_outcome"]
        joined.append(
            {
                "scenario_index": index,
                "root": grouping.get("root_scenario_id") or execution.get("scenario_id"),
                "physical_root_fingerprint": grouping.get("physical_root_fingerprint"),
                "family": _human_family(str(grouping.get("scenario_family"))),
                "category": _human_category(str(grouping.get("eval_category"))),
                "source_tier": grouping.get("source_tier"),
                "error_cardinality": error_count,
                "measurement_errors": len(measurement_errors),
                "parameter_errors": len(parameter_errors),
                "topology_errors": len(topology_errors),
                "fault_bearing": int(error_count > 0),
                "outcome": outcome,
                "invalid_actions": int(episode["invalid_actions"]),
                "steps": int(episode["steps"]),
                "first_error": episode.get("first_error"),
            }
        )

    if len({row["root"] for row in joined}) != 65:
        raise ValueError("Expected 65 distinct reconstructed root IDs")
    if len({row["physical_root_fingerprint"] for row in joined}) != 65:
        raise ValueError("Expected 65 distinct physical roots")

    connection = sqlite3.connect(":memory:")
    connection.execute(
        """
        CREATE TABLE joined_eval (
            scenario_index INTEGER,
            root TEXT,
            physical_root_fingerprint TEXT,
            family TEXT,
            category TEXT,
            source_tier TEXT,
            error_cardinality INTEGER,
            measurement_errors INTEGER,
            parameter_errors INTEGER,
            topology_errors INTEGER,
            fault_bearing INTEGER,
            outcome TEXT,
            invalid_actions INTEGER,
            steps INTEGER,
            first_error TEXT
        )
        """
    )
    connection.executemany(
        """
        INSERT INTO joined_eval VALUES (
            :scenario_index, :root, :physical_root_fingerprint, :family, :category,
            :source_tier, :error_cardinality, :measurement_errors, :parameter_errors,
            :topology_errors, :fault_bearing, :outcome, :invalid_actions, :steps,
            :first_error
        )
        """,
        joined,
    )

    result = {
        "inputs": {
            "evaluation": {"path": str(EVAL_PATH), "sha256": _sha256(EVAL_PATH)},
            "scenarios": {"path": str(SCENARIO_PATH), "sha256": _sha256(SCENARIO_PATH)},
            "join": "list position; evaluation scenario_id is null in all 65 rows",
        },
        "sql": {
            "headline": HEADLINE_SQL,
            "outcomes": OUTCOME_SQL,
            "families": FAMILY_SQL,
            "categories": CATEGORY_SQL,
            "cardinality": CARDINALITY_SQL,
            "horizon_roots": HORIZON_SQL,
        },
        "datasets": {
            "headline": _query(connection, HEADLINE_SQL),
            "outcome_summary": _query(connection, OUTCOME_SQL),
            "family_summary": _query(connection, FAMILY_SQL),
            "category_summary": _query(connection, CATEGORY_SQL),
            "cardinality_summary": _query(connection, CARDINALITY_SQL),
            "horizon_roots": _query(connection, HORIZON_SQL),
            "joined_eval": joined,
        },
    }

    headline = result["datasets"]["headline"][0]
    assert headline["total_roots"] == 65
    assert headline["fault_bearing_roots"] == 61
    assert headline["fault_bearing_resolved"] == 0
    assert headline["terminal_resolved"] == 4
    assert headline["escalation_roots"] == 49
    assert headline["horizon_roots"] == 12
    assert headline["all_invalid_actions"] == 149
    assert headline["horizon_invalid_actions"] == 135

    OUTPUT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT_PATH), "headline": headline}, indent=2))


if __name__ == "__main__":
    main()
