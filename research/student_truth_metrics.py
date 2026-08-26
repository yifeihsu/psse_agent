"""Physical-truth metrics for student checkpoints (review point 3).

Walks each traced evaluation episode's candidate lifecycle (correction opens a
candidate; a later successful commit accepts it; a successful rollback
discards it) and scores the ACCEPTED corrections against the scenario's
privileged truth.  Produces the physical level of the three-level metric
decomposition for every checkpoint whose eval carries per-step actions.
"""

import json
from pathlib import Path

SCRATCH = Path(
    r"C:/Users/Holiday/AppData/Local/Temp/claude/"
    r"C--Users-Holiday-Documents-ChatGPT-PSSE-Agent/"
    r"92729157-788a-4c24-ac77-8b1bd2e55183/scratchpad"
)

CHECKPOINTS = {
    "E2B R1 (guarded proxy)": "eval_round1_e2b_guarded.json",
    "E2B R2 clean": "eval_round2_clean_e2b.json",
    "E2B R2 upweighted": "eval_round2_upw_e2b.json",
    "12B R1": "eval_round1_12b_actions.json",
    "12B R1 (guarded)": "eval_round1_12b_guarded.json",
}
CORRECTIONS = ("correct_measurements", "correct_parameters", "correct_topology")


def parse_arguments(action):
    try:
        return json.loads(action.get("arguments") or "{}")
    except (json.JSONDecodeError, TypeError):
        return {}


def measurement_indices(arguments):
    indices = set()
    group = arguments.get("suspect_group")
    if isinstance(group, list):
        indices.update(int(i) for i in group if isinstance(i, int))
    updates = arguments.get("measurement_updates")
    if isinstance(updates, dict):
        for key in updates:
            try:
                indices.add(int(key))
            except (TypeError, ValueError):
                pass
    return indices


def parameter_lines(arguments):
    return {
        int(arguments[key])
        for key in ("line_index", "line_index1")
        if isinstance(arguments.get(key), int)
    }


def accepted_corrections(actions):
    accepted_m, accepted_p = set(), set()
    open_m, open_p = set(), set()
    candidate_open = False
    for action in actions:
        tool, status = action.get("tool"), action.get("status")
        if tool in CORRECTIONS and status == "success":
            arguments = parse_arguments(action)
            open_m, open_p = measurement_indices(arguments), parameter_lines(arguments)
            candidate_open = True
        elif tool == "commit_state" and status == "success" and candidate_open:
            accepted_m |= open_m
            accepted_p |= open_p
            open_m, open_p, candidate_open = set(), set(), False
        elif tool == "rollback_state" and status == "success" and candidate_open:
            open_m, open_p, candidate_open = set(), set(), False
    return accepted_m, accepted_p


def main() -> int:
    scenarios = json.load(open(SCRATCH / "research" / "eval_scenarios.json", encoding="utf-8"))
    truths = []
    for scenario in scenarios:
        truth = (scenario.get("audit") or {}).get("truth") or {}
        truths.append(
            (
                {int(e["index"]) for e in truth.get("true_measurement_errors") or [] if "index" in e},
                {int(e["line_index1"]) for e in truth.get("true_parameter_errors") or [] if "line_index1" in e},
            )
        )

    header = f"{'checkpoint':<24} {'m-recall':>8} {'p-recall':>8} {'false':>5} {'exact':>7}"
    print(header)
    report = {}
    for label, name in CHECKPOINTS.items():
        path = SCRATCH / "research" / name
        if not path.is_file():
            print(f"{label:<24} (no trace file)")
            continue
        data = json.load(open(path, encoding="utf-8"))
        hits_m = hits_p = total_m = total_p = false = exact = 0
        for (true_m, true_p), episode in zip(truths, data["per_episode"]):
            acc_m, acc_p = accepted_corrections(episode.get("actions") or [])
            hits_m += len(true_m & acc_m)
            hits_p += len(true_p & acc_p)
            total_m += len(true_m)
            total_p += len(true_p)
            false += len(acc_m - true_m) + len(acc_p - true_p)
            if (
                true_m <= acc_m
                and true_p <= acc_p
                and not (acc_m - true_m)
                and not (acc_p - true_p)
            ):
                exact += 1
        entry = {
            "measurement_recall": hits_m / total_m if total_m else None,
            "parameter_recall": hits_p / total_p if total_p else None,
            "false_corrections": false,
            "episodes_all_errors_exactly_corrected": exact,
        }
        report[label] = entry
        print(
            f"{label:<24} {hits_m/total_m:>8.1%} {hits_p/total_p:>8.1%} "
            f"{false:>5} {exact:>4}/65"
        )
    print()
    print("teacher reference:        m-recall 53.2%  p-recall 100%  false 0  exact 47/65")
    json.dump(report, open(SCRATCH / "research" / "student_truth_metrics.json", "w"), indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
