"""Score a triage fine-tune: the LLM's first action on every scored prompt, through the pipeline's own policy.

    python -m research.classifier_triage.llm_score --adapter OUT/lora --score DATA/score.jsonl --output OUT/scores.json

Runs on the cluster (it loads the base model and the LoRA adapter).  Each
prompt of ``score.jsonl`` (``llm_dataset``) is the model-visible state after
the opening WLS; the script calls the research policy on that state exactly
as closed-loop evaluation does (same processor, rendering, greedy decoding
and action validation), with the triage system prompt installed in place of
the profile paragraph so scoring reads the prompt the fine-tune was trained
on.  The output lists each row's first tool; ``benchmark --llm-scores``
turns it into the request and first-family studies.  Finished ids are kept,
so a preempted job resumes.

An LLM's first action is a decision, not a score, so no threshold is fitted
and the calibration rows are not needed: by default the test rows are scored
first, then ``--probe-per-cell`` probe rows of each kind and background.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.classifier_triage.llm_dataset import FIRST_ACTION, triage_system_prompt  # noqa: E402

#: Canonical tool name to triage decision (the canonical surface keeps these four names).
DECISION_OF_TOOL = {tool: decision for decision, tool in FIRST_ACTION.items()}
SAVE_EVERY = 25


def state_of(row: Mapping[str, Any]) -> dict[str, Any]:
    """The model-visible state of a scored row, after checking it carries the triage prompt."""
    system, user = row["messages"][0], row["messages"][1]
    if system["role"] != "system" or system["content"] != triage_system_prompt():
        raise ValueError(f"row {row.get('id')} does not carry the triage system prompt")
    return json.loads(user["content"])["state"]


def decision_of(action: Any) -> str:
    """``request``, a balanced family, ``other`` (another valid tool) or ``invalid``."""
    if not isinstance(action, Mapping):
        return "invalid"
    tool = str(action.get("tool") or action.get("name") or "")
    if tool in DECISION_OF_TOOL:
        return DECISION_OF_TOOL[tool]
    return "invalid" if tool in ("", "__invalid_action__") else "other"


def score_rows(rows: Sequence[Mapping[str, Any]], act: Callable[[Mapping[str, Any]], Any], output: Path, *,
               log=print) -> dict[str, Any]:
    """Call ``act`` on every row's state; resumable through ``output``."""
    done: dict[str, Any] = {}
    if output.is_file():
        done = json.loads(output.read_text(encoding="utf-8")).get("rows") or {}
    started = time.perf_counter()
    fresh = 0
    for row in rows:
        row_id = str(row["id"])
        if row_id in done:
            continue
        began = time.perf_counter()
        try:
            action = act(state_of(row))
            error = None
        except Exception as exc:  # an unparseable generation is a decision too: an invalid action
            action, error = None, f"{type(exc).__name__}: {str(exc)[:200]}"
        tool = str(action.get("tool") or "") if isinstance(action, Mapping) else None
        done[row_id] = {"decision": decision_of(action), "tool": tool, "error": error,
                        "seconds": round(time.perf_counter() - began, 3)}
        fresh += 1
        if fresh % SAVE_EVERY == 0:
            _save(output, done)
            log(f"[llm-score] {len(done)}/{len(rows)} rows, {time.perf_counter() - started:.0f} s")
    _save(output, done)
    return {"rows": done}


def select_rows(rows: Sequence[Mapping[str, Any]], labels: Sequence[Mapping[str, Any]], probe_per_cell: int) -> list[Mapping[str, Any]]:
    """Test rows first, then up to ``probe_per_cell`` probe rows per kind and background (smallest ids)."""
    label = {str(item["id"]): item for item in labels}
    test = [row for row in rows if label[str(row["id"])]["kind"] != "probe" and label[str(row["id"])]["split"] == "test"]
    cells: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in sorted((r for r in rows if label[str(r["id"])]["kind"] == "probe"), key=lambda r: str(r["id"])):
        item = label[str(row["id"])]
        cells.setdefault((str(item.get("probe_kind")), str(item["family"]).rsplit("_", 1)[-1]), []).append(row)
    probe = [row for key in sorted(cells) for row in cells[key][:probe_per_cell]]
    return test + probe


def _save(output: Path, done: Mapping[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps({"rows": done}), encoding="utf-8")
    temporary.replace(output)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--score", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-model")
    parser.add_argument("--labels", help="score_labels.json (default: next to --score)")
    parser.add_argument("--probe-per-cell", type=int, default=100, help="probe rows per kind and background")
    parser.add_argument("--all-rows", action="store_true", help="score every row, calibration included")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    with Path(args.score).open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not args.all_rows:
        labels_path = Path(args.labels) if args.labels else Path(args.score).with_name("score_labels.json")
        rows = select_rows(rows, json.loads(labels_path.read_text(encoding="utf-8")), args.probe_per_cell)
    if args.limit:
        rows = rows[:args.limit]

    import psse_env.dagger.research_policy_factory as factory

    # The policy builds its system prompt from the observation's profile; the
    # triage rows were trained on the triage paragraph instead.
    factory.system_prompt_for_observation = lambda _prompt, _observation: triage_system_prompt()
    policy = factory.research_gemma_policy_factory(args.adapter, base_model=args.base_model)
    result = score_rows(rows, policy.act_model_observation, Path(args.output))
    decisions: dict[str, int] = {}
    for item in result["rows"].values():
        decisions[item["decision"]] = decisions.get(item["decision"], 0) + 1
    print(f"[llm-score] wrote {args.output}: {len(result['rows'])} rows, decisions {decisions}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
