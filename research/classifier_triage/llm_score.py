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

``--probabilities`` reads the same fine-tune as a scored classifier instead
(decision G1 thresholds a score): for every row, the probability that greedy
decoding starts each of the four first actions, taken from the next-token
distribution at the token where the four tool calls part (one forward pass
of the prompt; the tokens the calls share are fed as given).  The request
probability is then thresholded on the calibration split like every other
classifier's score, so this mode scores the calibration rows too
(``benchmark --llm-probabilities``).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.classifier_triage.llm_dataset import FIRST_ACTION, triage_system_prompts  # noqa: E402

#: Canonical tool name to triage decision (the canonical surface keeps these four names).
DECISION_OF_TOOL = {tool: decision for decision, tool in FIRST_ACTION.items()}
SAVE_EVERY = 25


def triage_variant_of(row: Mapping[str, Any]) -> str:
    """The prompt variant whose system prompt the row carries."""
    system = row["messages"][0]
    if system["role"] == "system":
        for variant, prompt in triage_system_prompts().items():
            if system["content"] == prompt:
                return variant
    raise ValueError(f"row {row.get('id')} does not carry a triage system prompt")


def state_of(row: Mapping[str, Any]) -> dict[str, Any]:
    """The model-visible state of a scored row, after checking it carries a triage prompt."""
    triage_variant_of(row)
    return json.loads(row["messages"][1]["content"])["state"]


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


def select_rows(rows: Sequence[Mapping[str, Any]], labels: Sequence[Mapping[str, Any]], probe_per_cell: int, *,
                calibration: bool = False) -> list[Mapping[str, Any]]:
    """Test rows first, then up to ``probe_per_cell`` probe rows per kind and background (smallest ids).

    ``calibration`` puts the calibration rows after the test rows (a thresholded reading needs them).
    """
    label = {str(item["id"]): item for item in labels}
    test = [row for row in rows if label[str(row["id"])]["kind"] != "probe" and label[str(row["id"])]["split"] == "test"]
    held = [row for row in rows if label[str(row["id"])]["kind"] != "probe" and label[str(row["id"])]["split"] == "calibration"]
    cells: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in sorted((r for r in rows if label[str(r["id"])]["kind"] == "probe"), key=lambda r: str(r["id"])):
        item = label[str(row["id"])]
        cells.setdefault((str(item.get("probe_kind")), str(item["family"]).rsplit("_", 1)[-1]), []).append(row)
    probe = [row for key in sorted(cells) for row in cells[key][:probe_per_cell]]
    return test + (held if calibration else []) + probe


# ------------------------------------------------------------- probabilities

def candidate_log_probabilities(prefix: Sequence[int], targets: Mapping[str, Sequence[int]],
                                step: Callable[[list[int]], Any]) -> dict[str, float]:
    """Log-probability that decoding from ``prefix`` starts each candidate token sequence.

    ``step(ids)`` returns the next-token log-probabilities after ``ids``,
    indexable by token id.  Tokens every remaining candidate shares are fed as
    given, a candidate is charged the token where it leaves the others, and
    once it is alone the rest of its tokens are taken as certain.  One call of
    ``step`` when all candidates part at the same position.
    """
    result: dict[str, float] = {}

    def walk(context: list[int], remaining: dict[str, list[int]], base: float) -> None:
        if len(remaining) == 1:
            result[next(iter(remaining))] = base
            return
        sequences = list(remaining.values())
        shared = 0
        while all(len(s) > shared for s in sequences) and len({s[shared] for s in sequences}) == 1:
            shared += 1
        if any(len(s) <= shared for s in sequences):
            raise ValueError("a candidate's tokens are a prefix of another's")
        context = context + sequences[0][:shared]
        log_p = step(context)
        groups: dict[int, dict[str, list[int]]] = {}
        for name, sequence in remaining.items():
            groups.setdefault(int(sequence[shared]), {})[name] = sequence[shared + 1:]
        for token, members in groups.items():
            walk(context + [token], members, base + float(log_p[token]))

    walk([int(t) for t in prefix], {name: [int(t) for t in sequence] for name, sequence in targets.items()}, 0.0)
    return result


class FirstActionScorer:
    """The fine-tune's probability of each triage first action, through the policy's own prompt rendering."""

    def __init__(self, bundle: Any) -> None:
        from psse_env.dagger.preliminary_e2b_eval import canonical_prompt_tool_schemas
        from psse_env.sft.gemma_text import get_stop_token_ids, resolve_pad_token_id
        from psse_env.sft.training import infer_required_side_input_names

        self.bundle = bundle
        self.tools = canonical_prompt_tool_schemas()
        self.required = infer_required_side_input_names(bundle.model, bundle.processor, bundle.model_id)
        self.stop_ids = get_stop_token_ids(bundle.processor)
        self.pad_token_id = resolve_pad_token_id(bundle.processor)
        self._targets: dict[str, dict[str, list[int]]] = {}
        self.meta: dict[str, Any] = {"rows": 0, "training_renders_checked": 0, "prompt_differs_from_training_render": 0,
                                     "user_text_differs_from_row": 0, "forward_passes": 0}

    def _step(self, ids: list[int]) -> Any:
        import torch
        from psse_env.dagger.release_factories import _model_input_device

        device = _model_input_device(self.bundle.model)
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        inputs = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
        for name in self.required:
            inputs.setdefault(name, torch.zeros_like(input_ids))
        with torch.inference_mode():
            generated = self.bundle.model.generate(
                **inputs, max_new_tokens=1, do_sample=False, use_cache=True, eos_token_id=self.stop_ids,
                pad_token_id=self.pad_token_id, output_logits=True, output_scores=True, return_dict_in_generate=True)
        logits = getattr(generated, "logits", None) or generated.scores
        self.meta["forward_passes"] += 1
        return torch.log_softmax(logits[0][0].float(), dim=-1).cpu()

    def _call_tokens(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], alias: str,
                     prompt_ids: list[int]) -> dict[str, list[int]]:
        """Token ids of the four tool calls as the trainer renders them (they depend on the state alias only)."""
        from psse_env.sft.gates import prepare_example

        targets: dict[str, list[int]] = {}
        for decision, tool in FIRST_ACTION.items():
            call = {"role": "assistant", "content": "", "tool_calls": [{
                "type": "function", "id": "call_0", "function": {"name": tool, "arguments": {"case_path": alias}}}]}
            example = prepare_example({"messages": messages + [call], "tools": tools}, self.bundle.processor,
                                      max_length=1 << 20, row_label=f"triage:{decision}")
            first = next(index for index, label in enumerate(example.labels) if label != -100)
            targets[decision] = [int(t) for t in example.input_ids[first:]]
            # The trainer's prompt render against the policy's: the same tokens unless the two paths differ.
            self.meta["training_renders_checked"] += 1
            self.meta["prompt_differs_from_training_render"] += int([int(t) for t in example.input_ids[:first]] != prompt_ids)
        decode = getattr(self.bundle.processor, "decode", None) or self.bundle.processor.tokenizer.decode
        self.meta.setdefault("candidate_tokens", {})[alias] = {
            name: [decode([t]) for t in sequence] for name, sequence in targets.items()}
        self.meta.setdefault("prompt_tokens_first_row", len(prompt_ids))
        return targets

    def __call__(self, row: Mapping[str, Any]) -> dict[str, Any]:
        from psse_env.dagger.dataset_builder import tool_schemas_for_observation, validate_policy_payload
        from psse_env.sft.gemma_text import render_eval_text, tokenize_rendered_text

        state = state_of(row)
        payload = {"state": state}
        validate_policy_payload(payload)
        tools = tool_schemas_for_observation(self.tools, state)
        user = json.dumps(payload, sort_keys=True, allow_nan=False)
        messages = [{"role": "system", "content": row["messages"][0]["content"]}, {"role": "user", "content": user}]
        # The prompt exactly as the policy renders and tokenizes it for a greedy decision.
        rendered = render_eval_text(self.bundle.processor, messages, tools, enable_thinking=False,
                                    inject_empty_thought_channel=False)
        prompt_ids = [int(t) for t in tokenize_rendered_text(self.bundle.processor, rendered)["input_ids"][0].tolist()]
        alias = str(state["active_state_id"])
        if alias not in self._targets:
            self._targets[alias] = self._call_tokens(messages, tools, alias, prompt_ids)
        log_p = candidate_log_probabilities(prompt_ids, self._targets[alias], self._step)
        self.meta["rows"] += 1
        self.meta["user_text_differs_from_row"] += int(user != row["messages"][1]["content"])
        return {"p": {name: math.exp(value) for name, value in log_p.items()}}


def probability_rows(rows: Sequence[Mapping[str, Any]], scorer: Callable[[Mapping[str, Any]], dict[str, Any]], output: Path, *,
                     meta: Callable[[], Mapping[str, Any]] | None = None, log=print) -> dict[str, Any]:
    """Call ``scorer`` on every row; resumable through ``output``."""
    done: dict[str, Any] = {}
    if output.is_file():
        done = json.loads(output.read_text(encoding="utf-8")).get("rows") or {}
    started = time.perf_counter()
    fresh = 0

    def save() -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(".tmp")
        temporary.write_text(json.dumps({"rows": done, "meta": dict(meta()) if meta else {}}), encoding="utf-8")
        temporary.replace(output)

    for row in rows:
        row_id = str(row["id"])
        if row_id in done:
            continue
        began = time.perf_counter()
        item = scorer(row)
        done[row_id] = {**item, "seconds": round(time.perf_counter() - began, 3)}
        fresh += 1
        if fresh % SAVE_EVERY == 0:
            save()
            log(f"[llm-score] probabilities {len(done)}/{len(rows)} rows, {time.perf_counter() - started:.0f} s")
    save()
    return {"rows": done}


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
    parser.add_argument("--probabilities", action="store_true",
                        help="write the probability of each first action (calibration rows included) instead of decisions")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    with Path(args.score).open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not args.all_rows:
        labels_path = Path(args.labels) if args.labels else Path(args.score).with_name("score_labels.json")
        rows = select_rows(rows, json.loads(labels_path.read_text(encoding="utf-8")), args.probe_per_cell,
                           calibration=args.probabilities)
    if args.limit:
        rows = rows[:args.limit]

    import psse_env.dagger.research_policy_factory as factory

    # The policy builds its system prompt from the observation's profile; the
    # triage rows were trained on their variant's triage paragraph instead.
    variants = {triage_variant_of(row) for row in rows}
    if len(variants) != 1:
        raise ValueError(f"scored rows carry {len(variants)} prompt variants: {sorted(variants)}")
    system_prompt = rows[0]["messages"][0]["content"]
    factory.system_prompt_for_observation = lambda _prompt, _observation: system_prompt
    print(f"[llm-score] prompt variant {next(iter(variants))}, {len(rows)} rows")

    if args.probabilities:
        bundle, _ = factory._load_research_bundle(
            adapter_path=args.adapter, base_model=args.base_model, base_revision=None, load_in_4bit=True,
            local_files_only=True, trust_remote_code=False, prompt_profile=None, use_cache=True)
        scorer = FirstActionScorer(bundle)
        result = probability_rows(rows, scorer, Path(args.output), meta=lambda: scorer.meta)
        print(f"[llm-score] wrote {args.output}: {len(result['rows'])} rows, {json.dumps(scorer.meta)[:1500]}")
        return 0

    policy = factory.research_gemma_policy_factory(args.adapter, base_model=args.base_model)
    result = score_rows(rows, policy.act_model_observation, Path(args.output))
    decisions: dict[str, int] = {}
    for item in result["rows"].values():
        decisions[item["decision"]] = decisions.get(item["decision"], 0) + 1
    print(f"[llm-score] wrote {args.output}: {len(result['rows'])} rows, decisions {decisions}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
