"""The probability a model gives named actions on the prompts it read: teacher-forced, nothing is generated.

    python -m research.minimal_env.score_actions --adapter ADAPTER --inputs scoring_inputs_dev.jsonl \\
        [--inputs scoring_inputs_train.jsonl] --output-dir OUT [--shard S/N]

Each input line is one state of ``research.minimal_env.stopping_states``: the observation an episode's last action
was chosen on and the actions to ask about there (``own``: the model's own last action; ``repairing``: the first
actions of the truth-built continuation from that state; ``other_finish``: the other ``finish``).  For each action
the prompt is rendered as the policy renders it (``rl_train.prompt_functions``), the action as the supervised pass
taught it (``rl_train.target_function``), and the log-probability of every completion token is read in one forward
pass of the model the RL update loads (``rl_train.load_trainable_model``, ``psse_env.rl.update.completion_logprobs``),
at the sampling temperature, under ``torch.no_grad``.  An action's probability is the product over its tokens: the
chance that one sample at that temperature writes exactly that call.

One file per state under ``OUT/states``, so a preempted job loses at most the state it was on; ``--shard S/N``
scores every N-th state from S and writes no summary; a call without ``--shard`` scores what is left and writes
``OUT/summary.json``.  The summary reads, per split, how much probability the model puts on a step toward the
repair at the state where it stopped, and how likely one of K samples is to take such a step there.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCORES_CONTRACT = "minimal_action_scores_v1"
ROLES = ("own", "repairing", "other_finish")
#: The sample counts the summary reads "one of K samples takes a step toward the repair" at.
SAMPLE_COUNTS = (1, 8, 32)
THRESHOLDS = (0.5, 0.1, 0.01, 0.001)


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name) or "state"


def state_key(item: Mapping[str, Any]) -> str:
    return f"{item['split']}_{item['case']}_{int(item['step'])}"


def read_inputs(paths: Sequence[Path]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("contract") != "minimal_scoring_inputs_v1":
                raise ValueError(f"{path}: not a scoring input of research.minimal_env.stopping_states")
            key = state_key(item)
            if key in seen:
                raise ValueError(f"{path}: state {key} appears twice")
            seen.add(key)
            items.append(item)
    return items


def parse_shard(value: str | None) -> tuple[int, int] | None:
    if not value:
        return None
    match = re.fullmatch(r"(\d+)/(\d+)", value.strip())
    if not match or int(match.group(2)) < 1 or int(match.group(1)) >= int(match.group(2)):
        raise ValueError(f"--shard must be S/N with 0 <= S < N, got {value!r}")
    return int(match.group(1)), int(match.group(2))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def score_state(item: Mapping[str, Any], *, view: Callable[[Mapping[str, Any]], Mapping[str, Any]],
                render: Callable[[Mapping[str, Any]], str], prompt_ids: Callable[[str], tuple[Sequence[int], int]],
                render_target: Callable[[Mapping[str, Any], Mapping[str, Any]], Sequence[int]],
                logprobs: Callable[[Sequence[int], Sequence[int]], Sequence[float]],
                tokens: Callable[[Sequence[int]], Sequence[str]] | None = None) -> dict[str, Any]:
    """Every action of one state: its completion tokens, their log-probabilities and the action's probability."""
    observation = view(item["observation"])
    ids, original = prompt_ids(render(observation))
    ids = [int(value) for value in ids]
    out: dict[str, Any] = {"contract": SCORES_CONTRACT, "case": item["case"], "split": item["split"], "class": item.get("class"),
                           "step": int(item["step"]), "point": str(item.get("point") or "stop"), "prompt_tokens": len(ids),
                           "prompt_cut": int(original) != len(ids), "actions": []}
    for entry in item["actions"]:
        record: dict[str, Any] = {"role": entry["role"], "action": entry["action"]}
        try:
            completion = [int(value) for value in render_target(observation, entry["action"])]
        except ValueError as exc:
            record.update({"unrenderable": f"{type(exc).__name__}: {exc}"[:300]})
            out["actions"].append(record)
            continue
        values = [float(value) for value in logprobs(ids, completion)]
        if len(values) != len(completion):
            raise RuntimeError(f"{len(values)} log-probabilities for {len(completion)} completion tokens")
        total = sum(values)
        record.update({"tokens": len(completion), "logprob": total, "probability": math.exp(total),
                       "token_logprobs": [round(value, 5) for value in values]})
        if tokens is not None:
            record["token_text"] = list(tokens(completion))
        out["actions"].append(record)
    return out


def score_all(items: Sequence[Mapping[str, Any]], *, directory: Path, shard: tuple[int, int] | None,
              scorer: Callable[[Mapping[str, Any]], dict[str, Any]],
              progress: Callable[[Mapping[str, Any]], None] | None = None) -> int:
    """Score every state of the shard that has no file yet.  Returns how many were scored."""
    pending = [item for index, item in enumerate(items)
               if (shard is None or index % shard[1] == shard[0]) and not (directory / f"{_safe(state_key(item))}.json").is_file()]
    for number, item in enumerate(pending, start=1):
        started = time.perf_counter()
        result = scorer(item)
        result["seconds"] = round(time.perf_counter() - started, 3)
        _write_json(directory / f"{_safe(state_key(item))}.json", result)
        if progress is not None:
            progress({"state": state_key(item), "done": number, "of": len(pending), "seconds": result["seconds"],
                      "prompt_tokens": result["prompt_tokens"]})
    return len(pending)


def _quantiles(values: Sequence[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)

    def at(share: float) -> float:
        return ordered[min(len(ordered) - 1, int(share * len(ordered)))]

    return {"p10": at(0.10), "p25": at(0.25), "median": at(0.5), "p75": at(0.75), "p90": at(0.90)}


def role_probability(result: Mapping[str, Any], role: str) -> float | None:
    """The probability of the role's actions together (they are different calls, so their probabilities add)."""
    values = [float(action["probability"]) for action in result["actions"] if action["role"] == role and "probability" in action]
    return sum(values) if values else None


def summarize(results: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    by_split: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for result in results:
        # The stopping state and the first step off the truth's path are read apart.
        grouped[f"{result['split']}:{result.get('point') or 'stop'}"].append(result)
    for split, items in sorted(grouped.items()):
        counts: Counter = Counter()
        probabilities: dict[str, list[float]] = {role: [] for role in ROLES}
        reach: dict[int, float] = {count: 0.0 for count in SAMPLE_COUNTS}
        for result in items:
            counts["states"] += 1
            counts["prompt_cut"] += int(bool(result.get("prompt_cut")))
            counts["unrenderable_actions"] += sum(1 for action in result["actions"] if "unrenderable" in action)
            for role in ROLES:
                value = role_probability(result, role)
                if value is not None:
                    probabilities[role].append(value)
            repairing = role_probability(result, "repairing")
            if repairing is None:
                counts["states_without_a_repairing_action"] += 1
                continue
            for threshold in THRESHOLDS:
                counts[f"repairing_at_least_{threshold:g}"] += int(repairing >= threshold)
            for count in SAMPLE_COUNTS:
                reach[count] += 1.0 - (1.0 - min(max(repairing, 0.0), 1.0)) ** count
        by_split[split] = {
            "counts": dict(sorted(counts.items())),
            "probability": {role: _quantiles(values) for role, values in probabilities.items()},
            "expected_states_where_one_of_k_samples_steps_toward_the_repair": {str(count): round(value, 2) for count, value in reach.items()},
        }
    return {"contract": SCORES_CONTRACT, "by_split": by_split}


def read_results(directory: Path) -> list[dict[str, Any]]:
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(Path(directory).glob("*.json"))]


def model_scorer(adapter: str, output_dir: Path, *, model_choice: str, max_length: int, max_input_tokens: int,
                 table_versions: int, max_actions: int, temperature: float) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
    """The model the RL update loads, with the policy's renderers: a scorer of one state."""
    import torch

    from psse_env.rl.rows import build_step_row
    from psse_env.rl.train import apply_prompt_limits
    from psse_env.rl.update import completion_logprobs
    from research.minimal_env.rl_train import (
        MinimalRLConfig,
        load_trainable_model,
        model_view,
        prompt_functions,
        target_function,
    )

    config = MinimalRLConfig(output_dir=str(output_dir), training_suite="(not read)", initial_adapter=str(adapter),
                             model_choice=model_choice, load_in_4bit=False, max_actions=int(max_actions),
                             table_versions=int(table_versions), max_length=int(max_length),
                             max_input_tokens=int(max_input_tokens), temperature=float(temperature))
    config.validate()
    limits = apply_prompt_limits(config)
    print(f"[score-actions] prompt limits {json.dumps(limits, sort_keys=True)}", flush=True)
    model, processor, side_inputs = load_trainable_model(config, str(adapter))
    model.eval()
    render, prompt_ids = prompt_functions(config, str(adapter), processor)
    render_target = target_function(config, str(adapter), processor)
    tokenizer = getattr(processor, "tokenizer", processor)

    def logprobs(prompt: Sequence[int], completion: Sequence[int]) -> list[float]:
        row = build_step_row(root="score", sample_index=0, step=0, advantage=0.0, prompt_ids=prompt,
                             completion_ids=completion, max_length=int(config.max_length), required_side_inputs=side_inputs)
        if row.prompt_truncated:
            raise RuntimeError("a scored prompt does not fit max_length")
        with torch.no_grad():
            values = completion_logprobs(model, row, use_logits_to_keep=True, temperature=float(config.temperature))
        return [float(value) for value in values.detach().float().cpu().tolist()]

    def tokens(ids: Sequence[int]) -> list[str]:
        return list(tokenizer.convert_ids_to_tokens([int(value) for value in ids]))

    def scorer(item: Mapping[str, Any]) -> dict[str, Any]:
        return score_state(item, view=model_view, render=render, prompt_ids=prompt_ids, render_target=render_target,
                           logprobs=logprobs, tokens=tokens)

    return scorer


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--inputs", required=True, type=Path, action="append")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--shard", default=None)
    parser.add_argument("--model-choice", default="12b")
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--max-input-tokens", type=int, default=32768)
    parser.add_argument("--max-actions", type=int, default=16)
    parser.add_argument("--table-versions", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args(argv)
    shard = parse_shard(args.shard)
    items = read_inputs(args.inputs)
    directory = args.output_dir / "states"
    started = time.time()
    pending = [item for index, item in enumerate(items) if (shard is None or index % shard[1] == shard[0])
               and not (directory / f"{_safe(state_key(item))}.json").is_file()]
    print(f"[score-actions] {len(items)} states in the inputs, {len(pending)} left for this call", flush=True)
    if pending:
        scorer = model_scorer(str(args.adapter), args.output_dir, model_choice=args.model_choice, max_length=args.max_length,
                              max_input_tokens=args.max_input_tokens, table_versions=args.table_versions,
                              max_actions=args.max_actions, temperature=args.temperature)

        def progress(record: Mapping[str, Any]) -> None:
            print(f"[score-actions{'' if shard is None else f' shard {shard[0]}/{shard[1]}'}] {json.dumps(dict(record), sort_keys=True)}",
                  flush=True)

        score_all(items, directory=directory, shard=shard, scorer=scorer, progress=progress)
    if shard is not None:
        print(json.dumps({"shard": list(shard), "seconds": round(time.time() - started, 1)}))
        return 0
    results = read_results(directory)
    missing = len(items) - len(results)
    if missing:
        raise SystemExit(f"{missing} states have no scores after the run")
    summary = {**summarize(results), "adapter": str(args.adapter), "inputs": [str(path) for path in args.inputs],
               "settings": {"temperature": args.temperature, "max_input_tokens": args.max_input_tokens, "precision": "bf16"}}
    _write_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary["by_split"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
