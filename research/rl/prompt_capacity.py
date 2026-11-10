"""How long a prompt a card holds: one generation and one update row at each length (workplan of 2026-11-10, stage 1).

    python -m research.rl.prompt_capacity --config $WORK/configs/arm_b_complete.json \\
        --lengths 8192 16384 24576 32768 40960 49152 --output $WORK/out/prompt_capacity/a100

The model's window is not what bounds a prompt here (262,144 positions for
gemma-4-12B-it); the card is.  The series' prompt limit was set by
extrapolating the first arm's memory at bounded prompts of at most about
13,000 tokens, so before an arm trains on complete histories this reads the
real figures on the card the job got:

* **generation**: the prompt's prefill and a few new tokens with the cache
  on, as a rollout or evaluation step does;
* **update row**: the frozen reference's pass (when the config has a KL
  term), then the trained adapter's forward and backward under gradient
  checkpointing, as :func:`psse_env.rl.update.policy_gradient_update` does
  for one sampled row.

Each is run at every length in turn, shortest first, on synthetic tokens,
with the peak device memory and the time; a length the device cannot hold
ends that series.  The report names the longest length that fitted and the
longest whose peak leaves the chosen headroom, which is the figure to hold
``max_input_tokens`` against.  Two rollout workers share a card: the report
also says what two generations at once would need.

Nothing here changes a run.  The driver's own safety net is separate: a
complete-history prompt that does not fit at generation is refused and
counted (``PromptLimitError``); an update row that does not fit ends the
update process with the device's own error.
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

CAPACITY_CONTRACT = "prompt_capacity_v1"
DEFAULT_LENGTHS = (8192, 16384, 24576, 32768, 40960, 49152)
#: Completion tokens of the synthetic update row (a tool call is a few dozen tokens).
ROW_COMPLETION_TOKENS = 48
#: New tokens the generation probe decodes after the prefill.
GENERATED_TOKENS = 8
MIB = 1024 * 1024


class DeviceMemory:
    """Peak device memory between ``reset`` and ``read``; every figure is ``None`` without a CUDA device."""

    def __init__(self, torch_module: Any | None = None) -> None:
        self.torch = torch_module
        self.cuda = bool(torch_module is not None and torch_module.cuda.is_available())

    def describe(self) -> dict[str, Any]:
        if not self.cuda:
            return {"name": None, "total_mib": None}
        properties = self.torch.cuda.get_device_properties(0)
        return {"name": properties.name, "total_mib": int(properties.total_memory // MIB)}

    def held(self) -> int | None:
        return int(self.torch.cuda.memory_allocated() // MIB) if self.cuda else None

    def reset(self) -> None:
        if self.cuda:
            import gc

            gc.collect()
            self.torch.cuda.empty_cache()
            self.torch.cuda.reset_peak_memory_stats()

    def read(self) -> dict[str, Any]:
        if not self.cuda:
            return {"peak_allocated_mib": None, "peak_reserved_mib": None}
        return {"peak_allocated_mib": int(self.torch.cuda.max_memory_allocated() // MIB),
                "peak_reserved_mib": int(self.torch.cuda.max_memory_reserved() // MIB)}


def is_out_of_memory(error: BaseException) -> bool:
    from psse_env.dagger.research_policy_factory import is_out_of_memory_error

    return is_out_of_memory_error(error)


def probe(lengths: Sequence[int], steps: Mapping[str, Callable[[int], None]], memory: DeviceMemory) -> dict[str, list[dict[str, Any]]]:
    """Run every step at every length, shortest first; a step the device cannot hold at a length is not tried at a longer one."""

    results: dict[str, list[dict[str, Any]]] = {name: [] for name in steps}
    for name, step in steps.items():
        for length in sorted({int(value) for value in lengths}):
            memory.reset()
            started = time.perf_counter()
            fitted = True
            try:
                step(length)
            except Exception as exc:
                if not is_out_of_memory(exc):
                    raise
                fitted = False
            seconds = time.perf_counter() - started
            # Read before the release: the peak of a failed attempt is where it failed.
            entry = {"tokens": length, "fits": fitted, "seconds": round(seconds, 2), **memory.read()}
            results[name].append(entry)
            memory.reset()
            if not fitted:
                break
    return results


def summarize(results: Mapping[str, Sequence[Mapping[str, Any]]], device: Mapping[str, Any], *, weights_mib: int | None,
              headroom: float = 0.9) -> dict[str, Any]:
    """The longest length each step fitted, and the longest whose peak stays under ``headroom`` of the card."""

    total = device.get("total_mib")
    summary: dict[str, Any] = {}
    for name, entries in results.items():
        fitted = [entry for entry in entries if entry["fits"]]
        within = [entry for entry in fitted if total is None or entry["peak_reserved_mib"] is None
                  or entry["peak_reserved_mib"] <= headroom * total]
        failed = [entry["tokens"] for entry in entries if not entry["fits"]]
        # Memory per prompt token beyond the weights, from the longest length that fitted.
        longest = fitted[-1] if fitted else None
        per_token = None
        if longest is not None and longest["peak_allocated_mib"] is not None and weights_mib is not None:
            per_token = (longest["peak_allocated_mib"] - weights_mib) * 1024.0 / longest["tokens"]  # KiB per token
        summary[name] = {
            "longest_that_fits": fitted[-1]["tokens"] if fitted else None,
            "longest_within_headroom": within[-1]["tokens"] if within else None,
            "first_that_does_not_fit": failed[0] if failed else None,
            "kib_per_token_beyond_the_weights": per_token,
        }
    generation = results.get("generation") or []
    two_workers = None
    if total is not None and weights_mib is not None:
        # Two workers each hold the weights and their own prompt's working memory.
        fitting = [entry["tokens"] for entry in generation if entry["fits"] and entry["peak_reserved_mib"] is not None
                   and 2 * entry["peak_reserved_mib"] <= total]
        two_workers = fitting[-1] if fitting else None
    limits = [entry["longest_within_headroom"] for entry in summary.values()]
    return {
        "steps": summary,
        "two_generation_workers_at_once_up_to": two_workers,
        "headroom": headroom,
        # What the prompt limit can be held against on this card: every step, with the headroom.
        "supported_prompt_tokens": min(limits) if limits and all(value is not None for value in limits) else None,
    }


def markdown(report: Mapping[str, Any]) -> str:
    device, summary = report["device"], report["summary"]
    lines = [f"# Prompt capacity on {device.get('name') or 'this machine'}", "",
             f"Card memory {device.get('total_mib')} MiB; held after loading the model {report.get('weights_mib')} MiB; "
             f"config limit {report['prompt_limits']['max_input_tokens']} tokens; model window {report['prompt_limits']['model_context_window']}.", ""]
    for name, entries in report["results"].items():
        lines += [f"## {name.replace('_', ' ').capitalize()}", "", "| Prompt tokens | Fits | Peak allocated, MiB | Peak reserved, MiB | Seconds |",
                  "| --- | --- | --- | --- | --- |"]
        lines += [f"| {entry['tokens']} | {'yes' if entry['fits'] else 'no'} | {entry['peak_allocated_mib']} | {entry['peak_reserved_mib']} | "
                  f"{entry['seconds']} |" for entry in entries]
        step = summary["steps"][name]
        per_token = step["kib_per_token_beyond_the_weights"]
        lines += ["", f"Longest that fits: {step['longest_that_fits']}; within {int(summary['headroom'] * 100)}% of the card: "
                      f"{step['longest_within_headroom']}; memory beyond the weights: "
                      f"{'n/a' if per_token is None else f'{per_token:.0f} KiB per prompt token'}.", ""]
    lines += [f"**Supported prompt length on this card: {summary['supported_prompt_tokens']} tokens** (the shorter of the two steps, with the "
              f"headroom). Two generation workers at once: up to {summary['two_generation_workers_at_once_up_to']} tokens each.", ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------- the device steps


def synthetic_ids(processor: Any, length: int) -> list[int]:
    """``length`` token ids of ordinary prompt text: a JSON-like block tiled, after the tokenizer's start token."""

    tokenizer = getattr(processor, "tokenizer", processor)
    block = tokenizer('{"bus": 4, "vm_pu": 1.0123, "normalized_residual": -3.87, "branch": "4-7", "p_from_mw": 28.07, "status": "closed"}, ',
                      add_special_tokens=False)["input_ids"]
    start = [int(tokenizer.bos_token_id)] if getattr(tokenizer, "bos_token_id", None) is not None else []
    body = (block * (length // len(block) + 1))[: max(0, length - len(start))]
    return start + [int(value) for value in body]


def device_steps(config: Any, adapter: str, *, skip: Sequence[str] = ()) -> tuple[dict[str, Callable[[int], None]], DeviceMemory, int | None]:
    """The generation and update-row steps on the config's model, loaded as the update loads it."""

    import torch

    from psse_env.rl import train
    from psse_env.rl.rows import build_step_row
    from psse_env.rl.update import completion_logprobs, peft_reference_logprobs

    settings = config.update_settings(0)
    reference_adapter = config.initial_adapter if settings.beta_kl > 0.0 else None
    model, processor, side_inputs = train._load_trainable_model(config, adapter, reference_adapter=reference_adapter)
    memory = DeviceMemory(torch)
    memory.reset()
    weights = memory.held()
    reference = (peft_reference_logprobs(model, use_logits_to_keep=settings.use_logits_to_keep, temperature=settings.temperature)
                 if reference_adapter else None)
    tokenizer = getattr(processor, "tokenizer", processor)
    pad = tokenizer.pad_token_id if getattr(tokenizer, "pad_token_id", None) is not None else 0
    device = next(model.parameters()).device

    def generation(length: int) -> None:
        ids = torch.tensor([synthetic_ids(processor, length)], dtype=torch.long, device=device)
        inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
        for name in side_inputs:
            inputs.setdefault(name, torch.zeros_like(ids))
        model.eval()
        # no_grad, not inference_mode: the model is loaded as the update loads it, and its input embeddings are
        # set to require gradients for gradient checkpointing, which an inference tensor cannot take.
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=GENERATED_TOKENS, do_sample=False, use_cache=True, pad_token_id=pad)

    def update_row(length: int) -> None:
        ids = synthetic_ids(processor, length)
        row = build_step_row(root="capacity", sample_index=0, step=0, advantage=1.0, prompt_ids=ids[:-ROW_COMPLETION_TOKENS],
                             completion_ids=ids[-ROW_COMPLETION_TOKENS:], max_length=length, required_side_inputs=side_inputs)
        model.train()
        try:
            anchor = reference(row).detach() if reference is not None else None
            logprobs = completion_logprobs(model, row, use_logits_to_keep=settings.use_logits_to_keep, temperature=settings.temperature)
            loss = -logprobs.mean()
            if anchor is not None:
                delta = anchor.to(logprobs.device) - logprobs
                loss = loss + settings.beta_kl * (torch.exp(delta) - delta - 1.0).mean()
            loss.backward()
        finally:
            model.zero_grad(set_to_none=True)

    steps = {"generation": generation, "update_row": update_row}
    return {name: step for name, step in steps.items() if name not in skip}, memory, weights


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--adapter", default=None, help="default: the config's start adapter")
    parser.add_argument("--lengths", nargs="+", type=int, default=list(DEFAULT_LENGTHS))
    parser.add_argument("--headroom", type=float, default=0.9, help="the share of the card a supported length may reserve")
    parser.add_argument("--skip", nargs="*", default=[], choices=("generation", "update_row"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    from psse_env.rl.train import RLConfig, apply_prompt_limits

    config = RLConfig.load(args.config)
    limits = apply_prompt_limits(config)
    window = limits.get("model_context_window")
    lengths = [length for length in args.lengths if window is None or length <= window]
    steps, memory, weights = device_steps(config, args.adapter or config.initial_adapter, skip=args.skip)
    results = probe(lengths, steps, memory)
    device = memory.describe()
    report = {"contract": CAPACITY_CONTRACT, "config": str(args.config), "adapter": args.adapter or config.initial_adapter,
              "precision": "4bit" if config.load_in_4bit else "bf16", "device": device, "weights_mib": weights,
              "prompt_limits": limits, "lengths": lengths, "results": results,
              "summary": summarize(results, device, weights_mib=weights, headroom=args.headroom)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    text = markdown(report)
    args.output.with_suffix(".md").write_text(text, encoding="utf-8", newline="\n")
    print(text, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
