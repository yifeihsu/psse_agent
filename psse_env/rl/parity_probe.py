"""Localize the decode-versus-forward gap of a parity run on its worst rows.

:mod:`psse_env.rl.parity` found the sampler's record equal to ``generate``'s
cached decode path and the two loaders' full forwards equal to each other,
with the whole gap between the cached decode path and the full-sequence
forward, concentrated on the first completion tokens of a few prompts.  This
probe recomputes the first completion token's log-probability of those rows
(and of a few agreeing rows as controls) under variants that differ in one
thing each, so the asymmetry can be placed:

``forward_keep``
    the update's computation (full sequence, ``logits_to_keep``);
``forward_all``
    the full sequence without ``logits_to_keep`` (every logit);
``prompt_only``
    the prompt alone, no cache, last position (causal: must equal
    ``forward_keep`` unless the appended completion leaks backwards);
``prompt_cache``
    the prompt alone with ``use_cache=True`` (the prefill as ``generate``
    runs it, without ``generate``);
``generate_one``
    ``generate`` for one token with ``output_logits`` (the record's path);
``no_side_inputs``
    the full sequence without the zero-filled side inputs
    (``mm_token_type_ids`` and the like);
``no_attention_mask``
    the full sequence without an attention mask;
``prompt_only_no_side``
    the prompt alone without side inputs.

    python -m psse_env.rl.parity_probe --config C --iteration DIR --parity DIR/parity.json --output DIR/parity_probe.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from psse_env.rl.rows import TrainingRow

PROBE_CONTRACT = "rl_sampling_parity_probe_v1"


def select_probe_rows(tokens: Sequence[Mapping[str, Any]], *, worst: int = 8, controls: int = 2, threshold: float = 1.0) -> dict[str, list[tuple[str, int, int]]]:
    """Row keys (root, sample, step) whose first-token record-versus-forward gap exceeds ``threshold``, and agreeing controls."""

    first = [t for t in tokens if int(t.get("position", -1)) == 0 and "record" in t and "forward" in t]
    gaps = {}
    for t in first:
        key = (str(t["root"]), int(t["sample"]), int(t["step"]))
        a, b = float(t["record"]), float(t["forward"])
        gaps[key] = abs(a - b) if math.isfinite(a) and math.isfinite(b) else float("inf")
    bad = sorted((k for k, g in gaps.items() if g > threshold), key=lambda k: -gaps[k])[:worst]
    good = sorted((k for k, g in gaps.items() if g <= threshold), key=lambda k: gaps[k])[:controls]
    return {"bad": bad, "control": good}


def _first_token_logprob(logits_last: Any, token: int, temperature: float, torch_module: Any) -> float:
    row = logits_last.float() / float(temperature)
    return float(torch_module.log_softmax(row, dim=-1)[int(token)])


def probe_row(model: Any, row: TrainingRow, *, temperature: float, pad_token_id: int | None, use_logits_to_keep: bool = True) -> dict[str, Any]:
    """The first completion token's log-probability under each variant, for one row."""

    import torch

    from psse_env.rl.update import _model_device, _row_tensors, completion_logprobs

    device = _model_device(model)
    inputs = _row_tensors(row, device, torch)
    n = row.completion_tokens
    first = int(inputs["input_ids"][0, -n].item())
    prompt = {key: value[:, :-n] for key, value in inputs.items()}
    core = {k: v for k, v in inputs.items() if k in ("input_ids", "attention_mask")}
    prompt_core = {k: v for k, v in prompt.items() if k in ("input_ids", "attention_mask")}
    out: dict[str, Any] = {"first_token_id": first, "prompt_tokens": int(prompt["input_ids"].shape[1]), "completion_tokens": n,
                           "side_inputs": sorted(k for k in inputs if k not in ("input_ids", "attention_mask"))}
    with torch.no_grad():
        values = completion_logprobs(model, row, use_logits_to_keep=use_logits_to_keep, temperature=temperature)
        out["forward_keep"] = float(values[0])
        logits = model(**inputs, use_cache=False).logits
        out["forward_all"] = _first_token_logprob(logits[0, -n - 1], first, temperature, torch)
        logits = model(**prompt, use_cache=False).logits
        out["prompt_only"] = _first_token_logprob(logits[0, -1], first, temperature, torch)
        logits = model(**prompt, use_cache=True).logits
        out["prompt_cache"] = _first_token_logprob(logits[0, -1], first, temperature, torch)
        kwargs = {"pad_token_id": int(pad_token_id)} if pad_token_id is not None else {}
        with torch.inference_mode():
            generated = model.generate(**prompt, max_new_tokens=1, do_sample=False, use_cache=True, output_logits=True,
                                       return_dict_in_generate=True, **kwargs)
        out["generate_one"] = _first_token_logprob(generated.logits[0][0], first, temperature, torch)
        logits = model(**core, use_cache=False).logits
        out["no_side_inputs"] = _first_token_logprob(logits[0, -n - 1], first, temperature, torch)
        logits = model(input_ids=inputs["input_ids"], use_cache=False).logits
        out["no_attention_mask"] = _first_token_logprob(logits[0, -n - 1], first, temperature, torch)
        logits = model(**prompt_core, use_cache=False).logits
        out["prompt_only_no_side"] = _first_token_logprob(logits[0, -1], first, temperature, torch)
    return out


def probe_row_pass2(model: Any, row: TrainingRow, *, temperature: float, pad_token_id: int | None, seed: int = 0) -> dict[str, Any]:
    """Second pass: does the leak depend on the appended tokens' content, their count, the attention mask or the attention kernel?

    ``completion_masked``: the full sequence with the completion's attention mask zeroed (a causal kernel that
    honours padding must give the prompt-only value); ``prompt_plus_pad`` / ``prompt_plus_random``: the prompt
    followed by as many pad or random text tokens as the completion has (a length-only effect gives the leaked
    value, a content effect does not); ``prompt_plus_k``: the prompt followed by the first ``k`` completion tokens;
    ``eager_forward``: the full sequence under eager attention when the model can switch kernels.
    """

    import torch

    from psse_env.rl.update import _model_device, _row_tensors

    device = _model_device(model)
    inputs = _row_tensors(row, device, torch)
    n = row.completion_tokens
    first = int(inputs["input_ids"][0, -n].item())
    prompt_ids = inputs["input_ids"][:, :-n]
    length = int(prompt_ids.shape[1])
    out: dict[str, Any] = {"attn_implementation": str(getattr(getattr(model, "config", None), "_attn_implementation", None))}

    def value(ids: Any, mask: Any) -> float:
        logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
        return _first_token_logprob(logits[0, length - 1], first, temperature, torch)

    with torch.no_grad():
        mask = torch.ones_like(inputs["input_ids"])
        mask[:, length:] = 0
        out["completion_masked"] = value(inputs["input_ids"], mask)
        pad = int(pad_token_id) if pad_token_id is not None else 0
        padded = torch.cat([prompt_ids, torch.full((1, n), pad, dtype=prompt_ids.dtype, device=device)], dim=1)
        out["prompt_plus_pad"] = value(padded, torch.ones_like(padded))
        generator = torch.Generator(device="cpu").manual_seed(seed)
        random_ids = torch.randint(1000, 30000, (1, n), generator=generator).to(device=device, dtype=prompt_ids.dtype)
        randomized = torch.cat([prompt_ids, random_ids], dim=1)
        out["prompt_plus_random"] = value(randomized, torch.ones_like(randomized))
        for k in (1, 2, 4, 8):
            if k > n:
                break
            partial = inputs["input_ids"][:, : length + k]
            out[f"prompt_plus_{k}"] = value(partial, torch.ones_like(partial))
        setter = getattr(model, "set_attn_implementation", None)
        current = out["attn_implementation"]
        if callable(setter) and current not in ("eager", "None"):
            try:
                setter("eager")
                out["eager_forward"] = value(inputs["input_ids"], torch.ones_like(inputs["input_ids"]))
                prompt_mask = torch.ones_like(prompt_ids)
                logits = model(input_ids=prompt_ids, attention_mask=prompt_mask, use_cache=False).logits
                out["eager_prompt_only"] = _first_token_logprob(logits[0, -1], first, temperature, torch)
            except Exception as exc:  # pragma: no cover - live kernel switch
                out["eager_forward_error"] = f"{type(exc).__name__}: {exc}"
            finally:
                try:
                    setter(current)
                except Exception:
                    pass
    return out


def run_probe(*, config_path: str, iteration: Path, parity_path: Path, output: Path, worst: int = 8, controls: int = 2,
              pass2: bool = False) -> dict[str, Any]:
    import torch

    from psse_env.dagger.research_policy_factory import _load_research_bundle
    from psse_env.rl.parity import _iteration_adapter, _load_config
    from psse_env.rl.rows import build_rows
    from psse_env.rl.train import TrajectoryStore, _prompt_functions, score_groups
    from psse_env.sft.gemma_text import resolve_pad_token_id
    from psse_env.sft.training import infer_required_side_input_names

    config = _load_config(config_path)
    iteration = Path(iteration)
    iteration_index = int(iteration.name.split("_")[-1]) if iteration.name.startswith("iteration_") else 0
    adapter = _iteration_adapter(iteration, config.initial_adapter)
    temperature = float(config.update_settings(iteration_index).temperature)
    parity = json.loads(Path(parity_path).read_text(encoding="utf-8"))
    selection = select_probe_rows(parity["tokens"], worst=worst, controls=controls)
    wanted = {key: label for label in ("bad", "control") for key in selection[label]}
    started = time.perf_counter()

    store = TrajectoryStore(iteration / "trajectories")
    groups, _summary = score_groups(config, store)
    bundle, identity = _load_research_bundle(
        adapter_path=adapter, base_model=config.base_model, base_revision=config.base_revision,
        load_in_4bit=config.load_in_4bit, local_files_only=config.local_files_only, trust_remote_code=False,
        prompt_profile=config.prompt_profile, use_cache=False, architecture=config.architecture,
    )
    processor = bundle.processor
    render, prompt_ids = _prompt_functions(config, adapter, processor)
    side_inputs = infer_required_side_input_names(bundle.model, processor, config.base_model)
    rows, _report = build_rows(
        groups, render_prompt=render, prompt_token_ids=prompt_ids, max_length=config.max_length,
        required_side_inputs=side_inputs, settings=config.row_settings(iteration_index),
    )
    pad_token_id = resolve_pad_token_id(processor)
    decoder = processor if callable(getattr(processor, "decode", None)) else getattr(processor, "tokenizer", None)
    bundle.model.eval()
    results: list[dict[str, Any]] = []
    for row in rows:
        key = (row.root, int(row.sample_index), int(row.step))
        if key not in wanted:
            continue
        entry = {"label": wanted[key], "root": row.root, "sample": row.sample_index, "step": row.step,
                 "record": float(row.sampling_logprobs[0]) if row.sampling_logprobs else None}
        entry.update(probe_row(bundle.model, row, temperature=temperature, pad_token_id=pad_token_id))
        if pass2:
            entry.update(probe_row_pass2(bundle.model, row, temperature=temperature, pad_token_id=pad_token_id))
        entry["first_token"] = decoder.decode([entry["first_token_id"]]) if decoder is not None else None
        results.append(entry)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in entry.items() if k != "side_inputs"}), flush=True)
    report = {
        "contract": PROBE_CONTRACT, "config": str(config_path), "iteration": str(iteration), "adapter": adapter,
        "sampling_loader": {**identity, "load_in_4bit": bool(config.load_in_4bit)}, "temperature": temperature,
        "required_side_inputs": list(side_inputs), "selection": {k: [list(x) for x in v] for k, v in selection.items()},
        "rows": results, "seconds": round(time.perf_counter() - started, 1),
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--iteration", required=True)
    parser.add_argument("--parity", required=True, help="the parity.json of that iteration")
    parser.add_argument("--output", required=True)
    parser.add_argument("--worst", type=int, default=8)
    parser.add_argument("--controls", type=int, default=2)
    parser.add_argument("--pass2", action="store_true", help="add the content/length/mask/kernel variants")
    args = parser.parse_args(argv)
    run_probe(config_path=args.config, iteration=Path(args.iteration), parity_path=Path(args.parity), output=Path(args.output),
              worst=args.worst, controls=args.controls, pass2=args.pass2)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
