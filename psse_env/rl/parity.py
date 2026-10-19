"""Sampling-versus-training parity of the stored rows of a rollout.

The update recomputes the log-probability of every sampled token with a
full-sequence forward pass and compares it with the probability the sampler
recorded when it drew the token (``on_policy_check`` of the update report).
The gap is small on average and large on a few tokens; this diagnostic says
where it comes from.  For every training row of an iteration, built exactly
as the update builds them, it takes the log-probability of each completion
token under four computations:

``record``
    the sampler's recorded sampling probability (the softmax of the processed
    scores ``generate`` drew from);
``decode``
    ``generate``'s cached decode path under the sampling loader, teacher-forced
    through the recorded completion, read from the raw logits of each step
    (the same kernels and cache the sampler used, without the draw);
``forward``
    the full-sequence forward under the sampling loader, as the update's
    :func:`~psse_env.rl.update.completion_logprobs` computes it;
``training``
    the same full-sequence forward under the update's loader (the training
    preparation of the base and the adapter attached trainable), which is the
    clip baseline of the update.

The pairwise gaps, overall and by bucket of the recorded probability,
separate the candidates: ``record`` against ``decode`` is the sampling-time
nondeterminism (and the record's own arithmetic); ``decode`` against
``forward`` the cached decode path against the full forward on one loader;
``forward`` against ``training`` the two loaders.  The worst tokens are
listed with their text so a tail draw (a token sampled at a probability of
1e-6) can be told from a decision token the two passes rank differently.

    python -m psse_env.rl.parity --config configs/rl_hard_screen_bf16_smoke.json \
        --iteration out/smoke_bf16/iteration_000 --output out/smoke_bf16/parity.json
"""

from __future__ import annotations

import argparse
import gc
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

PARITY_CONTRACT = "rl_sampling_parity_v1"
SOURCES = ("record", "decode", "forward", "training")
PAIRS = (
    ("record", "decode"),
    ("decode", "forward"),
    ("forward", "training"),
    ("record", "forward"),
    ("record", "training"),
    ("decode", "training"),
)
#: Buckets of the recorded sampling probability of a token.
BUCKETS = (
    ("p<1e-4", 0.0, 1e-4),
    ("1e-4<=p<1e-2", 1e-4, 1e-2),
    ("1e-2<=p<0.5", 1e-2, 0.5),
    ("0.5<=p<0.99", 0.5, 0.99),
    ("p>=0.99", 0.99, 1.0 + 1e-9),
)
CLIP_THRESHOLD = math.log(1.2)


def bucket_of(probability: float) -> str:
    """The bucket of ``probability`` (the recorded sampling probability of a token)."""

    for name, low, high in BUCKETS:
        if low <= probability < high:
            return name
    return BUCKETS[-1][0] if probability >= BUCKETS[-1][1] else BUCKETS[0][0]


def gap_statistics(first: Sequence[float], second: Sequence[float], *, threshold: float = CLIP_THRESHOLD) -> dict[str, Any]:
    """Mean and largest absolute difference of two log-probability sequences, and how many differ beyond ``threshold``."""

    if len(first) != len(second):
        raise ValueError("the sequences differ in length")
    gaps = [abs(float(a) - float(b)) for a, b in zip(first, second)]
    finite = [g for g in gaps if math.isfinite(g)]
    return {
        "tokens": len(gaps),
        "nonfinite": len(gaps) - len(finite),
        "mean_abs": (sum(finite) / len(finite)) if finite else None,
        "max_abs": max(finite) if finite else None,
        "beyond_threshold": sum(1 for g in finite if g > threshold),
        "threshold": threshold,
    }


class ForcedSequence:
    """A ``generate`` logits processor that forces a fixed token sequence, one token per step."""

    def __init__(self, token_ids: Sequence[int]) -> None:
        self.token_ids = [int(value) for value in token_ids]
        self.step = 0

    def __call__(self, input_ids: Any, scores: Any) -> Any:
        if self.step >= len(self.token_ids):
            raise RuntimeError("generate asked for more tokens than the forced sequence holds")
        forced = scores.new_full(scores.shape, float("-inf"))
        forced[:, self.token_ids[self.step]] = 0.0
        self.step += 1
        return forced


def forced_decode_logprobs(model: Any, row: TrainingRow, *, temperature: float = 1.0, pad_token_id: int | None = None) -> list[float]:
    """Log-probabilities of the row's completion under ``generate``'s cached decode path, teacher-forced.

    The prompt is prefilled as the sampler prefilled it and every completion
    token is forced in turn; the raw logits of each step (``output_logits``,
    before any processor) give the distribution the sampler drew from.
    """

    import torch
    from transformers import LogitsProcessorList

    from psse_env.rl.update import _model_device, _row_tensors

    device = _model_device(model)
    inputs = _row_tensors(row, device, torch)
    n = row.completion_tokens
    if n <= 0:
        raise ValueError("row has no completion tokens")
    prompt = {key: value[:, :-n] for key, value in inputs.items()}
    completion = [int(value) for value in inputs["input_ids"][0, -n:].tolist()]
    kwargs: dict[str, Any] = {}
    if pad_token_id is not None:
        kwargs["pad_token_id"] = int(pad_token_id)
    with torch.inference_mode():
        generated = model.generate(
            **prompt,
            max_new_tokens=n,
            do_sample=False,
            use_cache=True,
            logits_processor=LogitsProcessorList([ForcedSequence(completion)]),
            output_logits=True,
            return_dict_in_generate=True,
            **kwargs,
        )
    produced = [int(value) for value in generated.sequences[0, -len(generated.logits):].tolist()]
    if produced != completion[: len(produced)]:
        raise RuntimeError("the forced decode did not reproduce the recorded completion")
    values: list[float] = []
    for step, logits in enumerate(generated.logits):
        row_logits = logits[0].float() / float(temperature)
        values.append(float(torch.log_softmax(row_logits, dim=-1)[completion[step]]))
    return values


def _finite(values: Sequence[float]) -> list[float]:
    return [float(v) if math.isfinite(float(v)) else float("nan") for v in values]


def summarize_parity(tokens: Sequence[Mapping[str, Any]], *, worst: int = 20) -> dict[str, Any]:
    """Pairwise gap statistics over ``tokens`` (one mapping per token with the source values), overall and by bucket."""

    available = [source for source in SOURCES if any(source in token for token in tokens)]
    pairs = [(a, b) for a, b in PAIRS if a in available and b in available]
    summary: dict[str, Any] = {"tokens": len(tokens), "sources": available, "pairs": {}, "buckets": {}}
    for a, b in pairs:
        rows_ = [t for t in tokens if a in t and b in t]
        summary["pairs"][f"{a}_vs_{b}"] = gap_statistics([t[a] for t in rows_], [t[b] for t in rows_])
    for name, _low, _high in BUCKETS:
        members = [t for t in tokens if t.get("bucket") == name]
        entry: dict[str, Any] = {"tokens": len(members)}
        for a, b in pairs:
            rows_ = [t for t in members if a in t and b in t]
            entry[f"{a}_vs_{b}"] = gap_statistics([t[a] for t in rows_], [t[b] for t in rows_])
        summary["buckets"][name] = entry
    key_pair = ("record", "training") if "training" in available else ("record", "forward")
    if key_pair[1] in available:
        ranked = sorted(
            (t for t in tokens if key_pair[0] in t and key_pair[1] in t),
            key=lambda t: -abs(float(t[key_pair[0]]) - float(t[key_pair[1]])),
        )
        summary["worst_pair"] = f"{key_pair[0]}_vs_{key_pair[1]}"
        summary["worst"] = [dict(t) for t in ranked[:worst]]
    return summary


def _load_config(path: str):
    from psse_env.rl.train import RLConfig

    return RLConfig.load(path)


def _iteration_adapter(iteration: Path, default: str) -> str:
    summary = iteration / "rollout_summary.json"
    if summary.is_file():
        try:
            payload = json.loads(summary.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            payload = {}
        for key in ("adapter", "adapter_in", "policy_adapter"):
            value = payload.get(key)
            if value:
                return str(value)
    return default


def _release(*objects: Any) -> None:
    for item in objects:
        del item
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # pragma: no cover - torch optional in tests
        pass


def run_parity(
    *,
    config_path: str,
    iteration: Path,
    output: Path,
    adapter: str | None = None,
    max_rows: int | None = None,
    skip_decode: bool = False,
    skip_training: bool = False,
    worst: int = 20,
) -> dict[str, Any]:
    import torch

    from psse_env.dagger.research_policy_factory import _load_research_bundle
    from psse_env.rl.rows import build_rows
    from psse_env.rl.train import TrajectoryStore, _load_trainable_model, _prompt_functions, score_groups
    from psse_env.rl.update import completion_logprobs
    from psse_env.sft.gemma_text import resolve_pad_token_id
    from psse_env.sft.training import infer_required_side_input_names

    config = _load_config(config_path)
    iteration = Path(iteration)
    iteration_index = int(iteration.name.split("_")[-1]) if iteration.name.startswith("iteration_") else 0
    adapter = adapter or _iteration_adapter(iteration, config.initial_adapter)
    temperature = float(config.update_settings(iteration_index).temperature)
    started = time.perf_counter()
    store = TrajectoryStore(iteration / "trajectories")
    groups, score_summary = score_groups(config, store)

    bundle, identity = _load_research_bundle(
        adapter_path=adapter, base_model=config.base_model, base_revision=config.base_revision,
        load_in_4bit=config.load_in_4bit, local_files_only=config.local_files_only, trust_remote_code=False,
        prompt_profile=config.prompt_profile, use_cache=False, architecture=config.architecture,
    )
    processor = bundle.processor
    render, prompt_ids = _prompt_functions(config, adapter, processor)
    side_inputs = infer_required_side_input_names(bundle.model, processor, config.base_model)
    rows, rows_report = build_rows(
        groups, render_prompt=render, prompt_token_ids=prompt_ids, max_length=config.max_length,
        required_side_inputs=side_inputs, settings=config.row_settings(iteration_index),
    )
    rows = [row for row in rows if row.sampling_logprobs is not None]
    if max_rows is not None:
        rows = rows[: int(max_rows)]
    pad_token_id = resolve_pad_token_id(processor)
    decoder = processor if callable(getattr(processor, "decode", None)) else getattr(processor, "tokenizer", None)

    tokens: list[dict[str, Any]] = []
    per_row: list[dict[str, Any]] = []
    decode_short = 0
    bundle.model.eval()
    for index, row in enumerate(rows):
        n = row.completion_tokens
        completion = row.input_ids[-n:]
        record = _finite(row.sampling_logprobs or [])
        with torch.no_grad():
            forward = _finite(completion_logprobs(bundle.model, row, temperature=temperature).detach().float().cpu().tolist())
        decode: list[float] | None = None
        if not skip_decode:
            try:
                decode = _finite(forced_decode_logprobs(bundle.model, row, temperature=temperature, pad_token_id=pad_token_id))
            except RuntimeError:
                decode = None
            if decode is not None and len(decode) < n:
                decode_short += 1
                decode = decode + [float("nan")] * (n - len(decode))
        entry = {"row": index, "root": row.root, "sample": row.sample_index, "step": row.step, "tokens": n,
                 "prompt_tokens": len(row.input_ids) - n}
        per_row.append(entry)
        for position in range(n):
            token: dict[str, Any] = {
                "row": index, "root": row.root, "sample": row.sample_index, "step": row.step, "position": position,
                "token_id": int(completion[position]),
                "token": decoder.decode([int(completion[position])]) if decoder is not None else None,
                "record": record[position], "forward": forward[position],
                "bucket": bucket_of(math.exp(record[position]) if math.isfinite(record[position]) else 0.0),
            }
            if decode is not None:
                token["decode"] = decode[position]
            tokens.append(token)
        if (index + 1) % 16 == 0 or index + 1 == len(rows):
            print(json.dumps({"parity_rows_done": index + 1, "rows": len(rows), "seconds": round(time.perf_counter() - started, 1)}), flush=True)
    sampling_seconds = time.perf_counter() - started

    training_identity: dict[str, Any] | None = None
    if not skip_training:
        _release(bundle)
        bundle = None
        model, _processor, _side = _load_trainable_model(config, adapter, reference_adapter=None)
        model.eval()
        training_identity = {"load_in_4bit": bool(config.load_in_4bit), "adapter": str(adapter), "gradient_checkpointing": True}
        cursor = 0
        with torch.no_grad():
            for index, row in enumerate(rows):
                values = _finite(completion_logprobs(model, row, temperature=temperature).detach().float().cpu().tolist())
                for position in range(row.completion_tokens):
                    tokens[cursor]["training"] = values[position]
                    cursor += 1
                if (index + 1) % 16 == 0 or index + 1 == len(rows):
                    print(json.dumps({"parity_training_rows_done": index + 1, "rows": len(rows), "seconds": round(time.perf_counter() - started, 1)}), flush=True)
        _release(model)

    summary = summarize_parity(tokens, worst=worst)
    report = {
        "contract": PARITY_CONTRACT,
        "config": str(config_path),
        "iteration": str(iteration),
        "adapter": str(adapter),
        "temperature": temperature,
        "sampling_loader": {**identity, "load_in_4bit": bool(config.load_in_4bit)},
        "training_loader": training_identity,
        "rollout": {k: score_summary.get(k) for k in ("groups", "informative_groups", "episodes")},
        "rows_report": rows_report,
        "rows_compared": len(rows),
        "decode_short_rows": decode_short,
        "seconds": {"sampling_loader": round(sampling_seconds, 1), "total": round(time.perf_counter() - started, 1)},
        "summary": summary,
        "tokens": tokens,
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    printable = {k: v for k, v in report.items() if k != "tokens"}
    printable["summary"] = {k: v for k, v in summary.items() if k != "worst"}
    print(json.dumps(printable, indent=2))
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--iteration", required=True, help="an iteration directory with trajectories/")
    parser.add_argument("--output", required=True)
    parser.add_argument("--adapter", default=None, help="the adapter that sampled the iteration (default: the rollout summary's, else the config's initial adapter)")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--skip-decode", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--worst", type=int, default=20)
    args = parser.parse_args(argv)
    run_parity(
        config_path=args.config, iteration=Path(args.iteration), output=Path(args.output), adapter=args.adapter,
        max_rows=args.max_rows, skip_decode=args.skip_decode, skip_training=args.skip_training, worst=args.worst,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
