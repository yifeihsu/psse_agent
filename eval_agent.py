

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_BASE_MODEL    = "unsloth/gpt-oss-20b-unsloth-bnb-4bit"
DEFAULT_ADAPTER_PATH  = "/scratch/hk4488/psse_agent/outputs/gpt_oss_sft_v2/lora"
DEFAULT_TEST_FILE     = "/home/hk4488/psse_agent/data/split_test.jsonl"
DEFAULT_OUTPUT_JSON   = "/scratch/hk4488/psse_agent/outputs/eval_results.json"
DEFAULT_CKPT_FILE     = "/scratch/hk4488/psse_agent/outputs/eval_checkpoint.jsonl"
DEFAULT_MAX_NEW_TOKENS = 512


FORCED_PREFIX = '<|message|>final{"has_error":'


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate PSSE agent on test split")
    p.add_argument("--base-model",     default=DEFAULT_BASE_MODEL)
    p.add_argument("--adapter-path",   default=DEFAULT_ADAPTER_PATH)
    p.add_argument("--test-file",      default=DEFAULT_TEST_FILE)
    p.add_argument("--output-json",    default=DEFAULT_OUTPUT_JSON)
    p.add_argument("--checkpoint-file", default=DEFAULT_CKPT_FILE,
                   help="JSONL file written after each sample for preemption recovery")
    p.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    p.add_argument("--max-seq-length", type=int, default=4096)
    p.add_argument("--load-in-4bit",   action="store_true", default=True)
    p.add_argument("--no-load-in-4bit", dest="load_in_4bit", action="store_false")
    p.add_argument("--wandb",          action="store_true", default=False)
    p.add_argument("--limit",          type=int, default=0,
                   help="Evaluate only first N samples (0 = all)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def load_checkpoint(ckpt_path: Path) -> dict[int, dict]:
    """Load already-evaluated samples from the checkpoint JSONL.
    Returns a dict keyed by sample index.
    Rows where pred is None are skipped so they get re-evaluated on resume."""
    done: dict[int, dict] = {}
    if not ckpt_path.exists():
        return done
    for line in ckpt_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            if row.get("pred") is None:
                continue  # treat failed predictions as not done
            done[row["idx"]] = row
        except (json.JSONDecodeError, KeyError):
            pass
    return done


def append_checkpoint(ckpt_path: Path, row: dict) -> None:
    """Append a single result row to the checkpoint JSONL (atomic-ish)."""
    with ckpt_path.open("a") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()


# ---------------------------------------------------------------------------
# Ground-truth extraction
# ---------------------------------------------------------------------------

def extract_ground_truth(messages: list[dict]) -> str | None:
    for msg in reversed(messages):
        if msg.get("role") == "assistant" and msg.get("content"):
            raw = msg["content"]
            raw = re.sub(r"^```[a-z]*\s*", "", raw.strip())
            raw = re.sub(r"\s*```$", "", raw)
            try:
                data = json.loads(raw)
                if "verdict" in data and isinstance(data["verdict"], dict):
                    return data["verdict"].get("error_family")
                return data.get("error_family")
            except json.JSONDecodeError:
                m = re.search(r'"error_family"\s*:\s*"([^"]+)"', raw)
                if m:
                    return m.group(1)
    return None


# ---------------------------------------------------------------------------
# Build inference prompt
# ---------------------------------------------------------------------------

def build_prompt_messages(messages: list[dict]) -> list[dict]:
    cutoff = len(messages)
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if msg.get("role") == "assistant" and msg.get("content"):
            cutoff = i
            break
    return messages[:cutoff]


# ---------------------------------------------------------------------------
# Parse model output
# ---------------------------------------------------------------------------

def extract_prediction(generated_text: str) -> dict | None:
    text = re.sub(r"^```[a-z]*\s*", "", generated_text.strip())
    text = re.sub(r"\s*```$", "", text)

    
    final_match = re.search(r"(?:^|\b)final(\{)", text)
    if final_match:
        text = text[final_match.start(1):]

    # Try full parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Scan for first complete {...} block
    depth, start = 0, -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start != -1:
                candidate = text[start : i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    pass

    
    verdict_match = re.search(r'"verdict"\s*:\s*(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})', text)
    if verdict_match:
        try:
            verdict = json.loads(verdict_match.group(1))
            return {"verdict": verdict}
        except json.JSONDecodeError:
            pass

    
    fam_match = re.search(
        r'"error_family"\s*:\s*(?:"([^"]+)"|\[\s*"([^"]+)")',
        text,
    )
    if fam_match:
        return {"error_family": fam_match.group(1) or fam_match.group(2)}

    return None


# ---------------------------------------------------------------------------
# Aggregate + print metrics from a list of result rows
# ---------------------------------------------------------------------------

def compute_and_print_metrics(results: list[dict], total_samples: int) -> dict:
    per_class: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        gt = r["gt"]
        per_class[gt]["total"] += 1
        if r["correct"]:
            per_class[gt]["correct"] += 1

    total   = len(results)
    correct = sum(r["correct"] for r in results)
    overall_acc = correct / total if total > 0 else 0.0

    confs     = [r["confidence"] for r in results if r["confidence"] is not None]
    mean_conf = sum(confs) / len(confs) if confs else None
    min_conf  = min(confs) if confs else None
    max_conf  = max(confs) if confs else None

    print("\n" + "=" * 60)
    print(f"OVERALL ACCURACY: {correct}/{total} (of {total_samples} total) = {overall_acc*100:.2f}%")
    print("=" * 60)

    # ── Build confusion matrix ───────────────────────────────────────────────
    families = sorted(per_class.keys())
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in results:
        if r["gt"] and r["pred"]:
            confusion[r["gt"]][r["pred"]] += 1

   
    pred_counts: dict[str, int] = defaultdict(int)
    for r in results:
        if r["pred"]:
            pred_counts[r["pred"]] += 1

    print(f"\n{'Class':<25}  {'Prec':>6}  {'Recall':>6}  {'F1':>6}  {'Support':>8}")
    print("-" * 60)
    class_accs, class_prec, class_rec, class_f1 = {}, {}, {}, {}
    for family in families:
        tp = per_class[family]["correct"]
        support = per_class[family]["total"]
        rec  = tp / support if support > 0 else 0.0
        prec = tp / pred_counts[family] if pred_counts[family] > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        class_accs[family] = rec
        class_prec[family] = prec
        class_rec[family]  = rec
        class_f1[family]   = f1
        print(f"  {family:<25}  {prec*100:>5.1f}%  {rec*100:>5.1f}%  {f1*100:>5.1f}%  {support:>8}")

    # Macro averages
    macro_prec = sum(class_prec.values()) / len(class_prec) if class_prec else 0.0
    macro_rec  = sum(class_rec.values())  / len(class_rec)  if class_rec  else 0.0
    macro_f1   = sum(class_f1.values())   / len(class_f1)   if class_f1   else 0.0
    print("-" * 60)
    print(f"  {'macro avg':<25}  {macro_prec*100:>5.1f}%  {macro_rec*100:>5.1f}%  {macro_f1*100:>5.1f}%  {total:>8}")

    if mean_conf is not None:
        print(f"\nConfidence — mean={mean_conf:.4f}  min={min_conf:.4f}  max={max_conf:.4f}")

    # null-prediction rate (model produced no parseable output)
    n_null = sum(1 for r in results if r["pred"] is None)
    print(f"Null predictions (parse failure): {n_null}/{total} ({n_null/total*100:.1f}%)")

    print("\nConfusion matrix (rows=GT, cols=pred):")
    print(f"{'':>25}" + "".join(f"{f[:10]:>13}" for f in families))
    for gt_fam in families:
        print(f"{gt_fam:<25}" + "".join(f"{confusion[gt_fam][p]:>13}" for p in families))

    return {
        "overall_accuracy":   overall_acc,
        "correct":            correct,
        "total":              total,
        "macro_precision":    macro_prec,
        "macro_recall":       macro_rec,
        "macro_f1":           macro_f1,
        "null_predictions":   n_null,
        "per_class_accuracy": class_accs,
        "per_class_precision": class_prec,
        "per_class_recall":   class_rec,
        "per_class_f1":       class_f1,
        "per_class_counts":   {k: dict(v) for k, v in per_class.items()},
        "mean_confidence":    mean_conf,
        "min_confidence":     min_conf,
        "max_confidence":     max_conf,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # 1. Load test samples
    # ------------------------------------------------------------------
    test_path = Path(args.test_file)
    if not test_path.exists():
        test_path = Path(__file__).resolve().parent / "split_test.jsonl"
    if not test_path.exists():
        sys.exit(f"[error] test file not found: {args.test_file}")

    samples = [json.loads(l) for l in test_path.read_text().splitlines() if l.strip()]
    if args.limit > 0:
        samples = samples[: args.limit]
    print(f"Loaded {len(samples)} test samples from {test_path}")

    # ------------------------------------------------------------------
    # 2. Resume from checkpoint if available
    # ------------------------------------------------------------------
    ckpt_path = Path(args.checkpoint_file)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    done = load_checkpoint(ckpt_path)
    if done:
        print(f"Resuming from checkpoint: {len(done)}/{len(samples)} samples already done.")
    else:
        print("No checkpoint found — starting fresh.")

    # Collect already-done results in order
    results: list[dict] = [done[i] for i in sorted(done.keys())]

    # Determine which samples still need evaluation
    remaining = [(idx, s) for idx, s in enumerate(samples) if idx not in done]
    if not remaining:
        print("All samples already evaluated. Skipping model load.")
        metrics = compute_and_print_metrics(results, len(samples))
        _save_and_log(args, results, metrics, ckpt_path)
        return

    print(f"{len(remaining)} samples remaining to evaluate.")

    # ------------------------------------------------------------------
    # 3. Load model + tokenizer
    # ------------------------------------------------------------------
    import unsloth  # must be first
    from unsloth import FastLanguageModel
    import torch

    adapter_path = Path(args.adapter_path)
    if not adapter_path.exists():
        sys.exit(f"[error] adapter not found: {adapter_path}")

    print(f"\nLoading LoRA adapter: {adapter_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_path),
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=args.load_in_4bit,
    )
    FastLanguageModel.for_inference(model)
    # Truncate from the LEFT so WLS tool results + generation prompt are always kept
    tokenizer.truncation_side = "left"
    print("Model ready.\n")

    # ------------------------------------------------------------------
    # 4. W&B init
    # ------------------------------------------------------------------
    wandb_run = None
    if args.wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project="psse-agent-sft",
                job_type="eval",
                resume="allow",
                config={
                    "adapter_path":   str(adapter_path),
                    "test_file":      str(test_path),
                    "n_samples":      len(samples),
                    "max_new_tokens": args.max_new_tokens,
                },
            )
            print(f"W&B run: {wandb_run.url}")
        except Exception as e:
            print(f"[warn] W&B init failed: {e}")
            wandb_run = None

    # ------------------------------------------------------------------
    # 5. Evaluate remaining samples
    # ------------------------------------------------------------------
    max_input_len = args.max_seq_length - args.max_new_tokens

    for idx, sample in remaining:
        messages = sample["messages"]

        gt = extract_ground_truth(messages)
        if gt is None:
            print(f"[warn] sample {idx}: no ground truth, skipping")
            continue

        prompt_messages = build_prompt_messages(messages)
        clean_msgs = [{k: v for k, v in m.items() if v is not None} for m in prompt_messages]
        prompt_text = tokenizer.apply_chat_template(
            clean_msgs, tokenize=False, add_generation_prompt=True,
        )
        
        prompt_text = prompt_text + FORCED_PREFIX

        inputs = tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_len,
        ).to(model.device)
        input_len = inputs["input_ids"].shape[1]

        torch.cuda.empty_cache()
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=1.0,
                use_cache=True,
            )
        del inputs

        # Prepend the forced prefix so extract_prediction sees the full JSON.
        generated = FORCED_PREFIX + tokenizer.decode(
            output_ids[0][input_len:], skip_special_tokens=True
        ).strip()
        del output_ids
        torch.cuda.empty_cache()

        # Debug: print raw output for first 5 samples
        if idx < 5:
            print(f"  [debug] raw output: {generated[:400]!r}")

        pred_data = extract_prediction(generated)
        if pred_data is not None:
            verdict = pred_data.get("verdict") if isinstance(pred_data.get("verdict"), dict) else pred_data
            pred_family = verdict.get("error_family")
            if isinstance(pred_family, list):
                pred_family = pred_family[0] if pred_family else None
            confidence = verdict.get("confidence")
        else:
            pred_family = None
            confidence  = None

        correct = (pred_family == gt)

        row = {
            "idx":        idx,
            "gt":         gt,
            "pred":       pred_family,
            "correct":    correct,
            "confidence": confidence,
            "raw_output": generated,
        }
        results.append(row)

        # Save to checkpoint immediately after each sample
        append_checkpoint(ckpt_path, row)

        # Running accuracy across all evaluated so far (including resumed)
        evaluated = sorted(results, key=lambda r: r["idx"])
        n_correct  = sum(r["correct"] for r in evaluated)
        acc_so_far = n_correct / len(evaluated) * 100
        status     = "✓" if correct else "✗"
        print(
            f"[{len(evaluated):>3}/{len(samples)}] {status}"
            f"  gt={gt:<22}  pred={str(pred_family):<22}"
            f"  conf={confidence}  acc={acc_so_far:.1f}%"
        )

        # Log running accuracy to W&B
        if wandb_run is not None:
            try:
                wandb_run.log({
                    "eval/running_accuracy": acc_so_far / 100,
                    "eval/samples_done":     len(evaluated),
                })
            except Exception:
                pass

    # ------------------------------------------------------------------
    # 6. Final metrics + save
    # ------------------------------------------------------------------
    results_sorted = sorted(results, key=lambda r: r["idx"])
    metrics = compute_and_print_metrics(results_sorted, len(samples))
    _save_and_log(args, results_sorted, metrics, ckpt_path, wandb_run)


def _save_and_log(args, results, metrics, ckpt_path, wandb_run=None):
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        **metrics,
        "adapter_path": args.adapter_path,
        "test_file":    args.test_file,
        "samples":      results,
    }
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nFull results saved → {out_path}")
    print(f"Checkpoint file    → {ckpt_path}")

    if wandb_run is not None:
        try:
            import wandb
            log_dict = {
                "eval/overall_accuracy": metrics["overall_accuracy"],
                "eval/macro_precision":  metrics["macro_precision"],
                "eval/macro_recall":     metrics["macro_recall"],
                "eval/macro_f1":         metrics["macro_f1"],
                "eval/null_rate":        metrics["null_predictions"] / metrics["total"],
                "eval/correct":          metrics["correct"],
                "eval/total":            metrics["total"],
            }
            if metrics["mean_confidence"] is not None:
                log_dict["eval/mean_confidence"] = metrics["mean_confidence"]
                log_dict["eval/min_confidence"]  = metrics["min_confidence"]
                log_dict["eval/max_confidence"]  = metrics["max_confidence"]
            for fam, acc in metrics["per_class_accuracy"].items():
                log_dict[f"eval/acc_{fam}"]   = acc
                log_dict[f"eval/prec_{fam}"]  = metrics["per_class_precision"].get(fam, 0)
                log_dict[f"eval/f1_{fam}"]    = metrics["per_class_f1"].get(fam, 0)
            wandb_run.log(log_dict)

            table = wandb.Table(columns=["gt", "pred", "correct", "confidence"])
            for r in results:
                table.add_data(r["gt"], r["pred"], r["correct"], r["confidence"])
            wandb_run.log({"eval/predictions": table})
            wandb_run.finish()
            print("W&B results logged.")
        except Exception as e:
            print(f"[warn] W&B logging failed: {e}")


if __name__ == "__main__":
    main()
