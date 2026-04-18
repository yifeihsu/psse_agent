

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_HF_REPO      = "harshith0214/psse-agent-gpt-oss-20b-lora"
DEFAULT_TEST_FILE    = "/home/hk4488/psse_agent/data/split_test.jsonl"
DEFAULT_OUTPUT       = "/scratch/hk4488/psse_agent/outputs/eval_hf_results.json"
DEFAULT_CKPT_FILE    = "/scratch/hk4488/psse_agent/outputs/eval_hf_checkpoint.jsonl"
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_MAX_SEQ_LEN    = 16384

# GPT-OSS forced-prefix: completes the assistant turn header and jumps
# directly to the flat JSON verdict, bypassing the commentary channel.
FORCED_PREFIX = '<|message|>final{"has_error":'

# Valid error families
VALID_FAMILIES = {"measurement_error", "parameter_error", "topology_error", "no_error"}

# Valid tool names
VALID_TOOLS = {
    "wls_from_path",
    "correct_measurements_from_path",
    "correct_parameters_from_path",
    "correct_topology_from_path",
    "run_hse_from_path",
}

# Expected correction tool for each error family (if any correction is needed)
EXPECTED_CORRECTION_TOOL = {
    "measurement_error": "correct_measurements_from_path",
    "parameter_error":   "correct_parameters_from_path",
    "topology_error":    "correct_topology_from_path",
    "no_error":          None,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate PSSE agent loaded from HuggingFace")
    p.add_argument("--hf-repo",        default=DEFAULT_HF_REPO,
                   help="HuggingFace repo ID for the LoRA adapter")
    p.add_argument("--test-file",      default=DEFAULT_TEST_FILE)
    p.add_argument("--output",         default=DEFAULT_OUTPUT)
    p.add_argument("--checkpoint-file", default=DEFAULT_CKPT_FILE)
    p.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    p.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LEN)
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
                continue
            done[row["idx"]] = row
        except (json.JSONDecodeError, KeyError):
            pass
    return done


def append_checkpoint(ckpt_path: Path, row: dict) -> None:
    with ckpt_path.open("a") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()


# ---------------------------------------------------------------------------
# Tool-call trace verification
# ---------------------------------------------------------------------------

def verify_tool_calls(messages: list[dict], gt_family: str) -> dict:
    """
    Inspect the ground-truth tool-call trace and return a structured report.

    Checks:
      - Every tool_call has a valid name, non-empty id, and parseable JSON arguments
      - Every tool response has a matching tool_call_id and parseable JSON content
      - All tool IDs are paired (call ↔ response with same id)
      - wls_from_path is the first call
      - The appropriate correction tool is present for error-class samples
      - Tool responses report success (or expected failure) correctly
    """
    issues: list[str] = []

    # Collect all tool calls and responses
    tool_calls: list[dict] = []   # {id, name, arguments}
    tool_resps: dict[str, dict] = {}  # tool_call_id → {name, content}

    for i, m in enumerate(messages):
        if m.get("tool_calls"):
            for tc in m["tool_calls"]:
                fn   = tc.get("function", {})
                name = fn.get("name", "")
                tid  = tc.get("id", "")
                args = fn.get("arguments", "")

                if not tid:
                    issues.append(f"msg[{i}]: tool_call missing id")
                if name not in VALID_TOOLS:
                    issues.append(f"msg[{i}]: unknown tool '{name}'")
                try:
                    json.loads(args)
                except (json.JSONDecodeError, TypeError):
                    issues.append(f"msg[{i}]: tool_call '{name}' has invalid JSON arguments")

                tool_calls.append({"id": tid, "name": name, "arguments": args})

        if m.get("role") == "tool":
            tid     = m.get("tool_call_id", "")
            name    = m.get("name", "")
            content = m.get("content", "")
            if not tid:
                issues.append(f"msg[{i}]: tool response missing tool_call_id")
            try:
                json.loads(content)
            except (json.JSONDecodeError, TypeError):
                issues.append(f"msg[{i}]: tool response for '{name}' has invalid JSON content")
            tool_resps[tid] = {"name": name, "content": content}

    # Verify every call has a matching response
    call_ids = {tc["id"] for tc in tool_calls if tc["id"]}
    resp_ids = set(tool_resps.keys())
    unmatched_calls = call_ids - resp_ids
    unmatched_resps = resp_ids - call_ids
    if unmatched_calls:
        issues.append(f"tool calls with no response: {unmatched_calls}")
    if unmatched_resps:
        issues.append(f"tool responses with no matching call: {unmatched_resps}")

    tool_names_seq = [tc["name"] for tc in tool_calls]

    # wls_from_path must be first
    if tool_names_seq and tool_names_seq[0] != "wls_from_path":
        issues.append(f"first tool is '{tool_names_seq[0]}', expected 'wls_from_path'")

    # wls_from_path must be present in every trace
    if "wls_from_path" not in tool_names_seq:
        issues.append("wls_from_path never called")

    # Check correction tool presence for error families
    expected_corr = EXPECTED_CORRECTION_TOOL.get(gt_family)
    if expected_corr is not None:
        if expected_corr not in tool_names_seq:
            issues.append(
                f"expected correction tool '{expected_corr}' for '{gt_family}' not called"
            )
    else:
        # no_error — correction tools should not appear
        correction_tools_called = [
            n for n in tool_names_seq
            if n in {"correct_measurements_from_path",
                     "correct_parameters_from_path",
                     "correct_topology_from_path"}
        ]
        if correction_tools_called:
            issues.append(
                f"no_error sample but correction tool(s) called: {correction_tools_called}"
            )

    return {
        "ok":              len(issues) == 0,
        "issues":          issues,
        "tool_sequence":   tool_names_seq,
        "n_tool_calls":    len(tool_calls),
        "n_tool_resps":    len(tool_resps),
    }


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
# Build inference prompt (strips final assistant turn)
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
# Parse model output → error_family + confidence
# ---------------------------------------------------------------------------

def extract_prediction(generated_text: str) -> dict | None:
    text = re.sub(r"^```[a-z]*\s*", "", generated_text.strip())
    text = re.sub(r"\s*```$", "", text)

    final_match = re.search(r"(?:^|\b)final(\{)", text)
    if final_match:
        text = text[final_match.start(1):]

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
                candidate = text[start: i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    pass

    # Regex fallback for error_family
    fam_match = re.search(
        r'"error_family"\s*:\s*(?:"([^"]+)"|\[\s*"([^"]+)")',
        text,
    )
    if fam_match:
        return {"error_family": fam_match.group(1) or fam_match.group(2)}

    return None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(results: list[dict], total_samples: int) -> dict:
    per_class: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        per_class[r["gt"]]["total"] += 1
        if r["correct"]:
            per_class[r["gt"]]["correct"] += 1

    total   = len(results)
    correct = sum(r["correct"] for r in results)
    overall_acc = correct / total if total > 0 else 0.0

    confs     = [r["confidence"] for r in results if r["confidence"] is not None]
    mean_conf = sum(confs) / len(confs) if confs else None

    families = sorted(per_class.keys())
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in results:
        if r["gt"] and r["pred"]:
            confusion[r["gt"]][r["pred"]] += 1

    pred_counts: dict[str, int] = defaultdict(int)
    for r in results:
        if r["pred"]:
            pred_counts[r["pred"]] += 1

    print("\n" + "=" * 65)
    print(f"OVERALL ACCURACY : {correct}/{total} (of {total_samples} total) = {overall_acc*100:.2f}%")
    print("=" * 65)

    print(f"\n{'Class':<26} {'Prec':>7} {'Recall':>7} {'F1':>7} {'Support':>9}")
    print("-" * 65)
    class_prec, class_rec, class_f1 = {}, {}, {}
    for fam in families:
        tp      = per_class[fam]["correct"]
        support = per_class[fam]["total"]
        rec  = tp / support           if support > 0          else 0.0
        prec = tp / pred_counts[fam]  if pred_counts[fam] > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        class_prec[fam] = prec
        class_rec[fam]  = rec
        class_f1[fam]   = f1
        print(f"  {fam:<26} {prec*100:>6.1f}% {rec*100:>6.1f}% {f1*100:>6.1f}% {support:>9}")

    macro_prec = sum(class_prec.values()) / len(class_prec) if class_prec else 0.0
    macro_rec  = sum(class_rec.values())  / len(class_rec)  if class_rec  else 0.0
    macro_f1   = sum(class_f1.values())   / len(class_f1)   if class_f1   else 0.0
    print("-" * 65)
    print(f"  {'macro avg':<26} {macro_prec*100:>6.1f}% {macro_rec*100:>6.1f}% {macro_f1*100:>6.1f}% {total:>9}")

    if mean_conf is not None:
        print(f"\nMean confidence  : {mean_conf:.4f}")

    n_null = sum(1 for r in results if r["pred"] is None)
    print(f"Null predictions : {n_null}/{total} ({n_null/total*100:.1f}%)")

    # Confusion matrix
    print("\nConfusion matrix (rows=GT, cols=pred):")
    header = f"{'':>26}" + "".join(f"{f[:12]:>14}" for f in families)
    print(header)
    for gt_fam in families:
        row_str = f"{gt_fam:<26}" + "".join(f"{confusion[gt_fam][p]:>14}" for p in families)
        print(row_str)

    return {
        "overall_accuracy":    overall_acc,
        "correct":             correct,
        "total":               total,
        "macro_precision":     macro_prec,
        "macro_recall":        macro_rec,
        "macro_f1":            macro_f1,
        "null_predictions":    n_null,
        "mean_confidence":     mean_conf,
        "per_class_precision": class_prec,
        "per_class_recall":    class_rec,
        "per_class_f1":        class_f1,
        "per_class_counts":    {k: dict(v) for k, v in per_class.items()},
    }


def print_tool_call_report(results: list[dict]) -> dict:
    """Aggregate and print tool-call verification results."""
    total         = len(results)
    traces_ok     = sum(1 for r in results if r["tool_call_verification"]["ok"])
    traces_issues = [r for r in results if not r["tool_call_verification"]["ok"]]

    # Sequence distribution
    seq_dist: dict[str, int] = defaultdict(int)
    for r in results:
        seq = " → ".join(r["tool_call_verification"]["tool_sequence"])
        seq_dist[seq] += 1

    # Issue distribution
    all_issues: list[str] = []
    for r in traces_issues:
        all_issues.extend(r["tool_call_verification"]["issues"])
    issue_dist: dict[str, int] = defaultdict(int)
    for iss in all_issues:
        issue_dist[iss] += 1

    print("\n" + "=" * 65)
    print("TOOL-CALL TRACE VERIFICATION")
    print("=" * 65)
    print(f"Traces passing all checks : {traces_ok}/{total} ({traces_ok/total*100:.1f}%)")

    print("\nTool sequence distribution:")
    for seq, cnt in sorted(seq_dist.items(), key=lambda x: -x[1]):
        print(f"  {cnt:>4}x  {seq}")

    if issue_dist:
        print("\nIssues found:")
        for iss, cnt in sorted(issue_dist.items(), key=lambda x: -x[1]):
            print(f"  {cnt:>4}x  {iss}")
    else:
        print("\nNo issues found in any trace.")

    return {
        "traces_ok":          traces_ok,
        "traces_total":       total,
        "traces_pass_rate":   traces_ok / total if total > 0 else 0.0,
        "sequence_distribution": dict(seq_dist),
        "issue_distribution": dict(issue_dist),
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
        sys.exit(f"[error] test file not found: {args.test_file}")

    samples = [json.loads(l) for l in test_path.read_text().splitlines() if l.strip()]
    if args.limit > 0:
        samples = samples[: args.limit]
    print(f"Loaded {len(samples)} test samples from {test_path}")

    # ------------------------------------------------------------------
    # 2. Resume from checkpoint
    # ------------------------------------------------------------------
    ckpt_path = Path(args.checkpoint_file)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    done = load_checkpoint(ckpt_path)
    if done:
        print(f"Resuming from checkpoint: {len(done)}/{len(samples)} done.")
    else:
        print("No checkpoint — starting fresh.")

    results: list[dict] = [done[i] for i in sorted(done.keys())]
    remaining = [(idx, s) for idx, s in enumerate(samples) if idx not in done]

    if not remaining:
        print("All samples already evaluated.")
        tc_report = print_tool_call_report(results)
        metrics   = compute_metrics(results, len(samples))
        _save(args, results, metrics, tc_report, ckpt_path)
        return

    print(f"{len(remaining)} samples to evaluate.")

    # ------------------------------------------------------------------
    # 3. Load model from HuggingFace
    # ------------------------------------------------------------------
    import unsloth  # must be imported first
    from unsloth import FastLanguageModel
    import torch

    print(f"\nLoading adapter from HuggingFace: {args.hf_repo}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = args.hf_repo,
        max_seq_length = args.max_seq_length,
        dtype          = None,
        load_in_4bit   = args.load_in_4bit,
    )
    FastLanguageModel.for_inference(model)
    tokenizer.truncation_side = "left"
    print("Model ready.\n")

    # ------------------------------------------------------------------
    # 4. W&B
    # ------------------------------------------------------------------
    wandb_run = None
    if args.wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project  = "psse-agent-sft",
                job_type = "eval_hf",
                resume   = "allow",
                config   = {
                    "hf_repo":        args.hf_repo,
                    "test_file":      str(test_path),
                    "n_samples":      len(samples),
                    "max_new_tokens": args.max_new_tokens,
                    "max_seq_length": args.max_seq_length,
                },
            )
            print(f"W&B: {wandb_run.url}")
        except Exception as e:
            print(f"[warn] W&B init failed: {e}")

    # ------------------------------------------------------------------
    # 5. Evaluate
    # ------------------------------------------------------------------
    max_input_len = args.max_seq_length - args.max_new_tokens

    for idx, sample in remaining:
        messages = sample["messages"]

        gt = extract_ground_truth(messages)
        if gt is None:
            print(f"[warn] sample {idx}: no ground truth, skipping")
            continue

        # ── Tool-call trace verification (uses ground-truth trace) ──
        tc_verify = verify_tool_calls(messages, gt)

        # ── Build prompt (drop final assistant turn) ──
        prompt_msgs = build_prompt_messages(messages)
        clean_msgs  = [{k: v for k, v in m.items() if v is not None} for m in prompt_msgs]
        prompt_text = tokenizer.apply_chat_template(
            clean_msgs, tokenize=False, add_generation_prompt=True,
        )
        # Forced prefix: `<|start|>assistant<|message|>final{"has_error":`
        prompt_text = prompt_text + FORCED_PREFIX

        inputs    = tokenizer(
            prompt_text, return_tensors="pt",
            truncation=True, max_length=max_input_len,
        ).to(model.device)
        input_len = inputs["input_ids"].shape[1]

        torch.cuda.empty_cache()
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens = args.max_new_tokens,
                do_sample      = False,
                temperature    = 1.0,
                use_cache      = True,
            )
        del inputs

        generated = FORCED_PREFIX + tokenizer.decode(
            output_ids[0][input_len:], skip_special_tokens=True
        ).strip()
        del output_ids
        torch.cuda.empty_cache()

        if idx < 5:
            print(f"  [debug idx={idx}] {generated[:300]!r}")

        # ── Parse verdict ──
        pred_data = extract_prediction(generated)
        if pred_data is not None:
            verdict     = pred_data.get("verdict") if isinstance(pred_data.get("verdict"), dict) else pred_data
            pred_family = verdict.get("error_family")
            if isinstance(pred_family, list):
                pred_family = pred_family[0] if pred_family else None
            confidence  = verdict.get("confidence")
        else:
            pred_family = None
            confidence  = None

        correct = (pred_family == gt)

        row = {
            "idx":                   idx,
            "gt":                    gt,
            "pred":                  pred_family,
            "correct":               correct,
            "confidence":            confidence,
            "raw_output":            generated,
            "tool_call_verification": tc_verify,
        }
        results.append(row)
        append_checkpoint(ckpt_path, row)

        evaluated  = sorted(results, key=lambda r: r["idx"])
        n_correct  = sum(r["correct"] for r in evaluated)
        acc_so_far = n_correct / len(evaluated) * 100
        tc_flag    = "" if tc_verify["ok"] else "  [TC-ISSUE]"
        print(
            f"[{len(evaluated):>3}/{len(samples)}] {'✓' if correct else '✗'}"
            f"  gt={gt:<22} pred={str(pred_family):<22}"
            f" conf={confidence}  acc={acc_so_far:.1f}%{tc_flag}"
        )

        if wandb_run is not None:
            try:
                wandb_run.log({
                    "eval/running_accuracy": acc_so_far / 100,
                    "eval/samples_done":     len(evaluated),
                })
            except Exception:
                pass

    # ------------------------------------------------------------------
    # 6. Final report
    # ------------------------------------------------------------------
    results_sorted = sorted(results, key=lambda r: r["idx"])
    tc_report      = print_tool_call_report(results_sorted)
    metrics        = compute_metrics(results_sorted, len(samples))
    _save(args, results_sorted, metrics, tc_report, ckpt_path, wandb_run)


def _save(args, results, metrics, tc_report, ckpt_path, wandb_run=None):
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "hf_repo":              args.hf_repo,
        "test_file":            args.test_file,
        "verdict_metrics":      metrics,
        "tool_call_report":     tc_report,
        "samples":              results,
    }
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nResults saved → {out_path}")
    print(f"Checkpoint    → {ckpt_path}")

    if wandb_run is not None:
        try:
            import wandb
            log_dict = {
                "eval/overall_accuracy":  metrics["overall_accuracy"],
                "eval/macro_precision":   metrics["macro_precision"],
                "eval/macro_recall":      metrics["macro_recall"],
                "eval/macro_f1":          metrics["macro_f1"],
                "eval/null_rate":         metrics["null_predictions"] / metrics["total"],
                "eval/tc_pass_rate":      tc_report["traces_pass_rate"],
            }
            for fam in metrics["per_class_precision"]:
                log_dict[f"eval/prec_{fam}"]   = metrics["per_class_precision"][fam]
                log_dict[f"eval/recall_{fam}"]  = metrics["per_class_recall"][fam]
                log_dict[f"eval/f1_{fam}"]      = metrics["per_class_f1"][fam]
            wandb_run.log(log_dict)

            table = wandb.Table(
                columns=["gt", "pred", "correct", "confidence", "tc_ok", "tc_issues"]
            )
            for r in results:
                tc = r["tool_call_verification"]
                table.add_data(
                    r["gt"], r["pred"], r["correct"], r["confidence"],
                    tc["ok"], "; ".join(tc["issues"]),
                )
            wandb_run.log({"eval/predictions": table})
            wandb_run.finish()
            print("W&B logged.")
        except Exception as e:
            print(f"[warn] W&B logging failed: {e}")


if __name__ == "__main__":
    main()
