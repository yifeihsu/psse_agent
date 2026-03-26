"""
eval_sft_agent.py  –  Evaluate a fine-tuned LoRA adapter on the held-out test split.

For each test conversation the script:
  1. Feeds the system prompt + user observation to the model.
  2. Lets the model generate a response (either a tool call or final verdict).
  3. If the model emits a tool call, executes it against the real Python tools
     and appends the tool response, then loops back to step 2.
  4. Once the model emits a final JSON verdict, compares it against the ground-
     truth verdict from the test trace.
  5. Prints per-sample results and aggregate accuracy.

Usage:
    python eval_sft_agent.py                          # defaults
    python eval_sft_agent.py --adapter outputs/gpt_oss_sft_power_agent_4k/lora
    python eval_sft_agent.py --max-samples 10         # quick smoke test
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from collections import Counter
from typing import Any

# ---------------------------------------------------------------------------
# 0.  Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Evaluate fine-tuned PSSE agent")
    p.add_argument("--adapter",    default="outputs/gpt_oss_sft_power_agent_4k/lora",
                   help="Path to the LoRA adapter directory")
    p.add_argument("--test-file",  default="out_traces_balanced/sft_traces.test.jsonl",
                   help="JSONL file with test conversations")
    p.add_argument("--max-samples", type=int, default=None,
                   help="Cap on the number of samples to evaluate")
    p.add_argument("--max-turns",  type=int, default=8,
                   help="Max tool-call turns before forcing stop (safety)")
    p.add_argument("--max-new-tokens", type=int, default=2048,
                   help="Max tokens per generation step")
    p.add_argument("--max-seq-length", type=int, default=8192,
                   help="Max context length for the model")
    p.add_argument("--output",     default="eval_results.jsonl",
                   help="Where to write per-sample results")
    p.add_argument("--verbose",    action="store_true",
                   help="Print every generation step")
    return p.parse_args()

# ---------------------------------------------------------------------------
# 1.  Import tools from the MCP server module (direct Python calls)
# ---------------------------------------------------------------------------
# We import the raw Python functions, stripping the FastMCP decorator overhead.
# Each function uses keyword-only arguments and returns a dict.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mcp_server.matpower_server import (          # noqa: E402
    wls_from_path,
    correct_measurements_from_path,
    correct_parameters_from_path,
    correct_topology_from_path,
    run_hse_from_path,
)

TOOL_MAP = {
    "wls_from_path":                    wls_from_path,
    "correct_measurements_from_path":   correct_measurements_from_path,
    "correct_parameters_from_path":     correct_parameters_from_path,
    "correct_topology_from_path":       correct_topology_from_path,
    "run_hse_from_path":                run_hse_from_path,
}


def execute_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Call one of the matpower tools and return its JSON-serialisable result."""
    tool_obj = TOOL_MAP.get(name)
    if tool_obj is None:
        return {"success": False, "error": f"Unknown tool: {name}"}
    try:
        # tool_obj is a FastMCP FunctionTool, we need to extract the raw python function
        fn = getattr(tool_obj, "fn", tool_obj)
        return fn(**arguments)
    except Exception as exc:
        return {"success": False, "error": f"{type(exc).__name__}: {exc}"}


# ---------------------------------------------------------------------------
# 2.  Extract the ground-truth verdict from a test conversation
# ---------------------------------------------------------------------------
def extract_ground_truth(messages: list[dict]) -> dict | None:
    """Return the parsed JSON verdict from the last assistant message."""
    for msg in reversed(messages):
        if msg.get("role") == "assistant" and "content" in msg and msg["content"]:
            try:
                obj = json.loads(msg["content"])
                if "verdict" in obj:
                    return obj
            except (json.JSONDecodeError, TypeError):
                pass
    return None


# ---------------------------------------------------------------------------
# 3.  Parse a model generation into either a tool-call or a final answer
# ---------------------------------------------------------------------------
def parse_generation(text: str, tokenizer) -> dict:
    """
    Parse the raw generated text from a gpt-oss model.

    The gpt-oss chat template uses channel routing tokens:
      Tool call:  to=functions.TOOL_NAME<|channel|>commentary json<|message|>{ARGS}<|call|>
      Verdict:    <|channel|>final<|message|>{VERDICT_JSON}<|return|>

    After skip_special_tokens=True decode, these become:
      Tool call:  "to=functions.TOOL_NAMEcommentary json{ARGS}"
      Verdict:    "final{VERDICT_JSON}"  or just "{VERDICT_JSON}"

    Returns a dict with key 'type':
      - {'type': 'tool_call', 'name': ..., 'arguments': ..., 'id': ...}
      - {'type': 'verdict', 'content': ...}
      - {'type': 'unparseable', 'raw': ...}
    """
    import re
    text = text.strip()

    # ---------------------------------------------------------------
    # Strategy 1: Strict Qwen2.5-Coder FastMCP Tool Call
    # Target format: " to=functions.TOOL_NAME<|channel|>commentary json<|message|>"{\"case_path..."
    # ---------------------------------------------------------------
    if "to=functions." in text and "<|message|>" in text:
        tool_start = text.find("to=functions.") + len("to=functions.")
        
        # Sometimes <|channel|> might be skipped if hallucinated, but <|message|> should be there
        if "<|channel|>" in text:
            tool_end = text.find("<|channel|>", tool_start)
        else:
            tool_end = text.find("commentary", tool_start)
            if tool_end == -1:
                tool_end = text.find("<|message|>", tool_start)
                
        tool_name = text[tool_start:tool_end].strip()
        
        msg_start = text.find("<|message|>") + len("<|message|>")
        payload = text[msg_start:].strip()
        
        # Clean trailing tokens
        for token in ["<|call|>", "<|end|>", "<|return|>"]:
            if payload.endswith(token):
                payload = payload[:-len(token)].strip()
                
        # Handle stringified JSON wrappers
        if payload.startswith('"') and payload.endswith('"'):
            payload = payload[1:-1].replace(r'\"', '"').replace(r'\\', '\\')
            
        try:
            args = json.loads(payload)
            if isinstance(args, str):
                args = json.loads(args)
            return {"type": "tool_call", "name": tool_name, "arguments": args, "id": f"call_{int(time.time())}"}
        except json.JSONDecodeError as e:
            print(f"DEBUG: Strict parser JSON decode failure: {e} on payload: {payload[:50]}...")
            
    # ---------------------------------------------------------------
    # Strategy 2: Strict Verdict
    # Target format: "<|message|>{"verdict": ..."
    # ---------------------------------------------------------------
    if '{"verdict"' in text:
        if "<|message|>" in text:
            payload = text[text.find("<|message|>") + len("<|message|>"):].strip()
        else:
            payload = text[text.find('{"verdict"'):].strip()
            
        for token in ["<|call|>", "<|end|>", "<|return|>"]:
            if payload.endswith(token):
                payload = payload[:-len(token)].strip()
                
        # Un-stringify if double-encoded
        if payload.startswith('"') and payload.endswith('"'):
            payload = payload[1:-1].replace(r'\"', '"').replace(r'\\', '\\')
            
        try:
            args = json.loads(payload)
            if isinstance(args, str):
                args = json.loads(args)
            if "verdict" in args and "action" in args:
                return {"type": "verdict", "content": args}
        except json.JSONDecodeError as e:
            print(f"DEBUG: Strict verdict JSON decode failure: {e}")

    # ---------------------------------------------------------------
    # Strategy 3: Fallback Regex for Hallucinations
    # ---------------------------------------------------------------
    json_blocks = re.findall(r'\{.*\}', text, re.DOTALL)
    if json_blocks:
        for block in reversed(json_blocks):
            try:
                if r'\"' in block:
                    block = block.replace(r'\"', '"').replace(r'\\', '\\')
                obj = json.loads(block)
                if isinstance(obj, str):
                    obj = json.loads(obj)

                if isinstance(obj, dict):
                    if "verdict" in obj and "action" in obj:
                        return {"type": "verdict", "content": obj}
                    if "case_path" in obj and "z" in obj and "suspect_group" not in obj:
                        return {"type": "tool_call", "name": "wls_from_path", "arguments": obj, "id": f"call_wls_{int(time.time())}"}
                    if "suspect_group" in obj and "alpha" in obj:
                        return {"type": "tool_call", "name": "correct_measurements_from_path", "arguments": obj, "id": f"call_corr_{int(time.time())}"}
                    if "harmonic_measurements" in obj:
                        return {"type": "tool_call", "name": "run_hse_from_path", "arguments": obj, "id": f"call_hse_{int(time.time())}"}
                    if "name" in obj and "arguments" in obj and obj["name"] in TOOL_MAP:
                        return {"type": "tool_call", "name": obj["name"], "arguments": obj["arguments"], "id": f"call_oi_{int(time.time())}"}
            except json.JSONDecodeError:
                continue

    # ---------------------------------------------------------------
    # Strategy 3: plain JSON verdict (entire text or embedded)
    # ---------------------------------------------------------------
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "verdict" in obj:
            return {"type": "verdict", "content": obj}
    except json.JSONDecodeError:
        pass

    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        try:
            obj = json.loads(json_match.group(0))
            if isinstance(obj, dict):
                if "verdict" in obj:
                    return {"type": "verdict", "content": obj}
                # Fallback: OpenAI-style {"name": ..., "arguments": ...}
                if "name" in obj and "arguments" in obj and obj["name"] in TOOL_MAP:
                    return {"type": "tool_call", "name": obj["name"],
                            "arguments": obj["arguments"],
                            "id": f"call_{obj['name']}_{int(time.time())}"}
        except json.JSONDecodeError:
            pass

    return {"type": "unparseable", "raw": text[:500]}


# ---------------------------------------------------------------------------
# 4.  Run one sample through the agent loop
# ---------------------------------------------------------------------------
def run_one_sample(
    messages_gt: list[dict],
    model, tokenizer,
    *,
    max_turns: int,
    max_new_tokens: int,
    verbose: bool,
) -> dict:
    """
    Run the model on a single test case and return evaluation metadata.
    """
    import torch

    conversation: list[dict] = []
    # Start with system + user (first 2 messages) from ground truth
    conversation.append(messages_gt[0])  # system
    conversation.append(messages_gt[1])  # user

    tool_calls_made: list[str] = []
    predicted_verdict = None
    error_msg = None

    for turn in range(max_turns):
        # Format prompt
        try:
            prompt = tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
            )
            # CRITICAL FIX: Qwen's template defaults to `<|start|>assistant\n` when forced. 
            # We must strip the trailing newline to match our actual tool-call training format,
            # which expects no newline (`<|start|>assistant to=functions...`).
            if prompt.endswith("<|start|>assistant\n"):
                prompt = prompt[:-1]
        except Exception as exc:
            error_msg = f"Chat template error: {exc}"
            break

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=tokenizer.model_max_length or 8192)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Identify stop tokens for gpt-oss tooling
        stop_strings = ["<|end|>", "<|call|>", "<|return|>"]
        stop_ids = []
        for s in stop_strings:
            tid = tokenizer.convert_tokens_to_ids(s)
            if tid is not None and tid != getattr(tokenizer, "unk_token_id", None):
                stop_ids.append(tid)
        if getattr(tokenizer, "eos_token_id", None) is not None:
            if isinstance(tokenizer.eos_token_id, list):
                stop_ids.extend(tokenizer.eos_token_id)
            elif tokenizer.eos_token_id not in stop_ids:
                stop_ids.append(tokenizer.eos_token_id)
        
        # Deduplicate
        stop_ids = list(set(stop_ids))

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                temperature=0.0,
                do_sample=False,
                eos_token_id=stop_ids,
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
        response_text = tokenizer.decode(new_tokens, skip_special_tokens=False)

        if verbose:
            print(f"\n  ======== [Turn {turn+1}] Generated ({len(new_tokens)} tokens) ========")
            print(f"  {response_text}")
            print("  ======================================================\n")

        parsed = parse_generation(response_text, tokenizer)

        if parsed["type"] == "tool_call":
            tool_name = parsed["name"]
            tool_args = parsed["arguments"]
            tool_calls_made.append(tool_name)

            if verbose:
                print(f"  -> Tool call: {tool_name}")

            # Add assistant message with tool call to conversation
            conversation.append({
                "role": "assistant",
                "tool_calls": [{
                    "type": "function",
                    "id": parsed["id"],
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(tool_args),
                    }
                }]
            })

            # Execute the tool
            tool_result = execute_tool(tool_name, tool_args)

            # Add tool response to conversation
            conversation.append({
                "role": "tool",
                "tool_call_id": parsed["id"],
                "name": tool_name,
                "content": json.dumps(tool_result),
            })

        elif parsed["type"] == "verdict":
            predicted_verdict = parsed["content"]
            # Add it to conversation for completeness
            conversation.append({
                "role": "assistant",
                "content": json.dumps(predicted_verdict),
            })
            break

        else:
            # Unparseable — try to treat as final answer anyway
            error_msg = f"Unparseable output at turn {turn+1}: {parsed.get('raw','')[:200]}"
            break

    # Extract ground truth
    gt_verdict = extract_ground_truth(messages_gt)

    # Compare
    gt_family = gt_verdict.get("verdict", {}).get("error_family") if gt_verdict else None
    pred_family = predicted_verdict.get("verdict", {}).get("error_family") if predicted_verdict else None

    gt_has_error = gt_verdict.get("verdict", {}).get("has_error") if gt_verdict else None
    pred_has_error = predicted_verdict.get("verdict", {}).get("has_error") if predicted_verdict else None

    return {
        "gt_error_family":   gt_family,
        "pred_error_family": pred_family,
        "gt_has_error":      gt_has_error,
        "pred_has_error":    pred_has_error,
        "family_correct":    (gt_family == pred_family),
        "detection_correct": (gt_has_error == pred_has_error),
        "tool_calls":        tool_calls_made,
        "num_turns":         len(tool_calls_made) + (1 if predicted_verdict else 0),
        "predicted_verdict": predicted_verdict,
        "error":             error_msg,
    }


# ---------------------------------------------------------------------------
# 5.  Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    # Validate paths
    if not Path(args.adapter).exists():
        print(f"ERROR: Adapter not found at {args.adapter}")
        sys.exit(1)
    if not Path(args.test_file).exists():
        print(f"ERROR: Test file not found at {args.test_file}")
        sys.exit(1)

    # ---- Load model ----
    print(f"Loading adapter from {args.adapter} ...")
    try:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.adapter,
            max_seq_length=args.max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
        print("Model loaded via Unsloth.\n")
    except ImportError:
        print("Unsloth not available, falling back to transformers + peft ...")
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel

        # Read the base model name from the adapter config
        adapter_cfg_path = Path(args.adapter) / "adapter_config.json"
        with open(adapter_cfg_path, "r") as f:
            adapter_cfg = json.load(f)
        base_model_name = adapter_cfg["base_model_name_or_path"]
        print(f"  Base model: {base_model_name}")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        model = PeftModel.from_pretrained(base_model, args.adapter)
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(args.adapter)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("Model loaded via transformers + peft.\n")

    # ---- Load test data ----
    test_samples = []
    with open(args.test_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                test_samples.append(json.loads(line))
    if args.max_samples:
        test_samples = test_samples[: args.max_samples]
    print(f"Loaded {len(test_samples)} test samples.\n")

    # ---- Run evaluation ----
    results = []
    family_correct = 0
    detection_correct = 0
    errors = 0
    family_counts: Counter = Counter()
    family_correct_counts: Counter = Counter()

    out_file = open(args.output, "w", encoding="utf-8")

    for idx, sample in enumerate(test_samples):
        messages = sample["messages"]
        print(f"[{idx+1}/{len(test_samples)}] ", end="", flush=True)

        try:
            result = run_one_sample(
                messages, model, tokenizer,
                max_turns=args.max_turns,
                max_new_tokens=args.max_new_tokens,
                verbose=args.verbose,
            )
        except Exception as exc:
            traceback.print_exc()
            result = {
                "gt_error_family": None, "pred_error_family": None,
                "family_correct": False, "detection_correct": False,
                "tool_calls": [], "num_turns": 0, "predicted_verdict": None,
                "error": str(exc),
            }

        results.append(result)
        out_file.write(json.dumps(result, default=str) + "\n")
        out_file.flush()

        gt_fam  = result["gt_error_family"] or "unknown"
        family_counts[gt_fam] += 1

        if result["family_correct"]:
            family_correct += 1
            family_correct_counts[gt_fam] += 1
        if result["detection_correct"]:
            detection_correct += 1
        if result["error"]:
            errors += 1

        status = "✓" if result["family_correct"] else "✗"
        # Stringify lists safely for formatting
        pred_fam_str = str(result["pred_error_family"]) if result["pred_error_family"] else "NONE"
        gt_fam_str = str(gt_fam)

        print(f"{status}  gt={gt_fam_str:<20s}  pred={pred_fam_str:<20s}  "
              f"tools={result['tool_calls']}")

    out_file.close()

    # ---- Print summary ----
    n = len(results)
    print("\n" + "=" * 60)
    print(f"{'EVALUATION SUMMARY':^60}")
    print("=" * 60)
    print(f"  Total samples:          {n}")
    print(f"  Error detection acc:    {detection_correct}/{n}  ({100*detection_correct/n:.1f}%)")
    print(f"  Error family acc:       {family_correct}/{n}  ({100*family_correct/n:.1f}%)")
    print(f"  Parse/runtime errors:   {errors}")
    print()
    print("  Per-family breakdown:")
    for fam in sorted(family_counts.keys()):
        total = family_counts[fam]
        correct = family_correct_counts[fam]
        print(f"    {fam:<25s}  {correct}/{total}  ({100*correct/total:.1f}%)")
    print("=" * 60)
    print(f"\nDetailed results saved to: {args.output}")


if __name__ == "__main__":
    main()
