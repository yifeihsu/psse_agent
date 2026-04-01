import json
import random
from collections import defaultdict

def evaluate_traces(filepath, samples_per_class=20):
    traces_by_class = defaultdict(list)
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            trace = json.loads(line)
            messages = trace.get("messages", [])
            
            final_content = ""
            for m in reversed(messages):
                if m["role"] == "assistant" and not m.get("tool_calls"):
                    final_content = m.get("content", "")
                    break
                    
            cls = "unknown"
            if final_content:
                try:
                    payload = json.loads(final_content)
                    if "verdict" in payload:
                        cls = payload["verdict"].get("error_family", "unknown")
                    else:
                        cls = payload.get("error_type", "unknown")
                except Exception:
                    pass
            
            trace["_line_idx"] = idx
            traces_by_class[cls].append(trace)
            
    # Now evaluate samples
    for cls, traces in traces_by_class.items():
        print(f"\n{'='*60}")
        print(f"CLASS: {cls.upper()} (Total: {len(traces)})")
        print(f"{'='*60}")
        
        sample_size = min(samples_per_class, len(traces))
        samples = random.sample(traces, sample_size)
        
        valid_count = 0
        issues = []
        
        for trace in samples:
            trace_id = f"Line {trace['_line_idx']}"
            messages = trace.get("messages", [])
            
            tool_calls = set()
            tool_results = []
            final_content = None
            
            for m in messages:
                if m["role"] == "assistant" and "tool_calls" in m:
                    for tc in m["tool_calls"]:
                        tool_calls.add(tc["function"]["name"])
                elif m["role"] == "tool":
                    tool_results.append(m)
                elif m["role"] == "assistant" and not m.get("tool_calls"):
                    final_content = m.get("content", "")
                    
            if not final_content:
                issues.append(f"{trace_id}: Missing final assistant answer")
                continue
                
            is_valid = True
            
            has_wls = any("wls" in tc for tc in tool_calls)
            
            if cls == "no_error":
                if not has_wls:
                    issues.append(f"{trace_id}: Did not call WLS tool")
                    is_valid = False
            
            elif cls == "measurement_error":
                if not has_wls:
                    issues.append(f"{trace_id}: Did not call WLS tool")
                    is_valid = False
                    
            elif cls == "parameter_error":
                if not has_wls:
                    issues.append(f"{trace_id}: Did not call WLS tool")
                    is_valid = False
                if not any("correct_parameters" in tc for tc in tool_calls):
                    issues.append(f"{trace_id}: Did not call correct_parameters tool")
                    is_valid = False
                    
            elif cls == "topology_error":
                if not has_wls:
                    issues.append(f"{trace_id}: Did not call WLS tool")
                    is_valid = False
                    
            elif cls == "harmonic_anomaly":
                pass
                
            if is_valid:
                valid_count += 1
                
        print(f"Logical Flow Valid: {valid_count} / {sample_size}")
        if issues:
            print(f"Issues found ({len(issues)} total):")
            for issue in issues[:5]: 
                print(f"  - {issue}")
            
        if traces:
            print(f"\n[Example Required Tools Called For Output]")
            tc_list = []
            for m in samples[0]["messages"]:
                if m["role"] == "assistant" and m.get("tool_calls"):
                    for t in m["tool_calls"]:
                        tc_list.append(t["function"]["name"])
            print(f"Tools Used -> {tc_list}")

if __name__ == "__main__":
    evaluate_traces("d:/ps_llm_agent/out_traces_balanced/sft_traces.jsonl", samples_per_class=30)
