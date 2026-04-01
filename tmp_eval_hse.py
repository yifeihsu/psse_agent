import json

ground_truths = {}
with open(r'd:\ps_llm_agent\out_measurements_balanced\samples.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        sample = json.loads(line)
        if sample['label'].get('error_type') == 'harmonic_anomaly':
            z_obs_tuple = tuple(sample['z_obs'])
            ground_truths[z_obs_tuple] = sample['label']['source_bus']

matches = 0
total_hse = 0
results = []

with open(r'd:\ps_llm_agent\out_traces_balanced\sft_traces.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        trace = json.loads(line)
        user_msg = json.loads(trace['messages'][1]['content'])
        
        hse_tool_output = None
        for msg in trace['messages']:
            if msg['role'] == 'tool' and msg.get('name') == 'run_hse_from_path':
                hse_tool_output = json.loads(msg['content'])
                break
                
        if hse_tool_output:
            z_obs_tuple = tuple(user_msg['z_obs'])
            true_src = ground_truths.get(z_obs_tuple)
            predicted_src = hse_tool_output.get('best_candidate_bus_1based')
            
            top3 = [x['bus_1based'] for x in hse_tool_output.get('ranking_top10', [])[:3]]
            
            is_correct = (true_src == predicted_src)
            if is_correct: matches += 1
            
            if total_hse < 15:
                status_str = "CORRECT" if is_correct else "INCORRECT"
                results.append(f"Sample {total_hse+1:2d}: True={true_src:2d} | Pred={predicted_src:2d} | Top 3: {top3} -> {status_str}")
            total_hse += 1

print("\n--- HSE Trace Accuracy ---")
print(f"Total HSE Traces Evaluated: {total_hse}")
print(f"Correct Source Identifications: {matches} ({(matches/total_hse)*100:.1f}%)")
print("\n--- First 15 Samples ---")
for res in results:
    print(res)
