import difflib

files_to_compare = [
    (r"d:\ps_llm_agent\Transmission\generate_measurements.py", r"d:\ps_llm_agent\revision\generate_measurements_revised.py"),
    (r"d:\ps_llm_agent\Transmission\build_sft_traces.py", r"d:\ps_llm_agent\revision\build_sft_traces_revised_v2.py"),
    (r"d:\ps_llm_agent\mcp_server\matpower_server.py", r"d:\ps_llm_agent\revision\matpower_server_revised.py"),
]

with open(r"d:\ps_llm_agent\tmp_diff_output.txt", "w", encoding="utf-8") as out:
    for orig, rev in files_to_compare:
        out.write(f"--- Diff for {orig} vs {rev} ---\n")
        try:
            with open(orig, 'r', encoding='utf-8') as f1, open(rev, 'r', encoding='utf-8') as f2:
                orig_lines = f1.readlines()
                rev_lines = f2.readlines()
                
                diff = difflib.unified_diff(orig_lines, rev_lines, fromfile=orig, tofile=rev)
                out.write("".join(diff))
        except Exception as e:
            out.write(f"Error comparing {orig} and {rev}: {e}\n")
        out.write("\n" + "="*80 + "\n\n")
