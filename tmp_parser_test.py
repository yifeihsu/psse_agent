import re
import numpy as np
import ast

def _parse_matpower_case(case_text: str) -> dict:
    # 1. Extract baseMVA
    base_mva_match = re.search(r'mpc\.baseMVA\s*=\s*([\d\.]+);', case_text)
    baseMVA = float(base_mva_match.group(1)) if base_mva_match else 100.0
    
    # 2. Extract matrices (bus, gen, branch)
    def extract_matrix(name: str) -> np.ndarray:
        # Match mpc.bus = [ ... ];
        # Allow optional spaces and newlines
        pattern = rf'mpc\.{name}\s*=\s*\[(.*?)\];'
        match = re.search(pattern, case_text, re.DOTALL)
        if not match:
            return np.array([])
            
        matrix_str = match.group(1)
        # Parse into a list of lists of floats
        rows = []
        for line in matrix_str.split('\n'):
            # Strip comments
            line = line.split('%')[0].strip()
            if not line:
                continue
            # Split by whitespace or commas, parse as floats
            row_vals = [float(val) for val in re.split(r'[\s,;]+', line) if val]
            if row_vals:
                rows.append(row_vals)
        return np.array(rows)

    bus = extract_matrix('bus')
    gen = extract_matrix('gen')
    branch = extract_matrix('branch')

    return {
        "baseMVA": baseMVA,
        "bus": bus,
        "gen": gen,
        "branch": branch
    }

if __name__ == "__main__":
    with open("d:/ps_llm_agent/mcp_server/case14.m", "r") as f:
        case_text = f.read()
    
    ppc = _parse_matpower_case(case_text)
    print(f"baseMVA: {ppc['baseMVA']}")
    print(f"bus shape: {ppc['bus'].shape}")
    print(f"gen shape: {ppc['gen'].shape}")
    print(f"branch shape: {ppc['branch'].shape}")
    print(ppc['bus'][0, :5])
