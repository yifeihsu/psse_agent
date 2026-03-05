import csv
import os
import shutil
from typing import Dict, Tuple

import report_sequences as rs


def read_bus_voltages(path: str) -> Dict[str, Tuple[float, float, float]]:
    out = {}
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            out[row["bus"].lower()] = (
                float(row["V1_pu"]),
                float(row["V1_kV_LN"]),
                float(row["V1_deg"]),
            )
    return out


def read_elem_currents(path: str) -> Dict[Tuple[str, int], Tuple[float, float]]:
    out = {}
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            out[(row["element"], int(row["terminal"]))] = (
                float(row["I1_A"]),
                float(row["I1_deg"]),
            )
    return out


def read_elem_powers(path: str) -> Dict[Tuple[str, int], Tuple[float, float]]:
    out = {}
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            out[(row["element"], int(row["terminal"]))] = (
                float(row["P1_MW"]),
                float(row["Q1_Mvar"]),
            )
    return out


def main():
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    v_path = os.path.join(repo_dir, "IEEE_14_PosSeq_Voltages.csv")
    c_path = os.path.join(repo_dir, "IEEE_14_PosSeq_Currents.csv")
    s_path = os.path.join(repo_dir, "IEEE_14_PosSeq_Powers.csv")

    have_baseline = all(os.path.exists(p) for p in [v_path, c_path, s_path])
    if not have_baseline:
        print("No baseline CSVs found. Generating a baseline first...")
        rs.run_and_report()
        have_baseline = True

    base_v = read_bus_voltages(v_path)
    base_c = read_elem_currents(c_path)
    base_s = read_elem_powers(s_path)

    # Save copies of the baseline files
    v_base = v_path.replace(".csv", "_BASELINE.csv")
    c_base = c_path.replace(".csv", "_BASELINE.csv")
    s_base = s_path.replace(".csv", "_BASELINE.csv")
    shutil.copyfile(v_path, v_base)
    shutil.copyfile(c_path, c_base)
    shutil.copyfile(s_path, s_base)

    # Run new results (will overwrite default CSVs)
    rs.run_and_report()

    # Save copies of the new files
    v_new = v_path.replace(".csv", "_NEW.csv")
    c_new = c_path.replace(".csv", "_NEW.csv")
    s_new = s_path.replace(".csv", "_NEW.csv")
    shutil.copyfile(v_path, v_new)
    shutil.copyfile(c_path, c_new)
    shutil.copyfile(s_path, s_new)

    new_v = read_bus_voltages(v_path)
    new_c = read_elem_currents(c_path)
    new_s = read_elem_powers(s_path)

    # Compare
    tol_v = 1e-6
    tol_i = 1e-3  # Amps
    tol_p = 1e-6  # MW
    tol_q = 1e-6  # Mvar

    max_dv = 0.0
    worst_bus = None
    for bus, (pu0, _, _) in base_v.items():
        if bus in new_v:
            pu1 = new_v[bus][0]
            dv = abs(pu1 - pu0)
            if dv > max_dv:
                max_dv = dv
                worst_bus = (bus, pu0, pu1)

    max_di = 0.0
    worst_i = None
    for key, (i0, _) in base_c.items():
        if key in new_c:
            i1 = new_c[key][0]
            di = abs(i1 - i0)
            if di > max_di:
                max_di = di
                worst_i = (key, i0, i1)

    max_dp = 0.0
    worst_p = None
    max_dq = 0.0
    worst_q = None
    for key, (p0, q0) in base_s.items():
        if key in new_s:
            p1, q1 = new_s[key]
            dp = abs(p1 - p0)
            dq = abs(q1 - q0)
            if dp > max_dp:
                max_dp = dp
                worst_p = (key, p0, p1)
            if dq > max_dq:
                max_dq = dq
                worst_q = (key, q0, q1)

    print("Baseline files saved:")
    print(" ", v_base)
    print(" ", c_base)
    print(" ", s_base)
    print("New files saved:")
    print(" ", v_new)
    print(" ", c_new)
    print(" ", s_new)

    print("\nEquivalence summary (abs diffs):")
    print(f"  Max |ΔV1_pu| = {max_dv:.8g} at {worst_bus[0] if worst_bus else 'N/A'}")
    print(f"  Max |ΔI1_A|  = {max_di:.8g} at {worst_i[0] if worst_i else 'N/A'}")
    print(f"  Max |ΔP1|   = {max_dp:.8g} MW at {worst_p[0] if worst_p else 'N/A'}")
    print(f"  Max |ΔQ1|   = {max_dq:.8g} Mvar at {worst_q[0] if worst_q else 'N/A'}")

    ok = (max_dv <= tol_v) and (max_di <= tol_i) and (max_dp <= tol_p) and (max_dq <= tol_q)
    print("\nResult:")
    if ok:
        print("  PASS: New results are equivalent within tolerances.")
    else:
        print("  FAIL: Differences exceed tolerances.")


if __name__ == "__main__":
    main()

