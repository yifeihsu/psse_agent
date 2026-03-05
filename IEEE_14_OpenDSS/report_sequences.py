import csv
import math
import os

import opendssdirect as dss


def run_and_report():
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    dss.Basic.DataPath(repo_dir)
    dss.run_command("Redirect Run_IEEE14Bus.dss")

    if dss.Circuit.NumBuses() == 0:
        raise RuntimeError("No active circuit after loading Run_IEEE14Bus.dss")

    volt_rows = []
    for bus in dss.Circuit.AllBusNames():
        dss.Circuit.SetActiveBus(bus)
        kvbase_ln = dss.Bus.kVBase()
        seq = dss.Bus.SeqVoltages()
        cseq = dss.Bus.CplxSeqVoltages()
        v1_mag_kv = seq[1] / 1000.0
        v1_re, v1_im = cseq[2], cseq[3]
        v1_ang_deg = math.degrees(math.atan2(v1_im, v1_re))
        v1_pu = v1_mag_kv / (kvbase_ln if kvbase_ln else 1.0)
        volt_rows.append({
            "bus": bus,
            "V1_kV_LN": f"{v1_mag_kv:.6f}",
            "V1_pu": f"{v1_pu:.6f}",
            "V1_deg": f"{v1_ang_deg:.6f}",
        })

    cur_rows = []
    pwr_rows = []
    for elem in dss.Circuit.AllElementNames():
        dss.Circuit.SetActiveElement(elem)
        nterm = dss.CktElement.NumTerminals()
        seqI = dss.CktElement.SeqCurrents()
        cseqI = dss.CktElement.CplxSeqCurrents()
        seqS = dss.CktElement.SeqPowers()
        for t in range(nterm):
            i1_mag = seqI[t * 3 + 1] if len(seqI) >= (t + 1) * 3 else 0.0
            i1_re = cseqI[t * 6 + 2] if len(cseqI) >= (t + 1) * 6 else 0.0
            i1_im = cseqI[t * 6 + 3] if len(cseqI) >= (t + 1) * 6 else 0.0
            i1_ang = math.degrees(math.atan2(i1_im, i1_re)) if (i1_re or i1_im) else 0.0
            cur_rows.append({
                "element": elem,
                "terminal": t + 1,
                "I1_A": f"{i1_mag:.6f}",
                "I1_deg": f"{i1_ang:.6f}",
            })

            p1_kw = seqS[t * 6 + 2] if len(seqS) >= (t + 1) * 6 else 0.0
            q1_kvar = seqS[t * 6 + 3] if len(seqS) >= (t + 1) * 6 else 0.0
            pwr_rows.append({
                "element": elem,
                "terminal": t + 1,
                "P1_kW": f"{p1_kw:.6f}",
                "Q1_kvar": f"{q1_kvar:.6f}",
                "P1_MW": f"{(p1_kw/1000.0):.6f}",
                "Q1_Mvar": f"{(q1_kvar/1000.0):.6f}",
            })

    vfile = os.path.join(repo_dir, "IEEE_14_PosSeq_Voltages.csv")
    cfile = os.path.join(repo_dir, "IEEE_14_PosSeq_Currents.csv")
    sfile = os.path.join(repo_dir, "IEEE_14_PosSeq_Powers.csv")

    with open(vfile, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["bus", "V1_kV_LN", "V1_pu", "V1_deg"])
        w.writeheader()
        w.writerows(volt_rows)

    with open(cfile, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["element", "terminal", "I1_A", "I1_deg"])
        w.writeheader()
        w.writerows(cur_rows)

    with open(sfile, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["element", "terminal", "P1_kW", "Q1_kvar", "P1_MW", "Q1_Mvar"])
        w.writeheader()
        w.writerows(pwr_rows)

    return vfile, cfile, sfile, volt_rows[:5], cur_rows[:5], pwr_rows[:5]


if __name__ == "__main__":
    vfile, cfile, sfile, vprev, cprev, sprev = run_and_report()
    print("Wrote:")
    print(" ", vfile)
    print(" ", cfile)
    print(" ", sfile)
    print("\nVoltages preview:")
    for r in vprev:
        print(r)
    print("\nCurrents preview:")
    for r in cprev:
        print(r)
    print("\nPowers preview:")
    for r in sprev:
        print(r)

