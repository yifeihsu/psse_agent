import os
from typing import Dict, Tuple

import opendssdirect as dss
from opendssdirect import Text


def p1q1_by_element() -> Dict[Tuple[str, int], Tuple[float, float]]:
    out: Dict[Tuple[str, int], Tuple[float, float]] = {}
    for elem in dss.Circuit.AllElementNames():
        dss.Circuit.SetActiveElement(elem)
        sp = dss.CktElement.SeqPowers()  # per terminal: [P0, Q0, P1, Q1, P2, Q2]
        nterm = dss.CktElement.NumTerminals()
        for t in range(nterm):
            idx = t * 6
            if len(sp) >= idx + 4:
                p1_mw = sp[idx + 2] / 1000.0
                q1_mvar = sp[idx + 3] / 1000.0
                out[(elem, t + 1)] = (p1_mw, q1_mvar)
    return out


def main():
    repo = os.path.dirname(os.path.abspath(__file__))
    dss.Basic.DataPath(repo)
    Text.Command("Redirect Run_IEEE14Bus.dss")
    Text.Command("Solve")

    imbal = p1q1_by_element()

    # Toggle to a balanced 3ph load at B3, disabling the three 1ph loads
    Text.Command("Edit Load.B3A enabled=no")
    Text.Command("Edit Load.B3B enabled=no")
    Text.Command("Edit Load.B3C enabled=no")
    # Add a balanced equivalent (if exists, it will be overwritten)
    Text.Command("New Load.B3 Bus1=B3 kV=1 kW=94200 kvar=19000 Phases=3")
    Text.Command("Solve")

    base = p1q1_by_element()

    # Compare Lines and Transformers at terminal 1
    def is_line(name: str) -> bool:
        return name.lower().startswith("line.")

    def is_trafo(name: str) -> bool:
        return name.lower().startswith("transformer.")

    diffs = []
    for (ename, term), (p_mw_i, q_mvar_i) in imbal.items():
        if term != 1:
            continue
        if (ename, term) not in base:
            continue
        if not (is_line(ename) or is_trafo(ename)):
            continue
        p_mw_b, q_mvar_b = base[(ename, term)]
        diffs.append((abs(p_mw_i - p_mw_b), ename, p_mw_i, p_mw_b, q_mvar_i, q_mvar_b))

    diffs.sort(reverse=True)
    print("Top 10 |ΔP1| among Lines/Transformers (imbalanced - balanced):")
    for d, ename, p_i, p_b, q_i, q_b in diffs[:10]:
        print(f"  {ename:14}  ΔP1={d:+.6f} MW   P1_i={p_i:+.6f}  P1_b={p_b:+.6f}   ΔQ1={(q_i-q_b):+.6f} Mvar")

    max_dp = diffs[0][0] if diffs else 0.0
    print(f"\nMax |ΔP1| MW across Lines/Transformers: {max_dp:.6f}")


if __name__ == "__main__":
    main()
