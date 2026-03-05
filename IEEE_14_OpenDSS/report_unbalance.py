import math
import os

import opendssdirect as dss


def vuf_percent(bus: str) -> float:
    dss.Circuit.SetActiveBus(bus)
    seq = dss.Bus.SeqVoltages()  # [V0_mag(V), V1_mag(V), V2_mag(V)]
    if not seq or len(seq) < 3:
        return 0.0
    v1 = seq[1]
    v2 = seq[2]
    return 100.0 * (abs(v2) / abs(v1)) if v1 else 0.0


def main():
    repo = os.path.dirname(os.path.abspath(__file__))
    dss.Basic.DataPath(repo)
    dss.run_command("Redirect Run_IEEE14Bus.dss")

    buses = dss.Circuit.AllBusNames()
    data = []
    for b in buses:
        vuf = vuf_percent(b)
        data.append((b, vuf))

    data.sort(key=lambda x: x[1], reverse=True)
    print("Top buses by voltage unbalance (VUF = |V2|/|V1| %):")
    for b, v in data[:10]:
        print(f"  {b:>4}: {v:8.5f} %")

    # Show per-phase VLN at B3 to illustrate impact
    target_bus = "b3"
    if target_bus in [x.lower() for x in buses]:
        dss.Circuit.SetActiveBus(target_bus)
        magang = dss.Bus.VMagAngle()
        kvbase = dss.Bus.kVBase()
        print(f"\n{target_bus.upper()} phase-to-neutral voltages (kV, deg):")
        for ph in range(3):
            vm = magang[2 * ph] / 1000.0
            va = magang[2 * ph + 1]
            print(f"  Phase {ph+1}: {vm:.6f} kV  @ {va:.2f}°  (base LN {kvbase:.6f} kV)")


if __name__ == "__main__":
    main()
