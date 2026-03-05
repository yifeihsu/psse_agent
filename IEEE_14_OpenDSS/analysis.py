
import pandapower as pp
import pandapower.networks as pn

def run_pandapower():
    """
    Runs the pandapower power flow on the IEEE 14 bus system.
    """
    print("Running pandapower analysis...")
    net = pn.case14()
    pp.runpp(net)
    print("Pandapower results:")
    print(net.res_bus)
    print(net.res_line)
    print("Pandapower bus info:")
    print(net.bus)
    return net

import opendssdirect as dss
import os
import math

def run_opendss():
    """
    Runs the OpenDSS power flow on the IEEE 14 bus system.
    """
    print("\nRunning OpenDSS analysis...")

    # Use the OpenDSS files from the repository root (not Bak/)
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    dss.Basic.DataPath(repo_dir)

    candidate = None
    root_master = os.path.join(repo_dir, "Run_IEEE14Bus.dss")
    alt_master = os.path.join(repo_dir, "Run_IEEE14Bus (1).dss")
    if os.path.isfile(root_master):
        candidate = os.path.basename(root_master)
    elif os.path.isfile(alt_master):
        candidate = os.path.basename(alt_master)

    if not candidate:
        raise FileNotFoundError(
            "Could not find an OpenDSS master file in repo root. Checked: "
            f"{os.path.join(repo_dir, 'Run_IEEE14Bus.dss')}, "
            f"{os.path.join(repo_dir, 'Run_IEEE14Bus (1).dss')}"
        )

    # Execute the selected loading strategy (redirect to the run script)
    dss.run_command(f"Redirect {candidate}")

    # Ensure a circuit is active
    if dss.Circuit.NumBuses() == 0:
        raise RuntimeError("OpenDSS did not create an active circuit after redirect.")

    # Use positive-sequence quantities for robust cross-tool comparison
    voltages = {}
    for bus in dss.Circuit.AllBusNames():
        dss.Circuit.SetActiveBus(bus)
        seq = dss.Bus.SeqVoltages()  # [V0_mag(V), V1_mag(V), V2_mag(V)]
        kvbase_ln = dss.Bus.kVBase()  # LN base kV
        v1_kv_ln = (seq[1] / 1000.0) if len(seq) >= 2 else 0.0
        v1_pu = (v1_kv_ln / kvbase_ln) if kvbase_ln else 0.0
        # Store directly as per-unit magnitude
        voltages[bus.lower()] = v1_pu

    powers = {}
    for elem in dss.Circuit.AllElementNames():
        dss.Circuit.SetActiveElement(elem)
        sp = dss.CktElement.SeqPowers()  # per terminal: [P0, Q0, P1, Q1, P2, Q2]
        total_mw = 0.0
        if sp and len(sp) >= 4:
            p1_kw = sp[2]  # terminal 1 positive-sequence active power (kW)
            total_mw = p1_kw / 1000.0
        powers[elem.lower()] = total_mw

    print("OpenDSS results:")
    print("Bus Voltages (p.u.):")
    for bus, vpu in voltages.items():
        print(f"  {bus}: {abs(vpu):.4f} pu")

    return voltages, powers



def compare_results(pandapower_net, opendss_voltages, opendss_powers):
    """
    Compares the pandapower and OpenDSS results.
    """
    print("\nComparison of bus voltages (p.u.):")
    print("-" * 40)
    print("{:<10} {:<15} {:<15}".format("Bus", "Pandapower", "OpenDSS"))
    print("-" * 40)

    bus_mapping = {
        0: "b1",
        1: "b2",
        2: "b3",
        3: "b4",
        4: "b5",
        5: "b6",
        6: "b7",
        7: "b8",
        8: "b9",
        9: "b10",
        10: "b11",
        11: "b12",
        12: "b13",
        13: "b14",
    }

    for pp_bus_index, opendss_bus_name in bus_mapping.items():
        pp_voltage = pandapower_net.res_bus.vm_pu.iloc[pp_bus_index]
        opendss_voltage = abs(opendss_voltages.get(opendss_bus_name.lower(), 0))
        print("{:<10} {:<15.4f} {:<15.4f}".format(f"Bus {pp_bus_index + 1}", pp_voltage, opendss_voltage))

    print("\nComparison of line power flows (MW):")
    print("-" * 50)
    print("{:<15} {:<20} {:<20}".format("Line", "Pandapower (p_from_mw)", "OpenDSS (MW)"))
    print("-" * 50)

    line_mapping = {
        0: "Line.1-2",
        1: "Line.1-5",
        2: "Line.2-3",
        3: "Line.2-4",
        4: "Line.2-5",
        5: "Line.3-4",
        6: "Line.4-5",
        7: "Line.6-11",
        8: "Line.6-12",
        9: "Line.6-13",
        10: "Line.9-10",
        11: "Line.9-14",
        12: "Line.10-11",
        13: "Line.12-13",
        14: "Line.13-14",
    }

    for pp_line_index, opendss_line_name in line_mapping.items():
        if pp_line_index < len(pandapower_net.res_line):
            pp_power = pandapower_net.res_line.p_from_mw.iloc[pp_line_index]
            opendss_power = opendss_powers.get(opendss_line_name.lower(), 0)
            if isinstance(opendss_power, complex):
                opendss_power = opendss_power.real / 1000  # Convert from kW to MW
            print("{:<15} {:<20.4f} {:<20.4f}".format(f"L{pp_line_index + 1}", pp_power, opendss_power))

if __name__ == '__main__':
    pandapower_net = run_pandapower()
    opendss_voltages, opendss_powers = run_opendss()
    compare_results(pandapower_net, opendss_voltages, opendss_powers)
