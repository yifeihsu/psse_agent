
import sys
import os
import json
import numpy as np
import math
import random
from tqdm import tqdm

# Add project roots to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Harmonics')))

from Harmonics import ieee14_verification as h_ver
from Harmonics.ieee14_verification import (
    BUS, BRANCH, BASE_MVA, ABB_6PULSE_WITH_CHOKE,
    fundamental_bus_voltages, fundamental_load_currents,
    make_harmonic_current_injections, solve_all_harmonics,
    build_ybus,
    LegacyAnalogWattVarTransducer
)

def complex_to_list(c):
    return [float(c.real), float(c.imag)]

def build_trace(
    source_bus_1based: int,
    target_thd: float,
    seed: int,
    harmonic_orders=[5, 7, 11, 13, 17, 19]
):
    rng = np.random.default_rng(seed)
    nb = BUS.shape[0]
    
    # 1. Fundamental
    V1 = fundamental_bus_voltages(BUS)
    Ybus1 = build_ybus(BUS, BRANCH, BASE_MVA)
    I1_load = fundamental_load_currents(BUS, V1, BASE_MVA)
    
    # 2. Harmonic Injections (Unit Scale)
    src_idx = [source_bus_1based - 1]
    spectrum = ABB_6PULSE_WITH_CHOKE
    
    # Inject at unit scale to find scaling factor
    Iinj_unit = make_harmonic_current_injections(nb, src_idx, spectrum, I1_load, rng, inj_scale=1.0)
    V_unit = solve_all_harmonics(BUS, BRANCH, harmonic_orders, Iinj_unit, BASE_MVA, slack_bus=0)
    
    # 3. Scale to Target THD
    # THD is defined at the source bus usually, or max THD? 
    # Let's target THD at the source bus.
    thd_unit_at_source = h_ver.voltage_thd(V_unit, source_bus_1based - 1, harmonic_orders)
    
    if thd_unit_at_source < 1e-12:
        scale = 0.0
    else:
        scale = target_thd / thd_unit_at_source
        
    Iinj_final = {h: scale * Ivec for h, Ivec in Iinj_unit.items()}
    Iinj_final[1] = Ybus1 @ V1 # Fundamental injection for SCADA consistency (net)
    
    V_final = solve_all_harmonics(BUS, BRANCH, harmonic_orders, Iinj_final, BASE_MVA, slack_bus=0)
    V_final[1] = V1 # Ensure fundamental is correct
    
    # 4. Generate SCADA Measurements (Legacy Transducer)
    # Reconstruct time-domain or use approximation for legacy transducer?
    # ieee14_verification.LegacyAnalogWattVarTransducer uses V_by_h/I_by_h to compute readings
    
    # Need branch currents (h)
    Ibranch_f_by_h = {}
    Ibranch_t_by_h = {}
    from Harmonics.ieee14_verification import branch_terminal_currents_both
    
    # We need to process harmonic current calc. 
    # V_final contains 1 and harmonics.
    all_h = [1] + harmonic_orders
    
    for h in all_h:
        if h == 1:
            If, It = branch_terminal_currents_both(V1, BRANCH, 1)
        else:
            If, It = branch_terminal_currents_both(V_final[h], BRANCH, h)
        Ibranch_f_by_h[h] = If
        Ibranch_t_by_h[h] = It
            
    # Measure
    meas_sim = LegacyAnalogWattVarTransducer()
    
    # Fundamental SCADA measurements (Vm, P, Q, Pf, Qf, Pt, Qt)
    z_true = []
    
    # Order: Vm(nb), Pinj(nb), Qinj(nb), Pf(nl), Qf(nl), Pt(nl), Qt(nl)
    # Vm
    vm_true = []
    for i in range(nb):
        vm_val = meas_sim.measure_voltage_magnitude(i, V_final, harmonic_orders)
        vm_true.append(vm_val)
        z_true.append(vm_val)
        
    # Pinj, Qinj
    for i in range(nb):
        p, q = meas_sim.measure_injection_power(i, V_final, Iinj_final, harmonic_orders)
        z_true.append(p)
    for i in range(nb):
        p, q = meas_sim.measure_injection_power(i, V_final, Iinj_final, harmonic_orders)
        z_true.append(q)
        
    # Flows
    nl = BRANCH.shape[0]
    
    # Helper for branch power
    def get_flow_reading(k, side_is_to):
        bus_idx = int(BRANCH[k, 1]) - 1 if side_is_to else int(BRANCH[k, 0]) - 1
        I_dict = Ibranch_t_by_h if side_is_to else Ibranch_f_by_h
        
        Vpoint = {}
        Ipoint = {}
        for h in all_h:
            if h in V_final: Vpoint[h] = V_final[h][bus_idx]
            if h in I_dict: Ipoint[h] = I_dict[h][k]
            
        return meas_sim.measure_PQ_pu(Vpoint, Ipoint, harmonic_orders=harmonic_orders)

    # Pf
    for k in range(nl):
        p, q = get_flow_reading(k, side_is_to=False)
        z_true.append(p)
    # Qf
    for k in range(nl):
        p, q = get_flow_reading(k, side_is_to=False)
        z_true.append(q)
    # Pt
    for k in range(nl):
        p, q = get_flow_reading(k, side_is_to=True)
        z_true.append(p)
    # Qt
    for k in range(nl):
        p, q = get_flow_reading(k, side_is_to=True)
        z_true.append(q)
        
    # Add noise to z
    sigma_v = 0.001
    sigma_pq = 0.01
    z_noise = []
    
    # Vm
    for _ in range(nb): z_noise.append(rng.standard_normal() * sigma_v)
    # Pinj
    for _ in range(nb): z_noise.append(rng.standard_normal() * sigma_pq)
    # Qinj
    for _ in range(nb): z_noise.append(rng.standard_normal() * sigma_pq)
    # Flows (4 * nl)
    for _ in range(4 * nl): z_noise.append(rng.standard_normal() * sigma_pq)
    
    z_meas = (np.array(z_true) + np.array(z_noise)).tolist()
    
    # Harmonic Phasors (PMU)
    # Add small noise (e.g. 1e-4 pu)
    sigma_pmu = 1e-4
    vh_meas = {}
    
    for h in harmonic_orders:
        vh_true = V_final[h]
        noise = (sigma_pmu / math.sqrt(2)) * (rng.standard_normal(nb) + 1j * rng.standard_normal(nb))
        vh_noisy = vh_true + noise
        
        vh_meas[str(h)] = []
        for i in range(nb):
             vh_meas[str(h)].append({
                 "bus_1based": i + 1,
                 "V_complex_true": complex_to_list(vh_true[i]),
                 "V_complex_noisy": complex_to_list(vh_noisy[i]),
                 "sigma": sigma_pmu
             })

    actual_thd = h_ver.voltage_thd(V_final, source_bus_1based - 1, harmonic_orders)
            
    return {
        "source_bus_1based": source_bus_1based,
        "target_thd": target_thd,
        "actual_thd": actual_thd,
        "z_scada_meas": z_meas,
        "z_scada_true": z_true,
        "harmonic_phasors": vh_meas
    }

def main():
    out_file = os.path.join(os.path.dirname(__file__), 'hse_samples.jsonl')
    print(f"Generating HSE traces to {out_file}...")
    
    num_samples = 50
    candidates = [2, 3, 4, 5, 9, 10, 11, 12, 13, 14] # Load buses/generator buses (skip slack 1)
    
    with open(out_file, 'w') as f:
        for i in tqdm(range(num_samples)):
            seed = 1000 + i
            rng = random.Random(seed)
            
            src = rng.choice(candidates)
            thd = rng.uniform(0.10, 0.20)
            
            trace = build_trace(src, thd, seed)
            trace["id"] = f"hse_trace_{i:03d}"
            
            f.write(json.dumps(trace) + "\n")
            
    print("Done.")

if __name__ == "__main__":
    main()
