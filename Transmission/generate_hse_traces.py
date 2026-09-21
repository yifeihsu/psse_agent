
import sys
import os
import json
import numpy as np
import math
import random
from numbers import Integral
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

HARMONIC_THD_PROFILES = {
    "sensitivity": (0.01, 0.05),
    "stress": (0.10, 0.20),
}


def draw_harmonic_target_thd(rng, profile: str = "stress") -> float:
    """Draw source-bus voltage THD from a named research population.

    These profiles are experimental cohorts, not regulatory acceptance limits.
    A separate draw lets paired cases share one explicit operating point.
    """
    if profile not in HARMONIC_THD_PROFILES:
        raise ValueError(f"unknown harmonic THD profile: {profile!r}")
    low, high = HARMONIC_THD_PROFILES[profile]
    return float(rng.uniform(low, high))


def _positive_sigma(value, name: str) -> float:
    parsed = float(value)
    if isinstance(value, bool) or not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{name} must be finite and positive; use z_scada_true for noiseless means")
    return parsed


def complex_to_list(c):
    return [float(c.real), float(c.imag)]

def build_trace(
    source_bus_1based: int,
    target_thd: float,
    seed: int,
    harmonic_orders=[5, 7, 11, 13, 17, 19],
    bus=None,
    branch=None,
    *,
    sigma_vm: float = 0.001,
    sigma_pq: float = 0.01,
    voltage_measurement: str = "true_rms",
):
    """Synthesize one harmonic-distortion snapshot.

    ``bus``/``branch`` optionally supply the fundamental operating point (MATPOWER
    columns; VM/VA and PD/QD are read from ``bus``). They default to the module's
    stored case14 planning solution, which is the legacy corpus behaviour.

    ``voltage_measurement`` selects only Vm: ``fundamental`` exports |V1|;
    the compatible default ``true_rms`` exports sqrt(|V1|^2 + sum_h |Vh|^2).
    P includes fundamental plus harmonic active power and Q retains the
    existing 60-Hz-tuned legacy quadrature transducer. Changing the voltage
    convention does not redefine power channels. SCADA sigmas are absolute
    per-unit standard deviations.
    """
    if voltage_measurement not in {"fundamental", "true_rms"}:
        raise ValueError("voltage_measurement must be fundamental or true_rms")
    sigma_vm = _positive_sigma(sigma_vm, "sigma_vm")
    sigma_pq = _positive_sigma(sigma_pq, "sigma_pq")
    target_thd = float(target_thd)
    if not math.isfinite(target_thd) or target_thd < 0:
        raise ValueError("target_thd must be a finite nonnegative voltage-THD ratio")
    harmonic_orders = list(harmonic_orders)
    if (not harmonic_orders or any(isinstance(h, bool) or not isinstance(h, Integral)
                                  or h not in ABB_6PULSE_WITH_CHOKE for h in harmonic_orders)
            or len(set(harmonic_orders)) != len(harmonic_orders)):
        raise ValueError("harmonic_orders must be distinct supported six-pulse harmonic orders")
    rng = np.random.default_rng(seed)
    bus = BUS if bus is None else np.asarray(bus, dtype=float)[:, :13]
    branch = BRANCH if branch is None else np.asarray(branch, dtype=float)[:, :13]
    nb = bus.shape[0]
    if (isinstance(source_bus_1based, bool) or not isinstance(source_bus_1based, Integral)
            or not 1 <= source_bus_1based <= nb):
        raise ValueError("source_bus_1based must identify one bus in the supplied network")
    source_bus_1based = int(source_bus_1based)
    
    # 1. Fundamental
    V1 = fundamental_bus_voltages(bus)
    Ybus1 = build_ybus(bus, branch, BASE_MVA)
    I1_load = fundamental_load_currents(bus, V1, BASE_MVA)
    
    # 2. Harmonic Injections (Unit Scale)
    src_idx = [source_bus_1based - 1]
    spectrum = ABB_6PULSE_WITH_CHOKE
    
    # Inject at unit scale to find scaling factor
    Iinj_unit = make_harmonic_current_injections(nb, src_idx, spectrum, I1_load, rng, inj_scale=1.0)
    V_unit = solve_all_harmonics(bus, branch, harmonic_orders, Iinj_unit, BASE_MVA, slack_bus=0)
    
    # 3. Scale to Target THD
    # The requested severity is source-bus voltage THD, not network maximum.
    thd_unit_at_source = h_ver.voltage_thd(V_unit, source_bus_1based - 1, harmonic_orders)
    
    if thd_unit_at_source < 1e-12:
        if target_thd > 0:
            raise ValueError("the selected source has no harmonic response to scale to target_thd")
        scale = 0.0
    else:
        scale = target_thd / thd_unit_at_source
        
    Iinj_final = {h: scale * Ivec for h, Ivec in Iinj_unit.items()}
    Iinj_final[1] = Ybus1 @ V1 # Fundamental injection for SCADA consistency (net)
    
    V_final = solve_all_harmonics(bus, branch, harmonic_orders, Iinj_final, BASE_MVA, slack_bus=0)
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
            If, It = branch_terminal_currents_both(V1, branch, 1)
        else:
            If, It = branch_terminal_currents_both(V_final[h], branch, h)
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
        vm_val = (float(abs(V1[i])) if voltage_measurement == "fundamental" else
                  meas_sim.measure_voltage_magnitude(i, V_final, harmonic_orders))
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
    nl = branch.shape[0]
    
    # Helper for branch power
    def get_flow_reading(k, side_is_to):
        bus_idx = int(branch[k, 1]) - 1 if side_is_to else int(branch[k, 0]) - 1
        I_dict = Ibranch_t_by_h if side_is_to else Ibranch_f_by_h
        
        Vpoint = {}
        Ipoint = {}
        for h in all_h:
            if h in V_final: Vpoint[h] = V_final[h][bus_idx]
            if h in I_dict: Ipoint[h] = I_dict[h][k]
            
        # Branch power includes the fundamental just like bus injections.
        # Restricting to h > 1 made the zero-harmonic control have zero flows.
        return meas_sim.measure_PQ_pu(Vpoint, Ipoint, harmonic_orders=all_h)

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
    sigma_z = [sigma_vm] * nb + [sigma_pq] * (2 * nb + 4 * nl)
    z_noise = []
    
    # Vm
    for _ in range(nb): z_noise.append(rng.standard_normal() * sigma_vm)
    # Pinj
    for _ in range(nb): z_noise.append(rng.standard_normal() * sigma_pq)
    # Qinj
    for _ in range(nb): z_noise.append(rng.standard_normal() * sigma_pq)
    # Flows (4 * nl)
    for _ in range(4 * nl): z_noise.append(rng.standard_normal() * sigma_pq)
    
    z_meas = (np.array(z_true) + np.array(z_noise)).tolist()
    
    # Harmonic Phasors (PMU)
    # Add small noise (e.g. 1e-4 pu)
    sigma_pmu = 1e-4  # RMS of the complex error, not either component's sigma.
    sigma_pmu_component = sigma_pmu / math.sqrt(2)
    vh_meas = {}
    
    for h in harmonic_orders:
        vh_true = V_final[h]
        noise = sigma_pmu_component * (rng.standard_normal(nb) + 1j * rng.standard_normal(nb))
        vh_noisy = vh_true + noise
        
        vh_meas[str(h)] = []
        for i in range(nb):
             vh_meas[str(h)].append({
                 "bus_1based": i + 1,
                 "V_complex_true": complex_to_list(vh_true[i]),
                 "V_complex_noisy": complex_to_list(vh_noisy[i]),
                 "sigma": sigma_pmu_component,
                 "sigma_semantics": "per_component",
                 "sigma_complex_rms": sigma_pmu,
             })

    actual_thd = h_ver.voltage_thd(V_final, source_bus_1based - 1, harmonic_orders)
    bus_thd = [float(h_ver.voltage_thd(V_final, i, harmonic_orders)) for i in range(nb)]
    maximum_bus = int(np.argmax(bus_thd))
            
    return {
        "source_bus_1based": source_bus_1based,
        "target_thd": target_thd,
        "actual_thd": actual_thd,
        "measurement_semantics": {
            "voltage_measurement": voltage_measurement,
            "voltage_formula": "abs(V1)" if voltage_measurement == "fundamental" else "sqrt(abs(V1)**2 + sum(abs(Vh)**2 for h > 1))",
            "active_power": "sum real(Vh * conj(Ih)) including fundamental",
            "reactive_power": "legacy 60-Hz-tuned all-pass quadrature, including fundamental",
            "quadrature_transfer": "H(h) = (1 - j*h) / (1 + j*h)",
            "power_measurement_unchanged_by_voltage_mode": True,
            "fundamental_operating_point_held_fixed": True,
            "measurement_order": ["Vm", "Pinj", "Qinj", "Pf", "Qf", "Pt", "Qt"],
        },
        "physical_severity": {
            "source_voltage_thd": float(actual_thd),
            "maximum_voltage_thd": bus_thd[maximum_bus],
            "maximum_voltage_thd_bus_1based": maximum_bus + 1,
            "voltage_thd_by_bus": bus_thd,
            "thd_units": "ratio_to_fundamental_voltage_magnitude",
            "source_count": 1 if target_thd > 0 else 0,
            "source_bus_1based": source_bus_1based,
            "harmonic_orders": [int(h) for h in harmonic_orders],
            "current_injection_scale": float(scale),
        },
        "z_scada_meas": z_meas,
        "z_scada_true": z_true,
        "sigma_z": sigma_z,
        "noise_contract": {
            "distribution": "gaussian", "sigma_semantics": "per_component",
            "scada_applied_matches_declared": True,
            "scada_sigma_vm": sigma_vm,
            "scada_sigma_power": sigma_pq,
            "harmonic_phasor_representation": "complex_rectangular",
            "harmonic_sigma_per_component": sigma_pmu_component,
            "harmonic_sigma_complex_rms": sigma_pmu,
            "seed": int(seed),
            "reference_role": "same_physics_noiseless_reference",
            "selection_filter": "none_on_noise_draws",
        },
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
