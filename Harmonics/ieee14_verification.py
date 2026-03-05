import math
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np


# -----------------------------
# 1) IEEE 14-bus (MATPOWER-like) data (bus, branch)
# -----------------------------
BASE_MVA = 100.0

# BUS columns:
# [BUS_I, BUS_TYPE, PD, QD, GS, BS, BUS_AREA, VM, VA(deg), BASE_KV, ZONE, VMAX, VMIN]
BUS = np.array([
    [1,3,0,0,0,0,1,1.06,0,0,1,1.06,0.94],
    [2,2,21.7,12.7,0,0,1,1.045,-4.98,0,1,1.06,0.94],
    [3,2,94.2,19,0,0,1,1.01,-12.72,0,1,1.06,0.94],
    [4,1,47.8,-3.9,0,0,1,1.019,-10.33,0,1,1.06,0.94],
    [5,1,7.6,1.6,0,0,1,1.02,-8.78,0,1,1.06,0.94],
    [6,2,11.2,7.5,0,0,1,1.07,-14.22,0,1,1.06,0.94],
    [7,1,0,0,0,0,1,1.062,-13.37,0,1,1.06,0.94],
    [8,2,0,0,0,0,1,1.09,-13.36,0,1,1.06,0.94],
    [9,1,29.5,16.6,0,19,1,1.056,-14.94,0,1,1.06,0.94],
    [10,1,9,5.8,0,0,1,1.051,-15.1,0,1,1.06,0.94],
    [11,1,3.5,1.8,0,0,1,1.057,-14.79,0,1,1.06,0.94],
    [12,1,6.1,1.6,0,0,1,1.055,-15.07,0,1,1.06,0.94],
    [13,1,13.5,5.8,0,0,1,1.05,-15.16,0,1,1.06,0.94],
    [14,1,14.9,5,0,0,1,1.036,-16.04,0,1,1.06,0.94],
], dtype=float)

# BRANCH columns:
# [F_BUS, T_BUS, R, X, B, RATEA, RATEB, RATEC, RATIO, ANGLE, STATUS, ANGMIN, ANGMAX]
BRANCH = np.array([
    [1,2,0.01938,0.05917,0.0528,1,0,0,0,0,1,-60,60],
    [1,5,0.05403,0.22304,0.0492,489,0,0,0,0,1,-60,60],
    [2,3,0.04699,0.19797,0.0438,552,0,0,0,0,1,-60,60],
    [2,4,0.05811,0.17632,0.034,605,0,0,0,0,1,-60,60],
    [2,5,0.05695,0.17388,0.0346,614,0,0,0,0,1,-60,60],
    [3,4,0.06701,0.17103,0.0128,611,0,0,0,0,1,-60,60],
    [4,5,0.01335,0.04211,0.0,2543,0,0,0,0,1,-60,60],
    [4,7,0.0,0.20912,0.0,537,0,0,0.978,0,1,-60,60],
    [4,9,0.0,0.55618,0.0,202,0,0,0.969,0,1,-60,60],
    [5,6,0.0,0.25202,0.0,445,0,0,0.932,0,1,-60,60],
    [6,11,0.09498,0.1989,0.0,509,0,0,0,0,1,-60,60],
    [6,12,0.12291,0.25581,0.0,395,0,0,0,0,1,-60,60],
    [6,13,0.06615,0.13027,0.0,769,0,0,0,0,1,-60,60],
    [7,8,0.0,0.17615,0.0,637,0,0,0,0,1,-60,60],
    [7,9,0.0,0.11001,0.0,1021,0,0,0,0,1,-60,60],
    [9,10,0.03181,0.0845,0.0,1244,0,0,0,0,1,-60,60],
    [9,14,0.12711,0.27038,0.0,376,0,0,0,0,1,-60,60],
    [10,11,0.08205,0.19207,0.0,537,0,0,0,0,1,-60,60],
    [12,13,0.22092,0.19988,0.0,377,0,0,0,0,1,-60,60],
    [13,14,0.17093,0.34802,0.0,289,0,0,0,0,1,-60,60],
], dtype=float)


# -----------------------------
# 2) Harmonic source spectra (ABB "typical harmonic current components")
#    Values are Ih/I1 at 100% load.
# -----------------------------
# 6-pulse rectifier without choke: 5th 63%, 7th 54%, 11th 10%, 13th 6.1%, 17th 6.7%, 19th 4.8% :contentReference[oaicite:4]{index=4}
ABB_6PULSE_NO_CHOKE = {5: 0.63, 7: 0.54, 11: 0.10, 13: 0.061, 17: 0.067, 19: 0.048}
# 6-pulse rectifier with choke: 5th 30%, 7th 12%, 11th 8.9%, 13th 5.6%, 17th 4.4%, 19th 4.1% :contentReference[oaicite:5]{index=5}
ABB_6PULSE_WITH_CHOKE = {5: 0.30, 7: 0.12, 11: 0.089, 13: 0.056, 17: 0.044, 19: 0.041}


# -----------------------------
# 3) Network modeling helpers (Ybus and branch terminal currents)
# -----------------------------
def _tap_complex(ratio: float, angle_deg: float) -> complex:
    """MATPOWER convention: tap on 'from' side. ratio=0 means 1.0; angle in degrees."""
    if ratio == 0:
        ratio = 1.0
    return ratio * np.exp(1j * np.deg2rad(angle_deg))


def build_ybus(bus: np.ndarray, branch: np.ndarray, base_mva: float) -> np.ndarray:
    """Fundamental Ybus from MATPOWER-like bus/branch arrays."""
    nb = bus.shape[0]
    Y = np.zeros((nb, nb), dtype=complex)

    # Bus shunts (Gs + jBs) in per unit
    Gs = bus[:, 4] / base_mva
    Bs = bus[:, 5] / base_mva
    Y[np.arange(nb), np.arange(nb)] += (Gs + 1j * Bs)

    for br in branch:
        f = int(br[0]) - 1
        t = int(br[1]) - 1
        r, x, b = float(br[2]), float(br[3]), float(br[4])
        tap = _tap_complex(float(br[8]), float(br[9]))

        y = 0.0 if (r == 0 and x == 0) else 1.0 / complex(r, x)
        ysh = 1j * b / 2.0

        Yff = (y + ysh) / (tap * np.conj(tap))
        Ytt = (y + ysh)
        Yft = -y / np.conj(tap)
        Ytf = -y / tap

        Y[f, f] += Yff
        Y[t, t] += Ytt
        Y[f, t] += Yft
        Y[t, f] += Ytf

    return Y


def scale_branch_params(r1: float, x1: float, b1: float, h: int,
                        r_model: str = "sqrt") -> Tuple[float, float, float]:
    """
    Simple frequency scaling:
      - X(h) = h X(1)  (inductive)
      - B(h) = h B(1)  (capacitive line charging)
      - R(h): choose a skin-effect surrogate; default sqrt(h)
    """
    if r_model == "sqrt":
        r = r1 * math.sqrt(h)
    elif r_model == "linear":
        r = r1 * h
    elif r_model == "const":
        r = r1
    else:
        raise ValueError("Unknown r_model")

    x = x1 * h
    b = b1 * h
    return r, x, b


def build_ybus_h(bus: np.ndarray, branch: np.ndarray, h: int, base_mva: float,
                 r_model: str = "sqrt") -> np.ndarray:
    """Harmonic Ybus(h). Uses the same topology, with frequency-scaled branch parameters."""
    nb = bus.shape[0]
    Y = np.zeros((nb, nb), dtype=complex)

    # Bus shunts: simplistic scaling of susceptance with harmonic order
    Gs = bus[:, 4] / base_mva
    Bs = bus[:, 5] / base_mva
    Y[np.arange(nb), np.arange(nb)] += (Gs + 1j * (Bs * h))

    for br in branch:
        f = int(br[0]) - 1
        t = int(br[1]) - 1
        r1, x1, b1 = float(br[2]), float(br[3]), float(br[4])
        r, x, b = scale_branch_params(r1, x1, b1, h, r_model=r_model)
        tap = _tap_complex(float(br[8]), float(br[9]))

        y = 0.0 if (r == 0 and x == 0) else 1.0 / complex(r, x)
        ysh = 1j * b / 2.0

        Yff = (y + ysh) / (tap * np.conj(tap))
        Ytt = (y + ysh)
        Yft = -y / np.conj(tap)
        Ytf = -y / tap

        Y[f, f] += Yff
        Y[t, t] += Ytt
        Y[f, t] += Yft
        Y[t, f] += Ytf

    return Y


def solve_harmonic_voltages(Yh: np.ndarray, Iinj: np.ndarray,
                            slack: int = 0, Vslack: complex = 0.0 + 0.0j,
                            reg_eps: float = 1e-9) -> np.ndarray:
    """
    Solve Y(h) V(h) = I(h) with slack harmonic voltage fixed (ideal source has no harmonics).
    Vslack=0 is a common harmonic-study assumption.
    """
    nb = Yh.shape[0]
    idx = [i for i in range(nb) if i != slack]
    Yuu = Yh[np.ix_(idx, idx)] + reg_eps * np.eye(len(idx))
    Iu = Iinj[idx] - Yh[np.ix_(idx, [slack])].flatten() * Vslack

    Vu = np.linalg.solve(Yuu, Iu)
    V = np.zeros(nb, dtype=complex)
    V[slack] = Vslack
    V[idx] = Vu
    return V


def branch_terminal_currents(V: np.ndarray, branch: np.ndarray, h: int = 1,
                             r_model: str = "sqrt") -> np.ndarray:
    """
    Return If[k] = current injected into branch k at the FROM terminal.
    """
    nbranch = branch.shape[0]
    If = np.zeros(nbranch, dtype=complex)

    for k, br in enumerate(branch):
        f = int(br[0]) - 1
        t = int(br[1]) - 1

        r1, x1, b1 = float(br[2]), float(br[3]), float(br[4])
        if h == 1:
            r, x, b = r1, x1, b1
        else:
            r, x, b = scale_branch_params(r1, x1, b1, h, r_model=r_model)

        tap = _tap_complex(float(br[8]), float(br[9]))
        y = 0.0 if (r == 0 and x == 0) else 1.0 / complex(r, x)
        ysh = 1j * b / 2.0

        Yff = (y + ysh) / (tap * np.conj(tap))
        Yft = -y / np.conj(tap)

        If[k] = Yff * V[f] + Yft * V[t]

    return If

def branch_terminal_currents_both(V: np.ndarray, branch: np.ndarray, h: int = 1,
                             r_model: str = "sqrt") -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (If, It) where:
      If[k] = current injected into branch k at the FROM terminal.
      It[k] = current injected into branch k at the TO terminal.
    """
    nbranch = branch.shape[0]
    If = np.zeros(nbranch, dtype=complex)
    It = np.zeros(nbranch, dtype=complex)

    for k, br in enumerate(branch):
        f = int(br[0]) - 1
        t = int(br[1]) - 1

        r1, x1, b1 = float(br[2]), float(br[3]), float(br[4])
        if h == 1:
            r, x, b = r1, x1, b1
        else:
            r, x, b = scale_branch_params(r1, x1, b1, h, r_model=r_model)

        tap = _tap_complex(float(br[8]), float(br[9]))
        y = 0.0 if (r == 0 and x == 0) else 1.0 / complex(r, x)
        ysh = 1j * b / 2.0

        # From side
        Yff = (y + ysh) / (tap * np.conj(tap))
        Yft = -y / np.conj(tap)
        If[k] = Yff * V[f] + Yft * V[t]
        
        # To side
        Ytt = (y + ysh)
        Ytf = -y / tap
        It[k] = Ytf * V[f] + Ytt * V[t]

    return If, It



# -----------------------------
# 4) Legacy analog watt/var transducer model
# -----------------------------
@dataclass
class LegacyAnalogWattVarTransducer:
    """
    MW transducer: average of v(t)*i(t) -> sum_h Re{V(h) I(h)*}
    MVar transducer: average of v_q(t)*i(t), where v_q is voltage through a 90° shift network.
    We model the 90° shift network as a 1st-order all-pass filter:
        H(jω) = (1 - j ω RC) / (1 + j ω RC)
    with angle(H) = -2 arctan(ωRC), and -90° at ω=1/RC. :contentReference[oaicite:6]{index=6}

    This captures the key legacy behavior: the shift is tuned at 60 Hz, not constant vs frequency.
    """
    f0_hz: float = 60.0
    RC: Optional[float] = None  # if None, pick RC so that ω0 RC = 1

    def __post_init__(self):
        self.omega0 = 2 * math.pi * self.f0_hz
        if self.RC is None:
            self.RC = 1.0 / self.omega0  # quadrature at fundamental

    def H_allpass(self, h: int) -> complex:
        omega = h * self.omega0
        RC = float(self.RC)
        return (1 - 1j * omega * RC) / (1 + 1j * omega * RC)

    def measure_PQ_pu(self, V_by_h: Dict[int, complex], I_by_h: Dict[int, complex], harmonic_orders: List[int]=None) -> Tuple[float, float]:
        """
        Return (P_pu, Q_pu). Uses only harmonic orders present in both dictionaries.
        """
        P = 0.0
        Q = 0.0
        
        # If specific harmonics list provided, restrict to that. Else intersection.
        keys = set(V_by_h.keys()).intersection(I_by_h.keys())
        if harmonic_orders is not None:
             keys = keys.intersection(harmonic_orders)
             
        for h in sorted(keys):
            Vh = V_by_h[h]
            Ih = I_by_h[h]
            Sh = Vh * np.conj(Ih)
            P += Sh.real
            Vqh = self.H_allpass(h) * Vh
            Q += (Vqh * np.conj(Ih)).real
        return P, Q

    def measure_voltage_magnitude(self, bus_idx: int, V_by_h: Dict[int, np.ndarray], harmonic_orders: List[int]) -> float:
        """
        Measure RMS voltage magnitude at a bus.
        True RMS = sqrt(sum |Vh|^2).
        """
        sq_sum = 0.0
        for h in harmonic_orders:
            if h in V_by_h:
                sq_sum += abs(V_by_h[h][bus_idx])**2
        
        # Add fundamental if not in list but present in V_by_h?
        # Usually harmonic_orders includes 1. If not, we should probably check.
        if 1 not in harmonic_orders and 1 in V_by_h:
             sq_sum += abs(V_by_h[1][bus_idx])**2
             
        return math.sqrt(sq_sum)

    def measure_injection_power(self, bus_idx: int, 
                              V_by_h: Dict[int, np.ndarray], 
                              Iinj_by_h: Dict[int, np.ndarray],
                              harmonic_orders: List[int]) -> Tuple[float, float]:
        """
        Measure P/Q injection at a bus.
        """
        # Slice for this bus
        Vpoint = {}
        Ipoint = {}
        
        # Fundamental (always needed)
        if 1 in V_by_h: Vpoint[1] = V_by_h[1][bus_idx]
        if 1 in Iinj_by_h: Ipoint[1] = Iinj_by_h[1][bus_idx]
        
        for h in harmonic_orders:
            if h == 1: continue
            if h in V_by_h: Vpoint[h] = V_by_h[h][bus_idx]
            if h in Iinj_by_h: Ipoint[h] = Iinj_by_h[h][bus_idx]
            
        return self.measure_PQ_pu(Vpoint, Ipoint)

    def measure_branch_power(self, branch_idx: int,
                           V_by_h: Dict[int, np.ndarray],
                           Ibranch_by_h: Dict[int, np.ndarray],
                           harmonic_orders: List[int]) -> Tuple[float, float, float, float]:
        """
        Measure P/Q flow at FROM and TO ends of a branch.
        Note: Ibranch_by_h[h] is typically FROM current.
        TO current needs to be calculated if not provided. This legacy model assumes we have access to currents.
        For generated traces, we usually only generate FROM flows in z?
        Wait, verify_harmonics_wls generated Pf, Qf, Pt, Qt.
        So we need TO currents as well.
        """
        # This helper assumes Ibranch_by_h contains FROM currents.
        # To get TO currents, we need to calculate them, or assume usage where we only need FROM.
        # But z contains Pt/Qt.
        # Let's assume for now we only return FROM flow if TO current not available, or calc TO current.
        # Actually, let's just return FROM flow P/Q and let caller handle TO if they have It.
        # Wait, the signature I designed in generate_hse_traces was: pf, qf, pt, qt = ...
        # So I need to compute It.
        pass # Placeholder reasoning
        
        # Since I am modifying the class, I can't easily compute It inside here without Ybus/Branch info.
        # But I don't have branch info inside the class instance (it's a dataclass with just f0/RC).
        # So I will restrict this method to measure power given V and I scalars (or dictionaries of scalars).
        # Or I can pass the computed It dict.
        
        # Better: define `measure_complex_power(V_dict, I_dict)` which is `measure_PQ_pu`.
        # And let the caller (my script) manage extracting V and I for From/To ends.
        return 0,0,0,0 # Dummy



# -----------------------------
# 5) Harmonic injection builder + calibration to a target voltage THD
# -----------------------------
def fundamental_bus_voltages(bus: np.ndarray) -> np.ndarray:
    """Use VM/VA as the 'true' fundamental phasors (good enough for measurement synthesis)."""
    Vm = bus[:, 7]
    Va = np.deg2rad(bus[:, 8])
    return Vm * np.exp(1j * Va)


def fundamental_load_currents(bus: np.ndarray, V1: np.ndarray, base_mva: float) -> np.ndarray:
    """Approximate load current phasor at each bus from (PD,QD) and V1."""
    Pd = bus[:, 2]
    Qd = bus[:, 3]
    Sload_pu = (Pd + 1j * Qd) / base_mva
    # S = V * conj(I) => I = conj(S/V)
    I = np.conj(Sload_pu / V1)
    return I


def make_harmonic_current_injections(
    nb: int,
    source_bus_idx: List[int],
    spectrum: Dict[int, float],
    I1_load: np.ndarray,
    rng: np.random.Generator,
    inj_scale: float = 1.0,
) -> Dict[int, np.ndarray]:
    """
    Create harmonic current injections I(h) as a fraction of the local fundamental load current magnitude.

    Convention: Iinj is net injection into the network.
    A current-drawing nonlinear load corresponds to NEGATIVE injection at harmonic orders.
    """
    Iinj_by_h: Dict[int, np.ndarray] = {}
    for h, ratio in spectrum.items():
        Ivec = np.zeros(nb, dtype=complex)
        for i in source_bus_idx:
            mag = inj_scale * ratio * abs(I1_load[i])
            phi = rng.uniform(0.0, 2 * math.pi)
            Ivec[i] = -mag * np.exp(1j * phi)
        Iinj_by_h[h] = Ivec
    return Iinj_by_h


def voltage_thd(V_by_h: Dict[int, np.ndarray], bus_idx: int, harmonics: List[int]) -> float:
    """THD_V = sqrt(sum_{h>1} |V(h)|^2) / |V(1)|."""
    V1 = V_by_h[1][bus_idx]
    num = 0.0
    for h in harmonics:
        if h == 1:
            continue
        num += abs(V_by_h[h][bus_idx]) ** 2
    return math.sqrt(num) / abs(V1)


def solve_all_harmonics(
    bus: np.ndarray,
    branch: np.ndarray,
    harmonics: List[int],
    Iinj_by_h: Dict[int, np.ndarray],
    base_mva: float,
    slack_bus: int = 0,
    r_model: str = "sqrt",
) -> Dict[int, np.ndarray]:
    V_by_h: Dict[int, np.ndarray] = {}
    V_by_h[1] = fundamental_bus_voltages(bus)

    for h in harmonics:
        if h == 1:
            continue
        Yh = build_ybus_h(bus, branch, h, base_mva, r_model=r_model)
        Ivec = Iinj_by_h.get(h, np.zeros(bus.shape[0], dtype=complex))
        V_by_h[h] = solve_harmonic_voltages(Yh, Ivec, slack=slack_bus, Vslack=0.0 + 0.0j)

    return V_by_h


# -----------------------------
# 6) Measurement synthesis: SCADA MW/MVar line flows with legacy transducers
# -----------------------------
def scada_line_flow_measurements(
    bus: np.ndarray,
    branch: np.ndarray,
    harmonics: List[int],
    V_by_h: Dict[int, np.ndarray],
    transducer: LegacyAnalogWattVarTransducer,
    measured_branch_idx: List[int],
    base_mva: float,
    r_model: str = "sqrt",
    sigma_p_mw: float = 0.2,   # measurement noise (MW)
    sigma_q_mvar: float = 0.2, # measurement noise (MVar)
    rng: Optional[np.random.Generator] = None,
) -> List[Dict]:
    """
    Produce SCADA MW/MVar flow measurements at the FROM terminal of selected branches.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    # currents per harmonic
    If_by_h: Dict[int, np.ndarray] = {}
    for h in harmonics:
        If_by_h[h] = branch_terminal_currents(V_by_h[h], branch, h=h, r_model=r_model)

    z = []
    for k in measured_branch_idx:
        f_bus = int(branch[k, 0]) - 1

        Vpoint = {h: V_by_h[h][f_bus] for h in harmonics}
        Ipoint = {h: If_by_h[h][k] for h in harmonics}

        P_pu, Q_pu = transducer.measure_PQ_pu(Vpoint, Ipoint)
        P_mw = P_pu * base_mva
        Q_mvar = Q_pu * base_mva

        # add noise (legacy analog transducer + SCADA acquisition noise)
        P_meas = P_mw + rng.normal(0.0, sigma_p_mw)
        Q_meas = Q_mvar + rng.normal(0.0, sigma_q_mvar)

        z.append({
            "type": "SCADA_FLOW",
            "branch_idx": int(k),
            "from_bus": int(branch[k, 0]),
            "to_bus": int(branch[k, 1]),
            "P_MW": float(P_meas),
            "Q_MVar": float(Q_meas),
            "P_true_MW": float(P_mw),
            "Q_true_MVar": float(Q_mvar),
        })

    return z


# -----------------------------
# 7) Example: time series where harmonics turn on (for triggering experiments)
# -----------------------------
def run_time_series_demo(
    T: int = 60,
    dt_sec: float = 2.0,
    harmonic_on_step: int = 20,
    source_buses_1based: List[int] = [3],            # nonlinear load bus(es)
    spectrum: Dict[int, float] = ABB_6PULSE_WITH_CHOKE,
    harmonics: List[int] = [1, 5, 7, 11, 13, 17, 19],
    target_thd_bus_1based: int = 3,
    target_thd: float = 0.02,                        # target THD_V at calibration bus (2%)
    measured_branch_idx: List[int] = [2, 4, 9, 15],  # pick a few flows to observe
    seed: int = 7,
) -> List[Dict]:
    rng = np.random.default_rng(seed)
    nb = BUS.shape[0]

    # Fundamental baseline
    V1 = fundamental_bus_voltages(BUS)
    I1_load = fundamental_load_currents(BUS, V1, BASE_MVA)

    src_idx = [b - 1 for b in source_buses_1based]
    thd_idx = target_thd_bus_1based - 1

    # Build "unit-scale" harmonic injections once (fixed phases for persistence)
    Iinj_unit = make_harmonic_current_injections(nb, src_idx, spectrum, I1_load, rng, inj_scale=1.0)
    # Solve once to calibrate injection scaling to target THD
    V_unit = solve_all_harmonics(BUS, BRANCH, harmonics, Iinj_unit, BASE_MVA, slack_bus=0)
    thd_unit = voltage_thd(V_unit, thd_idx, harmonics)
    inj_scale = 0.0 if thd_unit < 1e-12 else (target_thd / thd_unit)

    # Fixed injections with calibrated scale
    Iinj_cal = {h: inj_scale * Ivec for h, Ivec in Iinj_unit.items()}

    # Transducer model (legacy analog): var uses 90° phase-shift network tuned at 60 Hz :contentReference[oaicite:7]{index=7}
    transducer = LegacyAnalogWattVarTransducer(f0_hz=60.0)

    traces = []
    for t in range(T):
        harm_active = (t >= harmonic_on_step)
        Iinj = Iinj_cal if harm_active else {}

        V_by_h = solve_all_harmonics(BUS, BRANCH, harmonics, Iinj, BASE_MVA, slack_bus=0)
        thd_now = voltage_thd(V_by_h, thd_idx, harmonics)

        meas = scada_line_flow_measurements(
            BUS, BRANCH, harmonics, V_by_h, transducer,
            measured_branch_idx=measured_branch_idx,
            base_mva=BASE_MVA,
            sigma_p_mw=0.2,
            sigma_q_mvar=0.2,
            rng=rng,
        )

        traces.append({
            "t_step": t,
            "t_sec": float(t * dt_sec),
            "harmonics_active": bool(harm_active),
            "inj_scale": float(inj_scale),
            "THD_V_at_bus": float(thd_now),
            "bus_cal": int(target_thd_bus_1based),
            "measurements": meas,
        })

    return traces


if __name__ == "__main__":
    traces = run_time_series_demo()

    # Write JSONL trace file for your LLM-agent pipeline
    with open("scada_legacy_transducer_harmonics_traces.jsonl", "w") as f:
        for row in traces:
            f.write(json.dumps(row) + "\n")

    # Quick console check: show THD around the switching event
    for row in traces[18:23]:
        print(row["t_step"], "harm_on?", row["harmonics_active"], "THD@bus", row["bus_cal"], "=", row["THD_V_at_bus"])
    print("Wrote scada_legacy_transducer_harmonics_traces.jsonl")