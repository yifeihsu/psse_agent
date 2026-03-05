# IEEE 14-Bus OpenDSS / pandapower Project – Agent Notes

This file summarizes the current state of the project and the conventions used so future agents can pick up work quickly.

## Overview

Goal: model the IEEE 14-bus system in both:

- A detailed three‑phase OpenDSS model.
- A positive‑sequence pandapower / MATPOWER-style model.

Then:

- Introduce three‑phase unbalance at selected buses.
- Generate consistent measurement vectors in per-unit, matching a MATLAB reference script (`generate_measurements_ieee14.m`).

## Core Files

Top-level DSS and orchestration:

- `IEEE14BusMaster.dss` – main OpenDSS circuit definition (lines, transformers, loads, caps, generators).
- `Run_IEEE14Bus.dss` – compiles the master, sets `maxiterations`, and runs the power flow.

Network data:

- `IEEE14Lines.DSS` – three‑phase line models using `LineCode` with 1 kV base; maps to MATPOWER line data.
- `IEEE14Trafo.DSS` – transformers 4‑7, 4‑9, 5‑6, with comments mapping 132/33 kV buses.
- `IEEE14Loads.DSS` – load definitions (this is where unbalance is configured).
- `IEEE14Gen.DSS`, `IEEE14Cap.DSS` – generators and shunts.

Python analysis utilities:

- `analysis.py` – compares pandapower `case14` results vs OpenDSS (positive‑sequence bus voltages and line flows).
- `report_unbalance.py` – computes bus voltage unbalance factor (VUF) and prints per‑phase VLN at a chosen bus.
- `compare_posseq_flows.py` – compares positive‑sequence line/trafo powers between unbalanced and balanced cases.
- `report_sequences.py`, `verify_equivalence.py`, `debug_opendss.py` – export/compare OpenDSS sequence quantities and CSV snapshots.

Measurement series exporters:

- `export_measurement_series_balanced.py` – generates a balanced reference measurement vector using pandapower `case14` in MATPOWER style.
- `export_measurement_series.py` – generates a three‑phase unbalanced measurement vector from the OpenDSS model (currently with Bus 3 unbalanced).

## Current Unbalance Configuration

As of this file:

- Three‑phase unbalance is implemented at **Bus 3** in `IEEE14Loads.DSS`:
  - The original 3‑phase load at Bus 3 (94.2 MW + 19 Mvar) is replaced by three 1‑phase loads:
    - `Load.B3A` at `B3.1`, `kW=47100`, `kvar=9500`
    - `Load.B3B` at `B3.2`, `kW=28260`, `kvar=5700`
    - `Load.B3C` at `B3.3`, `kW=18840`, `kvar=3800`
  - All other loads (including Bus 5, 12, 14) are balanced 3‑phase, as per the standard IEEE‑14 data.

Related scripts:

- `report_unbalance.py`:
  - Runs `Run_IEEE14Bus.dss`.
  - Computes VUF at all buses.
  - Prints per‑phase VLN at **B3** to illustrate the unbalance.
- `compare_posseq_flows.py`:
  - Baseline: current unbalanced Bus 3 configuration.
  - Balanced reference: disables `Load.B3A/B3B/B3C` and creates `Load.B3` (94.2 MW + 19 Mvar, 3‑phase).
  - Compares positive‑sequence P1/Q1 across lines/transformers between unbalanced and balanced cases.

## Measurement Vector Convention

The measurement vector matches the MATLAB script `generate_measurements_ieee14.m` (MATPOWER style):

- `nb = 14` buses, `nl = 20` branches.
- Layout (0‑based indices):
  - `0 … 13`   → `Vm(1..14)` – bus voltage magnitudes (pu).
  - `14 … 27`  → `Pinj(1..14)` – active power injections (pu).
  - `28 … 41`  → `Qinj(1..14)` – reactive power injections (pu).
  - `42 … 61`  → `Pf(1..20)` – active branch flows (from side, pu).
  - `62 … 81`  → `Qf(1..20)` – reactive branch flows (from side, pu).
  - `82 … 101` → `Pt(1..20)` – active branch flows (to side, pu).
  - `102 … 121`→ `Qt(1..20)` – reactive branch flows (to side, pu).

Injections follow MATPOWER `makeSbus` convention:

- `Sbus = (Pg − Pl) + j(Qg − Ql)` in per-unit on 100 MVA.
- `Pinj(i) = (Pg_i − Pl_i)/baseMVA`
- `Qinj(i) = (Qg_i − Ql_i)/baseMVA`

Branch order (MATPOWER IEEE‑14 `branch` table order, used everywhere):

1. 1–2  
2. 1–5  
3. 2–3  
4. 2–4  
5. 2–5  
6. 3–4  
7. 4–5  
8. 4–7 (trafo)  
9. 4–9 (trafo)  
10. 5–6 (trafo)  
11. 6–11  
12. 6–12  
13. 6–13  
14. 7–8 (modeled as trafo in pandapower)  
15. 7–9 (trafo)  
16. 9–10  
17. 9–14  
18. 10–11  
19. 12–13  
20. 13–14  

## Generated Measurement Files

Balanced reference (pandapower, MATPOWER‑style):

- Script: `export_measurement_series_balanced.py`
  - Uses `pn.case14()` from pandapower and `pp.runpp(net)`.
  - Reconstructs `Pinj/Qinj` using the MATPOWER `makeSbus` logic (summing load/gen/ext_grid per bus).
  - Extracts branch flows from `net.res_line` / `net.res_trafo` in the branch order above.
  - Writes a 122‑entry vector (one value per line) to:
    - `measurement_series_balanced_pu.txt`

Unbalanced case (three‑phase unbalance at Bus 3, OpenDSS):

- Script: `export_measurement_series.py`
  - Runs `Run_IEEE14Bus.dss` (current unbalanced Bus 3 model).
  - Computes `Vm` from phase‑1 VLN, `Pinj/Qinj` from element powers (loads, generators, slack) using the same `Pg − Pl` convention.
  - Computes `Pf/Qf/Pt/Qt` from OpenDSS branch powers in the same branch order.
  - Writes a 122‑entry vector to:
    - `measurement_series_bus3_pu.txt`

Both vectors are directly paste‑able into MATLAB as row vectors using the documented index map.

## Notes for Future Work

- To move the unbalance to a different bus:
  - Edit `IEEE14Loads.DSS` (swap the 3‑phase load there for three 1‑phase loads on `.1/.2/.3`).
  - Optionally update `report_unbalance.py` to print per‑phase VLN at the new bus.
  - Rerun `export_measurement_series.py` to regenerate the unbalanced measurement series.

- To change the balanced reference model:
  - Adjust `export_measurement_series_balanced.py` (e.g., different pandapower case or modified parameters).
  - Keep the `makeSbus`‑style P/Q reconstruction and branch order intact if you want compatibility with the existing MATLAB tooling.

