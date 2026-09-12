# Full IEEE-14 node/breaker model — implementation and integration notes

Model ID: `ieee14_full_schematic_v1`
Reviewed repository: `yifeihsu/psse_agent`
Reviewed branch: `local/relaxed-current`
Reviewed commit: `a5db72c0ac6a84939b4760d8880f94262c0a4fc4`
Prepared: 2026-09-10

## Delivered scope

This is an **additive full-system topology model**, not a replacement of the
historical pocket123 experiment or a completed production WLS/DAgger migration.
The GitHub branch has **not** been changed or pushed. The accompanying patch
adds the new model, tests, model manifest, numerical fixture, validation CLI,
and documentation. No existing source file is modified.

The model implements the **73 visible switches** and **65 connectivity nodes**
in the user-supplied schematic. The normal state is **53 closed / 20 open**;
ideal-switch contraction gives **14 topological buses and 20 original electrical
branches**. All 14 planning-bus locations are represented. This is not 14
independent, identically equipped substations.

### Yard inventory

| Planning bus / yard | Schematic construction | Nodes | CBs | Closed | Open |
|---|---|---:|---:|---:|---:|
| 1 | Two three-breaker strings between busbars | 6 | 6 | 5 | 1 |
| 2 | Five-node ring | 5 | 5 | 4 | 1 |
| 3 | Two busbars, two connections per line terminal | 4 | 4 | 3 | 1 |
| 4 | Three three-breaker strings | 8 | 9 | 7 | 2 |
| 5 | Five dual-connected bays plus a coupler | 7 | 11 | 7 | 4 |
| 6 | Five dual-connected bays plus a coupler | 7 | 11 | 6 | 5 |
| 7 | Transformer star-equivalent junction | 1 | 0 | 0 | 0 |
| 8 | Generator terminal, no drawn switch | 1 | 0 | 0 | 0 |
| 9 | Two three-breaker strings plus injection bay | 7 | 8 | 6 | 2 |
| 10 / 14 | Shared yard with two normally separate topological buses | 8 | 9 | 6 | 3 |
| 11 | Two dual-connected line bays | 4 | 4 | 4 | 0 |
| 12 | Single busbar and two line-terminal CBs | 3 | 2 | 2 | 0 |
| 13 | Four-node ring | 4 | 4 | 3 | 1 |
| **Total** | | **65** | **73** | **53** | **20** |

“Nodes” includes busbars, ring nodes and equipment terminals, not only bus sections.
The counts are this transcription of the supplied image, not a claim about a
canonical official IEEE breaker-level dataset.

## Critical interpretation details

### Shared 10/14 yard

The top busbar in the upper-right yard is bus 14; the bottom is bus 10.
Each of the three vertical strings contains an open switch in the normal
state. Thus the yard has two electrical components, not one. Closing any of
`CB_Y1014_14B_10N1`, `CB_Y1014_14N2_10B` or `CB_Y1014_I14_I10` merges them.
No fictional 10–14 transmission branch is created. These CBs have yard ID
`10_14`, rather than being ambiguously assigned to one independent station.

### Transformer representation

The three-terminal transformer symbol is represented by the existing case14
star-equivalent involving 4–7, 7–9 and 7–8. The 4–9 and 5–6 branches are also
retained exactly. Do not add a three-winding transformer on top of those
existing equivalent branches; that would double-count the device.
No new switches at buses 7 or 8 are inferred from blank areas of the diagram.

### Declared equipment-placement assumptions

The figure does not identify every injection arrow as generator versus load,
or show a separate shunt bay. The default attachments are explicit:

- Bus 1: the slack generator attaches to `1N1`. The second arrow does not cause
  an invented load or a duplicate generator.
- Bus 3: generator at `3B1`, load at `3B2`. Swapping those assignments is a
  permissible alternative transcription and changes split-topology results,
  although it does not change normal-state equivalence.
- Bus 9: its reference-case shunt attaches at `9|I`, with the local load.
- Other reference equipment attaches to the corresponding drawn injection
  terminal; units are never split into multiple voltage controllers.

These are modelling assumptions, not externally verified physical terminal
identities. They are recorded in the manifest. To change an attachment, modify
`model.equipment[table][planning_bus]`, then pass that model explicitly; its
fingerprint changes. Use a new experiment/model ID for a changed interpretation.

## Files and API

`Transmission/ieee14_full_topology.py` provides:

- `build_full_topology()`: the complete connectivity/CB/terminal model.
- `topology_to_matpower(reference_case, status_map, model=...)`: exact ideal-switch
  contraction and a MATPOWER/PYPOWER case dictionary plus explicit node mapping.
- `build_nb_ieee14_full(status_map, reference_net=..., model=...)`: optional
  pandapower adapter using **ideal bus-bus switches**.
- `single_flip_audit()`: all 73 flips, including topology-equivalent changes.
- `write_matpower_case(...)`: a text `.m` export for the existing text parser.

Example with an existing PYPOWER case:

```python
from pypower.api import case14
from Transmission.ieee14_full_topology import (
    build_full_topology, topology_to_matpower, write_matpower_case,
)

model = build_full_topology()
# Omitted switch keys keep their schematic-normal state.
# With cumulative corrections, pass the COMPLETE current reported-state map.
case, mapping = topology_to_matpower(
    case14(), {"CB_6_B1_B2": "open"}, model=model
)
write_matpower_case(case, "case14_full_split6.m", "case14_full_split6")
print(mapping["node_to_bus"])
```

For pandapower:

```python
import pandapower as pp
from Transmission.ieee14_full_topology import build_nb_ieee14_full

net, sec_bus, cb_idx, line_idx, trafo_idx = build_nb_ieee14_full()
pp.runpp(net, init="flat", calculate_voltage_angles=True)
# IMPORTANT: cb_idx indexes net.switch, not net.impedance.
print(net.switch.loc[list(cb_idx.values()), ["name", "closed"]])
```

The pandapower adapter is implemented but was **not executed in this sandbox**,
where pandapower is not installed. Its optional pytest integration check is
included and was skipped here. Do not pass its switch IDs to the historical
`run_pf_and_measure()` impedance-based CB-flow extractor.

### What the electrical processor preserves

Every original electrical branch row is retained with its R, X, charging,
taps, phase shift, ratings and service status unchanged. Generator row count,
P/Q data, voltage setpoints, status, limits and costs remain unchanged; only
terminal indices are remapped. Bus P/Q loads and G/B shunts are assigned to
explicit physical terminals and aggregated only after ideal-switch contraction.

Open breaker ends do not automatically make their transmission branch out of
service. A disconnected line end remains a terminal so its charging can still
be represented. No degree-one physical branch is pruned. Empty, isolated
busbar-only components are retained as type-4 buses; unsupplied load/generator
islands are reported rather than silently shed.

## Validation actually executed

Run from the package/repository root:

```bash
python -m pytest -q tests/test_ieee14_full_topology.py
python scripts/validate_ieee14_full_topology.py --validate-ac --export-single-flips
```

Observed in this sandbox:

| Check | Result |
|---|---|
| Unit/regression tests | 106 passed; 1 optional pandapower integration test skipped |
| Normal bus, gen, branch, gencost matrices | Exactly equal to the reference fixture |
| Maximum normal-state Ybus difference | 0 |
| Normal polar AC PF residual | 1.76248e-14 pu |
| Normal voltage / branch-power difference | 0 / 0, because contracted numerical cases are identical |
| Single-CB inventory processed | 73 / 73 |
| All flips preserve branch parameters, generator non-bus columns and load/shunt totals | Pass |
| Single-flip topology changes | 45 splits, 3 merges, 25 equivalent |
| Single-flip AC checker | 64 solved; 9 flagged as islands without a reference |

The AC checker is an independent NumPy/SciPy polar residual implementation.
It is **not** MATLAB/MATPOWER `runpf`, a production WLS test, OPF, an N-1/security
assessment, or a check of generator Q / voltage / equipment limits. It makes
no claim that all solved switching states are operationally admissible.
The nine islanded cases are explicit audit outcomes, not hidden omissions.

The zero normal-state mismatch verifies the case-preserving construction; it
is not independent physical evidence for the unresolved arrow identities.

`validation/validation_report.json`, `validation/pytest_results.txt`, and
`validation/single_flip_audit.csv` contain the recorded results. The 73 `.m`
files are static switch-state cases, **not 73 labelled training episodes**.

## Integration into the current topology-error experiments

The current code explicitly limits `_choose_random_cb_open()` in
`Transmission/generate_measurements.py` to substations 1–3. It also averages
section voltage magnitudes by planning-bus number in `_nb_to_operator_z()`.
That average is not a physical voltage measurement after a bus split.
`Transmission/nb_to_matpower.py` rebuilds selected electrical table fields;
the new direct processor avoids that copy/rebuild path and dangling-line
pruning for the full model.

Retain `pocket123` for reproducibility. Introduce an explicit model/profile
selector in generation, runtime context, correction and verification before
making this model the training default. Such a CLI/runtime selector is **not**
added to existing commands by this additive package.

Required migration work:

1. **One versioned inventory everywhere.** Generation, topology context,
   correction and the validity oracle must use the same model ID, fingerprint,
   CB identifiers, native terminal mapping and cumulative reported-status map.
   The shared yard needs `yard_id="10_14"` and affected planning buses `[10,14]`;
   do not rely solely on an integer station-name prefix. The 15 historical CB
   IDs at buses 1–3 are preserved, but old and new episode models must not be
   mixed because equipment placement and switch numerics differ.
2. **Separate physical truth from reported status.** Generate physical signals
   under true CB states; expose an independently corrupted reported topology.
   A correction edits the reported model, not the physical plant or the
   observed measurement values. Do not regenerate cleaner measurements from a
   candidate topology and use them as evidence that the candidate was correct.
3. **Keep measurement identity fixed.** Store each voltage/flow channel at a
   physical node or branch terminal. Map that fixed registry through each
   candidate's node-to-topological-bus map. Do not average voltages across split
   sections or rebuild the observation vector from the candidate's solved PF.
   Branch orientation and stable original row IDs are already explicit here.
   Injection-channel aggregation needs an explicit measurement model; summing
   or duplicating a topological-bus injection across original buses is not a
   general solution after a split or merge.
4. **Classify topology-equivalent status discrepancies separately.** Of the
   73 single flips, 25 leave even the equipment-terminal connectivity partition
   unchanged. They cannot be detected from ordinary external analog measurements
   under the ideal-switch model. Exact CB-state scoring would need independent
   digital/status/CB-flow evidence; otherwise score the admissible topology
   equivalence class. Not every one of the other 48 flips is automatically
   identifiable under a limited/noisy measurement set.
5. **Qualify before retraining.** Run the optional pandapower regression, target
   MATLAB/PYPOWER exports, fixed-measurement WLS tests, and the actual correction /
   expert/transaction loop. Stratify useful roots by yard, split/merge, islanding,
   measurement availability and identifiability. Preserve equivalent and islanded
   cases as explicit separate categories. Add simultaneous CB errors only after
   these single-flip boundaries pass. Hold out entire yards or structural events,
   not merely noise replicas of the same switch signature.

Ideal switch-loop currents are not necessarily uniquely determined by external
terminal injections. This model contains redundant closed paths, particularly
at yards 5 and 11. Do not label arbitrarily distributed loop flows as unique
physical CB telemetry.

## Sources inspected

- User-supplied 546 × 639 IEEE-14 bus-section/switch schematic; file hash recorded
  in `models/ieee14_full_source_provenance.json`.
- `yifeihsu/psse_agent`, reviewed commit above:
  `Transmission/nodebreaker_pp14.py`,
  `Transmission/generate_measurements.py`,
  `Transmission/nb_to_matpower.py`.
- `MATPOWER/matpower`, `data/case14.m`, blob
  `a67682c6654077a0dd71f0eab936de2ec830903d`: reference electrical numeric data.
- Official pandapower documentation, “Switch” and “Known Problems and Caveats”,
  accessed 2026-09-10: ideal bus-bus fusion versus small-impedance modelling.

## Applying the additive patch

The patch has not been applied to the remote repository. From your checkout,
with the downloaded patch in the current directory:

```bash
git switch local/relaxed-current
git switch -c feature/ieee14-full-topology
git apply --check ieee14_full_detail.patch
git apply ieee14_full_detail.patch
python -m pytest -q tests/test_ieee14_full_topology.py
python scripts/validate_ieee14_full_topology.py --validate-ac --export-single-flips
```

`git apply --check` deliberately refuses conflicting existing files. The patch
contains no replacement for the existing repository README or pocket123 code.
The ZIP also contains pre-generated validation outputs and a self-contained
interactive HTML viewer. Open `docs/ieee14_full_model_viewer.html` in a browser.
The viewer changes graph connectivity only; it does not run PF or WLS.
