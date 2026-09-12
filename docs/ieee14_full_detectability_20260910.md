# Full IEEE-14 node/breaker model: detectability audit and inclusion decision

Date: 2026-09-10
Branch: `feature/ieee14-full-topology` (created from `local/relaxed-current` at `a5db72c`)
Model: `ieee14_full_schematic_v1`, fingerprint `a15f32c5e702…` (see `models/ieee14_full_schematic_v1.json`)
Audit script: `scripts/audit_ieee14_full_detectability.py`
Audit outputs: `models/ieee14_full_detectability/single_flip_detectability.{json,csv}`

## What was checked

1. The delivered additive patch (`ieee14_full_detail.patch`) applies cleanly and touches no
   existing file. Its 107 tests pass from the repository root, including the pandapower
   adapter test that the author could not run (pandapower 3.1.2 is installed here).
2. The transcription was compared against the supplied schematic. Yards 1–3 reproduce the
   existing pocket model exactly (same 15 CB names and normal states). Yards 4, 5, 6, 9, 11,
   12, 13 and the shared 10/14 yard match the visible closed/open squares. What the image
   cannot settle: the fifth bay and coupler at bus 5, which bus-3 arrow is the generator,
   the second arrow at bus 1, and which line lands on which string node in yards 4, 9 and
   10/14. None of these change the normal-state equivalence; they do change which
   section a given split isolates. Confirm them in `docs/ieee14_full_model_viewer.html`.
3. The admission rule was tested with the pipeline's own WLS, not with graph counting.
   For each of the 73 switches the flipped state is the physical truth and the schematic
   normal state is the reported model (`case14`). Measurements are synthesized from a
   PYPOWER power flow of the true ideal-switch topology with fixed physical identity
   (voltage at one meter node per planning bus, unit/load meters summed per planning bus,
   terminal flows of the 20 original branches), laid out in the 122-entry operator order,
   and passed to `mcp_server.matpower_server._wls_json` on `case14`. Detection uses the
   round-0 gate: chi-square objective above 1.25 × χ²(dof 95, α 0.01) = 162.5, at load
   scales 0.85, 1.0 and 1.15, noiseless plus ten noisy replicates each. The normal state
   reproduces h(x) to 1e-12 and gives J ≈ 1e-10. The table below is that power-flow run.
   After the integration the audit was rerun on the AC-OPF operating point, which is now
   the script's default (`--solver runopf`): the dangling-terminal picture is identical
   (24 detectable, all 26 clean after the branch-status fix, corrected J at most 66),
   the merges reach J 70 to 143 across the load scales, and two bus splits
   (`CB_13R4_13R1`, `CB_3_L34_B2`) leave the OPF infeasible at load scale 1.0 or 1.15,
   which will matter only once bus splits are sampled. The committed JSON and CSV are
   from the OPF run.

## Result: 34 of 73 single-switch errors are admissible

| Category | Count | WLS on `case14` detects | Fix representable today | Decision |
|---|---:|---|---|---|
| Partition unchanged | 25 | No. Measurements are bit-identical to normal. | n/a | Exclude (by construction). |
| Dangling line terminal (one line end isolated, nothing else) | 26 | 24 yes at all load scales. 2 no: `CB_12_B1_L1213` (line 12-13, J = 19.7) and `CB_Y1014_10N1_10N2` (line 10-11, J = 142). | Yes, all 26: setting that branch out of service in `case14` leaves J ≤ 66 < 130. | Include the 24. The existing branch-status `correct_topology` already fixes them. |
| Bus split (a section keeps two or more terminals, or a terminal plus equipment) | 10 | Yes, J from 389 to 1.3e5, 100 % noisy detection. | No. Needs a bus-split correction and a 15-bus candidate case. | Include, but only after the node-breaker correction primitive exists. |
| Merge of buses 10 and 14 | 3 | Marginal. Measurements move up to 11 σ, yet J = 70 to 118 stays under the 130 limit noiseless; noisy detection 60–100 % depending on load. | No. Needs a bus-merge correction and a 13-bus candidate case. | Exclude, or keep as a separate weakly-observable stratum. |
| Unsupplied island (a load or unit bay is cut off) | 9 | No for 8 of 9 with a voltage meter on the energized section: J ≈ 0, the WLS sees an ordinary operating point. `CB_9_I_B2` reaches J = 224 only because the 19 MVAr bus-9 shunt dies with the load bay. | No. It is an equipment outage, not a branch or bus change. | Exclude from the topology family. |

Admissible under the stated rule: 10 bus splits + 24 dangling terminals = 34. Yard coverage:
1 (5), 2 (4), 3 (2), 4 (6), 5 (2), 6 (5), 9 (5), 10/14 (1), 12 (1), 13 (3), 11 (0, all four
flips there are equivalent).

### Two findings that change how the model should be wired in

**Voltage-meter placement decides what "detectable" means for islands.** The manifest's
`voltage_anchors` sit on the injection bays. Used as Vm meters, an islanded bay reads 0 pu;
that makes three island cases "detectable" through a dead-bus reading and makes the WLS
solver fail outright in five others. With the meter on the energized main section the same
cases are clean. The audit therefore reports three placements (`anchor`, `busbar`, `main`)
and bases admission on `main`. Treat the anchors as equipment nodes, not meter locations.

**Merges are nearly invisible to the 14-bus WLS.** Buses 10 and 14 both hang off bus 9 with
similar voltages, so tying them changes flows by several sigma without producing a
chi-square excess that the round-0 gate accepts at every load scale.

## Integration performed (2026-09-10, same branch)

Superseded on 2026-09-12 for the correction step: the agent now requests substation
measurements, runs the node/breaker normalized-multiplier estimator and fixes the
named breaker; see `docs/ieee14_node_breaker_nlm_20260912.md`. The sampling class,
operating point, noise and load-scale decisions below are unchanged.

Stage 1 is wired into the round-0 scenario generator, together with the three
cross-family fixes identified in the legacy-versus-DAgger comparison.

**Topology roots are node-breaker switch errors.**
`Round0ScenarioGenerator._topology_scenario` now draws one of the 26
`dangling_line_terminal` switches from `Transmission/ieee14_full_measurements.py`,
contracts the true topology with `topology_to_matpower`, and synthesizes telemetry
through the fixed-identity mapping (`operator_measurements` with the main-section
voltage meter). The reported model stays `case14`; the bus-branch fix is still
`correct_topology(line_index, status=0)` on the branch at the isolated terminal, so
the environment, expert, verifier and audits are unchanged. The truth row carries
`cb_name`, `cb_yard`, `reported_cb_closed`, `true_cb_closed`, `physical_effect`,
`topology_model_id` and `topology_model_fingerprint` next to the branch keys every
downstream matcher already reads; `canonical_branch_target` resolves row keys before
named keys, so the switch name never changes target matching. Nothing about the
switch reaches policy-visible metadata (tested).

**Operating point.** Both synthesized families now solve an AC OPF (`runopf`) on a
load scale drawn from the corpus range 0.80 to 1.25, the same solver and range the
tabular no-error, measurement and parameter rows were built with. The old
power-flow point with generator setpoints clamped to 1.06 pu is gone.

**Noise.** `topology_noise_scale` defaults to 1.0, so topology roots carry the full
empirical noise profile like every other family.

**Harmonic load scale.** Harmonic roots are synthesized on the fly
(`_synthesized_harmonic_row`) by running the legacy harmonic synthesis, unchanged in
source buses, THD range, spectrum, transducer and noise, on the OPF operating point at
a corpus-range load scale. `build_trace` gained optional `bus`/`branch` arguments for
this; its legacy call is untouched. The tracked harmonic corpus rows, all at unit
load on planning voltages, are no longer read, and `harmonic` left the
source-partition family list. `hidden_truth` records `load_scale` and
`operating_point` for both families.

**What the branch-status fix approximates.** A dangling terminal keeps the line's
charging connected at the far end, while the corrected `case14` removes the line
entirely. Noiseless, the corrected model fits within J of 3 to 66 for every switch;
with noise the heavily charged lines 1-2, 1-5 and 2-3 sometimes exceed the 130 limit
and the existing `_require_clean` gate rejects those draws
(`corrected_configuration_still_anomalous`). Measured on seed 7: 24 roots from 29
attempts in 11 s, 15 distinct switches over 12 lines. Draws on lines 12-13 and 10-11
at light load fail `_require_anomalous`, as before.

**Not offered yet.** The 10 bus splits, the 3 merges of 10/14 and the 9 islanded
bays are catalogued but not sampled: they need a breaker-level correction with a
variable bus count and measurement re-projection (stage 2 below).

Tests: `tests/test_ieee14_full_measurements.py` (catalogue counts, h(x) agreement,
equivalent flips bit-identical, dangling flips fully energized, meter placement,
dead bay reads zero) and `SynthesizedFamilyOperatingPointTests` in
`psse_env/providers/test_scenario_generator.py`.

## Where the pipeline stands

The research DAgger path does not use the node-breaker pocket at all. Its topology roots
are branch outages synthesized on `case14` in `psse_env/providers/scenario_generator.py`
(`_topology_scenario`), and its correction, screening and truth matching are branch-status
operations (`psse_env/providers/matpower.py`: `correct_topology`, `get_topology_context`,
`_flip_creates_island`, `_target_evidence`; `psse_env/state_store.py::_apply_topology_update`;
`psse_env/private_target_matching.py`). The corpus `topology_error` rows from the pocket
generator are explicitly skipped by that path because a `case14` branch-status correction
cannot represent them.

The legacy SFT generator (`Transmission/generate_measurements.py`) still draws only from
substations 1–3, averages section voltages by planning bus in `_nb_to_operator_z`, and the
MCP `correct_topology_from_path` regenerates measurements from the candidate's own power
flow. The reviewer's objections to those three behaviours are confirmed by reading the code.

## Integration path

Stage 1, no new correction primitive, 24 roots per operating point (done, see above):

- Add a node-breaker source to `_topology_scenario` that samples one of the 24 admissible
  dangling-terminal switches, contracts with `topology_to_matpower`, solves with PYPOWER,
  and builds `z` with the fixed-identity mapping from the audit script (main-section Vm,
  unit/load meters per planning bus, terminal flows).
- Truth: `{cb_name, model_id, model_fingerprint, expected_cb_status, branch_row0,
  expected_status: 0}`. The agent's existing `correct_topology(line_index, status=0)` is
  the correct fix; the audit shows the corrected `case14` is clean for every one of them.
- Keep `_require_anomalous` / `_require_clean` as they are; they already reject the two
  light-load lines the audit found blind.

Stage 2, bus splits, 10 more roots per operating point:

- New action arguments `cb_name` + `status` on `correct_topology`, resolved through the
  versioned model to a derived case with 15 buses, plus re-projection of the 122-entry
  observation into the candidate's bus order. The equipment-free section gets a
  zero-injection pseudo-measurement; its voltage slot needs a missing-measurement path in
  `tools/lagrangian_port.py` (or a very large sigma), because no meter exists there.
- `_target_evidence` proves topology targets by branch-status equality today; it needs a
  CB-status equality test. `_flip_creates_island` needs the node-breaker equivalent.
- `TopologyExpert`, `state_store`, `private_target_matching` and the counterfactual
  injectors match targets on branch rows; add the `cb_name` route.
- Hypothesis enumeration in `get_topology_context`: screen the 34 admissible single flips
  with the same non-mutating WLS lookahead (34 solves per call on IEEE-14 is cheap).

Not recommended: enrolling merges or unsupplied islands as topology-error roots. Merges
fail the detectability gate at light load, and islands are equipment outages that the
bus-branch WLS treats as legitimate operating points.

## Reproduce

```bash
python -m pytest -q tests/test_ieee14_full_topology.py
python scripts/audit_ieee14_full_detectability.py --seeds 10
```

Runtime is about two minutes for three load scales and ten noisy replicates per flip.
