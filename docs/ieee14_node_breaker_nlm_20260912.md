# Breaker-level topology identification: substation measurements and the node/breaker NLM

Date: 2026-09-12
Branch: `feature/ieee14-full-topology` (uncommitted)
Builds on: `docs/ieee14_full_detectability_20260910.md` (model audit, stage-1 integration)

## What the agent does now for a topology error

1. `run_wls` on the operator's 14-bus model flags the anomaly and mints the usual
   branch-multiplier signatures.
2. `get_topology_context` **requests the substation measurements** bound to the
   state (node voltages, unit and load meters, terminal flows, breaker flows) and runs
   the **node/breaker generalized state estimator with normalized Lagrange
   multipliers** on the reported breaker statuses. It ranks every breaker, re-estimates
   with each top-ranked breaker flipped, maps each flip to its consequence in the
   operator model, and screens the admissible flips on the operator model with the
   same non-mutating candidate-quality lookahead used for line hypotheses.
3. `correct_topology(state_id, cb_name, status, line_index)` **sets the reported
   status of the named breaker**. The executor resolves the bus-branch consequence
   through the topology processor, refuses anything the 14-bus model cannot carry,
   derives the operator case, and records the breaker in the state metadata.
4. Verification runs the operator-model WLS on the derived case; the target test now
   also requires that the derived case was produced by exactly the requested breaker.

The expert follows this route unchanged: it proposes the context call on topology
signatures and then the advertised breaker-level correction. The private truth is
retired only by a correction naming the true breaker; another breaker whose flip
yields the same operator case is not a fix.

## The estimator

`Transmission/ieee14_full_gse.py::gse_topology_nlm`. State
`[theta(65), V(65), P_cb(73), Q_cb(73)]`, physical branches with their admittances,
breaker flows as explicit states. Analog rows: node voltages, injection meters,
terminal flows, breaker flows. Equality constraints: zero injection at switching
nodes, `theta_a = theta_b` and `V_a = V_b` for each reported-closed breaker,
`P_k = Q_k = 0` for each reported-open breaker. The constrained problem is solved
with the Hachtel (KKT) system; the multiplier covariance is minus the constraint
block of the KKT inverse and the normalized multiplier is `lambda / sqrt(cov)`
(Clements and Simões Costa, 1998). A tiny regularization of the constraint block keeps
the system solvable inside closed-breaker loops; the two loop-closing breakers of the
normal state (`CB_5_T56_B2`, `CB_11_L1110_B2`) are detected structurally and reported
with zero multipliers, since flipping them changes no partition.

**Ranking.** Hard equality constraints in series carry the same first-order multiplier
(the tension of a chain), so along a breaker-and-a-half string every angle multiplier
ties exactly. `rank_breakers` orders by the larger constraint multiplier and breaks
near-ties with the other constraint of the pair. **Confirmation.**
`screen_breaker_flips` re-estimates with each candidate flipped; a flip that leaves the
substation chi-square below its threshold explains the telemetry. When no flip is
clean because another fault remains (a gross meter error on a mixed root), flips that
remove at least half of the substation chi-square are still offered, best first, and
the operator-model screening decides between a partial and a final repair.

**Validation** (`tests/test_ieee14_full_gse.py`, plus the probe recorded here):

| Check | Result |
|---|---|
| Normal state, 150 noise draws | normalized multipliers have unit variance (median std 0.96); chi-square below its limit |
| 26 dangling-terminal errors, max-score ranking | true breaker first in 15, within top 7 in all |
| Same, tie-aware ranking | true breaker first in 26 of 26 |
| Top-8 flip confirmation | true breaker has the only clean post-flip chi-square in 26 of 26 (240 to 333 vs limit 351) |
| True statuses | chi-square clean for every case |
| Cost | 0.2 s per estimate; about 2 s per context call including 8 confirmations and screening |

The soft-constraint formulation tried first (tightly weighted pseudo-measurements)
produced normalized values that were numerical artefacts in the chain-tied yards; the
exact Lagrangian form replaced it.

## The physical truth and the telemetry channel

`Transmission/ieee14_full_substation.py`. An AC OPF on the ideal contracted true
topology fixes the dispatch (as for every synthesized family); a power flow of the
65-node network with closed breakers as tiny series impedances (the historical pocket
values, 5e-6 + j5e-5 pu) at that dispatch supplies one physical solution. Breaker
flows are therefore determinate even in yards with parallel closed paths. The
substation telemetry and the operator's 122-entry vector are read from that same
solution through the fixed meter identity, so every shared meter reads the same value
(bus voltage at the main-section meter node, bus injection as the sum of the bus's
unit and load meters, terminal flows). Independent Gaussian noise at the nominal
sensor accuracies (1e-3 pu voltages, 1e-2 pu powers) is added to the telemetry and
the operator vector inherits it. Newton diverges from a flat start with near-zero
breaker impedances, so nodes start at their topological bus's OPF voltage.

The scenario carries `metadata.substation_telemetry`, `metadata.reported_breaker_status`
(the schematic-normal statuses), `metadata.operator_voltage_meter_nodes` and the model
id and fingerprint; `substation_telemetry` is advertised as an evidence channel. At
context time the provider overwrites the shared meters in the telemetry with the
state's current measurements, so an accepted meter correction or an overlaid gross
error reaches the estimator. Admission adds three gates to the operator-model gates:
the estimator on the reported statuses must exceed the anomaly margin, the true
statuses must estimate clean, and the true breaker must rank first
(`enforce_topology_ranking`, default on; `topology_ranking` records the rank and
scores without naming the breaker).

## Contract changes

- `correct_topology` accepts `cb_name` with `status` (breaker status, 0 open / 1
  closed) and optionally the `line_index` it affects; a breaker name may accompany
  exactly one numeric row in `psse_env/actions.py` and in the environment's action
  signature, which keeps two breakers on the same line distinguishable. Failures:
  `topology_correction_unsupported_effect` (islanded bay, equivalent; splits and merges are rendered, see below),
  `topology_correction_inconsistent_target`, `topology_correction_unknown_breaker`,
  `topology_correction_breaker_unsupported` (no telemetry bound),
  `topology_correction_no_change`.
- `get_topology_context` on a bound state returns `substation_measurements_requested`,
  `substation_measurement_inventory`, `node_breaker_estimate`, `breaker_findings`
  (rank, score, multipliers, post-flip chi-square and progress, bus-branch effect,
  affected line) beside the legacy keys; `evidence_source` is
  `deployment_context:node_breaker_nlm_candidate_screened`. States without the channel
  take the unchanged line-hypothesis path.
- Verification adds `topology_target_breaker` and
  `topology_target_breaker_matches_requested`. Private-truth retirement and the offline
  teacher audit require breaker identity to agree when both sides name one.
- The mixed-topology admission gate accepts the breaker-level action schema.

## Files

New: `Transmission/ieee14_full_substation.py`, `Transmission/ieee14_full_gse.py`,
`tests/test_ieee14_full_gse.py`, `psse_env/providers/test_node_breaker_topology.py`.
Modified: `psse_env/providers/matpower.py`, `psse_env/providers/scenario_generator.py`,
`psse_env/actions.py`, `psse_env/transactional_env.py`,
`psse_env/private_target_matching.py`, `psse_env/dagger/offline_teacher_target_audit.py`,
`psse_env/providers/test_scenario_generator.py`.

## Bus splits (added the same day)

A breaker whose true state splits a bus into two energized sections is now sampled
and correctable. The operator model is re-rendered rather than approximated:
`Transmission/ieee14_full_substation.py::operator_model_from_map` contracts a reported
breaker map the way an EMS topology processor does. A section with one line terminal
and no equipment is a dangling terminal, so that line goes out of service and keeps
its row on the main section of its planning bus; an empty section is dropped; an
unsupplied section drops its loads and switches its units off; every other section
is a bus. Planning buses keep numbers 1 to 14 through their main section (the one
holding the operator's voltage meter), and further sections follow, so the normal
map reproduces `case14` exactly and a split appends bus 15.
`operator_vector_for_layout` reads the operator vector of any rendering from the
substation meters: section voltage at the meter node, injection as the sum of the
section's unit and load meters, terminal flows unchanged.

- **Generator.** `topology_effects` defaults to dangling terminals and bus splits;
  `merge` is a supported third class (rendered as 13 buses) but not a default, see
  below. A topology root is admitted only when the operator's WLS evidence is
  branch-dominant (largest normalized multiplier above 1.2 times the largest
  normalized residual), the discrimination the deployed solve routes on; otherwise
  the expert would take the measurement route first and commit a false meter fix.
  `require_branch_dominant_topology` switches the gate.
  A split root's `clean_case` is the rendered 15-bus case of the true map and its
  `clean_measurements` the projection into that layout; the root itself stays on
  `case14` with 122 measurements. The truth names the breaker, its expected status,
  `physical_effect: bus_split` and the affected planning bus; there is no branch row.
  Mixed measurement-plus-topology roots keep drawing dangling terminals so the meter
  indices of the overlaid fault stay valid.
- **Executor.** `correct_topology(cb_name, status)` for a split or merge renders the
  new map on the state's current case, re-projects the measurements, and records
  the layout, the derived case and `operator_layout_changed` in the metadata. A
  `line_index` on such a request is refused as inconsistent. Islanded bays remain
  refused as equipment outages.
- **Context.** Structural flips are confirmed by re-estimation like any other and
  offered without a line index.
- **Verification and audits.** The target test for a breaker with no branch row is
  the recorded breaker status (`breaker_status_mismatch`); the branch-multiplier
  ambiguity test does not apply. The candidate-quality oracle accepts the executor's
  re-rendering when the candidate case is the recorded derived case. Private-truth
  retirement resolves such a breaker by reported status. The release audit compares
  the final case and measurements with the clean rendering as a whole and judges
  non-regression on the breaker status.
- **Synchronization.** Shared-meter synchronization follows the recorded operator
  layout, so it works on 15-bus states.

Validation: the normal map renders `case14` bit-for-bit; all nine feasible splits
render 15 buses and the projected vector verifies clean (J 74 to 124 against a limit
of 131) while the reported 14-bus model shows J from 527 to 52,731; the expert fixes a
split root end to end with the private truth retired and the release audit clean.
The tenth split, `CB_13R4_13R1`, leaves the OPF infeasible at most load scales and is
rejected per draw. The three 10/14 merges are the only merges the model can produce
(no other yard serves two planning buses, and no pair of wrongly closed breakers joins
any other pair of buses); the estimator ranks each first with a clean flip, and the
rendered 13-bus case verifies clean. On the operator's 14-bus model, however, a merge is
decisively residual-dominant: over 26 draws the largest normalized branch multiplier
was 2.5 to 3.9 against a largest residual of 5 to 9 (ratio 0.25 to 0.64), while every
admitted dangling or split root shows ratios of 1.4 to 2.5. The ordinary WLS therefore
detects a merge but attributes it to a meter, and with the dominance gate off the
expert commits a false meter correction before the breaker route repairs the merge.
That is why merges stay out of the default mix even though the breaker-level tools
handle them.

## Limits

Islanded bays are ranked and confirmed by the estimator like any other breaker, but
their corrections are refused with the effect named: they are equipment outages the
operator's WLS cannot see. Eight candidates are confirmed per context call; the audit never needed
more than seven. After a split, later meter-index bookkeeping (mixed roots,
research truth metrics keyed by line index) assumes the 14-bus layout, which is
why compositions draw dangling terminals only.
