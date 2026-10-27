# Compound errors, stage 1: the rule expert's ceiling on randomly combined errors (2026-10-27)

Stage 1 of the compound-error extension (the assessment of 2026-10-27:
multiple parameter errors with meter errors, parameter plus breaker errors,
two breakers).  A compound synthesizer in the round-0 generator, a 1,275-root
suite of fourteen classes, the `baseline_compare` expert rolled out under
`scada_only` with the node/breaker telemetry, the decision audit on every
failed root and a sample of solved ones, four identifiability probes of the
pairs the assessment flagged, and the same suite under a flagged
topology-first order of the teacher.  CPU only, nothing trained.  Research
only; journal scope; the PES GM manuscript is untouched.

In one paragraph: the teacher solves 495 of 1,275 compound roots (39%; 44%
of the roots whose structural atoms each alarm alone).  Three mechanisms
account for most of the rest.  On every breaker-bearing root the parameter
route runs first and explains the open terminal by an inflated impedance
that the verifier accepts; putting the topology route first behind a flag
takes the ceiling to 609 (48%) and the single-breaker control to 50 of 50,
at the cost of the mirror-image masking on 38 parameter roots.  A bus-split
correction re-renders the operator model and the node/breaker binding then
fails on every later topology action, so two splits are unsolvable (0 of
45).  The impedance of a line whose terminal is open is not identifiable
and the audit still asks for it (0 of 75).  The adjacent-line pair is an
estimator problem (one more alternating pass brings 112 of 131 estimates
within tolerance), the same-line meter pair a verifier problem (the correct
impedance is rejected on 61 of 64 roots because the line's own bad meter
holds its multiplier up), and a progress floor divided by the number of open
faults would admit four wrong corrections for every true one it rescues.

Branch `claude/nostalgic-kalam-be5f7a` (from `claude/clever-roentgen-03b19c`
d6e6acc).  Code: `psse_env/providers/scenario_generator.py` (the `compound`
family, `CompoundSpec`, `DEFAULT_COMPOUND_CLASSES`, `build_compound`),
`psse_env/providers/test_compound_scenarios.py`,
`research/compound_errors/` (`build_suite`, `rollout`, `audit`,
`identifiability`, `report`), `research/rl/decision_audit.py` (breaker-level
truth probe).  Data: `output/compound_errors_20261027/` (gitignored):
`suite.json`, `manifest.json`, `atoms.jsonl`, `expert_baseline_compare/`,
`audit/`, `identifiability/`, `report/`.

## 1. The synthesizer

A compound root draws a set of error atoms from a class specification
(`CompoundSpec`): a meter group of one to three gross errors, one or two line
R/X errors, one or two breaker-status errors, with placement constraints.

**Structural atoms change the physical truth and are re-solved.**  The
scaled planning case (load scale 0.80 to 1.25, as every synthesized family)
receives the true impedances (both factors drawn independently from the
corpus's ranges [0.1, 0.5] or [2, 5]; the corpus also draws R-only and X-only
errors, this sampler always changes both), the true breaker map is
contracted (`flipped_case`), an AC OPF on the true contracted topology fixes
the dispatch, and the 65-node node/breaker power flow gives one physical
solution from which the substation telemetry (noisy, one draw) and the
operator vector on the schematic-normal layout are read, exactly as the
topology family does.  Eight repeated scans are drawn about the noiseless
operating point with the layout's sensor sigmas, which is how the tabular
parameter corpus draws its scans (`Transmission/generate_measurements.py`:
`z_true` plus fresh noise, same operating point).  The structural-zero rows
of the operator vector (sections without an injection meter) are exact WLS
constraints; the scans declare the nominal injection sigma on them because
the multi-scan corrector takes a strictly positive variance vector.  The
corrected operator model is the planning case with the true impedances and
the true breaker map rendered (a bus split re-renders the layout, a dangling
terminal keeps the 14 buses with its line out).  The meter overlay is the
existing additive one (0.10 to 0.30 pu, lifted to the 10-sigma floor) on P/Q
channels; `on_fault` placement puts the first meter on a flow channel of a
faulted line.

**Placement is read as bus-graph distance.**  Two lines are `adjacent` when
they share a bus and `distant` when no bus of one is within one branch of a
bus of the other.  A breaker is `same_line` when it isolates a terminal of
the faulted line, `shared_bus` when its yard is an endpoint of the line but
not its own terminal, `distant` at two or more branches.  Two breakers must
open exactly two more sections than the normal map and keep their catalogued
effects beside each other (the deployed classifier decides); two terminals
of one line are not sampled together.

**Admission is physical only**: the reported model alarms (chi-square at
alpha 0.01 or a normalized residual at 4.0, margin 1) and the corrected
operator model with the overlay removed is clean.  No ranking, estimator or
teacher-solvability gate runs.  Rejections are the topology family's
(`compound_true_model_not_clean`, mostly the retained half-charging of a
dangling terminal; `opf_diverged` and islands on some breaker pairs).

**What is recorded offline.**  The hidden truth lists every atom in the
existing fields; the `compound` block (under `audit` in the suite) records
the atoms, their pairwise relations, each structural atom's *solo effect*
(its noiseless operator vector's chi-square ratio on the reported model,
alone), the deployed parameter ranking's view of the true lines and the
node/breaker estimator's view of the true breakers.  The policy-visible
metadata carries the same fields for every class: substation telemetry, the
schematic-normal reported map, the layout, the operator noise, the model
identity and the parameter scans.  That uniformity is one more constraint of
the strict evidence boundary, and it has a consequence (section 5): in the
conventional families only parameter roots carry scans, so the parameter
route is never actionable on a breaker root; here it always is.

The existing families' draws are unchanged (the family is built last; the
committed and the new generator produce identical ids and vectors on the same
seed), and `psse_env/providers/test_compound_scenarios.py` covers the
specification, the placements, the re-solved truth, the admission, the scans,
the boundary, the determinism and the deployed executors on compound roots.

## 2. The suite

`build_suite.py`, seed 20261027, one generator per class seeded from the suite
seed and the class name, `scada_only`, 10-sigma meter floor, 278 s on 14
workers.  Fourteen classes, 1,275 roots:

| class | n | strata of the deployed parameter ranking (dominant / ambiguous / misranked / unranked) | roots whose structural atoms each alarm alone | meters on a faulted element | alarm ratio, median |
| --- | --- | --- | --- | --- | --- |
| two R/X lines, adjacent | 150 | 91 / 46 / 13 / 0 | 90 | 0 | 17.8 |
| two R/X lines, distant | 150 | 93 / 37 / 20 / 0 | 29 | 0 | 19.0 |
| R/X + breaker, same line | 75 | 70 / 3 / 2 / 0 | 46 | 0 | 13.1 |
| R/X + breaker, shared bus | 75 | 4 / 5 / 41 / 25 | 47 | 0 | 31.2 |
| R/X + breaker, distant | 150 | 27 / 20 / 53 / 50 | 88 | 0 | 25.1 |
| meter + two R/X | 150 | 86 / 37 / 26 / 1 | 58 | 12 | 18.3 |
| meter on the faulted line + R/X | 75 | 50 / 16 / 6 / 3 | 41 | 75 | 5.5 |
| R/X + breaker + meter | 150 | 29 / 17 / 45 / 59 | 80 | 12 | 27.0 |
| two breakers, both dangling | 60 | n/a | 52 | 0 | 29.7 |
| two breakers, dangling + split | 45 | n/a | 38 | 0 | 96.3 |
| two breakers, both splits | 45 | n/a | 45 | 0 | 125.8 |
| control: one R/X line | 50 | 38 / 4 / 7 / 1 | 33 | 0 | 3.6 |
| control: one breaker | 50 | n/a | 46 | 0 | 18.3 |
| control: 1 to 3 meters (15 / 13 / 22) | 50 | n/a | n/a | 0 | 6.2 |

Reading the columns.  The stratum is the production threshold 1.2 applied to
the deployed parameter context's ranking on the root (the best-ranked true
line's rank; `unranked` when no true line is among the ranked lines, which on
the breaker classes means the dangling line outranks every true line).  A
root's structural atoms "each alarm alone" when every one of them, alone at
the same operating point and without noise, pushes the reported model's
chi-square over its threshold (the `solo effect`); the other roots carry an
atom whose own effect is below the alarm, which the physical admission
allows and the tables below separate.  The alarm ratio is the root's
chi-square over the threshold.

Admission cost: the two-line and meter classes admit almost every draw (0 to
6 rejections per class, the reported model not alarming); the breaker
classes reject 29 to 90 draws each as `compound_true_model_not_clean`, the
topology family's own rejection (the operator model with a line out does not
carry the retained charging of the physical dangling terminal), and the
breaker pairs also lose draws to OPF divergence and islanding (9 + 8 of 149
attempts for two dangling terminals, 22 + 2 of 70 for two splits).  Every
class reached its planned count; the manifest records every class seed and
every rejection.

## 3. The expert

`rollout.py`: `baseline_compare` under `scada_only` with the node/breaker
telemetry (the 2026-10-22 conventional cell's configuration), 40-action
budget, the paired-evaluation contract, every root once.

**Per class** (`expert_baseline_compare/summary.{json,md}`, `report/report.md`;
14.6 min on 14 workers).  "Solved" is the truth-audited success of the
evaluator (every atom corrected within the release tolerances, nothing
healthy touched; a bounded handoff counts).  "Atoms each alarm alone" is the
subset of section 2.  The verdicts are the deployment verifier's dispositions
of the expert's corrections (the offline disposition the commit or rollback
step carried), split by whether the target is a true one; "refused" is an
executor failure, here always the multi-scan corrector not converging.

| class | n | solved | solved, atoms each alarm alone | mean actions | mean WLS calls | comparisons | true corrections: final / partial / rejected / refused | wrong-target corrections accepted | handoff | budget exhausted |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| two R/X lines, adjacent | 150 | 50 (33%) | 39 of 90 (43%) | 16.7 | 4.25 | 229 | 100 / 182 / 21 / 35 | 61 | 150 | 0 |
| two R/X lines, distant | 150 | 72 (48%) | 25 of 29 (86%) | 10.5 | 2.75 | 90 | 140 / 91 / 2 / 8 | 5 | 150 | 0 |
| R/X + breaker, same line | 75 | 0 (0%) | 0 of 46 (0%) | 16.6 | 5.11 | 50 | 62 / 2 / 0 / 53 | 46 | 75 | 0 |
| R/X + breaker, shared bus | 75 | 32 (43%) | 21 of 47 (45%) | 21.0 | 6.11 | 93 | 46 / 59 / 9 / 21 | 61 | 74 | 1 |
| R/X + breaker, distant | 150 | 73 (49%) | 52 of 88 (59%) | 18.1 | 5.35 | 139 | 112 / 105 / 17 / 14 | 77 | 150 | 0 |
| meter + two R/X | 150 | 61 (41%) | 34 of 58 (59%) | 16.7 | 4.41 | 132 | 116 / 249 / 26 / 24 | 16 | 150 | 0 |
| meter on the faulted line + R/X | 75 | 14 (19%) | 4 of 41 (10%) | 14.8 | 4.19 | 26 | 31 / 45 / 37 / 3 | 17 | 74 | 1 |
| R/X + breaker + meter | 150 | 55 (37%) | 32 of 80 (40%) | 23.0 | 6.70 | 142 | 112 / 224 / 34 / 21 | 83 | 150 | 0 |
| two breakers, both dangling | 60 | 16 (27%) | 13 of 52 (25%) | 25.7 | 7.55 | 108 | 24 / 36 / 0 / 0 | 92 | 60 | 0 |
| two breakers, dangling + split | 45 | 1 (2%) | 1 of 38 (3%) | 27.2 | 8.98 | 70 | 3 / 39 / 0 / 0 | 55 | 21 | 24 |
| two breakers, both splits | 45 | 0 (0%) | 0 of 45 (0%) | 26.8 | 8.71 | 80 | 2 / 23 / 0 / 0 | 61 | 20 | 25 |
| control: one R/X line | 50 | 43 (86%) | 31 of 33 (94%) | 7.5 | 2.06 | 16 | 47 / 0 / 0 / 1 | 2 | 50 | 0 |
| control: one breaker | 50 | 31 (62%) | 27 of 46 (59%) | 14.8 | 4.68 | 27 | 32 / 2 / 0 / 0 | 25 | 49 | 1 |
| control: 1 to 3 meters | 50 | 47 (94%) | 47 of 50 (94%) | 21.3 | 5.60 | 29 | 48 / 58 / 0 / 0 | 0 | 50 | 0 |
| all | 1,275 | 495 (39%) | 326 of 743 (44%) | 17.9 | 5.13 | 1,231 | | | 1,223 | 52 |

No episode has a false commit by the verifier's own standard (every commit
was of an accepted candidate) and no invalid action or loop was flagged; the
failures are wrong targets the verifier accepted, true targets it refused or
the corrector could not estimate, and the budget on the breaker pairs.
Mean WLS calls count the evaluator's solves and verifications; the 1,231
bounded comparisons add their own internal refits.

**By stratum of the deployed parameter ranking** (solved / n): two adjacent
lines, dominant 26/91, ambiguous 18/46, misranked 6/13; two distant lines
47/93, 16/37, 9/20; meter + two R/X 33/86, 17/37, 11/26; the breaker classes
are mostly `misranked` or `unranked` because the dangling line outranks the
true line (R/X + breaker distant: dominant 17/27, ambiguous 12/20, misranked
25/53, unranked 19/50; R/X + breaker + meter 9/29, 5/17, 18/45, 23/59).  The
single-line control is 33/38 dominant, 4/4 ambiguous, 6/7 misranked.

**What the corrections were, class by class** (the per-tool verdict split is
in `expert_baseline_compare/summary.json`).

*Two R/X lines.*  On distant pairs the expert corrects the first line
(140 final verdicts: the solve is clean after one line on most roots) and
hands off; 66 of the 78 failures leave the second line in place
(`resolved_episode_has_remaining_true_faults`) with almost no wrong target
accepted (5).  121 of the 150 distant roots carry a line whose own effect is
below the alarm, and on the 29 roots where both lines alarm alone the expert
solves 25.  On adjacent pairs the picture is different: 229 comparisons
(1.5 per root), 35 true corrections refused by the corrector, 21 true
corrections rejected, and 61 wrong targets accepted (parameters 11 final + 29
partial, meters 4 final + 16 partial); 43% on the roots whose atoms each
alarm alone.  The adjacent line's multiplier is close to the true line's
(median dominance ratio 1.07 in the probes of section 5) and a correction
of the wrong neighbour makes partial progress that the verifier accepts.

*One R/X line + one breaker.*  Every one of the 300 episodes starts with the
parameter route: the parameter context is actionable on every root (the
scans are present), its top line is usually the dangling line itself
(section 5), and the structural-first order puts the parameter correction
before the topology context.  The corrector fails on an isolated line
(`parameter_correction_failure`, 124 + 126 + 236 refusals on the three
classes) or, when it converges, explains the open terminal as an inflated
impedance that the verifier accepts (wrong parameter corrections accepted:
40, 52 and 76 on the same-line, shared-bus and distant classes; the audit
reasons `accepted_parameter_target_outside_same_family_truth` 12, 32 and 57).
On the same-line class the breaker is fixed on 49 of 75 roots (47 final + 2
partial verdicts) and the true line's correction is refused on 53: after the
breaker fix the line carries no flow and its impedance has no effect on the
measurements, so the corrector has nothing to fit, and the truth audit still
asks for the true R/X within tolerance (`final_case_outside_clean_tolerance`
on all 75).  No root of that class is solvable under the current audit.

*Meter + lines.*  With a meter and two lines the expert solves 61 of 150
(59% of the roots whose atoms each alarm alone); the true meter is corrected
on 138 roots (41 final + 97 partial) and 24 true line corrections are refused
by the corrector.  With the meter on the faulted line (75 roots) it solves
14: on 22 roots the true line's correction is rejected (its own flow meter
keeps the line's multiplier above the target test) and the meter route never
opens; on 13 the meter correction alone makes the solve clean
(`ACCEPT_FINAL`) and the line error is masked (`final_measurements_outside_
clean_tolerance` 39 of 61 failures); 12 of the 14 successes correct the line
first and then the meter.  The three-atom class R/X + breaker + meter solves
55 of 150 with 23.0 actions and 6.7 WLS calls per episode, the costliest
class with a meter.

*Two breakers.*  Both dangling: 41 of 60 episodes fix at least one
breaker and 19 fix both (16 of them solved; the other 3 also accepted a
wrong impedance); on 22 the first breaker is fixed and the remaining anomaly
is then explained by a parameter correction, usually of the second dangling
line, accepted as final, so the episode hands off with that line's R/X
inflated instead of its terminal opened; on 19 no breaker is fixed at all
(14 of them accept a parameter correction on a dangling line first).  In
all, 92 wrong targets are accepted in the class.  Dangling + split solves 1
of 45 and two splits none: after a bus-split correction re-renders the
operator model to 15 buses, the node/breaker estimator indexes the model's
line terminals by the re-rendered bus numbers and fails (`KeyError: (15, 7)`
in `node_network`), so `get_topology_context` returns `dispatch_error`, a
second breaker correction fails with `topology_correction_telemetry_incomplete`,
and the expert, with no route left, exhausts the 40-action budget (26 of the
45 dangling + split episodes fix the split first and 24 of them run out of
budget; 28 of the 45 two-split episodes fix one split and 25 run out; the
remaining episodes of both classes fix no breaker and accept a wrong
impedance instead).  This is the interface limit the legacy pipeline's
"curriculum only" label on parameter + topology combinations anticipated,
now located: the binding reads the current case as the electrical
reference, which is right for corrected impedances and wrong for a
re-rendered layout.

*Controls.*  The single R/X line is at 86% (94% where the line alarms alone;
the 7 failures are 3 lines left in place below tolerance and 2 wrong
targets).  Meters are at 94%.  The single breaker is at 62% (59% where the
breaker alarms alone), against 100% on the conventional cell's topology
roots: the difference is the scans.  A
conventional topology root carries no `parameter_scans`, so its parameter
route is never actionable and the expert goes to the topology context at
once; a compound root carries scans on every class (the boundary demands it),
the parameter route is actionable, and the expert corrects the dangling
line's impedance first (23 wrong parameter corrections accepted, 82 refused
attempts, every one of the 50 episodes starts with the parameter route).
Every one of the 19 failures carries a wrong parameter correction accepted
before or beside the breaker route: 16 never reach the breaker fix and 3 fix
the breaker but keep the inflated impedance.  The ordering, not the
evidence, costs them: the breaker fix alone resolves 40 of 40 probed roots
(section 5), and the 9 bus-split roots the first scoring pass failed were
accepted fixes whose truth rows lacked the `expected_status` the audit reads
(fixed in the synthesizer and in the suite; those three classes were
re-rolled).

### 3.1 The same suite under a topology-first order (flagged variant)

Is the parameter-before-breaker order a teacher defect or an artifact of the
suite's metadata?  Both, and the two are the same fact seen from two sides.
The conventional suites never show the teacher a breaker root with parameter
scans, so the family leaks through scan availability and the parameter route
is simply never actionable on a topology root; the compound suite closes
that leak (every root carries scans and the node/breaker channel, as the
boundary requires) and the teacher's structural tier then runs
parameter-first on every breaker root.  The teacher's rules for the seven
conventional families are untouched by what follows: the order is a flag,
`ExpertPolicyOracle(topology_first=True)`, off by default, exposed as the
variant `baseline_compare_topology_first` (`psse_env/oracle/expert_variants.py`;
`psse_env/oracle/test_topology_first_order.py` checks that `baseline_compare`
keeps its order).  With the flag the topology context and corrections precede
the parameter context, comparison and corrections inside the structural
tier; the meter tier and the order inside each group are unchanged.  The same
1,275 roots rolled out under it (`expert_baseline_compare_topology_first/`,
18 min on 14 workers), paired root by root with the baseline run:

| class | baseline_compare: solved, mean actions | topology first: solved, mean actions | roots won / lost |
| --- | --- | --- | --- |
| two R/X lines, adjacent | 50, 16.7 | 42, 18.9 | 0 / 8 |
| two R/X lines, distant | 72, 10.5 | 63, 14.2 | 0 / 9 |
| R/X + breaker, same line | 0, 16.6 | 0, 7.0 | 0 / 0 |
| R/X + breaker, shared bus | 32, 21.0 | 43, 13.5 | 12 / 1 |
| R/X + breaker, distant | 73, 18.1 | 109, 12.2 | 39 / 3 |
| meter + two R/X | 61, 16.7 | 51, 19.8 | 0 / 10 |
| meter on the faulted line + R/X | 14, 14.8 | 14, 15.0 | 0 / 0 |
| R/X + breaker + meter | 55, 23.0 | 89, 16.3 | 35 / 1 |
| two breakers, both dangling | 16, 25.7 | 53, 11.0 | 37 / 0 |
| two breakers, dangling + split | 1, 27.2 | 11, 14.0 | 10 / 0 |
| two breakers, both splits | 0, 26.8 | 0, 15.4 | 0 / 0 |
| control: one R/X line | 43, 7.5 | 37, 10.0 | 0 / 6 |
| control: one breaker | 31, 14.8 | 50, 7.0 | 19 / 0 |
| control: 1 to 3 meters | 47, 21.3 | 47, 21.4 | 0 / 0 |
| all | 495 (39%), 17.9 | 609 (48%), 14.7 | 152 / 38 |

On the roots whose atoms each alarm alone the ceiling goes from 326 to 419
of 743 (44% to 56%); mean WLS calls per episode from 5.13 to 3.87 and
bounded comparisons from 1,231 to 618.  The breaker classes are repaired
(the single-breaker control at 50 of 50 in 7.0 actions, both-dangling pairs
at 53 of 60, R/X + breaker distant at 109 of 150 with 77 of 88 where both
atoms alarm alone), and every episode of the dangling + split class that
fixes the dangling terminal first now fixes the split too (11 of 11); the
34 that fix the split first still run into the re-render `KeyError` and the
budget, as do all 45 two-split episodes.  The same-line class stays at 0:
the breaker is fixed first on all 75 roots in 7 actions and the open line's
impedance is then not estimable (section 5.2).

The losses are the mirror image of the gains.  On the parameter-only
classes the topology context now comes first and the node/breaker screening
sometimes admits a breaker flip whose partial progress clears the floor on a
root that has no breaker error: a wrong flip is accepted on 6 single-line
controls, 19 adjacent and 28 distant pairs, 17 meter + two R/X roots
(`accepted_topology_target_outside_same_family_truth` on 6, 9 and 10 of the
lost roots of the single-line, distant and meter classes), and 38 roots that
the baseline solved are lost.  An order alone therefore trades one masking
for the other; what separates the two cases is observable and is the
stage-2 item: a dangling line's terminal flows sit at noise level and its
flip explains the substation measurements completely, while a wrong flip on
a parameter root only makes partial progress.  Topology first with a
clean-flip or flow-at-noise condition is the recommended rule (section 6);
the flag as it stands is the measured upper bound of what the order alone
buys, 114 net roots.

## 4. The decision audit

RESULTS_AUDIT

## 5. Identifiability of the flagged pairs

`identifiability.py` (`identifiability/summary.json`, 11 min on 8 workers;
`identifiability_masking/` for the fourth probe, 102 s).  Every probe runs on
the deployed executors and verifier, each candidate in a fresh research
environment, on two paths: the *gated* path is the environment with its
process gates (a correction the fresh context did not advertise is refused
with `correction_not_supported_by_current_context`; the meter route stands
down under branch dominance; after an `ACCEPT_FINAL` commit only the
confirmation handoff is allowed, `post_correction_confirmation_required`),
the *ungated* path applies the same executor and the deployment candidate
oracle without those gates, as the provider's own non-mutating screening
does.  "Case ok" and "measurements ok" are the truth audit's final-case and
final-measurement checks applied to the end state.

### 5.1 Same-line impedance + flow meter (75 roots, with a twin each)

Each root's twin carries the same meter offset moved to a healthy P/Q
channel, so the pair's own effect is read against a paired control.

| | original: meter on the faulted line | twin: meter elsewhere |
| --- | --- | --- |
| WLS dominance at the root (residual / branch / none) | 41 / 27 / 7 | 40 / 29 / 6 |
| true line ranked first by the parameter context | 66 of 75 | 48 of 75 |
| true meter among the supported meter groups | 48 of 75 | 10 of 75 |
| meter corrected alone, gated (final / partial / refused by the gate) | 13 / 35 / 27 | 12 / 34 / 29 |
| ... solve clean afterwards | 13 | 12 |
| line corrected alone, gated (partial / rejected / refused or failed) | 3 / 61 / 11 | 35 / 25 / 15 |
| ... estimate within the release tolerance | 55 of 64 estimates | 54 of 60 |
| line corrected alone, ungated (partial / rejected / failed) | 1 / 67 / 7 | 34 / 34 / 7 |
| meter then line, ungated: both checks pass | 29 of 75 | 36 of 75 |
| line then meter, ungated: both checks pass | 1 of 75 | 33 of 75 |

The multi-scan corrector is not the problem: with the eight clean historical
scans it estimates the true impedance within tolerance on 55 of 64 roots
(median error 3% on X, 6% on R), the same as on the twins.  The verifier is:
the correct line correction is rejected on 61 of 64 gated and 67 of 68
ungated roots, against 25 of 60 on the twins, because the bad flow meter on
the same line keeps that line's own multiplier (the target test of a
parameter candidate) above its threshold after the impedance is fixed.  On
the originals the target metric sits at 2.6 times its threshold after the
fix (median; 1.6 to 4.4 from the 10th to the 90th percentile) with zero
target progress, and the rejection classes are `no_target_progress` (57) and
`healthy_component_corruption` (10); on the twins the same fix brings the
metric to 0.41 of the threshold with full target progress, and the verdicts
are 34 partial acceptances and 29 rejections for `insufficient_global_progress`
(the meter error elsewhere still holds the global statistic up).  The
meter correction alone is accepted and on 13 roots leaves a clean solve that
hides the impedance error; the order that works, meter first and line
second, passes both checks on 29 of 75 ungated roots and is what 12 of the
expert's 14 successes did.  So the pair is identifiable in principle (the
estimator and the meter residual both point at the right targets), but not
through the current single-target verification: this is an interface limit,
not an evidence limit.  Among the expert's 61 failures, 22 rejected the
correct line correction and never opened the meter route, and 13 accepted
the meter correction as final.

### 5.2 Same-terminal impedance + breaker (75 roots)

| | gated | ungated |
| --- | --- | --- |
| breaker alone: verdict final; solve clean; measurements ok; case ok | 75; 75; 75; 0 | 75; 75; 75; 0 |
| breaker then line: line correction | refused 75 (`post_correction_confirmation_required`) | failed 75 (corrector diverges) |
| line alone: corrector fails / converges | 60 / 15 | 60 / 15 |
| ... converged estimates: median R, X error | 455x, 137x | 455x, 137x |
| ... verdict on the converged estimates | `ACCEPT_FINAL` 15 | `ACCEPT_FINAL` 15 |
| line then breaker: breaker verdict | final 60, refused 15 after the masking commit | final 75 |

The WLS is branch-dominant on 72 roots and the parameter context ranks the
true line first on 73, because the dangling terminal is on that line.  The
line's four terminal-flow readings sit below three sigma on 66 of 75 roots
(median 0.5 to 0.8 sigma): the line carries no current, so its impedance has
no effect on any measurement.  The breaker fix alone is final on every root
and the solve is clean, yet the truth audit fails all 75 on
`final_case_outside_clean_tolerance` because the true R/X is not in the final
case.  When the corrector is tried first it diverges on 60 roots and on the
other 15 inflates the impedance by two orders of magnitude, which the
verifier accepts as final (the open terminal explained as a near-infinite
line).  This is an evidence limit by construction: the impedance of an
isolated line is not identifiable from any snapshot, and the right behaviour
is the breaker fix plus an explicit abstention on that line's parameters,
which the audit should score as not applicable.

### 5.3 Two wrong lines: adjacent (150) against distant (150)

| | adjacent | distant |
| --- | --- | --- |
| both true lines in the context's top 2 / top 3 | 71 / 95 | 15 / 33 |
| a true line ranked first | 137 | 130 |
| one true line unranked | 30 | 108 |
| ranking dominant (median ratio) | 92 (1.32) | 94 (1.50) |
| bounded comparison of the supported candidates: best is a true line / no best | 127 / 6 of 148 | 148 / 0 of 149 |
| first line, estimated on the stale model: within tolerance (median X error) | 73 of 126 (26%) | 143 of 147 (1%) |
| second line, on the derived model: within tolerance (median X error) | 65 of 129 (20%) | 106 of 136 (9%) |
| first line again, one alternating round, ungated: within tolerance (median X error) | 112 of 131 (6%) | 147 of 148 (1%) |
| after the round, ungated: solve clean / case ok | 102 / 59 | 140 / 103 |
| gated: second line refused after a final commit | 19 | 57 |

Adjacent pairs are the identifiability case the assessment flagged: a line's
impedance estimated while its neighbour is still wrong is off by a quarter
(26% median on X, within tolerance on 73 of 126), the neighbour's estimate
on the derived model is off by a fifth, and one more pass over the first
line brings it within tolerance on 112 of 131.  The ranking itself is not
the obstacle (a true line is first on 137 of 150 and the bounded comparison
picks a true line on 127 of 148); the single-line corrector applied once
per line is.  Distant pairs separate: the first estimate is already within
tolerance on 143 of 147 and the open question there is detection, since on
108 roots one line is not even ranked (its own effect is below the alarm)
and after the first line's final commit the environment admits no second
correction (57 refusals), which is the regime where the expert's 86% on the
roots whose atoms each alarm alone comes from.

### 5.4 The parameter route on breaker roots (40 roots of four classes)

| class | parameter route actionable | context's top line is the dangling line | top-line correction: corrector fails | converged: median X inflation | gated verdict on the top line (final / partial) | breaker(s) alone: solve clean |
| --- | --- | --- | --- | --- | --- | --- |
| control: one breaker | 40 | 26 | 28 | 63x | 11 / 1 | 40 |
| R/X + breaker, distant | 40 | 26 | 19 | 2.6x | 0 / 21 | 2 (the line remains) |
| R/X + breaker, shared bus | 40 | 29 | 25 | 4.7x | 2 / 13 | 5 (the line remains) |
| two breakers, both dangling | 40 | 38 | 32 | 88x | 0 / 8 | 34 (after both) |

The parameter context is actionable on every root because every root
carries scans, its top line is the dangling line on 26 to 38 of 40, the
corrector diverges on that line on 19 to 32 of 40 and otherwise explains the
open terminal by an inflated impedance that the verifier accepts (final on 11
of 40 single-breaker controls).  The breaker fix alone leaves the solve
clean on 40 of 40 controls.  The structural-first order of the expert
(parameter context before topology context, parameter correction before
either) therefore does not hold on parameter + breaker roots, nor on breaker
roots whose metadata does not reveal the family: every one of the 300
R/X + breaker episodes and all 50 single-breaker controls open with the
parameter route, and the wrong parameter corrections it commits before the
breaker fix are the main failure of those classes (section 3).

## 6. Evidence limits, interface limits, and the stage-2 recommendation

### 6.1 Which failures are which

AUDIT_BINS_SUMMARY

*Evidence limits* (no correction the interface can represent would pass the
truth audit):

- the impedance of a line whose terminal is open (`R/X + breaker, same
  line`, 75 roots, and the same pair whenever it arises inside a larger
  set): unidentifiable from any snapshot, section 5.2;
- a structural atom whose own effect is below the alarm (the second line of
  121 of the 150 distant pairs, 60 of the adjacent pairs, 92 of the meter +
  two R/X roots and so on, section 2): after the first fix the solve is
  clean, the environment admits no further correction, and the audit still
  asks for the atom.  On the roots whose atoms each alarm alone the expert
  solves 326 of 743 (44%) against 495 of 1,275 overall.

*Interface limits* (the truth probe or an alternative action solves the
root; the deployed rule, verifier or executor does not):

- the parameter route before the topology route on every breaker root, and
  the executor's inflated-impedance explanation of an open terminal that
  the verifier accepts (sections 3 and 5.4): the single-breaker control
  loses 19 of 50 roots to it, the R/X + breaker classes 12 to 57 roots each
  (`accepted_parameter_target_outside_same_family_truth`), both-dangling
  pairs 41 of 44 failures;
- the node/breaker binding after a bus-split re-render (`KeyError` in
  `node_network`; section 3): no second breaker correction and no topology
  context on 50 of 90 dangling + split and two-split episodes;
- the single-target verification of a line whose own flow meter is bad
  (section 5.1): the correct impedance is rejected on 61 of 64 roots while
  the estimate is within tolerance on 55;
- the single-line corrector applied once per line on adjacent pairs
  (section 5.3): the first estimate is off by a quarter and within
  tolerance on 73 of 126, one more pass brings 112 of 131 within tolerance;
- the post-commit confirmation gate after an `ACCEPT_FINAL` verdict, which
  forbids the second correction when the first one leaves a clean solve
  (57 distant-pair and 19 adjacent-pair roots in the probes): an interface
  rule that is right for single faults and wrong when a second atom is known
  to the operator but not to the alarm.

### 6.2 The stage-2 tool changes, with the evidence

1. **Topology before parameters whenever the node/breaker channel is
   present, and no parameter correction on a line whose terminal flows are
   at noise level.**  Evidence: every breaker-bearing episode (350 + 150
   two-breaker) opens with the parameter route; the breaker fix alone
   resolves 40 of 40 single-breaker probes and 34 of 40 double-dangling
   probes; the parameter corrector diverges on the dangling line on 19 to
   32 of 40 probed roots and otherwise inflates its reactance 63 to 88
   times (median), which the verifier accepts as final on 11 of 40
   controls.  The dangling line's four flow readings sit below three sigma
   on 66 of 75 same-line roots: a flow-at-noise test on the candidate line
   is observable and would both block the masking correction and route to
   the topology context.  Measured (section 3.1): the order alone, as the
   flagged variant `baseline_compare_topology_first`, wins 152 roots and
   loses 38 (495 to 609 of 1,275; the single-breaker control 31 to 50 of 50,
   both-dangling pairs 16 to 53 of 60, R/X + breaker distant 73 to 109 of
   150) and halves the actions on the breaker classes; the 38 losses are
   wrong breaker flips accepted on parameter roots, which the flow-at-noise
   or clean-flip condition is there to prevent.  This is the change that
   moves the most roots.

2. **Repair the node/breaker binding for re-rendered layouts.**  Evidence:
   `get_topology_context` fails with `dispatch_error` and a second breaker
   correction with `topology_correction_telemetry_incomplete` on every
   episode that fixes a bus split first (26 of 45 dangling + split, 28 of 45
   two-split), which then exhaust the budget.  The binding should read the
   planning case through the layout's `node_to_bus` instead of indexing the
   model's terminals by the current case's bus numbers.  Without it the
   two-split class is unsolvable at 0 of 45.

3. **Joint pair refits in the bounded comparison.**  Evidence: on adjacent
   pairs the first single-line estimate is within tolerance on 73 of 126
   (median X error 26%), the second on 65 of 129, and one alternating pass
   over the first line raises it to 112 of 131 (6%); the ranking already
   puts a true line first on 137 of 150 and the comparison picks a true line
   on 127 of 148, so the information is there and the one-shot single-line
   estimator is what loses it.  A pair refit (both lines' R and X in one
   multi-scan fit, or the alternating pass) inside the comparison of the top
   two candidates would turn the 33% adjacent class toward the 86% the
   distant pairs reach where both atoms alarm.  The step-1 screen already
   has an offline joint R/X refit (`docs/hypothesis_ranking_step1_20260930.md`).

4. **Progress floors scaled by the number of open suspects: not supported
   as a plain rule; replace the floor by target-local evidence.**
   Evidence (section 4's floor table from the 2,136 verified true and 1,548
   wrong corrections of the rollout): true corrections verified with two
   faults still open are rejected 75 times, 43 of them below the branch or
   meter floor, and a floor divided by the number of open faults would
   rescue 24 of them; but the same rule would admit 106 of the wrong
   corrections rejected at two open faults and 49 at three, against 22 at
   one.  The verifier already accepts 195 wrong corrections at two open
   faults and 44 at three (362 at one); loosening the global floor makes
   that worse.  What separates the rejected true corrections from the wrong
   ones in the same state is target-local: a true correction's own target
   metric clears its threshold (the twins of 5.1: metric at 0.41 of the
   threshold, full target progress) while a wrong one's does not, and the
   true correction's global progress distribution (median 0.73 at two open
   faults) sits above the wrong ones'.  So the recommendation is a
   target-first partial acceptance (target metric cleared and global
   progress positive) with the global floor kept only for corrections
   whose target metric is not cleared, together with the pair refit of
   item 3 for the adjacent case.

5. **An abstention rule for unidentifiable pairs, and the audit to match.**
   Evidence: 75 of 75 same-terminal roots end with the breaker fixed, a
   clean solve and a failed final-case check; the corrector has nothing to
   fit (75 of 75 ungated failures after the breaker fix).  The rule: after a
   breaker fix that isolates a line, declare that line's impedance *not
   identifiable from this snapshot* in the handoff (an explicit abstention,
   not a correction), and let the truth audit score an isolated line's
   parameters as not applicable.  The same abstention applies to a
   structural atom whose own effect is below the alarm once the solve is
   clean, where today the episode is scored a failure for a fault no
   snapshot statistic can see.

6. **Does the structural-first ordering hold for parameter + breaker
   roots?**  No, in its current form.  Both are structural, so the order
   between them is decided by the context tie-break (parameter context
   before topology context) and by the parameter correction's higher
   confidence once its context is fresh; on 300 of 300 R/X + breaker
   episodes the parameter route runs first and on 40 to 76 roots per class
   it commits a wrong impedance before the breaker is touched.  Under the
   flagged topology-first order (section 3.1) the breaker is fixed first on
   70 of 75 shared-bus, 111 of 150 distant and 103 of 150 three-atom roots,
   the remaining line is then estimated on a model with the true topology,
   and those classes go to 43, 109 and 89 solved (from 32, 73 and 55); the
   same-line class stays at 0 for the reason of item 5.  The structural-first
   rule should become topology-first within the structural tier, with item
   1's flow-at-noise or clean-flip condition as the observable trigger so
   the parameter classes do not pay for it (38 roots under the bare flag).

### 6.3 What stage 2 should build, in order

Items 1 and 2 are executor and routing fixes with the largest and cleanest
gains (the breaker classes and the two-breaker pairs); item 5 is an audit
and handoff contract; items 3 and 4 change the comparison and the verifier
and need the stage-1 suite as their test bed.  None needs training: the
suite, the rollout runner and the probes here re-run in about half an hour
on 14 CPU workers, so each change can be read off the same 1,275 roots
before any DAgger round is drawn on compound classes.
