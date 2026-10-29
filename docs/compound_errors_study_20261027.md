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
45).  The impedance of a line whose terminal is open is not identifiable;
the audit asked for it until the decision of 2026-10-28 scored it as not
applicable in the compound work, which re-scores that class to 47 of 75
(75 of 75 under topology-first) and the ceilings to 546 (43%) and 692
(54%), section 3.2.  The adjacent-line pair is an
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

### 3.2 The isolated-line rule (decision (b), 2026-10-28)

The rule, implemented in `psse_env/dagger/release_audit.py` and switched on
by a mark only compound roots carry: a parameter fault whose truth row says
`not_applicable: line_isolated_by_breaker`, and whose branch row is also the
row of a `dangling_line_terminal` topology fault of the same truth, is
excluded from the remaining-fault count (derived and supplied ledger), from
the final-case comparison and from the accepted-target non-regression check
(a committed estimate on that line is recorded as `not_applicable`, neither
regression nor repair).  It stays a declared target for the accepted-target
and healthy-case checks (an edit on it is not a false target), the breaker
fault stays required, and every other check is untouched, so a root with a
wrong target beside the breaker fix still fails.  The mark counts only
beside the isolating breaker's own truth row: a mark without that breaker,
or an unmarked fault, is required exactly as before, and the seven
conventional families, whose truths never carry the mark, score as they
always did (`IsolatedLineNotApplicableTests` in
`psse_env/dagger/test_release_audit.py`; the generator writes the mark,
`psse_env/providers/test_compound_scenarios.py`; the suite's 84 marked rows
were patched in place).  The rule touches 84 roots: the 75 same-line roots
and 9 three-atom roots whose breaker atom happened to isolate the faulted
line.  Those roots were re-run under the rule for both teachers
(`isolated_line_rescore/`, `expert_baseline_compare_rule_b/`,
`expert_baseline_compare_topology_first_rule_b/`); every trajectory is
identical to the original run, so this is a re-scoring of the same
rollouts.

| class | baseline_compare: before, after | topology first: before, after |
| --- | --- | --- |
| R/X + breaker, same line (75) | 0, 47 | 0, 75 |
| R/X + breaker + meter, the 9 roots with an isolated line | 0, 4 | 0, 8 |
| all 1,275 | 495 (39%), 546 (43%) | 609 (48%), 692 (54%) |

What stays failed under the baseline teacher (28 same-line roots): on 26
the breaker is never fixed because a parameter correction, usually on a
neighbouring line, made the solve clean first and the episode handed off
(`resolved_episode_has_remaining_true_faults` 26, with
`accepted_parameter_target_outside_same_family_truth` on 12 and wrong meter
targets on 6); on 2 the breaker is fixed but a wrong target was accepted
beside it.  Under topology-first every same-line root passes (the breaker
is fixed first in 7 actions and the line is left alone) and the one
remaining three-atom failure is a meter outside tolerance.

Two readings the rule leaves open.  A line isolated at one terminal is the
only case the suite contains (the sampler never opens both terminals of one
line); it is de-energised from that end, carries no current, and the rule
applies.  An isolated line the expert nevertheless corrected: the rule
records the committed estimate as not applicable rather than as a
regression (the lenient reading); a stricter reading that counts such a
commit against the root changes nothing here, because on every root where
the expert committed an estimate of the isolated line the solve was clean
afterwards and the episode handed off with the breaker unfixed, which fails
either way (47 and 75 of 75 under both readings).

## 4. The decision audit

`audit.py` over `research/rl/decision_audit.py` (`audit/`, 46 min on 14
workers): every root the expert failed (780 after the re-roll) and six solved
roots per class, 847 in all, under the same profile and teacher.  For each
root the audit replays the reference trajectory, takes at the first two
decision states (where the expert's own menu holds at least two admissible
actions) up to two menu alternatives each and lets the expert finish, and at
the first decision state offering a correction applies the truth corrections
(every true meter, line and breaker, breakers by name) with the expert
verifying and committing between them.  Two probe repairs were needed for
compound roots and are committed: the probe requests the family context the
process gate demands before each truth correction (without it every
correction of a family other than the expert's current one was refused as
`missing_precondition`), and it retries a correction the context did not
advertise once the others are applied (the topology context offers one
breaker of a pair at a time).  Bins as the plan defines them: `rule_best`
(no alternative beats the reference), `legal_alternative` (a menu alternative
solves a root the expert failed), `interface_limit` (only the truth probe
solves it), `evidence_limit` (nothing does).

| class | audited (failed + solved) | legal alternative | interface limit | evidence limit | rule best | truth probe solves failed roots | the alternative that solves |
| --- | --- | --- | --- | --- | --- | --- | --- |
| two R/X lines, adjacent | 100 + 6 | 6 | 1 | 93 | 5 | 4 of 99 | the other true line at the first correction (5) |
| two R/X lines, distant | 78 + 6 | 0 | 0 | 78 | 6 | 0 of 78 | |
| R/X + breaker, same line | 75 + 0 | 0 | 0 | 75 | 0 | 0 of 75 | |
| R/X + breaker, shared bus | 43 + 6 | 15 | 8 | 26 | 0 | 17 of 43 | `get_topology_context` first (9) |
| R/X + breaker, distant | 77 + 6 | 33 | 13 | 35 | 2 | 42 of 77 | `get_topology_context` first (29) |
| meter + two R/X | 89 + 6 | 1 | 1 | 87 | 6 | 2 of 89 | |
| meter on the faulted line + R/X | 61 + 6 | 0 | 0 | 61 | 6 | 0 of 61 | |
| R/X + breaker + meter | 95 + 6 | 29 | 9 | 60 | 3 | 35 of 95 | `get_topology_context` first (26) |
| two breakers, both dangling | 44 + 6 | 16 | 27 | 7 | 0 | 37 of 44 | `get_topology_context` first (10) |
| two breakers, dangling + split | 44 + 1 | 1 | 10 | 34 | 0 | 10 of 44 | |
| two breakers, both splits | 45 + 0 | 0 | 0 | 45 | 0 | 0 of 45 | |
| control: one R/X line | 7 + 6 | 1 | 0 | 5 | 6 | 1 of 6 | |
| control: one breaker | 19 + 6 | 25 | 0 | 0 | 0 | 19 of 19 | `get_topology_context` first (19) |
| control: 1 to 3 meters | 3 + 6 | 5 | 1 | 1 | 2 | 2 of 3 | a different meter (2) |
| all | 847 | 132 | 70 | 607 | 36 | | |

The legal alternatives are almost all one action: requesting the topology
context instead of the parameter context at the first decision state (step
1 of the trajectory) on breaker-bearing roots, which is what the flagged
variant of section 3.1 does by rule.  On the single-breaker control every
failed root is solved by that alternative and by the truth probe; on the two
R/X + breaker classes and the three-atom class 29, 9 and 26 roots are.

**What the evidence-limit bin contains.**  The bin is the plan's mechanical
definition (no alternative and no truth probe solves the root) and it is not
the same as "the evidence cannot separate the atoms".  Read with the probe's
own refusals and the suite's solo effects, it decomposes as follows (the
probe's refusals are counted over the failed roots of each class):

- *an atom whose own effect is below the alarm*: 74 of the 78 distant
  pairs, 63 of 87 meter + two R/X, 47 of 93 adjacent pairs, 29 of 35
  R/X + breaker distant, 34 of 60 three-atom roots, 24 of 61 same-line
  meter roots carry one; after the first fix the solve is clean and the
  probe's next correction is refused by the post-commit confirmation gate
  (`post_correction_confirmation_required`: 55 distant, 15 adjacent, 14
  meter + two R/X, 13 same-line meter) or, when it runs, has nothing to fit;
- *the isolated line* (all 75 same-line roots; the probe's line correction
  fails 60 times and is refused by the confirmation gate 15 times after the
  breaker fix);
- *the corrector or the context, not the evidence*: the probe's line
  correction fails to converge (`parameter_correction_failure`: 59 adjacent,
  55 meter + two R/X, 27 same-line meter, 20 shared-bus, 13 distant
  breaker roots) or the context does not advertise the true target
  (`correction_not_supported_by_current_context`: the true meter under
  branch dominance on 49 meter + two R/X, 27 same-line meter and 44
  three-atom roots; the true line on 26 adjacent, 25 meter + two R/X, 22
  distant-breaker and 23 three-atom roots).  These are interface limits in
  substance that the probe cannot certify because the probe itself goes
  through the same gates; sections 5.1 and 5.3 certify them directly.

So the honest reading per class is: the single-breaker control and the
R/X + breaker classes are legal-alternative and interface limits (the
ordering and the masking executor), the two-breaker classes are interface
limits (the re-rendered binding; the probe fixes both dangling terminals on 37 of 44 failed both-dangling roots once it retries the breaker the context did not advertise first, 10 of 44 dangling + split roots, and none of the two-split roots, where the context itself fails after the first split), the same-line breaker class
and the undetectable second atoms are evidence limits by construction, and
the adjacent-line and same-line-meter classes are interface limits the
probe misses (the single-line corrector and the single-target verifier,
sections 5.3 and 5.1).

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
which the audit now scores as not applicable in the compound work (section
3.2: 47 of 75 under the baseline teacher, 75 of 75 under topology-first).

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

The decision audit (section 4) bins the 847 audited roots as 36 rule-best, 132 legal-alternative, 70 interface-limit and 607 evidence-limit by its mechanical definition; read with the probe's refusals and the suite's solo effects (section 4), the honest split is below.  The legal alternatives are the topology context requested first on breaker roots; the interface limits the probe certifies are the breaker fixes the expert never reaches; the evidence-limit bin holds the undetectable second atoms and the isolated lines, but also the probe's own refusals by the corrector and the context gates, which are interface limits the probes of section 5 certify.

*Evidence limits* (no correction the interface can represent would pass the
truth audit):

- the impedance of a line whose terminal is open (`R/X + breaker, same
  line`, 75 roots, and the same pair on 9 three-atom roots): unidentifiable
  from any snapshot, section 5.2; since the decision of 2026-10-28 the
  compound audit scores it as not applicable (section 3.2), so the class
  is now limited by the breaker fix alone (47 of 75 reached under the
  baseline teacher, 75 of 75 under topology-first);
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
   snapshot statistic can see.  The audit half of this item is done for the
   compound work (section 3.2); the handoff half, an explicit abstention
   statement by the agent, is stage 2.

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

### 6.4 Decisions (2026-10-28)

On the ordering flag (item 1 and item 6), the user's decision, in their
words: "Keep topology-first flagged to the compound work until the
conditioned version is validated on the conventional suite."  So
`baseline_compare` stays the teacher for every profile, and
`baseline_compare_topology_first` stays a flagged variant of the compound
work until a run under the flow-at-noise or clean-flip condition shows the
conventional suite's expert still at 192 of 192 and the manuscript's numbers
unchanged.  On the audit (item 5), the user's decision, in their words: "Yes: change
the truth audit so an isolated line's impedance is scored not applicable,
in the compound work only."  Implemented and re-scored in section 3.2 (the
rule, the mark that scopes it to compound roots, the tests that show the
conventional families unchanged, 84 roots re-scored on identical
trajectories: 51 change from failed to solved under the baseline teacher
and 83 under topology-first).

On the metadata convention (section 3.1), the user's decision, in their
words: "Yes: keep the uniform-metadata convention, parameter scans on
every root, for future suites."  Future compound suites therefore carry the
scans and the node/breaker channel on every root, as this one does, so the
parameter route is actionable on every class and the ordering question of
section 3.1 stays in view; the conventional suites behind the PES GM
manuscript keep their existing metadata (no scans on topology roots).

Stage 2 was then requested, in the user's words: "Yes, start stage 2 in the
compound session: the tool repairs in the note's section 6.3 order, re-run
on the same suite, no DAgger round yet."  Section 7 records it item by
item.

On the handoff half of item 5 (the explicit abstention on an isolated
line), the user's decision of 2026-10-28, in their words: "Reading (C):
add the explicit 'not identifiable: line N isolated by breaker M' statement
at handoff, recorded and reported, but never required for the solved
verdict; switch to (B) only if a DAgger round is drawn on compound
classes."  So the statement rides the existing handoff request as a
structured field on compound roots, the evaluator and the audit record and
report it (how often each teacher emits it, and on which roots), no verdict
depends on it, the conventional families are untouched, and the stricter
reading (B), in which the statement is required for a solved verdict on a
root that carries such a line, is a future decision tied to a compound
DAgger round.  Implemented in its turn in the order of section 6.3
(section 7.4).

On adopting the guarded teacher (section 7.1), the user's decision of
2026-10-28, in their words: "Adopt the guarded topology-first teacher as
the default for new work once an independent check on the validation and
hard suites shows no change in success counts; the paper's recorded
results stay as they are."  The check: `baseline_compare` and
`baseline_compare_topology_guarded` on the conventional cell's validation
suite and on the hard draw it evaluated (the suites the budget study's
specs name, neither consulted when the residual floor was set), with the
end-to-end runner of section 7.1, the same code for both teachers, 40
actions, scada_only; it passes only with identical success counts per
family on both suites and no false commit.  On a pass the guarded variant
becomes the default teacher for new work only (what a pipeline env or a
research script picks up when it names none), `baseline_compare` stays
available by name, every recorded cell's env, the paper's evidence folder
and `docs/papers` stay untouched, the tests that pin `baseline_compare`
stay and one is added for the default; on any failing count nothing is
adopted and the differing roots are reported.  The outcome is recorded in
section 7.3: the check failed on one root of the hard draw, so nothing was
adopted and the guarded teacher stays a flagged variant.

After that failure, the user's decision of 2026-10-28, in their words:
"Yes: after item 4, try the clean-flip condition (breaker route first only
when a flip makes the node/breaker estimate clean) and re-run the same
independent check on the validation and hard suites."  So one more item
follows item 4 (section 7.7): a flagged teacher variant whose breaker
route goes first only when a candidate flip makes the node/breaker
estimate clean (its substation chi-square under its limit, not merely
better), evaluated on item 1's protocol (the compound suite under the
baseline, the always-on flag, the flow-at-noise guarded variant and the
new one; the conventional 192-root suite; then the independent check on
the validation and hard suites under the same pass criterion and the same
adoption rule), with the cost of the clean-flip test per episode (the
extra node/breaker solves) reported beside the counts.

The outcome (2026-10-28): the check passed (identical per-family success
on the validation suite and the hard draw, the torch files, no false
commit; section 7.7), so under the user's rule the clean-flip teacher
became the default for new work (commit `b546d2d`: `DEFAULT_EXPERT_VARIANT`
in `psse_env/oracle/expert_variants.py` and the
`EXPERT_VARIANT=${EXPERT_VARIANT:-...}` default in
`research/hpc/full_pipeline_20260907/pipeline.env`), with its provider
screen turned on wherever that teacher was built.  On the compound suite it
recovered fewer roots than the guarded or always-on order (700 against 788
and 789), and its screen cost about 23 node/breaker solves per compound
episode.

The same day the user reversed it, in their words: "Revert the default
teacher to baseline_compare and remove the clean-flip screen entirely; then
test the cue-guarded teacher with tightened flip acceptance and a
flow-scaled cue floor under the same independent check."  Their reasoning,
which governs from now on: the clean-flip screen is exhaustive screening
moved to the node/breaker model, no different in kind from the WLS screen
the triage classifier replaced; the agent must decide when to look, on
observable evidence, and brute force may never be adopted to raise the
success rate.  The same judgement applies to the always-on topology-first
order, an always-look policy: it stays a flagged research variant, never a
default.  So item 6 was removed as brute-force screening (on
`claude/upbeat-driscoll-ff493f`, the merge candidate of the stage-2 review):
the default teacher for new work is `baseline_compare` again, in
`DEFAULT_EXPERT_VARIANT` and in the pipeline default, and the provider
option `clean_flip_screen`, the `clean_flip` report and its ledger field,
the per-state estimate cache that served it, the expert mode and the
variant `baseline_compare_topology_clean_flip`, the runners' node/breaker
cost fields and the mixed-channel guard the stage-2 review had added for
the screen are gone.  Its measured numbers stay in section 7.7, in the
text only, as a brute-force upper bound for the breaker ordering.  Items 1
(the open-terminal cue, reported only when the guarded teacher or the
target-first verifier reads it), 2, 3 and 5 stay, and item 4 stays in the
tree off by default.  The two recorded override envs that name no teacher
(`ieee57_transfer_20260914.env`, `ieee118_balanced_20260925.env`, both from
before the teacher knob existed) pin `EXPERT_VARIANT=baseline` since the
review, so they re-run as recorded under either default.

## 7. Stage 2: the tool repairs, one at a time (2026-10-28)

The ground rules, from the request: the items of section 6.3 in order, one
commit each, every item re-run on the same 1,275 roots under both the
baseline teacher and the flagged topology-first variant before the next
item is started; CPU only, no training, no DAgger round, nothing under
`docs/papers/pesgm2027/`.  Standing constraints: the topology-first order
stays flagged, and its conditioned version has to be validated on the
conventional 192-scenario test suite of the manuscript's cell (the cell's
`development.json`, copied to `output/conventional_reference/`) with the
expert still at 192 of 192 and every conventional family unchanged before
anyone proposes adopting it; `baseline_compare` keeps its default
behaviour; the isolated-line audit rule of section 3.2 is in; every suite
drawn carries the uniform metadata.  Per item the record below gives what
changed and where (flag or shared path, with the tests that pin the seven
conventional families), the per-class before and after under both
teachers, the overall ceilings, what broke, and the commit.

### 7.1 Item 1: the topology-first order conditioned on an open-terminal cue

**The cue.**  A dangling terminal leaves its line carrying no current: both
terminal P-flow meters read noise, the Q-flow meter at the open end reads
noise (the energised end still shows charging), while the operator's model,
which has the breaker closed, routes the pre-fault flow through the line,
so the WLS solve leaves a large P-flow residual on it.  The provider now
computes, for the branch candidates it ranks (the top-k Lagrangian
multipliers of `run_wls`, and the ranked lines of `get_parameter_context`),
whether a line reads like that: both |P| flow readings below
`flow_noise_sigma` (3) standard deviations, the smaller |Q| flow reading
below it, and the larger normalized P-flow residual above
`flow_noise_residual` (8).  The lines that do are reported, 1-based, as
`flow_at_noise_lines` in the WLS tool metrics (durable in the observation's
`fresh_context_evidence["wls"]` beside the alarm fields) and as
`parameter_flow_at_noise_lines` in the parameter context.  Everything is
derived from the measurement vector, the declared sigma and the solve, so
nothing privileged enters the observation; the fields are additive, so no
default route decision changes (`psse_env/providers/test_flow_at_noise.py`,
and the cue on the compound fixtures in
`psse_env/providers/test_compound_scenarios.py`).

**The order.**  `ExpertPolicyOracle(topology_first=...)` now takes three
modes: off (every existing variant), on everywhere
(`baseline_compare_topology_first`, section 3.1), and `"flow_at_noise"`
(the new variant `baseline_compare_topology_guarded`), in which the breaker
route leads the structural tier only in a state whose latest bound WLS
solve reports at least one flow-at-noise line; without the cue the order is
the baseline's, so a root without the cue is handled exactly as
`baseline_compare` handles it.  Nothing is excluded: if the topology route
finds no supported flip, the parameter route follows as before.
`baseline_compare` is untouched (`psse_env/oracle/test_topology_first_order.py`
pins its options and its order).

**Setting the residual floor.**  The cue was measured on the first solve of
every root of both suites (scope: any of the top-k ranked lines), against
whether the root carries a breaker atom (the topology-first order is right)
or not (the parameter-first order is right):

| residual floor (sigma) | compound breaker roots that fire (of 650) | compound roots without a breaker atom that fire (of 625) | conventional topology and measurement+topology roots that fire (of 56) | other conventional roots that fire (of 136) |
| --- | --- | --- | --- | --- |
| 3 | 551 | 62 | 52 | 3 |
| 5 | 525 | 24 | 52 | 2 |
| 8 | 504 | 5 | 51 | 0 |

The floor of 8 was set on the compound suite's roots without a breaker atom
(5 false fires, 4 of them on adjacent-pair roots and 1 on a meter plus two
lines, against 24 and 62 at the lower floors), at the cost of 47 of the 551
breaker roots the loosest floor catches.  The conventional suite was then
read as a check: at 8 no conventional root without a breaker fires (the two
that fire at 5 are a parameter root and a measurement+parameter root whose
true line carries almost no flow, residuals 5.5 and 7.7 sigma), and the
dangling-terminal roots of the topology and mixed families fire on 51 of
56.  The conventional suite would have given the same floor, but the
reader should know it was looked at before the validation run below, so
that run is a check of the chosen floor, not an independent test of the
rule.  Per class at the floor of 8 the cue fires on 58 of 60 two-dangling
roots, 31 of 45 mixed pairs, 13 of 45 two-split roots (a bus split only
reads as an open terminal when a line's end lands on an empty section), 33
of 50 single-breaker controls, 128 of 150 and 114 of 150 of the two
distant R/X plus breaker classes, 58 of 75 same-line and 69 of 75
shared-bus roots; on the conventional topology family 27 of 32 and on
measurement+topology 24 of 24.  The cue is re-evaluated on every solve, so
a root that does not show it at the first state can show it after a meter
or parameter correction changes the solve.  (The counts in this paragraph
are for the cue on the five top-ranked branch candidates, the scope of the
first run; the committed version examines every line, see "Scope of the
cue" below.)

**Validation on the conventional suite.**  The manuscript's 192-root test
suite (`output/conventional_reference/development.json`, the cell's
`development.json`) was rolled out with the standard end-to-end runner
(`research/classifier_triage/expert_e2e.py --profile scada_only`, the
40-action budget and the paired evaluation contract) under the unchanged
`baseline_compare` and under `baseline_compare_topology_guarded`, on the
same code:

| family (roots) | baseline_compare | guarded | identical trajectories | roots with the cue at the first solve | mean actions: baseline, guarded |
| --- | --- | --- | --- | --- | --- |
| measurement (32) | 32 | 32 | 32 | 0 | 7.00, 7.00 |
| measurement+parameter (32) | 32 | 32 | 32 | 0 | 11.22, 11.22 |
| measurement+topology (24) | 24 | 24 | 0 | 24 | 12.00, 11.00 |
| multi_measurement (24) | 24 | 24 | 24 | 0 | 20.58, 20.58 |
| no_error (16) | 16 | 16 | 16 | 0 | 2.00, 2.00 |
| parameter (32) | 32 | 32 | 32 | 0 | 7.56, 7.56 |
| topology (32) | 32 | 32 | 5 | 27 | 8.00, 7.16 |
| all (192) | 192 | 192 | 141 | 51 | 9.87, 9.60 |

Both teachers solve 192 of 192 with no false commit; the baseline's
per-family means reproduce the cell's.  Of the 51 trajectories that
differ, every one is a topology or measurement+topology root on which the
cue fired, and every one is the baseline's trajectory with the
parameter-context fetch removed (51 of 51 are exactly one
action shorter; the breaker fix, the meter fix and the handoff are the
same actions).  No parameter, measurement, mixed measurement+parameter,
multi-measurement or no-error root changed at all.  The topology roots
without the cue keep the baseline's trajectory.  So the guarded order
meets the standing condition (expert 192 of 192, every family unchanged)
and the only effect on the conventional families is one fewer action on
51 of the 56 breaker roots.  This is a check of the chosen floor, as
said above, not an independent test of the rule.

**The compound suite.**  The same 1,275 roots under the three teachers
(the baseline and the always-on variant re-run on this code reproduce
their stage-1 trajectories on 1,275 of 1,275 roots, so the before columns
are the stage-1 results under the isolated-line rule of section 3.2):

| class (roots) | baseline_compare | topology first, always | topology first, guarded | guarded vs baseline, won / lost | roots with the cue at the first solve | mean actions: baseline, always, guarded |
| --- | --- | --- | --- | --- | --- | --- |
| two R/X, adjacent (150) | 50 | 42 | 50 | +0 / -0 | 6 | 16.7, 18.9, 16.7 |
| two R/X, distant (150) | 72 | 63 | 72 | +0 / -0 | 0 | 10.5, 14.2, 10.5 |
| R/X + breaker, same line (75) | 47 | 75 | 71 | +24 / -0 | 58 | 16.6, 7.0, 8.6 |
| R/X + breaker, shared bus (75) | 32 | 43 | 46 | +14 / -0 | 70 | 21.0, 13.5, 13.0 |
| R/X + breaker, distant (150) | 73 | 109 | 109 | +37 / -1 | 129 | 18.1, 12.2, 12.0 |
| meter + two R/X (150) | 61 | 51 | 61 | +0 / -0 | 1 | 16.7, 19.8, 16.7 |
| meter on the R/X line (75) | 14 | 14 | 14 | +0 / -0 | 0 | 14.8, 15.0, 14.8 |
| R/X + breaker + meter (150) | 59 | 97 | 97 | +38 / -0 | 116 | 23.0, 16.3, 16.7 |
| two breakers, dangling (60) | 16 | 53 | 46 | +30 / -0 | 58 | 25.7, 11.0, 12.8 |
| two breakers, mixed (45) | 1 | 11 | 2 | +1 / -0 | 37 | 27.2, 14.0, 18.6 |
| two breakers, bus splits (45) | 0 | 0 | 0 | +0 / -0 | 21 | 26.8, 15.4, 23.1 |
| control: one R/X (50) | 43 | 37 | 43 | +0 / -0 | 0 | 7.5, 10.0, 7.5 |
| control: one breaker (50) | 31 | 50 | 44 | +13 / -0 | 33 | 14.8, 7.0, 10.5 |
| control: one meter (50) | 47 | 47 | 47 | +0 / -0 | 0 | 21.3, 21.4, 21.3 |
| all (1275) | 546 (43%) | 692 (54%) | 702 (55%) | +157 / -1 | 529 | 17.9, 14.7, 14.3 |

The guarded order reaches 702 of 1,275 (55%) against 546 for the
baseline and 692 for the always-on flag: 157 roots won and 1 lost against the
baseline.  On the 625 roots without a breaker atom the guarded teacher
is the baseline (618 of 625 identical trajectories; the cue fired on
7 of them, 2 solved either way), so the always-on flag's 33 losses on
the parameter and meter classes are recovered in full.  On the breaker
classes it tracks the always-on flag where the cue fires and the baseline
where it does not:

| breaker class | cue at the first solve: roots | always | guarded | no cue: roots | baseline | always | guarded |
| --- | --- | --- | --- | --- | --- | --- | --- |
| R/X + breaker, same line | 58 | 58 | 58 | 17 | 13 | 17 | 13 |
| R/X + breaker, shared bus | 70 | 42 | 43 | 5 | 1 | 1 | 3 |
| R/X + breaker, distant | 129 | 96 | 99 | 21 | 10 | 13 | 10 |
| R/X + breaker + meter | 116 | 84 | 85 | 34 | 12 | 13 | 12 |
| two breakers, dangling | 58 | 52 | 46 | 2 | 0 | 1 | 0 |
| two breakers, mixed | 37 | 11 | 2 | 8 | 0 | 0 | 0 |
| two breakers, bus splits | 21 | 0 | 0 | 24 | 0 | 0 | 0 |
| control: one breaker | 33 | 33 | 33 | 17 | 11 | 17 | 11 |

Where the cue fires at the first solve the guarded teacher solves as many
roots as the always-on flag or more, except on the two-breaker classes:
after the first breaker is fixed the second atom has to show the cue on
its own, and a bus split rarely does (a split reads as an open terminal
only when a line's end lands on an empty section), nor does a lightly
loaded dangling line.  The replay of one missed two-dangling root
(`r0_f337fe6e3610`) shows the second: its two open lines read 1.3 and
0.9 sigma of P flow with 33.3 and 2.1 sigma of P residual at the first
solve, so the first is flagged and fixed; at the second solve the
remaining line's residual is 3.7 sigma, below the floor, the parameter
route leads, the estimator inflates that line's reactance and the
verifier accepts the edit as final (section 5.4's masking), and the
breaker is never fixed.  Lowering the floor to 3 would catch it at the
price of 65 false fires on the compound suite's parameter and meter roots
(the table of floors above).  The one root lost against the baseline
(`r0_522d8a1342a4`, R/X + breaker, distant) is a breaker root on which the cue fired
correctly but the topology context's first supported flip was a wrong
breaker that verified as partial progress; the baseline reached the same
breaker only after its parameter correction, when the wrong flip was no
longer offered.  That is the always-on flag's loss mechanism of section
3.1 (38 roots there), reduced to one here because the parameter roots no
longer take the topology route.

**Scope of the cue.**  The first run of the guarded teacher evaluated the
cue on the top five ranked branch candidates only
(`expert_baseline_compare_topology_guarded_top5/`: 702 of 1,275, the
same one root lost).  Evaluating it on every line (the committed version)
adds 18 breaker roots that show the cue at the first solve (504 to 522 of
650; two-breaker mixed pairs 31 to 37, two bus splits 13 to 21) for two
more false fires on adjacent-pair roots (5 to 7 of 625), because the line
a bus split leaves on an empty section and the second terminal of a pair
need not carry a top multiplier; 20 trajectories differ between the two
scopes and no root changes outcome.  The cue is still absent at the first
solve on 128 of the 650 breaker roots: of the suite's 650 dangling
terminals 115 do not show it, 104 because the P residual is below the
floor (lightly loaded lines) and 11 because the P flow is not at noise,
and the other roots without it are bus splits that leave no line on an
empty section.

**What broke.**  The first version of the provider change bound the cue
only on the branch-ranking path, so a resolved or waveform-dominated
solve raised an `UnboundLocalError` inside `run_wls`; the oracle
regression suite caught it (`test_expert_recovery`) before any run was
kept, and every result above is from the fixed code (471 tests pass).
The compound rollout harness cannot roll out the conventional suite,
whose roots name the tracked `case14` rather than a case file (every
re-solve after a correction fails with `dispatch_error`), which is why
the conventional validation uses the end-to-end runner; the harness is
unchanged.

**Commit.**  The commit that adds this section (branch
`claude/nostalgic-kalam-be5f7a`, 2026-10-28).  `baseline_compare` is
untouched and stays the teacher; `baseline_compare_topology_guarded` is a
named variant of the compound work, and adopting it elsewhere is the
user's decision.

### 7.2 Item 2: the node/breaker binding after a structural re-render

**The defect.**  A breaker flip that splits or merges a bus re-renders
the operator model: the candidate's case has operator sections as buses
(15 after one split, 16 after two).  The binding the topology route
builds for every later call took the state's current case as the
electrical reference, and both the node/breaker estimator and the
topology processor (`node_network`, `operator_model_from_map` in
`Transmission/`) look a line up by its planning-bus endpoints, so the
first branch ending on section 15 raised a `KeyError` inside
`get_topology_context` and `correct_topology`, which the environment
reported as `dispatch_error`.  Every root that needs a second structural
flip (the 45 two-split roots, and the mixed pairs whose split comes
second) ended with that breaker unfixable: section 6.2, item 2.

**The repair (shared path, exact for states that never re-rendered).**
The first structural re-render now records the planning-bus case it
rendered from (`planning_reference_case`, admitted through the scada_only
and gated evidence boundaries beside the other node/breaker fields,
`psse_env/evidence_profile.py`).  A state descending from a re-render
builds its binding from that case with the current case's branch
parameters copied row by row (an accepted impedance correction survives)
and every line in service (statuses are derived from the breaker map, as
before), and a second structural flip renders from that reference rather
than from the sectioned case; the record is carried forward.  A state
whose case is not on planning buses and that carries no reference reports
`topology_reference_unavailable` instead of raising.  A state that never
re-rendered takes exactly the path it took before (its own case), so a
single split, the conventional topology family's case, is rendered as it
was.  The test
(`test_two_bus_splits_are_corrected_from_the_planning_reference`,
`psse_env/providers/test_compound_scenarios.py`) fixes both splits of a
two-split root in sequence through the topology context and the
executor, 15 then 16 operator buses, and checks that the solve is clean
afterwards and that the binding's reference is the 14-bus case.

**Results.**  The same 1,275 roots under the three teachers, before (the
item-1 runs of section 7.1) and after the repair, same code otherwise:

| class (roots) | baseline_compare: before, after | topology first, always: before, after | topology first, guarded: before, after | mean actions after: baseline, always, guarded |
| --- | --- | --- | --- | --- |
| two R/X, adjacent (150) | 50, 50 | 42, 42 | 50, 50 | 16.7, 18.8, 16.7 |
| two R/X, distant (150) | 72, 72 | 63, 63 | 72, 72 | 10.5, 14.2, 10.5 |
| R/X + breaker, same line (75) | 47, 47 | 75, 75 | 71, 71 | 16.6, 7.0, 8.6 |
| R/X + breaker, shared bus (75) | 32, 32 | 43, 43 | 46, 46 | 21.0, 13.7, 13.2 |
| R/X + breaker, distant (150) | 73, 73 | 109, 109 | 109, 109 | 18.1, 12.2, 12.0 |
| meter + two R/X (150) | 61, 61 | 51, 51 | 61, 61 | 16.7, 19.8, 16.7 |
| meter on the R/X line (75) | 14, 14 | 14, 14 | 14, 14 | 14.7, 14.9, 14.7 |
| R/X + breaker + meter (150) | 59, 59 | 97, 97 | 97, 97 | 23.0, 16.3, 16.7 |
| two breakers, dangling (60) | 16, 16 | 53, 53 | 46, 46 | 25.7, 10.9, 12.7 |
| two breakers, mixed (45) | 1, 20 | 11, 43 | 2, 30 | 25.6, 11.0, 16.0 |
| two breakers, bus splits (45) | 0, 22 | 0, 39 | 0, 28 | 25.0, 11.3, 20.6 |
| control: one R/X (50) | 43, 43 | 37, 37 | 43, 43 | 7.5, 10.0, 7.5 |
| control: one breaker (50) | 31, 31 | 50, 50 | 44, 44 | 14.6, 7.0, 10.3 |
| control: one meter (50) | 47, 47 | 47, 47 | 47, 47 | 21.3, 21.4, 21.3 |
| all (1,275) | 546 (43%), 587 (46%) | 692 (54%), 763 (60%) | 702 (55%), 758 (59%) | 17.8, 14.5, 14.1 |

Every gain is on the two classes that need a second structural flip, 41,
71 and 56 roots under the three teachers, and no root is lost under any of
them (identical trajectories on 1,221, 1,187 and 1,206 of 1,275; the rest
are roots where a second structural flip was attempted, some of them
without a change of outcome).  The conventional suite is untouched: the
baseline teacher re-run on this code solves 192 of 192 with all 192
trajectories identical to the control of section 7.1 (a single split
renders from the state's own case, as before).

What still fails on the two-split class is the ordering problem of
section 3.1, not the binding: under the baseline teacher 23 of 45 roots
end with the second split unfixed because the parameter route runs first
on the re-rendered state and commits a wrong impedance edit (94 parameter
corrections refused along the way, the accepted ones modifying healthy
lines); the guarded teacher recovers 6 more (28 of 45) but a bus split
shows the open-terminal cue only when a line's end lands on an empty
section, so 17 roots take the parameter route at the second state; the
always-on flag, which asks the topology context again regardless, reaches
39 of 45 (the 6 left end with a wrong edit beside the second split).  The
mixed pairs follow the same pattern (20, 43 and 30 of 45).  So item 2 is
the executor repair it set out to be, and the two-breaker classes are now
limited by the cue's blindness to bus splits and by the parameter route's
masking of a breaker fault (item 4's verifier question).

**Commit.**  The commit that adds this section (branch
`claude/nostalgic-kalam-be5f7a`, 2026-10-28); shared path, exact for every
state that never re-rendered; 482 tests pass.

### 7.3 The independent check of the guarded teacher (2026-10-28): not adopted

The two suites the user named, neither consulted when the residual floor
was set: the conventional cell's validation suite (96 roots, 16/16/12/12/
8/16/16 over measurement, measurement+parameter, measurement+topology,
multi-measurement, no-error, parameter, topology) and the hard draw it
evaluated (193 roots: 24/32/8/72/8/41/8).  Correction of 2026-10-28: these
runs used the local smoke build of the cell's extra suites
(`conventional_only_extra_suites_v1`, built with a 14-root D0 exclusion),
whose hashes match that build's own report, not the files the cell built
on torch (370-root D0 exclusion), as first stated here.  Read back over
SSH, the torch files hold the same 96 and 193 roots with identical
execution payloads (cases, measurement vectors, metadata) and identical
truth; the hard draw differs only in serialization and the validation
suite only in each root's provenance fingerprint and the root order, none
of which the runner's outcome depends on.  The clean-flip check of section
7.7 was re-run on the torch files themselves.  Both teachers ran through
the end-to-end runner of section 7.1 on the same code (item 2's binding
repair included), 40 actions, scada_only.

| suite | family (roots) | baseline_compare | guarded | identical trajectories |
| --- | --- | --- | --- | --- |
| validation | measurement (16) | 16 | 16 | 16 |
| validation | measurement+parameter (16) | 16 | 16 | 16 |
| validation | measurement+topology (12) | 12 | 12 | 4 |
| validation | multi_measurement (12) | 12 | 12 | 12 |
| validation | no_error (8) | 8 | 8 | 8 |
| validation | parameter (16) | 16 | 16 | 16 |
| validation | topology (16) | 16 | 16 | 4 |
| validation | all (96) | 96 | 96 | 76 |
| hard draw | measurement (24) | 24 | 24 | 24 |
| hard draw | measurement+parameter (32) | 32 | 32 | 32 |
| hard draw | measurement+topology (8) | 8 | 8 | 3 |
| hard draw | multi_measurement (72) | 64 | 64 | 72 |
| hard draw | no_error (8) | 8 | 8 | 8 |
| hard draw | parameter (41) | 41 | 40 | 40 |
| hard draw | topology (8) | 8 | 8 | 3 |
| hard draw | all (193) | 185 | 184 | 182 |

No false commit on either suite under either teacher.  On the validation
suite the 20 trajectories that differ are the 8 measurement+topology and
12 topology roots on which the cue fires at the first solve, each the
baseline's trajectory minus the parameter-context fetch; the cue fires on
no root without a breaker atom.  On the hard draw 10 of the 11 differing
trajectories are the same one-action shortening on 5 measurement+topology
and 5 topology roots.  The eleventh is the failure: on the parameter root
`r0_344bb11720e7` (ambiguous stratum, one true error on line 3 at 0.43 of
its nominal impedance) the cue fires at the first solve on line 6, a
lightly loaded healthy line (1.2 sigma of P flow, 2.2 sigma of Q, 22.7
sigma of P residual): the operator's model, with line 3's impedance too
high, routes through line 6 flow that the meters do not see, the very
signature the cue reads as an open terminal.  The guarded teacher fetches
the topology context first, which offers a flip opening line 6's terminal;
the verifier accepts it as partial progress; the parameter correction on
line 3 follows and is accepted, then a meter edit is tried and rolled back,
and the episode hands off with the wrong breaker edit committed (the
baseline compares lines, corrects line 3 and hands off in 8 actions).  So
the hard draw's parameter family goes from 41 to 40 and the check fails on
one count.

Under the user's rule (identical success counts per family on both suites)
nothing is adopted: `baseline` stays the code default, `baseline_compare`
stays the conventional cell's teacher, and `baseline_compare_topology_guarded`
stays a flagged variant of the compound work.  Two readings for later, not
acted on: the cue alone cannot tell an open terminal from a healthy line
starved by a neighbour's impedance error (the compound suite shows the
same on 7 of 625 roots without a breaker atom, 6 of them adjacent pairs),
so a clean-flip condition on the topology side (admit the breaker route
first only when a flip makes the node/breaker estimate clean, not merely
better) or a residual floor that scales with the line's modelled flow
would be the next candidates; and the loss mechanism is again a wrong flip
accepted as partial progress (section 7.1's one lost root), which item 4's
target-first acceptance may also touch.

### 7.4 Item 5: the explicit abstention at handoff (decision C)

**The statement.**  An accepted `correct_topology` that opened a breaker
on a single line terminal (status open, with a `line_index`) de-energised
that line, so its impedance is not identifiable from the snapshot.  With
the expert option `abstain_isolated_lines` every operator handoff the
expert proposes (`ask_for_more_evidence` with an `operator_escalation:*`
request) carries the statements its accepted breaker corrections justify,
as a structured `abstentions` argument: one `{kind:
line_isolated_by_breaker, line_index: N, cb_name: M}` per isolated line,
read from the observation's own `accepted_corrections`
(`psse_env/actions.py`: `isolated_line_abstentions`,
`with_isolated_line_abstentions`).  The environment records the declared
statements in the handoff's tool metrics (`abstentions`), each marked
`supported` when an accepted breaker correction of that lineage isolated
the line; a malformed or unsupported statement is recorded as such and
never a precondition (`abstention_record`).  The truth audit reports it
beside the not-applicable check of section 3.2
(`isolated_line_abstention`: `declared`, `not_declared`, or
`declared_without_isolated_fault`) with no problem attached, so no
verdict depends on it, under either reading of the user's decision the
stricter one (B) would be a one-line change in that check.  The rollout
records the statements per episode and the per-class table counts the
episodes that abstained and the unsupported statements.

**Scope.**  The option is off in every variant, so no existing teacher's
actions change unless a run names it; the compound runs below name it for
all three teachers (`--expert-option abstain_isolated_lines=1`, the
research run options added for stage 2 together with `--env-option`),
and the conventional suite is run once with it on to show that the
statement changes no conventional verdict.  Tests:
`psse_env/test_isolated_line_abstention.py` (the statements, the
attachment, the record), the audit report in
`psse_env/dagger/test_release_audit.py`.

**Results.**  The same 1,275 roots under the three teachers with the
option on (before: the item-2 runs):

| class (roots) | baseline: solved before, after; handoffs with the statement | always: solved before, after; with the statement | guarded: solved before, after; with the statement |
| --- | --- | --- | --- |
| two R/X, adjacent (150) | 50, 50; 1 | 42, 42; 15 | 50, 50; 4 |
| two R/X, distant (150) | 72, 72; 2 | 63, 63; 25 | 72, 72; 2 |
| R/X + breaker, same line (75) | 47, 47; 49 | 75, 75; 75 | 71, 71; 71 |
| R/X + breaker, shared bus (75) | 32, 32; 47 | 43, 43; 72 | 46, 46; 70 |
| R/X + breaker, distant (150) | 73, 73; 95 | 109, 109; 147 | 109, 109; 142 |
| meter + two R/X (150) | 61, 61; 1 | 51, 51; 16 | 61, 61; 1 |
| meter on the R/X line (75) | 14, 14; 3 | 14, 14; 4 | 14, 14; 3 |
| R/X + breaker + meter (150) | 59, 59; 90 | 97, 97; 139 | 97, 97; 135 |
| two breakers, dangling (60) | 16, 16; 41 | 53, 53; 57 | 46, 46; 56 |
| two breakers, mixed (45) | 20, 20; 37 | 43, 43; 45 | 30, 30; 41 |
| two breakers, bus splits (45) | 22, 22; 0 | 39, 39; 1 | 28, 28; 1 |
| control: one R/X (50) | 43, 43; 0 | 37, 37; 6 | 43, 43; 0 |
| control: one breaker (50) | 31, 31; 23 | 50, 50; 35 | 44, 44; 35 |
| control: one meter (50) | 47, 47; 0 | 47, 47; 0 | 47, 47; 0 |
| all (1,275) | 587, 587; 389 | 763, 763; 637 | 758, 758; 561 |

As the decision requires, nothing else moves: every teacher's tool
sequence is identical on 1,275 of 1,275 roots and no outcome flips.  No
statement is unsupported.  On the 84 roots that carry an isolated line
(section 3.2) every solved root hands off with the statement (51 of 51,
83 of 83 and 79 of 79 under the three teachers), so under reading (B) the
same roots would stay solved.  The statements on roots without an
isolated truth line (the adjacent pairs: 1, 15 and 4; the distant pairs:
2, 25 and 2; and so on) are truthful about what the agent did, a wrong
flip that opened a healthy terminal, and the audit reports them as
`declared_without_isolated_fault`; their count per class is a direct
reading of the wrong-flip mechanism of sections 3.1 and 7.3 (the
always-on flag's 15 on adjacent pairs against the guarded teacher's 4).
On a bus split no statement is made, because no single line is isolated.
The conventional suite with the option on, baseline teacher: 192 of 192,
all 192 tool sequences identical to the control; the statement rides the
handoff on the 24 measurement+topology roots and on the 25 topology roots
whose flip opened a line terminal (the 7 bus-split roots carry none).
The end-to-end runner's compact record now carries the statements too.

**Commit.**  The commit that adds this section (branch
`claude/nostalgic-kalam-be5f7a`, 2026-10-28); 516 tests pass.

### 7.5 Item 3: one alternating pair refit inside the second correction of an adjacent pair

**The defect.**  On an adjacent pair the one-shot single-line estimate of
each line is biased by the other's error: the first line's refit on the
stale model is within tolerance on 73 of 126 adjacent roots (median
reactance error 26%), the second's on 65 of 129, and one more pass over
the first line on the model that carries the second line's estimate
brings it to 112 of 131 (6%), section 5.3.  The stage-1 probe also showed
where that pass cannot run: on the gated environment a third correction
after the second commit is refused on 77 of 150 roots
(`post_correction_confirmation_required` 67, `correction_not_supported`
10), because the solve is clean or the context no longer supports the
line, so the expert cannot schedule the pass as an action of its own.

**The repair (provider option `pair_refit_adjacent`, off by default).**
Inside `correct_parameters`, when the state's last committed parameter
correction (`last_parameter_correction` in the metadata, now admitted
through the evidence boundaries) is on a line that shares a bus with the
line being corrected, the executor runs one alternating pass after the
single-line fit: the earlier line is refit on the model that carries this
line's estimate, then this line again; both rows go into the candidate,
the output reports `pair_refit` (the partner line and both refits) and the
final values, and the metadata records the partner.  A failed pass leaves
the single-line estimate and is reported as skipped.  Nothing changes for
a line without an adjacent committed predecessor, so a root with one
parameter error, the conventional families among them, is handled exactly
as before (`MatpowerDeploymentProviders(pair_refit_adjacent=False)` is
the default; the research runs name it with `--env-option
pair_refit_adjacent=1`).  Tests: the pass on the adjacent fixture and the
partner rule (`psse_env/providers/test_compound_scenarios.py`).

**What the first run broke.**  The candidate-quality oracle's collateral
check held every parameter correction to exactly one changed branch row,
the declared target, so the pair-refit candidate (two rows changed) was
labelled a healthy-component corruption; and because that verdict carried
none of the rejection evidence the environment requires of a verified
candidate (`rejection_evidence_missing`), the verification solve itself
was refused as insufficient evidence and the expert looped on it until
the budget ran out (a six-root smoke: a solved root lost, 29 actions a
root).  The oracle now reads the executor's declaration, the partner line
recorded in the candidate's `last_parameter_correction`, and accepts a
change of that row's impedance as the correction's own work; an
undeclared second row, a partner that only the declaration names, a
status change on the partner, or an unchanged target are collateral as
before (`psse_env/oracle/test_pair_refit_collateral.py`).  The gap this
exposed, a collateral rejection that the evidence check cannot see, is
pre-existing and left as it is.

**Results.**  The same 1,275 roots under the three teachers, before (the
item-5 runs, which equal the item-2 outcomes) and with the pair refit on:

| class (roots) | baseline_compare: before, after | topology first, always: before, after | topology first, guarded: before, after | mean actions after: baseline, always, guarded |
| --- | --- | --- | --- | --- |
| two R/X, adjacent (150) | 50, 76 (+27 / -1) | 42, 65 (+24 / -1) | 50, 76 (+27 / -1) | 14.1, 16.2, 14.2 |
| two R/X, distant (150) | 72, 72 | 63, 63 | 72, 72 | 10.5, 14.2, 10.5 |
| R/X + breaker, same line (75) | 47, 47 | 75, 75 | 71, 71 | 15.8, 7.0, 8.6 |
| R/X + breaker, shared bus (75) | 32, 31 (-1) | 43, 43 | 46, 45 (-1) | 20.4, 13.7, 13.1 |
| R/X + breaker, distant (150) | 73, 73 | 109, 109 | 109, 109 | 17.9, 12.2, 12.0 |
| meter + two R/X (150) | 61, 66 (+5) | 51, 54 (+3) | 61, 66 (+5) | 16.1, 19.1, 16.1 |
| meter on the R/X line (75) | 14, 14 | 14, 14 | 14, 14 | 14.7, 14.9, 14.7 |
| R/X + breaker + meter (150) | 59, 59 | 97, 97 | 97, 97 | 22.9, 16.3, 16.7 |
| two breakers, dangling (60) | 16, 16 | 53, 53 | 46, 46 | 25.9, 10.9, 12.7 |
| two breakers, mixed (45) | 20, 20 | 43, 43 | 30, 30 | 25.7, 11.0, 16.0 |
| two breakers, bus splits (45) | 22, 22 | 39, 39 | 28, 28 | 25.0, 11.3, 20.6 |
| control: one R/X (50) | 43, 43 | 37, 37 | 43, 43 | 7.5, 10.0, 7.5 |
| control: one breaker (50) | 31, 31 | 50, 50 | 44, 44 | 14.6, 7.0, 10.3 |
| control: one meter (50) | 47, 47 | 47, 47 | 47, 47 | 21.3, 21.4, 21.3 |
| all (1,275) | 587 (46%), 617 (48%) (+32 / -2) | 763 (60%), 789 (62%) (+27 / -1) | 758 (59%), 788 (62%) (+32 / -2) | 17.3, 14.1, 13.7 |

The pass moves the adjacent-pair class from a third to a half (50 to 76
of 150 under the baseline and guarded teachers, 42 to 65 under the
always-on flag) and picks up 5, 3 and 5 roots of the meter plus two
lines class whose two lines happen to share a bus; every other class is
unchanged (identical trajectories on 1,130, 1,205 and 1,186 roots, the
rest being roots where a pass ran without changing the outcome).  Two
roots are lost.  One adjacent pair (`r0_f9fefc01d3f2`, under all three
teachers): the first line is accepted, the second true line's correction,
now carrying the pass, is rejected, and the three wrong lines tried
afterwards are rejected too, so the episode ends with one line
uncorrected.  One shared-bus root under the baseline and another under
the guarded teacher: the breaker-affected line and a healthy neighbour of
the last committed line form the pair, and the pass hands the verifier a
wrong edit it accepts.  The conventional suite with the option on,
baseline teacher: 192 of 192, all 192 tool sequences identical (no
conventional root carries two line errors, so the pass never runs).

What still fails on the adjacent class (74 of 150 under the baseline and
guarded teachers): on 73 the final case is outside tolerance, and on 56
of those one true line is never corrected, which is the pattern of the
stage-1 probe where the first estimate lands too far from the truth for
one pass to recover (median reactance error 26% before the pass), or the
second line's correction is rejected on the first line's biased model.  A
second sweep, or a joint fit of both lines' R and X in one multi-scan
problem, is the next step on this class; the executor and the oracle now
carry the declaration such a candidate needs.

**Commit.**  The commit that adds this section (branch
`claude/nostalgic-kalam-be5f7a`, 2026-10-28); provider option, off by
default; 521 tests pass.

### 7.6 Item 4: target-first partial acceptance

**The defect.**  With several faults open, a true correction's own target
test clears while the global statistic moves little, and the verifier's
partial-progress floor (20% of the parent anomaly for a meter, 30% for a
branch) refuses it: section 4 counted 75 true corrections rejected at two
open faults, 43 of them below the floor, against wrong corrections whose
target test does not clear.  A floor scaled by the number of open
suspects would admit 106 wrong corrections for 24 true ones (section
6.2, item 4), so the recommendation was target-local: keep a correction
whose own target metric clears and whose global progress is not negative,
whatever the floor, and keep the floor for corrections whose target does
not clear.

**The repair (verifier option `target_first_partial`, off by default).**
`CandidateQualityOracle(target_first_partial=True)` adds one route to the
branch of its decision where the target is fixed, the solve is not
resolved and global progress is below the floor: a parameter correction
with global progress at or above zero is kept as `ACCEPT_PARTIAL`
(`target_first_partial`).  Parameters only: the first version admitted
meters and breakers too, and on the conventional suite it cost three
multi-meter roots (192 to 189), because a breaker edit's "target fixed"
is the requested status itself, so the topology screening began to
advertise wrong flips that the expert then took and could not roll back
cleanly; a meter edit likewise zeroes its own residual by construction.
For those two families the floor stays.  The existing branch-first waiver
keeps precedence where it applies; a negative global progress or an
unfixed target is refused as before, and every other disposition is
untouched.  The deployment provider passes the option
through (`MatpowerDeploymentProviders(target_first_partial=...)`), the
research factory reads it from the run options (`--env-option
target_first_partial=1`), and nothing changes unless a run names it.
Tests: `psse_env/oracle/test_target_first_partial.py`.  The runs below
name it on top of items 3 and 5, so the before columns are the item-3
runs.

**Results, the rule as recommended.**  A clear loss, and nothing won:

| class (roots) | baseline_compare: before, after | topology first, always: before, after | topology first, guarded: before, after |
| --- | --- | --- | --- |
| two R/X, adjacent (150) | 76, 76 | 65, 65 | 76, 76 |
| two R/X, distant (150) | 72, 72 | 63, 63 | 72, 72 |
| R/X + breaker, same line (75) | 47, 21 | 75, 75 | 71, 63 |
| R/X + breaker, shared bus (75) | 31, 11 | 43, 43 | 45, 44 |
| R/X + breaker, distant (150) | 73, 30 | 109, 109 | 109, 102 |
| meter + two R/X (150) | 66, 63 | 54, 52 | 66, 63 |
| meter on the R/X line (75) | 14, 14 | 14, 14 | 14, 14 |
| R/X + breaker + meter (150) | 59, 30 | 97, 96 | 97, 90 |
| two breakers, dangling (60) | 16, 7 | 53, 53 | 46, 42 |
| two breakers, mixed (45) | 20, 19 | 43, 43 | 30, 30 |
| two breakers, bus splits (45) | 22, 20 | 39, 39 | 28, 26 |
| control: one R/X (50) | 43, 43 | 37, 37 | 43, 43 |
| control: one breaker (50) | 31, 20 | 50, 50 | 44, 39 |
| control: one meter (50) | 47, 44 | 47, 44 | 47, 44 |
| all (1,275) | 617 (48%), 470 (37%), 147 lost | 789 (62%), 783 (61%), 6 lost | 788 (62%), 748 (59%), 40 lost |

Every loss is on a root that still carries a breaker fault (or a meter
fault, the three one-meter controls) when the parameter route runs: there
a neighbouring line's refit does clear that line's own multiplier while
the fault stands, which is exactly the masking the floor was guarding
against.  Under the baseline teacher the wrong parameter corrections
accepted as partial progress go from 366 to 641 across the suite (true
ones 515 to 554), and the roots that end with the breaker never fixed
follow; the always-on flag, which fixes the breaker before any parameter
correction, loses only 6 (meter roots), and the guarded teacher 40, on
the breaker roots whose cue did not fire.  The stage-1 reading that "a
wrong line's target test does not clear" held for the parameter and
meter classes of section 4, not for lines next to an open terminal.  So
the rule as recommended is not kept; the option stays available, off, for
the record.

**A refinement, measured after the loss.**  The verified solve carries
item 1's open-terminal cue, so the route can be closed while that cue
stands on any line (`flow_at_noise_lines` non-empty): a wrong refit next
to an open terminal does not clear the cue, a true refit on a settled
root never sees one.  It does not rescue the rule:

| teacher | item 3 (before) | rule as recommended | refined rule |
| --- | --- | --- | --- |
| baseline_compare | 617 (48%) | 470 (37%) | 549 (43%) |
| topology first, always | 789 (62%) | 783 (61%) | 783 (61%) |
| topology first, guarded | 788 (62%) | 748 (59%) | 748 (59%) |

Nothing is won under any teacher with either rule.  The refinement halves
the baseline teacher's loss (147 to 68 roots) but leaves the guarded
teacher's 40 and the always-on flag's 6 unchanged: a wrong refit next to an
open terminal lowers that line's residual below the cue's floor, so the
verified solve no longer shows the cue and the route reopens (the guarded
teacher's 40 losses are the same roots under both rules, 1,273 of 1,275
trajectories identical).  The conventional suite is unchanged under either
rule (192 of 192, all 192 tool sequences identical, baseline teacher).  So
item 4 ends as a negative result: target-local acceptance is not safe while
an unresolved breaker fault can bias a neighbouring line's estimate, and
the option stays in the code, off, with the refinement and the
parameter-only restriction, for the record.  What section 4 measured on
true corrections refused below the floor (43 at two open faults) is
better addressed by fixing the structural atom first, which items 1, 2
and 6 do.

### 7.7 Item 6: the clean-flip condition (removed as brute-force screening)

**Removed (2026-10-28).**  The user removed this item the day it was
adopted (section 6.4 gives the decision in their words): the screen ran the
node/breaker estimate and its top-ranked flip re-estimates on every
anomalous solve of every state that carries the channel, whatever the
evidence, which is exhaustive screening moved to the node/breaker model,
and brute force may never be adopted to raise the success rate.  The code
is gone.  This section records the item as it was run, and its numbers
stand only as a brute-force upper bound for the breaker ordering: what
putting the breaker route first on a clean flip buys when the flip test is
computed everywhere, not a teacher that may be adopted.

**The condition.**  On an anomalous solve of a state that carries the
node/breaker channel, the provider (option `clean_flip_screen`, off by
default; the research runs named it with `--env-option
clean_flip_screen=1`) ran the node/breaker estimate on the reported
breaker map, re-estimated with each of the top-ranked breaker flips (the
same `breaker_candidate_count` the topology context screens), and reported
in the WLS metrics (`clean_flip`, kept in the observation's durable WLS
ledger) whether any flip brought the substation chi-square under its limit
(`available`, the breakers that did, the solves spent).  The estimate and
its flip screens were computed once per state (keyed on the case, the
measurement vector and the reported map), so a topology context fetched on
the same state reused them: the screen's own cost was the solves it spent
on states where no context followed.  The expert mode
`topology_first="clean_flip"` (variant `baseline_compare_topology_clean_flip`)
put the breaker route first only in a state whose latest bound solve
reported such a flip; everything else was the baseline's order.  Its tests
(the screen on a single-breaker fixture and its reuse by the context, the
mode and the variant) went with the code.  The rollout recorded the
screens, their solves and the topology-context fetches per episode, so the
cost below was read directly from the runs.

**Results on the conventional suites (the independent check).**  Both
teachers on the same code (items 1 to 6 in the tree), the end-to-end runner
of section 7.1, 40 actions, scada_only; the validation suite and the hard
draw are the files the cell built on torch (read back over SSH, section
7.3), and the conventional test suite's file matches torch's byte for byte:

| suite | baseline_compare | clean flip | identical trajectories | false commits | clean flip on a root without a breaker fault | node/breaker solves per episode: baseline, clean flip |
| --- | --- | --- | --- | --- | --- | --- |
| conventional test suite (192) | 192 | 192 | 159 | 0 | 0 | 2.6, 2.7 |
| validation suite, torch file (96) | 96 | 96 | 79 | 0 | 0 | 2.6, 2.7 |
| hard draw, torch file (193) | 185 | 185 | 184 | 0 | 0 | 0.7, 0.8 |

The success counts are identical family by family on all three suites (the
validation suite 96 of 96 for both teachers; the hard draw 24/24, 32/32,
8/8, 64/72, 8/8, 41/41 and 8/8 for both), with no false commit, so the
check passes.  The trajectories that differ are the topology roots (every
one: the parameter-context fetch is skipped, 8.0 to 7.0 actions) and one
measurement+topology root per suite (a clean flip on the second state
after the meter fix).  On a measurement+topology root the meter error
stays in the substation telemetry, so no single flip makes the estimate
clean at the first state and the order stays the baseline's; and the
parameter root of the hard draw that sank the guarded teacher
(`r0_344bb11720e7`) shows no clean flip.  The same runs on the local
smoke copies gave identical trajectories (96 of 96 and 193 of 193 for
both teachers), as the root-by-root comparison of section 7.3 predicted.

**Results on the compound suite (a brute-force upper bound for the breaker
ordering).**  The clean-flip teacher with the item-3 and item-5 options
(pair refit, abstention) against the item-3 runs of the other three
teachers, and an instrumented re-run of the baseline for its node/breaker
solves (its trajectories equal the item-3 baseline's on 1,275 of 1,275
roots):

| class (roots) | baseline_compare | topology first, always | topology first, guarded | topology first, clean flip | clean flip against the baseline, won / lost | node/breaker solves per episode: baseline, clean flip | mean actions: baseline, clean flip |
| --- | --- | --- | --- | --- | --- | --- | --- |
| two R/X, adjacent (150) | 76 | 65 | 76 | 76 | +0 / -0 | 1.6, 22.9 | 14.1, 14.1 |
| two R/X, distant (150) | 72 | 63 | 72 | 72 | +0 / -0 | 0.6, 15.4 | 10.5, 10.5 |
| R/X + breaker, same line (75) | 47 | 75 | 71 | 75 | +28 / -0 | 6.1, 9.0 | 15.8, 7.0 |
| R/X + breaker, shared bus (75) | 31 | 43 | 45 | 33 | +2 / -0 | 6.8, 24.7 | 20.4, 19.5 |
| R/X + breaker, distant (150) | 73 | 109 | 109 | 90 | +17 / -0 | 6.2, 18.5 | 17.9, 14.7 |
| meter + two R/X (150) | 66 | 54 | 66 | 66 | +0 / -0 | 1.7, 27.9 | 16.1, 16.1 |
| meter on the R/X line (75) | 14 | 14 | 14 | 14 | +0 / -0 | 5.3, 28.4 | 14.7, 14.7 |
| R/X + breaker + meter (150) | 59 | 97 | 97 | 60 | +1 / -0 | 6.7, 31.0 | 22.9, 22.3 |
| two breakers, dangling (60) | 16 | 53 | 46 | 25 | +9 / -0 | 9.9, 29.4 | 25.9, 22.5 |
| two breakers, mixed (45) | 20 | 43 | 30 | 27 | +7 / -0 | 14.8, 27.2 | 25.7, 23.7 |
| two breakers, bus splits (45) | 22 | 39 | 28 | 22 | +0 / -0 | 13.8, 32.8 | 25.0, 24.6 |
| control: one R/X (50) | 43 | 37 | 43 | 43 | +0 / -0 | 0.2, 9.5 | 7.5, 7.5 |
| control: one breaker (50) | 31 | 50 | 44 | 50 | +19 / -0 | 6.8, 9.0 | 14.6, 7.0 |
| control: one meter (50) | 47 | 47 | 47 | 47 | +0 / -0 | 7.9, 41.8 | 21.3, 21.3 |
| all (1,275) | 617 (48%) | 789 (62%) | 788 (62%) | 700 (55%) | +83 / -0 | 5.1, 23.1 | 17.3, 15.7 |

The clean-flip teacher wins 83 roots against the baseline and loses
none, but stays well below the always-on and guarded orders (700 against
789 and 788; against them +45 / -134 and +28 / -116).  It does what the
always-on flag does where the breaker is the only fault the substation
sees (the same-line class and the single-breaker control at 75 of 75 and
50 of 50), and much less where a second fault contaminates the substation
estimate, so that no single flip makes it clean (R/X + breaker distant 90
against 109, R/X + breaker + meter 60 against 97, two bus splits 22 as
the baseline).  It is the most conservative of the three orders: on the
625 roots without a breaker atom it reported a clean flip in
32 episodes, always after a line correction had left a moderate residual
that re-partitioning one substation happened to fit, and none of them
changed an outcome (605 of 625 trajectories equal the baseline's).

**Cost.**  The screen ran the node/breaker estimate and up to
8 flip re-estimates (9 solves) on every anomalous solve of a state that
carries the node/breaker channel, and the topology context on the same
state reused them.  On the conventional suites only the breaker families
carry the channel: 9 solves per breaker episode under both teachers, an
extra 0.4 to 1.1 per episode on the measurement+topology family, nothing
elsewhere (0.1 per episode over a whole suite).  On the compound suite every
root carries the channel (the uniform-metadata convention), so the screen
ran on every anomalous state of every root: 23.1 node/breaker solves per
episode against the baseline's 5.1, from 9 on a single-breaker root to about
40 on a meter root, and the rollout of the 1,275 roots took 1578 s against
843 s on the same 14 workers (the clean-flip run shared the machine with
other jobs, so the time ratio is an upper bound).

**Commit.**  Added by `6ff5293` (branch `claude/nostalgic-kalam-be5f7a`,
2026-10-28) and made the default by `b546d2d`; removed, with the default
back on `baseline_compare`, by the commit that records the removal on
`claude/upbeat-driscoll-ff493f` (section 6.4).

### 7.8 Stage 2 in sum

Roots solved of the 1,275, under the teachers as they stood after each
item (items in the order they were done; every run on the same suite and
seed, scada_only, 40 actions):

| after | baseline_compare | topology first, always | topology first, guarded | topology first, clean flip (removed; brute-force bound) |
| --- | --- | --- | --- | --- |
| stage 1 (with the isolated-line rule, section 3.2) | 546 (43%) | 692 (54%) | | |
| item 1: the open-terminal cue, guarded variant | 546 | 692 | 702 (55%) | |
| item 2: the node/breaker binding after a re-render | 587 (46%) | 763 (60%) | 758 (59%) | |
| item 5: the abstention statement (outcomes unchanged) | 587 | 763 | 758 | |
| item 3: one alternating pair refit (option) | 617 (48%) | 789 (62%) | 788 (62%) | |
| item 4: target-first acceptance (negative, left off) | 470 or 549 | 783 | 748 | |
| item 6: the clean-flip condition | 617 | 789 | 788 | 700 (55%) |

So the stage moved the baseline teacher's ceiling from 546 to 617 with
two shared or optional tool repairs (the binding repair, 41 roots; the
pair refit, 30) and the topology-first orders from 692 to 788 and 789;
the always-on order is still the strongest on this suite, and the guarded
order matches it within one root and loses none against the baseline
here.  The clean-flip order passed the independent check on the
conventional suites and was the default teacher for new work for part of a
day, then was removed as brute-force screening (section 6.4, the user's
decision in their words): a teacher decides when to look from observable
evidence, and brute force may never be adopted to raise the success rate.
Its 700 stays as a brute-force upper bound for the breaker ordering, the
always-on order (an always-look policy) stays a flagged research variant,
never a default, and the default teacher for new work is
`baseline_compare`.  Every item kept the conventional 192-root suite at
192 of 192 under the baseline teacher.  What is left,
by class: two bus splits and mixed pairs depend on the second structural
atom being routed first (the cue is blind to a split, the clean flip to a
contaminated substation); the adjacent pairs need a second sweep or a
joint two-line fit; the meter on the faulted line stays a verifier
limit; and nothing in stage 2 changes the identifiability limits of
section 5.  No DAgger round has been drawn on compound classes.

## 8. Item 7: hypothesis confidence, ambiguity and recovery (2026-10-28)

After the clean-flip screen was removed as brute-force screening (sections
6.4 and 7.7), the user set the rules for what follows, in their words:
"On lightly loaded lines, the power flow itself is small, making it
difficult to distinguish between parameter errors and topology errors.
However, if correcting for a topology error successfully lowers the
objective function value, we should still treat it as a viable option
rather than strictly requiring it to drop below the threshold... If, after
several correction cycles, the agent ultimately finds that it cannot
resolve the issue, it should proactively roll back its changes and explore
other error types... An alternative approach is to have the agent assign
confidence scores to different error types during classification. If it
cannot clearly distinguish between a parameter error and a topology
error, it could either hand off and report the case directly, or attempt
one hypothesis first and switch to the other upon failure... you cannot
try to fix every failure scenario. If that scenario is impossible to
succeed under our research goal, just report it as a failure."  So no
tightened flip acceptance, no rule whose purpose is to rescue a named
scenario, and the always-on order stays a labelled always-look reference,
never a candidate.  Item 7 has five steps, one commit each: a cue floor
scaled by the modelled flow (8.1), a parameter-versus-topology confidence
computed from what the solve already produced (8.2), an ambiguity
statement at handoff (8.3), the two recovery contracts, designed and not
built (8.4), and the measurement (8.5).

### 8.1 Step 1: a cue floor scaled by the modelled flow (a negative result; the fixed floor stays)

The request: replace the open-terminal cue's fixed 8-sigma residual floor
by one that scales with the line's modelled flow, keep the 3-sigma
flow-at-noise part, set it on the IEEE 14 compound suite's 625 roots
without a breaker fault as before, and check it on the IEEE 118 compound
suite (the pensive-darwin session's 625 roots, none with a breaker fault,
where the fixed floor fires on 54).

The modelled flow costs nothing: a line's P flow in the operator's model
at the estimate is the measurement minus the raw residual the solve
already returns.  `research/compound_errors/cue_floor.py` records, at
every root's first solve and for every line whose flows read at noise, the
normalized residual, the modelled flow, the line's branch multiplier and
its rank, and scores the floors (output in
`output/compound_errors_20261027/cue_floor/`):

| floor on the larger normalized P residual | IEEE 14: false fires (of 625) | IEEE 14: open lines caught (of 639 at noise) | IEEE 118: false fires (of 625) |
| --- | --- | --- | --- |
| fixed 3 | 108 | 612 | 161 |
| fixed 5 | 30 | 570 | 99 |
| fixed 8 (the cue today) | 7 | 535 | 54 |
| fixed 10 | 5 | 471 | 34 |
| max(8, 0.9 x modelled flow) | 7 | 535 | 53 |
| max(8, 1.0 x modelled flow) | 7 | 524 | 47 |
| max(8, 1.05 x modelled flow) | 5 | 442 | 41 |
| max(8, 1.1 x modelled flow) | 4 | 323 | 35 |
| max(8, 1.2 x modelled flow) | 3 | 66 | 22 |
| max(6, 1.0 x modelled flow) | 12 | 545 | 63 |
| max(10, 1.0 x modelled flow) | 5 | 462 | 31 |

On a line whose meters read at noise, the normalized residual is the
modelled flow divided by the square root of one minus the meter's
leverage: 1.02 to 1.25 times the modelled flow for the open lines that
fire (5th to 95th percentile), and 0.94 to 1.45 times for the IEEE 118
false fires.  A healthy, lightly loaded line that the model assigns a
neighbour's flow (the hard draw's `r0_344bb11720e7`, section 7.3) and an
open terminal both read "no current where the model routes some", in
proportion to the modelled flow.  So a floor that scales with the
modelled flow tests the meter's leverage, not the line's state: a factor
below 1 changes nothing (0.9 is the fixed floor within one root on both
systems), and any factor that removes false fires removes open lines at a
similar or higher rate (1.1: IEEE 14 false fires 7 to 4, open lines 535 to
323; IEEE 118 54 to 35).  Set on IEEE 14 as before, no scaled floor has
fewer false fires there without losing open lines, so the fixed 8-sigma
floor stays and the code is not changed: the cue false-fires on 7 of 625
IEEE 14 roots (1.1%) and on 54 of 625 IEEE 118 roots (8.6%).

What does separate the two is the cued line's standing among the branch
candidates, which the solve also produced already.  The open lines that
fire carry a branch multiplier at a median 1.0 of the strongest
candidate's (rank 1 on 387 of 535); the IEEE 118 false fires sit at a
median 0.36 (rank 1 on 1 of 67), the IEEE 14 ones at 0.47 to 0.89.
Requiring the cued line to carry at least half the top multiplier would
leave 6 false fires of 625 on IEEE 14 and 19 on IEEE 118 (496 open lines
caught); at 0.8 of the top, 2 and 11 (438).  That is evidence for the
confidence of step 2 rather than a floor, and step 2 uses it.

**Commit.**  The commit that adds this section (branch
`claude/nostalgic-kalam-be5f7a`, 2026-10-28): the analysis script and
this section; no provider or teacher code changes.

### 8.2 Step 2: a parameter-versus-open-terminal confidence from the solve

**The confidence.**  Computed by the provider on every alarming WLS solve
(`route_confidence` in the WLS metrics, kept in the durable WLS ledger),
only when a run reads the cue (`flow_at_noise_cue`, which only the guarded
teacher turns on, so every other teacher's student observation stays as
it was), and only from quantities the solve already produced: no extra
solve.  For each of the five top-ranked branch candidates and every line
showing the open-terminal cue:

- a line whose measured P flow is above the noise (3 sigma) carries
  current and cannot be open: topology 0 (`measurable_flow`);
- a line whose flows read at noise with the cue's residual (above the
  8-sigma floor of 8.1) gets, as its topology confidence, its branch
  multiplier relative to the strongest candidate's (`open_terminal_cue`),
  the quantity 8.1 found to separate open terminals from false fires;
- a line whose flows read at noise without that residual, a lightly
  loaded line, gets half of it (`flow_at_noise_without_cue`): the solve
  cannot tell an open terminal from an impedance error there, which is
  the user's lightly loaded case.

The line with the highest topology confidence is reported (else the
strongest candidate with 0), with parameter as the complement, the
multiplier ratio, the line's flow and residual in sigmas and whether the
solve is branch-dominant.  When the agent asks for the topology context
on a state with the node/breaker channel, the context reports the same
confidence refined by its own breaker multipliers: a screened flip of a
breaker on that line that lowers the substation objective (the user's
rule: lowering it is a viable step, no threshold) raises the topology
confidence to that breaker's multiplier relative to the strongest
breaker's; a flip that does not, or no breaker of that line among the
screened ones, leaves it and is reported as evidence.  Tests:
`psse_env/providers/test_route_confidence.py`, the gating and the
refinement on the compound fixtures in
`psse_env/providers/test_compound_scenarios.py`.

**The guarded teacher uses it.**  `baseline_compare_topology_guarded` now
puts the breaker route first when the confidence names the open terminal
as the more likely hypothesis (topology above 0.5), and the bare cue only
on an observation without a confidence
(`psse_env/oracle/test_topology_first_order.py`).  The 0.5 is the
decision boundary between the two hypotheses, not a tuned threshold.

**What it changes at the first solve** (the 1,900 roots of 8.1; the
measurement in closed loop is step 5):

| roots | the bare cue turns the order | confidence above 0.5 | confidence between 0.25 and 0.75 |
| --- | --- | --- | --- |
| IEEE 14 compound, without a breaker fault (625) | 7 | 6 | 75 |
| IEEE 14 compound, with a breaker fault (650) | 522 | 489 | 125 |
| IEEE 118 compound, none with a breaker fault (625) | 54 | 19 | 42 |

So the confidence cuts the IEEE 118 false topology-first routes by about
two thirds (54 to 19), at the price of 33 IEEE 14 breaker roots on which
the cue fired on a line that is not the leading candidate (two bus splits
21 to 14, mixed pairs 37 to 32, R/X + breaker distant 129 to 119,
R/X + breaker + meter 116 to 108; the same-line, shared-bus, two-dangling
and single-breaker classes stay within two roots).  Those are cases where
the solve's evidence for the open terminal is weaker than for an impedance
error elsewhere; whether the episode recovers is step 5's measurement, not
something this step tunes.

**Commit.**  The commit that adds this section (branch
`claude/nostalgic-kalam-be5f7a`, 2026-10-28).

### 8.3 Step 3: the ambiguity statement at handoff

**The statement.**  When the teacher hands off with the alarm still
standing and the latest solve cannot clearly tell an impedance error from
an open terminal on the leading line, the handoff states "ambiguous: line
N, parameter or breaker" in the structured field item 5 introduced
(`abstentions`, kind `ambiguous_parameter_or_breaker`, beside any
isolated-line statement): the line, the topology and parameter
confidences, the confidence's basis, the breaker evidence when the agent
fetched the topology context on that state, and the hypotheses the episode
already tried on that line (from its committed and rejected corrections).
"Cannot clearly tell" is a topology confidence between 0.25 and 0.75:
neither hypothesis has three-to-one support.  The confidence is step 2's,
refined by the topology context when that context names the same line
(`psse_env/actions.py`: `ambiguity_statements`,
`with_ambiguity_statements`, `with_handoff_statements`).  A handoff is
the teacher's own statement that its recovery options are exhausted, so
the statement rides every operator handoff that meets the condition.

**Recorded and reported, never required.**  The environment records the
statement in the handoff's tool metrics, marked supported when the latest
solve's confidence names the same line inside the band; the truth audit
reports it in its own check (`ambiguity_statement`) with what the hidden
truth holds on that line (impedance error, open terminal, both or
neither), never as a problem, and the isolated-line check of item 5 reads
only its own statements.  The expert option `state_ambiguity` is off in
every variant (research runs name it: `--expert-option
state_ambiguity=1`), and the rollout summary counts the episodes that
stated an ambiguity (`ambiguity_episodes`) apart from the isolated-line
ones.  Tests: `psse_env/test_ambiguity_statement.py`, the audit report in
`psse_env/dagger/test_release_audit.py`.  Step 5 measures how often each
teacher states it and on which roots.

**Commit.**  The commit that adds this section (branch
`claude/nostalgic-kalam-be5f7a`, 2026-10-28).

### 8.4 Step 4: recovering from a wrong committed correction, two contracts (designed, not built)

The user's rule asks the agent to roll back after fruitless cycles and try
the other error type.  Today it cannot: `rollback_state` discards only the
open candidate, and a committed correction has no undo.  How much is at
stake, from the item-3 runs (pair refit and abstention options):

| teacher | failed roots (of 1,275) | with a wrong correction committed | of those, a true correction also tried | failed without a wrong commit |
| --- | --- | --- | --- | --- |
| baseline_compare | 658 | 318 | 226 | 340 |
| topology first, guarded | 487 | 130 | 92 | 357 |

So an undo could reach at most the 130 guarded failures (318 under the
baseline teacher) whose final model carries a wrong edit: 130 wrong
parameter edits, 78 wrong meter edits and 24 wrong breaker flips under the
guarded teacher.  The other 357 failures end with no wrong edit kept (a
true fault left uncorrected, a verifier limit, a silent atom) and neither
contract changes their outcome.  The user is choosing between the two
contracts below; neither is built until the decision is relayed.

**A. A gated revert action.**
- *Action.*  `revert_correction(state_id, to_state_id)`: make an ancestor
  of the active state in the committed lineage active again, discarding
  the committed corrections after it.  The state store already keeps every
  committed state with its parent, so the revert is a pointer move with an
  exact hash check, as `rollback_state` is for a candidate.
- *Gate* (process oracle, observable quantities only).  Allowed only to an
  ancestor of the active state, only after K correction cycles since that
  ancestor without global progress (the WLS objective not lower than at the
  ancestor; K = 2 is the natural first setting), at most once per
  ancestor, and with no open candidate.  The reverted corrections move to a
  policy-visible `reverted_corrections` list with their hypothesis family
  and line, so the expert and a student see what was tried and switch to
  the other family on that line.
- *Teacher.*  After K fruitless cycles on the same alarm, revert to the
  state before the first correction of the current hypothesis and take the
  other route on the leading line (parameter after a failed breaker flip,
  breaker after a failed impedance edit), using step 2's confidence for the
  order; after the second hypothesis also fails, hand off with step 3's
  statement.
- *Scoring.*  The audit scores the final model as today; a reverted edit is
  neither kept nor counted as a false commit (it was committed under the
  gate's evidence, then withdrawn), and a revert that removes a true
  correction shows as an uncorrected fault in the final model.  New
  metrics: reverts per episode, roots recovered after a revert.
- *Costs and risks.*  A new tool in the action vocabulary (schema, the
  process oracle, the student's tool list, DAgger labels); revert cycles,
  bounded by once-per-ancestor and the action budget; the K-cycle rule
  spends actions before it fires (two to three cycles of five to six
  actions under the 40-action budget).

**B. Hand-off only.**
- *No undo.*  The contract stays as it is.  After K correction cycles on
  the same alarm without global progress the teacher stops and hands off,
  with step 3's ambiguity statement and the confidence, instead of
  continuing on a model that may carry its own wrong edit.
- *Teacher.*  The first route is chosen by step 2's confidence (the
  "attempt one hypothesis first" half of the user's rule), and a failed
  first hypothesis ends in an honest handoff rather than a switch (the
  "hand off and report the case" half).
- *Scoring.*  Unchanged: such roots stay failures, reported with the
  statement; a handoff that keeps a wrong edit fails on that edit as today.
- *Costs and risks.*  None in the contract; it cannot recover any of the
  130 (318) roots above, it only stops them earlier and labels them.

**A third shape, for reference.**  Trying a hypothesis on an isolated
clone and committing only what verifies, as `compare_parameter_candidates`
already does for impedance candidates, would avoid the undo altogether
(a "compare routes" step that fits the breaker flip and the impedance edit
on clones and reports both verdicts).  It costs one action and the clone
solves an agent asked for, so it is not a screen; it is listed because it
answers the same need, not proposed in place of the user's two options.

**Commit.**  The commit that adds this section (branch
`claude/nostalgic-kalam-be5f7a`, 2026-10-28): design only, no code.

### 8.5 Step 5: the guarded teacher with steps 1-3, measured

All runs on the same code (steps 1 to 3 committed), scada_only, 40
actions, with the pair refit, the isolated-line abstention and the
ambiguity statement switched on for every teacher (the statement only
appears under the guarded teacher, the only one whose solve carries the
confidence).  The always-look order (`baseline_compare_topology_first`)
is shown as a labelled reference only, never as a candidate.

**Compound suite** (roots solved; the item-3 runs in brackets):

| class (roots) | baseline_compare | guarded | always-look (reference) | guarded against baseline, won / lost | handoffs with an ambiguity statement (guarded) |
| --- | --- | --- | --- | --- | --- |
| two R/X, adjacent (150) | 76 (76) | 76 (76) | 65 (65) | +0 / -0 | 6 |
| two R/X, distant (150) | 72 (72) | 72 (72) | 63 (63) | +0 / -0 | 3 |
| R/X + breaker, same line (75) | 47 (47) | 71 (71) | 75 (75) | +24 / -0 | 0 |
| R/X + breaker, shared bus (75) | 31 (31) | 45 (45) | 43 (43) | +14 / -0 | 5 |
| R/X + breaker, distant (150) | 73 (73) | 111 (109) | 109 (109) | +38 / -0 | 0 |
| meter + two R/X (150) | 66 (66) | 66 (66) | 54 (54) | +0 / -0 | 6 |
| meter on the R/X line (75) | 14 (14) | 14 (14) | 14 (14) | +0 / -0 | 8 |
| R/X + breaker + meter (150) | 59 (59) | 97 (97) | 97 (97) | +38 / -0 | 5 |
| two breakers, dangling (60) | 16 (16) | 46 (46) | 53 (53) | +30 / -0 | 1 |
| two breakers, mixed (45) | 20 (20) | 30 (30) | 43 (43) | +10 / -0 | 2 |
| two breakers, bus splits (45) | 22 (22) | 27 (28) | 39 (39) | +5 / -0 | 3 |
| control: one R/X (50) | 43 (43) | 43 (43) | 37 (37) | +0 / -0 | 1 |
| control: one breaker (50) | 31 (31) | 44 (44) | 50 (50) | +13 / -0 | 0 |
| control: one meter (50) | 47 (47) | 47 (47) | 47 (47) | +0 / -0 | 0 |
| all (1,275) | 617 (617) | 789 (788) | 789 (789) | +172 / -0 | 40 |

Mean actions: 17.3 (baseline), 13.8 (guarded), 14.1 (always-look); no
false commit under any teacher.  The baseline and the always-look
reference reproduce their item-3 trajectories on 1,275 of 1,275 roots;
the guarded teacher changes 34, 33 of them on classes with a breaker
error, where the confidence moved the order (R/X + breaker distant +2, two
bus splits -1; the other 31 keep their outcome), and one on two R/X
adjacent with the outcome unchanged.

**Conventional suites** (end-to-end runner; the validation suite and the
hard draw are the torch files):

| suite | baseline_compare | guarded | always-look (reference) | guarded trajectories equal to the baseline's |
| --- | --- | --- | --- | --- |
| conventional test suite (192) | 192 | 192 | 178 | 141 |
| validation (96) | 96 | 96 | 89 | 76 |
| hard draw (193) | 185 | 184 | 163 | 182 |

No false commit anywhere.  Per family, the guarded teacher equals the
baseline everywhere except the hard draw's parameter family (41 to 40):
the same root as in section 7.3, `r0_344bb11720e7`.  There the confidence
is 0.854 for an open terminal on line 6: a lightly loaded line (1.2 sigma
of P flow, 22.7 sigma of P residual) that is the runner-up branch
candidate at 0.854 of the top multiplier, while the true error is an
impedance error on the top-ranked line 3.  The breaker route goes first,
the flip opening line 6 lowers the objective and is kept (a viable
partial step under the user's rule), line 3 is then corrected, and the
episode hands off with the wrong flip on the model.  This is the user's
lightly loaded ambiguity, and it is reported as a failure.  Recovering it
needs the revert of section 8.4 (now step 6), not a rule aimed at it.
The always-look reference loses 14, 7 and 22 conventional roots, all
parameter or measurement+parameter roots on which the topology context
offers a line outage that verifies as partial progress.

**The remaining guarded failures on the compound suite, by mechanism**
(486 roots; in brackets, how many of them carry a true fault on a line
whose flows read at noise without the cue at the first solve, the lightly
loaded ambiguity the user accepts as a failure):

| mechanism | roots | lightly loaded | where |
| --- | --- | --- | --- |
| an impedance fault never corrected | 174 | 74 | two R/X distant 57, meter + two R/X 39, adjacent 20, R/X + breaker distant 19 |
| every fault touched, the final model outside tolerance | 70 | 19 | meter + two R/X 15, adjacent 14, distant 11, meter on the line 11 |
| the action budget spent, or a loop | 70 | 19 | meter + two R/X 14, adjacent 13, R/X + breaker + meter 10, two bus splits 10 |
| a wrong impedance edit kept | 58 | 16 | across ten classes |
| an impedance correction attempted and not accepted | 43 | 4 | meter on the line 21 (the verifier limit of section 5.1) |
| a wrong impedance and a wrong meter edit kept | 19 | 3 | adjacent 6, two bus splits 5 |
| a wrong meter edit kept | 16 | 5 | adjacent 8, meter on the line 4 |
| a breaker fault left, no wrong edit | 16 | 14 | R/X + breaker + meter 7, distant 4 |
| a wrong breaker flip kept (alone or with a wrong meter edit) | 12 | 1 | |
| a meter fault never corrected | 6 | 0 | |
| a wrong breaker flip and a wrong impedance edit kept | 2 | 0 | adjacent 2 |

155 of the 486 failures carry the lightly loaded ambiguity; 129 keep a
wrong committed edit (the 107 of the wrong-edit rows above, which ended in
a handoff, and 22 of the budget and loop row), the population the revert of
step 6 can reach; the rest are faults the teacher never reached or could
not fit (silent second atoms, the adjacent-pair estimator, the verifier
limit on a meter on the faulted line).  None of them is addressed by a rule
here.  (Corrected on 2026-10-28: the first version of this section said 113.)

**The ambiguity statements** (guarded teacher, compound suite): 40
handoffs stated one (39 of them failed roots).  The stated line carried
an impedance error on 11, an open terminal on 8 and neither on 21; 37 of
the 40 rest on a lightly loaded line without the cue
(`flow_at_noise_without_cue`).  So the statement names a faulted line a
little under half the time: it reports the solve's ambiguity honestly, it
does not locate the fault.  On the conventional suites it appears once
(the hard draw's `r0_098e749ba7ac`, a solved measurement+parameter root,
line 19 at 0.343).

**Commit.**  The commit that adds this section (branch
`claude/nostalgic-kalam-be5f7a`, 2026-10-28).  Item 7's commits so far:
713c0ff (step 1), 0952ab1 (step 2), e76b18e (step 3), 619e659 (step 4).

### 8.6 Step 6: the gated revert, the confidence over all categories, and the teacher that walks it

The user chose design A of section 8.4 ("extend rollback") and widened step
2's confidence to every balanced category: "this confidence mechanism doesn't
need to be limited to parameter versus topology errors. If we can introduce a
confidence score across all error categories, we could either trigger a
handoff or explore different candidate paths whenever competing hypotheses
arise."  Two commits: the contract, then the teacher rule.

**The contract (`revert_state`).**  `revert_state(to_state_id)` makes a
committed ancestor of the active state active again.  It exists only in an
environment built with `revert_fruitless_cycles=K` (research runs: the
`baseline_compare_guarded_walk` variant, or `--env-option
revert_fruitless_cycles=K`); every other environment refuses it as an unknown
tool and its observation carries no revert ledger, so every existing teacher,
the paper contract and the student's tool surface are as they were.  The
process gate admits a revert only
- with no open candidate,
- to a committed ancestor of the active state,
- at most once per ancestor (a second revert to it is a lifecycle violation),
- after at least K fruitless correction cycles since that ancestor's commit
  on the active lineage (a rolled-back candidate or an executor failure; a
  process refusal tested nothing), fewer being a lifecycle violation,
- with the alarm still standing on the active state;
and it costs an action.  On a revert the controller's flags and the private
truth return to their snapshot at the ancestor's commit: the accepted
corrections are those of the new active lineage, and the contexts fetched on
the ancestor and its latest WLS are current again (no new solve).  The
correction withdrawn on the ancestor joins the ancestor's rejected hypotheses
(kind `reverted_after_fruitless_cycles`), so it is not tried there again, it
counts as tested for a handoff, and it opens the meter route under branch
dominance exactly as a rejected branch hypothesis does.  Every withdrawn
correction stays in the policy-visible `revert_ledger` (the committed lineage
with the fruitless cycles since each ancestor's commit and whether it was
reverted to, the withdrawn corrections, the reverts).  The truth audit reads
the final state, so it scores the corrections on the active lineage at the
handoff and healthy-component preservation on the final active state; the
compound rollout marks each withdrawn correction `reverted`.  The evaluator
records each revert's outcome (`revert_accounting`): the first correction
executed after it tests another (family, line) hypothesis, a recovery, or the
withdrawn one again, a loop (`loop_detected`), or none follows.  Files:
`psse_env/state_store.py` (`revert`), `psse_env/oracle/process_validity.py`
(`revert_refusal`), `psse_env/transactional_env.py` (`_step_revert`,
`_revert_ledger`), `psse_env/dagger/evaluator.py` (`update_revert_events`);
tests in `psse_env/test_gated_revert.py`.

**The confidence over meter, parameter and topology.**  From quantities the
solve already produced, no extra solve (`hypothesis_confidence` beside step
2's route confidence, under the same provider option, so only the guarded
teachers' solves carry it):
- meter (bad data) = r / (r + lambda), the largest normalized residual
  against the largest normalized branch multiplier, the classical Lagrangian
  test the routes already use;
- the branch share lambda / (r + lambda) splits into topology (breaker) and
  parameter (R/X) by step 2's open-terminal confidence c: topology = branch x
  c, parameter = branch x (1 - c);
- the topology context refines c with its breaker multipliers (step 2's
  refinement) and so refines the vector; the parameter and measurement
  contexts re-read the same solve and leave it as it is.
The ranking uses the bands the routes already use, so no new threshold: the
meter leads only when the solve is meter dominant (r > 1.2 lambda; the meter
is weighed against the branch block, because the Lagrangian test separates
meter from branch, not meter from one branch hypothesis), the open terminal
leads the branch block when c > 0.5 (step 2's rule).  Two hypotheses compete
when the rule that orders them cannot separate them: meter and branch inside
the 1.2 band, parameter and topology inside step 3's 3:1 band (c between 0.25
and 0.75).  Waveform families stay out of scope: their confidences would come
from the triage classifier's family scores (`family_scores`, 2026-10-13),
which already rank harmonic, HIF and unbalance against the balanced families
under `classifier_gated_diagnostics`.

**The ambiguity statement, generalized.**  At a handoff with the alarm
standing, when the latest solve's vector has competitors, the statement
(`ambiguous_hypotheses`) lists the leading hypothesis and every competing
one, each with its confidence and target (meter index or line), the ranking,
the families the lineage already tried and those withdrawn on the state.
It supersedes step 3's parameter-or-breaker statement wherever the solve
carries the vector, is recorded and reported (the truth audit adds what the
truth holds on each target), and is never required.

**The teacher rule (`baseline_compare_guarded_walk`).**  The guarded teacher
with `hypothesis_walk`:
- the balanced routes follow the vector's ranking (the meter/branch order
  is the existing dominance rule, so this changes only where the breaker
  context's refinement moves c across 0.5);
- where it would hand off with the alarm standing, it reverts instead, when
  the gate admits it and the budget still closes a lifecycle after the revert
  (the budget handoff's floor of 4, plus the revert): to the state before the
  first commit of the current hypothesis (the family of the latest commit,
  back through its run of commits);
- on the reverted state the withdrawn family goes behind the next one in the
  ranking (a withdrawn breaker flip puts the parameter route first, a
  withdrawn impedance edit the breaker route; a withdrawn meter edit lets the
  branch families propose on a meter-dominant solve);
- when the ranking is exhausted on a state that cannot be reverted again, it
  hands off with the statement.
The start handoff ("no category stands clearly above at the start") is the
other half of the user's rule.  It is measured from the walk run itself: the
decision reads only the first solve, so the handoff arm's outcome on every
start-ambiguous root is a handoff, and on every other root the walk's own
outcome.  Running it live needs one more escalation request in the contract
(today's handoffs require the same-state routes exhausted); not built.

**Choosing K.**  From the guarded teacher's step-5 trajectories (fruitless
cycles on each committed state before the episode moved on):

| committed states | 0 | 1 | 2 | 3 | 4 | 5 |
| --- | --- | --- | --- | --- | --- | --- |
| solved roots, states that moved on to a further commit (805) | 729 (91%) | 17 (93%) | 24 (96%) | 15 (98%) | 16 (100%) | 4 |
| solved roots, the final state (789) | 788 | 1 | | | | |

| failed roots with a wrong commit (129): fruitless cycles since the first wrong commit | 0 | 1 | 2 | 3 | 4 | 5 | 6+ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| roots (cumulative) | 63 (49%) | 8 (55%) | 12 (64%) | 6 (69%) | 15 (81%) | 17 (94%) | 8 |

96% of the committed states of solved roots moved on within 2 fruitless
cycles, so K = 3: a hypothesis still fruitless after 3 cycles is past what a
productive state needed (the baseline teacher's productive states needed more,
92% within 3, because its order delays the breaker route).  At K = 3 a revert
is admissible by the end on 46 of the 129 guarded failures with a wrong
commit.  63 of them never had a fruitless cycle after the wrong commit: on 47
the state it produced offered no correction to try and the teacher handed off
at once, and on 16 every later correction was accepted until the handoff.  A
gate that counts fruitless cycles cannot admit a revert on them at any K;
admitting one on an exhausted state with nothing tried is a different
contract (section 8.7).
K and "when to abandon a hypothesis", "which hypothesis next" (the ranking
walk) and "when to hand off" (start ambiguity against exploring) are
candidates for learning (RL); the rules here are their hand-set starting
points.

**DAgger labelling with a revert (design only).**
- Vocabulary: `revert_state(state_id, to_state_id)` is outside the
  student's tool surface (`MACRO_ACTIONS`, the protocol bridge's schemas);
  training on the walk teacher adds it there and to the dataset builder, and
  renders the revert ledger in the student prompt.  Nothing of that is built.
- Labels: the teacher's revert is a function of the observation alone (the
  ledger's lineage and fruitless cycles, the bound WLS alarm, the budget),
  so it is imitable; it fires only at would-be handoffs, so revert rows are
  rare (one per reverting episode) and need weighting or oversampling.
- Student states: on states the student reached through its own fruitless
  commits the teacher relabels from the same ledger, so aggregated rows teach
  when to abandon; a student revert the gate refuses (too early, a second
  one) is a lifecycle violation, a recovery row like a refused commit.
- After a revert the teacher's next label walks the ranking with the
  withdrawn family behind; a student that repeats the withdrawn hypothesis on
  the same line is flagged as a loop by the evaluator.
- K is part of the environment contract, so supervised data must hold it
  fixed; learning when to abandon happens inside the window the gate admits.

**Measured** (compound suite, the settings of step 5; the conventional
suites through the end-to-end runner, the validation suite and the hard draw
being the torch files; walk = `baseline_compare_guarded_walk`, guarded = the
step-5 run, unchanged by this step):

| class (roots) | baseline | guarded | walk | always-look (reference) | walk against guarded, won / lost | episodes with a revert |
| --- | --- | --- | --- | --- | --- | --- |
| two R/X, adjacent (150) | 76 | 76 | 76 | 65 | +0 / -0 | 7 |
| two R/X, distant (150) | 72 | 72 | 72 | 63 | +0 / -0 | 2 |
| R/X + breaker, same line (75) | 47 | 71 | 71 | 75 | +0 / -0 | 0 |
| R/X + breaker, shared bus (75) | 31 | 45 | 45 | 43 | +0 / -0 | 2 |
| R/X + breaker, distant (150) | 73 | 111 | 111 | 109 | +0 / -0 | 2 |
| meter + two R/X (150) | 66 | 66 | 66 | 54 | +0 / -0 | 10 |
| meter on the R/X line (75) | 14 | 14 | 14 | 14 | +0 / -0 | 6 |
| R/X + breaker + meter (150) | 59 | 97 | 97 | 97 | +0 / -0 | 7 |
| two breakers, dangling (60) | 16 | 46 | 46 | 53 | +0 / -0 | 1 |
| two breakers, mixed (45) | 20 | 30 | 30 | 43 | +0 / -0 | 0 |
| two breakers, bus splits (45) | 22 | 27 | 28 | 39 | +1 / -0 | 4 |
| control: one R/X (50) | 43 | 43 | 43 | 37 | +0 / -0 | 0 |
| control: one breaker (50) | 31 | 44 | 44 | 50 | +0 / -0 | 1 |
| control: one meter (50) | 47 | 47 | 47 | 47 | +0 / -0 | 0 |
| all (1,275) | 617 | 789 | 790 | 789 | +1 / -0 | 42 |

| suite | baseline | guarded | walk | always-look (reference) |
| --- | --- | --- | --- | --- |
| conventional test suite (192) | 192 | 192 | 192 | 178 |
| validation (96) | 96 | 96 | 96 | 89 |
| hard draw (193) | 185 | 184 | 184 | 163 |

Mean actions on the compound suite: 17.3 (baseline), 13.8 (guarded), 14.2
(walk), 14.1 (always-look).  No false commit under any teacher on any suite.
The baseline and the always-look reference reproduce their step-5 runs on
1,275 of 1,275 compound roots, and the baseline reproduces step 5 root for
root on the conventional suites (192 of 192, 96 of 96, 193 of 193), with its
observation identical to d056b35 (0 of 172 steps differ on 14 conventional
roots, two per family).  Away from its reverts the walk teacher's
trajectories are the guarded teacher's on every root (the 1,233 compound
episodes without a revert and all 481 conventional roots): the vector's
ranking reproduces the guarded order everywhere, so the 42 reverts are the
whole behavioural difference.  It never reverts on a conventional root.  The hard draw's `r0_344bb11720e7`
stays lost: its wrong flip on line 6 is followed by an accepted correction of
line 3 and no fruitless cycle, so the gate never admits a revert (the
lightly loaded ambiguity, reported as a failure).

**The reverts.**  42 compound episodes reverted, each once.  The correction
executed next tested another hypothesis on 41 and the withdrawn one again on
1 (a second breaker on the same line); the first run of this measurement
labelled four more as loops because a meter edit carried no target, so a
different meter counted as the same hypothesis, fixed in 11f1209 (labels
only; the trajectories are the same).  The reverting episodes used 36.1
actions on average: 40 ended in a handoff, 1 ran out of budget and 1 was
solved (`r0_d8b8f11b8711`, two bus splits: a wrong impedance edit on line 15
was withdrawn after five fruitless parameter attempts, and two breaker flips
then solved the root).  The walk run carries 134 loop flags against the
guarded run's 130: four reverting episodes, all failed either way, re-ran the
same solve after corrections that failed numerically on the reverted-to
state (the evaluator's ordinary non-advancing repeat).

What the reverts withdrew decides whether the rule can work: 28 of the 42
withdrew only true corrections (13 meter edits, 7 impedance edits, 8 breaker
flips), 13 only wrong ones, 1 both.  In 24 of the 42 the fruitless attempts
after the commit belonged to another family: the next fault's corrections
failing on top of a correct commit, which is no evidence against it.
Restricting the revert to hypotheses chosen while another competed (the
vector's competitors on the ancestor) does not separate them either: of the
32 reverts to the root it keeps 6, and 4 of those withdrew true corrections.
So a commit followed by fruitless cycles is wrong less often than right, and
the rule as specified withdraws true corrections two to one.  It costs no
root, because it fires only where the episode was failing anyway.

**The guarded failures that ended with a wrong commit (129)**, under the
walk teacher:

| class | roots | solved | reverted, failed without a wrong edit | reverted, still a wrong edit | no revert |
| --- | --- | --- | --- | --- | --- |
| two R/X, adjacent | 25 | 0 | 3 | 1 | 21 |
| two R/X, distant | 3 | 0 | 0 | 0 | 3 |
| R/X + breaker, same line | 1 | 0 | 0 | 0 | 1 |
| R/X + breaker, shared bus | 10 | 0 | 0 | 1 | 9 |
| R/X + breaker, distant | 7 | 0 | 0 | 0 | 7 |
| meter + two R/X | 12 | 0 | 0 | 2 | 10 |
| meter on the R/X line | 11 | 0 | 1 | 1 | 9 |
| R/X + breaker + meter | 12 | 0 | 0 | 0 | 12 |
| two breakers, dangling | 10 | 0 | 0 | 1 | 9 |
| two breakers, mixed | 13 | 0 | 0 | 0 | 13 |
| two breakers, bus splits | 17 | 1 | 0 | 3 | 13 |
| control: one R/X | 2 | 0 | 0 | 0 | 2 |
| control: one breaker | 6 | 0 | 0 | 1 | 5 |
| all | 129 | 1 | 4 | 10 | 114 |

Why 114 never reverted (every one of them ended in a handoff): 75 had no
fruitless cycle since the first commit of the hypothesis the teacher would
revert (the state the commit produced offered nothing to try, or every later
correction was accepted), 23 had one or two (fewer than K = 3), and 16 had
too little budget left for a revert and a new lifecycle.

**Calibration of the confidence vector** (the first solve of each root):
the top-ranked category is a true error family on 1,198 of 1,275 compound
roots (94%), and the top two contain one on all 1,275.  It is weakest where
a breaker error looks like an impedance error: bus splits 14 of 45 (31%; the
parameter family leads on 31), the one-breaker control 32 of 50 (64%), two
breakers mixed 32 of 45 (71%); 93% on two adjacent R/X errors, 99-100% on the
other classes.  On the conventional suites, among the roots whose first solve
alarms: 171 of 176, 81 of 88 and 160 of 185.

**The start handoff, measured from the walk run.**  192 compound roots are
start-ambiguous (no category clearly above at the first solve); the walk
teacher solved 115 of them, so handing off at the start would solve 675 of
1,275 instead of 790.  On the conventional suites 9, 5 and 28 roots are
start-ambiguous and the walk solved 9, 5 and 27 of them: the handoff arm
would score 183 of 192, 91 of 96 and 157 of 193.  Exploring is clearly
better; the start handoff hands over roots the teacher solves.

**The competing-hypotheses statement.**  40 compound handoffs carried it
(all failed roots).  At least one listed hypothesis named a true fault on its
target on 27 of them, none on 13; per hypothesis, the parameter one was on a
faulted line 22 of 40 times, the breaker one 6 of 25, the meter one 3 of 16.

**Reading.**  The contract works as specified (gated, once per ancestor,
with the truth and the flags restored, labelled in the evaluator, the
baseline untouched), and the confidence vector ranks a true family first on
94% of compound roots.  The revert-and-walk teacher does not pay: +1 root,
0.4 more actions per episode, and its reverts withdraw true corrections two
to one, because the signal it acts on, fruitless cycles after a commit, comes
mostly from other faults.  It stays a labelled research variant; the guarded
teacher of step 5 remains the compound teacher.

**Candidates for learning (RL)**, flagged as the user asked: when to abandon
a hypothesis (the hand rule above withdraws the wrong commit only one time
in three, so the decision needs a credit signal the rule does not read: which
commit the remaining alarm belongs to), which hypothesis next (the ranking
walk), when to hand off (the start handoff against exploring: exploring wins
by 115 compound roots), and K itself, which the supervised data must hold
fixed as part of the environment contract.

### 8.7 Open questions

1. *Exhausted states.*  75 of the 114 unreached wrong-commit failures hand
   off with no fruitless cycle since the hypothesis's first commit (nothing
   left to try, or every later correction accepted), so the gate's count
   stays at zero.  Admitting a revert there is a different contract; given
   that the latest commit is wrong one time in three, it would mostly
   withdraw more true corrections.  The recommendation is not to widen it.
2. *Telling the wrong commit.*  What would identify it is evidence about the
   commit itself: its own target metric re-read on the current state, or,
   better, not committing a parameter-or-breaker guess at all: comparing the
   two routes on isolated clones before committing (the third shape of
   section 8.4, `compare_parameter_candidates` widened to a breaker flip
   against an impedance edit on the leading line).  It costs one action and
   the clone solves the agent asked for, so it is not a screen.  Not built.
3. *A live start handoff* needs one more escalation request in the contract
   (today's handoffs require the same-state routes exhausted).  The derived
   arm above suggests it is not worth building for the teacher.

**Commits.**  4e37f2d (the contract), 7197c0a (the teacher rule), 11f1209
(the loop label), and the commit that adds this section (branch
`claude/nostalgic-kalam-be5f7a`, 2026-10-28).  Outputs (gitignored):
`output/item7_step6/` (compound and conventional runs, logs, the test logs).
