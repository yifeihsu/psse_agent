# Compound errors on IEEE 118: the rule expert's ceiling on the classes that need no breaker (2026-10-28)

The IEEE 14 compound study (`docs/compound_errors_study_20261027.md`) left an
open question: IEEE 14 has 20 branches, a random pair of its perturbable lines
shares a bus 24% of the time, and "distant" pairs are never more than three
hops apart, so its findings about distant pairs and silent second atoms may be
artifacts of the small system.  This note measures the same rule expert,
`baseline_compare` under `scada_only` with 40 actions, on the six compound
classes that need no breaker, drawn on IEEE 118 (186 branches) by the same
sampler with every atom admitted at an effect ratio of at least 1.5 on its
own, and compares class by class at matched effect ratios.  CPU only, nothing
trained, research only; `baseline_compare` is unchanged and nothing under
`docs/papers/` was touched.

In one paragraph: the expert solves 456 of 625 IEEE 118 roots as deployed
(73%) and 471 with the adjacent-pair refit (75%), against 287 and 318 of the
625 IEEE 14 roots of the same classes.  Most of the difference is not the
network: on the IEEE 14 roots whose atoms each reach the floor, and on an IEEE
14 control drawn by the very synthesizer and admission used for IEEE 118,
the distant pairs are already at 77 to 79% (48% overall), so the IEEE 14
distant-pair finding was the silent second atom, and silent atoms are not a
small-system artifact: on IEEE 118, 56% of the physically admissible distant
draws carry a line below the floor, against 55% on IEEE 14 under the same
recipe.  What is an artifact of the small system is the clustering: random
pairs share a bus 3.4% of the time on IEEE 118 (24% on IEEE 14) and adjacent
footprints overlap a third as much (median 0.14 against 0.42 to 0.48), so the
adjacent pairs rise from 41% to 63% (57% to 73% with the pair refit) at
matched effect ratios and the meter + two R/X class, whose random pairs are
mostly adjacent or one hop apart on IEEE 14, from 59% to 87%.  The same-line
meter pair is a local verifier problem and holds on both systems (15 to 19%).
The open-terminal cue fires falsely on 54 of the 625 IEEE 118 roots (8.6%)
against 7 of 625 on IEEE 14, and ranking ambiguity does not grow at
high-degree buses: it shrinks.

Section 9 (2026-10-29) adds an action-budget sweep.  At 16 actions the expert
keeps 424 of its 456 IEEE 118 successes.  With the true targets, 30 of the 34
lost roots fit in the budget, but a single different decision recovers only
1.  The loss is one recurring route choice: after a committed meter
correction, the measurement route or the parameter route.  Its cue, the
maximum normalized residual, nearly separates the two with one threshold on
both systems.  At the deployed 40 actions the route-choice headroom is 10 of
625 roots.

Section 10 (2026-10-29) runs that threshold as a labelled rule, an opt-in
option of the rule expert (`meter_route_residual=11`, now also the named
variant `baseline_compare_meter_route`; `baseline_compare` is unchanged).  It gains 30 IEEE 118 roots at 16 actions (424 to 454) and 28 on
the IEEE 14 control (328 to 356), all of them meter controls, and 5 and 1 at
40 actions.  It loses no root on either system at either budget.  Its cost
is a few more wrong meters committed on roots that fail under both teachers.

Section 11 (2026-10-29) repeats section 10 on a fresh draw of both suites
(suite seed 20261029, disjoint from the first draw), with the threshold left
at 11.  The gains hold: 32 IEEE 118 roots at 16 actions (405 to 437) and 22
on the control (341 to 362), all of them meter controls, and 4 and 1 at 40
actions.  On IEEE 118 the threshold separates the fresh states as cleanly as
the in-sample ones.  The no-loss result does not hold: on the control the rule
loses one meter + two R/X root at both budgets, through the masking section
10.3 named.  It commits a healthy voltage meter at a terminal of a bad line,
and the line's true correction is then rejected.  One `baseline_compare`
trajectory of the fresh draw could not be reproduced.  Section 11.8 traces it
to a Windows-only race between concurrent runs in the provider's
derived-case cache.  Re-run in isolation, every other fresh episode and every
in-sample episode checked reproduces, so no count changes.  The cache has since
been fixed.

Branch `claude/pensive-darwin-33a0d8` (fast-forwarded from the compound
branch at a52a7e5).  Commits: e8c7ddc (the bus-branch synthesizer, the pair
metrics, the scripts' system switch, tests), d148464 (the comparison script,
the guard option of the multi-scan corrector, the tracked summaries in
`docs/compound_errors_ieee118_20261028/`) and the commit that adds this note.  Data (gitignored): `output/compound_errors_ieee118_20261028/`
(`suite.json`, `manifest.json`, `atoms.jsonl`, `expert_baseline_compare/`,
`expert_baseline_compare_pair_refit/`, `audit/`, `identifiability/`,
`compare/`) and the IEEE 14 control `output/compound_errors_ieee14_busbranch_20261028/`.
The IEEE 14 study's suite and rollouts are read from its own worktree
(`.claude/worktrees/nostalgic-kalam-be5f7a/output/compound_errors_20261027/`),
never written.

## 1. The synthesizer on the bus-branch model

IEEE 118 has no node/breaker model (the registry declares detailed topology
for IEEE 14 only), so the compound sampler's IEEE 14 path (the 65-node power
flow, the substation telemetry, the operator vector on the schematic-normal
layout) cannot run there.  A second physics path,
`compound_physics="bus_branch"` (the default on every system but IEEE 14),
draws the classes that need no breaker on the system's own case through the
registry (`psse_env/systems/registry.py`):

- the scaled case (load scale 0.80 to 1.25, as every synthesized family)
  receives the true impedances (both factors drawn from the corpus ranges
  [0.1, 0.5] or [2, 5], as on IEEE 14) and is dispatched by its own AC OPF;
  the solution passes the balanced corpus's physical checks
  (`psse_env/providers/balanced_corpus.py`: voltage and generator bounds,
  generator-minus-load injections, solver flows);
- the measurement vector is the solution's measurement function with the
  registry's deployed WLS noise model (0.001 pu on voltage magnitudes, 0.01 pu
  on powers) and one sensor-noise draw; eight scans are drawn about the same
  noiseless operating point, as the balanced corpus draws them;
- the meter atoms are the IEEE 14 study's overlay (0.10 to 0.30 pu on P/Q
  channels, lifted to the ten-sigma floor; `on_fault` puts the first meter on
  a flow channel of the faulted line), now a helper both paths share;
- the corrected model is the planning case with the true impedances.

The classes are the IEEE 14 study's six that need no breaker, under the same
names (`BUS_BRANCH_COMPOUND_CLASSES`): two R/X lines adjacent (they share a
bus) and distant, meter + two R/X (any pair), meter on the faulted line + R/X,
and the controls (one R/X line; one to three meters).  "Distant" is now a
field of the class, `distant_hops`, the smallest bus-graph distance between
the closest endpoints of the two lines: 2 on IEEE 14 (the study's definition;
only 8 of the 105 pairs of IEEE 14's perturbable lines are three hops apart
and none is four), 3 on IEEE 118.

**Admission at the effect floor.**  Every atom is admitted only if its own
effect reaches an effect ratio of 1.5, the low end of the effect-targeted band
of `research/classifier_triage` (`effect_sampler.py`: the chi-square excess
over the excess of the alarm threshold, so a ratio of 1 is "just alarming" at
any network size: dof 95 on IEEE 14, 863 on IEEE 118).  An atom's own effect is
the noiseless chi-square of the root's operating point with that atom alone,
on the reported model, over the threshold's excess, which is the expected
effect ratio of a noisy snapshot: a line atom alone is the network with only
that line's true impedance, re-dispatched by its own OPF at the root's load;
a meter atom alone is its offset on the healthy network's solution.  On top of
that the admission is the IEEE 14 study's physical one (the reported model
alarms, chi-square at alpha 0.01 or a normalized residual of 4, and the true
model is clean).  Each root also records every atom's effect with the root's
own noise draw and whether that atom alone would alarm on this snapshot
(`silent_atoms`, `alarms_only_jointly`), and every refused draw records whether
it alarmed jointly and was clean, so the draws the floor excludes are counted,
not hidden.

**Uniform metadata.**  Every root carries the same policy-visible fields: the
declared noise (`sigma_z`), the evidence profile and the eight parameter
scans.  The IEEE 118 provider offers no substation or node/breaker channel and
`scada_only` admits no auxiliary stream, so there is nothing else to carry.

**Offline record.**  Beside the IEEE 14 study's fields (atoms, relations, solo
effects, the deployed parameter ranking), each pair relation records its hop
count (`bus_distance`) and the overlap of the two lines' footprints: the first
canonical correlation of their (R, X) blocks in the covariance of the
branch-parameter multipliers the deployed solve computes
(`psse_env/providers/compound_pairs.py`), the cosine of the smallest angle
between the residuals the two errors leave (1: no snapshot tells them apart).
The root also records how many lines' multipliers lie within the comparison
band (1.5) of the top one and the bus degree of each true line.

**Nothing else moves.**  The IEEE 14 node/breaker path is untouched: one root
per class of all fourteen classes, drawn before and after the change on the
same seed, is byte-identical (ids, vectors, truth, the offline record and the
rejections).  The compound, bus-branch, provider, systems,
candidate-comparison, pair-refit, topology-order, corrector and guard test
files pass (157 tests, 57 subtests), among them 12 new tests for the
bus-branch path (`psse_env/providers/test_compound_scenarios_bus_branch.py`),
3 for the scripts' system switch (`research/compound_errors/test_system_switch.py`)
and 4 for the guard option of section 3 (`tools/test_multi_scan_rcond.py`).  A
generator for a system other than IEEE 14 still requires its fresh corpus
unless it is built `compound_only`.

## 2. The suites

`build_suite.py --system case118`, seed 20261027, ten generators per class
(seeded from the suite seed, the class name and the shard), `scada_only`, the
ten-sigma meter floor, 1,325 draws for 625 roots in 20 min on 26 workers, no
OPF divergence, 655 draws refused for an atom below the floor and 45 for a
true model that is not clean (the normalized-residual test at 4 on 1,098 clean
measurements).  The matched-recipe IEEE 14 control (`--system case14
--physics bus_branch --distant-hops 2`, the same seed, 1,052 draws, 342 s) is
drawn by the same synthesizer, noise model, admission and metadata, so it
differs from the IEEE 118 suite only in the network.

| class | n | draws | refused below the floor although physically admissible (share of the physically admissible draws) | strata of the deployed ranking (dominant / ambiguous / misranked / unranked) | weakest atom's effect ratio, median | hops, median | footprint overlap, median |
| --- | --- | --- | --- | --- | --- | --- | --- |
| IEEE 118: two R/X lines, adjacent | 150 | 248 | 72 (32%) | 103 / 38 / 9 / 0 | 7.9 | 0 | 0.137 |
| IEEE 118: two R/X lines, distant | 150 | 378 | 194 (56%) | 94 / 46 / 10 / 0 | 4.6 | 6 | 0.000 |
| IEEE 118: meter + two R/X | 150 | 412 | 232 (61%) | 104 / 32 / 14 / 0 | 2.6 | 5 | 0.001 |
| IEEE 118: meter on the faulted line + R/X | 75 | 130 | 42 (36%) | 51 / 13 / 11 / 0 | 3.7 | | |
| IEEE 118: control, one R/X line | 50 | 80 | 14 (22%) | 36 / 12 / 2 / 0 | 14.4 | | |
| IEEE 118: control, 1 to 3 meters | 50 | 77 | 24 (32%) | n/a | 2.6 | | |
| IEEE 14 control: two R/X lines, adjacent | 150 | 221 | 66 (31%) | 71 / 61 / 18 / 0 | 26.6 | 0 | 0.417 |
| IEEE 14 control: two R/X lines, distant | 150 | 349 | 185 (55%) | 83 / 48 / 19 / 0 | 3.6 | 2 | 0.021 |
| IEEE 14 control: meter + two R/X | 150 | 262 | 106 (41%) | 75 / 48 / 27 / 0 | 4.3 | 1 | 0.224 |
| IEEE 14 control: meter on the faulted line + R/X | 75 | 96 | 21 (22%) | 50 / 15 / 7 / 3 | 5.7 | | |
| IEEE 14 control: control, one R/X line | 50 | 73 | 6 (11%) | 37 / 9 / 4 / 0 | 32.6 | | |
| IEEE 14 control: control, 1 to 3 meters | 50 | 51 | 1 (2%) | n/a | 6.0 | | |

No IEEE 118 root has an atom that would stay silent alone on its own snapshot
(`silent_atoms` empty on all 625; six IEEE 14 control roots have one), so no
IEEE 118 root alarms only jointly.  The floor does real work on both systems,
and the same work: of the draws the IEEE 14 study's physical admission would
have kept, the floor refuses 32% and 31% of the adjacent pairs and 56% and 55%
of the distant pairs on IEEE 118 and IEEE 14; only the meters differ (the
refused draws hold 116 meters below the floor on IEEE 118 against 1 on IEEE
14), because a ten-sigma meter offset is a smaller effect against 863 degrees
of freedom than against 95.  The random pairs of the meter + two R/X class share a bus on 3 of
150 IEEE 118 roots, 43 of 150 on the IEEE 14 control and 37 of 150 in the
IEEE 14 study.

## 3. The expert on IEEE 118

`rollout.py --system case118 --expert baseline_compare`: `scada_only`, the
40-action budget, the paired-evaluation contract, every root once, as deployed
and with `--env-option pair_refit_adjacent=1`.  Two settings belong to the
system, not the teacher, and are recorded in the summaries:

- **The progress basis.**  A system other than IEEE 14 runs with
  `chi_square_excess`, the basis the IEEE 118 cells have used since
  2026-10-11 (the verifier's partial-progress floors were set on IEEE 14's 95
  degrees of freedom; against 863 the whole-statistic basis makes them several
  times stricter).  On the IEEE 14 control the two bases give 374 and 375 of
  625 with identical trajectories on 579 roots, so the basis is not a
  confound of the comparison.
- **The conditioning guard of the multi-scan corrector.**  The corrector
  aborts when the 2-norm reciprocal condition of its eight-scan gain falls
  below 1e-14, computed by a dense SVD that takes about ten seconds per
  iteration on IEEE 118's 1,882-column gain (110 of the 112 s of one
  correction).  The runs set `PSSE_MULTI_SCAN_RCOND=sparse_symmetric`, an
  option of `tools/correct_parameter_group_multi_scan_port.py`, off by
  default, that reads the same quantity from the symmetric gain's extreme
  eigenvalues (Lanczos and shift-invert Lanczos, 0.02 s) and hands every
  estimate within three decades of the floor, every gain that cannot be
  factored and every Lanczos run that does not converge to the dense SVD, so
  the guard's decision stays the SVD's.  Checked before use: on 169 IEEE 118
  gain matrices of true- and wrong-line corrections the two agree (relative
  gap at most 2e-4, on the two near-singular gains where the guard fires and
  the SVD decides anyway; 1e-12 elsewhere); the IEEE 14 control rolled out
  both ways gives 625 of 625 identical trajectories, and 12 IEEE 118 roots
  (two per class) give 12 of 12 (10 solved both ways, 11,053 against 975 CPU
  seconds).

**Per class** (`expert_baseline_compare/summary.{json,md}`; 44 min on 12
workers beside the pair-refit run).  The columns are the IEEE 14 study's:
"solved" is the truth-audited success; the verdicts are the deployment
verifier's dispositions of the expert's corrections, split by whether the
target is a true one; "refused" is an executor failure (the corrector not
converging).

| class | n | solved | mean actions | mean WLS calls | comparisons | true corrections: final / partial / rejected / refused | wrong-target corrections accepted / rejected | handoff |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| two R/X lines, adjacent | 150 | 95 (63%) | 15.7 | 4.11 | 187 | 114 / 171 / 6 / 35 | 45 / 95 | 150 |
| two R/X lines, distant | 150 | 130 (87%) | 13.0 | 3.27 | 183 | 131 / 146 / 2 / 13 | 21 / 28 | 150 |
| meter + two R/X | 150 | 130 (87%) | 17.1 | 4.26 | 202 | 136 / 287 / 3 / 10 | 16 / 37 | 150 |
| meter on the faulted line + R/X | 75 | 11 (15%) | 22.6 | 6.52 | 64 | 15 / 34 / 70 / 5 | 50 / 241 | 75 |
| control: one R/X line | 50 | 45 (90%) | 8.0 | 2.16 | 28 | 46 / 0 / 0 / 3 | 3 / 6 | 50 |
| control: 1 to 3 meters | 50 | 45 (90%) | 21.6 | 5.78 | 31 | 45 / 51 / 0 / 0 | 0 / 143 | 50 |
| all | 625 | 456 (73%) | 16.1 | 4.21 | 695 | | | 625 |

No episode has a false commit by the verifier's own standard and none runs
out of budget unannounced; every episode ends in an operator handoff.  The
evaluator's invalid-action and loop counters (102 and 50 episodes) count
corrector failures and repeated corrections of one line; the IEEE 14 study's
summaries count them the same way (84 and 40 episodes on these six classes)
and the IEEE 14 control 100 and 41.

By stratum of the deployed ranking (solved / n): adjacent pairs dominant
68/103, ambiguous 26/38, misranked 1/9; distant pairs 82/94, 40/46, 8/10;
meter + two R/X 92/104, 30/32, 8/14; meter on the line 4/51, 2/13, 5/11; one
line 32/36, 11/12, 2/2.

**With the pair refit** (`expert_baseline_compare_pair_refit/`): 471 of 625.
Only the adjacent pairs move, 95 to 110 of 150 (+16 / -1 roots, paired; 121 of
150 trajectories identical); every other class is unchanged (the meter + two
R/X class has 3 adjacent pairs), against +27 / -1 on the IEEE 14 study's
adjacent pairs and +26 / -1 on the IEEE 14 control.

**What fails.**  *Adjacent pairs* (55 failures): the final case outside
tolerance on 50, a wrong target accepted beside the true ones on 31 (a wrong
line's impedance on 18, a meter edit on 10, a line-outage edit on 12; a root
can carry several); 35 true corrections refused by the corrector.  *Distant pairs* (20): the final case outside tolerance on 18, a
wrong target on 13.  *Meter on the faulted line* (64): the true line's
correction is rejected 68 times on 60 of the 75 roots, 27 failed episodes
commit nothing at all, and all 11 successes correct the meter before the
line, the order the IEEE 14 probes found to be the one that works (section
5).  *The meter controls* (5): all five are three-meter roots that hand off
with `recovery_budget_exhausted` at action 38 with two of the three meters
fixed: after each meter fix the structural-first order tries the parameter
route, which the scans on every root make actionable, and four wrong lines are
rejected before the meter route returns (the uniform-metadata cost of the IEEE
14 study's section 3.1, with meters in place of breakers).  *A wrong
line-outage edit*: without a node/breaker channel the topology context offers
a line-status flip read from the branch multipliers, and the expert commits a
wrong one on 32 IEEE 118 roots (12 adjacent, 1 distant, 3 meter + two R/X, 16
meter on the line) and 38 IEEE 14 control roots.  It is a late recovery step
on every one of them (never the first correction, always after a refused or
rejected one, at action 13 on average) and every one of those roots fails; it
is the bus-branch counterpart of the IEEE 14 study's wrong breaker flips (8 of
its 625 roots of these classes).

## 4. The decision audit

`audit.py --system case118` over `research/rl/decision_audit.py`
(`audit/summary.{json,md}`, 42 min on 20 workers): every root the deployed
expert failed (169) and six solved roots per class, 205 in all, with the IEEE
14 audit's settings (the first two decision states, two menu alternatives
each, the truth probe that applies every true correction with the expert
verifying and committing between them, seed 20261019).  Bins as the plan
defines them: `rule_best`, `legal_alternative` (a menu alternative solves a
root the expert failed), `interface_limit` (only the truth probe solves it),
`evidence_limit` (nothing does).

| class | audited (failed + solved) | legal alternative | interface limit | evidence limit | rule best | truth probe solves failed roots | IEEE 14 study: legal / interface / evidence (truth probe) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| two R/X lines, adjacent | 55 + 6 | 3 | 0 | 52 | 6 | 1 of 55 | 6 / 1 / 93 (4 of 99) |
| two R/X lines, distant | 20 + 6 | 1 | 1 | 18 | 6 | 2 of 20 | 0 / 0 / 78 (0 of 78) |
| meter + two R/X | 20 + 6 | 1 | 2 | 18 | 5 | 2 of 20 | 1 / 1 / 87 (2 of 89) |
| meter on the faulted line + R/X | 64 + 6 | 0 | 1 | 63 | 6 | 1 of 64 | 0 / 0 / 61 (0 of 61) |
| control: one R/X line | 5 + 6 | 0 | 0 | 5 | 6 | 0 of 5 | 1 / 0 / 5 (1 of 6) |
| control: 1 to 3 meters | 5 + 6 | 9 | 1 | 0 | 1 | 5 of 5 | 5 / 1 / 1 (2 of 3) |
| all | 205 | 14 | 5 | 156 | 30 | | |

The bins read as on IEEE 14, with one difference in what the evidence-limit
bin holds.  On IEEE 14 most of it was atoms whose own effect is below the
alarm (74 of the 78 distant pairs carried one); on IEEE 118 the floor admits
none, so the bin is, in substance, interface limits the probe cannot certify
because it goes through the same gates: on the meter-on-line roots the true
meter correction is refused because the context does not advertise it (the
meter route stands down under branch dominance) and the true line's correction
is rejected by the verifier (section 5.2); on the adjacent pairs the corrector
fails to converge on a true line or the context no longer supports it after
the first commit; on the distant pairs and meter + two R/X the same refusals
on the 20 failures each.  The only class the audit resolves is the meter
controls: every failed three-meter root is solved by the truth probe and by
the legal alternative of taking the measurement route at the decision state
where the expert took the parameter route, the ordering cost of section 3.
No class shows the IEEE 14 study's dominant legal alternative (the topology
context first on breaker roots), which needs a breaker.

## 5. Identifiability

`identifiability.py --system case118 --experiments same_line_meter
adjacent_lines` (`identifiability/summary.json`, 62 min on 8 workers), the two
of the IEEE 14 study's four probes that apply without a breaker, on every
root of their classes; the same run on the IEEE 14 control
(`output/compound_errors_ieee14_busbranch_20261028/identifiability/`, 22 min).
The *gated* path is the environment with its process gates, the *ungated*
path the same executors and candidate oracle without them (the study's
section 5); on IEEE 118 the ungated path measures progress on the chi-square
excess, as the gated one does.

### 5.1 Two wrong lines: adjacent against distant

| | IEEE 14 study: adjacent / distant | IEEE 14 control: adjacent / distant | IEEE 118: adjacent / distant |
| --- | --- | --- | --- |
| a true line ranked first by the parameter context | 137 / 130 | 132 / 131 | 141 / 140 |
| both true lines in the top 2 / top 3 | 71, 95 / 15, 33 | 75, 103 / 16, 37 | 62, 91 / 44, 74 |
| one true line unranked | 30 / 108 | 23 / 89 | 32 / 45 |
| ranking dominant (median ratio) | 92 (1.32) / 94 (1.50) | 76 (1.23) / 85 (1.34) | 104 (1.48) / 94 (1.35) |
| bounded comparison: best is a true line / no best | 127, 6 / 148, 0 | 122, 11 / 146, 1 | 133, 6 / 146, 2 |
| first line on the stale model: within tolerance (median X error) | 73 of 126 (26%) / 143 of 147 (1%) | 71 of 124 (20%) / 144 of 147 (1%) | 107 of 124 (5.5%) / 143 of 143 (1%) |
| second line on the derived model: within tolerance (median X error) | 65 of 129 (20%) / 106 of 136 (9%) | 88 of 125 (12%) / 128 of 138 (5%) | 118 of 135 (3.8%) / 142 of 142 (1.8%) |
| first line again, one alternating round, ungated: within tolerance (median X error) | 112 of 131 (6%) / 147 of 148 (1%) | 110 of 126 (3.5%) / 145 of 147 (1%) | 131 of 132 (1.4%) / 143 of 143 (0.9%) |
| after the round, ungated: case ok | 59 / 103 | 84 / 121 | 111 / 135 |
| gated: the third correction refused after a final commit | 67 / 138 | 59 / 132 | 87 / 132 |

The adjacent pair's estimator problem is the IEEE 14 study's finding, and it
is mostly the small network's: on IEEE 14 (study and control alike) the first
single-line estimate is off by a fifth to a quarter and within tolerance on
57 to 58% of the roots; on IEEE 118 it is off by 5.5% and within tolerance on
86%, and one alternating round settles 131 of 132.  The ranking was never the
obstacle on either system (a true line first on 137 to 141 of 150, the
comparison picks a true line on 122 to 133).  The distant pairs separate on
both systems; on IEEE 118 the second line is estimated within tolerance
every time (142 of 142, against 106 of 136 in the study, where it was often
below the alarm).  The gated refusals of the third correction are the
post-commit confirmation gate after the solve went clean, which on these
roots is the right stop: the round it blocks is the one the pair refit runs
inside the second correction instead (section 3).

### 5.2 Same-line impedance and flow meter (75 roots, with a twin each)

| | IEEE 14 study: original / twin | IEEE 14 control: original / twin | IEEE 118: original / twin |
| --- | --- | --- | --- |
| WLS dominance at the root (residual / branch / none) | 41, 27, 7 / 40, 29, 6 | 24, 45, 6 / 23, 46, 6 | 11, 44, 20 / 7, 48, 20 |
| true line ranked first | 66 / 48 | 65 / 64 | 64 / 71 |
| true meter among the supported meter groups | 48 / 10 | 29 / 9 | 30 / 9 |
| line corrected alone, gated: rejected (of the estimates) | 61 of 64 / 25 of 60 | 66 of 70 / 23 of 74 | 68 of 72 / 2 of 73 |
| ... estimate within the release tolerance | 55 of 64 / 54 of 60 | 67 of 70 / 71 of 74 | 71 of 72 / 72 of 73 |
| meter then line, ungated: both checks pass | 29 / 36 | 16 / 35 | 15 / 35 |
| line then meter, ungated: both checks pass | 1 / 33 | 1 / 49 | 1 / 69 |

The verifier finding holds unchanged on IEEE 118: the corrector estimates the
true impedance within tolerance on 71 of 72 roots, and the verifier rejects
it on 68, because the line's own bad flow meter keeps the line's multiplier
above its target test after the impedance is fixed; with the meter moved to a
healthy channel the same estimate is rejected on 2 of 73.  The order that
works is the IEEE 14 one, meter first and line second, and it passes both
checks on only 15 of 75 ungated roots, because the meter route stands down
under branch dominance or does not advertise the true meter (supported on 30
of 75).  This is an interface limit, local to the line, and the network size
does not touch it.

## 6. IEEE 14 against IEEE 118, class by class

Three IEEE 14 references: the study's suite (node/breaker telemetry, physical
admission; the stage-2 item 2 rollout of `baseline_compare` as deployed and
the item 3 rollout with the pair refit, 287 and 318 of these 625 roots, the
counts the study's section 7 reports for these classes), the study's roots
whose every atom reaches the floor ("matched";
a line's ratio from its recorded solo effect, a meter's from the symmetric
difference of its offset on the true model's clean snapshot, which agrees with
the bus-branch suite's own healthy-network record to a median 0.03% on the
326 IEEE 118 meters), and the IEEE 14 control.  Everything in this section is
`research/compound_errors/system_compare.py` (`compare/report.{json,md}`,
`compare/probes.jsonl`).

### 6.1 How compound errors cluster

| | IEEE 14 | IEEE 118 |
| --- | --- | --- |
| perturbable lines / pairs | 15 / 105 | 177 / 15,576 |
| a random pair shares a bus | 23.8% | 3.4% |
| within one hop | 54.3% | 10.3% |
| at least three hops apart | 7.6% (never more than 3) | 79.7% (median 5, up to 14) |
| largest bus degree | 5 | 12 |
| footprint overlap of a random pair at 0 / 1 / 2 / 3+ hops (median, unit-load OPF, registry noise) | 0.46 / 0.15 / 0.023 / 0.010 | 0.19 / 0.06 / 0.018 / 0.0005 |
| adjacent pairs whose footprints overlap above 0.5 | 44% | 21% |
| lines at noise level at unit load (both P flows under 3 sigma) | 2 of 15 | 14 of 177 |

Two adjacent IEEE 118 lines overlap less than two adjacent IEEE 14 lines
because each still has its own four flow meters while the shared bus is one
of many, and a random pair on IEEE 118 is almost never adjacent.  The suite's
pairs: adjacent 0.137 on IEEE 118 against 0.478 (study) and 0.417 (control);
distant 0.000 against 0.021; the random pairs of meter + two R/X 0.001
against 0.035 and 0.224.

### 6.2 The expert ceiling per class

Solved, as deployed (pair refit in parentheses where it differs); "matched" is
the study's subset whose every atom reaches the floor.

| class | IEEE 14 study, all | IEEE 14 study, matched | IEEE 14 control | IEEE 118 |
| --- | --- | --- | --- | --- |
| two R/X lines, adjacent | 50 of 150, 33% (76, 51%) | 48 of 120, 40% (73, 61%) | 61 of 150, 41% (86, 57%) | 95 of 150, 63% (110, 73%) |
| two R/X lines, distant | 72 of 150, 48% | 56 of 71, 79% | 115 of 150, 77% | 130 of 150, 87% |
| meter + two R/X | 61 of 150, 41% (66) | 54 of 90, 60% (59) | 88 of 150, 59% | 130 of 150, 87% |
| meter on the faulted line + R/X | 14 of 75, 19% | 10 of 52, 19% | 13 of 75, 17% | 11 of 75, 15% |
| control: one R/X line | 43 of 50, 86% | 41 of 45, 91% | 48 of 50, 96% | 45 of 50, 90% |
| control: 1 to 3 meters | 47 of 50, 94% | 47 of 50, 94% | 49 of 50, 98% | 45 of 50, 90% |
| all six classes | 287 of 625, 46% (318) | 256 of 428, 60% (286) | 374 of 625, 60% (399) | 456 of 625, 73% (471) |

The matched study subset and the control agree class by class (40 / 41%, 79 /
77%, 60 / 59%, 19 / 17%), so the node/breaker measurement layout and the
substation channel change little once the atoms' effects are matched, and the
control isolates the network.  By the weakest atom's effect ratio (deployed,
solved / n, IEEE 14 control against IEEE 118): adjacent pairs 10/48 and 29/56
at 1.5 to 5, 8/22 and 36/54 at 5 to 20, 43/80 and 30/40 above 20; distant
pairs 73/105 and 65/80 at 1.5 to 5, 41/44 and 50/54 at 5 to 20; meter + two
R/X 50/90 and 108/128 at 1.5 to 5, 38/59 and 22/22 at 5 to 20; meter on the
line 9/30 and 10/54 at 1.5 to 5, 4/43 and 1/21 at 5 to 20.  The IEEE 118
advantage on adjacent pairs and on meter + two R/X holds in every effect bin;
on distant pairs it is confined to the weakest atoms (1.5 to 5).

### 6.3 The silent second atom

| | IEEE 14 study | IEEE 14 control | IEEE 118 |
| --- | --- | --- | --- |
| physically admissible random draws with an atom below 1.5: adjacent / distant / meter + two R/X | (admitted: 30 / 79 / 60 of 150) | 31% / 55% / 41% | 32% / 56% / 61% |
| truth probe: one atom corrected exactly leaves the rest silent (adjacent / distant / meter on the line) | 25 / 58 / 11 | 8 / 4 / 1 | 5 / 0 / 0 |
| trajectories: the first committed true correction verified final with an atom left (adjacent / distant / meter on the line) | 20 of 121 / 57 of 147 / 13 of 49 | 9 of 121 / 4 of 144 / 1 of 31 | 5 of 130 / 0 of 144 / 0 of 26 |

The silent second atom is a property of the error recipe, not of the
network: a factor of 2 to 5 or 0.1 to 0.5 on a line that carries little
flow barely changes any measurement, and IEEE 118 has proportionally as many
such draws as IEEE 14.  It mattered on IEEE 14 because the study admitted
those draws (79 of its 150 distant pairs carry a line below the floor, and 58
of them go silent once the other line is fixed); with the floor, the second
atom stays visible after an exact fix on both systems, and the expert's
"final with an atom left" verdicts nearly vanish.

### 6.4 Ranking ambiguity and bus degree

Lines whose normalized multiplier lies within the comparison band (1.5) of
the top one, by the larger endpoint degree of the root's true lines:

| system | true lines' degree | roots | lines within the band, mean | two or more | top line is a true line |
| --- | --- | --- | --- | --- | --- |
| IEEE 14 study | 1 to 3 | 43 | 1.88 | 18 | 39 |
| IEEE 14 study | 4 to 5 | 532 | 1.75 | 296 | 460 |
| IEEE 14 control | 4 to 5 | 565 | 1.86 | 353 | 489 |
| IEEE 118 | 1 to 3 | 20 | 2.45 | 13 | 14 |
| IEEE 118 | 4 to 5 | 167 | 2.40 | 118 | 150 |
| IEEE 118 | 6 to 8 | 293 | 2.13 | 171 | 272 |
| IEEE 118 | 9 or more | 95 | 1.53 | 35 | 93 |

High-degree buses do not make the ranking more ambiguous: a line at a hub is
the best-observed line in the network, and the top multiplier is a true line
on 93 of 95 such roots.  IEEE 118's larger band counts at low and middle
degree come from the pairs, where the second true line raises its own peak
(single lines: 1.93, 1.88, 1.63 and 1.42 by degree bin).  The deployed
comparison trigger (a ranking not dominant or dominant by less than 1.5)
fires on about the same share of roots on both systems (adjacent 79, distant
91, meter + two R/X 99 of 150 on IEEE 118; 112, 84, 103 on the control).

### 6.5 The open-terminal cue on roots without a breaker

`flow_at_noise_lines` at the first solve of the deployed `run_wls` (every
firing is false here: no root carries a breaker atom):

| class | IEEE 14 study | IEEE 14 control | IEEE 118 |
| --- | --- | --- | --- |
| two R/X lines, adjacent | 6 of 150 | 4 of 150 | 15 of 150 |
| two R/X lines, distant | 0 | 1 | 16 |
| meter + two R/X | 1 | 0 | 15 |
| meter on the faulted line + R/X | 0 | 0 | 3 of 75 |
| control: one R/X line | 0 | 0 | 4 of 50 |
| control: 1 to 3 meters | 0 | 0 | 1 of 50 |
| all | 7 of 625 (1.1%) | 5 of 625 (0.8%) | 54 of 625 (8.6%) |

The study's 7 of 625 is reproduced.  On IEEE 118 the cue fires on a true line
on 4 of the 54 roots and on a healthy line elsewhere on 50, with two or three
lines at once on 12: lightly loaded healthy lines next to an impedance error
read as open terminals (14 of the 177 perturbable lines carry less than 3 sigma
of P flow at unit load, against 2 of 15 on IEEE 14).  A guarded teacher whose
cue sits at the 8-sigma residual floor would take the topology route first on
those 54 roots; the cue's floor was set on IEEE 14 (section 7.1 of the study)
and fires too easily at this scale.

### 6.6 Mean actions

Mean actions as deployed (IEEE 14 study / control / IEEE 118): adjacent pairs
16.7 / 17.6 / 15.7, distant pairs 10.5 / 12.4 / 13.0, meter + two R/X 16.7 /
18.6 / 17.1, meter on the line 14.7 / 19.1 / 22.6, one line 7.5 / 7.7 / 8.0,
meters 21.3 / 21.6 / 21.6; WLS calls per episode 4.21 on IEEE 118 against 4.29
on the control.  The cost per root does not grow with the network; on IEEE
118 the distant pairs take slightly more actions (the second line is now
corrected instead of going silent) and the meter-on-line roots more (more
rejected lines and late recovery steps before the handoff).

### 6.7 Which IEEE 14 findings hold

- **Distant pairs fail on the silent second atom** (study: 66 of 78 failures
  leave the second line in place, 86% where both alarm alone): *holds, and it
  is not a small-system artifact.*  At matched effect ratios IEEE 14 itself is
  at 77 to 79%, IEEE 118 at 87%, and silent atoms are as common in IEEE 118's
  random draws as in IEEE 14's (56% against 55% of the physically admissible
  distant draws).  The finding is about the error recipe and the admission,
  and the floor settles it on either system.
- **Adjacent pairs are an estimator problem** (study: the first single-line
  estimate is off by a quarter, one alternating pass recovers it, the pair
  refit takes the class from 33% to 51%): *weaker on IEEE 118.*  The footprints
  overlap a third as much, the first single-line estimate is off by 5.5%
  instead of 20 to 26% (within tolerance on 86% of the roots instead of 58%,
  section 5.1), the class rises to 63% as deployed and 73% with the pair refit
  at matched effect ratios, and the refit's gain shrinks (+16 against +26
  roots); the adjacent-pair difficulty is real but its size on IEEE 14 is
  mostly the small network's crowding.
- **Meter + two R/X** (study 41%, matched 60%): *a small-system artifact.*  On
  IEEE 14 its random pairs share a bus or sit one hop apart on 53% (study) and
  69% (control) of the roots, so the class inherits the adjacent-pair problem;
  on IEEE 118 they do on 10% and the class behaves like the distant pairs
  (87%), with no gain from the pair refit.
- **The same-line meter is a verifier problem** (study: the correct impedance
  is rejected while the line's own flow meter is bad; 19%): *holds on IEEE
  118* (15%): the true line's correction is rejected on 60 of 75 roots and
  every success fixes the meter first.  It is local to the line, so the
  network does not change it.
- **The controls** (86% and 94% on IEEE 14): one line 90% and meters 90% on
  IEEE 118; the one-line failures are corrector refusals and wrong targets,
  the meter failures the budget cost of the parameter route on three-meter
  roots (section 3), a uniform-metadata effect the IEEE 14 meter roots, with
  fewer candidate lines, pay less.
- **The open-terminal cue's false fires** (7 of 625): *grow eightfold* (54 of
  625); the cue as tuned on IEEE 14 does not transfer.
- **Ranking ambiguity at high-degree buses**: *not observed*; the comparison
  band holds fewer candidates at hubs, not more.
- **Wrong structural edits accepted as partial progress**: *holds in its
  bus-branch form*, a wrong line-outage edit committed late on 32 IEEE 118
  roots (38 on the IEEE 14 control).

## 7. Decisions and follow-ups

For the user's decision:

1. **The guard option** (`PSSE_MULTI_SCAN_RCOND=sparse_symmetric`).  It lives
   in a deployed tool (`tools/correct_parameter_group_multi_scan_port.py`),
   off by default, and every run of this note that corrects a line on IEEE 118
   set it; it reproduces the dense SVD's trajectories on every root checked
   (625 of 625, 12 of 12) at a tenth of the CPU on IEEE 118.  Keeping it opt-in
   costs nothing on IEEE 14 (the SVD there takes 15 ms); making it the default
   would speed every IEEE 118 parameter correction about tenfold.  Unchanged
   until decided.
2. **The open-terminal cue does not transfer.**  Its residual floor (8 sigma),
   set on IEEE 14, fires on 54 of 625 IEEE 118 roots that carry no breaker,
   on lightly loaded healthy lines next to an impedance error.  The guarded
   teacher (`baseline_compare_topology_guarded`) would take the topology route
   first on those roots.  The clean-flip teacher, the default for new work on
   the compound branch since b546d2d, screens flips with the node/breaker
   estimate, which IEEE 118 does not have, so (as the sibling design note
   records; not run here) it orders as `baseline_compare` on IEEE 118.  Any
   use of the cue at this scale needs a floor that scales with the line's
   modelled flow.

Follow-ups this note does not start:

- The same-line meter pair is the largest loss on both systems (15 to 19%)
  and an interface limit: a meter-first rule when a line's own flow meter is
  among the top residuals, tested on both suites, is the obvious candidate.
- The three-meter controls on IEEE 118 hand off on the budget because the
  parameter route, actionable on every root, costs four rejected lines per
  meter fix; on residual-dominant solves the route order could be the
  measurement tier's first, which is a teacher change for the conventional
  families too and so the user's call.  Section 9 measures this cost under a
  budget.
- Breaker classes on IEEE 118 need either a node/breaker model or the
  bus-branch line-status route (`get_topology_context` already screens
  status flips without a node/breaker binding, as the 32 line-outage edits of
  section 3 show); the sibling design note
  `docs/sparse_multi_error_design_20261028.md` (branch
  `claude/agitated-sammet-93f94e`) plans variable-count clusters on IEEE 118
  on top of this synthesizer and finds the same distance law with a linear
  separability probe (random matched pairs separable 11% on IEEE 14, 67% on
  IEEE 118).
- The IEEE 14 study's section 3 says no invalid action or loop was flagged;
  its own summaries count 84 invalid-action and 40 loop episodes on these six
  classes (corrector failures and repeated corrections, as here), a wording
  to correct in that note.

## 8. Reproduction

```
python -m research.compound_errors.build_suite --system case118 --output-dir output/compound_errors_ieee118_20261028 --seed 20261027 --shards 10 --workers 26
python -m research.compound_errors.build_suite --system case14 --physics bus_branch --distant-hops 2 --output-dir output/compound_errors_ieee14_busbranch_20261028 --seed 20261027 --shards 10 --workers 4
PSSE_MULTI_SCAN_RCOND=sparse_symmetric python -m research.compound_errors.rollout --suite output/compound_errors_ieee118_20261028/suite.json --output-dir output/compound_errors_ieee118_20261028/expert_baseline_compare --system case118 --expert baseline_compare --workers 12
PSSE_MULTI_SCAN_RCOND=sparse_symmetric python -m research.compound_errors.rollout --suite output/compound_errors_ieee118_20261028/suite.json --output-dir output/compound_errors_ieee118_20261028/expert_baseline_compare_pair_refit --system case118 --expert baseline_compare --env-option pair_refit_adjacent=1 --workers 12
python -m research.compound_errors.rollout --suite output/compound_errors_ieee14_busbranch_20261028/suite.json --output-dir output/compound_errors_ieee14_busbranch_20261028/expert_baseline_compare --expert baseline_compare --workers 10
python -m research.compound_errors.rollout --suite output/compound_errors_ieee14_busbranch_20261028/suite.json --output-dir output/compound_errors_ieee14_busbranch_20261028/expert_baseline_compare_pair_refit --expert baseline_compare --env-option pair_refit_adjacent=1 --workers 10
PSSE_MULTI_SCAN_RCOND=sparse_symmetric python -m research.compound_errors.audit --suite output/compound_errors_ieee118_20261028/suite.json --episodes output/compound_errors_ieee118_20261028/expert_baseline_compare/episodes.jsonl --output-dir output/compound_errors_ieee118_20261028/audit --system case118 --expert baseline_compare --max-states 2 --max-alternatives 2 --solved-per-class 6 --seed 20261019 --workers 20
PSSE_MULTI_SCAN_RCOND=sparse_symmetric python -m research.compound_errors.identifiability --suite output/compound_errors_ieee118_20261028/suite.json --output-dir output/compound_errors_ieee118_20261028/identifiability --system case118 --experiments same_line_meter adjacent_lines --workers 8
python -m research.compound_errors.identifiability --suite output/compound_errors_ieee14_busbranch_20261028/suite.json --output-dir output/compound_errors_ieee14_busbranch_20261028/identifiability --experiments same_line_meter adjacent_lines --workers 4
python -m research.compound_errors.system_compare --suite ieee14_study=<study>/suite.json --suite ieee14_busbranch=output/compound_errors_ieee14_busbranch_20261028/suite.json --suite ieee118=output/compound_errors_ieee118_20261028/suite.json --rollout ieee14_study:deployed=<study>/stage2_item2/expert_baseline_compare/episodes.jsonl --rollout ieee14_study:pair_refit=<study>/stage2_item3/expert_baseline_compare/episodes.jsonl --rollout ieee14_busbranch:deployed=... --rollout ieee14_busbranch:pair_refit=... --rollout ieee118:deployed=... --rollout ieee118:pair_refit=... --output-dir output/compound_errors_ieee118_20261028/compare --workers 10
```

`<study>` is `.claude/worktrees/nostalgic-kalam-be5f7a/output/compound_errors_20261027`.
The validation runs of the guard option (section 3) and the progress-basis
check are `output/compound_errors_ieee14_busbranch_20261028/expert_baseline_compare_{sparse_rcond,excess}/`
and `output/compound_errors_ieee118_20261028/validation_{dense_svd,sparse_rcond}/`;
their outcomes are in `docs/compound_errors_ieee118_20261028/guard_validation.json`.

## 9. Action budget and route-choice headroom (2026-10-29)

The RL watchlist names route choice under ambiguity as a decision a learned
policy might improve on.  Before any learning, this section measures how much
of the expert's success depends on the action budget, and how much a
different decision could recover.  Same suites, same expert (`baseline_compare`,
`scada_only`, rollout seed 20261006; the IEEE 118 runs with the sparse guard),
budgets of 12, 16 and 20 actions against the deployed 40; a decision audit of
every root the 16-action budget loses (four decision states, three menu
alternatives each and the truth probe, all under 16 actions:
`audit.py --max-steps 16 --reference-episodes <40-action run>`); and the
40-action audit of the 169 IEEE 118 failures widened from two states and two
alternatives to four and three.  `research/compound_errors/budget_headroom.py`
does the tabulation; the tracked tables are `budget_headroom.{json,md}`,
`*_budget{12,16,20}_summary.*`, `*_audit_budget16_summary.*` and
`ieee118_audit_wide40_summary.*` in `docs/compound_errors_ieee118_20261028/`.
Success is the truth-audited resolution of section 3, so an episode that
resolves the state and then hands off with `recovery_budget_exhausted`
counts as solved (152 of the 303 budget handoffs at 16 actions on IEEE 118).

### 9.1 Success against the budget

| class | IEEE 118: 12 / 16 / 20 / 40 actions | IEEE 14 control: 12 / 16 / 20 / 40 actions |
| --- | --- | --- |
| two R/X lines, adjacent (150) | 94 / 97 / 96 / 95 | 44 / 59 / 59 / 61 |
| two R/X lines, distant (150) | 130 / 130 / 130 / 130 | 115 / 115 / 115 / 115 |
| meter + two R/X (150) | 0 / 129 / 129 / 130 | 0 / 77 / 85 / 88 |
| meter on the faulted line + R/X (75) | 4 / 4 / 9 / 11 | 11 / 11 / 11 / 13 |
| control: one R/X line (50) | 45 / 45 / 45 / 45 | 48 / 48 / 48 / 48 |
| control: 1 to 3 meters (50) | 17 / 19 / 25 / 45 | 14 / 18 / 28 / 49 |
| all (625) | 290 / 424 / 434 / 456 | 232 / 328 / 346 / 374 |

- Sixteen actions keep 93% of the deployed successes on IEEE 118 (424 of
  456) and 88% on the control (328 of 374); twelve keep 64% and 62%.
- The drop at 12 actions comes from the interface, not from a decision.  No
  meter + two R/X root is solved at 12 on either system.  Three corrections
  cost at least 13 actions: one WLS, then for each correction a context, the
  correction, the verifying WLS and the commit.  The expert's shortest such
  solve at 40 actions takes 15.
- Between 16 and 40 actions, 33 of the 34 IEEE 118 losses fall in two
  classes: the meter controls (19 solved at 16, 45 at 40) and the meter on the
  faulted line (4 and 11).  On the control the meter + two R/X class (77 and 88) and the
  adjacent pairs (59 and 61) lose roots as well.
- A tighter budget sometimes wins: 2 adjacent pairs on IEEE 118 and 5 roots
  on the control are solved at 16 actions and not at 40.  This was not
  examined further.

### 9.2 What a 16-action budget loses, and what recovers it

| class | IEEE 118: lost / legal alternative / truth probe only / neither | IEEE 14 control: lost / legal / truth only / neither |
| --- | --- | --- |
| two R/X lines, adjacent | 0 | 6 / 4 / 0 / 2 |
| meter + two R/X | 1 / 0 / 0 / 1 | 12 / 0 / 2 / 10 |
| meter on the faulted line + R/X | 7 / 0 / 7 / 0 | 2 / 1 / 1 / 0 |
| control: 1 to 3 meters | 26 / 1 / 23 / 2 | 31 / 4 / 19 / 8 |
| all | 34 / 1 / 30 / 3 | 51 / 9 / 22 / 20 |

The paths show where the budget goes.  On a lost two-meter control the
40-action path typically runs as follows:

1. `run_wls, get_measurement_context, correct_measurements, run_wls,
   commit_state` fixes the first meter.
2. The expert then takes the parameter route:
   `get_parameter_context, compare_parameter_candidates`, then
   `correct_parameters, run_wls, rollback_state` once for each candidate line,
   two to five times.
3. `get_topology_context, get_measurement_context, correct_measurements,
   run_wls, commit_state` fixes the second meter.

A three-meter root pays the detour twice.  The 26 lost IEEE 118 meter
controls average 28.5 actions and 3.9 rolled-back corrections at 40 actions.
The truth probe takes the measurement route for each true meter and needs 12
actions for two meters and 16 for three (13.8 on average).  The meter-on-line
roots follow the same pattern with the parameter route first: 21.7 actions and
2.0 rollbacks, against 12 for the truth probe.

Why a single deviation does not recover them: at the state after the first
commit the menu is the parameter, topology and measurement contexts.  The
audit takes the measurement context; the expert, called again, goes on with
`compare_parameter_candidates`.  The expert re-chooses the route at every
call, so a one-action deviation that only fetches a context is undone at the
next call.  Wherever a decision recurs, the legal-alternative bin is a lower
bound on route-choice headroom.  The truth probe, which knows the true
targets, is the upper bound; a policy can approach it only by choosing the
measurement route again at each recurrence, from what it observes.

The control's other losses have different causes:

- The adjacent pairs' four legal alternatives replace the comparison at the
  first decision state with a direct correction of one of the two compared
  lines (the first on two roots, the second on two).  That order resolves the
  pair in 11 to 16 actions; the expert's order needs more than 16.
- Ten meter + two R/X roots need more than 16 actions even with the true
  targets.  The context refuses the true meter correction while the lines
  dominate, and the verifier accepts the two true line corrections only as
  partial.  The case then needs further passes that do not fit in 16 actions;
  the 40-action runs make them.
- The eight meter controls in "neither" are not route choices.  The context
  refuses the truth probe's joint meter correction.  The verifier then rejects
  one true single-meter correction, which is rolled back (the probe corrects
  the meters in a fixed order), so a true meter is still in place when the
  budget ends.

### 9.3 The deployed budget, audited wider

At 40 actions the four-by-three audit of the 169 IEEE 118 failures finds 10
legal alternatives, 4 interface limits and 155 evidence limits.  The two-by-two
audit of section 4 found 8, 5 and 156 on the same roots.  The two additions
are an adjacent pair and a meter + two R/X root.  Both are recovered at the
fourth audited state, where the expert compares two candidate lines and the
alternative corrects the second candidate directly.  Four of the five failed
meter controls are recovered by the measurement context after the first
commit: with 40 actions, the expert's continuation after it fits in the
budget.  No class changes character.  At the deployed budget the failures are
the evidence and interface limits of section 4, and the route-choice headroom
is 10 of 625 roots (1.6%).

### 9.4 What the route choice can see

After the first committed correction, while the alarm is still on, the
verifying `run_wls` reports a maximum normalized residual.  It is on the
agent's observation before the next action.  In the 40-action runs, grouped by
the committed correction and the next true correction the episode committed:

| committed, then next true correction | IEEE 118: n / median / at least 11 / at least 12 | IEEE 14 control: n / median / at least 11 / at least 12 |
| --- | --- | --- |
| meter, then meter | 32 / 19.4 / 32 / 32 | 39 / 15.5 / 37 / 32 |
| meter, then line | 19 / 9.6 / 4 / 1 | 36 / 7.4 / 4 / 3 |
| line, then line | 347 / 14.5 / | 287 / 9.5 / |
| line, then meter | 44 / 19.8 / | 64 / 17.3 / |

After a committed meter correction, the residual separates another bad meter
from a bad line with one threshold on both systems.  At 11 it catches 32 of 32
and 37 of 39 meter-then-meter states and admits 4 of 19 and 4 of 36
meter-then-line states.  After a committed line correction it does not
separate them (75% against 100% of the IEEE 118 states reach 10).  These are
the expert's own states, in sample, and conditional on its first commit being
a meter.

### 9.5 What this means for RL

- At the deployed 40 actions there is little route choice to learn: 10 of 625
  IEEE 118 roots.  The rest are evidence and interface limits, which the RL
  watchlist places outside what a policy can learn around.
- Under a 16-action budget there is headroom.  With the true targets, 30 of
  the 34 lost IEEE 118 roots fit in the budget (31 of 51 on the control, 9 of
  them by a legal alternative).  On IEEE 118 the loss is one recurring
  decision: after a committed meter correction with the alarm still on, take
  the measurement route or the parameter route.  Its cue is observable, and
  one threshold nearly separates it on both systems.  The first candidate is
  therefore a rule, not a policy.  A labelled rule probe would stay on the
  measurement route after a committed meter correction while the maximum
  normalized residual is at least 11.  It should run over both 625-root suites
  at 16 and 40 actions, counting what it recovers on the meter roots against
  what it costs where the next error is a line (meter + two R/X, meter on the
  faulted line).  It reads an observable cue and adds no action, so it is not
  a screen.  A learned router has a claim only on what such a rule leaves: the
  overlap band (7 to 11 on the control), the line-or-meter choice after a line
  commit, and the adjacent pairs' choice between comparing and correcting.
- A 12-action budget measures the interface, not decisions.  Budgeted RL
  experiments on these classes should use 16 or 20 actions.

The rule probe itself is section 10.

### 9.6 Reproduction

```
PSSE_MULTI_SCAN_RCOND=sparse_symmetric python -m research.compound_errors.rollout --suite output/compound_errors_ieee118_20261028/suite.json --output-dir output/compound_errors_ieee118_20261028/budget_16 --system case118 --expert baseline_compare --max-steps 16 --workers 9
python -m research.compound_errors.rollout --suite output/compound_errors_ieee14_busbranch_20261028/suite.json --output-dir output/compound_errors_ieee14_busbranch_20261028/budget_16 --expert baseline_compare --max-steps 16 --workers 4
PSSE_MULTI_SCAN_RCOND=sparse_symmetric python -m research.compound_errors.audit --suite output/compound_errors_ieee118_20261028/suite.json --episodes output/compound_errors_ieee118_20261028/budget_16/episodes.jsonl --reference-episodes output/compound_errors_ieee118_20261028/expert_baseline_compare/episodes.jsonl --output-dir output/compound_errors_ieee118_20261028/audit_budget16 --system case118 --expert baseline_compare --max-steps 16 --max-states 4 --max-alternatives 3 --solved-per-class 0 --seed 20261019 --workers 14
python -m research.compound_errors.audit --suite output/compound_errors_ieee14_busbranch_20261028/suite.json --episodes output/compound_errors_ieee14_busbranch_20261028/budget_16/episodes.jsonl --reference-episodes output/compound_errors_ieee14_busbranch_20261028/expert_baseline_compare/episodes.jsonl --output-dir output/compound_errors_ieee14_busbranch_20261028/audit_budget16 --expert baseline_compare --max-steps 16 --max-states 4 --max-alternatives 3 --solved-per-class 0 --seed 20261019 --workers 4
PSSE_MULTI_SCAN_RCOND=sparse_symmetric python -m research.compound_errors.audit --suite output/compound_errors_ieee118_20261028/suite.json --episodes output/compound_errors_ieee118_20261028/expert_baseline_compare/episodes.jsonl --output-dir output/compound_errors_ieee118_20261028/audit_wide40 --system case118 --expert baseline_compare --max-states 4 --max-alternatives 3 --solved-per-class 0 --seed 20261019 --workers 14
python -m research.compound_errors.budget_headroom --run ieee118:40=<D>/expert_baseline_compare/episodes.jsonl --run ieee118:12=<D>/budget_12/episodes.jsonl --run ieee118:16=<D>/budget_16/episodes.jsonl --run ieee118:20=<D>/budget_20/episodes.jsonl --run ieee14_control:40=<C>/expert_baseline_compare/episodes.jsonl --run ieee14_control:12=<C>/budget_12/episodes.jsonl --run ieee14_control:16=<C>/budget_16/episodes.jsonl --run ieee14_control:20=<C>/budget_20/episodes.jsonl --audit ieee118:16=<D>/audit_budget16/roots.jsonl --audit ieee14_control:16=<C>/audit_budget16/roots.jsonl --output-dir <D>/budget_headroom
```

The 12- and 20-action rollouts differ only in `--max-steps` and the output
folder (`budget_12/`, `budget_20/`).  `<D>` is
`output/compound_errors_ieee118_20261028`, `<C>` is
`output/compound_errors_ieee14_busbranch_20261028`.  Run times: the IEEE 118
rollouts took 30 to 36 min on 9 workers, the control's 11 to 13 min on 4, the
budget audits 17 min (IEEE 118, 14 workers) and 12 min (control, 4), and the
wide audit 82 min.

## 10. The meter-route rule probe (2026-10-29)

Section 9.5 proposed a labelled rule for the one recurring decision the
16-action budget loses on.  After a committed meter correction with the alarm
still on, the rule stays on the measurement route while the maximum
normalized residual is at least 11.  This section runs it.  The rule is an
opt-in option of the rule expert, `ExpertPolicyOracle(meter_route_residual=11)`,
run as `rollout.py --expert baseline_compare --expert-option
meter_route_residual=11`, and since then also selectable as the named
variant `baseline_compare_meter_route` (section 10.5).  `baseline_compare`
and the earlier named variants are unchanged.  The rule acts in a state where
two things hold: the latest
committed correction is a meter correction, and the active state's solve
reports a maximum normalized residual of at least 11.  There it changes two
things:

1. The meter route leads.  The measurement context, and the meter
   corrections it supports, go ahead of the parameter and topology tiers,
   which otherwise come first.
2. The measurement expert's post-commit branch guard stands down.  After a
   partial meter commit, that guard holds meter corrections back while a
   branch-multiplier signature is current and a branch route is untried.  The
   guard, not only the order, is what forced the parameter detour of
   section 9.2.  The branch-dominance suppression still applies: no meter
   correction while a branch multiplier dominates the solve and no residual
   does.

The rule reads only what the solve reports to the agent and adds no action,
so it is not a screen.  The runs use the suites, rollout seed and guard
settings of section 9.  `research/compound_errors/meter_route_probe.py` pairs
every root with `baseline_compare` on the same budget.  The tracked tables are
`meter_route_probe.{json,md}` and `*_meter_route11_{16,40}_summary.*` in
`docs/compound_errors_ieee118_20261028/`.  As a check that the option touches
nothing else: on every root where `baseline_compare`'s episode never reaches a
state the rule fires in, the two trajectories are identical action for
action.

### 10.1 Success

| class | IEEE 118, 16 actions: baseline_compare / rule | IEEE 118, 40 actions | IEEE 14 control, 16 actions | IEEE 14 control, 40 actions |
| --- | --- | --- | --- | --- |
| two R/X lines, adjacent (150) | 97 / 97 | 95 / 95 | 59 / 59 | 61 / 61 |
| two R/X lines, distant (150) | 130 / 130 | 130 / 130 | 115 / 115 | 115 / 115 |
| meter + two R/X (150) | 129 / 129 | 130 / 130 | 77 / 77 | 88 / 88 |
| meter on the faulted line + R/X (75) | 4 / 4 | 11 / 11 | 11 / 11 | 13 / 13 |
| control: one R/X line (50) | 45 / 45 | 45 / 45 | 48 / 48 | 48 / 48 |
| control: 1 to 3 meters (50) | 19 / 49 | 45 / 50 | 18 / 46 | 49 / 50 |
| all (625) | 424 / 454 | 456 / 461 | 328 / 356 | 374 / 375 |

- **No losses.** The rule loses no root in any of the four pairings (2,500
  paired episodes).  Every gain is a meter control.
- **16 actions.** The rule gains 30 roots on IEEE 118 and 28 on the
  control.  On IEEE 118 it solves 454 roots, two fewer than
  `baseline_compare` with 40 actions (456).  On the meter controls it beats
  that 40-action total: 49 against 45.
- **40 actions.** The rule gains 5 roots on IEEE 118 and 1 on the control,
  and solves all 50 meter controls on both systems.  The six roots
  `baseline_compare` fails at 40 are all three-meter controls.  On each it
  commits two true meters, tries and rolls back seven or eight line
  corrections, and hands off with `recovery_budget_exhausted` after 38
  actions.  The rule solves each in 15 actions.

Against the decision audit of section 9.2 (the roots the 16-action budget
loses):

- **IEEE 118:** the rule recovers 25 of the 26 lost meter controls within 16
  actions:
  - the 23 that only the truth probe recovered;
  - the one with a legal alternative;
  - one of the two in "neither".  There the truth probe's fixed meter order
    failed and the expert's own order does not.
- **IEEE 14 control:** the rule recovers 27 of 31: 17 of the 19 truth-only
  roots, all 4 legal ones and 6 of the 8 in "neither".
- **Meter on the faulted line:** the lost roots stay lost (7 on IEEE 118, 2
  on the control).  Their detour is the parameter route, taken before any
  meter commit.  The rule does not touch that decision.

### 10.2 What is left on the meter controls

The rule still fails five meter controls at 16 actions, 1 on IEEE 118 and 4
on the control; all five solve at 40.  In each, a true meter commit leaves a
maximum normalized residual between 10.3 and 11.0, just under the threshold,
with another bad meter still in place.  The rule stays quiet, the expert
takes the parameter route, and the budget ends.  These five sit in the
overlap band of section 9.4.

### 10.3 What the rule costs

- **Actions.** On the meter controls both teachers solve with 40 actions,
  the mean falls from 19.8 to 10.8 actions on IEEE 118 and from 21.3 to 12.4
  on the control.  Over all classes, budget handoffs at 40 fall from 22 to 12
  and from 32 to 21; on the meter controls, from 7 to 0 and from 7 to 1.
- **Other classes.** Outside the meter controls the rule changes 26 IEEE 118
  and 23 control trajectories at 40 actions: adjacent pairs, the two
  meter-and-line classes, and one distant pair on the control.  No outcome
  changes, and in total those episodes are shorter (by 60 and 30 actions).
- **Committed wrong meters.** On IEEE 118 at 40 actions they rise from 48 to
  54, on 4 roots; on the control from 49 to 57, on 6 roots.  At 16 actions
  they rise from 5 to 6 and from 9 to 10, on one root each.  Every one of
  these roots fails under both teachers.  This is the masking the lifted
  guard admits.  After a wrong meter commit that hides a bad line (mostly
  the meter-on-line class), the residual still reads as meter-led, and the
  rule keeps editing meters.  No outcome changes, but these failures end
  with more healthy meters edited.

### 10.4 The cue in the states the rule acted on

| rule decisions, by the meter commit before them | IEEE 118, 16 actions | IEEE 118, 40 actions | IEEE 14 control, 16 actions | IEEE 14 control, 40 actions |
| --- | --- | --- | --- | --- |
| after a true meter commit: next true correction meter / line / none | 50 / 4 / 5 | 50 / 7 / 4 | 47 / 3 / 2 | 47 / 4 / 6 |
| after a wrong meter commit: next true correction meter / line / none | 0 / 0 / 4 | 6 / 0 / 22 | 2 / 0 / 3 | 5 / 0 / 26 |

- **After a true meter commit.** The next true correction is a meter in 47
  to 50 of the 52 to 61 states where the rule decides, and a line in 3 to 7.
  That is close to the in-sample separation of section 9.4, now on states
  the rule itself led the expert into.  When a line comes next, the rule
  spends the measurement context, and any meter candidates it offers, before
  the line route resumes.  None of those roots was lost.
- **After a wrong meter commit.** Every decision the rule takes there is on
  a root that fails under both teachers.

### 10.5 What this means for RL

- **The recurring decision of section 9 needs no learning.** A
  one-threshold rule on an observable cue settles it:
  - it recovers 25 of the 26 IEEE 118 meter controls the 16-action budget
    lost, and 27 of 31 on the control;
  - it loses no root on either system;
  - it brings the 16-action IEEE 118 total within 2 roots of the deployed
    40-action total.
- **In sample by root.** The threshold of 11 was read off these same two
  suites (section 9.4, from `baseline_compare`'s own states).  The states the
  rule decides in are new, but the roots are not.  Section 11 tests the
  threshold on a fresh draw: the gains hold, and the rule loses one root.
- **What remains for a learned router is smaller:**
  1. The threshold band from 10.3 to 11: five meter controls at 16 actions.
  2. The first route on meter-on-line roots, before any commit.  The truth
     probe recovers 7 IEEE 118 roots at 16 actions; the rule recovers none.
  3. The adjacent pairs' order, compare first or correct first: 4 control
     roots at 16 actions.
  4. When to leave the meter route after a wrong meter commit, the rule's
     only cost.  This is the credit-assignment problem of hypothesis
     abandonment (RL watchlist item 1), not a threshold.
- **The named variant.** After this probe the rule was registered as the
  teacher variant `baseline_compare_meter_route` (`psse_env/oracle/expert_variants.py`,
  `baseline_compare` plus `meter_route_residual=11`).  The HPC pipeline's
  `EXPERT_VARIANT` accepts it, so a cell or a DAgger collection can select it
  by name.  `baseline_compare` is unchanged, and no existing cell names the
  new variant.  No cell has run it yet.

### 10.6 Reproduction

```
PSSE_MULTI_SCAN_RCOND=sparse_symmetric python -m research.compound_errors.rollout --suite output/compound_errors_ieee118_20261028/suite.json --output-dir output/compound_errors_ieee118_20261028/meter_route11_40 --expert baseline_compare --expert-option meter_route_residual=11 --max-steps 40 --workers 11
python -m research.compound_errors.rollout --suite output/compound_errors_ieee14_busbranch_20261028/suite.json --output-dir output/compound_errors_ieee14_busbranch_20261028/meter_route11_40 --expert baseline_compare --expert-option meter_route_residual=11 --max-steps 40 --workers 5
python -m research.compound_errors.meter_route_probe --threshold 11 --pair ieee118:40=<D>/expert_baseline_compare/episodes.jsonl,<D>/meter_route11_40/episodes.jsonl --pair ieee118:16=<D>/budget_16/episodes.jsonl,<D>/meter_route11_16/episodes.jsonl --pair ieee14_control:40=<C>/expert_baseline_compare/episodes.jsonl,<C>/meter_route11_40/episodes.jsonl --pair ieee14_control:16=<C>/budget_16/episodes.jsonl,<C>/meter_route11_16/episodes.jsonl --output-dir <D>/meter_route_probe
```

The 16-action rollouts differ only in `--max-steps 16`, the output folder
(`meter_route11_16/`) and the worker count (9 and 4).  `<D>` and `<C>` are as
in section 9.6.  All four rollouts ran at once on 32 cores.  The IEEE 118
runs took 29 min at 40 actions (11 workers) and 31 min at 16 (9 workers);
the control's took 12 and 13 min (5 and 4 workers).

With the named variant, `--expert baseline_compare_meter_route` replaces
`--expert baseline_compare --expert-option meter_route_residual=11`.  As a
check, the variant was rolled out at 16 actions on two subsets: the first six
meter controls and six meter-on-line roots of IEEE 118, and the 275 control
roots of the three classes with a bad meter.  Its trajectories match the
option run's on all of them, action for action and correction for correction
(`variant_check_16/` in both output folders).

## 11. The meter-route rule on a fresh draw (2026-10-29)

Section 10.5 left the threshold in sample by root: 11 was read off the suites
the rule was then scored on.  This section draws both suites again and
repeats section 10's pairing, with nothing else changed:

- **The draw.** Section 8's two `build_suite.py` commands with suite seed
  20261029 instead of 20261027.
- **The runs.** `baseline_compare` and the named variant
  `baseline_compare_meter_route` (threshold 11), `scada_only`, 16 and 40
  actions, rollout seed 20261006, the IEEE 118 runs with the sparse guard.
  The variant reproduces the option runs action for action (section 10.6).
- **Nothing tuned.** Nothing was tuned on the fresh draw.

`meter_route_probe.py` pairs the roots as before.  It now also reports three
more things:

- whether each committed meter correction was true;
- section 9.4's separation on `baseline_compare`'s own states;
- the check that the rule changes no root whose `baseline_compare` episode
  never reaches a firing state.

With `--reference`, it sets the in-sample probe's counts beside the fresh
ones.

### 11.1 The fresh draw

Each class shard's generator is seeded from a hash of the suite seed, the
class name and the shard, so the two draws share no generator stream.
`draw_overlap.py` confirms it class by class: on neither system do they share
a generator seed, a scenario id or a root (every atom and the load scale).

Repeated targets at other magnitudes are common only where targets are few:

- **IEEE 118:** 86 of the 625 fresh roots repeat an in-sample root's lines
  and meters, 57 of them adjacent pairs.
- **IEEE 14 control:** 409 of 625 do, since the control has only 20
  branches.
- **Meter controls**, where every in-sample gain was: 0 of 50 IEEE 118 roots
  and 3 of 50 control roots repeat an in-sample meter set.

The fresh suites took 1,195 draws on IEEE 118 (1,325 in sample) and 1,007 on
the control (1,052).  Section 2's profile changes little:

- the strata of the deployed ranking move by at most 17 roots in a class:
  the IEEE 118 meter + two R/X class has 90 dominant and 49 ambiguous
  rankings, against 104 and 32 in sample;
- no IEEE 118 root has a silent atom;
- 5 control roots have one, against 6 in sample.

The composition of the meter controls changes more:

| meter controls with 1 / 2 / 3 bad meters | in sample | fresh |
| --- | --- | --- |
| IEEE 118 | 17 / 15 / 18 | 16 / 17 / 17 |
| IEEE 14 control | 13 / 19 / 18 | 18 / 22 / 10 |

### 11.2 Success, against section 10.1

Each cell is `baseline_compare` / rule, with the roots gained and lost where
there are any.

| IEEE 118 | 16 actions, in sample | 16 actions, fresh | 40 actions, in sample | 40 actions, fresh |
| --- | --- | --- | --- | --- |
| two R/X lines, adjacent (150) | 97 / 97 | 99 / 99 | 95 / 95 | 101 / 101 |
| two R/X lines, distant (150) | 130 / 130 | 125 / 125 | 130 / 130 | 125 / 125 |
| meter + two R/X (150) | 129 / 129 | 114 / 114 | 130 / 130 | 117 / 117 |
| meter on the faulted line + R/X (75) | 4 / 4 | 5 / 5 | 11 / 11 | 8 / 8 |
| control: one R/X line (50) | 45 / 45 | 44 / 44 | 45 / 45 | 44 / 44 |
| control: 1 to 3 meters (50) | 19 / 49 (+30) | 18 / 50 (+32) | 45 / 50 (+5) | 46 / 50 (+4) |
| all (625) | 424 / 454 (+30, -0) | 405 / 437 (+32, -0) | 456 / 461 (+5, -0) | 441 / 445 (+4, -0) |

| IEEE 14 control | 16 actions, in sample | 16 actions, fresh | 40 actions, in sample | 40 actions, fresh |
| --- | --- | --- | --- | --- |
| two R/X lines, adjacent (150) | 59 / 59 | 56 / 56 | 61 / 61 | 60 / 60 |
| two R/X lines, distant (150) | 115 / 115 | 117 / 117 | 115 / 115 | 117 / 117 |
| meter + two R/X (150) | 77 / 77 | 77 / 76 (-1) | 88 / 88 | 88 / 87 (-1) |
| meter on the faulted line + R/X (75) | 11 / 11 | 16 / 16 | 13 / 13 | 17 / 17 |
| control: one R/X line (50) | 48 / 48 | 48 / 48 | 48 / 48 | 48 / 48 |
| control: 1 to 3 meters (50) | 18 / 46 (+28) | 27 / 49 (+22) | 49 / 50 (+1) | 49 / 50 (+1) |
| all (625) | 328 / 356 (+28, -0) | 341 / 362 (+22, -1) | 374 / 375 (+1, -0) | 379 / 379 (+1, -1) |

| meter controls at 16 actions, n: `baseline_compare` / rule | 1 meter | 2 meters | 3 meters |
| --- | --- | --- | --- |
| IEEE 118, in sample | 17: 17 / 17 | 15: 2 / 15 | 18: 0 / 17 |
| IEEE 118, fresh | 16: 16 / 16 | 17: 2 / 17 | 17: 0 / 17 |
| IEEE 14 control, in sample | 13: 13 / 13 | 19: 3 / 19 | 18: 2 / 14 |
| IEEE 14 control, fresh | 18: 18 / 18 | 22: 9 / 21 | 10: 0 / 10 |

- **The gains hold.** At 16 actions the rule gains 32 IEEE 118 roots (30 in
  sample) and 22 control roots (28).  At 40 actions it gains 4 and 1 (5 and
  1).  Every gain is a meter control.  On IEEE 118 the rule solves all 50
  meter controls at both budgets.  Its 16-action IEEE 118 total (437) is 4
  roots short of `baseline_compare`'s at 40 actions (441); in sample it was 2
  short.
- **Per multi-meter root, the rule does as well as in sample.** Both
  teachers solve the one-meter controls.  On the controls with two or three
  bad meters, at 16 actions, the rule solves 34 of 34 on IEEE 118
  (`baseline_compare` 2) and 31 of 32 on the control (`baseline_compare` 9).
  In sample it solved 32 of 33 (2) and 33 of 37 (5).  The control gains
  fewer roots than in sample for two reasons.  Its fresh draw has 10
  three-meter controls against 18.  And `baseline_compare` already solves 9
  of the 22 two-meter roots, against 3 of 19 in sample.
- **One root lost.** On the control, the meter + two R/X root
  `r0_038f9574c00a` is solved by `baseline_compare` and lost by the rule, at
  both budgets.  That is 2 losses in the 2,500 fresh paired episodes, against
  none in the 2,500 in-sample ones.  Section 11.3 follows it.
- **The one meter control still failed.** The rule fails one fresh control
  meter control at 16 actions.  Its residual is 10.6 after a true meter
  commit, the threshold band of section 10.2.
- **`baseline_compare` itself.** It solves 441 fresh IEEE 118 roots at 40
  actions, against 456 in sample.  Most of the gap is the meter + two R/X
  class (117 against 130), whose dominant and ambiguous rankings both solve
  less often, and the distant pairs (125 against 130).  This was not
  examined further.  The
  pairing compares the two teachers root by root, so it does not enter the
  gains.

### 11.3 The root the rule loses

`r0_038f9574c00a` has three errors:

- R/X errors on line 1 (buses 1 to 2) and line 17 (buses 9 to 14), two hops
  apart;
- a reactive-flow meter on line 18, 25.9 sigma off.

Both teachers start the same way:

1. They correct line 1 and the meter and commit each, partially accepted.
2. After the meter commit, the solve reports a maximum normalized residual
   of 14.8.  The rule fires, although the remaining error is a line.  Section
   9.4 found this case, a meter-then-line state above the threshold, after 4
   of 36 first commits on the in-sample control.  Here the meter commit is
   the episode's second.

Then they part:

- **`baseline_compare`** takes the parameter route.  It corrects line 17 and
  resolves the state with its 13th action.
- **The rule** takes the measurement route.
  1. The context offers the voltage meter at bus 14, a terminal of line 17.
     The synthesizer perturbs only P and Q meters, so in these suites a
     voltage meter is never a true target.
  2. The verifier accepts that correction as partial: the chi-square ratio
     falls from 3.60 to 1.91, the maximum normalized residual to 6.8.  It is
     committed.
  3. The edit absorbs part of line 17's signature.  The true correction of
     line 17 is then rejected (residual 10.8 against 6.8), and so is the next
     candidate, line 20.
  4. The episode hands off with a healthy meter edited, after 15 actions at
     16 and after 24 at 40.

Run alone, each teacher repeats its trajectory at both budgets.  This is the
masking section 10.3 named, a healthy meter edit that hides a bad line.  In
sample it fell only on roots both teachers fail; here it costs a root.  The
healthy meter is a voltage meter at a terminal of the bad line, the
neighborhood of section 5.2's same-line meter problem.

### 11.4 The cue out of sample

Section 9.4's separation is the residual after `baseline_compare`'s first
committed correction, at 40 actions.  On the fresh draw:

| first commit, then next true correction: n / median / at least 11 | IEEE 118, in sample | IEEE 118, fresh | IEEE 14 control, in sample | IEEE 14 control, fresh |
| --- | --- | --- | --- | --- |
| meter, then meter | 32 / 19.4 / 32 | 34 / 19.0 / 34 | 39 / 15.5 / 37 | 33 / 13.8 / 28 |
| meter, then line | 19 / 9.6 / 4 | 28 / 7.5 / 3 | 36 / 7.4 / 4 | 52 / 7.0 / 5 |

- **IEEE 118.** The threshold separates the fresh states as cleanly as the
  in-sample ones.  All 34 meter-then-meter states reach 11, and 3 of 28
  meter-then-line states do (4 of 19 in sample).
- **IEEE 14 control.** It admits as few line-next states (5 of 52) but
  catches fewer meter-next ones: 28 of 33, against 37 of 39 in sample.
  - The five below 11 are all two-meter controls.  The fresh two-meter
    controls sit lower after the first commit (median 13.3 against 15.2).
  - Below the threshold, `baseline_compare`'s own route still finishes four
    of the five within 16 actions.
  - The fifth, at 10.6, is the meter control of section 11.2 that both
    teachers fail.

The rule's own decisions, in the view of section 10.4:

| rule decisions, by the meter commit before them: next true correction meter / line / none | IEEE 118, 16 actions: in sample, fresh | IEEE 118, 40 actions | IEEE 14 control, 16 actions | IEEE 14 control, 40 actions |
| --- | --- | --- | --- | --- |
| after a true meter commit | 50 / 4 / 5, 51 / 1 / 5 | 50 / 7 / 4, 51 / 6 / 7 | 47 / 3 / 2, 35 / 6 / 3 | 47 / 4 / 6, 35 / 7 / 7 |
| after a healthy meter commit | 0 / 0 / 4, 0 / 0 / 6 | 6 / 0 / 22, 1 / 0 / 30 | 2 / 0 / 3, 1 / 0 / 9 | 5 / 0 / 26, 2 / 2 / 24 |

- **After a true meter commit, IEEE 118.** The next true correction is a
  meter in 51 of 57 and 51 of 64 fresh decisions (50 of 59 and 50 of 61 in
  sample).
- **After a true meter commit, IEEE 14 control.** It is a meter in 35 of 44
  and 35 of 49 fresh decisions, against 47 of 52 and 47 of 57 in sample.
  The control's fresh draw offers fewer meter-after-meter states (it has
  fewer three-meter roots) and a few more line-after-meter ones (6 and 7,
  against 3 and 4).
- **When a line came next** in the rule's own episode, the root was not
  lost: the line route resumed after the measurement context, as in sample.
  The lost root is a "none" decision: after its fired meter commit, the
  rule's episode commits no further true correction.

### 11.5 What the rule costs

- **Actions.** On the meter controls both teachers solve at 40 actions, the
  mean falls from 20.4 to 10.7 actions on IEEE 118 (19.8 to 10.8 in sample)
  and from 16.7 to 10.8 on the control (21.3 to 12.4).  Over all classes,
  budget handoffs at 40 actions fall from 31 to 17 and from 17 to 13; on the
  meter controls, from 9 to 0 and from 2 to 0.
- **Other classes.** At 40 actions the rule changes 27 IEEE 118 trajectories
  outside the meter controls: 21 meter-on-line roots, 5 meter + two R/X roots
  and 1 distant pair, 33 actions fewer in total.  On the control it changes
  29: 13 meter-on-line roots, 8 meter + two R/X roots and 8 adjacent pairs,
  54 actions fewer.  In sample the counts were 26 and 23.  The only outcome
  that changes is the lost root.
- **Committed wrong meters.** At 40 actions they rise from 52 to 58 on IEEE
  118 (on 5 roots) and from 59 to 65 on the control (5 roots).  At 16 they
  rise from 11 to 12 (1 root) and from 18 to 23 (4 roots).  Except the lost
  root, every one of these roots fails under both teachers.
- **The same-trajectory check.** Take the roots where `baseline_compare`
  never reaches a state the rule fires in (559 to 581 per run).  In three of
  the four fresh pairings the rule's trajectories on them are identical.
  The exception is the control at 16 actions, on the adjacent pair
  `r0_a590aacb3bc1`, where no meter is ever committed, so the rule cannot
  fire:
  - The two runs part at the second line correction: the
    `baseline_compare` run tries line 5 before line 3, the rule run line 3.
  - Run alone, `baseline_compare` takes line 3 three times out of three.
    So the difference is in the `baseline_compare` run, not in the rule.
  - Both teachers solve the root, so no count changes.
  - Section 11.8 finds the cause: a race between concurrent runs in the
    provider's derived-case cache, not state carried between roots.

### 11.6 What this means

- **The gain holds out of sample.** The threshold of 11, read off the
  20261027 suites, recovers the same kind of root on a disjoint draw.  At 16
  actions it gains 32 on IEEE 118 and 22 on the control (30 and 28 in
  sample).  Every gain is a meter control with two or three bad meters, and
  on IEEE 118 the fresh states separate as cleanly as the in-sample ones.
- **The no-loss result does not.** On a fresh control root, the rule commits
  a healthy meter at a terminal of a bad line.  `baseline_compare` would
  correct the line there.  Over both draws and both budgets, the rule wins
  123 of the 5,000 paired episodes and loses 2.
- **For the named variant.** `baseline_compare_meter_route` is still the
  better teacher on these classes, but not a strict improvement.  A DAgger
  collection under it would also teach that step.
- **For RL.** The residue of section 10.5 stands:
  1. the threshold band, which leaves one fresh control meter control at
     10.6;
  2. the first route on meter-on-line roots;
  3. the adjacent pairs' order.

  Item 4, the rule's cost, now loses a root.  On that root the rule leaves
  the meter route by itself: the residual falls to 6.8 after the healthy
  meter commit.  But the committed edit stays and masks line 17.  Avoiding
  the loss means refusing the partial meter candidate at the terminal of a
  line that may be bad.  Recovering from it means withdrawing a committed
  correction, the hypothesis-abandonment decision of RL watchlist item 1.
- **One trajectory was not reproducible.** Section 11.8 traces it to a
  race between concurrent runs on Windows.  Re-run in isolation, every other
  episode of the fresh draw reproduces, and so does every in-sample episode
  where the rule fires.  So no count of sections 10 and 11 changes.  The
  cache has since been fixed (section 11.8).

### 11.7 Reproduction

```
python -m research.compound_errors.build_suite --system case118 --output-dir output/compound_errors_ieee118_20261029 --seed 20261029 --shards 10 --workers 26
python -m research.compound_errors.build_suite --system case14 --physics bus_branch --distant-hops 2 --output-dir output/compound_errors_ieee14_busbranch_20261029 --seed 20261029 --shards 10 --workers 4
python -m research.compound_errors.draw_overlap --pair ieee118=<F>,<D> --pair ieee14_control=<G>,<C> --output-dir <F>/draw_overlap
PSSE_MULTI_SCAN_RCOND=sparse_symmetric python -m research.compound_errors.rollout --suite <F>/suite.json --output-dir <F>/baseline_compare_40 --system case118 --expert baseline_compare --max-steps 40 --workers 7
PSSE_MULTI_SCAN_RCOND=sparse_symmetric python -m research.compound_errors.rollout --suite <F>/suite.json --output-dir <F>/meter_route_40 --system case118 --expert baseline_compare_meter_route --max-steps 40 --workers 7
python -m research.compound_errors.rollout --suite <G>/suite.json --output-dir <G>/baseline_compare_40 --expert baseline_compare --max-steps 40 --workers 3
python -m research.compound_errors.join_runs --suite <F>/suite.json --part <F>/meter_route_16/episodes_head.jsonl --part <F>/meter_route_16_rest/episodes.jsonl --metadata <F>/meter_route_16_rest/summary.json --output-dir <F>/meter_route_16
python -m research.compound_errors.meter_route_probe --threshold 11 --pair ieee118:40=<F>/baseline_compare_40/episodes.jsonl,<F>/meter_route_40/episodes.jsonl --pair ieee118:16=<F>/baseline_compare_16/episodes.jsonl,<F>/meter_route_16/episodes.jsonl --pair ieee14_control:40=<G>/baseline_compare_40/episodes.jsonl,<G>/meter_route_40/episodes.jsonl --pair ieee14_control:16=<G>/baseline_compare_16/episodes.jsonl,<G>/meter_route_16/episodes.jsonl --reference docs/compound_errors_ieee118_20261028/meter_route_probe.json --output-dir <F>/meter_route_probe
```

`<F>` is `output/compound_errors_ieee118_20261029` and `<G>` is
`output/compound_errors_ieee14_busbranch_20261029`; `<D>` and `<C>` are as in
section 9.6.  The other rollouts differ from those shown only in the expert,
`--max-steps 16` and the folder (`baseline_compare_16/`, `meter_route_16/`),
with 6 workers for the IEEE 118 16-action runs.

Every rollout ran in two parts:

- **IEEE 14 control.** The four runs started on one worker each and were
  stopped after 93 to 118 roots.  `rollout.py --roots` ran the rest on three
  workers (`*_rest/`).
- **IEEE 118.** A restart of the machine stopped the four runs after 471 to
  506 roots, and the rest ran on six workers.

`join_runs.py` joins each run's first part, kept as `episodes_head.jsonl`,
with its rest into the run's `episodes.jsonl`, in suite order.  It also
rebuilds `summary.{json,md}` over all 625 episodes.

Run times:

- the two suites, built at once: 21 min (IEEE 118) and 10 min (control);
- the control's first parts: 20 min; its rests: 16 to 20 min;
- the IEEE 118 first parts: 50 min, beside the control's; their rests: 8 to
  17 min.

The tracked outputs are in `docs/compound_errors_ieee118_20261028/fresh_20261029/`:
`draw_overlap.*`, `meter_route_probe.*`, the two suite manifests, and the eight
joined run summaries `{ieee118,ieee14_busbranch}_{baseline_compare,meter_route}_{16,40}_summary.*`.
The in-sample `meter_route_probe.*` in the folder above was regenerated to
carry the probe's new tables; its earlier figures are unchanged.

### 11.8 Why one trajectory was not reproducible

On `r0_a590aacb3bc1` at 16 actions, the `baseline_compare` run tried line 5
before line 3, and every rerun took line 3 (section 11.5).  The cause is a
race between concurrent runs in the provider's derived-case cache.  State
carried between roots is ruled out:

- **Not history.** A replay of the first 32 roots of that run, in one
  process and in the same order, reproduces roots 1 to 31.  On root 32 it
  takes line 3.
- **Not hashing.** Six standalone runs under different `PYTHONHASHSEED`
  values all take line 3.
- **Not a near tie.** Standalone, the comparison before that correction
  ranks line 3 first by a wide margin.  It accepts line 3 as partial, with
  global progress 0.92 and a maximum normalized residual of 4.0.  Line 5 is
  rejected (progress 0.14), and line 6's correction fails.  For the run to
  try line 5 first, its comparison must have reported line 3's fit as
  failed, or as worse than line 5's.

**The race:**

1. Every correction writes the corrected case to a content-addressed file
   (`MatpowerDeploymentProviders._derived_case`).  The folder,
   `%TEMP%\psse_derived_cases`, is shared by every run on the machine.
2. It writes a sibling file, then replaces the target with `os.replace`.
3. On Windows that replace fails with `PermissionError` while another
   process holds the target open.  A read can fail the same way while the
   file is being replaced.
4. A comparison records such an exception as a failed candidate
   (`dispatch_error`), and the expert moves on to the next line.

Stressing the real `_derived_case` from 2 and 4 processes that write the same
new files raises `PermissionError` on 25% and 55% of the calls.  The two
16-action control first parts were the most exposed setting.  Each ran on one
worker, on the same roots in the same order, in lockstep (section 11.7).

**Why the record did not show it.** The rollout's slim trace looks for the
comparison under key names it does not use (`best_line`, `ranked_lines`,
`verdicts`).  So no comparison verdict is recorded, failed or not.

**What it changed.** Each run was re-run with a private temp folder, so no
two runs share a derived-case file, and compared root by root with the
recorded run:

| re-run in isolation | episodes | trajectories that differ | outcomes that differ |
| --- | --- | --- | --- |
| fresh draw: all eight runs, all roots | 5,000 | 1 (`r0_a590aacb3bc1`, control, 16 actions, `baseline_compare`) | 0 |
| in sample: both teachers at both budgets, on the roots where the rule fires in either teacher's episode | 430 | 0 | 0 |

In sample, the roots where the rule never fires were already identical
between the `baseline_compare` and rule runs, which ran a day apart.  That is
2,285 paired episodes over both systems and budgets, section 11.5's check.
So no count of sections 10 and 11 changes.

**Exposure:**

- **Windows only.** POSIX `os.replace` replaces an open file, so the HPC
  runs are not exposed.
- **On this machine,** the race can strike two runs on the same suite at
  once.  It needs both to write a file that neither, nor any earlier run, has
  written yet.  One run alone is safe, since its workers hold different roots.
- **The fix** came after this section's measurements, so they ran on the
  old cache.  `_derived_case` now accepts a target that already holds the
  same bytes, since the path is content-addressed.  It publishes a new case
  with `os.rename`, which on Windows fails on an existing target instead of
  replacing a file another process may be reading.  Only stale bytes are
  replaced.  A sharing refusal of the read or the publish is retried for up
  to about 3 s.
- **What the fix removes.** Under the stress test, failures fall from 25%
  and 55% of the calls (2 and 4 writers) to none with 2, 4 and 8 writers,
  readers included.  A four-process test (`DerivedCaseConcurrencyTests` in
  `psse_env/providers/test_matpower_providers.py`) fails on the old writer
  and passes on the new one.
- **The exposed setting, re-checked.** Four single-worker runs, two per
  teacher at 16 actions, ran in lockstep on the first 40 control roots,
  sharing one fresh temp folder (155 new case files).  All four reproduce
  the isolated re-runs on every root, `r0_a590aacb3bc1` included.
- **The trace.** The rollout's slim trace now records the comparison's
  ranking and each candidate's status and error.

The isolated re-runs are `rerun_isolated/` in both fresh folders and
`rerun_isolated_fired/` in both in-sample folders.
