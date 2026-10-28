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
  families too and so the user's call.
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
