# Sparse multi-error scenarios with cluster decomposition: design, and a separability probe (2026-10-28)

The compound-error study (`docs/compound_errors_study_20261027.md`) drew fixed pairs of errors on IEEE 14.
The next step the user asked for draws a variable number of errors per scenario on larger systems,
exploits that errors are sparse and mostly localized, groups the evidence of one WLS solve into clusters
and corrects the clusters one after another or side by side.  This note designs that step and grounds its
key assumption, that the evidence separates into clusters, with a CPU probe on data that exists.  Nothing
here changes a contract: no provider, oracle, verifier or audit code was touched.  The design is a
proposal for the user to accept, change or drop (section 12).  Research only; journal scope; the PES GM
manuscript and everything under `docs/papers/` are untouched.

In one paragraph.  Separability is a function of the bus-graph distance between two errors, not of the
system's size: a pair at matched effect ratios never separates within one hop, separates a third of the
time at three hops and almost always from five (IEEE 118: 0, 8, 33, 67, 88 and 98% at 0-1, 2, 3, 4, 5
and 6+ hops; IEEE 14: 22, 48 and 78% at 2, 3 and 4 hops, and its diameter is 5).  What the size decides
is how often random errors land that far apart.  So random pairs separate on 11% of IEEE 14 draws and on
67% of IEEE 118 draws, triples on 0.7% and 27% (16.8%, 72.8%, 1.2% and 33.4% with the line flag at 4,
the level this note recommends), and a noisy solve recovers the partition on 8% and 63% of pairs.  At
one and two errors per hundred branches on IEEE 118, 61% and 37% of the atoms sit alone in their cluster;
at four (eight errors, median) only 13% do, which is where "sparse" ends.  The cluster's own ranking puts
a line error's line first on 80% of IEEE 118 pairs against 60% for the global list.  With per-atom
admission at effect ratio 1.5 no atom goes silent after the others are fixed (99.8% or more still alarm
on IEEE 118; on the IEEE 14 suite, which has no per-atom admission, 121 of 150 distant pairs carry an
atom below the alarm alone, and the probe's linear model counts 119).  The dilution that matters on a
large system is in the verifier, not in the detector: the deployed partial-progress rule keeps 48% of
the correct corrections of an isolated error in an IEEE 118 pair, 24% in a triple.  A cluster-local
progress floor would keep them all but would also keep the best wrong line on 55% and the best wrong
meter on 40% of those clusters (stage-2 item 4's lesson in another form), so the verification change
this note proposes is not a local floor but cluster resolution: a correction is kept when its cluster's
own chi-square goes quiet and the global statistic does not rise (99% of correct corrections kept), and
when the cluster's candidates are compared on cloned states only the one leaving the lowest local
statistic may resolve it: 97% of correct corrections and 2.3% of the best wrong lines kept (no wrong
meter), against the deployed rule's 48% and 7.6% on the same IEEE 118 pairs.  The cluster tool is worth
building on IEEE 118 as a read-only context first (stage S2, about a day of work, CPU only), and not on
IEEE 14.

Branch `claude/agitated-sammet-93f94e` (from `claude/nostalgic-kalam-be5f7a` a52a7e5; the compound branch
has since added item 6, the clean-flip default and its stage summary, 6ff5293, b546d2d and 6e5c478, whose
numbers section 1 cites; the probe needs none of their code).  Code:
`research/compound_errors/sparse_probe.py` (the probe), `research/compound_errors/sparse_report.py` (the
tables).  Outputs (gitignored, like every study's): `output/sparse_multi_error_probe_20261028/` (line flag
3, the deployed lambda threshold) and `output/sparse_multi_error_probe_20261028/lambda4/` (line flag 4):
`summary.json`, `report.md`, one `combos_<system>_<set>.jsonl` per draw set, `footprints_<system>.jsonl`,
`suite14_natural.jsonl`.  T1 to T10 below are the tables of `report.md`.  To reproduce (about 6 minutes
per configuration on 7 workers):

    python -m research.compound_errors.sparse_probe --suite output/compound_errors_20261027/suite.json \
        --ieee118-dir output/classifier_triage_20261012/ieee118_effect \
        --output-dir output/sparse_multi_error_probe_20261028 --workers 7 [--lambda-flag 4]
    python -m research.compound_errors.sparse_report --probe output/sparse_multi_error_probe_20261028 \
        --variant output/sparse_multi_error_probe_20261028/lambda4 \
        --output output/sparse_multi_error_probe_20261028/report.md

The defaults are the run's settings (seed 20261028, 8 states, 1,200 pairs and triples, 40 noise draws).

## 1. What the earlier work established

The design answers these results of the compound study (stage 1 and stage 2, IEEE 14, 1,275 roots):

- After the stage-2 tool repairs (the node/breaker re-render binding, the isolated-line rule, the
  adjacent pair refit) the rule expert solves 617 of 1,275 fixed-pair roots under `baseline_compare`
  (48%), 789 and 788 under the always-on and guarded topology-first orders, and 700 (55%, +83 / -0
  against `baseline_compare`) under the clean-flip order, the one that passed the independent check on the
  conventional suites and became the default teacher for new work (`baseline_compare_topology_clean_flip`,
  commit b546d2d; the compound note's section 7.8 at 6e5c478).  Its screen costs 9 node/breaker solves
  per anomalous state, 23.1 per compound episode against 5.1.
- A second error often goes silent after the first fix: 121 of the 150 distant line pairs carry a line
  whose own noiseless chi-square is below the alarm, 108 have one line unranked, and after an
  `ACCEPT_FINAL` commit the post-commit confirmation gate refused the second correction on 57 distant
  and 19 adjacent pairs (sections 5.3, 6.1).
- Relaxing the global progress floor admits wrong corrections: a floor divided by the number of open
  faults would rescue 24 true corrections and admit 106 + 49 wrong ones (section 6.2 item 4); the
  target-first partial acceptance of item 4 lost 147, 40 and 6 roots under the three teachers and raised
  the wrong partial acceptances from 366 to 641, because a wrong neighbouring refit clears its own
  multiplier while a breaker fault stands (section 7.6).
- Multi-row candidates with a declared partner can be verified: the adjacent pair refit (item 3,
  provider option `pair_refit_adjacent`) moved the adjacent class from 50 to 76 of 150 with the
  candidate-quality oracle accepting the declared partner row (section 7.5).
- Every root carries parameter scans and the node/breaker channel (the uniform-metadata convention, user
  decision of 2026-10-28), so no family leaks through what a root carries.

A sibling session (`claude/pensive-darwin-33a0d8`, commit e8c7ddc) has since put the compound synthesizer
on a bus-branch physics path for IEEE 118, with each atom admitted at an effect-ratio floor on its own
(`compound_effect_floor`) and the pair's hops, footprint overlap and multiplier canonical correlation
recorded (`psse_env/providers/compound_pairs.py`).  The error-count model below builds on it.

## 2. The probe

### 2.1 Method

The probe linearizes the deployed WLS and never calls anything else.  At a recorded state, the deployed
Lagrangian solve (`tools.lagrangian_port.lagrangian_m_singlephase_details`, the solve the MATPOWER provider
runs through `_wls_json`) returns the measurement Jacobian, the residual covariance Omega with the exact
structural rows folded in, and the degrees of freedom; the whitened residual projector is
`S = R^-1/2 Omega R^-1/2` (rank dof).  The shared branch Jacobian (`tools.branch_param_jacobian`) gives
every line's R and X directions through the same projector, so the normalized multiplier of any
hypothesis (a meter: its normalized residual; a line: its R and X multipliers) and the correlation of
any two hypotheses (the normalized-multiplier covariance) are products of columns.

An atom is a deviation `d` of the measured vector from the reported model's prediction at the true
state: a gross meter error (its offset on one channel), a line R/X error (the exact change of the line's
flows and terminal injections, or its first-order direction when its size is matched), or an open line
terminal read on the bus-branch model (the line's whole contribution removed).  Its expected whitened
residual is `u = S R^-1/2 d`, its noiseless chi-square `|u|^2`, its global effect ratio
`|u|^2 / (threshold - dof)` (the effect-targeted sampler's ratio in expectation), and at a fixed state the
measurement function is additive over branches, so a combination of atoms adds their `u`.  Noise enters
through the same projector, so one noisy solve's alarms, flags and clusters are drawn exactly.  A
correction removes its atom's deviation and absorbs what its hypothesis can explain of the rest: a meter
correction projects out the meter's direction, a line correction its R/X tangent, a breaker flip removes
the deviation and absorbs nothing.  This is the stage-1 masking mechanism in closed form: correcting
atom b leaves atom a `1 - rho^2` of its energy, rho their hypotheses' correlation.

**Validation** (T10, on the 1,170 IEEE 14 compound roots without a bus split, each at its own true
state).  The chi-square of the summed atoms equals the nonlinear WLS chi-square of the root's own
noiseless vector (the true operator model at the true state plus the meter offsets, solved on the
reported model): median 1.000 to 1.010 per class, p10 to p90 within 0.98 to 1.05 on every class, so
superposition holds.  Atom by atom at the same state the agreement is the same (an atom on a two-breaker
root shows a low p10 only because its vector alone violates bus 7's zero injection when the other open
line ends there).  Against the suite's recorded solo effects, which re-solve the power flow with the atom
alone, the ratio is 1.007 [0.95, 1.10] on the single-line control, 1.008 [0.94, 1.07] on the single-breaker
control and 1.007 [0.86, 1.20] on distant pairs (median [p10, p90]), and 1.03 [0.48, 1.99] on adjacent
pairs: there the neighbour's error moves the physical state, an interaction through the power flow that
a fixed-state model leaves out.  Adjacent atoms are inseparable in the probe anyway, so the omission does
not flatter the result.  The linear model also reproduces stage 1's count of roots with a structural atom
below the alarm alone (noiseless chi-square under the threshold): 119 against the recorded 121 distant
pairs, 60 against 60 adjacent pairs, 90 against 92 meter + two R/X roots.  One limit was found and
designed around: an open line into a zero-injection bus (IEEE 14's bus 7, branch rows 7 and 14) cannot be
read at a healthy recorded state, because its flow at that bus would break the exact constraint, so the
suite's roots are read at their own true states and the drawn outages on IEEE 14 keep off those two lines
(IEEE 118 has no exact rows).

### 2.2 Data and draws

| | IEEE 14 | IEEE 118 |
| --- | --- | --- |
| reported model, meters, dof, chi-square threshold (alpha 0.01), threshold excess | `case14`, 120 used rows (2 exact), 95, 130.0, 35.0 | `case118`, 1,098, 863, 962.6, 99.6 |
| diameter in hops; bus degree max / mean | 5; 5 / 2.9 | 14; 9 / 3.0 |
| operating points (8 each) | control-meter roots of the compound suite | healthy samples of the effect-targeted corpus shards |
| meter atoms | 482 suite meter atoms (offsets, P/Q channels) | 864 meters of the `measurement` and `multi_measurement` roots |
| line R/X atoms | 1,475 suite line atoms (R and X factors) | 192 lines of the `parameter` roots |
| open-terminal atoms | 15 of the 17 lines the suite's dangling terminals use (not the two into bus 7) | every non-radial line (179), synthesized at the recorded states: IEEE 118 has no node/breaker model |

Bus splits are left out: a re-rendered layout is not a branch deviation of the reported model (stage 2
item 2 handles it in the environment).  Draw sets per system: 1,200 random pairs and 1,200 random
triples at a matched effect ratio (log-uniform over the effect-targeted band 1.5 to 20; an open terminal
keeps its natural magnitude and the others are matched to it), 600 meter + line pairs, 400 pairs at each
of the ratios 1.5, 3 and 10, 120 pairs per distance bin (0 to 6+ hops), and 300 variable-count roots at
each density of 1, 2 and 4 errors per hundred branches (truncated Poisson, 1 to 12 atoms, each atom
log-uniform in the band; open terminals in the band, and a second variant at their natural magnitude).
Forty noise draws per combination (twelve or more for the variable-count roots), eight states, seed
20261028.  Runtime: 6.5 minutes on 7 workers per configuration.

### 2.3 The cluster tool as the probe models it, and the separability criterion

Flags are the meters whose normalized residual reaches 4 (the deployed alarm level) and the lines whose
normalized multiplier reaches the line flag (3, the deployed `lambda_threshold`, or 4).  Each flag sits
on a bus set (a bus meter on its bus, a flow meter or a line on its two terminals); two flags are linked
when their bus sets are within one hop, and the clusters are the connected components.  A cluster's
measurement set is the bus meters on its buses closed by one hop and the flows of lines with both
terminals inside; its local statistic is the whitened residual energy on that set, with its own degrees
of freedom (the trace of the projector's block) and a Satterthwaite threshold at alpha 0.01 (its false
alarm on unselected one-hop sets is 1.0 to 1.1%, at most 1.45%, T1).

**Separable** (the stated criterion), on the noiseless expected evidence: (S1) every atom's own element
is flagged with the others present (its meter, or its line or one of the line's flow meters); (S2) the
atoms' own elements fall in distinct clusters and no cluster holds flags owned by two atoms (a flag
belongs to the atom whose expected evidence on it is largest); (S3) correcting the other atoms keeps at
least 90% of each atom's own chi-square energy.  The noisy reading is S1 and S2 on the flags of noisy
solves (`noisy partition recovered`).  An atom is *isolated* when it alone satisfies the three.

### 2.4 Results

**Separability of pairs and triples** (T4; first number line flag 3, second line flag 4):

| draw set | IEEE 14 separable | IEEE 14 noisy partition | IEEE 118 separable | IEEE 118 noisy partition |
| --- | --- | --- | --- | --- |
| random pairs (1,200) | 11.3%, 16.8% | 8.3%, 14.4% | 67.1%, 72.8% | 63.5%, 70.3% |
| random triples (1,200) | 0.7%, 1.2% | 0.3%, 0.8% | 27.3%, 33.4% | 21.7%, 31.1% |
| meter + line pairs (600) | 12.8%, 17.8% | 8.1%, 14.6% | 67.8%, 73.2% | 63.3%, 70.9% |
| pairs at ratio 1.5 / 3 / 10 (400 each) | 20.5 / 11.8 / 7.8% (line flag 3) | | 74.8 / 70.5 / 59.5% (line flag 3) | |

S1 (every atom visible with the others present) holds on 99.6% of IEEE 14 pairs and 100% of IEEE 118 pairs,
and S3 (no masking beyond 10%) on 89.3% and 99.2%: the binding condition is S2, footprints that touch.
Larger errors have larger footprints and separate less (75% at ratio 1.5 against 60% at ratio 10 on IEEE
118).  Linking flags only when they share a bus (link distance 0) would raise IEEE 118 pairs to 78-82%,
linking at two hops would lower them to 53-58%.  By kind, line-line pairs separate least (IEEE 118: 63%
against 74% for meter-meter; triples of three lines 15% against 35% for three meters).  The IEEE 14
compound suite's own roots at their natural magnitudes and states agree: 14% of its distant line pairs
(all at two or three hops) and at most 6% of any other class are separable (T10).

**By distance** (T5, 120 pairs per bin, line flag 3; the second separable figure is line flag 4):

| hops between the two atoms | 0 | 1 | 2 | 3 | 4 | 5 | 6+ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| IEEE 118 separable | 0% | 0% | 8%, 13% | 33%, 44% | 67%, 76% | 88%, 92% | 98%, 100% |
| IEEE 118 noisy partition recovered | 0% | 0% | 5% | 25% | 57% | 76% | 96% |
| IEEE 118 own-hypothesis correlation, p90 | 0.55 | 0.15 | 0.07 | 0.03 | 0.009 | 0.003 | 0.0004 |
| IEEE 118 cross-cluster block max, p90 | 0.45 | 0.42 | 0.47 | 0.44 | 0.22 | 0.15 | 0.04 |
| IEEE 14 separable | 0% | 0% | 22%, 34% | 48%, 76% (67 pairs) | 78%, 93% (27 pairs) | | |
| IEEE 14 noisy partition recovered | 0% | 0% | 14% | 38% | 71% | | |
| IEEE 14 own-hypothesis correlation, p90 | 0.57 | 0.15 | 0.08 | 0.01 | 0.01 | | |

The whole hypothesis set decays the same way (T3): on IEEE 118 the line-line multiplier correlation has
p99 0.99, 0.79, 0.35, 0.21, 0.08, 0.03 and 0.006 at 0 to 6+ hops (maxima 0.9995, 0.99, 0.73, 0.60, 0.58,
0.23, 0.04), meter-meter p99 0.37, 0.17, 0.05, 0.02, 0.007; IEEE 14 line-line p99 0.93, 0.52, 0.55, 0.02.

**Footprints** (T2, every element alone at a fixed ratio).  At ratio 1.5 a meter error flags 2 buses on
IEEE 14 and 3 (p90 4) on IEEE 118, a line error 3 (p90 6) and 5 (p90 8), an open terminal 3 and 5; at
ratio 20, 4, 7, 8 and 13 buses (p90 21 for a line on IEEE 118): a line error at ratio 20 covers 57% of
IEEE 14's buses (p90 65%) and 11% of IEEE 118's (p90 18%).  The footprint radius (the farthest flag from
the atom) is at most 3 hops at the p90 on IEEE 118 for ratios up to 3 and 4 hops at ratio 10 and above
(at most 2 on IEEE 14).

**Variable-count draws on IEEE 118** (T7, line flag 4; line flag 3 in brackets):

| errors per 100 branches | atoms per root p10 / median / p90 | clusters per root | roots separable | atoms isolated | peel rounds per atom (median) | line atom ranked first: global / in its cluster |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 / 2 / 4 | 1.7 | 43% (36%) | 61% (56%) | 0.50 | 53% / 74% |
| 2 | 1 / 4 / 7 | 2.3 | 23% (20%) | 37% (32%) | 0.50 | 33% / 55% |
| 4 | 4 / 8 / 11 | 2.6 | 1.3% (1.3%) | 13% (10%) | 0.56 | 17% / 33% |
| 2, open terminals at natural magnitude | 1 / 4 / 7 | 2.1 | 22% | 32% | 0.57 | 34% / 54% |

On IEEE 14 the same densities draw one error on almost every root (0.2 to 0.8 expected), so the fixed-pair
compound suite (ten errors per hundred branches) is the only multi-error IEEE 14 material and it is
mostly inseparable.  *Peel* re-partitions after each round, corrects every lone atom of the round side by
side and the strongest atom of each merged cluster: on IEEE 118 the rounds come to half the atom count
(two corrections per round), and no atom is left unflagged at the end.

**An atom's own effect** (T6).  Alone, a meter error at ratio 1.5 alarms on every draw of both systems
(the 4-sigma residual rule carries it: a meter at ratio 1.5 is a 7.2-sigma residual on IEEE 14 and a
12.2-sigma one on IEEE 118), a line error or an open terminal at exactly 1.5 on 87 to 88% of IEEE 14
draws and on all IEEE 118 draws (T2); the drawn atoms, log-uniform from 1.5 up, alarm alone on 99% and
100%.  With the others present its own element is flagged on more than 99.5% of combinations on both
systems.  After the others are fixed it still alarms on 98.5% of IEEE 14 and 99.96% of IEEE 118 pair
draws; the few silent draws (39 of 96,000 on IEEE 118 pairs) are flagged by the cluster-local chi-square
in 8% of cases.  On the IEEE 14 suite at its natural
magnitudes, without per-atom admission, 22% of the distant-pair atoms are silent after the other is fixed
and the local chi-square flags 7% of those (13% on the noiseless cluster): they are weak by
construction, and admission, not a local test, is the remedy.  Below the band the local test does add
power for spread errors: a line error at global ratio 0.6 alarms the global test on 72% of IEEE 118 draws
and its cluster's local test on 91% (27% and 37% on IEEE 14); for a meter the 4-sigma residual rule is
already the sharper test (99.99% global, 98% local at ratio 0.6 on IEEE 118).

**What the verifier's rules would keep** (T8).  For every atom, the true correction and the best wrong
line and the best wrong meter inside its cluster (the strongest flagged element no atom names) are fitted
on the noisy solve; a fit always clears its own multiplier, so, as in item 4, the target test cannot
separate them and the rules below are what does.  Atoms alone in their cluster, line flag 3:

| draws | rule | true correction kept | best wrong line kept | best wrong meter kept |
| --- | --- | --- | --- | --- |
| IEEE 118 pairs (1,614 atoms) | deployed: final when the alarm is silent, else partial on the J-basis floor | 48% | 7.6% | 0.4% |
| | final, else partial on the excess-basis floor (`chi_square_excess`) | 97% | 33% | 3.4% |
| | partial on a cluster-local floor | 100% | 55% | 40% |
| | cluster resolved: the cluster's local alarm silent, global J not risen | 99% | 18% | 0% |
| | resolved and best among the cluster's candidates (true, best wrong line, best wrong meter fitted side by side) | 97% | 2.3% | 0% |
| IEEE 118 triples (1,627) | deployed / excess / local floor / resolved / resolved and best | 24 / 85 / 100 / 99 / 96% | 1.3 / 17 / 53 / 18 / 2.6% | 0.1 / 0.7 / 39 / 0 / 0% |
| IEEE 118, 2 per 100 branches (376) | deployed / excess / local floor / resolved / resolved and best | 33 / 49 / 100 / 99 / 97.5% | 6.1 / 13 / 48 / 12 / 1.2% | 0.6 / 4.8 / 34 / 0 / 0% |
| IEEE 118, 4 per 100 branches (227) | deployed / excess / local floor / resolved / resolved and best | 14 / 22 / 100 / 99 / 96% | 1 / 3 / 47 / 14 / 2.6% | 0 / 0 / 31 / 0 / 0% |
| IEEE 14 pairs (272) | deployed / excess / local floor / resolved / resolved and best | 84 / 99.5 / 100 / 99 / 98% | 4.2 / 7.9 / 31 / 7.8 / 1.0% | 0.04 / 2.0 / 20 / 1.2 / 0% |

Three things follow.  The deployed J-basis rule loses most correct corrections of a multi-error root on
IEEE 118 (the statistic's noise baseline, 863, dwarfs one error's energy), and the excess basis that the
IEEE 118 cells use degrades as the clusters multiply (97% with two, 85% with three, 49% at two per hundred
branches: a correct fix removes 1/k of the excess).  A cluster-local floor keeps every correct correction
and keeps wrong ones on a third to a half of the clusters: it is item 4's relaxation restated, and the
probe rules it out as plainly as item 4 did.  Cluster resolution keeps 99% of correct corrections at any
cluster count and admits the best wrong meter almost never; what it admits is the best wrong line on 12
to 18% of IEEE 118 clusters, the near-collinear neighbour (line-line correlation p99 0.99 at 0 hops and
0.79 at 1 hop), which the in-cluster comparison of candidates reduces to 1.2 to 2.6% while still keeping
96 to 97.5% of correct corrections: twice the deployed rule's correct corrections at a third of its wrong
lines.  In merged clusters (two or more atoms) resolution keeps 0 to 2% of correct corrections by
construction (the other atom is still there), so merged clusters keep today's rules.  On single-error
roots the deployed rule already keeps the correct correction (98 to 100% on the one-error draws, the
solve goes clean) and nothing here changes it.

**Ranking** (T9).  A line error's line ranks first in the global list on 60% of IEEE 118 pairs, 43% of
triples and 33% of the roots at two per hundred branches, and first in its own cluster on 80%, 69% and
55% (line flag 4); within the top three, 94%, 84% and 58% globally against 99%, 95% and 81% in the cluster.
On IEEE 14 the two lists coincide (57% / 57%), since its clusters are most of the network.

**Noise** (T1).  With the line flag at 3, 58.5% of clean IEEE 118 snapshots flag something (0.83 noise
clusters per snapshot) and a noisy solve with two atoms carries 0.52 spurious clusters; at 4, 7.6% and
0.064.  On IEEE 14, 9.8% against 0.85%.  The local test fires on a noise-formed cluster 7.4% of the time at
flag 3 and 35% at flag 4 (a cluster seeded by a 4-sigma noise flag is selected on its own noise), so a
false local alarm reaches 6.2% and 2.7% of clean IEEE 118 snapshots; the deployed global alarm itself
fires on 6.65% of them (its 4-sigma residual rule over 1,098 meters; the chi-square part alone 1.15%).

**Hubs and redundancy.**  IEEE 118 pairs touching a bus of degree 6 or more separate 70% of the time
against 74 to 78% for the others (line flag 4), triples 30% against 43%: hubs cost some separability but
do not dominate it.  No measurement is critical under the full metering both systems carry (smallest
redundancy 1 - h_ii: 0.40 on IEEE 14, 0.22 on IEEE 118, five meters below 0.3).

## 3. The error-count model

- **Count.**  N ~ Poisson(delta x branches / 100), truncated to 1 <= N <= 10, delta in {1, 2} for the
  decomposition study.  At delta = 1 and 2 on IEEE 118 the median root has 2 and 4 errors and 61% and 37%
  of its atoms are isolated; at delta = 4 (median 8) only 13% are and roots are separable on 1.3% (T7),
  so delta = 4 belongs to a later stress set, not the main suite.  IEEE 14 cannot host the study (delta
  x 0.2 errors per root); it stays the regression suite with its fixed pairs.
- **Admission per atom**: each atom alone, noiseless, at a global effect ratio of at least 1.5 on the
  reported model (the sibling session's `compound_effect_floor`, e8c7ddc), plus today's physical
  admission of the root (the reported model alarms, the corrected model is clean).  With it 99.8% of
  IEEE 118 atoms still alarm after the others are fixed (T6); without it the IEEE 14 suite carries 121
  of 150 distant pairs with an atom below the alarm alone.  The ratio is global, so the same admission
  makes IEEE 118 errors locally loud (a 12.2-sigma meter at 1.5 against 7.2 sigma on IEEE 14);
  admission on the cluster's local ratio is the alternative (decision 5).
- **Magnitudes**: meters and lines log-uniform over the effect-targeted band 1.5 to 20, the meter offset
  scaled exactly and the line factors by the effect sampler's bisection (`fit_scale`); breakers keep
  their natural magnitude and are admitted inside the band.  On IEEE 118 the natural ratio of a
  non-radial line outage has quantiles 0.44 / 4.7 / 65 (10th, 50th, 90th): 30% fall below 1.5 and 29%
  above 20, and admitting the large ones at their natural size costs isolation (32% against 37% at
  delta = 2, T7) through larger footprints (a line at ratio 20 flags 13 buses, p90 21).
- **Kinds**: meter, line R/X and breaker status per atom, uniform; on IEEE 118 a breaker atom is a line
  status error on the bus-branch model until a node/breaker model exists (decision 3).  The deployed
  tools already correct it there: without a node/breaker binding, `get_topology_context` screens branch
  status flips on the lines its multipliers rank and `correct_topology` flips a branch row.  Without the
  node/breaker channel the clean-flip screen never runs, so on IEEE 118 the default teacher orders exactly
  as `baseline_compare`.  Distinct elements only; no placement rule.
- **Interaction recorded, not forced**: in the audit block, never in the policy view, every root records
  per atom its solo effect and kind, per pair the hops, the own-hypothesis canonical correlation, the
  footprint overlap and the cross-cluster block maximum, and per root the noiseless partition at the
  cluster tool's thresholds, each atom's masking keep and the peel rounds; `compound_pairs.py` already
  computes the pair fields.  Analyses stratify by these fields after the fact, so the suite stays a
  random draw.
- **Uniform metadata**: parameter scans on every root and the node/breaker channel wherever it is
  modelled, as the 2026-10-28 decision requires.

## 4. The cluster tool

A context tool, `get_cluster_context`, read-only, computed from the solve the state already has.

- **Inputs from one solve**: normalized residuals, normalized branch multipliers (R and X), the residual
  covariance (its diagonal for the local degrees of freedom, its blocks for the local threshold) and the
  bus graph; the deployed solve returns all of them (`r`, `lambdaN`, `residual_covariance`, `ea`).
- **Flags**: meters at a normalized residual of 4 (the deployed alarm level); lines at a normalized
  multiplier of 4 for clustering, not the deployed 3 (on IEEE 118, 3 flags 58.5% of clean snapshots and
  leaves 0.52 spurious clusters beside two real errors; 4 flags 7.6% and leaves 0.064), while the
  parameter context keeps 3 for its own list.
- **Hop closure and components**: two flags are linked when their bus sets are within one hop; clusters
  are the connected components; a cluster's region is its flagged buses closed by one hop.  Link 1 is
  the conservative choice (link 0 separates 78 to 82% of IEEE 118 pairs instead of 67 to 73%, but two
  clusters one hop apart still have a cross-cluster block of 0.42 at the p90).
- **Per-cluster local statistic**: the whitened residual energy over the region's measurements (bus
  meters inside, flows of lines with both ends inside), its degrees of freedom (median 119 for IEEE 118
  pair clusters, 80 for IEEE 14, which is most of IEEE 14's 95), a Satterthwaite threshold at alpha 0.01
  (nominal false alarm 1.0 to 1.1%), the local alarm and the local effect ratio; an atom's own energy
  sits 99.5% inside its cluster's set (median, IEEE 118 pairs).  The region is frozen when the cluster
  forms and reused for every later local statistic of that cluster, so a correction cannot redraw its
  own test.
- **Dominance tags and ranked candidates per cluster**: the deployed WLS dominance (residual-dominant,
  branch-dominant, none) and the ranking dominance ratio (top over runner-up, 1.2 as deployed) computed
  on the cluster's own flags; the cluster's top five lines, top five meters and, with the node/breaker
  channel, the topology screen's flips restricted to the cluster's yards (which also bounds the
  clean-flip screen's 9 solves per anomalous state to the yards a cluster touches); without the channel,
  the bus-branch status flips of the cluster's lines.  The cluster's own list puts
  the true line first on 80% of IEEE 118 pairs, 69% of triples and 74% and 55% of the roots at one and two
  errors per hundred branches, where the global list does on 60%, 43%, 53% and 33% (T7, T9).
- **Noise clusters**: a cluster whose local alarm is silent is listed as such and offers no target; at
  line flag 4 a false local alarm reaches 2.7% of clean IEEE 118 snapshots, below the global alarm's own
  6.65%.
- **What the policy reads**: the WLS ledger (`fresh_context_evidence["wls"]`) gains a `clusters` list
  (id, buses, region size, local dof, statistic, threshold, alarm, ratio, dominance tags, ranked
  candidates) on states with two or more alarmed clusters; the existing contexts accept an optional
  `cluster_id` that scopes their ranking to the cluster; a correction may name its cluster.  A state
  with one cluster shows exactly today's evidence, so every conventional single-error root keeps its
  observation and its trajectory (to be checked on the 192-root suite, stage S3).

## 5. Verification

- **Cluster resolution** (new disposition, behind a verifier option, off by default): a correction on a
  multi-cluster state is kept for its cluster when the verification solve shows that cluster's local
  alarm silent on its frozen region, the global chi-square not above the parent's, no other cluster's
  local statistic above its parent value by more than two of its standard deviations, and today's
  target test passed.  The cluster closes; the episode continues on the others.  This keeps 99% of the
  correct corrections of an isolated atom at any cluster count, where the deployed rule keeps 48% (pairs)
  to 24% (triples) on IEEE 118 (T8).
- **With the in-cluster comparison**: the cluster's ranked candidates (in the probe the true correction
  against the strongest wrong line and wrong meter; in the tool its top three) are fitted on cloned
  states, as `compare_parameter_candidates` already does per line, and only the one leaving the lowest
  local statistic may resolve the cluster: 96 to 97.5% of correct corrections kept on IEEE 118, the best
  wrong line 1.2 to 2.6% against 12 to 18% without the comparison, the best wrong meter never (T8).  It
  needs the true element among the compared candidates: a line error's line is in its cluster's top three
  on 99% of IEEE 118 pairs, 95% of triples and 81% of the roots at two per hundred branches (T9), so the
  comparison costs about three extra solves per cluster and is the step that matters most.
- **The global floors stay as they are**: partial progress on a correction that does not resolve its
  cluster is judged exactly as today (J basis by default, the excess basis where a cell already sets it).
  No cluster-local floor: it would keep the best wrong line on 55% and the best wrong meter on 40% of
  isolated IEEE 118 clusters, against 7.6% and 0.4% under the deployed rule (T8).  This is why stage-2
  item 4's result rules out relaxing the global floor: its target-first rule waived the floor because
  the target cleared, and a wrong neighbouring refit clears its own target too (366 to 641 wrong partial
  acceptances); dividing the floor's denominator by the cluster's share is the same relaxation.
  Resolution asks for something a wrong correction rarely achieves, the cluster's whole evidence gone.
- **Merged clusters** (two or more atoms in one cluster: 27 to 33% of IEEE 118 pairs, two thirds of
  triples) are today's compound problem in a smaller region.  Resolution of the cluster requires all of
  its atoms; in between, the existing rules and stage-2 item 3's declared multi-row candidate apply, with
  the declaration generalized from "the partner line of the last committed correction" to "rows inside
  this cluster's region", and the collateral check unchanged outside it.
- **The post-commit confirmation gate becomes per cluster**: after a resolution, that cluster accepts no
  further correction; another cluster whose local alarm stands keeps the episode open.  Stage 1's refused
  second corrections (57 distant, 19 adjacent pairs) were all on roots without per-atom admission.
- **Termination** stays global: the episode resolves when the global alarm is silent and every cluster's
  local alarm is silent; a handoff with alarmed clusters left lists them.

## 6. Budget and truth audit

- **Budget**: 40 actions today.  Per alarmed cluster the stage-1 rollouts spent 7.5 actions on a single
  line, 14.8 on a single breaker (7.0 under the topology-first orders, the clean flip among them), 21.3 on
  one to three meters, 10.5 to 21.0 on a pair
  and 23.0 on the three-atom class; the proposal is 40 + 20 per additional alarmed cluster at the first
  solve, capped at 100 (about 66 for the 2.3 clusters of a delta = 2 root).  The cluster count is
  observable, so the budget leaks nothing.
- **Truth audit**: per cluster, success when every atom whose own element falls in the cluster is
  corrected within the release tolerances and no healthy element of the cluster's region changed;
  reported beside the root verdict, which stays all-or-nothing (every atom corrected, nothing healthy
  touched anywhere), as today.  The isolated-line rule of decision (b) carries over unchanged.

## 7. Concurrency, in two steps

1. **Candidates per cluster on cloned states, merged in sequence.**  For every alarmed cluster with a
   supported top candidate, the candidate is executed and verified on a clone (the cloning
   `compare_parameter_candidates` uses, one clone per candidate, discarded after the verdict); the
   resolved candidates are then applied to the real state one at a time, each re-verified after the
   merge (one solve per merge), and a merge that breaks another cluster's resolution is rolled back and
   retried alone.  Two clusters are serialized instead when their regions share a bus or their
   cross-cluster block reaches 0.3 (IEEE 118 p90 0.22 from 4 hops, 0.44 at 3 hops, T5).  The peel rounds
   say what this can save: half as many rounds as errors on IEEE 118 at every density (T7), so about two
   corrections per solve round.
2. **Partitioned sub-networks** (later): a cluster's region estimated alone with its boundary
   injections fixed from the global estimate, which turns one 1,098-meter solve into several solves of
   about a hundred measurements (the clusters' local sets: median 119, p90 235 on IEEE 118 pairs) and holds
   99.5% of the atom's energy (median).  Worth it only on systems well beyond IEEE 118, and only after
   step 1 works.

## 8. What the expert does per cluster

The teacher's rules unchanged inside a cluster: the cluster with the largest local effect ratio first, its
family from the cluster's dominance tags and ranked candidates (meter, line, breaker, with the
topology-first variants' order applied per cluster), its usual correct-verify-commit or rollback
sequence, the in-cluster comparison of its top candidates before a resolution; on resolution the next cluster;
on a merged cluster the strongest atom first, then a re-clustering solve.  Every run names its teacher:
`baseline_compare` explicitly (the default for new work is now `baseline_compare_topology_clean_flip`,
so an unnamed run would silently change the reference arm), the clustered variant built on it
(`baseline_compare_clustered`), and on IEEE 14 the clean-flip teacher with its clustered counterpart as a
second arm.  With one cluster each clustered variant is its teacher exactly, which the conventional
192-root suite must confirm (identical tool sequences), as every stage-2 item did.

## 9. Training implications

- **Observation**: cluster-centric on multi-cluster states (the `clusters` ledger and the cluster-scoped
  contexts), the present observation otherwise.
- **Labels**: per cluster; the teacher's action on a multi-cluster state names the cluster it works on,
  and a DAgger relabel is the teacher's action for the cluster the student chose, so the student is not
  penalized for a different but legal cluster order.
- **Episode length**: roughly 10 to 15 actions per atom (stage 1), so about 20 to 30 at delta = 1 (median
  two errors) and 40 to 60 at delta = 2 (median four) on IEEE 118, against today's 40-action ceiling and
  the longer prompts that come with it; the step-1 concurrency saves solves and rounds, not policy
  actions, unless one action can carry the candidates of several clusters (decision 9).  No DAgger round
  before the teacher's ceiling on the variable-count suite is measured (stage S3).

## 10. Risks, with the probe numbers beside each

| risk | what the probe shows | what the design does |
| --- | --- | --- |
| overlapping clusters | inseparable at 0 to 1 hops on both systems, 8 to 33% separable at 2 to 3 hops (IEEE 118); 27 to 33% of random IEEE 118 pairs merged; 13% of atoms isolated at 4 per 100 branches | merged clusters keep today's rules and item 3's declared multi-row candidates; densities above 2 are a stress set |
| dilution | detection: none with per-atom admission (99.8% alarm after the others are fixed); progress: the deployed rule keeps 48% / 24% of correct corrections in IEEE 118 pairs / triples, the excess basis 97% / 85% / 49% at 2 / 3 clusters / delta 2 | cluster resolution, global floors untouched |
| wrong corrections | a cluster-local floor keeps the best wrong line on 55% and the best wrong meter on 40% of isolated IEEE 118 clusters; resolution 18% and 0%; with the comparison 2.3% and 0% (pairs) | resolution plus the in-cluster comparison; no local floor |
| hub injections | pairs touching a degree-6+ bus 70% separable against 74 to 78% (triples 30% against 43%) | none beyond the link rule; watch the per-degree split in S2 |
| critical measurements | none under full metering (smallest redundancy 0.22 on IEEE 118, 0.40 on IEEE 14) | untested under reduced metering, the main gap: S2 must add a reduced-metering draw before any claim on real metering |
| noise clusters | line flag 3: 58.5% of clean IEEE 118 snapshots flagged, 0.52 spurious clusters beside two errors; flag 4: 7.6% and 0.064; a false local alarm on 2.7% of clean snapshots | line flag 4 for clustering, silent clusters offer no target |
| locally loud admission | a meter at global ratio 1.5 is 12.2 sigma on IEEE 118 | decision 5 (admission on a local ratio) |
| linear model | superposition at the true state within 0.98 to 1.05 of the nonlinear WLS on every IEEE 14 class; adjacent pairs' physical interaction (0.48 to 1.99 of the recorded solo effect) and bus splits are not modelled; drawn outages are read at a healthy recorded state, so none ends at a zero-injection bus | S1 and S2 measure on the nonlinear solves of the real suite |

## 11. The staged plan, with costs

| stage | what | cost | depends on |
| --- | --- | --- | --- |
| S0 | this probe and note | done (CPU, about 6 minutes per configuration on 7 workers) | |
| S1 | variable-count generator on IEEE 118: the bus-branch synthesizer of e8c7ddc with N ~ Poisson(delta x branches / 100), delta in {1, 2}, per-atom admission at 1.5, the band, meter / line / line-status atoms, the recorded interaction block; a 600-root suite (300 per density) plus a 120-root reduced-metering draw | about a day of code, a few hours of CPU build | e8c7ddc merged or rebased |
| S2 | `get_cluster_context` read-only and its ledger block, no gate or verifier change; an offline partition audit on the S1 suite's noisy nonlinear solves against the recorded structure (expected from the probe: about 70% of pairs, 30% of triples recovered) | about a day of code, an hour of CPU | S1 |
| S3 | cluster resolution (verifier option), per-cluster confirmation gate and per-cluster audit, expert variant `baseline_compare_clustered`; runs, every teacher named explicitly: the S1 suite under `baseline_compare` and `baseline_compare_clustered` (on IEEE 118 the clean-flip default orders as the baseline), the IEEE 14 compound suite under `baseline_compare`, `baseline_compare_topology_clean_flip` and their clustered variants (regression; little change expected, its clusters merge), the conventional 192-root suite (identical tool sequences required) | two days of code, about an hour of CPU per run set | S2, the user's decisions 6 to 8 |
| S4 | concurrency step 1 (per-cluster candidates on clones, sequential merge), behind a flag; actions, solves and success against S3 | a day | S3 |
| S5 | a DAgger round on the cluster-centric observation | GPU cell | S3's teacher ceiling worth teaching |
| S6 | partitioned sub-network estimation | research | S4, a larger system |

## 12. Decisions for the user

1. **Build the cluster tool?**  Recommended: yes, on IEEE 118, read-only first (S2); not on IEEE 14,
   where 11% of pairs separate.
2. **Where the variable-count suite lives**: IEEE 118 on the bus-branch path (recommended), IEEE 14 kept
   as the regression suite.
3. **Breakers on IEEE 118**: line-status atoms on the bus-branch model now, corrected through the
   existing bus-branch topology route (recommended), or a node/breaker model for IEEE 118 first (several
   days, the stage-2 binding repairs to port, and only then does the clean-flip default differ from
   `baseline_compare` there).
4. **The line flag for clustering**: 4 (recommended; 7.6% of clean IEEE 118 snapshots flagged against
   58.5% at 3), with the contexts' own lists at 3.
5. **Admission**: per atom at a global effect ratio of 1.5 as asked, or on the cluster's local ratio,
   which admits smaller errors on large systems.
6. **Verification**: cluster resolution with the in-cluster comparison, no cluster-local floor, global
   floors unchanged (recommended); or a stage without any verification change, the tool used for
   ordering only.
7. **Audit**: root verdict all-or-nothing as today with per-cluster success reported (recommended), or
   per-cluster scoring.
8. **Budget**: 40 + 20 per additional alarmed cluster at the first solve, capped at 100, or another rule.
9. **Batched corrections**: whether one policy action may carry several clusters' candidates (only then
   does concurrency shorten episodes for the student), or one cluster per action.
10. **Density**: delta in {1, 2} for the main suite with delta = 4 as a stress set (recommended), or
    another range.
