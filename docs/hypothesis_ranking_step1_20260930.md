# Hypothesis ranking, step 1: what the balanced evidence can separate (2026-09-30)

Results of step 1 of `docs/hypothesis_ranking_plan_20260930.md`: an offline
dataset of balanced WLS ledgers and balanced screen reports over every
IEEE-14 scenario family, labelled with offline truth and split by physical
parent, plus the same screen on the IEEE 57 study roots of 2026-09-11.
Code: `research/hypothesis_ranking/` (builder, report, IEEE 57 check) and the
offline-only fields added to `psse_env/providers/hif_screen.py`. Data and
reports: `output/hypothesis_ranking_20260930/{ieee14,ieee57}/` (gitignored;
`report.md`, `metrics.json`, `manifest.json`, JSONL rows). Research only.

> Step 2 (same day) changed the screen in response to these findings and
> re-ran this dataset through it; the `unexplained` rates in section 2.2
> describe the step 1 screen. The superseding numbers and the rule that
> shipped are in `docs/hypothesis_ranking_step2_20260930.md`.

## 1. What was built

The round-0 generator's own family builders were called row by row under the
suspicion-gated profile with the DAgger suites' admission (WLS alarm at margin
1.25, parameter ranking at the development threshold 1.0 with rank allowance
2, meter errors lifted to ten sigma, dangling-terminal and bus-split breaker
errors), so every root records the physical parent it came from: the tabular
corpus row, the HIF or unbalance window (shared by a window's pure root and its
measurement overlay), or its own synthesized operating point. Splits are by
parent, 60/20/20 train, calibration, test (recorded, not yet used).

| Set | Count | Content |
| --- | --- | --- |
| Roots | 2,398 | 250 each of no_error, measurement, parameter, measurement+parameter, topology, measurement+topology, harmonic; 224 multi_measurement; all 131 HIF windows twice (hif, measurement+hif); all 162 unbalance windows |
| Children | 1,769 | truth-corrected states of the mixed, multi-meter, measurement and parameter roots (overlay meter restored, largest meter restored, all but the smallest meter restored, branch parameter restored) |
| Healthy alarms | 69 of 3,500 | the tabular corpus's clean windows whose noise alone alarms the WLS (2.0%) |
| Mimic | 200 | two flow meters of one candidate line biased 10 to 15 sigma, same sign and opposite sign |

Every admitted fault root alarms (admission requires it); no analysis failed.
A screen round costs about 1 s on one core for IEEE 14 and 54 s for IEEE 57.

The screen's full report gained offline fields the policy never sees: per class
the best refit's `J`, the alarm rule it would have to clear (chi-square at the
class's remaining degrees of freedom, alpha 0.01, and max normalized residual
below 4), ranked candidates; two extra hypotheses the production comparison
does not run (a joint R and X refit on the top three parameter branches; the
winner's model with one more meter set aside); and `unexplained`, true when the
round's base solve alarms and no single balanced cause clears it. The six
existing screen tests and four new ones pass; the compact policy-visible report
is unchanged.

## 2. Results on IEEE 14

### 2.1 The production winner is right on single-cause roots and never HIF on a non-HIF root

Winner of the penalized comparison, alarm test ignored (what the shipped screen decides):

| family | roots | meter | parameter | topology | hif |
| --- | --- | --- | --- | --- | --- |
| measurement | 250 | 247 | 2 | 1 | 0 |
| multi_measurement | 224 | 223 | 1 | 0 | 0 |
| parameter | 250 | 1 | 247 | 2 | 0 |
| measurement+parameter | 250 | 1 | 248 | 1 | 0 |
| topology | 250 | 0 | 45 | 205 | 0 |
| measurement+topology | 250 | 0 | 3 | 247 | 0 |
| harmonic | 250 | 227 | 22 | 1 | 0 |
| hif | 131 | 1 | 0 | 0 | 130 |
| measurement+hif | 131 | 2 | 1 | 0 | 128 |
| three_phase_unbalance | 162 | 129 | 33 | 0 | 0 |

The HIF class wins on 130 of 131 HIF roots and 128 of 131 mixed HIF roots,
always on the true line, and on none of the 1,474 alarmed non-HIF roots; it
wins on 1 of the 69 alarmed healthy windows. This reproduces the 2026-09-26
feasibility result on a fresh draw. Target ranking inside the winning class:
the true meter is first on 250 of 250 single-meter roots; the true branch is
first on 194 of 194 dominant and 45 of 48 ambiguous parameter roots and on 3
of the 8 misranked roots where the multiplier ranking itself puts a neighbour
first (top-2 on all 250); the isolated line is first on 170 of 170
dangling-terminal roots. Unbalance and harmonic roots are explained as a bad
meter (129 and 227) or a branch parameter (33 and 22): on balanced SCADA they
have no class of their own, as the feasibility study said.

### 2.2 `unexplained` separates the no-route families only partly, and fires on roots the screen simply stopped on

Rate of `unexplained` among alarmed roots, under the single-cause definition
and with the two offline extras added:

| family | alarmed | single cause | + joint R/X | + one more meter | + both |
| --- | --- | --- | --- | --- | --- |
| measurement | 250 | 0.0% | 0.0% | 0.0% | 0.0% |
| multi_measurement | 224 | 73.2% | 73.2% | 50.4% | 50.4% |
| parameter | 250 | 18.8% | 0.0% | 13.6% | 0.0% |
| measurement+parameter | 250 | 66.4% | 61.2% | 18.0% | 15.6% |
| topology | 250 | 28.4% | 27.6% | 28.0% | 27.2% |
| measurement+topology | 250 | 91.2% | 91.2% | 0.8% | 0.8% |
| harmonic | 250 | 89.6% | 89.2% | 84.4% | 84.4% |
| hif | 131 | 35.1% | 35.1% | 6.9% | 6.9% |
| measurement+hif | 131 | 41.2% | 41.2% | 9.2% | 9.2% |
| three_phase_unbalance | 162 | 66.7% | 66.7% | 48.1% | 48.1% |
| healthy windows (alarmed) | 69 | 11.6% | 10.1% | 5.8% | 5.8% |

Against the plan's decision rule (capture at least 80% on unbalance plus
harmonic, false rate under 5% on single-cause roots):

| definition | no-route capture | single-cause false rate | healthy-alarm false rate |
| --- | --- | --- | --- |
| single cause (option A as written) | 80.6% | 18.6% | 11.6% |
| + joint R/X | 80.3% | 13.1% | 10.1% |
| + one more meter | 70.1% | 12.8% | 5.8% |
| + both | 70.1% | 8.7% | 5.8% |

Option A as written fails the rule on the false-rate side. Where the false
rate comes from is the useful part:

- **Parameter roots (47 of 250).** Every one has both R and X wrong (the
  corpus draws the factors independently); the screen refits one of them. The
  joint R/X refit clears all 47. This is a screen defect, cheap to fix (20 more
  fits per round).
- **Topology roots (71 of 250).** 68 of the 80 bus-split roots: the screen has
  an outage class but no bus-split class. The remaining 2 of 170
  dangling-terminal roots clear with one more meter. Bus splits are the
  substation route's business (`get_topology_context` is admitted without any
  suspicion), so the ledger must run that route before an alarm can count as
  unexplained.
- **HIF roots (46 of 131).** The split-line shunt wins but does not clear the
  alarm; setting one more channel aside does, and that channel is a phase-A
  voltage reading in 45 of the 46 cases (36 clear, 9 do not). A single-phase
  fault shifts the phase-A magnitude at nearby buses by its negative- and
  zero-sequence effect, which the balanced shunt cannot carry; the
  phasor-conditioned prediction in `suspicion_gated.py` already models exactly
  this offset once phasors exist.
- **Mixed roots (m+parameter 166, m+topology 228 of 250).** The two-round rule
  only continues after a meter win, so a physical win leaves the overlay meter
  in place. One more meter set aside clears 85% of m+parameter and 99% of
  m+topology roots. The truth-corrected children confirm the sequential
  picture: after the branch parameter is fixed the remaining meter is named on
  250 of 250; after the overlay meter is restored the remaining fault is named
  on 192 of 250 (parameter, 52 of the rest are the R/X defect), 250 of 250
  (topology) and 130 of 131 (HIF line).
- **Multi-meter roots (164 of 224).** Two rounds cannot set aside three or more
  meters; after the largest meter is restored, 111 are explained and 113 still
  need more rounds; with all but the smallest restored, 161 of 164 are
  explained as that meter. The real controller commits one meter per
  transaction and re-screens, so this is a budget question, not a separation
  failure.

With those three screen changes (joint R/X, continue after any win, bus
splits routed first) the single-cause false rate would be about 1% (9 of 801
non-bus-split single-cause roots), and acquisitions on roots whose truth needs
no phasors fall from 676 of 1,474 (45.9%, option A as written) to about 227 of
1,543 (14.7%, dominated by multi-meter roots that need more rounds).

### 2.3 The half of unbalance roots that look like a bad meter look like a bad *voltage* meter

Unbalance roots explained by the single-cause screen: 48 are explained as a
meter, and in all 48 the meter the screen sets aside is a phase-A voltage
channel (73 picks, every one `Vm`). Harmonic roots explained as a meter: 24 of
25 involve a `Vm` pick. Among the 46 unexplained HIF roots, the one extra
channel that clears the alarm is `Vm` in 45. The pattern is physical: the
phase-A voltage magnitude is the one SCADA channel that a single-phase or
distorted phenomenon moves while the three-phase power totals stay consistent,
so the balanced model's cheapest explanation is a bad voltage meter.

That gives a third, label-free acquisition trigger: **a leading voltage-meter
hypothesis**. Measured on this dataset, "HIF wins, or the balanced ladder ends
unexplained, or the leading hypothesis edits a `Vm` channel" would open phasors
on 161 of 162 unbalance roots and 249 of 250 harmonic roots. Its cost is the
roots whose truth is a bad voltage meter or whose noise lands on one:

| family | roots with a `Vm` meter pick |
| --- | --- |
| measurement (single meter; 35 of 250 have a true `Vm` error) | 33 of 250 |
| multi_measurement | 27 of 224 |
| healthy windows (alarmed) | 10 of 69 |
| parameter, measurement+parameter | 1, 1 of 250 |
| topology, measurement+topology, hif, measurement+hif | 0 |

So about 13% of single-meter roots and 14% of alarmed healthy windows would
acquire phasors before a voltage-meter edit, and the phasors then decide
(balanced three-phase: proceed with the meter edit; unbalance: the NLM route;
HIF-like: the HIF route). This is not in the approved rule (C1 = option A
covers `unexplained` only) and is recorded here as decision **D3** for step 2.
It also removes a false-commit risk the plan flagged: under an observable-only
verifier a `Vm` edit on an unbalance root would be accepted.

### 2.4 The known HIF mimic

Two flow meters at the two ends of one candidate line, both biased the same
way by 10 to 15 sigma, are flagged as an HIF on that line in 37 of 100 roots
(the feasibility study's synthetic test gave 16 to 26%); biased in opposite
directions, 0 of 100. Under the gated contract this costs a phasor
acquisition, which the zero-sequence screen then refutes; it is the main
reason the HIF suspicion must stay "a reason to request phasors, not a
diagnosis", and a natural target for the learned ranker (step 4).

## 3. IEEE 57 transfer check

The 2026-09-11 IEEE 57 study roots (four OpenDSS models: diagonal and coupled
at 0.8 and 1.0 load; 756 resistive HIF, 336 unbalance, 400 healthy noise
realizations) were solved with the repository WLS on the canonical case57 and
screened when the alarm rule fired. The WLS layer reproduces the study: all
1,492 saved chi-square statistics within 1e-5 relative, and at the study's
alpha of 0.05 the alarm decision agrees on all 1,492 roots (267 alarms); the
repository rule (alpha 0.01 or a normalized residual at or above 4) alarms 184.

case57 carries no base kV, so `default_hif_lines` gives the screen no candidate
line there; the check passed every in-service untapped branch (65 lines)
explicitly. A transfer would have to supply candidate lines some other way.

| family | n | alarmed | screen final class (alarmed) | HIF suspected | HIF on true line |
| --- | --- | --- | --- | --- | --- |
| healthy | 400 | 26 | meter 20, parameter 1, topology 3, unexplained 2 | 0 | |
| hif | 756 | 112 | hif 14, meter 41, parameter 9, topology 4, unexplained 44 | 16 | 16 |
| unbalance | 336 | 46 | meter 20, parameter 2, topology 4, unexplained 20 | 0 | |

| HIF R (pu) | n | alarmed | HIF suspected | true line first in the HIF class |
| --- | --- | --- | --- | --- |
| 10 | 252 | 93 | 16 | 52 |
| 100 | 252 | 10 | 0 | 0 |
| 1000 | 252 | 9 | 0 | 0 |

Three things differ from IEEE 14. First, the HIF class rarely wins even when
it ranks the true line first (52 of 93 at R = 10 pu): the competing meter
explanation wins, and 43 of the 69 meter wins on HIF roots set aside a `Vm`
channel. The phase-A offset of section 2.3 dominates on this network, and the
faults here cycle through all three phases (phase 1: 10 of 48 flagged, phase
3: 4 of 48, phase 2: 2 of 16). Second, the R at or above 100 pu faults barely
alarm (19 of 504), the balanced-screen bound of 2026-09-10. Third, 26 of the
400 healthy windows alarm at alpha 0.01 (6.5%): the OpenDSS 57-bus model and
the canonical case57 disagree at the level of the detector, as the IEEE 14
corpora did before their shunt-convention and dispatch fixes. The screen
explains 20 of those 26 as a meter and calls 2 unexplained. Unbalance at delta
0.2: 19 of 35 alarmed roots unexplained; at delta 0.05: 1 of 11.

The transfer conclusion is narrow: the balanced HIF hypothesis needs the
phase-A voltage offset in its model (or the voltage-meter trigger of 2.3)
before it is worth evaluating on IEEE 57, and the 57-bus healthy baseline
needs the same physics audit the IEEE 14 corpora received.

## 4. What this means for step 2

1. **`unexplained` is the ledger's terminal condition, not a first-alarm
   trigger.** Evaluated at the first alarm it fires on 46% of the roots whose
   truth needs no phasors; evaluated after the balanced ladder (meter rounds up
   to the budget, continue after a physical win, substation route for bus
   splits) it would fire on about 15%, mostly multi-meter roots that need more
   transactions. Option A should be implemented as "the balanced ledger ended
   with the alarm unexplained", which is also where the current code hands off
   with `recovery_options_exhausted`.
2. **Three screen changes before any rule is wired:** a joint R/X parameter
   hypothesis (fixes 47 of 47 parameter misses), rounds that continue after
   any class wins (fixes 85 to 99% of mixed roots), and the HIF hypothesis with
   the phase-A voltage offset at the buses of the split line (fixes 36 of 46
   HIF misses; also the IEEE 57 problem). Each is a production change to the
   screen's decision and must be re-verified on the 2026-09-26 feasibility set
   and on this dataset's test parents before it ships.
3. **Decision D3 (new, for the user):** should a leading voltage-meter
   hypothesis open phasors before the edit is committed? It lifts unbalance
   capture from about 48% to 99% and harmonic to 99.6%, at the cost of a
   phasor acquisition on about 13% of single-meter roots and 14% of alarmed
   healthy windows, and it closes the observable-verifier false-commit path on
   unbalance roots. Without it, half of the unbalance roots remain
   indistinguishable from a bad voltage meter on balanced evidence and end in
   either a wrong meter edit (deployment verifier) or a handoff.
4. **Spectra (C2).** Harmonic roots are `unexplained` in 84 to 90% of cases
   and involve a `Vm` pick in most of the rest, so under option A plus D3
   nearly every harmonic root reaches the phasor acquisition; the phasors come
   back balanced (the harmonic corpus is a fundamental-frequency synthesis),
   which is the second-tier condition for spectra the user approved.
5. **For the learned ranker (step 4)** the dataset carries, per state, the
   WLS ledger (signed top residuals with channel types, per-branch multiplier
   ranking, dominance tags), the per-class refit scores, margins and `J` after
   the refit, the extras, the ranked candidates, and the parent split. The two
   discriminations where rules are weakest and a ranker could add value are the
   same-sign flow-meter pair against a real HIF (37% mimic rate) and a
   voltage-meter error against an unbalance (identical on balanced evidence
   unless the pattern across neighbouring buses carries information).

## 5. Reproduce

```bash
python -m research.hypothesis_ranking.build_dataset --output-dir output/hypothesis_ranking_20260930/ieee14 --workers 16
python -m research.hypothesis_ranking.report --dataset-dir output/hypothesis_ranking_20260930/ieee14
python -m research.hypothesis_ranking.ieee57_screen --output-dir output/hypothesis_ranking_20260930/ieee57 --workers 12
python -m pytest tests/test_hif_screen.py tests/test_hif_screen_offline_fields.py -q
```

Seed 20260930, commit 266e231 plus the uncommitted step-1 changes; the IEEE 14
build took about an hour (15 minutes of it the parameter admission), the
IEEE 57 screen about 25 minutes on 12 workers.
