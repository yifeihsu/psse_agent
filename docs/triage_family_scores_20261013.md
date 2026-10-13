# Triage family scores and the spectra-first route (2026-10-13)

Branch `codex/classifier-triage-20261004`. Follows
`docs/effect_targeted_magnitude_20261012.md` and
`docs/classifier_triage_ieee118_transfer_20261007.md`.

## Why

The triage GNN has always trained three heads: the request flag (`needs_aux`),
a six-family multi-label head and a first-family head for ordering the
balanced routes. The runtime report (`research/classifier_triage/runtime.py`)
exposed only the request decision and the first balanced family; the
hif/unbalance/harmonic probabilities were computed and dropped (2026-10-07,
over the IEEE 14 simulator-artifact concern). The IEEE 118 agentic cell lost
all 16 harmonic roots because the frozen IEEE 14 student never took the second
step from balanced phasors to spectra, while the gate itself opened on every
one of them. The question was whether the heads the model already has carry
the routing information the agent is missing.

## What the other heads know (effect-run models, view `residual_v3`, 5 seeds)

Scored from the saved checkpoints of the effect-targeted transfer run
(`output/classifier_triage_20261012/transfer_v3`) on both test splits. "Gate"
is the dedicated request flag at its calibrated threshold; "waveform heads as
gate" is the largest of the three waveform heads thresholded at the same
unneeded rate; "routing" is the largest waveform head against the true
waveform family among the roots the gate flags.

| trained → scored | gate recall | waveform heads as gate | routing hif / unbalance / harmonic |
| --- | --- | --- | --- |
| 14 → 14 | 99.2% | 96.5% | 139/141, 56/57, 55/55 |
| 14 → 118 | 66.6% | 52.1% | 227/227, 14/14, 1/12 |
| 118 → 14 | 77.6% | 80.0% | 131/131, 2/23, 44/44 |
| 118 → 118 | 98.7% | 95.8% | 240/242, 81/82, 51/51 |
| joint → 14 | 99.6% | 99.6% | 142/142, 56/57, 55/55 |
| joint → 118 | 98.7% | 100% | 241/242, 82/82, 51/51 |

Three readings.

* More heads do not lift the gate: the multi-task regularization is already
  in the training, and the waveform heads used as a gate are never better
  than the dedicated flag except for a point in the reverse direction.
* In-domain and under the joint model the family head routes the stream
  almost perfectly: among flagged roots it names the harmonic ones 55/55 on
  IEEE 14 and 51/51 on IEEE 118.
* Zero-shot the routing fails the same way the gate does, and in a telling
  way: the IEEE 14 model calls 53 of 55 IEEE 118 harmonics "unbalance" and
  the IEEE 118 model calls 53 of 57 IEEE 14 unbalances "harmonic". A small
  network's harmonic footprint has the spread of a large network's unbalance.
* Misses are not uncertainty: in-domain the one to five missed roots score
  0.3 to 0.84 against thresholds of 0.71 to 0.86, but zero-shot the IEEE 118
  model scores its missed IEEE 14 unbalances at a median of 0.003 (28 of 34
  below half the threshold). A confidence or abstain output would not flag
  what the model is sure about.

## What the tracked exports would do

The runtime rule below was checked on the two exports the cells use, on
their own test splits (`research/classifier_triage/data.split_rows`, excess
energy admission). "Spectra first" counts flagged roots whose largest
waveform head is harmonic.

| export | scored on | harmonic flagged → spectra first | other families sent to spectra first | harmonic head median on flagged harmonics |
| --- | --- | --- | --- | --- |
| `triage_gnn_ieee14_20261004` (residual) | IEEE 14 | 50/50 → 50 | none | 0.992 |
| `triage_gnn_joint_20261011` (residual_v2) | IEEE 14 | 50/50 → 50 | 6 of 29 unbalance | 0.962 |
| `triage_gnn_joint_20261011` (residual_v2) | IEEE 118 | 87/88 → 86 | 3 of 51 unbalance, 1 meter, 1 multi-meter | 0.936 |

An absolute threshold (harmonic ≥ 0.5) routes 84/87 on IEEE 118 and sends 4
unbalances instead of 3; the argmax rule was kept. A misroute costs one
spectra request that comes back clean; the phasors then follow as before.

## The change

* **Report** (`research/classifier_triage/runtime.py`): `family_scores` (all
  six heads), `waveform_family` (the largest of hif/unbalance/harmonic),
  `spectra_admitted` (the request is admitted and the waveform family is
  harmonic) and `spectra_rule`. No re-export: the tracked models already
  carry the heads; the manifest and model ids are unchanged. An unavailable
  report carries `spectra_admitted: False` (runtime and provider fallback).
* **Gate** (`psse_env/actions.py`): `triage_admits_spectra`,
  `triage_waveform_family`, `spectra_ledger`, `spectra_show_distortion`.
  Under `classifier_gated_diagnostics` the harmonic suspicion that admits
  `get_harmonic_context` and `run_hse_from_path` is now the report's
  harmonic call on the current WLS, or, as before, phasors that came back
  balanced. The process oracle, the environment and the provider all read
  that suspicion through `current_suspicion`, so the three gates agree.
* **Expert** (`psse_env/oracle/diagnostics_expert.py`,
  `classifier_screening_proposals`): on a harmonic call the spectra are the
  first tier. Spectra that show a distortion hand over to the harmonic
  ladder's HSE rung, and an accepted HSE closes the diagnosis as before.
  The phasors are acquired only when the spectra came back clean, or when
  the HSE answered and the harmonic signature still stands. Nothing changes
  for a request whose waveform family is HIF or unbalance.
* **Prompt** (`psse_env/dagger/dataset_builder.py`, the classifier-gated
  paragraph): names the new fields and the spectra-first rule, so the
  student reads the same contract the teacher demonstrates.

A harmonic root's expert episode becomes `run_wls → get_harmonic_context →
run_hse_from_path → finalize_diagnosis` (four steps) instead of six with the
phasor detour.

## Tests

`psse_env/oracle/test_classifier_gated_routing.py` gains three tests (a
harmonic call opens the spectra and the three gates agree; clean spectra are
followed by the admitted phasors and the spectra are not asked for again;
a detected distortion runs the HSE first, a rejected HSE is followed by the
phasors, an accepted HSE by no acquisition), and the report fixtures carry
the new fields. `research/classifier_triage/tests/test_runtime.py` asserts
the fields and the argmax rule. The oracle, dagger and classifier-triage
suites pass (844 tests).

## Expert end-to-end

Filled in below from `research/classifier_triage/expert_e2e.py`.

### IEEE 14, full development suite (160 roots, export `triage_gnn_ieee14_20261004`)

Baseline: the 2026-10-07 classifier cell's expert on the same suite
(`output/classifier_triage_20261004/e2e/classifier_baseline`). New run:
`output/classifier_triage_20261013/e2e_ieee14_spectra_first` (3.8 min).

| | before | spectra-first |
| --- | --- | --- |
| truth-audited success | 159/160 | 159/160 |
| phasors acquired / unnecessary | 59 / 19 | 43 / 3 |
| spectra acquired / unnecessary | 18 / 2 | 18 / 2 |
| harmonic roots: success, mean steps, phasors acquired | 16/16, 6.0, 16 | 16/16, 4.0, 0 |
| false commits / healthy components touched | 0 / 5 | 0 / 5 |

Every harmonic root ran `run_wls -> get_harmonic_context -> run_hse_from_path
-> finalize_diagnosis`. The three phasor and two spectra requests still
counted unnecessary belong to balanced roots whose exhausted routes open the
tiers (a parameter root the balanced routes could not close, a multi-meter
root, a meter+topology root), as before. The one miss is the same parameter
root (`r0_4d7aa8a5a01f`) the baseline missed.

### IEEE 118, waveform families (56 roots, joint export, progress basis `chi_square_excess`)

Baseline: `output/classifier_triage_20261011/e2e_ieee118_excess_waveform_fix`
(the run after the unbalance noise-floor fix). New run:
`output/classifier_triage_20261013/e2e_ieee118_spectra_first_waveform`
(4.8 min, 4.6 before).

| | before | spectra-first |
| --- | --- | --- |
| truth-audited success | 54/56 | 54/56 |
| phasors acquired / unnecessary | 55 / 16 | 39 / 0 |
| spectra acquired / unnecessary | 16 / 0 | 20 / 4 |
| harmonic: success, mean steps, phasors acquired | 16/16, 6.0, 16 | 16/16, 4.0, 0 |
| unbalance: success, mean steps, sent to spectra first | 16/16, 4.0, 0 | 16/16, 4.2, 3 |
| HIF, meter+HIF success | 16/16, 6/8 | 16/16, 6/8 |
| false commits / healthy components touched | 0 / 3 | 0 / 3 |

All sixteen harmonic roots ran the four-step trace. Three unbalance roots and
one meter+HIF root drew the harmonic call and paid one clean spectra request
before the phasors, then closed exactly as before. The two meter+HIF misses
are the baseline's.

### Reading

The route removes the phasor detour from every harmonic root on both
networks at no cost in success, false commits or healthy components, and the
misroutes are the clean spectra requests the offline tables predicted. For
the student this changes the teacher's demonstration: the spectra request on
a harmonic root is now one step keyed on a report field the student reads
(`spectra_admitted`), not an inference from balanced phasors plus a standing
alarm, which is the cue that did not survive the move to IEEE 118. Measuring
that needs the deferred student cells (an IEEE 118 collection round and the
reverse evaluation).

## IEEE 118 student cells (2026-10-13)

Two cells on torch measure what the route does for the frozen IEEE 14
student and what one DAgger round adds (`research/hpc/full_pipeline_20260907/
overrides/ieee118_student_20261013.env`, `ieee14_reverse_20261013.env`,
`launch_student_cells_20261013.sh`; commit 7634b46 after two launcher and
stage fixes). Cell A keeps every setting of the 2026-10-11 transfer cell and
its 124 development roots: the suite and corpora are copied from that cell,
the expert aggregate is regenerated with the spectra-first teacher, and
`check_suite_exclusion.py` confirmed the regenerated aggregate drew the same
92 roots and none of the suite's. The aggregate's harmonic demonstrations
are the four-step trace (40 rows instead of 60); the rest is row for row the
2026-10-11 aggregate. Round 1: 24 training roots, 111 safe rows, tool
disagreement 21%, a 222-row mixture, 56 optimizer steps continuing the frozen
adapter (one preemption, resumed from checkpoint 48; final evaluation loss
0.0106).

### Cell A: the same 124 IEEE 118 roots

| policy | success | harmonic | unbalance | meter+parameter | HIF | meter+HIF | false commits | false finalizations | episodes with invalid actions | mean steps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| frozen IEEE 14 student, 2026-10-11 environment | 101/124 | 0/16 | 16/16 | 15/16 | 14/16 | 6/8 | 12 on the harmonic roots alone | 0 | 13 harmonic loops | |
| frozen IEEE 14 student, spectra-first environment | 115/124 | 16/16 | 14/16 | 15/16 | 14/16 | 6/8 | 20 | 0 | 30 | 8.3 |
| adapted after one IEEE 118 round | 114/124 | 16/16 | 14/16 | 14/16 | 14/16 | 6/8 | 7 | 10 | 21 | 7.6 |
| expert | 119/124 | 16/16 | 16/16 | 16/16 | 14/16 | 6/8 | 0 | 0 | 4 | 7.0 |

Every harmonic root, for all three policies, runs `run_wls ->
get_harmonic_context -> run_hse_from_path -> finalize_diagnosis`. The frozen
student, which never requested spectra on IEEE 118 before, now does so on
all sixteen with nothing retrained: the environment change alone (the
report's harmonic call and the one-step admission it carries) closed the
whole harmonic gap, and the student's 101 became 115. The remaining nine
misses are the expert's own five (two HIF, two meter+HIF, one parameter),
two unbalance loops, one multi-meter and one meter+parameter root.

The DAgger round does not add to that. The adapted adapter scores 114: the
same roots as the frozen student except one meter+parameter root, where
after a rejected parameter correction it asks for more evidence instead of
trying the meter repair and loops. It does change habits: false commits fall
from 20 to 7 and the measurement-context requests from 166 to 102 (the
expert makes 91), HIF roots close in five steps instead of seven, but it
picks up ten premature finalizations on six HIF and meter+HIF roots after
the estimate, which cost it nothing except on a root the expert also
misses.

### Cell B: the adapted adapter on the 160 IEEE 14 roots

Cell B (`research_full_pipeline_20261013_ieee14_reverse`) links the 2026-10-07
classifier cell's expert aggregate and cell A's adapted adapter, copies that
cell's suite, collects one IEEE 14 round with the adapted adapter (122 roots,
tool disagreement 9.4%, invalid actions 0.9%, 488 safe rows) and then scores
the adapter and the expert zero-shot on the 160 development roots under the
joint gate (43 minutes).

| policy, 160 IEEE 14 roots | success | harmonic | unbalance | parameter | false commits | false finalizations | episodes with invalid actions | mean steps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IEEE 14 round-2 student on its own suite (2026-10-07 cell, IEEE 14 export) | 159/160 | | | | | | | |
| adapted adapter after one IEEE 118 round, joint gate | 156/160 | 16/16 | 13/16 | 15/16 | 8 | 6 | 19 | 7.3 |
| expert, joint gate | 159/160 | 16/16 | 16/16 | 15/16 | 0 | 0 | 0 | 7.0 |

Every other family is perfect for both policies, and every harmonic root
runs the four-step trace. The adapted adapter misses the parameter root the
expert misses too (`r0_4d7aa8a5a01f`, the same root the expert end-to-end
missed) and the three unbalance roots the joint gate sends to spectra first
on IEEE 14, the misroute the offline table predicted. The expert pays one
clean spectra request there and falls back to the phasors; the adapted
student runs the HSE on clean spectra, asks for the spectra again and loops.
Cell A's two unbalance losses are the same pattern on IEEE 118.

### Reading

* The spectra-first route is what fixed the IEEE 118 student: 101 to 115 of
  124 with nothing retrained, every harmonic root solved. The DAgger round
  on IEEE 118 did not add to the count on either network; it changed habits
  (fewer false commits and context requests, premature finalizations on HIF
  roots) and kept the IEEE 14 competence intact (156 of 160, with the three
  losses explained below).
* The one thing neither student has learned is the fallback after the
  route's misroute: clean spectra on a harmonic call must be followed by the
  phasors. The expert demonstrates it rarely (one of ten unbalance roots in
  the IEEE 118 aggregate), so the student has little to imitate. Two cheap
  remedies: a margin on the routing rule (open the spectra first only when
  the harmonic head leads the unbalance head clearly; the offline tables put
  the cost at one or two of 87 IEEE 118 harmonic roots), or an aggregate plan
  that oversamples misrouted unbalance roots so the fallback is demonstrated.


## Caveats

* The 2026-10-07 reason for hiding the heads stands as a caution: on IEEE 14
  the waveform families are separated from a bad voltage meter partly by
  simulator background. The request flag has the same exposure, and the
  heads are used here only to choose which stream an admitted request opens
  first, never to admit a request or to name a fault.
* Zero-shot across networks the routing fails with the gate (the swap above),
  so the spectra-first route is only as good as the gate's transfer; the
  joint export is what the IEEE 118 cells run.
* The misroutes (3 to 6 unbalance roots per network) cost one clean spectra
  request each; `unnecessary_spectra` in the e2e summary tracks them.

## Commands

```bash
python research/classifier_triage/head_study.py   # the effect-run heads table (output/classifier_triage_20261012/transfer_v3/head_study.json)
python research/classifier_triage/route_rule.py   # the tracked-export routing table (output/classifier_triage_20261013/route_rule.json)
python -m pytest psse_env/oracle/test_classifier_gated_routing.py research/classifier_triage/tests/test_runtime.py -q
python -m research.classifier_triage.expert_e2e --suite output/classifier_triage_20261004/suites/development.json --profile classifier_gated_diagnostics --output-dir output/classifier_triage_20261013/e2e_ieee14_spectra_first
python -m research.classifier_triage.expert_e2e --suite output/classifier_triage_20261011/suite_ieee118/development.json --profile classifier_gated_diagnostics --triage-model psse_env/oracle/models/triage_gnn_joint_20261011 --progress-basis chi_square_excess --families harmonic three_phase_unbalance hif measurement+hif --output-dir output/classifier_triage_20261013/e2e_ieee118_spectra_first_waveform
```
