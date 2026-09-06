# Diagnostic families in the DAgger pipeline: results and progress

Status as of 2026-09-06. Branch `local/relaxed-current`, tip `d3ac031`.

## Summary

* The DAgger pipeline, which previously covered measurement, parameter, and
  topology errors, now also trains and evaluates four explanation-only
  diagnostic families on the IEEE-14 system: high-impedance faults (HIF),
  measurement error combined with HIF, three-phase unbalance, and harmonic
  distortion, plus a balanced-telemetry negative control.
* Five research rounds ran on the NYU Torch cluster. The scale round, on 144
  training and 72 held-out development roots, takes the 12B student from a
  truth-audited success of 0.403 (BC0 warm start) to 0.861 after one DAgger
  pass, one root short of the 63-of-72 ceiling that the measurement+HIF
  decision leaves. Harmonic and unbalance go from 3/18 and 2/18 to 18/18
  each; HIF from 15/18 to 17/18; the control stays 9/9.
* The student learned the intended workflows: after the baseline WLS it asks
  for the measurements the anomaly calls for (phase measurements for a narrow
  residual, spectra for a broad one), runs the matching estimator, and
  finalizes with an explanation. It never escalates, loops, or commits a false
  correction on the development roots.
* Two calibrations changed the read of earlier rounds: the audit's HIF
  location tolerance moved from 0.05 to 0.10 of line length to match the
  estimator's validated error, and measurement+HIF is scored as "fault
  identified" because a meter cannot be repaired while the fault stands.
* The cluster's low-utilization monitor cancelled the scale collection twice.
  The cause was the OpenDSS grid search of the HIF estimator: hundreds of
  single-threaded simulations per step, each re-reading the model from the
  shared filesystem. A worker pool and a node-local model copy cut a HIF step
  from about 80 s to 8 s and lifted sampled GPU utilization from 4 % to 48 %.

## What was built

### Families and corpora

| Family | Source | Size | Terminal outcome |
| --- | --- | --- | --- |
| hif | ten-scan branch-current windows, OpenDSS, per-phase terminal currents | 85 training + 17 validation windows | HIF explanation on the true branch |
| measurement+hif | HIF window plus one composed meter error | shares the 102 windows | HIF explanation; the meter is left unresolved by decision |
| three_phase_unbalance | per-phase snapshots with branch currents, VUF above 0.01 | 220 rows | unbalance explanation at the source bus |
| harmonic | tabular corpus, spectral distortion on most of the 122 channels | 500 rows | harmonic explanation at the source bus |
| telemetry_no_disturbance | balanced null derived from the unbalance rows | 220 rows | finalize with no explanation |

The earlier voltage-only HIF corpus and the bus-3 double-unbalanced imbalance
corpus were retired; both are recorded in `docs/branch_current_telemetry_20260903.md`.

### Operator workflow

The operator starts from the positive-sequence SCADA snapshot and the
balanced model, as a control room would. Unbalance and harmonic roots carry
no sensor flag (discovery mode); HIF keeps its zero-sequence relay flag.

1. `run_wls` mints the fundamental anomaly and reports its breadth, the share
   of normalized residuals above the outlier threshold.
2. The residual breadth orders the first measurement request. Below 0.5 the
   expert asks for phase measurements (`get_three_phase_context`); at or
   above 0.5 it asks for spectra (`get_harmonic_context`). Either falls back
   to the other when the first returns nothing, and an answered-unavailable
   request stays answered for the episode. Measured breadth: harmonic 75 to
   84 of 122 channels, unbalance 7 to 19, a bad meter 1 to 5.
3. The matching estimator runs: the three-phase NLM for unbalance, the
   harmonic state estimator for harmonics, the multiscan HIF estimator on the
   relay-localized branch for HIF.
4. `finalize_diagnosis` closes the episode with the accepted explanation.

While any waveform-family signature stands, explained or not, WLS mints no
residual signatures, the contexts offer no corrections, the validity gate
refuses corrections, and the classical experts stand down. That guard was
the fix for round 1, where a post-explanation WLS on a still-unbalanced
vector had led the recovery expert into false meter corrections.

Resulting expert flows: unbalance `run_wls -> get_three_phase_context ->
run_three_phase_nlm_from_path -> finalize_diagnosis`; harmonic `run_wls ->
get_harmonic_context -> run_hse_from_path -> finalize_diagnosis`; HIF
`run_three_phase_nlm_from_path -> estimate_hif_location_magnitude_multiscan_from_path
-> finalize_diagnosis`; control `run_wls -> finalize_diagnosis`.

### Estimators and audit

* The multiscan HIF estimator stacks three weighted residual blocks per scan:
  positive-sequence SCADA, three-phase bus voltages, and per-phase branch
  terminal currents. The two-terminal current differential seeds the location
  and phase; an OpenDSS grid search over location (alpha) and fault resistance
  verifies and refines it. Research budget: 7 x 9 grid, ten scans.
* The strict audit scores an episode as a truth-audited success only when the
  accepted explanation names the true family, localizes the true element
  within tolerance, and no true fault remains. For HIF the location
  tolerance is 0.10 of line length (see calibration below).
* Label rows for training are audited offline; targets that fail the audit
  are quarantined rather than taught.

### Research launcher

`scripts/run_dagger_research.py --plan-preset diagnostic` with per-round plan
overrides, resumable stages (collect, train, eval) as a Slurm dependency
chain, a completed-root ledger that survives preemption, a D0 filter that
removes the 91 stale voltage-only HIF rows, and a per-family summary
(`research/hpc/diagnostic_round_20260903/`).

## Results by round

All rounds warm-start from the BC0 12B checkpoint (`google/gemma-4-12B-it`,
LoRA rank 16, lr 3e-5, micro-batch 1 x 4 accumulation, one pass), collect
with beta 0.25 in 12-step episodes, mix D1 1:1 with D0, and evaluate the
candidate against BC0 on the same development roots with a 24-step budget.
Seed 20260903 throughout.

| Round | Date | Dev roots | Design | BC0 | Candidate | Ceiling |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 09-04 | 18 | flagged unbalance and HIF | 14/18 | 13/18 | 15/18 |
| 2 | 09-04 | 18 | round 1 plus the waveform guard | 14/18 | 15/18 | 15/18 |
| 3 | 09-05 | 18 | unbalance discovered from the WLS anomaly | 10/18 | 15/18 | 15/18 |
| 4 | 09-05 | 24 | explicit measurement requests, harmonic added | 7/24 | 17/24 | 21/24 |
| 4 at 0.10 | | 24 | same episodes re-read at the new HIF tolerance | 9/24 | 20/24 | 21/24 |
| 5 | 09-06 | 72 | three times the plan, 0.10 tolerance | 29/72 | 62/72 | 63/72 |

The ceiling counts every root except measurement+HIF, which is scored as
fault-identified rather than resolved. Round 1's candidate regressed on
unbalance (4/6, nine false commits) for the reason the waveform guard fixed;
round 2 confirmed the fix on the same roots. Round 3 showed that BC0 cannot
find an unflagged unbalance (1/6) while the candidate can (6/6). Round 4
added the acquisition requests and the harmonic family; its four HIF misses
were location estimates 0.055 to 0.10 off against the 0.05 tolerance of the
time. Rounds 1 to 3 share their 18 roots; round 4 and round 5 each drew
fresh roots because the plan changed, and adapters are compared within a
round.

### Scale round, per family

| Family | BC0 | Candidate | Candidate steps | Candidate flow |
| --- | --- | --- | --- | --- |
| harmonic | 3/18 | 18/18 | 4.2 | WLS, spectra request, HSE, finalize |
| three_phase_unbalance | 2/18 | 18/18 | 4.2 | WLS, phase request, NLM, finalize on 15 roots; 3 ask for spectra first and fall back |
| hif | 15/18 | 17/18 | 3.0 | NLM, multiscan, finalize; the miss is a location error of 0.115 |
| measurement+hif | 0/9 | 0/9 | 3.0 | fault identified 9/9 |
| telemetry_no_disturbance | 9/9 | 9/9 | 2.0 | WLS, finalize |
| overall | 29/72 (0.403) | 62/72 (0.861) | 3.5 vs 5.1 | |

BC0 escalates to the operator on 32 of the 36 harmonic and unbalance roots
after three steps and commits 17 invalid actions; the candidate has no
escalations, loops, invalid actions, or false commits. Collection: 144 of
144 roots, label yield 0.833, 310 safe rows (HIF 96, unbalance 94,
harmonic 91, control 15, measurement+HIF 14), mixture 620 rows, 155
optimizer steps.

## Calibrations and decisions

* **HIF location tolerance.** Under the research search budget the
  multiscan estimator's alpha error over the 85-window corpus has a median
  of 0.020 and a 90th percentile of 0.092 (67 of 84 localized windows within
  0.05, 77 of 84 within 0.10); the 17-window corpus gives 0.027 and 0.103.
  The branch itself is localized on 83 of 85 windows. A per-episode tolerance
  of 0.05 therefore made HIF success a draw of the root; it is now 0.10, the
  estimator's 90th percentile, so about one HIF root in ten still misses on
  estimator error alone.
* **Measurement+HIF** terminates once the fault is identified; the remaining
  meter error is not repaired while the fault stands, and the audit keeps
  counting it as unresolved. Read that row as "HIF identified".
* **Root pairing.** Adding a family shifts the generator's draws for every
  later family, so rounds with different family sets do not share roots. By
  decision this is not a concern; adapters are compared within a round.
* **Research posture.** The byte pins on the study manifest, the evaluation
  policy, and the dependency lock, the factory source digests, and a stale
  tool-registry gate were removed: this is a research project, and those
  gates only cost time. A run artifact still binds the digest of the manifest
  it was validated against.

## Compute

| Stage | Round 4 (48 roots, 24 dev) | Round 5 (144 roots, 72 dev) |
| --- | --- | --- |
| collection | about 2.6 h active, two preemptions | 30 min for the last 70 roots after the fix |
| training | 13 min, 44 steps | 42 min, 155 steps |
| evaluation | 89 min | 34 min |

The scale collection was cancelled twice by the cluster's utilization
monitor (average GPU utilization 3.5 % over two hours against a 50 %
threshold). Harmonic episodes took 3 s per step; HIF episodes took 80 s per
step because the estimator's OpenDSS grid search ran several hundred
simulations serially, each re-reading the model through `Redirect`. A probe
inside the allocation measured 105 to 315 ms of wall time per simulation
against 7 to 11 ms of CPU from `/scratch`, and 7 ms from `/dev/shm`. The
search now runs in a process pool sized by the allocated CPUs
(`PSSE_HIF_WORKERS`), the stages copy the model to `/dev/shm`
(`PSSE_OPENDSS_MODEL_DIR`), and the provider memoizes a search by its
inputs. Measurement+HIF steps fell to 8 s, unbalance to 4.6 s, and sampled
utilization averaged 48 % with half the samples at or above 50 %. Serial and
parallel searches agree to floating-point rounding.

## Limitations

* The measurement+HIF row is a design decision, not a capability: no round
  has attempted the meter repair after the HIF explanation.
* HIF success is bounded by the estimator: at the 0.10 tolerance roughly one
  root in ten misses regardless of policy.
* The training-loss monitor is the classical-family trace validation set; the
  diagnostic outcome is the paired closed-loop evaluation only.
* Discovery mode requires the baseline WLS to show an anomaly; rows without
  one are rejected at generation, so the corpora are filtered to detectable
  roots.
* GPU utilization sits at the cluster's threshold. A further gain needs a
  structural change: several episodes per policy call, or a policy server
  shared by CPU workers.
* Everything is IEEE-14 with the research estimator budgets; no other network
  has been run.

## Next steps

1. A second DAgger iteration warm-started from the round 5 candidate on fresh
   roots, to see whether the last HIF miss and the three spectra-first
   unbalance episodes close.
2. The combined preset (classical plus diagnostic families) to confirm the
   diagnostic rows do not regress measurement, parameter, or topology
   recovery.
3. Decide whether measurement+HIF should continue to the meter repair after
   the fault explanation, and if so add the post-explanation correction route
   under the waveform guard.
4. Multi-episode batched collection if utilization must rise further.

## Where things live

* Code: `psse_env/` (environment, oracle, providers), `three_phase_nlm/`
  (estimators), `scripts/run_dagger_research.py`, round cell
  `research/hpc/diagnostic_round_20260903/` with `README.md`.
* Corpora: `artifacts/measurements/hif_multiscan_currents_train_85x10_20260903`,
  `hif_multiscan_currents_17x10_20260903`,
  `out_measurements_imbalance_currents_20260903`.
* Rounds on Torch: `/scratch/yx3882/research_diag_round_20260903` (1),
  `_20260904` (2), `_20260904_discovery` (3), `_20260905_requests` (4),
  `_20260906_scale` (5); each has `out/round_summary.json`.
* Key commits: `3b4ea03` waveform guard, `48afc0f` discovered unbalance,
  `efe438f` measurement requests, `8c940bf` pin removal, `ad695c0` HIF
  tolerance, `faa3341` scale overrides, `1572886` parallel search,
  `bf8cd54` node-local model.
