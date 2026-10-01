# Hypothesis ranking, step 2: the acquisition rule and the screen that feeds it (2026-09-30)

Step 2 of `docs/hypothesis_ranking_plan_20260930.md`, built on the step 1
findings (`docs/hypothesis_ranking_step1_20260930.md`) and the user's three
decisions: option A (an unexplained balanced alarm may open phasors), C2
(spectra as a second tier once phasors come back balanced) and D3 (a leading
voltage-meter hypothesis opens phasors before the meter is edited). Everything
here is on `codex/cleanup-20260928`, uncommitted; research only.

## 1. What changed, in one table

| Layer | Before (2026-09-27 contract) | After (step 2) |
| --- | --- | --- |
| Balanced screen (`psse_env/providers/hif_screen.py`) | four single-cause refits, two rounds, continue only after a meter win | adds a joint R and X parameter variant; rounds continue after a parameter or topology win (three comparisons, HIF compared while at most one meter is set aside); reports `explained`, `unexplained`, `voltage_meter_channels`, `accepted_hypotheses`, `phasor_suspicion` |
| What opens phasors | an HIF win | an HIF win, a phase-A voltage channel set aside as a bad meter, or an alarm no hypothesis sequence explains (`DIAGNOSTIC_SUSPICION_REQUIREMENTS`: `phasor`) |
| What opens spectra | nothing (refused) | phasors acquired on the state came back balanced (`harmonic`) |
| HIF estimator | the screen's HIF suspicion | the screen's HIF suspicion, or an HIF-like zero-sequence differential the acquired phasors showed (the NLM mints `hif_suspected_zero_sequence line=L`) |
| Voltage-meter edits | allowed like any meter edit | wait for the phasors (`correction_route_not_actionable`, detail `measurement_voltage_meter_correction_requires_phase_measurements`, repair = the phasor request) |
| Expert ordering | HIF suspicion acts at once; unbalance and harmonic roots end in an uncredited handoff | HIF and voltage-meter suspicions act at once; an unexplained alarm opens phasors after the balanced routes are exhausted, spectra after balanced phasors (at once when the screen already called the alarm unexplained); the handoff becomes `operator_escalation:unexplained_balanced_discrepancy` once both tiers answered |
| Audit | bounded branch handoff credited | adds the basis `unexplained_discrepancy_handoff`: a correction-free handoff after both tiers, with only waveform-family truth left and no healthy component changed |
| Every root carries | PMU phasors of its true state at 1e-4 | phasors and noise-only spectra (14 monitors, orders 5 to 19, the harmonic traces' per-component sigma), so neither request answers by availability |

The WLS ledger now also carries `bus_count`, which is how the process gate
tells a voltage channel from a power channel. The policy-visible compact
screen report gained `explained`, `unexplained`, `accepted_hypotheses`,
`voltage_meter_channels`, `phasor_suspicion` and `hif_variant`; two new
signatures ride on the WLS solve, `wls_voltage_meter_suspected index=I` and
`wls_unexplained_balanced_discrepancy`, neither carrying a waveform marker, so
neither blocks the balanced routes by itself.

## 2. Screen changes, measured on the step 1 dataset

The step 1 states (2,398 roots, 1,769 truth-corrected children, 69 alarmed
healthy windows, 200 mimics, same parents and split) were re-run through the
screen at each stage (`build_dataset --reanalyze`): v1 is the step 1 screen,
v2 adds the joint R/X variant, the continued rounds and an HIF variant with
the end buses' phase-A voltage channels set aside, v3 is v2 with that HIF
variant switched off. v3 is what ships.

| family | alarmed | unexplained v1 | unexplained v3 | HIF flags v3 | voltage-meter picks v3 | rule v3 acquisitions |
| --- | --- | --- | --- | --- | --- | --- |
| measurement | 250 | 0 | 0 | 0 | 33 | 33 |
| multi_measurement | 224 | 164 | 113 | 0 | 28 | 126 |
| parameter | 250 | 47 | 0 | 0 | 1 | 1 |
| measurement+parameter | 250 | 166 | 0 | 0 | 1 | 1 |
| topology | 250 | 71 | 55 | 0 | 11 | 55 |
| measurement+topology | 250 | 228 | 0 | 0 | 0 | 0 |
| harmonic | 250 | 224 | 208 | 0 | 248 | 249 |
| hif | 131 | 46 | 0 | 130 | 0 | 130 |
| measurement+hif | 131 | 54 | 0 | 128 | 0 | 128 |
| three_phase_unbalance | 162 | 108 | 77 | 0 | 160 | 161 |
| healthy windows (alarmed) | 69 | 8 | 2 | 1 | 10 | 13 |

Rule v3 is the approved rule: an HIF win, an unexplained alarm, or a
voltage-meter pick. Read from the first alarm it reaches 161 of 162 unbalance
and 249 of 250 harmonic roots, keeps every HIF flag (130 of 131 and 128 of
131, each on the true line), and fires on 216 of the 1,474 alarmed roots whose
truth needs no phasors (14.7%). Most of those are multi-meter roots with four
or more meters (113) and bus splits (55), which the expert reaches only after
its balanced routes, so the expert's actual acquisitions are lower (section
4); the rest are the single-meter roots whose true bad channel is a voltage
reading (33 of 250) and the alarmed healthy windows (13 of 69).

What the two shipped screen changes bought: parameter roots with both R and X
wrong went from 47 unexplained to 0; mixed roots now name both components
(measurement+parameter 243 of 250 against 92, measurement+topology 248 of 250
against 22, measurement+HIF 119 of 131 as before); a misranked parameter root
has its true branch first in the screen's refit ranking on 5 of 8 roots where
the multiplier ranking has it first on 0; HIF flags did not move (130 of 131,
no false flag on any of the 1,474 non-HIF roots; 1 of 69 healthy alarms as
before); the mean screen time rose from 0.96 s to 1.16 s per alarm.

The HIF variant with phase-A voltage offsets was switched off after
measurement: on the same states it added no HIF detection (130 of 131 either
way) and flagged 61 non-HIF roots as HIF (31 unbalance, 16 harmonic, 7
multi-meter, 7 topology). It stays in the code behind
`HifScreenConfig.hif_vm_offset_variant` for the IEEE 57 work, where the
voltage-meter suspicion covers the same physics.

Two reference points for the ranker (step 4) are unchanged: two flow meters of
one line biased the same way are flagged as an HIF on that line 37 times in
100, and the same pair biased in opposite directions is explained as a branch
parameter 31 times in 100 by the new joint R/X variant.

## 3. Verification

| Suite | Result |
| --- | --- |
| `tests/test_hif_screen.py` (the 2026-09-26 decisions pinned on real roots) | 6 pass, unchanged |
| `tests/test_hif_screen_offline_fields.py` (study fields, v2 outputs, voltage-meter suspicion, compact report untouched) | 5 pass |
| `tests/test_suspicion_gated_contract.py` (rewritten: the `phasor`/`hif`/`harmonic` requirements, the process gate including the D3 edit hold and its repair, phasors and spectra on every family, the expert on nine roots of seven families) | 9 pass |
| `psse_env/oracle` (all), providers, WLS-gated and SCADA-only boundaries and generation, research DAgger minimal, release audit, HPC pipeline test | 578 pass, 184 subtests |

## 4. Expert end to end, every family

The rule-based expert on 48 fresh roots, 4 per family, seed 20260930, the
40-step budget, the research environment's deployment-mode (observable-only)
verifier
(`research/hypothesis_ranking/expert_e2e.py`, results under
`output/hypothesis_ranking_20260930/expert_e2e_v2/`). Before this step the
same expert could not credit an unbalance or harmonic root at all.

| family | roots | truth-audited success | terminal outcome | phasors | NLM | spectra | HSE | mean steps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| no_error | 4 | 4 | resolved | 0 | 0 | 0 | 0 | 2.0 |
| telemetry_no_disturbance | 4 | 4 | resolved | 0 | 0 | 0 | 0 | 2.0 |
| measurement | 4 | 4 | post-correction handoff | 0 | 0 | 0 | 0 | 7.0 |
| multi_measurement | 4 | 4 | post-correction handoff | 0 | 0 | 0 | 0 | 19.0 |
| parameter | 4 | 4 | post-correction handoff | 0 | 0 | 0 | 0 | 7.0 |
| measurement+parameter | 4 | 4 | post-correction handoff | 0 | 0 | 0 | 0 | 11.0 |
| topology | 4 | 4 | post-correction handoff | 2 | 2 | 2 | 0 | 9.5 |
| measurement+topology | 4 | 4 | post-correction handoff | 0 | 0 | 0 | 0 | 12.0 |
| hif | 4 | 4 | resolved | 4 | 4 | 0 | 0 | 6.0 |
| measurement+hif | 4 | 4 | post-correction handoff | 4 | 4 | 0 | 0 | 11.0 |
| three_phase_unbalance | 4 | 4 | resolved | 4 | 4 | 0 | 0 | 4.0 |
| harmonic | 4 | 4 | resolved | 4 | 4 | 4 | 4 | 6.0 |

Every episode passes the truth audit on the counterfactual-resolution basis;
no false commit; 18 phasor acquisitions, of which 6 on roots whose truth needs
none (2 bus-split topology roots through the unexplained tier, the rest the
measurement+HIF roots, which need them); 6 spectra requests, 2 of them on the
same two bus-split roots, which the topology route then resolved anyway. The
traces are the intended ones: an unbalance root runs `run_wls`,
`get_three_phase_context`, `run_three_phase_nlm_from_path`, `finalize_diagnosis`
(the voltage-meter suspicion opens the phasors, the NLM explains the source);
a harmonic root adds `get_harmonic_context` and `run_hse_from_path` before
finalizing (the screen called the alarm unexplained, the phasors came back
balanced, the spectra followed at once; with the spectra ordered after the
balanced routes the same roots took 22 to 24 steps and five rejected meter
edits each, which is why the early tier was added). The `healthy_components_preserved`
flag is false on the three bus-split topology roots whose correct breaker
correction re-renders the operator model; their truth audit passes, and the
flag is the pre-existing comparison of a re-rendered case, not a wrong edit.
No episode ended with the new `unexplained_balanced_discrepancy` request:
every root in this draw was explained by one of the tiers.


## 5. IEEE 57 with the final screen

The 1,492 IEEE 57 study roots of 2026-09-11 (four OpenDSS models) were
re-screened with the final screen, candidate lines again supplied explicitly
(case57 carries no base kV); results under
`output/hypothesis_ranking_20260930/ieee57_v3/`. The WLS layer reproduces the
study on every root as before; 184 roots alarm under the repository rule. A
57-bus screen now takes 79 s per alarmed root (three rounds, 80 branches with
three parameter variants each, 65 candidate lines), which is a cost to keep
in mind for any 57-bus episode.

| family | alarmed | HIF flags (on true line) | voltage-meter picks | unexplained | rule v3 acquisitions | step 1 screen (HIF flags) |
| --- | --- | --- | --- | --- | --- | --- |
| hif | 112 | 19 (17) | 46 | 33 | 73 | 16 |
| three_phase_unbalance | 46 | 0 | 27 | 10 | 28 | 0 |
| healthy | 26 | 0 | 3 | 0 | 3 | 0 |

By resistance the picture is the balanced-screen bound of 2026-09-10: at
R = 10 pu the rule reaches 72 of 93 alarmed HIF roots (the shunt alone flags
18), at 100 pu 1 of 10, at 1000 pu none of 9 (and those faults barely alarm,
19 of 504). The HIF class alone still loses to the voltage-meter explanation
on this network (46 of 112 alarmed HIF roots are explained as a phase-A
voltage meter), which is exactly the case D3 covers: those roots now acquire
phasors, and the zero-sequence screen on the phasors is what identifies the
fault. Two of the 19 HIF flags name a wrong line, the first wrong-line flags
seen on either network; both are 10 pu faults on this 57-bus model, and they
cost an acquisition the phasors then correct. Unbalance roots reach the
phasors in 28 of 46 cases (delta 0.2: 25 of 35; delta 0.05: 3 of 11, where
the balanced alarm itself is marginal). Healthy windows that alarm (26 of
400 on this model, the OpenDSS-versus-case57 baseline offset the step 1
report describes) acquire phasors in 3 cases and never spectra.

The transfer conclusion is unchanged in kind but better in degree: on IEEE 57
the acquisition rule, not the HIF class, carries the HIF detection, and the
weak faults (100 pu and above) remain below what any single balanced snapshot
can show.


## 6. What step 3 starts from

- The ledger expert replaces three special cases with one hypothesis ledger
  over the screen's ranked candidates; the screen now hands it the accepted
  sequence (`accepted_hypotheses`), the per-class ranked targets and whether
  the sequence explains the alarm.
- Every verdict above is an observable-only one: the research environment
  verifies candidates in deployment mode, which ignores the truth the
  environment passes along (the plan's first version said otherwise; corrected
  2026-10-01). The D3 hold closes the one false-commit path the step 1 data
  exposed (a voltage-meter edit on an unbalance root), and the audit now
  scores an honest handoff after both tiers.
- Open items carried forward: bus-split roots reach the phasor tier through
  `unexplained` when the topology route does not resolve them first; the same-sign
  flow-meter pair still mimics an HIF (an acquisition, refuted by the
  zero-sequence screen); the 57-bus screen needs candidate lines from
  somewhere other than base kV.
