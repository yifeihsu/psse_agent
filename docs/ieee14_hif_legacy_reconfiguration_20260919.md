# IEEE14 legacy-stack physical-ohm HIF reconfiguration

Implemented 2026-09-20 from the 2026-09-19 revision plan, in the PSSE_Agent checkout.

## Decision and scope

Fault severity is specified in physical ohms at the line's local nominal voltage. The checked-in OpenDSS files remain a normalized 1 kV LL, 100 MVA circuit. Their historical 132/33/11 kV comment annotations do not set engine voltage bases. Convert with:

```
R_pu = R_physical_ohm / (kV_local_LL**2 / 100)
R_model_ohm = R_pu * 0.01
```

Physical kV and physical ohms never enter model-ohm injector or legacy NLM parameters. The injector checks this boundary. Shunt, load, tap and network pu parameters are preserved.

| Buses | kV LL | Zbase (ohm) | Ibase (A) | Nominal phase kV |
| --- | ---: | ---: | ---: | ---: |
| 1-5 | 69 | 47.6100 | 836.740 | 39.8372 |
| 6-7, 9-14 | 13.8 | 1.9044 | 4183.698 | 7.9674 |
| 8 | 18 | 3.2400 | 3207.501 | 10.3923 |

## Resistance conversion and populations

Nominal current below is Vphase/R, not a solved fault current or an arcing model.

| Physical ohm | Approximate current at 69 kV (A) | pu at 69 kV | pu at 18 kV | pu at 13.8 kV |
| ---: | ---: | ---: | ---: | ---: |
| 50 | 796.743 | 1.0502 | 15.4321 | 26.2550 |
| 100 | 398.372 | 2.1004 | 30.8642 | 52.5100 |
| 200 | 199.186 | 4.2008 | 61.7284 | 105.0200 |
| 500 | 79.674 | 10.5020 | 154.3210 | 262.5499 |
| 1000 | 39.837 | 21.0040 | 308.6420 | 525.0998 |
| 2000 | 19.919 | 42.0080 | 617.2840 | 1050.1995 |
| 5000 | 7.967 | 105.0200 | 1543.2099 | 2625.4988 |

The old 20-200 pu generation band corresponds to 952-9,522 ohm at 69 kV, or 38-381 ohm at 13.8 kV. The old 5-1000 pu estimator box corresponds to 238 ohm through 47.6 kOhm at 69 kV. A resistance of 1000 pu is not a 1000 ohm HIF.

Main training uses 100-1000 ohm on seven 69 kV lines, in 100-200, 200-500 and 500-1000 ohm bands. Full-corpus weights are 1:1:1; the early curriculum can use 2:2:1. Detection-limit evaluation uses 1000-5000 ohm. The discrete evaluation covers 50, 100, 200, 500, 1000, 2000 and 5000 ohm at all sixteen same-voltage lines and all three fault phases. Cross-voltage Line.7-8 is excluded from new generation but remains accepted by legacy estimators on its 13.8 kV from-bus base.

Classification boundaries are lower-inclusive:

| Physical resistance (ohm) | Classification |
| --- | --- |
| below 50 | low_resistance_fault |
| 50 to below 100 | moderately_resistive |
| 100 to below 200 | moderately_high_resistance |
| 200 to below 500 | representative_hif |
| 500 to below 1000 | weak_hif |
| 1000 to below 5000 | extreme_weak_hif |
| 5000 and above | near_open_circuit |

## Labels, tools and observation conventions

New labels carry `resistance_units="ohm_local_base"`; `r_hif_ohm` is physical, `r_hif_model_ohm` is normalized-model ohms, and `r_hif_pu` uses the local base. They declare `local_kv_ll`, `local_kv_ln`, `zbase_ohm`, `voltage_base_profile`, `resistance_class`, `expected_detectability`, `voltage_stratum` and nominal current/power. The legacy `kv_ln` field stays at the model value 0.57735 with explicit semantics. Read all old and new labels through `three_phase_nlm.hif_units.label_*`.

Estimator results report physical ohms, amperes and siemens; kW remains invariant. Their `estimated` and `search` payloads preserve local bases, units and the actual search bounds. Physical defaults are 50-5000 ohm, converted separately for each candidate line. Explicit pu bounds are optional overrides; mixing units fails. `MatpowerDeploymentProviders(hif_resistance_search="legacy_pu")` selects the historical 5-1000 pu box for replay.

The exporter historically included the bus-9 capacitor in measured Qinj while WLS also kept it in Ybus. New physical corpora use `measurement_convention.shunt_convention="ybus"` to exclude capacitor injection from makeSbus measurements. All scans, scenario metadata, estimator simulations and conditioned replays preserve that declaration. An absent marker retains `legacy_injection`. Physical KCL validation adds the actual capacitor injection back only when comparing bus injections with external branch terminal powers.

Tools, SFT context hydration, eval schemas and provider summaries preserve physical units. The NLM bridge's `r_hif_ohm` argument continues to mean model ohms; its context also provides separately named physical resistance. Registry-schema hash drift is recorded rather than blocking research evaluation, while action/schema and physical checks remain active.

## Reproduction

Only new corpus directories are generated. Historical corpora and checked-in DSS files are preserved. From the repository root:

```powershell
$G = 'Transmission/generate_measurements_hif_ieee14.py'
$A = 'artifacts/measurements'
python $G --out "$A/hif_physical69_main_train_84x10_20260919" --n-hif 84 --n-no-error 20 --seed 20260919 --scans-per-window 10 --resistance-units ohm --r-hif-ohm-bands '100:200,200:500,500:1000' --r-hif-ohm-band-weights '1,1,1' --voltage-stratum 69kv
python $G --out "$A/hif_physical69_main_valid_21x10_20260919" --n-hif 21 --n-no-error 5 --seed 20260920 --scans-per-window 10 --resistance-units ohm --r-hif-ohm-bands '100:200,200:500,500:1000' --r-hif-ohm-band-weights '1,1,1' --voltage-stratum 69kv
python $G --out "$A/hif_physical69_detection_limit_21x10_20260919" --n-hif 21 --n-no-error 0 --seed 20260921 --scans-per-window 10 --resistance-units ohm --r-hif-ohm-min 1000 --r-hif-ohm-max 5000 --voltage-stratum 69kv
python $G --out "$A/hif_physical_sweep_eval_336x10_20260919" --n-hif 336 --n-no-error 0 --seed 20260922 --scans-per-window 10 --resistance-units ohm --r-hif-ohm-sweep '50,100,200,500,1000,2000,5000' --voltage-stratum all_same_voltage
```

For each new directory run `scripts/validate_hif_samples.py` with `--meta` and `--allow-non-top3-detectability weak,extreme`, `scripts/validate_hif_multiscan_dataset.py` with `--meta --strict-physics --output`, and `scripts/validate_branch_current_localization.py --output`. Weak localization results remain reported; the exemption does not waive physical validity.

## Results

Generated on 2026-09-20. All four requested populations are complete: **462 HIF windows, 4,620 scans and 25 healthy controls**. The sweep covers every one of the 336 line/resistance/phase cells exactly once. Main train and validation populations have no shared physical-root fingerprints. All 4,620 clean scans replayed exactly under their declared convention, with no physical-validation or WLS solve failures. The preserved [generation, test and audit evidence](../output/hif_physical_revision_20260920/) includes commands and source snapshots.

| Corpus | HIF windows | Healthy controls | Legacy NLM top-1 / top-3 | Reference-scan WLS alarms | Discovered-mode scenario admissions |
| --- | ---: | ---: | ---: | ---: | ---: |
| Main training | 84 | 20 | 84 / 84 | 37 | 25 |
| Main validation | 21 | 5 | 21 / 21 | 10 | 7 |
| Detection limit | 21 | 0 | 21 / 21 | 0 | 0 |
| Full sweep | 336 | 0 | 332 / 336 | 67 | 51 |

The main training band counts are **23 / 30 / 31** for 100-200 / 200-500 / 500-1000 ohm; validation counts are **8 / 5 / 8**. These are independent draws with equal band weights. `expected_detectability` counts are representative=53, weak=31 for training; representative=13, weak=8 for validation; extreme=21 for the detection-limit corpus. These tags describe the configured population, not measured WLS visibility.

WLS uses sigma(Vm)=0.001 pu and sigma(P/Q)=0.01 pu, exactly matching added noise. The alarm is the inclusive OR of the 1% chi-square gate and max absolute normalized residual >=4. There are 122 measurements, 27 states and 95 dof; the global threshold is 129.972679. The following table counts the first scan of each window. All-scan rates are provided separately because ten scans within an event are not ten independent faults.

| kV LL | Ohm | Windows | NLM top-1 | NLM top-3 | WLS first-scan alarms | Paired healthy alarms | New first-scan alarms | All-scan WLS alarms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 13.8 | 50 | 27 | 27 | 27 | 1 | 1 | 0 | 5/270 |
| 13.8 | 100 | 27 | 27 | 27 | 0 | 0 | 0 | 2/270 |
| 13.8 | 200 | 27 | 27 | 27 | 1 | 1 | 0 | 5/270 |
| 13.8 | 500 | 27 | 27 | 27 | 0 | 0 | 0 | 2/270 |
| 13.8 | 1000 | 27 | 27 | 27 | 0 | 0 | 0 | 3/270 |
| 13.8 | 2000 | 27 | 27 | 27 | 1 | 1 | 0 | 7/270 |
| 13.8 | 5000 | 27 | 27 | 27 | 0 | 0 | 0 | 5/270 |
| 69 | 50 | 21 | 18 | 21 | 21 | 0 | 21 | 210/210 |
| 69 | 100 | 21 | 20 | 21 | 21 | 0 | 21 | 210/210 |
| 69 | 200 | 21 | 21 | 21 | 18 | 1 | 17 | 188/210 |
| 69 | 500 | 21 | 21 | 21 | 3 | 1 | 2 | 10/210 |
| 69 | 1000 | 21 | 21 | 21 | 1 | 1 | 0 | 4/210 |
| 69 | 2000 | 21 | 21 | 21 | 0 | 0 | 0 | 3/210 |
| 69 | 5000 | 21 | 21 | 21 | 0 | 0 | 0 | 5/210 |

The 25 independently generated healthy controls produced **1/25 alarms (4%)**. Paired no-HIF controls use the same OpenDSS operating point and exact same SCADA noise as each fault's first scan: main training **1/84**, validation **0/21**, detection limit **0/21**, sweep **6/336**, totaling **7/462 (1.52%)**. The entire sweep's 13.8 kV group produced **no new first-scan alarms**. Shared healthy/fault alarms are not attributed to the fault. The healthy-control populations and their denominators are different and should not be pooled without explanation.

**One strict ranking check remains failed as a scientific finding.** The prescribed `validate_hif_samples.py` command fails on four sweep rows because the target is not top-1: Line.1-2 at 50 ohm on A/B/C and at 100 ohm on A. All four targets remain in top-3. Their observations, rankings and original failed-validator log are retained. The source units, topology, noise and physical replay all pass. No ranking was replaced by the hidden target and no trace was deleted to make this check pass. Legacy NLM results also do not represent autonomous diagnosis from the noisy acquisition: the separate branch-current report records the actual noisy-telemetry localization results.

Discovered-mode admission applies the current stricter **1.25 anomaly margin**, so its **25/84 and 7/21** main-corpus admissions are lower than the raw WLS alarm counts. [The admission manifest](../output/hif_physical_revision_20260920/detection_audit/main_hif_admission_manifest.jsonl) preserves accepted and rejected IDs and reasons. This gate does not certify a successful expert rollout or a training-ready SFT trace. Quiet detection-limit cases remain evaluation data.

**HPC capacity finding:** the new main pool contains 105 physical HIF windows, but only 32 pass this discovered-mode gate before source partitioning. The existing D0 request for 38 pure HIF roots exceeds even this unpartitioned accepted pool; all-round plans require still more. Corpus paths and argument plumbing are implemented, but those quotas need a larger observable pool or a revised study plan before launch. This work does not silently change the requested resistance distribution, reduce admission standards or launch a cluster job.

### 2026-09-21 expansion of the 69 kV main population

The capacity shortfall was resolved by generating more windows of the same population rather than by relaxing admission. Two supplementary corpora use the identical recipe (69 kV lines, 100-200 / 200-500 / 500-1000 ohm bands with weights 1:1:1, location 0.25-0.75, all phases, ten diverse scans, shunt convention ybus) with independent seeds 20260923 and 20260924:

```
python Transmission/generate_measurements_hif_ieee14.py --out artifacts/measurements/hif_physical69_main_train_extra_252x10_20260921 --n-hif 252 --n-no-error 60 --seed 20260923 --scans-per-window 10 --resistance-units ohm --r-hif-ohm-bands "100:200,200:500,500:1000" --r-hif-ohm-band-weights "1,1,1" --voltage-stratum 69kv
python Transmission/generate_measurements_hif_ieee14.py --out artifacts/measurements/hif_physical69_main_valid_extra_63x10_20260921 --n-hif 63 --n-no-error 15 --seed 20260924 --scans-per-window 10 --resistance-units ohm --r-hif-ohm-bands "100:200,200:500,500:1000" --r-hif-ohm-band-weights "1,1,1" --voltage-stratum 69kv
```

Generation took 148 s for the 252-window corpus. Both corpora passed the strict-physics replay (2,520 and 630 clean scans), the branch-current localization report, and the sample validator; the sample validator records one top-1 ranking miss in the training corpus (Line.1-2, phase B, 105.9 ohm, ranked behind Line.1-5, in top-3), the same slack-adjacent pattern as the four sweep misses. Every window is a new physical root: no fingerprint (line, phase, location, resistance, load scale) repeats within the new corpora or against the 2026-09-19 corpora. Window ids reuse the generator's default numbering, so ids repeat across corpora; the scenario generator derives scenario ids from the id and the pooled position, and the legacy 85-window and 17-window corpora already shared 17 ids, so pooling is unaffected.

| Corpus | HIF windows | Healthy controls | Legacy NLM top-1 / top-3 | Reference-scan WLS alarms | Paired healthy alarms | Discovered-mode admissions (margin 1.25) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Main training extra | 252 | 60 | 251 / 252 | 117 | 6 | 69 |
| Main validation extra | 63 | 15 | 63 / 63 | 28 | 1 | 17 |

The 60 and 15 healthy controls produced 1 and 0 alarms. The audit for these corpora is under `output/hif_physical_revision_20260921/detection_audit/` (WLS observations, cohort progress, and an admission manifest in the same format as the 2026-09-20 one).

Combined admitted pool at margin 1.25, all four main corpora:

| Population | 100-200 ohm | 200-500 ohm | 500-1000 ohm | Total |
| --- | ---: | ---: | ---: | ---: |
| Training (84 + 252 windows) | 84 / 95 | 10 / 125 | 0 / 116 | 94 / 336 |
| Validation (21 + 63 windows) | 23 / 25 | 1 / 30 | 0 / 29 | 24 / 84 |
| All | 107 / 120 | 11 / 155 | 0 / 145 | 118 / 420 |

The admitted pool of 118 windows now exceeds the full plan demand of 78 pure-HIF roots (D0 38, two rounds of 12, development 16) and the 40 measurement+HIF roots. The pipeline names all four corpora: `pipeline.env` defines `HIF_CORPUS_TRAIN_EXTRA` and `HIF_CORPUS_VALID_EXTRA`, the collection arguments and `stage_d0.sbatch` pass them alongside the original two, `prerequisites.sh` checks them and records their digests, and `PHYSICAL_HIF_SAMPLE_PATHS` lists all four. The configuration tests pass (27).

The composition of the admitted pool is the honest consequence of discovered-mode admission at power sigma 0.01: 107 of the 118 admitted windows are 100-200 ohm (2.1-4.2 pu), 11 are 200-500 ohm, and none is 500-1000 ohm. A policy trained from this pool learns the strongest third of the declared population. The representative and weak classes remain available in the raw corpora for evaluation and for a flagged or telemetry-triggered acquisition mode. No training job was launched.


Validation: the 508-test integration selection passed. The full DAgger run passed 907 tests with 41 skipped and found four test-interface/stale-budget failures; all four were corrected and passed in a five-test targeted rerun. The fixed code therefore covers 911 passing DAgger tests, with the original and rerun receipts preserved separately. Physical-profile, exporter, generator and conditioned-recovery selections passed after their documented fixture fixes. A fresh two-scan physical fit recovered 500 ohm as 499.999999999986 ohm on the 69 kV base. See the [final validation receipt](../output/hif_physical_revision_20260920/validation_summary.json) for the separate evidence and limitations.

Reproduce the result audit with `scripts/audit_hif_physical_corpora.py --corpus-dirs <the four directories above> --out <fresh output directory>`. It reads preserved corpus observations, runs the current operator WLS and actual scenario admission, and writes detailed observations and paired controls. It does not refit or relabel faults to force admission.


### 2026-09-21 unbalance corpus under the WLS shunt convention

The three-phase unbalance corpus had the same exporter defect as the legacy HIF corpora: `out_measurements_imbalance_currents_20260903` counted the bus-9 capacitor in Qinj, so all 220 of its rows alarmed the operator WLS (median chi-square 246 against a threshold of 130, largest residual at the bus-9 reactive injection in 202 rows) regardless of the unbalance. Removing that injection analytically left 75 of 220 alarms at the plain detector and 62 at margin 1.25, graded by the voltage unbalance factor at the labeled bus.

The unbalance generator now takes `--shunt-convention` (default `ybus`) and `--telemetry-bases` (default `physical_local_bases`), writes `measurement_convention` on every row and in `meta.json`, and records the operator voltage channel explicitly: **Vm is the phase-A line-to-neutral magnitude** (decision 2026-09-21). Under unbalance that channel departs from the positive-sequence magnitude, which is single-phase-meter physics rather than an export artifact, so the resulting WLS alarms are legitimate balanced-screen detections. Balanced controls from pypower carry the ybus convention by construction.

```
python Transmission/generate_measurements_imbalance.py --out artifacts/measurements/out_measurements_imbalance_currents_ybus_440_20260921 --n-imbalance 440 --n-no-error 60 --seed 20260925
```

Generation took 106 s. In the new corpus the bus-9 reactive injection of every unbalanced solve agrees with the balanced pypower reference within 0.002 pu (the legacy offset was about 0.19 pu). The audit script `scripts/audit_unbalance_physical_corpus.py` refits every row with the operator WLS and runs the scenario generator's discovered-mode admission; its outputs are under `output/hif_physical_revision_20260921/unbalance_audit/`.

| Quantity | Value |
| --- | ---: |
| Unbalance windows | 440 |
| Balanced controls | 60 |
| Reference-scan WLS alarms (margin 1.0) | 218 / 440 |
| Alarms at margin 1.25 | 160 / 440 |
| Discovered-mode admissions (margin 1.25) | 160 / 440 |
| Rejected as unobservable (VUF below the telemetry gate) | 3 |
| Healthy-control alarms, margin 1.0 / 1.25 | 2 / 60 and 0 / 60 |

| VUF at the labeled bus | Windows | Alarms (1.0) | Admitted (1.25) |
| --- | ---: | ---: | ---: |
| below 0.5 % | 108 | 9 | 2 |
| 0.5 to 1 % | 97 | 34 | 14 |
| 1 to 2 % | 116 | 71 | 47 |
| 2 to 3 % | 57 | 46 | 41 |
| above 3 % | 62 | 58 | 56 |

Detection is graded by severity and by the size of the unbalanced load: buses 5 and 11 (7.6 MW and 3.5 MW) produced no admitted window out of 68, while buses 3 and 9 produced 36 of 42 and 32 of 43. The admitted pool of 160 exceeds the plan demand of 90 unbalance roots (D0 50, two rounds of 12, development 16) and the 40 balanced-control roots. `pipeline.env` now names this corpus as `IMBALANCE_CORPUS`, the collection stage passes it explicitly, and `PHYSICAL_IMBALANCE_SAMPLE_PATH` records it in the scenario generator. The legacy corpus is untouched and remains the module default for resuming older cells.

A caveat that applied to both corpora until the fix described in the next section: `z_true` was a pypower optimal-power-flow solution of the balanced case at the same total load, while the OpenDSS model runs a fixed generator dispatch. The two agree on the network but not on dispatch, so the slack and generator injections of `z_true` can differ from the OpenDSS sensor mean by tenths of a pu even for a nearly balanced split. `z_true` is a balanced network reference, not a paired healthy counterfactual for the generation channels; the paired controls in the audits use the same OpenDSS operating point instead.

### 2026-09-21 balanced reference fix (`z_true`)

The row-level balanced reference `z_true` was a pypower optimal-power-flow solution of the balanced case at the same total load. OpenDSS runs the fixed generator schedule of the DSS model, so the two differed in dispatch: in the 252-window training corpus about 52 of 122 channels differed from the sensor mean by more than 0.05 pu (slack injections by 0.5 to 0.6 pu, Line 1-2 flows by 0.35 pu, some voltage magnitudes by up to 0.14 pu). That vector fed `clean_measurements`, which the release audit compares against each mixed root's declared meter truth, and it was the SCADA vector of the no-disturbance control while that control's telemetry came from an OpenDSS solve. Both generators now compute `z_true` as the balanced OpenDSS solve at the same operating point (same dispatch, load scale or profile, voltage setpoints and shunt convention) with the disturbance removed: the HIF generator replays scan 0's operating point without the fault, the unbalance generator re-solves with every load balanced. The pypower vector is kept as `z_reference_opf`; `balanced_reference` and `z_true_semantics` name the choice on every row and in `meta.json`. Legacy pu mode keeps the OPF reference so historical corpora can still be reproduced (`--balanced-reference`).

All seven corpora were regenerated with unchanged seeds, so labels, observations and admission are identical to the tables above (main-corpora HIF admissions 25 + 7 + 69 + 17 = 118; unbalance admissions 160 / 440; the same five top-1 ranking misses on Line 1-2). Every strict-physics replay passed (7,770 clean scans). The reference now differs from the sensor mean only by the disturbance itself:

| Corpus | Windows | max abs(z_true - z_clean), pu | Channels above 0.05 pu per window | Former OPF reference: max, channels above 0.05 |
| --- | ---: | ---: | ---: | ---: |
| Main training | 84 | 0.156 | 1.3 | 1.00, 52.1 |
| Main validation | 21 | 0.159 | 1.8 | 0.99, 49.5 |
| Detection limit | 21 | 0.014 | 0.0 | 0.97, 49.4 |
| Main training extra | 252 | 0.167 | 1.2 | 1.04, 52.3 |
| Main validation extra | 63 | 0.152 | 1.3 | 0.98, 48.5 |
| Full sweep | 336 | 0.351 | 1.1 | 1.12, 51.6 |
| Unbalance | 440 | 0.115 | 0.2 | 1.08, 28.8 |

The remaining channel is the faulted injection: a 101 ohm fault at 69 kV dissipates 15.8 MW, exactly the 0.158 pu gap at that bus, and the 0.35 pu sweep maximum is a 50 ohm fault. For unbalance the residual gap is the phase-A voltage magnitude at the unbalanced bus. `z_true` is therefore a same-physics paired healthy reference for every channel, the no-disturbance control's SCADA and telemetry now come from one operating point, and the mixed-root clean-vector check compares like with like. Regression tests: `tests/test_generate_measurements_hif_physical.py` and `tests/test_generate_measurements_imbalance_physical.py`.

### 2026-09-21 HPC cell on the detectable corpora

The IEEE-14 full pipeline was launched on the cluster with the updated corpora restricted to the windows the operator WLS can discover. Subsets of the regenerated corpora were written with only the admitted windows (discovered mode, anomaly margin 1.25, reference scan) and their balanced controls, each `meta.json` recording the source corpus, its digest and the audit that produced the filter:

| Corpus (committed) | Windows | Controls |
| --- | ---: | ---: |
| `hif_physical69_main_train_detectable_25x10_20260921` | 25 | 20 |
| `hif_physical69_main_valid_detectable_7x10_20260921` | 7 | 5 |
| `hif_physical69_main_train_extra_detectable_69x10_20260921` | 69 | 60 |
| `hif_physical69_main_valid_extra_detectable_17x10_20260921` | 17 | 15 |
| `out_measurements_imbalance_currents_ybus_detectable_160_20260921` | 160 | 60 |

A re-audit of the subsets admits every window (118 of 118 HIF, 160 of 160 unbalance). `pipeline.env` names these five corpora; the plans are unchanged (78 pure-HIF roots, 40 measurement+HIF, 90 unbalance, 40 balanced controls) and fit the admitted pools. The cell is `/scratch/yx3882/research_full_pipeline_20260921_physical`, deployed from commit `07880e1` on `codex/wls-screen-gnn` and submitted on 2026-09-21 at 18:13 UTC as one dependency chain: d0 18205117, bc0 18205119, r1c 18205121, r1t 18205123, r1e 18205124, r2c 18205125, r2t 18205129, r2e 18205131. Results will appear in `out/pipeline_summary.json`. Two things to keep in mind when reading them: the HIF roots are the WLS-visible subset, 91 percent of them in the 100-200 ohm band, not the declared 100-1000 ohm population; and the sweep and detection-limit corpora remain evaluation-only and untracked.

### 2026-09-23 generator reactive-limit fix and regenerated HIF corpora

Every HIF corpus above was simulated without voltage regulation. `apply_hif_operating_point` writes each generator's kW through the OpenDSS API, and a kW write makes OpenDSS recompute `maxkvar`/`minkvar` from the nominal power factor (0.88 by default, so the limits become ±1.0795 × kW). The model file declares the MATPOWER limits (bus 2: 50 / -40 MVAr; the 1 kW synchronous condensers at buses 3, 6 and 8: 40 / 0, 24 / -6 and 24 / -6 MVAr). After the write the condensers had about ±1.08 kvar, so every PV generator sat at its tiny limit instead of holding its setpoint. PV-bus voltages were a median 3 to 12 percent below setpoint, and bus 8, which carries only its condenser, injected no reactive power in any HIF window (|Qinj| ≤ 1.1e-5 pu, against 0.14 to 0.24 pu in the unbalance corpus). A GNN screen trained on the 2026-09-22 DAgger-aligned corpus learned this as a label cue; the DAgger student could use it the same way. The unbalance generator never writes kW and was unaffected.

The fix keeps the model's limits: `capture_operating_point_baseline` records each generator's `maxkvar`/`minkvar` and `apply_hif_operating_point` restores them after the kW and Vpu writes (from the baseline, or the live values for a baseline captured before this change). All callers (corpus generation, the paired `z_true` reference, candidate simulation in the single- and multi-scan estimators and conditioned recovery) go through this function, so the estimator's forward model and the corpora change together. Across all 7,770 stored operating points every solve converges and keeps the declared limits; median PV-bus voltages sit on their setpoints, and at high load the limits bind as expected (bus 2 at 50 MVAr in 48 percent of scans, bus 8 at 24 MVAr in 13 percent). Regression tests: `tests/test_hif_operating_point_regulation.py`.

The six corpora were regenerated with unchanged recipes and seeds by `scripts/regenerate_hif_physical_corpora.py`, run from a clean worktree of commit `9967e14` plus the fix. Labels and operating points are identical to the earlier corpora; only the physics differs (bus-8 Qinj now 0.055 to 0.240 pu; `z_true` moves by a median 0.7 to 0.8 pu per window). Receipts, validator logs and audits are under `output/hif_physical_revision_20260923/`. Strict-physics replay and branch-current localization pass for all six corpora; the sample validator reports the same legacy-NLM top-1 ranking misses as before on the same rows (4 in the sweep, 1 in the training extra, all in the top 3).

| Corpus (2026-09-23) | HIF windows | Reference-scan WLS alarms (before → after) | Healthy-control alarms | Admitted, margin 1.25 (before → after) | Admitted by band 100-200 / 200-500 / 500-1000 ohm |
| --- | ---: | ---: | ---: | ---: | --- |
| `hif_physical69_main_train_84x10_20260923` | 84 | 37 → 39 | 1 / 20 | 25 → 27 | 21 / 6 / 0 |
| `hif_physical69_main_valid_21x10_20260923` | 21 | 10 → 11 | 0 / 5 | 7 → 8 | 8 / 0 / 0 |
| `hif_physical69_main_train_extra_252x10_20260923` | 252 | 117 → 126 | 1 / 60 | 69 → 77 | 67 / 10 / 0 |
| `hif_physical69_main_valid_extra_63x10_20260923` | 63 | 28 → 31 | 0 / 15 | 17 → 19 | 17 / 2 / 0 |
| `hif_physical69_detection_limit_21x10_20260923` | 21 | 0 → 0 | none | evaluation only | |
| `hif_physical_sweep_eval_336x10_20260923` | 336 | 67 → 70 | none | 51 → 57 (evaluation only) | |

The detectable subsets `hif_physical69_main_train_detectable_27x10_20260923`, `hif_physical69_main_valid_detectable_8x10_20260923`, `hif_physical69_main_train_extra_detectable_77x10_20260923` and `hif_physical69_main_valid_extra_detectable_19x10_20260923` keep every control and the 131 admitted windows: 113 at 100-200 ohm and 18 at 200-500 ohm, none above 500 ohm (previously 118: 107 and 11). No window was lost; the 13 additions were borderline before (largest normalized residual 4.1 to 5.0) and cross the admission margin because the fault dissipates more power once the PV buses hold their voltage. A re-audit admits all 131. The pool still covers the plans (78 pure-HIF roots, 40 measurement+HIF). The admitted population remains the strongest part of the declared 100-1000 ohm range. Each new `meta.json` records `hif.generator_reactive_limits.policy = model_file_limits_kept`; pre-fix corpora lack the key.

Two checks say nothing about the new physics and should not be read as evidence for it: the healthy controls are pypower OPF solves and are byte-identical to the pre-fix ones, and the legacy NLM diagnostic (`nlm_diagnostic`, the sample validator's top-1 ranking) uses a load-scale-only bridge that never sees the operating point, so its four sweep misses and one training-extra miss are unchanged by construction. Strict-physics replay is the check that would have caught a mismatch: every pre-fix corpus fails it under the fixed simulator (0 of 210 scans of the 2026-09-19 validation corpus match), and the tracked 20260903 corpora fail it too.

What the fix does not change: the OpenDSS solve tolerance stays at its default 1e-4. With four regulating PV units a single solve stops about one power-channel sigma short of the converged power flow, and the HIF operating-point path and the unbalance generator's path (which only scales loads) stop on slightly different sides, so at the same load scale their healthy channels at buses 2 and 3 differ by about 1.5 sigma (162 sigma before the fix). The corpora are self-consistent with the simulator that generated them; removing the residual offset needs a tight tolerance (1e-8) in both compile paths and a regeneration of the unbalance corpus as well. `voltage_setpoints_pu` are targets, not realized voltages: in about half of the scans at least one PV unit is at a reactive limit and behaves as a PQ bus.

Consumers switched with this revision: `research/hpc/full_pipeline_20260907/pipeline.env` (`HIF_CORPUS_*`, 131-window capacity), `research/test_hpc_full_pipeline.py`, the `PHYSICAL_HIF_*` constants in `psse_env/providers/scenario_generator.py`, `research/gnn_screen/dagger_corpus.py` (defaults; refuses any HIF corpus whose stored `z_true` the simulator cannot reproduce) and its README. The pre-fix corpora are kept for reproducing earlier runs but no longer match the simulator: the HIF estimator's forward model, conditioned recovery and the telemetry replay all go through the fixed path, so resuming an earlier cell, or re-scoring frozen roots built from pre-fix corpora with the fixed source, mixes physics. The DAgger `telemetry_no_disturbance` controls also change, because their auxiliary telemetry comes from `_simulate_base` while their SCADA comes from the unbalance path. The 2026-09-22 GNN DAgger-aligned corpus and its results were built from pre-fix corpora and need a rebuild.

### 2026-09-23 (b) OpenDSS solve tolerance and the regenerated HIF and unbalance corpora

The `_20260923` corpora above were still solved at OpenDSS's default convergence tolerance of 1e-4. With four regulating PV units that is not a converged power flow: against a 1e-10 reference, one solve of a stored operating point was up to 1.7 measurement sigmas off in the power channels (median 0.24 sigma), and the HIF operating-point path and the unbalance generator's load-scaling path stopped on different sides of the fixed point, 1.55 sigma apart on the bus-2/3 injections at the same load scale. `Run_IEEE14Bus.dss` now sets `tolerance=1e-8` (and `maxiterations=200`); every consumer compiles the model through that file, including the unbalance generator, the estimators, the GNN corpus converter and the legacy NLM bridge. At 1e-8 the single-solve error is below 1e-4 sigma and the two paths agree within 3e-4 sigma; solves take a median 48 and at most 79 iterations at no measurable cost in time. Both generators now refuse to export a non-converged solve, and the strict replay of the unbalance corpus (every balanced reference and unbalanced sensor mean re-solved) is part of `scripts/regenerate_hif_physical_corpora.py --families hif,unbalance`. Regression test: `tests/test_ieee14_opendss_solve_tolerance.py`.

The six HIF corpora and the unbalance corpus were regenerated with unchanged seeds as `*_20260923b` (unbalance: `out_measurements_imbalance_currents_ybus_440_20260923b`); labels and operating points are identical to the earlier corpora, the healthy references move by a median 0.8 sigma per window, and every replay is exact. WLS admission is unchanged to the window: 27 + 8 + 77 + 19 = 131 HIF windows (reference-scan alarms 39, 11, 126, 31; healthy-control alarms 1, 0, 1, 0) and 160 of 440 unbalance windows (218 alarms at margin 1.0, 0 of 60 controls at margin 1.25). The detectable subsets are `hif_physical69_main_train_detectable_27x10_20260923b`, `hif_physical69_main_valid_detectable_8x10_20260923b`, `hif_physical69_main_train_extra_detectable_77x10_20260923b`, `hif_physical69_main_valid_extra_detectable_19x10_20260923b` and `out_measurements_imbalance_currents_ybus_detectable_160_20260923b`; re-audits admit every window. The legacy NLM ranking misses are unchanged (4 in the sweep, 1 in the training extra). Receipts, replay reports, audits and the implementation manifest are under `output/hif_physical_revision_20260923b/`; `pipeline.env`, its test, the scenario-generator constants and the GNN converter defaults name these corpora.

One engine behaviour surfaced during the parallel regeneration and is handled rather than understood: about once per 8,000 solves, a fault solve diverged to NaN (base case converged normally, ordinary circuit, an operating point and fault that converge on every other run, in serial runs and in twelve parallel runs on shared and private model folders). A converged solve is deterministic, so `_simulate_base`, `_simulate_candidate`, the unbalance generator and the GNN converter now redo the whole compile, edit and solve from scratch when a solve does not converge, up to three times, before failing; the replay validators check every exported row afterwards.

## Historical scope and limitations

Tracked 20260903/20260714 corpora and the frozen BC0 suite retain their original system-pu labels and conventions. `CURRENT_TELEMETRY_HIF_SAMPLE_PATHS` retains its old paths; the new `PHYSICAL_HIF_*` constants and explicit HPC arguments select the physical corpora. No cluster job is launched by this revision.

The historical IEEE57 study was normalized and had no physical kV map. The separate [IEEE57 138/69 kV reconstruction](ieee57_physical_hif_20260919.md) is now available and must not be confused with those historical results.

These are synthetic fundamental-frequency resistive surrogates. A quiet WLS trace is retained for detection-limit evaluation, not automatically admitted as an expert-action training label. A WLS alarm does not establish correct localization, identifiable fault distance, physical repair or learned-policy success.
