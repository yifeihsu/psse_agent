# Research experiments

Experiments built on the `psse_env` environment, expert oracle and DAgger
tooling. Research only; nothing here is release evidence.

| Path | Contents |
|---|---|
| `hpc/full_pipeline_20260907/` | Slurm cell for the full pipeline: expert aggregate (D0), BC0, DAgger rounds 1 and 2, paired evaluation. Its README records every run and the stage layout. |
| `gnn_screen/` | `WLSScreenGNN`, the auxiliary screen that runs after a balanced WLS alarm: corpora, training, calibration, evaluation and results. |
| `reviewed_fault_scenarios.py`, `reviewed_observable_context.py`, `reviewed_expert_admission.py`, `filter_reviewed_training.py` | The September 2026 reviewed fault-scenario cohorts and their WLS-observable training admission (`docs/fault_scenario_review_20260917.md`, `docs/wls_observable_training_20260917.md`). |
| `ieee57/` | Recorded evidence of the IEEE 57 balanced, logical-topology and transfer work. |

The August 2026 research runner (`collect.py`, `train.py`, `evaluate.py`,
`run_dagger.py`, ...), its occupancy and exposure-curve cells, the
2026-09-03 diagnostic round and the 2026-09-22 revised-evaluation cell were
removed on 2026-09-28; they remain in the git history before that date.
