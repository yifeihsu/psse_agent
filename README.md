# PSSE Agent

Research code for agentic AI in power system state estimation. A Gemma 4 agent
diagnoses and corrects bad measurements, line-parameter errors, breaker
topology errors, high-impedance faults (HIF), load unbalance and harmonics by
calling state-estimation tools inside a transactional environment. It is
trained by behaviour cloning on a rule-based expert (BC0) and then by DAgger.
This is an academic research repository, not release software.

## Layout

| Path | Contents |
|---|---|
| `psse_env/` | The environment (`transactional_env.py`, `actions.py`, `evidence_profile.py`), tool providers (`providers/`: WLS and correction tools, the HIF screen, the scenario generator), the rule-based expert (`oracle/`), DAgger collection, datasets and closed-loop evaluation (`dagger/`), and Gemma 4 SFT (`sft/`). Design notes are in `psse_env/README.md`. |
| `scripts/run_dagger_research.py` | DAgger collection and paired closed-loop evaluation. |
| `research/hpc/full_pipeline_20260907/` | Slurm cell for the full pipeline: expert aggregate (D0), BC0, DAgger rounds and evaluation. Its README records every run. |
| `research/` | The WLS screening GNN (`gnn_screen/`), the reviewed fault-scenario cohorts and the IEEE 57 evidence (`research/README.md`). |
| `three_phase_model/`, `three_phase_nlm/`, `IEEE_14_OpenDSS/` | Three-phase OpenDSS models (IEEE 14, 57, 118), HIF and unbalance simulation, multi-scan HIF estimation and branch-current analysis. |
| `Transmission/` | Measurement-corpus generators (tabular, HIF, unbalance, the IEEE-14 node/breaker model) and the original MATLAB state-estimation code. |
| `logical_topology/` | IEEE 57 logical (circuit-breaker) topology simulator and estimator. |
| `tools/` | Python ports of the Lagrangian-multiplier parameter and topology estimators. |
| `mcp_server/` | MATPOWER-backed tool implementations (also servable over MCP) and the IEEE cases. |
| `Harmonics/`, `models/`, `schema/` | Harmonic state estimation, the IEEE-14 node/breaker model data, the decision JSON schema. |
| `data/`, `artifacts/` | Tracked corpora and inputs that the scenario generator reads (`artifacts/README.md`). |
| `docs/` | Dated research notes and run reports. |
| `tests/`, `test_*.py` | Tests. |

## Running

Tests run from the repository root, for example
`python -m pytest -q psse_env/dagger/test_research_dagger_minimal.py`.

Training and evaluation run on the cluster through the pipeline cell; its
README covers staging (`deploy_remote.sh`), submission (`submit_pipeline.sh`)
and every stage. The GPU environment is pinned in
`psse_env/requirements-sft-research.txt`; the IEEE 57 numerical testbeds need
`psse_env/requirements-ieee57.txt`. The HIF and unbalance corpora are
regenerated with `scripts/regenerate_hif_physical_corpora.py`, and balanced
IEEE 57/118 corpora with `scripts/build_balanced_corpus.py`.

Physical HIF configuration: [IEEE 14 physical experiment](docs/ieee14_physical_hif_20260918.md),
[legacy-stack physical-ohm reconfiguration](docs/ieee14_hif_legacy_reconfiguration_20260919.md),
and [IEEE 57 reconstruction](docs/ieee57_physical_hif_20260919.md).

## History

The original root-level Gemma SFT pipeline (`preprocess.py`,
`gpt_oss_power_sft_revised_v3.py`, `eval_sft_agent_gemma_v4.py` and their
launchers), the August 2026 research runner and launchers, and the DAgger-1
release tooling were removed on 2026-09-28; they remain in the git history.

## License

See `LICENSE.md`.
