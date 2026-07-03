# Three-Phase NLM / HIF Legacy Code

This folder contains the imported three-phase NLM and HIF study code as a
standalone legacy reference. The active IEEE-14 integration lives in the
repository-level `three_phase_nlm/` adapter package and the
`Transmission/generate_measurements_hif_ieee14.py` generator.

## Layout

```text
Three_phase_NLM-cable-burnout/
├── src/three_phase_nlm_legacy/  # Core legacy NLM, parser, PF, and utilities
├── scripts/                     # Legacy study and batch-run scripts
├── models/
│   ├── ieee4/                   # IEEE 4-node OpenDSS assets
│   └── ieee342/                 # 342/LV feeder OpenDSS assets and source CSVs
├── docs/                        # Original notes and detection summaries
└── artifacts/                   # Generated plots, notebooks, and batch temp output
```

## Running Legacy Scripts

The scripts were imported from a flat working directory, so run them with the
legacy source directory on `PYTHONPATH`:

```bash
PYTHONPATH=Three_phase_NLM-cable-burnout/src/three_phase_nlm_legacy \
python Three_phase_NLM-cable-burnout/scripts/case_4.py
```

Most legacy runs also assume the matching OpenDSS files are the process working
directory. For reproducible new IEEE-14 HIF trace generation, use the adapter
package at the repository root instead of these legacy scripts.
