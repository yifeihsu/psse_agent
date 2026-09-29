# Artifacts

`measurements/` holds the corpora that the scenario generator and the pipeline
read: the physical-ohm HIF and unbalance subsets (`*_20260923opf` used by the
pipeline, `*_20260923b` kept for comparison), the 2026-09-03 branch-current
corpora, the 2026-07-14 multi-scan HIF benchmark, and the balanced tabular
measurements. Full corpora and new regenerations are written here too but are
ignored by git; see `scripts/regenerate_hif_physical_corpora.py`.

`research_dagger_trace_20260823/trace_validation.jsonl` is the 27-row
validation set that every generated scenario suite keeps out
(`TRACE_VALIDATION` in `research/hpc/full_pipeline_20260907/pipeline.env`).
