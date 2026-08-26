"""A minimal research prototype of the DAgger pipeline.

The modules under ``psse_env`` implement a release-grade pipeline: content
addressed provenance, frozen commit bindings, study-manifest pinning, approved
accelerator classes and reproducibility receipts.  That machinery is valuable
for a production artifact and is pure overhead for an academic demonstration.

This package keeps the science and drops the scaffolding.  It reuses the
environment, the expert oracle, the rollout collector, the SFT rendering and
masking, and the canonical policy classes unchanged, so results remain
comparable with the release path.  It does not reuse the gate wrappers.

Nothing produced here is release evidence.
"""

from __future__ import annotations

__all__ = ["collect", "evaluate", "model", "train"]
