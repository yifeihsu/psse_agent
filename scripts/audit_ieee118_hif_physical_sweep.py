"""IEEE118 physical HIF sweep on the case's own 138/161/345 kV bases.

Primary gate is the pipeline detector (chi-square 1% OR normalized residual 4),
with the IEEE57 study's 5% gate re-thresholded on the same fitted observations.
Eligible HIF lines are the 175 active same-voltage zero-tap lines (165 at
138 kV, 10 at 345 kV); 86-87 and 68-116 join different source bases and are
excluded. The disjoint-shard runner and merger are the IEEE57 ones.
"""
from __future__ import annotations

from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts import audit_ieee57_hif_physical_sweep as parallel


def main(argv=None):
    return parallel.main(argv, default_system="case118", default_seed=20260924)


if __name__ == "__main__":
    raise SystemExit(main())
