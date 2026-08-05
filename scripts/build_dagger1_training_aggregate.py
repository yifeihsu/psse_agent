#!/usr/bin/env python3
"""Build the provenance-valid D0 union D1 aggregate used by STAGE=round1."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from psse_env.dagger.build_dagger1_aggregate import main


if __name__ == "__main__":
    raise SystemExit(main())
