#!/usr/bin/env python3
"""Classify a DAgger-1 collection job: GO, fail-closed NO-GO, or infrastructure failure."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from psse_env.dagger.collection_result import main


if __name__ == "__main__":
    raise SystemExit(main())
