#!/usr/bin/env python3
"""Collect rank-one recovery labels on learner-visited DAgger-1 states."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from psse_env.dagger.collect_dagger1 import main


if __name__ == "__main__":
    raise SystemExit(main())
