#!/usr/bin/env python3
"""CLI wrapper for ``psse_env.dagger.diagnostic_suite``."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from psse_env.dagger.diagnostic_suite import main


if __name__ == "__main__":
    raise SystemExit(main())
