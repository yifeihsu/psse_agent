#!/usr/bin/env python3
"""CLI wrapper for the deterministic BC0 release-suite builder."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from psse_env.dagger.suite_builder import main


if __name__ == "__main__":
    raise SystemExit(main())
