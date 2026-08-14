"""Executable entry point for closed-loop release evaluation.

Run ``python -m psse_env.dagger.evaluate_release --help`` for the JSON suite,
factory, coverage, and output arguments.
"""

from __future__ import annotations

from psse_env.dagger.evaluator import main


if __name__ == "__main__":
    raise SystemExit(main())
