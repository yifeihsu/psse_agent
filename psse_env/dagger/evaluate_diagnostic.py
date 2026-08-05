"""Run a closed-loop suite as diagnostic-only, never as release evidence.

This wrapper deliberately reuses the exact release environment, policy
identity, and scenario-schema checks.  Its only semantic difference is
irreversible: the emitted artifact has the diagnostic artifact type and is
explicitly ineligible for release evidence and training.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from pathlib import Path

from psse_env.dagger.evaluator import main as _evaluation_main


def _input_value(arguments: Sequence[str]) -> str | None:
    for index, argument in enumerate(arguments):
        if argument == "--input" and index + 1 < len(arguments):
            return arguments[index + 1]
        if argument.startswith("--input="):
            return argument.partition("=")[2]
    return None


def _derive_required_suites(arguments: Sequence[str]) -> list[str]:
    """Bind coverage to every suite present in the temporary replay file."""

    if any(
        argument == "--required-suite"
        or argument.startswith("--required-suite=")
        for argument in arguments
    ):
        return []
    input_value = _input_value(arguments)
    if not input_value:
        return []
    payload = json.loads(Path(input_value).expanduser().read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload:
        return []
    required: list[str] = []
    for suite in sorted(str(key) for key in payload):
        required.extend(("--required-suite", suite))
    return required


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    required_suites = _derive_required_suites(arguments)
    return _evaluation_main(
        [*arguments, *required_suites, "--diagnostic-only"]
    )


if __name__ == "__main__":
    raise SystemExit(main())
