#!/usr/bin/env bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
cd "$REPO"
test "$(git rev-parse HEAD)" = "$COMMIT" && test -z "$(git status --porcelain)"
"$PY" -m pip check
"$PY" "$LOG_ROOT/input_audit.py"
test -s "$LOG_ROOT/input_audit.json" && test -z "$(git status --porcelain)"
