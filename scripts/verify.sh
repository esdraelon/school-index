#!/usr/bin/env bash
# Quality gates for school-index (per school-assistant/docs/CODING-STANDARD.md).
# Runs every gate in order and fails fast on the first failure.
# Mirrors school-info-adapter's verify.sh, minus its Chromium/system
# stage — this project has no browser surface.
#
# Assumes the dev dependencies are installed in the active environment:
#   python3 -m venv .venv && source .venv/bin/activate
#   pip install -e '.[dev]'
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> [1/4] Unit tests"
python -m pytest tests/unit

echo "==> [2/4] Coverage gate (fail_under = 90, see pyproject.toml)"
python -m coverage run -m pytest tests/unit
python -m coverage report

echo "==> [3/4] Type check (mypy strict)"
python -m mypy src/school_index

echo "==> [4/4] LOC cap (500 lines per file under src/)"
python scripts/check_loc.py src

echo "All gates passed."
