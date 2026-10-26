#!/usr/bin/env bash
# Same test entry point locally and on GitHub Actions; preserve failing exit codes.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
export MPLBACKEND=Agg
mkdir -p tests/output/ci
python scripts/check_version.py
for test in tests/unit/test_*.py tests/lowlevel/diff_*.py tests/model/first_plot.py; do
    python -u "$test" 2>&1 | tee "tests/output/ci/$(basename "$test" .py).log"
done
COMPARE_VECTOR=svg,pdf,eps COMPARE_OUT="$PWD/tests/output/ci/compare" \
    python -u tests/compare/compare.py --rasterize-eps polar_more/polar_more.eps \
    2>&1 | tee tests/output/ci/compare.log
