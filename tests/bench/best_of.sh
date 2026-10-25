#!/bin/sh
# best-of-N phase timings: tests/bench/best_of.sh mpl|rs [N]
n=${2:-5}
for i in $(seq $n); do MATPLOTLIBRC=tests/mplrc python3 tests/bench/phases.py $1; done | python3 -c "
import sys, re
rows = [list(map(float, re.findall(r'([0-9.]+) ms', l))) for l in sys.stdin]
best = [min(c) for c in zip(*rows)]
print('$1 best of $n: import %.0f  build %.0f  first savefig %.0f  later savefig %.0f ms' % tuple(best))"
