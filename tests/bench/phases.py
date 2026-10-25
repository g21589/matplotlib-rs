"""Time the phases of a typical script: import, plotting calls, first and
repeated savefig.  Usage: python phases.py mpl|rs"""
import os
import sys
import time

t0 = time.perf_counter()
if sys.argv[1] == "rs":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))
    import matplotlib_rs
    matplotlib_rs.install()
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
t1 = time.perf_counter()

x = np.linspace(0, 10, 1000)
fig, axs = plt.subplots(2, 2, figsize=(8, 6))
for i, ax in enumerate(axs.flat):
    ax.plot(x, np.sin(x + i), label="sin")
    ax.plot(x, np.cos(x + i), "--", label="cos")
    ax.scatter(x[::50], np.sin(x[::50]), c=x[::50])
    ax.set_title(f"panel {i}")
    ax.set_xlabel("x")
    ax.legend()
    ax.grid(True)
fig.tight_layout()
t2 = time.perf_counter()
fig.savefig("/tmp/claude-0/phase.png")
t3 = time.perf_counter()
N = 10
for k in range(N):
    axs[0, 0].set_title(f"frame {k}")
    fig.savefig("/tmp/claude-0/phase.png")
t4 = time.perf_counter()
print(f"{sys.argv[1]}: import {1e3*(t1-t0):7.1f} ms  build {1e3*(t2-t1):7.1f} ms  "
      f"first savefig {1e3*(t3-t2):7.1f} ms  later savefig {1e3*(t4-t3)/N:7.1f} ms")
