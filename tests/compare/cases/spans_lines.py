import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot(np.arange(10), np.arange(10) ** 0.5)
ax.axhline(1.5, color="r", ls="--")
ax.axvline(4, color="g", lw=3)
ax.axhspan(0.5, 1.0, alpha=0.3)
ax.axvspan(6, 7, color="orange", alpha=0.4)
ax.axline((0, 0), slope=0.3, color="k")
ax.fill([1, 2, 3], [2, 2.8, 2.1], "m", alpha=0.5)
ax.hlines([2.5, 2.7], 1, 8, colors="c")
ax.vlines([8, 8.5], 0, 2, linestyles="dotted")
fig.savefig("spans_lines.png")
