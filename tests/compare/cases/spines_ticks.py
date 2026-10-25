import numpy as np
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 2, figsize=(8, 3.5))
ax = axs[0]
ax.plot(np.linspace(-1, 1, 50), np.linspace(-1, 1, 50) ** 3)
ax.spines["left"].set_position("zero")
ax.spines["bottom"].set_position(("data", 0))
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(direction="inout", length=8, colors="tab:blue")
ax = axs[1]
ax.plot([1, 2, 3], [3, 1, 2])
ax.minorticks_on()
ax.grid(which="major", color="k", lw=0.8)
ax.grid(which="minor", ls=":", lw=0.5)
ax.tick_params(axis="x", which="minor", length=4, color="r")
ax.yaxis.tick_right()
ax.xaxis.set_label_position("top")
ax.set_xlabel("top label")
fig.savefig("spines_ticks.png")
