import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(7)
x, y = rng.random(80), rng.random(80)
c = rng.random(80)
s = 500 * rng.random(80)
fig, ax = plt.subplots()
sc = ax.scatter(x, y, c=c, s=s, alpha=0.6, cmap="viridis", edgecolors="k", linewidths=0.5)
fig.colorbar(sc, label="value")
ax.scatter([0.5], [0.5], marker="*", s=300, c="r")
ax.scatter(x[:10], y[:10] * 0.5, marker="^", facecolors="none", edgecolors="b")
fig.savefig("scatter_cmap.png")
