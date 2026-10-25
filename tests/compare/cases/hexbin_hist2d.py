import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(3)
x = rng.normal(size=2000)
y = x * 0.5 + rng.normal(size=2000)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 4))
a1.hexbin(x, y, gridsize=20, cmap="Blues")
a2.hist2d(x, y, bins=25)
fig.savefig("hexbin_hist2d.png")
