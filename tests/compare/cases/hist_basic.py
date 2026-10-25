import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
fig, ax = plt.subplots()
ax.hist(rng.normal(size=1000), bins=30)
fig.savefig("hist_basic.png")
