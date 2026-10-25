import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(2)
fig, ax = plt.subplots()
ax.boxplot([rng.normal(size=100), rng.normal(1, 2, size=100)])
fig.savefig("boxplot_basic.png")
