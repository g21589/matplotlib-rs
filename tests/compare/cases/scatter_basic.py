import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(1)
x, y = rng.normal(size=(2, 200))
c = rng.uniform(size=200)
s = rng.uniform(5, 60, size=200)
fig, ax = plt.subplots()
ax.scatter(x, y, c=c, s=s, alpha=0.7)
fig.savefig("scatter_basic.png")
