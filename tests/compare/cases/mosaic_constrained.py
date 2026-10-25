import numpy as np
import matplotlib.pyplot as plt

fig, axd = plt.subplot_mosaic([["a", "b"], ["a", "c"]], layout="constrained", figsize=(7, 4))
axd["a"].plot(np.random.default_rng(0).random(20))
axd["b"].hist(np.random.default_rng(1).normal(size=200), bins=15)
axd["c"].imshow(np.arange(16).reshape(4, 4))
for k, ax in axd.items():
    ax.set_title(k)
fig.legend(["data"], loc="outside upper right")
fig.savefig("mosaic_constrained.png")
