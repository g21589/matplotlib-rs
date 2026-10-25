import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 5, 50)
fig, axs = plt.subplots(2, 2, figsize=(7, 5))
for i, ax in enumerate(axs.flat):
    ax.plot(x, x ** (i + 1) / 10)
    ax.set_title(f"power {i + 1}")
    ax.set_xlabel("x")
fig.suptitle("Grid")
fig.savefig("subplots_grid.png")
